from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlqqq.live import (
    CROSS_ASSET_FEATURE_NAMES,
    FEATURE_NAMES,
    LEGACY_MODEL_VERSION,
    LIVE_FEATURE_NAMES,
    MODEL_VERSION,
    FrozenActorEnsemble,
    append_forward_log,
    build_feature_frame,
    build_feature_frame_v10,
    build_signal_payload,
    load_checked_market_frames,
    replay_frozen_policy,
    validate_forward_log,
    validate_latest_market_frames,
    write_signal_json,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"
LEGACY_MODEL = ROOT / "models" / "live" / f"{LEGACY_MODEL_VERSION}.npz"


def checked_replay():
    bundle = FrozenActorEnsemble.load(MODEL)
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame_v10(frames)
    return bundle, frames, features, replay_frozen_policy(
        bundle, features, frames["QQQ"]
    )


def test_latest_market_feeds_must_share_the_qqq_session():
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    assert validate_latest_market_frames(frames) == pd.Timestamp("2026-07-31")

    # a feed that runs AHEAD is truncated to the common session
    ahead = {**frames, "^VIX": frames["^VIX"].copy()}
    extra = ahead["^VIX"].iloc[[-1]].copy()
    extra.index = [pd.Timestamp("2026-08-03")]
    ahead["^VIX"] = pd.concat([ahead["^VIX"], extra])
    assert validate_latest_market_frames(ahead) == pd.Timestamp("2026-07-31")
    assert pd.Timestamp(ahead["^VIX"].index[-1]) == pd.Timestamp("2026-07-31")

    # a feed that lags moves the common session BACK for everyone
    lagging = {k: v.copy() for k, v in frames.items()}
    lagging["^VIX"] = lagging["^VIX"].iloc[:-1]
    assert validate_latest_market_frames(lagging) == pd.Timestamp("2026-07-30")


def test_online_features_exactly_match_training_snapshot():
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    actual = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    expected = pd.read_parquet(
        ROOT / "data" / "processed" / "features_QQQ.parquet"
    ).join(
        pd.read_parquet(ROOT / "data" / "processed" / "context.parquet")[
            ["vix", "term_spread_10y_3m", "vix_chg_5d"]
        ]
    )
    dates = actual.index.intersection(expected.dropna().index)
    # Rolling-window implementations can differ by a few floating-point ULPs
    # across the pinned pandas/NumPy runtime and the version that wrote the
    # Parquet snapshot.  Keep the tolerance far below policy-relevant scale.
    np.testing.assert_allclose(
        actual.loc[dates, FEATURE_NAMES].to_numpy(),
        expected.loc[dates, FEATURE_NAMES].to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_v4_numpy_actors_match_saved_sb3_holdout_exposures():
    series_dir = ROOT / "results" / "series"
    expected_paths = [
        series_dir / f"holdout_v4_resid_QQQ_H2026_s{seed}.npz"
        for seed in range(10)
    ]
    if not all(path.is_file() for path in expected_paths):
        pytest.skip("optional gitignored v4 research holdout series are unavailable")

    bundle = FrozenActorEnsemble.load(LEGACY_MODEL)
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    replay = replay_frozen_policy(bundle, features, frames["QQQ"])
    actual = replay.attrs["actor_exposure"]
    for seed, expected_path in enumerate(expected_paths):
        with np.load(expected_path) as expected:
            dates = pd.to_datetime(expected["test_dates"])
            positions = replay.index.get_indexer(dates)
            assert np.all(positions >= 0)
            np.testing.assert_allclose(
                actual[positions, seed],
                expected["test_exposure"],
                rtol=0,
                atol=1e-6,
            )


def test_current_actor_uses_no_calendar_features():
    bundle = FrozenActorEnsemble.load(MODEL)
    assert list(bundle.feature_names) == LIVE_FEATURE_NAMES + CROSS_ASSET_FEATURE_NAMES
    assert "dow" not in bundle.feature_names
    assert "month" not in bundle.feature_names
    assert bundle.model_version == MODEL_VERSION
    assert bundle.vt_target == 0.20
    assert bundle.max_exposure == 1.5
    assert list(bundle.residual_multipliers) == [0.5, 0.75, 1.0, 1.25, 1.5]


def test_signal_contract_reproduces_latest_snapshot(tmp_path):
    bundle, _, _, replay = checked_replay()
    payload = build_signal_payload(
        replay,
        bundle,
        source_name="Yahoo Finance checked snapshot",
        generated_at=datetime(2026, 8, 1, 12, tzinfo=timezone.utc),
    )
    assert payload["asOf"] == "2026-07-31"
    assert payload["stale"] is False
    assert payload["market"]["price"] == 687.99
    assert payload["signal"]["stance"] == "Reduced risk"
    assert abs(payload["signal"]["vt10Exposure"] - 0.84) <= 0.01
    assert abs(payload["signal"]["learnedMean"] - 0.88) <= 0.01
    assert payload["signal"]["learnedMin"] < payload["signal"]["learnedMean"]
    assert payload["signal"]["learnedMax"] > payload["signal"]["learnedMean"]
    assert len(payload["history"]["dates"]) == 90
    assert payload["model"]["trainCutoff"] == "2023-12-31"
    assert payload["model"]["version"] == MODEL_VERSION
    assert payload["model"]["displayName"] == "v10 macro (leveraged)"
    assert payload["model"]["featureCount"] == 28

    signal_path = tmp_path / "signal.json"
    log_path = tmp_path / "forward.csv"
    write_signal_json(payload, signal_path)
    assert append_forward_log(payload, log_path) is True
    assert append_forward_log(payload, log_path) is False
    validate_forward_log(replay, log_path)
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-07-31"
    assert rows[0]["model_version"] == MODEL_VERSION
    assert signal_path.read_text(encoding="utf-8").endswith("\n")
    assert "\r" not in log_path.read_text(encoding="utf-8")
