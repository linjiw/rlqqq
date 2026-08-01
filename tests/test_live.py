from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rlqqq.live import (
    FEATURE_NAMES,
    LEGACY_MODEL_VERSION,
    LIVE_FEATURE_NAMES,
    MODEL_VERSION,
    FrozenActorEnsemble,
    append_forward_log,
    build_feature_frame,
    build_signal_payload,
    load_checked_market_frames,
    replay_frozen_policy,
    validate_forward_log,
    write_signal_json,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"
LEGACY_MODEL = ROOT / "models" / "live" / f"{LEGACY_MODEL_VERSION}.npz"


def checked_replay():
    bundle = FrozenActorEnsemble.load(MODEL)
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    return bundle, frames, features, replay_frozen_policy(
        bundle, features, frames["QQQ"]
    )


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
    np.testing.assert_array_equal(
        actual.loc[dates, FEATURE_NAMES].to_numpy(),
        expected.loc[dates, FEATURE_NAMES].to_numpy(),
    )


def test_v4_numpy_actors_match_saved_sb3_holdout_exposures():
    bundle = FrozenActorEnsemble.load(LEGACY_MODEL)
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    replay = replay_frozen_policy(bundle, features, frames["QQQ"])
    actual = replay.attrs["actor_exposure"]
    for seed in range(10):
        with np.load(
            ROOT
            / "results"
            / "series"
            / f"holdout_v4_resid_QQQ_H2026_s{seed}.npz"
        ) as expected:
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
    assert list(bundle.feature_names) == LIVE_FEATURE_NAMES
    assert "dow" not in bundle.feature_names
    assert "month" not in bundle.feature_names
    assert bundle.model_version == MODEL_VERSION


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
    assert payload["signal"]["stance"] == "Defensive"
    assert abs(payload["signal"]["vt10Exposure"] - 0.42) <= 0.01
    assert abs(payload["signal"]["learnedMean"] - 0.46) <= 0.01
    assert payload["signal"]["learnedMin"] < payload["signal"]["learnedMean"]
    assert payload["signal"]["learnedMax"] > payload["signal"]["learnedMean"]
    assert len(payload["history"]["dates"]) == 90
    assert payload["model"]["trainCutoff"] == "2023-12-31"
    assert payload["model"]["version"] == MODEL_VERSION
    assert payload["model"]["displayName"] == "v8 no-calendar"
    assert payload["model"]["featureCount"] == 22

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
