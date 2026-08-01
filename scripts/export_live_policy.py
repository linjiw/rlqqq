"""Train and export the frozen v8 ensemble as a pure-NumPy actor bundle.

This is a release-time operation, not part of the daily workflow. It does not
append trial records or alter the historical evaluation series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.data import Normalizer, load_market
from rlqqq.live import (
    LEGACY_MODEL_VERSION,
    LIVE_FEATURE_NAMES,
    MODEL_VERSION,
    TRAIN_CUTOFF,
    FrozenActorEnsemble,
    build_feature_frame,
    load_checked_market_frames,
    replay_frozen_policy,
)
from rlqqq.walkforward import FoldSpec

OUTPUT = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"
LEGACY_OUTPUT = ROOT / "models" / "live" / f"{LEGACY_MODEL_VERSION}.npz"
MANIFEST = ROOT / "models" / "live" / "manifest.json"
POLICY_NAME = "v8 no-calendar"
CONFIG_NAME = "ppo_v8_nocal"

FOLD = FoldSpec(
    name="H2026",
    train_start="1994-01-01",
    train_end=TRAIN_CUTOFF,
    val_start="2024-01-31",
    val_end="2025-12-31",
    test_start="2026-01-01",
    test_end="2026-12-31",
)
HYPERPARAMETERS = {
    "n_boot_paths": 3,
    "residual": True,
    "switch_penalty_bps": 5.0,
    "ent_coef": 0.005,
}
STATE_KEYS = {
    "layer1_weight": "mlp_extractor.policy_net.0.weight",
    "layer1_bias": "mlp_extractor.policy_net.0.bias",
    "layer2_weight": "mlp_extractor.policy_net.2.weight",
    "layer2_bias": "mlp_extractor.policy_net.2.bias",
    "action_weight": "action_net.weight",
    "action_bias": "action_net.bias",
}


def train_seed(seed: int, timesteps: int) -> dict:
    """Reproduce one holdout actor without writing to the experiment ledger."""
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rlqqq.env import ExposureTradingEnv, run_policy
    from rlqqq.synth import bootstrap_path
    from rlqqq.walkforward import make_ppo

    torch.set_num_threads(1)
    market = load_market("QQQ", drop_calendar=True)
    train = market.slice(FOLD.train_start, FOLD.train_end)
    test = market.slice(FOLD.test_start, FOLD.test_end)
    normalizer = Normalizer.fit(train.feat)

    train_sets = [train for _ in range(4)]
    for index in range(3):
        train_sets[index + 1] = bootstrap_path(
            train, seed=seed * 100 + index
        )

    def make_env(index: int):
        def factory():
            return ExposureTradingEnv(
                train_sets[index],
                normalizer,
                cost_bps=2.0,
                discrete=True,
                episode_len=252,
                seed=seed * 1000 + index,
                residual=True,
                switch_penalty_bps=5.0,
                max_exposure=1.0,
                vt_target=0.10,
            )

        return factory

    vector_env = DummyVecEnv([make_env(index) for index in range(4)])
    model = make_ppo(vector_env, seed, HYPERPARAMETERS)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    run = run_policy(
        test,
        normalizer,
        model,
        cost_bps=2.0,
        discrete=True,
        residual=True,
        max_exposure=1.0,
        vt_target=0.10,
    )

    state = model.policy.state_dict()
    arrays = {
        output_name: state[state_name].detach().cpu().numpy().astype(np.float32)
        for output_name, state_name in STATE_KEYS.items()
    }
    vector_env.close()
    return {
        "seed": seed,
        "arrays": arrays,
        "sb3_dates": test.index.to_numpy(dtype="datetime64[ns]").astype("int64"),
        "sb3_exposure": run["exposure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        raise SystemExit(f"{OUTPUT} already exists; pass --force to replace it")

    market = load_market("QQQ", drop_calendar=True)
    train = market.slice(FOLD.train_start, FOLD.train_end)
    normalizer = Normalizer.fit(train.feat)
    if market.feat_names != LIVE_FEATURE_NAMES:
        raise AssertionError(
            f"Training features changed: {market.feat_names} != "
            f"{LIVE_FEATURE_NAMES}"
        )

    print(
        f"Training 10 frozen actors through {TRAIN_CUTOFF} "
        f"({args.timesteps:,} steps each)"
    )
    completed: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(train_seed, seed, args.timesteps): seed
            for seed in range(10)
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"  seed {result['seed']}: trained and evaluated in memory",
                flush=True,
            )

    completed.sort(key=lambda result: result["seed"])
    arrays = {
        name: np.stack([result["arrays"][name] for result in completed])
        for name in STATE_KEYS
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        model_version=np.array(MODEL_VERSION),
        policy_name=np.array(POLICY_NAME),
        config_name=np.array(CONFIG_NAME),
        train_cutoff=np.array(TRAIN_CUTOFF),
        seeds=np.arange(10, dtype=np.int16),
        feature_names=np.array(LIVE_FEATURE_NAMES),
        normalizer_mean=normalizer.mean.astype(np.float64),
        normalizer_std=normalizer.std.astype(np.float64),
        **arrays,
    )

    bundle = FrozenActorEnsemble.load(OUTPUT)
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    replay = replay_frozen_policy(bundle, features, frames["QQQ"])
    live_exposures = replay.attrs["actor_exposure"]
    live_dates = replay.index

    numpy_errors = []
    for result in completed:
        seed = result["seed"]
        expected_exposure = result["sb3_exposure"]
        expected_dates = result["sb3_dates"]
        matched = live_dates.get_indexer(
            np.asarray(expected_dates, dtype="datetime64[ns]")
        )
        if np.any(matched < 0):
            raise AssertionError(f"Seed {seed}: NumPy replay is missing dates")
        error = float(
            np.max(
                np.abs(
                    live_exposures[matched, seed]
                    - expected_exposure
                )
            )
        )
        numpy_errors.append(error)
        if error > 1e-6:
            raise AssertionError(
                f"Seed {seed}: NumPy actor differs from SB3 by {error:.3g}"
            )

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    legacy_digest = hashlib.sha256(LEGACY_OUTPUT.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "currentModelVersion": MODEL_VERSION,
        "exportedAt": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "models": {
            MODEL_VERSION: {
                "role": "current_reference",
                "policyName": POLICY_NAME,
                "config": CONFIG_NAME,
                "artifact": OUTPUT.name,
                "artifactSha256": digest,
                "trainCutoff": TRAIN_CUTOFF,
                "forwardReplayStart": "2026-01-01",
                "ensembleSeeds": list(range(10)),
                "architecture": "24 -> 64 tanh -> 64 tanh -> 3 categorical logits",
                "residualMultipliers": [0.5, 1.0, 1.5],
                "featureNames": LIVE_FEATURE_NAMES,
                "normalizer": (
                    f"QQQ train rows {train.index[0].date()} through "
                    f"{train.index[-1].date()}"
                ),
                "training": {
                    "timesteps": args.timesteps,
                    "bootstrapPaths": 3,
                    "episodeLength": 252,
                    "switchPenaltyBps": 5.0,
                    "entropyCoefficient": 0.005,
                    "calendarFeatures": False,
                },
                "validation": {
                    "numpyVsInMemorySb3MaxAbsError": max(numpy_errors),
                    "comparisonWindowEnd": "2026-07-30",
                    "holdoutStatus": (
                        "The 2026 holdout was already spent before v8 selection; "
                        "this is implementation parity, not a fresh model test."
                    ),
                },
            },
            LEGACY_MODEL_VERSION: {
                "role": "audited_legacy_and_historical_replay",
                "policyName": "v4 residual",
                "config": "ppo_v4_resid",
                "artifact": LEGACY_OUTPUT.name,
                "artifactSha256": legacy_digest,
                "trainCutoff": TRAIN_CUTOFF,
                "featureCount": 24,
                "validation": {
                    "savedHoldoutParity": "Covered by tests/test_live.py",
                    "savedHoldoutEnd": "2026-07-30",
                },
            },
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
