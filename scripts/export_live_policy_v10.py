"""Train and export the frozen v10 macro ensemble as a NumPy actor bundle.

v10 contract differences from v8:
  - 28 raw features (22 core + 6 cross-asset/macro)
  - residual baseline = min(0.20 / realized_vol_21, 1.0), warmup 0.5
  - 5 residual multipliers {0.5, 0.75, 1.0, 1.25, 1.5}
  - exposure cap 1.5 (leveraged; financing modeled at T-bill + 50bp)
  - benchmark-relative training reward (inference is reward-free)

Release-time operation; does not append to the experiment registry.
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
from rlqqq.walkforward import FoldSpec

MODEL_VERSION = "ppo_v10_macro_frozen_2023_v1"
POLICY_NAME = "v10 macro (leveraged)"
CONFIG_NAME = "ppo_v10_macro"
TRAIN_CUTOFF = "2023-12-31"
OUTPUT = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"

FOLD = FoldSpec(
    name="H2026",
    train_start="1994-01-01",
    train_end=TRAIN_CUTOFF,
    val_start="2024-01-31",
    val_end="2025-12-31",
    test_start="2026-01-01",
    test_end="2026-12-31",
)
HP = {
    "n_boot_paths": 3,
    "residual": True,
    "switch_penalty_bps": 5.0,
    "ent_coef": 0.005,
    "vt_target": 0.20,
    "max_exposure": 1.5,
    "relative_reward": True,
    "n_multipliers": 5,
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
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rlqqq.env import ExposureTradingEnv, run_policy
    from rlqqq.synth import bootstrap_path
    from rlqqq.walkforward import make_ppo

    torch.set_num_threads(1)
    market = load_market("QQQ", drop_calendar=True, with_cross_asset=True)
    train = market.slice(FOLD.train_start, FOLD.train_end)
    test = market.slice(FOLD.test_start, FOLD.test_end)
    normalizer = Normalizer.fit(train.feat)

    train_sets = [train for _ in range(4)]
    for index in range(3):
        train_sets[index + 1] = bootstrap_path(train, seed=seed * 100 + index)

    def make_env(index: int):
        def factory():
            return ExposureTradingEnv(
                train_sets[index], normalizer, cost_bps=2.0, discrete=True,
                episode_len=252, seed=seed * 1000 + index, residual=True,
                switch_penalty_bps=5.0, max_exposure=1.5, vt_target=0.20,
                relative_reward=True, n_multipliers=5,
            )
        return factory

    vector_env = DummyVecEnv([make_env(index) for index in range(4)])
    model = make_ppo(vector_env, seed, HP)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    run = run_policy(test, normalizer, model, cost_bps=2.0, discrete=True,
                     residual=True, max_exposure=1.5, vt_target=0.20,
                     n_multipliers=5)
    state = model.policy.state_dict()
    arrays = {
        out: state[key].detach().cpu().numpy().astype(np.float32)
        for out, key in STATE_KEYS.items()
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

    market = load_market("QQQ", drop_calendar=True, with_cross_asset=True)
    train = market.slice(FOLD.train_start, FOLD.train_end)
    normalizer = Normalizer.fit(train.feat)
    print(f"v10 features ({len(market.feat_names)}): {market.feat_names}")

    completed: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_seed, s, args.timesteps): s
                   for s in range(10)}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"  seed {result['seed']}: trained", flush=True)

    completed.sort(key=lambda r: r["seed"])
    arrays = {name: np.stack([r["arrays"][name] for r in completed])
              for name in STATE_KEYS}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        model_version=np.array(MODEL_VERSION),
        policy_name=np.array(POLICY_NAME),
        config_name=np.array(CONFIG_NAME),
        train_cutoff=np.array(TRAIN_CUTOFF),
        seeds=np.arange(10, dtype=np.int16),
        feature_names=np.array(market.feat_names),
        normalizer_mean=normalizer.mean.astype(np.float64),
        normalizer_std=normalizer.std.astype(np.float64),
        residual_multipliers=np.array([0.5, 0.75, 1.0, 1.25, 1.5]),
        vt_target=np.array(0.20),
        max_exposure=np.array(1.5),
        **arrays,
    )
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    # store SB3 parity data alongside for the validation step
    parity = ROOT / "models" / "live" / f"{MODEL_VERSION}.parity.npz"
    np.savez_compressed(
        parity,
        dates=completed[0]["sb3_dates"],
        exposure=np.stack([r["sb3_exposure"] for r in completed]),
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size/1024:.1f} KiB)")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
