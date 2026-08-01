"""PPO pilot on SPY: 8 walk-forward folds x N seeds, parallelized.

Usage: .venv/bin/python scripts/run_pilot.py [--seeds 10] [--timesteps 150000]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# named experiment arms: config -> hyperparameter overrides.
# "with_har" is a data flag (adds HAR-RV vol-forecast features to the state).
CONFIGS = {
    "ppo_v1": {},
    "ppo_v2_volpen": {"reward_lambda": 2.0},
    "ppo_v2_boot": {"n_boot_paths": 3},
    "ppo_v2_boot_volpen": {"n_boot_paths": 3, "reward_lambda": 2.0},
    "ppo_v3_cont": {"n_boot_paths": 3, "discrete": False},
    "ppo_v3_cont_harv": {"n_boot_paths": 3, "discrete": False, "with_har": True},
    "ppo_v3_harv": {"n_boot_paths": 3, "with_har": True},
    # v4: residual actions (multipliers on causal vol-target) + switch penalty,
    # on top of the winning bootstrap recipe. ent_coef lowered: residual mode
    # needs less exploration (action 1 = baseline is already good).
    "ppo_v4_resid": {"n_boot_paths": 3, "residual": True,
                     "switch_penalty_bps": 5.0, "ent_coef": 0.005},
    "ppo_v4_resid_nosp": {"n_boot_paths": 3, "residual": True,
                          "ent_coef": 0.005},
}


def one_run(args):
    fold_idx, seed, timesteps, symbol, config = args
    # imports inside worker to keep processes independent
    from rlqqq.data import load_market
    from rlqqq.walkforward import load_folds, train_and_eval_one

    hp = dict(CONFIGS.get(config, {}))
    with_har = hp.pop("with_har", False)
    market = load_market(symbol, with_har=with_har)
    fold = load_folds()[fold_idx]
    rec = train_and_eval_one(
        market, fold, seed=seed, config_name=config, timesteps=timesteps,
        hp=hp,
    )
    return (fold.name, seed, rec.val_sharpe, rec.test_sharpe, rec.test_cagr,
            rec.test_avg_exposure, rec.test_ann_turnover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--config", default="ppo_v1")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--folds", type=int, default=8)
    args = ap.parse_args()

    jobs = [
        (f, s, args.timesteps, args.symbol, args.config)
        for f in range(args.folds)
        for s in range(args.seed_start, args.seed_start + args.seeds)
    ]
    print(f"{len(jobs)} runs ({args.folds} folds x {args.seeds} seeds), "
          f"{args.workers} workers, {args.timesteps} steps each")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_run, j): j for j in jobs}
        for fut in as_completed(futs):
            fold, seed, vs, ts, tc, ae, to = fut.result()
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            print(f"[{done:3d}/{len(jobs)}] {fold} s{seed}: val_sh={vs:5.2f} "
                  f"test_sh={ts:5.2f} cagr={tc:6.3f} exp={ae:4.2f} to={to:5.1f} "
                  f"| {el/60:4.1f}m elapsed, ~{eta/60:4.1f}m left", flush=True)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. "
          f"Registry: results/registry.jsonl")


if __name__ == "__main__":
    main()
