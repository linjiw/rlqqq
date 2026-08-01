"""ONE-SHOT 2026 holdout evaluation, per docs/holdout_preregistration.md.

Primary (pre-registered, frozen 2026-08-01):
  w = 0.5 * mean(ppo_v2_boot QQQ seeds 0-14) + 0.5 * vol_target_10
  agents trained with train <= 2023-12-31, val 2024-2025, embargo 21d,
  evaluated once on 2026-01-01+.

Secondary (exploratory, NOT pre-registered — labeled as such):
  ppo_v4_resid ensemble (seeds 0-9) on the same fold.

This script is intended to be run ONCE. Results land in
results/holdout_2026_report.md and the registry.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlqqq.walkforward import FoldSpec

HOLDOUT_FOLD = FoldSpec(
    name="H2026",
    train_start="1994-01-01",
    train_end="2023-12-31",
    val_start="2024-01-31",
    val_end="2025-12-31",
    test_start="2026-01-01",
    test_end="2026-12-31",
)

CONFIGS = {
    "holdout_v2_boot": {"n_boot_paths": 3},
    "holdout_v4_resid": {"n_boot_paths": 3, "residual": True,
                         "switch_penalty_bps": 5.0, "ent_coef": 0.005},
}


def one_run(args):
    config, seed = args
    from rlqqq.data import load_market
    from rlqqq.walkforward import train_and_eval_one

    market = load_market("QQQ")
    rec = train_and_eval_one(market, HOLDOUT_FOLD, seed=seed,
                             config_name=config, timesteps=150_000,
                             hp=CONFIGS[config])
    return (config, seed, rec.test_sharpe, rec.test_cagr)


def main():
    jobs = [("holdout_v2_boot", s) for s in range(15)] + \
           [("holdout_v4_resid", s) for s in range(10)]
    print(f"{len(jobs)} holdout training runs (train<=2023-12-31)")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(one_run, j): j for j in jobs}
        for fut in as_completed(futs):
            cfg, seed, ts, tc = fut.result()
            done += 1
            print(f"[{done:2d}/{len(jobs)}] {cfg} s{seed}: 2026_sh={ts:5.2f} "
                  f"2026_cagr={tc:6.3f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
