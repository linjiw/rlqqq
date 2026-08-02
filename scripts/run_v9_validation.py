"""Validation runs for v9_rel5 (benchmark-relative, 5-level, leveraged):
  A) NDX era-holdout folds (test 2000-2009) - crisis-decade validation
  B) QQQ 2026 fold (train<=2023) - exploratory current-year check

Usage: .venv/bin/python scripts/run_v9_validation.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlqqq.walkforward import FoldSpec

HP = {"n_boot_paths": 3, "residual": True, "switch_penalty_bps": 5.0,
      "ent_coef": 0.005, "vt_target": 0.20, "max_exposure": 1.5,
      "relative_reward": True, "n_multipliers": 5}

H2026 = FoldSpec("H2026", "1994-01-01", "2023-12-31", "2024-01-31",
                 "2025-12-31", "2026-01-01", "2026-12-31")


def one_run(args):
    kind, fold_idx, seed = args
    from rlqqq.data import load_market
    from rlqqq.walkforward import era_holdout_folds, train_and_eval_one

    hp = dict(HP)
    if kind == "era":
        market = load_market("NDX", drop_calendar=True)
        fold = era_holdout_folds()[fold_idx]
        cfg = "ppo_v9_rel5_era"
    else:
        market = load_market("QQQ", drop_calendar=True)
        fold = H2026
        cfg = "holdout_v9_rel5"
    rec = train_and_eval_one(market, fold, seed=seed, config_name=cfg,
                             timesteps=150_000, hp=hp)
    return (cfg, fold.name, seed, rec.test_sharpe, rec.test_cagr)


def main():
    jobs = [("era", f, s) for f in range(5) for s in range(10)] + \
           [("h2026", 0, s) for s in range(10)]
    print(f"{len(jobs)} validation runs")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(one_run, j): j for j in jobs}
        for fut in as_completed(futs):
            cfg, fold, seed, ts, tc = fut.result()
            done += 1
            print(f"[{done:2d}/{len(jobs)}] {cfg} {fold} s{seed}: sh={ts:5.2f} "
                  f"cagr={tc:6.3f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
