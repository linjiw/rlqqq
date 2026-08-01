"""Era-holdout validation: frozen v4 recipe on NDX with 2000-2009 test folds.

The recipe (residual actions on vol-target, 5bp switch penalty, 3 bootstrap
paths, mean-exposure ensemble) was developed entirely on 2010-2025 test
windows. Here it runs untouched on the dot-com crash / 2003-07 bull / GFC.

Usage: .venv/bin/python scripts/run_era_holdout.py [--seeds 10]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HP = {"n_boot_paths": 3, "residual": True, "switch_penalty_bps": 5.0,
      "ent_coef": 0.005}


def one_run(args):
    fold_idx, seed, timesteps = args
    from rlqqq.data import load_market
    from rlqqq.walkforward import era_holdout_folds, train_and_eval_one

    market = load_market("NDX")
    fold = era_holdout_folds()[fold_idx]
    rec = train_and_eval_one(market, fold, seed=seed,
                             config_name="ppo_v4_resid_era",
                             timesteps=timesteps, hp=HP)
    return (fold.name, seed, rec.val_sharpe, rec.test_sharpe, rec.test_cagr,
            rec.test_avg_exposure, rec.test_ann_turnover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    jobs = [(f, s, args.timesteps) for f in range(5) for s in range(args.seeds)]
    print(f"{len(jobs)} era-holdout runs on NDX")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_run, j): j for j in jobs}
        for fut in as_completed(futs):
            fold, seed, vs, ts, tc, ae, to = fut.result()
            done += 1
            print(f"[{done:3d}/{len(jobs)}] {fold} s{seed}: test_sh={ts:5.2f} "
                  f"cagr={tc:6.3f} exp={ae:4.2f} to={to:5.1f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
