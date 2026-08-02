"""v10_macro era-holdout validation on NDX 2000-2009."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HP = {"n_boot_paths": 3, "residual": True, "switch_penalty_bps": 5.0,
      "ent_coef": 0.005, "vt_target": 0.20, "max_exposure": 1.5,
      "relative_reward": True, "n_multipliers": 5}


def one_run(args):
    fold_idx, seed = args
    from rlqqq.data import load_market
    from rlqqq.walkforward import era_holdout_folds, train_and_eval_one
    market = load_market("NDX", drop_calendar=True, with_cross_asset=True)
    fold = era_holdout_folds()[fold_idx]
    rec = train_and_eval_one(market, fold, seed=seed,
                             config_name="ppo_v10_macro_era",
                             timesteps=150_000, hp=HP)
    return (fold.name, seed, rec.test_sharpe)


def main():
    jobs = [(f, s) for f in range(5) for s in range(10)]
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(one_run, j): j for j in jobs}
        for fut in as_completed(futs):
            fold, seed, ts = fut.result()
            done += 1
            print(f"[{done}/50] {fold} s{seed}: sh={ts:.2f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
