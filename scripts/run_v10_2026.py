"""v10_macro 2026 evaluation: train<=2023, evaluate on 2026 YTD (exploratory)."""
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


def one_run(seed):
    from rlqqq.data import load_market
    from rlqqq.walkforward import train_and_eval_one
    market = load_market("QQQ", drop_calendar=True, with_cross_asset=True)
    rec = train_and_eval_one(market, H2026, seed=seed,
                             config_name="holdout_v10_macro",
                             timesteps=150_000, hp=HP)
    return (seed, rec.test_sharpe, rec.test_cagr)


def main():
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(one_run, s): s for s in range(10)}
        for fut in as_completed(futs):
            seed, ts, tc = fut.result()
            done += 1
            print(f"[{done}/10] s{seed}: sh={ts:.2f} cagr={tc:.3f}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
