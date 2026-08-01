"""Analyze pilot results: per-fold IQM vs baselines, concatenated OOS curve,
paired bootstrap, exposure decomposition, DSR from the trial registry.

Usage: .venv/bin/python scripts/analyze_pilot.py [--config ppo_v1] [--symbol SPY]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.data import load_market
from rlqqq.baselines import run_baselines
from rlqqq.metrics import perf, turnover_stats
from rlqqq.stats import (dsr, exposure_decomposition, iqm, iqm_ci,
                         paired_bootstrap_delta, sharpe)
from rlqqq.walkforward import load_folds

RESULTS = ROOT / "results"


def load_registry(config: str, symbol: str) -> pd.DataFrame:
    rows = []
    with (RESULTS / "registry.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["config"] == config and r["symbol"] == symbol:
                rows.append(r)
    return pd.DataFrame(rows)


def load_series(config: str, symbol: str, fold: str, seed: int):
    p = RESULTS / "series" / f"{config}_{symbol}_{fold}_s{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {
        "net": z["test_net"], "exposure": z["test_exposure"],
        "dates": pd.DatetimeIndex(z["test_dates"].astype("datetime64[ns]")),
    }


def median_seed_by_val(reg: pd.DataFrame, fold: str) -> int:
    """Deterministic seed selection: median validation Sharpe (never the best)."""
    sub = reg[reg["fold"] == fold].sort_values("val_sharpe")
    return int(sub.iloc[len(sub) // 2]["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ppo_v1")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--cost_bps", type=float, default=2.0)
    args = ap.parse_args()

    reg = load_registry(args.config, args.symbol)
    if reg.empty:
        sys.exit(f"no registry rows for {args.config}/{args.symbol}")
    market = load_market(args.symbol)
    folds = load_folds()
    fold_names = [f.name for f in folds if f.name in set(reg["fold"])]

    print(f"=== {args.config} on {args.symbol}: {len(reg)} runs, "
          f"{len(fold_names)} folds, {reg['seed'].nunique()} seeds ===\n")

    # ---- per-fold table: agent IQM vs baselines --------------------------
    rows = []
    agent_concat, bh_concat, ma_concat, vt_concat, cash_concat = [], [], [], [], []
    exp_concat = []
    for f in folds:
        if f.name not in fold_names:
            continue
        test = market.slice(f.test_start, f.test_end)
        base = run_baselines(test, cost_bps=args.cost_bps)
        sub = reg[reg["fold"] == f.name]
        seed_sh = sub.set_index("seed")["test_sharpe"]

        # concatenated OOS: median-validation seed per fold (deterministic,
        # selection uses val only)
        med_seed = median_seed_by_val(reg, f.name)
        ser = load_series(args.config, args.symbol, f.name, med_seed)
        if ser is not None:
            agent_concat.append(ser["net"])
            exp_concat.append(ser["exposure"])
            bh_concat.append(base["buy_hold"]["net"])
            ma_concat.append(base["ma200"]["net"])
            vt_concat.append(base["vol_target_10"]["net"])
            cash_concat.append(test.cash)

        rows.append({
            "fold": f.name,
            "test_window": f"{f.test_start[:7]}..{f.test_end[:7]}",
            "agent_iqm_sharpe": round(iqm(seed_sh.values), 2),
            "agent_sh_min": round(float(seed_sh.min()), 2),
            "agent_sh_max": round(float(seed_sh.max()), 2),
            "bh_sharpe": perf(base["buy_hold"]["net"], test.cash)["sharpe"],
            "ma200_sharpe": perf(base["ma200"]["net"], test.cash)["sharpe"],
            "vt10_sharpe": perf(base["vol_target_10"]["net"], test.cash)["sharpe"],
            "agent_iqm_cagr": round(iqm(sub.set_index("seed")["test_cagr"].values), 3),
            "bh_cagr": perf(base["buy_hold"]["net"], test.cash)["cagr"],
        })
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))

    # fold-level win counts
    wins_bh = (tab["agent_iqm_sharpe"] > tab["bh_sharpe"]).sum()
    print(f"\nfolds where agent IQM Sharpe > B&H: {wins_bh}/{len(tab)}")

    # ---- concatenated OOS analysis ---------------------------------------
    if agent_concat:
        a = np.concatenate(agent_concat)
        b = np.concatenate(bh_concat)
        m = np.concatenate(ma_concat)
        v = np.concatenate(vt_concat)
        c = np.concatenate(cash_concat)
        w = np.concatenate(exp_concat)

        print("\n=== Concatenated OOS (median-val seed per fold) ===")
        for name, x in [("agent", a), ("buy_hold", b), ("ma200", m), ("vol_target", v)]:
            p = perf(x, c)
            print(f"{name:<11} CAGR={p['cagr']:6.3f} Sharpe={p['sharpe']:5.2f} "
                  f"MaxDD={p['max_dd']:7.3f} Sortino={p['sortino']:5.2f}")
        to = turnover_stats(w)
        print(f"agent avg_exposure={to['avg_exposure']:.2f} "
              f"ann_turnover={to['ann_turnover']:.1f} "
              f"days_full={to['pct_days_full']:.0%} days_cash={to['pct_days_cash']:.0%}")

        print("\n--- paired stationary bootstrap (10k reps), agent vs B&H ---")
        for metric in ["sharpe", "cagr"]:
            d = paired_bootstrap_delta(a, b, n_reps=10_000, metric=metric)
            print(f"delta_{metric}: {d['delta']:+.3f}  "
                  f"95% CI [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  "
                  f"P(>0)={d['p_gt_zero']:.2f}  significant={d['significant']}")

        dec = exposure_decomposition(a, w, np.concatenate(
            [market.slice(f.test_start, f.test_end).ret for f in folds if f.name in fold_names]), c)
        print(f"\nexposure decomposition: avg_w={dec['avg_exposure']:.2f} "
              f"ann_alpha={dec['ann_alpha']:+.4f} t(NW)={dec['alpha_tstat_nw']:+.2f}")

        # ---- DSR using full registry as trial universe --------------------
        all_reg = pd.DataFrame([json.loads(l) for l in
                                (RESULTS / "registry.jsonl").open()])
        n_trials = len(all_reg)
        daily_sr = all_reg["test_sharpe"].dropna() / np.sqrt(252)
        d = dsr(a, n_trials=n_trials, var_sharpe_daily=float(daily_sr.var()))
        print(f"\nDSR (N={n_trials} registry trials): {d:.3f} "
              f"(need > 0.95 for a credible positive)")

        np.savez_compressed(RESULTS / f"oos_{args.config}_{args.symbol}.npz",
                            agent=a, buy_hold=b, exposure=w, cash=c)

    # ---- seed-distribution summary ---------------------------------------
    print("\n=== seed distribution of test Sharpe (all folds pooled) ===")
    ic = iqm_ci(reg["test_sharpe"].dropna().values)
    print(f"IQM={ic['iqm']:.2f} 95% CI [{ic['ci_lo']:.2f}, {ic['ci_hi']:.2f}] "
          f"min={reg['test_sharpe'].min():.2f} max={reg['test_sharpe'].max():.2f}")


if __name__ == "__main__":
    main()
