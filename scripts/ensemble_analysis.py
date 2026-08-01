"""Seed-ensemble analysis from saved series — no retraining needed.

For each fold, average the 10 seeds' daily exposure series (mean policy) or
take their median, re-run the accounting identity, and compare against the
median-val-seed selection we've been reporting. Also: ensemble-size curve,
turnover effect (averaging smooths switches), and an explicit DCA comparison
of the best agent.

Usage: .venv/bin/python scripts/ensemble_analysis.py --config ppo_v2_boot --symbol QQQ
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
from rlqqq.baselines import run_baselines, dca_returns
from rlqqq.env import portfolio_returns
from rlqqq.metrics import perf, turnover_stats
from rlqqq.stats import paired_bootstrap_delta
from rlqqq.walkforward import load_folds

RESULTS = ROOT / "results"


def load_fold_exposures(config, symbol, fold_name, seeds):
    out = {}
    for s in seeds:
        p = RESULTS / "series" / f"{config}_{symbol}_{fold_name}_s{s}.npz"
        if p.exists():
            out[s] = np.load(p)["test_exposure"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ppo_v2_boot")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--cost_bps", type=float, default=2.0)
    args = ap.parse_args()

    reg = pd.DataFrame([json.loads(l) for l in (RESULTS / "registry.jsonl").open()])
    reg = reg[(reg["config"] == args.config) & (reg["symbol"] == args.symbol)]
    seeds = sorted(reg["seed"].unique())
    market = load_market(args.symbol)
    folds = load_folds()

    variants = {
        "median_val_seed": [], "mean_exposure": [], "median_exposure": [],
        "mean_exp_rounded": [],
    }
    bh_all, vt_all, cash_all, ret_all, dca_all = [], [], [], [], []

    for f in folds:
        test = market.slice(f.test_start, f.test_end)
        exps = load_fold_exposures(args.config, args.symbol, f.name, seeds)
        if len(exps) < len(seeds):
            print(f"({f.name}: only {len(exps)} series)")
        E = np.stack(list(exps.values()))          # (S, T)

        sub = reg[reg["fold"] == f.name].sort_values("val_sharpe")
        med_seed = int(sub.iloc[len(sub) // 2]["seed"])

        w_sel = exps[med_seed]
        w_mean = E.mean(axis=0)
        w_med = np.median(E, axis=0)
        # snap mean to the discrete grid (nearest of 0, .5, 1) -> lower churn
        w_snap = np.round(w_mean * 2) / 2

        for name, w in [("median_val_seed", w_sel), ("mean_exposure", w_mean),
                        ("median_exposure", w_med), ("mean_exp_rounded", w_snap)]:
            variants[name].append(
                (w, portfolio_returns(w, test.ret, test.cash, args.cost_bps)))

        base = run_baselines(test, cost_bps=args.cost_bps)
        bh_all.append(base["buy_hold"]["net"])
        vt_all.append(base["vol_target_10"]["net"])
        dca_all.append(dca_returns(test))
        cash_all.append(test.cash)
        ret_all.append(test.ret)

    cash = np.concatenate(cash_all)
    bh = np.concatenate(bh_all)
    vt = np.concatenate(vt_all)
    dca = np.concatenate(dca_all)

    print(f"=== {args.config} / {args.symbol}: ensemble variants, "
          f"concatenated OOS, {args.cost_bps} bps ===\n")
    rows = []
    for name, series in variants.items():
        w = np.concatenate([x[0] for x in series])
        net = np.concatenate([x[1] for x in series])
        p = perf(net, cash)
        to = turnover_stats(w)
        d = paired_bootstrap_delta(net, bh, n_reps=5000)
        rows.append({"variant": name, "CAGR": p["cagr"], "Sharpe": p["sharpe"],
                     "MaxDD": p["max_dd"], "Calmar": p["calmar"],
                     "turnover": to["ann_turnover"], "avg_w": to["avg_exposure"],
                     "dSh_vs_BH": round(d["delta"], 3),
                     "CI": f"[{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}]"})
    for nm, x in [("[buy_hold]", bh), ("[vol_target_10]", vt), ("[dca_monthly]", dca)]:
        p = perf(x, cash)
        rows.append({"variant": nm, "CAGR": p["cagr"], "Sharpe": p["sharpe"],
                     "MaxDD": p["max_dd"], "Calmar": p["calmar"]})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- ensemble-size curve (mean exposure over first k seeds) ----------
    print("\nensemble-size curve (Sharpe of mean-exposure ensemble):")
    for k in [1, 2, 3, 5, 7, 10]:
        nets = []
        for f in folds:
            test = market.slice(f.test_start, f.test_end)
            exps = load_fold_exposures(args.config, args.symbol, f.name, seeds[:k])
            E = np.stack(list(exps.values()))
            nets.append(portfolio_returns(E.mean(axis=0), test.ret, test.cash,
                                          args.cost_bps))
        net = np.concatenate(nets)
        print(f"  k={k:2d}: Sharpe={perf(net, cash)['sharpe']:.3f} "
              f"CAGR={perf(net, cash)['cagr']:.3f}")

    # ---- explicit DCA comparison (drip schedule matched) ------------------
    print("\n--- agent (mean-exposure) vs DCA and vs vol-target, paired bootstrap ---")
    net_mean = np.concatenate([x[1] for x in variants["mean_exposure"]])
    for nm, ref in [("DCA", dca), ("vol_target", vt)]:
        d = paired_bootstrap_delta(net_mean, ref, n_reps=5000)
        print(f"  vs {nm:<11} dSharpe={d['delta']:+.3f} CI [{d['ci_lo']:+.3f},"
              f"{d['ci_hi']:+.3f}] sig={d['significant']}")


if __name__ == "__main__":
    main()
