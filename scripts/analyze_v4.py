"""Analyze v4 residual arms: seed-mean ensemble of exposures, comparison vs
vol-target / B&H / v2_boot ensemble, paired bootstrap, per-fold table.

Usage: .venv/bin/python scripts/analyze_v4.py --configs ppo_v4_resid ppo_v4_resid_nosp
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
from rlqqq.env import portfolio_returns
from rlqqq.metrics import perf, turnover_stats
from rlqqq.stats import paired_bootstrap_delta
from rlqqq.walkforward import load_folds

RESULTS = ROOT / "results"


def ensemble_net(config, symbol, folds, market, seeds, cost_bps):
    nets, ws = [], []
    for f in folds:
        test = market.slice(f.test_start, f.test_end)
        E = []
        for s in seeds:
            p = RESULTS / "series" / f"{config}_{symbol}_{f.name}_s{s}.npz"
            if p.exists():
                E.append(np.load(p)["test_exposure"])
        if not E:
            return None, None
        w = np.stack(E).mean(axis=0)
        ws.append(w)
        nets.append(portfolio_returns(w, test.ret, test.cash, cost_bps))
    return np.concatenate(nets), np.concatenate(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["ppo_v4_resid", "ppo_v4_resid_nosp"])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--cost_bps", type=float, default=2.0)
    args = ap.parse_args()

    market = load_market(args.symbol)
    folds = load_folds()
    seeds = list(range(20))

    bh_all, vt_all, cash_all = [], [], []
    for f in folds:
        test = market.slice(f.test_start, f.test_end)
        b = run_baselines(test, cost_bps=args.cost_bps)
        bh_all.append(b["buy_hold"]["net"])
        vt_all.append(b["vol_target_10"]["net"])
        cash_all.append(test.cash)
    bh, vt, cash = map(np.concatenate, (bh_all, vt_all, cash_all))

    rows = []
    for nm, x in [("[buy_hold]", bh), ("[vol_target_10]", vt)]:
        p = perf(x, cash)
        rows.append({"policy": nm, "CAGR": p["cagr"], "Sharpe": p["sharpe"],
                     "MaxDD": p["max_dd"], "Calmar": p["calmar"]})

    for cfg in ["ppo_v2_boot"] + args.configs:
        net, w = ensemble_net(cfg, args.symbol, folds, market, seeds, args.cost_bps)
        if net is None:
            print(f"({cfg}: no series)")
            continue
        p = perf(net, cash)
        to = turnover_stats(w)
        d_bh = paired_bootstrap_delta(net, bh, n_reps=5000)
        d_vt = paired_bootstrap_delta(net, vt, n_reps=5000)
        rows.append({
            "policy": f"{cfg} (ens)", "CAGR": p["cagr"], "Sharpe": p["sharpe"],
            "MaxDD": p["max_dd"], "Calmar": p["calmar"],
            "turnover": to["ann_turnover"], "avg_w": to["avg_exposure"],
            "dSh_BH": f"{d_bh['delta']:+.2f} [{d_bh['ci_lo']:+.2f},{d_bh['ci_hi']:+.2f}]"
                      + ("*" if d_bh["significant"] else ""),
            "dSh_VT": f"{d_vt['delta']:+.2f} [{d_vt['ci_lo']:+.2f},{d_vt['ci_hi']:+.2f}]"
                      + ("*" if d_vt["significant"] else ""),
        })

    pd.set_option("display.width", 250)
    print(f"=== {args.symbol} ensembles (mean exposure), concat OOS, "
          f"{args.cost_bps} bps ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
