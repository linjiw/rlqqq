"""Compare experiment arms side by side on the concatenated OOS window.

Usage: .venv/bin/python scripts/compare_configs.py ppo_v1 ppo_v2_volpen ppo_v2_boot
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
from rlqqq.stats import iqm, iqm_ci, paired_bootstrap_delta
from rlqqq.walkforward import load_folds

RESULTS = ROOT / "results"


def decode_dates(raw):
    d = pd.DatetimeIndex(raw.astype("datetime64[ns]"))
    if d.min().year > 1971:
        return d
    return pd.DatetimeIndex((raw * 1000).astype("datetime64[ns]"))


def concat_oos(config: str, symbol: str, reg: pd.DataFrame, folds) -> dict | None:
    nets, exps = [], []
    for f in folds:
        sub = reg[(reg["config"] == config) & (reg["fold"] == f.name)]
        if sub.empty:
            return None
        med_seed = int(sub.sort_values("val_sharpe").iloc[len(sub) // 2]["seed"])
        p = RESULTS / "series" / f"{config}_{symbol}_{f.name}_s{med_seed}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        nets.append(z["test_net"])
        exps.append(z["test_exposure"])
    return {"net": np.concatenate(nets), "exposure": np.concatenate(exps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--cost_bps", type=float, default=2.0)
    args = ap.parse_args()

    reg = pd.DataFrame([json.loads(l) for l in (RESULTS / "registry.jsonl").open()])
    reg = reg[reg["symbol"] == args.symbol]
    market = load_market(args.symbol)
    folds = [f for f in load_folds()]

    # baselines on the same concatenated OOS window
    base_nets = {k: [] for k in ["buy_hold", "ma200", "vol_target_10"]}
    cash_all = []
    for f in folds:
        test = market.slice(f.test_start, f.test_end)
        b = run_baselines(test, cost_bps=args.cost_bps)
        for k in base_nets:
            base_nets[k].append(b[k]["net"])
        cash_all.append(test.cash)
    cash = np.concatenate(cash_all)
    bh = np.concatenate(base_nets["buy_hold"])

    rows = []
    for k in ["buy_hold", "ma200", "vol_target_10"]:
        x = np.concatenate(base_nets[k])
        p = perf(x, cash)
        rows.append({"config": f"[{k}]", "CAGR": p["cagr"], "Sharpe": p["sharpe"],
                     "Sortino": p["sortino"], "MaxDD": p["max_dd"],
                     "Calmar": p["calmar"]})

    for cfg in args.configs:
        oos = concat_oos(cfg, args.symbol, reg, folds)
        sub = reg[reg["config"] == cfg]
        if oos is None or sub.empty:
            print(f"({cfg}: missing runs, skipped)")
            continue
        p = perf(oos["net"], cash)
        to = turnover_stats(oos["exposure"])
        d = paired_bootstrap_delta(oos["net"], bh, n_reps=5000)
        ic = iqm_ci(sub["test_sharpe"].dropna().values)
        rows.append({
            "config": cfg, "CAGR": p["cagr"], "Sharpe": p["sharpe"],
            "Sortino": p["sortino"], "MaxDD": p["max_dd"], "Calmar": p["calmar"],
            "avg_w": to["avg_exposure"], "turnover": to["ann_turnover"],
            "dSh_vs_BH": round(d["delta"], 3),
            "dSh_CI": f"[{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}]",
            "seed_IQM_Sh": round(ic["iqm"], 2),
            "seed_CI": f"[{ic['ci_lo']:.2f},{ic['ci_hi']:.2f}]",
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\n=== {args.symbol}, concatenated OOS 2010-2025, {args.cost_bps} bps ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
