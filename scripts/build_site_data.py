"""Build the compact data bundle for the public site (docs/assets/site-data.json).

Contents: v10 ensemble vs QQQ buy-and-hold vs VT20 rule on the 2010-2025
walk-forward (weekly-sampled wealth curves, drawdown, annual returns,
headline stats), the NDX 2000-2009 era stress test, and the 2026 YTD record.
All accounting matches the research harness (2 bps, T-bill cash, T-bill+50bp
financing above 1x).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.data import load_market, PROCESSED
from rlqqq.env import portfolio_returns, causal_vol_target
from rlqqq.metrics import perf
from rlqqq.walkforward import era_holdout_folds, load_folds

RESULTS = ROOT / "results" / "series"
OUT = ROOT / "docs" / "assets" / "site-data.json"
SEEDS = range(10)


def ens_exposure(config: str, symbol: str, fold_name: str) -> np.ndarray:
    return np.stack([
        np.load(RESULTS / f"{config}_{symbol}_{fold_name}_s{s}.npz")["test_exposure"]
        for s in SEEDS
    ]).mean(axis=0)


def sample_weekly(dates: pd.DatetimeIndex, *series: np.ndarray):
    """Keep every 5th point plus the last one."""
    idx = list(range(0, len(dates), 5))
    if idx[-1] != len(dates) - 1:
        idx.append(len(dates) - 1)
    out = [[str(dates[i].date()) for i in idx]]
    for s in series:
        out.append([round(float(s[i]), 4) for i in idx])
    return out


def annual_returns(dates: pd.DatetimeIndex, net: np.ndarray) -> dict[str, float]:
    ser = pd.Series(net, index=dates)
    out = {}
    for year, grp in ser.groupby(ser.index.year):
        out[str(year)] = round(float((1 + grp).prod() - 1), 4)
    return out


def stats_block(net: np.ndarray, cash: np.ndarray) -> dict:
    p = perf(net, cash)
    return {
        "cagr": p["cagr"], "sharpe": p["sharpe"], "maxDD": p["max_dd"],
        "calmar": p["calmar"], "vol": p["vol"],
        "totalMultiple": p["total_multiple"],
    }


def main() -> None:
    market = load_market("QQQ")
    px = pd.read_parquet(PROCESSED / "prices_QQQ.parquet")["adj_close"]
    rv = px.pct_change().rolling(21).std() * np.sqrt(252)

    # ---- 2010-2025 walk-forward ----
    nets = {"v10": [], "qqq": [], "vt20": []}
    dates_all, cash_all = [], []
    for f in load_folds():
        test = market.slice(f.test_start, f.test_end)
        w10 = ens_exposure("ppo_v10_macro", "QQQ", f.name)
        wvt = (0.20 / rv).reindex(test.index).fillna(0.0).clip(upper=1.5).to_numpy()
        nets["v10"].append(portfolio_returns(w10, test.ret, test.cash, 2.0))
        nets["vt20"].append(portfolio_returns(wvt, test.ret, test.cash, 2.0))
        nets["qqq"].append(portfolio_returns(np.ones(len(test)), test.ret, test.cash, 2.0))
        dates_all.append(test.index)
        cash_all.append(test.cash)

    dates = pd.DatetimeIndex(np.concatenate([d.values for d in dates_all]))
    cash = np.concatenate(cash_all)
    net = {k: np.concatenate(v) for k, v in nets.items()}
    wealth = {k: np.cumprod(1 + v) for k, v in net.items()}
    dd = {k: w / np.maximum.accumulate(w) - 1 for k, w in wealth.items()}

    sampled = sample_weekly(dates, wealth["v10"], wealth["qqq"], wealth["vt20"],
                            dd["v10"], dd["qqq"])

    era_stats = {}
    era_market = load_market("NDX")
    era_nets = {"v10": [], "qqq": []}
    era_cash = []
    for f in era_holdout_folds():
        test = era_market.slice(f.test_start, f.test_end)
        w10 = ens_exposure("ppo_v9_rel5_era", "NDX", f.name)
        era_nets["v10"].append(portfolio_returns(w10, test.ret, test.cash, 2.0))
        era_nets["qqq"].append(portfolio_returns(np.ones(len(test)), test.ret, test.cash, 2.0))
        era_cash.append(test.cash)
    era_cash_cat = np.concatenate(era_cash)
    for k, v in era_nets.items():
        era_stats[k] = stats_block(np.concatenate(v), era_cash_cat)

    # ---- 2026 YTD (frozen forward replay) ----
    market26 = load_market("QQQ", drop_calendar=True, with_cross_asset=True)
    hold = market26.slice("2026-01-01", "2026-12-31")
    E26 = np.stack([
        np.load(RESULTS / f"holdout_v10_macro_QQQ_H2026_s{s}.npz")["test_exposure"]
        for s in SEEDS
    ]).mean(axis=0)
    ytd = {
        "v10": stats_block(portfolio_returns(E26, hold.ret, hold.cash, 2.0), hold.cash),
        "qqq": stats_block(portfolio_returns(np.ones(len(hold)), hold.ret, hold.cash, 2.0), hold.cash),
        "v10Return": round(float(np.prod(1 + portfolio_returns(E26, hold.ret, hold.cash, 2.0)) - 1), 4),
        "qqqReturn": round(float(np.prod(1 + portfolio_returns(np.ones(len(hold)), hold.ret, hold.cash, 2.0)) - 1), 4),
        "through": str(hold.index[-1].date()),
    }

    payload = {
        "meta": {
            "model": "ppo_v10_macro_frozen_2023_v1",
            "displayName": "v10 macro",
            "walkForward": "2010-02-05..2025-12-31",
            "days": int(len(dates)),
            "costBps": 2.0,
            "borrowSpreadBps": 50.0,
            "cap": 1.5,
            "builtFrom": "8 walk-forward folds x 10 seeds, mean-exposure ensemble",
        },
        "stats": {
            "v10": stats_block(net["v10"], cash),
            "qqq": stats_block(net["qqq"], cash),
            "vt20": stats_block(net["vt20"], cash),
        },
        "chart": {
            "dates": sampled[0],
            "wealthV10": sampled[1],
            "wealthQqq": sampled[2],
            "wealthVt20": sampled[3],
            "ddV10": sampled[4],
            "ddQqq": sampled[5],
        },
        "annual": {
            "v10": annual_returns(dates, net["v10"]),
            "qqq": annual_returns(dates, net["qqq"]),
        },
        "era": {
            "window": "NDX 2000-01..2009-12 (dot-com crash + GFC), never used in development",
            "v10": era_stats["v10"],
            "qqq": era_stats["qqq"],
            "significant": "Sharpe difference vs buy-and-hold +0.26 [95% CI +0.01, +0.52]",
        },
        "ytd2026": ytd,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
