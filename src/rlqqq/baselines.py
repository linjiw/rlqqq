"""Baseline strategies, all expressed as exposure series in [0, 1] and run
through the exact same accounting identity as the RL agent (env.portfolio_returns).

Signals are computed from information through close of day t and produce the
exposure held over t -> t+1 — the same timing convention as the environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketData, PROCESSED
from .env import portfolio_returns


def _prices_for(data: MarketData) -> pd.Series:
    px = pd.read_parquet(PROCESSED / f"prices_{data.symbol}.parquet")
    return px["adj_close"].reindex(data.index)


def buy_and_hold(data: MarketData) -> np.ndarray:
    return np.ones(len(data))


def all_cash(data: MarketData) -> np.ndarray:
    return np.zeros(len(data))


def ma_rule(data: MarketData, window: int = 200) -> np.ndarray:
    """Long when close > MA(window) at close t, else cash."""
    px = pd.read_parquet(PROCESSED / f"prices_{data.symbol}.parquet")["adj_close"]
    sig = (px > px.rolling(window).mean()).astype(float)
    return sig.reindex(data.index).fillna(0.0).to_numpy()


def vol_target(data: MarketData, target: float = 0.10, lookback: int = 21) -> np.ndarray:
    """Exposure = min(1, target / realized vol), computed through close t."""
    px = pd.read_parquet(PROCESSED / f"prices_{data.symbol}.parquet")["adj_close"]
    r = px.pct_change()
    rv = r.rolling(lookback).std() * np.sqrt(252)
    w = (target / rv).clip(upper=1.0)
    return w.reindex(data.index).fillna(0.0).to_numpy()


def tsmom(data: MarketData, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """12-1 time-series momentum: long if return over [t-252, t-21] > 0."""
    px = pd.read_parquet(PROCESSED / f"prices_{data.symbol}.parquet")["adj_close"]
    mom = px.shift(skip) / px.shift(lookback) - 1.0
    sig = (mom > 0).astype(float)
    return sig.reindex(data.index).fillna(0.0).to_numpy()


def dca_returns(data: MarketData, contribution_every: int = 21) -> np.ndarray:
    """Dollar-cost averaging with periodic contributions. Not an exposure
    series (cash flows differ), so it returns the money-weighted daily return
    of the DCA portfolio for reporting purposes: value-weighted blend of the
    already-invested portfolio (earning asset return) and the not-yet-invested
    cash (earning T-bill).

    The fair 'DCA benchmark' comparison for an agent with the same drip
    schedule is handled at analysis time; this gives the reference curve.
    """
    n = len(data)
    n_contrib = n // contribution_every + 1
    invested_frac = np.minimum(
        np.arange(n) // contribution_every + 1, n_contrib
    ) / n_contrib
    # invested fraction earns asset return, remainder earns cash
    return invested_frac * data.ret + (1 - invested_frac) * data.cash


BASELINES = {
    "buy_hold": buy_and_hold,
    "cash": all_cash,
    "ma200": ma_rule,
    "vol_target_10": vol_target,
    "tsmom_12_1": tsmom,
}


def run_baselines(data: MarketData, cost_bps: float = 2.0) -> dict[str, dict]:
    """Run every exposure-based baseline through the shared accounting."""
    out = {}
    for name, fn in BASELINES.items():
        w = fn(data)
        net = portfolio_returns(w, data.ret, data.cash, cost_bps)
        out[name] = {"exposure": w, "net": net}
    out["dca_monthly"] = {"exposure": None, "net": dca_returns(data)}
    return out
