"""Performance metrics computed on daily net-return series."""

from __future__ import annotations

import numpy as np

FREQ = 252


def perf(net: np.ndarray, cash: np.ndarray | None = None) -> dict:
    net = np.asarray(net, dtype=np.float64)
    n = len(net)
    if n < 2:
        return {}
    curve = np.cumprod(1.0 + net)
    years = n / FREQ
    cagr = curve[-1] ** (1 / years) - 1.0
    vol = net.std(ddof=1) * np.sqrt(FREQ)
    excess = net - (cash if cash is not None else 0.0)
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(FREQ) if excess.std(ddof=1) > 0 else np.nan
    downside = net[net < 0]
    sortino = (net.mean() * FREQ) / (downside.std(ddof=1) * np.sqrt(FREQ)) if len(downside) > 1 else np.nan
    peak = np.maximum.accumulate(curve)
    mdd = float((curve / peak - 1.0).min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {
        "n_days": n,
        "years": round(years, 2),
        "total_multiple": round(float(curve[-1]), 4),
        "cagr": round(float(cagr), 4),
        "vol": round(float(vol), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd": round(mdd, 4),
        "calmar": round(float(calmar), 3),
    }


def turnover_stats(exposure: np.ndarray, w_init: float = 0.0) -> dict:
    w = np.asarray(exposure, dtype=np.float64)
    prev = np.concatenate([[w_init], w[:-1]])
    t = np.abs(w - prev)
    return {
        "ann_turnover": round(float(t.sum() / (len(w) / FREQ)), 2),
        "avg_exposure": round(float(w.mean()), 3),
        "pct_days_full": round(float((w >= 0.99).mean()), 3),
        "pct_days_cash": round(float((w <= 0.01).mean()), 3),
    }
