"""Statistical inference: paired block bootstrap, PSR/DSR/MinTRL, IQM.

References:
  - Bailey & Lopez de Prado 2014 (Deflated Sharpe Ratio), JPM 40(5).
  - Politis & Romano 1994 stationary bootstrap via `arch`.
  - Agarwal et al. 2021 (IQM + stratified bootstrap), NeurIPS.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

FREQ = 252


# ---------------------------------------------------------------- Sharpe ----

def sharpe(net: np.ndarray, cash: np.ndarray | None = None) -> float:
    x = np.asarray(net, dtype=np.float64)
    if cash is not None:
        x = x - cash
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(FREQ)) if s > 0 else np.nan


def psr(net: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark), accounting for
    skew/kurtosis and track length (Bailey & LdP 2012). Inputs daily; SR terms
    are computed in per-period (daily) units as the formula requires."""
    x = np.asarray(net, dtype=np.float64)
    n = len(x)
    sr_hat = x.mean() / x.std(ddof=1)          # daily units
    sr0 = sr_benchmark / np.sqrt(FREQ)          # convert annual benchmark
    g3 = sps.skew(x)
    g4 = sps.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat**2))
    z = (sr_hat - sr0) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] over n_trials of null strategies (daily units), Bailey & LdP
    2014 eq. using EMC (Euler-Mascheroni)."""
    if n_trials <= 1:
        return 0.0
    emc = 0.5772156649015329
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sharpe) * ((1 - emc) * z1 + emc * z2))


def dsr(net: np.ndarray, n_trials: int, var_sharpe_daily: float) -> float:
    """Deflated Sharpe Ratio: PSR against E[max SR] of the trial universe.
    var_sharpe_daily: cross-sectional variance of DAILY Sharpe estimates
    across all trials in the registry."""
    sr_star_daily = expected_max_sharpe(n_trials, var_sharpe_daily)
    return psr(net, sr_benchmark=sr_star_daily * np.sqrt(FREQ))


# ------------------------------------------------------- paired bootstrap ----

def paired_bootstrap_delta(
    net_a: np.ndarray,
    net_b: np.ndarray,
    n_reps: int = 10_000,
    seed: int = 0,
    metric: str = "sharpe",
) -> dict:
    """Stationary-bootstrap CI for metric(A) - metric(B), resampling the SAME
    time blocks for both series (they share market state)."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    a = np.asarray(net_a, dtype=np.float64)
    b = np.asarray(net_b, dtype=np.float64)
    assert a.shape == b.shape
    blen = float(optimal_block_length(a).iloc[0]["stationary"])
    blen = max(blen, 5.0)

    def stat(x, y):
        if metric == "sharpe":
            sa = x.mean() / x.std(ddof=1) * np.sqrt(FREQ)
            sb = y.mean() / y.std(ddof=1) * np.sqrt(FREQ)
            return sa - sb
        if metric == "cagr":
            na, nb = len(x), len(y)
            return ((np.prod(1 + x)) ** (FREQ / na) - 1) - ((np.prod(1 + y)) ** (FREQ / nb) - 1)
        raise ValueError(metric)

    bs = StationaryBootstrap(blen, a, b, seed=seed)
    deltas = np.array([stat(*pos_args) for pos_args, _ in bs.bootstrap(n_reps)])
    point = stat(a, b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_gt_zero": float((deltas > 0).mean()),
        "block_length": blen,
        "significant": bool(lo > 0 or hi < 0),
    }


# ----------------------------------------------------------------- IQM ------

def iqm(values: np.ndarray) -> float:
    """Interquartile mean (middle 50%)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(v[lo:hi].mean())


def iqm_ci(values: np.ndarray, n_reps: int = 5000, seed: int = 0) -> dict:
    """Percentile-bootstrap CI for the IQM across seeds/folds."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    boots = np.array([
        iqm(rng.choice(v, size=len(v), replace=True)) for _ in range(n_reps)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"iqm": iqm(v), "ci_lo": float(lo), "ci_hi": float(hi)}


# --------------------------------------------- exposure decomposition --------

def exposure_decomposition(net: np.ndarray, exposure: np.ndarray,
                           ret: np.ndarray, cash: np.ndarray) -> dict:
    """How much of the strategy return is explained by passive exposure?
    alpha_t = net_t - [w_bar * ret_t + (1-w_bar) * cash_t] where w_bar is the
    average exposure. Reports annualized alpha and its t-stat (HAC via
    Newey-West with 21 lags)."""
    w_bar = float(np.mean(exposure))
    passive = w_bar * ret + (1 - w_bar) * cash
    alpha = net - passive
    n = len(alpha)
    # Newey-West standard error
    lags = 21
    mu = alpha.mean()
    e = alpha - mu
    gamma0 = float(e @ e) / n
    s = gamma0
    for k in range(1, lags + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2 * (1 - k / (lags + 1)) * gk
    se = np.sqrt(s / n)
    t = mu / se if se > 0 else np.nan
    return {
        "avg_exposure": w_bar,
        "ann_alpha": float(mu * FREQ),
        "alpha_tstat_nw": float(t),
    }
