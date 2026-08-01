"""WP-D: stationary block-bootstrap synthetic training paths.

Approach (van Staden et al. 2023 / Karzanov et al. 2025 template):
  - Resample contiguous blocks of TRAIN-window rows (geometric block lengths,
    mean BLOCK_MEAN days) and stitch them into a synthetic day sequence.
  - Path-dependent features (multi-horizon returns, vols, MA gaps, RSI, MACD,
    drawdown) are RECOMPUTED from the stitched return path, so state and
    future reward stay causally consistent on the synthetic path.
  - Day-local features (intraday range, gap, volume ratio, calendar, VIX,
    term spread) are carried from the sampled source days; within-block
    co-movement is preserved because blocks are contiguous.
  - Cash returns are carried from source days.

The generator only ever sees the training window - synthetic paths cannot
leak validation/test information by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketData

BLOCK_MEAN = 63  # mean block length in trading days (one quarter)

PATH_DERIVED = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_5d", "vol_21d", "vol_63d",
    "ma_gap_21", "ma_gap_50", "ma_gap_200",
    "rsi_14", "macd_hist", "drawdown",
]
WARMUP = 260


def _stationary_indices(n_source: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence."""
    idx = np.empty(n_out, dtype=np.int64)
    p = 1.0 / BLOCK_MEAN
    pos = 0
    while pos < n_out:
        start = int(rng.integers(0, n_source))
        length = int(rng.geometric(p))
        for k in range(length):
            if pos >= n_out:
                break
            idx[pos] = (start + k) % n_source
            pos += 1
    return idx


def _recompute_path_features(close: pd.Series) -> pd.DataFrame:
    """Path-derived features from a synthetic close series (same formulas as
    scripts/prepare_dataset.py::make_features)."""
    c = close
    logret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=c.index)
    for h in [1, 5, 10, 21, 63, 126, 252]:
        f[f"ret_{h}d"] = np.log(c / c.shift(h))
    for h in [5, 21, 63]:
        f[f"vol_{h}d"] = logret.rolling(h).std() * np.sqrt(252)
    for h in [21, 50, 200]:
        f[f"ma_gap_{h}"] = c / c.rolling(h).mean() - 1.0
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # all-up window => RSI is 100 by definition (dn == 0 after warmup)
    f["rsi_14"] = rsi.where(dn > 0, 100.0)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - sig) / c
    f["drawdown"] = c / c.cummax() - 1.0
    return f


def bootstrap_path(train: MarketData, seed: int) -> MarketData:
    """One synthetic path with the same length as the training window."""
    rng = np.random.default_rng(seed)
    n = len(train)
    total = n + WARMUP
    src = _stationary_indices(n, total, rng)

    ret = train.ret[src]
    cash = train.cash[src]

    # synthetic close path: ret[t] is the t -> t+1 return, so close[t] is the
    # cumulative product of returns up to (but excluding) t
    growth = np.concatenate([[1.0], np.cumprod(1.0 + ret[:-1])])
    idx = pd.RangeIndex(total)
    close = pd.Series(growth, index=idx)

    derived = _recompute_path_features(close)
    feat = pd.DataFrame(
        train.feat[src], columns=train.feat_names, index=idx
    )
    for col in PATH_DERIVED:
        if col in feat.columns:
            feat[col] = derived[col]

    # drop warmup rows where recomputed features are NaN
    keep = slice(WARMUP, total)
    feat_arr = feat.iloc[keep].to_numpy(dtype=np.float64)
    assert not np.isnan(feat_arr).any(), "NaNs in synthetic features"

    dates = pd.date_range("1900-01-01", periods=total - WARMUP, freq="B")
    return MarketData(
        symbol=f"{train.symbol}_boot{seed}",
        index=pd.DatetimeIndex(dates),
        ret=ret[keep],
        cash=cash[keep],
        feat=feat_arr,
        feat_names=train.feat_names,
    )


def stylized_fact_report(real: MarketData, synth: MarketData) -> dict:
    """Quick fidelity check: annualized vol, excess kurtosis, ACF of squared
    returns (vol clustering), and 12-1 momentum autocorrelation proxy."""
    def stats(r):
        r = np.asarray(r)
        acf_sq = [
            float(np.corrcoef(r[:-k] ** 2, r[k:] ** 2)[0, 1]) for k in (1, 5, 21)
        ]
        return {
            "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
            "kurtosis": float(pd.Series(r).kurtosis()),
            "acf_sq_1_5_21": [round(a, 3) for a in acf_sq],
            "ann_mean": float(r.mean() * 252),
        }
    return {"real": stats(real.ret), "synth": stats(synth.ret)}
