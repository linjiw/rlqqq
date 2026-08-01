"""Data loading utilities: aligned market frames for the environment.

The environment consumes a MarketData bundle:
  - ret:   next-day total return the position earns (close_t -> close_{t+1})
  - cash:  next-day cash return (3m T-bill, ^IRX, accrued daily)
  - feat:  raw (unnormalized) feature matrix, information through close of t
Row t therefore pairs "what you knew at close t" with "what you earn from
close t to close t+1". Normalization is fit later, per training window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


@dataclass
class MarketData:
    symbol: str
    index: pd.DatetimeIndex          # decision dates
    ret: np.ndarray                  # (T,) asset total return t->t+1
    cash: np.ndarray                 # (T,) cash return t->t+1
    feat: np.ndarray                 # (T, F) raw features known at close t
    feat_names: list[str]

    def slice(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> "MarketData":
        m = (self.index >= pd.Timestamp(start)) & (self.index <= pd.Timestamp(end))
        return MarketData(self.symbol, self.index[m], self.ret[m],
                          self.cash[m], self.feat[m], self.feat_names)

    def __len__(self) -> int:
        return len(self.index)


def har_rv_features(px: pd.DataFrame) -> pd.DataFrame:
    """HAR-RV (Corsi 2009) style volatility features + a causal forecast.

    Components: daily/weekly/monthly realized vol (already partially in the
    base features) plus an expanding-window HAR forecast of next-21d RV fitted
    ONLY on past data (coefficients re-fit monthly on data through t). The
    forecast at row t uses coefficients estimated from rows <= t-21 (targets
    fully realized), so there is no look-ahead.
    """
    r = np.log(px["adj_close"] / px["adj_close"].shift(1))
    rv_d = r.abs() * np.sqrt(252)
    rv_w = r.rolling(5).std() * np.sqrt(252)
    rv_m = r.rolling(21).std() * np.sqrt(252)
    target = r.shift(-1).rolling(21).std().shift(-20) * np.sqrt(252)  # RV over t+1..t+21

    X = pd.DataFrame({"d": rv_d, "w": rv_w, "m": rv_m})
    out = pd.Series(np.nan, index=px.index)
    dates = px.index
    # refit monthly, predict daily with last coefficients
    coef = None
    last_fit = None
    for i in range(504, len(dates)):
        if coef is None or (i - last_fit) >= 21:
            # rows whose 21d-forward target is fully realized by day i
            hist = slice(21, i - 21)
            Xh = X.iloc[hist].dropna()
            yh = target.iloc[hist].reindex(Xh.index).dropna()
            Xh = Xh.reindex(yh.index)
            if len(yh) > 100:
                A = np.column_stack([np.ones(len(Xh)), Xh.values])
                coef, *_ = np.linalg.lstsq(A, yh.values, rcond=None)
                last_fit = i
        if coef is not None and not X.iloc[i].isna().any():
            out.iloc[i] = coef[0] + coef[1:] @ X.iloc[i].values
    f = pd.DataFrame(index=px.index)
    f["har_rv_fcst"] = out
    f["rv_d"] = rv_d
    return f


def load_market(symbol: str = "SPY", with_context: bool = True,
                with_har: bool = False) -> MarketData:
    px = pd.read_parquet(PROCESSED / f"prices_{symbol}.parquet")
    feats = pd.read_parquet(PROCESSED / f"features_{symbol}.parquet")

    # next-day total return earned by holding from close t to close t+1
    nxt_ret = px["adj_close"].pct_change().shift(-1)

    # daily cash return from 3m T-bill discount yield. NOTE: unlike ^TNX/^FVX
    # (which quote yield*10), ^IRX quotes the discount rate in percent
    # directly (verified: 1990-06 raw value 7.68 vs historical 3m T-bill
    # ~7.7%). Do NOT divide by 10.
    irx = pd.read_csv(RAW / "yf_IDX_IRX.csv", parse_dates=["Date"]).set_index("Date")["Close"]
    cash_daily = (irx.reindex(px.index).ffill() / 100.0 / 252.0).fillna(0.0)

    cols = list(feats.columns)
    F = feats.copy()
    if with_context:
        ctx = pd.read_parquet(PROCESSED / "context.parquet")
        ctx = ctx[["vix", "term_spread_10y_3m", "vix_chg_5d"]].reindex(px.index).ffill()
        F = F.join(ctx)
        cols = list(F.columns)
    if with_har:
        F = F.join(har_rv_features(px))
        cols = list(F.columns)

    df = pd.DataFrame({"ret": nxt_ret, "cash": cash_daily}, index=px.index).join(F)
    # drop warmup rows (features with 252d lookback) and the final row (no t+1)
    df = df.dropna()

    return MarketData(
        symbol=symbol,
        index=pd.DatetimeIndex(df.index),
        ret=df["ret"].to_numpy(dtype=np.float64),
        cash=df["cash"].to_numpy(dtype=np.float64),
        feat=df[cols].to_numpy(dtype=np.float64),
        feat_names=cols,
    )


@dataclass
class Normalizer:
    """Per-feature affine normalization fit on a training window only."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feat: np.ndarray) -> "Normalizer":
        mean = feat.mean(axis=0)
        std = feat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        z = (feat - self.mean) / self.std
        return np.clip(z, -10.0, 10.0)
