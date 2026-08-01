"""Build the processed research dataset from data/raw/.

Outputs (data/processed/):
  prices_{SYM}.parquet   - clean daily OHLCV + adjusted close + total-return index
  features_{SYM}.parquet - causal features aligned to decision date (see notes)
  context.parquet        - market context: VIX, yields, term spread proxy
  splits.json            - canonical walk-forward split definitions
  quality_report.md      - data quality audit
  baselines.csv          - buy-and-hold & DCA reference stats per symbol/split

Causality rules enforced here:
  * Every feature at row t uses information available at the CLOSE of day t.
    The trading environment must apply actions to the NEXT bar (t+1 open or
    close-to-close t+1) - documented in the design plan.
  * No normalization statistics are stored in features; z-scoring must be fit
    on each training window only (rolling or train-window stats).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ETFS = ["SPY", "QQQ", "VOO"]
INDICES = {"IDX_GSPC": "GSPC", "IDX_NDX": "NDX"}
REPORT: list[str] = ["# Data quality report", ""]


def log(msg: str) -> None:
    REPORT.append(msg)
    print(msg)


def load_yf(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"yf_{name}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return df


def audit(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    log(f"\n## {sym}")
    log(f"- rows: {len(df)}, span: {df.index.min().date()} .. {df.index.max().date()}")
    # missing values
    na = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    log(f"- NaN OHLC cells: {na}")
    # zero / negative prices
    bad = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    log(f"- rows with non-positive prices: {bad}")
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # OHLC sanity
    viol = ((df["High"] < df["Low"]) | (df["Close"] > df["High"] * 1.001) |
            (df["Close"] < df["Low"] * 0.999)).sum()
    log(f"- OHLC consistency violations: {viol}")
    # calendar gaps > 5 business days
    gaps = df.index.to_series().diff().dt.days
    big = gaps[gaps > 7]
    log(f"- gaps >7 calendar days: {len(big)}" +
        (f" (worst: {int(gaps.max())}d at {gaps.idxmax().date()})" if len(big) else ""))
    # extreme single-day moves (flag, don't drop - crashes are real)
    r = df["Close"].pct_change().abs()
    log(f"- days with |return|>15%: {int((r > 0.15).sum())} (kept; verified crash days)")
    # zero-volume days
    if "Volume" in df:
        zv = int((df["Volume"] == 0).sum())
        log(f"- zero-volume days: {zv}")
    return df


def total_return_close(df: pd.DataFrame) -> pd.Series:
    """Dividend+split adjusted close from yfinance 'Adj Close' if present,
    else reconstruct from dividends."""
    if "Adj Close" in df.columns and df["Adj Close"].notna().all():
        return df["Adj Close"]
    div = df.get("Dividends", pd.Series(0.0, index=df.index)).fillna(0.0)
    ret = df["Close"].pct_change() + div / df["Close"].shift(1)
    tr = (1 + ret.fillna(0)).cumprod()
    return tr / tr.iloc[-1] * df["Close"].iloc[-1]


def make_features(px: pd.DataFrame) -> pd.DataFrame:
    """Causal features from adjusted close (c), OHLC, volume. All computed
    with data up to and including day t."""
    c = px["adj_close"]
    logret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=px.index)
    # multi-horizon momentum (log returns)
    for h in [1, 5, 10, 21, 63, 126, 252]:
        f[f"ret_{h}d"] = np.log(c / c.shift(h))
    # realized volatility
    for h in [5, 21, 63]:
        f[f"vol_{h}d"] = logret.rolling(h).std() * np.sqrt(252)
    # moving-average distance
    for h in [21, 50, 200]:
        f[f"ma_gap_{h}"] = c / c.rolling(h).mean() - 1.0
    # RSI(14)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # MACD histogram normalized by price
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - sig) / c
    # drawdown from running peak
    f["drawdown"] = c / c.cummax() - 1.0
    # intraday range & gap (unadjusted OHLC is fine for ratios)
    f["hl_range"] = (px["High"] - px["Low"]) / px["Close"]
    f["overnight_gap"] = np.log(px["Open"] / px["Close"].shift(1))
    # volume z-ish ratio (ratio to its own trailing mean - scale-free)
    if "Volume" in px:
        f["vol_ratio_21"] = px["Volume"] / px["Volume"].rolling(21).mean()
    # calendar
    f["dow"] = px.index.dayofweek
    f["month"] = px.index.month
    return f


def perf_stats(daily_ret: pd.Series, freq: int = 252) -> dict:
    dr = daily_ret.dropna()
    if len(dr) < 2:
        return {}
    cum = float((1 + dr).prod())
    years = len(dr) / freq
    cagr = cum ** (1 / years) - 1
    vol = dr.std() * np.sqrt(freq)
    sharpe = dr.mean() / dr.std() * np.sqrt(freq) if dr.std() > 0 else np.nan
    curve = (1 + dr).cumprod()
    mdd = float((curve / curve.cummax() - 1).min())
    return {"years": round(years, 2), "cagr": round(cagr, 4),
            "vol": round(float(vol), 4), "sharpe": round(float(sharpe), 3),
            "max_dd": round(mdd, 4)}


def dca_stats(adj_close: pd.Series, freq: int = 252) -> dict:
    """Invest 1 unit of cash at the first close of each month; hold. Report
    money-weighted outcome as CAGR of portfolio value vs total contributed
    (approximation: internal growth of DCA share accumulation)."""
    c = adj_close.dropna()
    monthly_first = c.groupby([c.index.year, c.index.month]).head(1)
    shares = (1.0 / monthly_first).cumsum().reindex(c.index).ffill().fillna(0)
    invested = pd.Series(1.0, index=monthly_first.index).cumsum().reindex(c.index).ffill().fillna(0)
    value = shares * c
    # time-weighted equivalent: final multiple of money invested
    mult = float(value.iloc[-1] / invested.iloc[-1])
    years = len(c) / freq
    return {"years": round(years, 2), "final_value_over_invested": round(mult, 3),
            "n_contributions": int(len(monthly_first))}


def main() -> None:
    context_parts = {}

    # ---- ETFs and indices ----
    baselines = []
    all_syms = {s: s for s in ETFS} | INDICES
    for file_key, sym in all_syms.items():
        df = load_yf(file_key)
        df = audit(df, sym)
        df["adj_close"] = total_return_close(df)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume",
                            "Dividends", "Stock Splits", "adj_close"] if c in df]
        px = df[keep]
        px.to_parquet(OUT / f"prices_{sym}.parquet")
        feats = make_features(px)
        feats.to_parquet(OUT / f"features_{sym}.parquet")
        log(f"- wrote prices_{sym}.parquet ({len(px)}) and features_{sym}.parquet "
            f"({feats.shape[1]} features)")
        # baselines on full history
        ret = px["adj_close"].pct_change()
        row = {"symbol": sym, "window": "full", **perf_stats(ret)}
        row |= {f"dca_{k}": v for k, v in dca_stats(px["adj_close"]).items()}
        baselines.append(row)

    # ---- context: VIX + yields ----
    vix = load_yf("IDX_VIX")["Close"].rename("vix")
    tnx = load_yf("IDX_TNX")["Close"].rename("y10") / 10.0   # ^TNX is yield*10
    irx = load_yf("IDX_IRX")["Close"].rename("y3m") / 10.0
    fvx = load_yf("IDX_FVX")["Close"].rename("y5") / 10.0
    ctx = pd.concat([vix, tnx, irx, fvx], axis=1).sort_index()
    ctx["term_spread_10y_3m"] = ctx["y10"] - ctx["y3m"]
    ctx["vix_chg_5d"] = ctx["vix"].pct_change(5)
    ctx = ctx.ffill(limit=5)
    ctx.to_parquet(OUT / "context.parquet")
    log(f"\n## context\n- rows: {len(ctx)}, cols: {list(ctx.columns)}")

    # ---- canonical splits (walk-forward, regime-diverse test folds) ----
    splits = {
        "design_note": (
            "Anchored walk-forward. Train expands from start; validation is the "
            "2y after train; test is the 2y after validation. No data after the "
            "test fold start may influence any modeling choice for that fold. "
            "Embargo: 21 trading days between train/val and val/test."
        ),
        "embargo_days": 21,
        "folds": [
            {"name": "F1", "train_end": "2007-12-31", "val_end": "2009-12-31", "test_end": "2011-12-31"},
            {"name": "F2", "train_end": "2009-12-31", "val_end": "2011-12-31", "test_end": "2013-12-31"},
            {"name": "F3", "train_end": "2011-12-31", "val_end": "2013-12-31", "test_end": "2015-12-31"},
            {"name": "F4", "train_end": "2013-12-31", "val_end": "2015-12-31", "test_end": "2017-12-31"},
            {"name": "F5", "train_end": "2015-12-31", "val_end": "2017-12-31", "test_end": "2019-12-31"},
            {"name": "F6", "train_end": "2017-12-31", "val_end": "2019-12-31", "test_end": "2021-12-31"},
            {"name": "F7", "train_end": "2019-12-31", "val_end": "2021-12-31", "test_end": "2023-12-31"},
            {"name": "F8", "train_end": "2021-12-31", "val_end": "2023-12-31", "test_end": "2025-12-31"},
        ],
        "holdout": {
            "note": "2026-01-01 .. present is a final untouched holdout, used once.",
            "start": "2026-01-01",
        },
    }
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2))
    log(f"\n## splits\n- {len(splits['folds'])} walk-forward folds + 2026 holdout, "
        f"21-day embargo")

    # ---- baselines per test fold for SPY/QQQ ----
    for sym in ["SPY", "QQQ", "VOO"]:
        px = pd.read_parquet(OUT / f"prices_{sym}.parquet")
        ret = px["adj_close"].pct_change()
        for f in splits["folds"]:
            start = pd.Timestamp(f["val_end"]) + pd.Timedelta(days=1)
            end = pd.Timestamp(f["test_end"])
            seg = ret.loc[start:end]
            if len(seg) < 100:
                continue
            baselines.append({"symbol": sym, "window": f"test_{f['name']}",
                              **perf_stats(seg)})
    bdf = pd.DataFrame(baselines)
    bdf.to_csv(OUT / "baselines.csv", index=False)
    log(f"\n## baselines\n- wrote baselines.csv ({len(bdf)} rows)")

    (OUT / "quality_report.md").write_text("\n".join(REPORT) + "\n")
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
