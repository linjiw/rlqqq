"""Download daily OHLCV data for the RL-vs-buy-and-hold study.

Primary source: Stooq (no API key, generous history, includes indices).
Secondary source: yfinance (dividend-adjusted ETF series + dividends/splits),
used when reachable; failures are logged and non-fatal.

Everything lands in data/raw/ as CSV, one file per (source, symbol).
A manifest with row counts, date ranges, and SHA256 hashes is written to
data/raw/manifest.json for reproducibility.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Stooq symbol -> our name. Stooq US tickers use .us suffix; indices use ^.
STOOQ_SYMBOLS = {
    "spy.us": "SPY",       # S&P 500 ETF, inception 1993-01
    "qqq.us": "QQQ",       # Nasdaq-100 ETF, inception 1999-03
    "voo.us": "VOO",       # Vanguard S&P 500 ETF, inception 2010-09
    "^spx": "GSPC",        # S&P 500 index (long history)
    "^ndx": "NDX",         # Nasdaq-100 index (1985+)
    "^ndq": "IXIC",        # Nasdaq Composite
    "tlt.us": "TLT",       # 20y treasury ETF (regime/context feature)
    "gld.us": "GLD",       # gold ETF (context feature)
}

YF_SYMBOLS = ["SPY", "QQQ", "VOO", "^GSPC", "^NDX", "^VIX", "TLT", "GLD"]

# FRED series (no key needed for fredgraph CSV endpoint)
FRED_SERIES = {
    "DGS10": "10y treasury constant maturity yield",
    "DGS2": "2y treasury constant maturity yield",
    "DFF": "effective federal funds rate",
    "T10Y2Y": "10y-2y term spread",
    "VIXCLS": "CBOE VIX close",
}

UA = {"User-Agent": "Mozilla/5.0 (research; rlqqq-study)"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_stooq(symbol: str, name: str) -> dict | None:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    if len(r.text) < 200 or "No data" in r.text:
        print(f"  stooq {symbol}: NO DATA", file=sys.stderr)
        return None
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    out = RAW / f"stooq_{name}.csv"
    df.to_csv(out, index=False)
    return {
        "source": "stooq",
        "symbol": symbol,
        "file": out.name,
        "rows": len(df),
        "start": str(df["Date"].min().date()),
        "end": str(df["Date"].max().date()),
        "sha256": sha256(out),
    }


def fetch_yfinance(symbol: str) -> list[dict]:
    import yfinance as yf

    t = yf.Ticker(symbol)
    entries = []
    # auto_adjust=False keeps raw Close + Adj Close + Dividends/Splits columns
    hist = t.history(period="max", auto_adjust=False, actions=True)
    if hist.empty:
        print(f"  yfinance {symbol}: EMPTY", file=sys.stderr)
        return entries
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    safe = symbol.replace("^", "IDX_")
    out = RAW / f"yf_{safe}.csv"
    hist.to_csv(out, index=False)
    entries.append({
        "source": "yfinance",
        "symbol": symbol,
        "file": out.name,
        "rows": len(hist),
        "start": str(hist["Date"].min().date()),
        "end": str(hist["Date"].max().date()),
        "sha256": sha256(out),
    })
    return entries


def fetch_fred(series: str, desc: str) -> dict | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or len(df.columns) < 2:
        print(f"  fred {series}: NO DATA", file=sys.stderr)
        return None
    df.columns = ["Date", series]
    df[series] = pd.to_numeric(df[series], errors="coerce")
    out = RAW / f"fred_{series}.csv"
    df.to_csv(out, index=False)
    return {
        "source": "fred",
        "symbol": series,
        "desc": desc,
        "file": out.name,
        "rows": len(df),
        "start": df["Date"].iloc[0],
        "end": df["Date"].iloc[-1],
        "sha256": sha256(out),
    }


def main() -> None:
    manifest: list[dict] = []

    print("== Stooq ==")
    for sym, name in STOOQ_SYMBOLS.items():
        try:
            e = fetch_stooq(sym, name)
            if e:
                manifest.append(e)
                print(f"  {name}: {e['rows']} rows {e['start']} .. {e['end']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  stooq {sym}: FAILED {exc}", file=sys.stderr)
        time.sleep(1.0)

    print("== FRED ==")
    for series, desc in FRED_SERIES.items():
        try:
            e = fetch_fred(series, desc)
            if e:
                manifest.append(e)
                print(f"  {series}: {e['rows']} rows {e['start']} .. {e['end']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  fred {series}: FAILED {exc}", file=sys.stderr)
        time.sleep(1.0)

    print("== yfinance ==")
    for sym in YF_SYMBOLS:
        try:
            for e in fetch_yfinance(sym):
                manifest.append(e)
                print(f"  {sym}: {e['rows']} rows {e['start']} .. {e['end']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  yfinance {sym}: FAILED {exc}", file=sys.stderr)
        time.sleep(1.5)

    meta = {
        "downloaded_on": str(date.today()),
        "entries": manifest,
    }
    (RAW / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {len(manifest)} datasets; manifest at {RAW / 'manifest.json'}")


if __name__ == "__main__":
    main()
