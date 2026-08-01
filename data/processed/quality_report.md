# Data quality report


## SPY
- rows: 8433, span: 1993-01-29 .. 2026-07-31
- NaN OHLC cells: 0
- rows with non-positive prices: 0
- OHLC consistency violations: 0
- gaps >7 calendar days: 0
- days with |return|>15%: 0 (kept; verified crash days)
- zero-volume days: 0
- wrote prices_SPY.parquet (8433) and features_SPY.parquet (21 features)

## QQQ
- rows: 6891, span: 1999-03-10 .. 2026-07-31
- NaN OHLC cells: 0
- rows with non-positive prices: 0
- OHLC consistency violations: 0
- gaps >7 calendar days: 0
- days with |return|>15%: 1 (kept; verified crash days)
- zero-volume days: 0
- wrote prices_QQQ.parquet (6891) and features_QQQ.parquet (21 features)

## VOO
- rows: 3997, span: 2010-09-09 .. 2026-07-31
- NaN OHLC cells: 0
- rows with non-positive prices: 0
- OHLC consistency violations: 0
- gaps >7 calendar days: 0
- days with |return|>15%: 0 (kept; verified crash days)
- zero-volume days: 0
- wrote prices_VOO.parquet (3997) and features_VOO.parquet (21 features)

## GSPC
- rows: 24762, span: 1927-12-30 .. 2026-07-31
- NaN OHLC cells: 0
- rows with non-positive prices: 0
- OHLC consistency violations: 0
- gaps >7 calendar days: 1 (worst: 12d at 1933-03-15)
- days with |return|>15%: 2 (kept; verified crash days)
- zero-volume days: 5497
- wrote prices_GSPC.parquet (24762) and features_GSPC.parquet (21 features)

## NDX
- rows: 10287, span: 1985-10-01 .. 2026-07-31
- NaN OHLC cells: 0
- rows with non-positive prices: 0
- OHLC consistency violations: 0
- gaps >7 calendar days: 0
- days with |return|>15%: 2 (kept; verified crash days)
- zero-volume days: 0
- wrote prices_NDX.parquet (10287) and features_NDX.parquet (21 features)

## context
- rows: 16661, cols: ['vix', 'y10', 'y3m', 'y5', 'term_spread_10y_3m', 'vix_chg_5d']

## splits
- 8 walk-forward folds + 2026 holdout, 21-day embargo

## baselines
- wrote baselines.csv (29 rows)
