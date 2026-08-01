# Tooling, Data & Methodology Review

Research-agent report, verified 2026-07-31 by live fetches (GitHub API, PyPI,
official docs, regulator/broker pages, Internet Archive).

## 1. Data sources — final picks

- **yfinance ≥1.5.2** (active, v1.5.2 2026-07-23): primary for
  QQQ/SPY/VOO/^GSPC/^NDX. Trap: `auto_adjust` defaulted to True in v0.2.51 —
  always pass `auto_adjust=False` explicitly and keep Adj Close + dividend/
  split events. ToS: personal use, no redistribution. 2025 had real rate-limit
  breakage windows (issue #2422); pin recent versions.
- **Stooq**: CSV endpoints now behind a JS proof-of-work anti-bot wall
  (confirmed — matches our own failed download); pandas-datareader 0.11 removed
  its Stooq reader. Bulk DB dumps (captcha-gated) are the one-time cross-check
  route. ^SPX pre-1928 rows are synthetic monthly placeholders — filter.
- **Tiingo**: best adjustment data (raw + adj OHLCV + divCash/splitFactor,
  CRSP methodology) but free tier **prohibits persistent storage** — a cached
  research dataset violates ToS unless on the $30/mo plan. No index levels.
- **Alpha Vantage**: adjusted daily is premium-only now; free = 100 bars. Skip.
- **Polygon → Massive** (rebranded 2025-10): free tier 2y history. Skip.
- **FRED**: VIXCLS (citation required) OK; SP500 series is pre-approval + 10y
  window — not a ^GSPC source; DTB3/DGS3MO/DFF public domain = risk-free rate.
  Site 403s bot fetches (matches our experience) — use API/fredapi.
- **CBOE direct** (verified live, no key):
  `https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv` —
  VIX OHLC 1990-01-02→present. Best ^VIX source. Siblings: VVIX, VIX9D.
- **Ken French library**: daily Mkt-RF/SMB/HML/RF 1926-07→present, no auth,
  monthly refresh; RF = 1-mo T-bill. pandas-datareader `famafrench` works.
- **Nasdaq Data Link**: WIKI dead since 2018. Skip. **Databento**: cleanest
  licensing if scaling up. **Norgate**: survivorship-bias-free upgrade path.

Instrument first dates (verified): ^GSPC 1927-12-30, ^NDX 1985-10-01, ^VIX
1990-01-02, SPY 1993-01-29, QQQ 1999-03-10, VOO 2010-09-09.

## 2. RL frameworks

Finance-specific harnesses all fail the rigor bar: FinRL (305 open issues,
PyPI frozen 0.3.7, unpinned deps; look-ahead concern open since 2023 #938;
#1248: tutorial failed to beat B&H even with future returns injected into the
state); FinRL-Meta (idea catalog); TradeMaster (stale, ray 1.13/TF 2.11 pins;
PRUDEX-Compass eval toolkit worth reading); gym-anytrading (2-action, no cash);
Gym-Trading-Env (best design — position-fraction actions, fees, borrow
interest — but winding down, open interest bug #23; fork as template).

General libraries: **Stable-Baselines3 v2.9.0** (2026-06-15, Gymnasium ≤1.3,
RecurrentPPO/TQC/CrossQ in sb3-contrib v2.9.0) = recommended trainer. **SBX**
(SB3+JAX, v0.28.1 2026-07-24, ~20x faster) = seed-sweep throughput. CleanRL =
algorithm-surgery reference. TorchRL most active but assemble-yourself. RLlib/
Tianshou: overkill / mid-transition.

**Conclusion: custom Gymnasium env (~200–400 lines) — the env IS the
experiment. SB3 for training, SBX for sweeps, check_env-validated.**

## 3. Backtesting engines

- **vectorbt REVIVED**: v1.0.0 2026-04-22 (optional Rust engine), v1.1.0
  2026-07-05 (py3.11–3.14, numpy 2.4+, pandas 3). Fair-code Apache-2.0 +
  Commons Clause (fine for research). `Portfolio.from_orders(size_type=
  'targetpercent', price=open_, fees=…, slippage=…)` = exactly what an RL
  policy emits; whole fee×slippage grid in one vectorized call. Pin 0.28.5 if
  version floors are a problem.
- backtrader: abandoned (2023). zipline-reloaded: maintained but bundle
  friction disproportionate for 3 ETFs. NautilusTrader: very active, overkill.
- Metrics: **empyrical-reloaded** 0.5.12 (correctness) + quantstats 0.0.81
  (tearsheets; has open metric bugs #535/#537/#514 — cross-check numbers).

## 4. Evaluation methodology mechanics

- **Walk-forward**: anchored expanding primary + rolling-10y sensitivity
  (if conclusions flip, that's a finding). Train ≥8–10y → val 1y → purge/
  embargo ~63 trading days → test 1y, step 1y → ~14–16 folds from ~2010,
  concatenated into one OOS equity curve. Regime-label every fold. skfolio
  `WalkForward` supports both.
- **CPCV**: skfolio `CombinatorialPurgedCV` (v0.20.1, BSD-3, active) — the
  maintained implementation (timeseriescv dead; mlfinlab now GBP 100/mo).
  N=10,k=2 → 45 splits, 5 OOS paths → Sharpe distribution. RL caveats: use on
  frozen-policy returns as an annex, never headline.
- **DSR/PSR**: quantstats has PSR only; pypbo has PBO/PSR/DSR/MinTRL but AGPL.
  Write ~20 lines of own NumPy, unit-test against pypbo. DSR needs the trial
  registry (N = every config ever scored, and their cross-sectional variance).
- **Block bootstrap**: arch v8.0.0 (active) — `StationaryBootstrap`,
  `CircularBlockBootstrap`, `optimal_block_length`. **Paired resampling**:
  pass strategy and benchmark returns together so each replicate resamples the
  same time blocks; Δ-Sharpe per replicate, ≥10k reps.
- **Multi-seed**: **rliable archived 2025-10-15** — vendor the IQM/stratified-
  bootstrap code. ≥10 seeds/(config, fold); IQM; never best-seed.
- **SPA/Reality Check**: `arch.bootstrap.SPA` (+ StepM, MCS) — run all configs'
  loss series vs B&H for a snooping-corrected p-value.

## 5. Cost model (2025–2026, verified)

- Commissions: $0 retail (Fidelity/Robinhood/E*TRADE/IBKR Lite verified; IBKR
  Pro ≈ 0.05 bps on a $700 ETF).
- Spreads: SPY penny-wide ≈ **0.07 bps half-spread**; QQQ ~0.1–0.5 bps full;
  VOO ~0.5–1.5 bps full. MOC orders ≈ zero spread cost at small size.
- Regulatory: SEC §31 volatile — $0 from 2025-05-14, **$20.60/$1M (0.206 bps,
  sells only) from 2026-04-04**; FINRA TAF $0.000195/sh capped ≈ negligible.
- Literature's 5–30 bps is **1–2 orders of magnitude too high** for
  retail-size SPY/QQQ today; measurable costs ~0.1–0.5 bps/side.
- **Taxes: the dominant real-world cost.** STCG up to 40.8% (37% + 3.8% NIIT)
  vs LTCG 23.8% deferred; stylized drag ≈ **250–400 bps/yr** for annual STCG
  realization. Headline = pre-tax (IRA framing); after-tax overlay as
  sensitivity. Wash-sale complications for daily single-ticker trading.
- Expense ratios (SPY 0.0945%, QQQ 0.18% post-Dec-2025 conversion, VOO 0.03%)
  embedded in NAV — identical for B&H, exclude from cost model.
- **Idle cash earns T-bill/EFFR (~3.63% now)** — 0% cash modeling unfairly
  penalizes timing strategies.
- Recommended grid (one-way per unit turnover): optimistic 0.5 bps | **base
  2 bps (train here)** | conservative 5 bps | legacy-literature 10–25 bps for
  comparability. Report turnover + break-even cost.

## 6. Recommended protocol spec (pre-registration draft)

Outer: anchored walk-forward, train ≥8–10y → val 1y → 63-day embargo → test 1y
(~14–16 regime-labeled folds); rolling variant sensitivity. Inner: 10 seeds per
(config, fold), frozen test policies, full trial registry. Aggregation: IQM +
stratified bootstrap CIs (vendored rliable), full seed distributions.
Inference: paired StationaryBootstrap (10k reps) on Δ-Sharpe/Δ-CAGR/Δ-MaxDD vs
B&H and DCA; PSR + MinTRL; DSR from trial registry; Hansen SPA across all
configs. Annexes: CPCV(10,2) on frozen policies; cost grid + break-even;
per-regime table. **Pass criterion: IQM excess Sharpe vs B&H > 0 with 95% CI
excluding 0, DSR > 0.95, SPA p < 0.05, at ≥2 bps costs — test folds only.**
