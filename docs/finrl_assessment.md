# Assessment: FinRL & FinRL-Trading as a foundation for this study

Cloned 2026-07-31 into `vendor/` (shallow, depth 50).

| | FinRL | FinRL-Trading ("FinRL-X") |
|---|---|---|
| License | MIT | Apache 2.0 |
| Last commit | 2026-07-12 (active) | 2026-05-02 (active) |
| Stack | Gymnasium + Stable-Baselines3 (also ElegantRL, RLlib) | ML selection + weight-centric pipeline + `bt` backtester + Alpaca live |
| Focus | DRL training environments (multi-stock, portfolio) | Full-stack stock-selection/rotation platform (paper arXiv:2603.21330) |

## What FinRL actually gives us

The heart is `finrl/meta/env_stock_trading/env_stocktrading.py` (~570 lines,
read in full) plus a thin `DRLAgent` wrapper over Stable-Baselines3 with
default hyperparameters in `finrl/config.py`.

Design of their env vs what our study needs:

| Aspect | FinRL `StockTradingEnv` | Our design requirement |
|---|---|---|
| Action | integer **shares** to buy/sell, scaled by `hmax` | target **exposure** in [0,1] |
| Reward | **dollar** PnL change × `reward_scaling=1e-4` (scale-dependent) | log return net of costs |
| Execution | trades at the **same close** the agent just observed | decision at close t, documented fill rule, next-open robustness check |
| Cash | earns **zero** interest | earns 3-month T-bill (^IRX) — critical, otherwise "go to cash" is unfairly penalized vs reality |
| Dividends | handled — `YahooDownloader._adjust_prices` back-adjusts OHLC by adj-close ratio | same (verified our pipeline independently) |
| Costs | flat pct per buy/sell — fine | same, plus sensitivity grid |
| Evaluation | none built in — examples train one 20k-step run, test on one window, no seeds/walk-forward | 8-fold walk-forward, ≥10 seeds, deflated Sharpe |
| Data plumbing | integer-indexed df with `tic` column; fragile | our parquet features |

Also inherited from FinRL if we adopt it: matplotlib/CSV side effects inside
`step()`, turbulence-based forced liquidation entangled with env logic, and a
state vector of raw prices/balances (unnormalized dollars) rather than
scale-free features.

## What FinRL-Trading (FinRL-X) actually is

It has pivoted (2026 rewrite) into a **multi-stock selection + rotation
platform**: ML/fundamental stock screening over S&P 500 constituents,
weight-centric strategy composition (selection → allocation → timing → risk
overlay), backtesting via the `bt` library with pluggable cost models
(default 10 bps flat), and Alpaca paper/live execution. README claims
Rolling-selection 5.98x vs QQQ 4.02x over 2018–2025 and +19.8% paper-trading
Oct 2025–Mar 2026; treat as unaudited marketing until reproduced — this is
exactly the class of claim our study is designed to stress-test.

Notably useful assets in the repo regardless:

- `data/sp500_historical_constituents.csv` — point-in-time index membership
  (survivorship-bias control; valuable if we ever extend beyond single-index).
- `src/strategies/tsmomsignal.py` — clean TSMOM (Moskowitz 2012) reference
  implementation, one of our planned baselines.
- `src/backtest/backtest_engine.py` — `bt`-based engine with cost-model API;
  usable as an **independent validator** for our env's trade accounting.
- Code quality caveat: research-grade, mixed Chinese/English comments,
  sklearn/xgboost glue; their "DRL" pieces still route through FinRL's env.

## Verdict

**Learn from both, build on neither as the core.** Their problem (multi-stock
selection/rotation) is not our problem (single liquid index product vs
buy-and-hold, with statistically honest evaluation). The four things our study
lives or dies on — exposure action space, T-bill cash, strict decision/fill
timing, and walk-forward multi-seed evaluation — are all absent from FinRL's
env, and retrofitting them means rewriting `step()` anyway. A purpose-built
env is ~200 lines and must pass a buy-and-hold-reproduction integration test
regardless of origin.

Concrete reuse plan:

1. **Adopt the stack pattern**: Gymnasium env + Stable-Baselines3 directly
   (FinRL validates this combo works; skip their wrapper).
2. **Seed hyperparameters** from `finrl/config.py` PPO/A2C/SAC defaults as the
   starting grid — they are community-tuned for daily-bar trading envs.
3. **Run FinRL's own env as a comparison arm**: train their `StockTradingEnv`
   on our SPY data with their defaults, evaluate on our folds. This directly
   tests "does the published framework beat buy-and-hold under honest
   evaluation?" — a headline result for the study.
4. **Cross-validate accounting** with FinRL-X's `bt` backtest engine: feed our
   agent's daily weights in, confirm PnL/cost agreement within tolerance.
5. **Borrow baselines/data**: TSMOM reference, S&P constituents file.

Licensing is clean for all of this (MIT/Apache-2.0).
