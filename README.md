# rlqqq — Can RL beat buy-and-hold on US index products?

Research project investigating whether a well-trained reinforcement learning
agent trading NASDAQ-100 / S&P 500 products (QQQ, SPY, VOO and the underlying
indices) can outperform buy-and-hold and dollar-cost-averaging baselines,
after realistic costs, with statistically honest evaluation.

## Interactive dashboard

The [RLQQQ policy replay](https://linjiw.github.io/rlqqq/) animates the
2010-2025 walk-forward decisions, compares normalized wealth with QQQ and
SPY, exposes the volatility base and learned residual tilt at every date, and
publishes the v8 no-calendar reference's latest delayed-close output. The
animated historical series remains the audited v4 replay and is labeled as
such. The dashboard also exports the replay as WebM video or a PNG frame.

Rebuild its checked-in data bundle with:

```bash
.venv/bin/python scripts/build_web_data.py
```

The live signal is generated in GitHub Actions and served as static JSON; no
market API key or model runtime is exposed in the browser. Rebuild the frozen
v8 actor bundle and checked snapshot with:

```bash
.venv/bin/python scripts/export_live_policy.py --workers 7 --force
.venv/bin/python scripts/update_live_signal.py \
  --provider checked \
  --generated-at 2026-08-01T12:00:00Z
```

See [docs/live_deployment.md](docs/live_deployment.md) for feature-parity
controls, scheduling behavior, provider tradeoffs, and the production upgrade
path.

## Layout

```
docs/
  literature_review.md   # synthesis of the six surveys below
  lit_review/            # detailed verified surveys: core positive results,
                         # skeptical/negative results, 2022-2026 advances,
                         # tooling/data/evaluation, transformers, diffusion/FM
  design_plan.md         # full experimental design plan (v1)
  investigation_plan_new_models.md  # transformer/diffusion/flow-matching arms
  finrl_assessment.md    # verdict on building atop FinRL / FinRL-Trading
data/
  raw/                   # downloaded CSVs + manifest.json (source of truth)
  processed/             # parquet prices/features, splits.json, baselines.csv,
                         # quality_report.md
src/rlqqq/
  data.py                # MarketData bundle + per-window Normalizer
  env.py                 # ExposureTradingEnv + shared accounting identity
  baselines.py           # B&H, DCA, MA200, vol-target, TSMOM via same accounting
  metrics.py             # CAGR/Sharpe/Sortino/MaxDD/Calmar/turnover
  stats.py               # paired block bootstrap, PSR/DSR, IQM, exposure decomp
  walkforward.py         # fold specs, PPO train/eval, trial registry
scripts/
  download_data.py       # fetch daily data (yfinance primary; Stooq/FRED blocked
                         # from this host as of 2026-07)
  prepare_dataset.py     # clean, audit, feature-build, splits, baselines
  run_pilot.py           # 8 folds x N seeds PPO experiment (parallel)
  analyze_pilot.py       # per-fold IQM vs baselines, bootstrap CIs, DSR
  export_live_policy.py  # reproducible SB3 -> NumPy frozen actor release
  update_live_signal.py  # delayed market refresh + static signal/log output
tests/
  test_env.py            # incl. buy-and-hold reproduction invariant
  test_live.py           # feature, actor, schema, and append-only log parity
results/
  registry.jsonl         # trial registry (every scored run, for DSR)
  series/                # per-run test exposure/return series
  forward_log.csv        # one immutable frozen-policy record per market date
models/live/             # deployable actor weights + release manifest
.venv/                   # python 3.12 venv (pandas, torch-cpu, sb3, arch)
```

## Reproduce

```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install pandas numpy pyarrow requests yfinance matplotlib
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare_dataset.py
```

## Data snapshot (2026-07-31)

| series | span | rows |
|---|---|---|
| SPY (ETF, div-adjusted) | 1993-01-29 .. 2026-07-31 | 8,433 |
| QQQ (ETF, div-adjusted) | 1999-03-10 .. 2026-07-31 | 6,891 |
| VOO (ETF, div-adjusted) | 2010-09-09 .. 2026-07-31 | 3,997 |
| ^GSPC (index) | 1927-12-30 .. 2026-07-31 | 24,762 |
| ^NDX (index) | 1985-10-01 .. 2026-07-31 | 10,287 |
| ^VIX | 1990-01-02 .. | 9,213 |
| ^TNX/^IRX/^FVX (yields) | 1960s .. | ~16k each |
| TLT, GLD (context ETFs) | 2002/2004 .. | ~6k each |

Buy-and-hold reference (full history, total return): SPY CAGR 10.8%,
Sharpe 0.65, MaxDD −55%; QQQ CAGR 10.7%, Sharpe 0.51, MaxDD −83%.
These are the numbers any RL agent has to beat. See
`data/processed/baselines.csv` for per-fold values.
