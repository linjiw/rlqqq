# rlqqq — Can RL beat buy-and-hold on US index products?

Research project investigating whether a well-trained reinforcement learning
agent trading NASDAQ-100 / S&P 500 products (QQQ, SPY, VOO and the underlying
indices) can outperform buy-and-hold and dollar-cost-averaging baselines,
after realistic costs, with statistically honest evaluation.

## Interactive dashboard

The [RLQQQ policy replay](https://linjiw.github.io/rlqqq/) publishes only the
selected v8 no-calendar core model as its current delayed-close target. A
checked-in benchmark compares v8 and v4 under identical accounting and keeps
the old v4/composite animation clearly separated as an archived research
replay. The dashboard also exports that archive as WebM video or a PNG frame.

Rebuild its checked-in data bundle with:

```bash
.venv/bin/python scripts/build_web_data.py
```

The latest policy panel runs the official frozen v8 ensemble in the browser.
Its source is the checked-in NPZ release; `scripts/export_browser_policy.py`
converts all ten actors into one hash-versioned, float64 ONNX model. The page
uses the self-hosted ONNX Runtime Web 1.27 WASM runtime with one thread and
replays the complete state path before it reveals the latest exposure.

GitHub Actions still performs the scheduled delayed-market refresh. It writes
static raw policy inputs, an independent Python reference, and the latest
signal metadata, so the browser calls no market-data API and receives no API
key. Verify the checked release and run the same end-to-end browser check
locally with:

```bash
.venv/bin/python scripts/update_live_signal.py \
  --provider checked \
  --no-log
.venv/bin/python scripts/export_browser_policy.py --check
node scripts/verify_browser_inference.mjs
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_live.py tests/test_browser_contract.py
```

Any model hash, version, feature schema, replay date, golden-vector, Python
parity, or freshness failure withholds the browser result instead of falling
back to the precomputed answer. An `asOf` close produces an unscored target
for the next close-to-close session; it is not part of performance through
that same close.

Reproduce the deployment comparison after generating both configurations'
eight-fold, ten-seed walk-forward series with:

```bash
.venv/bin/python scripts/evaluate_model_benchmark.py
```

The evaluator reports the trained cores, post-hoc leverage overlays, simple
VT10/VT20 rules, QQQ, frozen 2026 replay, and a conservative one-close-lag
sensitivity. Its selection rule limits deployment eligibility to trained core
models; v8 wins the near-tie because it removes two questionable calendar
features, trades less, and is also ahead under the historical lagged timing
sensitivity. The short 2026 lag sensitivity reverses the core ranking, so the
composite overlays and 2026 record are never treated as selection evidence.

See [docs/live_deployment.md](docs/live_deployment.md) for the release
contract, scheduled generation, fail-closed parity checks, and signal timing.

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
  export_browser_policy.py  # frozen NPZ -> hash-versioned float64 ONNX bundle
  update_live_signal.py  # delayed refresh + static input/reference/signal data
  verify_browser_inference.mjs  # local ORT-Web replay and parity check
tests/
  test_env.py            # incl. buy-and-hold reproduction invariant
  test_live.py           # feature, actor, schema, and append-only log parity
  test_browser_contract.py  # immutable ONNX/runtime/replay release contract
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
.venv/bin/pip install --requirement requirements-research.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare_dataset.py
```

The critical research stack is pinned for future reruns. Historical artifacts
created before this lock remain recipe-level results: hardware and the earlier
unrecorded dependency state can still prevent byte-identical PPO weights.

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
