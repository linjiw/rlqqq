# Live Policy Deployment

## Boundary

The public page runs only the selected frozen v10 macro (leveraged) **core**
policy in the browser. It does not expose the post-hoc composite as a live target and
does not call a market-data API. The two release paths are
deliberately separate:

1. `models/live/ppo_v10_macro_frozen_2023_v1.npz` is the source release. It
   contains the shared training normalizer and all ten deterministic actors.
   Its `2023-12-31` cutoff is a decision-date label: the last training feature
   row is `2023-12-29`, and that row's reward realizes on `2024-01-02`.
2. `scripts/export_browser_policy.py` converts that NPZ, without retraining,
   into one float64 ONNX graph with ten actors. The filename contains the
   first 12 characters of its SHA-256; the manifest records the complete ONNX
   and source-NPZ hashes, feature contract, training cutoff, and runtime.
3. GitHub Actions generates delayed, static market inputs and an independent
   Python replay. GitHub Pages serves those files with the model and runtime.
4. The browser verifies the release, runs ONNX, and reveals the current
   exposure only after its own replay agrees with the Python reference.

The static input contains 28 raw market features per date (22 core plus six
cross-asset macro features from SPX, TLT, GLD, VIX, and the yield curve). The
browser normalizes them with the manifest statistics, then builds an ONNX
input of shape `[10, 30]`: 28 normalized values, each actor's own previous
exposure, and the common current volatility anchor
(`min(0.20 / realized_vol_21, 1.0)`). The previous-exposure state is
independent for every actor. The output is `[10, 5]` logits for residual
multipliers `0.5`, `0.75`, `1.0`, `1.25`, and `1.5`; the resulting core
exposure is capped at `1.5`, with exposure above `1.0` financed at
T-bill + 50 bps in all benchmark accounting.

Because previous exposure is an observation, latest-row inference is not
equivalent to the deployed policy. The browser starts all ten actor states at
zero on the manifest activation date (`2026-01-02`) and sequentially replays
the complete published input history through the latest close.

## Browser runtime and fail-closed checks

The site vendors ONNX Runtime Web 1.27 and its WASM files under
`docs/assets/ort/1.27.0/`. It uses the WASM execution provider, float64 tensors,
and exactly one thread; no CDN runtime, quantized model, web worker, or
cross-origin isolation is required.

Before showing an exposure, `docs/assets/browser-inference.mjs` verifies:

- the runtime version, manifest schema, `[10, 24]` contract, model SHA-256,
  golden-vector SHA-256, and static-input SHA-256;
- matching model/source hashes and feature schema across the artifacts, the
  manifest activation date in the raw replay, and one `asOf` date shared by
  the raw replay, Python reference, and live metadata;
- sorted replay dates, freshness, finite feature rows, and the exact 22-feature
  order;
- representative golden logits and actions, then every replay row against the
  Python normalizer, logits, actions, actor exposures, latest aggregates, and
  final actor-state hash.

Actions must match exactly. Numeric tolerances are recorded in the manifest
(`1e-6` for normalization/exposure and `1e-5` for logits). Any missing, stale,
misdated, differently hashed, or out-of-tolerance artifact leaves the current
exposure unavailable; the page never substitutes the Python-computed answer.
The audited historical v4 animation remains usable independently.

## Scheduled static generation

The weekday Pages workflow runs at 22:37 UTC, after the US close and away from
the busiest top-of-hour period. `scripts/update_live_signal.py` fetches delayed
QQQ, VIX, 10-year, and 3-month Treasury data, constructs the 28 raw features,
and writes:

- `docs/assets/live-signal.json`, the latest metadata and Python summary;
- `docs/assets/data/policy-input-history.json`, the browser's raw replay path;
- `docs/assets/data/python-reference.json`, independent normalized features,
  logits, actions, exposures, and final state;
- one immutable decision row in `results/forward_log.csv`.

The workflow validates the frozen browser bundle and executes the same
ORT-Web replay before deployment. A revision that changes an already logged
actor-state hash, or any data/model/parity failure, stops the refresh. The
browser also rechecks freshness when the page is opened because scheduled
GitHub Actions are best-effort.

Yahoo Finance through `yfinance` is a delayed, research-grade source. A
credentialed replacement belongs in the scheduled provider adapter and CI
secrets, never in browser assets; its adjustment and session conventions must
pass the same full-replay parity checks.

## Local release and validation

Install the inference and browser-export dependencies:

```bash
.venv/bin/pip install --requirement requirements-browser.txt
```

Regenerate the ONNX release only when the frozen NPZ changes:

```bash
.venv/bin/python scripts/export_browser_policy.py
```

For the checked snapshot, rebuild the static replay without touching the
forward log, verify byte-for-byte release reproducibility, run the actual
self-hosted browser runtime, and run the Python tests:

```bash
.venv/bin/python scripts/update_live_signal.py \
  --provider checked \
  --no-log
.venv/bin/python scripts/export_browser_policy.py --check
node scripts/verify_browser_inference.mjs
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_live.py tests/test_browser_contract.py
```

Use `.venv/bin/python scripts/update_live_signal.py` for a live Yahoo refresh.
That command writes static browser inputs and the Python reference as well as
the summary JSON.

## Signal timing and research status

An `asOf: T` signal consumes only data available in the completed close at
`T` and is labeled for the next close-to-close session. It has no subsequent
return yet, so performance reported through `T` ends with the prior decision;
the `T` target must not be counted until another close exists. It is a delayed
research target, not an intraday price or an order executable at the already
observed close.

The primary and only deployed stance is the frozen v8 core ensemble. The
historical animation is an archived v4 replay; its tilt-on-VT20 composite is a
post-hoc leverage overlay, not another model or a deployment candidate.

`scripts/evaluate_model_benchmark.py` provides the checked deployment gate.
It evaluates 8 folds x 10 seeds per version with the same cash, financing, and
2 bp one-way cost accounting, then reports frozen 2026 performance and a
conservative one-close-lag sensitivity. The full rerun makes v4 and v8 core a
near-tie on the optimistic research timing (Sharpe 1.021 vs 1.026); v8 has
two fewer questionable calendar features, lower turnover, the marginally
better Calmar, a stronger historical lag sensitivity, and a stronger
same-close frozen-2026 comparison. The predeclared simplification rule
therefore selects v8. The difference is not proof of statistical superiority,
and simple VT10 remains a demanding benchmark.

Those two supporting comparisons use different qualifiers: v8 leads v4 on
the **historical** one-close-lag sensitivity and on the **same-close** frozen
2026 replay. The short 2026 lag sensitivity reverses the core ranking (v4
Sharpe 0.820 versus v8 0.603), so it reinforces the research-only boundary
rather than serving as promotion evidence.

The v8 reference was selected after the 2026 holdout had been opened, so its
2026 comparison is not a new untouched test. Retraining can also differ from
the checked results because the historical training stack was not pinned.
None of these outputs is personalized investment advice or an execution
order.


## Why the browser does not fetch market data itself

Investigated 2026-08-04 for a fully serverless "browser fetches the latest
bars" design. Verified empirically (real browser, deployed origin): none of
the free daily-bar feeds — Yahoo chart API, Cboe delayed-quote CDN, Stooq
CSV — send `Access-Control-Allow-Origin`, so every in-page `fetch` is blocked
by CORS before any data arrives. The only workarounds are a proxy we would
have to run (a server) or an untrusted third-party CORS proxy (a silent
data-integrity hole in a fail-closed system). Both rejected.

The serverless answer is the scheduled GitHub Actions refresh, hardened:

1. `validate_latest_market_frames` aligns all feeds to the **latest common
   completed session** instead of failing when feeds publish on different
   clocks (the actual cause of the 2026-08-03 stale page: an early VIX print
   for the next session aborted the whole refresh).
2. The refresh runs three times per trading day (22:37 UTC, 01:07 UTC,
   10:07 UTC next morning) — early runs are safe (they publish the common
   session) and later runs advance it once laggard feeds finalize.
3. The page shows explicit freshness copy ("auto-refreshes three times each
   trading day" / "refresh overdue") so staleness is visible, never silent.
