# Live Policy Deployment

## Architecture

The public dashboard does not run the model or call a market-data API in the
browser:

1. The current ten-seed v8 no-calendar actor bundle and audited legacy v4
   bundle live in `models/live/`.
2. GitHub Actions fetches delayed end-of-day QQQ, VIX, 10-year, and 3-month
   Treasury series.
3. `scripts/update_live_signal.py` reproduces the full provider feature
   frame, selects the 22 features named by the v8 artifact, replays each
   actor from the 2026 reset, and writes `docs/assets/live-signal.json`.
4. The same run appends one immutable row per market date to
   `results/forward_log.csv`.
5. GitHub Pages serves static HTML, JavaScript, and generated JSON.

This boundary keeps API credentials and model execution out of the client.
The generated payload includes source date, generation time, model version,
training cutoff, artifact checksum, and stale status.

## Frozen actor

`scripts/export_live_policy.py` is a release-time command. It retrains the
calendar-free `ppo_v8_nocal` recipe through 2023-12-31 and exports only
deterministic actor weights, model metadata, and normalization statistics.
It refuses the export unless:

- pure-NumPy inference matches each in-memory SB3 actor on the same dates;
- online feature construction exactly matches the checked training snapshot.

The daily job needs NumPy and pandas, not PyTorch, Gymnasium, or
Stable-Baselines3. The v8 actor observation is 22 normalized market features
plus current exposure and the causal VT10 anchor. The unchanged v4 artifact
retains 24 features; its NumPy output is still checked against all ten saved
SB3 holdout series.

The v8 reference was selected from the 2010-2025 calendar ablation after the
2026 holdout had already been opened. Its 2026 comparison is therefore an
implementation-parity check, not a new untouched model test. Prospective v8
evidence begins only after this release.

## Data source

The initial public feed is Yahoo Finance through
[`yfinance`](https://ranaroussi.github.io/yfinance/). It is suitable
for this delayed research demonstration because it requires no client-side
secret and matches the historical source. `yfinance` states that it is an
unaffiliated open-source tool for research and education and that Yahoo data
is intended for personal use. The dashboard therefore labels the feed
delayed and research-grade, never real-time or execution-grade.

For a production service, replace `fetch_yahoo_market_frames()` with a
credentialed provider adapter:

- [Alpaca](https://docs.alpaca.markets/us/docs/historical-stock-data-1)
  supports split, dividend, spin-off, or all-adjusted stock bars. Its free IEX
  feed represents a single exchange and roughly 2.5% of volume; the paid SIP
  feed covers all US exchanges.
- [Twelve Data](https://twelvedata.com/docs) requires an API key for full
  access. Any key belongs in GitHub Actions secrets and must never be
  serialized into frontend JavaScript.

Feature formulas, adjustment choices, and session calendars must still pass
the snapshot parity test after a provider change.

## Schedule and failure behavior

`.github/workflows/pages.yml` refreshes at 22:37 UTC Monday through Friday,
away from GitHub's busiest top-of-hour period. [GitHub documents scheduled
Actions](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
as best-effort: runs can be delayed or dropped under load, execute only on
the default branch, and are disabled in public repositories after 60 days
without repository activity.

The frontend therefore derives freshness again at view time. A missing,
invalid, or old payload produces an unavailable/stale state while the
historical replay remains usable. The workflow fails before deployment when
market data, feature construction, or actor inference is incomplete.
Previously logged per-seed exposure state is checksummed; a provider revision
that changes the published policy path also fails closed instead of silently
rewriting history.

## Local operation

Rebuild the release artifact:

```bash
.venv/bin/python scripts/export_live_policy.py --workers 7 --force
```

Regenerate from the checked reproducibility snapshot:

```bash
.venv/bin/python scripts/update_live_signal.py \
  --provider checked \
  --generated-at 2026-08-01T12:00:00Z
```

Refresh from Yahoo:

```bash
.venv/bin/python scripts/update_live_signal.py
```

The primary displayed stance is the frozen v8 no-calendar ensemble. The
historical animation remains explicitly labeled as the audited v4 replay.
The higher-return tilt-on-VT20 composite is shown separately and remains
labeled as a post-hoc candidate. None of these outputs is personalized
investment advice or an execution order.

## Legacy v4 decision-date correction

The original July 31 prose snapshot reported a 0.35x ensemble output. That was
the saved July 30 action paired with July 31 market statistics: scored
evaluation frames omit the last feature row because no next-day return exists.
The deployed inference path does not need a realized future return, so it
correctly evaluates the final close. On the checked July 31 features the
frozen actors produce 0.40x (0.21x-0.63x), while retaining exact parity with
all saved actions through July 30.
