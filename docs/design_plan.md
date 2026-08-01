# Design Plan: RL Trading Agents vs Buy-and-Hold / DCA on US Index Products

Status: v1 — literature review complete (see `literature_review.md` and
`lit_review/`); new-model arms specified in `investigation_plan_new_models.md`.

## 1. Research question

> Can a well-trained RL agent, trading only liquid US index products
> (QQQ / SPY / VOO and their underlying indices), reliably outperform
> buy-and-hold and dollar-cost-averaging **after realistic transaction costs**,
> evaluated with statistically honest, walk-forward methodology?

Hypotheses:

- **H0** (null): after costs, no RL agent variant achieves a statistically
  significant improvement in risk-adjusted return (deflated Sharpe) over
  buy-and-hold across walk-forward test folds.
- **H1a** (weak positive): an RL agent matches buy-and-hold return with
  materially lower drawdown (a defensible win — better risk-adjusted).
- **H1b** (strong positive): an RL agent beats buy-and-hold on absolute CAGR
  after costs across regime-diverse folds.

We treat H1a as the realistic target and H1b as a stretch goal. A rigorous
confirmation of H0 is itself a publishable/useful result.

## 2. Literature grounding

Full synthesis in `literature_review.md`; six detailed reports in
`lit_review/`. Key facts shaping this design:

- The three studies closest to our exact question all found ~nothing:
  Théate & Ernst 2021 (TDQN Sharpe = B&H exactly on SPY/QQQ/DIA), Zhang,
  Zohren & Roberts 2020 (long-only beats all RL on the equity-index
  sub-portfolio), Kashif & Ślepaczuk 2026 (Nasdaq-100 2003–2026, 16 folds,
  costs, HAC: no significant excess return). H0 is the literature's answer;
  our contribution is either the first credible exception or a pre-registered
  confirmation.
- The replicated positive mechanism across 2022–2026 is **drawdown/exposure
  control in bear/high-vol regimes** (CPPO, regime conditioning) — hence H1a
  as primary target.
- Sullivan/Timmermann/White 1999, Welch & Goyal 2008, and Huang et al. 2020
  catalog why index-timing signals decay or reduce to vol-scaled drift —
  hence the TSMOM and vol-target baselines with return attribution.
- Backtest-overfitting machinery (deflated Sharpe, PBO/CSCV, SPA, trial
  registries, seed distributions) is mandatory, not optional (§7).
- Lu 2023: standard off-policy DRL fails on noisy rewards even in known-optimum
  simulators; PPO-class needs "~8,000 years of daily prices" — hence on-policy
  algorithms, synthetic-path training (WP-D/E), and a known-optimum simulator
  sanity check before any claim on real data.

## 3. Data (DONE — see `data/` and scripts)

- Source: yfinance daily bars, dividend/split-adjusted; `Adj Close` used as
  total-return series (verified: SPY 17x price-only vs 31x total-return since
  1993 — dividends matter and are included on both agent and baseline sides).
- Instruments: SPY (1993–), QQQ (1999–), VOO (2010–), ^GSPC (1927–),
  ^NDX (1985–). Context: ^VIX, ^TNX/^IRX/^FVX yields, TLT, GLD.
- Audited: no NaNs/OHLC violations in ETF series; crash days verified and kept.
- 21 causal features per instrument (momentum 1d–252d, realized vol, MA gaps,
  RSI, MACD, drawdown, range/gap, volume ratio, calendar).
- Canonical splits in `data/processed/splits.json`: 8 anchored walk-forward
  folds with 2-year validation and 2-year test windows covering 2008–2025,
  21-day embargo, plus an untouched 2026 holdout used exactly once.

## 4. Baselines (computed, `data/processed/baselines.csv`)

1. Buy-and-hold, total return, per fold and full history.
2. Monthly DCA into the same instrument.
3. 60/40 SPY/TLT monthly-rebalanced (risk-adjusted reference).
4. 200-day moving-average timing rule (the classic non-ML timing baseline —
   any RL agent must beat the dumb version of itself).
5. Volatility targeting (10% target, scaled exposure) — second non-ML baseline.

## 5. Environment design

- Gymnasium environment, daily decision frequency.
- **Action**: target exposure in {0, 0.5, 1.0} (discrete) and [0, 1]
  (continuous variant). No leverage, no shorting in the core study (keeps the
  comparison to buy-and-hold honest; a levered extension is out of scope).
- **Timing rule (causality)**: state at day *t* uses information through the
  close of *t*; the chosen exposure is applied from the close of *t* (fills at
  close *t*, alternative: open of *t+1* as robustness check).
- **Costs**: 5 bps round-trip base case (SPY/QQQ spreads are ~1 bp, so this is
  conservative); sensitivity grid {1, 5, 10, 25} bps. Cash earns the 3-month
  T-bill rate (^IRX) — an agent that goes to cash gets paid, as in reality.
- **Reward**: log portfolio return net of costs; variants with
  vol-penalized and drawdown-penalized shaping.

## 6. Agents

(Algorithm shortlist to be finalized from the recent-advances survey; planned
core grid: DQN, PPO, SAC-discrete via Stable-Baselines3 + one offline-RL or
sequence-model variant.)

Framework decision (see `finrl_assessment.md`): use Gymnasium +
Stable-Baselines3 directly with a purpose-built ~200-line env; FinRL's env
lacks exposure actions, T-bill cash, and strict decision/fill timing. FinRL's
tuned hyperparameters seed our grid; FinRL's own `StockTradingEnv` runs as a
comparison arm under our honest evaluation; FinRL-X's `bt` backtest engine
cross-validates our trade accounting.

## 7. Evaluation protocol

- Anchored walk-forward over the 8 canonical folds; hyperparameters tuned on
  validation windows only, frozen before touching each test window.
- ≥10 random seeds per (algorithm, fold); report interquartile mean and 95%
  stratified-bootstrap CIs (rliable-style), never best-seed.
- Deflated Sharpe ratio accounting for the full number of configurations
  tried; report the trial count honestly.
- Regime breakdown: 2008 crisis, 2011, 2015–16, 2018 Q4, 2020 COVID, 2022
  bear, 2023–25 bull — a strategy that only wins in one regime is a fit, not
  a strategy.
- Final holdout (2026 YTD) evaluated once, after all decisions are frozen.

## 8. Risks & mitigations

From `lit_review/skeptical_negative_results.md` (full checklist there):

| Risk | Mitigation |
|---|---|
| Backtest overfitting via many configs/checkpoints/seeds | Trial registry from day one; deflated Sharpe with N = registry size; PBO via CSCV (skfolio); Hansen SPA over all configs vs B&H (arch) |
| Seed lottery | ≥10 seeds per (config, fold); IQM + stratified bootstrap CIs (vendored rliable); never best-seed |
| Look-ahead via preprocessing | All normalization fit per training window; features audited for causality; pretrained-model corpus cutoffs documented per fold |
| Reward hacking / env bugs | Independent re-simulation of every policy's trades in vectorbt (`targetpercent`, next-open fills); agent-holds-1.0 must reproduce B&H exactly (integration test) |
| Single-regime evaluation | 8 regime-labeled folds 2008–2025 + untouched 2026 holdout; per-regime reporting mandatory |
| Long-bias masquerading as skill | Exposure decomposition: excess return after subtracting exposure × index return; TSMOM + vol-target + 200d-MA attribution regressions |
| Cost mis-specification | Grid {0.5, 2, 5, 10, 25} bps + break-even cost; idle cash earns T-bill; pre-tax headline with after-tax (STCG 40.8% vs LTCG 23.8%) overlay |
| Non-stationarity / one-history overfitting | Synthetic-path training arms (block bootstrap → generators gated by stylized-fact + signal-survival scorecard); stress-scenario deck for all agents |
| RL sample inefficiency (Lu 2023) | On-policy PPO-class primary; known-optimum simulator sanity check; data-budget statement in the report |
| Publication/interpretation bias | Hypotheses and pass criteria pre-registered in this doc before any test fold is touched; null result written up as a first-class outcome |

Pre-registered pass criterion for H1b (from tooling review): IQM excess Sharpe
vs B&H > 0 with 95% CI excluding 0, DSR > 0.95, SPA p < 0.05, at ≥2 bps costs,
on test folds only. H1a analog: equal-or-better CAGR CI with MaxDD improvement
CI excluding 0.

## 9. Milestones

- M0: data + lit review (this phase).
- M1: environment + baselines reproduced in env (agent that holds must
  exactly reproduce buy-and-hold numbers — integration test).
- M2: core agent grid on SPY.
- M3: transfer to QQQ/VOO, cost sensitivity, ablations.
- M4: statistical analysis + report.
