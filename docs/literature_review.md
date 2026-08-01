# Literature Review — Synthesis

Full reports (100+ verified primary sources, checked 2026-07-31) live in
`lit_review/`:

- `core_positive_rl_trading.md` — foundational + deep-RL trading papers
- `skeptical_negative_results.md` — EMH, backtest overfitting, replication failures
- `recent_advances_2022_2026.md` — modern RL, LLM agents, simulators, competitions
- `tooling_data_evaluation.md` — data sources, frameworks, evaluation mechanics, costs
- `transformers_for_trading.md` — TSFMs, supervised transformers, encoders in RL
- `diffusion_flow_matching.md` — generative models, synthetic data, stress scenarios

## The answer the literature gives to our research question

**No published study credibly shows an RL agent beating buy-and-hold on
SPY/QQQ/^GSPC/^NDX as a single instrument, after costs, with statistical
significance.** Three findings anchor this:

1. **The papers that tested exactly our question found ~nothing.**
   Théate & Ernst 2021 (TDQN): Sharpe on SPY/QQQ/DIA *identical* to
   buy-and-hold — the agent "learns to tend toward a passive trading
   strategy." Zhang, Zohren & Roberts 2020: across 50 futures RL wins, **except
   equity indices, where long-only beats every RL algorithm**. Kashif &
   Ślepaczuk 2026 (Nasdaq-100, 2003–2026, 16 walk-forward folds, costs, HAC
   inference): no statistically significant excess return.
2. **The positive literature's wins come from elsewhere**: multi-asset
   allocation into safe assets (Moody & Saffell 1970–94), hand-crafted
   risk-off overlays (Yang et al. turbulence index), crypto/FX regime
   artifacts, short bull-market windows, and missing costs. Claim strength is
   inversely correlated with methodological quality — the audit literature
   (FINSABER KDD'26, StockBench, Alpha Illusion) demolishes the LLM-agent
   headline numbers via pretraining contamination and absent costs.
3. **The self-deception machinery is fully catalogued** (deflated Sharpe, PBO,
   SPA/Reality Check, seed variance, look-ahead via normalization or
   pretrained components, single-path evaluation). A credible positive result
   must clear all of it; the checklist is in
   `skeptical_negative_results.md` §Synthesis and is baked into
   `design_plan.md` §7–8.

## What is genuinely promising (and adopted into our design)

- **Risk-adjusted reframing**: the replicated positive mechanism is drawdown/
  exposure control in bear/high-vol regimes (CPPO, regime conditioning) — so
  H1a (match B&H return at materially lower drawdown, beat DCA risk-adjusted)
  is the realistic target, not raw CAGR (H1b).
- **Synthetic-path training** (block bootstrap first, generative second) as
  the answer to n=1-history overfitting; generative **stress-scenario decks**
  for evaluation regardless.
- **Volatility forecasting features** (the one TSFM task with statistically
  defensible wins) and **finance-pretrained embeddings** (Kronos) as feature
  arms — plan in `investigation_plan_new_models.md`.
- **Modern evaluation infrastructure**: SB3 2.9 + custom Gymnasium env,
  vectorbt 1.1 as independent accountant, arch (paired stationary bootstrap,
  SPA), skfolio (walk-forward, CPCV), vendored rliable IQM code.
- **Honest costs**: retail costs on SPY/QQQ today are ~0.1–0.5 bps/side —
  literature's 5–30 bps is 1–2 orders too high — but **taxes (up to ~250–400
  bps/yr drag for short-term trading) are the real-world killer**; headline
  results pre-tax (IRA framing), after-tax overlay as sensitivity.

## Bottom line

The user's hypothesis ("RL and finance ML improved a lot — we should have a
chance") is **not supported for raw outperformance** by 2022–2026 evidence:
what improved is training speed, tail-risk control, and evaluation honesty,
none of which addresses the binding constraint (low signal, non-stationarity,
~250 decisions/year on the world's most efficient instruments). The project
remains valuable on three counts: (a) a rigorous, pre-registered test that
either finds the first credible edge or produces a publishable confirmation of
the null; (b) the realistic H1a win (risk-adjusted) is achievable and useful;
(c) several genuine open niches (SynthER-on-index-trading, cross-asset
pretrain → SPY fine-tune, signal-survival scoring of generators) where we can
contribute new results either way.
