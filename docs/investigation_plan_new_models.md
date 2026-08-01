# Investigation Plan: New AI Models for Feature Engineering & RL in Index Trading

Scope: can transformer-based models, diffusion models, and flow matching
improve our core study (RL agents on SPY/QQQ/VOO daily bars vs buy-and-hold /
DCA)? Grounded in two verified surveys:
`lit_review/transformers_for_trading.md` and
`lit_review/diffusion_flow_matching.md` (40+ and 35+ primary sources
respectively, checked 2026-07-31).

## 0. Headline conclusions from the surveys

1. **Zero-shot time-series foundation models do NOT forecast daily returns.**
   The best evidence (Rahimikia et al. 2511.18578; Cheung 2607.12248 on
   NASDAQ-100/S&P specifically) is negative: no directional skill over the
   base rate. What survives scrutiny: **volatility forecasting** (fine-tuned
   TimesFM beats econometric benchmarks with proper DM/GW tests, 2505.11163),
   **cross-series context** within a related group (Chronos-2 multivariate,
   2605.21504), and **finance-native pretraining** (Kronos +93% RankIC,
   2508.02739).
2. **On noisy daily returns, linear baselines ≈ transformers** (Zeng AAAI'23
   oral; Toner & Darlow 2403.14587). Any transformer gain must be demonstrated
   against ridge/DLinear, not against nothing.
3. **Diffusion/flow value is in DATA, not policies.** Our action space is 1-D
   exposure — unimodal — so diffusion/flow policies buy nothing (that entire
   literature exists for multimodal high-dim actions). But synthetic-path
   training demonstrably fixes variance/overfitting (Deeper Hedging
   2310.18755: train-on-synthetic → win-on-real; SynthER NeurIPS'23; TSDiff;
   Gao-Zha-Zhou 2509.08731), and generative stress scenarios (Tail-GAN,
   GuidedDiffTime 2307.01717) upgrade evaluation regardless.
4. **A bad generator actively hurts** (Kwon & Lee 2410.09850) — every
   generator must pass a stylized-fact + signal-survival scorecard before its
   samples touch training.
5. **Honest framing:** synthetic training reliably buys robustness; there is
   no published proof it manufactures alpha in directional daily index
   trading. Clean SynthER-on-index-trading and "pretrain cross-asset →
   fine-tune on SPY" are open niches this project can fill.

## 1. Work packages

Ordered by evidence-per-unit-compute. WP-A/B/C are feature-engineering arms
(transformers); WP-D/E/F are data arms (diffusion/FM). Each has a kill
criterion so we don't ride a dead horse.

### WP-A: Foundation-model embeddings as state features (cheap, do first)
- **A1 — Chronos-Bolt-small (48M, Apache-2.0):** `ChronosBoltPipeline.embed()`
  (API verified in source) over trailing 512-day windows of adjusted close and
  realized-vol series → mean-pool → PCA to 32 dims → append to the classical
  21-feature state. Precomputed offline: one CPU batch job (~minutes–1h for
  ~8k days × 3 ETFs), zero cost during RL training.
- **A2 — Kronos-small (24.7M, MIT):** the only open finance-pretrained model
  (12B K-lines, 45 exchanges). Tokenize our daily OHLCV, expose hidden states
  (~30 lines), same PCA treatment. Hypothesis: domain pretraining > generic
  pretraining (the literature's clearest positive signal).
- **A3 — MOMENT-1 (MIT, `task_name='embedding'`)** as a third embedding source
  if A1/A2 disagree.
- **Evaluation:** identical PPO config ± embedding features, 10 seeds × 8
  walk-forward folds; also a *probe test* — ridge regression from embeddings
  to next-5d return/vol vs ridge from classical features (if embeddings add
  no linear signal, they won't add RL signal).
- **Kill criterion:** no IQM validation-Sharpe improvement on ≥2 folds → drop.

### WP-B: TSFM volatility forecasts as features (the evidenced use)
- Chronos-2 (120M) zero-shot quantile forecasts of 5d/21d realized vol,
  univariate and multivariate over {SPY, QQQ, VOO, ^VIX}; append forecast
  quantiles (p10/p50/p90) to state. Baseline to beat: HAR-RV (the classic
  vol model, ~5 lines) — if Chronos-2 doesn't beat HAR-RV on QLIKE/DM test,
  use HAR-RV features instead and close the WP.
- Rationale: vol is forecastable and drives position sizing; this is where
  fine-tuned TSFMs have statistically defensible wins.

### WP-C: Learned sequence encoder in the RL loop (expensive arm, gated)
- Small PatchTST or iTransformer encoder (~1–5M params) over the trailing
  63-day feature window replacing the MLP trunk in PPO; optional TS2Vec
  contrastive pretraining on all pre-fold data (MIT, CPU-trainable);
  StockFormer-style auxiliary short-horizon prediction loss as a variant.
- Also test the Zohren-benchmark winner shapes: VSN+LSTM and xLSTM heads
  (2603.01820: best Sharpe / best cost buffer on daily futures).
- **Gate:** only run if WP-A or WP-B shows any positive signal, or as the
  single "modern architecture" ablation for the paper. Pre-registered
  expectation: null (Kashif & Ślepaczuk 2605.17307). Modest GPU, hours/run.
- **Skip explicitly:** LLM-backbone forecasters (NeurIPS'24 ablation),
  Moirai (NC license), TimeGPT (closed), diffusion/flow policies (1-D action),
  Decision Transformer arm (4 years of near-empty results on market data).

### WP-D: Block-bootstrap synthetic training (the non-neural baseline, week 1)
- Circular/stationary block bootstrap (arch v8, `optimal_block_length`) of SPY
  log-returns → thousands of synthetic training paths; PPO trained across
  resampled paths (Karzanov 2502.02619 template), tested on real held-out
  folds. Mix ratios {0, 25, 50, 75}% synthetic.
- This is the bar every neural generator must clear. Negligible compute.
- **Deliverable either way:** "does path-diversity training reduce seed/fold
  variance and improve OOS Sharpe on real data?"

### WP-E: Neural generators (diffusion + one non-diffusion), gated on WP-D
- **E1 — Stylized-fact + signal-survival scorecard (prerequisite):** tail
  index, ACF of |r| and r², leverage effect, variance-ratio profile, AND our
  own addition — do momentum/MA-crossover signals retain their historical
  IC on generated paths? (The literature gap: nobody checks whether the
  conditional structure an agent must learn survives generation.)
- **E2 — Diffusion-TS (MIT, single GPU)** on 252-day return windows;
  **E3 — SynthER (MIT)** on transition tuples (state, action, reward, next
  state) — the first published SynthER-on-index-trading result either way;
  **E4 — Path Shadowing MC / scattering spectra (non-neural, CPU, S&P-native,
  best conditional-structure evidence)** as the strongest challenger.
- Train agents on real+synthetic mixes; same evaluation as WP-D.
- **Kill criterion:** any generator failing E1 never reaches training; if no
  generator beats block bootstrap on OOS metrics, report that (valuable
  negative result) and keep bootstrap.
- Flow matching (TSFlow/FlowTS-style rectified flow) enters only as a
  **sampling-speed upgrade** if E2 wins and generation volume becomes the
  bottleneck; both reference repos are unlicensed — reimplement, don't vendor.

### WP-F: Generative stress-scenario evaluation (always runs; independent value)
- Build a stress deck for the final evaluation harness:
  (i) resampled/amplified historical crisis windows (2008, 2020, 2022);
  (ii) conditional generation of crash/vol-spike regimes — GuidedDiffTime-style
  constraint-guided sampling (condition on drawdown/vol at sampling time, no
  retraining) using whichever generator passed E1, or CoFinDiff-style
  trend/vol conditioning;
  (iii) Tail-GAN-inspired metric: compare the *P&L distribution tails* of
  agent vs buy-and-hold across the deck.
- Every trained agent (including classical-feature ones from the core study)
  gets scored on this deck. This WP pays off even if every other WP nulls.

## 2. Sequencing & compute budget

| Phase | WPs | Compute | Calendar |
|---|---|---|---|
| 1 | D (bootstrap) + A1 probe + B (vol forecasts) | CPU only | week 1–2 |
| 2 | A1/A2/A3 full RL ablations + E1 scorecard | CPU + small GPU bursts | week 2–4 |
| 3 | E2/E3/E4 generators + mix-ratio training | 1 modest GPU, hours/run | week 4–7 |
| 4 | C (encoder arm) if gated in; F stress deck final | GPU hours | week 7–9 |

All feature-engineering arms are offline batch jobs feeding the same frozen
RL pipeline, so they parallelize with the core study and never touch the
walk-forward test folds until final evaluation.

## 3. Methodological guardrails (inherit from skeptical checklist)

- Every pretrained model's **training-data cutoff must precede each test
  fold** it feeds — Chronos/TimesFM/Kronos corpus vintages get documented per
  fold; where a cutoff is unknown, that model's features are excluded from
  earlier test folds (look-ahead via pretraining is the Alpha-Illusion trap).
- Embeddings are features, so PCA/normalization fit on training windows only.
- Each WP counts toward the **trial registry** for the Deflated Sharpe
  computation — more arms = a higher significance bar; that's the price.
- Generators train only on data before the fold boundary (one generator per
  fold, or anchored-training generator reused forward).
- Unlicensed reference code (StockFormer, finrl-dt, TSFlow, FlowTS) is read
  for ideas, never vendored.

## 4. Expected outcomes (pre-registered priors)

- WP-B (vol features): most likely positive — modest Sharpe/drawdown gain via
  better sizing. WP-D: likely reduces variance across seeds/folds; unclear if
  it lifts mean. WP-A: coin flip; A2 (Kronos) > A1 (generic) if anything.
  WP-E: expected to roughly match, not beat, bootstrap. WP-C: expected null.
  WP-F: guaranteed methodological value.
- Publishable regardless of direction: "which of the 2023–2026 model advances
  actually move the needle on honest, cost-inclusive daily index trading" is
  the contribution.
