# Diffusion Models & Flow Matching for RL Index Trading

Research-agent report, sources verified 2026-07-31 (arXiv, GitHub, publisher
pages). Raw material for `docs/investigation_plan_new_models.md`.

## 1. Generative models for financial time series

### General-purpose TS diffusion
- **TimeGrad** (ICML 2021, 2101.12072) — AR denoising diffusion; promising but
  code stale (`pytorch-ts`). **ScoreGrad** (2106.10121) skip. **CSDI** (NeurIPS
  2021, 2107.03502, MIT) imputation-focused, skip. **SSSD** (TMLR, 2208.09399)
  skip at our scale.
- **Diffusion-TS** (ICLR 2024, 2403.01742) — transformer diffusion w/
  trend/seasonality decomposition + Fourier loss. **Usable-now**: MIT, includes
  a Stocks dataset, single-GPU, maintained (classifier guidance Feb 2025).
- **TimeDiT** (2409.02322) foundation-scale, overkill. **TSDiff** (NeurIPS 2023,
  2307.11494) — unconditional diffusion + self-guidance; **synthetic data
  sometimes trains better downstream forecasters than real data** — key
  evidence for synthetic training.

### Finance-specific diffusion
- **Takahashi & Mizuno** (2410.18897): DDPM on wavelet-transformed price/volume;
  explicitly evaluates fat tails, vol clustering, seasonality — all satisfied.
  **Usable-now as design reference** + stylized-fact scoring template.
- **CoFinDiff** (2503.04164): conditional diffusion (cross-attn on trend/vol
  conditions); reproduces stylized facts, improves downstream deep hedging.
  **Usable-now conceptually** — regime-conditioned training data.
- FTS-Diffusion (ICLR 2024): ~"100 years" synthetic significantly improves LSTM
  next-day prediction; robust at 70% synthetic mix; **code NOT released**.
- DiffsFormer (2402.06656): diffusion factor augmentation, +7–28% relative
  annualized return (CSI300/800, supervised). Lesniewski & Trigila (2412.00036):
  CvM-validated simulation. Denoiser (2409.02138) skip.

### LOB/microstructure (context only — intraday, skip for daily bars)
TRADES/DeepMarket (2502.07071), LOB-image inpainting (2509.05107), **LOB-Bench**
(ICML 2025, 2502.09172 — evaluation-methodology template), MarS (ICLR 2025,
2409.07486 — MIT code but no weights, 128 GPUs).

### GAN lineage (stylized-facts evidence base)
- TimeGAN (NeurIPS 2019, Apache-2.0, TF1-era) — baseline citation only.
- **QuantGAN** (1907.06673): TCN generator reproduces vol clustering, leverage
  effect — the stylized-facts reference a diffusion model must beat.
- **Tail-GAN** (2203.01664, Management Science): trains against VaR/ES so
  **tails of trading-strategy P&L distributions are preserved** — the single
  most strategy-relevant evaluation criterion found. **Usable-now
  (methodologically)** for stress evaluation.
- Sig-WGAN / Conditional Sig-WGAN (2006.05421, 2111.01207): signature-Wasserstein
  → cheap, stable path generation. Promising low-compute alternative.
- Fin-GAN (Quant. Finance 2024) forecasting-oriented, promising.
- **Kwon & Lee** (2410.09850): stylized-fact capture varies sharply with
  architecture; naive GANs fail. **Key caution: a bad generator injects a wrong
  prior.**
- Market-GAN (AAAI 2024, 2309.07708): context-conditioned generation (regime +
  ticker), DJIA 2000–2023; beats 4 baselines on downstream usability.
- VolGAN (IV surfaces, SPX) — out of scope unless options enter.

### The gap
Almost nobody measures whether the **conditional signal structure**
(momentum/mean-reversion an agent must learn) survives generation. Tail-GAN's
strategy-aware loss and Path Shadowing MC come closest. We must build this
evaluation ourselves.

## 2. Flow matching for time series & finance

- Stochastic Interpolants (2303.08797) — theoretical backbone.
- **FlowTS** (2411.07506): rectified flow for TS generation, SOTA context-FID on
  Stock data; code thin, **no license**. Promising.
- **TSFlow** (ICLR 2025, 2410.03024): conditional FM with GP priors; best-
  credentialed FM-for-TS; code official but **no license**.
- Functional FM (AISTATS 2024, 2305.17209) overkill. Föllmer-interpolant
  forecasting (2403.13724) promising direction. PHINN (2606.15452): FM +
  persistent-homology conditioning for rare-event generation — watch.
- **FM vs diffusion verdict:** verified advantage is sampling cost (1–few ODE
  steps vs 50–1000) and simpler training — matters for mass path generation; no
  finance-specific fidelity advantage shown yet.
- **Neural-SDE / signature alternative (strong in low-data finance):**
  Gierjatowicz et al. (2007.04154); **Issa, Horvath, Lemercier, Salvi
  (2305.16274)** — non-adversarial neural-SDE via signature-kernel scoring
  rules, outperforms on rough-vol/FX/LOB; Buehler et al. (2006.14498) —
  signature-VAE "market simulator for small data environments"; Wiese et al.
  (2112.06823) — flow-based multi-asset simulator.

## 3. Synthetic data for RL training — the crux

- **SynthER** (NeurIPS 2023, 2303.06614, MIT): diffusion upsamples the replay
  buffer; residual-MLP denoiser designed for low-dim transitions — **directly
  portable to a daily-bar trading MDP. Usable-now; most transferable artifact.**
- Policy-Guided Diffusion (2404.06356): trajectory-level successor. Promising.
- DIAMOND (NeurIPS 2024 spotlight, 2405.12399): RL inside diffusion world model
  (Atari); no finance application found — Dreamer-on-markets is unexplored.
- **Gao, Zha, Zhou (2509.08731)**: conditional diffusion generates SDE paths for
  RL in continuous-time mean-variance portfolio selection; KL bounds +
  empirical RL gains. **Closest published match to our exact question.**
- **Deeper Hedging** (2310.18755): DDPG hedger **trained purely on calibrated
  synthetic ABM data outperforms baselines on real data** — the
  train-on-synthetic → test-on-real pattern works when the generator matches
  stylized facts.
- **Path Shadowing Monte-Carlo** (Morel, Mallat, Bouchaud, 2308.01486, Quant.
  Finance 2024): maximum-entropy Scattering-Spectra generator; SOTA realized-vol
  forecasts and S&P 500 smile predictions. Non-neural, low-compute,
  S&P-500-native, code available. **Usable-now; best conditional-structure
  evidence of anything surveyed.**
- **van Staden, Forsyth, Li (2303.08968)**: trains neural strategies on
  parametric vs **block bootstrap** vs GAN returns — the direct
  bootstrap-vs-generative comparison template.
- Filos (1909.09571), Yu et al. (1901.08740): early positive augmentation
  results, moderate evidence.
- "History Is Not Enough" (2601.10143): drift-aware augmentation + curation for
  financial TS, improves RL trading robustness. Promising, verify code/venue.
- Karzanov et al. (2502.02619): PPO trained on **circular block-bootstrap**
  paths + 20-agent averaging + regret reward improves on a 60/40 benchmark —
  closest published template for daily allocation.
- Riera Abbade & Reali Costa (2603.29086): realistic impact modeling **changed
  algorithm rankings entirely** (DDPG OOS Sharpe −2.1 → 0.3); cost-free envs
  actively mislead.

**State of evidence:** synthetic-path training reliably fixes
variance/overfitting (hedging, forecasting); **no published proof it
manufactures alpha in directional daily index trading**; a bad generator
actively hurts (Kwon & Lee; FTS-Diffusion's degrading baselines). Clean
SynthER-on-index-trading experiment = open niche for this project.

## 4. Diffusion/flow policies in RL — verdict: skip

Diffuser (2205.09991), Decision Diffuser (2211.15657), Diffusion Policy
(2303.04137), Diffusion-QL (2208.06193), IDQL (2304.10573), Consistency Policy
(2405.07503), FQL (2502.02538), π0 (2410.24164). Trading-specific: FlowHFT
(2505.05784, imitation/HFT), FlowOE (2506.05755, execution), DiffSTOCK
(2403.14063, prediction).

**Honest assessment:** the value of diffusion/flow policies is multimodal,
high-dimensional action distributions. Our action is 1-D exposure in [0,1] —
unimodal; a Gaussian/categorical head loses nothing. **Spend the diffusion
budget on data, not policies.**

## 5. Generative stress scenarios for evaluation

- **Tail-GAN** — strategy-P&L-tail-preserving scenarios (Management Science).
  Gold standard.
- **GuidedDiffTime** (JPMorgan, 2307.01717): constraint-guided diffusion for
  financial scenarios; new constraints need **no retraining** — best fit for
  "generate 2008-like paths" (condition on drawdown/vol at sampling time).
- CoFinDiff — trend/vol-conditioned crash paths. GAR (2603.08553) minimax risk
  scenarios, new. Flaig & Junike (2109.10072) regulatory precedent. Nehemya et
  al. (2010.09246) adversarial-perturbation red-team framing. True adversarial
  scenario generation vs RL trading agents = gap.

## Synthesis

**Ranked uses for our pipeline:**
1. **Stress-scenario evaluation** (lowest risk, immediate value): conditioned
   generation or resampled/scaled historical crisis windows; score every agent
   vs buy-and-hold on them. Templates: Tail-GAN criterion, GuidedDiffTime
   sampling, CoFinDiff conditioning.
2. **SynthER-pattern replay augmentation**: our MDP is tiny — exactly SynthER's
   regime. MIT code, hours on one GPU.
3. **Path-level generation for episode diversity**: block bootstrap first, then
   Diffusion-TS (neural) and **Path Shadowing MC / scattering spectra**
   (non-neural, stronger evidence) and Sig-WGAN (low-data-native).
4. **Flow matching as efficiency upgrade** to #2/#3 once diffusion baseline
   exists (not a fidelity upgrade).
5. **Diffusion/flow policies: don't.**

**Starting point:** week 1 = stationary/circular block bootstrap + stylized-fact
scorecard (tail exponent, ACF of |r| and r², leverage effect, variance-ratio
profile + signal-survival check); weeks 2–4 = one diffusion generator (SynthER
transitions or Diffusion-TS windows), mix synthetic:real {0,25,50,75}%, rerun;
then stress evaluation via conditional sampling. Everything fits one modest GPU
or largely CPU.

**Key repos:** conglu1997/SynthER (MIT) · Y-debug-sys/Diffusion-TS (MIT) ·
rudymorel/shadowing + scattering_spectra · marcelkollovieh/TSFlow (no license) ·
UNITES-Lab/FlowTS (no license) · issaz/sigker-nsdes (no license) ·
jsyoon0823/TimeGAN (Apache-2.0) · microsoft/MarS (MIT, no weights).
