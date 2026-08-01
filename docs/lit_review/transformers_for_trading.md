# Transformers for Index-Trading RL: Feature Engineering & Policy

Research-agent report, sources verified 2026-07-31 (arXiv abstracts, GitHub source,
HuggingFace model cards). Raw material for `docs/investigation_plan_new_models.md`.

## 1. Time-series foundation models (TSFMs) as feature extractors

| Model | Arch | Params | Context | License | Finance pretraining? | Embedding API? |
|---|---|---|---|---|---|---|
| TimesFM 2.5 (Google) | decoder-only, patched | 200M | 16,384 | Apache-2.0 | No | No documented API; LoRA via PEFT |
| Chronos / Chronos-Bolt / Chronos-2 (Amazon) | T5 enc-dec / patch-based / encoder-only | 9M–710M; Chronos-2 120M | 512 / 2048 / 8192 | Apache-2.0 | Not stated | **Yes — verified in source**: `ChronosPipeline.embed()` and `ChronosBoltPipeline.embed()` (returns patch embeddings + loc/scale) |
| Moirai 1.1 / 2.0, Moirai-MoE (Salesforce) | masked-encoder / decoder-only | 11–91M | n/s | **CC-BY-NC-4.0** | fred_md, bitcoin only | No |
| Lag-Llama | decoder-only w/ lags | small | trained at 32 | Apache-2.0 | unspecified | No |
| TimeGPT (Nixtla) | undisclosed | undisclosed | undisclosed | **closed, API-only** | undisclosed | No |
| MOMENT-1 (CMU, ICML 2024) | T5-encoder, masked modeling | 341M | 512 | **MIT** | Time-series Pile | **Yes** — `task_name='embedding'` first-class |
| Time-MoE (ICLR 2025) | sparse-MoE decoder | 50–200M activated | 4096 | Apache-2.0 | unspecified | No |
| Sundial (Tsinghua 2025) | transformer + TimeFlow flow-matching loss | 128M | 2880 | Apache-2.0 | no breakdown | No; CPU timings published (249–949 ms/forecast, M1 Pro) |
| TTM / granite-ttm-r2 (IBM) | MLP-mixer, ~1–5M | 805k–5M | 512–1536 | Apache-2.0 | Bitcoin only | No; CPU-trivial |
| **Kronos** (AAAI 2026) | **finance-native** hierarchical OHLCV tokenizer + AR decoder | 4.1M/24.7M/102.3M | 512 (mini 2048) | **MIT** | **Yes — 12B+ K-line records, 45 exchanges** | Hidden states extractable (~30 lines of surgery) |

Key arXiv: 2310.10688 (TimesFM), 2403.07815 (Chronos), 2510.15821 (Chronos-2),
2402.02592/2410.10469 (Moirai/-MoE), 2310.08278 (Lag-Llama), 2402.03885 (MOMENT),
2409.16040 (Time-MoE), 2502.00816 (Sundial), 2508.02739 (Kronos).

## 1.2 Do TSFMs work on financial data? (critical evidence)

- **Rahimikia, Ni & Wang, arXiv:2511.18578** ("Re(Visiting) TSFMs in Finance"): on a
  large panel of daily excess returns, off-the-shelf TSFMs "perform poorly in
  zero-shot and fine-tuning settings"; from-scratch pretraining on financial data
  achieves substantial improvements. **Most on-point paper: zero-shot TSFM
  features on daily returns are weak; domain pretraining is what pays.**
- **Cheung, arXiv:2607.12248** ("When Directional Accuracy Lies"): LoRA-tuned
  TimesFM on NASDAQ-100 + S&P 500, walk-forward + McNemar/DM under BH-FDR: the
  ~80% directional accuracy is a base-rate artifact (always-up rule matches);
  "no directional skill over the base rate at any horizon." **Rigorous negative
  on exactly our asset class.**
- **Noguer i Alonso & Franklin, arXiv:2606.27100**: TSFMs vs NBEATS/PatchTST/
  iTransformer on daily returns of 5 US large caps: TSFM gains vs random walk
  "small and sparse" (DM rejects equality in only 2 comparisons). iTransformer
  was the only from-scratch model to beat all TSFMs on some assets.
- **Goel, Pasricha, Magris & Kanniainen, arXiv:2505.11163**: TimesFM for
  **realized volatility**: fine-tuned variants statistically outperform
  traditional models (DM + Giacomini-White). **Vol, not returns, is where TSFMs
  demonstrably add value.**
- **Marconi, arXiv:2507.07296**: Chronos/TTM zero-shot beat naive benchmarks on
  vol and spreads (2 of 3 tasks).
- **Das, Goyal & Yadav, arXiv:2605.21504**: Chronos-2 multivariate context
  consistently beats univariate within a related asset group; mixing equities
  with rates hurts.
- **Kronos, arXiv:2508.02739**: finance pretraining boosts price-forecast RankIC
  +93% over leading TSFM. Same vein: FinCast 2508.19609, LENS 2408.10111.
- Fu, Hirano & Imajo, arXiv:2412.09880: continual pretraining of TimesFM on 100M
  financial points improves mock-trading metrics.

Verdicts: Chronos-Bolt/Chronos-2 embeddings **usable-now**; MOMENT usable-now;
Kronos **promising/usable-now** (only open finance-pretrained model); TimesFM
promising (no embed API; negative directional evidence); Moirai skip (NC
license); Lag-Llama skip; TimeGPT skip; Time-MoE/Sundial/TTM promising.

## 2. Supervised transformers vs the skeptical line

- **PatchTST** (ICLR 2023, 2211.14730): patching + channel independence; default
  strong supervised baseline. **iTransformer** (ICLR 2024, 2310.06625): attention
  over variates; direct positive evidence on daily US equity returns. **TFT**
  (1912.09363): interpretability option.
- **Skeptics:** Zeng et al. AAAI 2023 oral (2205.13504) — DLinear/NLinear beat
  all LTSF transformers, often by a large margin. Toner & Darlow (2403.14587) —
  linear TS models ≈ unconstrained linear regression; closed-form beats trained
  in 72% of settings. TSMixer (TMLR 2023, 2303.06053) — all-MLP matches SOTA;
  cross-variate info helps on rich data. Tan et al. NeurIPS 2024 Spotlight
  (2406.16964) — ablating the LLM out of LLM-forecasters doesn't hurt.
- **Mamba/SSM:** S-Mamba (2403.11144), TimeMachine (2403.09898) promising for
  cheap long context; MambaStock (2402.18959) skip (weak evidence).
- **FinTSB** (2502.18834): financial forecasting benchmark with transaction fees
  — evaluation-harness reference.

Conclusion: on noisy daily returns, burden of proof sits on attention; carry a
ridge/DLinear baseline always. Attention earns keep via cross-series/covariate
structure and vol forecasting, not single-series return memorization.

## 3. Transformers in the RL loop for trading

- **Decision Transformer** (2106.01345) — foundational only.
- **Yun, arXiv:2411.17900** (ICAIF'24 wksp): GPT-2+LoRA DT on FinRL trajectories;
  competitive with CQL/IQL/BC; **no buy-and-hold victory claimed**; repo
  unlicensed.
- **StockFormer** (IJCAI 2023): 3 transformer branches (long/short-term,
  relational predictive coding) → latents fused by SAC; critic gradients
  backprop into encoders; beats baselines on CSI-300/NASDAQ/crypto. Canonical
  "transformer encoder + RL" template; repo unlicensed; Chinese-market-centric.
- **Kashif & Ślepaczuk, arXiv:2605.17307**: SAC with LSTM vs Transformer
  encoders, Nasdaq-100/Nikkei/EuroStoxx, 2003–2026, 16 walk-forward folds,
  costs, HAC inference: **"no strategy achieves statistically significant excess
  returns relative to Buy and Hold"** (except EuroStoxx). **Null-hypothesis
  anchor for our project.**
- Raj 2509.14385 (Transformer-PPO best risk-adjusted; weak baselines); Kar et al.
  2507.19639 (supervised Crossformer with profit-guided loss beats PPO/DDPG —
  transformer value may not need RL); "Red Queen's Trap" 2512.15732 (validation
  →live decay warning); 2604.10996 (valid LLM features lose to price-only PPO
  under regime shift).
- Trajectory-transformer/world-model trading papers: none credible found — gap.

## 4. Self-supervised representations

- **TS2Vec** (AAAI 2022, 2106.10466, MIT): hierarchical contrastive; linear head
  on top beats prior SOTA; small dilated-CNN encoder, CPU-trainable.
  **Usable-now.**
- CoST (ICLR 2022, 2202.01575): seasonal-trend disentangle — arguably
  mis-specified for returns. SimMTM (NeurIPS 2023, 2302.00861): masked modeling,
  promising. MOMENT again = the practical "financial BERT" shape. DGRCL
  (2412.04034) cross-sectional; THEME (2508.16936) off-target.
- Nobody has published "pretrain cross-asset → fine-tune on SPY" — gap we could
  fill.

## Synthesis

**Ranked integrations for our pipeline:**
1. **Chronos-Bolt-small embeddings** of trailing 512–2048-day windows → PCA to
   32–64 dims → concat with classical features. Verified API, Apache-2.0,
   CPU-feasible, precomputed offline (zero RL-training cost).
2. **Kronos-small hidden states** — finance-native embeddings, MIT, needs ~30
   lines to expose hidden states; highest expected signal per domain-pretraining
   evidence.
3. **TSFM vol forecasts as features** (TimesFM/Chronos-2 quantile forecasts of
   next 5/21-day realized vol) — the one task with statistical wins; helps
   position sizing. Chronos-2 multivariate over {SPY,QQQ,VOO} jointly.
4. **StockFormer-style predictive-coding encoder** (small PatchTST/iTransformer,
   optional critic-gradient flow) — the expensive arm; pre-register the Kashif &
   Ślepaczuk null as expected outcome.

**Skip:** Moirai (NC), TimeGPT (closed), Lag-Llama (context 32), LLM-backbone
forecasters, MambaStock, unlicensed code reuse (StockFormer, finrl-dt) without
clarification.

**Compute:** all embedding/forecast integrations are offline batch jobs
(minutes–hours on 8-core CPU); only the in-loop encoder needs a modest GPU
(hours).
