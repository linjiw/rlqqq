# Recent Advances (2022–2026) in RL/ML Relevant to the Study

Research-agent report, verified 2026-07-31 against arXiv listings/abstracts and
project pages.

## 1. Modern RL algorithms in trading

- **Distributional RL:** Hêche et al. 2501.04421 (C51/QR-DQN/IQN on gas futures,
  CVaR-tuned; +32% over baselines — commodities, no B&H framing). **Prasad
  2607.11607: 40–95% of distributional-RL risk claims refuted against MC ground
  truth.** Noguer i Alonso 2605.30464: equal-weight/Black-Litterman/Ledoit-Wolf
  all out-Sharpe every distributional method on a 25-asset DJIA backtest.
  → Distributional-RL-as-risk-silver-bullet is hype; audit any CVaR agent.
- **Offline RL / DT:** Yun 2411.17900 (GPT-2 LoRA DT, competitive with CQL/IQL/
  BC, RL-vs-RL only). The offline-RL trading record 2024–2026 is otherwise
  nearly empty — no credible beat-B&H result.
- **Model-based:** Huang et al. 2301.09297 (heavy-tail normalizing-flow world
  model; benefit = lower drawdown through COVID, not higher return).
- **Ensembles:** Holzer et al. 2501.10709 (FinRL contests, 2,048 parallel GPU
  envs; ensemble gains vs single agents: MaxDD −4.17%, Sharpe +0.21 — not vs
  index).
- **Risk-sensitive:** Benhenda 2502.07393 (FinRL-DeepSeek CPPO + LLM risk
  signals, Nasdaq news 2013–2023: **plain PPO wins bull markets, CPPO wins bear
  markets** — regime split is the useful finding). Koren et al. 2510.08226
  (UAMDP: CVaR-constrained planning, Sharpe 1.54→1.74, MDD halved — HFT).
- **Hierarchical:** EarnHFT 2309.12891 (AAAI'24), MacroHFT 2406.14537 (KDD'24)
  — crypto HFT; HRT 2410.14927 (Sharpe 1.06→1.24, 2020–2023, cross-sectional);
  Hi-DARTS 2509.12048 (17-month window, single stock — weak).
- **StockFormer** (IJCAI 2023): predictive-coding transformer branches + SAC in
  latent space; the most relevant "encoder + RL" template; CSI-centric.
- **Counter-evidence:** **Kashif & Ślepaczuk 2605.17307 — the most rigorous
  recent study on exactly our asset class (Nasdaq-100, 2003–2026, 16 folds,
  costs, HAC): no significant excess over B&H.** Ma 2509.12764 (Malliavin-
  calculus argument why RL portfolio strategies lose money under frictions;
  "phantom profit" from policy-gradient contamination). Maskiewicz & Sakowski
  2506.04658 (edge mainly from avoiding unfavorable conditions).

## 2. Sequence models

- **Saly-Kaufmann, Wood, Calliess & Zohren 2603.01820** (Oxford, 43pp
  benchmark): linear vs LSTM/xLSTM/PatchTST/SSMs on daily futures 2010–2025
  with significance tests and cost robustness. **VSN+LSTM best Sharpe; xLSTM
  largest break-even-cost buffer.** The best architecture guidance available
  for daily position sizing.
- Mamba cluster (FinMamba 2502.06707, SAMBA 2410.03707, CryptoMamba,
  MaGNet): prediction papers with bolted-on backtests; cheap long-context
  encoder candidates, unproven alpha.
- DTs for trading: nearly empty record; Weinberg 2512.15738 low credibility.

## 3. LLM + RL hybrids

Agent systems: FinMem 2311.13743, FinAgent 2402.18485 (KDD'24, "36% profit
improvement"), FinCon 2407.06567 (NeurIPS'24), TradingAgents 2412.20138,
StockAgent 2407.18957, FLAG-Trader 2502.11433, Trading-R1 2509.11420, DAPO
2505.06408 (230% cumulative on NDX news data — baseline is CPPO, not B&H),
FinRLlama 2502.01992.

**Audit literature (where the credibility lives):**
- **FINSABER (Li, Kim, Cucuringu & Ma), 2505.07078, KDD 2026 Oral:** backtests
  LLM strategies over ~two decades, 100+ symbols — **LLM advantages
  "deteriorate significantly" under longer windows; too conservative in bulls,
  too aggressive in bears.**
- **StockBench 2510.02209:** contamination-free benchmark — **most frontier
  LLMs fail to beat buy-and-hold.**
- **"The Alpha Illusion" 2605.16895:** reported Sharpe in FinMem/FinAgent/etc.
  inseparable from temporal contamination (test window inside LLM pretraining).
- **Agentic Trading survey 2605.19337 (77 studies):** of 19 primary studies,
  2 use time-consistent splits, 1 reports costs, 0 fully reproducible.
- **KTD-Fin 2605.28359:** with tickers/dates masked, LLM returns mostly explained
  by passive market/style exposure.
- Sentiment-into-RL: 2607.16028 (works in bear regime, big generalization gap);
  **2604.10996 (LLM features with IC>0.15 still lose to price-only PPO under
  macro shocks)**.

→ Verdict: LLM *agents* that trade = hype. LLMs as upstream feature extractors
= defensible but fragile under regime shift.

## 4. Market simulators & synthetic data

- JAX-LOB 2308.13289, JaxMARL-HFT 2511.02136, KineticSim 2606.21784 — GPU-
  parallel LOB sims (intraday; infrastructure signal).
- ABIDES-Gym 2110.14771 (ICAIF'21) — reactive multi-agent market env,
  still the reference.
- Market-GAN 2309.07708 (AAAI'24) — controllable regime/ticker-conditioned
  generation, DJIA 2000–2023.
- Diffusion price paths: Kim et al. 2507.19003 (GBM-score diffusion reproduces
  heavy tails/vol clustering/leverage); **SBBTS 2604.07159 (Schrödinger bridge:
  synthetic augmentation → higher accuracy AND Sharpe than real-only on S&P
  500)** — most direct augmentation-helps evidence.
- **Karzanov et al. 2502.02619: PPO trained on circular block-bootstrap paths,
  20-agent averaging, regret reward → beats 60/40 benchmark.** Closest template
  for daily allocation.
- **Riera Abbade & Reali Costa 2603.29086: realistic impact modeling changed
  algorithm rankings entirely (DDPG OOS Sharpe −2.1 → 0.3).** Cost-free envs
  actively mislead algorithm selection.

## 5. Benchmarks, frameworks, competitions

- FinRL-Meta 2211.03107 (NeurIPS'22 D&B) — infrastructure.
- TradeMaster (NeurIPS'23 D&B) + **PRUDEX-Compass 2302.00586 (TMLR)** — 6-axis
  17-measure evaluation emphasizing reliability over headline returns.
- **FinRL Contests 2504.02281: neither the paper nor the 2025 contest site
  publishes winner-vs-buy-and-hold results** — evaluation is peer-relative. The
  flagship competition has not produced an audited index-beating demonstration.
- FinWorld 2508.02292, Agent Market Arena 2510.11695 (framework design matters
  more than backbone).

## 6. 2024–2026 beat-the-index claims, scrutinized

| Paper | Claim | Verdict |
|---|---|---|
| EvoNash-MARL 2604.10911 | 19.6% ann. vs 11.7% SPY, walk-forward 2014–2024 | Self-reports **failing White RC / SPA-lite significance** — most honest strong claim |
| HMM+RL 2605.27848 | Beats SPY Sharpe/DD via SPY/TLT/GLD rotation | Multi-asset sleeve, no costs, single split |
| DAPO 2505.06408 | 230% cumulative NDX | Baseline is CPPO not B&H; IR 0.37 unremarkable |
| Hi-DARTS 2509.12048 | 25% vs SPY 20% | 17 months, single stock — not evidence |
| SBCA 2605.01384 | Beats B&H, 11y multi-asset, costed | Better than most; multi-asset though |
| Kar et al. 2507.19639 | Supervised Crossformer beats PPO/DDPG | Argues against RL machinery per se |
| **Kashif & Ślepaczuk 2605.17307** | **Negative on Nasdaq-100, 2003–2026, costed, HAC** | The rigorous benchmark |
| Deep hedging SPY 2512.12420 | Higher point Sharpe than long-SPY, CIs overlap | Model of honest reporting |

**Pattern: claim strength inversely correlates with methodological quality. No
paper credibly beats QQQ/SPY B&H after costs with statistical significance.**

## Synthesis

**Genuinely useful for us:** (1) evaluation/training infrastructure (parallel
envs, realistic cost envs, PRUDEX/FINSABER-grade protocols); (2) risk-sensitive
objectives (CPPO) + regime conditioning — the most replicated positive
mechanism (drawdown control in bear/high-vol regimes); (3) synthetic-path
training (block bootstrap first, then generative); (4) modern sequence encoders
(VSN+LSTM / xLSTM per Zohren benchmark); (5) ensembles + median policy.

**Hype for this problem:** LLM trading agents' headline Sharpe; distributional
RL as risk silver bullet; DTs/offline RL on markets; "RL got better therefore
alpha."

**Verdict on the project hypothesis:** 2022–2026 literature does NOT support
"RL improved → can now beat index B&H." What improved: training speed, tail-risk
control, honest evaluation. Realistic goal: match B&H return with materially
lower drawdown; beat DCA risk-adjusted. Statistically significant excess return
on NASDAQ/S&P products after costs remains undemonstrated.
