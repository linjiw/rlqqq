# Core Positive Literature: (Deep) RL for Trading & Portfolio Management

Research-agent report, verified against primary sources 2026-07-31 (arXiv PDFs,
NeurIPS proceedings, Crossref, publisher pages).

## 1. Foundational work (1995–2006)

- **Neuneier 1995/1996** (NIPS 8): Q-learning + NN value function for asset
  allocation as MDP; German market; "superior to a heuristic benchmark policy";
  toy scale, no B&H comparison.
- **Neuneier 1997** (NIPS 10): multi-asset Q-learning with **transaction costs
  and risk preferences inside the MDP** — early recognition costs must live in
  the decision problem.
- **Moody, Wu, Liao & Saffell 1998** (J. Forecasting 17:441–470): introduces
  **Recurrent Reinforcement Learning (RRL)** / direct reinforcement and the
  **differential Sharpe ratio**; position enters policy input (cost-aware
  recurrence).
- **Moody & Saffell 1998** (NIPS 11): monthly S&P 500 vs T-Bill allocation,
  test 1970–1994; RL systems "outperform the S&P 500 over a 25-year test
  period."
- **Moody & Saffell 2001** ⭐ (IEEE TNN 12(4):875–889, DOI 10.1109/72.935097),
  the canonical citation:
  - S&P 500/T-Bill monthly, train 1950–, test 1970–1994, 84 macro inputs,
    0.5% costs. Total profit: **B&H 1,348% | Q-Trader 3,359% | RRL 5,860%**;
    Sharpe 0.34 / 0.63 / 0.83. Dodged 1974 & 1987.
  - USD/GBP half-hourly 1996: ~15% ann., Sharpe ≈ 2.3.
  - Red flags: 84-feature selection surface on one path; ensemble voting;
    pre-2000 regime; never replicated out-of-period at scale.
- **Dempster & Leemans 2006** (ESWA 30:543–552): adaptive RRL on EUR/USD
  1-minute, 2000–2002: **26% p.a. vs B&H −8%** — but currency B&H is a strawman
  and authors note post-2002 profit decay ("diminishing slope").

## 2. Deep RL era (2016–2022)

- **Deng et al. 2016** (IEEE TNNLS 28(3), 792 cites): RDNN/FDDR — deep features
  + recurrent direct RL, fuzzy representations; Chinese index futures +
  commodities; no code, no independent replication.
- **Jiang, Xu & Liang 2017** (arXiv:1706.10059, EIIE): crypto portfolio,
  30-min bars, "4-fold returns in 50 days" incl. 0.25% commission. Severe red
  flags: 2017 crypto bull, 50-day windows, no impact modeling. Methodology
  (portfolio-vector memory) still influential.
- **Liu et al. 2018** (arXiv:1811.07522): DDPG on Dow 30, test 2016–2018:
  ann. 25.87%, Sharpe 1.79 vs DJIA 16.40%/1.27. **No transaction costs in
  results**; bull-only window.
- **Yang, Liu, Zhong & Walid 2020** ⭐ (ICAIF '20 ensemble; the most-cited
  FinRL-line result): PPO+A2C+DDPG quarterly-switched by validation Sharpe +
  **hand-crafted turbulence-index risk-off**; Dow 30, 0.1% costs, trade
  2016–2020-05. Ensemble Sharpe 1.30 vs DJIA 0.47. Red flags: DJIA benchmark
  measured to COVID trough; risk-off rule is not learned; selection layer on
  top of backtest; single path.
- **Zhang, Zohren & Roberts 2020** ⭐ (JFDS 2(2):25–40; cleanest study): 50
  liquid futures 2011–2019, DQN/PG/A2C, LSTM state, vol-scaled reward with
  20bp cost term. Portfolio-level: DQN Sharpe 1.288 vs long-only 0.058. **BUT
  equity-index sub-portfolio: long-only 0.688 BEATS DQN 0.648 / A2C 0.510 /
  PG 0.447** — authors: "except for the equity index where a long-only strategy
  is better."
- **Théate & Ernst 2021** ⭐ (ESWA 173:114632, TDQN; **directly tests SPY, QQQ,
  DIA**): train 2012–2017, test 2018–2019, 0.1% costs. **Sharpe: SPY 0.834 =
  B&H 0.834; QQQ 0.845 = 0.845; DIA 0.684 = 0.684** — the agent "efficiently
  learns to tend toward a passive trading strategy"; at higher costs it stops
  trading entirely. Most honest positive paper; answer for index ETFs is
  "≈ buy-and-hold."
- **FinRL ecosystem** (Liu et al.): FinRL 2011.09607 / ICAIF'21 2111.09395;
  FinRL-Podracer 2111.05188 (+12–35% ann. vs other DRL baselines, not B&H);
  FinRL-Meta NeurIPS'22 D&B 2211.03107 — names "low signal-to-noise,
  survivorship bias, model overfitting" as the field's central problems.
  Library papers = demos, not controlled experiments.
- Others: Jeong & Kim 2019 (ESWA 117; ×-improvements vs own baseline framing);
  Pendharkar & Cusatis 2018 (ESWA 103; two-asset stock+bond RL beats single
  assets — multi-asset timing); Almahdi & Yang 2017 (ESWA 87; Calmar-objective
  RRL, hedge-fund benchmark); Huang 2018 (1807.02787, FX DRQN);
  Kabbani & Duman 2022 (IEEE Access; TD3 "Sharpe 2.68" — classic overfit flag);
  **Zejnullahu et al. 2022** (2206.14267, E-mini S&P DDQN: best model Sharpe
  0.915 vs long-hold 0.883 in a **no-cost env**, other feature sets fail —
  within noise); Hirsa et al. 2021 (2106.08437, self-described "preliminary").

## 3. Surveys

- Fischer 2018 (FAU): critic/actor/actor-critic taxonomy.
- **Hambly, Xu & Yang 2023** (Math. Finance 33:437–503): authoritative; does
  **not** endorse beating-index claims.
- **Pricope 2021** (2106.00123): "no decent profitability level was obtained";
  proof-of-concept in unrealistic settings; field "in the very early stages."
- Millea 2021 (Data 6:119): community inconsistency impedes progress.
- Sun, Wang & An 2021 (2109.13851): non-stationarity + evaluation protocol
  inconsistency are the central challenges.

## 4. Synthesis

1. **Mechanism claim (robust):** RL natively embeds costs/positions/risk in the
   objective; agents demonstrably modulate trading frequency with cost levels
   (Théate: passive at 0.2%; Zhang: cost-tolerance).
2. **Cross-sectional/multi-asset claim (moderate):** diversified futures
   (Zhang: DQN 1.29 vs 0.06) and multi-stock+risk-off (Yang ensemble) beat
   passive — value comes from regime timing across assets, not out-trading one
   instrument.
3. **Single-instrument claim (weak):** headline wins are FX (falling-currency
   strawman, decayed) and crypto (regime artifact).

**For liquid US index products specifically, the best evidence CUTS AGAINST the
hypothesis:** Théate & Ernst (TDQN = B&H exactly on SPY/QQQ/DIA), Zhang et al.
(long-only beats all RL on equity indices), Zejnullahu (within-noise edge,
no-cost env). Positive index-adjacent results rely on (a) multi-asset
allocation into safe assets, (b) hand-crafted risk-off overlays, (c) pre-2000
or crisis-containing windows, (d) benchmark choice. **No credible,
cost-inclusive, multi-year demonstration of DRL beating B&H on SPY/QQQ/^GSPC/
^NDX as a single instrument exists in this literature. The DCA benchmark is
essentially untested.** A positive result from our project would be a NEW
result; systematic red flags to avoid: short windows, zero/unstated costs,
single paths, selection layers, survivor-of-the-literature bias.
