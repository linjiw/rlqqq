# Skeptical / Negative-Results Literature

Research-agent report, verified 2026-07-31 (Crossref, arXiv, publisher PDFs).

## 1. Market efficiency baseline

- **Fama 1970** (J. Finance 25:383–417): EMH — weak form says past prices carry
  no exploitable information after costs. Our null hypothesis; SPY/QQQ/VOO is
  the hardest possible target.
- **Malkiel 2003** (JEP 17:59–82): documented predictabilities are small,
  non-robust, decay post-discovery, rarely exploitable after costs. Report
  economic, not just statistical, significance.
- **Lo & MacKinlay 1988** (RFS 1:41–66): variance-ratio tests reject random
  walk for weekly index returns 1962–1985 — but signal concentrated pre-1985
  and has decayed. Honest counterweight.
- **Welch & Goyal 2008** (RFS 21:1455–1508): every proposed equity-premium
  predictor failed in- and out-of-sample; "a real-time investor could not have
  used them to time the market." Index timing has a graveyard.
- **Moskowitz, Ooi & Pedersen 2012** (JFE 104:228–250): time-series momentum in
  58 liquid futures incl. equity indices — the best-documented exception.
  TSMOM must be a baseline; regress agent returns on it.
- **Huang, Li, Wang & Zhou 2020** (JFE 135:774–794): "TSM: Is It There?" —
  asset-level evidence weak; TSMOM ≈ historical-mean strategy (vol-scaled
  long drift). Compare agent vs vol-scaled B&H too.

## 2. Backtest overfitting & multiple testing

- **Bailey, Borwein, López de Prado & Zhu 2014** (Notices AMS 61:458–471):
  enough trials guarantee high in-sample Sharpe on noise; expected max Sharpe
  of N trials; undisclosed trial counts = "scientifically meaningless."
- **Bailey & López de Prado 2014** (JPM 40(5):94–107): **Deflated Sharpe
  Ratio** — corrects for trials, track length, skew/kurtosis. (2026 extension:
  López de Prado, Lipton & Zoonekynd, JPM, DOI 10.3905/jpm.2026.1.837.)
- **Bailey et al. 2016** (J. Comp. Finance 20(4):39–69): **PBO via CSCV** —
  probability the in-sample winner underperforms the median OOS.
- **Harvey, Liu & Zhu 2016** (RFS 29:5–68): 316 factors; t > 3 required; "most
  claimed research findings in financial economics are likely false." Also
  Harvey & Liu 2014 (Sharpe haircuts), Harvey 2017 presidential address.
- **White 2000** (Econometrica 68:1097–1126) Reality Check; **Hansen 2005**
  (JBES 23:365–380) SPA; **Sullivan, Timmermann & White 1999** (J. Finance
  54:1647–1691): 7,846 technical rules on DJIA over 100 years — no significant
  edge after snooping correction; profits vanished in the latest decade.
- **Arnott, Harvey & Markowitz 2019** (JFDS 1:64–74): 7-point ML backtesting
  protocol — our methods-section skeleton.

## 3. Replication failures & ML/RL trading critiques

- **Pricope 2021** (2106.00123): DRL trading "very early stages"; "no decent
  profitability level was obtained."
- **Millea 2021** (Data 6:119): pervasive inconsistency; leakage-prone state
  representations.
- **Snow 2020** (JFDS 2:10–23): backtest-live gap systematic; simple supervised
  baselines often match RL. Include a boring ML baseline.
- **López de Prado 2018** (JPM 44(6):120–133): 10 reasons ML funds fail —
  purged/embargoed CV, run-once backtests, trial registries.
- **Israel, Kelly & Moskowitz 2020** (JOIM 18(2)): finance is structurally
  hostile to ML — one history, low SNR, adversarial adaptation, no simulator.
  RL is the most data-hungry paradigm on ~15k daily bars.
- **Lu 2023/2025** (2307.07694): on simulators with known optimum, DDPG/TD3/SAC
  fail on noisy rewards; PPO/A2C need >2M steps ≈ **"almost 8,000 years of
  daily prices."** Prefer on-policy; sanity-check on synthetic known-optimum
  markets; address the data budget.
- **Kashif & Ślepaczuk 2026** (2605.17307): SAC, Nasdaq-100/Nikkei/EuroStoxx,
  2003–2026, 16 walk-forward folds, costs, HAC: **no significant excess over
  B&H** (except EuroStoxx). The closest study to our exact question — null.
- **Deep, Deep & Lamptey 2025** (2512.12924): rigorous walk-forward on 100 US
  equities: 0.55% ann., Sharpe 0.33, p = 0.34. The typical fate of honest
  validation.
- **Gort, Liu et al. 2022** (2209.05559, FinRL group itself): prior DRL results
  "may suffer from the false positive issue due to overfitting"; hypothesis-test
  screen for agent-level overfitting. Also PRUDEX-Compass (2302.00586, TMLR):
  re-evaluates 8 methods, profit-only evaluation "far from satisfactory."

## 4. RL-specific methodological traps

- **Henderson et al. 2018** (AAAI, 1709.06560): seed variance makes single-run
  RL results uninterpretable. ≥10 seeds, distribution reporting.
- **Agarwal et al. 2021** (NeurIPS Outstanding Paper, 2108.13264): IQM +
  stratified bootstrap CIs + performance profiles (`rliable`).
- **Zhang, Vinyals, Munos & Bengio 2018** (1804.06893): deep RL overfits
  robustly; training reward says nothing about generalization. (Echoed in
  trading by 2506.20930.)
- **Skalse et al. 2022** (NeurIPS, 2209.13085): reward hacking formalized —
  every proxy is exploitable. Evaluate with an independent backtester, never
  the training env.
- **Data-side traps:** Glasserman & Lin 2023 (2309.17322) look-ahead via
  pretrained components; Blanchet et al. 2022 (2202.00871) full-sample
  normalization/imputation leaks; FinRL-Meta's own bias list; Kong et al. 2026
  (2602.14233) five recurring biases across 164 papers. All normalization must
  be point-in-time; pretrained components must predate test windows.

## 5. DCA vs lump-sum

- **Constantinides 1979** (JFQA 14:443–450): DCA is provably dominated —
  a behavioral baseline, not an efficiency frontier. Fair comparison requires
  matched cash-flow schedules.
- **Vanguard (Finlay & Zorn) 2023**: lump-sum beats 3-month cost averaging 68%
  of the time at 1y horizon (MSCI World 1976–2022); CA only wins in the worst
  tail. B&H is the strong benchmark; "beating DCA" alone is weak (an agent
  beats DCA merely by being fully invested). Report tail percentiles. Also
  Brennan, Li & Torous 2005 (Rev. Finance 9:509–535).

## Synthesis: credibility checklist (design requirements)

Failure mode to design against: *a long-biased agent, selected from many trials
on one bull-market history with leaky preprocessing and no cost model, "beats"
a mis-specified benchmark.*

- Baselines: lump-sum B&H (dividends reinvested); DCA with matched cash flows;
  vol-scaled B&H; 12-mo TSMOM; simple supervised-ML signal on same features.
- Data hygiene: point-in-time normalization inside each fold; no pretrained
  component overlapping test windows; survivorship-safe features.
- Protocol: purged/embargoed walk-forward, test touched once; regime-diverse
  test windows (2008, 2015/2018, 2020, 2022, post-2023); realistic execution
  incl. cash interest; independent backtester.
- Inference: ≥10 seeds, IQM + bootstrap CIs, never best-seed; full trial
  registry; DSR with N = trial count; PBO via CSCV; White RC / Hansen SPA vs
  B&H; HAC-robust t ≥ 3 pre-registered.
- Sanity: simulator known-optimum check (Lu 2023); data-budget statement;
  long-bias decomposition (excess return after subtracting exposure × index);
  pre-registered null expectation — a rigorous negative result is publishable.
