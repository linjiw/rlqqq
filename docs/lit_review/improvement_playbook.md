# Improvement Playbook (research agent, verified 2026-08-01)

Full report from targeted bottleneck research. Top-5 ranked changes:

1. **Residual/anchored policy around vol-target baseline**: multiplicative
   residual actions {0.5x, 1.0x, 1.5x} on vol-target exposure, critic warmed
   on baseline; residual RL evidence arXiv:1812.06298, DAWN 2602.10539
   (avoid critic cold-start + scale mismatch); KL-to-prior alternatives
   1905.01240, 2106.06860 (TD3+BC), JSRL 2204.02372, finance BC-warmstart
   2503.04218. Zero residual-RL papers in q-fin = novel.
2. **Hysteresis + amplified switch penalty**: Novy-Marx & Velikov (RFS 2016)
   sS bands cut turnover -41% with HIGHER net returns, dominating frequency
   reduction; No-Transaction Band Network 2103.01775; Policy Inertia
   2103.02287; train-at-higher-costs precedent 1911.10107 (20bp), turnover
   reg rescue 1904.04912 (Sharpe -5.31 -> +0.91). Sweep lambda in {1,2,5,10}bp,
   expect hump-shaped curve, cap turnover ~10x/yr. Anneal entropy to 0,
   deterministic eval.
3. **Mean-probability seed ensemble, ties->hold**: gas-futures DQN ensemble
   2301.08359 (Sharpe 1.20 vs 0.975 single, lower turnover); FinRL contest
   2501.10709 (std halved, N=3 saturation for small action spaces); prob
   average > majority vote (1704.01664). Don't Sharpe-weight seeds.
4. **Weight averaging kills selection**: SWA 1803.05407, LAWA 2209.14981,
   last-k theory 2411.13169; **Ensemble-of-Averages 2110.10832 restores
   val-test rank correlation under shift** — testable prediction for our
   -0.15. Save checkpoints last third, uniform-average state dicts, freeze
   normalizer. PBO/CSCV as diagnostic; uniform or (1-PBO) weighting when
   selection fails dominance.
5. **Regime-stratified bootstrap + HMM-posterior conditioning**: prob-DDPG
   2511.00190 (feed filtered posterior, not hard label); FR-LUX 2510.02986;
   Karzanov 2502.02619 has no regime successor = unclaimed; CAVEAT
   2604.14498: synthetic augmentation helps variance-dominant tasks (risk
   control), hurts bias-dominant directional prediction — expect gains on
   drawdown axis, not alpha.

**Frontier verdict (verified sweeps):** no credible published daily-bar
single-index strategy with post-2010 OOS Sharpe > 1.1 net of costs exists.
All >1.1 claims come from cross-sectional breadth, intraday, or gross/
contaminated evals. Vol-targeting sits ON the frontier (~1.0-1.1;
Moreira-Muir JF 2017, Cederburg JFE 2020 fragility caveat). Matching B&H
Sharpe at 2/3 drawdown = already at frontier; decisively beating 1.10 would
exceed all published precedent.

Additional facts: PBO literature formalizes our negative val-test corr
(Bailey et al: OOS-on-IS slope negative "in most practical cases";
SR_OOS = 0.87 - 0.75*SR_IS worked example). FinRL "ensemble" (ICAIF'20) is
quarterly best-val SELECTION, not ensembling — the broken operation.
DeepAries 2510.14985: PPO jointly learning rebalance interval beats fixed
frequencies. Garleanu-Pedersen JF 2013: partial adjustment toward aim is
optimal under impact costs (justifies EMA execution filter).
