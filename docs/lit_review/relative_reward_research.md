# Reward Design for Beating a Vol-Target Baseline End-to-End (verified 2026-08-02)

Key findings (all arXiv-verified):

1. Benchmark-relative excess-return rewards train stably in PPO when paired
   with a risk/correlation penalty (AlphaZeroBeta 2607.18001, Financial
   Innovation). Direct differentiable-Sharpe is the proven modern successor
   to Moody-Saffell (2005.13665, 1911.10107); turnover control decides
   whether edges survive costs (2607.00475).
2. Documented pathologies: over-conservatism is the better-documented
   failure of Sharpe/relative rewards (2511.11481), not excess risk;
   proxy-reward gap (2506.20930); penalty terms interact non-monotonically
   (2604.00031). Relative log-wealth games have well-defined non-degenerate
   optimal leverage (2503.02722) - theoretical license for our reward.
3. Residual RL: initialize at identity residual (m=1), normalize critic to
   the RESIDUAL reward scale (2602.10539 - cold-start + scale-mismatch are
   the two failure modes). Zero finance applications found: our construction
   is publishable as such.
4. Log-utility PPO CAN learn correct leverage (Lu 2307.07694: PPO converges
   near Kelly optimum); our v7 refusal was a data-distribution effect -
   bootstrap crash paths shift empirical Kelly down (Hsieh 1710.01786:
   heavy tails rationally collapse log-optimal sizing). The relative reward
   fixes this cleanly: holding m=1 on a crash path scores ZERO, so crash
   paths only teach "don't tilt UP into crashes" - the correct lesson.
5. Risk-sensitive dial: exponential risk-sensitive RL = log objective +
   quadratic-variation penalty (Jia 2404.12598); benchmarked risk-sensitive
   control has explicit fractional-Kelly decomposition (Lleo-Runggaldier
   2603.00738/2606.20903). Recursive-utility PPO beat plain discounted on
   all 10 splits (2603.22880).
6. Two-stage decomposition has a name: META-LABELING (Joubert et al., JFDS
   2022-2023). Evidence pattern 2024-2026: neither pure end-to-end nor pure
   two-stage wins - end-to-end ANCHORED TO the two-stage solution does
   (Guided Learning 2411.10496, trust-region DFL AAAI'25 2501.01874).
   Calibrate the multiplier output before deployment (JFDS .119).

Recommended spec (implemented as ppo_v9_rel/rel5): residual PPO on the
DEPLOYMENT baseline (vt20 cap1.5), reward = dlog-wealth vs holding that
baseline, quadratic-variation penalty as risk dial (todo: v9c), critic on
residual scale, ensemble smoothing retained, CIs mandatory.
