# v2 Report: Review + Vol-Penalty & Bootstrap Arms (2026-08-01)

## Review pass (all clean)

- 11/11 env tests pass; independent audit of all 80 saved v1 series:
  recomputed accounting matches saved returns to 0 error; exposures in [0,1];
  dates align with fold windows (fixed a pandas-3 date-serialization quirk in
  `walkforward.py` — old files decode fine with resolution detection).
- **Known-optimum validation added** (`tests/test_planted_signal.py`, the Lu
  2023 requirement): PPO on a synthetic market with a planted signal beats
  B&H by >1.0 Sharpe on held-out data; on a placebo (shuffled signal) it shows
  no edge. The stack can learn when signal exists and doesn't hallucinate
  when it doesn't → our null results on real data are informative.

## New arms (8 folds × 10 seeds each, 150k steps, 2 bps)

| config | CAGR | Sharpe | MaxDD | Calmar | ΔSharpe vs B&H [95% CI] |
|---|---|---|---|---|---|
| buy-and-hold | 15.7% | 0.92 | −33.7% | 0.47 | — |
| vol-target 10% (dumb baseline) | 10.6% | 1.04 | −12.2% | 0.87 | — |
| ppo_v1 | 8.8% | 0.72 | −17.9% | 0.49 | −0.20 [−0.47, +0.04] |
| ppo_v2_volpen (λ=2 net² penalty) | 9.3% | 0.70 | −33.6% | 0.28 | −0.22 [−0.48, +0.06] |
| **ppo_v2_boot (3 bootstrap paths)** | **10.6%** | **0.81** | −22.7% | 0.47 | **−0.11 [−0.36, +0.12]** |

Findings:

1. **Bootstrap-path training (WP-D) is the best RL arm so far**: +1.8pp CAGR
   and +0.09 Sharpe over v1, ΔSharpe gap to B&H roughly halved. Consistent
   with the literature (path diversity reduces single-history overfitting).
   Still: does not beat B&H, and the dumb vol-target baseline still dominates
   every RL arm on the risk-adjusted frontier. Seed dispersion did NOT
   shrink (0.35 vs 0.29) — the gain shows in the median policy, not variance.
2. **Vol-penalty arm failed**: reward −λ·net² raised average exposure to 0.70
   and worsened drawdown (−33.6%!) and Calmar. The quadratic penalty is too
   weak at daily scale (net² ~ 1e-4) relative to log-return differences —
   effectively a no-op that just changed the entropy landscape. Kill this
   shaping; a proper CVaR/drawdown-aware objective needs a different
   formulation (e.g., penalty on rolling drawdown in the state, or CPPO).
3. **The most important diagnostic: validation-test Sharpe correlation is
   NEGATIVE (≈ −0.15) for every config.** Validation-window performance does
   not predict test-window performance at all — model selection on val is
   noise (or slightly anti-signal). This is non-stationarity made concrete,
   and it caps what ANY selection-based improvement can deliver here. It also
   retroactively justifies using the median-val seed (not best-val) for
   reporting.

## Stylized-fact check for the bootstrap generator

Real (SPY train 1994–2007) vs synthetic (seed 0/1): ann. vol 17.3% vs
16.9–17.5%; kurtosis 3.7 vs 3.9–4.1; ACF(r²) at lags 1/5/21: 0.19/0.17/0.10 vs
0.19–0.21/0.16–0.18/0.04–0.09. Vol clustering partially preserved within
blocks (mean block 63 days), tails preserved. Good enough for WP-D; a
conditional generator (WP-E) would be needed to fix the lag-21 ACF decay.

## Where this leaves the study

- H1b (beat B&H CAGR): no arm is close; ΔCAGR CI for the best arm still
  excludes any positive value of note.
- H1a (match return, lower drawdown): bootstrap arm has B&H-like Calmar but
  −5pp CAGR; vol-target-10 still Pareto-dominates all RL arms. **The current
  honest verdict: a 10-line volatility-targeting rule beats every learned
  policy we've trained.**
- Next levers (in order): (a) give the agent the vol-target policy's
  information advantage explicitly — vol forecast features (WP-B, HAR-RV
  first); (b) continuous action space (vol-targeting is a continuous-exposure
  policy — the discrete {0,½,1} grid can't express it); (c) QQQ transfer
  (higher vol asset → timing worth more); (d) more bootstrap paths + longer
  training for the boot arm.
