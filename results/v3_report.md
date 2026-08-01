# v3 Report: HAR-RV Features & Continuous Actions (2026-08-01)

## Setup

Both arms build on the best v2 recipe (3 bootstrap paths):
- `ppo_v3_harv`: discrete {0,½,1} + HAR-RV features (expanding-window Corsi
  HAR forecast of next-21d realized vol, refit monthly, no look-ahead;
  forecast-vs-realized corr 0.52 on 2010+ — genuinely informative).
- `ppo_v3_cont_harv`: continuous Box[0,1] action + HAR features.

8 folds × 10 seeds × 150k steps each, 2 bps.

## Results (concatenated OOS 2010–2025, median-val seed per fold)

| config | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| buy-and-hold | 15.7% | 0.92 | −33.7% | 0.47 |
| vol-target 10% | 10.6% | 1.04 | −12.2% | 0.87 |
| **ppo_v2_boot (still best)** | **10.6%** | **0.81** | −22.7% | 0.47 |
| ppo_v3_harv | 9.0% | 0.75 | −21.5% | 0.42 |
| ppo_v3_cont_harv | 8.1% | 0.72 | −25.4% | 0.32 |

## Findings

1. **HAR vol-forecast features did not help** (WP-B partial kill). The
   forecast is real (corr 0.52 with future RV) but the agent already had
   vol_5/21/63d and VIX in state — the marginal information was small, and
   two extra features added estimation noise. WP-B's stronger variant
   (TSFM vol forecasts) now looks low-value: the cheap upper bound on
   "better vol info" barely moved anything.
2. **Continuous actions made things worse**, not better. The hypothesis was
   that vol-targeting needs continuous exposure; in practice the Gaussian
   policy under-commits (avg exposure 0.44) and wanders. With ~250
   decisions/year of weak signal, the coarser action grid acts as
   regularization. Keep discrete.
3. `ppo_v2_boot` (bootstrap paths, discrete, plain features) remains the
   best RL arm. Nothing beats the dumb vol-target baseline risk-adjusted;
   nothing approaches B&H on CAGR.

## Kill-criteria log (per investigation plan)

- WP-B (vol features): HAR arm killed after honest test. TSFM vol arm
  deprioritized (cheap proxy showed no headroom).
- Continuous-action variant: killed.
- WP-D (bootstrap): confirmed as default recipe for all future arms.

Next: QQQ transfer of ppo_v2_boot (running), then consolidated report.
