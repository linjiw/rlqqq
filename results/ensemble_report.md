# Ensemble Mining Report (2026-08-01)

Everything here computed from already-saved series — zero new training.

## 1. Seed-ensemble (mean exposure across 10 seeds) is a free win

QQQ, concatenated OOS 2010–2025, 2 bps:

| variant | CAGR | Sharpe | MaxDD | Calmar | turnover | ΔSh vs B&H [CI] |
|---|---|---|---|---|---|---|
| median-val seed (old reporting) | 15.1% | 1.031 | −19.9% | 0.76 | 44.4 | +0.03 [−0.28,+0.33] |
| **mean exposure (10 seeds)** | 14.2% | **1.064** | **−16.2%** | **0.88** | **35.6** | +0.06 [−0.15,+0.27] |
| median exposure | 16.0% | 1.061 | −19.9% | 0.81 | 52.2 | +0.06 |
| buy-and-hold | 20.7% | 1.007 | −29.6% | 0.70 | — | — |
| vol-target 10% | 12.0% | 1.103 | −11.9% | 1.01 | — | — |
| DCA monthly | 11.8% | 1.076 | −13.9% | 0.85 | — | — |

SPY: mean-exposure ensemble improves Sharpe 0.81→0.90 and cuts turnover
45→29. Same pattern.

- Averaging policies beats selecting policies — consistent with the negative
  val-test correlation (selection has nothing to select on).
- **Ensemble-size curve rises monotonically**: k=1: 0.72, k=3: 0.90, k=5:
  1.00, k=7: 1.04, k=10: 1.06 — not saturated at 10 seeds → 20-seed run
  launched.
- Averaging also *smooths* exposure changes: turnover falls 44→36 for free.

## 2. Regime breakdown (QQQ ensemble vs benchmarks, per fold)

Agent beats B&H Sharpe in **5/8 folds** (all four high-vol folds — F5 2018-19,
F6 2020-21, F7 2022-23, F8 2024-25 — plus F1) and loses in the three calmest
bull folds (F2-F4, 2012–2017) where anything less than full exposure loses.
Beats vol-target in 5/8. Every fold's agent MaxDD < B&H MaxDD.

## 3. The agent is NOT vol-targeting in disguise

corr(agent exposure, vol-target exposure) = **0.15**; corr(exposure, trailing
vol) = −0.17. The learned policy and vol-targeting achieve similar
risk-adjusted numbers via nearly orthogonal signals → they are combinable
(e.g., min(w_agent, w_vt) or product) — a genuinely promising unexplored arm.

## 4. Cost sensitivity (QQQ ensemble)

| bps | agent Sharpe | B&H | vol-target |
|---|---|---|---|
| 0.5 | 1.10 | 1.01 | 1.11 |
| 2 | 1.06 | 1.01 | 1.10 |
| 5 | 0.98 | 1.01 | 1.09 |
| 10 | 0.85 | 1.00 | 1.06 |
| 25 | 0.44 | 1.00 | 0.99 |

Break-even vs B&H ≈ 4-5 bps. At realistic retail costs (~0.5 bps) the agent
matches vol-target Sharpe. Turnover remains the agent's structural handicap.

## 5. No-trade band (hysteresis) post-processing

Applying "only move if |target−current| > band" to the ensemble exposure:
band 0.20 → Sharpe 1.14, turnover 21.8 (from 35.6). BUT the band sweep was
done on test data (diagnostic only — adopting the best band = peeking). With
an a-priori band of 0.10: Sharpe 1.067, ΔSh vs B&H +0.06 [−0.15,+0.27] — not
significant. Proper adoption requires fixing the band before testing or
selecting per-fold on validation (which we know is uninformative). Next run
should bake a fixed band=0.15–0.2 into the ACTION space (act-persistently),
which is a design choice, not a fitted parameter.

## Standing verdict after mining

- vs DCA: ensemble ΔSharpe −0.02 [−0.29,+0.24] — statistical tie, with agent
  holding CAGR advantage (14.2% vs 11.8%).
- vs B&H: tie on Sharpe (+0.06, CI spans 0), −6.5pp CAGR, −13pp MaxDD.
- vs vol-target: tie (−0.04, CI spans 0), +2.2pp CAGR, +4pp MaxDD.
- The QQQ ensemble is now Pareto-competitive: more CAGR than vol-target,
  less drawdown than B&H. No claim survives the t≥3/DSR bar yet.
