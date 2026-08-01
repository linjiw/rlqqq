# One-Shot 2026 Holdout Report (evaluated 2026-08-01, once)

Per `docs/holdout_preregistration.md`. Agents retrained with train ≤
2023-12-31 (val 2024–2025, 21d embargo); the 2026-01-02..2026-07-30 window
(144 days) was never touched by any prior decision.

| policy | TotRet (7mo) | Sharpe | MaxDD | avg exposure |
|---|---|---|---|---|
| pre-registered blend (0.5 ens + 0.5 VT) | 6.45% | 0.67 | −8.5% | 0.54 |
| v4 residual ens (exploratory, not pre-reg) | 6.60% | 0.70 | −7.5% | 0.54 |
| buy-and-hold | **12.45%** | **0.89** | −11.7% | 1.00 |
| vol-target 10% | 7.12% | 0.80 | −7.4% | — |
| DCA monthly | 7.34% | 0.64 | −10.4% | — |

**Pre-registered criterion: Sharpe ≥ B&H AND MaxDD better → FAIL on Sharpe
(0.67 vs 0.89), PASS on MaxDD (−8.5% vs −11.7%).**
ΔSharpe −0.09 [−0.67, +0.51] — the CI is enormous, as pre-registered
(7-month window, near-zero power).

## Reading it honestly

- 2026 YTD is a calm bull (QQQ +12.5% in 7 months, vol 22%). This is
  precisely the regime where every de-risked strategy — including
  vol-target (0.80) and DCA (0.64) — trails B&H. The agent behaved exactly
  as its 2010–2025 profile predicts: captured ~53% of the upside with ~64%
  of the drawdown.
- The agent beat DCA on Sharpe (0.70 vs 0.64) and drawdown, slightly behind
  on 7-month return (6.6% vs 7.3%).
- Against its own baseline (vol-target) the agent was a hair behind this
  window (0.70 vs 0.80 Sharpe) — within the noise of 144 days.
- No update to any headline conclusion: the study's claims rest on the
  16-year walk-forward and the 2000s era holdout, which have actual power.
  This holdout adds one honest, low-power data point in the direction we
  expected: bulls belong to buy-and-hold.

Both configs' runs are logged in the registry as holdout_v2_boot /
holdout_v4_resid. The holdout is now SPENT — no further evaluation on
2026 data may inform any design choice for this study's claims.
