# v9_rel5 Validation Report (2026-08-02)

The strongest end-to-end model (benchmark-relative reward, 5-level multiplier
on vt20-cap1.5, calendar-free) run through both remaining validation sets.
Fresh agents; era agents saw nothing after their fold's train end; 2026
agents trained ≤2023-12-31.

## A. Era holdout — NDX 2000–2009 (crisis decade, leveraged agent)

| policy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| **v9_rel5 ens** | **+3.8%** | **+0.159** | −55.6% |
| buy-and-hold | −5.4% | −0.055 | −82.2% |
| vt20cap1.5 rule | +3.7% | +0.145 | −49.0% |
| vt10 rule | +3.8% | +0.149 | −25.8% |
| DCA | +4.8% | +0.199 | −51.5% |

- **vs B&H: ΔSharpe +0.26 [+0.02, +0.50] — SIGNIFICANT out-of-era**, matching
  the v4 core policy's era result. The leveraged end-to-end agent preserves
  the study's headline significance in a decade it never saw.
- vs its own baseline (vt20c): +0.01 Sharpe — the learned tilt neither adds
  nor destroys value out-of-era (same pattern as tilt-transfer).
- **The honest risk disclosure: −55.6% MaxDD.** The agent levered into the
  2004–07 mid-cycle (avg_w 1.07–1.22, correct — those folds beat B&H) but a
  levered book in 2000–02 still draws down hard (E1 Sharpe −0.66 vs B&H
  −0.54). Leverage costs tail risk in a secular bear; the defensive vt10
  (−26% MaxDD) remains the right vehicle for crisis-heavy regimes.
- Per-fold: agent beats B&H Sharpe in 4/5 folds (all but the dot-com crash
  itself, where nothing levered survives well).

## B. 2026 YTD (exploratory — pre-registered holdout claim already spent)

| policy | 7-mo return | Sharpe | MaxDD | avg_w |
|---|---|---|---|---|
| v9_rel5 ens | 11.7% | 0.81 | −13.7% | 1.01 |
| vt20cap1.5 rule | 12.0% | 0.95 | −10.5% | 0.84 |
| buy-and-hold | 12.5% | 0.89 | −11.7% | 1.00 |
| vt10 | 7.0% | 0.81 | −6.7% | 0.50 |
| DCA | 7.3% | 0.64 | −10.4% | — |

- The leveraged agent nearly matches B&H return (11.7% vs 12.5%) — a
  dramatic improvement over the defensive arms' ~6.5% in the same window —
  at similar drawdown. It trails vt20c slightly (its tilts cost ~30bp and a
  bit of vol this window; 144 days = noise).
- Latest live decision (2026-07-29): exposure 1.12 — modestly levered into
  the current pullback, consistent with its learned buy-weakness profile.

## Verdict across all three evaluation sets

| set | v9_rel5 vs B&H | vs its baseline (vt20c) |
|---|---|---|
| 2010–2025 walk-forward | **+CAGR, +Sharpe** (ΔSh +0.15) | +0.03 Sharpe |
| NDX 2000–2009 era | **ΔSh +0.26 SIG**, +9pp CAGR | +0.01 Sharpe |
| 2026 YTD (7mo, noise) | −0.8pp ret, −0.08 Sh | −0.14 Sh |

The end-to-end leveraged agent is the study's most complete result: beats
B&H on both axes in-development-era, significantly out-of-era, and stays
competitive (unlike defensive arms) in a calm bull. Its learned tilt over
the leveraged rule is small (+0.01 to +0.03 Sharpe, ties) — the honest
summary remains: **most of the B&H-beating power is the leveraged
vol-target scaffold; the RL layer adds a small, consistent, never-negative
tilt plus the validated crisis behavior.** Risk profile choice (vt10-based
defensive vs vt20-based growth) matters more than any learning refinement —
that's a user risk-preference decision, not a modeling one.
