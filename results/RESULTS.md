# Consolidated Results Ledger

Updated 2026-08-01. All numbers: concatenated walk-forward OOS 2010–2025,
median-validation seed per fold, 2 bps one-way costs, T-bill on cash.
Full per-run data: `registry.jsonl` (564 trials logged so far).

## SPY

| strategy | CAGR | Sharpe | MaxDD | Calmar | vs B&H ΔSharpe [CI] |
|---|---|---|---|---|---|
| buy-and-hold | 15.7% | 0.92 | −33.7% | 0.47 | — |
| vol-target 10% | 10.6% | 1.04 | −12.2% | 0.87 | — |
| MA200 | 10.6% | 0.90 | −20.7% | 0.51 | — |
| ppo_v1 | 8.8% | 0.72 | −17.9% | 0.49 | −0.20 [−0.47,+0.04] |
| ppo_v2_volpen (killed) | 9.3% | 0.70 | −33.6% | 0.28 | −0.22 |
| **ppo_v2_boot** | 10.6% | 0.81 | −22.7% | 0.47 | −0.11 [−0.36,+0.12] |
| ppo_v3_harv (killed) | 9.0% | 0.75 | −21.5% | 0.42 | −0.17 |
| ppo_v3_cont_harv (killed) | 8.1% | 0.72 | −25.4% | 0.32 | −0.20 |

## QQQ (transfer of best config, no re-tuning)

| strategy | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| buy-and-hold | 20.7% | 1.01 | −29.6% | 0.70 |
| vol-target 10% | 12.0% | 1.10 | −11.9% | 1.01 |
| **ppo_v2_boot** | **15.1%** | **1.03** | **−19.9%** | **0.76** |

QQQ diagnostics: ΔSharpe vs B&H **+0.03** [−0.28, +0.34]; ΔCAGR −5.6pp
[−0.129, +0.013]; ann. alpha +3.2% (t_NW = 1.54, not significant);
DSR = 0.013 (N=484). Seed IQM Sharpe 0.97 [0.86, 1.08].

## Standing conclusions

1. **H1b (beat B&H CAGR after costs): NO** on both symbols, decisively.
2. **H1a (match risk-adjusted, cut drawdown): partially, on QQQ only** —
   the agent matches B&H Sharpe (1.03 vs 1.01) at 2/3 the drawdown and
   higher Calmar (0.76 vs 0.70). But it is **not** statistically
   distinguishable (CI spans zero), alpha t=1.54 < 3 bar, DSR ≈ 0, and the
   10-line vol-target rule still delivers a better Sharpe (1.10) and far
   better drawdown (−12%) with no learning at all.
3. **Ablation verdicts:** bootstrap-path training (WP-D) = only arm that
   helped, now default. Vol-penalty reward, HAR features, continuous
   actions = all hurt or neutral, killed with logged evidence.
4. **Validation-test Sharpe correlation ≈ −0.15 across all configs** —
   non-stationarity makes val-window selection useless. This is the binding
   constraint on any further "smarter selection" work.
5. Stack validity is established (planted-signal + placebo tests pass), so
   these nulls are informative about markets, not about broken code.

## Ensemble & combination mining (2026-08-01, from saved series)

- **Mean-exposure seed ensemble** (k=10-15) is a free win: QQQ Sharpe
  1.03→1.09, MaxDD −20%→−16%, turnover −25%. Ensemble curve saturates
  ~k=10-15. Averaging > selection (val-selection is uninformative anyway).
- Agent exposure is nearly orthogonal to vol-targeting (corr 0.15) → they
  combine: **0.5·agent + 0.5·vol-target on QQQ: Sharpe 1.143, CAGR 13.2%,
  MaxDD −13.8%, turnover 17.7** — better Sharpe than vol-target (1.103),
  better CAGR, vs B&H ΔSharpe +0.14 [−0.05, +0.33]. Still not significant,
  and combination rules were explored on test data (multiple-testing caveat
  logged; needs 2026-holdout confirmation).
- SPY: agent adds nothing over vol-target (blend 1.00 vs vt 1.04). The QQQ
  edge does not transfer to SPY — consistent with QQQ's higher vol making
  timing more valuable.
- Cost sensitivity: agent break-even vs B&H ≈ 4-5 bps; at retail 0.5 bps
  agent ≈ vol-target Sharpe.
- Regime: agent beats B&H Sharpe in 5/8 folds — all four high-vol folds;
  loses only in the calm 2012–2017 bulls. Every fold: agent MaxDD < B&H.

## v4: residual-on-vol-target agents (2026-08-01)

Playbook items 1-3 implemented: actions = multipliers {0.5, 1.0, 1.5} on a
causal vol-target baseline (action 1.0 ≡ baseline, verified by test), 5bp
training-only switch penalty, mean-exposure seed ensembles. QQQ, 8 folds ×
10 seeds each:

| policy | CAGR | Sharpe | MaxDD | Calmar | turnover |
|---|---|---|---|---|---|
| buy-and-hold | 20.7% | 1.007 | −29.6% | 0.70 | — |
| vol-target 10% | 12.0% | 1.103 | −11.9% | 1.01 | 5 |
| ppo_v2_boot ens | 13.5% | 1.064 | −14.9% | 0.90 | 33 |
| **ppo_v4_resid ens** | **12.7%** | **1.117** | **−12.7%** | 1.00 | **14** |
| ppo_v4_resid_nosp ens | 12.3% | 1.112 | −11.7% | 1.05 | 15 |

- **First arm to edge past vol-targeting on BOTH Sharpe (1.117 vs 1.103) and
  CAGR (12.7% vs 12.0%)** — ΔSharpe vs VT +0.01 [−0.07,+0.10], i.e. a
  statistical tie, but the agent now sits ON the published frontier for this
  asset class (research verdict: no credible daily single-index result
  >1.1 net Sharpe post-2010 exists).
- vs B&H: ΔSharpe +0.12 [−0.09,+0.32]; CAGR gap −8pp remains (structural:
  avg exposure 0.64).
- Turnover collapsed 33→14 (residual anchoring + switch penalty); the
  switch penalty itself added little beyond the anchoring (nosp ≈ same).
- Anchoring converted the seed lottery into "baseline ± small learned
  deviation" — worst case is now the baseline, as designed.

### v4 robustness checks

- **5 bps costs (QQQ):** v4_resid Sharpe 1.080 vs vol-target 1.088 — the
  edge over VT vanishes at 2.5x costs but the agent stays at the frontier;
  v2_boot degrades much faster (0.99). Low turnover = cost robustness.
- **SPY transfer:** v4_resid ens Sharpe 1.012 vs VT 1.036 (ΔSh −0.02
  [−0.10,+0.06]) — on SPY the residual agent matches but does not exceed
  the baseline (unlike QQQ). Pattern repeats: learned timing adds value on
  the higher-vol asset only.

## v5 scaling sweep + era holdout + audit (2026-08-01)

**Cash-rate bug fixed** (^IRX was /10 → T-bill understated 10x; all evals
re-run, conclusions unchanged, numbers shifted slightly).

**Scaling sweep (QQQ, each 8 folds × 10 seeds on the v4 recipe):**

| variant | Sharpe | CAGR | MaxDD | verdict |
|---|---|---|---|---|
| ppo_v4_resid (150k, 64×64, 3 paths) | 1.047 | 13.2% | −12.2% | reference |
| + LAWA checkpoint averaging (k=8) | 1.032 | 12.6% | −11.7% | no gain |
| + 7 bootstrap paths | 1.031 | 12.8% | −11.6% | no gain |
| + 500k steps (3.3× compute) | 1.044 | 13.0% | −12.6% | no gain |

**Scaling is saturated.** The recipe converged at 150k steps / 64×64 / 3
paths; the constraint is signal, not capacity or compute — consistent with
the frontier verdict from the literature research.

**Era-holdout validation (the big result — see `era_holdout_report.md`):**
frozen recipe on NDX with 2000–2009 test folds (dot-com + GFC, never used
in any design decision): agent +3.9% CAGR / −27% MaxDD vs B&H −5.4% / −82%.
**ΔSharpe vs B&H +0.37 [+0.09, +0.67] — the study's first statistically
significant beat of buy-and-hold.** Rank over vol-target preserved (+0.03).

**DCA scorecard (2010–2025, matched windows):** agent beats DCA on CAGR
(QQQ +0.8pp, SPY +1.1pp) and drawdown, ties on Sharpe — consistent
directional edge, not yet significant. In 2000–2009 DCA is the strongest
passive benchmark (Sharpe 0.20 vs agent 0.16) though with −51% vs −27% DD.

**What the agent learned (deep-dive, `deep_dive_report.md`):** an
interpretable contrarian tilt — add exposure above vol-target in high-vol
drawdown states (+5%/yr in those states), pay a small FOMO tax in calm
bulls. Deviation from baseline >0.1 on 27% of days; +0.8%/yr total value
over pure vol-targeting; survives out-of-era.

## 2026 holdout (SPENT) + leverage extension (2026-08-01)

**One-shot 2026 holdout** (`holdout_2026_report.md`): pre-registered blend on
QQQ 2026 YTD (calm bull, +12.5%): FAILs the Sharpe criterion (0.67 vs B&H
0.89), PASSes MaxDD (−8.5% vs −11.7%). Beat DCA on Sharpe. Behaved exactly
as the regime profile predicts; near-zero power as pre-registered. Holdout
is now spent.

**Leverage extension (clearly labeled, not core study).** Cap 1.5×,
financing at T-bill+50bp, QQQ 2010–2025:

| policy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| buy-and-hold | 20.7% | 0.947 | −29.6% |
| **vt20 rule, cap 1.5 (no learning!)** | **22.2%** | **1.041** | **−21.9%** |
| agent residual on vt10, cap 1.5 | 13.3% | 1.040 | −12.0% |
| agent residual on vt20, cap 1.5 | 19.1% | 1.010 | −21.8% |

Three lessons:
1. **Leveraged vol-targeting alone beats B&H on BOTH CAGR and Sharpe**
   (22.2%/1.04 vs 20.7%/0.95) — the Moreira-Muir "vol-managed portfolios"
   result reproduced on QQQ with honest financing costs. If "beat
   buy-and-hold" is the goal and modest leverage is allowed, a 5-line rule
   does it; dCAGR vs B&H not yet significance-tested but Sharpe edge is
   consistent across the frontier (targets 10%→24% all >1.0).
2. The agent's learned tilt adds value on the *defensive* baseline
   (vt10+cap1.5: +0.03 Sh, +0.6pp CAGR over its fair baseline) but
   SUBTRACTS on the aggressive one (vt20: −0.02 Sh, −3.1pp CAGR) — the
   contrarian buy-weakness signal lives in low-exposure states; when the
   baseline is already near-full exposure there is nothing to add.
3. The CAGR gap to B&H was never about intelligence — it's an exposure
   budget. Leverage closes it mechanically; learning doesn't.

## v7: dataset scaling + leverage-in-RL (2026-08-01)

**Tilt-transfer (free, from saved series):** applying the v4 agents' learned
multiplier to the leveraged vt20-cap1.5 baseline gives **CAGR 23.4% /
Sharpe 1.055 vs B&H 20.7% / 0.947** — the best CAGR+Sharpe combination in
the study (ΔSh +0.11 [−0.08,+0.29], ΔCAGR +2.8pp [−1.5,+7.0], not sig;
post-hoc construction, caveat logged). See `live_decision_snapshot.md`.

**Dataset scaling (pooled SPY+NDX+GSPC training, 2–4× data incl. pre-1999
regimes):** ppo_v7_pool Sharpe 1.011 vs v4's 1.047 (ΔSh −0.03 [−0.11,+0.05],
ΔCAGR −0.9pp, borderline significantly *negative*). **Pooling cross-asset
data does not help and slightly hurts** — QQQ-specific training data wins;
cross-asset regularization was already provided by bootstrap paths.
Combined with v5 (compute/model scaling neutral): **both scaling axes are
now confirmed saturated. The binding constraint is signal, not scale.**

**Leverage inside RL (ppo_v7_pool_lever, multiplier cap 1.5 on vt10):** the
trained agent stays defensive (avg_w 0.61, barely uses the leverage
headroom) — Sharpe 0.966, worse than v4. The log-wealth objective on
bootstrap-diversified paths (which include crash paths) rationally declines
to lever a defensive baseline. End-to-end leverage learning underperforms
the simple transfer construction above.

**Live decision snapshot (2026-07-31, QQQ at 24% vol, −7.7% off peak):**
vt10 rule → 0.42 exposure; vt20-cap1.5 rule → 0.84; v4 agent ensemble →
0.40 (0.95× its baseline — mildly defensive; corrected after exact actor
export exposed the original one-day action offset). Its contrarian
add-exposure signal triggers in deeper drawdowns than this.

## Model understanding + DCA granularity finding (2026-08-02)

**Distillation** (`model_understanding_report.md`): the ensemble's decision
is 45% linearly explainable. Core logic: *add exposure above vol-target when
vol is high and price below trend BUT volume isn't panicking; shade down in
calm overbought markets.* COVID crash is the exception that proves the
volume-gate: the agent CUT below baseline during panic liquidation (mult
0.92) while ADDing in all 8 grinding drawdowns (mult 1.08–1.16). Red flag:
`month` appears in the distilled tree — calendar ablation queued.

**Tilt-transfer validated out-of-era**: multiplier × vt20cap1.5 on NDX
2000–2009: ΔSharpe vs B&H **+0.26 [+0.02, +0.51] SIG**; ≈ neutral vs the
leveraged rule; −51% MaxDD discloses the leverage tail-risk cost.

**DCA granularity finding (honest correction of emphasis):** whether the
agent "beats DCA" depends on the DCA horizon. Against a LONG-horizon DCA
(16-year continuous drip, which converges to B&H): agent wins CAGR both
symbols (+0.8/+1.1pp), ties Sharpe. Against SHORT-horizon DCA (fresh 2-year
ramp per fold, avg exposure ~0.5, very defensive): pooled across 21
asset-fold segments the agent is **significantly WORSE on Sharpe (−0.16
[−0.29, −0.02])**, tie on CAGR. A 2-year DCA ramp is a surprisingly strong
defensive benchmark — stronger than the literature (which benchmarks
lump-sum) acknowledges. Claim to carry forward: "agent beats long-horizon
DCA on CAGR at equal risk; short-horizon DCA remains unbeaten on
risk-adjusted terms, like everything else on the frontier."

## v8: calendar ablation — distillation-driven fix confirmed (2026-08-02)

The distilled tree flagged `month` as a decision driver (likely spurious
seasonality). Retrained without dow/month (ppo_v8_nocal, 8 folds × 10
seeds): **Sharpe 1.052 (vs 1.047), MaxDD −11.5% (vs −12.2%), Calmar 1.123
(best of any arm), turnover 11.9 (vs 13.6).** Kill criterion said drop if
no >0.02 Sharpe loss — ablation actually improved everything. **Calendar
features permanently removed; ppo_v8_nocal is the new reference policy.**
This closes the loop: interpretability analysis → hypothesis → ablation →
cleaner model. The interpretation-driven workflow works.

## Honest bottom line (as of v3)

Everything found so far is consistent with the literature synthesis: RL
delivers *risk management*, not *alpha*, on liquid US index products — and
even its risk management is currently matched by trivial vol-targeting. The
remaining planned arms with genuine upside: transformer embeddings (WP-A,
Kronos finance-pretrained), longer-horizon training + more bootstrap paths,
CPPO-style proper risk objective, and the FinRL comparison arm for the paper.
