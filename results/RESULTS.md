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

## Honest bottom line (as of v3)

Everything found so far is consistent with the literature synthesis: RL
delivers *risk management*, not *alpha*, on liquid US index products — and
even its risk management is currently matched by trivial vol-targeting. The
remaining planned arms with genuine upside: transformer embeddings (WP-A,
Kronos finance-pretrained), longer-horizon training + more bootstrap paths,
CPPO-style proper risk objective, and the FinRL comparison arm for the paper.
