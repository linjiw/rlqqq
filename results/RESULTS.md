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

## Honest bottom line (as of v3)

Everything found so far is consistent with the literature synthesis: RL
delivers *risk management*, not *alpha*, on liquid US index products — and
even its risk management is currently matched by trivial vol-targeting. The
remaining planned arms with genuine upside: transformer embeddings (WP-A,
Kronos finance-pretrained), longer-horizon training + more bootstrap paths,
CPPO-style proper risk objective, and the FinRL comparison arm for the paper.
