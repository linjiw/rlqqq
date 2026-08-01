# Pilot Report: PPO v1 on SPY (2026-07-31)

**Setup:** PPO (SB3 2.9, MLP 64x64, FinRL-seeded hyperparameters), discrete
exposure {0, 0.5, 1}, 2 bps one-way costs, T-bill interest on cash, 150k steps,
random 252-day training episodes. 8 anchored walk-forward folds (test windows
2010–2025, 21-day embargo), 10 seeds per fold = 80 runs, ~16 min wall on 8 CPU
cores. Every run logged to `registry.jsonl`. Env verified to reproduce
buy-and-hold exactly under an always-long policy (`tests/test_env.py`).

## Headline result

**The pilot PPO agent does NOT beat buy-and-hold — consistent with the
literature's null.**

| | CAGR | Sharpe | MaxDD | Sortino |
|---|---|---|---|---|
| agent (median-val seed/fold, concatenated OOS) | 8.7% | 0.72 | −17.9% | 0.79 |
| buy-and-hold | 15.7% | 0.92 | −33.7% | 1.15 |
| MA200 rule | 10.6% | 0.90 | −20.7% | 1.06 |
| vol-target 10% | 10.6% | 1.04 | −12.2% | 1.37 |

- Fold-level: agent IQM Sharpe > B&H in **1 of 8 folds** (F7, the 2022 bear —
  matching the literature's "RL adds value in bear regimes" pattern).
- Paired stationary bootstrap (10k reps): ΔSharpe −0.20, 95% CI [−0.48, +0.04]
  (not significant); ΔCAGR −6.9pp, CI [−0.120, −0.018] (**significantly
  worse**).
- Exposure decomposition: avg exposure 0.54; annualized alpha +0.5% with
  t(NW) = 0.29 → the agent's return is ~fully explained by partial passive
  exposure. No timing skill detected.
- DSR = 0.03 (N=80 trials): nowhere near the 0.95 credibility bar.
- Seed spread is large (pooled test Sharpe min −0.08, max 2.25) — confirms
  Henderson-style seed variance; single-seed papers would cherry-pick 2.25.

## Interpretation

1. The harness works end-to-end and the honest-evaluation machinery
   (registry, IQM, paired bootstrap, decomposition) produces exactly the kind
   of diagnosis it was designed for.
2. The v1 agent behaves like a noisy 54%-exposure index fund with 50x/yr
   turnover — it hasn't learned timing, it has learned "be partially long."
   Even its Sharpe deficit vs B&H is mostly the drag of cash + churn.
3. It does beat B&H on MaxDD (−18% vs −34%) but so do the dumb baselines
   (vol-target: −12% at higher Sharpe) — H1a is NOT satisfied either, because
   vol-targeting dominates the agent on the risk-adjusted frontier.
4. **Next lever candidates** (in planned order): vol-penalized/drawdown-aware
   reward shaping; more context features (VIX regime already in state — check
   feature importances); longer training with entropy schedule; CPPO-style
   risk-sensitive objective; WP-D block-bootstrap path training (the highest-
   evidence robustness lever from the lit review); QQQ transfer.

## Reproduce

```bash
.venv/bin/python scripts/run_pilot.py --seeds 10 --timesteps 150000
.venv/bin/python scripts/analyze_pilot.py --config ppo_v1 --symbol SPY
```
