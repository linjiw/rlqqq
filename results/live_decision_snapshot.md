# Live Decision Snapshot — QQQ as of 2026-07-31

Market state: QQQ 687.99 (adj), trailing 21d realized vol **23.9%** (63d:
25.5%), 1-month return **−5.1%**, 3-month +3.1%, **−7.7% below its peak**.
A mild-correction / elevated-vol state.

| policy | today's target exposure | reading |
|---|---|---|
| vt10 (core defensive rule) | **0.42** | vol is 2.4x the 10% target → less than half invested |
| vt20 cap 1.5 (leveraged rule) | **0.84** | below full — the leverage rule de-levers in this vol |
| v4 residual ensemble (10 agents, trained ≤2023) | **0.35** (seed range 0.21–0.62) | 0.84x multiplier on its vt10 baseline — mildly *below* baseline |

Note the agent's behavior vs its crisis profile: the learned tilt adds
exposure in *deep* high-vol drawdowns (2008/2020/2022-like states); at a
−7.7% drawdown with 24% vol it stays slightly defensive instead — the
contrarian signal hasn't triggered. Recent trajectory (July 20–31): agent
held ~0.50, cut to 0.35 after the late-July vol uptick, tracking but not
mirroring the vt10 baseline (0.39→0.42).

(Reminder: research snapshot, not investment advice; agents were trained
through 2023 for holdout purity.)

# Tilt-transfer: learned signal × leveraged baseline (free experiment)

The v4 agents' multiplier (agent_w / vt10, clipped [0.5, 1.5]) applied to
the vt20-cap1.5 leveraged baseline, QQQ 2010–2025, 2 bps, financing at
T-bill+50bp:

| policy | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| buy-and-hold | 20.7% | 0.947 | −29.6% | 0.70 |
| vt20 cap1.5 rule | 22.2% | 1.041 | −21.9% | 1.02 |
| **tilt × vt20 cap1.5** | **23.4%** | **1.055** | −23.1% | 1.01 |

vs B&H: ΔSharpe +0.108 [−0.076, +0.290], ΔCAGR +2.8pp [−1.5, +7.0] — the
strongest CAGR+Sharpe combination in the study so far, though still not
statistically significant. vs the leveraged rule itself: +0.011 Sharpe,
+1.2pp CAGR — the learned tilt survives transplantation onto a baseline it
was never trained with (it was learned on vt10). Multiple-testing caveat:
this transfer rule was constructed post-hoc from test-fold series; the
end-to-end trained version (ppo_v7_pool_lever) is the honest test.
