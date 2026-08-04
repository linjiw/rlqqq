# RLQQQ v4 vs v8 deployment benchmark

Evaluated 2026-08-04 at `288e4278055e8a40ca8c9861207df063a021857c`.

## Deployment decision

**Winner: `ppo_v10_macro_frozen_2023_v1`.**

**Scope: Research and paper-trading signal; not qualified for capital deployment.**

Compare trained core policies (v4/v8 at the VT10/cap-1 budget; v10 at its own trained VT20/cap-1.5 budget with T-bill+50bp financing in the shared accounting). Treat a Sharpe difference within 0.02 as non-inferior, then prefer fewer features and lower turnover; a candidate that beats the band on Sharpe wins outright. Post-hoc composite overlays are ineligible.

## 2010-2025 walk-forward rerun

Decision dates 2010-02-05 through 2025-12-31; returns realized through 2026-01-02.

| Policy | CAGR | Sharpe | Max DD | Calmar | Avg exposure | Annual turnover |
|---|---:|---:|---:|---:|---:|---:|
| v8 core (eligible) | 12.90% | 1.052 | -11.49% | 1.123 | 0.63x | 12.03 |
| v4 core (eligible) | 13.19% | 1.047 | -12.21% | 1.081 | 0.64x | 13.70 |
| v8 composite (post-hoc dashboard convention) | 22.88% | 1.057 | -21.62% | 1.058 | 1.14x | 19.08 |
| v4 composite (post-hoc dashboard convention) | 23.45% | 1.055 | -23.14% | 1.013 | 1.17x | 21.12 |
| VT10 rule | 12.53% | 1.033 | -11.62% | 1.079 | 0.64x | 5.27 |
| VT20 rule | 22.22% | 1.041 | -21.86% | 1.017 | 1.17x | 6.62 |
| QQQ | 20.66% | 0.947 | -29.56% | 0.699 | 1.00x | 0.53 |

The composite rows above reproduce the archived dashboard's continuous-VT anchor. Exact fold-local actor-anchor variants are included in the JSON as `v4CompositeActorBaseline` and `v8CompositeActorBaseline`.

### One-close-lag sensitivity

| Policy | CAGR | Sharpe | Max DD | Calmar | Avg exposure | Annual turnover |
|---|---:|---:|---:|---:|---:|---:|
| v8 core (eligible) | 11.98% | 0.971 | -11.86% | 1.011 | 0.62x | 12.01 |
| v4 core (eligible) | 11.92% | 0.938 | -12.85% | 0.927 | 0.64x | 13.67 |
| v8 composite (post-hoc dashboard convention) | 21.18% | 0.983 | -24.24% | 0.874 | 1.14x | 19.05 |
| v4 composite (post-hoc dashboard convention) | 20.65% | 0.940 | -26.65% | 0.775 | 1.17x | 21.08 |
| VT10 rule | 12.03% | 0.988 | -11.89% | 1.012 | 0.64x | 5.26 |
| VT20 rule | 20.99% | 0.987 | -21.73% | 0.966 | 1.17x | 6.60 |
| QQQ | 20.66% | 0.947 | -29.56% | 0.699 | 1.00x | 0.53 |

## Frozen-policy 2026 replay

Decision dates 2026-01-02 through 2026-07-30; returns realized through 2026-07-31.

| Policy | YTD | July | Sharpe | Max DD | Avg exposure |
|---|---:|---:|---:|---:|---:|
| v8 core | 7.30% | -2.24% | 0.739 | -9.09% | 0.59x |
| v4 core | 6.60% | -2.91% | 0.695 | -7.47% | 0.54x |
| v8 composite | 12.45% | -4.83% | 0.794 | -17.23% | 1.17x |
| v4 composite | 10.45% | -6.14% | 0.697 | -15.10% | 1.08x |
| VT10 rule | 7.03% | -2.03% | 0.811 | -6.73% | 0.50x |
| VT20 rule | 11.42% | -4.39% | 0.809 | -13.72% | 1.00x |
| QQQ | 12.45% | -6.57% | 0.886 | -11.72% | 1.00x |

### Frozen 2026 one-close-lag sensitivity

| Policy | Total return | Sharpe | Max DD |
|---|---:|---:|---:|
| v8 core | 6.24% | 0.603 | -9.08% |
| v4 core | 7.47% | 0.820 | -7.23% |
| v8 composite | 10.75% | 0.688 | -16.96% |
| v4 composite | 12.09% | 0.812 | -14.66% |
| VT10 rule | 7.13% | 0.832 | -6.44% |
| VT20 rule | 11.65% | 0.830 | -13.17% |
| QQQ | 12.45% | 0.886 | -11.72% |

## Limitations

- The 2010-2025 folds were used during model research and are not a fresh holdout for v8 selection.
- The 2026 holdout was spent before the v8 ablation was selected; it is a forward sanity check, not an untouched selection set.
- Composite policies are post-hoc volatility-budget overlays and are not separately trained models.
- The backtest assumes a close-t decision can earn close-t to close-(t+1) returns; live execution after the close can differ.
- The one-close-lag sensitivity is conservative and is reported separately; it is not a simulation of next-open execution.
- The archived runs did not record their full training stack. requirements-research.txt now pins the current stack for future recipe reruns, but cannot make old PPO weights byte-identical.
- The recipe rerun did not numerically reproduce the archived headlines: published v4 core/composite CAGR was 13.19%/23.45% and published v8 core CAGR was 12.90%, versus 12.52%/22.45%/12.53% in this environment.
- The 0.005 same-close Sharpe edge does not establish statistical superiority; v8 is retained under the predeclared simplification and turnover rule.
- The final historical decision is dated 2025-12-31 but its close-to-close return realizes on 2026-01-02; excluding it does not change the ranking.
- The frozen artifact's 2023-12-31 cutoff is a decision-date cutoff: its last training feature row is 2023-12-29 and the associated final reward realizes on 2024-01-02.
