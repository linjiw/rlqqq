# Deep-Dive: Model & Framework Settings Audit (2026-08-01)

## 1. What the v4 QQQ agent actually learned (state-conditional analysis)

Deviation of the 10-seed ensemble exposure from its vol-target baseline,
2010–2025 OOS, bucketed by trailing 21d vol tercile × 63d momentum sign
(ann_value = annualized return added by deviating):

| state | avg deviation | ann. value added | days |
|---|---|---|---|
| low vol, mom+ | −0.06 (de-risk) | −0.6% | 1232 |
| mid vol, mom+ | +0.05 | +0.5% | 1065 |
| mid vol, mom− | +0.08 | **+4.6%** | 193 |
| high vol, mom+ | +0.03 | +0.2% | 743 |
| high vol, mom− | +0.04 | **+5.2%** | 515 |

Total deviation value: **+0.8%/yr over pure vol-targeting.** The learned
signal is concentrated and interpretable: *in high-vol drawdown states the
agent adds exposure above what vol-targeting allows* — buying weakness when
vol-target is maximally de-risked (a contrarian/rebound tilt). Its one loss
bucket is de-risking in calm bulls (the FOMO tax). This is a real, novel,
learned behavior — not a re-derivation of the baseline (deviations >0.1 on
27% of days; the tilt survives out-of-era in the NDX 2000s test).

Seed disagreement remains large (max−min exposure spread >0.25 on 90% of
days) — ensembling is what converts noisy per-seed opinions into the stable
tilt; this is why mean-exposure works and single seeds don't.

## 2. Hyperparameter audit — what's right and what to probe

| setting | current | assessment |
|---|---|---|
| gamma 0.99 | horizon ≈ 100 trading days | reasonable for regime effects; 0.90 (10d) worth one probe — daily rewards are nearly i.i.d. and shorter credit assignment may cut variance |
| n_steps 512 × 4 envs | 2048-step buffer, 73 updates | small update count; 500k-step probe running |
| net 64×64 MLP | ~6k params on 26-dim state | right-sized for tabular-ish features; 256×256 probe queued (expect overfit) |
| ent_coef 0.005 fixed | no annealing | churn source in v1-v3; residual anchoring mostly fixed it (turnover 14) |
| episode_len 252 random windows | decorrelates rollouts | good; bootstrap paths add cross-path diversity |
| 3 bootstrap paths / 4 envs | 75% synthetic exposure | 7-path probe running |
| eval deterministic=True | argmax policy | correct (no residual exploration noise at eval) |
| data budget | each train row seen ~27× | RL-sample-inefficiency guard: bootstrap paths effectively multiply history ~4× |

## 3. Cash-rate bug found & fixed during this audit

`load_market` divided ^IRX by 10 (copied from the ^TNX convention). ^IRX
already quotes percent (verified vs historical 3m T-bill: 7.68 vs ~7.7% in
1990-06). Impact: cash earned ~0.4% instead of ~4% in high-rate years —
understated every cash-holding strategy (vol-target, agent) and overstated
B&H's relative edge pre-2009 and post-2022. All evaluations re-run;
qualitative conclusions unchanged (v4 still leads VT on QQQ: 1.047 vs 1.033
post-fix); era-holdout run used the corrected rate from the start.

## 4. Evaluation/validation dataset structure (answering the direct question)

Three independent lines of defense now exist:
1. **Walk-forward test folds** (2010–2025, 8 folds, embargoed) — used for
   development, honestly reported with IQM/CIs/registry.
2. **Era holdout** (NDX 2000–2009, 5 folds) — regime-opposite validation the
   recipe never touched: **agent beats B&H significantly (ΔSh +0.37
   [+0.09,+0.67])**, keeps its edge over vol-target. See
   `era_holdout_report.md`.
3. **2026 forward holdout** — still untouched, pre-registered policy frozen,
   to be evaluated exactly once at study end.

Plus the synthetic known-optimum validation (planted-signal/placebo tests)
guarding the training stack itself.
