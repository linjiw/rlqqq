# Pre-registered holdout policy (frozen 2026-08-01)

To be evaluated ONCE on the 2026-01-01+ holdout at study end, per design_plan §7.

Policy: w_t = 0.5 * w_ensemble_t + 0.5 * w_voltarget_t
  - w_ensemble: mean exposure of ppo_v2_boot QQQ agents, seeds 0-14, trained
    per walk-forward protocol on F8 fold boundaries extended to 2025-12-31
    (train <= 2023-12-31, val 2024-2025, embargo 21d)
  - w_voltarget: min(1, 0.10 / realized_vol_21d), computed through close t
  - costs 2 bps one-way; T-bill on cash; discrete env accounting identity

Benchmarks: B&H QQQ total return; vol-target alone; DCA monthly.
Metrics: Sharpe, CAGR, MaxDD + paired stationary bootstrap on daily deltas.
Success criterion (H1a): Sharpe(policy) >= Sharpe(B&H) AND MaxDD better,
with the pre-registered caveat that one ~7-month window has near-zero power;
this is a directional sanity check, not a significance test.

Multiple-testing note: the 0.5/0.5 blend weight and the ensemble recipe were
chosen from test-fold evidence (logged in registry, N=644 trials at freeze
time). The holdout is the only untouched data. DSR accounting will use the
full registry N at evaluation time.
