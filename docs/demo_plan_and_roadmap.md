# Demo Plan & Research Roadmap (2026-08-02)

## Part 1 — Demo plan

Audience story: "an RL tilt on a volatility-target rule, honestly evaluated."
Three demo artifacts, in order of impact:

1. **Interactive policy replay** (live at linjiw.github.io/rlqqq, data via
   `scripts/build_web_data.py`): animate 2010–2025 walk-forward decisions —
   vol base, learned tilt, wealth curves vs QQQ/SPY. Add (next iteration):
   era-holdout replay toggle (NDX 2000–2009) showing the −82% → −27%
   drawdown compression live.
2. **"What would it do today?" card**: daily decision snapshot (see
   `results/live_decision_snapshot.md` 2026-07-31: vol 24%, vt10 0.42,
   agent 0.35). Regenerate with one script call; ideal live-demo moment.
3. **One-slide honest scorecard** (from RESULTS.md): three eras × four
   policies (B&H / DCA / VT / agent), with the two significant results
   starred (era-holdout ΔSharpe vs B&H; none in-era) and the frontier
   context (published ceiling ≈ 1.0–1.1 Sharpe).

Demo script skeleton (15 min): problem + honest-evaluation machinery (3) →
replay walkthrough incl. one crisis episode (5) → live decision card (2) →
scorecard + what didn't work (scaling, LLM-era hype) (3) → roadmap (2).

## Part 2 — Validation protocol going forward

- **Forward paper-trading test (the only remaining unbiased data source):**
  freeze `ppo_v4_resid` ensemble + vt10 baseline TODAY; log its daily
  target exposure vs realized QQQ; review quarterly. Zero look-ahead by
  construction; ~2 years to a meaningful sample. Cheap and decisive.
- **Monthly data refresh + decision log**: append-only
  `results/forward_log.csv` (date, vol, baseline w, agent w, realized ret).
- **No more touching**: 2026 holdout spent; NDX era holdout now used twice
  (core + tilt-transfer) — any further use requires multiplicity accounting;
  VOO remains fully untouched as a spare validation asset.

## Part 3 — Ranked research directions (each with kill criterion)

1. **Calendar-feature ablation** (retrain v4 minus dow/month). Distillation
   flagged `month` as a decision driver — likely spurious. KILL: if Sharpe
   drops >0.02, calendar stays; else drop features permanently. (1 run.)
2. **Volume-panic gate formalization**: the distilled COVID behavior
   (don't add when vol_ratio spikes) suggests a 27-line rule:
   vt10 × (1.15 if vol>12% & price<MA50 & vol_ratio<1.25 else 1.0 ...).
   Build the distilled-rule policy, run through the full harness. If it
   matches the ensemble, we have a fully interpretable frontier policy —
   a stronger publishable claim than the RL result itself. KILL: rule
   underperforms ensemble by >0.03 Sharpe.
3. **Regime-stratified bootstrap** (last unexplored playbook item):
   HMM-posterior conditioning + within-regime block resampling. Expected
   gain on drawdown axis. KILL: no MaxDD improvement on 2 of 3 assets.
4. **DCA-beat significance push**: pool agent-vs-DCA deltas across QQQ/SPY/
   NDX-era with stratified bootstrap — the single most user-relevant claim
   ("beats DCA") may already be significant in the pooled test. (Analysis
   only, no training.)
5. **State-dependent leverage** (agent picks WHEN to lever): v7 showed
   log-wealth won't lever voluntarily → try reward = excess-over-B&H
   (relative objective) in the leveraged action space. KILL: doesn't beat
   tilt-transfer construction on both CAGR and MaxDD.
6. **Second-asset diversification of the CLAIM** (not the model): run the
   frozen recipe on VOO (untouched) as a pre-registered one-shot — third
   independent asset validation.

## Part 4 — Reading list (for depth, per "read books" advice)

- Grinold & Kahn, *Active Portfolio Management* — breadth/IC framing
  explains WHY single-asset timing caps at ~1.1 Sharpe (IR = IC·√BR: one
  asset, 250 low-IC decisions/yr).
- López de Prado, *Advances in Financial ML* — chapters 7–12 (CV, backtest
  statistics) underpin our protocol; ch. 8 feature importance methods for
  upgrading the distillation.
- Moreira & Muir 2017 + Cederburg et al. 2020 — the vol-managed literature
  our baseline (and its fragility caveats) comes from.
- Sutton & Barto ch. 9–13 for the function-approximation failure modes that
  motivated residual anchoring.
