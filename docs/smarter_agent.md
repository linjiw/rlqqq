# What "Smarter" Means for Our RL Agent (2026-08-02)

Grounded in 1,600+ logged trials, three validation eras, and the verified
literature. "Smarter" decomposes into four distinct capabilities — our
evidence says they are NOT equally improvable.

## 1. Smarter = better conditioning information (MOST PROMISING)

The agent can only be as good as what it can see. Everything it learned —
the vol-state/trend/volume-panic tilt — is a function of its 22 features.
Evidence this axis has headroom:
- Its one clear failure mode (levering into 2000–02, era fold E1) is a
  regime a *macro-aware* observer arguably saw coming (inverted curve 2000,
  collapsing breadth) but our price-only state cannot represent.
- The literature's most replicated positive mechanism is regime
  conditioning; our state has only VIX + term spread as non-price context.
- Cross-asset signals (SPX/QQQ relative strength, bond trend, gold trend,
  credit conditions) are the standard economist's dashboard for risk
  appetite — none is in the state. (v10 arm tests exactly this;
  research agent verifying which have documented value.)
Caution from our own history: HAR features failed (redundant), calendar
features were actively spurious. Rule: every candidate feature needs an
economic mechanism + an IC screen before entering the state, and an
ablation after.

## 2. Smarter = better objectives (PROVEN AXIS, largely harvested)

The v9 breakthrough was not more capacity — it was the reward. Log-wealth
→ benchmark-relative advantage converted "never lever" into "lever except
into crashes." Remaining headroom here: the quadratic-variation penalty
(fractional-Kelly dial) and possibly CVaR-constrained variants for the
defensive arm. But diminishing: we've now aligned the objective with the
economic goal; further reward surgery risks reward-hacking territory.

## 3. Smarter = better learning machinery (EXHAUSTED — evidence strong)

More compute (3.3x), bigger nets (16x params), more data (4x pooled), more
bootstrap paths, checkpoint averaging: all neutral or negative. The
constraint is signal, not capacity. Any pitch that "a bigger/fancier model
will be smarter here" is contradicted by our own sweeps AND the frontier
literature (transformers ≈ linear on daily returns; TSFM zero-shot fails).
Do not spend here again until the information set changes (axis 1).

## 4. Smarter = knowing when not to act (VALIDATED, structural)

The residual architecture IS this: bounded deviations from a sane default,
worst case ≈ baseline. The distillation shows the agent's value is
concentrated in ~700 high-vol days; elsewhere it correctly does nothing.
"Smartness" in a low-signal domain looks like selective, small,
well-timed interventions — not constant cleverness. This reframes the
research question: not "make the agent smarter everywhere" but "expand the
set of states where it can recognize an intervention is warranted" —
which is again axis 1.

## The economics framing (per the user's instinct)

The efficient-market baseline says price history is mostly exhausted;
what's left lives in (a) risk premia that vary with economic conditions
(credit, term structure, variance risk premium), (b) slow institutional
flows (vol-targeting funds, rebalancing), (c) crisis microstructure
(forced selling → our buy-weakness tilt). Our agent already harvests (c)
and a bit of (a) via VIX. The v10 cross-asset state is the systematic test
of how much more of (a) is reachable. SPX specifically: QQQ/SPX relative
strength = growth-vs-broad risk appetite; SPX breadth vs QQQ concentration
= fragility signal. Bonds: flight-to-quality trend = the other side of
equity risk-off. These are the "how the economy works" variables that a
QQQ-only agent is blind to.

## Concrete program (in order)

1. v10 cross-asset state (SPX/bond/gold/credit features) — running.
2. IC-screen protocol: candidate features must show |IC| > threshold with
   forward vol/drawdown (not returns) on train folds before admission.
3. Fractional-Kelly dial (QV penalty sweep) on v9_rel5.
4. Regime-stratified bootstrap (still the last untried playbook item).
5. Distilled-rule benchmark of v9_rel5 (if a 30-line rule matches it, the
   learning is done and the paper writes itself).
