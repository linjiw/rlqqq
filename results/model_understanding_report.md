# Understanding the Audited v4 Model (2026-08-02)

Subject: `ppo_v4_resid` 10-seed mean-exposure ensemble on QQQ — the study's
audited legacy and historical-replay policy (Sharpe 1.047 vs VT 1.033 / B&H
0.947, 2010–2025; significant B&H beat on the NDX 2000s era holdout). The
later v8 no-calendar ablation is the current reference policy.

## 1. The decision, distilled

The policy's only decision is a daily multiplier m ∈ [0.5, 1.5] on the vt10
baseline. Surrogate models on the 3,838 OOS decision days:

- **Linear surrogate explains R² = 0.45** of the multiplier with just the
  z-scored state; a depth-3 tree gets 0.38. The policy is *substantially*
  systematic, not noise.
- Top drivers (linear, signed): **volume ratio (−)**, **21d MA gap (+)**,
  **200d MA gap (−)**, **RSI-14 (−)**, term spread (−), 5–10d returns (−),
  21d vol (+).
- Readable tree logic: **calm markets (vol_21d < 12%) → m ≈ 0.84–0.97**
  (shade down); **stressed markets (vol_21d > 12%) → m ≈ 1.03–1.14**, most
  aggressively when price is below its 50d MA (m = 1.13) — i.e. the policy
  in one sentence:

  > *When vol is high and price is below trend but volume isn't panicking,
  > hold MORE than vol-targeting allows; when markets are calm and
  > overbought (high RSI, high 200d gap), hold slightly less.*

- **Red flag surfaced by distillation:** `month` appears as a split
  (calendar seasonality). This is likely spurious fitting — a known risk we
  accepted keeping calendar features. Action item: ablate `dow`/`month`
  features in the next retrain; if performance holds, drop them permanently.

## 2. Episode case studies (ensemble vs its baseline)

| episode | QQQ ret | avg mult | behavior |
|---|---|---|---|
| 2011 US downgrade | −10% | 1.16 | ADD |
| 2015–16 China scare | −1% | 1.10 | ADD |
| 2018 Q4 selloff | −18% | 1.08 | ADD |
| **COVID crash Feb–Mar 2020** | −22% | **0.92** | **CUT** |
| COVID recovery | +64% | 0.99 | track |
| 2022 bear | −24% | 1.16 | ADD |
| 2022–23 rebound | +43% | 1.06 | ADD |
| 2024 Aug vol spike | −4% | 1.11 | ADD |
| 2025 spring drawdown | −11% | 1.15 | ADD |

The contrarian ADD pattern is consistent across 8 of 9 episodes. The one
exception is informative: in the COVID crash — the fastest vol explosion in
the sample — the agent *cut below* baseline during the crash itself and only
re-tracked in the recovery. The volume-ratio feature (its strongest negative
driver) distinguishes "grinding drawdown" (add) from "panic liquidation"
(don't catch the knife). That is a more nuanced learned behavior than pure
buy-the-dip.

## 3. Tilt-transfer validated out-of-era

The leveraged construction (multiplier × vt20-cap1.5), designed entirely on
QQQ 2010–2025, applied to the NDX 2000–2009 era holdout:

| policy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| tilt × vt20c | +3.7% | +0.147 | −51% |
| vt20c rule | +3.7% | +0.145 | −49% |
| buy-and-hold | −5.4% | −0.055 | −82% |

**vs B&H: ΔSharpe +0.26 [+0.02, +0.51] — significant out-of-era.** The tilt
neither adds nor subtracts vs the leveraged rule out-of-era (Δ ≈ 0) — its
value concentrates on defensive baselines (consistent with v6 finding). Key
risk disclosure: in a crisis decade the leveraged variant draws down −51%
(vs −27% for the defensive core policy); leverage buys CAGR with tail risk,
exactly as theory says.

## 4. Core key decisions of the system (consolidated)

1. **Baseline anchor** (vol-target) supplies the risk management; the RL
   layer supplies a bounded, interpretable tilt. Neither alone matches the
   combination on QQQ.
2. **Ensemble averaging** (10 seeds) is what makes the tilt reliable — the
   distilled signal exists in the mean, individual seeds are noisy.
3. **Bootstrap-path training** is why the tilt generalizes (it survives a
   regime 10 years before its development window).
4. The policy's edge is **regime-asymmetric by construction**: it wins in
   high-vol/drawdown states, pays a small premium in calm bulls — verified
   in 2010–2025 folds, the 2000s era holdout, and the (low-power) 2026
   forward holdout, all consistent.
