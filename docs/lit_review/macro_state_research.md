# Conditioning Information for QQQ Risk-Timing (verified 2026-08-02)

Verdict framing: the defensible channel is RISK-TIMING, not return
prediction (Goyal-Welch-Zafirov 2024: half of 29 post-2008 predictors fail
OOS; return prediction is dead).

## Verified anchors
- Vol management: Moreira-Muir JF 2017 (the baseline we exploit);
  Cederburg JFE 2020 (fragile except market factor).
- Credit: Gilchrist-Zakrajsek AER 2012 excess bond premium leads activity
  and asset prices ~12mo (FRED BAMLH0A0HYM2 from 1996, BAA10Y longer).
- Curve slope: Estrella-Mishkin REStat 1998 - best single recession
  predictor, works best ALONE (don't add curve variants).
- Financial conditions: Adrian-Boyarchenko-Giannone AER 2019 (growth-at-
  risk; NFCI shifts downside quantile only) - weakest tier.
- Cross-asset TSM: Pitkajarvi et al. JFE 2020 - bond returns predict equity
  returns; +40% TSM Sharpe. THE cross-asset signal with direct evidence.
- VRP: Bollerslev-Tauchen-Zhou RFS 2009 (quarterly horizon, in-sample);
  Pyun JFE 2019 (only verified OOS variant); Bekaert-Hoerova 2014: for
  risk-off detection, conditional vol beats VRP - VRP complements only.
- VIX term structure: Johnson JFQA 2017 (slope carries the information);
  **Cheng RFS 2019: declining VIX premium PRECEDES rising market risk** -
  the highest-conviction addition (VIX3M/VIX - 1; VXVCLS from Dec-2007).
- Stock-bond corr: Campbell-Pflueger-Viceira JPE 2020 (macro-regime
  object); AQR JPM 2023 + Molenaar FAJ 2024: use SLOW (~252d) window;
  60d rolling is unvalidated noise.
- State discipline: Zhang et al. IJCAI-23 generalization bound - trading-RL
  overfitting driven by large obs space + limited samples; compact contexts.
  Harvey-Liu-Zhu t>3 + deflated Sharpe before any feature enters.

## Ranked additions
1. VIX term slope (VXVCLS/VIXCLS - 1) + backwardation flag [needs 2007+;
   for era tests substitute VIX-RV spread]
2. HY OAS z-score + 20d change (lag 1d for publication)
3. VRP (complement, monthly tilt)
4. Stock-bond corr 252d (slow regime flag; NOT 60d)
5. Bond TSM 63d sign (Pitkajarvi)
6. NFCI 4w change (add last, cut first)

## Do NOT add
- Market breadth (verified absence of peer-reviewed OOS anchor)
- **QQQ/SPY ratio momentum (no academic support; mechanically redundant
  with own momentum - the HAR failure mode)** <- our v10 spx_ratio_mom_63
  is exactly this; flag for ablation
- Valuation predictors / extra curve variants (dead OOS)

## Verified gaps (publishable)
- No RL-trading paper ablates cross-asset vs price-only states.
- No academic validation of QQQ/SPY ratio as risk-appetite feature.
