"""Build the static data bundle used by the GitHub Pages research dashboard.

The headline composite is reconstructed exactly as documented in
results/live_decision_snapshot.md:

    learned = mean exposure from ppo_v4_resid seeds 0..9
    multiplier = clip(learned / VT10, 0.5, 1.5)
    VT20 = min(1.5, 2 * VT10)
    composite = min(1.5, multiplier * VT20)

Every strategy is re-accounted fold by fold at 2 bps. This preserves the
walk-forward resets and the embargo gaps used by the research reports.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.baselines import vol_target
from rlqqq.data import PROCESSED, load_market
from rlqqq.env import portfolio_returns
from rlqqq.metrics import perf, turnover_stats
from rlqqq.stats import paired_bootstrap_delta
from rlqqq.walkforward import load_folds

RESULTS = ROOT / "results" / "series"
OUTPUT = ROOT / "docs" / "assets" / "policy-data.json"

EVENTS = [
    {
        "date": "2011-08-08",
        "label": "US downgrade selloff",
        "detail": "The US credit downgrade and euro-area stress drove a volatility shock.",
    },
    {
        "date": "2015-08-24",
        "label": "China growth shock",
        "detail": "A global equity selloff tested how quickly the policy reduced its volatility budget.",
    },
    {
        "date": "2018-12-24",
        "label": "Q4 risk-off",
        "detail": "Tightening fears and slowing growth pushed QQQ into a sharp fourth-quarter drawdown.",
    },
    {
        "date": "2020-03-16",
        "label": "COVID crash",
        "detail": "Pandemic liquidation produced the highest realized volatility in the replay.",
    },
    {
        "date": "2022-06-13",
        "label": "Inflation and rates",
        "detail": "Persistent inflation and rapid rate repricing deepened the technology bear market.",
    },
    {
        "date": "2025-04-04",
        "label": "Tariff shock",
        "detail": "A policy-driven selloff caused a fast volatility and exposure adjustment.",
    },
]

EXPECTED = {
    "composite": {"cagr": 0.2345, "sharpe": 1.055, "max_dd": -0.2314},
    "learned": {"cagr": 0.1319, "sharpe": 1.047, "max_dd": -0.1221},
    "vt20": {"cagr": 0.2222, "sharpe": 1.041, "max_dd": -0.2186},
    "qqq": {"cagr": 0.2066, "sharpe": 0.947, "max_dd": -0.2956},
}


def _rounded(values: np.ndarray, digits: int = 6) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values):
        number = float(value)
        out.append(round(number, digits) if np.isfinite(number) else None)
    return out


def _metric_record(
    returns: np.ndarray,
    cash: np.ndarray,
    exposure: np.ndarray | None = None,
) -> dict:
    values = perf(returns, cash)
    record = {
        "cagr": values["cagr"],
        "sharpe": values["sharpe"],
        "maxDrawdown": values["max_dd"],
        "volatility": values["vol"],
        "sortino": values["sortino"],
        "calmar": values["calmar"],
        "totalMultiple": values["total_multiple"],
    }
    if exposure is not None:
        activity = turnover_stats(exposure)
        record.update(
            {
                "averageExposure": activity["avg_exposure"],
                "annualTurnover": activity["ann_turnover"],
                "fullExposureDays": activity["pct_days_full"],
            }
        )
    return record


def _risk_label(realized_vol: float) -> str:
    if realized_vol >= 0.40:
        return "Extreme volatility"
    if realized_vol >= 0.25:
        return "High volatility"
    if realized_vol >= 0.18:
        return "Elevated volatility"
    return "Contained volatility"


def _stance(exposure: float) -> str:
    if exposure < 0.65:
        return "Defensive"
    if exposure < 0.95:
        return "Reduced risk"
    if exposure < 1.20:
        return "Fully invested"
    return "Levered"


def _decision_summary(
    exposure: float,
    base: float,
    multiplier: float,
    realized_vol: float,
) -> str:
    if base >= 1.49:
        base_read = "Low volatility put the rule at its 1.50x cap"
    elif base <= 0.75:
        base_read = f"Volatility cut the rule base to {base:.2f}x"
    else:
        base_read = f"Volatility set the rule base at {base:.2f}x"

    if multiplier >= 1.10:
        tilt_read = f"the learned tilt added risk ({multiplier:.2f}x)"
    elif multiplier <= 0.90:
        tilt_read = f"the learned tilt trimmed risk ({multiplier:.2f}x)"
    else:
        tilt_read = f"the learned tilt stayed near neutral ({multiplier:.2f}x)"

    return (
        f"{_risk_label(realized_vol)}. {base_read}; {tilt_read}. "
        f"Final target: {exposure:.2f}x."
    )


def _nearest_index(dates: pd.DatetimeIndex, target: str) -> int:
    return int(dates.get_indexer([pd.Timestamp(target)], method="nearest")[0])


def _wealth(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return wealth, drawdown


def build() -> dict:
    qqq = load_market("QQQ")
    folds = load_folds()

    qqq_price = pd.read_parquet(PROCESSED / "prices_QQQ.parquet")["adj_close"]
    spy_price = pd.read_parquet(PROCESSED / "prices_SPY.parquet")["adj_close"]
    spy_next_return = spy_price.pct_change().shift(-1)

    qqq_close_return = qqq_price.pct_change()
    market_signals = pd.DataFrame(index=qqq_price.index)
    market_signals["price"] = qqq_price
    market_signals["realized_vol_21"] = (
        qqq_close_return.rolling(21).std() * np.sqrt(252)
    )
    market_signals["momentum_21"] = qqq_price.pct_change(21)
    market_signals["momentum_63"] = qqq_price.pct_change(63)
    market_signals["market_drawdown"] = (
        qqq_price / qqq_price.cummax() - 1.0
    )

    chunks: dict[str, list[np.ndarray]] = {
        key: []
        for key in [
            "dates",
            "cash",
            "composite",
            "learned",
            "vt20",
            "qqq",
            "spy",
            "composite_exposure",
            "learned_exposure",
            "vt20_exposure",
            "vt10_exposure",
            "multiplier",
        ]
    }
    fold_records: list[dict] = []
    offset = 0

    for fold in folds:
        test = qqq.slice(fold.test_start, fold.test_end)
        exposures = []
        for seed in range(10):
            path = RESULTS / f"ppo_v4_resid_QQQ_{fold.name}_s{seed}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing required series: {path}")
            saved = np.load(path)
            saved_dates = pd.to_datetime(saved["test_dates"])
            if not saved_dates.equals(test.index):
                raise ValueError(f"{path.name} dates do not match {fold.name}")
            exposures.append(saved["test_exposure"])

        learned_exposure = np.stack(exposures).mean(axis=0)
        vt10_exposure = vol_target(test, target=0.10)
        multiplier = np.clip(
            np.divide(
                learned_exposure,
                vt10_exposure,
                out=np.ones_like(learned_exposure),
                where=vt10_exposure > 1e-12,
            ),
            0.5,
            1.5,
        )
        vt20_exposure = np.minimum(1.5, 2.0 * vt10_exposure)
        composite_exposure = np.minimum(
            1.5, multiplier * vt20_exposure
        )

        spy_return = spy_next_return.reindex(test.index).to_numpy(dtype=float)
        if np.isnan(spy_return).any():
            raise ValueError(f"SPY price returns do not cover {fold.name}")

        returns = {
            "composite": portfolio_returns(
                composite_exposure, test.ret, test.cash, 2.0
            ),
            "learned": portfolio_returns(
                learned_exposure, test.ret, test.cash, 2.0
            ),
            "vt20": portfolio_returns(
                vt20_exposure, test.ret, test.cash, 2.0
            ),
            "qqq": portfolio_returns(
                np.ones(len(test)), test.ret, test.cash, 2.0
            ),
            "spy": portfolio_returns(
                np.ones(len(test)), spy_return, test.cash, 2.0
            ),
        }

        chunks["dates"].append(test.index.to_numpy())
        chunks["cash"].append(test.cash)
        for name, values in returns.items():
            chunks[name].append(values)
        chunks["composite_exposure"].append(composite_exposure)
        chunks["learned_exposure"].append(learned_exposure)
        chunks["vt20_exposure"].append(vt20_exposure)
        chunks["vt10_exposure"].append(vt10_exposure)
        chunks["multiplier"].append(multiplier)

        fold_records.append(
            {
                "name": fold.name,
                "startIndex": offset,
                "endIndex": offset + len(test) - 1,
                "startDate": str(test.index[0].date()),
                "endDate": str(test.index[-1].date()),
            }
        )
        offset += len(test)

    merged = {name: np.concatenate(parts) for name, parts in chunks.items()}
    dates = pd.DatetimeIndex(merged["dates"])
    signals = market_signals.reindex(dates)
    if signals.isna().any().any():
        raise ValueError("Market signals contain missing values on replay dates")

    wealth: dict[str, np.ndarray] = {}
    drawdowns: dict[str, np.ndarray] = {}
    for name in ["composite", "learned", "vt20", "qqq", "spy"]:
        wealth[name], drawdowns[name] = _wealth(merged[name])

    policy_exposures = {
        "composite": merged["composite_exposure"],
        "learned": merged["learned_exposure"],
        "vt20": merged["vt20_exposure"],
    }
    policy_names = {
        "composite": {
            "name": "Tilt x VT20",
            "shortName": "Composite",
            "status": "Highest observed return - post-hoc",
            "description": (
                "The v4 learned multiplier transferred onto a 20% volatility "
                "target with a 1.5x cap."
            ),
        },
        "learned": {
            "name": "PPO v4 residual ensemble",
            "shortName": "Learned core",
            "status": "Best robust learned model",
            "description": (
                "Mean exposure across 10 PPO seeds, learned as a residual "
                "around a causal 10% volatility target."
            ),
        },
        "vt20": {
            "name": "VT20, 1.5x cap",
            "shortName": "Rule baseline",
            "status": "Best simple policy",
            "description": (
                "A non-learning 20% volatility target, capped at 1.5x and "
                "charged the same trading and financing costs."
            ),
        },
        "qqq": {
            "name": "QQQ buy and hold",
            "shortName": "QQQ",
            "status": "Primary benchmark",
            "description": "One dollar continuously invested in QQQ.",
        },
        "spy": {
            "name": "SPY buy and hold",
            "shortName": "S&P 500",
            "status": "Broad-market benchmark",
            "description": "One dollar continuously invested in SPY.",
        },
    }

    policies: dict[str, dict] = {}
    for name, copy in policy_names.items():
        exposure = policy_exposures.get(name)
        policies[name] = {
            **copy,
            "metrics": _metric_record(
                merged[name], merged["cash"], exposure=exposure
            ),
        }

    for policy, expected_metrics in EXPECTED.items():
        actual = policies[policy]["metrics"]
        mapping = {
            "cagr": actual["cagr"],
            "sharpe": actual["sharpe"],
            "max_dd": actual["maxDrawdown"],
        }
        for metric, expected in expected_metrics.items():
            if not np.isclose(mapping[metric], expected, atol=5e-5):
                raise AssertionError(
                    f"{policy} {metric} drifted: "
                    f"{mapping[metric]} != {expected}"
                )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rlqqq-matplotlib")
    sharpe_inference = paired_bootstrap_delta(
        merged["composite"] - merged["cash"],
        merged["qqq"] - merged["cash"],
        n_reps=10_000,
        seed=0,
        metric="sharpe",
    )
    cagr_inference = paired_bootstrap_delta(
        merged["composite"],
        merged["qqq"],
        n_reps=10_000,
        seed=0,
        metric="cagr",
    )

    annual: list[dict] = []
    years = dates.year
    exposure_change = np.abs(
        merged["composite_exposure"]
        - np.concatenate(
            [[0.0], merged["composite_exposure"][:-1]]
        )
    )
    for year in sorted(np.unique(years)):
        indices = np.flatnonzero(years == year)
        annual_vol = signals["realized_vol_21"].to_numpy()[indices]
        annual_dd = signals["market_drawdown"].to_numpy()[indices]
        score = (
            0.45 * np.clip((annual_vol - 0.12) / 0.35, 0.0, 1.0)
            + 0.35 * np.clip(np.abs(annual_dd) / 0.35, 0.0, 1.0)
            + 0.20 * np.clip(exposure_change[indices] / 1.0, 0.0, 1.0)
        )
        key_index = int(indices[int(np.argmax(score))])
        base = float(merged["vt20_exposure"][key_index])
        target = float(merged["composite_exposure"][key_index])
        multiplier = float(merged["multiplier"][key_index])
        realized_vol = float(
            signals["realized_vol_21"].iloc[key_index]
        )

        annual.append(
            {
                "year": int(year),
                "sampledDays": int(len(indices)),
                "partial": bool(len(indices) < 240),
                "returns": {
                    name: round(
                        float(np.prod(1.0 + merged[name][indices]) - 1.0),
                        6,
                    )
                    for name in [
                        "composite",
                        "learned",
                        "vt20",
                        "qqq",
                        "spy",
                    ]
                },
                "averageExposure": round(
                    float(merged["composite_exposure"][indices].mean()), 4
                ),
                "averageMultiplier": round(
                    float(merged["multiplier"][indices].mean()), 4
                ),
                "keyIndex": key_index,
                "keyDate": str(dates[key_index].date()),
                "keyStance": _stance(target),
                "keyDecision": _decision_summary(
                    target, base, multiplier, realized_vol
                ),
            }
        )

    events: list[dict] = []
    for definition in EVENTS:
        index = _nearest_index(dates, definition["date"])
        events.append(
            {
                **definition,
                "index": index,
                "date": str(dates[index].date()),
                "stance": _stance(
                    float(merged["composite_exposure"][index])
                ),
                "decision": _decision_summary(
                    float(merged["composite_exposure"][index]),
                    float(merged["vt20_exposure"][index]),
                    float(merged["multiplier"][index]),
                    float(signals["realized_vol_21"].iloc[index]),
                ),
            }
        )

    series = {}
    for name in ["composite", "learned", "vt20", "qqq", "spy"]:
        series[name] = {
            "dailyReturn": _rounded(merged[name], 7),
            "wealth": _rounded(wealth[name], 6),
            "drawdown": _rounded(drawdowns[name], 6),
        }

    all_replay_dates = qqq.index[
        (qqq.index >= dates[0]) & (qqq.index <= dates[-1])
    ]
    omitted_days = len(all_replay_dates.difference(dates))

    return {
        "meta": {
            "title": "RLQQQ policy replay",
            "sourceAsOf": str(qqq_price.index[-1].date()),
            "replayStart": str(dates[0].date()),
            "replayEnd": str(dates[-1].date()),
            "sampledDays": int(len(dates)),
            "sampleYears": round(len(dates) / 252.0, 2),
            "omittedEmbargoDays": int(omitted_days),
            "costBps": 2.0,
            "borrowSpreadBps": 50.0,
            "policyCap": 1.5,
            "disclaimer": "Research result, not investment advice.",
        },
        "policies": policies,
        "comparison": {
            "benchmark": "qqq",
            "sharpeDelta": round(float(sharpe_inference["delta"]), 4),
            "sharpeCi": [
                round(float(sharpe_inference["ci_lo"]), 4),
                round(float(sharpe_inference["ci_hi"]), 4),
            ],
            "sharpeProbabilityPositive": round(
                float(sharpe_inference["p_gt_zero"]), 4
            ),
            "cagrDelta": round(float(cagr_inference["delta"]), 4),
            "cagrCi": [
                round(float(cagr_inference["ci_lo"]), 4),
                round(float(cagr_inference["ci_hi"]), 4),
            ],
            "significant": False,
            "bootstrap": "Paired stationary bootstrap, 10,000 reps",
        },
        "dates": [str(date.date()) for date in dates],
        "series": series,
        "signals": {
            "compositeExposure": _rounded(
                merged["composite_exposure"], 5
            ),
            "learnedExposure": _rounded(
                merged["learned_exposure"], 5
            ),
            "vt20Exposure": _rounded(merged["vt20_exposure"], 5),
            "vt10Exposure": _rounded(merged["vt10_exposure"], 5),
            "multiplier": _rounded(merged["multiplier"], 5),
            "price": _rounded(signals["price"].to_numpy(), 4),
            "realizedVol21": _rounded(
                signals["realized_vol_21"].to_numpy(), 6
            ),
            "momentum21": _rounded(
                signals["momentum_21"].to_numpy(), 6
            ),
            "momentum63": _rounded(
                signals["momentum_63"].to_numpy(), 6
            ),
            "marketDrawdown": _rounded(
                signals["market_drawdown"].to_numpy(), 6
            ),
        },
        "folds": fold_records,
        "events": events,
        "annual": annual,
        "evidence": {
            "eraHoldout": {
                "period": "NDX 2000-2009",
                "learnedCagr": 0.039,
                "learnedSharpe": 0.16,
                "learnedMaxDrawdown": -0.272,
                "benchmarkCagr": -0.054,
                "benchmarkSharpe": -0.06,
                "benchmarkMaxDrawdown": -0.822,
                "sharpeDelta": 0.37,
                "sharpeCi": [0.09, 0.67],
                "significant": True,
            },
            "forwardHoldout": {
                "period": "QQQ 2026-01-02 to 2026-07-30",
                "status": "Preregistered Sharpe criterion failed",
                "blendTotalReturn": 0.0645,
                "blendSharpe": 0.67,
                "blendMaxDrawdown": -0.085,
                "benchmarkTotalReturn": 0.1245,
                "benchmarkSharpe": 0.89,
                "benchmarkMaxDrawdown": -0.117,
            },
        },
    }


def main() -> None:
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)} "
        f"({OUTPUT.stat().st_size / 1024:.1f} KiB)"
    )
    for key in ["composite", "learned", "vt20", "qqq", "spy"]:
        metrics = payload["policies"][key]["metrics"]
        print(
            f"{key:>9}: CAGR {metrics['cagr']:.2%}, "
            f"Sharpe {metrics['sharpe']:.3f}, "
            f"MaxDD {metrics['maxDrawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
