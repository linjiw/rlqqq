"""Evaluate the deployable v4/v8 actors and their research overlays.

The 2026 section always replays the checked-in frozen actor artifacts against
the checked market snapshot.  The historical section is added when the
gitignored walk-forward seed series are available (typically after running
``scripts/run_pilot.py`` for both configs).

Selection is deliberately limited to trained core policies.  Composite
policies are reported for research, but are not deployment candidates because
they are post-hoc risk-budget overlays rather than separately trained models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.live import (  # noqa: E402
    FrozenActorEnsemble,
    build_feature_frame,
    load_checked_market_frames,
    replay_frozen_policy,
)

CONFIGS = {
    "v4": "ppo_v4_resid",
    "v8": "ppo_v8_nocal",
}
FROZEN_MODELS = {
    "v4": ROOT / "models" / "live" / "ppo_v4_resid_frozen_2023_v1.npz",
    "v8": ROOT / "models" / "live" / "ppo_v8_nocal_frozen_2023_v1.npz",
}
SEEDS = tuple(range(10))
COST_BPS = 2.0
BORROW_SPREAD_BPS = 50.0
SHARPE_TIE_BAND = 0.02


@dataclass
class FoldEvaluation:
    name: str
    dates: pd.DatetimeIndex
    asset_return: np.ndarray
    cash_return: np.ndarray
    exposures: dict[str, np.ndarray]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def series_inventory(series_root: Path) -> dict:
    """Content-address the 160 ignored walk-forward inputs used by the report."""
    paths = sorted(
        path
        for config in CONFIGS.values()
        for path in series_root.glob(f"{config}_QQQ_F*_s*.npz")
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    try:
        root_label = str(series_root.relative_to(ROOT))
    except ValueError:
        root_label = f"<external>/{series_root.name}"
    repo_root = series_root.parent.parent
    payload: dict[str, str | int | dict] = {
        "seriesRoot": root_label,
        "seriesFiles": len(paths),
        "seriesInventorySha256": digest.hexdigest(),
        "trainingRecipeRevision": git_revision(repo_root),
        "marketInputs": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                ROOT / "data" / "processed" / "prices_QQQ.parquet",
                ROOT / "data" / "raw" / "yf_IDX_IRX.csv",
            )
        },
        "commands": {
            version: (
                "scripts/run_pilot.py --symbol QQQ "
                f"--config {config} --seeds 10 --seed_start 0 --folds 8 "
                "--timesteps 150000 --workers 7"
            )
            for version, config in CONFIGS.items()
        },
    }
    training_python = repo_root / ".venv" / "bin" / "python"
    if training_python.is_file():
        package_code = """
import importlib.metadata as metadata
import json
import platform

names = [
    "numpy", "pandas", "pyarrow", "torch", "stable-baselines3",
    "gymnasium", "scipy", "arch",
]
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}))
"""
        try:
            completed = subprocess.run(
                [str(training_python), "-c", package_code],
                check=True,
                capture_output=True,
                text=True,
            )
            payload["trainingEnvironment"] = json.loads(completed.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            payload["trainingEnvironment"] = {"status": "unavailable"}
        try:
            frozen = subprocess.run(
                [str(training_python), "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            payload["pipFreezeSha256"] = hashlib.sha256(
                frozen.encode("utf-8")
            ).hexdigest()
        except subprocess.CalledProcessError:
            uv = shutil.which("uv")
            if uv is None:
                payload["pipFreezeSha256"] = None
            else:
                frozen = subprocess.run(
                    [uv, "pip", "freeze", "--python", str(training_python)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                payload["pipFreezeSha256"] = hashlib.sha256(
                    frozen.encode("utf-8")
                ).hexdigest()
    return payload


def git_revision(repository: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _daily_cash(index: pd.DatetimeIndex) -> np.ndarray:
    irx = pd.read_csv(
        ROOT / "data" / "raw" / "yf_IDX_IRX.csv",
        parse_dates=["Date"],
    ).set_index("Date")["Close"]
    return (irx.reindex(index).ffill() / 100.0 / 252.0).to_numpy(dtype=float)


def _continuous_vt10(index: pd.DatetimeIndex) -> np.ndarray:
    price = pd.read_parquet(
        ROOT / "data" / "processed" / "prices_QQQ.parquet"
    )["adj_close"]
    realized = price.pct_change().rolling(21).std() * np.sqrt(252.0)
    return (0.10 / realized).clip(upper=1.0).reindex(index).to_numpy(dtype=float)


def _fold_local_vt10(asset_return: np.ndarray) -> np.ndarray:
    """The exact residual baseline used by ExposureTradingEnv on a test fold."""
    returns = pd.Series(np.asarray(asset_return, dtype=float))
    realized = (returns.rolling(21).std() * np.sqrt(252.0)).shift(1)
    return (0.10 / realized).clip(upper=1.0).fillna(0.5).to_numpy(dtype=float)


def _portfolio_returns(
    exposure: np.ndarray,
    asset_return: np.ndarray,
    cash_return: np.ndarray,
    *,
    initial_exposure: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    exposure = np.asarray(exposure, dtype=float)
    previous = np.concatenate([[initial_exposure], exposure[:-1]])
    turnover = np.abs(exposure - previous)
    cash_leg = np.where(
        exposure <= 1.0,
        (1.0 - exposure) * cash_return,
        -(exposure - 1.0)
        * (cash_return + BORROW_SPREAD_BPS / 10_000.0 / 252.0),
    )
    net = (
        exposure * asset_return
        + cash_leg
        - COST_BPS / 10_000.0 * turnover
    )
    return net, turnover


def _metrics(
    net_return: np.ndarray,
    cash_return: np.ndarray,
    exposure: np.ndarray,
    turnover: np.ndarray,
) -> dict[str, float | int]:
    net = np.asarray(net_return, dtype=float)
    cash = np.asarray(cash_return, dtype=float)
    curve = np.cumprod(1.0 + net)
    years = len(net) / 252.0
    excess = net - cash
    annual_volatility = net.std(ddof=1) * np.sqrt(252.0)
    excess_volatility = excess.std(ddof=1)
    sharpe = (
        excess.mean() / excess_volatility * np.sqrt(252.0)
        if excess_volatility > 0
        else np.nan
    )
    downside = net[net < 0]
    sortino = (
        net.mean() * 252.0 / (downside.std(ddof=1) * np.sqrt(252.0))
        if len(downside) > 1
        else np.nan
    )
    curve_with_initial = np.concatenate([[1.0], curve])
    drawdown = (
        curve_with_initial / np.maximum.accumulate(curve_with_initial) - 1.0
    )
    cagr = curve[-1] ** (1.0 / years) - 1.0
    maximum_drawdown = float(drawdown.min())
    annual_turnover = float(turnover.sum() / years)
    return {
        "days": int(len(net)),
        "totalReturn": float(curve[-1] - 1.0),
        "totalMultiple": float(curve[-1]),
        "cagr": float(cagr),
        "volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "maximumDrawdown": maximum_drawdown,
        "calmar": float(cagr / abs(maximum_drawdown)),
        "averageExposure": float(np.mean(exposure)),
        "maximumExposure": float(np.max(exposure)),
        "annualTurnover": annual_turnover,
        "cumulativeTurnover": float(turnover.sum()),
        "transactionCostBps": float(turnover.sum() * COST_BPS),
    }


def _evaluate_folds(folds: list[FoldEvaluation]) -> dict[str, dict]:
    names = tuple(folds[0].exposures)
    output: dict[str, dict] = {}
    for name in names:
        nets: list[np.ndarray] = []
        turnovers: list[np.ndarray] = []
        cash: list[np.ndarray] = []
        exposure: list[np.ndarray] = []
        for fold in folds:
            net, activity = _portfolio_returns(
                fold.exposures[name], fold.asset_return, fold.cash_return
            )
            nets.append(net)
            turnovers.append(activity)
            cash.append(fold.cash_return)
            exposure.append(fold.exposures[name])
        output[name] = _metrics(
            np.concatenate(nets),
            np.concatenate(cash),
            np.concatenate(exposure),
            np.concatenate(turnovers),
        )
    return output


def _evaluate_one_close_lag(folds: list[FoldEvaluation]) -> dict[str, dict]:
    """Sensitivity where a close-t signal first earns close-(t+1) to close-(t+2).

    This is deliberately conservative.  It replaces the same-close fill assumed
    by the research backtest with the previous decision's exposure, resetting to
    cash at the beginning of every held-out fold.
    """
    lagged: list[FoldEvaluation] = []
    for fold in folds:
        exposures = {
            name: (
                values.copy()
                if name == "qqq"
                else np.concatenate([[0.0], values[:-1]])
            )
            for name, values in fold.exposures.items()
        }
        lagged.append(
            FoldEvaluation(
                name=fold.name,
                dates=fold.dates,
                asset_return=fold.asset_return,
                cash_return=fold.cash_return,
                exposures=exposures,
            )
        )
    return _evaluate_folds(lagged)


def _load_seed_exposures(
    series_root: Path,
    config: str,
    fold: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    dates: pd.DatetimeIndex | None = None
    rows: list[np.ndarray] = []
    for seed in SEEDS:
        path = series_root / f"{config}_QQQ_{fold}_s{seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as saved:
            candidate_dates = pd.DatetimeIndex(
                saved["test_dates"].astype("datetime64[ns]")
            )
            if dates is None:
                dates = candidate_dates
            elif not dates.equals(candidate_dates):
                raise ValueError(f"Seed dates differ in {config}/{fold}")
            rows.append(saved["test_exposure"].astype(float))
    assert dates is not None
    return dates, np.stack(rows)


def historical_evaluation(series_root: Path) -> dict:
    prices = pd.read_parquet(
        ROOT / "data" / "processed" / "prices_QQQ.parquet"
    )["adj_close"]
    folds: list[FoldEvaluation] = []
    overlay_sensitivity: list[dict] = []
    for fold_number in range(1, 9):
        fold_name = f"F{fold_number}"
        dates_v4, seeds_v4 = _load_seed_exposures(
            series_root, CONFIGS["v4"], fold_name
        )
        dates_v8, seeds_v8 = _load_seed_exposures(
            series_root, CONFIGS["v8"], fold_name
        )
        if not dates_v4.equals(dates_v8):
            raise ValueError(f"v4/v8 dates differ in {fold_name}")
        dates = dates_v4
        asset_return = (
            prices.pct_change().shift(-1).reindex(dates).to_numpy(dtype=float)
        )
        cash_return = _daily_cash(dates)
        continuous_vt10 = _continuous_vt10(dates)
        actor_vt10 = _fold_local_vt10(asset_return)
        if not (
            np.isfinite(asset_return).all()
            and np.isfinite(cash_return).all()
            and np.isfinite(continuous_vt10).all()
        ):
            raise ValueError(f"Market data are incomplete for {fold_name}")

        v4_core = seeds_v4.mean(axis=0)
        v8_core = seeds_v8.mean(axis=0)
        vt20 = np.minimum(1.5, 2.0 * continuous_vt10)
        actor_vt20 = np.minimum(1.5, 2.0 * actor_vt10)

        exposures: dict[str, np.ndarray] = {
            "v4Core": v4_core,
            "v8Core": v8_core,
            "v4Composite": np.minimum(
                1.5,
                np.clip(v4_core / continuous_vt10, 0.5, 1.5) * vt20,
            ),
            "v8Composite": np.minimum(
                1.5,
                np.clip(v8_core / continuous_vt10, 0.5, 1.5) * vt20,
            ),
            "v4CompositeActorBaseline": np.minimum(
                1.5,
                np.clip(v4_core / actor_vt10, 0.5, 1.5) * actor_vt20,
            ),
            "v8CompositeActorBaseline": np.minimum(
                1.5,
                np.clip(v8_core / actor_vt10, 0.5, 1.5) * actor_vt20,
            ),
            "vt10": continuous_vt10,
            "vt20": vt20,
            "qqq": np.ones(len(dates)),
        }
        folds.append(
            FoldEvaluation(
                name=fold_name,
                dates=dates,
                asset_return=asset_return,
                cash_return=cash_return,
                exposures=exposures,
            )
        )
        overlay_sensitivity.append(
            {
                "fold": fold_name,
                "warmupDays": min(21, len(dates)),
                "meanAbsoluteVt10Difference": float(
                    np.abs(continuous_vt10 - actor_vt10).mean()
                ),
                "maximumAbsoluteVt10Difference": float(
                    np.abs(continuous_vt10 - actor_vt10).max()
                ),
            }
        )

    metrics = _evaluate_folds(folds)
    first_decision = folds[0].dates[0]
    last_decision = folds[-1].dates[-1]
    realized_start = prices.index[prices.index.get_loc(first_decision) + 1]
    realized_end = prices.index[prices.index.get_loc(last_decision) + 1]
    return {
        "period": {
            "decisionStart": str(first_decision.date()),
            "decisionEnd": str(last_decision.date()),
            "realizedStart": str(realized_start.date()),
            "realizedEnd": str(realized_end.date()),
            "days": int(sum(len(fold.dates) for fold in folds)),
            "folds": len(folds),
            "seedsPerFold": len(SEEDS),
            "embargoedStitchedReplay": True,
        },
        "metrics": metrics,
        "oneCloseLagSensitivity": _evaluate_one_close_lag(folds),
        "overlayBaselineSensitivity": overlay_sensitivity,
        "provenance": series_inventory(series_root),
    }


def frozen_2026_evaluation() -> dict:
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    replay = {
        version: replay_frozen_policy(
            FrozenActorEnsemble.load(path), features, frames["QQQ"]
        )
        for version, path in FROZEN_MODELS.items()
    }
    decision_dates = replay["v8"].index[:-1]
    realized_dates = pd.DatetimeIndex(replay["v8"].index[1:])
    qqq = frames["QQQ"]["adj_close"]
    asset_return = (
        qqq.pct_change().shift(-1).reindex(decision_dates).to_numpy(dtype=float)
    )
    cash_return = (
        frames["^IRX"]["Close"].reindex(decision_dates).ffill().to_numpy(dtype=float)
        / 100.0
        / 252.0
    )
    vt10 = replay["v8"]["vt10_exposure"].iloc[:-1].to_numpy(dtype=float)
    exposures = {
        "v4Core": replay["v4"]["learned_mean"].iloc[:-1].to_numpy(dtype=float),
        "v8Core": replay["v8"]["learned_mean"].iloc[:-1].to_numpy(dtype=float),
        "v4Composite": replay["v4"]["composite_exposure"].iloc[:-1].to_numpy(dtype=float),
        "v8Composite": replay["v8"]["composite_exposure"].iloc[:-1].to_numpy(dtype=float),
        "vt10": vt10,
        "vt20": np.minimum(1.5, 2.0 * vt10),
        "qqq": np.ones(len(decision_dates)),
    }
    fold = FoldEvaluation(
        name="H2026",
        dates=decision_dates,
        asset_return=asset_return,
        cash_return=cash_return,
        exposures=exposures,
    )
    metrics = _evaluate_folds([fold])
    one_close_lag = _evaluate_one_close_lag([fold])
    july = realized_dates.month == 7
    for name, exposure in exposures.items():
        net, _ = _portfolio_returns(exposure, asset_return, cash_return)
        metrics[name]["julyReturn"] = float(np.prod(1.0 + net[july]) - 1.0)
    latest = {
        version: {
            "asOf": str(frame.index[-1].date()),
            "coreExposure": float(frame["learned_mean"].iloc[-1]),
            "coreMinimum": float(frame["learned_min"].iloc[-1]),
            "coreMaximum": float(frame["learned_max"].iloc[-1]),
            "vt10Exposure": float(frame["vt10_exposure"].iloc[-1]),
            "tiltMultiplier": float(frame["tilt_multiplier"].iloc[-1]),
            "compositeExposure": float(frame["composite_exposure"].iloc[-1]),
        }
        for version, frame in replay.items()
    }
    return {
        "period": {
            "decisionStart": str(decision_dates[0].date()),
            "decisionEnd": str(decision_dates[-1].date()),
            "realizedStart": str(realized_dates[0].date()),
            "realizedEnd": str(realized_dates[-1].date()),
            "days": int(len(decision_dates)),
            "latestSignalScored": False,
        },
        "metrics": metrics,
        "oneCloseLagSensitivity": one_close_lag,
        "latestUnscoredSignal": latest,
        "artifacts": {
            version: {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
            }
            for version, path in FROZEN_MODELS.items()
        },
        "marketInputs": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                ROOT / "data" / "raw" / "yf_QQQ.csv",
                ROOT / "data" / "raw" / "yf_IDX_VIX.csv",
                ROOT / "data" / "raw" / "yf_IDX_TNX.csv",
                ROOT / "data" / "raw" / "yf_IDX_IRX.csv",
            )
        },
    }


def select_deployment(historical: dict | None) -> dict:
    if historical is None:
        return {
            "winner": None,
            "status": "Historical series required for deployment selection",
        }
    metrics = historical["metrics"]
    candidates = ["v4Core", "v8Core"]
    best_sharpe = max(metrics[name]["sharpe"] for name in candidates)
    tied = [
        name
        for name in candidates
        if best_sharpe - metrics[name]["sharpe"] <= SHARPE_TIE_BAND
    ]
    feature_counts = {"v4Core": 24, "v8Core": 22}
    winner = min(
        tied,
        key=lambda name: (
            feature_counts[name],
            metrics[name]["annualTurnover"],
        ),
    )
    lagged = historical["oneCloseLagSensitivity"]
    return {
        "winner": winner,
        "modelVersion": (
            "ppo_v8_nocal_frozen_2023_v1"
            if winner == "v8Core"
            else "ppo_v4_resid_frozen_2023_v1"
        ),
        "status": "Selected for browser research deployment",
        "deploymentScope": "Research and paper-trading signal; not qualified for capital deployment",
        "capitalDeploymentQualified": False,
        "eligibleCandidates": candidates,
        "ineligibleResearchOverlays": ["v4Composite", "v8Composite"],
        "rule": (
            "Compare trained core policies at the same VT10/cap-1 risk budget. "
            f"Treat a Sharpe difference within {SHARPE_TIE_BAND:.2f} as "
            "non-inferior, then prefer fewer features and lower turnover. "
            "Post-hoc composite overlays are ineligible."
        ),
        "sharpeTieBand": SHARPE_TIE_BAND,
        "tieCandidates": tied,
        "winnerMetrics": metrics[winner],
        "winnerEvidence": {
            "sameCloseSharpeDeltaVsV4": float(
                metrics["v8Core"]["sharpe"] - metrics["v4Core"]["sharpe"]
            ),
            "oneCloseLagSharpeDeltaVsV4": float(
                lagged["v8Core"]["sharpe"] - lagged["v4Core"]["sharpe"]
            ),
            "annualTurnoverDeltaVsV4": float(
                metrics["v8Core"]["annualTurnover"]
                - metrics["v4Core"]["annualTurnover"]
            ),
            "oneCloseLagSharpeDeltaVsVt10": float(
                lagged["v8Core"]["sharpe"] - lagged["vt10"]["sharpe"]
            ),
        },
        "capitalDeploymentReason": (
            "Under the conservative one-close lag, the simple VT10 rule has "
            "higher Sharpe and lower turnover than either RL core."
        ),
    }


def build_payload(series_root: Path) -> dict:
    historical: dict | None
    try:
        historical = historical_evaluation(series_root)
        historical_status = "complete"
    except FileNotFoundError as error:
        historical = None
        historical_status = f"missing: {error}"
    frozen = frozen_2026_evaluation()
    return {
        "schemaVersion": 1,
        "evaluatedOn": str(date.today()),
        "sourceRevision": git_revision(),
        "evaluationImplementation": {
            "path": "scripts/evaluate_model_benchmark.py",
            "sha256": file_sha256(Path(__file__)),
            "sourceRevisionMeaning": (
                "Git revision of the training recipe and checked market/model inputs; "
                "the evaluator itself is content-addressed separately."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "accounting": {
            "asset": "QQQ adjusted close total return",
            "cash": "^IRX / 100 / 252",
            "oneWayCostBps": COST_BPS,
            "leveragedFinancing": "T-bill + 50 bps",
            "initialExposure": 0.0,
        },
        "historicalStatus": historical_status,
        "historicalWalkForward": historical,
        "frozen2026": frozen,
        "deploymentSelection": select_deployment(historical),
        "limitations": [
            "The 2010-2025 folds were used during model research and are not a fresh holdout for v8 selection.",
            "The 2026 holdout was spent before the v8 ablation was selected; it is a forward sanity check, not an untouched selection set.",
            "Composite policies are post-hoc volatility-budget overlays and are not separately trained models.",
            "The backtest assumes a close-t decision can earn close-t to close-(t+1) returns; live execution after the close can differ.",
            "The one-close-lag sensitivity is conservative and is reported separately; it is not a simulation of next-open execution.",
            "The archived runs did not record their full training stack. requirements-research.txt now pins the current stack for future recipe reruns, but cannot make old PPO weights byte-identical.",
            "The recipe rerun did not numerically reproduce the archived headlines: published v4 core/composite CAGR was 13.19%/23.45% and published v8 core CAGR was 12.90%, versus 12.52%/22.45%/12.53% in this environment.",
            "The 0.005 same-close Sharpe edge does not establish statistical superiority; v8 is retained under the predeclared simplification and turnover rule.",
            "The final historical decision is dated 2025-12-31 but its close-to-close return realizes on 2026-01-02; excluding it does not change the ranking.",
            "The frozen artifact's 2023-12-31 cutoff is a decision-date cutoff: its last training feature row is 2023-12-29 and the associated final reward realizes on 2024-01-02.",
        ],
    }


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _metric_row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {_pct(metrics['cagr'])} | {metrics['sharpe']:.3f} | "
        f"{_pct(metrics['maximumDrawdown'])} | {metrics['calmar']:.3f} | "
        f"{metrics['averageExposure']:.2f}x | {metrics['annualTurnover']:.2f} |"
    )


def render_markdown(payload: dict) -> str:
    selection = payload["deploymentSelection"]
    lines = [
        "# RLQQQ v4 vs v8 deployment benchmark",
        "",
        f"Evaluated {payload['evaluatedOn']} at `{payload['sourceRevision']}`.",
        "",
        "## Deployment decision",
        "",
        f"**Winner: `{selection.get('modelVersion') or 'unresolved'}`.**",
        "",
        f"**Scope: {selection.get('deploymentScope', selection['status'])}.**",
        "",
        selection["rule"] if selection.get("rule") else selection["status"],
        "",
    ]
    historical = payload["historicalWalkForward"]
    if historical is not None:
        lines.extend(
            [
                "## 2010-2025 walk-forward rerun",
                "",
                (
                    f"Decision dates {historical['period']['decisionStart']} through "
                    f"{historical['period']['decisionEnd']}; returns realized through "
                    f"{historical['period']['realizedEnd']}."
                ),
                "",
                "| Policy | CAGR | Sharpe | Max DD | Calmar | Avg exposure | Annual turnover |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "v8Core": "v8 core (eligible)",
            "v4Core": "v4 core (eligible)",
            "v8Composite": "v8 composite (post-hoc dashboard convention)",
            "v4Composite": "v4 composite (post-hoc dashboard convention)",
            "vt10": "VT10 rule",
            "vt20": "VT20 rule",
            "qqq": "QQQ",
        }
        for key in labels:
            lines.append(_metric_row(labels[key], historical["metrics"][key]))
        lines.extend(
            [
                "",
                (
                    "The composite rows above reproduce the archived dashboard's "
                    "continuous-VT anchor. Exact fold-local actor-anchor variants "
                    "are included in the JSON as `v4CompositeActorBaseline` and "
                    "`v8CompositeActorBaseline`."
                ),
                "",
                "### One-close-lag sensitivity",
                "",
                "| Policy | CAGR | Sharpe | Max DD | Calmar | Avg exposure | Annual turnover |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in labels:
            lines.append(
                _metric_row(
                    labels[key],
                    historical["oneCloseLagSensitivity"][key],
                )
            )
        lines.append("")

    forward = payload["frozen2026"]
    lines.extend(
        [
            "## Frozen-policy 2026 replay",
            "",
            "Decision dates 2026-01-02 through 2026-07-30; returns realized through 2026-07-31.",
            "",
            "| Policy | YTD | July | Sharpe | Max DD | Avg exposure |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "v8Core": "v8 core",
        "v4Core": "v4 core",
        "v8Composite": "v8 composite",
        "v4Composite": "v4 composite",
        "vt10": "VT10 rule",
        "vt20": "VT20 rule",
        "qqq": "QQQ",
    }
    for key, label in labels.items():
        metric = forward["metrics"][key]
        lines.append(
            f"| {label} | {_pct(metric['totalReturn'])} | "
            f"{_pct(metric['julyReturn'])} | {metric['sharpe']:.3f} | "
            f"{_pct(metric['maximumDrawdown'])} | {metric['averageExposure']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "### Frozen 2026 one-close-lag sensitivity",
            "",
            "| Policy | Total return | Sharpe | Max DD |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in labels.items():
        metric = forward["oneCloseLagSensitivity"][key]
        lines.append(
            f"| {label} | {_pct(metric['totalReturn'])} | "
            f"{metric['sharpe']:.3f} | {_pct(metric['maximumDrawdown'])} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series-root",
        type=Path,
        default=ROOT / "results" / "series",
        help="Directory containing the gitignored per-seed walk-forward NPZ files.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "results" / "model_benchmark.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "results" / "model_benchmark.md",
    )
    parser.add_argument(
        "--web-output",
        type=Path,
        default=ROOT / "docs" / "assets" / "model-benchmark.json",
    )
    args = parser.parse_args()

    payload = build_payload(args.series_root)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    for output in (args.json_output, args.web_output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {display_path(args.json_output)}")
    print(f"Wrote {display_path(args.markdown_output)}")
    print(f"Wrote {display_path(args.web_output)}")
    print(json.dumps(payload["deploymentSelection"], indent=2))


if __name__ == "__main__":
    main()
