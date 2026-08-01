"""Frozen-policy inference and live signal serialization.

The deployable actor bundle contains only the deterministic PPO actor layers
and train-window normalization statistics. Daily inference therefore needs
NumPy and pandas, not PyTorch, Gymnasium, or Stable-Baselines3.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

MODEL_VERSION = "ppo_v8_nocal_frozen_2023_v1"
LEGACY_MODEL_VERSION = "ppo_v4_resid_frozen_2023_v1"
TRAIN_CUTOFF = "2023-12-31"
FORWARD_START = "2026-01-01"
RESIDUAL_MULTIPLIERS = np.array([0.5, 1.0, 1.5], dtype=np.float64)

BASE_FEATURE_NAMES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_252d",
    "vol_5d",
    "vol_21d",
    "vol_63d",
    "ma_gap_21",
    "ma_gap_50",
    "ma_gap_200",
    "rsi_14",
    "macd_hist",
    "drawdown",
    "hl_range",
    "overnight_gap",
    "vol_ratio_21",
    "dow",
    "month",
]
CONTEXT_FEATURE_NAMES = ["vix", "term_spread_10y_3m", "vix_chg_5d"]
FEATURE_NAMES = BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES
LIVE_FEATURE_NAMES = [
    name for name in FEATURE_NAMES if name not in {"dow", "month"}
]

FORWARD_LOG_FIELDS = [
    "date",
    "generated_at",
    "model_version",
    "source",
    "qqq_close",
    "qqq_close_return",
    "realized_vol_21",
    "momentum_21",
    "drawdown",
    "volume_ratio_21",
    "vix",
    "vt10_exposure",
    "learned_mean",
    "learned_min",
    "learned_max",
    "tilt_multiplier",
    "vt20_exposure",
    "composite_exposure",
    "stance",
    "actor_state_sha256",
]


@dataclass(frozen=True)
class FrozenActorEnsemble:
    """Ten deterministic MLP actors plus one shared feature normalizer."""

    model_version: str
    policy_name: str
    train_cutoff: str
    feature_names: tuple[str, ...]
    normalizer_mean: np.ndarray
    normalizer_std: np.ndarray
    layer1_weight: np.ndarray
    layer1_bias: np.ndarray
    layer2_weight: np.ndarray
    layer2_bias: np.ndarray
    action_weight: np.ndarray
    action_bias: np.ndarray
    artifact_sha256: str

    @property
    def ensemble_size(self) -> int:
        return int(self.layer1_weight.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> "FrozenActorEnsemble":
        artifact = Path(path)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        with np.load(artifact, allow_pickle=False) as saved:
            model_version = str(saved["model_version"].item())
            default_policy_names = {
                LEGACY_MODEL_VERSION: "v4 residual",
                MODEL_VERSION: "v8 no-calendar",
            }
            bundle = cls(
                model_version=model_version,
                policy_name=(
                    str(saved["policy_name"].item())
                    if "policy_name" in saved.files
                    else default_policy_names.get(model_version, model_version)
                ),
                train_cutoff=str(saved["train_cutoff"].item()),
                feature_names=tuple(saved["feature_names"].astype(str).tolist()),
                normalizer_mean=saved["normalizer_mean"].astype(np.float64),
                normalizer_std=saved["normalizer_std"].astype(np.float64),
                layer1_weight=saved["layer1_weight"].astype(np.float64),
                layer1_bias=saved["layer1_bias"].astype(np.float64),
                layer2_weight=saved["layer2_weight"].astype(np.float64),
                layer2_bias=saved["layer2_bias"].astype(np.float64),
                action_weight=saved["action_weight"].astype(np.float64),
                action_bias=saved["action_bias"].astype(np.float64),
                artifact_sha256=digest,
            )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        seeds = self.ensemble_size
        features = len(self.feature_names)
        expected = {
            "normalizer_mean": (features,),
            "normalizer_std": (features,),
            "layer1_weight": (seeds, 64, features + 2),
            "layer1_bias": (seeds, 64),
            "layer2_weight": (seeds, 64, 64),
            "layer2_bias": (seeds, 64),
            "action_weight": (seeds, 3, 64),
            "action_bias": (seeds, 3),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}; expected {shape}")
        if not self.model_version or not self.policy_name or not self.train_cutoff:
            raise ValueError("Actor bundle metadata is incomplete")
        unknown = set(self.feature_names).difference(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Actor bundle has unsupported features: {sorted(unknown)}")
        expected_order = tuple(
            name for name in FEATURE_NAMES if name in set(self.feature_names)
        )
        if len(set(self.feature_names)) != features:
            raise ValueError("Actor feature names must be unique")
        if tuple(self.feature_names) != expected_order:
            raise ValueError("Actor feature order does not match the live pipeline")
        if np.any(self.normalizer_std <= 0):
            raise ValueError("Normalizer standard deviations must be positive")

    def normalize(self, raw_features: np.ndarray) -> np.ndarray:
        values = (raw_features - self.normalizer_mean) / self.normalizer_std
        return np.clip(values, -10.0, 10.0)

    def actions(self, observations: np.ndarray) -> np.ndarray:
        """Return deterministic categorical actions for one row per actor."""
        obs = np.asarray(observations, dtype=np.float64)
        expected = (self.ensemble_size, len(self.feature_names) + 2)
        if obs.shape != expected:
            raise ValueError(f"observations have shape {obs.shape}; expected {expected}")
        hidden1 = np.tanh(
            np.einsum("si,soi->so", obs, self.layer1_weight)
            + self.layer1_bias
        )
        hidden2 = np.tanh(
            np.einsum("si,soi->so", hidden1, self.layer2_weight)
            + self.layer2_bias
        )
        logits = (
            np.einsum("si,soi->so", hidden2, self.action_weight)
            + self.action_bias
        )
        return np.argmax(logits, axis=1)


def make_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``prepare_dataset.make_features`` exactly."""
    close = prices["adj_close"]
    log_return = np.log(close / close.shift(1))
    features = pd.DataFrame(index=prices.index)
    for horizon in [1, 5, 10, 21, 63, 126, 252]:
        features[f"ret_{horizon}d"] = np.log(close / close.shift(horizon))
    for horizon in [5, 21, 63]:
        features[f"vol_{horizon}d"] = (
            log_return.rolling(horizon).std() * np.sqrt(252)
        )
    for horizon in [21, 50, 200]:
        features[f"ma_gap_{horizon}"] = (
            close / close.rolling(horizon).mean() - 1.0
        )

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    features["rsi_14"] = 100 - 100 / (1 + up / down.replace(0, np.nan))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features["macd_hist"] = (macd - signal) / close
    features["drawdown"] = close / close.cummax() - 1.0
    features["hl_range"] = (
        (prices["High"] - prices["Low"]) / prices["Close"]
    )
    features["overnight_gap"] = np.log(
        prices["Open"] / prices["Close"].shift(1)
    )
    features["vol_ratio_21"] = (
        prices["Volume"] / prices["Volume"].rolling(21).mean()
    )
    features["dow"] = prices.index.dayofweek
    features["month"] = prices.index.month
    return features


def build_feature_frame(
    qqq: pd.DataFrame,
    vix: pd.DataFrame,
    tnx: pd.DataFrame,
    irx: pd.DataFrame,
) -> pd.DataFrame:
    """Build every raw feature supported by the frozen actor pipeline.

    The yield scaling intentionally preserves the frozen training convention.
    Changing it would make deployed observations incompatible with the actor.
    """
    features = make_market_features(qqq)
    context = pd.concat(
        [
            vix["Close"].rename("vix"),
            (tnx["Close"] / 10.0).rename("y10"),
            (irx["Close"] / 10.0).rename("y3m"),
        ],
        axis=1,
        sort=True,
    ).sort_index()
    context["term_spread_10y_3m"] = context["y10"] - context["y3m"]
    context["vix_chg_5d"] = context["vix"].pct_change(5)
    context = context.reindex(qqq.index).ffill()
    combined = features.join(context[CONTEXT_FEATURE_NAMES])
    return combined[FEATURE_NAMES].dropna()


def _clean_provider_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        if symbol in data.columns.get_level_values(-1):
            data = data.xs(symbol, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)
    data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data.columns = [str(column) for column in data.columns]
    return data


def _price_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = _clean_provider_frame(frame, symbol)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{symbol} is missing provider columns: {sorted(missing)}")
    adjusted = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["adj_close"] = pd.to_numeric(data[adjusted], errors="coerce")
    return data.dropna(subset=["Open", "High", "Low", "Close", "adj_close"])


def load_checked_market_frames(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the checked-in 2026-07-31 Yahoo snapshot."""
    root = Path(raw_dir)

    def read(filename: str, symbol: str) -> pd.DataFrame:
        frame = pd.read_csv(root / filename, parse_dates=["Date"]).set_index("Date")
        return _price_frame(frame, symbol)

    return {
        "QQQ": read("yf_QQQ.csv", "QQQ"),
        "^VIX": read("yf_IDX_VIX.csv", "^VIX"),
        "^TNX": read("yf_IDX_TNX.csv", "^TNX"),
        "^IRX": read("yf_IDX_IRX.csv", "^IRX"),
    }


def fetch_yahoo_market_frames(
    retries: int = 3,
    retry_delay: float = 4.0,
) -> dict[str, pd.DataFrame]:
    """Fetch delayed end-of-day bars with yfinance.

    This feed is appropriate for the public research demo, not execution.
    """
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    for symbol in ["QQQ", "^VIX", "^TNX", "^IRX"]:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                history = yf.Ticker(symbol).history(
                    period="max",
                    auto_adjust=False,
                    actions=False,
                    timeout=30,
                )
                if history.empty:
                    raise RuntimeError(f"{symbol} returned no rows")
                frames[symbol] = _price_frame(history, symbol)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(retry_delay * (attempt + 1))
        else:
            raise RuntimeError(f"Could not fetch {symbol}: {last_error}")
    return frames


def replay_frozen_policy(
    ensemble: FrozenActorEnsemble,
    features: pd.DataFrame,
    qqq: pd.DataFrame,
    start: str = FORWARD_START,
) -> pd.DataFrame:
    """Replay actors from the frozen forward-test reset through latest close."""
    evaluation = features.loc[pd.Timestamp(start):].copy()
    if len(evaluation) < 22:
        raise ValueError("At least 22 forward rows are required for live inference")

    close_return = qqq["adj_close"].pct_change().reindex(evaluation.index)
    realized_vol = close_return.rolling(21).std() * np.sqrt(252)
    vt10 = (0.10 / realized_vol).clip(upper=1.0)
    vt10.iloc[:21] = 0.5
    if vt10.isna().any():
        raise ValueError("Forward volatility baseline contains missing values")

    actor_exposure = np.zeros(ensemble.ensemble_size, dtype=np.float64)
    exposure_rows = np.empty((len(evaluation), ensemble.ensemble_size))
    action_rows = np.empty((len(evaluation), ensemble.ensemble_size), dtype=int)

    normalized = ensemble.normalize(
        evaluation[list(ensemble.feature_names)].to_numpy(dtype=np.float64)
    )
    for index, (feature_row, baseline) in enumerate(
        zip(normalized, vt10.to_numpy(dtype=np.float64), strict=True)
    ):
        observations = np.column_stack(
            [
                np.repeat(feature_row[None, :], ensemble.ensemble_size, axis=0),
                actor_exposure,
                np.full(ensemble.ensemble_size, baseline),
            ]
        )
        actions = ensemble.actions(observations)
        actor_exposure = np.clip(
            baseline * RESIDUAL_MULTIPLIERS[actions], 0.0, 1.0
        )
        action_rows[index] = actions
        exposure_rows[index] = actor_exposure

    result = pd.DataFrame(index=evaluation.index)
    result["price"] = qqq["adj_close"].reindex(evaluation.index)
    result["daily_change"] = close_return
    result["realized_vol_21"] = evaluation["vol_21d"]
    result["momentum_21"] = np.expm1(evaluation["ret_21d"])
    result["drawdown"] = evaluation["drawdown"]
    result["volume_ratio_21"] = evaluation["vol_ratio_21"]
    result["vix"] = evaluation["vix"]
    result["vt10_exposure"] = vt10
    result["learned_mean"] = exposure_rows.mean(axis=1)
    result["learned_min"] = exposure_rows.min(axis=1)
    result["learned_max"] = exposure_rows.max(axis=1)
    result["tilt_multiplier"] = np.clip(
        result["learned_mean"] / result["vt10_exposure"], 0.5, 1.5
    )
    result["vt20_exposure"] = np.minimum(
        1.5, 2.0 * result["vt10_exposure"]
    )
    result["composite_exposure"] = np.minimum(
        1.5, result["tilt_multiplier"] * result["vt20_exposure"]
    )
    result.attrs["actions"] = action_rows
    result.attrs["actor_exposure"] = exposure_rows
    result.attrs["model_version"] = ensemble.model_version
    return result


def stance_for_exposure(exposure: float) -> str:
    if exposure < 0.65:
        return "Defensive"
    if exposure < 0.95:
        return "Reduced risk"
    if exposure < 1.20:
        return "Fully invested"
    return "Levered"


def _iso_utc(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _number(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def actor_state_sha256(exposure: np.ndarray) -> str:
    canonical = np.round(
        np.asarray(exposure, dtype=np.float64), 5
    ).astype("<f8", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_signal_payload(
    replay: pd.DataFrame,
    ensemble: FrozenActorEnsemble,
    source_name: str,
    generated_at: datetime | None = None,
    history_days: int = 90,
) -> dict:
    if replay.empty:
        raise ValueError("Cannot serialize an empty policy replay")
    latest = replay.iloc[-1]
    latest_actor_exposure = replay.attrs.get("actor_exposure")
    if latest_actor_exposure is None:
        raise ValueError("Replay is missing per-actor exposure state")
    as_of = pd.Timestamp(replay.index[-1])
    generated = generated_at or datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    age_days = (
        generated.astimezone(timezone.utc).date() - as_of.date()
    ).days
    stale = age_days > 4

    learned = float(latest["learned_mean"])
    baseline = float(latest["vt10_exposure"])
    multiplier = float(latest["tilt_multiplier"])
    display_multiplier = round(multiplier, 2)
    stance = stance_for_exposure(learned)
    if display_multiplier < 0.9:
        tilt_read = "trimmed the volatility anchor"
    elif display_multiplier >= 1.1:
        tilt_read = "added risk above the volatility anchor"
    else:
        tilt_read = "kept the volatility anchor nearly unchanged"

    recent = replay.tail(history_days)
    payload = {
        "schemaVersion": 1,
        "asOf": str(as_of.date()),
        "generatedAt": _iso_utc(generated),
        "stale": stale,
        "source": {
            "provider": source_name,
            "frequency": "Delayed end-of-day",
            "instrument": "QQQ",
        },
        "model": {
            "version": ensemble.model_version,
            "displayName": ensemble.policy_name,
            "trainCutoff": ensemble.train_cutoff,
            "featureCount": len(ensemble.feature_names),
            "ensembleSize": ensemble.ensemble_size,
            "artifactSha256": ensemble.artifact_sha256,
            "decisionTiming": "After close for the next close-to-close session",
        },
        "market": {
            "price": _number(latest["price"], 4),
            "dailyChange": _number(latest["daily_change"]),
            "realizedVol21": _number(latest["realized_vol_21"]),
            "momentum21": _number(latest["momentum_21"]),
            "drawdown": _number(latest["drawdown"]),
            "volumeRatio21": _number(latest["volume_ratio_21"]),
            "vix": _number(latest["vix"], 4),
        },
        "signal": {
            "vt10Exposure": _number(baseline, 5),
            "learnedMean": _number(learned, 5),
            "learnedMin": _number(latest["learned_min"], 5),
            "learnedMax": _number(latest["learned_max"], 5),
            "tiltMultiplier": _number(multiplier, 5),
            "vt20Exposure": _number(latest["vt20_exposure"], 5),
            "compositeExposure": _number(latest["composite_exposure"], 5),
            "stance": stance,
            "actorStateSha256": actor_state_sha256(latest_actor_exposure[-1]),
            "researchPosture": (
                "Risk budget below full exposure"
                if learned < 0.95
                else "Risk budget at or above full exposure"
            ),
            "explanation": (
                f"Trailing volatility set the VT10 anchor at {baseline:.2f}x. "
                f"The current ten-seed ensemble {tilt_read} with a "
                f"{multiplier:.2f}x residual, producing {learned:.2f}x."
            ),
        },
        "history": {
            "dates": [str(pd.Timestamp(date).date()) for date in recent.index],
            "price": [_number(value, 4) for value in recent["price"]],
            "vt10Exposure": [
                _number(value, 5) for value in recent["vt10_exposure"]
            ],
            "learnedMean": [
                _number(value, 5) for value in recent["learned_mean"]
            ],
            "learnedMin": [
                _number(value, 5) for value in recent["learned_min"]
            ],
            "learnedMax": [
                _number(value, 5) for value in recent["learned_max"]
            ],
            "compositeExposure": [
                _number(value, 5) for value in recent["composite_exposure"]
            ],
        },
        "limitations": [
            "Research output, not personalized investment advice or an execution order.",
            "Yahoo data is delayed and may be revised; the signal is not intraday.",
            (
                "The learned ensemble was frozen after training through "
                f"{ensemble.train_cutoff}."
            ),
            "The displayed composite is a post-hoc research candidate, not a validated deployment policy.",
        ],
    }
    return payload


def write_signal_json(payload: Mapping, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def append_forward_log(payload: Mapping, output: str | Path) -> bool:
    """Append one unique decision date. Existing rows are never rewritten."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_dates: set[str] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing_dates = {
                row["date"] for row in csv.DictReader(handle) if row.get("date")
            }
    if str(payload["asOf"]) in existing_dates:
        return False

    market = payload["market"]
    signal = payload["signal"]
    row = {
        "date": payload["asOf"],
        "generated_at": payload["generatedAt"],
        "model_version": payload["model"]["version"],
        "source": payload["source"]["provider"],
        "qqq_close": market["price"],
        "qqq_close_return": market["dailyChange"],
        "realized_vol_21": market["realizedVol21"],
        "momentum_21": market["momentum21"],
        "drawdown": market["drawdown"],
        "volume_ratio_21": market["volumeRatio21"],
        "vix": market["vix"],
        "vt10_exposure": signal["vt10Exposure"],
        "learned_mean": signal["learnedMean"],
        "learned_min": signal["learnedMin"],
        "learned_max": signal["learnedMax"],
        "tilt_multiplier": signal["tiltMultiplier"],
        "vt20_exposure": signal["vt20Exposure"],
        "composite_exposure": signal["compositeExposure"],
        "stance": signal["stance"],
        "actor_state_sha256": signal["actorStateSha256"],
    }
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FORWARD_LOG_FIELDS,
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return True


def validate_forward_log(replay: pd.DataFrame, input_path: str | Path) -> None:
    """Reject a refresh that would rewrite an already published actor state."""
    path = Path(input_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    actor_exposure = replay.attrs.get("actor_exposure")
    if actor_exposure is None:
        raise ValueError("Replay is missing per-actor exposure state")
    model_version = replay.attrs.get("model_version")
    if not model_version:
        raise ValueError("Replay is missing its model version")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        logged_model = row.get("model_version", "")
        if not logged_model:
            raise ValueError(
                f"Forward log date {row.get('date', 'unknown')} has no model version"
            )
        if logged_model != model_version:
            continue
        date = pd.Timestamp(row["date"])
        if date not in replay.index:
            raise ValueError(f"Forward log date {date.date()} is absent from replay")
        position = int(replay.index.get_loc(date))
        expected_hash = actor_state_sha256(actor_exposure[position])
        logged_hash = row.get("actor_state_sha256", "")
        if not logged_hash:
            raise ValueError(f"Forward log date {date.date()} has no actor-state hash")
        if logged_hash != expected_hash:
            raise ValueError(
                f"Forward actor state drifted on {date.date()}; "
                "refusing to rewrite the published path"
            )
