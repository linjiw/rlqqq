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

MODEL_VERSION = "ppo_v10_macro_frozen_2023_v1"
V8_MODEL_VERSION = "ppo_v8_nocal_frozen_2023_v1"
LEGACY_MODEL_VERSION = "ppo_v4_resid_frozen_2023_v1"
TRAIN_CUTOFF = "2023-12-31"
FORWARD_START = "2026-01-01"
RESIDUAL_MULTIPLIERS = np.array([0.5, 1.0, 1.5], dtype=np.float64)
CROSS_ASSET_FEATURE_NAMES = [
    "spx_ratio_mom_63",
    "bond_trend",
    "gold_trend",
    "stock_bond_corr_63",
    "vrp_proxy",
    "curve_slope",
]

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
    # residual-policy contract; legacy bundles (v4/v8) default to the
    # 3-multiplier defensive vt10 profile.
    residual_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    vt_target: float = 0.10
    max_exposure: float = 1.0

    @property
    def ensemble_size(self) -> int:
        return int(self.layer1_weight.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.action_bias.shape[1])

    @classmethod
    def load(cls, path: str | Path) -> "FrozenActorEnsemble":
        artifact = Path(path)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        with np.load(artifact, allow_pickle=False) as saved:
            model_version = str(saved["model_version"].item())
            default_policy_names = {
                LEGACY_MODEL_VERSION: "v4 residual",
                V8_MODEL_VERSION: "v8 no-calendar",
                MODEL_VERSION: "v10 macro (leveraged)",
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
                residual_multipliers=(
                    tuple(saved["residual_multipliers"].astype(float).tolist())
                    if "residual_multipliers" in saved.files
                    else (0.5, 1.0, 1.5)
                ),
                vt_target=(
                    float(saved["vt_target"])
                    if "vt_target" in saved.files
                    else 0.10
                ),
                max_exposure=(
                    float(saved["max_exposure"])
                    if "max_exposure" in saved.files
                    else 1.0
                ),
            )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        seeds = self.ensemble_size
        features = len(self.feature_names)
        actions = len(self.residual_multipliers)
        expected = {
            "normalizer_mean": (features,),
            "normalizer_std": (features,),
            "layer1_weight": (seeds, 64, features + 2),
            "layer1_bias": (seeds, 64),
            "layer2_weight": (seeds, 64, 64),
            "layer2_bias": (seeds, 64),
            "action_weight": (seeds, actions, 64),
            "action_bias": (seeds, actions),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}; expected {shape}")
        if not self.model_version or not self.policy_name or not self.train_cutoff:
            raise ValueError("Actor bundle metadata is incomplete")
        supported = set(FEATURE_NAMES) | set(CROSS_ASSET_FEATURE_NAMES)
        unknown = set(self.feature_names).difference(supported)
        if unknown:
            raise ValueError(f"Actor bundle has unsupported features: {sorted(unknown)}")
        expected_order = tuple(
            name
            for name in FEATURE_NAMES + CROSS_ASSET_FEATURE_NAMES
            if name in set(self.feature_names)
        )
        if len(set(self.feature_names)) != features:
            raise ValueError("Actor feature names must be unique")
        if tuple(self.feature_names) != expected_order:
            raise ValueError("Actor feature order does not match the live pipeline")
        if np.any(self.normalizer_std <= 0):
            raise ValueError("Normalizer standard deviations must be positive")
        if not (0.0 < self.vt_target <= 0.5 and 1.0 <= self.max_exposure <= 2.0):
            raise ValueError("Residual baseline contract is out of range")
        if list(self.residual_multipliers) != sorted(self.residual_multipliers):
            raise ValueError("Residual multipliers must be ascending")

    def normalize(self, raw_features: np.ndarray) -> np.ndarray:
        values = (raw_features - self.normalizer_mean) / self.normalizer_std
        return np.clip(values, -10.0, 10.0)

    def logits(self, observations: np.ndarray) -> np.ndarray:
        """Return categorical actor logits (one observation row per actor)."""
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
        return (
            np.einsum("si,soi->so", hidden2, self.action_weight)
            + self.action_bias
        )

    def actions(self, observations: np.ndarray) -> np.ndarray:
        """Return deterministic categorical actions for one row per actor."""
        return np.argmax(self.logits(observations), axis=1)


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


def build_cross_asset_frame(
    qqq: pd.DataFrame,
    vix: pd.DataFrame,
    tnx: pd.DataFrame,
    irx: pd.DataFrame,
    spx: pd.DataFrame,
    tlt: pd.DataFrame,
    gld: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-asset macro features matching rlqqq.data.cross_asset_features.

    All causal through close of day t; early NaNs (pre-inception TLT/GLD)
    are neutral-zero exactly as in training.
    """
    index = qqq.index
    own = qqq["adj_close"]
    counterpart = spx["adj_close"].reindex(index).ffill()
    tlt_a = tlt["adj_close"].reindex(index).ffill()
    gld_a = gld["adj_close"].reindex(index).ffill()
    vix_a = vix["Close"].reindex(index).ffill()
    spx_a = counterpart

    frame = pd.DataFrame(index=index)
    ratio = own / counterpart
    frame["spx_ratio_mom_63"] = np.log(ratio / ratio.shift(63))
    frame["bond_trend"] = np.log(tlt_a / tlt_a.shift(63))
    frame["gold_trend"] = np.log(gld_a / gld_a.shift(63))
    own_r = own.pct_change()
    frame["stock_bond_corr_63"] = own_r.rolling(63).corr(tlt_a.pct_change())
    spx_r = spx_a.pct_change()
    frame["vrp_proxy"] = (vix_a / 100.0) ** 2 - spx_r.rolling(21).var() * 252
    # ^TNX and ^IRX both quote percent directly (no /10) in this frame,
    # matching rlqqq.data.cross_asset_features exactly.
    tnx_pct = tnx["Close"].reindex(index).ffill()
    irx_pct = irx["Close"].reindex(index).ffill()
    frame["curve_slope"] = tnx_pct - irx_pct
    return frame.fillna(0.0)


def build_feature_frame_v10(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """28-feature frame for the v10 macro bundle (22 core minus calendar
    is 22 -> the v10 core set, plus 6 cross-asset features)."""
    core = build_feature_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"]
    )
    cross = build_cross_asset_frame(
        frames["QQQ"], frames["^VIX"], frames["^TNX"], frames["^IRX"],
        frames["^GSPC"], frames["TLT"], frames["GLD"],
    )
    combined = core.join(cross.reindex(core.index))
    live_names = [n for n in FEATURE_NAMES if n not in {"dow", "month"}]
    return combined[live_names + CROSS_ASSET_FEATURE_NAMES].dropna()


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

    frames = {
        "QQQ": read("yf_QQQ.csv", "QQQ"),
        "^VIX": read("yf_IDX_VIX.csv", "^VIX"),
        "^TNX": read("yf_IDX_TNX.csv", "^TNX"),
        "^IRX": read("yf_IDX_IRX.csv", "^IRX"),
    }
    # cross-asset feeds for the v10 macro bundle (Volume not required)
    for symbol, filename in [("^GSPC", "yf_IDX_GSPC.csv"), ("TLT", "yf_TLT.csv"),
                             ("GLD", "yf_GLD.csv")]:
        frame = pd.read_csv(root / filename, parse_dates=["Date"]).set_index("Date")
        cleaned = _clean_provider_frame(frame, symbol)
        adjusted = "Adj Close" if "Adj Close" in cleaned.columns else "Close"
        cleaned["adj_close"] = pd.to_numeric(cleaned[adjusted], errors="coerce")
        frames[symbol] = cleaned.dropna(subset=["Close", "adj_close"])
    return frames


def fetch_yahoo_market_frames(
    retries: int = 3,
    retry_delay: float = 4.0,
) -> dict[str, pd.DataFrame]:
    """Fetch delayed end-of-day bars with yfinance.

    This feed is appropriate for the public research demo, not execution.
    """
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    for symbol in ["QQQ", "^VIX", "^TNX", "^IRX", "^GSPC", "TLT", "GLD"]:
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
                if symbol in ("^GSPC", "TLT", "GLD"):
                    cleaned = _clean_provider_frame(history, symbol)
                    adjusted = (
                        "Adj Close" if "Adj Close" in cleaned.columns else "Close"
                    )
                    cleaned["adj_close"] = pd.to_numeric(
                        cleaned[adjusted], errors="coerce"
                    )
                    frames[symbol] = cleaned.dropna(subset=["Close", "adj_close"])
                else:
                    frames[symbol] = _price_frame(history, symbol)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(retry_delay * (attempt + 1))
        else:
            raise RuntimeError(f"Could not fetch {symbol}: {last_error}")
    return frames


def validate_latest_market_frames(
    frames: Mapping[str, pd.DataFrame],
) -> pd.Timestamp:
    """Require every live input feed to cover the same completed session.

    Context features are forward-filled to reproduce training, but a current
    decision must never be produced by silently carrying an old VIX or yield
    observation into a newer QQQ close.
    """
    required = ("QQQ", "^VIX", "^TNX", "^IRX")
    missing = [symbol for symbol in required if symbol not in frames]
    if missing:
        raise ValueError(f"Market inputs are missing feeds: {missing}")

    latest: dict[str, pd.Timestamp] = {}
    for symbol in required:
        frame = frames[symbol]
        if frame.empty:
            raise ValueError(f"{symbol} market input is empty")
        date = pd.Timestamp(frame.index[-1]).tz_localize(None).normalize()
        close = float(frame["Close"].iloc[-1])
        if not np.isfinite(close):
            raise ValueError(f"{symbol} latest close is not finite")
        latest[symbol] = date

    # Feeds publish on different clocks (e.g. a VIX quote can carry the next
    # session's stamp while QQQ still ends on the prior close). The safe,
    # causal rule is to align every feed to the LATEST COMMON completed
    # session: truncate feeds that run ahead, and fail only when a feed is
    # missing that common session entirely.
    common = min(latest.values())
    for symbol in required:
        frame = frames[symbol]
        truncated = frame.loc[:common]
        if truncated.empty or pd.Timestamp(
            truncated.index[-1]
        ).tz_localize(None).normalize() != common:
            raise ValueError(
                f"{symbol} is missing the common market session {common.date()}"
            )
        if len(truncated) != len(frame):
            frame.drop(frame.index[len(truncated):], inplace=True)
    return common


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
    # the observation anchor is the bundle's own baseline; the vt10 column
    # is retained in the result for display/back-compat.
    anchor = (ensemble.vt_target / realized_vol).clip(upper=1.0)
    anchor.iloc[:21] = 0.5
    vt10 = (0.10 / realized_vol).clip(upper=1.0)
    vt10.iloc[:21] = 0.5
    if anchor.isna().any() or vt10.isna().any():
        raise ValueError("Forward volatility baseline contains missing values")

    multipliers = np.asarray(ensemble.residual_multipliers, dtype=np.float64)
    actor_exposure = np.zeros(ensemble.ensemble_size, dtype=np.float64)
    exposure_rows = np.empty((len(evaluation), ensemble.ensemble_size))
    action_rows = np.empty((len(evaluation), ensemble.ensemble_size), dtype=int)

    normalized = ensemble.normalize(
        evaluation[list(ensemble.feature_names)].to_numpy(dtype=np.float64)
    )
    for index, (feature_row, baseline) in enumerate(
        zip(normalized, anchor.to_numpy(dtype=np.float64), strict=True)
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
            baseline * multipliers[actions], 0.0, ensemble.max_exposure
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
    result["vt10_exposure"] = anchor  # the bundle's own observation anchor
    result["learned_mean"] = exposure_rows.mean(axis=1)
    result["learned_min"] = exposure_rows.min(axis=1)
    result["learned_max"] = exposure_rows.max(axis=1)
    result["tilt_multiplier"] = np.clip(
        result["learned_mean"] / result["vt10_exposure"],
        multipliers[0], multipliers[-1],
    )
    result["vt20_exposure"] = np.minimum(
        1.5, 2.0 * result["vt10_exposure"]
    )
    result["composite_exposure"] = np.minimum(
        1.5, result["tilt_multiplier"] * result["vt20_exposure"]
    )
    result["anchor_exposure"] = anchor
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


def canonical_json_bytes(payload: Mapping) -> bytes:
    """Serialize a public browser contract deterministically for hashing."""
    return (
        json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: Mapping) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_compact_json(payload: Mapping, output: str | Path) -> str:
    """Write one deterministic public JSON contract and return its digest."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def browser_feature_schema(ensemble: FrozenActorEnsemble) -> dict:
    return {
        "featureNames": list(ensemble.feature_names),
        "normalizerClip": [-10.0, 10.0],
        "observationOrder": [
            f"normalizedFeatures[{len(ensemble.feature_names)}]",
            "previousExposureForActor",
            "vt10Exposure",
        ],
    }


def build_browser_replay_payloads(
    replay: pd.DataFrame,
    ensemble: FrozenActorEnsemble,
    features: pd.DataFrame,
) -> tuple[dict, dict]:
    """Build browser inputs plus an independent NumPy replay reference.

    The browser receives raw features and the causal VT10 anchor, then must
    preserve one previous-exposure state per actor while replaying from the
    frozen activation date.  The reference payload is intentionally separate
    so a browser cannot silently display the server-computed answer as if it
    had run the actor itself.
    """
    if replay.empty:
        raise ValueError("Cannot build browser assets from an empty replay")
    actor_exposure = replay.attrs.get("actor_exposure")
    expected_actions = replay.attrs.get("actions")
    if actor_exposure is None or expected_actions is None:
        raise ValueError("Replay is missing per-actor state or actions")

    dates = pd.DatetimeIndex(replay.index)
    raw = features.reindex(dates)[list(ensemble.feature_names)].to_numpy(
        dtype=np.float64
    )
    if raw.shape != (len(replay), len(ensemble.feature_names)):
        raise ValueError("Browser feature matrix has an unexpected shape")
    if not np.isfinite(raw).all():
        raise ValueError("Browser feature matrix contains non-finite values")

    normalized = ensemble.normalize(raw)
    previous = np.vstack(
        [np.zeros((1, ensemble.ensemble_size)), actor_exposure[:-1]]
    )
    baselines = replay["vt10_exposure"].to_numpy(dtype=np.float64)
    logits_rows = np.empty(
        (len(replay), ensemble.ensemble_size, ensemble.n_actions),
        dtype=np.float64,
    )
    margins = np.empty((len(replay), ensemble.ensemble_size), dtype=np.float64)

    for index, (feature_row, baseline) in enumerate(
        zip(normalized, baselines, strict=True)
    ):
        observations = np.column_stack(
            [
                np.repeat(feature_row[None, :], ensemble.ensemble_size, axis=0),
                previous[index],
                np.full(ensemble.ensemble_size, baseline),
            ]
        )
        logits = ensemble.logits(observations)
        actions = np.argmax(logits, axis=1)
        if not np.array_equal(actions, expected_actions[index]):
            raise AssertionError(f"Browser reference action mismatch at {dates[index]}")
        logits_rows[index] = logits
        ordered = np.sort(logits, axis=1)
        margins[index] = ordered[:, -1] - ordered[:, -2]

    feature_schema = browser_feature_schema(ensemble)
    feature_schema_sha = payload_sha256(feature_schema)
    date_strings = [str(date.date()) for date in dates]
    input_payload = {
        "schemaVersion": 1,
        "modelVersion": ensemble.model_version,
        "sourceArtifactSha256": ensemble.artifact_sha256,
        "featureSchemaSha256": feature_schema_sha,
        "activationDate": str(dates[0].date()),
        "asOf": str(dates[-1].date()),
        "rowCount": len(dates),
        "dates": date_strings,
        "featureNames": list(ensemble.feature_names),
        "rawFeatures": raw.tolist(),
        "vt10Exposure": baselines.tolist(),
    }
    input_digest = payload_sha256(input_payload)

    latest = replay.iloc[-1]
    latest_actions = expected_actions[-1].astype(int)
    vote_counts = np.bincount(
        latest_actions, minlength=ensemble.n_actions
    )
    reference_payload = {
        "schemaVersion": 1,
        "modelVersion": ensemble.model_version,
        "sourceArtifactSha256": ensemble.artifact_sha256,
        "featureSchemaSha256": feature_schema_sha,
        "inputPayloadSha256": input_digest,
        "activationDate": str(dates[0].date()),
        "asOf": str(dates[-1].date()),
        "rowCount": len(dates),
        "dates": date_strings,
        "normalizedFeatures": normalized.tolist(),
        "actions": expected_actions.astype(int).tolist(),
        "logits": logits_rows.tolist(),
        "topTwoMargins": margins.tolist(),
        "actorExposure": np.asarray(actor_exposure, dtype=np.float64).tolist(),
        "latest": {
            "actions": latest_actions.tolist(),
            "voteCounts": vote_counts.astype(int).tolist(),
            "actorExposure": np.asarray(actor_exposure[-1], dtype=np.float64).tolist(),
            "actorStateSha256": actor_state_sha256(actor_exposure[-1]),
            "vt10Exposure": float(latest["vt10_exposure"]),
            "learnedMean": float(latest["learned_mean"]),
            "learnedMin": float(latest["learned_min"]),
            "learnedMax": float(latest["learned_max"]),
            "tiltMultiplier": float(latest["tilt_multiplier"]),
            "vt20Exposure": float(latest["vt20_exposure"]),
            "compositeExposure": float(latest["composite_exposure"]),
        },
    }
    return input_payload, reference_payload


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
                f"The model uses a {ensemble.train_cutoff} decision-date cutoff; "
                "its final training feature row is 2023-12-29 and the associated "
                "reward realizes on 2024-01-02."
            ),
            (
                "The composite is retained only for parity and research audit; "
                "the deployed target is the v8 core exposure."
            ),
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
