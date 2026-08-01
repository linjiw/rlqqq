"""Generate the public latest-policy JSON and append the forward decision log."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.live import (
    MODEL_VERSION,
    FrozenActorEnsemble,
    append_forward_log,
    build_feature_frame,
    build_signal_payload,
    fetch_yahoo_market_frames,
    load_checked_market_frames,
    replay_frozen_policy,
    write_signal_json,
)

DEFAULT_MODEL = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"
DEFAULT_SIGNAL = ROOT / "docs" / "assets" / "live-signal.json"
DEFAULT_LOG = ROOT / "results" / "forward_log.csv"


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["yahoo", "checked"],
        default="yahoo",
        help="Live Yahoo feed or the checked-in reproducibility snapshot.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--forward-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--generated-at",
        help="Optional ISO-8601 timestamp for deterministic regeneration.",
    )
    args = parser.parse_args()

    ensemble = FrozenActorEnsemble.load(args.model)
    if args.provider == "checked":
        frames = load_checked_market_frames(ROOT / "data" / "raw")
        source_name = "Yahoo Finance checked snapshot"
    else:
        frames = fetch_yahoo_market_frames()
        source_name = "Yahoo Finance via yfinance"

    features = build_feature_frame(
        frames["QQQ"],
        frames["^VIX"],
        frames["^TNX"],
        frames["^IRX"],
    )
    replay = replay_frozen_policy(ensemble, features, frames["QQQ"])
    generated_at = (
        datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
        if args.generated_at
        else None
    )
    payload = build_signal_payload(
        replay,
        ensemble,
        source_name=source_name,
        generated_at=generated_at,
    )
    write_signal_json(payload, args.output)
    appended = (
        append_forward_log(payload, args.forward_log)
        if not args.no_log
        else False
    )

    market = payload["market"]
    signal = payload["signal"]
    print(
        f"{payload['asOf']} QQQ {market['price']:.2f} | "
        f"VT10 {signal['vt10Exposure']:.2f}x | "
        f"v4 {signal['learnedMean']:.2f}x "
        f"[{signal['learnedMin']:.2f}, {signal['learnedMax']:.2f}] | "
        f"{signal['stance']}"
    )
    print(f"Wrote {display_path(args.output)}")
    if not args.no_log:
        action = "Appended to" if appended else "Already present in"
        print(f"{action} {display_path(args.forward_log)}")


if __name__ == "__main__":
    main()
