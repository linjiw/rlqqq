"""Generate the public latest-policy JSON and append the forward decision log."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.live import (
    MODEL_VERSION,
    FrozenActorEnsemble,
    append_forward_log,
    build_browser_replay_payloads,
    build_feature_frame_v10,
    build_signal_payload,
    fetch_yahoo_market_frames,
    load_checked_market_frames,
    replay_frozen_policy,
    validate_latest_market_frames,
    validate_forward_log,
    write_compact_json,
    write_signal_json,
)

DEFAULT_MODEL = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"
DEFAULT_SIGNAL = ROOT / "docs" / "assets" / "live-signal.json"
DEFAULT_LOG = ROOT / "results" / "forward_log.csv"
DEFAULT_BROWSER_INPUT = (
    ROOT / "docs" / "assets" / "data" / "policy-input-history.json"
)
DEFAULT_BROWSER_REFERENCE = (
    ROOT / "docs" / "assets" / "data" / "python-reference.json"
)


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
    parser.add_argument(
        "--browser-input-output",
        type=Path,
        default=DEFAULT_BROWSER_INPUT,
    )
    parser.add_argument(
        "--browser-reference-output",
        type=Path,
        default=DEFAULT_BROWSER_REFERENCE,
    )
    parser.add_argument("--no-browser-assets", action="store_true")
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

    latest_market_date = validate_latest_market_frames(frames)

    features = build_feature_frame_v10(frames)
    if features.empty or pd.Timestamp(features.index[-1]) != latest_market_date:
        feature_date = "missing" if features.empty else str(features.index[-1].date())
        raise ValueError(
            "Latest complete feature row does not match the market session: "
            f"features={feature_date}, market={latest_market_date.date()}"
        )
    replay = replay_frozen_policy(ensemble, features, frames["QQQ"])
    if not args.no_log:
        validate_forward_log(replay, args.forward_log)
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
    browser_files: list[Path] = []
    if not args.no_browser_assets:
        browser_input, browser_reference = build_browser_replay_payloads(
            replay,
            ensemble,
            features,
        )
        write_compact_json(browser_input, args.browser_input_output)
        write_compact_json(browser_reference, args.browser_reference_output)
        browser_files = [
            args.browser_input_output,
            args.browser_reference_output,
        ]
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
        f"{payload['model']['displayName']} {signal['learnedMean']:.2f}x "
        f"[{signal['learnedMin']:.2f}, {signal['learnedMax']:.2f}] | "
        f"{signal['stance']}"
    )
    print(f"Wrote {display_path(args.output)}")
    for browser_file in browser_files:
        print(f"Wrote {display_path(browser_file)}")
    if not args.no_log:
        action = "Appended to" if appended else "Already present in"
        print(f"{action} {display_path(args.forward_log)}")


if __name__ == "__main__":
    main()
