from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rlqqq.live import (
    RESIDUAL_MULTIPLIERS,
    FrozenActorEnsemble,
    actor_state_sha256,
    browser_feature_schema,
    canonical_json_bytes,
    payload_sha256,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
MODELS = ASSETS / "models"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deployed_browser_model_matches_the_checked_benchmark_winner():
    web_report_path = ASSETS / "model-benchmark.json"
    research_report_path = ROOT / "results" / "model_benchmark.json"
    benchmark = read_json(web_report_path)
    manifest = read_json(MODELS / "model-manifest.json")
    signal = read_json(ASSETS / "live-signal.json")

    assert web_report_path.read_bytes() == research_report_path.read_bytes()
    assert benchmark["schemaVersion"] == 1
    assert benchmark["historicalStatus"] == "complete"
    assert benchmark["historicalWalkForward"]["provenance"]["seriesFiles"] == 243
    implementation = benchmark["evaluationImplementation"]
    assert implementation["sha256"] == file_sha256(ROOT / implementation["path"])

    historical = benchmark["historicalWalkForward"]
    assert historical["period"]["realizedEnd"] == "2026-01-02"
    assert historical["oneCloseLagSensitivity"]["qqq"] == historical["metrics"]["qqq"]
    assert "v10Core" in benchmark["frozen2026"]["oneCloseLagSensitivity"]

    selection = benchmark["deploymentSelection"]
    assert selection["winner"] == "v10Core"
    assert selection["modelVersion"] == manifest["modelVersion"]
    assert selection["modelVersion"] == signal["model"]["version"]
    assert selection["capitalDeploymentQualified"] is False
    assert set(selection["ineligibleResearchOverlays"]) == {
        "v4Composite",
        "v8Composite",
    }

    forward = benchmark["frozen2026"]
    assert forward["period"]["latestSignalScored"] is False
    # the benchmark artifact is frozen at evaluation time; the live signal
    # moves forward daily, so it may only be at-or-after the benchmark date
    assert forward["latestUnscoredSignal"]["v10"]["asOf"] <= signal["asOf"]


def test_page_exposes_only_the_benchmark_gated_v10_core_as_current():
    # public page: fail-closed live signal, honest framing, archive link
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    site_js = (ASSETS / "site.js").read_text(encoding="utf-8")
    signal = read_json(ASSETS / "live-signal.json")
    assert "verifyBrowserPolicy" in site_js
    assert "verification failed" in site_js.lower()
    assert "not investment advice" in html
    assert "research.html" in html
    assert "browser-inference.mjs" in site_js
    assert 'data-live-period="1m"' in html
    assert 'data-live-period="1y"' in html
    assert "S&amp;P 500" in html
    assert "RLQQQ portfolio status" in html
    assert "RLQQQ strategy design" in html
    assert 'id="decision-allocation"' in html
    assert 'id="live-performance-read"' in html
    assert "renderLivePerformance" in site_js
    assert "allocationSummary" in site_js
    assert "latestSignalScored" in signal["performance"]
    assert signal["performance"]["latestSignalScored"] is False
    assert signal["performance"]["through"] == signal["asOf"]
    assert signal["performance"]["unscoredSignalAsOf"] == signal["asOf"]
    assert set(signal["performance"]["periods"]) == {
        "1m",
        "3m",
        "ytd",
        "1y",
        "all",
    }

    # research console (archived audit page) keeps the full benchmark contract
    research = (ROOT / "docs" / "research.html").read_text(encoding="utf-8")
    javascript = (ASSETS / "dashboard.js").read_text(encoding="utf-8")
    assert "v10 macro core is the browser deployment winner" in research
    assert "v10 macro core is the sole deployed policy" in research
    assert 'id="benchmark"' in research
    assert "loadVerifiedDeployment" in javascript
    assert "Live model does not match the checked benchmark winner" in javascript

    historical_table = research.split(
        "Official 2010 through 2025 policy benchmark", 1
    )[1].split("</table>", 1)[0]
    forward_table = research.split(
        "Official frozen 2026 policy benchmark", 1
    )[1].split("</table>", 1)[0]
    assert "CAGR" in historical_table
    assert "Lag Sharpe" in historical_table
    assert "2026 YTD" in forward_table


def test_static_browser_bundle_is_hash_bound_to_the_frozen_actor():
    manifest_path = MODELS / "model-manifest.json"
    manifest = read_json(manifest_path)
    source_path = ROOT / manifest["sourceArtifact"]["path"]
    ensemble = FrozenActorEnsemble.load(source_path)

    assert manifest_path.read_bytes() == canonical_json_bytes(manifest)
    assert manifest["modelVersion"] == ensemble.model_version
    assert manifest["sourceArtifact"]["sha256"] == ensemble.artifact_sha256
    assert file_sha256(source_path) == ensemble.artifact_sha256

    onnx_path = MODELS / manifest["onnx"]["path"]
    assert list(MODELS.glob("*.onnx")) == [onnx_path]
    assert onnx_path.stat().st_size == manifest["onnx"]["bytes"]
    assert file_sha256(onnx_path) == manifest["onnx"]["sha256"]
    assert manifest["onnx"]["sha256"].startswith(
        manifest["onnx"]["path"].split(".")[-2]
    )

    golden_path = MODELS / manifest["replay"]["goldenVectorsPath"]
    golden = read_json(golden_path)
    assert golden_path.read_bytes() == canonical_json_bytes(golden)
    assert file_sha256(golden_path) == manifest["replay"]["goldenVectorsSha256"]
    assert golden["modelVersion"] == ensemble.model_version
    assert golden["sourceArtifactSha256"] == ensemble.artifact_sha256
    assert golden["onnxArtifactSha256"] == manifest["onnx"]["sha256"]

    schema = browser_feature_schema(ensemble)
    assert manifest["features"]["schemaSha256"] == payload_sha256(schema)
    for key, value in schema.items():
        assert manifest["features"][key] == value
    assert manifest["features"]["normalizerMean"] == ensemble.normalizer_mean.tolist()
    assert manifest["features"]["normalizerStd"] == ensemble.normalizer_std.tolist()

    runtime = manifest["runtime"]
    runtime_dir = ASSETS / "ort" / runtime["version"]
    assert runtime["executionProviders"] == ["wasm"]
    assert runtime["wasmThreads"] == 1
    assert manifest["validation"]["failClosed"] is True
    for filename in (
        "ort.wasm.min.mjs",
        "ort-wasm-simd-threaded.mjs",
        "ort-wasm-simd-threaded.wasm",
    ):
        assert (runtime_dir / filename).is_file()


def test_published_replay_is_hash_bound_and_recursively_reproducible():
    manifest = read_json(MODELS / "model-manifest.json")
    input_path = ASSETS / "data" / "policy-input-history.json"
    reference_path = ASSETS / "data" / "python-reference.json"
    signal = read_json(ASSETS / "live-signal.json")
    replay_input = read_json(input_path)
    reference = read_json(reference_path)
    ensemble = FrozenActorEnsemble.load(
        ROOT / manifest["sourceArtifact"]["path"]
    )

    assert input_path.read_bytes() == canonical_json_bytes(replay_input)
    assert reference_path.read_bytes() == canonical_json_bytes(reference)
    assert reference["inputPayloadSha256"] == file_sha256(input_path)
    assert reference["inputPayloadSha256"] == payload_sha256(replay_input)

    shared = (
        "schemaVersion",
        "modelVersion",
        "sourceArtifactSha256",
        "featureSchemaSha256",
        "activationDate",
        "asOf",
        "rowCount",
    )
    for key in shared:
        assert reference[key] == replay_input[key]
    assert replay_input["modelVersion"] == manifest["modelVersion"]
    assert replay_input["sourceArtifactSha256"] == ensemble.artifact_sha256
    assert replay_input["featureSchemaSha256"] == manifest["features"][
        "schemaSha256"
    ]
    assert replay_input["featureNames"] == list(ensemble.feature_names)
    assert replay_input["dates"] == reference["dates"]

    row_count = replay_input["rowCount"]
    actor_count = ensemble.ensemble_size
    raw = np.asarray(replay_input["rawFeatures"], dtype=np.float64)
    baselines = np.asarray(replay_input["vt10Exposure"], dtype=np.float64)
    expected_normalized = np.asarray(
        reference["normalizedFeatures"], dtype=np.float64
    )
    expected_logits = np.asarray(reference["logits"], dtype=np.float64)
    expected_actions = np.asarray(reference["actions"], dtype=np.int64)
    expected_exposure = np.asarray(reference["actorExposure"], dtype=np.float64)

    assert raw.shape == (row_count, len(ensemble.feature_names))
    assert baselines.shape == (row_count,)
    assert expected_logits.shape == (row_count, actor_count, len(ensemble.residual_multipliers))
    np.testing.assert_allclose(
        ensemble.normalize(raw), expected_normalized, rtol=0, atol=0
    )

    previous = np.zeros(actor_count, dtype=np.float64)
    for index in range(row_count):
        observations = np.column_stack(
            [
                np.repeat(expected_normalized[index][None, :], actor_count, axis=0),
                previous,
                np.full(actor_count, baselines[index]),
            ]
        )
        logits = ensemble.logits(observations)
        actions = np.argmax(logits, axis=1)
        bundle_multipliers = np.asarray(ensemble.residual_multipliers)
        exposure = np.clip(
            baselines[index] * bundle_multipliers[actions],
            0.0,
            ensemble.max_exposure,
        )
        # libm tanh can differ by a few ULPs across runner architectures.
        np.testing.assert_allclose(
            logits, expected_logits[index], rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(actions, expected_actions[index])
        np.testing.assert_allclose(
            exposure, expected_exposure[index], rtol=0, atol=0
        )
        previous = exposure

    latest = reference["latest"]
    assert latest["actions"] == expected_actions[-1].tolist()
    assert latest["actorExposure"] == expected_exposure[-1].tolist()
    assert latest["actorStateSha256"] == actor_state_sha256(expected_exposure[-1])
    assert latest["voteCounts"] == np.bincount(
        expected_actions[-1], minlength=len(ensemble.residual_multipliers)
    ).tolist()
    assert latest["learnedMean"] == pytest.approx(float(previous.mean()))
    assert signal["asOf"] == reference["asOf"]
    assert signal["model"]["artifactSha256"] == ensemble.artifact_sha256
    assert signal["signal"]["actorStateSha256"] == latest["actorStateSha256"]
    for key in (
        "vt10Exposure",
        "learnedMean",
        "learnedMin",
        "learnedMax",
        "tiltMultiplier",
        "vt20Exposure",
        "compositeExposure",
    ):
        assert signal["signal"][key] == pytest.approx(latest[key], abs=5e-6)
