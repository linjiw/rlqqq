"""Export the frozen ten-actor policy to a browser-verifiable ONNX bundle.

The source ``.npz`` already contains every deterministic actor layer and the
shared train-window normalizer.  SB3 checkpoints are therefore unnecessary:
this script builds a logits-only ONNX graph directly from the released actor
weights and emits immutable metadata plus golden parity vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlqqq.live import (  # noqa: E402
    MODEL_VERSION,
    FrozenActorEnsemble,
    browser_feature_schema,
    build_feature_frame_v10,
    canonical_json_bytes,
    load_checked_market_frames,
    payload_sha256,
    replay_frozen_policy,
)

DEFAULT_SOURCE = ROOT / "models" / "live" / f"{MODEL_VERSION}.npz"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "models"
ORT_WEB_VERSION = "1.27.0"
OPSET = 17


def build_onnx(ensemble: FrozenActorEnsemble) -> bytes:
    seeds = ensemble.ensemble_size
    observation_size = len(ensemble.feature_names) + 2
    inputs = [
        helper.make_tensor_value_info(
            "observations", TensorProto.DOUBLE, [seeds, observation_size]
        )
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.DOUBLE, [seeds, ensemble.n_actions])
    ]
    nodes = []
    initializers = []
    actor_outputs = []

    for seed in range(seeds):
        prefix = f"actor_{seed}"
        index_name = f"{prefix}_index"
        initializers.append(
            numpy_helper.from_array(np.array([seed], dtype=np.int64), index_name)
        )
        nodes.append(
            helper.make_node(
                "Gather",
                ["observations", index_name],
                [f"{prefix}_observation"],
                axis=0,
                name=f"{prefix}_gather",
            )
        )

        layer_specs = [
            (
                "layer1",
                ensemble.layer1_weight[seed],
                ensemble.layer1_bias[seed],
                "Tanh",
            ),
            (
                "layer2",
                ensemble.layer2_weight[seed],
                ensemble.layer2_bias[seed],
                "Tanh",
            ),
            (
                "action",
                ensemble.action_weight[seed],
                ensemble.action_bias[seed],
                None,
            ),
        ]
        current = f"{prefix}_observation"
        for layer_name, weight, bias, activation in layer_specs:
            weight_name = f"{prefix}_{layer_name}_weight"
            bias_name = f"{prefix}_{layer_name}_bias"
            affine = f"{prefix}_{layer_name}_affine"
            initializers.extend(
                [
                    numpy_helper.from_array(
                        np.asarray(weight, dtype=np.float64).T,
                        weight_name,
                    ),
                    numpy_helper.from_array(
                        np.asarray(bias, dtype=np.float64),
                        bias_name,
                    ),
                ]
            )
            nodes.append(
                helper.make_node(
                    "Gemm",
                    [current, weight_name, bias_name],
                    [affine],
                    name=f"{prefix}_{layer_name}_gemm",
                )
            )
            if activation:
                activated = f"{prefix}_{layer_name}_output"
                nodes.append(
                    helper.make_node(
                        activation,
                        [affine],
                        [activated],
                        name=f"{prefix}_{layer_name}_{activation.lower()}",
                    )
                )
                current = activated
            else:
                current = affine
        actor_outputs.append(current)

    nodes.append(
        helper.make_node(
            "Concat",
            actor_outputs,
            ["logits"],
            axis=0,
            name="ensemble_logits",
        )
    )
    graph = helper.make_graph(
        nodes,
        "rlqqq_v10_ten_actor_logits",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="rlqqq",
        producer_version="1",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = (
        "Frozen RLQQQ v10 ten-seed actor logits. Normalization and recursive "
        "per-actor exposure state are supplied by the browser contract."
    )
    model.metadata_props.add(key="model_version", value=ensemble.model_version)
    model.metadata_props.add(
        key="source_artifact_sha256", value=ensemble.artifact_sha256
    )
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    return model.SerializeToString(deterministic=True)


def observations_for_row(
    ensemble: FrozenActorEnsemble,
    normalized: np.ndarray,
    previous_exposure: np.ndarray,
    vt10: float,
) -> np.ndarray:
    return np.column_stack(
        [
            np.repeat(normalized[None, :], ensemble.ensemble_size, axis=0),
            previous_exposure,
            np.full(ensemble.ensemble_size, vt10),
        ]
    ).astype(np.float64)


def build_golden_vectors(
    ensemble: FrozenActorEnsemble,
    model_bytes: bytes,
    onnx_digest: str,
) -> tuple[dict, float]:
    frames = load_checked_market_frames(ROOT / "data" / "raw")
    features = build_feature_frame_v10(frames)
    replay = replay_frozen_policy(ensemble, features, frames["QQQ"])
    actor_exposure = replay.attrs["actor_exposure"]
    actions = replay.attrs["actions"]
    raw = features.reindex(replay.index)[list(ensemble.feature_names)].to_numpy(
        dtype=np.float64
    )
    normalized = ensemble.normalize(raw)
    previous = np.vstack(
        [np.zeros((1, ensemble.ensemble_size)), actor_exposure[:-1]]
    )
    candidates = [0, 20, 21, len(replay) // 2, len(replay) - 1]
    indexes = list(dict.fromkeys(candidates))

    session = ort.InferenceSession(
        model_bytes,
        providers=["CPUExecutionProvider"],
    )
    vectors = []
    max_error = 0.0
    for index in indexes:
        observations = observations_for_row(
            ensemble,
            normalized[index],
            previous[index],
            float(replay["vt10_exposure"].iloc[index]),
        )
        onnx_logits = session.run(
            ["logits"], {"observations": observations}
        )[0]
        numpy_logits = ensemble.logits(observations.astype(np.float64))
        error = float(np.max(np.abs(onnx_logits - numpy_logits)))
        max_error = max(max_error, error)
        onnx_actions = np.argmax(onnx_logits, axis=1)
        if not np.array_equal(onnx_actions, actions[index]):
            raise AssertionError(
                f"ONNX action mismatch on {replay.index[index].date()}"
            )
        ordered = np.sort(onnx_logits, axis=1)
        vectors.append(
            {
                "date": str(replay.index[index].date()),
                "observations": observations.tolist(),
                "expectedLogits": onnx_logits.astype(np.float64).tolist(),
                "expectedActions": onnx_actions.astype(int).tolist(),
                "topTwoMargins": (
                    ordered[:, -1] - ordered[:, -2]
                ).astype(np.float64).tolist(),
            }
        )

    payload = {
        "schemaVersion": 1,
        "modelVersion": ensemble.model_version,
        "sourceArtifactSha256": ensemble.artifact_sha256,
        "onnxArtifactSha256": onnx_digest,
        "inputName": "observations",
        "outputName": "logits",
        "logitAbsoluteTolerance": 1e-5,
        "vectors": vectors,
    }
    return payload, max_error


def build_manifest(
    ensemble: FrozenActorEnsemble,
    onnx_name: str,
    onnx_digest: str,
    onnx_size: int,
    golden_digest: str,
    max_error: float,
) -> dict:
    feature_schema = browser_feature_schema(ensemble)
    parameter_count = int(
        ensemble.layer1_weight.size
        + ensemble.layer1_bias.size
        + ensemble.layer2_weight.size
        + ensemble.layer2_bias.size
        + ensemble.action_weight.size
        + ensemble.action_bias.size
    )
    return {
        "schemaVersion": 1,
        "modelVersion": ensemble.model_version,
        "displayName": ensemble.policy_name,
        "sourceArtifact": {
            "path": "models/live/" + f"{MODEL_VERSION}.npz",
            "sha256": ensemble.artifact_sha256,
            "trainCutoff": ensemble.train_cutoff,
        },
        "onnx": {
            "path": onnx_name,
            "sha256": onnx_digest,
            "bytes": onnx_size,
            "opset": OPSET,
            "input": {
                "name": "observations",
                "dtype": "float64",
                "shape": [ensemble.ensemble_size, len(ensemble.feature_names) + 2],
            },
            "output": {
                "name": "logits",
                "dtype": "float64",
                "shape": [ensemble.ensemble_size, ensemble.n_actions],
            },
        },
        "runtime": {
            "name": "onnxruntime-web",
            "version": ORT_WEB_VERSION,
            "executionProviders": ["wasm"],
            "wasmThreads": 1,
            "quantized": False,
        },
        "ensemble": {
            "seeds": list(range(ensemble.ensemble_size)),
            "parameterCount": parameter_count,
            "residualMultipliers": list(ensemble.residual_multipliers),
            "maximumCoreExposure": ensemble.max_exposure,
            "initialActorExposure": [0.0] * ensemble.ensemble_size,
        },
        "features": {
            **feature_schema,
            "schemaSha256": payload_sha256(feature_schema),
            "normalizerMean": ensemble.normalizer_mean.tolist(),
            "normalizerStd": ensemble.normalizer_std.tolist(),
        },
        "replay": {
            "activationDate": "2026-01-02",
            "inputHistoryPath": "../data/policy-input-history.json",
            "pythonReferencePath": "../data/python-reference.json",
            "goldenVectorsPath": "golden-vectors.json",
            "goldenVectorsSha256": golden_digest,
        },
        "validation": {
            "pythonOnnxMaxAbsoluteLogitError": max_error,
            "requiredActionMatch": 1.0,
            "browserNormalizerAbsoluteTolerance": 1e-6,
            "browserLogitAbsoluteTolerance": 1e-5,
            "browserExposureAbsoluteTolerance": 1e-6,
            "failClosed": True,
        },
    }


def check_bytes(path: Path, expected: bytes) -> None:
    if not path.exists():
        raise SystemExit(f"Missing generated browser artifact: {path}")
    actual = path.read_bytes()
    if actual != expected:
        raise SystemExit(f"Stale generated browser artifact: {path}")


def load_canonical_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing generated browser artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid {label}: {path}: {exc}") from exc
    check_bytes(path, canonical_json_bytes(payload))
    return payload


def verify_existing_bundle(
    ensemble: FrozenActorEnsemble,
    model_bytes: bytes,
    onnx_path: Path,
    onnx_name: str,
    onnx_digest: str,
    golden_path: Path,
    manifest_path: Path,
) -> float:
    """Verify frozen release bytes without regenerating CPU-dependent logits.

    Tanh implementations can differ by a few ULPs across macOS/arm64 and
    Linux/x86_64.  The released golden numbers are therefore immutable test
    data: CI checks their hash and evaluates them numerically within the
    recorded tolerance instead of requiring cross-platform byte regeneration.
    """
    check_bytes(onnx_path, model_bytes)
    golden = load_canonical_json(golden_path, "golden vectors")
    golden_digest = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    manifest = load_canonical_json(manifest_path, "model manifest")

    if golden.get("modelVersion") != ensemble.model_version:
        raise SystemExit("Golden-vector model version mismatch")
    if golden.get("sourceArtifactSha256") != ensemble.artifact_sha256:
        raise SystemExit("Golden-vector source artifact mismatch")
    if golden.get("onnxArtifactSha256") != onnx_digest:
        raise SystemExit("Golden-vector ONNX artifact mismatch")

    recorded_error = float(
        manifest.get("validation", {}).get(
            "pythonOnnxMaxAbsoluteLogitError", float("nan")
        )
    )
    if not np.isfinite(recorded_error):
        raise SystemExit("Model manifest has no finite Python/ONNX error")
    expected_manifest = build_manifest(
        ensemble,
        onnx_name,
        onnx_digest,
        len(model_bytes),
        golden_digest,
        recorded_error,
    )
    check_bytes(manifest_path, canonical_json_bytes(expected_manifest))

    tolerance = float(golden.get("logitAbsoluteTolerance", float("nan")))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise SystemExit("Golden-vector tolerance is invalid")
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    maximum_error = 0.0
    for vector in golden.get("vectors", []):
        observations = np.asarray(vector["observations"], dtype=np.float64)
        expected_logits = np.asarray(vector["expectedLogits"], dtype=np.float64)
        expected_actions = np.asarray(vector["expectedActions"], dtype=np.int64)
        expected_shape = (ensemble.ensemble_size, len(ensemble.feature_names) + 2)
        if observations.shape != expected_shape or expected_logits.shape != (
            ensemble.ensemble_size,
            ensemble.n_actions,
        ):
            raise SystemExit("Golden-vector tensor shape mismatch")
        onnx_logits = session.run(
            ["logits"], {"observations": observations}
        )[0]
        numpy_logits = ensemble.logits(observations)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(onnx_logits - expected_logits))),
            float(np.max(np.abs(numpy_logits - expected_logits))),
        )
        if not np.array_equal(np.argmax(onnx_logits, axis=1), expected_actions):
            raise SystemExit(f"ONNX golden action mismatch on {vector['date']}")
        if not np.array_equal(np.argmax(numpy_logits, axis=1), expected_actions):
            raise SystemExit(f"NumPy golden action mismatch on {vector['date']}")
    if not golden.get("vectors"):
        raise SystemExit("Golden-vector release contains no cases")
    if maximum_error > tolerance:
        raise SystemExit(
            f"Golden logits exceeded tolerance: {maximum_error} > {tolerance}"
        )
    return maximum_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    ensemble = FrozenActorEnsemble.load(args.source)
    if ensemble.model_version != MODEL_VERSION:
        raise SystemExit(
            f"Expected {MODEL_VERSION}; received {ensemble.model_version}"
        )
    model_bytes = build_onnx(ensemble)
    onnx_digest = hashlib.sha256(model_bytes).hexdigest()
    onnx_name = f"rlqqq-v10-ensemble.{onnx_digest[:12]}.onnx"
    onnx_path = args.output_dir / onnx_name
    golden_path = args.output_dir / "golden-vectors.json"
    manifest_path = args.output_dir / "model-manifest.json"
    if args.check:
        max_error = verify_existing_bundle(
            ensemble,
            model_bytes,
            onnx_path,
            onnx_name,
            onnx_digest,
            golden_path,
            manifest_path,
        )
        print(
            f"Browser policy bundle verified: {onnx_name}, "
            f"max golden error {max_error:.3g}"
        )
        return

    golden, max_error = build_golden_vectors(
        ensemble,
        model_bytes,
        onnx_digest,
    )
    golden_bytes = canonical_json_bytes(golden)
    golden_digest = hashlib.sha256(golden_bytes).hexdigest()
    manifest = build_manifest(
        ensemble,
        onnx_name,
        onnx_digest,
        len(model_bytes),
        golden_digest,
        max_error,
    )
    manifest_bytes = canonical_json_bytes(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(model_bytes)
    golden_path.write_bytes(golden_bytes)
    manifest_path.write_bytes(manifest_bytes)
    print(f"Wrote {onnx_path.relative_to(ROOT)} ({len(model_bytes):,} bytes)")
    print(f"Wrote {golden_path.relative_to(ROOT)}")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    print(f"ONNX SHA256 {onnx_digest}")
    print(f"NumPy vs ONNX max logits error {max_error:.3g}")


if __name__ == "__main__":
    main()
