import * as ort from "./ort/1.27.0/ort.wasm.min.mjs";

const ORT_VERSION = "1.27.0";
const DEFAULT_MANIFEST_URL = new URL(
  "models/model-manifest.json",
  import.meta.url,
);
const DEFAULT_ORT_ASSET_URL = new URL("ort/1.27.0/", import.meta.url);

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteNumber(value, label) {
  invariant(Number.isFinite(value), `${label} must be finite`);
}

function sameArray(left, right) {
  return (
    Array.isArray(left) &&
    Array.isArray(right) &&
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function maxAbsoluteDifference(left, right) {
  invariant(left.length === right.length, "Parity arrays have different lengths");
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
  }
  return maximum;
}

export async function sha256Hex(value) {
  const bytes = value instanceof ArrayBuffer
    ? value
    : value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
}

async function fetchBytes(url) {
  const response = await fetch(url, { cache: "no-store" });
  invariant(response.ok, `Could not load ${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

function parseJson(bytes, label) {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function strictArgmax(logits, offset) {
  let action = 0;
  let best = logits[offset];
  for (let index = 1; index < 3; index += 1) {
    const value = logits[offset + index];
    if (value > best) {
      best = value;
      action = index;
    }
  }
  return action;
}

function topTwoMargin(logits, offset) {
  const values = [logits[offset], logits[offset + 1], logits[offset + 2]];
  values.sort((left, right) => right - left);
  return values[0] - values[1];
}

function normalizedFeatures(raw, mean, standardDeviation, clip) {
  return raw.map((value, index) =>
    Math.max(
      clip[0],
      Math.min(clip[1], (value - mean[index]) / standardDeviation[index]),
    ));
}

function observationsForActors(normalized, previousExposure, vt10) {
  const featureCount = normalized.length;
  const observationSize = featureCount + 2;
  const observations = new Float64Array(previousExposure.length * observationSize);
  for (let seed = 0; seed < previousExposure.length; seed += 1) {
    const offset = seed * observationSize;
    observations.set(normalized, offset);
    observations[offset + featureCount] = previousExposure[seed];
    observations[offset + featureCount + 1] = vt10;
  }
  return observations;
}

function flattenMatrix(matrix) {
  const flattened = [];
  matrix.forEach((row) => flattened.push(...row));
  return flattened;
}

function stanceForExposure(exposure) {
  if (exposure < 0.65) return "Defensive";
  if (exposure < 0.95) return "Reduced risk";
  if (exposure < 1.2) return "Fully invested";
  return "Levered";
}

async function actorStateSha256(exposures) {
  const bytes = new ArrayBuffer(exposures.length * 8);
  const view = new DataView(bytes);
  exposures.forEach((value, index) => {
    const rounded = Math.round(value * 100000) / 100000;
    view.setFloat64(index * 8, rounded, true);
  });
  return sha256Hex(bytes);
}

function validateManifest(manifest) {
  invariant(manifest?.schemaVersion === 1, "Unknown model manifest schema");
  invariant(manifest.modelVersion, "Model manifest has no version");
  invariant(manifest.runtime?.version === ORT_VERSION, "ORT runtime version mismatch");
  invariant(manifest.runtime?.wasmThreads === 1, "Browser runtime must use one thread");
  invariant(manifest.onnx?.input?.name, "Manifest is missing the ONNX input");
  invariant(manifest.onnx?.output?.name, "Manifest is missing the ONNX output");
  invariant(
    sameArray(manifest.onnx.input.shape, [10, 24]),
    "Unexpected ONNX input shape",
  );
  invariant(
    sameArray(manifest.onnx.output.shape, [10, 3]),
    "Unexpected ONNX output shape",
  );
  invariant(manifest.onnx.input.dtype === "float64", "ONNX input must be float64");
  invariant(manifest.features?.featureNames?.length === 22, "Expected 22 raw features");
  invariant(manifest.features.normalizerMean.length === 22, "Normalizer mean mismatch");
  invariant(manifest.features.normalizerStd.length === 22, "Normalizer std mismatch");
  manifest.features.normalizerStd.forEach((value, index) => {
    finiteNumber(value, `normalizerStd[${index}]`);
    invariant(value > 0, `normalizerStd[${index}] must be positive`);
  });
  invariant(
    sameArray(manifest.ensemble?.residualMultipliers, [0.5, 1, 1.5]),
    "Residual multiplier contract mismatch",
  );
  invariant(
    manifest.ensemble?.initialActorExposure?.length === 10,
    "Initial actor state must contain ten values",
  );
}

function validateReplayContracts(manifest, inputs, reference, liveSignal) {
  [inputs, reference].forEach((payload) => {
    invariant(payload?.schemaVersion === 1, "Unknown replay payload schema");
    invariant(
      payload.modelVersion === manifest.modelVersion,
      "Replay model version does not match the manifest",
    );
    invariant(
      payload.sourceArtifactSha256 === manifest.sourceArtifact.sha256,
      "Replay source artifact hash does not match the manifest",
    );
    invariant(
      payload.featureSchemaSha256 === manifest.features.schemaSha256,
      "Replay feature schema does not match the manifest",
    );
  });
  invariant(
    liveSignal?.model?.version === manifest.modelVersion,
    "Live signal model version does not match the manifest",
  );
  invariant(
    liveSignal.model.artifactSha256 === manifest.sourceArtifact.sha256,
    "Live signal source artifact hash does not match the manifest",
  );
  invariant(liveSignal.stale !== true, "The latest market signal is stale");
  invariant(inputs.asOf === reference.asOf, "Replay input/reference dates differ");
  invariant(inputs.asOf === liveSignal.asOf, "Replay is not current with the market signal");
  invariant(
    inputs.activationDate === manifest.replay.activationDate,
    "Replay activation date does not match the manifest",
  );
  invariant(
    inputs.rowCount === inputs.dates.length && inputs.rowCount >= 22,
    "Replay input row count is invalid",
  );
  invariant(
    reference.rowCount === inputs.rowCount &&
      reference.dates.length === inputs.rowCount,
    "Python reference row count is invalid",
  );
  invariant(
    sameArray(inputs.featureNames, manifest.features.featureNames),
    "Replay features are out of order",
  );
  for (let index = 0; index < inputs.rowCount; index += 1) {
    invariant(inputs.dates[index] === reference.dates[index], "Replay dates differ");
    if (index > 0) {
      invariant(inputs.dates[index] > inputs.dates[index - 1], "Replay dates are not sorted");
    }
    const raw = inputs.rawFeatures[index];
    invariant(Array.isArray(raw) && raw.length === 22, "A raw feature row is invalid");
    raw.forEach((value, feature) => finiteNumber(value, `rawFeatures[${index}][${feature}]`));
    const vt10 = inputs.vt10Exposure[index];
    finiteNumber(vt10, `vt10Exposure[${index}]`);
    invariant(vt10 > 0 && vt10 <= 1, `vt10Exposure[${index}] is out of range`);
    invariant(reference.actions[index]?.length === 10, "Reference action row is invalid");
    invariant(reference.logits[index]?.length === 10, "Reference logits row is invalid");
    invariant(
      reference.actorExposure[index]?.length === 10,
      "Reference exposure row is invalid",
    );
  }
  const marketAgeDays =
    (Date.now() - new Date(`${liveSignal.asOf}T23:59:59Z`).getTime()) / 86_400_000;
  invariant(marketAgeDays <= 6, "The latest completed market session is too old");
}

async function runGoldenVectors(session, manifest, golden) {
  invariant(golden?.schemaVersion === 1, "Unknown golden-vector schema");
  invariant(golden.modelVersion === manifest.modelVersion, "Golden model mismatch");
  invariant(
    golden.sourceArtifactSha256 === manifest.sourceArtifact.sha256,
    "Golden source artifact mismatch",
  );
  invariant(golden.onnxArtifactSha256 === manifest.onnx.sha256, "Golden ONNX mismatch");
  let maximumLogitError = 0;
  let actionMatches = 0;
  for (const vector of golden.vectors) {
    invariant(vector.observations.length === 10, "Golden observation shape mismatch");
    const observations = Float64Array.from(flattenMatrix(vector.observations));
    const feeds = {
      [manifest.onnx.input.name]: new ort.Tensor(
        "float64",
        observations,
        manifest.onnx.input.shape,
      ),
    };
    const result = await session.run(feeds);
    const logits = result[manifest.onnx.output.name].data;
    const expected = flattenMatrix(vector.expectedLogits);
    maximumLogitError = Math.max(
      maximumLogitError,
      maxAbsoluteDifference(logits, expected),
    );
    for (let seed = 0; seed < 10; seed += 1) {
      const action = strictArgmax(logits, seed * 3);
      invariant(action === vector.expectedActions[seed], "Golden action mismatch");
      actionMatches += 1;
    }
  }
  invariant(
    maximumLogitError <= golden.logitAbsoluteTolerance,
    `Golden logits exceeded tolerance: ${maximumLogitError}`,
  );
  return { maximumLogitError, actionMatches };
}

async function replayAndVerify(session, manifest, inputs, reference, liveSignal) {
  const started = performance.now();
  const multipliers = manifest.ensemble.residualMultipliers;
  const exposureTolerance = manifest.validation.browserExposureAbsoluteTolerance;
  const logitTolerance = manifest.validation.browserLogitAbsoluteTolerance;
  const normalizerTolerance = manifest.validation.browserNormalizerAbsoluteTolerance;
  let previousExposure = [...manifest.ensemble.initialActorExposure];
  let maximumNormalizerError = 0;
  let maximumLogitError = 0;
  let maximumExposureError = 0;
  let minimumMargin = Infinity;
  let latestLogits = null;
  let latestActions = null;
  let latestMargins = null;
  let actionMatches = 0;

  for (let row = 0; row < inputs.rowCount; row += 1) {
    const normalized = normalizedFeatures(
      inputs.rawFeatures[row],
      manifest.features.normalizerMean,
      manifest.features.normalizerStd,
      manifest.features.normalizerClip,
    );
    maximumNormalizerError = Math.max(
      maximumNormalizerError,
      maxAbsoluteDifference(normalized, reference.normalizedFeatures[row]),
    );
    const observations = observationsForActors(
      normalized,
      previousExposure,
      inputs.vt10Exposure[row],
    );
    const result = await session.run({
      [manifest.onnx.input.name]: new ort.Tensor(
        "float64",
        observations,
        manifest.onnx.input.shape,
      ),
    });
    const logits = result[manifest.onnx.output.name].data;
    const expectedLogits = flattenMatrix(reference.logits[row]);
    maximumLogitError = Math.max(
      maximumLogitError,
      maxAbsoluteDifference(logits, expectedLogits),
    );
    const actions = [];
    const margins = [];
    const nextExposure = [];
    for (let seed = 0; seed < 10; seed += 1) {
      const offset = seed * 3;
      const action = strictArgmax(logits, offset);
      const margin = topTwoMargin(logits, offset);
      invariant(action === reference.actions[row][seed], `Action mismatch at ${inputs.dates[row]}`);
      actionMatches += 1;
      actions.push(action);
      margins.push(margin);
      minimumMargin = Math.min(minimumMargin, margin);
      nextExposure.push(
        Math.min(
          manifest.ensemble.maximumCoreExposure,
          inputs.vt10Exposure[row] * multipliers[action],
        ),
      );
    }
    maximumExposureError = Math.max(
      maximumExposureError,
      maxAbsoluteDifference(nextExposure, reference.actorExposure[row]),
    );
    previousExposure = nextExposure;
    if (row === inputs.rowCount - 1) {
      latestLogits = Array.from(logits);
      latestActions = actions;
      latestMargins = margins;
    }
  }

  invariant(maximumNormalizerError <= normalizerTolerance, "Browser normalizer mismatch");
  invariant(maximumLogitError <= logitTolerance, "Browser logits mismatch");
  invariant(maximumExposureError <= exposureTolerance, "Browser exposure mismatch");
  const actorHash = await actorStateSha256(previousExposure);
  invariant(actorHash === reference.latest.actorStateSha256, "Final actor state hash mismatch");

  const learnedMean = previousExposure.reduce((sum, value) => sum + value, 0) / 10;
  const learnedMin = Math.min(...previousExposure);
  const learnedMax = Math.max(...previousExposure);
  const vt10Exposure = inputs.vt10Exposure.at(-1);
  const tiltMultiplier = Math.max(0.5, Math.min(1.5, learnedMean / vt10Exposure));
  const vt20Exposure = Math.min(1.5, 2 * vt10Exposure);
  const compositeExposure = Math.min(1.5, tiltMultiplier * vt20Exposure);
  const latest = {
    vt10Exposure,
    learnedMean,
    learnedMin,
    learnedMax,
    tiltMultiplier,
    vt20Exposure,
    compositeExposure,
    stance: stanceForExposure(learnedMean),
  };
  const referenceValues = [
    "vt10Exposure",
    "learnedMean",
    "learnedMin",
    "learnedMax",
    "tiltMultiplier",
    "vt20Exposure",
    "compositeExposure",
  ];
  for (const key of referenceValues) {
    invariant(
      Math.abs(latest[key] - reference.latest[key]) <= exposureTolerance,
      `Latest Python reference mismatch for ${key}`,
    );
    invariant(
      Math.abs(latest[key] - liveSignal.signal[key]) <= Math.max(1e-5, exposureTolerance),
      `Latest static signal mismatch for ${key}`,
    );
  }
  const voteCounts = [0, 0, 0];
  latestActions.forEach((action) => { voteCounts[action] += 1; });
  invariant(sameArray(voteCounts, reference.latest.voteCounts), "Vote count mismatch");

  return {
    signal: latest,
    actions: latestActions,
    actorExposure: previousExposure,
    logits: latestLogits,
    margins: latestMargins,
    voteCounts,
    actorStateSha256: actorHash,
    verification: {
      status: "verified",
      rowCount: inputs.rowCount,
      actionMatches,
      actionChecks: inputs.rowCount * 10,
      maximumNormalizerError,
      maximumLogitError,
      maximumExposureError,
      minimumMargin,
      latestMinimumMargin: Math.min(...latestMargins),
      elapsedMilliseconds: performance.now() - started,
    },
  };
}

export async function verifyBrowserPolicy(options = {}) {
  const manifestUrl = new URL(options.manifestUrl || DEFAULT_MANIFEST_URL, import.meta.url);
  const ortAssetUrl = new URL(options.ortAssetUrl || DEFAULT_ORT_ASSET_URL, import.meta.url);
  const liveSignal = options.liveSignal;
  invariant(liveSignal, "A Python live reference is required");

  const manifestBytes = await fetchBytes(manifestUrl);
  const manifest = parseJson(manifestBytes, "Model manifest");
  validateManifest(manifest);
  const modelUrl = new URL(manifest.onnx.path, manifestUrl);
  const inputUrl = new URL(manifest.replay.inputHistoryPath, manifestUrl);
  const referenceUrl = new URL(manifest.replay.pythonReferencePath, manifestUrl);
  const goldenUrl = new URL(manifest.replay.goldenVectorsPath, manifestUrl);
  const [modelBytes, inputBytes, referenceBytes, goldenBytes] = await Promise.all([
    fetchBytes(modelUrl),
    fetchBytes(inputUrl),
    fetchBytes(referenceUrl),
    fetchBytes(goldenUrl),
  ]);
  const [modelDigest, inputDigest, goldenDigest] = await Promise.all([
    sha256Hex(modelBytes),
    sha256Hex(inputBytes),
    sha256Hex(goldenBytes),
  ]);
  invariant(modelDigest === manifest.onnx.sha256, "Downloaded ONNX hash mismatch");
  invariant(goldenDigest === manifest.replay.goldenVectorsSha256, "Golden-vector hash mismatch");

  const inputs = parseJson(inputBytes, "Policy input history");
  const reference = parseJson(referenceBytes, "Python reference");
  const golden = parseJson(goldenBytes, "Golden vectors");
  invariant(inputDigest === reference.inputPayloadSha256, "Input-history hash mismatch");
  validateReplayContracts(manifest, inputs, reference, liveSignal);

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = ortAssetUrl.href;
  invariant(
    ort.env.versions?.web === ORT_VERSION,
    `Loaded ONNX Runtime Web ${ort.env.versions?.web || "unknown"}, expected ${ORT_VERSION}`,
  );
  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  let goldenResult;
  let replayResult;
  try {
    goldenResult = await runGoldenVectors(session, manifest, golden);
    replayResult = await replayAndVerify(
      session,
      manifest,
      inputs,
      reference,
      liveSignal,
    );
  } finally {
    await session.release();
  }
  return {
    ...replayResult,
    model: {
      version: manifest.modelVersion,
      sourceArtifactSha256: manifest.sourceArtifact.sha256,
      onnxArtifactSha256: manifest.onnx.sha256,
      onnxBytes: manifest.onnx.bytes,
      runtimeVersion: ORT_VERSION,
    },
    verification: {
      ...replayResult.verification,
      modelHashVerified: true,
      goldenActionMatches: goldenResult.actionMatches,
      goldenMaximumLogitError: goldenResult.maximumLogitError,
    },
  };
}
