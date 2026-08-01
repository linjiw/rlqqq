#!/usr/bin/env node

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { webcrypto } from "node:crypto";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "..");
const documentRoot = resolve(repositoryRoot, "docs");
const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
};

function safePath(requestUrl) {
  const pathname = decodeURIComponent(new URL(requestUrl, "http://localhost").pathname);
  const candidate = resolve(documentRoot, `.${normalize(pathname)}`);
  if (candidate !== documentRoot && !candidate.startsWith(`${documentRoot}/`)) {
    throw new Error("Invalid path");
  }
  return candidate;
}

const server = createServer(async (request, response) => {
  try {
    let path = safePath(request.url || "/");
    if ((await stat(path)).isDirectory()) path = join(path, "index.html");
    const body = await readFile(path);
    response.writeHead(200, {
      "cache-control": "no-store",
      "content-type": contentTypes[extname(path)] || "application/octet-stream",
    });
    response.end(body);
  } catch {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
});

await new Promise((resolveListen, rejectListen) => {
  server.once("error", rejectListen);
  server.listen(0, "127.0.0.1", resolveListen);
});

const address = server.address();
const baseUrl = new URL(`http://127.0.0.1:${address.port}/`);
try {
  const liveResponse = await fetch(new URL("assets/live-signal.json", baseUrl));
  if (!liveResponse.ok) throw new Error(`Live signal HTTP ${liveResponse.status}`);
  const liveSignal = await liveResponse.json();
  const moduleUrl = pathToFileURL(
    join(documentRoot, "assets", "browser-inference.mjs"),
  );
  const { verifyBrowserPolicy } = await import(moduleUrl.href);
  const result = await verifyBrowserPolicy({
    liveSignal,
    manifestUrl: new URL("assets/models/model-manifest.json", baseUrl),
    ortAssetUrl: pathToFileURL(join(documentRoot, "assets", "ort", "1.27.0", "/")),
  });
  const tamperedSignal = structuredClone(liveSignal);
  tamperedSignal.model.artifactSha256 = "0".repeat(64);
  let failClosedVerified = false;
  try {
    await verifyBrowserPolicy({
      liveSignal: tamperedSignal,
      manifestUrl: new URL("assets/models/model-manifest.json", baseUrl),
      ortAssetUrl: pathToFileURL(join(documentRoot, "assets", "ort", "1.27.0", "/")),
    });
  } catch (error) {
    failClosedVerified = /source artifact hash/.test(String(error?.message));
  }
  if (!failClosedVerified) {
    throw new Error("Tampered live signal did not fail closed");
  }
  console.log(JSON.stringify({
    status: result.verification.status,
    modelVersion: result.model.version,
    asOf: liveSignal.asOf,
    learnedMean: result.signal.learnedMean,
    compositeExposure: result.signal.compositeExposure,
    voteCounts: result.voteCounts,
    actionMatches: result.verification.actionMatches,
    actionChecks: result.verification.actionChecks,
    maximumNormalizerError: result.verification.maximumNormalizerError,
    maximumLogitError: result.verification.maximumLogitError,
    maximumExposureError: result.verification.maximumExposureError,
    goldenMaximumLogitError: result.verification.goldenMaximumLogitError,
    minimumMargin: result.verification.minimumMargin,
    failClosedVerified,
    elapsedMilliseconds: result.verification.elapsedMilliseconds,
  }, null, 2));
} finally {
  await new Promise((resolveClose, rejectClose) => {
    server.close((error) => (error ? rejectClose(error) : resolveClose()));
  });
}
