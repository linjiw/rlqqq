# ONNX Runtime Web runtime

This directory vendors the WASM-only browser runtime from
`onnxruntime-web@1.27.0` under the upstream MIT license.

Published package integrity:
`sha512-ogDLsqIozHZwifPuN37OproAo0byX6t43/bP8GzeZWBWD6MOGExswFAx3up4NS/vvWBOg2u2PXomDt3rMmdQSg==`

Files used by RLQQQ:

- `ort.wasm.min.mjs` (`9608e98a3fba9716f0cbd0bca0c94808b71dae2e9265ca3fe742ddc393a53ebc`)
- `ort-wasm-simd-threaded.mjs` (`0a1e718d99c41b22c21f2520ff4f9e883a6b5533856e398d21816ee8eb8185d3`)
- `ort-wasm-simd-threaded.wasm` (`d1ab1b94b16a65b29d710d0b587b29e7bed336827577623913479b8afe8113e6`)

RLQQQ forces the WASM execution provider and one thread because GitHub Pages
does not provide cross-origin isolation headers required for shared-memory
threading. The JavaScript entry point and WASM binaries must always be updated
together from the same package version.
