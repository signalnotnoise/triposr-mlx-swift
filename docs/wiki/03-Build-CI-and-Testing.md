# Build, CI, and testing

Ranges are checked by content hash, not by commit ancestry. Router: [AGENTS.md](../../AGENTS.md).

## Package pin

`Package.swift:1` declares Swift tools 6.2. Platforms are macOS 14, iOS 17, and visionOS 1 (`Package.swift:9-13`). Products are library `TripoSRRuntime` and executable `TripoSRCLI` (`Package.swift:14-22`).

mlx-swift is `0.31.3..<0.31.5` (`Package.swift:24-28`). 0.31.5 and later require Swift tools 6.3. Do not resolve past 0.31.4 on this manifest. The M3 Max note says that machine's `xcodebuild` resolved 0.31.4 (`README.md:117`).

## Metal library

Do not use `swift build` or `swift test` for this package. mlx-swift compiles Metal shaders and bundles `default.metallib` through Xcode. CPU tests still load that library. The runtime tests set `Device.setDefault(device: Device(.cpu))` in `init` (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift#TripoSRRuntimeTests.init`).

```bash
xcodebuild build -scheme TripoSRCLI -destination 'platform=macOS' -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
xcodebuild test -scheme TripoSR-Package -destination 'platform=macOS' -skipPackagePluginValidation -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO
```

The same commands are in `README.md:100-113`. CI uses them with an explicit derived-data path (`.github/workflows/ci.yml:42-58`).

## CI

The Metal job is named `macOS build and test` and runs on `macos-26` (`.github/workflows/ci.yml:16-18`). That job, through `.github/workflows/ci.yml:58`, shows Xcode and Swift, ensures the Metal toolchain, lists schemes, builds `TripoSRCLI`, then tests `TripoSR-Package`. A later Ubuntu job runs `python3 scripts/verify_docs.py`, then copies the tree into a fresh one-commit repository and runs the verifier again so a squash merge cannot revive an ancestry check.

A separate Ubuntu job runs `python3 scripts/verify_docs.py`. That check does not compile MLX.

## Tests and the weights env var

`TripoSRRuntimeTests` (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift#TripoSRRuntimeTests`) covers preprocess, tokenizer round-trip, renderer shapes, a missing weights file, and the optional published file.

`tripoSRPublishedWeightsURL` reads `TRIPOSR_WEIGHTS_PATH`, then `triposr_swift_ready.safetensors` and `triposr_weights/triposr_swift_ready.safetensors` (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift#tripoSRPublishedWeightsURL`). It does not look up the string `TEST_RUNNER_TRIPOSR_WEIGHTS_PATH`.

`README.md:119` records an `xcodebuild` observation: a shell `TRIPOSR_WEIGHTS_PATH` alone did not reach the test process, and the `TEST_RUNNER_` prefix did. To pass the path through `xcodebuild`, set `TEST_RUNNER_TRIPOSR_WEIGHTS_PATH`. The test process then sees `TRIPOSR_WEIGHTS_PATH`.

`loadWeightsFromPublishedFileWhenPresent` (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift:108-118`) is skipped when the file is absent. When the file is present it only calls `loadWeights(from:)`. It does not run a forward pass.

`smokeExitsNonzeroWhenWeightsAreMissing` (`Tests/TripoSRRuntimeTests/TripoSRCLIExitTests.swift:34-86`) runs the built executable. CI does not download the weights, so the published-file test stays skipped there.
