# AGENTS.md

Router for this repository. Read the linked docs before broad searches.

## Identity and scope

`TripoSRRuntime` (library) and `TripoSRCLI` (executable) run the TripoSR image-to-3D forward pass on Apple MLX. Declared platforms are macOS 14, iOS 17, and visionOS 1 (`Package.swift:9-13`).

`forwardTriplanes(image:)` (`Sources/TripoSRRuntime/TripoSRModel.swift#TripoSRModel.forwardTriplanes`) takes NHWC float RGB in `[0, 1]`. For the default module sizes the result is `[B, 3, 64, 64, 40]`. The CLI smoke uses batch 1, so that run prints `[1, 3, 64, 64, 40]`.

`evaluate(triplanes:positions:)` (`Sources/TripoSRRuntime/TripoSRModel.swift#TripoSRModel.evaluate`) returns `(density, color)`. It does not return a mesh.

Out of scope in this repository:

- No mesh extraction and no marching cubes.
- No Swift texture baking.
- No background removal.
- No Python scripts that generate `.npy` parity files. `TripoSRCLI --parity` reads `parity/image.npy` and `parity/positions.npy` when they are supplied (`Sources/TripoSRCLI/main.swift#main`). Missing inputs or weights print `Failed:` and exit nonzero (`Sources/TripoSRCLI/main.swift:139-145`).

## Documentation index

- [docs/engineering_standards.md](docs/engineering_standards.md): invariants for tools, tensors, weights, and memory.
- [docs/knowledge-graph.json](docs/knowledge-graph.json): symbol, file, and tensor index. Read this before a broad grep.
- [docs/knowledge-graph.md](docs/knowledge-graph.md): diagrams and the where-to-find-things table.
- [docs/wiki/01-Architecture-and-Dataflow.md](docs/wiki/01-Architecture-and-Dataflow.md)
- [docs/wiki/02-Weights-and-Parity.md](docs/wiki/02-Weights-and-Parity.md)
- [docs/wiki/03-Build-CI-and-Testing.md](docs/wiki/03-Build-CI-and-Testing.md)
- [docs/wiki/04-Memory-and-Performance.md](docs/wiki/04-Memory-and-Performance.md)

## Build and test

Do not use `swift build` or `swift test`. mlx-swift needs Xcode to compile Metal shaders and bundle `default.metallib`. The unit tests set the MLX default device to CPU (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift#TripoSRRuntimeTests.init`) and still need that library.

```bash
xcodebuild build -scheme TripoSRCLI -destination 'platform=macOS' -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
xcodebuild test -scheme TripoSR-Package -destination 'platform=macOS' -skipPackagePluginValidation -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO
```

The optional weights test reads `TRIPOSR_WEIGHTS_PATH` (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift#tripoSRPublishedWeightsURL`). Under `xcodebuild`, the README records that a bare shell variable did not reach the test process and that the `TEST_RUNNER_` prefix did (`README.md:119`). Example: `TEST_RUNNER_TRIPOSR_WEIGHTS_PATH=/path/to/triposr_swift_ready.safetensors`.

Check citations with `python3 scripts/verify_docs.py` before pushing and before tagging. If line-range citations fail because code moved, `python3 scripts/verify_docs.py --fix` retargets them when the hashed text still exists uniquely in the same file. See CONTRIBUTING.md.

## Zero-drift rule

A commit that adds, moves, renames, or changes files, public APIs, MLX layers, or `Package.swift` constraints must update `docs/knowledge-graph.json`, `docs/knowledge-graph.md`, and the affected `docs/wiki/*.md` pages in that same commit. Re-read the cited lines and refresh each `content_sha256`. `verified_commit` is a note only. `scripts/verify_docs.py` checks the working tree and does not require that commit to be an ancestor of HEAD.
