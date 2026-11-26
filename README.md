`TripoSRRuntime` loads `triposr_swift_ready.safetensors` and runs an RGB image through an MLX Swift port of TripoSR, returning triplanes plus density and color. It does not extract a mesh.

## Install

There is no version tag yet. Depend on the `main` branch. The package identity is the repository name. You need read access to this repository.

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Example",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    dependencies: [
        .package(url: "https://github.com/signalnotnoise/triposr-mlx-swift.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "Example",
            dependencies: [
                .product(name: "TripoSRRuntime", package: "triposr-mlx-swift"),
            ]
        ),
    ]
)
```

Products:

| Product | Kind |
| --- | --- |
| `TripoSRRuntime` | Library. Import this. |
| `TripoSRCLI` | Executable. Smoke forward pass and optional parity dump. |

`mlx-swift` is capped at `0.31.3..<0.31.5` in `Package.swift`. The source package's lockfile selected `0.31.3`, and `0.31.4` is the last release that still builds with this manifest. Starting with `0.31.5`, `mlx-swift` sets `swift-tools-version` to 6.3. This package stays on Swift tools 6.2, so the range stops before `0.31.5` and SwiftPM will not select a dependency this manifest cannot resolve.

## Usage

```swift
import MLX
import TripoSRRuntime

let model = TripoSRModel()
try model.loadWeights(from: weightsURL)

// image: [batch, height, width, 3] float RGB in [0, 1], NHWC.
// ImageNet mean and std are applied inside forwardTriplanes.
let triplanes = model.forwardTriplanes(image: image)

// positions: [batch, pointCount, 3], roughly inside radius 0.87.
let (density, color) = model.evaluate(triplanes: triplanes, positions: positions)
```

`loadWeights(from:)` reads a `.safetensors` file, checks tensor shapes, and rejects unused keys. A missing file throws.

`TripoSRCLI` looks for weights in this order: `TRIPOSR_WEIGHTS_PATH`, `./triposr_swift_ready.safetensors`, `./triposr_weights/triposr_swift_ready.safetensors`, then the older monorepo relative paths. If none exist, the smoke command prints `Failed:` and exits nonzero. Any other smoke failure also exits nonzero. `--parity` additionally needs `parity/image.npy` and `parity/positions.npy`. The Python scripts that write those dumps are not in this repository. The runtime result is density and color, not a mesh.

## Weights

Download the canonical file from [signalNotNoise/triposr-mlx](https://huggingface.co/signalNotNoise/triposr-mlx) and check the SHA-256 before calling `loadWeights`. The Hub commit recorded for this extraction is `76f3028746d0b5f39b6ba0205ede9e28835f106d`. That commit's `checksums.sha256` lists:

```text
83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb  triposr_swift_ready.safetensors
```

The file is 1,674,793,528 bytes. It is not in this git repository.

```bash
revision=76f3028746d0b5f39b6ba0205ede9e28835f106d
curl -L -o triposr_swift_ready.safetensors \
  "https://huggingface.co/signalNotNoise/triposr-mlx/resolve/${revision}/triposr_swift_ready.safetensors"
curl -L -o checksums.sha256 \
  "https://huggingface.co/signalNotNoise/triposr-mlx/resolve/${revision}/checksums.sha256"

# One-file check of the weights the Swift loader accepts:
echo "83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb  triposr_swift_ready.safetensors" | shasum -a 256 -c -

# Bundle check (every file named in checksums.sha256, including the weights):
grep 'triposr_swift_ready.safetensors$' checksums.sha256 | shasum -a 256 -c -
```

`shasum -a 256 -c` exits non-zero when the digest does not match. Do not load a file that fails the check.

To run the optional weight test, point `TRIPOSR_WEIGHTS_PATH` at the verified file. That test only calls `loadWeights(from:)` and does not run a forward pass. If that variable is unset and the file is not in the working directory, the test is skipped. `TripoSRRuntime` outputs density and color. It does not output a mesh.

## Platforms

`Package.swift` declares macOS 14, iOS 17, and visionOS 1.

What this repository's CI actually runs: one macOS job on `macos-26`, using the image's default Xcode (Swift 6.2 or newer) and `xcodebuild`. The job builds `TripoSRCLI` and runs `TripoSRRuntimeTests`. It does not download the weights, so `loadWeightsFromPublishedFileWhenPresent` is skipped there. It does not build an iOS or visionOS destination.

`swift build` and `swift test` are not the CI path. mlx-swift compiles Metal shaders and bundles `default.metallib` through Xcode. The unit tests set the MLX default device to CPU, and they still need that Metal library to load.

Build and test locally:

```bash
xcodebuild build \
  -scheme TripoSRCLI \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO

xcodebuild test \
  -scheme TripoSR-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
```

## Limitations

- No mesh extraction. The library stops at density and color.
- No Swift texture baking.
- No background removal. Callers have to supply the image they want the model to see.
- This repository does not re-run the Python-to-Swift parity scripts. Those scripts stayed in the source tree. The CLI still has a `--parity` mode if you already have the `.npy` inputs.
- DINO positional embeddings are resized with bilinear interpolation in `DINOv2Model`. The source notes that the Hugging Face ViT uses bicubic interpolation.
- Real-weight inference is not part of CI. The published-file test runs only when the safetensors file is already on disk.

## Where the code came from

Clean copy, not a history rewrite. The Swift sources and the four deterministic unit tests come from the `TripoSR` git submodule of [apple-mlx-foundations](https://github.com/signalnotnoise/apple-mlx-foundations) `main`, submodule commit [`6da4a3cd3f8454f7c98607249476a26c836509ee`](https://github.com/signalnotnoise/TripoSR/tree/6da4a3cd3f8454f7c98607249476a26c836509ee) on `agent/triposr-swift-integration`. That submodule history is the upstream TripoSR project plus the Swift port, so it was not replayed here. Layout in this repository is the Swift package itself: `Sources/`, `Tests/`, and a root `Package.swift`. No test fixtures were in the source test target.

## License

This package is MIT, Copyright (c) 2026 Ismael Alex Otero. See [LICENSE](LICENSE). signalnotnoise maintains the project.

Upstream TripoSR is MIT, Copyright (c) 2024 Tripo AI & Stability AI. The license text is copied in [LICENSES/TripoSR.txt](LICENSES/TripoSR.txt). The upstream project states that the MIT license includes the source code and the pretrained models. Attribution for TripoSR, the DINO initialization inside that checkpoint, and MLX is in [NOTICE](NOTICE).

This is an independent port. It is not affiliated with or endorsed by Tripo AI, Stability AI, Meta AI, or Apple.
