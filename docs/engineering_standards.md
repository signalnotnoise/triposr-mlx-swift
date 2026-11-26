# Engineering standards

Invariants of this repository. Generic Swift style is not covered here.

Cited ranges are locked by `content_sha256` in [knowledge-graph.json](knowledge-graph.json). `verified_commit` there is informational and is not an ancestry check. Index: [knowledge-graph.md](knowledge-graph.md). Router: [AGENTS.md](../AGENTS.md).

## Swift tools and mlx-swift

`Package.swift:1` is `swift-tools-version: 6.2`. The mlx-swift requirement is `0.31.3..<0.31.5` (`Package.swift:24-28`). Release 0.31.5 and later set Swift tools 6.3, which this manifest does not require. Do not bump mlx-swift past 0.31.4 while the tools version stays 6.2.

## Tensor layout

- Input RGB is NHWC `[batch, height, width, 3]`, float in `[0, 1]` (`Sources/TripoSRRuntime/TripoSRModel.swift:64-67`).
- ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` are applied inside `forwardTriplanes` unless `alreadyNormalized` is true (`Sources/TripoSRRuntime/TripoSRModel.swift:53-69`).
- Default modules produce triplanes `[B, 3, 64, 64, 40]`: plane size 32, upsample doubles spatial size, 40 channels (`Sources/TripoSRRuntime/TripoSRModel.swift:26-37` and `Sources/TripoSRRuntime/TripoSRModel.swift:80-84`). The leading `1` in `[1, 3, 64, 64, 40]` is the CLI smoke batch (`Sources/TripoSRCLI/main.swift:211`), not a hardcoded batch size.
- Query positions are `[B, N, 3]`, divided by radius `0.87` inside the renderer (`Sources/TripoSRRuntime/TripoSRRenderer.swift:8-10` and `Sources/TripoSRRuntime/TripoSRRenderer.swift:93-95`). The code does not clamp positions into that radius.
- `evaluate` returns density `[B, N, 1]` and color `[B, N, 3]` (`Sources/TripoSRRuntime/TripoSRRenderer.swift:119-123`). It does not return a mesh.

## Weights

`loadWeights(from:)` loads a safetensors file and calls `update` with `verify: [.shapeMismatch, .noUnusedKeys]` (`Sources/TripoSRRuntime/TripoSRModel.swift:44-51`). This repository does not contain a tensor key-mapping table. Keys must already match the Swift module parameter names.

Canonical file recorded in the README: `triposr_swift_ready.safetensors`, 1,674,793,528 bytes, Hub commit `76f3028746d0b5f39b6ba0205ede9e28835f106d`, SHA-256 `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb` (`README.md:121`). Do not commit it. `.gitignore:12-13` ignores `*.safetensors` and `triposr_weights/`.

`TripoSRCLI` search order (`Sources/TripoSRCLI/main.swift:97-112`):

1. `TRIPOSR_WEIGHTS_PATH` when that variable is non-empty.
2. `./triposr_swift_ready.safetensors`
3. `./triposr_weights/triposr_swift_ready.safetensors`
4. `../triposr_weights/triposr_swift_ready.safetensors`
5. `TripoSR/triposr_weights/triposr_swift_ready.safetensors`
6. `../../TripoSR/triposr_weights/triposr_swift_ready.safetensors`

Items 4 through 6 are the legacy monorepo paths. The first existing file wins.

Smoke with no file throws `CLIError.weightsNotFound` before the model is built and the process exits 1 (`Sources/TripoSRCLI/main.swift:199-201` and `Sources/TripoSRCLI/main.swift:234-240`).

## Positional embeddings

`interpolatePosEmbed` is a manual bilinear resize (`Sources/TripoSRRuntime/DINOv2Model.swift:99-126`). The comment on `Sources/TripoSRRuntime/DINOv2Model.swift:66` calls that resize "bicubic-ish bilinear". The README says the Hugging Face ViT uses bicubic interpolation (`README.md:150`). There is no bicubic implementation in this repository.

`--parity` is a real flag (`Sources/TripoSRCLI/main.swift:227`). It expects `image.npy` and `positions.npy` in the parity directory (`Sources/TripoSRCLI/main.swift:137-138`). If either file is missing, it throws `CLIError.parityInputsMissing` and names that directory (`Sources/TripoSRCLI/main.swift:139-142`). If the weights are missing, it throws `CLIError.weightsNotFound` (`Sources/TripoSRCLI/main.swift:143-145`). Both print `Failed:` and exit nonzero (`Sources/TripoSRCLI/main.swift:234-240`). This repository does not contain the script that writes those `.npy` files. Do not change the bilinear resize without a parity comparison that actually has those inputs.

## Memory and batching

`runSmoke` sets `Memory.cacheLimit` to `512 * 1024 * 1024` (`Sources/TripoSRCLI/main.swift:203`). `runParityDump` sets `256 * 1024 * 1024` (`Sources/TripoSRCLI/main.swift:131`). The library does not set a cache limit.

This package does not batch point queries. `queryTriplane` evaluates the positions tensor it is given in one call (`Sources/TripoSRRuntime/TripoSRRenderer.swift:84-126`). The 8192-point batches and the 128^3 grid are a local harness recorded in `README.md:131-138`, outside this package. Do not treat 8192 as a constant in `Sources/`.

`LRMTransformer1D` calls `eval` after every block (`Sources/TripoSRRuntime/LRMTransformer1D.swift:53-57`). That is the in-tree guard against fusing the 16-layer attention, not a point batch size.
