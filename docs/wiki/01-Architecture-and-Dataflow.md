# Architecture and dataflow

Ranges are checked by content hash, not by commit ancestry. Index: [knowledge-graph.md](../knowledge-graph.md). Standards: [engineering_standards.md](../engineering_standards.md).

`TripoSRModel` (`Sources/TripoSRRuntime/TripoSRModel.swift#TripoSRModel`) owns the modules. `init` (`Sources/TripoSRRuntime/TripoSRModel.swift#TripoSRModel.init`) builds:

- `DINOv2Model` at image size 224, patch 16, 3 channels, embed 768, depth 12, 12 heads (`Sources/TripoSRRuntime/TripoSRModel.swift:17-24`). Runtime 512 inputs interpolate position embeddings.
- `Triplane1DTokenizer` plane 32, 1024 channels (`Sources/TripoSRRuntime/TripoSRModel.swift:26`).
- `LRMTransformer1D` 1024 channels, 16 heads, head dim 64, 16 layers, cross-attention 768, 32 groups (`Sources/TripoSRRuntime/TripoSRModel.swift:28-35`).
- `TriplaneUpsampleNetwork` 1024 to 40 (`Sources/TripoSRRuntime/TripoSRModel.swift:37`).
- `NeRFMLP` 120 inputs, 64 neurons, 9 hidden layers (`Sources/TripoSRRuntime/TripoSRModel.swift:39`).
- `TripoSRRenderer` radius 0.87 (`Sources/TripoSRRuntime/TripoSRModel.swift:40`).

## forwardTriplanes

`Sources/TripoSRRuntime/TripoSRModel.swift#TripoSRModel.forwardTriplanes`:

1. ImageNet normalize unless `alreadyNormalized` (`Sources/TripoSRRuntime/TripoSRModel.swift:69`).
2. `DINOv2Model.callAsFunction` (`Sources/TripoSRRuntime/DINOv2Model.swift#DINOv2Model.callAsFunction`) patch-embeds, prepends CLS, adds interpolated positions, runs the blocks, and applies the final norm. Output comment: `[B, SeqLen, 768]`.
3. `Triplane1DTokenizer.callAsFunction` (`Sources/TripoSRRuntime/Triplane1DTokenizer.swift#Triplane1DTokenizer.callAsFunction`) yields `[B, 3 * 32 * 32, 1024]`, which is `[B, 3072, 1024]`.
4. `LRMTransformer1D` (`Sources/TripoSRRuntime/LRMTransformer1D.swift#LRMTransformer1D.callAsFunction`) cross-attends the latent sequence to the image tokens and `eval`s every block.
5. `detokenize` (`Sources/TripoSRRuntime/Triplane1DTokenizer.swift#Triplane1DTokenizer.detokenize`) restores `[B, 3, 32, 32, 1024]`.
6. `TriplaneUpsampleNetwork` (`Sources/TripoSRRuntime/TriplaneUpsampleNetwork.swift#TriplaneUpsampleNetwork.callAsFunction`) doubles the spatial size and writes 40 channels: `[B, 3, 64, 64, 40]`.

DINO pieces: `DINOv2PatchEmbed` (`Sources/TripoSRRuntime/DINOv2PatchEmbed.swift#DINOv2PatchEmbed`), `DINOv2Layer` (`Sources/TripoSRRuntime/DINOv2Layer.swift#DINOv2Layer`), `DINOv2Attention` (`Sources/TripoSRRuntime/DINOv2Attention.swift#DINOv2Attention`), `DINOv2MLP` (`Sources/TripoSRRuntime/DINOv2MLP.swift#DINOv2MLP`). LRM pieces: `LRMBasicTransformerBlock` (`Sources/TripoSRRuntime/LRMBasicTransformerBlock.swift#LRMBasicTransformerBlock`), `LRMAttention` (`Sources/TripoSRRuntime/LRMAttention.swift#LRMAttention`), `LRMFeedForward` (`Sources/TripoSRRuntime/LRMFeedForward.swift#LRMFeedForward`).

Position resize is bilinear. Details are in [02-Weights-and-Parity.md](02-Weights-and-Parity.md).

## evaluate

`evaluate` (`Sources/TripoSRRuntime/TripoSRModel.swift#TripoSRModel.evaluate`) calls `queryTriplane` (`Sources/TripoSRRuntime/TripoSRRenderer.swift#TripoSRRenderer.queryTriplane`).

The renderer divides positions by `radius` and does not clamp (`Sources/TripoSRRuntime/TripoSRRenderer.swift:93-95`). It samples the XY, XZ, and YZ planes with `gridSampleBilinear` (`Sources/TripoSRRuntime/TripoSRRenderer.swift#TripoSRRenderer.gridSampleBilinear`), concatenates them to `[B, N, 120]`, and runs `NeRFMLP` (`Sources/TripoSRRuntime/NeRFMLP.swift#NeRFMLP.callAsFunction`). Density is `exp(raw - 1)` and color is `sigmoid` (`Sources/TripoSRRuntime/TripoSRRenderer.swift:119-123`).

The whole positions tensor is one call. There is no 8192 loop in this file. Density and color are the outputs. A mesh is not.
