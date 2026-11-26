# Knowledge graph

Human index for [knowledge-graph.json](knowledge-graph.json).

Verified commit: `aafdeee6081dd780986479a72f34baab1a4f3c09` on `cursor/triposr-mlx-swift-package-45c1`. Those line numbers were read from that commit. This page does not change Swift sources. If `Sources/`, `Tests/`, `Package.swift`, `.github/workflows/ci.yml`, `.gitignore`, or `README.md` change, update the JSON and this page in the same commit. Standards: [engineering_standards.md](engineering_standards.md). Router: [AGENTS.md](../AGENTS.md).

No `actor` or `protocol` types are declared under `Sources/` or `Tests/`.

## Tensor dataflow

```mermaid
flowchart TD
  img["image NHWC B,H,W,3 in 0 to 1"]
  pre["preprocessImage unless alreadyNormalized"]
  dino["DINOv2Model to B, seq, 768"]
  tok["Triplane1DTokenizer to B, 3072, 1024"]
  lrm["LRMTransformer1D"]
  detok["detokenize to B, 3, 32, 32, 1024"]
  up["TriplaneUpsampleNetwork to B, 3, 64, 64, 40"]
  qry["queryTriplane"]
  mlp["NeRFMLP to B, N, 4"]
  out["density B,N,1 and color B,N,3"]
  img --> pre --> dino --> lrm
  tok --> lrm --> detok --> up --> qry --> mlp --> out
```

`forwardTriplanes` is `Sources/TripoSRRuntime/TripoSRModel.swift:67-87`. `evaluate` is `Sources/TripoSRRuntime/TripoSRModel.swift:90-92`. The CLI smoke batch is 1 (`Sources/TripoSRCLI/main.swift:219`).

## Where to find things

| Task | Location |
| --- | --- |
| Weight load and shape or unused-key checks | `Sources/TripoSRRuntime/TripoSRModel.swift:44-51` |
| ImageNet mean, std, and preprocess | `Sources/TripoSRRuntime/TripoSRModel.swift:53-62` |
| DINO forward and bilinear position resize | `Sources/TripoSRRuntime/DINOv2Model.swift:36-129` |
| Image to triplanes | `Sources/TripoSRRuntime/TripoSRModel.swift:67-87` |
| Triplane sample | `Sources/TripoSRRuntime/TripoSRRenderer.swift:84-126` |
| NeRF MLP | `Sources/TripoSRRuntime/NeRFMLP.swift:5-39` |
| CLI entry and `--parity` | `Sources/TripoSRCLI/main.swift:233-247` |
| Weight search order | `Sources/TripoSRCLI/main.swift:94-109` |
| Parity dump, including the exit-0 early returns | `Sources/TripoSRCLI/main.swift:127-205` |
| Smoke, 512 MiB cache, nonzero missing-weights exit | `Sources/TripoSRCLI/main.swift:207-231` |
| Published-weights test, load only | `Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift:108-118` |
| Env var the test actually reads | `Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift:9-26` |
| CLI nonzero-exit test | `Tests/TripoSRRuntimeTests/TripoSRCLIExitTests.swift:34-86` |
| Package manifest | `Package.swift:1-46` |
| macOS CI | `.github/workflows/ci.yml:16-58` |
| M3 Max numbers, as recorded, not re-run here | `README.md:116-142` |

Wiki: [01 Architecture](wiki/01-Architecture-and-Dataflow.md), [02 Weights](wiki/02-Weights-and-Parity.md), [03 Build and CI](wiki/03-Build-CI-and-Testing.md), [04 Memory](wiki/04-Memory-and-Performance.md).
