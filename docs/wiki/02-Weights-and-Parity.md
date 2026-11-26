# Weights and parity

Verified against `aafdeee6081dd780986479a72f34baab1a4f3c09`. Standards: [engineering_standards.md](../engineering_standards.md).

## What loadWeights checks

`loadWeights(from:)` (`Sources/TripoSRRuntime/TripoSRModel.swift:44-51`) reads the safetensors file with `loadArrays` and calls `update` with `verify: [.shapeMismatch, .noUnusedKeys]`. A missing file throws. The missing-file test is `Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift:96-102`.

There is no key-renaming table in this repository. The file must already use Swift module parameter names. The Python remap script that produced `triposr_swift_ready.safetensors` is not checked in here.

## Canonical file

Recorded in `README.md:121`, not vendored:

- Name: `triposr_swift_ready.safetensors`
- Bytes: 1,674,793,528
- Hub commit: `76f3028746d0b5f39b6ba0205ede9e28835f106d`
- SHA-256: `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`

`.gitignore:12-13` ignores `*.safetensors` and `triposr_weights/`.

```bash
revision=76f3028746d0b5f39b6ba0205ede9e28835f106d
curl -L -o triposr_swift_ready.safetensors \
  "https://huggingface.co/signalNotNoise/triposr-mlx/resolve/${revision}/triposr_swift_ready.safetensors"
echo "83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb  triposr_swift_ready.safetensors" | shasum -a 256 -c -
```

## CLI search order

`resolveWeights` (`Sources/TripoSRCLI/main.swift:94-109`):

1. `TRIPOSR_WEIGHTS_PATH`
2. `./triposr_swift_ready.safetensors`
3. `./triposr_weights/triposr_swift_ready.safetensors`
4. `../triposr_weights/triposr_swift_ready.safetensors`
5. `TripoSR/triposr_weights/triposr_swift_ready.safetensors`
6. `../../TripoSR/triposr_weights/triposr_swift_ready.safetensors`

Smoke throws before model construction when none exist (`Sources/TripoSRCLI/main.swift:207-209`) and the top-level catch exits 1 (`Sources/TripoSRCLI/main.swift:242-247`). The test is `Tests/TripoSRRuntimeTests/TripoSRCLIExitTests.swift:34-86`.

## --parity

`main` selects `runParityDump` when argv contains `--parity` (`Sources/TripoSRCLI/main.swift:233-236`). The function is `Sources/TripoSRCLI/main.swift:127-205`.

It forces CPU (`Sources/TripoSRCLI/main.swift:129`) and sets the cache limit to 256 MiB (`Sources/TripoSRCLI/main.swift:130`). It looks for `parity/image.npy` and `parity/positions.npy` under `resolveParityDir` (`Sources/TripoSRCLI/main.swift:111-125` and `Sources/TripoSRCLI/main.swift:136-137`).

If those files are missing, it prints a Python command that is not in this repo and returns (`Sources/TripoSRCLI/main.swift:138-146`). If weights are missing, it prints and returns (`Sources/TripoSRCLI/main.swift:156-158`). Both of those returns leave the process exit code at 0. A thrown error later still hits the `Failed:` catch and exits 1.

The optional weights unit test only calls `loadWeights` and does not run a forward or `--parity` (`Tests/TripoSRRuntimeTests/TripoSRRuntimeTests.swift:104-118`).

Bilinear position resize, which `--parity` is meant to compare, is `Sources/TripoSRRuntime/DINOv2Model.swift:67-129`. The implementation is bilinear (`Sources/TripoSRRuntime/DINOv2Model.swift:99-126`). Line 66 of that file says "bicubic-ish bilinear". `README.md:150` says the Hugging Face ViT uses bicubic. This tree has no bicubic path.
