# Memory and performance

Verified against `aafdeee6081dd780986479a72f34baab1a4f3c09`. The numbers below are copied from the README record of one M3 Max run. This page does not re-measure them.

## Cache limit in this repository

| Call site | Limit | Lines |
| --- | --- | --- |
| `runSmoke` | `512 * 1024 * 1024` (512 MiB) | `Sources/TripoSRCLI/main.swift:211` |
| `runParityDump` | `256 * 1024 * 1024` (256 MiB) | `Sources/TripoSRCLI/main.swift:130` |
| Library modules | not set | `Sources/TripoSRRuntime/TripoSRModel.swift:14-41` |

The 512 MiB value is the smoke command only. `--parity` uses 256 MiB.

`LRMTransformer1D` evaluates the graph after every block (`Sources/TripoSRRuntime/LRMTransformer1D.swift:53-57`). The comment there is about avoiding a fused 16-layer attention hang. It is not a point-query batch size.

## What is not in the library

`queryTriplane` (`Sources/TripoSRRuntime/TripoSRRenderer.swift:84-126`) runs the NeRF MLP on the positions it is given. There is no loop of 8192 points in `Sources/` or `Tests/`.

`README.md:131-138` records a harness outside this package: a 128^3 density grid, batches of 8192, then marching cubes and vertex colors. Those steps are not Swift sources here. The README says the harness left the default MLX cache limit, which read 36,721,970,380 bytes (`README.md:125`), rather than the CLI's 512 MiB.

## M3 Max record in the README

`README.md:116-142` is the source of these figures. Machine: Apple M3 Max (Mac15,11, 14 cores: 10 performance and 4 efficiency, 36 GB), macOS 27.3 build 26C5047e, Xcode 27.0 build 27A5252f, Metal toolchain 27A5252f, mlx-swift 0.31.4, Debug, sources at commit `7655b26` (`README.md:117`).

Tests with `TEST_RUNNER_TRIPOSR_WEIGHTS_PATH`: 6 tests in 2 suites, 0.773 seconds. `loadWeightsFromPublishedFileWhenPresent` 0.041 seconds. `smokeExitsNonzeroWhenWeightsAreMissing` 0.690 seconds (`README.md:119`).

CLI smoke on `Device(gpu, 0)`, random 512 image, 512 MiB cache: 0.75 seconds real, maximum resident set 1,733,754,880 bytes, peak memory footprint 2,231,355,360 bytes. Printed triplanes `[1, 3, 64, 64, 40]`, density min 0.08230684 max 2572.311, color min 0.2522956 max 0.6929118 (`README.md:123`).

Chair harness phases, seconds (`README.md:127-134`): weight load 0.095, first forward 0.538, second forward 0.514, density query 2.375, marching cubes 0.058, vertex colors 0.016. Clock from weight load through vertex colors 4.010 seconds, including the second forward. Process `/usr/bin/time -l`: 5.26 seconds real, maximum resident set 1,773,879,296 bytes, peak memory footprint 3,503,343,872 bytes. MLX `Memory.peakMemory` 2,629,506,977 bytes (`README.md:136`).

Grid stats (`README.md:138`): triplanes `[1, 3, 64, 64, 40]`, density from 1.271e-7 to 3516.801, 22,565 of 2,097,152 samples at or above 25, marching cubes at threshold 25 produced 10,066 vertices and 20,132 faces, 0 invalid faces, 0 boundary edges, 0 non-manifold edges, vertex colors from 0.00431 to 0.96070. The README states marching cubes ran outside this package.

The picture is [chair-m3-max.png](../chair-m3-max.png), linked from `README.md:142`.
