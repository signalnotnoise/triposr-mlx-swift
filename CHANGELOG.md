# Changelog

All notable changes to this package are documented in this file.

The format is based on Keep a Changelog, and this package does not have a
version tag yet.

## [Unreleased]

### Added

- Standalone Swift package for the TripoSR MLX runtime.
- Library product `TripoSRRuntime` and executable product `TripoSRCLI`.
- Sources and unit tests copied from the `TripoSR` submodule of
  `signalnotnoise/apple-mlx-foundations` at submodule commit
  `6da4a3cd3f8454f7c98607249476a26c836509ee`.
- Weight-loading test that skips when `triposr_swift_ready.safetensors`
  is not on disk.
- GitHub Actions workflow that builds and tests on macOS with `xcodebuild`.
- Swift Package Index manifest (`.spi.yml`).

### Changed

- `mlx-swift` is constrained to `0.31.3..<0.31.5`. The source manifest
  allowed `0.29.1` and newer, and its lockfile selected `0.31.3`.
  `mlx-swift` 0.31.5 and later require Swift tools 6.3.
- `TripoSRCLI` also searches `TRIPOSR_WEIGHTS_PATH` and
  `./triposr_swift_ready.safetensors`.
- `TripoSRCLI` smoke exits nonzero when weights are missing or the run throws.
