// swift-tools-version: 6.2
import PackageDescription

// Standalone package for the TripoSR MLX runtime.
// Product names match the TripoSR submodule manifest so a downstream
// package can keep depending on TripoSRRuntime and TripoSRCLI.
let package = Package(
    name: "TripoSR",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "TripoSRRuntime",
            targets: ["TripoSRRuntime"]
        ),
        .executable(
            name: "TripoSRCLI",
            targets: ["TripoSRCLI"]
        ),
    ],
    dependencies: [
        // Source lockfile pin. mlx-swift 0.31.5 and later set
        // swift-tools-version 6.3, which this manifest does not require.
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.3"..<"0.31.5"),
    ],
    targets: [
        .target(
            name: "TripoSRRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "TripoSRCLI",
            dependencies: ["TripoSRRuntime"]
        ),
        .testTarget(
            name: "TripoSRRuntimeTests",
            dependencies: ["TripoSRRuntime", "TripoSRCLI"]
        ),
    ]
)
