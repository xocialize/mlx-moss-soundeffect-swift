// swift-tools-version: 6.2
import PackageDescription

// mlx-moss-soundeffect-swift — the MLXEngine `soundEffect` package over MOSS-SoundEffect-v2.0
// (1.3B flow-matching DiT + continuous DAC VAE + Qwen3 encoder, 48 kHz SFX). A thin conformance
// layer wrapping the standalone inference engine moss-soundeffect-mlx-swift (product
// `MossSoundEffectMLX`), the same way mlx-voxcpm2-tts-swift wraps mlx-voxcpm-swift. The engine
// contract (MLXToolKit) is a local-path dep for in-workspace dev; the model core is pinned to a
// tagged release.
//
// First package for the `soundEffect` capability (contract 1.2.0).
let package = Package(
    name: "mlx-moss-soundeffect-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXMossSoundEffect", targets: ["MLXMossSoundEffect"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/moss-soundeffect-mlx-swift.git", from: "0.1.2"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXMossSoundEffect",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MossSoundEffectMLX", package: "moss-soundeffect-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            // MossSoundEffectPipeline (MLX arrays + tokenizer) isn't Sendable-audited; the engine
            // serializes all lifecycle on InferenceActor, so v5 mode keeps strict region-isolation
            // a warning while @InferenceActor isolation still holds — same lever as VoxCPM2/Kokoro.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXMossSoundEffectTests",
            dependencies: [
                "MLXMossSoundEffect",
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
