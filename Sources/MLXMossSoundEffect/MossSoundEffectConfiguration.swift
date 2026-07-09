import Foundation
import MLXToolKit

/// Init-time configuration for `MossSoundEffectPackage` (C9): which mlx-community checkpoint
/// to load and the diffusion defaults applied when a request leaves them unset. Per-request
/// prompt / duration / steps / guidance / seed ride the `SoundEffectRequest` envelope, not here.
///
/// Conforms to `QuantConfigured` so the memory governor charges the *selected* checkpoint's
/// declared `QuantFootprint` (bf16 vs int4 DiT — different resident floors) instead of the
/// largest-that-fits heuristic. Quant is derived from the repo suffix (`…-4bit` → int4, else bf16),
/// the same signal the core loader reads from model_index.json.
public struct MossSoundEffectConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// HuggingFace repo in the published MLX layout (model_index.json + mlx/{dit,vae}.safetensors
    /// + text_encoder + tokenizer). `…-bf16` or `…-4bit`; the loader auto-detects quantization
    /// from model_index.json.
    public var repo: String

    /// The selected checkpoint's DiT quantization, from the repo suffix (`…-4bit` → int4, else bf16).
    /// Exposed for `QuantConfigured` so the governor matches the right per-quant footprint.
    public var quant: Quant {
        repo.hasSuffix("-4bit") || repo.contains("4bit") ? .int4 : .bf16
    }
    /// Default denoising steps when the request doesn't set `steps` (reference default 100).
    public var defaultSteps: Int
    /// Default CFG scale when the request doesn't set `guidanceScale` (reference default 4.0).
    public var defaultGuidanceScale: Float
    /// Default output duration when the request doesn't set `durationSeconds`.
    public var defaultDurationSeconds: Double
    /// Explicit snapshot directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/MOSS-SoundEffect-v2.0-bf16",
                defaultSteps: Int = 100,
                defaultGuidanceScale: Float = 4.0,
                defaultDurationSeconds: Double = 10.0,
                modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.defaultDurationSeconds = defaultDurationSeconds
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, defaultSteps, defaultGuidanceScale, defaultDurationSeconds
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension MossSoundEffectConfiguration: WeightSourcing {
    /// Presence probe: what `MossSoundEffectPipeline.load(from:)` reads — the quantization
    /// index, both mlx checkpoints, the Qwen3 text-encoder shard index, and the tokenizer.
    /// The Qwen3-1.7B encoder rides the SAME consolidated repo (`text_encoder/`), not a
    /// separate one.
    static let requiredFiles = [
        "model_index.json",
        "mlx/dit.safetensors",
        "mlx/vae.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer.json",
    ]
    /// Download globs — the loadable snapshot (README/scheduler config are skipped; the
    /// scheduler is code-side).
    static let snapshotGlobs = [
        "model_index.json", "mlx/*.safetensors", "text_encoder/*", "tokenizer/*",
    ]

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: repo, matching: Self.snapshotGlobs)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func has(_ dir: URL) -> Bool {
            Self.requiredFiles.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        // Explicit local directory first (dev escape hatch), then the ModelStore layout.
        if let dir = modelDirectory, has(dir) { return [] }
        if let dir = ModelStore(root: storeRoot).directory(for: repo), has(dir) { return [] }
        return weightSources
    }

    /// The configuration with a nil `modelDirectory` resolved to the store layout — what `load()`
    /// uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> MossSoundEffectConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension MossSoundEffectConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved snapshot directory; the prewarmer scans it recursively (DiT/VAE +
        // text-encoder shards) and skips it when absent (first launch).
        [resolved(storeRoot: modelsRootDirectory).modelDirectory].compactMap { $0 }
    }
}
