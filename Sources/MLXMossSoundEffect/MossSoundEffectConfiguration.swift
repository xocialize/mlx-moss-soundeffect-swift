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
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable` (environment-specific).
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/MOSS-SoundEffect-v2.0-bf16",
                defaultSteps: Int = 100,
                defaultGuidanceScale: Float = 4.0,
                defaultDurationSeconds: Double = 10.0,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.defaultDurationSeconds = defaultDurationSeconds
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, defaultSteps, defaultGuidanceScale, defaultDurationSeconds
    }
}
