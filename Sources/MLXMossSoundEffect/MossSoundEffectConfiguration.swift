import Foundation
import MLXToolKit

/// Init-time configuration for `MossSoundEffectPackage` (C9): which mlx-community checkpoint
/// to load and the diffusion defaults applied when a request leaves them unset. Per-request
/// prompt / duration / steps / guidance / seed ride the `SoundEffectRequest` envelope, not here.
public struct MossSoundEffectConfiguration: PackageConfiguration, ModelStorable {
    /// HuggingFace repo in the published MLX layout (model_index.json + mlx/{dit,vae}.safetensors
    /// + text_encoder + tokenizer). `…-bf16` or `…-4bit`; the loader auto-detects quantization
    /// from model_index.json.
    public var repo: String
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
