import Foundation
import Hub
import MLX
import MLXToolKit
import MossSoundEffectMLX

/// Errors specific to the MOSS-SoundEffect package boundary.
public enum MossSoundEffectError: Error, Equatable {
    /// Requested duration exceeds the model's 30 s ceiling.
    case durationOutOfRange(requested: Double, max: Double)
}

/// An MLXEngine `soundEffect` package over **MOSS-SoundEffect-v2.0** (OpenMOSS) — text →
/// sound effect (foley / ambience / creature / action), 48 kHz, ≤ 30 s. A thin conformance
/// wrapper over the standalone `MossSoundEffectMLX` engine (moss-soundeffect-mlx-swift);
/// all model logic lives there, parity-locked against the Python oracle.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `MossSoundEffectConfiguration`,
/// pages weights in with `load()` (downloads the mlx-community snapshot on first run), drives
/// `run(_:)`, and reclaims with `unload()`. Returns the canonical `Audio` (.wav, 48 kHz mono).
///
/// Behavior notes:
/// - The model always denoises a fixed 30 s latent and crops to `durationSeconds`
///   (duration is conditioned via a trained prompt suffix, not latent length).
/// - An empty `negativePrompt` is the trained unconditional path (all-zero context).
/// - Cancellation is honored per denoising step (~0.5 s granularity at bf16 on M-class).
@InferenceActor
public final class MossSoundEffectPackage: ModelPackage {
    public typealias Configuration = MossSoundEffectConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Weights Apache-2.0 (OpenMOSS); the Swift port (moss-soundeffect-mlx-swift) is
            // Apache-2.0 as well.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "mlx-community/MOSS-SoundEffect-v2.0-bf16", revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // Multi-component pipeline: 1.3B DiT + 1.5 GB fp32 VAE + 1.7B Qwen3 encoder.
                // Measured peaks on M5 Max at 100 steps: bf16 14.2 GB, int4 12.2 GB —
                // budgeted with headroom for the 30 s decode.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 15_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 13_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // 100-step CFG diffusion over a 1500-token latent — a capability floor as a
                // sanity marker; the MemoryGovernor still gates on the footprint.
                chipFloor: .pro
            ),
            specialties: [],
            surfaces: [
                SoundEffectContract.descriptor(
                    name: "moss-soundeffect",
                    summary: "MOSS-SoundEffect-v2.0 text-to-sound-effect (48 kHz .wav, up to "
                        + "30 s): foley, ambience, creature, and action audio from EN/ZH captions."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: MossSoundEffectPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        // Download (or reuse the cached) mlx-community snapshot. When the engine has set a
        // model-store root, point the Hub download base there.
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? .shared
        let directory = try await hub.snapshot(from: configuration.repo)
        // The model-core loader verifies with `.noUnusedKeys` — a mismatched checkpoint
        // fails here at load, never silently at inference.
        pipeline = try await MossSoundEffectPipeline.load(from: directory)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .soundEffect, let sfx = request as? SoundEffectRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let seconds = sfx.durationSeconds ?? configuration.defaultDurationSeconds
        guard seconds > 0, seconds <= Double(pipeline.maxInferenceSeconds) else {
            throw MossSoundEffectError.durationOutOfRange(
                requested: seconds, max: Double(pipeline.maxInferenceSeconds))
        }

        let waveform = try pipeline.generate(
            prompt: sfx.prompt,
            seconds: seconds,
            negativePrompt: sfx.negativePrompt ?? "",
            numInferenceSteps: sfx.steps ?? configuration.defaultSteps,
            cfgScale: sfx.guidanceScale.map(Float.init) ?? configuration.defaultGuidanceScale,
            seed: sfx.seed ?? 0,
            // Cancellation yield point per denoising step (C13): a governor-initiated
            // cancellation aborts the loop so the engine can reclaim and requeue.
            onStep: { _ in try Task.checkCancellation() }
        )
        eval(waveform)

        let audio = Audio(
            format: .wav,
            data: Self.encodeWAV(waveform, sampleRate: pipeline.sampleRate),
            sampleRate: pipeline.sampleRate,
            channels: 1
        )
        return SoundEffectResponse(audio: audio)
    }

    /// 16-bit PCM mono WAV from a (1, 1, T) waveform in [-1, 1].
    nonisolated static func encodeWAV(_ waveform: MLXArray, sampleRate: Int) -> Data {
        let samples: [Float] = waveform.reshaped(-1).asArray(Float.self)
        var pcm = Data(capacity: samples.count * 2)
        for s in samples {
            let clamped = max(-1, min(1, s))
            var value = Int16(clamped * 32767).littleEndian
            withUnsafeBytes(of: &value) { pcm.append(contentsOf: $0) }
        }

        var data = Data()
        func append(_ string: String) { data.append(contentsOf: string.utf8) }
        func append32(_ value: UInt32) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        func append16(_ value: UInt16) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        append("RIFF")
        append32(UInt32(36 + pcm.count))
        append("WAVE")
        append("fmt ")
        append32(16)                          // PCM chunk size
        append16(1)                           // PCM format
        append16(1)                           // mono
        append32(UInt32(sampleRate))
        append32(UInt32(sampleRate * 2))      // byte rate (16-bit mono)
        append16(2)                           // block align
        append16(16)                          // bits per sample
        append("data")
        append32(UInt32(pcm.count))
        data.append(pcm)
        return data
    }
}
