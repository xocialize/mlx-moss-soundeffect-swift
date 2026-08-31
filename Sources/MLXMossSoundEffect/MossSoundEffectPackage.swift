import Foundation
import MLX
import MLXToolKit
import MossSoundEffectMLX

/// Errors specific to the MOSS-SoundEffect package boundary.
public enum MossSoundEffectError: Error, Equatable {
    /// Requested duration exceeds the model's 30 s ceiling.
    case durationOutOfRange(requested: Double, max: Double)
    /// Weight sources are missing and there is no store root (or resolved directory) to
    /// materialize into.
    case missingWeights(String)
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
                // Split footprint (contract 1.14). Multi-component pipeline: DiT + fp32 VAE + Qwen3-1.7B
                // text encoder. The encoder (~4 GB fp32 shards, mlx-community text_encoder/) encodes the
                // prompt ONCE, then is EVICTED before the 100-step CFG denoise (encoder-evict lever, in
                // moss-soundeffect-mlx-swift Pipeline.generate) — so it is NOT part of the
                // resident-through-denoise floor.
                //
                // resident (post-evict, held through the denoise = DiT + fp32 VAE):
                //   bf16: dit 2.8 GB + vae 1.5 GB → ~5 GB    int4: dit 0.8 GB + vae 1.5 GB → ~3 GB
                // (on-disk mlx-community bytes: bf16 dit 2832 MB / int4 dit 831 MB / vae 1486 MB fp32).
                // The VAE stays fp32 in both quants; only the DiT is quantized, hence the small delta.
                //
                // peakActivationBytes = the 100-step CFG denoise (two DiT forwards/step at the 1500-token
                // latent) + the 30 s fp32 VAE decode transient — the activation driver is the
                // 1500-token / 100-step envelope, largely dtype-independent (the LTX co-residency finding).
                //
                // PHYS RE-BASELINED 2026-08-31 (AB-T-0107): direct-load harness, task_vm_info
                // phys_footprint at 50 ms through generate (10 steps — per-step CFG shapes are
                // step-count-independent — plus the full 30 s fp32 VAE decode), two runs. bf16:
                // post-load floor 7.32 GB; in-run peak 25.33 GB both runs. resident = the load
                // floor (a warm-but-idle model; cache above it is reclaimable), peakActivation =
                // peak − floor. int4 is DERIVED from the bf16 measurement: floor minus the
                // on-disk DiT delta (2832 → 831 MB ≈ 2.0 GB); activations dtype-independent.
                // ⚠️ Separate finding: after unload() + MLX clearCache the process still held
                // 19.3 GB (Metal-heap retention below MLX's cache — the old flat 14.2 GB claim
                // was itself an underread). Filed with the sweep receipt; eviction reclaim is
                // NOT yet a full return-to-floor for this package.
                footprints: [
                    QuantFootprint(quant: .bf16,
                                   residentBytes: 7_400_000_000,
                                   peakActivationBytes: 18_000_000_000),
                    QuantFootprint(quant: .int4,
                                   residentBytes: 5_400_000_000,
                                   peakActivationBytes: 18_000_000_000),
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
        // Auto-materialize the missing snapshot into the engine store (dir-less configs only;
        // explicit directories never touch the network), forwarding progress via
        // WeightDownloadProgress so the engine's PreparationMonitor surfaces `.downloading`.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw MossSoundEffectError.missingWeights(
                    "no models root set and sources missing: \(missing.map(\.role).joined(separator: ", "))")
            }
            try await WeightMaterializer.materialize(missing, into: storeRoot)
        }
        try Task.checkCancellation()
        guard let directory = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw MossSoundEffectError.missingWeights("unresolved model directory (no store root)")
        }
        // The model-core loader verifies with `.noUnusedKeys` — a mismatched checkpoint
        // fails here at load, never silently at inference.
        pipeline = try await MossSoundEffectPipeline.load(from: directory)
    }

    public func unload() async {
        pipeline = nil
        // Drop the DiT / VAE / encoder buffers from MLX's pool too — niling the ref alone leaves
        // them cached, so phys_footprint doesn't fall and engine.evict / R-MEM-1 can't reclaim
        // (RSS then grows monotonically across model switches). Contract 1.14 requirement.
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the throwing onStep hook below fires before every
        // CFG denoise step AND once more before the VAE decode (core ≥ 0.2.0), rethrowing the
        // CancellationError unchanged through the core's rethrows chain.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .soundEffect, let sfx = request as? SoundEffectRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let seconds = sfx.durationSeconds ?? configuration.defaultDurationSeconds
        guard seconds > 0, seconds <= Double(pipeline.maxInferenceSeconds) else {
            throw MossSoundEffectError.durationOutOfRange(
                requested: seconds, max: Double(pipeline.maxInferenceSeconds))
        }

        let steps = sfx.steps ?? configuration.defaultSteps
        let waveform = try pipeline.generate(
            prompt: sfx.prompt,
            seconds: seconds,
            negativePrompt: sfx.negativePrompt ?? "",
            numInferenceSteps: steps,
            cfgScale: sfx.guidanceScale.map(Float.init) ?? configuration.defaultGuidanceScale,
            seed: sfx.seed ?? 0,
            // Cancellation yield points (C13/CAN gate): the core fires this before each
            // denoising step and once more (index == steps) before the final VAE decode
            // (core ≥ 0.2.0) — a cancellation aborts the loop / skips the decode so the
            // engine can reclaim and requeue. The same hook carries V2 run progress: the
            // engine binds `RunProgress` around run(), so per-step reports surface in the
            // host's RunMonitor (ENGINE-NEEDS V2; a fixed 30 s latent means the step count
            // is the whole story — every render walks all `steps` regardless of duration).
            onStep: { i in
                if i < steps {
                    RunProgress.report(.denoise, step: i + 1, totalSteps: steps)
                } else {
                    RunProgress.report(.decode)
                }
                try Task.checkCancellation()
            }
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

extension MossSoundEffectPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MossSoundEffectPackage.self)
    }
}
