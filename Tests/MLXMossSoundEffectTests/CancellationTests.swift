// CancellationTests.swift — MOSS-SoundEffect through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before
// notLoaded validation or weights); CAN-3 is the document of record for the checkpoint cadence:
// the wrapper passes a throwing `onStep` closure (`try Task.checkCancellation()`) that the core
// (moss-soundeffect-mlx-swift ≥ 0.2.0, Pipeline.generate) fires before EVERY CFG denoise step
// (100 by default, ~0.5 s each at bf16) and once more before the final 30 s fp32 VAE decode —
// the CancellationError rethrows unchanged through the core's rethrows chain.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXMossSoundEffect

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = MossSoundEffectPackage(configuration: MossSoundEffectConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: SoundEffectRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // soundEffect is a long-run capability (and the 10 GB declared transient independently
        // implies long runs) — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MossSoundEffectPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: MossSoundEffectPackage.manifest,
            posture: .cadence([
                // The throwing onStep hook fires before each CFG denoise step
                // (MossSoundEffectPackage.run → Pipeline.generate denoise loop).
                .init(phase: .denoise, unit: .step),
                // ... and once more immediately before the whole-latent VAE decode — one seam,
                // one chunk: the 30 s decode itself is a single monolithic eval (core ≥ 0.2.0,
                // Pipeline.generate pre-decode fire).
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
