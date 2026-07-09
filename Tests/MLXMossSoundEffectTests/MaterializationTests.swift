// MaterializationTests.swift — MOSS-SoundEffect through the engine's MAT gate (offline, no
// network): the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction,
// and the store-layout probe/resolution. One consolidated repo (DiT/VAE + Qwen3 text_encoder +
// tokenizer all ride the same snapshot) — one declaration per configured repo.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXMossSoundEffect

final class MaterializationTests: XCTestCase {

    /// Temp dir holding probe files (nested repo layout) that make an explicit-dir config read
    /// as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "moss-mat-\(UUID().uuidString)")
        for sub in ["mlx", "text_encoder", "tokenizer"] {
            try FileManager.default.createDirectory(
                at: dir.appending(path: sub), withIntermediateDirectories: true)
        }
        for f in MossSoundEffectConfiguration.requiredFiles {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Engine MAT gate

    func testMATGate() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: MossSoundEffectConfiguration(),
            satisfiedConfiguration: MossSoundEffectConfiguration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testDeclaresSingleConsolidatedSource() {
        let sources = MossSoundEffectConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main"])
        XCTAssertEqual(sources[0].repo, "mlx-community/MOSS-SoundEffect-v2.0-bf16")
        XCTAssertEqual(sources[0].matching, MossSoundEffectConfiguration.snapshotGlobs)
        // A 4bit repo choice keys the declaration by repo (same layout, different checkpoint).
        let int4 = MossSoundEffectConfiguration(repo: "mlx-community/MOSS-SoundEffect-v2.0-4bit")
        XCTAssertEqual(int4.weightSources[0].repo, "mlx-community/MOSS-SoundEffect-v2.0-4bit")
        XCTAssertEqual(int4.quant, .int4)
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "moss-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = MossSoundEffectConfiguration()
        // Empty store: the source is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // Populate the expected (nested) layout.
        let dir = root.appending(path: cfg.repo)
        for sub in ["mlx", "text_encoder", "tokenizer"] {
            try FileManager.default.createDirectory(
                at: dir.appending(path: sub), withIntermediateDirectories: true)
        }
        for f in MossSoundEffectConfiguration.requiredFiles {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout; an explicit dir always wins.
        XCTAssertEqual(cfg.resolved(storeRoot: root).modelDirectory?.path, dir.path)
        let explicit = MossSoundEffectConfiguration(modelDirectory: URL(fileURLWithPath: "/x"))
            .resolved(storeRoot: root)
        XCTAssertEqual(explicit.modelDirectory?.path, "/x")
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = MossSoundEffectConfiguration(modelsRootDirectory: root)
        XCTAssertEqual(cfg.prewarmPaths.map(\.path),
                       [root.appending(path: "mlx-community/MOSS-SoundEffect-v2.0-bf16").path])
    }

    func testCodableRoundTrip() throws {
        let cfg = MossSoundEffectConfiguration(defaultSteps: 50,
                                               modelDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(MossSoundEffectConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertEqual(decoded.defaultSteps, 50)
        XCTAssertNil(decoded.modelDirectory)   // environment-specific, never encoded
    }
}
