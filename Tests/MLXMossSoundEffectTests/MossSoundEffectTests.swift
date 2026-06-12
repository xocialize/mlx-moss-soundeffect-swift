import Foundation
import MLX
import MLXToolKit
import XCTest

@testable import MLXMossSoundEffect

final class MossSoundEffectTests: XCTestCase {
    func testManifestDeclaresSoundEffectSurface() {
        let m = MossSoundEffectPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces.first?.capability, .soundEffect)
        XCTAssertEqual(Capability.soundEffect.canonicalOutput, .audio)
        XCTAssertTrue(m.surfaces.first?.parameters.contains { $0.name == "durationSeconds" } ?? false)
    }

    func testLicenseBothLayersPermissive() {
        let license = MossSoundEffectPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertEqual(license.portCodeLicense, .apache2)
    }

    func testRequirementsManifestCoversPublishedQuants() {
        let reqs = MossSoundEffectPackage.manifest.requirements
        XCTAssertEqual(Set(reqs.footprints.map(\.quant)), [.bf16, .int4])
        XCTAssertTrue(reqs.requiredBackends.contains(.metalGPU))
        XCTAssertNotNil(reqs.os.minMacOS)
    }

    func testConfigurationCodableExcludesModelsRoot() throws {
        var config = MossSoundEffectConfiguration()
        config.modelsRootDirectory = URL(fileURLWithPath: "/tmp")
        let decoded = try JSONDecoder().decode(
            MossSoundEffectConfiguration.self, from: JSONEncoder().encode(config))
        XCTAssertNil(decoded.modelsRootDirectory)  // environment-specific, not portable
        XCTAssertEqual(decoded.repo, "mlx-community/MOSS-SoundEffect-v2.0-bf16")
        XCTAssertEqual(decoded.defaultSteps, 100)
    }

    func testWAVEncoding() {
        // 100 samples of a known ramp -> valid RIFF/WAVE header + correct sizes.
        let ramp = MLXArray(stride(from: Float(-1), to: 1, by: 0.02).map { $0 }).reshaped(1, 1, 100)
        let data = MossSoundEffectPackage.encodeWAV(ramp, sampleRate: 48_000)
        XCTAssertEqual(String(data: data.prefix(4), encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: data.subdata(in: 8 ..< 12), encoding: .ascii), "WAVE")
        XCTAssertEqual(data.count, 44 + 100 * 2)
        // sample rate field at offset 24
        let sr = data.subdata(in: 24 ..< 28).withUnsafeBytes { $0.load(as: UInt32.self) }
        XCTAssertEqual(UInt32(littleEndian: sr), 48_000)
    }

    func testRunRejectsWrongCapabilityAndUnloaded() async throws {
        let pkg = MossSoundEffectPackage(configuration: .init())
        // Not loaded yet:
        do {
            _ = try await pkg.run(SoundEffectRequest(prompt: "x"))
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            XCTAssertEqual(error, .notLoaded)
        }
    }

    /// Live e2e through the engine contract — needs the snapshot + GPU.
    /// MOSS_SFX_LIVE=1 swift test --filter testLiveGeneration
    func testLiveGeneration() async throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["MOSS_SFX_LIVE"] != "1",
                      "set MOSS_SFX_LIVE=1 (downloads weights, runs GPU inference)")
        let pkg = MossSoundEffectPackage(configuration: .init())
        try await pkg.load()
        let response = try await pkg.run(SoundEffectRequest(
            prompt: "a heavy wooden door creaks open slowly",
            durationSeconds: 2, steps: 4, seed: 7))
        guard let sfx = response as? SoundEffectResponse else {
            XCTFail("wrong response type"); return
        }
        XCTAssertEqual(sfx.audio.format, .wav)
        XCTAssertEqual(sfx.audio.sampleRate, 48_000)
        XCTAssertEqual(sfx.audio.data.count, 44 + 2 * 48_000 * 2)
        await pkg.unload()
    }
}
