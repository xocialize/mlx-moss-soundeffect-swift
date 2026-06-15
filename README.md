# mlx-moss-soundeffect-swift

The MLXEngine **`soundEffect`** capability package (contract 1.3.0 — first package for the
capability) over [MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0):
text → sound effect (foley / ambience / creature / action), 48 kHz `.wav`, up to 30 s, EN/ZH.

A thin conformance layer (`MossSoundEffectPackage`, module **`MLXMossSoundEffect`**) wrapping the
standalone inference engine
[moss-soundeffect-mlx-swift](https://github.com/xocialize/moss-soundeffect-mlx-swift) (product
`MossSoundEffectMLX`, pinned `from: 0.1.2`), which is itself parity-locked against the Python oracle
in [moss-soundeffect-mlx](https://github.com/xocialize/moss-soundeffect-mlx). All model logic lives
in the engine repo; this package adds the engine-owned lifecycle (C13), the `SoundEffectRequest` /
`SoundEffectResponse` surface, the `PackageManifest` (license + requirements), Hub weight download,
and WAV artifact encoding.

> Repo relationship (three repos, deliberately distinct):
> - **moss-soundeffect-mlx** — Python MLX parity oracle + weight conversion + golden fixtures.
> - **moss-soundeffect-mlx-swift** — standalone MLX-Swift inference engine (the math).
> - **mlx-moss-soundeffect-swift** (this repo) — MLXEngine capability wrapper (lifecycle + I/O contract).

## Weights (auto-downloaded on `load()`)

| Repo | DiT | Working set (M5 Max, 100 steps) |
|---|---|---|
| [mlx-community/MOSS-SoundEffect-v2.0-bf16](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-bf16) | bf16 | ~14.2 GB, 60 s |
| [mlx-community/MOSS-SoundEffect-v2.0-4bit](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-4bit) | int4 g64 | ~12.2 GB, 45 s |

`MossSoundEffectConfiguration` defaults to the `-bf16` repo; set `repo: "…-4bit"` to switch
(quantization is auto-detected from `model_index.json`). It also carries `defaultSteps` (100),
`defaultGuidanceScale` (4.0), and `defaultDurationSeconds` (10.0), applied when a request leaves
them unset.

## Use

```swift
import MLXMossSoundEffect
import MLXToolKit

let package = MossSoundEffectPackage(configuration: .init())  // or .init(repo: "…-4bit")
try await package.load()
let response = try await package.run(SoundEffectRequest(
    prompt: "a heavy wooden door creaks open slowly",
    durationSeconds: 5, seed: 42)) as! SoundEffectResponse
// response.audio: canonical Audio (.wav, 48 kHz mono)
```

Behavior notes: the model always denoises a fixed 30 s latent and crops to
`durationSeconds` (duration is conditioned via a trained prompt suffix); an empty
`negativePrompt` is the trained unconditional path; cancellation is honored per
denoising step (`onStep` → `Task.checkCancellation()`, ~0.5 s granularity).

## Dependencies / platform

- `mlx-engine-swift` (`MLXToolKit`, `MLXServeCore` for tests) — local-path dep for in-workspace dev
- `moss-soundeffect-mlx-swift` (`MossSoundEffectMLX`) — tagged release
- `mlx-swift`, `swift-transformers` (`Hub`)

Platform: macOS 26+. Swift language mode v5 (the pipeline's MLX/tokenizer state is not yet
Sendable-audited; the engine serializes all lifecycle on its inference actor).

Apache-2.0 (both layers: weights and port code).
