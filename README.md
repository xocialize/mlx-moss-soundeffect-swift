# mlx-moss-soundeffect-swift

The MLXEngine **`soundEffect`** capability package (contract 1.3.0 — first package for the
capability) over [MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0):
text → sound effect (foley / ambience / creature / action), 48 kHz `.wav`, up to 30 s, EN/ZH.

A thin conformance layer wrapping the standalone inference engine
[moss-soundeffect-mlx-swift](https://github.com/xocialize/moss-soundeffect-mlx-swift)
(parity-locked against the Python oracle in
[moss-soundeffect-mlx](https://github.com/xocialize/moss-soundeffect-mlx)); all model logic
lives there. This package adds the engine-owned lifecycle (C13), the canonical
`SoundEffectRequest`/`SoundEffectResponse` surface, license declaration, the requirements
manifest, Hub weight download, and WAV artifact encoding.

## Weights (auto-downloaded on `load()`)

| Repo | DiT | Working set (M5 Max, 100 steps) |
|---|---|---|
| [mlx-community/MOSS-SoundEffect-v2.0-bf16](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-bf16) | bf16 | ~14.2 GB, 60 s |
| [mlx-community/MOSS-SoundEffect-v2.0-4bit](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-4bit) | int4 g64 | ~12.2 GB, 45 s |

## Use

```swift
import MLXMossSoundEffect
import MLXToolKit

let package = MossSoundEffectPackage(configuration: .init())  // or repo: "…-4bit"
try await package.load()
let response = try await package.run(SoundEffectRequest(
    prompt: "a heavy wooden door creaks open slowly",
    durationSeconds: 5, seed: 42)) as! SoundEffectResponse
// response.audio: canonical Audio (.wav, 48 kHz mono)
```

Behavior notes: the model always denoises a fixed 30 s latent and crops to
`durationSeconds` (duration is conditioned via a trained prompt suffix); an empty
`negativePrompt` is the trained unconditional path; cancellation is honored per
denoising step (~0.5 s granularity).

Apache-2.0 (both layers: weights and port code).
