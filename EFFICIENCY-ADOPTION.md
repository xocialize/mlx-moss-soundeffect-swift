# Efficiency Adoption Brief — `mlx-moss-soundeffect-swift` (MOSS SoundEffect, `soundEffect`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + ALL of **"Measurement
> findings"**) + references/memory-harness.md. The MEATY one of this batch — a **diffusion** model with a real
> transient + a possible **encoder-evict**. Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXMossSoundEffect` (`MossSoundEffectPackage: ModelPackage`). Capability **`soundEffect`**
  (text→sound-effect generation). Engine pinned `from: "0.3.0"`. Component: `pipeline: MossSoundEffectPipeline?`.
- **Footprints today (FLAT residentBytes only, NO transient):** bf16 **15 GB** · int4 **13 GB**.
- The comment says it's a **"100-step CFG diffusion over a 1500-token latent"** → a real multi-step denoise
  transient that the flat number currently bakes into residency.
- `unload()` (~line 81) — verify it `MLX.Memory.clearCache()`s.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.17.0 | **P0** |
| 1. Split footprint | ❌ | flat 15/13 GB, transient (100-step denoise) baked in | **P1 (headline)** |
| 2. Per-stage evict | ❓ INVESTIGATE | text→sfx implies a **text conditioner** that encodes the prompt ONCE then idles through the 100-step denoise — if so, evict it before the loop (Lens/Boogu pattern). Read `MossSoundEffectPipeline`. | **P2 (if a conditioner exists)** |
| 3. mmap/lazy | 🟡 verify | confirm lazy load (no eager full copy) | note |
| 4. BudgetAware | 🟡 maybe | bf16/int4 quant lever; defer unless trivial | defer |

## Plan
- **P0:** `swift package update` → 0.17.0; build + fix any drift.
- **P2 (INVESTIGATE FIRST):** read `MossSoundEffectPipeline`. If a text/prompt conditioner encodes once upfront
  and is idle through the 100-step CFG denoise, refactor to load → encode → `eval`/retain → **evict** (`nil` +
  `Memory.clearCache()`) before the denoise loop (the proven encoder-evict pattern; Swift 6 `#isolation` gotcha
  if it goes async — `isolated (any Actor)? = #isolation`). If the components interleave per step, P2 is N/A —
  note the reason.
- **P1:** split per quant. `residentBytes` = the weights resident through the denoise (post-evict if P2 applies);
  `peakActivationBytes` = the **100-step CFG denoise + decode transient** at the 1500-token latent envelope —
  measure it (the denoise working set, not whole-model). Adopt `QuantConfigured` (bf16/int4).
- **`unload()` must `MLX.Memory.clearCache()`**.

## Measurement — IMPORTANT
This is the heaviest of the batch (15 GB + a 100-step denoise). Declare `residentBytes` from the measured
weight floor (solid) + a **FLAGGED** `peakActivationBytes` from the smoke (in-app phys ~2.5–2.9× higher —
admission basis). A 100-step diffusion smoke may be slow but shouldn't trip the watchdog like video; if it does,
fall back to weight-floor + flag and don't fight it. Note the 1500-token / 100-step envelope as the activation
driver.

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; P2 (encoder-evict if a conditioner exists, else N/A-with-reason); P1
      split per quant; `unload()` clearCache.
- [ ] residentBytes = weight floor (post-evict if P2); peakActivationBytes = denoise transient (FLAGGED).
- [ ] Smoke green (valid sound-effect audio); split recorded; activation flagged.
- [ ] Registry: moss-soundeffect row Eff ⬜→✅ (note "P2 finding; activation phys re-baseline pending"), Eng→0.17.0.

## Report back
flat→split per quant, the P2 conditioner-evict finding (+ effect if applied), the denoise transient (flagged),
drift, effort, commit SHA. STAY IN SCOPE — four-lever adoption + this brief + registry row only; verify
`git show --stat`; stop-and-report if the refactor needs to touch the diffusion math or anything bigger.
