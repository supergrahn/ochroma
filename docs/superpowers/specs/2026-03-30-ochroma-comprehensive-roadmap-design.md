# Ochroma Engine — Comprehensive Production Roadmap

**Date:** 2026-03-30
**Status:** Approved
**Approach:** Domain Completion — each domain reaches production quality before the next begins
**Target audience:** Open-source engine for external developers to adopt
**Platform scope:** Windows / Linux / Mac desktop (web via WebGPU; consoles deferred)

---

## Context

Ochroma is a spectral Gaussian splatting engine (~165K LOC, 15 crates, 479 files). Most core domains are 80–95% complete. This roadmap covers the remaining gaps to reach production quality across all 12 domains.

The engine's differentiator is spectral Gaussian splatting — 8 spectral bands (380–700nm) driving rendering, physics damage, audio synthesis, material fingerprinting, and AI perception simultaneously. Every domain spec and example game must make this differentiator *visible and meaningful*, not incidental.

---

## Why This Engine Can Surpass Unreal

Unreal Engine's advantages are real but they are ecosystem and momentum advantages — not architectural ones. Its core weaknesses are structural and not patchable:

- **Lumen's GI is a screen-space approximation** built on a rasterizer with no concept of wavelength. Physical spectral accuracy requires replacing the renderer, not extending it. They cannot patch their way there.
- **Blueprint's runtime slowness is architectural.** It compiles to interpreted bytecode. Fifteen years of optimization have not changed the fundamental model.
- **Their networking predates QUIC by 15 years.** The abstractions are wrong at the protocol level.
- **C++ is not patchable to memory safety.** Every year the Ochroma codebase grows, the safety gap widens.
- **The 8-band spectral space is not addable.** Ochroma's renderer, physics, audio, AI perception, and material recognition all share the same representation from day one. Unreal's systems were built in isolation over 25 years and speak different languages. A spectral rendering mode can be bolted on; spectral coherence across all systems cannot.

Gaussian splatting is also the right bet for the next decade of content creation: as photogrammetry and NeRF-to-splat pipelines mature, the ability to capture the real world at spectral fidelity becomes more valuable than any artist-created asset library. Megascans is impressive today. A spectral capture pipeline that produces physically accurate materials from a phone camera is the future.

The goal of this roadmap is not parity with Unreal. It is to surpass Unreal on the axes that define what rendering engines will look like in 2030.

---

## Execution Order

```
Build/Platform → Audio → UI → Scripting → Asset Pipeline → Rendering →
Networking → Character → Editor → Physics → AI/LLM → Spectral Frontier
```

Docs are written *per domain* as each completes. The getting-started guide and contributor guide are written after Build/Platform — the first point at which there is something runnable to document.

Rationale: Build/Platform first because developers cannot evaluate or contribute without being able to compile and run the engine. Each subsequent domain is ordered from user-facing gaps (audio, UI, scripting) down to advanced internals (physics, AI).

---

## Domain 1 — Build/Platform

**Current state:** 85% — Linux primary, Windows/Mac partial, web is framework only, no CI, no crates.io publishing.

**Completion spec:**

- **cargo-dist** for cross-platform artifact generation, installers, and GitHub release attachments. Replaces the ad-hoc `scripts/build_release.sh` and `scripts/package.sh`.
- **Web:** Target WebGPU directly (stable across Chrome/Firefox/Safari by 2026). The software rasterizer already present serves as fallback for non-WebGPU browsers. No separate WASM-specific code path needed beyond the `--features web` flag.
- **CI:** GitHub Actions matrix job (ubuntu-latest / windows-latest / macos-latest):
  - `cargo test --workspace`
  - `cargo clippy --workspace --deny warnings`
  - `cargo doc --no-deps`
  - cargo-dist dry-run
  - One integration smoke test per binary (engine_runner, walking_sim, platformer)
- **Publishing:** `ochroma_engine` façade crate + all domain crates published as `ochroma-*` via cargo-dist's crates.io workflow.
- **Docs checkpoint:** Getting-started guide (install → first scene → first game in <15 minutes) and CONTRIBUTING.md written and verified runnable on all three platforms.

**Completion criterion:** `cargo dist build` produces installable artifacts for all three desktop platforms and a working web bundle that loads in Chrome.

---

## Domain 2 — Audio

**Current state:** 85% — rich DSP math (HRTF, SDF reverb, adaptive music, AV sync) but the rodio backend is Linux-only in practice, and the `AudioGraph` DSP chain is custom with no signal combinator model.

**Completion spec:**

- **CPAL** as the cross-platform device backend (WASAPI on Windows, CoreAudio on Mac, ALSA/PipeWire on Linux). Rodio becomes an optional feature; CPAL is the default.
- **fundsp** for DSP signal graph. Replace the custom `AudioGraph` node implementations with fundsp signal chains. fundsp's combinator model (`>>`, `&`, `|` operators over signal generators) maps directly to the graph architecture already designed. Gain, EQ, compressor, reverb send, and HRTF insert become fundsp graph nodes.
- **File playback:** `.wav` via hound, `.ogg` via lewton — no external runtime dependencies.
- **Docs checkpoint:** Audio API reference: device setup, spatial audio, DSP graph, HRTF usage.

**Completion criterion:** Footstep `.wav` plays with spatial HRTF falloff and room reverb on Windows, Mac, and Linux simultaneously.

---

## Domain 3 — UI

**Current state:** 75% — custom `UiRoot` renderer with 4 widget types (Panel, Text, Button, Slider), no layout system, no data binding, no game UI / editor UI split.

**Completion spec:**

- **Vello** (Linebender's GPU vector renderer, production-quality by 2026) as the 2D rendering backend for game UI. Renders to a wgpu texture composited into the final frame.
- **Parley** for text layout (pairs with Vello; handles shaping, line breaking, bidi).
- **Taffy** for layout (flexbox engine, pure Rust). Flex-row / flex-column with min/max size constraints — not full CSS, just enough for HUDs and menus.
- **New widgets:** `Dropdown`, `TreeView`, `Table`, `Tooltip`, `ProgressBar`, `ColorPicker`.
- **Reactive data binding:** `Bindable<T>` wrapper that auto-queues a redraw when the inner value changes.
- **Clean split:** game UI renders through `vox_render::spatial_ui` via Vello/Taffy; editor UI stays on egui permanently. No mixing.
- **Docs checkpoint:** UI widget reference with code examples for each widget type.

**Completion criterion:** A full in-game HUD (health bar, minimap, inventory slot grid, notification toasts) buildable entirely from `vox_ui` without touching egui.

---

## Domain 4 — Scripting

**Current state:** 85% — Rhai runtime with hot-reload, but `spawn()` returns a stub entity ID, API surface is narrow, and Rhai is niche (not known to most game developers).

**Completion spec:**

- **Replace Rhai with mlua (Lua 5.4).** Lua is the de facto game scripting standard (Roblox, Love2D, Defold, Neovim). Larger ecosystem, better-known by game developers, better performance for hot loops, native coroutine library.
- Visual scripting graph compiles to Lua as its backend target (replacing Rhai codegen).
- WASM sandbox via wasmtime remains for untrusted plugin code.
- **Expanded engine API:** `physics.raycast()`, `animation.play_clip(entity, clip_name)`, `audio.play(path, position)`, `scene.find_by_name(name)`, `scene.spawn(prefab)` returning a real `EntityId`.
- **Coroutines:** `wait_frames(n)` and `wait_seconds(t)` implemented as Lua coroutine yields coordinated by a per-script frame scheduler in the engine.
- **Visual scripting node library expanded:** Event nodes (`OnCollide`, `OnTrigger`, `OnTimer`), Action nodes (`MoveToward`, `LookAt`, `PlayEffect`, `SetVisible`).
- **Docs checkpoint:** Scripting guide covering Lua API reference, coroutine patterns, and visual scripting node catalogue.

**Completion criterion:** The walking-sim game logic (collect orbs, play sound, trigger music state, win condition) fully authored in a `.lua` script with zero Rust changes.

---

## Domain 5 — Asset Pipeline

**Current state:** 90% — GLTF import exists but produces "reference quality" splats (vertex colors only, no texture sampling); no photogrammetry entrypoint.

**Completion spec:**

- **Texture import:** `TextureImporter` samples UV-mapped albedo texture pixels at each splat's surface position, converts the sampled RGB to an 8-band spectral estimate via `SpectralUpliftLut`, writes result into `GaussianSplat.spectral`. This makes imported scenes spectrally meaningful, not just gray.
- **GLTF seeding quality:** Mesh-to-splat seeding uses surface normals + albedo texture sampling (not just vertex colors). Splat scale is derived from local surface curvature.
- **Batch import CLI:** `ochroma-tools import --gltf model.glb --out scene.vxm`
- **Hot-reload:** Modifying a `.glb` triggers re-import and live scene update within 1 second via the existing `vox_data::hot_reload` file watcher.
- **Photogrammetry entrypoint:** `ochroma-tools capture --images ./photos/ --out scene.vxm` — a thin subprocess wrapper around COLMAP (user installs COLMAP separately). COLMAP handles camera calibration and sparse reconstruction; the engine converts the resulting point cloud to a splat cloud and runs spectral uplift on any reference images. No Rust-native SfM — production-quality structure-from-motion is not feasible in pure Rust at this time.
- **Docs checkpoint:** Asset pipeline guide: GLTF import workflow, texture requirements, COLMAP capture setup, `.vxm` format reference.

**Completion criterion:** Stanford Bunny `.glb` produces a visually recognizable splat cloud with correct spectral color mapping. COLMAP capture pipeline documented end-to-end with a worked example.

---

## Domain 6 — Rendering

**Current state:** 85% — material graphs evaluated on CPU (not compiled to shaders), SVT is skeleton code, DOF has stub buffer bindings, denoiser is framework only.

**Completion spec:**

- **Material shader compilation:** `MaterialGraph::compile()` produces a `naga::Module` (naga is wgpu's own shader IR — type-safe, validated, consumed directly by wgpu without string WGSL generation). `GpuMaterial` caches the compiled pipeline. Graph edits trigger live recompilation.
- **SVT (Sparse Virtual Texturing):** `SvtCache` streams 128×128 tiles from disk on demand, evicts LRU entries when the cache budget is exceeded. Used initially for terrain albedo.
- **DOF pass:** Complete CoC (circle of confusion) computation from depth buffer + bokeh scatter pass. Resolve the stub buffer bindings.
- **Denoiser:** Use **candle** (HuggingFace's pure-Rust ML framework) to run a small learned denoising CNN (U-Net style) in-process for the offline render path. No OIDN FFI, no Python dependency. Model weights distributed as a bundled safetensors file (GGUF is for LLMs; CNNs use safetensors).
- **Docs checkpoint:** Rendering architecture overview; material graph tutorial (build a two-node material from scratch); offline render guide.

**Completion criterion:** A custom two-node material (base color × roughness mask) compiles via naga and renders correctly on the Stanford Bunny import. Offline render of a scene with a character produces a denoised output image.

---

## Domain 7 — Networking

**Current state:** 80% — TCP-only transport (plain text, no encryption), no UDP option, no packet recovery strategy, no rate limiting.

**Completion spec:**

- **Drop TCP entirely. Use Quinn (QUIC) for all transport.** QUIC provides TLS 1.3 encryption, multiplexed streams, and UDP-level performance in a single protocol. This eliminates the need for a separate TLS layer and a separate UDP transport.
  - Reliable ordered streams → lobby, auth, asset sync
  - Unreliable datagrams → game state (position, splat deltas); packet loss is handled by rollback, not retransmission
- **Packet recovery:** Handled natively by QUIC on reliable streams. Unreliable datagrams are intentionally not recovered — the rollback netcode in `vox_net::rollback` already handles divergence from dropped game state packets.
- **Rate limiting:** Per-connection token bucket in `NetworkConfig { max_bytes_per_sec: u32 }`.
- **Docs checkpoint:** Networking architecture doc; multiplayer setup guide (host a server, connect two clients).

**Completion criterion:** Two instances of the walking-sim connect over QUIC, see each other move in real time, and recover cleanly from a simulated 500ms packet loss burst without desync.

---

## Domain 8 — Character Controller

**Current state:** 80% — the custom `CharacterController` in `vox_core` detects ground by comparing Y position to the flat plane `y <= height * 0.5 + 0.05`. It has no collision detection against actual geometry (terrain, buildings, physics bodies).

**Completion spec:**

- **Migrate to Rapier's `KinematicCharacterController`** (`rapier3d::control::KinematicCharacterController`) for all movement resolution. Rapier's KCC performs real capsule sweeps against the physics world, handles arbitrary terrain geometry, slopes, and steps correctly.
- **Keep existing math helpers** (`is_walkable_slope`, `compute_slope_slide`, `slide_along_wall`, `try_step_up`) as utility functions that feed the context move predicates — they are not deleted, just no longer the ground truth for collision.
- **Context moves** added as state predicates on top of the KCC:
  - **Vault:** triggered when sprinting + jumping toward an obstacle ≤1.2m tall; character repositions over it playing a vault animation clip
  - **Mantle:** triggered when a jump apex reaches a ledge within arm's reach; plays a pull-up animation
  - **Ledge-hang:** sustained grip state on a ledge edge; shimmy left/right while hanging
  - **Wall-climb:** vertical surface detection, limited by a stamina float that drains while climbing
- `CharacterController::evaluate_context() -> ContextMove` enum drives state selection.
- **Docs checkpoint:** Character controller guide: setup, input binding, context move configuration.

**Completion criterion:** A test level with a wall, a ledge, a low barrier, and a climbable vertical surface — character navigates all four context moves correctly without changing input bindings.

---

## Domain 9 — Editor

**Current state:** 90% — `vox_render::gizmos` is 200+ lines of state management; verify it is wired into the wgpu render pass. Bone editing, vertex painting, and cage deformation are absent.

**Completion spec:**

- **Verify and fix gizmo wiring:** Confirm `gizmos.rs` state drives actual draw calls in the wgpu render pass (not just state management). Fix if the render pass is not consuming gizmo geometry.
- **Bone gizmos:** Per-joint rotate handles overlaid on the skinned mesh in the scene viewport. Dragging a handle updates the joint's local rotation and drives the animation system live.
- **Vertex paint:** Brush tool writes per-splat color overrides into a `SplatOverrideLayer` stored alongside the base splat buffer. Override layer is blended at render time. Undo/redo via the existing `vox_core::undo` system.
- **Cage deformer:** Place a low-poly control cage around a splat cloud. Moving cage vertices deforms interior splats via trilinear interpolation of cage-space coordinates. Useful for coarse shape editing without modifying the source asset.
- **Docs checkpoint:** Editor user guide: scene navigation, gizmos, bone editing, vertex paint, cage deform.

**Completion criterion:** An artist can rig a character, paint vertex colors, and cage-deform a prop entirely within the editor without writing or modifying any Rust code.

---

## Domain 10 — Physics

**Current state:** 95% — Rapier integration is solid. GPU fluid, ragdoll automation, and runtime destruction are framework-only.

**Completion spec:**

- **GPU fluid — Position-Based Fluids (PBF)** on a wgpu compute shader. PBF is preferred over SPH: better incompressibility, more numerically stable, cleaner GPU implementation. Target: 50k fluid particles at 60fps on a mid-range GPU (RTX 3060 class).
  - Compute passes: density estimation → lambda solve → position correction → velocity update
  - Fluid particles rendered as spectral splats (spectral color driven by fluid temperature / composition)
- **Ragdoll:** `RagdollBuilder::from_skeleton(skeleton: &Skeleton) -> RagdollConfig` auto-generates Rapier rigid bodies and joints from a bone hierarchy. Joint limits derived from bone orientation ranges. Activated on `DeathEvent`.
- **Runtime destruction:** Voronoi fracture computed at import time and stored in the `.vxm` asset. `DestructibleBody::fracture_at(impulse: f32, point: Vec3)` activates the pre-fractured pieces as independent Rapier rigid bodies. Spectral damage is applied to each fragment at fracture time.
- **Docs checkpoint:** Physics feature reference: rigid bodies, character controller, fluids, ragdoll, destruction.

**Completion criterion:** A destructible crate shatters on impact with spectral damage applied to fragments, a ragdoll character collapses on death, and a PBF fluid emitter splashes — all simultaneously at 60fps.

---

## Domain 11 — AI/LLM

**Current state:** 70% — LLM client uses remote API only (requires API key and network), NPC dialogue framework only, scene graph disconnected from render world.

**Completion spec:**

- **Local LLM via candle** (HuggingFace's pure-Rust ML framework). `LlmBackend::Local` loads a GGUF-format model (llama3-8b, phi-3-mini, or gemma-2b) in-process. No external process, no Python, no ollama dependency. Engine selects local backend automatically if a model file is present in `~/.ochroma/models/`. `LlmBackend::Remote(OpenAiClient)` remains as fallback when no local model is available.
- **NPC dialogue:** `DialogueTree` driven by LLM completions. Responses cached per NPC ID and conversation context to avoid redundant inference. Graceful fallback to static canned lines when inference budget is exceeded or LLM unavailable.
- **Scene graph ↔ render bridge:** `SceneGraph::sync_to_world()` writes entity positions, materials, and spectral data back into the ECS render world so LLM-generated scene layouts become visible.
- **Docs checkpoint:** AI/LLM integration guide: local model setup, NPC dialogue authoring, text-to-city usage.

**Completion criterion:** An NPC in the walking-sim holds a short context-aware conversation using in-process llama3 GGUF inference, with automatic fallback to canned lines when no model file is present.

---

## The Three Example Games

These exist specifically to make the spectral differentiator *visible*. A developer's first contact with the engine must show what spectral splatting means in practice.

### `examples/hello_splat`
Static scene: a single `.vxm` file loaded, orbit camera, no game logic. Features demonstrated:
- Spectral tonemapping (watch the scene color shift as you scrub the tone curve)
- 8-band viewport overlay showing per-band energy in real time as the camera orbits
- Platform: runs in browser via WebGPU

Purpose: the first thing a new developer sees. It shows the spectral pipeline, not a gray box.

### `examples/walking_sim`
Character controller, ambient audio, collectible orbs, win condition, all game logic in Lua. Features demonstrated:
- Orbs are spectrally distinct materials: metal orbs (cold blue-violet spectral profile), fire orbs (warm red-orange). `SpectralFingerprintDb::classify()` identifies which type was collected.
- Collecting a fire orb triggers a combustion `AvEvent` via `AvSyncProcessor` — plays a crackle sound, shifts adaptive music to `MusicState::Combat` via `AdaptiveMusicPlayer`, applies `DamageType::Fire` spectral shift to nearby splats.
- Spatial HRTF audio: footsteps and ambient sounds positioned in 3D via `SpatialHrtfMixer`.
- All game logic in `.lua` (zero Rust changes to add a new game mechanic).

Purpose: shows fingerprinting + AV sync + adaptive music + scripting working together in a playable game.

### `examples/spectral_showcase`
Non-interactive fly-through. Features demonstrated:
- A rusting metal beam: `DamageType::Rust` spectral shift accumulating over simulated time
- A skin-material character: subsurface scattering driven by spectral band 2–4 energy
- A fire emitter: spectral splat particles + HRTF spatial audio panned as the camera moves past
- SDF soft shadows: the fire casts penumbra shadows on surrounding surfaces
- A debug overlay in the corner shows live 8-band spectral energy as a bar chart that changes as the camera moves through spectrally distinct regions

Purpose: a demo reel. Shows every "what makes this different" feature in one uninterrupted sequence.

---

---

## Domain 12 — Spectral Frontier

**Current state:** The spectral pipeline exists at the rendering level. It does not yet drive light transport dynamically, does not capture real-world spectral reflectance from photographs, and does not couple optical properties to physics behavior. These are the features that make Ochroma architecturally superior to any rasterizer-first engine — they need to be built to production quality.

**Completion spec:**

### 12a — Real-Time Spectral Global Illumination

Unreal's Lumen traces screen-space rays and approximates indirect lighting. Ochroma can do better: each Gaussian splat already carries an 8-band spectral emission and reflectance profile. Real-time spectral GI propagates actual wavelength-dependent radiance between splats.

- **`SpectralRadianceCache`**: a spatial hash of splat clusters. Each cluster stores an incoming radiance estimate per spectral band, updated each frame from neighbouring emissive splats.
- **Propagation pass** (compute shader): for each splat, gather radiance from its `N` nearest emissive neighbours (N=8, distance-weighted). Attenuate by the receiving splat's per-band reflectance. Accumulate into a temporal buffer with exponential moving average (α=0.1) for stability.
- **Output**: each splat's `spectral` field at render time = base reflectance × (sun contribution + radiance cache contribution). The spectral tonemapper already consumes this correctly — no render pipeline changes needed.
- **Performance target**: 500k splats, full radiance propagation, <3ms on an RTX 3060 compute pass.
- Physical accuracy claim: a red wall illuminated by a blue light source will show correct spectral mixing (purple-shifted reflectance). Lumen cannot do this without per-wavelength ray tracing.

### 12b — Spectral Atmosphere (Rayleigh + Mie per wavelength)

Unreal's sky atmosphere is a single-scattering approximation with RGB values. Ochroma can simulate the actual physics: Rayleigh scattering is proportional to λ⁻⁴, meaning violet (380nm) scatters ~9× more than red (700nm). This is why the sky is blue. It is *not* a texture or a gradient — it is a wavelength-dependent physical process.

- **`SpectralAtmosphere`**: replaces the existing `atmosphere.rs` with a physically grounded model.
- Per-frame sky radiance computed as: for each of the 8 spectral bands, apply Rayleigh coefficient `β_R(λ) = 8π³(n²-1)²/(3Nλ⁴)` and Mie coefficient `β_M` (wavelength-independent, particle-size dependent). Sun angle drives optical depth per band.
- Output: an 8-band sky radiance value fed into the spectral GI cache as the primary light source.
- Result: correct blue sky, correct orange sunset, correct purple twilight — from physics, not from artist-painted sky spheres.
- **Aerosol support**: `AerosolProfile { particle_radius_nm, density }` modulates Mie scattering for haze, fog, volcanic ash.

### 12c — Spectral Material Capture Pipeline

Photogrammetry today captures RGB. Ochroma should capture spectral reflectance — the actual optical properties of a material across the 8 bands, not an artist's approximation of it.

- **Multi-band capture protocol**: photograph subjects under 3 known light conditions (neutral daylight, warm tungsten, cool LED). These three images, combined with the known spectral power distributions of the light sources, overdetermine the 8-band reflectance via a least-squares solve.
- **`SpectralCaptureProcessor`**: reads 3 RAW photographs + 3 `LightSpd` spectral power distributions → outputs a `SpectralMaterialProfile { reflectance: [f32; 8], variance: [f32; 8] }`.
- **CLI**: `ochroma-tools capture-spectral --images a.dng b.dng c.dng --lights daylight.json tungsten.json led.json --out material.spm`
- **Integration with asset pipeline**: `.spm` files are referenced by `GaussianSplat.spectral` during import, replacing the `SpectralUpliftLut` approximation with measured data where available.
- Implication: a marble floor captured with this pipeline reflects light *exactly as marble does*, not as an artist imagined it. No other shipping engine can claim this.

### 12d — Spectral Resonance Physics

Materials have optical-acoustic coupling: their spectral absorption profile is related to their molecular structure, which also determines their acoustic resonance and fracture behaviour. This is not an approximation — it is solid-state physics.

- **`SpectralResonanceProfile`**: derived from a material's `[u16; 8]` spectral data. Metal (low absorption in visible, high reflectance) has high stiffness and metallic ring. Glass (sharp absorption edges) has brittle fracture and high-frequency resonance. Wood (chlorophyll-like mid-band absorption) has organic fracture patterns.
- **`SpectralFracture::compute_planes(splat_cloud, impact: Vec3, impulse: f32) -> Vec<FracturePlane>`**: fracture planes are oriented along the principal stress axes, modulated by the material's resonance profile. Crystalline materials (high spectral regularity) fracture in planes. Amorphous materials (irregular spectral profile) fracture in curves.
- **Integration with `vox_physics::destruction`**: `DestructibleBody::fracture_at()` calls `SpectralFracture::compute_planes()` instead of pure Voronoi. Pre-fracture geometry is generated at import time but plane orientations are spectral-material-dependent, not random.
- **Acoustic emission**: fracture events emit sound via the AV sync system using the material's resonance frequency (derived from peak spectral band). A glass window shattering produces a high-frequency ring; a wooden plank breaking produces a low-frequency crack.

### 12e — Spectral Neural Compression

The 8-band spectral representation is 4× larger than RGB. For streaming and storage, this needs to compress efficiently while preserving physical accuracy. RGB compression (JPEG, BC7) destroys spectral information. A learned spectral codec can outperform both.

- **`SpectralCodec`**: a small candle-based autoencoder (encoder: 8→4 latent, decoder: 4→8 reconstruction). Trained on the engine's own spectral data distribution.
- **Compression target**: 4 latent floats per splat instead of 8 unsigned shorts — 50% size reduction with <2% mean spectral error.
- **`SpectralCodec::encode(spectral: [f32; 8]) -> [f16; 4]`** and **`::decode(latent: [f16; 4]) -> [f32; 8]`**
- **Usage**: `.vxm` format v3 stores latent codes; decompression happens on GPU during the splat upload pass. Network replication sends latent codes (8 bytes vs 16 bytes per spectral field).
- **Training data**: generated from `SpectralUpliftLut` on a large set of RGB images — no external dataset needed.

---

**Domain 12 completion criterion:** A real-world object photographed under 3 light conditions produces a `.spm` material profile. That profile is applied to a splat cloud, which then participates in the real-time spectral GI pass, shatters along spectrally-determined fracture planes, emits acoustically correct sound at fracture, and transmits over the network using spectral neural compression — all simultaneously, at 60fps.

---

## Cross-Cutting Requirements

**Testing:** Each domain must ship with tests covering its completion criterion scenario, not just unit tests of internal functions. Integration tests live in `tests/` adjacent to the relevant crate.

**Error handling:** All public APIs return `Result` with typed errors (using `thiserror`). No `unwrap()` or `expect()` in library code paths (only in tests and binary entry points).

**Performance budgets:**
- Rendering: 60fps at 1080p on RTX 3060 class hardware with 500k splats visible
- Spectral GI: <3ms per frame for radiance propagation pass (500k splats)
- Spectral atmosphere: <0.5ms per frame
- Audio: <2ms latency on CPAL callback thread
- Physics: 50k PBF particles + full Rapier world at 60fps
- Scripting: Lua frame budget <1ms per entity per frame for typical game logic
- Spectral neural compression: encode/decode <0.1ms per frame for active splat set

**Spectral invariant:** Every system that touches a `GaussianSplat` must preserve or intentionally modify its `.spectral: [u16; 8]` field. Systems must not zero out or ignore spectral data as a convenience shortcut.

---

## Out of Scope for This Roadmap

- Console targets (PS5, Xbox Series X, Switch) — deferred post-v1.0
- Mobile (iOS, Android) — deferred post-v1.0
- Multiplayer voice chat
- LLM training / fine-tuning (inference only)
- Rust-native SfM (not feasible at production quality currently; COLMAP subprocess used instead)
- Full neural Gaussian splatting training pipeline (3DGS optimization from raw images — compute-intensive, Python ecosystem owns this; Ochroma imports the output)
- Steam achievements / leaderboards (framework exists; integration deferred)
