# Ochroma Engine — Comprehensive Production Roadmap

**Date:** 2026-03-30
**Status:** Approved
**Approach:** Domain Completion — each domain reaches production quality before the next begins
**Target audience:** Open-source engine for external developers to adopt
**Platform scope:** Windows / Linux / Mac desktop (web via WebGPU; consoles deferred)

---

## Context

Ochroma is a spectral Gaussian splatting engine (~165K LOC, 15 crates, 479 files). Most core domains are 80–95% complete. This roadmap does not aim for parity with Unreal Engine. It aims to surpass it on the axes that define what game engines will look like in 2030.

The engine's core invariant: **the 8-band spectral representation is the lingua franca of every system**. Rendering, physics, audio, scripting, networking, AI perception, and authoring tools all speak the same language. This is not a feature — it is an architectural decision that compounds across every domain and cannot be replicated by bolting spectral awareness onto 25-year-old isolated subsystems.

---

## Why This Engine Surpasses Unreal

Unreal's advantages are ecosystem and momentum — not architecture. Its structural weaknesses cannot be patched:

- **Lumen is a screen-space heuristic.** Physically correct spectral GI requires replacing the renderer. They cannot get there from here.
- **Blueprint is interpreted at runtime.** This is not a performance bug — it is a design decision baked into the bytecode model.
- **Their networking stack predates QUIC by 15 years.** The abstractions are wrong at the protocol level.
- **C++ cannot be made memory-safe.** The safety gap widens with every line of Ochroma that ships.
- **Their spectral coverage is zero.** Ochroma's renderer, physics, audio, AI, and editor all share `[u16; 8]` spectral data from day one. You cannot add spectral coherence to systems built in isolation. You can only rebuild from scratch — which is what this engine is.
- **Megascans is impressive today.** A spectral capture pipeline that produces physically accurate material profiles from three phone photographs is the future. The Megascans library is measured under controlled conditions and mapped to RGB. Ochroma's capture pipeline measures actual reflectance curves. The data is better at the source.

The goal of this roadmap: every domain is designed to surpass Unreal on its own terms, not merely to match it. Where Unreal has a feature, Ochroma has the physics behind it.

---

## Execution Order

```
Build/Platform → Audio → UI → Scripting → Asset Pipeline → Rendering →
Networking → Character → Editor → Physics → AI/LLM → Spectral Frontier
```

Docs are written per domain as each completes. Getting-started guide and CONTRIBUTING.md after Build/Platform — first point at which something runnable exists.

---

## Domain 1 — Build/Platform

**Current state:** 85% — Linux primary, Windows/Mac partial, web is framework only, no CI, no crates.io publishing.

**Ambition:** The easiest engine to get started with, on any platform, in any browser. A developer goes from `git clone` to a spectrally-correct scene running in Chrome in under 15 minutes — with no local GPU required. No other engine ships a physically-accurate spectral rendering demo that runs in a browser tab.

**Completion spec:**

- **cargo-dist** replaces ad-hoc scripts. Cross-platform artifacts, installers, GitHub release attachments, crates.io publish in a single workflow.
- **Web:** WebGPU is stable across Chrome/Firefox/Safari by 2026. Target it directly. Software rasterizer remains as fallback. The `hello_splat` web demo must show something genuinely novel — spectral band scrubbing, real-time atmospheric shift — not just "a triangle loads."
- **CI:** GitHub Actions matrix (ubuntu/windows/macos):
  - `cargo test --workspace`
  - `cargo clippy --workspace --deny warnings`
  - `cargo doc --no-deps`
  - cargo-dist dry-run
  - Spectral regression benchmark: measures GI propagation time and asserts it stays under budget. CI fails on performance regression, not just test failure.
- **Publishing:** `ochroma_engine` + all `ochroma-*` domain crates via cargo-dist crates.io workflow.
- **Docs checkpoint:** Getting-started guide verified runnable on all three platforms. CONTRIBUTING.md with architecture overview and spectral invariant explanation.

**Completion criterion:** `cargo dist build` produces installable artifacts for all three platforms. `hello_splat` loads in Chrome and demonstrates spectral band isolation with a single button press — something Unreal cannot ship in a browser.

---

## Domain 2 — Audio

**Current state:** 85% — rich DSP math (HRTF, SDF reverb, adaptive music, AV sync) but rodio backend is Linux-only in practice, and `AudioGraph` has no signal combinator model.

**Ambition:** Audio that emerges from physics, not from a sound bank. No pre-recorded samples required for basic gameplay sounds. Every surface has an intrinsic acoustic signature derived from its spectral material profile — strike it, shatter it, walk on it, and the sound is synthesized from what it *is*, not from what an audio designer recorded. Unreal requires a sound bank. Ochroma requires a spectral material.

**Completion spec:**

- **CPAL** as cross-platform device backend (WASAPI/CoreAudio/ALSA). Rodio becomes optional.
- **fundsp** for DSP signal graph. fundsp's combinator model (`>>`, `&`, `|`) replaces custom `AudioGraph` nodes. Gain, EQ, compressor, reverb send, HRTF insert are fundsp nodes.
- **Spectral material synthesis:** `SpectralSynth::strike(material_spectral: &[u16; 8], impulse: f32) -> AudioSignal` — synthesizes a physically plausible impact sound from spectral material properties. Resonance frequency from `SpectralResonanceProfile::from_spectral()`, harmonic decay envelope from stiffness, brightness from short-wavelength reflectance. No WAV file needed for metal clangs, wooden thuds, or glass rings.
- **Spectral reverb from GI cache:** room reverb tail per spectral band derived from the surrounding splats' reflectance profiles. Stone walls (high uniform reflectance) → long reverb. Fabric-covered surfaces (mid-band absorption) → short, warm reverb. The reverb emerges from the spectral geometry of the space, not from a preset.
- **Spectral field-driven ambient soundscape:** `SpectralSoundscape` monitors the local spectral energy field (from `SpectralRadianceCache`) and continuously adjusts ambient sound mix. Entering a region with high red-band energy (fire, heat) raises combustion ambience. Entering green-band (vegetation, calm) shifts to natural soundscape. No scripted triggers needed.
- **File playback:** `.wav` via hound, `.ogg` via lewton — for music and voice acting.
- **Docs checkpoint:** Audio guide with worked examples: synthesizing a glass impact, building a spectral reverb chain, writing a Lua script that responds to spectral audio events.

**Completion criterion:** A glass object shatters with a high-frequency ring synthesized from its spectral profile. A stone room has longer reverb than a carpeted room, derived automatically from their spectral geometry. Both work on Windows, Mac, and Linux. No WAV files used for either effect.

---

## Domain 3 — UI

**Current state:** 75% — 4 widget types, no layout system, no data binding, no game/editor UI split.

**Ambition:** A UI system that understands the spectral world it lives in. HUDs that respond to the physical state of the scene — not through scripted hooks, but because the UI renderer has access to spectral data and can visualize it directly. The spectral band overlay and material classifier view are first-class UI primitives, not debug tools.

**Completion spec:**

- **Vello** as 2D GPU vector rendering backend for game UI. **Parley** for text layout. **Taffy** for flexbox layout (row/column, min/max constraints).
- **Widget library:** `Dropdown`, `TreeView`, `Table`, `Tooltip`, `ProgressBar`, `ColorPicker`, `SpectralBandDisplay` (8-bar spectral energy visualizer), `MaterialClassBadge` (shows `SpectralFingerprintDb` classification for a selected object).
- **Reactive data binding:** `Bindable<T>` auto-queues redraw on value change.
- **Spectral-aware HUD:** `SpectralHud` widget reads from `SpectralRadianceCache` and renders a real-time band energy display. Developers can bind any HUD element's color temperature to the dominant spectral band in the scene — a health bar that subtly shifts warmer as nearby fire intensity increases, derived from physics not from scripted state.
- **Adaptive HUD tint:** global HUD tint responds to scene spectral state — combat (red-dominant) shifts HUD warm, stealth (blue-dominant) shifts cool, exploration (green-dominant) shifts neutral. Emerges from spectral GI data, requires zero scripting.
- **Clean split:** game UI via Vello/Taffy; editor UI stays on egui permanently.
- **Docs checkpoint:** UI guide with spectral widget examples; tutorial building a HUD that responds to spectral scene state.

**Completion criterion:** A full HUD — health bar, spectral band display, material classifier badge for the currently-aimed-at object, notification toasts — built entirely in `vox_ui` without touching egui. The health bar color-shifts based on scene spectral energy automatically.

---

## Domain 4 — Scripting

**Current state:** 85% — Rhai runtime with hot-reload, stub entity handles, narrow API surface.

**Ambition:** Spectral data is a first-class scripting primitive. A Lua script can read, write, and respond to spectral conditions in the world without named trigger zones or scripted state machines. Game logic that says "play fire audio when the nearby spectral field has band 7 > 0.8" is responding to actual physics — and it's three lines of Lua.

**Completion spec:**

- **Replace Rhai with mlua (Lua 5.4).** Industry standard for game scripting. Native coroutines. Larger ecosystem.
- Visual scripting compiles to Lua. WASM sandbox stays for plugins.
- **Spectral API — first-class primitives in Lua:**
  - `splat.spectral[b]` — read/write individual spectral bands on any entity
  - `spectral.classify(entity)` → `MaterialClass` string
  - `spectral.damage(entity, "fire", intensity)` — apply spectral damage
  - `spectral.field_energy(position, radius, band)` → float — query spectral energy in a region
  - `spectral.on_threshold(position, radius, band, threshold, callback)` — register a callback that fires when spectral conditions are met in a region. No trigger zones, no scripts polling every frame — the engine calls back when physics says so.
- **Full engine API:** `physics.raycast()`, `animation.play_clip(entity, clip_name)`, `audio.play(path, position)`, `audio.synthesize_material(entity, impulse)`, `scene.find_by_name(name)`, `scene.spawn(prefab)` → real `EntityId`.
- **Coroutines:** `wait_frames(n)` / `wait_seconds(t)` as Lua coroutine yields.
- **Visual scripting spectral nodes:** `SpectralThreshold` (fires event when band exceeds value), `SpectralClassify` (outputs material class), `SpectralDamage` (applies damage), `SpectralField` (samples energy in radius).
- **Docs checkpoint:** Scripting guide with spectral API reference. Tutorial: write a trap that activates when a fire-spectrally-classified object enters its radius — entirely in Lua, no Rust.

**Completion criterion:** A game mechanic where fire orbs trigger audio synthesis, spectral damage on nearby splats, and music state transition — entirely authored in `.lua` using `spectral.on_threshold()` and `spectral.damage()`. Zero Rust changes. The script is responding to physics, not to named game events.

---

## Domain 5 — Asset Pipeline

**Current state:** 90% — GLTF produces reference-quality splats (vertex colors only); no photogrammetry.

**Ambition:** Every imported asset is spectrally annotated automatically. An artist imports a GLTF and every surface region is classified, tagged with a physical material profile, and ready to participate in spectral GI, resonance physics, and material-driven audio — without manual annotation. The pipeline makes the spectral world, not the artist.

**Completion spec:**

- **Texture import:** `TextureImporter` UV-samples albedo texture → per-splat RGB → `SpectralUpliftLut` → `GaussianSplat.spectral`. Imported scenes are spectrally meaningful immediately.
- **GLTF seeding quality:** Surface normals + albedo texture sampling. Splat scale from local surface curvature.
- **Automatic spectral classification on import:** after texture-to-spectral conversion, `SpectralFingerprintDb::classify()` runs on each surface region. Classification result tags the region's `MaterialClass`, which automatically assigns: resonance profile (for fracture physics), reverb response (for audio), damage susceptibility (for `SpectralDamageComponent`). No artist annotation required for standard materials.
- **Progressive spectral enhancement:** assets import at "spectral level 1" (fast, uplift-approximated) immediately. A background task runs `SpectralCaptureProcessor` refinement if reference photographs are available, upgrading to "spectral level 3" (measured). The `.vxm` format stores which level each splat cluster has reached.
- **Batch import CLI:** `ochroma-tools import --gltf model.glb --out scene.vxm`
- **Photogrammetry:** `ochroma-tools capture --images ./photos/ --out scene.vxm` via COLMAP subprocess. COLMAP for calibration; engine converts point cloud to spectrally-annotated splats.
- **Hot-reload:** `.glb` modification triggers re-import and live scene update in <1 second.
- **Docs checkpoint:** Asset pipeline guide: import workflow, spectral classification tagging, progressive enhancement, COLMAP setup.

**Completion criterion:** Stanford Bunny `.glb` imports with correct spectral surface profiles, automatic `MaterialClass::Stone` classification, and correct resonance profile — produces physically correct fracture sound on impact without any artist configuration. COLMAP capture pipeline documented end-to-end.

---

## Domain 6 — Rendering

**Current state:** 85% — material graphs CPU-only, SVT skeleton, DOF stub bindings, denoiser framework.

**Ambition:** The renderer expresses physics that no rasterizer can represent. Every rendering feature is grounded in the spectral representation — materials, lighting, caustics, dispersion, and denoising all operate in the 8-band spectral space. Viewing modes expose the underlying physics directly, making Ochroma the only engine where an artist can literally see the wavelength-dependent behavior of their scene.

**Completion spec:**

- **Material shader compilation:** `MaterialGraph::compile()` → `naga::Module` consumed directly by wgpu. No string WGSL generation. Graph edits trigger live recompilation. Material nodes operate on 8-band spectral values, not RGB — a `SpectralAbsorption` node applies per-band extinction, not a color multiply.
- **Spectral caustics:** light passing through transmissive materials (glass, water) refracts each spectral band by a different angle (dispersion). The caustic pattern separates into spectral components. This is rainbow caustics from first principles — not a post-process effect, not a baked texture.
- **Spectral emission as illumination:** any splat with non-zero emissive spectral bands is automatically a light source contributing to the `SpectralRadianceCache`. No separate light entity needed. A fire particle illuminates its surroundings because its spectral emissive value is non-zero — by physics, not by configuration.
- **Viewing modes:** `SpectralViewMode` enum — `Physical` (normal rendering), `BandIsolate(u8)` (shows single band energy as grayscale), `FalseColor` (maps band distribution to visible color), `MaterialClass` (false-colors by `SpectralFingerprintDb` classification), `AsSeenBy(species)` where species can be `Human`, `Bee` (UV-sensitive), `Mantis` (16 bands mapped from 8). No other engine ships a "bee's-eye view" rendering mode.
- **SVT:** `SvtCache` streams 128×128 tiles with LRU eviction.
- **DOF:** Complete CoC + bokeh scatter pass. Spectral bokeh: each band's CoC is slightly different (chromatic aberration from dispersion), producing physically correct lens blur color fringing.
- **Denoiser:** candle U-Net CNN (safetensors weights) for offline render path.
- **Docs checkpoint:** Rendering architecture overview. Tutorial: build a spectral glass material that produces rainbow caustics. Guide: using spectral viewing modes for debugging.

**Completion criterion:** A glass prism scatters a white light beam into a rainbow pattern — caustics showing band-separated colors, derived from Snell's law applied per spectral band, not from a texture. Bee's-eye view rendering mode shows UV-range spectral data (band 0 mapped to visible). Material graph compiles via naga with live hot-reload.

---

## Domain 7 — Networking

**Current state:** 80% — TCP-only, plain text, no recovery, no rate limiting.

**Ambition:** The most bandwidth-efficient game networking stack for any splat-based world. Spectral neural compression means Ochroma transmits 50% less data per splat than any RGB-based engine. Spectral relevance filtering means clients only receive splat updates that are physically visible or audible to them — a better relevance filter than geometry-based visibility culling because it accounts for spectral occlusion (smoke obscuring in specific bands, darkness suppressing all bands).

**Completion spec:**

- **QUIC (Quinn) for all transport.** TCP dropped entirely. TLS 1.3 built-in. Reliable streams for lobby/auth/assets; unreliable datagrams for game state. No separate TLS layer.
- **Spectral-compressed replication:** splat delta packets transmit `SpectralCodec` latent values (4 floats) instead of raw 8-band data (8 uint16s) — 8 bytes vs 16 bytes per updated splat. 50% spectral bandwidth reduction compared to any engine transmitting RGB or full spectral data.
- **Spectral relevance filtering:** `SpectralRelevanceFilter` evaluates whether a splat's spectral energy is above perceptual threshold for a given client's position and orientation. Splats that are spectrally dark (negligible energy in all 8 bands from client's perspective) are culled from replication entirely. This is a physics-based LOD system for network traffic, not an artist-placed visibility volume.
- **Deterministic spectral simulation:** spectral GI propagation and spectral damage are deterministic given the same inputs. The rollback system can resimulate spectral state exactly — not approximately. When a client corrects from a server rollback, the spectral appearance of the world is guaranteed to match.
- **Spectral state broadcast:** the server publishes `SpectralRadianceCache` region updates so all clients share the same dynamic lighting state. A fire explosion that illuminates a room spectrally is seen by all clients simultaneously, derived from the same physics.
- **Rate limiting:** per-connection token bucket in `NetworkConfig`.
- **Docs checkpoint:** Networking architecture guide. Tutorial: build a two-player scene where a fire explosion dynamically illuminates both clients' views simultaneously via spectral state broadcast.

**Completion criterion:** Two clients connected over QUIC see a fire explosion illuminate a room — the spectral GI update propagates to both clients simultaneously with spectral-compressed replication. Bandwidth measured and confirmed <50% of equivalent RGB replication. 500ms packet loss burst recovered without desync.

---

## Domain 8 — Character Controller

**Current state:** 80% — flat-plane ground detection only, no actual physics collision.

**Ambition:** Movement that is aware of the physical world it traverses, derived from spectral material properties — not from scripted zones or named surface types. Ice is slippery because its spectral profile matches the `MaterialClass::Ice` reflectance signature, not because an artist tagged it "slippery_surface." Fire hurts because standing in high red-band spectral energy applies `DamageType::Fire` — the same spectral data driving the renderer is driving the character's physical response.

**Completion spec:**

- **Rapier `KinematicCharacterController`** for movement resolution. Real capsule sweeps against physics world. Custom KCC math helpers kept as utilities.
- **Spectral surface response:** `CharacterController::update()` queries the `SpectralRadianceCache` at foot position each frame. The dominant `MaterialClass` of the surface underfoot drives:
  - Movement speed multiplier (ice → 1.4×, sand → 0.7×, normal → 1.0×)
  - Friction coefficient (ice → near-zero, carpet → high)
  - Damage accumulation rate (`DamageType::Fire` if standing in high band-7 energy)
  - No scripted zones. No artist tags. The spectral field drives all of it.
- **Footstep synthesis from spectral material:** `audio.synthesize_material(surface_splat, footstep_impulse)` called on each footstep contact. Sound derived from `SpectralResonanceProfile` of the surface. No footstep sound bank needed.
- **Context moves:** vault (≤1.2m obstacle + sprint+jump), mantle (ledge pull-up), ledge-hang (shimmy), wall-climb (stamina drain). All trigger context-appropriate animation clips.
- **Spectral camouflage detection:** `CharacterController` reports its spectral signature to the AI perception system — a character standing in shadow has a different spectral profile than one standing in firelight. AI agents perceive characters spectrally, not by name-tag.
- **Docs checkpoint:** Character controller guide: setup, spectral surface response configuration, footstep synthesis, context moves.

**Completion criterion:** Character walks onto an ice surface (classified from spectral profile), slides with reduced friction and slipping animation. Walks into fire zone (high band-7 energy), takes damage and plays combustion sound — synthesized from surface spectral profile. All derived from spectral physics, no scripted zones.

---

## Domain 9 — Editor

**Current state:** 90% — gizmos are state management only, bone editing absent, vertex paint absent.

**Ambition:** The first game editor where artists work directly with physical light. Painting a surface means painting spectral reflectance — choosing how the surface responds to each wavelength, not picking an RGB color approximation. A spectral material library populated with measured real-world profiles means artists paint with physics, not with guesswork.

**Completion spec:**

- **Verify and fix gizmo wiring:** confirm gizmos.rs state drives wgpu draw calls. Fix if render pass is not consuming gizmo geometry.
- **Bone gizmos:** per-joint rotate handles on skinned mesh, live animation preview on drag.
- **Spectral paint** (replaces simple vertex paint): brush tool writes per-band spectral values directly. The color picker shows a true spectral curve editor — drag the curve per band, not a hex color. A `SpectralSwatchLibrary` provides measured profiles: "Aged Bronze," "Fresh Snow," "Dry Concrete," "Oak Wood" — each derived from `SpectralMaterialProfile` data, physically accurate. Drag a swatch onto a surface to apply its measured spectral reflectance.
- **Spectral viewing modes in editor:** toggle between Physical / BandIsolate / MaterialClass views mid-edit. See immediately which regions are mis-classified, which bands are incorrectly painted, where spectral energy is leaking. No guess-and-check with RGB approximations.
- **Live spectral GI preview:** editor renders spectral GI in real time as the artist edits. Placing an emissive splat immediately shows its influence on surrounding materials spectrally. No "bake lighting" step — the GI is live.
- **Cage deformer:** control cage around splat cloud, trilinear interior deformation.
- **Undo/redo:** all spectral paint operations integrated with `vox_core::undo`.
- **Docs checkpoint:** Editor guide: gizmos, spectral paint, spectral swatch library, viewing modes, live GI preview, cage deform.

**Completion criterion:** Artist paints a rust damage effect using spectral curve editing — drags the band 0–2 absorption up, shifts band 5–6 reflectance to simulate iron oxide absorption. Views result in MaterialClass mode to confirm it classifies as `Rust`. Applies "Aged Bronze" swatch from library to another surface. All in editor, zero Rust code.

---

## Domain 10 — Physics

**Current state:** 95% — Rapier solid, GPU fluid/ragdoll/destruction are framework-only.

**Ambition:** Physics that changes the appearance of the world and the world's appearance changes the physics. Fluid particles carry spectral composition — blood in water shifts the fluid red spectrally, which the renderer shows, which the AI perceives, which changes NPC behavior. Thermal emission from hot objects illuminates neighboring surfaces via GI. Fracture sounds are synthesized from material spectral profiles. The physical world and the spectral world are the same world.

**Completion spec:**

- **PBF GPU fluid:** 50k particles at 60fps on RTX 3060. Fluid particles carry `spectral: [f32; 8]` composition. Water: high bands 2–3. Blood: shifts bands 5–7 red. Fire smoke: absorbs mid-bands, making objects behind smoke spectrally desaturated correctly. Fluid spectral values contribute to `SpectralRadianceCache` — a pool of glowing fluid illuminates its surroundings.
- **Thermal dynamics:** objects with high temperature (tracked as `thermal_energy: f32` on physics bodies) emit in bands 5–7 (near-infrared analog). Emission rate feeds into `SpectralRadianceCache` — hot objects heat neighboring materials spectrally. Cooling is tracked per-frame. A forge heats the sword inside it; the sword glows red-orange because of heat energy in bands 5–7 visible in spectral GI, not because an artist set it to glow.
- **Spectral resonance destruction:** `DestructibleBody::fracture_at()` calls `SpectralFracture::compute_planes()` — fracture planes from optical-acoustic coupling. Acoustic emission synthesized via `SpectralSynth` from material resonance profile.
- **Ragdoll:** `RagdollBuilder::from_skeleton()` auto-generates Rapier bodies + joints. Joint stiffness modulated by material spectral profile (rigid crystal vs flexible organic). Activated on `DeathEvent`. On ragdoll impact, `SpectralSynth::strike()` synthesizes impact sound.
- **Docs checkpoint:** Physics reference. Tutorial: build a destructible forge — it heats the sword inside spectrally, the sword glows, the forge shatters with material-correct fracture sound, the molten metal splashes with spectral fluid composition.

**Completion criterion:** A forge heats a sword via thermal spectral emission (no scripted glow). The forge shatters with resonance-correct fracture planes and synthesized sound. Molten metal splashes as PBF fluid with high-band spectral emission illuminating nearby surfaces. All simultaneously at 60fps.

---

## Domain 11 — AI/LLM

**Current state:** 70% — remote LLM only, NPC dialogue framework, scene graph disconnected from render world.

**Ambition:** AI agents that perceive, reason about, and are fooled by the spectral world — exactly as any physical sensor would be. An NPC doesn't know "that's fire" from a game tag. They detect high band-7 energy in their perception radius and reason from that physical observation. They can be deceived by spectral camouflage. Their emotional state is driven by the spectral environment around them. The LLM that generates their dialogue has access to the actual physical state of the world.

**Completion spec:**

- **Local LLM via candle** — in-process GGUF model inference (llama3-8b, phi-3-mini). No external process. `LlmBackend::Remote` stays as fallback.
- **Spectral perception:** NPCs perceive the world through `SpectralPerceptionComponent`. Their sight, hearing, and "threat detection" are all spectral:
  - `spectral.field_energy(npc_pos, sight_range, band)` — what they can spectrally "see"
  - High band-7 energy nearby → fire detected → flee or fight response
  - High band-0 energy (violet/UV) → unusual lighting → curiosity or fear
  - Object spectral profile matching `MaterialClass::Skin` in sight cone → character detected
  - **Spectral camouflage works:** a player wearing a `MaterialClass::Stone` spectral profile in a stone room is harder to detect — the NPC's perception threshold rises because the spectral signature blends with background
- **Spectral emotion:** `NpcEmotionalState` derives from surrounding `SpectralRadianceCache`. Dominant red-band energy → anxiety. Dominant green-band → calm. Dominant violet-band → unease. This drives dialogue tone, movement speed, and behavioral tree priorities — without scripted triggers.
- **LLM with spectral context:** NPC dialogue prompts include current spectral state: "The forge fire emits strong red-orange energy. The sword inside glows. Nearby splats show thermal heating." The LLM generates contextually aware responses grounded in the actual physical state of the scene.
- **Scene graph ↔ render bridge:** `SceneGraph::sync_to_world()` writes positions, materials, and spectral data back into the ECS render world.
- **Docs checkpoint:** AI guide: spectral perception setup, spectral camouflage mechanics, emotional state from spectral environment, LLM dialogue with spectral context injection.

**Completion criterion:** NPC detects player via spectral skin-profile recognition (not name tag). Player hides in shadow — NPC's detection threshold rises because player's spectral signature decreases. NPC emotional state shifts to anxious because nearby forge emits high red-band energy. NPC dialogue reflects actual physical scene state via LLM with spectral context.

---

## The Three Example Games

These must make spectral splatting viscerally, immediately obvious to any developer who opens them for the first time.

### `examples/hello_splat`

Static scene, orbit camera, no game logic. Runs in browser via WebGPU.

- **Spectral band scrubber:** a slider isolates individual spectral bands. Move it and watch the scene transform — metals disappear in some bands, foliage pops in others, fire becomes invisible in short-wavelength bands. This is impossible in any other engine because no other engine has per-band scene data.
- **Atmospheric shift:** a time-of-day control shifts sun angle. Watch the scene colors change as Rayleigh scattering shifts — correct blue sky midday, orange sunset, purple twilight. From physics, not from color grading.
- **No download.** Runs at `ochroma.dev/hello_splat` in Chrome.

Purpose: the first thing any developer sees. Within 30 seconds they understand what spectral splatting means and why it matters.

### `examples/walking_sim`

Character controller, spatial audio, collectible orbs, win condition, all game logic in Lua.

- Metal orbs (cold blue-violet profile) ring metallically when collected — sound synthesized from spectral resonance profile, no WAV file.
- Fire orbs (high band-7) apply `DamageType::Fire` to nearby splats, synthesize combustion sound from spectral resonance, shift adaptive music to `MusicState::Combat`, and illuminate the surrounding scene via spectral GI — all from one spectral event.
- Walking on different surfaces produces different footstep sounds synthesized from their spectral material profiles.
- The HUD tints warmer as fire orbs are collected because spectral GI energy increases in bands 5–7.
- All game logic in `.lua` using `spectral.on_threshold()` callbacks.

Purpose: shows that spectral coherence *simplifies* game development. Fewer scripts, no trigger zones, no named material tags — just physics.

### `examples/spectral_showcase`

Non-interactive fly-through. Designed as a tech demo / release trailer capture.

- **Prism scene:** white light entering a glass prism separates into a rainbow caustic — each spectral band refracted at a different angle, from Snell's law, not from a texture.
- **Forge scene:** a hot forge heats a sword spectrally. The sword begins glowing in bands 5–7 as thermal energy accumulates. The forge is struck and shatters — spectral resonance fracture planes, synthesized metallic crash. Molten metal splashes as PBF fluid with spectral emission.
- **Forest at sunset:** Rayleigh atmosphere shifts from neutral to orange. Foliage spectral profile (strong mid-band) pops against the warm sky. An NPC walks through — their spectral skin profile is distinct against the vegetation background.
- **Spectral band overlay:** corner display shows 8-band energy as bars, updating in real time as the camera flies through each scene.

Purpose: shows what the engine looks like when all 12 domains are complete. This is the "impossible in Unreal" moment.

---

## Domain 12 — Spectral Frontier

**Current state:** Spectral pipeline exists at rendering level. Does not yet drive light transport dynamically, does not capture real-world spectral reflectance, does not couple optical properties to physics behavior.

**Completion spec:**

### 12a — Real-Time Spectral Global Illumination

- **`SpectralRadianceCache`:** spatial hash of splat clusters, per-band radiance estimate updated each frame.
- **Propagation compute pass (WGSL):** gather radiance from N nearest emissive neighbours, attenuate by per-band reflectance, accumulate with exponential moving average (α=0.1).
- **Output:** each splat's render-time spectral value = base reflectance × (solar irradiance + GI cache). Spectral tonemapper consumes unchanged.
- **Spectral sun:** `SpectralAtmosphere::solar_irradiance()` seeds the GI cache as primary light source each frame.
- **Performance:** <3ms for 500k splats on RTX 3060 compute pass.
- **Physical claim:** red wall lit by blue source shows purple-shifted reflectance. Lumen cannot reproduce this.

### 12b — Spectral Atmosphere (Rayleigh + Mie per wavelength)

- `SpectralAtmosphere` with Rayleigh β(λ) = β_ref × (550nm/λ)⁴ and Mie coefficient per band.
- Produces correct blue sky, orange sunset, purple twilight — from physics, not artist-painted sky spheres.
- `AerosolProfile { particle_radius_nm, density }` for haze, fog, volcanic ash.
- Output feeds directly into spectral GI as primary illuminant.

### 12c — Spectral Material Capture

- 3-photo protocol under neutral/tungsten/cool-LED light conditions.
- `SpectralCaptureProcessor::from_three_images()` → `SpectralMaterialProfile { reflectance: [f32; 8], variance: [f32; 8] }`.
- `.spm` binary format (64 bytes per profile).
- CLI: `ochroma-tools capture-spectral --images a.dng b.dng c.dng --lights daylight.json tungsten.json led.json --out material.spm`
- Replaces `SpectralUpliftLut` approximation with measured data where available.

### 12d — Spectral Resonance Physics

- `SpectralResonanceProfile::from_spectral()` — resonance frequency, regularity, stiffness from spectral data.
- `SpectralFracture::compute_planes()` — axis-aligned planes for crystalline materials, curved for amorphous.
- Wired into `DestructibleBody::fracture_at()`.
- Acoustic emission via `SpectralSynth` at fracture time.

### 12e — Spectral Neural Compression

- `SpectralCodec` — 8→4 linear autoencoder (PCA-derived weights initially, candle-trained weights as upgrade).
- 50% spectral data size reduction. <2% mean spectral error.
- Used in `.vxm` v3 format and network replication.

**Domain 12 completion criterion:** A real-world object photographed under 3 conditions produces a `.spm` profile. That profile drives spectral GI, shatters along resonance fracture planes, emits acoustically correct sound, is transmitted via neural-compressed replication — simultaneously, at 60fps.

---

## Cross-Cutting Requirements

**Testing:** Each domain ships tests covering its completion criterion scenario, not just unit tests. Integration tests in `tests/` adjacent to the crate.

**Error handling:** All public APIs return `Result` with typed errors (`thiserror`). No `unwrap()` or `expect()` in library paths.

**The spectral invariant:** Every system that touches a `GaussianSplat` must preserve or intentionally modify its `.spectral: [u16; 8]` field. Zeroing spectral data for convenience is a spec violation.

**Performance budgets:**
- Rendering: 60fps at 1080p, RTX 3060, 500k splats
- Spectral GI propagation: <3ms per frame (500k splats)
- Spectral atmosphere: <0.5ms per frame
- Audio synthesis: <1ms per impact event
- Physics: 50k PBF particles + full Rapier world at 60fps
- Scripting: Lua frame budget <1ms per entity
- Spectral neural compression: encode/decode <0.1ms per frame for active splat set
- Network spectral replication: <50% bandwidth of equivalent RGB replication

**The standard:** every domain spec should be answerable with "Unreal cannot do this" — not "Unreal does this differently." If a feature is only as good as Unreal's equivalent, it needs to be redesigned until the physics behind it makes it genuinely superior.

---

## Out of Scope for This Roadmap

- Console targets (PS5, Xbox Series X, Switch) — deferred post-v1.0
- Mobile (iOS, Android) — deferred post-v1.0
- Multiplayer voice chat
- LLM training / fine-tuning (inference only)
- Rust-native SfM (COLMAP subprocess used; pure-Rust SfM not production-quality yet)
- Full 3DGS training pipeline (Python ecosystem owns this; Ochroma imports the output)
- Steam achievements / leaderboards (framework exists; deferred)
