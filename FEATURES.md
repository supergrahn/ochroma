# Ochroma Engine — Features

A spectral Gaussian-splatting game engine in Rust. Light is carried as **16 spectral
bands (380–755 nm)** end-to-end — through materials, GI, atmosphere, physics, audio,
and even AI perception — rather than RGB.

Status legend: **live** = wired into a shipped binary and exercised by a headless
smoke gate · **library** = implemented + tested, not yet driving a shipped binary ·
**experimental** = present, depth-limited.

Maintained as features land. Last updated: 2026-06-07.

---

## Rendering

| Feature | Status | Description |
|---|---|---|
| Anisotropic 3DGS rasteriser | live | True EWA per-splat projection with front-to-back **16-band spectral** per-pixel compositing (CPU reference path; `OCHROMA_RASTER=legacy\|gaussian` A/B lever) |
| GPU rasteriser (wgpu) | live | Depth-sorted splat upload + tile-based EWA compute path; indexed rendering from LOD selections |
| Spectra Vulkan path | library | `spectra-native` feature: path-traced spectral rendering via the sibling Spectra renderer (renders on RADV iGPU) |
| Atom-budget LOD selector | live | "Nanite for splats": cluster-BVH frustum cull → solid-angle scoring → per-cluster LOD with a hard per-frame splat budget (144k scene → ≤24k selected in ~1.5 ms) |
| Scale-proven LOD pipeline | live | Measured at **2.05M splats**: selector build 138 ms, select median 216 µs / p99 1.1 ms across a 100-frame camera flight, zero budget violations (`scale_trial` bin, hard-asserted) |
| Spectral material compiler | live | Node-graph BSDF (Substrate-style layering, Fresnel, blackbody emitters, per-wavelength math) **compiling to naga-validated WGSL** — GPU output matches the CPU reference to 2.8e-6 per band on real hardware |
| Cluster acceleration (CLAS) | live | Deterministic spatial clustering + median-split BVH over splats (culling, LOD, GI subset queries) |
| Hierarchical LOD + crossfade | live | 4-level per-cluster LOD chains (full → billboard) with opacity crossfade bands |
| HLOD baker | library | K-means++ multi-level merged-splat representations for far hierarchies |
| World partition + streaming | library | 3D cell streaming, tile manager with active radius, level streaming, splat GPU buffer pooling |
| Disk tile streamer | live | Background-thread .vxm tile loading: priority queue, generation-based cancellation (teleports drop obsolete loads without I/O), byte-exact LRU residency cache with budget eviction |
| Spectral framebuffer | live | 16-channel spectral + depth/normals/motion/object-id/albedo G-buffer |
| Tone mapping + post | live | ACES/Reinhard/Filmic with exposure/white-point; bloom, vignette, fog, god rays, chromatic aberration |
| Temporal accumulation | library | Reprojected accumulation buffer with blend alpha (denoising support) |
| Bilateral spectral denoiser | library | Edge-aware spatial + spectral kernel denoise |
| Order-independent transparency | library | OIT accumulate + resolve passes |
| DLSS-style upscaling | library | Quality/Balanced/Perf/UltraPerf internal-resolution modes + frame generation scaffolding |
| Splat-VFX graph | live | Niagara-shaped typed node DAG where particles ARE spectral splats (blackbody fire, deterministic for rollback); unifies the 5 legacy particle modules |
| Particles (CPU/GPU/splat) | library | Emitters with spectral emission; Gaussian splats as particles; particle death drives audio synthesis |
| Hybrid mesh+splat compositing | live | One-pass depth-correct compositing of triangle meshes with splats: perspective-correct 1/z depth interpolation, Sutherland-Hodgman near+far clipping |
| Many-light sampler | live | ReSTIR-style weighted reservoir light selection (O(1) per shade point over arbitrary light counts), spatial-grid candidate culling, 16-band spectral path — unbiasedness proven to <1% vs brute force |
| Render graph | live | RDG-style pass DAG: declared reads/writes, topo scheduling, dead-pass culling, cycle detection, declared-access enforcement — drives the postprocess chain bit-identically to the legacy path |
| Spectral splat ray tracing | live | 3DGRT-style CPU reference: closed-form ray-vs-Gaussian peaks, CLAS-BVH traversal (footprint-padded, bit-identical to brute force), 16-band front-to-back compositing with hard budget, shadow-ray transmittance — cross-checked vs the EWA rasterizer to 3% |
| Cascaded shadow maps | live | Multi-cascade directional shadows + shadow atlas, SDF soft shadows, shadow catcher |
| Cinematic camera | library | Keyframed camera, depth-of-field with bokeh shapes, movie render to disk |
| Multi-viewport | library | Perspective/Top/Front/Right simultaneous editor views |
| Water + caustics | library | Flowing water surface with spectral wet blending and caustic modulation |
| Hair simulation | library | Strand curves with spectral melanin coloring, mass-spring dynamics → splats |
| Visual effects graph | library | Data-driven VFX definitions (declarative emitter/effect parameters) + editor UI |

## Lighting & Global Illumination

| Feature | Status | Description |
|---|---|---|
| Spectral GI (CPU) | live | Per-splat radiance cache: 1/d² emitter gather + sky-ambient seed, applied to splats every GI tick in both shipped binaries |
| **Spectral GI (GPU)** | live | `GpuGi` WGSL compute mirror — **bit-identical to the CPU path** (shared emitter bound + hour→zenith mapping by construction); selectable per-run via `OCHROMA_GI=gpu` or `EngineLoop::use_gpu_gi()`, permanent graceful CPU fallback |
| Spectral atmosphere | live | Rayleigh/Mie physically-based sky: blue zenith / red horizon emerge from wavelength, not textures |
| Day/night cycle | live | Continuous time-of-day drives sun, GI sky, and shadows (smoke-asserted ~3× noon/midnight luma ratio) |
| Sun model | live | Latitude + hour + day → sun direction (London default) |
| GI baker | library | Offline multi-bounce spectral irradiance bake (K-means clusters) + runtime cache application |
| Many-light sampling | live | MegaLights analogue shipped: see "Many-light sampler" under Rendering (ReSTIR-style reservoir selection, spectral path) |

## Scene & Asset Pipeline

| Feature | Status | Description |
|---|---|---|
| VXM native format | live | Zstd-compressed splat container, v1–v3 version-aware reader, material IDs, spectral-level tagging |
| PLY import/export | live | Standard 3DGS PLY (verified at 308k splats) |
| SPZ format | live | Niantic compressed splats (<35 % of PLY), hardened against hostile headers |
| KHR glTF splat interop | live | Export/import `KHR_gaussian_splatting` glTF; raw-JSON accessor validation (the typed crate's validator rejects the draft semantics); fuzz-hardened |
| **USD scene import** | live | `vox_usd` on the pure-Rust openusd-rs sibling: full pcp composition of `.usdc` stages → meshes (sampled to 2DGS), PointInstancer (→3DGS), lights, camera, materials→spectrum. CLI `usd-import` + content-browser integration. First-mover: no major engine reads USD scenes natively into splats |
| glTF mesh→splat conversion | live | Surfel sampling of triangle meshes to 2DGS with RGB→spectral uplift (CLI `gltf2splat`) |
| Spectral material capture | live | 3-photo reflectance estimation (daylight/tungsten/cool-LED) → per-band materials |
| Spectral upsampling | live | Smits RGB→16-band; spectral material library (11 materials) |
| Importance pruning | live | HGSC-style offline splat optimization with a render-diff perceptual guard (CLI `prune`: 50% size cut at 0.003 mean pixel diff) |
| Neural splat compression | library | MLP-based 2–10× splat compression tiers |
| COLMAP photogrammetry | library | Subprocess pipeline: photos → sparse reconstruction → splats |
| OSM import | library | OpenStreetMap roads/buildings ingestion |
| Procedural splat generation | live | Terrain/building/tree/city-block splatizers (drive the shipped demo scenes) |
| Save/scene serialization | live | Human-readable world saves (entities, lights, colliders, custom data round-trip) |
| Asset hot-reload | live | mtime+length watching for scripts/assets/maps |

## Editor

| Feature | Status | Description |
|---|---|---|
| Content browser | live | UE-style asset panel IN THE DOCK: real asset scan, working type filters + ranked search, kind-colored tiles with header-peek splat-count badges, double-click load with honest Output Log receipts |
| Node-graph PCG editor | live | Typed DAG (10 port types, type-checked connections, deterministic topo-sort) with 11 node kinds (Terrain/Biome/Moisture/Vegetation/Building/Plot/Splatize/…) |
| Live viewport graph execution | live | PCG-style: param edits re-cook ONLY the dirty subgraph (multi-edit safe, per-subgraph trailing-edge throttle) and update the 3D viewport without restart |
| Subgraphs / graph functions | live | Collapse any selection into a reusable, registry-searchable node — byte-identical evaluation proven before/after collapse and expand, nested with typed depth guard |
| Graph templates | live | Data-driven starter graphs instantiated through the node registry (cannot drift from real nodes) |
| Node registry + search insertion | live | UE-Blueprint-style typed fuzzy palette; wire-drag compatibility filtering; port metadata probed from real descriptors |
| Node preview thumbnails | live | Real data-derived mini-renders per output (heightfield maps, splat scatters, spectral bar charts), regenerated only when a node's cook actually ran |
| Comment boxes + wire inspection | live | UE-style group boxes that move members; per-wire value chips showing the actual data that flowed |
| Gizmo pipeline | live | Canonical drag→transform with mode honoring + snapping |
| Scene hierarchy / inspector / picking | live | Entity tree, property panels, ray-based splat picking, undo stack |
| Editor SOTA shell (Phase 1) | live | Tokenized theme (JSON-swappable), Phosphor icons, egui_dock drag-docking, plain-language chrome, headless `shell_snapshot` proof — bitmap font deleted from the editor |
| Real typography | live | parley shaping + swash rasterization (anti-aliased vector text, headless-capable); 5×7 bitmap font retired from HUD + editor |
| Vello GPU UI path | live | SpectralHUD renders through a real `vello::Renderer` (headless-provable pixel readback) in the default binary; opt-in `game-ui` feature; CPU fallback always present |
| Command palette + CommandRegistry | live | Single command surface: menus, fuzzy palette (Ctrl+P), shortcuts and AI intents all dispatch the same registry — duplicate ids replace, never shadow |
| NodeCanvas 2.0 | live | Bezier gradient wires (port-type colored endpoints, 32-segment AA), dot-grid zoom/pan/snap, minimap, category headers — shared `vox_ui` canvas for every graph tab |
| Live engine viewport in dock | live | Real `SoftwareRasteriser` frames presented as the docked Viewport texture (not a mock) |
| Graph bridge | live | Real `OchromaNodeGraph` → canvas projection: param edits re-cook through `live_cook`, cook errors surfaced in-panel |
| Ask Ochroma (AI intents) v2 | live | IntentBackend seam: deterministic parser default, opt-in LLM (env) emitting schema-validated strict JSON with parser fallback — unvalidated output can never touch the graph; receipts carry provenance; every AI action undoable |
| Ecosystem plugins (Crucible · Forge · FloraPrime) | live | UE-style host-plugin model: `PluginCtx` exposes only tokens/widgets/canvas (structurally enforced); three visual-editor plugins live in the dock |
| Behavior-tree editor, sequencer, anim editor, material editor | library | Authoring UIs present; not yet driving shipped binaries |

## Simulation & Gameplay

| Feature | Status | Description |
|---|---|---|
| Unified EngineLoop | live | One per-frame driver (scripts→physics→audio→GI→shadows) shared by the game and editor shells, with per-system opt-out masks |
| Rhai scripting + hot-reload | live | Live game scripts: edit the `.rhai` mid-run and behavior changes without restart; compile errors keep the last-good AST running + HUD banner (never crashes the game) |
| Character controller | live | Capsule KCC with gravity, slopes, step-over |
| Physics (Rapier3D) | live | Dynamic bodies, colliders, ECS sync; drop-box + impact events in the shipped game |
| Spectral physics | live | Per-band fracture thresholds, impact-driven splat fracture, wetness/drip spectral blending; resonance/damage models |
| Cloth / rope / vehicle | library | Mass-spring cloth, rope chains, wheel-suspension vehicles |
| Destruction | library | Fracture patterns, debris, constraint breaking, progressive destruction masks |
| City simulation | experimental | 30+ subsystems: zoning, roads, citizens, economy, traffic, weather, seasons, disasters, ecosystem |
| Large-world coordinates | library | Tile-based LWC streaming (1 km tiles) |
| Navmesh + A* | library | SDF-walkable-surface navmesh generation and pathfinding |
| Save/undo/autosave | live | Editor undo stack, periodic autosave, world persistence |

## Animation

| Feature | Status | Description |
|---|---|---|
| Blend trees + skeletal clips | live | Keyframed clips, looping, lerp/slerp blend trees on a humanoid skeleton with a splat-skinning bridge |
| Motion matching | live | PoseDatabase + nearest_continuing + inertial blending driving the walking_sim avatar from real locomotion state — smoke asserts clip selection, pose continuity, determinism |
| IK (FABRIK) | library | Forward-and-backward reaching chain solver |
| Morph targets, facial, GPU skinning | library | Blend shapes, face rigs, compute-shader skinning paths |

## Audio

| Feature | Status | Description |
|---|---|---|
| Spectral synthesis | live | Impact/collect sounds synthesized from material spectra — no WAV files; glass strikes are spectrally brighter than stone (test-asserted) |
| Spatial audio | live | 3D distance attenuation + panning; CPAL device backend (headless-safe) |
| Biome soundscapes | live | Ambient beds selected by geometric biome classification |
| SDF reverb | live | Room acoustics estimated from scene geometry (Sabine per band group); GI-derived reverb |
| HRTF, acoustic raytracer, adaptive music | library | Head-related transfer functions, early-reflection simulation, context-aware music transitions |

## Networking

| Feature | Status | Description |
|---|---|---|
| Rollback netcode | live | Predict/rollback/resimulate with 8-frame history — proven over a real QUIC (TLS 1.3) loopback session with convergence assertions |
| Two-process session harness | live | `net_session` host/client/selftest binary: real cross-process QUIC probes (self-spawn via `current_exe`, self-reported LISTENING, bounded host-kill death detection) |
| Transport tuning policy split | live | `TransportTuning::game()` (30 s idle / 5 s keep-alive) vs `test_harness()` (5 s / 1 s) — harness latency requirements no longer dictate engine-wide transport policy |
| Entity replication | library | Delta compression, snapshots, replication loop/packets |
| Lobby / world hosting / CRDT | experimental | Session scaffolding, conflict-free types |

## AI

| Feature | Status | Description |
|---|---|---|
| Spectral perception | live | NPCs sense the world through spectral signatures (the shipped NPC flees the player's fire-band radiance; guards could see fire through fog) |
| Behavior trees | library | Sequence/Selector/Repeater/Inverter evaluation + authoring editor |
| Text-to-city generation | experimental | Prompt → district layout → buildings/props/vegetation → splats (LLM-assisted, deterministic stub offline) |
| Spectral damage/fingerprints | library | Per-band damage attenuation; entity spectral identity |

## Platform, Tooling & Quality

| Feature | Status | Description |
|---|---|---|
| Headless smoke gates | live | Both binaries run `--smoke`: real sim frames + rendering with behavioral pixel/state assertions (the CI runtime gate) |
| CLI asset pipeline | live | `vox_tools`: gltf2splat, splats2gltf, gltf2splats, usd-import, turnaround capture, cross-platform build |
| WebGPU/WASM | experimental | Browser canvas + adapter bring-up (clear-color proof; splat pipeline pending) |
| CI | library | Three-sibling-repo checkout (ochroma+spectra+crucible; openusd-rs pending), default-feature builds, clippy `-D warnings` gate — awaiting repo-push + PAT to observe green |
| Profiling | library | Puffin frame profiling, render telemetry, benchmark harness |
| Localization | library | Locale bundles + i18n manager (not yet wired to UI) |
| Dev-profile debuginfo diet | live | line-tables-only workspace / no dep debuginfo: full cold build = 40 s / 5.7 GB target |

---

*The deeper paper trail: `docs/spec/unreal-gap-analysis.md` (honest per-system gaps),
`docs/superpowers/specs/` (design docs incl. atom-budget renderer + USD import),
`docs/superpowers/plans/` (the blitz roadmap with dated status notes).*
