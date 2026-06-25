# Design: Engine-Side Render Runtime — One Method, One Config (2026-06-25, v2)

**Status:** Draft (v2 — rewritten after adversarial review; v1 named the wrong boundary type)
**Scope:** Make `vox_render` the single engine-side render runtime and reduce `urban_horizon` to a `vox_scene::SceneDelta` emitter — one render method, one config — restructuring `spectra` / `vox_scene` / `vox_render` / `urban_horizon` as needed (we own all of them; no crate boundary is "external").
**Related:** [Scene Architecture: AURORA](./scene-architecture-aurora.md) · audit/migration map `scratchpad/migration_map.md` · v1 adversarial review (this session).

> **No artificial constraints.** We control every crate — `spectra-types`, `spectra-renderer`, `vox_scene`, `vox_render`, `urban_horizon`. "Out of scope: spectra internals", "forbidden dependency inversion", and "can't delete a symbol with N callers" are NOT real limits here. If the right structure needs a spectra-types change or a dep restructure or refactoring all 19 `near_realtime` callers, we do it. The only hard limits are language facts (Rust can't re-export inherent methods) and the ship gates (determinism + the 30fps floor).

---

## 1. Problem Statement

- The renderer is split: ~1.3K engine lines (`ResidentSceneRenderer`) vs **~10K game lines** (`urban_horizon/src/render_gpu/mod.rs` 6195 + `spectra_frame.rs` 3793) holding device context, GPU upload, the frame loop, present, image conversion, and generic geometry/SDF utilities — all game-agnostic, all in the game crate.
- **The clean game→render boundary already exists but is bypassed.** `vox_scene::SceneDelta` (enum `AddProto`/`EditProto`/`AddNode`/`RemoveNode`/`SetTransform`/`SetProto`, `vox_scene/src/lib.rs:190`) + `vox_render::scene_delta_adapter` (`RetainedRenderMirror`, `RetainedDeltaPlan`) is engine-side, game-agnostic, and deterministic. But structural scene changes route through the game's `build_instanced_scene` → full upload, **not** through this seam — so the boundary isn't the single contract it should be.
- **Multiple render methods:** live `render_camera`; the banned wgpu `HybridComposeGpu` held as a field on the live `SceneRenderer` (constructed at `play.rs:381` for default `MeshPreferred`); `#[cfg(feature="legacy-raster")]` `new_city_instanced`; dead `splat_rt_gpu`.
- **Four config sources:** (1) game `render.ron` → game `RenderConfig` (look/grade/sky) via `SPECTRA_*` env bridges; (2) `RenderSettings::for_tier` (cost knobs: spp/bounces/denoiser/spectral) — **hardcoded match arms in `spectra-types/src/render_settings.rs:934`**; (3) `RenderConfig::near_realtime/preview` base constructors (`spectra-renderer`, 19 callers); (4) game `FidelityTierSetting` enum (`menu/settings.rs`). No single source.
- v1 of this design named `vox_render::SceneDelta` (a `{layers_rebuilt,layers_reused}` **report**) as the boundary — wrong type. This v2 corrects it.

---

## 2. Done When

On the box (`tomespensin`), after migration, with the live config (no `OCHROMA_RES_IN` override):
```
ssh tomespensin '... <city render, performance tier, 96 lots>'
```
prints `internal 512x288 ... STEADY <≤33ms> (≥30 fps)` — city still renders, still clears the 30fps floor (no regression vs `d9f1a5b`).

Locally (engine compiled with the feature that actually includes this code):
1. `cargo build -p vox_render --features spectra-native` is green. (Default features are `["crucible"]` and `resident_renderer.rs` is `#![cfg(feature="spectra-native")]` — a default build compiles **none** of this, so every local clause uses `--features spectra-native`.)
2. `grep -rn "<moved render-runtime symbols>" urban_horizon/src` → **0** (they resolve via `use vox_render::...`).
3. `grep -rn "HybridComposeGpu\|splat_rt_gpu\|new_city_instanced\|hybrid_renderer_for" {ochroma,urban_horizon}/src` → **0** (physically deleted, not just unreachable).
4. `cargo test -p vox_render --features spectra-native scene_delta_replay_exact` passes (authored by the plan): same `vox_scene::SceneDelta` log applied twice → byte-identical `download_resident_instances()`.
5. Perturbing `render.ron`'s `[performance].max_bounces` and re-rendering changes `[shot-map] mean_luma` — and **no** code path (`for_tier`, `near_realtime`) overrides it. `for_tier` now reads `render.ron`.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Engine owns the render runtime | moved fns live under `vox_render/src/{frame,device,util}/`; game `render_gpu` has 0 of them (`grep`) | empty mod compiles |
| Game is a `SceneDelta` emitter | `cargo test -p urban_horizon scene_proxy_emits_deltas --features spectra-native -- --nocapture` builds a 3-building `GameState`, asserts the emitted `Vec<vox_scene::SceneDelta>` contains 3 `AddNode` + terrain, and `RetainedRenderMirror::plan_deltas` yields `structural_rebuild==true` then a stable mapping | `assert!(deltas.len() > 0)` |
| Determinism boundary | `scene_delta_replay_exact` (Done-When 4) — byte-identical instance buffer from a replayed delta log | `assert!(plan.refits.len() > 0)` |
| One render method | box city render writes `WROTE` + `STEADY ≥30 fps` via the single `render_camera`; no other path constructs a renderer | build passes |
| One config | `for_tier` reads `render.ron` tier blocks; `render_config_perturbation_coverage` (exists, `spectra_frame.rs:3725`, box-only) shows every tier knob changes the frame | `assert_eq!(cfg.max_bounces, 3)` vs a preset |

---

## 4. Architecture

### 4.1 The boundary — already real, just under-used

UE5.8's structural *lesson* (renderer is a self-contained engine module; the project feeds it), our *primitives* (no render thread, no RHI command list, no `FPrimitiveSceneProxy`, no `ENQUEUE_RENDER_COMMAND`, no RDG). The boundary already exists:

```
 game: GameState ──► Vec<vox_scene::SceneDelta>      [the structural delta LOG — game-agnostic, deterministic source]
        (scene_proxy factory)        │
        ─────────────────────────────┼─────────  determinism boundary = the engine render API
                                      ▼
 engine vox_render:
   scene_delta_adapter::RetainedRenderMirror.plan_deltas(&deltas)  ──► RetainedDeltaPlan
        (id-sorted BTreeMap, latest-wins, instance-sorted — PROVEN deterministic)
                                      │  structural → full upload ; transform-only → TLAS refit
                                      ▼
   ResidentSceneRenderer: GPU-resident scene · render_camera → trace → denoise → DLSS → present
                                      [FREE to be non-deterministic for speed — pixels/BVH are NOT the artifact]
                                      ▼
 spectra backend (≈RHI): scene + camera + tier → frame   [PT-native; no raster command layer]
```

**Reproducible artifact = the `vox_scene::SceneDelta` log** (the game's deterministic output), not pixels/BVH. Replay = re-apply the same log → byte-identical `download_resident_instances()`. This is *already* what `scene_delta_adapter` guarantees (`transform_refits_are_latest_wins_and_instance_sorted` test). The refactor's job is to make this seam the **only** game→render path: today structural edits bypass it via `build_instanced_scene`→full upload; the target routes structural edits through `vox_scene::SceneDelta::{AddNode,RemoveNode,...}` too (the GPU mirror owning proto/node allocation is the AURORA follow-on; until then structural deltas keep the documented full-rebuild fallback).

### 4.2 Engine module surface — 4 dirs, not 9

The audit's per-symbol map (`migration_map.md` §1) is correct about *what* is game-agnostic; v1's 9-module split was UE sprawl. Land it in **four** top-level dirs (`scene_renderer/ frame/ device/ util/`, ~10 files) plus the existing files — materially leaner than 9, stated honestly:

```
vox_render/src/
  scene_renderer/      (mod = ResidentSceneRenderer; keep resident_renderer.rs, scene_delta_adapter.rs)
                       + config.rs (the SPECTRA_*/OCHROMA_* RenderConfig assembly)
                       + glass.rs  (glass_floor_for_tier, GLASS_MIN_BOUNCES, scene_has_glass — engine cost policy)
  frame/               host_frame.rs (frame_with_scene_deltas), gpu_frame.rs (frame_gpu_with_scene_deltas, rr_guides),
                       present.rs (set_render_target, rr_guide_ptrs, rr_jitter, internal_resolution)
  device/              GPU_CTX, gpu_context, create/install_headless_gpu_context  (from render_gpu 4545–4620)
  util/                geometry (push_oriented_box_mesh, push_pitched_roof_mesh, point_in_outline, triangle/subdivide,
                                 include_*_bounds), camera math (screen_to_ground, world_to_screen, orbit_camera,
                                 unproject, scene_camera_forward), CameraAtomFilter, image (hybrid_image_to_srgb*,
                                 srgb_to_linear, box_project_uv, color math, brighten_splats)
```

The `set_*` upload setters **stay as inherent methods on `ResidentSceneRenderer`** (Rust can't re-export inherent methods — v1's `upload/` facade was a non-compiling phantom). The SDF *render* helpers (`resident_sdf_debug_splats`, `render_sdf_debug_raymarch`) move to `util/`; the SDF **physics** bridge (`resident_sdf_physics_*`, `sync_resident_sdf_physics_world(&mut bevy_ecs::World)`) is **out of scope** — it's physics, not render runtime, tracked separately.

### 4.3 Game — the `SceneDelta` feed

The game keeps only `GameState → render-scene` translation (reads `GameState`/`AssetCatalog`/`MapTerrain`/sim-tick): `city_scene_inner`, `build_instanced_scene`, terrain/water meshing, placers, CPU PBR + texture cache, `rig_for_tick`, agent hydration, the sim-driven light/glass/weathering policy. Its output is the `vox_scene::SceneDelta` log + the per-frame camera. **`CityRenderScene` → `RenderScene`** (move the type to `vox_scene` or a shared crate) is a **prerequisite** of moving any code that consumes it (it can't land in `vox_render` while named/owned by the game) — done first, not deferred.

### 4.4 One config — the game loads `render.ron`; `spectra-types` stays IO-free

`spectra-types` has **no filesystem deps today** and its own `fidelity_tier_cost_strictly_monotonic` test asserts the hardcoded tier arms — so we must NOT push `render.ron` reading into it (that breaks the in-crate test and every headless still/cine caller that has no `render.ron`). Instead:

- The **game** owns `render.ron` and loads a `TierTable` from **new** `[performance]`/`[balanced]`/`[beauty]` blocks. This is a **new schema**: `render.ron` is *flat* today (a single `resolution_in`/`spp:1`/`spp_photo:64`, `assets/config/render.ron:70,80`) — the migration adds the tier blocks and reconciles them with the existing realtime-vs-photo split.
- It passes the table into a pure `RenderSettings::for_tier_from_ron(tier, &TierTable)`. The existing hardcoded `for_tier(tier)` arms become the **absent-table fallback** — headless/still/cine/in-crate-test callers keep working with no `render.ron`. The 19 `near_realtime`/`preview` callers move onto this path.
- The audit floors (ReSTIR OFF, NRC off, SVGF pinned — `render_settings.rs:941`) are **clamps applied *after* the table read**, not config values — so the perturbation-coverage law isn't fooled by a dead knob, and "one config" stays honest (the table is the only *source*; the clamps are correctness guards).

Lever 2 (bounces) becomes a one-line `render.ron` edit — still floored for glass by `glass_floor_for_tier` (`resident_renderer.rs:1029`), which must be documented.

---

## 5. Data Models

```rust
// vox_scene — the boundary contract (EXISTS; this is the real "delta log").
pub enum SceneDelta {                         // vox_scene/src/lib.rs:190
    AddProto { proto: ProtoId, mesh: ProtoMesh },
    EditProto { proto: ProtoId, mesh: ProtoMesh },
    AddNode { id: NodeId, proto: ProtoId, transform: SceneTransform, material_base: u32 },
    RemoveNode { id: NodeId },
    SetProto { id: NodeId, proto: ProtoId },
    SetTransform { id: NodeId, transform: SceneTransform },
}

// vox_render::scene_delta_adapter — the engine-side deterministic planner (EXISTS).
pub struct RetainedRenderMirror { node_to_instance: BTreeMap<NodeId, usize>, instance_to_node: Vec<NodeId> }
impl RetainedRenderMirror {
    pub fn plan_deltas(&self, deltas: &[SceneDelta]) -> Result<RetainedDeltaPlan, RetainedDeltaError>;
    // id-sorted, latest-wins, instance-sorted refits — the replay-exact spine.
}

// RENAME to end the collision: the rebuild/reuse report v1 wrongly called "SceneDelta".
pub struct SceneSyncReport { layers_rebuilt: u32, layers_reused: u32 }  // was vox_render::SceneDelta
```

### 6. API

Exact signatures are read from source and pinned in the **plan** (per the template's IMPORTANT-NOTES rule — NOT invented here; v2's first draft invented `apply_scene_deltas`/`RenderError`/`InstanceRecord`, which do not exist). This section states the **real call path**:

- **The boundary mirror is GAME-side today.** `LiveFrameSource` (game, `spectra_frame.rs:2050`) owns the `RetainedRenderMirror` field and calls `apply_retained_scene_deltas` / `drain_scene_deltas` (`spectra_frame.rs:2321,2376`), which run `RetainedRenderMirror::plan_deltas(&[vox_scene::SceneDelta]) -> RetainedDeltaPlan` (errors on structural; transform-only → refit via `queue_resident_refits` → `update_instance_transform`).
- **Real signatures as they stand** (the plan moves, not rewrites, these): `ResidentSceneRenderer::new_with_tier(w,h, rig: LightRig, tier: FidelityTier, initial: SceneState) -> Result<Self, String>`; `download_resident_instances(&self) -> Result<Vec<u32>, String>` (the determinism witness — `Vec<u32>` is byte-comparable).
- **Explicit migration task:** move the `RetainedRenderMirror` + the `apply_retained_scene_deltas`/`drain_scene_deltas` feed **out of game `LiveFrameSource` into the engine** (`scene_renderer`), so the boundary `vox_scene::SceneDelta → mirror → refit` is engine-owned and the game only emits `Vec<vox_scene::SceneDelta>`. This relocation IS a core step, not a given — the current split is exactly why the renderer is "in the game."

No render thread; single GPU-context owner. Tier knobs resolve via `for_tier_from_ron` (§4.4).

### 7. Wiring

| Component | Called from | File |
|---|---|---|
| `vox_render::device::gpu_context` | engine runtime init | `vox_render/src/device/mod.rs` |
| `RetainedRenderMirror::plan_deltas` | `render_camera` delta drain | `vox_render/src/scene_renderer/` (already wired via scene_delta_adapter) |
| `RenderSettings::for_tier` (reads render.ron) | `ResidentSceneRenderer::new_with_tier` | `spectra-types/src/render_settings.rs` |
| game `scene_proxy::city_scene_inner` → `Vec<SceneDelta>` | `produce_live_frame` | `urban_horizon/src/render_gpu/` |

---

## 8. Open Questions

- [x] What is the boundary type? → `vox_scene::SceneDelta` enum + `scene_delta_adapter` (NOT the report struct). Resolved.
- [x] Is spectra editable for one-config? → Yes, we own it. Resolved.
- [ ] Do structural edits get routed through `vox_scene::SceneDelta::{AddNode,RemoveNode}` now, or keep the `build_instanced_scene` full-upload fallback until the GPU mirror owns proto/node allocation (AURORA follow-on)? Proposed: keep the fallback this pass; route transform deltas through the seam (already works); land structural-via-delta as a fast-follow.
- [ ] Does `internal_resolution` (`spectra_frame.rs:287`) read tier/FSR state that ties it to game config? Verify before moving to `frame/present`.

## 9. Out of Scope

- Spectra **render-algorithm** changes (kernels, denoiser, DLSS math) — this is relocation + config unification, not new render tech. (Editing `spectra-types` config plumbing IS in scope — that's not an algorithm change.)
- The SDF **physics** bridge (`sync_resident_sdf_physics_world`, `resident_sdf_physics_*`) — physics, separate target.
- The sim/economy/wake-queue port (separate AURORA work).
- Routing *structural* deltas through the GPU-resident mirror (fast-follow; see §8).

## 10. Related

- Depends on: [AURORA](./scene-architecture-aurora.md) (GPU-resident scene; `scene_delta_adapter` as canonical seam; determinism decoupled).
- Required before: the render-runtime *plan* (`docs/templates/plan.md`) driving the corrected, dep-aware migration sequence (CityRenderScene→RenderScene first; SceneRenderer teardown scheduled; spectra-native builds; real witnesses authored).
- Related: lever-1 commit `d9f1a5b` (render.ron 512 internal → city 33fps); one-config folds lever 2 (bounces) into render.ron.
