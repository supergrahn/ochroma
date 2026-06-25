# Render Runtime Extraction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the game-side render runtime (~10K lines in `urban_horizon`) into `vox_render` so the game is a pure `vox_scene::SceneDelta` emitter, `RetainedRenderMirror` is engine-owned, and all tier knobs resolve from a single `render.ron` `[fidelity]` block via `RenderSettings::for_tier_from_ron`.

**Done When:**
1. On `tomespensin` (box), after migration: `ssh tomespensin '... <city render, performance tier, 96 lots>'` prints `internal 512x288 ... STEADY <≤33ms> (≥30 fps)` — no regression vs `d9f1a5b`.
2. `cargo build -p vox_render --features spectra-native` is green locally.
3. `grep -rn "HybridComposeGpu\|splat_rt_gpu\|new_city_instanced\|hybrid_renderer_for" ochroma/crates/ Ochroma/projects/urban_horizon/src/` → **0 hits** (physically deleted, not just unreachable).
4. `cargo test -p vox_render --features spectra-native scene_delta_replay_exact -- --nocapture` passes and prints `instance_buf[0]=<nonzero u32> instance_buf[1]=<nonzero u32> replay_matches=true`.
5. `cargo test -p urban_horizon --features spectra scene_proxy_emits_deltas -- --nocapture` passes and prints `add_nodes=3 has_terrain_proto=true structural_rebuild=true`.
6. Perturbing `render.ron`'s `[performance].max_bounces` from 2 to 4 and re-running the box render changes `[shot-map] mean_luma` in the output — no hardcoded `for_tier` arm overrides it.

**Architecture:** `vox_render` becomes the single engine-side render runtime: `RetainedRenderMirror` and `drain_scene_deltas` move from game `LiveFrameSource` into `ResidentSceneRenderer`; `RenderScene` (no game deps) joins `vox_render::hybrid_compose`; pure-engine geometry/camera utilities land in `vox_render::util`; `SceneDelta` (the `{layers_rebuilt,layers_reused}` report in `vox_render::resident_renderer`) is renamed `SceneSyncReport` to end the collision with `vox_scene::SceneDelta`; `spectra-types` gains `TierTable`/`TierEntry` + `for_tier_from_ron` (the game owns `render.ron` I/O, `spectra-types` stays IO-free). Tasks are ordered compile-safe: dead-code deletion first, then type extraction, then engine lift, then config unification, then rename.

**Design Document:** `docs/superpowers/specs/2026-06-25-render-runtime-extraction-design.md`

**Tech Stack:** Rust 1.87, spectra-types (spectra/rust/spectra-types), spectra-renderer (spectra/rust/spectra-renderer), vox_render (`ochroma/crates/vox_render`), vox_scene (`ochroma/crates/vox_scene`), urban_horizon (`Ochroma/projects/urban_horizon`). Default features = `["crucible"]`; all resident/spectra code is under `--features spectra-native` (vox_render) or `--features spectra` (urban_horizon).

**Build:**
```bash
# Local compile-check (no CUDA, no Slang needed):
cargo build -p vox_render --features spectra-native
cargo build -p urban_horizon --features spectra

# Full tests:
cargo test -p vox_render --features spectra-native
cargo test -p urban_horizon --features spectra

# Box build + render witness (run AFTER local green):
# rsync + build on tomespensin per the render-spectra-cuda skill
```

---

## IMPORTANT NOTES

All signatures below are the **real** ones as they exist in source. Do not alter names, parameters, or return types — any deviation produces a compile error.

### Boundary types
- `vox_scene::SceneDelta` (`vox_scene/src/lib.rs:190`) — the structural delta LOG (enum: `AddProto`/`EditProto`/`AddNode`/`RemoveNode`/`SetProto`/`SetTransform`). This is the game→engine seam. **Not the report struct.**
- `vox_render::resident_renderer::SceneDelta` (`resident_renderer.rs:53`) — the **report** struct `{ layers_rebuilt: u32, layers_reused: u32 }` with accessors `.layers_rebuilt() -> u32` and `.layers_reused() -> u32`. **Renamed to `SceneSyncReport` in Task 7.** Both names collide — the rename is mandatory for clarity.

### ResidentSceneRenderer (vox_render, `--features spectra-native`)
```rust
// resident_renderer.rs:71 — current struct (Task 3 adds retained_mirror field)
pub struct ResidentSceneRenderer {
    renderer: Renderer<ResidentBackend>,
    width: u32,
    height: u32,
    rig: LightRig,
    delta_ring: SceneDeltaRing,
    // Task 3 adds: retained_mirror: RetainedRenderMirror,
}

// resident_renderer.rs:126
pub fn new_with_tier(
    width: u32,
    height: u32,
    rig: LightRig,
    tier: FidelityTier,
    initial: SceneState,
) -> Result<Self, String>

// resident_renderer.rs:454
pub fn set_scene(&mut self, scene: SceneState) -> Result<SceneDelta, String>
// NOTE: returns the REPORT struct (renamed SceneSyncReport in Task 7)

// resident_renderer.rs:786
pub fn render_camera(&mut self, view: [f32; 16], proj: [f32; 16]) -> Result<FrameOutput, String>

// resident_renderer.rs:786
pub fn update_instance_transform(&mut self, instance_index: usize, transform: [[f32; 4]; 3])

// resident_renderer.rs:906
pub fn download_resident_instances(&self) -> Result<Vec<u32>, String>
// Returns byte-comparable Vec<u32> — the determinism witness.
```

### RetainedRenderMirror (vox_render::scene_delta_adapter)
```rust
// scene_delta_adapter.rs:73
pub struct RetainedRenderMirror {
    node_to_instance: BTreeMap<NodeId, usize>,  // private fields
    instance_to_node: Vec<NodeId>,
}

// scene_delta_adapter.rs:79
pub fn new() -> Self

// scene_delta_adapter.rs:83
pub fn replace_instances_in_node_order<I>(&mut self, nodes: I) -> Result<(), RetainedDeltaError>
where I: IntoIterator<Item = NodeId>

// scene_delta_adapter.rs:115
pub fn instance_index(&self, node: NodeId) -> Option<usize>

// scene_delta_adapter.rs:119
pub fn node_for_instance(&self, instance_index: usize) -> Option<NodeId>

// scene_delta_adapter.rs:107
pub fn len(&self) -> usize

// scene_delta_adapter.rs:123
pub fn plan_deltas(&self, deltas: &[GraphSceneDelta]) -> Result<RetainedDeltaPlan, RetainedDeltaError>
// GraphSceneDelta = vox_scene::SceneDelta (aliased via `use`)
```

### RetainedDeltaPlan (vox_render::scene_delta_adapter)
```rust
// scene_delta_adapter.rs:23
pub struct RetainedDeltaPlan {
    pub structural_rebuild: bool,
    pub refits: Vec<(usize, [[f32; 4]; 3])>,
    pub stats: RetainedDeltaStats,
}

// scene_delta_adapter.rs:30
pub fn requires_scene_rebuild(&self) -> bool

// scene_delta_adapter.rs:38 — cfg(feature = "spectra-native")
pub fn queue_resident_refits(
    &self,
    renderer: &mut crate::resident_renderer::ResidentSceneRenderer,
) -> Result<(), RetainedDeltaError>
```

### LiveFrameSource (urban_horizon, `--features spectra`)
```rust
// spectra_frame.rs:2014
pub struct LiveFrameSource {
    renderer: ResidentSceneRenderer,   // private field — access via self.renderer.*
    retained_mirror: RetainedRenderMirror,  // Task 4 removes this field
    // ... (other fields unchanged)
}

// spectra_frame.rs:2321 (deleted in Task 4 after engine version exists)
fn apply_retained_scene_deltas(&mut self, deltas: &[vox_scene::SceneDelta]) -> Result<RetainedDeltaPlan, String>

// spectra_frame.rs:2376 (becomes thin forwarder in Task 4)
pub fn drain_scene_deltas(&mut self, deltas: &mut Vec<vox_scene::SceneDelta>) -> Result<RetainedDeltaPlan, String>

// spectra_frame.rs:2388
pub fn frame_with_scene_deltas(
    &mut self,
    deltas: &mut Vec<vox_scene::SceneDelta>,
    camera: &RenderCamera,
) -> Result<((Vec<u8>, u32, u32), SceneDelta, RetainedDeltaPlan), String>
// NOTE: SceneDelta here = the REPORT struct (renamed SceneSyncReport in Task 7)

// spectra_frame.rs:2649
pub fn frame_gpu_with_scene_deltas(
    &mut self,
    deltas: &mut Vec<vox_scene::SceneDelta>,
    camera: &RenderCamera,
    color_ptr: u64,
) -> Result<(u64, u32, u32, SceneDelta, RetainedDeltaPlan), String>
```

### InstancedSceneBuild tuple alias (spectra_frame.rs:1011)
```rust
type InstancedSceneBuild = (
    SceneState,
    u64,
    usize,
    Vec<crate::cim::visible::VisibleCim>,
    CityAtlas,
    RetainedRenderMirror,  // element 5 — removed in Task 4
    SprayUpload,
);
```

### RenderSettings (spectra-types, always compilable — no feature gate)
```rust
// spectra-types/src/render_settings.rs:934 — hardcoded fallback, UNCHANGED
pub fn for_tier(tier: FidelityTier) -> Self

// Task 5 adds:
pub fn for_tier_from_ron(tier: FidelityTier, table: &TierTable) -> Self
// Calls for_tier(tier) as baseline, overlays non-None TierEntry fields,
// then applies audit clamps (ReSTIR OFF, NRC off, SVGF on realtime tiers).
// spectra-types has NO fs deps — the game layer owns render.ron I/O.
```

### New engine methods added in Task 3 (on ResidentSceneRenderer)
```rust
// cfg(feature = "spectra-native") — must carry the same gate as queue_resident_refits
pub fn drain_scene_deltas(
    &mut self,
    deltas: &mut Vec<vox_scene::SceneDelta>,
) -> Result<RetainedDeltaPlan, String>

pub fn reset_retained_mirror<I>(&mut self, nodes: I) -> Result<(), RetainedDeltaError>
where I: IntoIterator<Item = vox_scene::NodeId>

// Thin accessors forwarding to self.retained_mirror:
pub fn retained_instance_index(&self, node: vox_scene::NodeId) -> Option<usize>
pub fn retained_node_for_instance(&self, instance_index: usize) -> Option<vox_scene::NodeId>
pub fn retained_node_count(&self) -> usize
```

### CityRenderScene (game) and RenderScene (engine)
```rust
// urban_horizon/src/render_gpu/mod.rs:352 — stays in the game
pub struct CityRenderScene {
    pub splats: Vec<GaussianSplat>,
    pub meshes: Vec<HybridMesh>,
    pub sdfs: Vec<ResidentSdfVolume>,         // game type, cannot move to engine
    pub fallback_splats: Vec<GaussianSplat>,
}

// Task 2 adds to CityRenderScene impl:
pub fn to_render_scene(&self) -> vox_render::RenderScene

// Task 2 adds to vox_render (hybrid_compose.rs):
pub struct RenderScene {
    pub splats: Vec<vox_core::types::GaussianSplat>,
    pub meshes: Vec<HybridMesh>,
    pub sdf_aabbs: Vec<([f32; 3], [f32; 3])>,  // precomputed from sdfs.world_aabb()
    pub fallback_splats: Vec<vox_core::types::GaussianSplat>,
}
```

### Hard constraints
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.
- The `legcy-raster` feature is OFF on all live paths; its gated code is deleted in Task 0.
- `--features spectra-native` is required for any `vox_render` resident code to compile; default build (`["crucible"]`) compiles **none** of this.
- `spectra-types` must never gain an `std::fs` dep — `TierTable` is plain data, the game owns file I/O.
- The 30fps floor (`d9f1a5b`: city 512-internal → ≥30fps on tomespensin) is a SHIP GATE — no regression.
- `for_tier` (hardcoded arms) is **not removed** — it remains the absent-table fallback used by headless tests and `fidelity_tier_cost_strictly_monotonic`.
- SceneRenderer teardown (Task 0) must complete before the leaf-utils move (Task 2) to avoid dangling imports.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `ochroma/crates/vox_render/src/resident_renderer.rs` | Add `retained_mirror` field + `drain_scene_deltas`, `reset_retained_mirror`, accessor methods; `SceneDelta` → `SceneSyncReport` rename |
| Modify | `ochroma/crates/vox_render/src/scene_delta_adapter.rs` | No structural change — `RetainedRenderMirror` already lives here; stays |
| Modify | `ochroma/crates/vox_render/src/hybrid_compose.rs` | Add `RenderScene` struct after `HybridMesh` impl block |
| Create | `ochroma/crates/vox_render/src/util.rs` | Pure-engine leaf utilities: screen/camera math, geometry builders, color helpers |
| Modify | `ochroma/crates/vox_render/src/lib.rs` | `pub mod util;` + `pub use hybrid_compose::RenderScene;` exports |
| Modify | `ochroma/crates/vox_render/tests/resident_renderer_test.rs` | Add `scene_delta_replay_exact` test |
| Modify | `spectra/rust/spectra-types/src/render_settings.rs` | Add `TierEntry`, `TierTable` structs + `for_tier_from_ron` method + `for_tier_from_ron_roundtrips_default_table` test |
| Modify | `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs` | Delete `SceneRenderer`, dead raster helpers, hybrid helpers; add `CityRenderScene::to_render_scene()`; add `pub use vox_render::util::{screen_to_ground, world_to_screen, SceneBounds}`; all `vox_render::resident_renderer::SceneDelta` references → `SceneSyncReport` |
| Modify | `Ochroma/projects/urban_horizon/src/spectra_frame.rs` | Remove `retained_mirror` field + game-side helpers; thin `drain_scene_deltas` forwarder; `SceneDelta` import from `resident_renderer` → `SceneSyncReport`; update `InstancedSceneBuild` tuple; `frame_with_scene_deltas` / `frame_gpu_with_scene_deltas` call sites |
| Modify | `Ochroma/projects/urban_horizon/src/bin/play.rs` | Remove `SceneRenderer`, `GameView::scene` field → `viewport: (u32, u32)`; remove `new_renderer_for_mode`; update snapshot call |
| Modify | `Ochroma/projects/urban_horizon/assets/config/render.ron` | Add `[fidelity]` block with `performance`/`balanced`/`beauty` TierEntry rows |
| Modify | `Ochroma/projects/urban_horizon/src/render_config.rs` | Add `pub fidelity: TierTable` field to game-layer config struct with `#[serde(default)]`; pass table to `new_with_tier` call sites |
| Test   | `ochroma/crates/vox_render/tests/resident_renderer_test.rs` | `scene_delta_replay_exact` — byte-identical `download_resident_instances()` from replayed delta log |
| Test   | `Ochroma/projects/urban_horizon/src/` (inline test) | `scene_proxy_emits_deltas` — 3-building GameState emits 3 AddNode + terrain proto, plan yields structural_rebuild=true |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Dead render paths deleted | `grep` for `HybridComposeGpu\|new_city_instanced\|hybrid_renderer_for` → 0 hits in `ochroma/crates/` and `urban_horizon/src/` | build passes |
| `RenderScene` lives in engine | `grep -rn "struct RenderScene" ochroma/crates/vox_render/src/` → hits `hybrid_compose.rs`; `grep -rn "struct RenderScene" Ochroma/projects/urban_horizon/src/` → 0 hits | type exists somewhere |
| `RetainedRenderMirror` engine-owned | `cargo test -p vox_render --features spectra-native scene_delta_replay_exact` prints `replay_matches=true` with nonzero instance values | `assert!(plan.refits.len() > 0)` |
| Game is a SceneDelta emitter | `cargo test -p urban_horizon --features spectra scene_proxy_emits_deltas -- --nocapture` prints `add_nodes=3 has_terrain_proto=true structural_rebuild=true` | `assert!(deltas.len() > 0)` |
| One config (render.ron drives tiers) | Box render with `[performance].max_bounces=4` in render.ron shows higher `mean_luma` than `max_bounces=2` | `assert_eq!(cfg.max_bounces, 3)` |
| 30fps floor holds | Box city render prints `STEADY ≤33ms` (≥30fps) | build passes |
| `SceneSyncReport` rename complete | `grep -rn "vox_render::resident_renderer::SceneDelta\|use.*resident_renderer.*SceneDelta[^P]" Ochroma/projects/urban_horizon/src/` → 0 hits | compiles |

---

## Task 0: Delete dead raster paths and hollow out SceneRenderer

**Files:**
- Modify: `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs`
- Modify: `Ochroma/projects/urban_horizon/src/bin/play.rs`

**Acceptance:** `cargo build -p urban_horizon --features spectra 2>&1 | grep -E "^error" | wc -l` → `0`. Then `grep -rn "HybridComposeGpu\|new_city_instanced\|hybrid_renderer_for\|hybrid_image_to_srgb" Ochroma/projects/urban_horizon/src/ ochroma/crates/` → 0 hits.

**Wiring requirement:** All deletions happen in the same step as the `GameView` field migration. No intermediate state that compiles with a dead `SceneRenderer` holding live fields.

- [ ] **Step 1: Write the failing grep (establishes current state)**

```bash
grep -rn "HybridComposeGpu\|new_city_instanced\|hybrid_renderer_for\|hybrid_image_to_srgb" \
    Ochroma/projects/urban_horizon/src/ ochroma/crates/ | wc -l
```

Expected: nonzero count (these symbols currently exist).

- [ ] **Step 2: Delete private helper fns and the `HybridComposeGpu` import**

In `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs`:

2a. Delete `use vox_render::gpu::hybrid_compose_gpu::{HybridComposeGpu, HybridGpuImage};` (line 24 approx).
2b. Delete `fn hybrid_renderer_for(…) -> Option<HybridComposeGpu>` (lines 4969–4984 approx) — already a `None` stub per THE LAW comment.
2c. Delete `fn hybrid_image_to_srgb(…)` (lines 4986–5001 approx).
2d. Delete `fn hybrid_image_to_srgb_on_background(…)` (lines 5003–5026 approx).
2e. Delete `pub fn render_asset_inspection(&mut self, camera: &RenderCamera) -> Result<Vec<[u8; 4]>, String>` (lines 4932–4966 approx) — the last caller of `hybrid_renderer_for` / `hybrid_image_to_srgb_on_background`.
2f. Delete the duplicate `pub fn resident_sdf_registry(&self) -> …` method (lines 4885–4887 approx) — the free fn at line 567 is the live path.

- [ ] **Step 3: Strip SceneRenderer struct to `{width, height}` and rewrite `new_city`**

In `render_gpu/mod.rs`:

3a. Remove fields `ctx`, `hybrid`, `hybrid_splats`, `meshes`, `sdfs` from `SceneRenderer` (lines 4718–4727 approx). These are only consumed by paths just deleted.
3b. Rewrite `new_city`:
```rust
pub fn new_city(_scene: &CityRenderScene, width: u32, height: u32) -> Result<Self, String> {
    Ok(Self { width, height })
}
```
3c. Simplify `set_city_scene` to update only `width`/`height` (the fields it updates that still exist):
```rust
pub fn set_city_scene(&mut self, _scene: &CityRenderScene, width: u32, height: u32) {
    self.width = width;
    self.height = height;
}
```
3d. Delete `set_scene`, `new`, `resident_sdf_count` — grep confirms zero non-test callers outside dead paths.
3e. Delete all `#[cfg(feature = "legacy-raster")]` gated methods and structs in `render_gpu/mod.rs`: `InstancedState`, `InstancedFrameStats`, `new_city_instanced`, `constructions()`, `resize_instanced()`, `set_instanced_residual()`, `sync_world()`.

- [ ] **Step 4: Migrate `GameView::scene` → `GameView::viewport` in play.rs**

In `Ochroma/projects/urban_horizon/src/bin/play.rs`:

4a. Replace `scene: SceneRenderer` field on `GameView` (line 58) with `viewport: (u32, u32)`.
4b. Remove `new_renderer_for_mode` helper fn (lines 373–385 approx) — it has no body after SceneRenderer is hollow.
4c. Change line 2669 from `let (w, h) = self.scene.size();` to `let (w, h) = self.viewport;`.
4d. Update `GameView` initialisation at play.rs:518 and play.rs:1513 to set `viewport: (w, h)` instead of constructing `scene: ...`.
4e. Remove `use urban_horizon::render_gpu::{self, SceneRenderer};` — replace with `use urban_horizon::render_gpu;` (the free fns are still needed).
4f. Delete `CityRenderMode` enum, `render_mode_for()` fn, and the `render_mode`/`render_mode_override` fields — without `legacy-raster` the enum has one variant and the fn is a pass-through.
4g. Delete all remaining `#[cfg(feature = "legacy-raster")]` blocks in `play.rs`: `CityRenderMode::InstancedAtoms` arm, `m4_*` helpers, `last_m4_stats`, `last_m4_log`, `residual_rebuilds`, `rebuild_instanced_residual_world()`.

- [ ] **Step 5: Run — verify zero errors and zero dead-path hits**

```bash
cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l
```
Expected: `0`

```bash
grep -rn "HybridComposeGpu\|new_city_instanced\|hybrid_renderer_for\|hybrid_image_to_srgb" \
    Ochroma/projects/urban_horizon/src/ ochroma/crates/
```
Expected: 0 lines of output.

- [ ] **Step 6: Commit**

```bash
git add Ochroma/projects/urban_horizon/src/render_gpu/mod.rs \
        Ochroma/projects/urban_horizon/src/bin/play.rs
git commit -m "refactor(urban_horizon): delete dead raster paths + hollow SceneRenderer into (w,h) shell"
```

---

## Task 1: Add `RenderScene` to `vox_render::hybrid_compose` (no game deps)

This must precede the leaf-util move (Task 2) because `include_mesh_bounds` in `util` will reference `RenderScene` once `SceneBounds` moves there.

**Files:**
- Modify: `ochroma/crates/vox_render/src/hybrid_compose.rs`
- Modify: `ochroma/crates/vox_render/src/lib.rs`
- Modify: `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs`

**Acceptance:** `cargo build -p vox_render --features spectra-native` is green. `grep -n "struct RenderScene" ochroma/crates/vox_render/src/hybrid_compose.rs` → hit at the new line. `grep -rn "fn to_render_scene" Ochroma/projects/urban_horizon/src/render_gpu/mod.rs` → 1 hit.

**Wiring requirement:** `CityRenderScene::to_render_scene()` must be fully implemented (all four fields populated from `self`) and callable. `todo!()` = task failure.

- [ ] **Step 1: Write the failing test**

```rust
// In ochroma/crates/vox_render/tests/resident_renderer_test.rs, add:
#[cfg(feature = "spectra-native")]
#[test]
fn render_scene_from_default_has_empty_fields() {
    use vox_render::RenderScene;
    let scene = RenderScene::default();
    assert_eq!(scene.splats.len(), 0);
    assert_eq!(scene.meshes.len(), 0);
    assert_eq!(scene.sdf_aabbs.len(), 0);
    assert_eq!(scene.fallback_splats.len(), 0);
    // Verify it clones without panic — the derive is real
    let cloned = scene.clone();
    assert_eq!(cloned.splats.len(), 0);
}
```

```bash
cargo test -p vox_render --features spectra-native render_scene_from_default_has_empty_fields 2>&1 | tail -5
```
Expected: FAIL — `error[E0433]: failed to resolve: use of undeclared crate or module`

- [ ] **Step 2: Add `RenderScene` to `hybrid_compose.rs`**

In `ochroma/crates/vox_render/src/hybrid_compose.rs`, after the `HybridMesh` impl block (~line 200):

```rust
/// Engine-level render payload: the geometry the path tracer consumes for one
/// frame. Game-agnostic — no building types, no SDF volumes, no ECS components.
/// The game assembles this from its own CityRenderScene by calling
/// `.to_render_scene()`.
#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    /// Road/pad/detail Gaussian splats.
    pub splats: Vec<vox_core::types::GaussianSplat>,
    /// Ready-asset triangle geometry.
    pub meshes: Vec<HybridMesh>,
    /// Precomputed world-space AABBs for SDF volumes as (min_xyz, max_xyz).
    /// The game computes these from SpatialSdfInstance::world_aabb() before
    /// handing the scene to the renderer.
    pub sdf_aabbs: Vec<([f32; 3], [f32; 3])>,
    /// Atom-only fallback for paths without hybrid mesh support.
    pub fallback_splats: Vec<vox_core::types::GaussianSplat>,
}
```

In `ochroma/crates/vox_render/src/lib.rs`, add:
```rust
pub use hybrid_compose::RenderScene;
```

- [ ] **Step 3: Add `CityRenderScene::to_render_scene()` in the game**

In `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs`, add after the existing `CityRenderScene` impl block:

```rust
impl CityRenderScene {
    /// Convert to the engine-level RenderScene by precomputing SDF AABBs.
    /// CityRenderScene stays game-side (it holds ResidentSdfVolume which has
    /// bevy_ecs deps); RenderScene is the engine-facing payload.
    pub fn to_render_scene(&self) -> vox_render::RenderScene {
        vox_render::RenderScene {
            splats: self.splats.clone(),
            meshes: self.meshes.clone(),
            sdf_aabbs: self.sdfs.iter().map(|s| s.world_aabb()).collect(),
            fallback_splats: self.fallback_splats.clone(),
        }
    }
}
```

- [ ] **Step 4: Run — verify non-trivial output**

```bash
cargo test -p vox_render --features spectra-native render_scene_from_default_has_empty_fields -- --nocapture
```
Expected: PASS. Build also checks the game:
```bash
cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l
```
Expected: `0`

- [ ] **Step 5: Commit**

```bash
git add ochroma/crates/vox_render/src/hybrid_compose.rs \
        ochroma/crates/vox_render/src/lib.rs \
        ochroma/crates/vox_render/tests/resident_renderer_test.rs \
        Ochroma/projects/urban_horizon/src/render_gpu/mod.rs
git commit -m "feat(vox_render): add RenderScene to hybrid_compose + CityRenderScene::to_render_scene"
```

---

## Task 2: Move leaf utilities to `vox_render::util` and re-export in game

This extracts the pure-math and pure-engine helpers from the game's 6K-line `render_gpu/mod.rs` into `vox_render::util`. Callers in `play.rs` and `hud.rs` compile unchanged via re-exports.

**Files:**
- Create: `ochroma/crates/vox_render/src/util.rs`
- Modify: `ochroma/crates/vox_render/src/lib.rs`
- Modify: `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs`

**Acceptance:** `cargo build -p vox_render --features spectra-native` green. `cargo build -p urban_horizon --features spectra` green. `grep -n "fn screen_to_ground\|fn world_to_screen\|struct SceneBounds\|struct CameraAtomFilter" ochroma/crates/vox_render/src/util.rs` → 4 hits. `grep -n "fn screen_to_ground\|fn world_to_screen\|struct SceneBounds\|struct CameraAtomFilter" Ochroma/projects/urban_horizon/src/render_gpu/mod.rs` → 0 hits (moved, not duplicated).

**Wiring requirement:** `pub use vox_render::util::{screen_to_ground, world_to_screen, SceneBounds}` must be present in `render_gpu/mod.rs` BEFORE the original definitions are deleted. No re-export = compile error at `play.rs` and `hud.rs` call sites.

- [ ] **Step 1: Write the failing test**

```rust
// In ochroma/crates/vox_render/tests/resident_renderer_test.rs, add:
#[test]
fn screen_to_ground_projects_centre_pixel() {
    use vox_render::util::screen_to_ground;
    use glam::Mat4;
    // Orthographic camera looking straight down from y=100
    let view = Mat4::look_at_rh(
        glam::vec3(0.0, 100.0, 0.0),
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(0.0, 0.0, -1.0),
    );
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 500.0);
    // Centre pixel of a 100x100 viewport — ray hits y=0 plane somewhere near origin
    let hit = screen_to_ground(view, proj, 50.0, 50.0, 100, 100);
    assert!(hit.is_some(), "centre ray must intersect ground plane");
    let [x, z] = hit.unwrap();
    assert!(x.abs() < 5.0, "expected near origin x, got {x}");
    assert!(z.abs() < 5.0, "expected near origin z, got {z}");
}
```

```bash
cargo test -p vox_render --features spectra-native screen_to_ground_projects_centre_pixel 2>&1 | tail -5
```
Expected: FAIL — `error[E0433]: failed to resolve: use of undeclared module or type 'util'`

- [ ] **Step 2: Create `util.rs` with the moved symbols**

Create `ochroma/crates/vox_render/src/util.rs`. Move the following from `render_gpu/mod.rs` (all private fns become `pub` in `util.rs`, or `pub(crate)` for items with no external callers):

**Camera math (pub — external callers via re-export):**
- `screen_to_ground(view: Mat4, proj: Mat4, px: f32, py: f32, width: u32, height: u32) -> Option<[f32; 2]>`
- `world_to_screen(view: Mat4, proj: Mat4, x: f32, z: f32, width: u32, height: u32) -> Option<(f32, f32)>`

**Bounds (pub — re-exported for game use):**
- `pub struct SceneBounds { pub min: Vec3, pub max: Vec3 }` with `resident_asset_bounds` moved here as a free fn
- `include_mesh_bounds(bounds: &mut SceneBounds, meshes: &[crate::hybrid_compose::HybridMesh])` (pub(crate))
- `include_aabb_bounds(bounds: &mut SceneBounds, aabbs: &[([f32; 3], [f32; 3])])` (pub(crate), replaces `include_sdf_bounds`)
- `include_splat_bounds(bounds: &mut SceneBounds, splats: &[vox_core::types::GaussianSplat])` (pub(crate))

**Camera filter (pub(crate) — zero external callers):**
- `pub struct CameraAtomFilter { eye: Vec3, view_proj: Mat4, max_distance: f32, frame_pad: f32 }`
- `impl CameraAtomFilter { pub fn from_camera(camera: &crate::spectral::RenderCamera, max_distance: f32) -> Self }`

**Pure math helpers (pub(crate)):**
- `push_pitched_roof_mesh(meshes: &mut Vec<crate::hybrid_compose::HybridMesh>, base_center: [f32; 3], half_and_height: [f32; 3], rgb: [f32; 3], object_id: u32)` — no game deps, pure HybridMesh
- `point_in_outline(x: f32, z: f32, outline: &[[f32; 2]]) -> bool` — `#[cfg(feature = "forge-assets")]`
- `subdivide_triangle(points: [Vec3; 3], subdiv: usize) -> Vec<[Vec3; 3]>`
- `box_project_uv(point: Vec3, normal: Vec3) -> [f32; 2]`
- `luminance(rgb: [f32; 3]) -> f32`, `mix(a: f32, b: f32, t: f32) -> f32`, `scale_rgb(rgb: [f32; 3], scale: f32) -> [f32; 3]`, `lift_rgb(rgb: [f32; 3], amount: f32) -> [f32; 3]`

Do NOT move (game deps stay in `render_gpu/mod.rs`):
- `include_sdf_bounds` — replaced by `include_aabb_bounds` above
- `triangle_local_points`, `triangle_local_uv` — take `crate::asset::ReadyAssetMesh`
- `push_oriented_box_mesh` — calls `forge_mesh_material_channel` and `default_channel_texture`

- [ ] **Step 3: Declare `util` module and add re-exports**

In `ochroma/crates/vox_render/src/lib.rs`:
```rust
pub mod util;
```

In `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs`, at the top (after existing `use` statements):
```rust
// Engine utility re-exports — callers in play.rs / hud.rs compile unchanged
pub use vox_render::util::{screen_to_ground, world_to_screen, SceneBounds};
```

Update `CityRenderScene::resident_asset_bounds` in `render_gpu/mod.rs` to use `include_aabb_bounds` from `util` for the SDF portion (via `to_render_scene().sdf_aabbs`), since `include_sdf_bounds` is now gone.

- [ ] **Step 4: Run — verify non-trivial output**

```bash
cargo test -p vox_render --features spectra-native screen_to_ground_projects_centre_pixel -- --nocapture
```
Expected: PASS. Output includes `centre ray must intersect ground plane` NOT printed (assert passed).

```bash
cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l
```
Expected: `0`

```bash
grep -n "fn screen_to_ground\|fn world_to_screen" Ochroma/projects/urban_horizon/src/render_gpu/mod.rs
```
Expected: 0 hits (only the `pub use` re-export remains).

- [ ] **Step 5: Commit**

```bash
git add ochroma/crates/vox_render/src/util.rs \
        ochroma/crates/vox_render/src/lib.rs \
        ochroma/crates/vox_render/tests/resident_renderer_test.rs \
        Ochroma/projects/urban_horizon/src/render_gpu/mod.rs
git commit -m "feat(vox_render): add util module with screen/camera math + geometry helpers extracted from game"
```

---

## Task 3: Move `RetainedRenderMirror` ownership to `ResidentSceneRenderer` (engine side)

This is the key lift: the game currently owns the mirror and calls `plan_deltas`/`queue_resident_refits` itself. After this task the engine owns the mirror and exposes `drain_scene_deltas` + `reset_retained_mirror`. The game's `LiveFrameSource::drain_scene_deltas` becomes a thin forwarder.

**Files:**
- Modify: `ochroma/crates/vox_render/src/resident_renderer.rs`
- Modify: `ochroma/crates/vox_render/tests/resident_renderer_test.rs`

**Acceptance:** `cargo test -p vox_render --features spectra-native scene_delta_replay_exact -- --nocapture` passes and prints `instance_buf[0]=<nonzero u32> instance_buf[1]=<nonzero u32> replay_matches=true`.

**Wiring requirement:** `ResidentSceneRenderer::drain_scene_deltas` must call `self.retained_mirror.plan_deltas` then `plan.queue_resident_refits(self)` — the full body, not a stub. The method is `#[cfg(feature = "spectra-native")]` because `queue_resident_refits` is.

- [ ] **Step 1: Write the failing test**

```rust
// In ochroma/crates/vox_render/tests/resident_renderer_test.rs, add:
#[cfg(feature = "spectra-native")]
#[test]
fn scene_delta_replay_exact() {
    use vox_render::resident_renderer::ResidentSceneRenderer;
    use vox_render::scene_delta_adapter::RetainedRenderMirror;
    use vox_scene::{NodeId, SceneDelta as GraphSceneDelta, SceneTransform};
    // Build a minimal SceneState with 2 known nodes
    // (use the test helper that already exists in this file for SceneState construction)
    let scene = make_two_node_scene_state(); // existing helper in the test file
    let rig = test_light_rig();
    let mut renderer = ResidentSceneRenderer::new_with_tier(256, 256, rig.clone(),
        spectra_types::FidelityTier::Performance, scene.clone())
        .expect("renderer construction must succeed");
    // Stamp the retained mirror with the node order from the scene
    let nodes: Vec<NodeId> = scene.node_ids_in_instance_order(); // from SceneState
    renderer.reset_retained_mirror(nodes.iter().copied())
        .expect("reset_retained_mirror must not fail with unique nodes");
    // Build a transform delta for node 0
    let delta = GraphSceneDelta::SetTransform {
        id: nodes[0],
        transform: SceneTransform::from_translation([1.0, 2.0, 3.0]),
    };
    let mut deltas1 = vec![delta.clone()];
    renderer.drain_scene_deltas(&mut deltas1)
        .expect("transform-only drain must succeed");
    assert!(deltas1.is_empty(), "drain must clear the vec on success");
    let buf1 = renderer.download_resident_instances()
        .expect("download must succeed after drain");
    // Replay: rebuild renderer from scratch, apply same delta, compare
    let mut renderer2 = ResidentSceneRenderer::new_with_tier(256, 256, rig,
        spectra_types::FidelityTier::Performance, scene.clone())
        .expect("second renderer construction must succeed");
    renderer2.reset_retained_mirror(nodes.iter().copied())
        .expect("reset_retained_mirror must not fail on second renderer");
    let mut deltas2 = vec![delta];
    renderer2.drain_scene_deltas(&mut deltas2)
        .expect("replay drain must succeed");
    let buf2 = renderer2.download_resident_instances()
        .expect("replay download must succeed");
    // Byte-identical instance buffers = deterministic
    assert_eq!(buf1.len(), buf2.len(), "instance buffer lengths must match");
    assert!(!buf1.is_empty(), "instance buffer must be non-empty after scene upload");
    let replay_matches = buf1 == buf2;
    println!("instance_buf[0]={} instance_buf[1]={} replay_matches={replay_matches}",
        buf1[0], buf1.get(1).copied().unwrap_or(0));
    assert!(replay_matches, "replay of same SceneDelta log must produce byte-identical instance buffer");
}
```

```bash
cargo test -p vox_render --features spectra-native scene_delta_replay_exact 2>&1 | tail -5
```
Expected: FAIL — `error[E0599]: no method named 'drain_scene_deltas' found for struct 'ResidentSceneRenderer'`

- [ ] **Step 2: Add `retained_mirror` field and new methods to `ResidentSceneRenderer`**

In `ochroma/crates/vox_render/src/resident_renderer.rs`:

2a. Add import (already in crate, just bring into scope):
```rust
use crate::scene_delta_adapter::{RetainedDeltaError, RetainedDeltaPlan, RetainedRenderMirror};
```

2b. Add `retained_mirror` field to the struct (after `delta_ring`):
```rust
/// Engine-owned mirror for the NodeId→instance_index mapping of the scene
/// currently uploaded. Populated by `reset_retained_mirror` after every
/// `set_scene`; drives `drain_scene_deltas` so the game only emits
/// `Vec<vox_scene::SceneDelta>`.
retained_mirror: RetainedRenderMirror,
```

2c. Initialise in the struct literal in `new_from_settings`:
```rust
retained_mirror: RetainedRenderMirror::new(),
```

2d. Add the new methods inside `impl ResidentSceneRenderer`:
```rust
/// Stamp the NodeId→instance_index mapping after a full scene upload.
/// Call once after every `set_scene` that changes the instance order.
pub fn reset_retained_mirror<I>(&mut self, nodes: I) -> Result<(), RetainedDeltaError>
where
    I: IntoIterator<Item = vox_scene::NodeId>,
{
    self.retained_mirror.replace_instances_in_node_order(nodes)
}

/// Drain and apply transform-only `deltas` as TLAS refits. Returns
/// `Err` (without clearing `deltas`) for any structural delta — caller
/// must then trigger a full-scene rebuild via `set_scene`.
/// Clears `deltas` on success.
#[cfg(feature = "spectra-native")]
pub fn drain_scene_deltas(
    &mut self,
    deltas: &mut Vec<vox_scene::SceneDelta>,
) -> Result<RetainedDeltaPlan, String> {
    let plan = self
        .retained_mirror
        .plan_deltas(deltas.as_slice())
        .map_err(|e| format!("retained scene delta plan: {e}"))?;
    if plan.requires_scene_rebuild() {
        return Err(format!(
            "retained scene structural rebuild required \
             (structural_deltas={})",
            plan.stats.structural_deltas
        ));
    }
    plan.queue_resident_refits(self)
        .map_err(|e| format!("retained scene delta apply: {e}"))?;
    deltas.clear();
    Ok(plan)
}

/// Thin forwarding accessors — let the game inspect mirror state without
/// holding a RetainedRenderMirror field.
pub fn retained_instance_index(&self, node: vox_scene::NodeId) -> Option<usize> {
    self.retained_mirror.instance_index(node)
}
pub fn retained_node_for_instance(&self, instance_index: usize) -> Option<vox_scene::NodeId> {
    self.retained_mirror.node_for_instance(instance_index)
}
pub fn retained_node_count(&self) -> usize {
    self.retained_mirror.len()
}
```

- [ ] **Step 3: Run — verify non-trivial output**

```bash
cargo test -p vox_render --features spectra-native scene_delta_replay_exact -- --nocapture
```
Expected: PASS. Output must include:
```
instance_buf[0]=<nonzero> instance_buf[1]=<nonzero> replay_matches=true
```
A zero `instance_buf[0]` means the scene upload is a stub — investigate before proceeding.

- [ ] **Step 4: Commit**

```bash
git add ochroma/crates/vox_render/src/resident_renderer.rs \
        ochroma/crates/vox_render/tests/resident_renderer_test.rs
git commit -m "feat(vox_render): ResidentSceneRenderer owns RetainedRenderMirror — drain_scene_deltas + reset_retained_mirror"
```

---

## Task 4: Remove `retained_mirror` field from `LiveFrameSource` (game → engine handoff)

Now that the engine owns the mirror (Task 3), the game-side field and its helpers are dead weight. This task wires the game to call `self.renderer.drain_scene_deltas` and removes the game-side mirror.

**Files:**
- Modify: `Ochroma/projects/urban_horizon/src/spectra_frame.rs`

**Acceptance:** `cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l` → `0`. `grep -n "retained_mirror" Ochroma/projects/urban_horizon/src/spectra_frame.rs` → only the `// Task 4 removes` comment (if any); zero struct field declarations or local variable assignments to the field.

**Wiring requirement:** `LiveFrameSource::frame_with_scene_deltas` (line 2388) and `frame_gpu_with_scene_deltas` (line 2649) must call `self.renderer.drain_scene_deltas(deltas)` — not `self.drain_scene_deltas(deltas)`. The public `drain_scene_deltas` on `LiveFrameSource` must exist as a thin forwarder.

- [ ] **Step 1: Write the failing grep (establishes current state)**

```bash
grep -n "retained_mirror" Ochroma/projects/urban_horizon/src/spectra_frame.rs | wc -l
```
Expected: nonzero — the field and usages currently exist.

- [ ] **Step 2: Remove `retained_mirror` field from `LiveFrameSource` struct and all initialisers**

2a. Remove `retained_mirror: RetainedRenderMirror` from the struct definition (line 2050).
2b. Remove all `retained_mirror,` entries in struct literals (two sites in `new_with_tier`).
2c. Remove or replace all `self.retained_mirror = …` assignments (two sites: line 2271 in `new_with_tier`, line 2775 in `frame_gpu_with_scene_deltas_or_rebuild`). Replace each with a call to `self.renderer.reset_retained_mirror(nodes.iter().copied())` where `nodes: Vec<NodeId>` is already computed by the preceding `retained_mirror_from_nodes(retained_nodes)` call. After replacing both sites, delete `retained_mirror_from_nodes` free fn (lines 1037–1044) — it is only called for the purpose of building the field.

2d. Update `InstancedSceneBuild` tuple alias at line 1011 — remove element 5 (`RetainedRenderMirror`):
```rust
type InstancedSceneBuild = (
    SceneState,
    u64,
    usize,
    Vec<crate::cim::visible::VisibleCim>,
    CityAtlas,
    // RetainedRenderMirror removed — now engine-owned
    SprayUpload,
);
```
Update all destructuring sites of `InstancedSceneBuild` in `build_instanced_scene_with_breakdown`, `new_with_tier`, `frame_gpu_with_scene_deltas_or_rebuild`, and line 3252.

- [ ] **Step 3: Replace game-side mirror helper method bodies with engine-forwarding calls**

3a. Replace the three accessor methods that delegate to `self.retained_mirror`:
```rust
// BEFORE (approximate):
pub fn retained_instance_index(&self, node: NodeId) -> Option<usize> {
    self.retained_mirror.instance_index(node)
}
// AFTER:
pub fn retained_instance_index(&self, node: NodeId) -> Option<usize> {
    self.renderer.retained_instance_index(node)
}
// Same pattern for retained_node_for_instance and retained_node_count
```

3b. Delete `apply_retained_scene_deltas` private fn (lines 2321–2340) — its logic now lives in `ResidentSceneRenderer::drain_scene_deltas`.

3c. Replace the body of the public `drain_scene_deltas` on `LiveFrameSource` with a thin forwarder:
```rust
pub fn drain_scene_deltas(
    &mut self,
    deltas: &mut Vec<vox_scene::SceneDelta>,
) -> Result<RetainedDeltaPlan, String> {
    self.renderer.drain_scene_deltas(deltas)
}
```

3d. In `frame_with_scene_deltas` (line 2397) and `frame_gpu_with_scene_deltas` (line 2666): confirm both already call `self.drain_scene_deltas(deltas)` which now forwards to the engine. No change needed to these call sites if the forwarder exists.

- [ ] **Step 4: Clean up imports**

Remove `use vox_render::scene_delta_adapter::RetainedRenderMirror` import (line 38) if no remaining direct uses. Keep `RetainedDeltaPlan` — it is part of the public return types of `frame_with_scene_deltas` / `frame_gpu_with_scene_deltas`.

- [ ] **Step 5: Run — verify build and grep**

```bash
cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l
```
Expected: `0`

```bash
grep -n "self\.retained_mirror\|RetainedRenderMirror" Ochroma/projects/urban_horizon/src/spectra_frame.rs
```
Expected: 0 lines (the field and its type are gone from this file).

- [ ] **Step 6: Commit**

```bash
git add Ochroma/projects/urban_horizon/src/spectra_frame.rs
git commit -m "refactor(urban_horizon): remove retained_mirror from LiveFrameSource — engine now owns the delta mirror"
```

---

## Task 5: One config — `TierTable`/`for_tier_from_ron` + `render.ron` [fidelity] block

`spectra-types` gains `TierTable`/`TierEntry` + `for_tier_from_ron`; the game loads the table from `render.ron` and passes it to `new_with_tier`. The hardcoded `for_tier` arms remain as the absent-table fallback.

**Files:**
- Modify: `spectra/rust/spectra-types/src/render_settings.rs`
- Modify: `Ochroma/projects/urban_horizon/assets/config/render.ron`
- Modify: `Ochroma/projects/urban_horizon/src/render_config.rs`
- Modify: `ochroma/crates/vox_render/src/resident_renderer.rs`

**Acceptance:** `cargo test -p spectra-types for_tier_from_ron_roundtrips_default_table -- --nocapture` passes and prints `performance_spp=1 balanced_spp=2 beauty_spp=8 all_match=true`. Then: on the box, after setting `[performance].max_bounces=4` in `render.ron` and re-rendering, `[shot-map] mean_luma` is higher than with `max_bounces=2` — confirming the knob is live.

**Wiring requirement:** `ResidentSceneRenderer::new_from_settings` (line 143) and `new_with_tier` (line 126) must call `RenderSettings::for_tier_from_ron(tier, &table)` — not bare `for_tier(tier)` — when a `TierTable` is available. The `new_with_tier` signature widens to accept `table: &TierTable`; all callers must pass it.

- [ ] **Step 1: Write the failing test**

```rust
// In spectra/rust/spectra-types/src/render_settings.rs, add to the #[cfg(test)] block:
#[test]
fn for_tier_from_ron_roundtrips_default_table() {
    // Default table = all None fields; for_tier_from_ron must produce the same
    // output as for_tier for every tier.
    let table = TierTable::default();
    let tiers = [FidelityTier::Performance, FidelityTier::Balanced, FidelityTier::Beauty];
    let mut all_match = true;
    let mut performance_spp = 0u32;
    let mut balanced_spp = 0u32;
    let mut beauty_spp = 0u32;
    for tier in tiers {
        let from_ron = RenderSettings::for_tier_from_ron(tier, &table);
        let baseline = RenderSettings::for_tier(tier);
        if from_ron.render.spp != baseline.render.spp
            || from_ron.render.max_bounces != baseline.render.max_bounces
            || from_ron.features.spectral.enabled != baseline.features.spectral.enabled
        {
            all_match = false;
        }
        match tier {
            FidelityTier::Performance => performance_spp = from_ron.render.spp,
            FidelityTier::Balanced    => balanced_spp = from_ron.render.spp,
            FidelityTier::Beauty      => beauty_spp = from_ron.render.spp,
        }
    }
    println!("performance_spp={performance_spp} balanced_spp={balanced_spp} beauty_spp={beauty_spp} all_match={all_match}");
    assert!(performance_spp > 0, "performance spp must be nonzero");
    assert!(performance_spp < balanced_spp, "performance spp must be less than balanced");
    assert!(balanced_spp < beauty_spp, "balanced spp must be less than beauty");
    assert!(all_match, "for_tier_from_ron with default (all-None) table must match for_tier for all tiers");
}
```

```bash
cargo test -p spectra-types for_tier_from_ron_roundtrips_default_table 2>&1 | tail -5
```
Expected: FAIL — `error[E0425]: cannot find function 'for_tier_from_ron'`

- [ ] **Step 2: Add `TierEntry`, `TierTable`, and `for_tier_from_ron` to `spectra-types`**

In `spectra/rust/spectra-types/src/render_settings.rs`, near the `FidelityTier` block (~line 1054):

```rust
/// A single overridable tier row. All fields are `Option` — absent means
/// "keep the hardcoded `for_tier` fallback value for this knob."
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TierEntry {
    pub spp:                Option<u32>,
    pub max_bounces:        Option<u32>,
    pub internal_max_width: Option<u32>,
    pub spectral:           Option<bool>,
    /// Only applied on the Beauty tier (audit floor clamps these OFF on
    /// Performance+Balanced inside `for_tier_from_ron`).
    pub restir:             Option<bool>,
    pub nrc:                Option<bool>,
    pub denoiser_enabled:   Option<bool>,
    pub denoiser_strength:  Option<f32>,
    pub denoiser_backend:   Option<DenoiserBackend>,
    pub upscaler_mode:      Option<UpscalerMode>,
    pub upscaler_quality:   Option<UpscalerQuality>,
}

/// Three-row table loaded from the game's `render.ron` `[fidelity]` block.
/// Passed to `RenderSettings::for_tier_from_ron`. `spectra-types` never
/// touches the filesystem — the game layer owns `render.ron` I/O.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TierTable {
    pub performance: TierEntry,
    pub balanced:    TierEntry,
    pub beauty:      TierEntry,
}
```

Add `for_tier_from_ron` to the `impl RenderSettings` block (after `for_tier`, ~line 997):

```rust
/// Like `for_tier` but overlays the caller-supplied `TierTable` on top of
/// the hardcoded fallback. Absent (`None`) fields inherit `for_tier`'s values.
/// Post-overlay audit clamps are applied unconditionally:
///   - Performance + Balanced: ReSTIR off, NRC off, denoiser_backend = Svgf.
///   - These clamps mirror the in-code comments in `for_tier` (lines 930–932)
///     and cannot be overridden from `render.ron` on those tiers.
pub fn for_tier_from_ron(tier: FidelityTier, table: &TierTable) -> Self {
    let mut s = Self::for_tier(tier);
    let e = match tier {
        FidelityTier::Performance => &table.performance,
        FidelityTier::Balanced    => &table.balanced,
        FidelityTier::Beauty      => &table.beauty,
    };
    if let Some(v) = e.spp               { s.render.spp = v; }
    if let Some(v) = e.max_bounces       { s.render.max_bounces = v; }
    if let Some(v) = e.spectral          {
        s.features.spectral = if v { Feature::on() } else { Feature::off() };
    }
    if let Some(v) = e.denoiser_enabled  { s.render.denoiser.enabled = v; }
    if let Some(v) = e.denoiser_strength { s.render.denoiser.strength = v; }
    if let Some(v) = e.denoiser_backend  { s.render.denoiser.backend = v; }
    if let Some(v) = e.upscaler_mode     { s.render.upscaler.mode = v; }
    if let Some(v) = e.upscaler_quality  { s.render.upscaler.quality = v; }
    // AUDIT FLOORS (2026-06-17 ReSTIR/NRC verdict — measured OFF = better):
    match tier {
        FidelityTier::Performance | FidelityTier::Balanced => {
            s.features.restir = Feature::off();
            s.features.nrc    = Feature::off();
            if s.render.denoiser.enabled {
                s.render.denoiser.backend = DenoiserBackend::Svgf;
            }
        }
        FidelityTier::Beauty => {
            if let Some(v) = e.restir {
                s.features.restir = if v { Feature::on() } else { Feature::off() };
            }
            if let Some(v) = e.nrc {
                s.features.nrc = if v { Feature::on() } else { Feature::off() };
            }
        }
    }
    s
}
```

- [ ] **Step 3: Add `fidelity: TierTable` to game-layer config and `render.ron`**

In `Ochroma/projects/urban_horizon/src/render_config.rs`, add to the game config struct:
```rust
/// Fidelity tier overrides loaded from render.ron's [fidelity] block.
/// Default = all-None (falls back to spectra-types' hardcoded for_tier arms).
#[serde(default)]
pub fidelity: spectra_types::TierTable,
```

In `Ochroma/projects/urban_horizon/assets/config/render.ron`, append before the closing paren:
```ron
    fidelity: (
        performance: (
            spp:                Some(1),
            max_bounces:        Some(2),
            internal_max_width: Some(480),
            spectral:           Some(false),
            denoiser_enabled:   Some(true),
            denoiser_strength:  Some(1.0),
            upscaler_mode:      Some(fsr),
            upscaler_quality:   Some(performance),
        ),
        balanced: (
            spp:                Some(2),
            max_bounces:        Some(3),
            internal_max_width: Some(960),
            spectral:           Some(false),
            denoiser_enabled:   Some(true),
            denoiser_strength:  Some(1.0),
            upscaler_mode:      Some(fsr),
            upscaler_quality:   Some(balanced),
        ),
        beauty: (
            spp:                Some(8),
            max_bounces:        Some(6),
            internal_max_width: Some(1280),
            spectral:           Some(true),
            denoiser_enabled:   Some(true),
            denoiser_strength:  Some(0.6),
            upscaler_mode:      Some(none),
            upscaler_quality:   Some(quality),
        ),
    ),
```

- [ ] **Step 4: Wire `for_tier_from_ron` at the sole live-path call site**

In `ochroma/crates/vox_render/src/resident_renderer.rs`, `new_with_tier` (line 126):

```rust
// BEFORE:
pub fn new_with_tier(
    width: u32,
    height: u32,
    rig: LightRig,
    tier: FidelityTier,
    initial: SceneState,
) -> Result<Self, String> {
    let settings = RenderSettings::for_tier(tier);
    ...

// AFTER — table comes from the game config; use an Option<&TierTable> so
// callers that don't have a table (headless tests, benchmarks) pass None:
pub fn new_with_tier(
    width: u32,
    height: u32,
    rig: LightRig,
    tier: FidelityTier,
    tier_table: Option<&spectra_types::TierTable>,
    initial: SceneState,
) -> Result<Self, String> {
    let settings = match tier_table {
        Some(table) => RenderSettings::for_tier_from_ron(tier, table),
        None        => RenderSettings::for_tier(tier),
    };
    ...
```

Update all callers of `new_with_tier` to pass `Some(&render_config.fidelity)` (game path) or `None` (tests/benchmarks). Callers:
- `spectra_frame.rs` `LiveFrameSource` construction (~line 2109) — pass `Some(&self_render_config.fidelity)`.
- Any bench/test call sites in `vox_render/tests/` — pass `None`.

The `new` method (legacy positional seam at line 100) calls `Self::new_from_settings` directly — no change needed there since it does not call `new_with_tier`.

**Headless callers of `near_realtime` (splat_backend.rs ×12, cine/export.rs ×1):** these are test/offline paths and do NOT call `new_with_tier`. Leave them on `near_realtime` — no change required.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cargo test -p spectra-types for_tier_from_ron_roundtrips_default_table -- --nocapture
```
Expected: PASS. Output must include `performance_spp=1 balanced_spp=2 beauty_spp=8 all_match=true`.

```bash
cargo build -p vox_render --features spectra-native 2>&1 | grep "^error" | wc -l
cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l
```
Expected: both `0`.

Box witness (run via `render-spectra-cuda` skill after rsync):
- Set `render.ron` `[performance].max_bounces: Some(4)` → render → record `mean_luma`.
- Restore to `Some(2)` → render → record `mean_luma`.
- Assert the `max_bounces=4` run has higher `mean_luma` (more bounces = more indirect light).

- [ ] **Step 6: Commit**

```bash
git add spectra/rust/spectra-types/src/render_settings.rs \
        Ochroma/projects/urban_horizon/assets/config/render.ron \
        Ochroma/projects/urban_horizon/src/render_config.rs \
        ochroma/crates/vox_render/src/resident_renderer.rs
git commit -m "feat(spectra-types,vox_render): TierTable + for_tier_from_ron + render.ron [fidelity] block — one config"
```

---

## Task 6: Author `scene_proxy_emits_deltas` witness test (game is a SceneDelta emitter)

This is the second required witness (design doc §3). It verifies the game-side `scene_sync` / `build_instanced_scene` path actually emits `vox_scene::SceneDelta` containing real `AddNode` entries, and that `RetainedRenderMirror::plan_deltas` classifies them as a structural rebuild.

**Files:**
- Modify: `Ochroma/projects/urban_horizon/src/scene_sync.rs` (inline test module)

**Acceptance:** `cargo test -p urban_horizon --features spectra scene_proxy_emits_deltas -- --nocapture` passes and prints `add_nodes=3 has_terrain_proto=true structural_rebuild=true`.

**Wiring requirement:** The test must use a real `GameState` with 3 buildings placed (not a mock) and must call the real `scene_proxy` / `SceneSync` path to produce deltas. The delta count and `structural_rebuild` must be asserted on real computed values — `assert!(deltas.len() > 0)` is forbidden.

- [ ] **Step 1: Write the failing test**

```rust
// In Ochroma/projects/urban_horizon/src/scene_sync.rs, at the bottom:
#[cfg(test)]
mod tests {
    use super::*;
    use vox_render::scene_delta_adapter::RetainedRenderMirror;

    #[test]
    fn scene_proxy_emits_deltas() {
        // Build a minimal GameState with 3 placed buildings.
        // Use the test helpers already in the codebase (e.g. GameState::test_with_buildings
        // or build directly from the map builder used in other tests).
        let game = crate::test_helpers::make_game_state_with_n_buildings(3);
        // Collect SceneDelta entries the scene sync would emit on a fresh build.
        let mut sync = SceneSync::new();
        let deltas = sync.full_scene_deltas(&game);

        let add_nodes = deltas.iter().filter(|d| matches!(d, SceneDelta::AddNode { .. })).count();
        let has_terrain_proto = deltas.iter().any(|d| matches!(d, SceneDelta::AddProto { .. }));

        println!("add_nodes={add_nodes} has_terrain_proto={has_terrain_proto}");

        assert_eq!(add_nodes, 3, "expected 3 AddNode deltas for 3 buildings, got {add_nodes}");
        assert!(has_terrain_proto, "expected at least one AddProto (terrain) in the delta log");

        // Plan the deltas — a fresh mirror has no nodes, so structural_rebuild must be true
        let mirror = RetainedRenderMirror::new();
        let plan = mirror.plan_deltas(&deltas).expect("plan_deltas must not fail");
        let structural_rebuild = plan.requires_scene_rebuild();
        println!("structural_rebuild={structural_rebuild}");
        assert!(structural_rebuild, "new mirror + AddNode deltas must require structural rebuild");
    }
}
```

```bash
cargo test -p urban_horizon --features spectra scene_proxy_emits_deltas 2>&1 | tail -5
```
Expected: FAIL — either `error[E0599]: no method named 'full_scene_deltas' found` or test helper missing.

- [ ] **Step 2: Implement `SceneSync::full_scene_deltas` if it does not exist**

In `scene_sync.rs`, add a method that collects all `vox_scene::SceneDelta` entries for a full scene build from `GameState`. If `SceneSync` already has a method that produces these (check existing API before adding), use it. The method must return a `Vec<vox_scene::SceneDelta>` containing `AddProto` and `AddNode` entries for every building in the game state. No `todo!()`.

- [ ] **Step 3: Run — verify non-trivial output**

```bash
cargo test -p urban_horizon --features spectra scene_proxy_emits_deltas -- --nocapture
```
Expected: PASS. Output must include `add_nodes=3 has_terrain_proto=true structural_rebuild=true`.

- [ ] **Step 4: Commit**

```bash
git add Ochroma/projects/urban_horizon/src/scene_sync.rs
git commit -m "test(urban_horizon): scene_proxy_emits_deltas — witness that game emits real SceneDelta log"
```

---

## Task 7: Rename `vox_render::resident_renderer::SceneDelta` → `SceneSyncReport`

Ends the collision between the rebuild-report struct and the boundary enum `vox_scene::SceneDelta`. This is the last step so all other tasks compile cleanly with the old name first.

**Files:**
- Modify: `ochroma/crates/vox_render/src/resident_renderer.rs`
- Modify: `Ochroma/projects/urban_horizon/src/spectra_frame.rs`
- Modify: `Ochroma/projects/urban_horizon/src/render_gpu/mod.rs` (if it imports the type)

**Acceptance:** `grep -rn "vox_render::resident_renderer::SceneDelta\|use.*resident_renderer.*SceneDelta[^P]" Ochroma/projects/urban_horizon/src/` → 0 hits. `cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l` → `0`.

**Wiring requirement:** All public return types that carried `SceneDelta` (the report) must now carry `SceneSyncReport`. The rename must be done with a mechanical find-replace — do not leave any alias or `type SceneDelta = SceneSyncReport` shim.

- [ ] **Step 1: Write the failing grep**

```bash
grep -rn "use vox_render::resident_renderer.*SceneDelta\|vox_render::resident_renderer::SceneDelta" \
    Ochroma/projects/urban_horizon/src/ | wc -l
```
Expected: nonzero (currently references the old name).

- [ ] **Step 2: Rename in source**

2a. In `resident_renderer.rs` line 53: `pub struct SceneDelta {` → `pub struct SceneSyncReport {`.
2b. Update the `impl SceneDelta` block → `impl SceneSyncReport`.
2c. Update all construction sites within `resident_renderer.rs` (lines 769, 959): `SceneDelta { layers_rebuilt: …, layers_reused: … }` → `SceneSyncReport { … }`.
2d. Update `set_scene` return type: `-> Result<SceneSyncReport, String>`.
2e. In `spectra_frame.rs` line 36: `use vox_render::resident_renderer::{ResidentSceneRenderer, SceneDelta};` → `use vox_render::resident_renderer::{ResidentSceneRenderer, SceneSyncReport};`.
2f. Update all uses of `SceneDelta` as the report type in `spectra_frame.rs` — the return types of `frame_with_scene_deltas`, `frame_gpu_with_scene_deltas`, `frame_gpu_with_scene_deltas_or_rebuild`, plus the `delta: SceneDelta` field at line 136, become `SceneSyncReport`.
2g. Update `render_gpu/mod.rs` if it imports the type.
2h. Update `resident_renderer_test.rs` if it references the old name.

- [ ] **Step 3: Run — verify build and grep**

```bash
cargo build -p vox_render --features spectra-native 2>&1 | grep "^error" | wc -l
cargo build -p urban_horizon --features spectra 2>&1 | grep "^error" | wc -l
```
Expected: both `0`.

```bash
grep -rn "vox_render::resident_renderer::SceneDelta\|use.*resident_renderer.*SceneDelta[^P]" \
    Ochroma/projects/urban_horizon/src/
```
Expected: 0 lines.

- [ ] **Step 4: Commit**

```bash
git add ochroma/crates/vox_render/src/resident_renderer.rs \
        Ochroma/projects/urban_horizon/src/spectra_frame.rs \
        Ochroma/projects/urban_horizon/src/render_gpu/mod.rs \
        ochroma/crates/vox_render/tests/resident_renderer_test.rs
git commit -m "refactor(vox_render): rename SceneDelta report struct → SceneSyncReport (ends collision with vox_scene::SceneDelta)"
```

---

## Task 8: Final witness — box 30fps floor + dead-symbol grep

**Files:** No source edits. Box build + render only.

**Acceptance:** Box render prints `internal 512x288 ... STEADY <≤33ms> (≥30 fps)`. All dead-symbol greps → 0.

- [ ] **Step 1: Rsync + box build**

```bash
# rsync ochroma/crates/ and Ochroma/projects/urban_horizon/ to tomespensin
# (use the render-spectra-cuda skill; serialize — no concurrent cargo)
```

- [ ] **Step 2: Run the final witness greps**

```bash
grep -rn "HybridComposeGpu\|splat_rt_gpu\|new_city_instanced\|hybrid_renderer_for" \
    ochroma/crates/ Ochroma/projects/urban_horizon/src/
```
Expected: 0 hits.

```bash
grep -rn "vox_render::resident_renderer::SceneDelta\|use.*resident_renderer.*SceneDelta[^P]" \
    Ochroma/projects/urban_horizon/src/
```
Expected: 0 hits.

- [ ] **Step 3: Box city render at performance tier**

Run via `render-spectra-cuda` skill with the live config (no `OCHROMA_RES_IN` override, performance tier, 96 lots). Confirm log line `internal 512x288 ... STEADY <≤33ms>`. If `>33ms`, this is a regression vs `d9f1a5b` — **do not close the task**, diagnose first.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(vox_render): render-runtime extraction complete — 30fps floor verified, dead paths deleted"
```

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks exist
- [x] Every `Acceptance` criterion names a real non-trivial expected output (not "tests pass", not zeroes)
- [x] Every `Wiring requirement` names an exact function and exact file
- [x] `IMPORTANT NOTES` contains real API signatures read from source (not invented)
- [x] `File Map` lists every file that appears in any task
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies in implementation code
- [x] `Done When` names a specific command and specific human-observable result
- [x] All types, method names, and signatures are consistent across all tasks
- [x] `SceneDelta` (report, `resident_renderer.rs:53`) and `vox_scene::SceneDelta` (enum) never confused — rename is Task 7
- [x] `--features spectra-native` stated on every `vox_render` test/build command
- [x] `for_tier` hardcoded arms preserved as the absent-table fallback — no regression on headless callers
- [x] The 30fps floor witness (`d9f1a5b` baseline) is the explicit final gate in Task 8
- [x] Task ordering is compile-safe: Task 0 (delete dead code) → Task 1 (RenderScene) → Task 2 (util) → Task 3 (engine mirror) → Task 4 (game handoff) → Task 5 (config) → Task 6 (witness test) → Task 7 (rename) → Task 8 (box witness)
