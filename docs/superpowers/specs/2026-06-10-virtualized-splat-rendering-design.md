# Design: Virtualized Splat Rendering — instanced, budget-driven city-scale atoms (2026-06-10)

**Status:** Draft
**Scope:** Make the interactive wgpu splat path render a full city — 10,000 textured buildings — at 60 fps on the AMD 780M by virtualizing the cooked-asset atom sets: one asset-space atom library, per-placement instances, GPU-assisted budget selection, and an on-device expand pass feeding the existing frozen tiled chain. Spectra (Vulkan path tracer) stays the cinematic-stills path, untouched.
**Related:** [SOTA City Block Phase 1 plan](../plans/2026-06-10-sota-city-block-phase1.md) (Task 9 produces this doc), [Atom-Budget Splat Renderer design](./2026-06-06-atom-budget-splat-renderer-design.md) (the shipped per-frame budget stage this builds on), Nanite (Karis, "Nanite: A Deep Dive", SIGGRAPH 2021) as **philosophy reference only** — see §4.2 for what we explicitly refuse to copy.

---

## 1. Problem Statement

- **The cooked-asset path duplicates every atom per placement.** A cooked Forge asset carries 3,150–8,868 atoms (`assets/buildings/forge_starter/atoms/*.json`: craftsman 8,868, school 5,339, rowhouse 4,504, police 3,283, victorian 3,150 — avg ≈ 5,000). `city_scene_inner` (`urban_horizon/src/render_gpu/mod.rs`) CPU-transforms each placed lot's full atom set into world-space `GaussianSplat`s. At 10,000 buildings that is ≈ 50M world atoms: 96 B/`GaussianSplat` = **4.8 GB host** plus 80 B `GpuSplatFull` + 32 B transforms = **5.6 GB GPU** — impossible on a 780M iGPU sharing system RAM, and O(city) CPU work per scene rebuild.
- **Every city change rebuilds the renderer and re-uploads everything.** `SceneRenderer::set_city_scene` (`render_gpu/mod.rs:2280`) constructs a brand-new `TiledSplatRenderer` — full buffer re-allocation + full splat re-upload — whenever a lot develops, a building is plopped, or the window resizes. A growing city hitches on every growth tick.
- **The shipped budget selector cannot see instances.** `AtomBudgetSelector` / `AtomBudgetGpu` (`crates/vox_render/src/atom_budget.rs`, `gpu/atom_budget_gpu.rs`) operate on one flat static splat array. Their `Selection` (indices + crossfade) is produced and **consumed by nothing on the GPU render path** — `TiledSplatRenderer` has no entry point that renders an index subset, so the proven "Nanite for splats" stage is still unwired to the real game frame.
- **Worst-case tile-entry sizing caps the tiled chain at ~131k splats under default limits.** `TiledSplatRenderer::new` sizes sort scratch as `splat_count × MAX_TILES_PER_SPLAT(256) × 4 B` per buffer; with wgpu `Limits::default()` (`max_storage_buffer_binding_size` = 128 MiB) the ceiling is 131,072 splats — two orders of magnitude below a city frame's working set.
- **Far buildings have no cheap representation.** The per-cluster LOD floor is L3 = 1 splat *per cluster* (~70 clusters for the craftsman ⇒ ~70 splats per far building). 10,000 distant buildings would still cost ~700k atoms before the budget sheds whole clusters arbitrarily — there is no asset-level imposter between "70 splats" and "invisible".

---

## 2. Done When

**Headline (all milestones landed), on the AMD 780M (RADV):**
running `cd ~/src/ochroma && cargo run --release --bin scale_trial -- --instanced --buildings 10000` prints

```
[scale_trial] instanced: 10000 buildings | library_atoms=<L> virtual_atoms=<V≈50M> | budget=<B> selected=<S≤B> | select+expand+render p50=<X> ms p99=<Y> ms @ 1280x720
```

with **X ≤ 16.6** (60 fps) and **S ≥ 0.9·B**, writes `scale_trial_instanced.png` where a human sees a recognizable city grid of distinct buildings (≥ 25% non-background coverage, printed), and in the game `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play` shows a HUD line `atoms <S>/<B> | frame <ms>` that stays ≤ 16.6 ms while orbiting a 10k-building city, with **zero** renderer reconstructions after startup (proven by a printed construction counter). A human at the keyboard verifies all of this without reading code.

**Phased Done When per milestone** (each is independently demo-able; later milestones depend on earlier ones):

- **M1 — Instance-aware selection (CPU oracle).** Done When `cargo test -p vox_render --lib instanced_selector -- --nocapture` prints `[instanced] 10000 instances x 5000-atom asset = 50000000 virtual atoms | budget=1000000 selected=<S> visible_instances=<V> select_ms=<T>` with `0.9·budget ≤ S ≤ budget`, a near (<50 m) instance emitting its full L0 cluster detail while a >400 m instance emits ≤ 64 imposter atoms (both counts printed), and two identical calls printing byte-identical selections.
- **M2 — GPU expand into the tiled chain (no re-upload).** Done When `scripts/build-spectra-native.sh test -p vox_render --lib expand_matches_cpu -- --nocapture` (or plain `cargo test` — wgpu path needs no Slang) prints a per-atom max deviation < 1e-5 between the GPU expand output (`GpuSplatFull` + transform pairs read back) and a CPU transform oracle over the same `ClusterDraw` list, AND a second test renders two consecutive frames from different cameras over one `TiledSplatRenderer` built once via `new_with_capacity`, printing both frames' non-black pixel counts (> 0) and asserting the splat/transform buffers were never re-created.
- **M3 — The 60 fps gate at city scale.** Done When the headline `scale_trial -- --instanced --buildings 10000` line above holds on the 780M, AND with `--buildings 1000` the same binary prints a strictly smaller p50 (budget controller idles below the cap) — both lines printed in one run.
- **M4 — Game wiring.** Done When `cargo run --release --bin play` on a saved 10k-lot city prints `[scene] renderer constructed 1x` at exit after a session that includes zoning growth and plopping a building, and the HUD frame-ms stays ≤ 16.6 during orbit (human-observable; HUD is the readout).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Instance dedup memory bound | build library from 5 cooked payloads + 10k instances; `assert!(lib.total_atoms() < 30_000)` and `assert!(selector_resident_bytes < 64<<20)` with both values printed | `assert!(lib.asset_count() > 0)` |
| Budget bound at 10k instances | `select(&cam, 1_000_000, &mut out)`: `assert!(out.atom_count() <= 1_000_000 && out.atom_count() >= 900_000)` on the 50M-virtual-atom scene, printed | `assert!(out.atom_count() <= budget)` on 10 instances — trivially true |
| Far-instance imposter | identical instance at 30 m vs 600 m, budget ∞: `assert!(near_atoms >= 50 * far_atoms && far_atoms >= 1)` with real emitted counts printed | `assert!(lod(600.0) > lod(30.0))` without counting emitted atoms |
| GPU expand correctness | read back expanded `GpuSplatFull` positions; `assert!(max_abs_dev < 1e-5)` vs CPU `instance_quat * (atom_pos) + instance_pos` oracle over ≥ 100k atoms, dev printed | `assert!(readback.len() > 0)` |
| No-rebuild placement | add 1 instance, re-render: `assert_eq!(renderer_constructions, 1)`; frame non-black count changes (new building visible), both printed | checking only `is_ok()` |
| Frame-time gate | `scale_trial --instanced` asserts p50 ≤ 16.6 ms over ≥ 120 frames on the 780M and prints p50/p99 | one warm frame timed with `Instant` around the whole process |
| Deterministic selection | two `select` calls, same camera/instances/budget: `assert_eq!(a.draws, b.draws)` exact vec equality | comparing lengths |
| Crossfade carried through expand | a cluster in the 130–150 m band: read back `opacity_color.w`; `assert!(o > 0.0 && o < full)` on real values, printed | `assert!(scale <= 1.0)` |

---

## 4. Architecture

### 4.1 Honest inventory — what exists today vs what is missing

**Exists and is load-bearing (do not rebuild):**

- `vox_render::atom_budget::AtomBudgetSelector` — CPU budget selector: BVH frustum walk → score (`total_opacity·r²/d²`) → distance LOD → demote/promote heaps → shed → emit indices + crossfade `opacity_scale`. Deterministic, tested to 2M+ splats (`vox_app/src/bin/scale_trial.rs` builds a ~2,048,080-splat scene and proves the `selected ≤ budget` bound over a flight path). Streaming hook `set_cluster_resident(cluster_id, resident)` already exists.
- `vox_render::gpu::atom_budget_gpu::AtomBudgetGpu` — GPU scoring (one thread per cluster, `atom_budget_gpu.wgsl`) + host budget walk, **byte-identical** to the CPU oracle (proven by `gpu_selection_exactly_equals_cpu_oracle`). The GPU/host split is settled doctrine: parallel scoring on device, the tiny sequential heap walk on host. **Gap:** it owns its own `wgpu::Device` (`AtomBudgetGpu::new(max_clusters)`, no `new_with_context`), so its scores live on a different device than the renderer and selection always round-trips the CPU.
- `vox_render::clas` — `build_clusters(splats, target_size)` (grid partition, deterministic ids), `build_cluster_bvh` (median split), `SplatCluster { id, splat_indices, aabb_min/max, center, total_opacity }`. Flat clustering — no cluster-group hierarchy, which is fine (see §4.2).
- `vox_render::hierarchical_lod` — `LOD_LEVEL_COUNT = 4`, fractions `[1.0, 0.4, 0.1, 0.0]` (opacity-sorted nested prefixes; L3 = 1 splat), distances `[0, 50, 150, 400]` m, `select_lod_level(distance, screen_size)`, `crossfade_factor(distance, level)`.
- `vox_render::gpu::tiled_splat_renderer::TiledSplatRenderer` — the resident-frame GPU loop's raster half: frozen, bit-exact-validated four-pass chain `tile_assign → radix_sort → tile_range_build → splat_raster` (TILE_SIZE 16, 8 spectral bands into a 4-layer rgba32float array), persistent buffers uploaded **once** in `new`, one `tile_count` host readback per frame, `GpuTimers` ms readings, the Y-flip camera reconciliation. `splat_buf()` accessor already exposed for the resident GI fold. **Gaps:** splat set is fixed at construction (no capacity-without-upload constructor, no active-count, no `transform_buf` accessor); `tile_assign` writes per-splat depth+conic back into `splat_buf` each frame (good: expand needn't compute view-dependent state); per-frame entry buffers are allocated inside `tile_assign.dispatch` each frame; worst-case `splat_count × 256` entry sizing caps default-limit devices at ~131k splats.
- `vox_render::gpu::resident_gi_raster::ResidentGiRaster` + `GiCombinePass` — the proven resident pattern this design imitates: one shared `GpuContext`, GI radiance folded into `splat_buf` on-device, **zero CPU readback at the seam** (bit-identical to the readback oracle). `GpuContext::from_parts(device, queue, info)` is the shared-device handle; `GpuGi::new_with_context` is the precedent for context-taking twins.
- `vox_render::gpu::splat_buffer` — `GpuSplatFull` (80 B; `position_depth`, `conic` [device-written], `opacity_color`, `spectral[8]`), `gaussian_splat_to_gpu_full`, `gaussian_splats_to_transforms` (2×vec4 = 32 B/splat: scale.xyz then quat xyzw — exactly the per-instance-rotatable form the expand pass needs), plus `SplatBufferAllocator` (8M-slot free-list buffer, the residency precedent).
- `vox_render::world_partition` (3D `CellCoord` cell streaming) and `gpu::instancing::InstanceManager` (CPU batching by asset uuid) — both exist, **neither is wired to any render path**. This design supersedes `InstanceManager`'s role for atoms; world_partition stays the future terrain/unique-atom streaming hook.
- Game (urban_horizon): `ReadyAssetPayload.atoms: Vec<ReadyAssetAtom>` (`position/scale/color/opacity/channel`, asset-local, frontage on +Z) is the cooked atom format; `AssetCatalog::ready_asset(id)`; lots carry `center`, `facing_angle` (yaw-only rotation), `developed`; `game.lot_asset(l) -> Option<&BuildingAsset>`. `SceneRenderer` (persistent `GpuContext` via `gpu_context()` OnceLock) renders via `TiledSplatRenderer` + atmosphere composite. `CameraAtomFilter` does CPU per-asset sphere culling at scene-build time.
- Spectra path tracer (`vox_render::splat_backend`, feature `spectra-native`) — seconds-per-still on the 780M; the cinematics path. Phase 1 Tasks 3–8 harden it. **Not part of the interactive loop and not modified here.**

**Missing (what this design adds):** an asset-space atom library with per-asset cluster/LOD/imposter tables; an instance-aware selector whose unit of work is (instance × cluster) with an instance-level coarse cut; a GPU expand pass that writes the selected, instance-transformed atoms into the tiled chain's persistent buffers (no re-upload, no renderer reconstruction); shared-`GpuContext` scoring; budget-derived (not worst-case) tile-entry capacity; a frame-time budget controller; the game wiring that replaces "rebuild the world Vec + renderer" with "update instances".

### 4.2 Philosophy from Nanite — and what we explicitly reject

We adopt Nanite's *invariants*: frame cost proportional to a budget, never to scene size; visibility-driven, cluster-granular detail; GPU-parallel culling/scoring; pay only for what is resident and visible. We **reject** its triangle mechanisms wherever they exist to solve triangle problems splats do not have:

- **No cluster DAG with boundary-locked simplification.** Nanite's hierarchy exists so simplified triangle clusters share exact boundary edges and never crack. Splats have no shared edges and no cracks: our LODs are strictly nested opacity-sorted *prefix subsets* of the same atoms (already shipped in `atom_budget`), switchable per cluster with a crossfade multiplier. Building a DAG would add cost and solve nothing.
- **No software micro-poly rasterizer.** Nanite rasterizes sub-pixel triangles in software because HW raster wastes 2×2 quads on them. A sub-pixel splat is just a small Gaussian footprint — the tiled EWA raster already handles it natively. The detail problem at distance is *aggregation*, not rasterization: a Gaussian mixture is approximated by a smaller Gaussian mixture (imposter levels, §4.4) — something triangles fundamentally cannot do without separate imposter authoring. This is the splat-native advantage; lean into it.
- **No visibility buffer + deferred material pass.** Nanite shades from a V-buffer assuming one opaque surface per pixel. Splats alpha-blend many translucent contributions per pixel front-to-back; a V-buffer fights the primitive. Shading stays in `splat_raster`'s blend loop.
- **No two-pass HZB occlusion in v1.** Nanite's reprojection-based occlusion pays off under extreme depth complexity. The city-from-orbit camera has modest overdraw, frustum + distance LOD + budget shedding dominate first; HZB-style occlusion of clusters is a *measured later milestone* (Open Question), not a v1 requirement.
- **Page streaming demoted.** Nanite streams cluster pages because film meshes exceed VRAM. Instancing collapses our residency problem: ≤ a few hundred unique assets × ≤ 16k atoms ≈ tens of MB resident (the current starter set is 5 assets, 25,144 atoms ≈ 2.4 MB). The whole library stays resident; `set_cluster_resident` and `world_partition` remain the hooks if unique geometry (terrain detail) ever needs paging.

### 4.3 `AssetAtomLibrary` — cook-once, asset-space (new: `crates/vox_render/src/atom_instances.rs`)

Built once at load from each unique asset's asset-local atoms (`Vec<GaussianSplat>` converted by the game from `ReadyAssetAtom`s). Per asset: `build_clusters(atoms, 128)` + the opacity-prefix LOD tables (reusing the exact `atom_budget` builder logic), **plus two new asset-level imposter sets**: I0 = deterministic voxel-grid downsample of the whole asset to ≤ 64 aggregate atoms (position = opacity-weighted cell mean, scale = cell-fitted, color = opacity-weighted spectral mean), I1 = 1 atom (asset bounds + mean color). All atoms (full + imposters) for all assets are concatenated into one library-order array, uploaded once to a GPU `library_buf` in asset-local form (position/scale/quat/opacity/spectral — 64 B packed, *not* `GpuSplatFull`, since conic/depth are per-frame device-derived anyway). CPU side keeps the cluster AABBs/LOD index tables. Single owner, immutable after build, `Send + Sync` (shared `Arc` between selector and GPU pass).

### 4.4 `InstancedSelector` — two-level cut (new, same module)

The per-frame unit of work is virtual: (instance, cluster). 10k instances × ~70 clusters = 700k virtual clusters — too many for the host heap walk that `atom_budget` proved at ~hundreds of clusters. So the cut is two-level, mirroring distance reality instead of Nanite's DAG:

1. **Instance pass** (GPU when M3 lands, CPU in the M1 oracle): one thread/iteration per instance — frustum test on the instance's transformed bounding sphere, distance, screen size. Output classes: **culled**, **far** (`distance ≥ 150 m` ⇒ the instance is ONE work unit using imposter levels I0/I1 — its "LOD chain" is [I0(≤64), I1(1)]), or **near** (< 150 m ⇒ expand to its per-cluster work units at their distance LODs). City reality bounds the near set: only instances within 150 m can go cluster-granular, typically 100–500 instances ⇒ ≤ ~35k cluster work units + ≤ 10k far units — a host-walkable count.
2. **Budget walk** (host, verbatim oracle logic): the same score (`total_opacity·r²/d²`), demote/promote heaps, deterministic shed — extended so far-instance units demote I0→I1 and near-cluster units demote L0→L3, all comparable under one score. Emits `Vec<ClusterDraw>` (instance, asset, cluster-or-imposter, lod, opacity_scale) plus a prefix-summed atom offset per draw. Deterministic: ordering keys are (score, instance_id, cluster_id).

This is "Nanite's hierarchical cut" translated: the hierarchy is **placement → cluster**, given by the city itself, instead of a precomputed geometry DAG.

### 4.5 GPU expand pass (new: `gpu/expand_draws.rs` + `expand_draws.wgsl`)

Per frame the host uploads the draw list (≤ ~50k × 32 B ≈ 1.6 MB) and instance transforms (dirty-only; 10k × 32 B = 320 KB worst case). One compute thread per *selected atom* (dispatch over the prefix-summed total): binary-search (or per-draw workgroup) its `ClusterDraw`, read the asset-local atom via the library LOD index tables (flattened to GPU buffers at build), apply `pos' = instance_quat ⊗ atom_pos + instance_pos`, `quat' = instance_quat ⊗ atom_quat`, multiply opacity by `opacity_scale`, and write `GpuSplatFull` (conic zeroed — `tile_assign` writes it) + the 2×vec4 transform pair into `TiledSplatRenderer`'s persistent `splat_buf`/`transform_buf` at the draw's offset. Disjoint slots by prefix sum ⇒ deterministic buffer contents. Then `TiledSplatRenderer::render` runs unchanged over the active count. The frozen four passes are **not touched** — same contract as `GiCombinePass` writing into `splat_buf`.

### 4.6 Capacity, not worst case — the tile-entry budget

The renderer is constructed once with `new_with_capacity(ctx, max_splats = budget_cap)`. Entry/sort buffers are sized `budget_cap × ENTRY_HEADROOM × 4 B` with `ENTRY_HEADROOM = 16` (measured typical coverage is single-digit tiles/splat; `MAX_TILES_PER_SPLAT = 256` remains the per-splat clamp), validated against device limits at construction exactly like today's `ExceedsDeviceLimits` contract. At budget 1M: splats 80 MB + transforms 32 MB + entries 64 MB + 3 sort tmps 192 MB ≈ 370 MB — fits the 780M's UMA comfortably where the worst-case sizing could not. Overflow beyond capacity clamps (entries dropped ⇒ a pathological full-screen close-up loses some splats in some tiles) — never an abort. The clamp count is exposed for the HUD.

### 4.7 Budget controller — how "60 fps" is actually held

A simple proportional controller on the measured frame time (`TiledFrame::raster_gpu_ms`, falling back to `wall_ms`): each frame `budget ← clamp(budget · (target_ms / ema_ms)^k, 100k, budget_cap)` with `k≈0.5`, target 14.5 ms (headroom under 16.6). The *design guarantee* is the `atom_budget` invariant — cost bounded by budget — so 60 fps is a controller setpoint, and the M3 gate verifies the controller actually settles there at 10k buildings on the 780M. "Textured" in the interactive path means texture-*baked* atoms: the cook samples the building's PBR textures at each atom's surface position into the atom's 16-band spectral color (cook-side work, game repo, sequenced with Phase 1's texture pipeline); true per-pixel texture sampling remains the Spectra stills path. This is stated honestly: interactive texture detail rides on atom density (~0.15–0.3 m pitch), stills resolve finer.

### 4.8 Threading / device model

Library build and CPU selection run on the sim/render thread (single `&mut` owner, scratch reuse, no locks — same as `AtomBudgetSelector`). All GPU passes share the game's one `GpuContext` (`gpu_context()` OnceLock). The instance-scoring GPU port (M3) takes `new_with_context(&GpuContext, …)` — the `GpuGi::new_with_context` precedent — so its score buffer feeds the host walk with one small readback (≤ 10k × 16 B = 160 KB/frame), and the expand pass binds the renderer's buffers directly with zero CPU round-trip of atom data.

---

## 5. Data Models

```rust
/// One placed copy of a library asset. POD, GPU-uploadable. 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct AtomInstance {
    position: [f32; 3],   // world metres
    asset: u32,           // index into AssetAtomLibrary
    rotation: [f32; 4],   // quat xyzw (yaw-only in civitas, general here)
    instance_id: u32,     // stable game-side id (lot/ploppable)
    _pad: [u32; 3],
}
impl AtomInstance {
    pub fn new(asset: u32, instance_id: u32, position: [f32; 3], rotation: [f32; 4]) -> Self;
    pub fn asset(&self) -> u32; pub fn instance_id(&self) -> u32;
}

/// Immutable per-process atom library. Built once; shared via Arc.
pub struct AssetAtomLibrary {
    atoms: Vec<GaussianSplat>,        // all assets concatenated, asset-local space
    assets: Vec<AssetEntry>,          // private: atom range, clusters, lod tables, imposters, bounds
}
impl AssetAtomLibrary {
    pub fn build(assets: &[Vec<GaussianSplat>], target_cluster_size: usize) -> Self;
    pub fn asset_count(&self) -> usize;
    pub fn total_atoms(&self) -> usize;          // library atoms incl. imposters (NOT virtual)
    pub fn asset_bounds(&self, asset: u32) -> (Vec3, f32); // local center, radius
}

/// One budget-selected draw unit: a cluster (near) or imposter level (far).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterDraw {
    instance: u32,
    asset: u32,
    unit: DrawUnit,        // Cluster { id: u32, lod: u8 } | Imposter { level: u8 }
    opacity_scale: f32,    // crossfade, 1.0 outside transition bands
    atom_offset: u32,      // prefix-summed slot in the frame buffer
    atom_count: u32,
}

/// Selection output + stats (mirrors atom_budget::SelectionStats shape).
pub struct InstancedSelection { draws: Vec<ClusterDraw> }       // accessors: draws(), atom_count()
pub struct InstancedStats {
    pub budget: usize, pub selected: usize,
    pub instances_visible: usize, pub instances_culled: usize, pub instances_far: usize,
    pub lod_histogram: [usize; 6],   // L0..L3, I0, I1
    pub select_us: u64,
}
```

GPU-side layout constraints: `AtomInstance` is the uniform/storage upload format (48 B, 16-byte aligned). The flattened library on GPU packs each atom as 64 B (`pos[3]+opacity_u8x4`, `scale[3]+pad`, `quat[4]`, `spectral8[8×f16→4×u32]`); LOD index tables flatten to one `u32` index buffer + per-(asset,cluster,lod) `(offset, len)` table.

---

## 6. API

```rust
// crates/vox_render/src/atom_instances.rs  (new, CPU)
impl InstancedSelector {
    /// Build over an immutable library. O(1) — tables precomputed in the library.
    pub fn new(library: std::sync::Arc<AssetAtomLibrary>) -> Self;
    /// Replace/update placements. Cheap (copies 48 B/instance); call on city change.
    pub fn set_instances(&mut self, instances: &[AtomInstance]);
    /// Deterministic budget selection. Clears + fills `out`. Never panics
    /// (no instances → selected = 0). Single-threaded, scratch-reusing.
    pub fn select(&mut self, camera: &RenderCamera, budget: usize,
                  out: &mut InstancedSelection) -> InstancedStats;
}

// crates/vox_render/src/gpu/expand_draws.rs  (new, GPU; shared-context house pattern)
impl ExpandDrawsPass {
    /// Shared-device constructor (GpuGi::new_with_context precedent). Uploads the
    /// flattened library ONCE. Errors mirror AtomBudgetGpuError (NoAdapter never
    /// occurs — context is given; ExceedsDeviceLimits validated up front).
    pub fn new_with_context(ctx: &GpuContext, library: &AssetAtomLibrary,
                            max_instances: u32, budget_cap: u32)
                            -> Result<Self, ExpandDrawsError>;
    /// Record the expand dispatch: draw list + instances → splat_buf/transform_buf.
    /// Returns the expanded atom count (== selection.atom_count()).
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder,
                  selection: &InstancedSelection, instances: &[AtomInstance],
                  splat_buf: &wgpu::Buffer, transform_buf: &wgpu::Buffer) -> u32;
}

// crates/vox_render/src/gpu/tiled_splat_renderer.rs  (ADDITIVE ONLY — existing
// TiledSplatRenderer::new / render signatures keep compiling and stay frozen)
impl TiledSplatRenderer {
    /// Allocate persistent buffers for up to `max_splats` WITHOUT an upload.
    /// Entry/sort scratch sized budget-derived (§4.6), device-limit validated.
    pub fn new_with_capacity(ctx: GpuContext, max_splats: u32,
                             width: u32, height: u32) -> Result<Self, TiledRenderError>;
    /// The persistent transform buffer (2×vec4/splat), mirror of splat_buf().
    pub fn transform_buf(&self) -> &wgpu::Buffer;
    /// Number of leading splats the four-pass chain processes next render().
    /// Must be ≤ the constructed capacity. Cheap (uniform rewrite).
    pub fn set_active_splat_count(&mut self, n: u32);
}
// Threading: all of the above render-thread-owned (&mut), like today.
```

Existing signatures relied on (verbatim, do not re-derive): `AtomBudgetSelector::select(&mut self, camera: &RenderCamera, budget: usize, out: &mut Selection) -> SelectionStats`; `AtomBudgetGpu::score(&self, camera: &RenderCamera) -> Result<Vec<GpuClusterScore>, AtomBudgetGpuError>`; `TiledSplatRenderer::render(&mut self, camera: &RenderCamera) -> Result<TiledFrame, TiledRenderError>`; `GpuContext::from_parts(device: &wgpu::Device, queue: &wgpu::Queue, info: &wgpu::AdapterInfo) -> Self`; `gaussian_splats_to_transforms(splats: &[GaussianSplat]) -> Vec<[f32; 4]>`; `GaussianSplat::volume(position, scales, rotation: Quat, opacity: u8, spectral: [u16; 16])`.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `AssetAtomLibrary::build` | `scale_trial` `--instanced` setup; game: `SceneRenderer::new_city` library init from `AssetCatalog` ready payloads | `crates/vox_app/src/bin/scale_trial.rs`; `urban_horizon/src/render_gpu/mod.rs` | once per process |
| `InstancedSelector::set_instances` | game city-change path (replaces full `city_scene_inner` splat rebuild for lots/ploppables); scale_trial per-scenario | `urban_horizon/src/render_gpu/mod.rs` (`SceneRenderer`) | roads/ground stay non-instanced splats appended to the frame |
| `InstancedSelector::select` | every frame before render | `SceneRenderer::render`; scale_trial frame loop | sim thread |
| `ExpandDrawsPass::encode` | same frame, one encoder, before the tiled chain | `SceneRenderer::render` | writes into `splat_buf()`/`transform_buf()` |
| `TiledSplatRenderer::new_with_capacity` | `SceneRenderer::new_city` (ONCE) — `set_city_scene` stops reconstructing the renderer | `urban_horizon/src/render_gpu/mod.rs:2240/2280` | construction counter printed at exit (M4 gate) |
| `set_active_splat_count` + `render` | every frame after expand | `SceneRenderer::render` | frozen chain unchanged |
| Budget controller | wraps `select` budget arg from `TiledFrame` ms | `SceneRenderer::render` | HUD line `atoms S/B \| frame ms` in `play.rs` |
| Imposter builder | inside `AssetAtomLibrary::build` | `crates/vox_render/src/atom_instances.rs` | deterministic voxel downsample |
| M1 CPU oracle tests | `cargo test -p vox_render --lib instanced_selector` | `atom_instances.rs` tests | synthetic 10k-instance scene |

---

## 8. Open Questions

- [ ] **Host-walk ceiling.** If a camera puts > ~50k cluster work units in the near band (dense downtown at street level), does the host budget walk exceed ~2 ms? Measure in M1; if yes, the answer is a GPU prefix-scan budget pass over score-sorted bins (the `AtomBudgetGpu` doctrine extended), not a smaller near radius. Decide from M1 printed `select_ms` before the M3 plan is written.
- [ ] **Per-instance score readback vs CPU scoring.** 10k instance frustum+score on CPU may be fast enough (~10k sphere tests) to skip the GPU instance pass entirely in v1. M1 measures; M3 only ports it if CPU select_ms > 2 ms.
- [ ] **HZB-style cluster occlusion** — adopt only if M3 profiling shows raster ms dominated by occluded far-side splats at street level. Explicitly out of v1 (§4.2).
- [ ] **Radix-sort nondeterminism** (equal-key `atomicAdd` scatter, the known landmine): selection and expand are deterministic; the rendered pixels still are not where splats depth-tie. The M2/M3 gates therefore assert at the buffer seam and on aggregate pixel stats, never bit-exact framebuffers. A stable-sort pass remains a separate follow-up.
- [ ] **Texture-baked atom cook** (per-atom spectral sampled from the asset's textures) lives in the game cook (`game_asset_cook.rs`) and depends on Phase 1's texture pipeline — sequence it in the Phase 2 plan; the engine side is texture-agnostic (atoms arrive colored).

---

## 9. Out of Scope

- Spectra/Vulkan path-tracer changes — stills pipeline is Phase 1 Tasks 3–8; the interactive path designed here never calls it.
- Disk streaming / paging of the atom library (`world_partition`, `TileManager` wiring) — the library is resident by design at city scale; terrain-detail streaming is its own future design.
- Modifying the frozen four-pass tiled chain shaders (`tile_assign.wgsl`, `radix_sort*.wgsl`, `tile_range_build.wgsl`, `splat_raster.wgsl`) — all integration is via buffers and additive Rust APIs.
- Skinned/deforming instances, per-instance non-uniform scale, and atom-level animation.
- Live building *gameplay* state (households/jobs/save) — that is the companion design `2026-06-10-living-building-instances-design.md` (Task 10); this design only consumes `(asset, transform, instance_id)`.
- GI for the instanced set — `ResidentGiRaster` keeps working on whatever is in `splat_buf`, but tuning GI budgets for 10k buildings is not addressed here.

---

## 10. Related Plans / Designs

- Depends on: [Design: Atom-Budget Splat Renderer](./2026-06-06-atom-budget-splat-renderer-design.md) (shipped; this design instances it)
- Required before: the Phase 2 "Virtualized city rendering" plan named in [SOTA City Block Phase 1](../plans/2026-06-10-sota-city-block-phase1.md) §Phase 2+
- Related: [Living building instances design](./2026-06-10-living-building-instances-design.md) (Task 10 — supplies `instance_id` semantics), AAA Spec 05/11 (TiledSplatRenderer / ResidentGiRaster — the resident-frame foundation), `aaa-phase2-resident-frame` memory (tile_assign latent-bug + radix nondeterminism landmines)
