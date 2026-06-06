# Design: Atom-Budget Splat Renderer — unified budget-driven LOD selection (2026-06-06)

**Status:** Draft
**Scope:** A per-frame, fixed-atom-budget splat selection stage in `vox_render` that unifies the existing-but-dead cluster BVH (`clas.rs`), LOD chains (`hierarchical_lod.rs`), and frustum culling (`frustum.rs`) into one wired pipeline — so frame cost is bounded by a chosen splat budget instead of scene size ("Nanite for splats", competitive-research candidate #4).
**Related:** [Engine competitive research](./2026-06-06-engine-competitive-research.json) (candidate #4, LODGE/Cesium-LOD/NanoGS), [Blitz roadmap](../plans/2026-06-05-ochroma-blitz-7day-full-roadmap.md)

---

## 1. Problem Statement

- `walking_sim::render()` (`crates/vox_app/src/bin/walking_sim.rs:1509`) clones the full ~65k-splat scene into a fresh `Vec` **every frame** (`self.terrain_splats.clone()` + 8 `extend` calls), then the rasteriser CPU-depth-sorts **all** of it (`gpu_rasteriser.rs:444`). Frame cost is O(scene), unbounded as scenes grow.
- Zero culling is wired: splats behind the camera are sorted, converted via `splats_to_gpu()`, and uploaded each frame. `Frustum::contains_sphere` (`crates/vox_render/src/frustum.rs:45`) exists and is never called on the render path.
- The CLAS machinery (`crates/vox_render/src/clas.rs`: `build_clusters`, `build_cluster_bvh` — walking_sim logs "CLAS: 1481 clusters, BVH depth 12") is built at startup and then **used for nothing** per frame.
- `hierarchical_lod.rs` (`LodChain`, `select_lod_level`, `crossfade_factor`) and `streaming.rs` (`TileManager`) are fully implemented, tested, and **wired to no caller** — classic "wire later" debt.
- There is no global atom budget anywhere on the render path: the only budgeted subsystem is GI (`GI_NEAREST_K: usize = 2000`, `walking_sim.rs:136`), which re-sorts the whole scene by distance each GI step to find its subset — another O(scene · log scene) pass that cluster queries would make cheap.

---

## 2. Done When

Running `cargo run --bin walking_sim -- --smoke` prints a line of the exact shape

```
[walking_sim] ATOM BUDGET: budget=24000 selected=<S> of 65000+ clusters_visible=<V>/1481 clusters_culled=<C> lod_histogram=[L0:<a> L1:<b> L2:<c> L3:<d>] select_us=<T>
```

where a human can read off that **S ≤ 24000** while the scene holds 65k+ splats, **C > 0** (frustum culling did real work), the LOD histogram has **at least two non-zero levels** (near clusters at L0, far clusters degraded), and **T < 2000** (selection under 2 ms). The smoke then re-renders with `budget=2000` and prints a second line whose `selected ≤ 2000`, and SMOKE PASS still holds (the frame stays non-black with ≥ 40 distinct colors — the budgeted frame is still the scene, not garbage).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Budget enforcement | `select(&cam, 2000, &mut out)`: `assert!(out.len() <= 2000 && out.len() >= 1900)` on the 65k walking_sim-like scene (budget is *spent*, not just bounded) | `assert!(out.len() <= budget)` on an empty scene — passes trivially |
| Frustum culling via cluster BVH | camera at origin looking +Z, all splats at z < −10: `assert_eq!(out.len(), 0)`; flip camera 180°: `assert!(out.len() > 0)` | `assert!(stats.clusters_culled >= 0)` — always true |
| Distance-driven LOD | two identical 1k-splat clusters, one at 5 m one at 300 m, budget ∞: near cluster contributes `assert!(near_count >= 4 * far_count)` real index counts | `assert!(lod_level(300.0) > lod_level(5.0))` without checking emitted splats |
| Budget pressure degrades LOD globally | same scene, `select` with budget 32k vs 4k: `assert!(histogram_4k[0] < histogram_32k[0] && histogram_4k[3] > histogram_32k[3])` (L0 shrinks, L3 grows) | checking only that both calls return Ok |
| Boundary crossfade | cluster exactly at a LOD transition distance: emitted splats carry `opacity` scaled by `crossfade_factor` — `assert!(faded_opacity < full_opacity && faded_opacity > 0)` on real u8 values | `assert!(crossfade_factor(d, l) <= 1.0)` |
| Deterministic selection | two `select` calls, same camera/budget: `assert_eq!(out_a, out_b)` (exact index vec equality) | comparing only lengths |
| GI subset reuse | `nearest_clusters(player, k)` replaces the per-step full-scene sort in `recompute_gi`: assert the returned subset's max distance ≤ old-path max distance + cluster radius, on real scene data | `assert!(subset.len() == k)` |

---

## 4. Architecture

### 4.1 `AtomBudgetSelector` (new, `crates/vox_render/src/atom_budget.rs`)

Owns the prebuilt `Vec<SplatCluster>` + `ClusterBVHNode` (from `clas.rs`) plus one per-cluster LOD index table built once at scene load. Per frame: (1) walk the BVH against `Frustum::from_view_proj(camera.view_proj())`, collecting visible cluster ids; (2) score each visible cluster by projected solid angle × `total_opacity` (`score = total_opacity * r² / d²` with r = AABB bounding-sphere radius, d = distance to camera — no trig needed); (3) assign each cluster its distance-LOD via `hierarchical_lod::select_lod_level`, then while the summed splat count exceeds the budget, demote the lowest-score clusters one LOD level at a time (a binary-heap pass, O(V log V)); (4) emit splat indices for each cluster at its final level into a caller-owned `Vec<u32>`, applying `crossfade_factor` as an opacity multiplier for clusters within the transition band. Single-threaded, allocation-free after warm-up (reuses internal scratch vecs); runs on the sim thread right before rasterisation. Send + Sync not required (one owner).

### 4.2 Per-cluster LOD index tables (build-time)

`LodChain`'s global stride-sampling is repurposed per cluster: for each `SplatCluster`, store 4 index lists — L0 = all `splat_indices`, L1/L2 = stride-sampled at the existing `LOD_FRACTIONS` (0.4 / 0.1), L3 = the single highest-opacity splat scaled up to the cluster AABB (billboard analogue, reusing the `hierarchical_lod.rs` merge rule). Sampling is opacity-weighted (sort cluster indices by opacity once at build, take prefixes) so L1/L2 keep the most visible atoms. Memory cost: ≈ 1.5 × one `u32` per splat (~400 KB at 65k splats) — negligible next to the 96-byte splats themselves.

### 4.3 Frame assembly without the per-frame clone (vox_app change)

`walking_sim` keeps **static** scene splats (terrain, buildings, trees) in one concatenated `Vec<GaussianSplat>` built once, with the selector built over it. Per frame the selector yields indices into that vec; dynamic splats (orbs, windmill blades, NPC, dropped boxes, GI-lit overlay — a few thousand) are appended unbudgeted after selection, exactly as today. The rasteriser gains an index-slice entry point (§6) so the static set uploads from indices without materialising a cloned `Vec<GaussianSplat>`.

### 4.4 GI subset query (replaces the full-scene sort)

`nearest_clusters(pos, k)`: BVH best-first walk returning cluster ids until ≥ k splats are covered; the GI step gathers those clusters' indices instead of sorting all 65k splats per GI recompute. Same `GI_NEAREST_K` budget semantics, now O(k + log V).

### 4.5 Streaming hook (deferred wiring, decided interface)

The selector exposes `set_cluster_resident(id, bool)`; non-resident clusters are skipped during selection. `TileManager::update_camera` (already in `streaming.rs`) decides residency; actual disk-chunk eviction/load stays out of scope (§9) but the selector-side interface lands now so streaming wires without touching selection logic later.

---

## 5. Data Models

```rust
/// Per-cluster precomputed LOD index lists (built once at scene load).
pub struct ClusterLod {
    cluster_id: u32,
    levels: [Vec<u32>; LOD_LEVEL_COUNT], // indices into the global splat array; L0 ⊇ L1 ⊇ L2, L3 = 1 billboard
}

/// Budget-driven splat selector over a static splat array.
pub struct AtomBudgetSelector {
    clusters: Vec<SplatCluster>,        // from clas::build_clusters
    bvh: ClusterBVHNode,                // from clas::build_cluster_bvh
    lods: Vec<ClusterLod>,              // parallel to clusters
    resident: Vec<bool>,                // streaming hook, all true by default
    scratch: SelectScratch,             // reused per-frame heaps/vecs (private)
}

/// What happened during one select() — the smoke prints this verbatim.
pub struct SelectionStats {
    pub budget: usize,
    pub selected: usize,
    pub clusters_visible: usize,
    pub clusters_culled: usize,
    pub lod_histogram: [usize; LOD_LEVEL_COUNT], // clusters per final LOD level
    pub select_us: u64,                          // wall time of the select call
}

/// Selected splat + its crossfade multiplier (1.0 = fully opaque band).
/// Emitted as parallel arrays to stay GPU-upload friendly.
pub struct Selection {
    pub indices: Vec<u32>,
    pub opacity_scale: Vec<f32>, // same length; 1.0 except in transition bands
}
```

All fields private in the implementation except `SelectionStats`/`Selection` outputs (plain data carriers); accessors per template rule for `AtomBudgetSelector`.

---

## 6. API

```rust
// crates/vox_render/src/atom_budget.rs
impl AtomBudgetSelector {
    /// Build over a static splat array. O(n log n), call once at scene load.
    pub fn build(splats: &[GaussianSplat], target_cluster_size: usize) -> Self;

    /// Select ≤ budget splat indices for this camera. Deterministic.
    /// Clears + fills `out`; returns stats. Panics: never (empty scene → selected=0).
    pub fn select(&mut self, camera: &RenderCamera, budget: usize, out: &mut Selection) -> SelectionStats;

    /// Cluster ids nearest `pos` until ≥ min_splats are covered (GI subset query).
    pub fn nearest_clusters(&self, pos: Vec3, min_splats: usize) -> Vec<u32>;

    /// Splat indices of one cluster at L0 (for GI gather).
    pub fn cluster_indices(&self, cluster_id: u32) -> &[u32];

    /// Streaming hook: non-resident clusters are skipped by select().
    pub fn set_cluster_resident(&mut self, cluster_id: u32, resident: bool);
}
// Threading: single owner, call from the sim/render-prep thread. &mut self because of scratch reuse.

// crates/vox_render/src/gpu/gpu_rasteriser.rs — new indexed entry point alongside render():
pub fn render_indexed(
    &self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_view: &wgpu::TextureView,
    static_splats: &[GaussianSplat],
    selection: &Selection,          // indices + opacity_scale into static_splats
    dynamic_splats: &[GaussianSplat], // appended unbudgeted (orbs, physics, NPC)
    camera: &RenderCamera,
    illuminant: &Illuminant,
);
// Semantics: identical to render() called on the materialised concatenation;
// existing render() becomes a thin wrapper (full-range selection, scale 1.0).
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `AtomBudgetSelector::build` | `WalkingSim::build_scene` (after the existing CLAS stats log) | `crates/vox_app/src/bin/walking_sim.rs:~845` | replaces the stats-only CLAS build; same clusters now load-bearing |
| `AtomBudgetSelector::select` | `WalkingSim::render`, before rasterise | `crates/vox_app/src/bin/walking_sim.rs:1509` | kills the per-frame `terrain_splats.clone()` chain for static splats |
| `render_indexed` | `WalkingSim::render` | same | dynamic splats still appended per frame |
| `nearest_clusters` + `cluster_indices` | `WalkingSim::recompute_gi` | `crates/vox_app/src/bin/walking_sim.rs:954` | replaces the full-scene distance sort; keeps `GI_NEAREST_K` semantics |
| `SelectionStats` print | `run_smoke` | `crates/vox_app/src/bin/walking_sim.rs` (smoke section) | the §2 "Done When" lines, budget 24000 then 2000 |
| `set_cluster_resident` | nothing yet (interface lands now) | — | streaming wiring is explicitly out of scope (§9) but the hook ships so it needs no selector change later |

Engine/game rule check: selector, LOD tables, stats all live in `vox_render` (game-agnostic — splats, clusters, camera only). `vox_app` owns scene composition and budget constants.

---

## 8. Open Questions

*(decided at design time — kept for the record)*

- **CPU or GPU selection?** CPU. At 1.5k–50k clusters the select is O(V log V) ≪ 2 ms; GPU selection (compute culling à la Nanite) only pays off past ~10⁶ splats and is out of scope.
- **Re-sort cost after selection?** Unchanged: the existing rasteriser depth-sort now runs on ≤ budget + dynamic splats instead of the whole scene — strictly less work; no new sort needed.
- **Budget constant?** `vox_app` decides per shell. walking_sim default: 24 000 (≈ ⅓ of today's scene → visible win, zero visible loss at the demo's draw distances per LOD_DISTANCES 50/150/400 m).

---

## 9. Out of Scope

- Disk streaming / chunk eviction (TileManager wiring, async VXM chunk loads) — interface hook only (§4.5).
- GPU-side (compute) culling/selection and indirect draw.
- Importance *pruning* at capture/export time (competitive-research candidate #15 — complements this, separate design).
- LOD *training* (LODGE-style learned LODs); we stride-sample by opacity, we do not optimise representations.
- Changing splat formats or `GaussianSplat` layout.
- The dynamic splat sets (orbs, physics debris, NPC) stay unbudgeted — they are O(hundreds) and game-owned.

---

## 10. Related Plans / Designs

- Depends on: `clas.rs` clusters/BVH, `hierarchical_lod.rs` (`select_lod_level`, `crossfade_factor`, `LOD_FRACTIONS`), `frustum.rs` — all already in `vox_render`, currently unwired.
- Required before: any large-scene streaming work (candidate #4's streaming half); many-light sampler (candidate #19) benefits from the same cluster queries.
- Related: GI atom budget (`GI_NEAREST_K`) — §4.4 unifies its subset query onto the cluster BVH; [GpuGi](../plans/2026-03-30-domain-12a-spectral-gi-wiring.md) EngineLoop wiring (pending) is orthogonal.
