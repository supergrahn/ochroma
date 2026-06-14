# Design: WT-1 — Two-Level Instancing + LOD as One System (2026-06-14)

**Status:** Draft
**Scope:** Replace the single identity-instance TLAS over one flattened triangle-soup BLAS with a real PointInstancer-shaped two-level instancing system (one BLAS per prototype, N hardware instances), unified with a screen-error LOD selector, serving first-party buildings/props, dense foliage at millions of instances, USD ingestion, and user-generated (modding) assets.
**Related:** [Render direction: MESH + RTX Mega Geometry](../../../../.claude/projects/-home-tom-espen-src-ochroma/memory/hybrid-atom-sdf-lod-direction.md), [USD interchange decision](../../../../.claude/projects/-home-tom-espen-src-ochroma/memory/civitas-usd-interchange-decision.md), dead `slang/instancing.slang` two-level traverse (spectra)

---

## 1. Problem Statement

- **No real instancing exists on the hardware-RT path.** `build_triangle_tlas` builds exactly ONE BLAS over the whole flattened scene and inserts exactly ONE `AccelerationStructureInstanceKHR` with an identity transform and `instance_custom_index = 0`, then builds the TLAS with `.primitive_count(1)` (`spectra/rust/spectra-gpu/src/vulkan_backend.rs:837-845`, `:923`, `:956`). A city of 10k buildings + millions of grass blades becomes one giant monolithic BLAS — no per-instance transform, no per-instance material lookup, no sharing of a prototype BLAS across instances, and a full rebuild on any change.
- **The per-instance transform buffer is uploaded but unused.** `uploader.rs` uploads `state.geometry.instance_transforms` (16 floats/instance) into `gpu_scene.instance_transforms` (`spectra/rust/spectra-scene-upload/src/uploader.rs:424-439`, `:630`), but `build_triangle_tlas` never reads it — the transforms reach the GPU and are dropped on the floor.
- **A half-built two-level *software* traverse is dead code.** `spectra/slang/instancing.slang` already implements a TLAS-node + prototype-BVH traverse with per-instance `transform`/`inv_xform`, `blas_start`/`bvh_start`, `material_override`, and a t0→t1 transform lerp for motion blur (`instancing.slang:25-34`, `:168-184`, `:230-264`, `:329-334`). Its buffers `g_tlas_nodes`, `g_proto_bvh_nodes`, `g_instance_transforms_t0/t1` have **zero producers** anywhere in the Rust scene-upload code. It is a PointInstancer-shaped design that was never wired.
- **LOD is fragmented and CPU-side.** `vox_render/src/hierarchical_lod.rs:24` defines a `LodChain` (`[full, 40%, 10%, billboard]`) and `vox_render/src/atom_instances.rs` has an `AtomInstance` + `AssetAtomLibrary` + `DrawUnit{Cluster,Imposter}` selector, but none of this drives the hardware-RT TLAS — there is no path that picks a *prototype BLAS by screen error* and emits a hardware instance against it. LOD transitions, where they exist, are CPU distance bands with hard pops.
- **USD PointInstancer is flattened, losing the instance structure.** `vox_usd/src/lib.rs:602` `instancer_to_splats` expands a PointInstancer into one volume splat per instance position and *does not descend into prototypes* (`:414`) — the native runtime never sees prototypes + per-instance arrays, only a baked splat soup. This contradicts the standing "native primitives are runtime, USD is interchange" decision: there is no native PointInstancer to lower *into*.
- **No modding path.** There is no pipeline that takes an untrusted user mesh (grass/prop/foliage), validates it, atomizes it, auto-generates an LOD chain, and registers it as an instanceable prototype with a stable id under a resource budget.

---

## 2. Done When

Running:

```bash
cargo test -p spectra-scene-upload --test instanced_tlas -- --nocapture
```

prints exactly:

```
instanced_tlas: 3 prototypes, 1_000_000 instances, TLAS prims=1000000, identity_instances=0
per-instance custom_index round-trip OK (instance 999999 -> custom_index 999999 -> proto 2)
```

AND running:

```bash
cargo run -p vox_app --bin engine_runner -- --scene meadow_1M --frames 240
```

renders a field of 1,000,000 grass instances drawn from 3 prototypes at ≥ 30 FPS shown in the window title bar, with NO visible LOD pop when the camera dollies in (a human at the keyboard sees the grass densify smoothly as it nears, not snap), and the far field reads as a continuous green carpet rather than sparse dots — verifiable without reading code.

AND running:

```bash
cargo run -p vox_app --bin engine_runner -- --import-mod ./mods/wild_grass.glb --scene meadow_mod --frames 1
```

prints `mod 'wild_grass' atomized: proto_id=0x7f3a..., lods=4, atoms[L0]=812, vram=1.9MB <= budget 64MB; registered` and renders the modded grass instanced in the same scene with no engine code change.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| `build_instanced_tlas` emits N real instances | `assert_eq!(tlas_stats.primitive_count, 1_000_000); assert_eq!(tlas_stats.identity_instances, 0)` over a real 1M-instance buffer | `assert!(tlas_handle.is_ok())` — passes for the old 1-instance build |
| Per-instance `instance_custom_index` round-trips | trace a ray at instance 999999, `assert_eq!(hit.instance_custom_index, 999999); assert_eq!(proto_of(999999), 2)` | `assert!(hit.instance_custom_index >= 0)` |
| One BLAS per prototype, shared by many instances | `assert_eq!(backend.blas_count(), 3)` after building 1M instances of 3 prototypes | `assert!(backend.blas_count() > 0)` |
| `refit_instanced_tlas` reuses AS+scratch | `let h0 = build(...); refit(h0, &dirty); assert_eq!(tlas_handle_after, h0); assert!(refit_ms < build_ms * 0.3)` | `assert!(refit(...).is_ok())` |
| Screen-error LOD picks coarser proto far away | `assert_eq!(select_lod(footprint_px=0.8), 3); assert_eq!(select_lod(footprint_px=40.0), 0)` | `assert!(lod <= 3)` |
| Stochastic LOD dissolve has no hard pop | render 8 accumulated samples across the transition band, `assert!(max_abs_luma_delta_between_adjacent_pixels < 0.04)` | `assert!(image.is_some())` |
| Energy-preserving far LOD conserves average color | `let avg_l0 = mean_spectral(L0); let avg_l3 = mean_spectral(card_L3); assert!((avg_l0 - avg_l3).abs().max() < 0.05)` | `assert!(card.atoms.len() < l0.atoms.len())` |
| USD PointInstancer lowers to native protos+instances | import `meadow.usda`, `assert_eq!(native.prototypes.len(), 3); assert_eq!(native.instances.len(), 1_000_000); assert_eq!(native.instances[42].proto_index, usd_protoIndices[42])` | `assert!(import.instancers > 0)` |
| Modded mesh atomizes + auto-LODs + registers | import `wild_grass.glb`, `assert_eq!(proto.lod_chain.levels.len(), 4); assert!(proto.lod_chain.levels[0].atoms.len() > 0); assert_eq!(registry.get(proto_id).is_some(), true)` | `assert!(import_mod(...).is_ok())` |
| Mod resource budget rejects oversize asset | feed a 500MB mesh with `budget=64MB`, `assert!(matches!(import_mod(...), Err(ModError::BudgetExceeded { needed_mb: 500, .. })))` | `assert!(result.is_err())` |
| Stable proto id across reloads | `let id1 = register(asset_bytes); drop(); let id2 = register(asset_bytes); assert_eq!(id1, id2)` | `assert!(id1 != 0)` |

---

## 4. Architecture

### 4.1 Native instance model (PointInstancer-shaped)

We steal the USD `PointInstancer` schema as the **data model only** — USD stays interchange/ingestion; the native primitives below are the runtime/scene/save authority (per the standing USD-interchange decision). The model is a **prototypes table** plus a **structure-of-arrays (SoA) instance buffer**:

- `PrototypeTable` — one entry per unique instanceable asset. Each prototype owns: a cooked atom representation (`Vec<GaussianSplat>` per LOD via the existing `LodChain`, `vox_render/src/hierarchical_lod.rs:24`), a built BLAS handle per active LOD, a stable `proto_id`, and a bounding sphere. The triangle BLAS is built from the prototype's mesh LOD (decimated via `forge::mesh::decimate`, `forge/crates/mesh/src/lib.rs:27`); the atom representation drives the splat/aggregate far-LOD.
- `InstanceSoa` — parallel arrays, NOT array-of-struct, so a 1M-instance foliage field streams as flat GPU buffers: `transform` (3×4 row-major, matching `instancing.slang:25` and the AS instance transform), `proto_index: u32`, `id: u32` (becomes `instance_custom_index`), `visible_mask: u8` (drives the USD `invisibleIds` semantics + frustum/budget culling), and optional `velocity`/`angular_velocity` for the t0/t1 motion path already present in `instancing.slang:256-257`.

The existing `AtomInstance` (48 B POD, `vox_render/src/atom_instances.rs:48`) is the *placed-instance AoS* form used by the gameplay/scene layer; the `InstanceSoa` is its de-interleaved GPU upload form. `AtomInstance.asset` maps to `proto_index`, `AtomInstance.instance_id` maps to `id`. We keep `AtomInstance` as the authoring/scene type and add a converter — no duplicate concept.

Threading: `PrototypeTable` and `InstanceSoa` are owned by the render thread; BLAS builds run on the GPU via the backend (Send + Sync `GpuBackend`). The SoA is `Send` plain-data; mutation (refit dirty set) happens on the render thread only.

### 4.2 `build_instanced_tlas` (the WT-1 fix)

Generalizes `build_triangle_tlas` (`vulkan_backend.rs:710`). Instead of one BLAS + one identity instance:

1. **One BLAS per prototype.** Reuse the existing BLAS build block (`vulkan_backend.rs:776-826`) per prototype, keyed by `proto_id`, cached so re-use across frames is free. Each prototype is built once with `PREFER_FAST_TRACE` (matching `:793`).
2. **N `AccelerationStructureInstanceKHR`.** Build a host buffer of N instances (generalizing the single-instance buffer at `:854-862`). For instance `k`: `transform` = `InstanceSoa.transform[k]`; `instance_custom_index_and_mask = vk::Packed24_8::new(id[k] & 0xFFFFFF, visible_mask[k])`; `acceleration_structure_reference.device_handle =` the per-prototype BLAS device address (the call at `:831-836`, now indexed by `proto_index[k]`); flags carry `TRIANGLE_FACING_CULL_DISABLE` as today (`:848`).
3. **TLAS with `.primitive_count(N)`.** Change `:884` size query `&[1u32]` → `&[N]` and `:956` range `.primitive_count(1)` → `.primitive_count(N)`. The instance custom index gives the kernel a per-instance handle for material/prototype lookup at the hit (replacing the hard-coded `0` at `:845`).

This is the WT-1 deliverable: real per-instance transforms + a per-instance custom index, one shared BLAS per prototype.

### 4.3 `refit_instanced_tlas` (WT-4)

For animated foliage (wind) and movers, rebuilding the TLAS every frame over 1M instances is wasteful. Build the TLAS with `BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE | PREFER_FAST_TRACE`, then refit with `BuildAccelerationStructureModeKHR::UPDATE`, `src_acceleration_structure = dst_acceleration_structure = tlas`, reusing the same AS storage and a retained scratch buffer (update scratch size from the size query). Only the changed `AccelerationStructureInstanceKHR` records are re-uploaded (a dirty-instance set indexed by instance id); the BLASes are untouched (rigid prototypes). RADV supports `MODE_UPDATE` for TLAS refit. Movers with deforming geometry are out of scope for refit (they get a per-frame BLAS rebuild on a separate path); rigid sub-parts (e.g. a garage door on an ambulance) ride the TLAS-refit path as separate instances per the articulated-parts budget note.

### 4.4 LOD as one system with instancing

LOD selection and instance emission are **one pass**, not two systems. For each instance, compute the **ray footprint / screen-space error**: project the prototype bounding sphere through the camera (and, for secondary rays, the ray cone/differential half-angle) to a pixel coverage `footprint_px`. The five "done-right" requirements:

- **(a) Screen-error-driven counts.** `footprint_px` selects the LOD index into the prototype's `LodChain` (`hierarchical_lod.rs:24`) AND, for aggregate foliage, the *instance count* kept in a tile (decimate the instance set itself, not just each prototype).
- **(b) Stochastic transitions resolved by temporal accumulation.** A transition between LOD `i` and `i+1` is a per-instance (or per-atom) stochastic coin flip seeded by `frame_seed(frame_index, sample_idx)` (`spectra/rust/spectra-renderer/src/renderer/sample.rs:9`) crossed with the instance `id`. Across accumulated samples the dissolve averages to a smooth cross-fade — NO hard pop. This is the explicit lean on the just-landed temporal work: we never blend two LODs in one sample; we pick one stochastically and let accumulation resolve it.
- **(c) Aggregate / energy-preserving far LOD.** Distant foliage must keep average color, GI bleed, and silhouette. Atom-LOD: merge/decimate atoms into fewer, larger atoms whose spectral payload is the **energy-average** of the merged atoms (the spectral pair-averaging already done in `pack_atom`, `atom_instances.rs:98-118`, generalized to N→1 merges). Instance-LOD: far instances collapse to cards / a single aggregate atom per tile (the `DrawUnit::Imposter` path, `atom_instances.rs:123`). The test asserts mean spectral radiance is conserved within 0.05 (capability table).
- **(d) Ray-footprint-consistent LOD.** Shadow/GI rays must select the SAME LOD as the primary ray that lit the point, or energy leaks (a primary hit on L0 geometry shadowed by an L3 card mismatches). We key LOD off the ray footprint (cone width at the hit distance), so a secondary ray with a wider cone consistently lands on a coarser, matching LOD. This is enforced by routing the same `select_lod(footprint_px)` function on both primary and secondary rays.
- **(e) Streaming the near-field set.** The high-detail L0 prototypes for the near field are streamed/virtualized (SVT-style residency, `spectra-svt-mgr` is the seam); far prototypes hold only their card LOD resident. Eviction is driven by the same screen-error metric.

Both ladders run: **atom-LOD** (merge atoms at distance, energy-preserving) for the splat substrate, and **instance-LOD** (fewer instances → simpler prototype BLAS → card) for the hardware-RT mesh path.

### 4.5 USD ingestion lowering

Because the native model is PointInstancer-shaped, USD ingestion is a *straight lowering*, not a bake. Replace the splat-flattening `instancer_to_splats` (`vox_usd/src/lib.rs:602`) with a `lower_instancer` that: descends into the PointInstancer's prototype prims (today explicitly skipped at `:414`), cooks each to a native `Prototype` (mesh→BLAS + atomize→LodChain), and fills `InstanceSoa` directly from the USD arrays. This is interchange only — no Hydra, no USD at runtime; the native buffers are the runtime authority.

### 4.6 Modding — atomize + instance user-generated assets

A modder drops in a grass/prop/foliage mesh and it travels the *same path* as first-party assets. Pipeline:

1. **Validate + repair (untrusted input).** Parse via the existing format-agnostic importer (`vox_data/src/import_pipeline.rs:60` `import_asset`, which already handles `.ply/.glb/.gltf/.vxm`; USD via `vox_usd`). Reject on: NaN/Inf positions, degenerate/zero-area triangles, unbounded scale, AABB beyond a sane world bound, triangle/vertex count over cap. Repair what is safe (weld, drop degenerates, recompute normals via `forge::mesh::compute_normals`, `forge/crates/mesh/src/lib.rs:20`).
2. **Atomize.** Convert the validated mesh to the engine atom/SDF/Gaussian representation (`GaussianSplat`, `vox_core/src/types.rs:29`) — the same target the importer already produces.
3. **Auto-generate the LOD chain.** Run `generate_lod_chain(&splats)` (`hierarchical_lod.rs:35`) for the atom ladder and `forge::mesh::decimate` (`forge/crates/mesh/src/lib.rs:27`) at the LOD ratios for the BLAS ladder, plus the aggregate/card far-LOD.
4. **Register with a stable proto id.** `proto_id = blake3(canonicalized_asset_bytes)` — content-addressed, so the same mod re-loads to the same id (determinism + stable references across save files). Register in the `PrototypeTable`; instances reference it by `proto_index` resolved from `proto_id`.
5. **Budget + sandbox.** Before registration, sum VRAM (BLAS + atoms + LOD chain) and check against a per-mod budget; reject with `ModError::BudgetExceeded { needed_mb, budget_mb }`. Enforce an instance cap per mod. Parsing runs with no filesystem/network access beyond the asset bytes (the importer is pure data-in/data-out). Hot-load/unload: register/evict a prototype + its BLASes without touching first-party assets; unload frees the BLAS via the existing `free_accel` (`backend.rs:204`).

### 4.7 Cross-vendor

KHR ray-query (`VK_KHR_acceleration_structure` + inline `RayQuery`) is the baseline and is real on RADV / the AMD 780M (the existing `build_triangle_tlas` path runs there). The DMM/CLAS micro-mesh path is the NVIDIA swap-in behind ONE per-prototype interface (`ProtoGeometry` trait, §6) — it is a **stub today**; we design the seam (a prototype can report a micro-mesh geometry instead of a triangle BLAS) but depend only on the KHR path. State per major decision: BLAS build (AMD: triangle BLAS; NVIDIA: DMM swap-in via the seam), refit (both: `MODE_UPDATE`), LOD (both: same screen-error selector; NVIDIA may push displacement to CLAS LOD).

---

## 5. Data Models

```rust
/// One instanceable prototype — owns its BLAS-per-LOD and atom LOD chain.
/// Stable across mod reloads via content-addressed `proto_id`.
pub struct Prototype {
    proto_id: u128,                 // private — blake3 of canonical asset bytes
    bounding_sphere: ([f32; 3], f32),
    lod_chain: LodChain,            // atom ladder (vox_render::hierarchical_lod::LodChain)
    blas_per_lod: [Option<u64>; 4], // GPU BLAS handle per LOD level (None = card-only)
    geometry: ProtoGeometry,        // KHR triangles | (stub) NVIDIA DMM micro-mesh
}

impl Prototype {
    pub fn proto_id(&self) -> u128 { self.proto_id }
    pub fn lod_chain(&self) -> &LodChain { &self.lod_chain }
    pub fn blas_for_lod(&self, lod: u8) -> Option<u64> { self.blas_per_lod[lod as usize] }
}

/// The prototype registry. proto_index (used in InstanceSoa) is the dense slot;
/// proto_id is the stable content hash. First-party and mod assets share it.
pub struct PrototypeTable {
    prototypes: Vec<Prototype>,            // index = proto_index
    by_id: std::collections::HashMap<u128, u32>, // proto_id -> proto_index
}

impl PrototypeTable {
    pub fn register(&mut self, proto: Prototype) -> u32; // returns proto_index
    pub fn proto_index(&self, id: u128) -> Option<u32>;
    pub fn get(&self, idx: u32) -> Option<&Prototype>;
    pub fn unregister(&mut self, id: u128) -> bool;
}

/// Structure-of-arrays per-instance data — the GPU upload form. PointInstancer-shaped.
/// All Vecs are parallel and equal length = instance_count.
pub struct InstanceSoa {
    transform: Vec<[f32; 12]>,      // private — 3x4 row-major, matches instancing.slang:25
    proto_index: Vec<u32>,
    id: Vec<u32>,                   // becomes instance_custom_index (24-bit usable)
    visible_mask: Vec<u8>,          // AS instance mask; 0 = USD invisibleIds / culled
    velocity: Option<Vec<[f32; 3]>>,        // t0->t1 motion (instancing.slang:256)
    angular_velocity: Option<Vec<[f32; 3]>>,
}

impl InstanceSoa {
    pub fn len(&self) -> usize { self.proto_index.len() }
    pub fn transform(&self, k: usize) -> &[f32; 12] { &self.transform[k] }
    pub fn proto_index(&self, k: usize) -> u32 { self.proto_index[k] }
    pub fn id(&self, k: usize) -> u32 { self.id[k] }
    /// Build from the scene-layer AoS form. AtomInstance.asset -> proto_index,
    /// AtomInstance.instance_id -> id.
    pub fn from_atom_instances(insts: &[AtomInstance]) -> Self;
}

/// Stats returned by build_instanced_tlas, asserted by the Done-When test.
pub struct TlasStats {
    pub primitive_count: u32,       // == instance count
    pub identity_instances: u32,    // must be 0 (proves transforms are real)
    pub blas_count: u32,            // == distinct active (proto, lod) pairs
    pub build_ms: f32,
}

/// USD PointInstancer attr -> native field mapping (lowering, not bake):
///   positions      -> InstanceSoa.transform (translation)
///   orientations   -> InstanceSoa.transform (rotation)
///   scales         -> InstanceSoa.transform (scale)
///   protoIndices   -> InstanceSoa.proto_index
///   ids            -> InstanceSoa.id
///   invisibleIds   -> InstanceSoa.visible_mask (cleared bits)
///   velocities     -> InstanceSoa.velocity
///   purpose        -> render/proxy/guide filter at lowering time
///   prototypes[]   -> PrototypeTable entries (mesh->BLAS, atomize->LodChain)
pub struct LoweredInstancer {
    pub prototypes: Vec<Prototype>,
    pub instances: InstanceSoa,
}

#[derive(Debug)]
pub enum ModError {
    Malformed { reason: String },
    BudgetExceeded { needed_mb: u32, budget_mb: u32 },
    InstanceCapExceeded { cap: u32 },
}
```

---

## 6. API

```rust
// ---- Backend (spectra-gpu GpuBackend trait, extends backend.rs:181) ----

/// Build one BLAS per distinct (proto_index, lod) and a TLAS of N instances.
/// Replaces the single-instance build_triangle_tlas for the instanced path.
/// `instances` provides per-instance transform / proto_index / id / mask.
/// `proto_blas` maps proto_index -> per-LOD BLAS handles (built once, reused).
/// Returns the TLAS accel handle + stats. Blocks until the GPU build completes.
/// Threading: call from the render thread; backend is Send + Sync.
fn build_instanced_tlas(
    &self,
    instances: &InstanceSoa,
    proto_blas: &[[Option<u64>; 4]],   // per proto_index
    instance_lod: &[u8],               // selected LOD per instance (from select_lod)
    flags: BuildFlags,                 // ALLOW_UPDATE for the refit path
) -> Result<(u64, TlasStats), GpuError>;

/// Refit an ALLOW_UPDATE TLAS in place (MODE_UPDATE). Reuses AS + retained
/// scratch. Only re-uploads the dirty instance records. BLASes untouched.
/// `tlas` must be a handle from build_instanced_tlas built with ALLOW_UPDATE.
/// Panics: if `tlas` was built without ALLOW_UPDATE.
fn refit_instanced_tlas(
    &self,
    tlas: u64,
    instances: &InstanceSoa,
    dirty: &[u32],                     // instance ids to re-upload
) -> Result<f32, GpuError>;            // returns refit_ms

// One per-prototype geometry seam. KHR triangles today; NVIDIA DMM is a STUB.
pub enum ProtoGeometry {
    KhrTriangles { vertices: GpuBufferHandle, vertex_stride_floats: u32,
                   vertex_count: u32, triangles: GpuBufferHandle,
                   triangle_stride_u32: u32, triangle_count: u32 },
    NvDisplacementMicroMesh { /* stub: not built today */ },
}

// ---- LOD selection (vox_render) ----

/// Pick the LOD index for a screen/ray footprint. Used on BOTH primary and
/// secondary rays so shadow/GI LOD matches primary (requirement (d)).
/// footprint_px: projected pixel coverage of the prototype bounding sphere.
pub fn select_lod(footprint_px: f32, chain: &LodChain) -> u8;

/// Stochastic dissolve coin for a LOD transition, seeded to resolve under
/// temporal accumulation (requirement (b)). Reuses spectra frame_seed.
/// Returns the chosen LOD for THIS sample (no in-sample blending).
pub fn stochastic_lod(footprint_px: f32, chain: &LodChain,
                      instance_id: u32, frame_seed: u32) -> u8;

// ---- USD lowering (vox_usd) ----

/// Lower a USD PointInstancer to native prototypes + InstanceSoa. Interchange
/// only — no runtime USD. Replaces instancer_to_splats (lib.rs:602).
pub fn lower_instancer(prim: &usd::Prim, world: DMat4)
    -> Result<LoweredInstancer, UsdError>;

// ---- Modding (vox_data + vox_render) ----

/// Take untrusted asset bytes -> validated, atomized, auto-LOD'd Prototype with a
/// content-addressed stable proto_id, under a VRAM budget. Same path as 1st-party.
pub fn import_mod_prototype(asset_bytes: &[u8], fmt: AssetFormat,
                            budget_mb: u32, instance_cap: u32)
    -> Result<Prototype, ModError>;
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `build_instanced_tlas` | `Uploader::upload_from_state` (replaces the `build_triangle_tlas` call) | `spectra/rust/spectra-scene-upload/src/uploader.rs:396` | reads the now-USED `instance_transforms`/SoA buffers (`:424-439`) |
| `refit_instanced_tlas` | render loop, when wind/movers dirty instances | `spectra/rust/spectra-renderer/src/renderer/sample.rs` (per-frame, before tracing) | ALLOW_UPDATE TLAS from build; dirty set from sim |
| `select_lod` | both primary and secondary ray emission | spectra ray-gen + shadow/GI kernels (via uniform LOD table) | requirement (d): same fn both paths |
| `stochastic_lod` | LOD table build per frame | render thread, fed `frame_seed(frame_index, sample_idx)` | `sample.rs:9` |
| `InstanceSoa::from_atom_instances` | scene upload | `vox_render/src/atom_instances.rs` (alongside `AtomInstance`) | bridges scene AoS -> GPU SoA |
| `lower_instancer` | USD import walk | `vox_usd/src/lib.rs:414` (the PointInstancer match arm) | replaces flatten at `:602` |
| `import_mod_prototype` | mod loader + content browser | `vox_app/src/content_browser.rs` | hot-load/unload; registers in `PrototypeTable` |
| `PrototypeTable::register` | both USD lowering and mod import | `vox_render` (render-thread owned) | first-party + mod share it |
| `free_accel` (BLAS evict) | mod unload | `spectra/rust/spectra-gpu/src/backend.rs:204` | already exists |

---

## 8. Open Questions

- [ ] LOD selection cadence: per-frame CPU rebuild of the per-instance LOD table, or a GPU compute pass over `InstanceSoa`? (GPU-first directive favors compute; 1M instances/frame on CPU may stall.) Resolve before the plan; leaning GPU compute pass writing `instance_lod[]`.
- [ ] Per-instance custom index is 24-bit (`Packed24_8`, `vulkan_backend.rs:845`) — caps at ~16.7M instances per TLAS. For >16M, do we segment into multiple TLASes or pack a tile id instead of a global id? (1M target fits; foliage "millions" fits; decide the ceiling story.)
- [ ] Atom merge for energy-preserving far LOD: greedy spatial cluster + spectral average, or a precomputed octree reduction? Determinism matters for mod cooking.
- [ ] Mod sandbox depth: is "pure data-in/data-out importer, no fs/net" sufficient, or do we need a separate process / wasm sandbox for untrusted parsers (the glTF/USD parsers themselves are attack surface)?

---

## 9. Out of Scope

- Deforming (skinned) movers on the refit path — refit covers rigid instances + rigid sub-parts only; skinned geometry gets a separate per-frame BLAS rebuild path.
- NVIDIA DMM/CLAS micro-mesh *implementation* — only the per-prototype `ProtoGeometry` seam is designed; the DMM build is a stub.
- Multi-GPU TLAS distribution.
- Runtime USD / Hydra — USD remains strictly ingestion/interchange.
- Replacing the existing software-traversal `instancing.slang` as the production path — it is reused as the algorithmic reference and the motion-blur t0/t1 source, but the hardware KHR TLAS is the production traverse.

---

## 10. Related Plans / Designs

- Depends on: the existing hardware-RT path (`build_triangle_tlas`, `spectra/rust/spectra-gpu/src/vulkan_backend.rs:710`) and the temporal accumulation work (`frame_seed`, `spectra/rust/spectra-renderer/src/renderer/sample.rs:9`).
- Reuses: the dead two-level traverse `spectra/slang/instancing.slang` (PointInstancer layout, t0/t1 transform lerp, prototype-BVH/TLAS-node structure) as the reference for the buffer layout and the motion path; the existing `LodChain`/`generate_lod_chain` (`vox_render/src/hierarchical_lod.rs`), `AtomInstance`/`AssetAtomLibrary`/`DrawUnit` (`vox_render/src/atom_instances.rs`), the format-agnostic importer (`vox_data/src/import_pipeline.rs:60`), and the USD importer (`vox_usd/src/lib.rs`).
- Required before: the dense-foliage scatter-field plan and the mesh mega-geometry render plan.

---

## 11. Implementation Plan (phases)

Each phase names an exact `Done When` command + exact human-visible output.

**Phase 1 — Native model + SoA bridge (no GPU).**
`cargo test -p vox_render --test instance_soa -- --nocapture` prints `from_atom_instances: 1000000 instances, 3 protos, proto_index[999999]=2, id[999999]=999999`.

**Phase 2 — `build_instanced_tlas` (the WT-1 fix).**
`cargo test -p spectra-scene-upload --test instanced_tlas -- --nocapture` prints `instanced_tlas: 3 prototypes, 1_000_000 instances, TLAS prims=1000000, identity_instances=0` and the custom-index round-trip line (see §2). Replaces the call at `uploader.rs:396`.

**Phase 3 — Screen-error LOD + stochastic dissolve.**
`cargo test -p vox_render --test lod_select -- --nocapture` prints `select_lod(0.8)=3 select_lod(40.0)=0; stochastic dissolve luma-delta max=0.031 (< 0.04)` over a rendered transition band.

**Phase 4 — Energy-preserving aggregate far LOD.**
`cargo test -p vox_render --test aggregate_lod -- --nocapture` prints `L0 atoms=812 -> L3 card atoms=1; mean spectral delta max=0.04 (< 0.05)`.

**Phase 5 — `refit_instanced_tlas` (WT-4).**
`cargo test -p spectra-gpu --test tlas_refit -- --nocapture` prints `refit reused handle=<h>; refit_ms=2.1 build_ms=18.4 (refit < 30% build)`.

**Phase 6 — USD lowering.**
`cargo test -p vox_usd --test pointinstancer_lower -- --nocapture` prints `lowered: protos=3 instances=1000000 proto_index[42] matches USD protoIndices[42]`.

**Phase 7 — Modding pipeline.**
`cargo run -p vox_app --bin engine_runner -- --import-mod ./mods/wild_grass.glb --scene meadow_mod --frames 1` prints `mod 'wild_grass' atomized: proto_id=0x..., lods=4, atoms[L0]=812, vram=1.9MB <= budget 64MB; registered` and `cargo test -p vox_data --test mod_budget -- --nocapture` prints `oversize mesh rejected: BudgetExceeded { needed_mb: 500, budget_mb: 64 }` and `stable id: reload matches (0x7f3a...)`.

**Phase 8 — Full scene integration.**
`cargo run -p vox_app --bin engine_runner -- --scene meadow_1M --frames 240` renders 1M grass instances from 3 prototypes at ≥ 30 FPS (title bar), no visible LOD pop on dolly-in, continuous far carpet.

---

## 12. Risks

- **TLAS build quality on dense, overlapping instances.** Millions of overlapping grass instances produce heavy AABB overlap, degrading TLAS traversal. Mitigation: aggregate/card far-LOD collapses overlapping far instances into single primitives (§4.4c); measure traversal cost in Phase 8 and tune the aggregate distance.
- **Refit cost + drift.** `MODE_UPDATE` refits degrade TLAS quality over many frames (no re-fit of the tree topology). Mitigation: periodic full rebuild (e.g. every N frames or when the dirty fraction exceeds a threshold).
- **Bandwidth.** 1M instances × (3×4 transform + indices) re-uploaded per frame is ~50+ MB/frame on the naive path. Mitigation: upload only the dirty set on the refit path; keep `InstanceSoa` resident and patch in place.
- **24-bit custom-index ceiling.** `Packed24_8` caps at 16.7M instances per TLAS (Open Question). Foliage "millions" fits; document the ceiling and the tile-id fallback.
- **Untrusted-asset safety.** Mod parsers (glTF/USD) are attack surface. Mitigation: strict validation + budgets + caps before registration; Open Question on process/wasm sandboxing the parser itself.
- **GPU-side LOD vs CPU.** Per-frame CPU LOD over 1M instances may stall the render thread (Open Question); the GPU compute LOD pass is the GPU-first answer but adds a kernel.

---

## 13. Contradictions found in the code

- **The standing "USD is interchange, native primitives are runtime" decision is currently violated.** There is no native PointInstancer to lower into — `instancer_to_splats` (`vox_usd/src/lib.rs:602`) bakes instances to a splat soup at import and explicitly refuses to descend into prototypes (`:414`). This design *creates* the native PointInstancer (`PrototypeTable` + `InstanceSoa`) so the decision can actually hold.
- **`instance_transforms` is uploaded but dead.** `uploader.rs:424-439` ships per-instance transforms to the GPU that `build_triangle_tlas` never consumes (`vulkan_backend.rs:837` hard-codes identity). The data plumbing half-exists; this design connects it.
- **`instancing.slang` is a complete two-level *software* traverse with no Rust producers.** It proves the intended layout but is orphaned. We adopt its layout for the hardware path and reuse its t0/t1 motion lerp, but it is not the production traverse — the hardware KHR TLAS is. Anyone reading `instancing.slang` should treat it as a reference, not a live path.
