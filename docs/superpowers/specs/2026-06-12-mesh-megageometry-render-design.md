# Design: Mesh + Mega Geometry Rendering — the Real-Geometry Visible World (2026-06-12)

**Status:** Draft
**Scope:** The visible world (100K building instances + 1M movers) renders as REAL triangle meshes through Spectra's path tracer on a tiered, cross-vendor acceleration backend (RTX Mega Geometry on NVIDIA, KHR HW-RT + our software cluster LOD on AMD/Intel, software-compute BVH floor on the 780M); SDF is retired from rendering and kept first-class as the engine SUBSTRATE (destruction, terraforming, collision, volumetric water).
**Related:** [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (one-renderer premise survives; its SDF-primary primitive is superseded), [SDF Pillar](./2026-06-10-sdf-pillar-design.md) (substrate role unchanged), [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (targets refined here), [Virtualized Splat Rendering M1–M3.2](../plans/2026-06-10-virtualized-splat-rendering-m1-m2.md) (visible-set machinery reused), Wave-1 plan: [Mesh Render M1](../plans/2026-06-12-mesh-render-wave1.md).

---

## 0. The Decision (settled, 2026-06-12)

The user's governing value: **"I don't want to fake anything. I want the real things."**

At the cooked SDF resolution (64³-class, ~15–22 cm voxels) a building is only *real* down to the voxel; recessed windows, muntins, trim and cornices are sub-voxel and therefore **faked in shading** (normal maps, procedural muntins, parallax) — exactly what the user rejects. The SDF building-quality track measured the ceiling: the stopped Tasks-2/3 agent reached detail-energy **1.92× against a 3.0 gate** — the faking ceiling, not a tuning problem.

Therefore:

1. **Render primitive = MESH.** Real triangle geometry for everything visible: static buildings AND the 1M moving objects. Forge already generates the mesh (`ReadyAssetPayload.mesh`); the path tracer already renders triangles natively and GPU-verified (`pathtrace_mesh_to_rgba` family).
2. **Acceleration = RTX Mega Geometry as a TIERED cross-vendor backend** (§4.3). NVIDIA gets CLAS cluster LOD + incremental TLAS refit; AMD/Intel get portable KHR HW ray tracing with OUR software-managed cluster LOD/refit; the floor is the software-compute BVH that runs on the 780M today.
3. **SDF = engine substrate only.** Destruction, terraforming, collision, placement, coverage fields, volumetric water. The SDF *render* path (M0/M1/M2 sphere-trace, the A1 cell-grid gather parked on spectra `feat/sdf-atom-cellgrid`) is retired from rendering; it survives as a debug view and as the substrate's query machinery.

Why mesh is the right call (not a taste decision):

- **Real sub-feature geometry.** A triangle mesh carries the recessed window reveal, the muntin bar, the trim profile, the cornice overhang as actual surfaces with actual silhouettes and actual shadows — no shading tricks.
- **Hardware intersection.** Triangles are the ONLY primitive every GPU vendor accelerates in hardware (RT cores / ray accelerators). An SDF march is always a compute shader; a triangle hit on Tier 1/2 is fixed-function silicon.
- **The proven shipping-class path.** Cyberpunk 2077 Overdrive path-traces a mesh city denser per-view than ours at 4K60 — *with DLSS reconstruction + frame generation* — on 40-series hardware weaker than the 4070 Ti (native it is ~8 fps at 1440p on a 4070, per the perf-verdict anchors). That is exactly why reconstruction is a **named dependency** of this design (§4.8), not an optimization.
- **The measured SDF scaling verdict.** Our own numbers: the interval-culled SDF march is O(N-instances) per ray — "dead at 10k instances" (perf verdict, measured on the 780M). The scaling structure for 100K instances is a TLAS over per-type BLASes, which is precisely the triangle path's native shape.

---

## 1. Problem Statement

- The SDF render path fakes sub-voxel detail and measurably cannot reach the quality gate: detail-energy stalled at **1.92× vs the 3.0 gate** on the building-quality track; windows/muntins/trim at 64³ are shading tricks, violating the no-faking directive.
- The Vulkan engine-frame renderer (`spectra-renderer`) traces **one flattened triangle soup** per scene: `KernelSet` registers only `(BVH_TRAVERSE, "bvh_traverse", "trace_bvh")`; rendering 100K building instances today would require duplicating every building's triangles per instance in `GeometryLayer` — there is no dispatched per-type BLAS sharing, so memory and BVH build cost scale with instance count instead of type count.
- The complete kernel-side instancing machinery **exists but is unwired**: `slang/instancing.slang` has `trace_instances` (TLAS BVH traversal, per-prototype BLAS, motion-blur-ready) and `megakernel.slang` already shades instance hits (`is_instance_hit` branch, line ~2249) — but no Rust code registers the kernel, packs `g_instance_data`, or builds `g_tlas_nodes`. `spectra-tlas` is wired into `spectra-renderer` **CPU-side only** (an `Accelerator` is rebuilt in `load_scene_state` from `GeometryLayer.instance_transforms`; the GPU never consumes it).
- 1M movers cannot be SDF at quality (no articulated characters); they need skinned/instanced triangle meshes regardless — a second render primitive for movers would mean two pipelines, violating the one-renderer decision.
- Mods/user assets make the building type library unbounded (~200–400 types base, more with mods): without content-addressed BLAS residency + an import-cook quality gate, every subscribed asset permanently bloats GPU memory and can ship broken (open/non-manifold) meshes into the tracer.

---

## 2. Done When

**Wave 1 (Mesh M1, verifiable on the 780M today):** running

```
SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release \
  --lib mesh_city_block_renders_instanced_buildings -- --nocapture --test-threads=1
```

prints a `[mesh_block]` report containing `instances solid : >= 10/12 (per-instance span-fill >= 0.80)`, `mean span-fill : >= 0.80`, `distinct transforms : 12/12`, `materials : >= 10/12 instances show wall+roof as distinct colours`, and writes `ochroma_mesh_city_block.png` (in the temp dir) showing **12 solid triangle-mesh buildings from 3 shared BLAS prototypes at distinct positions/rotations on a ground plane, with correct inter-building occlusion and real sun shadows** — verifiable by a human with no code reading. (Exact gates: the Wave-1 plan.)

**Design end-state (last rung of §10):** on tomespensin (RTX 4070 Ti), the game view renders a 100K-building-instance / 1M-mover save at **≥ 60 fps at 4K output with reconstruction enabled**, with the FPS counter in the window title showing ≥ 60 for 30 consecutive seconds of free camera flight. Every intermediate rung has its own named command + output (§10).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Instanced mesh primary visibility | `mesh_city_block_renders_instanced_buildings` prints `instances solid : 12/12` from per-instance projected-AABB span-fill ≥ 0.80 over real rendered pixels | `assert!(rgba.len() > 0)` — passes on a black frame |
| Per-type BLAS shared across instances | pack test asserts prototype 1's `bvh_start == prototype 0's num_nodes` and instance records carry the SAME `blas_start` for same-type instances while transforms differ | `assert!(packed.is_some())` |
| Per-triangle materials through the instance path | overlap test: wall-pixel mean colour within ΔE of wall material AND roof-pixel mean within ΔE of roof material, printed | "material buffer exists", returns unit |
| Instance shadows (no fake) | shadow test: ground luma in the slab's shadow band < 0.55 × lit-ground luma, both printed | `assert!(shadow_pass_ran)` |
| Tiered accel probe | `tier_probe` test prints the detected tier + extension list on this box: `tier=3 software (RADV exposes KHR ray_query: yes — unwired)` | hardcoded `Tier::Three` |
| BLAS residency under quota | eviction test: insert types until quota; assert `resident_bytes() <= quota` AND a re-request of an evicted hash re-cooks from `ContentCache` (hit counter increments) | `assert!(cache.read(h).is_some())` after writing h |
| Mover TLAS refit (not rebuild) | refit bench prints `refit 50_000 visible instances: X.X ms (rebuild would be Y.Y ms)` with X measured from `update_transform` + GPU re-upload of the instance buffer only | "refit function exists" |
| 4K reconstruction | upscale test: 1080p internal → 4K output PSNR vs native-4K reference printed and ≥ threshold | `assert!(output.len() == 4*3840*2160)` |

---

## 4. Architecture

### 4.1 One path tracer, mesh primitive

Spectra's path tracer remains the single runtime renderer (settled 2026-06-12; the game owns no raster path). The visible world is triangle geometry: the camera-visible subset of building instances and movers enters the frame as **per-type prototype BLASes + per-instance transforms**, traced by the tier-appropriate backend (§4.3), shaded by the existing megakernel. The GPU-verified single-mesh path (`pathtrace_mesh_to_rgba` / `_textured` / `_lit`, vox_render `splat_backend.rs`) is the proven foundation: scene packing, the 4-light rig + sky dome, the EWA texture atlas, the material packer (`pack_vulkan_mesh_material`, 156-float `MaterialData`) all carry over unchanged. What this design adds is **instancing between the camera and that proven shading** — never a second renderer. Threading model unchanged: vox_render packs `SceneState` on the caller's thread; `Renderer<VulkanSlangBackend>` owns all GPU state, single-threaded per renderer instance.

### 4.2 Primitive / responsibility table

| World element | Primitive | Renders via | Notes |
|---|---|---|---|
| Buildings (100K instances, 200–400+ types) | **Triangle mesh** (Forge-cooked, `ReadyAssetPayload.mesh`) | per-type BLAS + per-instance TLAS entry | real windows/muntins/trim/cornices as geometry |
| Movers — people/vehicles (1M world, visible subset drawn) | **Triangle mesh** (instanced, animated; BLOD library §4.7) | shared BLOD BLASes + per-frame TLAS transform refit | never per-agent BLAS rebuild |
| Terrain | Mesh surface for rendering; **SDF substrate** for terraform/queries | heightfield mesh BLAS (rtxmg displaced-terrain path on Tier 1) | terraform edits SDF → re-mesh patch |
| Destruction | **SDF substrate** (boolean ops) → re-meshed surface | damaged region re-meshed → BLAS patch rebuild | SDF is the authority, mesh is the view |
| Water (volumetric) | **SDF substrate** + volume shading | volume integration in the megakernel | not a surface-mesh problem |
| Physics collision | triangles extracted from the SDF substrate via the existing `vox_physics`/Rapier path | n/a (not rendered) | SDF stays the collision authority |
| Vegetation/scatter | mesh instances (same TLAS machinery) | scatter-field design survives with mesh blades/billboards per LOD | Gaussian-blade runtime track stays superseded |
| Placement/coverage/clearance | **SDF + 2D fields** (SDF pillar, unchanged) | n/a | first-class substrate, not render |

### 4.3 The tiered Mega Geometry acceleration backend

One scene representation (Morton meshlet clusters per type — `spectra-cluster` is vendor-neutral), three bottom-level build strategies. **Only the AS-build call differs per tier; the scene, materials, lights, and shading are identical.** Industry parallel: UE5 ships reduced-detail proxy meshes for non-NVIDIA HW-RT; RTX Mega Geometry is NVIDIA's enhancement that puts full Nanite-class detail in the BVH. Our Tier 2 is *better than a static proxy*: software-selected cluster LOD over the same meshlets.

| Tier | Hardware | Exact Vulkan surface | Bottom-level strategy | LOD | Dynamic (movers) |
|---|---|---|---|---|---|
| **1** | NVIDIA RTX (tomespensin, 4070 Ti) | `VK_NV_cluster_acceleration_structure` + `VK_KHR_acceleration_structure` + `VK_KHR_ray_tracing_pipeline`/`VK_KHR_ray_query`, `VK_KHR_deferred_host_operations` | CLAS from meshlet clusters (RTX Mega Geometry) | per-cluster LOD selection in the AS | incremental TLAS refit + CLAS template instantiation |
| **2** | AMD RDNA2+ / Intel Arc (and the 780M's RDNA3 silicon, once wired via RADV) | `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (ray queries from the compute megakernel — preserves the wavefront architecture) + `VK_KHR_deferred_host_operations` | per-cluster-LOD KHR BLAS per type (our software-managed cluster LOD; pick the LOD's BLAS per instance per frame) | discrete cluster-LOD chain, software-selected | `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR` TLAS refit |
| **3** | anything with Vulkan compute (the 780M **today** — the proven dev floor) | plain compute: `trace_bvh` + `trace_instances` (instancing.slang) software BVH/TLAS kernels | CPU `build_median_bvh` per type (`BLASEntry`), 10-float nodes | discrete LOD chain (separate prototype per LOD) | CPU TLAS rebuild/refit + instance-buffer re-upload |

**Graceful-degradation contract** (generalize `spectra-rtxmg`'s existing behavior — `SubdivisionPipeline::is_available() -> bool`, `build_cas(&surf) -> Option<u64>` returns `None` without HW, `trace(...)` returns `f32::INFINITY` no-hits, error type `RtxmgError::NotAvailable`): every tier-N feature MUST run on tier N+1 with an **identical image** (same hits, same shading), only slower. Tier selection is a startup probe (`vkEnumerateDeviceExtensionProperties`), never a compile-time switch; a probe report is printed at init (`[accel] tier=2 (KHR ray_query) — NV CLAS absent`). Tier 3 is permanently maintained: it is the CI/dev floor and the renderer correctness oracle for Tiers 1–2 (same-image diff tests).

**What exists today, honestly:** Tier 3's kernels are complete in `.slang` (trace_instances has TLAS traversal, per-prototype BLAS, motion-blur transforms) but **no Rust code dispatches them** — Wave 1 wires exactly this. Tier 2 has zero wired code (RADV on the 780M exposes KHR RT — usable for development before tomespensin). Tier 1 has `spectra-rtxmg` (1,120 LOC) for **subd surfaces + displaced terrain only**, with the degradation contract proven; general building/agent CLAS is unbuilt, and `VK_NV_cluster_acceleration_structure` will need raw `ash` bindings (Open Question §9).

### 4.4 Scene model: type library + instance table + visible subset

- **BLAS library is per-TYPE.** ~200–400 unique building types (more with mods) → one BLAS chain per type, content-hash keyed (`ContentCache`, xxhash128, `~/.cache/spectra/`), shared by every instance. 100K instances cost 100K × 24-float instance records (Tier 3) / 64-byte `VkAccelerationStructureInstanceKHR` (Tier 1–2) — **9.6 MB / 6.4 MB**, not 100K building meshes.
- **TLAS holds cheap per-instance transforms.** Tier 3: the 10-float-node software TLAS built over instance world-AABBs (new `build_aabb_tlas`, Wave 1). Tier 1/2: driver TLAS build/refit.
- **Render cost is bound by the camera-visible subset, NOT world counts.** The existing `InstancedSelector` (`vox_render/src/atom_instances.rs`: `set_instances(&[AtomInstance])`, `select(...)` → `InstancedSelection`/`ClusterDraw`) + `FrameBudgetGovernor` (`new(target_ms, initial, min, max)`, `update(frame_ms)`, `budget()`) become the **scene selector feeding the TLAS**: each frame they emit the visible instance set within the frame budget; only those instances enter (or stay in) the TLAS. The governor's adaptive budget is the knob that keeps the refit + trace inside 16.6 ms. This machinery is built and shipped (virtualized M1–M3.2); this design only repoints its output from the wgpu rasterizer to the TLAS instance table.

### 4.5 Residency + import-cook (mods make scale a residency problem, not a render problem)

**Residency:** an LRU GPU residency manager over the BLAS library, mirroring the SDF-atlas eviction pattern that already shipped (SDF pillar M1: quota bytes, `resident_bytes()`, eviction counters, `[sdf_atlas] 200 assets requested | resident=36 MB (quota 64) | evicted=0`). Backing store is the **existing content-addressed BLAS cache**: `ContentCache::{read(hash: u128) -> Option<BLASEntry>, write(hash, &BLASEntry)}` (`spectra-scene-upload/src/content_cache.rs`), where `BLASEntry { vertex_count, triangle_count, bvh_node_count, aabb_min/max, vertices, triangles, bvh_nodes /* ×10 floats */, prim_indices }` is exactly the GPU node format the kernels traverse. Evicted type → instances drop to the coarsest resident LOD or imposter; `FeedbackBuffer::record_miss(blas_id)` (spectra-tlas) is the readback channel for demand-loading.

**Import-cook** (one pipeline for Forge assets AND mods/user meshes):
1. **Validate/repair:** Forge mesh-sealing + outward orientation (forge `building/src/seal.rs`; `orient_outward`/GWN are private fns — re-implement the check engine-side, the SDF-pillar GWN precedent). Reject open meshes with a printed fractional-winding diagnostic.
2. **Decimate into an LOD chain:** quadric edge-collapse exists in `spectra-decimate` but is **pyo3-only** (`#[pyfunction] fn decimate`) — needs a Rust-native export (known gap, Wave 3).
3. **Meshlet-cluster:** `spectra-cluster` Morton clustering (412 LOC; also pyo3-facing — its inner `cluster_raw(vertices, tris, max_tris_per_meshlet) -> (meshlet_offsets, meshlet_counts, reordered_indices)` is the Rust-native core to export).
4. **BLAS build + cache write:** `BLASEntry::build(&LayerGeom)` → `ContentCache::write(hash, &blas)`. Cook once per content hash, ever.
5. **Quality gates:** per-asset triangle budget by class (e.g. ≤ 60K tris LOD0 residential, ≤ 200K landmark), auto-LOD mandatory, budget violations rejected at import with a human-readable report — mods cannot degrade the frame.

### 4.6 LOD strategy

- **Tier 1:** cluster LOD inside the AS (Mega Geometry's core win): per-cluster detail selection, no per-instance BLAS swaps.
- **Tier 2/3:** discrete LOD chains — N prototypes per type (LOD0…LOD3 from the decimation cook); per-frame, per-instance LOD pick by projected solid angle; an LOD switch is a `blas_id`/record swap in the instance table (cheap), never a geometry re-upload.
- **Reframed for a path tracer:** LOD = *what is in the acceleration structure at what detail* + *ray/sample budget* (ReSTIR et al.). One renderer, one budget; the governor (§4.4) owns the trade.

### 4.7 The 1M-moving-object plan

- **Instanced animated meshes** from a small **BLOD (body-LOD) library**: far = rigid keyframe-baked pose prototypes (pure transform instances — thousands share one BLAS); mid = bone-palette/blend-baked pose variants (a few hundred BLASes covering the pose space); near = true per-frame skinned vertices with **BLAS refit** for only the few dozen close-ups.
- **Per-frame TLAS transform refit for the visible movers — NOT per-agent BLAS rebuild.** Tier 1: incremental TLAS refit (Mega Geometry's headline dynamic feature). Tier 2: `UPDATE_KHR`-mode TLAS refit. Tier 3: CPU TLAS rebuild over the visible subset + instance-buffer re-upload (`TlasBuilder::update_transform(index, [[f32;4];3])` + `build()` is the existing CPU shape).
- **⚠ MEASURE REFIT FIRST.** The TLAS refit at visible-mover scale (10K–50K instances changing transforms every frame) is the **#1 unmeasured risk in this design** — it gates the 1M-mover promise and no datapoint exists on any tier. The agents rung (§10 L4) opens with a refit micro-benchmark BEFORE any crowd rendering work; if Tier-2/3 refit blows the budget, the fallback is refit-every-N-frames for far movers + per-frame for near (documented degradation, not a redesign).
- The 1M-agent **simulation** is a separate CPU/ECS workstream (vox_physics ECS) — this design covers only their render.

### 4.8 Reconstruction (named dependency)

4K60 is reached by path tracing at **1080p–1440p internal** and reconstructing to 4K. This is how the shipping-class anchor does it (CP2077 Overdrive: ~8 fps native 1440p on a 4070; 4K60 *only* with DLSS + FG) and how our budget closes: 8–16 ms full PT at 720p-class internal on the 4070 Ti is the honest projected tier (perf verdict), and reconstruction bridges to 4K. Existing surface: `spectra-upscale` (`create_upscaler(...)`, `UpscaleParams::default_at(color, output, depth, motion_vectors, stream)`, `fsr_render_resolution_for_test`) and `spectra-dlss` (PACK_RGBA / PACK_MOTION_VECTORS kernels already registered in `KernelSet`). Tier 1 uses DLSS; Tier 2/3 use the FSR-class path. Motion vectors exist in the megakernel vertex path (velocity in `Vertex`) but are **zero on the instance path today** — instance motion vectors are a reconstruction-rung work item (§10 L5), required before upscaling moving crowds.

### 4.9 SDF substrate boundary (what SDF keeps, exactly)

SDF remains a first-class pillar judged by its four goals (flexibility, power, robustness, speed) for: boolean destruction, terraforming, physics collision (triangles extracted via Rapier path), placement/clearance, coverage fields, AO/distance queries, volumetric water. The SDF **render** entry points (`pathtrace_sdf_to_rgba`, `pathtrace_sdf_scene_to_rgba`, `pathtrace_sdf_scene_with_atoms_to_rgba`, `pathtrace_sdf_scene_textured_to_rgba`) stay in-tree as substrate debug views and regression anchors but receive no further quality investment; the `feat/sdf-atom-cellgrid` branch stays parked. The mesh path REPLACES them as the building-render path. Nothing from the SDF perf week is wasted: the SPIR-V disk cache, the Vulkan timing sink, and the perf-harness pattern (`SdfScenePerfKnobs`/`SdfScenePerfReport`) carry to the mesh path unchanged.

---

## 5. Data Models

House precedent: the `splat_backend.rs` scene-carrier structs (`SdfSceneInstance`, `SdfVolumeInput`, `PbrMaterial`) and the `spectra_scene_state` layers (`GeometryLayer`, `SdfLayer`) are plain pub-field data carriers; the new types follow the same convention for consistency with the module they live in. Stateful managers (residency, selector) keep private fields + accessors per template rule.

```rust
// ── vox_render/src/splat_backend.rs (feature spectra-native) — Wave-1 carriers ──

/// One unique building TYPE: object-space triangle geometry shared by every
/// instance that references it (one BLAS per prototype).
pub struct MeshPrototype {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
    /// Per-triangle material ids into the `materials` slice (empty = all 0).
    pub material_ids: Vec<u8>,
}

/// One placed building: a prototype reference + a world transform + an
/// optional whole-instance material override (-1 = per-triangle materials).
pub struct MeshSceneInstance {
    pub prototype_index: u32,
    pub position: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub uniform_scale: f32,
    pub material_override: i32,
}

/// Optional ground quad packed into the regular (non-instanced) soup — also
/// serves as the non-empty-soup guarantee (a zero-triangle scene hangs the GPU;
/// see the SDF sentinel precedent).
pub struct GroundPlane {
    pub y: f32,
    pub half_extent: f32,
    pub material_id: u8,
}

// ── spectra-scene-state/src/layers.rs — the new scene layer ──

/// Host-side prototype mesh for instanced tracing (positions-only BLAS in M1;
/// normals/uvs land with the textured-instances milestone).
pub struct MeshPrototypeData {
    pub positions: Vec<f32>,     // V*3
    pub indices: Vec<u32>,       // T*3
    pub material_ids: Vec<u32>,  // T (empty => all 0)
    pub vertex_count: usize,
    pub triangle_count: usize,
}

pub struct MeshInstanceRecord {
    pub prototype_index: u32,
    pub transform: [[f32; 4]; 3], // row-major 3x4 object→world
    pub material_override: i32,   // -1 = per-triangle
}

pub struct MeshInstanceLayer {
    pub prototypes: Vec<MeshPrototypeData>,
    pub instances: Vec<MeshInstanceRecord>,
}

// ── spectra-scene-upload — packed GPU buffers (exact kernel contracts) ──

/// Packed buffers matching instancing.slang + megakernel.slang declarations.
pub struct PackedMeshInstances {
    pub instance_data: Vec<f32>,      // N*24 — InstanceData layout (below)
    pub instance_inv_xform: Vec<f32>, // N*12 — row-major 3x4 world→object
    pub blas_vertices: Vec<f32>,      // total_tris*9 — v0v1v2 positions, BVH-SORTED order
    pub blas_tri_material: Vec<u32>,  // total_tris — SAME sorted order
    pub proto_bvh_nodes: Vec<f32>,    // total_nodes*10 — child indices prototype-RELATIVE
    pub tlas_nodes: Vec<f32>,         // tlas_node_count*10 — same BvhNode format
    pub tlas_prim_order: Vec<u32>,    // leaf slot → instance index
    pub instance_count: u32,
    pub total_bvh_nodes: u32,
    pub tlas_node_count: u32,
}
```

**GPU layouts (verbatim from the kernels — do not deviate):**

- `InstanceData` (instancing.slang, 24 floats): `[0–11]` row-major 3×4 object→world; `[12–14]` world AABB min; `[15–17]` world AABB max; `[18]` `blas_start` (first triangle index in the BVH-reordered BLAS flat buffer, int-as-float bits via `asint(asuint(...))`); `[19]` `blas_count`; `[20]` `material_override` (−1 = per-triangle); `[21]` `bvh_start` (absolute node offset of the prototype's root in `g_proto_bvh_nodes`); `[22–23]` pad.
- BVH node (`bvh_build.rs` ↔ `bvh_traverse.slang`/`instancing.slang`, 10 floats): `[min.xyz, max.xyz, left, right, prim_start, prim_count]` — leaf when `prim_count > 0`. **Prototype BLAS child indices are prototype-relative (root = 0); leaf `prim_start` is prototype-local and the kernel reads triangle `(blas_start + prim_start + k)` DIRECTLY from `g_blas_vertices` — so the BLAS triangle buffer MUST be pre-permuted by `BvhBuildResult.sorted_indices` (no `g_prim_indices` indirection on the instance path).**
- `TlasInstance` (spectra-tlas, CPU): 64-byte packed record, `transform: [[f32;4];3]`, `blas_id`, `instance_id`, mask — Tier-1/2's `VkAccelerationStructureInstanceKHR`-shaped future; Tier 3 consumes the 24-float `InstanceData` instead.

Residency manager (Wave 3, accessor-style):

```rust
pub struct BlasResidency { /* private: quota_bytes, resident: LruMap<u128, ResidentBlas>, cache: ContentCache */ }
impl BlasResidency {
    pub fn new(quota_bytes: u64, cache: ContentCache) -> Self;
    pub fn request(&mut self, content_hash: u128) -> ResidencyOutcome; // Resident | Loaded | Evicted{victims} | Missing
    pub fn resident_bytes(&self) -> u64;
    pub fn evicted_count(&self) -> u64;
}
```

---

## 6. API

```rust
// ── EXISTING, GPU-verified — the foundation; signatures are law ──────────────

// vox_render::splat_backend (feature = "spectra-native")
pub fn pathtrace_mesh_to_rgba(
    positions: &[[f32; 3]], normals: &[[f32; 3]], uvs: &[[f32; 2]],
    indices: &[[u32; 3]], material_ids: &[u8], materials: &[PbrMaterial],
    eye: [f32; 3], target: [f32; 3], fov_y: f32,
    width: u32, height: u32, spp: u32, sun_dir: [f32; 3],
) -> Result<Vec<u8>, String>;

pub fn pathtrace_mesh_textured_to_rgba(
    /* mesh args as above */ materials: &[PbrMaterial], textures: &[TextureImage],
    /* camera args */ sun_dir: [f32; 3],
) -> Result<Vec<u8>, String>;

pub fn pathtrace_mesh_lit_to_rgba(
    /* mesh + texture args */ rig: &LightRig,
) -> Result<Vec<u8>, String>;
// PbrMaterial { base_color, roughness, metallic, emission_strength,
//               albedo_tex/roughness_tex/normal_tex: i32 (-1=none), uv_scale: [f32;2] } + Default
// TextureImage { width, height, channels, data: Vec<f32> /* LINEAR */ }
// LightRig { sun_dir, sun_color, sun_intensity, sky_intensity, camera_fill,
//            rim_fill, sky_dome_intensity, sky_dome_zenith, sky_dome_horizon } + Default

// The structural model for the instanced entry point:
pub fn pathtrace_sdf_scene_to_rgba(
    volumes: &[SdfVolumeInput], instances: &[SdfSceneInstance],
    eye: [f32; 3], target: [f32; 3], fov_y: f32,
    width: u32, height: u32, spp: u32, rig: &LightRig,
) -> Result<Vec<u8>, String>;

// spectra-tlas (CPU TLAS, WIRED into spectra-renderer host-side only)
impl TlasInstance {
    pub fn new(blas_id: u32, instance_id: u32) -> Self;
    pub fn with_translation(blas_id: u32, instance_id: u32, tx: f32, ty: f32, tz: f32) -> Self;
    pub fn pack_bytes(&self) -> [u8; 64];
}
pub fn make_trs_transform(translate: [f32; 3], rotate_y_rad: f32, scale: f32) -> [[f32; 4]; 3];
impl TlasBuilder {
    pub fn update_transform(&mut self, index: usize, transform: [[f32; 4]; 3]) -> bool;
    pub fn build(&mut self);
    pub fn packed_bytes(&self) -> &[u8];
}
impl Accelerator {
    pub fn add_blas(&mut self, blas_id: u32, local_aabb_min: [f32; 3], local_aabb_max: [f32; 3]);
    pub fn add_instance(&mut self, instance: TlasInstance) -> usize;
    pub fn rebuild_if_dirty(&mut self) -> bool;
    pub fn traverse_ray(&mut self, ray_origin: [f32; 3], ray_dir: [f32; 3]) -> Vec<usize>;
}

// spectra-scene-upload
impl BLASEntry { pub fn build(layer: &LayerGeom<'_>) -> Self; }
pub struct LayerGeom<'a> { pub positions: &'a [f32], pub normals: &'a [f32],
    pub uvs: &'a [f32], pub indices: &'a [u32], pub material_ids: &'a [u32] }
pub fn build_median_bvh(vertices: &[f32], triangles: &[u32] /* 4 u32/tri */,
    vertex_stride: usize, max_leaf_prims: usize) -> BvhBuildResult;
// BvhBuildResult { node_data: Vec<f32> /* ×10 */, sorted_indices: Vec<u32>, num_nodes: u32 }
impl ContentCache {
    pub fn read(&self, hash: u128) -> Option<BLASEntry>;
    pub fn write(&self, hash: u128, blas: &BLASEntry);
}
pub fn xxhash128_file(path: &Path) -> Option<u128>;
impl GpuScene { pub fn bind_to_map(&self, map: &mut spectra_gpu::BindingMap); }

// spectra-cluster (Rust core behind the pyo3 wrapper — export this natively)
fn cluster_raw(vertices: &[[f32; 3]], tris: &[[i32; 3]], max_tris_per_meshlet: usize)
    -> (Vec<i32> /* offsets */, Vec<i32> /* counts */, Vec<i32> /* reordered indices */);

// spectra-rtxmg (the degradation contract to generalize)
impl SubdivisionPipeline { pub fn is_available(&self) -> bool;
    pub fn build_cas(&self, surf: &RTXMGSubdSurface) -> Option<u64>; }
impl TerrainDisplacementPipeline { pub fn refit_cas(&self, cas: Option<u64>, mesh: &RTXMGTerrainMesh) -> Option<u64>; }

// ── NEW (Wave 1 — exact contracts; the plan's IMPORTANT NOTES copy these) ────

// vox_render::splat_backend
pub fn pathtrace_mesh_scene_to_rgba(
    prototypes: &[MeshPrototype], instances: &[MeshSceneInstance],
    materials: &[PbrMaterial], ground: Option<GroundPlane>,
    eye: [f32; 3], target: [f32; 3], fov_y: f32,
    width: u32, height: u32, spp: u32, rig: &LightRig,
) -> Result<Vec<u8>, String>;
// Errors (explicit, never silent): empty prototypes/instances; prototype_index
// out of range; uniform_scale <= 0; any referenced material with a *_tex >= 0
// (textured instancing is a later milestone — error, don't drop).

// spectra-scene-state
impl MeshInstanceLayer {
    pub fn empty() -> Self;
    pub fn from_parts(prototypes: Vec<MeshPrototypeData>, instances: Vec<MeshInstanceRecord>) -> Self;
    pub fn is_empty(&self) -> bool;
}
impl SceneState { pub fn mark_mesh_instances_changed(&mut self); } // field: pub mesh_instances: MeshInstanceLayer

// spectra-scene-upload
pub fn pack_mesh_instances(layer: &MeshInstanceLayer) -> PackedMeshInstances;
pub fn build_aabb_tlas(world_aabbs: &[([f32; 3], [f32; 3])], max_leaf_prims: usize) -> BvhBuildResult;

// spectra-renderer kernel registrations (kernel_set.rs names module)
pub const TRACE_INSTANCES: &str = "trace_instances";          // instancing.slang :: trace_instances
pub const SHADOW_TRACE_INSTANCES: &str = "trace_shadow_instances"; // instancing.slang :: trace_shadow_instances (new entry)
```

GPU buffer bind names (gpu_scene.rs `bind_to_map`, matching the kernel declarations exactly): `g_instance_data`, `g_instance_inv_xform`, `g_blas_vertices`, `g_blas_tri_material` (new declaration in megakernel.slang), `g_proto_bvh_nodes`, `g_tlas_nodes`, `g_tlas_prim_order`. Uniforms set at dispatch: `u_num_instances`, `u_total_bvh_nodes`, `u_tlas_node_count`, `u_instance_motion_blur = 0`. `RayQueue` (spectra-integrator) gains `hit_tri_in_blas` (alloc'd `fill_i32(-1)`, bound as `g_hit_tri_in_blas` — `hit_instance` already exists with exactly this pattern).

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `MeshInstanceLayer` pack (`pack_mesh_instances`) | `Uploader::upload_from_state` when `!scene.mesh_instances.is_empty()` | `spectra/rust/spectra-scene-upload/src/uploader.rs` | same gate pattern as the SDF layer |
| Instance buffer binds | `GpuScene::bind_to_map` | `spectra/rust/spectra-scene-upload/src/gpu_scene.rs` | bind only when present (SDF precedent) |
| `build_aabb_tlas` | inside `pack_mesh_instances` | `spectra/rust/spectra-scene-upload/src/bvh_build.rs` | leaf order → `g_tlas_prim_order` |
| `TRACE_INSTANCES` registration | `KernelSet::new` `core_kernels` table | `spectra/rust/spectra-renderer/src/kernel_set.rs` | `(TRACE_INSTANCES, "instancing", "trace_instances")` |
| `trace_instances` dispatch | immediately after EVERY `BVH_TRAVERSE` dispatch in the bounce loop, gated on `mesh_instance_count > 0` | `spectra/rust/spectra-renderer/src/renderer.rs` | locate by the BVH_TRAVERSE dispatch call, never by line; reads `best_t = g_hit_t[ray]` so soup↔instance occlusion composes |
| `trace_shadow_instances` dispatch | immediately after every `shadow_trace` dispatch (`renderer.rs` shadow site), same gate | `spectra/rust/spectra-renderer/src/renderer.rs` | early-outs on `g_shadow_vis == 0` |
| `g_hit_tri_in_blas` alloc + bind | `RayQueue` construction + `bind_to_map` | `spectra/rust/spectra-integrator/src/ray_queue.rs` | mirror `hit_instance` (-1 fill) |
| Per-triangle instance materials | megakernel instance branch (`is_instance_hit`) | `spectra/slang/megakernel.slang` | `material_override >= 0 ? override : g_blas_tri_material[blas_start + tri_in_blas]` |
| `pathtrace_mesh_scene_to_rgba` | game/test callers (M1: the milestone test; later: civitas view path) | `ochroma/crates/vox_render/src/splat_backend.rs` | mirrors `pathtrace_sdf_scene_to_rgba` shape |
| `InstancedSelector`/`FrameBudgetGovernor` → TLAS instance table | frame loop (Wave 3 — visible-set rung) | `ochroma/crates/vox_render/src/atom_instances.rs` (existing) + new glue | repoint, don't rebuild |
| `BlasResidency` | asset streaming on type request/eviction (Wave 3) | new `ochroma/crates/vox_render/src/blas_residency.rs` | wraps `ContentCache` |
| Tier probe | renderer/backend init, printed once | `spectra/rust/spectra-gpu/src/vulkan_backend.rs` (Wave 2) | `vkEnumerateDeviceExtensionProperties` |

---

## 8. Risks & Unknowns (honest)

- **Every 4070 Ti number in this document is a PROJECTION until tomespensin runs it.** The only defensible multiplier is 5–7× over the 780M (TFLOPS/bandwidth ratio, perf verdict); production anchors (JCGT SDF-grid PT 9.7–17.3 ms on a 3090; CP2077 Overdrive ~8 fps native 1440p on a 4070) bracket "8–16 ms full PT at ~720p–1080p internal" as the honest tier. 4K60 therefore **requires** reconstruction; native 4K60 full PT has no supporting datapoint on this card class.
- **The 1M-mover TLAS refit is the #1 unmeasured risk.** No tier has a refit datapoint at 10K–50K dynamic instances/frame. It is measured FIRST in the agents rung, before any crowd work builds on it. Known mitigation shape if it fails: refit cadence stratified by distance.
- **The megakernel instance path shades flat today:** geometric normals only, `hit_uv = (0,0)`, single material (override-or-0), zero motion vectors, no weathering. Wave 1 adds per-triangle materials only; smooth normals + UVs + textures on instances (vertex-attribute fetch for BLAS hits) is its own milestone, and **textured-instancing quality is unproven until then**. The M1 image is deliberately flat-shaded multi-material.
- **Instanced geometry casts no shadows until `trace_shadow_instances` lands** (`trace_shadow` traverses only the soup BVH) — Wave 1 closes this; it is a correctness item, not polish ("no faking" includes shadows).
- **`VK_NV_cluster_acceleration_structure` has no released ash binding we have verified** — Tier 1 may start on raw FFI or a vendored extension binding; risk of API churn (the extension is new). Tier 2 KHR ray-query from Slang compute on RADV is likewise unproven in OUR toolchain (slangc → SPIR-V ray-query on RADV) — a spike rung gates it.
- **Reconstruction integration is unproven in our frame loop** (spectra-upscale/dlss exist as crates + kernels, never wired end-to-end here), and instance-path motion vectors are zero today — moving crowds cannot be reconstructed until that lands.
- **Non-NVIDIA stays a tier below on dense dynamic scenes** even when Tier 2 is complete: no CLAS means coarser LOD in the AS and costlier refits. This is the accepted industry-parallel position (UE5 proxies), stated honestly, not hidden.
- **`spectra-cluster` and `spectra-decimate` are pyo3-facing** — both need Rust-native exports before the cook can use them (mechanical, but unbudgeted work if forgotten).
- **Mesh lights × instancing:** `g_mesh_light_*` CDFs are built from soup geometry; emissive INSTANCED meshes won't be sampled as lights until the light path learns instances (later wave; M1 materials are non-emissive).
- The 780M Tier-3 floor renders M1 at **seconds per frame** (SDF M1 precedent: 0.39 s/frame after the SPIR-V cache) — fine for correctness gates, useless for interactivity; do not promise an interactive 780M game view from Tier 3.

---

## 9. Open Questions

- [ ] Tier 1 CLAS bindings: raw `ash` FFI vs vendored extension wrapper — decide at the Wave-2 spike (blocked on tomespensin access for any validation).
- [ ] Tier 2 mechanism: KHR ray-query from the compute megakernel (preserves wavefront architecture — the working assumption) vs a full RT pipeline port — decided by the slangc→SPIR-V ray-query spike on RADV (the 780M can host this spike today).
- [ ] Upscaler per tier: DLSS (Tier 1) is settled; FSR3 vs XeSS vs our own TAAU for Tier 2/3 — decide at the reconstruction rung with measured quality on the actual city content.
- [ ] Near-mover skinning: refit skinned BLAS per frame vs deformation in cluster templates (Tier 1 CLAS templates support this; Tier 2/3 need refit) — decide after the refit benchmark.
- [ ] Import quality gates' exact budgets (60K/200K starting points) — calibrate against real Forge output at the import-cook rung.

---

## 10. Build Ladder

Each rung is independently shippable, has one verifiable outcome, and the next rung builds on it. The ladder runs on the 780M (Tier 3) until L6.

| Rung | Name | One-line verifiable outcome |
|---|---|---|
| **L1** | **Mesh M1 — instanced city block** (the Wave-1 plan, fully specified) | the §2 command prints `instances solid : >= 10/12 … distinct transforms : 12/12 … materials : >= 10/12` and writes `ochroma_mesh_city_block.png` a human eyeballs: 12 solid, occluding, shadowed buildings from 3 shared BLAS prototypes |
| **L2** | **M2 Mega Geometry — tiered accel backend** | `tier_probe` prints the detected tier + extensions on each box; Tier-2 KHR ray-query spike renders the L1 block image-identical to Tier 3 (mean | Δ | < 2/255 printed); Tier-1 CLAS path builds on tomespensin or degrades per contract with the printed reason |
| **L3** | **LOD + visible set** | the L1 test at 5,000 instances with the `InstancedSelector`+governor prints `visible=<N≪5000> selected, frame N.N ms within budget B`, and a zoom-out swaps LODs with the per-LOD instance counts printed |
| **L4** | **Agents — movers + refit (MEASURE REFIT FIRST)** | rung OPENS with `refit_bench` printing `refit 50_000 instances: X.X ms` per tier BEFORE crowd work; then 10K visible animated movers render with per-frame transform refit and the bench line stays under the budget split |
| **L5** | **Reconstruction** | 1080p-internal → 4K-output frame of the L3 scene with PSNR vs native-4K printed ≥ gate, and instance motion vectors verified non-zero on a moving instance (printed max |mv|) |
| **L6** | **tomespensin validation** | on the 4070 Ti: the 100K-instance/1M-mover save at 4K output with reconstruction shows ≥ 60 fps in the window title for 30 s of camera flight; every projection in §8 replaced by a measured number in a committed report |

---

## 11. Out of Scope

- Building interiors, interior mapping, and the aperture/glass classification track (SDF-era work; revisit after textured instancing).
- Skinned-animation authoring, blend trees, facial animation (vox_render animation stack is a separate concern; L4 consumes baked poses).
- The Gaussian-splat runtime track (atoms remain appearance/scene data; `splats_to_lit_scene` stays a cinematic tool, not the building path).
- Multi-GPU, planetary-scale coordinates (NYC-extent f32 holds per the world-scale design).
- DLSS frame generation specifics (L5 covers super-resolution only; FG is a tomespensin-era follow-up).
- The 1M-agent simulation/ECS throughput (separate workstream; this design renders whatever the sim publishes).
- USD ingestion (interchange-only decision stands; import-cook consumes meshes, wherever they came from).

---

## 12. Related Plans / Designs

- Wave-1 plan (build this first): [`2026-06-12-mesh-render-wave1.md`](../plans/2026-06-12-mesh-render-wave1.md)
- Supersedes as render path: [SDF housing quality](./2026-06-12-sdf-housing-quality-design.md) (render goals only; its Forge-mass improvements survive as mesh improvements), the SDF scene-content roadmap's render rungs.
- Depends on: [SDF Pillar](./2026-06-10-sdf-pillar-design.md) (substrate, unchanged), the virtualized visible-set machinery (shipped), [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (one-renderer premise).
- Required before: the civitas game-view repoint (game renders through `pathtrace_mesh_scene_to_rgba`-descended frame path), Tier-1 work on tomespensin.

---

## IMPORTANT NOTES (real API signatures — implementers use these verbatim)

- **Run recipe (the only way GPU tests run on this box):** `SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release --lib <filter> -- --nocapture --test-threads=1`. The script exports `SLANG_DIR`/`LD_LIBRARY_PATH` (picks `~/slang-sdk`). Kernel dir resolves via `resolve_slang_kernel_dir()` = `SPECTRA_SLANG_DIR` env else `$CARGO_MANIFEST_DIR/../../../spectra/slang`.
- **Never run the full `cargo test -p vox_render --lib`** — the pre-existing `resident_gi_seam` radix-sort nondeterminism flake fails full runs; always use targeted filters.
- `pathtrace_mesh_to_rgba(positions: &[[f32;3]], normals: &[[f32;3]], uvs: &[[f32;2]], indices: &[[u32;3]], material_ids: &[u8], materials: &[PbrMaterial], eye: [f32;3], target: [f32;3], fov_y: f32, width: u32, height: u32, spp: u32, sun_dir: [f32;3]) -> Result<Vec<u8>, String>` — delegates to `pathtrace_mesh_textured_to_rgba(…, textures: &[TextureImage], …)` which delegates to `pathtrace_mesh_lit_to_rgba(…, rig: &LightRig)`. All `#[cfg(feature = "spectra-native")]` in `crates/vox_render/src/splat_backend.rs`.
- `PbrMaterial { base_color: [f32;3], roughness: f32, metallic: f32, emission_strength: f32, albedo_tex: i32, roughness_tex: i32, normal_tex: i32, uv_scale: [f32;2] }` (+ `Default`); packed by `pack_vulkan_mesh_material` into the **156-float Vulkan `MaterialData`** (`VULKAN_MATERIAL_FLOATS`); texture indices: albedo=a[28], roughness=a[29], normal=a[30], uv_scale=a[94..95].
- `Renderer<G: GpuBackend>` flow (exact order, landmines included): `VulkanSlangBackend::new(0)` → `RenderConfig::near_realtime(w, h)` + `config.slang_kernel_dir = resolve_slang_kernel_dir()` + `config.target_spp = spp` → `Renderer::new(gpu, config)` → `load_scene_state(scene)` → (`set_sky_gradient` and `set_texture_atlas` only AFTER `load_scene_state` — both silently no-op before) → `set_camera_view_matrix(cam.view_matrix)` → `set_view_proj(cam.view_matrix)` → `render() -> FrameOutput { beauty: Vec<f32> /* RGBA linear */, width, height }`. `renderer.timing_sink: pub Option<Vec<(String, f32)>>` — set `Some(Vec::new())` before render to record every kernel dispatch `(label, ms)`; Vulkan dispatch is fence-synchronous so wall-clock is real.
- `SceneState { pub geometry: GeometryLayer, pub materials: MaterialLayer, pub lights: LightLayer, pub sdf: SdfLayer, pub camera: CameraLayer, pub dirty: DirtyFlags, pub width, pub height, pub name }`; `SceneState::new(w, h)`; dirty marks `mark_geometry_changed()` / `mark_materials_changed()` / `mark_lights_changed()` / `mark_sdf_changed()` — the new layer adds `mark_mesh_instances_changed()`.
- `GeometryLayer` flat fields: `positions/normals/uvs: Vec<f32>`, `indices: Vec<u32>`, `material_ids: Vec<u32>`, `instance_transforms: Vec<f32>` (16/instance, row-major 4×4 — **translation in elements 12/13/14**, consumed CPU-side only today), counts kept in sync by the caller.
- `spectra_tlas::{TlasInstance::new(blas_id, instance_id), ::with_translation(..), ::pack_bytes() -> [u8;64], make_trs_transform(translate, rotate_y_rad, scale) -> [[f32;4];3], TlasBuilder::{push, update_transform(index, transform) -> bool, build(), packed_bytes()}, Accelerator::{add_blas(blas_id, aabb_min, aabb_max), add_instance, rebuild_if_dirty() -> bool, traverse_ray(origin, dir) -> Vec<usize>, FeedbackBuffer::record_miss(blas_id)}}` — CPU structures; in `spectra-renderer` the Accelerator is rebuilt in `load_scene_state` from `geom.instance_count`/`instance_transforms` (TlasInstance wants the upper 3 rows of the 4×4).
- `spectra_scene_upload`: `BLASEntry { vertex_count, triangle_count, bvh_node_count, aabb_min/max: [f32;3], vertices: Vec<f32> /* V×VERTEX_FLOATS */, triangles: Vec<u32> /* T×4 [v0,v1,v2,mat] */, bvh_nodes: Vec<f32> /* N×10 */, prim_indices: Vec<u32> }`, `BLASEntry::build(&LayerGeom)`; `build_median_bvh(vertices: &[f32], triangles: &[u32], vertex_stride: usize, max_leaf_prims: usize) -> BvhBuildResult { node_data, sorted_indices, num_nodes }`; `ContentCache::{new(), with_dir(PathBuf), read(u128) -> Option<BLASEntry>, write(u128, &BLASEntry)}` (blob magic `SPECTRA_C`, default dir `~/.cache/spectra/`); `xxhash128_file(&Path) -> Option<u128>`; `GpuScene::bind_to_map(&self, &mut BindingMap)`; `Uploader::upload_from_state(&SceneState, None, &mut gpu, None)`.
- Kernel-side instancing (ALL EXISTS, unwired): `instancing.slang` — `InstanceData` 24-float layout (§5), `load_instance(StructuredBuffer<float>, idx)`, `trace_instances` compute entry `[numthreads(64,1,1)]` consuming `g_ray_ox/oy/oz/dx/dy/dz`, `g_ray_alive`, `u_num_rays`, `g_instance_data`, `g_instance_inv_xform` (12 fl/inst), `g_blas_vertices` (9 fl/tri, **BVH-sorted**), `g_proto_bvh_nodes` (10 fl/node, prototype-relative children), `g_tlas_nodes` + `g_tlas_prim_order` + `u_tlas_node_count`, `u_total_bvh_nodes`, `u_instance_motion_blur` (set 0; `g_instance_transforms_t0/t1`/`g_ray_time_in` unread then) — writes `g_hit_instance` (−1 on miss, ALWAYS written), `g_hit_tri_in_blas` (prototype-local), `g_hit_t/u/v` only on improvement (reads `best_t = g_hit_t[ray]` initial → composes occlusion with the soup pass). `megakernel.slang` shade: `is_instance_hit = (g_hit_instance[ray] >= 0)` branch at ~2249 — geometric normal from `g_blas_vertices`, `hit_uv=(0,0)`, `tri_material_id = material_override >= 0 ? override : 0` (Wave 1 replaces the `: 0` with `g_blas_tri_material`). `bvh_traverse.slang::trace_shadow` reads `g_shadow_ox..dz/max_t/u_num_shadow_rays`, writes `g_shadow_vis[ray] = occluded ? 0 : 1` — soup only.
- `KernelSet::new` registration shape: `(names::CONST, "slang_file_stem", "entry_fn")` rows in `core_kernels`; dispatch via `self.dispatch_kernel("label", id, pixel_threads, &bindings)`; uniforms via `bindings.set_uniform_i32("u_name", v)`. Declared-but-unbound buffers are tolerated by the Vulkan backend (megakernel declares `g_instance_data` today and every existing test passes without binding it).
- `RayQueue` (spectra-integrator/src/ray_queue.rs): `hit_instance` is `alloc_u32(n)` + `fill_i32(&h, -1)`; bound in `bind_to_map` as `g_hit_instance`; **`g_hit_tri_in_blas` does NOT exist there yet** — Wave 1 adds it with the identical pattern.
- Cooked asset (game side, civitas `src/asset/mod.rs`): `ReadyAssetPayload { version, asset_id: String, asset_dir: Option<PathBuf>, bounds, materials: Vec<ReadyAssetPbrMaterial>, surfaces, snap_points, mesh: Option<ReadyAssetMesh>, sdf: Option<ReadyAssetSdfVolume>, …, atoms }` where `ReadyAssetMesh { positions: Vec<[f32;3]>, normals: Vec<[f32;3]>, uvs: Vec<[f32;2]>, indices: Vec<[u32;3]>, material_ids: Vec<u8>, exterior_surface_ids: Vec<String> }` — the per-type prototype source.
- Visible-set machinery (vox_render/src/atom_instances.rs): `AtomInstance::new(asset: u32, instance_id: u32, position: [f32;3], rotation: [f32;4])` (accessors `.asset()`, `.instance_id()`); `InstancedSelector::{new(Arc<AssetAtomLibrary>), set_instances(&[AtomInstance]), select(…)}`; `FrameBudgetGovernor::{new(target_ms: f32, initial: usize, min: usize, max: usize), budget() -> usize, ema_ms() -> f32, update(frame_ms: f32)}`.
- Perf-truth pins: first `Renderer` per process pays ~0.3–9 s of slangc compile depending on the SPIR-V disk cache (`~/.cache/spectra-spirv`, on spectra `vulkan-fallback-backend`); the 780M→4070 Ti multiplier is **5–7×**; `SPECTRA_RENDER_PROFILE=1` prints render() phase ms.
