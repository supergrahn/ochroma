# Mesh Render Wave 1 (Mesh M1) — Instanced City Block Through the Path Tracer Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render an instanced city block of building meshes through Spectra's path tracer on the 780M (Tier-3 software BVH/TLAS): multiple building instances drawn from a small prototype TYPE library, **one shared BLAS per type**, per-instance world transforms, per-instance/per-type PBR materials, solid surfaces, correct inter-building occlusion, and real instance-cast sun shadows — the mesh twin of the proven SDF M1 city block.

**Done When:** Running

```
SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release \
  --lib mesh_city_block_renders_instanced_buildings -- --nocapture --test-threads=1
```

passes and prints (real measured values, shape exact):

```
[mesh_block] INSTANCED MESH CITY BLOCK (3 BLAS prototypes, 12 instances, Tier-3 software TLAS):
  instances placed    : 12 (3 prototypes)
  instances solid     : <S>/12 (per-instance span-fill >= 0.80)
  mean span-fill      : <F>
  distinct transforms : <T>/12 (boxes lit, pairwise centers >= 8 px apart)
  materials           : <M>/12 (wall vs roof colour split > 0.12)
  occlusion           : overlap dist to front wall = <a>, to rear wall = <b> (front wins)
  total lit coverage  : <C> of frame
  seconds/frame       : <X>s
[mesh_block] wrote <tmpdir>/ochroma_mesh_city_block.png
```

with gates **S ≥ 10, F ≥ 0.80, T = 12, M ≥ 10, a < b, C ≥ 0.15** asserted in the test, and `ochroma_mesh_city_block.png` showing — to a human eyeball — 12 solid flat-shaded buildings of 3 visibly different types at distinct positions and rotations on a grey ground plane, front buildings occluding rear ones, with sun shadows on the ground. NO "tests pass" — the printed counts and the PNG are the gate.

**Architecture:** The kernel-side instancing machinery already exists in Spectra (`instancing.slang::trace_instances` — TLAS traversal over per-prototype BLASes; the megakernel already shades instance hits) but **nothing in the Rust runtime packs, binds, or dispatches it**. This wave adds: a `MeshInstanceLayer` scene layer + exact-kernel-layout packing (per-type BLAS via the existing `build_median_bvh`, 24-float `InstanceData` records, a 10-float-node software TLAS over instance world-AABBs), kernel registration + per-bounce dispatch composing occlusion with the existing soup pass, per-triangle materials and an any-hit shadow pass for instance hits, and the public `pathtrace_mesh_scene_to_rgba` entry point in vox_render mirroring the proven `pathtrace_sdf_scene_to_rgba` shape. M1 is **flat-shaded** on the instance path (geometric normals, no textures — the kernel's instance branch has no UVs yet); per-type colour identity comes from per-triangle PBR materials. No RTXMG anywhere in this wave.

**Design Document:** `docs/superpowers/specs/2026-06-12-mesh-megageometry-render-design.md`
**Tech Stack:** Rust (workspace toolchains as-is), Slang→SPIR-V via slangc (`~/slang-sdk` 2025.24.3), Vulkan/RADV on the AMD 780M, glam 0.29, spectra path deps (`crates/vox_render/Cargo.toml` → `../../../spectra/rust/*`).
**Build:** `scripts/build-spectra-native.sh <cargo args>` for anything touching `spectra-native`; plain `cargo test -p spectra-scene-upload` / `-p spectra-scene-state` for the CPU-side spectra tasks (run from `~/src/spectra/rust`).

---

## BRANCH HYGIENE (do this before Task 1)

- **ochroma:** create `feat/mesh-render-m1` off **`master`** (verified: master already carries `pathtrace_mesh_to_rgba` and `pathtrace_sdf_scene_to_rgba`). Do NOT branch off `blitz/day1-foundation` and do NOT touch its uncommitted work.
- **spectra:** create `feat/mesh-render-m1` off **`vulkan-fallback-backend`** (verified: that branch carries SDF M0–M2 *and* the SPIR-V disk cache + interval-march perf commits — warm renderer frames are sub-second). Check it out **in `~/src/spectra` directly** — ochroma's vox_render path-deps resolve against whatever `~/src/spectra` has checked out. If `~/src/spectra` must stay parked on `feat/sdf-atom-cellgrid` for another workflow, use a worktree + a LOCAL (uncommitted) repoint of `crates/vox_render/Cargo.toml`, per the worktree-gotcha memory — never commit the repoint.
- Commit per task, on these feature branches only.

---

## IMPORTANT NOTES

- **Run recipe for every GPU test** (vox_render, from `~/src/ochroma`): `SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release --lib <filter> -- --nocapture --test-threads=1`. The script exports `SLANG_DIR`/`LD_LIBRARY_PATH`. Kernel dir: `resolve_slang_kernel_dir()` = `SPECTRA_SLANG_DIR` env override, else `$CARGO_MANIFEST_DIR/../../../spectra/slang`.
- **Never run a full `cargo test -p vox_render --lib`** — the pre-existing `resident_gi_seam` radix-sort flake fails full runs. Targeted filters only. Line numbers in this plan are landmarks — **locate code by function name, never by line**.
- **Existing signatures (code-verified 2026-06-12; these are law):**
  - `pathtrace_mesh_lit_to_rgba(positions: &[[f32;3]], normals: &[[f32;3]], uvs: &[[f32;2]], indices: &[[u32;3]], material_ids: &[u8], materials: &[PbrMaterial], textures: &[TextureImage], eye: [f32;3], target: [f32;3], fov_y: f32, width: u32, height: u32, spp: u32, rig: &LightRig) -> Result<Vec<u8>, String>` — the scene/lights/camera/renderer packing in its body is the template for Task 5; copy its structure, do not invent a new one.
  - `pathtrace_sdf_scene_to_rgba(volumes, instances, eye, target, fov_y, width, height, spp, rig)` — the multi-instance entry-point shape Task 5 mirrors (validation-error style: explicit `Err(format!(...))`, never silent drops).
  - `PbrMaterial { base_color: [f32;3], roughness: f32, metallic: f32, emission_strength: f32, albedo_tex: i32, roughness_tex: i32, normal_tex: i32, uv_scale: [f32;2] }` + `Default`; packed by the existing `pack_vulkan_mesh_material(m: PbrMaterial) -> [f32; VULKAN_MATERIAL_FLOATS /* 156 */]`. `LightRig { sun_dir, sun_color, sun_intensity, sky_intensity, camera_fill, rim_fill, sky_dome_intensity, sky_dome_zenith, sky_dome_horizon }` + `Default`.
  - Renderer flow (ORDER MATTERS): `VulkanSlangBackend::new(0)` → `RenderConfig::near_realtime(w,h)`; `config.slang_kernel_dir = resolve_slang_kernel_dir()`; `config.target_spp = spp` → `Renderer::new(gpu, config)` → `load_scene_state(scene)` → `set_sky_gradient(zenith, horizon, intensity)` (AFTER load — no-ops before) → `set_camera_view_matrix(cam.view_matrix)` → `set_view_proj(cam.view_matrix)` → `render() -> FrameOutput { beauty: Vec<f32> /* linear RGBA */, width, height }`. `renderer.timing_sink: pub Option<Vec<(String, f32)>>` — set `Some(Vec::new())` to record every dispatch `(label, ms)`.
  - `SceneState { pub geometry: GeometryLayer, pub materials: MaterialLayer, pub lights: LightLayer, pub sdf: SdfLayer, pub camera: CameraLayer, pub dirty: DirtyFlags, pub width, pub height, pub name }`; `SceneState::new(w,h)`; `camera_layer(view: [f32;16], fov_y, w, h)` from `crate::splat_convert`; `pack_vulkan_directional_light(dir, color, intensity)`; `MaterialLayer { params, spectral_spd: Default::default(), material_count }`.
  - `build_median_bvh(vertices: &[f32], triangles: &[u32] /* 4 u32/tri: v0,v1,v2,mat */, vertex_stride: usize, max_leaf_prims: usize) -> BvhBuildResult { node_data: Vec<f32> /* 10 fl/node */, sorted_indices: Vec<u32>, num_nodes: u32 }` (`spectra-scene-upload/src/bvh_build.rs`). Node layout: `[min.xyz, max.xyz, left, right, prim_start, prim_count]`, ints stored as f32 BITS (`f32::from_bits(i as u32)` to pack, `asint(asuint(x))` kernel-side); leaf when `prim_count > 0`; child indices are **array-relative (root = 0)**.
  - `GpuScene::bind_to_map(&self, map: &mut spectra_gpu::BindingMap)` (`gpu_scene.rs`) — the ONE place buffers get bound by name; `Uploader::upload_from_state(&SceneState, None, &mut gpu, None)` is the upload entry; SDF layer packing in `uploader.rs` is the gate-pattern to copy (`if !scene.mesh_instances.is_empty() { ... }`).
  - `RayQueue` (`spectra-integrator/src/ray_queue.rs`): `hit_instance` is `backend.alloc_u32(n)` + `backend.fill_i32(&h, -1)`, bound as `"g_hit_instance"` in `bind_to_map` — **copy this exact pattern for the new `hit_tri_in_blas`**.
  - `KernelSet::new` (`kernel_set.rs`): kernels register as `(names::CONST, "slang_file_stem", "entry_fn")` rows in the `core_kernels` table; dispatch is `self.dispatch_kernel("label", id, pixel_threads, &bindings)?` in `renderer.rs`; uniforms via `bindings.set_uniform_i32("u_name", v)`.
- **Kernel contracts (verbatim from `instancing.slang` / `megakernel.slang` — deviation = wrong image):**
  - `InstanceData` = 24 floats: `[0–11]` row-major 3×4 object→world; `[12–14]`/`[15–17]` world AABB min/max; `[18]` `blas_start` (first triangle of the prototype in the BVH-REORDERED flat BLAS buffer, int bits); `[19]` `blas_count`; `[20]` `material_override` (−1 = per-triangle); `[21]` `bvh_start` (ABSOLUTE node offset of the prototype root in `g_proto_bvh_nodes`); `[22–23]` pad.
  - `g_blas_vertices` = 9 floats/tri (v0,v1,v2 object-space positions) and **MUST be pre-permuted by `BvhBuildResult.sorted_indices`** — the instance path reads triangle `(blas_start + prim_start + k)` DIRECTLY, with **no `g_prim_indices` indirection**. The new `g_blas_tri_material` (1 u32/tri) must use the SAME permutation.
  - `g_proto_bvh_nodes`: concatenated per-prototype node arrays; children prototype-relative, so concatenation needs NO index fixup — only record each prototype's `bvh_start` offset and leaf `prim_start` stays prototype-local.
  - `trace_instances` consumes `g_ray_ox/oy/oz/dx/dy/dz`, `g_ray_alive`, `u_num_rays` (all already bound by `RayQueue::bind_to_map`), plus `g_instance_data`, `g_instance_inv_xform` (12 fl/inst world→object), `g_blas_vertices`, `g_proto_bvh_nodes` + `u_total_bvh_nodes`, `g_tlas_nodes` + `g_tlas_prim_order` + `u_tlas_node_count`, `u_num_instances`, `u_instance_motion_blur` (**set 0**; `g_instance_transforms_t0/t1` and `g_ray_time_in` are then unread — declared-but-unbound buffers are tolerated by the Vulkan backend: the megakernel declares `g_instance_data` today and every existing test passes without it bound). Writes: `g_hit_instance` (−1 on miss, ALWAYS written), `g_hit_tri_in_blas` (prototype-local), and `g_hit_t/u/v` only on improvement — it initializes `best_t = g_hit_t[ray_idx]`, so dispatching it AFTER `trace_bvh` composes soup↔instance occlusion automatically.
  - Megakernel shade: `is_instance_hit = (g_hit_instance[ray_idx] >= 0)`; the instance branch (find `if (is_instance_hit)` in `shade_and_bounce`) uses the geometric normal from `g_blas_vertices`, `hit_uv = (0,0)`, and `tri_material_id = (material_override_inst >= 0) ? material_override_inst : 0` — Task 3 replaces ONLY the `: 0` fallback.
  - `bvh_traverse.slang::trace_shadow` reads `g_shadow_ox/oy/oz/dx/dy/dz`, `g_shadow_max_t`, `u_num_shadow_rays` and writes `g_shadow_vis[i] = occluded ? 0 : 1` — Task 4's new entry uses the SAME buffer names (re-declared in instancing.slang; binding is by name per dispatch).
- **NEW API contracts (implement EXACTLY — design §6):** `MeshPrototypeData { positions: Vec<f32>, indices: Vec<u32>, material_ids: Vec<u32>, vertex_count, triangle_count }`, `MeshInstanceRecord { prototype_index: u32, transform: [[f32;4];3], material_override: i32 }`, `MeshInstanceLayer::{empty(), from_parts(prototypes, instances), is_empty()}`, `SceneState::mark_mesh_instances_changed()`; `pack_mesh_instances(layer: &MeshInstanceLayer) -> PackedMeshInstances` (fields per design §5); `build_aabb_tlas(world_aabbs: &[([f32;3],[f32;3])], max_leaf_prims: usize) -> BvhBuildResult`; kernel name consts `TRACE_INSTANCES: &str = "trace_instances"`, `SHADOW_TRACE_INSTANCES: &str = "trace_shadow_instances"`; vox_render `MeshPrototype { positions: Vec<[f32;3]>, indices: Vec<[u32;3]>, material_ids: Vec<u8> }`, `MeshSceneInstance { prototype_index: u32, position: [f32;3], rotation_xyzw: [f32;4], uniform_scale: f32, material_override: i32 }`, `GroundPlane { y: f32, half_extent: f32, material_id: u8 }`, `pathtrace_mesh_scene_to_rgba(prototypes: &[MeshPrototype], instances: &[MeshSceneInstance], materials: &[PbrMaterial], ground: Option<GroundPlane>, eye: [f32;3], target: [f32;3], fov_y: f32, width: u32, height: u32, spp: u32, rig: &LightRig) -> Result<Vec<u8>, String>`.
- **M1 scope pins:** materials with any `*_tex >= 0` on the instanced path are an explicit `Err` (textured instancing is a later milestone — error, never silently drop, mirroring the `SdfApertureRect` precedent). Instance materials must be non-emissive (mesh-light CDFs don't know instances yet). The instance path is flat-shaded (geometric normals) — expected, not a bug.
- **Transform math:** object→world 3×4 from `(position, rotation_xyzw, uniform_scale)` via `glam::Affine3A::from_scale_rotation_translation(Vec3::splat(s), Quat::from_array(q).normalize(), Vec3::from(p))`; rows = the affine's matrix3 rows with translation as the 4th column (ROW-major 3×4); inverse via `.inverse()` packed the same way. World AABB = transform of all 8 prototype-local AABB corners (copy the corner loop from `pathtrace_sdf_scene_perf`).
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task. Every test asserts a real computed value and PRINTS it (`assert!(x.is_some())`-style tests are forbidden). Zero-triangle scenes hang the GPU — the soup must always contain ≥ 1 triangle (ground quad or the SDF-path's degenerate sentinel).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `~/src/spectra/rust/spectra-scene-state/src/layers.rs` | `MeshPrototypeData`, `MeshInstanceRecord`, `MeshInstanceLayer` |
| Modify | `~/src/spectra/rust/spectra-scene-state/src/lib.rs` | `SceneState.mesh_instances` field, `mark_mesh_instances_changed()`, re-exports |
| Modify | `~/src/spectra/rust/spectra-scene-state/src/dirty.rs` | `mesh_instances` dirty flag (copy the sdf flag pattern) |
| Create | `~/src/spectra/rust/spectra-scene-upload/src/mesh_instances.rs` | `pack_mesh_instances` + `PackedMeshInstances` + pack tests |
| Modify | `~/src/spectra/rust/spectra-scene-upload/src/bvh_build.rs` | `build_aabb_tlas` over instance world-AABBs |
| Modify | `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` | pack+upload mesh-instance buffers when layer non-empty |
| Modify | `~/src/spectra/rust/spectra-scene-upload/src/gpu_scene.rs` | buffer handles + `bind_to_map` names + `mesh_instance_count()` |
| Modify | `~/src/spectra/rust/spectra-scene-upload/src/lib.rs` | `pub mod mesh_instances;` |
| Modify | `~/src/spectra/rust/spectra-integrator/src/ray_queue.rs` | `hit_tri_in_blas` alloc (−1 fill) + `g_hit_tri_in_blas` bind |
| Modify | `~/src/spectra/rust/spectra-renderer/src/kernel_set.rs` | register `TRACE_INSTANCES` + `SHADOW_TRACE_INSTANCES` |
| Modify | `~/src/spectra/rust/spectra-renderer/src/renderer.rs` | dispatch both kernels (bounce loop + shadow site) + uniforms |
| Modify | `~/src/spectra/slang/megakernel.slang` | declare `g_blas_tri_material`; per-triangle material fallback in the instance branch |
| Modify | `~/src/spectra/slang/instancing.slang` | new `trace_shadow_instances` entry point |
| Modify | `~/src/ochroma/crates/vox_render/src/splat_backend.rs` | `MeshPrototype`/`MeshSceneInstance`/`GroundPlane`, `pathtrace_mesh_scene_to_rgba`, all GPU tests (in-module `#[cfg(test)]`, house style) |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Per-type BLAS shared by instances | pack test: two same-type instances carry IDENTICAL `blas_start/blas_count/bvh_start` bits while transforms differ; prototype 1's `bvh_start == prototype 0's num_nodes` — values printed | `assert!(packed.instance_data.len() > 0)` |
| Software TLAS over instances | `build_aabb_tlas` test: root node AABB == exact union of 12 inputs, leaf `prim_count` sum == 12, `sorted_indices` is a permutation — all printed | `assert!(result.num_nodes > 0)` |
| Instanced trace dispatched + hits | GPU probe: `timing_sink` contains `trace_instances`, wall-region ΔLuma vs no-instance render > 0.10 — printed | "kernel registered" without a render |
| Per-triangle materials on instance hits | GPU probe: red-wall region mean R > mean B + 0.10 with walls mat-red / top mat-blue in ONE prototype | `assert!(rgba.iter().any(\|p\| *p > 0))` |
| Instance-cast shadows | shadow test: ground luma under slab < 0.6 × open-ground luma, both printed | `assert!(shadow_kernel_compiles)` |
| Public scene entry point | two-instance test: left-box mean R > 2× mean B AND right-box mean B > 2× mean R, printed | `assert!(result.is_ok())` |
| The M1 milestone | `mesh_city_block_renders_instanced_buildings` prints the full `[mesh_block]` gate report + PNG | any test not printing per-instance numbers |

---

## Task 1: `MeshInstanceLayer` + exact-kernel-layout packing, wired into upload + bind map

**Files:**
- Modify: `~/src/spectra/rust/spectra-scene-state/src/layers.rs`, `src/lib.rs`, `src/dirty.rs`
- Create: `~/src/spectra/rust/spectra-scene-upload/src/mesh_instances.rs`
- Modify: `~/src/spectra/rust/spectra-scene-upload/src/{lib.rs, uploader.rs, gpu_scene.rs}`
- Test: in-module tests in `mesh_instances.rs`

**Acceptance:** `cd ~/src/spectra/rust && cargo test -p spectra-scene-upload pack_mesh_instances -- --nocapture` → prints `[pack] protoB.bvh_start=<NA> (== protoA num_nodes) | inst1 blas_start/count = 0/12 (shared with inst0) | inst2 mat_override=4 | inst1.tx=12.5 | inv_check=<e < 1e-5> | tri_mat permuted OK` with every value real and non-zero where stated.

**Wiring requirement:** `pack_mesh_instances` must be called from `Uploader::upload_from_state` (gated `if !scene.mesh_instances.is_empty()`), its buffers uploaded and registered in `GpuScene`, and `GpuScene::bind_to_map` must bind `g_instance_data`, `g_instance_inv_xform`, `g_blas_vertices`, `g_blas_tri_material`, `g_proto_bvh_nodes` (TLAS buffers land in Task 2) — in THIS task. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — real geometry, real packed-bit assertions

```rust
// spectra-scene-upload/src/mesh_instances.rs  (tests module)
#[test]
fn pack_mesh_instances_two_prototypes() {
    use spectra_scene_state::{MeshInstanceLayer, MeshInstanceRecord, MeshPrototypeData};
    let cube = unit_cube_proto();          // helper: 8 verts, 12 tris, walls mat 1, top 2 tris mat 2
    let quad = ground_quad_proto(4.0, 3);  // helper: 4 verts, 2 tris, mat 3
    let xf = |tx: f32, yaw: f32, s: f32| trs_3x4([tx, 0.0, 0.0], yaw, s); // helper, row-major 3x4
    let layer = MeshInstanceLayer::from_parts(
        vec![cube, quad],
        vec![
            MeshInstanceRecord { prototype_index: 0, transform: xf(0.0, 0.0, 1.0), material_override: -1 },
            MeshInstanceRecord { prototype_index: 0, transform: xf(12.5, std::f32::consts::FRAC_PI_2, 1.0), material_override: -1 },
            MeshInstanceRecord { prototype_index: 1, transform: xf(0.0, 0.0, 2.0), material_override: 4 },
        ],
    );
    let p = pack_mesh_instances(&layer);
    assert_eq!(p.instance_count, 3);
    assert_eq!(p.instance_data.len(), 3 * 24);
    let bits = |f: f32| f32::to_bits(f) as i32;
    // shared BLAS: instances 0 and 1 reference the SAME prototype range
    assert_eq!(bits(p.instance_data[18]), 0, "inst0 blas_start");
    assert_eq!(bits(p.instance_data[19]), 12, "inst0 blas_count");
    assert_eq!(p.instance_data[24 + 18], p.instance_data[18], "inst1 shares blas_start");
    // prototype 1 starts where prototype 0's triangles/nodes end
    let proto_a_nodes = /* recompute: build_median_bvh on the cube */ expected_cube_nodes();
    assert_eq!(bits(p.instance_data[2 * 24 + 21]), proto_a_nodes as i32, "protoB.bvh_start");
    assert_eq!(bits(p.instance_data[2 * 24 + 18]), 12, "protoB blas_start after 12 cube tris");
    assert_eq!(bits(p.instance_data[2 * 24 + 20]), 4, "inst2 material_override");
    assert!((p.instance_data[24 + 3] - 12.5).abs() < 1e-6, "inst1 tx in transform row 0 col 3");
    // inverse really inverts: max |(inv ∘ fwd) - I| over the 12 elements
    let e = compose_3x4_max_dev(&p.instance_data[24..36], &p.instance_inv_xform[12..24]);
    assert!(e < 1e-5, "inv_check {e}");
    // tri materials follow the BVH permutation: every cube triangle's material
    // survives the reorder (multiset equal: ten 1s + two 2s in the first 12 slots)
    let cube_mats: Vec<u32> = p.blas_tri_material[0..12].to_vec();
    assert_eq!(cube_mats.iter().filter(|&&m| m == 2).count(), 2, "2 roof tris");
    assert_eq!(p.blas_vertices.len(), (12 + 2) * 9);
    println!("[pack] protoB.bvh_start={proto_a_nodes} | inst1 blas_start/count = 0/12 (shared with inst0) | inst2 mat_override=4 | inst1.tx=12.5 | inv_check={e:.7} | tri_mat permuted OK");
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd ~/src/spectra/rust && cargo test -p spectra-scene-upload pack_mesh_instances 2>&1 | tail -5
```

Expected: FAIL — `error[E0432]: unresolved import spectra_scene_state::MeshInstanceLayer` (the types must not exist yet).

- [ ] **Step 3: Implement** — full logic, no stubs

In `spectra-scene-state`: the three structs + `MeshInstanceLayer::{empty, from_parts, is_empty}` (layers.rs, doc-commented like `GeometryLayer`); `pub mesh_instances: MeshInstanceLayer` on `SceneState` (initialized `empty()` in `new`), `mark_mesh_instances_changed()` + a dirty flag following the `sdf` flag in dirty.rs exactly.

In `mesh_instances.rs`: for each prototype — pack triangles to the 4-u32 form (`[v0,v1,v2,mat]`, mat from `material_ids` or 0), `build_median_bvh(&proto.positions, &tris, 3, 4)`, then emit `g_blas_vertices`/`g_blas_tri_material` **in `sorted_indices` order**, append `node_data` recording `bvh_start = nodes_so_far`, `blas_start = tris_so_far`; record each prototype's local AABB. For each instance — transform the 8 local-AABB corners for world AABB; pack the 24 floats (ints as `f32::from_bits`); pack the 12-float affine inverse. Return `PackedMeshInstances` (TLAS fields empty until Task 2 — `tlas_nodes: vec![]`, `tlas_node_count: 0`; document that Task 2 fills them).

- [ ] **Step 4: Wire at exact callsite**

In `Uploader::upload_from_state` (locate the SDF-layer gate, mirror it):

```rust
// After the SDF layer block:
if !scene.mesh_instances.is_empty() {
    let packed = crate::mesh_instances::pack_mesh_instances(&scene.mesh_instances);
    gpu_scene.mesh_instance_data = Some(gpu.upload_f32(&packed.instance_data)?);
    gpu_scene.mesh_instance_inv_xform = Some(gpu.upload_f32(&packed.instance_inv_xform)?);
    gpu_scene.mesh_blas_vertices = Some(gpu.upload_f32(&packed.blas_vertices)?);
    gpu_scene.mesh_blas_tri_material = Some(gpu.upload_u32(&packed.blas_tri_material)?);
    gpu_scene.mesh_proto_bvh_nodes = Some(gpu.upload_f32(&packed.proto_bvh_nodes)?);
    gpu_scene.mesh_instance_count = packed.instance_count;
    gpu_scene.mesh_total_bvh_nodes = packed.total_bvh_nodes;
}
```

(match the file's actual upload-call idioms and error plumbing); in `GpuScene::bind_to_map`, bind each present handle by its `g_*` name following the `g_sdf_*` rows.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cd ~/src/spectra/rust && cargo test -p spectra-scene-upload pack_mesh_instances -- --nocapture
```

Expected: PASS with the `[pack]` line showing real node counts and `inv_check` ~1e-7 (not 0.0000000 from an identity-only stub — the test's instance 1 is rotated, so a wrong inverse fails loudly).

- [ ] **Step 6: Commit**

```bash
cd ~/src/spectra && git add rust/spectra-scene-state rust/spectra-scene-upload && \
git commit -m "feat(scene): MeshInstanceLayer + prototype-BLAS/instance packing bound for trace_instances"
```

---

## Task 2: Software TLAS over instance world-AABBs (`build_aabb_tlas`), packed + bound

**Files:**
- Modify: `~/src/spectra/rust/spectra-scene-upload/src/bvh_build.rs` (new fn + tests)
- Modify: `~/src/spectra/rust/spectra-scene-upload/src/{mesh_instances.rs, uploader.rs, gpu_scene.rs}`

**Acceptance:** `cd ~/src/spectra/rust && cargo test -p spectra-scene-upload build_aabb_tlas -- --nocapture` → prints `[tlas] nodes=<N>=instance leaves 12 | root=(-1.0,0.0,-1.0)..(33.0,9.0,17.0) exact union | prim_order is a permutation of 0..12` with the root min/max equal to the hand-computed union of the 12 input boxes (exact float equality — it's mins/maxes, no arithmetic).

**Wiring requirement:** `build_aabb_tlas` must be called inside `pack_mesh_instances` (filling `tlas_nodes`/`tlas_prim_order`/`tlas_node_count`), uploaded in `Uploader::upload_from_state` and bound as `g_tlas_nodes`/`g_tlas_prim_order` in `GpuScene::bind_to_map` — in THIS task.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn build_aabb_tlas_grid_of_12() {
    // 4x3 grid of 8m-spaced 6x9x6 boxes starting at (-1,0,-1)
    let boxes: Vec<([f32;3],[f32;3])> = (0..12).map(|i| {
        let (gx, gz) = ((i % 4) as f32, (i / 4) as f32);
        ([gx*8.0 - 1.0, 0.0, gz*8.0 - 1.0], [gx*8.0 + 5.0, 9.0, gz*8.0 + 5.0])
    }).collect();
    let r = build_aabb_tlas(&boxes, 1);
    // root AABB == exact union
    assert_eq!(&r.node_data[0..3], &[-1.0, 0.0, -1.0]);
    assert_eq!(&r.node_data[3..6], &[29.0, 9.0, 21.0]);
    // every input box appears in exactly one leaf
    let mut leaf_prims = 0u32;
    for n in 0..(r.num_nodes as usize) {
        let pc = f32::to_bits(r.node_data[n*10 + 9]) as i32;
        if pc > 0 { leaf_prims += pc as u32; }
    }
    assert_eq!(leaf_prims, 12);
    let mut seen: Vec<u32> = r.sorted_indices.clone(); seen.sort();
    assert_eq!(seen, (0..12u32).collect::<Vec<_>>(), "prim order is a permutation");
    println!("[tlas] nodes={}=instance leaves 12 | root=({},{},{})..({},{},{}) exact union | prim_order is a permutation of 0..12",
        r.num_nodes, r.node_data[0], r.node_data[1], r.node_data[2], r.node_data[3], r.node_data[4], r.node_data[5]);
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cd ~/src/spectra/rust && cargo test -p spectra-scene-upload build_aabb_tlas 2>&1 | tail -5
```

Expected: FAIL — `cannot find function build_aabb_tlas`.

- [ ] **Step 3: Implement** — median-split over AABB centroids, SAME 10-float node format and int-bit packing as `build_median_bvh` (reuse its internal node-emission helpers; leaf `prim_start` indexes `sorted_indices`, which the kernel reads through `g_tlas_prim_order` → instance index). No stubs.

- [ ] **Step 4: Wire at exact callsite** — in `pack_mesh_instances`, after the instance loop: `let tlas = build_aabb_tlas(&world_aabbs, 1); packed.tlas_nodes = tlas.node_data; packed.tlas_prim_order = tlas.sorted_indices; packed.tlas_node_count = tlas.num_nodes;` — then the Task-1 uploader block uploads + `bind_to_map` binds both (`g_tlas_nodes`, `g_tlas_prim_order`) and `gpu_scene.mesh_tlas_node_count` is stored.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
cd ~/src/spectra/rust && cargo test -p spectra-scene-upload build_aabb_tlas -- --nocapture && \
cargo test -p spectra-scene-upload pack_mesh_instances -- --nocapture
```

Expected: both PASS; the pack test now also prints a non-zero `tlas_node_count` (extend its print).

- [ ] **Step 6: Commit**

```bash
cd ~/src/spectra && git add rust/spectra-scene-upload && \
git commit -m "feat(scene-upload): software TLAS over instance world-AABBs (g_tlas_nodes/g_tlas_prim_order)"
```

---

## Task 3: Dispatch `trace_instances` + `g_hit_tri_in_blas` + per-triangle instance materials — GPU-verified

**Files:**
- Modify: `~/src/spectra/rust/spectra-renderer/src/kernel_set.rs` (register), `src/renderer.rs` (dispatch + uniforms)
- Modify: `~/src/spectra/rust/spectra-integrator/src/ray_queue.rs` (`hit_tri_in_blas`)
- Modify: `~/src/spectra/slang/megakernel.slang` (`g_blas_tri_material` declaration + fallback)
- Test: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` tests module (`mesh_instance_trace_probe`)

**Acceptance:** `SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release --lib mesh_instance_trace_probe -- --nocapture --test-threads=1` → prints `[inst_probe] trace_instances dispatches=<N>=1 | wall mean rgb=(<r>,<g>,<b>) r-b=<d>=0.10 | delta vs no-instance render = <e>=0.10` with N ≥ 1, r − b ≥ 0.10 (the cube's red walls shaded through the per-triangle material path) and e ≥ 0.10 (removing the instance changes the pixels — the hit really came from `trace_instances`).

**Wiring requirement:** `TRACE_INSTANCES` registered in `KernelSet::new` `core_kernels` as `(TRACE_INSTANCES, "instancing", "trace_instances")`; dispatched in `renderer.rs` **immediately after every `BVH_TRAVERSE` dispatch in the bounce loop** (locate by the dispatch call), gated on `gs.mesh_instance_count() > 0`, with `u_num_instances`/`u_total_bvh_nodes`/`u_tlas_node_count`/`u_instance_motion_blur=0` set on the SAME bindings map; `hit_tri_in_blas` allocated (−1 fill) + bound in `RayQueue`. Empty bodies = **task failure**.

- [ ] **Step 1: Write the failing test** (in vox_render's `#[cfg(test)]` module; builds `SceneState` directly — the public entry point arrives in Task 5)

```rust
#[test]
#[cfg(feature = "spectra-native")]
fn mesh_instance_trace_probe() {
    // helper shared with Task 4's test: ground-quad soup + N instances scene
    let (mut scene, cam) = mesh_probe_scene(/*with_cube=*/true);   // cube: walls mat1 RED, top mat2 BLUE
    let rgba_with = render_scene_collecting(&mut scene, &cam, |sink| {
        let n = sink.iter().filter(|(l, _)| l.contains("trace_instances")).count();
        assert!(n >= 1, "trace_instances never dispatched (sink: {:?})", sink);
        println!("[inst_probe] trace_instances dispatches={n}");
    });
    let (mut scene2, _) = mesh_probe_scene(false);
    let rgba_without = render_scene_collecting(&mut scene2, &cam, |_| {});
    let (r, g, b) = region_mean_rgb(&rgba_with, WALL_REGION);   // centre-screen wall pixels
    let delta = region_mean_luma_diff(&rgba_with, &rgba_without, WALL_REGION);
    println!("[inst_probe] wall mean rgb=({r:.3},{g:.3},{b:.3}) r-b={:.3} | delta vs no-instance render = {delta:.3}", r - b);
    assert!(r - b >= 0.10, "red walls must dominate: r={r:.3} b={b:.3}");
    assert!(delta >= 0.10, "instance must visibly change the frame, delta={delta:.3}");
}
```

(`render_scene_collecting` = the Task-implemented local helper: full Renderer flow from IMPORTANT NOTES with `timing_sink = Some(Vec::new())`, 96×96, spp 2, then the closure sees the sink. `mesh_probe_scene` packs: ground quad in `scene.geometry` (mat 0 grey), `scene.mesh_instances` with the cube prototype at scale 2 when `with_cube`, materials 0–2 via `pack_vulkan_mesh_material`, the 4-light rig via `pack_vulkan_directional_light`, `camera_layer`, all dirty marks.)

- [ ] **Step 2: Run to verify failure**

```bash
SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release \
  --lib mesh_instance_trace_probe -- --nocapture --test-threads=1 2>&1 | tail -8
```

Expected: FAIL — either compile error (helpers absent) or, once helpers exist but spectra is unwired, `trace_instances never dispatched`.

- [ ] **Step 3: Implement** — three surgical pieces, no stubs:
  1. `kernel_set.rs`: `pub const TRACE_INSTANCES: &str = "trace_instances";` in `names` + the `core_kernels` row `(TRACE_INSTANCES, "instancing", "trace_instances")`.
  2. `ray_queue.rs`: `pub hit_tri_in_blas: GpuBufferHandle` — `alloc_u32(n)` + `fill_i32(&h, -1)` (copy the `hit_instance` block verbatim), bound as `"g_hit_tri_in_blas"` in `bind_to_map`, `dummy()` in the stub constructors.
  3. `renderer.rs`: after each BVH_TRAVERSE dispatch in the bounce loop —

```rust
if gs.mesh_instance_count() > 0 {
    bindings.set_uniform_i32("u_num_instances", gs.mesh_instance_count() as i32);
    bindings.set_uniform_i32("u_total_bvh_nodes", gs.mesh_total_bvh_nodes() as i32);
    bindings.set_uniform_i32("u_tlas_node_count", gs.mesh_tlas_node_count() as i32);
    bindings.set_uniform_i32("u_instance_motion_blur", 0);
    let id = self.kernels.mgr.get(crate::kernel_set::names::TRACE_INSTANCES, &mut self.gpu)?;
    self.dispatch_kernel("trace_instances", id, pixel_threads, &bindings)?;
}
```

  (adapt to the file's actual `get`/dispatch idiom — copy the neighbouring BVH_TRAVERSE lines).
  4. `megakernel.slang`: declare `StructuredBuffer<uint> g_blas_tri_material;` next to `g_blas_vertices`; in the `is_instance_hit` branch replace the material fallback:

```slang
// Before:
tri_material_id = (material_override_inst >= 0) ? material_override_inst : 0;
// After:
tri_material_id = (material_override_inst >= 0)
    ? material_override_inst
    : int(g_blas_tri_material[blas_start_inst + tri_blas_local]);
```

- [ ] **Step 4: Wire at exact callsite** — Step 3 IS the wiring (registration table, bounce loop, bind map). Confirm `GpuScene` accessor names added in Tasks 1–2 (`mesh_instance_count()`, `mesh_total_bvh_nodes()`, `mesh_tlas_node_count()`) are the ones renderer.rs calls — one naming authority, no drift.

- [ ] **Step 5: Run — verify non-trivial output** (the Step-2 command). Expected: PASS printing `dispatches>=1`, `r-b >= 0.10`, `delta >= 0.10`. If the wall region is black, check dirty marks + that `bind_to_map` ran after upload; if grey (mat 0), the per-triangle material buffer or permutation is wrong.

- [ ] **Step 6: Commit**

```bash
cd ~/src/spectra && git add rust/spectra-renderer rust/spectra-integrator slang/megakernel.slang && \
git commit -m "feat(renderer): dispatch trace_instances — instanced BLAS hits shade with per-triangle materials"
cd ~/src/ochroma && git add crates/vox_render/src/splat_backend.rs && \
git commit -m "test(vox_render): GPU probe — instanced mesh trace dispatch + per-tri materials"
```

---

## Task 4: `trace_shadow_instances` — instanced geometry casts real sun shadows

**Files:**
- Modify: `~/src/spectra/slang/instancing.slang` (new entry point)
- Modify: `~/src/spectra/rust/spectra-renderer/src/{kernel_set.rs, renderer.rs}`
- Test: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (`mesh_instance_casts_shadow`)

**Acceptance:** the run-recipe command with `--lib mesh_instance_casts_shadow` → prints `[inst_shadow] ground luma: shaded=<a> open=<b> ratio=<a/b> (< 0.60)` with both lumas real (open ≥ 0.15 so the scene is actually lit) and ratio < 0.60.

**Wiring requirement:** `SHADOW_TRACE_INSTANCES` registered as `(SHADOW_TRACE_INSTANCES, "instancing", "trace_shadow_instances")`; dispatched in `renderer.rs` **immediately after every `shadow_trace` dispatch** (locate the `self.dispatch_kernel("shadow_trace", ...)` site), gated on `gs.mesh_instance_count() > 0`, same shadow-ray thread count.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
#[cfg(feature = "spectra-native")]
fn mesh_instance_casts_shadow() {
    // Big ground quad (soup) + one 8x0.5x8 slab prototype instanced at y=3.
    // Sun nearly overhead; fills/dome dimmed so the shadow is measurable.
    let rig = LightRig { sun_dir: [0.15, 1.0, 0.1], sun_intensity: 5.0, sky_intensity: 0.15,
                         camera_fill: 0.10, rim_fill: 0.05, sky_dome_intensity: 0.10, ..Default::default() };
    let (mut scene, cam) = shadow_probe_scene(&rig);  // camera looks down-forward at the ground
    let rgba = render_scene_collecting(&mut scene, &cam, |_| {});
    let shaded = region_mean_luma(&rgba, UNDER_SLAB_REGION);  // ground directly under the slab
    let open   = region_mean_luma(&rgba, OPEN_GROUND_REGION); // ground far from the slab
    println!("[inst_shadow] ground luma: shaded={shaded:.3} open={open:.3} ratio={:.3} (< 0.60)", shaded / open);
    assert!(open >= 0.15, "scene not lit — open ground luma {open:.3}");
    assert!(shaded / open < 0.60, "instance casts no shadow: {shaded:.3}/{open:.3}");
}
```

- [ ] **Step 2: Run to verify failure** (run-recipe + filter). Expected: compiles after helpers, then FAILS the ratio assert ≈ 1.0 — the soup-only `trace_shadow` cannot see the instanced slab.

- [ ] **Step 3: Implement** — in `instancing.slang`, a second compute entry sharing the existing instance/TLAS buffers; shadow-ray I/O re-declared with the `bvh_traverse.slang` names:

```slang
StructuredBuffer<float> g_shadow_ox;  // … oy, oz, dx, dy, dz, g_shadow_max_t
RWStructuredBuffer<int> g_shadow_vis;
uniform int u_num_shadow_rays;

[shader("compute")]
[numthreads(64, 1, 1)]
void trace_shadow_instances(uint3 tid: SV_DispatchThreadID)
{
    int i = int(tid.x);
    if (i >= u_num_shadow_rays) return;
    if (g_shadow_vis[i] == 0) return;            // soup pass already occluded it
    float3 ro = float3(g_shadow_ox[i], g_shadow_oy[i], g_shadow_oz[i]);
    float3 rd = float3(g_shadow_dx[i], g_shadow_dy[i], g_shadow_dz[i]);
    float max_t = g_shadow_max_t[i];
    // TLAS + per-instance BLAS traversal copied from trace_instances, with
    // ANY-HIT semantics: on the first triangle hit with 0.001 < world_t < max_t,
    // write g_shadow_vis[i] = 0 and return. Static transforms only (no motion blur).
}
```

  Full traversal body — every branch implemented (clone the `trace_instances` loop, replace best-hit bookkeeping with early-exit). Register `SHADOW_TRACE_INSTANCES`; dispatch after the `shadow_trace` site with the same thread count and the Task-3 uniform block.

- [ ] **Step 4: Wire at exact callsite** — the dispatch in `renderer.rs` directly below `self.dispatch_kernel("shadow_trace", ...)`, same gate. Step 3's registration row completes the wiring.

- [ ] **Step 5: Run — verify non-trivial output** (run-recipe + filter). Expected: PASS, e.g. `shaded=0.071 open=0.221 ratio=0.321` — a printed ratio near 1.0 means the kernel ran but hit nothing (check `g_shadow_max_t` semantics / world-t comparison).

- [ ] **Step 6: Commit**

```bash
cd ~/src/spectra && git add slang/instancing.slang rust/spectra-renderer && \
git commit -m "feat(renderer): trace_shadow_instances — instanced meshes occlude shadow rays"
cd ~/src/ochroma && git add crates/vox_render/src/splat_backend.rs && \
git commit -m "test(vox_render): instanced slab casts a measured ground shadow"
```

---

## Task 5: `pathtrace_mesh_scene_to_rgba` — the public instanced-scene entry point

**Files:**
- Modify: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (types + entry point + test)

**Acceptance:** run-recipe with `--lib mesh_scene_two_instances` → prints `[mesh_scene] left rgb=(<r>,<g>,<b>) right rgb=(<r2>,<g2>,<b2>) | left R>2B: true | right B>2R: true | both lit >= 0.15` with the left instance measurably red and the right measurably blue from per-instance `material_override`s.

**Wiring requirement:** `pathtrace_mesh_scene_to_rgba` is `pub` in `splat_backend.rs`, packs `SceneState` (ground/sentinel soup + `MeshInstanceLayer` + materials + 4-light rig + camera) and drives the full Renderer flow itself — it must NOT call the probe-test helpers; the helpers from Tasks 3–4 get rewritten to call IT (one scene-packing authority). Validation errors (empty inputs, bad indices, scale ≤ 0, textured/emissive instanced materials) are explicit `Err`s with the offending index in the message.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
#[cfg(feature = "spectra-native")]
fn mesh_scene_two_instances() {
    let cube = MeshPrototype { positions: cube_positions(), indices: cube_indices(), material_ids: vec![] };
    let materials = [
        PbrMaterial { base_color: [0.45, 0.45, 0.45], ..Default::default() }, // 0 ground/default
        PbrMaterial { base_color: [0.80, 0.10, 0.08], ..Default::default() }, // 1 red
        PbrMaterial { base_color: [0.08, 0.12, 0.80], ..Default::default() }, // 2 blue
    ];
    let instances = [
        MeshSceneInstance { prototype_index: 0, position: [-2.0, 0.0, 0.0],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0], uniform_scale: 1.0, material_override: 1 },
        MeshSceneInstance { prototype_index: 0, position: [ 2.0, 0.0, 0.0],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0], uniform_scale: 1.0, material_override: 2 },
    ];
    let rgba = pathtrace_mesh_scene_to_rgba(
        &[cube], &instances, &materials,
        Some(GroundPlane { y: 0.0, half_extent: 20.0, material_id: 0 }),
        [0.0, 3.0, 9.0], [0.0, 0.8, 0.0], 0.9, 160, 120, 4,
        &LightRig::default(),
    ).expect("instanced mesh scene should render");
    let l = region_mean_rgb(&rgba, left_box_region());
    let r = region_mean_rgb(&rgba, right_box_region());
    println!("[mesh_scene] left rgb=({:.3},{:.3},{:.3}) right rgb=({:.3},{:.3},{:.3}) | left R>2B: {} | right B>2R: {} | both lit >= 0.15",
        l.0, l.1, l.2, r.0, r.1, r.2, l.0 > 2.0 * l.2, r.2 > 2.0 * r.0);
    assert!(l.0 > 2.0 * l.2, "left instance not red: {l:?}");
    assert!(r.2 > 2.0 * r.0, "right instance not blue: {r:?}");
    assert!(luma3(l) >= 0.15 && luma3(r) >= 0.15, "instances not lit: {l:?} {r:?}");
}
```

- [ ] **Step 2: Run to verify failure** (run-recipe + filter). Expected: FAIL — `cannot find function pathtrace_mesh_scene_to_rgba`.

- [ ] **Step 3: Implement** — `MeshPrototype`/`MeshSceneInstance`/`GroundPlane` structs (doc-commented, `#[cfg(feature = "spectra-native")]`, pub fields per the `SdfSceneInstance` precedent) + the entry point: validate; build the soup (`GroundPlane` → 2-tri quad with upward normals, or the SDF-path's degenerate sentinel when `None`); convert prototypes/instances to `MeshPrototypeData`/`MeshInstanceRecord` (transform via the IMPORTANT-NOTES glam recipe); `scene.mesh_instances = MeshInstanceLayer::from_parts(...)`; materials via `pack_vulkan_mesh_material`; the 4-light rig + sky-dome + camera + renderer flow copied from `pathtrace_mesh_lit_to_rgba`'s body; RGBA8 conversion identical. Full body, every validation branch real.

- [ ] **Step 4: Wire at exact callsite** — rewrite Task 3/4's `mesh_probe_scene`/`shadow_probe_scene` helpers to build their scenes THROUGH `pathtrace_mesh_scene_to_rgba` (the probe variant keeping `timing_sink` uses a small internal fn the public entry point also calls — single packing authority, no duplicated scene code). Re-run both probe tests to prove the refactor held.

- [ ] **Step 5: Run — verify non-trivial output**

```bash
SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  scripts/build-spectra-native.sh test -p vox_render --features spectra-native --release \
  --lib mesh_scene -- --nocapture --test-threads=1
```

Expected: `mesh_scene_two_instances`, `mesh_instance_trace_probe`, `mesh_instance_casts_shadow` ALL PASS with their printed numbers.

- [ ] **Step 6: Commit**

```bash
cd ~/src/ochroma && git add crates/vox_render/src/splat_backend.rs && \
git commit -m "feat(spectra-native): pathtrace_mesh_scene_to_rgba — instanced prototype meshes via TLAS"
```

---

## Task 6: The M1 milestone — `mesh_city_block_renders_instanced_buildings`

**Files:**
- Modify: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (scene helper + milestone test)

**Acceptance:** the **Done When** command and printed block, verbatim, with gates S ≥ 10, F ≥ 0.80, T = 12, M ≥ 10, occlusion front-wins, C ≥ 0.15 — and the PNG written for human eyeballing.

**Wiring requirement:** the test calls `pathtrace_mesh_scene_to_rgba` (nothing else); `mesh_city_block_scene()` is a tests-module helper returning `(Vec<MeshPrototype>, Vec<MeshSceneInstance>, Vec<PbrMaterial>, eye, center, fov_y, (w, h), LightRig)`. Reuse the SDF M1 test's projection/span-fill code shape (`sdf_city_block_renders_multiple_solid_buildings` — projected world-AABB → screen box → per-row span-fill) and its `write_png_rgba`/`luma` helpers; per-instance world AABBs come from the prototype's local AABB through each instance transform.

- [ ] **Step 1: Write the failing test** — gates first:

```rust
#[test]
#[cfg(feature = "spectra-native")]
fn mesh_city_block_renders_instanced_buildings() {
    let (protos, instances, materials, eye, center, fov_y, (w, h), rig) = mesh_city_block_scene();
    assert_eq!(protos.len(), 3); assert_eq!(instances.len(), 12);
    let t0 = std::time::Instant::now();
    let rgba = pathtrace_mesh_scene_to_rgba(&protos, &instances, &materials,
        Some(GroundPlane { y: 0.0, half_extent: 80.0, material_id: 0 }),
        eye, center, fov_y, w, h, 4, &rig).expect("city block should render");
    let secs = t0.elapsed().as_secs_f64();
    write_png_rgba(&png_path("ochroma_mesh_city_block.png"), &rgba, w, h);
    // Per-instance: project the instance's world AABB (prototype local AABB
    // through its transform) → screen box → span-fill (the SDF M1 metric).
    let stats = per_instance_span_fill(&rgba, &protos, &instances, eye, center, fov_y, w, h);
    // Materials: within each solid box, mean colour of the top-third lit rows
    // (roof) vs bottom-two-thirds (walls) must split by > 0.12 (L1 distance).
    let mat_ok = per_instance_wall_roof_split(&rgba, &stats, 0.12);
    // Transforms: all boxes lit AND pairwise projected centers >= 8 px apart.
    let distinct = pairwise_distinct_centers(&stats, 8.0);
    // Occlusion: instance 9 (tower, front) is placed on the view ray to
    // instance 1 (rowhouse, rear); in the overlap box the mean colour must be
    // nearer the TOWER wall colour than the ROWHOUSE wall colour.
    let (d_front, d_rear) = occlusion_pair_colour_distance(&rgba, &stats, 9, 1, &materials);
    /* prints the exact [mesh_block] report from Done When, then: */
    assert!(stats.solid >= 10, "solid {}/12, fills {:?}", stats.solid, stats.fills);
    assert!(stats.mean_fill >= 0.80, "mean span-fill {:.3}", stats.mean_fill);
    assert_eq!(distinct, 12, "transforms collapsed: {distinct}/12 distinct");
    assert!(mat_ok >= 10, "wall/roof material split on {mat_ok}/12");
    assert!(d_front < d_rear, "occlusion wrong: front {d_front:.3} rear {d_rear:.3}");
    assert!(stats.coverage >= 0.15, "block lost in sky: {:.3}", stats.coverage);
}
```

- [ ] **Step 2: Run to verify failure** (Done-When command). Expected: FAIL — `cannot find function mesh_city_block_scene`.

- [ ] **Step 3: Implement the scene helper** — 3 procedural prototypes, each multi-material REAL geometry (no textures, flat shading is the M1 contract):
  - **rowhouse** (~9×7×8 m): box body + gable prism + recessed door + 4 recessed window insets (each inset = 5 quads sunk 0.3 m — real reveals, the no-faking point), wall mat `warm brick [0.62,0.30,0.22]`, roof mat `slate [0.18,0.20,0.24]`, trim mat `[0.85,0.82,0.75]`;
  - **tower** (~8×18×8 m): stacked box + stepped parapet, wall `render grey-green [0.45,0.52,0.42]`, same roof/trim mats;
  - **workshop** (~14×5×10 m): low box + monitor-roof ridge, wall `ochre [0.70,0.55,0.25]`.
  ~100–300 triangles each, `material_ids` per triangle. 12 instances on a 4×3 street grid (~10 m pitch), yaws {0°, 90°, 180°, 45°}, scales 0.9–1.2, prototypes interleaved; instance 9 = a tower placed on the eye→instance-1 ray for the occlusion pair. Camera: oblique elevated 3/4 view framing all 12 (mirror the SDF `city_block_scene` framing math), 640×400, `LightRig { sun_dir: [0.4, 0.7, 0.3], ..Default::default() }`. Implement the four metric helpers with the SDF M1 test's projection code as the base.

- [ ] **Step 4: Wire at exact callsite** — the milestone test IS the wiring proof of the whole wave (entry point → layer → pack → TLAS → dispatch → per-tri materials → shadows). Nothing else to connect; confirm the test file contains NO direct `SceneState` construction.

- [ ] **Step 5: Run — verify non-trivial output** — the Done-When command. Expected: PASS, the full `[mesh_block]` report with real numbers, PNG written. **Open the PNG and look at it** (12 distinct solid buildings, 3 silhouette families, rotations visible, front tower hiding the rear rowhouse, shadows on the ground). If mean fill is high but the image is wrong, the metrics are lying — fix the metrics, not the gates.

- [ ] **Step 6: Commit**

```bash
cd ~/src/ochroma && git add crates/vox_render/src/splat_backend.rs && \
git commit -m "feat(spectra-native): Mesh M1 — instanced city block of real building meshes (12 instances / 3 BLAS)"
```

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — Task 1 packs *and* uploads *and* binds; Task 3 registers *and* dispatches *and* is GPU-verified; no "wire later" anywhere
- [x] Every `Acceptance` names a real non-trivial printed output (packed-bit values, dispatch counts, colour means, luma ratios, span-fills) — never "tests pass", never zeroes
- [x] Every `Wiring requirement` names an exact function and exact file (`Uploader::upload_from_state`, `GpuScene::bind_to_map`, `KernelSet::new` core_kernels, the BVH_TRAVERSE/shadow_trace dispatch sites, `RayQueue::bind_to_map`)
- [x] `IMPORTANT NOTES` carries the real, code-verified signatures (pathtrace_mesh_*, build_median_bvh/BvhBuildResult, the 24-float InstanceData and 10-float node layouts, RayQueue/KernelSet patterns, Renderer flow order) from the design doc
- [x] `File Map` lists every file that appears in any task
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies — Task 4's kernel skeleton explicitly demands the full traversal loop
- [x] `Done When` names one exact command and a human-observable result (the printed `[mesh_block]` block + the PNG a human eyeballs)
- [x] Names are consistent across tasks (`pack_mesh_instances`, `build_aabb_tlas`, `pathtrace_mesh_scene_to_rgba`, `mesh_city_block_scene`, `GpuScene::mesh_instance_count()` — Task 3 Step 4 pins the accessor-name authority)
