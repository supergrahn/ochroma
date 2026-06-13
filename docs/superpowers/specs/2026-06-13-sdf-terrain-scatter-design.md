# Design: SDF Terrain + Default Scatter (2026-06-13)

**Status:** Draft — REVISED 2026-06-13 (3D-SDF pivot)
**Scope:** Unify Forge's terrain SDF output, the engine's terrain merge/render path, the terraform tool's data path against SDF terrain, and a default PolyHaven scatter manifest (ground textures + grass/bush/tree models) into one coherent pipeline that extends — does not contradict — the hybrid-atom-sdf-lod direction and the existing scatter/water/scene-content designs.
**Related:** `[hybrid-atom-sdf-lod-direction](../../../.claude/projects/-home-tom-espen-src-ochroma/memory/hybrid-atom-sdf-lod-direction.md)`, `[3d-sdf-terrain-directive](../../../.claude/projects/-home-tom-espen-src-ochroma/memory/3d-sdf-terrain-directive.md)`, `[SDF Water](./2026-06-12-sdf-water-design.md)`, `[Scatter Field](./2026-06-12-scatter-field-design.md)`, `[SDF Scene Content Roadmap](./2026-06-12-sdf-scene-content-roadmap.md)`, `[Terraform Tool](./2026-06-11-terraform-tool-design.md)`, `[FloraPrime Vegetation Plan](../plans/2026-06-12-floraprime-vegetation-plan.md)`

> **REVISION NOTE (2026-06-13 — corrects the original draft).** The first draft of this doc made the **2D heightfield authoritative** and derived the SDF from it ("edit heightfield, re-derive SDF regionally"). That is **wrong**: a heightfield is 2.5D and *cannot* represent overhangs, vertical cliff faces, or interior cave voids. The user requires terraforming to sculpt **cliffs and caves**, so per `3d-sdf-terrain-directive.md` the **3D SDF is now authoritative and directly editable**; a heightfield only *seeds* the base ground. Sections 1, 2, 4.1–4.4, 5, 6 (terrain APIs), 7, 8, 9 are rewritten accordingly. The scatter-asset manifest (§4.5, §6 tables) is **unaffected** and carried forward verbatim.

---

## 1. Problem Statement

**The terrain must be a TRUE 3D SDF so terraforming can sculpt cliffs (overhangs/vertical faces) and caves (interior voids). A heightfield cannot represent either — it is the wrong authoritative layer.** What exists today is fragmented and 2.5D-shaped:

- **Forge generates terrain as a 2.5D heightfield with an empty SDF slot.** `forge-cli terrain` (`~/src/forge/crates/forge-cli/src/cmd/terrain.rs:55-71`) builds `HeightfieldSpatial` and leaves `sdf: None` (`~/src/forge/crates/volume/src/spatial.rs:30`); nothing populates the `SdfSpatial` the module doc promises (`volume/src/lib.rs:23`). A heightfield is single-valued in Y — it physically cannot carry overhangs or voids.
- **The only true-3D SDF the codebase cooks is a single DENSE box per asset.** `ready_sdf` (`~/Ochroma/projects/civitas_care/src/bin/game_asset_cook.rs:2909`) emits `ReadyAssetSdfVolume { resolution:[u16;3], origin:[f32;3], voxel_size, narrow_band, sign, distances_snorm16 }` (`~/Ochroma/projects/civitas_care/src/asset/mod.rs:372-383`) — true 3D and narrow-band (±4 voxels, `SDF_GPU_TARGET` `~/.../game_asset_cook.rs:2820`), but a **single 64³-class box sized for a building**, not a sparse paged field over a km-scale map. There is no brick/clipmap paging layer; the GPU samples one dense volume per instance (`megakernel_sdf.slang::sdf_sample_volume_local:96`, one `origin`/`voxel_size`/`distance_offset` per volume header). A dense 3D box over a 4 km map is multi-TB — a non-starter.
- **Three disjoint terrain code paths exist and none is wired end-to-end from the game.** NDF-chunk sphere-trace (`~/src/spectra/slang/terrain_sdf.slang:70`, gated `g_terrain_enabled` at `megakernel.slang:1342`, **not wired from `crates/vox_render/`**); baked per-building snorm SDF (`splat_backend.rs:776`, fully wired); RTX Mega-Geometry displaced mesh (`~/src/spectra/.../rtxmg/src/terrain.rs`, Python-mirror maturity, not in the resident frame). The displaced-mesh path is heightfield-only by construction (displacement along a base sheet) — caves/overhangs are out of its reach.
- **GPU CSG already exists and already has the carve/union ops terraform needs.** `sdf_eval.slang` (`~/src/spectra/slang/sdf_eval.slang`) defines `SDFOpGPU` (16 floats/node) with `OP_UNION/OP_SUBTRACT/OP_INTERSECT/OP_SMOOTH_UNION/OP_SMOOTH_SUBTRACT` (`:28-31`) and the combine dispatch `min/max(-)/smooth_*` is implemented (`:127-133`), plus `eval_sdf_tree` (`:88`) and a batch eval compute kernel with central-difference gradients (`eval_sdf_batch:158`). Primitives: ground-plane/sphere/box/capsule/noise-field (`:37-62`). **This is the engine that 3D-CSG terraform brushes map onto** — subtract = carve cave/cliff cut, smooth-union = overhang build-up.
- **Surface extraction exists but is split between a real GPU path and a stub CPU path.** GPU `marching_cubes.slang` (`~/src/spectra/slang/marching_cubes.slang`) has the full 256-entry edge/tri tables (`g_mc_edge_table`/`g_mc_tri_table:7-8`), edge interpolation (`interp_vertex:55`), and emits a vertex/normal/attribute triangle soup — but it consumes **ASDF octree cells** (`g_cell_centers/g_cell_data`), not our snorm16 bricks. The Rust `spectra-flexicubes` crate (`~/src/spectra/rust/spectra-flexicubes/src/lib.rs`) is advertised as dual marching cubes but is a **SIMPLIFIED STUB**: it fan-triangulates crossing edges from a cell with no proper MC lookup table (`:111-118` comment: *"For a proper implementation we'd use the full MC lookup table"*), takes a single dense `res³` field (`extract:46`), and is not wired to anything game-side (only `spectra-rs` re-exports it). So: a working GPU MC kernel exists for ASDF input; **a brick-input, cliff-edge-preserving extractor for our terrain bricks is net-new** (see §4.3).
- **The terraform brush edits a heightfield, not an SDF.** `sculpt_heightmap` (`~/Ochroma/projects/civitas_care/src/map/sculpt.rs:73`) → `apply_stamp` (`src/map/terrain.rs:122`) → `rederive_region` (`src/map/derive.rs:43`) bumps a 2D `CellClass` mask + `version`; no SDF is touched and the brush is structurally incapable of overhangs/voids.
- **Only one ground texture (`leafy_grass`) and zero scatter models are wired.** `STEM_SETS` (`src/asset/textures.rs:49`) + `PINNED_SETS` (`src/bin/polyhaven_fetch.rs:28`) carry a single ground set; the biome LUT `BIOME_SPECIES_PROBS` (`~/src/spectra/slang/biome_gpu.slang:44-63`) names species slots (oak/beech/grass/...) with no asset behind any slot.

---

## 2. Done When

Running `cargo run --bin civitas_care` and entering a fresh temperate map, then dragging the terraform brushes over open ground, produces — in the live window, verifiable by a human at the keyboard:

1. The **Carve (subtract) brush dug horizontally into a hillside leaves a real CAVE**: a human walks/flies the camera *inside* the hill and sees an enclosed void with terrain above, below, and on three sides — i.e. the surface is genuinely multi-valued in Y at that (x,z). (A heightfield cannot produce this; it is the proof the SDF is 3D-authoritative.)
2. The **Carve brush cutting downward against a raised lip leaves a vertical CLIFF face / overhang** — the cut face is near-vertical and there is terrain that hangs *over* empty space beneath it, again multi-valued in Y.
3. The new geometry under every brush is a **re-extracted mesh skin** (real triangles, not a height displacement) that updates in real time — ≥60 FPS shown in title on the 4070 Ti, sub-frame dirty-brick re-extract — and the cave interior / cliff face is correctly lit and textured (grey rock on the steep/overhanging surfaces).
4. **grass tufts, shrubs, and trees** scatter on the new flat/gentle ground with biome-correct density (dense grass on flat, none on the steep cliff/overhang faces, none inside the cave, none under the water line), textured by the temperate ground set.
5. Dragging a brush near a placed building is **refused** (preview tints red), with no terrain/building seam desync.

A human confirms all five by eye without reading code. (1) and (2) are the load-bearing acceptance: they are impossible under the superseded heightfield design.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| 3D narrow-band brick field stores caves | `cargo test -p vox_render terrain_brick_cave_topology` builds a field, carves a cave with a subtract stamp, and asserts a vertical ray at the cave (x,z) crosses the zero-set **≥3 times** (ground→roof→floor→ground) — multi-valued in Y, impossible for a heightfield | `assert!(field.brick_count() > 0)` |
| Sparse brick allocation tracks surface | `cargo test -p vox_render terrain_brick_sparsity` asserts a 1 km flat field allocates `O(area/brick_xz²)` bricks (one band layer), and carving a cave allocates **only** the bricks the cave passes through (count == covered bricks ±0) | `assert!(field.bricks().len() < dense_count)` |
| 3D CSG subtract carves the band | `cargo test -p vox_render terraform_csg_subtract` applies a sphere-subtract stamp; asserts every voxel inside the sphere flipped sign (was `<0`, now `>0`) and ONLY bricks overlapping the stamp AABB are marked dirty | `assert!(field.version() > old)` |
| 3D CSG smooth-union builds overhang | `cargo test -p vox_render terraform_csg_overhang` smooth-unions a blob offset laterally above a slope; asserts a probe under the blob lip reads `<0` (solid) while the same (x,z) one band-width lower reads `>0` (the void under the overhang) | `assert!(field.distances.iter().any(\|d\|*d<0.0))` |
| Surface extraction handles cave topology | `cargo test -p vox_render extract_cave_mesh` extracts a brick containing a carved cave and asserts the mesh has the cave's interior facing triangles (≥1 triangle with normal·(+Y) < −0.3, i.e. a downward-facing roof) AND is watertight-manifold (every edge shared by exactly 2 tris) | `assert!(!mesh.triangles.is_empty())` |
| Dirty-brick re-extract is local | `cargo test -p vox_render extract_dirty_only` carves one brick, asserts re-extraction touches ONLY the dirty brick's chunk vertices (untouched chunks' vertex buffers byte-identical) | `assert!(chunk.dirty)` |
| Heightfield SEEDS the 3D field | `cargo test -p vox_render seed_field_from_heightmap` asserts `vol.sample(x, h(x,z)+voxel, z) ∈ (0, voxel*1.5)` and `< 0` one voxel below `h(x,z)` for 100 sampled cells | `assert!(vol.distances.iter().any(\|d\|*d!=0.0))` |
| Biome-gated scatter placement | `cargo test -p vox_render scatter_biome_gate -- --nocapture` prints `trees=0 on Water/Slope/cave, grass>500 on flat temperate` and asserts those counts | `assert!(instances.is_some())` |
| Default ground texture blend | `cargo run --bin civitas_care` then screenshot diff: flat cell RGB greener than rock cell (G>R on flat, R≈G on slope/cliff) | texture file exists on disk |

---

## 4. Architecture

### 4.1 Governing reconciliation (what this design commits to)

Per `hybrid-atom-sdf-lod-direction.md`: **MESH renders the visible world; SDF is the engine substrate (collision/destruction/terraforming).** Per `3d-sdf-terrain-directive.md`: **the substrate is a TRUE 3D SDF, authoritative and directly editable** — because cliffs and caves are required and only a 3D field can carry them. Applied to terrain:

- **3D SDF terrain = the authoritative, editable substrate.** A signed-distance field over a 3D *volume* (not a height grid), stored as **sparse narrow-band voxel bricks, paged/clipmapped** — bricks exist only where a surface passes through them, so caves and overhangs cost bricks *only where they exist* and a flat km-scale map costs one thin band layer. This is sim/collision/water-CSG/foundation truth and the Tier-3 software render floor.
- **The visible terrain = a mesh skin extracted from the 3D SDF by surface extraction** (marching cubes / dual contouring — arbitrary topology, so caves and overhangs mesh correctly), traced as triangles in the unified TLAS alongside buildings and movers (one closest-hit, no per-class branch). NOT a heightfield displacement — displacement is single-valued in Y and would silently drop every overhang and cave.
- **Authoring is DIRECTLY on the 3D SDF** via replayable 3D CSG brush stamps (§4.4). A 2D heightfield is retained ONLY as the *seed* of the base ground (§4.3) and for sim layers that legitimately want a 2D field (drainage, the `CellClass` slope/water mask); it is never the editable terrain geometry and is never re-derived back from the SDF.

This **extends** the existing docs but **corrects this doc's own first draft**: terrain authoring moves from "edit heightfield → re-derive SDF" to "edit the 3D SDF directly." The water design's `is_water` unification (`2026-06-12-sdf-water-design.md:121`) and CSG against the terrain field are unchanged; the scatter-field placement machine (`scatter_field.rs`) is unchanged. The older `hybrid-atom-sdf-lod-design.md` SDF-near/splat-mid premise stays superseded per the direction memo, its clipmap tier-selection reasoning salvaged here for terrain brick-clipmap LOD.

### 4.2 The 3D SDF terrain representation (sparse narrow-band bricks)

The field is `d(p) = min(d_seed, eval_edits(p))` evaluated/stored as bricks. **No dense 3D box** (a 4 km × vertical-extent dense volume is multi-TB) and **no heightfield-analytic march as the authoritative form** — the field must be free to be multi-valued in Y.

- **Brick = a small dense narrow-band tile in the EXISTING `ReadyAssetSdfVolume` snorm16 format** (`~/Ochroma/projects/civitas_care/src/asset/mod.rs:372-383`: `resolution:[u16;3]`, `origin`, `voxel_size`, `narrow_band`, `sign`, `distances_snorm16`) — the format already cooks true-3D narrow-band volumes (`ready_sdf`, `~/.../game_asset_cook.rs:2909`; `SDF_GPU_TARGET` band = ±4 voxels, `:2820`). A terrain brick is one such volume covering e.g. a 8 m × 8 m × 8 m cube at ~0.25 m voxels (≈32³). Reuse means buildings, props, and terrain residented identically in the GPU SDF atlas and sampled by the SAME shader path (`megakernel_sdf.slang::sdf_sample_volume_local:96`, one header per volume).
- **Sparsity is the whole point.** A brick is allocated ONLY if the zero-set passes within `narrow_band` of it. Flat ground → one band layer of bricks. A cave or overhang → the extra bricks the void/lip occupy, nowhere else. This is what makes a 3D field over a km-scale map affordable.
- **Paging / clipmap** keeps only bricks near the camera (and any region under active sim) GPU-resident, in concentric LOD rings (coarser voxel size per ring). The salvaged `hybrid-atom-sdf-lod-design.md` clipmap tier-selection logic drives ring assignment.
- **`sign` semantics:** terrain bricks are `ReadyAssetSdfSign::Closed` (`asset/mod.rs:393`) — interior negative distances are authoritative (you can be *inside* the hill, which is exactly what a cave needs). The reserved `Shell` value is not used here.

**Net-new vs existing:** the per-brick volume format and its GPU sampling EXIST; the **sparse brick index / allocator / paging-clipmap layer over them is net-new** (today there is one dense volume per asset instance, no spatial brick index — `megakernel_sdf.slang` indexes volumes by instance, not by world-space brick coordinate).

### 4.3 Seeding the base ground from the heightmap (one-time, not authoritative)

The initial terrain is seeded into the brick field from `MapTerrain`'s heightmap (`~/Ochroma/projects/civitas_care/src/map/terrain.rs:18` `MapTerrain{cell_size, origin, heights:Heightmap}`, `:72-75`). For each brick the seed surface passes through, the cook writes `d(p) = signed distance to the heightmap surface` (negative below `h(x,z)`, positive above) into the brick's snorm16 band:

```
GAME (civitas_care)                  ENGINE (vox_render)                 SPECTRA
MapTerrain.heights ----seed (once)--> seed_terrain_bricks (NEW)          (substrate)
  (Heightmap, cell_size, origin)        -> sparse narrow-band bricks      sdf_sample_volume_local
  terrain.rs:18,72                          in ReadyAssetSdfVolume fmt     megakernel_sdf.slang:96
bricks (post-seed + edits) ----extract--> extract_terrain_skin (NEW)  --> per-chunk triangle mesh
  (dirty bricks only)                       (surface nets / DC)            -> CLAS -> TLAS (the SKIN)
is_water / sea_level ----cook--------> WaterSurfaceInput (water §193) -->  analytic plane, CSG'd
  terrain.rs:303                                                           max(p.y-h_w, -d_terrain)
```

The seed is a heightfield→SDF *projection* and so produces no overhangs at t=0 (correct — the base map has none). All overhangs/caves arrive via §4.4 edits **into the same bricks**. The heightmap is consulted once at seed; thereafter the SDF is truth. Generic grids/brick arrays cross the engine seam — never `MapTerrain`/`CellClass` (CLAUDE.md engine rule).

The skin is **just another resident in the unified TLAS**, one CLAS/BLAS entry per extracted terrain chunk, **incrementally refit** on edit (no full rebuild). Three tiers: T1 NVIDIA full CLAS extracted-terrain chunks; T2 AMD/Intel portable KHR-RT per-cluster BLAS; **T3 software floor = sphere-trace the brick SDF directly** (the substrate is already a sampleable 3D field — `sdf_sample_volume_local` — so the software floor needs no separate representation; this also serves shadow/AO/water-bottom queries).

Buildings sit on terrain by querying the **3D substrate** at placement (downward ray to the first zero-crossing = ground height, even over a cave roof). Because the skin is *extracted from* the substrate, the building base meets the skin at exactly the foundation-query surface — no z-fight, no double surface.

### 4.4 Terraforming = 3D CSG brush ops on the SDF (this is what enables cliffs + caves)

The terraform brush authors the 3D SDF directly via **replayable CSG stamps**, mapping 1:1 onto Spectra's existing `sdf_eval.slang` ops (`OP_SUBTRACT`/`OP_UNION`/`OP_SMOOTH_UNION`/`OP_SMOOTH_SUBTRACT`, `:28-31`; combine dispatch `:127-133`). Each stroke appends an immutable stamp record `{op_type, shape, center, radius/half_extents, blend_k}` to the terrain's edit log:

- **Subtract (carve):** `d ← max(d, −d_brush)` — digs a CAVE (subtract a sphere/capsule swept horizontally into a hill) or cuts a CLIFF face (subtract a box against a raised lip). The void is genuine: the field goes positive *inside* the carved region with solid terrain remaining around it.
- **Add / union (build up, overhang):** `d ← min(d, d_brush)` / `smooth_union` — raises ground, and (offsetting the brush laterally above a slope) builds an OVERHANG: solid hangs over empty space, multi-valued in Y.
- **Smooth / flatten:** `smooth_union`/`smooth_subtract` with a blend radius for natural blends; a flatten variant pulls the band toward a target plane.

**Application is dirty-brick-local, no BVH rebuild:**
1. Compute the stamp's world AABB; mark every brick it overlaps **dirty** (allocate new bricks if the stamp's band reaches empty space — this is how a cave grows into previously-unstored interior).
2. Re-evaluate the field in each dirty brick's narrow band (`d_new = op(d_old, d_brush)`), re-encode snorm16, **re-upload only those bricks** to the SDF atlas. The existing `eval_sdf_batch` kernel (`sdf_eval.slang:158`) or a dedicated per-brick stamp kernel does this on-GPU; reference for cheap dirty-band recompute is `physics_sdf_gen.slang`'s 3-pass JFA.
3. **Re-extract the skin for the dirty bricks only** (§4.3 extractor), producing replacement triangles for the affected chunk → **TLAS refit** of that one chunk (CLAS/BLAS chunk rebuild, never the whole map). Sub-frame on the 4070 Ti at brush scale (a handful of 32³ bricks).

**Replay & save:** the edit log of stamps is the authoritative, bit-exact, replayable record (it supersedes the old `AuthoredAction::Terraform` heightfield-delta; a stamp list is far smaller than a snorm16 delta-grid and *is* the overhang/cave capability the old design explicitly refused). On load, seed from heightmap (§4.3) then replay stamps.

**Surface extraction method (net-new component — recommend DUAL CONTOURING).** No brick-input extractor exists today: the GPU `marching_cubes.slang` consumes ASDF octree cells (`g_cell_centers/g_cell_data`, `~/src/spectra/slang/marching_cubes.slang:11-25`), not our world-space snorm16 bricks, and the Rust `spectra-flexicubes` is a simplified stub (no MC table, fan-triangulation, single dense field — `~/src/spectra/rust/spectra-flexicubes/src/lib.rs:111-118,46`). Recommendation: **dual contouring** over the bricks — it places one vertex per cell from the Hermite (distance + gradient, both available: gradient via the same central difference as `eval_sdf_batch:177-183`) and **preserves sharp cliff edges and crisp cave mouths**, where plain marching cubes / surface nets round them off. The existing `marching_cubes.slang` edge/tri tables (`:7-8,55`) are a usable starting point for a surface-nets fallback if DC sharp-feature placement proves fiddly; the stub flexicubes crate should be either finished into real DC or bypassed. Extraction must run per-brick with one-voxel apron overlap into neighbors so chunk seams are watertight.

**Constraints preserved:** edits remain **blocked under the built city** (`terraform_blocked`, terraform design §4.5) — terrain bricks and building SDF stay separate atlas instances; the brush cannot desync them because it cannot edit under buildings. Water-vs-terrain stays a render-time CSG (`max(p.y−h_w, −d_terrain)`), unchanged.

### 4.5 Default scatter (ground textures + models) — plugs into the existing machine

Three existing pieces define the contract: biome classifier writes `g_biome_id` + per-cell `g_species_probs` from `BIOME_SPECIES_PROBS` (`biome_gpu.slang:44-63`; **temperate = id 3**, grassland = id 8); instancer consumes ONLY `{pos, uniform-scale, yaw}` (`instancer_gpu.slang:43-50` → 12-float TRS); ground skin is textured by PolyHaven sets fetched idempotently (`polyhaven_fetch::fetch_set:107`, maps `Diffuse/nor_gl/Rough`). Scatter placement gates are density × clearance-SDF × CellClass (scatter-field-design §4.4) with deterministic `pcg3d` hash placement, T0/T1 instanced model, T2 ≥150 m volumetric/billboard shell.

Adding the default temperate biome = (1) add ground stems to `STEM_SETS`/`PINNED_SETS`, (2) a species-slot→model table the instancer scatters, (3) the density mask the Scatter Field reads. PolyHaven models are the **bootstrap default**; FloraPrime species replace the oak/beech/hornbeam slots when cooked (PolyHaven has no broadleaf oak — its tree library is thin, which is *why* FloraPrime exists). Mass near-field grass is the procedural `LEAF_STRAP` blade, not a model instance; PolyHaven `grass_medium_01` is the mid-distance hero tuft + shape reference. Every PolyHaven model needs cook-time LOD generation (trunk→SDF, decimated mid-LOD, far billboard/shell) — PolyHaven ships no LODs.

**See §6 for the full asset manifest.**

---

## 5. Data Models

```rust
/// The authoritative 3D terrain field: a sparse, world-space index of narrow-band
/// bricks. Bricks exist only where the zero-set passes — caves/overhangs cost
/// bricks only where they are. NET-NEW (no spatial brick index exists today).
pub struct TerrainSdfField {
    brick_extent_m: f32,                 // world size of one brick cube (e.g. 8.0)
    voxel_size_m:   f32,                 // per-brick voxel (e.g. 0.25 -> ~32^3)
    bricks: HashMap<BrickCoord, Brick>,  // sparse: keyed by world brick coordinate
    edit_log: Vec<TerraformStamp>,       // replayable authoritative edit history
    dirty: HashSet<BrickCoord>,          // bricks needing re-upload + re-extract
}
pub struct BrickCoord { x: i32, y: i32, z: i32 }   // world / brick_extent_m
/// One brick = the EXISTING per-asset volume format, reused verbatim
/// (asset/mod.rs:372): resolution[u16;3], origin, voxel_size, narrow_band,
/// sign(=Closed), distances_snorm16. Residented in the same GPU SDF atlas and
/// sampled by megakernel_sdf.slang::sdf_sample_volume_local.
pub struct Brick { volume: ReadyAssetSdfVolume, atlas_slot: Option<u32> }

/// A replayable 3D CSG terraform stamp. Maps 1:1 onto sdf_eval.slang SDFOp.
pub struct TerraformStamp {
    op: SdfOp,                 // Subtract | Union | SmoothUnion | SmoothSubtract (sdf_eval.slang:28-31)
    shape: BrushShape,         // Sphere{r} | Box{half} | Capsule{a,b,r}  (sdf_eval.slang:1-3 prims)
    center: [f32; 3],
    blend_k: f32,              // smooth-op blend radius (combine uses params1.w, sdf_eval.slang:126)
}

/// Extracted skin chunk for the TLAS. One per N bricks; re-extracted when any
/// member brick is dirty. NET-NEW extractor (dual contouring recommended).
pub struct TerrainSkinChunk { aabb: Aabb, vertices: Vec<[f32;3]>, normals: Vec<[f32;3]>, tris: Vec<[u32;3]>, blas: Option<BlasHandle> }
```

```rust
/// Species slot -> default bootstrap model. Indexed by BIOME_SPECIES_PROBS column.
pub struct ScatterSlotTable {
    // biome 3 {oak,beech,hornbeam,rose,bramble,grass}
    //   -> {tree_small_02, FLORAPRIME, FLORAPRIME, shrub_01, shrub_02, grass_medium_01}
    // biome 8 {grass,wildflower,shrub,tree}
    //   -> {grass_medium_01, dandelion_01, shrub_03, tree_small_02}
    slots: Vec<(BiomeId, [Option<PolyHavenSlug>; 8])>,
}
```

---

## 6. API + Default Scatter Asset Manifest

```rust
// Engine (vox_render) — 3D SDF terrain substrate. All NET-NEW.

/// Seed the sparse brick field from the base heightmap, ONCE. Allocates only
/// bricks the seed surface passes through. d<0 below h(x,z), d>0 above.
/// Panics if heights.len() != w*d. Pure; call from cook thread.
pub fn seed_terrain_bricks(
    heights: &[f32], w: usize, d: usize, cell_size: f32, origin: [f32; 3],
    brick_extent_m: f32, voxel_size_m: f32,
) -> TerrainSdfField;

/// Apply ONE replayable 3D CSG stamp. Marks/allocates only the bricks the
/// stamp AABB overlaps; re-evaluates their narrow bands; records the stamp in
/// the edit log; returns the set of dirty bricks. NO BVH rebuild.
pub fn apply_terraform_stamp(field: &mut TerrainSdfField, stamp: TerraformStamp)
    -> Vec<BrickCoord>;

/// Replay the full edit log onto a freshly-seeded field (load path).
pub fn replay_edit_log(field: &mut TerrainSdfField);

/// Surface-extract the skin for the given (dirty) bricks. Dual contouring
/// (recommended — preserves sharp cliff edges); one-voxel apron into neighbor
/// bricks so chunk seams stay watertight. NET-NEW: no brick-input extractor
/// exists (marching_cubes.slang is ASDF-cell input; spectra-flexicubes is a
/// stub). Returns replacement chunk meshes for caller to refit into the TLAS.
pub fn extract_terrain_skin(field: &TerrainSdfField, bricks: &[BrickCoord])
    -> Vec<TerrainSkinChunk>;

/// World-space queries on the 3D field (foundation/scatter/collision).
/// Downward ray to first zero-crossing = ground height (works over cave roofs).
pub fn sample_terrain_distance(field: &TerrainSdfField, p: [f32; 3]) -> f32;
```

### Default GROUND textures — temperate (all CC0, PolyHaven, API-verified)

| Stem (`STEM_SETS`) | PolyHaven slug | Role |
|---|---|---|
| `lawn` | `leafy_grass` | manicured/park grass (ALREADY WIRED) |
| `meadow_grass` | `aerial_grass_rock` | wild grassland (biome 8) |
| `forest_floor` | `forrest_ground_03` | under-canopy deciduous floor (biome 3) |
| `forest_litter` | `forest_leaves_02` | autumn leaf-litter variant |
| `dirt` | `dirt` | bare earth / worn patches |
| `mud_leaves` | `brown_mud_leaves_01` | damp low ground / riverbank |
| `rock_ground` | `rocks_ground_06` | slope/scree (CellClass::Slope) |
| `rock_face` | `rock_face_03` | exposed bedrock / cliff |
| `dirt_path` | `grass_path_2` | worn footpath |
| `gravel_path` | `gravel` | gravel walkway |

**Minimum first-ship (4):** `leafy_grass` (have), `aerial_grass_rock`, `forrest_ground_03`, `rocks_ground_06`.

### Default SCATTER models (CC0, PolyHaven `models?c=nature`, slug-verified)

- **Trees:** `tree_small_02` (generic broadleaf default / oak slot), `pine_tree_01`, `fir_tree_01`, `pine_sapling_medium`, `fir_sapling_medium`, `dead_tree_trunk`, `tree_stump_01`. *(Oak/beech/hornbeam slots → FloraPrime when cooked; PolyHaven has no true broadleaf oak.)*
- **Bushes/shrubs:** `shrub_01`, `shrub_02`, `shrub_03`, `shrub_04` (rose/bramble/shrub slots, rotate by hash), `shrub_sorrel_01`, `wild_rooibos_bush`, `nettle_plant`, `periwinkle_plant`.
- **Grass + forbs:** `grass_medium_01`, `grass_medium_02` (mid-distance hero tufts), `grass_bermuda_01`, `fern_02`, `dandelion_01` + `flower_ursinia`/`flower_gazania`/`flower_heliophila` (wildflower slot), `weed_plant_02`. *(Mass near-field grass = procedural `LEAF_STRAP` blade, not a model.)*

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `seed_terrain_bricks` | `game_asset_cook` terrain pass / map load | `crates/vox_render/src/splat_backend.rs` (new) | one-time heightmap→sparse 3D brick field; NET-NEW |
| Sparse brick index + atlas paging | terrain field resident | `crates/vox_render/src/gpu/` (new) | world-keyed bricks; clipmap rings; residents in existing SDF atlas; NET-NEW |
| `apply_terraform_stamp` | `MapTerrain::apply_stamp` (replace heightfield edit) | `~/Ochroma/.../src/map/terrain.rs:122` | 3D CSG into dirty bricks; append edit log; NET-NEW |
| `extract_terrain_skin` (dual contouring) | terraform end-of-stamp + initial build | `crates/vox_render/src/gpu/` (new) | dirty-brick re-extract → chunk mesh; NET-NEW (no brick-input extractor today) |
| `eval_sdf_batch` / per-brick stamp kernel | stamp band recompute | `~/src/spectra/slang/sdf_eval.slang:158` | existing CSG ops + central-diff gradient; reuse for stamp eval |
| `sdf_sample_volume_local` | substrate queries + T3 software floor | `~/src/spectra/slang/megakernel_sdf.slang:96` | already samples a narrow-band volume; bricks reuse it per-brick |
| Ground stems / slugs | provisioner | `src/asset/textures.rs:49`, `src/bin/polyhaven_fetch.rs:28` | `cargo run --bin polyhaven_fetch` |
| `ScatterSlotTable` | scatter field build | `crates/vox_render/src/gpu/scatter_field.rs` | density × clearance-SDF × CellClass, §4.5 |
| Terraform refit loop | `end_stroke` / `water_flips` | `~/Ochroma/.../src/map/terrain.rs` | re-eval+upload dirty bricks → re-extract chunk → TLAS refit |

---

## 8. Open Questions

- [ ] **Brick extent & voxel size:** 8 m brick / 0.25 m voxel (~32³) is the starting proposal — balances atlas slot count against finest sculptable cliff/cave detail. Pin after measuring atlas residency on a town-scale map on the 4070 Ti.
- [ ] **Extraction method:** recommend dual contouring (sharp cliff/cave-mouth edges). Surface nets is the simpler fallback if DC's QEF vertex placement proves fiddly at seams. Decide whether to finish the stub `spectra-flexicubes` into real DC or write a fresh brick-input extractor. (Lean: fresh brick-input DC; the stub's single-dense-field shape is the wrong input.)
- [ ] **Stamp eval on GPU vs CPU:** brush-scale edits touch a handful of 32³ bricks — is `eval_sdf_batch` (GPU) worth the round-trip vs a CPU SIMD band recompute? Measure; the dirty set is tiny.
- [ ] **Clipmap ring policy:** voxel-size doubling per ring, ring radii, and whether sim-active regions (under construction/destruction) force a high-res ring regardless of camera distance.
- [ ] **PolyHaven model fetch:** build the missing `polyhaven_models_fetch` sibling (current fetcher is textures-only) in this design's scope, or stub models behind procedural placeholders until FloraPrime?

---

## 9. Out of Scope

- **Re-seating placed buildings/roads on terraformed ground** (they seat at y≈0 today; terraform design §9 — a later wave).
- **Sub-voxel / micro-detail displacement on the extracted skin** (the brick voxel size bounds geometric detail; finer rock relief is a material/normal-map concern, not this design).
- **FloraPrime QSM+PROSPECT species cooking** — covered by the FloraPrime plan; this design uses PolyHaven bootstrap models in the same placement machine.
- **Non-temperate biome manifests** (desert/boreal/tropical ground+model sets) — temperate only here.

> Note: overhangs and caves as an *interactive terraform brush capability* were OUT of scope in the original draft. The 3D-SDF pivot makes them the **primary in-scope deliverable** (§4.4) — that is the whole point of the correction.

---

## 9b. Phased Plan (3D-SDF-first ordering)

Each phase names files + a measurable Done-When. Phase 1 is the 3D SDF brick representation; terraform (P3) is 3D CSG with surface-extraction re-meshing (P2).

**Phase 1 — 3D SDF brick representation + storage (the foundation).**
Build `TerrainSdfField` (sparse world-keyed bricks over `ReadyAssetSdfVolume`) and `seed_terrain_bricks`; resident bricks in the GPU SDF atlas via a world-coordinate brick index (clipmap rings deferred to P5).
Files: `crates/vox_render/src/splat_backend.rs`, `crates/vox_render/src/gpu/terrain_field.rs` (new).
*Done When:* `cargo test -p vox_render seed_field_from_heightmap` and `terrain_brick_sparsity` GREEN — flat 1 km map allocates one band layer (`O(area/brick_xz²)` bricks), seeded surface SDF is `>0` one voxel above `h(x,z)` and `<0` one voxel below for 100 sampled cells.

**Phase 2 — surface extraction → skin → TLAS (caves/overhangs mesh correctly).**
Net-new brick-input dual-contouring extractor `extract_terrain_skin` (apron overlap for watertight seams); wire extracted chunks as TLAS residents.
Files: `crates/vox_render/src/gpu/terrain_extract.rs` (new); TLAS wiring in the resident frame builder.
*Done When:* `cargo test -p vox_render extract_cave_mesh` GREEN — a brick with a carved cave extracts ≥1 downward-facing-roof triangle (normal·(+Y) < −0.3) and is edge-manifold; `cargo run --bin civitas_care` shows the seeded base terrain as a lit, textured mesh skin at ≥60 FPS (title) on the 4070 Ti.

**Phase 3 — 3D CSG terraform (cliffs + caves, the headline).**
`apply_terraform_stamp` (subtract/union/smooth via `sdf_eval.slang` ops) writing dirty bricks + edit log; per-stamp band recompute + atlas re-upload + dirty-brick re-extract + single-chunk TLAS refit; replace the heightfield brush in `MapTerrain::apply_stamp`.
Files: `~/Ochroma/projects/civitas_care/src/map/terrain.rs:122`, `crates/vox_render/src/gpu/terrain_field.rs`, `crates/vox_render/src/gpu/terrain_extract.rs`.
*Done When:* `cargo test -p vox_render terraform_csg_subtract`, `terraform_csg_overhang`, `terrain_brick_cave_topology`, `extract_dirty_only` all GREEN; and `cargo run --bin civitas_care` Done-When (2)§ acceptance: a human carves a cave into a hillside and flies the camera inside an enclosed void; carves a downward cut and sees a vertical cliff/overhang — both at ≥60 FPS.

**Phase 4 — replay + save.**
Persist the `edit_log`; load = `seed_terrain_bricks` then `replay_edit_log`.
Files: `~/Ochroma/projects/civitas_care/src/map/` save path, `crates/vox_render/src/gpu/terrain_field.rs`.
*Done When:* `cargo test -p civitas_care terraform_replay_roundtrip` GREEN — a map with a carved cave saved+loaded reproduces the cave's multi-crossing ray (≥3 zero-set crossings) bit-for-bit vs pre-save.

**Phase 5 — clipmap paging + perf.**
Concentric LOD rings (coarser voxels outward), camera + sim-active residency; profile dirty-brick edit latency.
Files: `crates/vox_render/src/gpu/terrain_field.rs`.
*Done When:* `cargo test -p vox_render terrain_clipmap_residency` asserts only near-camera bricks are atlas-resident; a measured terraform stamp on a town-scale map stays sub-frame (<16.6 ms end-to-end: stamp eval + upload + re-extract + refit) on the 4070 Ti, printed by the test harness.

**Phase 6 — default scatter (unchanged from §4.5).**
Ground stems/slugs, `ScatterSlotTable`, biome-gated placement — gates now read the 3D field (no scatter inside caves / on overhang underside / on cliff faces).
Files: `src/asset/textures.rs:49`, `src/bin/polyhaven_fetch.rs:28`, `crates/vox_render/src/gpu/scatter_field.rs`.
*Done When:* `cargo test -p vox_render scatter_biome_gate` GREEN with `trees=0 on Water/Slope/cave, grass>500 on flat temperate`; live screenshot diff: flat cell greener than rock/cliff cell.

---

## 10. Related Plans / Designs

- Corrected by: `[3d-sdf-terrain-directive](../../../.claude/projects/-home-tom-espen-src-ochroma/memory/3d-sdf-terrain-directive.md)` (heightfield→3D-SDF pivot)
- Depends on: `[SDF Water](./2026-06-12-sdf-water-design.md)` (render-time water/terrain CSG; dirty-region precedent), `[Terraform Tool](./2026-06-11-terraform-tool-design.md)` (the brush UI/gating, now re-pointed at 3D CSG)
- Reuses (Spectra): `sdf_eval.slang` CSG ops + `eval_sdf_batch`, `megakernel_sdf.slang::sdf_sample_volume_local`, `ReadyAssetSdfVolume` brick format (`asset/mod.rs:372`)
- Extends: `[Scatter Field](./2026-06-12-scatter-field-design.md)`, `[SDF Scene Content Roadmap](./2026-06-12-sdf-scene-content-roadmap.md)`
- Supersedes premise of: `[Hybrid Atom SDF LOD](./2026-06-12-hybrid-atom-sdf-lod-design.md)` (SDF-near/splat-mid), per the hybrid-atom-sdf-lod direction memo
- Related: `[FloraPrime Vegetation Plan](../plans/2026-06-12-floraprime-vegetation-plan.md)`, `[FloraPrime Audit](./2026-06-12-floraprime-audit.md)`
