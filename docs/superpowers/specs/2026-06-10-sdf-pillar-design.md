# Design: The SDF Pillar — one canonical distance-field stack for engine and game (2026-06-10)

**Status:** Draft
**Scope:** Replace the six fragmented, mostly consumer-less SDF representations across vox_physics / vox_render / vox_terrain / urban_horizon / forge / spectra with one canonical three-tier distance-field stack — per-asset dense fields cooked on GPU, an incrementally-composited camera clipmap, and city-scale 2D distance fields — and wire the consumers that make SDFs pay: placement/conform, care-coverage queries, splat-renderer AO + far imposters, destruction, physics, navmesh. Engine crates stay game-agnostic.
**Related:** [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (sibling — SDF imposters/occlusion proxies slot into its far-instance path), [Living Building Instances](./2026-06-10-living-building-instances-design.md) (placement/inspect queries), [SOTA City Block Phase 1 plan](../plans/2026-06-10-sota-city-block-phase1.md)

**Goal axes (the directive, verbatim):** *"SDF is important to me and the reasons are flexibility, power, robustness and speed as SOTA."* Every decision below is argued against **flexibility** (one representation serves placement, physics, rendering, AI, and game fields), **power** (capabilities splat/mesh pipelines cannot express: CSG destruction, geodesic coverage, analytic-quality shadows), **robustness** (signed correctness, validation gates, graceful degradation), and **speed** (GPU generation and queries scoped to the AMD 780M).

---

## 1. Problem Statement

Concrete, observable symptoms (verified 2026-06-10):

- **Six representations, one real consumer.** The codebase carries six incompatible SDF types — `vox_physics::sdf::SdfVolume` (dense f32, CPU), `vox_render::sdf_scene::RenderSdfScene` (flat f32 + POD headers), `vox_terrain::TerrainVolume` (dense f32 + materials), civitas `ReadyAssetSdfVolume` (dense snorm16, ≤24³), forge `SdfSpatial` (dense f32, bounds-based), and spectra's Slang stack (`sdf_eval.slang` analytic op-tree, `physics_sdf_gen.slang` JFA, `physics_sdf_collide.slang`) — yet the only end-to-end consumer of the cooked asset SDF is the **debug binary** `urban_horizon/src/bin/city_coverage.rs` (writes `play_sdf_debug.png` / `play_sdf_raymarch.png`). The play path collects `CityRenderScene.sdfs` and never samples them. `PbfSimulation::cpu_step_with_sdf` exists but has zero callers outside its own tests.
- **The cooked SDF is too coarse to be useful.** `game_asset_cook.rs::ready_sdf` clamps to `SDF_TARGET_MAX_RES = 24` with `voxel_size ≥ 0.20 m` — a 14 m craftsman house gets **0.61 m voxels** (narrow band ±4.9 m). At that resolution contact normals, clearance checks, AO, and crack geodesics are all mush; nothing downstream *can* want it.
- **The cook is CPU brute force and won't survive a resolution bump.** `ready_sdf` is O(voxels × triangles) point-triangle distance plus an O(voxels × triangles) parity-raycast sign (`point_inside_mesh`, single fixed ray, hits deduped within 1e-3 — fragile on coplanar/grazing geometry). Raising res to a useful 64³ on an 8.9k-triangle asset is ≈4.7G triangle ops single-threaded — minutes per asset — while `~/src/spectra/slang/physics_sdf_gen.slang` already implements the 3-pass GPU answer (scatter-min seed → JFA → sign) and sits unused by the cook.
- **Render-side SDF plumbing is upload-only dead ends.** `RenderSdfScene::to_spectra_sdf_layer` → `spectra-scene-upload/src/uploader.rs:327` uploads volume/instance headers + distances to GPU buffers, but **no Slang kernel reads them** (grep over `~/src/spectra/slang/` finds no consumer of `sdf_volume_headers`/`sdf_distances`). `vox_render/src/gpu/volumetric_pass.rs` binds an anonymous "sdf volume" buffer at scatter binding 3 with no defined producer — and its resolve pipeline is built from `scatter_compute.wgsl` with entry point `"scatter_compute"` against a *different* bind-group layout (lines 242–258), a latent bug.
- **Incremental anything is fake.** `vox_terrain::navmesh_bridge::extract_region` "incrementally" re-extracts the **entire volume** then filters by sphere (O(world) per edit). `TerrainVolume` initializes air to `1.0` — distances are non-metric beyond one voxel, so it is an occupancy field wearing an SDF's clothes.
- **The game's signature mechanic uses Euclidean radii.** Care coverage is `coverage_radius_m` circle tests (`game_mechanics/care/service.rs:47`) — a childcare across a river "covers" homes it cannot reach. The engine has no distance-field service to do better, even though a 1024² GPU jump-flood costs well under a millisecond on the 780M.

---

## 2. Done When

**Headline (all milestones landed), on the AMD 780M (RADV):**
running `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin city_coverage` prints

```
[sdf] resident: assets=<N> atlas=<A> MB (quota 64) | clipmap 6 MB | fields <F>x4 MB
[coverage] childcare: reachable <R>% mean <M> m p95 <P> m | build <B> ms (gpu jfa 1024^2)
wrote coverage_childcare.png   (distance-shaded field, roads visible as corridors)
wrote city_sdf_ao.png          (building bases visibly grounded by contact shadow)
wrote play_sdf_raymarch.png    (gpu raymarch, <K> sdf pixels)
```

with **B ≤ 2.0 ms** and **K > 50,000**, and the three PNGs pass eyeball inspection: the coverage field darkens with road distance (not as concentric circles), the AO still shows dark contact bands where buildings meet ground, and the raymarch resolves recognizable building silhouettes. In the game, `cargo run --release --bin play` shows a red footprint plus the HUD line `blocked: clearance 0.4 m` when hovering a ploppable into a building, and green elsewhere. A human at the keyboard verifies all of it without reading code.

**Phased Done When per milestone:**

- **M1 — Canonical type + GPU cook.** `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin game_asset_cook` prints per asset a line shaped exactly like `[sdf] forge.house.craftsman 64x56x51 vox=0.222m band=±0.89m 183 KB sign=gwn closed=yes cook=142ms (gpu)` and a final `[sdf-validate] 5/5 inside-negative OK | |grad|-1 p95 < 0.15 | open-mesh 0` gate line; a deliberately holed test mesh makes the cook exit non-zero printing `[sdf-validate] FAIL <id>: open mesh (|w|=0.43 at centroid)`.
- **M2 — GPU atlas + query parity.** `cargo test -p vox_render --lib sdf_atlas -- --nocapture` prints `[atlas] 5 assets packed, 0.92 MB | gpu-vs-cpu max |Δd| = <D> m over 100000 samples` with **D < voxel/2**, sampled through hardware trilinear filtering against the CPU `SdfVolume::sample_local` oracle on real cooked payloads.
- **M3 — Placement clearance in the game.** In `play`, hovering a clinic ploppable so its footprint overlaps a developed house renders the footprint red and the HUD prints `blocked: clearance <c> m` with c < 1.0; moving 3 m away flips it green. `cargo test --lib placement_clearance -- --nocapture` prints the measured clearance at both probe points (negative inside, > 1.0 outside).
- **M4 — Care coverage fields.** The headline `city_coverage` run prints the `[coverage]` line with build ≤ 2.0 ms, AND plopping a second childcare in `play` changes the info-window line `Childcare reach <R>%` upward within one tick (both values human-visible in the HUD).
- **M5 — Render consumers at city scale.** `cd ~/src/ochroma && cargo run --release --bin scale_trial -- --instanced --buildings 10000 --sdf-shadows` prints `sdf_ao=<X> ms` with **X ≤ 2.0** inside a frame p50 ≤ 16.6 ms, and prints `imposter_raymarch: <H>/<T> rays hit, max |Δt| vs cpu = <E>` with **E < 0.05 m**; `scale_trial_sdf.png` shows contact-darkened building bases (mean luminance in the 0.5 m contact band printed, ≥ 15% below open ground).
- **M6 — Destruction through the field.** `cargo test -p vox_physics --lib sdf_fracture -- --nocapture` prints `crack: <L> m geodesic, <V> voxels carved, debris settled at y=<Y> (surface ±<vox>)` where the crack length L is computed by fast-marching through a real cooked wall SDF from a real impact point, the carve is a CSG subtract visible as a sign flip across ≥ V > 500 voxels, and Y lies within one voxel of the SDF zero crossing.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| GPU cook sign correctness | cook a closed 2 m cube mesh: `assert!(sample(center) < -0.8)` and `assert!(sample(corner+pad) > 0.4)` — extends the existing `ready_sdf_marks_closed_mesh_inside_negative` (game_asset_cook.rs:3271) to the GPU path, both values printed | `assert!(sdf.is_some())` |
| GPU cook ≡ CPU oracle | same mesh through `SdfBaker` and the CPU `ready_sdf`: `assert!(max_abs_dev < voxel * 0.5)` over all voxels in band, dev printed | comparing resolutions only |
| Eikonal validity | 10k random in-band sample pairs: `assert!((grad_len - 1.0).abs() p95 < 0.15)` printed; catches non-metric fields like `TerrainVolume`'s 1.0-air | `assert!(grad.is_finite())` |
| HW-filtered atlas parity | 100k random world points: GPU readback vs `SdfVolume::sample_local`: `assert!(max_dev < voxel/2)`, printed | `assert!(texture exists)` |
| Clearance query | probe inside a cooked house: `assert!(d < 0.0)`; 3 m outside: `assert!(d > 1.0)`; both printed with asset id | `assert!(result.is_some())` |
| Road-aware coverage | wall a childcare off from homes by a solid mask line: `assert!(field_distance > 3.0 * euclidean_distance)` at a home cell, both printed — proves geodesic, not radius | `assert!(distance > 0.0)` |
| Incremental clipmap update | place one building: `assert!(updated_voxels < 0.05 * total)` and re-query at the new facade flips sign; counts printed | full rebuild passing as "update" |
| Imposter raymarch parity | GPU sphere-trace hit t vs CPU `SdfColliderSet::raycast_sdf` over 1k rays: `assert!(max_dev < 0.05)`, printed | `assert!(hits > 0)` |
| CSG carve + crack geodesic | subtract a capsule from a wall SDF: count sign flips `> 500`; fast-march crack length within 10% of hand-computed geodesic, both printed | `assert!(carved)` boolean |
| Field memory quota | load 200 synthetic assets: `assert!(atlas_bytes <= 64 << 20)` with eviction count printed | no-op when under quota |

---

## 4. Architecture

### 4.1 Honest inventory — the fragmented surface this design unifies

| # | Representation | Storage | Producer | Real consumers today | CPU/GPU |
|---|---|---|---|---|---|
| 1 | `vox_physics::sdf::SdfVolume` + `SdfInstance`/`SdfColliderSet` | dense f32 grid, asset-local, narrow-band decode from snorm16 | civitas `spatial_field.rs` decode; tests | `PbfSimulation::cpu_step_with_sdf` (no external callers), `ecs::SdfColliderRuntime`, render packing | CPU |
| 2 | `vox_render::sdf_scene::RenderSdfScene` | flat f32 `distances` + `GpuSdfVolumeHeader`/`GpuSdfInstanceHeader` POD | `from_collider_set` | spectra `SdfLayer` bridge — **uploaded, never sampled by any kernel** | CPU→GPU, unread |
| 3 | `vox_render::gpu::volumetric_pass` froxel scatter | anonymous storage buffer, binding 3 | none defined (caller passes any buffer) | the scatter kernel; resolve pipeline mis-built from the scatter entry point (latent bug) | GPU |
| 4 | `vox_terrain::TerrainVolume` | dense f32 + u8 materials; air = 1.0 (non-metric) | sculpt ops, `generate_demo_volume`, deform/brushes | `navmesh_bridge` (O(world) "incremental"), `volume_to_splats`, vox_app terrain editor | CPU |
| 5 | civitas `ReadyAssetPayload.sdf: ReadyAssetSdfVolume` | dense snorm16, ≤ 24³, voxel ≥ 0.2 m, band = 8·voxel | `game_asset_cook::ready_sdf` (CPU O(V·T), parity sign) | `spatial_field.rs` registry → #1; end-user = `city_coverage` debug bin only | CPU |
| 6 | forge `SdfSpatial` (`~/src/forge/crates/volume/src/spatial.rs`) | dense f32, bounds_min/max, x-major | forge volume generators (heightfield edge treatment) | forge-internal; `SpatialRepr::Sdf` variant | CPU |
| 7 | spectra Slang: `sdf_eval.slang` (analytic op-tree), `physics_sdf_gen.slang` (scatter-min + JFA + parity sign), `physics_sdf_collide.slang` | procedural ops / dense f32 buffers, cube res only | Phorcics physics gen | Phorcics particle projection | GPU |

The unification rule: **one descriptor + one quantized payload (Tier 1), one composite field (Tier 2), one 2D field service (Tier 3)** — everything in the table either becomes a producer into these tiers, a consumer of them, or is deleted. *Flexibility* is the argument: today each consumer would have to pick one of six types and re-implement sampling; after this design every consumer (placement, fluid, AO, coverage, navmesh, destruction) speaks to the same three types through the same query API.

### 4.2 The tier scheme and its memory math (10k buildings, AMD 780M UMA)

The 780M shares system RAM; the sibling virtualized-splat design already budgets ≈ 370 MB for the splat chain at a 1M-atom budget. The SDF pillar's hard quota is **≤ 102 MB (typical ≈ 75 MB)** — about 28% of the splat chain, and it scales with *unique assets*, never with the 10k placed instances (the same dedup argument as the `AssetAtomLibrary`).

**Tier 1 — per-asset dense snorm8 field, cooked once on GPU.**
Resolution: `voxel = clamp(longest_padded_extent / 63, 0.15 m, ∞)`, axis res ≤ 64 (hero assets opt into 96, see below). The craftsman house (12×10×9 m + 1 m pad each side → 14×12×11 m) gets **0.222 m voxels**, grid 64×56×51 ≈ 183k voxels, narrow band ±4 voxels = ±0.89 m, snorm8 step = 0.89/127 ≈ **7 mm** of distance precision.

- Per asset: ≈ 183 KB typical, 256 KB hard cap (64³).
- 10k buildings dedup to ~200 unique catalog assets (starter set today: 5): **200 × 183 KB ≈ 36 MB**. Today's 5 assets: 0.9 MB.
- Hero destruction-enabled assets at 96³ snorm8 = 884 KB, opt-in ≤ 16: +14 MB.
- "Wounded" per-instance copies after destruction (§4.8): ≤ 32 × 256 KB = +8 MB.
- **Tier-1 quota: 64 MB** (atlas-managed, distance-based eviction).
- Counterfactuals that justify the design: per-*instance* cooking at the same res = 10k × 183 KB = **1.8 GB** (impossible); keeping today's f32 flat-buffer render path (`RenderSdfScene.distances`) at the new res = 4 B/voxel = 146 MB for 200 assets, 4× the snorm8 atlas, with no hardware filtering.

**Deliberately dense, not bricked — the honest anti-cargo-cult call.** A building at 64³ is mostly *shell*: ≈ 40% of its voxels lie within the ±0.89 m band (636 m² of facade/floor surface × 1.78 m band thickness ≈ 750 m³ of 1,848 m³ bounds). Sparse 8³ bricks with the 1-voxel filtering apron cost ~1.95× payload per stored voxel, so at 40% occupancy bricks need ≈ 0.78× of dense — a wash that buys two indirections and a packer. Dense wins on **robustness** (no indirection to corrupt, HW trilinear everywhere) and **speed** (one fetch), and loses nothing material on memory at ≤ 64³. Bricks are reconsidered only where occupancy is genuinely low (Tier 2 storage, Open Question #4).

**Tier 2 — composited city clipmap, built incrementally on placement.**
Three concentric levels following the camera, each 128×64×128 voxels (XZ×Y) of snorm8 = 1 MB; voxel sizes 0.25 / 0.5 / 1.0 m → footprints 32×16×32 m, 64×32×64 m, 128×64×128 m. Double-buffered for in-flight updates: **6 MB total, fixed**. Contents = min-composite of terrain + every `SdfInstance` AABB-intersecting the level, evaluated by a compute pass that samples the Tier-1 atlas through each instance's inverse transform (the exact GPU analog of `SdfColliderSet::sample_world`). Updates are *brick-granular dirty regions* (8³ update tiles): placing one house re-composites only the tiles its world AABB touches (< 5% of a level, asserted in §3); camera motion uses toroidal addressing so only newly exposed slabs recompute — this is the UE5 Global Distance Field clipmap discipline, scoped to three small levels. *Power*: cross-asset queries (road conform, debris collision, froxel shadowing) need a world-space field no per-asset SDF can answer alone. *Speed*: every consumer pays one texture fetch instead of N instance transforms.

**Tier 3 — coarse global 2D fields for coverage/audio/AI.**
City extent 2,048×2,048 m at 2 m cells = 1024×1024. One u8 walkability/cost mask (1 MB, rasterized from roads + lots + Tier-2 ground occupancy) and per-field outputs of f16 distance + u16 nearest-seed-id (4 MB/field). Eight resident fields (game decides their meaning; the engine never knows "childcare") = **32 MB + 8 MB transient JFA ping-pong**. A full 1024² jump flood is 10 passes of trivial ALU — **< 1 ms on the 780M** — so fields rebuild lazily on dirty (new facility, new road) rather than streaming. *Flexibility*: the same `DistanceField2d` type later serves pollution, noise, land value, AI flow fields without new machinery.

### 4.3 GPU mesh→SDF cook — scatter-min + JFA + winding-number sign

New engine module `vox_render::gpu::sdf_bake::SdfBaker` (wgpu/WGSL — the cook binary already links vox_render; WGSL keeps the cook free of the Slang toolchain prereq that the spectra path deps carry). It is a port-and-upgrade of spectra's proven `physics_sdf_gen.slang` three-pass structure, generalized from cube-only `u_resolution` to anisotropic `[nx,ny,nz]`:

1. **Seed (scatter-min):** one thread per triangle writes exact closest-point distances into voxels within 2 voxels of the triangle (Ericson closest-point, as the Slang kernel does). Race-y min writes are acceptable for seeding, exactly as documented in the Slang source.
2. **JFA propagation:** log2(64) = 6 passes, 26-neighbor jump flood carrying closest-point coordinates; distances re-derived from carried points, so JFA's approximation error stays ≤ 1 voxel — and the band is then *re-exact*: voxels whose carried point lies within the narrow band recompute true distance to it.
3. **Sign via generalized winding number, not parity.** One thread per voxel sums signed solid angles over all triangles (`w ≥ 0.5` ⇒ inside). 262k voxels × 8.9k tris ≈ 2.3G solid-angle evaluations ≈ 80 GFLOP → **30–80 ms on the 780M** — affordable at cook time, and *robust on the exact failure modes parity has*: coplanar shared edges, grazing rays, slivers. This is the same oracle forge just adopted engine-wide for winding repair (`~/src/forge/crates/building/src/lib.rs::orient_outward` / `generalized_winding_number`), so cook sign and mesh orientation are judged by one mathematical authority. Parity (`RAY_DIR` perturbation, the Slang pass-3 approach) remains as a debug cross-check the validator can print disagreements from.

Scorecard — **GPU JFA cook**: flexibility ★★★ (any mesh, any res, anisotropic); power ★★★ (unlocks 64–96³ where CPU caps at 24³); robustness ★★☆ (JFA ≤ 1-voxel error outside band, exact in band; GWN sign); speed ★★★ (≈ 0.2 s/asset vs ≈ 60–120 s CPU brute force at 64³ — ~500×). **Fits-on-780M: yes** (cook-time, tens of ms per pass).

### 4.4 GPU residency — one snorm8 3D atlas with hardware filtering

`vox_render::gpu::sdf_atlas::SdfAtlasGpu`: a single `R8Snorm` 3D texture (max 512×512×256 = 64 MB at the quota) shelf-packing each asset's volume as a box with a 1-voxel replicated border (filtering never bleeds across assets); a small storage buffer of per-asset headers `{atlas_offset, resolution, origin, voxel_size, narrow_band}` mirrors `GpuSdfVolumeHeader`'s role. R8Snorm is core-filterable in WebGPU, so every in-shader sample is **one hardware trilinear fetch** — this is the entire speed case against the current `RenderSdfScene` flat-f32-buffer path, which would force 8 buffer loads + manual lerp per sample at 4× the memory. Instances stay what they already are: `GpuSdfInstanceHeader` (position/rotation/scale/AABB) uploaded from `SdfColliderSet` — the proven dedup path `RenderSdfScene::from_collider_set` provides, retargeted at texture residency.

Scorecard — **snorm8 atlas + HW filtering**: flexibility ★★★ (every GPU consumer binds one texture + two buffers); power ★★☆; robustness ★★★ (border-padded, quota-evicted, headers validated at insert); speed ★★★ (1 fetch/sample). **Fits-on-780M: yes** — 3D textures and snorm filtering are baseline wgpu/RADV.

### 4.5 Ray-marched queries in wgpu compute

One WGSL library (`sdf_common.wgsl`: header decode, atlas sample, instance world↔local, sphere-trace loop with bounded steps) shared by three consumers:

- **`SdfQueryPass`** — batched point/ray queries (clearance probes, placement previews, AI line-of-sight): upload ≤ 4k queries, one dispatch, readback next frame (one-frame latency is acceptable for previews; the CPU `SdfColliderSet` answers anything that cannot wait). This is the GPU twin of the `SdfQuery` trait.
- **AO / contact shadows** (§4.7-R1) — half-res cone-march against Tier 2.
- **Far-imposter raymarch** (§4.7-R2) — per-pixel sphere-trace against Tier-1 boxes for distant instances.

Scorecard — **wgpu compute raymarch**: flexibility ★★★; power ★★★ (soft penumbrae and AO from the same data physics uses); robustness ★★☆ (bounded 24-step marches, miss = conservative no-hit); speed ★★☆ (bandwidth-bound on UMA; all passes ≤ half-res or distance-gated). **Fits-on-780M: yes, with the stated res caps.**

### 4.6 The 2D distance-transform service (engine) and coverage fields (game)

Engine: `vox_render::gpu::distance_field_2d::DistanceTransform2d` — seeds-from-mask JFA (rg16uint ping-pong, 10 passes at 1024²), output f16 distance + u16 nearest-seed id. Input semantics are opaque to the engine: a *cost mask* (impassable cells = ∞) and *seed cells*. Game (civitas `src/coverage_field.rs`, new): builds the mask from roads/lots/water, seeds from care facilities per service, owns the meaning ("childcare"), feeds `services` reach numbers from field lookups instead of `coverage_radius_m` circles. The JFA over a walkable-cost mask gives **obstacle-aware geodesic distance** — the river finally blocks the clinic. Exact road-graph travel time stays a possible refinement (Open Question #6); the field is the right 90% at 1/1000th the code.

Scorecard — **GPU JFA 2D fields**: flexibility ★★★ (any mask, any seeds — services, pollution, flow); power ★★★ (geodesic coverage is a visible gameplay upgrade in a *care-first* game); robustness ★★★ (JFA is exact for 2D nearest-seed up to ≤ 1-cell error); speed ★★★ (< 1 ms full rebuild — no incrementalism even needed). **Fits-on-780M: trivially.**

### 4.7 Consumer roadmap, ranked by payoff (value against the four axes ÷ effort)

1. **Placement / clearance / conform (game, M3).** `SdfColliderSet::project_point` and `SpatialFieldRegistry::is_solid` already exist with tests — they are simply not called from `place_at`. Wiring clearance into ploppable placement and lot conform (snap a foundation's corners to `sample_world == 0` of the terrain field) is days of work for a permanently-visible robustness win: no building ever intersects another.
2. **Care coverage / service fields (game, M4).** The identity mechanic of Urban Horizon, upgraded from circles to geodesics by §4.6, < 1 ms per rebuild. Highest power-per-effort in the whole design; also the headline demo.
3. **Rendering: contact AO + far imposters (engine, M5).** (R1) Half-res cone-march AO against the Tier-2 clipmap grounds every splat building with contact shadows — the single most visible image-quality gap in current stills, ≤ 2 ms budget. (R2) The virtualized-splat design's far-instance imposters (I0 ≤ 64 atoms / I1 = 1 atom) gain an SDF alternative: sphere-trace the instance's Tier-1 box in the same expand budget slot, giving *exact silhouettes* at any distance from 183 KB shared per asset — and Tier-1 boxes double as occlusion proxies when that design's deferred HZB question is revisited. Composes through buffers; the frozen tiled chain is untouched.
4. **Destruction (engine vox_physics + game, M6).** `FractureSystem`/`spectral_fracture` compute fracture planes today but have no medium to carve. The field is that medium (§4.8) — the biggest pure-*power* item, ranked after R-consumers only because it needs M1+M2 landed first.
5. **Physics queries (engine, continuous).** PBF fluid (`cpu_step_with_sdf`), cloth, and character controllers query `SdfColliderSet` as-is — they inherit the resolution upgrade for free the moment cooked payloads improve, then optionally move to `SdfQueryPass`/clipmap for GPU particle collision (spectra's `physics_sdf_collide.slang` is the proven shape).
6. **Navmesh (engine vox_terrain, last).** Re-point `navmesh_bridge` at Tier 2 so walkability sees *buildings*, not just terrain, and make `extract_region` honestly incremental (re-extract only dirty clipmap tiles — fixing the O(world) lie).

### 4.8 Destruction through the field

A damaged instance is **promoted to unique**: its Tier-1 box is copied into the atlas's wounded region (≤ 32 × 256 KB), and damage becomes a GPU CSG op list applied to the copy — `max(d, -cutter)` subtract ops exactly as `sdf_eval.slang` defines them (capsules along `FracturePlane` intersections, ellipsoids at impacts). Crack propagation is fast-marching through the surface band: seed at the impact voxel, march along `|d| < voxel` voxels with step cost modulated by `spectral_fracture::fracture_threshold` of the local material — the crack path is a *geodesic in the damage-weighted distance field*, something neither splats nor meshes can express. Output: carved SDF (debris collision + re-render), crack polylines (visual splat re-tint + `spectral_shift_on_break`), dirty Tier-2 tiles. Debris simulates against the clipmap (`project_point` per particle). Robustness: CSG on a discrete SDF can only *underestimate* distances (min/max of valid lower bounds), so collision stays conservative — never tunneling.

### 4.9 Robustness — sign correctness, validation gates, graceful degradation

- **Watertightness contract on forge meshes.** Forge meshes are CW-wound with outward normals (`forge_mesh::Mesh` doc, line 5) and the just-landed engine-wide winding repair (`orient_outward` in `~/src/forge/crates/building/src/lib.rs`) drives every triangle to the generalized-winding-number verdict before export. The cook *verifies rather than trusts*: GWN at N=64 stratified interior/exterior probes must read |w| > 0.5 − margin inside and < 0.5 − margin outside; fractional w (open mesh) **fails the cook loudly** (non-zero exit + `[sdf-validate] FAIL <id>: open mesh (|w|=…)`) instead of cooking a garbage sign. An explicit `shell` flag in the payload is the only sanctioned escape hatch (unsigned field, placement-only semantics, never used for inside tests).
- **Sign guarantees downstream.** Tier 1: sign comes from GWN (§4.3) — winding-independent, coplanar-safe; the existing cook test `ready_sdf_marks_closed_mesh_inside_negative` (game_asset_cook.rs:3271) is kept and extended to the GPU path plus a deliberately-holed negative case. Tier 2: min-composition of correctly-signed fields preserves sign by construction. Tier 3: unsigned by definition (distance-to-seed).
- **Metric validity gate.** The eikonal check (`|∇d| ≈ 1`, p95 < 0.15 over in-band samples) runs in the cook validator and in `SdfVolume::from_*` debug builds — it is the test that today's `TerrainVolume` (air = 1.0) fails, making the Tier rebase of terrain mechanically checkable.
- **Quantization honesty.** snorm8 distance step is printed per asset at cook (`band/127`); any asset whose geometric features are thinner than 2 voxels gets a cook warning naming the feature size, so coarse fields are a *visible decision*, not a silent surprise.
- **Degradation order under quota pressure:** wounded copies evict first (re-derivable from op lists), then distant hero volumes downgrade 96³→64³, then distant assets evict entirely (CPU `SdfVolume` remains authoritative for queries — GPU residency is an accelerator, never the source of truth).

### 4.10 Ownership, boundaries, threading

- **`vox_core::sdf` (new)** owns the canonical *representation*: `SdfDesc`, `SdfField` (snorm8/snorm16 payload + decode), sampling math, eikonal/sign validators. Pure types + math, no game concepts — exactly vox_core's charter. `vox_physics::sdf` re-exports these (`pub use`) so `SdfVolume`/`SdfVolumeDesc` paths keep compiling, and keeps the *query* layer (`SdfInstance`, `SdfColliderSet`, `SdfQuery`) it already owns.
- **`vox_render`** owns GPU residency + passes (`sdf_bake`, `sdf_atlas`, `sdf_clipmap`, `sdf_query_pass`, `distance_field_2d`, the AO/imposter WGSL). The froxel volumetric pass's binding-3 buffer gets its first real producer (the clipmap) and its resolve-pipeline entry-point bug is fixed in passing.
- **`vox_terrain`** rebases `TerrainVolume` onto `vox_core::sdf::SdfField` + its material array (a milestone after M2; sculpt API unchanged).
- **Game (`urban_horizon`)** owns the cook invocation (calls the engine `SdfBaker`), `ReadyAssetSdfVolume` v2 (adds `encoding: Snorm8|Snorm16` + `sign: Closed|Shell`, serde-default so v1 payloads load), placement rules, and all service-field *semantics*. Engine crates never learn the words childcare, lot, or road.
- **Forge** keeps `SdfSpatial` as terrain interchange; the loader converts to `SdfField` at import.
- **Spectra** keeps its `SdfLayer` upload path; giving the path tracer a sampling kernel is Open Question #2, not this design's work.
- **Threading:** CPU queries are `&self` on `Arc<SdfVolume>`-holding sets — already `Send + Sync`, callable from sim or render thread. All GPU passes share the one `GpuContext` (`gpu_context()` OnceLock; `new_with_context` constructor pattern per `GpuGi`). Cook is offline single-process. Clipmap updates are encoder-recorded on the render thread; no locks anywhere new.

---

## 5. Data Models

```rust
// crates/vox_core/src/sdf.rs (new) — canonical representation, engine-wide.

/// Grid descriptor shared by every tier-1 field. Mirrors (and replaces the
/// divergent copies of) SdfVolumeDesc / ReadyAssetSdfVolume / SdfSpatial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfDesc {
    resolution: [u32; 3],   // each ≥ 2, ≤ 96
    origin: Vec3,           // asset-local position of voxel (0,0,0)
    voxel_size: f32,        // metres, > 0
    narrow_band: f32,       // metres; payload saturates at ±narrow_band
    sign: SdfSign,          // Closed (GWN-verified) | Shell (unsigned, placement-only)
}
impl SdfDesc {
    pub fn resolution(&self) -> [u32; 3] { self.resolution }
    pub fn voxel_size(&self) -> f32 { self.voxel_size }
    pub fn narrow_band(&self) -> f32 { self.narrow_band }
    pub fn sign(&self) -> SdfSign { self.sign }
    // origin(), sample_count(), local_bounds() …
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfSign { Closed, Shell }

#[derive(Debug, Clone, PartialEq)]
pub enum SdfPayload { Snorm8(Vec<i8>), Snorm16(Vec<i16>) }

/// One asset-local distance field. Construction validates desc/payload and
/// (debug builds) the eikonal property. Immutable after build; share via Arc.
#[derive(Debug, Clone, PartialEq)]
pub struct SdfField {
    desc: SdfDesc,
    payload: SdfPayload,    // X-fastest, then Y, then Z (the existing layout)
}
impl SdfField {
    pub fn desc(&self) -> SdfDesc { self.desc }
    pub fn sample_local(&self, p: Vec3) -> Option<f32>;   // trilinear, decoded metres
    pub fn normal_local(&self, p: Vec3) -> Option<Vec3>;
    pub fn payload_bytes(&self) -> &[u8];                  // for atlas upload
}

/// Validation report printed by the cook gate (§4.9).
#[derive(Debug, Clone, PartialEq)]
pub struct SdfValidation {
    inside_negative: bool,
    eikonal_p95: f32,          // | |∇d| − 1 |, in-band samples
    winding_min_abs: f32,      // min |w| over interior probes; < 0.5 ⇒ open mesh
    open_mesh: bool,
}
// accessors: inside_negative(), eikonal_p95(), open_mesh() …
```

```rust
// crates/vox_render/src/gpu/sdf_atlas.rs (new)

/// GPU-side header, 48 bytes, std430-compatible. One per resident asset field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct GpuSdfFieldHeader {
    atlas_offset_voxels: [u32; 3], _pad0: u32,
    resolution:          [u32; 3], _pad1: u32,
    origin_voxel:        [f32; 4],  // origin.xyz, voxel_size
    band_sign:           [f32; 4],  // narrow_band, sign(±1), 0, 0
}

/// Owns the R8Snorm 3D atlas texture + header buffer. Render-thread-owned.
pub struct SdfAtlasGpu { /* texture, headers, shelf allocator, quota */ }

/// Three-level camera-following composite field (§4.2 Tier 2). 6 MB fixed.
pub struct SdfClipmap { /* 3 × R8Snorm 3D textures + level uniforms */ }

/// 2D distance transform output (§4.6 Tier 3). Engine-semantic-free.
pub struct DistanceField2d { /* f16 distance + u16 seed-id textures, 1024² */ }
```

`SdfColliderSet`, `SdfInstance`, `SdfVolumeCache`, `RenderSdfScene`'s instance headers, and civitas `SpatialFieldRegistry` keep their existing shapes — they re-point at `vox_core::sdf` types via `pub use` (the `SdfVolume = SdfField` aliasing is the unification, not a rewrite of the query layer that already works).

---

## 6. API

```rust
// ── vox_render::gpu::sdf_bake (new; called by the game cook) ────────────────
impl SdfBaker {
    /// Shared-context constructor (GpuGi::new_with_context precedent).
    pub fn new_with_context(ctx: &GpuContext) -> Result<Self, SdfBakeError>;
    /// Mesh → SdfField on GPU: scatter-min seed, JFA, GWN sign, band re-exact,
    /// snorm8 encode, validation probes. Blocking (cook-time tool).
    /// Errors: DegenerateMesh, OpenMesh { min_abs_winding: f32 }, DeviceLimits.
    pub fn bake(&mut self, positions: &[[f32; 3]], indices: &[[u32; 3]],
                target: SdfBakeTarget) -> Result<(SdfField, SdfValidation), SdfBakeError>;
}
pub struct SdfBakeTarget { pub max_axis_res: u32 /* 64 | 96 */, pub min_voxel: f32,
                           pub pad: f32, pub band_voxels: f32 }
// Threading: cook/tool thread; owns its encoder submissions; no frame coupling.

// ── vox_render::gpu::sdf_atlas ───────────────────────────────────────────────
impl SdfAtlasGpu {
    pub fn new_with_context(ctx: &GpuContext, quota_bytes: u64) -> Result<Self, SdfAtlasError>;
    /// Upload (or return resident) field; evicts per §4.9 order when over quota.
    pub fn insert(&mut self, asset: SdfAssetId, field: &SdfField) -> Result<u32, SdfAtlasError>;
    pub fn resident_bytes(&self) -> u64;
    pub fn texture(&self) -> &wgpu::Texture;          // bind in consumer passes
    pub fn header_buf(&self) -> &wgpu::Buffer;
}

// ── vox_render::gpu::sdf_clipmap ─────────────────────────────────────────────
impl SdfClipmap {
    pub fn new_with_context(ctx: &GpuContext) -> Result<Self, SdfClipmapError>;
    /// Re-center on camera (toroidal) and re-composite dirty tiles only.
    /// `instances` = the same GpuSdfInstanceHeader set RenderSdfScene packs.
    pub fn encode_update(&mut self, encoder: &mut wgpu::CommandEncoder,
                         camera_pos: Vec3, atlas: &SdfAtlasGpu,
                         instances: &wgpu::Buffer, instance_count: u32,
                         dirty_world_aabbs: &[(Vec3, Vec3)]) -> ClipmapUpdateStats;
}
pub struct ClipmapUpdateStats { pub tiles_updated: u32, pub tiles_total: u32 }

// ── vox_render::gpu::distance_field_2d ──────────────────────────────────────
impl DistanceTransform2d {
    pub fn new_with_context(ctx: &GpuContext, size: u32 /* 1024 */) -> Result<Self, Df2dError>;
    /// mask: u8 cost grid (255 = impassable). seeds: cell indices. ~10 JFA passes.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder,
                  mask: &wgpu::Texture, seeds: &[[u32; 2]]) -> ();
    pub fn distance_texture(&self) -> &wgpu::TextureView;   // f16 metres
    pub fn seed_id_texture(&self) -> &wgpu::TextureView;    // u16 nearest seed
    /// CPU readback for sim-side queries (coverage %, per-home distance).
    pub fn read_back(&self, ctx: &GpuContext) -> Df2dReadback;
}

// ── existing signatures relied on verbatim (do not re-derive) ───────────────
// vox_physics::sdf::SdfVolume::from_snorm16_grid(desc, &[i16]) -> Result<Self, SdfVolumeError>
// vox_physics::sdf::SdfColliderSet::{sample_world(Vec3) -> Option<SdfSample>,
//     project_point(Vec3, clearance: f32) -> Option<SdfContact>,
//     raycast_sdf(origin, dir, max_d, eps) -> Option<SdfRayHit>}
// vox_physics::sdf trait SdfQuery { sample_world, normal_world, project_point, raycast_sdf }
// vox_render::sdf_scene::RenderSdfScene::from_collider_set(&SdfColliderSet) -> Self
// civitas spatial_field::{SpatialFieldRegistry::is_solid([f32;3], f32) -> bool,
//     stable_sdf_asset_id(&str) -> SdfAssetId, ready_asset_sdf_desc(&ReadyAssetSdfVolume)}
// civitas asset::ReadyAssetSdfVolume { resolution: [u16;3], origin: [f32;3],
//     voxel_size: f32, narrow_band: f32, distances_snorm16: Vec<i16> } + decode_distance(i16)
// GpuContext::from_parts(&wgpu::Device, &wgpu::Queue, &wgpu::AdapterInfo) -> Self
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `SdfBaker::bake` | `ready_sdf` (replaces the CPU O(V·T) body; CPU path stays as the test oracle) | `urban_horizon/src/bin/game_asset_cook.rs:1450` | per recipe in the cook loop (line ~65); prints the M1 `[sdf]` + `[sdf-validate]` lines |
| `vox_core::sdf` re-exports | `pub use` shims | `crates/vox_physics/src/sdf.rs`, `crates/vox_render/src/sdf_scene.rs` | existing paths keep compiling; no consumer churn |
| `SdfAtlasGpu::insert` | `SceneRenderer::set_city_scene` / asset residency tick | `urban_horizon/src/render_gpu/mod.rs` (~2280) | from the same `SpatialFieldRegistry` instances; once per unique asset |
| `SdfClipmap::encode_update` | per-frame before AO, dirty AABBs from develop/plop/destroy | `urban_horizon/src/render_gpu/mod.rs` `SceneRenderer::render` | camera recenter + ≤ 5% tile updates (§3 gate) |
| Clearance check | ploppable hover + `place_at` gate | `urban_horizon/src/bin/play.rs` (MouseInput chain) + `src/game/mod.rs` | `SpatialFieldRegistry::is_solid(probe, 1.0)` — CPU, no GPU latency |
| Lot/road conform | foundation snap at develop time | `urban_horizon/src/render_gpu/mod.rs` `place_ready_asset_*` | terrain-field `project_point` per foundation corner |
| `DistanceTransform2d::encode` | coverage rebuild on facility/road dirty flag | new `urban_horizon/src/coverage_field.rs`, ticked from `CivitasGame::tick` | per-service seeds; `read_back` feeds `services` reach numbers |
| Coverage HUD line | `GameView::info_lines` | `urban_horizon/src/bin/play.rs:322` | `Childcare reach <R>%` from field readback (M4 gate) |
| SDF AO pass | after expand, before atmosphere composite | `urban_horizon/src/render_gpu/mod.rs` + `crates/vox_render/src/gpu/` | half-res march vs clipmap; `sdf_ao` ms printed by scale_trial (M5) |
| Imposter raymarch hook | far-instance branch of the virtualized expand path | per [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) §4.5 | budget slot unchanged; SDF hit → splat-compatible fragment |
| Froxel binding-3 producer + resolve-entry-point fix | `VolumetricPass::dispatch_scatter` callers | `crates/vox_render/src/gpu/volumetric_pass.rs:242` | clipmap buffer becomes the defined producer; fix the resolve pipeline's `scatter_compute` entry point |
| CSG carve + fast-march crack | impact handler in `FractureSystem::apply_impact` | `crates/vox_physics/src/destruction.rs:64` + new `sdf_fracture.rs` | wounded-copy promotion; dirty AABBs → clipmap |
| `cpu_step_with_sdf` colliders | fluid demo/bench wiring (first real caller) | `crates/vox_physics/src/pbf.rs:90` ← `vox_app` demo scene | inherits cooked-res upgrade for free |
| Navmesh from clipmap | `navmesh_bridge` rebase + honest `extract_region` | `crates/vox_terrain/src/navmesh_bridge.rs:79` | dirty-tile re-extraction only (kills the O(world) filter) |

---

## 8. Open Questions

- [ ] **snorm16 hero fields:** destruction carving accumulates quantization; do ≤ 16 hero assets need snorm16 (2× memory, 0.03 mm steps) or does snorm8 + op-list re-evaluation (re-carve from pristine each promotion) hold up? Decide from M6's carve-error print.
- [ ] **Spectra-native sampling kernel:** the `SdfLayer` upload exists end-to-end with no Slang reader. Wire `sample_scene_sdf` into the path tracer (occlusion proxies for stills) when the game view routes through Spectra (per the staged plan in memory) — or delete the upload to stop paying for a dead end. Decide at M5.
- [ ] **Clipmap Y extent:** 64 voxels of Y (16/32/64 m) caps composited height; towers above that exist only in Tier 1. Acceptable for Civitas' current asset set — re-measure when high-density assets land.
- [ ] **Tier-2 sparse storage:** if profiling shows the city's air fraction makes dense clipmap levels wasteful at larger footprints, revisit 8³-brick storage for L2 only (the occupancy math in §4.2 that rejects bricks for Tier 1 flips in favor below ~25%).
- [ ] **Exact road-graph coverage:** grid JFA over the walkable mask approximates network distance (no travel speed, diagonal error ≤ ~8%). If care-balance design needs travel *time*, layer a road-graph Dijkstra seeding pass that stamps edge times into the mask — field machinery unchanged.
- [ ] **TerrainVolume rebase sequencing:** after M2 (types proven) but independent of M3–M6; needs its own small plan because sculpt ops and the material channel must survive the move.

---

## 9. Out of Scope

- **SDF global illumination (SDFGI/Lumen-style):** GI stays on the resident splat path (`ResidentGiRaster`); SDFs feed it nothing in this design.
- **Neural SDFs, hash-grid fields, NanoVDB/OpenVDB adoption:** the tiers above hit the memory/speed targets with one indirection level or none; a full VDB tree is unjustified complexity at ≤ 96³ assets and a 6 MB clipmap.
- **Hardware sparse/tiled textures:** not exposed by wgpu; quota + eviction does the job.
- **Fluid surface reconstruction to SDF** (marching PBF particles into a field) — future; `marching_cubes.slang` exists in spectra when it's wanted.
- **CSG authoring/modeling tools** — `sdf_eval.slang`'s op-tree remains a runtime damage representation, not an editor feature.
- **Changing the frozen tiled splat chain or the Spectra path tracer's render loop** — all rendering consumers compose via buffers/textures, per the sibling design's contract.
- **Save-format work:** wounded-instance op lists ride the existing replay-log persistence (deterministic re-carve), no schema change.

---

## 10. Related Plans / Designs

- Depends on: nothing unshipped — binds to `vox_physics::sdf` (shipped, tested), `RenderSdfScene` (shipped), the cook's `ready_sdf` path + `ready_sdf_marks_closed_mesh_inside_negative` test, forge's `orient_outward` winding repair, spectra's `physics_sdf_gen.slang` as the ported algorithm reference, `GpuContext::from_parts` shared-device pattern.
- Required before: the SDF Pillar implementation plan (M1–M6, one plan per milestone pair), the destruction plan (M6 expands into vox_physics), the terrain rebase plan (Open Question #6).
- Related: [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (imposter slot + occlusion proxies + the 370 MB UMA budget this pillar fits beside), [Living Building Instances](./2026-06-10-living-building-instances-design.md) (placement/inspect picking can upgrade from rotated-rect to SDF clearance), `aaa-phase2-resident-frame` memory (GpuContext/resident-pass doctrine), [Atom-Budget Splat Renderer](./2026-06-06-atom-budget-splat-renderer-design.md) (the dedup-by-unique-asset philosophy Tier 1 mirrors).
