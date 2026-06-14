# Design: Hybrid Atom + SDF LOD Rendering Pipeline (2026-06-12)

**Status:** Draft
**Scope:** Unify the two pillars built this week — virtualized atom splatting (M1–M3.2) and the GPU-baked signed SDF (SDF Pillar M1) — into ONE screen-coverage-driven LOD pipeline that resolves the FIX-3 sparse-atom finding: cooked building atoms are a sparse voxel point-cloud that splat as *confetti, not surfaces*. The resolution is a three-tier contract — **T0** sphere-traces the per-asset baked SDF for solid near-field surfaces, **T1** keeps the M1–M3.2 instanced atom splats for mid coverage, **T2** is aggregate imposters far — composited in one frame under one `FrameBudgetGovernor`. Engine crates (`vox_core`/`vox_render`) stay game-agnostic; the game supplies assets, instances, and lighting only.
**Related:** [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (the T1 tier — `InstancedSelector` L0–L3, `select_resident`, `ExpandDrawsPass`, the frozen tiled chain, `FrameBudgetGovernor`); [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) (the T0 substrate — `SdfBaker` GWN-signed snorm8 cook, `SdfAtlasGpu` hardware-trilinear atlas, `SdfAtlasProbe`); sibling [Scatter Field design](./2026-06-12-scatter-field-design.md) (shares this tier contract for grass/props); direction memory `hybrid-atom-sdf-lod-direction`.

---

## 1. Problem Statement

Concrete, observable symptoms (audited 2026-06-12; the FIX-3 finding):

- **Cooked building atoms render as confetti through the splat path, never a surface.** One building's **15,272 atoms** through the plain instanced splat chain (`InstancedSelector` → `ExpandDrawsPass` → `TiledSplatRenderer`) is a sparse Gaussian point-cloud at a ~0.15–0.30 m pitch — gaps between atoms exceed a pixel at any near camera, so the façade reads as scattered fragments, not a wall. The `--shot-m4` parity gate measures this as low instanced coverage versus the legacy mesh frame at the same camera.
- **Everything that looks solid today renders MESHES, not atoms.** The good-looking interactive city uses the legacy triangle-mesh hybrid path in `civitas_care`; the Spectra path-traced stills (`pathtrace_splats_to_rgba`) trace meshes. The scalable instanced-**atom** path (M1–M3.2) and the solid-**mesh** path are disjoint code: one is fast and sparse, the other is solid and unscalable. Nothing that is both scalable AND solid exists.
- **The splat raster writes no depth, so it cannot occlude or be occluded by a second geometry source.** `splat_raster.wgsl` stores only a 4-layer `Rgba32Float` texture — layers 0/1 = 8 spectral bands, layer 3 = transmittance; **layer 2 is written by no pass and read by no path** (`tiled_splat_renderer.rs:155`). `SpectralPresenter` keys "geometry present" off `channel_sum > 18` (`spectral_present.wgsl:92`). There is no per-pixel depth to reconcile a near SDF surface against mid splats.
- **The baked SDF is geometry-only — distance, no color.** `SdfField` is a snorm8 distance payload (`vox_core::sdf`); `SdfAtlasProbe` returns `distance = sample * narrow_band` and a gradient-normal. A T0 sphere-trace hit has a position and a normal but **no albedo, no spectral color** — and a city of grey untextured surfaces is not shippable.
- **M4's honest `--shot-m4` coverage failure has no fix on the atom path.** Densifying atoms to fill the façade re-creates the 50M-world-atom memory blowup the whole virtualization effort exists to avoid (Virtualized §1). The coverage gap is structural to splatting a sparse point set up close — it cannot be closed by spending more atoms.

---

## 2. Done When

**Headline (all milestones landed), on the AMD 780M (RADV); the 4070 Ti is the target where SDF-march throughput climbs:**

running `cd ~/src/ochroma && cargo run --release --bin scale_trial -- --hybrid --buildings 10000 --shot scale_trial_hybrid.png` prints

```
[hybrid] T0 sdf-march: instances=<N0> facade_coverage=<C0≥0.97> march_ms=<M0> p50 | T1 splats: selected=<S>/<B> | T2 imposters: instances=<N2>
[hybrid] frame p50=<X≤16.6> ms p99=<Y> ms @ 1280x720 | governor budget=<B> ema=<E> ms | renderer constructed 1x
[hybrid] occlusion: near-sdf-over-mid-splat pixels=<Ko>0> | mid-splat-over-far-imposter pixels=<Kf>0>
```

with **X ≤ 16.6** (60 fps) and **C0 ≥ 0.97** (the near building's façade is filled, where the plain atom path printed confetti). `scale_trial_hybrid.png` shows a human a city where the nearest buildings are **solid, gap-free walls** (not scattered points), mid buildings are recognizable splatted massing, and far buildings are colored boxes — with correct occlusion at every tier seam. A human at the keyboard verifies all of this without reading code.

**Phased Done When per milestone** (each independently demo-able; later depend on earlier):

- **M1 — One building SDF-marched solid (the FIX-3 proof).** Done When `cargo test -p vox_render --lib sdf_march_facade -- --nocapture` cooks one real 15,272-atom building's mesh to an `SdfField`, residents it in `SdfAtlasGpu`, sphere-traces it at a 12 m near camera, and prints `[sdf_march] facade_coverage=<C0> (sdf) vs <Ca> (atom-splat same camera) | hit_pixels=<H> normal_valid=<V%>` with **C0 ≥ 0.97 AND C0 ≥ 3·Ca** — the SDF fills the façade where the atom splat of the *same building, same camera* gives confetti, both coverage fractions computed and printed.
- **M2 — Hybrid frame: T0 + T1 + T2 composited with correct occlusion.** Done When `cargo test -p vox_render --lib hybrid_composite -- --nocapture` renders a scene with one near SDF-marched building partially in front of a mid splatted building in front of a far imposter, reads back the composite, and prints `[hybrid] sdf_over_splat=<Ko>0> splat_over_sdf=<Ks>0> splat_over_imposter=<Kf>0> depth_seam_artifacts=<A==0>` — proving each tier correctly occludes and is occluded by its neighbors per the shared depth buffer (no tier draws over a nearer tier; `A` counts pixels where the farther source wins and must be 0).
- **M3 — The budget governor over all three tiers at city scale.** Done When the headline `scale_trial --hybrid --buildings 10000` line holds on the 780M with `X ≤ 16.6`, AND a second run `--buildings 1000` prints a strictly smaller p50 (the unified governor idles below the cap), AND a printed `[hybrid] budget split: sdf_march_ms=<m0> splat_ms=<m1> imposter_ms=<m2> sum≈frame` showing the governor accounting all three costs against one frame budget — all in one run.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| SDF fills the façade atoms leave sparse | cook one 15,272-atom building, sphere-trace at 12 m: `assert!(facade_coverage >= 0.97 && facade_coverage >= 3.0 * atom_coverage_same_camera)`, both printed | `assert!(hit_pixels > 0)` |
| T0 hit carries a real surface color | sphere-trace a hit, resolve color via the chosen path (§4.3): `assert!((rgb - nearest_atom_spectral_rgb).length() < 0.15)` on a hand-placed atom field, printed | `assert!(color != black)` |
| Per-pixel depth written by the splat raster | render 2 splats at known view depths, read back the new depth layer: `assert!((depth_px - expected_view_z).abs() < 0.01)`, printed | `assert!(layer2_written)` |
| Near SDF occludes mid splat | place SDF surface at depth 5 m over a splat at 20 m: `assert!(composite_pixel == sdf_color)` for the overlap, count printed | `assert!(composite.is_ok())` |
| Mid splat occludes far imposter | splat at 60 m over imposter box at 500 m: `assert!(overlap_pixel == splat_color)`, count printed | comparing only lengths |
| Tier selection by screen coverage | one instance at 12 m → tier T0; same at 80 m → T1; at 500 m → T2; `assert_eq!(tier(12.0), T0)` etc., all three printed | `assert!(tier(12.0) != tier(500.0))` |
| Unified budget accounts SDF + splat cost | run governor with a 2-term cost model (per-pixel march + per-atom splat); `assert!((settled_ms - target).abs() < 0.1·target)`, both printed | `assert!(budget <= cap)` |
| One renderer at city scale | 10k buildings, orbit + plop: `assert_eq!(constructions, 1)` and `p50 <= 16.6`, both printed | `assert!(render.is_ok())` per frame |

---

## 4. Architecture

### 4.1 Honest inventory — what the two pillars give us, and the exact seam that is missing

**T1 substrate (shipped, frozen — the Virtualized pillar).** `InstancedSelector` (CPU oracle) / `InstancedSelectGpu::select_resident` (GPU-resident walk, bit-identical via the shared `budget_walk_and_emit`) classify each placement cull/far/near with LOD levels L0–L3 + imposters I0/I1, emit prefix-summed `ClusterDraw`s; `ExpandDrawsPass::encode_indirect` transforms the selected library atoms into `TiledSplatRenderer`'s persistent `splat_buf`/`transform_buf`; the frozen four-pass chain (`tile_assign → radix_sort → tile_range_build → splat_raster`) rasters them emissively into the 4-layer `Rgba32Float` texture. `FrameBudgetGovernor` (`atom_instances.rs:984`) holds 60 fps by modulating the per-frame atom budget. **This is T1 and T2** — the M1–M3.2 imposter levels I0 (≤ 64 atoms) / I1 (1 atom) ARE the T2 aggregate-imposter mechanism, already in the selector.

**T0 substrate (shipped — the SDF pillar M1).** `SdfBaker` cooks any sealed Forge mesh to a 64³ snorm8 `SdfField` with GWN-correct sign (Forge meshes are now SEALED watertight, so signs are correct — SDF pillar §4.9); `SdfAtlasGpu` residents up to ~200 unique assets in one `R8Snorm` 3D atlas (36 MB at 64³) with hardware trilinear filtering; `SdfAtlasProbe` is the proven first consumer — `g = (p - origin)/voxel`, `uvw = (atlas_offset + g + 0.5)/dims`, `d = textureSampleLevel(...).r * narrow_band`. **This is the T0 surface source.**

**The exact missing seam (what this design adds):** ONE `HybridFramePass` that (1) extends `splat_raster` to write a per-pixel splat depth into the **already-allocated, currently-unused output texture layer 2**; (2) adds a `SdfMarchPass` that sphere-traces the T0 instances against the atlas, writing surface spectral color + view-space depth into the SAME 4-layer texture under a depth test; (3) selects the tier per instance by screen coverage inside the existing instance pass; (4) drives all three tiers from ONE `FrameBudgetGovernor` with a two-term cost model. No new color pipeline, no new present path — the composite lands in the same texture `SpectralPresenter`/`composite_lit` already reads.

### 4.2 Tier selection — extending the instance pass with a T0 cut (resolves Q3, half)

The Virtualized instance pass (`InstancedSelector::select`, `atom_instances.rs:639`) already computes, per visible instance, `world_center`, `inst_distance`, and a projected screen size. Today it branches **far** (`inst_distance ≥ FAR_INSTANCE_M = 150 m` → imposter chain) vs **near** (→ per-cluster L0–L3). We add ONE coarser cut at the front, keyed on **screen coverage**, not raw distance, because coverage is what makes atoms look sparse:

```
coverage = projected_radius_px(world_radius, inst_distance, viewport) / min(width, height)
tier =  T0  if coverage ≥ T0_COVERAGE (default 0.18 — a few objects fill the screen)
        T1  if coverage ≥ T1_COVERAGE (default 0.012)
        T2  otherwise
```

- **T0 (≥ ~18% of the smaller screen axis):** the instance is sphere-traced (§4.4); it emits **zero** atom draws. This is the bounded set the SDF-march budget pays for — at a street camera typically 1–8 buildings, never the city.
- **T1:** the instance emits its per-cluster L0–L3 work units exactly as today — the M1–M3.2 path is the **correct mid tier, unchanged**.
- **T2:** the instance emits its I0/I1 imposter chain exactly as today.

`T0_COVERAGE` replaces *nothing* in the existing far/near logic — it sits above it: an instance that is not T0 falls through to the existing `FAR_INSTANCE_M` near/far split (T1 vs T2). So the diff to `InstancedSelector` is one classification branch and a new `tier: Tier` field on the instance-pass output; the budget walk over T1/T2 work units is byte-identical. **This is the "2026 LOD" cut translated honestly:** like LODGE/CLoD-GS (continuous splat-count LOD by camera distance) for T1/T2, but with an *implicit-surface* near tier (NGLOD/sphere-tracing lineage) that splat LOD fundamentally cannot provide — exactly the hybrid-SDF-Gaussian direction the 2024–25 literature converges on (Gaussians aligned to an SDF zero-level set).

### 4.3 Color on the SDF surface — THE CRUX (resolves Q2)

A T0 sphere-trace hit has `hit_pos` (world) and `normal` (decoded SDF gradient) but no color. Four options, with the arithmetic:

**(a) Bake a colored SDF — per-voxel albedo alongside distance.** A 64³ field with an RGB (or 8-bin spectral) channel beside snorm8 distance. Cost: distance is 1 B/voxel = 256 KB at 64³ (the shipped atlas); adding RGBA8 albedo is **+4 B/voxel = +1 MB/asset**, ×200 unique assets = **+200 MB** — over 5× the entire 36 MB distance atlas, blowing the 64 MB Tier-1 quota by 3×. Rejected on memory; also the cook would have to project the atoms' spectral color into a dense grid, re-introducing a sparse→dense fill at cook time.

**(b) Sample the nearest atoms' spectral color at the hit — atoms become the color field, SDF the surface (the genuine hybrid).** The atoms we already cook and resident in the T1 `AssetAtomLibrary` (`packed_atoms`, 64 B each, 8-bin spectral) ARE a colored point cloud of the asset surface. At a T0 hit, gather the *k* nearest library atoms (asset-local, via a small per-asset uniform-grid acceleration built at library cook) and blend their spectral color by Gaussian weight on distance-to-hit. Cost: **zero new storage** — reuses `packed_atoms` already resident for T1; per-hit cost is a *k*-NN gather (k = 4–8) over a coarse grid, a handful of texture/​buffer reads per traced pixel. The atoms are *too sparse to splat as a surface* but *perfectly dense to color one* — a hit on the gap-free SDF wall samples the same atoms that would have been confetti, and reads their color. This is precisely the 2024–25 "explicit Gaussians aligned to the SDF zero-level set" hybrid: SDF carries topology/silhouette, Gaussians carry appearance.

**(c) Triplanar-project the material-zone textures onto the SDF surface.** Sample the asset's PBR/material-zone textures by world-or-local position blended across the three axis projections weighted by `normal²`. Cost: requires the material-zone textures resident on GPU (they are a cook output, game-side) and a per-zone classification at the hit — more plumbing, and the game's material-zone semantics would leak toward the engine. Strong for tiled materials, but heavier and game-coupled.

**Decision: (b) — atoms color, SDF surfaces.** It is the only option with **zero added atlas cost** (the 200 MB in (a) is disqualifying on the 780M's shared RAM beside the 370 MB splat chain + 36 MB SDF atlas), it reuses data already resident for T1, it keeps the engine game-agnostic (no material-zone semantics), and it is the architecturally honest unification — the two pillars become one field: *the SDF is the surface, the atoms are its color*. The k-NN gather is bounded (coarse asset-local grid, k ≤ 8, only at traced T0 pixels, of which there are few by construction). The fallback when a hit's gather is empty (degenerate atom-free region) is the SDF gradient-normal shaded with the asset's I1 mean spectral color — never black. **Honest cost note:** the k-NN color is lower-frequency than per-texel sampling; fine texture detail on a near wall is softer than a Spectra still — recorded, not hidden. Sharper near-field texture is a future option-(c) overlay, sequenced after the surface is solid.

### 4.4 `SdfMarchPass` — T0 sphere-trace into the shared texture (resolves Q1, the depth + color write)

A new `vox_render::gpu::sdf_march::SdfMarchPass` (WGSL compute, shared `GpuContext` per the `new_with_context` house pattern). Per T0 instance the host uploads a `T0Instance { inv_transform: mat3x4, header_slot: u32, asset: u32 }`. One thread per screen pixel in each T0 instance's projected AABB (a coarse tile dispatch, not full-screen — T0 instances are few):

1. Reconstruct the world ray from the pixel NDC and the camera `inv_view_proj` (the exact recipe `spectral_present.wgsl::backdrop_color` already uses).
2. Transform ray origin+dir into the instance's asset-local space (`inv_transform`).
3. Sphere-trace the atlas field (bounded 24–48 steps; `d = textureSampleLevel(atlas, samp, uvw, 0).r * band` — the `SdfAtlasProbe` sample, in a march loop), stop at `|d| < hit_eps`.
4. On hit: gradient-normal by central differences on the field; resolve spectral color by the §4.3(b) k-NN atom gather; compute **view-space depth** of the hit (`-( view * world_hit ).z`).
5. **Depth-tested store into the shared 4-layer texture:** read the current depth at the pixel from **layer 2**; if the hit is nearer, store the 8 spectral bands into layers 0/1, the new depth into layer 2, and leave layer 3 (transmittance) at 0 (a solid SDF surface is fully opaque — `transmittance = 0` means "no background shows here", which is exactly how `composite_lit` already treats covered pixels).

The march runs **before** the splat chain in the frame, seeding layer 2 with near-surface depths; the splat raster then runs with the depth-test extension (§4.5) so mid splats behind a near SDF wall are killed, and splats in *front* of the SDF surface overwrite it correctly. This is the textbook depth-buffer-fusion of a raymarched and a rasterized source (write `gl_FragDepth`-equivalent from both, compare per pixel) — grounded SOTA, not invention.

### 4.5 Depth reconciliation — the one pinned sort order (resolves Q1, the contract)

The pillars compose through **one shared per-pixel depth buffer = output-texture layer 2** (free today). The pinned order:

| Step | Pass | Depth behavior |
|---|---|---|
| 0 | Clear | layer 2 ← `+INF` (far), layers 0/1/3 ← 0 (the existing zero-fill, extended to layer 2) |
| 1 | `SdfMarchPass` (T0) | per pixel: if `march_depth < layer2`, write color (0/1) + depth (2), transmittance (3)←0 |
| 2 | `splat_raster` (T1+T2) | **extended:** before accumulating a splat into a pixel, if `splat_view_depth > layer2` (behind a solid SDF surface) the splat's contribution is skipped for that pixel; front-to-back accumulation otherwise proceeds unchanged; at pixel finish, if the accumulated splat front is **nearer** than the SDF depth, the splat color replaces (composites over) — handled by the existing transmittance blend since the SDF wrote `transmittance=0` and the splats blend front-to-back from the camera |

The splat raster already iterates **front-to-back within each tile using pre-sorted indices** (`splat_raster.wgsl:4`) and already has `position_depth.w = view depth` written by `tile_assign` (`tile_assign.wgsl:189`). The extension is: (i) the per-splat depth is already available; the raster gains a single compare against `layer2` to early-out splats behind the SDF surface, and (ii) writes the *nearest accumulated splat depth* back to layer 2 so a near splat correctly occludes a farther T2 imposter splat in the same buffer. T2 imposters are themselves splats in the same chain, so **mid-vs-far is the chain's existing front-to-back sort** — no new mechanism. The only genuinely new reconciliation is **T0(SDF) vs T1/T2(splats)**, and it is one depth compare against a buffer that already exists physically.

**Honesty rail (carried from the splat chain):** the radix sort is non-deterministic on depth ties (the known landmine, `aaa-phase2-resident-frame` memory), so the M2 occlusion gate asserts on **aggregate pixel counts and the absence of farther-source-wins artifacts** (`A == 0`), never bit-exact framebuffers.

### 4.6 The unified budget — one governor, two cost terms (resolves Q3, the budget half)

SDF-march cost is **per-pixel and expensive** (a 24–48-step trace per T0-covered pixel, k-NN color gather on hit); splat cost is **per-atom and cheap** (one EWA eval per tile-splat). They cannot share a single scalar "budget" naively. The `FrameBudgetGovernor` (shipped, `atom_instances.rs:984`) already holds a *frame-time* setpoint by modulating a budget — we keep that exact loop and feed it the **honest full-frame wall ms** of T0+T1+T2, but split the controlled budget into two coupled knobs:

- **T0 knob — march step ceiling × T0 pixel budget.** The march's cost is `≈ traced_pixels × steps × per_step_cost`. The governor caps `traced_pixels` by promoting the *largest-coverage* instances to T0 first (descending coverage) until the T0 pixel budget is hit; instances that would exceed it fall to T1. Step ceiling is fixed (24 near, scaling to 48 on the 4070 Ti where march throughput is cheap — RT/compute is exactly what sphere-tracing loves).
- **T1/T2 knob — the existing atom budget `B`,** spent by `budget_walk_and_emit` over the T1/T2 work units, unchanged.

The governor's single setpoint (14.5 ms) drives **both** knobs proportionally: when the measured `ema_ms` runs hot, it shrinks the T0 pixel budget AND the atom budget together (the controlled `budget` scalar maps to both via fixed ratios calibrated once); when cool, it grows both. The design guarantee is unchanged — *frame cost is bounded by the budget, never by scene size* — now across two cost models. The M3 gate prints `sdf_march_ms + splat_ms + imposter_ms ≈ frame` to prove the accounting is honest, and verifies the loop settles at 60 fps on the 780M at 10k buildings. This is the GPU-driven, budget-bounded discipline the 2024 Nanite-materials / GPU-work-generation line formalizes (make each dispatch do meaningful work; bound cost by a budget not the scene), applied to a two-primitive frame.

### 4.7 Cook implications — atoms get sparser; the division of labor (resolves Q4)

If the SDF carries near-field surfaces, **atoms are no longer responsible for looking solid up close** — that job moves to T0. Atoms are now needed only for: (1) **T1 mid-coverage massing** (where sub-pixel sparseness is masked by distance — splatting's natural strength), and (2) **the color field for T0 hits** (§4.3b). Both want *adequate surface coverage of color*, not *gap-free density*. Consequence for the cook:

- **Target atom pitch can loosen.** The current ~0.15–0.30 m pitch was implicitly chasing near-field solidity it never achieved. For T1 legibility at ≥ ~12 m and T0 color sampling (k-NN, k ≤ 8), a coarser pitch suffices. A building that cooked at 15,272 atoms for a (failed) solid look can target a **color-adequate** density — illustratively a 0.30–0.40 m pitch — roughly halving atom count per asset. With T1 budget already bounding the *rendered* atom count, the win is **library memory and cook time**, not frame time: ~200 unique assets at half the atoms ≈ halved `packed_atoms` residency.
- **The SDF is the new near-field source of truth.** Per asset the cook now emits BOTH: the snorm8 `SdfField` (64³, ~183 KB — already an SDF-pillar cook output) and the colored atom set (now sparser). At 200 assets: SDF atlas 36 MB (unchanged) + atom library roughly halved. The net payload is *lower* than the pre-hybrid atom-only target, because the expensive solidity was being bought (and failed) in atoms and is now bought (and achieved) in a cheaper field.
- **Division of labor, pinned:** **SDF = silhouette + solidity + occlusion proxy** (geometry); **atoms = color + mid massing** (appearance). Neither duplicates the other. This is the data-model unification the direction memo names — "SDF the surface, atoms the color field."

The cook changes are **game-side** (`game_asset_cook.rs` emits the looser-pitch atoms beside the already-specified `SdfBaker` call); the engine is pitch-agnostic (atoms arrive colored, the SDF arrives signed). Recorded as a follow-up plan, not gated here.

### 4.8 M4 reconciliation (resolves Q5)

State it explicitly: **M4's honest `--shot-m4` coverage failure is fixed by ADDING T0 (SDF near), not by densifying atoms.** The M4 atom path — `select_resident` → `encode_indirect` → tiled chain — is **correct and unchanged as the T1 tier**. The `--shot-m4` parity gate (instanced coverage vs legacy mesh coverage at the same camera, `c2 ≥ 15%`) failed at near cameras precisely because near buildings are the confetti case; with the near building promoted to T0 SDF-march, the near façade is solid (coverage ≥ 0.97), and the `--shot-m4` comparison flips from "atoms lose to mesh" to "hybrid matches mesh, scalably." No M4 code is reverted; M4 becomes the T1 leg of a three-leg frame. The legacy mesh path can then be retired as the *redundant* solid-near source — T0 is the scalable replacement for it.

### 4.9 Threading / device / ownership

- **All passes share the one game `GpuContext`** (`gpu_context()` OnceLock; `new_with_context` constructors — `SdfMarchPass`, the extended raster, `ExpandDrawsPass`, `SdfAtlasGpu` all take it). Render-thread-owned `&mut`, no new locks.
- **Tier selection** runs in the existing CPU/GPU instance pass (`InstancedSelector` / `InstancedSelectGpu`) — one new classification branch, no new pass.
- **Frame order (pinned):** governor budget → instance pass (classify T0/T1/T2) → clear (incl. layer 2 ← +INF) → `SdfMarchPass` (T0) → `ExpandDrawsPass` (T1+T2 atoms) → `splat_raster` (depth-tested against layer 2) → `resolve_to_srgb` → `composite_lit` (the game's lighting, unchanged) → `governor.update(full_loop_wall_ms)`.
- **Engine stays game-agnostic:** the engine sees assets, instances, SDF fields, atoms, a coverage threshold, a budget. It never sees buildings, lots, roads, or childcare. The game supplies the cooked assets and the `composite_lit` lighting.

---

## 5. Data Models

```rust
// crates/vox_render/src/atom_instances.rs — ADDITIVE to the shipped selector.

/// Which tier an instance renders in this frame. Chosen by screen coverage in
/// the instance pass (§4.2). T1/T2 keep the shipped near/far atom mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Sphere-trace the baked SDF — solid near-field surface. Emits 0 atoms.
    SdfNear,        // T0
    /// Per-cluster atom splats L0–L3 (the M1–M3.2 path, unchanged).
    SplatMid,       // T1
    /// Aggregate imposter atoms I0/I1 (the M1–M3.2 path, unchanged).
    ImposterFar,    // T2
}

/// Coverage thresholds; private with accessors. Defaults are the §4.2 pins.
pub struct TierThresholds {
    t0_coverage: f32,   // ≥ this fraction of min(w,h) → SdfNear (default 0.18)
    t1_coverage: f32,   // ≥ this → SplatMid; else ImposterFar (default 0.012)
}
impl TierThresholds {
    pub fn new(t0: f32, t1: f32) -> Self;
    pub fn classify(&self, coverage: f32) -> Tier;
    pub fn t0_coverage(&self) -> f32;
    pub fn t1_coverage(&self) -> f32;
}
```

```rust
// crates/vox_render/src/gpu/sdf_march.rs (new)

/// One T0 instance to sphere-trace: its inverse world transform (world→local),
/// the atlas header slot from SdfAtlasGpu::insert, and the asset id for the
/// color gather. 64 B, std430. Private fields + accessors.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct T0Instance {
    inv_transform: [f32; 12],   // mat3x4 world→asset-local (rotation+translation)
    header_slot: u32,           // index into SdfAtlasGpu header_buf
    asset: u32,                 // index into the color-gather grid table
    _pad: [u32; 2],
}
impl T0Instance {
    pub fn new(inv_transform: [f32; 12], header_slot: u32, asset: u32) -> Self;
}

/// Per-frame T0 march budget (governed §4.6). Private fields + accessors.
pub struct MarchBudget {
    max_traced_pixels: u32,   // governor-controlled ceiling
    max_steps: u32,           // 24 (780M) … 48 (4070 Ti)
    hit_eps: f32,             // metres; default 0.5·voxel
}
```

The shipped `GpuSplatFull`, `ClusterDraw`, `InstancedSelection`, `FrameBudgetGovernor`, `SdfField`, `SdfAtlasGpu`, `GpuSdfFieldHeader` are reused **unchanged** — this design adds the tier cut, the march pass, the depth-layer write, and the color-gather grid; it changes no existing struct layout.

---

## 6. API

```rust
// crates/vox_render/src/gpu/sdf_march.rs (new) — shared-context house pattern.
impl SdfMarchPass {
    /// Uploads the per-asset color-gather grid ONCE (built from the library's
    /// packed_atoms). Errors mirror SdfAtlasError; ExceedsDeviceLimits up front.
    pub fn new_with_context(ctx: &GpuContext, library: &AssetAtomLibrary,
                            atlas: &SdfAtlasGpu) -> Result<Self, SdfMarchError>;
    /// Record the T0 march: for each instance, sphere-trace the atlas field and
    /// depth-test-store color + view depth into the shared 4-layer texture's
    /// layers 0/1 (spectral), 2 (depth), 3 (transmittance←0). Returns the count
    /// of pixels that produced a hit (the M1 facade_coverage numerator source).
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder,
                  camera: &RenderCamera, instances: &[T0Instance],
                  budget: MarchBudget,
                  output_texture: &wgpu::Texture /* the renderer's 4-layer tex */)
                  -> u32;
}

// crates/vox_render/src/gpu/tiled_splat_renderer.rs — ADDITIVE only.
impl TiledSplatRenderer {
    /// View of the output texture for SdfMarchPass to depth-test/store into.
    pub fn output_texture(&self) -> &wgpu::Texture;
    /// Enable the depth test in splat_raster against layer 2 (default off keeps
    /// the frozen chain bit-identical for non-hybrid callers).
    pub fn set_sdf_depth_test(&mut self, enabled: bool);
}

// crates/vox_render/src/atom_instances.rs — ADDITIVE; existing select() unchanged.
impl InstancedSelector {
    /// Classify each visible instance into a Tier by screen coverage, returning
    /// the T0 set (for SdfMarchPass) AND filling `out` with the T1/T2 atom draws
    /// (the existing budget walk, now over T1/T2 work units only). Deterministic.
    pub fn select_hybrid(&mut self, camera: &RenderCamera, budget: usize,
                         thresholds: &TierThresholds, out: &mut InstancedSelection)
                         -> (Vec<T0Instance>, InstancedStats);
}
// Threading: render-thread-owned (&mut), like the shipped select().
```

Existing signatures relied on (verbatim): `SdfAtlasGpu::{insert(SdfAssetId, &SdfField) -> Result<u32, SdfAtlasError>, texture(), header_buf()}`; `SdfAtlasProbe` sample math (`g = (p-origin)/voxel`, `uvw = (off+g+0.5)/dims`, `d = sample.r * band`); `FrameBudgetGovernor::{new(target_ms, initial, min, max), budget(), update(frame_ms)}`; `InstancedSelectGpu::select_resident`; `ExpandDrawsPass::encode_indirect`; `TiledSplatRenderer::{new_with_capacity, set_active_splat_count, render, resolve_to_srgb}`; `tile_assign` writes `position_depth.w = -p_view.z`; the output texture is a 4-layer `Rgba32Float` with layer 2 unused.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `TierThresholds` + `select_hybrid` | per frame before render | `civitas_care/src/render_gpu/mod.rs` `render_instanced`; `scale_trial --hybrid` | one classify branch; T1/T2 walk unchanged |
| `SdfMarchPass::new_with_context` | city load (ONCE) | `SceneRenderer::new_city_instanced` | uploads color-gather grid; renderer still constructed 1× |
| `SdfMarchPass::encode` | per frame, before the atom expand+raster | `render_instanced` frame order (§4.9) | writes the shared 4-layer texture |
| `TiledSplatRenderer::set_sdf_depth_test(true)` | once after construction in hybrid mode | `render_instanced` setup | default off keeps non-hybrid callers frozen |
| splat_raster depth-test extension | inside the frozen-chain raster (additive uniform + compare) | `crates/vox_render/src/gpu/splat_raster.wgsl` | single compare vs layer 2; no new pass |
| clear extension (layer 2 ← +INF) | per frame | `tiled_splat_renderer.rs` clear | the only frozen-chain clear change |
| Two-term governor mapping | wraps T0 pixel budget + atom budget | `render_instanced` | one setpoint → both knobs (§4.6) |
| `SdfAtlasGpu::insert` per unique asset | city load | `SceneRenderer::new_city_instanced` | same fields the SDF-pillar M1 residents |
| color-gather grid build | inside `SdfMarchPass::new_with_context` | `sdf_march.rs` | reuses `AssetAtomLibrary::packed_atoms` (no new storage) |

---

## 8. Open Questions

- [ ] **k-NN color gather frequency.** Does k ≤ 8 over a coarse asset-local grid give acceptable façade color, or do tiled materials need the option-(c) triplanar overlay? Measure in M1 with the real building: print color error vs the legacy mesh frame; decide before M2.
- [ ] **T0 pixel budget vs atom budget ratio.** The §4.6 fixed mapping from the one governed scalar to two knobs needs a calibrated ratio. M3 measures `sdf_march_ms` vs `splat_ms` at 10k buildings and pins the ratio; if T0 march dominates at street level, the answer is a lower `T0_COVERAGE` (fewer T0 instances), not a smaller step ceiling.
- [ ] **SDF depth precision in layer 2.** `Rgba32Float` layer 2 holds view-space depth at f32 — ample; but the splat raster's per-tile front-to-back order plus the SDF compare must not z-fight at a splat-grazes-SDF seam. M2's `depth_seam_artifacts == 0` gate measures this; if it fails, add a small bias.
- [ ] **T0 transition band.** A building crossing the `T0_COVERAGE` threshold pops from solid SDF to splatted L0. Does an L0-splat-over-SDF crossfade band (render both, blend by coverage) hide the pop, or is the L0↔SDF visual delta small enough to hard-switch? Measure when M2 lands.
- [ ] **Far SDF occlusion proxies.** The SDF-pillar §4.7 names Tier-1 boxes as HZB occlusion proxies. Whether T0 SDF surfaces should also seed an occluder buffer to cull T2 imposters behind them is a measured later milestone, not v1.

---

## 9. Out of Scope

- **Spectra path-tracer changes.** The cinematic-stills path (`pathtrace_splats_to_rgba`) is untouched; this design is the interactive loop. Routing the game view through Spectra is the separate texture-bridge plan.
- **The frozen tiled chain's sort/raster algorithm.** The only raster edits are additive: write depth to layer 2, and one compare against it. `tile_assign`/`radix_sort`/`tile_range_build` are unchanged.
- **A new lighting model.** `composite_lit` (the game's emissive-splat + sky composite) is the lighting, unchanged; T0 SDF surfaces feed it spectral color exactly like splats do. `LightRig` stays the Spectra-only rig.
- **The cook's looser-pitch atom emission** (§4.7) — the engine is pitch-agnostic; the game cook change is a sequenced follow-up plan.
- **SDF destruction/CSG, contact AO, the clipmap, 2D coverage fields** — those are the SDF pillar's own later milestones (M2–M6); this design consumes only the M1 atlas + probe.
- **Scatter-field grass/props** — the sibling scatter design shares this tier contract but is its own doc.

---

## 10. Related Plans / Designs

- Depends on: [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (shipped M1–M3.2 — the T1/T2 substrate, governor, expand pass, frozen chain) and [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) + its [M1 plan](../plans/2026-06-10-sdf-pillar-m1.md) (shipped `SdfBaker`/`SdfAtlasGpu`/`SdfAtlasProbe` — the T0 substrate).
- Reconciles: [Virtualized M4 game wiring](../plans/2026-06-11-virtualized-m4-game-wiring.md) — M4's atom path is the T1 tier; its `--shot-m4` coverage failure is fixed by ADDING T0 (§4.8).
- Required before: the Hybrid LOD implementation plan (M1 SDF-march-solid → M2 hybrid composite → M3 unified budget at city scale).
- Related: sibling [Scatter Field design](./2026-06-12-scatter-field-design.md) (shares the tier contract); `aaa-phase2-resident-frame` memory (the resident-frame/GpuContext doctrine + the radix non-determinism honesty rail); direction memory `hybrid-atom-sdf-lod-direction`.
- SOTA grounding (cited, not padded): hierarchical/continuous splat LOD for large scenes — H3DGS ([arXiv 2406.12080](https://arxiv.org/abs/2406.12080)), LODGE ([arXiv 2505.23158](https://arxiv.org/html/2505.23158v2)), CLoD-GS ([arXiv 2510.09997](https://arxiv.org/pdf/2510.09997)) — informs T1/T2; implicit-surface sphere-tracing LOD — NGLOD ([arXiv 2101.10994](https://arxiv.org/abs/2101.10994)) — and the hybrid-SDF-Gaussian (Gaussians aligned to an SDF zero-level set) line inform T0 + §4.3(b); depth-buffer fusion of raymarched + rasterized sources ([Industrial Arithmetic, 2014](http://industrialarithmetic.blogspot.com/2014/11/depth-buffer-fusion-for-real-time.html); jimbo00000, raymarching+rasterization) grounds §4.4–4.5; GPU work-generation / budget-bounded GPU-driven rendering (Nanite GPU-driven materials, GDC 2024) grounds §4.6.
```
