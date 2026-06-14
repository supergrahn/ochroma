# Design: Spectra Universal Renderer — one path tracer intersects the engine's real primitives for both interactive and cinematic frames (2026-06-12)

**Status:** Draft
**Scope:** Make the engine-owned Spectra path tracer the SINGLE runtime renderer for every Ochroma frame — interactive and cinematic alike — by having Spectra NATIVELY intersect the engine's real scene primitives (instanced SDF volumes and 3D Gaussian atoms) inside the trace, instead of the game owning a separate wgpu render path and feeding Spectra lossy camera-facing quads. The game stops owning a render path; SDF atlas + atom library + scatter generator + terrain become scene DATA that one path tracer traces. This design resolves the trace-primitive fork, the engine→Spectra scene-feeding seam, and the correctness milestones; it defers the realtime millisecond ladder to the cited realtime design.
**Related:**
- Depends on the renderer-ownership boundary: [Spectra Primary Renderer Bridge](./2026-06-11-spectra-primary-renderer-bridge.md) (the `ViewIntent`/`GameViewSource`/`live_update` contract this extends).
- Defers realtime to: [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (M0–M6, the 1–2 ms budget contract, the M2 renegotiation trigger, the 4070-Ti gating, the backend matrix — the realtime half is cited, never restated here).
- Consumes as scene data: [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) (`SdfBaker`/`SdfAtlasGpu` — GWN-signed snorm8 fields from sealed watertight meshes), the [Atom-Budget Splat Renderer](./2026-06-06-atom-budget-splat-renderer-design.md) (`AssetAtomLibrary`), [Scatter Field](./2026-06-12-scatter-field-design.md) (the generator becomes a per-tile atom producer), [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (the `InstancedSelector` + `FrameBudgetGovernor` become SCENE SELECTION/LOD feeding the BVH, not a rasterizer).
- **Supersedes:** [Hybrid Atom + SDF LOD Rendering Pipeline](./2026-06-12-hybrid-atom-sdf-lod-design.md). That design composited a near SDF-march pass and a mid/far splat-raster pass into one wgpu framebuffer via a shared depth layer — two engine renderers reconciled by a depth buffer. The user rejected that. This design keeps that doc's *tier/division-of-labor reasoning* (SDF = solid surface, atoms = appearance/scatter) but moves both into ONE path tracer's intersection stage. See §4.2 for the explicit distinction.

---

## 1. Problem Statement

Concrete, observable symptoms (every claim verified against code 2026-06-12):

- **The engine→Spectra bridge is lossy by construction — it throws away the primitive.** `crates/vox_render/src/splat_convert.rs` tessellates every splat into a flat quad (4 verts / 2 triangles): its own module doc says *"the current `spectra-renderer` is a triangle-mesh spectral path tracer"* and *"a volume splat becomes a **camera-facing** quad sized by its scale."* Per-splat spectral colour is preserved on the Rust struct's API surface but **`material_ids` is left empty so the uploader assigns material 0 to every triangle** (`splat_convert.rs:21-22`). A 3D anisotropic Gaussian — the engine's whole appearance primitive — reaches the path tracer as a grey flat card. This is FIX-3 confetti made worse: not just sparse, but flattened and decolored.
- **The game owns a renderer, against the stated boundary.** The Spectra Primary Renderer Bridge (2026-06-11) names the end state — "Civitas does not own `render`, `render_gpu`, presenter, shader, or GPU HUD modules" (M4) — yet the interactive frame today is the game's wgpu tiled splat chain, and the four superseded/sibling render designs (virtualized splat, hybrid atom+SDF, scatter) all build *engine* render paths the game drives. Two renderers (game wgpu chain; Spectra stills) exist; the directive is one.
- **Spectra's native splat path is unwired into the production renderer.** The megakernel DOES contain a Gaussian-splat intersection stage — `traverse_gaussian_bvh(gs_ray, gs_t_max)` at `slang/megakernel.slang:1148`, backed by a real ellipsoid ray-intersection (`gaussian_ray_alpha`, `slang/splat_intersect.slang:54`) with SH + a 4-band spectral-embedding decode. But **`rust/spectra-renderer/` binds none of `g_splat_buffer` / `g_gaussian_bvh_nodes` / `u_num_gaussians`** (grep over `spectra-renderer/` and `spectra-scene-upload/` returns zero matches). Those buffers are populated only by the separate `spectra-gaussian-render` crate (its own SAH BVH builder in `bvh.rs`, its own rasterizer) and the Python harness — never by the engine's `SceneState`→`GpuScene` upload path. So the native splat capability exists in shader but is not reachable from an Ochroma frame.
- **Native SDF intersection exists only in fragments, and the *instanced engine-SDF* variant is unread.** The megakernel sphere-traces a *terrain* SDF (`terrain_intersect`, `slang/terrain_sdf.slang:70`, a 256-step sphere trace over an NDF/SVO chunk grid, gated by `g_terrain_enabled`) and there is a procedural op-tree evaluator (`slang/sdf_eval.slang` — `eval_sdf_tree`, sphere/box/capsule/CSG). But the engine's *baked instanced* fields — the SDF Pillar's `SdfField` snorm8 volumes — are uploaded by `spectra-scene-upload/src/uploader.rs:327` (`sdf_distances` / `sdf_volume_headers` / `sdf_instance_headers`) and **read by no megakernel kernel** (confirmed in the SDF Pillar §1 and by grep — no Slang consumer of `sdf_volume_headers`). The instanced-SDF intersection the universal renderer needs — sphere-trace many transformed `SdfField` boxes from the snorm8 atlas inside the trace — is **UNBUILT**. `slang/splat_intersect.slang` is NOT a stub (it is a complete ellipsoid intersector), but the instanced engine-SDF reader is the genuine hole.
- **All Spectra traversal is software compute; Spectra has ZERO wired hardware RT.** Per the realtime design §4.3 (verified): every ray in the frame loop goes through `bvh_traverse.slang` / the Gaussian stack-BVH as plain compute; `render_state.rs` hardcodes `optix_traversal: None`; the OptiX/VK-RT crates expose host-batch APIs incompatible with a GPU-resident wavefront. This is a *realtime* problem (cited, not solved here), but it bounds what M0/M1 can claim: correctness now, milliseconds later.

The throughline: **"Spectra renders everything" is blocked on one buildable correctness gap — Spectra natively intersecting the engine's real primitives (instanced Gaussians + instanced baked SDF) from the engine's own scene-feed — and one hardware-gated performance gap (realtime), which is a separate, later, `tomespensin`-only effort.**

---

## 2. Done When

**Headline (M0 + M1 landed), on the AMD 780M (RADV, Linux) — correctness, seconds/frame acceptable, NO millisecond gate:**

Running

```bash
cd ~/src/ochroma && cargo run --release --bin spectra_universal_probe -- \
  --scene one_building --primitive both --width 1280 --height 720 --spp 64
```

prints

```
[spectra-universal] gpu=AMD Radeon 780M backend=vk-compute trace=sdf+gaussian (native, no quad bridge)
[spectra-universal] one_building: sdf_instances=1 atoms=15272 scatter_blades=0
[spectra-universal] facade_coverage: native=0.991 vs lossy-quad-bridge=0.310  (gate native>=0.97 AND native>=3x bridge) -> PASS
[spectra-universal] hit color: mean |rgb - nearest_atom_spectral_rgb| = 0.061 (gate < 0.15, spectral preserved) -> PASS
[spectra-universal] frame=4.2 s  (seconds/frame is acceptable on the floor — correctness first)
wrote universal_one_building.png   (solid gap-free coloured facade, NOT grey confetti)
```

with **`facade_coverage native ≥ 0.97`** and **`native ≥ 3× the lossy-quad-bridge coverage`**, **hit-colour error < 0.15** (proving spectral colour survives the native trace where the quad bridge dropped it to material 0), the `trace=` line naming the primitive types intersected, and `universal_one_building.png` passing eyeball: a solid, gap-free, *coloured* facade where `splat_convert.rs`'s quad bridge produced grey confetti. A human at the keyboard verifies this without reading code.

**Then M1 — the full engine scene through one path tracer (still the 780M, seconds/frame):**

```bash
cd ~/src/ochroma && cargo run --release --bin spectra_universal_probe -- \
  --scene city_block --primitive both --grass --spp 64
```

prints `[spectra-universal] city_block: sdf_instances=<N> atoms=<A> scatter_blades=<G> terrain_chunks=<T>` and `[spectra-universal] one renderer, one trace, all four sources -> PASS`, and `universal_city_block.png` shows solid buildings (SDF), coloured massing/detail (atoms), grass (scatter atoms), and ground (terrain SDF) in a SINGLE traced image with no second renderer and no quad bridge in the path (`bridge=none` printed).

**Realtime (1–2 ms) is explicitly NOT in this doc's Done When.** It is validatable only on `tomespensin` (RTX 4070 Ti) and is owned, gated, and kill-criteria'd by [Spectra Realtime](./2026-06-10-spectra-realtime-design.md). This design's bar is correctness on the floor; the realtime ladder takes over after M1.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Native SDF intersection beats the lossy quad bridge | cook one 15,272-atom building to `SdfField`, resident in `SdfAtlasGpu`, trace at 12 m: `assert!(native_cov >= 0.97 && native_cov >= 3.0 * bridge_cov)`, both fractions printed from the same camera | `assert!(hit_pixels > 0)` |
| Spectral colour survives the native trace | trace an SDF hit, resolve colour by k-NN atom gather: `assert!((hit_rgb - nearest_atom_spectral_rgb).length() < 0.15)` on a hand-placed coloured atom field, printed — the quad bridge scores ≈ uniform material-0 grey here | `assert!(color != black)` |
| Native Gaussian intersection from the engine feed | upload 1000 engine atoms via `SceneState`, trace: `assert!(traced_alpha_coverage > 0.5)` AND `assert!((traced_rgb - cpu_atom_rgb).length() < 0.1)` over the splat footprints, printed — proves the engine→`g_splat_buffer` path, not the Python harness | `assert!(num_gaussians > 0)` |
| One renderer, two primitive types in one trace | scene with 1 SDF instance in front of an atom cloud in front of terrain: per-pixel nearest-source map, `assert!(sdf_pixels>0 && atom_pixels>0 && terrain_pixels>0 && farther_source_wins==0)`, all four printed | `assert!(image.len() > 0)` |
| No quad bridge in the universal path | run the universal probe: `assert_eq!(quads_tessellated, 0)` (the `splat_convert.rs` counter), printed `bridge=none` | checking the image is non-empty |
| Scene feed is generic (no game terms) | feed SDF+atoms+scatter+terrain through `SceneUpdateBatch`: `assert!(batch_carries_no_building_zone_road_symbols)` — the upload types are `SceneResource::{SdfVolume,InstanceBatch,Geometry,Material}` only | renderer compiles with a `Building` type |
| Static BVH build vs per-frame refit split | static SDF instance → BVH built once (`builds==1` after 10 frames); a moved scatter tile → refit only (`refits>=1, rebuilds==0`), both counts printed | `assert!(bvh.is_some())` |

---

## 4. Architecture

### 4.1 The honest precondition, stated first and not buried

**Spectra is not "triangle-only" today — but it cannot render the engine's real primitives from the engine's scene feed today, and that is the gap this design exists to close.** Verified state of `~/src/spectra`:

| Primitive | Shader intersection exists? | Wired into `spectra-renderer` production frame? | Fed by the engine's `SceneState`/`GpuScene` upload? |
|---|---|---|---|
| Triangles (mesh BVH) | yes (`bvh_traverse.slang`, megakernel primary path) | **yes** | yes (this is what the lossy quad bridge abuses) |
| 3D Gaussian atoms | **yes** — `gaussian_ray_alpha` ellipsoid intersect (`splat_intersect.slang:54`), `traverse_gaussian_bvh` called at `megakernel.slang:1148`, SH + 4-band spectral decode | **no** — `g_splat_buffer`/`g_gaussian_bvh_nodes` bound only by `spectra-gaussian-render`/Python, not `spectra-renderer` | **no** — no `SceneState`→splat-buffer upload exists |
| Terrain SDF (sphere trace) | yes — `terrain_intersect` 256-step trace (`terrain_sdf.slang:70`), gated `g_terrain_enabled` | yes (when terrain present) | partial (NDF/SVO chunk path, not the baked `SdfField`) |
| **Instanced baked engine SDF** (`SdfField` snorm8 atlas, transformed per instance) | **NO** — uploaded by `uploader.rs:327` but read by no kernel | no | uploaded-but-unread (dead-end) |

So the correctness work is precisely two items, both buildable and verifiable on the 780M with Vulkan compute at seconds/frame:

1. **Wire the existing native Gaussian path to the engine feed** — bind `g_splat_buffer`/`g_gaussian_bvh_nodes` in `spectra-renderer` and feed them from `SceneState`'s atom instances (not from the Python harness, not via quads). Carry the engine's 16-band spectral, not material-0 grey.
2. **Build the instanced engine-SDF intersection** — a megakernel stage that sphere-traces each transformed `SdfField` box out of the snorm8 atlas `uploader.rs` already uploads, giving gap-free opaque surfaces.

The quad bridge (`splat_convert.rs`) is retired from the universal path the moment (1) lands (it stays only as a debug/comparison oracle for the M0 coverage gate — see the deprecation ledger §4.8).

**The realtime half is a different, harder, later problem.** Spectra has zero wired hardware RT (realtime design §4.3); the software-compute Gaussian stack-BVH and the 256-step SDF march are *seconds-per-frame correct* and *milliseconds-impossible*. Getting either primitive into the 1–2 ms budget needs HW traversal (VK ray-query / OptiX), custom intersection programs for the non-triangle primitives, ReSTIR/cache amortization, and DLSS-RR — all owned by the realtime design and validatable only on `tomespensin`. **This doc does not restate or weaken any of that.** It delivers the primitive-correctness precondition the realtime ladder then accelerates.

### 4.2 The trace-primitive fork — decision (c), with arithmetic, and the distinction from the rejected hybrid

**The fork:** what does the one path tracer intersect?

- **(a) SDF only** — sphere-trace instanced `SdfField` volumes; gap-free opaque surfaces (resolves FIX-3 confetti); appearance from a k-NN atom gather at the hit (the superseded hybrid's §4.3(b) recommendation) or a colored field.
- **(b) Gaussian only** — intersect atoms directly as volumetric anisotropic Gaussians (true 3DGS-in-a-path-tracer; this is the 3DGRT line — `nv-tlabs/3dgrut`, CVPR 2025); grass/scatter are naturally this.
- **(c) BOTH** — SDF for opaque hard surfaces (buildings, terrain), Gaussians for volumetric/scatter (grass, smoke, foliage), as two primitive types in ONE path tracer's BVH/intersection stage.

**Decision: (c).** The arithmetic and the engine's content shape both point here, and Spectra's megakernel is *already structured for it* (§4.1 shows triangles + Gaussian + terrain-SDF composited in one trace today).

**Why not (a):** an SDF-only city forfeits the engine's appearance primitive. Grass, foliage, smoke, and destruction debris are *volumetric and semi-transparent* — they are Gaussians by nature (the Scatter Field design makes every blade an anisotropic `GaussianSplat::volume`). Forcing them into opaque sphere-traced surfaces loses the sub-pixel anti-aliasing that is splatting's whole advantage for sub-pixel content (Mip-Splatting / Multi-Scale-3DGS), and an SDF cannot represent a translucent blade at all. (a) also makes near-field colour a permanent k-NN approximation even where dense atoms exist.

**Why not (b):** the FIX-3 finding is structural — a 15,272-atom building is a sparse point cloud whose inter-atom gaps exceed a pixel at any near camera, so even intersected natively it reads as confetti up close unless densified, which re-creates the 50M-world-atom memory blowup virtualization exists to avoid (hybrid design §1, §4.8). Hard surfaces *want* the SDF's gap-free solidity (the snorm8 atlas gives it from 183 KB/asset). 3DGRT's own results bind each Gaussian in a proxy mesh for the BVH — efficient for view-synthesis clouds, but it does not make a sparse procedural-building atom set look solid; the SDF does, cheaper.

**Cost/quality arithmetic:**

| | (a) SDF only | (b) Gaussian only | **(c) BOTH** |
|---|---|---|---|
| Near building facade (780M, correctness) | solid (cov ≥ 0.97), colour = k-NN approx | confetti unless densified (50M-atom blowup) | **solid SDF + k-NN/atom colour** |
| Grass / foliage / smoke | impossible (opaque only) | natural (volumetric Gaussians, free AA) | **natural (Gaussian tier)** |
| Memory (200 unique assets, 780M UMA) | SDF atlas 36 MB | atom library only, but must densify → GBs | **SDF atlas 36 MB + atom library (sparser, hybrid §4.7) ≈ 36 MB + ~half-atoms** |
| 4070 Ti realtime (cited, not solved here) | 1 custom SDF-march intersection program | 1 custom Gaussian intersection program (3DGRT/Blackwell ray-sphere) | **2 custom intersection programs in 1 BVH — strictly the union; the realtime design's job** |
| Matches engine content | partial | partial | **exact** |

The 4070-Ti realtime cost of (c) is the *union* of the two custom-intersection programs' costs, not a multiplication — a path tracer that handles spheres + triangles + Gaussians runs each ray against whichever primitive its BVH leaf holds; adding a second primitive type adds intersection-program divergence, not a second full trace. SOTA confirms this is the working direction: SDF grids ray-traced inside a path tracer via custom intersection (JCGT 2022 "Ray Tracing of Signed Distance Function Grids"; Sphere-Tracing SDFs with OptiX), 3DGRT intersecting Gaussians in the same trace as meshes, and Blackwell RT cores natively supporting ray-sphere intersection — all point to "one tracer, several primitive types," which is exactly (c).

**THE DISTINCTION the user must not mistake this for.** The rejected hybrid (2026-06-12-hybrid-atom-sdf-lod-design.md) ran **two renderers** — a wgpu `SdfMarchPass` and the wgpu `splat_raster` chain — and reconciled them by writing depth into a shared framebuffer layer (its §4.4–4.5: "depth-buffer-fusion of a raymarched and a rasterized source"). That is *two renderers composited by a depth buffer.* This design is **one renderer** — the Spectra path tracer — whose ray, per intersection, tests both an SDF primitive and a Gaussian primitive in its acceleration structure and keeps the nearest/composited hit, exactly as any path tracer resolves a scene of spheres + triangles + meshes. There is no second renderer, no framebuffer depth reconciliation, no compositing pass: the "reconciliation" is the trace's own nearest-hit/transmittance logic, in shader, per ray (the megakernel already does this — SDF/Gaussian/triangle all update the same `hit_t`/`throughput`/film at `megakernel.slang:971–1168`). **One path tracer, two primitive types ≠ two renderers composited.** That sentence is the load-bearing difference between this design and the one the user rejected.

### 4.3 Appearance on the SDF surface — atoms are the colour field (carried from the superseded hybrid)

An SDF sphere-trace hit has position + gradient-normal but no colour. The superseded hybrid resolved this (its §4.3) and the analysis transfers unchanged because it was never wgpu-specific: **(b) sample the nearest engine atoms' 16-band spectral colour at the hit** — the atoms already resident for the Gaussian tier ARE a coloured point cloud of the asset surface; too sparse to look solid (that's the SDF's job) but dense enough to colour a surface. At a T0 SDF hit, gather k≤8 nearest asset-local atoms via a small per-asset uniform grid (built at library cook), blend their spectral SPD by Gaussian distance weight, and shade. Zero added atlas storage (reuses the atom library), engine stays game-agnostic (no material-zone semantics leak), and it is the architecturally honest unification: **the SDF is the surface, the atoms are its colour.** Inside Spectra this is a megakernel function called from the new instanced-SDF hit branch; the spectral SPD feeds the existing spectral film path (`megakernel.slang:1053-1081`) so colour is *real spectral*, not the material-0 grey the quad bridge produced. The rejected alternatives (per-voxel coloured SDF = +200 MB, disqualified on the 780M UMA; triplanar material-zone textures = game-coupled) are rejected for the same reasons stated there.

### 4.4 LOD under one path tracer = acceleration-structure detail + ray/sample budget, never technique-switching

The directive's "no matter what scene" is honored not by swapping renderers per distance (the rejected hybrid's tier-switch) but by the two levers every production path tracer uses, both of which the engine already has machinery for:

**1. Acceleration-structure detail (the `InstancedSelector` feeds the BVH, not a rasterizer).** The shipped `InstancedSelector` / `InstancedSelectGpu::select_resident` (virtualized design) already classify each placement cull/far/near with LOD levels L0–L3 + imposters I0/I1 under a budget, emitting a prefix-summed selection. In the universal renderer this selection becomes **scene selection feeding Spectra's TLAS**, not a draw list feeding a raster:
- **Instance-level cull/LOD:** the selector decides which instances are resident in the TLAS this frame and at what LOD; culled instances are simply absent from the acceleration structure (the cheapest possible LOD — no BVH leaf at all). This is the `FrameBudgetGovernor` bounding TLAS size, hence trace cost, by scene selection.
- **SDF mip:** a far building uses a coarser mip of its snorm8 `SdfField` (fewer march steps to converge) — an LOD on the primitive's own data, inside the intersection program.
- **Gaussian count per instance:** a far atom-instance contributes fewer atoms to its BLAS (the existing I0 ≤ 64-atom / I1 = 1-atom imposter levels become "how many Gaussians this instance puts in the BVH"); scatter tiles past the blade→shell distance contribute one shell Gaussian instead of thousands (Scatter Field §4.5 cost-collapse, now a BVH-population decision instead of a draw-count decision).

**2. Sample/ray budget (ReSTIR reservoirs — owned by the realtime design).** On the realtime tier, quality is the free variable and milliseconds are fixed: per-pixel ray/sample counts are constant, and ReSTIR temporal reservoirs + the world-space radiance cache carry convergence across frames (realtime design §4.1–4.2). The universal renderer does not redefine this; it states only that **LOD = AS detail (lever 1) + sample budget (lever 2), never technique-switching**, so the same trace handles a near street and a far skyline by changing what is in the BVH and how many samples accumulate, not by switching to a different renderer.

**FrameBudgetGovernor → ray/sample budget mapping.** The shipped `FrameBudgetGovernor` (`atom_instances.rs:984`) is a proportional controller on measured frame time that today modulates an atom budget. In the universal renderer it maps onto two coupled knobs (the same two-term shape the superseded hybrid §4.6 derived, now over a trace not a raster): **(i) TLAS population budget** — how many instances/atoms/SDF-mips the selector residents (lever 1), and **(ii) the internal-resolution / sample knob** the realtime governor owns (lever 2, realtime design §4.4). On the 780M (correctness) only lever 1 runs and the frame is seconds; on `tomespensin` (realtime) both run and hold 2 ms by trimming TLAS population and internal resolution before ever lengthening the frame. There is ONE governor type across the engine (the merge obligation the realtime design §4.8 and scatter design pin); this design adds the "budget maps to BVH population" reading of it.

### 4.5 The engine→Spectra scene-feeding seam

The engine feeds four sources into one `SceneState`; Spectra builds one acceleration structure over the resulting primitives. The seam EXTENDS the shipped `live_update` contract (`rust/spectra-scene-state/src/live_update.rs`, verified) — which already carries `SceneResource::{Geometry, Material, InstanceBatch, SdfVolume, SdfInstanceBatch, LightSet}`, `DirtyRange{Kind}` (incl. `InstanceTransforms`), `ResidencyChange`, and `FrameOutputTarget::GpuResident` — so most of the contract exists; this design adds the *atom/Gaussian* resource and the *SDF-as-traced-primitive* reader.

| Engine source | Becomes Spectra scene data | Upload path (extends live_update) | BVH lifetime |
|---|---|---|---|
| `SdfAtlasGpu` (snorm8 fields, GWN-signed, per-asset) | `SceneResource::SdfVolume` (the atlas) + `SdfInstanceBatch` (per-placement transforms) — and the NEW megakernel reader that sphere-traces them | `uploader.rs` already uploads `sdf_distances`/`sdf_volume_headers`/`sdf_instance_headers` (today dead-ended); this design WIRES the reader | **static BVH build** for placed buildings (transforms stable); refit on plop/destroy only |
| `AssetAtomLibrary` atoms (16-band spectral) | `SceneResource::InstanceBatch` of Gaussians → bound to `g_splat_buffer` + a Gaussian BLAS/BVH | NEW: `SceneState` atom-instance upload binding `g_splat_buffer`/`g_gaussian_bvh_nodes` in `spectra-renderer` (today only `spectra-gaussian-render`/Python bind these) | **static** per unique asset BLAS (atoms are asset-local); TLAS refit on instance move |
| Scatter generator (grass/props, per-tile hash-placed) | Gaussians emitted into the same `g_splat_buffer` range per frame; far tiles → one shell Gaussian | NEW: scatter's `ScatterField::encode` writes into the Spectra-bound atom buffer at a disjoint slot range (the producer seam it already has for the wgpu chain, retargeted) | **per-frame refit** (animated/regenerated each frame); capped, amortized (realtime design §4.5/M5) |
| Terrain (NDF/SVO chunks; baked terrain `SdfField`) | `SceneResource::SdfVolume` terrain + existing `terrain_intersect` | existing terrain upload + `g_terrain_enabled` | **static** per chunk; refit on sculpt |

**Static build vs per-frame refit, pinned:** buildings, terrain, and per-asset atom BLASes are **static BVH builds** (camera-independent geometry; built once at residency, the cheapest steady state). Animation and scatter are **per-frame refits** — transform updates via `DirtyRangeKind::InstanceTransforms` (a TLAS refit, not rebuild), and the scatter blade buffer regenerated + refit each frame under the capped AS-update budget the realtime design §4.5/M5 owns. A destroyed building's wounded `SdfField` copy (SDF Pillar §4.8) triggers one BLAS rebuild, queued across frames. The engine decides residency/LOD (lever 1 §4.4); Spectra receives only generic `SceneUpdateBatch`es of handles + dirty ranges — no `Building`/`Zone`/`Road` ever crosses the seam (the Primary Bridge ownership rule, enforced by the `spectra_bridge.rs` boundary test).

### 4.6 Interactive AND cinematic from one renderer — the only difference is sample/time budget

"Both interactive and cinematic frames" is one renderer, two budgets, zero code forks in the trace:
- **Cinematic (stills):** the existing spp loop to convergence, 32-band spectral film, full bounces/caustics — the offline integrator, unchanged. The universal change is that it now traces native SDF + Gaussian engine primitives instead of lossy quads, so the still is of the *real scene* at full quality.
- **Interactive:** the realtime frame graph (realtime design) — fixed 1–2 ms, ReSTIR reservoirs + radiance cache + DLSS-RR, quality converging across frames. Same scene, same primitives, same BVH; the only difference is "accumulate to convergence" (cinematic) vs "one budgeted frame, amortize over time" (interactive).

This is why "Spectra renders everything" is coherent: cinematic and interactive are the *same trace of the same primitives* differing only in sample/time budget — not two renderers, and emphatically not the game's wgpu chain for interactive + Spectra for stills.

### 4.7 Threading / device / ownership

- **Engine owns scene selection + residency + budget** (the `InstancedSelector`, `FrameBudgetGovernor`, `SdfAtlasGpu`, `AssetAtomLibrary`, scatter generator); these run render-thread-owned `&mut` on the shared `GpuContext`, exactly as today, but now *emit `SceneUpdateBatch`es to Spectra* instead of driving a raster.
- **Spectra owns the trace** — TLAS/BLAS build+refit, the SDF+Gaussian+triangle intersection stages, spectral lighting, reconstruction, GPU-resident output. Single-owner render thread (the realtime design's `&mut self` model).
- **The boundary is the `live_update` `SceneUpdateBatch`** — generic handles + dirty ranges + residency + `FrameOutputTarget::GpuResident`. The `vox_render::spectra_bridge` ownership test (`renderer_owned_items_are_exactly_the_primary_bridge_set`) is the compile-time guard that no game/runtime concept leaks across.
- **Game owns nothing render** — it supplies cooked assets (SDF fields, atom libraries, scatter density fields) and lighting intent through `ViewIntent`/`GameViewSource`, and never calls Spectra directly (Primary Bridge M4).

### 4.8 Deprecation / salvage ledger

| Asset | Fate under the universal renderer | Why |
|---|---|---|
| Game wgpu tiled splat rasterizer (`tiled_splat_renderer.rs`, frozen 4-pass chain) | **Demoted to fast preview / debug, then retired as runtime.** Per Primary Bridge M4: "the WGPU tiled path is demoted to fallback/debug/transition path." Kept while the realtime trace tier is unproven on hardware below the 4070 Ti; not the product renderer. | The directive: one renderer (the path tracer). The raster is a transition preview, not a coexisting runtime. |
| `splat_convert.rs` lossy quad bridge | **Retired from the universal path; kept ONLY as the M0 comparison oracle.** The native Gaussian trace replaces it. | It flattens + decolors the primitive (§1); the native path is the whole point. |
| `SdfBaker` / `SdfAtlasGpu` (SDF Pillar M1) | **Load-bearing scene data.** Becomes the traced opaque-surface primitive (§4.5). | Sealed-mesh GWN-signed snorm8 fields are the gap-free surface source. |
| `AssetAtomLibrary` atoms | **Load-bearing scene data.** The traced Gaussian primitive AND the SDF-hit colour field (§4.3). | The engine's appearance primitive — now traced natively with spectral preserved. |
| Scatter generator (`ScatterField`) | **Load-bearing per-frame producer.** Emits Gaussians into the Spectra atom buffer (§4.5). | Grass/props are volumetric Gaussians — exactly what the Gaussian tier wants. |
| `InstancedSelector` + `FrameBudgetGovernor` | **Load-bearing scene SELECTION/LOD.** Feed the TLAS + map budget→BVH population (§4.4). | LOD becomes AS detail; the governor bounds trace cost by selection. |
| Hybrid `SdfMarchPass` + depth-layer composite (superseded design) | **Superseded — not built.** Its tier/colour reasoning is salvaged into §4.2–4.4; its two-renderer-depth-composite mechanism is discarded. | The user rejected two-renderer compositing; this is one tracer. |
| `spectra-gaussian-render` crate (separate BVH + rasterizer) | **Salvage the BVH builder + ellipsoid intersection into the engine-fed path; retire the standalone rasterizer.** | Its SAH `GaussianBvh::build` and `gaussian_ray_alpha` are the working native-Gaussian pieces; only the engine-feed wiring is missing. |

### 4.9 SOTA grounding (verified 2026-06-12)

- **3D Gaussians intersected in a path tracer alongside meshes** is shipped research: 3DGRT (`nv-tlabs/3dgrut`, CVPR 2025) bounds each Gaussian in a proxy mesh for the BVH and gathers per-ray intersections, integrating with mesh path tracing; RaySplats (arXiv 2501.19196), 3DGUT/3DGRUT (distorted cameras + secondary rays), and GRTX (arXiv 2601.20429) confirm the line. This is exactly the Gaussian tier of decision (c).
- **SDF grids ray-traced inside a path tracer via custom intersection** is established: "Ray Tracing of Signed Distance Function Grids" (JCGT 2022) and "Sphere-Tracing Signed Distance Fields with OptiX" run a sphere-trace in a custom intersection program invoked when a voxel BLAS is hit — the SDF tier of (c).
- **Hardware support is arriving for the non-triangle primitives:** NVIDIA Blackwell RT cores natively support ray-sphere intersection, exploitable for both Gaussian (sphere/ellipsoid proxy) and SDF voxel BLASes — relevant to the realtime tier (cited design), not the 780M correctness tier.

The novel-but-grounded claim of this design is the *combination fed by a game engine's live scene*: one path tracer intersecting instanced baked SDF + instanced spectral Gaussians + per-frame scatter Gaussians + terrain SDF, driven by the engine's `InstancedSelector`/`FrameBudgetGovernor` as scene selection — which is the union of two proven techniques, not an unproven invention.

---

## 5. Data Models

```rust
// rust/spectra-scene-state/src/live_update.rs — EXTEND the shipped contract.
// (SceneResource already has Geometry/Material/InstanceBatch/SdfVolume/
//  SdfInstanceBatch/LightSet; DirtyRangeKind already has InstanceTransforms.)

/// Handle to a resident Gaussian-atom instance batch (the engine's appearance
/// primitive, intersected natively — NOT tessellated to quads). New variant.
handle_type!(AtomBatchHandle);

// SceneResource gains:
//   AtomBatch(AtomBatchHandle)   // 16-band-spectral 3D Gaussians, traced as ellipsoids
// DirtyRangeKind gains:
//   AtomTransforms               // per-frame scatter/animation refit
//   AtomSpectral                 // recolour without rebuild
```

```rust
// rust/spectra-renderer — the universal trace config. Additive to the offline
// Renderer; chooses which primitive intersection stages the megakernel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePrimitives {
    TrianglesOnly,         // today's default offline path
    SdfPlusGaussian,       // decision (c): instanced SDF + Gaussian atoms + terrain
}

/// Per-frame proof the universal probe prints (§2). All values measured.
pub struct UniversalFrameReport {
    sdf_instances: u32,
    atoms: u32,
    scatter_blades: u32,
    terrain_chunks: u32,
    facade_coverage_native: f32,   // M0 gate: >= 0.97
    facade_coverage_bridge: f32,   // the lossy-quad oracle, same camera
    hit_color_err: f32,            // mean |rgb - nearest_atom_spectral_rgb|, gate < 0.15
    quads_tessellated: u32,        // MUST be 0 in the universal path
    bvh_builds: u32,               // static builds
    bvh_refits: u32,               // per-frame refits (scatter/animation)
    frame_seconds: f32,            // floor: seconds/frame is acceptable
}
impl UniversalFrameReport {
    pub fn facade_coverage_native(&self) -> f32 { self.facade_coverage_native }
    pub fn quads_tessellated(&self) -> u32 { self.quads_tessellated }
    // ... accessors per field; private fields per the template rule.
}
```

GPU layout constraints (verified against shader, frozen by this design): the megakernel's `GaussianSplat` is 64 floats / 256 B (`splat_intersect.slang:12-22`, `pos`+`opacity`+`scale`+`rotation`+`sh_dc`+`sh_rest[45]`+`spectral_embedding[4]`+pad) — the engine atom-instance upload must pack into this layout (the 16-band engine spectral resolves into `sh_dc` + the 4-band `spectral_embedding`, NOT dropped to material 0). `SDFOpGPU` / instanced `SdfField` headers reuse the `uploader.rs` `sdf_volume_headers` / `sdf_instance_headers` layout already uploaded — the new megakernel reader consumes them; no new upload format.

---

## 6. API

```rust
// rust/spectra-renderer — additive; offline render()/spp loop unchanged.
impl<G: GpuBackend> Renderer<G> {
    /// Configure which native primitive intersection stages the megakernel runs.
    /// SdfPlusGaussian binds g_splat_buffer/g_gaussian_bvh_nodes AND the instanced
    /// SDF reader, and refuses to run the lossy quad bridge.
    /// Errors: kernel compile failure (named kernel + GpuError), missing atom/SDF
    /// buffers when primitives require them.
    pub fn set_trace_primitives(&mut self, p: TracePrimitives) -> Result<(), RenderError>;

    /// Apply a generic SceneUpdateBatch (residency + dirty ranges + output target).
    /// Builds static BLAS for newly-resident SDF/atom assets, refits the TLAS for
    /// InstanceTransforms/AtomTransforms dirty ranges. Game terms never appear here.
    pub fn apply_scene_update(&mut self, batch: &SceneUpdateBatch) -> Result<UpdateStats, RenderError>;
}

// Existing signatures relied on verbatim (do NOT re-derive — verified in code):
//   megakernel:  traverse_gaussian_bvh(Ray, float t_max) -> GaussianHitResult   // splat_intersect.slang
//                gaussian_ray_alpha(Ray, GaussianSplat, out t_hit, out normal) -> float
//                terrain_intersect(o, d, out normal, out chunk) -> float          // terrain_sdf.slang
//   uploader:    pack_sdf_volume_headers(&SdfLayer)->Vec<f32>, pack_sdf_instance_headers(&SdfLayer)->Vec<f32>
//   live_update: SceneUpdateBatch::{new(u64), with_output, make_resident, mark_dirty, remove}
//                SceneResource::{Geometry,Material,InstanceBatch,SdfVolume,SdfInstanceBatch,LightSet}
//                DirtyRange::new(resource, kind, start, count), DirtyRangeKind::InstanceTransforms
//                FrameOutputTarget::GpuResident(RenderTargetHandle)
//   engine:      SdfAtlasGpu::{insert, texture, header_buf}, AssetAtomLibrary::packed_atoms,
//                InstancedSelector::select_resident(...), FrameBudgetGovernor::{new,budget,update}
//   bridge:      vox_render::spectra_bridge::{SpectraSceneUpdateBatch, SpectraSceneResource, ...}
// Threading: render-thread-owned &mut on the GpuContext-owning thread (today's model).
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `spectra_universal_probe` (M0/M1) | new bin target | `crates/vox_render/src/bin/spectra_universal_probe.rs` (or `vox_app`) | the truth harness; prints §2 table; floor-only, no ms gate |
| `set_trace_primitives(SdfPlusGaussian)` | probe setup; game view init | `rust/spectra-renderer/src/renderer.rs` | refuses the quad bridge; binds atom + instanced-SDF buffers |
| Engine-atom → `g_splat_buffer` binding (NEW) | scene upload | `rust/spectra-scene-upload/src/uploader.rs` (+ `spectra-renderer` bind group) | the missing wire: bind what `spectra-gaussian-render` binds, fed from `SceneState`, spectral preserved |
| Instanced-SDF megakernel reader (NEW) | megakernel intersection stage | `~/src/spectra/slang/megakernel.slang` (new `sdf_instance_intersect` block beside the terrain/Gaussian blocks) + new `slang/sdf_instance.slang` | sphere-traces `sdf_volume_headers`/`sdf_instance_headers` the uploader already ships |
| k-NN atom colour gather at SDF hit | the new SDF hit branch | `~/src/spectra/slang/sdf_instance.slang` | reuses the resident atom buffer; resolves 16-band spectral, not grey |
| `apply_scene_update` | engine render thread emits batches | `crates/vox_render/src/spectra_bridge.rs` → `rust/spectra-renderer` | static build vs refit per §4.5 |
| `InstancedSelector::select_resident` → TLAS population | per-frame scene selection | `crates/vox_render/src/atom_instances.rs` → `SceneUpdateBatch` | selection feeds the BVH, not a raster (§4.4 lever 1) |
| Scatter `ScatterField::encode` → atom buffer | per-frame, before trace | `crates/vox_render/src/gpu/scatter_field.rs` | writes Gaussians into the Spectra atom range (disjoint slots) |
| `FrameBudgetGovernor` → TLAS+sample budget | wraps the frame | shared governor type (realtime §4.8 merge obligation) | budget maps to BVH population + (realtime) internal res |
| `splat_convert.rs` quad bridge | M0 comparison oracle ONLY | `crates/vox_render/src/splat_convert.rs` | retired from the universal path; counts quads for the gate |

---

## 8. Open Questions

- [ ] **Gaussian BLAS proxy for HW-RT (realtime tier).** 3DGRT bounds each Gaussian in an icosahedron/proxy mesh so a triangle-RT-core BVH can host it; Blackwell can do ray-sphere natively. Which proxy Spectra adopts is a *realtime* decision (custom intersection vs proxy mesh vs HW ray-sphere) — measured on `tomespensin`, owned by the realtime design. For M0/M1 (compute BVH) the existing `GaussianBvh` SAH suffices.
- [ ] **k-NN colour frequency on near SDF facades.** k≤8 over a coarse asset-local grid gives lower-frequency colour than per-texel; the superseded hybrid recorded this honestly. Measure colour error vs a triangle-mesh reference in M0; a triplanar overlay is a later option, not v1.
- [ ] **SDF-mip vs Gaussian-count as the dominant LOD lever per content class.** Buildings lean SDF-mip; scatter leans Gaussian-count. Whether one budget knob suffices or each class needs its own falloff is measured at M1 over the city scene.
- [ ] **Scatter refit cost in the per-frame AS-update budget.** Regenerating + refitting the scatter blade BVH every frame must fit the capped AS-update budget (realtime §4.5/M5). On the floor (seconds/frame) this is free; on `tomespensin` it is the realtime design's M5 problem — flagged, not solved here.
- [ ] **Does the offline cinematic path also drop the quad bridge immediately, or keep it until the native Gaussian path passes the colour gate?** Decided at M0: the native path must pass the colour gate (< 0.15) before the quad bridge is removed from stills.

---

## 9. Out of Scope

- **The realtime millisecond ladder.** M0–M6 of [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) — HW traversal, ReSTIR, radiance cache, DLSS-RR, the 1–2 ms budget, the scene-doubling gate — are owned, gated, and kill-criteria'd there and validatable only on `tomespensin`. This doc delivers the primitive-correctness precondition; it does not restate, weaken, or re-promise realtime.
- **The Windows/`tomespensin` build reality** (Slang.dll, OptiX FFI, OPAQUE_WIN32 interop, Streamline) — realtime design §4.6.
- **Hardware-RT primitive proxies** (Gaussian icosahedron BLAS, SDF voxel BLAS, Blackwell ray-sphere) — realtime tier, measured on hardware.
- **Game-side asset cook changes** (looser atom pitch now that the SDF carries solidity — superseded hybrid §4.7) — a sequenced game-side follow-up; the engine is pitch-agnostic.
- **Save-format / determinism of the trace** — the scene-feed is deterministic (the engine's existing determinism surface); the trace's own sample sequence is the renderer's concern.
- **2D coverage fields, destruction CSG, navmesh** — SDF Pillar's own later milestones; consumed only as scene data here.

---

## 10. Related Plans / Designs

- **Supersedes:** [Hybrid Atom + SDF LOD Rendering Pipeline](./2026-06-12-hybrid-atom-sdf-lod-design.md) (two wgpu renderers composited by depth — rejected; its tier/colour reasoning salvaged into §4.2–4.4, its mechanism discarded).
- **Depends on:** [Spectra Primary Renderer Bridge](./2026-06-11-spectra-primary-renderer-bridge.md) (the `live_update`/`ViewIntent` ownership contract this extends), [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) + its M1 results (`SdfBaker`/`SdfAtlasGpu`, sealed watertight meshes → sign-correct fields), [Atom-Budget Splat Renderer](./2026-06-06-atom-budget-splat-renderer-design.md) (`AssetAtomLibrary`), [Scatter Field](./2026-06-12-scatter-field-design.md) (the per-frame Gaussian producer), [Virtualized Splat Rendering](./2026-06-10-virtualized-splat-rendering-design.md) (`InstancedSelector` + `FrameBudgetGovernor` as scene selection/LOD).
- **Required before:** the realtime ladder takes over after M1 — [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) M0–M6 on `tomespensin`. This design's own implementation plan is M0 (native one-building) → M1 (full scene), each using `docs/templates/plan.md`.
- **Related:** `spectra-pathtracer-runs` memory (dev-floor run recipe, kernel-dir fix), `spectra-crucible-deps` memory (Slang SDK prereq + build-break gotcha), `aaa-phase2-resident-frame` memory (GpuContext/resident-frame doctrine; the radix-raster nondeterminism honesty rail).
- **SOTA grounding (verified 2026-06-12):** 3DGRT/3DGUT — [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut), [RaySplats arXiv 2501.19196](https://arxiv.org/abs/2501.19196), [GRTX arXiv 2601.20429](https://arxiv.org/pdf/2601.20429) (Gaussians intersected in a path tracer alongside meshes); [Ray Tracing of SDF Grids, JCGT 2022](https://jcgt.org/published/0011/03/06/paper-lowres.pdf) and Sphere-Tracing SDFs with OptiX (SDF custom-intersection inside a path tracer); NVIDIA Blackwell native ray-sphere RT-core intersection (HW support for the non-triangle primitives, realtime tier).

---

## Milestone Ladder (honest, floor-first)

**M0 — Spectra natively renders ONE building CORRECTLY on the 780M (the precondition proof).** Wire the existing native Gaussian path to the engine atom feed AND build the instanced-SDF megakernel reader; trace one real 15,272-atom building (SDF surface + atom colour) on the 780M at seconds/frame.
*Done When:* `spectra_universal_probe --scene one_building --primitive both` prints `facade_coverage native >= 0.97 AND >= 3x lossy-quad-bridge`, `hit color err < 0.15` (spectral preserved), `quads_tessellated=0`, `bridge=none`, and writes a solid coloured-facade PNG. Seconds/frame is acceptable and printed. The lossy-quad bridge is run as the comparison oracle and prints its (low) coverage side by side. **This is the FIX-3 proof done natively in the path tracer, not in a second wgpu pass.**

**M1 — The full engine scene (SDF + atoms + scatter + terrain) feeds ONE path tracer and renders correctly on the 780M.** All four sources through `SceneUpdateBatch`es; static builds for buildings/terrain, per-frame refit for scatter/animation; one trace, one image.
*Done When:* `spectra_universal_probe --scene city_block --primitive both --grass` prints all four source counts, `one renderer, one trace, all four sources -> PASS`, the static-build/refit split (`builds`/`refits`), and writes a city PNG with solid buildings, coloured massing, grass, and ground in a single traced frame — no second renderer, `bridge=none`. Seconds/frame on the floor.

**Then the realtime ladder (4070 Ti) takes over — referenced, not restated.** [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) M0–M6 accelerate the *same* trace of the *same* primitives into the 1–2 ms budget on `tomespensin`. Its kill criteria (M1 HW/SW ratio < 5×, M2 trace subtotal > 1.2 ms, etc.) and the M2 renegotiation trigger govern whether realtime holds — owned there, gated there, validatable only there.
