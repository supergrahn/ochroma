# Design: FloraPrime Audit + the SDF-Primary Foliage Decision — what FloraPrime IS, and how ultra-realistic vegetation renders in an SDF-only path tracer (2026-06-12)

**Status:** Draft
**Scope:** (1) A blunt audit of the standalone repo at `~/src/floraprime` — what it is, what language, what it produces, what state (real vs stub). (2) The load-bearing rendering decision the engine direction forces: the runtime is **SDF-only sphere-traced path tracing** (geometry = SDF; appearance = k-NN atom gather), but foliage — thin translucent leaf cards, fine twigs, grass blades — is *not* natural SDF content. This doc decides, with arithmetic and SOTA, **how vegetation renders**, and reconciles FloraPrime with the existing Forge vegetation cook and the Scatter Field design. Every cost is parameterized on the per-frame ray/sample budget `B` the separate Spectra-realtime perf agent is setting NOW; nothing here is budget-blind.
**Related:**
- The runtime contract this must obey: [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (one path tracer; SDF + atoms; the `MAT_VEGETATION` Gaussian/disc path already in `megakernel.slang`) and the SDF-only override in memory `hybrid-atom-sdf-lod-direction` (Gaussian-AS-GEOMETRY rejected; atoms survive as appearance only).
- The grass machine this reconciles with: [The Scatter Field](./2026-06-12-scatter-field-design.md) (per-tile hash placement, `FrameBudgetGovernor`, T0/T1/T2 collapse — its "Gaussian blade" mechanism is superseded by this doc's leaf-card decision; its density/mask/determinism reasoning survives intact).
- Consumes as scene data: [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) (`SdfBaker`/`SdfAtlasGpu` — sealed watertight mesh → GWN-signed snorm8, ~183 KB/asset).
- Defers realtime ms to: [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (the perf agent's verdict sets `B`).

---

## 1. Problem Statement

Concrete, verified facts (every claim checked against code 2026-06-12):

- **FloraPrime is a real, substantial, biology-grounded vegetation GENERATOR — not a stub, not a dataset, not a renderer.** `~/src/floraprime` is ~14,300 lines: a Rust core crate `floraprime-rs` (~10k LOC, the production engine), a Python package `floraprime_gen` (~2k LOC, an ML research scaffold), a PyO3/maturin binding crate `floraprime-rs-py`, and 21 species `.toml` data files. It produces **tree geometry + physically-based leaf optics**, not pixels. The Rust core is genuine: `qsm.rs::build_branching_qsm` implements six named biological growth mechanisms (apical dominance, self-pruning, phototropism, variable vigor, branch mortality, trunk curvature, root flare) over a validated parent-pointer cylinder DAG; `prospect.rs` is a real PROSPECT-PRO Allen-plate leaf radiative-transfer model (8 bands, Beer–Lambert × Fresnel adding equations, energy-conserving `R+T≤1`); `splatting.rs` turns a QSM into bark + foliage Gaussians with phyllotaxis, petioles, droop, twist, and sun/shade dimorphism. Tests assert real computed outcomes (covariance symmetry to 1e-6, splats on the cylinder surface to 1%, PROSPECT energy conservation, spectral diversity), not `is_some()`.

- **FloraPrime was co-designed with Spectra's vegetation path and then orphaned.** Its `LeafGeometry::shader_code()` emits exactly `{Broadleaf=2, Needle=3, Frond=4, Scale=5, Strap=6}` — byte-identical to Spectra's `splat_hit.slang` `LEAF_*` constants. Its foliage splat carries `spectral_embedding: [R450,R560,R680,T450,T560,T680]` — the exact reflectance/transmittance split `splat_hit.slang::leaf_transmittance` consumes. The repo history says it was "overlaid from aetherspectra" — it is the asset cooker that *was* part of the AetherSpectra/Spectra monorepo, extracted to standalone. **But nothing in Ochroma builds it.** The game's flora cook (`urban_horizon/src/bin/game_asset_cook.rs::cook_flora`) calls the *other*, weaker generator — `forge-vegetation` — not FloraPrime.

- **The engine ships a second, weaker tree generator that the cook actually uses.** `~/src/forge/crates/vegetation` is a 599-line L-system + space-colonization tree mesher (`Species::{Pine,Spruce,Oak,Birch,Shrub,TallGrass,Fern,Cactus}`) emitting a `forge_mesh::Mesh` with material id 0 = trunk/branch and id 2 = leaf billboards. `cook_flora` runs it, then **voxel-clusters the triangle mesh into ~1–2k "Vegetation-channel atoms"** (`atomize_vegetation`) and writes `veg.{species}.atoms.json` as a `FloraAsset`. So today a tree is a ~1–2k-atom blob — far below FloraPrime's 80k–350k-leaf realism (`red_oak.toml: leaf_count = [80000, 350000]`), and lossy (leaf silhouette, translucency, and PROSPECT optics all discarded in the voxel cluster).

- **The runtime cannot render foliage as SDF, and the direction forbids Gaussian-as-geometry.** The override (memory `hybrid-atom-sdf-lod-direction`) is hard: SDF is the *only* geometry primitive; atoms are an appearance field; "Grass/scatter becomes an SDF problem … NOT Gaussian blades." But an SDF of a 350k-leaf oak canopy is either a blobby green mass (no leaf silhouettes, no light-through-leaf translucency) or astronomically expensive at leaf resolution (a snorm8 volume resolving 5 mm leaves over a 12 m crown is ~2400³ voxels ≈ 14 GB — absurd). Unreal's own docs confirm it: two-sided distance fields "work well for foliage but come at a higher ray-marching cost," and even then they are a coarse occlusion proxy, never the hero surface. **Foliage is the one place the SDF-only rule cannot stand unqualified.** This doc resolves exactly that.

- **Spectra ALREADY has the foliage primitive the SDF-only world needs — and it is wired in the megakernel.** `megakernel.slang` lines 2291–2371: on a `MAT_VEGETATION` (id 20) hit it reads `veg_leaf_type`, remaps the hit to disc-local `[-1,1]²`, runs `vegetation_opacity_test_typed` (the per-geometry leaf-silhouette **alpha test** — broadleaf serration, needle, frond ribs, scale diamond, strap), applies `vegetation_albedo_typed` (vein networks), packs `leaf_type → mat.sss_radius`, and (line 2164) treats vegetation backfaces as **translucent two-sided** for real light-through-leaf. This is a *leaf-card* path tracer, already in the production kernel. The open question this doc answers: **is foliage where that non-SDF primitive earns its keep?** (It is.)

---

## 2. Done When

This is an audit + decision doc; its "Done When" is that the implementation plan ([FloraPrime Vegetation Plan](../plans/2026-06-12-floraprime-vegetation-plan.md)) can be executed without re-litigating the foliage primitive. Concretely, the decision is settled when a reader can answer all three:

1. **What is FloraPrime, in one sentence, with no hedging?** → *A standalone, biology-grounded, deterministic Rust vegetation generator (QSM tree skeletons + PROSPECT-PRO leaf optics + per-species data) with an untrained Python diffusion research scaffold bolted on; it produces geometry and spectra, never pixels; it was co-designed with Spectra's `MAT_VEGETATION` leaf-card path and is currently unbuilt by Ochroma.* (§4.1–4.2.)

2. **How does ultra-realistic vegetation render under SDF-only path tracing?** → *Trunk and structural branches as SDF (they ARE solid volumetric geometry — a QSM cylinder is `sdf_capsule` exactly; bake to per-asset snorm8 like buildings). Foliage and grass as the **leaf-card primitive** the path tracer ALSO intersects (alpha-tested oriented discs + translucent PROSPECT BSDF — Spectra's existing `MAT_VEGETATION` path), this being the single justified non-SDF exception. Far-LOD canopy collapses to a **volumetric green shell** (participating medium), never a solid SDF blob.* (§4.3 — the decision, with the arithmetic rejecting the pure-SDF alternatives.)

3. **Where does FloraPrime fit relative to Forge and the Scatter Field?** → *FloraPrime REPLACES forge-vegetation as the cook-time tree generator (it is SpeedTree-class; forge is a 599-line toy). Its QSM trunk feeds `SdfBaker`; its foliage splats become `MAT_VEGETATION` leaf cards; its PROSPECT spectra become the appearance field. The Scatter Field machine is UNCHANGED in structure (hash place, budget cap, LOD collapse) but its per-instance primitive switches from "Gaussian blade" to "leaf-card / strap-leaf"; FloraPrime is the realistic generator whose cooked output the scatter system instances.* (§4.6–4.7.)

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| FloraPrime trunk → SDF, gap-free | bake a `red_oak` QSM (200+ capsule nodes) via `mesh.rs` → `SdfBaker` → snorm8; sphere-trace at 8 m: `assert!(trunk_coverage >= 0.97)` AND `assert!(snorm8_bytes < 256_000)` per asset, both printed | `assert!(sdf.is_some())` |
| Foliage renders as alpha-tested leaf cards, not a blob | trace a backlit oak: `assert!(silhouette_gap_fraction > 0.25)` (lacy sky-gaps between leaves, measured against crown bbox) AND `assert!(backlit_transmittance_mean > 0.10)` (green light through leaves), both printed | `assert!(canopy_pixels > 0)` |
| PROSPECT spectral survives to the rendered SPD | render a summer broadleaf, integrate the hit SPD: `assert!(green_560 > red_680 && red_edge_740 > red_680)` (chlorophyll green peak + red-edge), printed vs the `prospect::compute` reference | `assert!(color != black)` |
| Far canopy collapses to a volumetric shell, not a solid SDF | classify one tree at 20/120/600 m: `assert!(leaf_cards_20 > leaf_cards_120 && leaf_cards_600 == 0 && volumetric_shell_600 == 1)`, printed | `assert!(lod(600) != lod(20))` |
| Foliage cost is budget-bounded, not population-bounded | trace a 50-tree grove at sub-budget `β·B`: `assert!(alpha_test_rays <= beta * B)` and the card→shell distance auto-set, both printed | timing one frame with no count |
| FloraPrime is deterministic (replay-safe cook) | cook `red_oak` seed=42 twice: `assert_eq!(sha256(asset_a), sha256(asset_b))`, printed | `assert!(!atoms.is_empty())` |
| Grass = strap-leaf cards through the same machine | scatter `tall_grass` (LEAF_STRAP) over a tile: `assert!(every_blade.mat == MAT_VEGETATION && every_blade.leaf_type == 6)`, printed | `assert!(blade_count > 0)` |

---

## 4. Architecture

### 4.1 What FloraPrime IS — the blunt inventory (file by file, verified)

**Rust core (`rust/floraprime-rs/src/`, ~10k LOC — REAL, production-grade):**

| Module | LOC | What it actually is | State |
|---|---|---|---|
| `qsm.rs` | 902 | Quantitative Structure Model: `CylinderNode {position, direction, radius, length, branch_order, parent_idx, depth_*, sub_tree_volume, angles, joint_type}`; `build_branching_qsm(BranchingConfig)` — parametric tree with 6 biological mechanisms; `validate()` (DAG, taper, acyclicity); `compute_derived_features()`. | **Real.** This is SpeedTree-class procedural structure, deterministic from a seed. |
| `prospect.rs` | 253 | PROSPECT-PRO leaf optics: `compute(LeafParams{n,cab,car,cbrown,cw,cm,anth}) -> SpectralRt{reflectance[8], transmittance[8]}` over bands 380–780 nm via Allen plate model (Fresnel + Beer–Lambert + adding equations). Energy-conserving. | **Real, physically-based.** The realism engine. The one thing forge-vegetation has no analogue for. |
| `splatting.rs` | 1309 | `qsm_to_bark_splats` (cylinder-surface discs) + `qsm_to_foliage_splats` (leaf Gaussians along branches: phyllotactic golden-angle, petiole offset, curl/droop, twist, sun/shade size dimorphism). Emits `GaussianSplatCpu{position, covariance[3][3], opacity, leaf_type:i32, spectral_embedding:[f32;6]}`. `CrownShape{Sphere,NarrowEllipsoid,Cone,Columnar}`. | **Real.** The leaf-card source. Note the module doc: it "mirrors two Slang shaders `tree_to_splats.slang` / `scatter_foliage.slang`" — it was built to feed Spectra. |
| `species.rs` | 2317 | `SpeciesRegistry` (data-driven, runtime-extensible), `LeafGeometry{Broadleaf=2..Strap=6}`, `SeasonalProspect{spring,summer,autumn,winter}` with `interpolate(season)`, `ParamBounds`, MLP-output validation (`validate_params`). Loads the 21 TOML species. | **Real.** The species DB. `shader_code()` matches Spectra `LEAF_*` exactly. |
| `mesh.rs` | 524 | `qsm_to_mesh -> TreeMesh{vertices,normals,uvs,indices}` — radius-continuous, tapered, junction-sealed tube mesh, max-branch-order filter (drops tiny twigs). | **Real.** This is the watertight mesh the `SdfBaker` needs (the trunk→SDF bridge). |
| `cage.rs` | 496 | `qsm_to_control_cage -> ControlCageOutput` — Catmull-Clark quad cage with crease edges at branch forks. | **Real.** A subdivision-surface path (not needed for SDF-only runtime; salvage value: clean cage for offline/USD). |
| `scatter.rs` | 1822 | `scatter_species_forest(name, positions, rotations)` — replicate a species QSM template at world positions with jitter + PROSPECT color. Pure-Rust replacement for the old `species_scatter.py`. | **Real.** Overlaps the Ochroma Scatter Field's job — reconciled in §4.7. |
| `modal_wind.rs` + `lbs.rs` | 320+296 | Modal-analysis wind (`ModalBasis`, `integrate_modal_step`, `modal_to_joint_displacement`) + linear blend skinning over the QSM skeleton. | **Real.** Wind animation; feeds the per-instance phase the Scatter Field already plans for. |
| `spectral_pca.rs` | 566 | `SpeciesClass`, PCA spectral basis (compress 8-band PROSPECT → compact embedding). | **Real.** Relevant to the engine's spectral packing seam. |
| `fern.rs`, `gpu_scatter.rs` | 204+382 | Fern leaf-disc geometry; cudarc GPU scatter. | Real but peripheral (`gpu_scatter` is CUDA — Ochroma is wgpu/Vulkan; do not port). |
| `usd.rs` | 205 | `TemporalTree`/`SkeletonSnapshot` + `export_temporal_tree_json`. | **STUB.** Module doc: "Stub implementation — full USD export blocked on crucible-usd Phase 3." Emits a JSON *summary* (ages, node counts, PROSPECT params), not a USD stage. Matches the memory `civitas-usd-interchange-decision` (USD is interchange-only; not the runtime). Harmless. |

**Python (`python/floraprime_gen/`, ~2k LOC — RESEARCH SCAFFOLD, untrained):**
- A MiDi-style **graph diffusion** generative model for QSMs: `graph_diffusion.py` (topology + geometry diffusion, MST projection per reverse step), `model/{topology,geometry,denoiser,conditioning}.py` (SE(3)-equivariant denoiser, EDM continuous diffusion), `data/{qsm_dataset,augmentation}.py`, `spectral/pca_basis.py`, `foliage/point_diffusion.py`, `training/trainer.py`.
- **No trained checkpoints exist** (`find` for `*.pt/*.pth/*.safetensors/*.ckpt` → empty). This is *aspirational* — a learned generator that, once trained on QSM datasets (the code references BioDiv-3DTrees/TreeQSM-class data), would produce novel species variety. **Not on the production path.** The deterministic parametric `build_branching_qsm` + species TOML is the shippable generator.

**Verdict (blunt):** FloraPrime is a **strong, real, deterministic vegetation generator with the best leaf-optics model in the building** — and an unbuilt, orphaned, untrained-ML-scaffolded standalone repo that Ochroma never compiles. It is 80% of a SpeedTree competitor's *generation* side. It has *zero* renderer (correctly — Spectra is the renderer). Its single most valuable, non-replicable asset is `prospect.rs` + the seasonal species DB.

### 4.2 The seam that already exists — FloraPrime was built for this renderer

The match is not a coincidence:

| FloraPrime emits | Spectra consumes | Status |
|---|---|---|
| `LeafGeometry::shader_code()` ∈ {2,3,4,5,6} | `splat_hit.slang::LEAF_{BROADLEAF,NEEDLE,FROND,SCALE,STRAP}` ∈ {2,3,4,5,6} | **identical** |
| foliage `spectral_embedding[R450,R560,R680,T450,T560,T680]` | `splat_hit.slang::leaf_transmittance` (per-geometry R/T) + `MAT_VEGETATION` BSDF | **same R/T split** |
| `GaussianSplatCpu{position, covariance, opacity, leaf_type}` | `splat_intersect.slang::GaussianSplat` (256 B) + `megakernel` `MAT_VEGETATION` disc path | **lossy-but-mappable** (covariance→quat/scale via `covariance_to_quat_scale`, already in `splatting.rs`) |
| QSM `CylinderNode{position,direction,radius,length}` | `sdf_eval.slang::sdf_capsule(p, a, b, radius)` | **exact** (`a = position`, `b = position + direction·length`) |

The only gaps are *wiring* (nobody runs FloraPrime in the cook) and *spectral width* (FloraPrime exports a 6-float compaction; the engine wants its 16-band / Spectra's 4-band `spectral_embedding` + `sh_dc`). Both are mechanical (§ plan).

### 4.3 THE VEGETATION CRUX — how foliage renders under SDF-only, decided with arithmetic + SOTA

**The fork (from the task):**
- **(a)** trunk/branches as SDF + foliage as a SEPARATE primitive the tracer ALSO intersects (alpha-tested leaf cards / oriented discs + translucent BSDF).
- **(b)** SDF-density volumetric canopy (leaves as a green absorbing/scattering participating medium).
- **(c)** SDF silhouette (crown envelope) + a procedural leaf shader that fakes leaf detail at the hit.

**Decision: (a) for trunk + near/mid foliage, with (b) as the FAR-LOD tier ONLY. (c) is rejected outright.** This is not a hybrid of two *renderers* (the thing the user rejected) — it is one path tracer intersecting two primitive *types* per ray (SDF surfaces + alpha-tested vegetation discs), plus a volumetric medium for the far tier, exactly as the Universal Renderer already composites SDF + Gaussian + triangle in one `hit_t`/`throughput` loop. The "exception" to SDF-only is foliage, and it is justified below.

**Why a leaf is not an SDF (the structural argument).** A leaf is a ~0.1–0.3 mm-thick, ~5–15 cm, *two-sided translucent* surface. Three things define photoreal foliage and an SDF kills all three:
1. **Silhouette.** Realistic foliage is *lacy* — you see sky through thousands of leaf-gaps, and the crown edge is a stochastic stipple, not a smooth boundary. An SDF crown is a closed manifold: its edge is a smooth blob. (c) and a coarse (b) both fail here.
2. **Translucency.** A backlit leaf glows green because light transmits through it (PROSPECT's `T` channel). The SDF-only appearance model (k-NN atom gather → opaque surface shade) has no transmission term; you get a dark opaque leaf. Spectra's `MAT_VEGETATION` path *does* (line 2164 double-sided + `leaf_transmittance`).
3. **Sub-pixel AA.** Almost every leaf in a frame is sub-pixel. An alpha-tested oriented disc (or anisotropic Gaussian) integrates to a sub-pixel footprint cleanly (Mip-Splatting line); a sphere-traced SDF leaf needs supersampling to not shimmer.

**Why (a) is the production-proven answer (SOTA, with the cost datum).** Every shipping path-traced-foliage system uses alpha-tested leaf cards/instances, *not* SDF leaves:
- **Disney *Moana* on a single AMD GPU (2024):** multi-level instancing (leaf instanced on a tree, tree instanced in a forest) via HIP RT 2.2 — alpha-tested leaf geometry, instanced, in a path tracer. The canonical "billions of leaves" path tracer.
- **Indiana Jones (Dec 2024), full path tracing:** alpha-tested vegetation is the dominant `TraceMain` cost; **Opacity Micromaps cut it from 7.90 → 3.58 ms on an RTX 5080 (−55%)** by skipping anyhit on provably-opaque/transparent micro-regions. This is the single most important budget datum in this doc (§ budget below).
- **The Witcher 4 RTX Mega Geometry (2025):** even "real geometry" leaves (10 M tris/tree) are clustered alpha-tested geometry in the BVH, not SDF.
- **Distant foliage → volumetric / stochastic aggregate:** Cook et al. "Stochastic Simplification of Aggregate Detail" (prune leaves, enlarge survivors, preserve aggregate albedo) and volumetric-canopy LOD ("distant trees use a volumetric technique that avoids geometry, single ray per pixel") — this is exactly tier (b), and it is a FAR-LOD, not a near representation.

Nobody ships SDF-leaf foliage because (b)-as-near loses silhouette/translucency and (c) loses both. The SDF wins decisively for *trunk and structural branches* — they ARE solid volumetric wood, `sdf_capsule` is exact, gap-free, and cheap — and loses decisively for *leaves*. So the honest split is **wood = SDF, leaves = cards, far canopy = volume.**

**The cost arithmetic, parameterized on the perf agent's budget `B` (rays/frame).** Let:
- `B` = total primary+secondary rays/frame the perf agent's verdict allows (e.g. at 1280×720 internal × 1 spp + a few bounces, `B ≈ 1–4 M` rays for a 1–2 ms 4070-Ti frame).
- `β` = foliage's fraction of `B` (the `FrameBudgetGovernor` sub-budget; the lever).
- `L̄` = mean leaf-card overlaps per ray that enters a canopy (depth complexity); near hero tree `L̄ ≈ 8–30`, stochastically reduced with distance.
- `c_at` = ALU cost of one leaf alpha-test (`splat_hit` opacity tests ≈ 30–60 ALU); with OMM on Ada/Blackwell, ~55 % of tests are skipped → effective `c_at_eff ≈ 0.45·c_at` (the Indiana Jones datum).

Foliage trace cost ≈ `(β·B) · L̄ · (c_at_eff + c_bvh)`. **Two anchors fix the constants:**
- *Whole-frame anchor:* Indiana Jones vegetation `TraceMain` = 3.58 ms (OMM) on a 5080 at ~4K. A 4070 Ti is ≈ 0.55–0.65× a 5080 and we target 1280×720-internal (≈ 1/4 the 4K pixels) → an *un-budgeted* vegetation-dense frame's foliage trace ≈ **2–4 ms on the 4070 Ti**. That is already over a 1–2 ms total. **Therefore foliage MUST be budgeted** — `β·B` capped, `L̄` reduced by distance LOD, far tiers volumetric. This is not optional; it is the whole reason the card→shell governor exists.
- *Per-tier affordance:* given `β·B` alpha-test rays, the near tier affords `N_near = (β·B)/(L̄·c_at_eff·k)` leaf-card *hits* before exceeding budget (`k` = the trace-overhead multiplier, measured at M-cost milestone). The **card→volumetric-shell transition distance `d_v`** is set so the integrated `L̄` over the visible canopy stays under `β·B` — i.e. exactly the Scatter Field's governor mechanism, now over leaf cards instead of Gaussian blades.

**Why not pure (b) for near.** A volumetric canopy is `march_steps × medium_eval` per ray = cheap and *great far* (one ray, no per-leaf BVH — the SOTA forest-LOD result), but near it cannot produce the lacy silhouette or the crisp single-leaf translucency; it reads as green fog. Use it as the FAR shell (the card→shell collapse), never the hero.

**Why not (c).** SDF crown envelope + procedural leaf shader gives ONE cheap SDF hit but: a smooth blob silhouette (no sky-gaps), no real transmission, and parallax-faked "leaves" that break under a moving camera and secondary rays (shadows/GI see a solid blob). It fails all three realism tests in §3. Rejected.

**Memory arithmetic (per UNIQUE tree asset, 780M UMA-bound, parameterized).** A `red_oak` is `leaf_count ∈ [80k, 350k]`. Options:
- Trunk SDF (snorm8, SDF Pillar): **~183 KB/asset** (gap-free wood). Cheap. Always resident.
- Foliage cards: a hero tree at 350k cards × (oriented disc ≈ 64–96 B packed) ≈ **22–34 MB/asset at full density** — too heavy to store per unique asset at scale. So foliage is **cooked as a representative card SET** (a few k canonical leaf cards + per-branch attachment frames) and **scatter-generated to fill** under budget (the Scatter Field per-tile generation, §4.7), exactly like grass — NOT stored at 350k. Resident foliage per visible tree = `min(N_near, governor cap)`, not population. This is the same population-vs-coverage collapse the Scatter Field already banks.

### 4.4 The runtime data flow under the decision (one path tracer, two primitive types + a volume tier)

```
 cook time (FloraPrime, deterministic):
   species TOML + seed
        └─ build_branching_qsm ──► QSM (capsule skeleton)
              ├─ mesh.rs (watertight tube) ──► SdfBaker ──► snorm8 SDF  [WOOD = SDF]
              ├─ qsm_to_foliage_splats   ──► canonical leaf-card set + attachment frames
              │                              (leaf_type, PROSPECT R/T, covariance→quat/scale)  [LEAVES = CARDS]
              └─ prospect::compute(season) ──► per-species/season spectral appearance field

 run time (Spectra megakernel, per ray):
   sphere-trace instanced trunk SDF        ── gap-free wood surface, k-NN atom colour at hit
   ∪ traverse MAT_VEGETATION leaf-card BVH  ── alpha-test silhouette + translucent PROSPECT BSDF  (near/mid)
   ∪ march far-canopy volumetric shell       ── green participating medium, 1 ray/px            (far)
   → one hit_t / throughput / spectral film  (the existing megakernel composite, lines 971–2371)
```

The "two primitive types in one trace" is the Universal Renderer's decision (c) — *but* with the user's SDF-only override honored: SDF is the only *geometry/surface* primitive (wood, buildings, terrain); the leaf card is explicitly the *one justified non-SDF exception* (a translucent thin surface an SDF cannot represent), and it is exactly the primitive Spectra already intersects. No second renderer, no depth composite — the megakernel's own nearest-hit/transmittance loop reconciles them, as it does for triangles + Gaussian + terrain-SDF today.

### 4.5 Appearance — PROSPECT IS the foliage appearance field (atoms are for wood)

The Universal Renderer colours an SDF hit by k-NN atom gather. For **wood**, that holds (bark atoms colour the trunk SDF). For **leaves**, the appearance is not a gathered atom colour — it is the **PROSPECT BSDF**: front-face Lambertian+GGX-cuticle with the species/season reflectance, back-face chromatic transmission with `leaf_transmittance`. FloraPrime *computes* this physically (`prospect::compute` → seasonal R/T); the cook bakes per-species/season spectra into the leaf-card material, and `splat_hit.slang` already evaluates it. So: **wood appearance = atom k-NN (Universal Renderer §4.3); leaf appearance = PROSPECT (FloraPrime + `MAT_VEGETATION`).** Two appearance models, one per primitive, no conflict.

### 4.6 Reconciliation with Forge — FloraPrime replaces forge-vegetation as the generator

`forge-vegetation` (599 LOC L-system) and FloraPrime (10k LOC QSM + PROSPECT) both make trees. They are not peers:

| | forge-vegetation | FloraPrime |
|---|---|---|
| Structure | L-system + space colonization, triangle mesh | QSM cylinder DAG, 6 biological mechanisms, validated taper |
| Leaf optics | none (material id 2 "billboard", flat color) | PROSPECT-PRO 8-band R/T, seasonal, per-species |
| Species | 8 hardcoded enums | 21 data-driven TOML, runtime-extensible |
| Realism | placeholder | SpeedTree-class |
| Used by cook today | **yes** (`cook_flora`) | no |

**Decision:** FloraPrime becomes the cook-time generator; `forge-vegetation` is **demoted to a fast preview / fallback** (kept for cheap placeholder trees during iteration, retired from the product cook). The civitas `cook_flora` path (`forge.generate_vegetation → atomize_vegetation → veg.*.atoms.json`) is replaced by `floraprime → {SdfBaker trunk, leaf-card set, PROSPECT appearance}`. The `FloraAsset` registration seam stays (still not a `BuildingAsset`); only the producer behind it changes.

### 4.7 Reconciliation with the Scatter Field — same machine, the primitive switches to leaf cards

The Scatter Field design's *mechanism* is fully intact and is exactly what foliage needs: per-visible-tile hash placement, `density × clearance-SDF × CellClass` mask, distance-LOD into T0/T1/T2, one `FrameBudgetGovernor`, far-tile collapse to a shell, determinism by `(tx,tz,k)` hash, wind by per-instance phase. The **only** change the SDF-only override forces is the per-instance primitive:

| Scatter Field said | Under this decision |
|---|---|
| grass blade = anisotropic **Gaussian** `GaussianSplat::volume` | grass blade = **strap-leaf card** (`MAT_VEGETATION`, `leaf_type = LEAF_STRAP = 6`), alpha-tested + translucent |
| far tile → flat ground-tint **Gaussian** shell | far tile → **volumetric green shell** (participating medium, §4.3 tier b) |
| prototype scatter-class = cooked **atom set** | prototype scatter-class = FloraPrime **leaf-card set + trunk SDF** (trees) or strap-card (grass) |

FloraPrime is the realistic generator whose cooked output the Scatter Field instances: a tree scatter-class is a FloraPrime asset (trunk SDF + canonical leaf cards); a grass scatter-class is FloraPrime's `tall_grass`/dry-forb PROSPECT params driving strap-cards. The Scatter Field's "billion-blade field at fixed budget, never billion per frame" claim is unchanged — leaf cards collapse to volumetric shells with distance exactly as blades did, and the `β·B` alpha-test ceiling (§4.3) is the honest per-frame cap.

### 4.8 Threading / device / ownership

- **Cook is offline, host-side, deterministic.** FloraPrime runs in the asset cook (`game_asset_cook`-class binary), single-threaded-deterministic per asset (rayon-parallel *within* an asset is fine — its outputs are order-independent: bark/foliage splats collected by `flat_map`, mesh by parallel ring gen). Output: `{snorm8 SDF, leaf-card set, PROSPECT appearance, attachment frames}` written to the asset pack. No GPU, no engine runtime dependency at cook time.
- **Runtime is Spectra's, unchanged in ownership.** The trunk SDF rides the `SdfAtlasGpu`/`SdfInstanceBatch` upload (Universal Renderer §4.5); leaf cards ride the `MAT_VEGETATION` Gaussian/disc upload (the `g_splat_buffer` bind the Universal Renderer is wiring); the far shell is a volumetric medium the megakernel marches. Render-thread-owned `&mut`, no new locks. The Scatter Field generator emits leaf cards into the same `g_splat_buffer` range per frame, as it already plans to for blades.
- **Drop the CUDA path.** `gpu_scatter.rs` (cudarc) does not belong in a wgpu/Vulkan engine; the Scatter Field's wgsl/Slang generation replaces it. `cage.rs` (Catmull-Clark) and `usd.rs` (stub) are not on the runtime path; keep `cage.rs` for offline/interchange, leave `usd.rs` as-is until crucible-usd lands.

### 4.9 SOTA grounding (verified 2026-06-12)

- **Path-traced foliage = alpha-tested instanced leaf cards, accelerated by OMM.** Disney Moana multi-level instancing on AMD/HIP RT 2.2; Indiana Jones Opacity Micromaps 7.90→3.58 ms on RTX 5080 (the budget datum); Witcher 4 RTX Mega Geometry clustered alpha-tested leaves. Nobody ships SDF-leaf foliage.
- **Distant foliage = stochastic aggregate / volumetric canopy.** Cook–Halstead "Stochastic Simplification of Aggregate Detail"; NVIDIA "Appearance-Driven Automatic 3D Model Simplification" (arXiv 2104.03989, differentiable leaf simplification, 1.7M→6.5k tris at preserved appearance); volumetric-canopy forest LOD ("single ray per pixel"). This is tier (b)-as-far.
- **Two-sided distance fields for foliage are an occlusion proxy, not the hero surface** (Unreal docs: "works well for foliage but higher ray-marching cost") — confirms SDF leaves are a coarse approximation, never photoreal foliage.
- **QSM is the established realistic tree structure** (InverseTampere/TreeQSM, AdQSM, SmartQSM 2026, L1-Tree 2024, SfQSM 2025) — FloraPrime's `qsm.rs` is in this lineage; cylinder hierarchies → `sdf_capsule` is exact.
- **3D-Gaussian vegetation with structure + LOD is current** (GaussianPlant arXiv 2512.14087 — structure-aligned GS extracting branch structure + leaf instances; LoD-of-Gaussians) — confirms the appearance-field framing (atoms/Gaussians as appearance) and that leaf-wise structure is recoverable, but the runtime *geometry* stays SDF wood + leaf cards per the override.

The novel-but-grounded combination: a biology-grounded QSM generator (FloraPrime) feeding an SDF-only path tracer as **wood-SDF + alpha-tested PROSPECT leaf cards + volumetric far-canopy**, budget-governed by the same `FrameBudgetGovernor` as buildings and grass — the union of three proven techniques (QSM structure, path-traced leaf cards + OMM, volumetric aggregate LOD), not an unproven invention.

---

## 5. Data Models

```rust
// The cook-time vegetation asset FloraPrime produces for the SDF-only runtime.
// Engine-generic: no game terms. Lives in the asset pack the cook writes.
/// One cooked vegetation asset: wood-as-SDF + leaves-as-cards + PROSPECT appearance.
pub struct CookedVegetationAsset {
    species_id: u32,                  // private — .species_id()
    trunk_sdf: SdfFieldHandle,        // snorm8 volume from mesh.rs → SdfBaker (~183 KB)
    leaf_cards: Vec<LeafCard>,        // canonical card set (a few k), scatter-filled at runtime
    appearance: SeasonalAppearance,   // PROSPECT R/T per season, per leaf_type
    bounds: [f32; 6],                 // crown AABB for LOD / shell
}

/// One alpha-tested oriented leaf disc — the MAT_VEGETATION primitive.
/// Maps to Spectra splat_intersect.slang::GaussianSplat (256 B) on upload.
pub struct LeafCard {
    position: [f32; 3],               // private accessors throughout
    quat: [f32; 4],                   // from covariance_to_quat_scale (already in splatting.rs)
    scale: [f32; 3],                  // disc half-extents (anisotropic)
    opacity: f32,
    leaf_type: i32,                   // LeafGeometry::shader_code() ∈ {2..6} == Spectra LEAF_*
    spectral: SpectralAppearance,     // PROSPECT R/T → engine spectral (widen from the 6-float export)
}

/// PROSPECT-derived appearance — NOT a k-NN atom gather (that is for wood).
pub struct SeasonalAppearance {
    // per season (spring/summer/autumn/winter): reflectance[B] + transmittance[B]
    // B = the engine spectral width (16-band native / Spectra 4-band embedding + sh_dc),
    // widened from FloraPrime's lossy [R450,R560,R680,T450,T560,T680] 6-float export.
}
```

GPU layout constraints (verified, frozen): leaf cards pack into Spectra's `GaussianSplat` (256 B, `splat_intersect.slang:12-22`); `leaf_type` rides `mat.sss_radius` (megakernel:2370) and `velocity.y` (megakernel:2291) per the existing path; trunk SDF reuses the `uploader.rs` `sdf_volume_headers`/`sdf_instance_headers` layout. No new GPU format — the cook targets layouts the megakernel already reads.

---

## 6. API

```rust
// Cook-time entry (host, deterministic). New thin crate `vox_data::flora_cook`
// wrapping floraprime-rs; engine-generic, no game terms.
pub fn cook_vegetation(
    species: &SpeciesTemplate,   // floraprime-rs species (from TOML)
    seed: u32,
    season: f32,                 // 0=spring..0.75=winter (SeasonalProspect::interpolate)
) -> Result<CookedVegetationAsset, FloraCookError>;
// Deterministic: same (species, seed, season) → byte-identical asset (sha256 gate).

// Existing FloraPrime signatures relied on verbatim (verified in code — do NOT re-derive):
//   build_branching_qsm(cfg: &BranchingConfig) -> TreeQSM
//   qsm_to_mesh(qsm: &TreeQSM, max_order: u32) -> TreeMesh           // → SdfBaker input
//   qsm_to_foliage_splats(qsm, total_leaf_count, leaf_w, leaf_h, light_dir,
//        tree_seed, leaf_geometry, foliage_cluster_ratio, petiole, droop, twist)
//        -> Vec<GaussianSplatCpu>
//   covariance_to_quat_scale(cov: &[[f32;3];3]) -> ([f32;4], [f32;3])
//   prospect::compute(p: &LeafParams) -> SpectralRt { reflectance[8], transmittance[8] }
//   SeasonalProspect::interpolate(season: f32) -> LeafParams
//   LeafGeometry::shader_code(&self) -> i32          // {2,3,4,5,6} == Spectra LEAF_*
// Existing engine signatures (verified):
//   SdfBaker: sealed watertight Mesh -> snorm8 SdfField (~183 KB)   // SDF Pillar
//   SdfAtlasGpu::{insert, texture, header_buf}
//   ScatterField::encode(...) -> ScatterStats                        // Scatter Field §6
// Existing Spectra runtime (verified, no change to these):
//   megakernel.slang: MAT_VEGETATION path (2291–2371), traverse_gaussian_bvh (1794),
//                     double-sided veg transparency (2164)
//   splat_hit.slang: vegetation_opacity_test_typed, vegetation_albedo_typed,
//                    leaf_transmittance, eval_vegetation_bsdf_typed
//   sdf_eval.slang: sdf_capsule(p, a, b, radius)                     // QSM cylinder == this
// Threading: cook is host/offline, rayon-within-asset OK (order-independent outputs).
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `cook_vegetation` | the asset cook | `crates/vox_data/src/flora_cook.rs` (new) wrapping `~/src/floraprime` | replaces the `forge.generate_vegetation` call in `cook_flora` |
| FloraPrime `qsm_to_mesh` → `SdfBaker` | `cook_vegetation` | `flora_cook.rs` → SDF Pillar `SdfBaker` | wood-as-SDF, ~183 KB/asset |
| FloraPrime `qsm_to_foliage_splats` → `LeafCard` set | `cook_vegetation` | `flora_cook.rs` (via `covariance_to_quat_scale`) | leaf cards for `MAT_VEGETATION` |
| `prospect::compute(season)` → `SeasonalAppearance` | `cook_vegetation` | `flora_cook.rs` | the foliage appearance field (not atoms) |
| trunk SDF instance | runtime SDF intersect | Spectra `uploader.rs` + megakernel SDF reader (Universal Renderer M0) | gap-free wood |
| leaf-card upload → `g_splat_buffer` | runtime veg intersect | Spectra `MAT_VEGETATION` path (already in megakernel) | alpha-test + PROSPECT BSDF |
| far-canopy volumetric shell | far-LOD tier | Spectra megakernel volume march (new far tier) | Cook stochastic-aggregate / volumetric LOD |
| tree/grass scatter-class = FloraPrime asset | `ScatterField` per tile | `crates/vox_render/src/gpu/scatter_field.rs` | leaf-card primitive replaces "Gaussian blade" |
| `cook_flora` (game) | game asset cook | `urban_horizon/src/bin/game_asset_cook.rs` | calls `vox_data::flora_cook`, not forge |
| `forge-vegetation` | preview/fallback only | `forge/crates/vegetation` | demoted, not retired in-code |

---

## 8. Open Questions

- [x] **Foliage primitive under SDF-only?** → leaf cards (a) near/mid + volumetric shell (b) far; wood = SDF; (c) rejected. (§4.3.) **Resolved.**
- [x] **FloraPrime vs Forge?** → FloraPrime is the generator; Forge demoted to preview. (§4.6.) **Resolved.**
- [x] **FloraPrime vs Scatter Field?** → Scatter mechanism unchanged; per-instance primitive becomes leaf-card. (§4.7.) **Resolved.**
- [ ] **Spectral widening.** FloraPrime exports 6-float `[R450,R560,R680,T450,T560,T680]`; the engine wants 16-band / Spectra's `sh_dc`+4-band. Carry full 8-band PROSPECT (FloraPrime computes it internally) and let the cook compress to the engine width — decide the exact mapping at cook M1 (the `spectral_pca.rs` basis is a candidate compressor).
- [ ] **Leaf-card cook density vs scatter-fill density.** How many canonical cards to bake per asset (a few k) vs scatter-generate per visible tree (up to the `β·B` budget)? Measure at the cost milestone — the Scatter Field's per-tile generation answers it, but the canonical-set size that reads correctly at the card→shell distance is empirical.
- [ ] **Untrained Python diffusion generator.** Keep as a research lever for novel species variety (needs a QSM training set), or delete to reduce surface? Decide after the parametric generator ships — it is not on the production path either way.
- [ ] **Wind under the leaf-card path.** FloraPrime `modal_wind.rs` (modal LBS over the QSM) vs the Scatter Field's per-instance hash-phase sway. The skeleton-modal wind is richer for hero trees; the hash-phase is free for mass scatter. Pick per tier at the wind milestone.

---

## 9. Out of Scope

- **The realtime millisecond ladder.** `B` is set by the Spectra-realtime perf agent on the 4070 Ti; this doc parameterizes cost on `B`, it does not set it.
- **Training the Python diffusion model.** No checkpoints exist; the parametric generator is the shippable path. Training is a future research effort, not this plan.
- **USD export.** `usd.rs` stays a stub until crucible-usd; USD is interchange-only (memory `civitas-usd-interchange-decision`), never the runtime.
- **The CUDA `gpu_scatter.rs`.** Not ported; the engine's wgsl/Slang Scatter Field replaces it.
- **Catmull-Clark subdivision runtime.** `cage.rs` is kept for offline/interchange; the runtime is SDF wood + leaf cards, not subdivision surfaces.
- **Animation rigging beyond wind** (growth-over-time, breakage) — `usd.rs::TemporalTree` exists but is downstream of getting one tree to render correctly.

---

## 10. Related Plans / Designs

- **Implements into:** [FloraPrime Vegetation Plan](../plans/2026-06-12-floraprime-vegetation-plan.md) (the get-into-shape milestones M0–M4, each with a realism + cost metric).
- **Depends on:** [Spectra Universal Renderer](./2026-06-12-spectra-universal-renderer-design.md) (the one-tracer / SDF + Gaussian primitive seam, the `g_splat_buffer` wiring, the SDF instance reader), [The SDF Pillar](./2026-06-10-sdf-pillar-design.md) (`SdfBaker`/`SdfAtlasGpu` for wood), [The Scatter Field](./2026-06-12-scatter-field-design.md) (the per-tile generation / budget / LOD machine foliage rides).
- **Supersedes (partially):** the Scatter Field's "Gaussian blade" per-instance primitive (now a strap-leaf card); `forge-vegetation` as the product tree generator (demoted to preview).
- **Required before:** the realtime ladder accelerates the *same* leaf-card + SDF trace into `B` on `tomespensin` — [Spectra Realtime](./2026-06-10-spectra-realtime-design.md) (sets `B`, owns OMM/anyhit acceleration for the leaf cards).
- **SOTA grounding (verified 2026-06-12):** Disney Moana on AMD/HIP RT 2.2 (multi-level instanced leaf cards); [Indiana Jones OMM path-tracing optimizations, NVIDIA](https://developer.nvidia.com/blog/path-tracing-optimizations-in-indiana-jones-opacity-micromaps-and-compaction-of-dynamic-blass/) (7.90→3.58 ms vegetation, the budget datum); Cook–Halstead stochastic aggregate simplification + [NVIDIA appearance-driven simplification arXiv 2104.03989](https://arxiv.org/pdf/2104.03989) (far-canopy LOD); [GaussianPlant arXiv 2512.14087](https://arxiv.org/abs/2512.14087) (structure-aligned vegetation appearance); TreeQSM / AdQSM lineage (QSM structure → `sdf_capsule`); Unreal two-sided distance fields for foliage (SDF-leaf = occlusion proxy only).
