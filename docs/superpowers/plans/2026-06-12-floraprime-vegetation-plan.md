# FloraPrime Vegetation — Get-Into-Shape Plan (SDF wood + PROSPECT leaf cards) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the orphaned, real, biology-grounded FloraPrime generator into the Ochroma cook so that ultra-realistic vegetation renders in the SDF-only Spectra path tracer as **SDF wood (trunk/branches) + alpha-tested PROSPECT leaf cards (foliage/grass) + a volumetric far-canopy shell**, budget-governed by the one `FrameBudgetGovernor`, replacing the 599-line forge-vegetation toy the cook uses today.
**Done When:** `cargo run --release --bin flora_probe -- --species red_oak --season summer --camera backlit --gate-rays $B` prints a realism block (`silhouette_gap=<g≥0.25> translucency=<t≥0.10> spectral: green560>red680 & redEdge740>red680 PASS`), a cost block (`alpha_test_rays=<r ≤ β·B> card→shell d_v=<d>m frame=<ms> @ tier=<adapter>`), a determinism block (`cook sha256 identical across 2 runs`), and writes `flora_red_oak_backlit.png` showing a backlit oak with **lacy sky-gaps between leaves, green light glowing through the canopy, and a solid gap-free trunk** — verifiable by a human at the keyboard without reading code. forge-vegetation no longer appears in the product cook path.
**Architecture:** FloraPrime stays the deterministic offline GENERATOR. A new `vox_data::flora_cook` wraps it: QSM → `mesh.rs` → `SdfBaker` (wood SDF, ~183 KB/asset); QSM → `qsm_to_foliage_splats` → `LeafCard` set (`MAT_VEGETATION`, `leaf_type`, covariance→quat/scale); `prospect::compute(season)` → `SeasonalAppearance`. The Spectra megakernel already intersects the leaf-card primitive (lines 2291–2371) and double-sided translucency (2164); the trunk SDF rides the Universal Renderer's SDF instance reader; the far shell is a new volumetric march tier. The Scatter Field instances FloraPrime assets with the per-instance primitive switched from "Gaussian blade" to "leaf card". Every cost is parameterized on the perf agent's per-frame ray budget `B` and foliage sub-budget `β·B`.
**Design Document:** `docs/superpowers/specs/2026-06-12-floraprime-audit.md`
**Tech Stack:** Rust (engine 2021 ed.), `floraprime-rs` (nalgebra, rayon, serde), Spectra Slang megakernel (Vulkan compute on 780M floor / RTX 4070 Ti target), wgpu engine side.
**Build:** `cargo build` / `cargo test` / FloraPrime is a path-dep crate (`floraprime-rs`), built from `~/src/floraprime/rust`.

---

## IMPORTANT NOTES

- **Do NOT add a renderer to FloraPrime.** It generates geometry + spectra; Spectra renders. Every "render" in this plan happens in the Spectra megakernel, which already has the `MAT_VEGETATION` path — do not re-implement leaf shading.
- **The SDF-only override is hard for GEOMETRY.** Wood/trunk/branches/buildings/terrain = SDF. The ONLY justified non-SDF primitive is the **leaf card** (a thin translucent surface an SDF cannot represent). Do not "SDF the canopy" near; do not turn leaves into Gaussian *geometry* (rejected by the user) — they are alpha-tested vegetation discs the tracer intersects.
- **Exact FloraPrime signatures (verified — do not invent):**
  - `build_branching_qsm(cfg: &BranchingConfig) -> TreeQSM`
  - `qsm_to_mesh(qsm: &TreeQSM, max_order: u32) -> TreeMesh` (watertight; `SdfBaker` input)
  - `qsm_to_foliage_splats(qsm, total_leaf_count, leaf_w, leaf_h, light_dir, tree_seed, leaf_geometry: i32, foliage_cluster_ratio, petiole_length, leaf_droop_range, leaf_twist_max) -> Vec<GaussianSplatCpu>`
  - `covariance_to_quat_scale(cov: &[[f32;3];3]) -> ([f32;4], [f32;3])`
  - `prospect::compute(p: &LeafParams) -> SpectralRt { reflectance: [f32;8], transmittance: [f32;8] }`
  - `SeasonalProspect::interpolate(season: f32) -> LeafParams`
  - `LeafGeometry::shader_code(&self) -> i32` ∈ {2,3,4,5,6} == Spectra `LEAF_{BROADLEAF,NEEDLE,FROND,SCALE,STRAP}`
  - `GaussianSplatCpu { position:[f32;3], covariance:[[f32;3];3], opacity:f32, leaf_type:i32, spectral_embedding:[f32;6] }` — fields are `pub`; the **engine-side** `LeafCard`/`CookedVegetationAsset` MUST use private fields + accessors (cross-crate rule).
- **Exact engine/Spectra signatures (verified — do not invent):**
  - `SdfBaker`: sealed watertight `Mesh` → snorm8 `SdfField` (~183 KB); `SdfAtlasGpu::{insert, texture, header_buf}` (SDF Pillar).
  - Spectra `sdf_eval.slang::sdf_capsule(p, a, b, radius)` — a QSM cylinder is `a = position`, `b = position + direction*length`.
  - Spectra `splat_hit.slang::{vegetation_opacity_test_typed, vegetation_albedo_typed, leaf_transmittance, eval_vegetation_bsdf_typed}`; `material_types.slang::MAT_VEGETATION = 20`.
  - `megakernel.slang`: leaf_type packed into `velocity.y` (2291) and `mat.sss_radius` (2370); double-sided veg transparency (2164); `traverse_gaussian_bvh` (1794).
  - `ScatterField::encode(encoder, camera, clearance, wind_t, budget, base_slot, splat_buf, transform_buf) -> ScatterStats` (Scatter Field §6).
  - `FrameBudgetGovernor::{new, budget, update}` (`atom_instances.rs:984`).
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.
- **Every milestone names a tier + adapter when it quotes ms** (780M floor = seconds OK; 4070 Ti target = `B`-bounded). Never quote a number without its tier (the realtime-design discipline).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_data/src/flora_cook.rs` | wrap `floraprime-rs`: cook one species → `CookedVegetationAsset` (SDF wood + leaf cards + PROSPECT appearance), deterministic |
| Create | `crates/vox_data/src/flora_asset.rs` | `CookedVegetationAsset`, `LeafCard`, `SeasonalAppearance` (private fields + accessors) |
| Modify | `crates/vox_data/src/lib.rs` | export `flora_cook`, `flora_asset`; add `floraprime-rs` path dep |
| Modify | `crates/vox_data/Cargo.toml` | `floraprime-rs = { path = "../../../floraprime/rust/floraprime-rs" }` |
| Create | `crates/vox_render/src/bin/flora_probe.rs` | the truth harness: cook → trace → print realism + cost + determinism blocks, write PNG |
| Modify | `~/src/spectra/slang/megakernel.slang` | add the far-canopy volumetric-shell march tier (card→shell collapse) — owned by Spectra side |
| Modify | `crates/vox_render/src/gpu/scatter_field.rs` | tree/grass scatter-class emits the `LeafCard` (`MAT_VEGETATION`) primitive, not a Gaussian blade |
| Modify | `~/Ochroma/projects/urban_horizon/src/bin/game_asset_cook.rs` | `cook_flora` calls `vox_data::flora_cook`, not `forge.generate_vegetation` |
| Test   | `crates/vox_data/tests/flora_cook_test.rs` | trunk-SDF coverage, leaf-card mapping, PROSPECT spectral, cook determinism |
| Test   | `crates/vox_render/tests/flora_render_test.rs` | silhouette-gap, translucency, far-shell collapse, alpha-test-ray budget |

> **NOTE on the parallel perf agent:** the Spectra megakernel edits (far-shell tier) and the `g_splat_buffer` leaf-card wiring may collide with the perf agent currently editing the megakernel. Sequence the Spectra-side tasks (3) AFTER the Universal Renderer M0 lands its `MAT_VEGETATION`/`g_splat_buffer` bind, and coordinate on `megakernel.slang` ownership. The `vox_data`/`vox_render` engine-side tasks (1, 2, 5) are collision-free and can start immediately.

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Trunk → gap-free SDF wood | `assert!(trunk_coverage >= 0.97 && snorm8_bytes < 256_000)`, both printed | `assert!(sdf.is_some())` |
| Foliage = lacy alpha-tested cards | `assert!(silhouette_gap_fraction >= 0.25)` (sky-gaps in crown bbox), printed | `assert!(canopy_pixels > 0)` |
| Real leaf translucency | `assert!(backlit_transmittance_mean >= 0.10)` (green-through-leaf), printed | `assert!(color != black)` |
| PROSPECT spectral survives | `assert!(green560 > red680 && redEdge740 > red680)` vs `prospect::compute`, printed | `assert!(spectral[0] > 0.0)` |
| Far canopy → volumetric shell | `assert!(cards_600m == 0 && shell_600m == 1 && cards_20m > cards_120m)`, printed | `assert!(lod(600)!=lod(20))` |
| Foliage cost ≤ budget | `assert!(alpha_test_rays <= beta * B)`, `d_v` auto-set, both printed | timing one frame, no count |
| Grass = strap-leaf card | `assert!(blade.mat==MAT_VEGETATION && blade.leaf_type==6)`, printed | `assert!(blades>0)` |
| Cook is deterministic | `assert_eq!(sha256(asset_a), sha256(asset_b))` (seed=42 twice), printed | `assert!(!cards.is_empty())` |
| forge gone from product cook | `grep -L "forge.*generate_vegetation" game_asset_cook.rs` in the product path; cook output identical species set via FloraPrime | cook still calls forge |

---

## Task 1 (M0): Trunk/branches → gap-free SDF wood — FloraPrime QSM through the SdfBaker

**Files:**
- Create: `crates/vox_data/src/flora_asset.rs`, `crates/vox_data/src/flora_cook.rs`
- Modify: `crates/vox_data/src/lib.rs`, `crates/vox_data/Cargo.toml`
- Test: `crates/vox_data/tests/flora_cook_test.rs`

**Acceptance:** `cargo test -p vox_data flora_trunk_sdf_gapfree -- --nocapture` → prints `trunk_coverage=0.99x snorm8_bytes=<<256000 nodes=<N≥200>` with `trunk_coverage >= 0.97` and `snorm8_bytes < 256_000`. A solid sphere-traced trunk, not a confetti point cloud.

**Wiring requirement:** `cook_vegetation` must be called from `flora_probe.rs` (Task setup) and the test; it must run `build_branching_qsm → qsm_to_mesh → SdfBaker` end to end. `todo!()`/empty bodies = task failure.

- [ ] **Step 1: Write the failing test** — bake a `red_oak` QSM and assert real gap-free coverage.

```rust
#[test]
fn flora_trunk_sdf_gapfree() {
    let species = load_species("red_oak");           // floraprime-rs TOML
    let asset = cook_vegetation(&species, 42, 0.25).expect("cook");  // summer
    let cov = sphere_trace_coverage(asset.trunk_sdf(), camera_at_m(8.0));
    println!("trunk_coverage={cov:.3} snorm8_bytes={} nodes={}",
             asset.trunk_sdf_bytes(), asset.qsm_node_count());
    assert!(cov >= 0.97, "trunk must be gap-free SDF, got {cov:.3}");
    assert!(asset.trunk_sdf_bytes() < 256_000, "snorm8 too big");
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test -p vox_data flora_trunk_sdf_gapfree 2>&1 | tail -5` → FAIL (`cannot find function cook_vegetation`).

- [ ] **Step 3: Implement** — `cook_vegetation`: build QSM from the species `BranchingConfig` mid-range bounds + seed; `qsm_to_mesh(&qsm, max_order=3)` (drops sub-3 twigs to foliage); feed the watertight mesh to `SdfBaker`; store the snorm8 `SdfField` in `CookedVegetationAsset`. `LeafCard`/`CookedVegetationAsset` use private fields + accessors (cross-crate rule). No CUDA, no Catmull-Clark on this path.

- [ ] **Step 4: Wire at exact callsite** — `flora_probe.rs` setup calls `cook_vegetation(&species, seed, season)` and uploads `asset.trunk_sdf()` to the Spectra SDF instance reader (Universal Renderer §4.5). The test calls it directly.

- [ ] **Step 5: Run — verify non-trivial output** — `cargo test -p vox_data flora_trunk_sdf_gapfree -- --nocapture` → PASS, `trunk_coverage=0.99x` (NOT 0.0), `snorm8_bytes` a real ~183k.

- [ ] **Step 6: Commit** — `git commit -m "feat(vox_data): FloraPrime QSM trunk → gap-free SDF wood via SdfBaker"`.

---

## Task 2 (M1): Foliage → PROSPECT leaf cards — backlit oak with lacy silhouette + real translucency

**Files:**
- Modify: `crates/vox_data/src/flora_cook.rs`, `crates/vox_data/src/flora_asset.rs`
- Create: `crates/vox_render/src/bin/flora_probe.rs`
- Test: `crates/vox_render/tests/flora_render_test.rs`

**Acceptance:** `cargo run --release --bin flora_probe -- --species red_oak --season summer --camera backlit` → prints `silhouette_gap=<g≥0.25> translucency=<t≥0.10> spectral: green560>red680 redEdge740>red680 PASS` and writes `flora_red_oak_backlit.png` showing a backlit oak with sky visible between leaves and green glow through the canopy. (780M floor: seconds/frame OK, printed.)

**Wiring requirement:** `cook_vegetation` must additionally produce `LeafCard`s via `qsm_to_foliage_splats` + `covariance_to_quat_scale` and a `SeasonalAppearance` via `prospect::compute(SeasonalProspect::interpolate(season))`; `flora_probe` must upload them to the Spectra `MAT_VEGETATION`/`g_splat_buffer` path and trace. `todo!()` = failure.

- [ ] **Step 1: Write the failing test** — trace a backlit cooked oak; assert lacy gaps + transmission + spectral shape.

```rust
#[test]
fn flora_backlit_oak_is_lacy_and_translucent() {
    let asset = cook_vegetation(&load_species("red_oak"), 42, 0.25).unwrap();
    let img = trace_spectra(&asset, Camera::backlit(), spp(64));
    let gap = sky_gap_fraction(&img, asset.crown_bbox());     // sky pixels inside crown bbox
    let trans = backlit_green_transmittance(&img);             // mean G through canopy
    let spd = integrate_canopy_spd(&img);
    println!("silhouette_gap={gap:.3} translucency={trans:.3} \
              green560={:.3} red680={:.3} redEdge740={:.3}",
             spd.band(560), spd.band(680), spd.band(740));
    assert!(gap >= 0.25, "canopy must be lacy, got gap={gap:.3} (SDF blob would be ~0)");
    assert!(trans >= 0.10, "leaves must transmit, got {trans:.3} (opaque SDF would be ~0)");
    assert!(spd.band(560) > spd.band(680) && spd.band(740) > spd.band(680),
            "PROSPECT green peak + red-edge must survive to the rendered SPD");
}
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`trace_spectra`/`cook_vegetation` foliage path missing).

- [ ] **Step 3: Implement** — extend `cook_vegetation`: `qsm_to_foliage_splats(&qsm, leaf_count_from_bounds, leaf_w, leaf_h, light, seed, LeafGeometry::Broadleaf.shader_code(), cluster_ratio, petiole, droop, twist)`; map each `GaussianSplatCpu` → `LeafCard` via `covariance_to_quat_scale`; widen the 6-float embedding to the engine spectral from the full 8-band `prospect::compute`. `SeasonalAppearance` = per-season `SpectralRt`. `flora_probe` uploads leaf cards to `g_splat_buffer` with `leaf_type` packed per the megakernel path; trace with the existing `MAT_VEGETATION` shader (no shader change this task).

- [ ] **Step 4: Wire at exact callsite** — `flora_probe::main` cooks, uploads trunk SDF + leaf cards, calls the Spectra trace, computes the three metrics, writes the PNG. The test calls the same `trace_spectra` helper.

- [ ] **Step 5: Run — verify non-trivial output** — the run prints `... PASS` and a human opens `flora_red_oak_backlit.png` to confirm lacy backlit foliage (not a green blob, not black leaves).

- [ ] **Step 6: Commit** — `git commit -m "feat(vox_data,vox_render): FloraPrime foliage → MAT_VEGETATION leaf cards, backlit oak renders lacy + translucent"`.

---

## Task 3 (M2): Far-canopy volumetric shell + distance-LOD collapse + the β·B budget governor

**Files:**
- Modify: `~/src/spectra/slang/megakernel.slang` (new far-canopy volumetric march tier) — **coordinate with the perf agent**
- Modify: `crates/vox_render/src/bin/flora_probe.rs`
- Test: `crates/vox_render/tests/flora_render_test.rs`

**Acceptance:** `cargo run --release --bin flora_probe -- --species red_oak --grove 50 --gate-rays $B --tier 4070ti` → prints `cards@20m=<a> cards@120m=<b<a> cards@600m=0 shell@600m=1 | alpha_test_rays=<r ≤ β·B> card→shell d_v=<d>m | frame=<ms> @ tier=4070ti adapter=<name>` with `alpha_test_rays <= β·B` and `d_v` auto-derived from the governor. Far trees are green volumetric shells; near trees are lacy cards.

**Wiring requirement:** the card→shell transition `d_v` must be driven by `FrameBudgetGovernor` (the foliage sub-budget `β·B`), not a constant; the far tier must march a volumetric medium, not sphere-trace an SDF blob. `todo!()` = failure.

- [ ] **Step 1: Write the failing test** — classify one tree at 20/120/600 m and assert the collapse + budget bound.

```rust
#[test]
fn flora_far_canopy_collapses_and_respects_budget() {
    let asset = cook_vegetation(&load_species("red_oak"), 42, 0.25).unwrap();
    let b = perf_budget_rays();                 // B from the realtime tier
    let beta = 0.35_f32;                         // foliage sub-budget fraction
    let r20  = classify(&asset, dist_m(20.0));
    let r120 = classify(&asset, dist_m(120.0));
    let r600 = classify(&asset, dist_m(600.0));
    let frame = trace_grove(&asset, 50, b, beta);
    println!("cards@20m={} cards@120m={} cards@600m={} shell@600m={} \
              alpha_test_rays={} d_v={:.0}m",
             r20.cards, r120.cards, r600.cards, r600.shells,
             frame.alpha_test_rays, frame.d_v_m);
    assert!(r20.cards > r120.cards && r600.cards == 0 && r600.shells == 1);
    assert!(frame.alpha_test_rays <= (beta * b as f32) as u64,
            "foliage must stay under β·B, got {}", frame.alpha_test_rays);
}
```

- [ ] **Step 2: Run to verify it fails** — FAIL (no far-shell tier; no governor-driven `d_v`).

- [ ] **Step 3: Implement** — Spectra: a volumetric-canopy march tier (green absorbing/scattering medium parameterized by the crown bbox + a coarse density from the leaf-card count, Cook-stochastic-aggregate albedo) entered when an instance's eye distance > `d_v`. Engine: `FrameBudgetGovernor` sets `d_v` so the integrated leaf-card `L̄` over the visible grove stays ≤ `β·B` (the §4.3 affordance). Near/mid keep cards; far is one shell march per far instance.

- [ ] **Step 4: Wire at exact callsite** — `flora_probe` grove loop drives the governor each frame; `d_v` feeds the per-instance tier select before upload; the megakernel branches card-BVH vs shell-march on the instance tier flag.

- [ ] **Step 5: Run — verify non-trivial output** — `cards@600m=0 shell@600m=1`, `alpha_test_rays ≤ β·B`, `d_v` a real metres value, `frame` ms with its tier+adapter named.

- [ ] **Step 6: Commit** — `git commit -m "feat(spectra,vox_render): far-canopy volumetric shell + β·B governor — foliage cost bounded, not population-bounded"`.

---

## Task 4 (M3): Scatter reconciliation — tree & grass scatter-classes emit the leaf-card primitive

**Files:**
- Modify: `crates/vox_render/src/gpu/scatter_field.rs`
- Test: `crates/vox_render/tests/flora_render_test.rs`

**Acceptance:** `cargo test -p vox_render flora_grass_is_strap_leaf_cards -- --nocapture` → prints `blades=<N> mat=MAT_VEGETATION leaf_type=6 (LEAF_STRAP) | far_tile=volumetric_shell` with every generated grass primitive a strap-leaf card (not a Gaussian blade), and a tree scatter-class instancing a FloraPrime asset (trunk SDF + leaf cards). Confirms the Scatter Field machine unchanged, primitive switched.

**Wiring requirement:** `ScatterField::encode` (or its generation pass) must emit `LeafCard` (`MAT_VEGETATION`, `leaf_type`) primitives for grass/tree classes, replacing the superseded "Gaussian blade"; far-tile collapse must use the volumetric shell from Task 3. `todo!()` = failure.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn flora_grass_is_strap_leaf_cards() {
    let cls = ScatterClass::grass_strap(/*density*/ 16.0, /*h*/ [0.12, 0.30]);
    let tile = scatter_generate(cls, tile(17, 42), budget(10_000));
    println!("blades={} mat={:?} leaf_type={} far_tile={:?}",
             tile.count(), tile.first().mat(), tile.first().leaf_type(), tile.far_repr());
    assert!(tile.count() > 0 && tile.count() <= 10_000);
    assert!(tile.iter().all(|p| p.mat() == MAT_VEGETATION && p.leaf_type() == 6),
            "grass must be strap-leaf cards, not Gaussian blades");
    assert!(matches!(tile.far_repr(), FarRepr::VolumetricShell));
}
```

- [ ] **Step 2: Run to verify it fails** — FAIL (scatter still emits Gaussian blades / `ScatterClass::grass_strap` missing).

- [ ] **Step 3: Implement** — `ScatterClass::grass_strap` and the tree prototype class produce `LeafCard`s: grass = `LEAF_STRAP` cards with the per-blade hash placement/lean/phase the Scatter Field already computes; trees = instance a FloraPrime `CookedVegetationAsset` (trunk SDF instance + leaf-card set). Mask/density/determinism/wind reused verbatim from the Scatter Field design — only the emitted primitive changes.

- [ ] **Step 4: Wire at exact callsite** — `ScatterField::encode` generation pass writes the `LeafCard` layout into `g_splat_buffer` (the layout the megakernel `MAT_VEGETATION` path reads), far tiles tagged for the volumetric shell.

- [ ] **Step 5: Run — verify non-trivial output** — `mat=MAT_VEGETATION leaf_type=6`, count within budget, `far_tile=VolumetricShell`.

- [ ] **Step 6: Commit** — `git commit -m "feat(vox_render): Scatter Field emits MAT_VEGETATION leaf cards (grass=strap), Gaussian-blade primitive superseded"`.

---

## Task 5 (M4): Replace forge in the product cook — deterministic FloraPrime cook, forge demoted to preview

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/bin/game_asset_cook.rs`
- Test: `crates/vox_data/tests/flora_cook_test.rs`

**Acceptance:** `cargo run --release --bin game_asset_cook -- --flora-only` → prints `flora: oak_tree birch_tree shrub tall_grass via floraprime (sdf+cards) | sha256 deterministic` and the product cook path no longer calls `forge.generate_vegetation`. `cargo test -p vox_data flora_cook_deterministic -- --nocapture` → `sha256(asset_a) == sha256(asset_b)` for seed=42 cooked twice.

**Wiring requirement:** `cook_flora` in `game_asset_cook.rs` must call `vox_data::flora_cook::cook_vegetation`, producing `FloraAsset`s backed by FloraPrime (SDF wood + leaf cards + PROSPECT), not the forge atomization. forge stays in-code as a preview-only path. `todo!()` = failure.

- [ ] **Step 1: Write the failing test** — cook determinism gate.

```rust
#[test]
fn flora_cook_deterministic() {
    let s = load_species("red_oak");
    let a = cook_vegetation(&s, 42, 0.25).unwrap();
    let b = cook_vegetation(&s, 42, 0.25).unwrap();
    let (ha, hb) = (a.sha256(), b.sha256());
    println!("sha256_a={ha} sha256_b={hb}");
    assert_eq!(ha, hb, "FloraPrime cook must be byte-deterministic (replay-safe)");
}
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`cook_vegetation`/`sha256` not yet on the asset, or cook still forge-backed).

- [ ] **Step 3: Implement** — replace the `forge.generate_vegetation → atomize_vegetation → veg.*.atoms.json` body of `cook_flora` with `cook_vegetation` per recipe, writing the `CookedVegetationAsset` (SDF + cards + appearance). Keep the `FloraAsset` registration (still NOT a `BuildingAsset`). Leave a `--preview-forge` flag that routes to the old forge path for fast placeholders. `CookedVegetationAsset::sha256` hashes the canonical serialized bytes.

- [ ] **Step 4: Wire at exact callsite** — inside `cook_flora`, swap the producer; the `flora_recipes()` list (oak_tree, birch_tree, shrub, tall_grass) now maps to FloraPrime species + seeds.

- [ ] **Step 5: Run — verify non-trivial output** — the cook prints the FloraPrime-backed species line; the determinism test prints two identical sha256s.

- [ ] **Step 6: Commit** — `git commit -m "feat(civitas): cook_flora uses FloraPrime (SDF wood + PROSPECT leaf cards), forge demoted to preview"`.

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks (each Task has a Wiring requirement + Step 4 callsite).
- [x] Every `Acceptance` names a real non-trivial output (coverage ≥ 0.97, gap ≥ 0.25, transmittance ≥ 0.10, spectral inequality, `alpha_test_rays ≤ β·B`, identical sha256) — never "tests pass", never zeroes.
- [x] Every `Wiring requirement` names an exact function/file (`cook_vegetation`, `cook_flora`, `ScatterField::encode`, `flora_probe::main`).
- [x] `IMPORTANT NOTES` carries the real FloraPrime + engine + Spectra signatures from the audit's §6.
- [x] `File Map` lists every file any task touches.
- [x] No step contains `todo!()`/`unimplemented!()`/stub bodies in implementation code.
- [x] `Done When` names a specific command (`flora_probe ... --gate-rays $B`) and a specific human-observable result (lacy backlit oak PNG + printed realism/cost/determinism blocks).
- [x] Types/signatures consistent across tasks (`cook_vegetation`, `CookedVegetationAsset`, `LeafCard`, `MAT_VEGETATION`, `leaf_type` used identically).
- [x] Budget-parameterized throughout: costs stated as `B` / `β·B`, tier + adapter named whenever ms is quoted; the perf agent's verdict sets `B`.
- [x] Honest about 780M-floor (seconds/frame OK, correctness first) vs 4070-Ti-target (`B`-bounded ms).
