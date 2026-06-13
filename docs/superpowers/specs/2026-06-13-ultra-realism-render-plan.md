# Design: Ultra-Realism Render Plan — Buildings + Perfect Rendering (2026-06-13)

**Status:** Draft
**Scope:** One ranked, sequenced implementation plan to take Ochroma building renders from "object-on-turntable, blotchy-gray, flat-prism" to SOTA arch-viz photorealism — synthesizing five dimension surveys (geometry, materials, lighting, pathtracer, refgap) and reconciling them with the existing 6-wave facade-photorealism roadmap. Affects `forge` (geometry), `civitas_care` cook (materials/weathering), `spectra` (kernel + bindings), `ochroma/vox_render` (host wiring, LightRig/LookPreset, denoiser).
**Related:** `facade-photorealism-roadmap.md` (the 6 waves — this plan EXTENDS and RE-SEQUENCES it, does not duplicate); `building-content-pipeline.md`; `hybrid-atom-sdf-lod-direction.md`; `spectra-texture-bridge.md`; `aaa-phase2-resident-frame.md`.

---

## 1. Problem Statement

Grounded in the six current renders (`/tmp/block_hiq.png`, `/tmp/block_lowq.png`, `/tmp/look_{aces,acesbright,filmic,flat}.png`) and code:

- **Blotchy gray "static" on every solid wall** — a complete 7-channel per-vertex weathering engine (`weathering_masks.slang:8,51`, `megakernel.slang:179,2713`) is bound NOWHERE in `spectra-renderer/src/renderer.rs`; the cook fills `forge_weathering: None` (`game_asset_cook.rs:1770`). Zero tonal break-up.
- **Dead-flat facades, no shadow lines** — punched-window `build_wall` (`forge .../wall.rs:48-128`) emits a flat plane; spandrel/string-course/parapet machinery exists only in `curtain_wall.rs:147-213`. Reveals are depth-capped at 6 cm (`wall.rs:370-405`) by an over-cautious reading of the SDF winding oracle.
- **Visible 2.5 m brick-tile stamp** — UVs are baked at a single global `UV_TILE_METRES = 2.5` (`forge .../lib.rs:36,362`); `PbrMaterial.uv_scale` is always `[1,1]` (`splat_backend.rs:177`). Textures are downsampled to 256/128 px to dodge a mip-less EWA sampler (`textures.rs:189-194`), so distance facades dissolve.
- **House-door on a 12-storey tower** — `wall.rs:111-115` always emits one residential `emit_door_cell`; no storefront/lobby.
- **Buildings float on a void plane under flat frontal daylight** — no grounding/context; the physical atmosphere model (`atmosphere.slang`) and HDRI path (`megakernel.slang:1825`) are unbound; lit window panes glow but are NOT NEE-registered (`u_num_emissive_tris` never uploaded) so interiors don't light the street.
- **Realtime PT has no denoise/temporal/adaptive plumbing** — SVGF/temporal/adaptive kernels (`realtime_denoise.slang`, `temporal_reproject.slang`, `adaptive.slang`) are written but unregistered in `kernel_set.rs`; the only active denoiser on AMD is a detail-destroying CPU 8-bit bilateral (`denoiser.rs:20`).

---

## 2. Done When

Running the per-task render gate below produces the named, measured human-visible output. The headline acceptance for the whole plan:

Running `cargo run --features spectra-native --bin forge_pathtrace -- --look dusk --time 18.5 --weathering on --atmosphere on --out /tmp/hero_block.png` produces a path-traced city block where (a) wall surfaces show visible vertical soot streaks + sill water-stains (wall-crop luminance variance ≥1.5× the weathering-off render), (b) each floor line casts a distinct horizontal shadow band (≥ floors-1 dark bands in a vertical luminance profile), (c) brick reads ≥6 courses/metre (not one 2.5 m stamp), (d) lit window panes cast a measurable warm pool on adjacent wall (ΔL ≥ 0.03 linear vs emission-off), and (e) the building sits on a sidewalk with contact-AO darkening at its base (bottom-5% wall crop <0.85× mid-wall luminance). A human at the keyboard sees a believable dusk city block, not a flat-gray prism on a void.

---

## 3. Capabilities

Ordered strictly by **visual-leverage-per-effort with dependencies honored**. Each row = one shippable item; the leftmost number is the global rank. Tracks: **[S]tills-now** items improve the offline beauty path immediately; **[R]ealtime** items only matter for 60fps@4K. Items 1-12 serve BOTH tracks (a better material/geometry/light is seen in both); items marked [R] diverge into the realtime-only pipeline.

| # | Capability | Real behavior test (the render-gate assertion) | Stub test (forbidden) |
|---|---|---|---|
| 1 | **Wire dormant weathering** (refgap T1 / materials M2 / Wave 3-partial) | `assert!(var(wall_crop, weathering_on) >= 1.5 * var(wall_crop, weathering_off))` AND `assert!(mean_lum(bottom_20pct) <= 0.92 * mean_lum(top_20pct))` (soot gradient); New-condition cook byte-identical to today | `assert!(masks.len() > 0)` — passes with zeros fed |
| 2 | **Bind sun direction + physical atmosphere** (lighting L1 / Wave 6) | `assert!((sky_top_R_on - sky_top_R_off).abs() > 0.04)` — two renders differ at horizon proving the uniform reaches the GPU (today byte-identical, uniform unbound) | render runs, no exception |
| 3 | **Per-material UV scale** (materials M1 / kills 2.5 m stamp) | `assert!(brick_row_crossings_per_m >= 6.0)` (was ~0.4) on a vertical pixel column over the spandrel | `assert!(mat.uv_scale != [1,1])` — passes without re-tiling |
| 4 | **Mip chains in EWA sampler** (pathtracer A2 / materials M0 / Wave 6) | `assert!(distant_wall_row_variance < baseline)` (alias→stable) AND `assert!(mid_distance_detail_variance > baseline_256px)` | sampler returns a texel |
| 5 | **Coping parapet + multi-step cornice** (geometry T1+T2 / Wave 3) | `assert!(silhouette_width(top + coping_h*0.5) >= wall_width + 2*coping_proj)` AND cornice outward tris at `>=3` distinct Z for 3-step style | cornice mesh non-empty |
| 6 | **Spandrel bands + string courses on punched wall** (geometry T4 / refgap T2 / Wave 3) | `assert!(dark_horizontal_bands(wall_column_crop) >= floors - 1)` AND `assert_watertight_body(mesh)` | `assert!(trim_tris > 0)` |
| 7 | **Deepen window reveals (decouple from frame depth)** (geometry T4 / materials) | `assert!(deepest_reveal_z >= 0.15)` AND `assert_watertight_body` AND cook reports closed SDF (no Shell fallback) | reveal depth field set, geometry unchanged |
| 8 | **Storefront base + glazed lobby** (geometry T3 / refgap T3 / Wave 2) | `assert!(door_tris == 0)` on a 10-storey tower AND `assert!(glass_tris / band_tris > 0.6)` in ground band | `assert!(storefront.is_some())` |
| 9 | **Register emissive tris as NEE lights** (lighting L4 / Wave 6) | `assert!(wall_lum_below_pane(emit_on) - wall_lum_below_pane(emit_off) >= 0.03)` (linear) | pane glows when hit (already true) |
| 10 | **Sun disk + HDRI environment upload** (lighting L2+L3 / Wave 6) | brightest pixel cluster within 3 px of projected sun dir AND >5× surrounding sky lum; chrome sphere reflects HDRI within 4 px | background non-gradient |
| 11 | **Grounding + context plate (GAME layer)** (refgap T5) | `assert!(lum(bottom_5pct_wall) / lum(mid_wall) < 0.85)` (contact AO) | ground plane exists |
| 12 | **Dusk/GoldenHour LookPreset + TimeOfDay helper** (lighting L5 / refgap T6 / Wave 6) | `assert!(shadow_pixel_extent(dusk) >= 2.0 * shadow_pixel_extent(noon))` AND dusk mean-hue toward orange | preset enum compiles |
| 13 | **Per-volume material breaks (podium ≠ shaft)** (geometry T8 / materials / Wave 3) | `assert!(podium_band_material_id != tower_band_material_id)` AND sampled mean-color delta > 3% | two materials assigned |
| 14 | **Projecting bays / oriels** (geometry T5 / Wave 4) | `assert!(bayed_column_max_outward_z >= 0.4)` AND watertight AND shadow gradient adjacent to bay | bay mesh non-empty |
| 15 | **Balconies (slab + rail)** (geometry T6 / residential) | `assert!(balcony_slab_outward_z >= 1.2)` AND watertight AND slab casts shadow on floor below | slab tris > 0 |
| 16 | **Rooftop mechanical / penthouse** (geometry T7 / crown content) | `assert!(roof_region_pixel_height_variance > threshold)` (top-down) — plant breaks flat plane | `MAT_MECH` tris > 0 |
| 17 | **Block composition: multi-volume frontage** (Wave 5) | `assert!(distinct_volume_heights >= 2 && distinct_facade_families >= 2)` on a wide lot | massing produces N volumes |
| 18 | **Straight-skeleton pitched roofs over L/U/T** (geometry T9) | `assert!(void_cov < 0.02)` (L-notch empty) AND roof-over-arm normals non-horizontal | L/U/T mesh non-empty |
| 19 | **AO + height/POM maps in reveals** (materials M3) | horizontal scan across brick shows AO valleys ≥15% lum dip at mortar lines (absent before) | AO field non-None |
| 20 | **Per-instance hue/dirt variation** (materials M4 / 100K) | `assert!(pairwise_mean_color_delta >= 0.03)` for ≥8 of 10 instances (today pixel-identical) | per-instance buffer bound |
| 21 | **[R] GPU spectral denoiser replaces CPU bilateral** (pathtracer A1) | `assert!(sobel_edge_sharpness_gpu >= 1.4 * cpu)` on the mullion crop | denoiser dispatches |
| 22 | **[R] Realtime SVGF (spatial+temporal) at 1spp** (pathtracer B1) | `assert!(mean_abs(frame7 - frame6) < 2.0/255)` AND `assert!(flat_wall_noise_1spp_denoised <= raw_1spp/3)` | denoise kernel registered |
| 23 | **[R] Variance-adaptive sampling on** (pathtracer B2) | `assert!(samples_adaptive <= 0.6 * samples_fixed && psnr_delta.abs() < 0.5)` | `u_adaptive_mode==1` |
| 24 | **[R] Resident zero-readback PT frame loop** (pathtracer B3) | `assert!(readback_bytes_per_frame == 0)`; prints `mean_frame_ms` | render_resident returns |
| 25 | **[R] HW-RT TLAS proven + scaled to 100K** (pathtracer C1) | `assert!(trace_ms_100k / trace_ms_1k < 10.0)` (sub-linear) | TLAS builds |
| 26 | **[R] 4K/60 composite (FSR + SVGF + adaptive + resident)** (pathtracer C2) | `assert!(p99_frame_ms < 16.6)` on projected-4070Ti (=p50/6); 780M number printed as floor | renders at 4K |

---

## 4. Architecture

### 4.1 Sequencing rationale (why this order)

The dominant principle is **wiring-before-authoring**: items 1, 2, 9, 21-23 light up engine code that already exists and is dead, so their leverage-per-effort is maximal. The geometry-depth items (5-7, 13-18) require new Forge generators but each reuses proven closed-solid emitters (`emit_wall_box`, `emit_trim_slab`, `push_box`) so the SDF winding oracle stays green. Dependencies that reorder vs. the surveys:

- **Item 4 (mip chains) is a hard prerequisite for items 3, 19, 20 being VISIBLE** — without it, higher-res/re-tiled textures alias worse. But item 3 (UV scale) is itself cheap and visible even at 256 px, so 3 precedes 4 (3 is S-effort wiring, 4 is M-effort cross-repo). They are independent in code; both land before any 2k-tier work.
- **Item 1 (weathering) before item 6 (spandrels)**: weathering is the single highest-leverage fix (kills the blotchy-gray defect with three call-site bindings) and is independent of geometry.
- **Item 2 (atmosphere/sun bind) before item 12 (dusk preset)**: the preset needs a sun that actually appears in the sky.
- **Items 9 + 10 (emissive NEE + sun disk/HDRI)** complete the lighting spine; 10 depends on 2's uniform-binding pattern.
- **Realtime track (21-26) is fully separable** and should run as a parallel workstream; only item 21 (GPU denoiser) also benefits stills. The realtime track does NOT block any stills item.

### 4.2 Reconciliation with the 6-wave roadmap

| Roadmap wave | This plan's items | Change |
|---|---|---|
| Wave 1 (composed glazing + glow) | DONE (`wall.rs:200-224`) — not re-listed | confirmed complete |
| Wave 2 (storefront/lobby) | Item 8 | re-ranked LATER than weathering/lighting (depth-of-field gives more payoff first) |
| Wave 3 (facade depth + per-volume + weathering) | Items 1, 5, 6, 7, 13 | SPLIT: weathering (1) pulled to rank #1; depth geometry (5,6,7) mid; per-volume (13) after lighting |
| Wave 4 (bays/oriels) | Item 14 | unchanged position |
| Wave 5 (block composition) | Item 17 | unchanged |
| Wave 6 (beauty rig: dusk/HDR/glow/denoise) | Items 2, 9, 10, 12, 21 | EXPANDED and pulled FORWARD — lighting spine (2,9,10) ranks above bays/blocks because it lifts every render |

The net reorder vs. the strict wave sequence: **weathering and the lighting spine jump ahead of mid/late geometry** because they are dead-code-wiring (low effort) with whole-image payoff (high leverage), whereas bays/balconies/roof-plant are new-geometry (higher effort) with localized payoff.

### 4.3 Engine/game split discipline

Sun/sky/IBL/look knobs live on `LightRig`/`LookPreset` in `vox_render` (rendering concepts — engine-legal). Weathering masks, UV-scale tables, material assignment, and the context/grounding plate (item 11) live in `civitas_care` cook / showcase scene builders (game layer). Forge generators stay game-agnostic (a "storefront" is a facade primitive, not a "business zone"). No building/zoning/traffic concept enters `vox_*`, `spectra-*`, or `forge`.

### 4.4 Closed-solid invariant (cross-cutting)

Every new geometry primitive (parapet, cornice steps, spandrel pocket, deepened reveal, storefront, bay, balcony, roof plant) MUST be a closed solid so the cook's generalized-winding-number gate stays green. Verified per-item by `assert_watertight_body` (forge `lib.rs:1441`) and by `game_asset_cook` reporting closed SDF (not Shell fallback) for the full catalog. All new `description.rs` fields are additive with defaults that preserve existing directives byte-for-byte (`box_massing_regression_is_byte_identical_to_pre_massing` stays green).

---

## 5. Data Models

```rust
// vox_render/src/splat_backend.rs — LightRig gains the lighting-spine knobs (engine-legal)
pub struct LightRig {
    // ...existing sun/sky-dome fields...
    atmosphere_enabled: bool,      // item 2 — drives the Bruneton model
    sun_radiance: [f32; 3],        // item 2/10 — sun disk + atmosphere coupling
    weathering_enabled: bool,      // item 1 — default true
    env: EnvLight,                 // item 10 — Gradient | Hdri(TextureImage)
    extra_lights: Vec<AreaLight>,  // item (lighting L7, deferred) rect/point
}
impl LightRig {
    pub fn for_time_of_day(hour: f32) -> LightRig; // item 12 — sun elev/azimuth/temp
    pub fn weathering_enabled(&self) -> bool { self.weathering_enabled }
}

pub enum LookPreset { AcesFilm, AcesBright, Filmic, SoftReview, Flat, Dusk, Night } // +Dusk/Night, item 12

// civitas_care/src/asset/mod.rs — ReadyAssetPbrMaterial gains per-material tiling
pub struct ReadyAssetPbrMaterial {
    // ...existing...
    uv_scale: [f32; 2],            // item 3 — default [1,1], skip-if-default in serde
    // ambient_occlusion: stop hardcoding None (item 19)
}

// forge .../description.rs — additive, default-preserving fields
pub struct BuildingDescription {
    // ...existing...
    ground_floor: GroundFloor,     // item 8 — Punched(default) | Storefront | Lobby
    bay_columns: BayPolicy,        // item 14 — default Off
    balcony_policy: BalconyPolicy, // item 15 — default Off
}

// forge .../facade/extrude.rs — new material ids (renderer table extended to match)
// MAT_MECH 8 (item 16), MAT_WALL_ACCENT 9, MAT_WALL_BASE 10 (item 13)
```

---

## 6. API

```rust
// spectra-renderer/src/renderer.rs — the missing bindings (mirror set_texture_atlas pattern,
// MUST be called AFTER load_scene_state, like the atlas at :1361-1366)
pub fn set_weathering_masks(&mut self, masks: &[f32]);   // item 1 — uploads g_weathering_masks, sets u_weathering_enabled=1
pub fn set_sun(&mut self, dir: [f32;3], radiance: [f32;3]); // item 2 — binds u_sun_direction/u_sun_radiance
pub fn set_atmosphere(&mut self, enabled: bool, mie: f32, rayleigh: f32); // item 2
pub fn set_hdri(&mut self, data: &[f32], w: u32, h: u32, channels: u32);  // item 10 — uploads g_hdri_data

// forge .../facade/cornice.rs
pub fn generate_parapet(start: Vec3, end: Vec3, base_y: f32, outward: Vec3,
                        upstand_h: f32, coping_proj: f32, coping_h: f32) -> Mesh; // item 5 — two stacked closed boxes
pub fn generate_cornice(edges: &[Edge], profile: &[(f32, f32)]) -> Mesh;          // item 5 — (proj,height) per step

// forge .../facade/storefront.rs (new)
pub fn generate_storefront(edge: Edge, floor_h: f32, outward: Vec3, style: Style) -> Mesh; // item 8

// forge .../facade/{bay,balcony}.rs (new)
pub fn generate_bay(origin: Vec3, tangent: Vec3, outward: Vec3, width: f32,
                    sill_y: f32, height: f32, projection: f32, style: Style) -> Mesh; // item 14
pub fn generate_balcony(origin: Vec3, tangent: Vec3, outward: Vec3, width: f32,
                        y: f32, depth: f32, style: Style) -> Mesh;                    // item 15

// game_asset_cook.rs (civitas) — the producers
fn weathering_masks_for(mesh: &Mesh, cond: ForgeWeathering) -> Vec<f32>;  // item 1 — 7 floats/vertex
fn uv_scale_for_channel(channel: MaterialChannel) -> [f32; 2];           // item 3 — per-channel table
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `set_weathering_masks` | after `load_scene_state` in lit-mesh render | `vox_render/src/splat_backend.rs` ~`:519` (alongside `set_sky_gradient`) | item 1 |
| `weathering_masks_for` | payload assembly | `civitas .../game_asset_cook.rs:1770` (replace `forge_weathering: None`) | item 1 |
| `set_sun` / `set_atmosphere` | after `set_sky_gradient` | `vox_render/src/splat_backend.rs` ~`:519` | item 2 |
| `uv_scale` propagation | `PbrMaterial` build | `civitas .../forge_pathtrace.rs:941` (drop `..Default` for uv_scale) | item 3 |
| mip pyramid build | atlas upload | `vox_render/src/splat_backend.rs:2208` `build_texture_atlas`; consume in `spectra/slang/texture_atlas.slang` | item 4 |
| `generate_parapet` | per-volume roof assembly, Flat roofs | `forge .../lib.rs:95-115, 275-292` | item 5 |
| spandrel/string-course emit | after cell loop in `build_wall` | `forge .../wall.rs` (reuse `emit_trim_slab:489`, `emit_wall_box:723`) | item 6 |
| reveal depth decouple | reveal emit | `forge .../wall.rs:663-718` (revise the `:370-405` cap comment) | item 7 |
| `generate_storefront` | grade-band routing | `forge .../wall.rs:105-126` (gate `emit_door_cell` to residential) | item 8 |
| emissive-tri → `g_lights` + `u_num_emissive_tris` upload | scene upload | `spectra .../gpu_scene.rs:245`; mark tris in `layers.rs` | item 9 |
| `set_hdri` / sun-disk uniforms | after `set_sun` | `vox_render/src/splat_backend.rs`; sky-miss branch `spectra .../megakernel.slang:1840` | item 10 |
| context/grounding plate | showcase scene builder (GAME) | `civitas` showcase / `vox_app` — NOT `vox_render` core | item 11 |
| `for_time_of_day` + Dusk/Night | `LookPreset::resolve` + rig build | `vox_render/src/splat_backend.rs:252,269` | item 12 |
| per-volume material tag | massing assembly | `forge .../lib.rs:259-270`, `massing.rs:24-29`; map in `civitas material.rs` | item 13 |
| `generate_bay` / `generate_balcony` | deterministic column subset | `forge .../wall.rs:88-92` (column hash) | items 14, 15 |
| `generate_rooftop_plant` | after roof on Flat volumes | `forge .../roof_plant.rs` (new) → `lib.rs` | item 16 |
| SVGF/temporal/adaptive kernels | dispatch graph | `spectra .../kernel_set.rs` (register), `renderer.rs:580-694, 1413` | items 21-23 [R] |
| `render_resident` | resident present path | `spectra .../lib.rs`; `vox_render/src/gpu/` (ResidentGiRaster present) | item 24 [R] |
| TLAS scale / LBVH fallback | `is_hw_rt_available` branch | `spectra .../renderer.rs:98,188-220`; `spectra-tlas` | item 25 [R] |

---

## 8. Open Questions

Resolved for this plan:

- **Realtime vs stills priority?** → Stills track (items 1-20) ships first; it is what the user judges each wave on. Realtime track (21-26) runs as a parallel workstream gated on the resident-frame substrate (already green per `aaa-phase2-resident-frame`).
- **Where does the 256px cap get lifted?** → Item 4 (mip pyramid) is the unlock; raise `max_size_for` to 1024 only after mips land, else aliasing regresses.
- **2k texture tier?** → Deferred (materials M5) until item 4 proves the mip select holds; not in the ranked 26 (hero-shot polish, low whole-catalog leverage).
- **Rect/portal area lights (lighting L7) and auto-exposure (L6)?** → Deferred; emissive-tri NEE (item 9) covers the dusk-interior need; fixed-EV presets suffice until a night-streetlight scene is required.

---

## 9. Out of Scope

- CUDA-only SOTA (OptiX denoise/SER, DLSS, NRC) — stubs on the 780M dev box; the realtime track deliberately routes around them (compute SVGF replaces OptiX denoise, FSR replaces DLSS, compute-LBVH replaces OptiX traversal).
- GI/indirect rebuild — the path tracer already does multi-bounce indirect; do not touch.
- Decals, parallax beyond reveal POM, and 4k texture tier — polish beyond the ranked 26.
- Any building/zoning/traffic logic in engine crates.

---

## 10. Related Plans / Designs

- Extends: `facade-photorealism-roadmap.md` (the 6 waves) — see §4.2 reconciliation.
- Depends on: `aaa-phase2-resident-frame` (resident GPU loop, for items 24-26).
- Related: `building-content-pipeline.md`, `forge-directive-factory.md`, `spectra-texture-bridge.md`, `spectra-sdf-perf-verdict.md`, `hybrid-atom-sdf-lod-direction.md`.
