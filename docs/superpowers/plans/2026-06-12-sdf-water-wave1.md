# SDF Water — Wave 1 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **EXECUTION GATE (coordinate with the perf agent):** A Fable 5 perf agent is editing `~/src/spectra/slang/megakernel.slang` RIGHT NOW. Before Task 1: `git -C ~/src/spectra log --oneline -3 && git -C ~/src/spectra status --porcelain` — start only from a CLEAN tree on the perf agent's landed result, and re-verify the M0–M2 anchor points cited below (`sdf_render_intersect`, the glass branch, the `u_sdf_render_enabled` gate) still exist where stated; if the perf agent moved them, the water block goes beside their NEW location — the mechanism is unchanged. The perf agent's verdict also sets the budget **B** that Task 4's report parameterizes on. All water code is additive and gated by `u_water_enabled = 0` default, so every existing render stays byte-identical. Run in the main repos, never an isolated worktree (the abs-path gotcha memory).

**Goal:** v1 static-body SDF water in the Spectra path tracer on the 780M floor: analytic mask-clipped water planes, the M2 dielectric mechanism (Fresnel/Snell continuation + Beer-Lambert), Gerstner wave normals, shore-distance foam/wet-edge, hard budget instrumentation, and the game-map unification still — every milestone with a measured coverage/quality metric AND a measured ray cost.
**Done When:** `cd ~/src/ochroma && cargo test --release -p vox_render --features spectra-native sdf_water -- --nocapture` exits green printing `[sdf-water] reflection-evidence px: <E> (gate > 800)`, `[sdf-water] shallow bottom visible: <V> (gate > 0.50)`, `[sdf-water] deep mean transmittance: <T> (gate < 0.05)`, `[sdf-water] deep B/R: <D> vs shallow B/R: <S> (gate deep > 2x shallow)`, `[sdf-water] foam px: <F>, max |shore_dist| = <M> m (gate: inside the 6.0 m band, count > 200)`, `[sdf-water] waves: t=0.0 vs t=1.5 differ on <W> water px (gate > 0.3 * water_px)`, `[sdf-water] rays: continuations=<C> reflect=<R> refract=<X> skipped_deep=<K>` (K > 0), `[sdf-water] cost: water frame <a> s vs dry frame <b> s = <m>x (gate <= 2.5x)`, and writes `sdf_water_calm.png` / `sdf_water_choppy.png` / `sdf_water_opaque_control.png` where a human sees the red-roofed building MIRRORED in the water, the bottom through the shallows, and foam at the shore — none of which exist in the control. AND `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play -- --shot-water water_shots` exits 0 printing `[water] mask unification: render==sim on 65536/65536 cells` and writes `water_shots/water_tidewater.png`.
**Architecture:** Water is a new analytic primitive in the megakernel's SDF block (NOT a baked volume, NOT a march): per-plane `t = (h_k − ro.y)/rd.y` + one mask texel, competing on nearest-t with `sdf_render_intersect`. The hit runs the PROVEN M2 glass continuation (`sample_glass`, IOR 1.333) with Beer-Lambert depth absorption from a cooked depth field, Gerstner normal perturbation (no geometry), and a deep-water refraction skip. Three 2D fields (height/shore-distance/depth, ≈0.8 MB) carry every shore effect for zero extra rays. Counters in `g_water_stats` make the ray cost observable every run.
**Design Document:** `docs/superpowers/specs/2026-06-12-sdf-water-design.md`
**Tech Stack:** Rust (edition 2021/2024 per crate), Slang (slangc via the spectra kernel dir — the `spectra-pathtracer-runs` kernel-dir fix), spectra repo `~/src/spectra`, engine `~/src/ochroma` (`vox_render`, feature `spectra-native`), game repo `~/Ochroma/projects/urban_horizon` (Task 5 only). AMD 780M / RADV Vulkan floor.
**Build:** engine: `cd ~/src/ochroma && cargo build --release -p vox_render --features spectra-native`; spectra kernels compile at run (ensure_compiled — loud on failure); game: `cd ~/Ochroma/projects/urban_horizon && cargo build --release`.

---

## IMPORTANT NOTES

- **Repos / commits:** Tasks 1–4 land in `~/src/spectra` (slang + rust crates) and `~/src/ochroma` (`crates/vox_render`) — paired commits per task (the M0–M2 precedent: spectra `1480e2c` + ochroma `160469b`). Task 5 lands in `~/Ochroma/projects/urban_horizon`. Every commit ends with the house footer `Co-Authored-By:` line.
- **Verified signatures (code-checked 2026-06-12 — code wins over docs; re-verify line numbers after the perf agent lands):**
  - `BSDFSample sample_glass(float3 wo, float3 n, float ior, float roughness, float3 absorption_color, float absorption_depth, float distance, float u1, float u2, float u3)` — `~/src/spectra/slang/brdf_glass.slang:227`. Handles TIR. **Reuse it; do NOT write a second water BSDF.**
  - `float3 beer_lambert(float3 absorption_color, float absorption_depth, float distance)` = `exp(-(absorption_color/absorption_depth)·distance)` — `brdf_glass.slang:77`. For σ in 1/m pass `absorption_color = σ`, `absorption_depth = 1.0`.
  - The M2 glass continuation pattern to mirror EXACTLY (one stochastic next ray, throughput `*= f·|n·wi|/pdf`, transmitted-side tint, `RAY_BIAS_EPSILON` origin offset on the correct side, `g_next_ray_*`/`g_next_throughput_*`/`g_next_ray_alive` writes, AOV writes gated `u_bounce == 0`) — `megakernel.slang:1266–1338`. The SDF block runs on EVERY bounce (:1205–1207) so refracted rays hit the bottom naturally.
  - `float sdf_render_intersect(float3 ro, float3 rd, float t_max, out float3 hit_normal, out int hit_inst)` — `megakernel.slang:687` (`MAX_STEPS = 384`). The water test must pass `min(t_water - eps, t_max)`-style bounds so nearest-primitive-wins is exact.
  - Gating precedent: `u_sdf_render_enabled` / `u_num_sdf_instances` (`megakernel.slang:1208`) — water adds `u_water_enabled` (0 default), same discipline.
  - `pathtrace_sdf_scene_with_atoms_to_rgba(volumes: &[SdfVolumeInput], instances: &[SdfSceneInstance], atoms_per_instance: &[Vec<SdfSceneAtom>], eye: [f32;3], target: [f32;3], fov_y: f32, width: u32, height: u32, spp: u32, rig: &LightRig) -> Result<Vec<u8>, String>` — `crates/vox_render/src/splat_backend.rs:982`; it sets `config.use_nrc = false` and forces `config.max_bounces >= 3` (:1204–1206). **The with-water entry keeps both** (water needs the ≥3 bounce floor: camera→water→scene→light).
  - `SdfVolumeInput { resolution: [u32;3], origin: [f32;3], voxel_size: f32, narrow_band: f32, distances: Vec<f32> }` (:454); `SdfSceneInstance { volume_index, position, rotation_xyzw, uniform_scale, albedo }` (:660); `SdfSceneAtom { position, color, channel }` (:677); `LightRig { sun_dir, sun_color, sun_intensity, sky_intensity, camera_fill }` (:192).
  - `SdfLayer::from_parts_with_atoms(volumes: Vec<SdfVolumeHeader>, instances: Vec<SdfInstanceHeader>, distances: Vec<f32>, albedo: Vec<f32>, atom_positions: Vec<f32>, atom_colors: Vec<f32>, atom_channels: Vec<f32>, instance_atom_range: Vec<f32>) -> Self` — `~/src/spectra/rust/spectra-scene-state/src/layers.rs:271`. The new `WaterLayer` sits beside `SdfLayer`, same style.
  - GAME (Task 5): `MapTerrain` fields private; use `is_water(wx, wz)`, `sample_y`, `heights()`, `version()` (`src/map/terrain.rs`); `WaterMask { pub bodies: Vec<WaterBody>, pub sea_level: f32 }`, `WaterBody { name, kind, surface_height, outline }` (`src/map/layers.rs`); bundled maps are 256² @ 4 m, `tidewater_flats` has real water (`src/map/bundled.rs`); harness precedent `--shot-landvalue` (`play.rs` — prints measured gates, exits Err on failure).
- **New-type rule:** all new cross-crate types have **private fields + accessors** (`WaterSurfaceInput`, `WaterMaterial`, `WaterFrameReport` — see design §5; constructors validate grid lens == w·h and planes ≤ 4).
- **Water material constants (design §4.3):** IOR **1.333**; default σ = **(0.45, 0.12, 0.06)** 1/m; roughness `lerp(0.02, 0.18, choppiness)`; foam 4 m; wet edge 2 m; deep cutoff ε = 0.05 ⇒ `d_cut = ln(20)/min(σ)`; deep color (0.012, 0.035, 0.05). Wave amplitudes capped ≤ 0.3 m (normal-only bump honesty).
- **Budget discipline:** every harness/test prints the `[sdf-water] rays:` counter line and the cost multiplier — a task whose output lacks the ray accounting is NOT done. `u_water_roulette_p` stays 1.0 in wave 1 (floor tier); the uniform + throughput division ship anyway so the realtime tier is a knob, not a refactor.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.
- Tests live in in-module `#[cfg(test)]` blocks in `splat_backend.rs` (the M0–M2 house style); GPU-needing tests print a skip line and stay green on a GPU-less box.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `~/src/spectra/slang/water_surface.slang` | analytic plane intersect, mask/shore/depth field reads, Gerstner `wave_normal(xz, t)`, foam/wet helpers |
| Modify | `~/src/spectra/slang/megakernel.slang` | `g_water_*` bindings + uniforms + `g_water_stats`; the water branch in the SDF block (nearest-t vs SDF; dielectric continuation; deep skip; foam/wet shading) |
| Modify | `~/src/spectra/rust/spectra-scene-state/src/layers.rs` | `WaterLayer::from_parts(...)` beside `SdfLayer` |
| Modify | `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` | pack/upload the water grids + uniforms (the `pack_sdf_volume_headers` precedent) |
| Modify | `~/src/spectra/rust/spectra-renderer/src/renderer.rs` | bind `g_water_*` + read back `g_water_stats` |
| Modify | `~/src/ochroma/crates/vox_render/src/splat_backend.rs` | `WaterPlane`/`WaterSurfaceInput`/`WaterMaterial`/`WaveOctave`/`WaterFrameReport`; `heightfield_to_sdf_volume`; `cook_fluid_shore_fields`; `pathtrace_sdf_scene_with_water_to_rgba`; all wave-1 tests |
| Create | `~/Ochroma/projects/urban_horizon/src/map/water_fields.rs` | `build_water_surface_input(terrain, water, material)` — the unification seam |
| Modify | `~/Ochroma/projects/urban_horizon/src/map/mod.rs` | `pub mod water_fields;` |
| Modify | `~/Ochroma/projects/urban_horizon/src/bin/play.rs` | `--shot-water` harness |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Analytic water plane, dielectric hit | shore scene dielectric vs opaque-control render: `reflection_evidence_px > 800` (building-hue water px absent in control), `shallow bottom visible > 0.5`, both printed | `assert!(water_px > 0)` |
| Nearest-primitive correctness | a building standing IN the water: pixels where the building is nearer show facade, not water (`facade_in_water_region_px > 500`), printed | rendering water alone |
| Waves animate, zero re-upload | t=0.0 vs t=1.5 frames differ on > 0.3·water_px; `scene_uploads == 1`; choppy luma variance > calm | comparing image lengths |
| Beer-Lambert + deep skip + foam | `deep_T < 0.05`, `deep B/R > 2× shallow B/R`, `skipped_deep > 0`, foam px > 200 all with `|shore_dist| < 6 m` — every number printed | asserting deep darker |
| Budget observability | `WaterFrameReport` printed every run: continuations/reflect/refract/skipped/cost-multiplier ≤ 2.5×, plus the tier table for r ∈ {1, 1.5, 2, 3} computed from MEASURED counters | no counters |
| Game unification | tidewater_flats: `render==sim on 65536/65536 cells`; PNG with the game's shore mirrored | sampling 3 cells |
| Off = byte-identical | M0/M1/M2 tests green unmodified with `u_water_enabled` defaulted 0 | skipping regression |

---

## Task 1: The water-plane dielectric primitive (megakernel branch + WaterLayer + uploader + the with-water host entry) — calm mirror water over a real bottom

**Files:**
- Create: `~/src/spectra/slang/water_surface.slang`
- Modify: `~/src/spectra/slang/megakernel.slang`, `~/src/spectra/rust/spectra-scene-state/src/layers.rs`, `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs`, `~/src/spectra/rust/spectra-renderer/src/renderer.rs`
- Modify: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (types + `heightfield_to_sdf_volume` + the entry + test)

**Acceptance:** `cargo test --release -p vox_render --features spectra-native sdf_water_plane_reflects_and_transmits -- --nocapture` → prints `[sdf-water] water px: <n> (f_w=<f>)`, `[sdf-water] reflection-evidence px: <E>` with E > 800, `[sdf-water] shallow bottom visible: <V>` with V > 0.5, `[sdf-water] facade-in-water px: <P>` with P > 500, and writes `sdf_water_calm.png` + `sdf_water_opaque_control.png`. M0/M1/M2 tests (`sdf_building_renders_solid_surface`, `sdf_city_block`, `sdf_craftsman_atom_material_and_glass`) green UNMODIFIED.

**Wiring requirement:** the water branch must live INSIDE the megakernel's gated SDF block (beside `megakernel.slang:1208–1338`), run every bounce, and compete on nearest-t with `sdf_render_intersect` and the triangle `g_hit_t`; `pathtrace_sdf_scene_with_water_to_rgba` in `splat_backend.rs` must build the `WaterLayer`, upload via the uploader, and keep `use_nrc = false` + `max_bounces >= 3`. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — in `splat_backend.rs` `#[cfg(test)]` (the M2 test's skeleton: skip-print without a GPU):

```rust
#[test]
fn sdf_water_plane_reflects_and_transmits() {
    // Bottom: a procedural island heightfield (REAL data — a radial island,
    // h = 8*exp(-r²/1800) - 3, 128x128 @ 2 m), so there is a genuine shore.
    let (heights, w, d, cs) = island_heightfield_128();
    let bottom = heightfield_to_sdf_volume(&heights, w, d, cs, 4.0, 1.0).expect("bottom sdf");
    // Two cooked buildings at the shore — one RED-roofed (the reflection probe),
    // loaded exactly like sdf_craftsman_atom_material_and_glass does.
    let (volumes, instances, atoms) = shore_scene_with_red_roof(bottom);
    let water = WaterSurfaceInput::new(
        vec![WaterPlane::new(0.0)], w as u32, d as u32, cs, [0.0, 0.0],
        water_height_from(&heights, 0.0),           // wet iff h < 0.0
        vec![0.0; w * d], depth_from(&heights, 0.0), // Task 3 cooks real shore fields; depth real now
        WaterMaterial::calm_default(),
    ).expect("water input");
    let (img, report) = pathtrace_sdf_scene_with_water_to_rgba(
        &volumes, &instances, &atoms, &water,
        [60.0, 18.0, 90.0], [0.0, 0.0, 0.0], 0.9, 320, 320, 32, &LightRig::default_day(),
    ).expect("water render");
    let (ctrl, _) = pathtrace_sdf_scene_with_water_to_rgba( // opaque control: dielectric off
        &volumes, &instances, &atoms, &water.with_opaque_control(),
        [60.0, 18.0, 90.0], [0.0, 0.0, 0.0], 0.9, 320, 320, 32, &LightRig::default_day(),
    ).expect("control render");
    save_png("sdf_water_calm.png", &img, 320, 320);
    save_png("sdf_water_opaque_control.png", &ctrl, 320, 320);

    let wm = water_pixel_mask(&water, /*camera*/ ...); // analytic: which px hit water
    let evidence = red_roof_hue_px(&img, &wm) - red_roof_hue_px(&ctrl, &wm);
    let shallow_visible = bottom_hue_fraction(&img, &wm, /*shallow band*/ ...);
    println!("[sdf-water] water px: {} (f_w={:.2})", report.water_px(), report.water_px() as f32 / (320.0*320.0));
    println!("[sdf-water] reflection-evidence px: {evidence} (gate > 800)");
    println!("[sdf-water] shallow bottom visible: {shallow_visible:.2} (gate > 0.50)");
    assert!(evidence > 800, "the water must MIRROR the red roof, got {evidence}");
    assert!(shallow_visible > 0.5, "the shallows must show the bottom, got {shallow_visible}");
    let in_water_facade = facade_px_inside_water_region(&img, &wm);
    println!("[sdf-water] facade-in-water px: {in_water_facade} (gate > 500)");
    assert!(in_water_facade > 500, "nearest-primitive: a building IN water occludes it");
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --release -p vox_render --features spectra-native sdf_water_plane 2>&1 | tail -5` → FAIL: `pathtrace_sdf_scene_with_water_to_rgba` / `WaterSurfaceInput` not found.
- [ ] **Step 3: Implement.** (a) `water_surface.slang`: `water_plane_intersect(ro, rd, t_max, out t, out cell)` — per plane `t_k = (h_k − ro.y)/rd.y`, accept nearest with `g_water_cell_height[cell] == h_k` (±1e-3) and `rd.y` guarded; `water_grid_cell(xz)` from `u_water_grid`. (b) megakernel: bindings/uniforms per design §5; in the SDF block compute `t_water` BEFORE shading the SDF hit; if `0 < t_water < min(t_sdf_or_miss, sdf_t_max)` → the water branch: geometric normal +Y (waves come in Task 2), `sample_glass(wo, n, u_water_ior, u_water_roughness, float3(0), 1.0, 0.0, u1, u2, u3)` with fresh `mlt_sample_1d` dims (the :1275–1277 pattern, distinct dim offsets), continuation written EXACTLY like :1301–1330 (transmitted side gets NO tint yet — absorption is Task 3), AOVs at `u_bounce == 0`, `g_water_stats` increments via `InterlockedAdd`. Opaque-control mode: `u_water_opaque_control != 0` → shade the water hit as Lambert `(0.05, 0.10, 0.22)` (today's splat tint — the honest "before"). (c) scene-state `WaterLayer::from_parts` + uploader packing + renderer binding (mirror the SdfLayer/atom-buffer flow end to end). (d) host: the §5/§6 types with private fields + accessors; `heightfield_to_sdf_volume` (sample `d = (p.y − h(x,z))·inv_lipschitz`, `inv_lipschitz = 1/√(1+max|∇h|²)`, clamp to narrow band); `pathtrace_sdf_scene_with_water_to_rgba` mirroring the :982 entry + `WaterLayer` + stats readback into `WaterFrameReport` (dry-frame timing comes in Task 4 — until then `dry_frame_seconds = 0`, NOT printed as a gate).
- [ ] **Step 4: Wire at exact callsites** — the water branch inside the `u_sdf_render_enabled` block of `shade_and_bounce` (beside :1208); `WaterLayer` consumed where `SdfLayer` is consumed in `uploader.rs`; the entry calls `ensure_compiled` so a kernel error is loud (the M0 fix precedent).
- [ ] **Step 5: Run — verify non-trivial output** — the acceptance command prints E > 800, V > 0.5, P > 500 with real numbers; M0/M1/M2 + `vulkan_backend_renders_quad` green unmodified; eyeball the two PNGs (mirror image present / absent).
- [ ] **Step 6: Commit** (both repos)

```bash
cd ~/src/spectra && git add slang/water_surface.slang slang/megakernel.slang rust/spectra-scene-state/src/layers.rs rust/spectra-scene-upload/src/uploader.rs rust/spectra-renderer/src/renderer.rs
git commit -m "feat(megakernel): analytic water-plane dielectric — mask-clipped, Fresnel/Snell continuation (W0)"
cd ~/src/ochroma && git add crates/vox_render/src/splat_backend.rs
git commit -m "feat(spectra-native): pathtrace_sdf_scene_with_water_to_rgba — water as an SDF primitive over a heightfield bottom (W0)"
```

---

## Task 2: Gerstner wave normals + sea-state roughness (animated, zero re-upload)

**Files:**
- Modify: `~/src/spectra/slang/water_surface.slang`, `~/src/spectra/slang/megakernel.slang` (wave uniforms only)
- Modify: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (`WaveOctave` plumb-through + test)

**Acceptance:** `cargo test --release -p vox_render --features spectra-native sdf_water_waves_animate_and_roughen -- --nocapture` → prints `[sdf-water] waves: t=0.0 vs t=1.5 differ on <W> water px (gate > 0.3 * water_px)`, `[sdf-water] scene uploads: 1 (geometry never re-uploaded)`, and `[sdf-water] choppy luma variance <c> > calm <a>` with c > 1.5·a — all real measured numbers.

**Wiring requirement:** `wave_normal(float2 xz, float t)` in `water_surface.slang` called from the megakernel water branch (replacing the Task-1 flat +Y normal); `u_water_time` and the four `WaveOctave`s flow from `WaterMaterial` through the uploader. Per-hit ALU stays bounded: exactly 4 octaves, no loops over buffers. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — render the Task-1 scene at `time = 0.0` and `time = 1.5` (a `with_time(f32)` builder on `WaterSurfaceInput`) with choppiness 0.7, count differing water pixels (|ΔRGB| > 8/255); render calm (choppiness 0.0) vs choppy at t=0 and compare water-pixel luma variance; assert the three gates; print all numbers.
- [ ] **Step 2: Run to verify failure** — `2>&1 | tail -5` → FAIL: `with_time` not found / frames identical (W = 0).
- [ ] **Step 3: Implement.** `wave_normal`: sum 4 Gerstner octaves — per octave `phase = dot(dir, xz)·(2π/λ) + t·speed·(2π/λ)`, accumulate `∂h/∂x, ∂h/∂z` analytically, normal = `normalize(float3(-∂x, 1, -∂z))`; octave table from uniforms (defaults: λ = 11/5.3/2.7/1.3 m, A = choppiness·(0.18/0.09/0.04/0.02) m, directions spread ±40°, speeds from the deep-water dispersion `√(gλ/2π)`). Roughness `lerp(0.02, 0.18, choppiness)` into the `sample_glass` call. Blend the wave normal toward flat within the last 2 m of `shore_dist` (no choppy foam-line artifacts; field exists from Task 1's depth/height inputs).
- [ ] **Step 4: Wire at exact callsite** — the megakernel water branch calls `wave_normal(hit.xz, u_water_time)`; the host entry uploads time/octaves per render with NO scene re-upload (assert the upload counter).
- [ ] **Step 5: Run — verify non-trivial output** — gates pass with printed W, variance pair, uploads=1; write `sdf_water_choppy.png`; eyeball: broken-up reflection vs Task 1's mirror.
- [ ] **Step 6: Commit** (both repos, same shape as Task 1, message `feat(megakernel): Gerstner wave normals + sea-state roughness on the water dielectric (W1)`).

---

## Task 3: Beer-Lambert depth + shore fields (foam, wet edge, deep refraction skip)

**Files:**
- Modify: `~/src/spectra/slang/water_surface.slang`, `~/src/spectra/slang/megakernel.slang` (absorption + skip + foam/wet in the water and opaque-SDF branches)
- Modify: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (`cook_fluid_shore_fields` + test)

**Acceptance:** `cargo test --release -p vox_render --features spectra-native sdf_water_depth_absorption_and_foam -- --nocapture` → prints `[sdf-water] deep mean transmittance: <T> (gate < 0.05)`, `[sdf-water] deep B/R: <D> vs shallow B/R: <S> (gate deep > 2x shallow)`, `[sdf-water] foam px: <F>, max |shore_dist| = <M> m (gate: inside the 6.0 m band, count > 200)`, `[sdf-water] wet-edge px: <E> (darker terrain inside 2 m of shore, gate > 100)`, `[sdf-water] rays: ... skipped_deep=<K>` with K > 0.

**Wiring requirement:** `cook_fluid_shore_fields(water_cell_height, terrain_heights, w, h, cell_size)` (two-pass chamfer; returns `(shore_distance, water_depth)`) must be called inside the test's scene build AND be the function Task 5's game cook calls — one cook, two callers. The megakernel water branch must (a) on refract: `throughput *= beer_lambert(u_water_absorption, 1.0, depth/max(|wi.y|,0.2))`, (b) skip the refract branch when `depth > u_water_deep_cut` (add `(1-F)`-weighted `u_water_deep_color` to the film, count `skipped_deep`), (c) foam-shade where `shore_dist < u_water_foam_width` modulated by the Task-2 wave noise; the opaque SDF/bottom branch must wet-darken where `-u_water_wet_width < shore_dist < 0`. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — deepen the island scene (min depth ~−40 m so a genuinely deep basin exists); cook real shore fields; render; measure: mean water-pixel RGB in the deep region vs the bottom-albedo-normalized shallow region → transmittance proxy + B/R ratios; count foam px (luma > 0.75 inside water) and their max `|shore_dist|` via the cooked field; count wet-edge px (terrain px inside 2 m, darker than the same terrain rendered with `u_water_wet_width = 0` — render the A/B pair); assert all gates; print everything.
- [ ] **Step 2: Run to verify failure** — FAIL: `cook_fluid_shore_fields` not found / deep water identical to shallow.
- [ ] **Step 3: Implement** the cook (chamfer 3-4-distance two-pass over 65k cells, signed by wet/dry, metres = chamfer·cell_size/3; depth = `surface_y − terrain_h` clamped ≥ 0), the three megakernel effects, and the σ plumb-through. Sun term at underwater bottom hits additionally `*= exp(−σ·depth_at_hit)` (the vertical column — design §4.3).
- [ ] **Step 4: Wire at exact callsites** — cook called from the test scene build; fields uploaded in the Task-1 layer; effects in the water branch + the opaque-SDF shade path (wet edge).
- [ ] **Step 5: Run — verify non-trivial output** — all five printed gates pass; eyeball `sdf_water_calm.png` (re-written): blue-green deepening, white shore band, dark wet rim.
- [ ] **Step 6: Commit** (both repos, `feat(megakernel): Beer-Lambert water depth + shore-distance foam/wet-edge + deep refraction skip (W2)`).

---

## Task 4: The water-ray budget report — counters, cost multiplier, and the perf-tier affordance table

**Files:**
- Modify: `~/src/ochroma/crates/vox_render/src/splat_backend.rs` (dry-frame A/B timing in the with-water entry, `WaterFrameReport` completion, the tier table, the `sdf_water` umbrella test)
- Modify: `~/src/spectra/rust/spectra-renderer/src/renderer.rs` (stats readback if not finished in Task 1), `~/src/spectra/slang/megakernel.slang` (`u_water_roulette_p` + throughput division — shipped at p=1.0)

**Acceptance:** `cargo test --release -p vox_render --features spectra-native sdf_water_budget_report -- --nocapture` → prints `[sdf-water] rays: continuations=<C> reflect=<R> refract=<X> skipped_deep=<K> rouletted=0` with C == R+X+K consistency asserted (`assert_eq!(report.continuations(), report.reflections()+report.refractions(), ...)` modulo the skip accounting — exact identity printed), `[sdf-water] cost: water frame <a> s vs dry frame <b> s = <m>x (gate <= 2.5x)`, and the tier table:

```
[sdf-water] budget ladder (measured k_w=<k>, f_w=<f>):
  r=3.0  -> full water        (needs <n3> traversals/frame, fits)
  r=2.0  -> reduced-TLAS refl (needs <n2>)
  r=1.5  -> roulette p=<p>    (clamped to (r-1)*P*S)
  r=1.0  -> analytic fallback (0 extra traversals — headline OFF, renegotiate)
```

computed from the MEASURED counters (k_w = (R·1.0 + X·0.3)/C etc.), not hardcoded.

**Wiring requirement:** the dry frame is the SAME scene rendered with `u_water_enabled = 0` inside `pathtrace_sdf_scene_with_water_to_rgba` (one entry, both timings — callers cannot forget the control); `u_water_roulette_p` is applied in the megakernel (continuation spawn gated by a roulette draw, throughput `/= p`) and pinned to 1.0 by wave-1 hosts. The umbrella `sdf_water` test name must match every Task 1–4 test prefix so the Done-When command runs them all. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — call the entry on the Task-3 scene, assert counter consistency, `cost_multiplier() <= 2.5`, `skipped_deep > 0`, `rouletted == 0` at p=1.0, and that the ladder lines print with non-zero traversal estimates.
- [ ] **Step 2: Run to verify failure** — FAIL: `dry_frame_seconds == 0` / no ladder print.
- [ ] **Step 3: Implement** — the A/B timing, the ladder computation from counters (`k_w`, `f_w`, `P`, `S` all measured), roulette uniform + kernel division.
- [ ] **Step 4: Wire** — report printing inside the entry (always-on, the observability rule); roulette in the water branch continuation gate.
- [ ] **Step 5: Run — verify non-trivial output** — full `cargo test --release -p vox_render --features spectra-native sdf_water -- --nocapture` = the plan's Done-When engine block, all gates green, three PNGs written.
- [ ] **Step 6: Commit** (both repos, `feat(spectra-native): water-ray budget instrumentation — counters, dry-frame cost multiplier, perf-tier affordance ladder (W2)`).

---

## Task 5: The game's water, unified — `--shot-water` on tidewater_flats (GAME repo)

**Files:**
- Create: `~/Ochroma/projects/urban_horizon/src/map/water_fields.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/map/mod.rs`, `src/bin/play.rs`

**Acceptance:** `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play -- --shot-water water_shots` exits 0 printing `[water] tidewater_flats: planes=<N> water cells=<W> shore cells=<S>` (W > 1000), `[water] mask unification: render==sim on 65536/65536 cells (CellClass::Water is THE clip field)`, the `[sdf-water] rays:`/cost lines from the engine report, and writes `water_shots/water_tidewater.png` — the game map's water with shore buildings mirrored. `cargo test --release water_fields -- --nocapture` prints `unification: 65536/65536` headlessly (cook-only, no GPU needed; the render gate lives in the binary harness which skip-prints without an adapter).

**Wiring requirement:** `build_water_surface_input(terrain: &MapTerrain, water: &WaterMask, material: WaterMaterial) -> Result<WaterSurfaceInput, String>` in `src/map/water_fields.rs` — wet iff `terrain.is_water(centre)` (NEVER re-derive from heights; the mask is authoritative), per-cell height = the controlling body's `surface_height` else `sea_level`, fields via the ENGINE's `cook_fluid_shore_fields`; `shot_water(out_dir)` parsed beside `--shot-landvalue` in `play.rs::main`, building 2–3 shore-adjacent cooked-building SDF instances + a `heightfield_to_sdf_volume` bottom from `terrain.heights()` and calling `pathtrace_sdf_scene_with_water_to_rgba`. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — in `water_fields.rs` `#[cfg(test)]`: load `tidewater_flats` via `Map::load_dir`/bundled, build the input, assert per-cell agreement count == 65536 with both counts printed, `wet_cell_count() > 1000`, and every wet cell's height equals its body's `surface_height` (or `sea_level`) exactly.
- [ ] **Step 2: Run to verify failure** — FAIL: module not found.
- [ ] **Step 3: Implement** the cook + the harness (the `--shot-landvalue` skeleton: real prints, `Err(String)` on gate failure, `save_rgba`).
- [ ] **Step 4: Wire** — `pub mod water_fields;`; the `--shot-water` arm in `main`; camera aimed at a probed wet-shore site (scan the mask — no magic coordinates).
- [ ] **Step 5: Run — verify the full Done-When** — both Done-When commands green end to end; eyeball the PNG (the headline shot: the game's own water reflecting the game's own buildings — the thing CS2 fakes in screen space).
- [ ] **Step 6: Commit** (game repo, `feat(water): path-traced map water unified with CellClass::Water — --shot-water harness + unification gate`).

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — the megakernel branch, layer, uploader, host entry and test land together in Task 1; no "wire later"
- [x] Every `Acceptance` names real non-trivial output (evidence px, transmittance, B/R ratios, foam band metres, counter identities, 65536-cell agreement) — never "tests pass"
- [x] Every `Wiring requirement` names exact functions/files (the SDF block beside `megakernel.slang:1208–1338`, `uploader.rs` packing, `play.rs::main` arg parse, `cook_fluid_shore_fields` one-cook-two-callers)
- [x] `IMPORTANT NOTES` carries code-checked signatures (`sample_glass` :227, `beer_lambert` :77, `sdf_render_intersect` :687, the :982 entry + its NRC/bounce forcing, `from_parts_with_atoms` :271, game accessors) and the water constants
- [x] `File Map` lists every file appearing in any task
- [x] No step contains `todo!()` / stub bodies
- [x] `Done When` names exact commands and exact human-visible prints + PNGs in both repos
- [x] Names consistent across tasks: `WaterSurfaceInput`, `WaterMaterial`, `WaterFrameReport`, `pathtrace_sdf_scene_with_water_to_rgba`, `heightfield_to_sdf_volume`, `cook_fluid_shore_fields`, `wave_normal`, `build_water_surface_input`, `u_water_enabled`
- [x] Budget discipline enforced structurally: counters print in every run; the dry control lives inside the one entry; the roulette knob ships at p=1.0 so the perf tier is a parameter, not a refactor
- [x] Execution gate stated: clean spectra tree AFTER the perf agent lands; anchors re-verified; everything `u_water_enabled`-gated so existing renders are byte-identical
- [x] v1.5 (terraform flood/drain re-cook) and v2 (fluid.rs/pbf.rs surfacing, FFT, caustics) explicitly NOT in wave 1 — seams named in the design §4.7–4.8
