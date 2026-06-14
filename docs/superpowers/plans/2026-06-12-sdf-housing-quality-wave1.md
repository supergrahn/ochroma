# SDF Housing Quality — Wave 1 (M1 textured building + M2 detail crispness) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the M2 SDF craftsman from k-NN-blended flat colour to crisp textured PBR — cell-grid atom gather (~80× cheaper), per-channel PolyHaven textures box-projected at the SDF hit, normal maps, exact aperture-rect windows with procedural muntins/bevels, and interior-mapped glass — proven by measured detail-energy/consistency/muntin/interior gates on the AMD 780M.
**Done When:** `cd ~/src/ochroma && SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --lib sdf_craftsman_textured_quality -- --nocapture --test-threads=1` prints `gather speedup=<N>x` (N ≥ 20, identical image), `facade detail energy <E>x flat-blend (gate >= 3.0x) -> PASS`, `consistency mean dev <D> (gate < 0.12) -> PASS`, `normals variance <V>x (gate >= 4.0x) -> PASS`, `muntins v=<v> h=<h> (expects v=1 h=1) -> PASS`, `interior A/B <P> px mean|ΔRGB|=<M> (gate >= 200 px, >= 0.15) -> PASS`, and writes `sdf_craftsman_textured.png` showing clapboard courses, muntin bars, and a lit room behind the glass where `sdf_craftsman_flat_control.png` (M2 path, same camera) shows flat colour. A human verifies by eyeball + the printed gates.
**Architecture:** All kernel work is additive and uniform-gated (`u_sdf_textured`, default 0 ⇒ byte-identical M2 frames) in the megakernel's SDF shade block; host work extends `splat_backend.rs` with a new textured entry point that reuses the GPU-verified texture-atlas bridge (`set_texture_atlas` AFTER `load_scene_state`). The cook gains deterministic aperture-rect derivation from glass-zone triangles. Quality features are per-hit shading costs (≈ +5% of one sphere-trace), never extra rays — see design §4.1 cost table.
**Design Document:** `docs/superpowers/specs/2026-06-12-sdf-housing-quality-design.md`
**Tech Stack:** Rust (workspace toolchain), Slang (megakernel, slangc runtime compile), wgpu-independent Vulkan via spectra `GpuBackend`, serde JSON cooked payloads.
**Build:** `scripts/build-spectra-native.sh build -p vox_render --features spectra-native` (sets SLANG_DIR=~/slang-sdk + LD_LIBRARY_PATH); cook: `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin game_asset_cook`. GPU tests need `SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` and `--test-threads=1`.

---

## IMPORTANT NOTES

- **A separate perf agent is editing `~/src/spectra/slang/megakernel.slang` concurrently.** Before EVERY task touching spectra: `cd ~/src/spectra && git pull/rebase` onto its head; all kernel additions MUST be new uniform-gated blocks (`u_sdf_textured != 0`, `u_sdf_lighting_tier > 0`) so default frames stay byte-identical. Merge conflicts are resolved by keeping the perf agent's trace changes and re-applying these shading blocks.
- Exact existing signatures (verified in code — do NOT re-derive or alter):
  - `pathtrace_sdf_scene_with_atoms_to_rgba(volumes: &[SdfVolumeInput], instances: &[SdfSceneInstance], atoms_per_instance: &[Vec<SdfSceneAtom>], eye: [f32;3], target: [f32;3], fov_y: f32, width: u32, height: u32, spp: u32, rig: &LightRig) -> Result<Vec<u8>, String>` — `crates/vox_render/src/splat_backend.rs:982`.
  - `SdfSceneAtom { position: [f32;3], color: [f32;3], channel: u32 }`; channel ids `SDF_ATOM_CH_FACADE=0 … SDF_ATOM_CH_OTHER=8`; `sdf_atom_channel_from_str(&str) -> u32`.
  - `PbrMaterial { base_color: [f32;3], roughness: f32, metallic: f32, emission_strength: f32, albedo_tex: i32, roughness_tex: i32, normal_tex: i32, uv_scale: [f32;2] }` (+`Default`); `TextureImage { width, height, channels, data: Vec<f32> }` — data is LINEAR; sRGB-decode diffuse only.
  - `Renderer::set_texture_atlas(texture_descs: &[u32], texture_data: &[f32], num_textures: u32) -> RenderResult<()>` — descs are `[offset_in_FLOATS, width, height, channels]` per texture; **silently no-ops if called before `load_scene_state`** — always call AFTER.
  - `SdfLayer::from_parts_with_atoms(volumes, instances, distances, albedo, atom_positions, atom_colors, atom_channels, instance_atom_range) -> SdfLayer` — `~/src/spectra/rust/spectra-scene-state/src/layers.rs:271`. Do not change it; add a sibling.
  - Megakernel: `sdf_gather_atom_material(int inst, float3 world_p, float radius) -> SdfAtomGather{color,channel,is_glass,nearest_d}` (line ~463); SDF shade block at line ~1208 (`u_sdf_render_enabled`); glass routes through `sample_glass(wo, n, ior, rough, float3(0), 1.0, 0.0, u1, u2, u3) -> BSDFSample`; texture fetch is `sample_texture_ewa(g_tex_descs, g_tex_data, u_num_textures, tex_id, uv, ddx_uv, ddy_uv) -> float4` (bilinear sibling exists).
  - Cook UV convention (`game_asset_cook.rs:1711` `forge_box_uv`): `p = [local.x + forge_width*0.5, local.y, forge_depth*0.5 - local.z]`, `s = 1/2.5`; dominant |normal| axis: Y ⇒ `[p.x*s, p.z*s]`, X ⇒ `[p.z*s, p.y*s]`, Z ⇒ `[p.x*s, p.y*s]`. The kernel MUST replicate this exactly (consistency gate depends on it).
  - Cooked per-channel materials: `ReadyAssetPbrMaterial { id, name, channel: ReadyAssetMaterialChannel, base_color_factor: [f32;4], metallic_factor, roughness_factor, textures: ReadyAssetTextureSet }`; texture URIs resolve via `urban_horizon::asset::textures::TextureCache::load(uri, tint)` (downsampled ≤256²/≤128², tint-normalized — use it, do not re-decode JPGs).
- All new types use **private fields + accessors ONLY where the design says so**; the `splat_backend.rs` input structs (`SdfUvParams`, `SdfApertureRect`, `SdfChannelMaterials`) are plain-pub input data like the shipped `SdfSceneAtom` — match the file's existing convention.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.
- GPU tests are `#[ignore]`-free but require the Vulkan env; follow the existing `sdf_craftsman_atom_material_and_glass` test (splat_backend.rs:2633) for asset loading, env guards, and PNG writing.
- Cooked craftsman fixtures: SDF + atoms for `forge.house.craftsman` under `~/Ochroma/projects/urban_horizon/assets/buildings/forge_starter/` (atoms: `atoms/…craftsman….atoms.json`); PolyHaven sets under `assets/buildings/forge_starter/textures/polyhaven/<set>/1k/`. The existing M2 test shows the exact loading code — reuse its helpers.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `~/src/spectra/slang/megakernel.slang` | grid gather, box-UV textured shade, TBN normal maps, aperture rects + muntins/bevel, interior mapping (all `u_sdf_textured`-gated) |
| Modify | `~/src/spectra/rust/spectra-scene-state/src/layers.rs` | `SdfLayer` sibling ctor `from_parts_textured` + new fields (grid, uv params, channel materials, apertures) |
| Modify | `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` | upload + bind the new buffers in `bind_to_map` (mirror M2 atom-buffer pattern) |
| Modify | `~/src/spectra/rust/spectra-renderer/src/renderer.rs` | set `u_sdf_textured` when the textured buffers are present (mirror `u_sdf_atom_material`) |
| Modify | `crates/vox_render/src/splat_backend.rs` | cell-grid build, `SdfUvParams`/`SdfApertureRect`/`SdfChannelMaterials`, `pathtrace_sdf_scene_textured_to_rgba`, wave-1 GPU tests |
| Modify | `~/Ochroma/projects/urban_horizon/src/asset/mod.rs` | `ReadyAssetApertureRect` + `ReadyAssetPayload.apertures` (serde-default) |
| Modify | `~/Ochroma/projects/urban_horizon/src/bin/game_asset_cook.rs` | `derive_aperture_rects` from glass-zone triangles + payload wiring + cook print + tests |
| Test | `crates/vox_render/src/splat_backend.rs` (mod tests) | `sdf_gather_grid_matches_linear`, `sdf_craftsman_textured_quality` (the Done-When gates) |
| Test | `~/Ochroma/projects/urban_horizon/src/bin/game_asset_cook.rs` (mod tests) | `derive_aperture_rects_finds_craftsman_windows` |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Grid gather ≡ linear, ~80× fewer atom reads | `sdf_gather_grid_matches_linear`: same craftsman frame both ways, prints `speedup=<N>x max|Δrgb|=<d>`; asserts `d < 1.0/255.0 && N >= 20` | `assert!(gather.is_some())` |
| Textured facade shows sub-atom detail | `sdf_craftsman_textured_quality` prints facade-crop gradient energy textured vs flat-blend; asserts ratio ≥ 3.0 | `assert!(png.len() > 0)` |
| Texture layer consistent with cooked atoms | same test prints `mean |hit_rgb - atom_rgb| = <D>`; asserts D < 0.12 over ≥ 2,000 facade px | `assert!(color != black)` |
| Per-channel PBR dispatch | same test prints roof vs facade roughness-AOV means; asserts `|Δ| > 0.05` | single material everywhere passing |
| Normal maps bend SDF normals | same test prints facade normal-AOV variance ratio mapped/unmapped; asserts ≥ 4.0 | `assert!(normal.length() ≈ 1)` |
| Aperture rects from the cook | civitas test prints `craftsman: <n> aperture rects` and per-rect extents; asserts `n >= 6` and every rect half-extent in [0.15, 2.5] m | `assert!(!rects.is_empty())` only |
| Muntins anchored to rects | textured test counts dark bar crossings in a window crop; asserts `(v,h) == (1,1)` for the craftsman's DoubleHung | `assert!(window_px > 0)` |
| Interior-mapped glass | textured test renders A (interior) vs B (hollow control), prints `<P> px mean|ΔRGB|=<M>`; asserts P ≥ 200 && M ≥ 0.15 | checking pane is "not black" |

---

## Task 1: Cell-grid atom gather — build host-side, read in the megakernel, prove parity + speedup

**Files:**
- Modify: `crates/vox_render/src/splat_backend.rs`
- Modify: `~/src/spectra/rust/spectra-scene-state/src/layers.rs`
- Modify: `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs`
- Modify: `~/src/spectra/rust/spectra-renderer/src/renderer.rs`
- Modify: `~/src/spectra/slang/megakernel.slang`
- Test: `crates/vox_render/src/splat_backend.rs` (`sdf_gather_grid_matches_linear`)

**Acceptance:** `SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --lib sdf_gather_grid_matches_linear -- --nocapture --test-threads=1` → prints `gather: grid=<G> atoms/hit (avg) vs linear=12229, speedup=<N>x, max|Δrgb|=<d>` with N ≥ 20 and d < 1/255, plus the M0/M1/M2 tests staying green (`sdf_building_renders_solid_surface` ≥ 0.97, `sdf_craftsman_atom_material_and_glass` unchanged).

**Wiring requirement:** Grid build called from inside the new `pathtrace_sdf_scene_textured_to_rgba` (created here as a thin extension of the M2 body — later tasks grow it); kernel gather rewritten inside `sdf_gather_atom_material` behind `u_sdf_textured != 0` with the linear loop kept verbatim for `u_sdf_textured == 0`. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — in `splat_backend.rs` tests, clone the M2 test's craftsman loading; render once via `pathtrace_sdf_scene_with_atoms_to_rgba` (linear oracle) and once via the new `pathtrace_sdf_scene_textured_to_rgba` with `u_sdf_textured` ON but materials empty (gather-only mode: same flat-blend shading); compare RGBA byte-wise (`max|Δrgb|`), print the speedup from the per-instance atom/cell counts (host-computed: avg atoms tested per probe = sum over 3×3×3 cells of counts at 1,000 random surface points).
- [ ] **Step 2: Run to verify it fails** — `... --lib sdf_gather_grid_matches_linear` → FAIL: `cannot find function pathtrace_sdf_scene_textured_to_rgba`.
- [ ] **Step 3: Implement host grid** — per instance: cell size = `max(0.5, 2.0 * median_atom_spacing)` (cap grid ≤ 32³, floor ≥ 4³ — derive dims from the instance's local AABB); counting-sort atoms by cell into the SoA order; emit `g_sdf_atom_grid_headers` (cell_dims[3], cell_size, table_offset, atom_base — 6 floats/instance) + `g_sdf_atom_cell_table` ((start,count) per cell). Add `SdfLayer::from_parts_textured(...)` carrying the two new vecs (plus pass-through of the M2 fields); upload + bind in `uploader.rs::bind_to_map` exactly as the four M2 atom buffers are; in `renderer.rs`, set `u_sdf_textured = 1` when grid buffers are non-empty (mirror the `u_sdf_atom_material` site).
- [ ] **Step 4: Implement kernel gather** — in `sdf_gather_atom_material`, branch on `u_sdf_textured != 0`: world hit → instance-local cell coords (atoms are world-space in M2's layout — grid is built over the WORLD positions of the instance's atoms, so cell math is world-space per instance); iterate the 3×3×3 neighbourhood clamped to grid dims; identical K=6 insertion sort + glass-radius logic over only those atoms. Keep the linear loop for the off path.
- [ ] **Step 5: Run — verify non-trivial output** — expected: `speedup=78x` (order-of), `max|Δrgb|=0.0000`–`0.0039`, M0/M1/M2 tests green.
- [ ] **Step 6: Commit** — spectra: `feat(megakernel): cell-grid SDF atom gather (parity-gated, u_sdf_textured)`; ochroma: `feat(spectra-native): grid-accelerated atom gather + textured SDF entry point scaffold`.

---

## Task 2: Per-channel PBR materials + texture atlas bound to the SDF path (host packing)

**Files:**
- Modify: `crates/vox_render/src/splat_backend.rs` (`SdfChannelMaterials`, `SdfUvParams`, extend `pathtrace_sdf_scene_textured_to_rgba` signature to the design §6 form)
- Modify: `~/src/spectra/rust/spectra-scene-state/src/layers.rs`, `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` (channel-material + uv-param buffers)
- Test: extend `sdf_gather_grid_matches_linear`'s harness into a shared helper; new assertions live in Task 3's test

**Acceptance:** `scripts/build-spectra-native.sh build -p vox_render --features spectra-native` compiles; a unit test `sdf_channel_material_packing` (CPU-only, no GPU) asserts the packed buffer is `9 * instances` floats with the craftsman's facade slot resolving to a material whose `albedo_tex >= 0` and roughness == the cooked `roughness_factor`, values printed.

**Wiring requirement:** `pathtrace_sdf_scene_textured_to_rgba` calls `renderer.set_texture_atlas(...)` AFTER `load_scene_state` (the verified-order landmine), packing `materials`/`textures` through the same `build_texture_atlas` helper `pathtrace_mesh_textured_to_rgba` uses (splat_backend.rs:1232). Channel→material table uploaded via the Task-1 layer/uploader path. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — `sdf_channel_material_packing`: build `SdfChannelMaterials` from a hand-rolled cooked-material list (facade→mat 0 with albedo_tex 0, roof→mat 1, glass→-1), pack, assert layout + values printed.
- [ ] **Step 2: Run to verify failure** — type doesn't exist yet.
- [ ] **Step 3: Implement** — `SdfChannelMaterials { material_for_channel: [i32; 9] }`; extend the entry point with `uv_params`, `channel_materials`, `materials`, `textures`, `apertures_per_instance` (empty OK), `lighting_tier` (0 only for now); pack `g_sdf_channel_material` (9 floats/instance) + `g_sdf_volume_uv_params` (4 floats/volume); reuse the mesh path's material/atlas packing verbatim (materials → `MaterialLayer`, textures → `set_texture_atlas`).
- [ ] **Step 4: Wire at exact callsite** — inside `pathtrace_sdf_scene_textured_to_rgba`, after `renderer.load_scene_state(&scene)`: `renderer.set_texture_atlas(&tex_descs, &tex_data, textures.len() as u32)` (copy the guard + error mapping from `pathtrace_mesh_textured_to_rgba:424-429`).
- [ ] **Step 5: Run — verify non-trivial output** — packing test prints the 9-slot table with real indices; build green.
- [ ] **Step 6: Commit** — `feat(spectra-native): per-channel material table + texture atlas on the SDF path`.

---

## Task 3: Box-UV textured shade in the megakernel — the M1 gate

**Files:**
- Modify: `~/src/spectra/slang/megakernel.slang` (opaque SDF branch)
- Test: `crates/vox_render/src/splat_backend.rs` → `sdf_craftsman_textured_quality` (created here; grows in Tasks 4–6)

**Acceptance:** the Done-When command prints `M1 texture: facade detail energy <E>x flat-blend (gate >= 3.0x) -> PASS` and `M1 consistency: mean |hit_rgb - atom_gather_rgb| = <D> (gate < 0.12) -> PASS` and `roughness roof=<a> facade=<b>` with `|a-b| > 0.05`, and writes `sdf_craftsman_textured.png` + `sdf_craftsman_flat_control.png`.

**Wiring requirement:** inside the SDF shade block's opaque branch (megakernel.slang, after the `SdfAtomGather` at ~line 1250), gated `u_sdf_textured != 0 && channel material >= 0`. The flat M2 path must remain byte-identical when the gate is off. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — `sdf_craftsman_textured_quality`: load craftsman SDF + atoms + cooked per-channel materials (parse the cooked payload's `ReadyAssetPbrMaterial` list via serde from the pack JSON — same files the M2 test reads), build `PbrMaterial`s/`TextureImage`s through `TextureCache` (use the test-local copy of the resolver pattern: diffuse sRGB→linear+tint, normal/rough raw), render textured + flat-control; compute (a) mean |∇luminance| over a fixed facade crop (front wall, the M2 test's window rows give the crop rect) for both renders, (b) per-pixel |hit − atom_gather| via a second flat render diff on facade-classified pixels, (c) roughness AOV means per crop (extend the RGBA return or read `g_aov_roughness` via the frame's AOV outputs — the M2 test pattern shows beauty-only; add an `aov_roughness: bool` debug flag to the entry point that swaps roughness into the alpha channel for the test). Assert the three gates.
- [ ] **Step 2: Run to verify failure** — renders are identical (kernel not implemented) → energy ratio ≈ 1.0 → FAIL printed with both energies.
- [ ] **Step 3: Implement kernel** — in the opaque branch: instance-local hit `lp = quat_rotate(inv_rot, (hit - pos)/scale)`; UV per the cook convention (IMPORTANT NOTES) using `g_sdf_volume_uv_params`; material id = `g_sdf_channel_material[inst*9 + channel]`; if id ≥ 0: fetch `MaterialData`, sample albedo (EWA on `u_bounce==0` with isotropic footprint `t * u_pixel_angle / 2.5` as both axes — add the uniform if absent; bilinear otherwise), roughness map → `mat.roughness`; footprint > 1 tile ⇒ skip fetches, keep gathered atom colour (design §4.2 LOD); shade with the existing Lambert/light loop using the texel albedo. Keep glass routing untouched.
- [ ] **Step 4: Wire at exact callsite** — the branch replaces `albedo = g.color` with the textured resolve when gated; AOV albedo writes use the texel.
- [ ] **Step 5: Run — verify non-trivial output** — energy ratio ≥ 3.0 printed (clapboard/shingle texture visible in the PNG), consistency < 0.12, roughness split printed.
- [ ] **Step 6: Commit** — spectra: `feat(megakernel): box-UV per-channel PBR textures at SDF hits (M1)`; ochroma: `feat(spectra-native): textured craftsman quality gate (M1)`.

---

## Task 4: Normal maps on the SDF surface (box-projection TBN)

**Files:**
- Modify: `~/src/spectra/slang/megakernel.slang`
- Test: extend `sdf_craftsman_textured_quality`

**Acceptance:** Done-When command additionally prints `M2 normals: facade normal-AOV variance <V>x flat (gate >= 4.0x) -> PASS`; the PNG shows clapboard relief responding to the sun direction (eyeball line in the test output names the sun azimuth used).

**Wiring requirement:** same gated branch; tangent = the UV plane's first axis, bitangent = second (per dominant-axis case), normal perturbed `normalize(T*n.x + B*n.y + N*n.z)` from `normal_tex` (raw [0,1] → [-1,1]); skipped when footprint-LOD skips textures. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test extension** — render with normal maps off (uniform-driven debug: reuse `u_sdf_textured == 1` vs `== 2` levels — 1 = albedo/rough only, 2 = +normals) and compute facade normal-AOV variance ratio; assert ≥ 4.0.
- [ ] **Step 2: Run to verify failure** — ratio ≈ 1.0.
- [ ] **Step 3: Implement** — TBN per dominant axis (matches the UV cases exactly: Y-plane → T=+X,B=+Z(flipped per the cook's `d/2 − z`); X-plane → T=+Z(flipped),B=+Y; Z-plane → T=+X,B=+Y); fetch `mat.normal_tex`, perturb, renormalize; feed the perturbed normal to the light loop AND the normal AOV; keep the geometric gradient normal for ray offsets.
- [ ] **Step 4: Wire** — `u_sdf_textured >= 2` enables the perturbation inside the same branch.
- [ ] **Step 5: Run — verify** — variance ratio printed ≥ 4.0; M1 gates still green.
- [ ] **Step 6: Commit** — `feat(megakernel): box-projection TBN normal maps on SDF surfaces (M2)`.

---

## Task 5: Aperture rects at cook — exact windows from glass-zone triangles

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/asset/mod.rs` (`ReadyAssetApertureRect`, `ReadyAssetPayload.apertures` serde-default)
- Modify: `~/Ochroma/projects/urban_horizon/src/bin/game_asset_cook.rs` (`derive_aperture_rects` + wiring + `[apertures]` print)
- Test: cook test `derive_aperture_rects_finds_craftsman_windows`

**Acceptance:** `cd ~/Ochroma/projects/urban_horizon && cargo test --bin game_asset_cook derive_aperture_rects_finds_craftsman_windows -- --nocapture` prints `craftsman: <n> aperture rects` with n ≥ 6 and per-rect `<w>x<h> m @ normal (<nx>,<ny>,<nz>)` lines, every half-extent in [0.15, 2.5] m, every normal within 5° of an axis; and `cargo run --release --bin game_asset_cook` prints one `[apertures] <id>: <n> rects` line per asset with glass zones.

**Wiring requirement:** `derive_aperture_rects(mesh, zones, desc)` called from the recipe cook body where `atomize_mesh_triangles` is called (same inputs in scope), result stored on the payload. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — load the craftsman forge contract + mesh exactly as the existing cook tests at game_asset_cook.rs:4955 do; call `derive_aperture_rects`; assert count/extents/axis-alignment; print every rect.
- [ ] **Step 2: Run to verify failure** — function not found.
- [ ] **Step 3: Implement** — collect triangles whose zone material name contains `glass`/`window_glass`; union-find clusters on shared vertices (positions quantized to 1 mm); per cluster: average normal (assert near-planar: max vertex deviation from the mean plane < 2 cm, else split by plane), in-plane PCA for `right`/`up`, half-extents from projected bounds; `style_id` from the description JSON's window style string (`DoubleHung=0, Casement=1, Bay=2, Lancet=3, Sash=4`, default 0).
- [ ] **Step 4: Wire** — store on `ReadyAssetPayload.apertures`; print the `[apertures]` cook line.
- [ ] **Step 5: Run — verify** — n ≥ 6 real rects printed with plausible window dimensions (~0.5–1.0 m half-extents).
- [ ] **Step 6: Commit** — `feat(cook): derive glass aperture rects from forge zones (windows as exact rectangles)`.

---

## Task 6: Rect-anchored glass + procedural muntins/sill bevel + interior mapping (kernel)

**Files:**
- Modify: `crates/vox_render/src/splat_backend.rs` (`SdfApertureRect`, world-transform + upload via the layer; test extension)
- Modify: `~/src/spectra/rust/spectra-scene-state/src/layers.rs`, `~/src/spectra/rust/spectra-scene-upload/src/uploader.rs` (aperture buffers)
- Modify: `~/src/spectra/slang/megakernel.slang` (point-in-rect classification, muntin/bevel detail, interior box-room on transmission)
- Test: extend `sdf_craftsman_textured_quality`

**Acceptance:** Done-When command prints all three M2 lines: `muntins: window crop crossings v=1 h=1 (DoubleHung expects v=1 h=1) -> PASS`, `interior: <P> window px, A/B vs hollow mean|ΔRGB|=<M> (gate >= 200 px, >= 0.15) -> PASS`, and `glass px outside rects = 0`; final PNG shows muntin bars and a lit room behind the panes.

**Wiring requirement:** glass classification in the SDF shade block prefers point-in-rect (hit within 1.5 surf_eps of a rect plane AND inside extents) over the k-NN radius when `u_num apertures > 0` for the instance; muntin/bevel detail computed in rect-local (s,t); transmission branch replaces the scene re-trace with the analytic box room when `u_sdf_textured != 0` (reflection continuation unchanged). Stubs = **task failure**.

- [ ] **Step 1: Write the failing test extension** — (a) muntin metric: in the brightest window's crop (rect projected to screen via the known camera), count dark vertical/horizontal bar crossings on the crop's center row/column (luminance dips > 25% below crop median, deduped within 3 px); assert `(v,h) == (1,1)` (DoubleHung: center sash split + meeting rail). (b) interior A/B: render with interior mapping vs `u_sdf_textured` glass-hollow control; count window px (from rect projection) with |ΔRGB| ≥ 0.15; assert ≥ 200 px and print mean. (c) classification: assert zero glass-classified px whose ray hit lies outside all rect projections (sample the AOV crypto/albedo to identify glass px — the glass tint albedo write marks them).
- [ ] **Step 2: Run to verify failure** — crossings (0,0); A/B diff ≈ 0.
- [ ] **Step 3: Implement** — host: transform rects to world per instance (rotate axes, scale extents), pack 15 floats/rect + per-instance (offset,count); kernel: (i) classification — distance to rect plane + in-extent test; (ii) muntins — `s,t` in pane-local metres; DoubleHung: bar at `|s| < 0.02` and `|t| < 0.025` (meeting rail) ⇒ reclassify TRIM (use trim channel material, bend normal by a 45° box-bevel profile); (iii) sill/frame band — within 0.06 m of the rect border ⇒ trim material + bevel normal tilted toward the rect center + 0.85 AO factor (the recess read); (iv) interior — on transmission, rect-local box room depth 3.0 m: intersect the refracted dir with the 5 inner faces, shade per-face albedo (floor 0.35 warm, walls 0.55 neutral, ceiling 0.8) lit by `0.5 + 0.5*hash(rect_id)` daylight factor + warm emissive `hash(rect_id) > 0.6` evening windows OFF by day (use sun elevation from the rig: emissive only when sun_dir.y < 0.2); multiply by glass tint; write to film and terminate (no re-trace).
- [ ] **Step 4: Wire** — all inside the existing glass branch, gated; the k-NN radius path stays for instances with zero rects.
- [ ] **Step 5: Run — verify** — all M2 gates PASS with real printed values; the PNG eyeball line printed (`muntin bars + lit interiors visible — verify sdf_craftsman_textured.png`).
- [ ] **Step 6: Commit** — spectra: `feat(megakernel): aperture-rect glass, procedural muntins/bevel, interior-mapped panes (M2)`; ochroma + civitas wiring commits.

---

## Task 7: Wave-1 ledger + green-suite proof

**Files:**
- Modify: `crates/vox_render/src/splat_backend.rs` (final print block in `sdf_craftsman_textured_quality`)

**Acceptance:** the Done-When command prints, after all gates, one ledger line: `[sdf-quality] wave1 ledger: gather <G> loads/hit (was ~49k), textures <T> MB (<S> sets), rects <R>, frame <F>s @ 320x320 spp<N> on 780M` with measured values — and the full prior suite (`sdf_building_renders_solid_surface`, `sdf_city_block`, `sdf_craftsman_atom_material_and_glass`, `vulkan_backend_renders_quad`) passes unchanged in the same run command set.

**Wiring requirement:** ledger computed from real counters (host grid stats, atlas byte size, rect count, wall-clock around the render call) — no constants. Stubs = **task failure**.

- [ ] **Step 1: Add the ledger assertions** — frame time printed (no gate — the 780M is the floor, the budget verdict belongs to the perf track); texture MB == sum of uploaded `TextureImage` float bytes; assert gather loads/hit < 1,000.
- [ ] **Step 2: Run the full SDF suite** — all listed tests `--test-threads=1`; expected: every prior gate value unchanged (M0 0.976 ±0.005, M1 12/12, M2 hue buckets ≥ 3 + glass ≥ 200 px).
- [ ] **Step 3: Verify the PNGs by eyeball** — textured vs control; record the verdict in the commit message.
- [ ] **Step 4: Commit** — `feat(spectra-native): wave-1 SDF housing quality ledger — textured craftsman proven on the 780M`.

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks exist
- [x] Every `Acceptance` criterion names a real non-trivial expected output (printed measured values, gates, PNGs — not "tests pass")
- [x] Every `Wiring requirement` names an exact function and exact file
- [x] `IMPORTANT NOTES` contains the real API signatures from the design doc (verified against code 2026-06-12)
- [x] `File Map` lists every file that appears in any task
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies in the implementation code
- [x] `Done When` names a specific command and specific human-observable result
- [x] All types, method names, and signatures are consistent across all tasks (`pathtrace_sdf_scene_textured_to_rgba` is created in Task 1, grown in Task 2, consumed in Tasks 3–7)
- [x] Concurrency rail: every spectra task rebases onto the perf agent's megakernel head; all kernel changes uniform-gated default-off
