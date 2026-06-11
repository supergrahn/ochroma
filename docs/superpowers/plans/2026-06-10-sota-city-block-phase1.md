# SOTA City Block — Phase 1 Plan (Debt, Vertical Slice, Asset Factory Prereqs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One beautiful, technically honest city block: cooked Forge assets render in Spectra with real PBR textures from a clean pipeline, on a dressed lot, with deterministic inspection gates — and the upstream debt that today's texture work exposed is paid off.
**Done When:** `cd ~/Ochroma/projects/civitas_care && LD_LIBRARY_PATH=$HOME/slang-sdk/lib SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json FORGE_PREVIEW_SPP=32 cargo run --release --features spectra --bin forge_pathtrace` writes `renders/spectra/craftsman/{iso,front,side,porch,debug_mat,debug_uv}.png` **and** `renders/spectra/rowhouse/{iso,front,side,porch,debug_mat,debug_uv}.png`, the binary prints `gates: PASS` from the pixel-stat checks, and a human looking at `craftsman/porch.png` sees a panelled door leaf and at `craftsman/iso.png` sees a dressed lot (no loud 12× concrete repeat).
**Architecture:** Keep the proven path — Forge generates triangle meshes + material zones with `polyhaven://` URIs → game cook produces ReadyAssetPayload (mesh + atoms + zones + sockets + SDF) → `vox_render::pathtrace_mesh_textured_to_rgba` renders through Spectra/Vulkan with a flat float atlas. Phase 1 hardens that path (winding, interior proxies, first-frame race), parameterizes lighting, and feeds it more CC0 texture variety via the PolyHaven API. Later SOTA items (virtualized rendering, living instances) get design docs here, full plans after.
**Design Document:** `docs/superpowers/specs/2026-06-10-virtualized-splat-rendering-design.md`, `docs/superpowers/specs/2026-06-10-living-building-instances-design.md` (produced by Tasks 9–10)
**Tech Stack:** Rust (workspace toolchains as-is), Spectra Slang→SPIR-V on Vulkan/RADV, wgpu interactive path untouched, PolyHaven public API (CC0), image 0.25
**Build:** per-repo `cargo test`; GPU verification needs the env in **Done When**; forge repo has **NO git** — edit with care.

---

## IMPORTANT NOTES

- **Repos:** engine `~/src/ochroma` (branch `blitz/day1-foundation`, uncommitted work present), `~/src/spectra` (git, uncommitted work), `~/src/forge` (**NOT a git repo**), game `~/Ochroma/projects/civitas_care` (git master, uncommitted). **Do NOT run `git commit` anywhere** — the user commits. Template commit steps are replaced by "leave uncommitted".
- `vox_render::splat_backend` (feature `spectra-native`; game feature `spectra`):
  - `pub fn pathtrace_mesh_textured_to_rgba(positions: &[[f32;3]], normals: &[[f32;3]], uvs: &[[f32;2]], indices: &[[u32;3]], material_ids: &[u8], materials: &[PbrMaterial], textures: &[TextureImage], eye: [f32;3], target: [f32;3], fov_y: f32, width: u32, height: u32, spp: u32, sun_dir: [f32;3]) -> Result<Vec<u8>, String>` — additive changes only; existing callers must keep compiling.
  - `pub struct PbrMaterial { base_color: [f32;3], roughness: f32, metallic: f32, emission_strength: f32, albedo_tex: i32, roughness_tex: i32, normal_tex: i32, uv_scale: [f32;2] }` (+ `Default`).
  - `pub struct TextureImage { width: u32, height: u32, channels: u32, data: Vec<f32> }` — data is **LINEAR**; sRGB decode is the caller's job (diffuse only, never normal/roughness).
  - Vulkan `MaterialData` packing is the 156-float layout in `pack_vulkan_mesh_material`: tex ids at `a[28..=31]`, uv transform at `a[94..=98]`. Do not re-derive; calibration anchors are in the function.
- Spectra atlas: `Renderer::set_texture_atlas(texture_descs: &[u32], texture_data: &[f32], num_textures: u32)` — descs are `[offset(FLOATS), width, height, channels]`; must be called **after** `load_scene_state` (silent no-op before). Kernel samples via `sample_texture_ewa` (mip-less, 8×8 window cap — keep textures ≤256², normals ≤128²).
- Game texture cache: `TextureCache::load(&mut self, uri: &str, tint: [f32;3]) -> Option<Arc<TextureData>>` in `civitas_care/src/asset/textures.rs`; mapping table `STEM_SETS` is the single source of truth; diffuse texels are pre-multiplied by zone tint (kernel REPLACES albedo with texel).
- Forge UV projection: `forge_mesh::apply_box_projection(mesh, density)` **multiplies** (`uv = pos * density`); 1 tile per 2.5 m ⇒ pass `1.0/2.5`. Called in `generate_asset` (lib.rs), style-agnostic.
- Forge material ids (per-triangle u8): 0=wall 1=roof 2=glass 3=reveal 4=trim 5=cornice 6=door; ground plane in the harness is 7.
- Known in-flight fixes (do not re-litigate): megakernel backface pass-through now gated on `MAT_VEGETATION`; EWA wraps UVs; harness warm-up frame absorbs the first-frame compile race until Task 3 lands the real fix.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `~/src/forge/crates/building/src/lib.rs` (or new `orient.rs`) | outward-winding pass in `generate_asset` |
| Test   | `~/src/forge/crates/building/src/lib.rs` tests | outwardness ≥99% on craftsman + victorian |
| Modify | `~/src/forge/crates/building/src/interior.rs` | interior proxies inside the +z body, not mirrored outside |
| Modify | `~/src/forge/crates/building/src/wall.rs` / `facade/` | panelled door leaf geometry |
| Modify | `~/src/spectra/rust/spectra-renderer/src/renderer.rs` | first `render()` waits for kernel compilation (no mismatched fallback frame) |
| Modify | `~/src/ochroma/crates/vox_render/src/splat_backend.rs` | additive `LightRig` + `pathtrace_mesh_lit_to_rgba` wrapper |
| Create | `~/Ochroma/projects/civitas_care/src/bin/polyhaven_fetch.rs` | PolyHaven API fetcher, pinned manifest, CC0 |
| Modify | `~/Ochroma/projects/civitas_care/src/asset/textures.rs` | mapping table expansion for fetched sets |
| Modify | `~/Ochroma/projects/civitas_care/src/bin/forge_pathtrace.rs` | lot treatment, LightRig use, rowhouse views, `gates: PASS` pixel-stat checks |
| Modify | `~/Ochroma/projects/civitas_care/src/bin/game_asset_cook.rs` | drop interior-strip workaround once Task 2 lands (keep test) |
| Create | `~/src/ochroma/docs/superpowers/specs/2026-06-10-virtualized-splat-rendering-design.md` | SOTA item 3 design (metric-led) |
| Create | `~/src/ochroma/docs/superpowers/specs/2026-06-10-living-building-instances-design.md` | SOTA item 4 design |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Outward winding | ≥99% of generated craftsman triangles have geometric normal agreeing with outward orientation (printed count) | `assert!(mesh.indices.len() > 0)` |
| Interior containment | all interior-proxy vertices satisfy `0 < z < depth` (printed min/max) | `assert!(proxies.is_some())` |
| First-frame correctness | first `render()` of a fresh process produces the same mean luma (±2%) as the second render of the same scene | `assert!(frame.is_ok())` |
| LightRig | same scene rendered with `LightRig{sun_intensity: 0.5×}` is darker by a measured margin; `Default` byte-identical to old rig | function exists |
| PolyHaven fetch | pinned manifest downloads N sets; files exist with nonzero size; resolver maps new stems | HTTP 200 asserted only |
| Pixel-stat gates | harness prints `gates: PASS` only when per-view mean/variance in bands; corrupting a render makes it print `gates: FAIL <view>` | always-true bands |

---

## Task 1: Outward triangle winding in `generate_asset`

**Files:** Modify `~/src/forge/crates/building/src/lib.rs` (new private `orient_outward(&mut Mesh)`), tests in same file.
**Acceptance:** `cd ~/src/forge && cargo test -p forge-building winding -- --nocapture` → prints `craftsman outward: <n>/<total>` with n/total ≥ 0.99, and full suite stays green.
**Wiring requirement:** called from `generate_asset` after `apply_box_projection`, before `material_zones_for_mesh` (zone ranges depend on triangle order — do NOT reorder triangles, only swap index pairs within a triangle).

- [ ] Step 1: failing test — generate default craftsman, compute per-triangle geometric normal vs. solid-angle/winding-number orientation oracle (point `centroid + ε·ĝ` outside vs inside via generalized winding number), assert ≥99% outward; print the count.
- [ ] Step 2: run, verify it fails with today's ~50% inward walls.
- [ ] Step 3: implement `orient_outward`: per triangle, evaluate generalized winding number at `centroid ± ε·ĝ`; if the `+ĝ` side is more interior, swap `indices[t][1]` and `indices[t][2]`. O(n²) on ≤3k tris is fine. Recompute `compute_normals` + `compute_tangents` after.
- [ ] Step 4: wire in `generate_asset` exactly as stated; victorian + school paths get it free.
- [ ] Step 5: full `cargo test -p forge-building` and `cargo test -p forge-cli --test plugin_contract` green; print non-trivial counts.
- [ ] Step 6: leave uncommitted (forge has no git).

## Task 2: Interior proxies inside the body

**Files:** Modify `~/src/forge/crates/building/src/interior.rs`; tests there.
**Acceptance:** `cargo test -p forge-building interior -- --nocapture` → prints proxy AABB strictly inside `[0,width]×[0,depth]` for a 10×8 craftsman (real numbers, not 0..0).
**Wiring requirement:** `generate()` with `generate_interior: true` calls fixed `generate_interior_proxies`; the game-side strip in `game_asset_cook.rs` (`strip_misplaced_interior_proxies`) must find **nothing** to strip after this (its test flips to assert zero stripped).

- [ ] Step 1: failing test asserting all proxy vertices have `0.0 < z < depth`.
- [ ] Step 2: verify fails (today proxies mirror to negative/outside z).
- [ ] Step 3: fix the z-sign/origin assumption in `interior.rs` (body occupies +z, front wall at z=0).
- [ ] Step 4: wired already via `generate()`; update game cook test to assert `stripped == 0` after recook.
- [ ] Step 5: forge suite + `cargo test --bin game_asset_cook` green.
- [ ] Step 6: leave uncommitted.

## Task 3: Spectra first-frame correctness

**Files:** Modify `~/src/spectra/rust/spectra-renderer/src/renderer.rs` (and `kernel_set` if that is where late kernels are skipped).
**Acceptance:** `cd ~/src/ochroma && SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json scripts/build-spectra-native.sh test -p vox_render --features spectra-native --lib first_frame_matches_second -- --nocapture` → prints two mean lumas within 2% on a textured quad, first try, fresh process.
**Wiring requirement:** the fix lives in `Renderer::render()` (or its kernel-compile path) so **every** caller gets a correct first frame; the warm-up in `forge_pathtrace.rs` becomes redundant (leave it, it is cheap).

- [ ] Step 1: failing test in vox_render (new `#[test] first_frame_matches_second`) rendering the checker quad twice in one process, asserting ≤2% mean-luma delta — currently the first frame diverges.
- [ ] Step 2: verify failure on RADV.
- [ ] Step 3: root-cause in spectra-renderer: `try_compile_all` results vs. dispatch skipping (`Kernel compile failed (skipping)`) on the first sample loop; make `render()` block until required kernels are compiled or return `Err` — never silently shade through a mismatched path.
- [ ] Step 4: wired by construction (inside `render()`).
- [ ] Step 5: both vox_render GPU tests + `vulkan_backend_renders_quad` green; print lumas.
- [ ] Step 6: leave uncommitted (spectra repo).

## Task 4: `LightRig` parameterization (additive)

**Files:** Modify `~/src/ochroma/crates/vox_render/src/splat_backend.rs`.
**Acceptance:** `scripts/build-spectra-native.sh test -p vox_render --features spectra-native --lib light_rig -- --nocapture` → prints default-rig vs half-sun mean lumas with the half-sun image measurably darker (>10%), and default-rig output byte-identical to the legacy entry point on the same scene.
**Wiring requirement:** new `pub struct LightRig { pub sun_dir: [f32;3], pub sun_color: [f32;3], pub sun_intensity: f32, pub sky_intensity: f32, pub camera_fill: f32, pub rim_fill: f32 }` with `Default` reproducing today's hardcoded rig exactly; new `pathtrace_mesh_lit_to_rgba(..., rig: &LightRig)`; existing `pathtrace_mesh_textured_to_rgba` delegates with `LightRig{sun_dir, ..Default::default()}`.

- [x] Steps 1–5 per template: failing test → implement → delegate wiring → green with printed lumas.
- [x] Step 6: leave uncommitted.

## Task 5: PolyHaven fetcher + mapping expansion

**Files:** Create `~/Ochroma/projects/civitas_care/src/bin/polyhaven_fetch.rs`; modify `src/asset/textures.rs`.
**Acceptance:** `cargo run --bin polyhaven_fetch` → prints one line per pinned set ending `OK (cached)` or `OK (downloaded)`; then `cargo test --lib asset::textures` green including a new test resolving every pinned stem to an on-disk diffuse.
**Wiring requirement:** pinned manifest (const in the bin) of ~8–12 CC0 sets chosen for the block (painted wood siding, wood shingles/roof tiles, asphalt, grass/lawn, brick variants, plaster, concrete pavers, metal roof); downloads `1k` diffuse/normal/roughness JPGs via `https://api.polyhaven.com/files/<id>` into `assets/buildings/forge_starter/textures/polyhaven/<set>/1k/{diffuse,normal,roughness}.jpg`; idempotent; `STEM_SETS` gains the new stems used by Tasks 7–8. CC0 needs no attribution — still print the license line.

## Task 6: Real door leaf (forge)

**Files:** Modify `~/src/forge/crates/building/src/wall.rs` (door cell emission) or `facade/door_leaf.rs` (new).
**Acceptance:** `cargo test -p forge-building door_leaf -- --nocapture` → prints door-leaf triangle count ≥ 60 and panel inset depth 0.02–0.04 m measured from vertex data (today the leaf is 2 triangles).
**Wiring requirement:** `emit_door_cell` emits the leaf (stiles/rails/2 recessed panels, 0.06 m proud of the reveal, MAT_DOOR) so every archetype with a door gets it.

## Task 7: Lot treatment + lighting pass (game harness)

**Files:** Modify `~/Ochroma/projects/civitas_care/src/bin/forge_pathtrace.rs`.
**Acceptance:** rendered `craftsman/iso.png` ground is a lawn/pavers lot (fetched sets) with no visible 12× repeat; `front.png` mean luma in 110–150 (no white-wash) — printed by the gates of Task 8.
**Wiring requirement:** ground quad split into lawn + walkway strip to the door socket (two materials, uv_scale ≤4); views use `pathtrace_mesh_lit_to_rgba` with a tuned `LightRig` (lower fills, sun azimuth lighting the front-left corner of the iso view); porch view eye pulled to 5.5 m / 2.1 m height so the leaf + columns frame.

## Task 8: Rowhouse through the harness + pixel-stat gates

**Files:** Modify `forge_pathtrace.rs` (+ small `textures.rs` mapping additions).
**Acceptance:** the **Done When** command prints `gates: PASS` and writes both view sets; corrupting any one PNG band (test mode `FORGE_GATE_SELFTEST=1`) prints `gates: FAIL <view>`.
**Wiring requirement:** generalize the cooked-asset view block over `[craftsman, rowhouse]` payloads; per-view gate = mean luma band + per-channel std floor (catches flat/washed frames) computed from the freshly rendered buffer before PNG write; `gates:` summary printed last.

## Task 9: Design doc — virtualized splat rendering (SOTA item 3)

**Acceptance:** `test -f ~/src/ochroma/docs/superpowers/specs/2026-06-10-virtualized-splat-rendering-design.md && grep -c "Done When" <same>` ≥ 1; doc uses `docs/templates/design.md`, is metric-led (e.g. “10k textured buildings @ 60 fps interactive on the 780M, Spectra stills for cinematics”), inventories existing `atom_budget`/`clas`/resident-frame code, and explicitly rejects copying Nanite's triangle-cluster design where it fights splats (Nanite docs are the philosophy reference, not the blueprint).

## Task 10: Design doc — living building instances (SOTA item 4)

**Acceptance:** same shape as Task 9 for `2026-06-10-living-building-instances-design.md`; covers cooked-asset → live instance (households/jobs/property value/state/save), names the real civitas types it binds to (parcels, demand, save), and a `Done When` someone can observe in the running game.

---

## Phase 2+ (plans to be written after their design docs land — NOT executed by this plan)

- Virtualized city rendering (from Task 9 design) — own plan doc.
- Living building instances (from Task 10 design) — own plan doc.
- Deep simulation, City UX, AI asset factory (LLM → Forge directives → cook → Spectra validation; `asset_directive_from_llm.rs` already exists as the seed) — sequenced after instances.
- Asset-audit remediation (ventilation, variation, etc.) — driven by `~/Ochroma/projects/civitas_care/docs/asset_audit_2026-06-10.md` once the audit workflow lands.

## Self-Review Checklist

- [x] Every task implements AND wires in the same task
- [x] Every `Acceptance` names a real non-trivial expected output
- [x] Every `Wiring requirement` names an exact function and file
- [x] `IMPORTANT NOTES` contains real, verified API signatures
- [x] `File Map` lists every file in any task
- [x] No stubs/`todo!()` anywhere
- [x] `Done When` names a specific command and human-observable result
- [x] Names/signatures consistent across tasks
