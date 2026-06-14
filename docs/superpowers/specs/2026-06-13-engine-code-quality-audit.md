# Design: Engine Code Quality Audit — Cross-Repo Remediation Backlog (2026-06-13)

**Status:** Draft
**Scope:** Codebase-wide quality review across all four Ochroma repos (ochroma engine, spectra renderer, forge geometry, urban_horizon game) — synthesizes six audit slices into one prioritized remediation backlog ranked by (risk-reduction + iteration-speed) / effort.
**Related:** CLAUDE.md conventions; the recent `splat_backend_tests/` and `game_asset_cook/tests.rs` test-extraction precedent.

---

## 1. Problem Statement

Measured 2026-06-13. Totals: ochroma 190,449 LOC; spectra/rust 125,291; spectra/slang 65,255; forge 18,712; urban_horizon 57,033. Build cache 7.9 GB across four `target/` dirs (all correctly gitignored).

Concrete, observable symptoms:

- A plain `cargo build` of `spectra-bin` **cannot build on this AMD box** — `spectra-bin/Cargo.toml:15` defaults to `["cuda","dlss","optix-denoiser","optix-rt","vulkan-backend"]`, demanding CUDA+OptiX+DLSS+Vulkan SDKs simultaneously.
- The engine crate `vox_data` (game-agnostic by CLAUDE.md rule) contains **city-builder domain code, not just comments** — `asset_catalog.rs:17-20` (`ResidentialBuilding`/`CommercialBuilding`/…), `:52` (`enum BuildingStyle { Victorian, Modern, … }`), `templates.rs:39-48` (a "City Builder" template enumerating "Zoning system", "Citizen simulation", "Traffic simulation"). ~85 game-concept hits across 9 files; ~14 are code-level symbols.
- Five files exceed 3,700 LOC and concentrate 39% of the urban_horizon `src` tree: `game_asset_cook.rs` (5,333 src, not domain-carved), `render_gpu/mod.rs` (5,073), `play.rs` (4,711), `game/mod.rs` (3,758), `hud.rs` (3,115).
- The test-extraction convention (proven once on `splat_backend` and once on `game_asset_cook`) was **never propagated** — ~277 ochroma + ~248 spectra + ~75 cook files keep tests inline; the worst single case is `shell/mod.rs` carrying ~2,412 test LOC inside a 4,889-line file.
- **42 forbidden weak `assert!(x.is_some()/is_ok())` tests** (ochroma 18, spectra 24, forge 0, cook ~1-2) violate "every test checks a REAL computed outcome."
- Lying feature flags: `spectra-renderer/Cargo.toml` declares `restir`/`neural-nrc`/`photon`/`path-guide`/`temporal-denoiser`/`neural-bsdf` as empty `= []` aliases that gate nothing — `--no-default-features` still compiles the full stack.

---

## 2. Done When

The remediation is complete when, for the highest-leverage items:

- `cargo build -p spectra-bin --no-default-features --features vulkan-backend` succeeds on the AMD/RADV box without a CUDA/OptiX toolchain, and `cargo build -p spectra-bin` (defaults) also succeeds on the same box.
- `grep -rn "BuildingStyle\|ResidentialBuilding\|CommercialBuilding" crates/vox_data/src` returns **zero** matches — the city taxonomy lives in `vox_app`/urban_horizon instead, and `cargo build -p vox_data` is clean.
- `grep -rn 'assert!([^)]*\.is_some())\|assert!([^)]*\.is_ok())' --include='*.rs'` across all four repos returns **zero** matches in non-borderline sites (the 42 catalogued sites resolved).
- `wc -l vox_app/src/shell/mod.rs` reports < 2,500 (test mod extracted to `shell/tests/`); `wc -l urban_horizon/src/bin/game_asset_cook.rs` reports < 700 (domain-carved into `cook/{recipes,atomize,plot,flora,materials,surfaces,sdf,details,gates}.rs`).

A human can verify each line above at the keyboard without reading implementation code.

---

## 3. Overall Health Verdict

**The codebase is structurally healthier than its file sizes suggest, with two genuine hard-rule breaches and a large-but-mechanical hygiene backlog.**

- **Clean axes (all four repos):** `todo!()`/`unimplemented!()` = **0** everywhere. All `target/` dirs correctly gitignored (0 tracked artifacts). Production code paths use `Result`, not `unwrap` (the high unwrap counts cluster in test code). Crate granularity in spectra (84 sharp single-responsibility crates) and forge (well-decomposed, zero warnings, zero dead-code suppressions) is excellent.
- **The two real breaches:** (1) **engine/game leak** — `vox_data` ships city-builder taxonomy (the only CLAUDE.md hard rule broken in shipped code); (2) **hostile/lying build config** in spectra — defaults that can't build on AMD plus feature flags that gate nothing.
- **The diffuse debt:** monster files (mostly inflated by inline test mods) and 42 weak assertions. These are *mechanical-safe* once the precedent split is scripted, and they directly buy iteration speed (smaller files compile/navigate faster; real assertions catch regressions the stubs let through).
- **Per-repo ranking (cleanest → most debt):** forge (cleanest — one god file, no violations) < ochroma-core < spectra (great crates, a few god files + lying flags) < ochroma-render-app (god-object shell) < urban_horizon (5 monster files, but it's the game layer so domain concepts are expected).

---

## 4. Remediation Backlog — Grouped by Theme

Ranking metric: **(risk-reduction + iteration-speed gain) / effort**. Effort: S ≤ half day, M ≈ 1 day, L ≈ 1-2 days. "Mechanical-safe" = pure move/rename with a golden gate or compiler to catch breakage; "needs care" = semantic restructuring.

### Theme A — Build Hygiene (highest leverage; unblocks everyone)

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| A1 | Change `default` features to a buildable-everywhere set (`["vulkan-backend"]`); make `cuda`/`optix-rt`/`optix-denoiser`/`dlss` opt-in; add a `cuda-full` convenience feature | spectra `spectra-bin/Cargo.toml:15` | A plain `cargo build` currently fails on AMD/RADV — blocks every contributor on non-NVIDIA hardware | S | mechanical-safe |
| A2 | Replace the `shader-slang-sys` `build.rs` env panic with a `compile_error!`/clear message + a workspace `.cargo/config.toml` setting `SLANG_DIR`; add a one-line `BUILD.md` recipe | spectra `shader-slang-sys-0.1.0/build.rs:18`, repo root | Build panics (not a clean cargo error) when SLANG_DIR/VULKAN_SDK unset — poor first-run UX | S | needs care (vendored crate) |
| A3 | Make the `= []` feature aliases (`restir`, `neural-nrc`, `neural-niv`, `photon`, `path-guide`, `temporal-denoiser`, `neural-bsdf`) actually gate their deps + `render_state.rs` fields, OR delete the fake flags and document deps as always-on | spectra `spectra-renderer/Cargo.toml` | Flags lie — `--no-default-features` still pulls the full neural/restir stack; misleads anyone trying to slim a build | M | needs care (field-level cfg is the bulk) |
| A4 | Commit the 13 untracked design/plan docs + `HANDOFF-CODEX.md` under `docs/superpowers/` | ochroma repo root | Currently dangling/untracked — risk of loss | S | mechanical-safe |

### Theme B — Convention Violations: Engine/Game Separation (hard rule)

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| B1 | Move `Building`/`BuildingStyle`/`ResidentialBuilding`/city-template content out of engine `vox_data` into `vox_app`/urban_horizon; replace with a generic `AssetGenerator::Parametric` / data-driven `AssetKind` registry (string tags) | ochroma `vox_data/src/asset_catalog.rs:17-20,30,52,64,107-126,225-328`, `library.rs:6`, `templates.rs:39-48`, `marketplace.rs:27,29` | The **only CLAUDE.md hard rule broken in shipped code** — city-builder taxonomy hardcoded into the foundational engine crate; ~14 code-level symbols across 9 files | L | needs care (semantic; touches asset gen API) |

### Theme C — Tests-in-Own-Files (mechanical; biggest navigation/compile win)

The `splat_backend_tests/` + `game_asset_cook/tests.rs` pattern is proven but applied to only 2 files total. Script the `#[path = "..._tests.rs"]` extraction once, then batch-apply.

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| C1 | Extract the ~2,412-LOC top-level `mod tests` to `shell/tests/` | ochroma `vox_app/src/shell/mod.rs:2477+` | Single biggest win — drops the file 4,889 → ~2,477 | S-M | mechanical-safe |
| C2 | Extract ~1,845 inline test LOC (~49% of file) to `game/tests.rs` | cook `src/game/mod.rs:1914+` | Halves the file before any risky structural carve | S | mechanical-safe |
| C3 | Extract ~976 inline test LOC to `render_gpu/tests.rs` | cook `src/render_gpu/mod.rs:4098+` | Drops 5,073 → ~4,097 | S | mechanical-safe |
| C4 | Extract ~1,060 inline test LOC to `spectral_gi_tests/` | ochroma `vox_render/src/spectral_gi.rs` | 1,843 → ~783; this file only *looks* like a monster | S | mechanical-safe |
| C5 | Extract the ~730-LOC remaining test mod (@2700) to `cudarc_backend/tests.rs`; split `cuda_stub` into its own file | spectra `spectra-gpu/src/cudarc_backend.rs` | 3,429 file; note src is still 2,700 after (needs further carve, not just split) | M | mechanical-safe |
| C6 | Extract the 1,149-LOC inline test mod (95% of file!) to `tests.rs` | spectra `spectra-renderer/src/lib.rs` | File is 1,208 LOC, 95% tests | S | mechanical-safe |
| C7 | Batch-extract remaining heavy inline-test files: `expand_draws.rs` (857), `hybrid_compose_gpu.rs` (756), `hybrid_compose.rs` (745), `instanced_select_gpu.rs` (715), `tiled_splat_renderer.rs` (588), `relight.rs` (574), `splat_rt.rs` (505); cook `play.rs` (~516), `hud.rs` (~567); forge `building/lib.rs` (2 mods, 23 tests) + `wall/roof/detail/interior` | all four repos | Removes ~18k inline-test LOC from vox_render src alone; script-driven | L | mechanical-safe |
| C8 | De-duplicate modules tested both inline AND in `tests/` (`gltf_animation`, `materials` ×3, `script_interface`, `ply_loader`, `world_save`, `vxm`, `osm_import`, `import_pipeline`) | ochroma `vox_data` / `vox_core` | Dual convention = duplicated, drifting coverage | M | needs care (decide canonical home per module) |

### Theme D — Convention Violations: Weak Assertions (42 sites)

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| D1 | Replace `assert!(x.is_some()/is_ok())` with assertions on the unwrapped computed value (lengths, dims, field values, pixel values) | spectra `scene-upload/uploader.rs:886,887,913,1003-1005` (6×), `renderer/lib.rs:96,489,589,601`, `volume_mgr.rs:134-135`, `automaton.rs:325-326`; ochroma `shadow_atlas.rs:196`, `splat_buffer_pool.rs:305-307`, `material_graph.rs:205,215`, `gizmos.rs:710`, `vfx_editor.rs:216,258`, `import_pipeline.rs:307,389`, `forge_ecs.rs:196,240,261`, `materials.rs:133-136`, `tests/spatial_ui_test.rs:9,54,66`, `render_showcase.rs:380` | CLAUDE.md forbids these explicitly; they pass against empty stubs and catch no regressions — 42 sites total | M | needs care (must know the real expected value per site) |
| D2 | Tighten 1-2 borderline cook asserts (e.g. `game/mod.rs:2149` → assert employment sector/job id, not mere `is_some()`) | cook `src/game/mod.rs:2149` | Borderline but cheap to make a real outcome check | S | needs care |

### Theme E — Structure / Decomposition (god files & god objects)

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| E1 | Domain-carve the cook into `cook/{recipes,atomize,plot,flora,materials,surfaces,sdf,details,gates}.rs` per the fn→module map; keep `main`/`run`/`CookConfig`/cache/thumbs in the bin (target ≤ ~600 LOC) | cook `src/bin/game_asset_cook.rs` (5,333 src) | The explicitly-noted backlog item; ~130 free fns across 6+ domains in one flat file | L | needs care (large move; cook output golden re-verifies) |
| E2 | Break up the `EditorShell` god-object (50 methods, 1 impl block) into `shell/{planting,world_io,undo,chrome}.rs` as `impl EditorShell` blocks | ochroma `vox_app/src/shell/mod.rs:465-1993` | 6-7 distinct responsibilities welded to one type; sub-modules already exist as carve targets | M | needs care |
| E3 | Split `impl CivitasGame` (75 methods, ~1,550 LOC) into `game/{zoning,terraform,placement,economy,history,tick}.rs`; break the 229-LOC `tick` into per-system steps; extend existing `game/save.rs` | cook `src/game/mod.rs:363-1914` | One impl block doing 6 subsystems; `tick` is a 229-LOC method | M-L | needs care |
| E4 | Carve `renderer.rs` into `frame_loop`/`pass_orchestration`/`resource_binding` modules | spectra `spectra-renderer/src/renderer.rs` (2,836 src, ~0 tests) | Largest test-less god-file in spectra — genuine src complexity | L | needs care |
| E5 | Pull procedural content-gen out of the renderer: `render_gpu/decoration.rs` (HSV jitter, doors/chimneys, flora), `material_color.rs`, `splat_build.rs`, `scene_renderer.rs` | cook `src/render_gpu/mod.rs` (4,097 src) | Content generation leaking into the renderer; target mod.rs ≤ ~900 LOC | M | needs care |
| E6 | Move the 10 `shot_*`/`nyc_shot`/fixture fns (~1,080 LOC) to a separate `bin/shots.rs`; move `WgpuSurfacePresenter` to reusable `src/present/`; split the 1,380-LOC `impl GameView` | cook `src/bin/play.rs` (4,195 src) | A CI/screenshot harness embedded in the interactive binary; target play.rs ≤ ~1,500 LOC | M-L | needs care |
| E7 | Extract the ~290-LOC generalized-winding-number topology oracle into `forge-mesh/src/winding.rs`; move `generate_asset` → `asset.rs`, material-zoning fns → `material.rs` | forge `crates/building/src/lib.rs:415-707,349-413,708-931` | Pure mesh math + asset/material concerns wrongly housed in the building crate; existing homes already exist; golden hash gate re-verifies byte-identity. Drops lib.rs 2,372 → ~340 (with C7) | M | mechanical-safe (golden gate guards it) |
| E8 | Move logic out of `lib.rs` into submodules (lib.rs should re-export) | spectra `spectra-accel/src/lib.rs` (1,525), `spectra-bridge/src/lib.rs` (1,115) | `lib.rs` doing the work — opposite of the well-factored crates | M | needs care |
| E9 | Diff the three large bins for shared window/camera/frame-loop setup; lift into `vox_app` lib modules so bins are thin entry points | ochroma `engine_runner.rs` (3,607), `walking_sim.rs` (3,145), `scale_trial.rs` (1,339) | Likely duplicated setup, zero test coverage | L | needs care |
| E10 | Extract slang SDF helpers (L389-800) and the IOR-stack buffer block (L263-282, dedup `g_ray_`/`g_next_ray_`) into includes; target < 2,500 LOC | spectra `slang/megakernel.slang` (3,800 LOC, 2 entrypoints) | Copy-paste IOR-stack-as-8-flat-buffers bloat; slang include plumbing is the risk | M | needs care |

### Theme F — Dead Code & Coverage Gaps

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| F1 | Delete the dead `Lcg` struct + impl + comment block (clears the 2 real vox_app warnings) | ochroma `vox_app/src/shell/forge_native.rs:111-124` | Confirmed never-constructed by compiler | XS | mechanical-safe |
| F2 | Wire `normal_to_rotation_i16` into the splat-orientation path or delete it + its `#[allow(dead_code)]` | ochroma `vox_data/src/proc_gs_advanced.rs:465` | A complete quaternion helper nothing calls | S | needs care |
| F3 | Audit the `#[allow(dead_code)]` suppressions (ochroma 24: `atom_instances.rs` ×7, `instanced_select_gpu.rs` ×3, `demo_asset.rs` ×5+; spectra 43: `spectrax.rs:65,73`, `vulkan_context.rs:229,341,361,364,368`, `stbvh.rs:110,131`) — wire in or delete | ochroma + spectra | Hidden debt masking unused API surface | M | needs care |
| F4 | Implement the bench harness or remove the stub binary from the manifest | spectra `spectra-bin/src/bench.rs:2` (`eprintln!("…not yet implemented")`) | A stub binary that ships | S | mechanical-safe (if delete) |
| F5 | Add U-shape and T-shape seal tests (`u_shaped_underside_*`, `t_shaped_*`) mirroring the `tri_area_and_normal_y` pattern | forge `crates/building/src/seal.rs` | Module docstring promises 4 watertight shapes; only Rect+L are verified — U/T seals are unverified-correctness gaps | S | mechanical-safe |
| F6 | Add SDF test coverage (SDF is a stated first-class pillar; 2 tests for 582 LOC is the thinnest ratio in the slice) | ochroma `vox_core/src/sdf.rs` | First-class pillar under-tested | S-M | needs care |
| F7 | Verify `resolve(&mut self,_w,_h){}` is an intentional no-op vs an unfinished stub; implement or document | ochroma `vox_ui/src/layout.rs:91` | Public layout entrypoint with empty body (not a trait default) | S | needs care |
| F8 | Empty test body asserts nothing — add a real feature/compile assertion or delete | spectra `spectra-renderer/tests/feature_gates.rs:12` | A no-op test that gives false confidence | S | mechanical-safe |

### Theme G — Error-Type Adoption

| # | WHAT | WHERE | WHY | EFFORT | Safety |
|---|------|-------|-----|--------|--------|
| G1 | Adopt `vox_core::error::EngineError` (or add a `DataError` variant) instead of `Result<_, String>`; wire `From<io::Error>` | ochroma `vox_data/src/import_pipeline.rs:60,89,146,224` + 27 `Result<_,String>` sites | A proper `EngineError` enum exists but is used by only 2 files — the good error type is abandoned while 27 sites use stringly-typed errors | M | needs care |

---

## 5. The Highest-Leverage 10 (do these first)

Ranked by (risk-reduction + iteration-speed) / effort. The first three unblock builds and close hard-rule breaches; the next four are cheap mechanical wins that shrink the worst files; the last three are the high-value structural carves.

1. **A1 — Fix spectra default features (S, mechanical).** One `Cargo.toml` line unblocks `cargo build` for every non-NVIDIA contributor. Highest leverage per minute spent.
2. **B1 — Remove city taxonomy from `vox_data` (L, needs care).** The only CLAUDE.md hard rule broken in shipped code; left unfixed it keeps leaking and corrupts the engine/game boundary the whole architecture rests on.
3. **A3 — Make spectra feature flags real or delete them (M, needs care).** Lying flags actively mislead anyone trying to slim a build; either gate the deps or stop pretending.
4. **C1 — Extract `shell/mod.rs` test mod (S-M, mechanical).** Single biggest file-shrink: 4,889 → ~2,477 with zero semantic risk.
5. **C2 — Extract `game/mod.rs` test mod (S, mechanical).** Halves the file (3,758 → ~1,913) before any risky structural carve; prerequisite that de-risks E3.
6. **E7 — Carve forge `building/lib.rs` into existing homes (M, mechanical — golden gate guards it).** Drops the crate's only monster 2,372 → ~340, and the GWN oracle becomes reusable by terrain/water sealing. Low risk because the golden hash gate re-verifies byte-identity.
7. **C5/C6 — Extract spectra `cudarc_backend.rs` + `renderer/lib.rs` test mods (M+S, mechanical).** Removes a 95%-test file and ~730 test LOC from the largest GPU backend file.
8. **D1 — Fix the 42 weak `is_some()`/`is_ok()` asserts (M, needs care).** These tests pass against empty stubs and catch no regressions — fixing them is pure risk-reduction and is the most widespread convention violation.
9. **E1 — Domain-carve the cook `game_asset_cook.rs` (L, needs care).** 5,333 src LOC across 6+ domains in one flat file; the single biggest structural-debt item in the game layer; cook output goldens re-verify correctness.
10. **F1 + F4 + F8 — Quick dead-code deletes (XS-S, mechanical).** Delete `Lcg`, the `bench.rs` stub, and the empty `feature_gates.rs` test. Clears the only 2 real ochroma warnings and removes false-confidence scaffolding in minutes.

**Sequencing note:** items 4, 5, 7 (test extractions) should land *before* the corresponding structural carves (E2, E3, E1) — extracting tests first halves the files and makes the semantic restructuring far safer to review. Script the `#[path]` extraction once (the `splat_backend_tests/` pattern) and the entire Theme C backlog becomes near-free.
