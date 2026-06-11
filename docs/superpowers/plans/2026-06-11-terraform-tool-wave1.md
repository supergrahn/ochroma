# Terraform Tool — Wave 1 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **EXECUTION GATE (sequenced honestly):** Tasks 3–6 must not start until **virtualized M4** (`docs/superpowers/plans/2026-06-11-virtualized-m4-game-wiring.md`) has executed and its Done-When prints are demonstrated (`[m4] frame: … path=gpu-resident`, `[scene] renderer constructed 1x`, the `--shot-m4` PNGs). Reasons: the terrain layer rides M4's residual-splat upload path; both waves rewrite `render_gpu/mod.rs` + `bin/play.rs` ownership; pre-M4 a per-stroke scene refresh reconstructs the renderer and makes continuous-apply unusable. **Tasks 1–2 are pure sim (kernel, action, replay/undo) and may run before M4 lands** if schedule demands. Also: a checkpoint-commit agent landed/is landing the current state of both repos — verify `git -C ~/Ochroma/projects/civitas_care status --porcelain` is clean before Task 1. Run in the main repos, never an isolated worktree (the abs-path gotcha).

**Goal:** A CS2-style terraforming brush (Raise/Lower/Level/Smooth) in Civitas Care: continuous-apply drag bundled into ONE replayable `AuthoredAction::Terraform`, kr-per-m³ cost through funds, BLOCK-under-the-city guard with the one-source verdict chip, water-mask → childcare-coverage invalidation, a live terrain splat layer, the `--shot-terraform` harness, and a W3 walkthrough.
**Done When:** `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play -- --shot-terraform terraform_shots` exits 0 printing `[terraform] raise stroke: <S> stamps · <V> m³ · <K> kr · funds <F0> -> <F1>` (S in 5..=200, V > 500, F1 = F0 − K), `[terraform] flood stroke: water cells flipped <W> · childcare field dirty: true` (W > 0), `[terraform] undo: heights bit-exact to pre-stroke: true · funds restored: true`, and `[terraform] after differs from before on <N> px` (N > 20000) — and writes `terraform_shots/terraform_{before,after,undo}.png` where a human sees a hill appear and disappear. Windowed: arm **Terrain → Raise** on Founders' Vale, drag → visible ground rises, release → status bar prints `Terraformed (Raise) — <V> m³ · <K> kr`, Ctrl+Z prints `Undid: Terraformed (Raise) — <V> m³ · <K> kr` with byte-identical numbers.
**Architecture:** The game-owned heightfield path (engine voxel terrain NOT adopted): the `MapEditor` sculpt kernel is factored into `src/map/sculpt.rs` and shared with a new `MapTerrain::apply_stamp` (kernel edit + regional `CellClass` re-derive + version bump). `CivitasGame` owns the stroke state machine (`begin_stroke`/`stroke_stamp`/`end_stroke`) — spacing-gated stamps mutate terrain continuously, mouse-up logs ONE `AuthoredAction::Terraform{mode, radius, amount, target, stamps, volume_m3, cost_kr}` (SAVE_VERSION **stays 5**, the AddRoad additive-variant precedent). Undo correctness comes from the new `base_terrain` pristine snapshot replacing the live-terrain carry-over in `rebuilt_with_log`.
**Design Document:** `docs/superpowers/specs/2026-06-11-terraform-tool-design.md`
**Tech Stack:** Rust (edition 2024), game repo `~/Ochroma/projects/civitas_care` (branch `master`, own repo), engine `~/src/ochroma` untouched. winit 0.30, tiny-skia, serde_json.
**Build:** `cd ~/Ochroma/projects/civitas_care && cargo build --release` / `cargo test --release` / bin tests `cargo test --release --bin play`.

---

## IMPORTANT NOTES

- **Repos / commits:** ALL code lands in the GAME repo `~/Ochroma/projects/civitas_care` (its own git repo, branch `master`). Engine crates stay untouched. Each task ends with a real commit, message per house convention ending with the footer line `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Verified GAME signatures (code-checked 2026-06-11; code wins over docs):**
  - `HeightBrush { pub radius: f32, pub strength: f32 }`, `HeightBrush::new(radius, strength)`, private `falloff(t)` = smoothstep `s*s*(3-2s)` with `s = 1-t`; `BrushOp { Raise(f32), Lower(f32), Flatten(f32), Smooth }`; `MapEditor::sculpt(&mut self, wx: f32, wz: f32, op: BrushOp)` — all `src/map/editor.rs:22-142`. The kernel iterates cells row-major in the brush AABB, distance-tests against `radius`, weight `falloff(dist/r) * brush.strength`; `Smooth` snapshots `hm.data` first. **This behavior must be preserved bit-for-bit when factored out** (existing editor tests `sculpting_raises_terrain_…` must pass unmodified).
  - `derive_buildable(hm: &Heightmap, water: &WaterMask, max_buildable_slope_deg: f32) -> CellGrid<CellClass>` (`src/map/derive.rs:19`); `SLOPE_BLOCKED_MAX: f32 = 20.0`; a cell is `Water` if `hm.sample(centre) <= water.sea_level` OR centre inside a `WaterBody.outline` polygon, else `TooSteep` over the slope cap, else `Buildable`.
  - `Heightmap { pub width: usize, pub height: usize, pub data: Vec<f32>, pub cell_size: f32, pub origin: [f32; 2] }`, `sample(wx, wz) -> f32`, `slope_at(wx, wz) -> f32`, `from_data(w, h, data, cell_size)` (**re-zeroes origin** — the R-Origin invariant), `bounds()` (engine `vox_terrain::heightmap`, read-only use).
  - `MapTerrain` fields are **private** (`cell_size, origin, buildable, heights, flat`); accessors `buildability/is_buildable/is_water/sample_y/is_flat/line_crosses_water`; hand-written `Clone` rebuilds `heights` via `from_data` (`src/map/terrain.rs:115`). `WaterMask { pub bodies: Vec<WaterBody>, pub sea_level: f32 }`, `WaterMask::none()` (`src/map/layers.rs`).
  - `CivitasGame.log: Vec<(u64, AuthoredAction)>`; the logging idiom is `self.log.push((self.tick_index, …)); self.undone.clear();` then mutate (`place_care`, `src/game/mod.rs:321`). Funds: `self.sim.budget.funds -= cost` (f64, may go negative — house behavior, do NOT add a refusal).
  - **The undo landmine this plan exists to fix:** `rebuilt_with_log` (`src/game/mod.rs:1378`) currently seeds `fresh.terrain = self.terrain.clone()` — the LIVE terrain. Task 2 replaces it with `base_terrain`. Touch nothing else in that function (it deliberately does NOT use `from_save_on`: catalog/style-pack divergence).
  - `GameSave::from_json` rejects malformed `action_ticks` and `version > SAVE_VERSION`; **`SAVE_VERSION` stays 5** (additive variant, AddRoad precedent — document the variant in the v5 doc comment; do NOT bump).
  - `GameRegistry::dispatch(&self, view: &mut ViewState, game: &mut CivitasGame, sdf: &SpatialFieldRegistry, id: &str) -> Result<String, Vec<String>>` (`src/command/mod.rs:472`); receipts push to `view.receipts` + set `view.status_receipt`. `describe_action(game: &CivitasGame, action: &AuthoredAction) -> String` (`:728`) — the undo receipt is `format!("Undid: {}", describe_action(...))`, so commit and undo MUST byte-match. `thousands(v: f64) -> String` (`:757`).
  - `tool_command_id`/`tool_title` (`src/command/mod.rs:773`) are **exhaustive matches** over `BuildTool` — adding `Terraform(TerraformMode)` compile-breaks them (intended, the I3 lock); the registry test pins `assert_eq!(count, 27, …)` (`:945`) → becomes **31** (11+5+8+2+4+1).
  - `build_categories() -> Vec<(&'static str, Vec<BuildTool>)>` (`src/ui/build_tools.rs:45`) — insert `("Terrain", vec![Terraform(Raise), Terraform(Lower), Terraform(Level), Terraform(Smooth)])` at index 4; Inspect stays appended LAST (shifts to index 5).
  - Hardcoded 4-build-tab sites that become 5: `hud.rs:861` (`toolbar_layout(w, 4)`), `hud.rs:1215` (`panel_anchor_x`: `toolbar_layout(w, 4)` + `if cat < 4`), `hud.rs:1585`; `play.rs` `&self.categories[..4]` ("the FOUR build categories only" comment). Audit with `grep -n "toolbar_layout(w, 4)\|\[\.\.4\]"`.
  - `hud::draw_world_circle(px: &mut [[u8;4]], w: u32, h: u32, view: glam::Mat4, proj: glam::Mat4, c: [f32;2], r: f32, rgb: (u8,u8,u8), a: f32)` (`hud.rs:902`) — the care-reach ring; the play binary already calls it (`play.rs:630`) with `hud::ACCENT`.
  - `ToolPanelOptionRow { pub label: &'a str, pub options: &'a [(String, bool)] }` (`hud.rs:1198`); option rows are built in `play.rs` ~line 700 (the `"Show"` row site); hit-tested via `tool_panel_option_at`.
  - `tool_preview(game, sdf, tool, draft, cursor) -> ToolPreview { caption, valid, parcels, cost }` and `plop_blocked(game, sdf, pos) -> Option<String>` (`src/ui/preview.rs`) — the one-source verdict pattern to mirror EXACTLY for `terraform_blocked`.
  - Coverage: `childcare_field_dirty: bool` (private, `src/game/mod.rs:240`), set by infant-care plops; consumed lazily by `rebuild_childcare_field_if_dirty` (`:713`) inside `tick`. The mask is water-only — only Water-class flips dirty it.
  - Land value reads **no terrain input** (`src/land_value/mod.rs` formula) — add NO invalidation hook.
  - Bundled real maps: `founders_vale` / `tidewater_flats` / `sketchpad`, 256² cells @ 4 m (`src/map/bundled.rs`); `Map::load_dir(dir, id)`; `MenuApp::start_game(scenario, map_id)`; `GameView::new(game, w, h, style_packs, asset_catalog)`. The Demo flat world REFUSES terraforming — every harness/walkthrough fixture must start on a bundled map.
  - Walkthrough: `WalkthroughStep { say, command: Option<String>, check: Check }`, `script(milestone: u8)`, `print(milestone)`, `run(milestone)`, `--walkthrough N` (`src/walkthrough/mod.rs`, `play.rs:2546`).
  - Harness precedents: `--shot-landvalue` (`play.rs:2009`, prints real measured lines, exits Err(String) on failed gates), `save_rgba`, `next_coverage_shot_path`. Mirror its structure for `--shot-terraform`.
  - **Determinism rules (the replay contract):** stamps spacing-gated at `radius * STAMP_SPACING_FRAC` (0.25) — never dt/time-gated; kernel is pure f32, fixed row-major order, no RNG; `amount`/`target` resolved at stroke time and stored IN the action; replay `debug_assert_eq!` on `volume_m3.to_bits()`. **Do not gate undo PNGs on pixel identity** (the known radix-raster nondeterminism) — gate on height-buffer bits.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.
- Tests live in in-module `#[cfg(test)]` blocks (house style), GPU-needing tests print a skip line and stay green on a GPU-less box (the `coverage_field` pattern).

---

## File Map

All paths relative to `~/Ochroma/projects/civitas_care`.

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/map/sculpt.rs` | the shared sculpt kernel `sculpt_heightmap` + `StampDelta` (moved from `MapEditor::sculpt`, behavior-identical) |
| Modify | `src/map/editor.rs` | `MapEditor::sculpt` delegates to the kernel; `BrushOp`/`HeightBrush` re-exported or relocated |
| Modify | `src/map/mod.rs` | `pub mod sculpt;` |
| Modify | `src/map/derive.rs` | `rederive_region(...) -> u32` (water-flip count) |
| Modify | `src/map/terrain.rs` | `MapTerrain` gains `water: WaterMask` + `version: u64`; `apply_stamp`, `version()`, `heights()`; Clone/from_map/flat_default updated |
| Modify | `src/game/save.rs` | `TerraformMode` + `AuthoredAction::Terraform` (additive, SAVE_VERSION stays 5; doc comment updated) |
| Modify | `src/game/mod.rs` | `base_terrain` + `stroke` fields; `begin_stroke`/`stroke_stamp`/`end_stroke`/`stroke_active`; `apply_action` Terraform arm; `rebuilt_with_log` base_terrain fix; consts `TERRAFORM_KR_PER_M3`/`STAMP_SPACING_FRAC`/`STAMP_AMOUNT_M` |
| Modify | `src/command/mod.rs` | `describe_action` Terraform arm; `UiAction::SetBrushRadius/SetBrushStrength`; `terrain.size.*`/`terrain.strength.*` commands; `ViewState.brush_radius/brush_strength`; exhaustive-match arms; tool-count test 27→31 |
| Modify | `src/ui/build_tools.rs` | `BuildTool::Terraform(TerraformMode)`; "Terrain" category at index 4 (Inspect shifts to 5); labels |
| Modify | `src/ui/preview.rs` | `terraform_blocked` one-source verdict; `tool_preview` Terraform arm |
| Modify | `src/render_gpu/hud.rs` | 5 build tabs: `toolbar_layout(w, 4)` sites + `panel_anchor_x` `cat < 4` → 5 |
| Modify | `src/render_gpu/mod.rs` | `terrain_splats(terrain, spc)`; inclusion in the M4 residual cook when `!terrain.is_flat()` |
| Modify | `src/bin/play.rs` | drag wiring (begin/stamp/end), brush ring, Size/Strength option rows, `[..4]`→`[..5]`, terrain-version watch, `--shot-terraform` |
| Modify | `src/walkthrough/mod.rs` | `script(3)` W3 terraform steps + `run(3)` headless twin |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Shared kernel + regional re-derive | stamp on founders_vale: printed `stamp: <cells> cells · <m³> m³ · water flips <w>` with cells > 50, m³ > 10; shore-lowering flips `is_water` false→true | `assert!(result.is_ok())` |
| One action per stroke, replay/undo exact | 12-stamp drag → `log.len()+1`, `stamps.len()==12`; save→`from_save_on`→ per-cell `to_bits()` equal; undo → bits equal to pre-stroke snapshot + funds restored | comparing a single sampled height |
| Cost through funds | funds delta == `volume_m3 as f64 * 10.0` exactly; receipt `Terraformed (Raise) — <V> m³ · <K> kr` | asserting funds merely changed |
| Commands/panel | `terrain.size.large` → receipt `Brush: 80 m` AND `view.brush_radius == 80.0`; registry covers 31 tools 1:1 | checking the command exists |
| One-source block verdict | preview caption string == `stroke_stamp` Blocked string over a placed lot; heights unchanged; nothing logged | two independently-written verdicts |
| Terrain layer | `terrain_splats` > 50_000 splats; stamped cell's splat y moves by the stamp Δ; water-cell splat darker than grass | `assert!(!splats.is_empty())` |
| Harness + walkthrough | the §Done-When prints with real numbers; `run(3)` prints `[x]` per asserted step | a harness that only writes PNGs |

---

## Task 1: The shared sculpt kernel + `MapTerrain::apply_stamp` (regional re-derive, version, water flips)

**Files:**
- Create: `src/map/sculpt.rs`
- Modify: `src/map/editor.rs`, `src/map/mod.rs`, `src/map/derive.rs`, `src/map/terrain.rs`

**Acceptance:** `cargo test --release map_terrain_stamp -- --nocapture` → prints `stamp: <cells> cells · <m3> m³ · water flips <w>` with cells > 50, m3 > 10.0, and the flood case prints `water flips <w>` with w > 0; `cargo test --release map::editor` (the pre-existing editor tests) green UNMODIFIED.

**Wiring requirement:** `MapEditor::sculpt` in `src/map/editor.rs` must delegate to `sculpt_heightmap` (its body becomes the delegation — duplicate kernels = task failure); `MapTerrain::apply_stamp` in `src/map/terrain.rs` must call `sculpt_heightmap` + `rederive_region` and bump `version`. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing tests** — in `src/map/terrain.rs` `#[cfg(test)]`:

```rust
#[test]
fn map_terrain_stamp_raises_ground_and_floods_the_shore() {
    // founders_vale: a real bundled map with sea-level water.
    let map = crate::map::bundled::build_founders_vale();
    let mut t = MapTerrain::from_map(&map);
    let v0 = t.version();

    // A dry buildable spot (probe the grid for one — no hardcoded luck).
    let (wx, wz) = find_buildable_spot(&t, &map);
    let y0 = t.sample_y(wx, wz);
    let brush = crate::map::sculpt::HeightBrush::new(40.0, 1.0);
    let stats = t
        .apply_stamp(&brush, wx, wz, crate::map::sculpt::BrushOp::Raise(1.5))
        .expect("stamp applies on a real map");
    let y1 = t.sample_y(wx, wz);
    println!("stamp: {} cells · {:.1} m³ · water flips {}", stats.cells_changed(), stats.volume_m3(), stats.water_flips());
    assert!(stats.cells_changed() > 50, "a 40 m brush on 4 m cells touches >50 cells");
    assert!(stats.volume_m3() > 10.0, "real volume moved, got {}", stats.volume_m3());
    assert!(y1 > y0 + 0.5, "centre must rise ~amount·falloff(0): {y0} -> {y1}");
    assert_eq!(t.version(), v0 + 1, "version bumps per stamp");

    // Flood: hammer a shore-adjacent dry cell below sea level → Water flips.
    let (sx, sz) = find_shore_dry_spot(&t, &map);
    assert!(!t.is_water(sx, sz));
    let mut flips = 0u32;
    for _ in 0..40 {
        flips += t.apply_stamp(&brush, sx, sz, crate::map::sculpt::BrushOp::Lower(1.5)).unwrap().water_flips();
    }
    println!("flood: water flips {flips}");
    assert!(flips > 0, "lowering below sea level must flip cells to Water");
    assert!(t.is_water(sx, sz), "the hammered cell is now Water");
}

#[test]
fn flat_world_refuses_stamps() {
    let mut t = MapTerrain::flat_default();
    let brush = crate::map::sculpt::HeightBrush::new(40.0, 1.0);
    let err = t.apply_stamp(&brush, 10.0, 10.0, crate::map::sculpt::BrushOp::Raise(1.5)).unwrap_err();
    println!("flat refusal: {err}");
    assert!(err.contains("real map"));
}
```

(`find_buildable_spot`/`find_shore_dry_spot` are small test helpers scanning the grid — real probing, no magic coordinates.)

- [ ] **Step 2: Run to verify failure** — `cargo test --release map_terrain_stamp 2>&1 | tail -5` → FAIL: `apply_stamp`/`sculpt` module not found.
- [ ] **Step 3: Implement.** (a) Move the kernel loop of `MapEditor::sculpt` verbatim into `src/map/sculpt.rs::sculpt_heightmap(hm, &HeightBrush, wx, wz, op) -> StampDelta` — same row-major iteration, same falloff, same Smooth snapshot; accumulate `cells_changed`, `volume_m3 += |after-before| * cell_size²`, and the inclusive cell AABB. Relocate (or re-export) `BrushOp`/`HeightBrush` so `MapEditor` keeps compiling. (b) `MapEditor::sculpt` body becomes `let _ = sculpt_heightmap(&mut self.map.heightmap, &self.brush, wx, wz, op); self.dirty = true;`. (c) `src/map/derive.rs::rederive_region(hm, water, grid, max_slope, aabb)` — the `derive_buildable` per-cell classification restricted to the AABB inflated by 1 cell; count and return Water-class crossings. (d) `MapTerrain`: add `water: WaterMask` + `version: u64` (update `flat_default` → `WaterMask::none()`, `from_map` → `map.layers.water.clone()`, the manual `Clone`); `apply_stamp` = flat-world `Err("Terraforming needs a real map")`, else kernel → `rederive_region` → `version += 1` → `StampStats`. Add `version()`, `heights()` accessors.
- [ ] **Step 4: Wire at exact callsites** — delegation inside `MapEditor::sculpt` (editor.rs:97) and the kernel call inside `MapTerrain::apply_stamp` (terrain.rs). The map-system tests prove the editor wiring; the new tests prove the terrain wiring.
- [ ] **Step 5: Run — verify non-trivial output** — `cargo test --release map_terrain_stamp -- --nocapture` PASS with the printed real cells/m³/flips; `cargo test --release map::` all green (editor tests untouched).
- [ ] **Step 6: Commit**

```bash
cd ~/Ochroma/projects/civitas_care
git add src/map/sculpt.rs src/map/editor.rs src/map/mod.rs src/map/derive.rs src/map/terrain.rs
git commit -m "feat(terraform): shared sculpt kernel + MapTerrain::apply_stamp with regional re-derive, version counter and water-flip stats

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: The stroke state machine + `AuthoredAction::Terraform` + replay-exact undo (the base_terrain fix)

**Files:**
- Modify: `src/game/save.rs`, `src/game/mod.rs`, `src/command/mod.rs` (the `describe_action` arm only)

**Acceptance:** `cargo test --release terraform_stroke -- --nocapture` → prints `stroke: 12 stamps · <V> m³ · <K> kr` (V > 100, K = V×10), `replay heights bit-exact: true`, `undo heights bit-exact: true · funds restored: true`, and `save version: 5`.

**Wiring requirement:** `begin_stroke`/`stroke_stamp`/`end_stroke` on `CivitasGame` in `src/game/mod.rs`; `end_stroke` MUST push the log entry via the house idiom and charge `self.sim.budget.funds`; `apply_action` MUST gain the `Terraform` arm; `rebuilt_with_log` MUST seed `fresh.terrain = self.base_terrain.clone()` (and carry `base_terrain`). Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — in `src/game/mod.rs` `#[cfg(test)]`:

```rust
#[test]
fn terraform_stroke_bundles_replays_and_undoes_exactly() {
    use crate::game::save::TerraformMode;
    let map = crate::map::bundled::build_founders_vale();
    let mut game = CivitasGame::new_on_map(&map);
    let funds0 = game.sim.budget.funds;
    let pre_bits: Vec<u32> = game.terrain.heights().data.iter().map(|h| h.to_bits()).collect();

    // A 12-stamp diagonal drag on dry ground (stamps spaced ≥ radius*0.25).
    game.begin_stroke(TerraformMode::Raise, 40.0, 1.0).unwrap();
    let mut applied = 0;
    for i in 0..12 {
        let (x, z) = (400.0 + i as f32 * 12.0, 400.0 + i as f32 * 12.0);
        if matches!(game.stroke_stamp(x, z), StampOutcome::Applied { .. }) { applied += 1; }
    }
    let receipt = game.end_stroke().expect("a real stroke logs");
    println!("{receipt}");
    assert_eq!(game.log.len(), 1, "ONE action for the whole drag");
    let (_, action) = game.log.last().unwrap().clone();
    let crate::game::save::AuthoredAction::Terraform { stamps, volume_m3, cost_kr, .. } = action.clone()
        else { panic!("a Terraform action") };
    println!("stroke: {} stamps · {volume_m3:.1} m³ · {cost_kr:.0} kr", stamps.len());
    assert_eq!(stamps.len(), applied);
    assert!(volume_m3 > 100.0);
    assert!((cost_kr - volume_m3 as f64 * TERRAFORM_KR_PER_M3).abs() < 1e-6);
    assert!((funds0 - game.sim.budget.funds - cost_kr).abs() < 1e-6, "charged through funds");
    assert!(receipt.starts_with("Terraformed (Raise) — "), "got {receipt}");

    // Replay-exact: save → from_save_on → heights bit-compare.
    let save = game.to_save();
    assert_eq!(save.version, 5, "SAVE_VERSION must NOT bump (AddRoad precedent)");
    let loaded = CivitasGame::from_save_on(&save, &map);
    let same = loaded.terrain.heights().data.iter().zip(game.terrain.heights().data.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("replay heights bit-exact: {same}");
    assert!(same);
    assert_eq!(loaded.sim.budget.funds, game.sim.budget.funds);

    // Undo: truncate-and-replay must restore PRISTINE heights (base_terrain fix).
    let undone = game.undo_last_action().expect("undoable");
    assert_eq!(format!("Undid: {}", crate::command::describe_action(&game, &undone)),
               format!("Undid: {receipt}"), "commit and undo receipts byte-match");
    let restored = game.terrain.heights().data.iter().map(|h| h.to_bits()).eq(pre_bits.iter().copied());
    println!("undo heights bit-exact: {restored} · funds restored: {}", game.sim.budget.funds == funds0);
    assert!(restored, "the live-terrain carry-over landmine: rebuilt_with_log must seed base_terrain");
    assert_eq!(game.sim.budget.funds, funds0);
    println!("save version: {}", save.version);
}

#[test]
fn flat_world_stroke_refused_and_logs_nothing() {
    let mut game = CivitasGame::new_small();
    let err = game.begin_stroke(crate::game::save::TerraformMode::Raise, 40.0, 1.0).unwrap_err();
    println!("flat: {err}");
    assert!(err.contains("real map"));
    assert!(game.log.is_empty());
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --release terraform_stroke 2>&1 | tail -5` → FAIL: `begin_stroke` not found.
- [ ] **Step 3: Implement.** (a) `save.rs`: `TerraformMode` enum + the `Terraform` variant exactly as the design §5; extend the v5 doc comment ("v5 also gains the additive Terraform action — older binaries reject such saves loudly, the AddRoad rule"). (b) `game/mod.rs`: consts; `base_terrain` + `stroke` fields (set `base_terrain` in `new_small`; `new_on_map`/`bind_map` set BOTH `terrain` and `base_terrain`); `ActiveStroke`; `begin_stroke` (flat refusal; resolved `amount = STAMP_AMOUNT_M * strength`); `stroke_stamp` (spacing gate vs `last_stamp`; `terraform_blocked` gate — until Task 4 lands, call a private placeholder-named REAL implementation `terraform_blocked_inner` here and move it to preview.rs in Task 4, never a stub; Level samples `target` at the first applied stamp; map mode→`BrushOp`; `terrain.apply_stamp`; accumulate volume/flips/stamps); `end_stroke` (empty → `None`; else compute `cost_kr`, push `(tick_index, Terraform{..})`, `undone.clear()`, `funds -=`, `childcare_field_dirty |= water_flips > 0`, return the receipt string built via `describe_action`-identical formatting); `apply_action` Terraform arm (re-apply stamps through `apply_stamp` with the stored params, recompute volume, `debug_assert_eq!(recomputed.to_bits(), volume_m3.to_bits())`, charge stored `cost_kr`, set coverage dirty on flips); `rebuilt_with_log`: replace `fresh.terrain = self.terrain.clone()` with `fresh.terrain = self.base_terrain.clone(); fresh.base_terrain = self.base_terrain.clone();`. (c) `command/mod.rs`: `describe_action` arm → `format!("Terraformed ({}) — {} m³ · {} kr", mode_label(mode), thousands(volume_m3 as f64), thousands(cost_kr))`.
- [ ] **Step 4: Wire at exact callsites** — `apply_action` (game/mod.rs:1340) gains the arm; `rebuilt_with_log` (game/mod.rs:1378-1386) seeds `base_terrain`; `end_stroke` charges at the same `self.sim.budget.funds -=` pattern as `place_care` (mod.rs:327).
- [ ] **Step 5: Run — verify non-trivial output** — `cargo test --release terraform_stroke -- --nocapture` PASS with real m³/kr; full `cargo test --release` green (notably every existing save/undo/replay test — the base_terrain change must be invisible to non-terraform logs).
- [ ] **Step 6: Commit**

```bash
git add src/game/save.rs src/game/mod.rs src/command/mod.rs
git commit -m "feat(terraform): stroke state machine — one AuthoredAction per drag, kr/m³ funds charge, base_terrain replay/undo exactness (SAVE_VERSION stays 5)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: The Terrain build tab + brush commands + option rows (GATED on virtualized M4)

**Files:**
- Modify: `src/ui/build_tools.rs`, `src/command/mod.rs`, `src/render_gpu/hud.rs`, `src/bin/play.rs`

**Acceptance:** `cargo test --release registry_covers_every_build_tool brush_commands -- --nocapture` → prints `31/31 tools mapped 1:1`, `Armed: Terrain: Raise`, `Brush: 80 m`, `Brush: Hard`, and asserts `view.brush_radius == 80.0` / `view.brush_strength == 2.0`.

**Wiring requirement:** the "Terrain" category at `build_categories()` index 4 in `src/ui/build_tools.rs`; `terrain.size.*`/`terrain.strength.*` dispatched through `GameRegistry::dispatch` mutating the new `ViewState` fields; `panel_anchor_x` in `src/render_gpu/hud.rs` handles 5 build tabs; `play.rs` builds the Size/Strength option rows when a Terrain tool is armed. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — in `src/command/mod.rs` tests:

```rust
#[test]
fn brush_commands_set_view_state() {
    let mut game = CivitasGame::new_small();
    let mut view = ViewState::default();
    let sdf = SpatialFieldRegistry::new();
    let registry = GameRegistry::standard();
    assert_eq!((view.brush_radius, view.brush_strength), (40.0, 1.0), "boot defaults");
    let arm = registry.dispatch(&mut view, &mut game, &sdf, "build.terrain.raise").unwrap();
    assert_eq!(arm, "Armed: Terrain: Raise");
    let r = registry.dispatch(&mut view, &mut game, &sdf, "terrain.size.large").unwrap();
    let s = registry.dispatch(&mut view, &mut game, &sdf, "terrain.strength.hard").unwrap();
    println!("{arm}"); println!("{r}"); println!("{s}");
    assert_eq!((r.as_str(), view.brush_radius), ("Brush: 80 m", 80.0));
    assert_eq!((s.as_str(), view.brush_strength), ("Brush: Hard", 2.0));
}
```

and update the pinned count: `assert_eq!(count, 31, "the palette is 31 tools (11+5+8+2+4+1)")`.

- [ ] **Step 2: Run to verify failure** — `cargo test --release brush_commands 2>&1 | tail -5` → FAIL: `build.terrain.raise` unknown / `BuildTool::Terraform` not found.
- [ ] **Step 3: Implement.** `build_tools.rs`: `Terraform(TerraformMode)` variant, the index-4 `("Terrain", …)` category, `terraform_label(mode)`, `tile_label` arm (update the appended-LAST comment honestly). `command/mod.rs`: exhaustive-match arms in `tool_command_id` (`build.terrain.<mode>`), `tool_title` (`Terrain: Raise`); `UiAction::SetBrushRadius(f32)`/`SetBrushStrength(f32)`; six `terrain.size.*`/`terrain.strength.*` commands with the `Brush: …` receipts; `ViewState` additive fields + `Default`. `hud.rs`: the three `toolbar_layout(w, 4)` sites + `panel_anchor_x` `cat < 4` → 5 (grep-audit). `play.rs`: `[..4]` → `[..5]`; category hotkey row picks up tab 5 automatically via `build_categories()` length; option rows when `matches!(self.current_tool(), BuildTool::Terraform(_))`: `Size` row `[("20 m", r==20.0), ("40 m", r==40.0), ("80 m", r==80.0)]`, `Strength` row `Soft/Medium/Hard` lit by `brush_strength`; clicking a value button dispatches the matching command id through the registry (the chrome rule: buttons and keys share ids).
- [ ] **Step 4: Wire at exact callsites** — `GameRegistry::standard()` registers all new commands; the play.rs option-row builder at the existing `"Show"` row site (~line 700); the tile click path arms via the registry exactly as the other 27 tools.
- [ ] **Step 5: Run — verify non-trivial output** — `cargo test --release brush_commands registry_covers -- --nocapture` PASS printing `31/31` + the three receipts; `cargo test --release` green (every hud hit-test/layout test).
- [ ] **Step 6: Commit**

```bash
git add src/ui/build_tools.rs src/command/mod.rs src/render_gpu/hud.rs src/bin/play.rs
git commit -m "feat(terraform): Terrain build tab (5th centre tab), brush size/strength commands + option rows — 31-tool registry

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: One-source block verdict + ghost caption + brush ring + drag wiring

**Files:**
- Modify: `src/ui/preview.rs`, `src/game/mod.rs` (move the Task-2 inner verdict out), `src/bin/play.rs`

**Acceptance:** `cargo test --release terraform_preview_blocked_matches_stamp -- --nocapture` → prints the SAME verdict string twice (`preview verdict: Terraforming blocked — under …` / `stamp verdict: <identical>`), `heights unchanged: true`, `nothing logged: true`; and prints the clear-ground caption `Raise · 40 m — drag to sculpt`.

**Wiring requirement:** `terraform_blocked` lives in `src/ui/preview.rs` and is called by BOTH `tool_preview` (the ghost) and `CivitasGame::stroke_stamp` (the refusal) — two call sites, one function. The play binary's mouse handlers drive `begin_stroke`/`stroke_stamp`/`end_stroke` and draw the ring via `hud::draw_world_circle` with `view.brush_radius`. `GameView` calls `end_stroke()` before any non-stamp dispatch, before save and on exit. Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — in `src/ui/preview.rs` tests (mirror `plop_preview_blocked_matches_dispatch`):

```rust
#[test]
fn terraform_preview_blocked_matches_stamp() {
    use crate::game::save::TerraformMode;
    let map = crate::map::bundled::build_founders_vale();
    let mut game = CivitasGame::new_on_map(&map);
    // Author a real district so a placed (vacant) lot exists to block on.
    let outline = [[400.0, 440.0], [460.0, 440.0], [460.0, 480.0], [400.0, 480.0]];
    assert!(game.zone_area(vox_sim::zoning::ZoneType::ResidentialLow, &outline) > 0);
    let lot_center = game.placed_lots[0].lots[0].center;

    let pv = tool_preview(&game, &SpatialFieldRegistry::new(),
        BuildTool::Terraform(TerraformMode::Raise), &[], lot_center);
    assert!(!pv.valid, "the ring over a placed lot must be invalid");
    println!("preview verdict: {}", pv.caption);

    let pre: Vec<u32> = game.terrain.heights().data.iter().map(|h| h.to_bits()).collect();
    game.begin_stroke(TerraformMode::Raise, 40.0, 1.0).unwrap();
    let StampOutcome::Blocked(v) = game.stroke_stamp(lot_center[0], lot_center[1]) else { panic!("blocked") };
    println!("stamp verdict: {v}");
    assert_eq!(pv.caption, v, "ONE verdict source — preview and stamp cannot diverge");
    assert!(game.end_stroke().is_none(), "an all-blocked stroke logs nothing");
    let unchanged = game.terrain.heights().data.iter().map(|h| h.to_bits()).eq(pre.iter().copied());
    println!("heights unchanged: {unchanged} · nothing logged: {}", game.log.len() == 1);
    assert!(unchanged);
    assert_eq!(game.log.len(), 1, "only the zone action");

    // Clear ground: the drag-prompt caption.
    let clear = tool_preview(&game, &SpatialFieldRegistry::new(),
        BuildTool::Terraform(TerraformMode::Raise), &[], [700.0, 700.0]);
    println!("{}", clear.caption);
    assert!(clear.valid);
    assert_eq!(clear.caption, "Raise · 40 m — drag to sculpt");
}
```

(Note: `tool_preview` for Terraform needs the brush radius — pass it via the existing signature's `draft` slot is WRONG; extend `tool_preview` with the armed brush radius read from a new parameter or have the Terraform arm read a `radius` argument added to the call — pick ONE: add `brush_radius: f32` to `tool_preview` and update its ~4 call sites. The test above then passes `40.0`.)

- [ ] **Step 2: Run to verify failure** — `cargo test --release terraform_preview_blocked 2>&1 | tail -5` → FAIL: `terraform_blocked`/`Terraform` preview arm missing.
- [ ] **Step 3: Implement.** `preview.rs`: `terraform_blocked(game, pos, radius) -> Option<String>` — disc-vs-footprint tests over `game.placed_lots` (developed AND vacant: `|dx| < radius + width/2 && |dz| < radius + depth/2` in lot frame), `game.care.buildings` + `game.sim.services.buildings` (disc vs position + 12 m footprint margin), road segments (point-segment distance < radius + `seg.width()/2`); verdicts `Terraforming blocked — under a zoned lot` / `under <care label>` / `under <service label>` / `under a road`. Move Task 2's inner check to call THIS function. `tool_preview` Terraform arm: blocked → `(caption=verdict, valid=false)`; flat world → `Terraforming needs a real map`; else `"{mode} · {radius:.0} m — drag to sculpt"`. `play.rs`: mouse-down with Terraform armed → `begin_stroke(mode, view.brush_radius, view.brush_strength)`; mouse-move while down → `stroke_stamp(cursor_world)`; mouse-up → `end_stroke()` receipt pushed to `view.receipts`/`status_receipt`; `end_stroke` also in the pre-dispatch/save/exit paths; ring drawn each frame at `cursor_world` with `view.brush_radius`, ACCENT when `pv.valid` else red `(220, 80, 80)`.
- [ ] **Step 4: Wire at exact callsites** — `stroke_stamp` (game/mod.rs) calls `crate::ui::preview::terraform_blocked`; the play.rs ring at the existing `draw_world_circle` site (~line 630); drag handlers in the existing mouse match.
- [ ] **Step 5: Run — verify non-trivial output** — `cargo test --release terraform_preview -- --nocapture` PASS printing the identical verdict twice; full suite green.
- [ ] **Step 6: Commit**

```bash
git add src/ui/preview.rs src/game/mod.rs src/bin/play.rs
git commit -m "feat(terraform): one-source block verdict (preview == stamp), brush ring + continuous drag wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Live terrain splat layer + coverage cascade proof

**Files:**
- Modify: `src/render_gpu/mod.rs`, `src/bin/play.rs`, `src/game/mod.rs` (test access only if needed)

**Acceptance:** `cargo test --release terrain_splats_layer coverage_severs_after_flood -- --nocapture` → prints `terrain layer: <N> splats (water <W> · steep <S> · grass <G>)` with N > 50_000 and W > 0; `stamped cell splat y: <y0> -> <y1>` with y1 − y0 > 0.5; and the coverage test prints `pre-flood reach: true → post-flood reach: false` (GPU box; prints a skip line and stays green without an adapter).

**Wiring requirement:** `terrain_splats(terrain, spc)` in `src/render_gpu/mod.rs`, included in the M4 residual-splat cook in `GameView` **only when `!terrain.is_flat()`**, re-cooked when `gv` sees `game.terrain.version()` differ from its cached value (the flat-world screenshot harnesses must be pixel-unaffected). Stubs = **task failure**.

- [ ] **Step 1: Write the failing tests** — `render_gpu/mod.rs` tests:

```rust
#[test]
fn terrain_splats_layer_tracks_heights_and_classes() {
    let map = crate::map::bundled::build_founders_vale();
    let mut t = crate::map::terrain::MapTerrain::from_map(&map);
    let splats = terrain_splats(&t, 1);
    let n = splats.len();
    assert!(n > 50_000, "256² map at spc=1 ⇒ ~65k cells, got {n}");
    // Water splats darker than grass (class tint is real, not uniform).
    // …count per-class via terrain.is_water at each splat's xz; assert W > 0 and
    // mean water luma < mean grass luma (real channel math, printed).
    let (wx, wz) = /* a dry cell */;
    let y0 = splat_y_at(&splats, wx, wz);
    let brush = crate::map::sculpt::HeightBrush::new(40.0, 2.0);
    for _ in 0..3 { t.apply_stamp(&brush, wx, wz, crate::map::sculpt::BrushOp::Raise(1.5)).unwrap(); }
    let y1 = splat_y_at(&terrain_splats(&t, 1), wx, wz);
    println!("stamped cell splat y: {y0:.2} -> {y1:.2}");
    assert!(y1 - y0 > 0.5, "the layer must track the edit");
}
```

and in `src/game/mod.rs` (or `coverage_field.rs`) the cascade test: founders_vale + injected `CoverageFieldService::new(256)` (skip-print on Err), a childcare facility and a probe home on the same bank, `tick()` → `reachable == true`; carve a flood channel between them with Lower strokes (real `begin/stamp/end`), assert `end_stroke` set the dirty flag, `tick()` → `reachable == false`; print `pre-flood reach: true → post-flood reach: false`.

- [ ] **Step 2: Run to verify failure** — `cargo test --release terrain_splats_layer 2>&1 | tail -5` → FAIL: `terrain_splats` not found.
- [ ] **Step 3: Implement.** `terrain_splats`: iterate cells (step `spc`), one `GaussianSplat::volume` per cell at `[wx, sample, wz]`, scale `[cell_size*0.5*spc, 0.4, cell_size*0.5*spc]`, spectral via `rgb_to_spectral` from the class tint (Water `(0.05,0.10,0.22)`, TooSteep `(0.32,0.30,0.28)`, Buildable `(0.10,0.30,0.12)`, Reserved `(0.20,0.20,0.20)`) with the proven display boost. `GameView`: cache `last_terrain_version: u64`; the residual cook appends `terrain_splats(&game.terrain, 1)` when `!is_flat()`; on `version()` change re-cook residuals and re-upload (the M4 `upload_splats_at` path — renderer NOT reconstructed; assert by eye that `[scene] renderer constructed 1x` still prints at exit).
- [ ] **Step 4: Wire at exact callsite** — the M4 residual-splat builder in `play.rs`/`render_gpu` (wherever M4 landed `upload_splats_at` feeding); the version watch in the per-frame update.
- [ ] **Step 5: Run — verify non-trivial output** — both tests PASS with printed splat counts, y delta and the reach flip; `cargo run --release --bin play -- --shot-chrome chrome_shots` still passes (flat world pixel-unaffected).
- [ ] **Step 6: Commit**

```bash
git add src/render_gpu/mod.rs src/bin/play.rs src/game/mod.rs
git commit -m "feat(terraform): live terrain splat layer (version-watched residuals) + flood-severs-childcare cascade proof

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: `--shot-terraform` harness + walkthrough W3

**Files:**
- Modify: `src/bin/play.rs`, `src/walkthrough/mod.rs`

**Acceptance:** `cargo run --release --bin play -- --shot-terraform terraform_shots` exits 0 with the full Done-When print set (real numbers, three PNGs, pixel-diff > 20000 before→after, heights-bits gate on undo — never a pixel gate on undo); `cargo run --release --bin play -- --walkthrough 3` prints the W3 checklist; `cargo test --release walkthrough_w3 -- --nocapture` prints `[x]`/`[~]` per step and `W3 headless: <n>/<n> steps (<m> manual)`.

**Wiring requirement:** `shot_terraform(out_dir)` parsed beside `--shot-landvalue` in `play.rs::main`; `script(3)` + `run(3)` in `src/walkthrough/mod.rs` with at least: arm `build.terrain.raise` (ReceiptContains `Armed: Terrain: Raise`), the drag step (manual in print; the twin drives `begin/stamp/end` directly and checks ReceiptContains `Terraformed (Raise)`), the undo step (ReceiptContains `Undid: Terraformed`), and a brush-row step (`terrain.size.large` → `Brush: 80 m`). Stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — walkthrough twin test asserting `run(3)` succeeds and `script(3).len() >= 4`; harness smoke is the binary run itself.
- [ ] **Step 2: Run to verify failure** — `cargo test --release walkthrough_w3 2>&1 | tail -5` → FAIL: `script(3)` empty.
- [ ] **Step 3: Implement `shot_terraform`** (the `shot_landvalue` skeleton): start `MenuApp::start_game(Demo, "founders_vale")` + `bind_map` + coverage service injection (the play.rs:1257 recipe), `GameView::new`, aim the camera at the stroke site, render+save `terraform_before.png`; scripted raise stroke (~15 spaced stamps over 200 m), print the `[terraform] raise stroke: …` line from the logged action + funds delta; render+save `terraform_after.png`, print the pixel-diff line (gate > 20000); scripted shore lower-strokes until `water_flips > 0`, print the flood line + `childcare field dirty: true` (expose via a `#[doc(hidden)] pub fn childcare_dirty(&self) -> bool` test accessor or assert via the reach flip); `edit.undo` ×2 through the registry, compare height bits to the pre-stroke snapshot + funds, print the undo line, render+save `terraform_undo.png`.
- [ ] **Step 4: Implement W3** — `script(3)` steps with honest `say` lines (the W2 voice); `run(3)` twin on founders_vale driving dispatch + the stroke API.
- [ ] **Step 5: Run — verify the full Done-When** — the three commands above, plus `cargo test --release` fully green.
- [ ] **Step 6: Commit**

```bash
git add src/bin/play.rs src/walkthrough/mod.rs
git commit -m "feat(terraform): --shot-terraform proof harness (before/after/undo PNGs + funds/water/undo gates) and walkthrough W3

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks exist (Task 2's verdict inner moves to its real home in Task 4 as a *relocation*, both ends real code)
- [x] Every `Acceptance` names real non-trivial output (stamp counts, m³, kr, bit-compares, reach flips, pixel diffs) — never "tests pass"
- [x] Every `Wiring requirement` names exact functions and files (`MapEditor::sculpt` editor.rs:97, `rebuilt_with_log` game/mod.rs:1378, `panel_anchor_x` hud.rs:1215, …)
- [x] `IMPORTANT NOTES` carries code-checked signatures (`HeightBrush`, `BrushOp`, `derive_buildable`, `dispatch`, `draw_world_circle`, `ToolPanelOptionRow`, the 27→31 pin, the AddRoad SAVE_VERSION precedent)
- [x] `File Map` lists every file appearing in any task
- [x] No step contains `todo!()` / stub bodies
- [x] `Done When` names the exact command and the exact human-visible prints + PNGs
- [x] Names consistent across tasks: `sculpt_heightmap`, `apply_stamp`, `begin_stroke`/`stroke_stamp`/`end_stroke`, `terraform_blocked`, `terrain_splats`, `TerraformMode`, `TERRAFORM_KR_PER_M3`
- [x] Execution gate stated and honest: Tasks 3–6 after virtualized M4; Tasks 1–2 de-gateable (pure sim)
- [x] Undo PNG gated on height bits, not pixels (the radix-raster nondeterminism landmine)
