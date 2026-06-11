# Game UI Chrome — Wave 1 (CS2-Parity Shell for Civitas Care) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> ## ⛔ EXECUTION GATE — runs AFTER Editor/UX Wave 1
> This plan **must not start** until every task of `docs/superpowers/plans/2026-06-10-editor-ux-wave1.md` reports complete. Verify before Task 1: `cd ~/Ochroma/projects/civitas_care && cargo test --lib command:: walkthrough:: -- --nocapture` is green and prints `W1 headless: 8/8 steps (2 manual)`, and `cargo run --release --bin play -- --walkthrough` prints the W1 checklist. Every task below names the wave-1 artifacts it builds on (`GameRegistry`/`ViewState`/`run_command`/`tool_preview`/`draw_tool_preview`/`walkthrough::script`); if any is missing, stop and report — do not reinvent them.

**Goal:** Give the playable city the bar the user mandated (`docs/references/ui-layout-spec.md`, normative): a three-zone bottom bar — **left**: REAL pause/1×/3× speed cluster + city-info chips + Info/Actions/Development buttons (Development dimmed, no fake content); **center** (widest): the build menu (Zones/Care/Services/Utilities icon tabs) with the priced-thumbnail tool panel popping above its tab; **right**: Inspect + overlay toggles + screenshot + a Statistics button — plus Actions/Statistics submenu panels carrying ONLY real sim content, in-world measurement chips + care-radius ring. **Every submenu anchors ABOVE the bar to its invoking button; the inspector stays a floating window.** Every control dispatches wave-1 `GameCommand`s.
**Done When:** `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play -- --shot-chrome chrome_shots` prints `wrote chrome_shots/chrome_status.png`, `…chrome_tool_panel.png`, `…chrome_zone_draft.png`, `…chrome_overlays.png`, `…chrome_actions.png`, `…chrome_stats.png` plus `radius ring: projected <R>px vs expected <R'>px (|Δ| <= 3)`; reading those PNGs shows the §Design anatomy (three-zone bottom bar: `⏸▶▶▶` cluster + city/Care/Pop/kr chips + Info/Actions/dimmed-Development left, four build tabs centre, Inspect/Land/Care/camera/Statistics right; Kindergarten tile `40,000 kr` with blue selection border + close `x`, panel anchored above its tab; draft edges with `<N> m` chips; lit right-zone overlay buttons + legends; the Actions panel above its button with exactly the three policy rows + `Save city` + `Exit to menu`; the Statistics panel above its button with the demand/care-ledger/budget lines). `cargo run --release --bin play -- --walkthrough 2` prints the verbatim `W2 — chrome, 90 seconds:` checklist; `cargo test --lib walkthrough::tests::w2_runs_headless -- --nocapture` ends `W2 headless: 11/11 steps (4 manual)`. Live: pressing `0` shows `Speed: 1×` in the strip and the date/population advance with no further input; `0` twice more shows `Speed: Paused` and the date freezes; clicking Actions → `Universal childcare` prints `Policy: Universal childcare ON`.
**Architecture:** Keep the deterministic CPU-raster HUD; add a `theme` token table + rounded primitive, rebuild the bottom bands (strip/toolbar/tool-panel) on it as three vertical zones, one shared submenu shell anchored above the bar for Info/Actions/Statistics, thumbnails come from a `--thumbs-only` cook crop of the existing path-traced renders into `visual.preview`, and time is a lib-side `SimClock` accumulator that only ever dispatches the wave-1 `sim.tick` command — ticks stay the authoritative, replay-safe unit. The Actions panel mutates only the sim-wired `CarePolicies` (already in the v5 save schema — **no `SAVE_VERSION` change**).
**Design Document:** `docs/superpowers/specs/2026-06-10-game-ui-chrome-design.md`
**Tech Stack:** Rust (workspace toolchain as-is), winit 0.30 + softbuffer 0.4, ab_glyph CPU raster, `image` crate (already a dependency — used by `save_rgba`). **No new dependencies.**
**Build:** `cd ~/Ochroma/projects/civitas_care && cargo build` / targeted `cargo test --lib <filter>` (lib tests are GPU-free; only `--bin play` runs/tests need the GPU box).

---

## IMPORTANT NOTES

- **Repos & house rules:** ALL code lives in `~/Ochroma/projects/civitas_care` (GAME layer). The engine repo (`~/src/ochroma`) is never edited, built, or tested by this plan. **Do NOT run `git commit`** — every task's final step is *leave uncommitted*; the user commits.
- **Normative layout (user directive, 2026-06-10 — `docs/references/ui-layout-spec.md`):** the bar is three zones — left: city info + Info/Actions/Development buttons; center: the build menu (widest); right: tools + Statistics. **Universal rule: every submenu/panel pops ABOVE the main bar, anchored to its invoking button** (tool panel to its tab; Info/Actions/Statistics to their buttons; clamped on-screen). The building **inspector stays a floating window** (the directive's exception). At most one of Info/Actions/Statistics is open at once. The Development button ships **disabled** (dimmed, tooltip + click receipt `Progression — coming soon`) — NO panel, NO fake tree content.
- **Wave-1 artifacts this plan binds to (created by editor-ux-wave1 — verify they exist, do not re-create):**
  - `GameRegistry::dispatch(&self, view: &mut ViewState, game: &mut CivitasGame, sdf: &SpatialFieldRegistry, id: &str) -> Result<String, Vec<String>>` and `GameRegistry::standard()` — `src/command/mod.rs`. Command ids used here: `sim.tick`, `build.commit_zone`, `build.place_here`, `edit.undo`, `view.info`, and the ArmTool ids (`build.care.kindergarten`, `build.zones.residential_low`, …).
  - `ViewState` (`src/command/mod.rs`): `cat`, `tool`, `selected_asset_id`, `draft: Vec<[f32;2]>`, `cursor_world: Option<[f32;2]>`, `info_open`, `inspected`, `palette`, `receipts: Vec<String>`, `status_receipt: Option<String>`. This plan adds **additive** fields only (`sim_speed`, `overlay_land`, `overlay_care`, `panel_open: bool` default **true**, `shot_requested`) — never reorder/remove wave-1 fields.
  - `ToolPreview { caption, valid, parcels, cost }` + `tool_preview(..)` + `plop_blocked(..)` — `src/ui/preview.rs`; `hud::draw_tool_preview(px, w, h, cursor, caption, valid)` — the ghost. Chrome must NOT introduce a second verdict source.
  - `fn run_command(gv: &mut GameView, w: u32, h: u32, id: &str)` in `src/bin/play.rs` — ALL new buttons/keys go through it.
  - `walkthrough::{WalkthroughStep, Check, script(u8), print(u8), run(u8)}` — `src/walkthrough/mod.rs`; W2 extends `script`, it does not fork the module.
- **Real signatures verified in code 2026-06-10 (do not reinvent):**
  - hud.rs raster kit (all **private** to hud.rs — new drawing fns MUST live in hud.rs): `fn fill(px: &mut [[u8;4]], w: usize, h: usize, x0: i32, y0: i32, rw: i32, rh: i32, rgb: (u8,u8,u8), a: f32)` (:41), `fn text(px, w, h, font: &FontVec, size: f32, x: f32, top: f32, s: &str, rgb)` (:61), `fn text_width(font, size, s) -> f32` (:97), `fn fit_label(font, size, s, max_w) -> String` (:103), `fn line(px, w, h, x0, y0, x1, y1, rgb, a)` (:120, 2 px), `fn disc(px, w, h, cx, cy, r: i32, rgb, a)` (:149), `fn fonts() -> &'static (FontVec, FontVec)` (:12).
  - Public hud surface today: `fill_world_quad(px, w: u32, h: u32, corners: &[(f32,f32);4], rgb, alpha)` (:176 — accepts degenerate quads → triangles for the play/fast icons), `draw_overlay_legend(px, w, h, title, ramp: &dyn Fn(f32)->(u8,u8,u8))` (:237 — its `y0` uses `BAR_H + CAT_H + ASSET_H + THEME_H`; re-base in Task 3), `draw_zone_draft(px, w, h, pts: &[(f32,f32)], cursor: (f32,f32))` (:301), consts `BAR_H=34, CAT_H=48, ASSET_H=58, THEME_H=74` (:375-381), colors `ACCENT=(52,136,188), ACCENT_DARK=(24,72,104), PANEL=(13,17,24), PANEL_2=(24,30,40), TEXT=(232,238,246), MUTED=(172,184,198)` (:394-399), `draw_status_bar(px, w, h, left, center, right, info_hot)` (:433) + `status_info_button_at(w,h,x,y)` (:491) [retired in Task 3], `draw_category_bar(px, w, h, cats: &[(&str,bool)])` (:498) + `category_at` (:565), `draw_asset_row`/`draw_catalog_asset_row(px,w,h,items: &[(String,bool)])` (:610/:617) + `asset_at`/`catalog_asset_at` (:676/:681) [subsumed in Task 6], `draw_theme_row` (:725) + `theme_at` (:792) [unchanged], `BottomBarUi<'a>` (:813: `demand_bars: &'a [(&'a str, f32, (u8,u8,u8))], themes, theme_preview, catalog_assets, assets, categories, city_name, center, right, info_open`) + `draw_bottom_bar_ui` (:827), `draw_city_info_window` (:844) + `info_window_close_at`/`info_window_contains` (:896/:907) [unchanged], `draw_demand_bars(px, w, h, bars, reserved_bottom: i32)` (:995 — keep the fn, retire the floating-panel call). **Callers of this kit: `src/bin/play.rs` and `src/bin/city_coverage.rs` only — update both whenever a signature changes.**
  - `render_gpu::world_to_screen(view: Mat4, proj: Mat4, x: f32, z: f32, width: u32, height: u32) -> Option<(f32,f32)>` (render_gpu/mod.rs:71); `screen_to_ground(view, proj, px, py, width, height) -> Option<[f32;2]>` (:40); `orbit_camera(center: Vec3, radius, yaw, pitch, w, h) -> RenderCamera` (:3030); `city_center_radius(&CivitasGame) -> (Vec3, f32)` (:3056).
  - Costs/radii: `CareKind::params() -> CareParams { capacity, coverage_radius_m: f32, build_cost: f64, monthly_cost, staff_required }` (care/service.rs:43-105; Kindergarten = 1500.0 m / 40_000.0); `CityServiceKind::build_cost() -> f64` (services/mod.rs:70); zone tiles show **no price** (zoning charges nothing at commit — wave-1 note 6). `command::thousands(v: f64) -> String` is the formatter (wave-1 moved it lib-side).
  - Assets: `BuildingAsset { id, name, style, usage, level, footprint, footprint_bucket, floors, attributes: BuildingAttributes, visual: Option<AssetVisualRef> }` (asset/mod.rs:463); `AssetVisualRef { atoms: String, source: Option<String>, preview: Option<String> }` (:165-170 — `preview` is the thumbnail slot, currently `null` in pack.json); `BuildingAttributes.build_cost: u32` (:83); `AssetCatalog { pub packs: Vec<AssetPack>, ready_assets (private) }` (:547) — pack-relative paths resolve against the pack.json parent dir in `append_asset_packs_from_dir` → `load_ready_asset_payloads(ready_assets, base_dir, &pack)` (:719-736); thumbnails follow the identical pattern. `AssetCatalog::get(&self, id) -> Option<&BuildingAsset>` (:656). Cook: `DEFAULT_OUTPUT_DIR = "assets/buildings/forge_starter"` (game_asset_cook.rs:25); recipe ids `forge.house.craftsman`, `forge.house.victorian`, … (:492+); existing renders `assets/buildings/forge_starter/renders/spectra/{cooked_house_craftsman,house_victorian,police_station,school}.png` (768×768) + `{craftsman,rowhouse}/iso.png`; variants `.v1`/`.v2` reuse the base id's render.
  - `Observer { latitude_deg, longitude_deg, day_of_year: u32, hour: f32 }` (render_gpu/atmosphere.rs; default start `day_of_year: 172, hour: 14.0` in `GameView::new`). Season boundaries (pin exactly): `<60 || >=335` Winter, `60..152` Spring, `152..244` Summer, `244..335` Autumn → day 172 = `Summer`.
  - winit 0.30: the event loop runs `ControlFlow::Wait` (play.rs main). The clock drive uses `fn about_to_wait(&mut self, event_loop: &ActiveEventLoop)` on `ApplicationHandler` + `event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(100)))` while unpaused, `ControlFlow::Wait` while paused. `--shot*` harnesses press Space → **Space stays `sim.tick`**; do not change its meaning.
  - SimClock rates (pin exactly): Paused 0.0, Normal 0.5, Fast 1.5 ticks/s; drain cap **4 ticks per call**. `ViewState::default()` ⇒ `SimSpeed::Paused` — all existing tests/harnesses stay semantics-identical.
  - Actions-panel real content (verified in code 2026-06-10 — ONLY these rows; anything else is fake): `CarePolicies { universal_childcare: bool, parental_leave_years: f32, elder_subsidy: bool }` (care/policy.rs:16-24) with `operating_cost_multiplier() -> f64` (1.0 base, +0.5 childcare, +0.3 elder), `infant_demand_exempt(age) -> bool`, `satisfaction_bonus() -> f32` — ALL consulted every tick by `CivitasGame::tick` (game/mod.rs:775-777 leave exemption, :864 satisfaction, :875 upkeep multiplier) and ALREADY in the v5 save schema (`GameSave.policies`, save.rs:90 — **no `SAVE_VERSION` bump**). `game.policies` is a pub field on `CivitasGame` (game/mod.rs:147). Save: `CivitasGame::to_save()` (game/mod.rs:1246) + `GameSave::to_json()` (save.rs:98) exist; the menu's Load screen already scans+lists `.civsave` slots in `saves_dir` (`data_dir.join("saves")`, play.rs:832; menu/saves.rs) — write `quicksave.civsave` there. Exit to menu: the existing Esc path (`GameView::key` returns true → play.rs:1011 `self.game = None`). Parental-leave stepper pins the cycle `0.0 → 0.5 → 1.0 → 2.0 → 0.0` yr. **TaxPolicy is real but OUT of the Actions panel** (directive scope: care policies + save/menu actions only).
  - Statistics-panel real content (verified): `game.demand() -> Demand { residential, commercial, industry, office, farming }` (game/mod.rs:970); `game.last_ledger: CareLedger { labor_tax, care_upkeep }` + `net()`/`is_surplus()` (game/mod.rs:151, game_mechanics/economy/mod.rs:33-48); `game.last_budget: CityBudget` with `taxes, service_fees, transit_fares, trade_exports, power_sales, service_upkeep, transit_operating, trade_imports, power_imports` + `net()/revenue()/spending()` (game/mod.rs:212, budget/mod.rs:84+); `game.tick() -> CareStats` carries `population, effective_employed, gated_workers, total_unmet, care_buildings` (game/mod.rs:116/719). The info window already formats these (play.rs `info_lines()` :345-372) — the Statistics panel shows the same real numbers in detail. **Graphs are out of scope** (no history buffer exists).
  - Info window today: `i` key / `view.info` → centred `draw_city_info_window(px, w, h, "City Information", &info_lines())` (play.rs:666, hit tests :1124/:1128). The Info button re-anchors this CONTENT above the bar via the submenu shell; the inspector call site (play.rs:672) keeps the floating window untouched.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task. Every visual task ends with the executor **Reading the emitted PNG with the Read tool** and confirming the described content — a black or chrome-less frame fails the task.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/render_gpu/hud.rs` | `theme` tokens, `fill_rounded`, `season_label`, `StatusStrip`+`draw_status_strip`+`speed_button_at`+`strip_city_chip_at`, three-zone `draw_toolbar`+`toolbar_category_at`+`toolbar_left_button_at`+`toolbar_tool_at`, `ToolPanelTile`+`draw_tool_panel`+3 hit tests, `draw_measure_chip`, `draw_world_circle`, `draw_submenu_panel`+`submenu_close_at`+`submenu_contains`, `draw_actions_panel`+`actions_row_at`, thumb blit; retire `draw_status_bar`/`status_info_button_at`/`draw_category_bar`/`category_at`/`draw_asset_row`/`draw_catalog_asset_row`+hit tests; re-base `draw_overlay_legend`; raster tests |
| Create | `src/ui/sim_clock.rs` | `SimSpeed`, `SimClock::ticks_due`; tests |
| Create | `src/ui/panels.rs` | `actions_rows(&CarePolicies)`, `stats_lines(&CivitasGame, &CareStats)` — lib-side, GPU-free; tests |
| Modify | `src/ui/mod.rs` | `pub mod sim_clock;` + `pub mod panels;` + re-exports |
| Modify | `src/command/mod.rs` | additive `UiAction` variants (`SetSpeed`, `CycleSpeed`, `Screenshot`, `ToggleOverlay`, `ClosePanel`, `ToggleActions`, `ToggleStats`, `TogglePolicy`, `CycleLeave`, `SaveCity`, `ExitToMenu`) + `OverlayKind`/`PolicyKind`, new commands in `standard()`, dispatch arms + receipts, additive `ViewState` fields; tests |
| Modify | `src/asset/mod.rs` | `AssetThumb`, `thumbnails` map + load beside `load_ready_asset_payloads`, `thumbnail(id)` accessor; test |
| Modify | `src/bin/game_asset_cook.rs` | `--thumbs-only` mode: crop/downscale existing renders → `thumbs/<id>.png`, patch `visual.preview`, rewrite pack.json |
| Modify | `src/bin/play.rs` | three-zone strip/toolbar/panel/submenu draw + hit-test routing through `run_command`, `SimClock` drive in `about_to_wait`, overlay/screenshot key re-routes, Info re-anchor, `save_requested`/`exit_requested` consumption, `--shot-chrome` harness, `--walkthrough 2` arm, `shot_requested` consumption; migrate off retired hud fns |
| Modify | `src/bin/city_coverage.rs` | migrate off retired hud fns (it calls the bottom-bar kit) |
| Modify | `src/walkthrough/mod.rs` | `script(2)`/`print(2)`/`run(2)` chrome checklist + headless twin; tests |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Rounded token kit | `hud::tests::rounded_fill_clips_corners`: on white, corner probe pixels stay white, centre + edge-midpoint probes tinted; prints both counts | `assert!(buffer changed)` anywhere |
| Sim clock | `ui::sim_clock::tests::ticks_accumulate_by_speed` prints `paused 0 · 1x 1 · 3x 3 (dt=2s) · carry 0+1` ; `burst_capped_at_4` prints `60 s stall -> 4 ticks` | `ticks_due >= 0` |
| Speed commands | `command::tests::speed_cycle_emits_receipts` prints `Speed: 1× / Speed: 3× / Speed: Paused` receipts and the round-tripped `view.sim_speed` | dispatch returned `Ok` |
| Status strip | `hud::tests::status_strip_speed_buttons_hit_test` prints 3 button centres round-tripped + `None` off-strip; `season_labels_match_day` prints the 4 derived labels | `is_some()` at (0,0) |
| Cooked thumbnails | `asset::tests::catalog_loads_cooked_thumbnails` prints `craftsman thumb 96x72, <C> distinct colors` with C ≥ 3 | `thumb.is_some()` on empty map |
| Tool panel | `hud::tests::tool_panel_rows_hit_test` prints `tiles 0..N round-trip OK, close button OK`; PNG read shows `Kindergarten` + `40,000 kr` + blue border | function returned `()` |
| Measure chips + radius | `hud::tests::measure_chip_prints_distance` prints `edge 25 m -> label "25 m"`; `--shot-chrome` prints `radius ring: projected <R>px vs expected <R'>px (|Δ| <= 3)` | fixed-radius circle, no world math |
| Overlay/screenshot commands | `command::tests::overlay_and_screenshot_receipts` prints the exact receipts `Overlay: Land value ON`, `Overlay: Care reach ON`, `Screenshot queued` + the flipped flags | dispatch returned `Ok` |
| Actions panel mutates the sim | `command::tests::policy_toggles_move_the_multiplier` prints `multiplier 1.0 -> 1.5 -> 1.8 · exempt(0.3y)@0.5y leave: true` + the exact `Policy: …` receipts | asserting `actions_open` flipped |
| Statistics panel = the ledger | `ui::panels::tests::stats_lines_carry_the_ledger` prints the demand/care/budget lines and asserts they embed the same figures as `game.demand()` / `last_ledger.net()` / `last_budget.net()` | `assert!(!lines.is_empty())` |
| W2 | `walkthrough::tests::w2_runs_headless` prints `[x]`/`[~]` per step, ends `W2 headless: 11/11 steps (4 manual)` | running steps without checks |

---

## Task 1: Theme tokens + rounded raster kit — the chrome's visual base

**Wave-1 dependency:** none (pure hud.rs raster) — but the gate above must already be verified.

**Files:**
- Modify: `src/render_gpu/hud.rs`

**Acceptance:** `cd ~/Ochroma/projects/civitas_care && cargo test --lib render_gpu::hud::tests::rounded_fill_clips_corners -- --nocapture` → prints `corners untouched: 4/4, interior tinted: <N> px` with N > 500. Then `cargo run --release --bin play -- --shot inspector_shot.png` still succeeds and **Read `inspector_shot.png` with the Read tool**: the bottom dock, drawer, theme card, and info window now have visibly rounded corners and the same dark/blue palette — no regression in any text or hit-target.

**Wiring requirement:** Every existing panel-backing `fill` call in `draw_category_bar`, `draw_asset_row_at`, `draw_theme_row`, `draw_city_info_window`, `draw_palette`, `draw_demand_bars`, `draw_overlay_legend`, and `draw_tool_preview`'s caption backing switches to `fill_rounded(.., theme::RADIUS, ..)` with colors/alphas/spacing read from the new `theme` module consts — zero magic literals remain in those backings. Hit-test fns are untouched (rounding is visual only; 6 px corners don't move click targets materially). `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing test** in hud.rs `#[cfg(test)]`:

```rust
#[test]
fn rounded_fill_clips_corners() {
    let (w, h) = (64usize, 64usize);
    let mut px = vec![[255u8; 4]; w * h];
    super::fill_rounded(&mut px, w, h, 8, 8, 48, 32, theme::RADIUS, (20, 30, 40), 1.0);
    // Exact corner pixels of the bounding rect lie OUTSIDE the 6px corner discs.
    let corners = [(8usize, 8usize), (55, 8), (8, 39), (55, 39)];
    let untouched = corners.iter().filter(|&&(x, y)| px[y * w + x] == [255, 255, 255, 255]).count();
    let interior = px.iter().filter(|p| p[0] < 250).count();
    assert_eq!(untouched, 4, "rect corners must stay outside the rounding");
    assert_eq!(px[24 * w + 32], [20, 30, 40, 255], "centre must be solid fill");
    assert!(interior > 500, "interior tinted: got {interior}");
    println!("corners untouched: {untouched}/4, interior tinted: {interior} px");
}
```

- [x] **Step 2: Run to verify it fails** — `cargo test --lib render_gpu::hud::tests::rounded_fill 2>&1 | tail -5` → FAIL: `cannot find function fill_rounded` / `could not find theme`.
- [x] **Step 3: Implement** — `mod theme` (pub(crate) consts: `PANEL`, `PANEL_RAISED`, `ACCENT`, `ACCENT_HI`, `ACCENT_DARK`, `TEXT`, `MUTED`, `GOOD`, `BAD`, `RADIUS: i32 = 6`, `SP1/SP2/SP3 = 6/12/24`, `STRIP_H: i32 = 40`, `TOOLBAR_H: i32 = 48`, `PANEL_H: i32 = 96`, alphas `A_PANEL = 0.90`, `A_CARD = 0.66`, `A_BTN = 0.96`) — values per the design §4.2 token table; the existing `ACCENT`/`PANEL`/… consts become re-exports of the theme values so old call sites compile unchanged. `fill_rounded` = `fill` with a per-pixel corner test: pixel `(x,y)` inside one of the four `r×r` corner squares must satisfy `(dx*dx + dy*dy) <= r*r` against that corner's disc centre, else skipped. Full body, every branch.
- [x] **Step 4: Wire at exact callsites** — replace the named backings' `fill(..)` with `fill_rounded(.., theme::RADIUS, ..)` and swap their literal colors/alphas to theme consts (the list in the wiring requirement; `grep -n "fill(px" src/render_gpu/hud.rs` to enumerate).
- [x] **Step 5: Run — verify non-trivial output** — the test prints real counts; `cargo test --lib render_gpu::hud:: -- --nocapture` keeps all existing raster tests green; run the `--shot` harness and **Read the PNG** per the acceptance.
- [x] **Step 6: Leave uncommitted** (the user commits).

---

## Task 2: SimClock + speed commands — the city runs without keypresses

**Wave-1 dependency:** `GameRegistry::standard()/dispatch`, `ViewState`, `run_command`, command id `sim.tick`.

**Files:**
- Create: `src/ui/sim_clock.rs` · Modify: `src/ui/mod.rs`
- Modify: `src/command/mod.rs` (UiAction variants `SetSpeed(SimSpeed)`/`CycleSpeed`, `ViewState.sim_speed: SimSpeed`, commands `sim.speed.pause/normal/fast/cycle`, receipts)
- Modify: `src/bin/play.rs` (clock drive in `about_to_wait`, `0` key → `sim.speed.cycle`)

**Acceptance:** `cargo test --lib ui::sim_clock:: -- --nocapture` → `ticks_accumulate_by_speed` prints `paused 0 · 1x 1 · 3x 3 (dt=2s) · carry 0+1`; `burst_capped_at_4` prints `60 s stall -> 4 ticks`. `cargo test --lib command::tests::speed_cycle_emits_receipts -- --nocapture` → prints the three receipts `Speed: 1×`, `Speed: 3×`, `Speed: Paused` in order. On the GPU box: `cargo run --release --bin play`, start the Demo, press `0` once — the strip date and Pop advance with hands off the keyboard; `0` ×2 more freezes them.

**Wiring requirement:** `SimClock::ticks_due(dt, view.sim_speed)` is called from a new `fn about_to_wait(&mut self, event_loop: &ActiveEventLoop)` on `impl ApplicationHandler for App` in `src/bin/play.rs`; each due tick calls `run_command(gv, w, h, "sim.tick")` (the wave-1 helper — NEVER `game.tick()` direct); control flow set to `WaitUntil(now + 100ms)` while `view.sim_speed != Paused`, `Wait` otherwise. The `0` key in the in-game `KeyboardInput` arm dispatches `sim.speed.cycle`. Space remains `sim.tick`. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing tests** — `src/ui/sim_clock.rs`:

```rust
#[test]
fn ticks_accumulate_by_speed() {
    let mut c = SimClock::default();
    let paused = c.ticks_due(2.0, SimSpeed::Paused);
    let mut c = SimClock::default();
    let normal = c.ticks_due(2.0, SimSpeed::Normal); // 0.5 t/s → 1 tick
    let mut c = SimClock::default();
    let fast = c.ticks_due(2.0, SimSpeed::Fast);     // 1.5 t/s → 3 ticks
    assert_eq!((paused, normal, fast), (0, 1, 3));
    // Fractional ticks carry across calls: 1.0 s + 1.0 s at 1× = 0 then 1.
    let mut c = SimClock::default();
    let (a, b) = (c.ticks_due(1.0, SimSpeed::Normal), c.ticks_due(1.0, SimSpeed::Normal));
    assert_eq!((a, b), (0, 1), "fractional ticks must carry: {a}+{b}");
    println!("paused {paused} · 1x {normal} · 3x {fast} (dt=2s) · carry {a}+{b}");
}
#[test]
fn burst_capped_at_4() {
    let mut c = SimClock::default();
    let n = c.ticks_due(60.0, SimSpeed::Normal);
    assert_eq!(n, 4, "a stalled window must not replay a burst");
    println!("60 s stall -> {n} ticks");
}
```

and in `src/command/mod.rs` tests: `speed_cycle_emits_receipts` — fresh `ViewState`/`CivitasGame::new_small()`/empty SDF registry; dispatch `sim.speed.cycle` ×3; assert `view.sim_speed` walks `Normal → Fast → Paused` and the receipts are exactly `Speed: 1×`, `Speed: 3×`, `Speed: Paused`; print them.

- [x] **Step 2: Run to verify failure** — `cargo test --lib ui::sim_clock 2>&1 | tail -5` → FAIL: module not found.
- [x] **Step 3: Implement** — `SimSpeed` (`Default = Paused`, `fn rate(self) -> f64` = 0.0/0.5/1.5, `fn label(self) -> &'static str` = `Paused`/`1×`/`3×`, `fn cycle(self) -> SimSpeed`), `SimClock { accum: f64 }` with the drain-and-cap loop (cap 4; the accumulator is **cleared, not carried,** past the cap so a stall doesn't dribble ticks for minutes). `command/mod.rs`: `ViewState.sim_speed: SimSpeed` (additive, after the wave-1 fields), `UiAction::SetSpeed(SimSpeed)`/`CycleSpeed`, four registry entries (`sim.speed.pause`/`normal`/`fast`/`cycle`, category "Sim", shortcut `"0"` on cycle), dispatch arms set the speed and push receipt `format!("Speed: {}", view.sim_speed.label())`.
- [x] **Step 4: Wire at exact callsite** — `play.rs`: `App` gains `last_wake: std::time::Instant`; implement `about_to_wait` per the wiring requirement (compute `dt`, call `ticks_due`, loop `run_command(gv, w, h, "sim.tick")`, `request_redraw` if any tick ran, set the control flow). Add `"0"` to the in-game `Key::Character` match → `run_command(gv, w, h, "sim.speed.cycle")`.
- [x] **Step 5: Run — verify non-trivial output** — all three tests print real tick counts/receipts; `cargo test --lib walkthrough::tests::w1_runs_headless -- --nocapture` STILL prints `W1 headless: 8/8 steps (2 manual)` (paused default = zero behavior drift).
- [x] **Step 6: Leave uncommitted.**

---

## Task 3: Bottom status strip — the LEFT zone's city information, plus the care chip CS2 doesn't have

**Wave-1 dependency:** the status-centre precedence contract (zoning prompt > `ToolPreview.caption` > `view.status_receipt` > date) and `view.info` command id.

**Files:**
- Modify: `src/render_gpu/hud.rs` (`season_label`, `StatusStrip`, `draw_status_strip`, `speed_button_at`, `strip_city_chip_at`; retire `draw_status_bar`/`status_info_button_at`; re-base `draw_overlay_legend` + `BottomBarUi`/`draw_bottom_bar_ui` onto the strip)
- Modify: `src/bin/play.rs` (strip data + click routing; `--shot-chrome` harness skeleton writing `chrome_status.png`)
- Modify: `src/bin/city_coverage.rs` (migrate off `draw_status_bar`)

**Acceptance:** `cargo test --lib render_gpu::hud::tests::status_strip_speed_buttons_hit_test -- --nocapture` → prints the three round-tripped centres + the off-strip `None`; `…::season_labels_match_day` prints `30->Winter 100->Spring 172->Summer 300->Autumn`. Then `cargo run --release --bin play -- --shot-chrome chrome_shots` prints `wrote chrome_shots/chrome_status.png`; **Read that PNG with the Read tool** and verify: the strip is at the BOTTOM edge, full width; the **LEFT zone** reads (left→right) the `⏸ ▶ ▶▶` cluster with ⏸ accent-lit (`Paused` boot state), then the city-information chips — city name, `Care <NN>%`, `Pop <N>`, `<N> kr`; the centre reads `Day 172 · 10:30 · Summer` (the precedence line's idle state); the strip's right portion is clear (the toolbar's right zone above carries the tools, Task 4); the old top bar is GONE (top 34 px rows show city pixels, not a panel).

**Wiring requirement:** `draw_status_strip` is called from `draw_bottom_bar_ui` (replacing the `draw_status_bar` call; `BottomBarUi` swaps `center/right/info_open` for a `strip: StatusStrip<'a>` field) — both callers (`play.rs::GameView::render` via `draw_bottom_bar_ui`, `city_coverage.rs`) updated in this task. Clicks: `speed_button_at` → `sim.speed.pause/normal/fast`; `strip_city_chip_at` → `view.info` — both through `run_command` in the `MouseInput` Left-press arm, ABOVE the ground-click fallthrough. `draw_overlay_legend`'s reserved-bottom math re-bases from `BAR_H + CAT_H + ASSET_H + THEME_H` to `theme::STRIP_H + theme::TOOLBAR_H + theme::PANEL_H`. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing tests** — hud.rs: `season_labels_match_day` (the four exact pairs above) and `status_strip_speed_buttons_hit_test`: draw a `StatusStrip { speed: SimSpeed::Paused, city: "Demo", center: "Day 172 · 10:30 · Summer", care_pct: 62.0, population: 1234, funds: 56789.0, info_hot: false }` onto a 640×480 black buffer; for `i in 0..3` assert `speed_button_at` round-trips the documented button centres and returns `None` at `(strip centre x, h - STRIP_H - 4)`; assert the buffer row at `y = h - STRIP_H/2` contains ≥ 200 non-black pixels (the strip actually rendered); print the centres.
- [x] **Step 2: Run to verify failure** — `cargo test --lib render_gpu::hud::tests::status_strip 2>&1 | tail -5` → FAIL: `cannot find function draw_status_strip`.
- [x] **Step 3: Implement** — `season_label` per the pinned boundaries; `draw_status_strip`: rounded `theme::PANEL` band at `y0 = h - STRIP_H`, the LEFT zone laid out left→right from `x = SP2`: speed cluster = three 28×28 `PANEL_RAISED` rounded buttons (pause = two 4 px `fill` bars; play = one triangle via `fill_world_quad` with a repeated corner; fast = two triangles), active button `ACCENT` with `ACCENT_HI` underline; then the city-info chips in order city name (`fit_label`, `info_hot` → `ACCENT`), `Care NN%` (care-first — leads the info chips), `Pop N`, funds (`GOOD`), each a rounded `PANEL_RAISED` chip sized by `text_width + 2*SP1`; centre text via `text_width` centring (the wave-1 precedence line — its idle state is the date, so name/pop/funds/date all stay readable in the bar per the layout spec); nothing right-aligned in the strip. Degrade order on narrow widths (min window 640): drop chip backings before text, truncate city first. `speed_button_at`/`strip_city_chip_at` mirror the geometry (shared private `fn strip_layout(w, h) -> …` so draw and hit tests cannot drift).
- [x] **Step 4: Wire at exact callsites** — `BottomBarUi`/`draw_bottom_bar_ui` reshaped; `play.rs` builds `StatusStrip` from `self.observer` (`Day {} · {:02}:{:02} · {}` + `season_label`), `self.stats.population`, `game.sim.budget.funds` (via `command::thousands`), `game.coverage.coverage_ratio(DependentKind::Infant) * 100.0`; the centre keeps the wave-1 precedence chain verbatim. Delete `draw_status_bar`/`status_info_button_at` and their play.rs/city_coverage.rs call sites; route the two new hit tests through `run_command`. Add the `--shot-chrome <dir>` arm to `main` beside `--shot`: Demo game, `observer.hour = 10.5`, render 1280×800, write `chrome_status.png`, print `wrote …` (later tasks append frames to this same harness).
- [x] **Step 5: Run — verify non-trivial output** — both raster tests print real values; `cargo build --bin play --bin city_coverage` clean; run `--shot-chrome` and **Read the PNG** per the acceptance (a strip at the top, a missing season, or an unlit pause button each fail the task).
- [x] **Step 6: Leave uncommitted.**

---

## Task 4: Toolbar row — the three zones: Info/Actions/Development + demand minis | build tabs | tools + Statistics

**Wave-1 dependency:** ArmTool command ids (`registry` index helper from wave-1 Task 2) — category tab clicks and the right-zone Inspect button dispatch them; `view.cat`/`view.tool`, `view.info`.

**Files:**
- Modify: `src/render_gpu/hud.rs` (three-zone `draw_toolbar` + `toolbar_category_at` + `toolbar_left_button_at` + `toolbar_tool_at`; retire `draw_category_bar`/`category_at`; retire the floating `draw_demand_bars` panel call — fn kept for the embedded minis)
- Modify: `src/bin/play.rs`, `src/bin/city_coverage.rs` (callers)

**Acceptance:** `cargo test --lib render_gpu::hud::tests::toolbar_category_hit_test -- --nocapture` → prints `4 tabs round-trip OK · left buttons 0..3 OK · tools 0..5 OK · demand minis at x=<X>` (every centre-tab centre maps to its index, the three left buttons and five right tools round-trip through their own hit tests, a click on the demand cluster returns `None`). `cargo run --release --bin play -- --shot-chrome chrome_shots` rewrites `chrome_status.png`; **Read the PNG**: directly above the strip sits one toolbar band in three zones — **left**: `Info` `Actions` `Development` buttons (Development visibly dimmed/`MUTED` vs its siblings) + five 6 px demand mini-bars (R/C/I/O/F with visible different fill heights); **centre** (widest): the FOUR build-category tabs with glyph boxes (`Z C S U` — Inspect is NOT a tab), the active tab accent-filled with the `ACCENT_HI` underline; **right**: the `Inspect`, `Land`, `Care`, camera, `Statistics` buttons; the old floating bottom-left `Demand` panel is GONE.

**Wiring requirement:** `draw_toolbar` is called from `draw_bottom_bar_ui` (replacing `draw_category_bar` + the separate `draw_demand_bars` call; `BottomBarUi.demand_bars` feeds the minis). `toolbar_category_at` clicks in play.rs dispatch the clicked category's **first tool's ArmTool command id** through `run_command` (exactly what wave-1 Tab does), replacing the `category_at` branch — the tab list is the four build categories only; the right-zone Inspect button dispatches the Inspect category's existing ArmTool id. `toolbar_left_button_at`: Info → `run_command(.., "view.info")`; Actions/Development and the right-zone Land/Care/camera/Statistics buttons are **drawn this task, direct-wired to today's fields** (`land_overlay`/`coverage_overlay` toggles, the `p`-key save path; Actions/Statistics no-op with a `MUTED` status line; Development → status receipt `Progression — coming soon`) and **re-routed through their command ids in Tasks 8/9** — this mirrors Task 6's `Show`-row note: the buttons ship working either way, never dead. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing test** — draw a toolbar with the 4 build categories (`Zones` active), `left_hot = [false, false]`, the 5-tool right cluster, + the 5-bar demand fixture from play.rs (`("R", .7, ..), ("C", .4, ..) …`) on 1280×800 black; assert `toolbar_category_at` round-trips all 4 tab centres, `toolbar_left_button_at` round-trips the 3 left-button centres, `toolbar_tool_at` round-trips the 5 tool centres, each returns `None` over the demand cluster and above the band; assert the demand columns rendered (probe one pixel inside the R bar fill = its rgb) and the Development button's fill is dimmer than Info's (probe both backings); print per the acceptance.
- [x] **Step 2: Run to verify failure** — `cargo test --lib render_gpu::hud::tests::toolbar 2>&1 | tail -5` → FAIL: function not found.
- [x] **Step 3: Implement** — band at `y0 = h - STRIP_H - TOOLBAR_H`; **left zone**: three rounded `PANEL_RAISED` buttons `Info`/`Actions`/`Dev` (`fit_label`; open-state = `ACCENT`; Development always `MUTED` text on un-raised backing — the disabled treatment) then 5 mini-bars (6 px wide, `TOOLBAR_H - 16` tall tracks, bottom-up fills, single-letter `MUTED` labels under — a compacted `draw_demand_bars` inline, the old floating panel geometry retired); **centre**: the existing tab treatment (glyph box + `fit_label` + accent/underline) on rounded `PANEL_RAISED`, centred in the remaining width — the widest zone; **right**: five compact buttons right-aligned from `w - SP2` (Inspect glyph box, `V`/`C` letter boxes accent-lit when on, camera = rect + disc, `Stats` label button); shared `fn toolbar_layout(w, n)` feeds draw and all three hit tests. On hover over Development (cursor from play.rs), draw the tooltip chip `Progression — coming soon` via the Task-7 measure-chip backing (until Task 7 lands, a plain rounded chip — same geometry). The Development button IS the reserved progression slot (design §4.1 — content deferred, no fake tree).
- [x] **Step 4: Wire at exact callsites** — `draw_bottom_bar_ui` calls `draw_toolbar`; delete `draw_category_bar`/`category_at` and the standalone `draw_demand_bars(…, reserved)` call + its `reserved` math; update both bin callers; play.rs `MouseInput` swaps `hud::category_at` for `hud::toolbar_category_at` → ArmTool dispatch (4 cats), adds `toolbar_left_button_at`/`toolbar_tool_at` arms per the wiring requirement; the `category_items` call site passes only the four build categories (Inspect stays in `BuildCategory` data for the right-zone button's ArmTool id).
- [x] **Step 5: Run — verify non-trivial output** — raster test green with printed coords; `--shot-chrome` + **Read the PNG** per the acceptance (a still-floating Demand panel, a fifth Inspect tab, text-only tabs, or an undimmed Development button fail).
- [x] **Step 6: Leave uncommitted.**

---

## Task 5: Thumbnail cook + catalog loading — `--thumbs-only`, no Forge required

**Wave-1 dependency:** none directly (asset layer), but Task 6 consumes its output.

**Files:**
- Modify: `src/bin/game_asset_cook.rs` (`--thumbs-only` mode)
- Modify: `src/asset/mod.rs` (`AssetThumb`, `thumbnails` map loaded beside `load_ready_asset_payloads`, `thumbnail(id)`; test)

**Acceptance:** `cargo run --bin game_asset_cook -- --thumbs-only` (no `GAME_FORGE_BIN` needed) prints `thumbs: <K> written, <M> assets keep text tiles` with K ≥ 6 (craftsman + victorian + police + school + the two iso variants' bases, plus `.v1`/`.v2` reuse lines) and the files exist under `assets/buildings/forge_starter/thumbs/`. Then `cargo test --lib asset::tests::catalog_loads_cooked_thumbnails -- --nocapture` prints `craftsman thumb 96x72, <C> distinct colors` with C ≥ 3.

**Wiring requirement:** the `--thumbs-only` branch runs in `game_asset_cook::run()` BEFORE `ForgeRunner::discover()` (and returns — it must work on a box with no Forge). Thumbnail loading is called from `append_asset_packs_from_dir`'s per-pack loop in `src/asset/mod.rs` (a `load_thumbnails(thumbnails, base_dir, &pack)` sibling of `load_ready_asset_payloads(..)` at :731), so every `AssetCatalog::load`/`load_project_assets` caller gets thumbs for free. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing test** — `src/asset/mod.rs` tests:

```rust
#[test]
fn catalog_loads_cooked_thumbnails() {
    let catalog = AssetCatalog::load_project_assets();
    let Some(t) = catalog.thumbnail("forge.house.craftsman") else {
        println!("no cooked thumbs on disk — run `cargo run --bin game_asset_cook -- --thumbs-only`");
        return; // tolerate a clean checkout; the cook acceptance covers emission
    };
    assert_eq!(t.size(), (96, 72));
    let distinct: std::collections::HashSet<[u8; 3]> =
        t.pixels().chunks_exact(4).map(|p| [p[0], p[1], p[2]]).collect();
    assert!(distinct.len() >= 3, "a real image, not a flat fill: {} colors", distinct.len());
    println!("craftsman thumb 96x72, {} distinct colors", distinct.len());
}
```

- [x] **Step 2: Run to verify failure** — `cargo test --lib asset::tests::catalog_loads 2>&1 | tail -5` → FAIL: `no method named thumbnail`.
- [x] **Step 3: Implement** — asset/mod.rs: `AssetThumb { w, h, rgba }` (+`size()`/`pixels()`), `thumbnails: HashMap<String, AssetThumb>` on `AssetCatalog` (Default/new/load paths initialize it), `load_thumbnails` decoding `base_dir.join(preview)` via the `image` crate for every asset whose `visual.preview` is `Some`, `thumbnail(&self, id)`. Cook: static `fn thumb_sources() -> &'static [(&'static str, &'static str)]` mapping recipe id → render path relative to `DEFAULT_OUTPUT_DIR` (`("forge.house.craftsman", "renders/spectra/cooked_house_craftsman.png")`, `("forge.house.victorian", "renders/spectra/house_victorian.png")`, `("civic.police", "renders/spectra/police_station.png")`, `("civic.school", "renders/spectra/school.png")` — verify the civic ids against pack.json at implementation time and extend with the `craftsman/iso.png` / `rowhouse/iso.png` archetypes' actual ids); `--thumbs-only` loads `pack.json`, for each mapped id (and each `<id>.v<n>` variant present, reusing the base render) centre-crops square, resizes 96×72 (`image::imageops::FilterType::Lanczos3`), writes `thumbs/<asset_id>.png`, sets `visual.preview = Some("thumbs/<asset_id>.png")`, serializes pack.json back (`serde_json::to_vec_pretty`), prints the acceptance line. Unmapped assets stay `preview: null` — counted in `<M>`.
- [x] **Step 4: Wire at exact callsite** — the `load_thumbnails` call inside `append_asset_packs_from_dir` (mod.rs:719) next to `load_ready_asset_payloads`; `--thumbs-only` arg branch at the top of `run()` (game_asset_cook.rs:39) before Forge discovery.
- [x] **Step 5: Run — verify non-trivial output** — run the cook (prints K/M), re-run the lib test (prints dims + distinct colors), and `git -C ~/Ochroma/projects/civitas_care diff --stat assets/` shows only `pack.json` + new `thumbs/*.png`.
- [x] **Step 6: Leave uncommitted.**

---

## Task 6: Contextual tool panel — option rows, priced thumbnail grid, close X

**Wave-1 dependency:** ArmTool command ids per tile (`registry` index helper), `view.cat/tool/selected_asset_id`, `view.panel_open` (added here, default `true`).

**Files:**
- Modify: `src/render_gpu/hud.rs` (`ToolPanelTile`, `draw_tool_panel`, `tool_panel_thumb_at`, `tool_panel_option_at`, `tool_panel_close_at`, private thumb blit; retire `draw_asset_row`/`draw_catalog_asset_row` + `asset_at`/`catalog_asset_at`)
- Modify: `src/command/mod.rs` (`UiAction::ClosePanel`, command `view.panel.close`, `ViewState.panel_open: bool` default `true`)
- Modify: `src/bin/play.rs`, `src/bin/city_coverage.rs` (callers; tile/option/close click routing; `--shot-chrome` gains `chrome_tool_panel.png`)

**Acceptance:** `cargo test --lib render_gpu::hud::tests::tool_panel_rows_hit_test -- --nocapture` → prints `tiles 0..<N> round-trip OK, close button OK, option row (0,1) OK, anchored at tab x=<X>`. `cargo run --release --bin play -- --shot-chrome chrome_shots` prints `wrote chrome_shots/chrome_tool_panel.png`; **Read the PNG**: above the toolbar, **x-anchored to the Care tab** (the universal submenu rule — the panel's horizontal centre tracks the invoking tab, clamped on-screen), sits the Care panel — left column shows a labelled `Show` option row; the grid shows the five care tiles with names and prices (`Kindergarten` tile reads `40,000 kr`), the armed tile carries a 2 px blue border + lighter header bar, tiles with cooked thumbs (if any care asset maps) show imagery while the rest are text tiles, and a `x` button sits at the panel's top-right. Pressing the close `x` in a live run prints receipt `Closed: panel` and collapses the panel leaving strip+toolbar.

**Wiring requirement:** `draw_tool_panel` is called from `GameView::render` in play.rs when `view.panel_open` (replacing the `draw_asset_row` + `draw_catalog_asset_row` calls; `BottomBarUi` drops `assets`/`catalog_assets` in favor of the panel inputs). Tile data: label from `BuildTool::tile_label`, price = `Some(CareKind::params().build_cost)` / `Some(CityServiceKind::build_cost())` / `None` for zones+Inspect, thumb = `catalog.thumbnail(asset_id)` for the zone tool's pinned/level-1 asset where one exists, selected = `i == view.tool`. Option rows: Zones → `("Growable", catalog_asset_items())` clicking through the existing `select_catalog_asset` (direct-wired, wave-1 note); Care/Services → `("Show", [("Value", view.overlay_land), ("Care", view.overlay_care)])` dispatching `view.overlay.*` once Task 8 lands (until then the row renders from the GameView fields and clicks toggle them directly — swapped in Task 8). Tile clicks dispatch the tile's ArmTool id; `tool_panel_close_at` → `view.panel.close` (receipt `Closed: panel`); selecting any category sets `panel_open = true` (in the ArmTool dispatch arm). `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing test** — hud.rs: build 5 `ToolPanelTile`s (one `selected`, one with a synthetic 96×72 two-color `AssetThumb`, prices `Some(40_000.0)` ×2 / `None` ×3) + one option row; draw on 1280×800 black; assert all tile centres round-trip through `tool_panel_thumb_at`, the close button round-trips, option entry `(0,1)` round-trips, `None` outside the panel; assert the selected tile's border pixel is `ACCENT_HI` and the thumb tile's centre differs from the text tiles' background; print per acceptance.
- [x] **Step 2: Run to verify failure** — `cargo test --lib render_gpu::hud::tests::tool_panel 2>&1 | tail -5` → FAIL: function not found.
- [x] **Step 3: Implement** — panel band at `y0 = h - STRIP_H - TOOLBAR_H - PANEL_H` (PANEL_H = 96 fits a 64 px thumb + name + price line), rounded `theme::PANEL`, **width sized to content and x-centred on the invoking tab's centre (from `toolbar_layout`), clamped to `[SP2, w - SP2]`** — not full-width, so the anchoring is visible; left option column ~190 px (label `MUTED` 12 px + rounded value buttons, the theme-row idiom); grid right: tiles `72×(PANEL_H - 2*SP2)` with `fit_label` name, `MUTED` price line via `command::thousands` + ` kr`, thumb blitted letterboxed into the upper 64×48 (private `fn blit_thumb` alpha-1 copy with nearest scaling), selection = `ACCENT_HI` 2 px inset border + `ACCENT` header strip; close `x` = 20×20 rounded button at panel top-right. One private `fn panel_layout(w, h, n_tiles, n_rows, anchor_x)` shared by draw + all three hit tests.
- [x] **Step 4: Wire at exact callsites** — per the wiring requirement; delete the four retired fns and migrate `city_coverage.rs`; `--shot-chrome` arms the Care category (`run_command(.., "build.care.kindergarten")`), sets a mid-screen `cursor_world`, writes `chrome_tool_panel.png`.
- [x] **Step 5: Run — verify non-trivial output** — raster test prints the round-trip lines; `--shot-chrome` then **Read the PNG** per the acceptance (a missing price, missing border, or the old flat asset rows fail).
- [x] **Step 6: Leave uncommitted.**

---

## Task 7: In-world measurement chips + care-radius ring

**Wave-1 dependency:** `view.draft`, `view.cursor_world`, `tool_preview` ghost (`draw_tool_preview` stays the only verdict chip — this task adds geometry feedback, not verdicts).

**Files:**
- Modify: `src/render_gpu/hud.rs` (`draw_measure_chip`, `draw_world_circle`)
- Modify: `src/bin/play.rs` (edge-length chips beside `draw_zone_draft`; radius ring when a Care tool is armed; `--shot-chrome` gains `chrome_zone_draft.png` + the ring assertion)

**Acceptance:** `cargo test --lib render_gpu::hud::tests::measure_chip_prints_distance -- --nocapture` → prints `edge 25 m -> label "25 m"` (label computed from world coords by the play-side helper under test, chip rendered with non-zero backing). `cargo run --release --bin play -- --shot-chrome chrome_shots` prints `wrote chrome_shots/chrome_zone_draft.png` AND `radius ring: projected <R>px vs expected <R'>px (|Δ| <= 3)` with the assertion enforced (exit 1 on miss); **Read the PNG**: a 3-corner zone draft shows the existing node discs + edges PLUS a small dark chip at each edge midpoint reading `<N> m`, and (second half of the frame sequence) the armed-Kindergarten cursor carries a blue circle whose extent visibly spans the city at 1500 m scale plus the wave-1 red/green ghost disc.

**Wiring requirement:** play.rs `GameView::render`: after the existing `draw_zone_draft` call, for each draft edge (`view.draft` world pairs, plus last→cursor rubber-band edge) compute `((dx*dx+dz*dz).sqrt())`, project the midpoint via `render_gpu::world_to_screen`, call `hud::draw_measure_chip(px, w, h, mid, &format!("{:.0} m", len))`; when `matches!(current_tool, BuildTool::Care(k))` and `view.cursor_world` is `Some(c)`, call `hud::draw_world_circle(px, w, h, cam.view, cam.proj, c, k.params().coverage_radius_m, theme::ACCENT, 0.8)` BEFORE the ghost so the verdict disc draws on top. The `--shot-chrome` ring check: `expected R' = |world_to_screen(c + [1500, 0]) − world_to_screen(c)|.x`, `projected R` measured by scanning the rendered row through the cursor for the accent-blue ring pixels. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing test** — hud.rs raster test: 200×200 black buffer, `draw_measure_chip(.., (100.0, 100.0), "25 m")`; assert the anchor-adjacent backing region has > 40 non-black pixels and that text pixels exist (any pixel ≥ 200 luminance inside the chip rect); plus a play.rs-side pure helper `fn edge_label(a: [f32;2], b: [f32;2]) -> String` unit test: `edge_label([0.0,0.0],[25.0,0.0]) == "25 m"`; print both.
- [x] **Step 2: Run to verify failure** — `cargo test --lib render_gpu::hud::tests::measure_chip 2>&1 | tail -5` → FAIL: function not found.
- [x] **Step 3: Implement** — `draw_measure_chip`: rounded `theme::PANEL` backing sized `text_width(label)+2*SP1` × 18, `TEXT` 12 px label, anchored centred on the point (the `draw_tool_preview` caption idiom). `draw_world_circle`: 64 chords; project each `c + r·(cosθ, sinθ)` through `world_to_screen`, draw `line` between consecutive `Some` pairs, skip segments with either end `None` (off-screen-safe).
- [x] **Step 4: Wire at exact callsites** — per the wiring requirement; extend `--shot-chrome`: build the 3-corner draft via the wave-1 dispatch path (arm `build.zones.residential_low`, push the fixture corners into `view.draft`), write `chrome_zone_draft.png`; then arm `build.care.kindergarten`, set `cursor_world` mid-frame, render, run the ring measurement, print + assert, write into `chrome_tool_panel.png` (the Task-6 frame now includes the ring).
- [x] **Step 5: Run — verify non-trivial output** — tests print the label + pixel counts; run `--shot-chrome`, confirm the ring line prints a real `|Δ| <= 3`, and **Read both PNGs** per the acceptance (chips missing on any edge, or a ring that doesn't scale with zoom, fail).
- [x] **Step 6: Leave uncommitted.**

---

## Task 8: Overlay/screenshot commands — the right-zone tools go through the registry

**Wave-1 dependency:** `GameRegistry` dispatch, receipts (I11); overlays move INTO `ViewState`.

**Files:**
- Modify: `src/command/mod.rs` (`UiAction::Screenshot`/`ToggleOverlay(OverlayKind)` + `OverlayKind`; `ViewState.overlay_land/overlay_care/shot_requested`; commands `view.screenshot` (`p`), `view.overlay.land` (`v`), `view.overlay.care` (`c`); receipts `Screenshot queued`, `Overlay: Land value ON/OFF`, `Overlay: Care reach ON/OFF`)
- Modify: `src/bin/play.rs` (overlay fields migrate `GameView.land_overlay/coverage_overlay` → `view.overlay_*` everywhere incl. the two play-gate bin tests; `v`/`c`/`p` keys + the Task-4 right-zone Land/Care/camera buttons → `run_command`; `shot_requested` consumed after present; Task-6's `Show` option row re-routed through the overlay commands; `--shot-chrome` gains `chrome_overlays.png`)

**Acceptance:** `cargo test --lib command::tests::overlay_and_screenshot_receipts -- --nocapture` → prints the exact receipts `Overlay: Land value ON`, `Overlay: Care reach ON`, `Screenshot queued` and the flipped `view.overlay_land`/`overlay_care`/`shot_requested` flags; `cargo run --release --bin play -- --shot-chrome chrome_shots` prints `wrote chrome_shots/chrome_overlays.png`; **Read the PNG**: both overlays active — tinted lot quads + `Land value` legend AND coverage discs + `Childcare reach` legend — with the right-zone `Land`/`Care` buttons accent-lit and the camera button visible in the right zone (NOT in any screen corner). On the GPU box, clicking the right-zone camera button prints `wrote play_coverage_<NN>.png` on the next frame.

**Wiring requirement:** the Task-4 `toolbar_tool_at` arms for Land/Care/camera re-route from the direct field toggles to `run_command` with `view.overlay.land` / `view.overlay.care` / `view.screenshot` (deleting the direct wiring — this completes Task 4's explicit re-route note). The `v`/`c`/`p` `Key::Character` arms re-route through the same ids (delete the direct field toggles and the `p` inline render+save — the flag path replaces it: `App::redraw` checks `gv.view.shot_requested` after `buf.present()`, clears it, renders once more and saves via the existing `next_coverage_shot_path`/`save_rgba`, prints `wrote …`). The overlay render branches in `GameView::render` read `view.overlay_land`/`view.overlay_care`. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing test** — `command::tests::overlay_and_screenshot_receipts` (dispatch the three ids; assert flags + the exact receipts above; print them).
- [x] **Step 2: Run to verify failure** — `cargo test --lib command::tests::overlay 2>&1 | tail -5` → FAIL: no variant `ToggleOverlay`.
- [x] **Step 3: Implement** — the UiAction variants + dispatch arms + `ViewState` fields (buttons and keys share one command id each — that is the point).
- [x] **Step 4: Wire at exact callsites** — per the wiring requirement, including migrating `land_overlay`/`coverage_overlay` reads in the two bin play-gate tests (`placement_clearance_play_gate_blocks_on_building` untouched; `childcare_coverage_play_gate_before_after`'s `gv.key("c")` now lands on the dispatch path — same observable behavior) and re-routing Task 6's `Show` option row + Task 4's right-zone buttons through the overlay command ids; `--shot-chrome` frame: both overlays on via dispatch, write `chrome_overlays.png`.
- [x] **Step 5: Run — verify non-trivial output** — the command test prints the real receipts; **Read `chrome_overlays.png`** per the acceptance (overlay buttons in a screen corner instead of the right zone fail).
- [x] **Step 6: Leave uncommitted.**

---

## Task 9: Actions + Statistics submenu shells, the Info re-anchor, and the W2 walkthrough

**Wave-1 dependency:** `walkthrough::{script, print, run, Check}`, `GameRegistry` dispatch, receipts (I11), `view.info`.

**Files:**
- Create: `src/ui/panels.rs` (`actions_rows`, `stats_lines`; tests) · Modify: `src/ui/mod.rs`
- Modify: `src/command/mod.rs` (`UiAction::ToggleActions`/`ToggleStats`/`TogglePolicy(PolicyKind)`/`CycleLeave`/`SaveCity`/`ExitToMenu` + `PolicyKind`; `ViewState.actions_open/stats_open/save_requested/exit_requested`; commands `view.actions`, `view.stats`, `view.development` (receipt-only `Progression — coming soon` — mutates NOTHING, the honest disabled-button dispatch; Task 4's direct status line re-routes through it), `policy.childcare.universal`, `policy.elder.subsidy`, `policy.leave.cycle`, `game.save`, `game.exit_menu`; receipts per design §4.7; opening one of actions/stats/info closes the other two)
- Modify: `src/render_gpu/hud.rs` (`draw_submenu_panel` + `submenu_close_at`/`submenu_contains`, `draw_actions_panel` + `actions_row_at`)
- Modify: `src/bin/play.rs` (Info window re-anchored through the submenu shell — inspector call site untouched; Actions/Statistics panels drawn + clicks routed; Task-4's Actions/Statistics buttons re-routed through `view.actions`/`view.stats`; `save_requested` → write `game.to_save().to_json()` to `saves_dir/quicksave.civsave` + print `saved: <path>`; `exit_requested` → the existing `self.game = None` path; `--shot-chrome` gains `chrome_actions.png` + `chrome_stats.png`; `--walkthrough 2` arm)
- Modify: `src/walkthrough/mod.rs` (`script(2)` + `w2_runs_headless`)

**Acceptance:** `cargo test --lib command::tests::policy_toggles_move_the_multiplier -- --nocapture` → prints `multiplier 1.0 -> 1.5 -> 1.8 · exempt(0.3y)@0.5y leave: true` and the exact receipts `Policy: Universal childcare ON`, `Policy: Elder subsidy ON`, `Policy: Parental leave 0.5 yr` (asserts `game.policies` actually mutated and `operating_cost_multiplier()` walked 1.0→1.5→1.8 — the panel moves the REAL sim levers). `cargo test --lib ui::panels::tests -- --nocapture` → `actions_rows_carry_only_wired_policies` prints the exactly-five rows (`Universal childcare`/`Parental leave`/`Elder subsidy`/`Save city`/`Exit to menu` — nothing else exists to show) and `stats_lines_carry_the_ledger` prints the demand/care/budget lines asserting they embed the same figures as `game.demand()`, `game.last_ledger.net()`, `game.last_budget.net()`. `cargo test --lib walkthrough::tests::w2_runs_headless -- --nocapture` → prints one `[x]`/`[~]` line per step ending `W2 headless: 11/11 steps (4 manual)`; `cargo run --release --bin play -- --walkthrough 2` prints the checklist below **verbatim**; `cargo run --release --bin play -- --shot-chrome chrome_shots` prints `wrote chrome_shots/chrome_actions.png` and `wrote chrome_shots/chrome_stats.png`; **Read both PNGs**: `chrome_actions.png` shows the Actions panel popped above the Actions button (bottom edge meeting the toolbar's top, panel x near the button — NOT centred mid-screen) with the five rows and the toggled states; `chrome_stats.png` shows the Statistics panel above the Statistics button (right zone) with the real demand figures, the care ledger (labor tax vs upkeep, net), and the budget summary lines — no chart axes, no invented series.

The W2 script (`script(2)` → printed by `--walkthrough 2`):

```
W2 — chrome, 90 seconds:
 1. [ ] Start the Demo. The bar hugs the BOTTOM edge in three zones —
        left: ⏸ lit + city/Care/Pop/kr chips + Info|Actions|Development
        (dimmed) buttons; centre: the four build tabs; right: Inspect,
        V/C overlays, camera, Statistics.
 2. [ ] Press 0. Receipt "Speed: 1×"; the date and Pop now advance
        hands-off. 0 twice more: "Speed: Paused" freezes them.
 3. [ ] Zones → Res Low, click 3 corners: every edge wears an "N m"
        chip and the ghost caption tracks "≈ N parcels".
 4. [ ] Care → Kindergarten: the tool panel pops ABOVE its tab with
        priced tiles ("Kindergarten · 40,000 kr", blue border when
        armed) and the cursor carries the 1500 m reach ring + ghost.
 5. [ ] Click the panel's x. Receipt "Closed: panel"; the bar stays.
 6. [ ] Press v then c. Receipts "Overlay: Land value ON", "Overlay: Care
        reach ON"; the right-zone V/C buttons light; legends stack.
 7. [ ] Click the right-zone camera button. Receipt "Screenshot queued",
        then the log prints "wrote play_coverage_NN.png".
 8. [ ] Click a speed button (▶▶). Receipt "Speed: 3×" — the same control
        the 0 key cycles; buttons and keys share one command id.
 9. [ ] Click Actions. The panel pops above the button; toggle Universal
        childcare → receipt "Policy: Universal childcare ON".
10. [ ] Click Statistics. The panel pops above the button: demand detail,
        care ledger (labor tax vs upkeep), budget net — real numbers,
        no graphs.
11. [ ] Click Development. It stays dimmed; the tooltip/receipt reads
        "Progression — coming soon" — no tree opens, nothing fake.
```

**Wiring requirement:** `draw_submenu_panel`/`draw_actions_panel` called from `GameView::render` after the bar when `view.info_open`/`actions_open`/`stats_open` (Info content = the existing `info_lines()` verbatim — play.rs:666 migrates to the shell anchored at the Info button's centre from `toolbar_layout`; the **inspector call site play.rs:672 keeps the floating `draw_city_info_window` untouched**); `submenu_close_at`/`submenu_contains`/`actions_row_at` routed in the `MouseInput` Left-press arm ABOVE ground handling — Actions rows dispatch `policy.childcare.universal`/`policy.leave.cycle`/`policy.elder.subsidy`/`game.save`/`game.exit_menu` by index from `ui::panels::actions_rows` order. The `policy.*` dispatch arms mutate `game.policies` directly (the sim consults it next tick — game/mod.rs:775/864/875; pin the leave cycle `0.0 → 0.5 → 1.0 → 2.0 → 0.0`). `game.save` sets `view.save_requested` (receipt `Save queued`); the bin consumes it after present and writes `game.to_save().to_json()` to `saves_dir/quicksave.civsave` (the dir play.rs:832 already builds and the menu Load screen already lists) printing `saved: <path>` — **no `SAVE_VERSION` change** (policies are already schema fields, save.rs:90). `game.exit_menu` sets `view.exit_requested` (receipt `Exiting to menu`); the bin consumes it via the existing `self.game = None` path (play.rs:1011). `script(2)` steps map: 1 → `Manual`, 2 → dispatch `sim.speed.cycle` + `ReceiptContains("Speed: 1×")` then ×2 + `ReceiptContains("Speed: Paused")`, 3 → the W1 zone fixture + `PreviewEqualsCommit` + `Manual` for the chips, 4 → arm kindergarten + `ReceiptContains("Armed: Build: Kindergarten")` + `Manual` for ring/tiles, 5 → dispatch `view.panel.close` + `ReceiptContains("Closed: panel")`, 6 → dispatch both overlay ids + `ReceiptContains("Overlay: Care reach ON")` + assert both flags, 7 → dispatch `view.screenshot` + `ReceiptContains("Screenshot queued")` + assert `shot_requested`, 8 → dispatch `sim.speed.fast` + `ReceiptContains("Speed: 3×")`, 9 → dispatch `view.actions` + `policy.childcare.universal` + `ReceiptContains("Policy: Universal childcare ON")` + assert `game.policies.universal_childcare`, 10 → dispatch `view.stats` + `ReceiptContains("Opened: statistics")` + `Manual` for the panel content, 11 → dispatch `view.development` + `ReceiptContains("Progression — coming soon")` + assert NO panel flag flipped. `--walkthrough 2` arm parses the optional milestone (`--walkthrough` alone still prints W1 — wave-1 behavior preserved). `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing tests** — `command::tests::policy_toggles_move_the_multiplier` (fresh `ViewState`/`CivitasGame::new_small()`; dispatch the three policy ids; assert `game.policies` mutation + the multiplier walk 1.0→1.5→1.8 + `infant_demand_exempt(0.3)` after one leave cycle + exact receipts; print), `ui::panels::tests::{actions_rows_carry_only_wired_policies, stats_lines_carry_the_ledger}` (the latter ticks a small game once, then asserts the formatted lines embed `game.demand().residential` `{:.2}`, `last_ledger.net()` `{:.0}`, `last_budget.net()` `{:+.0}`; print all lines), and `walkthrough::tests::w2_runs_headless` (calls `walkthrough::run(2)`).
- [x] **Step 2: Run to verify failure** — `cargo test --lib command::tests::policy 2>&1 | tail -5` → FAIL: no command `policy.childcare.universal` / module `panels` not found / `script(2)` panics on unknown milestone.
- [x] **Step 3: Implement** — `ui/panels.rs` (`actions_rows` formats the three policies — `("Universal childcare", "ON"/"OFF", flag)`, `("Parental leave", "off"/"<N> yr", >0)`, `("Elder subsidy", …)` — plus the fixed `("Save city", "quicksave", false)` and `("Exit to menu", "", false)` rows; `stats_lines` formats the §IMPORTANT-NOTES ledger fields, one section header line each for `Demand`, `Care ledger`, `City budget`); the UiAction variants + dispatch arms + `ViewState` fields (mutual exclusion: each `Toggle*` clears the other two open-flags + `info_open`); hud.rs `draw_submenu_panel` (rounded `theme::PANEL`, title bar + close `x` — the `draw_city_info_window` idiom re-anchored: bottom edge at `h - STRIP_H - TOOLBAR_H - SP1`, x clamped around `anchor_x`) + `draw_actions_panel` (shell + rounded value buttons per row, lit `ACCENT` when on) + hit tests sharing one private `fn submenu_layout(w, h, anchor_x, rows)`; `script(2)`/`run(2)` per the mapping (run(2) reuses the W1 headless rig builder).
- [x] **Step 4: Wire at exact callsites** — per the wiring requirement (Info migration at play.rs:666, click routing, the two flag consumptions, Task-4 button re-routes); `--shot-chrome`: dispatch `view.actions` + one policy toggle, write `chrome_actions.png`; dispatch `view.stats`, write `chrome_stats.png`.
- [x] **Step 5: Run — verify non-trivial output** — the four acceptance commands print the real multiplier walk/rows/lines/checklist/`11/11`; **Read `chrome_actions.png` and `chrome_stats.png`** per the acceptance (a centred info-style window, a sixth invented Actions row, or chart-like content fail); on the GPU box run the W2 checklist at the keyboard end-to-end.
- [x] **Step 6: Leave uncommitted.**

---

## Code-reality notes binding this plan (code wins over the design where they disagree)

1. `draw_demand_bars` is not deleted — Task 4 inlines a compact variant inside `draw_toolbar`; the public fn stays until no caller remains (then delete in the same task).
2. The design's `fill_rounded` is private (like `fill`); only the composed draw fns are public — hud.rs keeps its private-raster-kit discipline.
3. `chrome_tool_panel.png` carries BOTH Task 6 (panel) and Task 7 (ring) content — the harness grows monotonically (six PNGs by Task 9); earlier acceptances re-run green after later tasks.
4. Care tools have no cooked renders today — the Kindergarten tile in `chrome_tool_panel.png` is a **text tile with a price**; thumbnail tiles are proven on zone tiles whose assets map (craftsman/victorian). The PNG checks say so explicitly: do not fake a care thumbnail.
5. `ViewState`/`UiAction` extensions are additive; if wave-1's actual landed shapes differ from the artifacts listed in IMPORTANT NOTES, **stop and reconcile against the landed code first** — never fork a second ViewState.
6. Inspect leaves the centre tab strip (four build categories), but stays in the `BuildCategory` data — the right-zone Inspect button dispatches the same ArmTool id its tab did. If wave-1's Tab-cycling assumes five categories, reconcile against the landed code (cycle the four build tabs; Inspect reachable via its button) — do not delete the Inspect tool.
7. `CarePolicies` is ALREADY a v5 save-schema field (save.rs:90) — the Actions panel and `quicksave.civsave` write the existing format; **nothing in this plan touches `SAVE_VERSION`**.
8. Task 4 draws every bar button; Actions/Statistics open no panel until Task 9 and Land/Care/camera are direct-wired until Task 8 — each interim state is stated in Task 4's wiring requirement and each button works (or honestly receipts) at every point. No dead chrome ships in any task.
9. The Actions panel's row set is CLOSED: the three `CarePolicies` levers + Save city + Exit to menu. A new policy field added to the struct without tick-wiring must NOT be surfaced (`actions_rows` is the single gatekeeper — its test pins the row count).

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — no "wire later" tasks exist (Task 4's direct-wired buttons and Task 6's `Show` row name their Task-8/9 re-routes explicitly and ship working either way; note 8 pins the rule)
- [x] Every `Acceptance` criterion names a real non-trivial expected output (printed tick counts, exact receipts, round-tripped pixel coords, `40,000 kr` on a tile, a measured ring Δ ≤ 3 px, the multiplier walk `1.0 -> 1.5 -> 1.8`, the exactly-five Actions rows — never "tests pass")
- [x] Every `Wiring requirement` names an exact function and exact file
- [x] `IMPORTANT NOTES` contains real, code-verified signatures (hud.rs/render_gpu/asset/cook line numbers cited; CarePolicies/CareLedger/CityBudget/GameSave fields + game/mod.rs tick-wiring lines verified 2026-06-10) and names the wave-1 artifacts as external dependencies to verify, not re-create
- [x] `File Map` lists every file that appears in any task (incl. the easily-missed `city_coverage.rs` caller and the new `src/ui/panels.rs`)
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies
- [x] `Done When` names specific commands and specific human-observable results (six named PNGs + the printed W2 checklist + the live `0`-key and policy-receipt behavior)
- [x] All types, method names, and signatures are consistent across tasks (`draw_status_strip`/`SimClock::ticks_due`/`thumbnail`/`draw_world_circle`/`draw_submenu_panel`/`actions_rows` referenced identically throughout)
- [x] Commit steps replaced by "leave uncommitted" everywhere (the user commits)
- [x] Every visual task ends with a Read-the-PNG verification description naming what must be visible
- [x] The execution gate on editor-UX wave 1 is stated at the top with a concrete verification command
- [x] The normative three-zone layout + submenus-above-the-bar rule (`docs/references/ui-layout-spec.md`) is stated in IMPORTANT NOTES and enforced per task (strip = left-zone info, toolbar = three zones, tool panel + Info/Actions/Statistics anchored to their buttons; inspector floating)
- [x] No fake content anywhere: Development ships disabled with the `Progression — coming soon` receipt (W2 step 11 checks no panel opens); Actions rows are only the tick-wired `CarePolicies` + the pre-existing save machinery and exit path (note 9); Statistics is real per-tick numbers, graphs named out of scope
