# Design: Game UI Chrome — CS2-Parity Shell for Civitas Care (2026-06-10)

**Status:** Draft
**Scope:** Translate the Cities: Skylines 2 game-UI anatomy (reference: `~/Ochroma/projects/civitas_care/docs/references/cs2-ui-reference.png`) onto Civitas Care's existing CPU-raster HUD, organized into the **three-zone bottom bar the user mandated** (`~/Ochroma/projects/civitas_care/docs/references/ui-layout-spec.md`, the NORMATIVE layout): **left** = speed cluster + city information + Info/Actions/Development buttons; **center** = the build menu (the widest zone); **right** = tools + Statistics. Real pause/speed, contextual tool panel with priced thumbnails, Actions and Statistics submenu panels carrying only real sim content, in-world measurement feedback — all dispatched through the editor-UX wave-1 command spine. **Universal rule (normative): every submenu/panel pops up ABOVE the main bar, anchored to its invoking button — never sideways, never floating free; the building inspector stays a floating window.** Game layer only (`~/Ochroma/projects/civitas_care`); no engine crates touched.
**Related:** [Main-bar layout spec (normative)](~/Ochroma/projects/civitas_care/docs/references/ui-layout-spec.md) · [Editor/UX North Star](./2026-06-10-editor-ux-north-star-design.md) · [Editor/UX Wave 1 Plan](../plans/2026-06-10-editor-ux-wave1.md) (MUST complete first) · [Living Building Instances](./2026-06-10-living-building-instances-design.md)

---

## 1. Problem Statement

The user uploaded a CS2 screenshot and said: *"Game UI in Cities: Skylines 2 looks like this. I want something similar."* Measured against that reference, today's HUD (`src/render_gpu/hud.rs`, `src/bin/play.rs`) falls short in specific, observable ways:

- **No time controls.** CS2's bottom-left shows `⏸ PAUSED` + speed steps; Civitas Care only advances when the player presses Space (`GameView::key`, play.rs — Space → one `game.tick()`). There is no paused/1×/3× model, so the city never lives on its own and the date/population in the status bar only move on keypress.
- **Status strip is at the top and thin.** CS2 anchors the city vitals (speed, city name, date/season, temperature, population, money) in a full-width bottom strip; ours is a 34 px top bar (`draw_status_bar`, hud.rs:433) with three text slots and no chips, and the care headline (childcare reach %) — the game's core fantasy — is buried inside the `i` info window.
- **Asset tiles are text-only.** CS2's tool panel shows a thumbnail grid with prices and a blue selection border; our `draw_asset_row`/`draw_catalog_asset_row` (hud.rs:610/617) draw flat text buttons with no imagery and no cost, even though the cook already produces 768×768 path-traced renders (`assets/buildings/forge_starter/renders/spectra/*.png` + `contact_sheet.png`) and `pack.json` already carries `visual.preview: null` plus `attributes.build_cost` per asset.
- **In-world tool feedback has no measurements.** CS2 draws white guide lines, dashed extensions, node circles, a radius circle, and floating chips with lengths/angles. Our `draw_zone_draft` (hud.rs:301) has the lines/nodes/dashed close-hint but no length chips, and placing a Kindergarten (coverage_radius_m = 1500, `CareKind::params()`) shows no reach circle at the cursor.
- **Panels are ad-hoc.** Colors/spacing are scattered constants (`ACCENT`, `PANEL`, magic 6/8/12/14/16/24 px offsets across ten draw functions); every rect is sharp-cornered while the reference is uniformly dark-translucent rounded panels with one blue accent. There is no token table, so each new panel re-invents the style.
- **Tools are undiscoverable keys.** Our screenshot is a hidden `p` key and the v/c overlay toggles are undiscoverable keys; the user's layout spec puts these in the bar's **right zone** as visible tool buttons.
- **City management has no surface.** `CarePolicies` (universal childcare / parental leave / elder subsidy — `src/game_mechanics/care/policy.rs`, all three consulted every tick in `CivitasGame::tick` at game/mod.rs:775/864/875 and already persisted in the v5 save schema, save.rs:90) can only be set from test code; the demand/care-ledger/budget numbers are computed every tick (`game.demand()`, `game.last_ledger`, `game.last_budget`) but only visible buried inside the `i` window. The user's layout spec mandates an **Actions** button (left zone) and a **Statistics** button (right zone) opening submenu panels above the bar.

---

## 2. Done When

Running `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play -- --shot-chrome chrome_shots` writes `chrome_shots/chrome_status.png`, `chrome_tool_panel.png`, `chrome_zone_draft.png`, `chrome_overlays.png`, `chrome_actions.png`, `chrome_stats.png` and prints one `wrote …` line per file; opening `chrome_status.png` a human sees the three-zone anatomy — a full-width **bottom** bar whose **left zone** holds a `⏸/▶/▶▶` speed cluster reading `Paused`, the city-name, `Care NN%`, `Pop N`, and `NNN,NNN kr` chips on rounded dark panels plus `Info` / `Actions` / `Development` buttons (Development visibly dimmed), whose **centre zone** (the widest) holds the four build-category tabs, and whose **right zone** holds the Inspect / Land / Care / camera / `Statistics` tool buttons. Opening `chrome_tool_panel.png` sees the Care tool panel popped **above the bar anchored to its tab** with a priced thumbnail/text tile grid (`Kindergarten · 40,000 kr`), a blue border on the armed tile, an `x` close button, and a blue coverage-radius ring around the cursor. Opening `chrome_actions.png` sees the Actions panel above its button listing exactly the three real `CarePolicies` rows (`Universal childcare`, `Parental leave`, `Elder subsidy`) plus `Save city` and `Exit to menu` — nothing else. Opening `chrome_stats.png` sees the Statistics panel above its button with the real demand figures, care-ledger lines, and budget summary the info window already computes. In a live `cargo run --release --bin play`, pressing `0` prints the receipt `Speed: 1×` in the strip and the date/population then advance with **no further input**; pressing `0` twice more returns to `Speed: Paused` and the date freezes; clicking Actions → `Universal childcare` prints the receipt `Policy: Universal childcare ON` and the next tick's care upkeep is 1.5× the un-subsidised figure.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Sim clock (paused/1×/3×) | `ui::sim_clock::tests::ticks_accumulate_by_speed`: `ticks_due(2.0)` returns 0 (Paused), 1 (Normal @ 0.5 t/s), 3 (Fast @ 1.5 t/s); fractional ticks carry across calls; a 60 s stall caps at 4; prints all counts | `assert!(clock.ticks_due(1.0) >= 0)` — always true |
| Speed commands dispatch | `command::tests::speed_cycle_emits_receipts`: dispatching `sim.speed.cycle` ×3 yields receipts `Speed: 1×`, `Speed: 3×`, `Speed: Paused` and `view.sim_speed` round-trips | asserting dispatch returned `Ok` |
| Bottom status strip geometry | `hud::tests::status_strip_speed_buttons_hit_test`: `speed_button_at` returns `Some(0/1/2)` at each drawn button centre, `None` 4 px above the strip; prints the three centres | `assert!(speed_button_at(..).is_some())` on (0,0) |
| Season derivation | `hud::tests::season_labels_match_day`: day 172→`Summer`, 30→`Winter`, 100→`Spring`, 300→`Autumn`; printed | asserting the fn returns a non-empty str |
| Cooked thumbnails load | `asset::tests::catalog_loads_cooked_thumbnails`: after `--thumbs-only` cook, `catalog.thumbnail("forge.house.craftsman")` is `Some` with ≥3 distinct RGB values (a real image, not a flat fill); prints dims + distinct-color count | `assert!(thumb.is_some())` against an empty map |
| Tool panel grid + prices | `hud::tests::tool_panel_rows_hit_test`: `tool_panel_thumb_at` round-trips every drawn tile centre; the rendered Kindergarten tile contains the exact substring price `40,000 kr` rendered (probe: text pixels differ from panel bg at the price baseline); prints the roundtrip count | function returns `()` |
| Measure chips carry real metres | `hud::tests::measure_chip_prints_distance`: a 25 m draft edge renders a chip whose label is `25 m` (label computed from world coords, asserted before raster); prints the label | asserting a chip rect was filled |
| Radius ring at world scale | `--shot-chrome` writes `chrome_tool_panel.png`; the ring's projected pixel radius equals `world_to_screen(cursor±1500 m)` span within 3 px (asserted in the harness, printed) | drawing a fixed 100 px circle |
| Actions panel mutates the real sim | `command::tests::policy_toggles_move_the_multiplier`: dispatching `policy.childcare.universal` flips `game.policies.universal_childcare` and `operating_cost_multiplier()` walks 1.0→1.5; `policy.elder.subsidy` → 1.8; `policy.leave.cycle` makes `infant_demand_exempt(0.3)` true; receipts exactly `Policy: Universal childcare ON`, `Policy: Elder subsidy ON`, `Policy: Parental leave 0.5 yr`; all printed | asserting dispatch returned `Ok` |
| Statistics panel carries the real ledger | `ui::panels::tests::stats_lines_carry_the_ledger`: after one `game.tick()`, the demand line contains the same `{:.2}` figures as `game.demand()`, the care line the same net as `game.last_ledger.net()`, the budget line the same net as `game.last_budget.net()`; lines printed | `assert!(!lines.is_empty())` |
| W2 walkthrough twin | `walkthrough::tests::w2_runs_headless` prints `[x]`/`[~]` per step ending `W2 headless: 11/11 steps (4 manual)` | running commands without checks |

---

## 4. Architecture

### 4.1 Layout anatomy — what we adopt, adapt, defer

**Normative layout directive (user, 2026-06-10 — supersedes any earlier zone placement in this doc):** the bar has three zones — **left**: city information + buttons opening Info / Actions / Development tree; **center**: the zoning + service-buildings build menu (the widest zone); **right**: tools + Statistics. **Every submenu pops up above the main bar, anchored to its invoking button.** Source: `~/Ochroma/projects/civitas_care/docs/references/ui-layout-spec.md`. The bar keeps its two stacked bands (`STRIP_H` row at the very bottom, `TOOLBAR_H` row above it — exactly the CS2 reference's stacking); the three zones are vertical columns spanning both bands.

Mapping every element visible in the reference screenshot:

| CS2 element (reference) | Decision | Civitas Care translation |
|---|---|---|
| Bottom status strip: pause/speed cluster, city name, date + season, temperature, population, money | **Adopt (adapted)** | New full-width 40 px strip at the **bottom** (replaces the 34 px top bar). **Left zone:** `⏸/▶/▶▶` 3-button speed cluster (Paused/1×/3×, active = accent), then the city-information chips — city name (click = info window, same dispatch as the Info button; lit while open), **childcare-reach chip `Care NN%`** (care-first — CS2 has no equivalent; it gets first-class placement), `Pop N` chip, funds chip in kr. Centre: `Day N · HH:MM · <Season>` — season derived from `Observer.day_of_year` (Winter <60 or ≥335, Spring 60–151, Summer 152–243, Autumn 244–334); the centre slot keeps the wave-1 precedence contract verbatim: zoning prompt > preview caption > sticky receipt > date line (so the date stays readable in the bar whenever nothing transient outranks it — the fold that keeps name/pop/funds/date all readable per the layout spec). Strip-right stays clear (the toolbar row's right zone above it carries the tools). **Temperature: deferred** — no weather/temperature simulation exists; displaying an invented number is dishonest chrome. |
| Toolbar row: city chip + milestone/XP progress bars + icon category buttons | **Adapt (three zones)** | Toolbar row sits directly above the strip (existing `CAT_H` dock restyled). **Left zone:** three buttons — `Info` (opens the existing city-information content as a submenu panel above the bar, anchored to this button), `Actions` (opens the Actions submenu panel — real content only, §4.6), `Development` (**present but disabled**: dimmed `MUTED` treatment, hover tooltip and click receipt `Progression — coming soon`; NO panel, NO fake tree content — the button placement is the user's explicit ask, the tree ships when a progression system exists, own design) — followed by the five demand mini-bars (R/C/I/O/F, 6 px columns; the floating bottom-left `Demand` panel is retired into this slot — demand is city information, so it lives in the left zone). **Centre zone (widest):** the four **build** categories (Zones/Care/Services/Utilities) as icon+label tabs (glyph box + accent underline, both already drawn — restyled to tokens); Inspect leaves the category list for the right zone. **Right zone:** the tool cluster — `Inspect` (arms `BuildTool::Inspect` via its existing ArmTool command id), `Land`/`Care` overlay toggles (accent-lit when on), camera (screenshot), and the `Statistics` button (opens the Statistics submenu panel above the bar, §4.6). **Milestone/XP bars: deferred** — no progression system exists yet; the Development button IS the reserved progression slot, so wiring the tree later is purely additive. |
| Contextual tool panel: labelled option rows left (Tool Mode / Elevation / Parallel Mode), asset thumbnail grid with prices right, blue selection border, close X | **Adapt** | Panel opens **above the bar, x-anchored to the invoking category tab** (clamped on-screen — the universal submenu rule) when a category is selected. Left option column gets **tool-relevant** rows, not CS2's road rows: for Zones a `Growable` row (Auto / pinned catalog asset — today's catalog-pin row relocated); for Care/Services a `Show` row exposing the matching overlay (land value / care reach). **Elevation / Parallel Mode: deferred** — there is no road tool; inventing those rows would be cargo-culting. Right: the category's tools as a tile grid — thumbnail (when cooked) or text fallback, name, **price** (`CareKind::params().build_cost` / `CityServiceKind::build_cost()`; zone tiles show no price because zoning charges nothing at commit — lots are charged at develop time, see wave-1 note 6). Armed tile gets a 2 px accent border + lighter top bar (existing selection treatment, tokenized). Close `x` top-right collapses the panel (dock stays; re-clicking the active category reopens). |
| In-world feedback: white guide lines, dashed extensions, node circles, radius circle, floating measurement chips (length/angle) | **Adopt (generalize)** | `draw_zone_draft` already draws committed edges, node discs, the dashed close hint, and the rubber band — kept. Added: a **measure chip** at each edge midpoint showing real world length (`NN m`, computed from the draft's world XZ coords, projected via `world_to_screen`), including the live rubber-band edge; and a **coverage-radius ring** (blue, 64 segments through the existing `line` raster) around `cursor_world` whenever a Care tool is armed, at true world scale (`coverage_radius_m`). The wave-1 ghost (`draw_tool_preview`: red/green verdict disc + caption chip) is the CS2 floating-chip equivalent for placement verdicts and is reused untouched. **Angle chips: deferred** — angles are road-geometry feedback; zone polygons have no snapping angles worth reporting yet. |
| Top-corner utility buttons (round, blue) | **Relocate (right zone)** | The layout spec puts tools in the bar's right zone, so the screenshot button (camera glyph → `view.screenshot`) and the two overlay toggles (`view.overlay.land`, `view.overlay.care`, accent-lit when active) live in the toolbar's right-zone tool cluster — there are NO top-corner clusters. **Bulldoze: deferred** — an undo-aware demolish needs a new `AuthoredAction::Demolish` and a save-schema bump (wave 1 froze `SAVE_VERSION = 5`); shipping a bulldoze button that secretly calls `edit.undo` would lie about what it removes. Ctrl+Z already covers "remove the last thing"; real bulldoze is its own wave. |
| Submenus over the bar (normative layout spec, not a CS2 crib) | **Adopt (universal rule)** | Every submenu/contextual panel — the Info panel, the Actions panel, the Statistics panel, the contextual tool panel, overlay legends' stacking unchanged — renders **above the main bar with its bottom edge at the toolbar's top, x-anchored to the invoking button/tab** (clamped to the screen). Never sideways, never floating free. At most one of Info/Actions/Statistics is open at a time (opening one closes the others). **Exception, per the directive: the building inspector stays a floating window** (it is about a world object, not the bar). |
| Dark-translucent rounded panels, single blue accent | **Adopt** | One token table (§4.2) + a `fill_rounded` raster primitive; every chrome panel draws through tokens. |

### 4.2 Rendering approach — keep the CPU raster, pin the tokens

The HUD stays exactly what it is: pure CPU raster (`ab_glyph` text + alpha blends straight onto the frame's `[[u8;4]]`), because it is deterministic (same input → same pixels, byte-for-byte), testable per-pixel in plain `cargo test --lib` with no GPU, and already carries draw/hit-test pairs for every control. No egui, no GPU UI pass. What changes is discipline: a `theme` module in hud.rs pins the palette/spacing/iconography so panels stop being ad-hoc:

| Token | Value | Use |
|---|---|---|
| `PANEL` | (13, 17, 24) @ α 0.90 | strip, toolbar, tool panel base |
| `PANEL_RAISED` | (24, 30, 40) @ α 0.96 | buttons, chips |
| `ACCENT` | (52, 136, 188) | armed/selected fills, ring |
| `ACCENT_HI` | (154, 220, 255) | selection underline / border |
| `ACCENT_DARK` | (24, 72, 104) | strip top rule |
| `TEXT` / `MUTED` | (232, 238, 246) / (172, 184, 198) | labels / secondary |
| `GOOD` / `BAD` | (104, 196, 112) / (220, 50, 40) | funds, valid ghost / blocked ghost, uncovered |
| `RADIUS` | 6 px | every rounded panel corner |
| `SP1 / SP2 / SP3` | 6 / 12 / 24 px | intra-chip / inter-chip / section gaps |
| `STRIP_H / TOOLBAR_H / PANEL_H` | 40 / 48 / 96 px | the three bottom bands |

`fill_rounded(px, w, h, x0, y0, rw, rh, r, rgb, a)` = the existing `fill` plus a per-corner circle test (skip pixels outside radius `r` discs at the four corners) — O(area), same cost class as `fill`. Iconography stays geometric-raster, honestly scoped: pause = two `fill` bars, play = triangle via `fill_world_quad` with a repeated corner (convex-quad code accepts degenerate quads), fast = two triangles, camera = rect + disc, category glyphs keep the existing single-letter boxes (`category_icon`). No icon-font or SVG pipeline this wave.

### 4.3 Asset thumbnails pipeline — crop the existing renders (decision)

**Decision: derive thumbnails from the already-cooked path-traced renders; do NOT add a dedicated per-asset render pass.** Cost analysis, honestly: a dedicated render means a Spectra path-trace per asset (minutes each on the 780M, and the cook then requires the GPU box + Forge), or building a new offline rasterizer harness — both unjustified for tile-sized images. The cook already produces 768×768 iso renders for the starter archetypes (`renders/spectra/cooked_house_craftsman.png`, `house_victorian.png`, `police_station.png`, `school.png`, `craftsman/iso.png`, `rowhouse/iso.png`) that read clearly at thumbnail size (verified by eye against the actual PNGs).

Mechanics: `game_asset_cook --thumbs-only` (a new light mode that **never invokes Forge/AssemblyPrime**) reads a static recipe-id → render-path map, centre-crops and Lanczos-downscales to 96×72, writes `assets/buildings/forge_starter/thumbs/<asset_id>.png`, and patches each asset's existing `visual.preview` field (currently `null`, `AssetVisualRef::preview: Option<String>`, asset/mod.rs:168) with the pack-relative path. Variant assets (`.v1`/`.v2`, today's per-asset variant pools) reuse the base asset's thumbnail — the same base-before-variant reuse the cook applies to AssemblyPrime exports. At load, `append_asset_packs_from_dir` already resolves pack-relative paths against the pack.json's parent dir for atoms (asset/mod.rs:731); thumbnail loading rides the identical pattern into a `thumbnails: HashMap<String, AssetThumb>` on `AssetCatalog` (96×72 RGBA ≈ 27 KB each — trivial). **Coverage is partial and that is fine:** procedural style-pack zonables and Care kinds have no renders; their tiles keep the current text treatment (name + price). The pipeline is monotone — every newly cooked render that gets a map entry upgrades its tile with zero UI changes.

### 4.4 In-world feedback — world pass geometry, screen-space labels, one rasterizer

All in-world feedback follows the pattern `draw_zone_draft` proved: geometry lives in world space, is projected per-frame with `render_gpu::world_to_screen(view, proj, x, z, w, h) -> Option<(f32,f32)>`, and is stamped by the hud rasterizer onto the same pixel buffer after the GPU city frame — so feedback is always camera-correct and never needs a GPU UI pass. Generalizations: `draw_measure_chip(px, w, h, anchor, label)` (rounded dark backing sized by `text_width`, the `draw_tool_preview` caption idiom) called at each draft-edge midpoint with `label = format!("{:.0} m", world_len)`; `draw_world_circle(px, w, h, view, proj, center_xz, radius_m, rgb, a)` projecting 64 chord points through the existing 2 px `line`. The wave-1 ghost (`draw_tool_preview`) supplies the verdict chip (`Blocked — clearance <c> m` / `≈ N parcels`) — chrome adds no second verdict source (invariant I1 stands: the chip text comes from `tool_preview`/`plop_blocked`).

### 4.5 Time controls — a real sim-speed model, replay-safe by construction

Today `play.rs` runs `ControlFlow::Wait` and only ticks on Space. Design: a lib-side `SimSpeed { Paused, Normal, Fast }` + `SimClock { accum: f64 }` with `ticks_due(&mut self, dt_s: f64, speed: SimSpeed) -> u32` — `accum += dt * rate` (Paused 0.0, Normal 0.5, Fast 1.5 ticks/s; one tick = one sim month, so 1× ≈ 2 s/month), drain whole ticks, **cap 4 ticks per frame** (a stalled window must not replay a burst). The bin holds the wall-clock (`Instant`), the speed lives in `ViewState.sim_speed` (default **Paused**, matching the reference screenshot and keeping every existing test/harness semantics-identical), and each due tick is dispatched as the wave-1 **`sim.tick` command** — never a direct `game.tick()`. That is the replay-safety argument in one line: ticks remain the sole authoritative time unit; the clock only decides *when* to dispatch the same command Space dispatches, so saves, undo (truncate-log-and-replay), the headless walkthrough twin, and all `--shot` harnesses (which press Space) are untouched. Event loop: `ControlFlow::WaitUntil(now + 100 ms)` while unpaused (drive `ticks_due` + redraw from `about_to_wait`), back to `Wait` when paused — no busy poll. Keys: `0` cycles Paused → 1× → 3× → Paused (`sim.speed.cycle`; digits 1–9 are taken by tool hotkeys); the strip's three buttons dispatch `sim.speed.pause/normal/fast` directly. **Space keeps meaning `sim.tick` (step once)** — changing it would break the `--shot`/walkthrough harnesses and single-step is genuinely useful while paused.

### 4.6 Left-zone buttons, right-zone tools, submenu panels — scoped against what exists

**One submenu shell, three users.** `draw_submenu_panel` (hud.rs) is the rounded `theme::PANEL` window with title bar + close `x`, drawn with its bottom edge at `h - STRIP_H - TOOLBAR_H - SP1` and its x clamped around the invoking button's centre — the universal above-the-bar anchoring in one place. Info, Actions, and Statistics all render through it; the contextual tool panel keeps its own grid layout but obeys the same anchor rule (§4.1). The **inspector keeps the floating centred `draw_city_info_window`** — it is a world-object window, exempted by the directive.

**Info (left zone).** The button dispatches the existing `view.info` command; the window's *content* is unchanged (the `info_lines()` the `i` key already shows) — what changes is *where*: it renders through `draw_submenu_panel` anchored to the Info button instead of centred mid-screen. The strip's city-name chip dispatches the same id (both light while open).

**Actions (left zone) — real content only, inventoried against the code.** Every row is wired to the sim today or it does not appear:
- `Universal childcare` toggle → `game.policies.universal_childcare`. Wired: `operating_cost_multiplier()` (+0.5) is applied to care upkeep every tick (game/mod.rs:875) and `satisfaction_bonus()` (+0.05) every tick (:864).
- `Parental leave` stepper cycling `0 → 0.5 → 1.0 → 2.0 → 0 yr` → `game.policies.parental_leave_years`. Wired: infants under the threshold are exempted from external care demand every tick (game/mod.rs:775-777) + satisfaction bonus.
- `Elder subsidy` toggle → `game.policies.elder_subsidy`. Wired: `operating_cost_multiplier()` (+0.3) + `satisfaction_bonus()` (+0.04) every tick.
- `Save city` → writes `game.to_save().to_json()` (both exist today: game/mod.rs:1246, save.rs:98) to `<saves_dir>/quicksave.civsave` — the same dir the menu's Load screen already scans and lists (menu/saves.rs). All three policies are already fields of the v5 save schema (save.rs:90) — **no `SAVE_VERSION` change**. The button is the save system's first in-game trigger; the format, writer, and loader all pre-exist.
- `Exit to menu` → the existing Esc-ladder exit (play.rs:1011 `self.game = None`), surfaced as a button.

Explicitly NOT in the Actions panel: tax-rate editing (`TaxPolicy` is real and consulted, but the directive scopes Actions to care policies + the save/menu actions — a tax UI is its own design), and any policy not consulted by `CivitasGame::tick`. There are no such phantom policies today; if one is added to the struct before it is wired, it must NOT appear here.

**Development (left zone) — button present, content honestly absent.** The user's directive places the button NOW; no progression system exists, so the button renders dimmed (`MUTED`), hover draws a tooltip chip `Progression — coming soon` (the measure-chip backing), and a click emits the same line as a status receipt. No panel, no fake tree. When progression lands (own design), the button gains its submenu purely additively.

**Statistics (right zone) — the real numbers, no graphs.** The panel shows, via `ui::panels::stats_lines(&CivitasGame, &CareStats)`, exactly what the sim already computes per tick: the five demand figures in detail (`game.demand()` — residential/commercial/industry/office/farming), the care ledger (`game.last_ledger: CareLedger { labor_tax, care_upkeep }` + `net()`/`is_surplus()`, plus gated workers / unmet care / care sites / reach % from `CareStats`), and the budget summary (`game.last_budget: CityBudget` — revenue/spending/net with the taxes/fees/trade/power lines the info window already formats at play.rs:365-366). **Graphs/time-series are out of scope this wave** — there is no history buffer; charting one tick would be dishonest. Numbers only, said plainly on the panel ordering.

**Right-zone tools.** Screenshot button → `view.screenshot` command. Dispatch can't render (the renderer lives in the bin), so the arm sets `view.shot_requested = true` with receipt `Screenshot queued`; `GameView::render`'s caller in the bin consumes the flag after the next presented frame and writes via the existing `next_coverage_shot_path()`/`save_rgba` path (`p` key re-routes through the same command). `view.overlay.land` / `view.overlay.care` toggle buttons — `land_overlay`/`coverage_overlay` move from `GameView` fields into `ViewState` (additive `bool` fields) so the buttons, the `v`/`c` keys, and the headless twin share one dispatch path; buttons render accent-lit when active. `Save city`/`Exit to menu` follow the same flag idiom (`view.save_requested`/`view.exit_requested`, consumed by the bin — the bin owns the saves dir and the `self.game = None` transition). Bulldoze: explicitly out (see §4.1 table) — `AuthoredAction::Demolish` + `SAVE_VERSION 6` is its own future wave.

### 4.7 Command-spine integration (the wave-1 contract)

Every chrome control dispatches through `GameRegistry::dispatch(&mut ViewState, &mut CivitasGame, &SpatialFieldRegistry, id)` — no control calls game logic around the registry. New command ids (additive `UiAction` variants): `sim.speed.cycle` (`0`), `sim.speed.pause`, `sim.speed.normal`, `sim.speed.fast`, `view.screenshot` (`p`), `view.overlay.land` (`v`), `view.overlay.care` (`c`), `view.panel.close`, `view.actions` (toggle the Actions panel), `view.stats` (toggle the Statistics panel), `view.development` (receipt-only — opens NOTHING; the honest disabled-button dispatch), `policy.childcare.universal`, `policy.elder.subsidy`, `policy.leave.cycle`, `game.save`, `game.exit_menu`. Receipts follow I11 (`Speed: 1×`, `Overlay: Land value ON`, `Screenshot queued`, `Opened: actions`/`Closed: actions`, `Opened: statistics`/`Closed: statistics`, `Progression — coming soon`, `Policy: Universal childcare ON/OFF`, `Policy: Elder subsidy ON/OFF`, `Policy: Parental leave 0.5 yr`/`… off`, `Save queued`, `Exiting to menu`). The Esc `CloseTop` ladder is **not** extended this wave (the tool panel and the Info/Actions/Statistics submenus close via their `x` / button re-click only) so wave-1's `view.close_top` tests stay byte-stable. The walkthrough grows `script(2)` (chrome checklist) beside `script(1)`, printed via `--walkthrough 2`, with a `w2_runs_headless` twin.

---

## 5. Data Models

```rust
/// src/ui/sim_clock.rs — lib-side, winit-free, fully testable headless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimSpeed {
    #[default]
    Paused,   // boot state — matches the CS2 reference and freezes all harnesses
    Normal,   // 0.5 ticks/s  (one sim month ≈ 2 s)
    Fast,     // 1.5 ticks/s
}

/// Wall-clock → whole-tick accumulator. The ONLY thing it ever does is decide
/// how many `sim.tick` commands to dispatch; ticks stay the authoritative unit.
#[derive(Debug, Clone, Default)]
pub struct SimClock {
    accum: f64, // private — fractional ticks carried between frames
}

impl SimClock {
    /// Whole ticks due after `dt_s` seconds at `speed`; drains the accumulator,
    /// capped at 4 per call so a stalled window never replays a burst.
    pub fn ticks_due(&mut self, dt_s: f64, speed: SimSpeed) -> u32;
}

/// src/asset/mod.rs — decoded thumbnail, loaded once per pack at catalog load.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetThumb {
    w: u32,
    h: u32,
    rgba: Vec<u8>, // w*h*4 — private; .pixels() accessor for the hud blit
}

// AssetCatalog gains (private field + accessor):
//   thumbnails: HashMap<String, AssetThumb>   // asset id → decoded preview
//   pub fn thumbnail(&self, id: &str) -> Option<&AssetThumb>

// ViewState (src/command/mod.rs, wave-1 type) gains additive fields:
//   pub sim_speed: SimSpeed,
//   pub overlay_land: bool,      // moved from GameView.land_overlay
//   pub overlay_care: bool,      // moved from GameView.coverage_overlay
//   pub panel_open: bool,        // contextual tool panel collapsed via its X
//   pub shot_requested: bool,    // set by view.screenshot, consumed by the bin
//   pub actions_open: bool,      // Actions submenu (opening closes stats/info)
//   pub stats_open: bool,        // Statistics submenu (opening closes actions/info)
//   pub save_requested: bool,    // set by game.save, consumed by the bin (owns saves_dir)
//   pub exit_requested: bool,    // set by game.exit_menu, consumed by the bin (game = None)

// UiAction (wave-1 enum) gains additive variants:
//   SetSpeed(SimSpeed), CycleSpeed, Screenshot,
//   ToggleOverlay(OverlayKind), ClosePanel,
//   ToggleActions, ToggleStats,
//   Development,                            // receipt-only: "Progression — coming soon"
//   TogglePolicy(PolicyKind), CycleLeave,   // mutate game.policies directly (sim-wired)
//   SaveCity, ExitToMenu                    // flag idiom, like Screenshot
// #[derive(Debug, Clone, Copy, PartialEq)] pub enum OverlayKind { LandValue, CareReach }
// #[derive(Debug, Clone, Copy, PartialEq)] pub enum PolicyKind { UniversalChildcare, ElderSubsidy }
```

---

## 6. API

```rust
// ===== src/render_gpu/hud.rs (all drawing lives here — text/fill/text_width are private to this file) =====

/// Rounded-rect fill: `fill` + four corner-disc tests. r <= min(rw, rh)/2.
fn fill_rounded(px: &mut [[u8;4]], w: usize, h: usize,
                x0: i32, y0: i32, rw: i32, rh: i32, r: i32, rgb: (u8,u8,u8), a: f32);
// (private, like fill — public surface below)

/// Season label from day-of-year: <60|>=335 Winter, 60..152 Spring, 152..244 Summer, 244..335 Autumn.
pub fn season_label(day_of_year: u32) -> &'static str;

/// The bottom status strip (STRIP_H=40, full width, y = h - STRIP_H).
/// LEFT zone: speed cluster + city-info chips (name, Care %, Pop, kr);
/// centre: the precedence line; strip-right stays clear (tools live in the toolbar's right zone).
pub struct StatusStrip<'a> {
    pub speed: SimSpeed,
    pub city: &'a str,
    pub center: &'a str,     // zoning prompt > preview caption > receipt > "Day N · HH:MM · Season"
    pub care_pct: f32,       // 0..100, the childcare-reach headline — LEFT-zone chip
    pub population: u32,     // LEFT-zone chip
    pub funds: f64,          // LEFT-zone chip
    pub info_hot: bool,      // city chip lit while the info panel is open
}
pub fn draw_status_strip(px: &mut [[u8;4]], w: u32, h: u32, s: &StatusStrip<'_>);
pub fn speed_button_at(w: u32, h: u32, x: f32, y: f32) -> Option<usize>;   // 0=pause 1=normal 2=fast
pub fn strip_city_chip_at(w: u32, h: u32, x: f32, y: f32) -> bool;          // replaces status_info_button_at

/// Toolbar row above the strip — the three zones:
/// left: Info/Actions/Development buttons + demand minis · centre: 4 build-category tabs (widest)
/// · right: Inspect, Land, Care, camera, Statistics.
/// `left_hot` = open-state for Info/Actions (Development always renders dimmed);
/// `tools` = (label, lit) for the right-zone cluster.
pub fn draw_toolbar(px: &mut [[u8;4]], w: u32, h: u32,
                    demand: &[(&str, f32, (u8,u8,u8))], cats: &[(&str, bool)],
                    left_hot: [bool; 2], tools: &[(&str, bool)]);
pub fn toolbar_category_at(w: u32, h: u32, x: f32, y: f32, n: usize) -> Option<usize>;
pub fn toolbar_left_button_at(w: u32, h: u32, x: f32, y: f32) -> Option<usize>; // 0=Info 1=Actions 2=Development
pub fn toolbar_tool_at(w: u32, h: u32, x: f32, y: f32, n: usize) -> Option<usize>; // 0=Inspect 1=Land 2=Care 3=Shot 4=Stats

/// The shared submenu shell — EVERY submenu anchors ABOVE the bar at its invoking
/// button (universal rule): bottom edge at h - STRIP_H - TOOLBAR_H - SP1, x clamped
/// around `anchor_x`. Used by the Info, Actions, and Statistics panels.
pub fn draw_submenu_panel(px: &mut [[u8;4]], w: u32, h: u32, anchor_x: f32,
                          title: &str, lines: &[String]);
pub fn submenu_close_at(w: u32, h: u32, anchor_x: f32, lines: usize, x: f32, y: f32) -> bool;
pub fn submenu_contains(w: u32, h: u32, anchor_x: f32, lines: usize, x: f32, y: f32) -> bool;
/// Actions panel: submenu shell + clickable rows ((label, value, on), e.g.
/// ("Universal childcare", "ON", true) / ("Parental leave", "1.0 yr", true)).
pub fn draw_actions_panel(px: &mut [[u8;4]], w: u32, h: u32, anchor_x: f32,
                          rows: &[(String, String, bool)]);
pub fn actions_row_at(w: u32, h: u32, anchor_x: f32, n: usize, x: f32, y: f32) -> Option<usize>;

/// Contextual tool panel: left option rows, right tile grid (thumb-or-text + name + price).
/// `anchor_x` = the invoking category tab's centre (universal rule: the panel pops
/// above the bar anchored to its tab, clamped on-screen) — shared by draw + hit tests.
pub struct ToolPanelTile<'a> {
    pub label: &'a str,
    pub price: Option<f64>,             // None for zones (charged at develop time)
    pub thumb: Option<&'a AssetThumb>,  // None → text tile
    pub selected: bool,
}
pub fn draw_tool_panel(px: &mut [[u8;4]], w: u32, h: u32, anchor_x: f32,
                       option_rows: &[(&str, &[(String, bool)])],  // ("Growable", [("Auto", true), ..])
                       tiles: &[ToolPanelTile<'_>]);
pub fn tool_panel_thumb_at(w: u32, h: u32, anchor_x: f32, x: f32, y: f32, n: usize) -> Option<usize>;
pub fn tool_panel_option_at(w: u32, h: u32, anchor_x: f32, x: f32, y: f32, rows: &[usize]) -> Option<(usize, usize)>;
pub fn tool_panel_close_at(w: u32, h: u32, anchor_x: f32, x: f32, y: f32) -> bool;

/// In-world feedback.
pub fn draw_measure_chip(px: &mut [[u8;4]], w: u32, h: u32, anchor: (f32, f32), label: &str);
#[allow(clippy::too_many_arguments)]
pub fn draw_world_circle(px: &mut [[u8;4]], w: u32, h: u32, view: Mat4, proj: Mat4,
                         center_xz: [f32; 2], radius_m: f32, rgb: (u8,u8,u8), a: f32);

// ===== src/ui/panels.rs (lib-side, GPU-free testable) =====
/// The Actions rows, derived ONLY from the real CarePolicies + the two host actions
/// (Save city / Exit to menu). The single gatekeeper of the panel's content.
pub fn actions_rows(p: &CarePolicies) -> Vec<(String, String, bool)>;
/// The Statistics lines: demand detail (game.demand()), care ledger (game.last_ledger),
/// budget summary (game.last_budget) — the same numbers the info window computes. No graphs.
pub fn stats_lines(game: &CivitasGame, stats: &CareStats) -> Vec<String>;

// ===== src/ui/sim_clock.rs =====
pub fn ticks_due(&mut self, dt_s: f64, speed: SimSpeed) -> u32;  // see §5; pure, no clock inside

// ===== src/asset/mod.rs =====
impl AssetCatalog { pub fn thumbnail(&self, id: &str) -> Option<&AssetThumb> }
impl AssetThumb { pub fn size(&self) -> (u32, u32); pub fn pixels(&self) -> &[u8] }

// ===== src/bin/game_asset_cook.rs =====
// `--thumbs-only`: no Forge/AssemblyPrime discovery; read pack.json + renders/spectra,
// write thumbs/<asset_id>.png (96×72), set visual.preview, rewrite pack.json.
// Prints: `thumbs: <K> written, <M> assets keep text tiles`.
// Threading: all of the above is single-threaded CPU raster on the frame buffer —
// called from GameView::render / the cook main, no Send/Sync requirements.
```

Deletions: `draw_status_bar` + `status_info_button_at` (top bar retired — `city_coverage.rs` and `play.rs` are the only external callers of the bottom-bar kit and both migrate), `draw_category_bar`/`category_at` (subsumed by `draw_toolbar`), `draw_asset_row`/`draw_catalog_asset_row` + their hit tests (subsumed by `draw_tool_panel`), the standalone `draw_demand_bars` panel call in play.rs (the function stays for the toolbar's mini-bars). `BAR_H` reserved-space arithmetic in `draw_overlay_legend` (hud.rs:246) is re-based on `STRIP_H + TOOLBAR_H + PANEL_H`. The city-info call site in play.rs (`:666`) migrates from the centred `draw_city_info_window` to `draw_submenu_panel` (anchored to the Info button) — `draw_city_info_window` + `info_window_close_at`/`info_window_contains` **stay** for the inspector's floating window, which is exempt from the anchoring rule.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `SimClock::ticks_due` | `App::about_to_wait` (new winit arm) | `src/bin/play.rs` | each due tick → `run_command(gv, w, h, "sim.tick")` |
| `sim.speed.*` dispatch arms | `GameRegistry::dispatch` | `src/command/mod.rs` | mutate `view.sim_speed`, receipt `Speed: …` |
| `draw_status_strip` | `GameView::render`, after the dock | `src/bin/play.rs` | replaces the `draw_status_bar` call inside `draw_bottom_bar_ui` |
| `speed_button_at` / `strip_city_chip_at` | `MouseInput` Left-press arm | `src/bin/play.rs` | route to `sim.speed.*` / `view.info` command ids |
| `draw_toolbar` | `GameView::render` | `src/bin/play.rs` | demand minis from `game.demand()`; centre tabs = the 4 build categories from `category_items` (Inspect excluded); left buttons + right tool cluster drawn here |
| `toolbar_left_button_at` / `toolbar_tool_at` | `MouseInput` Left-press arm | `src/bin/play.rs` | left → `view.info` / `view.actions` / Development receipt; right → Inspect ArmTool id / `view.overlay.*` / `view.screenshot` / `view.stats` |
| `draw_tool_panel` + hit tests | `GameView::render` / `MouseInput` | `src/bin/play.rs` | tiles dispatch the wave-1 ArmTool command ids; Growable row keeps `select_catalog_asset` (direct-wired, per wave-1 note); `x` → `view.panel.close` |
| `--thumbs-only` mode | `run()` arg branch | `src/bin/game_asset_cook.rs` | pure file transform; no Forge |
| thumbnail load | `load_ready_asset_payloads` sibling | `src/asset/mod.rs` | same `base_dir.join(rel)` resolution as atoms (mod.rs:731) |
| `draw_measure_chip` (edge lengths) | `GameView::render`, beside `draw_zone_draft` | `src/bin/play.rs` | world lengths from `view.draft`, projected per edge |
| `draw_world_circle` (care radius) | `GameView::render`, when Care armed + `cursor_world` | `src/bin/play.rs` | radius = `CareKind::params().coverage_radius_m` |
| `view.shot_requested` consumption | `App::redraw`, after present | `src/bin/play.rs` | existing `next_coverage_shot_path` + `save_rgba` |
| `draw_submenu_panel` (Info/Actions/Stats) | `GameView::render`, after the bar | `src/bin/play.rs` | Info = existing `info_lines()`; Actions = `ui::panels::actions_rows(&game.policies)` + Save/Exit rows; Stats = `ui::panels::stats_lines(&game, &stats)` |
| `policy.*` dispatch arms | `GameRegistry::dispatch` | `src/command/mod.rs` | mutate `game.policies` directly (sim consults it next tick); receipts `Policy: …` |
| `game.save` / `game.exit_menu` flags | dispatch sets; `App::redraw`/event loop consumes | `src/command/mod.rs` / `src/bin/play.rs` | save: `game.to_save().to_json()` → `saves_dir/quicksave.civsave` (the dir the menu scans); exit: the existing `self.game = None` path |
| `walkthrough::script(2)` / `print(2)` / `run(2)` | `--walkthrough 2` arm / `w2_runs_headless` | `src/bin/play.rs` / `src/walkthrough/mod.rs` | extends the wave-1 module, same Check enum |
| `--shot-chrome <dir>` | `main()` arg branch beside `--shot` | `src/bin/play.rs` | writes the six chrome proof PNGs (status, tool panel, zone draft, overlays, actions, stats) |

---

## 8. Open Questions

All resolved at design time:

- ~~Where does temperature come from?~~ → Deferred; no weather sim exists (§4.1).
- ~~Dedicated thumbnail renders or crops?~~ → Crops of existing cooked renders; `--thumbs-only` cook mode (§4.3).
- ~~Does Space become pause-toggle (CS2-style)?~~ → No; Space stays `sim.tick` — every `--shot` harness and the walkthrough twin press Space, and single-step is the paused workflow. `0` + strip buttons own speed (§4.5).
- ~~Is bulldoze in scope?~~ → No; needs `AuthoredAction::Demolish` + save schema v6, frozen by wave 1 (§4.6).
- ~~Does Esc close the tool panel?~~ → No; panel `x` / category re-click only, keeping wave-1 `CloseTop` ladder tests stable (§4.7). Same answer for the Info/Actions/Statistics submenus.
- ~~Does the info window stay centred?~~ → No; the user's layout directive anchors every submenu above the bar at its button — the Info panel follows it. The **inspector** stays a floating window (the directive's explicit exception).
- ~~Does the Development button ship content?~~ → No; the button placement is the user's explicit ask, the content needs a progression system that doesn't exist. Dimmed button + `Progression — coming soon` tooltip/receipt; zero fake tree (§4.6).
- ~~Does the Actions panel edit taxes?~~ → No; `TaxPolicy` is real but out of the directive's Actions scope (care policies + save/menu actions only). Only the three sim-wired `CarePolicies` levers appear (§4.6).
- ~~Does Statistics get graphs?~~ → No; there is no history buffer — real per-tick numbers only, charting is a future wave (§4.6).

---

## 9. Out of Scope

- Roads/networks tooling and therefore CS2's Elevation / Parallel Mode rows, snapping-angle chips, and node-editing guides.
- Milestone/XP progression UI and the **Development tree's content** (no progression system exists; the Development button ships dimmed with the `Progression — coming soon` tooltip — that button IS the reserved slot, nothing behind it).
- Statistics graphs/time-series (no history buffer exists; the Statistics panel is real per-tick numbers only).
- Tax-rate editing UI (`TaxPolicy` is real, but the Actions panel is scoped by the layout directive to care policies + save/menu actions; a tax UI is its own design).
- Temperature/weather display (no simulation behind it).
- Bulldoze/demolish (save-schema change; future wave), and any `AuthoredAction`/`SAVE_VERSION` change at all.
- GPU-side UI rendering, icon fonts/SVG assets, animation/easing of panels.
- Engine repo (`~/src/ochroma`) changes of any kind — this is GAME-layer chrome.
- New thumbnail *renders* (only crops of existing cooked renders; uncovered assets keep text tiles).
- Theme-row redesign (the top-right style-pack card stays as-is this wave) and main-menu styling.

---

## 10. Related Plans / Designs

- Normative layout source: `~/Ochroma/projects/civitas_care/docs/references/ui-layout-spec.md` (user directive, 2026-06-10) — the three-zone bar + submenus-above-the-bar rule this design implements.
- Depends on: [Editor/UX Wave 1 Plan](../plans/2026-06-10-editor-ux-wave1.md) — command registry, `ViewState`, `tool_preview`/`plop_blocked`, `draw_tool_preview`, walkthrough harness. **This design's plan must not execute until wave 1 reports complete.**
- Depends on: [Editor/UX North Star](./2026-06-10-editor-ux-north-star-design.md) — invariants I1 (preview=commit), I3 (one registry), I11 (receipts).
- Implemented by: [Game UI Chrome Wave 1 Plan](../plans/2026-06-10-game-ui-chrome-wave1.md).
- Related: [Living Building Instances](./2026-06-10-living-building-instances-design.md) (inspector window the chrome coexists with), the cooked-asset pipeline (`game_asset_cook`, `renders/spectra`).
