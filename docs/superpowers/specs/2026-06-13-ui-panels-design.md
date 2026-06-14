# Design: Urban Horizon UI Panel System (2026-06-13)

**Status:** Draft
**Scope:** A cohesive, dockable panel system for the city builder (working repo `~/Ochroma/projects/urban_horizon`, game name **Urban Horizon**), extending the existing `render_gpu/hud.rs` bottom bar + info-window style. Defines the shared widget toolkit, charts/overlays, and every major panel (milestones, economy/budget, public transit, services + mental-health, building inspector, info overlays, demand/RCI, notifications, policies), each grounded in our real sim data and aligned with the feature roadmap (F1-F15).
**Related:** [SOTA Feature Roadmap](./2026-06-13-sota-citybuilder-feature-roadmap.md) (panels here surface F1-F8 MVP data); memory `civitas-public-services-scope` (full public-services + mental-health flagship), `super-realism-northstar` (game = Urban Horizon), `cs2-asset-model`, `config-first-no-hardcode-directive`.

---

## 1. Problem Statement

- We render a CS2-style toolbar, status strip, tool panel, and a few care panels (`render_gpu/hud.rs:955-2015`, `ui/panels.rs`), but there is **no general panel system** — no docking, tabs, open/close lifecycle, or shared chrome. Every new surface is bespoke draw + bespoke `*_at` hit-test.
- Sim data the model already computes is **not surfaced**: `ownership::Ledger` (`ownership/mod.rs`, owner/rent-roll/tenants/net) has no inspector; `ServiceAccessReport` (`services_access/mod.rs:48`) is computed and discarded by the UI; `CityBudget`'s two-sided revenue/spending (`budget/mod.rs:34-118`) has no breakdown panel; `TaxPolicy` (`finance/mod.rs:14`) is not player-editable in UI; `notifications::CareNotification` (`game_mechanics/notifications/mod.rs`) is computed but never rendered (roadmap F3).
- There are **2 info overlays vs CS2's 33**, **no time-series graphs** (`ui/panels.rs:8` says graphs are "out of scope"), and no milestone/progression, transit, or policy panels at all.
- CS2's UI is praised for guidance but criticized for **colorblind-unsafe green/blue ramps**, **fixed UI scale** (too large on non-4K, dev-only scaling), and **info overlays that auto-take-over the screen**. A *caring* game must do accessibility better.
- The game is a **full public-services builder with MENTAL HEALTH as the flagship**; no panel surface today expresses mental-health institutions/programs/policies or the wellbeing loop that makes them matter.

---

## 2. Done When

Running `cargo run --bin play`, a human can: press a hotkey or click a toolbar button to open the **Economy panel** docked at the right edge with Budget / Taxes / Production tabs showing live revenue and expense line items from `CityBudget`; click any building to open the **Building Inspector** showing the owner label, occupied units/slots, per-tick rent roll and net (from `ownership::Ledger`), the named resident households (from `HouseholdRegistry`), and each dependent's care-gap status; open the **Wellbeing/Care overlay** and see blocks tint along a colorblind-safe sequential ramp with a legend; open the **History panel** and see a `draw_line_graph` of wellbeing-over-time with a hover-scrub readout; and click the **bell** to see the severity-ranked notification feed — every panel sharing one title bar, close button, and theme, all openable, movable, and closeable without reading code. `cargo test -p urban_horizon ui_panel_hit_tests -- --nocapture` prints the resolved panel id and widget id for a set of seeded click coordinates, proving the hit-tester mirrors the layout.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Panel registry open/close/focus | open Economy, open Inspector, assert Inspector is focused (top of z-order) and Economy still in registry; close Economy → registry len drops by 1 | `assert!(registry.is_some())` |
| Shared hit-tester routes clicks | seeded click inside Economy "Taxes" tab returns `PanelHit{ panel: Economy, widget: TabTaxes }`; click in dead space returns `None` | `assert!(hit.is_some())` |
| Budget panel reads real ledger | with a seeded `CityBudget{ taxes: 1200.0, service_upkeep: 800.0, .. }`, panel rows sum to `revenue()=...`/`spending()=...` and net line equals `net()` | `assert!(rows.len() > 0)` |
| Building inspector reads ownership | seeded `Ledger{ owner_label:"Maple Holdings", occupied_units:3, rent_roll:450.0, net: 90.0 }` → inspector text contains "Maple Holdings", "3", "450", "90" | inspector text contains "Building" |
| Line graph plots real series | feed history ring buffer `[0.5,0.55,0.6]`, `draw_line_graph` polyline's last y is above its first y (rising); hover at x=tick prints `0.6` | `assert!(buffer.len() > 0)` |
| Overlay ramp is colorblind-safe | overlay maps value→color via a named sequential ramp; assert the ramp's endpoints differ in luminance by ≥ threshold (not just hue) | `assert!(color != [0,0,0,0])` |
| Demand/RCI panel from compute_demand | `compute_demand(&profile,&supply)` → panel bar heights equal `Demand{residential,..}` fields scaled | `assert!(bars.len()==5)` |
| Notification feed renders + bell badge | seeded 1 Critical + 2 Warning → bell badge text == "1" (count_at_least Critical); feed lists 3 rows newest-first | feed struct exists |
| Mental-health panel surfaces the loop | with a wellbeing axis dropped, panel shows the axis value AND the migration/labor consequence line (not just the number) | `assert!(mental_health.is_finite())` |

---

## 4. Architecture

The system is **immediate-mode over the existing software framebuffer** (`px: &mut [[u8;4]]`), preserving the proven `hud.rs` draw/`*_at` split: every panel has a pure layout step that both *draws* and *answers hit-tests* from the same geometry struct (the `ToolbarLayout` pattern, `hud.rs:919`). No retained-mode widget tree, no engine touch — this is GAME-layer UI under `src/render_gpu/` and `src/ui/`, never in `vox_*` crates.

### 4.1 Panel registry & lifecycle (the spine)
A game-side `PanelManager` (new, `src/ui/panel.rs`) holds an ordered `Vec<OpenPanel>` (z-order: last = focused/top). Each `OpenPanel` carries a `PanelKind` enum (Economy, Inspector, Transit, Services, MentalHealth, Milestones, Policies, History, Notifications, Demand), a screen `Rect`, a dock side (`Dock::{Left,Right,Float,Bottom}`), and per-kind state (e.g. active tab index, selected building `InstanceId`, scroll offset). Opening a panel that is already open re-focuses it (pushes to top) rather than duplicating. This mirrors the determinism discipline used elsewhere (`household/mod.rs:65`): no HashMap iteration over panels for draw order — a stable `Vec`. The manager is owned by the same struct that owns HUD state today (the `bin/play.rs` UI state; roadmap F4 moves inspector logic lib-side to `src/ui/`).

Lifecycle: `open(kind)` / `close(idx)` / `focus(idx)` / `move_to(idx, dock)`. Draw order iterates the `Vec` front-to-back; hit-test iterates **back-to-front** (topmost panel wins a click) and returns the first `PanelHit{ panel: PanelKind, widget: WidgetId }`, else falls through to the world (building pick → opens Inspector).

### 4.2 Shared widget toolkit
A small pure-function widget set in `src/render_gpu/widgets.rs`, all built on the existing primitives (`fill`, `fill_rounded` `:126`, `text` `:167`, `line` `:226`, `disc` `:255`) and the locked design tokens (`hud.rs:16-55`: `PANEL`, `ACCENT`, `TEXT`, `MUTED`, `GOOD`, `BAD`, `SP1/2/3`, corner-radius 6). Each widget is `fn draw_X(px,w,h, rect, ...) ` plus a matching `fn X_at(rect, mx,my) -> Option<WidgetId>`:
- **PanelFrame** — rounded backing at `A_PANEL`, a title bar (title text + close X + optional dock-grip), an optional tab row. One call gives every panel identical chrome. Returns the content `Rect`.
- **TabRow** — N tabs, active gets `ACCENT` bg + `ACCENT_HI` underline (reuse toolbar tab styling `hud.rs:1090`).
- **Row / KeyValueRow** — label (MUTED) + value (TEXT), right-aligned value; `GOOD`/`BAD` tint for signed numbers (net, surplus/deficit).
- **Button / Toggle / Stepper / Slider** — for policies & tax rates (Stepper/Slider emit a value the caller writes back into `TaxPolicy`/`CarePolicies`).
- **Bar** — horizontal proportion bar (budget line shares, coverage %); reuses the demand mini-bar idea (`hud.rs` demand tracks) horizontally.
- **Sparkline / LineGraph** — see §4.3.
- **Legend** — swatches + labels, mandatory companion to every overlay (accessibility).
- **Tooltip** — deferred hover card; opt-in, never blocks input.

### 4.3 Charts: history ring buffer + line graph (roadmap F5)
A game-side `src/history/` fixed-capacity ring buffer (`HistoryBuffer`, save-version gated) sampling per-tick scalars already computed: population, funds, `CityBudget::net()`, total unmet care, mean wellbeing, mean health, demand bars. `draw_line_graph(px,w,h, rect, series: &[f32], range, label)` plots axis + polyline + min/max gridlines; `line_graph_scrub_at(rect, series, mx)` returns the `(tick, value)` under the cursor for a hover readout (removes the `ui/panels.rs:8` "graphs out of scope" note). `Sparkline` is the inline one-row variant embedded in KeyValueRows (e.g. funds trend next to the funds chip).

### 4.4 Data-viz overlays (roadmap F6-era, accessibility-first)
An `OverlayDescriptor` registry: `{ id, label, sampler: fn(world_pos)->f32, ramp: Ramp, command_id }`. Overlays tint lots via the existing `fill_world_quad` (`hud.rs:282`) path. **Three hard accessibility rules** baked into the descriptor type: (1) ramps are **sequential, colorblind-safe** (viridis-like or single-hue luminance ramps; the `Ramp` constructor asserts endpoint luminance separation — the §3 test), never raw red↔green; (2) overlays are **opt-in and never auto-open** (fixing CS2's screen-takeover); (3) every active overlay forces an on-screen **Legend** with numeric anchors. MVP overlays: Land Value (`land_value/mod.rs`), Demand RCI, Care Reach by stage (`care/mod.rs` `CareCoverage::coverage_ratio`), Wellbeing & Health (roadmap F1/F2 `WellbeingRegistry`), Service Access (`ServiceAccessReport`), Pollution (`vox_sim/pollution`), Mental-Health Access (the flagship — see §4.5.5).

### 4.5 The individual panels
Each subsection: what it shows · our data source (file:line) · interactions · wireframe.

#### 4.5.1 Milestones / Progression
- **Shows:** current milestone, XP/progress bar toward next, the unlock list (zones, services, buildings) gained at each tier, and rewards. CS2 model: milestones grant Development/Expansion Points spent in a Progression tree (Paradox DD#10). We adopt milestones-with-unlocks; **Urban-Horizon twist:** progress is driven not only by population but by a **Wellbeing/Care score** so the game rewards caring, not just sprawl.
- **Data:** population/funds from `CareStats` (`game/mod.rs:226`); unlock gating is a new small `src/progression/` table (config-first per `config-first-no-hardcode-directive`) keyed off population + mean wellbeing (`WellbeingRegistry`, roadmap F1). Tools/categories it unlocks map to existing toolbar `cats`/`tools` (`hud.rs:955`).
- **Interactions:** click a milestone to preview its unlocks; reward "claim" toast routes through the notification feed (§4.5.8).
```
+-- Progression ------------------------------[x]-+
| Milestone 3: Growing Town          [#####...]   |
|   next at Pop 2,500  +  Wellbeing >= 0.55       |
|-------------------------------------------------|
|  v Tier 1  Founding        [unlocked]           |
|  v Tier 2  Hamlet          [unlocked]           |
|  > Tier 3  Growing Town    [3 of 5 ]            |
|       Unlocks: Clinic - Secondary School -      |
|                Bus Depot - Mental-Health Clinic |
|       Reward:  +Care budget, Park policy        |
+-------------------------------------------------+
```

#### 4.5.2 Economy / Budget (tabs: Budget · Taxes · Production)
- **Shows (Budget tab):** two-sided revenue vs spending with per-line breakdown and a net line; mirrors CS2's hover-to-break-down budget (Economy 2.0). **Care-first lines (roadmap F8):** a "Cost of neglect" row (lost income tax + lost company output from care-gated parents) and a "Workers freed by care" row — the economic thesis statement, present in no competitor.
- **Data:** `CityBudget` (`budget/mod.rs:34-118`): revenue rows `taxes / service_fees / transit_fares / trade_exports / power_sales`, spending rows `service_upkeep / transit_operating / trade_imports / power_imports`, totals via `revenue()/spending()/net()`. Care ledger from `CareLedger` (`game_mechanics/economy/mod.rs:31`: `labor_tax`, `care_upkeep`, `net()`, `is_surplus()`). "Cost of neglect" derives from `CareStats.gated_workers` × `CareEconomy.tax_per_worker`.
- **Taxes tab:** per-sector rate **steppers/sliders** writing back into `TaxPolicy` (`finance/mod.rs:14`: `property/income/commercial/industry/office/farming/care_levy`); a Sparkline of revenue response (roadmap F8 over-taxation bends demand → `demand/mod.rs:90`).
- **Production tab:** trade balance from `GoodsFlow` (`goods/mod.rs`: `food/goods/retail` Balances, `export_revenue()/import_cost()/trade_balance()/self_sufficiency()`).
```
+-- Economy --[ Budget ][ Taxes ][ Production ]-[x]-+
|  REVENUE                        +14,200 / mo      |
|    Taxes ........................ +9,400  [bar]   |
|    Service fees ................. +1,100  [bar]   |
|    Transit fares ................   +700  [bar]   |
|    Trade exports / Power sales .. +3,000  [bar]   |
|  SPENDING                        -11,600 / mo     |
|    Service upkeep ............... -6,200  [bar]   |
|    Transit operating ............ -1,400  [bar]   |
|    Imports (trade + power) ...... -4,000  [bar]   |
|---------------------------------------------------|
|  Cost of neglect (gated parents) ... -2,300       |
|  Workers freed by care ............. +18 jobs     |
|  NET ............................... +2,600 / mo  |
+---------------------------------------------------+
```

#### 4.5.3 Public Transit
- **Shows:** list of lines (name, length, # stops, current passengers, **line-usage %**), a per-mode summary (lines / monthly riders), and ridership; CS2's Transportation info view + Line panel model.
- **Data:** routes ride on the road graph — `RoadNetwork` (`road/mod.rs:40-119`: `nodes`, `segments`, `RoadClass`) is geometry-only today; transit becomes a directive-cooked overlay network on the **routable graph** (roadmap F6/F13). Fares/operating from `budget::Transit` (`budget/mod.rs`: `operating`, `fares`). MVP surfaces fares + operating + stub line list; ridership populates once F13 lands. Wait-time-as-edge-cost is called out as the CS2 fix.
- **Interactions:** select line → highlight stops on map (reuse stop-highlight pattern); rename; click a stop to see boardings. **Care lens:** flag lines serving low-mobility / vulnerable households ("can they get there", roadmap F6/F12).
```
+-- Transit ----------------------------------[x]-+
|  Mode      Lines   Riders/mo                     |
|  Bus         3      12,400                        |
|  Tram        1       4,800                        |
|--------------------------------------------------|
|  Line          Len   Stops  Pax   Usage          |
|  > Riverside    6km    11    220   [#####--] 71% |
|    Hilltop      4km     8    140   [###----] 48% |
|  Fares +700  /  Operating -1,400  (net -700/mo)  |
+--------------------------------------------------+
```

#### 4.5.4 Services (coverage) + 4.5.5 Mental Health (flagship)
- **Services shows:** per-service coverage and capacity vs demand for the full public-services suite (health, education, police, fire, garbage+recycling, water+sewage, power) — memory `civitas-public-services-scope`. Bars green→amber→red on coverage (colorblind-safe ramp + numeric labels).
- **Data:** `ServiceAccessReport` (`services_access/mod.rs:10-175`: `school_seats`, `enrolled`, `unplaced_children`, `health_capacity`, `health_in_reach`), `AccessKind` enum; care coverage from `CareCoverage` (`care/mod.rs`: `demand/served/capacity` per `DependentKind`, `coverage_ratio(stage)`, `total_unmet()`); power from `budget::PowerBalance`. After roadmap F6, "coverage" is **network travel time**, not crow-flight.
```
+-- Services ---------------------------------[x]-+
|  Health     cap 800 / reach 640   [####--] 80%  |
|  Schools    seats 1,200 / enr 1,050  unplaced 60|
|  Eldercare  served 210 / demand 260  [###-] 81% |
|  Childcare  served 480 / demand 520  [####] 92% |
|  Police/Fire/Garbage/Water/Power ... [coverage] |
+-------------------------------------------------+
```

##### 4.5.5 Mental Health (the differentiator — no panel exists in CS2)
- **Shows:** the **mental-health axis** of the population (mean + distribution), the institutions (clinics, crisis centers), **programs** and **policies**, and — critically — the **downstream consequence lines** so it visibly *matters*: how mental-health drops feed `WellbeingRegistry → satisfaction → migration/labor/economy` (roadmap F1; memory `civitas-public-services-scope`: "must MATTER, not wallpaper"). A "you cared / cost of neglect" before-after readout.
- **Data:** `WellbeingRegistry` mental-health axis (roadmap F1, new `src/game_mechanics/wellbeing/`); institutions reuse the `CareBuilding`/`CareKind` registry pattern (`care/mod.rs:21`); programs/policies join `CarePolicies` (`ui/panels.rs:31`) config-first. Consequence line ties to `demand/mod.rs:90` (migration flight) and `CareStats.effective_employed` (labor).
- **Interactions:** toggle programs/policies (Toggle/Slider widgets → write to policy struct); the panel shows the projected labor/migration delta so the player sees the loop close. Pairs with the **Mental-Health Access overlay** (§4.4).
```
+-- Mental Health ----------------------------[x]-+
|  City wellbeing index      0.62  [sparkline ~~] |
|  At-risk residents          340   (8%)          |
|  Institutions: 2 clinics, 1 crisis center       |
|--------------------------------------------------|
|  Programs                                        |
|   [x] Community outreach     [ ] Youth support   |
|   [x] Workplace wellbeing    [ ] Crisis hotline  |
|--------------------------------------------------|
|  IF YOU CUT FUNDING:                             |
|   wellbeing -0.08  ->  +120 out-migration        |
|                        -45 effective workers     |
+--------------------------------------------------+
```

#### 4.5.6 Building Inspector (roadmap F4; the user's CS2 reference)
- **Shows:** owner label, occupancy (units/slots), per-tick rent roll / upkeep / fees / **net**, the tenant companies, and the **named resident households with per-dependent care-gap status** — the care-first twist CS2's SIP lacks. Plus building age/maintenance (weathering) and a small wellbeing readout for residents.
- **Data:** `ownership::Ledger` (`ownership/mod.rs`: `owner_label`, `business_slots/occupied_business_slots`, `residential_units/occupied_units`, `tenants: Vec<CompanyId>`, `rent_roll`, `upkeep`, `fees`, `net`); residents from `HouseholdRegistry` / `HouseholdRecord` (`household/mod.rs:24`: `residence`, `workers`, `dependents`, `anchor`); company detail from `Company` (`ownership/mod.rs`: `name`, `balance`, `jobs`) — net-profit line once company P&L lands (roadmap F8); weathering from `instance` (`InstanceVisual`, `WeatheringConfig`, maintenance signal); care-gap via `household_has_care_gap` + `CareCoverage`.
- **Interactions:** opened by world building-pick (§4.1 fallthrough); click a resident household → (later) Lifepath inspector (roadmap F14); click company → focus its other holdings.
```
+-- 42 Maple Street -------------------------[x]-+
|  Owner:  Maple Holdings (Landlord)              |
|  Residential  3 / 4 units      Jobs  0          |
|  Rent roll +450  Upkeep -210  Fees -90          |
|  NET  +150 / tick                               |
|-------------------------------------------------|
|  Residents                                      |
|   * Family A   2 workers  1 child   care: OK    |
|   * Family B   1 worker   1 elder   care: GAP ! |
|   * Family C   2 workers             well: 0.71 |
|  Condition  78%   (built lvl 1, light wear)     |
+-------------------------------------------------+
```

#### 4.5.7 Demand / RCI
- **Shows:** the 5 sector demand bars (R / C / I / O / F) large, with supply context (jobs available vs filled) and a short "why" (e.g. "Commercial low: not enough shoppers"). The toolbar already draws mini demand bars (`hud.rs` demand tracks); this is the expanded read.
- **Data:** `Demand` (`demand/mod.rs:20-100`: `residential/commercial/industry/office/farming` 0..1), `Supply` (`housing`, `*_jobs`), `CityProfile` (`population`, `employed`, `wealth`); via `compute_demand(&CityProfile,&Supply)`.
```
+-- Demand --------------------------[x]-+
|  R [############----]   C [#####-----] |
|  I [#######-------]     O [###-------] |
|  F [######--------]                    |
|  Office low: workforce undereducated   |
+----------------------------------------+
```

#### 4.5.8 Notifications feed (roadmap F3)
- **Shows:** severity-ranked, newest-first feed; a **bell badge** in the status strip showing `count_at_least(Critical)`. Click bell → feed panel. Each row: severity dot (icon + color, not color-alone — accessibility), title, message, and a "go to" that focuses the map on the subject (who/where, not "unmet: 12").
- **Data:** `CareNotification` (`game_mechanics/notifications/mod.rs:13`: `severity: Severity{Info,Warning,Critical}`, `title`, `message`), `care_notifications(...)`, `seat_notifications(...)`, `most_urgent(...)`. Badge from `count_at_least(Critical)`.
- **Wiring:** bell lives in `StatusStrip` (`hud.rs:1340`), badge drawn next to the funds chip; `command/mod.rs` id `view.notifications`.
```
status strip:  ... [ Pop 4,820 ] [ $128k ] [ (!)3 ]
+-- Notifications --------------------------[x]-+
|  (!) Critical  3 parents care-gated off work  |
|       Riverside block - no childcare in reach |
|  (!) Warning   Eldercare demand unmet (50)    |
|   i  Info      Milestone 3 reached            |
+-----------------------------------------------+
```

#### 4.5.9 Policies
- **Shows:** city-wide policy toggles grouped by category (Services · Taxation · City Planning · Care/Wellbeing), mirroring CS2's City Information → Policies tab + Taxation. (District policies later, after a district tool.)
- **Data:** `CarePolicies` via `actions_rows(&CarePolicies)` (`ui/panels.rs:31`: universal childcare ON/OFF, parental leave years, elder subsidy); tax rates link to the Economy→Taxes tab (`TaxPolicy`). New mental-health programs/policies (§4.5.5) register here. All config-first (`config-first-no-hardcode-directive`).
```
+-- Policies --[ Services ][ Taxes ][ Care ]-[x]-+
|  Care & Wellbeing                               |
|   [x] Universal childcare                       |
|   [ ] Parental leave        < 1 year >          |
|   [x] Elder subsidy                             |
|   [ ] Workplace wellbeing program               |
|  ( each row shows its budget delta inline )     |
+-------------------------------------------------+
```

### 4.6 Accessibility (a caring UI for a caring game)
Directly answering CS2's documented complaints: (1) **No color-alone encoding** — every overlay/severity uses icon + shape + numeric label alongside hue, and overlay ramps are sequential luminance ramps with asserted endpoint contrast (§4.4, §3 test). (2) **UI scale is a first-class setting** (not dev-only as in CS2) — a scale factor multiplies the `hud.rs` token heights/spacing; panels reflow off the same `ToolbarLayout`-style geometry. (3) **Overlays never auto-take-over** — opt-in, dismissible, world stays visible. (4) **Plain-language "why" lines** on demand/notifications/mental-health (who/where, consequence) rather than raw counts. (5) **Tooltips are opt-in hover, never input-blocking.**

---

## 5. Data Models

```rust
/// Game-side panel registry (src/ui/panel.rs). Owned by the UI-state struct.
pub struct PanelManager {
    open: Vec<OpenPanel>,        // private; z-order, last = focused
    ui_scale: f32,               // accessibility scale (default 1.0)
}

pub struct OpenPanel {
    kind: PanelKind,             // discriminant + per-kind state
    rect: Rect,                  // screen-space, recomputed on resize/scale
    dock: Dock,
}

pub enum PanelKind {
    Milestones,
    Economy { tab: EconomyTab },                 // Budget | Taxes | Production
    Transit { selected_line: Option<LineId> },
    Services,
    MentalHealth,
    Inspector { building: InstanceId },
    Demand,
    Notifications,
    Policies { tab: PolicyTab },
    History { series: HistoryId },
}

pub enum Dock { Left, Right, Bottom, Float }

/// Result of routing a click through the open panels (back-to-front).
pub struct PanelHit { pub panel: PanelKind, pub widget: WidgetId }

/// Per-tick scalar history (src/history/mod.rs) — save-version gated, fixed cap.
pub struct HistoryBuffer { /* ring of f32 per HistoryId; private */ }

/// Accessibility-safe overlay descriptor (src/ui/overlay.rs).
pub struct OverlayDescriptor {
    pub id: OverlayId,
    pub label: &'static str,
    pub command_id: &'static str,
    // sampler: fn(world_xz: [f32;2]) -> f32   (stored as fn ptr / boxed)
    ramp: Ramp,                  // private; constructor asserts luminance separation
}
```
All panel state is plain data read each frame from the existing sim structs (§4.5); the manager holds no copies of sim data — it holds selection (`InstanceId`, tab, line) and re-reads live every draw, matching the current `StatusStrip`/`ToolPanel` borrow-and-draw pattern (`hud.rs:1340`, `:1638`).

---

## 6. API

```rust
// --- Panel manager ---
impl PanelManager {
    pub fn open(&mut self, kind: PanelKind);            // re-focuses if already open
    pub fn close(&mut self, idx: usize);
    pub fn focus(&mut self, idx: usize);                // moves to top of z-order
    pub fn hit_test(&self, mx: u32, my: u32) -> Option<(usize, PanelHit)>; // back-to-front
    pub fn ui_scale(&self) -> f32;
}

// --- Shared widgets (src/render_gpu/widgets.rs); draw + matching *_at hit-test ---
pub fn draw_panel_frame(px:&mut[[u8;4]], w:u32,h:u32, rect:Rect, title:&str, tabs:&[&str], active:usize) -> Rect; // returns content rect
pub fn panel_frame_at(rect:Rect, tabs:&[&str], mx:u32,my:u32) -> Option<WidgetId>; // close / tab / grip
pub fn draw_kv_row(px:&mut[[u8;4]], w:u32,h:u32, rect:Rect, label:&str, value:&str, tint:Option<[u8;3]>);
pub fn draw_bar(px:&mut[[u8;4]], w:u32,h:u32, rect:Rect, frac:f32, color:[u8;3]);
pub fn draw_stepper(px:&mut[[u8;4]], w:u32,h:u32, rect:Rect, value:&str) ; pub fn stepper_at(rect:Rect, mx:u32,my:u32) -> Option<StepperHit>; // Inc | Dec
pub fn draw_line_graph(px:&mut[[u8;4]], w:u32,h:u32, rect:Rect, series:&[f32], range:(f32,f32), label:&str);
pub fn line_graph_scrub_at(rect:Rect, series:&[f32], mx:u32) -> Option<(u64,f32)>; // (tick, value)
pub fn draw_legend(px:&mut[[u8;4]], w:u32,h:u32, rect:Rect, entries:&[(&str,[u8;3])]);

// --- Ramp (accessibility) ---
impl Ramp {
    pub fn sequential(stops:&[[u8;3]]) -> Ramp; // panics if endpoint luminance delta < MIN_LUMA_SEP
    pub fn sample(&self, t:f32) -> [u8;3];
}
```
Threading: all draw + hit-test run single-threaded on the UI/render path against the borrowed framebuffer — no `Send`/`Sync` needed, no locks (same as today's `hud.rs`). Panics: `Ramp::sequential` panics on insufficient luminance separation (caught by the §3 test, not at runtime in shipping ramps).

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `PanelManager` (own + tick) | UI state owner | `bin/play.rs` UI state (F4 moves logic to `src/ui/`) | one instance; holds z-order + ui_scale |
| `PanelManager::hit_test` | input handler, before world pick | `bin/play.rs` mouse handler | back-to-front; falls through to building pick → `open(Inspector)` |
| `draw_panel_frame` + per-panel draw | HUD draw, after world, before toolbar overlay | `render_gpu/hud.rs` (new `panels_draw`) | front-to-back z-order |
| Budget/Taxes/Production read | Economy panel draw | `budget/mod.rs`, `finance/mod.rs`, `goods/mod.rs` | live read each frame |
| Inspector read | Inspector panel draw | `ownership/mod.rs` (`Ledger`), `household/mod.rs`, `instance/mod.rs` | selection = `InstanceId` |
| Services/MentalHealth read | those panels' draw | `services_access/mod.rs`, `care/mod.rs`, `wellbeing/` (F1) | coverage → travel-time after F6 |
| `draw_line_graph` | History panel + Statistics submenu | `render_gpu/hud.rs`, `history/mod.rs` | reads `HistoryBuffer` |
| overlay tint | overlay pass | `render_gpu/hud.rs::fill_world_quad` (`:282`) | `OverlayDescriptor` sampler + `Ramp` + forced `Legend` |
| bell badge + feed | `StatusStrip` + Notifications panel | `render_gpu/hud.rs:1340`, `notifications/mod.rs` | badge = `count_at_least(Critical)` |
| Policy/Tax writes | Policies panel + Economy→Taxes | `ui/panels.rs:31` (`CarePolicies`), `finance/mod.rs:14` (`TaxPolicy`) | Toggle/Stepper/Slider write-back |
| command ids | command registry | `command/mod.rs` | every panel registers `view.*` ids (keys + palette + future Ask-Civitas) |

---

## 8. Open Questions

- [ ] Docking model depth: simple edge-snap (Left/Right/Bottom/Float) vs full split-dock? (Recommend edge-snap for MVP — float by default, snap to an edge when dragged near it.)
- [ ] Panel state persistence in saves: remember which panels were open + their positions? (Recommend yes but additive/version-gated, mirroring roadmap's staged save-version policy.)
- [ ] One omni-panel (CS2 City-Info-style multi-tab) vs many small panels? (Recommend many small dockable panels — better for the multi-monitor / power-user audience and avoids CS2's screen-takeover, but group related ones, e.g. Economy tabs.)
- [ ] UI-scale granularity: continuous slider vs presets (0.75/1.0/1.25/1.5)? (Recommend presets for crisp pixel snapping in the software framebuffer.)

---

## 9. Out of Scope

- Retained-mode / reactive widget framework — we stay immediate-mode over the software framebuffer to match `hud.rs`; no engine-side UI in `vox_*`.
- District-level policies and a district tool (CS2 milestone-4 feature) — city-wide policies only for MVP.
- The Lifepath / follow-a-citizen inspector (roadmap F14) — the building inspector lists residents but per-citizen drill-down is later.
- Live transit ridership numbers and route editing UX — depend on the routable graph + directive-cooked transit (roadmap F6/F13); MVP transit panel shows fares/operating + a stub line list.
- Asset-authoring / pack-manager panel (roadmap F15) — separate design.

---

## 10. Related Plans / Designs

- Depends on: [SOTA Feature Roadmap](./2026-06-13-sota-citybuilder-feature-roadmap.md) — F3 (notifications), F4 (inspector), F5 (history/graphs), F8 (tax UI + budget care lines) are the MVP data sources; F1 (`WellbeingRegistry`) and F6 (routable graph) deepen the Mental-Health/Services/overlay panels.
- Required before: a per-panel implementation plan in `docs/superpowers/plans/` using `docs/templates/plan.md` (each `Done When` names an exact command + exact on-screen output; no `todo!()`, no wire-later).
- Related: memory `civitas-public-services-scope` (full services + mental-health flagship), `super-realism-northstar` (Urban Horizon), `cs2-asset-model`, `config-first-no-hardcode-directive`, `gpu-first-directive`.
- Research sources: CS2 progression ([Paradox DD#10](https://forum.paradoxplaza.com/forum/threads/development-diary-10-game-progression.1596392/)), Economy 2.0 ([Paradox dev diary](https://www.paradoxinteractive.com/games/cities-skylines-ii/news/dev-diary-economy-part-one)), transit/info-views/services ([CS2 wiki](https://cs2.paradoxwikis.com/Info_views)), building SIP/happiness ([CS2 wiki Services](https://cs2.paradoxwikis.com/Services)), accessibility criticism ([colorblind thread](https://steamcommunity.com/app/949230/discussions/0/3877095833479033441/), [UI scale thread](https://steamcommunity.com/app/949230/discussions/0/601913251731809415/)).
```
