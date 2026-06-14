# Living Building Instances Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every developed parcel in Urban Horizon becomes a persistent, inspectable `BuildingInstance` — frozen household/job caps, live engine occupancy, an evolving property value, an UnderConstruction → Active → Abandoned state machine driven through the engine's `operational` flag — with replay-exact save/load and a click-to-inspect window in the running game.
**Done When:** `cd ~/Ochroma/projects/urban_horizon && cargo test` is green, **and** `cargo run --release --bin play` → New Game → **Demo** → Start Game → press **Space** 8 times → **Tab** to the **Inspect** category (last in the bottom bar) → left-click a risen Forge house → a window titled `Building — Craftsman House` opens showing exactly six lines in this shape (real values from that lot):

```
Asset        Craftsman House  (forge.house.craftsman)
State        Occupied   built tick 3
Households   1 / 2
Jobs         -
Value        134,890
Zone         Res Low   Level 1
```

Pressing **Space** twice and re-clicking the same house shows the `Value` (and/or `Households`) line **changed**; clicking a parcel that developed within the last 2 presses shows `State        Building   N ticks left`; pressing `i` shows a City Information line `Instances         N active · M building · K abandoned` whose three numbers match what is on the ground. A human at the keyboard verifies all of this without reading code.
**Architecture:** A `Vec`-backed `BuildingRegistry` owned by `CivitasGame` captures the engine building id that `develop_parcels` currently discards, freezes the same seeded numbers the engine capacity already uses, and ticks a deterministic state machine as step 1c of `CivitasGame::tick`. `sector_supply()` / `assessed_value()` keep their exact signatures but read Active instances. Persistence is the existing action-log replay — **no `SAVE_VERSION` bump**; determinism of the registry tick is what makes reload exact. The inspector is a new `BuildTool::Inspect` + `CivitasGame::instance_at` picking + the existing `hud::draw_city_info_window`. State-driven visuals are the LAST task and are **gated on render_gpu ownership release**.
**Design Document:** `docs/superpowers/specs/2026-06-10-living-building-instances-design.md`
**Tech Stack:** Rust (workspace toolchain as-is), game repo `~/Ochroma/projects/urban_horizon`, engine crates read-only (`vox_sim::buildings` pub fields only)
**Build:** `cd ~/Ochroma/projects/urban_horizon && cargo build && cargo test` (the play binary needs a desktop display)

---

## IMPORTANT NOTES

- **OWNERSHIP — DO NOT TOUCH:** `~/Ochroma/projects/urban_horizon/src/render_gpu/**` and `src/bin/forge_pathtrace.rs` are **OWNED BY ANOTHER WORKSTREAM right now**. Tasks 1–5 must NOT edit any file under those paths. **Calling** their existing `pub` functions from `play.rs` (`render_gpu::screen_to_ground`, `render_gpu::world_to_screen`, `hud::draw_city_info_window`, `hud::info_window_close_at`, `hud::info_window_contains`) is allowed and required. The design's §4.6 state-driven visuals (`InstanceVisual`, the three `place_ready_asset_*` placers, `city_scene_inner`) are **Task 6, the LAST task, explicitly gated on "render_gpu ownership released"** — everything before it is observable through the inspector window and the `i` info line instead.
- **Repos / commits:** game `~/Ochroma/projects/urban_horizon` (git master, uncommitted work present); engine `~/src/ochroma` is **not edited by this plan at all**. **Do NOT run `git commit` anywhere** — the user commits. Every task's final step is "leave uncommitted".
- **Save format:** `SAVE_VERSION` stays **5** (`src/game/save.rs:27`). The registry is **not serialized** — `CivitasGame::replay` rebuilds it exactly because every registry rule is a pure function of replayed state. Hard determinism rules inside `BuildingRegistry`: no RNG, no wall clock, no `HashMap` iteration anywhere; instances are visited in creation (Vec) order. No serde derives on any instance type.
- Engine types used (read/write of **existing pub fields only**, `~/src/ochroma/crates/vox_sim/src/buildings.rs` — no engine edits):
  - `pub struct Building { pub id: u32, pub building_type: BuildingType, pub position: [f32; 2], pub capacity: u32, pub occupants: u32, pub operational: bool }`
  - `pub struct BuildingManager { pub buildings: Vec<Building>, .. }` — find a building with `buildings.buildings.iter_mut().find(|b| b.id == id)` (same pattern as `CivitasGame::tick` step 4a, src/game/mod.rs:686–694).
  - `match_housing` / `match_employment` (vox_sim/src/employment.rs) both assign through `BuildingManager::find_nearest_with_vacancy`, which filters `b.operational` — so `operational = false` is the whole construction/abandonment gate. **Known accepted quirk:** `CitySim::vacant_housing()` (migration attraction, city_sim.rs:210) does NOT filter `operational`; migrants may be attracted 2–4 ticks early and wait unhoused. Do not "fix" the engine.
- Real game signatures this plan binds to (verified in code — **code wins over the design doc**):
  - `fn develop_lot(&mut self, l: &crate::lot::layout::PlannedLot) -> u32` (src/game/mod.rs:284) — returns the engine building id; its only caller `develop_parcels` (build pass at src/game/mod.rs:887–897) currently discards it.
  - `fn seeded_range(range: [u32; 2], seed: u64) -> Option<u32>` (src/game/mod.rs:99) — returns `None` when `hi == 0`. Freeze seeds: homes `lot.seed ^ 0xA551_0001`, jobs `^ 0xA551_0002`, value `^ 0xA551_0003` (via the existing private helpers `lot_asset_homes` / `lot_asset_jobs` / `lot_assessed_value` — call them, do not re-derive).
  - `pub fn sector_supply(&self) -> (crate::finance::SectorJobs, u32)` (src/game/mod.rs:768) and `pub fn assessed_value(&self) -> f64` (src/game/mod.rs:912) — **signatures unchanged**, bodies become registry iterations. Callers that must keep compiling untouched: `demand()`, `wealth()`, `finances()`, `goods_flow()`, `city_budget()`.
  - `PlannedLot` (src/lot/layout.rs:71) is `Copy` with all-pub fields: `kind, zone, program, center: [f32;2], width, depth, facing_angle, seed, developed, asset_id: Option<crate::asset::AssetKey>`. `lot.occupancy() -> Occupancy` (`.homes`, `.total_jobs()`); `lot.capacity()` is the engine fallback capacity.
  - `AssetKey` (src/asset/mod.rs:411) is `Copy + PartialEq + Eq`, `AssetKey::new(&str) -> Option<AssetKey>`, `.as_str()`. **Design deviation (code wins):** `BuildingInstance.asset` must be `Option<AssetKey>`, not bare `AssetKey` — parks/agricultural/mixed-use lots can develop with **no** catalog asset (`lot_asset` returns `None`; the vanilla style packs generate no Park/Agricultural/MixedUse zonables).
  - `BuildingAsset.level: u8`, `.name: String`, `.id: String`, `.attributes: BuildingAttributes { households: [u32;2], jobs: [u32;2], property_value: [u32;2], build_cost: u32, upkeep: i32, class }` (src/asset/mod.rs). Catalog lookup: `game.asset_catalog.get(id: &str) -> Option<&BuildingAsset>`.
  - `crate::demand::Demand { residential, commercial, industry, office, farming: f32 }` (all pub). `crate::finance::SectorJobs::add(&mut self, s: Sector, jobs: u32)`; `zone_job_sector(zone) -> Option<Sector>` is a private fn in src/game/mod.rs (Residential/Park → `None`).
  - `zone_sector_budget` (src/game/mod.rs:57) is **private** — the instance module gets its own `fn zone_demand_bar(zone: ZoneType, d: &Demand) -> Option<f32>` with the identical zone→bar mapping (Park → `None`).
  - HUD (read-only render_gpu calls): `hud::draw_city_info_window(px: &mut [[u8; 4]], w: u32, h: u32, title: &str, lines: &[String])` (hud.rs:719), `hud::info_window_close_at(w, h, x, y, lines: usize) -> bool` (:771), `hud::info_window_contains(w, h, x, y, lines: usize) -> bool` (:782), `render_gpu::screen_to_ground(view: Mat4, proj: Mat4, px, py, width, height) -> Option<[f32; 2]>` (mod.rs:39).
  - `BuildTool` (src/ui/build_tools.rs:10) derives only `Clone, Copy` — **no PartialEq**; test the tool with `matches!(gv.current_tool(), BuildTool::Inspect)`. `tile_label` has signature `pub fn tile_label(self, game: &CivitasGame) -> String` (the design omitted the `game` param); `selected_zone`'s `_ => None` arm already covers `Inspect`. The `place_at` match in play.rs:530–534 is exhaustive — add `BuildTool::Inspect => false`.
- **Fallback-freeze equivalence (why one `jobs_cap: u32` is exact):** today's `sector_supply` fallback for asset-less lots sums `occ.businesses` per `biz.kind.sector()`. Every default `BuildingProgram` (src/lot/occupancy.rs:240) is single-sector and that sector equals `zone_job_sector(lot.zone)` for every zone the game can create lots in (ShoppingCenter/Shop→Commercial, OfficeComplex→Office, Industry→Industry, Farm→Farming, Residence/Civic→none). So freezing `jobs_cap = lot_asset_jobs(lot).unwrap_or(occ.total_jobs() if zone_job_sector is Some else 0)` and attributing it to `zone_job_sector(lot.zone)` reproduces today's numbers exactly. Same for housing: `households_cap = lot_asset_homes(lot).unwrap_or(occ.homes if Residence else 0)` — and both agree with `lot_engine_capacity` by construction.
- **Construction-tick rule (pinned, design refined):** `develop()` at game tick T stores `ticks_left = build_ticks(level)`. `BuildingRegistry::tick` takes `now_tick: u64` (an **additive param over the design's signature** — needed so "after `build_ticks` more ticks" is exact) and skips instances with `built_tick == now_tick` (the creation tick). The instance flips Active — and `operational = true` — during game tick `T + build_ticks(level)`. L1 house: developed at T, Occupied at end of T+3.
- **Behavioral delta, intentional:** supply and assessed value now start when construction **completes**, not when the parcel develops. Two named tests in src/game/mod.rs shift by a few ticks and are updated explicitly in Task 3: `the_general_budget_taxes_property_and_business_and_pays_service_upkeep`, `zoning_commercial_adds_commercial_supply_and_relieves_commercial_demand`; `selected_asset_attributes_drive_live_capacity_cost_and_value` changes its assessed-value expectation to the registry formula.
- Constants (named, in `src/instance/mod.rs` — tests assert against the constants, never literals): `ABANDON_AFTER: u32 = 12`, `RECOVER_AFTER: u32 = 3`, `APPRECIATION_PER_TICK: f64 = 0.002`, `APPRECIATION_CAP: f64 = 0.25`, `pub fn build_ticks(level: u8) -> u32 { 2 + level as u32 }`.
- Value formula (exact): `value = base_value × (0.70 + 0.30 × occ_ratio) × (1.0 + appreciation)`, `occ_ratio = occupants/capacity` (`1.0` if capacity 0); appreciation `+0.002` per tick while `occ_ratio ≥ 0.5`, capped at `0.25`, never decays; reset to `0.0` on recovery from Abandoned.
- **Done-When asset facts (from `assets/buildings/forge_starter/pack.json` — the design's example text was illustrative, these are the real numbers):** `forge.house.craftsman` is named **"Craftsman House"** (not "Forge Craftsman House"), level 1, households `[1, 2]`, property_value `[110000, 185000]`. The **Demo** scenario pins its Res Low district to `forge.house.craftsman`, which is why the Done When can name the exact window title.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `~/Ochroma/projects/urban_horizon/src/instance/mod.rs` | `BuildingInstance`, `InstanceState`, `BuildingRegistry`, constants, `zone_demand_bar`, (Task 6: `InstanceVisual` + `visual_for`), inline tests |
| Modify | `~/Ochroma/projects/urban_horizon/src/lib.rs` | `pub mod instance;` |
| Modify | `~/Ochroma/projects/urban_horizon/src/game/mod.rs` | `instances` field, `develop_parcels` capture, tick step 1c, registry-backed `sector_supply`/`assessed_value`, `instance_at`, test updates + new tests |
| Modify | `~/Ochroma/projects/urban_horizon/src/ui/build_tools.rs` | `BuildTool::Inspect` variant, `tile_label` arm, `("Inspect", …)` appended **last** in `build_categories()` |
| Modify | `~/Ochroma/projects/urban_horizon/src/bin/play.rs` | `inspected` field, inspect click branch, inspector window, Esc/close handling, `Instances …` info line, `thousands` formatter, `place_at` Inspect arm |
| Modify (GATED, Task 6 only) | `~/Ochroma/projects/urban_horizon/src/render_gpu/mod.rs` | `InstanceVisual` applied in `city_scene_inner` + the three `place_ready_asset_*` placers — **only after render_gpu ownership is released** |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Instance created on development, linked to the engine | `cargo test --lib instance::tests::developing_a_parcel_creates_a_linked_instance -- --nocapture`: zone a 100 m Res Low strip, tick 30; `game.instances.len()` equals developed-parcel count **and** every instance's engine building has `capacity == inst.capacity()`; prints `instances: 8, capacities: [4, 4, ...]` | `assert!(game.instances.len() > 0)` |
| Construction gates the engine | fresh instance is `UnderConstruction` with engine `operational == false`; flips `Active`/`operational == true` exactly `build_ticks(level)` game ticks after `built_tick`; prints both tick indices | `assert!(inst.state != InstanceState::Abandoned)` |
| Occupancy is the engine's real move-ins | after 30 ticks an Active residential instance's `occupants` equals its engine building's `occupants` and is `> 0`; prints `occupied 3/4 (engine occupants 3)` | `assert!(occ <= cap)` |
| Property value evolves | `inst.value()` equals `base × (0.70 + 0.30·occ_ratio) × (1 + appreciation)` recomputed independently, and strictly rises over 2 more occupied ticks; prints `base 128000 -> value 134890` | `assert!(value > 0.0)` |
| Abandonment removes a building from the economy | overbuilt commercial: an instance hits `ABANDON_AFTER` vacant ticks → `Abandoned`, engine `operational == false`; `assessed_value()` excludes it (Task 3 assert); prints assessed before/after the flip | `assert!(state.is_some())` |
| Recovery hysteresis | unit-level: an Abandoned instance fed a demand bar of `0.9` for `RECOVER_AFTER` ticks flips `Active`, `operational == true`, `appreciation == 0.0`; prints the flip tick | always-true on fresh state |
| Registry-backed economy, unchanged signatures | updated `selected_asset_attributes_drive_live_capacity_cost_and_value`: supply == Active count, `assessed_value()` == Σ `inst.value()` over Active; new test prints the develop→supply warm-up delta == `build_ticks` | comparing `len()` only |
| Replay-exact save/load, no version bump | `to_save()` → `from_save()` → `assert_eq!(loaded.instances, game.instances)` field-by-field with mixed states and nonzero occupancy; `save.version == 5`; prints `roundtrip: 12 instances identical` | comparing only `len()` |
| Inspector picking | `game.instance_at(lot.center)` returns that lot's id; `game.instance_at([1.0e6, 1.0e6])` returns `None`; prints picked id + asset name | `assert!(result.is_some())` with no None case |
| Inspector in the running game | the **Done When** sequence, verified by a human | window opens with placeholder text |
| State-driven visuals (GATED) | construction renders at `y_scale = 0.3 + 0.7·progress` with grey tint, pops to full asset on completion; `visual_for` unit test prints the y_scale series | identity visual everywhere |

---

## Task 1: `BuildingRegistry` — instances created at develop time, construction gates the engine

**Files:**
- Create: `~/Ochroma/projects/urban_horizon/src/instance/mod.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/lib.rs` (add `pub mod instance;` to the alphabetical module list)
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs` (field + `develop_parcels` + tick step 1c)

**Acceptance:** `cd ~/Ochroma/projects/urban_horizon && cargo test --lib instance:: -- --nocapture` → `developing_a_parcel_creates_a_linked_instance` prints `instances: <n>, capacities: [..]` with n ≥ 1 and every capacity > 0; `construction_gates_the_engine_until_build_ticks_elapse` prints `developed at tick <T>, active at tick <T+3>` (L1 house) with `operational` false in between. Full `cargo test` stays green.

**Wiring requirement:** `BuildingRegistry::develop` is called from the build pass of `CivitasGame::develop_parcels` (src/game/mod.rs:887–897), capturing the id `self.develop_lot(&lot)` currently discards. `BuildingRegistry::tick` is called from `CivitasGame::tick` as step **1c**, immediately after `self.develop_parcels();` (src/game/mod.rs:636) and before the demographics step. `todo!()` / stubs = **task failure**.

- [x] **Step 1: Write the failing tests** in `src/instance/mod.rs` `#[cfg(test)] mod tests` (they drive a real `CivitasGame`):

```rust
#[test]
fn developing_a_parcel_creates_a_linked_instance() {
    let mut game = crate::CivitasGame::new_small();
    let x = 50_000.0;
    game.zone_area(
        vox_sim::zoning::ZoneType::ResidentialLow,
        &[[x, 0.0], [x + 100.0, 0.0], [x + 100.0, 100.0], [x, 100.0]],
    );
    for _ in 0..30 { game.tick(); }
    let developed = game.placed_lots.iter().flat_map(|p| p.lots.iter())
        .filter(|l| l.developed).count();
    assert!(developed >= 1, "demand develops the strip");
    assert_eq!(game.instances.len(), developed, "one instance per developed parcel");
    let mut caps = Vec::new();
    for inst in game.instances.iter() {
        let b = game.sim.buildings.buildings.iter()
            .find(|b| b.id == inst.engine_building)
            .expect("instance links a real engine building");
        assert_eq!(b.capacity, inst.capacity(), "engine and instance capacity agree");
        caps.push(b.capacity);
    }
    assert!(caps.iter().all(|&c| c > 0), "res-low capacities are real, got {caps:?}");
    println!("instances: {}, capacities: {:?}", game.instances.len(), caps);
}

#[test]
fn construction_gates_the_engine_until_build_ticks_elapse() {
    let mut game = crate::CivitasGame::new_small();
    let x = 50_000.0;
    game.zone_area(
        vox_sim::zoning::ZoneType::ResidentialLow,
        &[[x, 0.0], [x + 40.0, 0.0], [x + 40.0, 40.0], [x, 40.0]],
    );
    // Tick until the first instance exists.
    while game.instances.is_empty() { game.tick(); }
    let inst = game.instances.get(0).unwrap().clone();
    let t0 = inst.built_tick;
    assert!(matches!(inst.state, InstanceState::UnderConstruction { .. }));
    let op = |g: &crate::CivitasGame| g.sim.buildings.buildings.iter()
        .find(|b| b.id == inst.engine_building).unwrap().operational;
    assert!(!op(&game), "a construction site is not operational");
    let bt = build_ticks(inst.level) as u64; // level 1 -> 3
    while game.tick_index < t0 + bt {
        assert!(!op(&game), "still building at tick {}", game.tick_index);
        game.tick();
    }
    let after = game.instances.get(0).unwrap();
    assert_eq!(after.state, InstanceState::Active, "active after {bt} more ticks");
    assert!(op(&game), "an Active building is operational");
    println!("developed at tick {t0}, active at tick {}", game.tick_index);
}
```

- [x] **Step 2: Run to verify failure** — `cargo test --lib instance:: 2>&1 | tail -5` → FAIL: `error[E0433]: ... could not find 'instance' in the crate root` (module must not exist yet).

- [x] **Step 3: Implement `src/instance/mod.rs`** — full logic, no stubs. Types exactly (design §5/§6, with the two verified deviations: `asset: Option<AssetKey>` and `tick(.., now_tick: u64)`):

```rust
pub type InstanceId = u32;

pub const ABANDON_AFTER: u32 = 12;
pub const RECOVER_AFTER: u32 = 3;
pub const APPRECIATION_PER_TICK: f64 = 0.002;
pub const APPRECIATION_CAP: f64 = 0.25;
pub fn build_ticks(level: u8) -> u32 { 2 + level as u32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceState {
    UnderConstruction { ticks_left: u32 },
    Active,
    Abandoned,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuildingInstance {
    pub engine_building: u32,
    pub plan_idx: u32,
    pub lot_idx: u32,
    /// None for asset-less developed lots (parks, farms): the procedural fallback.
    pub asset: Option<crate::asset::AssetKey>,
    pub state: InstanceState,
    pub households_cap: u32,
    pub jobs_cap: u32,
    pub base_value: f64,
    pub appreciation: f64,
    pub occupants: u32,
    pub vacant_streak: u32,
    pub recover_streak: u32,
    pub level: u8,
    pub built_tick: u64,
}

impl BuildingInstance {
    pub fn value(&self) -> f64 {
        let cap = self.capacity();
        let occ_ratio = if cap == 0 { 1.0 } else { f64::from(self.occupants) / f64::from(cap) };
        self.base_value * (0.70 + 0.30 * occ_ratio) * (1.0 + self.appreciation)
    }
    pub fn capacity(&self) -> u32 { self.households_cap.max(self.jobs_cap) }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildingRegistry { instances: Vec<BuildingInstance> }
```

`impl BuildingRegistry` (all single-threaded, called only from `CivitasGame`): `develop(&mut self, buildings: &mut vox_sim::buildings::BuildingManager, engine_building: u32, plan_idx: u32, lot_idx: u32, asset: Option<crate::asset::AssetKey>, households_cap: u32, jobs_cap: u32, base_value: f64, level: u8, now_tick: u64) -> InstanceId` — pushes the instance with `state: UnderConstruction { ticks_left: build_ticks(level) }`, `built_tick: now_tick`, `occupants: 0`, `appreciation: 0.0`, streaks 0, and sets the engine building's `operational = false`; returns `(len - 1) as u32`. Plus `tick(&mut self, buildings: &mut BuildingManager, lots: &[crate::lot::layout::LotPlan], demand: &crate::demand::Demand, now_tick: u64)` — **this task implements only the UnderConstruction arm** (skip instances with `built_tick == now_tick`; decrement `ticks_left`; at 0 → `Active` + engine `operational = true`; Active/Abandoned arms are no-ops until Task 2 — they match and do nothing **except** that Active must still refresh `inst.occupants` from the engine so Task 1 ships nothing dead). Plus accessors: `get(&self, id: InstanceId) -> Option<&BuildingInstance>`, `for_lot(&self, plan_idx: u32, lot_idx: u32) -> Option<&BuildingInstance>`, `iter(&self) -> impl Iterator<Item = &BuildingInstance>`, `len`, `is_empty`, `active_count`, `constructing_count`, `abandoned_count`, and the private `fn zone_demand_bar(zone: ZoneType, d: &crate::demand::Demand) -> Option<f32>` mapping (used in Task 2; mark `#[allow(dead_code)]` only if needed this task).

- [x] **Step 4: Wire at the exact callsites.** (a) `pub mod instance;` in `src/lib.rs`. (b) `CivitasGame` gains `pub instances: crate::instance::BuildingRegistry`, initialized `Default::default()` in `new_small()`. (c) `develop_parcels` build pass — capture and freeze (compute the freezes from `&self` **before** `develop_lot`):

```rust
// src/game/mod.rs, build pass (~line 887). Before: `self.develop_lot(&lot);`
for (pi, li) in todo {
    let mut lot = self.placed_lots[pi].lots[li];
    let asset_id = self
        .lot_asset(&lot)
        .and_then(|asset| crate::asset::AssetKey::new(&asset.id));
    lot.asset_id = asset_id;
    let level = self.lot_asset(&lot).map(|a| a.level).unwrap_or(1);
    let households_cap = self.lot_asset_homes(&lot).unwrap_or_else(|| {
        match lot.program {
            crate::lot::occupancy::BuildingProgram::Residence { .. } => lot.occupancy().homes,
            _ => 0,
        }
    });
    let jobs_cap = self.lot_asset_jobs(&lot).unwrap_or_else(|| {
        if zone_job_sector(lot.zone).is_some() { lot.occupancy().total_jobs() } else { 0 }
    });
    let base_value = self.lot_assessed_value(&lot);
    let engine_building = self.develop_lot(&lot);
    self.instances.develop(
        &mut self.sim.buildings,
        engine_building,
        pi as u32, li as u32,
        asset_id, households_cap, jobs_cap, base_value, level,
        self.tick_index,
    );
    let live_lot = &mut self.placed_lots[pi].lots[li];
    live_lot.developed = true;
    live_lot.asset_id = asset_id;
}
```

(d) `CivitasGame::tick` step 1c, directly after `self.develop_parcels();` at line 636:

```rust
// 1c. Advance every building instance (construction now; occupancy/value/
//     abandonment in Task 2) before the budget settle sees this tick's states.
let instance_demand = self.demand();
self.instances.tick(
    &mut self.sim.buildings,
    &self.placed_lots,
    &instance_demand,
    self.tick_index,
);
```

- [x] **Step 5: Run — verify non-trivial output.** `cargo test --lib instance:: -- --nocapture` → both tests PASS with printed real tick indices and capacities. Then full `cargo test` — green (supply still reads lots in this task, so no economy timing shifts yet; if any care/employment-timing test wobbles from delayed move-ins, fix forward here, do not defer).
- [x] **Step 6: Leave uncommitted** (the user commits).

---

## Task 2: Per-tick instance life — occupancy mirror, evolving value, abandonment, recovery

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/instance/mod.rs` (Active/Abandoned arms of `tick` + tests)

**Acceptance:** `cargo test --lib instance:: -- --nocapture` → `occupancy_mirrors_engine_move_ins` prints `occupied <k>/<cap> (engine occupants <k>)` with k > 0; `value_appreciates_with_occupancy` prints `base <b> -> value <v>` with v matching the formula and rising across 2 more occupied ticks; `vacancy_abandons_after_ABANDON_AFTER_ticks` prints the abandon tick with engine `operational == false`; `demand_recovers_an_abandoned_instance` prints the recovery tick with `appreciation == 0.0`. Full `cargo test` green.

**Wiring requirement:** all rules live inside `BuildingRegistry::tick` (already wired as step 1c in Task 1) — wired by construction; no new callsites. Rules are pure functions of (engine building state, stored fields, `demand`, `now_tick`), visited in creation order.

- [x] **Step 1: Write the failing tests.** Occupancy/value tests drive a real game (new_small + Res Low strip at `x = 50_000.0`, 30 ticks, find an Active instance with `occupants > 0`, recompute `value()` independently from pub fields, tick twice more while occupied, assert strict increase from appreciation). Abandonment test: new_small + a 600×120 m `CommercialRegional` strip, 60 ticks → assert at least one instance is `Abandoned`, its engine building `operational == false` and `occupants == 0`, and `vacant_streak`-driven flip happened at exactly `ABANDON_AFTER` consecutive vacant ticks (track via per-tick polling in the test loop); print the flip tick. Recovery test is **unit-level** (the only deterministic way to hold a bar > 0.5): hand-build a `BuildingManager` (`add_building(BuildingType::Commercial, [0.0, 0.0], 10)`), a one-lot `LotPlan` (struct literal — `PlannedLot` is all-pub `Copy`; `zone: ZoneType::CommercialLocal`), `registry.develop(..)`, run `tick` with `Demand { residential: 0.0, commercial: 0.0, industry: 0.0, office: 0.0, farming: 0.0 }` until construction completes and then `ABANDON_AFTER` vacant ticks flip it Abandoned; then feed `Demand { commercial: 0.9, .. }` for `RECOVER_AFTER` ticks → assert `Active`, engine `operational == true`, `appreciation == 0.0`, `vacant_streak == 0`; print both flip ticks.
- [x] **Step 2: Run to verify failure** — the Active arm currently only mirrors occupants; value/abandon/recover asserts FAIL with real (not compile) failures.
- [x] **Step 3: Implement the arms** in `tick` (no stubs):

```rust
InstanceState::Active => {
    let b = buildings.buildings.iter_mut()
        .find(|b| b.id == inst.engine_building);
    debug_assert!(b.is_some(), "instance {i} points at a dead engine building");
    let Some(b) = b else { continue };
    inst.occupants = b.occupants;
    let cap = inst.capacity();
    let occ_ratio = if cap == 0 { 1.0 } else { f64::from(inst.occupants) / f64::from(cap) };
    if occ_ratio >= 0.5 {
        inst.appreciation = (inst.appreciation + APPRECIATION_PER_TICK).min(APPRECIATION_CAP);
    }
    if cap > 0 && inst.occupants == 0 { inst.vacant_streak += 1; } else { inst.vacant_streak = 0; }
    if inst.vacant_streak >= ABANDON_AFTER {
        inst.state = InstanceState::Abandoned;
        inst.recover_streak = 0;
        b.operational = false;
    }
}
InstanceState::Abandoned => {
    let lot = &lots[inst.plan_idx as usize].lots[inst.lot_idx as usize];
    if let Some(bar) = zone_demand_bar(lot.zone, demand) {
        if bar > 0.5 { inst.recover_streak += 1; } else { inst.recover_streak = 0; }
        if inst.recover_streak >= RECOVER_AFTER {
            inst.state = InstanceState::Active;
            inst.vacant_streak = 0;
            inst.recover_streak = 0;
            inst.appreciation = 0.0;
            if let Some(b) = buildings.buildings.iter_mut()
                .find(|b| b.id == inst.engine_building) { b.operational = true; }
        }
    }
}
```

- [x] **Step 4: Wiring is by construction** (inside the already-wired `tick`); confirm the step-1c call in `CivitasGame::tick` passes `&self.placed_lots` and the freshly computed demand.
- [x] **Step 5: Run — verify non-trivial output.** All four tests PASS with printed occupant counts, value series, and flip ticks. Full `cargo test` green.
- [x] **Step 6: Leave uncommitted.**

---

## Task 3: Registry-backed `sector_supply` / `assessed_value` + the shifted-test updates

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs` (two method bodies, signatures untouched; the named test updates; one new test)

**Acceptance:** `cargo test --lib game:: -- --nocapture` → updated `selected_asset_attributes_drive_live_capacity_cost_and_value` passes with `assessed_value()` equal to Σ `inst.value()` over Active instances (printed); new `supply_starts_at_construction_complete_not_at_develop` prints `developed at tick <D>, first supply at tick <D+5>` (ComReg defaults to a level-3 asset → `build_ticks(3) == 5`). Full `cargo test` green.

**Wiring requirement:** the bodies of `pub fn sector_supply(&self) -> (crate::finance::SectorJobs, u32)` (src/game/mod.rs:768) and `pub fn assessed_value(&self) -> f64` (src/game/mod.rs:912) — every existing caller (`demand`, `wealth`, `finances`, `goods_flow`, `city_budget`, all tests) compiles unchanged.

- [x] **Step 1: Write the failing test** `supply_starts_at_construction_complete_not_at_develop`: zone a ComReg strip on new_small; tick one at a time recording (a) the first tick the engine building count rises (develop tick D) and (b) the first tick `sector_supply().0.commercial > 0`; assert `(b) - (a) == crate::instance::build_ticks(3) as u64` and that supply is `0` at every tick in between; print both ticks.
- [x] **Step 2: Run to verify failure** — today supply appears at tick D (delta 0): real assertion failure.
- [x] **Step 3: Implement the two bodies:**

```rust
pub fn sector_supply(&self) -> (crate::finance::SectorJobs, u32) {
    let mut jobs = crate::finance::SectorJobs::default();
    let mut housing = 0u32;
    for inst in self.instances.iter()
        .filter(|i| i.state == crate::instance::InstanceState::Active)
    {
        housing += inst.households_cap;
        if inst.jobs_cap > 0 {
            let lot = &self.placed_lots[inst.plan_idx as usize].lots[inst.lot_idx as usize];
            if let Some(sector) = zone_job_sector(lot.zone) {
                jobs.add(sector, inst.jobs_cap);
            }
        }
    }
    (jobs, housing)
}

pub fn assessed_value(&self) -> f64 {
    self.instances.iter()
        .filter(|i| i.state == crate::instance::InstanceState::Active)
        .map(crate::instance::BuildingInstance::value)
        .sum()
}
```

(The doc comments on both methods must be updated to say supply/value start at construction-complete and exclude Abandoned instances.)

- [x] **Step 4: Update the named tests explicitly** (the design's §4.4 commitment):
  - `selected_asset_attributes_drive_live_capacity_cost_and_value`: extend the loop `30 → 35` ticks (so every developed Victorian, `build_ticks(1) == 3`, is Active); replace `assert_eq!(game.sector_supply().1, developed.len() as u32)` with equality against the **Active** instance count; replace the `expected_value` sum of `lot_assessed_value` with `game.instances.iter().filter(Active).map(value).sum::<f64>()` and additionally assert each Active `inst.value() >= 0.70 * inst.base_value` and `inst.base_value` is in the authored `95_000..=155_000` range (the frozen number is still the asset's). Print one instance's `base -> value`.
  - `the_general_budget_taxes_property_and_business_and_pays_service_upkeep` and `zoning_commercial_adds_commercial_supply_and_relieves_commercial_demand`: extend their `0..30` tick loops to `0..35` (the 5-tick ComReg warm-up) — assertions themselves stand. If either still fails, the fix is in the implementation, not looser asserts.
  - Run the **whole** suite; `residential_development_converges_and_does_not_sprawl` and `over_zoning_stops_developing_once_demand_is_satisfied` are expected to still pass (the construction lag only delays convergence) — if not, report, don't weaken.
- [x] **Step 5: Run — verify non-trivial output.** New test prints the exact 5-tick delta; full `cargo test` green. Also extend the Task 2 abandonment test now: assert `game.assessed_value()` equals the Active-only sum after the flip (the "drops by exactly that instance's value" check), printing assessed before/after.
  - **REPORTED (per Step 4's "report, don't weaken"):** `residential_development_converges_and_does_not_sprawl` and `over_zoning_stops_developing_once_demand_is_satisfied` now FAIL at their `vacant_parcels() > 0` asserts — not from the construction lag, but from the abandonment feedback loop: in a deliberately over-zoned strip only ~2 instances ever get occupants, the rest sit fully vacant, hit `ABANDON_AFTER` (12), drop out of Active-only supply, the demand bar spikes back above `DEV_THRESHOLD` (0.12), and the remaining vacant parcels develop — so the city consumes every parcel (traced per-tick: vacant 14 → 0 across ticks 18–21 as 30+ instances abandon). `DEV_THRESHOLD` (0.12) < the recovery bar (0.5) means the dead zone re-develops instead of recovering. The two tests were left untouched; resolving the design tension (e.g. counting Abandoned stock in the development gate, or recovery before new development) is a follow-up decision, not a Task 3 edit.
- [x] **Step 6: Leave uncommitted.**

---

## Task 4: Replay-exact save/load — registry-equality roundtrip, no `SAVE_VERSION` bump

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs` (one new test beside the save tests)

**Acceptance:** `cargo test --lib game::tests::save_roundtrip_reproduces_instances_exactly -- --nocapture` → prints `roundtrip: <n> instances identical (save version 5)` with n ≥ 2 and at least one instance showing `occupants > 0` or `appreciation > 0.0` (printed), proving the equality is non-trivial. `src/game/save.rs` has **zero diff**.

**Wiring requirement:** none new — `CivitasGame::replay` (src/game/mod.rs:1030) rebuilds the registry because `develop_parcels` + `BuildingRegistry::tick` are deterministic; this task **proves** it with a field-by-field equality (derived `PartialEq` on `BuildingRegistry`/`BuildingInstance`/`InstanceState` compares every field in creation order). If the test fails, the bug is nondeterminism in Tasks 1–3 and must be fixed there.

- [x] **Step 1: Write the failing-or-passing test** (it must fail against any seeded nondeterminism; write it before reviewing the tick code again):

```rust
#[test]
fn save_roundtrip_reproduces_instances_exactly() {
    let mut game = CivitasGame::new_small();
    let x = 50_000.0;
    game.zone_area(ZoneType::CommercialRegional,
        &[[x, 0.0], [x + 200.0, 0.0], [x + 200.0, 100.0], [x, 100.0]]);
    for _ in 0..5 { game.tick(); }
    game.zone_area(ZoneType::ResidentialLow,
        &[[x, 300.0], [x + 120.0, 300.0], [x + 120.0, 400.0], [x, 400.0]]);
    for _ in 0..35 { game.tick(); }
    assert!(game.instances.len() >= 2, "need a real registry to compare");
    assert!(
        game.instances.iter().any(|i| i.occupants > 0 || i.appreciation > 0.0),
        "the registry must carry lived-in state, not defaults"
    );
    let save = game.to_save();
    assert_eq!(save.version, crate::game::save::SAVE_VERSION);
    assert_eq!(save.version, 5, "this plan must not bump SAVE_VERSION");
    let loaded = CivitasGame::from_save(&save);
    assert_eq!(loaded.instances, game.instances,
        "replay rebuilds every instance field-by-field");
    assert_eq!(loaded.sim.budget.funds, game.sim.budget.funds);
    println!("roundtrip: {} instances identical (save version {})",
        game.instances.len(), save.version);
}
```

- [x] **Step 2: Run it** — passed first run: `roundtrip: 16 instances identical (save version 5)` with a lived-in instance (occupants 8) — no nondeterminism found.
- [x] **Step 3–4: No implementation/wiring expected** — no code change was needed; the registry replayed exactly.
- [x] **Step 5: Run — verify** the printed instance count and the existing save tests (`save_then_load_reconstructs_the_city_losslessly`, `interleaved_zoning_and_ticking_replays_losslessly`, `save_load_on_a_real_map_is_lossless_and_flat_replay_diverges`) all still pass — they now implicitly cover registry-driven funds.
- [x] **Step 6: Leave uncommitted.**

---

## Task 5: Click-to-inspect in the running game — `instance_at`, the Inspect tool, the inspector window, the `Instances` info line

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs` (`instance_at` + picking test)
- Modify: `~/Ochroma/projects/urban_horizon/src/ui/build_tools.rs` (`Inspect` variant, label, category appended last)
- Modify: `~/Ochroma/projects/urban_horizon/src/bin/play.rs` (everything the Done When shows)

**Acceptance:** `cargo test --lib game::tests::instance_at_picks_the_lot_under_a_point -- --nocapture` prints the picked id + asset id and proves the `None` case; then the plan-level **Done When** sequence in `cargo run --release --bin play` is verified by a human (exact window title `Building — Craftsman House`, six exact-label lines, Value/Households change after 2 Space presses, `Building   N ticks left` on a fresh parcel, `Instances …` line matching the ground).

**Wiring requirement:** `instance_at` is called from the Inspect branch of the left-click chain in `play.rs` (`WindowEvent::MouseInput`, ~line 848, inserted after the `gv.is_zoning()` branch and **before** `gv.place_at`); the inspector window draws in `GameView::render` beside the info window (~line 474); the `Instances` line is appended in `GameView::info_lines` (play.rs:322); the `place_at` match (play.rs:530–534) gains `BuildTool::Inspect => false`. **No file under `src/render_gpu/` is edited** — only its existing pub fns are called.

- [x] **Step 1: Write the failing picking test** in src/game/mod.rs tests:

```rust
#[test]
fn instance_at_picks_the_lot_under_a_point() {
    let mut game = CivitasGame::new_small();
    let x = 50_000.0;
    game.zone_area(ZoneType::ResidentialLow,
        &[[x, 0.0], [x + 100.0, 0.0], [x + 100.0, 100.0], [x, 100.0]]);
    for _ in 0..30 { game.tick(); }
    let inst = game.instances.get(0).expect("a developed instance exists").clone();
    let lot = game.placed_lots[inst.plan_idx as usize].lots[inst.lot_idx as usize];
    let picked = game.instance_at(lot.center);
    assert_eq!(picked, Some(0), "the lot centre picks its own instance");
    assert_eq!(game.instance_at([1.0e6, 1.0e6]), None, "empty ground picks nothing");
    // A vacant (undeveloped) parcel must also pick nothing.
    if let Some(vacant) = game.placed_lots.iter().flat_map(|p| p.lots.iter())
        .find(|l| !l.developed)
    {
        assert_eq!(game.instance_at(vacant.center), None, "vacant parcels have no instance");
    }
    println!("picked id {:?} -> asset {:?}", picked,
        inst.asset.map(|k| k.as_str().to_string()));
}
```

- [x] **Step 2: Run to verify failure** — `error[E0599]: no method named 'instance_at'`.
- [x] **Step 3: Implement.** (a) `CivitasGame::instance_at` — rotated-rect test in the lot frame (`facing_angle` carries local +Z onto `(sin yaw, cos yaw)`, so the inverse rotation is `lx = c·dx − s·dz`, `lz = s·dx + c·dz`):

```rust
/// The instance under a world XZ point. None over roads, vacant parcels,
/// ploppables, empty ground. Lots do not overlap; first hit wins.
pub fn instance_at(&self, p: [f32; 2]) -> Option<crate::instance::InstanceId> {
    for (i, inst) in self.instances.iter().enumerate() {
        let lot = &self.placed_lots[inst.plan_idx as usize].lots[inst.lot_idx as usize];
        let dx = p[0] - lot.center[0];
        let dz = p[1] - lot.center[1];
        let (s, c) = lot.facing_angle.sin_cos();
        let lx = c * dx - s * dz;
        let lz = s * dx + c * dz;
        if lx.abs() <= lot.width * 0.5 && lz.abs() <= lot.depth * 0.5 {
            return Some(i as u32);
        }
    }
    None
}
```

(b) `build_tools.rs`: add unit variant `Inspect` to `BuildTool`; `tile_label` gains `BuildTool::Inspect => "Inspect".to_string()` (signature is `tile_label(self, game: &CivitasGame) -> String` — the `game` param stays); `selected_zone` needs no change (`_ => None`); `build_categories()` appends `("Inspect", vec![Inspect])` **last** so existing category indices and 1–9 hotkeys are untouched.
(c) `play.rs`: `GameView` gains `inspected: Option<u32>` (init `None` in `new`); method

```rust
/// Inspector window content: (title, the six exact lines) for an instance.
fn inspector(&self, id: u32) -> Option<(String, Vec<String>)> {
    use urban_horizon::instance::InstanceState;
    let inst = self.game.instances.get(id)?;
    let lot = &self.game.placed_lots[inst.plan_idx as usize].lots[inst.lot_idx as usize];
    let (name, asset_id) = inst.asset
        .and_then(|k| self.game.asset_catalog.get(k.as_str()))
        .map(|a| (a.name.clone(), a.id.clone()))
        .unwrap_or_else(|| ("Procedural".to_string(), "-".to_string()));
    let state = match inst.state {
        InstanceState::UnderConstruction { ticks_left } =>
            format!("Building   {ticks_left} ticks left"),
        InstanceState::Active => format!("Occupied   built tick {}", inst.built_tick),
        InstanceState::Abandoned => "Abandoned".to_string(),
    };
    let cap_line = |cap: u32| if cap > 0 {
        format!("{} / {}", inst.occupants.min(cap), cap)
    } else { "-".to_string() };
    let lines = vec![
        format!("Asset        {name}  ({asset_id})"),
        format!("State        {state}"),
        format!("Households   {}", cap_line(inst.households_cap)),
        format!("Jobs         {}", cap_line(inst.jobs_cap)),
        format!("Value        {}", thousands(inst.value())),
        format!("Zone         {}   Level {}",
            build_tools::zone_label(lot.zone), inst.level),
    ];
    Some((format!("Building — {name}"), lines))
}
```

plus the free fn `fn thousands(v: f64) -> String` (round to i64, insert `,` every three digits — real implementation, no locale crate) and

```rust
fn inspect_click(&mut self, px: f32, py: f32, w: u32, h: u32) {
    let cam = self.camera(w, h);
    let Some(pos) = render_gpu::screen_to_ground(cam.view, cam.proj, px, py, w, h) else { return };
    self.inspected = self.game.instance_at(pos);
    if self.inspected.is_some() { self.info_open = false; } // one window at a time
}
```

- [x] **Step 4: Wire at the exact callsites in `play.rs`:**
  - Left-click chain (~line 848): after the `gv.is_zoning()` branch, before `gv.place_at`: `} else if matches!(gv.current_tool(), BuildTool::Inspect) { gv.inspect_click(px, py, w, h); }`. Earlier in the same chain (before the `info_open` close branches) add the inspector-window close/swallow pair: if `gv.inspected` is `Some(id)` and `hud::info_window_close_at(w, h, px, py, gv.inspector(id).map_or(0, |(_, l)| l.len()))` → `gv.inspected = None`; else if it `info_window_contains` → swallow.
  - `GameView::render` (~line 474, after the info-window draw): `if let Some(id) = self.inspected { if let Some((title, lines)) = self.inspector(id) { hud::draw_city_info_window(&mut pixels, w, h, &title, &lines); } }` — re-derived every frame, so Space presses update it live.
  - `key()` Esc order becomes: draft → `inspected = None` → `info_open = false` → menu. The `"i"` toggle also clears `inspected` (one window at a time).
  - `info_lines()` (play.rs:322), after the `Buildings` line: `format!("Instances         {} active · {} building · {} abandoned", self.game.instances.active_count(), self.game.instances.constructing_count(), self.game.instances.abandoned_count()),`.
  - `place_at` match: add `BuildTool::Inspect => false`.
- [x] **Step 5: Run — verify.** `cargo test` green (the picking test prints real ids), then run the full **Done When** sequence by hand and confirm every exact string, including a `Building   N ticks left` state on a freshly developed parcel and the live `Value` change after two Space presses.
  - **VERIFIED (via an added `--shot` path in `play.rs` `main()` that drives the real `GameView` — `MenuApp::start_game(Demo)`, `key(Space)` ×8, `key(Tab)` to Inspect, `inspect_click` at the house's `world_to_screen` position — and saves the rendered frames to PNG):** picking test prints `picked id Some(0) -> asset Some("european.res_low.l1.georgian")`; the shot prints `inspector title: Building — Craftsman House` with the six exact lines (`Asset        Craftsman House  (forge.house.craftsman)` / `State        Occupied   built tick 0` / `Households   1 / 1` / `Jobs         -` / `Value        154,739` / `Zone         Res Low   Level 1`), `State        Building   2 ticks left` on a fresh Craftsman lot after Space press 1, `Value` 154,739 → 155,352 after two more presses (`value/households changed: true`), and `Instances         40 active · 0 building · 0 abandoned` matching the ground-truth counts. The rendered PNGs show both windows over the Demo city with **Inspect** highlighted last in the bottom bar. `cargo test --lib`: 239 passed; the only 2 failures are the pre-existing Task 3-REPORTED `residential_development_converges_and_does_not_sprawl` / `over_zoning_stops_developing_once_demand_is_satisfied`. (`cargo test` all-targets additionally hits a transient `game_asset_cook.rs` compile error from the other workstream's in-flight edit — file not touched by this plan.)
- [x] **Step 6: Leave uncommitted.**

---

## Task 6 (gate RELEASED 2026-06-10 — executed): State-driven visuals from the cooked payload

> **GATE: render_gpu ownership released.** `src/render_gpu/**` is owned by another workstream. This task must not begin until the user confirms that ownership is released. Everything before this task is already observable through the inspector window and the `i` info line. If the gate is still closed when Tasks 1–5 are done, **stop and report** — do not "just make a small edit".
> **GATE RELEASED:** the remediation workflow finished and handed `render_gpu/mod.rs` back (it gained socket consumers, per-instance `InstanceVariation`, and vegetation placement). Task 6 was implemented against that current code; the state treatment composes AFTER the variation tint jitter.

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/instance/mod.rs` (`InstanceVisual` + `visual_for` + unit test)
- Modify: `~/Ochroma/projects/urban_horizon/src/render_gpu/mod.rs` (the per-lot branch of `city_scene_inner` ~lines 689–760; the three placers `place_ready_asset_splats` (:1289), `place_ready_asset_sdf_instance` (:1319), `place_ready_asset_mesh_renderables` (:1338))

**Acceptance:** `cargo test --lib instance::tests::construction_visual_scales_height -- --nocapture` prints the y_scale series (e.g. `ticks_left 3 -> y_scale 0.30, 2 -> 0.53, 1 -> 0.77, active -> 1.00`); then in `cargo run --release --bin play` (Demo, Space pressed once after a parcel develops) the rising house renders at partial height with a grey tint and pops to the full cooked asset when the inspector says `Occupied` — a human sees it.

**Wiring requirement:** `city_scene_inner`'s developed-lot branch looks up `game.instances.for_lot(pi, li)` (the loop must switch to indexed iteration to know `pi`/`li`) and passes the derived visual into all three `place_ready_asset_*` calls; every **other** caller of those placers (ploppables/services) passes `InstanceVisual::IDENTITY`. The cooked `ReadyAssetPayload` is never mutated — `y_scale` multiplies atom Y scale/position and mesh vertex Y, `tint` multiplies atom color and triangle rgb (presentation only, renderer-agnostic).

- [x] **Step 1: Failing unit test** for the pure derivation in `instance/mod.rs`: `visual_for(&BuildingInstance) -> InstanceVisual` where `InstanceVisual { y_scale: f32, tint: [f32; 3] }` with `IDENTITY` const `{ y_scale: 1.0, tint: [1.0; 3] }`; UnderConstruction → `y_scale = 0.3 + 0.7 × progress` (`progress = 1.0 − ticks_left as f32 / build_ticks(level) as f32`), tint `[0.62, 0.60, 0.58]`; Active → identity; Abandoned → `y_scale 1.0`, tint `[0.45, 0.45, 0.45]`. Assert the printed series above against the constants.
- [x] **Step 2: Verify failure** (`error[E0425]: cannot find function 'visual_for'` + unresolved `InstanceVisual` imports), **Step 3: implement** `visual_for`, **Step 4: wire** the three placers (additive `visual: InstanceVisual` parameter; update every callsite in render_gpu, identity for non-instance callers) and the `for_lot` lookup in `city_scene_inner`.
- [x] **Step 5: Run** — unit test prints the real series; `cargo build --release --bin play` clean; human observes the grey partial-height construction in the Demo city.
  - **VERIFIED (automated; the by-eye Demo check remains for the user):** `construction_visual_scales_height` prints `ticks_left 3 -> y_scale 0.30, 2 -> 0.53, 1 -> 0.77, active -> 1.00 (abandoned tint [0.45, 0.45, 0.45])`. Three new render_gpu tests assert the treatment on real cooked-payload placements: `construction_visual_clamps_placed_splat_height` prints `height clamp: atoms 10.42 m -> 5.56 m (ratio 0.533), mesh 9.73 m -> 5.19 m (ratio 0.533), y_scale 0.533` (and proves the under-construction SDF suppression); `abandoned_visual_darkens_the_placed_splats` prints `mean atom luma: active 0.4624 -> abandoned 0.2081 (ratio 0.45, tint [0.45, 0.45, 0.45])`; `city_scene_renders_construction_progress_from_the_registry` proves the registry wiring end to end on the Demo city: `registry-driven scene: ticks_left 3 renders 3.14 m, active renders 10.47 m`. Full `cargo test --lib`: **256 passed; 0 failed** (252 baseline + 4 new). `cargo run --release --bin play -- --shot` re-ran clean: same six inspector lines, `value/households changed: true`, `Instances 40 active · 0 building · 0 abandoned` matching ground truth, PNGs written.
  - **Composition decisions (current code is post-remediation):** the state tint multiplies the display RGB **after** `InstanceVariation`'s facade/trim HSV jitter; the lived-in extras the remediation added (door-path/chimney socket consumers + yard/garden vegetation) are placed **only for the identity visual** — a rising site and an abandoned shell render the treated building alone; the resident SDF (no scale field on `SpatialSdfInstance`) is **suppressed while `y_scale < 1`** and kept for Abandoned (SDFs carry no color). Identity placements are bit-identical to pre-Task-6 output (all variation/vegetation determinism tests still pass).
- [x] **Step 6: Leave uncommitted.**

---

## Self-Review Checklist

- [x] Every task implements AND wires in the same task — Task 4 is a determinism proof whose only legal code change is a fix in already-wired code; Task 6 is gated, not deferred wiring
- [x] Every `Acceptance` names a real non-trivial expected output (printed tick indices, capacities, the exact six inspector lines, the 5-tick warm-up delta)
- [x] Every `Wiring requirement` names an exact function and exact file (with current line numbers)
- [x] `IMPORTANT NOTES` contains real, **code-verified** API signatures — including the four places the design doc and the code disagree (Option<AssetKey>, `tick(.., now_tick)`, `tile_label(self, game)`, the real asset name/caps in pack.json)
- [x] `File Map` lists every file that appears in any task; render_gpu appears only in the gated Task 6
- [x] No step contains `todo!()`, `unimplemented!()`, or stub bodies
- [x] `Done When` names a specific command and specific human-observable result (exact window title and line shapes for a real placed building)
- [x] All types, method names, and signatures are consistent across all tasks (`instances`, `instance_at`, `inspector`, `visual_for`, `build_ticks`)
- [x] No `git commit` anywhere — every task ends "leave uncommitted"
- [x] `SAVE_VERSION` stays 5; `src/game/save.rs` is not in the File Map
