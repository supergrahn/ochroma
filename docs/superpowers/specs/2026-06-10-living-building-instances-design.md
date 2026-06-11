# Design: Living Building Instances (2026-06-10)

**Status:** Draft
**Scope:** Give every developed parcel in Civitas Care a persistent, inspectable, simulated building instance — frozen households/jobs capacity, live occupancy read from the engine, an evolving property value, a construction → occupied → abandoned state machine, replay-exact save/load, and a click-to-inspect window in the running game. All new code lives in the GAME layer (`~/Ochroma/projects/civitas_care`); engine crates are untouched.
**Related:** `[SOTA City Block Phase 1 Plan](../plans/2026-06-10-sota-city-block-phase1.md)` (Task 10), `[Virtualized Splat Rendering Design](./2026-06-10-virtualized-splat-rendering-design.md)` (Task 9, sibling)

---

## 1. Problem Statement

Concrete symptoms in today's code (`~/Ochroma/projects/civitas_care`):

- **The engine-building link is discarded.** `CivitasGame::develop_lot` (src/game/mod.rs:284) returns the new `vox_sim::buildings::Building` id, and its only caller `develop_parcels` (src/game/mod.rs:893) drops it: `self.develop_lot(&lot);`. Once built, a `PlannedLot` has no way back to its engine building — nothing in the game can answer "who lives in this house?" even though the engine tracks `occupants` per building.
- **Per-building economics are re-derived from scratch on every read.** `sector_supply()` (src/game/mod.rs:768) recomputes `seeded_range(attrs.households, lot.seed ^ 0xA551_0001)` for every developed lot, and it is called four times per tick (`demand()`, `finances()`, `goods_flow()`, `city_budget()`). There is no place to hang per-building state because the numbers are stateless functions, not instances.
- **Buildings teleport in at full capacity.** `develop_parcels` flips `lot.developed = true` and the same tick the engine building exists with full `capacity`, full assessed value, and the complete cooked `ReadyAssetPayload` rendered — no construction phase, and no building can ever be abandoned. The only state a building has is one `bool`.
- **Property value is frozen forever.** `lot_assessed_value` (src/game/mod.rs:463) is a pure seeded range — a fully-occupied house in a well-served city is assessed identically to one that has stood empty for years.
- **Clicking a building in the running game does nothing.** `play` (src/bin/play.rs) routes left-clicks to HUD widgets, zoning, or plopping; the only readout is the city-wide info window (`info_lines`, src/bin/play.rs:322). The cooked assets (`ReadyAssetPayload` — mesh, material zones, SDF, sockets) render beautifully but are economically and interactively inert.

---

## 2. Done When

Running `cd ~/Ochroma/projects/civitas_care && cargo run --release --bin play`, choosing **New Game → any scenario → Start Game**, drawing a Res Low district (Zones → Res Low, click the frontage + corners, right-click to close), and pressing **Space** ~10 times, then:

1. **Inspector:** Tab to the new **Inspect** category (last in the bottom bar), left-click a risen house → a window titled `Building — <asset name>` opens showing real per-instance lines in exactly this shape:

   ```
   Asset        Forge Craftsman House  (forge.house.craftsman)
   State        Occupied   built tick 7
   Households   3 / 4
   Jobs         -
   Value        93,400
   Zone         Res Low   Level 1
   ```

   Pressing **Space** twice more and re-clicking the same house shows `Households` and/or `Value` **changed** (move-ins raise occupancy; occupied ticks appreciate value).
2. **Construction is visible:** in the first 2–3 Space presses after a parcel develops, its building renders at partial height with a grey tint, then pops to the full cooked asset when construction completes.
3. **City info:** pressing `i` shows a new line `Instances     8 active · 2 building · 0 abandoned` whose three numbers match what is on the ground.

A human at the keyboard verifies all three without reading code.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Instance created on development, linked to the engine | `cargo test --lib instance::tests::developing_a_parcel_creates_a_linked_instance -- --nocapture`: zone a 100 m Res Low strip, tick 30; assert `game.instances.len()` equals the developed-parcel count **and** for every instance the engine building found via `inst.engine_building` has `capacity == inst.households_cap.max(inst.jobs_cap)`; print `instances: 8, capacities: [4, 3, ...]` | `assert!(game.instances.len() > 0)` — passes with an unlinked registry |
| Construction state machine drives the engine | freshly developed instance is `UnderConstruction` and its engine building has `operational == false` (so `match_housing`/`match_employment` skip it); after `build_ticks` more ticks it is `Active` with `operational == true`; print both tick indices and the state transitions | `assert!(inst.state != State::Abandoned)` — true for any fresh value |
| Occupancy is the engine's real move-ins | after 30 ticks on a populated city, a residential instance's `occupied()` equals its engine building's `occupants` and is `> 0`; print `occupied 3/4 (engine occupants 3)` | `assert!(occ <= cap)` — true for a hardcoded 0 |
| Property value evolves | run 25 occupied ticks; assert `inst.value()` equals `base × (0.70 + 0.30·occ_ratio) × (1 + appreciation)` and is strictly greater than the frozen base; print `base 91300 -> value 96214` | `assert!(value > 0.0)` |
| Abandonment removes a building from the economy | overbuild a zero-population city so one instance stays at 0 occupants for `ABANDON_AFTER` ticks; assert it flips `Abandoned`, its engine building `operational == false`, and `game.assessed_value()` drops by exactly that instance's value; print assessed before/after | `assert!(state.is_some())` |
| Save/load reproduces instances exactly | play 40 ticks with zoning, `to_save()` → `CivitasGame::from_save()`; assert the two registries are equal field-by-field (state, built_tick, value, occupancy, engine link); print `roundtrip: 12 instances identical` | comparing only `len()` |
| Inspector picking | `game.instance_at(lot.center)` returns that lot's instance id; `game.instance_at([1.0e6, 1.0e6])` returns `None`; print the picked id and asset name | `assert!(result.is_some())` with no None-case check |

---

## 4. Architecture

The full chain this design closes: **cooked `ReadyAssetPayload`** (mesh + material zones + SDF + sockets, exported by Forge, loaded by `AssetCatalog::load_project_assets` into `ready_assets`) → **`BuildingAsset.attributes`** (`BuildingAttributes { households, jobs, property_value, build_cost, upkeep }` ranges) → **develop** (demand gate fires) → **`BuildingInstance`** (ranges frozen to numbers by the lot seed, engine building linked) → **per-tick life** (occupancy, value, state) → **inspector / overlay / save**.

### 4.1 `BuildingRegistry` (new module `src/instance/mod.rs`)

A `Vec<BuildingInstance>`-backed registry owned by `CivitasGame` (new field `pub instances: BuildingRegistry`). Single-threaded, owned game state like every other `CivitasGame` field — no locks, not shared across threads; the render path takes `&CivitasGame` as it does today. Instances are created **only** in `develop_parcels` (the one place parcels become buildings) and never removed; `Abandoned` is a state, not a deletion, so indices (`InstanceId = u32` index into the vec) are stable for the whole game and safe to keep in the UI. The registry replaces the per-call re-derivation: `sector_supply()` and `assessed_value()` become iterations over instances (Active only), keeping their exact signatures so `demand()`, `finances()`, `goods_flow()`, `city_budget()` and every test caller compile unchanged.

### 4.2 Instance creation at develop time

`develop_lot` (src/game/mod.rs:284) already returns the engine building id — `develop_parcels` stops discarding it and calls `self.instances.develop(...)` with: the engine id, the lot's `(plan_idx, lot_idx)` back-pointer, the chosen `AssetKey` (already computed there), the frozen capacities (`lot_asset_homes` / `lot_asset_jobs` — the same `seeded_range(attrs.households, lot.seed ^ 0xA551_0001)` / `^ 0xA551_0002` numbers used for engine capacity today, so engine `capacity` and instance capacity agree by construction), the frozen base value (`seeded_range(attrs.property_value, lot.seed ^ 0xA551_0003)`, today's `lot_assessed_value` number), the asset `level`, the current `tick_index`, and `build_ticks = 2 + asset.level as u64` (an L1 house takes 3 ticks ≈ 3 months). Immediately after creation the engine building's `operational` is set `false` (pub field on `vox_sim::buildings::Building`) so `find_nearest_with_vacancy` — the gate both `match_housing` and `match_employment` go through — assigns nobody to a construction site. **No engine-crate change is needed.** Known quirk, accepted: `CitySim::vacant_housing()` (migration attraction) does not filter `operational`, so migrants may be attracted 2–4 ticks early; they simply wait unhoused until construction completes.

### 4.3 Per-tick instance simulation (`BuildingRegistry::tick`)

Called from `CivitasGame::tick` as step **1c**, right after `develop_parcels()` (so the engine's housing/employment matching from step 1 is settled and freshly developed parcels are registered) and before the budget settle (so this tick's taxes see the new states). For each instance, in registry order, all rules pure functions of (engine building state, `tick_index`, frozen fields) — this purity is what keeps replay-based saves exact:

- **UnderConstruction { ticks_left }**: decrement; at 0 → `Active`, set engine `operational = true`.
- **Active**: read `occupants` off the linked engine building (the real move-ins / filled jobs). Update value: `value = base × (0.70 + 0.30 × occ_ratio) × (1.0 + appreciation)` where `occ_ratio = occupants/capacity` (1.0 if capacity 0) and `appreciation` grows `+0.002` per tick with `occ_ratio ≥ 0.5`, capped at `+0.25`, never decaying. Track `vacant_streak`: if `capacity > 0 && occupants == 0`, increment, else reset; at `ABANDON_AFTER = 12` consecutive ticks → `Abandoned`, set engine `operational = false` (the engine's `remove_departures` already decremented occupants as people left; flipping operational stops re-assignment).
- **Abandoned**: contributes nothing to `sector_supply()` or `assessed_value()`. Recovery hysteresis: when the instance's sector demand bar (`zone_sector_budget` mapping on the lot's zone) exceeds `0.5` for `RECOVER_AFTER = 3` consecutive ticks → back to `Active`, `operational = true`, `vacant_streak = 0`, `appreciation` reset to 0.

### 4.4 Economy reads move to the registry

`sector_supply()` iterates `instances.iter().filter(|i| i.state == Active)` summing `households_cap` into housing and `jobs_cap` into the sector from `zone_job_sector(lot.zone)` (lot reached via the back-pointer); lots with no asset attributes keep today's `occupancy()` fallback at develop time by freezing those numbers into the instance instead. `assessed_value()` sums `inst.value()` over Active instances. Behavioral delta, intentional: supply and assessed value now start when construction **completes**, not when the parcel develops — `the_general_budget_taxes_property_and_business...` and `zoning_commercial_adds_commercial_supply...` (src/game/mod.rs tests) gain 2–4 ticks of warm-up; the plan implementing this design updates those expectations explicitly.

### 4.5 Inspector + overlay (the Done When surface)

- New `BuildTool::Inspect` variant (src/ui/build_tools.rs) and a new **last** category `("Inspect", vec![Inspect])` in `build_categories()` — appended last so existing category indices and the 1–9 hotkeys are untouched. `tile_label` returns `"Inspect"`; `selected_zone` returns `None` (zoning UI stays off).
- Picking: `CivitasGame::instance_at(p: [f32; 2]) -> Option<u32>` rotates `p` into each developed lot's frame (`center`, `facing_angle`) and tests `|x| ≤ width/2 && |z| ≤ depth/2`; lots do not overlap, first hit wins.
- `play.rs`: `GameView` gains `inspected: Option<u32>`. In the left-click chain (src/bin/play.rs:815), after the HUD branches and before `gv.place_at`, add: if the current tool is `Inspect`, `screen_to_ground` → `game.instance_at(pos)` → set `inspected`. `GameView::render` draws it with the existing `hud::draw_city_info_window(&mut pixels, w, h, &format!("Building — {name}"), &inspect_lines)`; `hud::info_window_close_at` closes it; Esc clears it before falling through to the menu (same pattern as `draft`/`info_open`).
- `inspect_lines` (new `GameView` method) formats exactly the six lines in §2 from `BuildingInstance` + `BuildingAsset` (name/id/level via `asset_catalog.get(inst.asset.as_str())`).
- `info_lines()` (src/bin/play.rs:322) gains `format!("Instances         {} active · {} building · {} abandoned", r.active_count(), r.constructing_count(), r.abandoned_count())`.

### 4.6 State-driven visuals from the cooked payload

`city_scene_inner` (src/render_gpu/mod.rs, the loop at ~line 688 that calls `place_ready_asset_sdf_instance` / `place_ready_asset_mesh_renderables` / `place_ready_asset_splats` per developed lot) looks up `game.instances.for_lot(pi, li)` and derives an `InstanceVisual { y_scale: f32, tint: [f32; 3] }`: UnderConstruction → `y_scale = 0.3 + 0.7 × progress`, tint `[0.62, 0.60, 0.58]` (bare-concrete grey); Active → identity; Abandoned → `y_scale = 1.0`, tint `[0.45, 0.45, 0.45]`. The three `place_ready_asset_*` helpers gain the `InstanceVisual` parameter (defaulting to identity for ploppables/services, which have no instance) and apply it to atom scale/color and mesh triangle rgb. This is presentation-only — the cooked `ReadyAssetPayload` is never mutated — and is renderer-agnostic: the same scene feeds the wgpu interactive path and the Spectra still path.

### 4.7 Save/load: replay, not snapshot

No save-schema change; `SAVE_VERSION` stays 5. The save system (src/game/save.rs) is a replayable authored-action log — `GameSave { actions, action_ticks, ticks, ... }` interleaved by `CivitasGame::replay` — and every instance rule in §4.3 is a deterministic function of replayed state, so the registry is **rebuilt exactly** by `CivitasGame::from_save` / `from_save_on`. The design constraint this imposes (and the roundtrip test enforces): no RNG, no wall-clock, no HashMap-iteration-order dependence anywhere in `BuildingRegistry::tick` — instances are visited in registry (creation) order, the same order `develop_parcels` produces deterministically.

### 4.8 ENGINE/GAME boundary

Everything above lives in `civitas_care`. Engine usage is read/write of **existing** public `vox_sim::buildings::Building` fields (`id`, `capacity`, `occupants`, `operational`, `position`) — the same fields `CivitasGame::tick` already mutates for care-gating (src/game/mod.rs:686–694). No edits to any `~/src/ochroma` crate; `vox_render`/`vox_core` never learn what a household or a property value is.

---

## 5. Data Models

All types are game-internal (single crate); fields follow the codebase's existing `pub`-field convention (`PlannedLot`, `Building`), with the mutating transitions confined to `BuildingRegistry` methods.

```rust
// src/instance/mod.rs

/// Stable handle: index into BuildingRegistry.instances. Instances are never
/// removed (Abandoned is a state), so the id is valid for the whole game.
pub type InstanceId = u32;

/// Lifecycle of a developed building.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceState {
    /// Rising; engine building exists but operational == false.
    UnderConstruction { ticks_left: u32 },
    /// Built and part of the economy.
    Active,
    /// Vacant too long; out of the economy until demand recovers.
    Abandoned,
}

/// A live building on a developed parcel — the game-side identity of one
/// engine `vox_sim::buildings::Building`, carrying everything the engine
/// doesn't model: frozen capacities, evolving value, lifecycle state.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingInstance {
    /// `vox_sim::buildings::Building::id` — the link develop_lot used to drop.
    pub engine_building: u32,
    /// Back-pointer into `CivitasGame::placed_lots[plan_idx].lots[lot_idx]`.
    pub plan_idx: u32,
    pub lot_idx: u32,
    /// The cooked catalog asset this instance grew into.
    pub asset: crate::asset::AssetKey,
    pub state: InstanceState,
    /// Frozen at develop time from BuildingAttributes ranges + lot seed
    /// (same numbers as lot_engine_capacity, so engine capacity agrees).
    pub households_cap: u32,
    pub jobs_cap: u32,
    /// Frozen base assessed value (today's lot_assessed_value number).
    pub base_value: f64,
    /// Cumulative appreciation factor, 0.0..=0.25 (see §4.3).
    pub appreciation: f64,
    /// Last-read engine occupants (homes or filled jobs).
    pub occupants: u32,
    /// Consecutive Active ticks at zero occupancy (abandonment counter).
    pub vacant_streak: u32,
    /// Consecutive ticks of sector demand > 0.5 while Abandoned (recovery).
    pub recover_streak: u32,
    pub level: u8,
    pub built_tick: u64,
}

impl BuildingInstance {
    /// Current assessed value: base × occupancy factor × appreciation.
    pub fn value(&self) -> f64 { /* §4.3 formula */ }
    pub fn capacity(&self) -> u32 { self.households_cap.max(self.jobs_cap) }
}

/// Owned by CivitasGame; created in develop_parcels, ticked in step 1c.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildingRegistry {
    instances: Vec<BuildingInstance>,
}

/// Presentation-only state visual (src/render_gpu/mod.rs).
#[derive(Debug, Clone, Copy)]
pub struct InstanceVisual {
    pub y_scale: f32,   // 0.3..=1.0
    pub tint: [f32; 3], // multiplied into atom color / triangle rgb
}
```

Constants (in `src/instance/mod.rs`): `ABANDON_AFTER: u32 = 12`, `RECOVER_AFTER: u32 = 3`, `APPRECIATION_PER_TICK: f64 = 0.002`, `APPRECIATION_CAP: f64 = 0.25`, `fn build_ticks(level: u8) -> u32 { 2 + level as u32 }`.

---

## 6. API

```rust
// src/instance/mod.rs — all single-threaded, called only from CivitasGame.
impl BuildingRegistry {
    /// Register a freshly developed parcel. Called ONLY from
    /// CivitasGame::develop_parcels with the id develop_lot returns.
    /// Sets the engine building operational = false. Returns the InstanceId.
    pub fn develop(
        &mut self,
        buildings: &mut vox_sim::buildings::BuildingManager,
        engine_building: u32,
        plan_idx: u32,
        lot_idx: u32,
        asset: crate::asset::AssetKey,
        households_cap: u32,
        jobs_cap: u32,
        base_value: f64,
        level: u8,
        now_tick: u64,
    ) -> InstanceId;

    /// One simulation step (§4.3). `demand` is this tick's bars (for recovery).
    /// Deterministic: pure function of arguments + stored state, visited in
    /// creation order. Mutates engine `operational` on state transitions.
    pub fn tick(
        &mut self,
        buildings: &mut vox_sim::buildings::BuildingManager,
        lots: &[crate::lot::layout::LotPlan],
        demand: &crate::demand::Demand,
    );

    pub fn get(&self, id: InstanceId) -> Option<&BuildingInstance>;
    pub fn for_lot(&self, plan_idx: u32, lot_idx: u32) -> Option<&BuildingInstance>;
    pub fn iter(&self) -> impl Iterator<Item = &BuildingInstance>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn active_count(&self) -> usize;
    pub fn constructing_count(&self) -> usize;
    pub fn abandoned_count(&self) -> usize;
}

// src/game/mod.rs — additions to CivitasGame (existing signatures unchanged):
impl CivitasGame {
    /// The instance under a world XZ point (rotated-rect test on developed
    /// lots). None over roads, vacant parcels, ploppables, empty ground.
    pub fn instance_at(&self, p: [f32; 2]) -> Option<InstanceId>;
}
// CHANGED INTERNALS (same public signatures):
//   sector_supply(&self) -> (crate::finance::SectorJobs, u32)  — Active instances only
//   assessed_value(&self) -> f64                               — sum of inst.value()
//   develop_parcels(&mut self)                                  — captures develop_lot's id

// src/render_gpu/mod.rs — the three placers gain a visual parameter:
//   place_ready_asset_splats(..., visual: InstanceVisual) -> Option<Vec<GaussianSplat>>
//   place_ready_asset_mesh_renderables(..., visual: InstanceVisual) -> ...
//   place_ready_asset_sdf_instance(..., visual: InstanceVisual) -> ...
// (identity InstanceVisual { y_scale: 1.0, tint: [1.0; 3] } for non-instance callers)
```

Panics: none — `develop` with an unknown `engine_building` id is impossible by construction (the id comes from `develop_lot` in the same call stack); `tick` with a dangling back-pointer is a logic error guarded by `debug_assert!`.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `BuildingRegistry::develop()` | `CivitasGame::develop_parcels` build pass | `src/game/mod.rs` (~line 887) | captures the id `develop_lot` currently discards; same loop that sets `lot.developed = true` |
| `BuildingRegistry::tick()` | `CivitasGame::tick` step **1c** | `src/game/mod.rs` (after `develop_parcels()` call at line 636) | before demographics/budget so taxes see new states |
| registry-backed `sector_supply` / `assessed_value` | existing callers unchanged | `src/game/mod.rs:768, :912` | signatures identical; bodies iterate instances |
| `CivitasGame::instance_at()` | Inspect-tool left-click branch | `src/bin/play.rs` (MouseInput chain, ~line 848, before `place_at`) | via existing `render_gpu::screen_to_ground` |
| `BuildTool::Inspect` + `("Inspect", …)` category | `build_categories()` | `src/ui/build_tools.rs:42` | appended **last**; hotkeys/categories stable |
| `GameView::inspect_lines` + window draw | `GameView::render` | `src/bin/play.rs` (~line 474, beside the info window) | reuses `hud::draw_city_info_window` / `info_window_close_at` |
| `Instances … active · building · abandoned` line | `GameView::info_lines` | `src/bin/play.rs:322` | reads the three registry counters |
| `InstanceVisual` application | per-lot branch of `city_scene_inner` | `src/render_gpu/mod.rs` (~line 688–760) | `instances.for_lot(pi, li)`; identity for ploppables/services |
| registry equality in save roundtrip | new test beside the save tests | `src/game/mod.rs` tests | `to_save()` → `from_save()` → `assert_eq!(a.instances, b.instances)` |

---

## 8. Open Questions

- [ ] **Level-up / densification:** `BuildingAsset.level` (1–5) and `max_level(zone)` exist, and `lot_asset` documents "upgrade mechanics can later replace this". Should an Active instance with high appreciation + high sector demand re-enter `UnderConstruction` toward a level+1 asset? Deferred — the state machine has room for an `Upgrading` variant; decide when the upgrade plan is written.
- [ ] **Abandonment thresholds:** `ABANDON_AFTER = 12` / `RECOVER_AFTER = 3` / demand `> 0.5` are first-cut tunings. The implementing plan must keep them as named constants and may retune from playtest, but tests must assert against the constants, not literals.
- [ ] **Care/service ploppables as instances:** care buildings and city services have their own systems (`CareSystem`, `sim.services`). Folding them into the registry would unify the inspector but tangle three lifecycles. Out of scope here (see §9); revisit after zoned instances ship.

---

## 9. Out of Scope

- Level-up/downgrade between density levels (asset `level` stays frozen at develop time).
- Instances for ploppables: care buildings, city services, parks, roads — only zoned, demand-developed parcels get instances.
- Per-citizen ↔ per-building residency mapping beyond the engine's `occupants` counter (no "list the families in this house").
- Demolition/bulldoze tool and re-development of Abandoned parcels into different assets.
- Any engine-crate change: `vox_sim`'s `vacant_housing()` not filtering `operational` (early migration attraction during 2–4 construction ticks) is accepted, not fixed.
- Save-schema changes: no `SAVE_VERSION` bump, no instance snapshotting — replay determinism is the persistence mechanism.
- Spectra path-traced rendering of instance states (the still-render harness consumes the same scene; tuning its look is the Task 9 design's territory).

---

## 10. Related Plans / Designs

- Depends on: nothing unshipped — binds entirely to existing types: `CivitasGame` / `develop_parcels` / `develop_lot` (src/game/mod.rs), `PlannedLot` / `LotPlan` (src/lot/layout.rs), `BuildingAsset` / `BuildingAttributes` / `AssetKey` / `ReadyAssetPayload` (src/asset/mod.rs), `Demand` / `compute_demand` (src/demand/mod.rs), `SectorJobs` / `settle` (src/finance/mod.rs), `CityBudget` (src/budget/mod.rs), `GameSave` / `AuthoredAction` (src/game/save.rs), `vox_sim::buildings::{Building, BuildingManager}`.
- Required before: the **Living Building Instances implementation plan** (Phase 2+ item in `[SOTA City Block Phase 1](../plans/2026-06-10-sota-city-block-phase1.md)`), which turns each §3 capability row into a TDD task.
- Related: `[Virtualized Splat Rendering Design](./2026-06-10-virtualized-splat-rendering-design.md)` (Task 9) — instance-state visuals (§4.6) must stay compatible with whatever residency/LOD scheme it adopts, which is why they are expressed as a pure per-asset `InstanceVisual` transform, not baked geometry.
