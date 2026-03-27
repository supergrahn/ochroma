# Wave 1: Make It Playable — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Ochroma prototype into a playable city builder where a player can draw roads, zone land, grow buildings, manage citizens, and see a living city.

**Architecture:** All gameplay wiring happens in vox_app modules. Simulation logic in vox_sim is already implemented — this plan connects it to the ECS and UI.

**Tech Stack:** Existing Rust workspace, bevy_ecs 0.16, egui 0.31, wgpu 24.

---

### Task 1: Interactive Road Drawing

**Files:**
- Modify: `crates/vox_app/src/ui.rs`
- Modify: `crates/vox_app/src/placement.rs`
- Modify: `crates/vox_app/src/main.rs`

Connect the Road mode so clicking two points draws a road.

- [ ] **Step 1:** Add `road_start: Option<Vec3>` field to PlopUi. In `handle_viewport_click` for `UiMode::Road`: if `road_start` is None, store click as start. If start exists, push `UiAction::BuildRoad { start, end, road_type }` and clear start.

- [ ] **Step 2:** Add `BuildRoad { start: Vec3, end: Vec3 }` variant to `UiAction`.

- [ ] **Step 3:** In `placement.rs`, handle `BuildRoad`: create a `RoadSegment`, call `road_builder::spawn_road_visual()` to render it, add to `SimulationState.roads` (add a `roads: RoadNetwork` field to SimulationState if not present), add a corresponding traffic segment.

- [ ] **Step 4:** In the egui Road mode toolbar, show "Click to set start point" or "Click to set end point" feedback.

- [ ] **Step 5:** Commit: "feat(vox_app): interactive road drawing with two-click placement"

---

### Task 2: Zoning Paint Tool

**Files:**
- Modify: `crates/vox_app/src/placement.rs`
- Modify: `crates/vox_app/src/simulation.rs`

Connect the Zone mode so clicking creates zone plots and triggers growth.

- [ ] **Step 1:** In `placement.rs`, handle `UiAction::ZoneArea`: convert the zone_type string to `vox_sim::zoning::ZoneType`, call `sim.zoning.zone_plot()` with a 10×10m plot at the click position.

- [ ] **Step 2:** Add periodic growth ticking to main.rs: every 2 seconds of real time, call `growth::growth_tick(&mut self.world)`.

- [ ] **Step 3:** The growth system (already implemented in growth.rs) will spawn buildings on undeveloped plots when demand is sufficient.

- [ ] **Step 4:** Commit: "feat(vox_app): zoning paint creates plots that trigger building growth"

---

### Task 3: Service Placement Wiring

**Files:**
- Modify: `crates/vox_app/src/placement.rs`

Connect service placement to the simulation.

- [ ] **Step 1:** In `placement.rs`, handle `UiAction::PlaceService`: convert service_type string to `ServiceType`, call `sim.services.place_service()`, spawn a visual asset (use demo building as placeholder).

- [ ] **Step 2:** Each simulation tick, iterate citizens and update their `needs.health`, `needs.safety`, `needs.education` based on whether their residence position is within service coverage.

- [ ] **Step 3:** Commit: "feat(vox_app): service building placement connected to citizen needs"

---

### Task 4: Citizen Daily Routines

**Files:**
- Modify: `crates/vox_sim/src/citizen.rs`
- Create: `crates/vox_sim/src/routines.rs`
- Test: `crates/vox_sim/tests/routines_test.rs`

Citizens need daily schedules that generate agent movement.

- [ ] **Step 1:** Add `DailyState` enum to citizen.rs: `AtHome`, `Commuting`, `AtWork`, `Shopping`, `Returning`.

- [ ] **Step 2:** Create routines.rs with `update_daily_routines(citizens, hour_of_day, buildings)`:
```
6:00-8:00  → Workers leave home, state = Commuting
8:00-17:00 → At work
17:00-18:00 → Returning home
18:00-20:00 → Some go shopping (commercial districts)
20:00-6:00 → At home
```

- [ ] **Step 3:** Each routine transition generates an agent destination (home → workplace, workplace → home, etc.). The AgentManager then moves them.

- [ ] **Step 4:** Test: citizen starts AtHome at hour 5, transitions to Commuting at hour 7.

- [ ] **Step 5:** Commit: "feat(vox_sim): citizen daily routines with commute/work/shop/home cycle"

---

### Task 5: Building Functionality

**Files:**
- Create: `crates/vox_sim/src/buildings.rs`
- Modify: `crates/vox_sim/src/lib.rs`
- Test: `crates/vox_sim/tests/buildings_test.rs`

Buildings need to DO things — provide housing, jobs, goods.

- [ ] **Step 1:** Create buildings.rs:
```rust
pub struct Building {
    pub id: u32,
    pub building_type: BuildingType,
    pub position: [f32; 2],
    pub capacity: u32,      // housing units or job slots
    pub occupants: u32,
    pub operational: bool,   // requires utilities to function
}

pub enum BuildingType {
    Residential { housing_units: u32 },
    Commercial { job_slots: u32, goods_output: f32 },
    Industrial { job_slots: u32, resource_output: ResourceType, output_rate: f32 },
    Service { service_type: ServiceType },
}

pub struct BuildingManager {
    buildings: Vec<Building>,
    next_id: u32,
}
```

- [ ] **Step 2:** `BuildingManager` methods: `add_building()`, `find_nearest_with_vacancy(position, type)`, `total_housing()`, `total_jobs()`, `assign_occupant()`, `remove_occupant()`.

- [ ] **Step 3:** Test: add residential building, check capacity, assign occupant, verify count.

- [ ] **Step 4:** Commit: "feat(vox_sim): building functionality with housing, jobs, and goods production"

---

### Task 6: Resource Flow

**Files:**
- Modify: `crates/vox_sim/src/economy.rs`
- Create: `crates/vox_sim/src/supply_chain.rs`

Connect resource production to buildings and consumption to citizens.

- [ ] **Step 1:** Create supply_chain.rs with a `SupplyChainManager`:
```rust
pub fn tick(buildings: &BuildingManager, chain: &mut SupplyChain, dt: f32) {
    // Industrial buildings produce raw resources
    // Processing buildings convert raw → refined
    // Markets/commercial distribute to citizens
    // Citizens consume goods (food, materials)
}
```

- [ ] **Step 2:** Each industrial building adds to its resource stock per tick. Each commercial building consumes from supply chain and serves citizens.

- [ ] **Step 3:** Commit: "feat(vox_sim): resource production and consumption connected to buildings"

---

### Task 7: Overlay System

**Files:**
- Create: `crates/vox_app/src/overlays.rs`
- Modify: `crates/vox_app/src/ui.rs`

Visual overlays that show data on the terrain.

- [ ] **Step 1:** Create overlays.rs with overlay types:
```rust
pub enum OverlayType { None, Traffic, LandValue, ZoneColour, ServiceCoverage }
```

- [ ] **Step 2:** For each overlay, generate coloured translucent splats placed on the terrain grid:
- Traffic: red where density > 50%, green where low
- LandValue: green for high, grey for low
- ZoneColour: blue=residential, yellow=commercial, purple=industrial
- Coverage: blue circles around service buildings

- [ ] **Step 3:** Add overlay toggle buttons to the UI. When active, overlay splats are added to the visible set.

- [ ] **Step 4:** Commit: "feat(vox_app): toggle-able data overlays for traffic, zones, and services"

---

### Task 8: Procedural Map Generation

**Files:**
- Create: `crates/vox_core/src/mapgen.rs`
- Test: `crates/vox_core/tests/mapgen_test.rs`

Generate a terrain heightmap with hills, flat areas, and a river.

- [ ] **Step 1:** Implement simple noise-based terrain generation:
```rust
pub fn generate_map(seed: u64, size: f32) -> Vec<GaussianSplat> {
    // Use simple value noise for hills
    // Carve a river along a sine curve
    // Flat areas near river (good for building)
    // Hills at edges
}
```

- [ ] **Step 2:** Different surface materials based on height: water at bottom, grass on flats, rock on hills.

- [ ] **Step 3:** Replace the flat terrain in terrain_setup.rs with the procedural map.

- [ ] **Step 4:** Test: generated map has splats, height varies, river area exists.

- [ ] **Step 5:** Commit: "feat(vox_core): procedural map generation with hills, river, and varied terrain"

---

## Summary

| Task | Feature | Makes playable? |
|------|---------|----------------|
| 1 | Interactive road drawing | Yes — roads are the skeleton |
| 2 | Zoning paint tool | Yes — zones drive growth |
| 3 | Service placement | Yes — citizens need services |
| 4 | Citizen daily routines | Yes — city feels alive |
| 5 | Building functionality | Yes — buildings do things |
| 6 | Resource flow | Yes — economy works |
| 7 | Overlay system | Yes — player can see data |
| 8 | Map generation | Yes — terrain isn't flat |
