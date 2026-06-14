# Design: Deep Simulation (2026-06-10)

**Status:** Draft
**Scope:** SOTA item 5 — give Urban Horizon a deep, observable economy on top of the BuildingInstance registry: persistent households anchored to real houses, a per-lot land-value field that feeds assessed value and taxation, capacity-aware school/healthcare access wired into the care gate, per-instance goods inventory and multi-tenant business programs, flow-based traffic assignment on the player's road graph, and a zoning growth loop that reacts to services and land value. All new code lives in the GAME layer (`~/Ochroma/projects/urban_horizon`); engine crates (`~/src/ochroma`) are untouched.
**Related:** `[Living Building Instances Design](./2026-06-10-living-building-instances-design.md)` (SOTA item 4 — **prerequisite**, in implementation now), `[SOTA City Block Phase 1 Plan](../plans/2026-06-10-sota-city-block-phase1.md)` (sequencing: "Deep simulation … sequenced after instances")

---

## 1. Problem Statement

Honest inventory of today's code. What exists: the RCI demand model (`src/demand`), aggregate taxes/budget (`src/finance`, `src/budget`), a city-wide 3-good chain (`src/goods`), a real road graph with classes and engine export but no pathfinding (`src/road`), the care gate with capacity + staffing + radius + water barriers (`src/game_mechanics/care`), radius-only general-service coverage (`src/game_mechanics/services`), a multi-tenant `BuildingProgram` model (`src/lot/occupancy.rs`), and — landing now — the `BuildingRegistry` with frozen caps, evolving `value()`, and the Inspect window. What is missing, as concrete symptoms:

- **Households have no place and no identity.** `Demographics::build` (src/game_mechanics/care/demographics.rs:74) rebuilds households from scratch every tick and locates every one at the synthetic strip `[residence_id × 10, 0]` (`residence_pos`, line 65) — a kindergarten's 1500 m radius is measured to a phantom point, not to the house the family actually occupies on a developed lot. Engine `match_housing` (vox_sim/src/employment.rs:48) matches everyone from `[0,0]`, so houses fill in nearest-to-origin order regardless of where the player zoned.
- **Land value does not exist.** `BuildingInstance::value()` is `base × (0.70 + 0.30·occ) × (1 + appreciation)` — a house beside a park, a school, and an avenue is assessed identically to one wedged between two foundries on a back alley. `vox_sim::land_value::LandValueGrid` exists in the engine but is wired to nothing (`CitySim` never constructs one).
- **Schools and healthcare are booleans.** `ServiceBuilding` carries `capacity`/`current_users` (vox_sim/src/services.rs:26–27) but `current_users` is never written anywhere in either repo; `service_coverage` (src/game_mechanics/services/mod.rs:181) only checks radius presence. One PrimarySchool with 500 seats "covers" 5,000 children. The care gate — the game's core mechanic — never feels a seat shortage.
- **Goods are a city-wide scalar.** `compute_flow` (src/goods/mod.rs:110) balances aggregate production vs consumption; no shop has inventory, no factory has a customer, and a farm across the map is identical to one next door. `BuildingProgram` (malls = many shops + restaurants) exists but the asset path (`lot_asset_jobs`, src/game/mod.rs:438) flattens every instance to one number in one sector; MixedUse drops its homes entirely (`zone_building_type(MixedUse) = Commercial`).
- **Roads carry nothing.** `RoadNetwork` (src/road/mod.rs) has nodes, segments, widths — and no adjacency, no path query, no load. `vox_sim::traffic::TrafficNetwork` (Greenshields per segment) is wired to nothing. The player can build a city with one alley as its only artery and the sim never notices.
- **Zoning growth is blind to place.** `develop_parcels` (src/game/mod.rs:843) gates only on the sector demand bar and develops parcels in insertion order — an unserved fringe lot develops before the lot beside the school and park.

---

## 2. Done When

Umbrella: running `cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play`, choosing **New Game → Demo City → Start Game**, and pressing **Space** ~10 times, a human at the keyboard can verify every milestone gate below without reading code. Each milestone is independently shippable in this order.

### M1 — Land value & anchored households

Press `v`: every developed lot tints green→red by its land value (legend `Land value  low ▓ high` in the bottom-left). With the **Inspect** tool (from the instances design), clicking a house shows a new line `Land value   0.41`. Plop a **Park** (Services → Park) two lots away, press Space once, re-click: the line reads a strictly higher number (e.g. `0.41 → 0.53`) and the `Value` line is higher than it would have been (land factor `0.9 + 0.2·lv`). Pressing `i` shows `Land value        avg 0.52 · top 0.78` and the `City budget` taxes line moves with it.

### M2 — School seats & healthcare capacity (the care tie-in)

Pressing `i` shows `School seats      412 / 510 enrolled` and `Health capacity   1,200 cap · 980 in reach`. In a city whose school-age children exceed total school seats, the seatless children are uncovered: `Gated workers` rises and the notification feed shows `School seats full — N children unplaced`. Plopping a second **Primary School** and pressing Space twice drops `Gated workers` back and the enrolled line shows the new capacity. Clicking a house in Inspect shows `Access       school 0.4 km · health 1.1 km · care OK`.

### M3 — Traffic (flow-based assignment)

Press `t`: every road segment draws as a colored polyline — green (load < 50%), amber (< 80%), red (≥ 80%) — width scaled by road class. `i` shows `Traffic           load 34% avg · worst 87% · off-grid 8%`. Zone a residential district and a commercial district at opposite ends of a single avenue, Space ×6: the avenue turns red and `worst` exceeds 80%. Lay a parallel street connecting the same two areas, Space ×2: the avenue's color drops and the printed `worst` number falls.

### M4 — Goods inventory & business tenants

Inspect a commercial building: `Tenants      5 shops · 1 restaurant   (jobs 28)` and `Inventory    62% · supplier 1.8 km`. `i` shows `Goods local       food 64% · goods 41% · retail 100%`. In a city with zero industry, every shop's inventory sits at the import floor (`35%`) and `Working` in the info window is lower than the same city with industry zoned next door (throttled effective jobs); zoning industry within ~2 km and pressing Space ×6 raises the inspected shop's inventory percentage.

### M5 — Zoning growth feedback

Zone two equal Res Low strips: one beside the park + school, one on the bare fringe. Space ×6: the served strip visibly develops first (houses rise there before the fringe). `i` shows `Vacant lots       14 (3 stalled: low desirability)` — fringe lots with land value below the desirability floor stay vacant even under high residential demand, until the player services them.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Land value responds to amenities | `cargo test --lib land_value::tests::a_park_raises_neighbouring_lot_value -- --nocapture`: compute `lv` for a developed lot, plop a Park 100 m away, recompute; assert `lv_after > lv_before + 0.05` and both in `[0.05, 1.0]`; print both numbers | `assert!(lv > 0.0)` — true for the bare base constant |
| Instance value couples to land value | with `lv = 0.5` the instance `value()` equals the instances-design formula exactly (`base × (0.70+0.30·occ) × (1+appr)`); with `lv = 1.0` it is `1.1×` that; print all three values | `assert!(value > 0.0)` |
| Households anchor to their house | after 30 ticks on a zoned city, a household whose residence is an instance-linked engine building has `anchor()` equal to that lot's `center` (not `[id×10, 0]`); a seeded household (id ≥ 100_000) keeps `[id×10, 0]`; print both anchors | `assert!(anchor.is_some())` |
| School seats gate parents | city with 1 PrimarySchool (500 seats) and 600 enrolled-age children: assert `seats.enrolled == 500`, `seats.unplaced == 100`, and `gated_workers` strictly greater than the same city with 2 schools; print the triple | `assert!(report.rows.len() > 0)` |
| Healthcare capacity is counted | `ServiceAccess` writes `current_users` onto the engine `ServiceBuilding` so `utilisation()` returns `users/capacity > 0`; assert the exact assigned count equals min(in-range demand, capacity); print it | `assert!(b.operational)` |
| Traffic loads the shortest path | two clusters joined by one avenue (3 segments): all home→work trips land on those segments; assert each segment's `volume` equals the commuter count and `load_ratio == volume / cap(Avenue)`; print per-segment volumes | `assert!(model.segments() > 0)` |
| A parallel road relieves congestion | add a parallel street with lower total weighted cost for half the OD pairs; assert the avenue's `load_ratio` strictly decreases and total trips are conserved (sum of volumes constant); print before/after worst load | `assert!(load <= 1.0)` — vacuously true |
| Inventory falls with supply distance | commercial instance 0.5 km from an Active industry instance vs one 4 km away, same city totals: assert `inv_near > inv_far` and both `≥ IMPORT_FLOOR`; assert effective jobs = `jobs_cap × (0.6 + 0.4·inv)` rounded; print all four numbers | `assert!(inv <= 1.0)` |
| Tenant mix preserves frozen totals | realize tenants for a CommercialRegional instance with `jobs_cap = 38`: assert `tenants.iter().map(jobs).sum() == 38` and at least 2 tenant kinds; MixedUse instance contributes BOTH its `households_cap` to housing and `jobs_cap` to commercial in `sector_supply()`; print the mix | `assert!(!tenants.is_empty())` |
| Desirability orders development | two strips, one beside park+school: after 6 ticks the served strip has strictly more developed lots; a lot with `lv < DEV_DESIRABILITY_MIN` stays vacant for 60 ticks under residential bar > threshold; print developed counts and the stalled lot's lv | `assert!(developed > 0)` |
| Replay reproduces all deep-sim state | play 40 ticks with zoning + a park + a school, `to_save()` → `from_save()`; assert field-by-field equality of `LandValueField` outputs, `ServiceAccess` report, `TrafficModel` segment volumes, per-instance inventory, and household registry; print `roundtrip: identical (lots N, segments M, households K)` | comparing only `tick_index` |

---

## 4. Architecture

One new top-level pass — **step 1d, the deep-sim pass** — runs inside `CivitasGame::tick` after the registry tick (1c) and before demographics/care (2), in this fixed order: households → land value → service access → traffic → goods. Each stage is a pure, deterministic function of (engine state, registry state, prior stage outputs, `tick_index`); outputs are cached on `CivitasGame` fields and read by care, finances, the budget, the renderer overlays, and the inspector. No stage uses RNG, wall-clock, or unordered iteration — this is what keeps the action-log replay exact (§4.8).

```
sim.tick → develop_parcels → instances.tick ─┬─ 1d.1 households.reconcile (anchors)
                                             ├─ 1d.2 land_value.recompute (per lot)
                                             ├─ 1d.3 services_access.assign (seats/cap)
                                             ├─ 1d.4 traffic.assign (OD → segment loads)
                                             └─ 1d.5 goods_local.distribute (inventory)
        → demographics/care gate (now anchored + seat-aware) → care economy → city budget
```

### 4.1 `HouseholdRegistry` (new module `src/household/mod.rs`)

Persistent identity over the per-tick `Demographics` view. Keyed by residence id (engine building id, or the synthetic 100_000+ seed ids); a record is created on first sight (`formed_tick`), updated each tick (member ids, counts), and marked `dissolved_tick` when its residence empties. The headline fix is **anchoring**: `anchor(residence)` resolves residence → engine building → `BuildingRegistry` instance → that lot's `center`; seeded/unlinked residences keep today's `[id × 10, 0]` convention so existing scenarios and tests replay unchanged. `Demographics::build` gains a position-resolver parameter (defaulting to the legacy convention) and `CivitasGame::tick` passes the registry's resolver — care radii, water barriers, and service distances now measure to real houses. Intra-city moves (a household relocating to a better-served house) are **v2**; v1 household turnover remains the engine's existing arrivals/departures. Single-threaded owned game state, like every `CivitasGame` field.

### 4.2 `LandValueField` (new module `src/land_value/mod.rs`)

Per-**lot** (not per-cell — lots are the taxable, inspectable unit), recomputed every tick for developed and vacant lots alike (vacant lots need it for §4.7). For lot `L` with center `p`:

```
lv(L) = clamp( BASE
  + W_PARK    × falloff(nearest Park service, 500 m)
  + W_SCHOOL  × school_access(p)        // in radius AND seats not exhausted (last tick's report)
  + W_HEALTH  × health_access(p)        // clinic/hospital in radius, capacity-weighted
  + W_SAFETY  × police_coverage(p)
  + W_CARE    × care_in_reach(p)        // any non-full, staffed care building in radius
  + road_bonus(frontage class)          // Avenue 0.05 · Street 0.03 · Alley 0.0
  − W_IND     × falloff(nearest Active industrial instance, 300 m)
  − W_CONG    × frontage_load_ratio     // last tick's traffic on the nearest segment
, LV_MIN, 1.0)
```

Constants (named, test-asserted): `BASE = 0.35`, `W_PARK = 0.15`, `W_SCHOOL = 0.10`, `W_HEALTH = 0.10`, `W_SAFETY = 0.05`, `W_CARE = 0.10`, `W_IND = 0.15`, `W_CONG = 0.10`, `LV_MIN = 0.05`. Traffic and seat inputs are **last tick's** outputs (one-tick lag breaks the cycle land→traffic→land deterministically). Engine `vox_sim::land_value::LandValueGrid` is deliberately **not** used: it is cell-based, unwired, and would put game-tuned amenity weights in an engine crate.

**Value coupling:** `BuildingInstance::value()` gains a multiplicative land factor `× (0.9 + 0.2·lv)` — **neutral at lv = 0.5**, so every number in the instances design and its tests is preserved at neutral land value; the implementing plan updates only tests that build amenity-rich fixtures. Property tax then responds to land value through the existing `assessed_value × policy.property` line — no new tax line, no `TaxPolicy` change.

### 4.3 `ServiceAccess` (new module `src/services_access/mod.rs`)

The care system's capacity+radius assignment loop (src/game_mechanics/care/mod.rs:158), generalized to the engine services that carry real capacity: PrimarySchool (ages 6–12), SecondarySchool (12–18), Clinic, Hospital. Each tick, demanders are assigned nearest-first within radius up to capacity, in citizen-id order (deterministic, the care pattern). The assigned counts are written back onto the engine `ServiceBuilding.current_users` **pub field** (the same existing-pub-field pattern the care gate uses on `Building.occupants`; no engine code change), making `utilisation()` real for the first time.

**The care tie-in (the differentiator):** a school-age child is covered **only** when it holds both a school *seat* and an after-school slot — `DependentKind::SchoolChild` coverage becomes `seat AND AfterSchool`, implemented by intersecting the seat assignment into `CareCoverage.covered` before the gate runs. A full school therefore gates parents exactly like a missing kindergarten: the city's labor supply is hostage to its school capacity, not just its care buildings. Healthcare does **not** gate (no double-jeopardy on the same worker): clinic/hospital shortfall lowers land value (`W_HEALTH`), feeds the existing `Healthcare gap` notification with real numbers, and counts in the wellbeing modifier (§4.7). The engine education pipeline (`process_education` booleans) is left untouched — education *quality* stays engine-side; *access* is game-side.

### 4.4 `TrafficModel` (new module `src/traffic/mod.rs`) — v1 flow-based, honestly scoped

**Agent-based traffic is out of scope for v1** (the engine's `AgentManager` walks straight lines, and per-citizen routing on every tick × full replay would dominate load times). V1 is static all-or-nothing assignment, the SimCity 4 model, fully deterministic:

- **Graph:** built from `CivitasGame::roads` each time the network changes (segment count is the dirty key). Edge weight = `length_m / speed(class)` with `speed`: Avenue 50, Street 30, Alley 15 (km/h). Adjacency over `RoadNode` ids in id order.
- **Endpoints:** every instance, care building, and service maps to its nearest segment (`RoadNetwork::nearest_segment` exists) and then to that segment's nearer node. Trips whose home or work lies > 200 m from any road, or when the network is empty, are counted **off-grid** (reported, not routed — honesty over silent drops).
- **OD demand:** one trip per employed citizen, home anchor (§4.1) → workplace position (instance lot center via the registry's `engine_building` link, or the raw engine building position for synthetic `add_jobs` buildings). Freight: one trip per Active commercial instance from its supplier node (§4.5). Trips aggregate into (origin node, destination node) counts.
- **Assignment:** one Dijkstra per distinct origin node (origins in node-id order, neighbor relaxation in edge-insertion order, ties to lower node id), volumes accumulated per segment along each shortest path. No equilibrium iteration in v1 (MSA is v2).
- **Load:** `load_ratio = volume / cap(class)` with `TRIPS_CAP`: Avenue 600, Street 300, Alley 80 per tick. Outputs: per-segment `SegmentLoad`, city `avg_load` / `worst_load` / `offgrid_ratio`.
- **Feedback (mild, explicit):** congestion enters land value (`W_CONG`, §4.2, one-tick lag) and throttles business output: the business-tax base uses `jobs × (1 − 0.15 × worst_path_load)` for instances whose commute path's max load exceeds 0.8. No citizen satisfaction coupling in v1 (one feedback loop at a time).
- **Cost budget:** Dijkstra is O(origins × E log N); current cities have < 500 nodes — the design budget is **≤ 5 ms per tick at 1,000 instances**, asserted by a perf smoke test in the plan, because every tick of replay pays it on load (§4.8 risk).

The engine's unwired `vox_sim::traffic::TrafficNetwork` (Greenshields densities) is not used in v1; it is the natural v2 home for time-dynamic flow once assignment exists.

### 4.5 `GoodsLocal` (extends `src/goods`, new submodule `src/goods/local.rs`)

The aggregate chain (`compute_flow`, budget lines) stays exactly as is — it remains the city totals and the trade money. New: a per-instance distribution layer over it. Each Active commercial instance gets an inventory ratio:

```
proximity = 1 / (1 + dist_km / SUPPLY_HALF_KM)        // road-graph distance to nearest
                                                       // Active industry instance; 0 if none
inventory = clamp(IMPORT_FLOOR + (1 − IMPORT_FLOOR) × max(goods.self_sufficiency, proximity), 0, 1)
```

with `SUPPLY_HALF_KM = 2.0`, `IMPORT_FLOOR = 0.35` (imports always reach the shelf, at the premium the budget already charges). A shop beside a factory is well-stocked even in an import-heavy city; a remote shop leans on the city's aggregate self-sufficiency. **Effect:** `sector_supply()` counts a commercial instance's jobs as `effective_jobs = round(jobs_cap × (0.6 + 0.4 × inventory))` — under-supplied shops employ fewer people, which the demand bars, income/business tax, and the care economy all feel through the existing call chain, with zero signature changes. Each commercial instance's restock trip feeds the traffic OD (§4.4). Farms/industry get no inventory in v1 (their input shortfalls are already priced city-wide); per-good per-instance ledgers are v2.

### 4.6 Business tenant programs (extends `src/instance`)

At develop time the registry realizes a **tenant mix** for job-bearing instances by partitioning the already-frozen `jobs_cap` through the existing `BuildingProgram` machinery (`occupy`, src/lot/occupancy.rs) for the lot's zone — a CommercialRegional instance becomes N shops + M restaurants summing exactly to `jobs_cap`; an Office instance becomes office tenants; partition seeded by `lot.seed ^ 0xA551_0004`. Sector totals are unchanged by construction (the mix is a decomposition, not a new number), so nothing in finance/demand moves — the mix exists for the inspector, for v2 tenant churn, and to fix one real bug: **MixedUse instances now contribute both `households_cap` to housing and `jobs_cap` to commercial** in the registry-backed `sector_supply()` (today the homes are dropped). Tenant-level open/close lifecycle is v2; v1 tenants throttle uniformly with the instance's inventory.

### 4.7 Zoning growth feedback (modifies `develop_parcels` + demand read)

Two changes, both deterministic:

1. **Ordering:** the selection pass sorts candidate vacant lots by `(desirability desc, plan_idx, lot_idx)` where `desirability = lv(lot)` from §4.2 — the lot beside the school develops before the fringe. A lot with `lv < DEV_DESIRABILITY_MIN = 0.20` is skipped entirely (stalled) and counted for the info line; servicing the area un-stalls it. Insertion-order determinism is preserved by the explicit tie-break.
2. **Wellbeing modifier:** the residential bar the developer *and the player* see becomes `residential × (0.6 + 0.4 × wellbeing)` where `wellbeing = mean(seat_ratio, care_coverage_ratio, health_in_reach_ratio)` from last tick's reports — a city failing its children and elders attracts fewer residents, on top of the satisfaction-driven migration the engine already runs. Applied inside `CivitasGame::demand()` so every consumer (bars UI, develop gate) agrees; `compute_demand` itself is unchanged.

**Implemented baseline (2026-06-10) — abandoned stock suppresses development.** The develop gate no longer reads `CivitasGame::demand()`: it reads the private `development_demand()`, identical except its supply side counts **Abandoned** instances' frozen `households_cap`/`jobs_cap` as existing stock (Active + Abandoned via the shared `supply_where`/`demand_for_supply` helpers in `src/game/mod.rs`), so no new same-sector parcel develops while abandoned stock could absorb the demand — recovery (player-facing bar > 0.5 for `RECOVER_AFTER` ticks) is the only path that reabsorbs it before sprawl. `sector_supply()` and `assessed_value()` remain Active-only (taxes/services come only from living buildings), UnderConstruction supplies nothing to either read (the construction lag deliberately keeps pulling development), and the loop is proven by `game::tests::abandoned_stock_suppresses_development_until_recovery`. M5 builds on this: the wellbeing modifier and desirability ordering must be applied in the shared `demand_for_supply`/selection core so the player bar and the gate bar keep agreeing on everything *except* the abandoned-stock term, which stays gate-only by design.

### 4.8 Determinism, save/replay, and the engine boundary

- **Replay, not snapshot.** `SAVE_VERSION` stays 5; no new fields in `GameSave`; no new `AuthoredAction` variants (every deep-sim input is already authored: zoning, roads, care, services, jobs, households). All §4.1–4.7 state is recomputed identically by `CivitasGame::replay` because every rule is a pure function of replayed state with fixed iteration order: registries in creation order, services and care buildings in id order, graph nodes in id order, citizens in id order, lots in `(plan, lot)` order. **Forbidden anywhere in the pass:** RNG beyond `seeded_range` with lot seeds, wall-clock, HashMap/HashSet iteration without a sort, parallel reduction.
- **Replay cost is a first-class constraint.** Load time = ticks × per-tick cost; the pass budget is ≤ 10 ms total per tick at 1,000 instances (traffic ≤ 5 ms of it), measured by a perf test. Cross-machine float identity is *not* promised (same as the existing sim — replay is bit-exact on the same binary/machine, which is what the save system requires).
- **Engine/game boundary.** Zero edits to any `~/src/ochroma` crate. Engine touch points are reads/writes of **existing pub fields only**: `Building.{id, position, capacity, occupants, operational}`, `ServiceBuilding.{service_type, position, capacity, current_users, coverage_radius, operational}`, `Citizen.{id, age, residence, employment, lifecycle}` — the exact pattern the care gate and the instances design already use. The engine never learns what land value, a school seat, an inventory, or a commute is. `vox_sim`'s unwired `LandValueGrid` and `TrafficNetwork` stay unwired (documented above why).

### 4.9 Observable surface (overlays, inspector, info window)

All rendering is HUD-layer over the existing pixel buffer via `world_to_screen` — no renderer/scene changes, works identically over the wgpu path and screenshots.

- **`v` key — land-value overlay:** for each lot, a filled translucent quad (new `hud::fill_world_quad`) tinted by `lv` on a green→red ramp, plus the legend line. Toggles; mutually exclusive with `t`.
- **`t` key — traffic overlay:** per road segment, a polyline (reusing the `draw_zone_draft` line primitive, new `hud::draw_road_load`) colored by `load_ratio` (green < 0.5, amber < 0.8, red ≥ 0.8), thickness from `RoadClass::width()`.
- **Info window (`i`)** gains exactly these lines (formats are the contract):
  `Land value        avg {:.2} · top {:.2}` · `School seats      {} / {} enrolled` · `Health capacity   {} cap · {} in reach` · `Traffic           load {:.0}% avg · worst {:.0}% · off-grid {:.0}%` · `Goods local       food {:.0}% · goods {:.0}% · retail {:.0}%` · the existing `Buildings` line extended to `Vacant lots       {} ({} stalled: low desirability)`.
- **Inspect window** (instances design §4.5) gains: `Land value   {:.2}` · `Tenants      {mix}   (jobs {})` · `Inventory    {:.0}% · supplier {:.1} km` (commercial only) · `Access       school {:.1} km · health {:.1} km · care {OK|GAP}` · `Commute      {} workers · avg {:.1} km` (residential only).

---

## 5. Data Models

All game-internal; pub fields per the codebase convention, mutation confined to the owning registry's methods.

```rust
// src/household/mod.rs
/// Persistent identity for one residence's occupants across ticks.
#[derive(Debug, Clone, PartialEq)]
pub struct HouseholdRecord {
    pub residence: u32,          // engine building id, or synthetic 100_000+ seed id
    pub formed_tick: u64,
    pub dissolved_tick: Option<u64>,
    pub anchor: [f32; 2],        // real lot center via instance link, else [id*10, 0]
    pub workers: u32,
    pub dependents: u32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HouseholdRegistry {
    records: Vec<HouseholdRecord>,            // creation order — stable
    by_residence: Vec<(u32, u32)>,            // sorted (residence, index) — no HashMap iteration
}

// src/land_value/mod.rs
/// Per-lot land value for one tick, parallel to placed_lots' (plan, lot) indexing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LandValueField {
    values: Vec<Vec<f32>>,       // values[plan_idx][lot_idx], 0.05..=1.0
    pub avg: f32,
    pub top: f32,
}

// src/services_access/mod.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind { PrimarySchool, SecondarySchool, Clinic, Hospital }

/// One tick's capacity-aware assignment for the capacity-bearing services.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServiceAccessReport {
    pub school_seats: u32,       // total capacity, both school kinds
    pub enrolled: u32,           // children holding a seat
    pub unplaced_children: u32,  // school-age, in no seat — these gate parents
    pub health_capacity: u32,
    pub health_in_reach: u32,    // residents assigned within radius up to capacity
    seated: Vec<u32>,            // citizen ids holding seats, sorted — feeds CareCoverage
}

// src/traffic/mod.rs
/// Load on one road segment after assignment (parallel to roads.segments()).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SegmentLoad {
    pub volume: f32,             // trips assigned this tick
    pub load_ratio: f32,         // volume / TRIPS_CAP[class]
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrafficModel {
    loads: Vec<SegmentLoad>,
    pub avg_load: f32,
    pub worst_load: f32,
    pub offgrid_ratio: f32,      // trips that couldn't be routed / total trips
    graph_key: usize,            // roads.segment_count() when the graph was built
}

// src/goods/local.rs
/// Per-commercial-instance stock level, parallel to BuildingRegistry indices.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GoodsLocal {
    inventory: Vec<f32>,         // by InstanceId; 0.0 for non-commercial instances
    supplier_km: Vec<f32>,       // road-graph distance to nearest Active industry; f32::MAX if none
}

// src/instance (extension) — realized at develop time, frozen like the caps
/// Tenant decomposition of a job-bearing instance; jobs sum to jobs_cap exactly.
pub struct TenantMix { pub tenants: Vec<crate::lot::occupancy::Business> }
```

Constants (each in its module, test-asserted by name): `BASE = 0.35`, `W_PARK = 0.15`, `W_SCHOOL = 0.10`, `W_HEALTH = 0.10`, `W_SAFETY = 0.05`, `W_CARE = 0.10`, `W_IND = 0.15`, `W_CONG = 0.10`, `LV_MIN = 0.05`; `LAND_FACTOR = |lv| 0.9 + 0.2 * lv`; `SPEED_KMH = [50.0, 30.0, 15.0]`, `TRIPS_CAP = [600.0, 300.0, 80.0]` (Avenue/Street/Alley), `OFFGRID_MAX_M = 200.0`, `OUTPUT_THROTTLE = 0.15`, `CONGESTED = 0.8`; `SUPPLY_HALF_KM = 2.0`, `IMPORT_FLOOR = 0.35`, `JOBS_FLOOR = 0.6`; `DEV_DESIRABILITY_MIN = 0.20`, `WELLBEING_FLOOR = 0.6`.

---

## 6. API

```rust
// All single-threaded, called only from CivitasGame::tick (step 1d) or read paths.

// src/household/mod.rs
impl HouseholdRegistry {
    /// Reconcile against the current citizens + instance registry. Creates new
    /// records, refreshes anchors/counts, marks emptied residences dissolved.
    /// Deterministic: residences visited in sorted id order.
    pub fn reconcile(
        &mut self,
        citizens: &[vox_sim::citizen::Citizen],
        instances: &crate::instance::BuildingRegistry,
        lots: &[crate::lot::layout::LotPlan],
        now_tick: u64,
    );
    /// Anchor for a residence id — lot center if instance-linked, else [id*10, 0].
    pub fn anchor(&self, residence: u32) -> [f32; 2];
    pub fn active_count(&self) -> usize;
}

// src/game_mechanics/care/demographics.rs — additive; default preserves behavior:
impl Demographics {
    /// As build(), with a caller-supplied residence -> position resolver.
    pub fn build_with_positions(
        citizens: &[Citizen],
        pos_of: impl Fn(u32) -> [f32; 2],
    ) -> Self;
    // build() delegates: build_with_positions(citizens, |r| [r as f32 * 10.0, 0.0])
}

// src/land_value/mod.rs
impl LandValueField {
    /// Recompute every lot's value (§4.2). `traffic` and `access` are LAST
    /// tick's outputs (one-tick lag). Pure; fixed iteration order.
    pub fn recompute(&mut self, game: &LandValueInputs<'_>);   // borrows lots, instances,
                                                               // care, sim.services, roads,
                                                               // last traffic + access
    pub fn lot(&self, plan_idx: usize, lot_idx: usize) -> f32; // LV_MIN..=1.0
}

// src/services_access/mod.rs
/// Assign seats/capacity nearest-first in citizen-id order; writes
/// `current_users` onto engine ServiceBuildings (existing pub field).
pub fn assign(
    sim: &mut vox_sim::city_sim::CitySim,
    households: &HouseholdRegistry,
) -> ServiceAccessReport;
impl ServiceAccessReport {
    pub fn has_seat(&self, citizen_id: u32) -> bool;   // binary-search over `seated`
}

// src/traffic/mod.rs
impl TrafficModel {
    /// Rebuild the graph if roads changed, derive OD from employed citizens +
    /// commercial restock trips, run all-or-nothing assignment (§4.4).
    pub fn assign(&mut self, inputs: &TrafficInputs<'_>);      // roads, citizens,
                                                               // instances, lots,
                                                               // households, goods_local
    pub fn segment_load(&self, segment_idx: usize) -> SegmentLoad;
    /// Road-graph distance in km between two world points (also used by goods).
    pub fn route_km(&self, a: [f32; 2], b: [f32; 2]) -> Option<f32>;
}

// src/goods/local.rs
impl GoodsLocal {
    /// Per-commercial-instance inventory from the aggregate flow + road
    /// proximity to the nearest Active industry instance (§4.5).
    pub fn distribute(
        &mut self,
        flow: &crate::goods::GoodsFlow,
        instances: &crate::instance::BuildingRegistry,
        lots: &[crate::lot::layout::LotPlan],
        traffic: &TrafficModel,
    );
    pub fn inventory(&self, id: crate::instance::InstanceId) -> f32;
    pub fn supplier_km(&self, id: crate::instance::InstanceId) -> Option<f32>;
}

// src/instance — additive:
impl BuildingRegistry {
    /// The frozen tenant decomposition (realized at develop; jobs sum == jobs_cap).
    pub fn tenants(&self, id: InstanceId) -> &[crate::lot::occupancy::Business];
}

// src/game/mod.rs — CHANGED INTERNALS, same public signatures:
//   demand()         — residential bar × (WELLBEING_FLOOR + (1-WELLBEING_FLOOR)·wellbeing)
//   sector_supply()  — commercial jobs become inventory-throttled effective jobs;
//                      MixedUse adds BOTH homes and jobs
//   develop_parcels()— candidates sorted by (lv desc, plan, lot); lv < DEV_DESIRABILITY_MIN stalls
//   finances()       — business tax base throttled on congested commute paths (§4.4)
// NEW CivitasGame fields: households, land_value, access, traffic, goods_local
// NEW CivitasGame reads:  stalled_parcels() -> usize
```

Panics: none. Empty road network, zero services, zero instances are all valid (loads/ratios report 0, off-grid 100%, inventory at floor); guarded by tests, never by `unwrap` on player state.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `households.reconcile()` | `CivitasGame::tick` step 1d.1 | `src/game/mod.rs` (after `instances.tick`, before `Demographics::build`) | anchors must exist before care derivation |
| `Demographics::build_with_positions(…, registry resolver)` | `CivitasGame::tick` step 2 | `src/game/mod.rs:640` | replaces the `Demographics::build` call; legacy fallback inside the resolver |
| `land_value.recompute()` | step 1d.2 | `src/game/mod.rs` | reads LAST tick's `traffic`/`access` (one-tick lag) |
| `access = services_access::assign()` | step 1d.3 | `src/game/mod.rs` | writes `current_users` on engine `ServiceBuilding`s |
| seat intersection into `CareCoverage` | step 2b, after `assign_slots_with_barrier` | `src/game/mod.rs:661` | a SchoolChild stays covered only if `access.has_seat(id)` |
| `traffic.assign()` | step 1d.4 | `src/game/mod.rs` | rebuilds graph only when `roads.segment_count()` changed |
| `goods_local.distribute()` | step 1d.5 | `src/game/mod.rs` | uses `traffic.route_km` for supplier distance |
| inventory-throttled `sector_supply()` / MixedUse homes | existing callers unchanged | `src/game/mod.rs:768` | registry iteration gains the `effective_jobs` term |
| desirability ordering + stall gate | `develop_parcels` selection pass | `src/game/mod.rs:860` | sort by `(lv desc, pi, li)`; skip `lv < DEV_DESIRABILITY_MIN` |
| wellbeing modifier on residential bar | `CivitasGame::demand()` | `src/game/mod.rs:815` | single place; bars UI + develop gate agree |
| tenant realization | `BuildingRegistry::develop` | `src/instance/mod.rs` | partition `jobs_cap` via `occupy`, seed `lot.seed ^ 0xA551_0004` |
| `v` / `t` overlay toggles | `GameView::key` Character arm | `src/bin/play.rs:503` | mutually exclusive flags on `GameView` |
| `hud::fill_world_quad` / `hud::draw_road_load` | `GameView::render`, after city pixels, before bottom bar | `src/bin/play.rs:400` + `src/render_gpu/hud.rs` | pure pixel-buffer drawing via `world_to_screen` |
| six new info lines | `GameView::info_lines` | `src/bin/play.rs:322` | exact formats in §4.9 |
| five new inspect lines | `GameView::inspect_lines` (from instances design) | `src/bin/play.rs` | reads registry + `land_value` + `access` + `goods_local` + `traffic` |
| `School seats full — N children unplaced` alert | tick step 6 notifications | `src/game_mechanics/notifications/mod.rs` + `src/game/mod.rs:740` | severity Critical when `unplaced > 0` |
| replay equality of all new state | new test beside the save tests | `src/game/mod.rs` tests | field-by-field on households/land_value/access/traffic/goods_local |

---

## 8. Open Questions

- [ ] **Pollution as its own field:** v1 folds industrial nuisance into the land-value term (`W_IND`). Is a separate pollution/noise field (with its own overlay) wanted before v2 traffic? Leaning no — one new field per milestone keeps the inspector legible.
- [ ] **Seat gating severity:** a seatless school child gates *all* household workers (the existing gate rule). Should school seats gate only one guardian (a softer rule than care)? V1 keeps the uniform rule for mechanical clarity; revisit after the M2 playtest, as a constant-free rule change.
- [ ] **Congestion → output coupling strength:** `OUTPUT_THROTTLE = 0.15` at `load > 0.8` is a first cut. The plan must keep it a named constant and may retune from playtest; tests assert against the constant.
- [ ] **V2 agents:** reuse engine `AgentManager`/`crowd` for visible commuters sampled from the v1 flow field (presentation-only, sim stays flow-based), or full agent routing? Decide when v2 is planned; v1's per-segment loads are the interface either way.
- [ ] **Land value at scale:** per-lot recompute is O(lots × amenities) per tick. If cities reach ~10k lots, switch to a dirty-region scheme or a coarse grid sampled per lot — decide when a profile shows it, not before.

---

## 9. Out of Scope

- Agent-based traffic, route choice, transit lines, parking, or rendered vehicles — v1 is static all-or-nothing flow assignment, stated plainly in the UI as load percentages.
- Tenant churn (individual businesses opening/closing), business wealth/levels, and per-good per-instance ledgers — tenants exist and throttle uniformly in v1.
- Intra-city household moves and life-event simulation (marriage, births beyond engine aging) — household identity and anchoring only; turnover remains engine migration.
- Any engine-crate edit: `vox_sim`'s unwired `LandValueGrid`/`TrafficNetwork` stay unwired; `match_housing`'s nearest-to-origin fill order is accepted, not fixed.
- New tax instruments (land-value tax line, per-district rates) — land value flows through the existing assessed-value × property-rate line only.
- Save-schema changes: no `SAVE_VERSION` bump, no new `AuthoredAction` variants, no state snapshotting — replay determinism remains the persistence mechanism.
- Cross-machine float-identical replay (same constraint as the existing sim).
- Education quality/outcomes changes (`process_education` untouched); healthcare gating workers.
- Districts/policies per area, and level-up/densification (tracked in the instances design's open questions).

---

## 10. Related Plans / Designs

- Depends on: `[Living Building Instances Design](./2026-06-10-living-building-instances-design.md)` — **hard prerequisite, in implementation now**: this design binds to `BuildingRegistry` (`InstanceId`, `engine_building`, `for_lot`, frozen `households_cap`/`jobs_cap`, `value()` with the land factor inserted multiplicatively and neutral at lv = 0.5), the registry-backed `sector_supply()`/`assessed_value()`, and the Inspect window it adds to `play.rs`.
- Binds to existing types: `CivitasGame`/`develop_parcels`/`demand()` (src/game/mod.rs), `Demand`/`compute_demand` (src/demand/mod.rs), `SectorJobs`/`settle` (src/finance/mod.rs), `GoodsFlow`/`compute_flow` (src/goods/mod.rs), `CityBudget` (src/budget/mod.rs), `RoadNetwork`/`RoadClass`/`nearest_segment` (src/road/mod.rs), `CareSystem`/`CareCoverage`/`Demographics` (src/game_mechanics/care/), `BuildingProgram`/`occupy`/`Business` (src/lot/occupancy.rs), engine `vox_sim::{buildings, services, citizen, city_sim}` pub fields.
- Required before: the **Deep Simulation implementation plan** (one plan per milestone M1–M5, each turning its §3 rows into TDD tasks; M1 → M2 → M3 → M4 → M5, M4 needs M3's `route_km`, M5 needs M1 + M2).
- Related: `[SOTA City Block Phase 1 Plan](../plans/2026-06-10-sota-city-block-phase1.md)` (sequencing source), `[Virtualized Splat Rendering Design](./2026-06-10-virtualized-splat-rendering-design.md)` (overlays are HUD-layer precisely so they are independent of the residency/LOD scheme).
