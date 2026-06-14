# Deep Simulation M2 — School Seats & Healthcare Capacity Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use **superpowers:subagent-driven-development** (recommended) or **superpowers:executing-plans** to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the care-first flagship milestone: schools and healthcare stop being radius booleans — a new `ServiceAccess` pass assigns school seats and clinic/hospital capacity nearest-first with depletion (the `CareSystem` loop, generalized), writes the truth onto the engine's `ServiceBuilding.current_users`, and a full school gates parents off the workforce **exactly** like missing childcare (SchoolChild coverage = after-school slot AND a seat) — with the cause diagnosable on every chrome surface (feed, Statistics, info window, inspector).
**Done When:** All three, in `~/Ochroma/projects/urban_horizon`:
1. `cargo test --lib game::tests::an_overfilled_school_gates_parents_until_seats_open -- --nocapture` prints the tick timeline with real numbers:
   `[school-gate] t=3 seats 500/500 · unplaced 250 · gated 50` → `[school-gate] +PrimarySchool #2` → `[school-gate] t=5 seats 750/1000 · unplaced 0 · gated 0 · parents returned: true`, and `cargo test --lib services_access:: -- --nocapture` prints `[seats] enrolled 5 / 5 seats, unplaced 7, lowest ids seated: true` and `[access] current_users school 5/5 · clinic 3/3 (utilisation 1.00)` (depletion proven on hand-shrunk capacities, not vacuous zeros).
2. `cargo run --release --bin play -- --shot-school /tmp/school` prints, in order: the gating city's feed line `CRIT School seats full — 250 children unplaced — parents stay home until seats open.`, the Statistics line `School seats      500 / 500   unplaced 250`, then after the second school the live info line `School seats      750 / 1000 enrolled`; then on Demo City the service-inspector block containing `Seats        10 / 500`, the house line `Access       school 0.2 km · health 0.1 km · care GAP`, writes `/tmp/school/school_inspect.png` (the Demo school's window open over the city), exits 0 (exits 1 with a printed reason on any failed inequality).
3. Human at the keyboard: `cargo run --release --bin play` → New Game → **Demo City** → Start Game → Space ×10 → press **i**: `School seats      10 / 500 enrolled` and `Health capacity   1000 cap · <n> in reach` with `n > 0`; **Tab** to **Inspect**, click the Primary School: a floating window `Service — Primary School` with `Seats        10 / 500`; click a house: a new line `Access       school 0.N km · health 0.N km · care OK|GAP`; open **Stats**: the same school/health lines the info window shows.

**Architecture:** A new tick step **1d.3** runs after the land-value pass (1d.2) and before demographics (2): `services_access::assign` walks school-age children (and all housed residents for health) in citizen-id order, anchored through the M1 `HouseholdRegistry`, assigns each to the nearest in-reach service building with a free slot (squared-Euclidean nearest, ties to the lower index — the exact `CareSystem::assign_slots_with_coverage` pattern), and writes per-building counts onto the **existing pub field** `ServiceBuilding.current_users` — `utilisation()` becomes real with zero engine edits. At step 2b, after the care pass, `CareCoverage::require_seat` removes cover from every SchoolChild holding no seat, so the **unchanged** gate (`worker_is_free_to_work` + the revocation loop) fires for seat shortage exactly as it fires for a missing kindergarten. Reach reuses the tick's existing closure seam: Euclidean circle + water barrier everywhere, upgraded to the geodesic field for schools on real maps (a second reach field mirroring `childcare_field`, rebuilt on school plop). Healthcare counts and informs (land value, info lines, notifications) but never gates — no double jeopardy. Everything is recomputed by action-log replay: no `SAVE_VERSION` bump, no new `AuthoredAction`s, no serialized capacity state.
**Design Document:** `docs/superpowers/specs/2026-06-10-deep-simulation-design.md` (M2 = §2-M2, §4.3, §4.9 M2 rows, §5, §6; M1 results in `docs/superpowers/plans/2026-06-10-deep-sim-m1.md`)
**Tech Stack:** Rust (workspace toolchain as-is), game repo `~/Ochroma/projects/urban_horizon`. Engine crates (`~/src/ochroma`) **read-only**.
**Build:** `cd ~/Ochroma/projects/urban_horizon && cargo build && cargo test --lib` (the play binary needs a desktop display; `--shot-school` renders offscreen). No spectra features needed.

---

## IMPORTANT NOTES

- **Repos / commits (house convention — changed from M1):** game `~/Ochroma/projects/urban_horizon` (git master). **A checkpoint lands first:** before any M2 edit, commit the repo's current uncommitted baseline (M1 + instances + chrome work) as `checkpoint: pre-deep-sim-M2 baseline` — then every task ends with its own commit. Every commit message ends with the footer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Engine repo `~/src/ochroma` is **never edited and never committed** by this plan.
- **OWNERSHIP GATE (read before starting):** the World-Scale NYC M1 plan (`docs/superpowers/plans/2026-06-10-world-scale-nyc-m1.md`) is executing in cook/play space. Honest file-disjointness check: NYC M1's Group P **modifies `src/game/mod.rs` (`place_real_city`) and `src/bin/play.rs` (StartGame hook, inspector lines, `--nyc-shot`)** — the same two files every M2 task wires through. Disjointness FAILS; therefore **this entire plan executes only after NYC M1 completes** (its modules `src/realcity`, `src/bin/nyc_*` never collide, but the wiring files do). The M2-only modules (`src/services_access`, `care/`, `household/`, `coverage_field.rs`, `land_value/`, `ui/panels.rs`, `notifications/`, `scenario/`) have no other owner. Engine workstreams may transiently break `vox_render`/`vox_sim` builds via the path deps — retry after they land; **never edit `~/src/ochroma` to "fix" the build**, and never run the full `vox_render --lib` suite (documented `resident_gi_seam` flake).
- **CAPACITY SOURCE (the decision, from code):** seats/capacity come from the **existing engine pub field `ServiceBuilding.capacity`**, set once at plop by `ServiceManager::place_service` (vox_sim/src/services.rs:78–91): PrimarySchool **500 seats / 1000 m**, SecondarySchool **1000 / 2000 m**, Clinic **200 / 1000 m**, Hospital **1000 / 3000 m**. It is already frozen-at-placement by construction (services don't grow), so nothing new to freeze — the `jobs_cap` analogy is satisfied by the engine's own table. **NOT** `CareKind` params (those are care buildings: Kindergarten/AfterSchool/…), **NOT** registry instance caps (schools are plopped services, never zoned instances — M1's park note: zoned Service lots become engine `Building`s, not `ServiceBuilding`s). `CityServiceKind` (game) carries only `build_cost`/`service_type()`.
- **THE PARENT-GATING MECHANISM (found, mirrored, untouched):** the infant-childcare gate is `CareSystem::worker_is_free_to_work` (care/mod.rs:241–254 — a worker is employable only when **every** dependent in `demo.household_of(worker)` has `coverage.is_covered(dep)`) plus the tick revocation loop (game/mod.rs:840–878 — steps 3/4a/4b: collect gated ids, free the engine building slot, null `employment`/`workplace`, crater `needs.employment`). **M2 changes neither.** The mirror is upstream: `CareCoverage::require_seat` removes seatless SchoolChildren from the private `covered` set at step 2b, so the identical gate fires for seats. `coverage.served`/`unmet()` keep **slot semantics** (the after-school gap) — the seat gap reports separately through `ServiceAccessReport.unplaced_children`, so the two causes can never masquerade as each other (diagnosability requirement).
- **Engine boundary:** zero `~/src/ochroma` edits, **verified not needed**: `ServiceBuilding { pub capacity, pub current_users, pub coverage_radius_m, pub operational, pub position, .. }` (vox_sim/src/services.rs:22–34) and `utilisation()` (:57) are pub; `current_users` is written **nowhere** in either repo today (grepped 2026-06-11) — the game becomes its single writer, the same existing-pub-field pattern the gate uses on `Building.occupants`. Engine `process_education` (booleans via `has_service`, city_sim.rs:357–365) stays untouched: education *quality* is engine-side, *access* is game-side.
- **Verified game signatures (code-checked 2026-06-11; code wins over the design doc):**
  - `CivitasGame::tick` (game/mod.rs:737): 1c instances.tick (:751) → 1d.1 `households.reconcile` (:762) → 1d.2 `land_value.recompute` + `apply_land_values` (:771–783) → step 2 `Demographics::build_with_positions` (:790–796) → 2b `assign_care_staff` (:808) + `assign_slots_with_coverage` (:827–838) → step 3 gating (:840) → notifications assembly (:912–917). **Step 1d.3 slots between :783 and :785; the seat intersection slots between :838 and :840.**
  - `CareSystem::assign_slots_with_coverage(&mut self, demo, in_reach: impl Fn(usize, [f32;2], [f32;2], f32) -> bool) -> CareCoverage` (care/mod.rs:180) — the closure shape M2's assign reuses. The tick's live closure (:829–838): geodesic field arm for `Infant` when `field.contains(dp)`, else `d² ≤ r² && !terrain.line_crosses_water(bp, dp)`.
  - `CareCoverage` (care/mod.rs:58–67): `covered: HashSet<u32>` is **private** — `require_seat` must be an `impl CareCoverage` method in care/mod.rs. `HashSet::remove` is lookup, not iteration — determinism-safe.
  - `HouseholdRegistry::anchor(&self, residence: u32) -> [f32; 2]` (household/mod.rs) — lot center for instance-linked residences, `[id*10, 0]` otherwise. Reconciled at 1d.1, so 1d.3 reads this tick's anchors.
  - `CoverageFieldService` (coverage_field.rs): `rebuild_childcare(&mut self, terrain, extent, facilities: &[[f32;2]]) -> ChildcareField` (:47) is **already seed-generic** (water mask + arbitrary seed points; only the name is childcare-specific); `rebuild_from_mask` is private. `ChildcareField::{contains, distance_m, reachable}` answer from the CPU integer read-back — bit-deterministic. The dirty-rebuild pattern to mirror: `childcare_field`/`childcare_field_dirty` fields (game/mod.rs:237–240), `rebuild_childcare_field_if_dirty` (:713–734), dirty set in `place_care` (:328–333) and `set_coverage_field` (:292–301). **Caveat carried over unchanged:** the field service is injected by `play` only; saves/replays/headless/flat worlds run the legacy circle+barrier path — the school field inherits the exact same convention.
  - `Citizen { id, age, residence: Option<u32>, lifecycle, employment, .. }`; `lifecycle_for_age` (vox_sim/src/citizen.rs:82): Child < 6, Student 6–18, Worker 18–65, Retired ≥ 65. `DependentKind::SchoolChild` == Student == ages [6, 18). **Age bands for seats (pin, matches engine education):** PrimarySchool serves [6, 12), SecondarySchool [12, 18) — exact cover of SchoolChild, no gap, no overlap.
  - `CitySim::YEARS_PER_TICK = 0.02` and `seed_population(24)` ages 19–58 **all working-age** (city_sim.rs) — scenarios start with **zero** students/infants/elders from the engine; school-age children exist only where a test/scenario seeds them (ages 6–17.99) — children effectively never age across test horizons (50 ticks/year).
  - `AfterSchool` params (care/service.rs:84–90): capacity 120, radius 1800 m, staff 8. `assign_care_staff` fills buildings in id order from `labor_pool` = count of `Worker`-lifecycle citizens (game/mod.rs:801–808) — gated workers still count toward the pool.
  - Save/replay: `SAVE_VERSION = 5` (save.rs:27); `GameSave { version, map_id, actions, action_ticks, style_pack, policies, economy, ticks }` — **no sim state**; `PlaceService`/`PlaceCare`/`AddJobs`/`SeedHousehold`/`AddRoad` all replay through `apply_action` (game/mod.rs:1349–1365). `current_users` and the access report are **derivable**: replay re-runs step 1d.3 every replayed tick, recomputing them bit-for-bit. **No GameSave change, no new AuthoredAction** (everything M2 reads is already authored).
  - Chrome surfaces: `stats_lines` (ui/panels.rs:57 — its test asserts by `contains`, additive lines are safe); `info_lines` (play.rs:398, `Care sites` line at :410); `inspector` (play.rs:428 — currently **seven** lines; all close/swallow handlers read counts dynamically via `gv.inspector(id).map_or(0, |(_, l)| l.len())`, play.rs:1560–1577, so adding an 8th line and a second window kind is safe); window draw block (play.rs:894–935, `hud::draw_city_info_window` for the floating inspector, `hud::draw_submenu_panel` for Info/Stats); `inspect_click` (:470–479); `ViewState` lives in **src/command/mod.rs:112** (`pub inspected: Option<u32>` :119, init :158, Esc-clear :625) — `inspected_service` is added **there**, not in play.rs; notifications feed: `care_notifications(stats, coverage)` + `service_notifications(sim)` assembled and severity-sorted at game/mod.rs:912–917; `Severity::{Info, Warning, Critical}` with `label()` → `INFO|WARN|CRIT`.
- **Design-doc deviations (pinned here; the plan wins):** (1) §6's `assign(sim, households)` gains the `in_reach` closure parameter — the geodesic/barrier seam, mirroring `assign_slots_with_coverage`; (2) the M1 plan note "M2 adds `access` to `LandValueInputs`" is **dropped** — seat-exhaustion rides on `ServiceBuilding.current_users`, which `LandValueInputs.services` already carries, and because 1d.2 runs *before* 1d.3 those values are **last tick's** — the design's one-tick lag falls out of tick order with zero new plumbing; (3) §4.3's "capacity-weighted" health term is pinned to the binary form `capacity > 0 && current_users < capacity` (same rule as schools; an exactly-full building reads exhausted — boundary documented); (4) info-line formats use plain integers (no `1,200` thousands separators — matches every existing info line); (5) the design's §8 seat-severity open question is resolved as written: a seatless child gates **all** household workers (the uniform existing rule).
- **Determinism rules (design §4.8, unchanged):** no RNG, no wall clock, no HashMap/HashSet **iteration** (lookups fine); citizens in sorted-id order, service buildings in Vec order (engine id order), ties to the lower index. `ServiceAccessReport` derives `PartialEq` and is **never serialized**.
- `todo!()` / `unimplemented!()` / empty function bodies are **forbidden** — they fail the task.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `~/Ochroma/projects/urban_horizon/src/services_access/mod.rs` | `AccessKind`, `ServiceAccessReport`, `assign` (nearest-first, seat depletion, `current_users` writes), `has_seat`, in-module tests |
| Modify | `~/Ochroma/projects/urban_horizon/src/lib.rs` | `pub mod services_access;` (alphabetical, between `scenario` and `spatial_field`) |
| Modify | `~/Ochroma/projects/urban_horizon/src/game/mod.rs` | `access` + `school_field`(+dirty) fields, tick step 1d.3, seat intersection at 2b, school-dirty in `place_city_service`/`set_coverage_field`, `rebuild_school_field_if_dirty`, seat-notification call, timeline + replay tests |
| Modify | `~/Ochroma/projects/urban_horizon/src/coverage_field.rs` | `ReachField` rename-by-alias + seed-generic `rebuild_reach` (childcare delegates) |
| Modify | `~/Ochroma/projects/urban_horizon/src/game_mechanics/care/mod.rs` | `CareCoverage::require_seat` (seat∩slot intersection), unit test |
| Modify | `~/Ochroma/projects/urban_horizon/src/land_value/mod.rs` | school/health terms gain the capacity-exhaustion predicate, tests |
| Modify | `~/Ochroma/projects/urban_horizon/src/game_mechanics/notifications/mod.rs` | `seat_notifications` — `School seats full — N children unplaced` (Critical) |
| Modify | `~/Ochroma/projects/urban_horizon/src/ui/panels.rs` | Statistics `School seats` / `Health capacity` lines |
| Modify | `~/Ochroma/projects/urban_horizon/src/scenario/mod.rs` | Demo City school-age mix (printed justification), scenario verification test |
| Modify | `~/Ochroma/projects/urban_horizon/src/command/mod.rs` | `ViewState.inspected_service` field + Esc/one-window handling |
| Modify | `~/Ochroma/projects/urban_horizon/src/bin/play.rs` | info lines, house `Access` line, service-building inspector window, `--shot-school` harness |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Seats deplete nearest-first, deterministically | shrink a school's pub `capacity` to 5 with 12 in-reach children: `enrolled == 5`, `unplaced == 7`, and the seated ids are exactly the 5 lowest child ids; printed | `assert!(report.school_seats > 0)` |
| `current_users` is truthful | after one tick, each school's `current_users` equals its assigned count and `utilisation()` returns `1.00` on the full one; clinic `current_users == min(in-reach residents, capacity)`; printed per building | `assert!(b.operational)` |
| A full school gates parents like missing childcare | 150 households × 5 children (8 y/o), slots ample, 1 school: `gated == 50` (the seatless households' workers), strictly greater than the 2-school run (`gated == 0`); the revocation loop and `worker_is_free_to_work` are **untouched** (git diff proves it); tick timeline printed | `assert!(stats.gated_workers >= 0)` — vacuous |
| Geodesic reach severs school coverage across water | on a river map with the field injected, a school across the river covers nobody on the far bank (`enrolled == 0`) until a same-bank school plops; flat-world run bit-matches the barrier-only closure | `assert!(field.is_some())` |
| Land value feels seat exhaustion | a lot in radius of a school whose seats filled last tick loses exactly `W_SCHOOL` vs the unexhausted run; printed before/after | `assert!(lv > 0.0)` |
| The collapse is diagnosable | with unplaced > 0 the feed carries `CRIT School seats full — N children unplaced…`, Statistics carries `School seats  a / b   unplaced N`, and the slot-side `No after-school care` alert does **not** fire when slots are ample (causes never masquerade); printed | `assert!(!notifications.is_empty())` |
| Scenario shapes survive M2 | Founders' Vale: 0 students / 0 seats / 0 unplaced — the eldercare teaching gap is untouched (existing tests byte-identical); Demo: 10 students enrolled 10/500, eldercare gap intact; both triples printed as the recalibration justification | skipping the scenarios |
| Replay reproduces capacity state | 40 ticks with school + students + clubs → save/load → `assert_eq!` on the access report, per-building `current_users`, coverage maps; `save.version == 5`; counts printed | comparing only `tick_index` |

---

## Pre-flight: checkpoint commit (before Task 1)

- [ ] In `~/Ochroma/projects/urban_horizon`: `cargo test --lib 2>&1 | tail -3` green, then
```bash
git add -A && git commit -m "checkpoint: pre-deep-sim-M2 baseline (M1 land value + instances + chrome)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 1: `ServiceAccess` — capacity-aware nearest-first assignment, truthful `current_users`, wired as tick step 1d.3

**Files:**
- Create: `~/Ochroma/projects/urban_horizon/src/services_access/mod.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/lib.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs`

**Acceptance:** `cd ~/Ochroma/projects/urban_horizon && cargo test --lib services_access:: -- --nocapture` → `seats_deplete_nearest_first_in_id_order` prints `[seats] enrolled 5 / 5 seats, unplaced 7, lowest ids seated: true` and `current_users_become_real` prints `[access] current_users school 5/5 · clinic 3/3 (utilisation 1.00)`. Full `cargo test --lib` stays green.

**Wiring requirement:** `crate::services_access::assign(...)` is called from `CivitasGame::tick` as step **1d.3**, immediately after the 1d.2 `apply_land_values` block (game/mod.rs:783) and before step 2 (:785). `CivitasGame` gains `pub access: crate::services_access::ServiceAccessReport` (init `Default::default()` in `new_small`). `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing tests** in `src/services_access/mod.rs` `#[cfg(test)] mod tests` (driving a real `CivitasGame` — the care-test pattern):
  - `seats_deplete_nearest_first_in_id_order`: `CivitasGame::new_small()`; seed 12 one-child households at residences 200_000..200_011 (`seed_household_at(r, 34.0, &[8.0])` — anchors x = 2_000_000..2_000_110, a 110 m strip); `place_city_service(CityServiceKind::PrimarySchool, [2_000_055.0, 0.0])`; **shrink the seat pool through the pub field** `game.sim.services.buildings.iter_mut().find(|b| b.service_type == ServiceType::PrimarySchool).unwrap().capacity = 5;` then `game.tick()`. Assert `game.access.school_seats == 5`, `enrolled == 5`, `unplaced_children == 7`; collect the 12 child ids, assert `has_seat` is true for exactly the 5 lowest and false for the rest; print the acceptance line.
  - `current_users_become_real`: same city plus `place_city_service(CityServiceKind::Clinic, [2_000_055.0, 0.0])` with clinic `capacity = 3` (same pub-field shrink); tick; assert the school building's `current_users == 5` and `utilisation()` returns `1.0` (± 1e-6), the clinic's `current_users == 3` (12 workers + 12 children + engine pop are out of its 1000 m reach except the strip's 24 — assert the exact computed count == capacity since in-reach residents ≥ 3); print per building.
  - `out_of_reach_children_are_unplaced_not_assigned`: school at `[0.0, 0.0]` (engine block), children on the 2e6 strip → `enrolled == 0`, `unplaced == 12`, every `current_users == 0`; print.
- [ ] **Step 2: Run to verify failure** — `cargo test --lib services_access:: 2>&1 | tail -5` → FAIL: `error[E0433]`/`E0609` (module/field absent).
- [ ] **Step 3: Implement** — full logic, design §5 types (verified against code):

```rust
/// The engine services that carry real capacity (design §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind { PrimarySchool, SecondarySchool, Clinic, Hospital }

impl AccessKind {
    pub fn of(st: vox_sim::services::ServiceType) -> Option<AccessKind> { /* 4-arm match, else None */ }
    pub fn is_school(self) -> bool { matches!(self, Self::PrimarySchool | Self::SecondarySchool) }
}

/// One tick's capacity-aware assignment. Never serialized — replay recomputes it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServiceAccessReport {
    pub school_seats: u32,       // total capacity, operational schools (both kinds)
    pub enrolled: u32,           // children holding a seat
    pub unplaced_children: u32,  // school-age, in no seat — these gate parents
    pub health_capacity: u32,    // operational clinic+hospital capacity
    pub health_in_reach: u32,    // residents assigned within reach up to capacity
    seated: Vec<u32>,            // citizen ids holding seats, ascending — binary search
}

impl ServiceAccessReport {
    pub fn has_seat(&self, citizen_id: u32) -> bool { self.seated.binary_search(&citizen_id).is_ok() }
}

/// Assign seats (schools, ages [6,12) primary / [12,18) secondary) and health
/// capacity (clinic|hospital, every housed citizen) nearest-first with
/// depletion, demanders in ascending citizen-id order, candidates in service
/// Vec order, nearest by squared Euclidean with ties to the lower index — the
/// CareSystem::assign_slots_with_coverage pattern. `in_reach(svc_idx,
/// building_pos, demander_pos, radius_m)` owns reach semantics (the caller's
/// geodesic/barrier closure). Writes the per-building counts onto the engine's
/// pub `ServiceBuilding.current_users` (the four kinds only; others untouched).
pub fn assign(
    sim: &mut vox_sim::city_sim::CitySim,
    households: &crate::household::HouseholdRegistry,
    in_reach: impl Fn(usize, [f32; 2], [f32; 2], f32) -> bool,
) -> ServiceAccessReport
```

  Implementation pins: (a) snapshot candidates first — `Vec<(svc_idx, AccessKind, position, coverage_radius_m, capacity)>` from `sim.services.buildings` where `operational && AccessKind::of(..).is_some()`, with local `assigned: Vec<u32>` counts (no borrow fights); (b) **one** id-sorted pass over school-age housed citizens (`residence.is_some() && 6.0 <= age < 18.0`), each child's candidate set filtered by its band's kind, position = `households.anchor(residence)`; (c) a second id-sorted pass over **all** housed citizens for Clinic∪Hospital (one slot per resident across both kinds); (d) write-back `current_users = assigned[i]` for the snapshot's indices, totals into the report; `seated` is ascending by construction (pass order). Zero-capacity or out-of-reach demand reports honestly (`unplaced`, untouched `current_users == 0`). Register `pub mod services_access;` in lib.rs.
- [ ] **Step 4: Wire at the exact callsite** — game/mod.rs, after :783, before step 2:

```rust
// 1d.3 Capacity-aware service access: school seats + clinic/hospital capacity
//      assigned nearest-first with depletion, written onto the engine's
//      ServiceBuilding.current_users (its single writer). Runs AFTER land
//      value (1d.2 reads LAST tick's current_users — the one-tick lag) and
//      BEFORE demographics/care (the seat intersection at 2b reads it).
let terrain = &self.terrain;
self.access = crate::services_access::assign(
    &mut self.sim,
    &self.households,
    |_idx, bp, dp, r| {
        let (dx, dz) = (bp[0] - dp[0], bp[1] - dp[1]);
        dx * dx + dz * dz <= r * r && !terrain.line_crosses_water(bp, dp)
    },
);
```

  (Task 2 replaces this closure body with the geodesic-school version; landing the barrier form first keeps each task shippable.) Field doc on `pub access`: "Rebuilt every tick at 1d.3; never serialized — replay recomputes it."
- [ ] **Step 5: Run — verify non-trivial output.** All three tests PASS with the printed real numbers; full `cargo test --lib` green.
- [ ] **Step 6: Commit** — `git add src/services_access/mod.rs src/lib.rs src/game/mod.rs && git commit -m "feat(deep-sim-m2): ServiceAccess seat/capacity assignment wired as tick 1d.3 — current_users becomes real"` + footer.

---

## Task 2: School geodesic reach — generalize the coverage field, mirror the childcare dirty-rebuild

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/coverage_field.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs`

**Acceptance:** `cargo test --lib coverage_field::tests::school_reach_severs_across_water -- --nocapture` prints `[school-field] far bank enrolled 0 (severed), same-bank school enrolled <n> > 0` on a GPU machine (prints the documented `no adapter — skipping` line otherwise), and `cargo test --lib game::tests::school_access_flat_world_matches_barrier_closure -- --nocapture` prints `[school-field] flat-world parity: reports identical (enrolled <n>)` with `n > 0`. Full `cargo test --lib` green.

**Wiring requirement:** `CivitasGame` gains `school_field: Option<crate::coverage_field::ReachField>` + `school_field_dirty: bool` (private, beside `childcare_field` at game/mod.rs:237–240, init in `new_small`); `rebuild_school_field_if_dirty` mirrors :713–734 with seeds = operational `PrimarySchool|SecondarySchool` positions; `place_city_service` (:340) sets `school_field_dirty = true` when `matches!(kind, CityServiceKind::PrimarySchool | CityServiceKind::SecondarySchool)`; `set_coverage_field` (:292) clears + dirties it; the 1d.3 closure gains the school arm. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing tests.**
  - `coverage_field::tests::school_reach_severs_across_water` (GPU-skippable like its siblings): the `coverage_field_build_time_1024` river fixture (full-width river z ∈ [200, 260] on the Town map); game on that map with the service injected; households with 8-year-olds on the south bank (`seed_household_at` cannot place on-map — use the zoned-strip pattern from `household::tests` to get instance-linked anchors south of the river, or anchor via seeded ids and `field.contains` guard: pin whichever the existing river test used — it drives `rebuild_childcare` directly; this test drives `game.tick()` with a school plopped NORTH only and asserts `game.access.enrolled == 0`, then plops a south-bank school and asserts `enrolled > 0` after one tick); print both.
  - `game::tests::school_access_flat_world_matches_barrier_closure`: flat world, 12 children + school (Task 1 fixture); tick; clone the report; assert the live report equals a direct `services_access::assign` re-run with the pure barrier closure (no field exists on flat worlds — parity is bit-exact); print.
- [ ] **Step 2: Run to verify failure** — `cargo test --lib school_reach 2>&1 | tail -5` → FAIL (`ReachField` unknown / field absent).
- [ ] **Step 3: Implement.** In coverage_field.rs: rename `ChildcareField` → `ReachField` with `pub type ChildcareField = ReachField;` (zero churn for existing callers/tests), and add the seed-generic entry `pub fn rebuild_reach(&mut self, terrain: &MapTerrain, extent: ([f32;2],[f32;2]), seeds: &[[f32;2]]) -> ReachField` — today's `rebuild_childcare` body verbatim; `rebuild_childcare` becomes a one-line delegate (doc: "childcare-named convenience"). In game/mod.rs: the two fields, `rebuild_school_field_if_dirty` (flat world ⇒ `None`; no service ⇒ no-op; seeds from `sim.services.buildings` filtered to operational school types), the dirty hooks.
- [ ] **Step 4: Wire the closure** — 1d.3 becomes:

```rust
self.rebuild_school_field_if_dirty();
let terrain = &self.terrain;
let school_field = self.school_field.as_ref();
let is_school: Vec<bool> = self.sim.services.buildings.iter()
    .map(|b| matches!(b.service_type,
        vox_sim::services::ServiceType::PrimarySchool
        | vox_sim::services::ServiceType::SecondarySchool))
    .collect();
self.access = crate::services_access::assign(&mut self.sim, &self.households, |idx, bp, dp, r| {
    if let Some(field) = school_field {
        if is_school[idx] && field.contains(dp) {
            return field.reachable(dp, r);
        }
    }
    let (dx, dz) = (bp[0] - dp[0], bp[1] - dp[1]);
    dx * dx + dz * dz <= r * r && !terrain.line_crosses_water(bp, dp)
});
```

  Pins: the school field mixes both school kinds as seeds — the **same nearest-seed approximation** the childcare field already accepts across Kindergarten (1500 m) and Daycare (800 m); health stays on the barrier arm in M2 (health never gates and only informs — the closure seam is where a health field plugs in later, documented in the module doc).
- [ ] **Step 5: Run — verify non-trivial output.** Both tests pass with printed numbers; `cargo test --lib coverage_field::` keeps the three existing tests green (the alias proves the rename is invisible).
- [ ] **Step 6: Commit** — `feat(deep-sim-m2): geodesic school reach — ReachField generalization + school_field mirror` + footer.

---

## Task 3: The seat gate — SchoolChild coverage = slot AND seat, parents gated exactly like missing childcare

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/game_mechanics/care/mod.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs`

**Acceptance:** `cargo test --lib game::tests::an_overfilled_school_gates_parents_until_seats_open -- --nocapture` prints the tick timeline:
`[school-gate] t=3 seats 500/500 · unplaced 250 · gated 50` then `[school-gate] +PrimarySchool #2` then `[school-gate] t=5 seats 750/1000 · unplaced 0 · gated 0 · parents returned: true`. And `cargo test --lib care:: -- --nocapture` includes `require_seat_revokes_only_seatless_schoolchildren` printing `revoked 1 of 2 schoolchildren; infant cover untouched; served stays slot-semantics`. Full `cargo test --lib` green; `git diff` shows **zero** changes to `worker_is_free_to_work` or tick steps 3/4a/4b.

**Wiring requirement:** `CareCoverage::require_seat` is called in `CivitasGame::tick` step **2b**, immediately after the `assign_slots_with_coverage` block (game/mod.rs:838) and before step 3 (:840). `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing tests.**
  - care/mod.rs `require_seat_revokes_only_seatless_schoolchildren` (unit): demographics with one household holding an infant + two schoolchildren; a CareSystem with Kindergarten + AfterSchool in range; `assign_slots` covers all three; `let revoked = cov.require_seat(&demo, |id| id != <second schoolchild id>);` assert `revoked == 1`, `is_covered(infant)` and `is_covered(first)` still true, `is_covered(second)` false, and `cov.served[&SchoolChild]`/`unmet(SchoolChild)` **unchanged** (slot semantics preserved); print.
  - game/mod.rs `an_overfilled_school_gates_parents_until_seats_open` (the Done-When timeline): `new_small`; seed **150 households × 5 children age 8.0** at residences 300_000..300_149 (`seed_household_at(r, 34.0, &[8.0; 5])` — anchors span x = 3_000_000..3_001_490, 1,490 m; center `[3_000_745.0, 0.0]`); `add_jobs(center, 400)`; **7× `place_care(CareKind::AfterSchool, center)`** (840 slots ≥ 750 children — slots are deliberately ample so the seat is the only binding constraint; radius 1800 m covers the 745 m half-span); **1× `place_city_service(CityServiceKind::PrimarySchool, center)`** (500 seats, radius 1000 m covers all). Tick 3×; assert `access.school_seats == 500`, `enrolled == 500`, `unplaced_children == 250`, `stats.gated_workers == 50` (the 50 seatless households' single workers — seats deplete in child-id = household order), and `coverage.unmet(DependentKind::SchoolChild) == 0` (slots ample — the cause is seats, not clubs); print t=3 line. Plop the second PrimarySchool at center; tick 2×; assert `school_seats == 1000`, `enrolled == 750`, `unplaced == 0`, `gated_workers == 0`, and effective_employed strictly higher than the gated tick's; print the t=5 line + `parents returned: true`.
- [ ] **Step 2: Run to verify failure** — `cargo test --lib require_seat 2>&1 | tail -5` → FAIL `E0599` (no method `require_seat`).
- [ ] **Step 3: Implement** in care/mod.rs (`impl CareCoverage`):

```rust
/// The M2 school tie-in: a SchoolChild is covered only while it holds BOTH an
/// after-school slot AND a school seat. Removes cover from every SchoolChild
/// `has_seat` rejects and returns how many were revoked. `served`/`unmet()`
/// keep slot semantics — the seat gap reports through
/// `ServiceAccessReport::unplaced_children`, so the two causes stay
/// distinguishable on every surface. Deterministic: households/dependents in
/// demographics order; HashSet::remove is lookup, not iteration.
pub fn require_seat(&mut self, demo: &Demographics, has_seat: impl Fn(u32) -> bool) -> u32 {
    let mut revoked = 0;
    for h in &demo.households {
        for d in &h.dependents {
            if d.kind == DependentKind::SchoolChild
                && !has_seat(d.citizen_id)
                && self.covered.remove(&d.citizen_id)
            {
                revoked += 1;
            }
        }
    }
    revoked
}
```

- [ ] **Step 4: Wire at the exact callsite** — game/mod.rs, after :838:

```rust
// 2c. The school tie-in (design §4.3): a SchoolChild stays covered only with
//     BOTH a slot and a seat, so the UNCHANGED gate below (steps 3/4) pulls
//     parents off the workforce for a full school exactly as it does for a
//     missing kindergarten.
let access = &self.access;
self.coverage.require_seat(&self.demographics, |id| access.has_seat(id));
```

- [ ] **Step 5: Run — verify non-trivial output.** Timeline prints the exact numbers (500/500/250/50 → 750/1000/0/0); full `cargo test --lib` green — Founders' Vale/Demo tests untouched (no students exist there yet; Task 6 proves it with printed numbers).
- [ ] **Step 6: Commit** — `feat(deep-sim-m2): a full school gates parents — SchoolChild coverage = slot AND seat` + footer.

---

## Task 4: Land value feels seats — exhaustion-aware school/health terms (one-tick lag for free)

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/land_value/mod.rs`

**Acceptance:** `cargo test --lib land_value::tests::seat_exhaustion_drops_the_school_term -- --nocapture` prints `[land_value] school term: free seats lv <a> -> exhausted lv <b> (delta -0.100)` with `a - b == W_SCHOOL` (± 1e-4). Full `cargo test --lib` green (the M1 land-value tests use never-exhausted services — `current_users == 0 < capacity` — and stay byte-identical).

**Wiring requirement:** inside `LandValueField::recompute`, the `school(p)` predicate becomes *operational school type in radius AND `capacity > 0 && current_users < capacity`*; the `health(p)` predicate becomes *operational `Clinic|Hospital` in radius AND `capacity > 0 && current_users < capacity`*. **No `LandValueInputs` change**: `inputs.services` already carries `current_users`, and because 1d.2 runs before 1d.3 those are **last tick's** values — the design's one-tick lag by tick order. Park/police terms untouched. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing test** — `seat_exhaustion_drops_the_school_term`: zoned strip + in-radius PrimarySchool; shrink its `capacity` to the strip's child count via the pub field; seed that many 8-year-olds in reach; tick 2× (tick 1 fills seats at 1d.3; tick 2's 1d.2 reads the full school); record `lv_full = game.land_value.lot(0, 0)`; comparison run without the children (seats free): `lv_free`; assert `lv_free - lv_full == W_SCHOOL` ± 1e-4 and both in bounds; print.
- [ ] **Step 2: Run to verify failure** — `cargo test --lib seat_exhaustion 2>&1 | tail -5` → FAIL (assertion: delta 0.0 — the term is still presence-only).
- [ ] **Step 3: Implement** — the two predicate changes only; doc comment names the lag ("reads last tick's current_users — 1d.2 runs before 1d.3") and the boundary ("a precisely-full building reads exhausted").
- [ ] **Step 4: Wiring is the existing 1d.2 call** — confirm with the test (no new callsites).
- [ ] **Step 5: Run** — the new test passes with the printed `-0.100` delta; `cargo test --lib land_value::` keeps all M1 tests green unchanged.
- [ ] **Step 6: Commit** — `feat(deep-sim-m2): land value feels seat/capacity exhaustion (one-tick lag via current_users)` + footer.

---

## Task 5: Diagnosable chrome (lib-side) — the seat alert + Statistics lines

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/game_mechanics/notifications/mod.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs` (one `notes.extend` line at :915)
- Modify: `~/Ochroma/projects/urban_horizon/src/ui/panels.rs`

**Acceptance:** `cargo test --lib notifications::tests::seat_shortage_raises_a_critical_diagnosable_alert -- --nocapture` prints `CRIT School seats full — 250 children unplaced — parents stay home until seats open.` and `slot alert absent: true` (the after-school warning must NOT fire when slots are ample — causes never masquerade); `cargo test --lib panels::tests::stats_lines_carry_seat_and_health_capacity -- --nocapture` prints the two Statistics lines with the live report's numbers embedded. Full `cargo test --lib` green.

**Wiring requirement:** `seat_notifications(&self.access)` is extended into the feed at game/mod.rs:915 (between `service_notifications` and the severity sort); the Statistics lines are inserted in `stats_lines` (ui/panels.rs:57) directly after the `Unmet care` line, reading `game.access`. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing tests.**
  - notifications test: drive the Task 3 overfill fixture 3 ticks; assert the feed contains a `Severity::Critical` note with `title == "School seats full"` and message starting `"250 children unplaced"`; assert **no** `"No after-school care"` note (slots ample — `unmet_schoolchild == 0` because Task 3 kept served slot-pure); assert the existing `"Parents can't work"` Critical fires alongside (the WHY chain: gated → cause). Print the formatted feed line + `slot alert absent: true`.
  - panels test: same fixture; `stats_lines(&game, &stats)` joined contains `format!("School seats      {} / {}   unplaced {}", a.enrolled, a.school_seats, a.unplaced_children)` and `format!("Health capacity   {} cap · {} in reach", a.health_capacity, a.health_in_reach)`; print both lines.
- [ ] **Step 2: Run to verify failure** — `tail -5` → FAIL `E0425` (`seat_notifications` not found) / missing lines.
- [ ] **Step 3: Implement.**

```rust
/// The seat-shortage alert — the diagnosable cause behind seat-gated parents
/// (design M2's named risk: a labor collapse must name its reason).
pub fn seat_notifications(access: &crate::services_access::ServiceAccessReport) -> Vec<CareNotification> {
    if access.unplaced_children == 0 { return Vec::new(); }
    vec![CareNotification::new(
        Severity::Critical,
        "School seats full",
        format!("{} children unplaced — parents stay home until seats open.", access.unplaced_children),
    )]
}
```

  panels.rs: the two lines after `Unmet care` (exact formats above — plain integers, the info-window contract).
- [ ] **Step 4: Wire** — game/mod.rs:915 gains `notes.extend(crate::game_mechanics::notifications::seat_notifications(&self.access));` before the sort.
- [ ] **Step 5: Run** — both tests pass with the printed lines; full suite green.
- [ ] **Step 6: Commit** — `feat(deep-sim-m2): diagnosable seat shortage — CRIT feed alert + Statistics seat/health lines` + footer.

---

## Task 6: Scenario balance verification + replay equality — adjust only with printed justification

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/scenario/mod.rs`
- Modify: `~/Ochroma/projects/urban_horizon/src/game/mod.rs` (the replay test)

**Acceptance:** `cargo test --lib scenario:: -- --nocapture` → `m2_scenario_shapes_survive_seats` prints
`[scenario] founders-vale: students 0 · seats 0 · unplaced 0 · gate untouched (eldercare gap intact)` and
`[scenario] demo: students 10 · enrolled 10 / 500 seats · unplaced 0 · gated unchanged · eldercare gap intact — justification: school line must read non-zero in the M2 human pass`; all four pre-existing scenario tests pass **unmodified except** `demo_city`'s seeding line. And `cargo test --lib game::tests::replay_reproduces_service_access -- --nocapture` prints `roundtrip: identical (enrolled 750, current_users [500, 250], version 5)`. Full `cargo test --lib` green.

**Wiring requirement:** the ONLY balance change permitted is `demo_city`'s child-age mix: every third family (`i % 3 == 0`) seeds its child at age `8.0` (school-age) instead of `2.0 + (i % 4)` — 10 students, enrolled 10/500 at the existing PrimarySchool, after-school covered by the existing club (cap 120), **no new gating, the eldercare teaching gap untouched**. Founders' Vale is verified and **not** changed (engine pop is all 19–58; its seeded children are 2–5-year-old infants; 0 students ⇒ seats never bind ⇒ the eldercare lesson is intact — print this as the no-change justification). If verification shows anything else firing (e.g. unexpected gating), **stop and report — do not retune silently**. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Write the failing test** `m2_scenario_shapes_survive_seats`: build both scenarios; tick Founders' Vale 9× (its existing test horizon) — assert `access.unplaced_children == 0`, `access.school_seats == 0`, `unmet_infant == 0`, `unmet_elder > 0`, `gated_workers > 0` (elders, as today); tick Demo 6× — assert `access.enrolled == 10`, `school_seats == 500`, `unplaced_children == 0`, `coverage.total_unmet() > 0` (eldercare gap); print both justification lines.
- [ ] **Step 2: Run to verify failure** — Demo asserts fail (`enrolled == 0` — all children are infants today).
- [ ] **Step 3: Implement** the demo_city seeding change (one expression: `let child_age = if i % 3 == 0 { 8.0 } else { 2.0 + (i % 4) as f32 };`) with a comment citing this task's justification.
- [ ] **Step 4: The replay test** — `game::tests::replay_reproduces_service_access`: the Task 3 overfill city (both schools placed) run 40 ticks; `to_save()`/`from_save()`; assert `loaded.access == game.access` (derived `PartialEq`), the per-building `current_users` vectors equal, `loaded.coverage.demand/served/capacity` maps equal and per-dependent `is_covered` parity (the coverage_field flat-world test's loop pattern), `save.version == 5`; print the roundtrip line. This is the **derivability proof**: capacity state is recomputed by replayed ticks, never persisted — `git diff src/game/save.rs` must be empty.
- [ ] **Step 5: Run** — both tests pass with printed justifications; the four pre-existing scenario tests + both pre-existing save round-trip tests green.
- [ ] **Step 6: Commit** — `feat(deep-sim-m2): scenario seat verification (demo gains 10 students, justified) + replay equality of access state` + footer.

---

## Task 7: The observable surface — info lines, house Access line, service-building inspector, `--shot-school` proof harness

**Files:**
- Modify: `~/Ochroma/projects/urban_horizon/src/command/mod.rs` (`ViewState.inspected_service`)
- Modify: `~/Ochroma/projects/urban_horizon/src/bin/play.rs`

**Acceptance:** `cargo run --release --bin play -- --shot-school /tmp/school` prints, in order: the overfill city's timeline (`[shot-school] t=3 seats 500/500 · unplaced 250 · gated 50`), the feed line (`CRIT School seats full — 250 children unplaced — parents stay home until seats open.`), the Statistics school line, the post-fix line (`[shot-school] t=5 seats 750/1000 · unplaced 0 · gated 0 · parents returned: true`), the live info line (`School seats      750 / 1000 enrolled`); then on Demo City (Space ×10): the full service-inspector block for the Primary School including `Seats        10 / 500`, one house's inspector block including its `Access       school 0.N km · health 0.N km · care OK|GAP` line, `wrote /tmp/school/school_inspect.png`, exit 0 (exit 1 with a printed reason on any failed inequality). Then the human Done-When pass.

**Wiring requirement:** `info_lines` (play.rs:398) inserts, directly after the `Care sites` line (:410): `format!("School seats      {} / {} enrolled", a.enrolled, a.school_seats)` and `format!("Health capacity   {} cap · {} in reach", a.health_capacity, a.health_in_reach)`. `inspector` (play.rs:428) inserts the eighth line between `Land value` and `Zone`: `Access       school {:.1} km · health {:.1} km · care {OK|GAP}` — distances Euclidean from the lot center to the nearest operational school / clinic-or-hospital (display-only; `-` when none exists), `care` = GAP iff the household at `inst.engine_building` has any uncovered dependent (no household / no dependents ⇒ OK). New `fn service_inspector(&self, sid: u32) -> Option<(String, Vec<String>)>` returns (`Service — {label}`, lines `Type / Seats|Capacity / Radius / Status`): `Seats        {current_users} / {capacity}` for school types, `Capacity     {current_users} / {capacity}` for Clinic/Hospital, `-` for capacity-0 kinds; label via `CityServiceKind::all().find(|k| k.service_type() == b.service_type)`. `inspect_click` (:470): instance first; else the nearest operational service building whose ground distance ≤ `SERVICE_PICK_M = 25.0` (ties to lower id) into `view.inspected_service`; opening either window clears the other + `info_open`; Esc and the close/swallow handlers mirror the instance-window pattern (:1560–1577) with `service_inspector`'s dynamic line count; render after the instance-inspector block (:935) via `hud::draw_city_info_window`. `todo!()` / stubs = **task failure**.

- [ ] **Step 1: Add `inspected_service: Option<u32>`** to `ViewState` (command/mod.rs:112, init :158, cleared at the Esc arm :625 and wherever `inspected` is cleared) — `cargo build` green.
- [ ] **Step 2: Implement the two info lines + the Access line + `service_inspector` + pick/draw/click plumbing** per the wiring requirement; `cargo build --release --bin play` clean.
- [ ] **Step 3: Write the harness** `--shot-school <out_dir>` (extend the `main()` arg scan at :2554-pattern, model on `shot_landvalue` :2009): phase 1 headless `CivitasGame` — the Task 3 overfill fixture, tick 3, print timeline/feed/stats lines (exit 1 if `gated != 50` or the CRIT line is absent), plop school #2, tick 2, print the recovery line (exit 1 if `gated != 0`); phase 2 — Demo City `GameView`, Space ×10, print `info_lines()` (must contain the School seats line with `10 / 500`), `inspect_click` on the Primary School's position (`[95, 200]` → its window), print the service block, click a house instance, print its Access line, re-open the service window, render + `save_rgba` to `school_inspect.png` (`.png` extension required — documented image-crate behavior).
- [ ] **Step 4: Run the acceptance command** — exits 0 with every line printed; open the PNG: the Demo city with the `Service — Primary School` window showing `Seats        10 / 500`.
- [ ] **Step 5: Full-suite + regression sanity** — `cargo test --lib` green; `cargo run --release --bin play -- --shot /tmp/shot_regression.png` still exits 0 (the 8-line inspector is read dynamically).
- [ ] **Step 6: Commit** — `feat(deep-sim-m2): observable seat surface — info lines, Access line, service inspector, --shot-school proof` + footer. Then the **human Done-When pass** (§Done When item 3).

---

## Out of Scope (M3–M5 — future plans on the same design)

- `TrafficModel`, the `t` overlay, `W_CONG`'s real input (M3); `GoodsLocal`/tenants (M4); desirability ordering + wellbeing demand modifier (M5).
- Healthcare **gating** workers (design §4.3: no double jeopardy — health informs land value/info/alerts only); a geodesic health field (the closure seam is ready).
- Education quality/outcomes (`process_education` untouched); University seats (not an `AccessKind` in v1; the generic `ServiceType::School` is not placeable in civitas and contributes nothing).
- Engine-crate edits of any kind; `SAVE_VERSION` bumps; new `AuthoredAction`s; per-building geodesic distance (nearest-seed approximation, as childcare).
- Seat-gating softening (one guardian instead of all — design §8, revisit after the M2 playtest as a constant-free rule change).

---

## Self-Review Checklist

- [ ] Every task implements AND wires in the same task — assign wired at 1d.3 (Task 1), field arm in the live closure (Task 2), `require_seat` at 2b (Task 3), predicates inside the existing 1d.2 (Task 4), feed/stats wired into the tick + panel (Task 5), scenario/replay proven in-game (Task 6), windows/lines/harness live (Task 7)
- [ ] Every `Acceptance` names exact commands and exact non-trivial printed output (500/500/250/50 timeline, 0.100 delta, 10/500 Demo line, roundtrip counts) — never "tests pass"
- [ ] Every `Wiring requirement` names exact functions/files with current line anchors (game/mod.rs:783/838/915, panels.rs:57, play.rs:398/410/428/470, command/mod.rs:112)
- [ ] `IMPORTANT NOTES` carries code-verified signatures including every design/code divergence (in_reach param, no LandValueInputs change, binary capacity-weighting, plain-integer formats, uniform gate severity) and the capacity-source + gating-mechanism decisions
- [ ] `File Map` lists every file in any task; NYC-M1-colliding files justified by the plan-wide gate
- [ ] No step contains `todo!()`, `unimplemented!()`, or stub bodies; no engine edit anywhere
- [ ] `Done When` names specific commands and human-observable results (printed lines, a PNG, the keyboard pass) and the diagnosability chain (gated → CRIT cause → stats/info lines)
- [ ] Names/signatures consistent across tasks (`ServiceAccessReport`/`has_seat`/`assign`, `require_seat`, `ReachField`/`rebuild_reach`, `seat_notifications`, `service_inspector`)
- [ ] Replay safety proven, not asserted: `save.rs` zero-diff, version stays 5, `replay_reproduces_service_access` field-by-field
- [ ] Scenario numbers adjusted exactly once (Demo's 10 students) with the printed justification; Founders' Vale verified unchanged
