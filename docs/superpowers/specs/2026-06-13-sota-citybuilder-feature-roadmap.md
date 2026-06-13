# Design: SOTA Care-First City Builder — Feature Roadmap (2026-06-13)

**Status:** Draft
**Scope:** One ranked roadmap turning Civitas Care from a strong-economy-but-shallow-citizen prototype into a SOTA city builder whose differentiator is *care-first wellbeing* — ordered by (player value × differentiation) / effort, dependencies respected, MVP vs later waves marked.
**Related:** synthesizes 6 domain surveys (simulation, economy, traffic, growth, services_care, ux_meta). All work is GAME-LAYER (`~/Ochroma/projects/civitas_care/src/`) unless explicitly noted as an engine-agnostic primitive in `~/src/ochroma/crates/vox_*`.

---

## 1. Problem Statement

- The game is named **Care** but care is only an *employment gate* (`game_mechanics/care/mod.rs:263`). It never produces a wellbeing *outcome*: engine `Needs` default to `0.5` and never move (`vox_sim/src/citizen.rs:52`); `CitizenManager::tick` (`:133`) only ages and copies a frozen mean into `satisfaction`. Every non-employment citizen state is animated wallpaper — the exact CS1 trap.
- Services compute coverage reports that **feed nothing back to citizens**: `ServiceAccessReport.health_in_reach` (`services_access/mod.rs:48`) is computed and discarded; health/safety/leisure/education stay static `0.5`.
- "Reach" is **crow-flight Euclidean**, not network travel (`services_access/mod.rs:207`); the road network (`road/mod.rs`) is geometry-only with no routable graph — a clinic across an uncrossable motorway counts as "covered."
- Buildings **never level up**: `develop_lot` builds at level 1 and the comment at `game/mod.rs:713-714` admits "upgrade mechanics can later replace this." Land value is computed (`land_value/mod.rs`) but read by nothing.
- Companies **can't fail**: `Company` earns a flat `COMPANY_MARGIN_PER_JOB=8.0` (`ownership/mod.rs:49`), never runs a real P&L, never goes bankrupt — CS2 Economy-2.0's headline fix is missing.
- The UI has **no time-series graphs** (explicitly out of scope, `ui/panels.rs:8`), **2 info overlays vs CS2's 33**, a notification system that is **computed but never rendered**, and a building inspector that ignores the **ownership/resident/rent data the model already holds**.

---

## 2. Done When

Running `cargo test -p civitas_care wellbeing_loop_closes -- --nocapture` prints the out-migration delta between a care-and-clinic-covered household and an identical uncovered one, and a human running `cargo run --bin play`, deleting a city's only clinic, sees the **Wellbeing overlay** redden over the affected blocks, a **Critical notification** appear in the bell feed, the **Wellbeing time-series graph** trend downward, and within N ticks **residents out-migrate** — all visible on screen without reading code.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Wellbeing registry feeds migration | `assert!(uncovered.out_migration > covered.out_migration)` with two seeded households, clinic present/absent | `assert!(registry.is_some())` |
| Health is dynamic & care-addressable | `assert!(health_after_clinic_loss < health_before - 0.1)` over T ticks | `assert_eq!(health, 0.5)` |
| Network-aware access | route home→school across unbridged river returns `None`; old Euclidean returned `Some` | `assert!(route.is_ok())` |
| In-place level-up | building at sustained high land value repoints to level+1 asset; `assert_eq!(instance.level(), 2)` after `LEVEL_AFTER` ticks | `assert!(level >= 1)` |
| Company P&L + bankruptcy | unprofitable company's `balance` goes negative N ticks → slot freed; `assert_eq!(slot.state, Vacant)` | `assert!(company.balance() != 0.0)` |
| Loneliness axis (no competitor has it) | elder near eldercare+family: `assert!(loneliness < isolated_elder.loneliness)` | `assert!(loneliness.is_finite())` |
| Notification feed renders | `cargo run` shows bell badge count == `count_at_least(Critical)`; click lists `care_notifications` | feed struct exists |
| Wellbeing time-series graph | `draw_line_graph` over a recorded ring buffer; hover-scrub prints value at tick | buffer length > 0 |
| Rich building inspector | inspector lists named residents + per-dependent care status + rent from `ownership::TickFlows` | inspector prints "Building" |

---

## 4. Architecture

**Architectural law (CLAUDE.md):** engine crates (`vox_*`) never learn game concepts. Care, wellbeing, loneliness, health-as-sickness, levels, land value, companies, transit policy are GAME concepts. Two engine-agnostic primitives are the *only* permitted engine touches: (a) a generic `CitySim::pollution_at([f32;2]) -> f32` accessor + calling the already-generic `pollution.tick()`; (b) a routable road-graph + `route()` query on `vox_core::navmesh` / `vox_nn::street_layout` (graphs are game-agnostic). Everything else is a game-side registry reconciled each tick, mirroring the proven `HouseholdRegistry` determinism contract (`household/mod.rs:65`): id-sorted iteration, no RNG, no HashMap iteration, no wall clock, replay-recomputed, save-version-gated.

### 4.1 WellbeingRegistry (the spine — Wave 1)
A game-side `src/game_mechanics/wellbeing/` registry keyed by `citizen.id`, reconciled each tick **after** the coverage passes and **before** the satisfaction read at `game/mod.rs:1181`. It *consumes* the already-computed `ServiceAccessReport`s + care coverage + pollution field and produces per-citizen multi-axis state (`health`, `wellbeing`, `loneliness`, `community`) plus a derived `satisfaction` that replaces the dead `Needs::satisfaction` mean as the migration driver. The engine `Needs` struct stays the engine's thin proxy; the deep model is game-side. This is the neighbor every other care system has been missing — once it exists, health/loneliness/community/child-development are all "add a sampler + close a loop" rather than new machinery.

### 4.2 Dynamic Health (Wave 1)
Per-tick health update: baseline decay accelerated by pollution + uncovered eldercare/childcare, recovered by in-reach clinic/hospital capacity (`ServiceAccessReport.health_in_reach` — already computed). Low health → sickness flag → reuse the care-gate plumbing (`game/mod.rs:1166`) to pull the citizen off work; health modulates the existing death check and (later) birth rate. Care-specific contributions injected from the game via the same closure pattern as `assign_slots_with_coverage`'s `in_reach`.

### 4.3 Routable Road Graph + Network Access (Wave 2, the P0 traffic spine)
Engine: add edge adjacency + per-edge `travel_cost` and an A*/Dijkstra `route(from,to) -> (path, time)` reusing `vox_core::navmesh`'s proven spatial-grid `nearest_node` (<1ms/1000 nodes). Game: `road/mod.rs::to_routable_graph()`, then swap `services_access`'s `in_reach` callback from Euclidean-vs-radius to **network travel time ≤ threshold**. The entire nearest-first depletion machinery is unchanged — only the metric changes. This retroactively makes every care/service access *honest*.

### 4.4 In-Place Level-Up + Land-Value Loop (Wave 2)
Close the loop land value already computes: `assessed_value()` multiplies by `land_value.lot()`; `develop_parcels` orders by land value. Add a `level_streak` to the `instance` state machine and a `try_level_up` tick step: a building levels when, for `LEVEL_AFTER` consecutive ticks, land value ≥ `threshold(level)` AND sector bar satisfied AND the next-level directive `fits(w,d)` — then repoint the instance to the `level+1` catalog asset via the existing `lot_asset` path and re-cook geometry. **Care-first inversion of CS2's gentrification bug:** rent stays decoupled from raw land value; low-rent / care-supported tenancies near high land value do *not* auto-evict.

### 4.5 Company P&L + Bankruptcy (Wave 2)
Replace `COMPANY_MARGIN_PER_JOB` with `revenue = output × sale_price`, `costs = inputs × price + wages + rent + upkeep + fees`. Negative `balance` for N ticks (deterministic counter, no RNG) → mark slot vacant, free it via the instance registry, raise the sector demand bar. Wires into the existing `ownership::tick` and gives the CS2-style company-info panel a real net-profit line.

### 4.6 UX Surfacing Layer (Wave 1 + Wave 2)
The command registry (`command/mod.rs`) is the spine — every feature adds command ids so keys, palette, and future Ask-Civitas AI calls get them free. Wave 1: notification feed (bell badge + panel from the already-computed `notifications/mod.rs`), rich building inspector (residents + care status + rent from `ownership/`), and a `src/history/` ring buffer + `draw_line_graph` for the thesis metric ("wellbeing/care over time"). Wave 2: generalize `OverlayKind` into an overlay-descriptor registry (sampler + ramp + command id) with care-first views first (Care Reach by stage, Wellbeing, Service Access) and **accessibility-safe, opt-in, never-auto-open** ramps — a citable way to beat CS2's headache/epilepsy criticism.

### 4.7 Later Waves
W3: care-first wellbeing axes deepened (loneliness for elders via intergenerational proximity, community cohesion district meter, child-development from childcare *quality* raising future education ceiling); labor market v2 (typed worker/job tiers + commute + unfilled-jobs signal feeding demand); multi-stage production chains as directive data. W4: GPU-resident 1M-mover sim + visible commutes (renders into the resident splat frame); transit as a directive-cooked network with wait-time edge cost + ridership fares (fixes CS2's "transit ignores wait time"); citizen lifecycle (births, education progression, family formation); Lifepath / follow-a-citizen inspector; pack manager + in-game directive authoring panel (attacks CS2's "asset editor not intuitive").

---

## 5. Ranked Roadmap (the deliverable)

Ranked by (player value × differentiation) / effort, dependencies first. **F1–F8 = MVP (Waves 1–2). F9+ = later.**

### MVP — Wave 1 (the care-first identity becomes real)

**F1. WellbeingRegistry — the loop that closes** · MVP · Effort: M
- WHAT: game-side per-citizen multi-axis registry (`src/game_mechanics/wellbeing/`), reconciled in id order after coverage passes, driving `satisfaction` → migration.
- WHY / beats CS2: turns the whole sim from cosmetic to consequential; CS2 has the axes but they feel inert and "lifeless" (its own reviewers' word). This is the prerequisite neighbor every other care feature stands on.
- SYSTEMS: `wellbeing/mod.rs` (new), `game/mod.rs:1181` (tick wiring, mirror `household/mod.rs:65` determinism), reads `services_access/mod.rs:48`.

**F2. Dynamic, care-addressable Health** · MVP · Effort: M
- WHAT: per-tick health responding to clinic reach + pollution + uncovered care; low health → can't work → death/birth modulation.
- WHY / beats CS2: it's literally in the game's name; CS2 fakes health with a passive bonus, we make it a real care-driven loop. Depends on F1.
- SYSTEMS: `vox_sim/src/citizen.rs` (generic need update + closure injection), `wellbeing/`, reuse care-gate plumbing `game/mod.rs:1166`; engine accessor `CitySim::pollution_at` + `pollution.tick()` (`vox_sim/src/pollution.rs`, currently dead).

**F3. Notification feed (bell + panel)** · MVP · Effort: S
- WHAT: render the already-computed severity-ranked feed; bell badge `count_at_least(Critical)`, click opens list.
- WHY / beats CS2: cheapest high-impact win; data is pure & tested, only HUD wiring missing. Makes problems legible ("who/where," not "unmet: 12").
- SYSTEMS: `notifications/mod.rs` (have), `render_gpu/hud.rs` `StatusStrip`, `command/mod.rs` (`view.notifications`).

**F4. Rich building inspector (CS2 panel parity)** · MVP · Effort: S–M
- WHAT: extend inspector to read `ownership/` — owner, rent roll, company, named residents + per-dependent care-gap status.
- WHY / beats CS2: exactly the panels the user shared as reference; the model already holds the data, and the care-status-per-resident is the care-first twist CS2 lacks.
- SYSTEMS: `bin/play.rs:709` → move lib-side to `src/ui/`, `ownership/mod.rs`, `household/mod.rs` (`household_has_care_gap`).

**F5. Time-series history + graphs** · MVP · Effort: S–M
- WHAT: `src/history/` fixed-capacity ring buffer sampling per-tick figures already computed; `draw_line_graph` in the Statistics submenu; headline "Wellbeing/Care over time" line.
- WHY / beats CS2: SimCity 4 / W&R signature CS2 underdelivers; the care trend graph is the visual thesis no competitor has. Removes the `ui/panels.rs:8` "graphs out of scope" note.
- SYSTEMS: `history/mod.rs` (new, save-version bump), `render_gpu/hud.rs::draw_line_graph`, `ui/panels.rs`.

### MVP — Wave 2 (depth: honest access, growth, living economy)

**F6. Routable road graph + network access** · MVP · Effort: M–L
- WHAT: engine routable graph + `route()`; game swaps `services_access` `in_reach` to network travel time.
- WHY / beats CS2: makes all care/service access *honest* (the across-the-motorway clinic stops counting); foundation for commutes/transit/congestion. Care-first "can they actually get there?" + 15-minute-city scoring inverts CS2's car-centric framing.
- SYSTEMS: engine `vox_core/src/navmesh.rs` (reuse A*), `vox_nn/src/street_layout.rs`; game `road/mod.rs:303`, `services_access/mod.rs:207`.

**F7. In-place level-up + land-value loop** · MVP · Effort: M
- WHAT: land value feeds assessed value + dev ordering; `try_level_up` repoints to level+1 directive on sustained land value + wellbeing.
- WHY / beats CS2: the single biggest missing growth loop; cities stop being static L1 boxes. Care-first inversion fixes CS2's most-hated bug (gentrification eviction of the vulnerable) — the signature marketing mechanic.
- SYSTEMS: `instance/mod.rs` (`level_streak`), `game/mod.rs:713,1361,1579` , `land_value/mod.rs`, level-N directives already in catalog (`asset/directive.rs:38`).

**F8. Company P&L + bankruptcy + player-adjustable tax** · MVP · Effort: M
- WHAT: real per-company P&L → bankruptcy/vacancy; tax rates into UI + save; over-taxation bends demand → flight.
- WHY / beats CS2: CS2 Economy-2.0's headline (companies fail on bad lots) made real; answers "money is too simple." Care-first hook: surface "workers freed by care" and "cost of neglect" (lost tax + output when a parent is care-gated off work) as budget lines — the game's economic thesis statement.
- SYSTEMS: `ownership/mod.rs:49,220`, `finance/mod.rs` (`TaxPolicy` → UI/save), `demand/mod.rs:90` (flight feedback).

### Later — Wave 3 (the care-first wedge deepens; economy depth)

**F9. Care-first wellbeing axes: loneliness, community, child-development** · Later · Effort: M
- WHAT: elder loneliness lowered by eldercare + intergenerational proximity; district community-cohesion meter; childcare *quality* raises a child's future education ceiling.
- WHY / beats CS2: **no mainstream builder models any of these** (confirmed: CS2 `Game.dll` has zero childcare/eldercare/loneliness hits). The open lane. Mechanically a coverage pass on `assign_slots` — cheap once F1 exists.
- SYSTEMS: `wellbeing/`, `care/mod.rs:41` (efficiency→quality), `household/`.

**F10. Labor market v2 (typed tiers + commute + unfilled signal)** · Later · Effort: M–L
- WHAT: game-side `src/labor/` clearing workers-by-tier against jobs-by-sector-tier with down-fill + commute cost; feeds `unfilled_jobs` into demand + company output.
- WHY / beats CS2: replaces the crude 2-type matcher; care sector competes for the same worker pool (a tension no competitor has). Depends on F6 (commute) + F8 (output).
- SYSTEMS: `vox_sim/src/employment.rs` (spatial layer), new `labor/`, `demand/mod.rs:90`, `game/mod.rs:1256`.

**F11. Multi-stage production chains as directive data** · Later · Effort: M
- WHAT: `production` block in `asset/directive.rs` (inputs/outputs/jobs-by-tier); `goods::Good` becomes a small DAG; factory emits businesses *with recipes*.
- WHY / beats CS2: Anno/W&R "satisfaction of coordinating chains"; the directive-factory keystone applied to economy (authored, not coded).
- SYSTEMS: `asset/directive.rs`, `goods/mod.rs:110`, `factory/`.

### Later — Wave 4 (scale, movement, lifecycle, modding moat)

**F12. GPU-resident 1M-mover sim + visible commutes** · Later · Effort: L
- WHAT: promote `crowd.rs` to a GPU mesoscopic kernel writing into the resident splat instance buffer; trips generated from matched occupants.
- WHY / beats CS2: 1M legible, followable movers (movers are visual → `radix_sort` nondeterminism landmine is moot). Care-first vulnerable-mover lens (who is stranded). Depends on F6.
- SYSTEMS: engine `vox_sim/src/crowd.rs`, resident frame loop; game `src/trip/`.

**F13. Transit as directive-cooked network** · Later · Effort: M–L
- WHAT: transit lines/stops as a directive type added to the routable graph with wait-cost edges; mode choice by total travel time; ridership fares replace the pop-scaled subsidy.
- WHY / beats CS2: wait-time is an edge cost from day one (fixes CS2's "cims ignore transit wait"). Depends on F6 + forge-directive-factory.
- SYSTEMS: `asset/directive.rs`, `road/` graph, `budget/mod.rs:41` (fares).

**F14. Citizen lifecycle + Lifepath inspector** · Later · Effort: M–L
- WHAT: births into households, education progression gating jobs, family formation/move-out; per-citizen follow/Lifepath panel surfacing the wellbeing breakdown.
- WHY / beats CS2: the most-praised SOTA feature (follow-a-citizen) becomes the emotional payoff of care-first — watch a named resident's wellbeing recover when you care for them. Depends on F1.
- SYSTEMS: `vox_sim/src/citizen.rs`, `household/`, `ui/`.

**F15. Pack manager + in-game directive authoring** · Later · Effort: M
- WHAT: enable/disable packs in UI; sliders/dropdowns emit an `AssetDirective` JSON → Forge cook → live preview.
- WHY / beats CS2: directly attacks CS2's "asset editor not intuitive, shipped 5 months late"; the directive is JSON so an LLM (Ask Civitas) can author it.
- SYSTEMS: `asset/mod.rs:613`, `asset/directive.rs`, `render_gpu/mod.rs:3628` (preview path), `command/mod.rs`.

---

## 6. The care-first throughline

Every feature closes a loop back to *caring for named people*. The recursive care→labor gate already proves the team can make care mechanically load-bearing; this roadmap makes it **emotionally legible and visibly rewarding**: care raises wellbeing (F1) and health (F2), which the player sees in the inspector (F4), on the map overlay (F6-era), in the trend graph (F5), and feels in the economy as "workers freed by care" and "cost of neglect" (F8). The differentiators no competitor has: **loneliness/community/child-development axes (F9)**, **no-gentrification-eviction growth (F7)**, **the care-over-time thesis graph (F5)**, **the vulnerable-mover "can they get there" lens (F6/F12)**, and **accessibility-safe caring UI (F6-era overlays)**.

**Key risk (the CS1 trap):** every wellbeing axis added MUST close a feedback loop (axis → satisfaction → migration/labor/economy) the player can manage, or it's wallpaper. F1 enforces this contract; F9+ must obey it.

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `WellbeingRegistry::reconcile` | `CivitasGame::tick` step 2.5 | `src/game/mod.rs:~1181` | after coverage passes, before satisfaction read |
| `health update` | engine `CitizenManager::tick` + game closure | `vox_sim/src/citizen.rs:133` | care contributions injected via closure |
| `pollution.tick` / `pollution_at` | engine sim loop / `WellbeingRegistry` | `vox_sim/src/pollution.rs` | generic grid, currently unwired |
| notification feed render | HUD draw | `render_gpu/hud.rs` `StatusStrip` | bell badge from `count_at_least` |
| `draw_line_graph` | Statistics submenu | `render_gpu/hud.rs` | reads `history/` ring buffer |
| `route()` | `services_access` `in_reach` | `vox_core/src/navmesh.rs` + `road/mod.rs` | engine query, game metric |
| `try_level_up` | `tick` after `develop_parcels` | `src/game/mod.rs` | slot reserved at `:1016` |
| company P&L | `ownership::tick` | `src/ownership/mod.rs:220` | deterministic bankruptcy counter |

---

## 8. Open Questions

- [ ] Wellbeing granularity: per-citizen vs per-household registry? (Lean per-household for memory at 1M scale; per-citizen only for inspected/Lifepath cims.)
- [ ] Save-version strategy: one bump for the F1/F2/F5 additive fields, or staged per wave? (Recommend staged additive bumps mirroring the v3 `action_ticks` precedent.)
- [ ] Level-up footprint rule: regenerate via Forge per level (cost) vs pre-cooked level-N assets in catalog (memory)? (MVP uses pre-cooked catalog assets.)

---

## 9. Out of Scope

- CS2-grade deathcare hearse logistics, fire-truck pathing, crime-incident sims — table-stakes flavor that drained CS2's polish budget without making cities feel alive. Care wins on relationships, not logistics fidelity.
- Microscopic lane-level traffic (CS2's lane-cost instability is a known dead end at 1M) — we choose mesoscopic.
- Multi-GPU; physical goods transport (W&R-style) in the MVP economy (chains stay aggregate until F11).

---

## 10. Related Plans / Designs

- Depends on: engine resident GPU frame loop (memory `aaa-phase2-resident-frame`), forge-directive-factory (memory).
- Required before: per-wave plans in `docs/superpowers/plans/` (each must use `docs/templates/plan.md`; every `Done When` names an exact command + exact observable output; no `todo!()`, no wire-later).
- Related: `cs2-asset-model`, `building-content-pipeline`, `ochroma-beat-unreal-directive` (memory).
