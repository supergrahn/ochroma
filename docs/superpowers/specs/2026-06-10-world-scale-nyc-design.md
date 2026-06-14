# Design: World-Scale Real City — New York at 1:1 (2026-06-10)

**Status:** Draft — §2.0 scale targets are **normative until user veto** (reviewed 2026-06-10 without objection); amended 2026-06-11: footprints drive INSTANCES, not assets (archetype buckets, per-instance identity, fit-scale)
**Scope:** Build New York City at real scale inside Urban Horizon on the Ochroma engine: real NYC open data (DEM, building footprints, tax lots, streets) ingested into a deterministic snapshot, mapped onto Forge directives + the cooked-asset pipeline, placed at true coordinates, streamed at borough scale, and simulated under tiered sim-LOD — with an M1/M2/M3 Done-When ladder where every rung is a command a human can run.
**Related:** `[Virtualized Splat Rendering Design](./2026-06-10-virtualized-splat-rendering-design.md)`, `[Deep Simulation Design](./2026-06-10-deep-simulation-design.md)`, `[Living Building Instances Design](./2026-06-10-living-building-instances-design.md)`, `[SDF Pillar Design](./2026-06-10-sdf-pillar-design.md)`, plan `[World-Scale NYC M1](../plans/2026-06-10-world-scale-nyc-m1.md)`

**Layer rule (CLAUDE.md):** everything NYC-specific (datasets, BIN/BBL, addresses, zoning mapping) lives in `~/Ochroma/projects/urban_horizon`. Engine work in this design (M2 streaming) stays game-agnostic: cells, tiles, instances, splats. No engine crate ever learns what a "tax lot" is.

---

## 1. Problem Statement

- The game's largest map class is `MapSize::Huge` — 24.6 km span, `max_buildings: 24_000` (`src/map/size.rs`). NYC is ~49 km corner-to-corner with ~1.08M building footprints citywide (count verified at fetch time from the snapshot): the current ceiling is ~45× too small, and nothing in the codebase can ingest a real city's data.
- The engine proved virtualized instancing today — `scale_trial --instanced --buildings 10000` printed `virtual_atoms=50000000 | budget=1000000 selected=999997 | select+expand+render p50=72.46 ms` on the 780M — but there is **no path from a real footprint polygon to an `AtomInstance` placement**: placements today come from the zoning auto-layout (`plan_lots`), never from data.
- `vox_render::world_partition::WorldPartition` (3D `CellCoord` cells, LRU eviction, `SplatBudget { max_gpu_mb: 2048 }`) and `gpu::instancing::InstanceManager` are shipped and unit-tested but have **zero production callers** — the game streams terrain through its own `map::streaming::MapPartition` instead. Two parallel streaming systems, neither fed by real data.
- `forge-terrain` has a DEM loader (`dem_ingest::load_dem_png`) whose doc comment claims it is "Used when `TerrainParams.dem_source = Some(\"usgs\")`" — **`TerrainParams` has no `dem_source` field and `load_dem_png` has zero callers** (grep-verified). The DEM path exists only as dead code.
- The deep-sim design budgets **≤ 10 ms/tick at 1,000 instances** (traffic ≤ 5 ms of it). Manhattan alone is ~46k buildings; the city ~1.08M. Worse, `BuildingRegistry::tick` does `buildings.iter_mut().find(|b| b.id == ...)` per instance — an O(n²) scan that is invisible at 1k instances and fatal at 46k.
- `cargo run --release --bin play` today shows procedurally zoned fiction. Clicking a building can never show "96 Spring St, BIN 1008912" because no real-world identity exists anywhere in the data model.

---

## 2. Done When

### 2.0 Normative scale targets (normative until user veto — reviewed 2026-06-10 without objection)

The working SOTA targets every rung below is calibrated against:

- **World: 676 km²** — a single 26 × 26 km world window (f32 quantization ≈ 1.6 mm with the origin pinned at the window center; arithmetic in §4.1 — no floating origin needed).
- **Instances: 1,000,000 placed building instances** — placements, never unique assets; the unique-asset count stays bounded by archetype buckets (§4.3, §4.7).
- **Headline render claim: FRAME COST FLAT from 10k → 1M instances at a fixed atom budget** — proven by printed p50 at both ends of the same run/binary (`p50@10k=<a> ms`, `p50@1M=<b> ms`, `b ≤ 1.10 × a`): frame cost is a function of the budget, never of instance count.
- **Sim: full-fidelity deep-sim for ~50k instances near the interest set**, district aggregation beyond (the tier mechanism of the [Deep Simulation Design](./2026-06-10-deep-simulation-design.md); §4.6).
- **Perf gate: the 60 fps (16.6 ms) gate is DEFINED on `tomespensin` (RTX 4070 Ti)**; the 780M dev floor prints tracking numbers against its 33 ms floor with the mandatory tier+adapter gate line — per the M3.2 re-baseline (plan `../plans/2026-06-11-virtualized-splat-rendering-m3-2.md`).

A three-rung ladder. Each rung is independently observable; a rung is not "done" until its exact command prints its exact output (correctness rungs on the 780M; ms gates on `tomespensin` per the targets above).

**M1 — one real Lower-Manhattan tile (the plan `2026-06-10-world-scale-nyc-m1.md` executes exactly this):**
`cd ~/Ochroma/projects/urban_horizon && cargo run --release --bin play -- --nyc-shot nyc_m1_inspector.png` loads the cooked NYC tile (real 1-ft-DEM terrain; ~1,000 real footprints standing as **instances of ~25–40 cooked Forge archetypes** at true coordinates — **footprints drive instances, not assets**: one hand-tunable directive per shape × size × height bucket, every placement carrying archetype id + fit-scale + its own Address/BIN as per-instance metadata), headlessly clicks a pinned building, writes the PNG, and prints inspector lines including `Address      <a real street address from PLUTO>` and `BIN          1xxxxxx` (a real Manhattan BIN, leading digit 1) plus `latlon       40.72xxxx, -74.00xxxx (Δ <2.0 m vs snapshot)` — the printed lat/lon of the clicked world position matches the snapshot footprint centroid within 2 m, and the identity lines come **from the instance side-table, never from the asset**. The same tile is reachable by a human through the menu: New Game → map `NYC — Lower Manhattan M1` → Start Game → Inspect tool → click any building → the inspector window shows the Address/BIN lines.

**M2 — 100k instances, streamed at 60 fps:** `play` on the M2 map (Manhattan plus the facing Brooklyn/Queens waterfront — **~100k real footprints**, the `i` info line `Instances    ~100k` printing the real snapshot count) flies Battery → Inwood (~21 km); a `[stream]` line prints every 2 s shaped `[stream] cells loaded=<L> resident_mb=<M> evicted=<E> instances_visible=<V>` with `M ≤ 2048` (the `SplatBudget` default) for the whole flight and zero black frames; **frame gate: p50 ≤ 16.6 ms over the flight on `tomespensin`** (tier+adapter gate line printed; the 780M runs the identical flight printing its 33 ms tracking floor — M3.2 re-baseline protocol); the district sim tick prints `[sim] tier1 districts=<D> tick_ms=<t>` within the deep-sim budget.

**M3 — the 676 km² world, 1M instances, flat cost proven:** the 26 × 26 km world window centered on the city registered with **~1M placements** (real count printed at load; the window covers the dense boroughs' built mass — the bbox crop is printed, nothing silently dropped); teleporting to 5 pinned landmark coordinates renders each within the streaming budget, with the same `[stream]`/`[sim]` lines holding; clicking any building anywhere still shows its real Address/BIN from instance metadata; and **the headline claim is printed**: at the fixed atom budget, `p50@10k=<a> ms` and `p50@1M=<b> ms` from the same harness with `b ≤ 1.10 × a` — the 16.6 ms value of those p50s is gated on `tomespensin`, the 780M prints the same pair as tracking numbers (M3.2 re-baseline).

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Deterministic snapshot | second `nyc_fetch` run prints `OK (cached)` for every item with byte-identical sha256 (printed 12-hex prefix per file) | `assert!(path.exists())` |
| Projection correctness | per-footprint polygon area computed from projected local metres vs the dataset's own area attribute: median relative error printed, `< 1%` over ~1,000 footprints | asserting the projection function returns finite numbers |
| DEM ↔ footprint coherence | DEM sampled at each footprint centroid vs the footprint's `groundelev` attribute: `median |Δ| < 1.5 m` printed with n | `assert!(height > 0.0)` |
| Footprint → archetype bucketing (instances, not assets) | ~1,000 footprints collapse to ~25–40 archetype buckets (ONE directive each; `≤ 48` asserted); the actual histogram printed (bucket id → placement count); fit-scale p95 within [0.80, 1.25] printed; low-rise/placeholder split printed with real counts; `instance/asset ratio ≥ 20:1` printed | `assert!(!buckets.is_empty())` |
| Real-coordinate placement | clicked building's world position inverse-projected to lat/lon matches the snapshot centroid `< 2 m`, both printed | comparing instance count only |
| Real identity in inspector | inspector lines contain the PLUTO address string and the footprint BIN for the clicked instance, printed by `--nyc-shot` | a hard-coded address string |
| Streaming at 100k (M2) | `[stream]` line: `resident_mb ≤ 2048` across the full Battery→Inwood flight at ~100k registered instances, evictions > 0 (proves cells actually unload); p50 ≤ 16.6 ms on `tomespensin`, 780M tracking number printed | streaming "enabled" flag asserted |
| Flat frame cost (M3 headline) | same harness, same fixed atom budget: `p50@10k=<a> ms` and `p50@1M=<b> ms` both printed, `b ≤ 1.10 × a` | "fps looks similar" / a single-scale measurement |
| Tiered sim (M2/M3) | `[sim] tier1 ... tick_ms=<t ≤ budget>` at 100k (M2) / 1M (M3) registered instances with the T0 full-fidelity set ~50k near the interest set; T0 promotion shows per-instance inspector detail unchanged vs full-sim oracle on a 1k-instance test city; city totals invariant under promotion/demotion | tick "completes without panic" |

---

## 4. Architecture

### 4.1 Coordinate frame and precision (grounded in the real engine types)

**The world frame is a local metric tangent frame derived from EPSG:2263 (NAD83 / New York Long Island, US survey feet), translated to a pinned tile origin and scaled to metres** (`1 ftUS = 1200/3937 m`, exactly). NYC's authoritative rasters (1-ft DEM) and planimetric data are natively EPSG:2263, so using it as the source frame introduces **zero projection error between datasets**; footprints fetched as EPSG:4326 GeoJSON are projected into 2263 by a self-contained Lambert-Conformal-Conic (2SP) implementation in the game (constants from the EPSG registry, correctness gated by the area cross-check in §3, not by trust). World axes: `x = east`, `z = south` (heightmap row 0 = northernmost row, matching `Heightmap` row-major sampling), `y = elevation in metres (NAVD88)`.

**f32 precision at NYC extent — fine, with numbers.** `GaussianSplat` stores `position: [f32;3]` (96-byte struct, `vox_core/src/types.rs:29`); `AtomInstance` likewise carries `position: [f32;3]`. f32 ULP is 2⁻²³ of the magnitude bracket: at offsets < 16.4 km the ULP is ≤ 2.0 mm; in the 16.4–32.8 km bracket it is 3.9 mm. NYC's citywide bbox is ~47 × 50 km; with the city origin pinned at the geographic centroid no coordinate exceeds ~30 km, so **worst-case quantization is ~4 mm at the far tip of Staten Island and < 1 mm anywhere in an M1/M2 play session** — far below splat scales (cm–dm). **The normative world window (§2.0) tightens this further: 26 × 26 km with the origin at the window center keeps every coordinate ≤ ~18.4 km, so f32 quantization is ≈ 1.6 mm at the window rim (≤ 1 mm over most of the window)** — the 676 km² target costs nothing in precision. Camera-relative subtraction in f32 view space inherits the same ~4 mm bound; that is below visible jitter for this content.
**Compatibility note (explicitly NOT scope):** planet-scale floating origin is not needed at ≤ 50 km. If M3+ ever wants sub-mm fidelity at the city rim, the cheap upgrade is **cell-local coordinates** — positions stored relative to their 512 m streaming cell's origin (cells already exist in both `WorldPartition` and `MapPartition`) — not a floating-origin rewrite. Recorded here so nobody designs for Earth when the city is the requirement.

### 4.2 Data ingestion — the pinned NYC snapshot

One game-side fetcher (`nyc_fetch`, curl-via-`std::process::Command`, the proven `polyhaven_fetch.rs` house pattern) downloads each dataset once into `assets/source/nyc/raw/`, records `{url, fetched_at, sha256, license}` per item in `assets/source/nyc/manifest.json`, and never re-downloads a cached item — **the snapshot, not the live API, is the deterministic input to everything downstream.** Datasets (names/URLs verified 2026-06-10):

| Dataset | Source / URL | What we take | License |
|---|---|---|---|
| **Building Footprints** | NYC Open Data (OTI), `data.cityofnewyork.us/City-Government/Building-Footprints/5zhs-2jue` — GeoJSON (EPSG:4326) via the Socrata SODA API, bbox-sliced with `$where=within_box(the_geom, …)` | polygon, **BIN**, BBL, `heightroof` (ft above ground), `groundelev`, `cnstrct_yr` | NYC Open Data: free public data, no use restrictions (Open Data Law, LL11/2012; opendata.cityofnewyork.us/faq) |
| **PLUTO** (tax lots) | NYC Dept. of City Planning, Bytes of the Big Apple / NYC Open Data (`nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page`) — version pinned at fetch (e.g. the current release id recorded in the manifest) | per-BBL `address`, `zonedist1`, `landuse`, `numfloors`, `bldgclass`, `lotarea` | NYC Open Data terms as above; DCP Bytes products carry an informational-purposes/no-warranty disclaimer (recorded in the manifest) |
| **1-ft DEM** | NYC Open Data: `1 foot Digital Elevation Model (DEM)` `dpc8-z3jc`; Integer Raster variant `7kuu-zah7` (3.3 GB zipped citywide GeoTIFF, EPSG:2263). 2010 LiDAR, bare-earth, hydro-flattened | a cropped window per tile, downsampled to the map cell grid | NYC Open Data terms as above |
| **LION** (street centerlines) | DCP, `nyc.gov/content/planning/pages/resources/datasets/lion` / NYC Open Data `LION 2v4z-66xt` — address ranges per segment | **M2**: road graph + addresses; M1 only records the pin in the manifest | NYC Open Data / DCP terms as above |
| **OSM (fallback + water)** | Overpass API, pinned query for the tile bbox; transcoded at fetch time into the simplified `{nodes, ways}` schema `vox_data::osm_import::parse_osm_json` actually expects (it does NOT read raw Overpass `elements` JSON — verified) | water polygons for `import_real`'s `WaterMask` (the Hudson/East River shoreline); footprint fallback where city data is absent | **ODbL** — requires attribution "© OpenStreetMap contributors" and share-alike on derived *databases*; the manifest stores the attribution string and the map's `Attribution` struct displays it |

The licensing posture is honest: every dataset above is open; NYC Open Data is unrestricted; OSM alone carries obligations (attribution + share-alike for redistributed derived databases), which the snapshot manifest and the in-game `Attribution` line both satisfy.

### 4.3 Footprints → Forge directives — what maps, and what Forge cannot build today

Forge's footprint vocabulary (`~/src/forge/crates/building/src/footprint.rs`, code-verified): exactly four parametric shapes — `Rectangular | LShaped | UShaped | TShaped` — at **fixed internal proportions** (L is a half/half cut, U is exact thirds, T is thirds × 0.67 depth). `BuildingDescription` (`description.rs`): `floors: u8`, `floor_height: f32`, `RoofStyle { Flat | Gabled | Hip | Mansard }`, `SidingType { Clapboard | Brick | Stone | Stucco | Vinyl | TimberFrame }`, `Condition { New..Derelict }`, style keys as JSON data.

**Mapping (M1) — footprints drive INSTANCES, not assets:** per footprint, fit an oriented bounding box (rotating calipers) → `width/depth/yaw`; rectangularity = polygon area / OBB area. Rectangularity ≥ 0.78, OBB ≤ 30 × 30 m, floors (from PLUTO `numfloors`, else `heightroof`/3.3 m) ≤ 6 → **low-rise, cookable**: bucket on **shape class × footprint-size bucket × height bucket** (~25–40 occupied buckets expected for a low-rise tile; the actual histogram is an acceptance artifact) and emit **ONE archetype directive per bucket** in the shipped `.asset.json` schema — **hand-tunable files** (the generator never clobbers an existing directive), with the existing `variants`/condition machinery reused sparingly for intra-bucket variety and the shipped per-instance `InstanceVariation` (mirror/yaw/setback/tint off `lot.seed`) providing the rest for free. Each of the ~1,000 placements maps its footprint to `(archetype id, position, rotation, fit-scale within a pinned [0.80, 1.25] tolerance, real Address/BIN as per-instance metadata)`. Cooked-asset count is bounded by buckets, never by buildings — ~30 cooked archetypes ≈ ~100 MB of atom payloads and minutes of cook, vs per-footprint cooking at ~1,000 ≈ 2–3.5 GB and an hour+ — this is what keeps the SDF/atlas/library budgets sane (§4.7). L/U/T classification against Forge's fixed-proportion shapes is M2 (most real L/U/T footprints won't match Forge's hard-coded proportions anyway — flagged below).

**High-rise archetype gap analysis — what Forge CANNOT generate today (honest list, code-verified against the building crate):**
- **No high-rise archetype at all.** Styles validated in production are houses/rowhouses/schools/police at 1–6 floors. `floors: u8` accepts 255, but no facade/roof grammar exists for a tower; nothing emits a structural core, curtain wall, or repeated-floor shaft.
- **No setbacks** — the 1916-zoning "wedding cake" massing that defines pre-war Manhattan towers cannot be expressed (one extrusion per footprint, full height).
- **No podium + tower**, no mechanical penthouse/bulkhead, **no rooftop water towers** (the iconic NYC roofscape element).
- **Footprints limited to the 4 fixed-proportion shapes** — no courtyard "O" tenement donuts, no acute-corner lots (the Flatiron is unrepresentable), no arbitrary polygons.
- **No party-wall street walls** — every asset is freestanding; a continuous rowhouse block face is N independent meshes.
- The consequence, stated plainly: **skyscraper and odd-footprint lots are placed as honest placeholders** (M1: the existing procedural fallback massing, flagged and counted in the generator output; the inspector still shows their real Address/BIN). A real `highrise` Forge archetype (base/shaft/crown massing with setback parameters + curtain-wall grammar + roof bulkhead) is the single biggest M2 Forge work item — and Forge has **no git**, so that work carries elevated risk and needs file-level backups before each edit session.

### 4.4 Terrain — DEM through the existing (dead) Forge loader into the existing (live) map pipeline

`nyc_fetch` crops the citywide EPSG:2263 GeoTIFF to the tile window, downsamples to the map cell grid, and writes a **16-bit grayscale PNG + JSON sidecar** `{min_m, max_m, cell_size_m, origin_2263_ft}`. The importer calls the existing `forge_terrain::dem_ingest::load_dem_png(path, resolution, amplitude) -> Result<Vec<f32>, String>` — giving the dead code its first real caller without editing Forge — with `amplitude = max_m − min_m`, then adds `min_m` back (the loader normalizes to `[0, amplitude]`; the sidecar restores the absolute NAVD88 datum). The result feeds the game's already-shipped real-world import: `Heightmap::from_data` → `import_real(RealImport { heightmap, osm, origin_lat/lon, sea_level, attribution, .. })` → `Map::save_to_dir` (`.meta.json` + `.map.json` + `.r32` f32-LE + `.layers.json`), discovered by `MapCatalog::scan(./civitas_data/maps)`. Water comes from the OSM snapshot so the shoreline masks buildability exactly as `derive_buildable` already does. Tile sizes: M1 ~1.5 km at 2 m cells ≈ 750² ≈ 0.56M terrain splats (trivially resident vs the 8M ceiling); Manhattan at 3 m ≈ 6.6M (borderline resident — streamed anyway, §4.5); citywide at 3 m ≈ 86M (partitioned only).

### 4.5 Streaming — one authority, fed by real tiles (M2)

Two shipped systems exist: game-side `map::streaming::MapPartition` (bake-time tiler, 512 m cells, used by `region_demo`) and engine-side `vox_render::world_partition::WorldPartition` (runtime: 3D `CellCoord`, `LoadRadius { inner 256 / outer 512 / hlod 400 }`, `SplatBudget { max_gpu_mb: 2048 }`, LRU eviction, `CellEventHandler`, `StreamEvent` — fully unit-tested, zero production callers). **Decision: `MapPartition` remains the bake-time tiler; `WorldPartition` becomes the single runtime authority and gets its first production caller in M2.** Per 512 m cell the bake writes: a terrain splat tile (`.vxm` via the existing tile path), the cell's `AtomInstance` list (48 B each), and HLOD (already covered by the instanced selector's I0/I1 imposters — `WorldCell::hlod_splats` stays empty for buildings, used only for terrain). Runtime each frame: `compute_streaming_state(camera_pos)` → async tile IO → `complete_load`; the loaded cells' instance lists concatenate into `InstancedSelector::set_instances` (instances are tiny — even citywide they are ~52 MB host-resident, so **only terrain splats stream; building instances merely cull**). `gpu::instancing::InstanceManager` is **not** used — it is the older Uuid-batch registry superseded by `atom_instances`; recorded as a deprecation decision, not silently ignored. Engine work here stays game-agnostic: cells, tiles, instances.

### 4.6 Sim-LOD tiers — from the deep-sim budget (1k) to NYC (1.08M)

The deep-sim design's budget is ≤ 10 ms/tick at 1,000 instances. NYC is 46k (Manhattan) to 1.08M (city) — a 46–1,080× gap that no constant-factor work closes. **Tiering, with replay determinism preserved by construction:**
- **T0 — full per-instance deep-sim** for an *attention set* near the interest set, **normative target ~50k instances (§2.0)**. Membership is **sim-deterministic** (districts containing player actions in the last K ticks, plus the inspected instance's district) — never camera-derived, because camera state is not in the action log and replay must be exact. Honesty note: the deep-sim budget (≤ 10 ms/tick) was measured at 1k instances; reaching ~50k full-fidelity requires the M2 tick-index fix below plus measured per-stage scaling — the gate is the printed `tick_ms` at the printed T0 count, never an assumed constant.
- **T1 — district aggregates** (Manhattan: 12 community districts; citywide: 59): occupancy, value, abandonment as per-district rates updated from aggregate demand exactly once per tick. City totals (taxes, demand bars, care) are **always computed from T1 aggregates**, so promotion/demotion cannot change city totals — that invariant is what makes tiering replay-safe.
- **T2 — borough/city statistics**: the existing aggregate demand/finance/goods chain, unchanged.
- Per-instance detail for a T1 instance (inspector click) is derived on demand as a pure function of `(instance seed, district aggregate, tick)` — same trick as `PlannedLot::occupancy`, no stored state, no replay hazard.
- **Prerequisite fix:** `BuildingRegistry::tick`'s per-instance `find()` over `BuildingManager.buildings` is O(n²); M2 must index `engine_building → buildings` index once per tick (O(n)). At 1k instances this is invisible; at 46k it is ~2×10⁹ comparisons per tick.

### 4.7 Memory and perf math at borough scale — today's real numbers, extrapolated honestly

Measured today (sources: virtualized M1/M2 plan steps 5, SDF M1 plan, `assets/buildings/forge_starter` on disk):
- Cooked pack today: 9 assets; atoms payloads 24 MB on disk; **SDF payloads ≈ 75 MB at the current cook settings — scaling with *unique assets*, never with instances**. GPU SDF atlas: 64 MB quota, **200 assets ≈ 36 MB measured**.
- Splat library: 5 synthetic assets → 25,325 library atoms, `resident_bytes = 1,796,956` (~1.8 MB) → ~70 KB/asset at this fidelity; 200 unique assets ≈ 1M library atoms ≈ 64 MB GPU (packed 64 B/atom).
- Render chain at budget 1M: 80 MB splats + 32 MB transforms + 6 × 64 MB entry-class buffers ≈ **496 MB GPU, proven on the 780M**; `select+expand+render p50 = 72.46 ms` at 10k visible instances (select 19.56 ms of it) — the 60 fps gap is the virtualized M3 plan's problem and is not re-promised here.
- Instances: `AtomInstance` is 48 B → Manhattan 46k ≈ 2.2 MB, citywide 1.08M ≈ 52 MB (host, always resident). `BuildingInstance` ≈ 140 B → 46k ≈ 6.4 MB, 1.08M ≈ 151 MB host — acceptable RAM, but see the O(n²) tick fix above.
- Terrain: M1 tile 0.56M splats resident; Manhattan ≈ 6.6M at 3 m streamed under the 2,048 MB `SplatBudget` (80 B/splat → full borough would be 528 MB if resident; streaming keeps the live set at the load-radius ring, ~3–5 cells ≈ tens of MB).
- Cooked atom payloads measure **~1.5–3.5 MB each** on disk (`forge_starter/atoms/*.atoms.json`, 2026-06-10 cook; variants are full payloads too). The instance-not-asset arithmetic at M1 scale: **~30 archetypes ≈ ~100 MB cooked / minutes of cook**, vs per-footprint cooking at ~1,000 ≈ **2–3.5 GB / an hour+** — and the latter also breaks every dedup budget below.
- **The binding constraint is unique-asset count, not building count**: SDF (~0.4 MB/asset payload + atlas share — the SDF design's own counterfactual: per-instance cooking = 1.8 GB, "impossible"), splat library (~70 KB+/asset), and textures all scale with buckets. The bucketing layer must keep any resident set ≤ ~200 unique assets — which the district-themed bucket design satisfies by construction (~25–40 buckets per neighborhood theme; a handful of themes resident at once). At the normative 1M instances (§2.0) the instance side stays trivial (48 B × 1M ≈ 48 MB host) precisely because instances never multiply cooked assets.

### 4.8 Real identity — placements and the inspector

The generator emits `placements.json` per tile: one record per footprint `{bin, bbl, address, zone, center_m: [x,z], yaw, width, depth, floors, height_m, fit_scale: [sx, sz], asset_id: Option<bucket>, placeholder: Option<reason>}` — **every footprint becomes an instance of a shared archetype, and identity (Address/BIN) is per-instance metadata; the archetype carries none of it.** Placement converts records to the game's own `LotPlan/PlannedLot` (pub fields, code-verified) with `asset_id` pinned and `seed = BIN` (deterministic — and the seed the shipped `InstanceVariation` derives mirror/yaw-jitter/setback/tint from, so intra-bucket variety is free), develops them immediately through `BuildingRegistry::develop` (verified signature), and keeps an `InstanceId → RealBuildingInfo` side-table for the inspector — instance/registry types untouched. `fit_scale` (per-axis footprint-OBB / archetype dims, pinned to [0.80, 1.25] at generation) is recorded for every placement but **not applied at render time in M1** (render_gpu is locked; `InstanceFrame.scale` is the one-line application point when its owner unlocks it) — the tolerance bounds the visible mismatch at ≤ 25%. The inspector appends `Address` / `BIN` / `latlon` lines **from the side-table, never from the asset**. Save/replay of real-city games is M2 (a new authored action that replays the placement deterministically from the tile id); M1 loads fresh each run and says so.

---

## 5. Data Models

```rust
/// urban_horizon/src/realcity/mod.rs — all game-layer. Private fields, accessors.

/// One dataset item in the pinned snapshot manifest.
pub struct SnapshotItem {
    url: String,        // exact request URL (bbox-sliced where applicable)
    file: PathBuf,      // under assets/source/nyc/raw/
    sha256: String,     // pinned after first fetch; later fetches must match
    license: String,    // human-readable terms line (NYC Open Data / ODbL / DCP)
    fetched_at: String, // RFC3339; informational only, not part of determinism
}

/// A real building placement, one per footprint in the tile.
pub struct RealPlacement {
    bin: u64,                 // Building Identification Number (leading digit = borough)
    bbl: u64,                 // Borough-Block-Lot key into PLUTO
    address: String,          // PLUTO address ("96 SPRING ST"), empty if absent
    zone: ZoneType,           // mapped from PLUTO landuse (see plan's mapping table)
    center_m: [f32; 2],       // tile-local metres (EPSG:2263-derived frame, §4.1)
    yaw: f32,                 // OBB orientation, radians
    width_m: f32, depth_m: f32,
    floors: u8,               // numfloors, else heightroof / 3.3 m, clamped 1..=255
    height_m: f32,            // heightroof × 0.3048…, 0.0 if absent
    fit_scale: [f32; 2],      // per-axis footprint-OBB / archetype dims, pinned to
                              // [0.80, 1.25] at generation; recorded, not yet rendered (M1)
    asset_id: Option<String>, // shared archetype (bucket) directive id; None => placeholder
    placeholder: Option<PlaceholderReason>, // Tall | OddFootprint | OddFit | MissingData
}

/// Tile = origin pin + placements; serialized as placements.json next to the map.
pub struct RealCityTile {
    tile_id: String,           // "nyc_lower_manhattan_m1"
    origin_lat: f64, origin_lon: f64, // world (0,0) in WGS84, for the inverse print
    origin_2263_ft: [f64; 2],  // same pin in EPSG:2263 ftUS (DEM/footprint frame)
    placements: Vec<RealPlacement>,
}

/// Inspector side-table payload (held by GameView, never serialized in M1).
pub struct RealBuildingInfo { bin: u64, bbl: u64, address: String, lat: f64, lon: f64 }
```

---

## 6. API

```rust
// urban_horizon/src/realcity/mod.rs
/// EPSG:2263 (LCC 2SP, NAD83/Long Island, ftUS) from WGS84. Pure, deterministic.
/// Correctness is gated by the snapshot's own area attribute (§3), not by trust.
pub fn epsg2263_from_lat_lon(lat: f64, lon: f64) -> [f64; 2];
/// Inverse of the tile's local frame back to WGS84 (for the inspector print).
pub fn lat_lon_from_local(tile: &RealCityTile, x_m: f32, z_m: f32) -> (f64, f64);
/// OBB fit (rotating calipers) over a closed polygon ring in local metres.
pub fn fit_obb(ring: &[[f64; 2]]) -> ObbFit; // ObbFit { center, width, depth, yaw, rectangularity }
/// Load placements.json. Errors are strings (house style in the game bins).
pub fn load_tile(path: &Path) -> Result<RealCityTile, String>;
/// Convert placements to the game's lot plans (one LotPlan per placement,
/// asset pinned, seed = BIN). Pure; does not touch CivitasGame.
pub fn lot_plans(tile: &RealCityTile) -> Vec<crate::lot::layout::LotPlan>;

// urban_horizon/src/game/mod.rs  (M1 — the one gated game-core addition)
/// Push the plans and develop every lot NOW (no demand gate), fast-forwarding
/// construction by running the registry's own tick loop; returns instance ids
/// in placement order so the caller can build the RealBuildingInfo side-table.
pub fn place_real_city(&mut self, plans: Vec<crate::lot::layout::LotPlan>) -> Vec<crate::instance::InstanceId>;

// forge (existing, unmodified — first real caller):
// forge_terrain::dem_ingest::load_dem_png(path: &std::path::Path, resolution: u32, amplitude: f32) -> Result<Vec<f32>, String>
// civitas (existing, unmodified):
// map::import::import_real(input: RealImport) -> Result<Map, String>
// vox_data::osm_import::parse_osm_json(json: &str) -> Result<OsmData, String>  // simplified {nodes, ways} schema
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `nyc_fetch` snapshot | operator command | `urban_horizon/src/bin/nyc_fetch.rs` | curl house pattern; idempotent; manifest-pinned |
| `epsg2263_from_lat_lon` / `fit_obb` | directive generator | `urban_horizon/src/bin/nyc_directives.rs` | snapshot → directives + placements.json |
| `load_dem_png` (forge, dead today) | tile importer | `urban_horizon/src/bin/nyc_tile_import.rs` | first caller; sidecar restores datum |
| `import_real` / `Map::save_to_dir` | tile importer | same | map appears in `./civitas_data/maps` |
| game cook | operator command | `game_asset_cook` (no edits; directives only) | discovers `*.asset.json` recursively |
| `place_real_city` | `MenuAction::StartGame` arm, after `game.bind_map(&map)` | `urban_horizon/src/bin/play.rs` | only when `<map dir>/placements.json` exists |
| `RealBuildingInfo` lines | `GameView::inspector` | `urban_horizon/src/bin/play.rs` | appended when the side-table has the id |
| `WorldPartition` (M2) | game streaming driver | engine `vox_render/src/world_partition.rs` + game glue | first production caller; engine stays generic |
| Sim tiers (M2) | `CivitasGame::tick` deep-sim pass | `urban_horizon/src/game/mod.rs` | T1 aggregates authoritative for totals |

---

## 8. Open Questions

- [ ] **M2:** high-rise Forge archetype scope (setback massing + curtain grammar + roof bulkheads) — sized after M1 ships; Forge has no git, so the work plan must include file backups.
- [ ] **M2:** real-city save/replay — new authored action replaying `place_real_city(tile_id)` deterministically; interacts with `SAVE_VERSION`.
- [ ] **M2:** LION vs OSM as the road-graph source (LION has address ranges; OSM has better connectivity tags) — decide with data in hand.
- [ ] **M3:** whether citywide 1.08M `BuildingInstance` rows stay a flat Vec (151 MB, fine) or move behind the district tier entirely; decided by the M2 tick-index fix's measured headroom.

(M1 has no open questions — the plan documents every chosen answer.)

---

## 9. Out of Scope

- Planet-scale / floating-origin coordinates (compatibility note in §4.1 only; ≤ 50 km needs none of it).
- The 60 fps render gate (virtualized M3 plan owns it) and any GPU selector port.
- Real-time data feeds (311, traffic), photo-textures, interiors, subway/bridges/elevated infrastructure as geometry.
- High-rise Forge archetype implementation (gap analysis here; build in M2+).
- Editing `vox_render`, `render_gpu`, or Forge in M1 (see the plan's gates; M1 is ingestion + game-layer placement only).

---

## 10. Related Plans / Designs

- Executes first: `[World-Scale NYC M1 Plan](../plans/2026-06-10-world-scale-nyc-m1.md)` (the M1 rung only).
- Depends on: virtualized splat rendering M1+M2 (landed today), living building instances Tasks 1–5 (landed today), SDF pillar M1 (in flight — gates documented in the plan).
- Frame-gate machinery for §2.0's targets: the M3.2 re-baseline plan `[Virtualized Splat Rendering M3.2](../plans/2026-06-11-virtualized-splat-rendering-m3-2.md)` (gate tiers, `tomespensin` 16.6 ms manual procedure, 780M 33 ms tracking floor).
- Informs: deep-simulation plan (sim tiers), any future Forge high-rise archetype plan.
- Dataset sources (verified 2026-06-10): NYC Building Footprints `5zhs-2jue`; 1-ft DEM `dpc8-z3jc` / Integer Raster `7kuu-zah7`; PLUTO/MapPLUTO (DCP Bytes of the Big Apple); LION `2v4z-66xt`; NYC Open Data FAQ (no restrictions); OSM ODbL.
