---
name: forge-assets
description: Create or author assets and environments with Forge — buildings, terrain, vegetation, water, roads, plots, and full maps. Use when generating geometry+materials for the engine/game, authoring a BuildingDescription/blueprint, or assembling a map's content layer. Forge is a node-DAG asset factory (not a box generator); drive the DIRECTIVE path, never bare params.
---

# Create assets & environments with Forge

Forge (`~/src/forge`, workspace of `crates/*`) is the asset factory: geometry + materials. **Full verified inventory: `~/src/forge/CLAUDE.md`** — read it for exact signatures/file:lines. This skill is the workflow.

## The cardinal rule
**Drive the directive/asset path; never bare params.** A `BuildingDescription` → `generate_asset` carries materials, glass, weathering, zones. Bare `BuildingParams` → `generate` returns mesh-only (flat, no glass) — the recurring failure mode.

## Two authoring surfaces (buildings)
- **Flat directive (default):** `forge_building::generate_asset(BuildingDescription) -> ForgeAsset`. Returns `mesh` + `material_zones` (incl. transmissive glass) + `sockets` + per-vertex `weathering_masks` + UVs + tags. This is what the offline cook + `forge-cli building` use.
- **Node-DAG (hero/curved/multi-volume):** `forge_building::blueprint::evaluate(BlueprintGraph) -> ForgeAsset`. Compositional op-tree (footprint_curve, extrude, cylinder, sweep_profile, boolean union, setback, facade_assign, crown_kit, place_volume/part, asset_output). `boolean` only `union` works — compose voids as multi-volume union, never CSG subtract.
- **AVOID:** bare `generate(BuildingParams) -> Mesh` (no zones, no glass spec).

## CLI
- `forge run '{"command":"building","params":{...,"description":{<BuildingDescription>}}}'` — **`description` present → `generate_asset`** (full contract under `asset`); absent → bare mesh. Passing `description` is what unlocks glass/massing/zones.
- `forge blueprint <graph.json|->` → `blueprint::evaluate`.
- `forge catalog [--json]` → live node/socket/enum registry (author graphs against this, don't guess enums).

## BuildingDescription quality levers
- `facade_system`: `punched_window` (solid wall + apertures) | `curtain_wall` (mullion grid + **transmissive vision glass**).
- `massing_mode`: `box` | `podium_tower` (podium_floors+tower_floors==floors) | `setback` (ziggurat).
- `footprint.shape`: `Rectangular|LShaped|UShaped|TShaped`. `roof.style`: `Flat|Gabled|Hip|Mansard`.
- `condition`: `New|Aged|Weathered|Derelict` (drives weathering). `spec.siding`: `Clapboard|Brick|Stone|Stucco|Vinyl|TimberFrame`.
- `window_density`: **0.75–0.95** for glass families. `seed`: pass the object/lot id for deterministic, reproducible output.

## Glass & materials
- Glass = `material_id 2` / `MaterialSpec.base_color[3] < 0.5` (alpha = transmission ≈ 0.15). **Detect glass by `base_color[3] < 0.5`, not a hardcoded id.**
- `MaterialSpec` carries `base_color/normal/roughness/displacement` maps as `polyhaven://...` URIs — consume these; never re-derive flat RGB.

## Environments / maps (the world dressing — most NOT linked into the game yet)
- `forge_terrain::generate(TerrainParams{biomes:true, erosion:Some(..)})` — fBm heights + **7-biome classify + 4-ch splat weights** + a real physical-erosion suite (hydraulic/thermal/aeolian/SWE/glacial/coastal). Set `biomes=true` or splat weights stay zero. Output `TerrainGrid` (row-major `z*resolution+x`, **center-origin** `±world_size/2`).
- `forge_scatter::scatter(&TerrainGrid, ScatterParams)` — Bridson Poisson-disc instance transforms, biome/slope/altitude-gated. The **placement** algorithm; `asset_id` points at the proto to instance. (`ScatterInstances.transforms` are **column-major** glam — transpose to row-major for the GPU instancer.)
- `forge_water::{generate_ocean,generate_river}` (Gerstner, `time`-animated), `forge_road::generate`/`build_network` (crowned ribbons; curbs/intersections stubbed), `forge_plot::generate` (lot ground + fence/garden `asset_query` hooks). All output `forge_mesh::Mesh` (Y-up RH, **CW winding**, `indices: Vec<[u32;3]>`).

## Game content policy (Urban Horizon)
- **PolyHaven for vegetation + ground models + textures** — Forge's procedural L-system flora is NOT AAA-SOTA. Import PolyHaven GLBs via the game's `import_public_pack` (glTF → `ReadyAssetPayload`); scatter them by biome.
- **Forge for** terrain topography (erosion+biomes), buildings (`generate_asset` directives), water, roads, plots, **and `forge_scatter` as the placement algorithm** (its `asset_id` → a PolyHaven proto).
- The offline cook `urban_horizon/src/bin/game_asset_cook.rs` (`forge_description`) is the proven recipe; 79 authored `*.asset.json` directives live under `urban_horizon/assets/source/buildings/`.

## Honesty
Validate content on a **hero-camera rendered frame** (see `render-spectra-cuda` / `render-spectra-vulkan`), never on box-stub scenes or a passing unit test. A generator that returns a mesh is not the same as one that renders beautifully.
