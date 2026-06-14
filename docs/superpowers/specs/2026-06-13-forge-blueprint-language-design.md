# Design: Forge Blueprint Language — a typed NODE-GRAPH building grammar (Geometry-Nodes-for-buildings) (2026-06-13)

**Status:** Draft
**Scope:** Replace Forge's flat parametric building directive with a **typed node DAG** — Geometry-Nodes-for-buildings — that an LLM authors as JSON by querying a live introspection catalog, and that a Forge `blueprint` evaluator executes deterministically into the existing `ForgeAsset` payload (mesh + sockets + material zones + clusters). Curves are first-class (Bézier/spline footprints + a net-new sweep-profile-along-curve mesh op → CRISP swept geometry, not an SDF blob). Affects: forge `crates/building/src/{lib,description,massing,footprint,facade/*,roof,seal,winding,socket,material,asset}.rs`, **new** `crates/building/src/blueprint/` (graph, catalog, eval) and **new** `crates/mesh/src/sweep.rs`; forge-cli; civitas `src/asset/directive.rs` + `src/bin/game_asset_cook.rs`.
**Related:** memory `llm-blueprint-forge-directive` (EVOLUTION 2026-06-13, the node-graph steer — authoritative), `forge-directive-factory`, `building-content-pipeline`, `articulated-parts-budget`, `hybrid-atom-sdf-lod-direction`, `civitas-usd-interchange-decision`; Spectra `slang/sdf_eval.slang`; supersedes the first-cut "header + 3 sub-trees" revision of this same doc.

---

## 1. Problem Statement

- Today's directive (`civitas src/asset/directive.rs`, `forge description.rs:52,69,88`) is a **flat parameter set**: one `FootprintShape` from a closed 4-value enum (`Rectangular|LShaped|UShaped|TShaped`, `forge description.rs:52`), one `MassingMode` from a closed 3-value enum (`Box|PodiumTower|Setback`, `description.rs:88`), one `FacadeSystem` per band (`PunchedWindow|CurtainWall`, `description.rs:69`). An LLM varies knobs *within* a family but cannot describe an arbitrary bespoke form: "an L-shaped podium with a curved-glass entrance wing, a cylindrical tower notched by a light-well, and a stepped crown with an antenna mast." The grammar is fixed knobs, not a composable language.
- **Curves are unrepresentable as crisp mesh.** Massing is a stack of straight-edged inset prisms (`forge massing.rs:24 MassingVolume`, polygon loops only). There is **no sweep, loft, spline, or Bézier anything in forge** — a whole-repo grep for `fn .*(sweep|loft|spline|bezier)` returns only `gwn_sweep_512` (a GWN *probe* sweep, unrelated; `lib.rs:629`), and `forge-mesh` exposes only `merge / transform / flip_normals / apply_box_projection / weld_vertices` (`crates/mesh/src/ops.rs`) plus `boolean_union/boolean_subtract` (`crates/mesh/src/boolean.rs`). A curved atrium or swooping entrance wing has **no crisp authoring path** today. The prior cut of this design wrongly punted curves to SDF surface-extraction, which smears the crisp box-UV facade rhythm the renderer needs.
- The first-cut "header + 3 sub-trees" model is **not how a human designs** and not how the LLM should compose. It hard-codes three top-level slots (massing / skin / crown) and cannot express reuse (a stair-tower sub-graph used twice), data-flow (a footprint curve feeding both an extrude AND a sweep), or coordinate-free composition (snap the tower to the podium roof without typing y=10.8).
- Articulated parts (a hospital ambulance-bay door) and named placement have no first-class wiring: `socket.rs:9 build_residential_sockets` emits a fixed residential set; there is no graph-level way to expose a named anchor (a podium roof's `tower_base`) and snap a downstream node's socket onto it.
- The watertight oracle (`forge lib.rs:738 assert_watertight_body` → `boundary_edge_stats` non-manifold gate + `gwn_sweep_512` < 5% fractional + `cook_probe_gate`, `lib.rs:555,629,689`) is enforced **after** generation on a fixed pipeline. A free node-graph must make a broken body **unrepresentable**, not merely rejected.
- The LLM has **no way to know the real options**. Wall types, materials, crown kits, profiles live in Rust enums (`description.rs`); a prompt that lists them by hand goes stale the moment a variant is added, inviting hallucinated options.

---

## 2. Done When

Running

```bash
forge-cli run "$(cat assets/blueprints/civic.hospital_demo.bp.json)" --emit asset \
  && cargo run -p urban_horizon --bin game_asset_cook -- --only civic.hospital_demo --sdf-strict
```

produces, on stdout, the lines:

```
[graph] civic.hospital_demo: 14 nodes, 19 typed edges -> type-check OK (0 socket-type mismatches)
[graph] curve: entrance_wing sweep-profile-along-curve -> 1 crisp swept component (CCW spline, 96 segments, watertight cap+endcaps)
[graph] anchors: tower.base snapped to podium.roof_anchor 'tower_base' at y=10.80 (no literal coord authored)
[blueprint] civic.hospital_demo: 3 volume nodes, 1 void-subtract, 1 swept wing, 41 sub-meshes -> watertight OK (0 non-manifold edges, 0.0% fractional, interior 512/512)
[blueprint] articulated: 1 part 'ambulance_bay_door' pivot=(30,0,0) axis=Y -> emitted as socket+sub-mesh (not fused)
[sdf-validate] civic.hospital_demo: sign=gwn inside_negative=1.000 eikonal_p95<0.10
[cook] wrote atoms/civic.hospital_demo.atoms.json (NNNNN atoms, sdf sign=Closed)
```

A human reads "type-check OK", "1 crisp swept component" for the curved wing (NOT an "sdf-derived" warning), an anchor snapped without a literal coordinate, and "watertight OK / sign=Closed" — proof the language is a typed graph, curves are crisp mesh, anchors compose coordinate-free, and the seal held through free composition.

A second command proves introspection:

```bash
forge-cli catalog --json | jq '.node_types[] | select(.name=="facade_assign") | .sockets'
```

prints the live typed sockets of the `facade_assign` node (a `Geometry` input, an `Enum<FacadeSystem>` with the two real variants `punched_window`/`curtain_wall`, a `Material` input, …) — sourced from the Rust registry, so adding a `FacadeSystem` variant changes this output with no doc edit.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Graph deserializes & type-checks | `assert_eq!(graph.type_check().unwrap().edge_count, 19)` on the hospital graph; an edge wiring a `Curve` output into a `Geometry` input FAILS with `ForgeError::SocketTypeMismatch{from:Curve,to:Geometry}`; an unknown node `op` FAILS deserialization (`#[serde(deny_unknown_fields)]`) | `assert!(serde_json::from_str::<BlueprintGraph>(j).is_ok())` — passes on `{"nodes":[],"edges":[]}` |
| Catalog reflects live registry | `forge_catalog().node_type("facade_assign").input("system").enum_options()` equals the live `FacadeSystem` variant set `["punched_window","curtain_wall"]`; after adding a 3rd `FacadeSystem` variant the same call returns 3 with NO change to the catalog code | `assert!(!forge_catalog().node_types.is_empty())` |
| Sweep-profile-along-curve = CRISP mesh | sweeping a 5-vert wall profile along a 96-segment Bézier arc yields a closed mesh with `assert_eq!(connected_components(&m), 1)`, `boundary_edge_stats(&m).non_manifold == 0`, and `assert!(max_face_normal_deviation_from_swept_frame(&m) < 1e-3)` (faces follow the frame, not a marched iso-surface) | `assert!(swept.indices.len() > 0)` — passes on any extrude |
| Curved footprint produces crisp wall, NOT SDF | the entrance-wing node lowers via `sweep_profile_along_curve` and the eval log line contains `"crisp swept component"` and does NOT contain `"sdf-derived"`: `assert!(log.contains("crisp swept") && !log.contains("sdf-derived"))` | `assert!(asset.mesh.aabb().1[0] > 0.0)` |
| Named anchor snapping (coordinate-free) | the `extrude` podium node exposes anchor `tower_base` at the roof-cap centroid; the tower node's `base` socket is wired to that anchor and the evaluated tower's `y_base` equals the podium roof height within 1e-4 with NO literal y in the JSON: `assert!((tower_volume.y_base() - podium_roof_y).abs() < 1e-4)` and `assert!(!json.contains("\"translate_y\""))` | `assert_eq!(graph.anchors.len(), 1)` |
| Node-group reuse | a `stair_tower` node-group instanced twice yields two disjoint coherent components with identical local geometry hashes: `assert_eq!(blake3(comp_a_local), blake3(comp_b_local)) && components_disjoint(a,b)` | `assert!(graph.groups.contains_key("stair_tower"))` |
| Per-volume facade assignment | a graph assigning `curtain_wall` to the tower volume and `punched_window` to the podium yields ≥2 distinct `MaterialZone.material.base_color`, the glass zone tagged `MAT_GLASS=3`: `assert!(zones.iter().any(|z| z.material.is_glass()) && zones.iter().map(...).distinct().count() >= 2)` | `assert!(!zones.is_empty())` |
| Crown kit (parapet + penthouse + antenna) | the crown node emits ≥1 closed box above the top volume's wall plane AND a capsule mast: `assert!(mesh.aabb().1[1] > top_volume_height + antenna_height - 0.01)` | `assert!(crown_meshes.len() > 0)` |
| Articulated part as separate sub-mesh + socket | a `place_part` door node yields a `Socket{socket_type: Door, ...}` whose triangle range is NOT welded into the shell's coherent component: `assert!(part_component_disjoint_from_shell(&asset))` and `assert_eq!(asset.sockets.iter().filter(|s| s.socket_type==SocketType::Door).count(), 1)` | `assert!(asset.sockets.len() > 9)` |
| Watertight invariant under composition | `assert_watertight_body(&asset, "hospital")` returns without panic: 0 non-manifold edges, `gwn_sweep_512` fractional < 5%, interior ≥ 30, `cook_probe_gate` `Ok` (`lib.rs:738`) | `assert!(asset.mesh.indices.len() % 3 == 0)` |
| Today's directive is a graph preset | the live `rowhouse_01` directive lowered to a `BlueprintGraph` preset then evaluated produces a mesh byte-identical to today's `generate(params)` path: `assert_eq!(blake3(new_mesh.positions), blake3(legacy_mesh.positions))` | `assert!(legacy_directive_still_parses())` |

---

## 4. Architecture

### 4.1 Layering — where the language lives

The blueprint language is **ENGINE-level** (Forge), game-agnostic per CLAUDE.md: nodes know footprints, curves, volumes, facades, roofs, materials, anchors, parts — never "hospital" or "zone." A new module tree `crates/building/src/blueprint/{graph,catalog,eval,nodes}.rs` deserializes the node graph and lowers it to the existing extruder pipeline. The mesh-level sweep op is even more game-agnostic and lands in `crates/mesh/src/sweep.rs` (reusable by road/water/vegetation directives later — `forge-directive-factory`). The civitas GAME layer keeps `AssetDirective` (gameplay) and gains a thin lowering: directive → graph preset (§8). Forge stays the deterministic builder; the LLM (or a preset) is the designer.

### 4.2 The model: a typed node DAG (Geometry Nodes for buildings)

A `BlueprintGraph` is a directed acyclic graph of **typed nodes** connected by **typed edges**. This subsumes the old header+3-subtrees model: massing, skin, and crown are no longer fixed top-level slots but ordinary nodes wired by data-flow. The graph evaluates in topological order; each node consumes typed input sockets and produces typed output sockets.

**Socket types** (the wire type system; an edge is legal only if `from.ty == to.ty`, or via an explicit declared coercion):

| Type | Carries | Example producer → consumer |
|---|---|---|
| `Curve` | an ordered spline (polyline / Bézier control net) in XZ, open or closed, with per-vertex tangents | `footprint_curve` → `extrude` / `sweep_profile` |
| `Geometry` | a watertight mesh fragment + its named anchors + its material zones | `extrude` → `facade_assign` → `union` |
| `Profile` | a small open/closed 2D section (a wall cross-section, a cornice profile) | `profile_lib` → `sweep_profile` |
| `Anchor` | a named point+frame on a Geometry (position, normal, tangent) | `extrude.roof_anchor` → `place_volume.base` |
| `Float` / `Vector` / `Int` | scalars / 3-vectors / counts | literals or `value` nodes → any param socket |
| `Enum<T>` | one variant of a registry enum (`FacadeSystem`, `RoofStyle`, `Style`, `CrownKitKind`, `ProfileKind`) | literal → `facade_assign.system` |
| `Material` | a key into the graph's material table, resolved to `MaterialSpec` (`asset.rs:59`) | `material` node → `facade_assign.material` |

**Node types** (the registry; each declares its typed input/output sockets and its watertight invariant — see §4.6):

| Node | Inputs (typed) | Output | Lowers to |
|---|---|---|---|
| `footprint_curve` | control verts `[ [x,z], … ]`, `closed: bool`, `bezier: bool` | `Curve` | net-new spline eval (`blueprint::curve`) |
| `prism` / `extrude` | `Curve` (closed), `floors: Int`, `base: Anchor?` | `Geometry` (+ anchors: `roof`, side-face bands) | `MassingVolume` → boundary-rep extruders (`massing.rs`, `roof::flat_roof_over_polygon` `roof.rs:73`) |
| `sweep_profile` | `Curve` (path), `Profile`, `caps: bool` | `Geometry` | **net-new** `forge_mesh::sweep_profile_along_curve` (`crates/mesh/src/sweep.rs`) |
| `cylinder` | `radius: Float`, `sides: Int`, `floors: Int` | `Geometry` | N-gon `Curve` → `extrude` |
| `boolean` | `a: Geometry`, `b: Geometry`, `op: Enum<{union,subtract,intersect,smooth_union,smooth_subtract}>`, `k: Float?`, `void: bool` | `Geometry` | separable → ring footprint / abutting solids; blends → SDF extraction (§4.4) |
| `setback` | `a: Geometry`, `at_floor: Int`, `ratio: Float` | `Geometry` | `massing::massing_volumes` setback band (`massing.rs:82`) |
| `facade_assign` | `Geometry`, `system: Enum<FacadeSystem>`, `floors: [Int;2]?`, `faces: [Enum<Compass>]?`, `bay_width: Float`, `windows_per_bay: Int`, `window_density: Float`, `material: Material`, `glass_material: Material?` | `Geometry` (clad) | `facade::build_facade_walls` (`facade/mod.rs:27`) — **assignment, not CSG** |
| `crown_kit` | `Geometry`, `kit: Enum<CrownKitKind>`, kit params, `material: Material?` | `Geometry` | parapet/cornice (`facade/cornice.rs`) + new penthouse/mast solids |
| `place_volume` | `Geometry`, `base: Anchor` | `Geometry` (positioned) | translate volume so its base frame coincides with the anchor (coordinate-free, §4.5) |
| `place_part` | `Geometry`, `anchor: Anchor`, `kind: Enum<PartKind>`, `size: Vector`, `pivot_axis: Enum<Axis>`, `material: Material` | `Geometry` (+ a `Socket` + a `GeometryCluster`) | `facade::door_leaf::generate_door_leaf` (`door_leaf.rs:136`) as a disjoint component + `Socket` (`asset.rs:42`) |
| `scatter` / `instance_on_anchors` | `Geometry`, anchor-set, instance `Geometry` | `Geometry` | instance placement on named anchors (parts ride mover/refit path, `articulated-parts-budget`) |
| `material` | `base_color`, `roughness`, `metallic`, `glass: bool`, … | `Material` | `MaterialSpec` (`asset.rs:59`) |
| `asset_output` | `Geometry` (the final body) | — (terminal) | `generate_asset` tail: `winding::orient_outward` (`winding.rs:134`), `material::material_zones_for_mesh` (`material.rs:142`), sockets, bounds, clusters, `assert_watertight_body` (`lib.rs:738`) |

**Node-groups** are reusable typed sub-graphs: a `node_group` definition declares its own external input/output sockets (typed exactly like a node), and a `group_instance` node references it by name and wires its exposed sockets. This is how a `stair_tower` or a `bay_window_unit` is authored once and instanced N times — the Geometry-Nodes "group" concept, type-checked identically to a primitive node.

### 4.3 The JSON graph shape

A graph is a flat node list + a typed edge list (Geometry-Nodes' serialization, not a nested tree — DAGs aren't trees, and a tree can't express a footprint curve feeding two consumers):

```json
{
  "schema": 2,
  "id": "civic.hospital_demo",
  "seed": 7,
  "floor_height": 3.6,
  "style": "modern",
  "materials": { "stone": {...}, "vision_glass": {...}, "spandrel": {...}, "steel": {...} },
  "groups": { "stair_tower": { "inputs": [...], "outputs": [...], "nodes": [...], "edges": [...] } },
  "nodes": [
    { "id": "podium_fp", "op": "footprint_curve",
      "params": { "verts": [[0,0],[60,0],[60,40],[0,40]], "closed": true, "bezier": false } },
    { "id": "podium",    "op": "extrude", "params": { "floors": 3 } },
    { "id": "lightwell_fp", "op": "footprint_curve",
      "params": { "verts": [[26,16],[34,16],[34,24],[26,24]], "closed": true } },
    { "id": "lightwell", "op": "extrude", "params": { "floors": 3 } },
    { "id": "podium_court", "op": "boolean", "params": { "op": "subtract", "void": true } },
    { "id": "tower",     "op": "extrude", "params": { "floors": 9 } },
    { "id": "tower_fp",  "op": "footprint_curve", "params": { "verts": [[18,8],[42,8],[42,32],[18,32]], "closed": true } },
    { "id": "place_tower", "op": "place_volume" },
    { "id": "body",      "op": "boolean", "params": { "op": "union" } }
  ],
  "edges": [
    { "from": "podium_fp:curve",   "to": "podium:curve",        "ty": "Curve" },
    { "from": "lightwell_fp:curve","to": "lightwell:curve",     "ty": "Curve" },
    { "from": "podium:geom",       "to": "podium_court:a",      "ty": "Geometry" },
    { "from": "lightwell:geom",    "to": "podium_court:b",      "ty": "Geometry" },
    { "from": "tower_fp:curve",    "to": "tower:curve",         "ty": "Curve" },
    { "from": "tower:geom",        "to": "place_tower:geom",    "ty": "Geometry" },
    { "from": "podium:roof_anchor","to": "place_tower:base",    "ty": "Anchor" },
    { "from": "podium_court:geom", "to": "body:a",              "ty": "Geometry" },
    { "from": "place_tower:geom",  "to": "body:b",              "ty": "Geometry" }
  ]
}
```

An edge endpoint is `"node_id:socket_name"`; `ty` is redundant with the registry but stored so the validator (and the LLM) can read the wire type without resolving the node. `deny_unknown_fields` + the typed-edge check reject malformed graphs before any geometry runs.

### 4.4 Curves are first-class — the headline fix

The renderer renders **MESH** (`hybrid-atom-sdf-lod-direction`); a curved wall must therefore be **crisp swept mesh**, not a marched SDF iso-surface that smears box-UV facade rhythm. Modeled exactly on Blender's **Curve-to-Mesh**:

- A `footprint_curve` node emits a `Curve`: either a polyline or a **Bézier** control net, `closed` or open. Bézier segments are tessellated to a polyline at a seed-stable resolution (segments-per-unit-arc, pinned to the curve's bounding box so re-cooks are byte-identical). The result is an ordered list of points **with per-vertex tangent frames** (the frame is what the sweep orients against).
- A `sweep_profile` node takes a `Curve` (the path) and a `Profile` (the wall cross-section — a thin closed rectangle for a solid wall, or a section with a window reveal) and emits crisp swept `Geometry`: for each path segment it transports the profile along the **rotation-minimizing (parallel-transport) frame**, emitting a quad strip between consecutive profile rings, then caps the two ends (open path) or closes the loop (closed path). This is the curved-wall / curved-atrium / ramp generator.

**State of forge today (grep-verified):** there is **no** sweep, loft, spline, or Bézier code anywhere in forge. `crates/mesh/src/ops.rs` exposes only `merge / transform / flip_normals / apply_box_projection / weld_vertices`; `boolean.rs` exposes `boolean_union / boolean_subtract`; the only repo hit for "sweep" is the GWN probe `gwn_sweep_512` (`lib.rs:629`), unrelated. **Therefore the sweep op is genuinely net-new.** Spec:

```rust
// crates/mesh/src/sweep.rs  (NEW — engine-level, reusable by road/water/veg)

/// A 2D cross-section to sweep. Closed = a tube wall; open = a ribbon (e.g. a ramp deck).
pub struct Profile { verts: Vec<[f32; 2]>, closed: bool }  // local (u = right, v = up)

/// Ordered path samples with rotation-minimizing frames (parallel transport),
/// so the profile doesn't twist around tight bends.
pub struct CurveSamples { points: Vec<Vec3>, tangents: Vec<Vec3> }

/// Sweep `profile` along `path`. Emits crisp boundary-rep mesh: a quad strip per
/// path segment plus endcaps (open path) or a watertight loop seam (closed path).
/// Watertight invariant: closed profile + (capped open path OR closed path) =>
/// 0 boundary edges; degenerate (zero-length) segments are collapsed, never emitted.
/// Pure given inputs; deterministic vertex order (path-major, profile-minor).
pub fn sweep_profile_along_curve(path: &CurveSamples, profile: &Profile, caps: bool)
    -> Result<Mesh, ForgeError>;
```

A curved-glass entrance wing is thus a `footprint_curve(bezier) -> sweep_profile(wall_section, caps:true) -> facade_assign(curtain_wall)` chain producing crisp mesh — **no SDF approximation**. SDF surface-extraction is retained ONLY for genuine free-form *blends* (`smooth_union`/`smooth_subtract`, §4.7), which curves no longer need.

### 4.5 Named anchors / snapping points — coordinate-free composition

Every `Geometry` carries a set of **named anchors** (`Anchor` = position + normal + tangent frame). Generators expose semantically named anchors: an `extrude` exposes `roof_anchor` (roof-cap centroid, +Y normal) and per-face band anchors (`face_S_mid`, …); a `sweep_profile` exposes `start` / `end` caps. A downstream node consumes an `Anchor` on a typed `Anchor` socket and snaps to it. `place_volume` translates (and optionally rotates) its input `Geometry` so the geometry's own base frame coincides with the supplied anchor frame — so the tower is wired to `podium:roof_anchor` and **no literal `y=10.8` is ever authored** (the eval log prints the resolved value). Anchors are computed during node evaluation from the produced mesh (deterministic), exposed in the catalog (so the LLM knows `extrude` has a `roof_anchor`), and never require the author to know coordinates. This is the Geometry-Nodes "named attribute / sample-nearest" idea narrowed to discrete labeled connection points.

### 4.6 The watertight invariant, made structural per-node

The graph is **watertight-by-construction**, then gated. The survived insight from the prior design is preserved and generalized: **facades, crown, parts, and scatter are ASSIGNMENT/PLACEMENT nodes operating on already-sealed Geometry — never CSG subtractions.** Apertures stay *inside* the sealed wall generators (`facade::build_facade_walls`, `facade/mod.rs:27` — punched windows and curtain-wall mullion grids are emitted as part of a closed wall, not booleaned out of a solid). Each node type declares and locally enforces a watertight invariant:

1. **Type level (deserialize):** `#[serde(deny_unknown_fields)]` on every node + the typed-edge check (`Curve`↛`Geometry`); a closed-`Curve` consumer (`extrude`, `sweep_profile` caps) rejects an open curve.
2. **Node level (each producer is sealed):** `extrude`/`cylinder` → independently sealed boundary-rep solids (walls + flat-roof cap + soffit + underside, the `generate_stacked` loop, `lib.rs`); `sweep_profile` → 0-boundary-edge swept tube (§4.4 invariant); `boolean{void}` → ring footprint (sealed inner+outer loop via `footprint::offset_polygon` `footprint.rs:55` + `point_in_polygon` `footprint.rs:99`); `facade_assign`/`crown_kit`/`place_part` operate on / add to sealed geometry and add only closed sub-meshes. Abutting solids stay watertight by the documented overlap-band rule (`massing.rs:1-14`: an upper volume's underside is buried ≥0.22 m inside the lower cap; GWN reads |w|≈2 = interior).
3. **Gate level (terminal `asset_output`):** `assert_watertight_body` (`lib.rs:738`) — 0 non-manifold edges (`boundary_edge_stats` `lib.rs:555`), `gwn_sweep_512` < 5% fractional (`lib.rs:629`), `cook_probe_gate` `Ok` (`lib.rs:689`). The same gate every cooked building passes today.

**Contract:** any graph that type-checks and passes node-level validators produces geometry that passes the gate. The SDF-derived blend arm (§4.7) is closed by construction (extracted iso-surface of a continuous field). There is no representable middle ground that yields an open body.

### 4.7 The executor — `evaluate(graph) -> ForgeAsset`

`forge_building::blueprint::evaluate(g: &BlueprintGraph) -> Result<ForgeAsset, ForgeError>`:

1. **Type-check** the graph (typed edges, required sockets present, DAG acyclic). Return `Err(SocketTypeMismatch|MissingSocket|Cycle)` before any geometry.
2. **Topologically sort** nodes; evaluate each, caching its typed outputs. Fan-out (a `Curve` feeding two consumers) is just two edges from one cached output.
3. **Lower separable nodes LOSSLESSLY** onto today's extruders: `extrude`/`cylinder`/`setback` → `MassingVolume` (`massing.rs:24`) → boundary-rep walls/roof; `boolean{op:union}` of non-interpenetrating footprints → abutting sealed solids; `boolean{op:subtract,void:true}` → ring footprint. Crisp box-UV geometry preserved.
4. **`sweep_profile` nodes** → `forge_mesh::sweep_profile_along_curve` (§4.4) → crisp swept mesh.
5. **Only genuine free-form blends** (`smooth_union`/`smooth_subtract`) take the **SDF surface-extraction** escape hatch: build the merged narrow-band field from the `sdf_eval.slang` ops (`OP_SMOOTH_UNION=8`, `OP_SMOOTH_SUBTRACT=9`, `sdf_eval.slang:31,131`), dual-contour it (the terrain dual-contouring path, `crates/terrain`), grid pinned to the AABB for byte-stable re-cooks. The eval emits `[blueprint] WARN sdf-derived blend: facade rhythm approximate`. Curves NEVER hit this path now.
6. **`facade_assign` / `crown_kit` / `place_part` / `scatter`** apply onto sealed Geometry per §4.6.
7. **Terminal `asset_output`** runs the existing `generate_asset` tail (`orient_outward` `winding.rs:134`, `material_zones_for_mesh` `material.rs:142`, sockets, bounds, clusters) then `assert_watertight_body` (`lib.rs:738`).

### 4.8 LLM authoring via introspection tool-use

The LLM never sees a hand-maintained option list. A `forge_catalog()` reflects the **live registry** so adding an option auto-exposes it:

```rust
// crates/building/src/blueprint/catalog.rs (NEW)

/// The live node/socket/enum registry. Built from the same enums the executor
/// dispatches on (FacadeSystem description.rs:69, RoofStyle :104, Style :187,
/// CrownKitKind, ProfileKind, PartKind, Axis, Compass) — single source of truth.
pub fn forge_catalog() -> Catalog;

pub struct Catalog { pub node_types: Vec<NodeTypeInfo>, pub socket_types: Vec<&'static str> }
pub struct NodeTypeInfo { pub name: String, pub inputs: Vec<SocketInfo>, pub outputs: Vec<SocketInfo>, pub doc: String }
pub struct SocketInfo { pub name: String, pub ty: SocketTy, pub required: bool, pub enum_options: Option<Vec<String>> }

/// Narrow query: the legal options for one node's one socket — e.g.
/// get_options("facade_assign","system") == ["punched_window","curtain_wall"]
/// get_options("crown_kit","kit") == ["parapet","cornice_step","mechanical_penthouse","antenna_mast"]
pub fn get_options(node: &str, socket: &str) -> Result<Vec<String>, ForgeError>;
```

Authoring loop: the LLM calls `forge_catalog()` (and `get_options` to drill in), composes a graph using only returned node types / sockets / enum variants, and submits it. The graph is **schema-validated** (`serde` + `deny_unknown_fields` + the typed-edge check) before execution. Because the enum options come from the same Rust enums the executor matches on, **a new `FacadeSystem` variant (e.g. `ribbon`) appears in the catalog with zero prompt change, and the LLM cannot author a variant the executor can't build** (hallucinated options fail validation). Exposed over forge-cli as `forge-cli catalog --json` and, for the agent harness, an MCP/tool wrapper.

---

## 5. Data Models

```rust
// crates/building/src/blueprint/graph.rs (NEW)

/// The typed node DAG an LLM emits. Engine-level: no game concepts.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlueprintGraph {
    pub schema: u32,                                   // language version (2 = node-graph)
    pub id: String,
    pub seed: u64,
    pub floor_height: f32,
    pub style: String,                                 // keyed into Style, description.rs:187
    pub materials: BTreeMap<String, MaterialSpec>,     // asset.rs:59, referenced by id
    #[serde(default)] pub groups: BTreeMap<String, NodeGroup>, // reusable typed sub-graphs
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub op: String,                                    // registry key: "extrude","sweep_profile",…
    #[serde(default)] pub params: serde_json::Value,   // validated against the node's param schema
}

/// A typed wire. `from`/`to` are "node_id:socket_name". `ty` must equal both
/// endpoints' declared socket type or the graph fails type_check.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edge { pub from: String, pub to: String, pub ty: SocketTy }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum SocketTy { Curve, Geometry, Profile, Anchor, Float, Vector, Int, Material, Enum }

/// A reusable typed sub-graph (Geometry-Nodes "group"). Its declared inputs/
/// outputs are sockets a group_instance node exposes and wires.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeGroup {
    pub inputs:  Vec<SocketDecl>,
    pub outputs: Vec<SocketDecl>,
    pub nodes:   Vec<Node>,
    pub edges:   Vec<Edge>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SocketDecl { pub name: String, pub ty: SocketTy }
```

```rust
// Typed values flowing on wires during evaluation (internal, not deserialized).
pub enum SocketValue {
    Curve(CurveData),       // points + tangents + closed flag (blueprint::curve)
    Geometry(GeomFragment), // Mesh + Vec<NamedAnchor> + Vec<MaterialZone>
    Profile(forge_mesh::sweep::Profile),
    Anchor(NamedAnchor),
    Float(f32), Vector([f32;3]), Int(i64),
    Material(String),       // key into BlueprintGraph.materials
    Enum(String),           // a registry-validated variant string
}

pub struct NamedAnchor { pub name: String, pub position: [f32;3], pub normal: [f32;3], pub tangent: [f32;3] }
```

`MaterialSpec`, `FacadeSystem`, `RoofStyle`, `Style`, `Socket`, `SocketType`, `MaterialZone`, `GeometryCluster`, `MassingVolume`, `Compass`/`Axis` reuse existing Forge types (`asset.rs:30,42,59,103,114`, `description.rs:69,104,187`, `massing.rs:24`). The language introduces only the graph/node/edge/socket types above + `forge_mesh::sweep::{Profile, CurveSamples}` (§4.4).

---

## 6. API

```rust
// crates/building/src/blueprint/mod.rs (NEW)

/// Deserialize + type-check a graph WITHOUT building geometry (typed edges,
/// required sockets, acyclicity, node-level param schemas). For early authoring
/// rejection. Pure, any thread.
pub fn validate(json: &str) -> Result<BlueprintGraph, ForgeError>;

pub struct TypeCheck { pub node_count: usize, pub edge_count: usize }
pub fn type_check(g: &BlueprintGraph) -> Result<TypeCheck, ForgeError>;

/// Topologically evaluate the DAG to the same ForgeAsset every cooked building
/// yields. Lowers separable nodes losslessly to MassingVolume + boundary-rep
/// extruders; sweep_profile -> crisp swept mesh; only smooth_* blends use SDF
/// extraction. Ends with assert_watertight_body. Returns Err on any validator
/// failure; PANICS only inside assert_watertight_body if an invariant is
/// violated (a bug — validators guarantee valid input cannot reach it).
/// Pure given seed; rayon used internally for per-node meshing.
pub fn evaluate(g: &BlueprintGraph) -> Result<ForgeAsset, ForgeError>;

// catalog.rs
pub fn forge_catalog() -> Catalog;
pub fn get_options(node: &str, socket: &str) -> Result<Vec<String>, ForgeError>;
```

```rust
// crates/mesh/src/sweep.rs (NEW)
pub fn sweep_profile_along_curve(path: &CurveSamples, profile: &Profile, caps: bool)
    -> Result<Mesh, ForgeError>;     // §4.4 — crisp, watertight, deterministic order
```

```rust
// Reused unchanged (cited):
//   facade::build_facade_walls(...)         forge facade/mod.rs:27   (system: PunchedWindow|CurtainWall)
//   facade::storefront::emit_storefront_cell(...) forge facade/storefront.rs:33  (wire as a 3rd arm)
//   massing::massing_volumes(...)           forge massing.rs:82
//   footprint::offset_polygon / point_in_polygon   forge footprint.rs:55,99
//   roof::flat_roof_over_polygon(...)       forge roof.rs:73
//   facade::door_leaf::generate_door_leaf(bottom_left,tangent,outward,width,height) forge facade/door_leaf.rs:136
//   winding::orient_outward(...)            forge winding.rs:134
//   material::material_zones_for_mesh(...)  forge material.rs:142
//   forge_mesh::ops::{merge,transform}      forge crates/mesh/src/ops.rs:4,27
//   assert_watertight_body(...)             forge lib.rs:738   (the gate)
//   sdf_eval.slang OP_SMOOTH_UNION/SUBTRACT slang/sdf_eval.slang:31,131 (blend arm only)
```

```rust
// civitas src/asset/directive.rs — the bridge (§8)
/// Lower a legacy flat directive to an equivalent graph PRESET (fixed sub-graph).
/// Byte-identical mesh for box-massing directives (the regression gate).
pub fn directive_to_graph(d: &AssetDirective) -> BlueprintGraph;
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `blueprint::validate` / `type_check` | `forge-cli run` when input is `*.bp.json` (schema 2) | `forge crates/forge-cli` | new CLI arm beside `building` |
| `blueprint::evaluate` | forge-cli blueprint arm; returns `ForgeAsset` JSON | `forge crates/building/src/blueprint/eval.rs` | reuses `generate_asset` tail |
| `forge_catalog` / `get_options` | `forge-cli catalog --json`; agent MCP wrapper | `forge crates/building/src/blueprint/catalog.rs` | reflects live enums — single source of truth |
| `sweep_profile_along_curve` | `evaluate` `sweep_profile` node | `forge crates/mesh/src/sweep.rs` | net-new; reusable by road/water/veg |
| `evaluate` SDF blend arm | `evaluate` smooth_* nodes | `forge crates/building/src/blueprint/eval.rs` | dual-contour narrow-band; reuses terrain extraction; curves never reach it |
| `directive_to_graph` | `recipe_from_directive` | `civitas game_asset_cook.rs` | legacy directives lower to a Box-massing graph preset; cook sends `{"command":"blueprint",…}` |
| `assert_watertight_body` | end of `evaluate` (`asset_output`) | `forge lib.rs:738` | the gate — unchanged |
| `ready_sdf` (closed proof + atoms) | cook, after Forge returns | `civitas game_asset_cook.rs` | unchanged: GWN sign, `--sdf-strict` rejects open bodies |

Backward compat: the `"command":"building"` flat path stays for one release; `directive_to_graph` + a `box_massing_graph_is_byte_identical` gate (blake3 of positions vs the legacy `generate` path) proves the migration is lossless before the flat path retires.

---

## 8. Migration — the flat directive becomes a graph preset

The legacy `AssetDirective` lowers to a **fixed sub-graph preset** (not a hand-written graph): `directive_to_graph` emits `footprint_curve(shape preset) -> extrude(floors) -> facade_assign(system, material) -> [setback/podium nodes if massing_mode != Box] -> asset_output`. For `MassingMode::Box` the preset is exactly one `extrude` + one `facade_assign` + `asset_output`, lowering through the same `MassingVolume`→boundary-rep path the legacy `generate` (`lib.rs:40`) uses, so the output mesh is **byte-identical** (the `box_massing_graph_is_byte_identical` gate hashes positions). PodiumTower/Setback map to the `setback`/`place_volume` nodes over `massing::massing_volumes` (`massing.rs:82`). This lets the whole content library migrate behind the gate before the flat path is removed.

---

## 9. Worked Example — the CS2-class hospital, as a node graph WITH a curve

Hand-authored graph (abridged; full file `assets/blueprints/civic.hospital_demo.bp.json`). Adds a **curved-glass entrance wing** to prove curves are crisp mesh, plus the light-well courtyard, inset tower (anchor-snapped), mixed skin, crown kit, and a pivoted ambulance door.

```jsonc
{ "schema": 2, "id": "civic.hospital_demo", "seed": 7, "floor_height": 3.6, "style": "modern",
  "materials": { "stone": {...}, "vision_glass": {"glass": true, ...}, "spandrel": {...}, "steel": {...} },
  "nodes": [
    // --- podium with a courtyard light-well (CSG void) ---
    {"id":"podium_fp","op":"footprint_curve","params":{"verts":[[0,0],[60,0],[60,40],[0,40]],"closed":true}},
    {"id":"podium","op":"extrude","params":{"floors":3}},
    {"id":"lightwell_fp","op":"footprint_curve","params":{"verts":[[26,16],[34,16],[34,24],[26,24]],"closed":true}},
    {"id":"lightwell","op":"extrude","params":{"floors":3}},
    {"id":"court","op":"boolean","params":{"op":"subtract","void":true}},
    // --- inset tower, SNAPPED to the podium roof anchor (no literal y) ---
    {"id":"tower_fp","op":"footprint_curve","params":{"verts":[[18,8],[42,8],[42,32],[18,32]],"closed":true}},
    {"id":"tower","op":"extrude","params":{"floors":9}},
    {"id":"place_tower","op":"place_volume"},
    // --- CURVED-GLASS ENTRANCE WING: Bézier path + swept wall section (CRISP) ---
    {"id":"wing_path","op":"footprint_curve",
       "params":{"verts":[[60,12],[72,16],[72,28],[60,32]],"closed":false,"bezier":true}},
    {"id":"wing_section","op":"profile_lib","params":{"kind":"glass_wall","thickness":0.4,"height":7.2}},
    {"id":"wing","op":"sweep_profile","params":{"caps":true}},
    {"id":"wing_skin","op":"facade_assign",
       "params":{"system":"curtain_wall","bay_width":1.5,"windows_per_bay":1,"window_density":0.95,
                 "material":"spandrel","glass_material":"vision_glass"}},
    // --- assemble ---
    {"id":"body","op":"boolean","params":{"op":"union"}},
    {"id":"body2","op":"boolean","params":{"op":"union"}},
    // --- skin / crown / part ---
    {"id":"podium_skin","op":"facade_assign","params":{"system":"punched_window","floors":[0,3],"bay_width":3.0,"windows_per_bay":2,"window_density":0.55,"material":"stone"}},
    {"id":"tower_skin","op":"facade_assign","params":{"system":"curtain_wall","bay_width":1.5,"windows_per_bay":1,"window_density":0.92,"material":"spandrel","glass_material":"vision_glass"}},
    {"id":"crown","op":"crown_kit","params":{"kit":"antenna_mast","height":12.0,"radius":0.25,"material":"steel"}},
    {"id":"door","op":"place_part","params":{"kind":"door","size":[5.0,4.0],"pivot_axis":"Y","material":"steel"}},
    {"id":"out","op":"asset_output"}
  ],
  "edges": [
    {"from":"podium_fp:curve","to":"podium:curve","ty":"Curve"},
    {"from":"lightwell_fp:curve","to":"lightwell:curve","ty":"Curve"},
    {"from":"podium:geom","to":"court:a","ty":"Geometry"},
    {"from":"lightwell:geom","to":"court:b","ty":"Geometry"},
    {"from":"court:geom","to":"podium_skin:geom","ty":"Geometry"},
    {"from":"tower_fp:curve","to":"tower:curve","ty":"Curve"},
    {"from":"tower:geom","to":"tower_skin:geom","ty":"Geometry"},
    {"from":"tower_skin:geom","to":"place_tower:geom","ty":"Geometry"},
    {"from":"podium:roof_anchor","to":"place_tower:base","ty":"Anchor"},   // <-- coordinate-free snap
    {"from":"wing_path:curve","to":"wing:curve","ty":"Curve"},
    {"from":"wing_section:profile","to":"wing:profile","ty":"Profile"},
    {"from":"wing:geom","to":"wing_skin:geom","ty":"Geometry"},
    {"from":"podium_skin:geom","to":"body:a","ty":"Geometry"},
    {"from":"place_tower:geom","to":"body:b","ty":"Geometry"},
    {"from":"body:geom","to":"body2:a","ty":"Geometry"},
    {"from":"wing_skin:geom","to":"body2:b","ty":"Geometry"},
    {"from":"body2:geom","to":"crown:geom","ty":"Geometry"},
    {"from":"crown:geom","to":"door:geom","ty":"Geometry"},
    {"from":"podium:face_S_mid","to":"door:anchor","ty":"Anchor"},
    {"from":"door:geom","to":"out:geom","ty":"Geometry"}
  ] }
```

**Graph evaluation walk-through:**
1. **Type-check.** 19 typed edges; every `Curve→Curve`, `Geometry→Geometry`, `Anchor→Anchor`, `Profile→Profile`. `wing:curve` accepts the open Bézier (sweep needs no closed loop); `podium:curve` requires closed (an open curve here would `Err`). DAG acyclic → `[graph] … type-check OK`.
2. **Footprints.** `podium_fp`/`lightwell_fp`/`tower_fp` evaluate to closed polyline `Curve`s; `wing_path` tessellates its Bézier control net to a 96-segment polyline with parallel-transport frames.
3. **Volumes + courtyard.** `podium`/`lightwell`/`tower` `extrude` to `MassingVolume`s → boundary-rep solids. `court` = `subtract{void:true}` → a **ring footprint** (outer 60×40 + inner 8×8, both sealed via `offset_polygon`+`point_in_polygon`) — separable courtyard, NOT an SDF blob.
4. **Anchor snap.** `podium` exposes `roof_anchor` at its roof-cap centroid (y = 3×3.6 = 10.8). `place_tower` snaps the tower's base frame to it → tower `y_base = 10.80` with **no literal coord in the JSON** → `[graph] anchors: tower.base snapped … at y=10.80`.
5. **CURVED WING (the proof).** `wing` = `sweep_profile(wing_path, glass_wall section, caps:true)` → `forge_mesh::sweep_profile_along_curve` transports the 7.2 m glass-wall section along the 96-segment arc, quad-strips between rings, caps both ends → **1 crisp swept watertight component** (`connected_components==1`, `non_manifold==0`). `wing_skin` clads it `curtain_wall`. Log: `[graph] curve: entrance_wing … 1 crisp swept component` — and crucially **no `sdf-derived` warning**.
6. **Skin.** `podium_skin` = `punched_window` stone; `tower_skin` = `curtain_wall` spandrel + vision glass tagged `MAT_GLASS=3`. `facade::build_facade_walls` (`facade/mod.rs:27`) — apertures inside sealed walls, not booleaned.
7. **Crown + part.** `crown` adds a 12 m capsule antenna mast as a disjoint closed component (winding oracle protects it, `winding.rs`). `door` = `place_part` snapped to `podium:face_S_mid` → `generate_door_leaf` (`door_leaf.rs:136`) as a disjoint component + `Socket{Door, pivot (30,0,0), axis Y}` + its own `GeometryCluster` (mover/refit path) — **not fused**.
8. **Output + gate.** `out` runs `orient_outward` + `material_zones_for_mesh` + sockets + clusters, then `assert_watertight_body` (`lib.rs:738`): courtyard ring, abutting tower (|w|≈2 overlap band), swept wing (0 boundary edges by §4.4 invariant), free-standing mast & door all pass → `watertight OK` → cook writes `sign=Closed` atoms.

**Outcome.** The graph expressed the full CS2-class hospital — light-well courtyard (CSG void), anchor-snapped inset curtain-wall tower (coordinate-free), mixed punched/curtain-wall skin, antenna crown, pivoted ambulance door — **and a curved-glass entrance wing that produced CRISP swept mesh, not an SDF blob.** Curves are first-class; the old "swooping wing forces SDF / approximate facade" caveat is gone.

---

## 10. Open Questions

- [ ] Bézier tessellation resolution policy: fixed segments-per-metre vs curvature-adaptive? Adaptive gives better silhouettes but must stay seed/grid-stable for byte-identical re-cooks (the cook hashes outputs). Leaning: curvature-adaptive with the sample count pinned to the curve AABB.
- [ ] Bay rhythm on a swept curtain wall: `facade::build_facade_walls` (`facade/mod.rs:27`) consumes straight `WallEdge`s; cladding a curved wing needs per-segment bay fitting along the swept rings (`curtain_wall.rs` `bays_for_span`). Verify it tiles cleanly on a curve before specing `facade_assign` over swept geometry.
- [ ] Anchor frame ambiguity: a `roof_anchor` exposes position+normal; does `place_volume` also need a rotation reference (tangent) for rotated wings, or is +Y-up sufficient for towers? Spec the `tangent` field now (§4.5) but confirm rotated-wing snapping.
- [ ] `storefront` wiring: `facade/storefront.rs:33 emit_storefront_cell` exists but is NOT wired into `build_facade_walls` (only `PunchedWindow`/`CurtainWall` arms today). Adding it as a 3rd `FacadeSystem` variant auto-exposes it in the catalog — confirm the cell emitter produces a sealed wall band.
- [ ] Determinism of the SDF blend arm: dual-contouring must be seed-stable, grid-aligned (pin origin to volume AABB) so re-cooks are byte-identical.

---

## 11. Out of Scope

- Skinned/deforming articulated parts (rigid only; deformation is the costly class, `articulated-parts-budget`).
- Pitched/gabled roofs over non-rectangular or multi-volume massing (deferred straight-skeleton work, `lib.rs:151`); the hospital is flat-roofed.
- Loft (profile interpolation between two different sections along a curve) — the v1 sweep uses ONE profile; varying-section loft is a follow-up sweep mode.
- The LLM authoring *harness* (how the model is steered to query the catalog and iterate) — this design specs the graph + catalog API + executor; the agent loop is a follow-up.
- Non-building types (terrain/road/water/vegetation) — they get their own node sets under the same directive-factory envelope (`forge-directive-factory`); but `sweep_profile_along_curve` is built engine-level (`crates/mesh`) so they reuse it.
- USD ingestion path — unchanged; remains the answer for free-form artistic one-offs (`civitas-usd-interchange-decision`), though curves no longer force USD for curved buildings.

---

## 12. Related Plans / Designs

- Supersedes: the first-cut "header + 3 sub-trees" revision of this doc (the DAG subsumes it — header+subtrees is one possible sub-graph).
- Depends on: the box/podium/setback massing grammar (`forge massing.rs`, `description.rs`) and the watertight oracle (`forge lib.rs:738`, `winding.rs`) — shipped; the net-new `forge_mesh::sweep_profile_along_curve` (`crates/mesh/src/sweep.rs`).
- Reuses: Spectra `sdf_eval.slang` smooth_* ops (`sdf_eval.slang:31,131`) for the blend escape hatch only.
- Required before: an LLM authoring harness over `forge_catalog()`; the per-type directive-factory envelope expansion.
- Related: memory `llm-blueprint-forge-directive` (EVOLUTION 2026-06-13), `hybrid-atom-sdf-lod-direction`, `forge-directive-factory`, `building-content-pipeline`; `2026-06-12-building-composition-grammar` (civitas docs/specs).
