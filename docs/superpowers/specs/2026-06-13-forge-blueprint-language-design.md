# Design: Forge Blueprint Language — the compositional, LLM-authorable building grammar (2026-06-13)

**Status:** Draft
**Scope:** Replace Forge's flat parametric building directive with a composable OP-TREE an LLM emits as JSON, executed deterministically by a Forge `blueprint` evaluator into the existing `ForgeAsset` payload (mesh + sockets + material zones + clusters), so "if a human can design it, an LLM can blueprint it to Forge and Forge builds it." Affects: civitas `src/asset/directive.rs` + `src/bin/game_asset_cook.rs`; Forge `crates/building/src/{lib,description,massing,footprint,facade/*,roof,seal,winding,socket,material}.rs`.
**Related:** memory `llm-blueprint-forge-directive`, `forge-directive-factory`, `building-content-pipeline`, `articulated-parts-budget`; `2026-06-12-building-composition-grammar` (civitas docs/specs); Spectra `slang/sdf_eval.slang`.

---

## 1. Problem Statement

- Today's directive (`civitas src/asset/directive.rs:96-157`, `DirectiveForge`) is a **flat parameter set**: one `footprint_shape` from a closed 4-value enum (`Rectangular|LShaped|UShaped|TShaped`, `forge footprint.rs:122`), one `massing_mode` from a closed 3-value enum (`box|podium_tower|setback`, `forge description.rs:88`), one `facade_system` per band. An LLM can vary knobs **within** a family but cannot describe an arbitrary bespoke form: it cannot say "an L-shaped podium with a cylindrical tower notched by a light-well and a stepped crown carrying an antenna mast." The grammar is fixed knobs, not a composable language.
- Massing is a single linear stack of inset prisms (`forge massing.rs:82-174`, `MassingVolume` list). There is no union of two towers, no subtraction of a courtyard void, no smooth blend — even though the SDF-CSG that does exactly this (`union/subtract/intersect/smooth_union/smooth_subtract`, `spectra sdf_eval.slang:23-32`) already exists and was just generalized for terrain.
- Articulated parts (a hospital ambulance-bay door) have **no authoring path**: `socket.rs:9 build_residential_sockets` emits a fixed 9-socket residential set; there is no way for a blueprint to place a hinged door with a pivot as a separate rigid sub-instance (the requirement from memory `articulated-parts-budget`).
- The watertight oracle (`forge lib.rs:738 assert_watertight_body` → non-manifold-edge gate + `gwn_sweep_512` < 5% fractional + `cook_probe_gate`) is enforced **after** generation on a fixed pipeline. A free-composition language risks emitting bodies that fail the gate; the language must make a broken body **unrepresentable**, not merely rejected.

---

## 2. Done When

Running

```bash
forge-cli run "$(cat assets/blueprints/civic.hospital_demo.bp.json)" --emit asset \
  && cargo run -p civitas_care --bin game_asset_cook -- --only civic.hospital_demo --sdf-strict
```

produces, on stdout, the lines:

```
[blueprint] civic.hospital_demo: 5 volume nodes, 2 CSG ops, 41 sub-meshes -> watertight OK (0 non-manifold edges, 0.0% fractional, interior 512/512)
[blueprint] articulated: 1 part 'ambulance_bay_door' pivot=(x,y,z) axis=Y -> emitted as socket+sub-mesh (not fused)
[sdf-validate] civic.hospital_demo: sign=gwn inside_negative=1.000 eikonal_p95<0.10
[cook] wrote atoms/civic.hospital_demo.atoms.json (NNNNN atoms, sdf sign=Closed)
```

A human at the keyboard reads "watertight OK" and a `sign=Closed` SDF for a hand-authored, multi-volume, CSG-composed hospital blueprint — proof the language is expressive AND the seal invariant held through free composition.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Op-tree deserializes & validates | `assert_eq!(bp.massing.eval_volume_count(), 5)` on the hospital blueprint JSON; unknown op variant FAILS deserialization (`serde` `#[serde(deny_unknown_fields)]`) | `assert!(serde_json::from_str::<Blueprint>(j).is_ok())` — passes on empty `{}` |
| Massing CSG union/subtract | union of two prism volumes whose AABBs overlap yields a single connected mesh component: `assert_eq!(connected_components(&mesh), 1)` after `evaluate`; subtract of a courtyard prism yields `assert!(point_in_polygon_at_height(courtyard_center) == false)` | `assert!(mesh.indices.len() > 0)` — passes on either operand alone |
| Per-volume facade assignment | a blueprint assigning `curtain_wall` to the tower volume and `storefront` to the podium yields ≥2 distinct `MaterialZone.material.base_color` with the glass zone tagged `MAT_GLASS=3`: `assert!(zones.iter().any(|z| z.material.is_glass()) && zones.iter().filter(...).count() >= 2)` | `assert!(!zones.is_empty())` |
| Crown kit (parapet + penthouse + antenna) | the crown node emits ≥1 closed box above the top volume's wall plane AND a capsule mast: `assert!(mesh.aabb().max[1] > top_volume_height + antenna_height - 0.01)` | `assert!(crown_meshes.len() > 0)` |
| Articulated part as separate sub-mesh + socket | a `door` part with a pivot yields a `Socket{socket_type: Door, ...}` AND a sub-mesh whose triangle range is NOT welded into the shell's coherent component: `assert!(part_mesh_component_disjoint_from_shell(&asset))` and `assert_eq!(asset.sockets.iter().filter(|s| s.socket_type==Door).count(), 1)` | `assert!(asset.sockets.len() > 9)` |
| Watertight invariant under composition | `assert_watertight_body(&asset, "hospital")` returns without panic: 0 non-manifold edges, `gwn_sweep_512` fractional < 5%, interior ≥ 30, `cook_probe_gate` = `Ok` | `assert!(asset.mesh.indices.len() % 3 == 0)` — passes on any triangle soup |
| Today's directive is a preset | the live `rowhouse_01.asset.json` lowered to a `Blueprint` then evaluated produces a mesh byte-identical to today's `generate(params)` path: `assert_eq!(blake3(new_mesh.positions), blake3(legacy_mesh.positions))` | `assert!(legacy_directive_still_parses())` |

---

## 4. Architecture

### 4.1 Layering — where the language lives

The blueprint language is **ENGINE-level** (Forge), game-agnostic per CLAUDE.md: it knows volumes, facades, roofs, materials, parts — never "hospital" or "zone." A new Forge type `Blueprint` (in `crates/building`, the most-developed generator) deserializes the op-tree and lowers it to the existing `generate_asset`-equivalent pipeline. The civitas GAME layer keeps `AssetDirective` (gameplay: households/jobs/zone/care-category) and gains a thin lowering: directive → blueprint (`recipe_from_directive` in `game_asset_cook.rs:836` learns to emit blueprint JSON instead of the flat param block). Forge stays the deterministic builder; the LLM (or a preset) is the designer.

### 4.2 The op-tree shape (three sub-trees + a header)

A `Blueprint` is **not** one homogeneous CSG tree but a **header + three typed sub-trees**, because building authorship has three distinct concerns that compose differently:

1. **`massing`** — a CSG tree of *volumes* (the SOLID). This is the part that maps directly onto `sdf_eval.slang`'s `SDFOp`: `Volume` leaves (prism/cylinder/wedge from a footprint loop + height) combined by `union/subtract/intersect/smooth_union/smooth_subtract`. The CSG tree is evaluated **twice**: once on the GPU/CPU SDF path to *prove* the merged solid is closed (the seal oracle's substrate), and once as the mesh-generation driver — but mesh comes from the existing per-volume *boundary-rep* extruders, NOT from marching the SDF, so today's crisp wall/cornice/UV geometry is preserved (see §4.4).
2. **`skin`** — a list of *facade assignments*, each binding a face-selector (which volume + which side band) to a facade generator (`punched_window | curtain_wall | storefront`) with that generator's params (bay rhythm, window density, materials).
3. **`crown`** — the *designed top*: an ordered kit list (`parapet | mechanical_penthouse | antenna_mast | cornice_step`) placed on the top volume's wall plane.

Plus **`details`** (entrances/ornament) and **`parts`** (articulated: doors/gates with pivots) as flat lists with placement anchors, and a **`palette`/`materials`** table referenced by id.

This split is the key design decision: **massing is CSG (a tree), but facade/crown/parts are assignments/lists keyed to massing nodes**, because a curtain-wall skin is a *property of a volume's faces*, not a boolean of solids. Trying to express windows as CSG subtractions is how watertightness dies (every aperture becomes a hole to re-seal). Instead, apertures stay inside the facade generators that already produce sealed walls (`facade::build_facade_walls`, `forge facade/mod.rs:27`).

### 4.3 The executor — walking the tree to a `ForgeAsset`

`forge_building::evaluate(bp: &Blueprint) -> Result<ForgeAsset, ForgeError>` runs these stages, every stage enforcing the watertight invariant locally:

1. **Lower massing CSG → `Vec<MassingVolume>` + a residual CSG manifest.** For the *separable* case (the tree is a union of prisms whose footprints don't interpenetrate, plus subtractions that are courtyard voids cut cleanly through a volume's footprint), lower each leaf volume to a `MassingVolume{polygon, y_base, floors, facade_system}` — exactly the struct `massing.rs:24` already produces — and rewrite courtyard `subtract` ops as **ring/void footprint polygons** (the polygon offset machinery, `footprint.rs:55 offset_polygon`, plus an inner loop). Each volume is then built as an **independently sealed boundary-rep solid** by the *same* per-volume loop as `generate_stacked` (`lib.rs:296-388`): walls + flat-roof cap + soffit + underside. Stacked/abutting solids stay watertight by the documented overlap-band rule (`massing.rs:1-14`: an upper volume's underside is buried ≥0.22 m inside the lower cap; GWN probes read |w|≈2 = interior).
2. **For the *non-separable* case** (a `smooth_union` blend, or a free blob where boundary-rep extrusion can't represent the merged surface), the executor marks the sub-tree `SdfDerived` and produces its mesh by **surface extraction from the SDF** (dual contouring over the narrow band — the same path terrain uses, memory `3d-sdf-terrain-directive`). This is the escape hatch; it is lower-fidelity (no crisp box UVs) and the validator emits `[blueprint] WARN sdf-derived volume: facade rhythm approximate`. Most real buildings (≈95%, per `building-content-pipeline`) are separable and never touch this path.
3. **Apply skin.** For each `MassingVolume`, look up its facade assignment and call `facade::build_facade_walls(facade_system, &edges, floors, eff_floor_height, bay_width, windows_per_bay, window_density, style, seed, at_grade)` (`facade/mod.rs:27`) — already the dispatch point for punched/curtain-wall; `storefront` (`facade/storefront.rs:33`) is added as a third arm. Materials come from the palette table → `MaterialSpec` (`asset.rs:59`).
4. **Apply crown** on the top volume only (the existing `flat_roof_crown` path, `lib.rs:408`), extended: `parapet` → `generate_parapet`, `cornice_step` → `generate_cornice_stepped`, `mechanical_penthouse` → a new small inset sealed prism placed on the roof cap, `antenna_mast` → an `sdf_capsule`-shaped or thin-prism closed solid as a separate coherent component (so the winding oracle protects it, `winding.rs:161`).
5. **Apply details** (entrances, ornament) — parametric trim courses + the authored-kit instances (the HYBRID decision, `building-content-pipeline`), each a closed sub-mesh.
6. **Apply parts** (§4.5) — each articulated part is built as its OWN closed sub-mesh, NOT merged into the shell, and registered as a `Socket` + a separate cluster (so it rides the TLAS-refit mover path, memory `articulated-parts-budget`).
7. **Assemble + orient + gate.** Concatenate sub-meshes, run `winding::orient_outward` (`winding.rs:134`), compute `material_zones_for_mesh` (`material.rs:142`), `build_*_sockets`, bounds, clusters — i.e. the existing `generate_asset` tail (`lib.rs:478-525`). Finally call `assert_watertight_body(&asset, &bp.id)` (`lib.rs:738`). **A blueprint that would fail the gate fails the cook with `--sdf-strict`** — but the validators in steps 1/2 are designed so that any *deserializable* blueprint produces a sealed body (see §4.6).

### 4.4 Why mesh comes from boundary-rep, not the SDF (the fidelity decision)

The CSG tree is the *authoring abstraction* and the *seal proof*, but the visible mesh is built by the existing extruders. Reason: the renderer renders MESH (memory `hybrid-atom-sdf-lod-direction`), and the crisp clapboard/brick/window-frame/curtain-wall geometry that already cooks comes from `facade::build_facade_walls` + `roof::*` + box-projected UVs (`lib.rs:478 apply_box_projection`). Marching the SDF would smear all of that. So: **separable CSG (the common case) lowers losslessly to `MassingVolume`s and uses the boundary-rep extruders; only genuinely non-separable blends fall back to SDF surface extraction.** The SDF is always *also* computed (downstream by the cook's `ready_sdf`, `game_asset_cook.rs:2909`) as the closed-ness proof and the engine substrate, exactly as today.

### 4.5 Articulated parts

A `parts[]` entry is `{kind: door|gate|barrier, anchor: <volume-id + face + uv>, size, pivot: {point:[x,y,z], axis: X|Y|Z}, material}`. The executor:
- builds the leaf/panel geometry via the existing `facade::door_leaf::generate_door_leaf` (`facade/door_leaf.rs:136`) as a **separate coherent closed component**;
- emits a `Socket{socket_type: Door|GatePost, position, normal, channel: Structure, compatible_tags}` (`asset.rs:42`, `socket.rs`) at the pivot — the socket carries the pivot so the runtime can hinge it;
- emits a dedicated `GeometryCluster` (`asset.rs:114`) for the part so it can be promoted to its own BLAS instance on the mover/refit path. It is **never welded** into the shell's coherent component (the budget trap, `articulated-parts-budget`).

### 4.6 The watertight invariant, made structural

The language prevents broken bodies at three levels:
1. **Type level (deserialization):** `#[serde(deny_unknown_fields)]` on every node; `Volume` requires a closed footprint loop (≥3 verts); `subtract` children that are courtyard voids must be marked `void: true` so the lowering knows to make a ring footprint (a sealed inner+outer loop) rather than an open hole.
2. **Lowering level (validators, return `Err` before any geometry):** each `MassingVolume` validated by the existing `massing_volumes` rules (`massing.rs:89-173`: inset doesn't exhaust the half-extent, podium+tower floors sum, setback bands tile the height); CSG `subtract` validated so the void footprint is strictly interior with ≥ wall-thickness clearance (reuse `point_in_polygon`, `footprint.rs:99`).
3. **Gate level (post-assembly, the existing oracle):** `assert_watertight_body` (`lib.rs:738`) — 0 non-manifold edges (`boundary_edge_stats`), `gwn_sweep_512` < 5% fractional, `cook_probe_gate` `Ok`. This is the same gate every cooked building passes today; the language just feeds it.

The contract: **any blueprint that deserializes and passes the lowering validators produces geometry that passes the gate.** The SDF-derived fallback (§4.3 step 2) is closed by construction (it's an extracted iso-surface of a continuous field). The separable path is closed by the documented per-volume seal + overlap-band rule. There is no representable middle ground that yields an open body.

---

## 5. Data Models

```rust
/// The root op-tree an LLM emits. Engine-level: no game concepts.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Blueprint {
    pub schema: u32,                  // language version
    pub id: String,
    pub seed: u64,
    pub floor_height: f32,
    pub style: StyleKey,              // forge description.rs:135 (JSON-keyed, not an enum)
    pub massing: MassingNode,         // the CSG SOLID tree
    pub skin: Vec<FacadeAssignment>,  // face-selector -> facade generator
    #[serde(default)] pub crown: Vec<CrownKit>,
    #[serde(default)] pub details: Vec<DetailPlacement>,
    #[serde(default)] pub parts: Vec<ArticulatedPart>,
    pub materials: BTreeMap<String, MaterialSpec>, // forge asset.rs:59, referenced by id
}

/// Massing CSG tree. Maps onto spectra sdf_eval.slang SDFOp. Leaves are
/// volumes; internal nodes are booleans. Each Volume carries the id the
/// skin/crown/parts selectors reference.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum MassingNode {
    /// Leaf: an extruded prism / cylinder / wedge.
    Volume {
        id: String,
        shape: VolumeShape,           // Prism{footprint loop} | Cylinder{r,sides} | Wedge
        floors: u8,
        transform: Xform,             // translate/rotate in XZ (towers off-center, rotated wings)
    },
    Union     { a: Box<MassingNode>, b: Box<MassingNode> },
    /// `b` is carved out of `a`. If b.void==true, lowered to a ring footprint
    /// (sealed courtyard), else an SDF-derived notch.
    Subtract  { a: Box<MassingNode>, b: Box<MassingNode>, #[serde(default)] void: bool },
    Intersect { a: Box<MassingNode>, b: Box<MassingNode> },
    SmoothUnion    { a: Box<MassingNode>, b: Box<MassingNode>, k: f32 }, // -> SDF-derived
    SmoothSubtract { a: Box<MassingNode>, b: Box<MassingNode>, k: f32 }, // -> SDF-derived
}

/// Footprint shape for a Volume leaf. Prism reuses footprint.rs polygon ops;
/// the closed Rectangular/L/U/T enum becomes ONE preset of Prism's free loop.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VolumeShape {
    Prism { footprint: Vec<[f32; 2]> }, // closed loop, >=3 verts, CCW outer
    Cylinder { radius: f32, sides: u8 },
    Wedge { footprint: Vec<[f32; 2]>, taper: f32 },
}

/// Binds a face selector to a facade generator. The selector keys a Volume id
/// (+ optional side band by floor range or compass face).
#[derive(Debug, Clone, Deserialize)]
pub struct FacadeAssignment {
    pub volume: String,                       // Volume.id
    #[serde(default)] pub floors: Option<[u8; 2]>, // floor band, else whole volume
    #[serde(default)] pub faces: Vec<Compass>,     // N/E/S/W, else all faces
    pub system: FacadeSystem,                 // forge description.rs:69 + Storefront
    pub bay_width: f32,
    pub windows_per_bay: u8,
    pub window_density: f32,
    pub material: String,                     // key into Blueprint.materials
    #[serde(default)] pub glass_material: Option<String>,
}

/// The designed top. Placed on the top volume's wall plane, in order.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kit", rename_all = "snake_case", deny_unknown_fields)]
pub enum CrownKit {
    Parapet { height: f32, projection: f32 },
    CorniceStep { steps: u8, height: f32, projection: f32 },
    MechanicalPenthouse { inset: f32, height: f32, material: String },
    AntennaMast { height: f32, radius: f32, material: String },
}

/// Articulated part — a SEPARATE rigid sub-mesh + socket, never fused.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArticulatedPart {
    pub id: String,
    pub kind: PartKind,               // Door | Gate | Barrier
    pub anchor: FaceAnchor,           // volume id + face + uv on that face
    pub size: [f32; 2],
    pub pivot: Pivot,                 // point + axis (X|Y|Z)
    pub material: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Pivot { pub point: [f32; 3], pub axis: Axis }
```

`MaterialSpec`, `FacadeSystem`, `StyleKey`, `Socket`, `SocketType`, `MaterialZone`, `GeometryCluster`, `MassingVolume`, `Xform`/`Compass`/`Axis` reuse existing Forge types (cited in §6) — the language introduces only the node enums above.

---

## 6. API

```rust
// crates/building/src/blueprint.rs (NEW)

/// Deserialize + validate a blueprint without building geometry.
/// Runs the lowering validators (massing inset/floor/void rules) so an
/// authoring tool can reject early. Threading: pure, any thread.
pub fn validate(json: &str) -> Result<Blueprint, ForgeError>;

/// Evaluate a blueprint to the same ForgeAsset every cooked building yields.
/// Internally: lower massing -> Vec<MassingVolume> (+ SdfDerived sub-trees),
/// apply skin/crown/details/parts, orient_outward, material_zones, sockets,
/// then assert_watertight_body. Returns Err on any validator failure; PANICS
/// only inside assert_watertight_body if an invariant is violated (a bug, not
/// bad input — the validators guarantee it cannot happen for valid input).
/// Threading: pure given seed; rayon used internally for per-volume meshing.
pub fn evaluate(bp: &Blueprint) -> Result<ForgeAsset, ForgeError>;

// Reused unchanged (cited):
//   facade::build_facade_walls(...)      forge facade/mod.rs:27
//   massing::massing_volumes(...)        forge massing.rs:82
//   footprint::offset_polygon(...)       forge footprint.rs:55
//   roof::flat_roof_over_polygon(...)    forge roof.rs
//   seal::footprint_underside / roof_soffit_within_outline   forge seal.rs:69,120
//   winding::orient_outward(...)         forge winding.rs:134
//   material::material_zones_for_mesh(...) forge material.rs:142
//   <building>::assert_watertight_body(...) forge lib.rs:738   (the gate)
//   facade::door_leaf::generate_door_leaf(...) forge facade/door_leaf.rs:136
```

```rust
// civitas src/asset/directive.rs — the bridge (§7)
/// Lower a legacy flat directive to an equivalent Blueprint. Byte-identical
/// mesh for box-massing directives (the regression gate).
pub fn directive_to_blueprint(d: &AssetDirective) -> Blueprint;
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `blueprint::validate` | `forge-cli run` when input is a `*.bp.json` | `forge crates/cli` (dispatch by `"command":"blueprint"`) | new CLI arm beside `"building"` |
| `blueprint::evaluate` | `forge-cli` blueprint arm; returns `ForgeAsset` JSON | `forge crates/building/src/blueprint.rs` | reuses `generate_asset` tail |
| `directive_to_blueprint` | `recipe_from_directive` | `civitas game_asset_cook.rs:836` | legacy directives lower to a Box-massing blueprint; the cook then sends `{"command":"blueprint", ...}` instead of `{"command":"building"}` |
| `evaluate` SDF-derived arm | `evaluate` step 2 | `forge crates/building/src/blueprint.rs` | dual-contour the narrow-band SDF; reuses terrain extraction |
| `assert_watertight_body` | end of `evaluate` | `forge lib.rs:738` | the gate — unchanged |
| `ready_sdf` (closed proof + atoms) | cook, after Forge returns | `civitas game_asset_cook.rs:2909` | unchanged: GWN sign, `--sdf-strict` rejects open bodies |

Backward compat: the `"command":"building"` flat path stays for one release; `directive_to_blueprint` + a `blueprint_box_massing_is_byte_identical` gate (blake3 of positions vs the legacy `generate` path) proves the migration is lossless before the flat path is retired.

---

## 8. Worked Example — the CS2-class hospital

Hand-authored blueprint (abridged; full file `assets/blueprints/civic.hospital_demo.bp.json`):

```json
{
  "schema": 1, "id": "civic.hospital_demo", "seed": 7, "floor_height": 3.6,
  "style": "modern",
  "massing": {
    "op": "union",
    "a": {
      "op": "subtract", "void": true,
      "a": { "op": "volume", "id": "podium", "floors": 3,
             "shape": {"kind":"prism","footprint":[[0,0],[60,0],[60,40],[0,40]]},
             "transform": {} },
      "b": { "op": "volume", "id": "lightwell", "floors": 3,
             "shape": {"kind":"prism","footprint":[[26,16],[34,16],[34,24],[26,24]]},
             "transform": {} }
    },
    "b": { "op": "volume", "id": "tower", "floors": 9,
           "shape": {"kind":"prism","footprint":[[18,8],[42,8],[42,32],[18,32]]},
           "transform": {"translate_y": 10.8} }
  },
  "skin": [
    {"volume":"podium","floors":[0,1],"system":"storefront","bay_width":4.0,
     "windows_per_bay":1,"window_density":0.7,"material":"stone","glass_material":"vision_glass"},
    {"volume":"podium","floors":[1,3],"system":"punched_window","bay_width":3.0,
     "windows_per_bay":2,"window_density":0.55,"material":"stone"},
    {"volume":"tower","system":"curtain_wall","bay_width":1.5,"windows_per_bay":1,
     "window_density":0.92,"material":"spandrel","glass_material":"vision_glass"}
  ],
  "crown": [
    {"kit":"parapet","height":1.2,"projection":0.3},
    {"kit":"mechanical_penthouse","inset":4.0,"height":4.0,"material":"spandrel"},
    {"kit":"antenna_mast","height":12.0,"radius":0.25,"material":"steel"}
  ],
  "parts": [
    {"id":"ambulance_bay_door","kind":"door",
     "anchor":{"volume":"podium","face":"S","uv":[0.5,0.0]},
     "size":[5.0,4.0],"pivot":{"point":[30,0,0],"axis":"Y"},"material":"steel"}
  ],
  "materials": { "stone":{...}, "vision_glass":{...}, "spandrel":{...}, "steel":{...} }
}
```

**Forge execution walk-through:**
1. **Lower massing.** The `subtract{void:true}` of `lightwell` from `podium` → a ring footprint (outer 60×40 loop + inner 8×8 loop, both sealed by `offset_polygon` + `point_in_polygon`); a separable courtyard, NOT an SDF blob. The `union` of that podium with the off-center `tower` (seated at y=10.8 = 3 floors × 3.6) → two `MassingVolume`s; their footprints don't interpenetrate, so each builds as an independent sealed boundary-rep solid (`generate_stacked` loop). 5 effective volume nodes (podium-outer, lightwell-inner-loop, tower) → 2 CSG ops resolved at lowering. **Separable: no SDF fallback.**
2. **Skin.** Podium floors 0–1 → `storefront` (glazed base), floors 1–3 → `punched_window` stone; tower → `curtain_wall` with spandrel + vision glass tagged `MAT_GLASS=3` (the cook tags it transmissive). Three `build_facade_walls` calls, three+ `MaterialZone`s.
3. **Crown** on the tower (top volume): parapet box + a 4 m inset mechanical penthouse prism on the roof cap + a 12 m capsule antenna mast — each a separate coherent closed component the winding oracle protects (`winding.rs:161`).
4. **Parts.** The ambulance-bay door builds via `generate_door_leaf` as its own closed sub-mesh + a `Socket{Door, pivot point (30,0,0), axis Y}` + its own cluster — rides the mover/refit path, not fused.
5. **Assemble + gate.** `orient_outward`, `material_zones_for_mesh`, sockets, bounds, clusters, then `assert_watertight_body`: the ring-courtyard + abutting tower + free-standing crown/part components all satisfy the non-manifold + GWN gate (the courtyard's inner wall is a real sealed surface; the overlap band where the tower base meets the podium reads |w|≈2 = interior).

**Outcome — could the language express it? YES.** Podium + light-well courtyard (CSG subtract), inset curtain-wall tower (CSG union + off-center transform), mixed storefront/punched/curtain-wall skin, parapet + mechanical penthouse + antenna crown, and a pivoted ambulance door — all hand-authored, all separable, all watertight, no SDF fallback needed.

**What's missing / would stress the language:** (a) a *curved/blob* hospital wing (e.g. a swooping atrium) forces the SDF-derived path → approximate facade rhythm, the honest fidelity cost; that's where USD ingestion stays the better answer for one-off artistry (memory `civitas-usd-interchange-decision`). (b) A *cylinder* tower with a curtain wall needs `build_facade_walls` to accept a high-sided polygon (Cylinder lowers to an N-gon prism) — works, but bay rhythm on a curve needs the per-segment bay logic verified. (c) Sloped/gabled roofs over non-rectangular massing are still the deferred straight-skeleton work (`lib.rs:151-157`) — the hospital is flat-roofed so it's unaffected, but a pitched-roof bespoke form would hit it.

---

## 9. Open Questions

- [x] Mesh from SDF march, or boundary-rep? **Boundary-rep for separable CSG (the 95% case); SDF surface-extraction only for non-separable blends.** (§4.4)
- [x] Is windows-as-CSG-subtract viable? **No — apertures stay inside facade generators; CSG is massing-only.** (§4.2)
- [ ] Face-selector grammar precision: floor-band + compass faces enough, or do we need per-edge selection for asymmetric facades (e.g. a glass north face, stone elsewhere)? Leaning: add `edge_index` selector as a v2 escape hatch.
- [ ] Cylinder/N-gon bay rhythm: does `build_facade_walls` (`facade/mod.rs:27`) need a per-edge bay-fitting pass for high-sided polygons, or does the existing `bays_for_span` (`curtain_wall.rs:89`) suffice per segment? Verify before specing the Cylinder leaf.
- [ ] Determinism of the SDF-derived fallback: dual-contouring must be seed-stable and grid-aligned so re-cooks are byte-identical (the cook hashes outputs). Pin the grid origin to the volume AABB.

---

## 10. Out of Scope

- Skinned/deforming articulated parts (rigid only; deformation is the costly class, `articulated-parts-budget`).
- Pitched/gabled roofs over non-rectangular or multi-volume massing (deferred straight-skeleton work, `lib.rs:151`).
- The LLM authoring *prompt/grammar tooling* itself (how the model is steered to emit valid blueprints) — this design specs the language + executor; the authoring harness is a follow-up.
- Non-building types (terrain/road/water/vegetation) — they get their own per-type templates under the same directive-factory envelope (`forge-directive-factory`); this design is the building template.
- USD ingestion path — unchanged; remains the answer for free-form artistic one-offs (`civitas-usd-interchange-decision`).

---

## 11. Related Plans / Designs

- Depends on: the box/podium/setback massing grammar (`forge massing.rs`, `description.rs`) and the watertight oracle (`forge lib.rs:738`, `winding.rs`) — both shipped.
- Reuses: Spectra `sdf_eval.slang` CSG ops (the massing tree's semantic model + the SDF-derived fallback).
- Required before: an LLM authoring harness + the per-type directive-factory envelope expansion.
- Related: `2026-06-12-building-composition-grammar` (civitas docs/specs), memory `hybrid-atom-sdf-lod-direction`, `3d-sdf-terrain-directive`.
