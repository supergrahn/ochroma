# Design: Blueprint Visual Editor — three synced surfaces (3D viewport + node-graph + LLM chat) over the Forge node-DAG (2026-06-13)

**Status:** Draft
**Scope:** Specify the in-app VISUAL EDITOR for the Forge blueprint node-DAG: an interactive, path-traced view of an LLM-or-human-authored building wired to a Geometry-Nodes-style node-graph editor and an LLM chat, where manual edits (graph params/wiring AND direct 3D gizmo/anchor/curve manipulation) and LLM edits BOTH mutate the SAME `BlueprintGraph` (the single source of truth), the graph re-evaluates incrementally to a `ForgeAsset`, and `vox_render` re-renders the viewport. Engine tooling lives in `vox_app` (the GAME/app layer per CLAUDE.md); Forge is the deterministic builder; the LLM is a service. READ-ONLY design — no code changed.
**Related:** `2026-06-13-forge-blueprint-language-design.md` (the typed node DAG + `evaluate(graph)->ForgeAsset` + `forge_catalog()`/`get_options` + sweep curves + named anchors — authoritative), memory `llm-blueprint-forge-directive`, `forge-directive-factory`, `hybrid-atom-sdf-lod-direction`, `building-content-pipeline`.

---

## 1. Problem Statement

- The blueprint language (`2026-06-13-forge-blueprint-language-design.md`) makes a building a typed node DAG an LLM authors as JSON. But today there is **no way to SEE the building it describes interactively, nor to tweak it by hand.** The author edits JSON blind and re-cooks from the CLI (`forge-cli run … --emit asset` → `game_asset_cook`), then has to load the result in a separate render-gate test (`vox_render/src/splat_backend_tests/mesh_render.rs:5`) to see it. There is no orbit-inspect-tweak loop.
- The app shell already has the three surfaces in skeleton form but **wired to the wrong graph and to a CPU still:** the `NodeGraph` panel (`vox_app/src/shell/mod.rs:55,2233`) is bound via `GraphBridge` (`graph_bridge.rs:1`) to the *terrain/biome/vegetation* cook graph (`TEMPLATE_NAME = "Terrain → Biome → Vegetation → Splatize"`, `graph_bridge.rs:28`), not a Forge building blueprint; the `Viewport` panel (`mod.rs:2193`) draws a 480×320 CPU splat rasterization of a hard-coded demo scene (`viewport.rs:19,27,96`), not a path-traced `ForgeAsset`; the `Inspector` (`mod.rs:2089`) scrubs that graph's float params only. None of the three are connected to `blueprint::evaluate`.
- **There is no direct-manipulation path at all.** A user cannot click a wall in the 3D view and have its node highlight, cannot drag a gizmo on a volume, cannot drag a footprint anchor or a Bézier control point. The viewport is a static `egui::Image` (`mod.rs:2203`); it has no picking, no gizmos, no selection-sync to the node graph.
- **The LLM edit path is a one-shot text intent, not a graph patch.** `EditorShell::run_intent` (`mod.rs:847`) parses NL and dispatches through the `CommandRegistry` (`command_palette.rs:50,77`) — there is no loop that hands the LLM the Forge catalog + current graph, takes back a validated graph *diff*, runs it through the typed-edge + watertight validators, and re-evaluates. There is no undo across LLM + manual edits unified (today undo is `UndoEntry::ParamSet` for inspector scrubs only, `mod.rs:217,1177`).
- The live-eval loop, when it does exist for terrain (`graph_bridge.rs` `request_recook`/`live_cook`), **re-cooks the whole graph** on any param change; a building blueprint with a 96-segment swept wing + facade cladding + watertight gate must not full-re-evaluate on every gizmo-drag frame or the editor stalls.

---

## 2. Done When

Running the app and opening a blueprint:

```bash
cargo run -p vox_app --features spectra-native
# In-app: Content panel -> double-click assets/blueprints/civic.hospital_demo.bp.json
```

produces, with a human at the keyboard, ALL of the following observable outcomes (verifiable without reading code):

1. **Three synced surfaces populate from ONE graph.** The Viewport shows the path-traced hospital (podium + curtain-wall tower + curved-glass wing); the Node Graph shows 14 nodes / 19 typed wires with port-type-colored sockets; the Inspector is empty until selection. The Output log prints `[blueprint] civic.hospital_demo: 3 volume nodes … watertight OK` (the `evaluate` gate line from the language design §2).
2. **Selection syncs both ways.** Clicking the tower in the 3D viewport highlights the `tower` node (and its `tower_skin` clad node) in the Node Graph AND binds the Inspector to its params; clicking the `wing` node in the graph highlights the curved wing in 3D with a selection outline. The status bar reads `selected: tower (extrude) — 9 floors`.
3. **A manual graph edit re-renders live.** Scrubbing the `tower` node's `floors` from 9 → 14 in the Inspector makes the tower visibly grow in the viewport within ~1 s, and the Output log prints `[blueprint] re-eval: 2 dirty nodes (tower, tower_skin) refit; wing/podium reused`. The whole graph is NOT re-evaluated (the log names exactly the dirty subset).
4. **A direct-manipulation drag writes back to the graph.** Dragging the tower's height gizmo handle up in the 3D view changes the SAME `tower.floors` param — after the drag, the Inspector field reads the new value and the Node Graph node shows the new param. (The transform is not a one-off; the graph stays source of truth.)
5. **An LLM tweak mutates the graph through validation.** Typing "make the tower taller and give the wing more glass" into the Chat panel produces, in the Output log, `[llm] patch: set tower.floors=14, set wing_skin.window_density=0.98 -> validate: type-check OK, watertight OK -> re-eval 3 dirty nodes`, and the viewport updates. Typing "wire the wing curve into the podium extrude" (an illegal `Curve→`closed-curve-into-already-fed `Geometry` consumer) produces `[llm] patch REJECTED: SocketTypeMismatch{from:Curve,to:Geometry} — graph unchanged, nothing re-rendered`.
6. **Undo crosses both edit kinds.** After the manual scrub (step 3) and the LLM patch (step 5), pressing Ctrl+Z once reverts the LLM patch (tower back to 14→? and wing glass back), Ctrl+Z again reverts the manual scrub (tower back to 9) — the viewport re-renders to match at each step, and the Output log prints `[undo] reverted llm-patch (3 nodes)` then `[undo] reverted param tower.floors 14->9`.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Blueprint opens into all 3 surfaces | loading `civic.hospital_demo.bp.json` builds a `BlueprintDoc` whose `graph.nodes.len()==14`, whose `CanvasGraph` (`vox_ui::node_canvas`) has 14 `NodeView`s + 19 `WireView`s, AND whose first `evaluate` yields a `ForgeAsset` with `asset.mesh.indices.len()>0` rendered to a non-blank RGBA frame: `assert!(frame_luma_variance(&rgba) > 0.01)` | `assert!(doc.is_some())` — passes on an empty doc |
| Node-graph reflects the Forge catalog | the add-node menu lists exactly `forge_catalog().node_types` names (sourced from the live registry, language design §4.8); adding a `FacadeSystem` variant adds it to the `facade_assign.system` socket dropdown with NO editor code change: `assert_eq!(menu.node_names(), forge_catalog().node_types.iter().map(|n|&n.name).collect())` | `assert!(!menu.is_empty())` |
| Typed-edge gate in the graph UI | dragging a wire from a `Curve` output to a `Geometry` input is REFUSED at drop (the wire snaps back, no `Edge` is added to `graph.edges`) and the socket flashes the error token; the validator is `blueprint::type_check` (language design §6), not a UI re-implementation: `assert_eq!(graph.edges.len(), before)` after the illegal drop | `assert!(wire_drawn)` — passes on any drag |
| 3D pick → node highlight | a ray-cast against the rendered `ForgeAsset` triangle whose `material_id`/cluster maps to source node `tower` sets `selection.primary == NodeId(tower)` and the `CanvasGraph` node `tower` renders `NodeState::Selected`: `assert_eq!(shell.selected_node(), Some("tower"))` after `pick_at(tower_screen_px)` | `assert!(pick.is_some())` |
| Node highlight → 3D outline | selecting node `wing` in the canvas sets a viewport selection set = the swept wing's triangle range, drawn as an outline pass: `assert_eq!(viewport.selected_tri_range(), asset.node_tri_range("wing"))` | `assert!(viewport.has_selection)` |
| Gizmo drag writes the OWNING node param | dragging the tower height gizmo by +Δ sets `graph.node("tower").params["floors"]` to the value matching the new world height (NOT a transform on the mesh); after release `graph.node("tower").params["floors"] == 14` and re-eval reproduces it: `assert_eq!(graph.node("tower").param_i64("floors"), 14)` | `assert!(gizmo.dragged)` |
| Anchor / control-point drag → curve param | dragging `wing_path`'s 2nd Bézier control vertex updates `graph.node("wing_path").params["verts"][1]` and re-evaluates a different-shaped but still-watertight wing: `assert_ne!(verts_before, verts_after) && connected_components(&wing_mesh)==1` | `assert!(curve.edited)` |
| Incremental re-eval (dirty subset only) | scrubbing `tower.floors` marks `tower` + its descendants (`tower_skin`, `place_tower`, `body`, `body2`, `out`) dirty and re-evaluates ONLY those, reusing cached `wing`/`podium`/`lightwell` outputs: the eval reports `reused.contains("wing") && dirty == {tower,tower_skin,place_tower,body,body2,out}` | `assert!(reeval_ran)` |
| LLM returns a VALIDATED graph patch | the chat request "make the tower taller" yields a `GraphPatch` (a typed diff) that, applied to a clone, passes `blueprint::type_check` AND `assert_watertight_body` BEFORE it touches the live doc; a patch failing either is dropped with the reason logged and the live graph byte-unchanged: `assert_eq!(blake3(graph_before), blake3(graph_after))` on a rejected patch | `assert!(llm_response.contains("floors"))` |
| Unified undo across manual + LLM | a manual `ParamSet` then an `LlmPatch` then two Ctrl+Z restore the graph to its pre-LLM then pre-manual state byte-for-byte, re-rendering each time: `assert_eq!(blake3(graph), blake3(graph_at_step0))` after 2 undos | `assert_eq!(undo_stack.len(), 2)` |
| Live loop stays responsive | a gizmo-drag stream of 30 param-sets in 1 s coalesces (debounce) into ≤ N re-evals where N≤8 and each re-eval touches only dirty nodes; the viewport uses `SpectraRenderBackend::submit_frame`/`read_last_output` (one-frame latency, `splat_backend.rs:2700,2721`) so the UI never blocks: `assert!(reeval_count <= 8 && max_frame_block_ms < 16)` | `assert!(fps > 0.0)` |

---

## 4. Architecture

### 4.1 Where it lives — engine/game split

The visual editor is **app/GAME-layer tooling** in `vox_app/src/shell/` (CLAUDE.md: UI + game layer is `vox_app`). It composes three things that already exist as engine-agnostic services:

- **Forge** (external `~/src/forge`) is the deterministic builder. The editor calls only its public surface from the language design §6: `blueprint::validate`, `blueprint::type_check`, `blueprint::evaluate(&BlueprintGraph)->Result<ForgeAsset,ForgeError>`, `blueprint::forge_catalog()`, `blueprint::get_options(node,socket)`. The editor NEVER reimplements geometry, type rules, or the watertight gate — it calls them.
- **`vox_render`** is the renderer. The editor feeds the `ForgeAsset` mesh + material zones to the path tracer via the existing mesh-lit entry `pathtrace_mesh_lit_to_rgba` (`vox_render/src/splat_backend.rs:547`) for stills and, for the interactive loop, `SpectraRenderBackend` (`splat_backend.rs:39`, `submit_frame`/`read_last_output` at `2700`/`2721`) — the one-frame-latency resident frame. `LookPreset`/`LightRig` (`splat_backend.rs:224,292`) drive the look.
- **The LLM** is a SERVICE (an async tool-using agent call), behind a `BlueprintLlmService` trait so the model/provider is swappable and the editor depends only on the trait. It is given the Forge catalog (engine data) + current graph (game doc) + the user's text; it returns a graph patch. No LLM logic lives in Forge or `vox_render`.

This mirrors the existing pattern exactly: the `GraphBridge` (`graph_bridge.rs:1`) already maps a real builder graph onto the shared `vox_ui::node_canvas::CanvasGraph` (`graph_bridge.rs:24`) and routes param edits through a `request_recook`/`live_cook` loop. The blueprint editor is a SECOND bridge of the same shape, bound to the Forge building graph instead of the terrain graph.

### 4.2 The single source of truth — `BlueprintDoc`

One owned document holds the authoritative `BlueprintGraph` (the language-design type, `graph.rs`). Everything else is derived/cached from it. It is the ONLY thing undo/redo snapshots, the ONLY thing that serializes to `*.bp.json`, and the ONLY thing all three surfaces mutate (always via the dirty-marking mutators, never by editing a cached projection).

```text
                       ┌─────────────────────────────────────────┐
                       │  BlueprintDoc  (SINGLE SOURCE OF TRUTH)   │
                       │   graph: BlueprintGraph   (forge type)    │
                       │   dirty: DirtySet<NodeId>                 │
                       │   eval_cache: HashMap<NodeId, SocketValue>│
                       │   asset: Option<ForgeAsset>  (last eval)  │
                       │   source_map: TriRange/Cluster -> NodeId  │
                       │   history: Vec<DocEdit>  (undo/redo)      │
                       └───────────┬───────────────┬───────────────┘
        derive (read)              │ mutate (write) │            mutate (write)
   ┌───────────────────┐   ┌───────┴────────┐  ┌────┴───────────────┐
   │ 1. 3D VIEWPORT     │   │ 2. NODE GRAPH   │  │ 3. LLM CHAT         │
   │  path-traced asset │   │  CanvasGraph    │  │  BlueprintLlmService│
   │  pick + gizmos     │──▶│  add/rewire/    │◀─│  -> GraphPatch ->   │
   │  selection outline │   │  param scrub    │  │  validate -> apply  │
   └───────────────────┘   └────────────────┘  └────────────────────┘
        all three apply a DocEdit -> dirty-mark -> incremental evaluate -> re-render
```

Every mutation from any surface is funneled through ONE method (`BlueprintDoc::apply_edit(DocEdit) -> Result<(), ForgeError>`): it validates (cheap type-check for structural edits), records the edit on the history stack, marks the affected nodes + descendants dirty, and requests a re-eval. There is no path that mutates the graph without going through this funnel — that is what keeps the three surfaces consistent.

### 4.3 Surface 1 — the 3D viewport (path-traced, pickable, gizmos)

Replaces the static demo rasterization (`viewport.rs:27,96`) with a live render of the doc's `ForgeAsset`. Owns a `CameraController` (`vox_render/src/camera.rs:8`) for orbit/zoom/pan (`orbit`/`zoom`/`pan` at `camera.rs:69,72,75`); on idle it accumulates samples (cinematic 128-spp path via `SpectraRenderBackend::cinematic`, `splat_backend.rs:2612`), during interaction it drops to the realtime budget (`realtime`, 4 spp, `splat_backend.rs:2606`). The rendered RGBA is uploaded as an `egui::TextureHandle` and painted in the Viewport tab exactly as today (`mod.rs:2203`).

- **Picking:** the editor keeps a `source_map` from each `ForgeAsset` triangle range / `GeometryCluster` back to the originating node id (built during `evaluate`, see §4.6). A click does a CPU ray-cast (or a GPU id-buffer read-back) against the mesh → hit triangle → `source_map` → `NodeId` → set selection (§4.7). No new renderer feature needed for the CPU ray path; the id-buffer is an optimization.
- **Gizmos:** drawn in egui over the viewport image (a transform gizmo on the selected volume's AABB; draggable anchor dots on a footprint; Bézier control-point handles on a selected `Curve`). A gizmo handle is bound to a specific node param (§4.5) — dragging it emits a `DocEdit::SetParam` stream, NOT a mesh transform. The drag is debounced into the live loop (§4.6).
- **Selection outline:** the selected node's triangle range is drawn as an outline/tint pass over the beauty frame (a cheap screen-space edge or a second material override); this is the "highlight its node → 3D" direction.

### 4.4 Surface 2 — the node-graph editor (Geometry-Nodes style)

Reuses the existing engine-agnostic canvas `vox_ui::node_canvas::{CanvasGraph, NodeView, WireView, PortView}` (`node_canvas/mod.rs:147,63,123,44`) — the same pan/zoom/grid/minimap renderer the terrain bridge already paints (`graph_bridge.rs:24`). A new `BlueprintBridge` (sibling of `GraphBridge`) projects the `BlueprintGraph` onto a `CanvasGraph`:

- **Nodes:** each `Node` (`graph.rs` `op`) → a `NodeView`; the node title is the `op` ("extrude", "sweep_profile"); category color comes from a `category_for_op` map (extending the `category_for_kind` pattern at `graph_bridge.rs:68`) — e.g. `extrude/cylinder/setback` = Spatial, `footprint_curve/sweep_profile` = Generator, `facade_assign/crown_kit` = Field, `asset_output` = Sink.
- **Sockets + typed wires:** each typed socket (the language design §4.2 socket types: `Curve/Geometry/Profile/Anchor/Float/Vector/Int/Material/Enum`) → a `PortView` whose `PortType` drives the wire color (the canvas already colors wires by port type; extend `vox_ui::PortType` / the `map_port_type` mirror at `graph_bridge.rs:51` with the blueprint socket types, OR add a `Blueprint` port-type family). A wire drag is allowed to drop ONLY when `blueprint::type_check` would accept the edge — the validator is queried at drop time, the UI never re-implements the rule.
- **Add / remove / rewire:** an add-node menu populated from `forge_catalog().node_types` (so it can never list a node Forge can't build); removing a node removes its incident edges; rewiring is a `DocEdit::AddEdge`/`RemoveEdge`. Every structural edit re-runs `type_check` on the resulting graph before commit (cheap, no geometry) and is rejected (with the socket flashing the error token) if it would break typing or create a cycle.
- **Param editing:** clicking a node binds the Inspector (the existing `inspector()` panel, `mod.rs:2089`, and `ParamField` scrub mechanism, `graph_bridge.rs:32`) to that node's params. Param schemas come from the catalog `SocketInfo` (`enum_options` for `Enum` sockets renders a dropdown of the live variants, `get_options`, language design §4.8). Editing emits `DocEdit::SetParam`.

### 4.5 Direct-manipulation → graph-mutation mapping

The keystone of "graph stays source of truth": a gizmo never produces a transform matrix baked onto the mesh — it computes the **inverse mapping back to the owning node's authored param** and writes that. Each draggable handle declares a `ParamBinding { node: NodeId, param_path: JsonPointer, axis_to_param: fn }`:

| Direct manipulation | Owning node + param it writes | Mapping |
|---|---|---|
| Drag volume height gizmo | `extrude.floors` (Int) | new world height ÷ `graph.floor_height` → rounded floors |
| Drag volume footprint corner | `footprint_curve.verts[i]` ([x,z]) | screen drag → ground-plane XZ delta added to that control vert |
| Drag a Bézier control point | `footprint_curve.verts[i]` with `bezier:true` | same XZ projection; re-tessellation is the eval's job |
| Drag an anchor dot | the wired `place_volume.base` source — re-targets which `Anchor` socket feeds it (a `DocEdit::Rewire`), or nudges the producing node | coordinate-free: moving the snap target changes the wiring, not a literal coord (language design §4.5) |
| Drag a setback ratio handle | `setback.ratio` (Float) | handle offset ÷ floor footprint extent |

Because every handle resolves to a param-set or a rewire on a real node, the node graph and the JSON update in lockstep, the edit is a normal `DocEdit` on the undo stack, and a subsequent LLM patch sees the same numbers. Handles whose param cannot be inverted unambiguously (e.g. a free-form anchor with no governing param) are read-only (shown, not draggable) until the language exposes a param for them — an open question (§8), not a silent baked transform.

### 4.6 The live eval/render loop — incremental + debounced

A `DocEdit` does not synchronously re-cook. It marks dirty and schedules. Per frame, the shell's main loop (`EditorShell::ui`, `mod.rs:720`) drains pending edits (like `drain_requests`, `mod.rs:1313`) and runs the loop:

1. **Coalesce / debounce.** Successive `SetParam` edits to the same `(node, param)` within a drag coalesce (exactly the existing inspector drag-coalescing in `record_inspector_edit`, `mod.rs:1140`). A re-eval is scheduled at most every ~80–120 ms during a drag (and once on release at full quality).
2. **Dirty propagation.** `BlueprintDoc::dirty` is the set of edited nodes plus all topological descendants (the DAG reachability). Cached outputs of non-dirty nodes (`eval_cache`, keyed by node id + an input-hash) are reused. A param change to `tower` dirties `tower → tower_skin → place_tower → body → body2 → out`; `podium`, `lightwell`, `wing`, `wing_path` stay cached.
3. **Incremental evaluate.** Call a re-eval that evaluates only dirty nodes in topological order, pulling non-dirty inputs from `eval_cache`. (v1 may call `blueprint::evaluate` on the whole graph if Forge does not yet expose a partial-eval entry — see §8; even so the dirty set drives a *refit-vs-rebuild* decision: a pure-param change to a leaf-clad node like `facade_assign` refits cladding without rebuilding the swept wing.) The terminal `asset_output` always runs `assert_watertight_body` (language design §4.6) — the gate is never skipped.
4. **Build the source map.** During eval, each node's emitted triangle range / `GeometryCluster` is tagged with its producing `NodeId` and recorded in `source_map` (the picking + outline index). This is additive bookkeeping over the existing `material_zones_for_mesh` / cluster emission tail.
5. **Re-render.** Feed the new `ForgeAsset` to the viewport: during interaction `SpectraRenderBackend::submit_frame(Some(scene), camera)` at the realtime budget (non-blocking; read the prior frame via `read_last_output`, `splat_backend.rs:2721` — one-frame latency keeps the UI from blocking); on idle, re-submit at the cinematic budget so the still converges. Only the geometry/material layers that changed are re-uploaded (`mark_geometry_changed`/`mark_materials_changed`).

Responsiveness budget: a structural edit (add/rewire node) is rare and may pay a full eval; a param scrub is the common case and is debounced + dirty-scoped so the per-frame cost is one realtime-budget render of a small dirty subset. The UI thread never blocks on the GPU (the resident-frame backend owns its own thread, `splat_backend.rs:2748`).

### 4.7 Bidirectional selection sync

Selection lives once, in the doc/shell `Selection` (the existing multi-select, `mod.rs:129`), keyed by `NodeId` (not by mesh triangle). Both directions resolve to the same `NodeId`:

- **3D → graph:** pick → triangle → `source_map` → `NodeId` → `Selection::primary`. The `CanvasGraph` projection reads `Selection` and marks the matching `NodeView` `NodeState::Selected`; the Inspector binds to that node's params.
- **Graph → 3D:** click a `NodeView` → `NodeId` → `Selection::primary`. The viewport reads `Selection`, looks up the node's triangle range in `source_map`, and draws the outline pass.

Because selection is a single `NodeId` field both surfaces read, sync is automatic — there is no two-way mirror to keep coherent, only one value projected two ways (the same discipline as `GraphBridge`'s single live graph).

### 4.8 Surface 3 — the LLM chat / tweak loop

A new Chat panel (a 4th `PanelId`, `mod.rs:55`) hosts the conversation. A tweak round-trips:

1. **Context assembly.** The editor builds the LLM request from: (a) `forge_catalog()` serialized (the legal node types / sockets / enum options — language design §4.8, so the model composes only buildable ops), (b) the current `BlueprintGraph` as JSON (the graph design §4.3 shape), (c) the user's text, (d) optionally the current rendered viewport frame (the beauty RGBA) for visually-grounded asks like "more daylight in the atrium". The catalog + graph are the cacheable prefix.
2. **Patch, not rewrite.** The model is instructed to return a **`GraphPatch`** — a typed diff (`SetParam{node,path,value}` / `AddNode{node}` / `RemoveNode{id}` / `AddEdge{edge}` / `RemoveEdge{from,to}`), NOT a full graph. A diff is auditable, minimally invalidating (only touched nodes go dirty), and undoable as one unit. The patch is the same `DocEdit` vocabulary manual edits use (§4.2), so LLM and manual edits are literally the same mutation type on the same stack.
3. **Guardrails — validate before render.** The patch is applied to a CLONE of the graph; that clone runs `blueprint::type_check` (typed edges, required sockets, acyclicity) and, if it type-checks, a trial `blueprint::evaluate` whose terminal `assert_watertight_body` must pass. ONLY if both pass is the patch committed to the live doc (as one `DocEdit::LlmPatch(Vec<DocEdit>)` group) and rendered. A failing patch is dropped, the failure reason (`SocketTypeMismatch`, non-manifold, etc.) is shown in chat + Output log, and the model may be re-prompted with the error for a self-correction turn. The live graph is byte-unchanged on rejection (`Done When` #5).
4. **Re-eval + re-render** then run through the same incremental loop (§4.6) — the LLM patch's touched nodes are the dirty set.

This routes through the existing `CommandRegistry` philosophy (the one dispatch surface menus/palette/AI already share, `command_palette.rs:50`; `run_intent` at `mod.rs:847` is the current NL entry) — the chat is `run_intent` evolved from "parse → command" into "assemble context → LLM → validated GraphPatch → apply".

### 4.9 Unified undo / redo

The history stack stores `DocEdit`s (manual `SetParam`/`AddNode`/`Rewire` AND `LlmPatch(group)`), generalizing the existing `UndoEntry` (`mod.rs:217`, which already has a `Group` variant for multi-action, and `ParamSet` with prev/next, `mod.rs:1177`). Undo reverts the top `DocEdit` (an `LlmPatch` reverts as one group), re-marks the affected nodes dirty, re-evaluates, and re-renders — so undo is visible in the viewport, not just the data. Ctrl+Z is already wired (`mod.rs:734`); redo is the symmetric forward stack. Because both edit kinds are the same `DocEdit` type, a single linear history interleaves them correctly (`Done When` #6).

---

## 5. Data Models

```rust
// vox_app/src/shell/blueprint_editor/doc.rs (NEW — app/game layer)

/// The single source of truth for the visual editor. Owns the authoritative
/// graph; all three surfaces derive from it and mutate it only via apply_edit.
pub struct BlueprintDoc {
    graph: forge_building::blueprint::BlueprintGraph, // authoritative (language design graph.rs)
    path: Option<PathBuf>,                            // the *.bp.json file, for save
    dirty: HashSet<NodeId>,                           // nodes needing re-eval (edited + descendants)
    eval_cache: HashMap<NodeId, CachedOutput>,        // reused outputs of clean nodes
    asset: Option<ForgeAsset>,                        // last successful evaluate() result
    source_map: SourceMap,                            // ForgeAsset tri/cluster -> NodeId (picking)
    history: Vec<DocEdit>,                            // undo stack
    redo: Vec<DocEdit>,                               // redo stack
    selection: NodeId,                                // single source-of-truth selection
}

impl BlueprintDoc {
    pub fn open(json: &str) -> Result<Self, ForgeError>;   // validate + first evaluate
    pub fn apply_edit(&mut self, e: DocEdit) -> Result<(), ForgeError>; // the ONE funnel
    pub fn undo(&mut self) -> Option<()>;                  // revert top DocEdit, re-mark dirty
    pub fn redo(&mut self) -> Option<()>;
    pub fn evaluate_dirty(&mut self) -> Result<&ForgeAsset, ForgeError>; // incremental
    pub fn node_for_triangle(&self, tri: u32) -> Option<NodeId>;         // picking
    pub fn tri_range_for_node(&self, n: NodeId) -> Option<Range<u32>>;   // outline
    pub fn graph(&self) -> &BlueprintGraph;                // read-only projection source
    pub fn to_json(&self) -> String;                       // save
}

/// The unified edit vocabulary — manual AND LLM edits are the same type.
/// One DocEdit (or an LlmPatch group) is one undo step.
pub enum DocEdit {
    SetParam   { node: NodeId, path: JsonPointer, value: serde_json::Value },
    AddNode    { node: forge_building::blueprint::Node, at: egui::Pos2 },
    RemoveNode { id: NodeId },
    AddEdge    { edge: forge_building::blueprint::Edge },
    RemoveEdge { from: String, to: String },
    LlmPatch   { summary: String, edits: Vec<DocEdit> },   // one model turn, reverts as a unit
}

/// Maps rendered geometry back to its producing node, for picking + outline.
pub struct SourceMap { tri_to_node: Vec<NodeId>, node_to_tris: HashMap<NodeId, Range<u32>> }

pub struct CachedOutput { input_hash: u64, value: forge_building::blueprint::SocketValue }

/// Binds a draggable 3D handle to the node param it writes (graph = source of truth).
pub struct ParamBinding { node: NodeId, param_path: JsonPointer, kind: GizmoKind }
pub enum GizmoKind { VolumeHeight, FootprintVert(usize), BezierCtrl(usize), SetbackRatio, AnchorTarget }
```

```rust
// vox_app/src/shell/blueprint_editor/llm.rs (NEW — the LLM as a swappable service)

/// The LLM tweak service. The editor depends only on this trait; the concrete
/// impl (model/provider) is injected, so neither Forge nor vox_render depends on it.
#[async_trait]
pub trait BlueprintLlmService: Send + Sync {
    /// Given the catalog, current graph, user request, and optional beauty frame,
    /// return a typed graph patch (a diff), NOT a full graph.
    async fn propose_patch(&self, req: TweakRequest) -> Result<GraphPatch, LlmError>;
}

pub struct TweakRequest {
    pub catalog_json: String,        // forge_catalog() serialized — legal ops/sockets/enums
    pub graph_json: String,          // current BlueprintGraph
    pub user_text: String,
    pub beauty_png: Option<Vec<u8>>, // optional rendered frame for visual asks
}

/// What the model returns: a list of DocEdits + a human summary. Validated
/// (type_check + evaluate watertight) on a CLONE before it touches the live doc.
pub struct GraphPatch { pub summary: String, pub edits: Vec<DocEdit> }
```

Reused unchanged: `vox_ui::node_canvas::{CanvasGraph, NodeView, WireView, PortView}` (`node_canvas/mod.rs:147,63,123,44`), `vox_ui::{PortType, NodeCategory}` (`tokens.rs:28,54`), `EditorShell`/`Selection`/`UndoEntry`/`CommandRegistry` (`mod.rs:332,129,217`; `command_palette.rs:50`), `ParamField` (`graph_bridge.rs:32`); `vox_render::{SpectraRenderBackend, LightRig, LookPreset, CameraController}` (`splat_backend.rs:39,224,292`; `camera.rs:8`); Forge `blueprint::{validate,type_check,evaluate,forge_catalog,get_options}`, `BlueprintGraph`, `ForgeAsset` (language design §6).

---

## 6. API

```rust
// vox_app/src/shell/blueprint_editor/mod.rs (NEW)

/// Open a blueprint into the editor (validate + first evaluate + first render).
pub fn open_blueprint(path: &Path) -> Result<BlueprintDoc, ForgeError>;

/// Project the doc's graph onto the shared canvas (sibling of GraphBridge).
pub struct BlueprintBridge { /* doc ref, canvas, catalog */ }
impl BlueprintBridge {
    pub fn rebuild_canvas(&mut self, doc: &BlueprintDoc) -> CanvasGraph; // graph -> NodeViews/WireViews
    pub fn add_node_menu(&self) -> Vec<&str>;        // from forge_catalog() node_types
    pub fn enum_options(&self, node_op: &str, socket: &str) -> Vec<String>; // get_options
    /// Validate-then-apply a wire drop; refuses (returns false, no edge added) on type mismatch.
    pub fn try_connect(&mut self, doc: &mut BlueprintDoc, from: &str, to: &str) -> bool;
}

/// Pick a node from a viewport click (CPU ray-cast against the asset mesh).
pub fn pick_node(doc: &BlueprintDoc, ray: Ray) -> Option<NodeId>;

// Forge surface the editor calls (language design §6 — unchanged, READ-ONLY here):
//   blueprint::validate(json) -> Result<BlueprintGraph, ForgeError>
//   blueprint::type_check(&g) -> Result<TypeCheck, ForgeError>
//   blueprint::evaluate(&g)   -> Result<ForgeAsset, ForgeError>   (ends in assert_watertight_body)
//   blueprint::forge_catalog() -> Catalog
//   blueprint::get_options(node, socket) -> Result<Vec<String>, ForgeError>
//
// vox_render surface (unchanged):
//   SpectraRenderBackend::realtime/cinematic(w,h)         splat_backend.rs:2606,2612
//   SpectraRenderBackend::submit_frame(Option<SceneState>, CameraLayer)  splat_backend.rs:2700
//   SpectraRenderBackend::read_last_output() -> Arc<Vec<u8>>             splat_backend.rs:2721
//   pathtrace_mesh_lit_to_rgba(...)                       splat_backend.rs:547  (still fallback)
//   CameraController::{orbit,zoom,pan}                    camera.rs:69,72,75
//   LightRig / LookPreset                                 splat_backend.rs:224,292
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `open_blueprint` | Content panel double-click on `*.bp.json` | `vox_app/src/shell/content_panel.rs:72` (the existing double-click-load) | replaces/extends the asset-load path |
| `BlueprintDoc::apply_edit` | Inspector scrub, gizmo drag, wire drop, LLM patch | `vox_app/src/shell/blueprint_editor/doc.rs` (NEW) | THE single mutation funnel |
| `BlueprintBridge::rebuild_canvas` | per-frame canvas build, after any `DocEdit` | `vox_app/src/shell/blueprint_editor/mod.rs` (NEW), painted by `node_graph()` `mod.rs:2233` | sibling of `GraphBridge`/`graph_bridge.rs:24` |
| `BlueprintBridge::try_connect` | wire-drop in the Node Graph panel | same | queries `blueprint::type_check`; refuses illegal drops |
| `pick_node` | Viewport click | `vox_app/src/shell/viewport.rs` (extend; currently static image `mod.rs:2203`) | CPU ray-cast → `source_map` → `NodeId` |
| Gizmo overlay + `ParamBinding` | Viewport interaction (egui over the image) | `vox_app/src/shell/viewport.rs` (NEW gizmo module) | drag emits `DocEdit::SetParam`, never a mesh transform |
| `BlueprintDoc::evaluate_dirty` | the debounced live loop | `vox_app/src/shell/mod.rs:720` `ui()` (drains like `drain_requests` `mod.rs:1313`) | incremental; ends in `assert_watertight_body` |
| `SpectraRenderBackend::submit_frame`/`read_last_output` | the live loop, after eval | `vox_render/src/splat_backend.rs:2700,2721` | one-frame latency; UI never blocks |
| `BlueprintLlmService::propose_patch` | Chat panel send | `vox_app/src/shell/blueprint_editor/llm.rs` (NEW) | async; validated on a clone before commit |
| validate-on-clone (`type_check`+`evaluate`) | before committing any LLM `GraphPatch` | `doc.rs` `apply_edit` LlmPatch arm | guardrail: reject → live graph byte-unchanged |
| Unified undo/redo | Ctrl+Z (`mod.rs:734`) / Ctrl+Y | `doc.rs` `undo`/`redo`, generalizing `UndoEntry` `mod.rs:217,1177` | one `DocEdit`/`LlmPatch` per step; re-renders |
| New `PanelId::Chat` | dock layout | `vox_app/src/shell/mod.rs:55` (add variant), `ShellViewer::ui` `mod.rs:2036` | 4th surface |

---

## 8. Open Questions

- [ ] **Partial-eval API in Forge.** §4.6 wants to re-evaluate only dirty nodes. Does `forge_building::blueprint` expose (or will it expose) a node-level eval that accepts cached inputs, or does the editor call whole-graph `evaluate` each time and rely on Forge's own per-node memoization? If the latter, the dirty set still drives the realtime-vs-cinematic budget and the re-upload scope, but eval cost is not reduced — confirm Forge's eval is fast enough at the hospital scale, or request a `evaluate_with_cache(&g, &mut Cache)` entry. (This is the single biggest perf lever.)
- [ ] **Picking precision under instancing/clusters.** `place_part` parts and node-group instances are disjoint components (language design §4.6); the `source_map` must distinguish two instances of the same `stair_tower` group → which `NodeId` does a pick resolve to (the group node, or the instance)? Likely the `group_instance` node; confirm the cluster→node tagging carries instance identity.
- [ ] **Gizmo inverse for non-invertible params.** §4.5 makes handles read-only when no single param governs them (e.g. a swept wing's silhouette is governed by a whole Bézier net + a profile). Do we expose per-control-point drag only, or also a higher-level "scale wing" gizmo that writes multiple verts at once (a macro `DocEdit` group)? Decide before specing the gizmo set.
- [ ] **GPU id-buffer vs CPU ray-cast for picking.** v1 CPU ray-cast is simplest and needs no renderer change; a GPU triangle/cluster id buffer is faster for dense scenes but requires a `vox_render` output channel. Start CPU; measure; add the id-buffer only if picking lags.
- [ ] **LLM patch self-correction budget.** On a rejected patch (§4.8 step 3) how many auto-retry turns (re-prompt with the validator error) before surfacing failure to the user? Bound it (e.g. ≤2) to cap latency/cost.
- [ ] **Beauty-frame to the LLM: cost vs value.** Sending the rendered PNG enables visual asks ("more daylight") but adds image tokens every turn. Make it opt-in per request, or only when the user's text references appearance?
- [ ] **Multi-select edits.** The existing `Selection` is multi-select (`mod.rs:129`); should a gizmo drag on a multi-selection write the param on all selected nodes (a `DocEdit` group)? Useful for "raise all towers" but ambiguous across heterogeneous node types.

---

## 9. Out of Scope

- The Forge blueprint language itself (node types, sweep op, anchors, catalog, the `evaluate` executor, the watertight gate) — specified in `2026-06-13-forge-blueprint-language-design.md` and built in `~/src/forge`; this design only CONSUMES that surface.
- The LLM authoring HARNESS that first composes a blueprint from a blank prompt (the "author a hospital from scratch" loop) — that is the language design's deferred follow-up; this editor TWEAKS an existing graph (which may itself have been authored by that harness or by hand).
- Renderer internals (BSDFs, ReSTIR/NRC, atmosphere) — reused via `vox_render`'s public entries unchanged.
- Non-building blueprint types (terrain/road/water/vegetation node sets) — they ride the same directive-factory envelope (`forge-directive-factory`) and could reuse this editor shell later, but this design is specified against the building catalog.
- Collaborative / multi-user editing of one graph — single-user, single-doc.
- Save-format changes — the doc serializes to the existing `*.bp.json` (language design §4.3); no new on-disk format.

---

## 10. Related Plans / Designs

- Depends on: `2026-06-13-forge-blueprint-language-design.md` (the graph type, `evaluate`, `forge_catalog`/`get_options`, anchors, sweep — the entire builder surface this editor drives); the existing app shell (`vox_app/src/shell/mod.rs`), the engine-agnostic node canvas (`vox_ui::node_canvas`), and the resident-frame renderer (`vox_render` `SpectraRenderBackend`).
- Mirrors: `GraphBridge` (`vox_app/src/shell/graph_bridge.rs`) — the existing live-cook bridge for the terrain graph; `BlueprintBridge` is its building-graph sibling.
- Required before: a usable LLM-blueprint workflow end-to-end (author with the harness, then inspect + tweak here).
- Related: memory `llm-blueprint-forge-directive`, `forge-directive-factory`, `hybrid-atom-sdf-lod-direction`, `building-content-pipeline`, `facade-photorealism-roadmap`.
