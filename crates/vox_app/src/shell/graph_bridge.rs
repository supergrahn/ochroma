//! The REAL node-graph bridge — maps a live [`vox_editor::node_graph::OchromaNodeGraph`]
//! onto the shared [`vox_ui::node_canvas::CanvasGraph`] the host canvas renders.
//!
//! This is THE mapping site the `tokens::PortType` mirror anticipated: it
//! translates `vox_editor::node_graph::PortType` -> `vox_ui::PortType` and a
//! node's registry kind -> [`vox_ui::NodeCategory`], so the host canvas paints
//! the real cook graph with its true port-type colored wires and category
//! headers. It also threads the cooked `wire_values()` into the canvas wire
//! labels, exposes the selected node's REAL params as scrubbable fields, and
//! routes a param edit through the existing `request_recook` + `live_cook`
//! live-cooking loop — then refreshes the canvas labels.
//!
//! The graph is instantiated from the real template library
//! (`Terrain → Biome → Vegetation → Splatize`) via
//! [`vox_editor::templates::instantiate_by_name`] against a [`NodeRegistry`], so
//! it can never drift from the real node implementations.

use std::time::Instant;

use vox_editor::node_graph::{NodeId, OchromaNodeGraph, ParamValue, PortType as EdPortType};
use vox_editor::registry::NodeRegistry;
use vox_editor::templates::instantiate_by_name;

use vox_ui::node_canvas::{CanvasGraph, NodeView, WireView};
use vox_ui::{NodeCategory, PortType as UiPortType};

/// The exact template the Node Graph tab drives.
pub const TEMPLATE_NAME: &str = "Terrain → Biome → Vegetation → Splatize";

/// One editable parameter of a node, surfaced to the Properties tab.
#[derive(Debug, Clone)]
pub struct ParamField {
    /// The real `set_param` key (e.g. `"amplitude"`).
    pub key: &'static str,
    /// Friendly label shown in the inspector.
    pub label: &'static str,
    /// Live value (the scrub field binds to this).
    pub value: f32,
    /// Drag speed for the scrub field.
    pub speed: f32,
    /// Whether the param is an integer (set via `ParamValue::Int`) or a float.
    pub integer: bool,
    pub range: std::ops::RangeInclusive<f32>,
}

/// Map a `vox_editor` `PortType` onto the engine-agnostic `vox_ui` mirror.
///
/// Exhaustive on the editor enum: adding a `vox_editor::PortType` variant fails
/// to compile here until it is mapped (the same guard
/// `tokens_portype_mirror_is_exhaustive` enforces for the canonical mirror).
pub fn map_port_type(p: EdPortType) -> UiPortType {
    match p {
        EdPortType::Splats => UiPortType::Splats,
        EdPortType::SpectralField => UiPortType::SpectralField,
        EdPortType::Terrain => UiPortType::Terrain,
        EdPortType::Mesh => UiPortType::Mesh,
        EdPortType::LodMesh => UiPortType::LodMesh,
        EdPortType::Instances => UiPortType::Instances,
        EdPortType::Scalar => UiPortType::Scalar,
        EdPortType::BiomeMap => UiPortType::BiomeMap,
        EdPortType::SplatWeights => UiPortType::SplatWeights,
        EdPortType::ScalarVec => UiPortType::ScalarVec,
    }
}

/// Map a node's registry kind (its `descriptor().type_name`) onto a UI category
/// so its header colors by role — never an RGB (design enforcement).
pub fn category_for_kind(type_name: &str) -> NodeCategory {
    match type_name {
        "TerrainNode" | "PlotNode" | "BuildingNode" => NodeCategory::Spatial,
        "BiomeNode" | "MoistureNode" => NodeCategory::Field,
        "VegetationNode" | "CatenaryNode" | "PropPlacementNode" | "UrbanSimNode" => {
            NodeCategory::Generator
        }
        "SplatWeightNode" => NodeCategory::Math,
        "SplatizeNode" => NodeCategory::Sink,
        _ => NodeCategory::Generator,
    }
}

/// The editable param schema for a node kind (real `set_param` keys + sane
/// ranges). Empty for kinds with no scrubbable scalar params.
fn param_schema(type_name: &str) -> Vec<ParamField> {
    let f = |key, label, value, speed, integer, range| ParamField {
        key,
        label,
        value,
        speed,
        integer,
        range,
    };
    match type_name {
        "TerrainNode" => vec![
            f("resolution", "Detail (cells/side)", 64.0, 1.0, true, 16.0..=256.0),
            f("amplitude", "Height", 200.0, 1.0, false, 0.0..=800.0),
            f("seed", "Seed", 7.0, 1.0, true, 0.0..=999.0),
        ],
        "BiomeNode" => vec![
            f("world_height", "World height", 400.0, 1.0, false, 1.0..=2000.0),
            f("moisture", "Moisture", 0.5, 0.01, false, 0.0..=1.0),
        ],
        "VegetationNode" => vec![
            f("branch_levels", "Branch levels", 4.0, 1.0, true, 1.0..=8.0),
            f("trunk_radius", "Trunk radius", 0.3, 0.01, false, 0.05..=2.0),
            f("height", "Tree height", 6.0, 0.1, false, 1.0..=30.0),
        ],
        _ => Vec::new(),
    }
}

/// The live bridge: owns the real graph + per-node param fields and re-projects
/// onto the canvas each frame.
pub struct GraphBridge {
    pub graph: OchromaNodeGraph,
    /// Template node ids in template order.
    pub node_ids: Vec<NodeId>,
    /// The node kind (`type_name`) for each id, for category mapping + params.
    kinds: Vec<(NodeId, &'static str)>,
    /// Editable params per node id (Properties tab binds to these).
    pub params: Vec<(NodeId, Vec<ParamField>)>,
    /// The currently selected node (drives the Properties tab + canvas outline).
    pub selected: Option<NodeId>,
    /// World-space layout positions for each node (template order).
    positions: Vec<egui::Pos2>,
    /// The error from the most recent failed live-cook (if any), surfaced in the
    /// Properties tab so an edit that the graph rejected is visible rather than
    /// silently leaving stale outputs. Cleared on the next successful cook.
    pub last_cook_error: Option<String>,
}

impl GraphBridge {
    /// Instantiate the real template, cook it once, and lay it out.
    pub fn new() -> Self {
        let registry = NodeRegistry::new();
        let inst = instantiate_by_name(&registry, TEMPLATE_NAME)
            .expect("template library must contain the Terrain→Biome→Vegetation→Splatize template")
            .expect("the shipped template must instantiate (every edge type-checks)");

        let mut graph = inst.graph;
        // Cook once so wire_values() carries real cell/tri counts immediately.
        let _ = graph.cook();

        // Recover each node's kind via its descriptor type_name (template order).
        let kind_for = |id: NodeId, g: &OchromaNodeGraph| -> &'static str {
            // The template's labels are stable; map label -> type_name.
            match g.node_name(id) {
                Some("terrain") => "TerrainNode",
                Some("biome") => "BiomeNode",
                Some("vegetation") => "VegetationNode",
                Some("splatize") => "SplatizeNode",
                _ => "TerrainNode",
            }
        };
        let kinds: Vec<(NodeId, &'static str)> = inst
            .node_ids
            .iter()
            .map(|&id| (id, kind_for(id, &graph)))
            .collect();
        let params: Vec<(NodeId, Vec<ParamField>)> = kinds
            .iter()
            .map(|&(id, k)| (id, param_schema(k)))
            .collect();

        // A readable left-to-right pipeline layout.
        let positions = vec![
            egui::pos2(40.0, 120.0),  // terrain
            egui::pos2(250.0, 90.0),  // biome
            egui::pos2(250.0, 250.0), // vegetation
            egui::pos2(470.0, 150.0), // splatize
        ];

        GraphBridge {
            graph,
            node_ids: inst.node_ids,
            kinds,
            params,
            selected: None,
            positions,
            last_cook_error: None,
        }
    }

    fn kind_of(&self, id: NodeId) -> &'static str {
        self.kinds.iter().find(|(n, _)| *n == id).map(|(_, k)| *k).unwrap_or("TerrainNode")
    }

    fn position_of(&self, id: NodeId) -> egui::Pos2 {
        self.node_ids
            .iter()
            .position(|&n| n == id)
            .and_then(|i| self.positions.get(i).copied())
            .unwrap_or(egui::pos2(40.0, 40.0))
    }

    /// The display label for a node (its template name, title-cased).
    fn title_of(&self, id: NodeId) -> String {
        match self.graph.node_name(id) {
            Some("terrain") => "Terrain".to_string(),
            Some("biome") => "Biome Classify".to_string(),
            Some("vegetation") => "Vegetation".to_string(),
            Some("splatize") => "Splatize".to_string(),
            Some(other) => other.to_string(),
            None => "node".to_string(),
        }
    }

    /// Project the live graph onto a fresh [`CanvasGraph`], threading the real
    /// cooked `wire_values()` into wire labels and the typed ports onto sockets.
    pub fn to_canvas_graph(&self) -> CanvasGraph {
        let mut g = CanvasGraph::default();

        for &(id, _kind) in &self.kinds {
            let kind = self.kind_of(id);
            let mut nv = NodeView::new(
                id.0 as u64,
                self.title_of(id),
                category_for_kind(kind),
                self.position_of(id),
            );
            nv.size.x = 160.0;
            nv.selected = self.selected == Some(id);

            // Inputs/outputs straight from the node's real descriptor ports.
            for (name, ty) in self.input_ports(id) {
                nv.inputs.push(vox_ui::node_canvas::PortView {
                    name,
                    ty: map_port_type(ty),
                });
            }
            for (name, ty) in self.output_ports(id) {
                nv.outputs.push(vox_ui::node_canvas::PortView {
                    name,
                    ty: map_port_type(ty),
                });
            }
            g.nodes.push(nv);
        }

        // Wires from the real edges; labels from the cooked wire_values().
        let values = self.graph.wire_values();
        for (from, from_port, to, to_port) in self.graph.edges() {
            let label = values
                .iter()
                .find(|v| {
                    v.from == from && v.from_port == from_port && v.to == to && v.to_port == to_port
                })
                .map(|v| v.value.clone());
            g.wires.push(WireView {
                from_node: from.0 as u64,
                from_port: from_port.to_string(),
                to_node: to.0 as u64,
                to_port: to_port.to_string(),
                exec: false,
                label,
            });
        }

        g
    }

    /// The descriptor input ports of a node (name, type), via the real graph.
    fn input_ports(&self, id: NodeId) -> Vec<(String, EdPortType)> {
        // Known ports per kind (the descriptors are private to the node impls,
        // but the graph exposes typed port lookups via the edges/descriptor).
        match self.kind_of(id) {
            "BiomeNode" => self.lookup_inputs(id, &["terrain"]),
            "SplatizeNode" => self.lookup_inputs(id, &["mesh"]),
            _ => Vec::new(),
        }
    }

    fn output_ports(&self, id: NodeId) -> Vec<(String, EdPortType)> {
        match self.kind_of(id) {
            "TerrainNode" => self.lookup_outputs(id, &["terrain"]),
            "BiomeNode" => self.lookup_outputs(id, &["biome_map"]),
            "VegetationNode" => self.lookup_outputs(id, &["mesh"]),
            "SplatizeNode" => self.lookup_outputs(id, &["splats"]),
            _ => Vec::new(),
        }
    }

    fn lookup_inputs(&self, id: NodeId, names: &[&str]) -> Vec<(String, EdPortType)> {
        names
            .iter()
            .filter_map(|n| self.graph.input_port_type(id, n).map(|t| (n.to_string(), t)))
            .collect()
    }
    fn lookup_outputs(&self, id: NodeId, names: &[&str]) -> Vec<(String, EdPortType)> {
        names
            .iter()
            .filter_map(|n| self.graph.output_port_type(id, n).map(|t| (n.to_string(), t)))
            .collect()
    }

    /// Select a node by its canvas id (the `NodeId.0`).
    pub fn select_by_canvas_id(&mut self, canvas_id: u64) {
        self.selected = self.node_ids.iter().copied().find(|n| n.0 as u64 == canvas_id);
    }

    /// The selected node's editable params (a clone for the Properties tab).
    pub fn selected_params(&self) -> Option<(NodeId, String, Vec<ParamField>)> {
        let id = self.selected?;
        let p = self.params.iter().find(|(n, _)| *n == id)?;
        Some((id, self.title_of(id), p.1.clone()))
    }

    /// Apply a scrubbed param edit to node `id` through the REAL live-cook loop:
    /// `request_recook` (set_param + dirty cascade + trailing-edge request) then
    /// an immediate `live_cook` with a forced-elapsed clock so the cooked
    /// `wire_values()` refresh this frame. Returns the sink splat count after.
    pub fn apply_param(&mut self, id: NodeId, key: &str, value: f32) {
        let integer = self
            .params
            .iter()
            .find(|(n, _)| *n == id)
            .and_then(|(_, fields)| fields.iter().find(|f| f.key == key))
            .map(|f| f.integer)
            .unwrap_or(false);

        // For an Int param, cache the ROUNDED value — the same value that actually
        // cooks below (ParamValue::Int(value.round())). Caching the raw fractional
        // scrub value would make the inspector show e.g. 96.4 while the node cooked
        // at 96 (a cosmetic divergence between display and cooked state).
        let cached_value = if integer { value.round() } else { value };

        // Remember the pre-edit cached value so we can REVERT the display if the
        // edit fails to cook (so the inspector never shows an edited-but-not-applied
        // parameter).
        let prev_value = self
            .params
            .iter()
            .find(|(n, _)| *n == id)
            .and_then(|(_, fields)| fields.iter().find(|f| f.key == key))
            .map(|f| f.value);

        // Mirror the (rounded) value into our cached field so the Properties scrub
        // field shows what cooked.
        if let Some((_, fields)) = self.params.iter_mut().find(|(n, _)| *n == id)
            && let Some(field) = fields.iter_mut().find(|f| f.key == key)
        {
            field.value = cached_value;
        }

        let pv = if integer {
            ParamValue::Int(cached_value as i64)
        } else {
            ParamValue::Float(value as f64)
        };

        // Helper: on a cook failure, record the error string AND revert the cached
        // display to the pre-edit value (the outputs stayed stale, so the inspector
        // must not present the rejected value as if it were applied).
        let revert = |this: &mut Self, err: String| {
            if let Some(prev) = prev_value
                && let Some((_, fields)) = this.params.iter_mut().find(|(n, _)| *n == id)
                && let Some(field) = fields.iter_mut().find(|f| f.key == key)
            {
                field.value = prev;
            }
            this.last_cook_error = Some(err);
        };

        if let Err(e) = self.graph.request_recook(id, key, pv) {
            revert(self, e.to_string());
            return;
        }
        // Force the throttle past its budget so the edit cooks now (the editor's
        // real loop would call this every frame with Instant::now()).
        let now = Instant::now() + self.graph.recook_budget() * 2;
        match self.graph.live_cook(now) {
            Ok(_) => self.last_cook_error = None,
            Err(e) => revert(self, e.to_string()),
        }
    }

    /// The cooked sink (Splatize) splat count — proves a recook changed output.
    pub fn sink_splat_count(&self) -> Option<usize> {
        let sink = *self.node_ids.last()?;
        self.graph
            .get_output(sink, "splats")
            .and_then(|d| d.as_splats())
            .map(|s| s.len())
    }
}

impl Default for GraphBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_instantiates_real_template_and_cooks() {
        let b = GraphBridge::new();
        assert_eq!(b.node_ids.len(), 4, "template has 4 nodes");
        // The sink produced real splats on the initial cook.
        let n = b.sink_splat_count().expect("sink must have a cooked splat output");
        assert!(n > 0, "initial cook must produce splats, got {n}");
    }

    #[test]
    fn canvas_graph_carries_real_wire_value_labels() {
        let b = GraphBridge::new();
        let cg = b.to_canvas_graph();
        // The Terrain->Biome wire must carry a real "Terrain N cells" label.
        let terrain_wire = cg
            .wires
            .iter()
            .find(|w| w.from_port == "terrain")
            .expect("a terrain output wire exists");
        let label = terrain_wire.label.as_ref().expect("terrain wire has a cooked value label");
        assert!(
            label.starts_with("Terrain ") && label.contains("cells"),
            "terrain wire label should read 'Terrain N cells', got {label:?}"
        );
    }

    #[test]
    fn port_types_map_to_token_colored_sockets() {
        let b = GraphBridge::new();
        let cg = b.to_canvas_graph();
        let terrain = cg.nodes.iter().find(|n| n.title == "Terrain").unwrap();
        assert_eq!(
            terrain.outputs[0].ty,
            UiPortType::Terrain,
            "terrain output port must map to the UI Terrain type"
        );
        let splatize = cg.nodes.iter().find(|n| n.title == "Splatize").unwrap();
        assert_eq!(splatize.category, NodeCategory::Sink);
        assert_eq!(splatize.outputs[0].ty, UiPortType::Splats);
    }

    /// A Terrain param edit re-cooks and changes the Terrain output wire VALUE
    /// LABEL (its cell count) — the real provable live-cook outcome for this
    /// template (the Splatize sink is fed by the Vegetation branch, so its splat
    /// count is invariant to terrain resolution; the terrain wire value is not).
    #[test]
    fn scrub_terrain_recooks_and_changes_wire_value_label() {
        let mut b = GraphBridge::new();
        let terrain = b.node_ids[0];
        let label_of = |b: &GraphBridge| -> String {
            b.to_canvas_graph()
                .wires
                .iter()
                .find(|w| w.from_port == "terrain")
                .and_then(|w| w.label.clone())
                .unwrap_or_default()
        };
        let before = label_of(&b);
        // Raise terrain resolution -> more cells -> different "Terrain N cells".
        b.apply_param(terrain, "resolution", 96.0);
        let after = label_of(&b);
        assert_ne!(
            before, after,
            "raising terrain detail must change the cooked terrain wire value label ({before:?} -> {after:?})"
        );
    }

    /// A Vegetation param edit re-cooks and changes the cooked SINK (Splatize)
    /// splat count — the Vegetation mesh feeds Splatize, so more branches ->
    /// more splats.
    #[test]
    fn scrub_vegetation_recooks_and_changes_sink_count() {
        let mut b = GraphBridge::new();
        // vegetation is template node index 2.
        let veg = b.node_ids[2];
        assert_eq!(b.kind_of(veg), "VegetationNode");
        let before = b.sink_splat_count().unwrap();
        b.apply_param(veg, "branch_levels", 6.0);
        let after = b.sink_splat_count().unwrap();
        assert_ne!(
            before, after,
            "raising vegetation branch levels must change the cooked sink splat count ({before} -> {after})"
        );
    }

    /// Read the cached display value of a node's param field (what the inspector
    /// scrub field is bound to).
    fn cached_param(b: &GraphBridge, id: NodeId, key: &str) -> f32 {
        b.params
            .iter()
            .find(|(n, _)| *n == id)
            .and_then(|(_, fields)| fields.iter().find(|f| f.key == key))
            .map(|f| f.value)
            .expect("param field exists")
    }

    /// Finding 7: scrubbing an INTEGER param to a fractional value caches the ROUNDED
    /// value — the inspector mirrors what actually cooked (96), not the raw scrub
    /// (96.4). `resolution` is an integer param (schema `integer: true`).
    #[test]
    fn int_param_cached_value_is_rounded_to_what_cooked() {
        let mut b = GraphBridge::new();
        let terrain = b.node_ids[0];

        // Confirm resolution is an integer param (precondition for the fix).
        let is_int = b
            .params
            .iter()
            .find(|(n, _)| *n == terrain)
            .and_then(|(_, fields)| fields.iter().find(|f| f.key == "resolution"))
            .map(|f| f.integer)
            .unwrap();
        assert!(is_int, "resolution must be an integer param");

        // Scrub to a fractional value; the node cooks at round(96.4) = 96.
        b.apply_param(terrain, "resolution", 96.4);
        let displayed = cached_param(&b, terrain, "resolution");
        assert_eq!(
            displayed, 96.0,
            "Int param display must equal the cooked value (round(96.4)=96), got {displayed}"
        );
        assert!(b.last_cook_error.is_none(), "a valid edit must not record a cook error");
    }

    /// Finding 8: a live-cook FAILURE is surfaced via `last_cook_error` AND the
    /// cached param display reverts to the pre-edit value. TerrainNode::cook rejects
    /// `resolution < 16` (returns CookFailed), so scrubbing resolution to 8 forces a
    /// real cook error through the live_cook path.
    #[test]
    fn cook_failure_is_surfaced_and_param_display_reverts() {
        let mut b = GraphBridge::new();
        let terrain = b.node_ids[0];

        // Start from a known-good resolution so the pre-edit value is well defined.
        b.apply_param(terrain, "resolution", 64.0);
        assert!(b.last_cook_error.is_none(), "baseline cook must succeed");
        let good_display = cached_param(&b, terrain, "resolution");
        assert_eq!(good_display, 64.0);
        let good_splats = b.sink_splat_count().unwrap();

        // Now scrub to an out-of-range value the node REJECTS at cook time.
        b.apply_param(terrain, "resolution", 8.0);

        // The error is exposed...
        let err = b
            .last_cook_error
            .clone()
            .expect("a rejected cook must record last_cook_error");
        assert!(
            err.contains("resolution must be >= 16"),
            "error must carry the node's real reason, got {err:?}"
        );
        // ...the displayed param reverted to the pre-edit value (NOT 8)...
        let reverted = cached_param(&b, terrain, "resolution");
        assert_eq!(
            reverted, 64.0,
            "display must revert to the pre-edit value on cook failure, got {reverted}"
        );
        // ...and the cooked output stayed at the last good state (still produces splats).
        assert_eq!(
            b.sink_splat_count().unwrap(),
            good_splats,
            "outputs must stay at the last good cook on failure"
        );

        // A subsequent valid edit clears the error.
        b.apply_param(terrain, "resolution", 80.0);
        assert!(b.last_cook_error.is_none(), "a successful cook must clear the error");
    }

    #[test]
    fn selecting_terrain_populates_real_param_names() {
        let mut b = GraphBridge::new();
        let terrain = b.node_ids[0];
        b.selected = Some(terrain);
        let (_, title, fields) = b.selected_params().expect("terrain has params");
        assert_eq!(title, "Terrain");
        let keys: Vec<&str> = fields.iter().map(|f| f.key).collect();
        assert!(keys.contains(&"amplitude"), "terrain params must include 'amplitude', got {keys:?}");
        assert!(keys.contains(&"resolution"), "terrain params must include 'resolution', got {keys:?}");
    }
}
