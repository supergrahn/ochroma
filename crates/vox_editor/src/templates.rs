//! Graph templates (PCG / Niagara-style starter graphs).
//!
//! A [`GraphTemplate`] is pure DATA: an ordered list of node *kinds* (referenced
//! by their registry `type_name`) plus a list of typed edges between them. It
//! holds no node logic — instantiation goes through the [`NodeRegistry`] so the
//! template can never drift from the real node implementations.
//!
//! [`GraphTemplate::instantiate`] builds a fresh, correctly-connected, cookable
//! [`OchromaNodeGraph`]: every node is constructed from the registry, every edge
//! is type-checked by `graph.connect`, and the resulting graph evaluates to a
//! non-empty terminal output. The library mirrors the PCG "Terrain -> Biome ->
//! Vegetation -> Splatize" / "Building -> Plot -> Splatize" starter flows.

use crate::node_graph::{NodeId, OchromaNodeGraph};
use crate::registry::NodeRegistry;

/// One node placeholder in a template: which registry kind to construct and a
/// stable label used to name the instance in the built graph.
#[derive(Debug, Clone)]
pub struct TemplateNode {
    /// Registry `type_name` (e.g. `"TerrainNode"`).
    pub kind: &'static str,
    /// Human label for the instance (e.g. `"terrain"`).
    pub label: &'static str,
}

/// One edge in a template, referencing nodes by their index in
/// [`GraphTemplate::nodes`]. Ports are named exactly as the nodes expose them.
#[derive(Debug, Clone)]
pub struct TemplateEdge {
    pub from: usize,
    pub from_port: &'static str,
    pub to: usize,
    pub to_port: &'static str,
}

/// A named, data-only starter graph.
#[derive(Debug, Clone)]
pub struct GraphTemplate {
    pub name: &'static str,
    pub description: &'static str,
    pub nodes: Vec<TemplateNode>,
    pub edges: Vec<TemplateEdge>,
    /// Index (into `nodes`) of the terminal node whose output is the graph's
    /// result. For the splat-producing templates this is the `SplatizeNode`.
    pub terminal: usize,
    /// The output port on the terminal node carrying the final result.
    pub terminal_port: &'static str,
}

/// Errors raised while instantiating a template into a live graph.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("template references unknown node kind '{0}'")]
    UnknownKind(&'static str),
    #[error("template edge {edge} references out-of-range node index {index}")]
    BadEdgeIndex { edge: usize, index: usize },
    #[error("failed to connect template edge {edge}: {reason}")]
    ConnectFailed { edge: usize, reason: String },
}

/// The result of instantiating a template: the built graph plus the [`NodeId`]s
/// of every created node (in template order) and the terminal node id.
pub struct InstantiatedTemplate {
    pub graph: OchromaNodeGraph,
    pub node_ids: Vec<NodeId>,
    pub terminal_id: NodeId,
    pub terminal_port: &'static str,
}

impl GraphTemplate {
    /// Construct a fresh, correctly-connected, cookable graph from this template,
    /// pulling every node from `registry`. Each edge is type-checked by
    /// `graph.connect`; a type-mismatched or out-of-range edge is a hard error so
    /// a template can never produce an un-cookable graph.
    pub fn instantiate(&self, registry: &NodeRegistry) -> Result<InstantiatedTemplate, TemplateError> {
        let mut graph = OchromaNodeGraph::new();
        let mut node_ids = Vec::with_capacity(self.nodes.len());

        for tn in &self.nodes {
            let node = registry.create(tn.kind).ok_or(TemplateError::UnknownKind(tn.kind))?;
            node_ids.push(graph.add_node(tn.label, node));
        }

        for (ei, edge) in self.edges.iter().enumerate() {
            let from = *node_ids.get(edge.from).ok_or(TemplateError::BadEdgeIndex { edge: ei, index: edge.from })?;
            let to = *node_ids.get(edge.to).ok_or(TemplateError::BadEdgeIndex { edge: ei, index: edge.to })?;
            graph
                .connect(from, edge.from_port, to, edge.to_port)
                .map_err(|e| TemplateError::ConnectFailed { edge: ei, reason: e.to_string() })?;
        }

        let terminal_id = *node_ids
            .get(self.terminal)
            .ok_or(TemplateError::BadEdgeIndex { edge: usize::MAX, index: self.terminal })?;

        Ok(InstantiatedTemplate {
            graph,
            node_ids,
            terminal_id,
            terminal_port: self.terminal_port,
        })
    }
}

/// The shipped starter-template library. Returned in a stable order.
///
/// All three terminate in a [`SplatizeNode`] producing a `Splats` output, so a
/// freshly-instantiated template cooks straight to renderable Gaussian splats.
pub fn template_library() -> Vec<GraphTemplate> {
    vec![
        // 1. Terrain -> Biome -> Vegetation -> Splatize.
        //    Terrain feeds Biome classification; Vegetation grows a tree mesh that
        //    Splatize converts to spectral splats. (Vegetation has no typed input,
        //    so the biome branch runs in parallel and informs scatter context — the
        //    same shape PCG uses where a biome map gates a downstream scatter.)
        GraphTemplate {
            name: "Terrain → Biome → Vegetation → Splatize",
            description: "Classify terrain into biomes, grow vegetation, splatize to spectral Gaussians.",
            nodes: vec![
                TemplateNode { kind: "TerrainNode",    label: "terrain" },
                TemplateNode { kind: "BiomeNode",      label: "biome" },
                TemplateNode { kind: "VegetationNode", label: "vegetation" },
                TemplateNode { kind: "SplatizeNode",   label: "splatize" },
            ],
            edges: vec![
                TemplateEdge { from: 0, from_port: "terrain", to: 1, to_port: "terrain" },
                TemplateEdge { from: 2, from_port: "mesh",    to: 3, to_port: "mesh" },
            ],
            terminal: 3,
            terminal_port: "splats",
        },
        // 2. Terrain -> Biome -> SplatWeight, with Moisture feeding weights.
        //    Demonstrates the moisture-driven splat-weight branch; terminal is the
        //    Splatize of a vegetation mesh so it still yields splats.
        GraphTemplate {
            name: "Terrain → Moisture → Vegetation",
            description: "Combine drip moisture, classify biomes, weight + splatize vegetation.",
            nodes: vec![
                TemplateNode { kind: "TerrainNode",      label: "terrain" },
                TemplateNode { kind: "BiomeNode",        label: "biome" },
                TemplateNode { kind: "CatenaryNode",     label: "drip" },
                TemplateNode { kind: "MoistureNode",     label: "moisture" },
                TemplateNode { kind: "SplatWeightNode",  label: "splat_weight" },
                TemplateNode { kind: "VegetationNode",   label: "vegetation" },
                TemplateNode { kind: "SplatizeNode",     label: "splatize" },
            ],
            edges: vec![
                TemplateEdge { from: 0, from_port: "terrain",   to: 1, to_port: "terrain" },
                TemplateEdge { from: 2, from_port: "points",    to: 3, to_port: "drip" },
                TemplateEdge { from: 1, from_port: "biome_map", to: 4, to_port: "biome_map" },
                TemplateEdge { from: 3, from_port: "moisture",  to: 4, to_port: "moisture" },
                TemplateEdge { from: 5, from_port: "mesh",      to: 6, to_port: "mesh" },
            ],
            terminal: 6,
            terminal_port: "splats",
        },
        // 3. Building -> Plot -> Splatize.
        //    A plot's ground mesh is splatized; the building sits in the same graph.
        GraphTemplate {
            name: "Building → Plot → Splatize",
            description: "Lay out a plot + building, splatize the plot ground to spectral Gaussians.",
            nodes: vec![
                TemplateNode { kind: "BuildingNode", label: "building" },
                TemplateNode { kind: "PlotNode",     label: "plot" },
                TemplateNode { kind: "SplatizeNode", label: "splatize" },
            ],
            edges: vec![
                TemplateEdge { from: 1, from_port: "ground_mesh", to: 2, to_port: "mesh" },
            ],
            terminal: 2,
            terminal_port: "splats",
        },
    ]
}

/// Instantiate a template by exact `name` against `registry`. `None` if no
/// template with that name exists.
pub fn instantiate_by_name(
    registry: &NodeRegistry,
    name: &str,
) -> Option<Result<InstantiatedTemplate, TemplateError>> {
    template_library()
        .into_iter()
        .find(|t| t.name == name)
        .map(|t| t.instantiate(registry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    /// Every template instantiates: every node is constructed, every edge is
    /// type-valid (proven by `connect` succeeding inside `instantiate`), and a
    /// full cook produces a non-empty `Splats` terminal output with real,
    /// finite positions inside a sane world bound.
    #[test]
    fn every_template_instantiates_cooks_and_produces_splats() {
        let reg = NodeRegistry::new();
        let lib = template_library();
        assert_eq!(lib.len(), 3, "three starter templates ship");

        for tmpl in &lib {
            let mut inst = tmpl
                .instantiate(&reg)
                .unwrap_or_else(|e| panic!("template '{}' failed to instantiate: {e}", tmpl.name));

            // Every template node was actually created.
            assert_eq!(inst.node_ids.len(), tmpl.nodes.len(), "template '{}' node count", tmpl.name);
            // Every template edge is present in the built graph (type-checked on connect).
            assert_eq!(inst.graph.edge_count(), tmpl.edges.len(), "template '{}' edge count", tmpl.name);

            // Cook the whole DAG and pull the terminal Splats output.
            let result = inst
                .graph
                .evaluate()
                .unwrap_or_else(|e| panic!("template '{}' failed to evaluate: {e}", tmpl.name));
            let splats: &Vec<GaussianSplat> = result
                .get(inst.terminal_id, inst.terminal_port)
                .unwrap_or_else(|| panic!("template '{}' terminal has no '{}' output", tmpl.name, inst.terminal_port))
                .as_splats()
                .unwrap_or_else(|| panic!("template '{}' terminal output is not Splats", tmpl.name));

            assert!(!splats.is_empty(), "template '{}' produced zero splats", tmpl.name);

            // Real, finite positions within a generous world bound (meshes are tens
            // of units; the terrain world_size default is 1000).
            for s in splats {
                let p = s.position();
                for (axis, &c) in p.iter().enumerate() {
                    assert!(c.is_finite(), "template '{}' splat coord {axis} is non-finite: {c}", tmpl.name);
                    assert!(c.abs() <= 2000.0, "template '{}' splat coord {axis} out of world bound: {c}", tmpl.name);
                }
                // Spectral payload is non-zero (material was assigned).
                assert!(s.spectral().iter().any(|&v| v != 0), "template '{}' splat has empty spectral", tmpl.name);
            }
            println!("template '{}': {} splats", tmpl.name, splats.len());
        }
    }

    #[test]
    fn instantiate_by_name_round_trips_each_template() {
        let reg = NodeRegistry::new();
        for tmpl in template_library() {
            let inst = instantiate_by_name(&reg, tmpl.name)
                .unwrap_or_else(|| panic!("instantiate_by_name returned None for '{}'", tmpl.name))
                .unwrap_or_else(|e| panic!("instantiate_by_name('{}') errored: {e}", tmpl.name));
            assert_eq!(inst.node_ids.len(), tmpl.nodes.len());
        }
        assert!(instantiate_by_name(&reg, "no-such-template").is_none());
    }

    /// A template's edges are genuinely type-checked: a deliberately malformed
    /// template (wrong port type) must fail to instantiate, not silently build a
    /// broken graph.
    #[test]
    fn type_invalid_template_edge_is_rejected() {
        let reg = NodeRegistry::new();
        // Terrain "terrain" (Terrain) -> Splatize "mesh" (Mesh) is a type mismatch.
        let bad = GraphTemplate {
            name: "bad",
            description: "intentionally type-invalid",
            nodes: vec![
                TemplateNode { kind: "TerrainNode",  label: "terrain" },
                TemplateNode { kind: "SplatizeNode", label: "splatize" },
            ],
            edges: vec![TemplateEdge { from: 0, from_port: "terrain", to: 1, to_port: "mesh" }],
            terminal: 1,
            terminal_port: "splats",
        };
        match bad.instantiate(&reg) {
            Err(TemplateError::ConnectFailed { .. }) => {}
            Err(other) => panic!("expected ConnectFailed, got {other:?}"),
            Ok(_) => panic!("type-invalid template must not instantiate"),
        }
    }

    /// A template referencing an unknown node kind is rejected with a clear error.
    #[test]
    fn unknown_kind_template_is_rejected() {
        let reg = NodeRegistry::new();
        let bad = GraphTemplate {
            name: "bad-kind",
            description: "references a node that does not exist",
            nodes: vec![TemplateNode { kind: "NoSuchNode", label: "x" }],
            edges: vec![],
            terminal: 0,
            terminal_port: "out",
        };
        match bad.instantiate(&reg) {
            Err(TemplateError::UnknownKind("NoSuchNode")) => {}
            Err(other) => panic!("expected UnknownKind, got {other:?}"),
            Ok(_) => panic!("unknown-kind template must not instantiate"),
        }
    }
}
