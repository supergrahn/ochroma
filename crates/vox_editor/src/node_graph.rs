//! OchromaNodeGraph — procedural scene operations as a DAG.
//!
//! Direct port of CrucibleGraph from AetherSpectra with Ochroma-specific port types.
//! Key invariants:
//!   - Kahn's topological sort with BinaryHeap<Reverse<NodeId>> for determinism
//!   - Downstream dirty cascade on mark_dirty()
//!   - Idempotent duplicate-edge guard in connect()
//!   - Type-checked port connections via OchromaNode::descriptor()

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use hashbrown::HashMap;
use thiserror::Error;

use vox_core::types::GaussianSplat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortType {
    Splats,
    SpectralField,
    Terrain,
    Mesh,
    LodMesh,
    Instances,
    Scalar,
    BiomeMap,
    SplatWeights,
    ScalarVec,
}

impl std::fmt::Display for PortType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug)]
pub enum PortData {
    Splats(Vec<GaussianSplat>),
    SpectralField([f32; 16]),
    Terrain(HeightfieldSpatial),
    Mesh(EditorMesh),
    LodMesh(Vec<EditorMesh>),
    Instances(Vec<[f32; 3]>),
    Scalar(f64),
    BiomeMap(Vec<u8>),       // serialized BiomeKind per cell
    SplatWeights(Vec<[f32; 4]>),
    ScalarVec(Vec<f32>),
}

impl PortData {
    pub fn port_type(&self) -> PortType {
        match self {
            PortData::Splats(_)        => PortType::Splats,
            PortData::SpectralField(_) => PortType::SpectralField,
            PortData::Terrain(_)       => PortType::Terrain,
            PortData::Mesh(_)          => PortType::Mesh,
            PortData::LodMesh(_)       => PortType::LodMesh,
            PortData::Instances(_)     => PortType::Instances,
            PortData::Scalar(_)        => PortType::Scalar,
            PortData::BiomeMap(_)      => PortType::BiomeMap,
            PortData::SplatWeights(_)  => PortType::SplatWeights,
            PortData::ScalarVec(_)     => PortType::ScalarVec,
        }
    }
    pub fn as_terrain(&self)   -> Option<&HeightfieldSpatial> { match self { PortData::Terrain(t) => Some(t), _ => None } }
    pub fn as_mesh(&self)      -> Option<&EditorMesh>         { match self { PortData::Mesh(m) => Some(m), _ => None } }
    pub fn as_lod_mesh(&self)  -> Option<&Vec<EditorMesh>>    { match self { PortData::LodMesh(l) => Some(l), _ => None } }
    pub fn as_splats(&self)    -> Option<&Vec<GaussianSplat>>  { match self { PortData::Splats(s) => Some(s), _ => None } }
    pub fn as_scalar(&self)    -> Option<f64>                  { match self { PortData::Scalar(v) => Some(*v), _ => None } }
    pub fn as_scalar_vec(&self) -> Option<&Vec<f32>>           { match self { PortData::ScalarVec(v) => Some(v), _ => None } }
    pub fn as_biome_map(&self)  -> Option<&Vec<u8>>            { match self { PortData::BiomeMap(b) => Some(b), _ => None } }
    pub fn as_splat_weights(&self) -> Option<&Vec<[f32; 4]>>   { match self { PortData::SplatWeights(w) => Some(w), _ => None } }
}

pub type NodeInputs  = HashMap<String, PortData>;
pub type NodeOutputs = HashMap<String, PortData>;

#[derive(Clone, Debug)]
pub struct HeightfieldSpatial {
    pub heights:    Vec<f32>,
    pub resolution: u32,
    pub world_size: f32,
}

impl Default for HeightfieldSpatial {
    fn default() -> Self {
        Self { heights: Vec::new(), resolution: 0, world_size: 1000.0 }
    }
}

#[derive(Clone, Debug)]
pub struct EditorMesh {
    pub positions:   Vec<[f32; 3]>,
    pub normals:     Vec<[f32; 3]>,
    pub indices:     Vec<[u32; 3]>,
    pub material_id: u32,
}

impl EditorMesh {
    pub fn new() -> Self {
        Self { positions: Vec::new(), normals: Vec::new(), indices: Vec::new(), material_id: 0 }
    }
}

impl Default for EditorMesh {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Error)]
pub enum NodeError {
    #[error("missing required input port: {0}")]
    MissingInput(String),
    #[error("type mismatch on port {0}")]
    TypeMismatch(String),
    #[error("cook failed: {0}")]
    CookFailed(String),
    #[error("unknown parameter: {0}")]
    UnknownParam(String),
}

pub struct PortSpec {
    pub name:      &'static str,
    pub port_type: PortType,
    pub optional:  bool,
}

pub struct NodeDescriptor {
    pub type_name: &'static str,
    pub inputs:    Vec<PortSpec>,
    pub outputs:   Vec<PortSpec>,
}

impl NodeDescriptor {
    pub fn output_type(&self, port: &str) -> Option<PortType> {
        self.outputs.iter().find(|p| p.name == port).map(|p| p.port_type)
    }
    pub fn input_type(&self, port: &str) -> Option<PortType> {
        self.inputs.iter().find(|p| p.name == port).map(|p| p.port_type)
    }
}

pub trait OchromaNode: Send + Sync {
    fn descriptor(&self) -> NodeDescriptor;
    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError>;
    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError>;
    /// Deep-clone this node (including all its parameters) into a fresh box.
    /// Used by [`OchromaNodeGraph::snapshot`] / [`OchromaNodeGraph::restore`] so a
    /// graph can be saved and restored to an identical state (nodes + params + edges).
    fn clone_box(&self) -> Box<dyn OchromaNode>;
}

#[derive(Debug, Clone)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("node {0:?} not found")]
    NodeNotFound(NodeId),
    #[error("cycle detected: {from} → {to}")]
    CycleDetected { from: u32, to: u32 },
    #[error("type mismatch on port {port}: expected {expected}, got {got}")]
    TypeMismatch { port: String, expected: String, got: String },
    #[error("unknown port {port} on node {node:?}")]
    UnknownPort { node: NodeId, port: String },
    #[error("cook failed at node {node}: {reason}")]
    CookFailed { node: String, reason: String },
}

struct NodeEntry {
    name:        String,
    node:        Box<dyn OchromaNode>,
    dirty:       bool,
    last_output: Option<NodeOutputs>,
}

struct Edge {
    from:      NodeId,
    from_port: String,
    to:        NodeId,
    to_port:   String,
}

pub struct OchromaNodeGraph {
    nodes:   HashMap<NodeId, NodeEntry>,
    edges:   Vec<Edge>,
    next_id: u32,
}

impl Default for OchromaNodeGraph {
    fn default() -> Self { Self::new() }
}

impl OchromaNodeGraph {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new(), next_id: 0 }
    }

    pub fn add_node(&mut self, name: &str, node: Box<dyn OchromaNode>) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id, NodeEntry { name: name.to_string(), node, dirty: true, last_output: None });
        id
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    /// Number of edges currently in the graph.
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// Iterate the graph's edges as `(from, from_port, to, to_port)` tuples.
    /// Used by the editor panel to draw wires and by tests to assert connectivity.
    pub fn edges(&self) -> impl Iterator<Item = (NodeId, &str, NodeId, &str)> + '_ {
        self.edges.iter().map(|e| (e.from, e.from_port.as_str(), e.to, e.to_port.as_str()))
    }

    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&id) { return Err(GraphError::NodeNotFound(id)); }
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
        Ok(())
    }

    pub fn is_dirty(&self, id: NodeId) -> bool {
        self.nodes.get(&id).map(|e| e.dirty).unwrap_or(false)
    }

    pub fn connect(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&from) { return Err(GraphError::NodeNotFound(from)); }
        if !self.nodes.contains_key(&to)   { return Err(GraphError::NodeNotFound(to));   }
        let from_type = {
            let desc = self.nodes[&from].node.descriptor();
            match desc.output_type(from_port) {
                Some(t) => t,
                None    => return Err(GraphError::UnknownPort { node: from, port: from_port.to_string() }),
            }
        };
        {
            let desc = self.nodes[&to].node.descriptor();
            if let Some(expected) = desc.input_type(to_port) {
                if from_type != expected {
                    return Err(GraphError::TypeMismatch {
                        port: to_port.to_string(),
                        expected: expected.to_string(),
                        got: from_type.to_string(),
                    });
                }
            }
        }
        if self.can_reach(to, from) {
            return Err(GraphError::CycleDetected { from: from.0, to: to.0 });
        }
        if self.edges.iter().any(|e| e.from == from && e.from_port == from_port && e.to == to && e.to_port == to_port) {
            return Ok(());
        }
        self.edges.push(Edge { from, from_port: from_port.to_string(), to, to_port: to_port.to_string() });
        Ok(())
    }

    pub fn topo_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj:       HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.entry(id).or_insert(0);
            adj.entry(id).or_default();
        }
        for e in &self.edges {
            *in_degree.entry(e.to).or_insert(0) += 1;
            adj.entry(e.from).or_default().push(e.to);
        }
        let mut queue: BinaryHeap<Reverse<NodeId>> = in_degree.iter()
            .filter(|&(_, &d)| d == 0)
            .map(|(&id, _)| Reverse(id))
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(Reverse(node)) = queue.pop() {
            order.push(node);
            if let Some(succs) = adj.get(&node) {
                for &s in succs {
                    let d = in_degree.entry(s).or_insert(0);
                    *d = d.saturating_sub(1);
                    if *d == 0 { queue.push(Reverse(s)); }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(GraphError::CookFailed { node: "topo_sort".into(), reason: "cycle detected".into() });
        }
        Ok(order)
    }

    pub fn mark_dirty(&mut self, id: NodeId) {
        let mut stack = vec![id];
        let mut visited: HashSet<NodeId> = HashSet::new();
        while let Some(cur) = stack.pop() {
            if !visited.insert(cur) { continue; }
            if let Some(e) = self.nodes.get_mut(&cur) { e.dirty = true; }
            for edge in &self.edges {
                if edge.from == cur { stack.push(edge.to); }
            }
        }
    }

    pub fn mark_clean_all(&mut self) {
        for e in self.nodes.values_mut() { e.dirty = false; }
    }

    pub fn set_param(&mut self, id: NodeId, key: &str, value: ParamValue) -> Result<(), GraphError> {
        let entry = self.nodes.get_mut(&id).ok_or(GraphError::NodeNotFound(id))?;
        entry.node.set_param(key, value).map_err(|e| GraphError::CookFailed { node: entry.name.clone(), reason: e.to_string() })?;
        self.mark_dirty(id);
        Ok(())
    }

    pub fn cook(&mut self) -> Result<(), GraphError> {
        let order = self.topo_sort()?;
        for id in order {
            if !self.nodes.get(&id).map(|e| e.dirty).unwrap_or(false) { continue; }
            let inputs = self.assemble_inputs(id)?;
            let name   = self.nodes[&id].name.clone();
            let output = self.nodes[&id].node.cook(inputs).map_err(|e| GraphError::CookFailed { node: name.clone(), reason: e.to_string() })?;
            let entry = self.nodes.get_mut(&id).unwrap();
            entry.last_output = Some(output);
            entry.dirty = false;
        }
        Ok(())
    }

    pub fn get_output(&self, id: NodeId, port: &str) -> Option<&PortData> {
        self.nodes.get(&id)?.last_output.as_ref()?.get(port)
    }

    fn assemble_inputs(&self, id: NodeId) -> Result<NodeInputs, GraphError> {
        let mut inputs = NodeInputs::new();
        for e in &self.edges {
            if e.to != id { continue; }
            let data = self.nodes.get(&e.from)
                .and_then(|entry| entry.last_output.as_ref())
                .and_then(|out| out.get(&e.from_port))
                .cloned();
            if let Some(d) = data {
                inputs.insert(e.to_port.clone(), d);
            } else {
                return Err(GraphError::CookFailed {
                    node:   format!("{:?}", id),
                    reason: format!("upstream {:?} has no output for '{}'", e.from, e.from_port),
                });
            }
        }
        Ok(inputs)
    }

    fn can_reach(&self, from: NodeId, target: NodeId) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![from];
        while let Some(cur) = stack.pop() {
            if cur == target { return true; }
            if !visited.insert(cur) { continue; }
            for e in &self.edges {
                if e.from == cur { stack.push(e.to); }
            }
        }
        false
    }

    /// Capture the full graph state — every node (deep-cloned with its params),
    /// every edge, and the id allocator — so it can be restored identically later.
    ///
    /// The returned [`GraphSnapshot`] carries both a serializable description
    /// (`nodes` / `edges`, suitable for `to_json`) and the live cloned node boxes
    /// (`node_states`) needed for a faithful in-process round-trip via [`restore`].
    pub fn snapshot(&self) -> GraphSnapshot {
        let nodes = self.nodes.iter().map(|(id, entry)| NodeSnapshot {
            id: id.0,
            name: entry.name.clone(),
            type_name: entry.node.descriptor().type_name.to_string(),
            params: serde_json::Value::Null,
        }).collect();
        let edges = self.edges.iter().map(|e| EdgeSnapshot {
            from: e.from.0, from_port: e.from_port.clone(),
            to: e.to.0, to_port: e.to_port.clone(),
        }).collect();
        let node_states = self.nodes.iter().map(|(id, entry)| NodeState {
            id:   id.0,
            name: entry.name.clone(),
            node: entry.node.clone_box(),
        }).collect();
        GraphSnapshot { nodes, edges, node_states, next_id: self.next_id }
    }

    /// Restore the graph to exactly the state captured by [`snapshot`]: the same
    /// nodes (with identical params), the same edges, and the same id allocator.
    ///
    /// After `g.restore(g.snapshot())` the graph is byte-for-byte equivalent to the
    /// original as far as ids, edges and node parameters are concerned. All restored
    /// nodes are marked dirty so the next `cook()` recomputes their outputs.
    pub fn restore(&mut self, snap: GraphSnapshot) -> Result<(), NodeError> {
        self.nodes.clear();
        self.edges.clear();
        for state in snap.node_states {
            self.nodes.insert(NodeId(state.id), NodeEntry {
                name:        state.name,
                node:        state.node,
                dirty:       true,
                last_output: None,
            });
        }
        for e in &snap.edges {
            self.edges.push(Edge {
                from:      NodeId(e.from),
                from_port: e.from_port.clone(),
                to:        NodeId(e.to),
                to_port:   e.to_port.clone(),
            });
        }
        self.next_id = snap.next_id.max(
            self.nodes.keys().map(|id| id.0 + 1).max().unwrap_or(0),
        );
        Ok(())
    }
}

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub id:        u32,
    pub name:      String,
    pub type_name: String,
    pub params:    serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct EdgeSnapshot {
    pub from: u32, pub from_port: String,
    pub to: u32,   pub to_port:   String,
}

/// Live node state captured by [`OchromaNodeGraph::snapshot`] for in-process
/// round-tripping. Not serializable — it carries an actual cloned trait object.
pub struct NodeState {
    pub id:   u32,
    pub name: String,
    pub node: Box<dyn OchromaNode>,
}

#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub nodes: Vec<NodeSnapshot>,
    pub edges: Vec<EdgeSnapshot>,
    /// Next free node id at the time of the snapshot — restored verbatim so a
    /// restored graph keeps allocating ids exactly where the original left off.
    #[serde(default)]
    pub next_id: u32,
    /// Cloned live nodes. Skipped during (de)serialization (trait objects are not
    /// serializable); only populated when produced by [`OchromaNodeGraph::snapshot`].
    #[serde(skip)]
    pub node_states: Vec<NodeState>,
}

impl GraphSnapshot {
    pub fn to_json(&self) -> Result<String, serde_json::Error> { serde_json::to_string_pretty(self) }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}

/// Test helpers exposed so other modules (editor_panel) can use pass_node
pub mod tests_helpers {
    use super::*;

    pub struct PassNode;
    impl OchromaNode for PassNode {
        fn descriptor(&self) -> NodeDescriptor {
            NodeDescriptor {
                type_name: "pass",
                inputs:  vec![PortSpec { name: "in",  port_type: PortType::Scalar, optional: true  }],
                outputs: vec![PortSpec { name: "out", port_type: PortType::Scalar, optional: false }],
            }
        }
        fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), NodeError> {
            Err(NodeError::UnknownParam(key.into()))
        }
        fn cook(&self, _: NodeInputs) -> Result<NodeOutputs, NodeError> {
            let mut out = NodeOutputs::new();
            out.insert("out".into(), PortData::Scalar(1.0));
            Ok(out)
        }
        fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(PassNode) }
    }

    pub fn pass_node() -> Box<dyn OchromaNode> { Box::new(PassNode) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tests_helpers::*;

    fn pass() -> Box<dyn OchromaNode> { pass_node() }

    #[test]
    fn add_node_returns_unique_ids() {
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", pass());
        let b = g.add_node("b", pass());
        assert_ne!(a, b);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn connect_cycle_rejected() {
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", pass());
        let b = g.add_node("b", pass());
        g.connect(a, "out", b, "in").unwrap();
        let err = g.connect(b, "out", a, "in").unwrap_err();
        assert!(matches!(err, GraphError::CycleDetected { .. }), "got: {:?}", err);
    }

    #[test]
    fn duplicate_connect_is_idempotent() {
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", pass());
        let b = g.add_node("b", pass());
        g.connect(a, "out", b, "in").unwrap();
        g.connect(a, "out", b, "in").unwrap();
        let order = g.topo_sort().unwrap();
        assert_eq!(order.len(), 2);
        g.cook().unwrap();
        assert!(g.get_output(b, "out").is_some());
    }

    #[test]
    fn topo_sort_respects_dependency() {
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", pass());
        let b = g.add_node("b", pass());
        g.connect(a, "out", b, "in").unwrap();
        let order = g.topo_sort().unwrap();
        let pa = order.iter().position(|&x| x == a).unwrap();
        let pb = order.iter().position(|&x| x == b).unwrap();
        assert!(pa < pb, "a must come before b");
    }

    #[test]
    fn mark_dirty_cascades_transitive() {
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", pass());
        let b = g.add_node("b", pass());
        let c = g.add_node("c", pass());
        g.connect(a, "out", b, "in").unwrap();
        g.connect(b, "out", c, "in").unwrap();
        g.mark_clean_all();
        g.mark_dirty(a);
        assert!(g.is_dirty(b), "b should be dirty");
        assert!(g.is_dirty(c), "c should be dirty");
    }

    #[test]
    fn cook_skips_clean_nodes() {
        use std::sync::{Arc, Mutex};
        let count = Arc::new(Mutex::new(0u32));
        struct CountNode(Arc<Mutex<u32>>);
        impl OchromaNode for CountNode {
            fn descriptor(&self) -> NodeDescriptor {
                NodeDescriptor {
                    type_name: "count",
                    inputs: vec![],
                    outputs: vec![PortSpec { name: "out", port_type: PortType::Scalar, optional: false }],
                }
            }
            fn set_param(&mut self, k: &str, _: ParamValue) -> Result<(), NodeError> {
                Err(NodeError::UnknownParam(k.into()))
            }
            fn cook(&self, _: NodeInputs) -> Result<NodeOutputs, NodeError> {
                *self.0.lock().unwrap() += 1;
                let mut out = NodeOutputs::new();
                out.insert("out".into(), PortData::Scalar(1.0));
                Ok(out)
            }
            fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(CountNode(self.0.clone())) }
        }
        let mut g = OchromaNodeGraph::new();
        g.add_node("n", Box::new(CountNode(count.clone())));
        g.cook().unwrap();
        g.cook().unwrap();
        assert_eq!(*count.lock().unwrap(), 1, "node should cook exactly once");
    }

    #[test]
    fn type_mismatch_on_connect_is_rejected() {
        struct TerrainOutNode;
        impl OchromaNode for TerrainOutNode {
            fn descriptor(&self) -> NodeDescriptor {
                NodeDescriptor {
                    type_name: "terrain_out",
                    inputs: vec![],
                    outputs: vec![PortSpec { name: "terrain", port_type: PortType::Terrain, optional: false }],
                }
            }
            fn set_param(&mut self, k: &str, _: ParamValue) -> Result<(), NodeError> { Err(NodeError::UnknownParam(k.into())) }
            fn cook(&self, _: NodeInputs) -> Result<NodeOutputs, NodeError> { Ok(NodeOutputs::new()) }
            fn clone_box(&self) -> Box<dyn OchromaNode> { Box::new(TerrainOutNode) }
        }
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", Box::new(TerrainOutNode));
        let b = g.add_node("b", pass());
        let err = g.connect(a, "terrain", b, "in").unwrap_err();
        assert!(matches!(err, GraphError::TypeMismatch { .. }), "got: {:?}", err);
    }

    #[test]
    fn test_snapshot_round_trip() {
        use crate::nodes::terrain_node::TerrainNode;
        let mut graph = OchromaNodeGraph::new();
        graph.add_node("terrain", Box::new(TerrainNode::default()));
        let snap = graph.snapshot();
        let json = snap.to_json().unwrap();
        println!("snapshot JSON: {}", json);
        let restored = GraphSnapshot::from_json(&json).unwrap();
        assert_eq!(restored.nodes.len(), 1);
        assert_eq!(restored.nodes[0].type_name, "TerrainNode");
    }

    #[test]
    fn test_undo_restores_params() {
        use crate::nodes::terrain_node::TerrainNode;
        let mut graph = OchromaNodeGraph::new();
        // Start at resolution 32 (1024 heights), no erosion for determinism.
        let id = graph.add_node("terrain", Box::new(TerrainNode { resolution: 32, droplet_count: 0, ..Default::default() }));
        let snap_before = graph.snapshot();

        // Mutate the param to a different value and confirm it took effect.
        graph.set_param(id, "resolution", ParamValue::Int(64)).unwrap();
        graph.cook().unwrap();
        let mutated = graph.get_output(id, "terrain").unwrap().as_terrain().unwrap();
        assert_eq!(mutated.heights.len(), 64 * 64, "mutation must change resolution");

        // Undo: restore the pre-mutation snapshot. The restored node must carry
        // the ORIGINAL resolution (32), not the mutated 64.
        graph.restore(snap_before).unwrap();
        graph.cook().unwrap();
        let restored = graph.get_output(id, "terrain").unwrap().as_terrain().unwrap();
        assert_eq!(restored.heights.len(), 32 * 32, "restore must roll resolution back to 32");
        assert_eq!(restored.resolution, 32);
    }

    #[test]
    fn restore_round_trips_nodes_edges_and_params() {
        use crate::nodes::terrain_node::TerrainNode;
        use crate::nodes::biome_node::BiomeNode;

        // Build a graph with >=2 nodes and a real edge: terrain -> biome.
        let mut graph = OchromaNodeGraph::new();
        let terrain = graph.add_node("terrain", Box::new(TerrainNode { resolution: 48, seed: 7, droplet_count: 0, ..Default::default() }));
        let biome   = graph.add_node("biome",   Box::new(BiomeNode { world_height: 123.0, moisture: 0.25 }));
        graph.connect(terrain, "terrain", biome, "terrain").unwrap();

        // Sanity: original graph cooks and produces a biome map sized by resolution.
        graph.cook().unwrap();
        let orig_biome = graph.get_output(biome, "biome_map").unwrap().as_biome_map().unwrap().clone();
        assert_eq!(orig_biome.len(), 48 * 48);

        let snap = graph.snapshot();
        // Snapshot must describe BOTH nodes and the single edge.
        assert_eq!(snap.nodes.len(), 2);
        assert_eq!(snap.edges.len(), 1);
        assert_eq!(snap.edges[0].from, terrain.0);
        assert_eq!(snap.edges[0].to, biome.0);

        // Vandalize the graph: drop everything, then add an unrelated node so the
        // id allocator and topology genuinely differ from the snapshot.
        graph.remove_node(terrain).unwrap();
        graph.remove_node(biome).unwrap();
        graph.add_node("junk", Box::new(BiomeNode::default()));
        assert_eq!(graph.node_count(), 1);

        // Restore: the graph must come back IDENTICAL — same node ids, same edge,
        // same params (proven by an identical biome map after re-cook).
        graph.restore(snap).unwrap();
        assert_eq!(graph.node_count(), 2, "both nodes restored");

        let mut restored_ids: Vec<u32> = graph.node_ids().map(|n| n.0).collect();
        restored_ids.sort();
        assert_eq!(restored_ids, vec![terrain.0, biome.0], "node ids preserved");

        graph.cook().unwrap();
        let restored_biome = graph.get_output(biome, "biome_map").unwrap().as_biome_map().unwrap();
        // Same terrain seed/resolution AND same biome params => byte-identical map.
        assert_eq!(restored_biome, &orig_biome, "edge + params must survive restore identically");

        // The edge truly reconnects: with the edge present, biome has its terrain
        // input. Remove it and biome cooks would fail on missing input.
        assert_eq!(restored_biome.len(), 48 * 48);
    }
}
