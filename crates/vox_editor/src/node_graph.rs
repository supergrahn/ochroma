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
use std::time::{Duration, Instant};
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
    /// Monotonic count of how many times this node's `cook()` has actually
    /// executed. Probed by tests + the live loop to prove that clean upstream
    /// nodes are NOT re-executed and dirty downstream nodes ARE.
    cook_count:  u64,
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
    /// PCG-style live re-cook throttle. A dirty node marked since the last live
    /// cook is held as a pending trailing-edge request, keyed by the node whose
    /// param/topology changed; it only actually re-cooks once `budget` has
    /// elapsed since that subgraph last cooked. See [`request_recook`] /
    /// [`live_cook`].
    throttle: ThrottleState,
}

/// Per-subgraph trailing-edge throttle for live re-cooking.
///
/// `budget` is the minimum wall-clock gap between two cooks of the same dirty
/// subgraph. `pending` holds the most recent recook request (the root node that
/// changed) so a parameter scrubbed every frame collapses into at most
/// `ceil(window / budget)` cooks, and the LAST requested value is always the one
/// that finally cooks (trailing edge guaranteed).
struct ThrottleState {
    budget:       Duration,
    /// The root node of the dirty subgraph awaiting a cook, if any.
    pending_root: Option<NodeId>,
    /// When the subgraph rooted at `pending_root` last actually cooked.
    last_cook_at: Option<Instant>,
}

impl Default for ThrottleState {
    fn default() -> Self {
        Self { budget: Duration::from_millis(100), pending_root: None, last_cook_at: None }
    }
}

impl Default for OchromaNodeGraph {
    fn default() -> Self { Self::new() }
}

impl OchromaNodeGraph {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new(), next_id: 0, throttle: ThrottleState::default() }
    }

    pub fn add_node(&mut self, name: &str, node: Box<dyn OchromaNode>) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id, NodeEntry { name: name.to_string(), node, dirty: true, last_output: None, cook_count: 0 });
        id
    }

    /// How many times node `id`'s `cook()` has actually executed. `None` if the
    /// node does not exist. Used by tests + the live loop to assert that only the
    /// dirty subgraph re-cooks (clean nodes keep a flat count).
    pub fn cook_count(&self, id: NodeId) -> Option<u64> {
        self.nodes.get(&id).map(|e| e.cook_count)
    }

    /// Configure the minimum gap between two live re-cooks of the same dirty
    /// subgraph (the PCG-style scrub throttle). Defaults to 100ms.
    pub fn set_recook_budget(&mut self, budget: Duration) {
        self.throttle.budget = budget;
    }

    /// The configured live re-cook budget.
    pub fn recook_budget(&self) -> Duration { self.throttle.budget }

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

    /// The downstream closure of `id` (the node itself plus every node reachable
    /// from it along edges) restricted to nodes currently marked dirty. This is
    /// the exact set of nodes a live re-cook would execute — the "dirty subgraph".
    /// Returned in deterministic ascending [`NodeId`] order.
    pub fn dirty_subgraph(&self, id: NodeId) -> Vec<NodeId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            if !visited.insert(cur) { continue; }
            for edge in &self.edges {
                if edge.from == cur { stack.push(edge.to); }
            }
        }
        let mut out: Vec<NodeId> = visited
            .into_iter()
            .filter(|n| self.nodes.get(n).map(|e| e.dirty).unwrap_or(false))
            .collect();
        out.sort_unstable();
        out
    }

    /// PCG-style live param edit: set a parameter on `id`, mark its downstream
    /// subgraph dirty, and register a *trailing-edge* recook request rooted at
    /// `id`. The recook does NOT happen here — it happens on the next
    /// [`live_cook`] whose injected time has advanced past the throttle budget,
    /// so a parameter scrubbed every frame collapses into few actual cooks while
    /// the LAST value set is always the one that finally cooks.
    pub fn request_recook(&mut self, id: NodeId, key: &str, value: ParamValue) -> Result<(), GraphError> {
        self.set_param(id, key, value)?;
        // Trailing edge: always overwrite the pending root with the most recent
        // change so the final cook reflects the latest edit.
        self.throttle.pending_root = Some(id);
        Ok(())
    }

    /// Drive the live re-cook loop with an injected `now` (real `Instant::now()`
    /// in the editor, a fake clock in tests). If a recook is pending AND at least
    /// the throttle budget has elapsed since this subgraph last cooked, the dirty
    /// subgraph rooted at the pending node is cooked incrementally — clean
    /// upstream nodes are NOT re-executed — and a [`LiveCook`] report is returned.
    /// Returns `Ok(None)` when nothing was due (no pending request, or still
    /// inside the throttle window).
    pub fn live_cook(&mut self, now: Instant) -> Result<Option<LiveCook>, GraphError> {
        let Some(root) = self.throttle.pending_root else { return Ok(None) };

        // Throttle: only cook once the budget has elapsed since the last cook.
        if let Some(last) = self.throttle.last_cook_at {
            if now.duration_since(last) < self.throttle.budget {
                return Ok(None);
            }
        }

        // The subgraph we will actually execute (dirty downstream closure of root).
        let subgraph = self.dirty_subgraph(root);

        // Cook in topological order, but ONLY the dirty nodes. Clean upstream
        // nodes are skipped, so their cook_count stays flat — their cached
        // last_output is threaded into the dirty nodes via assemble_inputs.
        let dirty_set: HashSet<NodeId> = subgraph.iter().copied().collect();
        let order = self.topo_sort()?;
        let mut cooked = Vec::new();
        for id in order {
            if !dirty_set.contains(&id) { continue; }
            let inputs = self.assemble_inputs(id)?;
            let name = self.nodes[&id].name.clone();
            let output = self.nodes[&id].node.cook(inputs)
                .map_err(|e| GraphError::CookFailed { node: name.clone(), reason: e.to_string() })?;
            let entry = self.nodes.get_mut(&id).unwrap();
            entry.last_output = Some(output);
            entry.dirty = false;
            entry.cook_count += 1;
            cooked.push(id);
        }

        let root_name = self.nodes.get(&root).map(|e| e.name.clone()).unwrap_or_default();
        self.throttle.pending_root = None;
        self.throttle.last_cook_at = Some(now);

        Ok(Some(LiveCook { root, root_name, cooked }))
    }

    /// Is a trailing-edge recook currently pending (a param changed since the
    /// last live_cook flushed it)?
    pub fn has_pending_recook(&self) -> bool {
        self.throttle.pending_root.is_some()
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
            entry.cook_count += 1;
        }
        Ok(())
    }

    /// Fully evaluate the graph: topologically order every node and compute its
    /// output from its inputs threaded through the wired edges, regardless of the
    /// dirty cache. Returns an [`EvalResult`] holding every node's outputs plus the
    /// set of sink (terminal) nodes, so a caller can pull the final result(s).
    ///
    /// Unlike [`OchromaNodeGraph::cook`] (which skips clean nodes and returns `()`),
    /// `evaluate` always recomputes the whole DAG, so changing any upstream
    /// parameter is guaranteed to flow through to every downstream node's output.
    pub fn evaluate(&mut self) -> Result<EvalResult, GraphError> {
        let order = self.topo_sort()?;

        // Force a full recompute in topological order so every downstream node
        // sees the freshest upstream outputs.
        for &id in &order {
            let inputs = self.assemble_inputs(id)?;
            let name   = self.nodes[&id].name.clone();
            let output = self.nodes[&id].node.cook(inputs)
                .map_err(|e| GraphError::CookFailed { node: name.clone(), reason: e.to_string() })?;
            let entry = self.nodes.get_mut(&id).unwrap();
            entry.last_output = Some(output);
            entry.dirty = false;
            entry.cook_count += 1;
        }

        // Sink nodes = nodes with no outgoing edge; these carry the graph's results.
        let mut has_outgoing: HashSet<NodeId> = HashSet::new();
        for e in &self.edges {
            has_outgoing.insert(e.from);
        }
        let mut sinks: Vec<NodeId> = order.iter().copied()
            .filter(|id| !has_outgoing.contains(id))
            .collect();
        sinks.sort_unstable();

        let mut outputs: HashMap<NodeId, NodeOutputs> = HashMap::new();
        for &id in &order {
            if let Some(entry) = self.nodes.get(&id) {
                if let Some(out) = entry.last_output.as_ref() {
                    outputs.insert(id, out.clone());
                }
            }
        }

        Ok(EvalResult { order, sinks, outputs })
    }

    pub fn get_output(&self, id: NodeId, port: &str) -> Option<&PortData> {
        self.nodes.get(&id)?.last_output.as_ref()?.get(port)
    }

    /// Display name of a node, if it exists.
    pub fn node_name(&self, id: NodeId) -> Option<&str> {
        self.nodes.get(&id).map(|e| e.name.as_str())
    }

    /// The full set of cached outputs for a node after the last cook/evaluate,
    /// if it has cooked at least once. Used by the live preview thumbnail
    /// generator to pick a node's primary output without knowing its port names.
    pub fn node_outputs(&self, id: NodeId) -> Option<&NodeOutputs> {
        self.nodes.get(&id)?.last_output.as_ref()
    }

    /// Inspect the value that flowed through every wire after the last cook/evaluate.
    ///
    /// For each edge, looks up the upstream node's cached output for the wire's
    /// `from_port` and produces a short formatted snapshot of it (see
    /// [`format_port_data`]). Wires whose upstream output has not been computed
    /// yet are skipped. The result is the honest "data inspection" feed the UI
    /// renders as value chips: one entry per wire that actually carried data,
    /// keyed by `(from, from_port, to, to_port)`.
    pub fn wire_values(&self) -> Vec<WireValue> {
        let mut out = Vec::new();
        for e in &self.edges {
            let data = self
                .nodes
                .get(&e.from)
                .and_then(|entry| entry.last_output.as_ref())
                .and_then(|o| o.get(&e.from_port));
            if let Some(d) = data {
                out.push(WireValue {
                    from: e.from,
                    from_port: e.from_port.clone(),
                    to: e.to,
                    to_port: e.to_port.clone(),
                    value: format_port_data(d),
                });
            }
        }
        out
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
                cook_count:  0,
            });
        }
        self.throttle = ThrottleState { budget: self.throttle.budget, pending_root: None, last_cook_at: None };
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

/// The formatted snapshot of the value carried by one wire, produced by
/// [`OchromaNodeGraph::wire_values`]. `value` is a short human-readable string
/// (e.g. `"Terrain 1024 cells"`, `"Scalar 3.50"`) suitable for a UI value chip.
#[derive(Clone, Debug, PartialEq)]
pub struct WireValue {
    pub from: NodeId,
    pub from_port: String,
    pub to: NodeId,
    pub to_port: String,
    pub value: String,
}

/// Format a [`PortData`] as a compact, human-readable snapshot for wire inspection.
/// Always contains the data's kind plus a real magnitude/size cue (a count or a
/// number), so the UI chip conveys what actually flowed through the wire.
pub fn format_port_data(data: &PortData) -> String {
    match data {
        PortData::Splats(s)        => format!("Splats {}", s.len()),
        PortData::SpectralField(f) => {
            // Range notation means the actual min..max over all 16 bands —
            // not the first/last samples, which lie for any non-monotonic field.
            let (mn, mx) = f
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            format!("Spectral [{mn:.2}..{mx:.2}]")
        }
        PortData::Terrain(t)       => format!("Terrain {} cells", t.heights.len()),
        PortData::Mesh(m)          => format!("Mesh {} tris", m.indices.len()),
        PortData::LodMesh(l)       => format!("LodMesh {} levels", l.len()),
        PortData::Instances(i)     => format!("Instances {}", i.len()),
        PortData::Scalar(v)        => format!("Scalar {:.2}", v),
        PortData::BiomeMap(b)      => format!("BiomeMap {} cells", b.len()),
        PortData::SplatWeights(w)  => format!("SplatWeights {}", w.len()),
        PortData::ScalarVec(v)     => format!("ScalarVec {}", v.len()),
    }
}

/// Report of a single PCG-style incremental [`OchromaNodeGraph::live_cook`].
///
/// `root` is the node whose param/topology change triggered the recook,
/// `root_name` its display name, and `cooked` the exact set of nodes that were
/// re-executed this pass (the dirty subgraph), in topological order. Clean
/// upstream nodes are NOT in `cooked` — their cached outputs were reused.
#[derive(Clone, Debug)]
pub struct LiveCook {
    pub root:      NodeId,
    pub root_name: String,
    pub cooked:    Vec<NodeId>,
}

impl LiveCook {
    /// Number of nodes that actually re-cooked (the dirty subgraph size).
    pub fn dirty_subgraph_size(&self) -> usize { self.cooked.len() }
}

/// Result of a full graph [`OchromaNodeGraph::evaluate`] pass.
///
/// Holds the topological evaluation `order`, the `sinks` (terminal nodes with no
/// outgoing edge — the graph's final results), and every node's computed
/// `outputs` keyed by [`NodeId`].
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub order:   Vec<NodeId>,
    pub sinks:   Vec<NodeId>,
    pub outputs: HashMap<NodeId, NodeOutputs>,
}

impl EvalResult {
    /// Fetch a specific computed output port of a node.
    pub fn get(&self, id: NodeId, port: &str) -> Option<&PortData> {
        self.outputs.get(&id)?.get(port)
    }

    /// All outputs of a node, if it was evaluated.
    pub fn node_outputs(&self, id: NodeId) -> Option<&NodeOutputs> {
        self.outputs.get(&id)
    }

    /// The single terminal sink node, if the graph has exactly one. Useful when a
    /// linear pipeline produces one final result.
    pub fn sole_sink(&self) -> Option<NodeId> {
        if self.sinks.len() == 1 { Some(self.sinks[0]) } else { None }
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

    /// Build terrain -> biome, evaluate the whole graph, and verify the downstream
    /// biome_map is a real computed function of the upstream terrain output (one
    /// biome byte per terrain height cell, classified from the height value).
    #[test]
    fn evaluate_threads_terrain_into_biome_classification() {
        use crate::nodes::terrain_node::TerrainNode;
        use crate::nodes::biome_node::{BiomeNode, BiomeKind};

        let mut graph = OchromaNodeGraph::new();
        // amplitude 200 -> heights in [0, 200]; with world_height 400 no cell can
        // reach the Alpine band (norm_h >= 0.90 i.e. height >= 360).
        let terrain = graph.add_node("terrain", Box::new(TerrainNode {
            resolution: 32, amplitude: 200.0, droplet_count: 0, seed: 7, ..Default::default()
        }));
        let biome = graph.add_node("biome", Box::new(BiomeNode { world_height: 400.0, moisture: 0.5 }));
        // terrain output port "terrain" -> biome input port "terrain"
        graph.connect(terrain, "terrain", biome, "terrain").unwrap();

        let result = graph.evaluate().unwrap();

        // Topological order: terrain must precede biome.
        let pt = result.order.iter().position(|&x| x == terrain).unwrap();
        let pb = result.order.iter().position(|&x| x == biome).unwrap();
        assert!(pt < pb, "terrain must be evaluated before biome");

        // biome is the sole sink (terminal result) of this pipeline.
        assert_eq!(result.sole_sink(), Some(biome), "biome is the terminal node");

        // Downstream output exists and has exactly one biome byte per terrain cell.
        let heights = result.get(terrain, "terrain").unwrap().as_terrain().unwrap();
        let biome_map_low = result.get(biome, "biome_map").unwrap().as_biome_map().unwrap();
        assert_eq!(biome_map_low.len(), heights.heights.len(), "one biome cell per height cell");
        assert_eq!(heights.heights.len(), 32 * 32);

        // At amplitude 200 no cell reaches the Alpine band.
        let alpine_low = biome_map_low.iter().filter(|&&b| b == BiomeKind::Alpine as u8).count();
        assert_eq!(alpine_low, 0, "amplitude 200 cannot produce Alpine cells, got {}", alpine_low);

        // Now change the UPSTREAM param: raise amplitude so peaks reach the Alpine
        // band. fBm normalizes so the tallest cell equals `amplitude` exactly, and
        // 800 >= 360, guaranteeing at least one Alpine cell downstream.
        graph.set_param(terrain, "amplitude", ParamValue::Float(800.0)).unwrap();
        let result_hi = graph.evaluate().unwrap();

        let heights_hi = result_hi.get(terrain, "terrain").unwrap().as_terrain().unwrap();
        let biome_map_hi = result_hi.get(biome, "biome_map").unwrap().as_biome_map().unwrap();

        let max_height_hi = heights_hi.heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_height_hi >= 360.0, "raised amplitude should push a peak into Alpine band, max={}", max_height_hi);

        let alpine_hi = biome_map_hi.iter().filter(|&&b| b == BiomeKind::Alpine as u8).count();
        println!("alpine cells: low-amplitude={} high-amplitude={}", alpine_low, alpine_hi);
        assert!(alpine_hi > 0, "raising upstream amplitude must change downstream biome to include Alpine, got {}", alpine_hi);

        // The downstream result is genuinely a different computed value after the
        // upstream change — not merely a re-run of the same bytes.
        assert_ne!(biome_map_low, biome_map_hi, "downstream biome map must change when upstream param changes");
    }

    /// Wire data inspection (#9b): after evaluate(), the terrain->biome wire must
    /// carry a formatted snapshot of the Terrain value that flowed through it.
    #[test]
    fn wire_values_populate_terrain_into_biome_edge() {
        use crate::nodes::terrain_node::TerrainNode;
        use crate::nodes::biome_node::BiomeNode;

        let mut graph = OchromaNodeGraph::new();
        let terrain = graph.add_node("terrain", Box::new(TerrainNode {
            resolution: 32, droplet_count: 0, ..Default::default()
        }));
        let biome = graph.add_node("biome", Box::new(BiomeNode::default()));
        graph.connect(terrain, "terrain", biome, "terrain").unwrap();

        // Before evaluate, nothing has flowed.
        assert!(graph.wire_values().is_empty(), "no wire values before evaluate");

        graph.evaluate().unwrap();
        let wires = graph.wire_values();
        assert_eq!(wires.len(), 1, "exactly one wire carried data");
        let w = &wires[0];
        assert_eq!(w.from, terrain);
        assert_eq!(w.to, biome);
        assert_eq!(w.from_port, "terrain");
        assert_eq!(w.to_port, "terrain");
        // The Terrain value that flowed is 32*32 = 1024 cells; assert the real
        // formatted content, not just presence.
        assert_eq!(w.value, "Terrain 1024 cells");
        assert!(w.value.contains("1024"), "wire value must report the real cell count");
    }

    /// PCG-style incremental live re-cook: changing a Terrain param re-cooks the
    /// downstream Biome (cook_count +1, output bytes differ), while an unrelated
    /// parallel Terrain->Biome branch does NOT re-cook (cook_count unchanged).
    #[test]
    fn live_cook_recooks_only_dirty_subgraph() {
        use crate::nodes::terrain_node::TerrainNode;
        use crate::nodes::biome_node::BiomeNode;

        let mut g = OchromaNodeGraph::new();
        // Branch A: terrain_a -> biome_a (the one we'll edit).
        let terrain_a = g.add_node("terrain_a", Box::new(TerrainNode {
            resolution: 32, amplitude: 200.0, droplet_count: 0, seed: 7, ..Default::default()
        }));
        let biome_a = g.add_node("biome_a", Box::new(BiomeNode { world_height: 400.0, moisture: 0.5 }));
        g.connect(terrain_a, "terrain", biome_a, "terrain").unwrap();

        // Branch B: an independent terrain_b -> biome_b that must stay untouched.
        let terrain_b = g.add_node("terrain_b", Box::new(TerrainNode {
            resolution: 32, amplitude: 200.0, droplet_count: 0, seed: 99, ..Default::default()
        }));
        let biome_b = g.add_node("biome_b", Box::new(BiomeNode { world_height: 400.0, moisture: 0.5 }));
        g.connect(terrain_b, "terrain", biome_b, "terrain").unwrap();

        // Initial full cook establishes baselines for every node.
        g.cook().unwrap();
        let biome_a_before = g.get_output(biome_a, "biome_map").unwrap().as_biome_map().unwrap().clone();
        let cc_terrain_b_before = g.cook_count(terrain_b).unwrap();
        let cc_biome_b_before   = g.cook_count(biome_b).unwrap();
        let cc_terrain_a_before = g.cook_count(terrain_a).unwrap();
        let cc_biome_a_before   = g.cook_count(biome_a).unwrap();

        // Change terrain_a amplitude so the downstream biome classification changes.
        // Use a fresh clock far past the budget so the throttle fires immediately.
        let t0 = Instant::now();
        g.set_recook_budget(Duration::from_millis(100));
        g.request_recook(terrain_a, "amplitude", ParamValue::Float(800.0)).unwrap();

        // Dirty subgraph is exactly {terrain_a, biome_a}.
        let sub = g.dirty_subgraph(terrain_a);
        assert_eq!(sub, vec![terrain_a, biome_a], "dirty subgraph must be branch A only");

        let report = g.live_cook(t0 + Duration::from_millis(200)).unwrap().expect("a recook was due");
        assert_eq!(report.root, terrain_a);
        assert_eq!(report.dirty_subgraph_size(), 2, "only terrain_a + biome_a re-cook");

        // Branch A re-cooked: counts went up by exactly 1 and output changed.
        assert_eq!(g.cook_count(terrain_a).unwrap(), cc_terrain_a_before + 1);
        assert_eq!(g.cook_count(biome_a).unwrap(),   cc_biome_a_before + 1);
        let biome_a_after = g.get_output(biome_a, "biome_map").unwrap().as_biome_map().unwrap();
        assert_ne!(&biome_a_before, biome_a_after, "downstream biome_a output must change after upstream edit");
        let alpine_byte = crate::nodes::biome_node::BiomeKind::Alpine as u8;
        let alpine_after = biome_a_after.iter().filter(|&&b| b == alpine_byte).count();
        assert!(alpine_after > 0, "raised amplitude must push cells into Alpine, got {}", alpine_after);

        // Branch B never re-cooked: counts are byte-for-byte unchanged.
        assert_eq!(g.cook_count(terrain_b).unwrap(), cc_terrain_b_before, "unrelated terrain_b must NOT re-cook");
        assert_eq!(g.cook_count(biome_b).unwrap(),   cc_biome_b_before,   "unrelated biome_b must NOT re-cook");
    }

    /// Throttle (#3): scrub a param 10 times inside the window with a FAKE clock.
    /// Cooks happen at most ceil(window/budget) times, AND the final cooked output
    /// reflects the LAST scrubbed value (trailing edge).
    #[test]
    fn live_cook_throttles_scrub_and_keeps_last_value() {
        use crate::nodes::terrain_node::TerrainNode;
        use crate::nodes::biome_node::BiomeNode;

        let mut g = OchromaNodeGraph::new();
        let terrain = g.add_node("terrain", Box::new(TerrainNode {
            resolution: 32, amplitude: 200.0, droplet_count: 0, seed: 7, ..Default::default()
        }));
        let biome = g.add_node("biome", Box::new(BiomeNode { world_height: 400.0, moisture: 0.5 }));
        g.connect(terrain, "terrain", biome, "terrain").unwrap();
        g.cook().unwrap();

        let budget = Duration::from_millis(100);
        g.set_recook_budget(budget);
        let base = Instant::now();
        // Window = 250ms. ceil(250/100) = 3 cooks max. Scrub amplitude 10 times,
        // 25ms apart (frame-rate scrubbing). The amplitudes ramp 100..1000.
        let window = Duration::from_millis(250);
        let cc_before = g.cook_count(biome).unwrap();
        let mut cooks = 0u64;
        let last_amplitude = 1000.0f64;
        for i in 0..10u32 {
            let amp = 100.0 + (i as f64) * 100.0; // 100,200,...,1000
            let t = base + Duration::from_millis(25 * i as u64);
            g.request_recook(terrain, "amplitude", ParamValue::Float(amp)).unwrap();
            if g.live_cook(t).unwrap().is_some() {
                cooks += 1;
            }
        }
        // Flush the trailing edge: advance well past the budget so the LAST pending
        // request (amplitude 1000) definitely cooks.
        if g.has_pending_recook() {
            let t = base + window + budget + Duration::from_millis(1);
            if g.live_cook(t).unwrap().is_some() {
                cooks += 1;
            }
        }

        let ceil_window = (window.as_millis() as f64 / budget.as_millis() as f64).ceil() as u64;
        // +1 for the explicit trailing flush past the window.
        assert!(cooks <= ceil_window + 1, "cooked {} times, expected <= {}", cooks, ceil_window + 1);
        assert!(cooks >= 1, "at least one cook must happen");
        assert!(g.cook_count(biome).unwrap() > cc_before, "biome must have re-cooked at least once");
        assert!(!g.has_pending_recook(), "no recook should remain pending after the flush");

        // Trailing edge: the final cooked terrain reflects the LAST amplitude (1000),
        // proven by the tallest height equalling 1000 (fBm normalizes peak==amplitude).
        let final_terrain = g.get_output(terrain, "terrain").unwrap().as_terrain().unwrap();
        let max_h = final_terrain.heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max_h - last_amplitude as f32).abs() < 1.0,
            "final cook must reflect LAST scrubbed amplitude {}, got peak {}", last_amplitude, max_h);
    }

    #[test]
    fn live_cook_returns_none_when_nothing_pending() {
        use crate::nodes::terrain_node::TerrainNode;
        let mut g = OchromaNodeGraph::new();
        g.add_node("terrain", Box::new(TerrainNode { resolution: 32, droplet_count: 0, ..Default::default() }));
        g.cook().unwrap();
        assert!(g.live_cook(Instant::now()).unwrap().is_none(), "no pending request => no cook");
    }

    #[test]
    fn format_port_data_is_descriptive() {
        assert_eq!(format_port_data(&PortData::Scalar(3.5)), "Scalar 3.50");
        assert_eq!(format_port_data(&PortData::BiomeMap(vec![0u8; 9])), "BiomeMap 9 cells");
        let hf = HeightfieldSpatial { heights: vec![0.0; 16], resolution: 4, world_size: 1.0 };
        assert_eq!(format_port_data(&PortData::Terrain(hf)), "Terrain 16 cells");
    }
}
