//! Subgraphs / graph functions — collapse a selection of nodes into one
//! reusable [`SubgraphNode`], and the inverse re-inlining via [`expand_subgraph`].
//!
//! This mirrors UE Material Functions / PCG subgraphs and Unity Sub Graph: a
//! [`SubgraphDef`] is pure data — a named inner [`OchromaNodeGraph`] plus a
//! declared *interface* of exposed input/output ports. Each exposed port promotes
//! one inner `(node, port)` to an outer-facing port whose [`PortType`] is taken
//! verbatim from the inner port, so a [`SubgraphNode`]'s descriptor is type-correct
//! *by construction* and can be wired exactly like any built-in node.
//!
//! ## The load-bearing invariant
//!
//! For ANY valid selection, the graph BEFORE [`collapse_to_subgraph`] and AFTER it
//! produce identical [`OchromaNodeGraph::evaluate`] results at every surviving sink.
//! [`expand_subgraph`] is the exact inverse. Selections whose collapse would create
//! a cycle through the new node (a path that leaves the selection and re-enters it)
//! are REJECTED with [`SubgraphError::WouldCycle`]; the graph is left unchanged.
//!
//! ## Cooking
//!
//! [`SubgraphNode::cook`] clones the inner graph, feeds the exposed inputs onto the
//! promoted inner ports through tiny injected source nodes, evaluates the inner DAG,
//! and reads the exposed outputs back out. Nested subgraphs work (a subgraph whose
//! inner graph contains another [`SubgraphNode`]); a recursion-depth guard
//! ([`MAX_SUBGRAPH_DEPTH`]) turns runaway nesting into a typed
//! [`NodeError::CookFailed`] instead of a stack overflow.

use std::cell::Cell;

use hashbrown::{HashMap, HashSet};

use crate::node_graph::{
    NodeDescriptor, NodeError, NodeId, NodeInputs, NodeOutputs, OchromaNode,
    OchromaNodeGraph, ParamValue, PortData, PortSpec, PortType,
};

/// Maximum subgraph nesting depth before [`SubgraphNode::cook`] errors with a typed
/// [`NodeError::CookFailed`] rather than overflowing the stack.
pub const MAX_SUBGRAPH_DEPTH: u32 = 32;

thread_local! {
    /// Recursion-depth counter for the *current thread's* nested-cook chain.
    /// `OchromaNode::cook` runs synchronously on one thread, so a thread-local
    /// counter incremented on entry and decremented on exit faithfully tracks how
    /// deep THIS thread's nested-subgraph cook is — without a process-wide atomic
    /// that would conflate concurrent cooks on other threads.
    static COOK_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Errors raised while collapsing/expanding a selection.
#[derive(Debug, thiserror::Error)]
pub enum SubgraphError {
    #[error("selection is empty")]
    EmptySelection,
    #[error("selected node {0:?} does not exist in the graph")]
    NodeNotFound(NodeId),
    #[error("selected node {0:?} is duplicated in the selection")]
    DuplicateSelection(NodeId),
    #[error(
        "collapsing this selection would create a cycle: external node {external:?} both \
         consumes a selection output and feeds a selection input"
    )]
    WouldCycle { external: NodeId },
    #[error("internal graph error: {0}")]
    Graph(#[from] crate::node_graph::GraphError),
    #[error("node error: {0}")]
    Node(#[from] NodeError),
    #[error("node {0:?} is not a SubgraphNode and cannot be expanded")]
    NotASubgraph(NodeId),
}

/// One promoted port on a subgraph's interface: an outer-facing name + its type,
/// bound to a specific inner `(node, port)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposedPort {
    /// Outer-facing port name on the [`SubgraphNode`] (unique within the interface).
    pub outer_name: String,
    /// Inner node this port is bound to.
    pub inner_node: NodeId,
    /// Port name on the inner node.
    pub inner_port: String,
    /// The port's data type — copied verbatim from the inner port so the
    /// [`SubgraphNode`] descriptor is type-correct by construction.
    pub port_type: PortType,
}

/// A named, reusable inner graph plus its declared I/O interface.
///
/// Pure data + a live inner [`OchromaNodeGraph`]. Constructible programmatically
/// (build the inner graph, declare exposed ports) or via [`collapse_to_subgraph`].
pub struct SubgraphDef {
    /// Registry/display name; becomes the [`SubgraphNode`]'s `type_name`.
    pub name: String,
    /// The inner graph. Cloned (via `snapshot`/`restore`) every cook so the def
    /// itself is never mutated and nested cooks don't alias state.
    pub inner: OchromaNodeGraph,
    /// Inputs exposed to the outer graph (outer input -> inner `(node, port)`).
    pub inputs: Vec<ExposedPort>,
    /// Outputs exposed to the outer graph (inner `(node, port)` -> outer output).
    pub outputs: Vec<ExposedPort>,
}

impl SubgraphDef {
    /// Deep-clone this def, including a faithful clone of the inner graph (nodes +
    /// params + edges) via snapshot/restore.
    pub fn deep_clone(&self) -> SubgraphDef {
        let mut inner = OchromaNodeGraph::new();
        inner
            .restore(self.inner.snapshot())
            .expect("inner graph restore from its own snapshot is infallible");
        SubgraphDef {
            name: self.name.clone(),
            inner,
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }

    /// The leaked `&'static str` form of the name, needed because
    /// [`NodeDescriptor::type_name`] is `&'static str`. A subgraph's name is created
    /// once and lives for the program; leaking it is the standard pattern for
    /// dynamically-named node kinds.
    fn static_type_name(&self) -> &'static str {
        // SAFETY of intent: subgraph defs are long-lived editor objects; the leaked
        // name is small and bounded by the number of distinct subgraph kinds.
        Box::leak(self.name.clone().into_boxed_str())
    }
}

/// A node that evaluates an inner [`SubgraphDef`]: descriptor ports derive from the
/// def's interface; `cook` feeds the exposed inputs into the inner graph, evaluates
/// it, and reads the exposed outputs.
pub struct SubgraphNode {
    def: SubgraphDef,
    /// Stable static type name (the def's name, leaked once at construction).
    type_name: &'static str,
}

impl SubgraphNode {
    /// Wrap a [`SubgraphDef`] as a cookable node.
    pub fn new(def: SubgraphDef) -> Self {
        let type_name = def.static_type_name();
        SubgraphNode { def, type_name }
    }

    /// The inner def (read-only). Used by [`expand_subgraph`] to re-inline.
    pub fn def(&self) -> &SubgraphDef {
        &self.def
    }
}

impl OchromaNode for SubgraphNode {
    fn descriptor(&self) -> NodeDescriptor {
        let inputs = self
            .def
            .inputs
            .iter()
            .map(|e| PortSpec {
                name: leak_str(&e.outer_name),
                port_type: e.port_type,
                optional: false,
            })
            .collect();
        let outputs = self
            .def
            .outputs
            .iter()
            .map(|e| PortSpec {
                name: leak_str(&e.outer_name),
                port_type: e.port_type,
                optional: false,
            })
            .collect();
        NodeDescriptor {
            type_name: self.type_name,
            inputs,
            outputs,
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        // Namespaced "inner_node_label.param" routes the param to an inner node by
        // its display label. Unqualified keys are rejected so a typo never silently
        // no-ops.
        let (label, inner_key) = key
            .split_once('.')
            .ok_or_else(|| NodeError::UnknownParam(key.into()))?;
        let target = self
            .def
            .inner
            .node_ids()
            .find(|&id| self.def.inner.node_name(id) == Some(label));
        match target {
            Some(id) => self
                .def
                .inner
                .set_param(id, inner_key, value)
                .map_err(|e| NodeError::CookFailed(e.to_string())),
            None => Err(NodeError::UnknownParam(key.into())),
        }
    }

    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        // Recursion-depth guard: increment on entry, ALWAYS decrement on exit.
        let depth = COOK_DEPTH.with(|d| {
            let next = d.get() + 1;
            d.set(next);
            next
        });
        let _guard = DepthGuard;
        if depth > MAX_SUBGRAPH_DEPTH {
            return Err(NodeError::CookFailed(format!(
                "subgraph nesting exceeded MAX_SUBGRAPH_DEPTH ({MAX_SUBGRAPH_DEPTH})"
            )));
        }

        // Clone the inner graph so this cook never mutates the def and nested cooks
        // don't alias one another.
        let mut inner = OchromaNodeGraph::new();
        inner
            .restore(self.def.inner.snapshot())
            .map_err(|e| NodeError::CookFailed(e.to_string()))?;

        // For each exposed input, inject a constant source node that emits the
        // supplied PortData on the inner port the input is bound to, then wire it
        // onto that inner port. This feeds outer inputs into the inner DAG without
        // mutating the bound node's logic.
        for exposed in &self.def.inputs {
            let data = inputs.get(&exposed.outer_name).ok_or_else(|| {
                NodeError::MissingInput(exposed.outer_name.clone())
            })?;
            if data.port_type() != exposed.port_type {
                return Err(NodeError::TypeMismatch(exposed.outer_name.clone()));
            }
            let src = inner.add_node(
                &format!("__in_{}", exposed.outer_name),
                Box::new(ConstSource {
                    port: exposed.inner_port.clone(),
                    value: data.clone(),
                }),
            );
            inner
                .connect(src, &exposed.inner_port, exposed.inner_node, &exposed.inner_port)
                .map_err(|e| NodeError::CookFailed(e.to_string()))?;
        }

        let result = inner
            .evaluate()
            .map_err(|e| NodeError::CookFailed(e.to_string()))?;

        // Read the exposed outputs back out.
        let mut out = NodeOutputs::new();
        for exposed in &self.def.outputs {
            let data = result
                .get(exposed.inner_node, &exposed.inner_port)
                .ok_or_else(|| {
                    NodeError::CookFailed(format!(
                        "exposed output '{}' (inner {:?}.{}) produced nothing",
                        exposed.outer_name, exposed.inner_node, exposed.inner_port
                    ))
                })?
                .clone();
            out.insert(exposed.outer_name.clone(), data);
        }
        Ok(out)
    }

    fn clone_box(&self) -> Box<dyn OchromaNode> {
        Box::new(SubgraphNode::new(self.def.deep_clone()))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// RAII guard that decrements [`COOK_DEPTH`] when a subgraph cook returns (success
/// or error), so a failed nested cook never leaves the counter inflated.
struct DepthGuard;
impl Drop for DepthGuard {
    fn drop(&mut self) {
        COOK_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// A trivial inner source node that emits a fixed [`PortData`] on a named port.
/// Used to feed a [`SubgraphNode`]'s exposed inputs into the inner graph.
#[derive(Clone)]
struct ConstSource {
    port: String,
    value: PortData,
}

impl OchromaNode for ConstSource {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "__SubgraphInput",
            inputs: vec![],
            outputs: vec![PortSpec {
                name: leak_str(&self.port),
                port_type: self.value.port_type(),
                optional: false,
            }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), NodeError> {
        Err(NodeError::UnknownParam(key.into()))
    }
    fn cook(&self, _: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let mut out = NodeOutputs::new();
        out.insert(self.port.clone(), self.value.clone());
        Ok(out)
    }
    fn clone_box(&self) -> Box<dyn OchromaNode> {
        Box::new(self.clone())
    }
}

/// Leak a `String` into a `&'static str`. Used for descriptor port names, which the
/// [`NodeDescriptor`] type requires to be `'static`. Bounded by the number of
/// distinct subgraph ports authored in a session.
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// The result of [`collapse_to_subgraph`]: the def that was extracted and the id of
/// the single [`SubgraphNode`] that replaced the selection in the outer graph.
#[derive(Debug, Clone)]
pub struct Collapsed {
    pub def_name: String,
    pub node_id: NodeId,
}

/// Collapse `selection` (a set of node ids) into one [`SubgraphNode`] named `name`.
///
/// Extracts the induced subgraph (selected nodes + edges with BOTH endpoints in the
/// selection), computes the boundary:
///
/// - every edge from OUTSIDE into a selected node becomes an *exposed input*
///   (outer source -> SubgraphNode input -> inner bound port),
/// - every edge from a selected node OUT to a non-selected node becomes an
///   *exposed output*.
///
/// then replaces the selection with one [`SubgraphNode`] wired identically.
///
/// REJECTS (leaving the graph fully unchanged) selections whose collapse would
/// introduce a cycle through the new node — i.e. some external node both consumes a
/// selection output and (transitively, in the outer graph) feeds a selection input.
pub fn collapse_to_subgraph(
    graph: &mut OchromaNodeGraph,
    selection: &[NodeId],
    name: &str,
) -> Result<Collapsed, SubgraphError> {
    if selection.is_empty() {
        return Err(SubgraphError::EmptySelection);
    }
    let mut sel: HashSet<NodeId> = HashSet::new();
    for &id in selection {
        if graph.node_name(id).is_none() {
            return Err(SubgraphError::NodeNotFound(id));
        }
        if !sel.insert(id) {
            return Err(SubgraphError::DuplicateSelection(id));
        }
    }

    // Classify edges relative to the selection.
    // - internal: both endpoints selected            -> rebuilt inside the def
    // - incoming: from outside into a selected node   -> exposed INPUT
    // - outgoing: from selected to outside            -> exposed OUTPUT
    let mut internal: Vec<(NodeId, String, NodeId, String)> = Vec::new();
    let mut incoming: Vec<(NodeId, String, NodeId, String)> = Vec::new();
    let mut outgoing: Vec<(NodeId, String, NodeId, String)> = Vec::new();
    for (from, fp, to, tp) in graph.edges() {
        let from_in = sel.contains(&from);
        let to_in = sel.contains(&to);
        match (from_in, to_in) {
            (true, true) => internal.push((from, fp.into(), to, tp.into())),
            (false, true) => incoming.push((from, fp.into(), to, tp.into())),
            (true, false) => outgoing.push((from, fp.into(), to, tp.into())),
            (false, false) => {}
        }
    }

    // Cycle-through-selection detection.
    //
    // After collapse the whole selection becomes ONE node X. A cycle forms iff some
    // external node E is BOTH:
    //   (a) a consumer of the selection — reachable from an `outgoing` target, and
    //   (b) an ancestor of the selection — can reach an `incoming` source,
    // because then X -> ... -> E -> ... -> X. Equivalently: any `incoming` SOURCE
    // (or an external ancestor of it) that is reachable from any `outgoing` TARGET.
    //
    // We compute, over the outer graph restricted to EXTERNAL nodes, the set
    // reachable forward from every outgoing target, and check whether it contains
    // any incoming source. (Internal nodes collapse into X, so paths through them
    // are exactly the X self-loop we are testing for.)
    let external_reachable = external_forward_closure(
        graph,
        &sel,
        outgoing.iter().map(|(_, _, to, _)| *to),
    );
    for (src, _, _, _) in &incoming {
        if external_reachable.contains(src) {
            return Err(SubgraphError::WouldCycle { external: *src });
        }
    }

    // Build the inner graph: clone every selected node, remembering the mapping from
    // outer id -> inner id so we can rebuild internal edges and bind exposed ports.
    let mut inner = OchromaNodeGraph::new();
    let mut outer_to_inner: HashMap<NodeId, NodeId> = HashMap::new();
    // Deterministic order: ascending outer id.
    let mut sel_sorted: Vec<NodeId> = sel.iter().copied().collect();
    sel_sorted.sort_unstable();
    for &outer in &sel_sorted {
        let label = graph.node_name(outer).unwrap_or("node").to_string();
        let node = graph
            .clone_node(outer)
            .expect("selected node exists, just validated");
        let inner_id = inner.add_node(&label, node);
        outer_to_inner.insert(outer, inner_id);
    }
    for (from, fp, to, tp) in &internal {
        let inner_from = outer_to_inner[from];
        let inner_to = outer_to_inner[to];
        inner.connect(inner_from, fp, inner_to, tp)?;
    }

    // Exposed inputs: one per distinct (inner bound node, inner port) that an
    // incoming edge targets. Multiple outer edges to the same inner port share one
    // exposed input. Outer name = "label__port" (deterministic, unique by port id).
    let mut input_index: HashMap<(NodeId, String), usize> = HashMap::new();
    let mut exposed_inputs: Vec<ExposedPort> = Vec::new();
    // Map each incoming edge's (outer source, source port) so we can rewire it.
    // exposed input outer_name -> list of (outer_from, outer_from_port).
    let mut input_wiring: Vec<(String, NodeId, String)> = Vec::new();
    for (from, fp, to, tp) in &incoming {
        let inner_bound = outer_to_inner[to];
        let key = (inner_bound, tp.clone());
        let outer_name = if let Some(&idx) = input_index.get(&key) {
            exposed_inputs[idx].outer_name.clone()
        } else {
            let label = inner.node_name(inner_bound).unwrap_or("node");
            let outer_name = format!("{label}__{tp}");
            let port_type = port_type_of_input(&inner, inner_bound, tp)?;
            input_index.insert(key, exposed_inputs.len());
            exposed_inputs.push(ExposedPort {
                outer_name: outer_name.clone(),
                inner_node: inner_bound,
                inner_port: tp.clone(),
                port_type,
            });
            outer_name
        };
        input_wiring.push((outer_name, *from, fp.clone()));
    }

    // Exposed outputs: one per distinct (inner source node, source port) that an
    // outgoing edge originates from. Outer name = "label__port".
    let mut output_index: HashMap<(NodeId, String), usize> = HashMap::new();
    let mut exposed_outputs: Vec<ExposedPort> = Vec::new();
    // outgoing rewiring: exposed output outer_name -> (outer_to, outer_to_port).
    let mut output_wiring: Vec<(String, NodeId, String)> = Vec::new();
    for (from, fp, to, tp) in &outgoing {
        let inner_src = outer_to_inner[from];
        let key = (inner_src, fp.clone());
        let outer_name = if let Some(&idx) = output_index.get(&key) {
            exposed_outputs[idx].outer_name.clone()
        } else {
            let label = inner.node_name(inner_src).unwrap_or("node");
            let outer_name = format!("{label}__{fp}");
            let port_type = port_type_of_output(&inner, inner_src, fp)?;
            output_index.insert(key, exposed_outputs.len());
            exposed_outputs.push(ExposedPort {
                outer_name: outer_name.clone(),
                inner_node: inner_src,
                inner_port: fp.clone(),
                port_type,
            });
            outer_name
        };
        output_wiring.push((outer_name, *to, tp.clone()));
    }

    let def = SubgraphDef {
        name: name.to_string(),
        inner,
        inputs: exposed_inputs,
        outputs: exposed_outputs,
    };

    // Mutate the outer graph: add the SubgraphNode, rewire boundary edges, remove
    // the selected nodes (which drops all their edges). Order matters: add + rewire
    // BEFORE removing, so sources/targets still exist.
    let sub_id = graph.add_node(name, Box::new(SubgraphNode::new(def)));

    for (outer_name, outer_from, outer_from_port) in &input_wiring {
        graph.connect(*outer_from, outer_from_port, sub_id, outer_name)?;
    }
    for (outer_name, outer_to, outer_to_port) in &output_wiring {
        graph.connect(sub_id, outer_name, *outer_to, outer_to_port)?;
    }

    for &outer in &sel_sorted {
        graph.remove_node(outer)?;
    }

    Ok(Collapsed {
        def_name: name.to_string(),
        node_id: sub_id,
    })
}

/// Re-inline the [`SubgraphNode`] at `sub_id` back into `graph`: its inner nodes are
/// added back, internal edges restored, and the boundary edges (which currently land
/// on the SubgraphNode's promoted ports) are reconnected directly to the inner bound
/// ports. The SubgraphNode is then removed. The exact inverse of
/// [`collapse_to_subgraph`].
pub fn expand_subgraph(
    graph: &mut OchromaNodeGraph,
    sub_id: NodeId,
) -> Result<Vec<NodeId>, SubgraphError> {
    // Deep-clone the def out so we drop the immutable borrow on `graph` before we
    // start mutating it (re-adding inner nodes, rewiring, removing the sub node).
    let def = graph
        .subgraph_def(sub_id)
        .ok_or(SubgraphError::NotASubgraph(sub_id))?
        .deep_clone();

    // Snapshot the boundary edges currently touching the SubgraphNode.
    // incoming-to-sub: (outer_from, outer_from_port, sub_input_name)
    // outgoing-from-sub: (sub_output_name, outer_to, outer_to_port)
    let mut to_sub: Vec<(NodeId, String, String)> = Vec::new();
    let mut from_sub: Vec<(String, NodeId, String)> = Vec::new();
    for (from, fp, to, tp) in graph.edges() {
        if to == sub_id {
            to_sub.push((from, fp.into(), tp.into()));
        } else if from == sub_id {
            from_sub.push((fp.into(), to, tp.into()));
        }
    }

    // Re-create the inner nodes in the outer graph, mapping inner id -> new outer id.
    let mut inner_to_outer: HashMap<NodeId, NodeId> = HashMap::new();
    let mut new_ids: Vec<NodeId> = Vec::new();
    let mut inner_ids: Vec<NodeId> = def.inner.node_ids().collect();
    inner_ids.sort_unstable();
    for inner_id in &inner_ids {
        let label = def.inner.node_name(*inner_id).unwrap_or("node").to_string();
        let node = def
            .inner
            .clone_node(*inner_id)
            .expect("inner node exists");
        let new_id = graph.add_node(&label, node);
        inner_to_outer.insert(*inner_id, new_id);
        new_ids.push(new_id);
    }

    // Restore internal edges.
    let internal: Vec<(NodeId, String, NodeId, String)> = def
        .inner
        .edges()
        .map(|(f, fp, t, tp)| (f, fp.to_string(), t, tp.to_string()))
        .collect();
    // Resolve exposed-port bindings before we drop the borrow on `def`.
    let input_bind: HashMap<String, (NodeId, String)> = def
        .inputs
        .iter()
        .map(|e| (e.outer_name.clone(), (e.inner_node, e.inner_port.clone())))
        .collect();
    let output_bind: HashMap<String, (NodeId, String)> = def
        .outputs
        .iter()
        .map(|e| (e.outer_name.clone(), (e.inner_node, e.inner_port.clone())))
        .collect();

    for (f, fp, t, tp) in &internal {
        graph.connect(inner_to_outer[f], fp, inner_to_outer[t], tp)?;
    }

    // Reconnect boundary edges to the inner bound ports.
    for (outer_from, outer_from_port, sub_input_name) in &to_sub {
        let (inner_node, inner_port) = &input_bind[sub_input_name];
        let dst = inner_to_outer[inner_node];
        graph.connect(*outer_from, outer_from_port, dst, inner_port)?;
    }
    for (sub_output_name, outer_to, outer_to_port) in &from_sub {
        let (inner_node, inner_port) = &output_bind[sub_output_name];
        let src = inner_to_outer[inner_node];
        graph.connect(src, inner_port, *outer_to, outer_to_port)?;
    }

    graph.remove_node(sub_id)?;
    Ok(new_ids)
}

/// Forward reachability over EXTERNAL (non-selected) nodes only, starting from the
/// given seed nodes. Edges that enter the selection are NOT traversed (the selection
/// is the single collapsed node X; traversing into it is the self-loop we test for).
fn external_forward_closure(
    graph: &OchromaNodeGraph,
    sel: &HashSet<NodeId>,
    seeds: impl Iterator<Item = NodeId>,
) -> HashSet<NodeId> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = seeds.filter(|n| !sel.contains(n)).collect();
    while let Some(cur) = stack.pop() {
        if sel.contains(&cur) {
            continue;
        }
        if !visited.insert(cur) {
            continue;
        }
        for (from, _, to, _) in graph.edges() {
            if from == cur && !sel.contains(&to) {
                stack.push(to);
            }
        }
    }
    visited
}

fn port_type_of_input(
    graph: &OchromaNodeGraph,
    node: NodeId,
    port: &str,
) -> Result<PortType, SubgraphError> {
    graph
        .input_port_type(node, port)
        .ok_or(SubgraphError::NodeNotFound(node))
}

fn port_type_of_output(
    graph: &OchromaNodeGraph,
    node: NodeId,
    port: &str,
) -> Result<PortType, SubgraphError> {
    graph
        .output_port_type(node, port)
        .ok_or(SubgraphError::NodeNotFound(node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::{OchromaNodeGraph, ParamValue};
    use crate::nodes::biome_node::BiomeNode;
    use crate::nodes::splat_weight_node::SplatWeightNode;
    use crate::nodes::terrain_node::TerrainNode;
    use crate::registry::NodeRegistry;

    /// Build a deterministic Terrain -> Biome -> SplatWeight pipeline. Terrain is
    /// erosion-free + seeded so every run is byte-identical.
    fn build_pipeline() -> (OchromaNodeGraph, NodeId, NodeId, NodeId) {
        let mut g = OchromaNodeGraph::new();
        let terrain = g.add_node(
            "terrain",
            Box::new(TerrainNode {
                resolution: 32,
                amplitude: 300.0,
                droplet_count: 0,
                seed: 7,
                ..Default::default()
            }),
        );
        let biome = g.add_node("biome", Box::new(BiomeNode { world_height: 400.0, moisture: 0.5 }));
        let weight = g.add_node("weight", Box::new(SplatWeightNode));
        g.connect(terrain, "terrain", biome, "terrain").unwrap();
        g.connect(biome, "biome_map", weight, "biome_map").unwrap();
        (g, terrain, biome, weight)
    }

    fn sink_weights(g: &mut OchromaNodeGraph, weight: NodeId) -> Vec<[f32; 4]> {
        let r = g.evaluate().unwrap();
        r.get(weight, "splat_weights")
            .unwrap()
            .as_splat_weights()
            .unwrap()
            .clone()
    }

    /// Collapse equivalence: the sink output is byte-identical before and after
    /// collapsing the middle Biome node, and again after expanding it back.
    #[test]
    fn collapse_then_expand_preserves_sink_output_byte_for_byte() {
        let (mut g, _terrain, biome, weight) = build_pipeline();

        let before = sink_weights(&mut g, weight);
        // 32*32 = 1024 cells, each a [f32;4] weight tuple.
        assert_eq!(before.len(), 1024, "sink carries one weight per terrain cell");

        // Collapse just the middle Biome node into a subgraph.
        let collapsed = collapse_to_subgraph(&mut g, &[biome], "BiomeFn").unwrap();
        // Selection replaced by exactly one node; total node count is unchanged (3).
        assert_eq!(g.node_count(), 3, "biome replaced by one SubgraphNode");
        // The new node has one exposed input (Terrain) and one exposed output (BiomeMap).
        let def = g.subgraph_def(collapsed.node_id).expect("is a subgraph");
        assert_eq!(def.inputs.len(), 1);
        assert_eq!(def.outputs.len(), 1);
        assert_eq!(def.inputs[0].port_type, PortType::Terrain);
        assert_eq!(def.outputs[0].port_type, PortType::BiomeMap);

        let after_collapse = sink_weights(&mut g, weight);
        assert_eq!(after_collapse.len(), before.len(), "same length after collapse");
        assert_eq!(
            bytes_of(&after_collapse),
            bytes_of(&before),
            "collapse must not change the sink output bytes"
        );

        // Now expand the subgraph back to inline nodes and re-verify.
        expand_subgraph(&mut g, collapsed.node_id).unwrap();
        assert_eq!(g.node_count(), 3, "expand restores three inline nodes");
        let after_expand = sink_weights(&mut g, weight);
        assert_eq!(
            bytes_of(&after_expand),
            bytes_of(&before),
            "expand must not change the sink output bytes"
        );
    }

    /// Collapse a MULTI-node selection (Terrain + Biome) and prove equivalence at
    /// the surviving sink.
    #[test]
    fn collapse_multinode_selection_preserves_sink() {
        let (mut g, terrain, biome, weight) = build_pipeline();
        let before = sink_weights(&mut g, weight);

        let collapsed = collapse_to_subgraph(&mut g, &[terrain, biome], "TerrainBiomeFn").unwrap();
        // Terrain has no inputs; Biome's only input came from Terrain (internal) ->
        // zero exposed inputs. One exposed output: Biome's biome_map.
        let def = g.subgraph_def(collapsed.node_id).expect("is a subgraph");
        assert_eq!(def.inputs.len(), 0, "no boundary inputs (terrain is a source)");
        assert_eq!(def.outputs.len(), 1, "one boundary output: biome_map");
        assert_eq!(def.inner.node_count(), 2, "two inner nodes captured");
        assert_eq!(def.inner.edge_count(), 1, "internal terrain->biome edge captured");

        // weight + the subgraph node = 2 nodes.
        assert_eq!(g.node_count(), 2);
        let after = sink_weights(&mut g, weight);
        assert_eq!(bytes_of(&after), bytes_of(&before), "multi-node collapse preserves sink");
    }

    /// Re-entrant selection rejected: A->B->C with A,C selected but B outside must
    /// Err (collapsing A+C would cycle through B), and the graph is left unchanged.
    #[test]
    fn reentrant_selection_is_rejected_and_graph_unchanged() {
        // Linear Terrain(A) -> Biome(B) -> SplatWeight... but we need A and C to be
        // connected through B such that A feeds B and B feeds C, selecting {A, C}.
        let (mut g, terrain, biome, weight) = build_pipeline();
        let nodes_before = g.node_count();
        let edges_before = g.edge_count();
        let sink_before = sink_weights(&mut g, weight);

        // Select terrain (A) and weight (C), leaving biome (B) outside. A -> B -> C,
        // so collapsing {A,C} into X would form X -> B -> X: a cycle.
        let err = collapse_to_subgraph(&mut g, &[terrain, weight], "Bad").unwrap_err();
        assert!(matches!(err, SubgraphError::WouldCycle { .. }), "got {err:?}");

        // Graph fully unchanged: same node/edge counts and same sink output.
        assert_eq!(g.node_count(), nodes_before, "node count unchanged after rejection");
        assert_eq!(g.edge_count(), edges_before, "edge count unchanged after rejection");
        let _ = biome; // (kept outside; nothing collapsed)
        let sink_after = sink_weights(&mut g, weight);
        assert_eq!(bytes_of(&sink_after), bytes_of(&sink_before), "sink unchanged after rejection");
    }

    /// Nested: a subgraph whose inner graph contains another SubgraphNode cooks and
    /// yields the same sink output as the fully-inline pipeline.
    #[test]
    fn nested_subgraph_cooks_to_same_output() {
        let (mut g, _t, biome, weight) = build_pipeline();
        let inline = sink_weights(&mut g, weight);

        // First collapse the Biome node.
        let c1 = collapse_to_subgraph(&mut g, &[biome], "BiomeFn").unwrap();
        // Now collapse the SubgraphNode itself into ANOTHER subgraph -> nesting.
        let c2 = collapse_to_subgraph(&mut g, &[c1.node_id], "OuterFn").unwrap();

        // The outer subgraph's inner graph contains a SubgraphNode.
        let outer_def = g.subgraph_def(c2.node_id).expect("outer is a subgraph");
        let inner_has_subgraph = outer_def
            .inner
            .node_ids()
            .any(|id| outer_def.inner.subgraph_def(id).is_some());
        assert!(inner_has_subgraph, "outer subgraph must nest an inner SubgraphNode");

        let nested = sink_weights(&mut g, weight);
        assert_eq!(bytes_of(&nested), bytes_of(&inline), "nested subgraph cooks identically");
    }

    /// Depth-33 nesting errors with a typed CookFailed instead of overflowing.
    #[test]
    fn depth_overflow_errors_typed_not_stack_overflow() {
        // Build a chain of SubgraphNodes each wrapping the next, 33 deep, around a
        // trivial passthrough leaf. Cooking it must hit MAX_SUBGRAPH_DEPTH.
        use crate::node_graph::tests_helpers::pass_node;

        // Innermost def: a single pass node, exposing its scalar output.
        fn wrap_once(prev: SubgraphDef) -> SubgraphDef {
            let mut inner = OchromaNodeGraph::new();
            let node_id = inner.add_node("sub", Box::new(SubgraphNode::new(prev)));
            // Expose the wrapped node's "out" output.
            SubgraphDef {
                name: "wrap".to_string(),
                inner,
                inputs: vec![],
                outputs: vec![ExposedPort {
                    outer_name: "out".to_string(),
                    inner_node: node_id,
                    inner_port: "out".to_string(),
                    port_type: PortType::Scalar,
                }],
            }
        }

        // Leaf def: a pass node exposing "out".
        let mut leaf_inner = OchromaNodeGraph::new();
        let leaf_node = leaf_inner.add_node("pass", pass_node());
        let leaf = SubgraphDef {
            name: "leaf".to_string(),
            inner: leaf_inner,
            inputs: vec![],
            outputs: vec![ExposedPort {
                outer_name: "out".to_string(),
                inner_node: leaf_node,
                inner_port: "out".to_string(),
                port_type: PortType::Scalar,
            }],
        };

        // Wrap 40 times -> nesting depth far exceeds MAX_SUBGRAPH_DEPTH (32).
        let mut def = leaf;
        for _ in 0..40 {
            def = wrap_once(def);
        }
        let node = SubgraphNode::new(def);
        let err = node.cook(NodeInputs::new()).unwrap_err();
        match err {
            NodeError::CookFailed(msg) => {
                assert!(msg.contains("MAX_SUBGRAPH_DEPTH") || msg.contains("nesting"),
                    "depth overflow should be a typed cook error, got: {msg}");
            }
            other => panic!("expected typed CookFailed, got {other:?}"),
        }
    }

    /// Registry: a registered subgraph is creatable by name, searchable, and an
    /// instance cooks to the same output as the def's inner graph evaluated directly.
    #[test]
    fn registered_subgraph_is_creatable_searchable_and_cooks_identically() {
        // Build a standalone Biome subgraph def with one Terrain input + BiomeMap output.
        let (mut g, biome, _w) = {
            let (g, terrain, biome, weight) = build_pipeline();
            // Reduce to just the biome node as a def by collapsing it.
            (g, biome, (terrain, weight))
        };
        let collapsed = collapse_to_subgraph(&mut g, &[biome], "BiomeFn").unwrap();
        let def = g.subgraph_def(collapsed.node_id).unwrap().deep_clone();

        // Independently evaluate the def's inner graph with a known Terrain input fed
        // into the exposed input port, to get the reference output.
        let reference = {
            let node = SubgraphNode::new(def.deep_clone());
            // Build a Terrain input matching the exposed input's bound port.
            let mut terrain_g = OchromaNodeGraph::new();
            let t = terrain_g.add_node(
                "t",
                Box::new(TerrainNode { resolution: 16, amplitude: 250.0, droplet_count: 0, seed: 3, ..Default::default() }),
            );
            let tr = terrain_g.evaluate().unwrap();
            let terrain_data = tr.get(t, "terrain").unwrap().clone();
            let mut inputs = NodeInputs::new();
            inputs.insert(def.inputs[0].outer_name.clone(), terrain_data);
            node.cook(inputs).unwrap()
        };

        // Register the def and create an instance by name.
        let mut reg = NodeRegistry::new();
        let kind_name = reg.register_subgraph(def.deep_clone());
        assert_eq!(kind_name, "BiomeFn");
        assert!(reg.get("BiomeFn").is_some(), "registered kind retrievable");
        // Searchable like a built-in.
        let search_hits = reg.search("Biome");
        let hits: Vec<&str> = search_hits.iter().map(|h| h.name()).collect();
        assert!(hits.contains(&"BiomeFn"), "registered subgraph must be searchable, got {hits:?}");

        let instance = reg.create("BiomeFn").expect("creatable by name");

        // Feed the SAME terrain input to the created instance and compare outputs.
        let produced = {
            let mut terrain_g = OchromaNodeGraph::new();
            let t = terrain_g.add_node(
                "t",
                Box::new(TerrainNode { resolution: 16, amplitude: 250.0, droplet_count: 0, seed: 3, ..Default::default() }),
            );
            let tr = terrain_g.evaluate().unwrap();
            let terrain_data = tr.get(t, "terrain").unwrap().clone();
            let mut inputs = NodeInputs::new();
            inputs.insert(def.inputs[0].outer_name.clone(), terrain_data);
            instance.cook(inputs).unwrap()
        };

        let out_port = &def.outputs[0].outer_name;
        let r = reference[out_port].as_biome_map().unwrap();
        let p = produced[out_port].as_biome_map().unwrap();
        assert_eq!(p, r, "registry-created subgraph instance cooks identically to the def");
        assert_eq!(p.len(), 16 * 16, "biome map sized by the fed terrain");
    }

    /// Namespaced param edit through the live path recooks the SubgraphNode and
    /// changes the sink output; clean unrelated nodes do not recook.
    #[test]
    fn namespaced_param_edit_through_live_path_recooks_and_changes_sink() {
        let (mut g, terrain, _biome, weight) = build_pipeline();

        // Collapse the Terrain node alone into a subgraph. Its inner node keeps the
        // label "terrain", so the namespaced param "terrain.amplitude" routes to it.
        // The subgraph exposes terrain's "terrain" output, which still feeds Biome.
        let collapsed = collapse_to_subgraph(&mut g, &[terrain], "TerrainFn").unwrap();
        let sub_id = collapsed.node_id;

        // Establish baselines via a full cook.
        g.cook().unwrap();
        let sink_before = g.get_output(weight, "splat_weights").unwrap().as_splat_weights().unwrap().clone();
        let cc_weight_before = g.cook_count(weight).unwrap();
        let cc_sub_before = g.cook_count(sub_id).unwrap();

        // Edit the inner terrain node's amplitude via the namespaced param through
        // the live throttled path. "terrain" is the inner node's label.
        let t0 = std::time::Instant::now();
        g.set_recook_budget(std::time::Duration::from_millis(100));
        g.request_recook(sub_id, "terrain.amplitude", ParamValue::Float(1200.0)).unwrap();

        let report = g
            .live_cook(t0 + std::time::Duration::from_millis(200))
            .unwrap()
            .expect("a recook was due");
        assert_eq!(report.root, sub_id);

        // Subgraph + downstream weight recooked exactly once each.
        assert_eq!(g.cook_count(sub_id).unwrap(), cc_sub_before + 1, "subgraph recooked");
        assert_eq!(g.cook_count(weight).unwrap(), cc_weight_before + 1, "downstream sink recooked");

        // The sink output genuinely changed (raising amplitude reclassifies biomes,
        // changing splat weights).
        let sink_after = g.get_output(weight, "splat_weights").unwrap().as_splat_weights().unwrap();
        assert_ne!(
            bytes_of(sink_after),
            bytes_of(&sink_before),
            "namespaced inner param edit must change the sink output"
        );
    }

    /// Empty selection and unknown ids are typed errors.
    #[test]
    fn invalid_selections_error_typed() {
        let (mut g, _t, _b, _w) = build_pipeline();
        assert!(matches!(collapse_to_subgraph(&mut g, &[], "x"), Err(SubgraphError::EmptySelection)));
        assert!(matches!(
            collapse_to_subgraph(&mut g, &[NodeId(9999)], "x"),
            Err(SubgraphError::NodeNotFound(_))
        ));
    }

    /// Helper: flatten a Vec<[f32;4]> into its raw little-endian bytes for exact
    /// equality comparison (the "byte-equal sink output" invariant).
    fn bytes_of(weights: &[[f32; 4]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(weights.len() * 16);
        for w in weights {
            for &c in w {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    }
}
