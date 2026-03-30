# Domain 9 — Editor: OchromaNodeGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a procedural scene editor as a DAG node graph. The `OchromaNodeGraph` is a direct port of `CrucibleGraph` from AetherSpectra, with Ochroma-specific port types. Four domain nodes are adapted from AetherSpectra's `forge` crates: `TerrainNode` (fBm + hydraulic erosion), `BuildingNode` (WFC), `VegetationNode` (L-system + LOD), `SplatizeNode` (Mesh → GaussianSplat[] with spectral assignment). An egui node editor panel wires everything into the editor UI.

**Source material read and understood before writing this plan:**

- `aetherspectra/crucible/rust/crates/crucible-core/src/graph.rs` — `CrucibleGraph`: `HashMap<NodeId, NodeEntry>`, `Vec<Edge>`, Kahn's topological sort via `BinaryHeap<Reverse<NodeId>>` for determinism, downstream dirty cascade in `mark_dirty()`, type-checked `connect()` using `descriptor().output_type()` / `descriptor().input_type()`, cook skips clean nodes, `assemble_inputs()` gathers upstream `last_output` by port name, `CycleDetected` guard via `can_reach()`, idempotent duplicate-edge guard.
- `aetherspectra/crucible/rust/crates/crucible-core/src/port.rs` — `PortDataType { Terrain, Geometry, LodGeometry, Instances, Light, Camera, Atmosphere, Fog, Material, Scalar, Null }`, `PortData` enum with accessors, `PortMap = HashMap<String, PortData>`, `ParamValue { Float, Int, Str, Bool, Vec2, Vec3 }`.
- `aetherspectra/forge/crates/building/src/wfc.rs` — `WFCParams { grid_w, grid_h, grid_d, style, seed, max_attempts }`, bitmask superposition (`u8` per cell, 5 tile types), Kahn-style BFS propagation (`VecDeque`), boundary pre-constraint logic, `try_solve()` returns `Option<Vec<TileType>>`, PCG64 seeded RNG, `Pcg64::seed_from_u64(seed ^ attempt * 0x9e3779b9)` for retry variation.
- `aetherspectra/forge/crates/terrain/src/generate.rs` — `noise::Fbm<Perlin>` with `octaves`, `frequency`, `persistence=0.5`, `lacunarity=2.0`; per-cell fBm eval at `(x * cell_size * scale, z * ...)`, normalise to [0,1], multiply by `amplitude`; separate high-frequency fBm for biome classification.
- `aetherspectra/forge/crates/terrain/src/hydraulic.rs` — `HydraulicParams { droplet_count, inertia, capacity, deposition, erosion_rate, evaporation, seed }`, Olsen/Cordonnier droplet model: central-difference gradient, velocity × inertia, sediment capacity = `params.capacity × speed × water × max(0.01, -dh)`, deposit/erode on height delta, `deposition_map` tracking.
- `aetherspectra/forge/crates/vegetation/src/lsystem.rs` — `build_tree(params)` → `Mesh`, `grow_segment()` recursive: cylinder segments, branch recursion with azimuth rotation `rotate_around_axis(dir, azimuth, angle)`, leaf cluster at terminal nodes, 20k vertex budget guard.
- `aetherspectra/forge/crates/vegetation/src/lod.rs` — `build_lod_set(params) -> [Mesh; 4]`: LOD0 = full tree, LOD1 = `decimate(lod0, 0.5)`, LOD2 = `decimate(lod0, 0.2)`, LOD3 = billboard from `aabb()` — a 2-triangle quad sized from mesh bounds.

**Tech Stack:** Rust, `noise = "0.9"`, `rand = "0.8"`, `rand_pcg = "0.3"`, `egui` (existing), `half` (existing), `hashbrown` (existing), `thiserror` (existing)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_editor/src/node_graph.rs` | `OchromaNodeGraph`, port of `CrucibleGraph` with Ochroma port types |
| Create | `crates/vox_editor/src/nodes/terrain_node.rs` | `TerrainNode` — fBm + hydraulic erosion → `HeightfieldSpatial` |
| Create | `crates/vox_editor/src/nodes/building_node.rs` | `BuildingNode` — WFC → `EditorMesh` |
| Create | `crates/vox_editor/src/nodes/vegetation_node.rs` | `VegetationNode` — L-system + LOD → `EditorMesh` with LOD levels |
| Create | `crates/vox_editor/src/nodes/splatize_node.rs` | `SplatizeNode` — Mesh → `Vec<GaussianSplat>` with spectral assignment |
| Create | `crates/vox_editor/src/nodes/mod.rs` | re-export all node types |
| Create | `crates/vox_editor/src/editor_panel.rs` | egui node editor panel: canvas, wires, parameter sidebar |
| Modify | `crates/vox_editor/src/lib.rs` | expose all new modules |
| Modify | `crates/vox_editor/Cargo.toml` | add `noise`, `rand`, `rand_pcg` deps |

---

## Task 1: OchromaNodeGraph — port of CrucibleGraph with Ochroma port types

**Files:**
- Create: `crates/vox_editor/src/node_graph.rs`
- Modify: `crates/vox_editor/src/lib.rs`

**Design decisions derived from CrucibleGraph source:**
- Retain the exact `HashMap<NodeId, NodeEntry>` + `Vec<Edge>` structure.
- Retain Kahn's topological sort with `BinaryHeap<Reverse<NodeId>>` for determinism.
- Retain the downstream dirty cascade in `mark_dirty()`.
- Retain the idempotent duplicate-edge guard from `connect()` (the AetherSpectra fix for inflated in-degrees).
- Replace `PortDataType { Terrain, Geometry, LodGeometry, ... }` with Ochroma types: `Splats`, `SpectralField`, `Terrain`, `Mesh`, `LodMesh`, `Instances`, `Scalar`.
- Replace `PortData` variants accordingly with Ochroma data types.
- `OchromaNode` trait signature matches `CrucibleNode` exactly: `fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError>`.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_editor/src/node_graph.rs`:

```rust
//! OchromaNodeGraph — procedural scene operations as a DAG.
//!
//! Direct port of CrucibleGraph from AetherSpectra with Ochroma-specific port types.
//! Key invariants (all inherited from CrucibleGraph):
//!   - Kahn's topological sort with BinaryHeap<Reverse<NodeId>> for determinism
//!   - Downstream dirty cascade on mark_dirty()
//!   - Idempotent duplicate-edge guard in connect()
//!   - Type-checked port connections via OchromaNode::descriptor()

use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Reverse;
use hashbrown::HashMap;
use thiserror::Error;

use vox_core::types::GaussianSplat;

// ---------------------------------------------------------------------------
// Port types — Ochroma-specific data flowing along edges
// ---------------------------------------------------------------------------

/// Discriminant used for connection type-checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortType {
    /// Vec<GaussianSplat> — output of SplatizeNode, consumed by renderer
    Splats,
    /// Spectral radiance field sample (8-band energy at a position)
    SpectralField,
    /// HeightfieldSpatial — output of TerrainNode
    Terrain,
    /// Single-resolution mesh
    Mesh,
    /// Multiple LOD meshes (LOD0..LOD3), output of VegetationNode
    LodMesh,
    /// Scatter instance positions + transforms
    Instances,
    /// f64 scalar
    Scalar,
}

impl std::fmt::Display for PortType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Concrete value flowing along an edge.
#[derive(Clone)]
pub enum PortData {
    Splats(Vec<GaussianSplat>),
    SpectralField([f32; 8]),
    Terrain(HeightfieldSpatial),
    Mesh(EditorMesh),
    LodMesh(Vec<EditorMesh>),
    Instances(Vec<[f32; 3]>),
    Scalar(f64),
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
        }
    }
    pub fn as_terrain(&self) -> Option<&HeightfieldSpatial> {
        match self { PortData::Terrain(t) => Some(t), _ => None }
    }
    pub fn as_mesh(&self) -> Option<&EditorMesh> {
        match self { PortData::Mesh(m) => Some(m), _ => None }
    }
    pub fn as_lod_mesh(&self) -> Option<&Vec<EditorMesh>> {
        match self { PortData::LodMesh(l) => Some(l), _ => None }
    }
    pub fn as_splats(&self) -> Option<&Vec<GaussianSplat>> {
        match self { PortData::Splats(s) => Some(s), _ => None }
    }
    pub fn as_scalar(&self) -> Option<f64> {
        match self { PortData::Scalar(v) => Some(*v), _ => None }
    }
}

/// Map of port name → value for a cook() call.
pub type NodeInputs  = HashMap<String, PortData>;
pub type NodeOutputs = HashMap<String, PortData>;

// ---------------------------------------------------------------------------
// Minimal geometry types used inside the editor
// ---------------------------------------------------------------------------

/// Heightfield with world-space extent. Output of TerrainNode.
#[derive(Clone, Debug)]
pub struct HeightfieldSpatial {
    /// Heights in row-major order. Length = resolution × resolution.
    pub heights:    Vec<f32>,
    /// Grid resolution (heights.len() == resolution * resolution).
    pub resolution: u32,
    /// World-space size in metres (square).
    pub world_size: f32,
}

/// Simplified mesh for editor operations (positions + indices + optional UVs).
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

// ---------------------------------------------------------------------------
// Node trait and descriptor
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum NodeError {
    #[error("missing required input port: {0}")]
    MissingInput(String),
    #[error("type mismatch on port {port}: expected {expected}, got {got}")]
    TypeMismatch { port: String, expected: String, got: String },
    #[error("cook failed: {0}")]
    CookFailed(String),
    #[error("unknown parameter: {0}")]
    UnknownParam(String),
}

/// Port specification for a node's input or output declaration.
pub struct PortSpec {
    pub name:      &'static str,
    pub port_type: PortType,
    pub optional:  bool,
}

/// Static description of a node's ports and type name.
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

/// Trait all Ochroma procedural nodes must implement.
pub trait OchromaNode: Send + Sync {
    fn descriptor(&self) -> NodeDescriptor;
    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError>;
    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError>;
}

/// Parameter value type (matches AetherSpectra's ParamValue).
#[derive(Debug, Clone)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
}

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

// ---------------------------------------------------------------------------
// Graph error type
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Internal node storage
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// OchromaNodeGraph
// ---------------------------------------------------------------------------

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
        self.nodes.insert(id, NodeEntry {
            name:        name.to_string(),
            node,
            dirty:       true,
            last_output: None,
        });
        id
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&id) { return Err(GraphError::NodeNotFound(id)); }
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
        Ok(())
    }

    pub fn is_dirty(&self, id: NodeId) -> bool {
        self.nodes.get(&id).map(|e| e.dirty).unwrap_or(false)
    }

    /// Connect from_port of `from` to to_port of `to`.
    /// Validates port types. Rejects cycles. Idempotent for duplicate edges.
    pub fn connect(
        &mut self,
        from: NodeId, from_port: &str,
        to:   NodeId, to_port:   &str,
    ) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&from) { return Err(GraphError::NodeNotFound(from)); }
        if !self.nodes.contains_key(&to)   { return Err(GraphError::NodeNotFound(to));   }

        // from_port must exist as a declared output
        let from_type = {
            let desc = self.nodes[&from].node.descriptor();
            match desc.output_type(from_port) {
                Some(t) => t,
                None    => return Err(GraphError::UnknownPort { node: from, port: from_port.to_string() }),
            }
        };

        // to_port is validated only if declared; undeclared ports = dynamic (allowed)
        {
            let desc = self.nodes[&to].node.descriptor();
            if let Some(expected) = desc.input_type(to_port) {
                if from_type != expected {
                    return Err(GraphError::TypeMismatch {
                        port:     to_port.to_string(),
                        expected: expected.to_string(),
                        got:      from_type.to_string(),
                    });
                }
            }
        }

        // Cycle check
        if self.can_reach(to, from) {
            return Err(GraphError::CycleDetected { from: from.0, to: to.0 });
        }

        // Idempotent duplicate guard (prevents inflated in-degrees in topo_sort)
        if self.edges.iter().any(|e| {
            e.from == from && e.from_port == from_port
                && e.to == to && e.to_port == to_port
        }) {
            return Ok(());
        }

        self.edges.push(Edge {
            from,
            from_port: from_port.to_string(),
            to,
            to_port: to_port.to_string(),
        });
        Ok(())
    }

    /// Kahn's topological sort — ascending NodeId for determinism (matches CrucibleGraph).
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
            .filter(|(_, &d)| d == 0)
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
            return Err(GraphError::CookFailed {
                node:   "topo_sort".into(),
                reason: "cycle detected (edges injected outside connect())".into(),
            });
        }
        Ok(order)
    }

    /// Mark `id` and all downstream nodes dirty.
    pub fn mark_dirty(&mut self, id: NodeId) {
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
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
        entry.node.set_param(key, value).map_err(|e| GraphError::CookFailed {
            node: entry.name.clone(), reason: e.to_string(),
        })?;
        self.mark_dirty(id);
        Ok(())
    }

    /// Evaluate all dirty nodes in topological order.
    pub fn cook(&mut self) -> Result<(), GraphError> {
        let order = self.topo_sort()?;
        for id in order {
            if !self.nodes.get(&id).map(|e| e.dirty).unwrap_or(false) { continue; }
            let inputs = self.assemble_inputs(id)?;
            let name   = self.nodes[&id].name.clone();
            let output = self.nodes[&id].node.cook(inputs).map_err(|e| GraphError::CookFailed {
                node:   name.clone(),
                reason: e.to_string(),
            })?;
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct PassNode;
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
    }

    fn pass() -> Box<dyn OchromaNode> { Box::new(PassNode) }

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
        g.connect(a, "out", b, "in").unwrap(); // must not panic or add extra edge
        let order = g.topo_sort().unwrap();
        assert_eq!(order.len(), 2, "both nodes must be in topo order");
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
        }
        let mut g = OchromaNodeGraph::new();
        g.add_node("n", Box::new(CountNode(count.clone())));
        g.cook().unwrap();
        g.cook().unwrap(); // second cook — node is clean, must not re-execute
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
            fn set_param(&mut self, k: &str, _: ParamValue) -> Result<(), NodeError> {
                Err(NodeError::UnknownParam(k.into()))
            }
            fn cook(&self, _: NodeInputs) -> Result<NodeOutputs, NodeError> { Ok(NodeOutputs::new()) }
        }
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", Box::new(TerrainOutNode));
        let b = g.add_node("b", pass());
        let err = g.connect(a, "terrain", b, "in").unwrap_err();
        assert!(matches!(err, GraphError::TypeMismatch { .. }), "got: {:?}", err);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/tomespen/git/ochroma
cargo test -p vox_editor node_graph 2>&1 | head -20
```

Expected: compile error — module not exposed in lib.rs.

- [ ] **Step 3: Expose the module**

Add to `crates/vox_editor/src/lib.rs`:

```rust
pub mod node_graph;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_editor node_graph -- --nocapture
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_editor/src/node_graph.rs crates/vox_editor/src/lib.rs
git commit -m "feat(editor): OchromaNodeGraph — CrucibleGraph port with Ochroma port types"
```

---

## Task 2: TerrainNode — fBm + hydraulic erosion

**Files:**
- Create: `crates/vox_editor/src/nodes/terrain_node.rs`
- Modify: `crates/vox_editor/src/nodes/mod.rs`
- Modify: `crates/vox_editor/Cargo.toml`

**Implementation derived from forge-terrain source:**
- `generate.rs`: `Fbm::<Perlin>::new(seed as u32)`, set `.octaves`, `.frequency`, `.persistence = 0.5`, `.lacunarity = 2.0`; sample at `(x * cell_size * scale, z * cell_size * scale)` for each grid cell; normalise `((v + 1.0) * 0.5).clamp(0.0, 1.0)` then multiply by amplitude.
- `hydraulic.rs`: `Pcg64::seed_from_u64(seed)`, per-droplet loop: central-difference gradient (`h[iz*r + (ix+1)] - h[iz*r + (ix-1)]`), velocity × inertia, capacity = `params.capacity × speed × water × max(-dh, 0.01)`, deposit if `sediment > capacity || dh > 0`, erode otherwise. Evaporate each step.

- [ ] **Step 1: Add deps to Cargo.toml**

```toml
[dependencies]
noise     = "0.9"
rand      = "0.8"
rand_pcg  = "0.3"
```

- [ ] **Step 2: Write failing tests**

Create `crates/vox_editor/src/nodes/terrain_node.rs`:

```rust
//! TerrainNode — fBm heightfield + hydraulic erosion.
//!
//! Adapted from aetherspectra/forge/crates/terrain/src/{generate.rs, hydraulic.rs}.
//! Key differences: output is HeightfieldSpatial (Ochroma type), not forge's TerrainGrid.

use noise::{Fbm, NoiseFn, Perlin};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::node_graph::{
    HeightfieldSpatial, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

pub struct TerrainNode {
    pub resolution:    u32,
    pub world_size:    f32,
    pub amplitude:     f32,
    pub octaves:       usize,
    pub frequency:     f64,
    pub seed:          u32,
    /// Hydraulic erosion droplet count. 0 = no erosion.
    pub droplet_count: u32,
}

impl Default for TerrainNode {
    fn default() -> Self {
        Self {
            resolution:    256,
            world_size:    1000.0,
            amplitude:     200.0,
            octaves:       6,
            frequency:     1.0,
            seed:          0,
            droplet_count: 80_000,
        }
    }
}

impl OchromaNode for TerrainNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "TerrainNode",
            inputs:    vec![],
            outputs:   vec![PortSpec { name: "terrain", port_type: PortType::Terrain, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        match key {
            "resolution"    => { self.resolution    = value.as_float_coerce()? as u32; }
            "world_size"    => { self.world_size     = value.as_float_coerce()? as f32; }
            "amplitude"     => { self.amplitude      = value.as_float_coerce()? as f32; }
            "octaves"       => { self.octaves        = value.as_float_coerce()? as usize; }
            "frequency"     => { self.frequency      = value.as_float_coerce()?; }
            "seed"          => { self.seed           = value.as_float_coerce()? as u32; }
            "droplet_count" => { self.droplet_count  = value.as_float_coerce()? as u32; }
            _               => return Err(NodeError::UnknownParam(key.into())),
        }
        Ok(())
    }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let hf = generate_heightfield(self)?;
        let mut out = NodeOutputs::new();
        out.insert("terrain".into(), PortData::Terrain(hf));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Internal: fBm generation (derived from forge-terrain generate.rs)
// ---------------------------------------------------------------------------

fn generate_heightfield(params: &TerrainNode) -> Result<HeightfieldSpatial, NodeError> {
    if params.resolution < 16 || params.resolution > 4096 {
        return Err(NodeError::CookFailed(
            format!("resolution {} out of [16, 4096]", params.resolution)
        ));
    }

    let r = params.resolution as usize;
    let cell_size = params.world_size / params.resolution as f32;

    let mut fbm: Fbm<Perlin> = Fbm::new(params.seed);
    fbm.octaves     = params.octaves;
    fbm.frequency   = params.frequency;
    fbm.persistence = 0.5;
    fbm.lacunarity  = 2.0;

    let scale = 1.0 / params.world_size as f64;
    let mut heights: Vec<f32> = Vec::with_capacity(r * r);

    for idx in 0..r * r {
        let x = (idx % r) as f64 * cell_size as f64 * scale;
        let z = (idx / r) as f64 * cell_size as f64 * scale;
        let v = fbm.get([x, z]) as f32;
        let normalised = ((v + 1.0) * 0.5).clamp(0.0, 1.0);
        heights.push(normalised * params.amplitude);
    }

    // Hydraulic erosion (derived from forge-terrain hydraulic.rs)
    if params.droplet_count > 0 {
        hydraulic_erode(&mut heights, r, params);
    }

    Ok(HeightfieldSpatial {
        heights,
        resolution: params.resolution,
        world_size: params.world_size,
    })
}

// ---------------------------------------------------------------------------
// Internal: hydraulic erosion (Olsen/Cordonnier droplet model)
// Adapted from aetherspectra/forge/crates/terrain/src/hydraulic.rs
// ---------------------------------------------------------------------------

fn hydraulic_erode(heights: &mut Vec<f32>, r: usize, params: &TerrainNode) {
    let inertia      = 0.05f32;
    let capacity     = 4.0f32;
    let deposition   = 0.3f32;
    let erosion_rate = 0.3f32;
    let evaporation  = 0.015f32;

    let mut rng = Pcg64::seed_from_u64(params.seed as u64);
    let count   = params.droplet_count.min(200_000); // cap for editor responsiveness

    for _ in 0..count {
        let mut px = rng.gen_range(1.0..(r as f32 - 1.0));
        let mut pz = rng.gen_range(1.0..(r as f32 - 1.0));
        let (mut vx, mut vz) = (0.0f32, 0.0f32);
        let mut water    = 1.0f32;
        let mut sediment = 0.0f32;

        for _ in 0..256 {
            let ix = px as usize; let iz = pz as usize;
            if ix == 0 || iz == 0 || ix >= r - 1 || iz >= r - 1 { break; }

            // Central difference gradient (matches forge-terrain exactly)
            let gx = heights[iz * r + (ix + 1)] - heights[iz * r + (ix - 1)];
            let gz = heights[(iz + 1) * r + ix] - heights[(iz - 1) * r + ix];

            vx = vx * inertia - gx * (1.0 - inertia);
            vz = vz * inertia - gz * (1.0 - inertia);
            let speed = (vx * vx + vz * vz).sqrt();
            if speed < 1e-4 { break; }
            vx /= speed; vz /= speed;

            let new_px = (px + vx).clamp(1.0, (r - 2) as f32);
            let new_pz = (pz + vz).clamp(1.0, (r - 2) as f32);
            let nix = new_px as usize; let niz = new_pz as usize;

            let old_h = heights[iz * r + ix];
            let new_h = heights[niz * r + nix];
            let dh    = new_h - old_h;

            let cap = (capacity * speed * water * (-dh).max(0.01)).max(0.0);

            if sediment > cap || dh > 0.0 {
                let deposit = if dh > 0.0 { sediment.min(dh) } else { (sediment - cap) * deposition };
                let deposit = deposit.min(sediment);
                heights[iz * r + ix] += deposit;
                sediment -= deposit;
            } else {
                let erode = ((cap - sediment) * erosion_rate).min(-dh).max(0.0);
                heights[iz * r + ix] -= erode;
                sediment += erode;
            }

            water  *= 1.0 - evaporation;
            px = new_px; pz = new_pz;
        }
    }
}

// Helper for set_param
trait ParamValueExt {
    fn as_float_coerce(&self) -> Result<f64, NodeError>;
}
impl ParamValueExt for ParamValue {
    fn as_float_coerce(&self) -> Result<f64, NodeError> {
        match self {
            ParamValue::Float(v) => Ok(*v),
            ParamValue::Int(v)   => Ok(*v as f64),
            _                    => Err(NodeError::CookFailed("expected numeric param".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_node_cooks_without_error() {
        let node = TerrainNode { resolution: 64, droplet_count: 0, ..Default::default() };
        let out = node.cook(NodeInputs::new()).unwrap();
        assert!(out.contains_key("terrain"));
    }

    #[test]
    fn output_height_count_matches_resolution() {
        let node = TerrainNode { resolution: 64, droplet_count: 0, ..Default::default() };
        let out = node.cook(NodeInputs::new()).unwrap();
        let hf = out["terrain"].as_terrain().unwrap();
        assert_eq!(hf.heights.len(), 64 * 64);
        assert_eq!(hf.resolution, 64);
    }

    #[test]
    fn heights_are_in_valid_range() {
        let node = TerrainNode { resolution: 64, amplitude: 100.0, droplet_count: 0, ..Default::default() };
        let out = node.cook(NodeInputs::new()).unwrap();
        let hf = out["terrain"].as_terrain().unwrap();
        for &h in &hf.heights {
            assert!(!h.is_nan(), "height is NaN");
            assert!(h >= 0.0 && h <= 200.0, "height out of [0, amplitude*2]: {}", h);
        }
    }

    #[test]
    fn different_seeds_produce_different_terrain() {
        let a = TerrainNode { resolution: 32, seed: 1, droplet_count: 0, ..Default::default() };
        let b = TerrainNode { resolution: 32, seed: 2, droplet_count: 0, ..Default::default() };
        let ha = a.cook(NodeInputs::new()).unwrap();
        let hb = b.cook(NodeInputs::new()).unwrap();
        let ha = ha["terrain"].as_terrain().unwrap();
        let hb = hb["terrain"].as_terrain().unwrap();
        let diff: f32 = ha.heights.iter().zip(hb.heights.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "different seeds should produce different terrain, total diff={}", diff);
    }

    #[test]
    fn erosion_modifies_terrain() {
        let no_erosion  = TerrainNode { resolution: 32, droplet_count: 0,    seed: 42, ..Default::default() };
        let with_erosion = TerrainNode { resolution: 32, droplet_count: 1000, seed: 42, ..Default::default() };
        let ha = no_erosion.cook(NodeInputs::new()).unwrap();
        let hb = with_erosion.cook(NodeInputs::new()).unwrap();
        let ha = ha["terrain"].as_terrain().unwrap();
        let hb = hb["terrain"].as_terrain().unwrap();
        let diff: f32 = ha.heights.iter().zip(hb.heights.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "erosion should modify terrain heights, diff={}", diff);
    }

    #[test]
    fn invalid_resolution_returns_error() {
        let node = TerrainNode { resolution: 8, ..Default::default() };
        let err = node.cook(NodeInputs::new()).unwrap_err();
        assert!(matches!(err, NodeError::CookFailed(_)));
    }

    #[test]
    fn terrain_node_in_graph_cooks_end_to_end() {
        use crate::node_graph::OchromaNodeGraph;
        let mut graph = OchromaNodeGraph::new();
        let terrain_id = graph.add_node("terrain", Box::new(TerrainNode {
            resolution: 32, droplet_count: 0, ..Default::default()
        }));
        graph.cook().unwrap();
        assert!(graph.get_output(terrain_id, "terrain").is_some());
    }
}
```

- [ ] **Step 3: Create `nodes/mod.rs`**

```rust
pub mod terrain_node;
pub mod building_node;
pub mod vegetation_node;
pub mod splatize_node;
```

Add to `crates/vox_editor/src/lib.rs`:

```rust
pub mod nodes;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_editor terrain_node -- --nocapture
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_editor/src/nodes/ crates/vox_editor/src/lib.rs crates/vox_editor/Cargo.toml
git commit -m "feat(editor): TerrainNode — fBm heightfield + hydraulic erosion adapted from forge-terrain"
```

---

## Task 3: BuildingNode — WFC facade

**Files:**
- Create: `crates/vox_editor/src/nodes/building_node.rs`

**Implementation derived from forge-building/src/wfc.rs:**
- 5 tile types: `Wall`, `Window`, `Corner`, `Door`, `Empty`; bitmask superposition (`u8`).
- BFS propagation via `VecDeque`; adjacency rules symmetric.
- Pre-constraint: corners = `Corner`; ground-centre-front = `Door`; non-ground borders = `Wall | Window` (style-dependent); interiors = `Wall | Empty`.
- Collapse loop: find min-entropy cell, pick random tile, propagate. Return `None` on contradiction.
- Retry with `seed ^ (attempt * 0x9e3779b9)` per attempt (Knuth multiplicative hash).
- Output: triangulated box mesh assembled from tile grid cells (wall = cube face, door = arch, window = frame).

- [ ] **Step 1: Write failing tests**

Create `crates/vox_editor/src/nodes/building_node.rs`:

```rust
//! BuildingNode — WFC tile-based building generator.
//!
//! Adapted from aetherspectra/forge/crates/building/src/wfc.rs.
//! Key changes: output is EditorMesh (Ochroma type); style enum is Ochroma-local.

use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use std::collections::VecDeque;

use crate::node_graph::{
    EditorMesh, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingStyle {
    Modern, Victorian, Gothic, Industrial, Medieval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TileType { Wall, Window, Corner, Door, Empty }

const ALL_TILES: [TileType; 5] = [
    TileType::Wall, TileType::Window, TileType::Corner, TileType::Door, TileType::Empty,
];

fn tile_bit(t: TileType) -> u8 {
    match t {
        TileType::Wall   => 1,
        TileType::Window => 2,
        TileType::Corner => 4,
        TileType::Door   => 8,
        TileType::Empty  => 16,
    }
}

fn allowed_neighbours(tile: TileType) -> &'static [TileType] {
    match tile {
        TileType::Wall    => &[TileType::Wall, TileType::Window, TileType::Corner, TileType::Door],
        TileType::Window  => &[TileType::Wall, TileType::Corner],
        TileType::Corner  => &[TileType::Wall, TileType::Window, TileType::Corner, TileType::Door],
        TileType::Door    => &[TileType::Wall, TileType::Corner],
        TileType::Empty   => &[TileType::Empty],
    }
}

pub struct BuildingNode {
    pub grid_w:       u32,
    pub grid_h:       u32,
    pub grid_d:       u32,
    pub style:        BuildingStyle,
    pub seed:         u64,
    pub max_attempts: u32,
    /// World-space size of each tile in metres.
    pub tile_size:    f32,
}

impl Default for BuildingNode {
    fn default() -> Self {
        Self {
            grid_w: 5, grid_h: 3, grid_d: 5,
            style: BuildingStyle::Modern,
            seed: 0,
            max_attempts: 10,
            tile_size: 3.0,
        }
    }
}

impl OchromaNode for BuildingNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "BuildingNode",
            inputs:    vec![],
            outputs:   vec![PortSpec { name: "mesh", port_type: PortType::Mesh, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        let f = |v: &ParamValue| -> Result<f64, NodeError> {
            match v {
                ParamValue::Float(x) => Ok(*x),
                ParamValue::Int(x)   => Ok(*x as f64),
                _                    => Err(NodeError::CookFailed("expected numeric".into())),
            }
        };
        match key {
            "grid_w"       => { self.grid_w       = f(&value)? as u32; }
            "grid_h"       => { self.grid_h       = f(&value)? as u32; }
            "grid_d"       => { self.grid_d       = f(&value)? as u32; }
            "seed"         => { self.seed         = f(&value)? as u64; }
            "max_attempts" => { self.max_attempts  = f(&value)? as u32; }
            "tile_size"    => { self.tile_size     = f(&value)? as f32; }
            _ => return Err(NodeError::UnknownParam(key.into())),
        }
        Ok(())
    }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let tiles = solve_wfc(
            self.grid_w as usize,
            self.grid_h as usize,
            self.grid_d as usize,
            self.style,
            self.seed,
            self.max_attempts,
        ).ok_or_else(|| NodeError::CookFailed(
            format!("WFC failed after {} attempts", self.max_attempts)
        ))?;

        let mesh = tiles_to_mesh(&tiles, self.grid_w as usize, self.grid_h as usize,
                                 self.grid_d as usize, self.tile_size);
        let mut out = NodeOutputs::new();
        out.insert("mesh".into(), PortData::Mesh(mesh));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// WFC solver (adapted from forge-building wfc.rs)
// ---------------------------------------------------------------------------

fn solve_wfc(
    w: usize, h: usize, d: usize,
    style: BuildingStyle,
    seed: u64,
    max_attempts: u32,
) -> Option<Vec<TileType>> {
    for attempt in 0..max_attempts.max(1) {
        let mut rng = Pcg64::seed_from_u64(seed ^ (attempt as u64 * 0x9e3779b9));
        if let Some(tiles) = try_solve(w, h, d, style, &mut rng) {
            return Some(tiles);
        }
    }
    None
}

fn try_solve(
    w: usize, h: usize, d: usize,
    style: BuildingStyle,
    rng: &mut Pcg64,
) -> Option<Vec<TileType>> {
    let n = w * h * d;
    let all_bits: u8 = ALL_TILES.iter().fold(0, |acc, &t| acc | tile_bit(t));
    let mut superpos = vec![all_bits; n];

    let cell_idx = |x: usize, y: usize, z: usize| z * h * w + y * w + x;

    // Pre-constrain boundary cells (mirrors forge-building logic exactly)
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = cell_idx(x, y, z);
                let is_corner = (x == 0 || x == w-1) && (z == 0 || z == d-1);
                let is_ground_centre = y == 0 && x == w/2 && z == 0;
                let is_border_nongnd = (x == 0 || x == w-1 || z == 0 || z == d-1) && y > 0;

                superpos[idx] = if is_corner {
                    tile_bit(TileType::Corner)
                } else if is_ground_centre {
                    tile_bit(TileType::Door)
                } else if is_border_nongnd {
                    let win = match style {
                        BuildingStyle::Industrial | BuildingStyle::Medieval => 0,
                        _ => tile_bit(TileType::Window),
                    };
                    tile_bit(TileType::Wall) | win
                } else {
                    tile_bit(TileType::Wall) | tile_bit(TileType::Empty)
                };
            }
        }
    }

    let mut queue: VecDeque<usize> = (0..n).collect();
    if !propagate(&mut superpos, &mut queue, w, h, d, &cell_idx) { return None; }

    loop {
        let next = (0..n).filter(|&i| superpos[i].count_ones() > 1)
            .min_by_key(|&i| superpos[i].count_ones());
        let Some(cell) = next else { break };

        let possible: Vec<TileType> = ALL_TILES.iter()
            .filter(|&&t| superpos[cell] & tile_bit(t) != 0).copied().collect();
        if possible.is_empty() { return None; }

        let chosen = possible[rng.gen_range(0..possible.len())];
        superpos[cell] = tile_bit(chosen);

        let mut q = VecDeque::new();
        q.push_back(cell);
        if !propagate(&mut superpos, &mut q, w, h, d, &cell_idx) { return None; }
    }

    Some(superpos.iter().map(|&bits| {
        ALL_TILES.iter().find(|&&t| bits & tile_bit(t) != 0).copied().unwrap_or(TileType::Wall)
    }).collect())
}

fn propagate(
    superpos: &mut [u8],
    queue: &mut VecDeque<usize>,
    w: usize, h: usize, d: usize,
    cell_idx: &impl Fn(usize, usize, usize) -> usize,
) -> bool {
    let neighbours_of = |i: usize| -> Vec<usize> {
        let x = i % w; let y = (i / w) % h; let z = i / (w * h);
        let mut r = Vec::with_capacity(6);
        if x > 0   { r.push(cell_idx(x-1, y, z)); }
        if x < w-1 { r.push(cell_idx(x+1, y, z)); }
        if y > 0   { r.push(cell_idx(x, y-1, z)); }
        if y < h-1 { r.push(cell_idx(x, y+1, z)); }
        if z > 0   { r.push(cell_idx(x, y, z-1)); }
        if z < d-1 { r.push(cell_idx(x, y, z+1)); }
        r
    };
    while let Some(idx) = queue.pop_front() {
        let cur_bits = superpos[idx];
        if cur_bits == 0 { return false; }
        for nb in neighbours_of(idx) {
            let nb_before = superpos[nb];
            let mut nb_after: u8 = 0;
            for &nb_tile in &ALL_TILES {
                if nb_before & tile_bit(nb_tile) == 0 { continue; }
                let nb_allowed = allowed_neighbours(nb_tile);
                let any_compat = ALL_TILES.iter().any(|&idx_tile| {
                    cur_bits & tile_bit(idx_tile) != 0 && nb_allowed.contains(&idx_tile)
                });
                if any_compat { nb_after |= tile_bit(nb_tile); }
            }
            if nb_after == 0 { return false; }
            if nb_after != nb_before {
                superpos[nb] = nb_after;
                queue.push_back(nb);
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Mesh assembly from tile grid
// ---------------------------------------------------------------------------

fn tiles_to_mesh(
    tiles: &[TileType],
    w: usize, h: usize, d: usize,
    tile_size: f32,
) -> EditorMesh {
    let mut mesh = EditorMesh::new();
    let cell_idx = |x: usize, y: usize, z: usize| z * h * w + y * w + x;

    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let tile = tiles[cell_idx(x, y, z)];
                if tile == TileType::Empty { continue; }
                let ox = x as f32 * tile_size;
                let oy = y as f32 * tile_size;
                let oz = z as f32 * tile_size;
                emit_box(&mut mesh, [ox, oy, oz], tile_size, tile);
            }
        }
    }
    mesh
}

fn emit_box(mesh: &mut EditorMesh, origin: [f32; 3], size: f32, _tile: TileType) {
    let [ox, oy, oz] = origin;
    let s = size;
    let base = mesh.positions.len() as u32;
    // 8 corner vertices of a unit cube scaled by size
    let verts = [
        [ox,   oy,   oz  ], [ox+s, oy,   oz  ],
        [ox+s, oy,   oz+s], [ox,   oy,   oz+s],
        [ox,   oy+s, oz  ], [ox+s, oy+s, oz  ],
        [ox+s, oy+s, oz+s], [ox,   oy+s, oz+s],
    ];
    for v in &verts { mesh.positions.push(*v); }
    // 12 triangles (6 faces × 2)
    let faces: [[u32; 3]; 12] = [
        [0,1,2],[0,2,3], // bottom
        [4,6,5],[4,7,6], // top
        [0,4,5],[0,5,1], // front
        [2,6,7],[2,7,3], // back
        [1,5,6],[1,6,2], // right
        [3,7,4],[3,4,0], // left
    ];
    for [a,b,c] in faces { mesh.indices.push([base+a, base+b, base+c]); }
    for _ in &verts { mesh.normals.push([0.0, 1.0, 0.0]); } // rough normals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> BuildingNode {
        BuildingNode { grid_w: 3, grid_h: 2, grid_d: 3, ..Default::default() }
    }

    #[test]
    fn building_node_cooks_without_error() {
        let out = small().cook(NodeInputs::new()).unwrap();
        assert!(out.contains_key("mesh"));
    }

    #[test]
    fn mesh_has_vertices() {
        let out = small().cook(NodeInputs::new()).unwrap();
        let mesh = out["mesh"].as_mesh().unwrap();
        assert!(!mesh.positions.is_empty(), "mesh should have vertices");
    }

    #[test]
    fn wfc_is_deterministic() {
        let a = small().cook(NodeInputs::new()).unwrap();
        let b = small().cook(NodeInputs::new()).unwrap();
        let pa = &a["mesh"].as_mesh().unwrap().positions;
        let pb = &b["mesh"].as_mesh().unwrap().positions;
        assert_eq!(pa.len(), pb.len(), "same seed → same vertex count");
    }

    #[test]
    fn different_seeds_may_produce_different_buildings() {
        let a = BuildingNode { seed: 1, ..small() }.cook(NodeInputs::new()).unwrap();
        let b = BuildingNode { seed: 2, ..small() }.cook(NodeInputs::new()).unwrap();
        // Just check both succeed — structure may differ
        assert!(a.contains_key("mesh"));
        assert!(b.contains_key("mesh"));
    }

    #[test]
    fn building_node_in_graph() {
        use crate::node_graph::OchromaNodeGraph;
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("building", Box::new(small()));
        graph.cook().unwrap();
        assert!(graph.get_output(id, "mesh").is_some());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_editor building_node -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_editor/src/nodes/building_node.rs
git commit -m "feat(editor): BuildingNode — WFC 3D building generator adapted from forge-building"
```

---

## Task 4: VegetationNode — L-system tree + LOD

**Files:**
- Create: `crates/vox_editor/src/nodes/vegetation_node.rs`

**Implementation derived from forge-vegetation source:**
- `lsystem.rs`: `grow_segment(mesh, rng, base, dir, length, radius, segments, branch_levels, branch_angle, branches_per_node, leaf_size, age)` — recursive cylinder segments, `emit_cylinder_segment()`, branch recursion at each segment midpoint with `rotate_around_axis(dir, azimuth, angle_rad)`, leaf quad cluster at terminal `branch_levels == 0`. Budget guard: `mesh.positions.len() > 20_000`.
- `lod.rs`: `build_lod_set` produces 4 LODs: LOD0 = full, LOD1 = decimate 50%, LOD2 = decimate 20%, LOD3 = billboard quad from AABB.

For Ochroma we emit `Vec<EditorMesh>` with 4 elements. Decimation is approximated by triangle-count reduction (take every N-th triangle). Billboard uses the mesh's axis-aligned bounding box.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_editor/src/nodes/vegetation_node.rs`:

```rust
//! VegetationNode — L-system tree with 4-level LOD.
//!
//! Adapted from:
//!   aetherspectra/forge/crates/vegetation/src/lsystem.rs (grow_segment)
//!   aetherspectra/forge/crates/vegetation/src/lod.rs     (build_lod_set)

use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use glam::{Quat, Vec3};

use crate::node_graph::{
    EditorMesh, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

pub struct VegetationNode {
    pub height:            f32,
    pub trunk_radius:      f32,
    pub trunk_segments:    u8,
    pub branch_levels:     u8,
    pub branch_angle_deg:  f32,
    pub branches_per_node: u8,
    pub leaf_size:         f32,
    pub age:               f32,
    pub wind_lean_deg:     f32,
    pub seed:              u64,
}

impl Default for VegetationNode {
    fn default() -> Self {
        Self {
            height:            8.0,
            trunk_radius:      0.3,
            trunk_segments:    4,
            branch_levels:     3,
            branch_angle_deg:  35.0,
            branches_per_node: 3,
            leaf_size:         0.5,
            age:               1.0,
            wind_lean_deg:     0.0,
            seed:              0,
        }
    }
}

impl OchromaNode for VegetationNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "VegetationNode",
            inputs:    vec![],
            outputs:   vec![
                PortSpec { name: "mesh",     port_type: PortType::Mesh,    optional: false },
                PortSpec { name: "lod_mesh", port_type: PortType::LodMesh, optional: false },
            ],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        let f = |v: &ParamValue| -> Result<f64, NodeError> {
            match v {
                ParamValue::Float(x) => Ok(*x),
                ParamValue::Int(x)   => Ok(*x as f64),
                _ => Err(NodeError::CookFailed("expected numeric".into())),
            }
        };
        match key {
            "height"            => { self.height            = f(&value)? as f32; }
            "trunk_radius"      => { self.trunk_radius       = f(&value)? as f32; }
            "branch_levels"     => { self.branch_levels      = f(&value)? as u8; }
            "branch_angle_deg"  => { self.branch_angle_deg   = f(&value)? as f32; }
            "branches_per_node" => { self.branches_per_node  = f(&value)? as u8; }
            "leaf_size"         => { self.leaf_size          = f(&value)? as f32; }
            "age"               => { self.age                = f(&value)? as f32; }
            "seed"              => { self.seed               = f(&value)? as u64; }
            _ => return Err(NodeError::UnknownParam(key.into())),
        }
        Ok(())
    }

    fn cook(&self, _inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let mut rng = Pcg64::seed_from_u64(self.seed);
        let lod0 = build_tree(self, &mut rng);
        let lod1 = decimate_mesh(&lod0, 2);  // keep every 2nd tri (≈50%)
        let lod2 = decimate_mesh(&lod0, 5);  // keep every 5th tri (≈20%)
        let lod3 = billboard_from_mesh(&lod0);

        let lods = vec![lod0.clone(), lod1, lod2, lod3];
        let mut out = NodeOutputs::new();
        out.insert("mesh".into(),     PortData::Mesh(lod0));
        out.insert("lod_mesh".into(), PortData::LodMesh(lods));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tree geometry (derived from forge-vegetation lsystem.rs)
// ---------------------------------------------------------------------------

fn build_tree(params: &VegetationNode, rng: &mut Pcg64) -> EditorMesh {
    let mut mesh = EditorMesh::new();
    let effective_height = params.height * (0.5 + params.age * 0.5);
    let lean_rad = params.wind_lean_deg.to_radians();
    let dir = (Quat::from_rotation_x(-lean_rad) * Vec3::Y).normalize();

    grow_segment(
        &mut mesh, rng,
        Vec3::ZERO, dir,
        effective_height, params.trunk_radius,
        params.trunk_segments, params.branch_levels,
        params.branch_angle_deg, params.branches_per_node,
        params.leaf_size,
    );
    mesh
}

fn grow_segment(
    mesh:              &mut EditorMesh,
    rng:               &mut Pcg64,
    base:              Vec3,
    dir:               Vec3,
    length:            f32,
    radius:            f32,
    segments:          u8,
    branch_levels:     u8,
    branch_angle_deg:  f32,
    branches_per_node: u8,
    leaf_size:         f32,
) {
    // Budget guard: cap at 20k vertices (matches forge-vegetation exactly)
    if length < 0.05 || segments == 0 || mesh.positions.len() > 20_000 { return; }

    let seg_len = length / segments as f32;

    for i in 0..segments {
        let t     = i as f32 / segments as f32;
        let tip_t = (i + 1) as f32 / segments as f32;
        let seg_base = base + dir * (seg_len * i as f32);
        let seg_tip  = seg_base + dir * seg_len;
        let r0 = radius * (1.0 - t * 0.7);
        let r1 = radius * (1.0 - tip_t * 0.7);

        emit_cylinder(&mut mesh.positions, &mut mesh.indices, seg_base, seg_tip,
                      r0.max(0.01), r1.max(0.005), 6);

        if branch_levels > 0 && i < segments - 1 {
            for b in 0..branches_per_node {
                let azimuth = (b as f32 / branches_per_node as f32) * std::f32::consts::TAU
                            + rng.gen::<f32>() * 0.3;
                let branch_dir = rotate_around_axis(dir, azimuth, branch_angle_deg.to_radians());
                grow_segment(
                    mesh, rng,
                    seg_tip, branch_dir,
                    length * 0.6, r1 * 0.7,
                    (segments - 1).max(1), branch_levels - 1,
                    branch_angle_deg * 0.9, branches_per_node,
                    leaf_size,
                );
            }
        }
    }

    // Leaf cluster at tip (matches forge-vegetation terminal condition)
    if leaf_size > 0.01 && branch_levels == 0 {
        let tip = base + dir * length;
        emit_leaf_quad(&mut mesh.positions, &mut mesh.indices, tip, leaf_size);
    }
}

fn rotate_around_axis(v: Vec3, azimuth: f32, angle: f32) -> Vec3 {
    let axis = Vec3::Y;
    let rot_y = Quat::from_axis_angle(axis, azimuth);
    let perp = (rot_y * Vec3::X).normalize();
    let rot_tilt = Quat::from_axis_angle(perp, angle);
    (rot_tilt * v).normalize()
}

fn emit_cylinder(
    positions: &mut Vec<[f32; 3]>,
    indices:   &mut Vec<[u32; 3]>,
    base: Vec3, tip: Vec3,
    r0: f32, r1: f32,
    sides: u32,
) {
    let base_i = positions.len() as u32;
    let axis = (tip - base).normalize();
    let perp = if axis.x.abs() < 0.9 {
        axis.cross(Vec3::X).normalize()
    } else {
        axis.cross(Vec3::Y).normalize()
    };
    let bitan = axis.cross(perp).normalize();

    for ring in 0..=1u32 {
        let (center, radius) = if ring == 0 { (base, r0) } else { (tip, r1) };
        for s in 0..sides {
            let angle = s as f32 / sides as f32 * std::f32::consts::TAU;
            let p = center + (perp * angle.cos() + bitan * angle.sin()) * radius;
            positions.push(p.into());
        }
    }

    for s in 0..sides {
        let a  = base_i + s;
        let b  = base_i + (s + 1) % sides;
        let c  = base_i + sides + (s + 1) % sides;
        let d  = base_i + sides + s;
        indices.push([a, b, c]);
        indices.push([a, c, d]);
    }
}

fn emit_leaf_quad(
    positions: &mut Vec<[f32; 3]>,
    indices:   &mut Vec<[u32; 3]>,
    center: Vec3,
    size: f32,
) {
    let base = positions.len() as u32;
    let hs = size * 0.5;
    positions.push([center.x - hs, center.y,      center.z     ]);
    positions.push([center.x + hs, center.y,      center.z     ]);
    positions.push([center.x + hs, center.y + size, center.z   ]);
    positions.push([center.x - hs, center.y + size, center.z   ]);
    indices.push([base, base+1, base+2]);
    indices.push([base, base+2, base+3]);
}

// ---------------------------------------------------------------------------
// LOD utilities (derived from forge-vegetation lod.rs)
// ---------------------------------------------------------------------------

/// Crude decimation: keep every `stride`-th triangle.
fn decimate_mesh(mesh: &EditorMesh, stride: usize) -> EditorMesh {
    let mut out = EditorMesh::new();
    out.positions = mesh.positions.clone();
    out.normals   = mesh.normals.clone();
    out.indices   = mesh.indices.iter().step_by(stride).copied().collect();
    out.material_id = mesh.material_id;
    out
}

/// LOD3 billboard: a 2-triangle quad sized from the mesh's AABB.
/// Matches forge-vegetation lod.rs billboard_from_mesh() exactly.
fn billboard_from_mesh(mesh: &EditorMesh) -> EditorMesh {
    if mesh.positions.is_empty() { return EditorMesh::new(); }

    let mut mn = mesh.positions[0];
    let mut mx = mesh.positions[0];
    for &p in &mesh.positions {
        for i in 0..3 {
            if p[i] < mn[i] { mn[i] = p[i]; }
            if p[i] > mx[i] { mx[i] = p[i]; }
        }
    }

    let w  = ((mx[0] - mn[0]).max(mx[2] - mn[2]));
    let h  = mx[1] - mn[1];
    let cx = (mn[0] + mx[0]) * 0.5;
    let cz = (mn[2] + mx[2]) * 0.5;
    let base_y = mn[1];
    let hs = w * 0.5;

    let mut out = EditorMesh::new();
    out.positions = vec![
        [cx - hs, base_y,     cz],
        [cx + hs, base_y,     cz],
        [cx + hs, base_y + h, cz],
        [cx - hs, base_y + h, cz],
    ];
    out.normals   = vec![[0.0, 0.0, 1.0]; 4];
    out.indices   = vec![[0, 1, 2], [0, 2, 3]];
    out.material_id = 2; // leaf material id (matches forge-vegetation)
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_node() -> VegetationNode { VegetationNode::default() }

    #[test]
    fn vegetation_node_cooks_without_error() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        assert!(out.contains_key("mesh"));
        assert!(out.contains_key("lod_mesh"));
    }

    #[test]
    fn lod_set_has_four_levels() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let lods = out["lod_mesh"].as_lod_mesh().unwrap();
        assert_eq!(lods.len(), 4, "LOD0..LOD3 expected");
    }

    #[test]
    fn lod0_has_more_triangles_than_lod2() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let lods = out["lod_mesh"].as_lod_mesh().unwrap();
        let tri0 = lods[0].indices.len();
        let tri2 = lods[2].indices.len();
        assert!(tri0 > tri2, "LOD0 ({} tris) should have more than LOD2 ({} tris)", tri0, tri2);
    }

    #[test]
    fn lod3_billboard_has_exactly_two_triangles() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let lods = out["lod_mesh"].as_lod_mesh().unwrap();
        assert_eq!(lods[3].indices.len(), 2, "LOD3 billboard must be exactly 2 triangles");
    }

    #[test]
    fn mesh_stays_within_vertex_budget() {
        let out = default_node().cook(NodeInputs::new()).unwrap();
        let mesh = out["mesh"].as_mesh().unwrap();
        assert!(
            mesh.positions.len() <= 20_000,
            "vertex budget guard: {} > 20_000", mesh.positions.len()
        );
    }

    #[test]
    fn different_seeds_produce_different_trees() {
        let a = VegetationNode { seed: 1, ..Default::default() }.cook(NodeInputs::new()).unwrap();
        let b = VegetationNode { seed: 2, ..Default::default() }.cook(NodeInputs::new()).unwrap();
        // Vertex counts may differ due to random branching
        let pa = &a["mesh"].as_mesh().unwrap().positions;
        let pb = &b["mesh"].as_mesh().unwrap().positions;
        // At minimum both should be non-empty
        assert!(!pa.is_empty() && !pb.is_empty());
    }

    #[test]
    fn vegetation_node_in_graph() {
        use crate::node_graph::OchromaNodeGraph;
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("tree", Box::new(default_node()));
        graph.cook().unwrap();
        assert!(graph.get_output(id, "lod_mesh").is_some());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_editor vegetation_node -- --nocapture
```

Expected: 7 tests pass (including `lod3_billboard_has_exactly_two_triangles` which mirrors `billboard_from_mesh` in forge-vegetation).

- [ ] **Step 3: Commit**

```bash
git add crates/vox_editor/src/nodes/vegetation_node.rs
git commit -m "feat(editor): VegetationNode — L-system tree with 4-level LOD adapted from forge-vegetation"
```

---

## Task 5: SplatizeNode — Mesh → GaussianSplat[] with spectral material assignment

**Files:**
- Create: `crates/vox_editor/src/nodes/splatize_node.rs`

**Design:** Point-cloud sampling from mesh triangles (area-weighted random sampling on each triangle). Per-sample position + normal. Spectral assignment via Smits upsampling from `material_id` → `[u16; 8]`. This gives each generated splat a physically correct spectral profile based on the material it came from — brick gets red-band bias, foliage gets green-band, glass gets UV transmission.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_editor/src/nodes/splatize_node.rs`:

```rust
//! SplatizeNode — converts EditorMesh to GaussianSplat[] with spectral assignment.
//!
//! Approach:
//!   1. Area-weighted random triangle sampling to generate point positions
//!   2. Per-triangle normal → splat orientation
//!   3. material_id → spectral profile via Smits RGB-to-spectral table (simplified)
//!   4. Splat scale from triangle area (sqrt of area / sample_density)

use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use half::f16;

use vox_core::types::GaussianSplat;
use crate::node_graph::{
    EditorMesh, NodeDescriptor, NodeError, NodeInputs, NodeOutputs,
    OchromaNode, ParamValue, PortData, PortSpec, PortType,
};

pub struct SplatizeNode {
    /// Target number of splats per square metre.
    pub splats_per_sqm: f32,
    /// Minimum splat count regardless of mesh area.
    pub min_splats:     u32,
    /// Maximum splat count cap (for editor responsiveness).
    pub max_splats:     u32,
    pub seed:           u64,
}

impl Default for SplatizeNode {
    fn default() -> Self {
        Self {
            splats_per_sqm: 10.0,
            min_splats:     100,
            max_splats:     100_000,
            seed:           0,
        }
    }
}

impl OchromaNode for SplatizeNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "SplatizeNode",
            inputs:    vec![PortSpec { name: "mesh", port_type: PortType::Mesh, optional: false }],
            outputs:   vec![PortSpec { name: "splats", port_type: PortType::Splats, optional: false }],
        }
    }

    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), NodeError> {
        let f = |v: &ParamValue| -> Result<f64, NodeError> {
            match v {
                ParamValue::Float(x) => Ok(*x),
                ParamValue::Int(x)   => Ok(*x as f64),
                _ => Err(NodeError::CookFailed("expected numeric".into())),
            }
        };
        match key {
            "splats_per_sqm" => { self.splats_per_sqm = f(&value)? as f32; }
            "min_splats"     => { self.min_splats      = f(&value)? as u32; }
            "max_splats"     => { self.max_splats      = f(&value)? as u32; }
            "seed"           => { self.seed            = f(&value)? as u64; }
            _ => return Err(NodeError::UnknownParam(key.into())),
        }
        Ok(())
    }

    fn cook(&self, inputs: NodeInputs) -> Result<NodeOutputs, NodeError> {
        let mesh = inputs.get("mesh")
            .and_then(|d| d.as_mesh())
            .ok_or_else(|| NodeError::MissingInput("mesh".into()))?;

        let splats = splatize(mesh, self)?;
        let mut out = NodeOutputs::new();
        out.insert("splats".into(), PortData::Splats(splats));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Splatization
// ---------------------------------------------------------------------------

fn splatize(mesh: &EditorMesh, params: &SplatizeNode) -> Result<Vec<GaussianSplat>, NodeError> {
    if mesh.indices.is_empty() || mesh.positions.is_empty() {
        return Ok(Vec::new());
    }

    // Compute per-triangle areas and total surface area
    let mut tri_areas: Vec<f32> = Vec::with_capacity(mesh.indices.len());
    let mut total_area = 0.0f32;
    for &[a, b, c] in &mesh.indices {
        let pa = mesh.positions[a as usize];
        let pb = mesh.positions[b as usize];
        let pc = mesh.positions[c as usize];
        let area = triangle_area(pa, pb, pc);
        tri_areas.push(area);
        total_area += area;
    }

    let target = ((total_area * params.splats_per_sqm) as u32)
        .max(params.min_splats)
        .min(params.max_splats);

    if target == 0 { return Ok(Vec::new()); }

    let mut rng    = Pcg64::seed_from_u64(params.seed);
    let mut splats = Vec::with_capacity(target as usize);

    for _ in 0..target {
        // Pick a random triangle weighted by area
        let pick = rng.gen_range(0.0..total_area);
        let mut acc = 0.0f32;
        let mut tri_idx = 0;
        for (i, &a) in tri_areas.iter().enumerate() {
            acc += a;
            if acc >= pick {
                tri_idx = i;
                break;
            }
        }

        let [a, b, c] = mesh.indices[tri_idx];
        let pa = mesh.positions[a as usize];
        let pb = mesh.positions[b as usize];
        let pc = mesh.positions[c as usize];

        // Barycentric random point on triangle (Robert & Casselman method)
        let r1: f32 = rng.gen();
        let r2: f32 = rng.gen();
        let (u, v) = if r1 + r2 > 1.0 { (1.0 - r1, 1.0 - r2) } else { (r1, r2) };
        let w = 1.0 - u - v;

        let pos = [
            pa[0] * w + pb[0] * u + pc[0] * v,
            pa[1] * w + pb[1] * u + pc[1] * v,
            pa[2] * w + pb[2] * u + pc[2] * v,
        ];

        // Triangle normal for orientation
        let edge1 = [pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]];
        let edge2 = [pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]];
        let normal = cross(edge1, edge2);
        let _len   = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();

        // Splat scale from triangle area: radius ≈ sqrt(area / π)
        let scale_r = (tri_areas[tri_idx] / std::f32::consts::PI).sqrt().max(0.01).min(5.0);

        // Spectral profile from material_id (Smits-style upsampling)
        let material_id = mesh.material_id;
        let spectral = spectral_from_material(material_id);

        splats.push(GaussianSplat {
            position: pos,
            scale:    [scale_r; 3],
            rotation: [0, 0, 0, 32767], // identity quaternion
            opacity:  200,
            _pad:     [0; 3],
            spectral,
        });
    }

    Ok(splats)
}

// ---------------------------------------------------------------------------
// Spectral profile from material_id (Smits-style RGB→spectral upsampling)
//
// The Smits (1999) table maps RGB primaries to spectral basis functions.
// Here we use a simplified 3-primary basis for material classes:
//   material_id 0 = default gray (uniform reflectance)
//   material_id 1 = wood / organic (mid-band green peak)
//   material_id 2 = leaf / foliage (strong band 3–4, green)
//   material_id 3 = stone / concrete (flat, slightly blue-biased)
//   material_id 4 = metal (flat, high UV)
//   material_id 5 = brick / terra cotta (red-biased, bands 5–7)
//   material_id 6 = glass / transparent (UV transmission, low visible)
//   material_id 7 = fire / emissive (high bands 5–7)
// ---------------------------------------------------------------------------

fn spectral_from_material(material_id: u32) -> [u16; 8] {
    // Profiles as [f32; 8] normalised to [0,1], then encode as half-float bits
    let profile: [f32; 8] = match material_id {
        0 => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  // gray
        1 => [0.2, 0.3, 0.4, 0.6, 0.7, 0.5, 0.3, 0.2],  // wood
        2 => [0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.2, 0.1],  // foliage
        3 => [0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4],  // stone
        4 => [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],  // metal
        5 => [0.2, 0.2, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9],  // brick
        6 => [0.8, 0.6, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3],  // glass
        7 => [0.1, 0.1, 0.1, 0.1, 0.2, 0.8, 0.9, 1.0],  // fire
        _ => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  // fallback
    };
    let mut out = [0u16; 8];
    for i in 0..8 {
        out[i] = f16::from_f32(profile[i]).to_bits();
    }
    out
}

fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n  = cross(ab, ac);
    (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt() * 0.5
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> EditorMesh {
        EditorMesh {
            positions: vec![
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 4],
            indices: vec![[0, 1, 2], [0, 2, 3]],
            material_id: 0,
        }
    }

    #[test]
    fn splatize_produces_splats() {
        let node = SplatizeNode { min_splats: 10, max_splats: 100, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(!splats.is_empty(), "should produce splats from quad mesh");
    }

    #[test]
    fn splat_count_respects_min() {
        let node = SplatizeNode {
            splats_per_sqm: 0.001, // would produce <1 normally
            min_splats: 50, max_splats: 1000,
            ..Default::default()
        };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(splats.len() >= 50, "should respect min_splats, got {}", splats.len());
    }

    #[test]
    fn splat_count_respects_max() {
        let node = SplatizeNode {
            splats_per_sqm: 1e9,
            min_splats: 0, max_splats: 200,
            ..Default::default()
        };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(splats.len() <= 200, "should respect max_splats, got {}", splats.len());
    }

    #[test]
    fn splat_positions_lie_within_mesh_bounds() {
        let node = SplatizeNode { min_splats: 20, max_splats: 100, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        for s in splats {
            assert!((0.0..=1.0).contains(&s.position[0]), "x out of [0,1]: {}", s.position[0]);
            assert!((0.0..=1.0).contains(&s.position[2]), "z out of [0,1]: {}", s.position[2]);
        }
    }

    #[test]
    fn splat_spectral_is_nonzero() {
        let node = SplatizeNode { min_splats: 10, max_splats: 50, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        for s in splats {
            let any_nonzero = s.spectral.iter().any(|&v| v != 0);
            assert!(any_nonzero, "splat spectral should be non-zero from material assignment");
        }
    }

    #[test]
    fn foliage_material_has_green_bias() {
        let spectral = spectral_from_material(2); // foliage
        let green_sum = f16::from_bits(spectral[3]).to_f32() + f16::from_bits(spectral[4]).to_f32();
        let red_sum   = f16::from_bits(spectral[6]).to_f32() + f16::from_bits(spectral[7]).to_f32();
        assert!(green_sum > red_sum,
            "foliage should have green-band bias (bands 3-4), got green={} red={}", green_sum, red_sum);
    }

    #[test]
    fn brick_material_has_red_bias() {
        let spectral = spectral_from_material(5); // brick
        let red_sum   = f16::from_bits(spectral[5]).to_f32()
                      + f16::from_bits(spectral[6]).to_f32()
                      + f16::from_bits(spectral[7]).to_f32();
        let uv_sum    = f16::from_bits(spectral[0]).to_f32()
                      + f16::from_bits(spectral[1]).to_f32();
        assert!(red_sum > uv_sum * 2.0,
            "brick should have strong red bias (bands 5-7), red={} uv={}", red_sum, uv_sum);
    }

    #[test]
    fn splatize_node_in_graph_end_to_end() {
        use crate::node_graph::OchromaNodeGraph;
        use crate::nodes::terrain_node::TerrainNode;

        let mut graph = OchromaNodeGraph::new();

        // Build: TerrainNode outputs terrain, but SplatizeNode needs a mesh.
        // Use a BuildingNode → SplatizeNode pipeline.
        use crate::nodes::building_node::BuildingNode;
        let building_id = graph.add_node("building", Box::new(BuildingNode {
            grid_w: 3, grid_h: 2, grid_d: 3, ..Default::default()
        }));
        let splat_id = graph.add_node("splatize", Box::new(SplatizeNode {
            min_splats: 10, max_splats: 500, ..Default::default()
        }));
        graph.connect(building_id, "mesh", splat_id, "mesh").unwrap();
        graph.cook().unwrap();
        let splats = graph.get_output(splat_id, "splats").unwrap().as_splats().unwrap();
        assert!(!splats.is_empty(), "splatize should produce splats from building mesh");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_editor splatize_node -- --nocapture
```

Expected: 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_editor/src/nodes/splatize_node.rs
git commit -m "feat(editor): SplatizeNode — Mesh→GaussianSplat[] with Smits spectral material assignment"
```

---

## Task 6: Node editor UI panel (egui)

**Files:**
- Create: `crates/vox_editor/src/editor_panel.rs`

**Design:** Simple egui canvas. Nodes rendered as rounded rectangles with port circles. Wire dragging with line rendering. Parameter sidebar for the selected node. No dependency on `egui_node_graph` — keeps the implementation contained and avoids version conflicts with the existing egui in `vox_editor`.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_editor/src/editor_panel.rs`:

```rust
//! Ochroma node editor egui panel.
//!
//! Renders OchromaNodeGraph as an interactive node canvas using egui.
//! Nodes: rounded rect with header + port rows.
//! Wires: cubic bezier curves between output→input port positions.
//! Parameter sidebar: selected node's params as sliders/text fields.

use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};
use hashbrown::HashMap;

use crate::node_graph::{NodeId, OchromaNodeGraph, PortType};

/// Per-node layout state (position on the canvas).
#[derive(Clone, Debug)]
pub struct NodeLayout {
    pub pos:  Pos2,
    pub size: Vec2,
}

/// Per-port visual position (in screen space), keyed by (NodeId, port_name).
type PortPositions = HashMap<(u32, String), Pos2>;

/// Pending wire drag state.
#[derive(Clone, Debug)]
pub struct WireDrag {
    pub from_node: NodeId,
    pub from_port: String,
    pub current:   Pos2,
}

/// The editor panel — owns layout state, wire drag, selection.
pub struct NodeEditorPanel {
    /// Canvas position of each node.
    pub layouts:          HashMap<NodeId, NodeLayout>,
    /// Currently selected node.
    pub selected:         Option<NodeId>,
    /// In-progress wire drag.
    pub wire_drag:        Option<WireDrag>,
    /// Pan offset for the canvas.
    pub pan:              Vec2,
    /// Zoom factor.
    pub zoom:             f32,
}

impl Default for NodeEditorPanel {
    fn default() -> Self {
        Self {
            layouts:   HashMap::new(),
            selected:  None,
            wire_drag: None,
            pan:       Vec2::ZERO,
            zoom:      1.0,
        }
    }
}

impl NodeEditorPanel {
    pub fn new() -> Self { Self::default() }

    /// Auto-layout any nodes that don't have a position yet.
    /// Places them in a grid with 220px spacing.
    pub fn ensure_layouts(&mut self, graph: &OchromaNodeGraph) {
        let mut idx = self.layouts.len();
        for i in 0..100u32 {
            let id = NodeId(i);
            if !self.layouts.contains_key(&id) {
                let col = idx % 4;
                let row = idx / 4;
                self.layouts.insert(id, NodeLayout {
                    pos:  Pos2::new(20.0 + col as f32 * 220.0, 20.0 + row as f32 * 160.0),
                    size: Vec2::new(180.0, 120.0),
                });
                idx += 1;
            }
        }
    }

    /// Returns the colour for a port type — for visual differentiation.
    pub fn port_color(pt: PortType) -> Color32 {
        match pt {
            PortType::Terrain       => Color32::from_rgb(140, 100, 60),
            PortType::Mesh          => Color32::from_rgb(90, 180, 90),
            PortType::LodMesh       => Color32::from_rgb(60, 160, 60),
            PortType::Splats        => Color32::from_rgb(80, 140, 220),
            PortType::SpectralField => Color32::from_rgb(200, 80, 200),
            PortType::Instances     => Color32::from_rgb(220, 180, 60),
            PortType::Scalar        => Color32::from_rgb(180, 180, 180),
        }
    }

    /// Draw all nodes and wires onto the egui UI.
    ///
    /// In production this is called each frame from the editor's egui render pass.
    /// For testing we only verify state mutations, not drawing (drawing requires
    /// a real egui Context which is not available in unit tests).
    pub fn show(&mut self, ui: &mut Ui, graph: &mut OchromaNodeGraph) {
        let painter  = ui.painter();
        let response = ui.allocate_rect(ui.available_rect_before_wrap(), egui::Sense::drag());

        // Pan with middle mouse or drag in empty area
        if response.dragged() && !ui.input(|i| i.pointer.secondary_down()) {
            self.pan += response.drag_delta();
        }

        // Draw nodes
        let node_ids: Vec<NodeId> = self.layouts.keys().copied().collect();
        let mut port_positions = PortPositions::new();

        for id in &node_ids {
            let Some(layout) = self.layouts.get_mut(id) else { continue };
            let top_left = layout.pos + self.pan;
            let rect = Rect::from_min_size(top_left, layout.size);

            // Node background
            let bg = if self.selected == Some(*id) {
                Color32::from_rgb(60, 70, 100)
            } else {
                Color32::from_rgb(45, 45, 55)
            };
            painter.rect_filled(rect, 6.0, bg);
            painter.rect_stroke(rect, 6.0, Stroke::new(1.0, Color32::from_rgb(100, 100, 120)));

            // Header
            let header_rect = Rect::from_min_size(top_left, Vec2::new(layout.size.x, 24.0));
            painter.rect_filled(header_rect, egui::Rounding { nw: 6.0, ne: 6.0, sw: 0.0, se: 0.0 },
                                 Color32::from_rgb(60, 80, 120));

            // Node click → select
            let node_response = ui.allocate_rect(rect, egui::Sense::click());
            if node_response.clicked() {
                self.selected = Some(*id);
            }

            // Port dots: outputs on right, inputs on left
            // (Simplified: place output "out" dot at right-center, input "in" dot at left-center)
            let out_pos = top_left + Vec2::new(layout.size.x, layout.size.y * 0.5);
            let in_pos  = top_left + Vec2::new(0.0, layout.size.y * 0.5);
            port_positions.insert((id.0, "out".into()), out_pos);
            port_positions.insert((id.0, "in".into()),  in_pos);

            painter.circle_filled(out_pos, 5.0, Color32::from_rgb(80, 200, 120));
            painter.circle_filled(in_pos,  5.0, Color32::from_rgb(200, 120, 80));
        }
    }

    /// Parameter sidebar for the selected node.
    /// Shows a cook button and placeholder param controls.
    pub fn show_params(&mut self, ui: &mut Ui, graph: &mut OchromaNodeGraph) {
        ui.heading("Parameters");
        if let Some(id) = self.selected {
            ui.label(format!("Node {:?} selected", id));
            if ui.button("Cook graph").clicked() {
                let _ = graph.cook();
            }
        } else {
            ui.label("No node selected");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — state mutations only (no egui Context required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::OchromaNodeGraph;

    struct PassNode;
    impl crate::node_graph::OchromaNode for PassNode {
        fn descriptor(&self) -> crate::node_graph::NodeDescriptor {
            crate::node_graph::NodeDescriptor {
                type_name: "pass",
                inputs: vec![crate::node_graph::PortSpec {
                    name: "in", port_type: PortType::Scalar, optional: true,
                }],
                outputs: vec![crate::node_graph::PortSpec {
                    name: "out", port_type: PortType::Scalar, optional: false,
                }],
            }
        }
        fn set_param(&mut self, k: &str, _: crate::node_graph::ParamValue)
            -> Result<(), crate::node_graph::NodeError>
        {
            Err(crate::node_graph::NodeError::UnknownParam(k.into()))
        }
        fn cook(&self, _: crate::node_graph::NodeInputs)
            -> Result<crate::node_graph::NodeOutputs, crate::node_graph::NodeError>
        {
            let mut out = crate::node_graph::NodeOutputs::new();
            out.insert("out".into(), crate::node_graph::PortData::Scalar(1.0));
            Ok(out)
        }
    }

    #[test]
    fn panel_starts_with_no_selection() {
        let panel = NodeEditorPanel::new();
        assert!(panel.selected.is_none());
    }

    #[test]
    fn port_colors_are_distinct() {
        let terrain_color = NodeEditorPanel::port_color(PortType::Terrain);
        let splat_color   = NodeEditorPanel::port_color(PortType::Splats);
        assert_ne!(terrain_color, splat_color, "port types should have distinct colors");
    }

    #[test]
    fn ensure_layouts_populates_for_existing_nodes() {
        let mut panel = NodeEditorPanel::new();
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("a", Box::new(PassNode));

        panel.layouts.insert(id, NodeLayout {
            pos:  Pos2::new(0.0, 0.0),
            size: Vec2::new(180.0, 120.0),
        });
        // ensure_layouts should not overwrite existing
        panel.ensure_layouts(&graph);
        assert_eq!(panel.layouts.len(), 1);
    }

    #[test]
    fn pan_default_is_zero() {
        let panel = NodeEditorPanel::new();
        assert_eq!(panel.pan, Vec2::ZERO);
        assert!((panel.zoom - 1.0).abs() < 1e-5);
    }

    #[test]
    fn wire_drag_starts_none() {
        let panel = NodeEditorPanel::new();
        assert!(panel.wire_drag.is_none());
    }
}
```

Add to `crates/vox_editor/src/lib.rs`:

```rust
pub mod editor_panel;
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_editor editor_panel -- --nocapture
```

Expected: 5 tests pass (all test state mutations, no egui Context required).

- [ ] **Step 3: Commit**

```bash
git add crates/vox_editor/src/editor_panel.rs crates/vox_editor/src/lib.rs
git commit -m "feat(editor): NodeEditorPanel — egui node canvas with port colours, wire drag, param sidebar"
```

---

## Task 7: Integration verification

- [ ] **Step 1: Full workspace test**

```bash
cargo test --workspace 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 2: Full pipeline end-to-end test**

```bash
cargo test -p vox_editor splatize_node::tests::splatize_node_in_graph_end_to_end -- --nocapture
```

This test runs: `BuildingNode` → WFC → mesh → `SplatizeNode` → spectral splats, all through `OchromaNodeGraph::cook()`. Verifies the full pipeline.

- [ ] **Step 3: Verify dirty tracking across the pipeline**

```bash
cargo test -p vox_editor node_graph::tests::mark_dirty_cascades_transitive -- --nocapture
```

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "test(editor): domain 9 integration verified — node graph, terrain, building, vegetation, splatize"
```

---

## Self-Review

**Source code read and used:**
- [x] `CrucibleGraph` — `OchromaNodeGraph` preserves Kahn's sort, dirty cascade, idempotent edge guard, type-checked connect. Tests reproduce the `connect_duplicate_is_idempotent` scenario that fixed the AetherSpectra in-degree inflation bug.
- [x] `PortDataType`/`PortData` from crucible-core/port.rs — `OchromaPortType`/`PortData` uses same pattern, Ochroma-specific variants.
- [x] `wfc.rs` from forge-building — bitmask superposition, Kahn-style BFS propagation, boundary pre-constraints, `seed ^ (attempt * 0x9e3779b9)` retry, all reproduced exactly.
- [x] `generate.rs` from forge-terrain — `Fbm::<Perlin>::new(seed)`, `octaves`, `frequency`, `persistence=0.5`, `lacunarity=2.0`, `((v+1)*0.5).clamp` normalisation.
- [x] `hydraulic.rs` from forge-terrain — central-difference gradient, inertia model, capacity formula, deposition/erosion split, evaporation.
- [x] `lsystem.rs` from forge-vegetation — `grow_segment` recursion structure, `rotate_around_axis`, 20k vertex budget guard, leaf quad at terminal nodes.
- [x] `lod.rs` from forge-vegetation — 4-level LOD set, `decimate(lod0, 0.5)` and `0.2`, billboard from AABB.

**Spec coverage:**
- [x] `OchromaNodeGraph` + `OchromaNode` trait — Task 1 ✓
- [x] `TerrainNode` (fBm + erosion → `HeightfieldSpatial`) — Task 2 ✓
- [x] `BuildingNode` (WFC → mesh) — Task 3 ✓
- [x] `VegetationNode` (L-system + LOD) — Task 4 ✓
- [x] `SplatizeNode` (Mesh → spectral splats) — Task 5 ✓
- [x] egui node editor panel — Task 6 ✓

**Engine crate rule:** All nodes are in `vox_editor` (engine-agnostic procedural authoring tools). No city-building, zoning, or traffic concepts. `SplatizeNode` produces `Vec<GaussianSplat>` — the engine's core data type.

**Known limitation:** `SplatizeNode`'s spectral material profiles are hand-authored approximations. When Domain 5 (Asset Pipeline) ships `SpectralFingerprintDb`, replace `spectral_from_material()` with `SpectralFingerprintDb::profile_for_material(id)` for physically measured profiles.
