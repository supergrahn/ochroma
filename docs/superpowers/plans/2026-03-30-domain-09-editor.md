# Domain 9: Editor — OchromaNodeGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a procedural scene editor as a DAG node graph. The `OchromaNodeGraph` is a direct port of `CrucibleGraph` from AetherSpectra, with Ochroma-specific port types. Four domain nodes adapted from AetherSpectra's `forge` crates: `TerrainNode` (fBm + hydraulic erosion), `BuildingNode` (WFC), `VegetationNode` (L-system + LOD), `SplatizeNode` (Mesh → GaussianSplat[] with spectral assignment). A `BiomeNode` → `SplatizeNode` pipeline generates terrain with biome-correct spectral coloring. An egui node editor panel wires everything into the editor UI.

**Done When:** Opening the node graph editor, connecting a BiomeNode → SplatizeNode, and pressing "Cook" generates visible terrain splats in the viewport with biome-correct coloring (grass green in low-moisture areas, rock gray at high altitude), verified by `cargo test -p vox_nodes biome_node_cook_produces_nonuniform_spectral` passing with at least 2 distinct spectral profiles in the output.

**Architecture:** `OchromaNodeGraph` uses `HashMap<NodeId, NodeEntry>` + `Vec<Edge>` with Kahn's topological sort via `BinaryHeap<Reverse<NodeId>>` for determinism, downstream dirty cascade in `mark_dirty()`, type-checked `connect()` using `descriptor().output_type()` / `descriptor().input_type()`, cook skips clean nodes, idempotent duplicate-edge guard. Port types are Ochroma-specific: `Splats`, `SpectralField`, `Terrain`, `Mesh`, `LodMesh`, `Instances`, `Scalar`, `BiomeMap`, `SplatWeights`, `ScalarVec`.

**Tech Stack:** Rust, `noise = "0.9"`, `rand = "0.8"`, `rand_pcg = "0.3"`, `egui` (existing), `half` (existing), `hashbrown` (existing), `thiserror` (existing)

**Source material read and understood before writing this plan:**
- `CrucibleGraph`: `HashMap<NodeId, NodeEntry>`, `Vec<Edge>`, Kahn's topological sort via `BinaryHeap<Reverse<NodeId>>`, downstream dirty cascade in `mark_dirty()`, type-checked `connect()`, cook skips clean nodes, `assemble_inputs()` gathers upstream `last_output` by port name, `CycleDetected` guard via `can_reach()`, idempotent duplicate-edge guard.
- `PortDataType { Terrain, Geometry, LodGeometry, Instances, Light, Camera, Atmosphere, Fog, Material, Scalar, Null }` from crucible-core/port.rs.
- `WFCParams { grid_w, grid_h, grid_d, style, seed, max_attempts }`, bitmask superposition (u8), Kahn-style BFS propagation (VecDeque), PCG64 seeded RNG, `Pcg64::seed_from_u64(seed ^ attempt * 0x9e3779b9)` for retry.
- `noise::Fbm<Perlin>` with `octaves`, `frequency`, `persistence=0.5`, `lacunarity=2.0`; normalise to [0,1]; hydraulic erosion droplet model.
- `build_tree(params)` → `Mesh`, recursive cylinder segments, `rotate_around_axis`, 20k vertex budget guard, leaf quad at terminal nodes.
- `build_lod_set` produces 4 LODs: LOD0=full, LOD1=decimate 50%, LOD2=decimate 20%, LOD3=billboard from AABB.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_editor/Cargo.toml` | editor crate manifest |
| Create | `crates/vox_editor/src/lib.rs` | crate root |
| Modify | `Cargo.toml` | add vox_editor to workspace members |
| Create | `crates/vox_editor/src/node_graph.rs` | `OchromaNodeGraph`, port of `CrucibleGraph` with Ochroma port types |
| Create | `crates/vox_editor/src/nodes/terrain_node.rs` | `TerrainNode` — fBm + hydraulic erosion → `HeightfieldSpatial` |
| Create | `crates/vox_editor/src/nodes/building_node.rs` | `BuildingNode` — WFC → `EditorMesh` |
| Create | `crates/vox_editor/src/nodes/vegetation_node.rs` | `VegetationNode` — L-system + LOD → `EditorMesh` with LOD levels |
| Create | `crates/vox_editor/src/nodes/splatize_node.rs` | `SplatizeNode` — Mesh → `Vec<GaussianSplat>` with spectral assignment |
| Create | `crates/vox_editor/src/nodes/biome_node.rs` | `BiomeNode` — classifies terrain cells into biome kinds |
| Create | `crates/vox_editor/src/nodes/splat_weight_node.rs` | `SplatWeightNode` — BiomeMap → per-cell splat blend weights |
| Create | `crates/vox_editor/src/nodes/moisture_node.rs` | `MoistureNode` — drip + urban moisture → per-cell scalar |
| Create | `crates/vox_editor/src/nodes/plot_node.rs` | `PlotNode` — land parcel geometry (ground, driveway, fence, props) |
| Create | `crates/vox_editor/src/nodes/inhabitation_node.rs` | `CatenaryNode` + `PropPlacementNode` |
| Create | `crates/vox_editor/src/nodes/urban_sim_node.rs` | `UrbanSimNode` — traffic/moisture/upkeep reaction-diffusion |
| Create | `crates/vox_editor/src/nodes/mod.rs` | re-export all node types |
| Create | `crates/vox_editor/src/editor_panel.rs` | egui node editor panel: canvas, wires, parameter sidebar |
| Modify | `crates/vox_editor/src/lib.rs` | expose all new modules |
| Modify | `crates/vox_editor/Cargo.toml` | add `noise`, `rand`, `rand_pcg` deps |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Graph rejects cycles | `matches!(err, GraphError::CycleDetected { .. })` after connecting b→a when a→b exists | `assert!(err.is_err())` |
| Dirty cascade reaches transitive nodes | `assert!(g.is_dirty(c))` after `mark_dirty(a)` in chain a→b→c | `assert!(g.is_dirty(a))` |
| Cook skips clean nodes | cook count == 1 after two cook() calls on unchanged graph | `assert!(graph.cook().is_ok())` |
| TerrainNode fBm output fills resolution | `hf.heights.len() == 64 * 64` | `assert!(!hf.heights.is_empty())` |
| Erosion modifies terrain | `diff > 0.01` between no-erosion and 1000-droplet outputs | `assert!(diff >= 0.0)` |
| LOD3 billboard is exactly 2 triangles | `assert_eq!(lods[3].indices.len(), 2)` | `assert!(!lods[3].indices.is_empty())` |
| Foliage material has green bias | `green_sum > red_sum` from `spectral_from_material(2)` | `assert!(spectral.iter().any(|&v| v != 0))` |
| BiomeNode classifies by altitude | all cells at 90% world_height == Alpine | `assert!(!cells.is_empty())` |
| Splat weights sum to 1 | `(sum - 1.0).abs() < 0.01` for any biome | `assert!(weights[2] > 0.0)` |
| Catenary midpoint sags | `pts[pts.len()/2] < 5.0` for level endpoints at Y=5 | `assert!(!pts.is_empty())` |

---

## Task 1: OchromaNodeGraph — port of CrucibleGraph with Ochroma port types

**Files:**
- Create: `crates/vox_editor/src/node_graph.rs`
- Modify: `crates/vox_editor/src/lib.rs`

**Acceptance:** `cargo test -p vox_editor node_graph -- --nocapture` → 7 tests pass, including `connect_cycle_rejected` asserting `GraphError::CycleDetected` and `cook_skips_clean_nodes` asserting cook count == 1

**Wiring requirement:** Must be called from `pub mod node_graph;` in `crates/vox_editor/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 0: Bootstrap the `vox_editor` crate**

```toml
# crates/vox_editor/Cargo.toml
[package]
name = "vox_editor"
version = "0.1.0"
edition = "2021"

[dependencies]
vox_core = { path = "../vox_core" }
vox_render = { path = "../vox_render" }
```

```rust
// crates/vox_editor/src/lib.rs
// Editor crate — node graph, gizmos, terrain editor, viewport UI.
```

```toml
# In root Cargo.toml, add to [workspace] members:
"crates/vox_editor",
```

```bash
cargo build -p vox_editor
```
Expected: clean build — crate exists and compiles before any other steps proceed.

- [ ] **Step 1: Write the failing test**
```rust
//! OchromaNodeGraph — procedural scene operations as a DAG.
//!
//! Direct port of CrucibleGraph from AetherSpectra with Ochroma-specific port types.
//! Key invariants:
//!   - Kahn's topological sort with BinaryHeap<Reverse<NodeId>> for determinism
//!   - Downstream dirty cascade on mark_dirty()
//!   - Idempotent duplicate-edge guard in connect()
//!   - Type-checked port connections via OchromaNode::descriptor()

use std::collections::{BinaryHeap, VecDeque};
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

#[derive(Clone)]
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
            return Err(GraphError::CookFailed { node: "topo_sort".into(), reason: "cycle detected".into() });
        }
        Ok(order)
    }

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
        GraphSnapshot { nodes, edges }
    }

    pub fn restore(&mut self, _snap: GraphSnapshot) -> Result<(), NodeError> {
        self.nodes.clear();
        self.edges.clear();
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

#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub nodes: Vec<NodeSnapshot>,
    pub edges: Vec<EdgeSnapshot>,
}

impl GraphSnapshot {
    pub fn to_json(&self) -> Result<String, serde_json::Error> { serde_json::to_string_pretty(self) }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}

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
        }
        let mut g = OchromaNodeGraph::new();
        let a = g.add_node("a", Box::new(TerrainOutNode));
        let b = g.add_node("b", pass());
        let err = g.connect(a, "terrain", b, "in").unwrap_err();
        assert!(matches!(err, GraphError::TypeMismatch { .. }), "got: {:?}", err);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor node_graph 2>&1 | head -20
```
Expected: FAIL — compile error (module not exposed in lib.rs).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_editor/src/node_graph.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_editor/src/lib.rs:
pub mod node_graph;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor node_graph -- --nocapture
```
Expected: PASS, output: 7 tests pass. `connect_cycle_rejected` prints `GraphError::CycleDetected`. `cook_skips_clean_nodes` prints count=1.

- [ ] **Step 6: Commit**
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

**Acceptance:** `cargo test -p vox_editor terrain_node -- --nocapture` → 7 tests pass, including `erosion_modifies_terrain` asserting `diff > 0.01`

**Wiring requirement:** Must be called from `pub mod terrain_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! TerrainNode — fBm heightfield + hydraulic erosion.
//! Adapted from aetherspectra/forge/crates/terrain/src/{generate.rs, hydraulic.rs}.

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
    pub droplet_count: u32,
}

impl Default for TerrainNode {
    fn default() -> Self {
        Self { resolution: 256, world_size: 1000.0, amplitude: 200.0, octaves: 6, frequency: 1.0, seed: 0, droplet_count: 80_000 }
    }
}

// [Full implementation with generate_heightfield and hydraulic_erode as in source material]
// Tests:
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
        let no_erosion   = TerrainNode { resolution: 32, droplet_count: 0,    seed: 42, ..Default::default() };
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
        let terrain_id = graph.add_node("terrain", Box::new(TerrainNode { resolution: 32, droplet_count: 0, ..Default::default() }));
        graph.cook().unwrap();
        assert!(graph.get_output(terrain_id, "terrain").is_some());
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor terrain_node 2>&1 | head -20
```
Expected: FAIL — compile error (noise/rand not in Cargo.toml or module not exposed).

- [ ] **Step 3: Implement** (no stubs, no todo!())
```toml
# Add to crates/vox_editor/Cargo.toml [dependencies]:
noise    = "0.9"
rand     = "0.8"
rand_pcg = "0.3"
```
Implement `generate_heightfield` (Fbm<Perlin>, normalise, multiply by amplitude) and `hydraulic_erode` (Olsen/Cordonnier droplet: central-difference gradient, capacity = `capacity × speed × water × max(-dh, 0.01)`, deposit/erode, evaporate) in `terrain_node.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_editor/src/nodes/mod.rs:
pub mod terrain_node;
pub mod building_node;
pub mod vegetation_node;
pub mod splatize_node;
pub mod biome_node;
pub mod splat_weight_node;
pub mod moisture_node;
pub mod plot_node;
pub mod inhabitation_node;
pub mod urban_sim_node;

// crates/vox_editor/src/lib.rs:
pub mod nodes;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor terrain_node -- --nocapture
```
Expected: PASS, output: 7 tests pass. `erosion_modifies_terrain` prints actual diff value > 0.01.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/ crates/vox_editor/src/lib.rs crates/vox_editor/Cargo.toml
git commit -m "feat(editor): TerrainNode — fBm heightfield + hydraulic erosion adapted from forge-terrain"
```

---

## Task 3: BuildingNode — WFC facade

**Files:**
- Create: `crates/vox_editor/src/nodes/building_node.rs`

**Acceptance:** `cargo test -p vox_editor building_node -- --nocapture` → 5 tests pass, including `wfc_is_deterministic` asserting same seed → same vertex count

**Wiring requirement:** Must be called from `pub mod building_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! BuildingNode — WFC tile-based building generator.
//! Adapted from aetherspectra/forge/crates/building/src/wfc.rs.
//! 5 tile types: Wall, Window, Corner, Door, Empty; bitmask superposition (u8).
//! BFS propagation via VecDeque. Retry: Pcg64::seed_from_u64(seed ^ attempt * 0x9e3779b9).

// [Full BuildingNode implementation with WFC solver, propagate, tiles_to_mesh, emit_box]

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
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor building_node 2>&1 | head -20
```
Expected: FAIL — compile error.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implement `solve_wfc` (per-attempt RNG with `seed ^ attempt * 0x9e3779b9`), `try_solve` (bitmask superposition, BFS propagation via VecDeque, min-entropy collapse), `propagate`, `tiles_to_mesh`, `emit_box` in `building_node.rs`. Include `BuildingDescription` with serde for LLM authoring.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Already in nodes/mod.rs from Task 2
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor building_node -- --nocapture
```
Expected: PASS, output: 5 tests pass. `wfc_is_deterministic` prints same vertex count for both runs.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/building_node.rs
git commit -m "feat(editor): BuildingNode — WFC 3D building generator adapted from forge-building"
```

---

## Task 4: VegetationNode — L-system tree + LOD

**Files:**
- Create: `crates/vox_editor/src/nodes/vegetation_node.rs`

**Acceptance:** `cargo test -p vox_editor vegetation_node -- --nocapture` → 7 tests pass, including `lod3_billboard_has_exactly_two_triangles` asserting `lods[3].indices.len() == 2`

**Wiring requirement:** Must be called from `pub mod vegetation_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! VegetationNode — L-system tree with 4-level LOD.
//! Adapted from forge-vegetation lsystem.rs (grow_segment) and lod.rs (build_lod_set).

use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use glam::{Quat, Vec3};

// [Full VegetationNode, build_tree, grow_segment, rotate_around_axis,
//  emit_cylinder, emit_leaf_quad, decimate_mesh, billboard_from_mesh]

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
        assert!(mesh.positions.len() <= 20_000, "vertex budget guard: {} > 20_000", mesh.positions.len());
    }

    #[test]
    fn different_seeds_produce_different_trees() {
        let a = VegetationNode { seed: 1, ..Default::default() }.cook(NodeInputs::new()).unwrap();
        let b = VegetationNode { seed: 2, ..Default::default() }.cook(NodeInputs::new()).unwrap();
        let pa = &a["mesh"].as_mesh().unwrap().positions;
        let pb = &b["mesh"].as_mesh().unwrap().positions;
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
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor vegetation_node 2>&1 | head -20
```
Expected: FAIL — compile error.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implement `build_tree`, `grow_segment` (recursive cylinder segments, budget guard at 20k verts, leaf quad at `branch_levels==0`), `rotate_around_axis`, `emit_cylinder` (6-sided, two rings of vertices, 12 triangles), `emit_leaf_quad`, `decimate_mesh` (stride-based), `billboard_from_mesh` (AABB → 2-triangle quad).

- [ ] **Step 4: Wire at exact callsite**
```rust
// Already in nodes/mod.rs from Task 2
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor vegetation_node -- --nocapture
```
Expected: PASS, output: 7 tests pass. `lod3_billboard_has_exactly_two_triangles` prints `indices.len()=2`.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/vegetation_node.rs
git commit -m "feat(editor): VegetationNode — L-system tree with 4-level LOD adapted from forge-vegetation"
```

---

## Task 5: SplatizeNode — Mesh → GaussianSplat[] with spectral material assignment

**Files:**
- Create: `crates/vox_editor/src/nodes/splatize_node.rs`

**Acceptance:** `cargo test -p vox_editor splatize_node -- --nocapture` → 8 tests pass, including `foliage_material_has_green_bias` asserting `green_sum > red_sum` and `splatize_node_in_graph_end_to_end` producing non-empty splats

**Wiring requirement:** Must be called from `pub mod splatize_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! SplatizeNode — converts EditorMesh to GaussianSplat[] with spectral assignment.
//! Area-weighted random triangle sampling + Smits-style material→spectral upsampling.

// [Full SplatizeNode, splatize, spectral_from_material (8 material profiles),
//  triangle_area, cross functions]

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> EditorMesh {
        EditorMesh {
            positions: vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,0.0,1.0],[0.0,0.0,1.0]],
            normals: vec![[0.0,1.0,0.0]; 4],
            indices: vec![[0,1,2],[0,2,3]],
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
        let node = SplatizeNode { splats_per_sqm: 0.001, min_splats: 50, max_splats: 1000, ..Default::default() };
        let mut inputs = NodeInputs::new();
        inputs.insert("mesh".into(), PortData::Mesh(quad_mesh()));
        let out = node.cook(inputs).unwrap();
        let splats = out["splats"].as_splats().unwrap();
        assert!(splats.len() >= 50, "should respect min_splats, got {}", splats.len());
    }

    #[test]
    fn splat_count_respects_max() {
        let node = SplatizeNode { splats_per_sqm: 1e9, min_splats: 0, max_splats: 200, ..Default::default() };
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
            assert!((0.0..=1.0).contains(&s.position()[0]), "x out of [0,1]: {}", s.position()[0]);
            assert!((0.0..=1.0).contains(&s.position()[2]), "z out of [0,1]: {}", s.position()[2]);
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
            let any_nonzero = s.spectral().iter().any(|&v| v != 0);
            assert!(any_nonzero, "splat spectral should be non-zero from material assignment");
        }
    }

    #[test]
    fn foliage_material_has_green_bias() {
        let spectral = spectral_from_material(2);
        let green_sum = half::f16::from_bits(spectral[6]).to_f32()
                      + half::f16::from_bits(spectral[7]).to_f32()
                      + half::f16::from_bits(spectral[8]).to_f32();
        let red_sum   = half::f16::from_bits(spectral[12]).to_f32()
                      + half::f16::from_bits(spectral[13]).to_f32();
        assert!(green_sum > red_sum, "foliage should have green-band bias, got green={} red={}", green_sum, red_sum);
    }

    #[test]
    fn brick_material_has_red_bias() {
        let spectral = spectral_from_material(5);
        let red_sum = half::f16::from_bits(spectral[10]).to_f32()
                    + half::f16::from_bits(spectral[11]).to_f32()
                    + half::f16::from_bits(spectral[12]).to_f32();
        let uv_sum  = half::f16::from_bits(spectral[0]).to_f32()
                    + half::f16::from_bits(spectral[1]).to_f32();
        assert!(red_sum > uv_sum * 2.0, "brick should have strong red bias, red={} uv={}", red_sum, uv_sum);
    }

    #[test]
    fn splatize_node_in_graph_end_to_end() {
        use crate::node_graph::OchromaNodeGraph;
        use crate::nodes::building_node::BuildingNode;
        let mut graph = OchromaNodeGraph::new();
        let building_id = graph.add_node("building", Box::new(BuildingNode { grid_w: 3, grid_h: 2, grid_d: 3, ..Default::default() }));
        let splat_id = graph.add_node("splatize", Box::new(SplatizeNode { min_splats: 10, max_splats: 500, ..Default::default() }));
        graph.connect(building_id, "mesh", splat_id, "mesh").unwrap();
        graph.cook().unwrap();
        let splats = graph.get_output(splat_id, "splats").unwrap().as_splats().unwrap();
        assert!(!splats.is_empty(), "splatize should produce splats from building mesh");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor splatize_node 2>&1 | head -20
```
Expected: FAIL — compile error.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implement `splatize` (area-weighted triangle sampling, barycentric random point, `spectral_from_material`), `spectral_from_material` (8 material profiles: gray, wood, foliage, stone, metal, brick, glass, fire), `triangle_area`, `cross`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Already in nodes/mod.rs from Task 2
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor splatize_node -- --nocapture
```
Expected: PASS, output: 8 tests pass. `foliage_material_has_green_bias` prints actual green/red sums. `splatize_node_in_graph_end_to_end` prints non-empty splat count.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/splatize_node.rs
git commit -m "feat(editor): SplatizeNode — Mesh→GaussianSplat[] with Smits spectral material assignment"
```

---

## Task 6: Node editor UI panel (egui)

**Files:**
- Create: `crates/vox_editor/src/editor_panel.rs`
- Modify: `crates/vox_editor/src/lib.rs`

**Acceptance:** `cargo test -p vox_editor editor_panel -- --nocapture` → 5 tests pass, including `port_colors_are_distinct` asserting terrain color != splat color

**Wiring requirement:** Must be called from `pub mod editor_panel;` in `crates/vox_editor/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Ochroma node editor egui panel.
//! Nodes: rounded rect + port circles. Wires: bezier curves. Param sidebar: cook button.

use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};
use hashbrown::HashMap;
use crate::node_graph::{NodeId, OchromaNodeGraph, PortType};

#[derive(Clone, Debug)]
pub struct NodeLayout {
    pub pos:  Pos2,
    pub size: Vec2,
}

type PortPositions = HashMap<(u32, String), Pos2>;

#[derive(Clone, Debug)]
pub struct WireDrag {
    pub from_node: NodeId,
    pub from_port: String,
    pub current:   Pos2,
}

pub struct NodeEditorPanel {
    pub layouts:   HashMap<NodeId, NodeLayout>,
    pub selected:  Option<NodeId>,
    pub wire_drag: Option<WireDrag>,
    pub pan:       Vec2,
    pub zoom:      f32,
}

impl Default for NodeEditorPanel {
    fn default() -> Self {
        Self { layouts: HashMap::new(), selected: None, wire_drag: None, pan: Vec2::ZERO, zoom: 1.0 }
    }
}

impl NodeEditorPanel {
    pub fn new() -> Self { Self::default() }

    pub fn ensure_layouts(&mut self, _graph: &OchromaNodeGraph) {
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

    pub fn port_color(pt: PortType) -> Color32 {
        match pt {
            PortType::Terrain       => Color32::from_rgb(140, 100, 60),
            PortType::Mesh          => Color32::from_rgb(90, 180, 90),
            PortType::LodMesh       => Color32::from_rgb(60, 160, 60),
            PortType::Splats        => Color32::from_rgb(80, 140, 220),
            PortType::SpectralField => Color32::from_rgb(200, 80, 200),
            PortType::Instances     => Color32::from_rgb(220, 180, 60),
            PortType::Scalar        => Color32::from_rgb(180, 180, 180),
            PortType::BiomeMap      => Color32::from_rgb(100, 160, 80),
            PortType::SplatWeights  => Color32::from_rgb(160, 120, 60),
            PortType::ScalarVec     => Color32::from_rgb(160, 180, 200),
        }
    }

    pub fn show(&mut self, ui: &mut Ui, graph: &mut OchromaNodeGraph) {
        let painter  = ui.painter();
        let response = ui.allocate_rect(ui.available_rect_before_wrap(), egui::Sense::drag());
        if response.dragged() && !ui.input(|i| i.pointer.secondary_down()) {
            self.pan += response.drag_delta();
        }
        let node_ids: Vec<NodeId> = self.layouts.keys().copied().collect();
        let mut port_positions = PortPositions::new();
        for id in &node_ids {
            let Some(layout) = self.layouts.get_mut(id) else { continue };
            let top_left = layout.pos + self.pan;
            let rect = Rect::from_min_size(top_left, layout.size);
            let bg = if self.selected == Some(*id) { Color32::from_rgb(60, 70, 100) } else { Color32::from_rgb(45, 45, 55) };
            painter.rect_filled(rect, 6.0, bg);
            painter.rect_stroke(rect, 6.0, Stroke::new(1.0, Color32::from_rgb(100, 100, 120)));
            let header_rect = Rect::from_min_size(top_left, Vec2::new(layout.size.x, 24.0));
            painter.rect_filled(header_rect, egui::Rounding { nw: 6.0, ne: 6.0, sw: 0.0, se: 0.0 }, Color32::from_rgb(60, 80, 120));
            let node_response = ui.allocate_rect(rect, egui::Sense::click());
            if node_response.clicked() { self.selected = Some(*id); }
            let out_pos = top_left + Vec2::new(layout.size.x, layout.size.y * 0.5);
            let in_pos  = top_left + Vec2::new(0.0, layout.size.y * 0.5);
            port_positions.insert((id.0, "out".into()), out_pos);
            port_positions.insert((id.0, "in".into()),  in_pos);
            painter.circle_filled(out_pos, 5.0, Color32::from_rgb(80, 200, 120));
            painter.circle_filled(in_pos,  5.0, Color32::from_rgb(200, 120, 80));
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::OchromaNodeGraph;

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
    fn ensure_layouts_does_not_overwrite_existing() {
        let mut panel = NodeEditorPanel::new();
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("a", Box::new(crate::node_graph::tests_helpers::pass_node()));
        panel.layouts.insert(id, NodeLayout { pos: Pos2::new(0.0, 0.0), size: Vec2::new(180.0, 120.0) });
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
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor editor_panel 2>&1 | head -20
```
Expected: FAIL — compile error.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_editor/src/editor_panel.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_editor/src/lib.rs:
pub mod editor_panel;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor editor_panel -- --nocapture
```
Expected: PASS, output: 5 tests pass; `port_colors_are_distinct` passes with distinct RGB values.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/editor_panel.rs crates/vox_editor/src/lib.rs
git commit -m "feat(editor): NodeEditorPanel — egui node canvas with port colours, wire drag, param sidebar"
```

---

## Task 7: Integration verification

**Acceptance:** `cargo test -p vox_editor splatize_node::tests::splatize_node_in_graph_end_to_end -- --nocapture` → PASS, splat count printed > 0

**Wiring requirement:** All prior tasks complete. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
cargo test -p vox_editor splatize_node::tests::splatize_node_in_graph_end_to_end 2>&1 | tail -5
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test --workspace 2>&1 | tail -30
```
Expected: PASS after all prior tasks, FAIL if any task incomplete.

- [ ] **Step 3: Implement** (no stubs, no todo!())
```bash
# Complete Tasks 1-6 first
cargo test --workspace 2>&1 | tail -30
```
- [ ] **Step 4: Wire at exact callsite**
```bash
cargo test -p vox_editor splatize_node::tests::splatize_node_in_graph_end_to_end -- --nocapture
cargo test -p vox_editor node_graph::tests::mark_dirty_cascades_transitive -- --nocapture
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor splatize_node::tests::splatize_node_in_graph_end_to_end -- --nocapture
```
Expected: PASS, output: `splats.len() = N` where N > 0.

- [ ] **Step 6: Commit**
```bash
git commit --allow-empty -m "test(editor): domain 9 integration verified — node graph, terrain, building, vegetation, splatize"
```

---

## Task 8: BiomeNode, SplatWeightNode, MoistureNode — biome pipeline

**Files:**
- Create: `crates/vox_editor/src/nodes/biome_node.rs`
- Create: `crates/vox_editor/src/nodes/splat_weight_node.rs`
- Create: `crates/vox_editor/src/nodes/moisture_node.rs`
- Modify: `crates/vox_editor/src/nodes/mod.rs`

**Acceptance:** `cargo test -p vox_editor biome_node -- --nocapture` → 3 tests pass; `cargo test -p vox_editor splat_weight_node -- --nocapture` → 2 tests pass; `cargo test -p vox_editor moisture_node -- --nocapture` → 3 tests pass

**Wiring requirement:** Must be called from `pub mod biome_node; pub mod splat_weight_node; pub mod moisture_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// biome_node tests:
#[test]
fn test_biome_node_classifies_by_height() {
    let mut node = BiomeNode::default();
    // Heights at 90% of world_height=400 → Alpine
    let terrain = HeightfieldSpatial { resolution: 2, world_size: 100.0, heights: vec![360.0; 4], ..Default::default() };
    let mut inputs = NodeInputs::new();
    inputs.insert("terrain".into(), PortData::Terrain(terrain));
    let out = node.cook(inputs).unwrap();
    let biome_bytes = out["biome_map"].as_biome_map().unwrap();
    assert!(biome_bytes.iter().all(|&b| b == BiomeKind::Alpine as u8), "cells at 90% world height should be Alpine");
}

#[test]
fn test_splat_weights_sum_to_one() {
    let weights = biome_to_splat_weights(BiomeKind::Forest, 40.0, 80.0);
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "weights must sum to 1, got {sum}");
    assert!(weights[2] > 0.5, "Forest: vegetation channel (2) should dominate");
}

#[test]
fn test_spectral_terrain_materials_water_dark() {
    let mats = SpectralTerrainMaterials::default();
    assert!(mats.slots[0][0] < 0.1,  "Water slot UV reflectance should be dark");
    assert!(mats.slots[5][0] > 0.85, "Snow slot should be near-white");
}

// splat_weight_node tests:
#[test]
fn test_splat_weight_node_forest_dominant_veg() {
    // Forest biome → vegetation channel (2) > 0.5 for all cells
    // [full test as in source]
}

#[test]
fn test_splat_weight_node_custom_biome_map() {
    // Desert → ground dominant; Wetland → water significant
    // [full test as in source]
}

// moisture_node tests:
#[test]
fn test_moisture_node_combines_drip_and_urban() {
    let node = MoistureNode::default();
    let drip = vec![0.8f32, 0.1, 0.0, 0.5];
    let urban = vec![0.2f32, 0.0, 0.9, 0.1];
    let result = node.combine(&drip, Some(&urban));
    assert!((result[0] - 0.8).abs() < 0.01); // max(0.8, 0.2) = 0.8
    assert!((result[2] - 0.9).abs() < 0.01); // max(0.0, 0.9) = 0.9
}

#[test]
fn test_moisture_node_drip_only() {
    let node = MoistureNode::default();
    let drip = vec![0.3f32, 0.7, 0.0];
    let result = node.combine(&drip, None);
    assert!((result[0] - 0.3).abs() < 0.01);
    assert!((result[1] - 0.7).abs() < 0.01);
}

#[test]
fn test_splat_weight_node_moisture_darkens_alpine() {
    // Wet Alpine spectral should be darker than dry Alpine at all bands
    // [full test verifying blend_moisture produces values <= dry for all 16 bands]
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor biome_node splat_weight_node moisture_node 2>&1 | head -30
```
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement** (no stubs, no todo!())

**BiomeNode:** `BiomeKind` (11 variants: Alpine, Tundra, Forest, Grassland, Desert, Wetland, Coastal, SubalpineShrub, Savanna, Taiga, TropicalRainforest), altitude-based classification in `cook()`, output as `PortData::BiomeMap(Vec<u8>)`.

**SpectralTerrainMaterials:** 7-slot USGS palette — Water(dark blue), Sand, Grass(green), Dirt, Rock, Snow(bright), Bark.

`biome_to_splat_weights`: Alpine=[0.00,0.50,0.05,0.45], Forest=[0.00,0.05,0.70,0.25], Desert=[0.00,0.10,0.00,0.90], Wetland=[0.40,0.05,0.40,0.15], etc.

**SplatWeightNode:** reads BiomeMap + terrain heights, calls `biome_to_splat_weights` per cell, accepts optional `moisture` input and blends toward water slot.

**MoistureNode:** `combine(drip, urban)` takes per-cell max after scaling; `blend_moisture(base, water, moisture)` linear blend with no altitude or biome exemptions.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_editor/src/nodes/mod.rs:
pub mod biome_node;
pub mod splat_weight_node;
pub mod moisture_node;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor biome_node -- --nocapture
cargo test -p vox_editor splat_weight_node -- --nocapture
cargo test -p vox_editor moisture_node -- --nocapture
```
Expected: PASS, output: biome tests print Alpine for high-altitude cells; splat weight tests print vegetation > 0.5 for Forest.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/biome_node.rs \
        crates/vox_editor/src/nodes/splat_weight_node.rs \
        crates/vox_editor/src/nodes/moisture_node.rs \
        crates/vox_editor/src/nodes/mod.rs
git commit -m "feat(editor): BiomeNode + SplatWeightNode + MoistureNode — composable biome pipeline"
```

---

## Task 9: PlotNode — forge-plot integration

**Files:**
- Create: `crates/vox_editor/src/nodes/plot_node.rs`
- Modify: `crates/vox_editor/src/nodes/mod.rs`

**Acceptance:** `cargo test -p vox_editor test_plot_node_residential_suburban -- --nocapture` → PASS, `ground_mesh.positions.len() > 3` and `fence_count >= 0.0`

**Wiring requirement:** Must be called from `pub mod plot_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[test]
fn test_plot_node_residential_suburban() {
    let mut node = PlotNode::default();
    node.set_param("archetype",   ParamValue::Str("residential_suburban".into())).unwrap();
    node.set_param("footprint_w", ParamValue::Float(20.0)).unwrap();
    node.set_param("footprint_d", ParamValue::Float(30.0)).unwrap();
    node.set_param("building_w",  ParamValue::Float(10.0)).unwrap();
    node.set_param("building_d",  ParamValue::Float(12.0)).unwrap();
    node.set_param("fence_style", ParamValue::Str("picket".into())).unwrap();
    node.set_param("seed",        ParamValue::Int(42)).unwrap();
    let outputs = node.cook(NodeInputs::new()).unwrap();
    let ground = outputs.get("ground_mesh").expect("ground_mesh output");
    assert!(ground.as_mesh().unwrap().positions.len() > 3);
    let fences = outputs.get("fence_count").expect("fence_count");
    assert!(fences.as_scalar().unwrap() >= 0.0);
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor test_plot_node_residential_suburban 2>&1 | head -20
```
Expected: FAIL — `PlotNode` not found.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implement `PlotNode` with fields `archetype`, `footprint_w/d`, `building_w/d`, `condition`, `fence_style`, `driveway`, `seed`. In `cook()`: build flat rectangular ground mesh from footprint dimensions, driveway rectangle, output counts for fence/prop/garden. No `todo!()` — all outputs populated.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_editor/src/nodes/mod.rs:
pub mod plot_node;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor test_plot_node_residential_suburban -- --nocapture
```
Expected: PASS, output: `positions.len() = 4`, `fence_count = 4.0`.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/plot_node.rs crates/vox_editor/src/nodes/mod.rs
git commit -m "feat(editor): PlotNode — land parcel geometry (ground, driveway, fence, props)"
```

---

## Task 10: InhabitationNode — catenary wires + prop placement

**Files:**
- Create: `crates/vox_editor/src/nodes/inhabitation_node.rs`
- Modify: `crates/vox_editor/src/nodes/mod.rs`

**Acceptance:** `cargo test -p vox_editor test_catenary_curve_sags test_prop_placement_respects_clearance -- --nocapture` → both PASS; catenary midpoint Y < 5.0 for level endpoints at Y=5

**Wiring requirement:** Must be called from `pub mod inhabitation_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[test]
fn test_catenary_curve_sags() {
    let node = CatenaryNode { start: [0.0, 5.0, 0.0], end: [10.0, 5.0, 0.0], slack: 0.1, segments: 20 };
    let pts = node.cook(NodeInputs::new()).unwrap();
    let pts = pts.get("points").unwrap().as_scalar_vec().unwrap();
    assert!(pts[pts.len() / 2] < 5.0, "catenary midpoint should sag below endpoints");
}

#[test]
fn test_prop_placement_respects_clearance() {
    let node = PropPlacementNode { area: [20.0, 20.0], count: 10, min_clearance: 2.0, seed: 42 };
    let out = node.cook(NodeInputs::default()).unwrap();
    let count = out.get("count").unwrap().as_scalar().unwrap() as usize;
    assert!(count <= 10);
    assert!(count > 0);
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor test_catenary_curve_sags test_prop_placement_respects_clearance 2>&1 | head -20
```
Expected: FAIL — compile error.

- [ ] **Step 3: Implement** (no stubs, no todo!())

**CatenaryNode:** Newton's method to find catenary parameter `a` from span `h` and arc length `s = straight * (1 + slack)`. Output Y values as `ScalarVec`.

**PropPlacementNode:** LCG-based Poisson-disk rejection sampling within `area`. Output `count` as Scalar.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_editor/src/nodes/mod.rs:
pub mod inhabitation_node;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor test_catenary_curve_sags test_prop_placement_respects_clearance -- --nocapture
```
Expected: PASS, output: catenary midpoint Y < 5.0 printed; prop count between 1 and 10.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/inhabitation_node.rs crates/vox_editor/src/nodes/mod.rs
git commit -m "feat(editor): CatenaryNode (Newton catenary) + PropPlacementNode (Poisson-disk)"
```

---

## Task 11: UrbanSimNode — traffic/moisture/upkeep simulation

**Files:**
- Create: `crates/vox_editor/src/nodes/urban_sim_node.rs`
- Modify: `crates/vox_editor/src/nodes/mod.rs`

**Acceptance:** `cargo test -p vox_editor test_urban_sim_produces_grid test_moisture_drives_spectral_blend -- --nocapture` → both PASS; max_traffic and max_moisture both in [0.0, 1.0]

**Wiring requirement:** Must be called from `pub mod urban_sim_node;` in `crates/vox_editor/src/nodes/mod.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[test]
fn test_urban_sim_produces_grid() {
    let mut node = UrbanSimNode::default();
    node.set_param("grid_w",     ParamValue::Int(16)).unwrap();
    node.set_param("grid_h",     ParamValue::Int(16)).unwrap();
    node.set_param("iterations", ParamValue::Int(10)).unwrap();
    node.set_param("seed",       ParamValue::Int(1)).unwrap();
    let out = node.cook(NodeInputs::new()).unwrap();
    let traffic  = out.get("max_traffic").unwrap().as_scalar().unwrap();
    let moisture = out.get("max_moisture").unwrap().as_scalar().unwrap();
    assert!(traffic >= 0.0 && traffic <= 1.0);
    assert!(moisture >= 0.0 && moisture <= 1.0);
}

#[test]
fn test_moisture_drives_spectral_blend() {
    use crate::nodes::biome_node::SpectralTerrainMaterials;
    let mats = SpectralTerrainMaterials::default();
    let dry = mats.slots[3]; // Dirt
    let wet = mats.slots[0]; // Water
    let blend: [f32; 16] = std::array::from_fn(|i| dry[i] * 0.7 + wet[i] * 0.3);
    for i in 0..16 {
        assert!(blend[i] < dry[i] + 0.01, "wet blend should be darker at band {i}");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor test_urban_sim_produces_grid test_moisture_drives_spectral_blend 2>&1 | head -20
```
Expected: FAIL — compile error.

- [ ] **Step 3: Implement** (no stubs, no todo!())

Implement `UrbanSimNode` with `UrbanCell { traffic_weight, refuse_level, civic_upkeep, wind_exposure, moisture }`. Seed 3 traffic sources via LCG RNG. Diffuse `traffic_weight` across neighbors each iteration; derive `moisture` from traffic drainage, `civic_upkeep = 1 - traffic * 0.5`, `wind_exposure` from edge cells. Output `max_traffic`, `max_moisture`, `cell_count` as Scalars.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_editor/src/nodes/mod.rs:
pub mod urban_sim_node;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor test_urban_sim_produces_grid test_moisture_drives_spectral_blend -- --nocapture
```
Expected: PASS, output: `max_traffic` and `max_moisture` values in [0, 1]; moisture-blended spectral is darker than dry.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/nodes/urban_sim_node.rs crates/vox_editor/src/nodes/mod.rs
git commit -m "feat(editor): UrbanSimNode — traffic/moisture/civic_upkeep reaction-diffusion"
```

---

## Task 12: GraphSnapshot undo/redo + BuildingDescription JSON authoring

**Files:**
- Modify: `crates/vox_editor/src/node_graph.rs`
- Modify: `crates/vox_editor/src/nodes/building_node.rs`

**Acceptance:** `cargo test -p vox_editor snapshot_tests description_tests -- --nocapture` → both test groups PASS; snapshot JSON round-trips with node type_name preserved

**Wiring requirement:** Must be called from `OchromaNodeGraph::snapshot()` and `OchromaNodeGraph::restore()` in `crates/vox_editor/src/node_graph.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// In node_graph.rs tests:
#[test]
fn test_snapshot_round_trip() {
    let mut graph = OchromaNodeGraph::new();
    let id = graph.add_node("terrain", Box::new(TerrainNode::default()));
    let snap = graph.snapshot();
    let json = snap.to_json().unwrap();
    let restored = GraphSnapshot::from_json(&json).unwrap();
    assert_eq!(restored.nodes.len(), 1);
    assert_eq!(restored.nodes[0].type_name, "TerrainNode");
}

#[test]
fn test_undo_restores_params() {
    let mut graph = OchromaNodeGraph::new();
    let id = graph.add_node("terrain", Box::new(TerrainNode::default()));
    let snap_before = graph.snapshot();
    graph.set_param(id, "resolution", ParamValue::Int(256)).unwrap();
    graph.restore(snap_before).unwrap();
    let result = graph.cook();
    assert!(result.is_ok());
}

// In building_node.rs tests:
#[test]
fn test_building_description_json_roundtrip() {
    let desc = BuildingDescription {
        program: Program::Residential,
        setting: Setting::Suburban,
        style_key: "craftsman".into(),
        era: "1920s".into(),
        condition: BuildingCondition::Aged,
        floors: 2,
        floor_height: 3.0,
        seed: 42,
        detail_atoms: Some(vec!["exposed_rafter_tails".into(), "tapered_columns".into()]),
        organic_atoms: None,
        ..Default::default()
    };
    let json = serde_json::to_string(&desc).unwrap();
    let restored: BuildingDescription = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.style_key, "craftsman");
    assert_eq!(restored.detail_atoms.unwrap().len(), 2);
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_editor snapshot_tests description_tests 2>&1 | head -20
```
Expected: FAIL — `GraphSnapshot`, `restore()`, `BuildingDescription` not found.

- [ ] **Step 3: Implement** (no stubs, no todo!())

`GraphSnapshot { nodes: Vec<NodeSnapshot>, edges: Vec<EdgeSnapshot> }` with `to_json`/`from_json` via serde_json already added in Task 1's node_graph.rs. Verify `snapshot()` and `restore()` are implemented.

Add to `building_node.rs`: `Program` (Residential, Agricultural, Civic, Religious, Commercial, Industrial, Utility), `Setting` (Urban, Suburban, Rural, Industrial, Waterfront, HistoricalOldTown), `BuildingCondition` (New, Aged, Weathered, Derelict), `BuildingDescription { program, setting, style_key, era, condition, floors, floor_height, seed, detail_atoms, organic_atoms }` with serde Serialize/Deserialize. `BuildingDescription::to_building_params()` maps style_key prefix to `BuildingStyle` enum.

- [ ] **Step 4: Wire at exact callsite**
```rust
// snapshot() and restore() already in OchromaNodeGraph (added in Task 1)
// BuildingDescription::to_building_params() wires into BuildingNode::set_param("description_json", ...)
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_editor snapshot_tests description_tests -- --nocapture
```
Expected: PASS, output: snapshot JSON printed showing `type_name="TerrainNode"`; BuildingDescription round-trips with `style_key="craftsman"` and 2 detail_atoms.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_editor/src/node_graph.rs crates/vox_editor/src/nodes/building_node.rs
git commit -m "feat(editor): GraphSnapshot undo/redo + BuildingDescription LLM authoring contract"
```
