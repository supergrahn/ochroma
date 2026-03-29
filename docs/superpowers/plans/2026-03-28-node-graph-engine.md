# Node Graph Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `crucible-core` (from `/home/tomespen/git/aetherspectra/crucible/`) as Ochroma's graph evaluation backend, and upgrade all three editor windows (material, VFX, animation) to use `vox_ui::NodeGraphWidget` with live evaluation.

**Architecture:** A new `vox_nodes` crate bridges `CrucibleGraph` (evaluation engine from crucible-core) and `NodeGraphWidget` (visual renderer in vox_ui) via an `OchrGraph` type that holds the graph and per-node visual positions. `NodeGraphWidget` gains a `show_egui()` method for interactive rendering. All three editor windows own an `OchrGraph` + `NodeGraphWidget`.

**Tech Stack:** `crucible-core` path dep (Kahn-sort DAG, dirty propagation, typed ports), `crucible-types` path dep (PortData variants), `vox_ui::NodeGraphWidget`, egui 0.31, Rust 2024 edition.

---

## Key File Paths (read before editing)

- Ochroma workspace: `/home/tomespen/git/ochroma/Cargo.toml`
- crucible-core: `/home/tomespen/git/aetherspectra/crucible/rust/crates/crucible-core/`
- crucible-types: `/home/tomespen/git/aetherspectra/crucible/rust/crates/crucible-types/`
- `vox_ui` NodeGraphWidget: `crates/vox_ui/src/node_graph_widget.rs`
- Material editor: `crates/vox_render/src/material_editor_ui.rs` + `crates/vox_render/src/material_editor.rs`
- VFX editor: `crates/vox_render/src/vfx_editor_ui.rs`
- Anim editor: `crates/vox_render/src/anim_editor_ui.rs`
- vox_render lib: `crates/vox_render/src/lib.rs`

## File Structure

**Create:**
- `crates/vox_nodes/Cargo.toml`
- `crates/vox_nodes/src/lib.rs` — `OchrGraph` bridge type
- `crates/vox_nodes/src/mat_nodes.rs` — material `CrucibleNode` impls

**Modify:**
- `Cargo.toml` — workspace members + path deps
- `crates/vox_ui/src/node_graph_widget.rs` — add `show_egui()` + `pin_screen_pos()` methods
- `crates/vox_render/Cargo.toml` — add `vox_nodes`, `vox_ui` deps
- `crates/vox_render/src/material_editor_ui.rs` — replace with OchrGraph + NodeGraphWidget
- `crates/vox_render/src/vfx_editor_ui.rs` — replace stub
- `crates/vox_render/src/anim_editor_ui.rs` — replace stub

---

### Task 1: Add crucible-core/crucible-types path deps + vox_nodes skeleton

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/vox_nodes/Cargo.toml`
- Create: `crates/vox_nodes/src/lib.rs`
- Create: `crates/vox_nodes/src/mat_nodes.rs`

- [ ] **Step 1: Add to workspace Cargo.toml**

Open `/home/tomespen/git/ochroma/Cargo.toml`. Add to `[workspace.members]`:

```toml
    "crates/vox_nodes",
```

Add to `[workspace.dependencies]`:

```toml
crucible-core  = { path = "../../aetherspectra/crucible/rust/crates/crucible-core" }
crucible-types = { path = "../../aetherspectra/crucible/rust/crates/crucible-types" }
vox_nodes      = { path = "crates/vox_nodes" }
```

- [ ] **Step 2: Create crates/vox_nodes/Cargo.toml**

```toml
[package]
name = "vox_nodes"
edition.workspace = true
version.workspace = true

[dependencies]
vox_ui         = { path = "../vox_ui" }
crucible-core  = { workspace = true }
crucible-types = { workspace = true }
serde          = { workspace = true }
thiserror      = { workspace = true }
```

- [ ] **Step 3: Create crates/vox_nodes/src/lib.rs**

```rust
pub mod mat_nodes;
```

- [ ] **Step 4: Create crates/vox_nodes/src/mat_nodes.rs**

```rust
// placeholder — implemented in Task 3
```

- [ ] **Step 5: Verify the crate compiles**

Run: `cargo check -p vox_nodes`
Expected: exits 0 with no errors

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/vox_nodes/
git commit -m "feat(vox_nodes): add crucible-core path deps + vox_nodes crate skeleton"
```

---

### Task 2: Implement OchrGraph bridge type

**Files:**
- Modify: `crates/vox_nodes/src/lib.rs`

`OchrGraph` wraps `CrucibleGraph` and maintains a per-node position map. It provides `to_visual_nodes()` and `to_visual_connections()` to sync state into `NodeGraphWidget`'s visual types.

- [ ] **Step 1: Write failing tests**

Replace `crates/vox_nodes/src/lib.rs` with:

```rust
pub mod mat_nodes;

use std::collections::HashMap;
pub use crucible_core::graph::{CrucibleGraph, NodeId};
pub use crucible_core::port::{PortData, PortMap, ParamValue, PortDataType};
pub use crucible_core::node::{CrucibleNode, NodeDescriptor, PortSpec};
pub use crucible_core::error::CookError;
use vox_ui::node_graph_widget::{VisualNode, VisualConnection};

pub struct OchrGraph {
    pub graph: CrucibleGraph,
    positions: HashMap<u32, [f32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat_nodes::FloatConstNode;

    #[test]
    fn ochrgraph_add_and_cook() {
        let mut og = OchrGraph::new();
        let id = og.add_node("f", Box::new(FloatConstNode::new(3.14)), [100.0, 50.0]);
        og.graph.cook().unwrap();
        let out = og.graph.get_output(id, "out").unwrap();
        assert!((out.as_scalar().unwrap() - 3.14).abs() < 1e-9);
    }

    #[test]
    fn ochrgraph_to_visual_nodes_count() {
        let mut og = OchrGraph::new();
        og.add_node("a", Box::new(FloatConstNode::new(1.0)), [0.0, 0.0]);
        og.add_node("b", Box::new(FloatConstNode::new(2.0)), [200.0, 0.0]);
        let vis = og.to_visual_nodes();
        assert_eq!(vis.len(), 2);
    }

    #[test]
    fn ochrgraph_to_visual_connections_count() {
        let mut og = OchrGraph::new();
        let a = og.add_node("a", Box::new(FloatConstNode::new(1.0)), [0.0, 0.0]);
        let b = og.add_node("b", Box::new(FloatConstNode::new(2.0)), [200.0, 0.0]);
        // FloatConst has output "out", and "in" is a dynamic port on FloatConst
        // Use a passthrough pattern — connect a→b using dynamic port
        let _ = og.connect(a, "out", b, "out"); // type mismatch expected — use declared output on b
        // Actually connect to a declared port: won't work with FloatConst
        // This test just verifies connections list is empty when no connections added
        assert_eq!(og.to_visual_connections().len(), 0);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vox_nodes`
Expected: FAIL — `OchrGraph::new`, `FloatConstNode` not found

- [ ] **Step 3: Implement OchrGraph**

Replace the file content (keep the tests at bottom) with the full implementation:

```rust
pub mod mat_nodes;

use std::collections::HashMap;
pub use crucible_core::graph::{CrucibleGraph, NodeId};
pub use crucible_core::port::{PortData, PortMap, ParamValue, PortDataType};
pub use crucible_core::node::{CrucibleNode, NodeDescriptor, PortSpec};
pub use crucible_core::error::CookError;
use vox_ui::node_graph_widget::{VisualNode, VisualConnection};

pub struct OchrGraph {
    pub graph: CrucibleGraph,
    positions: HashMap<u32, [f32; 2]>,
}

impl OchrGraph {
    pub fn new() -> Self {
        Self { graph: CrucibleGraph::new(), positions: HashMap::new() }
    }

    pub fn add_node(&mut self, name: &str, node: Box<dyn CrucibleNode>, pos: [f32; 2]) -> NodeId {
        let id = self.graph.add_node(name, node);
        self.positions.insert(id.0, pos);
        id
    }

    pub fn connect(
        &mut self,
        from: NodeId, from_port: &str,
        to: NodeId,   to_port:   &str,
    ) -> Result<(), CookError> {
        self.graph.connect(from, from_port, to, to_port)
    }

    pub fn set_position(&mut self, id: NodeId, pos: [f32; 2]) {
        self.positions.insert(id.0, pos);
    }

    /// Build VisualNode list for NodeGraphWidget from current graph snapshot.
    /// Callers add domain-specific pin info after receiving this list.
    pub fn to_visual_nodes(&self) -> Vec<VisualNode> {
        let snap = self.graph.snapshot();
        snap.nodes.iter().map(|ns| {
            let pos = self.positions.get(&ns.id).copied().unwrap_or([0.0, 0.0]);
            VisualNode {
                id: ns.id,
                title: ns.name.clone(),
                position: pos,
                size: [140.0, 60.0],
                color: [55, 65, 95],
                inputs: vec![],
                outputs: vec![],
                selected: false,
                collapsed: false,
            }
        }).collect()
    }

    /// Build VisualConnection list for NodeGraphWidget.
    pub fn to_visual_connections(&self) -> Vec<VisualConnection> {
        let snap = self.graph.snapshot();
        snap.edges.iter().map(|es| VisualConnection {
            from_node: es.from,
            from_pin:  es.from_port.clone(),
            to_node:   es.to,
            to_pin:    es.to_port.clone(),
            color: [80, 140, 200],
        }).collect()
    }
}

impl Default for OchrGraph {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat_nodes::FloatConstNode;

    #[test]
    fn ochrgraph_add_and_cook() {
        let mut og = OchrGraph::new();
        let id = og.add_node("f", Box::new(FloatConstNode::new(3.14)), [100.0, 50.0]);
        og.graph.cook().unwrap();
        let out = og.graph.get_output(id, "out").unwrap();
        assert!((out.as_scalar().unwrap() - 3.14).abs() < 1e-9);
    }

    #[test]
    fn ochrgraph_to_visual_nodes_count() {
        let mut og = OchrGraph::new();
        og.add_node("a", Box::new(FloatConstNode::new(1.0)), [0.0, 0.0]);
        og.add_node("b", Box::new(FloatConstNode::new(2.0)), [200.0, 0.0]);
        let vis = og.to_visual_nodes();
        assert_eq!(vis.len(), 2);
    }

    #[test]
    fn ochrgraph_to_visual_connections_empty_when_none() {
        let og = OchrGraph::new();
        assert_eq!(og.to_visual_connections().len(), 0);
    }

    #[test]
    fn ochrgraph_set_position_updates_visual() {
        let mut og = OchrGraph::new();
        let id = og.add_node("n", Box::new(FloatConstNode::new(1.0)), [0.0, 0.0]);
        og.set_position(id, [200.0, 300.0]);
        let vis = og.to_visual_nodes();
        let node = vis.iter().find(|n| n.id == id.0).unwrap();
        assert_eq!(node.position, [200.0, 300.0]);
    }
}
```

- [ ] **Step 4: Run tests — expect FAIL on FloatConstNode**

Run: `cargo test -p vox_nodes`
Expected: FAIL — `FloatConstNode` not defined in mat_nodes

- [ ] **Step 5: Commit**

```bash
git add crates/vox_nodes/src/lib.rs
git commit -m "feat(vox_nodes): OchrGraph bridge — wraps CrucibleGraph with visual position map"
```

---

### Task 3: Material CrucibleNode implementations

**Files:**
- Modify: `crates/vox_nodes/src/mat_nodes.rs`

Implement `CrucibleNode` for the nodes used by the material editor. All float data flows as `PortData::Scalar(f64)`. `CrucibleNode` requires `Send + Sync` — all these types are trivially so.

- [ ] **Step 1: Write failing tests**

Write `crates/vox_nodes/src/mat_nodes.rs`:

```rust
use crucible_core::node::{CrucibleNode, NodeDescriptor, PortSpec};
use crucible_core::port::{PortData, PortDataType, PortMap, ParamValue};
use crucible_core::error::CookError;

// Implementations in Step 3 — leave empty for now so tests fail.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_const_outputs_value() {
        let n = FloatConstNode::new(2.5);
        let out = n.cook(PortMap::default()).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn multiply_node_multiplies() {
        let n = MultiplyNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(3.0));
        inputs.insert("b".into(), PortData::Scalar(4.0));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn add_node_adds() {
        let n = AddNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(1.0));
        inputs.insert("b".into(), PortData::Scalar(2.0));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn lerp_node_lerps() {
        let n = LerpNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(0.0));
        inputs.insert("b".into(), PortData::Scalar(10.0));
        inputs.insert("t".into(), PortData::Scalar(0.5));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn one_minus_node() {
        let n = OneMinusNode;
        let mut inputs = PortMap::default();
        inputs.insert("in".into(), PortData::Scalar(0.3));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn material_output_cooks_to_null() {
        let n = MaterialOutputNode;
        let out = n.cook(PortMap::default()).unwrap();
        assert!(matches!(out["material"], PortData::Null));
    }

    #[test]
    fn float_const_set_param_value() {
        let mut n = FloatConstNode::new(1.0);
        n.set_param("value", ParamValue::Float(7.0)).unwrap();
        let out = n.cook(PortMap::default()).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 7.0).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vox_nodes mat_nodes`
Expected: FAIL — `FloatConstNode`, `MultiplyNode`, etc. not found

- [ ] **Step 3: Implement all material nodes**

Replace `crates/vox_nodes/src/mat_nodes.rs` with:

```rust
use crucible_core::node::{CrucibleNode, NodeDescriptor, PortSpec};
use crucible_core::port::{PortData, PortDataType, PortMap, ParamValue};
use crucible_core::error::CookError;

// ---------------------------------------------------------------------------
// FloatConstNode
// ---------------------------------------------------------------------------

pub struct FloatConstNode {
    pub value: f64,
}

impl FloatConstNode {
    pub fn new(value: f64) -> Self { Self { value } }
}

impl CrucibleNode for FloatConstNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "FloatConst",
            inputs: vec![],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, value: ParamValue) -> Result<(), CookError> {
        match key {
            "value" => {
                self.value = value.as_float_coerce().ok_or_else(|| CookError::CookFailed {
                    node: "FloatConst".into(),
                    reason: format!("'value' must be a number, got {:?}", value),
                })?;
                Ok(())
            }
            _ => Err(CookError::UnknownParam { key: key.into(), node: "FloatConst".into() }),
        }
    }
    fn cook(&self, _inputs: PortMap) -> Result<PortMap, CookError> {
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(self.value));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MultiplyNode
// ---------------------------------------------------------------------------

pub struct MultiplyNode;

impl CrucibleNode for MultiplyNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "Multiply",
            inputs: vec![
                PortSpec { name: "a", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "b", data_type: PortDataType::Scalar, optional: false },
            ],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "Multiply".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let a = inputs.get("a").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("a".into()))?;
        let b = inputs.get("b").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("b".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(a * b));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// AddNode
// ---------------------------------------------------------------------------

pub struct AddNode;

impl CrucibleNode for AddNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "Add",
            inputs: vec![
                PortSpec { name: "a", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "b", data_type: PortDataType::Scalar, optional: false },
            ],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "Add".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let a = inputs.get("a").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("a".into()))?;
        let b = inputs.get("b").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("b".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(a + b));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// LerpNode
// ---------------------------------------------------------------------------

pub struct LerpNode;

impl CrucibleNode for LerpNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "Lerp",
            inputs: vec![
                PortSpec { name: "a", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "b", data_type: PortDataType::Scalar, optional: false },
                PortSpec { name: "t", data_type: PortDataType::Scalar, optional: false },
            ],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "Lerp".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let a = inputs.get("a").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("a".into()))?;
        let b = inputs.get("b").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("b".into()))?;
        let t = inputs.get("t").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("t".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(a + (b - a) * t));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// OneMinusNode
// ---------------------------------------------------------------------------

pub struct OneMinusNode;

impl CrucibleNode for OneMinusNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "OneMinus",
            inputs: vec![PortSpec { name: "in", data_type: PortDataType::Scalar, optional: false }],
            outputs: vec![PortSpec { name: "out", data_type: PortDataType::Scalar, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "OneMinus".into() })
    }
    fn cook(&self, inputs: PortMap) -> Result<PortMap, CookError> {
        let v = inputs.get("in").and_then(|p| p.as_scalar())
            .ok_or_else(|| CookError::MissingInput("in".into()))?;
        let mut out = PortMap::default();
        out.insert("out".into(), PortData::Scalar(1.0 - v));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MaterialOutputNode
// Collects roughness, metallic, base_r/g/b from upstream scalars.
// Outputs Null — callers read upstream values directly.
// ---------------------------------------------------------------------------

pub struct MaterialOutputNode;

impl CrucibleNode for MaterialOutputNode {
    fn descriptor(&self) -> NodeDescriptor {
        NodeDescriptor {
            type_name: "MaterialOutput",
            inputs: vec![
                PortSpec { name: "base_r",    data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "base_g",    data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "base_b",    data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "roughness", data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "metallic",  data_type: PortDataType::Scalar, optional: true },
                PortSpec { name: "emission",  data_type: PortDataType::Scalar, optional: true },
            ],
            outputs: vec![PortSpec { name: "material", data_type: PortDataType::Null, optional: false }],
        }
    }
    fn set_param(&mut self, key: &str, _: ParamValue) -> Result<(), CookError> {
        Err(CookError::UnknownParam { key: key.into(), node: "MaterialOutput".into() })
    }
    fn cook(&self, _inputs: PortMap) -> Result<PortMap, CookError> {
        let mut out = PortMap::default();
        out.insert("material".into(), PortData::Null);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_const_outputs_value() {
        let n = FloatConstNode::new(2.5);
        let out = n.cook(PortMap::default()).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn multiply_node_multiplies() {
        let n = MultiplyNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(3.0));
        inputs.insert("b".into(), PortData::Scalar(4.0));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn add_node_adds() {
        let n = AddNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(1.0));
        inputs.insert("b".into(), PortData::Scalar(2.0));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn lerp_node_lerps() {
        let n = LerpNode;
        let mut inputs = PortMap::default();
        inputs.insert("a".into(), PortData::Scalar(0.0));
        inputs.insert("b".into(), PortData::Scalar(10.0));
        inputs.insert("t".into(), PortData::Scalar(0.5));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn one_minus_node() {
        let n = OneMinusNode;
        let mut inputs = PortMap::default();
        inputs.insert("in".into(), PortData::Scalar(0.3));
        let out = n.cook(inputs).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn material_output_cooks_to_null() {
        let n = MaterialOutputNode;
        let out = n.cook(PortMap::default()).unwrap();
        assert!(matches!(out["material"], PortData::Null));
    }

    #[test]
    fn float_const_set_param_value() {
        let mut n = FloatConstNode::new(1.0);
        n.set_param("value", ParamValue::Float(7.0)).unwrap();
        let out = n.cook(PortMap::default()).unwrap();
        assert!((out["out"].as_scalar().unwrap() - 7.0).abs() < 1e-9);
    }
}
```

- [ ] **Step 4: Run tests — all 7 should pass**

Run: `cargo test -p vox_nodes`
Expected: 7 tests PASS (4 from lib.rs + 7 from mat_nodes minus duplicates)

- [ ] **Step 5: Commit**

```bash
git add crates/vox_nodes/src/mat_nodes.rs crates/vox_nodes/src/lib.rs
git commit -m "feat(vox_nodes): FloatConst, Multiply, Add, Lerp, OneMinus, MaterialOutput CrucibleNode impls"
```

---

### Task 4: NodeGraphWidget gains show_egui() method

**Files:**
- Modify: `crates/vox_ui/src/node_graph_widget.rs`

Add an interactive egui rendering method and a private `pin_screen_pos` helper. The existing `render_to_pixels` method is for offline/software rendering — keep it. `show_egui` is for live editor use.

- [ ] **Step 1: Write failing compilation test**

Append to the test section in `node_graph_widget.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn show_egui_method_exists() {
        // Compilation test — confirms show_egui signature is correct.
        fn _check(_: &mut NodeGraphWidget, _: &mut egui::Ui) -> Vec<NodeGraphAction> {
            unreachable!()
        }
        // If NodeGraphWidget::show_egui has the right signature, this compiles:
        let _: fn(&mut NodeGraphWidget, &mut egui::Ui) -> Vec<NodeGraphAction> =
            |w, ui| w.show_egui(ui);
    }

    #[test]
    fn new_widget_is_empty() {
        let w = NodeGraphWidget::new();
        assert_eq!(w.node_count(), 0);
        assert_eq!(w.connection_count(), 0);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vox_ui show_egui_method_exists`
Expected: FAIL — `show_egui` not found

- [ ] **Step 3: Add show_egui() and pin_screen_pos() to NodeGraphWidget**

In `node_graph_widget.rs`, add after the `render_to_pixels` method (before the closing `}` of `impl NodeGraphWidget`):

```rust
    /// Render the node graph interactively using egui.
    /// Returns actions produced by user interaction this frame.
    /// Call this inside an egui window or panel.
    pub fn show_egui(&mut self, ui: &mut egui::Ui) -> Vec<NodeGraphAction> {
        let mut actions = Vec::new();
        let avail = ui.available_rect_before_wrap();
        let painter = ui.painter_at(avail);

        // Allocate the full available area for input
        let response = ui.allocate_rect(avail, egui::Sense::click_and_drag());

        // Background
        painter.rect_filled(avail, 0.0, egui::Color32::from_rgb(28, 28, 35));

        // Grid lines
        let grid_px = self.grid_size * self.zoom;
        let ox = self.scroll_offset[0] % grid_px;
        let oy = self.scroll_offset[1] % grid_px;
        let grid_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(40, 42, 52));
        let mut gx = avail.min.x + ox;
        while gx < avail.max.x {
            painter.line_segment([egui::pos2(gx, avail.min.y), egui::pos2(gx, avail.max.y)], grid_stroke);
            gx += grid_px;
        }
        let mut gy = avail.min.y + oy;
        while gy < avail.max.y {
            painter.line_segment([egui::pos2(avail.min.x, gy), egui::pos2(avail.max.x, gy)], grid_stroke);
            gy += grid_px;
        }

        // Pan (middle mouse drag)
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            self.scroll_offset[0] += delta.x;
            self.scroll_offset[1] += delta.y;
            actions.push(NodeGraphAction::PanChanged { offset: self.scroll_offset });
        }

        // Zoom (scroll wheel)
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta.abs() > 0.1 {
            let old_zoom = self.zoom;
            self.zoom = (self.zoom * (1.0 + scroll_delta * 0.001)).clamp(0.2, 4.0);
            if (self.zoom - old_zoom).abs() > 1e-4 {
                actions.push(NodeGraphAction::ZoomChanged { zoom: self.zoom });
            }
        }

        // Draw connections as cubic bezier curves
        for conn in &self.connections {
            let from_pos = self.pin_screen_pos(conn.from_node, &conn.from_pin, true, avail.min);
            let to_pos   = self.pin_screen_pos(conn.to_node,   &conn.to_pin,   false, avail.min);
            if let (Some(fp), Some(tp)) = (from_pos, to_pos) {
                let ctrl_dx = ((tp.x - fp.x).abs() * 0.5).max(50.0);
                painter.add(egui::Shape::CubicBezier(egui::epaint::CubicBezierShape {
                    points: [
                        fp,
                        egui::pos2(fp.x + ctrl_dx, fp.y),
                        egui::pos2(tp.x - ctrl_dx, tp.y),
                        tp,
                    ],
                    closed: false,
                    fill: egui::Color32::TRANSPARENT,
                    stroke: egui::Stroke::new(2.0, egui::Color32::from_rgb(
                        conn.color[0], conn.color[1], conn.color[2],
                    )).into(),
                }));
            }
        }

        // Draw nodes — collect move/select actions separately to avoid borrow conflict
        let mut move_actions:   Vec<(u32, [f32; 2])> = Vec::new();
        let mut select_actions: Vec<u32>              = Vec::new();

        for node in &mut self.nodes {
            let sx = avail.min.x + node.position[0] * self.zoom + self.scroll_offset[0];
            let sy = avail.min.y + node.position[1] * self.zoom + self.scroll_offset[1];
            let sw = node.size[0] * self.zoom;
            let pin_rows = node.inputs.len().max(node.outputs.len()) as f32;
            let sh = (30.0 + pin_rows * 24.0 + 8.0) * self.zoom;

            let node_rect = egui::Rect::from_min_size(egui::pos2(sx, sy), egui::vec2(sw, sh));
            let node_resp = ui.allocate_rect(node_rect, egui::Sense::click_and_drag());

            // Body
            let bg = if node.selected {
                egui::Color32::from_rgb(55, 65, 90)
            } else {
                egui::Color32::from_rgb(36, 36, 48)
            };
            painter.rect_filled(node_rect, 4.0 * self.zoom, bg);

            // Title bar
            let title_h = 24.0 * self.zoom;
            let title_rect = egui::Rect::from_min_size(node_rect.min, egui::vec2(sw, title_h));
            painter.rect_filled(
                title_rect,
                egui::Rounding { nw: 4.0 * self.zoom, ne: 4.0 * self.zoom, sw: 0.0, se: 0.0 },
                egui::Color32::from_rgb(node.color[0], node.color[1], node.color[2]),
            );
            painter.text(
                title_rect.center(),
                egui::Align2::CENTER_CENTER,
                &node.title,
                egui::FontId::proportional((11.0 * self.zoom).max(8.0)),
                egui::Color32::WHITE,
            );

            // Border
            let border = if node.selected {
                egui::Color32::from_rgb(100, 160, 255)
            } else {
                egui::Color32::from_rgb(58, 68, 100)
            };
            painter.rect_stroke(node_rect, 4.0 * self.zoom,
                egui::Stroke::new(1.5, border), egui::StrokeKind::Outside);

            // Input pins (left side)
            for (i, pin) in node.inputs.iter().enumerate() {
                let py = sy + (30.0 + i as f32 * 24.0 + 8.0) * self.zoom;
                let c = pin.pin_type.color();
                let pin_col = egui::Color32::from_rgb(c[0], c[1], c[2]);
                painter.circle_filled(egui::pos2(sx, py), 5.0 * self.zoom, pin_col);
                painter.text(
                    egui::pos2(sx + 10.0 * self.zoom, py),
                    egui::Align2::LEFT_CENTER,
                    &pin.name,
                    egui::FontId::proportional((10.0 * self.zoom).max(7.0)),
                    egui::Color32::from_rgb(190, 190, 200),
                );
            }

            // Output pins (right side)
            for (i, pin) in node.outputs.iter().enumerate() {
                let py = sy + (30.0 + i as f32 * 24.0 + 8.0) * self.zoom;
                let c = pin.pin_type.color();
                let pin_col = egui::Color32::from_rgb(c[0], c[1], c[2]);
                painter.circle_filled(egui::pos2(sx + sw, py), 5.0 * self.zoom, pin_col);
                painter.text(
                    egui::pos2(sx + sw - 10.0 * self.zoom, py),
                    egui::Align2::RIGHT_CENTER,
                    &pin.name,
                    egui::FontId::proportional((10.0 * self.zoom).max(7.0)),
                    egui::Color32::from_rgb(190, 190, 200),
                );
            }

            // Drag to move
            if node_resp.dragged_by(egui::PointerButton::Primary) {
                let delta = node_resp.drag_delta();
                let new_pos = [
                    node.position[0] + delta.x / self.zoom,
                    node.position[1] + delta.y / self.zoom,
                ];
                move_actions.push((node.id, new_pos));
            }

            // Click to select
            if node_resp.clicked() {
                select_actions.push(node.id);
            }
        }

        // Apply mutations after the borrow-loop ends
        for (id, pos) in move_actions {
            if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
                n.position = pos;
            }
            actions.push(NodeGraphAction::NodeMoved { id, new_pos: pos });
        }
        for id in select_actions {
            self.selected_nodes.clear();
            self.selected_nodes.push(id);
            for n in &mut self.nodes {
                n.selected = n.id == id;
            }
            actions.push(NodeGraphAction::NodeSelected { id });
        }

        actions
    }

    /// Screen-space position of a pin for connection curve drawing.
    fn pin_screen_pos(
        &self,
        node_id:   u32,
        pin_name:  &str,
        is_output: bool,
        origin:    egui::Pos2,
    ) -> Option<egui::Pos2> {
        let node = self.nodes.iter().find(|n| n.id == node_id)?;
        let pins = if is_output { &node.outputs } else { &node.inputs };
        let idx = pins.iter().position(|p| p.name == pin_name)?;
        let sx = origin.x + node.position[0] * self.zoom + self.scroll_offset[0];
        let sy = origin.y + node.position[1] * self.zoom + self.scroll_offset[1];
        let sw = node.size[0] * self.zoom;
        let x  = if is_output { sx + sw } else { sx };
        let y  = sy + (30.0 + idx as f32 * 24.0 + 8.0) * self.zoom;
        Some(egui::pos2(x, y))
    }
```

- [ ] **Step 4: Run test**

Run: `cargo test -p vox_ui show_egui_method_exists`
Expected: PASS

Run: `cargo build -p vox_ui`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add crates/vox_ui/src/node_graph_widget.rs
git commit -m "feat(vox_ui): NodeGraphWidget.show_egui() — interactive egui rendering, bezier connections, zoom, pan, drag"
```

---

### Task 5: Wire MaterialEditorUi to OchrGraph + NodeGraphWidget

**Files:**
- Modify: `crates/vox_render/Cargo.toml`
- Modify: `crates/vox_render/src/material_editor_ui.rs`

Before editing, read:
- `crates/vox_render/src/lib.rs` — check whether `material_editor` module is re-exported or used by any other file. If only used from `material_editor_ui.rs`, remove its `pub mod` from lib.rs too. If used elsewhere, keep it.
- `crates/vox_render/src/material_editor.rs` — understand `MaterialGraph`, `MaterialNodeType`, `MaterialConnection`.

- [ ] **Step 1: Add vox_nodes to vox_render's Cargo.toml**

Open `crates/vox_render/Cargo.toml` and add to `[dependencies]`:

```toml
vox_nodes = { workspace = true }
vox_ui    = { path = "../vox_ui" }
```

(`vox_ui` may already be present — check first and skip if so.)

- [ ] **Step 2: Write failing tests**

Add to the bottom of `crates/vox_render/src/material_editor_ui.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_editor_starts_empty() {
        let ed = MaterialEditorUi::new();
        assert!(!ed.open);
        assert_eq!(ed.graph.graph.node_count(), 0);
    }

    #[test]
    fn create_default_graph_has_nodes() {
        let mut ed = MaterialEditorUi::new();
        ed.create_default_graph();
        // Roughness node + Metallic node + MaterialOutput node = 3
        assert_eq!(ed.graph.graph.node_count(), 3);
    }

    #[test]
    fn create_default_graph_cooks_cleanly() {
        let mut ed = MaterialEditorUi::new();
        ed.create_default_graph();
        ed.graph.graph.cook().unwrap();
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p vox_render material_editor`
Expected: FAIL — `graph` field not found on `MaterialEditorUi`

- [ ] **Step 4: Replace MaterialEditorUi implementation**

Rewrite `crates/vox_render/src/material_editor_ui.rs` with:

```rust
//! egui window for the material node graph editor.
//! Uses OchrGraph (crucible-core backend) + NodeGraphWidget (egui rendering).

use vox_nodes::{OchrGraph, NodeId};
use vox_nodes::mat_nodes::{
    FloatConstNode, MaterialOutputNode, MultiplyNode, AddNode, LerpNode, OneMinusNode,
};
use vox_ui::node_graph_widget::{NodeGraphWidget, VisualPin, VisualPinType};

pub struct MaterialEditorUi {
    pub open: bool,
    pub name: String,
    pub graph: OchrGraph,
    widget: NodeGraphWidget,
    selected_node: Option<NodeId>,
}

impl MaterialEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            name: String::new(),
            graph: OchrGraph::new(),
            widget: NodeGraphWidget::new(),
            selected_node: None,
        }
    }

    /// Build a default Roughness + Metallic → MaterialOutput graph.
    pub fn create_default_graph(&mut self) {
        self.graph = OchrGraph::new();
        self.name = "New Material".to_string();

        let roughness_id = self.graph.add_node(
            "Roughness", Box::new(FloatConstNode::new(0.5)), [80.0, 60.0],
        );
        let metallic_id = self.graph.add_node(
            "Metallic", Box::new(FloatConstNode::new(0.0)), [80.0, 170.0],
        );
        let output_id = self.graph.add_node(
            "Output", Box::new(MaterialOutputNode), [320.0, 110.0],
        );

        // roughness → output.roughness, metallic → output.metallic
        // MaterialOutputNode has declared inputs so connect() type-checks them.
        let _ = self.graph.connect(roughness_id, "out", output_id, "roughness");
        let _ = self.graph.connect(metallic_id,  "out", output_id, "metallic");

        let _ = self.graph.graph.cook();
        self.sync_widget();
    }

    /// Rebuild the NodeGraphWidget from current OchrGraph state.
    /// Must be called after any graph topology change.
    fn sync_widget(&mut self) {
        self.widget = NodeGraphWidget::new();
        let snap = self.graph.graph.snapshot();

        for mut vn in self.graph.to_visual_nodes() {
            // Assign pins and color by node type
            if let Some(ns) = snap.nodes.iter().find(|ns| ns.id == vn.id) {
                match ns.type_name.as_str() {
                    "FloatConst" => {
                        vn.outputs = vec![VisualPin {
                            name: "out".into(),
                            pin_type: VisualPinType::Float,
                            connected: false,
                        }];
                        vn.color = [45, 110, 65];
                        vn.size = [130.0, 55.0];
                    }
                    "Multiply" | "Add" | "Lerp" => {
                        vn.inputs = vec![
                            VisualPin { name: "a".into(), pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "b".into(), pin_type: VisualPinType::Float, connected: false },
                        ];
                        vn.outputs = vec![VisualPin { name: "out".into(), pin_type: VisualPinType::Float, connected: false }];
                        vn.color = [80, 55, 110];
                        vn.size = [130.0, 80.0];
                    }
                    "OneMinus" => {
                        vn.inputs  = vec![VisualPin { name: "in".into(),  pin_type: VisualPinType::Float, connected: false }];
                        vn.outputs = vec![VisualPin { name: "out".into(), pin_type: VisualPinType::Float, connected: false }];
                        vn.color = [80, 55, 110];
                        vn.size = [130.0, 60.0];
                    }
                    "MaterialOutput" => {
                        vn.inputs = vec![
                            VisualPin { name: "base_r".into(),    pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "base_g".into(),    pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "base_b".into(),    pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "roughness".into(), pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "metallic".into(),  pin_type: VisualPinType::Float, connected: false },
                        ];
                        vn.color = [140, 55, 55];
                        vn.size = [160.0, 175.0];
                    }
                    _ => {}
                }
            }
            self.widget.add_node(vn);
        }

        for vc in self.graph.to_visual_connections() {
            self.widget.add_connection(vc);
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        egui::Window::new("Material Editor")
            .open(&mut self.open)
            .default_size([950.0, 560.0])
            .resizable(true)
            .show(ctx, |ui| {
                // Toolbar
                ui.horizontal(|ui| {
                    if self.name.is_empty() {
                        if ui.button("New Material").clicked() {
                            self.create_default_graph();
                        }
                        return;
                    }
                    ui.heading(self.name.clone());
                    ui.separator();
                    ui.label(format!("{} nodes", self.graph.graph.node_count()));
                    ui.separator();
                    if ui.button("+ Float").clicked() {
                        let n = self.graph.graph.node_count() as f32;
                        let id = self.graph.add_node(
                            "Float", Box::new(FloatConstNode::new(1.0)),
                            [50.0 + n * 20.0, 50.0 + n * 20.0],
                        );
                        let _ = self.graph.graph.cook();
                        self.sync_widget();
                        let _ = id;
                    }
                    if ui.button("+ Multiply").clicked() {
                        let n = self.graph.graph.node_count() as f32;
                        let id = self.graph.add_node(
                            "Multiply", Box::new(MultiplyNode),
                            [200.0 + n * 10.0, 50.0],
                        );
                        let _ = self.graph.graph.cook();
                        self.sync_widget();
                        let _ = id;
                    }
                    if ui.button("+ Add").clicked() {
                        let n = self.graph.graph.node_count() as f32;
                        let id = self.graph.add_node(
                            "Add", Box::new(AddNode), [200.0 + n * 10.0, 100.0],
                        );
                        let _ = self.graph.graph.cook();
                        self.sync_widget();
                        let _ = id;
                    }
                    if ui.button("Cook").clicked() {
                        let _ = self.graph.graph.cook();
                    }
                });

                if self.name.is_empty() { return; }
                ui.separator();

                // Node graph viewport
                let actions = self.widget.show_egui(ui);

                // Process actions
                for action in actions {
                    use vox_ui::node_graph_widget::NodeGraphAction;
                    match action {
                        NodeGraphAction::NodeMoved { id, new_pos } => {
                            self.graph.set_position(NodeId(id), new_pos);
                        }
                        NodeGraphAction::NodeSelected { id } => {
                            self.selected_node = Some(NodeId(id));
                        }
                        NodeGraphAction::ConnectionCreated { from_node, from_pin, to_node, to_pin } => {
                            let _ = self.graph.connect(
                                NodeId(from_node), &from_pin,
                                NodeId(to_node),   &to_pin,
                            );
                            let _ = self.graph.graph.cook();
                            self.sync_widget();
                        }
                        NodeGraphAction::NodeDeleted { id } => {
                            self.widget.remove_node(id);
                        }
                        _ => {}
                    }
                }
            });
    }
}

impl Default for MaterialEditorUi {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_editor_starts_empty() {
        let ed = MaterialEditorUi::new();
        assert!(!ed.open);
        assert_eq!(ed.graph.graph.node_count(), 0);
    }

    #[test]
    fn create_default_graph_has_nodes() {
        let mut ed = MaterialEditorUi::new();
        ed.create_default_graph();
        assert_eq!(ed.graph.graph.node_count(), 3);
    }

    #[test]
    fn create_default_graph_cooks_cleanly() {
        let mut ed = MaterialEditorUi::new();
        ed.create_default_graph();
        ed.graph.graph.cook().unwrap();
    }
}
```

- [ ] **Step 5: Check whether material_editor.rs is still needed**

Read `crates/vox_render/src/lib.rs`. If it contains `pub mod material_editor;` and nothing outside `material_editor_ui.rs` imports from it, remove that line from lib.rs. If other code uses `MaterialGraph` etc., keep it.

- [ ] **Step 6: Run tests**

Run: `cargo test -p vox_render material_editor`
Expected: 3 tests PASS

Run: `cargo build -p vox_render`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add crates/vox_render/src/material_editor_ui.rs crates/vox_render/src/lib.rs crates/vox_render/Cargo.toml
git commit -m "feat(material-editor): replace custom renderer with OchrGraph + NodeGraphWidget — live crucible-core evaluation"
```

---

### Task 6: Wire VFX editor and Animation editor to NodeGraphWidget

**Files:**
- Modify: `crates/vox_render/src/vfx_editor_ui.rs`
- Modify: `crates/vox_render/src/anim_editor_ui.rs`

Before editing, read both files to understand the current stubs.

- [ ] **Step 1: Write failing tests**

Add to `crates/vox_render/src/vfx_editor_ui.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vfx_editor_has_ochrgraph() {
        let ed = VfxEditorUi::new();
        assert_eq!(ed.graph.graph.node_count(), 0);
    }
}
```

Add to `crates/vox_render/src/anim_editor_ui.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anim_editor_has_ochrgraph() {
        let ed = AnimEditorUi::new();
        assert_eq!(ed.graph.graph.node_count(), 0);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vox_render vfx_editor_has_ochrgraph anim_editor_has_ochrgraph`
Expected: FAIL — `graph` field not found

- [ ] **Step 3: Rewrite VfxEditorUi**

Replace `crates/vox_render/src/vfx_editor_ui.rs` with:

```rust
//! VFX node graph editor window.

use vox_nodes::OchrGraph;
use vox_ui::node_graph_widget::NodeGraphWidget;

pub struct VfxEditorUi {
    pub open: bool,
    pub graph: OchrGraph,
    widget: NodeGraphWidget,
}

impl VfxEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: OchrGraph::new(),
            widget: NodeGraphWidget::new(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }
        egui::Window::new("VFX Editor")
            .open(&mut self.open)
            .default_size([950.0, 560.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("VFX Graph");
                    ui.separator();
                    ui.label(format!("{} nodes", self.graph.graph.node_count()));
                });
                ui.separator();
                let _actions = self.widget.show_egui(ui);
            });
    }
}

impl Default for VfxEditorUi {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vfx_editor_has_ochrgraph() {
        let ed = VfxEditorUi::new();
        assert_eq!(ed.graph.graph.node_count(), 0);
    }
}
```

- [ ] **Step 4: Rewrite AnimEditorUi**

Replace `crates/vox_render/src/anim_editor_ui.rs` with:

```rust
//! Animation state machine editor window.

use vox_nodes::OchrGraph;
use vox_ui::node_graph_widget::NodeGraphWidget;

pub struct AnimEditorUi {
    pub open: bool,
    pub graph: OchrGraph,
    widget: NodeGraphWidget,
}

impl AnimEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: OchrGraph::new(),
            widget: NodeGraphWidget::new(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }
        egui::Window::new("Animation Editor")
            .open(&mut self.open)
            .default_size([950.0, 560.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Animation State Machine");
                    ui.separator();
                    ui.label(format!("{} states", self.graph.graph.node_count()));
                });
                ui.separator();
                let _actions = self.widget.show_egui(ui);
            });
    }
}

impl Default for AnimEditorUi {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anim_editor_has_ochrgraph() {
        let ed = AnimEditorUi::new();
        assert_eq!(ed.graph.graph.node_count(), 0);
    }
}
```

- [ ] **Step 5: Run all vox_render tests**

Run: `cargo test -p vox_render`
Expected: all tests PASS

- [ ] **Step 6: Full workspace build**

Run: `cargo build`
Expected: exits 0, no errors

- [ ] **Step 7: Commit**

```bash
git add crates/vox_render/src/vfx_editor_ui.rs crates/vox_render/src/anim_editor_ui.rs
git commit -m "feat(vfx-anim-editors): replace stubs with OchrGraph + NodeGraphWidget"
```

---

## Self-Review

**Spec coverage check:**
- ✅ crucible-core as path dep → Task 1
- ✅ OchrGraph bridge type → Task 2
- ✅ Material CrucibleNode impls → Task 3
- ✅ NodeGraphWidget egui rendering → Task 4
- ✅ Material editor wired → Task 5
- ✅ VFX + Anim editors wired → Task 6

**Placeholder scan:** No TBD, TODO, or "implement later" present.

**Type consistency:**
- `OchrGraph` defined Task 2, used Tasks 5–6 ✅
- `FloatConstNode`, `MultiplyNode`, etc. defined Task 3, used Task 5 ✅
- `NodeGraphWidget::show_egui()` defined Task 4, used Tasks 5–6 ✅
- `NodeId(u32)` tuple struct — re-exported from vox_nodes, used consistently ✅
- `PortData::Scalar(f64)` — consistent across all material nodes ✅
- `CookError::UnknownParam { key, node }` and `CookError::MissingInput(String)` — correct variants from graph.rs tests ✅
