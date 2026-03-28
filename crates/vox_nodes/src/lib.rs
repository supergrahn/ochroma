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
