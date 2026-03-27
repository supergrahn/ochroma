use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UtilityType { Water, Power, Sewage }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityNode {
    pub id: u32,
    pub utility_type: UtilityType,
    pub position: [f32; 2],
    pub capacity: f32,
    pub current_load: f32,
    pub is_source: bool, // true = generates, false = consumes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityEdge {
    pub from: u32,
    pub to: u32,
    pub capacity: f32,
    pub current_flow: f32,
}

pub struct UtilityNetwork {
    pub utility_type: UtilityType,
    pub nodes: Vec<UtilityNode>,
    pub edges: Vec<UtilityEdge>,
    next_id: u32,
}

impl UtilityNetwork {
    pub fn new(utility_type: UtilityType) -> Self {
        Self { utility_type, nodes: Vec::new(), edges: Vec::new(), next_id: 0 }
    }

    pub fn add_source(&mut self, position: [f32; 2], capacity: f32) -> u32 {
        let id = self.next_id; self.next_id += 1;
        self.nodes.push(UtilityNode { id, utility_type: self.utility_type, position, capacity, current_load: 0.0, is_source: true });
        id
    }

    pub fn add_consumer(&mut self, position: [f32; 2], demand: f32) -> u32 {
        let id = self.next_id; self.next_id += 1;
        self.nodes.push(UtilityNode { id, utility_type: self.utility_type, position, capacity: demand, current_load: 0.0, is_source: false });
        id
    }

    pub fn connect(&mut self, from: u32, to: u32, capacity: f32) {
        self.edges.push(UtilityEdge { from, to, capacity, current_flow: 0.0 });
    }

    /// Check if a consumer node is served (connected to a source with sufficient capacity).
    pub fn is_served(&self, node_id: u32) -> bool {
        // BFS from node to any source through edges
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = vec![node_id];

        while let Some(current) = queue.pop() {
            if let Some(idx) = self.nodes.iter().position(|n| n.id == current) {
                if visited[idx] { continue; }
                visited[idx] = true;
                if self.nodes[idx].is_source { return true; }
            }
            // Find connected nodes
            for edge in &self.edges {
                if edge.from == current { queue.push(edge.to); }
                if edge.to == current { queue.push(edge.from); }
            }
        }
        false
    }

    pub fn total_capacity(&self) -> f32 {
        self.nodes.iter().filter(|n| n.is_source).map(|n| n.capacity).sum()
    }

    pub fn total_demand(&self) -> f32 {
        self.nodes.iter().filter(|n| !n.is_source).map(|n| n.capacity).sum()
    }

    pub fn has_deficit(&self) -> bool { self.total_demand() > self.total_capacity() }
}
