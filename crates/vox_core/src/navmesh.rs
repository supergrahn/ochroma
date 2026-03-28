use glam::Vec2;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

/// A navigation mesh node.
#[derive(Debug, Clone)]
pub struct NavNode {
    pub id: u32,
    pub position: Vec2,
    pub walkable: bool,
}

/// An edge between two nav nodes.
#[derive(Debug, Clone)]
pub struct NavEdge {
    pub from: u32,
    pub to: u32,
    pub cost: f32, // distance or weighted cost
}

/// Navigation mesh for pathfinding.
pub struct NavMesh {
    pub nodes: Vec<NavNode>,
    pub edges: Vec<NavEdge>,
    adjacency: HashMap<u32, Vec<(u32, f32)>>, // node_id -> [(neighbor_id, cost)]
}

impl Default for NavMesh {
    fn default() -> Self {
        Self::new()
    }
}

impl NavMesh {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), adjacency: HashMap::new() }
    }

    pub fn add_node(&mut self, id: u32, position: Vec2, walkable: bool) {
        self.nodes.push(NavNode { id, position, walkable });
    }

    pub fn add_edge(&mut self, from: u32, to: u32) {
        let from_pos = self.nodes.iter().find(|n| n.id == from).map(|n| n.position);
        let to_pos = self.nodes.iter().find(|n| n.id == to).map(|n| n.position);
        if let (Some(fp), Some(tp)) = (from_pos, to_pos) {
            let cost = fp.distance(tp);
            self.edges.push(NavEdge { from, to, cost });
            self.adjacency.entry(from).or_default().push((to, cost));
            self.adjacency.entry(to).or_default().push((from, cost)); // bidirectional
        }
    }

    /// Find shortest path using A* algorithm.
    pub fn find_path(&self, start: u32, goal: u32) -> Option<Vec<u32>> {
        let goal_pos = self.nodes.iter().find(|n| n.id == goal)?.position;

        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<u32, u32> = HashMap::new();
        let mut g_score: HashMap<u32, f32> = HashMap::new();

        g_score.insert(start, 0.0);
        let start_pos = self.nodes.iter().find(|n| n.id == start)?.position;
        open.push(AStarEntry { node: start, f_score: start_pos.distance(goal_pos) });

        while let Some(current) = open.pop() {
            if current.node == goal {
                // Reconstruct path
                let mut path = vec![goal];
                let mut node = goal;
                while let Some(&prev) = came_from.get(&node) {
                    path.push(prev);
                    node = prev;
                }
                path.reverse();
                return Some(path);
            }

            let current_g = g_score[&current.node];

            if let Some(neighbors) = self.adjacency.get(&current.node) {
                for &(neighbor, cost) in neighbors {
                    // Check walkable
                    if !self.nodes.iter().find(|n| n.id == neighbor).map(|n| n.walkable).unwrap_or(false) {
                        continue;
                    }

                    let tentative_g = current_g + cost;
                    if tentative_g < *g_score.get(&neighbor).unwrap_or(&f32::MAX) {
                        came_from.insert(neighbor, current.node);
                        g_score.insert(neighbor, tentative_g);
                        let neighbor_pos = self.nodes.iter().find(|n| n.id == neighbor).unwrap().position;
                        let f = tentative_g + neighbor_pos.distance(goal_pos);
                        open.push(AStarEntry { node: neighbor, f_score: f });
                    }
                }
            }
        }

        None // No path found
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// Find the nearest walkable node to a world position.
    pub fn nearest_node(&self, position: Vec2) -> Option<u32> {
        self.nodes.iter()
            .filter(|n| n.walkable)
            .min_by(|a, b| {
                let da = a.position.distance(position);
                let db = b.position.distance(position);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .map(|n| n.id)
    }
}

#[derive(Debug)]
struct AStarEntry {
    node: u32,
    f_score: f32,
}

impl PartialEq for AStarEntry {
    fn eq(&self, other: &Self) -> bool { self.f_score == other.f_score }
}
impl Eq for AStarEntry {}
impl PartialOrd for AStarEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for AStarEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal) // min-heap
    }
}
