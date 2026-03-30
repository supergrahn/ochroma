//! SDF-derived walkable navmesh.
//!
//! `NavMesh` stores a graph of walkable positions extracted from the terrain SDF.
//! Pathfinding uses A* over this graph.
//! `invalidate_region` + `merge` enable cheap incremental updates after deformation.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// Spatial grid index for O(1) average nearest-node lookup.
#[derive(Default, Clone, Debug)]
struct NavMeshGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<u32>>,
}

impl NavMeshGrid {
    fn new(cell_size: f32) -> Self {
        Self { cell_size, cells: HashMap::new() }
    }
    fn cell(&self, pos: [f32; 3]) -> (i32, i32) {
        ((pos[0] / self.cell_size).floor() as i32, (pos[2] / self.cell_size).floor() as i32)
    }
    fn insert(&mut self, id: u32, pos: [f32; 3]) {
        self.cells.entry(self.cell(pos)).or_default().push(id);
    }
    fn candidates(&self, pos: [f32; 3]) -> Vec<u32> {
        let (cx, cz) = self.cell(pos);
        let mut out = Vec::new();
        for dx in -1..=1i32 {
            for dz in -1..=1i32 {
                if let Some(ids) = self.cells.get(&(cx + dx, cz + dz)) {
                    out.extend_from_slice(ids);
                }
            }
        }
        out
    }
}

/// A walkable position in the navmesh.
#[derive(Debug, Clone)]
pub struct NavNode {
    pub id: u32,
    pub world_pos: [f32; 3],
    /// Indices into `NavMesh::nodes` for adjacent walkable nodes.
    pub neighbours: Vec<u32>,
}

/// Walkable graph derived from terrain SDF.
#[derive(Debug)]
pub struct NavMesh {
    pub nodes: Vec<NavNode>,
    grid: NavMeshGrid,
}

impl Default for NavMesh {
    fn default() -> Self {
        Self { nodes: Vec::new(), grid: NavMeshGrid::new(5.0) }
    }
}

impl NavMesh {
    pub fn new() -> Self { Self::default() }

    /// Number of walkable nodes.
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Rebuild the spatial grid index from current nodes.
    /// Call after batch-inserting nodes or after `merge()` / `invalidate_region()`.
    pub fn rebuild_grid(&mut self) {
        self.grid = NavMeshGrid::new(5.0);
        for node in &self.nodes {
            self.grid.insert(node.id, node.world_pos);
        }
    }

    /// Find the nearest node to a world position.
    /// Uses spatial grid when available (O(1) avg), falls back to O(n) linear scan.
    pub fn nearest_node(&self, world_pos: [f32; 3]) -> Option<u32> {
        let candidates = self.grid.candidates(world_pos);
        if !candidates.is_empty() {
            return candidates.into_iter().min_by(|&a, &b| {
                let da = dist2(self.nodes[a as usize].world_pos, world_pos);
                let db = dist2(self.nodes[b as usize].world_pos, world_pos);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            });
        }
        self.nodes.iter()
            .min_by(|a, b| {
                let da = dist2(a.world_pos, world_pos);
                let db = dist2(b.world_pos, world_pos);
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
            .map(|n| n.id)
    }

    /// A* pathfinding from `start_node` to `goal_node`.
    /// Returns Some(path) as a list of world positions, or None if no path exists.
    /// # Panics
    /// Panics if `start_id` or `goal_id` are out of bounds for `self.nodes`.
    /// Use `nearest_node()` to obtain valid IDs.
    pub fn find_path(&self, start_id: u32, goal_id: u32) -> Option<Vec<[f32; 3]>> {
        if start_id == goal_id {
            return Some(vec![self.nodes[start_id as usize].world_pos]);
        }
        let goal_pos = self.nodes[goal_id as usize].world_pos;

        let mut open: BinaryHeap<AStarEntry> = BinaryHeap::new();
        let mut g_cost: HashMap<u32, f32> = HashMap::new();
        let mut came_from: HashMap<u32, u32> = HashMap::new();

        g_cost.insert(start_id, 0.0);
        open.push(AStarEntry { node: start_id, f: heuristic(self.nodes[start_id as usize].world_pos, goal_pos) });

        while let Some(AStarEntry { node: current, .. }) = open.pop() {
            if current == goal_id {
                return Some(self.reconstruct_path(&came_from, goal_id));
            }
            let current_g = *g_cost.get(&current).unwrap_or(&f32::MAX);
            for &neighbour in &self.nodes[current as usize].neighbours {
                let edge_cost = dist(self.nodes[current as usize].world_pos,
                                    self.nodes[neighbour as usize].world_pos);
                let new_g = current_g + edge_cost;
                if new_g < *g_cost.get(&neighbour).unwrap_or(&f32::MAX) {
                    g_cost.insert(neighbour, new_g);
                    came_from.insert(neighbour, current);
                    let f = new_g + heuristic(self.nodes[neighbour as usize].world_pos, goal_pos);
                    open.push(AStarEntry { node: neighbour, f });
                }
            }
        }
        None
    }

    /// Remove nodes in a sphere (call after terrain deformation).
    pub fn invalidate_region(&mut self, center: [f32; 3], radius: f32) {
        let r2 = radius * radius;
        let removed: HashSet<u32> = self.nodes.iter()
            .filter(|n| dist2(n.world_pos, center) <= r2)
            .map(|n| n.id)
            .collect();
        self.nodes.retain(|n| !removed.contains(&n.id));
        // Remove references to deleted nodes from neighbour lists
        for node in &mut self.nodes {
            node.neighbours.retain(|id| !removed.contains(id));
        }
    }

    /// Merge additional nodes (from re-extraction of deformed region) into this navmesh.
    pub fn merge(&mut self, new_nodes: Vec<NavNode>) {
        let id_offset = self.nodes.len() as u32;
        for mut node in new_nodes {
            let old_id = node.id;
            node.id = old_id + id_offset;
            node.neighbours = node.neighbours.iter().map(|&n| n + id_offset).collect();
            self.nodes.push(node);
        }
        self.rebuild_grid();
    }

    fn reconstruct_path(&self, came_from: &HashMap<u32, u32>, goal: u32) -> Vec<[f32; 3]> {
        let mut path = Vec::new();
        let mut current = goal;
        loop {
            path.push(self.nodes[current as usize].world_pos);
            match came_from.get(&current) {
                Some(&prev) => current = prev,
                None => break,
            }
        }
        path.reverse();
        path
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
    dx*dx + dy*dy + dz*dz
}
fn dist(a: [f32; 3], b: [f32; 3]) -> f32 { dist2(a, b).sqrt() }
fn heuristic(a: [f32; 3], b: [f32; 3]) -> f32 { dist(a, b) }

#[derive(PartialEq)]
struct AStarEntry { node: u32, f: f32 }
impl Eq for AStarEntry {}
impl Ord for AStarEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for AStarEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_navmesh(n: usize) -> NavMesh {
        let mut mesh = NavMesh::new();
        for i in 0..n {
            let id = i as u32;
            let neighbours = if i + 1 < n { vec![id + 1] } else { vec![] }
                .into_iter()
                .chain(if i > 0 { vec![id - 1] } else { vec![] })
                .collect();
            mesh.nodes.push(NavNode {
                id,
                world_pos: [i as f32, 0.0, 0.0],
                neighbours,
            });
        }
        mesh
    }

    #[test]
    fn find_path_linear_graph() {
        let mesh = make_linear_navmesh(5);
        let path = mesh.find_path(0, 4).expect("path must exist");
        assert_eq!(path.len(), 5, "path should visit all 5 nodes");
        assert!((path[0][0] - 0.0).abs() < 0.01);
        assert!((path[4][0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn find_path_same_node_returns_single() {
        let mesh = make_linear_navmesh(3);
        let path = mesh.find_path(1, 1).unwrap();
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn find_path_disconnected_returns_none() {
        let mut mesh = NavMesh::new();
        mesh.nodes.push(NavNode { id: 0, world_pos: [0.0; 3], neighbours: vec![] });
        mesh.nodes.push(NavNode { id: 1, world_pos: [5.0, 0.0, 0.0], neighbours: vec![] });
        assert!(mesh.find_path(0, 1).is_none());
    }

    #[test]
    fn invalidate_region_removes_nodes() {
        let mut mesh = make_linear_navmesh(5);
        // Nodes 1 and 2 are at x=1,2 — within radius 1.5 of x=1.5
        mesh.invalidate_region([1.5, 0.0, 0.0], 1.5);
        assert!(mesh.nodes.len() < 5, "some nodes should be removed");
        for node in &mesh.nodes {
            assert!(node.id != 1 && node.id != 2, "nodes 1 and 2 should be gone");
        }
    }

    #[test]
    fn nearest_node_finds_closest() {
        let mesh = make_linear_navmesh(4);
        let nearest = mesh.nearest_node([2.1, 0.0, 0.0]).unwrap();
        assert_eq!(nearest, 2, "nearest to 2.1 should be node 2");
    }
}
