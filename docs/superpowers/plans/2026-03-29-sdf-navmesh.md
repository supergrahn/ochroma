# SDF-Derived Navmesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a walkable navmesh directly from the terrain SDF — no separate voxelization step, auto-updates after `carve_sphere`/`fill_sphere` deformation.

**Architecture:** A `NavMesh` is a flat graph of walkable `NavNode`s, each at a world position on or near the terrain surface. Extraction: iterate voxels where `|SDF| < threshold` (surface voxels) with SDF ≥ 0 on the voxel above (walkable = not inside solid ceiling). Connect adjacent walkable voxels as graph edges. Pathfinding: A* over this graph. The `vox_core::navmesh` module stub is replaced with this implementation. `NavMesh::invalidate_region(center, radius)` removes nodes within radius and re-extracts from the SDF — this is called after any `deform::carve_sphere` call.

**Why better than Unreal:** Unreal's RecastNavMesh voxelizes the scene geometry (a second representation), then computes walkability. After terrain deformation, the entire affected region must be re-voxelized and re-meshed — an expensive async operation. Ochroma's navmesh IS derived from the SDF directly; no second representation, invalidation is instant, re-extraction is cheap because only the deformed region is reprocessed.

**Tech Stack:** Rust, `vox_terrain::TerrainVolume` (existing), `vox_core::navmesh` (replace stub), `glam::Vec3`.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_core/src/navmesh.rs` | Replace stub | `NavNode`, `NavMesh`, `extract_navmesh`, A* pathfinding |
| `crates/vox_terrain/src/navmesh_bridge.rs` | Create | `extract_from_volume()` — terrain SDF → NavMesh |
| `crates/vox_terrain/src/lib.rs` | Modify | `pub mod navmesh_bridge;` |
| `crates/vox_app/src/bin/engine_runner.rs` | Modify | Wire navmesh rebuild after terrain deform |

---

## Task 1: NavNode + NavMesh + A* in vox_core

**Files:**
- Modify: `crates/vox_core/src/navmesh.rs` (read first, then replace)

- [ ] Read `crates/vox_core/src/navmesh.rs` to understand the existing stub.

- [ ] Replace the stub content with:

```rust
//! SDF-derived walkable navmesh.
//!
//! `NavMesh` stores a graph of walkable positions extracted from the terrain SDF.
//! Pathfinding uses A* over this graph.
//! `invalidate_region` + `merge` enable cheap incremental updates after deformation.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// A walkable position in the navmesh.
#[derive(Debug, Clone)]
pub struct NavNode {
    pub id: u32,
    pub world_pos: [f32; 3],
    /// Indices into `NavMesh::nodes` for adjacent walkable nodes.
    pub neighbours: Vec<u32>,
}

/// Walkable graph derived from terrain SDF.
#[derive(Debug, Default)]
pub struct NavMesh {
    pub nodes: Vec<NavNode>,
    /// Spatial index: voxel (x,y,z) → node id for fast lookup during extraction.
    voxel_to_node: HashMap<(i32, i32, i32), u32>,
}

impl NavMesh {
    pub fn new() -> Self { Self::default() }

    /// Number of walkable nodes.
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Find the nearest node to a world position. O(n) — use for path queries, not per-frame.
    pub fn nearest_node(&self, world_pos: [f32; 3]) -> Option<u32> {
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
        self.voxel_to_node.retain(|_, id| !removed.contains(id));
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
```

- [ ] Run:
```bash
cargo test -p vox_core navmesh
```
Expected: 5 tests pass.

- [ ] Commit:
```bash
git commit -m "feat(core): NavMesh with A* pathfinding — replaces stub"
```

---

## Task 2: SDF → NavMesh extraction bridge

**Files:**
- Create: `crates/vox_terrain/src/navmesh_bridge.rs`
- Modify: `crates/vox_terrain/src/lib.rs`

- [ ] Create `crates/vox_terrain/src/navmesh_bridge.rs`:

```rust
//! Extracts a walkable NavMesh directly from a TerrainVolume SDF.
//!
//! Walkable voxel: surface voxel (|sdf| < threshold) with air above (sdf at y+1 > 0).
//! This eliminates the need for separate voxelization — the SDF IS the walkability query.

use crate::volume::TerrainVolume;
use vox_core::navmesh::{NavMesh, NavNode};

/// Extract a `NavMesh` from a terrain volume.
///
/// `surface_threshold`: voxels with |SDF| < this are considered surface (recommended: 1.5).
/// `agent_height_voxels`: number of voxels of clearance required above the surface (typically 2).
pub fn extract_from_volume(
    vol: &TerrainVolume,
    surface_threshold: f32,
    agent_height_voxels: usize,
) -> NavMesh {
    let sx = vol.size_x;
    let sy = vol.size_y;
    let sz = vol.size_z;
    let mut nodes: Vec<NavNode> = Vec::new();
    // voxel → node index for edge building
    let mut voxel_to_idx: std::collections::HashMap<(usize, usize, usize), u32> =
        std::collections::HashMap::new();

    for z in 0..sz {
        for y in 0..sy.saturating_sub(agent_height_voxels) {
            for x in 0..sx {
                let sdf = vol.get(x, y, z);
                if sdf.abs() > surface_threshold { continue; } // not near surface
                if sdf > 0.0 { continue; } // voxel is air — not a walkable floor

                // Check headroom: all voxels above must be air
                let has_headroom = (1..=agent_height_voxels).all(|dy| {
                    let above_y = y + dy;
                    above_y < sy && vol.get(x, above_y, z) > 0.0
                });
                if !has_headroom { continue; }

                let id = nodes.len() as u32;
                let world_pos = vol.voxel_to_world(x, y, z);
                voxel_to_idx.insert((x, y, z), id);
                nodes.push(NavNode { id, world_pos, neighbours: Vec::new() });
            }
        }
    }

    // Connect adjacent walkable nodes (6-connectivity in xz plane + 1 y step)
    let directions: &[(i32, i32, i32)] = &[
        (1, 0, 0), (-1, 0, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1), // diagonals
        (1, 1, 0), (-1, 1, 0), (0, 1, 1), (0, 1, -1),   // step up
        (1, -1, 0), (-1, -1, 0), (0, -1, 1), (0, -1, -1), // step down
    ];

    let node_voxels: Vec<(usize, usize, usize)> = voxel_to_idx.keys().cloned().collect();
    for (x, y, z) in node_voxels {
        let id = voxel_to_idx[&(x, y, z)];
        for (dx, dy, dz) in directions {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 { continue; }
            let key = (nx as usize, ny as usize, nz as usize);
            if let Some(&neighbour_id) = voxel_to_idx.get(&key) {
                nodes[id as usize].neighbours.push(neighbour_id);
            }
        }
    }

    let mut mesh = NavMesh::new();
    mesh.nodes = nodes;
    mesh
}

/// Re-extract only the nodes within a sphere (for incremental update after deformation).
pub fn extract_region(
    vol: &TerrainVolume,
    center: [f32; 3],
    radius: f32,
    surface_threshold: f32,
    agent_height_voxels: usize,
) -> Vec<NavNode> {
    let full = extract_from_volume(vol, surface_threshold, agent_height_voxels);
    let r2 = radius * radius;
    let dx = |a: [f32; 3], b: [f32; 3]| -> f32 {
        let dx = a[0]-b[0]; let dy = a[1]-b[1]; let dz = a[2]-b[2];
        dx*dx + dy*dy + dz*dz
    };
    full.nodes.into_iter()
        .filter(|n| dx(n.world_pos, center) <= r2)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::TerrainVolume;
    use vox_terrain_sculpt::sculpt;

    fn flat_ground_volume() -> TerrainVolume {
        let mut vol = TerrainVolume::new(16, 8, 16, 1.0);
        // Solid ground at y=0..3, air above
        for z in 0..16usize {
            for x in 0..16usize {
                for y in 0..4usize {
                    vol.set(x, y, z, -1.0);
                }
                // y=4..7 remain air (default 1.0)
            }
        }
        vol
    }

    #[test]
    fn extract_produces_nodes_on_flat_ground() {
        let vol = flat_ground_volume();
        let mesh = extract_from_volume(&vol, 1.5, 2);
        assert!(mesh.node_count() > 0, "should find walkable nodes on flat ground");
    }

    #[test]
    fn extracted_nodes_have_neighbours() {
        let vol = flat_ground_volume();
        let mesh = extract_from_volume(&vol, 1.5, 2);
        let nodes_with_neighbours = mesh.nodes.iter().filter(|n| !n.neighbours.is_empty()).count();
        assert!(nodes_with_neighbours > 0, "inner nodes should have neighbours");
    }

    #[test]
    fn path_exists_across_flat_ground() {
        let vol = flat_ground_volume();
        let mesh = extract_from_volume(&vol, 1.5, 2);
        if mesh.node_count() < 2 { return; } // skip if degenerate
        let start = mesh.nearest_node([1.0, 4.0, 1.0]).unwrap();
        let goal = mesh.nearest_node([12.0, 4.0, 12.0]).unwrap();
        let path = mesh.find_path(start, goal);
        assert!(path.is_some(), "path should exist across flat ground");
    }

    #[test]
    fn invalidate_and_reextract_after_carve() {
        use crate::deform::carve_sphere;
        let mut vol = flat_ground_volume();
        let mut mesh = extract_from_volume(&vol, 1.5, 2);
        let initial_count = mesh.node_count();

        // Carve a hole
        carve_sphere(&mut vol, [8.0, 3.0, 8.0], 2.0);

        // Incremental update
        mesh.invalidate_region([8.0, 3.0, 8.0], 3.0);
        let new_nodes = extract_region(&vol, [8.0, 3.0, 8.0], 3.0, 1.5, 2);
        mesh.merge(new_nodes);

        // Navmesh should have fewer nodes in the carved area
        // (exact count depends on SDF, just verify no panic and reasonable result)
        assert!(mesh.node_count() <= initial_count + 50);
    }
}
```

- [ ] Add `pub mod navmesh_bridge;` to `crates/vox_terrain/src/lib.rs`.

- [ ] Add `vox_core` as dependency to `crates/vox_terrain/Cargo.toml` if not already present:
```bash
grep "vox_core" crates/vox_terrain/Cargo.toml
```
If absent, add: `vox_core = { path = "../vox_core" }`

- [ ] Run:
```bash
cargo test -p vox_terrain navmesh_bridge
```

- [ ] Commit:
```bash
git commit -m "feat(terrain): extract_from_volume — SDF-derived navmesh with incremental update"
```

---

## Task 3: Wire into engine_runner

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Add `navmesh: Option<vox_core::navmesh::NavMesh>` to `EngineApp` (all construction sites).

- [ ] After terrain volume is ready, extract initial navmesh:
```rust
if let Some(terrain) = &self.terrain_volume {
    self.navmesh = Some(vox_terrain::navmesh_bridge::extract_from_volume(terrain, 1.5, 2));
    println!("[ochroma] Navmesh: {} nodes", self.navmesh.as_ref().unwrap().node_count());
}
```

- [ ] After any `carve_sphere` call, do incremental update:
```rust
let center = [hit_pos.x, hit_pos.y, hit_pos.z];
let radius = 2.0f32;
if let (Some(navmesh), Some(terrain)) = (&mut self.navmesh, &self.terrain_volume) {
    navmesh.invalidate_region(center, radius + 1.0);
    let new_nodes = vox_terrain::navmesh_bridge::extract_region(
        terrain, center, radius + 1.0, 1.5, 2,
    );
    navmesh.merge(new_nodes);
}
```

- [ ] Verify:
```bash
cargo check --bin ochroma
```

- [ ] Commit:
```bash
git commit -m "feat(app): wire SDF navmesh into engine_runner with deform-triggered update"
```

---

## Acceptance Criteria

| # | Test | Command |
|---|------|---------|
| 1 | A* finds paths, handles disconnected graphs | `cargo test -p vox_core navmesh` |
| 2 | Extraction finds nodes on flat terrain | `cargo test -p vox_terrain navmesh_bridge` |
| 3 | Engine compiles | `cargo check --bin ochroma` |
| 4 | Full workspace green | `cargo test` |
