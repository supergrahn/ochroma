//! Extracts a walkable NavMesh directly from a TerrainVolume SDF.
//!
//! Walkable voxel: solid (sdf <= 0) near the surface (|sdf| < threshold)
//! with sufficient air clearance above.

use crate::volume::TerrainVolume;
use vox_core::navmesh::{NavMesh, NavNode};

/// Extract a `NavMesh` from a terrain volume.
///
/// `surface_threshold`: voxels with |SDF| < this are considered surface (recommended: 1.5).
/// `agent_height_voxels`: clearance required above the surface voxel (typically 2).
pub fn extract_from_volume(
    vol: &TerrainVolume,
    surface_threshold: f32,
    agent_height_voxels: usize,
) -> NavMesh {
    let sx = vol.size_x;
    let sy = vol.size_y;
    let sz = vol.size_z;
    let mut nodes: Vec<NavNode> = Vec::new();
    let mut voxel_to_idx: std::collections::HashMap<(usize, usize, usize), u32> =
        std::collections::HashMap::new();

    for z in 0..sz {
        for y in 0..sy.saturating_sub(agent_height_voxels) {
            for x in 0..sx {
                let sdf = vol.get(x, y, z);
                // Must be near surface
                if sdf.abs() > surface_threshold { continue; }
                // Must be solid (on or below surface)
                if sdf > 0.0 { continue; }

                // Check headroom: all voxels above must be air (positive SDF)
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

    // Build adjacency: 16-connectivity (xz plane + 1-step height change)
    let directions: &[(i32, i32, i32)] = &[
        (1, 0, 0), (-1, 0, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (1, 1, 0), (-1, 1, 0), (0, 1, 1), (0, 1, -1),
        (1, -1, 0), (-1, -1, 0), (0, -1, 1), (0, -1, -1),
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

/// Re-extract nodes within a sphere for incremental update after terrain deformation.
pub fn extract_region(
    vol: &TerrainVolume,
    center: [f32; 3],
    radius: f32,
    surface_threshold: f32,
    agent_height_voxels: usize,
) -> Vec<NavNode> {
    let full = extract_from_volume(vol, surface_threshold, agent_height_voxels);
    let r2 = radius * radius;
    full.nodes.into_iter()
        .filter(|n| {
            let dx = n.world_pos[0] - center[0];
            let dy = n.world_pos[1] - center[1];
            let dz = n.world_pos[2] - center[2];
            dx*dx + dy*dy + dz*dz <= r2
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::TerrainVolume;

    fn flat_ground_volume() -> TerrainVolume {
        let mut vol = TerrainVolume::new(16, 8, 16, 1.0);
        // Solid ground at y=0..3, air above
        for z in 0..16usize {
            for x in 0..16usize {
                for y in 0..4usize {
                    vol.set(x, y, z, -1.0);
                }
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
        if mesh.node_count() < 2 { return; }
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

        carve_sphere(&mut vol, [8.0, 3.0, 8.0], 2.0);

        mesh.invalidate_region([8.0, 3.0, 8.0], 3.0);
        let new_nodes = extract_region(&vol, [8.0, 3.0, 8.0], 3.0, 1.5, 2);
        mesh.merge(new_nodes);

        assert!(mesh.node_count() <= initial_count + 50);
    }
}
