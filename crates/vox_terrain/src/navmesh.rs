//! NavMesh generation from TerrainVolume SDF.

use glam::Vec3;
use navmesh::{NavMesh, NavQuery, NavPathMode, NavTriangle, NavVec3};

use crate::volume::TerrainVolume;

#[derive(Debug, Clone, Copy)]
pub struct SurfacePoint {
    pub position: Vec3,
    pub normal_y: f32,
}

pub fn sample_walkable_surface(
    vol: &TerrainVolume,
    step: f32,
    max_slope_angle: f32,
) -> Vec<SurfacePoint> {
    let mut points = Vec::new();
    let cos_max = max_slope_angle.to_radians().cos();

    let world_min_x = vol.origin[0];
    let world_min_z = vol.origin[2];
    let world_max_x = vol.origin[0] + vol.size_x as f32 * vol.voxel_size;
    let world_max_z = vol.origin[2] + vol.size_z as f32 * vol.voxel_size;
    let world_max_y = vol.origin[1] + vol.size_y as f32 * vol.voxel_size;
    let world_min_y = vol.origin[1];

    let mut wx = world_min_x;
    while wx < world_max_x {
        let mut wz = world_min_z;
        while wz < world_max_z {
            let mut wy = world_max_y;
            let y_step = vol.voxel_size;
            let mut prev_sdf = vol.sample_world(wx, wy, wz);

            while wy > world_min_y {
                wy -= y_step;
                let sdf = vol.sample_world(wx, wy, wz);

                if prev_sdf > 0.0 && sdf <= 0.0 {
                    let t = prev_sdf / (prev_sdf - sdf);
                    let surface_y = (wy + y_step) - t * y_step;

                    let (vx, vy, vz) = vol.world_to_voxel(wx, surface_y, wz);
                    let grad = vol.gradient(vx, vy, vz);
                    let normal_y = grad[1];

                    if normal_y >= cos_max {
                        points.push(SurfacePoint {
                            position: Vec3::new(wx, surface_y, wz),
                            normal_y,
                        });
                    }
                    break;
                }

                prev_sdf = sdf;
            }

            wz += step;
        }
        wx += step;
    }

    points
}

#[derive(Debug, Clone)]
pub struct NavMeshConfig {
    pub sample_step: f32,
    pub agent_radius: f32,
    pub max_slope_angle: f32,
}

impl Default for NavMeshConfig {
    fn default() -> Self {
        Self { sample_step: 2.0, agent_radius: 0.5, max_slope_angle: 45.0 }
    }
}

pub fn build_navmesh(vol: &TerrainVolume, config: &NavMeshConfig) -> Option<NavMesh> {
    let points = sample_walkable_surface(vol, config.sample_step, config.max_slope_angle);
    if points.len() < 3 { return None; }

    let step = config.sample_step;
    let mut grid: std::collections::HashMap<(i32, i32), usize> = std::collections::HashMap::new();
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(points.len());

    for (idx, p) in points.iter().enumerate() {
        let gx = (p.position.x / step).floor() as i32;
        let gz = (p.position.z / step).floor() as i32;
        grid.insert((gx, gz), idx);
        vertices.push([p.position.x, p.position.y, p.position.z]);
    }

    let mut triangles: Vec<NavTriangle> = Vec::new();
    for (&(gx, gz), &idx) in &grid {
        let right = grid.get(&(gx + 1, gz)).copied();
        let below = grid.get(&(gx, gz + 1)).copied();
        let diag = grid.get(&(gx + 1, gz + 1)).copied();

        if let (Some(r), Some(b)) = (right, below) {
            let dy1 = (vertices[idx][1] - vertices[r][1]).abs();
            let dy2 = (vertices[idx][1] - vertices[b][1]).abs();
            if dy1 < step * 2.0 && dy2 < step * 2.0 {
                triangles.push(NavTriangle::from([idx as u32, r as u32, b as u32]));
            }
        }

        if let (Some(r), Some(b), Some(d)) = (right, below, diag) {
            let dy1 = (vertices[r][1] - vertices[d][1]).abs();
            let dy2 = (vertices[b][1] - vertices[d][1]).abs();
            if dy1 < step * 2.0 && dy2 < step * 2.0 {
                triangles.push(NavTriangle::from([r as u32, d as u32, b as u32]));
            }
        }
    }

    if triangles.is_empty() { return None; }

    let nav_vertices: Vec<NavVec3> = vertices
        .iter()
        .map(|v| NavVec3::new(v[0], v[1], v[2]))
        .collect();

    NavMesh::new(nav_vertices, triangles).ok()
}

#[derive(Debug, Clone)]
pub struct NavPath {
    pub waypoints: Vec<Vec3>,
}

pub fn find_path(mesh: &NavMesh, start: Vec3, end: Vec3) -> Option<NavPath> {
    let from = NavVec3::new(start.x, start.y, start.z);
    let to = NavVec3::new(end.x, end.y, end.z);

    match mesh.find_path(from, to, NavQuery::Accuracy, NavPathMode::MidPoints) {
        Some(path) => {
            let waypoints: Vec<Vec3> = path.iter().map(|p| Vec3::new(p.x, p.y, p.z)).collect();
            if waypoints.is_empty() { None } else { Some(NavPath { waypoints }) }
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::TerrainVolume;

    #[test]
    fn sample_walkable_surface_empty_volume_returns_empty() {
        let vol = TerrainVolume::new(8, 8, 8, 1.0); // all air
        let points = sample_walkable_surface(&vol, 1.0, 45.0);
        assert!(points.is_empty(), "All-air volume should have no walkable surface");
    }

    #[test]
    fn build_navmesh_empty_volume_returns_none() {
        let vol = TerrainVolume::new(8, 8, 8, 1.0);
        let mesh = build_navmesh(&vol, &NavMeshConfig::default());
        assert!(mesh.is_none());
    }

    #[test]
    fn navmesh_config_default_is_reasonable() {
        let config = NavMeshConfig::default();
        assert!(config.sample_step > 0.0);
        assert!(config.agent_radius > 0.0);
        assert!(config.max_slope_angle > 0.0 && config.max_slope_angle <= 90.0);
    }
}
