use vox_core::types::GaussianSplat;

pub struct ShadowCatcherMesh {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// Andrew's monotone chain algorithm for 2D convex hull.
/// Returns hull vertices in counter-clockwise order.
pub fn generate_convex_hull_2d(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));

    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    // Remove last point of each half (duplicate of first point of the other half)
    lower.pop();
    upper.pop();

    lower.extend_from_slice(&upper);
    lower
}

/// Generate a shadow catcher mesh from a set of Gaussian splats.
/// Projects splat positions to the ground plane (y=0) and builds a convex hull mesh.
pub fn generate_shadow_catcher(splats: &[GaussianSplat]) -> ShadowCatcherMesh {
    if splats.is_empty() {
        return ShadowCatcherMesh {
            vertices: vec![],
            indices: vec![],
        };
    }

    // Project to ground plane: use x and z
    let points_2d: Vec<[f32; 2]> = splats
        .iter()
        .map(|s| [s.position[0], s.position[2]])
        .collect();

    let hull = generate_convex_hull_2d(&points_2d);

    if hull.len() < 3 {
        return ShadowCatcherMesh {
            vertices: vec![],
            indices: vec![],
        };
    }

    // Create 3D vertices at y=0
    let vertices: Vec<[f32; 3]> = hull.iter().map(|p| [p[0], 0.0, p[1]]).collect();

    // Fan triangulate from first vertex
    let mut indices: Vec<u32> = Vec::new();
    for i in 1..(hull.len() as u32 - 1) {
        indices.push(0);
        indices.push(i);
        indices.push(i + 1);
    }

    ShadowCatcherMesh { vertices, indices }
}
