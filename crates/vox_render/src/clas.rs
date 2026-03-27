use glam::Vec3;
use vox_core::types::GaussianSplat;

/// A cluster of spatially-nearby Gaussian splats.
#[derive(Debug, Clone)]
pub struct SplatCluster {
    pub id: u32,
    pub splat_indices: Vec<u32>,    // indices into global splat array
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub center: Vec3,               // centroid
    pub lod_level: u8,
    pub total_opacity: f32,         // sum of opacities (for LOD culling)
}

impl SplatCluster {
    pub fn splat_count(&self) -> usize { self.splat_indices.len() }

    pub fn aabb_size(&self) -> Vec3 { self.aabb_max - self.aabb_min }

    /// Test if a point is inside the cluster's AABB.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.aabb_min.x && point.x <= self.aabb_max.x
            && point.y >= self.aabb_min.y && point.y <= self.aabb_max.y
            && point.z >= self.aabb_min.z && point.z <= self.aabb_max.z
    }

    /// Test AABB-AABB intersection.
    pub fn intersects_aabb(&self, other_min: Vec3, other_max: Vec3) -> bool {
        self.aabb_min.x <= other_max.x && self.aabb_max.x >= other_min.x
            && self.aabb_min.y <= other_max.y && self.aabb_max.y >= other_min.y
            && self.aabb_min.z <= other_max.z && self.aabb_max.z >= other_min.z
    }

    /// Test ray-AABB intersection. Returns (hit, t_near, t_far).
    pub fn ray_intersect(&self, origin: Vec3, inv_dir: Vec3) -> (bool, f32, f32) {
        // Expand zero-thickness AABBs by a small epsilon for robust intersection
        let eps = Vec3::splat(1e-4);
        let aabb_min = self.aabb_min - eps;
        let aabb_max = self.aabb_max + eps;
        let t1 = (aabb_min - origin) * inv_dir;
        let t2 = (aabb_max - origin) * inv_dir;
        let t_min = t1.min(t2);
        let t_max = t1.max(t2);
        let t_near = t_min.x.max(t_min.y).max(t_min.z);
        let t_far = t_max.x.min(t_max.y).min(t_max.z);
        (t_near <= t_far && t_far >= 0.0, t_near, t_far)
    }
}

/// Build clusters from a splat array using spatial grid partitioning.
/// target_size: desired number of splats per cluster (64-256).
pub fn build_clusters(splats: &[GaussianSplat], target_size: usize) -> Vec<SplatCluster> {
    if splats.is_empty() { return Vec::new(); }

    // Compute scene AABB
    let mut scene_min = Vec3::splat(f32::MAX);
    let mut scene_max = Vec3::splat(f32::MIN);
    for s in splats {
        let p = Vec3::from(s.position);
        scene_min = scene_min.min(p);
        scene_max = scene_max.max(p);
    }

    let scene_size = scene_max - scene_min;

    // Count non-degenerate axes for proper dimensionality handling
    let non_degenerate = (if scene_size.x > 1e-6 { 1 } else { 0 })
        + (if scene_size.y > 1e-6 { 1 } else { 0 })
        + (if scene_size.z > 1e-6 { 1 } else { 0 });

    if non_degenerate == 0 {
        // All splats at same point — one cluster
        return vec![make_cluster(0, splats, &(0..splats.len() as u32).collect::<Vec<_>>())];
    }

    // Grid cell size: target cluster_count = splats.len() / target_size
    let cluster_count = (splats.len() / target_size).max(1) as f32;
    // Use dimensionality-aware root for cells per axis
    let cells_per_axis = match non_degenerate {
        1 => cluster_count.ceil() as usize,
        2 => cluster_count.sqrt().ceil() as usize,
        _ => cluster_count.cbrt().ceil() as usize,
    }.max(1);
    let cell_size = Vec3::new(
        if scene_size.x > 1e-6 { scene_size.x / cells_per_axis as f32 } else { 1.0 },
        if scene_size.y > 1e-6 { scene_size.y / cells_per_axis as f32 } else { 1.0 },
        if scene_size.z > 1e-6 { scene_size.z / cells_per_axis as f32 } else { 1.0 },
    );

    // Assign splats to grid cells
    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<u32>> = std::collections::HashMap::new();

    for (i, s) in splats.iter().enumerate() {
        let p = Vec3::from(s.position) - scene_min;
        let cx = (p.x / cell_size.x.max(0.001)).floor() as i32;
        let cy = (p.y / cell_size.y.max(0.001)).floor() as i32;
        let cz = (p.z / cell_size.z.max(0.001)).floor() as i32;
        grid.entry((cx, cy, cz)).or_default().push(i as u32);
    }

    // Convert cells to clusters (split large cells, merge small ones)
    let mut clusters = Vec::new();
    let mut cluster_id = 0u32;

    for (_cell, indices) in &grid {
        if indices.len() <= target_size * 2 {
            // Small enough — one cluster
            clusters.push(make_cluster(cluster_id, splats, indices));
            cluster_id += 1;
        } else {
            // Too large — split into sub-clusters
            for chunk in indices.chunks(target_size) {
                clusters.push(make_cluster(cluster_id, splats, &chunk.to_vec()));
                cluster_id += 1;
            }
        }
    }

    clusters
}

fn make_cluster(id: u32, splats: &[GaussianSplat], indices: &[u32]) -> SplatCluster {
    let mut aabb_min = Vec3::splat(f32::MAX);
    let mut aabb_max = Vec3::splat(f32::MIN);
    let mut center_sum = Vec3::ZERO;
    let mut total_opacity = 0.0f32;

    for &idx in indices {
        let s = &splats[idx as usize];
        let p = Vec3::from(s.position);
        aabb_min = aabb_min.min(p);
        aabb_max = aabb_max.max(p);
        center_sum += p;
        total_opacity += s.opacity as f32 / 255.0;
    }

    let center = if indices.is_empty() { Vec3::ZERO } else { center_sum / indices.len() as f32 };

    SplatCluster {
        id,
        splat_indices: indices.to_vec(),
        aabb_min,
        aabb_max,
        center,
        lod_level: 0,
        total_opacity,
    }
}

/// BVH node over clusters.
#[derive(Debug, Clone)]
pub enum ClusterBVHNode {
    Leaf { cluster_id: u32 },
    Internal { aabb_min: Vec3, aabb_max: Vec3, left: Box<ClusterBVHNode>, right: Box<ClusterBVHNode> },
}

/// Build a BVH over clusters using median split.
pub fn build_cluster_bvh(clusters: &[SplatCluster]) -> Option<ClusterBVHNode> {
    if clusters.is_empty() { return None; }
    if clusters.len() == 1 { return Some(ClusterBVHNode::Leaf { cluster_id: clusters[0].id }); }

    // Find overall AABB
    let mut aabb_min = Vec3::splat(f32::MAX);
    let mut aabb_max = Vec3::splat(f32::MIN);
    for c in clusters {
        aabb_min = aabb_min.min(c.aabb_min);
        aabb_max = aabb_max.max(c.aabb_max);
    }

    // Split along longest axis
    let size = aabb_max - aabb_min;
    let axis = if size.x >= size.y && size.x >= size.z { 0 }
        else if size.y >= size.z { 1 } else { 2 };

    let mut sorted: Vec<&SplatCluster> = clusters.iter().collect();
    sorted.sort_by(|a, b| {
        let ca = match axis { 0 => a.center.x, 1 => a.center.y, _ => a.center.z };
        let cb = match axis { 0 => b.center.x, 1 => b.center.y, _ => b.center.z };
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = sorted.len() / 2;
    let left_clusters: Vec<SplatCluster> = sorted[..mid].iter().map(|c| (*c).clone()).collect();
    let right_clusters: Vec<SplatCluster> = sorted[mid..].iter().map(|c| (*c).clone()).collect();

    let left = build_cluster_bvh(&left_clusters);
    let right = build_cluster_bvh(&right_clusters);

    match (left, right) {
        (Some(l), Some(r)) => Some(ClusterBVHNode::Internal {
            aabb_min, aabb_max, left: Box::new(l), right: Box::new(r),
        }),
        (Some(n), None) | (None, Some(n)) => Some(n),
        (None, None) => None,
    }
}

/// Statistics about clustering.
#[derive(Debug)]
pub struct ClusterStats {
    pub total_splats: usize,
    pub cluster_count: usize,
    pub avg_splats_per_cluster: f32,
    pub min_splats_per_cluster: usize,
    pub max_splats_per_cluster: usize,
    pub bvh_depth: u32,
}

pub fn compute_stats(clusters: &[SplatCluster], bvh: &Option<ClusterBVHNode>) -> ClusterStats {
    let total: usize = clusters.iter().map(|c| c.splat_count()).sum();
    let min = clusters.iter().map(|c| c.splat_count()).min().unwrap_or(0);
    let max = clusters.iter().map(|c| c.splat_count()).max().unwrap_or(0);

    fn bvh_depth(node: &ClusterBVHNode) -> u32 {
        match node {
            ClusterBVHNode::Leaf { .. } => 1,
            ClusterBVHNode::Internal { left, right, .. } => {
                1 + bvh_depth(left).max(bvh_depth(right))
            }
        }
    }

    ClusterStats {
        total_splats: total,
        cluster_count: clusters.len(),
        avg_splats_per_cluster: if clusters.is_empty() { 0.0 } else { total as f32 / clusters.len() as f32 },
        min_splats_per_cluster: min,
        max_splats_per_cluster: max,
        bvh_depth: bvh.as_ref().map(|b| bvh_depth(b)).unwrap_or(0),
    }
}
