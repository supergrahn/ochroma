use vox_render::clas::*;
use vox_core::types::GaussianSplat;
use glam::Vec3;
use half::f16;

fn make_splat(pos: [f32; 3]) -> GaussianSplat {
    GaussianSplat {
        position: pos, scale: [0.1, 0.1, 0.1], rotation: [0, 0, 0, 32767],
        opacity: 200, _pad: [0; 3], spectral: [f16::from_f32(0.5).to_bits(); 8],
    }
}

#[test]
fn cluster_100_splats() {
    let splats: Vec<GaussianSplat> = (0..100)
        .map(|i| make_splat([i as f32, 0.0, 0.0]))
        .collect();
    let clusters = build_clusters(&splats, 32);
    assert!(!clusters.is_empty());
    let total: usize = clusters.iter().map(|c| c.splat_count()).sum();
    assert_eq!(total, 100, "All splats should be in clusters");
}

#[test]
fn cluster_respects_target_size() {
    let splats: Vec<GaussianSplat> = (0..1000)
        .map(|i| make_splat([(i % 50) as f32, (i / 50) as f32, 0.0]))
        .collect();
    let clusters = build_clusters(&splats, 64);
    for c in &clusters {
        assert!(c.splat_count() <= 128, "Cluster too large: {}", c.splat_count()); // allow 2x target
    }
}

#[test]
fn cluster_aabb_contains_all_splats() {
    let splats: Vec<GaussianSplat> = (0..50)
        .map(|i| make_splat([i as f32 * 0.1, (i as f32 * 0.7).sin(), 0.0]))
        .collect();
    let clusters = build_clusters(&splats, 50);
    for c in &clusters {
        for &idx in &c.splat_indices {
            let p = Vec3::from(splats[idx as usize].position);
            assert!(c.contains_point(p), "Splat {} should be inside cluster AABB", idx);
        }
    }
}

#[test]
fn ray_intersect_cluster() {
    let splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([1.0, 0.0, 0.0])];
    let clusters = build_clusters(&splats, 64);
    let cluster = &clusters[0];

    let origin = Vec3::new(-5.0, 0.0, 0.0);
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let inv_dir = Vec3::new(1.0 / dir.x, 1.0 / dir.y.max(1e-8), 1.0 / dir.z.max(1e-8));
    let (hit, _, _) = cluster.ray_intersect(origin, inv_dir);
    assert!(hit, "Ray should hit cluster");

    let miss_origin = Vec3::new(0.0, 100.0, 0.0);
    let miss_dir = Vec3::new(1.0, 0.0, 0.0);
    let miss_inv = Vec3::new(1.0 / miss_dir.x, 1.0 / miss_dir.y.max(1e-8), 1.0 / miss_dir.z.max(1e-8));
    let (hit2, _, _) = cluster.ray_intersect(miss_origin, miss_inv);
    assert!(!hit2, "Ray should miss cluster");
}

#[test]
fn build_bvh_over_clusters() {
    let splats: Vec<GaussianSplat> = (0..500)
        .map(|i| make_splat([(i % 50) as f32, (i / 50) as f32, 0.0]))
        .collect();
    let clusters = build_clusters(&splats, 32);
    let bvh = build_cluster_bvh(&clusters);
    assert!(bvh.is_some(), "BVH should be built");

    let stats = compute_stats(&clusters, &bvh);
    println!("CLAS stats: {:?}", stats);
    assert_eq!(stats.total_splats, 500);
    assert!(stats.bvh_depth > 1, "BVH should have depth > 1");
}

#[test]
fn large_scene_clustering() {
    let splats: Vec<GaussianSplat> = (0..100_000)
        .map(|i| make_splat([
            (i % 100) as f32,
            ((i / 100) % 100) as f32,
            (i / 10000) as f32,
        ]))
        .collect();

    let start = std::time::Instant::now();
    let clusters = build_clusters(&splats, 128);
    let cluster_time = start.elapsed();

    let start2 = std::time::Instant::now();
    let bvh = build_cluster_bvh(&clusters);
    let bvh_time = start2.elapsed();

    let stats = compute_stats(&clusters, &bvh);
    println!("100k splats: {} clusters, BVH depth {}, cluster time {:?}, BVH time {:?}",
        stats.cluster_count, stats.bvh_depth, cluster_time, bvh_time);

    assert!(stats.cluster_count > 100);
    assert!(cluster_time.as_millis() < 5000, "Clustering should be fast");
}
