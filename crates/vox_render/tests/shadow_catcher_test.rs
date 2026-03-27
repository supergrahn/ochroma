use vox_render::gpu::shadow_catcher::generate_convex_hull_2d;

#[test]
fn convex_hull_of_square() {
    let points = vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
    let hull = generate_convex_hull_2d(&points);
    assert_eq!(hull.len(), 4);
}

#[test]
fn convex_hull_of_triangle() {
    let points = vec![[0.0f32, 0.0], [1.0, 0.0], [0.5, 1.0]];
    let hull = generate_convex_hull_2d(&points);
    assert_eq!(hull.len(), 3);
}

#[test]
fn shadow_mesh_from_splat_positions() {
    use vox_core::types::GaussianSplat;
    use vox_render::gpu::shadow_catcher::generate_shadow_catcher;

    let splats: Vec<GaussianSplat> = (0..100)
        .map(|i| {
            let angle = i as f32 * 0.1;
            let radius = 5.0;
            GaussianSplat {
                position: [angle.cos() * radius, i as f32 * 0.1, angle.sin() * radius],
                scale: [0.1, 0.1, 0.1],
                rotation: [0, 0, 0, 32767],
                opacity: 255,
                _pad: [0; 3],
                spectral: [0; 8],
            }
        })
        .collect();

    let mesh = generate_shadow_catcher(&splats);
    assert!(mesh.vertices.len() >= 3);
    assert!(!mesh.indices.is_empty());
    for v in &mesh.vertices {
        assert!((v[1]).abs() < 0.01, "Shadow catcher verts should be on ground");
    }
}
