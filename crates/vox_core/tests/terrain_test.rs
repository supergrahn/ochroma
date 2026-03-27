use vox_core::terrain::{TerrainPlane, generate_terrain_splats};

#[test]
fn terrain_plane_has_dimensions() {
    let t = TerrainPlane::new(100.0, 100.0, 1.0);
    assert_eq!(t.width, 100.0);
}

#[test]
fn generate_splats_produces_correct_count() {
    let t = TerrainPlane::new(10.0, 10.0, 1.0);
    let splats = generate_terrain_splats(&t, "grass");
    assert!(!splats.is_empty());
    assert!(splats.len() > 50);
}

#[test]
fn all_terrain_splats_at_y_zero() {
    let t = TerrainPlane::new(10.0, 10.0, 1.0);
    let splats = generate_terrain_splats(&t, "asphalt_dry");
    for s in &splats {
        assert!((s.position[1]).abs() < 0.1);
    }
}
