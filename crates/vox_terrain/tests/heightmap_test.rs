use vox_terrain::heightmap::*;

#[test]
fn flat_terrain_constant_height() {
    let hm = Heightmap::flat(10, 10, 1.0, 5.0);
    assert!((hm.sample(5.0, 5.0) - 5.0).abs() < 0.01);
}

#[test]
fn bilinear_interpolation() {
    let mut data = vec![0.0f32; 4];
    data[0] = 0.0; // (0,0)
    data[1] = 10.0; // (1,0)
    data[2] = 0.0; // (0,1)
    data[3] = 10.0; // (1,1)
    let hm = Heightmap::from_data(2, 2, data, 1.0);
    let mid = hm.sample(0.5, 0.5);
    assert!((mid - 5.0).abs() < 0.1, "Midpoint should be ~5.0, got {}", mid);
}

#[test]
fn to_splats_produces_correct_count() {
    let hm = Heightmap::flat(10, 10, 1.0, 0.0);
    let zones = default_zones();
    let splats = hm.to_splats(&zones, 1);
    assert_eq!(splats.len(), 100); // 10x10 cells x 1 splat each
}

#[test]
fn material_zones_assign_by_height() {
    let mut data = vec![0.0f32; 4];
    data[0] = -1.0; // water
    data[1] = 0.0; // sand
    data[2] = 5.0; // grass
    data[3] = 20.0; // snow
    let hm = Heightmap::from_data(2, 2, data, 10.0);
    let zones = default_zones();
    let splats = hm.to_splats(&zones, 1);
    // Different heights should produce different spectral values
    assert_ne!(splats[0].spectral, splats[3].spectral);
}

#[test]
fn normal_points_up_on_flat() {
    let hm = Heightmap::flat(10, 10, 1.0, 0.0);
    let n = hm.normal_at(5.0, 5.0);
    assert!(n[1] > 0.99, "Normal on flat should point up: {:?}", n);
}

#[test]
fn slope_is_zero_on_flat() {
    let hm = Heightmap::flat(10, 10, 1.0, 0.0);
    assert!(
        hm.slope_at(5.0, 5.0) < 1.0,
        "Flat terrain should have ~0 slope"
    );
}

#[test]
fn test_heightmap_area() {
    let hm = Heightmap::flat(100, 100, 2.0, 0.0);
    assert!((hm.area() - 40000.0).abs() < 0.1); // 200m x 200m
}

#[test]
fn generate_test_heightmap_has_variation() {
    let hm = generate_test_heightmap(64, 64, 1.0, 42);
    let min = hm.data.iter().cloned().fold(f32::MAX, f32::min);
    let max = hm.data.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        max - min > 1.0,
        "Generated terrain should have height variation"
    );
}
