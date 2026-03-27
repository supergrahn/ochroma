use vox_core::mapgen::generate_map;

#[test]
fn generates_splats() {
    let splats = generate_map(42, 100.0, 1.0);
    assert!(!splats.is_empty());
    assert!(splats.len() > 1000);
}

#[test]
fn has_height_variation() {
    let splats = generate_map(42, 200.0, 0.5);
    let min_y = splats.iter().map(|s| s.position[1]).fold(f32::MAX, f32::min);
    let max_y = splats.iter().map(|s| s.position[1]).fold(f32::MIN, f32::max);
    assert!(max_y - min_y > 1.0, "Expected height variation, got range {}", max_y - min_y);
}

#[test]
fn has_water_areas() {
    let splats = generate_map(42, 200.0, 0.5);
    let water_count = splats.iter().filter(|s| s.position[1] < 0.0).count();
    assert!(water_count > 0, "Expected river/water areas");
}

#[test]
fn deterministic() {
    let a = generate_map(42, 50.0, 1.0);
    let b = generate_map(42, 50.0, 1.0);
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(b.iter()) {
        assert_eq!(sa.position, sb.position);
    }
}
