use vox_data::proc_gs_advanced::*;

#[test]
fn tree_has_trunk_and_canopy() {
    let splats = generate_tree(42, 8.0, 3.0);
    assert!(
        splats.len() > 500,
        "Tree should have many splats, got {}",
        splats.len()
    );
    // Check there are splats at different heights
    let max_y = splats
        .iter()
        .map(|s| s.position[1])
        .fold(f32::MIN, f32::max);
    assert!(max_y > 5.0, "Tree should be tall");
}

#[test]
fn tree_is_deterministic() {
    let a = generate_tree(42, 8.0, 3.0);
    let b = generate_tree(42, 8.0, 3.0);
    assert_eq!(a.len(), b.len());
}

#[test]
fn different_seeds_different_trees() {
    let a = generate_tree(42, 8.0, 3.0);
    let b = generate_tree(99, 8.0, 3.0);
    // Different seeds produce different trees (positions or counts differ)
    let differs = a.len() != b.len()
        || a.iter()
            .zip(b.iter())
            .any(|(sa, sb)| sa.position != sb.position);
    assert!(differs, "Different seeds should produce different trees");
}

#[test]
fn bench_has_seat_and_back() {
    let splats = generate_bench(42);
    assert!(
        splats.len() > 50,
        "Bench needs splats for frame + slats"
    );
    let max_y = splats
        .iter()
        .map(|s| s.position[1])
        .fold(f32::MIN, f32::max);
    assert!(
        max_y > 0.5,
        "Bench should have a back rest above seat height"
    );
}

#[test]
fn grass_patch_density_scales() {
    let sparse = generate_grass_patch(42, 10.0, 5.0);
    let dense = generate_grass_patch(42, 10.0, 50.0);
    assert!(
        dense.len() > sparse.len() * 5,
        "Higher density should produce more splats"
    );
}

#[test]
fn lamp_post_reaches_height() {
    let splats = generate_lamp_post(42, 5.0);
    let max_y = splats
        .iter()
        .map(|s| s.position[1])
        .fold(f32::MIN, f32::max);
    assert!(max_y > 4.5, "Lamp should reach near specified height");
}
