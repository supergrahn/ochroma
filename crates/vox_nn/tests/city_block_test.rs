use vox_nn::city_block::*;

#[test]
fn city_block_generates_all_elements() {
    let block = generate_city_block(42, 100.0);
    assert!(!block.road_splats.is_empty(), "Should have roads");
    assert!(!block.building_splats.is_empty(), "Should have buildings");
    assert!(!block.terrain_splats.is_empty(), "Should have terrain");
}

#[test]
fn city_block_is_deterministic() {
    let a = generate_city_block(42, 100.0);
    let b = generate_city_block(42, 100.0);
    assert_eq!(a.road_splats.len(), b.road_splats.len());
    assert_eq!(a.building_splats.len(), b.building_splats.len());
}

#[test]
fn city_block_total_splat_count() {
    let block = generate_city_block(42, 100.0);
    let total = block.road_splats.len()
        + block
            .building_splats
            .iter()
            .map(|(_, s)| s.len())
            .sum::<usize>()
        + block
            .tree_splats
            .iter()
            .map(|(_, s)| s.len())
            .sum::<usize>()
        + block
            .prop_splats
            .iter()
            .map(|(_, s)| s.len())
            .sum::<usize>()
        + block.terrain_splats.len();
    assert!(
        total > 10000,
        "A city block should have many splats, got {}",
        total
    );
    println!("Total splats in city block: {}", total);
}
