use vox_nn::multi_tile::expand_to_multi_tile;
use vox_nn::layout::{LayoutInterpreter, StubLayoutInterpreter};

#[test]
fn expand_2x2_produces_4_tiles() {
    let interp = StubLayoutInterpreter;
    let scene = interp.interpret("test").unwrap();
    let tiles = expand_to_multi_tile(&scene, 2, 2);
    assert_eq!(tiles.len(), 4);
}

#[test]
fn each_tile_has_offset_positions() {
    let interp = StubLayoutInterpreter;
    let scene = interp.interpret("test").unwrap();
    let tiles = expand_to_multi_tile(&scene, 2, 1);

    let first_building_x_0 = tiles[0].1.street.buildings[0].position[0];
    let first_building_x_1 = tiles[1].1.street.buildings[0].position[0];
    // Second tile should be offset by TILE_SIZE (1000m)
    assert!(
        (first_building_x_1 - first_building_x_0 - 1000.0).abs() < 1.0,
        "Expected 1000m offset, got {}",
        first_building_x_1 - first_building_x_0
    );
}

#[test]
fn expand_1x1_produces_single_tile() {
    let interp = StubLayoutInterpreter;
    let scene = interp.interpret("test").unwrap();
    let tiles = expand_to_multi_tile(&scene, 1, 1);
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0].0.x, 0);
    assert_eq!(tiles[0].0.z, 0);
}

#[test]
fn tile_coords_are_correct() {
    let interp = StubLayoutInterpreter;
    let scene = interp.interpret("test").unwrap();
    let tiles = expand_to_multi_tile(&scene, 2, 2);

    let coords: Vec<(i32, i32)> = tiles.iter().map(|(tc, _)| (tc.x, tc.z)).collect();
    assert!(coords.contains(&(0, 0)));
    assert!(coords.contains(&(0, 1)));
    assert!(coords.contains(&(1, 0)));
    assert!(coords.contains(&(1, 1)));
}

#[test]
fn z_offset_applied_correctly() {
    let interp = StubLayoutInterpreter;
    let scene = interp.interpret("test").unwrap();
    let tiles = expand_to_multi_tile(&scene, 1, 2);

    let first_building_z_0 = tiles[0].1.street.buildings[0].position[2];
    let first_building_z_1 = tiles[1].1.street.buildings[0].position[2];
    // Second tile along Z should be offset by TILE_SIZE (1000m)
    assert!(
        (first_building_z_1 - first_building_z_0 - 1000.0).abs() < 1.0,
        "Expected 1000m Z offset, got {}",
        first_building_z_1 - first_building_z_0
    );
}
