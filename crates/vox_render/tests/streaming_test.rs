use vox_render::streaming::TileManager;
use vox_core::lwc::{TileCoord, TileState};

#[test]
fn test_initial_state_all_cold() {
    let mgr = TileManager::new();
    let tile = TileCoord { x: 0, z: 0 };
    assert_eq!(mgr.tile_state(tile), TileState::Cold);
    assert_eq!(mgr.active_tiles().len(), 0);
}

#[test]
fn test_camera_update_activates_nearby_tiles() {
    let mut mgr = TileManager::new();
    let camera = TileCoord { x: 0, z: 0 };
    mgr.update_camera(camera);

    // With radius=1, should activate a 3x3 grid = 9 tiles
    let active = mgr.active_tiles();
    assert_eq!(active.len(), 9, "Expected 9 active tiles, got {}", active.len());

    // Center tile should be active
    assert_eq!(mgr.tile_state(TileCoord { x: 0, z: 0 }), TileState::Active);
    assert_eq!(mgr.tile_state(TileCoord { x: 1, z: 0 }), TileState::Active);
    assert_eq!(mgr.tile_state(TileCoord { x: -1, z: -1 }), TileState::Active);
}

#[test]
fn test_far_tiles_stay_cold() {
    let mut mgr = TileManager::new();
    let camera = TileCoord { x: 0, z: 0 };
    mgr.update_camera(camera);

    // Tiles outside radius should remain cold
    assert_eq!(mgr.tile_state(TileCoord { x: 5, z: 5 }), TileState::Cold);
    assert_eq!(mgr.tile_state(TileCoord { x: -3, z: 0 }), TileState::Cold);
    assert_eq!(mgr.tile_state(TileCoord { x: 2, z: 2 }), TileState::Cold);
}

#[test]
fn test_camera_move_evicts_far_tiles() {
    let mut mgr = TileManager::new();
    mgr.update_camera(TileCoord { x: 0, z: 0 });
    // Move camera far away
    mgr.update_camera(TileCoord { x: 10, z: 10 });

    // Old tiles should be evicted (no longer tracked)
    assert_eq!(mgr.tile_state(TileCoord { x: 0, z: 0 }), TileState::Cold);
    // New camera area should be active
    assert_eq!(mgr.tile_state(TileCoord { x: 10, z: 10 }), TileState::Active);
}
