use vox_core::lwc::{TileCoord, WorldCoord, TILE_SIZE};

#[test]
fn test_from_absolute_and_to_absolute_round_trip() {
    let x = 1234.567;
    let y = 42.0;
    let z = -789.1;
    let wc = WorldCoord::from_absolute(x, y, z);
    let (rx, ry, rz) = wc.to_absolute();
    assert!((rx - x).abs() < 1e-4, "x round-trip failed: {} vs {}", rx, x);
    assert!((ry - y).abs() < 1e-4, "y round-trip failed: {} vs {}", ry, y);
    assert!((rz - z).abs() < 1e-4, "z round-trip failed: {} vs {}", rz, z);
}

#[test]
fn test_tile_anchor_calculation() {
    let tile = TileCoord { x: 3, z: -2 };
    let (ax, az) = tile.anchor();
    assert_eq!(ax, 3.0 * TILE_SIZE);
    assert_eq!(az, -2.0 * TILE_SIZE);
}

#[test]
fn test_tile_assignment() {
    let wc = WorldCoord::from_absolute(2500.0, 0.0, 1500.0);
    assert_eq!(wc.tile.x, 2);
    assert_eq!(wc.tile.z, 1);
}

#[test]
fn test_sub_mm_precision_at_50km() {
    // 50km = 50_000m
    let x = 50_000.123_456_789;
    let y = 100.0;
    let z = 49_999.987_654_321;
    let wc = WorldCoord::from_absolute(x, y, z);
    let (rx, ry, rz) = wc.to_absolute();
    // sub-mm precision = < 1e-3
    assert!((rx - x).abs() < 1e-3, "x precision at 50km failed: diff={}", (rx - x).abs());
    assert!((ry - y).abs() < 1e-3, "y precision at 50km failed: diff={}", (ry - y).abs());
    assert!((rz - z).abs() < 1e-3, "z precision at 50km failed: diff={}", (rz - z).abs());
}
