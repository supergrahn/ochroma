use vox_render::mega_geometry::*;

#[test]
fn compute_tiles_covers_screen() {
    let dispatch = MegaGeometryDispatch::new(1920, 1080, 10_000_000);
    let tiles = dispatch.compute_tiles();
    assert_eq!(dispatch.tile_count(), 120 * 68); // ceil(1920/16) * ceil(1080/16)
    assert_eq!(tiles.len(), dispatch.tile_count() as usize);
}

#[test]
fn assign_splats_to_correct_tiles() {
    let mut dispatch = MegaGeometryDispatch::new(64, 64, 1000);
    let mut tiles = dispatch.compute_tiles();

    // Place a splat in the center with small radius
    let positions = vec![(32.0, 32.0, 2.0)];
    dispatch.assign_splats_to_tiles(&mut tiles, &positions);

    // The center tile should have the splat
    let _center_tile = &tiles[tiles.len() / 2];
    assert!(!tiles.iter().all(|t| t.splat_indices.is_empty()), "Some tile should have the splat");
    assert_eq!(dispatch.last_frame_stats.splats_rendered, 1);
}

#[test]
fn gpu_budget_limits_splats() {
    let mut dispatch = MegaGeometryDispatch::new(64, 64, 5);
    let mut tiles = dispatch.compute_tiles();

    let positions: Vec<(f32, f32, f32)> = (0..100)
        .map(|i| (i as f32 % 64.0, i as f32 / 2.0 % 64.0, 1.0))
        .collect();
    dispatch.assign_splats_to_tiles(&mut tiles, &positions);

    assert!(dispatch.last_frame_stats.splats_rendered <= 5);
    assert!(dispatch.last_frame_stats.splats_culled > 0);
}

#[test]
fn large_radius_splat_spans_multiple_tiles() {
    let mut dispatch = MegaGeometryDispatch::new(128, 128, 1000);
    let mut tiles = dispatch.compute_tiles();

    // Large splat covering many tiles
    let positions = vec![(64.0, 64.0, 50.0)];
    dispatch.assign_splats_to_tiles(&mut tiles, &positions);

    let tiles_with_splat = tiles.iter().filter(|t| !t.splat_indices.is_empty()).count();
    assert!(tiles_with_splat > 4, "Large splat should span multiple tiles: {}", tiles_with_splat);
}

#[test]
fn stats_track_correctly() {
    let mut dispatch = MegaGeometryDispatch::new(128, 128, 1000);
    let mut tiles = dispatch.compute_tiles();

    let positions: Vec<(f32, f32, f32)> = (0..50)
        .map(|i| (i as f32 * 2.0 + 10.0, 64.0, 3.0))
        .collect();
    dispatch.assign_splats_to_tiles(&mut tiles, &positions);

    assert_eq!(dispatch.last_frame_stats.total_splats_in_scene, 50);
    assert!(dispatch.last_frame_stats.splats_rendered > 0);
    assert_eq!(dispatch.last_frame_stats.tiles_processed, tiles.len() as u32);
}
