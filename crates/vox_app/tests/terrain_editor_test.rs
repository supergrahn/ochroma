use vox_app::terrain_editor::{TerrainEditorState, ActiveBrush};
use vox_terrain::brushes::BrushType;

#[test]
fn default_state_has_raise_brush() {
    let state = TerrainEditorState::default();
    assert!(matches!(state.brush.brush_type, BrushType::Raise));
    assert!((state.brush.radius - 5.0).abs() < f32::EPSILON);
    assert!((state.brush.strength - 0.5).abs() < f32::EPSILON);
}

#[test]
fn set_brush_type_updates_brush() {
    let mut state = TerrainEditorState::default();
    state.set_brush_type(BrushType::Flatten { target_height: 2.0 });
    assert!(matches!(state.brush.brush_type, BrushType::Flatten { .. }));
}

#[test]
fn active_brush_matches_enum() {
    let mut state = TerrainEditorState::default();
    assert_eq!(state.active_brush, ActiveBrush::Raise);
    state.active_brush = ActiveBrush::Lower;
    state.sync_brush();
    assert!(matches!(state.brush.brush_type, BrushType::Lower));
}

use bevy_ecs::prelude::*;
use vox_app::terrain_editor::apply_brush_stroke;
use vox_terrain::volume::TerrainVolume;
use glam::Vec3;

#[test]
fn apply_brush_stroke_does_not_panic() {
    let mut world = World::new();
    let vol = vox_terrain::volume::generate_demo_volume(42);
    world.insert_resource(vol);

    let center = Vec3::new(4.0, 2.0, 4.0);
    apply_brush_stroke(&mut world, center, BrushType::Raise, 3.0, 1.0, 0.016);
    // If we get here without panic, test passes
    let vol = world.resource::<TerrainVolume>();
    assert!(vol.solid_count() > 0);
}

use vox_app::terrain_editor::resplat_terrain;
use vox_core::ecs::SplatAssetComponent;
use uuid::Uuid;
use vox_terrain::volume::generate_demo_volume;

#[test]
fn resplat_updates_splat_asset() {
    let mut world = World::new();
    world.insert_resource(generate_demo_volume(42));

    let uuid = Uuid::new_v4();
    let entity = world.spawn(SplatAssetComponent {
        uuid,
        splat_count: 0,
        splats: vec![],
    }).id();

    resplat_terrain(&mut world, entity);

    let asset = world.entity(entity).get::<SplatAssetComponent>().unwrap();
    assert!(asset.splat_count > 0, "resplat should produce splats from the volume");
    assert_eq!(asset.splats.len(), asset.splat_count as usize);
}

#[test]
fn foliage_scatter_flag_round_trip() {
    let mut state = TerrainEditorState::default();
    assert!(!state.foliage_scatter_pending);
    state.foliage_scatter_pending = true;
    assert!(state.foliage_scatter_pending);
    state.foliage_scatter_pending = false;
    assert!(!state.foliage_scatter_pending);
}

#[test]
fn sync_brush_flatten_uses_flatten_height() {
    let mut state = TerrainEditorState::default();
    state.active_brush = ActiveBrush::Flatten;
    state.flatten_height = 3.5;
    state.sync_brush();
    if let BrushType::Flatten { target_height } = state.brush.brush_type {
        assert!((target_height - 3.5).abs() < f32::EPSILON);
    } else {
        panic!("expected Flatten brush");
    }
}
