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
