use vox_data::creator_tools::*;

#[test]
fn test_brush_stroke_creation() {
    let brush = BrushStroke::new([5.0, 0.0, 5.0], 3.0, "dirt", 0.6);
    assert_eq!(brush.position, [5.0, 0.0, 5.0]);
    assert!((brush.radius - 3.0).abs() < f32::EPSILON);
    assert_eq!(brush.material, "dirt");
    assert!((brush.pressure - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_brush_stroke_pressure_clamping() {
    let brush = BrushStroke::new([0.0; 3], 1.0, "stone", 5.0);
    assert!((brush.pressure - 1.0).abs() < f32::EPSILON);

    let brush2 = BrushStroke::new([0.0; 3], 1.0, "stone", -2.0);
    assert!((brush2.pressure - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_brush_contains_point() {
    let brush = BrushStroke::new([0.0, 0.0, 0.0], 10.0, "grass", 1.0);
    assert!(brush.contains_point([5.0, 0.0, 0.0]));
    assert!(brush.contains_point([0.0, 0.0, 0.0]));
    assert!(!brush.contains_point([20.0, 0.0, 0.0]));
}

#[test]
fn test_brush_area_of_influence() {
    let brush = BrushStroke::new([0.0; 3], 5.0, "rock", 1.0);
    let expected = std::f32::consts::PI * 25.0;
    assert!((brush.area_of_influence() - expected).abs() < 0.01);
}

#[test]
fn test_terrain_sculpt_raise() {
    let op = TerrainSculptOp::Raise(10.0);
    assert!((op.apply_to_height(5.0) - 15.0).abs() < f32::EPSILON);
}

#[test]
fn test_terrain_sculpt_lower() {
    let op = TerrainSculptOp::Lower(3.0);
    assert!((op.apply_to_height(10.0) - 7.0).abs() < f32::EPSILON);
}

#[test]
fn test_terrain_sculpt_flatten() {
    let op = TerrainSculptOp::Flatten(50.0);
    assert!((op.apply_to_height(10.0) - 50.0).abs() < f32::EPSILON);
    assert!((op.apply_to_height(100.0) - 50.0).abs() < f32::EPSILON);
}

#[test]
fn test_terrain_sculpt_smooth() {
    let op = TerrainSculptOp::Smooth(0.5);
    // Smoothing with strength 0.5 should halve the height (moving toward 0)
    assert!((op.apply_to_height(10.0) - 5.0).abs() < f32::EPSILON);
}

#[test]
fn test_terrain_sculpt_paint_no_height_change() {
    let op = TerrainSculptOp::Paint("sand".to_string());
    assert!((op.apply_to_height(42.0) - 42.0).abs() < f32::EPSILON);
}

#[test]
fn test_cutscene_timeline_creation() {
    let mut timeline = CutsceneTimeline::new("opening");
    assert_eq!(timeline.name, "opening");
    assert_eq!(timeline.total_duration(), 0.0);

    timeline.add_camera_keyframe(0.0, [0.0, 10.0, 0.0], [50.0, 0.0, 50.0], 60.0);
    timeline.add_camera_keyframe(3.0, [25.0, 10.0, 0.0], [50.0, 0.0, 50.0], 45.0);
    timeline.add_camera_keyframe(6.0, [50.0, 5.0, 0.0], [50.0, 0.0, 50.0], 60.0);

    assert_eq!(timeline.camera_keyframes.len(), 3);
    assert!((timeline.total_duration() - 6.0).abs() < f32::EPSILON);
}

#[test]
fn test_cutscene_audio_cues() {
    let mut timeline = CutsceneTimeline::new("test");
    timeline.add_audio_cue(0.0, "intro_music", 0.8);
    timeline.add_audio_cue(5.0, "explosion_sfx", 1.0);

    assert_eq!(timeline.audio_cues.len(), 2);
    assert_eq!(timeline.audio_cues[0].asset_id, "intro_music");
    assert!((timeline.audio_cues[1].volume - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_cutscene_entity_animations() {
    let mut timeline = CutsceneTimeline::new("test");
    timeline.add_entity_animation(1.0, "hero", "walk", 4.0);
    timeline.add_entity_animation(3.0, "villain", "attack", 2.0);

    assert_eq!(timeline.entity_animations.len(), 2);
    // Total duration should be max of (1.0 + 4.0, 3.0 + 2.0) = 5.0
    assert!((timeline.total_duration() - 5.0).abs() < f32::EPSILON);
}

#[test]
fn test_cutscene_camera_at_time() {
    let mut timeline = CutsceneTimeline::new("cam_test");
    timeline.add_camera_keyframe(0.0, [0.0; 3], [1.0, 0.0, 0.0], 60.0);
    timeline.add_camera_keyframe(5.0, [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 45.0);

    let cam = timeline.camera_at(2.5).unwrap();
    assert_eq!(cam.position, [0.0; 3]); // Before second keyframe, returns first

    let cam2 = timeline.camera_at(5.0).unwrap();
    assert_eq!(cam2.position, [10.0, 0.0, 0.0]);

    assert!(timeline.camera_at(-1.0).is_none());
}

#[test]
fn test_material_paint_brush() {
    let material = SpectralMaterial {
        name: "gold".to_string(),
        reflectance: vec![0.9, 0.85, 0.3, 0.1],
        roughness: 0.2,
        metallic: 1.0,
    };

    let brush = MaterialPaintBrush::new(material, 5.0, 0.7);
    assert_eq!(brush.material.name, "gold");
    assert!((brush.radius - 5.0).abs() < f32::EPSILON);
    assert_eq!(brush.falloff, BrushFalloff::Smooth);

    // At center, pressure should equal brush pressure
    let p_center = brush.pressure_at_distance(0.0);
    assert!((p_center - 0.7).abs() < 0.01);

    // At edge, pressure should be zero
    let p_edge = brush.pressure_at_distance(5.0);
    assert!((p_edge - 0.0).abs() < f32::EPSILON);

    // Outside radius, pressure should be zero
    let p_outside = brush.pressure_at_distance(10.0);
    assert!((p_outside - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_material_paint_brush_constant_falloff() {
    let material = SpectralMaterial {
        name: "test".to_string(),
        reflectance: vec![0.5],
        roughness: 0.5,
        metallic: 0.0,
    };

    let mut brush = MaterialPaintBrush::new(material, 10.0, 1.0);
    brush.falloff = BrushFalloff::Constant;

    // Constant falloff should give full pressure everywhere inside radius
    assert!((brush.pressure_at_distance(0.0) - 1.0).abs() < f32::EPSILON);
    assert!((brush.pressure_at_distance(5.0) - 1.0).abs() < f32::EPSILON);
    assert!((brush.pressure_at_distance(9.9) - 1.0).abs() < f32::EPSILON);
    assert!((brush.pressure_at_distance(10.0) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_material_paint_brush_linear_falloff() {
    let material = SpectralMaterial {
        name: "test".to_string(),
        reflectance: vec![0.5],
        roughness: 0.5,
        metallic: 0.0,
    };

    let mut brush = MaterialPaintBrush::new(material, 10.0, 1.0);
    brush.falloff = BrushFalloff::Linear;

    assert!((brush.pressure_at_distance(0.0) - 1.0).abs() < f32::EPSILON);
    assert!((brush.pressure_at_distance(5.0) - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_sculpt_command() {
    let brush = BrushStroke::new([10.0, 0.0, 10.0], 5.0, "terrain", 1.0);
    let cmd = SculptCommand {
        brush,
        operation: TerrainSculptOp::Raise(2.0),
    };

    assert_eq!(cmd.operation, TerrainSculptOp::Raise(2.0));
    assert!(cmd.brush.contains_point([12.0, 0.0, 10.0]));
}
