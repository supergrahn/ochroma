use vox_nn::layout::{LayoutError, LayoutInterpreter, StubLayoutInterpreter};
use vox_nn::scene_graph::{Season, Weather};

#[test]
fn stub_returns_valid_scene_graph() {
    let interpreter = StubLayoutInterpreter;
    let result = interpreter.interpret("A foggy Victorian street at dusk");
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

    let scene = result.unwrap();
    assert!(!scene.street.name.is_empty());
    assert!(scene.street.width_metres > 0.0);
    assert!(scene.street.length_metres > 0.0);
    assert!(!scene.street.buildings.is_empty());
    assert!(!scene.street.props.is_empty());
    assert!(!scene.street.vegetation.is_empty());
}

#[test]
fn stub_returns_victorian_style_scene() {
    let interpreter = StubLayoutInterpreter;
    let scene = interpreter.interpret("Victorian street").unwrap();

    assert!(scene.street.name.contains("Victorian"));
    assert_eq!(scene.atmosphere.weather, Weather::PartlyCloudy);
    assert_eq!(scene.atmosphere.season, Season::Autumn);
    assert!(scene.atmosphere.time_of_day_hour >= 0.0 && scene.atmosphere.time_of_day_hour < 24.0);
}

#[test]
fn empty_prompt_returns_error() {
    let interpreter = StubLayoutInterpreter;
    let result = interpreter.interpret("   ");
    assert!(matches!(result, Err(LayoutError::EmptyPrompt)));
}

#[test]
fn scene_graph_json_round_trip() {
    let interpreter = StubLayoutInterpreter;
    let original = interpreter.interpret("Victorian street scene").unwrap();

    let json = serde_json::to_string(&original).expect("serialization failed");
    assert!(!json.is_empty());

    let restored: vox_nn::scene_graph::SceneGraph =
        serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(original.street.name, restored.street.name);
    assert_eq!(original.street.buildings.len(), restored.street.buildings.len());
    assert_eq!(original.street.props.len(), restored.street.props.len());
    assert_eq!(original.street.vegetation.len(), restored.street.vegetation.len());
    assert_eq!(original.atmosphere.weather, restored.atmosphere.weather);
    assert_eq!(original.atmosphere.season, restored.atmosphere.season);
    assert!((original.atmosphere.time_of_day_hour - restored.atmosphere.time_of_day_hour).abs() < f32::EPSILON);
    assert!((original.atmosphere.fog_density - restored.atmosphere.fog_density).abs() < f32::EPSILON);
}

#[test]
fn building_slots_have_valid_positions() {
    let interpreter = StubLayoutInterpreter;
    let scene = interpreter.interpret("Victorian street").unwrap();

    for building in &scene.street.buildings {
        assert!(!building.style.is_empty());
        assert!(!building.facade_material.is_empty());
        assert!(building.height_metres > 0.0);
    }
}
