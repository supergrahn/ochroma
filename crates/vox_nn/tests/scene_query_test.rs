use std::collections::HashMap;
use vox_nn::scene_query::{EntityDescription, SceneQueryEngine};

fn make_engine() -> SceneQueryEngine {
    let mut engine = SceneQueryEngine::new();
    engine.register(
        1,
        EntityDescription {
            name: "City Hall".to_string(),
            entity_type: "building".to_string(),
            position: [10.0, 0.0, 20.0],
            details: HashMap::from([
                ("style".to_string(), "Victorian".to_string()),
                ("capacity".to_string(), "500".to_string()),
            ]),
        },
    );
    engine.register(
        2,
        EntityDescription {
            name: "Oak Park".to_string(),
            entity_type: "park".to_string(),
            position: [30.0, 0.0, 40.0],
            details: HashMap::new(),
        },
    );
    engine.register(
        3,
        EntityDescription {
            name: "Main Street".to_string(),
            entity_type: "road".to_string(),
            position: [15.0, 0.0, 25.0],
            details: HashMap::from([("lanes".to_string(), "4".to_string())]),
        },
    );
    engine.register(
        4,
        EntityDescription {
            name: "Elm School".to_string(),
            entity_type: "building".to_string(),
            position: [50.0, 0.0, 60.0],
            details: HashMap::from([("tag".to_string(), "education".to_string())]),
        },
    );
    engine
}

#[test]
fn describe_building() {
    let engine = make_engine();
    let desc = engine.describe_entity(1).unwrap();
    assert!(desc.contains("City Hall"));
    assert!(desc.contains("building"));
    assert!(desc.contains("10.0"));
}

#[test]
fn describe_missing_entity() {
    let engine = make_engine();
    assert!(engine.describe_entity(999).is_none());
}

#[test]
fn find_by_type() {
    let engine = make_engine();
    let results = engine.find_entities_matching("building");
    assert_eq!(results.len(), 2);
}

#[test]
fn find_by_name() {
    let engine = make_engine();
    let results = engine.find_entities_matching("Oak");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1.name, "Oak Park");
}

#[test]
fn find_multiple_matches() {
    let engine = make_engine();
    // "street" should match Main Street (name) only
    let results = engine.find_entities_matching("street");
    assert!(!results.is_empty());
}

#[test]
fn empty_query_returns_empty() {
    let engine = make_engine();
    let results = engine.find_entities_matching("");
    assert!(results.is_empty());
}

#[test]
fn suggest_action_traffic() {
    let engine = make_engine();
    let suggestion = engine.suggest_action("There is heavy traffic congestion");
    assert!(suggestion.to_lowercase().contains("transit") || suggestion.to_lowercase().contains("road"));
}

#[test]
fn suggest_action_unknown() {
    let engine = make_engine();
    let suggestion = engine.suggest_action("some obscure problem xyz");
    assert!(suggestion.contains("Analyse"));
}

#[test]
fn scene_caption_with_entities() {
    let engine = make_engine();
    let caption = engine.generate_scene_caption(&[1, 2, 3], "sunset", "light rain");
    assert!(caption.contains("3 entities"));
    assert!(caption.contains("sunset"));
    assert!(caption.contains("light rain"));
}

#[test]
fn scene_caption_empty() {
    let engine = make_engine();
    let caption = engine.generate_scene_caption(&[], "noon", "clear");
    assert!(caption.contains("empty scene"));
}
