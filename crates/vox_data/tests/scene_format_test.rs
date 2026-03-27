use vox_data::scene_format::*;

#[test]
fn create_and_populate_scene() {
    let mut scene = SceneFile::new("Test Scene");
    scene.add_entity("Player", Transform::default());
    scene.add_entity("Building", Transform { position: [10.0, 0.0, 5.0], ..Default::default() });
    assert_eq!(scene.entity_count(), 2);
}

#[test]
fn scene_round_trip_file() {
    let dir = std::env::temp_dir().join("ochroma_scene_fmt_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut scene = SceneFile::new("Round Trip");
    scene.description = "Test scene".to_string();
    scene.add_entity("Cube", Transform { position: [1.0, 2.0, 3.0], ..Default::default() });
    scene.settings.fog_enabled = true;

    let path = dir.join("test.ochroma_scene");
    scene.save(&path).unwrap();
    let loaded = SceneFile::load(&path).unwrap();

    assert_eq!(loaded.name, "Round Trip");
    assert_eq!(loaded.entity_count(), 1);
    assert!(loaded.settings.fog_enabled);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn entity_components_are_flexible() {
    let mut scene = SceneFile::new("Components");
    let id = scene.add_entity("Custom", Transform::default());
    scene.entities[id as usize].components.insert("health".into(), serde_json::json!(100));
    scene.entities[id as usize].components.insert("inventory".into(), serde_json::json!(["sword", "shield"]));

    let json = serde_json::to_string(&scene).unwrap();
    let loaded: SceneFile = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.entities[0].components["health"], 100);
}
