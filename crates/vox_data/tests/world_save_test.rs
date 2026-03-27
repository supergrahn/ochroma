use std::collections::HashMap;
use vox_data::world_save::*;

fn make_test_entity(name: &str) -> SavedEntity {
    SavedEntity {
        name: name.to_string(),
        position: [1.0, 2.0, 3.0],
        rotation: [0.0, 0.707, 0.0, 0.707],
        scale: [1.0, 1.0, 1.0],
        asset_path: Some("assets/models/cube.vxm".to_string()),
        scripts: vec!["rotate.rhai".to_string(), "bounce.rhai".to_string()],
        tags: vec!["prop".to_string(), "interactive".to_string()],
        custom_data: HashMap::new(),
        collider: None,
        audio: None,
        light: None,
    }
}

#[test]
fn test_save_and_load_round_trip() {
    let tmp = std::env::temp_dir().join("ochroma_test_round_trip.ochroma_save");
    let mut save = WorldSave::new("test_scene");
    save.add_entity(make_test_entity("EntityA"));
    save.add_entity(make_test_entity("EntityB"));

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();

    assert_eq!(loaded.scene_name, "test_scene");
    assert_eq!(loaded.entity_count(), 2);
    assert_eq!(loaded.entities[0].name, "EntityA");
    assert_eq!(loaded.entities[1].name, "EntityB");

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_entity_fields_round_trip() {
    let tmp = std::env::temp_dir().join("ochroma_test_fields.ochroma_save");
    let mut save = WorldSave::new("fields_scene");
    save.add_entity(make_test_entity("Player"));

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();
    let e = &loaded.entities[0];

    assert_eq!(e.position, [1.0, 2.0, 3.0]);
    assert_eq!(e.rotation, [0.0, 0.707, 0.0, 0.707]);
    assert_eq!(e.scale, [1.0, 1.0, 1.0]);
    assert_eq!(e.asset_path.as_deref(), Some("assets/models/cube.vxm"));
    assert_eq!(e.scripts, vec!["rotate.rhai", "bounce.rhai"]);
    assert_eq!(e.tags, vec!["prop", "interactive"]);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_custom_data_mixed_types() {
    let tmp = std::env::temp_dir().join("ochroma_test_custom_data.ochroma_save");
    let mut save = WorldSave::new("custom_scene");

    let mut entity = make_test_entity("CustomEntity");
    entity.custom_data.insert("health".to_string(), serde_json::json!(100));
    entity.custom_data.insert("name".to_string(), serde_json::json!("hero"));
    entity.custom_data.insert("active".to_string(), serde_json::json!(true));
    entity.custom_data.insert("inventory".to_string(), serde_json::json!(["sword", "shield"]));
    save.add_entity(entity);

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();
    let cd = &loaded.entities[0].custom_data;

    assert_eq!(cd["health"], serde_json::json!(100));
    assert_eq!(cd["name"], serde_json::json!("hero"));
    assert_eq!(cd["active"], serde_json::json!(true));
    assert_eq!(cd["inventory"], serde_json::json!(["sword", "shield"]));

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_empty_save_round_trip() {
    let tmp = std::env::temp_dir().join("ochroma_test_empty.ochroma_save");
    let save = WorldSave::new("empty_scene");
    assert_eq!(save.entity_count(), 0);

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();

    assert_eq!(loaded.scene_name, "empty_scene");
    assert_eq!(loaded.entity_count(), 0);
    assert_eq!(loaded.resources.time_of_day, 12.0);
    assert_eq!(loaded.resources.game_state, "playing");

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_collider_serialization() {
    let tmp = std::env::temp_dir().join("ochroma_test_colliders.ochroma_save");
    let mut save = WorldSave::new("collider_scene");

    let mut box_entity = make_test_entity("BoxCollider");
    box_entity.collider = Some(SavedCollider {
        shape_type: "box".to_string(),
        dimensions: vec![1.0, 2.0, 3.0],
    });

    let mut sphere_entity = make_test_entity("SphereCollider");
    sphere_entity.collider = Some(SavedCollider {
        shape_type: "sphere".to_string(),
        dimensions: vec![5.0],
    });

    let mut capsule_entity = make_test_entity("CapsuleCollider");
    capsule_entity.collider = Some(SavedCollider {
        shape_type: "capsule".to_string(),
        dimensions: vec![0.5, 2.0],
    });

    save.add_entity(box_entity);
    save.add_entity(sphere_entity);
    save.add_entity(capsule_entity);

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();

    let box_c = loaded.entities[0].collider.as_ref().unwrap();
    assert_eq!(box_c.shape_type, "box");
    assert_eq!(box_c.dimensions, vec![1.0, 2.0, 3.0]);

    let sphere_c = loaded.entities[1].collider.as_ref().unwrap();
    assert_eq!(sphere_c.shape_type, "sphere");
    assert_eq!(sphere_c.dimensions, vec![5.0]);

    let capsule_c = loaded.entities[2].collider.as_ref().unwrap();
    assert_eq!(capsule_c.shape_type, "capsule");
    assert_eq!(capsule_c.dimensions, vec![0.5, 2.0]);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_light_serialization() {
    let tmp = std::env::temp_dir().join("ochroma_test_lights.ochroma_save");
    let mut save = WorldSave::new("light_scene");

    let mut point_light = make_test_entity("PointLight");
    point_light.light = Some(SavedLight {
        light_type: "point".to_string(),
        color: [1.0, 0.8, 0.6],
        intensity: 100.0,
        radius: 25.0,
    });

    let mut dir_light = make_test_entity("DirLight");
    dir_light.light = Some(SavedLight {
        light_type: "directional".to_string(),
        color: [1.0, 1.0, 1.0],
        intensity: 50.0,
        radius: 0.0,
    });

    save.add_entity(point_light);
    save.add_entity(dir_light);

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();

    let pl = loaded.entities[0].light.as_ref().unwrap();
    assert_eq!(pl.light_type, "point");
    assert_eq!(pl.color, [1.0, 0.8, 0.6]);
    assert_eq!(pl.intensity, 100.0);
    assert_eq!(pl.radius, 25.0);

    let dl = loaded.entities[1].light.as_ref().unwrap();
    assert_eq!(dl.light_type, "directional");
    assert_eq!(dl.intensity, 50.0);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_quick_save_path_valid() {
    let path = WorldSave::quick_save_path();
    assert!(path.to_str().unwrap().contains("ochroma"));
    assert!(path.to_str().unwrap().ends_with("quicksave.ochroma_save"));
}

#[test]
fn test_version_preserved() {
    let tmp = std::env::temp_dir().join("ochroma_test_version.ochroma_save");
    let save = WorldSave::new("version_scene");
    assert_eq!(save.version, 1);
    assert_eq!(save.engine_version, "0.1.0");

    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();

    assert_eq!(loaded.version, 1);
    assert_eq!(loaded.engine_version, "0.1.0");

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_file_is_human_readable_json() {
    let tmp = std::env::temp_dir().join("ochroma_test_readable.ochroma_save");
    let mut save = WorldSave::new("readable_scene");
    save.add_entity(make_test_entity("HeroCharacter"));

    save.save_to_file(&tmp).unwrap();
    let raw = std::fs::read_to_string(&tmp).unwrap();

    // Human-readable: contains entity name as a plain string
    assert!(raw.contains("HeroCharacter"));
    assert!(raw.contains("readable_scene"));
    // Pretty-printed JSON has newlines
    assert!(raw.contains('\n'));

    std::fs::remove_file(&tmp).ok();
}
