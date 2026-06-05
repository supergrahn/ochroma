use std::collections::HashMap;
use vox_data::prefab::{Prefab, PrefabEntity};
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
        splats: vec![],
        prefab_ref: None,
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
    assert!(path.to_str().unwrap().contains("quicksave"));
    assert!(path.to_str().unwrap().ends_with(".json"));
}

/// A distinct, deterministic 16-band spectral signature so a corrupted band
/// (e.g. silently dropped, reordered, or rounded) makes the round-trip fail.
fn spectral_signature(seed: f32) -> [f32; SAVED_SPLAT_BANDS] {
    let mut bands = [0.0f32; SAVED_SPLAT_BANDS];
    for (i, b) in bands.iter_mut().enumerate() {
        // Mix index + seed; values are exactly representable in f32 JSON.
        *b = (i as f32) * 0.0625 + seed * 0.5;
    }
    bands
}

/// COMPLETE world save -> load round-trip across every seam:
/// multiple entities, full transforms, per-splat 16-band spectral data, and a
/// prefab instance (with a SavedPrefabRef). Asserts the loaded world is
/// field-identical to the original — real equality, not is_some().
#[test]
fn test_full_world_round_trip_spectral_and_prefab() {
    let tmp = std::env::temp_dir().join("ochroma_full_world_round_trip.ochroma_save");

    // --- Build a prefab and instantiate it into save records (prefab seam) ---
    let mut prefab = Prefab::new("streetlamp");
    prefab.add_entity(PrefabEntity {
        name: "post".to_string(),
        local_position: [0.0, 0.0, 0.0],
        local_rotation: [0.0, 0.0, 0.0, 1.0],
        local_scale: [1.0, 1.0, 1.0],
        asset_path: Some("assets/post.vxm".to_string()),
        scripts: vec![],
        tags: vec!["structure".to_string()],
        children_indices: vec![1],
        components: HashMap::new(),
    });
    prefab.add_entity(PrefabEntity {
        name: "bulb".to_string(),
        local_position: [0.0, 4.0, 0.0],
        local_rotation: [0.0, 0.0, 0.0, 1.0],
        local_scale: [0.5, 0.5, 0.5],
        asset_path: Some("assets/bulb.vxm".to_string()),
        scripts: vec!["flicker.rhai".to_string()],
        tags: vec!["light".to_string()],
        children_indices: vec![],
        components: HashMap::new(),
    });
    let prefab_world_pos = [10.0, 0.0, -5.0];
    let prefab_instances = prefab.instantiate_into_save(prefab_world_pos, 42);
    assert_eq!(prefab_instances.len(), 2, "prefab should produce 2 entities");

    // --- Build the world ---
    let mut save = WorldSave::new("full_world");

    // Entity 0: hand-placed, carries two spectral splats.
    let mut splat_entity = SavedEntity::new("splat_prop", [1.5, 2.5, 3.5]);
    splat_entity.rotation = [0.0, 0.7071, 0.0, 0.7071];
    splat_entity.scale = [2.0, 2.0, 2.0];
    splat_entity.asset_path = Some("assets/prop.vxm".to_string());
    splat_entity.tags = vec!["prop".to_string()];
    splat_entity.splats = vec![
        SavedSplat {
            position: [0.1, 0.2, 0.3],
            spectral: spectral_signature(1.0),
            opacity: 0.875,
        },
        SavedSplat {
            position: [-0.4, 0.5, -0.6],
            spectral: spectral_signature(2.0),
            opacity: 0.25,
        },
    ];
    save.add_entity(splat_entity);

    // Entity 1: plain transform-only entity.
    save.add_entity(SavedEntity::new("ground", [0.0, -1.0, 0.0]));

    // Entities 2 & 3: the prefab instance.
    for e in prefab_instances {
        save.add_entity(e);
    }

    // Snapshot the original BEFORE persisting (Clone) so equality is meaningful.
    let original = save.clone();
    assert_eq!(original.entity_count(), 4);

    // --- Persist and reload ---
    save.save_to_file(&tmp).unwrap();
    let loaded = WorldSave::load_from_file(&tmp).unwrap();

    // (1) Whole-world structural equality: every field of every entity +
    //     resources + metadata must match. This is the strongest assertion.
    assert_eq!(loaded, original, "loaded world must equal original world");

    // (2) Entity count.
    assert_eq!(loaded.entity_count(), 4);

    // (3) Positions / transforms preserved exactly.
    assert_eq!(loaded.entities[0].position, [1.5, 2.5, 3.5]);
    assert_eq!(loaded.entities[0].rotation, [0.0, 0.7071, 0.0, 0.7071]);
    assert_eq!(loaded.entities[0].scale, [2.0, 2.0, 2.0]);
    assert_eq!(loaded.entities[1].position, [0.0, -1.0, 0.0]);

    // (4) Spectral splat data: count, position, all 16 bands, opacity.
    let splats = &loaded.entities[0].splats;
    assert_eq!(splats.len(), 2, "both splats must survive the round-trip");
    assert_eq!(splats[0].position, [0.1, 0.2, 0.3]);
    assert_eq!(splats[0].spectral, spectral_signature(1.0));
    assert_eq!(splats[0].opacity, 0.875);
    assert_eq!(splats[1].spectral, spectral_signature(2.0));
    assert_eq!(splats[1].opacity, 0.25);
    // Band-by-band: the 7th band of splat 0 is index 6 -> 6*0.0625 + 0.5 = 0.875.
    assert_eq!(splats[0].spectral[6], 0.875);
    // The two splats must have genuinely different spectra (not aliased).
    assert_ne!(splats[0].spectral, splats[1].spectral);

    // (5) Prefab references preserved on instantiated entities; absent on others.
    assert!(loaded.entities[0].prefab_ref.is_none());
    assert!(loaded.entities[1].prefab_ref.is_none());

    let post_ref = loaded.entities[2]
        .prefab_ref
        .as_ref()
        .expect("prefab instance entity must carry a prefab_ref");
    assert_eq!(post_ref.prefab_name, "streetlamp");
    assert_eq!(post_ref.instance_position, prefab_world_pos);
    assert_eq!(post_ref.instance_id, 42);
    // Instantiated world-space position = local + world.
    assert_eq!(loaded.entities[2].name, "post");
    assert_eq!(loaded.entities[2].position, [10.0, 0.0, -5.0]);
    assert_eq!(loaded.entities[3].name, "bulb");
    assert_eq!(loaded.entities[3].position, [10.0, 4.0, -5.0]);
    assert_eq!(
        loaded.entities[3].prefab_ref.as_ref().unwrap().instance_id,
        42
    );

    std::fs::remove_file(&tmp).ok();
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
