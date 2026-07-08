use vox_data::scene_serialize::WorldSnapshot;

#[test]
fn snapshot_round_trip_bytes() {
    let mut snap = WorldSnapshot::new(123.456);
    snap.simulation.game_time_hours = 72.0;
    // Game-specific state goes into metadata.
    snap.metadata
        .insert("city_name".to_string(), "TestVille".to_string());
    snap.metadata
        .insert("citizen_count".to_string(), "5000".to_string());

    let entity = snap.add_entity(1);
    entity
        .components
        .insert("position".to_string(), serde_json::json!([1.0, 2.0, 3.0]));

    let bytes = snap.to_bytes().unwrap();
    let loaded = WorldSnapshot::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.metadata["citizen_count"], "5000");
    assert_eq!(loaded.entities.len(), 1);
    assert_eq!(loaded.metadata["city_name"], "TestVille");
    assert!((loaded.simulation.game_time_hours - 72.0).abs() < 1e-6);
}

#[test]
fn snapshot_round_trip_file() {
    let dir = std::env::temp_dir().join("ochroma_scene_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut snap = WorldSnapshot::new(456.789);
    snap.simulation.season_day = 42;
    // Game-specific numeric state stored in metadata as strings.
    snap.metadata
        .insert("funds".to_string(), "12345.0".to_string());
    for i in 0..100 {
        let entity = snap.add_entity(i);
        entity
            .components
            .insert("type".to_string(), serde_json::json!("structure"));
    }

    let path = dir.join("test_scene.ochroma");
    snap.save_to_file(&path).unwrap();
    let loaded = WorldSnapshot::load_from_file(&path).unwrap();

    assert_eq!(loaded.entities.len(), 100);
    assert_eq!(loaded.simulation.season_day, 42);
    let funds: f64 = loaded.metadata["funds"].parse().unwrap();
    assert!((funds - 12345.0).abs() < 0.01);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compression_reduces_size() {
    let mut snap = WorldSnapshot::new(0.0);
    for i in 0..1000 {
        let entity = snap.add_entity(i);
        entity.components.insert(
            "data".to_string(),
            serde_json::json!({"x": i, "y": 0, "z": 0}),
        );
    }

    let bytes = snap.to_bytes().unwrap();
    let json_size = serde_json::to_vec(&snap).unwrap().len();
    assert!(
        bytes.len() < json_size,
        "Compressed {} should be < uncompressed {}",
        bytes.len(),
        json_size
    );
}
