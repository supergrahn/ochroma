use vox_data::map_file::*;

#[test]
fn create_and_populate_map() {
    let mut map = MapFile::new("Test Map");
    map.place_object("Building", "buildings/house.ply", [10.0, 0.0, 5.0]);
    map.place_object("Tree", "trees/oak.ply", [20.0, 0.0, 15.0]);
    map.add_light("point", [10.0, 5.0, 5.0], [1.0, 0.9, 0.8], 50.0);

    assert_eq!(map.object_count(), 2);
    assert_eq!(map.light_count(), 1);
}

#[test]
fn map_save_load_round_trip() {
    let dir = std::env::temp_dir().join("ochroma_map_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut map = MapFile::new("Round Trip");
    map.description = "Test map".into();
    map.place_object("House", "house.ply", [1.0, 2.0, 3.0]);
    map.settings.time_of_day = 18.5;
    map.settings.weather = "rain".into();

    let path = dir.join("test.ochroma_map");
    map.save(&path).unwrap();
    let loaded = MapFile::load(&path).unwrap();

    assert_eq!(loaded.name, "Round Trip");
    assert_eq!(loaded.object_count(), 1);
    assert_eq!(loaded.placed_objects[0].asset_path, "house.ply");
    assert!((loaded.settings.time_of_day - 18.5).abs() < 0.01);
    assert_eq!(loaded.settings.weather, "rain");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn default_spawn_point() {
    let map = MapFile::new("Test");
    let spawn = map.default_spawn().unwrap();
    assert!(spawn.is_default);
    assert_eq!(spawn.position[1], 10.0);
}

#[test]
fn placed_object_with_scripts() {
    let mut map = MapFile::new("Test");
    map.placed_objects.push(PlacedObject {
        name: "NPC".into(),
        asset_path: "characters/npc.ply".into(),
        position: [5.0, 0.0, 5.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        scripts: vec!["Patrol".into(), "Dialogue".into()],
        properties: [("greeting".into(), "Hello!".into())].into(),
    });

    assert_eq!(map.placed_objects[0].scripts.len(), 2);
    assert_eq!(map.placed_objects[0].properties["greeting"], "Hello!");
}
