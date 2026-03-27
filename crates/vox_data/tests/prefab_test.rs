use std::collections::HashMap;
use vox_data::prefab::{Prefab, PrefabEntity};

fn make_entity(name: &str, pos: [f32; 3]) -> PrefabEntity {
    PrefabEntity {
        name: name.to_string(),
        local_position: pos,
        local_rotation: [0.0, 0.0, 0.0, 1.0],
        local_scale: [1.0, 1.0, 1.0],
        asset_path: None,
        scripts: Vec::new(),
        tags: Vec::new(),
        children_indices: Vec::new(),
        components: HashMap::new(),
    }
}

#[test]
fn create_prefab_and_add_entities() {
    let mut prefab = Prefab::new("TestPrefab");
    assert_eq!(prefab.entity_count(), 0);

    let idx0 = prefab.add_entity(make_entity("Root", [0.0, 0.0, 0.0]));
    let idx1 = prefab.add_entity(make_entity("Child", [1.0, 2.0, 3.0]));

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(prefab.entity_count(), 2);
    assert_eq!(prefab.name, "TestPrefab");
}

#[test]
fn save_load_round_trip() {
    let dir = std::env::temp_dir().join("vox_prefab_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.prefab.json");

    let mut prefab = Prefab::new("House");
    let mut root = make_entity("Foundation", [0.0, 0.0, 0.0]);
    root.asset_path = Some("meshes/foundation.vxm".to_string());
    root.tags = vec!["structure".to_string()];
    root.scripts = vec!["scripts/building.rhai".to_string()];
    root.children_indices = vec![1];
    prefab.add_entity(root);

    let mut child = make_entity("Roof", [0.0, 5.0, 0.0]);
    child.components.insert(
        "material".to_string(),
        serde_json::json!({"color": "red"}),
    );
    prefab.add_entity(child);

    prefab.save(&path).unwrap();
    let loaded = Prefab::load(&path).unwrap();

    assert_eq!(loaded.name, "House");
    assert_eq!(loaded.entity_count(), 2);
    assert_eq!(loaded.entities[0].name, "Foundation");
    assert_eq!(loaded.entities[0].asset_path.as_deref(), Some("meshes/foundation.vxm"));
    assert_eq!(loaded.entities[0].children_indices, vec![1]);
    assert_eq!(loaded.entities[1].name, "Roof");
    assert_eq!(
        loaded.entities[1].components.get("material").unwrap(),
        &serde_json::json!({"color": "red"})
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn instantiate_at_position_offsets() {
    let mut prefab = Prefab::new("Tree");
    prefab.add_entity(make_entity("Trunk", [0.0, 0.0, 0.0]));
    prefab.add_entity(make_entity("Canopy", [0.0, 3.0, 0.0]));

    let instances = prefab.instantiate([10.0, 20.0, 30.0]);

    assert_eq!(instances.len(), 2);
    assert_eq!(instances[0].name, "Trunk");
    assert_eq!(instances[0].world_position, [10.0, 20.0, 30.0]);
    assert_eq!(instances[1].name, "Canopy");
    assert_eq!(instances[1].world_position, [10.0, 23.0, 30.0]);
}

#[test]
fn children_indices_work() {
    let mut prefab = Prefab::new("Vehicle");
    let mut body = make_entity("Body", [0.0, 0.0, 0.0]);
    body.children_indices = vec![1, 2];
    prefab.add_entity(body);
    prefab.add_entity(make_entity("WheelFront", [1.0, -0.5, 0.0]));
    prefab.add_entity(make_entity("WheelRear", [-1.0, -0.5, 0.0]));

    let root = &prefab.entities[0];
    assert_eq!(root.children_indices.len(), 2);
    for &idx in &root.children_indices {
        assert!(idx < prefab.entity_count());
    }
    assert_eq!(prefab.entities[root.children_indices[0]].name, "WheelFront");
    assert_eq!(prefab.entities[root.children_indices[1]].name, "WheelRear");
}
