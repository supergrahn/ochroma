use uuid::Uuid;
use vox_data::library::{AssetEntry, AssetLibrary, AssetPipeline, AssetType};

#[test]
fn register_and_lookup_by_uuid() {
    let mut lib = AssetLibrary::new();
    let uuid = Uuid::new_v4();
    lib.register(AssetEntry {
        uuid,
        name: "Test Building".to_string(),
        asset_type: AssetType::Building,
        pipeline: AssetPipeline::ProcGS,
        tags: vec!["victorian".to_string(), "residential".to_string()],
        description: "A test building".to_string(),
    });
    let entry = lib.get(&uuid).unwrap();
    assert_eq!(entry.name, "Test Building");
}

#[test]
fn search_by_tag() {
    let mut lib = AssetLibrary::new();
    for i in 0..5 {
        lib.register(AssetEntry {
            uuid: Uuid::new_v4(),
            name: format!("Building {i}"),
            asset_type: AssetType::Building,
            pipeline: AssetPipeline::ProcGS,
            tags: vec!["tagged".to_string()],
            description: "".to_string(),
        });
    }
    lib.register(AssetEntry {
        uuid: Uuid::new_v4(),
        name: "Untagged".to_string(),
        asset_type: AssetType::Prop,
        pipeline: AssetPipeline::Turnaround,
        tags: vec![],
        description: "".to_string(),
    });

    let results = lib.search_by_tag("tagged");
    assert_eq!(results.len(), 5);
}

#[test]
fn search_by_type() {
    let mut lib = AssetLibrary::new();
    lib.register(AssetEntry {
        uuid: Uuid::new_v4(),
        name: "Tree".to_string(),
        asset_type: AssetType::Vegetation,
        pipeline: AssetPipeline::LyraCapture,
        tags: vec![],
        description: "".to_string(),
    });
    lib.register(AssetEntry {
        uuid: Uuid::new_v4(),
        name: "Car".to_string(),
        asset_type: AssetType::Vehicle,
        pipeline: AssetPipeline::Turnaround,
        tags: vec![],
        description: "".to_string(),
    });

    let veg = lib.search_by_type(&AssetType::Vegetation);
    assert_eq!(veg.len(), 1);
    assert_eq!(veg[0].name, "Tree");

    let vehicle = lib.search_by_type(&AssetType::Vehicle);
    assert_eq!(vehicle.len(), 1);
}

#[test]
fn count_and_all() {
    let mut lib = AssetLibrary::new();
    assert_eq!(lib.count(), 0);
    for _ in 0..3 {
        lib.register(AssetEntry {
            uuid: Uuid::new_v4(),
            name: "x".to_string(),
            asset_type: AssetType::Terrain,
            pipeline: AssetPipeline::NeuralInfill,
            tags: vec![],
            description: "".to_string(),
        });
    }
    assert_eq!(lib.count(), 3);
    assert_eq!(lib.all().count(), 3);
}
