use uuid::Uuid;
use vox_data::library::{AssetLibrary, AssetEntry, AssetType, AssetPipeline};

#[test]
fn save_and_load_index_toml() {
    let dir = std::env::temp_dir().join("ochroma_test_lib");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut lib = AssetLibrary::new();
    let uuid = Uuid::new_v4();
    lib.register(AssetEntry {
        uuid,
        name: "".to_string(),
        path: "buildings/house_01.vxm".into(),
        style: "victorian".into(),
        asset_type: AssetType::Building,
        description: "".to_string(),
        tags: vec!["victorian".into()],
        pipeline: AssetPipeline::ProcGS,
    });

    let path = dir.join("INDEX.toml");
    lib.save_index(&path).unwrap();
    assert!(path.exists());

    let loaded = AssetLibrary::load_index(&path).unwrap();
    let entry = loaded.get(uuid).unwrap();
    assert_eq!(entry.style, "victorian");

    let _ = std::fs::remove_dir_all(&dir);
}
