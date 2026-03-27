use std::path::PathBuf;
use vox_script::mod_manager::*;

#[test]
fn create_mod_manager() {
    let mgr = ModManager::new(PathBuf::from("/tmp/test_mods"));
    assert_eq!(mgr.mod_count(), 0);
}

#[test]
fn scan_empty_directory() {
    let dir = std::env::temp_dir().join("ochroma_mod_test_empty");
    let _ = std::fs::create_dir_all(&dir);
    let mut mgr = ModManager::new(dir.clone());
    mgr.scan();
    assert_eq!(mgr.mod_count(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scan_finds_mods() {
    let dir = std::env::temp_dir().join("ochroma_mod_test_scan");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Create a mock mod
    let mod_dir = dir.join("test_mod");
    std::fs::create_dir_all(&mod_dir).unwrap();
    std::fs::write(
        mod_dir.join("manifest.toml"),
        r#"
name = "test_mod"
version = "1.0.0"
author = "Tester"
description = "A test mod"
dependencies = []
entry_point = "mod.wasm"
"#,
    )
    .unwrap();

    let mut mgr = ModManager::new(dir.clone());
    mgr.scan();
    assert_eq!(mgr.mod_count(), 1);
    assert_eq!(mgr.mods[0].manifest.name, "test_mod");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn enable_disable_mod() {
    let dir = std::env::temp_dir().join("ochroma_mod_test_toggle");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mod_dir = dir.join("my_mod");
    std::fs::create_dir_all(&mod_dir).unwrap();
    std::fs::write(
        mod_dir.join("manifest.toml"),
        r#"
name = "my_mod"
version = "1.0"
author = "Me"
description = "My mod"
dependencies = []
entry_point = "main.wasm"
"#,
    )
    .unwrap();

    let mut mgr = ModManager::new(dir.clone());
    mgr.scan();
    assert_eq!(mgr.enabled_mods().len(), 1);
    mgr.disable("my_mod");
    assert_eq!(mgr.enabled_mods().len(), 0);
    mgr.enable("my_mod");
    assert_eq!(mgr.enabled_mods().len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dependency_check() {
    let mut mgr = ModManager::new(PathBuf::from("/tmp"));
    mgr.mods.push(LoadedMod {
        manifest: ModManifest {
            name: "mod_a".into(),
            version: "1.0".into(),
            author: "".into(),
            description: "".into(),
            dependencies: vec!["mod_b".into()],
            entry_point: "a.wasm".into(),
        },
        path: PathBuf::new(),
        enabled: true,
        load_order: 0,
    });
    // mod_b not present -> dependency error
    let errors = mgr.check_dependencies();
    assert!(!errors.is_empty());
    assert!(errors[0].contains("mod_b"));
}
