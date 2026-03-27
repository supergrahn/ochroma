use std::fs;
use std::path::PathBuf;
use vox_script::plugin_system::{PluginManager, PluginManifest, PluginState};

fn create_test_plugin_dir() -> (PathBuf, PathBuf) {
    let base = std::env::temp_dir().join(format!("ochroma_plugin_test_{}", std::process::id()));
    let plugin_dir = base.join("my_plugin");
    fs::create_dir_all(&plugin_dir).unwrap();

    let manifest = r#"
name = "my_plugin"
version = "1.0.0"
engine_version_min = "0.1.0"
engine_version_max = "1.0.0"
entry_point = "plugin.wasm"
dependencies = []
"#;
    fs::write(plugin_dir.join("manifest.toml"), manifest).unwrap();
    fs::write(plugin_dir.join("plugin.wasm"), b"fake wasm").unwrap();

    (base, plugin_dir)
}

fn cleanup(base: &PathBuf) {
    let _ = fs::remove_dir_all(base);
}

#[test]
fn discover_plugins() {
    let (base, _) = create_test_plugin_dir();
    let mut mgr = PluginManager::new("0.1.0");
    let found = mgr.discover(&base);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], "my_plugin");
    assert_eq!(mgr.plugin_count(), 1);
    cleanup(&base);
}

#[test]
fn load_and_unload_plugin() {
    let (base, _) = create_test_plugin_dir();
    let mut mgr = PluginManager::new("0.1.0");
    mgr.discover(&base);

    mgr.load("my_plugin").unwrap();
    assert_eq!(mgr.active_plugins(), vec!["my_plugin".to_string()]);
    assert_eq!(mgr.plugin_state("my_plugin"), Some(&PluginState::Active));

    mgr.unload("my_plugin").unwrap();
    assert!(mgr.active_plugins().is_empty());
    assert_eq!(mgr.plugin_state("my_plugin"), Some(&PluginState::Disabled));

    cleanup(&base);
}

#[test]
fn compatibility_check() {
    let mgr = PluginManager::new("0.5.0");
    let manifest = PluginManifest {
        name: "test".to_string(),
        version: "1.0.0".to_string(),
        engine_version_min: "0.1.0".to_string(),
        engine_version_max: "1.0.0".to_string(),
        entry_point: "main.wasm".to_string(),
        dependencies: vec![],
    };
    assert!(mgr.is_compatible(&manifest));

    let mgr_new = PluginManager::new("2.0.0");
    assert!(!mgr_new.is_compatible(&manifest));
}

#[test]
fn error_on_missing_entry_point() {
    let base = std::env::temp_dir().join(format!("ochroma_plugin_noentry_{}", std::process::id()));
    let plugin_dir = base.join("broken_plugin");
    fs::create_dir_all(&plugin_dir).unwrap();

    let manifest = r#"
name = "broken_plugin"
version = "1.0.0"
engine_version_min = "0.1.0"
engine_version_max = "1.0.0"
entry_point = "missing.wasm"
dependencies = []
"#;
    fs::write(plugin_dir.join("manifest.toml"), manifest).unwrap();
    // Do NOT create missing.wasm

    let mut mgr = PluginManager::new("0.1.0");
    mgr.discover(&base);

    assert_eq!(mgr.plugin_count(), 1);
    assert!(matches!(
        mgr.plugin_state("broken_plugin"),
        Some(PluginState::Error(_))
    ));

    // Loading should fail
    assert!(mgr.load("broken_plugin").is_err());

    cleanup(&base);
}

#[test]
fn active_list_empty_initially() {
    let mgr = PluginManager::new("0.1.0");
    assert!(mgr.active_plugins().is_empty());
    assert_eq!(mgr.plugin_count(), 0);
}

#[test]
fn load_unknown_plugin_fails() {
    let mut mgr = PluginManager::new("0.1.0");
    assert!(mgr.load("nonexistent").is_err());
}
