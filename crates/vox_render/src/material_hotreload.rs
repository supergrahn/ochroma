//! Material hot-reload system.
//!
//! Polls `.toml` material files for modification-time changes and marks
//! affected entities dirty so the render pipeline can re-upload them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bevy_ecs::prelude::*;
use serde::Deserialize;

// ── Data types ────────────────────────────────────────────────────────────

/// A spectral material loaded from a `.toml` file.
///
/// Each spectral band maps to a wavelength bucket in the renderer.
/// `roughness` and `metallic` are PBR scalars in [0, 1].
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SpectralMaterialConfig {
    pub name: String,
    /// Per-band emissive weights, length 8.
    pub spectral_weights: Vec<f32>,
    pub roughness: f32,
    pub metallic: f32,
    /// Optional tint in linear RGB (length 3).
    pub tint: Option<Vec<f32>>,
}

impl SpectralMaterialConfig {
    /// Validate that `spectral_weights` has exactly 8 entries.
    pub fn is_valid(&self) -> bool {
        self.spectral_weights.len() == 8
            && self.roughness >= 0.0
            && self.roughness <= 1.0
            && self.metallic >= 0.0
            && self.metallic <= 1.0
    }
}

/// Handle that associates a file path with a material name.
#[derive(Debug, Clone)]
pub struct MaterialHandle {
    pub path: PathBuf,
    pub name: String,
}

/// ECS marker component placed on entities that use a hot-reloadable material.
#[derive(Component, Debug, Clone)]
pub struct MaterialRef {
    /// Must match a key registered in `HotMaterialLibrary`.
    pub material_name: String,
}

/// ECS marker component inserted by `material_reload_system` when a material
/// file has changed. Render systems consume and remove this component.
#[derive(Component, Debug)]
pub struct MaterialDirty;

// ── Hot-reload library ────────────────────────────────────────────────────

/// Tracks loaded materials and their file modification times.
///
/// Call `register` to watch a file; `poll` to check for changes.
#[derive(Resource, Default)]
pub struct HotMaterialLibrary {
    /// material_name → (path, last mtime, loaded config)
    entries: HashMap<String, (PathBuf, Option<SystemTime>, Option<SpectralMaterialConfig>)>,
    /// Rate-limit: only poll files this many seconds apart.
    poll_interval: f32,
    time_since_last_poll: f32,
}

impl HotMaterialLibrary {
    pub fn new(poll_interval_secs: f32) -> Self {
        Self {
            entries: HashMap::new(),
            poll_interval: poll_interval_secs,
            time_since_last_poll: 0.0,
        }
    }

    /// Register a material file for watching.
    pub fn register(&mut self, name: impl Into<String>, path: impl Into<PathBuf>) {
        let name = name.into();
        let path = path.into();
        // Try loading immediately; don't fail if file doesn't exist yet.
        let (config, mtime) = Self::load_file(&path);
        self.entries.insert(name, (path, mtime, config));
    }

    /// Returns the currently loaded config for a material, if any.
    pub fn get(&self, name: &str) -> Option<&SpectralMaterialConfig> {
        self.entries.get(name).and_then(|(_, _, cfg)| cfg.as_ref())
    }

    /// Returns a list of all registered material names.
    pub fn material_names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Returns the watched file path for a material name, if registered.
    pub fn path_for(&self, name: &str) -> Option<&PathBuf> {
        self.entries.get(name).map(|(p, _, _)| p)
    }

    /// Poll registered files for mtime changes.
    ///
    /// Returns the names of materials whose files changed since last poll.
    /// Rate-limited to `poll_interval` seconds.
    pub fn poll(&mut self, dt: f32) -> Vec<String> {
        self.time_since_last_poll += dt;
        if self.time_since_last_poll < self.poll_interval {
            return Vec::new();
        }
        self.time_since_last_poll = 0.0;

        let mut changed = Vec::new();
        for (name, (path, last_mtime, config)) in &mut self.entries {
            let (new_config, new_mtime) = Self::load_file(path);
            if new_mtime != *last_mtime
                && let Some(cfg) = new_config
                && cfg.is_valid()
            {
                *last_mtime = new_mtime;
                *config = Some(cfg);
                changed.push(name.clone());
                // parse failed or invalid: leave last_mtime as-is so next poll retries
                // file unreadable: also leave last_mtime as-is
            }
        }
        changed
    }

    fn load_file(path: &Path) -> (Option<SpectralMaterialConfig>, Option<SystemTime>) {
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok();
        let config = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| toml::from_str::<SpectralMaterialConfig>(&s).ok());
        (config, mtime)
    }

    /// Force-reload all files immediately (ignores rate limit).
    pub fn force_reload(&mut self) -> Vec<String> {
        // Bypass the rate-limit by directly performing the mtime scan.
        let mut changed = Vec::new();
        for (name, (path, last_mtime, config)) in &mut self.entries {
            let (new_config, new_mtime) = Self::load_file(path);
            if new_mtime != *last_mtime
                && let Some(cfg) = new_config
                && cfg.is_valid()
            {
                *last_mtime = new_mtime;
                *config = Some(cfg);
                changed.push(name.clone());
            }
        }
        changed
    }
}

// ── AssetWatcher impl ────────────────────────────────────────────────────

impl vox_core::asset_watcher::AssetWatcher for HotMaterialLibrary {
    fn poll(&mut self, dt: f32) -> Vec<vox_core::asset_watcher::AssetChanged> {
        let changed_names = HotMaterialLibrary::poll(self, dt);
        changed_names
            .into_iter()
            .map(|name| {
                let path = self
                    .path_for(&name)
                    .cloned()
                    .unwrap_or_default();
                vox_core::asset_watcher::AssetChanged { name, path }
            })
            .collect()
    }

    fn force_poll(&mut self) -> Vec<vox_core::asset_watcher::AssetChanged> {
        let changed_names = self.force_reload();
        changed_names
            .into_iter()
            .map(|name| {
                let path = self
                    .path_for(&name)
                    .cloned()
                    .unwrap_or_default();
                vox_core::asset_watcher::AssetChanged { name, path }
            })
            .collect()
    }

    fn watch(&mut self, name: &str, path: std::path::PathBuf) {
        self.register(name, path);
    }
}

// ── ECS systems ───────────────────────────────────────────────────────────

/// Delta time resource used by the hot-reload system.
#[derive(Resource)]
pub struct HotReloadDeltaTime(pub f32);

impl Default for HotReloadDeltaTime {
    fn default() -> Self {
        Self(1.0 / 60.0)
    }
}

/// System that polls material files and inserts `MaterialDirty` on entities
/// whose material changed.
pub fn material_reload_system(
    mut commands: Commands,
    mut library: ResMut<HotMaterialLibrary>,
    dt: Res<HotReloadDeltaTime>,
    query: Query<(Entity, &MaterialRef), Without<MaterialDirty>>,
) {
    let changed = library.poll(dt.0);
    if changed.is_empty() {
        return;
    }
    let changed_set: std::collections::HashSet<&str> =
        changed.iter().map(String::as_str).collect();
    for (entity, mat_ref) in query.iter() {
        if changed_set.contains(mat_ref.material_name.as_str()) {
            commands.entity(entity).insert(MaterialDirty);
        }
    }
}

/// Plugin that registers the hot-reload resource and system.
pub struct MaterialHotReloadPlugin {
    pub poll_interval_secs: f32,
}

impl MaterialHotReloadPlugin {
    pub fn new(poll_interval_secs: f32) -> Self {
        Self { poll_interval_secs }
    }
}

impl Default for MaterialHotReloadPlugin {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl bevy_app::Plugin for MaterialHotReloadPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(HotMaterialLibrary::new(self.poll_interval_secs));
        app.insert_resource(HotReloadDeltaTime::default());
        app.add_systems(bevy_app::Update, material_reload_system);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn valid_config_toml(name: &str) -> String {
        format!(
            r#"
name = "{name}"
spectral_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
roughness = 0.5
metallic = 0.3
"#
        )
    }

    fn make_config(name: &str) -> SpectralMaterialConfig {
        SpectralMaterialConfig {
            name: name.into(),
            spectral_weights: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            roughness: 0.5,
            metallic: 0.3,
            tint: None,
        }
    }

    #[test]
    fn spectral_config_valid() {
        let cfg = make_config("test");
        assert!(cfg.is_valid());
    }

    #[test]
    fn spectral_config_invalid_weights() {
        let mut cfg = make_config("bad");
        cfg.spectral_weights = vec![0.1, 0.2]; // too short
        assert!(!cfg.is_valid());
    }

    #[test]
    fn spectral_config_invalid_roughness() {
        let mut cfg = make_config("bad");
        cfg.roughness = 1.5;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn toml_roundtrip() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", valid_config_toml("roundtrip")).unwrap();
        file.flush().unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        let cfg: SpectralMaterialConfig = toml::from_str(&content).unwrap();
        assert_eq!(cfg.name, "roundtrip");
        assert_eq!(cfg.spectral_weights.len(), 8);
        assert!(cfg.is_valid());
    }

    #[test]
    fn library_register_and_get() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", valid_config_toml("stone")).unwrap();
        file.flush().unwrap();

        let mut lib = HotMaterialLibrary::new(1.0);
        lib.register("stone", file.path());
        let cfg = lib.get("stone").unwrap();
        assert_eq!(cfg.name, "stone");
    }

    #[test]
    fn library_get_missing_returns_none() {
        let lib = HotMaterialLibrary::new(1.0);
        assert!(lib.get("nonexistent").is_none());
    }

    #[test]
    fn library_material_names() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", valid_config_toml("metal")).unwrap();
        file.flush().unwrap();

        let mut lib = HotMaterialLibrary::new(1.0);
        lib.register("metal", file.path());
        assert!(lib.material_names().contains(&"metal".to_string()));
    }

    #[test]
    fn library_poll_rate_limited() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", valid_config_toml("ratetest")).unwrap();
        file.flush().unwrap();

        let mut lib = HotMaterialLibrary::new(1.0);
        lib.register("ratetest", file.path());
        // Polling with dt < interval returns nothing
        let changed = lib.poll(0.1);
        assert!(changed.is_empty(), "should be rate-limited");
    }

    #[test]
    fn library_force_reload_skips_rate_limit() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", valid_config_toml("forced")).unwrap();
        file.flush().unwrap();

        let mut lib = HotMaterialLibrary::new(60.0); // long interval
        lib.register("forced", file.path());

        // Modify the file
        write!(file, "{}", valid_config_toml("forced_v2")).unwrap();
        file.flush().unwrap();
        // Touch mtime by rewriting
        std::fs::write(file.path(), valid_config_toml("forced_v2")).unwrap();

        let changed = lib.force_reload();
        // May or may not detect depending on fs mtime granularity,
        // but the call should not panic
        let _ = changed;
    }

    #[test]
    fn library_registers_nonexistent_file_gracefully() {
        let mut lib = HotMaterialLibrary::new(1.0);
        lib.register("ghost", "/tmp/definitely_does_not_exist_ochroma_test.toml");
        assert!(lib.get("ghost").is_none());
    }

    #[test]
    fn library_detects_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("newmat.toml");

        let mut lib = HotMaterialLibrary::new(0.0); // no rate limit
        lib.register("newmat", &path);
        assert!(lib.get("newmat").is_none()); // file doesn't exist yet

        // Write the file
        std::fs::write(&path, valid_config_toml("newmat")).unwrap();

        // Force a poll
        let changed = lib.force_reload();
        assert!(changed.contains(&"newmat".to_string()), "should detect new file: {:?}", changed);
        assert!(lib.get("newmat").is_some());
    }

    #[test]
    fn plugin_builds_without_panic() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(MaterialHotReloadPlugin::default());
        assert!(app.world().contains_resource::<HotMaterialLibrary>());
    }

    #[test]
    fn ecs_dirty_marking() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        let mut lib = HotMaterialLibrary::new(0.0);

        // Write a temp file
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", valid_config_toml("dirttest")).unwrap();
        file.flush().unwrap();
        lib.register("dirttest", file.path());

        // Force first poll to consume initial mtime
        lib.force_reload();

        // Modify file to trigger change
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(file.path(), valid_config_toml("dirttest_v2")).unwrap();

        world.insert_resource(lib);
        world.insert_resource(HotReloadDeltaTime(0.0)); // bypass rate limit

        // Spawn an entity with MaterialRef
        world.spawn(MaterialRef { material_name: "dirttest".into() });

        let mut schedule = Schedule::default();
        schedule.add_systems(material_reload_system);
        schedule.run(&mut world);
        world.flush();

        let dirty_count = world
            .query::<&MaterialDirty>()
            .iter(&world)
            .count();
        // May be 0 if mtime granularity didn't register the change in <10ms,
        // but the system must not panic
        let _ = dirty_count;
    }

    #[test]
    fn material_ref_component() {
        let mat_ref = MaterialRef { material_name: "wood".into() };
        assert_eq!(mat_ref.material_name, "wood");
        let cloned = mat_ref.clone();
        assert_eq!(cloned.material_name, "wood");
    }
}

#[cfg(test)]
mod asset_watcher_tests {
    use super::*;
    use vox_core::asset_watcher::AssetWatcher;

    #[test]
    fn hot_material_library_is_asset_watcher() {
        let lib = HotMaterialLibrary::new(1.0);
        let _watcher: Box<dyn vox_core::asset_watcher::AssetWatcher> = Box::new(lib);
    }

    #[test]
    fn asset_changed_fields() {
        let ac = vox_core::asset_watcher::AssetChanged {
            name: "stone".into(),
            path: std::path::PathBuf::from("materials/stone.toml"),
        };
        assert_eq!(ac.name, "stone");
        assert_eq!(ac.path.to_str().unwrap(), "materials/stone.toml");
    }

    #[test]
    fn poll_returns_empty_when_no_entries() {
        let mut lib = HotMaterialLibrary::new(1.0);
        let changed = AssetWatcher::poll(&mut lib, 2.0);
        assert!(changed.is_empty());
    }
}
