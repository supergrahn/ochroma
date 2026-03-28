use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, PartialEq)]
pub enum HotReloadEvent {
    ScriptChanged { path: PathBuf },
    AssetChanged { path: PathBuf },
    MapChanged { path: PathBuf },
}

pub struct HotReloadManager {
    watched_files: HashMap<PathBuf, SystemTime>,
    watch_directories: Vec<PathBuf>,
    poll_interval: f32,
    accumulator: f32,
    enabled: bool,
}

impl HotReloadManager {
    pub fn new(poll_interval: f32) -> Self {
        Self {
            watched_files: HashMap::new(),
            watch_directories: Vec::new(),
            poll_interval,
            accumulator: 0.0,
            enabled: true,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn watch_file(&mut self, path: PathBuf) {
        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.watched_files.insert(path, modified);
    }

    pub fn watch_directory(&mut self, path: PathBuf) {
        // Also scan existing files in the directory
        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.flatten() {
                let file_path = entry.path();
                if file_path.is_file() && !self.watched_files.contains_key(&file_path) {
                    let modified = std::fs::metadata(&file_path)
                        .and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    self.watched_files.insert(file_path, modified);
                }
            }
        }
        self.watch_directories.push(path);
    }

    pub fn unwatch(&mut self, path: &Path) {
        self.watched_files.remove(path);
        self.watch_directories.retain(|d| d != path);
    }

    /// Check for changes. Call every frame with dt.
    /// Returns events for files that changed since last check.
    pub fn poll(&mut self, dt: f32) -> Vec<HotReloadEvent> {
        self.accumulator += dt;
        if self.accumulator < self.poll_interval || !self.enabled {
            return Vec::new();
        }
        self.accumulator = 0.0;

        let mut events = Vec::new();

        // Check watched files for modifications
        for (path, last_modified) in &mut self.watched_files {
            if let Ok(metadata) = std::fs::metadata(path)
                && let Ok(modified) = metadata.modified()
                    && modified > *last_modified {
                        *last_modified = modified;
                        events.push(classify_reload_event(path));
                    }
        }

        // Check watched directories for new files
        let mut new_files = Vec::new();
        for dir in &self.watch_directories {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && !self.watched_files.contains_key(&path)
                        && let Ok(metadata) = std::fs::metadata(&path)
                            && let Ok(modified) = metadata.modified() {
                                new_files.push((path.clone(), modified));
                                events.push(classify_reload_event(&path));
                            }
                }
            }
        }

        for (path, modified) in new_files {
            self.watched_files.insert(path, modified);
        }

        events
    }

    pub fn watched_count(&self) -> usize {
        self.watched_files.len()
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

fn classify_reload_event(path: &Path) -> HotReloadEvent {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rhai") => HotReloadEvent::ScriptChanged {
            path: path.to_path_buf(),
        },
        Some("ochroma_map") => HotReloadEvent::MapChanged {
            path: path.to_path_buf(),
        },
        _ => HotReloadEvent::AssetChanged {
            path: path.to_path_buf(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread;
    use std::time::Duration;

    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "hot_reload_test_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn unchanged_file_produces_no_events() {
        let dir = tmp_dir();
        let file = dir.join("test.ply");
        fs::write(&file, "data").unwrap();

        let mut mgr = HotReloadManager::new(0.0);
        mgr.watch_file(file.clone());

        // First poll should find no changes (timestamp matches)
        let events = mgr.poll(1.0);
        assert!(events.is_empty(), "unchanged file should produce no events");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn modified_file_produces_event() {
        let dir = tmp_dir();
        let file = dir.join("test.ply");
        fs::write(&file, "data").unwrap();

        let mut mgr = HotReloadManager::new(0.0);
        mgr.watch_file(file.clone());

        // Drain initial state
        mgr.poll(1.0);

        // Wait briefly to ensure filesystem timestamp changes
        thread::sleep(Duration::from_millis(50));
        fs::write(&file, "modified data").unwrap();

        let events = mgr.poll(1.0);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            HotReloadEvent::AssetChanged {
                path: file.clone()
            }
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn poll_interval_respected() {
        let dir = tmp_dir();
        let file = dir.join("test.rhai");
        fs::write(&file, "script").unwrap();

        let mut mgr = HotReloadManager::new(1.0); // 1 second interval
        mgr.watch_file(file.clone());

        // dt too small, should not poll
        thread::sleep(Duration::from_millis(50));
        fs::write(&file, "modified").unwrap();

        let events = mgr.poll(0.1);
        assert!(events.is_empty(), "should not poll before interval elapsed");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_file_in_watched_directory_detected() {
        let dir = tmp_dir();
        fs::create_dir_all(&dir).unwrap();

        let mut mgr = HotReloadManager::new(0.0);
        mgr.watch_directory(dir.clone());

        // Drain initial state
        mgr.poll(1.0);

        // Create a new file
        let new_file = dir.join("new_asset.ply");
        fs::write(&new_file, "new data").unwrap();

        let events = mgr.poll(1.0);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            HotReloadEvent::AssetChanged {
                path: new_file.clone()
            }
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn classify_rhai_as_script_changed() {
        let path = PathBuf::from("scripts/player.rhai");
        let event = classify_reload_event(&path);
        assert_eq!(
            event,
            HotReloadEvent::ScriptChanged {
                path: path.clone()
            }
        );
    }

    #[test]
    fn classify_ply_as_asset_changed() {
        let path = PathBuf::from("assets/model.ply");
        let event = classify_reload_event(&path);
        assert_eq!(
            event,
            HotReloadEvent::AssetChanged {
                path: path.clone()
            }
        );
    }

    #[test]
    fn classify_ochroma_map_as_map_changed() {
        let path = PathBuf::from("maps/level1.ochroma_map");
        let event = classify_reload_event(&path);
        assert_eq!(
            event,
            HotReloadEvent::MapChanged {
                path: path.clone()
            }
        );
    }

    #[test]
    fn disabled_manager_produces_no_events() {
        let dir = tmp_dir();
        let file = dir.join("test.ply");
        fs::write(&file, "data").unwrap();

        let mut mgr = HotReloadManager::new(0.0);
        mgr.watch_file(file.clone());
        mgr.set_enabled(false);

        thread::sleep(Duration::from_millis(50));
        fs::write(&file, "modified").unwrap();

        let events = mgr.poll(1.0);
        assert!(events.is_empty(), "disabled manager should produce no events");
        assert!(!mgr.is_enabled());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn watch_unwatch_changes_count() {
        let dir = tmp_dir();
        let file1 = dir.join("a.ply");
        let file2 = dir.join("b.ply");
        fs::write(&file1, "a").unwrap();
        fs::write(&file2, "b").unwrap();

        let mut mgr = HotReloadManager::new(0.0);
        assert_eq!(mgr.watched_count(), 0);

        mgr.watch_file(file1.clone());
        assert_eq!(mgr.watched_count(), 1);

        mgr.watch_file(file2.clone());
        assert_eq!(mgr.watched_count(), 2);

        mgr.unwatch(&file1);
        assert_eq!(mgr.watched_count(), 1);

        let _ = fs::remove_dir_all(&dir);
    }
}
