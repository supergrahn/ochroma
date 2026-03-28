use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

/// Watches registered file paths for modifications via polling.
pub struct AssetWatcher {
    watched_paths: HashMap<PathBuf, SystemTime>,
    poll_interval_secs: f32,
    accumulator: f32,
}

impl AssetWatcher {
    pub fn new(poll_interval_secs: f32) -> Self {
        Self {
            watched_paths: HashMap::new(),
            poll_interval_secs,
            accumulator: 0.0,
        }
    }

    /// Register a path to watch. Records the current modification time.
    pub fn watch(&mut self, path: PathBuf) {
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.watched_paths.insert(path, mtime);
    }

    /// Unregister a path.
    pub fn unwatch(&mut self, path: &PathBuf) {
        self.watched_paths.remove(path);
    }

    /// Returns the number of currently watched paths.
    pub fn watched_count(&self) -> usize {
        self.watched_paths.len()
    }

    /// Check for changes. Returns paths that changed since last check.
    /// Accumulates `dt` until the poll interval is reached.
    pub fn poll(&mut self, dt: f32) -> Vec<PathBuf> {
        self.accumulator += dt;
        if self.accumulator < self.poll_interval_secs {
            return Vec::new();
        }
        self.accumulator = 0.0;

        let mut changed = Vec::new();
        for (path, last_modified) in &mut self.watched_paths {
            if let Ok(metadata) = std::fs::metadata(path)
                && let Ok(modified) = metadata.modified()
                    && modified > *last_modified {
                        *last_modified = modified;
                        changed.push(path.clone());
                    }
        }
        changed
    }
}
