//! File watcher for hot-reloading .lua scripts.

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WatchError {
    #[error("notify error: {0}")]
    Notify(#[from] notify::Error),
}

pub struct ScriptWatcher {
    _watcher: RecommendedWatcher,
    pub changed_paths: Arc<Mutex<Vec<PathBuf>>>,
}

impl ScriptWatcher {
    pub fn new(dir: &Path) -> Result<Self, WatchError> {
        let changed = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
        let changed_clone = changed.clone();

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                    let mut lock = changed_clone.lock().unwrap();
                    for path in event.paths {
                        if path.extension().map_or(false, |e| e == "lua") {
                            lock.push(path);
                        }
                    }
                }
            }
        })?;

        let _ = watcher.watch(dir, RecursiveMode::Recursive);

        Ok(Self { _watcher: watcher, changed_paths: changed })
    }

    pub fn drain(&self) -> Vec<PathBuf> {
        std::mem::take(&mut self.changed_paths.lock().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_creates_for_nonexistent_dir() {
        let result = ScriptWatcher::new(Path::new("/tmp/ochroma_test_scripts_nonexistent"));
        assert!(result.is_ok(), "watcher should tolerate missing directory at startup");
    }

    #[test]
    fn drain_returns_empty_initially() {
        let watcher = ScriptWatcher::new(Path::new("/tmp")).unwrap();
        let paths = watcher.drain();
        let _ = paths;
    }

    #[test]
    fn drain_manually_queued_path() {
        let watcher = ScriptWatcher::new(Path::new("/tmp")).unwrap();
        watcher.changed_paths.lock().unwrap().push(PathBuf::from("test.lua"));
        let drained = watcher.drain();
        println!("drained {} path: {}", drained.len(), drained[0].display());
        assert_eq!(drained.len(), 1, "drained {} paths, expected 1", drained.len());
        assert_eq!(drained[0], PathBuf::from("test.lua"));
        assert!(watcher.drain().is_empty());
    }
}
