//! Generic asset hot-reload interface.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AssetChanged {
    pub name: String,
    pub path: PathBuf,
}

pub trait AssetWatcher {
    /// Poll for changed assets. Rate-limited internally.
    fn poll(&mut self, dt: f32) -> Vec<AssetChanged>;

    /// Force a poll regardless of rate limit.
    fn force_poll(&mut self) -> Vec<AssetChanged>;

    /// Register a file path to watch under a given name.
    fn watch(&mut self, name: &str, path: PathBuf);
}
