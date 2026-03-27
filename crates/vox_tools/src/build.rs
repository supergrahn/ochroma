use serde::{Serialize, Deserialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildTarget {
    Windows,
    Linux,
    MacOS,
    SteamOS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildConfig {
    Debug,      // full debug info, assertions
    Release,    // optimised, no debug info
    Shipping,   // fully optimised, stripped, with Steam integration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildManifest {
    pub game_name: String,
    pub version: String,
    pub target: BuildTarget,
    pub config: BuildConfig,
    pub entry_scene: String,
    pub assets_dir: PathBuf,
    pub output_dir: PathBuf,
    pub icon: Option<PathBuf>,
    pub features: Vec<String>,
}

impl BuildManifest {
    pub fn new(game_name: &str, target: BuildTarget, config: BuildConfig) -> Self {
        Self {
            game_name: game_name.to_string(),
            version: "0.1.0".to_string(),
            target,
            config,
            entry_scene: "scenes/main.ochroma_scene".to_string(),
            assets_dir: PathBuf::from("assets"),
            output_dir: PathBuf::from("build"),
            icon: None,
            features: Vec::new(),
        }
    }

    pub fn output_binary_name(&self) -> String {
        match self.target {
            BuildTarget::Windows => format!("{}.exe", self.game_name),
            _ => self.game_name.clone(),
        }
    }

    pub fn cargo_target_triple(&self) -> &'static str {
        match self.target {
            BuildTarget::Windows => "x86_64-pc-windows-msvc",
            BuildTarget::Linux | BuildTarget::SteamOS => "x86_64-unknown-linux-gnu",
            BuildTarget::MacOS => "aarch64-apple-darwin",
        }
    }

    pub fn cargo_profile(&self) -> &'static str {
        match self.config {
            BuildConfig::Debug => "dev",
            BuildConfig::Release | BuildConfig::Shipping => "release",
        }
    }

    /// Generate the cargo build command.
    pub fn build_command(&self) -> String {
        format!(
            "cargo build --target {} --profile {} -p vox_app",
            self.cargo_target_triple(),
            self.cargo_profile(),
        )
    }
}
