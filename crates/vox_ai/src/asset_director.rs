//! AssetDirector — stage machine: TextPrompt → ColmapProcess → LoadVxm → PlaceInScene.
//! Pattern adapted from AetherSpectra Director: resumable, crash-safe, JSON artifacts.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetStageName {
    TextPrompt,
    ColmapProcess,
    LoadVxm,
    PlaceInScene,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetStageArtifact {
    ExpandedPrompt { text: String },
    ColmapOutput { sparse_path: PathBuf, image_count: usize },
    VxmLoaded { vxm_path: PathBuf, splat_count: usize },
    Placed { scene_id: u64, position: [f32; 3] },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetPipelineState {
    pub schema_version: u32,
    pub prompt: String,
    pub completed: Vec<AssetStageName>,
    pub failed_at: Option<AssetStageName>,
    pub artifacts: HashMap<String, String>,
}

impl AssetPipelineState {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            prompt: prompt.into(),
            completed: Vec::new(),
            failed_at: None,
            artifacts: HashMap::new(),
        }
    }

    pub fn save(&self, output_dir: &Path) -> Result<()> {
        std::fs::write(
            output_dir.join("asset_pipeline_state.json"),
            serde_json::to_string_pretty(self)?,
        )?;
        Ok(())
    }

    pub fn load(output_dir: &Path) -> Option<Self> {
        let text =
            std::fs::read_to_string(output_dir.join("asset_pipeline_state.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn is_complete(&self) -> bool {
        self.completed.contains(&AssetStageName::PlaceInScene)
    }

    pub fn mark_complete(&mut self, stage: AssetStageName, artifact: &AssetStageArtifact) -> Result<()> {
        self.artifacts.insert(format!("{:?}", stage), serde_json::to_string(artifact)?);
        self.completed.push(stage);
        self.failed_at = None;
        Ok(())
    }

    pub fn get_artifact(&self, stage: AssetStageName) -> Option<AssetStageArtifact> {
        serde_json::from_str(self.artifacts.get(&format!("{:?}", stage))?).ok()
    }
}

#[derive(Debug, Clone)]
pub struct AssetDirectorConfig {
    pub output_dir: PathBuf,
    pub colmap_bin: PathBuf,
    pub placement_pos: [f32; 3],
}

impl Default for AssetDirectorConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("/tmp/asset_director"),
            colmap_bin: PathBuf::from("colmap"),
            placement_pos: [0.0; 3],
        }
    }
}

pub struct AssetDirector {
    pub config: AssetDirectorConfig,
}

impl AssetDirector {
    pub fn new(config: AssetDirectorConfig) -> Self {
        Self { config }
    }

    pub async fn run(
        &self,
        prompt: &str,
        image_paths: &[PathBuf],
        resume: bool,
    ) -> Result<AssetStageArtifact> {
        std::fs::create_dir_all(&self.config.output_dir)?;
        let mut state =
            if self.config.output_dir.join("asset_pipeline_state.json").exists() {
                if !resume {
                    bail!("Pipeline state exists. Use resume=true to continue.");
                }
                AssetPipelineState::load(&self.config.output_dir)
                    .ok_or_else(|| anyhow::anyhow!("Failed to load pipeline state"))?
            } else {
                AssetPipelineState::new(prompt)
            };

        if !state.completed.contains(&AssetStageName::TextPrompt) {
            let a = AssetStageArtifact::ExpandedPrompt { text: format!("[expanded] {}", prompt) };
            state.mark_complete(AssetStageName::TextPrompt, &a)?;
            state.save(&self.config.output_dir)?;
        }
        if !state.completed.contains(&AssetStageName::ColmapProcess) {
            let sparse_path = self.config.output_dir.join("sparse");
            let a = AssetStageArtifact::ColmapOutput {
                sparse_path,
                image_count: image_paths.len(),
            };
            state.mark_complete(AssetStageName::ColmapProcess, &a)?;
            state.save(&self.config.output_dir)?;
        }
        if !state.completed.contains(&AssetStageName::LoadVxm) {
            let colmap = state
                .get_artifact(AssetStageName::ColmapProcess)
                .ok_or_else(|| anyhow::anyhow!("Missing ColmapProcess artifact"))?;
            let sparse_path = match &colmap {
                AssetStageArtifact::ColmapOutput { sparse_path, .. } => sparse_path.clone(),
                _ => bail!("Expected ColmapOutput"),
            };
            let a = AssetStageArtifact::VxmLoaded {
                vxm_path: sparse_path.with_extension("vxm"),
                splat_count: 0,
            };
            state.mark_complete(AssetStageName::LoadVxm, &a)?;
            state.save(&self.config.output_dir)?;
        }
        if !state.completed.contains(&AssetStageName::PlaceInScene) {
            let a = AssetStageArtifact::Placed {
                scene_id: 0,
                position: self.config.placement_pos,
            };
            state.mark_complete(AssetStageName::PlaceInScene, &a)?;
            state.save(&self.config.output_dir)?;
        }

        state
            .get_artifact(AssetStageName::PlaceInScene)
            .ok_or_else(|| anyhow::anyhow!("Pipeline complete but PlaceInScene artifact missing"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_director(dir: &TempDir) -> AssetDirector {
        AssetDirector::new(AssetDirectorConfig {
            output_dir: dir.path().to_path_buf(),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn pipeline_completes_successfully() {
        let dir = TempDir::new().unwrap();
        let result = make_director(&dir).run("a stone forge", &[], false).await.unwrap();
        assert!(matches!(result, AssetStageArtifact::Placed { .. }), "final artifact must be Placed");
    }

    #[tokio::test]
    async fn pipeline_state_persists_to_disk() {
        let dir = TempDir::new().unwrap();
        make_director(&dir).run("a wooden barrel", &[], false).await.unwrap();
        assert!(
            dir.path().join("asset_pipeline_state.json").exists(),
            "pipeline_state.json must be written to output_dir"
        );
    }

    #[tokio::test]
    async fn pipeline_refuses_rerun_without_resume() {
        let dir = TempDir::new().unwrap();
        make_director(&dir).run("a sword", &[], false).await.unwrap();
        let result = make_director(&dir).run("a sword", &[], false).await;
        assert!(result.is_err(), "re-running without resume must fail");
    }

    #[tokio::test]
    async fn resume_skips_completed_stages() {
        let dir = TempDir::new().unwrap();
        make_director(&dir).run("a lantern", &[], false).await.unwrap();
        let result = make_director(&dir).run("a lantern", &[], true).await;
        assert!(result.is_ok(), "resume on completed pipeline must succeed: {:?}", result);
    }

    #[test]
    fn pipeline_state_round_trips_json() {
        let dir = TempDir::new().unwrap();
        let mut state = AssetPipelineState::new("test prompt");
        let artifact = AssetStageArtifact::ExpandedPrompt { text: "expanded".into() };
        state.mark_complete(AssetStageName::TextPrompt, &artifact).unwrap();
        state.save(dir.path()).unwrap();
        let loaded = AssetPipelineState::load(dir.path()).unwrap();
        assert_eq!(loaded.prompt, "test prompt");
        assert!(loaded.completed.contains(&AssetStageName::TextPrompt));
    }

    #[test]
    fn stage_name_is_complete_after_all_stages() {
        let mut state = AssetPipelineState::new("p");
        for stage in [
            AssetStageName::TextPrompt,
            AssetStageName::ColmapProcess,
            AssetStageName::LoadVxm,
            AssetStageName::PlaceInScene,
        ] {
            let artifact = AssetStageArtifact::ExpandedPrompt { text: "x".into() };
            state.mark_complete(stage, &artifact).unwrap();
        }
        assert!(state.is_complete());
    }
}
