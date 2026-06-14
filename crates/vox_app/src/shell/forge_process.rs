//! Process boundary for driving standalone Forge as an asset-factory plugin.
//!
//! This bridge intentionally talks JSON over `forge` instead of
//! linking Forge crates. The editor can discover generator metadata with
//! `manifest`, then request an artifact with `run`; Forge decides how to build
//! it, and Ochroma decides how to import the returned artifact.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_FORGE_BINARY: &str = "forge";
const FORGE_BINARY_ENV: &str = "OCHROMA_FORGE_BIN";

#[derive(Debug, Clone)]
pub struct ForgeProcessHost {
    binary: PathBuf,
}

impl ForgeProcessHost {
    pub fn from_env() -> Self {
        let binary = std::env::var_os(FORGE_BINARY_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_FORGE_BINARY));
        Self { binary }
    }

    pub fn with_binary(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
        }
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }

    pub fn manifest(&self) -> Result<ForgeManifest, ForgeProcessError> {
        let stdout = self.run_command(["manifest"])?;
        serde_json::from_slice(&stdout).map_err(|source| ForgeProcessError::Json {
            context: "manifest".to_string(),
            source,
        })
    }

    pub fn run(
        &self,
        command: impl Into<String>,
        params: Value,
    ) -> Result<ForgeArtifact, ForgeProcessError> {
        let request = ForgeRunRequest {
            command: command.into(),
            params,
        };
        let request_json =
            serde_json::to_string(&request).map_err(|source| ForgeProcessError::Json {
                context: "run request".to_string(),
                source,
            })?;
        let stdout = self.run_command(["run", request_json.as_str()])?;
        serde_json::from_slice(&stdout).map_err(|source| ForgeProcessError::Json {
            context: "run artifact".to_string(),
            source,
        })
    }

    fn run_command<I, S>(&self, args: I) -> Result<Vec<u8>, ForgeProcessError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let output = Command::new(&self.binary)
            .args(args)
            .output()
            .map_err(|source| ForgeProcessError::Spawn {
                binary: self.binary.clone(),
                source,
            })?;

        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(ForgeProcessError::Failed {
                binary: self.binary.clone(),
                code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        }
    }
}

impl Default for ForgeProcessHost {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ForgeManifest {
    pub protocol_version: u32,
    pub generators: Vec<ForgeGenerator>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ForgeGenerator {
    pub command: String,
    pub title: String,
    pub category: String,
    pub output: String,
    #[serde(default)]
    pub params_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ForgeRunRequest {
    command: String,
    params: Value,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ForgeArtifact {
    pub command: String,
    pub output: String,
    pub payload: Value,
}

#[derive(Debug)]
pub enum ForgeProcessError {
    Spawn {
        binary: PathBuf,
        source: std::io::Error,
    },
    Failed {
        binary: PathBuf,
        code: Option<i32>,
        stderr: String,
    },
    Json {
        context: String,
        source: serde_json::Error,
    },
}

impl fmt::Display for ForgeProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { binary, source } => {
                write!(
                    f,
                    "failed to start Forge binary {}: {source}",
                    binary.display()
                )
            }
            Self::Failed {
                binary,
                code,
                stderr,
            } => {
                write!(
                    f,
                    "Forge binary {} exited with code {:?}: {}",
                    binary.display(),
                    code,
                    stderr
                )
            }
            Self::Json { context, source } => {
                write!(f, "failed to parse Forge {context}: {source}")
            }
        }
    }
}

impl std::error::Error for ForgeProcessError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_shape_matches_forge_plugin_contract() {
        let manifest: ForgeManifest = serde_json::from_value(serde_json::json!({
            "protocol_version": 1,
            "generators": [{
                "command": "building",
                "title": "Building",
                "category": "spatial",
                "output": "volume",
                "params_schema": { "type": "object" }
            }]
        }))
        .unwrap();

        assert_eq!(manifest.protocol_version, 1);
        assert_eq!(manifest.generators[0].command, "building");
        assert_eq!(manifest.generators[0].output, "volume");
    }

    #[test]
    fn run_request_serializes_to_contract_shape() {
        let request = ForgeRunRequest {
            command: "building".to_string(),
            params: serde_json::json!({ "seed": 7 }),
        };

        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["command"], "building");
        assert_eq!(json["params"]["seed"], 7);
    }

    #[test]
    fn artifact_shape_matches_forge_plugin_contract() {
        let artifact: ForgeArtifact = serde_json::from_value(serde_json::json!({
            "command": "building",
            "output": "volume",
            "payload": { "label": "building" }
        }))
        .unwrap();

        assert_eq!(artifact.command, "building");
        assert_eq!(artifact.payload["label"], "building");
    }
}
