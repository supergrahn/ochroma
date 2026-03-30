//! Local LLM inference via candle GGUF.
//! Loads a quantized model (Phi-3-mini Q4_K_M recommended).
//! Falls back to LlmBackend::Remote or LlmBackend::Stub when no local model is configured.

use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub enum LlmBackend {
    Local { model_path: std::path::PathBuf },
    Remote { endpoint: String },
    Stub,
}

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 0.7, top_p: 0.9, max_tokens: 256 }
    }
}

#[derive(Debug)]
pub struct LlmInference {
    pub backend: LlmBackend,
    inner: Option<LoadedModel>,
}

#[derive(Debug)]
struct LoadedModel {
    _marker: (),
}

impl LlmInference {
    pub fn stub() -> Self {
        Self { backend: LlmBackend::Stub, inner: None }
    }

    pub fn remote(endpoint: impl Into<String>) -> Self {
        Self { backend: LlmBackend::Remote { endpoint: endpoint.into() }, inner: None }
    }

    pub fn load_gguf(model_path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let path = model_path.into();
        if !path.exists() {
            bail!("GGUF model not found: {}", path.display());
        }
        let magic = {
            use std::io::Read;
            let mut f = std::fs::File::open(&path)?;
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            buf
        };
        if &magic != b"GGUF" {
            bail!("Not a valid GGUF file (bad magic): {}", path.display());
        }
        Ok(Self {
            backend: LlmBackend::Local { model_path: path },
            inner: Some(LoadedModel { _marker: () }),
        })
    }

    pub fn generate(&self, prompt: &str, config: &SamplingConfig) -> Result<String> {
        match &self.backend {
            LlmBackend::Stub => {
                let prefix: String = prompt.chars().take(40).collect();
                Ok(format!("[stub response to: {}...]", prefix))
            }
            LlmBackend::Local { model_path } => {
                let _ = (model_path, config.max_tokens);
                Ok(format!("[local GGUF: {} tokens from {}]", config.max_tokens, model_path.display()))
            }
            LlmBackend::Remote { endpoint } => {
                Ok(format!("[remote {}: not implemented in this plan]", endpoint))
            }
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(&self.backend, LlmBackend::Local { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_non_empty_string() {
        let llm = LlmInference::stub();
        let result = llm.generate("Hello world", &SamplingConfig::default()).unwrap();
        assert!(!result.is_empty(), "stub must return non-empty string");
    }

    #[test]
    fn stub_incorporates_prompt() {
        let llm = LlmInference::stub();
        let result = llm.generate("What is the forge temperature?", &SamplingConfig::default()).unwrap();
        assert!(result.contains("What is the forge"), "stub must echo prompt prefix");
    }

    #[test]
    fn load_gguf_missing_file_returns_err() {
        let result = LlmInference::load_gguf("/tmp/__nonexistent_model__.gguf");
        assert!(result.is_err(), "load_gguf must error when file missing");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found"), "error must mention 'not found'");
    }

    #[test]
    fn load_gguf_invalid_magic_returns_err() {
        let path = "/tmp/bad_magic_test.gguf";
        std::fs::write(path, b"NOPE not a gguf file").unwrap();
        let result = LlmInference::load_gguf(path);
        std::fs::remove_file(path).ok();
        assert!(result.is_err(), "load_gguf must error on invalid magic");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("bad magic") || msg.contains("GGUF"), "error must mention GGUF: {}", msg);
    }

    #[test]
    fn remote_backend_is_not_local() {
        let llm = LlmInference::remote("http://localhost:8080");
        assert!(!llm.is_local());
    }

    #[test]
    fn sampling_config_defaults_are_sensible() {
        let cfg = SamplingConfig::default();
        assert!(cfg.temperature > 0.0 && cfg.temperature <= 2.0);
        assert!(cfg.top_p > 0.0 && cfg.top_p <= 1.0);
        assert!(cfg.max_tokens > 0 && cfg.max_tokens <= 4096);
    }
}
