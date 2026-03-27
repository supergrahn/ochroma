use serde::{Deserialize, Serialize};

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmProvider {
    /// Local Ollama server.
    Ollama { url: String, model: String },
    /// Mock provider for testing (returns deterministic output).
    Mock,
}

impl Default for LlmProvider {
    fn default() -> Self {
        Self::Ollama {
            url: "http://localhost:11434".to_string(),
            model: "llama3.2".to_string(),
        }
    }
}

/// A structured prompt for the LLM.
#[derive(Debug, Clone)]
pub struct LlmPrompt {
    pub system: String,
    pub user: String,
    pub format_hint: Option<String>, // e.g. "Respond with valid TOML"
    pub temperature: f32,
    pub max_tokens: u32,
}

impl LlmPrompt {
    pub fn new(system: &str, user: &str) -> Self {
        Self {
            system: system.to_string(),
            user: user.to_string(),
            format_hint: None,
            temperature: 0.3,
            max_tokens: 2048,
        }
    }

    pub fn with_format(mut self, hint: &str) -> Self {
        self.format_hint = Some(hint.to_string());
        self
    }
}

/// LLM response.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub tokens_used: u32,
    pub model: String,
}

/// LLM client that handles communication with the model.
pub struct LlmClient {
    pub provider: LlmProvider,
}

impl LlmClient {
    pub fn new(provider: LlmProvider) -> Self {
        Self { provider }
    }

    /// Send a prompt to the LLM. Currently uses mock for offline development.
    pub fn complete(&self, prompt: &LlmPrompt) -> Result<LlmResponse, String> {
        match &self.provider {
            LlmProvider::Mock => self.mock_complete(prompt),
            LlmProvider::Ollama { url, model: _ } => {
                // In production: HTTP POST to Ollama API
                // For now, fall back to mock if Ollama not available
                eprintln!("[ochroma-llm] Ollama at {} not available, using mock", url);
                self.mock_complete(prompt)
            }
        }
    }

    fn mock_complete(&self, prompt: &LlmPrompt) -> Result<LlmResponse, String> {
        let text = if prompt.user.contains("Victorian") || prompt.user.contains("victorian") {
            self.mock_victorian_layout()
        } else if prompt.user.contains("rule") || prompt.user.contains("SplatRule") {
            self.mock_splat_rule(&prompt.user)
        } else if prompt.user.contains("modern") || prompt.user.contains("Modern") {
            self.mock_modern_layout()
        } else {
            self.mock_generic_layout()
        };

        Ok(LlmResponse {
            text,
            tokens_used: 256,
            model: "mock".to_string(),
        })
    }

    fn mock_victorian_layout(&self) -> String {
        r#"{
  "street": { "width": 8.0, "length": 120.0, "orientation_degrees": 0.0, "surface": "cobblestone" },
  "slots": [
    { "position": [0.0, 0.0, 5.0], "rule": "victorian_terraced", "seed": 1, "wear": 0.3 },
    { "position": [7.0, 0.0, 5.0], "rule": "victorian_terraced", "seed": 2, "wear": 0.4 },
    { "position": [14.0, 0.0, 5.0], "rule": "victorian_terraced", "seed": 3, "wear": 0.2 },
    { "position": [21.0, 0.0, 5.0], "rule": "victorian_corner_shop", "seed": 4, "wear": 0.5 },
    { "position": [0.0, 0.0, -5.0], "rule": "victorian_terraced", "seed": 5, "wear": 0.35 },
    { "position": [7.0, 0.0, -5.0], "rule": "victorian_terraced", "seed": 6, "wear": 0.45 }
  ],
  "props": [
    { "position": [3.5, 0.0, 2.0], "asset": "lamp_post_gas", "rotation": 0.0 },
    { "position": [10.5, 0.0, 2.0], "asset": "lamp_post_gas", "rotation": 0.0 },
    { "position": [17.5, 0.0, -2.0], "asset": "bench_victorian", "rotation": 3.14 }
  ],
  "vegetation": [
    { "position": [25.0, 0.0, 0.0], "rule": "oak_mature", "seed": 99 }
  ]
}"#
        .to_string()
    }

    fn mock_modern_layout(&self) -> String {
        r#"{
  "street": { "width": 12.0, "length": 200.0, "orientation_degrees": 0.0, "surface": "asphalt" },
  "slots": [
    { "position": [0.0, 0.0, 8.0], "rule": "modern_apartment", "seed": 10, "wear": 0.1 },
    { "position": [20.0, 0.0, 8.0], "rule": "modern_office", "seed": 11, "wear": 0.05 },
    { "position": [45.0, 0.0, 8.0], "rule": "modern_apartment", "seed": 12, "wear": 0.15 }
  ],
  "props": [
    { "position": [10.0, 0.0, 3.0], "asset": "modern_lamp", "rotation": 0.0 },
    { "position": [30.0, 0.0, 3.0], "asset": "modern_bench", "rotation": 0.0 }
  ],
  "vegetation": [
    { "position": [15.0, 0.0, -3.0], "rule": "birch_young", "seed": 50 },
    { "position": [35.0, 0.0, -3.0], "rule": "birch_young", "seed": 51 }
  ]
}"#
        .to_string()
    }

    fn mock_generic_layout(&self) -> String {
        self.mock_victorian_layout()
    }

    fn mock_splat_rule(&self, prompt: &str) -> String {
        let style = if prompt.contains("tall") {
            "modern"
        } else if prompt.contains("brick") {
            "victorian"
        } else {
            "suburban"
        };

        format!(
            r#"[rule]
asset_type = "House"
style = "{style}"

[geometry]
strategy = "structured_placement"
floor_count_min = 2
floor_count_max = 4
floor_height_min = 3.0
floor_height_max = 3.5
base_width_min = 5.0
base_width_max = 8.0
depth = 12.0

[[materials]]
tag = "facade"
spd = "brick_red"
density = 600.0
scale_min = 0.04
scale_max = 0.08
entity_id_zone = 1

[[materials]]
tag = "roof"
spd = "slate_grey"
density = 400.0
scale_min = 0.05
scale_max = 0.10
entity_id_zone = 2

[variation]
facade_color_shift = 0.1
wear_level_min = 0.0
wear_level_max = 0.4
"#
        )
    }
}

/// Convenience: generate a SceneGraph layout from a text prompt.
pub fn prompt_to_layout(client: &LlmClient, prompt: &str) -> Result<String, String> {
    let llm_prompt = LlmPrompt::new(
        "You are an urban planner AI. Given a description, generate a JSON scene layout with street, building slots, props, and vegetation. Each building slot has position [x,y,z], rule name, seed, and wear level.",
        prompt,
    ).with_format("JSON");

    client.complete(&llm_prompt).map(|r| r.text)
}

/// Convenience: generate a SplatRule TOML from a text description.
pub fn prompt_to_rule(client: &LlmClient, description: &str) -> Result<String, String> {
    let llm_prompt = LlmPrompt::new(
        "You are a procedural building rule generator. Given a building description, generate a SplatRule in TOML format with geometry, materials, and variation parameters.",
        &format!("Generate a SplatRule for: {}", description),
    )
    .with_format("TOML");

    client.complete(&llm_prompt).map(|r| r.text)
}
