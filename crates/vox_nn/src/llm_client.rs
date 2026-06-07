use serde::{Deserialize, Serialize};

/// Building vocabulary selected by prompt keywords. Fixes the rule names used
/// in the deterministic-stub layout so keyword-based callers stay stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildStyle {
    Victorian,
    Modern,
    Generic,
}

/// Tiny deterministic xorshift64 PRNG used only by the deterministic-stub
/// layout generator. Reproducible across runs and platforms for a given seed.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero state.
        Self { state: seed | 1 }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmProvider {
    /// Local Ollama server.
    Ollama { url: String, model: String },
    /// OpenAI-compatible server running on THIS box's GPU — llama.cpp's
    /// `llama-server` (default :8080) or Ollama's `/v1` API. "Use local GPU":
    /// inference runs on the local hardware GPU, never a remote box. The `model`
    /// field is for provenance/labelling; `llama-server` serves whatever model
    /// it has loaded. Real HTTP is implemented under the `local-llm` feature; the
    /// default build falls back to the labelled deterministic stub (no network).
    LocalServer { url: String, model: String },
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

impl LlmProvider {
    /// The local-GPU LLM: an OpenAI-compatible server on loopback. Defaults to
    /// the `llama-server` port (8080) and a provenance label; override the
    /// endpoint/model with `OCHROMA_LLM_URL` / `OCHROMA_LLM_MODEL`.
    pub fn local_gpu() -> Self {
        let url = std::env::var("OCHROMA_LLM_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
        let model = std::env::var("OCHROMA_LLM_MODEL").unwrap_or_else(|_| "local-gpu".to_string());
        Self::LocalServer { url, model }
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

    /// Send a prompt to the LLM.
    ///
    /// DEPTH LIMIT (honest): no real model is invoked. The [`LlmProvider::Ollama`]
    /// HTTP path is not implemented; it falls back to a DETERMINISTIC STUB that
    /// derives a reproducible layout from a hash of the prompt. Responses are
    /// labeled `model = "deterministic-stub"` so callers never mistake them for
    /// real inference. The same prompt always yields the same layout; different
    /// prompts yield different layouts.
    pub fn complete(&self, prompt: &LlmPrompt) -> Result<LlmResponse, String> {
        match &self.provider {
            LlmProvider::Mock => self.deterministic_complete(prompt),
            LlmProvider::Ollama { url, model: _ } => {
                // Real HTTP POST to the Ollama API is not implemented in this
                // build. Fall back to the deterministic stub so behaviour is
                // reproducible and clearly labeled (not silent fake inference).
                eprintln!(
                    "[ochroma-llm] Ollama at {url} not reachable in this build; \
                     using deterministic-stub layout (reproducible, NOT real inference)"
                );
                self.deterministic_complete(prompt)
            }
            LlmProvider::LocalServer { url, model } => {
                self.local_server_complete(url, model, prompt)
            }
        }
    }

    /// Real local-GPU inference path (feature `local-llm`): POST the prompt to
    /// an OpenAI-compatible server on loopback and return its completion. On any
    /// failure (server down, bad response, timeout) it falls back to the labelled
    /// deterministic stub — never silent fake inference, never a UI hang beyond
    /// the timeout. The schema-validated `IntentAction` seam downstream means an
    /// unreachable/garbage model output can never reach the scene graph.
    #[cfg(feature = "local-llm")]
    fn local_server_complete(
        &self,
        url: &str,
        model: &str,
        prompt: &LlmPrompt,
    ) -> Result<LlmResponse, String> {
        match self.try_local_server(url, model, prompt) {
            Ok(resp) => Ok(resp),
            Err(e) => {
                eprintln!(
                    "[ochroma-llm] local GPU server at {url} failed ({e}); \
                     falling back to deterministic stub (NOT real inference)"
                );
                self.deterministic_complete(prompt)
            }
        }
    }

    /// When the `local-llm` feature is OFF, the local server is never contacted —
    /// the default build is hermetic. Clearly labelled so callers know inference
    /// did not run on the GPU.
    #[cfg(not(feature = "local-llm"))]
    fn local_server_complete(
        &self,
        url: &str,
        _model: &str,
        prompt: &LlmPrompt,
    ) -> Result<LlmResponse, String> {
        eprintln!(
            "[ochroma-llm] local-llm feature OFF; {url} not called — using deterministic \
             stub. Rebuild with `--features local-llm` to run inference on the local GPU."
        );
        self.deterministic_complete(prompt)
    }

    /// The actual HTTP round-trip against the OpenAI `/v1/chat/completions` API.
    /// Short connect + overall timeouts so a wedged server can never freeze the
    /// UI thread for more than ~30s; on loopback a real reply lands in well under
    /// a second.
    #[cfg(feature = "local-llm")]
    fn try_local_server(
        &self,
        url: &str,
        model: &str,
        prompt: &LlmPrompt,
    ) -> Result<LlmResponse, String> {
        use std::time::Duration;

        let endpoint = format!("{}/v1/chat/completions", url.trim_end_matches('/'));
        let mut user = prompt.user.clone();
        if let Some(hint) = &prompt.format_hint {
            user.push_str(&format!("\n\nRespond with valid {hint} only. No prose, no code fences."));
        }
        let body = serde_json::json!({
            "model": model,
            "messages": [
                { "role": "system", "content": prompt.system },
                { "role": "user", "content": user },
            ],
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_tokens,
            "stream": false,
        });

        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(2))
            .timeout(Duration::from_secs(30))
            .build();

        let resp = agent
            .post(&endpoint)
            .send_json(body)
            .map_err(|e| format!("request to {endpoint} failed: {e}"))?;
        let v: serde_json::Value = resp
            .into_json()
            .map_err(|e| format!("could not parse response json: {e}"))?;

        let text = v["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| "response missing choices[0].message.content".to_string())?
            .trim()
            .to_string();
        if text.is_empty() {
            return Err("model returned an empty completion".to_string());
        }
        let tokens_used = v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;
        let model_name = v["model"].as_str().unwrap_or(model).to_string();

        Ok(LlmResponse {
            text,
            tokens_used,
            model: model_name,
        })
    }

    /// FNV-1a 64-bit hash of the full prompt. Stable across runs/platforms, so
    /// the generated layout is reproducible for a given prompt string.
    fn prompt_seed(prompt: &LlmPrompt) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut hash = FNV_OFFSET;
        for byte in prompt.system.bytes().chain([0u8]).chain(prompt.user.bytes()) {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn deterministic_complete(&self, prompt: &LlmPrompt) -> Result<LlmResponse, String> {
        let seed = Self::prompt_seed(prompt);

        let text = if prompt.user.contains("rule") || prompt.user.contains("SplatRule") {
            self.mock_splat_rule(&prompt.user)
        } else if prompt.user.contains("modern") || prompt.user.contains("Modern") {
            self.deterministic_layout(seed, BuildStyle::Modern)
        } else if prompt.user.contains("Victorian") || prompt.user.contains("victorian") {
            self.deterministic_layout(seed, BuildStyle::Victorian)
        } else {
            self.deterministic_layout(seed, BuildStyle::Generic)
        };

        Ok(LlmResponse {
            text,
            // Token count is itself seeded so it is reproducible yet prompt-varied.
            tokens_used: 128 + (seed % 256) as u32,
            model: "deterministic-stub".to_string(),
        })
    }

    /// Build a reproducible JSON street layout from a prompt-derived `seed`.
    ///
    /// The seed drives a tiny xorshift PRNG that varies the building count,
    /// per-slot sub-seeds, lateral jitter and wear. The chosen `style` fixes the
    /// rule vocabulary (so existing keyword-based callers/tests still see e.g.
    /// `victorian_terraced`). Same seed -> byte-identical output.
    fn deterministic_layout(&self, seed: u64, style: BuildStyle) -> String {
        let mut rng = XorShift64::new(seed);

        let (width, length, surface, terraced_rule, corner_rule, lamp, bench, tree_rule, base_wear) =
            match style {
                BuildStyle::Victorian => (
                    8.0, 120.0, "cobblestone", "victorian_terraced", "victorian_corner_shop",
                    "lamp_post_gas", "bench_victorian", "oak_mature", 0.30,
                ),
                BuildStyle::Modern => (
                    12.0, 200.0, "asphalt", "modern_apartment", "modern_office",
                    "modern_lamp", "modern_bench", "birch_young", 0.08,
                ),
                BuildStyle::Generic => (
                    10.0, 140.0, "paving", "suburban_house", "suburban_corner_store",
                    "lamp_post", "bench_plain", "maple_mature", 0.20,
                ),
            };

        // 4..=7 buildings, seeded.
        let n_buildings = 4 + (rng.next() % 4) as usize;
        let spacing = (length / (n_buildings as f64 + 1.0)) as f32;

        let mut slots = String::new();
        for i in 0..n_buildings {
            let x = spacing * (i as f32 + 1.0);
            // Lateral side alternates; small seeded jitter on z.
            let side = if i % 2 == 0 { 5.0 } else { -5.0 };
            let z_jitter = (rng.next() % 100) as f32 / 100.0 - 0.5; // [-0.5, 0.5)
            let z = side + z_jitter;
            // Last building on each side becomes the corner rule.
            let rule = if i + 1 == n_buildings { corner_rule } else { terraced_rule };
            let sub_seed = rng.next() % 100_000;
            let wear = (base_wear + ((rng.next() % 30) as f32 / 100.0)).min(0.95);
            if i > 0 {
                slots.push_str(",\n");
            }
            slots.push_str(&format!(
                "    {{ \"position\": [{:.2}, 0.0, {:.2}], \"rule\": \"{}\", \"seed\": {}, \"wear\": {:.2} }}",
                x, z, rule, sub_seed, wear
            ));
        }

        // Two seeded props.
        let p1x = spacing * 0.5 + (rng.next() % 20) as f32 / 10.0;
        let p2x = spacing * 1.5 + (rng.next() % 20) as f32 / 10.0;
        let props = format!(
            "    {{ \"position\": [{p1x:.2}, 0.0, 2.0], \"asset\": \"{lamp}\", \"rotation\": 0.0 }},\n    {{ \"position\": [{p2x:.2}, 0.0, -2.0], \"asset\": \"{bench}\", \"rotation\": 3.14 }}"
        );

        // One seeded tree.
        let tx = spacing * (n_buildings as f32) + (rng.next() % 50) as f32 / 10.0;
        let tree_seed = rng.next() % 100_000;
        let vegetation = format!(
            "    {{ \"position\": [{tx:.2}, 0.0, 0.0], \"rule\": \"{tree_rule}\", \"seed\": {tree_seed} }}"
        );

        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"_note\": \"DETERMINISTIC-STUB layout (not real LLM inference)\",\n");
        out.push_str(&format!("  \"layout_seed\": {seed},\n"));
        out.push_str(&format!(
            "  \"street\": {{ \"width\": {width:.1}, \"length\": {length:.1}, \"orientation_degrees\": 0.0, \"surface\": \"{surface}\" }},\n"
        ));
        out.push_str(&format!("  \"slots\": [\n{slots}\n  ],\n"));
        out.push_str(&format!("  \"props\": [\n{props}\n  ],\n"));
        out.push_str(&format!("  \"vegetation\": [\n{vegetation}\n  ]\n"));
        out.push('}');
        out
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

#[cfg(test)]
mod local_gpu_tests {
    use super::*;

    /// Hermetic: when the `local-llm` feature is OFF (the default test build),
    /// a `LocalServer` provider makes NO network call and returns the labelled
    /// deterministic stub. Guarantees the default build never depends on a server.
    #[cfg(not(feature = "local-llm"))]
    #[test]
    fn local_server_without_feature_falls_back_to_stub() {
        let client = LlmClient::new(LlmProvider::LocalServer {
            url: "http://127.0.0.1:9".to_string(), // discard port — must never be hit
            model: "local-gpu".to_string(),
        });
        let resp = client
            .complete(&LlmPrompt::new("sys", "make a modern street"))
            .expect("stub fallback always succeeds");
        assert_eq!(
            resp.model, "deterministic-stub",
            "feature-off LocalServer must produce the labelled stub, not real inference"
        );
        assert!(
            resp.text.contains("DETERMINISTIC-STUB"),
            "stub text must self-identify"
        );
    }

    /// `local_gpu()` honours the env overrides and defaults to the llama-server
    /// loopback port. Pure config assertion — no network.
    #[test]
    fn local_gpu_defaults_to_loopback_8080() {
        // Only assert the default when the override is unset, to stay hermetic.
        if std::env::var("OCHROMA_LLM_URL").is_err() {
            match LlmProvider::local_gpu() {
                LlmProvider::LocalServer { url, .. } => {
                    assert_eq!(url, "http://127.0.0.1:8080", "default local-GPU endpoint");
                }
                other => panic!("local_gpu() must be a LocalServer, got {other:?}"),
            }
        }
    }

    /// Real local-GPU inference. Network + GPU dependent → `#[ignore]` (run
    /// explicitly with `--features local-llm -- --ignored`). Proves Ask Ochroma's
    /// backend round-trips through the local llama-server: a non-stub model name
    /// and a non-empty completion come back. Mirrors the `#[ignore]` perf-bench
    /// convention — never gates CI on a running server.
    #[cfg(feature = "local-llm")]
    #[test]
    #[ignore]
    fn local_gpu_real_inference_round_trips() {
        let client = LlmClient::new(LlmProvider::local_gpu());
        let prompt = LlmPrompt::new(
            "You are a terse assistant. Answer in one word.",
            "Reply with exactly: hello",
        );
        let resp = client.complete(&prompt).expect("local server reachable");
        eprintln!(
            "[local-gpu] model={} tokens={} text={:?}",
            resp.model, resp.tokens_used, resp.text
        );
        assert_ne!(
            resp.model, "deterministic-stub",
            "expected REAL inference from the local GPU, got the stub — is llama-server up on :8080?"
        );
        assert!(!resp.text.trim().is_empty(), "real completion must be non-empty");
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
