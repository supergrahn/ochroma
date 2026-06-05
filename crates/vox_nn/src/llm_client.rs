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
        }
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
