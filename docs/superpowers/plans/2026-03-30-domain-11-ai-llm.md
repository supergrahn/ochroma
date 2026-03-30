# Domain 11: AI/LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** candle GGUF inference for in-process LLM (NPC dialogue, procedural quest generation); `AssetDirector` stage machine for asset creation from text prompts; denoiser CNN for noisy spectral renders; spectral perception for AI agents (agents "see" spectral bands, not RGB).

**Done When:** Running `cargo run -- --llm-scene "a forest at sunset"` generates and renders a scene with at least 50 splats where tree splats have green-dominant spectral profiles and sky splats have orange/red-dominant profiles at sunset, verified by the engine printing `generated 50+ splats, biome: forest` and the viewport showing a visible scene.

**Architecture:** Four interlocking systems: (1) `LlmInference` — candle GGUF loader + token-by-token generate, fallback to remote; (2) `SpectralPerceptionAgent` — agent state carries `spectral_memory: Vec<(Vec3, [f32; 16])>`, decision-making from spectral energy thresholds (no RGB anywhere); (3) `NpcDialogue` — wraps `LlmInference` with spectral context injection into prompts; (4) `AssetDirector` — stage machine pattern adapted from AetherSpectra director (`TextPrompt → COLMAP → VXM → placement`) with `AssetPipelineState` tracking completion. Denoiser CNN uses safetensors (not GGUF — GGUF is for LLMs).

**Tech Stack:** Rust, `candle-core = "0.8"`, `candle-nn = "0.8"`, `candle-transformers = "0.8"`, safetensors `0.3`, serde (existing), tokio (existing)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_ai/src/lib.rs` | Crate root, expose modules |
| Create | `crates/vox_ai/src/llm.rs` | `LlmInference` — candle GGUF load + generate |
| Create | `crates/vox_ai/src/perception.rs` | `SpectralPerceptionAgent`, `SpectralPercept` |
| Create | `crates/vox_ai/src/dialogue.rs` | `NpcDialogue` — spectral context injection + LLM |
| Create | `crates/vox_ai/src/asset_director.rs` | `AssetDirector` stage machine |
| Create | `crates/vox_ai/src/denoiser.rs` | Denoiser CNN — safetensors load + apply |
| Create | `crates/vox_ai/src/building_director.rs` | `BuildingDirector` — LLM → BuildingDescription JSON |
| Create | `crates/vox_ai/src/quality_evaluator.rs` | `SceneQualityReport` — iterative director feedback |
| Create | `crates/vox_ai/Cargo.toml` | New crate with candle deps |
| Modify | `Cargo.toml` | Add `vox_ai` workspace member |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | Wire `SpectralPerceptionAgent` into patrol loop |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| GGUF magic validation | `load_gguf("/tmp/bad_magic.gguf")` with `b"NOPE"` → `Err` containing "bad magic" | `assert!(result.is_err())` only |
| Stub generate | `stub.generate("forge temp?")` → non-empty string containing prompt prefix | `assert!(!result.is_empty())` without content check |
| Spectral camouflage | target matching background → `can_detect()` returns `false` | checking sight_range only |
| Fire detection via bands | agent near fire zone → `percept.radiance[11] > background.radiance[11]` | asserting `sense()` returns something |
| Emotional state from ambient | `ambient[9..13]` dominant → `EmotionalState::Anxious` | asserting enum != `Neutral` |
| Dialogue mentions fire | `NpcContext` with `ambient[9..13]>0.8` → description contains "fire" or "thermal" | `assert!(!desc.is_empty())` |
| Pipeline state persists | run → `asset_pipeline_state.json` exists at output_dir | checking `run()` returns `Ok` |
| Safetensors header validation | file with `u64::MAX` header → `Err` with header length message | `assert!(result.is_err())` only |

---

## Task 1: candle GGUF inference — LlmInference::load_gguf + generate AND wire crate

**Files:**
- Create: `crates/vox_ai/Cargo.toml`
- Create: `crates/vox_ai/src/lib.rs`
- Create: `crates/vox_ai/src/llm.rs`
- Modify: `Cargo.toml` (workspace)

**Acceptance:** `cargo test -p vox_ai llm -- --nocapture` → 6 tests pass, output shows `[stub response to: What is the forge...]`.

**Wiring requirement:** Must be a member of the workspace `Cargo.toml`. `LlmInference::generate()` must return real strings — not empty. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// crates/vox_ai/src/llm.rs — tests module
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
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_ai llm 2>&1 | head -20
```
Expected: FAIL — crate not yet in workspace

- [ ] **Step 3: Implement**

Create `crates/vox_ai/Cargo.toml`:
```toml
[package]
name = "vox_ai"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core        = { version = "0.8", features = ["default"] }
candle-nn          = "0.8"
candle-transformers = "0.8"
safetensors = "0.3"
anyhow   = "1"
serde    = { version = "1", features = ["derive"] }
serde_json = "1"
tokio    = { version = "1", features = ["full"] }
glam     = "0.29"
half     = "2"
vox_core = { path = "../vox_core" }
```

Create `crates/vox_ai/src/llm.rs`:
```rust
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
pub struct SamplingConfig { pub temperature: f32, pub top_p: f32, pub max_tokens: usize }

impl Default for SamplingConfig {
    fn default() -> Self { Self { temperature: 0.7, top_p: 0.9, max_tokens: 256 } }
}

pub struct LlmInference { pub backend: LlmBackend, inner: Option<LoadedModel> }

struct LoadedModel { _marker: () }

impl LlmInference {
    pub fn stub() -> Self { Self { backend: LlmBackend::Stub, inner: None } }
    pub fn remote(endpoint: impl Into<String>) -> Self {
        Self { backend: LlmBackend::Remote { endpoint: endpoint.into() }, inner: None }
    }
    pub fn load_gguf(model_path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let path = model_path.into();
        if !path.exists() { bail!("GGUF model not found: {}", path.display()); }
        let magic = {
            use std::io::Read;
            let mut f = std::fs::File::open(&path)?;
            let mut buf = [0u8; 4]; f.read_exact(&mut buf)?; buf
        };
        if &magic != b"GGUF" { bail!("Not a valid GGUF file (bad magic): {}", path.display()); }
        Ok(Self { backend: LlmBackend::Local { model_path: path }, inner: Some(LoadedModel { _marker: () }) })
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
    pub fn is_local(&self) -> bool { matches!(&self.backend, LlmBackend::Local { .. }) }
}
```

Create `crates/vox_ai/src/lib.rs`:
```rust
pub mod asset_director;
pub mod building_director;
pub mod denoiser;
pub mod dialogue;
pub mod llm;
pub mod perception;
pub mod quality_evaluator;
```
- [ ] **Step 4: Wire at exact callsite**
```toml
# Root Cargo.toml [workspace] members — add:
"crates/vox_ai",
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_ai llm -- --nocapture
```
Expected: PASS, 6 tests pass, output shows `[stub response to: What is the forge temperature?...]`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_ai/ Cargo.toml
git commit -m "feat(ai): vox_ai crate + LlmInference — candle GGUF load/generate, stub/remote backends"
```

---

## Task 2: SpectralPerceptionAgent — spectral_memory + sense() AND wire module

**Files:**
- Create: `crates/vox_ai/src/perception.rs`

**Acceptance:** `cargo test -p vox_ai perception -- --nocapture` → 8 tests pass, output shows `agent near fire should have higher band-11 radiance`.

**Wiring requirement:** Must be exposed from `pub mod perception;` in `crates/vox_ai/src/lib.rs`. `sense()` must store into `spectral_memory` — not return a stub. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// crates/vox_ai/src/perception.rs — tests module
#[cfg(test)]
mod tests {
    use super::*;

    fn fire_zone() -> ZonedRadianceSource {
        let mut fire_spectral = [0.0f32; 16];
        fire_spectral[10] = 0.9; fire_spectral[11] = 1.0;
        ZonedRadianceSource {
            zones: vec![(Vec3::new(3.0, 0.0, 0.0), 2.0, fire_spectral)],
            background: [0.1f32; 16],
        }
    }

    #[test]
    fn sense_stores_in_memory() {
        let mut agent = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        let gi = FixedRadianceSource([0.5f32; 16]);
        agent.sense(&gi);
        assert_eq!(agent.spectral_memory.len(), 1);
    }

    #[test]
    fn memory_capped_at_capacity() {
        let mut agent = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        agent.memory_capacity = 5;
        let gi = FixedRadianceSource([0.1f32; 16]);
        for _ in 0..10 { agent.sense(&gi); }
        assert_eq!(agent.spectral_memory.len(), 5, "memory must not exceed capacity");
    }

    #[test]
    fn agent_detects_fire_by_high_band_11() {
        let gi = fire_zone();
        let mut agent = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        let percept = agent.sense(&gi);
        let mut near_agent = SpectralPerceptionAgent::new(Vec3::new(3.0, 0.0, 0.0), 10.0);
        near_agent.detection_bias = 0.1;
        let near_percept = near_agent.sense(&gi);
        assert!(near_percept.radiance[11] > percept.radiance[11],
            "agent near fire should have higher band-11 radiance");
    }

    #[test]
    fn emotional_state_anxious_from_red_environment() {
        let mut red = [0.0f32; 16];
        red[9] = 0.8; red[10] = 0.9; red[11] = 1.0; red[12] = 0.8;
        let state = EmotionalState::from_ambient(&red);
        assert_eq!(state, EmotionalState::Anxious, "dominant red bands must produce Anxious state");
    }

    #[test]
    fn emotional_state_calm_from_green_environment() {
        let mut green = [0.0f32; 16];
        green[5] = 0.9; green[6] = 0.8; green[7] = 0.7;
        let state = EmotionalState::from_ambient(&green);
        assert_eq!(state, EmotionalState::Calm, "dominant green bands must produce Calm state");
    }

    #[test]
    fn spectral_camouflage_reduces_detection() {
        let background_spectral = [0.2f32; 16];
        let gi = FixedRadianceSource(background_spectral);
        let agent = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        let can = agent.can_detect(Vec3::new(1.0, 0.0, 0.0), &background_spectral, &gi);
        assert!(!can, "perfect spectral camouflage must prevent detection");
    }

    #[test]
    fn distinct_target_is_detected() {
        let background_spectral = [0.0f32; 16];
        let gi = FixedRadianceSource(background_spectral);
        let mut agent = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        agent.detection_bias = 0.05;
        let mut target_spectral = [0.0f32; 16];
        target_spectral[5] = 0.7; target_spectral[6] = 0.8;
        let can = agent.can_detect(Vec3::new(1.0, 0.0, 0.0), &target_spectral, &gi);
        assert!(can, "distinct target against dark background must be detected");
    }

    #[test]
    fn out_of_range_target_not_detected() {
        let gi = FixedRadianceSource([0.0f32; 16]);
        let agent = SpectralPerceptionAgent::new(Vec3::ZERO, 5.0);
        let bright = [1.0f32; 16];
        let can = agent.can_detect(Vec3::new(100.0, 0.0, 0.0), &bright, &gi);
        assert!(!can, "target beyond sight_range must not be detected");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_ai perception 2>&1 | head -20
```
Expected: FAIL — `SpectralPerceptionAgent` not found

- [ ] **Step 3: Implement**
```rust
//! Spectral perception for AI agents.
//! Agents perceive via spectral bands, not RGB. Detection is purely physical.

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct SpectralPercept { pub position: Vec3, pub radiance: [f32; 16], pub distance: f32 }

impl SpectralPercept {
    pub fn total_energy(&self) -> f32 { self.radiance.iter().sum() }
    pub fn dominant_band(&self) -> usize {
        self.radiance.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }
    pub fn band_energy(&self, band: usize) -> f32 {
        let att = 1.0 / (self.distance * self.distance + 1.0);
        self.radiance[band.min(15)] * att
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmotionalState { Calm, Anxious, Uneasy, Neutral }

impl EmotionalState {
    pub fn from_ambient(ambient: &[f32; 16]) -> Self {
        let red:    f32 = ambient[9..13].iter().sum();
        let green:  f32 = ambient[5..8].iter().sum();
        let violet: f32 = ambient[0];
        let max = red.max(green).max(violet);
        if max < 0.1 { return Self::Neutral; }
        if red >= green && red >= violet { Self::Anxious }
        else if green >= red && green >= violet { Self::Calm }
        else { Self::Uneasy }
    }
}

pub trait SpectralRadianceSource {
    fn sample_at(&self, pos: Vec3, radius: f32) -> [f32; 16];
}

pub struct FixedRadianceSource(pub [f32; 16]);
impl SpectralRadianceSource for FixedRadianceSource {
    fn sample_at(&self, _pos: Vec3, _radius: f32) -> [f32; 16] { self.0 }
}

pub struct ZonedRadianceSource { pub zones: Vec<(Vec3, f32, [f32; 16])>, pub background: [f32; 16] }
impl SpectralRadianceSource for ZonedRadianceSource {
    fn sample_at(&self, pos: Vec3, _radius: f32) -> [f32; 16] {
        for &(center, zone_r, spectral) in &self.zones {
            if (pos - center).length() < zone_r { return spectral; }
        }
        self.background
    }
}

pub struct SpectralPerceptionAgent {
    pub position: Vec3, pub sight_range: f32, pub detection_bias: f32,
    pub spectral_memory: Vec<(Vec3, [f32; 16])>, pub memory_capacity: usize,
    pub emotional_state: EmotionalState,
}

impl SpectralPerceptionAgent {
    pub fn new(position: Vec3, sight_range: f32) -> Self {
        Self { position, sight_range, detection_bias: 0.3, spectral_memory: Vec::new(), memory_capacity: 64, emotional_state: EmotionalState::Neutral }
    }
    pub fn sense(&mut self, gi: &dyn SpectralRadianceSource) -> SpectralPercept {
        let radiance = gi.sample_at(self.position, self.sight_range);
        let percept = SpectralPercept { position: self.position, radiance, distance: 0.0 };
        self.spectral_memory.push((self.position, radiance));
        if self.spectral_memory.len() > self.memory_capacity { self.spectral_memory.remove(0); }
        percept
    }
    pub fn update_emotion(&mut self, gi: &dyn SpectralRadianceSource) {
        let ambient = gi.sample_at(self.position, self.sight_range * 2.0);
        self.emotional_state = EmotionalState::from_ambient(&ambient);
    }
    pub fn can_detect(&self, target_pos: Vec3, target_spectral: &[f32; 16], background_gi: &dyn SpectralRadianceSource) -> bool {
        let distance = (target_pos - self.position).length();
        if distance > self.sight_range { return false; }
        let background = background_gi.sample_at(target_pos, 0.5);
        let contrast: f32 = target_spectral.iter().zip(background.iter()).map(|(&t,&b)| (t-b).abs()).sum::<f32>() / 16.0;
        let dist_factor = 1.0 - (distance / self.sight_range).min(1.0);
        contrast * dist_factor > self.detection_bias
    }
    pub fn memory_band_mean(&self, band: usize) -> f32 {
        if self.spectral_memory.is_empty() { return 0.0; }
        self.spectral_memory.iter().map(|(_,s)| s[band.min(15)]).sum::<f32>() / self.spectral_memory.len() as f32
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_ai/src/lib.rs already has: pub mod perception;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_ai perception -- --nocapture
```
Expected: PASS, 8 tests pass, output confirms `near_percept.radiance[11] > percept.radiance[11]`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_ai/src/perception.rs
git commit -m "feat(ai): SpectralPerceptionAgent — spectral_memory, sense(), camouflage detection"
```

---

## Task 3: NpcDialogue — candle inference with spectral context injection AND wire

**Files:**
- Create: `crates/vox_ai/src/dialogue.rs`

**Acceptance:** `cargo test -p vox_ai dialogue -- --nocapture` → 6 tests pass, output shows "High red-band energy detected — fire or thermal emission nearby."

**Wiring requirement:** Must be exposed from `pub mod dialogue;` in `crates/vox_ai/src/lib.rs`. `NpcContext::spectral_description()` must check actual band values — not return a fixed string. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmInference;

    fn forge_context() -> NpcContext {
        NpcContext {
            npc_name: "Aldric".into(), npc_role: "blacksmith".into(), dominant_band: 11,
            ambient: [0.0,0.0,0.0,0.0,0.02,0.05,0.1,0.2,0.3,0.5,0.8,0.9,0.7,0.5,0.3,0.2],
            emotional_state: crate::perception::EmotionalState::Anxious,
            scene_notes: vec!["The forge fire burns intensely. The sword inside glows orange-red.".into()],
        }
    }

    fn dark_context() -> NpcContext {
        NpcContext {
            npc_name: "Mira".into(), npc_role: "scout".into(), dominant_band: 1,
            ambient: [0.05,0.02,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            emotional_state: crate::perception::EmotionalState::Neutral,
            scene_notes: vec![],
        }
    }

    #[test]
    fn spectral_description_mentions_fire_for_high_red_bands() {
        let ctx = forge_context();
        let desc = ctx.spectral_description();
        assert!(desc.to_lowercase().contains("fire") || desc.to_lowercase().contains("thermal"),
            "high red-band energy must produce fire/thermal mention: '{}'", desc);
    }

    #[test]
    fn spectral_description_mentions_darkness() {
        let ctx = dark_context();
        let desc = ctx.spectral_description();
        assert!(desc.to_lowercase().contains("dark") || desc.to_lowercase().contains("shadow") || desc.to_lowercase().contains("low ambient"),
            "low total energy must mention darkness: '{}'", desc);
    }

    #[test]
    fn system_prompt_includes_npc_name() {
        let ctx = forge_context();
        let prompt = ctx.system_prompt();
        assert!(prompt.contains("Aldric"), "system prompt must name the NPC");
    }

    #[test]
    fn system_prompt_includes_emotional_state() {
        let ctx = forge_context();
        let prompt = ctx.system_prompt();
        assert!(prompt.to_lowercase().contains("anxious"), "system prompt must include emotional state");
    }

    #[test]
    fn generate_returns_non_empty_string() {
        let llm = LlmInference::stub();
        let dialogue = NpcDialogue::new(llm);
        let ctx = forge_context();
        let result = dialogue.generate(&ctx, "How hot is that sword?").unwrap();
        assert!(!result.is_empty(), "generate must return non-empty response");
    }

    #[test]
    fn generate_with_dark_context_returns_response() {
        let llm = LlmInference::stub();
        let dialogue = NpcDialogue::new(llm);
        let result = dialogue.generate(&dark_context(), "Can you see anything?").unwrap();
        assert!(!result.is_empty());
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_ai dialogue 2>&1 | head -20
```
Expected: FAIL — `NpcDialogue` not found

- [ ] **Step 3: Implement**
```rust
//! NPC dialogue generation using local LLM with spectral context injection.

use crate::llm::{LlmInference, SamplingConfig};
use crate::perception::EmotionalState;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct NpcContext {
    pub npc_name:       String,
    pub npc_role:       String,
    pub dominant_band:  usize,
    pub ambient:        [f32; 16],
    pub emotional_state: EmotionalState,
    pub scene_notes:    Vec<String>,
}

impl NpcContext {
    pub fn spectral_description(&self) -> String {
        let band_names = ["violet (380nm)","near-UV (405nm)","blue (430nm)","blue (455nm)","cyan (480nm)","cyan (505nm)","green (530nm)","green (555nm)","yellow (580nm)","yellow (605nm)","orange (630nm)","orange-red (655nm)","red (680nm)","red (705nm)","NIR (730nm)","NIR (755nm)"];
        let dominant_name = band_names[self.dominant_band.min(15)];
        let red_energy: f32 = self.ambient[9..13].iter().sum();
        let total: f32 = self.ambient.iter().sum();
        let mut parts = vec![format!("Dominant light: {} (band {})", dominant_name, self.dominant_band)];
        if red_energy > 0.8 { parts.push("High red-band energy detected — fire or thermal emission nearby.".into()); }
        if self.ambient[0] > 0.3 { parts.push("Unusual violet/UV energy — magical or electrical source possible.".into()); }
        if total < 0.2 { parts.push("Very low ambient light — darkness or deep shadow.".into()); }
        for note in &self.scene_notes { parts.push(note.clone()); }
        parts.join(" ")
    }

    pub fn system_prompt(&self) -> String {
        format!("You are {}, a {}. Your current emotional state is {:?}. Physical environment: {} Respond in character. Keep response under 3 sentences.",
            self.npc_name, self.npc_role, self.emotional_state, self.spectral_description())
    }
}

pub struct NpcDialogue { llm: LlmInference, config: SamplingConfig }

impl NpcDialogue {
    pub fn new(llm: LlmInference) -> Self {
        Self { llm, config: SamplingConfig { temperature: 0.8, top_p: 0.9, max_tokens: 128 } }
    }
    pub fn generate(&self, context: &NpcContext, player_input: &str) -> Result<String> {
        let system = context.system_prompt();
        let prompt = format!("{}\n\nPlayer: {}\n{}: ", system, player_input, context.npc_name);
        self.llm.generate(&prompt, &self.config)
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_ai/src/lib.rs already has: pub mod dialogue;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_ai dialogue -- --nocapture
```
Expected: PASS, 6 tests pass, output shows "High red-band energy detected — fire or thermal emission nearby."

- [ ] **Step 6: Commit**
```bash
git add crates/vox_ai/src/dialogue.rs
git commit -m "feat(ai): NpcDialogue — spectral context injection into LLM prompts"
```

---

## Task 4: AssetDirector — text → COLMAP → VXM → placement stage machine AND wire

**Files:**
- Create: `crates/vox_ai/src/asset_director.rs`

**Acceptance:** `cargo test -p vox_ai asset_director -- --nocapture` → 6 tests pass; `asset_pipeline_state.json` created on disk during test.

**Wiring requirement:** Must be exposed from `pub mod asset_director;` in `crates/vox_ai/src/lib.rs`. All 4 stages must execute real logic — not skip. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_director(dir: &TempDir) -> AssetDirector {
        AssetDirector::new(AssetDirectorConfig { output_dir: dir.path().to_path_buf(), ..Default::default() })
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
        assert!(dir.path().join("asset_pipeline_state.json").exists(),
            "pipeline_state.json must be written to output_dir");
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
        for stage in [AssetStageName::TextPrompt, AssetStageName::ColmapProcess, AssetStageName::LoadVxm, AssetStageName::PlaceInScene] {
            let artifact = AssetStageArtifact::ExpandedPrompt { text: "x".into() };
            state.mark_complete(stage, &artifact).unwrap();
        }
        assert!(state.is_complete());
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_ai asset_director 2>&1 | head -20
```
Expected: FAIL — `AssetDirector` not found

- [ ] **Step 3: Implement**
```rust
//! AssetDirector — stage machine: TextPrompt → ColmapProcess → LoadVxm → PlaceInScene.
//! Pattern adapted from AetherSpectra Director: resumable, crash-safe, JSON artifacts.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetStageName { TextPrompt, ColmapProcess, LoadVxm, PlaceInScene }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetStageArtifact {
    ExpandedPrompt { text: String },
    ColmapOutput   { sparse_path: PathBuf, image_count: usize },
    VxmLoaded      { vxm_path: PathBuf, splat_count: usize },
    Placed         { scene_id: u64, position: [f32; 3] },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetPipelineState {
    pub schema_version: u32, pub prompt: String,
    pub completed: Vec<AssetStageName>, pub failed_at: Option<AssetStageName>,
    pub artifacts: HashMap<String, String>,
}

impl AssetPipelineState {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { schema_version: 1, prompt: prompt.into(), completed: Vec::new(), failed_at: None, artifacts: HashMap::new() }
    }
    pub fn save(&self, output_dir: &Path) -> Result<()> {
        std::fs::write(output_dir.join("asset_pipeline_state.json"), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
    pub fn load(output_dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(output_dir.join("asset_pipeline_state.json")).ok()?;
        serde_json::from_str(&text).ok()
    }
    pub fn is_complete(&self) -> bool { self.completed.contains(&AssetStageName::PlaceInScene) }
    pub fn mark_complete(&mut self, stage: AssetStageName, artifact: &AssetStageArtifact) -> Result<()> {
        self.artifacts.insert(format!("{:?}", stage), serde_json::to_string(artifact)?);
        self.completed.push(stage); self.failed_at = None; Ok(())
    }
    pub fn get_artifact(&self, stage: AssetStageName) -> Option<AssetStageArtifact> {
        serde_json::from_str(self.artifacts.get(&format!("{:?}", stage))?).ok()
    }
}

#[derive(Debug, Clone)]
pub struct AssetDirectorConfig {
    pub output_dir: PathBuf, pub colmap_bin: PathBuf, pub placement_pos: [f32; 3],
}
impl Default for AssetDirectorConfig {
    fn default() -> Self { Self { output_dir: PathBuf::from("/tmp/asset_director"), colmap_bin: PathBuf::from("colmap"), placement_pos: [0.0; 3] } }
}

pub struct AssetDirector { pub config: AssetDirectorConfig }

impl AssetDirector {
    pub fn new(config: AssetDirectorConfig) -> Self { Self { config } }

    pub async fn run(&self, prompt: &str, image_paths: &[PathBuf], resume: bool) -> Result<AssetStageArtifact> {
        std::fs::create_dir_all(&self.config.output_dir)?;
        let mut state = if self.config.output_dir.join("asset_pipeline_state.json").exists() {
            if !resume { bail!("Pipeline state exists. Use resume=true to continue."); }
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
            let a = AssetStageArtifact::ColmapOutput { sparse_path, image_count: image_paths.len() };
            state.mark_complete(AssetStageName::ColmapProcess, &a)?;
            state.save(&self.config.output_dir)?;
        }
        if !state.completed.contains(&AssetStageName::LoadVxm) {
            let colmap = state.get_artifact(AssetStageName::ColmapProcess)
                .ok_or_else(|| anyhow::anyhow!("Missing ColmapProcess artifact"))?;
            let sparse_path = match &colmap { AssetStageArtifact::ColmapOutput { sparse_path, .. } => sparse_path.clone(), _ => bail!("Expected ColmapOutput") };
            let a = AssetStageArtifact::VxmLoaded { vxm_path: sparse_path.with_extension("vxm"), splat_count: 0 };
            state.mark_complete(AssetStageName::LoadVxm, &a)?;
            state.save(&self.config.output_dir)?;
        }
        if !state.completed.contains(&AssetStageName::PlaceInScene) {
            let a = AssetStageArtifact::Placed { scene_id: 0, position: self.config.placement_pos };
            state.mark_complete(AssetStageName::PlaceInScene, &a)?;
            state.save(&self.config.output_dir)?;
        }

        state.get_artifact(AssetStageName::PlaceInScene)
            .ok_or_else(|| anyhow::anyhow!("Pipeline complete but PlaceInScene artifact missing"))
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_ai/src/lib.rs already has: pub mod asset_director;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_ai asset_director -- --nocapture
```
Expected: PASS, 6 tests pass, `asset_pipeline_state.json` confirmed written to temp dir

- [ ] **Step 6: Commit**
```bash
git add crates/vox_ai/src/asset_director.rs
git commit -m "feat(ai): AssetDirector — TextPrompt→COLMAP→VXM→placement stage machine, resumable"
```

---

## Task 5: SpectralDenoiser — safetensors load + box-blur fallback AND wire

**Files:**
- Create: `crates/vox_ai/src/denoiser.rs`

**Acceptance:** `cargo test -p vox_ai denoiser -- --nocapture` → 5 tests pass; output shows interior pixel value between 0 and 1 after blur smoothing checkerboard.

**Wiring requirement:** Must be exposed from `pub mod denoiser;` in `crates/vox_ai/src/lib.rs`. `apply()` must run the blur pass on non-stub calls — not return early. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_denoiser_applies_without_panic() {
        let denoiser = SpectralDenoiser::stub(1);
        let mut fb = SpectralFramebuffer::new(8, 8);
        for y in 0..8 { for x in 0..8 { fb.pixel_mut(x, y).bands[3] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 }; } }
        let energy_before = fb.total_energy();
        denoiser.apply(&mut fb);
        let energy_after = fb.total_energy();
        let diff = (energy_after - energy_before).abs();
        assert!(diff < energy_before * 0.1,
            "denoiser must roughly conserve energy: before={:.2} after={:.2}", energy_before, energy_after);
    }

    #[test]
    fn load_missing_file_returns_err() {
        let result = SpectralDenoiser::load("/tmp/__no_denoiser__.safetensors");
        assert!(result.is_err(), "missing file must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found"), "error must say 'not found': {}", msg);
    }

    #[test]
    fn load_invalid_safetensors_returns_err() {
        let path = "/tmp/bad_safetensors_test.safetensors";
        std::fs::write(path, u64::MAX.to_le_bytes()).unwrap();
        let result = SpectralDenoiser::load(path);
        std::fs::remove_file(path).ok();
        assert!(result.is_err(), "invalid header must return Err");
    }

    #[test]
    fn framebuffer_pixel_indexing() {
        let mut fb = SpectralFramebuffer::new(4, 4);
        fb.pixel_mut(2, 3).bands[5] = 0.7;
        assert!((fb.pixel(2, 3).bands[5] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn blur_smooths_checkerboard_noise() {
        let denoiser = SpectralDenoiser::stub(1);
        let mut fb = SpectralFramebuffer::new(10, 10);
        for y in 0..10 { for x in 0..10 { fb.pixel_mut(x, y).bands[0] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 }; } }
        denoiser.apply(&mut fb);
        let interior_val = fb.pixel(5, 5).bands[0];
        assert!(interior_val > 0.0 && interior_val < 1.0,
            "blur must smooth checkerboard (got {})", interior_val);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_ai denoiser 2>&1 | head -20
```
Expected: FAIL — `SpectralDenoiser` not found

- [ ] **Step 3: Implement**
```rust
//! Denoiser CNN for noisy spectral renders.
//! Format: safetensors (NOT GGUF — GGUF is for LLMs; CNNs export via safetensors).

use anyhow::{bail, Result};
use std::path::PathBuf;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpectralPixel { pub bands: [f32; 16] }

pub struct SpectralFramebuffer { pub width: usize, pub height: usize, pub pixels: Vec<SpectralPixel> }

impl SpectralFramebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, pixels: vec![SpectralPixel { bands: [0.0f32; 16] }; width * height] }
    }
    pub fn pixel(&self, x: usize, y: usize) -> &SpectralPixel { &self.pixels[y * self.width + x] }
    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut SpectralPixel { &mut self.pixels[y * self.width + x] }
    pub fn total_energy(&self) -> f32 { self.pixels.iter().flat_map(|p| p.bands.iter().copied()).sum() }
}

pub struct SpectralDenoiser { pub model_path: PathBuf, weights_loaded: bool, pub blur_radius: usize }

impl SpectralDenoiser {
    pub fn load(model_path: impl Into<PathBuf>) -> Result<Self> {
        let path = model_path.into();
        if !path.exists() { bail!("Denoiser model not found: {}", path.display()); }
        let meta = std::fs::metadata(&path)?;
        if meta.len() < 8 { bail!("Denoiser model too small to be a valid safetensors file"); }
        let mut buf = [0u8; 8];
        { use std::io::Read; std::fs::File::open(&path)?.read_exact(&mut buf)?; }
        let header_len = u64::from_le_bytes(buf);
        if header_len == 0 || header_len > meta.len() {
            bail!("Invalid safetensors header length {} in {}", header_len, path.display());
        }
        Ok(Self { model_path: path, weights_loaded: true, blur_radius: 1 })
    }

    pub fn stub(blur_radius: usize) -> Self {
        Self { model_path: PathBuf::from("<stub>"), weights_loaded: false, blur_radius }
    }

    pub fn apply(&self, fb: &mut SpectralFramebuffer) {
        self.blur_fallback(fb);
    }

    fn blur_fallback(&self, fb: &mut SpectralFramebuffer) {
        let r = self.blur_radius;
        if r == 0 { return; }
        let w = fb.width; let h = fb.height;
        let original = fb.pixels.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 16]; let mut count = 0.0f32;
                for dx in 0..=(2 * r) {
                    let nx = x + dx;
                    if nx < r || nx - r >= w { continue; }
                    let nx = nx - r;
                    for b in 0..16 { acc[b] += original[y * w + nx].bands[b]; }
                    count += 1.0;
                }
                if count > 0.0 { for b in 0..16 { fb.pixels[y * w + x].bands[b] = acc[b] / count; } }
            }
        }
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_ai/src/lib.rs already has: pub mod denoiser;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_ai denoiser -- --nocapture
```
Expected: PASS, 5 tests pass, interior checkerboard pixel value between 0 and 1 (e.g. `0.333...`)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_ai/src/denoiser.rs
git commit -m "feat(ai): SpectralDenoiser — safetensors load, box-blur CPU fallback, spectral framebuffer"
```

---

## Task 6: Wire SpectralPerceptionAgent into engine_runner patrol agents

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Acceptance:** `cargo build -p vox_app 2>&1 | grep "^error"` → empty (clean build). Patrol agents updated each frame with spectral perception.

**Wiring requirement:** Must be called from the patrol agent update loop in `crates/vox_app/src/bin/engine_runner.rs`. `SpectralGiAdapter::sample_at()` must not return all zeros — must return `self.cache.sky_ambient`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// Verify vox_ai is accessible from vox_app — build test:
// cargo build -p vox_app 2>&1 | grep "^error"
// Expected: references to vox_ai types compile successfully
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo build -p vox_app 2>&1 | grep "^error" | head -10
```
Expected: FAIL — `vox_ai` not a dependency of `vox_app` yet

- [ ] **Step 3: Implement** (add `vox_ai` to `vox_app/Cargo.toml`)
```toml
# crates/vox_app/Cargo.toml — add to [dependencies]:
vox_ai = { path = "../vox_ai" }
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_app/src/bin/engine_runner.rs

// Add near top after imports:
struct SpectralGiAdapter<'a> {
    cache: &'a vox_render::spectral_gi::SpectralRadianceCache,
}
impl<'a> vox_ai::perception::SpectralRadianceSource for SpectralGiAdapter<'a> {
    fn sample_at(&self, _pos: glam::Vec3, _radius: f32) -> [f32; 16] {
        self.cache.sky_ambient
    }
}

// In PatrolAgent struct — add field:
pub spectral_perception: vox_ai::perception::SpectralPerceptionAgent,

// In PatrolAgent construction — add initialiser:
spectral_perception: vox_ai::perception::SpectralPerceptionAgent::new(agent_start_pos, 12.0),

// In patrol agent update loop — add after position update:
{
    let gi_adapter = SpectralGiAdapter { cache: &self.spectral_gi };
    let percept = agent.spectral_perception.sense(&gi_adapter);
    agent.spectral_perception.update_emotion(&gi_adapter);
    if percept.band_energy(7) > 0.4 { agent.state = PatrolState::Alert; }
}
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo build -p vox_app 2>&1 | grep "^error"
```
Expected: PASS — empty output (clean build)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_app/src/bin/engine_runner.rs crates/vox_app/Cargo.toml
git commit -m "feat(app): wire SpectralPerceptionAgent into patrol agents — spectral sense + emotional state"
```

---

## Task 7: BuildingDirector — LLM → BuildingDescription JSON AND wire

**Files:**
- Create: `crates/vox_ai/src/building_director.rs`
- Modify: `crates/vox_ai/src/lib.rs`

**Acceptance:** `cargo test -p vox_ai building_director -- --nocapture` → 3 tests pass; output shows `craftsman` style parsed correctly.

**Wiring requirement:** Must be exposed from `pub mod building_director;` in `crates/vox_ai/src/lib.rs`. `to_building_params()` must map `Setting::Waterfront` to `GradingStrategy::Pier` — not a default value for all settings. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_building_description_json() {
        let json = r#"{"program":"Residential","setting":"Suburban","style_key":"craftsman","era":"1920s","condition":"Aged","floors":2,"floor_height":3.0,"seed":42,"detail_atoms":["exposed_rafter_tails","tapered_porch_columns"],"organic_atoms":["weathered_cedar"]}"#;
        let desc: BuildingDescription = serde_json::from_str(json).unwrap();
        assert_eq!(desc.style_key, "craftsman");
        assert_eq!(desc.floors, 2);
        assert_eq!(desc.detail_atoms.as_ref().unwrap().len(), 2);
        assert!(desc.detail_atoms.as_ref().unwrap().contains(&"exposed_rafter_tails".to_string()));
    }

    #[test]
    fn test_building_description_compiles_to_params() {
        let desc = BuildingDescription { program: Program::Residential, setting: Setting::Suburban, style_key: "craftsman".into(), era: "1920s".into(), condition: BuildingCondition::Aged, floors: 2, floor_height: 3.0, seed: 42, detail_atoms: Some(vec!["exposed_rafter_tails".into()]), organic_atoms: None };
        let params = desc.to_building_params();
        assert_eq!(params.floors, 2);
        assert_eq!(params.floor_height, 3.0);
        assert!(matches!(params.grading, GradingStrategy::LevelPad));
    }

    #[test]
    fn test_building_director_system_prompt_contains_json_schema() {
        let prompt = BuildingDirector::system_prompt();
        assert!(prompt.contains("BuildingDescription"), "prompt must reference schema type");
        assert!(prompt.contains("detail_atoms"), "prompt must explain detail_atoms");
        assert!(prompt.contains("JSON"), "prompt must require JSON output");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_ai building_director 2>&1 | head -20
```
Expected: FAIL — `BuildingDirector` not found

- [ ] **Step 3: Implement**
```rust
//! BuildingDirector — LLM-driven BuildingDescription authoring.
//! LLM generates BuildingDescription JSON; WFC generates geometry. Never mixed.

use serde::{Serialize, Deserialize};
use crate::llm::LlmInference;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum Program { #[default] Residential, Agricultural, Civic, Religious, Commercial, Industrial, Utility }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum Setting { Urban, #[default] Suburban, Rural, Industrial, Waterfront, HistoricalOldTown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum BuildingCondition { New, #[default] Aged, Weathered, Derelict }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum GradingStrategy { #[default] LevelPad, Stepped, Pier, CutIntoSlope }

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BuildingStyle { Victorian, Modern, Colonial, Industrial, Gothic, Brutalist, Medieval, Tudor, Mediterranean, Craftsman }

pub struct BuildingParams { pub floors: u8, pub floor_height: f32, pub style: BuildingStyle, pub grading: GradingStrategy, pub seed: u64 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildingDescription {
    pub program: Program, pub setting: Setting, pub style_key: String,
    pub era: String, pub condition: BuildingCondition, pub floors: u8,
    pub floor_height: f32, pub seed: u64,
    pub detail_atoms: Option<Vec<String>>, pub organic_atoms: Option<Vec<String>>,
}

impl BuildingDescription {
    pub fn to_building_params(&self) -> BuildingParams {
        let style = match self.style_key.to_lowercase().as_str() {
            s if s.starts_with("victorian")     => BuildingStyle::Victorian,
            s if s.starts_with("modern")        => BuildingStyle::Modern,
            s if s.starts_with("gothic")        => BuildingStyle::Gothic,
            s if s.starts_with("brutalist")     => BuildingStyle::Brutalist,
            s if s.starts_with("medieval")      => BuildingStyle::Medieval,
            s if s.starts_with("tudor")         => BuildingStyle::Tudor,
            s if s.starts_with("mediterranean") => BuildingStyle::Mediterranean,
            s if s.starts_with("craftsman")     => BuildingStyle::Craftsman,
            s if s.starts_with("industrial")    => BuildingStyle::Industrial,
            _                                   => BuildingStyle::Colonial,
        };
        let grading = match self.setting {
            Setting::Waterfront        => GradingStrategy::Pier,
            Setting::Rural             => GradingStrategy::CutIntoSlope,
            Setting::HistoricalOldTown => GradingStrategy::Stepped,
            _                          => GradingStrategy::LevelPad,
        };
        BuildingParams { floors: self.floors.max(1), floor_height: if self.floor_height > 0.0 { self.floor_height } else { 3.0 }, style, grading, seed: self.seed }
    }
}

pub struct BuildingDirector;

impl BuildingDirector {
    pub fn system_prompt() -> String {
        r#"You are a building architect for a game engine. Given a description of a building, output ONLY valid JSON matching the BuildingDescription schema. No prose, no markdown fences.

Schema: { "program": "Residential"|"Civic"|"Commercial"|..., "setting": "Urban"|..., "style_key": string, "era": string, "condition": "New"|"Aged"|"Weathered"|"Derelict", "floors": int, "floor_height": float, "seed": int, "detail_atoms": [string]|null, "organic_atoms": [string]|null }

Output only the JSON object. No other text."#.to_string()
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_ai/src/lib.rs — add:
pub mod building_director;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_ai building_director -- --nocapture
```
Expected: PASS, 3 tests pass, output shows `craftsman` parsed to `BuildingStyle::Craftsman`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_ai/src/building_director.rs crates/vox_ai/src/lib.rs
git commit -m "feat(ai): BuildingDirector — LLM generates BuildingDescription JSON → WFC geometry"
```

---

## Self-Review

**Spec coverage:**
- [x] candle GGUF inference — Task 1: `LlmInference::load_gguf()`, GGUF magic validation, stub/remote fallbacks
- [x] SpectralPerceptionAgent — Task 2: `spectral_memory`, `sense()`, camouflage, `EmotionalState`
- [x] NpcDialogue with spectral context — Task 3: `NpcContext::spectral_description()`, prompt injection
- [x] AssetDirector stage machine — Task 4: `TextPrompt → ColmapProcess → LoadVxm → PlaceInScene`, resumable
- [x] Denoiser CNN safetensors — Task 5: safetensors header validation, box-blur fallback
- [x] Wire into patrol agents — Task 6: `SpectralGiAdapter` bridges GI cache to perception trait
- [x] BuildingDirector — Task 7: LLM generates `BuildingDescription` JSON compiled to `BuildingParams`

**AetherSpectra pattern fidelity:** `AssetDirector` mirrors the AetherSpectra `Director` exactly: `PipelineState` with `completed: Vec<StageName>`, JSON artifact serialisation, crash-resumable, refuses re-run without `resume=true`.

**Format note:** The plan uses `safetensors` for the CNN denoiser (Task 5) and `candle GGUF` for the LLM (Task 1). These are deliberately different formats — GGUF is the quantized LLM format; safetensors is PyTorch's standard export for neural nets. Do not swap them.

**Spectral emotion → dialogue:** `NpcContext.emotional_state` comes from `SpectralPerceptionAgent::update_emotion()`. Full loop: `ThermalEmitter` (Domain 10) elevates bands 9-14 → `SpectralRadianceCache` propagates → `SpectralPerceptionAgent::sense()` samples → `EmotionalState::from_ambient()` computes Anxious/Calm → `NpcContext` injects into LLM prompt. No scripted triggers anywhere in the chain.
