//! NPC dialogue generation using local LLM with spectral context injection.

use crate::llm::{LlmInference, SamplingConfig};
use crate::perception::EmotionalState;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct NpcContext {
    pub npc_name: String,
    pub npc_role: String,
    pub dominant_band: usize,
    pub ambient: [f32; 16],
    pub emotional_state: EmotionalState,
    pub scene_notes: Vec<String>,
}

impl NpcContext {
    pub fn spectral_description(&self) -> String {
        let band_names = [
            "violet (380nm)", "near-UV (405nm)", "blue (430nm)", "blue (455nm)",
            "cyan (480nm)", "cyan (505nm)", "green (530nm)", "green (555nm)",
            "yellow (580nm)", "yellow (605nm)", "orange (630nm)", "orange-red (655nm)",
            "red (680nm)", "red (705nm)", "NIR (730nm)", "NIR (755nm)",
        ];
        let dominant_name = band_names[self.dominant_band.min(15)];
        let red_energy: f32 = self.ambient[9..13].iter().sum();
        let total: f32 = self.ambient.iter().sum();
        let mut parts =
            vec![format!("Dominant light: {} (band {})", dominant_name, self.dominant_band)];
        if red_energy > 0.8 {
            parts.push("High red-band energy detected — fire or thermal emission nearby.".into());
        }
        if self.ambient[0] > 0.3 {
            parts.push(
                "Unusual violet/UV energy — magical or electrical source possible.".into(),
            );
        }
        if total < 0.2 {
            parts.push("Very low ambient light — darkness or deep shadow.".into());
        }
        for note in &self.scene_notes {
            parts.push(note.clone());
        }
        parts.join(" ")
    }

    pub fn system_prompt(&self) -> String {
        format!(
            "You are {}, a {}. Your current emotional state is {:?}. Physical environment: {} Respond in character. Keep response under 3 sentences.",
            self.npc_name,
            self.npc_role,
            self.emotional_state,
            self.spectral_description()
        )
    }
}

pub struct NpcDialogue {
    llm: LlmInference,
    config: SamplingConfig,
}

impl NpcDialogue {
    pub fn new(llm: LlmInference) -> Self {
        Self { llm, config: SamplingConfig { temperature: 0.8, top_p: 0.9, max_tokens: 128 } }
    }

    pub fn generate(&self, context: &NpcContext, player_input: &str) -> Result<String> {
        let system = context.system_prompt();
        let prompt =
            format!("{}\n\nPlayer: {}\n{}: ", system, player_input, context.npc_name);
        self.llm.generate(&prompt, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmInference;

    fn forge_context() -> NpcContext {
        NpcContext {
            npc_name: "Aldric".into(),
            npc_role: "blacksmith".into(),
            dominant_band: 11,
            ambient: [
                0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.7, 0.5, 0.3,
                0.2,
            ],
            emotional_state: EmotionalState::Anxious,
            scene_notes: vec![
                "The forge fire burns intensely. The sword inside glows orange-red.".into(),
            ],
        }
    }

    fn dark_context() -> NpcContext {
        NpcContext {
            npc_name: "Mira".into(),
            npc_role: "scout".into(),
            dominant_band: 1,
            ambient: [
                0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ],
            emotional_state: EmotionalState::Neutral,
            scene_notes: vec![],
        }
    }

    #[test]
    fn spectral_description_mentions_fire_for_high_red_bands() {
        let ctx = forge_context();
        let desc = ctx.spectral_description();
        assert!(
            desc.to_lowercase().contains("fire") || desc.to_lowercase().contains("thermal"),
            "high red-band energy must produce fire/thermal mention: '{}'",
            desc
        );
    }

    #[test]
    fn spectral_description_mentions_darkness() {
        let ctx = dark_context();
        let desc = ctx.spectral_description();
        assert!(
            desc.to_lowercase().contains("dark")
                || desc.to_lowercase().contains("shadow")
                || desc.to_lowercase().contains("low ambient"),
            "low total energy must mention darkness: '{}'",
            desc
        );
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
        assert!(
            prompt.to_lowercase().contains("anxious"),
            "system prompt must include emotional state"
        );
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
