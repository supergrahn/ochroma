//! Adaptive music system.
//!
//! Layers horizontal (intensity-based) and vertical (state-based) music tracks.
//! Music responds to gameplay state and spectral scene data.
//! Playback is driven by AudioHandle — this module manages state only.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MusicState {
    Exploration,
    Combat,
    Stealth,
    Cinematic,
    Menu,
}

pub struct MusicLayer {
    pub path: String,
    pub base_volume: f32,
    pub state: MusicState,
    /// 0=bass/percussion, 1=harmony, 2=melody, 3=ambient
    pub layer_index: u8,
}

pub struct MusicIntensity {
    pub combat: f32,
    pub tension: f32,
    pub exploration: f32,
}

pub struct AdaptiveMusicPlayer {
    state: MusicState,
    layers: Vec<MusicLayer>,
    intensity: MusicIntensity,
    transition_blend: f32,
    target_state: MusicState,
}

impl AdaptiveMusicPlayer {
    pub fn new() -> Self {
        Self {
            state: MusicState::Exploration,
            layers: Vec::new(),
            intensity: MusicIntensity { combat: 0.0, tension: 0.0, exploration: 0.0 },
            transition_blend: 1.0,
            target_state: MusicState::Exploration,
        }
    }

    /// Sets target_state and begins a crossfade transition (resets blend to 0.0).
    pub fn set_state(&mut self, new_state: MusicState) {
        self.state = self.target_state;
        self.target_state = new_state;
        self.transition_blend = 0.0;
    }

    /// Advances transition_blend toward 1.0 at rate 1/3 per second (3s crossfade).
    pub fn update(&mut self, dt: f32) {
        self.transition_blend = (self.transition_blend + dt / 3.0).min(1.0);
        if self.transition_blend >= 1.0 {
            self.state = self.target_state;
        }
    }

    pub fn add_layer(&mut self, layer: MusicLayer) {
        self.layers.push(layer);
    }

    /// Returns (layer_ref, effective_volume) for layers matching the current target state.
    /// Volume = layer.base_volume × transition_blend (weight reaches 1.0 when fully transitioned).
    pub fn active_layers(&self) -> Vec<(&MusicLayer, f32)> {
        self.layers
            .iter()
            .filter(|l| l.state == self.target_state)
            .map(|l| (l, l.base_volume * self.transition_blend))
            .collect()
    }

    /// Detect music state from spectral data:
    /// - band 7 > 0.7 → Combat (red/low-freq = explosions)
    /// - band 0 > 0.8 → Stealth (blue/high-freq = electric)
    /// - else → Exploration
    pub fn update_from_spectral(&mut self, spectral: &[f32; 8]) {
        self.intensity = spectral_to_intensity(spectral);
        let new_state = if spectral[7] > 0.7 {
            MusicState::Combat
        } else if spectral[0] > 0.8 {
            MusicState::Stealth
        } else {
            MusicState::Exploration
        };
        if new_state != self.target_state {
            self.set_state(new_state);
        }
    }
}

impl Default for AdaptiveMusicPlayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Derive MusicIntensity from spectral bands.
/// - combat: average of bands 5-7 (red/orange = fire/explosion)
/// - tension: band 0 (blue-violet = electric tension)
/// - exploration: average of bands 2-4 (green = nature/calm)
pub fn spectral_to_intensity(spectral: &[f32; 8]) -> MusicIntensity {
    let combat = (spectral[5] + spectral[6] + spectral[7]) / 3.0;
    let tension = spectral[0];
    let exploration = (spectral[2] + spectral[3] + spectral[4]) / 3.0;
    MusicIntensity { combat, tension, exploration }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_blend_advances_with_time() {
        let mut player = AdaptiveMusicPlayer::new();
        player.set_state(MusicState::Combat);
        player.update(1.5);
        assert!((player.transition_blend - 0.5).abs() < 0.05);
    }

    #[test]
    fn transition_blend_clamps_at_one() {
        let mut player = AdaptiveMusicPlayer::new();
        player.set_state(MusicState::Combat);
        player.update(10.0);
        assert_eq!(player.transition_blend, 1.0);
    }

    #[test]
    fn set_state_resets_blend() {
        let mut player = AdaptiveMusicPlayer::new();
        player.set_state(MusicState::Combat);
        assert_eq!(player.transition_blend, 0.0);
    }

    #[test]
    fn active_layers_returns_matching_state() {
        let mut player = AdaptiveMusicPlayer::new();
        player.set_state(MusicState::Combat);
        player.add_layer(MusicLayer {
            path: "combat_bass.wav".into(),
            base_volume: 1.0,
            state: MusicState::Combat,
            layer_index: 0,
        });
        player.add_layer(MusicLayer {
            path: "explore_melody.wav".into(),
            base_volume: 1.0,
            state: MusicState::Exploration,
            layer_index: 2,
        });
        let active = player.active_layers();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0.path, "combat_bass.wav");
    }

    #[test]
    fn spectral_to_intensity_high_red_is_combat() {
        let mut spectral = [0.0f32; 8];
        spectral[5] = 1.0;
        spectral[6] = 1.0;
        spectral[7] = 1.0;
        let intensity = spectral_to_intensity(&spectral);
        assert!(intensity.combat >= 0.9);
    }

    #[test]
    fn update_from_spectral_combat_detection() {
        let mut player = AdaptiveMusicPlayer::new();
        let mut spectral = [0.0f32; 8];
        spectral[7] = 0.8;
        player.update_from_spectral(&spectral);
        assert_eq!(player.target_state, MusicState::Combat);
    }
}
