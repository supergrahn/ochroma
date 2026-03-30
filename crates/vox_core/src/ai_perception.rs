//! AI perception: sight cones, hearing radii, and spectral signature detection.
//!
//! Unique to Ochroma: agents can perceive spectral signatures of nearby objects,
//! allowing guards to "see" fire (high red bands) through fog, or recognize
//! player disguises by spectral mismatch.

use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StimulusKind {
    Sight,
    Sound,
    Spectral,
}

#[derive(Debug, Clone)]
pub struct Stimulus {
    pub kind: StimulusKind,
    pub position: Vec3,
    pub intensity: f32,
    pub source_entity: u32,
}

#[derive(Debug, Clone)]
pub struct SightConfig {
    pub range: f32,
    pub half_angle_rad: f32,
    pub height_offset: f32,
}

impl Default for SightConfig {
    fn default() -> Self {
        Self {
            range: 15.0,
            half_angle_rad: 0.6,
            height_offset: 1.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HearingConfig {
    pub range: f32,
}

impl Default for HearingConfig {
    fn default() -> Self {
        Self { range: 8.0 }
    }
}

#[derive(Debug, Clone)]
pub struct SpectralPerceptionConfig {
    pub band: u8,
    pub threshold: f32,
    pub range: f32,
}

impl Default for SpectralPerceptionConfig {
    fn default() -> Self {
        Self {
            band: 6,
            threshold: 0.5,
            range: 20.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PerceptionComponent {
    pub sight: SightConfig,
    pub hearing: HearingConfig,
    pub spectral: SpectralPerceptionConfig,
    pub stimuli: Vec<Stimulus>,
}

impl PerceptionComponent {
    pub fn clear_stimuli(&mut self) {
        self.stimuli.clear();
    }
}

pub fn check_sight(
    observer_pos: Vec3,
    observer_forward: Vec3,
    target_pos: Vec3,
    cfg: &SightConfig,
) -> bool {
    let diff = target_pos - observer_pos;
    let distance = diff.length();
    if distance < 1e-4 {
        return false;
    }
    if distance > cfg.range {
        return false;
    }
    let dir = diff / distance;
    let fwd = observer_forward.normalize();
    let dot = fwd.dot(dir).clamp(-1.0, 1.0);
    let angle = dot.acos();
    angle <= cfg.half_angle_rad
}

pub fn check_hearing(observer_pos: Vec3, sound_pos: Vec3, cfg: &HearingConfig) -> bool {
    (sound_pos - observer_pos).length() <= cfg.range
}

pub fn check_spectral(
    observer_pos: Vec3,
    target_pos: Vec3,
    target_spectral: &[f32; 8],
    cfg: &SpectralPerceptionConfig,
) -> bool {
    let distance = (target_pos - observer_pos).length();
    if distance > cfg.range {
        return false;
    }
    let band_idx = cfg.band as usize;
    if band_idx >= 8 {
        return false;
    }
    target_spectral[band_idx] >= cfg.threshold
}

pub struct PerceptionSystem;

impl PerceptionSystem {
    /// agents: `(perception, position, forward_dir)`
    /// targets: `(entity_id, position, forward_dir, spectral)`
    pub fn tick(
        agents: &mut [(&mut PerceptionComponent, Vec3, Vec3)],
        targets: &[(u32, Vec3, Vec3, [f32; 8])],
    ) {
        for (perception, agent_pos, agent_fwd) in agents.iter_mut() {
            for &(entity_id, target_pos, _target_fwd, ref spectral) in targets {
                if check_sight(*agent_pos, *agent_fwd, target_pos, &perception.sight) {
                    perception.stimuli.push(Stimulus {
                        kind: StimulusKind::Sight,
                        position: target_pos,
                        intensity: 1.0,
                        source_entity: entity_id,
                    });
                }
                if check_hearing(*agent_pos, target_pos, &perception.hearing) {
                    perception.stimuli.push(Stimulus {
                        kind: StimulusKind::Sound,
                        position: target_pos,
                        intensity: 1.0,
                        source_entity: entity_id,
                    });
                }
                if check_spectral(*agent_pos, target_pos, spectral, &perception.spectral) {
                    perception.stimuli.push(Stimulus {
                        kind: StimulusKind::Spectral,
                        position: target_pos,
                        intensity: spectral[perception.spectral.band as usize],
                        source_entity: entity_id,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sight_detects_target_in_cone() {
        let observer = Vec3::ZERO;
        let forward = Vec3::Z;
        let target = Vec3::new(0.0, 0.0, 10.0);
        let cfg = SightConfig { range: 15.0, ..Default::default() };
        assert!(check_sight(observer, forward, target, &cfg));
    }

    #[test]
    fn sight_misses_target_behind() {
        let observer = Vec3::ZERO;
        let forward = Vec3::Z;
        let target = Vec3::new(0.0, 0.0, -10.0);
        let cfg = SightConfig::default();
        assert!(!check_sight(observer, forward, target, &cfg));
    }

    #[test]
    fn sight_misses_target_too_far() {
        let observer = Vec3::ZERO;
        let forward = Vec3::Z;
        let target = Vec3::new(0.0, 0.0, 20.0);
        let cfg = SightConfig { range: 15.0, ..Default::default() };
        assert!(!check_sight(observer, forward, target, &cfg));
    }

    #[test]
    fn hearing_detects_nearby_sound() {
        let observer = Vec3::ZERO;
        let sound = Vec3::new(5.0, 0.0, 0.0);
        let cfg = HearingConfig { range: 8.0 };
        assert!(check_hearing(observer, sound, &cfg));
    }

    #[test]
    fn hearing_misses_distant_sound() {
        let observer = Vec3::ZERO;
        let sound = Vec3::new(15.0, 0.0, 0.0);
        let cfg = HearingConfig { range: 8.0 };
        assert!(!check_hearing(observer, sound, &cfg));
    }

    #[test]
    fn spectral_detects_fire() {
        let observer = Vec3::ZERO;
        let target = Vec3::new(5.0, 0.0, 0.0);
        let mut spectral = [0.0f32; 8];
        spectral[6] = 0.8;
        let cfg = SpectralPerceptionConfig { band: 6, threshold: 0.5, range: 20.0 };
        assert!(check_spectral(observer, target, &spectral, &cfg));
    }

    #[test]
    fn perception_system_tick_populates_stimuli() {
        let mut perception = PerceptionComponent {
            sight: SightConfig { range: 15.0, half_angle_rad: 0.6, height_offset: 1.7 },
            hearing: HearingConfig { range: 8.0 },
            spectral: SpectralPerceptionConfig { band: 6, threshold: 0.5, range: 20.0 },
            stimuli: Vec::new(),
        };

        let agent_pos = Vec3::ZERO;
        let agent_fwd = Vec3::Z;

        let mut spectral = [0.0f32; 8];
        spectral[6] = 0.8;

        // Target directly ahead at 10m
        let targets: Vec<(u32, Vec3, Vec3, [f32; 8])> = vec![
            (42, Vec3::new(0.0, 0.0, 10.0), Vec3::NEG_Z, spectral),
        ];

        let mut agents: Vec<(&mut PerceptionComponent, Vec3, Vec3)> =
            vec![(&mut perception, agent_pos, agent_fwd)];

        PerceptionSystem::tick(&mut agents, &targets);

        // At least a sight stimulus should be present
        assert!(
            agents[0].0.stimuli.iter().any(|s| s.kind == StimulusKind::Sight && s.source_entity == 42),
            "Expected a Sight stimulus for entity 42"
        );
    }
}
