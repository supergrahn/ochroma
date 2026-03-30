//! Spectral perception for AI agents.
//! Agents perceive via spectral bands, not RGB. Detection is purely physical.

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct SpectralPercept {
    pub position: Vec3,
    pub radiance: [f32; 16],
    pub distance: f32,
}

impl SpectralPercept {
    pub fn total_energy(&self) -> f32 {
        self.radiance.iter().sum()
    }

    pub fn dominant_band(&self) -> usize {
        self.radiance
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn band_energy(&self, band: usize) -> f32 {
        let att = 1.0 / (self.distance * self.distance + 1.0);
        self.radiance[band.min(15)] * att
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmotionalState {
    Calm,
    Anxious,
    Uneasy,
    Neutral,
}

impl EmotionalState {
    pub fn from_ambient(ambient: &[f32; 16]) -> Self {
        let red: f32 = ambient[9..13].iter().sum();
        let green: f32 = ambient[5..8].iter().sum();
        let violet: f32 = ambient[0];
        let max = red.max(green).max(violet);
        if max < 0.1 {
            return Self::Neutral;
        }
        if red >= green && red >= violet {
            Self::Anxious
        } else if green >= red && green >= violet {
            Self::Calm
        } else {
            Self::Uneasy
        }
    }
}

pub trait SpectralRadianceSource {
    fn sample_at(&self, pos: Vec3, radius: f32) -> [f32; 16];
}

pub struct FixedRadianceSource(pub [f32; 16]);

impl SpectralRadianceSource for FixedRadianceSource {
    fn sample_at(&self, _pos: Vec3, _radius: f32) -> [f32; 16] {
        self.0
    }
}

pub struct ZonedRadianceSource {
    pub zones: Vec<(Vec3, f32, [f32; 16])>,
    pub background: [f32; 16],
}

impl SpectralRadianceSource for ZonedRadianceSource {
    fn sample_at(&self, pos: Vec3, _radius: f32) -> [f32; 16] {
        for &(center, zone_r, spectral) in &self.zones {
            if (pos - center).length() < zone_r {
                return spectral;
            }
        }
        self.background
    }
}

pub struct SpectralPerceptionAgent {
    pub position: Vec3,
    pub sight_range: f32,
    pub detection_bias: f32,
    pub spectral_memory: Vec<(Vec3, [f32; 16])>,
    pub memory_capacity: usize,
    pub emotional_state: EmotionalState,
}

impl SpectralPerceptionAgent {
    pub fn new(position: Vec3, sight_range: f32) -> Self {
        Self {
            position,
            sight_range,
            detection_bias: 0.3,
            spectral_memory: Vec::new(),
            memory_capacity: 64,
            emotional_state: EmotionalState::Neutral,
        }
    }

    pub fn sense(&mut self, gi: &dyn SpectralRadianceSource) -> SpectralPercept {
        let radiance = gi.sample_at(self.position, self.sight_range);
        let percept = SpectralPercept { position: self.position, radiance, distance: 0.0 };
        self.spectral_memory.push((self.position, radiance));
        if self.spectral_memory.len() > self.memory_capacity {
            self.spectral_memory.remove(0);
        }
        percept
    }

    pub fn update_emotion(&mut self, gi: &dyn SpectralRadianceSource) {
        let ambient = gi.sample_at(self.position, self.sight_range * 2.0);
        self.emotional_state = EmotionalState::from_ambient(&ambient);
    }

    pub fn can_detect(
        &self,
        target_pos: Vec3,
        target_spectral: &[f32; 16],
        background_gi: &dyn SpectralRadianceSource,
    ) -> bool {
        let distance = (target_pos - self.position).length();
        if distance > self.sight_range {
            return false;
        }
        let background = background_gi.sample_at(target_pos, 0.5);
        let contrast: f32 = target_spectral
            .iter()
            .zip(background.iter())
            .map(|(&t, &b)| (t - b).abs())
            .sum::<f32>()
            / 16.0;
        let dist_factor = 1.0 - (distance / self.sight_range).min(1.0);
        contrast * dist_factor > self.detection_bias
    }

    pub fn memory_band_mean(&self, band: usize) -> f32 {
        if self.spectral_memory.is_empty() {
            return 0.0;
        }
        self.spectral_memory.iter().map(|(_, s)| s[band.min(15)]).sum::<f32>()
            / self.spectral_memory.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fire_zone() -> ZonedRadianceSource {
        let mut fire_spectral = [0.0f32; 16];
        fire_spectral[10] = 0.9;
        fire_spectral[11] = 1.0;
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
        for _ in 0..10 {
            agent.sense(&gi);
        }
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
        assert!(
            near_percept.radiance[11] > percept.radiance[11],
            "agent near fire should have higher band-11 radiance"
        );
    }

    #[test]
    fn emotional_state_anxious_from_red_environment() {
        let mut red = [0.0f32; 16];
        red[9] = 0.8;
        red[10] = 0.9;
        red[11] = 1.0;
        red[12] = 0.8;
        let state = EmotionalState::from_ambient(&red);
        assert_eq!(state, EmotionalState::Anxious, "dominant red bands must produce Anxious state");
    }

    #[test]
    fn emotional_state_calm_from_green_environment() {
        let mut green = [0.0f32; 16];
        green[5] = 0.9;
        green[6] = 0.8;
        green[7] = 0.7;
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
        target_spectral[5] = 0.7;
        target_spectral[6] = 0.8;
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
