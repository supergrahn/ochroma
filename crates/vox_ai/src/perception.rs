//! Spectral perception for AI agents.
//! Agents perceive via spectral bands, not RGB. Detection is purely physical.

use glam::Vec3;
use half::f16;

/// A consumable behavior decision an NPC can act on, produced by classifying a
/// spectral percept. This is the *retrievable* output of perception: an NPC
/// reads it and chooses an action (run away, go look, raise an alarm, etc.).
///
/// Ordering is by escalation severity (Idle is least urgent, Flee is most),
/// so callers can compare states (`assessment.behavior >= BehaviorState::Alert`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BehaviorState {
    /// Nothing notable in the spectral field — resume default activity.
    Idle,
    /// Routine ambient signal — keep patrolling, mild awareness.
    Patrol,
    /// An unusual but non-threatening signature — go investigate the source.
    Investigate,
    /// A clearly hostile signature detected (e.g. weapons-grade laser band) —
    /// raise the alarm and engage.
    Alert,
    /// Imminent physical danger (intense fire/heat field) — flee.
    Flee,
}

impl BehaviorState {
    /// Human-readable tag, handy for debugging / NPC dialogue hooks.
    pub fn as_str(self) -> &'static str {
        match self {
            BehaviorState::Idle => "idle",
            BehaviorState::Patrol => "patrol",
            BehaviorState::Investigate => "investigate",
            BehaviorState::Alert => "alert",
            BehaviorState::Flee => "flee",
        }
    }
}

/// The full result of classifying a spectral percept: the chosen
/// [`BehaviorState`] plus the scalar features that drove the decision, so a
/// game NPC (or a designer tuning behavior) can inspect *why* it fired, not
/// just *what* it chose. This struct is what perception now *returns* instead
/// of discarding.
///
/// NOTE: this is a deterministic, hand-written rule classifier over physical
/// spectral features (fire-band energy, laser-band spike, novelty vs. ambient).
/// It is **not** a neural network — there are no learned weights here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreatAssessment {
    /// The consumable decision.
    pub behavior: BehaviorState,
    /// Continuous threat score in `[0, 1]` (used both for the decision and for
    /// blending/animation strength on the game side).
    pub threat: f32,
    /// Mean energy across the fire/heat bands (10–15).
    pub fire_energy: f32,
    /// Sharpness of a single dominant spike (laser-like) over the rest of the
    /// spectrum, in `[0, 1]`. High = monochromatic beam, low = broadband glow.
    pub spike_ratio: f32,
    /// Total radiant energy across all 16 bands.
    pub total_energy: f32,
    /// Index of the brightest band (0 = 380 nm violet … 15 = 755 nm IR).
    pub dominant_band: usize,
}

/// Band conventions (16 bands, 380–755 nm @ 25 nm steps, USGS grid):
/// - 0–4   UV / violet / blue   (radiation)
/// - 5–9   green / cyan / yellow (calm ambient)
/// - 10–15 red / orange / IR     (fire, heat, laser)
const FIRE_BANDS: std::ops::Range<usize> = 10..16;

/// Classify a raw spectral percept (`[u16; 16]`, f16-encoded band energies,
/// matching the engine's on-splat spectral storage) into a [`ThreatAssessment`].
///
/// This is a pure function: identical input always yields identical output.
/// `ambient` is the agent's learned baseline (mean of remembered spectra) used
/// to judge *novelty* — a signature that stands out from the background is more
/// interesting than one that blends in. Pass `[0.0; 16]` for "no baseline".
///
/// Hand-written rules (in priority order):
/// 1. Strong fire/heat field   -> `Flee`   (physical danger to the NPC body)
/// 2. Sharp single-band spike  -> `Alert`  (laser/weapon — directed threat)
/// 3. Novel vs. ambient        -> `Investigate`
/// 4. Some routine energy       -> `Patrol`
/// 5. Near-dark / featureless   -> `Idle`
pub fn classify_spectral(spectral: &[u16; 16], ambient: &[f32; 16]) -> ThreatAssessment {
    // Decode f16-as-u16 into linear band energies, matching the engine's
    // spectral storage convention (vox_core::spectral_damage::decode_spectral_u16).
    let mut bands = [0.0f32; 16];
    for i in 0..16 {
        bands[i] = f16::from_bits(spectral[i]).to_f32().max(0.0);
    }

    let total_energy: f32 = bands.iter().sum();

    let (dominant_band, peak) = bands
        .iter()
        .copied()
        .enumerate()
        .fold((0usize, 0.0f32), |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) });

    let fire_energy: f32 = bands[FIRE_BANDS].iter().sum::<f32>() / FIRE_BANDS.len() as f32;

    // Spike ratio: how concentrated is the spectrum in its single brightest band?
    // A laser is ~all energy in one band (ratio -> 1); a broad glow spreads it out.
    let spike_ratio = if total_energy > 1e-6 {
        let rest = (total_energy - peak).max(0.0);
        // peak vs. the mean of every other band; normalised into [0, 1].
        let rest_mean = rest / 15.0;
        let contrast = (peak - rest_mean).max(0.0);
        (contrast / (peak + rest_mean + 1e-6)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Novelty: the single largest per-band deviation from the learned ambient
    // baseline. Max (not mean) so a real signal concentrated in a few bands
    // still stands out instead of being averaged into the noise floor.
    let novelty: f32 = bands
        .iter()
        .zip(ambient.iter())
        .map(|(&b, &a)| (b - a).abs())
        .fold(0.0f32, f32::max);

    // Continuous threat score blends the physical-danger signals. Fire is the
    // dominant term (it can kill the NPC), a directed laser spike is secondary,
    // novelty contributes only mild unease.
    let threat = (fire_energy * 0.9 + spike_ratio * peak.min(1.0) * 0.2 + novelty * 0.05)
        .clamp(0.0, 1.0);

    let behavior = if fire_energy >= 0.45 {
        BehaviorState::Flee
    } else if spike_ratio >= 0.85 && peak >= 0.5 {
        BehaviorState::Alert
    } else if novelty >= 0.12 {
        BehaviorState::Investigate
    } else if total_energy >= 0.5 {
        BehaviorState::Patrol
    } else {
        BehaviorState::Idle
    };

    ThreatAssessment {
        behavior,
        threat,
        fire_energy,
        spike_ratio,
        total_energy,
        dominant_band,
    }
}

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
    /// The most recent behavior decision this agent reached. `None` until the
    /// first call to [`SpectralPerceptionAgent::assess_threat`]. A game NPC
    /// reads this each frame to decide what to do.
    last_assessment: Option<ThreatAssessment>,
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
            last_assessment: None,
        }
    }

    /// Mean of all remembered spectra — the agent's learned baseline of "what
    /// the world normally looks like here", used as the novelty reference when
    /// classifying. Returns `[0.0; 16]` if the agent has no memory yet.
    pub fn ambient_baseline(&self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        if self.spectral_memory.is_empty() {
            return out;
        }
        for (_, s) in &self.spectral_memory {
            for (acc, &v) in out.iter_mut().zip(s.iter()) {
                *acc += v;
            }
        }
        let n = self.spectral_memory.len() as f32;
        for acc in out.iter_mut() {
            *acc /= n;
        }
        out
    }

    /// Classify a raw spectral percept into a consumable [`ThreatAssessment`],
    /// store it as the agent's current decision, and return it. This is the
    /// retrievable decision API a game NPC calls: it both *acts on* the return
    /// value and can re-read it later via [`SpectralPerceptionAgent::last_assessment`].
    ///
    /// Deterministic: the same `spectral` against the same agent memory always
    /// yields the same assessment. It is a hand-written rule classifier, not a
    /// neural net.
    pub fn assess_threat(&mut self, spectral: &[u16; 16]) -> ThreatAssessment {
        let ambient = self.ambient_baseline();
        let assessment = classify_spectral(spectral, &ambient);
        self.last_assessment = Some(assessment);
        assessment
    }

    /// The agent's current behavior decision, if it has assessed anything yet.
    pub fn last_assessment(&self) -> Option<ThreatAssessment> {
        self.last_assessment
    }

    /// Shorthand: the current [`BehaviorState`] the NPC should act on, or
    /// [`BehaviorState::Idle`] if nothing has been assessed yet.
    pub fn current_behavior(&self) -> BehaviorState {
        self.last_assessment
            .map(|a| a.behavior)
            .unwrap_or(BehaviorState::Idle)
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

    /// Encode f32 band energies into the engine's f16-as-u16 spectral form,
    /// the inverse of how [`classify_spectral`] decodes them.
    fn encode(bands: [f32; 16]) -> [u16; 16] {
        let mut out = [0u16; 16];
        for i in 0..16 {
            out[i] = f16::from_f32(bands[i]).to_bits();
        }
        out
    }

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

    // ---- Behavior classification: the consumable, retrievable decision ----

    fn dark() -> [u16; 16] {
        encode([0.0; 16])
    }

    /// Intense heat field across the fire/IR bands (10–15).
    fn fire() -> [u16; 16] {
        let mut b = [0.0f32; 16];
        b[10..16].fill(0.8);
        encode(b)
    }

    /// A monochromatic laser-like spike in a single red band.
    fn laser() -> [u16; 16] {
        let mut b = [0.0f32; 16];
        b[11] = 0.9;
        encode(b)
    }

    /// Routine broadband green ambient — calm, energetic enough to patrol.
    fn green_ambient() -> [u16; 16] {
        let mut b = [0.0f32; 16];
        b[5] = 0.25;
        b[6] = 0.3;
        b[7] = 0.25;
        encode(b)
    }

    #[test]
    fn fire_field_yields_flee() {
        let a = classify_spectral(&fire(), &[0.0; 16]);
        assert_eq!(a.behavior, BehaviorState::Flee, "intense fire field must trigger Flee");
        assert!(
            a.fire_energy > 0.45,
            "fire_energy should reflect the heat-band field, got {}",
            a.fire_energy
        );
        assert!(a.threat > 0.5, "fire should produce a high threat score, got {}", a.threat);
    }

    #[test]
    fn laser_spike_yields_alert() {
        let a = classify_spectral(&laser(), &[0.0; 16]);
        assert_eq!(
            a.behavior,
            BehaviorState::Alert,
            "a sharp single-band spike must trigger Alert, got {:?}",
            a
        );
        assert_eq!(a.dominant_band, 11, "laser dominant band must be 11, got {}", a.dominant_band);
        assert!(
            a.spike_ratio >= 0.85,
            "laser must register a high spike ratio, got {}",
            a.spike_ratio
        );
    }

    #[test]
    fn dark_field_yields_idle() {
        let a = classify_spectral(&dark(), &[0.0; 16]);
        assert_eq!(a.behavior, BehaviorState::Idle, "a featureless dark field must stay Idle");
        assert_eq!(a.total_energy, 0.0, "dark field must carry zero energy, got {}", a.total_energy);
    }

    #[test]
    fn routine_ambient_yields_patrol_when_it_is_the_baseline() {
        // When the green ambient *is* the agent's learned baseline, novelty is
        // zero so it should fall through to Patrol, not Investigate.
        let g = green_ambient();
        let baseline = {
            let mut out = [0.0f32; 16];
            for i in 0..16 {
                out[i] = f16::from_bits(g[i]).to_f32();
            }
            out
        };
        let a = classify_spectral(&g, &baseline);
        assert_eq!(
            a.behavior,
            BehaviorState::Patrol,
            "familiar broadband ambient should be Patrol, got {:?}",
            a
        );
    }

    #[test]
    fn novel_signal_against_empty_baseline_yields_investigate() {
        // Same green ambient but with NO learned baseline => it is novel =>
        // the agent should go investigate rather than ignore it.
        let a = classify_spectral(&green_ambient(), &[0.0; 16]);
        assert_eq!(
            a.behavior,
            BehaviorState::Investigate,
            "a novel non-threatening signal should trigger Investigate, got {:?}",
            a
        );
    }

    #[test]
    fn different_inputs_yield_different_decisions() {
        let baseline = [0.0f32; 16];
        let decisions = [
            classify_spectral(&dark(), &baseline).behavior,
            classify_spectral(&green_ambient(), &baseline).behavior,
            classify_spectral(&laser(), &baseline).behavior,
            classify_spectral(&fire(), &baseline).behavior,
        ];
        // All four physically distinct spectra map to four distinct behaviors.
        assert_eq!(decisions[0], BehaviorState::Idle);
        assert_eq!(decisions[1], BehaviorState::Investigate);
        assert_eq!(decisions[2], BehaviorState::Alert);
        assert_eq!(decisions[3], BehaviorState::Flee);
        let mut sorted: Vec<BehaviorState> = decisions.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "distinct spectral inputs must yield distinct decisions");
    }

    #[test]
    fn classification_is_deterministic() {
        let input = laser();
        let baseline = [0.05f32; 16];
        let first = classify_spectral(&input, &baseline);
        for _ in 0..16 {
            let again = classify_spectral(&input, &baseline);
            assert_eq!(again, first, "same input must always yield the same assessment");
        }
    }

    #[test]
    fn agent_stores_and_retrieves_decision() {
        let mut agent = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        assert!(agent.last_assessment().is_none(), "fresh agent has no decision yet");
        assert_eq!(agent.current_behavior(), BehaviorState::Idle);

        let returned = agent.assess_threat(&fire());
        assert_eq!(returned.behavior, BehaviorState::Flee, "fire must make the agent flee");

        // The decision is retrievable later, not discarded.
        let stored = agent.last_assessment().expect("decision must be stored after assess_threat");
        assert_eq!(stored, returned, "stored decision must equal the one returned");
        assert_eq!(agent.current_behavior(), BehaviorState::Flee);
    }

    #[test]
    fn agent_memory_shapes_novelty_decision() {
        // Fill the agent's memory with green ambient so it becomes the learned
        // baseline; then the SAME green spectrum is familiar (Patrol), while a
        // fresh agent finds it novel (Investigate). Memory changes the decision.
        let g = green_ambient();
        let g_f32 = {
            let mut out = [0.0f32; 16];
            for i in 0..16 {
                out[i] = f16::from_bits(g[i]).to_f32();
            }
            out
        };

        let mut seasoned = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        for _ in 0..8 {
            seasoned.spectral_memory.push((Vec3::ZERO, g_f32));
        }
        let seasoned_decision = seasoned.assess_threat(&g).behavior;

        let mut fresh = SpectralPerceptionAgent::new(Vec3::ZERO, 10.0);
        let fresh_decision = fresh.assess_threat(&g).behavior;

        assert_eq!(seasoned_decision, BehaviorState::Patrol, "familiar ambient => Patrol");
        assert_eq!(fresh_decision, BehaviorState::Investigate, "unfamiliar ambient => Investigate");
        assert_ne!(
            seasoned_decision, fresh_decision,
            "learned memory must change the agent's decision on identical input"
        );
    }
}
