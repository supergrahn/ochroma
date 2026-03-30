use glam::Vec3;

/// A morph target — a named set of per-vertex position deltas.
pub struct MorphTarget {
    pub name: String,
    pub deltas: Vec<Vec3>,
}

/// Pre-defined facial expression presets (ARKit-compatible blend-shape names).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacialExpression {
    Neutral,
    Smile,
    Frown,
    Surprise,
    Angry,
    Blink,
}

/// Facial animation controller — blends between morph targets by weight.
pub struct FacialController {
    pub targets: Vec<MorphTarget>,
    pub weights: Vec<f32>,
}

impl FacialController {
    /// Create an empty controller with no targets.
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a morph target with an initial weight of 0.
    pub fn add_target(&mut self, name: &str, deltas: Vec<Vec3>) {
        self.targets.push(MorphTarget {
            name: name.to_string(),
            deltas,
        });
        self.weights.push(0.0);
    }

    /// Set the blend weight for a named target.  Silently ignored if not found.
    pub fn set_weight(&mut self, name: &str, weight: f32) {
        for (i, target) in self.targets.iter().enumerate() {
            if target.name == name {
                self.weights[i] = weight.clamp(0.0, 1.0);
                return;
            }
        }
    }

    /// Get the current blend weight for a named target (0.0 if not found).
    pub fn get_weight(&self, name: &str) -> f32 {
        for (i, target) in self.targets.iter().enumerate() {
            if target.name == name {
                return self.weights[i];
            }
        }
        0.0
    }

    /// Compute the blended position delta for a single vertex index.
    pub fn evaluate(&self, vertex_index: usize) -> Vec3 {
        let mut result = Vec3::ZERO;
        for (target, &weight) in self.targets.iter().zip(&self.weights) {
            if weight > 0.001 && vertex_index < target.deltas.len() {
                result += target.deltas[vertex_index] * weight;
            }
        }
        result
    }

    /// Apply all active morph targets to a set of base positions, returning new positions.
    pub fn apply(&self, base_positions: &[Vec3]) -> Vec<Vec3> {
        base_positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| pos + self.evaluate(i))
            .collect()
    }

    /// Apply a pre-defined expression preset.  Resets all weights first.
    pub fn set_expression(&mut self, expression: FacialExpression) {
        // Reset all weights to zero.
        for w in &mut self.weights {
            *w = 0.0;
        }
        match expression {
            FacialExpression::Neutral => {}
            FacialExpression::Smile => {
                self.set_weight("mouthSmileLeft", 1.0);
                self.set_weight("mouthSmileRight", 1.0);
                self.set_weight("cheekPuff", 0.3);
            }
            FacialExpression::Frown => {
                self.set_weight("mouthFrownLeft", 1.0);
                self.set_weight("mouthFrownRight", 1.0);
                self.set_weight("browDownLeft", 0.5);
                self.set_weight("browDownRight", 0.5);
            }
            FacialExpression::Surprise => {
                self.set_weight("browInnerUp", 1.0);
                self.set_weight("eyeWideLeft", 1.0);
                self.set_weight("eyeWideRight", 1.0);
                self.set_weight("jawOpen", 0.5);
            }
            FacialExpression::Angry => {
                self.set_weight("browDownLeft", 1.0);
                self.set_weight("browDownRight", 1.0);
                self.set_weight("noseSneerLeft", 0.5);
                self.set_weight("mouthFrownLeft", 0.7);
                self.set_weight("mouthFrownRight", 0.7);
            }
            FacialExpression::Blink => {
                self.set_weight("eyeBlinkLeft", 1.0);
                self.set_weight("eyeBlinkRight", 1.0);
            }
        }
    }
}

impl Default for FacialController {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// FACS Action Units
// ═════════════════════════════════════════════════════════════════════════════

/// A single FACS action unit with a normalized weight.
#[derive(Debug, Clone)]
pub struct ActionUnit {
    pub id: u8,
    pub name: String,
    pub weight: f32,
}

impl ActionUnit {
    pub fn new(id: u8, name: impl Into<String>) -> Self {
        Self { id, name: name.into(), weight: 0.0 }
    }
}

/// Maps one action unit to one morph target with a scalar influence.
#[derive(Debug, Clone)]
pub struct AuMorphMapping {
    pub au_id: u8,
    pub morph_name: String,
    pub influence: f32,
}

/// 44 standard FACS action units mapped to morph targets.
pub struct FacialRig {
    pub action_units: Vec<ActionUnit>,
    pub au_to_morph: Vec<AuMorphMapping>,
}

impl FacialRig {
    pub fn new() -> Self {
        let action_units = vec![
            ActionUnit::new(1,  "AU01_inner_brow_raiser"),
            ActionUnit::new(2,  "AU02_outer_brow_raiser"),
            ActionUnit::new(4,  "AU04_brow_lowerer"),
            ActionUnit::new(5,  "AU05_upper_lid_raiser"),
            ActionUnit::new(6,  "AU06_cheek_raiser"),
            ActionUnit::new(7,  "AU07_lid_tightener"),
            ActionUnit::new(9,  "AU09_nose_wrinkler"),
            ActionUnit::new(10, "AU10_upper_lip_raiser"),
            ActionUnit::new(11, "AU11_nasolabial_deepener"),
            ActionUnit::new(12, "AU12_lip_corner_puller"),
            ActionUnit::new(13, "AU13_cheek_puffer"),
            ActionUnit::new(14, "AU14_dimpler"),
            ActionUnit::new(15, "AU15_lip_corner_depressor"),
            ActionUnit::new(16, "AU16_lower_lip_depressor"),
            ActionUnit::new(17, "AU17_chin_raiser"),
            ActionUnit::new(18, "AU18_lip_puckerer"),
            ActionUnit::new(20, "AU20_lip_stretcher"),
            ActionUnit::new(22, "AU22_lip_funneler"),
            ActionUnit::new(23, "AU23_lip_tightener"),
            ActionUnit::new(24, "AU24_lip_pressor"),
            ActionUnit::new(25, "AU25_lips_part"),
            ActionUnit::new(26, "AU26_jaw_drop"),
            ActionUnit::new(27, "AU27_mouth_stretch"),
            ActionUnit::new(28, "AU28_lip_suck"),
            ActionUnit::new(41, "AU41_lid_droop"),
            ActionUnit::new(42, "AU42_slit"),
            ActionUnit::new(43, "AU43_eyes_closed"),
            ActionUnit::new(44, "AU44_squint"),
            ActionUnit::new(45, "AU45_blink"),
            ActionUnit::new(46, "AU46_wink"),
            ActionUnit::new(51, "AU51_head_turn_left"),
            ActionUnit::new(52, "AU52_head_turn_right"),
            ActionUnit::new(53, "AU53_head_up"),
            ActionUnit::new(54, "AU54_head_down"),
            ActionUnit::new(55, "AU55_head_tilt_left"),
            ActionUnit::new(56, "AU56_head_tilt_right"),
            ActionUnit::new(57, "AU57_forward"),
            ActionUnit::new(58, "AU58_back"),
            ActionUnit::new(61, "AU61_eyes_turn_left"),
            ActionUnit::new(62, "AU62_eyes_turn_right"),
            ActionUnit::new(63, "AU63_eyes_up"),
            ActionUnit::new(64, "AU64_eyes_down"),
            ActionUnit::new(65, "AU65_walleye"),
            ActionUnit::new(66, "AU66_cross_eye"),
        ];
        Self { action_units, au_to_morph: Vec::new() }
    }

    pub fn set_au(&mut self, au_id: u8, weight: f32) {
        if let Some(au) = self.action_units.iter_mut().find(|a| a.id == au_id) {
            au.weight = weight.clamp(0.0, 1.0);
        }
    }

    pub fn get_au(&self, au_id: u8) -> f32 {
        self.action_units.iter().find(|a| a.id == au_id).map(|a| a.weight).unwrap_or(0.0)
    }

    pub fn add_mapping(&mut self, au_id: u8, morph_name: impl Into<String>, influence: f32) {
        self.au_to_morph.push(AuMorphMapping { au_id, morph_name: morph_name.into(), influence });
    }

    /// Collapse AU weights into final morph weights.
    /// Returns Vec<(morph_name, weight)> sorted by morph_name.
    pub fn compute_morph_weights(&self) -> Vec<(String, f32)> {
        let mut morph_weights: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for mapping in &self.au_to_morph {
            let au_weight = self.get_au(mapping.au_id);
            *morph_weights.entry(mapping.morph_name.clone()).or_insert(0.0)
                += au_weight * mapping.influence;
        }
        let mut result: Vec<(String, f32)> = morph_weights.into_iter()
            .map(|(k, v)| (k, v.clamp(0.0, 1.0)))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

impl Default for FacialRig {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Phoneme classifier
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phoneme {
    Silence, P, B, M, F, V, Th, D, T, N, S, Z, Sh, Ch, Jh, G, K, Ng,
    Ah, Ae, Ey, Ih, Iy, Oh, Ow, Uh, Uw, Er, Aa, Aw, Oy, Ay,
}

/// Lightweight phoneme classifier using precomputed MFCC features.
/// Stores a linear weight matrix (39-input, 32-hidden, 32-phoneme output).
pub struct PhonemeClassifier {
    /// Weight matrix rows: 32 hidden neurons, each with 39 input weights + 1 bias = 40 values
    hidden_weights: Vec<[f32; 40]>,
    /// Output weights: 32 phoneme classes, each with 32 hidden weights + 1 bias = 33 values
    output_weights: Vec<[f32; 33]>,
    /// MFCC frame buffer: 3 frames x 13 coefficients = 39-element context
    frame_buffer: std::collections::VecDeque<[f32; 13]>,
}

impl PhonemeClassifier {
    /// Create with random-initialized weights (production would load from include_bytes!).
    pub fn new() -> Self {
        let hidden_weights: Vec<[f32; 40]> = (0..32)
            .map(|i| {
                let mut w = [0.0f32; 40];
                for (j, v) in w.iter_mut().enumerate() {
                    *v = ((i * 37 + j * 13) as f32 * 0.001) - 0.5;
                }
                w
            })
            .collect();
        let output_weights: Vec<[f32; 33]> = (0..32)
            .map(|i| {
                let mut w = [0.0f32; 33];
                for (j, v) in w.iter_mut().enumerate() {
                    *v = ((i * 31 + j * 17) as f32 * 0.001) - 0.5;
                }
                w
            })
            .collect();
        Self {
            hidden_weights,
            output_weights,
            frame_buffer: std::collections::VecDeque::with_capacity(3),
        }
    }

    /// Classify one 25ms PCM frame into a phoneme.
    /// Input: mono PCM samples at 16kHz.
    pub fn classify_frame(&mut self, pcm: &[f32]) -> Phoneme {
        let mfcc = self.compute_mfcc(pcm);
        self.frame_buffer.push_back(mfcc);
        while self.frame_buffer.len() > 3 { self.frame_buffer.pop_front(); }

        let mut feature = [0.0f32; 39];
        for (frame_idx, frame) in self.frame_buffer.iter().enumerate() {
            for (coeff_idx, &v) in frame.iter().enumerate() {
                feature[frame_idx * 13 + coeff_idx] = v;
            }
        }

        self.forward(&feature)
    }

    fn compute_mfcc(&self, pcm: &[f32]) -> [f32; 13] {
        let n = pcm.len().min(400);
        let mut mfcc = [0.0f32; 13];

        if n == 0 { return mfcc; }

        let band_size = n / 13;
        for (band, mfcc_val) in mfcc.iter_mut().enumerate() {
            let start = band * band_size;
            let end = (start + band_size).min(n);
            if end > start {
                let energy: f32 = pcm[start..end].iter().map(|&x| x * x).sum::<f32>()
                    / (end - start) as f32;
                *mfcc_val = (energy.max(1e-10)).ln();
            }
        }

        mfcc
    }

    fn forward(&self, feature: &[f32; 39]) -> Phoneme {
        let mut hidden = [0.0f32; 32];
        for (i, neuron) in self.hidden_weights.iter().enumerate() {
            let mut sum = neuron[39]; // bias
            for (j, &x) in feature.iter().enumerate() {
                sum += neuron[j] * x;
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        let mut logits = [0.0f32; 32];
        for (i, neuron) in self.output_weights.iter().enumerate() {
            let mut sum = neuron[32]; // bias
            for (j, &h) in hidden.iter().enumerate() {
                sum += neuron[j] * h;
            }
            logits[i] = sum;
        }

        let best = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        const PHONEMES: [Phoneme; 32] = [
            Phoneme::Silence, Phoneme::P, Phoneme::B, Phoneme::M, Phoneme::F, Phoneme::V,
            Phoneme::Th, Phoneme::D, Phoneme::T, Phoneme::N, Phoneme::S, Phoneme::Z,
            Phoneme::Sh, Phoneme::Ch, Phoneme::Jh, Phoneme::G, Phoneme::K, Phoneme::Ng,
            Phoneme::Ah, Phoneme::Ae, Phoneme::Ey, Phoneme::Ih, Phoneme::Iy, Phoneme::Oh,
            Phoneme::Ow, Phoneme::Uh, Phoneme::Uw, Phoneme::Er, Phoneme::Aa, Phoneme::Aw,
            Phoneme::Oy, Phoneme::Ay,
        ];
        PHONEMES[best]
    }
}

impl Default for PhonemeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Viseme table
// ═════════════════════════════════════════════════════════════════════════════

pub struct VisemeTable {
    pub phoneme_to_au_weights: std::collections::HashMap<Phoneme, Vec<(u8, f32)>>,
}

impl VisemeTable {
    pub fn default_english() -> Self {
        let mut map = std::collections::HashMap::new();
        map.insert(Phoneme::Silence, vec![]);
        map.insert(Phoneme::P,  vec![(20u8, 0.5), (24u8, 0.8)]);
        map.insert(Phoneme::B,  vec![(20u8, 0.5), (24u8, 0.8)]);
        map.insert(Phoneme::M,  vec![(24u8, 0.9)]);
        map.insert(Phoneme::F,  vec![(20u8, 0.3), (10u8, 0.4)]);
        map.insert(Phoneme::V,  vec![(20u8, 0.3), (10u8, 0.4)]);
        map.insert(Phoneme::D,  vec![(25u8, 0.3), (26u8, 0.2)]);
        map.insert(Phoneme::T,  vec![(25u8, 0.2)]);
        map.insert(Phoneme::N,  vec![(25u8, 0.3)]);
        map.insert(Phoneme::Sh, vec![(22u8, 0.6), (25u8, 0.3)]);
        map.insert(Phoneme::G,  vec![(26u8, 0.3)]);
        map.insert(Phoneme::K,  vec![(26u8, 0.3)]);
        map.insert(Phoneme::Ah, vec![(26u8, 0.7), (25u8, 0.5)]);
        map.insert(Phoneme::Iy, vec![(20u8, 0.7), (6u8, 0.3)]);
        map.insert(Phoneme::Uw, vec![(20u8, 0.8), (25u8, 0.4)]);
        map.insert(Phoneme::Ow, vec![(22u8, 0.7), (26u8, 0.4)]);
        map.insert(Phoneme::Er, vec![(22u8, 0.4), (26u8, 0.3)]);
        map.insert(Phoneme::Ay, vec![(26u8, 0.5), (20u8, 0.4)]);
        Self { phoneme_to_au_weights: map }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Audio-driven lip sync
// ═════════════════════════════════════════════════════════════════════════════

pub struct AudioLipSync {
    pub classifier: PhonemeClassifier,
    pub rig: FacialRig,
    pub viseme_table: VisemeTable,
    pub smoothing: f32,
}

impl AudioLipSync {
    pub fn new() -> Self {
        Self {
            classifier: PhonemeClassifier::new(),
            rig: FacialRig::new(),
            viseme_table: VisemeTable::default_english(),
            smoothing: 0.7,
        }
    }

    /// Process one audio frame (25ms PCM) and update the facial rig.
    pub fn process_audio_frame(&mut self, pcm: &[f32]) {
        let phoneme = self.classifier.classify_frame(pcm);
        let au_weights = self.viseme_table.phoneme_to_au_weights
            .get(&phoneme)
            .cloned()
            .unwrap_or_default();

        for au in &mut self.rig.action_units {
            let target = au_weights.iter()
                .find(|(id, _)| *id == au.id)
                .map(|(_, w)| *w)
                .unwrap_or(0.0);
            au.weight = au.weight * self.smoothing + target * (1.0 - self.smoothing);
        }
    }
}

impl Default for AudioLipSync {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Spectral emotion mapping
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionState { Neutral, Anger, Sadness, Fear, Joy }

pub struct SpectralEmotionMapping {
    pub emotion: EmotionState,
    pub spectral_bias: [f32; 8],
}

impl SpectralEmotionMapping {
    pub fn for_emotion(emotion: EmotionState) -> Self {
        let bias = match emotion {
            EmotionState::Neutral => [0.0; 8],
            EmotionState::Anger   => [0.0, 0.0, 0.0, 0.02, 0.03, 0.02, 0.0, 0.0],
            EmotionState::Joy     => [0.0, 0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0],
            EmotionState::Fear    => [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            EmotionState::Sadness => [0.0; 8],
        };
        Self { emotion, spectral_bias: bias }
    }

    /// Apply spectral bias to all 8 bands of a splat's spectral values.
    pub fn apply(&self, face_splat_spectral: &mut [f32; 8]) {
        for (b, bias) in self.spectral_bias.iter().enumerate() {
            face_splat_spectral[b] = (face_splat_spectral[b] + bias).clamp(0.0, 1.0);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests ────────────────────────────────────────────────────────

    fn make_controller() -> FacialController {
        let mut fc = FacialController::new();
        let names = [
            "mouthSmileLeft",
            "mouthSmileRight",
            "cheekPuff",
            "mouthFrownLeft",
            "mouthFrownRight",
            "browDownLeft",
            "browDownRight",
            "browInnerUp",
            "eyeWideLeft",
            "eyeWideRight",
            "jawOpen",
            "noseSneerLeft",
            "eyeBlinkLeft",
            "eyeBlinkRight",
        ];
        for name in &names {
            let deltas = vec![
                Vec3::new(0.1, 0.0, 0.0),
                Vec3::new(0.0, 0.2, 0.0),
                Vec3::new(0.0, 0.0, 0.3),
                Vec3::new(0.1, 0.1, 0.1),
            ];
            fc.add_target(name, deltas);
        }
        fc
    }

    #[test]
    fn neutral_has_zero_deltas() {
        let mut fc = make_controller();
        fc.set_expression(FacialExpression::Neutral);
        for i in 0..4 {
            let d = fc.evaluate(i);
            assert!(d.length() < 1e-6, "Neutral should produce zero deltas");
        }
    }

    #[test]
    fn smile_sets_mouth_weights() {
        let mut fc = make_controller();
        fc.set_expression(FacialExpression::Smile);
        assert!((fc.get_weight("mouthSmileLeft") - 1.0).abs() < 1e-6);
        assert!((fc.get_weight("mouthSmileRight") - 1.0).abs() < 1e-6);
        assert!((fc.get_weight("cheekPuff") - 0.3).abs() < 1e-6);
        assert!(fc.get_weight("browDownLeft").abs() < 1e-6);
    }

    #[test]
    fn evaluate_returns_weighted_sum() {
        let mut fc = FacialController::new();
        fc.add_target("a", vec![Vec3::new(1.0, 0.0, 0.0)]);
        fc.add_target("b", vec![Vec3::new(0.0, 2.0, 0.0)]);
        fc.set_weight("a", 0.5);
        fc.set_weight("b", 0.25);
        let d = fc.evaluate(0);
        assert!((d.x - 0.5).abs() < 1e-6);
        assert!((d.y - 0.5).abs() < 1e-6);
        assert!(d.z.abs() < 1e-6);
    }

    #[test]
    fn apply_modifies_positions() {
        let mut fc = FacialController::new();
        fc.add_target("smile", vec![Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, 0.2, 0.0)]);
        fc.set_weight("smile", 1.0);

        let base = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        let result = fc.apply(&base);
        assert!((result[0].y - 2.1).abs() < 1e-6);
        assert!((result[1].y - 5.2).abs() < 1e-6);
        assert!((result[0].x - 1.0).abs() < 1e-6);
        assert!((result[1].z - 6.0).abs() < 1e-6);
    }

    #[test]
    fn set_and_get_weight() {
        let mut fc = FacialController::new();
        fc.add_target("test", vec![Vec3::ZERO]);
        fc.set_weight("test", 0.75);
        assert!((fc.get_weight("test") - 0.75).abs() < 1e-6);
    }

    #[test]
    fn expression_presets_set_correct_targets() {
        let mut fc = make_controller();

        fc.set_expression(FacialExpression::Frown);
        assert!((fc.get_weight("mouthFrownLeft") - 1.0).abs() < 1e-6);
        assert!((fc.get_weight("browDownLeft") - 0.5).abs() < 1e-6);

        fc.set_expression(FacialExpression::Surprise);
        assert!(fc.get_weight("mouthFrownLeft").abs() < 1e-6);
        assert!((fc.get_weight("browInnerUp") - 1.0).abs() < 1e-6);
        assert!((fc.get_weight("jawOpen") - 0.5).abs() < 1e-6);

        fc.set_expression(FacialExpression::Angry);
        assert!((fc.get_weight("browDownLeft") - 1.0).abs() < 1e-6);
        assert!((fc.get_weight("noseSneerLeft") - 0.5).abs() < 1e-6);

        fc.set_expression(FacialExpression::Blink);
        assert!((fc.get_weight("eyeBlinkLeft") - 1.0).abs() < 1e-6);
        assert!((fc.get_weight("eyeBlinkRight") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn missing_target_weight_is_zero() {
        let fc = FacialController::new();
        assert!(fc.get_weight("nonexistent").abs() < 1e-6);
    }

    #[test]
    fn weight_clamped_to_01() {
        let mut fc = FacialController::new();
        fc.add_target("test", vec![Vec3::ZERO]);
        fc.set_weight("test", 2.0);
        assert!((fc.get_weight("test") - 1.0).abs() < 1e-6);
        fc.set_weight("test", -0.5);
        assert!(fc.get_weight("test").abs() < 1e-6);
    }

    // ── New FACS / phoneme tests ──────────────────────────────────────────────

    #[test]
    fn facial_rig_has_44_aus() {
        assert_eq!(FacialRig::new().action_units.len(), 44);
    }

    #[test]
    fn facial_rig_set_and_get_au() {
        let mut rig = FacialRig::new();
        rig.set_au(12, 0.7);
        assert!((rig.get_au(12) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn facial_rig_compute_morph_weights_sums_correctly() {
        let mut rig = FacialRig::new();
        rig.add_mapping(12, "smile_morph", 1.0);
        rig.set_au(12, 0.8);
        let weights = rig.compute_morph_weights();
        let w = weights.iter().find(|(name, _)| name == "smile_morph").map(|(_, w)| *w).unwrap();
        assert!((w - 0.8).abs() < 1e-5, "expected ~0.8, got {w}");
    }

    #[test]
    fn phoneme_classifier_returns_a_phoneme() {
        let mut clf = PhonemeClassifier::new();
        let silence = vec![0.0f32; 400];
        // Just verify it doesn't panic and returns a valid variant
        let _p: Phoneme = clf.classify_frame(&silence);
    }

    #[test]
    fn viseme_table_has_english_phonemes() {
        let table = VisemeTable::default_english();
        assert!(table.phoneme_to_au_weights.contains_key(&Phoneme::Silence));
        assert!(table.phoneme_to_au_weights.contains_key(&Phoneme::Ah));
    }

    #[test]
    fn spectral_emotion_mapping_anger_boosts_red() {
        let mapping = SpectralEmotionMapping::for_emotion(EmotionState::Anger);
        assert!(mapping.spectral_bias[4] > 0.0, "anger should boost band 4");
    }

    #[test]
    fn spectral_emotion_mapping_apply_clamps() {
        let mapping = SpectralEmotionMapping::for_emotion(EmotionState::Anger);
        let mut spectral = [0.99f32; 8];
        mapping.apply(&mut spectral);
        for &v in &spectral {
            assert!(v <= 1.0, "apply must clamp to [0,1], got {v}");
        }
    }

    #[test]
    fn audio_lip_sync_processes_frame() {
        let mut als = AudioLipSync::new();
        let silence = vec![0.0f32; 400];
        als.process_audio_frame(&silence); // must not panic
    }
}
