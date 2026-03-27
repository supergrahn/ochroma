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
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a controller with common ARKit-style blend shapes.
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
            // 4 vertices, each target moves them differently
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
        // Others should be zero
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
        // X and Z unchanged
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
        // Frown weights should be reset
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
}
