use serde::{Deserialize, Serialize};

/// A visual animation state machine graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimGraphDefinition {
    pub name: String,
    pub states: Vec<AnimState>,
    pub transitions: Vec<AnimTransition>,
    pub parameters: Vec<AnimParameter>,
    pub default_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimState {
    pub name: String,
    pub clip_path: String,
    pub speed: f32,
    pub looping: bool,
    pub position: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimTransition {
    pub from: String,
    pub to: String,
    pub condition: TransitionConditionDef,
    pub blend_duration: f32,
    pub has_exit_time: bool,
    pub exit_time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionConditionDef {
    BoolParam { name: String, value: bool },
    FloatThreshold { name: String, op: CompareOp, value: f32 },
    Trigger { name: String },
    Always,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompareOp {
    Greater,
    Less,
    Equal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimParameter {
    pub name: String,
    pub param_type: AnimParamType,
    pub default: AnimParamValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimParamType {
    Bool,
    Float,
    Trigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimParamValue {
    Bool(bool),
    Float(f32),
    Trigger(bool),
}

/// Blend space — 2D grid for blending animations by two parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendSpace2D {
    pub name: String,
    pub x_param: String,
    pub y_param: String,
    pub samples: Vec<BlendSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendSample {
    pub position: [f32; 2],
    pub clip_path: String,
    pub weight: f32,
}

impl BlendSpace2D {
    pub fn new(name: &str, x_param: &str, y_param: &str) -> Self {
        Self {
            name: name.to_string(),
            x_param: x_param.to_string(),
            y_param: y_param.to_string(),
            samples: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, position: [f32; 2], clip_path: &str) {
        self.samples.push(BlendSample {
            position,
            clip_path: clip_path.to_string(),
            weight: 1.0,
        });
    }

    /// Evaluate the blend space at (x, y), returning clip paths with weights
    /// based on inverse-distance weighting.
    pub fn evaluate(&self, x: f32, y: f32) -> Vec<(String, f32)> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        // Check for exact match first
        for sample in &self.samples {
            let dx = sample.position[0] - x;
            let dy = sample.position[1] - y;
            if dx * dx + dy * dy < 1e-10 {
                return vec![(sample.clip_path.clone(), 1.0)];
            }
        }

        // Inverse-distance weighting
        let mut weights: Vec<f32> = Vec::new();
        let mut total = 0.0_f32;
        for sample in &self.samples {
            let dx = sample.position[0] - x;
            let dy = sample.position[1] - y;
            let dist = (dx * dx + dy * dy).sqrt();
            let w = 1.0 / dist;
            weights.push(w);
            total += w;
        }

        if total < 1e-10 {
            return Vec::new();
        }

        self.samples
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| (s.clip_path.clone(), w / total))
            .collect()
    }
}

impl AnimGraphDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            states: Vec::new(),
            transitions: Vec::new(),
            parameters: Vec::new(),
            default_state: String::new(),
        }
    }

    pub fn add_state(&mut self, name: &str, clip: &str, looping: bool) {
        if self.states.is_empty() {
            self.default_state = name.to_string();
        }
        self.states.push(AnimState {
            name: name.to_string(),
            clip_path: clip.to_string(),
            speed: 1.0,
            looping,
            position: [0.0, 0.0],
        });
    }

    pub fn add_transition(
        &mut self,
        from: &str,
        to: &str,
        condition: TransitionConditionDef,
        blend: f32,
    ) {
        self.transitions.push(AnimTransition {
            from: from.to_string(),
            to: to.to_string(),
            condition,
            blend_duration: blend,
            has_exit_time: false,
            exit_time: 1.0,
        });
    }

    pub fn add_parameter(&mut self, name: &str, param_type: AnimParamType, default: AnimParamValue) {
        self.parameters.push(AnimParameter {
            name: name.to_string(),
            param_type,
            default,
        });
    }

    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn load_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph() {
        let graph = AnimGraphDefinition::new("PlayerAnim");
        assert_eq!(graph.name, "PlayerAnim");
        assert_eq!(graph.state_count(), 0);
        assert_eq!(graph.transition_count(), 0);
    }

    #[test]
    fn test_add_states_and_transitions() {
        let mut graph = AnimGraphDefinition::new("Player");
        graph.add_state("Idle", "anims/idle.vxa", true);
        graph.add_state("Run", "anims/run.vxa", true);
        graph.add_state("Jump", "anims/jump.vxa", false);

        graph.add_transition(
            "Idle",
            "Run",
            TransitionConditionDef::FloatThreshold {
                name: "speed".into(),
                op: CompareOp::Greater,
                value: 0.1,
            },
            0.2,
        );
        graph.add_transition(
            "Run",
            "Jump",
            TransitionConditionDef::Trigger { name: "jump".into() },
            0.1,
        );

        assert_eq!(graph.state_count(), 3);
        assert_eq!(graph.transition_count(), 2);
    }

    #[test]
    fn test_parameter_types() {
        let mut graph = AnimGraphDefinition::new("Test");
        graph.add_parameter("is_grounded", AnimParamType::Bool, AnimParamValue::Bool(true));
        graph.add_parameter("speed", AnimParamType::Float, AnimParamValue::Float(0.0));
        graph.add_parameter("jump", AnimParamType::Trigger, AnimParamValue::Trigger(false));

        assert_eq!(graph.parameters.len(), 3);
        assert_eq!(graph.parameters[0].param_type, AnimParamType::Bool);
        assert_eq!(graph.parameters[1].param_type, AnimParamType::Float);
        assert_eq!(graph.parameters[2].param_type, AnimParamType::Trigger);
    }

    #[test]
    fn test_blend_space_evaluation() {
        let mut bs = BlendSpace2D::new("Locomotion", "speed", "direction");
        bs.add_sample([0.0, 0.0], "anims/idle.vxa");
        bs.add_sample([1.0, 0.0], "anims/walk.vxa");
        bs.add_sample([2.0, 0.0], "anims/run.vxa");

        // At exact sample point
        let result = bs.evaluate(0.0, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "anims/idle.vxa");
        assert!((result[0].1 - 1.0).abs() < 0.001);

        // Between samples — should blend
        let result = bs.evaluate(0.5, 0.0);
        assert!(result.len() > 1);
        let total_weight: f32 = result.iter().map(|(_, w)| w).sum();
        assert!((total_weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_json_round_trip() {
        let mut graph = AnimGraphDefinition::new("Serialize");
        graph.add_state("Idle", "idle.vxa", true);
        graph.add_state("Walk", "walk.vxa", true);
        graph.add_transition(
            "Idle",
            "Walk",
            TransitionConditionDef::BoolParam {
                name: "moving".into(),
                value: true,
            },
            0.3,
        );
        graph.add_parameter("moving", AnimParamType::Bool, AnimParamValue::Bool(false));

        let json = graph.save_json().unwrap();
        let loaded = AnimGraphDefinition::load_json(&json).unwrap();
        assert_eq!(loaded.name, "Serialize");
        assert_eq!(loaded.state_count(), 2);
        assert_eq!(loaded.transition_count(), 1);
        assert_eq!(loaded.parameters.len(), 1);
    }

    #[test]
    fn test_default_state_set() {
        let mut graph = AnimGraphDefinition::new("Test");
        assert!(graph.default_state.is_empty());

        graph.add_state("Idle", "idle.vxa", true);
        assert_eq!(graph.default_state, "Idle");

        graph.add_state("Run", "run.vxa", true);
        // Default should still be the first one
        assert_eq!(graph.default_state, "Idle");
    }
}
