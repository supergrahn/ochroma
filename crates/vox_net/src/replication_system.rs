use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Replicated entity state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatedState {
    pub entity_id: u32,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub velocity: [f32; 3],
    pub custom_data: HashMap<String, serde_json::Value>,
    pub sequence: u64,
}

/// Delta compression — only send what changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDelta {
    pub entity_id: u32,
    pub sequence: u64,
    pub position: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub velocity: Option<[f32; 3]>,
    pub custom_changes: HashMap<String, serde_json::Value>,
}

impl StateDelta {
    /// Compute delta between old and new state.
    pub fn compute(old: &ReplicatedState, new: &ReplicatedState) -> Self {
        let pos_changed = old.position != new.position;
        let rot_changed = old.rotation != new.rotation;
        let vel_changed = old.velocity != new.velocity;

        let mut custom_changes = HashMap::new();
        for (k, v) in &new.custom_data {
            if old.custom_data.get(k) != Some(v) {
                custom_changes.insert(k.clone(), v.clone());
            }
        }

        Self {
            entity_id: new.entity_id,
            sequence: new.sequence,
            position: if pos_changed { Some(new.position) } else { None },
            rotation: if rot_changed { Some(new.rotation) } else { None },
            velocity: if vel_changed { Some(new.velocity) } else { None },
            custom_changes,
        }
    }

    /// Apply delta to a state.
    pub fn apply(&self, state: &mut ReplicatedState) {
        state.sequence = self.sequence;
        if let Some(pos) = self.position {
            state.position = pos;
        }
        if let Some(rot) = self.rotation {
            state.rotation = rot;
        }
        if let Some(vel) = self.velocity {
            state.velocity = vel;
        }
        for (k, v) in &self.custom_changes {
            state.custom_data.insert(k.clone(), v.clone());
        }
    }

    /// Is this delta empty (nothing changed)?
    pub fn is_empty(&self) -> bool {
        self.position.is_none()
            && self.rotation.is_none()
            && self.velocity.is_none()
            && self.custom_changes.is_empty()
    }

    /// Serialized size estimate (for bandwidth tracking).
    pub fn estimated_bytes(&self) -> usize {
        let mut size = 12; // entity_id + sequence
        if self.position.is_some() {
            size += 12;
        }
        if self.rotation.is_some() {
            size += 16;
        }
        if self.velocity.is_some() {
            size += 12;
        }
        size += self.custom_changes.len() * 32; // rough estimate
        size
    }
}

/// Client-side prediction — apply input immediately, reconcile with server.
pub struct ClientPrediction {
    pub pending_inputs: Vec<PredictedInput>,
    pub last_confirmed_sequence: u64,
}

#[derive(Debug, Clone)]
pub struct PredictedInput {
    pub sequence: u64,
    pub input: [f32; 3],
    pub timestamp: f64,
}

impl ClientPrediction {
    pub fn new() -> Self {
        Self {
            pending_inputs: Vec::new(),
            last_confirmed_sequence: 0,
        }
    }

    pub fn add_input(&mut self, sequence: u64, input: [f32; 3], timestamp: f64) {
        self.pending_inputs.push(PredictedInput {
            sequence,
            input,
            timestamp,
        });
    }

    /// Server confirmed up to this sequence. Remove confirmed inputs.
    pub fn confirm(&mut self, server_sequence: u64) {
        self.last_confirmed_sequence = server_sequence;
        self.pending_inputs.retain(|i| i.sequence > server_sequence);
    }

    /// Re-apply unconfirmed inputs after server reconciliation.
    pub fn replay_unconfirmed(&self) -> Vec<[f32; 3]> {
        self.pending_inputs.iter().map(|i| i.input).collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_inputs.len()
    }
}

impl Default for ClientPrediction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(entity_id: u32, pos: [f32; 3], seq: u64) -> ReplicatedState {
        ReplicatedState {
            entity_id,
            position: pos,
            rotation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 3],
            custom_data: HashMap::new(),
            sequence: seq,
        }
    }

    #[test]
    fn delta_detects_position_change() {
        let old = make_state(1, [0.0, 0.0, 0.0], 1);
        let new = make_state(1, [1.0, 2.0, 3.0], 2);
        let delta = StateDelta::compute(&old, &new);
        assert!(delta.position.is_some());
        assert_eq!(delta.position.unwrap(), [1.0, 2.0, 3.0]);
        assert!(!delta.is_empty());
    }

    #[test]
    fn empty_delta_when_unchanged() {
        let state = make_state(1, [1.0, 2.0, 3.0], 5);
        let delta = StateDelta::compute(&state, &state);
        assert!(delta.is_empty());
    }

    #[test]
    fn apply_delta_updates_state() {
        let mut state = make_state(1, [0.0, 0.0, 0.0], 1);
        let delta = StateDelta {
            entity_id: 1,
            sequence: 5,
            position: Some([10.0, 20.0, 30.0]),
            rotation: None,
            velocity: Some([1.0, 0.0, 0.0]),
            custom_changes: HashMap::new(),
        };
        delta.apply(&mut state);
        assert_eq!(state.position, [10.0, 20.0, 30.0]);
        assert_eq!(state.velocity, [1.0, 0.0, 0.0]);
        assert_eq!(state.sequence, 5);
        // rotation unchanged
        assert_eq!(state.rotation, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn prediction_add_and_confirm() {
        let mut pred = ClientPrediction::new();
        pred.add_input(1, [1.0, 0.0, 0.0], 0.0);
        pred.add_input(2, [0.0, 1.0, 0.0], 0.016);
        pred.add_input(3, [0.0, 0.0, 1.0], 0.032);
        assert_eq!(pred.pending_count(), 3);

        pred.confirm(2);
        assert_eq!(pred.pending_count(), 1);
        assert_eq!(pred.last_confirmed_sequence, 2);
    }

    #[test]
    fn replay_returns_unconfirmed() {
        let mut pred = ClientPrediction::new();
        pred.add_input(1, [1.0, 0.0, 0.0], 0.0);
        pred.add_input(2, [0.0, 1.0, 0.0], 0.016);
        pred.add_input(3, [0.0, 0.0, 1.0], 0.032);
        pred.confirm(1);

        let inputs = pred.replay_unconfirmed();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], [0.0, 1.0, 0.0]);
        assert_eq!(inputs[1], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn bandwidth_estimate() {
        let delta = StateDelta {
            entity_id: 1,
            sequence: 10,
            position: Some([1.0, 2.0, 3.0]),
            rotation: Some([0.0, 0.0, 0.0, 1.0]),
            velocity: None,
            custom_changes: HashMap::new(),
        };
        let bytes = delta.estimated_bytes();
        assert_eq!(bytes, 12 + 12 + 16); // base + pos + rot
    }
}
