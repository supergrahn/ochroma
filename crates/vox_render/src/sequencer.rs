//! Cinematic sequencer data model.
//!
//! Multi-track timeline for cinematics with keyframe interpolation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<SequenceTrack>,
    pub playback_speed: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceTrack {
    pub name: String,
    pub track_type: TrackType,
    pub muted: bool,
    pub keyframes: Vec<SequenceKeyframe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrackType {
    CameraTransform,
    EntityTransform { entity_name: String },
    EntityVisibility { entity_name: String },
    AudioTrigger,
    Event { event_name: String },
    FloatProperty { entity_name: String, property: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceKeyframe {
    pub time: f32,
    pub value: KeyframeValue,
    pub interpolation: Interpolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyframeValue {
    Transform {
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    },
    Float(f32),
    Bool(bool),
    AudioClip(String),
    EventTrigger,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Interpolation {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Step,
}

impl Sequence {
    pub fn new(name: &str, duration: f32) -> Self {
        Self {
            name: name.to_string(),
            duration,
            tracks: Vec::new(),
            playback_speed: 1.0,
        }
    }

    pub fn add_track(&mut self, name: &str, track_type: TrackType) -> usize {
        let idx = self.tracks.len();
        self.tracks.push(SequenceTrack {
            name: name.to_string(),
            track_type,
            muted: false,
            keyframes: Vec::new(),
        });
        idx
    }

    pub fn add_keyframe(&mut self, track_index: usize, kf: SequenceKeyframe) {
        if let Some(track) = self.tracks.get_mut(track_index) {
            // Insert sorted by time
            let pos = track
                .keyframes
                .iter()
                .position(|k| k.time > kf.time)
                .unwrap_or(track.keyframes.len());
            track.keyframes.insert(pos, kf);
        }
    }

    /// Evaluate all non-muted tracks at a given time.
    pub fn evaluate(&self, time: f32) -> Vec<(String, KeyframeValue)> {
        let mut results = Vec::new();

        for track in &self.tracks {
            if track.muted {
                continue;
            }
            if track.keyframes.is_empty() {
                continue;
            }

            if let Some(value) = evaluate_track(track, time) {
                results.push((track.name.clone(), value));
            }
        }

        results
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn load_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

fn evaluate_track(track: &SequenceTrack, time: f32) -> Option<KeyframeValue> {
    let kfs = &track.keyframes;
    if kfs.is_empty() {
        return None;
    }

    // Before first keyframe
    if time <= kfs[0].time {
        return Some(kfs[0].value.clone());
    }

    // After last keyframe
    if time >= kfs.last().unwrap().time {
        return Some(kfs.last().unwrap().value.clone());
    }

    // Find the two keyframes to interpolate between
    for window in kfs.windows(2) {
        let a = &window[0];
        let b = &window[1];
        if time >= a.time && time <= b.time {
            let t = (time - a.time) / (b.time - a.time);
            let t = apply_interpolation(t, b.interpolation);
            return Some(interpolate_values(&a.value, &b.value, t));
        }
    }

    Some(kfs.last().unwrap().value.clone())
}

fn apply_interpolation(t: f32, interp: Interpolation) -> f32 {
    match interp {
        Interpolation::Linear => t,
        Interpolation::EaseIn => t * t,
        Interpolation::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
        Interpolation::EaseInOut => {
            let t2 = t * t;
            t2 / (2.0 * (t2 - t) + 1.0)
        }
        Interpolation::Step => {
            if t < 1.0 {
                0.0
            } else {
                1.0
            }
        }
    }
}

fn interpolate_values(a: &KeyframeValue, b: &KeyframeValue, t: f32) -> KeyframeValue {
    match (a, b) {
        (
            KeyframeValue::Transform {
                position: pa,
                rotation: ra,
                scale: sa,
            },
            KeyframeValue::Transform {
                position: pb,
                rotation: rb,
                scale: sb,
            },
        ) => KeyframeValue::Transform {
            position: lerp3(pa, pb, t),
            rotation: lerp4(ra, rb, t),
            scale: lerp3(sa, sb, t),
        },
        (KeyframeValue::Float(fa), KeyframeValue::Float(fb)) => {
            KeyframeValue::Float(fa + (fb - fa) * t)
        }
        // Non-interpolatable types: use A before midpoint, B at/after
        _ => {
            if t < 0.5 {
                a.clone()
            } else {
                b.clone()
            }
        }
    }
}

fn lerp3(a: &[f32; 3], b: &[f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn lerp4(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sequence() {
        let seq = Sequence::new("intro_cutscene", 10.0);
        assert_eq!(seq.name, "intro_cutscene");
        assert_eq!(seq.duration, 10.0);
        assert_eq!(seq.track_count(), 0);
        assert_eq!(seq.playback_speed, 1.0);
    }

    #[test]
    fn test_add_tracks_and_keyframes() {
        let mut seq = Sequence::new("flyover", 5.0);
        let cam = seq.add_track("main_camera", TrackType::CameraTransform);
        let vis = seq.add_track(
            "hero_vis",
            TrackType::EntityVisibility {
                entity_name: "hero".into(),
            },
        );

        seq.add_keyframe(
            cam,
            SequenceKeyframe {
                time: 0.0,
                value: KeyframeValue::Transform {
                    position: [0.0, 5.0, -10.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                interpolation: Interpolation::Linear,
            },
        );
        seq.add_keyframe(
            cam,
            SequenceKeyframe {
                time: 5.0,
                value: KeyframeValue::Transform {
                    position: [10.0, 10.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                interpolation: Interpolation::Linear,
            },
        );
        seq.add_keyframe(
            vis,
            SequenceKeyframe {
                time: 2.0,
                value: KeyframeValue::Bool(true),
                interpolation: Interpolation::Step,
            },
        );

        assert_eq!(seq.track_count(), 2);
        assert_eq!(seq.tracks[cam].keyframes.len(), 2);
        assert_eq!(seq.tracks[vis].keyframes.len(), 1);
    }

    #[test]
    fn test_evaluate_midpoint_interpolation() {
        let mut seq = Sequence::new("test", 10.0);
        let track = seq.add_track("float_track", TrackType::FloatProperty {
            entity_name: "light".into(),
            property: "intensity".into(),
        });

        seq.add_keyframe(track, SequenceKeyframe {
            time: 0.0,
            value: KeyframeValue::Float(0.0),
            interpolation: Interpolation::Linear,
        });
        seq.add_keyframe(track, SequenceKeyframe {
            time: 10.0,
            value: KeyframeValue::Float(100.0),
            interpolation: Interpolation::Linear,
        });

        let results = seq.evaluate(5.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "float_track");
        if let KeyframeValue::Float(v) = results[0].1 {
            assert!((v - 50.0).abs() < 1e-3, "Expected ~50.0, got {}", v);
        } else {
            panic!("Expected Float keyframe value");
        }
    }

    #[test]
    fn test_json_round_trip() {
        let mut seq = Sequence::new("cinematic", 30.0);
        seq.playback_speed = 0.5;
        let cam = seq.add_track("cam", TrackType::CameraTransform);
        seq.add_keyframe(cam, SequenceKeyframe {
            time: 0.0,
            value: KeyframeValue::Transform {
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            interpolation: Interpolation::EaseInOut,
        });

        let json = seq.save_json().unwrap();
        let loaded = Sequence::load_json(&json).unwrap();

        assert_eq!(loaded.name, "cinematic");
        assert_eq!(loaded.duration, 30.0);
        assert_eq!(loaded.playback_speed, 0.5);
        assert_eq!(loaded.track_count(), 1);
        assert_eq!(loaded.tracks[0].keyframes.len(), 1);
    }

    #[test]
    fn test_muted_track_skipped() {
        let mut seq = Sequence::new("test", 10.0);
        let t0 = seq.add_track("active", TrackType::FloatProperty {
            entity_name: "a".into(),
            property: "x".into(),
        });
        let t1 = seq.add_track("muted", TrackType::FloatProperty {
            entity_name: "b".into(),
            property: "y".into(),
        });

        seq.add_keyframe(t0, SequenceKeyframe {
            time: 0.0,
            value: KeyframeValue::Float(1.0),
            interpolation: Interpolation::Linear,
        });
        seq.add_keyframe(t1, SequenceKeyframe {
            time: 0.0,
            value: KeyframeValue::Float(2.0),
            interpolation: Interpolation::Linear,
        });

        seq.tracks[t1].muted = true;

        let results = seq.evaluate(0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "active");
    }
}
