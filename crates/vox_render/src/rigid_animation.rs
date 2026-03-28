use glam::{Quat, Vec3};
use std::collections::HashMap;

/// A rigid animation clip — keyframed transforms over time.
///
/// For rigid animation, we don't modify individual splats. We modify the ENTITY
/// transform. The splats are offset from the entity origin at load time. When
/// the entity moves/rotates, all its splats move with it — like Lego pieces.
#[derive(Debug, Clone)]
pub struct RigidClip {
    pub name: String,
    pub duration: f32,
    pub looping: bool,
    pub keyframes: Vec<RigidKeyframe>,
}

/// A single keyframe storing position, rotation, and scale at a specific time.
#[derive(Debug, Clone)]
pub struct RigidKeyframe {
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl RigidKeyframe {
    /// Interpolate between two keyframes.
    /// Position: linear lerp. Rotation: slerp. Scale: linear lerp.
    pub fn interpolate(a: &RigidKeyframe, b: &RigidKeyframe, t: f32) -> RigidKeyframe {
        let t = t.clamp(0.0, 1.0);
        RigidKeyframe {
            time: a.time + (b.time - a.time) * t,
            position: a.position.lerp(b.position, t),
            rotation: a.rotation.slerp(b.rotation, t),
            scale: a.scale.lerp(b.scale, t),
        }
    }

    /// Identity keyframe at the given time.
    pub fn identity(time: f32) -> Self {
        RigidKeyframe {
            time,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl RigidClip {
    pub fn new(name: &str, duration: f32, looping: bool) -> Self {
        Self {
            name: name.to_string(),
            duration,
            looping,
            keyframes: Vec::new(),
        }
    }

    /// Sample the clip at a given time, returning an interpolated keyframe.
    pub fn sample(&self, mut time: f32) -> RigidKeyframe {
        if self.keyframes.is_empty() {
            return RigidKeyframe::identity(time);
        }

        // Handle looping
        if self.looping && self.duration > 0.0 {
            time %= self.duration;
            if time < 0.0 {
                time += self.duration;
            }
        } else {
            time = time.clamp(0.0, self.duration);
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return self.keyframes[0].clone();
        }

        // After last keyframe
        let last = &self.keyframes[self.keyframes.len() - 1];
        if time >= last.time {
            return last.clone();
        }

        // Find surrounding keyframes and interpolate
        for i in 0..self.keyframes.len() - 1 {
            let prev = &self.keyframes[i];
            let next = &self.keyframes[i + 1];
            if time >= prev.time && time <= next.time {
                let span = next.time - prev.time;
                let t = if span > 0.0 {
                    (time - prev.time) / span
                } else {
                    0.0
                };
                return RigidKeyframe::interpolate(prev, next, t);
            }
        }

        // Fallback (shouldn't reach here)
        self.keyframes.last().unwrap().clone()
    }

    /// Continuous rotation around an axis. Generates a full-loop clip.
    /// `speed` is rotations per second.
    pub fn rotation_loop(axis: Vec3, speed: f32) -> Self {
        let duration = 1.0 / speed.abs().max(f32::EPSILON);
        let axis = axis.normalize();
        let keyframes = vec![
            RigidKeyframe {
                time: 0.0,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.5,
                position: Vec3::ZERO,
                rotation: Quat::from_axis_angle(axis, std::f32::consts::PI),
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration,
                position: Vec3::ZERO,
                rotation: Quat::from_axis_angle(axis, std::f32::consts::TAU),
                scale: Vec3::ONE,
            },
        ];
        Self {
            name: "rotation_loop".to_string(),
            duration,
            looping: true,
            keyframes,
        }
    }

    /// Up-down bobbing motion. `height` is the peak offset, `speed` is cycles per second.
    pub fn bounce(height: f32, speed: f32) -> Self {
        let duration = 1.0 / speed.abs().max(f32::EPSILON);
        let keyframes = vec![
            RigidKeyframe {
                time: 0.0,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.25,
                position: Vec3::new(0.0, height, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.5,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.75,
                position: Vec3::new(0.0, -height, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ];
        Self {
            name: "bounce".to_string(),
            duration,
            looping: true,
            keyframes,
        }
    }

    /// Back-and-forth rotation (e.g., a swinging door).
    /// `angle` is the max angle in radians, `speed` is cycles per second, `axis` is the rotation axis.
    pub fn swing(angle: f32, speed: f32, axis: Vec3) -> Self {
        let duration = 1.0 / speed.abs().max(f32::EPSILON);
        let axis = axis.normalize();
        let keyframes = vec![
            RigidKeyframe {
                time: 0.0,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.25,
                position: Vec3::ZERO,
                rotation: Quat::from_axis_angle(axis, angle),
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.5,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration * 0.75,
                position: Vec3::ZERO,
                rotation: Quat::from_axis_angle(axis, -angle),
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: duration,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ];
        Self {
            name: "swing".to_string(),
            duration,
            looping: true,
            keyframes,
        }
    }
}

/// Condition that triggers a state transition.
#[derive(Debug, Clone)]
pub enum TransitionCondition {
    /// Transition after a specific time into the current clip.
    AfterTime(f32),
    /// Transition when a named trigger event fires.
    OnTrigger(String),
    /// Transition when a named boolean parameter matches the given value.
    OnBool(String, bool),
}

/// A transition from one state to another.
#[derive(Debug, Clone)]
pub struct RigidTransition {
    pub target: String,
    pub condition: TransitionCondition,
    pub blend_time: f32,
}

/// A named state containing a clip and its outgoing transitions.
#[derive(Debug, Clone)]
pub struct RigidState {
    pub clip: RigidClip,
    pub transitions: Vec<RigidTransition>,
}

/// Internal state tracking an in-progress transition blend.
#[derive(Debug, Clone)]
struct TransitionState {
    target: String,
    elapsed: f32,
    duration: f32,
}

/// Animation state machine — manages transitions between clips.
///
/// Each tick, the machine advances the current clip, checks transition
/// conditions, and if a transition is active, blends between the old
/// and new clip outputs over the blend duration.
#[derive(Debug, Clone)]
pub struct RigidStateMachine {
    states: HashMap<String, RigidState>,
    current: String,
    time: f32,
    speed: f32,
    transitioning: Option<TransitionState>,
    /// Pending trigger events (consumed on next tick).
    pending_triggers: Vec<String>,
    /// Boolean parameters for condition evaluation.
    bool_params: HashMap<String, bool>,
}

impl RigidStateMachine {
    pub fn new(initial_state: &str) -> Self {
        Self {
            states: HashMap::new(),
            current: initial_state.to_string(),
            time: 0.0,
            speed: 1.0,
            transitioning: None,
            pending_triggers: Vec::new(),
            bool_params: HashMap::new(),
        }
    }

    pub fn add_state(&mut self, name: &str, clip: RigidClip) {
        self.states.insert(
            name.to_string(),
            RigidState {
                clip,
                transitions: Vec::new(),
            },
        );
    }

    pub fn add_transition(&mut self, from: &str, transition: RigidTransition) {
        if let Some(state) = self.states.get_mut(from) {
            state.transitions.push(transition);
        }
    }

    /// Fire a trigger event. Consumed on the next tick.
    pub fn trigger(&mut self, event: &str) {
        self.pending_triggers.push(event.to_string());
    }

    /// Set a boolean parameter used by OnBool conditions.
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.bool_params.insert(name.to_string(), value);
    }

    /// Return the name of the current state.
    pub fn current_state(&self) -> &str {
        &self.current
    }

    /// Set playback speed multiplier.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Advance the state machine by `dt` seconds and return the interpolated transform.
    pub fn tick(&mut self, dt: f32) -> RigidKeyframe {
        let dt = dt * self.speed;

        // Advance current clip time
        self.time += dt;

        // Handle looping on the current clip
        if let Some(state) = self.states.get(&self.current) {
            let clip = &state.clip;
            if clip.looping && clip.duration > 0.0 {
                self.time %= clip.duration;
                if self.time < 0.0 {
                    self.time += clip.duration;
                }
            } else if clip.duration > 0.0 {
                self.time = self.time.clamp(0.0, clip.duration);
            }
        }

        // If we're in a transition, advance it and extract values before borrowing self again
        let transition_data = self.transitioning.as_mut().map(|trans| {
            trans.elapsed += dt.abs();
            (trans.elapsed, trans.duration, trans.target.clone())
        });
        if let Some((elapsed, duration, target_name)) = transition_data {
            if elapsed >= duration {
                // Transition complete — switch to target state
                self.current = target_name;
                self.time = elapsed; // carry over elapsed time into new clip
                if let Some(state) = self.states.get(&self.current) {
                    let clip = &state.clip;
                    if clip.looping && clip.duration > 0.0 {
                        self.time %= clip.duration;
                    } else if clip.duration > 0.0 {
                        self.time = self.time.clamp(0.0, clip.duration);
                    }
                }
                self.transitioning = None;
                // Sample the new current state
                return self.sample_current();
            } else {
                // Blend between current and target
                let blend_t = elapsed / duration;
                let current_sample = self.sample_current();
                let target_sample = if let Some(target_state) = self.states.get(&target_name) {
                    target_state.clip.sample(elapsed)
                } else {
                    RigidKeyframe::identity(0.0)
                };
                self.pending_triggers.clear();
                return RigidKeyframe::interpolate(&current_sample, &target_sample, blend_t);
            }
        }

        // Check transitions on the current state
        if let Some(state) = self.states.get(&self.current).cloned() {
            for transition in &state.transitions {
                let should_fire = match &transition.condition {
                    TransitionCondition::AfterTime(t) => self.time >= *t,
                    TransitionCondition::OnTrigger(event) => {
                        self.pending_triggers.contains(event)
                    }
                    TransitionCondition::OnBool(name, value) => {
                        self.bool_params.get(name).copied().unwrap_or(false) == *value
                    }
                };

                if should_fire && self.states.contains_key(&transition.target) {
                    if transition.blend_time > 0.0 {
                        self.transitioning = Some(TransitionState {
                            target: transition.target.clone(),
                            elapsed: 0.0,
                            duration: transition.blend_time,
                        });
                    } else {
                        // Instant transition
                        self.current = transition.target.clone();
                        self.time = 0.0;
                    }
                    break;
                }
            }
        }

        self.pending_triggers.clear();
        self.sample_current()
    }

    /// Sample the current state's clip at the current time.
    fn sample_current(&self) -> RigidKeyframe {
        if let Some(state) = self.states.get(&self.current) {
            state.clip.sample(self.time)
        } else {
            RigidKeyframe::identity(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_vec3(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    #[test]
    fn test_keyframe_interpolation() {
        let a = RigidKeyframe {
            time: 0.0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        let b = RigidKeyframe {
            time: 1.0,
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::from_axis_angle(Vec3::Y, PI),
            scale: Vec3::splat(2.0),
        };
        let mid = RigidKeyframe::interpolate(&a, &b, 0.5);
        assert!(approx_vec3(mid.position, Vec3::new(5.0, 0.0, 0.0), 0.01));
        assert!(approx_vec3(mid.scale, Vec3::splat(1.5), 0.01));
    }

    #[test]
    fn test_clip_sample_basic() {
        let mut clip = RigidClip::new("test", 2.0, false);
        clip.keyframes = vec![
            RigidKeyframe {
                time: 0.0,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: 2.0,
                position: Vec3::new(4.0, 0.0, 0.0),
                rotation: Quat::from_axis_angle(Vec3::Y, PI),
                scale: Vec3::ONE,
            },
        ];

        // At halfway, position should be halfway
        let kf = clip.sample(1.0);
        assert!(approx_vec3(kf.position, Vec3::new(2.0, 0.0, 0.0), 0.01));

        // At half-duration, rotation should be approximately halfway (PI/2 around Y)
        let angle = kf.rotation.to_axis_angle();
        assert!(approx_eq(angle.1, FRAC_PI_2, 0.05));
    }

    #[test]
    fn test_clip_looping() {
        let mut clip = RigidClip::new("loop_test", 2.0, true);
        clip.keyframes = vec![
            RigidKeyframe {
                time: 0.0,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            RigidKeyframe {
                time: 2.0,
                position: Vec3::new(10.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ];

        // 1.5x duration = 3.0 seconds, wraps to 1.0 which is 50% through
        let kf = clip.sample(3.0);
        assert!(approx_vec3(kf.position, Vec3::new(5.0, 0.0, 0.0), 0.01));
    }

    #[test]
    fn test_rotation_loop_helper() {
        let clip = RigidClip::rotation_loop(Vec3::Y, 1.0);
        assert_eq!(clip.keyframes.len(), 3);
        assert!(approx_eq(clip.duration, 1.0, 0.001));
        assert!(clip.looping);

        // At halfway, should be PI rotation
        let kf = clip.sample(0.5);
        let (axis, angle) = kf.rotation.to_axis_angle();
        assert!(approx_eq(angle, PI, 0.05));
        assert!(approx_eq(axis.y.abs(), 1.0, 0.05));
    }

    #[test]
    fn test_swing_helper() {
        let angle = FRAC_PI_2;
        let clip = RigidClip::swing(angle, 1.0, Vec3::Y);
        assert_eq!(clip.keyframes.len(), 5);
        assert!(clip.looping);

        // At 0.25 duration, should be at max positive angle
        let kf = clip.sample(0.25);
        let (_, rot_angle) = kf.rotation.to_axis_angle();
        assert!(approx_eq(rot_angle, angle, 0.05));

        // At 0.5 duration, should be back to identity
        let kf = clip.sample(0.5);
        let (_, rot_angle) = kf.rotation.to_axis_angle();
        assert!(approx_eq(rot_angle, 0.0, 0.05));

        // At 0.75 duration, should be at max negative angle
        let kf = clip.sample(0.75);
        let (_, rot_angle) = kf.rotation.to_axis_angle();
        assert!(approx_eq(rot_angle, angle, 0.05));
    }
}
