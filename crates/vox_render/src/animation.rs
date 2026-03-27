use glam::{Mat4, Quat, Vec3};

/// A single bone in a skeleton hierarchy.
#[derive(Debug, Clone)]
pub struct Bone {
    pub id: u8,
    pub parent_id: Option<u8>,
    pub local_transform: Mat4,
    pub name: String,
}

/// A skeleton composed of bones in a hierarchy.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self { bones: Vec::new() }
    }

    pub fn add_bone(&mut self, bone: Bone) {
        self.bones.push(bone);
    }

    /// Find a bone by its id.
    fn find_bone_index(&self, id: u8) -> Option<usize> {
        self.bones.iter().position(|b| b.id == id)
    }

    /// Compute the world transform for each bone by walking the hierarchy.
    /// Returns a Vec indexed by bone position in `self.bones`.
    pub fn compute_world_transforms(&self) -> Vec<Mat4> {
        let mut world = vec![Mat4::IDENTITY; self.bones.len()];
        for (i, bone) in self.bones.iter().enumerate() {
            let parent_world = match bone.parent_id {
                Some(pid) => {
                    if let Some(pi) = self.find_bone_index(pid) {
                        world[pi]
                    } else {
                        Mat4::IDENTITY
                    }
                }
                None => Mat4::IDENTITY,
            };
            world[i] = parent_world * bone.local_transform;
        }
        world
    }
}

/// A keyframe storing a transform at a specific time.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub time: f32,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

/// An animation clip with keyframes per bone.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    /// Keyframes indexed by bone_id.
    pub bone_keyframes: std::collections::HashMap<u8, Vec<Keyframe>>,
}

impl AnimationClip {
    pub fn new(name: &str, duration: f32) -> Self {
        Self {
            name: name.to_string(),
            duration,
            bone_keyframes: std::collections::HashMap::new(),
        }
    }

    pub fn add_keyframe(&mut self, bone_id: u8, keyframe: Keyframe) {
        self.bone_keyframes.entry(bone_id).or_default().push(keyframe);
    }

    /// Sample the transform for a bone at a given time using linear interpolation.
    pub fn sample(&self, bone_id: u8, time: f32) -> Option<Mat4> {
        let keyframes = self.bone_keyframes.get(&bone_id)?;
        if keyframes.is_empty() {
            return None;
        }
        if keyframes.len() == 1 {
            let kf = &keyframes[0];
            return Some(Mat4::from_scale_rotation_translation(kf.scale, kf.rotation, kf.position));
        }

        // Find the two keyframes surrounding `time`.
        let mut prev = &keyframes[0];
        let mut next = &keyframes[keyframes.len() - 1];
        for i in 0..keyframes.len() - 1 {
            if keyframes[i].time <= time && keyframes[i + 1].time >= time {
                prev = &keyframes[i];
                next = &keyframes[i + 1];
                break;
            }
        }

        let span = next.time - prev.time;
        let t = if span > 0.0 {
            ((time - prev.time) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let pos = prev.position.lerp(next.position, t);
        let rot = prev.rotation.slerp(next.rotation, t);
        let scl = prev.scale.lerp(next.scale, t);

        Some(Mat4::from_scale_rotation_translation(scl, rot, pos))
    }
}

/// Current animation playback state.
#[derive(Debug, Clone)]
pub struct AnimationState {
    pub clip_index: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl AnimationState {
    pub fn new(clip_index: usize) -> Self {
        Self {
            clip_index,
            time: 0.0,
            speed: 1.0,
            looping: true,
        }
    }
}

/// Plays animation clips on a skeleton, supporting blending between two clips.
pub struct AnimationPlayer {
    pub clips: Vec<AnimationClip>,
    pub primary: AnimationState,
    pub blend_target: Option<AnimationState>,
    pub blend_factor: f32,
    pub blend_duration: f32,
    blend_elapsed: f32,
}

impl AnimationPlayer {
    pub fn new(clips: Vec<AnimationClip>) -> Self {
        Self {
            clips,
            primary: AnimationState::new(0),
            blend_target: None,
            blend_factor: 0.0,
            blend_duration: 0.0,
            blend_elapsed: 0.0,
        }
    }

    /// Start blending from the current clip to another over `duration` seconds.
    pub fn blend_to(&mut self, clip_index: usize, duration: f32) {
        self.blend_target = Some(AnimationState::new(clip_index));
        self.blend_duration = duration;
        self.blend_elapsed = 0.0;
        self.blend_factor = 0.0;
    }

    /// Advance the animation by `dt` seconds.
    pub fn update(&mut self, dt: f32) {
        // Advance primary.
        self.advance_state(&mut self.primary.clone(), dt);
        let mut primary = self.primary.clone();
        Self::advance_state_inner(&mut primary, dt, &self.clips);
        self.primary = primary;

        // Advance blend target if active.
        if let Some(ref mut target) = self.blend_target {
            Self::advance_state_inner(target, dt, &self.clips);
            self.blend_elapsed += dt;
            self.blend_factor = (self.blend_elapsed / self.blend_duration).clamp(0.0, 1.0);
            if self.blend_factor >= 1.0 {
                self.primary = self.blend_target.take().unwrap();
                self.blend_factor = 0.0;
            }
        }
    }

    fn advance_state_inner(state: &mut AnimationState, dt: f32, clips: &[AnimationClip]) {
        state.time += dt * state.speed;
        if let Some(clip) = clips.get(state.clip_index) {
            if state.looping && state.time > clip.duration {
                state.time %= clip.duration;
            } else if !state.looping && state.time > clip.duration {
                state.time = clip.duration;
            }
        }
    }

    fn advance_state(&self, _state: &mut AnimationState, _dt: f32) {
        // Kept for API compatibility; actual work done in advance_state_inner.
    }

    /// Sample the skeleton pose at the current playback state.
    /// Returns the local transform override per bone_id.
    pub fn sample_pose(&self) -> std::collections::HashMap<u8, Mat4> {
        let mut pose = std::collections::HashMap::new();

        if let Some(primary_clip) = self.clips.get(self.primary.clip_index) {
            for &bone_id in primary_clip.bone_keyframes.keys() {
                if let Some(m) = primary_clip.sample(bone_id, self.primary.time) {
                    pose.insert(bone_id, m);
                }
            }
        }

        // Blend with target if active.
        if let Some(ref target_state) = self.blend_target {
            if let Some(target_clip) = self.clips.get(target_state.clip_index) {
                for &bone_id in target_clip.bone_keyframes.keys() {
                    if let Some(target_m) = target_clip.sample(bone_id, target_state.time) {
                        let blended = if let Some(&primary_m) = pose.get(&bone_id) {
                            lerp_mat4(primary_m, target_m, self.blend_factor)
                        } else {
                            target_m
                        };
                        pose.insert(bone_id, blended);
                    }
                }
            }
        }

        pose
    }
}

/// Linearly interpolate between two Mat4 transforms by decomposing into
/// translation, rotation, and scale components.
pub fn lerp_mat4(a: Mat4, b: Mat4, t: f32) -> Mat4 {
    let (a_scale, a_rot, a_trans) = a.to_scale_rotation_translation();
    let (b_scale, b_rot, b_trans) = b.to_scale_rotation_translation();
    let pos = a_trans.lerp(b_trans, t);
    let rot = a_rot.slerp(b_rot, t);
    let scl = a_scale.lerp(b_scale, t);
    Mat4::from_scale_rotation_translation(scl, rot, pos)
}

/// Binding that tells which bone a splat is attached to.
#[derive(Debug, Clone, Copy)]
pub struct BoneBinding {
    pub bone_id: u8,
}

/// Apply a skeleton's world transforms to splat positions based on bone bindings.
/// `splat_positions` are the bind-pose positions; returns transformed positions.
pub fn apply_skeleton_to_splats(
    skeleton: &Skeleton,
    splat_positions: &[[f32; 3]],
    bone_bindings: &[BoneBinding],
) -> Vec<[f32; 3]> {
    let world_transforms = skeleton.compute_world_transforms();
    let bone_id_to_index: std::collections::HashMap<u8, usize> = skeleton
        .bones
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    splat_positions
        .iter()
        .zip(bone_bindings.iter())
        .map(|(pos, binding)| {
            let transform = bone_id_to_index
                .get(&binding.bone_id)
                .and_then(|&idx| world_transforms.get(idx))
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let p = transform.transform_point3(Vec3::new(pos[0], pos[1], pos[2]));
            [p.x, p.y, p.z]
        })
        .collect()
}
