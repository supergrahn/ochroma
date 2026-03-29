//! Runtime animation driver — ticks animation time and produces deformed splats each frame.
//!
//! Bridges `vox_data::gltf_animation` (skeleton + evaluate + skin) into a single
//! struct the renderer can call once per frame to get updated splat positions.

use glam::Mat4;
use vox_core::types::GaussianSplat;
use vox_data::gltf_animation::{evaluate_animation, skin_splats, GltfAnimation, GltfSkeleton};
use crate::gpu::skinning_compute::{GpuJointTransform, SkinningCompute};
use crate::gpu::blend_skinning_compute::BlendSkinningCompute;
use wgpu;

/// Drives animation on a set of splats each frame.
pub struct AnimationDriver {
    pub skeleton: GltfSkeleton,
    pub animations: Vec<GltfAnimation>,
    pub current_animation: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
    pub base_splats: Vec<GaussianSplat>,
    /// Maps each splat index to the joint that drives it.
    pub joint_bindings: Vec<usize>,
    /// Optional GPU compute skinning pass.
    pub gpu_skinning: Option<SkinningCompute>,
    /// Optional GPU blend-tree skinning pass (up to 4 poses).
    pub blend_gpu: Option<BlendSkinningCompute>,
}

/// Describes a crossfade transition between two animation states.
#[derive(Clone, Debug)]
pub struct AnimTransition {
    pub from: usize,
    pub to: usize,
    pub duration: f32,
}

/// Simple two-state crossfade state machine.
pub struct AnimStateMachine {
    pub current: usize,
    pub next: Option<usize>,
    pub blend: f32,
    pub transition_duration: f32,
}

impl AnimStateMachine {
    pub fn new(start: usize) -> Self {
        Self { current: start, next: None, blend: 0.0, transition_duration: 0.2 }
    }

    pub fn transition_to(&mut self, target: usize) {
        if self.current == target { return; }
        self.next = Some(target);
        self.blend = 0.0;
    }

    /// Advance state machine. Returns `(current_idx, next_idx, blend_weight)`.
    pub fn tick(&mut self, dt: f32) -> (usize, usize, f32) {
        if let Some(next) = self.next {
            self.blend += dt / self.transition_duration;
            if self.blend >= 1.0 {
                self.current = next;
                self.next = None;
                self.blend = 0.0;
            }
            (self.current, next, self.blend.clamp(0.0, 1.0))
        } else {
            (self.current, self.current, 0.0)
        }
    }
}

impl AnimationDriver {
    pub fn new(skeleton: GltfSkeleton, base_splats: Vec<GaussianSplat>) -> Self {
        let bindings = vec![0; base_splats.len()];
        Self {
            skeleton,
            animations: Vec::new(),
            current_animation: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
            base_splats,
            joint_bindings: bindings,
            gpu_skinning: None,
            blend_gpu: None,
        }
    }

    pub fn add_animation(&mut self, anim: GltfAnimation) {
        self.animations.push(anim);
    }

    pub fn play(&mut self, index: usize) {
        if index < self.animations.len() {
            self.current_animation = index;
            self.time = 0.0;
        }
    }

    /// Advance time by `dt` and return deformed splats for the current frame.
    pub fn tick(&mut self, dt: f32) -> Vec<GaussianSplat> {
        if self.animations.is_empty() {
            return self.base_splats.clone();
        }

        self.time += dt * self.speed;
        let anim = &self.animations[self.current_animation];

        if self.looping && anim.duration > 0.0 {
            self.time %= anim.duration;
        }

        let transforms = evaluate_animation(&self.skeleton, anim, self.time);
        let inverse_binds: Vec<Mat4> = self
            .skeleton
            .joints
            .iter()
            .map(|j| j.inverse_bind_matrix)
            .collect();

        skin_splats(
            &self.base_splats,
            &self.joint_bindings,
            &transforms,
            &inverse_binds,
        )
    }

    pub fn is_playing(&self) -> bool {
        !self.animations.is_empty()
    }

    pub fn animation_count(&self) -> usize {
        self.animations.len()
    }

    /// Advance animation time and dispatch GPU skinning compute pass.
    /// Returns `true` if the dispatch was issued, `false` if no GPU skinning is configured
    /// or no animations are loaded.
    pub fn tick_gpu(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, dt: f32) -> bool {
        let gpu = match &self.gpu_skinning {
            Some(g) => g,
            None => return false,
        };
        if self.animations.is_empty() { return false; }

        self.time += dt * self.speed;
        let anim = &self.animations[self.current_animation];
        if self.looping && anim.duration > 0.0 {
            self.time %= anim.duration;
        }

        let transforms = evaluate_animation(&self.skeleton, anim, self.time);
        let inverse_binds: Vec<glam::Mat4> = self.skeleton.joints.iter()
            .map(|j| j.inverse_bind_matrix)
            .collect();

        let joint_transforms: Vec<GpuJointTransform> = transforms.iter()
            .zip(inverse_binds.iter())
            .map(|(world_t, inv_bind)| {
                let skin = *world_t * *inv_bind;
                GpuJointTransform { skin_matrix: skin.to_cols_array_2d() }
            })
            .collect();

        gpu.update_joints(queue, &joint_transforms);
        gpu.dispatch(encoder);
        true
    }

    /// GPU-accelerated tick with blend tree support (up to 2 active poses).
    /// Returns false if `blend_gpu` is None or no animations loaded.
    pub fn tick_blend_gpu(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        state_machine: &mut AnimStateMachine,
    ) -> bool {
        let blend_gpu = match &self.blend_gpu {
            Some(g) => g,
            None => return false,
        };
        if self.animations.is_empty() { return false; }

        self.time += dt * self.speed;

        let (cur_idx, next_idx, next_weight) = state_machine.tick(dt);
        let cur_weight = 1.0 - next_weight;

        let anim_count = self.animations.len();
        let cur_anim = &self.animations[cur_idx.min(anim_count - 1)];
        let next_anim = &self.animations[next_idx.min(anim_count - 1)];

        let cur_time = if self.looping && cur_anim.duration > 0.0 {
            self.time % cur_anim.duration
        } else {
            self.time
        };

        let inv_binds: Vec<glam::Mat4> = self.skeleton.joints.iter()
            .map(|j| j.inverse_bind_matrix)
            .collect();

        let cur_transforms = evaluate_animation(&self.skeleton, cur_anim, cur_time);
        let pose0: Vec<GpuJointTransform> = cur_transforms.iter()
            .zip(inv_binds.iter())
            .map(|(t, inv)| GpuJointTransform {
                skin_matrix: (*t * *inv).to_cols_array_2d(),
            })
            .collect();

        let next_transforms = evaluate_animation(&self.skeleton, next_anim, cur_time);
        let pose1: Vec<GpuJointTransform> = next_transforms.iter()
            .zip(inv_binds.iter())
            .map(|(t, inv)| GpuJointTransform {
                skin_matrix: (*t * *inv).to_cols_array_2d(),
            })
            .collect();

        blend_gpu.update_pose(queue, 0, &pose0);
        blend_gpu.update_pose(queue, 1, &pose1);
        blend_gpu.update_weights(queue, [cur_weight, next_weight, 0.0, 0.0]);
        blend_gpu.dispatch(encoder);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use vox_data::gltf_animation::{build_synthetic_animation, build_synthetic_skeleton};

    fn make_test_splat(x: f32, y: f32, z: f32) -> GaussianSplat {
        GaussianSplat {
            position: [x, y, z],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        }
    }

    #[test]
    fn no_animations_returns_base_splats() {
        let skel = build_synthetic_skeleton(&["root"]);
        let splats = vec![make_test_splat(1.0, 2.0, 3.0)];
        let mut driver = AnimationDriver::new(skel, splats.clone());
        let result = driver.tick(1.0 / 60.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].position, splats[0].position);
    }

    #[test]
    fn animation_modifies_splats() {
        // Use 3-joint skeleton: root, arm, hand
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        // Place splat at hand bind position (0, 2, 0) bound to hand joint
        let splat = make_test_splat(0.0, 2.0, 0.0);
        let mut driver = AnimationDriver::new(skel, vec![splat]);
        driver.joint_bindings[0] = 2; // bind to hand

        // Rotate arm joint — this moves the hand
        let anim = build_synthetic_animation(
            "rotate",
            1, // rotate arm
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        );
        driver.add_animation(anim);
        driver.play(0);

        // Tick to near-end of animation (not exactly 1.0 to avoid looping wrap)
        let result = driver.tick(0.99);
        assert_eq!(result.len(), 1);
        // Hand was at (0,2,0). After arm rotates 90 deg Z, hand moves to (-1,1,0).
        let p = result[0].position;
        let original = [0.0f32, 2.0, 0.0];
        let changed = (p[0] - original[0]).abs() > 0.01
            || (p[1] - original[1]).abs() > 0.01
            || (p[2] - original[2]).abs() > 0.01;
        assert!(changed, "animation should move the splat, got {:?}", p);
    }

    #[test]
    fn looping_wraps_time() {
        let skel = build_synthetic_skeleton(&["root"]);
        let mut driver = AnimationDriver::new(skel, vec![make_test_splat(0.0, 0.0, 0.0)]);
        let anim = build_synthetic_animation(
            "spin",
            0,
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_y(std::f32::consts::PI),
        );
        driver.add_animation(anim);
        driver.looping = true;

        // Tick past the animation duration
        driver.tick(1.5);
        assert!(
            driver.time < 1.0,
            "time should wrap when looping, got {}",
            driver.time
        );
    }

    #[test]
    fn play_resets_time() {
        let skel = build_synthetic_skeleton(&["root"]);
        let mut driver = AnimationDriver::new(skel, vec![]);
        let anim = build_synthetic_animation("a", 0, 2.0, Quat::IDENTITY, Quat::IDENTITY);
        driver.add_animation(anim);
        driver.time = 1.5;
        driver.play(0);
        assert!((driver.time).abs() < 1e-6, "play should reset time to 0");
    }

    #[test]
    fn animation_count_and_is_playing() {
        let skel = build_synthetic_skeleton(&["root"]);
        let mut driver = AnimationDriver::new(skel, vec![]);
        assert!(!driver.is_playing());
        assert_eq!(driver.animation_count(), 0);

        let anim = build_synthetic_animation("a", 0, 1.0, Quat::IDENTITY, Quat::IDENTITY);
        driver.add_animation(anim);
        assert!(driver.is_playing());
        assert_eq!(driver.animation_count(), 1);
    }
}
