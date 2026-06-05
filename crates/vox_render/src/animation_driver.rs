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

    /// Attach a GPU blend-tree skinning pass (builder style).
    /// Without this, `blend_gpu` is `None` and `tick_blend_gpu` is unreachable.
    pub fn with_blend_gpu(mut self, blend_gpu: BlendSkinningCompute) -> Self {
        self.blend_gpu = Some(blend_gpu);
        self
    }

    /// Attach a GPU blend-tree skinning pass to an existing driver.
    pub fn set_blend_gpu(&mut self, blend_gpu: BlendSkinningCompute) {
        self.blend_gpu = Some(blend_gpu);
    }

    /// Whether a GPU blend pass has been attached.
    pub fn has_blend_gpu(&self) -> bool {
        self.blend_gpu.is_some()
    }

    /// CPU blend-tree evaluation: compose two animations at their own times and
    /// linearly blend the resulting per-joint world transforms by `weight`
    /// (`weight = 0.0` → pure `anim_a`, `weight = 1.0` → pure `anim_b`).
    ///
    /// Each joint's blended transform is reconstructed from a per-component
    /// interpolation: scale and translation are lerped, rotation is slerped.
    /// This is the same math the GPU `blend_skinning` pass performs, exposed for
    /// headless callers and tests.
    pub fn blend_poses(
        &self,
        anim_a: usize,
        anim_b: usize,
        time_a: f32,
        time_b: f32,
        weight: f32,
    ) -> Vec<Mat4> {
        let count = self.animations.len();
        if count == 0 {
            return self
                .skeleton
                .joints
                .iter()
                .map(|j| j.local_transform)
                .collect();
        }
        let a = &self.animations[anim_a.min(count - 1)];
        let b = &self.animations[anim_b.min(count - 1)];
        let w = weight.clamp(0.0, 1.0);

        let pose_a = evaluate_animation(&self.skeleton, a, time_a);
        let pose_b = evaluate_animation(&self.skeleton, b, time_b);

        pose_a
            .iter()
            .zip(pose_b.iter())
            .map(|(ma, mb)| {
                let (sa, ra, ta) = ma.to_scale_rotation_translation();
                let (sb, rb, tb) = mb.to_scale_rotation_translation();
                let s = sa.lerp(sb, w);
                let r = ra.slerp(rb, w);
                let t = ta.lerp(tb, w);
                Mat4::from_scale_rotation_translation(s, r, t)
            })
            .collect()
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
        GaussianSplat::volume([x, y, z], [0.1, 0.1, 0.1], Quat::IDENTITY, 255, [0; 16])
    }

    #[test]
    fn no_animations_returns_base_splats() {
        let skel = build_synthetic_skeleton(&["root"]);
        let splats = vec![make_test_splat(1.0, 2.0, 3.0)];
        let mut driver = AnimationDriver::new(skel, splats.clone());
        let result = driver.tick(1.0 / 60.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].position(), splats[0].position());
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
        let p = result[0].position();
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
    fn blend_tree_interpolates() {
        // Two clips that drive the SAME root joint to two different rotations.
        // Clip A leaves the root at 0° about Z; Clip B rotates it +90° about Z.
        // At blend weight 0.5 the root's blended transform must be a real slerp
        // midpoint (+45°) that differs from BOTH endpoints, not equal either.
        let skel = build_synthetic_skeleton(&["root", "tip"]);
        let mut driver = AnimationDriver::new(skel, vec![]);

        let clip_a = build_synthetic_animation(
            "rest",
            0,
            1.0,
            Quat::IDENTITY,
            Quat::IDENTITY,
        );
        let clip_b = build_synthetic_animation(
            "rot90",
            0,
            1.0,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        );
        driver.add_animation(clip_a);
        driver.add_animation(clip_b);

        // Pure endpoints.
        let pose_a = driver.blend_poses(0, 1, 0.0, 0.0, 0.0);
        let pose_b = driver.blend_poses(0, 1, 0.0, 0.0, 1.0);
        // Blended midpoint.
        let mid = driver.blend_poses(0, 1, 0.0, 0.0, 0.5);

        // Inspect the root joint's rotation about Z (world transform at index 0).
        let z_angle = |m: &Mat4| -> f32 {
            let (_s, r, _t) = m.to_scale_rotation_translation();
            // Signed angle of (R * X) in the XY plane.
            let v = r * glam::Vec3::X;
            v.y.atan2(v.x)
        };
        let ang_a = z_angle(&pose_a[0]);
        let ang_b = z_angle(&pose_b[0]);
        let ang_mid = z_angle(&mid[0]);

        // Endpoints are 0° and +90°.
        assert!(ang_a.abs() < 1e-3, "ang_a should be ~0, got {ang_a}");
        assert!((ang_b - std::f32::consts::FRAC_PI_2).abs() < 1e-3, "ang_b should be ~90deg, got {ang_b}");
        // Slerp midpoint must be ~+45° — strictly between, differing from BOTH ends.
        let quarter = std::f32::consts::FRAC_PI_4;
        assert!((ang_mid - quarter).abs() < 1e-3, "midpoint angle should be ~45deg, got {ang_mid}");
        assert!((ang_mid - ang_a).abs() > 1e-2, "mid must differ from endpoint A");
        assert!((ang_mid - ang_b).abs() > 1e-2, "mid must differ from endpoint B");

        // And the translation of the tip joint (rotated by root) must also be a
        // real blend: tip lies at root-relative (0,1,0). At 0° its world is
        // (0,1); at 90° it swings to (-1,0). The +45° midpoint must land
        // strictly between on BOTH x and y.
        let tip_xy = |m: &Mat4| -> (f32, f32) {
            let t = m.to_scale_rotation_translation().2;
            (t.x, t.y)
        };
        let (ax, ay) = tip_xy(&pose_a[1]);
        let (bx, by) = tip_xy(&pose_b[1]);
        let (mx, my) = tip_xy(&mid[1]);
        // Endpoints: A=(0,1), B=(-1,0).
        assert!((ax - 0.0).abs() < 1e-3 && (ay - 1.0).abs() < 1e-3, "tip A=({ax},{ay})");
        assert!((bx + 1.0).abs() < 1e-3 && by.abs() < 1e-3, "tip B=({bx},{by})");
        // Midpoint (-sin45, cos45) ≈ (-0.707, 0.707) — strictly between both.
        assert!(mx < ax - 1e-2 && mx > bx + 1e-2, "mid x must lie strictly between: a={ax} m={mx} b={bx}");
        assert!(my < ay - 1e-2 && my > by + 1e-2, "mid y must lie strictly between: a={ay} m={my} b={by}");
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
