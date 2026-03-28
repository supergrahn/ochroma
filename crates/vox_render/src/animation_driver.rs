//! Runtime animation driver — ticks animation time and produces deformed splats each frame.
//!
//! Bridges `vox_data::gltf_animation` (skeleton + evaluate + skin) into a single
//! struct the renderer can call once per frame to get updated splat positions.

use glam::Mat4;
use vox_core::types::GaussianSplat;
use vox_data::gltf_animation::{evaluate_animation, skin_splats, GltfAnimation, GltfSkeleton};

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
