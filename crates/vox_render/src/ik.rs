use glam::{Quat, Vec3};

/// FABRIK (Forward And Backward Reaching IK) chain solver.
pub struct IKChain {
    pub joint_positions: Vec<Vec3>,
    pub joint_lengths: Vec<f32>,
    pub target: Vec3,
    pub pole_target: Option<Vec3>,
    pub iterations: u32,
    pub tolerance: f32,
}

impl IKChain {
    pub fn new(joint_positions: Vec<Vec3>) -> Self {
        let joint_lengths: Vec<f32> = joint_positions
            .windows(2)
            .map(|w| w[0].distance(w[1]))
            .collect();
        Self {
            joint_positions,
            joint_lengths,
            target: Vec3::ZERO,
            pole_target: None,
            iterations: 10,
            tolerance: 0.001,
        }
    }

    /// Solve IK using FABRIK algorithm.
    /// Returns new joint positions that reach toward the target.
    pub fn solve(&mut self) -> &[Vec3] {
        let total_length: f32 = self.joint_lengths.iter().sum();
        let root = self.joint_positions[0];
        let dist_to_target = root.distance(self.target);

        // If target is unreachable, stretch toward it
        if dist_to_target > total_length {
            for i in 0..self.joint_positions.len() - 1 {
                let dir = (self.target - self.joint_positions[i]).normalize();
                self.joint_positions[i + 1] = self.joint_positions[i] + dir * self.joint_lengths[i];
            }
            return &self.joint_positions;
        }

        // FABRIK iterations
        for _ in 0..self.iterations {
            // Forward pass: start from end effector
            *self.joint_positions.last_mut().unwrap() = self.target;
            for i in (0..self.joint_positions.len() - 1).rev() {
                let dir =
                    (self.joint_positions[i] - self.joint_positions[i + 1]).normalize();
                self.joint_positions[i] =
                    self.joint_positions[i + 1] + dir * self.joint_lengths[i];
            }

            // Backward pass: start from root (constrain root to original position)
            self.joint_positions[0] = root;
            for i in 0..self.joint_positions.len() - 1 {
                let dir =
                    (self.joint_positions[i + 1] - self.joint_positions[i]).normalize();
                self.joint_positions[i + 1] =
                    self.joint_positions[i] + dir * self.joint_lengths[i];
            }

            // Check convergence
            let end_effector = *self.joint_positions.last().unwrap();
            if end_effector.distance(self.target) < self.tolerance {
                break;
            }
        }

        &self.joint_positions
    }

    pub fn end_effector(&self) -> Vec3 {
        *self.joint_positions.last().unwrap_or(&Vec3::ZERO)
    }

    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    pub fn reached(&self) -> bool {
        self.end_effector().distance(self.target) < self.tolerance
    }
}

/// Two-bone IK (common for arms/legs). Analytical solution.
/// Returns (root_rotation, mid_rotation).
pub fn solve_two_bone_ik(
    root: Vec3,
    mid: Vec3,
    end: Vec3,
    target: Vec3,
    _pole: Vec3,
) -> (Quat, Quat) {
    let upper_len = root.distance(mid);
    let lower_len = mid.distance(end);
    let target_dist = root.distance(target).min(upper_len + lower_len - 0.001);

    // Law of cosines for the mid joint angle
    let cos_angle = ((upper_len * upper_len + lower_len * lower_len
        - target_dist * target_dist)
        / (2.0 * upper_len * lower_len))
        .clamp(-1.0, 1.0);
    let mid_angle = std::f32::consts::PI - cos_angle.acos();

    // Root rotation: point toward target
    let to_target = (target - root).normalize();
    let initial_dir = (mid - root).normalize();
    let root_rot = Quat::from_rotation_arc(initial_dir, to_target);

    // Mid rotation
    let mid_rot = Quat::from_axis_angle(Vec3::X, mid_angle);

    (root_rot, mid_rot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_reaches_reachable_target() {
        let joints = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let mut chain = IKChain::new(joints);
        chain.set_target(Vec3::new(1.5, 1.0, 0.0));
        chain.solve();
        assert!(
            chain.reached(),
            "End effector should reach target within tolerance, dist={}",
            chain.end_effector().distance(chain.target)
        );
    }

    #[test]
    fn unreachable_target_stretches_toward() {
        let joints = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let mut chain = IKChain::new(joints);
        chain.set_target(Vec3::new(100.0, 0.0, 0.0));
        chain.solve();
        // Root stays at origin
        assert!((chain.joint_positions[0] - Vec3::ZERO).length() < 0.001);
        // All joints should be stretched along X axis
        for i in 0..chain.joint_positions.len() - 1 {
            let dir = (chain.joint_positions[i + 1] - chain.joint_positions[i]).normalize();
            assert!(
                dir.dot(Vec3::X) > 0.99,
                "Joint {} should point toward target",
                i
            );
        }
    }

    #[test]
    fn convergence_within_tolerance() {
        let joints = vec![
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let mut chain = IKChain::new(joints);
        chain.tolerance = 0.01;
        chain.set_target(Vec3::new(1.0, 2.0, 0.0));
        chain.solve();
        let dist = chain.end_effector().distance(chain.target);
        assert!(
            dist < chain.tolerance,
            "Should converge within tolerance, dist={dist}"
        );
    }

    #[test]
    fn two_bone_ik_produces_valid_rotations() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 1.0, 0.0);
        let end = Vec3::new(0.0, 2.0, 0.0);
        let target = Vec3::new(1.0, 1.0, 0.0);
        let pole = Vec3::new(0.0, 0.0, 1.0);

        let (root_rot, mid_rot) = solve_two_bone_ik(root, mid, end, target, pole);

        // Rotations should be unit quaternions
        assert!(
            (root_rot.length() - 1.0).abs() < 0.001,
            "Root rotation should be unit quat"
        );
        assert!(
            (mid_rot.length() - 1.0).abs() < 0.001,
            "Mid rotation should be unit quat"
        );
    }
}
