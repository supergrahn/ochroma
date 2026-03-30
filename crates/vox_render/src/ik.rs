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

            // Apply pole vector constraint after backward pass
            if let Some(pole) = self.pole_target {
                let n = self.joint_positions.len();
                if n >= 3 {
                    let mid = n / 2;
                    let last = n - 1;
                    let root_to_tip =
                        (self.joint_positions[last] - self.joint_positions[0]).normalize();
                    let root_to_mid = self.joint_positions[mid] - self.joint_positions[0];
                    let mid_on_axis =
                        self.joint_positions[0] + root_to_tip * root_to_mid.dot(root_to_tip);
                    let current_vec = self.joint_positions[mid] - mid_on_axis;
                    let desired_vec = pole - mid_on_axis;
                    if current_vec.length() > 1e-6 && desired_vec.length() > 1e-6 {
                        let current_dir = current_vec.normalize();
                        let desired_dir = desired_vec.normalize();
                        let q = Quat::from_rotation_arc(current_dir, desired_dir);
                        self.joint_positions[mid] = mid_on_axis + q * current_vec;
                    }
                }
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

/// Higher-level IK chain spec (for skeleton-index-based usage).
pub struct IkChain {
    pub joints: Vec<usize>,
    pub target: Vec3,
    pub pole_vector: Option<Vec3>,
    pub max_reach: f32,
    pub iterations: u8,
}

impl IkChain {
    pub fn new(joints: Vec<usize>, max_reach: f32) -> Self {
        Self {
            joints,
            target: Vec3::ZERO,
            pole_vector: None,
            max_reach,
            iterations: 8,
        }
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

/// Two-bone IK returning world-space positions of mid and tip joints.
///
/// Given the root position, current mid position (used for pole fallback),
/// target tip position, and the two bone lengths, solve analytically in the
/// plane defined by (root, tip_target, pole).
///
/// Returns `(mid_pos, tip_pos)`.
pub fn two_bone_ik(
    root: Vec3,
    _mid: Vec3,
    tip_target: Vec3,
    l0: f32,
    l1: f32,
    pole: Vec3,
) -> (Vec3, Vec3) {
    let d = (tip_target - root).length().min(l0 + l1 - 1e-4);
    let cos_angle_mid =
        ((l0 * l0 + l1 * l1 - d * d) / (2.0 * l0 * l1)).clamp(-1.0, 1.0);
    let _angle_mid = cos_angle_mid.acos();

    let root_to_target = (tip_target - root).normalize();
    let pole_dir = pole - root;

    // Build orthonormal basis in the IK plane
    let axis_raw = root_to_target.cross(pole_dir);
    let axis = if axis_raw.length() > 1e-6 {
        axis_raw.normalize()
    } else {
        // Fallback: pick an arbitrary perpendicular axis
        let up = if root_to_target.dot(Vec3::Y).abs() < 0.99 {
            Vec3::Y
        } else {
            Vec3::Z
        };
        root_to_target.cross(up).normalize()
    };
    let perp = axis.cross(root_to_target).normalize();

    // Mid joint angle at root via law of cosines
    let cos_root = ((d * d + l0 * l0 - l1 * l1) / (2.0 * d * l0)).clamp(-1.0, 1.0);
    let angle_root = cos_root.acos();

    let mid = root
        + root_to_target * l0 * angle_root.cos()
        + perp * l0 * angle_root.sin();
    let tip = root + root_to_target * d;

    (mid, tip)
}

/// SDF-based procedural foot placement.
pub struct FootPlacement {
    pub left_foot_joint: usize,
    pub right_foot_joint: usize,
    pub hip_joint: usize,
    pub hip_compensation: f32,
    pub raycast_offset: f32,
    pub max_steps: u32,
    pub step_size: f32,
}

impl FootPlacement {
    pub fn new() -> Self {
        Self {
            left_foot_joint: 4,
            right_foot_joint: 8,
            hip_joint: 0,
            hip_compensation: 0.5,
            raycast_offset: 0.2,
            max_steps: 64,
            step_size: 0.02,
        }
    }

    /// Perform SDF-based foot placement for both feet.
    ///
    /// `joint_positions` — mutable slice of world-space joint positions.
    /// `sdf` — closure returning SDF value at a world position.
    /// `leg_bone_lengths` — `(upper_leg, lower_leg)` lengths for two-bone IK.
    pub fn update<F: Fn(Vec3) -> f32>(
        &mut self,
        joint_positions: &mut [Vec3],
        sdf: F,
        leg_bone_lengths: (f32, f32),
    ) {
        let (l0, l1) = leg_bone_lengths;

        for foot_idx in [self.left_foot_joint, self.right_foot_joint] {
            if foot_idx >= joint_positions.len() {
                continue;
            }

            let ankle_pos = joint_positions[foot_idx];
            let ray_start = ankle_pos + Vec3::Y * self.raycast_offset;

            if let Some(hit_point) = self.sdf_march(ray_start, Vec3::NEG_Y, &sdf) {
                let foot_offset = hit_point.y - ankle_pos.y;

                if foot_idx >= 2 && (foot_idx - 2) < joint_positions.len() {
                    let mid_joint = foot_idx - 1;
                    let root_joint = foot_idx - 2;

                    if mid_joint < joint_positions.len() && root_joint < joint_positions.len() {
                        let root_pos = joint_positions[root_joint];
                        let pole = joint_positions[mid_joint] + Vec3::Z * 0.5;

                        let (mid_pos, _) = two_bone_ik(
                            root_pos,
                            joint_positions[mid_joint],
                            hit_point,
                            l0,
                            l1,
                            pole,
                        );
                        joint_positions[mid_joint] = mid_pos;
                        joint_positions[foot_idx] = hit_point;
                    }
                }

                // Hip compensation
                if self.hip_joint < joint_positions.len() {
                    joint_positions[self.hip_joint].y -= foot_offset * self.hip_compensation;
                }
            }
        }
    }

    fn sdf_march<F: Fn(Vec3) -> f32>(
        &self,
        start: Vec3,
        dir: Vec3,
        sdf: &F,
    ) -> Option<Vec3> {
        let mut pos = start;
        for _ in 0..self.max_steps {
            let d = sdf(pos);
            if d < 0.005 {
                return Some(pos);
            }
            pos += dir * d.abs().max(self.step_size);
            if (pos - start).length() > 10.0 {
                break;
            }
        }
        None
    }
}

impl Default for FootPlacement {
    fn default() -> Self {
        Self::new()
    }
}

/// Procedural hand / arm IK solver.
pub struct HandIk {
    pub wrist_joint: usize,
    pub elbow_joint: usize,
    pub shoulder_joint: usize,
    pub upper_arm_length: f32,
    pub forearm_length: f32,
}

pub enum Arm {
    Left,
    Right,
}

impl HandIk {
    /// Move the wrist and elbow joints to grab `target_pos`.
    pub fn grab(
        &self,
        joint_positions: &mut [Vec3],
        target_pos: Vec3,
        _grab_normal: Vec3,
        _arm: Arm,
    ) {
        if self.shoulder_joint >= joint_positions.len()
            || self.elbow_joint >= joint_positions.len()
            || self.wrist_joint >= joint_positions.len()
        {
            return;
        }

        let shoulder = joint_positions[self.shoulder_joint];
        let elbow = joint_positions[self.elbow_joint];
        let pole = elbow + Vec3::new(0.0, 0.0, -1.0);

        let (mid_pos, tip_pos) = two_bone_ik(
            shoulder,
            elbow,
            target_pos,
            self.upper_arm_length,
            self.forearm_length,
            pole,
        );
        joint_positions[self.elbow_joint] = mid_pos;
        joint_positions[self.wrist_joint] = tip_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── existing tests ────────────────────────────────────────────────────────

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
        assert!((chain.joint_positions[0] - Vec3::ZERO).length() < 0.001);
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

        assert!(
            (root_rot.length() - 1.0).abs() < 0.001,
            "Root rotation should be unit quat"
        );
        assert!(
            (mid_rot.length() - 1.0).abs() < 0.001,
            "Mid rotation should be unit quat"
        );
    }

    // ── new tests ─────────────────────────────────────────────────────────────

    #[test]
    fn two_bone_ik_returns_positions() {
        let root = Vec3::ZERO;
        let mid_cur = Vec3::new(0.0, 1.0, 0.0);
        let target = Vec3::new(2.0, 0.0, 0.0);
        let pole = Vec3::new(0.0, 1.0, 0.0);
        let l0 = 1.0_f32;
        let l1 = 1.0_f32;

        let (mid_pos, tip_pos) = two_bone_ik(root, mid_cur, target, l0, l1, pole);

        // tip should be near (2, 0, 0) — the chain is fully extended
        assert!(
            tip_pos.distance(target) < 0.01,
            "tip should be near target, got {tip_pos}"
        );

        // mid should lie between root and tip (within the reach of l0)
        let root_to_mid = (mid_pos - root).length();
        assert!(
            (root_to_mid - l0).abs() < 0.01,
            "root-to-mid length should equal l0, got {root_to_mid}"
        );
    }

    #[test]
    fn foot_placement_flat_ground() {
        // Flat ground at y = 0: SDF = pos.y
        let sdf = |pos: Vec3| pos.y;

        // Simple 3-joint leg: root(idx 2), mid(idx 3), ankle(idx 4)
        let mut joints = vec![
            Vec3::new(0.0, 1.0, 0.0), // hip (idx 0) — also hip_joint
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0), // root of leg
            Vec3::new(0.0, 0.5, 0.0), // mid
            Vec3::new(0.0, 0.5, 0.0), // left ankle at y=0.5
        ];

        let mut fp = FootPlacement::new(); // left_foot_joint = 4
        fp.hip_joint = 0;
        fp.right_foot_joint = 99; // disable right foot for this test

        fp.update(&mut joints, sdf, (0.5, 0.5));

        // Left ankle (index 4) should now be near y = 0
        assert!(
            joints[4].y.abs() < 0.1,
            "Left foot should be near ground (y≈0), got y={}",
            joints[4].y
        );
    }

    #[test]
    fn hand_ik_grab_moves_wrist() {
        let shoulder_pos = Vec3::new(0.0, 2.0, 0.0);
        let elbow_pos = Vec3::new(0.0, 1.0, 0.0);
        let wrist_pos = Vec3::new(0.0, 0.0, 0.0);

        let mut joints = vec![shoulder_pos, elbow_pos, wrist_pos];

        let ik = HandIk {
            shoulder_joint: 0,
            elbow_joint: 1,
            wrist_joint: 2,
            upper_arm_length: 1.0,
            forearm_length: 1.0,
        };

        let target = Vec3::new(1.5, 1.0, 0.0);
        ik.grab(&mut joints, target, Vec3::Y, Arm::Right);

        // Wrist should have moved toward the target
        let dist_before = wrist_pos.distance(target);
        let dist_after = joints[2].distance(target);
        assert!(
            dist_after < dist_before,
            "Wrist should move closer to target: before={dist_before:.3}, after={dist_after:.3}"
        );
    }

    #[test]
    fn foot_placement_default_joints() {
        let fp = FootPlacement::new();
        assert_eq!(fp.left_foot_joint, 4);
        assert_eq!(fp.right_foot_joint, 8);
        assert_eq!(fp.hip_joint, 0);
    }
}
