use glam::Vec3;

/// A pose in the motion matching database.
#[derive(Clone, Debug)]
pub struct MotionPose {
    pub clip_index: usize,
    pub frame: usize,
    pub root_velocity: Vec3,
    pub root_facing: Vec3,
    /// Future positions at 0.2, 0.4, 0.6, 0.8, 1.0 seconds.
    pub trajectory: Vec<Vec3>,
    /// Key joint positions (feet, hands).
    pub joint_positions: Vec<Vec3>,
    pub cost: f32,
}

/// Motion matching database -- pre-computed from animation clips.
pub struct MotionDatabase {
    pub poses: Vec<MotionPose>,
}

impl MotionDatabase {
    pub fn new() -> Self {
        Self { poses: Vec::new() }
    }

    /// Add a pose to the database.
    pub fn add_pose(&mut self, pose: MotionPose) {
        self.poses.push(pose);
    }

    /// Find the best matching pose for the current state.
    pub fn find_best_match(
        &self,
        current_velocity: Vec3,
        current_facing: Vec3,
        desired_trajectory: &[Vec3],
    ) -> Option<&MotionPose> {
        self.poses.iter().min_by(|a, b| {
            let cost_a =
                compute_match_cost(a, current_velocity, current_facing, desired_trajectory);
            let cost_b =
                compute_match_cost(b, current_velocity, current_facing, desired_trajectory);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn pose_count(&self) -> usize {
        self.poses.len()
    }
}

impl Default for MotionDatabase {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_match_cost(
    pose: &MotionPose,
    velocity: Vec3,
    facing: Vec3,
    trajectory: &[Vec3],
) -> f32 {
    let vel_cost = pose.root_velocity.distance(velocity) * 1.0;
    let facing_cost = (1.0 - pose.root_facing.dot(facing).max(0.0)) * 2.0;
    let traj_cost: f32 = pose
        .trajectory
        .iter()
        .zip(trajectory)
        .map(|(a, b)| a.distance(*b))
        .sum::<f32>()
        * 0.5;
    vel_cost + facing_cost + traj_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pose(velocity: Vec3, facing: Vec3, trajectory: Vec<Vec3>) -> MotionPose {
        MotionPose {
            clip_index: 0,
            frame: 0,
            root_velocity: velocity,
            root_facing: facing,
            trajectory,
            joint_positions: vec![],
            cost: 0.0,
        }
    }

    #[test]
    fn best_match_returns_closest_velocity() {
        let mut db = MotionDatabase::new();
        db.add_pose(make_pose(Vec3::X * 5.0, Vec3::Z, vec![]));
        db.add_pose(make_pose(Vec3::X * 1.0, Vec3::Z, vec![]));
        db.add_pose(make_pose(Vec3::X * 10.0, Vec3::Z, vec![]));

        let best = db.find_best_match(Vec3::X * 1.1, Vec3::Z, &[]).unwrap();
        assert!(
            (best.root_velocity - Vec3::X * 1.0).length() < 0.01,
            "Should pick pose with velocity closest to query"
        );
    }

    #[test]
    fn facing_affects_selection() {
        let mut db = MotionDatabase::new();
        // Same velocity, different facing
        db.add_pose(make_pose(Vec3::X, Vec3::Z, vec![])); // facing Z
        db.add_pose(make_pose(Vec3::X, -Vec3::Z, vec![])); // facing -Z

        let best = db.find_best_match(Vec3::X, Vec3::Z, &[]).unwrap();
        assert!(
            best.root_facing.dot(Vec3::Z) > 0.9,
            "Should prefer pose facing the same direction"
        );
    }

    #[test]
    fn trajectory_affects_selection() {
        let mut db = MotionDatabase::new();
        let traj_a = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let traj_b = vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0)];
        db.add_pose(make_pose(Vec3::ZERO, Vec3::Z, traj_a));
        db.add_pose(make_pose(Vec3::ZERO, Vec3::Z, traj_b));

        let desired = vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0)];
        let best = db.find_best_match(Vec3::ZERO, Vec3::Z, &desired).unwrap();
        // Should match the trajectory going in Z direction
        assert!(
            best.trajectory[0].z > 0.5,
            "Should pick pose with matching trajectory"
        );
    }

    #[test]
    fn empty_database_returns_none() {
        let db = MotionDatabase::new();
        assert!(db.find_best_match(Vec3::ZERO, Vec3::Z, &[]).is_none());
    }
}
