use glam::Vec3;

// ---------------------------------------------------------------------------
// Existing types (preserved for backwards compatibility)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// New production types
// ---------------------------------------------------------------------------

/// Full pose record stored in the motion matching database.
#[derive(Clone, Debug)]
pub struct DatabasePose {
    pub joint_positions: Vec<Vec3>,
    pub joint_velocities: Vec<Vec3>,
    /// Predicted root positions at +0.1s, +0.3s, +0.5s, +1.0s.
    pub trajectory: [Vec3; 4],
    /// Foot contact flags: [left, right].
    pub foot_contacts: [bool; 2],
    /// Locomotion phase in [0, 2π].
    pub phase: f32,
    pub time_in_clip: f32,
    pub clip_id: u32,
}

// ---------------------------------------------------------------------------
// Helper free functions
// ---------------------------------------------------------------------------

/// Euclidean distance between two 9-dimensional feature vectors.
pub fn euclidean_dist_9d(a: &[f32; 9], b: &[f32; 9]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..9 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// Minimum angular distance on a circle [0, 2π].
pub fn min_angular_distance(a: f32, b: f32) -> f32 {
    use std::f32::consts::TAU;
    let diff = (b - a).rem_euclid(TAU);
    if diff > std::f32::consts::PI {
        TAU - diff
    } else {
        diff
    }
}

// ---------------------------------------------------------------------------
// PoseDatabase
// ---------------------------------------------------------------------------

/// Motion matching pose database with 9D feature vectors.
pub struct PoseDatabase {
    pub poses: Vec<DatabasePose>,
    /// Always 9.
    pub feature_dim: usize,
}

/// Default joint indices for feet.
const LEFT_FOOT_JOINT: usize = 4;
const RIGHT_FOOT_JOINT: usize = 8;

impl PoseDatabase {
    /// Extract the 9D feature vector from a pose.
    ///
    /// Feature layout:
    ///   [vel_x, vel_z, dir_x, dir_z, foot_l.x, foot_l.z, foot_r.x, foot_r.z, hip_vel_y]
    pub fn build_feature(pose: &DatabasePose) -> [f32; 9] {
        let root_vel = pose
            .joint_velocities
            .first()
            .copied()
            .unwrap_or(Vec3::ZERO);
        let traj0 = pose.trajectory[0];

        let foot_l = if pose.joint_positions.len() > LEFT_FOOT_JOINT {
            pose.joint_positions[LEFT_FOOT_JOINT]
        } else {
            Vec3::ZERO
        };
        let foot_r = if pose.joint_positions.len() > RIGHT_FOOT_JOINT {
            pose.joint_positions[RIGHT_FOOT_JOINT]
        } else {
            Vec3::ZERO
        };

        [
            root_vel.x,
            root_vel.z,
            traj0.x,
            traj0.z,
            foot_l.x,
            foot_l.z,
            foot_r.x,
            foot_r.z,
            root_vel.y,
        ]
    }

    /// Brute-force nearest-neighbour search with phase-matching penalty.
    /// Returns the index of the best-matching pose, or `None` if the database
    /// is empty.
    pub fn nearest(&self, query: &[f32; 9]) -> Option<usize> {
        if self.poses.is_empty() {
            return None;
        }
        // We need the query phase to compute the angular penalty.  Since the
        // caller only supplies a feature vector (not a full pose), we use the
        // phase of the *current best* candidate in the first pass.  A cleaner
        // API would pass phase explicitly, but the spec uses [f32; 9], so we
        // approximate by ignoring the angular penalty in the very first pass
        // and then do a single-pass search using each candidate's own phase as
        // the "query phase" — this is the standard brute-force approach when
        // no separate phase channel is provided in the query.
        //
        // Concretely: score = d_feature + 0.3 * min_angular_dist(query_phase, candidate.phase)
        // where query_phase is estimated from the best feature-only match first.

        // Step 1 – find query_phase via a feature-only nearest neighbour.
        let (seed_idx, _) = self
            .poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let feat = Self::build_feature(p);
                (i, euclidean_dist_9d(query, &feat))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let query_phase = self.poses[seed_idx].phase;

        // Step 2 – full scored search including phase penalty.
        let (best_idx, _) = self
            .poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let feat = Self::build_feature(p);
                let d_feature = euclidean_dist_9d(query, &feat);
                let d_phase = min_angular_distance(query_phase, p.phase);
                let score = d_feature + 0.3 * d_phase;
                (i, score)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        Some(best_idx)
    }

    /// Like `nearest` but applies a `continuation_bonus` subtracted from the
    /// score of `current_idx`, biasing the search toward staying on the current
    /// pose.  Always returns a valid index (panics only on empty database).
    pub fn nearest_continuing(
        &self,
        current_idx: usize,
        query: &[f32; 9],
        continuation_bonus: f32,
    ) -> usize {
        assert!(
            !self.poses.is_empty(),
            "PoseDatabase::nearest_continuing called on empty database"
        );

        // Estimate query phase from current pose if index is valid.
        let query_phase = if current_idx < self.poses.len() {
            self.poses[current_idx].phase
        } else {
            0.0
        };

        let (best_idx, _) = self
            .poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let feat = Self::build_feature(p);
                let d_feature = euclidean_dist_9d(query, &feat);
                let d_phase = min_angular_distance(query_phase, p.phase);
                let mut score = d_feature + 0.3 * d_phase;
                if i == current_idx {
                    score -= continuation_bonus;
                }
                (i, score)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        best_idx
    }
}

// ---------------------------------------------------------------------------
// PoseDatabaseBuilder
// ---------------------------------------------------------------------------

pub struct PoseDatabaseBuilder {
    db: PoseDatabase,
    /// Foot height (world Y) below which contact is considered true.
    pub contact_threshold: f32,
}

impl PoseDatabaseBuilder {
    pub fn new() -> Self {
        Self {
            db: PoseDatabase {
                poses: Vec::new(),
                feature_dim: 9,
            },
            contact_threshold: 0.05,
        }
    }

    /// Ingest a single animation clip sampled at 10 Hz.
    ///
    /// `eval(t)` returns the world-space joint positions at time `t`.
    /// Each sample becomes one `DatabasePose` in the database.
    pub fn ingest<F>(&mut self, clip_id: u32, duration: f32, num_joints: usize, eval: F)
    where
        F: Fn(f32) -> Vec<Vec3>,
    {
        use std::f32::consts::PI;

        const STEP: f32 = 0.1;
        const FD_DT: f32 = 0.01;

        // Number of samples: floor(duration / STEP) + 1  (inclusive of t=0)
        let num_samples = (duration / STEP).floor() as usize + 1;

        for sample in 0..num_samples {
            let t = sample as f32 * STEP;

            let positions = eval(t);

            // Finite-difference velocities — clamp t bounds to avoid negative time.
            let t_fwd = (t + FD_DT).min(duration);
            let t_bwd = (t - FD_DT).max(0.0);
            let pos_fwd = eval(t_fwd);
            let pos_bwd = eval(t_bwd);
            let inv_2dt = 1.0 / (t_fwd - t_bwd).max(f32::EPSILON);

            let joint_velocities: Vec<Vec3> = (0..num_joints)
                .map(|j| {
                    let pf = if j < pos_fwd.len() {
                        pos_fwd[j]
                    } else {
                        Vec3::ZERO
                    };
                    let pb = if j < pos_bwd.len() {
                        pos_bwd[j]
                    } else {
                        Vec3::ZERO
                    };
                    (pf - pb) * inv_2dt
                })
                .collect();

            // Trajectory: root joint at t+0.1, t+0.3, t+0.5, t+1.0.
            let traj_offsets = [0.1f32, 0.3, 0.5, 1.0];
            let trajectory: [Vec3; 4] = std::array::from_fn(|i| {
                let tp = (t + traj_offsets[i]).min(duration);
                let pts = eval(tp);
                if pts.is_empty() {
                    Vec3::ZERO
                } else {
                    pts[0]
                }
            });

            // Foot contacts.
            let left_contact = if positions.len() > LEFT_FOOT_JOINT {
                positions[LEFT_FOOT_JOINT].y < self.contact_threshold
            } else {
                false
            };
            let right_contact = if positions.len() > RIGHT_FOOT_JOINT {
                positions[RIGHT_FOOT_JOINT].y < self.contact_threshold
            } else {
                false
            };
            let foot_contacts = [left_contact, right_contact];

            // Phase from contact pattern.
            let phase = match (left_contact, right_contact) {
                (true, false) => 0.0,
                (false, true) => PI,
                (true, true) => PI / 2.0,
                (false, false) => 3.0 * PI / 2.0,
            };

            // Pad / truncate joint_positions to num_joints.
            let joint_positions: Vec<Vec3> = (0..num_joints)
                .map(|j| {
                    if j < positions.len() {
                        positions[j]
                    } else {
                        Vec3::ZERO
                    }
                })
                .collect();

            self.db.poses.push(DatabasePose {
                joint_positions,
                joint_velocities,
                trajectory,
                foot_contacts,
                phase,
                time_in_clip: t,
                clip_id,
            });
        }
    }

    pub fn build(self) -> PoseDatabase {
        self.db
    }
}

impl Default for PoseDatabaseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// InertialBlender
// ---------------------------------------------------------------------------

/// Smooths pose transitions using a per-joint critically damped spring.
pub struct InertialBlender {
    pub current_pose: Vec<Vec3>,
    /// Per-joint positional offset being damped toward zero.
    pub offset: Vec<Vec3>,
    /// Per-joint velocity of the offset.
    pub velocity: Vec<Vec3>,
    /// Spring half-life in seconds (default 0.1s).
    pub half_life: f32,
}

impl InertialBlender {
    pub fn new(joint_count: usize) -> Self {
        Self {
            current_pose: vec![Vec3::ZERO; joint_count],
            offset: vec![Vec3::ZERO; joint_count],
            velocity: vec![Vec3::ZERO; joint_count],
            half_life: 0.1,
        }
    }

    /// Call when transitioning to a new target pose.  The offset absorbs the
    /// discontinuity so the output remains continuous.
    pub fn begin_transition(&mut self, target_positions: &[Vec3]) {
        let n = self.current_pose.len().min(target_positions.len());
        for (j, offset) in self.offset[..n].iter_mut().enumerate() {
            *offset = self.current_pose[j] - target_positions[j];
        }
        // velocity is left unchanged (preserves momentum across transitions)
    }

    /// Advance one simulation step and return the blended joint positions.
    pub fn update(&mut self, target_positions: &[Vec3], dt: f32) -> Vec<Vec3> {
        let n = self.offset.len().min(target_positions.len());
        let hl = self.half_life.max(f32::EPSILON);

        for j in 0..n {
            // Critically damped spring integration (semi-implicit Euler).
            // Read current velocity before the mutable borrow of offset.
            let vel = self.velocity[j];
            self.offset[j] += vel * dt;
            let offset = self.offset[j];
            self.velocity[j] += (-offset / (hl * hl) - 2.0 * vel / hl) * dt;
        }

        let mut output = Vec::with_capacity(n);
        for (j, &tp) in target_positions[..n].iter().enumerate() {
            let blended = tp + self.offset[j];
            self.current_pose[j] = blended;
            output.push(blended);
        }
        output
    }
}

// ---------------------------------------------------------------------------
// MotionMatcher
// ---------------------------------------------------------------------------

/// Top-level interface combining database search and inertial pose blending.
pub struct MotionMatcher {
    pub db: PoseDatabase,
    pub blender: InertialBlender,
    pub current_pose_idx: usize,
    /// Weight on the phase penalty term (default 0.3).
    pub phase_lambda: f32,
}

impl MotionMatcher {
    pub fn new(db: PoseDatabase, joint_count: usize) -> Self {
        Self {
            db,
            blender: InertialBlender::new(joint_count),
            current_pose_idx: 0,
            phase_lambda: 0.3,
        }
    }

    /// Query the database for the best-matching pose and advance the blender.
    /// Returns the index of the matched `DatabasePose`.
    pub fn query(&mut self, desired_feature: [f32; 9], dt: f32) -> usize {
        let next_idx = self
            .db
            .nearest_continuing(self.current_pose_idx, &desired_feature, self.phase_lambda);

        let target_positions = self.db.poses[next_idx].joint_positions.clone();

        if next_idx != self.current_pose_idx {
            self.blender.begin_transition(&target_positions);
        }

        self.blender.update(&target_positions, dt);
        self.current_pose_idx = next_idx;
        next_idx
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Existing tests (must keep passing)
    // -----------------------------------------------------------------------

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
        db.add_pose(make_pose(Vec3::X, Vec3::Z, vec![]));
        db.add_pose(make_pose(Vec3::X, -Vec3::Z, vec![]));

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

    // -----------------------------------------------------------------------
    // New tests
    // -----------------------------------------------------------------------

    /// Helper: build a minimal DatabasePose with the given root velocity and
    /// enough joints to satisfy LEFT_FOOT_JOINT (4) and RIGHT_FOOT_JOINT (8).
    fn make_db_pose(root_vel: Vec3, phase: f32) -> DatabasePose {
        let n = 10;
        DatabasePose {
            joint_positions: vec![Vec3::ZERO; n],
            joint_velocities: {
                let mut v = vec![Vec3::ZERO; n];
                v[0] = root_vel;
                v
            },
            trajectory: [Vec3::ZERO; 4],
            foot_contacts: [false, false],
            phase,
            time_in_clip: 0.0,
            clip_id: 0,
        }
    }

    #[test]
    fn inertial_blender_converges() {
        let n = 3;
        let mut blender = InertialBlender::new(n);

        // Initialise with a large offset.
        blender.offset = vec![Vec3::splat(1.0); n];

        let target = vec![Vec3::ZERO; n];
        let dt = 1.0 / 60.0;

        let mut output = Vec::new();
        for _ in 0..60 {
            output = blender.update(&target, dt);
        }

        for pos in &output {
            assert!(
                pos.length() < 0.1,
                "Blender should converge to target within 60 frames; got {:?}",
                pos
            );
        }
    }

    #[test]
    fn pose_database_nearest_returns_valid_index() {
        let db = PoseDatabase {
            poses: vec![
                make_db_pose(Vec3::X, 0.0),
                make_db_pose(Vec3::Y, 1.0),
                make_db_pose(Vec3::Z, 2.0),
            ],
            feature_dim: 9,
        };

        let query = [0.0f32; 9];
        let idx = db.nearest(&query).expect("should return Some");
        assert!(idx < 3, "index {} must be in [0, 2]", idx);
    }

    #[test]
    fn euclidean_dist_9d_same_vector_is_zero() {
        let v: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(euclidean_dist_9d(&v, &v), 0.0);
    }

    #[test]
    fn min_angular_distance_wraps() {
        // min_angular_dist(6.0, 0.0) should be 2π - 6.0 ≈ 0.2832
        let d = min_angular_distance(6.0, 0.0);
        let expected = std::f32::consts::TAU - 6.0; // ≈ 0.2832
        assert!(
            (d - expected).abs() < 1e-5,
            "Expected ≈ {}, got {}",
            expected,
            d
        );
    }

    #[test]
    fn pose_database_builder_produces_poses() {
        let mut builder = PoseDatabaseBuilder::new();
        let num_joints = 10;

        // A trivial clip: all joints at origin for 1 second.
        builder.ingest(0, 1.0, num_joints, |_t| vec![Vec3::ZERO; num_joints]);

        let db = builder.build();

        // 10 Hz over 1 s → samples at 0.0, 0.1, …, 1.0 = 11 poses.
        assert_eq!(
            db.poses.len(),
            11,
            "Expected 11 poses, got {}",
            db.poses.len()
        );
    }
}
