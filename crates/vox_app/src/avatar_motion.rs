//! Avatar motion matching — game-layer glue that wires the engine's
//! [`vox_render::motion_matching`] library into a real consumer.
//!
//! The engine crate ships a complete motion-matching library (a
//! [`PoseDatabase`] built from animation clips, a brute-force nearest-pose
//! search with hysteresis, and an [`InertialBlender`] for continuous pose
//! transitions) but had zero consumers. This module is that consumer: it
//! procedurally builds a small locomotion database (idle / walk / run) and
//! selects an animation pose each frame from an avatar's REAL locomotion state
//! (root velocity + facing). It lives in `vox_app` (the GAME layer) because the
//! synthetic clips and the locomotion vocabulary (idle/walk/run) are
//! game-specific; the engine library stays game-agnostic.
//!
//! Determinism: the database is built from analytic joint trajectories (no
//! rng, no time seeds) and the matcher uses the library's brute-force search,
//! which iterates poses in insertion order. The same velocity sequence
//! therefore always yields the same matched pose indices — exercised by the
//! `determinism_*` unit test and the smoke's 4th assertion.
//!
//! ## What the matcher selects vs. what the cursor advances (finding [6])
//!
//! The engine's `build_feature` stores channels 2–7 as ABSOLUTE world positions
//! (the root trajectory point `speed*(t+0.1)` and the two world foot positions
//! `speed*t + fore`). A naive query that left those channels at 0/unit-vector
//! made them incommensurable with the stored clips, collapsing the 9D match to a
//! velocity threshold and pinning `nearest_continuing` to each clip's entry pose
//! (the old bug). We now build the query in the SAME clip-local space: a gait
//! cursor (`gait_time`, advanced by `dt` and wrapped at the clip duration) drives
//! a clip-local baseline `speed * gait_time`, and the trajectory + foot channels
//! are synthesized from the SAME analytic gait the clips use ([`eval_locomotion`])
//! evaluated at that cursor phase. So a query at gait phase 0.25 carries a
//! genuinely different foot signature than one at phase 0.75 and the matcher
//! discriminates poses WITHIN a clip — proven by
//! `walk_query_phase_discriminates_within_clip`.
//!
//! With the feature layout fixed, pose selection is driven FULLY through the
//! engine matcher (`nearest_continuing`): the returned index IS the played pose
//! (idle/walk/run classification AND within-clip phase). The gait cursor exists
//! only to author the phase-discriminating query; it is not a separate playback
//! clock layered over the matcher. This is the variant we shipped.

use glam::Vec3;
use vox_render::motion_matching::{InertialBlender, PoseDatabase, PoseDatabaseBuilder};

/// Clip id for the idle (near-stationary) locomotion clip.
pub const CLIP_IDLE: u32 = 0;
/// Clip id for the walk locomotion clip.
pub const CLIP_WALK: u32 = 1;
/// Clip id for the run locomotion clip.
pub const CLIP_RUN: u32 = 2;

/// Number of joints in each synthetic locomotion pose.
///
/// The engine's `build_feature` reads the LEFT foot at joint index 4 and the
/// RIGHT foot at joint index 8 (see `motion_matching::build_feature`), so the
/// skeleton must have at least 9 joints. We use a compact 9-joint rig:
///   0 = root (hip), 1 = spine, 2 = head, 3 = left hip, 4 = LEFT foot,
///   5 = right hip, 6 = left hand, 7 = right hand, 8 = RIGHT foot.
const NUM_JOINTS: usize = 9;
const LEFT_FOOT: usize = 4;
const RIGHT_FOOT: usize = 8;

/// Forward locomotion speed (m/s) of each clip's root.
///
/// NOTE / DEVIATION FROM SPEC EXAMPLE: the adoption brief suggested walk ≈
/// 1.5 m/s and run ≈ 4 m/s. The real consumer — the walking_sim
/// `CharacterController` — moves the avatar at `speed = 8.0` m/s when walking.
/// If the walk clip sat at 1.5 m/s, an 8 m/s query would match the RUN clip,
/// not WALK, and the smoke's "walk clip selected while walking" assertion would
/// fail. We therefore tune the clip speeds to the consumer's ACTUAL locomotion
/// band so matching reflects reality: idle 0, walk = the controller's 8 m/s,
/// run = a faster 16 m/s sprint band. The idle/walk/run structure, the forward
/// motion, and the sinusoidal gait phase are all exactly as specified.
pub const WALK_SPEED: f32 = 8.0;
const RUN_SPEED: f32 = 16.0;

/// Gait stride frequency (full foot cycles per second) for walk/run.
const WALK_GAIT_HZ: f32 = 1.6;
const RUN_GAIT_HZ: f32 = 2.6;

/// Clip durations (seconds). A couple of full gait cycles each, sampled at the
/// library's fixed 10 Hz so every clip contributes many poses.
const CLIP_DURATION: f32 = 2.0;

/// Continuation bonus passed to `nearest_continuing` — biases the search toward
/// staying on the current pose so the matched stream advances smoothly instead
/// of jittering between equidistant candidates (hysteresis).
const CONTINUATION_BONUS: f32 = 0.5;

/// Half ground-stride a foot sweeps fore/aft, scaled by clip speed so faster
/// clips have visibly longer strides. Feet also lift above the contact
/// threshold (0.05) on the swing half of the cycle so the engine's contact /
/// phase classification actually toggles.
const STRIDE_HALF_LEN: f32 = 0.45;
const FOOT_LIFT: f32 = 0.18;

/// Build a single analytic locomotion pose evaluator for a clip moving forward
/// (+Z, the controller's forward at yaw 0) at `speed`, with a gait of
/// `gait_hz`. Returns world-space joint positions at time `t`.
///
/// The root (joint 0) advances at `speed` along +Z. The two feet sweep
/// fore/aft in counter-phase and lift off the ground on their swing half so the
/// library's foot-contact / phase logic produces a real gait phase. All motion
/// is a closed-form function of `t` — fully deterministic, no rng.
fn eval_locomotion(t: f32, speed: f32, gait_hz: f32) -> Vec<Vec3> {
    use std::f32::consts::TAU;
    let root_z = speed * t;
    // Stride length scales mildly with speed so walk and run differ in feature
    // space beyond raw velocity.
    let stride = STRIDE_HALF_LEN * (1.0 + speed / RUN_SPEED);
    let phase = TAU * gait_hz * t;

    // Left foot leads, right foot trails by half a cycle.
    let l_fore = phase.sin() * stride;
    let r_fore = (phase + std::f32::consts::PI).sin() * stride;
    // Swing lift: foot is in the air (y > contact threshold) on the forward
    // half of its cycle, planted (y = 0) on the back half.
    let l_lift = phase.cos().max(0.0) * FOOT_LIFT;
    let r_lift = (phase + std::f32::consts::PI).cos().max(0.0) * FOOT_LIFT;

    let mut joints = vec![Vec3::ZERO; NUM_JOINTS];
    joints[0] = Vec3::new(0.0, 1.0, root_z); // root / hip
    joints[1] = Vec3::new(0.0, 1.4, root_z); // spine
    joints[2] = Vec3::new(0.0, 1.7, root_z); // head
    joints[3] = Vec3::new(-0.15, 0.9, root_z); // left hip
    joints[LEFT_FOOT] = Vec3::new(-0.15, l_lift, root_z + l_fore); // LEFT foot
    joints[5] = Vec3::new(0.15, 0.9, root_z); // right hip
    joints[6] = Vec3::new(-0.3, 1.2, root_z - r_fore * 0.5); // left hand (counter-swing)
    joints[7] = Vec3::new(0.3, 1.2, root_z - l_fore * 0.5); // right hand
    joints[RIGHT_FOOT] = Vec3::new(0.15, r_lift, root_z + r_fore); // RIGHT foot
    joints
}

/// Procedurally build the idle / walk / run locomotion database through the
/// REAL [`PoseDatabaseBuilder::ingest`]. Deterministic — no rng, no clocks.
pub fn build_locomotion_database() -> PoseDatabase {
    let mut builder = PoseDatabaseBuilder::new();

    // Idle: feet planted, near-zero root velocity. A tiny breathing sway keeps
    // the pose from being perfectly degenerate but the root velocity stays ~0.
    builder.ingest(CLIP_IDLE, CLIP_DURATION, NUM_JOINTS, |t| {
        use std::f32::consts::TAU;
        let sway = (TAU * 0.3 * t).sin() * 0.01;
        let mut joints = vec![Vec3::ZERO; NUM_JOINTS];
        joints[0] = Vec3::new(0.0, 1.0 + sway, 0.0);
        joints[1] = Vec3::new(0.0, 1.4 + sway, 0.0);
        joints[2] = Vec3::new(0.0, 1.7 + sway, 0.0);
        joints[3] = Vec3::new(-0.15, 0.9, 0.0);
        joints[LEFT_FOOT] = Vec3::new(-0.15, 0.0, 0.0); // planted
        joints[5] = Vec3::new(0.15, 0.9, 0.0);
        joints[6] = Vec3::new(-0.3, 1.2, 0.0);
        joints[7] = Vec3::new(0.3, 1.2, 0.0);
        joints[RIGHT_FOOT] = Vec3::new(0.15, 0.0, 0.0); // planted
        joints
    });

    builder.ingest(CLIP_WALK, CLIP_DURATION, NUM_JOINTS, |t| {
        eval_locomotion(t, WALK_SPEED, WALK_GAIT_HZ)
    });

    builder.ingest(CLIP_RUN, CLIP_DURATION, NUM_JOINTS, |t| {
        eval_locomotion(t, RUN_SPEED, RUN_GAIT_HZ)
    });

    builder.build()
}

/// The result of a single [`AvatarMotion::tick`]: which clip/pose the matcher
/// landed on, the inertial-blended joint positions, and the root speed.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchedPose {
    /// Clip id of the matched pose (`CLIP_IDLE` / `CLIP_WALK` / `CLIP_RUN`).
    pub clip_id: u32,
    /// Index of the matched pose within the database (insertion order).
    pub pose_index: usize,
    /// Root locomotion speed (m/s) of the matched pose — the magnitude of the
    /// matched pose's root joint velocity.
    pub root_speed: f32,
    /// True iff this tick switched to a different clip than the previous tick
    /// (an inertial transition was begun).
    pub clip_changed: bool,
    /// The inertial-blended joint positions for THIS tick (finding [4]). During a
    /// clip transition these interpolate from the previous pose toward the matched
    /// target, so a mid-transition root-joint value lies strictly between source
    /// and target — the InertialBlender's output is now observable, not discarded.
    pub blended_joints: Vec<Vec3>,
}

impl MatchedPose {
    /// Human-readable clip name for HUD / log surfacing.
    pub fn clip_name(&self) -> &'static str {
        clip_name(self.clip_id)
    }

    /// Root-joint (hip) world Y of the inertial-blended pose — a single scalar
    /// the HUD/log surfaces to make the blend observable (finding [4]). Mid-clip
    /// transition this reads between the source and target hip heights.
    pub fn blended_root_y(&self) -> f32 {
        self.blended_joints.first().map(|p| p.y).unwrap_or(0.0)
    }
}

/// Map a clip id to its display name.
pub fn clip_name(clip_id: u32) -> &'static str {
    match clip_id {
        CLIP_IDLE => "idle",
        CLIP_WALK => "walk",
        CLIP_RUN => "run",
        _ => "?",
    }
}

/// Count "non-continuing" steps in a `(clip_id, global_pose_index)` stream:
/// consecutive SAME-clip matches whose pose index moves BACKWARD by anything
/// other than a single wrap at the clip's end. A continuing playback stream has
/// zero such steps. Used by both the unit test and the smoke's assertion 3.
///
/// `avatar` supplies the clip ranges needed to translate global indices to
/// clip-relative ones (so a wrap from the clip's last pose back to its first is
/// recognised and NOT counted as a violation).
pub fn continuity_violations(avatar: &AvatarMotion, stream: &[(u32, usize)]) -> u32 {
    let mut v = 0u32;
    for w in stream.windows(2) {
        let (c0, g0) = w[0];
        let (c1, g1) = w[1];
        if c0 != c1 {
            continue; // clip change resets continuity
        }
        let (Some((_, l0)), Some((_, l1))) = (avatar.local_index(g0), avatar.local_index(g1))
        else {
            continue;
        };
        if l1 < l0 {
            // Backward move within the clip. Allow only a wrap: the previous
            // pose was at/near the clip's last index and the next is at/near 0.
            let clip_len = avatar
                .clip_ranges
                .iter()
                .find(|r| r.clip_id == c0)
                .map(|r| r.len())
                .unwrap_or(0);
            let looks_like_wrap = clip_len > 0 && l0 >= clip_len - 1 && l1 == 0;
            if !looks_like_wrap {
                v += 1;
            }
        }
    }
    v
}

/// Per-avatar motion-matching state: owns the locomotion database, an inertial
/// blender for smooth pose transitions, and the index of the last matched pose
/// (the seed for `nearest_continuing`'s hysteresis).
/// Half-open `[start, end)` index range of a clip's poses in the database.
#[derive(Clone, Copy, Debug)]
struct ClipRange {
    clip_id: u32,
    start: usize,
    end: usize,
}

impl ClipRange {
    fn len(&self) -> usize {
        self.end - self.start
    }
    fn contains(&self, idx: usize) -> bool {
        idx >= self.start && idx < self.end
    }
}

pub struct AvatarMotion {
    db: PoseDatabase,
    blender: InertialBlender,
    /// Per-clip index ranges in `db.poses` (clips are ingested contiguously).
    clip_ranges: Vec<ClipRange>,
    /// Current pose index into `db.poses` — the last matched pose, and the seed
    /// for `nearest_continuing`'s hysteresis.
    current_idx: usize,
    /// Clip id of the current pose (to detect clip changes).
    current_clip: u32,
    /// Gait cursor: clip-relative time (seconds) used to author the
    /// phase-discriminating query (finding [3]). Advances by `dt` each tick and
    /// wraps at [`CLIP_DURATION`] so the synthesized query foot signature sweeps
    /// through a full gait cycle, letting the matcher pick distinct within-clip
    /// poses. NOT a playback clock layered over the matcher — the matched index is
    /// the played pose.
    gait_time: f32,
}

impl AvatarMotion {
    /// Construct an avatar motion matcher over the synthetic locomotion DB.
    pub fn new() -> Self {
        let db = build_locomotion_database();
        let clip_ranges = Self::compute_clip_ranges(&db);
        // Seed on the first pose (idle clip, index 0).
        let current_clip = db.poses.first().map(|p| p.clip_id).unwrap_or(CLIP_IDLE);
        // Finding [4]: seed the blender's current pose to the initial (idle entry)
        // pose so the FIRST transition blends from a real pose, not the world
        // origin (the old cold-start computed a full-magnitude bogus offset).
        let mut blender = InertialBlender::new(NUM_JOINTS);
        if let Some(first) = db.poses.first() {
            blender.current_pose = first.joint_positions.clone();
        }
        Self {
            db,
            blender,
            clip_ranges,
            current_idx: 0,
            current_clip,
            gait_time: 0.0,
        }
    }

    /// Compute the contiguous `[start, end)` index range of each clip. Clips are
    /// ingested in order, so each clip occupies a single contiguous run.
    fn compute_clip_ranges(db: &PoseDatabase) -> Vec<ClipRange> {
        let mut ranges: Vec<ClipRange> = Vec::new();
        for (i, p) in db.poses.iter().enumerate() {
            match ranges.last_mut() {
                Some(r) if r.clip_id == p.clip_id => r.end = i + 1,
                _ => ranges.push(ClipRange {
                    clip_id: p.clip_id,
                    start: i,
                    end: i + 1,
                }),
            }
        }
        ranges
    }

    /// Local (clip-relative) pose index for a global database index, if the
    /// index belongs to a known clip.
    fn local_index(&self, global_idx: usize) -> Option<(u32, usize)> {
        self.clip_ranges
            .iter()
            .find(|r| r.contains(global_idx))
            .map(|r| (r.clip_id, global_idx - r.start))
    }

    /// Number of poses in the underlying database.
    pub fn pose_count(&self) -> usize {
        self.db.poses.len()
    }

    /// Number of poses belonging to `clip_id`.
    pub fn pose_count_for(&self, clip_id: u32) -> usize {
        self.db.poses.iter().filter(|p| p.clip_id == clip_id).count()
    }

    /// Raw (UN-blended) root-joint world position of the pose at `pose_index` —
    /// the blend target. Used to verify the inertial blend interpolates toward,
    /// but does not snap to, the matched pose (finding [4]).
    pub fn pose_root_position(&self, pose_index: usize) -> Vec3 {
        self.db
            .poses
            .get(pose_index)
            .and_then(|p| p.joint_positions.first().copied())
            .unwrap_or(Vec3::ZERO)
    }

    /// Select an animation pose for the avatar's CURRENT locomotion state.
    ///
    /// `root_velocity` is the avatar's world-space velocity (the
    /// CharacterController's `velocity`), `facing_yaw` is its heading (radians,
    /// the same yaw convention walking_sim uses: forward = (sin y, 0, -cos y)),
    /// and `dt` advances the inertial blender. Returns the [`MatchedPose`].
    ///
    /// The 9D query feature is built to match the engine's `build_feature`
    /// layout AND SEMANTICS exactly — `[vel_x, vel_z, dir_x, dir_z, foot_l.x,
    /// foot_l.z, foot_r.x, foot_r.z, vel_y]` — expressed in the avatar's
    /// FACING-LOCAL frame so "moving forward" always reads as +Z velocity
    /// regardless of world heading (the clips are all authored facing +Z). We
    /// rotate the world velocity by `-facing_yaw` into that local frame.
    ///
    /// Crucially (finding [3]) channels 2–7 carry the SAME absolute-position
    /// semantics the clips store: the trajectory point and foot positions are
    /// synthesized from the SAME analytic gait ([`eval_locomotion`]) at the gait
    /// cursor's clip-phase, anchored to the matched clip's BAND speed. That makes
    /// the query commensurate with the stored poses, so the matcher discriminates
    /// poses WITHIN a clip instead of pinning to the entry pose. The position
    /// channels use the quantized BAND speed (0/walk/run) rather than the raw
    /// instantaneous speed so a small velocity bob cannot shift the baseline and
    /// jolt the matched index backward (keeping the stream continuous); the
    /// velocity channels (0,1,8) still carry the real instantaneous velocity, which
    /// is what classifies idle/walk/run.
    pub fn tick(&mut self, root_velocity: Vec3, facing_yaw: f32, dt: f32) -> MatchedPose {
        // Rotate world velocity into the avatar's facing-local frame. walking_sim
        // forward = (sin yaw, 0, -cos yaw). Project onto the local basis vectors.
        let (s, c) = facing_yaw.sin_cos();
        let forward = Vec3::new(s, 0.0, -c); // local +Z axis in world space
        let right = Vec3::new(c, 0.0, s); // local +X axis in world space
        let local_vx = root_velocity.dot(right);
        let local_vz = root_velocity.dot(forward);
        let local_speed = Vec3::new(local_vx, 0.0, local_vz).length();

        // Advance the gait cursor (clip-relative time), wrapping each cycle.
        self.gait_time = (self.gait_time + dt).rem_euclid(CLIP_DURATION);

        // Quantize the avatar's speed to the nearest clip BAND (idle/walk/run) and
        // synthesize the position channels at that band's speed + gait frequency,
        // so they exactly mirror one clip's analytic motion. This stabilizes the
        // within-clip phase signal against velocity bob.
        let (phase_speed, gait_hz) = {
            let to_idle = local_speed;
            let to_walk = (local_speed - WALK_SPEED).abs();
            let to_run = (local_speed - RUN_SPEED).abs();
            if to_run <= to_walk && to_run <= to_idle {
                (RUN_SPEED, RUN_GAIT_HZ)
            } else if to_walk <= to_idle {
                (WALK_SPEED, WALK_GAIT_HZ)
            } else {
                (0.0, WALK_GAIT_HZ)
            }
        };

        // Synthesize the query's channels 2–7 from the SAME analytic gait the
        // clips use, at the cursor phase, anchored to the matched band's baseline
        // `phase_speed * gait_time`. When near-stationary the band is idle
        // (phase_speed 0), so the baseline and foot sweep collapse to ~0 — matching
        // the idle clip's planted feet and static trajectory.
        let query_joints = eval_locomotion(self.gait_time, phase_speed, gait_hz);
        // Trajectory[0] = root position a short step (0.1s) ahead, exactly as the
        // builder ingests it (clamped to the clip duration).
        let traj_t = (self.gait_time + 0.1).min(CLIP_DURATION);
        let traj0 = eval_locomotion(traj_t, phase_speed, gait_hz)[0];
        let foot_l = query_joints[LEFT_FOOT];
        let foot_r = query_joints[RIGHT_FOOT];

        // Query feature in the engine's build_feature layout/semantics.
        let query: [f32; 9] = [
            local_vx,        // vel_x
            local_vz,        // vel_z
            traj0.x,         // dir_x  (trajectory.x)
            traj0.z,         // dir_z  (trajectory.z)
            foot_l.x,        // foot_l.x
            foot_l.z,        // foot_l.z
            foot_r.x,        // foot_r.x
            foot_r.z,        // foot_r.z
            root_velocity.y, // hip_vel_y
        ];

        // Motion matching SELECTS the pose via the engine's hysteresis-aware
        // nearest search. With the feature layout fixed, this drives BOTH clip
        // classification AND within-clip phase — the returned index is the played
        // pose (finding [6]). The continuation bonus biases toward the current
        // pose so the stream advances smoothly instead of flickering.
        let next_idx = self
            .db
            .nearest_continuing(self.current_idx, &query, CONTINUATION_BONUS);
        let next_clip = self.db.poses[next_idx].clip_id;
        let clip_changed = next_clip != self.current_clip;

        // Begin an inertial transition only when the matched CLIP changes — a
        // within-clip advance is already continuous, but a clip switch is a
        // discontinuity the blender must absorb.
        let target = self.db.poses[next_idx].joint_positions.clone();
        if clip_changed {
            self.blender.begin_transition(&target);
        }
        let blended = self.blender.update(&target, dt);
        debug_assert_eq!(blended.len(), NUM_JOINTS);

        // Blended root speed: horizontal speed of the matched pose's root joint
        // velocity (the locomotion speed the pose represents).
        let root_vel = self.db.poses[next_idx]
            .joint_velocities
            .first()
            .copied()
            .unwrap_or(Vec3::ZERO);
        let root_speed = Vec3::new(root_vel.x, 0.0, root_vel.z).length();

        self.current_idx = next_idx;
        self.current_clip = next_clip;

        MatchedPose {
            clip_id: next_clip,
            pose_index: next_idx,
            root_speed,
            clip_changed,
            blended_joints: blended,
        }
    }
}

impl Default for AvatarMotion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A facing-local forward velocity query at the given world speed (yaw 0,
    /// so world == local and forward is +Z... walking_sim forward at yaw 0 is
    /// (0,0,-1), so "forward" world velocity is -Z; we feed that and rely on the
    /// tick's local-frame rotation to turn it into +Z locally).
    fn forward_world_velocity(speed: f32) -> Vec3 {
        // yaw 0 forward = (sin0, 0, -cos0) = (0, 0, -1).
        Vec3::new(0.0, 0.0, -speed)
    }

    #[test]
    fn database_has_poses_for_every_clip() {
        let db = build_locomotion_database();
        let idle = db.poses.iter().filter(|p| p.clip_id == CLIP_IDLE).count();
        let walk = db.poses.iter().filter(|p| p.clip_id == CLIP_WALK).count();
        let run = db.poses.iter().filter(|p| p.clip_id == CLIP_RUN).count();
        // 10 Hz over 2.0 s = 21 poses per clip (inclusive of t=0).
        assert_eq!(idle, 21, "idle clip pose count");
        assert_eq!(walk, 21, "walk clip pose count");
        assert_eq!(run, 21, "run clip pose count");
        assert_eq!(db.poses.len(), 63, "total pose count");
    }

    #[test]
    fn walk_velocity_query_matches_walk_clip() {
        let mut avatar = AvatarMotion::new();
        // Feed the controller's real walk speed (8 m/s) forward, yaw 0.
        let m = avatar.tick(forward_world_velocity(WALK_SPEED), 0.0, 1.0 / 60.0);
        assert_eq!(
            m.clip_id, CLIP_WALK,
            "8 m/s forward must match the WALK clip, got {} (speed {:.2})",
            m.clip_name(),
            m.root_speed
        );
    }

    #[test]
    fn idle_velocity_query_matches_idle_clip() {
        let mut avatar = AvatarMotion::new();
        let m = avatar.tick(Vec3::ZERO, 0.0, 1.0 / 60.0);
        assert_eq!(
            m.clip_id, CLIP_IDLE,
            "zero velocity must match the IDLE clip, got {}",
            m.clip_name()
        );
    }

    #[test]
    fn run_velocity_query_matches_run_clip() {
        let mut avatar = AvatarMotion::new();
        let m = avatar.tick(forward_world_velocity(RUN_SPEED), 0.0, 1.0 / 60.0);
        assert_eq!(
            m.clip_id, CLIP_RUN,
            "16 m/s forward must match the RUN clip, got {}",
            m.clip_name()
        );
    }

    #[test]
    fn clip_change_flag_set_on_transition() {
        let mut avatar = AvatarMotion::new();
        // First tick at idle (seed clip is idle) — no change.
        let m0 = avatar.tick(Vec3::ZERO, 0.0, 1.0 / 60.0);
        assert_eq!(m0.clip_id, CLIP_IDLE);
        assert!(!m0.clip_changed, "idle->idle should not flag a clip change");
        // Now walk — must flag a transition.
        let m1 = avatar.tick(forward_world_velocity(WALK_SPEED), 0.0, 1.0 / 60.0);
        assert_eq!(m1.clip_id, CLIP_WALK);
        assert!(m1.clip_changed, "idle->walk must flag a clip change");
    }

    #[test]
    fn determinism_same_sequence_same_indices() {
        // A scripted velocity sequence: idle, ramp to walk, hold, ramp to run,
        // back to idle. Two independent matchers must produce identical index
        // streams.
        let dt = 1.0 / 60.0;
        let seq: Vec<Vec3> = (0..120)
            .map(|i| {
                let speed = match i {
                    0..=19 => 0.0,
                    20..=59 => WALK_SPEED,
                    60..=89 => RUN_SPEED,
                    _ => 0.0,
                };
                forward_world_velocity(speed)
            })
            .collect();

        let mut a = AvatarMotion::new();
        let mut b = AvatarMotion::new();
        let idx_a: Vec<usize> = seq.iter().map(|v| a.tick(*v, 0.0, dt).pose_index).collect();
        let idx_b: Vec<usize> = seq.iter().map(|v| b.tick(*v, 0.0, dt).pose_index).collect();
        assert_eq!(idx_a, idx_b, "two matchers must reproduce identical indices");
        // And the stream is non-trivial (not stuck on one pose the whole run).
        let distinct = idx_a.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            distinct > 3,
            "matched index stream is degenerate ({distinct} distinct indices)"
        );
    }

    /// Finding [3]: the query now carries real foot/trajectory semantics, so the
    /// matcher discriminates poses WITHIN a clip by gait phase. A steady walk
    /// query sampled at gait phase 0.25 must match a DIFFERENT walk pose than the
    /// same query at phase 0.75 — proving channels 2–7 (feet/trajectory) genuinely
    /// distinguish within-clip phases (they were dead before the fix).
    #[test]
    fn walk_query_phase_discriminates_within_clip() {
        // Drive a steady walk and collect, per gait phase, which pose the matcher
        // picks. We advance the gait cursor by ticking and read the matched index
        // at the cursor times nearest phase 0.25 and 0.75 of the clip.
        let dt = 1.0 / 60.0;
        let phase_a_t = 0.25 * CLIP_DURATION; // 0.5 s
        let phase_b_t = 0.75 * CLIP_DURATION; // 1.5 s

        let pose_at = |target_t: f32| -> usize {
            let mut avatar = AvatarMotion::new();
            // Prime to WALK first (so we are within the walk clip), then advance
            // the gait cursor to the target clip-time.
            let mut last = 0usize;
            // gait_time starts at 0 and advances dt each tick; tick until it just
            // passes target_t. Each tick feeds a steady walk velocity.
            let steps = (target_t / dt).round() as usize;
            for _ in 0..steps {
                last = avatar
                    .tick(forward_world_velocity(WALK_SPEED), 0.0, dt)
                    .pose_index;
            }
            last
        };

        let a = pose_at(phase_a_t);
        let b = pose_at(phase_b_t);
        // Both must be WALK-clip poses.
        let walk_range_start = 21usize; // idle occupies 0..21
        let walk_range_end = 42usize;
        assert!(
            (walk_range_start..walk_range_end).contains(&a),
            "phase-0.25 match {a} is not a walk-clip pose"
        );
        assert!(
            (walk_range_start..walk_range_end).contains(&b),
            "phase-0.75 match {b} is not a walk-clip pose"
        );
        assert_ne!(
            a, b,
            "phase 0.25 and 0.75 matched the SAME walk pose ({a}) — feet/trajectory channels do not discriminate within the clip"
        );
    }

    /// Finding [4]: the InertialBlender output is returned in `MatchedPose` and
    /// actually interpolates across a clip transition. On idle->walk the blended
    /// root-joint Z lands STRICTLY between the idle (z≈0) and walk target Z — it
    /// neither stays at the source nor snaps to the target. Also proves the
    /// blender is seeded to a real pose (no origin cold-start snap).
    #[test]
    fn blend_output_interpolates_across_transition() {
        let dt = 1.0 / 60.0;
        let mut avatar = AvatarMotion::new();
        // Settle on idle for ~half a gait cycle so the gait cursor has advanced;
        // the idle pose stays near the origin (root z≈0). Switching to walk then
        // matches a MID-clip walk pose (root z well > 0), so the idle->walk
        // transition carries a real positional discontinuity for the blender to
        // absorb (the idle->walk ENTRY poses are near-identical, so a 1-frame
        // transition would have nothing to interpolate).
        let mut idle = avatar.tick(Vec3::ZERO, 0.0, dt);
        for _ in 0..30 {
            idle = avatar.tick(Vec3::ZERO, 0.0, dt);
        }
        assert_eq!(idle.blended_joints.len(), NUM_JOINTS, "blend output is surfaced");
        let source_z = idle.blended_joints[0].z;

        let walk = avatar.tick(forward_world_velocity(WALK_SPEED), 0.0, dt);
        assert!(walk.clip_changed, "idle->walk must begin a transition");
        // The blend starts AT the source on the transition frame (offset = full
        // discontinuity) and decays toward the target over the half-life. Sample a
        // few frames into the transition: the blended root Z must now lie STRICTLY
        // between the source (idle, z≈0) and the live walk target — proving the
        // blender is interpolating, not snapping to either endpoint.
        let mut walk = walk;
        for _ in 0..4 {
            walk = avatar.tick(forward_world_velocity(WALK_SPEED), 0.0, dt);
        }
        let target_z = avatar.pose_root_position(walk.pose_index).z;
        let blended_z = walk.blended_joints[0].z;
        let (lo, hi) = (source_z.min(target_z), source_z.max(target_z));
        assert!(
            (hi - lo) > 1e-3,
            "source ({source_z}) and target ({target_z}) root Z must differ for a meaningful test"
        );
        assert!(
            blended_z > lo + 1e-4 && blended_z < hi - 1e-4,
            "blended root Z {blended_z} must lie strictly between source {source_z} and target {target_z}"
        );
    }

    #[test]
    fn continuing_match_stream_has_no_backward_jumps() {
        // While walking steadily then varying speed slightly (real gait bob),
        // consecutive same-clip matches must never jump backward (except a wrap
        // at the clip's end) — the matched stream is continuing. This is the
        // unit-level mirror of the smoke's assertion 3.
        let mut avatar = AvatarMotion::new();
        let dt = 1.0 / 60.0;
        let mut stream = Vec::new();
        for i in 0..80 {
            // Slight speed bob around the walk band so the query (and hence the
            // matched gait phase) actually moves, exercising real advancement.
            let bob = (i as f32 * 0.4).sin() * 1.5;
            let m = avatar.tick(forward_world_velocity(WALK_SPEED + bob), 0.0, dt);
            stream.push((m.clip_id, m.pose_index));
        }
        let violations = super::continuity_violations(&avatar, &stream);
        assert_eq!(
            violations, 0,
            "continuing match stream had {violations} backward jumps within a clip"
        );
        // And it really did move through more than one pose (not pinned).
        let walk_indices: std::collections::HashSet<usize> = stream
            .iter()
            .filter(|(c, _)| *c == CLIP_WALK)
            .map(|(_, i)| *i)
            .collect();
        assert!(
            walk_indices.len() >= 2,
            "walk segment pinned a single pose ({}) — gait phase never advanced",
            walk_indices.len()
        );
    }
}
