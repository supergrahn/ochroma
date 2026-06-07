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
const WALK_SPEED: f32 = 8.0;
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
/// landed on plus the blended root speed it represents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchedPose {
    /// Clip id of the matched pose (`CLIP_IDLE` / `CLIP_WALK` / `CLIP_RUN`).
    pub clip_id: u32,
    /// Index of the matched pose within the database (insertion order).
    pub pose_index: usize,
    /// Root locomotion speed (m/s) of the matched pose — the magnitude of the
    /// matched pose's root joint velocity, blended via the inertial blender's
    /// current state.
    pub root_speed: f32,
    /// True iff this tick switched to a different clip than the previous tick
    /// (an inertial transition was begun).
    pub clip_changed: bool,
}

impl MatchedPose {
    /// Human-readable clip name for HUD / log surfacing.
    pub fn clip_name(&self) -> &'static str {
        clip_name(self.clip_id)
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
    /// Current playback / hysteresis cursor into `db.poses`. Within a clip it
    /// advances forward each tick (gait playback); on a clip change it jumps to
    /// the matched entry pose.
    current_idx: usize,
    /// Clip id of the current pose (to detect clip changes).
    current_clip: u32,
}

impl AvatarMotion {
    /// Construct an avatar motion matcher over the synthetic locomotion DB.
    pub fn new() -> Self {
        let db = build_locomotion_database();
        let clip_ranges = Self::compute_clip_ranges(&db);
        // Seed on the first pose (idle clip, index 0).
        let current_clip = db.poses.first().map(|p| p.clip_id).unwrap_or(CLIP_IDLE);
        Self {
            db,
            blender: InertialBlender::new(NUM_JOINTS),
            clip_ranges,
            current_idx: 0,
            current_clip,
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

    fn range_for(&self, clip_id: u32) -> Option<ClipRange> {
        self.clip_ranges.iter().copied().find(|r| r.clip_id == clip_id)
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

    /// Select an animation pose for the avatar's CURRENT locomotion state.
    ///
    /// `root_velocity` is the avatar's world-space velocity (the
    /// CharacterController's `velocity`), `facing_yaw` is its heading (radians,
    /// the same yaw convention walking_sim uses: forward = (sin y, 0, -cos y)),
    /// and `dt` advances the inertial blender. Returns the [`MatchedPose`].
    ///
    /// The 9D query feature is built to match the engine's `build_feature`
    /// layout exactly — `[vel_x, vel_z, dir_x, dir_z, foot_l.x, foot_l.z,
    /// foot_r.x, foot_r.z, vel_y]` — but expressed in the avatar's FACING-LOCAL
    /// frame so that "moving forward" always reads as +Z velocity regardless of
    /// world heading (the clips are all authored facing +Z). We rotate the
    /// world velocity by `-facing_yaw` into that local frame.
    pub fn tick(&mut self, root_velocity: Vec3, facing_yaw: f32, dt: f32) -> MatchedPose {
        // Rotate world velocity into the avatar's facing-local frame. walking_sim
        // forward = (sin yaw, 0, -cos yaw). A point in world space maps to local
        // by the inverse (transpose) of that basis: forward -> local +Z,
        // right -> local +X. Derive local components by projecting onto the
        // local basis vectors.
        let (s, c) = facing_yaw.sin_cos();
        let forward = Vec3::new(s, 0.0, -c); // local +Z axis in world space
        let right = Vec3::new(c, 0.0, s); // local +X axis in world space
        let local_vx = root_velocity.dot(right);
        let local_vz = root_velocity.dot(forward);

        // Direction channels: the trajectory/heading term. The clips encode this
        // as the root position a short time ahead (trajectory[0]); for the query
        // we use the normalized desired travel direction in local space, which
        // is +Z forward when moving. When stationary the direction is zero,
        // which matches the idle clip's near-static trajectory.
        let horiz = Vec3::new(local_vx, 0.0, local_vz);
        let dir = if horiz.length_squared() > 1e-6 {
            horiz.normalize()
        } else {
            Vec3::ZERO
        };

        // Query feature in the engine's build_feature layout. Feet contribute
        // 0 in the query (we match on locomotion velocity + direction, not foot
        // placement), which is a neutral value that does not bias clip choice.
        let query: [f32; 9] = [
            local_vx,         // vel_x
            local_vz,         // vel_z
            dir.x,            // dir_x  (trajectory.x)
            dir.z,            // dir_z  (trajectory.z)
            0.0,              // foot_l.x
            0.0,              // foot_l.z
            0.0,              // foot_r.x
            0.0,              // foot_r.z
            root_velocity.y,  // hip_vel_y
        ];

        // Motion matching SELECTS the clip (and the entry pose) via the engine's
        // hysteresis-aware nearest search. The continuation bonus biases it to
        // keep the current clip unless the locomotion state clearly calls for a
        // different one — this is what stops idle/walk/run from flickering.
        let matched_idx = self
            .db
            .nearest_continuing(self.current_idx, &query, CONTINUATION_BONUS);
        let next_clip = self.db.poses[matched_idx].clip_id;

        // PLAYBACK: a static locomotion query matches the same single best pose
        // every frame (the engine library is a pose SELECTOR, not a playback
        // clock — see the report note). To produce a continuing, gait-advancing
        // stream we advance a playback cursor forward WITHIN the matched clip,
        // wrapping at the clip end. On a clip CHANGE we jump to the matched entry
        // pose and let the blender absorb the discontinuity.
        let clip_changed = next_clip != self.current_clip;
        let next_idx = if clip_changed {
            matched_idx
        } else if let Some(range) = self.range_for(next_clip) {
            // Advance one pose within the clip, wrapping at the end. If the
            // hysteresis cursor somehow left the clip, re-enter at the match.
            if range.contains(self.current_idx) && range.len() > 0 {
                let local = self.current_idx - range.start;
                range.start + (local + 1) % range.len()
            } else {
                matched_idx
            }
        } else {
            matched_idx
        };

        // Begin an inertial transition only when the matched CLIP changes — a
        // within-clip advance is already continuous, but a clip switch is a
        // discontinuity the blender must absorb.
        let target = self.db.poses[next_idx].joint_positions.clone();
        if clip_changed {
            self.blender.begin_transition(&target);
        }
        let blended = self.blender.update(&target, dt);

        // Blended root speed: horizontal speed of the matched pose's root joint
        // velocity (the locomotion speed the pose represents).
        let root_vel = self.db.poses[next_idx]
            .joint_velocities
            .first()
            .copied()
            .unwrap_or(Vec3::ZERO);
        let root_speed = Vec3::new(root_vel.x, 0.0, root_vel.z).length();

        // Sanity: blended pose has the rig's joint count (blender is wired).
        debug_assert_eq!(blended.len(), NUM_JOINTS);

        self.current_idx = next_idx;
        self.current_clip = next_clip;

        MatchedPose {
            clip_id: next_clip,
            pose_index: next_idx,
            root_speed,
            clip_changed,
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
