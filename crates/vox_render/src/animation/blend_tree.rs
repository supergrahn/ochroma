//! Character animation blend trees (plan Task 4.2).
//!
//! This module provides the engine machinery for skeletal character animation:
//! keyframed [`AnimClip`]s sampled by time with looping, a [`BlendTree`] that
//! blends/layers clips, and a linear-blend-skinning bridge [`apply_pose`] that
//! deforms [`GaussianSplat`]s using `vox_core`'s [`SplatSkinData`] weights.
//!
//! ## Interpolation choice: linear keyframe lerp/slerp (not `BSpline`)
//!
//! `vox_core::skinning::BSpline::sample(t)` takes a *normalized* `t ∈ [0,1]` and
//! linearly walks adjacent control points assuming they are uniformly spaced; it
//! ignores its own knot vector. That is fine for evenly-spaced data but throws
//! away the per-keyframe `time` we need for (a) exact looping — where the wrap
//! segment must interpolate from the last keyframe back to the first so that
//! `sample(t) == sample(t + duration)` holds to 1e-5 — and (b) honest per-axis
//! translation lerps that the blend test asserts lie *between* endpoints.
//! We therefore store explicit `(time, JointPose)` keyframes and lerp/slerp
//! between the bracketing pair, with a final wrap segment back to keyframe 0.
//! We still reuse `vox_core`'s `SplatSkinData` weights for the skinning bridge.

use glam::{Quat, Vec3};
use vox_core::skinning::SplatSkinData;
use vox_core::types::GaussianSplat;

/// Number of joints in the built-in humanoid-ish skeleton shipped by
/// [`AnimClip::idle`] / [`AnimClip::walk`].
///
/// Layout: 0=root/hips, 1=spine, 2=left arm, 3=right arm, 4=left leg, 5=right leg.
pub const ANIM_JOINT_COUNT: usize = 6;

// ─────────────────────────────────────────────────────────────────────────────
// JointPose / Pose
// ─────────────────────────────────────────────────────────────────────────────

/// Local transform of a single joint at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointPose {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Default for JointPose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl JointPose {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
    };

    pub fn new(translation: Vec3, rotation: Quat, scale: f32) -> Self {
        Self { translation, rotation, scale }
    }

    /// Linear/spherical blend between two joint poses. Rotation uses shortest-arc
    /// slerp (glam's `slerp` already negates one quat when the dot is negative).
    pub fn blend(a: &Self, b: &Self, t: f32) -> Self {
        Self {
            translation: a.translation.lerp(b.translation, t),
            rotation: shortest_arc_slerp(a.rotation, b.rotation, t),
            scale: a.scale + (b.scale - a.scale) * t,
        }
    }

    /// Convert to a 4x4 column-major transform (scale, then rotate, then translate).
    pub fn to_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.translation,
        )
    }
}

/// A full-skeleton pose: one [`JointPose`] per joint, indexed by joint id.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub joints: Vec<JointPose>,
}

impl Pose {
    /// Rest pose with `count` identity joints.
    pub fn rest(count: usize) -> Self {
        Self { joints: vec![JointPose::IDENTITY; count] }
    }

    pub fn len(&self) -> usize {
        self.joints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Per-joint blend of two poses (`a` at t=0, `b` at t=1). The result has the
    /// length of the shorter input so it never indexes out of bounds.
    pub fn blend(a: &Pose, b: &Pose, t: f32) -> Pose {
        let n = a.joints.len().min(b.joints.len());
        let joints = (0..n)
            .map(|i| JointPose::blend(&a.joints[i], &b.joints[i], t))
            .collect();
        Pose { joints }
    }

    /// Additive layering: `self` plus a delta pose (joint-local). Translations and
    /// log-scale add; rotations compose. `weight` scales the additive contribution.
    pub fn add_layer(&self, additive: &Pose, weight: f32) -> Pose {
        let n = self.joints.len().min(additive.joints.len());
        let mut out = self.clone();
        for i in 0..n {
            let base = &self.joints[i];
            let add = &additive.joints[i];
            out.joints[i] = JointPose {
                translation: base.translation + add.translation * weight,
                rotation: (shortest_arc_slerp(Quat::IDENTITY, add.rotation, weight) * base.rotation)
                    .normalize(),
                scale: base.scale * (1.0 + (add.scale - 1.0) * weight),
            };
        }
        out
    }
}

/// Shortest-arc spherical interpolation. glam's [`Quat::slerp`] already flips the
/// sign of `b` when `a·b < 0`, guaranteeing the short path; we normalize the
/// endpoints first so callers can pass un-normalized quats safely.
pub fn shortest_arc_slerp(a: Quat, b: Quat, t: f32) -> Quat {
    a.normalize().slerp(b.normalize(), t)
}

// ─────────────────────────────────────────────────────────────────────────────
// AnimClip — keyframed, time-sampled, looping
// ─────────────────────────────────────────────────────────────────────────────

/// One keyframe: a full-skeleton [`Pose`] at a given local time (seconds).
#[derive(Debug, Clone)]
pub struct PoseKeyframe {
    pub time: f32,
    pub pose: Pose,
}

/// A keyframed character-animation clip. Sampled by time with looping via a
/// wrap-around segment from the last keyframe back to the first.
///
/// Keyframes must be sorted by ascending `time` and span `[0, duration]` is the
/// loopable range; the final wrap segment runs from the last keyframe's time to
/// `duration` and blends back to keyframe 0.
#[derive(Debug, Clone)]
pub struct AnimClip {
    pub name: String,
    pub duration: f32,
    pub joint_count: usize,
    pub keyframes: Vec<PoseKeyframe>,
}

impl AnimClip {
    pub fn new(name: impl Into<String>, duration: f32, joint_count: usize) -> Self {
        Self { name: name.into(), duration, joint_count, keyframes: Vec::new() }
    }

    /// Push a keyframe. Caller is responsible for ascending `time`.
    pub fn push(&mut self, time: f32, pose: Pose) {
        self.keyframes.push(PoseKeyframe { time, pose });
    }

    /// Sample the clip at local `time` (seconds), looping over `duration`.
    ///
    /// `sample(t) == sample(t + duration)` holds exactly (to f32 precision)
    /// because we map time into `[0, duration)` and the wrap segment interpolates
    /// from the final keyframe back to keyframe 0.
    pub fn sample(&self, time: f32) -> Pose {
        if self.keyframes.is_empty() {
            return Pose::rest(self.joint_count);
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].pose.clone();
        }

        let dur = self.duration.max(1e-9);
        // Wrap into [0, dur).
        let mut local = time % dur;
        if local < 0.0 {
            local += dur;
        }

        let kf = &self.keyframes;
        let last = kf.len() - 1;

        // Within an interior segment [i, i+1].
        for i in 0..last {
            let t0 = kf[i].time;
            let t1 = kf[i + 1].time;
            if local >= t0 && local <= t1 {
                let span = (t1 - t0).max(1e-9);
                let f = ((local - t0) / span).clamp(0.0, 1.0);
                return Pose::blend(&kf[i].pose, &kf[i + 1].pose, f);
            }
        }

        // Wrap segment: from last keyframe back to keyframe 0 over [last.time, dur].
        let t0 = kf[last].time;
        let span = (dur - t0).max(1e-9);
        if local >= t0 {
            let f = ((local - t0) / span).clamp(0.0, 1.0);
            return Pose::blend(&kf[last].pose, &kf[0].pose, f);
        }

        // Before the first keyframe's time (only if kf[0].time > 0): hold first.
        kf[0].pose.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BlendTree
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters fed to [`BlendTree::sample`]. Indexed by parameter id.
#[derive(Debug, Clone, Default)]
pub struct BlendParams {
    pub values: Vec<f32>,
}

impl BlendParams {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Read parameter `id`, defaulting to 0.0 if absent.
    pub fn get(&self, id: usize) -> f32 {
        self.values.get(id).copied().unwrap_or(0.0)
    }
}

/// A node in a blend tree.
///
/// Scope (documented honesty): we ship a **1D blend** (the idle↔walk by-speed
/// case the NPC needs) and an **additive layer** node, both composing recursively
/// over `Clip` leaves. This is the minimum-honest set the plan asks for; a 2D
/// blend is intentionally out of scope.
#[derive(Debug, Clone)]
pub enum BlendNode {
    /// Leaf: sample a clip by index into [`BlendTree::clips`] at the tree time.
    Clip(usize),
    /// 1D blend of two children by `params[param]` clamped to `[0, 1]`
    /// (0 → `low`, 1 → `high`).
    Blend1D { param: usize, low: Box<BlendNode>, high: Box<BlendNode> },
    /// Additive layer: `base` plus `additive`'s delta scaled by `params[param]`.
    Additive { param: usize, base: Box<BlendNode>, additive: Box<BlendNode> },
}

/// A blend tree over a shared clip set. Deterministic: identical
/// `(time, params)` → identical [`Pose`].
#[derive(Debug, Clone)]
pub struct BlendTree {
    pub clips: Vec<AnimClip>,
    pub root: BlendNode,
    pub joint_count: usize,
}

impl BlendTree {
    pub fn new(clips: Vec<AnimClip>, root: BlendNode, joint_count: usize) -> Self {
        Self { clips, root, joint_count }
    }

    /// Sample the tree at `time` (seconds, per-clip looping) under `params`.
    pub fn sample(&self, time: f32, params: &BlendParams) -> Pose {
        self.sample_node(&self.root, time, params)
    }

    fn sample_node(&self, node: &BlendNode, time: f32, params: &BlendParams) -> Pose {
        match node {
            BlendNode::Clip(i) => self
                .clips
                .get(*i)
                .map(|c| c.sample(time))
                .unwrap_or_else(|| Pose::rest(self.joint_count)),
            BlendNode::Blend1D { param, low, high } => {
                let t = params.get(*param).clamp(0.0, 1.0);
                let a = self.sample_node(low, time, params);
                let b = self.sample_node(high, time, params);
                Pose::blend(&a, &b, t)
            }
            BlendNode::Additive { param, base, additive } => {
                let w = params.get(*param).clamp(0.0, 1.0);
                let base = self.sample_node(base, time, params);
                let add = self.sample_node(additive, time, params);
                base.add_layer(&add, w)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Skinning bridge — linear blend skinning onto GaussianSplats
// ─────────────────────────────────────────────────────────────────────────────

/// Deform splats in place by a sampled [`Pose`] using `vox_core`'s
/// [`SplatSkinData`] weights (linear blend skinning).
///
/// For each splat:
///   * the new position is the weighted sum of `joint.to_matrix() * bind_position`
///     over its (up to 4) influencing joints, and
///   * the splat's orientation axes are rotated by the weighted-average joint
///     rotation (re-orthonormalized), so disks/ellipsoids follow the joints.
///
/// `skins`, `splats`, and `bind_positions` are parallel arrays; the loop runs over
/// the shortest of the three so a mismatch can never index out of bounds.
pub fn apply_pose(
    pose: &Pose,
    skins: &[SplatSkinData],
    splats: &mut [GaussianSplat],
    bind_positions: &[Vec3],
) {
    let n = skins.len().min(splats.len()).min(bind_positions.len());
    for i in 0..n {
        let skin = &skins[i];
        let bind = bind_positions[i];

        let mut new_pos = Vec3::ZERO;
        // Weighted-average rotation accumulated in quaternion space (with sign
        // alignment so we average along the short arc rather than cancelling).
        let mut acc_rot = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let mut ref_rot: Option<Quat> = None;
        let mut total_w = 0.0;

        for k in 0..4 {
            let w = skin.joint_weights[k];
            if w <= 0.0 {
                continue;
            }
            let jidx = skin.joint_indices[k] as usize;
            let jpose = pose.joints.get(jidx).copied().unwrap_or(JointPose::IDENTITY);

            // Position contribution (full skinning matrix on the bind position).
            new_pos += w * jpose.to_matrix().transform_point3(bind);

            // Rotation accumulation.
            let mut q = jpose.rotation.normalize();
            match ref_rot {
                None => ref_rot = Some(q),
                Some(r) => {
                    if r.dot(q) < 0.0 {
                        q = -q;
                    }
                }
            }
            acc_rot = Quat::from_xyzw(
                acc_rot.x + q.x * w,
                acc_rot.y + q.y * w,
                acc_rot.z + q.z * w,
                acc_rot.w + q.w * w,
            );
            total_w += w;
        }

        if total_w <= 0.0 {
            continue; // unskinned splat: leave untouched
        }

        new_pos /= total_w;
        splats[i].set_position([new_pos.x, new_pos.y, new_pos.z]);

        if acc_rot.length_squared() > 1e-12 {
            let rot = acc_rot.normalize();
            rotate_splat_axes(&mut splats[i], rot);
        }
    }
}

/// Rotate a splat's orientation by `rot`. For 2DGS surfaces we rotate the two
/// tangent axes via `set_tangents`. 3DGS volumes store a *quantized* rotation and
/// `vox_core` (read-only here) exposes no quaternion setter, so volume splats are
/// skinned positionally only — their stored orientation is left unchanged.
fn rotate_splat_axes(splat: &mut GaussianSplat, rot: Quat) {
    if splat.is_surface() {
        let tu = rot * Vec3::from(splat.tangent_u());
        let tv = rot * Vec3::from(splat.tangent_v());
        splat.set_tangents([tu.x, tu.y, tu.z], [tv.x, tv.y, tv.z]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in clips: a 6-joint humanoid-ish skeleton
// ─────────────────────────────────────────────────────────────────────────────

/// Joint ids for the built-in skeleton.
mod joint {
    pub const ROOT: usize = 0;
    pub const SPINE: usize = 1;
    pub const ARM_L: usize = 2;
    pub const ARM_R: usize = 3;
    pub const LEG_L: usize = 4;
    pub const LEG_R: usize = 5;
}

fn rot_x(deg: f32) -> Quat {
    Quat::from_rotation_x(deg.to_radians())
}

fn rot_z(deg: f32) -> Quat {
    Quat::from_rotation_z(deg.to_radians())
}

impl AnimClip {
    /// A subtle idle: the root and spine sway gently. 2 animated joints.
    /// Duration 2.0s; sinusoidal keyframes so the loop is seamless.
    pub fn idle() -> AnimClip {
        let dur = 2.0;
        let mut clip = AnimClip::new("idle", dur, ANIM_JOINT_COUNT);
        let steps = 8;
        for s in 0..steps {
            let phase = s as f32 / steps as f32; // [0,1)
            let theta = phase * std::f32::consts::TAU;
            let mut pose = Pose::rest(ANIM_JOINT_COUNT);
            // Root bob up/down a few cm.
            pose.joints[joint::ROOT].translation = Vec3::new(0.0, 0.02 * theta.sin(), 0.0);
            // Spine sway about Z.
            pose.joints[joint::SPINE].rotation = rot_z(3.0 * theta.sin());
            clip.push(phase * dur, pose);
        }
        clip
    }

    /// A walk cycle: arms and legs swing in anti-phase, spine counter-rotates,
    /// root bobs at twice the stride rate. 5 animated joints.
    /// Duration 1.0s; sinusoidal keyframes for a seamless loop.
    pub fn walk() -> AnimClip {
        let dur = 1.0;
        let mut clip = AnimClip::new("walk", dur, ANIM_JOINT_COUNT);
        let steps = 12;
        for s in 0..steps {
            let phase = s as f32 / steps as f32;
            let theta = phase * std::f32::consts::TAU;
            let swing = 30.0 * theta.sin(); // degrees
            let mut pose = Pose::rest(ANIM_JOINT_COUNT);
            // Legs swing about X in anti-phase.
            pose.joints[joint::LEG_L].rotation = rot_x(swing);
            pose.joints[joint::LEG_R].rotation = rot_x(-swing);
            // Arms swing opposite their same-side leg (contralateral gait).
            pose.joints[joint::ARM_L].rotation = rot_x(-swing * 0.8);
            pose.joints[joint::ARM_R].rotation = rot_x(swing * 0.8);
            // Spine counter-rotates about Y at half amplitude.
            pose.joints[joint::SPINE].rotation =
                Quat::from_rotation_y((8.0 * theta.cos()).to_radians());
            // Root bobs vertically at twice stride rate.
            pose.joints[joint::ROOT].translation =
                Vec3::new(0.0, 0.03 * (2.0 * theta).cos().abs(), 0.0);
            clip.push(phase * dur, pose);
        }
        clip
    }
}

/// Build the standard idle↔walk locomotion blend tree.
///
/// Clip 0 = idle, clip 1 = walk. Parameter 0 = locomotion `speed01` in `[0,1]`
/// (0 → idle, 1 → walk). This is the tree the walking-sim NPC hookup consumes.
pub fn locomotion_blend_tree() -> BlendTree {
    let clips = vec![AnimClip::idle(), AnimClip::walk()];
    let root = BlendNode::Blend1D {
        param: 0,
        low: Box::new(BlendNode::Clip(0)),
        high: Box::new(BlendNode::Clip(1)),
    };
    BlendTree::new(clips, root, ANIM_JOINT_COUNT)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn quat_angle(q: Quat) -> f32 {
        // Angle of the rotation in degrees.
        2.0 * q.normalize().w.clamp(-1.0, 1.0).acos().to_degrees()
    }

    #[test]
    fn blend_tree_interpolates() {
        let tree = locomotion_blend_tree();
        let t = 0.3;

        let idle = tree.sample(t, &BlendParams::new(vec![0.0]));
        let walk = tree.sample(t, &BlendParams::new(vec![1.0]));
        let mid = tree.sample(t, &BlendParams::new(vec![0.5]));

        // Joints animated by either clip: spine(1), arms(2,3), legs(4,5), root(0).
        let animated = [
            joint::ROOT,
            joint::SPINE,
            joint::ARM_L,
            joint::ARM_R,
            joint::LEG_L,
            joint::LEG_R,
        ];

        let mut saw_distinct = false;
        for &j in &animated {
            let mi = &idle.joints[j];
            let mw = &walk.joints[j];
            let mm = &mid.joints[j];

            // Only assert on joints that actually differ between idle and walk.
            let idle_walk_diff = (mi.translation - mw.translation).length()
                + (1.0 - mi.rotation.normalize().dot(mw.rotation.normalize()).abs());
            if idle_walk_diff < 1e-4 {
                continue;
            }
            saw_distinct = true;

            // Mid must differ from BOTH endpoints.
            let d_idle = (mm.translation - mi.translation).length()
                + (1.0 - mm.rotation.normalize().dot(mi.rotation.normalize()).abs());
            let d_walk = (mm.translation - mw.translation).length()
                + (1.0 - mm.rotation.normalize().dot(mw.rotation.normalize()).abs());
            assert!(d_idle > 1e-5, "joint {j} mid equals idle (d={d_idle})");
            assert!(d_walk > 1e-5, "joint {j} mid equals walk (d={d_walk})");

            // Lerped translation components lie between endpoints (within fp slack).
            for axis in 0..3 {
                let a = mi.translation[axis];
                let b = mw.translation[axis];
                let m = mm.translation[axis];
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                assert!(
                    m >= lo - 1e-5 && m <= hi + 1e-5,
                    "joint {j} axis {axis}: {m} not between {lo} and {hi}"
                );
            }
        }
        assert!(saw_distinct, "no joint differed between idle and walk");
    }

    #[test]
    fn slerp_blend_is_45_degrees() {
        let a = JointPose { rotation: Quat::IDENTITY, ..JointPose::IDENTITY };
        let b = JointPose { rotation: rot_z(90.0), ..JointPose::IDENTITY };
        let mid = JointPose::blend(&a, &b, 0.5);
        let angle = quat_angle(mid.rotation);
        assert!((angle - 45.0).abs() < 0.5, "expected ~45deg, got {angle}");
    }

    #[test]
    fn clip_loops_exactly() {
        let walk = AnimClip::walk();
        let dur = walk.duration;
        for &t in &[0.0_f32, 0.17, 0.33, 0.5, 0.71, 0.93] {
            let a = walk.sample(t);
            let b = walk.sample(t + dur);
            for j in 0..walk.joint_count {
                let da = (a.joints[j].translation - b.joints[j].translation).length();
                let dr = 1.0 - a.joints[j].rotation.dot(b.joints[j].rotation).abs();
                assert!(da < 1e-5, "joint {j} translation loop mismatch at t={t}: {da}");
                assert!(dr < 1e-5, "joint {j} rotation loop mismatch at t={t}: {dr}");
            }
        }
    }

    #[test]
    fn skinning_single_joint_translation() {
        // Joint 1 translated by (1,0,0); a splat fully weighted to it moves by (1,0,0).
        let mut pose = Pose::rest(ANIM_JOINT_COUNT);
        pose.joints[1].translation = Vec3::new(1.0, 0.0, 0.0);

        let bind = Vec3::new(0.0, 0.0, 0.0);
        let skins = vec![SplatSkinData::single(1)];
        let mut splats = vec![make_splat(bind)];

        apply_pose(&pose, &skins, &mut splats, &[bind]);
        let p = Vec3::from(splats[0].position());
        assert!((p - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5, "got {p:?}");
    }

    #[test]
    fn skinning_two_joint_average() {
        // Joint 0 → +X, joint 2 → +Z; 50/50 splat moves to the average.
        let mut pose = Pose::rest(ANIM_JOINT_COUNT);
        pose.joints[0].translation = Vec3::new(2.0, 0.0, 0.0);
        pose.joints[2].translation = Vec3::new(0.0, 0.0, 4.0);

        let bind = Vec3::ZERO;
        let skins = vec![SplatSkinData::two(0, 0.5, 2, 0.5)];
        let mut splats = vec![make_splat(bind)];

        apply_pose(&pose, &skins, &mut splats, &[bind]);
        let p = Vec3::from(splats[0].position());
        let expected = Vec3::new(1.0, 0.0, 2.0);
        assert!((p - expected).length() < 1e-5, "got {p:?}, expected {expected:?}");
    }

    #[test]
    fn blend_tree_is_deterministic() {
        let tree = locomotion_blend_tree();
        let params = BlendParams::new(vec![0.37]);
        let a = tree.sample(0.42, &params);
        let b = tree.sample(0.42, &params);
        assert_eq!(a, b);
    }

    fn make_splat(pos: Vec3) -> GaussianSplat {
        GaussianSplat::surface(
            [pos.x, pos.y, pos.z],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.1,
            0.1,
            255,
            [0u16; 16],
        )
    }
}
