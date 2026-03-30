//! Skeletal skinning data — parallel array to GaussianSplat for multi-joint blend weights.

use glam::{Vec2, Vec3, Quat};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// SplatSkinData — 4-joint blend weights per splat
// ─────────────────────────────────────────────────────────────────────────────

/// Per-splat skinning data. Parallel array to GaussianSplat (not embedded in it).
/// joint_weights must sum to 1.0; use trailing zeros for < 4 influences.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplatSkinData {
    pub joint_indices: [u8; 4],
    pub joint_weights: [f32; 4],
}

impl SplatSkinData {
    /// Single-joint binding (1.0 weight on joint_index, rest 0).
    pub fn single(joint_index: u8) -> Self {
        Self { joint_indices: [joint_index, 0, 0, 0], joint_weights: [1.0, 0.0, 0.0, 0.0] }
    }

    /// Two-joint blend. Weights normalized to sum to 1.0.
    pub fn two(j0: u8, w0: f32, j1: u8, w1: f32) -> Self {
        let total = (w0 + w1).max(1e-6);
        Self {
            joint_indices: [j0, j1, 0, 0],
            joint_weights: [w0 / total, w1 / total, 0.0, 0.0],
        }
    }

    /// Normalize weights in-place so they sum to 1.0.
    pub fn normalize(&mut self) {
        let sum: f32 = self.joint_weights.iter().sum();
        if sum > 1e-6 {
            for w in &mut self.joint_weights { *w /= sum; }
        }
    }

    /// Sum of weights (should be ~1.0 after normalization).
    pub fn weight_sum(&self) -> f32 {
        self.joint_weights.iter().sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// B-Spline curve for animation compression
// ─────────────────────────────────────────────────────────────────────────────

pub trait BSplineValue: Clone {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self;
    fn zero() -> Self;
}

impl BSplineValue for Vec3 {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self { a.lerp(*b, t) }
    fn zero() -> Self { Vec3::ZERO }
}

impl BSplineValue for Quat {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self { a.slerp(*b, t).normalize() }
    fn zero() -> Self { Quat::IDENTITY }
}

impl BSplineValue for f32 {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self { a + (b - a) * t }
    fn zero() -> Self { 0.0 }
}

impl BSplineValue for Vec2 {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self { a.lerp(*b, t) }
    fn zero() -> Self { Vec2::ZERO }
}

/// Cubic B-spline curve for animation compression.
/// Stores control points and knot vector; evaluates via De Boor's algorithm (simplified to linear segments for degree 1, or cubic).
#[derive(Debug, Clone)]
pub struct BSpline<T: BSplineValue> {
    pub control_points: Vec<T>,
    pub knot_vector: Vec<f32>,
    pub degree: u8,
}

impl<T: BSplineValue> BSpline<T> {
    /// Create from control points with a uniform knot vector.
    pub fn uniform(control_points: Vec<T>, degree: u8) -> Self {
        let n = control_points.len();
        let knot_count = n + degree as usize + 1;
        let knot_vector: Vec<f32> = (0..knot_count)
            .map(|i| {
                let clamped = (i as f32 - degree as f32).max(0.0)
                    .min((n - degree as usize) as f32);
                clamped / (n - degree as usize) as f32
            })
            .collect();
        Self { control_points, knot_vector, degree }
    }

    /// Evaluate the curve at parameter t ∈ [0.0, 1.0].
    /// Uses linear interpolation between adjacent control points (degree-1 approximation).
    /// For production, replace with full De Boor recursion.
    pub fn sample(&self, t: f32) -> T {
        let t = t.clamp(0.0, 1.0);
        let n = self.control_points.len();
        if n == 0 { return T::zero(); }
        if n == 1 { return self.control_points[0].clone(); }

        // Find which segment [i, i+1] the parameter falls in
        let seg = (t * (n - 1) as f32).floor() as usize;
        let seg = seg.min(n - 2);
        let local_t = t * (n - 1) as f32 - seg as f32;

        T::lerp(&self.control_points[seg], &self.control_points[seg + 1], local_t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnimationClip — B-spline compressed animation
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct JointTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for JointTransform {
    fn default() -> Self {
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

#[derive(Debug, Clone)]
pub struct JointCurve {
    pub joint_index: u16,
    pub rotation_spline: BSpline<Quat>,
    pub translation_spline: BSpline<Vec3>,
    pub scale_spline: BSpline<Vec3>,
}

impl JointCurve {
    pub fn sample(&self, t: f32) -> JointTransform {
        JointTransform {
            translation: self.translation_spline.sample(t),
            rotation: self.rotation_spline.sample(t),
            scale: self.scale_spline.sample(t),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RootMotionCurve {
    pub translation_xz: BSpline<Vec2>,
    pub rotation_y: BSpline<f32>,
}

impl RootMotionCurve {
    pub fn sample(&self, t: f32) -> (Vec2, f32) {
        (self.translation_xz.sample(t), self.rotation_y.sample(t))
    }
}

#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration_secs: f32,
    pub joint_curves: Vec<JointCurve>,
    pub root_motion: Option<RootMotionCurve>,
}

impl AnimationClip {
    pub fn new(name: impl Into<String>, duration_secs: f32) -> Self {
        Self { name: name.into(), duration_secs, joint_curves: Vec::new(), root_motion: None }
    }

    /// Sample all joint transforms at time t (in seconds).
    /// Returns a Vec of (joint_index, JointTransform) pairs.
    pub fn sample(&self, t: f32) -> Vec<(u16, JointTransform)> {
        let t_norm = if self.duration_secs > 0.0 {
            (t / self.duration_secs).clamp(0.0, 1.0)
        } else { 0.0 };
        self.joint_curves.iter().map(|c| (c.joint_index, c.sample(t_norm))).collect()
    }

    /// Extract root motion from the root joint curve (joint_index 0), removing XZ translation and Y rotation.
    /// Returns a RootMotionCurve. Modifies joint_curves[root] to have identity XZ translation.
    pub fn extract_root_motion(&mut self) -> Option<RootMotionCurve> {
        let root_curve = self.joint_curves.iter_mut().find(|c| c.joint_index == 0)?;

        // Extract XZ from translation control points
        let xz_points: Vec<Vec2> = root_curve.translation_spline.control_points.iter()
            .map(|v| Vec2::new(v.x, v.z))
            .collect();
        let xz_spline = BSpline::uniform(xz_points, root_curve.translation_spline.degree);

        // Zero out XZ in translation
        for cp in &mut root_curve.translation_spline.control_points {
            cp.x = 0.0;
            cp.z = 0.0;
        }

        // Extract Y rotation from rotation control points
        let ry_points: Vec<f32> = root_curve.rotation_spline.control_points.iter()
            .map(|q| {
                // Extract Y-axis rotation angle from quaternion
                let (axis, angle) = q.to_axis_angle();
                if axis.y.abs() > 0.5 { angle * axis.y.signum() } else { 0.0 }
            })
            .collect();
        let ry_spline = BSpline::uniform(ry_points, root_curve.rotation_spline.degree);

        let rmc = RootMotionCurve { translation_xz: xz_spline, rotation_y: ry_spline };
        self.root_motion = Some(rmc.clone());
        Some(rmc)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SkeletonRetargeter
// ─────────────────────────────────────────────────────────────────────────────

pub type JointName = String;

#[derive(Debug, Clone)]
pub struct SkeletonPose {
    /// World-space joint transforms, one per joint in the skeleton.
    pub joint_transforms: Vec<JointTransform>,
}

impl SkeletonPose {
    pub fn new(count: usize) -> Self {
        Self { joint_transforms: vec![JointTransform::default(); count] }
    }
}

pub struct SkeletonRetargeter {
    pub source_joint_names: Vec<JointName>,
    pub target_joint_names: Vec<JointName>,
    /// Maps source joint name → target joint name.
    pub joint_map: HashMap<JointName, JointName>,
}

impl SkeletonRetargeter {
    pub fn new(
        source_joint_names: Vec<JointName>,
        target_joint_names: Vec<JointName>,
    ) -> Self {
        Self { source_joint_names, target_joint_names, joint_map: HashMap::new() }
    }

    /// Add a joint mapping: source_name drives target_name.
    pub fn map(&mut self, source: impl Into<String>, target: impl Into<String>) {
        self.joint_map.insert(source.into(), target.into());
    }

    /// Retarget a source pose to target skeleton.
    /// Unmapped target joints remain in their default (bind) pose.
    pub fn retarget_pose(&self, source_pose: &SkeletonPose) -> SkeletonPose {
        let mut target_pose = SkeletonPose::new(self.target_joint_names.len());

        for (src_idx, src_name) in self.source_joint_names.iter().enumerate() {
            if let Some(tgt_name) = self.joint_map.get(src_name)
                && let Some(tgt_idx) = self.target_joint_names.iter().position(|n| n == tgt_name)
                && src_idx < source_pose.joint_transforms.len()
            {
                target_pose.joint_transforms[tgt_idx] =
                    source_pose.joint_transforms[src_idx].clone();
            }
        }

        target_pose
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec2, Quat};

    #[test]
    fn splat_skin_data_single_weight_sums_to_one() {
        let skin = SplatSkinData::single(3);
        assert!((skin.weight_sum() - 1.0).abs() < 1e-6);
        assert_eq!(skin.joint_indices[0], 3);
    }

    #[test]
    fn splat_skin_data_two_normalizes() {
        // Provide unequal weights that don't sum to 1
        let skin = SplatSkinData::two(0, 2.0, 1, 6.0);
        assert!((skin.weight_sum() - 1.0).abs() < 1e-6);
        assert!((skin.joint_weights[0] - 0.25).abs() < 1e-6);
        assert!((skin.joint_weights[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn bspline_sample_endpoints() {
        let spline = BSpline::uniform(vec![Vec3::ZERO, Vec3::ONE * 5.0], 1);
        let start = spline.sample(0.0);
        let end = spline.sample(1.0);
        assert!(start.length() < 1e-5, "start should be near ZERO, got {:?}", start);
        assert!((end - Vec3::ONE * 5.0).length() < 1e-5, "end should be near ONE*5, got {:?}", end);
    }

    #[test]
    fn bspline_sample_midpoint() {
        let spline = BSpline::uniform(vec![Vec3::ZERO, Vec3::ONE * 5.0], 1);
        let mid = spline.sample(0.5);
        let expected = Vec3::ONE * 2.5;
        assert!((mid - expected).length() < 1e-5, "midpoint should be {:?}, got {:?}", expected, mid);
    }

    #[test]
    fn animation_clip_sample_range() {
        let rot_spline = BSpline::uniform(vec![Quat::IDENTITY, Quat::IDENTITY], 1);
        let trans_spline = BSpline::uniform(vec![Vec3::ZERO, Vec3::X], 1);
        let scale_spline = BSpline::uniform(vec![Vec3::ONE, Vec3::ONE], 1);

        let curve = JointCurve {
            joint_index: 0,
            rotation_spline: rot_spline,
            translation_spline: trans_spline,
            scale_spline,
        };

        let mut clip = AnimationClip::new("test", 1.0);
        clip.joint_curves.push(curve);

        // Must not panic
        let _s0 = clip.sample(0.0);
        let _s05 = clip.sample(0.5);
        let _s1 = clip.sample(1.0);
    }

    #[test]
    fn retargeter_maps_joint() {
        let source_names = vec!["root".to_string(), "spine".to_string()];
        let target_names = vec!["pelvis".to_string(), "torso".to_string()];

        let mut retargeter = SkeletonRetargeter::new(source_names, target_names);
        retargeter.map("root", "pelvis");

        let mut source_pose = SkeletonPose::new(2);
        source_pose.joint_transforms[0].translation = Vec3::new(1.0, 2.0, 3.0);

        let target_pose = retargeter.retarget_pose(&source_pose);

        let pelvis_idx = retargeter.target_joint_names.iter().position(|n| n == "pelvis").unwrap();
        assert!((target_pose.joint_transforms[pelvis_idx].translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn joint_curve_sample() {
        let first_trans = Vec3::new(10.0, 20.0, 30.0);
        let first_rot = Quat::IDENTITY;
        let first_scale = Vec3::new(2.0, 2.0, 2.0);

        let curve = JointCurve {
            joint_index: 1,
            rotation_spline: BSpline::uniform(vec![first_rot, Quat::IDENTITY], 1),
            translation_spline: BSpline::uniform(vec![first_trans, Vec3::ZERO], 1),
            scale_spline: BSpline::uniform(vec![first_scale, Vec3::ONE], 1),
        };

        let xform = curve.sample(0.0);
        assert!((xform.translation - first_trans).length() < 1e-5);
        assert!((xform.scale - first_scale).length() < 1e-5);
    }
}
