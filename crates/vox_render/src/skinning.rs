//! Linear-blend skinning for the **path-traced mesh** path (glTF skinned cims).
//!
//! This is distinct from the legacy [`crate::animation`] module, which binds
//! Gaussian splats rigidly to a single bone (`u8` ids, no inverse-bind pose, no
//! per-vertex weights). Skinned glTF characters (Mixamo cims) need true
//! 4-influence linear-blend skinning with inverse-bind matrices, and the output
//! is a [`crate::splat_backend::BlasDesc`] the path tracer consumes — none of
//! which the legacy module provides.
//!
//! Scale model (see plan R42): characters are NOT skinned per-frame per-cim.
//! [`bake_pose_blas`] samples the clip at one time and produces a static posed
//! BLAS; the caller bakes K snapshots once and instances them on the TLAS, so
//! ~1M cims cost K BLAS builds, not 1M.

use glam::{Mat4, Quat, Vec3};

/// A joint's local transform as translation / rotation / scale.
/// `r` is a quaternion in `xyzw` order (glam [`Quat`] convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformTRS {
    pub t: [f32; 3],
    pub r: [f32; 4],
    pub s: [f32; 3],
}

impl TransformTRS {
    pub const IDENTITY: Self = Self {
        t: [0.0, 0.0, 0.0],
        r: [0.0, 0.0, 0.0, 1.0],
        s: [1.0, 1.0, 1.0],
    };

    fn to_mat4(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from_array(self.s),
            Quat::from_array(self.r),
            Vec3::from_array(self.t),
        )
    }
}

/// A skeleton: joint hierarchy + bind pose + inverse-bind matrices.
///
/// All three vectors are parallel and indexed by contiguous joint index. The
/// hierarchy MUST be topologically ordered: every joint's parent index is `< its
/// own index` (or `-1` for a root). [`joint_matrices`] relies on this.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// Parent joint index, or `-1` for a root. `joint_parents[j] < j`.
    pub joint_parents: Vec<i32>,
    /// Inverse bind-pose matrix per joint (column-major 4×4).
    pub inverse_bind: Vec<[[f32; 4]; 4]>,
    /// Bind-pose local transform per joint (the default when a clip omits it).
    pub local_bind: Vec<TransformTRS>,
    /// Non-joint glTF transform chain between the nearest parent joint and this
    /// joint. A scaled `Armature` wrapper is semantically part of the rig even
    /// though it does not appear in a skin's joint array.
    pub local_prefix: Vec<TransformTRS>,
}

impl Skeleton {
    pub fn joint_count(&self) -> usize {
        self.joint_parents.len()
    }

    /// True iff every parent precedes its child (the contract for [`joint_matrices`]).
    pub fn is_topologically_ordered(&self) -> bool {
        self.joint_parents
            .iter()
            .enumerate()
            .all(|(j, &p)| p < 0 || (p as usize) < j)
    }
}

/// Sampled TRS keyframes for one joint. Each of `t`/`r`/`s` may be empty (that
/// channel is then taken from the skeleton's bind pose). When present, the
/// outer length matches `times`.
#[derive(Debug, Clone)]
pub struct JointChannel {
    pub joint: usize,
    pub times: Vec<f32>,
    pub t: Vec<[f32; 3]>,
    pub r: Vec<[f32; 4]>,
    pub s: Vec<[f32; 3]>,
}

/// An animation clip: one or more per-joint TRS channels over `duration` seconds.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub duration: f32,
    pub channels: Vec<JointChannel>,
}

/// A skinned triangle mesh: geometry + per-vertex 4-joint influences + skeleton.
/// `joints`/`weights` are parallel to `positions`; `weights` rows need not be
/// pre-normalized ([`skin_mesh`] renormalizes).
#[derive(Debug, Clone)]
pub struct SkinnedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub skeleton: Skeleton,
}

/// Find the keyframe bracket `[i, i+1]` around `time` and the lerp factor in it.
/// Clamps at both ends. Returns `(i0, i1, frac)`.
fn bracket(times: &[f32], time: f32) -> (usize, usize, f32) {
    if times.len() <= 1 {
        return (0, 0, 0.0);
    }
    if time <= times[0] {
        return (0, 0, 0.0);
    }
    if time >= times[times.len() - 1] {
        let last = times.len() - 1;
        return (last, last, 0.0);
    }
    for i in 0..times.len() - 1 {
        if times[i] <= time && time <= times[i + 1] {
            let span = times[i + 1] - times[i];
            let frac = if span > 0.0 {
                ((time - times[i]) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return (i, i + 1, frac);
        }
    }
    let last = times.len() - 1;
    (last, last, 0.0)
}

/// Sample every joint's local TRS at `time_s`. Returns one [`TransformTRS`] per
/// joint (length == `skel.joint_count()`). Joints without a channel keep their
/// bind-pose local transform; per-component (T/R/S) fallback to bind is honored.
pub fn sample_clip(clip: &AnimationClip, skel: &Skeleton, time_s: f32) -> Vec<TransformTRS> {
    let mut out: Vec<TransformTRS> = skel.local_bind.clone();
    debug_assert_eq!(
        out.len(),
        skel.joint_count(),
        "local_bind must cover all joints"
    );

    for ch in &clip.channels {
        if ch.joint >= out.len() {
            continue;
        }
        let pose = &mut out[ch.joint];

        if !ch.t.is_empty() {
            let (i0, i1, f) = bracket(&ch.times, time_s);
            let a = Vec3::from_array(ch.t[i0.min(ch.t.len() - 1)]);
            let b = Vec3::from_array(ch.t[i1.min(ch.t.len() - 1)]);
            pose.t = a.lerp(b, f).to_array();
        }
        if !ch.r.is_empty() {
            let (i0, i1, f) = bracket(&ch.times, time_s);
            let a = Quat::from_array(ch.r[i0.min(ch.r.len() - 1)]);
            let b = Quat::from_array(ch.r[i1.min(ch.r.len() - 1)]);
            pose.r = a.slerp(b, f).normalize().to_array();
        }
        if !ch.s.is_empty() {
            let (i0, i1, f) = bracket(&ch.times, time_s);
            let a = Vec3::from_array(ch.s[i0.min(ch.s.len() - 1)]);
            let b = Vec3::from_array(ch.s[i1.min(ch.s.len() - 1)]);
            pose.s = a.lerp(b, f).to_array();
        }
    }
    for (pose, prefix) in out.iter_mut().zip(&skel.local_prefix) {
        let composed = prefix.to_mat4() * pose.to_mat4();
        let (scale, rotation, translation) = composed.to_scale_rotation_translation();
        *pose = TransformTRS {
            t: translation.to_array(),
            r: rotation.to_array(),
            s: scale.to_array(),
        };
    }
    out
}

/// Compose per-joint world matrices from local TRS poses, walking the hierarchy.
/// Requires `skel.is_topologically_ordered()` (parents before children).
/// Returns column-major 4×4 arrays, one per joint.
pub fn joint_matrices(skel: &Skeleton, local: &[TransformTRS]) -> Vec<[[f32; 4]; 4]> {
    debug_assert!(
        skel.is_topologically_ordered(),
        "joint_matrices needs parent index < child index"
    );
    debug_assert_eq!(local.len(), skel.joint_count());

    let mut world: Vec<Mat4> = vec![Mat4::IDENTITY; skel.joint_count()];
    for j in 0..skel.joint_count() {
        let local_m = local[j].to_mat4();
        let parent = skel.joint_parents[j];
        world[j] = if parent < 0 {
            local_m
        } else {
            world[parent as usize] * local_m
        };
    }
    world.iter().map(|m| m.to_cols_array_2d()).collect()
}

/// Linear-blend skinning. `skin_mats[j] = world_joint[j] * inverse_bind[j]`.
/// Returns deformed `(positions, normals)` parallel to `mesh.positions`.
/// Per-vertex weights are renormalized; normals use the same blended matrix
/// (rotation/scale only) and are re-normalized.
pub fn skin_mesh(
    mesh: &SkinnedMesh,
    skin_mats: &[[[f32; 4]; 4]],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mats: Vec<Mat4> = skin_mats
        .iter()
        .map(|m| Mat4::from_cols_array_2d(m))
        .collect();
    let n = mesh.positions.len();
    let mut out_pos = Vec::with_capacity(n);
    let mut out_nrm = Vec::with_capacity(n);

    for v in 0..n {
        let p = Vec3::from_array(mesh.positions[v]);
        let nrm = mesh
            .normals
            .get(v)
            .map(|x| Vec3::from_array(*x))
            .unwrap_or(Vec3::Y);
        let joints = mesh.joints[v];
        let w = mesh.weights[v];
        let wsum = w[0] + w[1] + w[2] + w[3];
        let inv = if wsum > 1e-6 { 1.0 / wsum } else { 0.0 };

        // Weighted sum of the influencing skin matrices (standard LBS).
        let mut blended = Mat4::ZERO;
        let mut any = false;
        for k in 0..4 {
            let wk = w[k] * inv;
            if wk == 0.0 {
                continue;
            }
            if let Some(m) = mats.get(joints[k] as usize) {
                blended += *m * wk;
                any = true;
            }
        }
        let blended = if any { blended } else { Mat4::IDENTITY };

        let pp = blended.transform_point3(p);
        let nn = blended.transform_vector3(nrm).normalize_or_zero();
        out_pos.push(pp.to_array());
        out_nrm.push(if nn == Vec3::ZERO {
            [0.0, 1.0, 0.0]
        } else {
            nn.to_array()
        });
    }
    (out_pos, out_nrm)
}

/// Volume-preserving dual-quaternion skinning for articulated characters.
///
/// Unlike linear matrix blending, DQS does not collapse elbows, shoulders or
/// hips when neighbouring joints rotate far apart. Scale is deliberately not
/// blended: production character skeletons are rigid transforms and authored
/// body proportions live in their bind poses.
pub fn skin_mesh_dual_quat(
    mesh: &SkinnedMesh,
    skin_mats: &[[[f32; 4]; 4]],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let dual_quats = skin_mats
        .iter()
        .map(|matrix| {
            let matrix = Mat4::from_cols_array_2d(matrix);
            let (_, rotation, translation) = matrix.to_scale_rotation_translation();
            let rotation = rotation.normalize();
            let translation = Quat::from_xyzw(translation.x, translation.y, translation.z, 0.0);
            let dual = (translation * rotation) * 0.5;
            (rotation, dual)
        })
        .collect::<Vec<_>>();
    let mut out_pos = Vec::with_capacity(mesh.positions.len());
    let mut out_nrm = Vec::with_capacity(mesh.positions.len());
    for vertex in 0..mesh.positions.len() {
        let joints = mesh.joints[vertex];
        let weights = mesh.weights[vertex];
        let weight_sum: f32 = weights.iter().sum();
        let mut real = [0.0_f32; 4];
        let mut dual = [0.0_f32; 4];
        let mut reference = None;
        for influence in 0..4 {
            let weight = weights[influence] / weight_sum.max(1.0e-8);
            if weight <= 0.0 {
                continue;
            }
            let Some(&(rotation, translation)) = dual_quats.get(joints[influence] as usize) else {
                continue;
            };
            let reference = *reference.get_or_insert(rotation);
            let sign = if reference.dot(rotation) < 0.0 {
                -1.0
            } else {
                1.0
            };
            for axis in 0..4 {
                real[axis] += rotation.to_array()[axis] * weight * sign;
                dual[axis] += translation.to_array()[axis] * weight * sign;
            }
        }
        let real_length =
            (real[0] * real[0] + real[1] * real[1] + real[2] * real[2] + real[3] * real[3]).sqrt();
        let (rotation, translation) = if real_length > 1.0e-8 {
            for axis in 0..4 {
                real[axis] /= real_length;
                dual[axis] /= real_length;
            }
            let real_dual_dot = real
                .iter()
                .zip(dual)
                .map(|(real, dual)| real * dual)
                .sum::<f32>();
            for axis in 0..4 {
                dual[axis] -= real[axis] * real_dual_dot;
            }
            let rotation = Quat::from_array(real).normalize();
            let dual = Quat::from_array(dual);
            let translation_quat = (dual * rotation.conjugate()) * 2.0;
            (
                rotation,
                Vec3::new(translation_quat.x, translation_quat.y, translation_quat.z),
            )
        } else {
            (Quat::IDENTITY, Vec3::ZERO)
        };
        let position = rotation * Vec3::from_array(mesh.positions[vertex]) + translation;
        let normal = (rotation * Vec3::from_array(mesh.normals[vertex])).normalize_or_zero();
        out_pos.push(position.to_array());
        out_nrm.push(if normal == Vec3::ZERO {
            [0.0, 1.0, 0.0]
        } else {
            normal.to_array()
        });
    }
    (out_pos, out_nrm)
}

/// Deterministic pose-space surface corrective for a finite shared snapshot.
///
/// Skeletal skinning supplies the pose and this projected constraint solve
/// restores the finished mesh's authored local edge lengths. The attachment
/// term prevents drift away from the animation. It runs once per shared pose
/// BLAS at residency time, never per instance or per frame.
pub fn correct_skin_surface(
    mesh: &SkinnedMesh,
    posed: &[[f32; 3]],
    iterations: u16,
    attachment: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    if posed.len() != mesh.positions.len() || iterations == 0 {
        return (posed.to_vec(), mesh.normals.clone());
    }
    let attachment = attachment.clamp(0.0, 1.0);
    let authored = posed
        .iter()
        .copied()
        .map(Vec3::from_array)
        .collect::<Vec<_>>();
    let mut corrected = authored.clone();
    for _ in 0..iterations {
        // Deterministic Gauss-Seidel projection. Applying each constraint
        // immediately converges much faster than averaging one Jacobi update
        // over a high-valence character vertex, while triangle/index order
        // keeps the result replay-exact.
        for triangle in &mesh.indices {
            for (a, b) in [
                (triangle[0] as usize, triangle[1] as usize),
                (triangle[1] as usize, triangle[2] as usize),
                (triangle[2] as usize, triangle[0] as usize),
            ] {
                if a >= corrected.len() || b >= corrected.len() || a == b {
                    continue;
                }
                let rest = Vec3::from_array(mesh.positions[a])
                    .distance(Vec3::from_array(mesh.positions[b]));
                if rest <= 1.0e-7 {
                    continue;
                }
                let edge = corrected[b] - corrected[a];
                let length = edge.length();
                if length <= 1.0e-7 {
                    continue;
                }
                let correction = edge * ((length - rest) / length * 0.5);
                corrected[a] += correction;
                corrected[b] -= correction;
            }
        }
        for vertex in 0..corrected.len() {
            corrected[vertex] = corrected[vertex].lerp(authored[vertex], attachment);
        }
    }

    let mut normals = vec![Vec3::ZERO; corrected.len()];
    for triangle in &mesh.indices {
        let [a, b, c] = triangle.map(|index| index as usize);
        if a >= corrected.len() || b >= corrected.len() || c >= corrected.len() {
            continue;
        }
        let normal = (corrected[b] - corrected[a]).cross(corrected[c] - corrected[a]);
        normals[a] += normal;
        normals[b] += normal;
        normals[c] += normal;
    }
    let positions = corrected
        .into_iter()
        .map(|position| position.to_array())
        .collect();
    let normals = normals
        .into_iter()
        .map(|normal| {
            let normal = normal.normalize_or_zero();
            if normal == Vec3::ZERO {
                [0.0, 1.0, 0.0]
            } else {
                normal.to_array()
            }
        })
        .collect();
    (positions, normals)
}

/// Sample `clip` at `time_s`, skin `mesh`, and produce a render-ready posed
/// BLAS. Geometry stays indexed with smooth per-vertex normals (characters need
/// smooth shading — unlike the flat-normal building path). The returned
/// `material_ids` is empty (all triangles default to material 0; the caller
/// overrides per-instance material on the TLAS).
#[cfg(feature = "spectra-native")]
pub fn bake_pose_blas(
    mesh: &SkinnedMesh,
    clip: &AnimationClip,
    time_s: f32,
    proto_id: u64,
) -> crate::splat_backend::BlasDesc {
    let local = sample_clip(clip, &mesh.skeleton, time_s);
    let world = joint_matrices(&mesh.skeleton, &local);
    let skin_mats: Vec<[[f32; 4]; 4]> = (0..mesh.skeleton.joint_count())
        .map(|j| {
            let wm = Mat4::from_cols_array_2d(&world[j]);
            let ibm = Mat4::from_cols_array_2d(&mesh.skeleton.inverse_bind[j]);
            (wm * ibm).to_cols_array_2d()
        })
        .collect();

    // Preserve the authored skin deformation. A global edge-length projection
    // is not a character corrective: it couples unrelated triangles across the
    // entire body and can collapse shoulders, torsos, hands, and faces. DQS is
    // the runtime deformation contract; asset-specific correctives belong in
    // authored morphs, not a whole-surface solver.
    let (positions, normals) = skin_mesh_dual_quat(mesh, &skin_mats);

    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for a in 0..3 {
            aabb_min[a] = aabb_min[a].min(p[a]);
            aabb_max[a] = aabb_max[a].max(p[a]);
        }
    }
    if positions.is_empty() {
        aabb_min = [0.0; 3];
        aabb_max = [0.0; 3];
    }

    crate::splat_backend::BlasDesc {
        proto_id,
        positions,
        normals,
        uvs: mesh.uvs.clone(),
        indices: mesh.indices.clone(),
        material_ids: Vec::new(),
        aabb_min,
        aabb_max,
        weathering_masks: Vec::new(),
    }
}
