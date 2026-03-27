//! GLTF skeletal animation — extract skeletons & animations, evaluate poses, skin splats.
//!
//! This module bridges GLTF skeletal data to the Gaussian splat pipeline:
//! bone hierarchies drive splat positions instead of traditional vertex skinning.

use std::path::Path;

use glam::{Mat4, Quat, Vec3, Vec4};
use vox_core::types::GaussianSplat;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A skeleton extracted from a GLTF file.
#[derive(Debug, Clone)]
pub struct GltfSkeleton {
    pub joints: Vec<GltfJoint>,
    pub root_joints: Vec<usize>,
}

/// A single joint (bone) in a skeleton.
#[derive(Debug, Clone)]
pub struct GltfJoint {
    pub name: String,
    pub index: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub inverse_bind_matrix: Mat4,
    pub local_transform: Mat4,
}

/// An animation clip extracted from a GLTF file.
#[derive(Debug, Clone)]
pub struct GltfAnimation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

/// One animated property on one joint.
#[derive(Debug, Clone)]
pub struct AnimationChannel {
    pub joint_index: usize,
    pub property: AnimationProperty,
    pub keyframes: Vec<Keyframe>,
}

/// Which transform component a channel animates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
}

/// A single keyframe sample.
#[derive(Debug, Clone, Copy)]
pub struct Keyframe {
    pub time: f32,
    /// translation xyz+0, rotation xyzw, scale xyz+0
    pub value: [f32; 4],
}

// ---------------------------------------------------------------------------
// Extraction from GLTF
// ---------------------------------------------------------------------------

/// Extract skeleton and animations from a GLTF file on disk.
pub fn extract_skeleton(path: &Path) -> Result<(GltfSkeleton, Vec<GltfAnimation>), String> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| format!("GLTF import error: {e}"))?;

    // --- skeleton -----------------------------------------------------------
    let skin = document
        .skins()
        .next()
        .ok_or_else(|| "No skin found in GLTF file".to_string())?;

    let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
    let joint_count = joint_nodes.len();

    // Build node-index -> joint-index map
    let mut node_to_joint: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (ji, node) in joint_nodes.iter().enumerate() {
        node_to_joint.insert(node.index(), ji);
    }

    // Read inverse bind matrices
    let ibms: Vec<Mat4> = if let Some(accessor) = skin.inverse_bind_matrices() {
        let reader = accessor.clone();
        let view = reader
            .view()
            .ok_or_else(|| "No buffer view for IBM accessor".to_string())?;
        let buf = &buffers[view.buffer().index()];
        let offset = view.offset() + accessor.offset();
        let stride = view.stride().unwrap_or(64);
        (0..joint_count)
            .map(|i| {
                let start = offset + i * stride;
                let cols: [[f32; 4]; 4] = std::array::from_fn(|c| {
                    std::array::from_fn(|r| {
                        let idx = start + (c * 4 + r) * 4;
                        f32::from_le_bytes([buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]])
                    })
                });
                Mat4::from_cols_array_2d(&cols)
            })
            .collect()
    } else {
        vec![Mat4::IDENTITY; joint_count]
    };

    // Build joints
    let mut joints: Vec<GltfJoint> = Vec::with_capacity(joint_count);
    for (ji, node) in joint_nodes.iter().enumerate() {
        let (t, r, s) = node.transform().decomposed();
        let local = Mat4::from_scale_rotation_translation(
            Vec3::from(s),
            Quat::from_array(r),
            Vec3::from(t),
        );

        // Find parent among joint nodes
        let parent = find_parent_joint(node, &node_to_joint, &document);

        let children: Vec<usize> = node
            .children()
            .filter_map(|c| node_to_joint.get(&c.index()).copied())
            .collect();

        joints.push(GltfJoint {
            name: node.name().unwrap_or("unnamed").to_string(),
            index: ji,
            parent,
            children,
            inverse_bind_matrix: ibms[ji],
            local_transform: local,
        });
    }

    let root_joints: Vec<usize> = joints
        .iter()
        .filter(|j| j.parent.is_none())
        .map(|j| j.index)
        .collect();

    let skeleton = GltfSkeleton { joints, root_joints };

    // --- animations ---------------------------------------------------------
    let animations: Vec<GltfAnimation> = document
        .animations()
        .map(|anim| extract_one_animation(&anim, &buffers, &node_to_joint))
        .collect();

    Ok((skeleton, animations))
}

/// Walk up the node tree to find the closest ancestor that is also a joint.
fn find_parent_joint(
    node: &gltf::Node,
    node_to_joint: &std::collections::HashMap<usize, usize>,
    document: &gltf::Document,
) -> Option<usize> {
    // GLTF nodes don't expose a direct parent pointer; we search all scenes.
    let target = node.index();
    for scene in document.scenes() {
        for root in scene.nodes() {
            if let Some(ji) = walk_for_parent(&root, target, node_to_joint, None) {
                return Some(ji);
            }
        }
    }
    None
}

fn walk_for_parent(
    current: &gltf::Node,
    target: usize,
    map: &std::collections::HashMap<usize, usize>,
    parent_joint: Option<usize>,
) -> Option<usize> {
    if current.index() == target {
        return Some(parent_joint?);
    }
    let next_parent = map.get(&current.index()).copied().or(parent_joint);
    for child in current.children() {
        if let Some(found) = walk_for_parent(&child, target, map, next_parent) {
            return Some(found);
        }
    }
    None
}

fn extract_one_animation(
    anim: &gltf::Animation,
    buffers: &[gltf::buffer::Data],
    node_to_joint: &std::collections::HashMap<usize, usize>,
) -> GltfAnimation {
    let mut channels = Vec::new();
    let mut duration: f32 = 0.0;

    for channel in anim.channels() {
        let target = channel.target();
        let joint_index = match node_to_joint.get(&target.node().index()) {
            Some(&ji) => ji,
            None => continue, // channel targets a non-joint node
        };

        let property = match target.property() {
            gltf::animation::Property::Translation => AnimationProperty::Translation,
            gltf::animation::Property::Rotation => AnimationProperty::Rotation,
            gltf::animation::Property::Scale => AnimationProperty::Scale,
            _ => continue,
        };

        let reader = channel.reader(|buf| Some(&buffers[buf.index()]));

        let times: Vec<f32> = reader
            .read_inputs()
            .map(|iter| iter.collect())
            .unwrap_or_default();

        let values: Vec<[f32; 4]> = match property {
            AnimationProperty::Translation => reader
                .read_outputs()
                .map(|out| match out {
                    gltf::animation::util::ReadOutputs::Translations(iter) => iter
                        .map(|t| [t[0], t[1], t[2], 0.0])
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
            AnimationProperty::Rotation => reader
                .read_outputs()
                .map(|out| match out {
                    gltf::animation::util::ReadOutputs::Rotations(iter) => {
                        iter.into_f32().map(|r| r).collect::<Vec<_>>()
                    }
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
            AnimationProperty::Scale => reader
                .read_outputs()
                .map(|out| match out {
                    gltf::animation::util::ReadOutputs::Scales(iter) => iter
                        .map(|s| [s[0], s[1], s[2], 0.0])
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
        };

        if let Some(&last_t) = times.last() {
            duration = duration.max(last_t);
        }

        let keyframes: Vec<Keyframe> = times
            .iter()
            .zip(values.iter())
            .map(|(&time, &value)| Keyframe { time, value })
            .collect();

        channels.push(AnimationChannel {
            joint_index,
            property,
            keyframes,
        });
    }

    GltfAnimation {
        name: anim.name().unwrap_or("unnamed").to_string(),
        duration,
        channels,
    }
}

// ---------------------------------------------------------------------------
// Animation evaluation
// ---------------------------------------------------------------------------

/// Compute world-space joint transforms for a given animation time.
///
/// Returns one `Mat4` per joint. The animation channels override the skeleton's
/// bind-pose local transforms, then the hierarchy is walked to produce world
/// transforms.
pub fn evaluate_animation(
    skeleton: &GltfSkeleton,
    animation: &GltfAnimation,
    time: f32,
) -> Vec<Mat4> {
    let joint_count = skeleton.joints.len();

    // Start with bind-pose local transforms
    let mut local_transforms: Vec<Mat4> = skeleton
        .joints
        .iter()
        .map(|j| j.local_transform)
        .collect();

    // Per-joint animated TRS, seeded from the bind pose decomposition
    let mut translations: Vec<Option<Vec3>> = vec![None; joint_count];
    let mut rotations: Vec<Option<Quat>> = vec![None; joint_count];
    let mut scales: Vec<Option<Vec3>> = vec![None; joint_count];

    for channel in &animation.channels {
        let ji = channel.joint_index;
        if ji >= joint_count {
            continue;
        }
        let v = interpolate_keyframes(&channel.keyframes, time, channel.property);
        match channel.property {
            AnimationProperty::Translation => {
                translations[ji] = Some(Vec3::new(v[0], v[1], v[2]));
            }
            AnimationProperty::Rotation => {
                rotations[ji] = Some(Quat::from_xyzw(v[0], v[1], v[2], v[3]).normalize());
            }
            AnimationProperty::Scale => {
                scales[ji] = Some(Vec3::new(v[0], v[1], v[2]));
            }
        }
    }

    // Compose local transforms from animated components where available
    for ji in 0..joint_count {
        if translations[ji].is_some() || rotations[ji].is_some() || scales[ji].is_some() {
            let (bind_s, bind_r, bind_t) =
                local_transforms[ji].to_scale_rotation_translation();
            let t = translations[ji].unwrap_or(bind_t);
            let r = rotations[ji].unwrap_or(bind_r);
            let s = scales[ji].unwrap_or(bind_s);
            local_transforms[ji] = Mat4::from_scale_rotation_translation(s, r, t);
        }
    }

    // Walk hierarchy to compute world transforms
    let mut world_transforms = vec![Mat4::IDENTITY; joint_count];
    for &root in &skeleton.root_joints {
        compute_world_recursive(root, Mat4::IDENTITY, &local_transforms, &skeleton.joints, &mut world_transforms);
    }

    world_transforms
}

fn compute_world_recursive(
    ji: usize,
    parent_world: Mat4,
    locals: &[Mat4],
    joints: &[GltfJoint],
    world: &mut [Mat4],
) {
    world[ji] = parent_world * locals[ji];
    for &child in &joints[ji].children {
        compute_world_recursive(child, world[ji], locals, joints, world);
    }
}

/// Interpolate keyframes at the given time.
fn interpolate_keyframes(
    keyframes: &[Keyframe],
    time: f32,
    property: AnimationProperty,
) -> [f32; 4] {
    if keyframes.is_empty() {
        return match property {
            AnimationProperty::Translation => [0.0, 0.0, 0.0, 0.0],
            AnimationProperty::Rotation => [0.0, 0.0, 0.0, 1.0],
            AnimationProperty::Scale => [1.0, 1.0, 1.0, 0.0],
        };
    }

    if keyframes.len() == 1 || time <= keyframes[0].time {
        return keyframes[0].value;
    }

    if time >= keyframes.last().unwrap().time {
        return keyframes.last().unwrap().value;
    }

    // Find surrounding keyframes
    let mut i = 0;
    while i < keyframes.len() - 1 && keyframes[i + 1].time < time {
        i += 1;
    }

    let kf0 = &keyframes[i];
    let kf1 = &keyframes[i + 1];
    let dt = kf1.time - kf0.time;
    let t = if dt > 1e-8 {
        ((time - kf0.time) / dt).clamp(0.0, 1.0)
    } else {
        0.0
    };

    match property {
        AnimationProperty::Rotation => {
            let q0 = Quat::from_xyzw(kf0.value[0], kf0.value[1], kf0.value[2], kf0.value[3]);
            let q1 = Quat::from_xyzw(kf1.value[0], kf1.value[1], kf1.value[2], kf1.value[3]);
            let result = q0.slerp(q1, t);
            [result.x, result.y, result.z, result.w]
        }
        AnimationProperty::Translation | AnimationProperty::Scale => {
            let a = Vec4::from(kf0.value);
            let b = Vec4::from(kf1.value);
            let result = a.lerp(b, t);
            result.to_array()
        }
    }
}

// ---------------------------------------------------------------------------
// Skinning
// ---------------------------------------------------------------------------

/// Skin Gaussian splats according to joint transforms.
///
/// For each splat, applies `joint_transform * inverse_bind * local_pos`.
/// `joint_bindings[i]` maps splat `i` to a joint index.
pub fn skin_splats(
    splats: &[GaussianSplat],
    joint_bindings: &[usize],
    joint_transforms: &[Mat4],
    inverse_bind_matrices: &[Mat4],
) -> Vec<GaussianSplat> {
    assert_eq!(splats.len(), joint_bindings.len());

    splats
        .iter()
        .zip(joint_bindings.iter())
        .map(|(splat, &ji)| {
            let ibm = inverse_bind_matrices
                .get(ji)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let world = joint_transforms
                .get(ji)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let skin_mat = world * ibm;

            let pos = Vec3::from(splat.position);
            let new_pos = skin_mat.transform_point3(pos);

            GaussianSplat {
                position: [new_pos.x, new_pos.y, new_pos.z],
                ..*splat
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Synthetic skeleton builder (for tests and procedural content)
// ---------------------------------------------------------------------------

/// Build a synthetic skeleton for testing.
///
/// Creates a linear chain of joints along the Y axis, each 1 unit apart.
pub fn build_synthetic_skeleton(joint_names: &[&str]) -> GltfSkeleton {
    let count = joint_names.len();
    let mut joints = Vec::with_capacity(count);

    for (i, &name) in joint_names.iter().enumerate() {
        let y_offset = i as f32;
        let local_translation = if i == 0 {
            Vec3::new(0.0, y_offset, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0) // each child 1 unit above parent
        };
        let local_transform =
            Mat4::from_translation(local_translation);

        // Inverse bind matrix: inverse of the world-space bind pose
        let world_y = y_offset;
        let inverse_bind_matrix = Mat4::from_translation(Vec3::new(0.0, -world_y, 0.0));

        let parent = if i == 0 { None } else { Some(i - 1) };
        let children = if i + 1 < count { vec![i + 1] } else { vec![] };

        joints.push(GltfJoint {
            name: name.to_string(),
            index: i,
            parent,
            children,
            inverse_bind_matrix,
            local_transform,
        });
    }

    let root_joints = if count > 0 { vec![0] } else { vec![] };

    GltfSkeleton { joints, root_joints }
}

/// Build a simple rotation animation on a single joint.
pub fn build_synthetic_animation(
    name: &str,
    joint_index: usize,
    duration: f32,
    start_rotation: Quat,
    end_rotation: Quat,
) -> GltfAnimation {
    GltfAnimation {
        name: name.to_string(),
        duration,
        channels: vec![AnimationChannel {
            joint_index,
            property: AnimationProperty::Rotation,
            keyframes: vec![
                Keyframe {
                    time: 0.0,
                    value: [
                        start_rotation.x,
                        start_rotation.y,
                        start_rotation.z,
                        start_rotation.w,
                    ],
                },
                Keyframe {
                    time: duration,
                    value: [
                        end_rotation.x,
                        end_rotation.y,
                        end_rotation.z,
                        end_rotation.w,
                    ],
                },
            ],
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_synthetic_skeleton_structure() {
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        assert_eq!(skel.joints.len(), 3);
        assert_eq!(skel.root_joints, vec![0]);

        assert!(skel.joints[0].parent.is_none());
        assert_eq!(skel.joints[1].parent, Some(0));
        assert_eq!(skel.joints[2].parent, Some(1));

        assert_eq!(skel.joints[0].children, vec![1]);
        assert_eq!(skel.joints[1].children, vec![2]);
        assert!(skel.joints[2].children.is_empty());
    }

    #[test]
    fn test_bind_pose_at_t0() {
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        let anim = build_synthetic_animation(
            "wave",
            1, // animate arm
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_z(FRAC_PI_2),
        );

        let transforms = evaluate_animation(&skel, &anim, 0.0);
        assert_eq!(transforms.len(), 3);

        // At t=0 the arm has identity rotation, so hand should be at (0, 2, 0)
        let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
        assert!(
            approx_eq(hand_pos.x, 0.0, 0.01)
                && approx_eq(hand_pos.y, 2.0, 0.01)
                && approx_eq(hand_pos.z, 0.0, 0.01),
            "hand at t=0 should be at (0,2,0), got ({}, {}, {})",
            hand_pos.x,
            hand_pos.y,
            hand_pos.z,
        );
    }

    #[test]
    fn test_interpolated_pose_at_half() {
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        let anim = build_synthetic_animation(
            "wave",
            1, // animate arm
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_z(FRAC_PI_2),
        );

        let transforms = evaluate_animation(&skel, &anim, 0.5);

        // Arm should be rotated ~45 degrees around Z.
        // Hand is 1 unit above arm in local space, so in world:
        // arm world pos is (0,1,0); hand offset rotated 45 deg: ~(0.707, 0.707, 0)
        let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
        // With arm at (0,1,0) and hand 1 unit offset rotated 45 deg:
        // expected ~ (-0.707, 1.707, 0)
        let expected_x = -std::f32::consts::FRAC_1_SQRT_2;
        let expected_y = 1.0 + std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            approx_eq(hand_pos.x, expected_x, 0.05)
                && approx_eq(hand_pos.y, expected_y, 0.05)
                && approx_eq(hand_pos.z, 0.0, 0.01),
            "hand at t=0.5 should be near ({}, {}, 0), got ({}, {}, {})",
            expected_x,
            expected_y,
            hand_pos.x,
            hand_pos.y,
            hand_pos.z,
        );
    }

    #[test]
    fn test_full_rotation_at_t1() {
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        let anim = build_synthetic_animation(
            "wave",
            1,
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_z(FRAC_PI_2),
        );

        let transforms = evaluate_animation(&skel, &anim, 1.0);

        // Arm rotated 90 deg around Z => hand offset (0,1,0) becomes (-1,0,0)
        // arm is at (0,1,0), hand world = (0,1,0) + (-1,0,0) = (-1,1,0)
        let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
        assert!(
            approx_eq(hand_pos.x, -1.0, 0.01)
                && approx_eq(hand_pos.y, 1.0, 0.01)
                && approx_eq(hand_pos.z, 0.0, 0.01),
            "hand at t=1 should be at (-1,1,0), got ({}, {}, {})",
            hand_pos.x,
            hand_pos.y,
            hand_pos.z,
        );
    }

    #[test]
    fn test_skin_splats_moves_bound_splats() {
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        let anim = build_synthetic_animation(
            "wave",
            1,
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_z(FRAC_PI_2),
        );

        // Place a splat at the hand's bind-pose world position (0, 2, 0)
        let splat = GaussianSplat {
            position: [0.0, 2.0, 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        };

        let joint_transforms = evaluate_animation(&skel, &anim, 1.0);
        let ibms: Vec<Mat4> = skel.joints.iter().map(|j| j.inverse_bind_matrix).collect();

        let skinned = skin_splats(&[splat], &[2], &joint_transforms, &ibms);
        assert_eq!(skinned.len(), 1);

        // The splat was at hand bind position (0,2,0). After skinning:
        // skin_mat = world_transform[hand] * ibm[hand]
        // ibm[hand] brings (0,2,0) -> (0,0,0) (hand local origin)
        // world_transform[hand] puts it at (-1,1,0)
        let p = skinned[0].position;
        assert!(
            approx_eq(p[0], -1.0, 0.05)
                && approx_eq(p[1], 1.0, 0.05)
                && approx_eq(p[2], 0.0, 0.01),
            "skinned splat should be at (-1,1,0), got ({}, {}, {})",
            p[0],
            p[1],
            p[2],
        );
    }

    #[test]
    fn test_hierarchy_propagation() {
        // Rotate root 90 deg — all children should move
        let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
        let anim = build_synthetic_animation(
            "spin",
            0, // animate ROOT
            1.0,
            Quat::IDENTITY,
            Quat::from_rotation_z(FRAC_PI_2),
        );

        let transforms = evaluate_animation(&skel, &anim, 1.0);

        // Root at origin, rotated 90 deg Z
        // Arm local offset (0,1,0) rotated => (-1,0,0), arm world = (-1,0,0)
        let arm_pos = transforms[1].transform_point3(Vec3::ZERO);
        assert!(
            approx_eq(arm_pos.x, -1.0, 0.01)
                && approx_eq(arm_pos.y, 0.0, 0.01),
            "arm world pos should be (-1,0,0) when root rotates 90 Z, got ({}, {}, {})",
            arm_pos.x,
            arm_pos.y,
            arm_pos.z,
        );

        // Hand local offset another (0,1,0) => after root rotation, (-2,0,0)
        let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
        assert!(
            approx_eq(hand_pos.x, -2.0, 0.01)
                && approx_eq(hand_pos.y, 0.0, 0.01),
            "hand world pos should be (-2,0,0), got ({}, {}, {})",
            hand_pos.x,
            hand_pos.y,
            hand_pos.z,
        );
    }

    #[test]
    fn test_interpolate_keyframes_empty() {
        let result = interpolate_keyframes(&[], 0.5, AnimationProperty::Translation);
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0]);

        let result = interpolate_keyframes(&[], 0.5, AnimationProperty::Rotation);
        assert_eq!(result, [0.0, 0.0, 0.0, 1.0]);

        let result = interpolate_keyframes(&[], 0.5, AnimationProperty::Scale);
        assert_eq!(result, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_interpolate_keyframes_single() {
        let kf = vec![Keyframe {
            time: 0.0,
            value: [1.0, 2.0, 3.0, 0.0],
        }];
        let result = interpolate_keyframes(&kf, 5.0, AnimationProperty::Translation);
        assert_eq!(result, [1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_interpolate_keyframes_clamping() {
        let kf = vec![
            Keyframe { time: 1.0, value: [0.0, 0.0, 0.0, 0.0] },
            Keyframe { time: 2.0, value: [10.0, 10.0, 10.0, 0.0] },
        ];
        // Before first keyframe
        let r = interpolate_keyframes(&kf, 0.0, AnimationProperty::Translation);
        assert_eq!(r, [0.0, 0.0, 0.0, 0.0]);
        // After last keyframe
        let r = interpolate_keyframes(&kf, 99.0, AnimationProperty::Translation);
        assert_eq!(r, [10.0, 10.0, 10.0, 0.0]);
    }
}
