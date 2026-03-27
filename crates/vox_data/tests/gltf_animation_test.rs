//! Integration tests for GLTF skeletal animation.

use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Quat, Vec3};
use vox_core::types::GaussianSplat;
use vox_data::gltf_animation::*;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

#[test]
fn test_extract_synthetic_skeleton() {
    let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
    assert_eq!(skel.joints.len(), 3);
    assert_eq!(skel.root_joints, vec![0]);
    assert_eq!(skel.joints[0].name, "root");
    assert_eq!(skel.joints[1].name, "arm");
    assert_eq!(skel.joints[2].name, "hand");

    assert!(skel.joints[0].parent.is_none());
    assert_eq!(skel.joints[1].parent, Some(0));
    assert_eq!(skel.joints[2].parent, Some(1));
}

#[test]
fn test_evaluate_bind_pose_at_t0() {
    let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
    let anim = build_synthetic_animation("wave", 1, 1.0, Quat::IDENTITY, Quat::from_rotation_z(FRAC_PI_2));

    let transforms = evaluate_animation(&skel, &anim, 0.0);
    assert_eq!(transforms.len(), 3);

    // root at origin
    let root_pos = transforms[0].transform_point3(Vec3::ZERO);
    assert!(approx_eq(root_pos.y, 0.0, 0.01));

    // arm at y=1
    let arm_pos = transforms[1].transform_point3(Vec3::ZERO);
    assert!(approx_eq(arm_pos.y, 1.0, 0.01));

    // hand at y=2
    let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
    assert!(approx_eq(hand_pos.y, 2.0, 0.01));
}

#[test]
fn test_evaluate_interpolated_at_half() {
    let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
    let anim = build_synthetic_animation("wave", 1, 1.0, Quat::IDENTITY, Quat::from_rotation_z(FRAC_PI_2));

    let transforms = evaluate_animation(&skel, &anim, 0.5);

    // Arm rotated 45 deg => hand offset (0,1,0) goes to (-sin45, cos45, 0)
    let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
    let expected_x = -std::f32::consts::FRAC_1_SQRT_2;
    let expected_y = 1.0 + std::f32::consts::FRAC_1_SQRT_2;
    assert!(
        approx_eq(hand_pos.x, expected_x, 0.05) && approx_eq(hand_pos.y, expected_y, 0.05),
        "hand at t=0.5: expected ({}, {}), got ({}, {})",
        expected_x, expected_y, hand_pos.x, hand_pos.y,
    );
}

#[test]
fn test_skin_splats_transforms_positions() {
    let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
    let anim = build_synthetic_animation("wave", 1, 1.0, Quat::IDENTITY, Quat::from_rotation_z(FRAC_PI_2));

    let splat = GaussianSplat {
        position: [0.0, 2.0, 0.0],
        scale: [0.05, 0.05, 0.05],
        rotation: [0, 0, 0, 32767],
        opacity: 200,
        _pad: [0; 3],
        spectral: [0; 8],
    };

    let joint_transforms = evaluate_animation(&skel, &anim, 1.0);
    let ibms: Vec<Mat4> = skel.joints.iter().map(|j| j.inverse_bind_matrix).collect();

    let skinned = skin_splats(&[splat], &[2], &joint_transforms, &ibms);
    assert_eq!(skinned.len(), 1);

    let p = skinned[0].position;
    assert!(
        approx_eq(p[0], -1.0, 0.05) && approx_eq(p[1], 1.0, 0.05) && approx_eq(p[2], 0.0, 0.01),
        "skinned position: expected (-1, 1, 0), got ({}, {}, {})", p[0], p[1], p[2],
    );

    // Non-position fields should be preserved
    assert_eq!(skinned[0].opacity, 200);
    assert_eq!(skinned[0].scale, [0.05, 0.05, 0.05]);
}

#[test]
fn test_hierarchy_propagation_root_rotation() {
    let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
    let anim = build_synthetic_animation("spin", 0, 1.0, Quat::IDENTITY, Quat::from_rotation_z(FRAC_PI_2));

    let transforms = evaluate_animation(&skel, &anim, 1.0);

    // Root rotated 90 deg Z at origin
    // arm: local (0,1,0) -> rotated (-1,0,0)
    let arm_pos = transforms[1].transform_point3(Vec3::ZERO);
    assert!(approx_eq(arm_pos.x, -1.0, 0.01) && approx_eq(arm_pos.y, 0.0, 0.01));

    // hand: another (0,1,0) offset, also rotated -> (-2,0,0)
    let hand_pos = transforms[2].transform_point3(Vec3::ZERO);
    assert!(approx_eq(hand_pos.x, -2.0, 0.01) && approx_eq(hand_pos.y, 0.0, 0.01));
}

#[test]
fn test_multi_splat_different_joints() {
    let skel = build_synthetic_skeleton(&["root", "arm", "hand"]);
    let anim = build_synthetic_animation("wave", 1, 1.0, Quat::IDENTITY, Quat::from_rotation_z(FRAC_PI_2));

    let make_splat = |y: f32| GaussianSplat {
        position: [0.0, y, 0.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        _pad: [0; 3],
        spectral: [0; 8],
    };

    let splats = vec![make_splat(0.0), make_splat(1.0), make_splat(2.0)];
    let bindings = vec![0, 1, 2]; // each splat bound to corresponding joint

    let joint_transforms = evaluate_animation(&skel, &anim, 1.0);
    let ibms: Vec<Mat4> = skel.joints.iter().map(|j| j.inverse_bind_matrix).collect();

    let skinned = skin_splats(&splats, &bindings, &joint_transforms, &ibms);

    // Root splat: root has no animation channel in this anim (only arm does),
    // so root stays at identity. IBM[0] takes (0,0,0)->(0,0,0), world is identity.
    let p0 = skinned[0].position;
    assert!(approx_eq(p0[0], 0.0, 0.01) && approx_eq(p0[1], 0.0, 0.01));

    // Arm splat at bind (0,1,0): IBM[1] -> (0,0,0), world[1] = (0,1,0) with 90deg rot
    let p1 = skinned[1].position;
    assert!(approx_eq(p1[0], 0.0, 0.05) && approx_eq(p1[1], 1.0, 0.05));

    // Hand splat at bind (0,2,0): moves to (-1,1,0)
    let p2 = skinned[2].position;
    assert!(approx_eq(p2[0], -1.0, 0.05) && approx_eq(p2[1], 1.0, 0.05));
}
