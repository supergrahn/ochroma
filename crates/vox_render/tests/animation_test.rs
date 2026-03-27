use glam::{Mat4, Quat, Vec3};
use vox_render::animation::*;

#[test]
fn identity_bone_does_not_move_splats() {
    let mut skeleton = Skeleton::new();
    skeleton.add_bone(Bone {
        id: 0,
        parent_id: None,
        local_transform: Mat4::IDENTITY,
        name: "root".to_string(),
    });

    let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let bindings = vec![BoneBinding { bone_id: 0 }, BoneBinding { bone_id: 0 }];

    let result = apply_skeleton_to_splats(&skeleton, &positions, &bindings);

    for (orig, transformed) in positions.iter().zip(result.iter()) {
        assert!((orig[0] - transformed[0]).abs() < 1e-5);
        assert!((orig[1] - transformed[1]).abs() < 1e-5);
        assert!((orig[2] - transformed[2]).abs() < 1e-5);
    }
}

#[test]
fn rotating_bone_rotates_splats() {
    let mut skeleton = Skeleton::new();
    // Rotate 90 degrees around Y axis.
    let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    skeleton.add_bone(Bone {
        id: 0,
        parent_id: None,
        local_transform: Mat4::from_quat(rot),
        name: "root".to_string(),
    });

    let positions = vec![[1.0, 0.0, 0.0]];
    let bindings = vec![BoneBinding { bone_id: 0 }];

    let result = apply_skeleton_to_splats(&skeleton, &positions, &bindings);

    // 90-degree Y rotation: (1,0,0) -> (0,0,-1)
    assert!((result[0][0] - 0.0).abs() < 1e-4, "x: {}", result[0][0]);
    assert!((result[0][1] - 0.0).abs() < 1e-4, "y: {}", result[0][1]);
    assert!((result[0][2] - (-1.0)).abs() < 1e-4, "z: {}", result[0][2]);
}

#[test]
fn animation_playback_advances_time() {
    let mut clip = AnimationClip::new("walk", 2.0);
    clip.add_keyframe(0, Keyframe {
        time: 0.0,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    });
    clip.add_keyframe(0, Keyframe {
        time: 2.0,
        position: Vec3::new(10.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    });

    let mut player = AnimationPlayer::new(vec![clip]);
    player.update(1.0); // Advance 1 second into a 2-second clip.

    assert!((player.primary.time - 1.0).abs() < 1e-5);

    let pose = player.sample_pose();
    let m = pose.get(&0).expect("bone 0 should have a pose");
    let (_, _, trans) = m.to_scale_rotation_translation();
    // At t=1.0 out of 2.0, position should be ~(5, 0, 0).
    assert!((trans.x - 5.0).abs() < 1e-3, "x: {}", trans.x);
}

#[test]
fn two_clip_blending_produces_intermediate() {
    let mut clip_a = AnimationClip::new("idle", 1.0);
    clip_a.add_keyframe(0, Keyframe {
        time: 0.0,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    });

    let mut clip_b = AnimationClip::new("run", 1.0);
    clip_b.add_keyframe(0, Keyframe {
        time: 0.0,
        position: Vec3::new(10.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    });

    let mut player = AnimationPlayer::new(vec![clip_a, clip_b]);
    player.blend_to(1, 1.0); // Blend to clip 1 over 1 second.
    player.update(0.5); // Half-way through blend.

    let pose = player.sample_pose();
    let m = pose.get(&0).expect("bone 0 should have a pose");
    let (_, _, trans) = m.to_scale_rotation_translation();
    // Should be roughly (5, 0, 0) — halfway between (0,0,0) and (10,0,0).
    assert!((trans.x - 5.0).abs() < 1.0, "x: {} (expected ~5.0)", trans.x);
}

#[test]
fn parent_child_bone_hierarchy() {
    let mut skeleton = Skeleton::new();
    skeleton.add_bone(Bone {
        id: 0,
        parent_id: None,
        local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        name: "root".to_string(),
    });
    skeleton.add_bone(Bone {
        id: 1,
        parent_id: Some(0),
        local_transform: Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        name: "child".to_string(),
    });

    let positions = vec![[0.0, 0.0, 0.0]];
    let bindings = vec![BoneBinding { bone_id: 1 }];

    let result = apply_skeleton_to_splats(&skeleton, &positions, &bindings);
    // Child world = parent(1,0,0) * child(2,0,0) = (3,0,0).
    assert!((result[0][0] - 3.0).abs() < 1e-4);
}
