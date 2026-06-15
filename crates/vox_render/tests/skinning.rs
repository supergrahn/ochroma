//! Real-behavior tests for linear-blend skinning (plan R42, Tasks 1-2).
//! Synthetic rigs only — no asset files needed. The CesiumMan import test lives
//! game-side (urban_horizon Task 3).

use vox_render::skinning::{
    joint_matrices, sample_clip, skin_mesh, AnimationClip, JointChannel, Skeleton, SkinnedMesh,
    TransformTRS,
};

/// A 2-joint chain: joint 0 root at origin, joint 1 child offset +1 on Y.
/// Inverse-bind = inverse of each joint's bind world transform.
fn two_joint_skeleton() -> Skeleton {
    use glam::Mat4;
    // Bind: joint0 identity at origin; joint1 local-translated +1 on Y.
    let j0_world = Mat4::IDENTITY;
    let j1_local = TransformTRS {
        t: [0.0, 1.0, 0.0],
        r: [0.0, 0.0, 0.0, 1.0],
        s: [1.0, 1.0, 1.0],
    };
    let j1_world = j0_world * glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0));
    Skeleton {
        joint_parents: vec![-1, 0],
        inverse_bind: vec![
            j0_world.inverse().to_cols_array_2d(),
            j1_world.inverse().to_cols_array_2d(),
        ],
        local_bind: vec![TransformTRS::IDENTITY, j1_local],
    }
}

/// Clip that rotates joint 1 from identity to +90° about Z over 1 second.
fn rotate_clip() -> AnimationClip {
    use glam::Quat;
    let q0 = Quat::IDENTITY.to_array();
    let q1 = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array();
    AnimationClip {
        duration: 1.0,
        channels: vec![JointChannel {
            joint: 1,
            times: vec![0.0, 1.0],
            t: vec![],
            r: vec![q0, q1],
            s: vec![],
        }],
    }
}

#[test]
fn clip_sampling_changes_pose() {
    let skel = two_joint_skeleton();
    let clip = rotate_clip();

    let p0 = sample_clip(&clip, &skel, 0.0);
    let pmid = sample_clip(&clip, &skel, 0.5);

    let r0 = p0[1].r;
    let rmid = pmid[1].r;
    let dq: f32 = (0..4).map(|i| (r0[i] - rmid[i]).abs()).sum();
    println!("joint 1 rot @t0: {:?}  @t0.5: {:?}  L1 delta: {dq}", r0, rmid);
    assert!(
        dq > 1e-3,
        "expected joint 1 rotation to change between t=0 and t=0.5, delta {dq}"
    );
    // Root (joint 0) has no channel -> stays at bind identity.
    assert_eq!(p0[0].r, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn skinned_cim_animates() {
    // A single vertex at the tip of joint 1 (world ~ (0,2,0)), fully weighted to
    // joint 1. Rotating joint 1 by +90° about Z swings the tip along -X/+? —
    // its world position must move measurably between clip phases.
    let skel = two_joint_skeleton();
    let clip = rotate_clip();

    let mesh = SkinnedMesh {
        positions: vec![[0.0, 2.0, 0.0]],
        normals: vec![[0.0, 1.0, 0.0]],
        uvs: vec![[0.0, 0.0]],
        indices: vec![],
        joints: vec![[1, 0, 0, 0]],
        weights: vec![[1.0, 0.0, 0.0, 0.0]],
        skeleton: skel.clone(),
    };

    let pose_at = |t: f32| {
        let local = sample_clip(&clip, &skel, t);
        let world = joint_matrices(&skel, &local);
        // skin = world * inverse_bind
        let skin: Vec<[[f32; 4]; 4]> = (0..skel.joint_count())
            .map(|j| {
                let w = glam::Mat4::from_cols_array_2d(&world[j]);
                let ibm = glam::Mat4::from_cols_array_2d(&skel.inverse_bind[j]);
                (w * ibm).to_cols_array_2d()
            })
            .collect();
        skin_mesh(&mesh, &skin).0[0]
    };

    let a = pose_at(0.0);
    let b = pose_at(0.5);
    let delta = (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0_f32, f32::max);
    println!("vertex @phase0: {a:?}  @phase0.5: {b:?}  max-axis delta: {delta}");
    assert!(
        delta > 0.05,
        "skinned vertex must move > 0.05 m between phases, got {delta}"
    );
    // At t=0 the rig is at bind pose -> vertex unmoved.
    assert!(
        (a[0]).abs() < 1e-4 && (a[1] - 2.0).abs() < 1e-4,
        "bind-pose vertex should be ~(0,2,0), got {a:?}"
    );
}

#[test]
fn lbs_deforms_vertex() {
    // Same as above but asserts the deformed direction: +90° about Z sends the
    // tip toward -X (the joint-1-local +Y axis rotates to -X in world... actually
    // +Y rotates to +X for +Z by right-hand rule; assert it left the Y axis).
    let skel = two_joint_skeleton();
    let clip = rotate_clip();
    let mesh = SkinnedMesh {
        positions: vec![[0.0, 2.0, 0.0]],
        normals: vec![[0.0, 1.0, 0.0]],
        uvs: vec![[0.0, 0.0]],
        indices: vec![],
        joints: vec![[1, 0, 0, 0]],
        weights: vec![[1.0, 0.0, 0.0, 0.0]],
        skeleton: skel.clone(),
    };
    let local = sample_clip(&clip, &skel, 1.0);
    let world = joint_matrices(&skel, &local);
    let skin: Vec<[[f32; 4]; 4]> = (0..skel.joint_count())
        .map(|j| {
            let w = glam::Mat4::from_cols_array_2d(&world[j]);
            let ibm = glam::Mat4::from_cols_array_2d(&skel.inverse_bind[j]);
            (w * ibm).to_cols_array_2d()
        })
        .collect();
    let d = skin_mesh(&mesh, &skin).0[0];
    println!("fully-rotated vertex: {d:?}");
    // The tip left the +Y axis: its X moved away from 0 by ~1 m (joint1 at y=1,
    // tip 1 above it rotated 90° -> ~1 unit horizontal displacement).
    assert!(
        d[0].abs() > 0.5,
        "expected horizontal displacement > 0.5 m, got x={}",
        d[0]
    );
}
