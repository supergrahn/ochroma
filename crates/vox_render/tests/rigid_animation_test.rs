use glam::{Quat, Vec3};
use std::f32::consts::{FRAC_PI_2, PI};
use vox_render::rigid_animation::*;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn approx_vec3(a: Vec3, b: Vec3, eps: f32) -> bool {
    approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
}

#[test]
fn rotation_clip_halfway_rotation_is_half() {
    let mut clip = RigidClip::new("rot", 2.0, false);
    clip.keyframes = vec![
        RigidKeyframe {
            time: 0.0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        RigidKeyframe {
            time: 2.0,
            position: Vec3::ZERO,
            rotation: Quat::from_axis_angle(Vec3::Y, PI),
            scale: Vec3::ONE,
        },
    ];

    let kf = clip.sample(1.0);
    let (axis, angle) = kf.rotation.to_axis_angle();
    assert!(
        approx_eq(angle, FRAC_PI_2, 0.05),
        "Expected ~PI/2 rotation at half duration, got {}",
        angle
    );
    assert!(axis.y.abs() > 0.9, "Expected rotation around Y axis");
}

#[test]
fn looping_clip_wraps_correctly() {
    let mut clip = RigidClip::new("loop", 2.0, true);
    clip.keyframes = vec![
        RigidKeyframe {
            time: 0.0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        RigidKeyframe {
            time: 2.0,
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
    ];

    // 1.5x duration = 3.0, wraps to 1.0 (50% through the 2.0 clip)
    let kf = clip.sample(3.0);
    assert!(
        approx_vec3(kf.position, Vec3::new(5.0, 0.0, 0.0), 0.01),
        "Expected position (5,0,0) after wrap, got {:?}",
        kf.position
    );
}

#[test]
fn state_machine_trigger_transitions_to_new_state() {
    let mut sm = RigidStateMachine::new("idle");

    let mut idle_clip = RigidClip::new("idle", 1.0, true);
    idle_clip.keyframes = vec![RigidKeyframe {
        time: 0.0,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    }];

    let mut walk_clip = RigidClip::new("walk", 1.0, true);
    walk_clip.keyframes = vec![
        RigidKeyframe {
            time: 0.0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
        RigidKeyframe {
            time: 1.0,
            position: Vec3::new(5.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
    ];

    sm.add_state("idle", idle_clip);
    sm.add_state("walk", walk_clip);
    sm.add_transition(
        "idle",
        RigidTransition {
            target: "walk".to_string(),
            condition: TransitionCondition::OnTrigger("start_walk".to_string()),
            blend_time: 0.0, // instant transition
        },
    );

    assert_eq!(sm.current_state(), "idle");

    // Fire trigger and tick
    sm.trigger("start_walk");
    sm.tick(0.016);

    assert_eq!(sm.current_state(), "walk");
}

#[test]
fn state_machine_blend_during_transition() {
    let mut sm = RigidStateMachine::new("a");

    let mut clip_a = RigidClip::new("a", 2.0, true);
    clip_a.keyframes = vec![RigidKeyframe {
        time: 0.0,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    }];

    let mut clip_b = RigidClip::new("b", 2.0, true);
    clip_b.keyframes = vec![RigidKeyframe {
        time: 0.0,
        position: Vec3::new(10.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    }];

    sm.add_state("a", clip_a);
    sm.add_state("b", clip_b);
    sm.add_transition(
        "a",
        RigidTransition {
            target: "b".to_string(),
            condition: TransitionCondition::OnTrigger("go_b".to_string()),
            blend_time: 1.0, // 1 second blend
        },
    );

    // Tick once to establish state
    sm.tick(0.0);

    // Fire trigger, then tick halfway through the blend
    sm.trigger("go_b");
    sm.tick(0.0); // This tick detects the trigger and starts the transition

    // Now tick 0.5s into the 1.0s blend
    let kf = sm.tick(0.5);

    // During blend, position should be between (0,0,0) and (10,0,0)
    // The exact value depends on blend progress, but it should not be at either extreme
    assert!(
        kf.position.x > 0.1 && kf.position.x < 9.9,
        "Expected blended position between states, got {:?}",
        kf.position
    );
}

#[test]
fn rotation_loop_generates_correct_keyframes() {
    let clip = RigidClip::rotation_loop(Vec3::Y, 2.0);
    assert_eq!(clip.keyframes.len(), 3);
    assert!(approx_eq(clip.duration, 0.5, 0.001));
    assert!(clip.looping);

    // First keyframe: identity rotation
    let (_, angle0) = clip.keyframes[0].rotation.to_axis_angle();
    assert!(approx_eq(angle0, 0.0, 0.01));

    // Middle keyframe: PI rotation
    let (_, angle1) = clip.keyframes[1].rotation.to_axis_angle();
    assert!(approx_eq(angle1, PI, 0.05));

    // Sampling at 25% should give ~PI/2
    let kf = clip.sample(clip.duration * 0.25);
    let (_, angle) = kf.rotation.to_axis_angle();
    assert!(
        approx_eq(angle, FRAC_PI_2, 0.1),
        "Expected ~PI/2 at 25%, got {}",
        angle
    );
}

#[test]
fn swing_generates_correct_back_and_forth() {
    let max_angle = FRAC_PI_2;
    let clip = RigidClip::swing(max_angle, 1.0, Vec3::Y);
    assert_eq!(clip.keyframes.len(), 5);
    assert!(clip.looping);

    // At 25% duration: max positive rotation
    let kf = clip.sample(clip.duration * 0.25);
    let (_, angle) = kf.rotation.to_axis_angle();
    assert!(
        approx_eq(angle, max_angle, 0.05),
        "Expected max angle at 25%, got {}",
        angle
    );

    // At 50% duration: back to identity
    let kf = clip.sample(clip.duration * 0.5);
    let (_, angle) = kf.rotation.to_axis_angle();
    assert!(
        approx_eq(angle, 0.0, 0.05),
        "Expected identity at 50%, got {}",
        angle
    );

    // At 75% duration: max negative rotation (same magnitude, opposite direction)
    let kf = clip.sample(clip.duration * 0.75);
    let (_, angle) = kf.rotation.to_axis_angle();
    assert!(
        approx_eq(angle, max_angle, 0.05),
        "Expected max angle at 75%, got {}",
        angle
    );
}

#[test]
fn bounce_helper_generates_up_down_motion() {
    let clip = RigidClip::bounce(5.0, 1.0);
    assert_eq!(clip.keyframes.len(), 5);
    assert!(clip.looping);

    // At 25% duration: peak height
    let kf = clip.sample(clip.duration * 0.25);
    assert!(
        approx_eq(kf.position.y, 5.0, 0.01),
        "Expected peak at 25%, got {}",
        kf.position.y
    );

    // At 75% duration: negative peak
    let kf = clip.sample(clip.duration * 0.75);
    assert!(
        approx_eq(kf.position.y, -5.0, 0.01),
        "Expected trough at 75%, got {}",
        kf.position.y
    );
}

#[test]
fn state_machine_bool_condition() {
    let mut sm = RigidStateMachine::new("off");

    let mut off_clip = RigidClip::new("off", 1.0, true);
    off_clip.keyframes = vec![RigidKeyframe::identity(0.0)];

    let mut on_clip = RigidClip::new("on", 1.0, true);
    on_clip.keyframes = vec![RigidKeyframe {
        time: 0.0,
        position: Vec3::new(0.0, 10.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    }];

    sm.add_state("off", off_clip);
    sm.add_state("on", on_clip);
    sm.add_transition(
        "off",
        RigidTransition {
            target: "on".to_string(),
            condition: TransitionCondition::OnBool("active".to_string(), true),
            blend_time: 0.0,
        },
    );

    sm.tick(0.016);
    assert_eq!(sm.current_state(), "off");

    sm.set_bool("active", true);
    sm.tick(0.016);
    assert_eq!(sm.current_state(), "on");
}

#[test]
fn state_machine_after_time_condition() {
    let mut sm = RigidStateMachine::new("intro");

    let mut intro_clip = RigidClip::new("intro", 2.0, false);
    intro_clip.keyframes = vec![
        RigidKeyframe::identity(0.0),
        RigidKeyframe::identity(2.0),
    ];

    let mut main_clip = RigidClip::new("main", 1.0, true);
    main_clip.keyframes = vec![RigidKeyframe::identity(0.0)];

    sm.add_state("intro", intro_clip);
    sm.add_state("main", main_clip);
    sm.add_transition(
        "intro",
        RigidTransition {
            target: "main".to_string(),
            condition: TransitionCondition::AfterTime(1.5),
            blend_time: 0.0,
        },
    );

    // Tick to 1.0s — should still be in intro
    sm.tick(1.0);
    assert_eq!(sm.current_state(), "intro");

    // Tick to 2.0s — should transition (time >= 1.5)
    sm.tick(1.0);
    assert_eq!(sm.current_state(), "main");
}
