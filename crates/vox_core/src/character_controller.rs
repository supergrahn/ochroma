use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::ecs::TransformComponent;

/// Physics-driven character controller component.
#[derive(Component, Debug, Clone)]
pub struct CharacterController {
    pub speed: f32,
    pub sprint_multiplier: f32,
    pub jump_force: f32,
    pub gravity: f32,
    pub velocity: Vec3,
    pub grounded: bool,
    pub height: f32,
    pub radius: f32,
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            speed: 5.0,
            sprint_multiplier: 2.0,
            jump_force: 8.0,
            gravity: 20.0,
            velocity: Vec3::ZERO,
            grounded: false,
            height: 1.8,
            radius: 0.3,
        }
    }
}

/// Advance the character controller by one tick.
///
/// This is a plain function (not a Bevy system) so it can be tested without a
/// `World`.  A thin Bevy system can call this each frame.
pub fn character_controller_tick(
    cc: &mut CharacterController,
    transform: &mut TransformComponent,
    move_input: Vec3, // normalized XZ movement direction
    jump_pressed: bool,
    dt: f32,
) {
    // Ground detection
    cc.grounded = transform.position.y <= cc.height * 0.5 + 0.05;

    // Gravity
    if !cc.grounded {
        cc.velocity.y -= cc.gravity * dt;
    } else {
        if cc.velocity.y < 0.0 {
            cc.velocity.y = 0.0;
        }
        transform.position.y = transform.position.y.max(cc.height * 0.5);
    }

    // Jump
    if jump_pressed && cc.grounded {
        cc.velocity.y = cc.jump_force;
        cc.grounded = false;
    }

    // Horizontal movement
    cc.velocity.x = move_input.x * cc.speed;
    cc.velocity.z = move_input.z * cc.speed;

    // Apply
    transform.position += cc.velocity * dt;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cc() -> CharacterController {
        CharacterController::default()
    }

    fn make_transform_on_ground(cc: &CharacterController) -> TransformComponent {
        TransformComponent {
            position: Vec3::new(0.0, cc.height * 0.5, 0.0),
            ..Default::default()
        }
    }

    fn make_transform_in_air(cc: &CharacterController) -> TransformComponent {
        TransformComponent {
            position: Vec3::new(0.0, cc.height * 0.5 + 5.0, 0.0),
            ..Default::default()
        }
    }

    #[test]
    fn character_on_ground_stays_grounded() {
        let mut cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, 1.0 / 60.0);
        assert!(cc.grounded);
    }

    #[test]
    fn character_falls_when_above_ground() {
        let mut cc = make_cc();
        let mut t = make_transform_in_air(&cc);
        let y_before = t.position.y;
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, 1.0 / 60.0);
        assert!(t.position.y < y_before, "character should fall");
        assert!(!cc.grounded);
    }

    #[test]
    fn character_cannot_fall_below_half_height() {
        let mut cc = make_cc();
        let half_h = cc.height * 0.5;
        let mut t = TransformComponent {
            position: Vec3::new(0.0, half_h, 0.0),
            ..Default::default()
        };
        cc.velocity.y = -100.0;
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, 1.0 / 60.0);
        assert!(
            t.position.y >= half_h - 0.001,
            "should not fall below half-height, got {}",
            t.position.y
        );
    }

    #[test]
    fn jump_sets_velocity_y() {
        let mut cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, true, 1.0 / 60.0);
        // After the tick where jump was pressed, velocity.y should have been
        // set to jump_force (then one tick of that velocity is applied).
        // The velocity should still be close to jump_force (minus one frame of gravity
        // is NOT subtracted because grounded was true at start of tick).
        assert!(
            (cc.velocity.y - cc.jump_force).abs() < 0.01,
            "velocity.y should equal jump_force, got {}",
            cc.velocity.y
        );
    }

    #[test]
    fn no_double_jump() {
        let mut cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        // First jump
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, true, 1.0 / 60.0);
        // Simulate a few frames in the air
        for _ in 0..5 {
            character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, 1.0 / 60.0);
        }
        assert!(!cc.grounded, "should be in air after several frames");
        let vel_before = cc.velocity.y;
        // Try to jump again in the air
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, true, 1.0 / 60.0);
        // Velocity should have decreased (gravity), not jumped back up
        assert!(
            cc.velocity.y < vel_before,
            "double jump should not work; vel {} should be less than {}",
            cc.velocity.y,
            vel_before
        );
    }

    #[test]
    fn movement_applies_at_correct_speed() {
        let mut cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        let dt = 1.0 / 60.0;
        let dir = Vec3::new(1.0, 0.0, 0.0);
        character_controller_tick(&mut cc, &mut t, dir, false, dt);
        let expected_dx = cc.speed * dt;
        assert!(
            (t.position.x - expected_dx).abs() < 0.001,
            "expected x ~ {}, got {}",
            expected_dx,
            t.position.x
        );
    }

    #[test]
    fn gravity_accumulates_over_time() {
        let mut cc = make_cc();
        let mut t = make_transform_in_air(&cc);
        let dt = 1.0 / 60.0;
        // Tick once
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, dt);
        let vel_after_1 = cc.velocity.y;
        // Tick again
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, dt);
        let vel_after_2 = cc.velocity.y;
        assert!(
            vel_after_2 < vel_after_1,
            "gravity should accumulate: {} should be less than {}",
            vel_after_2,
            vel_after_1
        );
    }

    #[test]
    fn zero_input_no_horizontal_movement() {
        let mut cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        let x_before = t.position.x;
        let z_before = t.position.z;
        character_controller_tick(&mut cc, &mut t, Vec3::ZERO, false, 1.0 / 60.0);
        assert!(
            (t.position.x - x_before).abs() < 1e-6,
            "x should not change"
        );
        assert!(
            (t.position.z - z_before).abs() < 1e-6,
            "z should not change"
        );
    }

    #[test]
    fn default_values_are_sensible() {
        let cc = CharacterController::default();
        assert!(cc.speed > 0.0);
        assert!(cc.jump_force > 0.0);
        assert!(cc.gravity > 0.0);
        assert!(cc.height > 0.0);
        assert!(cc.radius > 0.0);
        assert!(!cc.grounded);
    }
}
