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
    /// Maximum walkable slope angle in degrees. Slopes steeper than this cause sliding.
    pub max_slope_angle: f32,
    /// Maximum obstacle height that can be auto-stepped over.
    pub step_height: f32,
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
            max_slope_angle: 45.0,
            step_height: 0.3,
        }
    }
}

/// Check if a slope is walkable (below max_slope_angle).
pub fn is_walkable_slope(normal: Vec3, max_angle_degrees: f32) -> bool {
    let up_dot = normal.dot(Vec3::Y);
    let angle = up_dot.acos().to_degrees();
    angle < max_angle_degrees
}

/// Compute a slide-down velocity for a steep slope.
/// Returns the gravity-driven slide vector projected onto the slope surface.
pub fn compute_slope_slide(normal: Vec3, gravity: f32, dt: f32) -> Vec3 {
    let gravity_vec = Vec3::new(0.0, -gravity * dt, 0.0);
    // Project gravity onto the slope plane
    gravity_vec - normal * gravity_vec.dot(normal)
}

/// Attempt to step up a small obstacle.
/// Returns true if the step was successful (obstacle is below step_height).
pub fn try_step_up(
    cc: &CharacterController,
    transform: &mut TransformComponent,
    _move_dir: Vec3,
    obstacle_height: f32,
) -> bool {
    if obstacle_height <= cc.step_height && obstacle_height > 0.01 {
        transform.position.y += obstacle_height + 0.01;
        true
    } else {
        false
    }
}

/// Slide along a wall instead of stopping dead.
/// Projects velocity onto the wall plane so the player glides along it.
pub fn slide_along_wall(velocity: Vec3, wall_normal: Vec3) -> Vec3 {
    velocity - wall_normal * velocity.dot(wall_normal)
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
        assert!((cc.max_slope_angle - 45.0).abs() < f32::EPSILON);
        assert!((cc.step_height - 0.3).abs() < f32::EPSILON);
    }

    // --- Slope handling tests ---

    #[test]
    fn flat_ground_is_walkable() {
        assert!(super::is_walkable_slope(Vec3::Y, 45.0));
    }

    #[test]
    fn gentle_slope_is_walkable() {
        // ~30 degree slope
        let normal = Vec3::new(0.0, 0.866, 0.5).normalize();
        assert!(super::is_walkable_slope(normal, 45.0));
    }

    #[test]
    fn steep_slope_is_not_walkable() {
        // ~70 degree slope
        let normal = Vec3::new(0.0, 0.342, 0.94).normalize();
        assert!(!super::is_walkable_slope(normal, 45.0));
    }

    #[test]
    fn vertical_wall_is_not_walkable() {
        assert!(!super::is_walkable_slope(Vec3::Z, 45.0));
    }

    #[test]
    fn steep_slope_causes_sliding() {
        // A steep slope normal pointing mostly sideways
        let normal = Vec3::new(0.0, 0.342, 0.94).normalize();
        let slide = super::compute_slope_slide(normal, 20.0, 1.0 / 60.0);
        // Slide should have a downward Y component (player slides down)
        assert!(slide.y < 0.0, "slope slide should push downward, got y={}", slide.y);
        // Slide should have a horizontal component along the slope
        assert!(slide.length() > 0.0, "slope slide should have nonzero magnitude");
    }

    // --- Stair stepping tests ---

    #[test]
    fn step_up_small_obstacle() {
        let cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        let y_before = t.position.y;
        let result = super::try_step_up(&cc, &mut t, Vec3::X, 0.2);
        assert!(result, "should step over 0.2m obstacle");
        assert!(t.position.y > y_before, "position should move up after step");
    }

    #[test]
    fn cannot_step_over_tall_obstacle() {
        let cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        let y_before = t.position.y;
        let result = super::try_step_up(&cc, &mut t, Vec3::X, 0.5);
        assert!(!result, "should not step over 0.5m obstacle with 0.3m step_height");
        assert!((t.position.y - y_before).abs() < 1e-6, "position should not change");
    }

    #[test]
    fn step_ignores_tiny_obstacles() {
        let cc = make_cc();
        let mut t = make_transform_on_ground(&cc);
        let result = super::try_step_up(&cc, &mut t, Vec3::X, 0.005);
        assert!(!result, "should ignore obstacles below 0.01m threshold");
    }

    // --- Wall sliding tests ---

    #[test]
    fn slide_along_wall_produces_tangent_velocity() {
        let velocity = Vec3::new(1.0, 0.0, 1.0);
        let wall_normal = Vec3::X; // wall facing +X
        let slid = super::slide_along_wall(velocity, wall_normal);
        // X component should be removed, Z preserved
        assert!((slid.x).abs() < 1e-6, "x should be zero after sliding along X wall");
        assert!((slid.z - 1.0).abs() < 1e-6, "z should be preserved");
    }

    #[test]
    fn slide_along_wall_parallel_unchanged() {
        let velocity = Vec3::new(0.0, 0.0, 5.0);
        let wall_normal = Vec3::X;
        let slid = super::slide_along_wall(velocity, wall_normal);
        assert!((slid - velocity).length() < 1e-6, "parallel velocity should be unchanged");
    }

    #[test]
    fn slide_along_wall_head_on_stops() {
        let velocity = Vec3::new(3.0, 0.0, 0.0);
        let wall_normal = Vec3::X;
        let slid = super::slide_along_wall(velocity, wall_normal);
        assert!(slid.length() < 1e-6, "head-on into wall should produce zero velocity");
    }
}
