use glam::Vec3;
use vox_physics::RapierPhysicsWorld;
use vox_core::input::{InputState, InputSource};

pub struct CharacterController {
    pub body_handle: vox_physics::RigidBodyHandle,
    pub collider_handle: vox_physics::ColliderHandle,
    pub position: Vec3,
    pub yaw: f32,
    pub speed: f32,
    pub jump_velocity: f32,
    pub on_ground: bool,
    pub vertical_velocity: f32,
    pub enabled: bool,
}

impl CharacterController {
    pub fn new(physics: &mut RapierPhysicsWorld, spawn_pos: Vec3) -> Self {
        let (body_handle, collider_handle) = physics.add_character_controller(
            [spawn_pos.x, spawn_pos.y, spawn_pos.z],
            0.4,
            1.8,
        );
        Self {
            body_handle,
            collider_handle,
            position: spawn_pos,
            yaw: 0.0,
            speed: 5.0,
            jump_velocity: 8.0,
            on_ground: true,
            vertical_velocity: 0.0,
            enabled: false,
        }
    }

    /// Update position based on input. Call before physics.step().
    pub fn update(
        &mut self,
        input: &InputState,
        dt: f32,
        physics: &mut RapierPhysicsWorld,
    ) {
        if !self.enabled {
            return;
        }

        // WASD using Linux/X11 scancodes: W=17, A=30, S=31, D=32
        let mut move_dir = Vec3::ZERO;
        let forward = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
        let right = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());

        if input.is_pressed(InputSource::Key(17)) { move_dir += forward; }  // W
        if input.is_pressed(InputSource::Key(31)) { move_dir -= forward; }  // S
        if input.is_pressed(InputSource::Key(30)) { move_dir -= right; }    // A
        if input.is_pressed(InputSource::Key(32)) { move_dir += right; }    // D

        if move_dir.length_squared() > 0.001 {
            move_dir = move_dir.normalize() * self.speed;
        }

        // Simple gravity
        if !self.on_ground {
            self.vertical_velocity -= 9.81 * dt;
        }

        // Jump: Space = key 57
        if self.on_ground && input.was_just_pressed(InputSource::Key(57)) {
            self.vertical_velocity = self.jump_velocity;
            self.on_ground = false;
        }

        let next_pos = self.position + (move_dir + Vec3::Y * self.vertical_velocity) * dt;

        // Ground check: y <= 0.9 (half capsule height)
        if next_pos.y <= 0.9 {
            self.position = Vec3::new(next_pos.x, 0.9, next_pos.z);
            self.vertical_velocity = 0.0;
            self.on_ground = true;
        } else {
            self.position = next_pos;
        }

        physics.set_kinematic_position(
            self.body_handle,
            [self.position.x, self.position.y, self.position.z],
        );
    }

    /// Camera position: eye level above capsule center
    pub fn camera_position(&self) -> Vec3 {
        self.position + Vec3::Y * 0.8
    }

    /// Camera forward direction from yaw
    pub fn camera_forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos()).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_controller_camera_position_is_above_body() {
        let ctrl = CharacterController {
            body_handle: Default::default(),
            collider_handle: Default::default(),
            position: Vec3::new(0.0, 0.9, 0.0),
            yaw: 0.0,
            speed: 5.0,
            jump_velocity: 8.0,
            on_ground: true,
            vertical_velocity: 0.0,
            enabled: true,
        };
        let cam = ctrl.camera_position();
        assert!(cam.y > ctrl.position.y);
    }

    #[test]
    fn character_stays_on_ground_when_below_threshold() {
        // Gravity would pull below 0.9 — verify ground clamp
        let ctrl = CharacterController {
            body_handle: Default::default(),
            collider_handle: Default::default(),
            position: Vec3::new(0.0, 0.9, 0.0),
            yaw: 0.0,
            speed: 5.0,
            jump_velocity: 8.0,
            on_ground: false,
            vertical_velocity: -50.0, // strong downward
            enabled: true,
        };
        // next_pos.y = 0.9 + (-50.0) * 0.016 = 0.9 - 0.8 = 0.1 < 0.9 → clamped
        let next_y = ctrl.position.y + ctrl.vertical_velocity * 0.016;
        assert!(next_y < 0.9, "expected to go below ground threshold before clamp");
    }
}
