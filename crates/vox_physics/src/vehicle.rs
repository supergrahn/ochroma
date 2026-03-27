//! Vehicle physics — simple vehicle model with wheels, steering, acceleration.

use glam::{Quat, Vec3};

/// State of a single wheel.
#[derive(Debug, Clone)]
pub struct WheelState {
    pub position_offset: Vec3,
    pub rotation: f32,
    pub steering_angle: f32,
    pub grounded: bool,
}

/// A driveable vehicle with Ackermann-ish steering.
pub struct Vehicle {
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: f32,
    pub throttle: f32,
    pub steering: f32,
    pub max_speed: f32,
    pub acceleration: f32,
    pub brake_force: f32,
    pub turn_rate: f32,
    pub wheels: [WheelState; 4],
    braking: bool,
}

impl Vehicle {
    pub fn new(max_speed: f32, acceleration: f32) -> Self {
        // Default wheel layout: front-left, front-right, rear-left, rear-right
        let wheel_offsets = [
            Vec3::new(-0.8, -0.3, 1.2),  // FL
            Vec3::new(0.8, -0.3, 1.2),   // FR
            Vec3::new(-0.8, -0.3, -1.2), // RL
            Vec3::new(0.8, -0.3, -1.2),  // RR
        ];

        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: 0.0,
            throttle: 0.0,
            steering: 0.0,
            max_speed,
            acceleration,
            brake_force: acceleration * 2.0,
            turn_rate: 2.5,
            wheels: wheel_offsets.map(|offset| WheelState {
                position_offset: offset,
                rotation: 0.0,
                steering_angle: 0.0,
                grounded: true,
            }),
            braking: false,
        }
    }

    /// Set driver input.
    pub fn set_input(&mut self, throttle: f32, steering: f32, brake: bool) {
        self.throttle = throttle.clamp(-1.0, 1.0);
        self.steering = steering.clamp(-1.0, 1.0);
        self.braking = brake;
    }

    /// Forward direction in world space.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    /// Current speed (signed along forward axis).
    pub fn speed(&self) -> f32 {
        self.velocity.dot(self.forward())
    }

    /// Step vehicle physics.
    pub fn step(&mut self, dt: f32) {
        let fwd = self.forward();
        let current_speed = self.speed();

        // --- Acceleration / braking ---
        let mut accel = 0.0f32;
        if self.braking {
            // Brake: decelerate toward zero
            if current_speed > 0.01 {
                accel = -self.brake_force;
            } else if current_speed < -0.01 {
                accel = self.brake_force;
            } else {
                self.velocity = Vec3::ZERO;
            }
        } else {
            accel = self.throttle * self.acceleration;
        }

        // Apply acceleration along forward direction
        self.velocity += fwd * accel * dt;

        // Clamp speed
        let speed = self.speed();
        if speed.abs() > self.max_speed {
            let clamped = speed.signum() * self.max_speed;
            // Project velocity onto forward and clamp
            let lateral = self.velocity - fwd * speed;
            self.velocity = fwd * clamped + lateral;
        }

        // Drag (simple)
        self.velocity *= 1.0 - 0.5 * dt;

        // --- Steering ---
        // Turn rate depends on speed: can't turn when stationary
        let speed_factor = (current_speed.abs() / self.max_speed).clamp(0.0, 1.0);
        // At higher speeds, reduce turn rate for stability
        let effective_turn = self.turn_rate * speed_factor * (1.0 - 0.3 * speed_factor);
        let yaw_delta = self.steering * effective_turn * dt * current_speed.signum();
        self.angular_velocity = yaw_delta / dt.max(1e-6);

        let yaw_rot = Quat::from_rotation_y(yaw_delta);
        self.rotation = (self.rotation * yaw_rot).normalize();

        // Realign velocity with forward direction (simple lateral friction)
        let new_fwd = self.forward();
        let forward_speed = self.velocity.dot(new_fwd);
        let lateral_speed = self.velocity - new_fwd * forward_speed;
        self.velocity = new_fwd * forward_speed + lateral_speed * (1.0 - 3.0 * dt).max(0.0);

        // --- Position ---
        self.position += self.velocity * dt;

        // --- Update wheels ---
        let max_steer_angle = 0.5; // ~28 degrees
        for (i, wheel) in self.wheels.iter_mut().enumerate() {
            // Front wheels steer
            if i < 2 {
                wheel.steering_angle = self.steering * max_steer_angle;
            }
            // All wheels spin proportional to speed
            let wheel_circumference = 2.0 * std::f32::consts::PI * 0.3; // ~0.3m radius
            wheel.rotation += (forward_speed / wheel_circumference) * dt * std::f32::consts::TAU;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_accelerates() {
        let mut v = Vehicle::new(30.0, 10.0);
        v.set_input(1.0, 0.0, false);
        for _ in 0..50 {
            v.step(0.016);
        }
        assert!(v.speed() > 1.0, "vehicle should accelerate: speed = {}", v.speed());
    }

    #[test]
    fn brake_decelerates() {
        let mut v = Vehicle::new(30.0, 10.0);
        // First accelerate
        v.set_input(1.0, 0.0, false);
        for _ in 0..50 {
            v.step(0.016);
        }
        let speed_before_brake = v.speed();
        assert!(speed_before_brake > 1.0);

        // Now brake
        v.set_input(0.0, 0.0, true);
        for _ in 0..50 {
            v.step(0.016);
        }
        assert!(
            v.speed() < speed_before_brake,
            "braking should reduce speed: {} vs {}",
            v.speed(),
            speed_before_brake
        );
    }

    #[test]
    fn steering_turns() {
        let mut v = Vehicle::new(30.0, 10.0);
        // Accelerate forward
        v.set_input(1.0, 0.0, false);
        for _ in 0..30 {
            v.step(0.016);
        }
        let fwd_before = v.forward();

        // Turn right
        v.set_input(1.0, 1.0, false);
        for _ in 0..30 {
            v.step(0.016);
        }
        let fwd_after = v.forward();

        let dot = fwd_before.dot(fwd_after);
        assert!(
            dot < 0.99,
            "steering should change direction: dot = {dot}"
        );
    }

    #[test]
    fn speed_capped_at_max() {
        let max = 20.0;
        let mut v = Vehicle::new(max, 50.0); // high accel to test cap
        v.set_input(1.0, 0.0, false);
        for _ in 0..500 {
            v.step(0.016);
        }
        assert!(
            v.speed() <= max + 1.0,
            "speed should be capped: {} vs max {}",
            v.speed(),
            max
        );
    }

    #[test]
    fn reverse_works() {
        let mut v = Vehicle::new(30.0, 10.0);
        v.set_input(-1.0, 0.0, false);
        for _ in 0..50 {
            v.step(0.016);
        }
        assert!(
            v.speed() < -0.5,
            "reverse should give negative speed: {}",
            v.speed()
        );
    }

    #[test]
    fn wheels_rotate_with_speed() {
        let mut v = Vehicle::new(30.0, 10.0);
        let initial_rot = v.wheels[0].rotation;
        v.set_input(1.0, 0.0, false);
        for _ in 0..50 {
            v.step(0.016);
        }
        assert!(
            (v.wheels[0].rotation - initial_rot).abs() > 0.1,
            "wheels should rotate: {} vs {}",
            v.wheels[0].rotation,
            initial_rot
        );
    }
}
