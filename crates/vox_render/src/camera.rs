use glam::{Mat4, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    CityOverview,
    StreetLevel,
    Cinematic,
    FollowAgent,
}

pub struct CameraController {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub mode: CameraMode,
    pub orbit_angle: f32,
    pub orbit_distance: f32,
    pub altitude: f32,
    pub aspect_ratio: f32,
}

impl CameraController {
    pub fn new(aspect_ratio: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 50.0, 50.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 10000.0,
            mode: CameraMode::CityOverview,
            orbit_angle: 0.0,
            orbit_distance: 100.0,
            altitude: 50.0,
            aspect_ratio,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect_ratio, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Orbit around target by delta angle (radians).
    pub fn orbit(&mut self, delta_angle: f32) {
        self.orbit_angle += delta_angle;
        self.update_position();
    }

    /// Zoom by changing orbit distance.
    pub fn zoom(&mut self, delta: f32) {
        self.orbit_distance = (self.orbit_distance + delta).clamp(10.0, 10000.0);
        self.update_position();
    }

    /// Pan target position.
    pub fn pan(&mut self, dx: f32, dz: f32) {
        self.target.x += dx;
        self.target.z += dz;
        self.update_position();
    }

    /// Set altitude (clamped above 0).
    pub fn set_altitude(&mut self, alt: f32) {
        self.altitude = alt.max(1.0);
        self.update_position();
    }

    /// Public wrapper to recalculate position from orbit parameters.
    pub fn update_position_public(&mut self) {
        self.update_position();
    }

    fn update_position(&mut self) {
        self.position = Vec3::new(
            self.target.x + self.orbit_angle.cos() * self.orbit_distance,
            self.target.y + self.altitude,
            self.target.z + self.orbit_angle.sin() * self.orbit_distance,
        );
    }
}
