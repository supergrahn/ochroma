use glam::{Mat4, Vec3, Quat};

/// VR eye.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye { Left, Right }

/// Per-eye view configuration.
#[derive(Debug, Clone)]
pub struct EyeView {
    pub eye: Eye,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
    pub viewport_offset: [u32; 2],
    pub viewport_size: [u32; 2],
}

/// VR headset state (pose, input).
#[derive(Debug, Clone)]
pub struct HeadsetState {
    pub head_position: Vec3,
    pub head_rotation: Quat,
    pub left_hand_position: Option<Vec3>,
    pub left_hand_rotation: Option<Quat>,
    pub right_hand_position: Option<Vec3>,
    pub right_hand_rotation: Option<Quat>,
    pub ipd: f32, // inter-pupillary distance in metres
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_grip: f32,
    pub right_grip: f32,
}

impl Default for HeadsetState {
    fn default() -> Self {
        Self {
            head_position: Vec3::new(0.0, 1.7, 0.0), // standing height
            head_rotation: Quat::IDENTITY,
            left_hand_position: None,
            left_hand_rotation: None,
            right_hand_position: None,
            right_hand_rotation: None,
            ipd: 0.063, // average IPD
            left_trigger: 0.0,
            right_trigger: 0.0,
            left_grip: 0.0,
            right_grip: 0.0,
        }
    }
}

impl HeadsetState {
    /// Compute stereo eye views from headset state.
    pub fn compute_eye_views(&self, near: f32, far: f32, fov_y: f32, aspect: f32) -> (EyeView, EyeView) {
        let half_ipd = self.ipd * 0.5;

        let left_eye_offset = Vec3::new(-half_ipd, 0.0, 0.0);
        let right_eye_offset = Vec3::new(half_ipd, 0.0, 0.0);

        let left_pos = self.head_position + self.head_rotation * left_eye_offset;
        let right_pos = self.head_position + self.head_rotation * right_eye_offset;

        let forward = self.head_rotation * Vec3::NEG_Z;
        let up = self.head_rotation * Vec3::Y;

        let left_view = EyeView {
            eye: Eye::Left,
            view_matrix: Mat4::look_at_rh(left_pos, left_pos + forward, up),
            projection_matrix: Mat4::perspective_rh(fov_y, aspect, near, far),
            viewport_offset: [0, 0],
            viewport_size: [1440, 1600], // typical Quest resolution per eye
        };

        let right_view = EyeView {
            eye: Eye::Right,
            view_matrix: Mat4::look_at_rh(right_pos, right_pos + forward, up),
            projection_matrix: Mat4::perspective_rh(fov_y, aspect, near, far),
            viewport_offset: [1440, 0],
            viewport_size: [1440, 1600],
        };

        (left_view, right_view)
    }
}

/// VR rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrMode {
    /// No VR — standard desktop rendering.
    Desktop,
    /// Simulated VR — renders stereo but outputs to desktop window (for development).
    Simulated,
    /// Real VR — connected to OpenXR runtime.
    OpenXR,
}

/// Manages VR session lifecycle.
pub struct VrSession {
    pub mode: VrMode,
    pub headset: HeadsetState,
    pub target_framerate: u32,
    /// Foveated rendering: reduce splat density in peripheral vision.
    pub foveation_level: u32, // 0=none, 1=low, 2=medium, 3=high
}

impl VrSession {
    pub fn new_desktop() -> Self {
        Self {
            mode: VrMode::Desktop,
            headset: HeadsetState::default(),
            target_framerate: 72,
            foveation_level: 0,
        }
    }

    pub fn new_simulated() -> Self {
        Self {
            mode: VrMode::Simulated,
            headset: HeadsetState::default(),
            target_framerate: 72,
            foveation_level: 1,
        }
    }

    /// Simulate head movement for development without a headset.
    pub fn simulate_head_turn(&mut self, yaw: f32, pitch: f32) {
        self.headset.head_rotation = Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
    }

    /// Check if stereo rendering is needed.
    pub fn needs_stereo(&self) -> bool {
        self.mode != VrMode::Desktop
    }

    /// Get the target frame time in milliseconds.
    pub fn target_frame_ms(&self) -> f32 {
        1000.0 / self.target_framerate as f32
    }
}
