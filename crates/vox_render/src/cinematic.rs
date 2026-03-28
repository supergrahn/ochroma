use glam::Vec3;

/// Depth of field parameters.
#[derive(Debug, Clone)]
pub struct DepthOfField {
    pub focal_distance: f32, // distance to focus point
    pub aperture: f32,       // f-stop (lower = more blur)
    pub bokeh_shape: BokehShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BokehShape {
    Circle,
    Hexagon,
    Octagon,
}

impl DepthOfField {
    /// Calculate circle of confusion radius at a given depth.
    pub fn coc_radius(&self, depth: f32, sensor_size: f32) -> f32 {
        let coc = (depth - self.focal_distance).abs() * sensor_size
            / (self.aperture * self.focal_distance);
        coc.min(10.0) // cap for sanity
    }
}

/// Motion blur parameters.
#[derive(Debug, Clone)]
pub struct MotionBlur {
    pub shutter_speed: f32, // fraction of frame time (0.5 = 180 degree shutter)
    pub samples: u32,
    pub enabled: bool,
}

/// Lens effects.
#[derive(Debug, Clone)]
pub struct LensEffects {
    pub flare_enabled: bool,
    pub flare_intensity: f32,
    pub chromatic_aberration: f32, // strength (0 = none)
    pub film_grain: f32,           // 0 = none, 1 = heavy
    pub vignette_strength: f32,
}

impl Default for LensEffects {
    fn default() -> Self {
        Self {
            flare_enabled: false,
            flare_intensity: 0.5,
            chromatic_aberration: 0.0,
            film_grain: 0.0,
            vignette_strength: 0.15,
        }
    }
}

/// A keyframe in a camera animation path.
#[derive(Debug, Clone)]
pub struct CameraKeyframe {
    pub time: f32,
    pub position: Vec3,
    pub target: Vec3,
    pub fov: f32,
    pub roll: f32,
}

/// Cinematic camera with spline path, DOF, and lens effects.
pub struct CinematicCamera {
    pub keyframes: Vec<CameraKeyframe>,
    pub dof: DepthOfField,
    pub motion_blur: MotionBlur,
    pub lens: LensEffects,
    pub playback_time: f32,
    pub playback_speed: f32,
    pub looping: bool,
}

impl Default for CinematicCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl CinematicCamera {
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            dof: DepthOfField {
                focal_distance: 10.0,
                aperture: 2.8,
                bokeh_shape: BokehShape::Circle,
            },
            motion_blur: MotionBlur {
                shutter_speed: 0.5,
                samples: 8,
                enabled: false,
            },
            lens: LensEffects::default(),
            playback_time: 0.0,
            playback_speed: 1.0,
            looping: false,
        }
    }

    pub fn add_keyframe(&mut self, time: f32, position: Vec3, target: Vec3, fov: f32) {
        self.keyframes.push(CameraKeyframe {
            time,
            position,
            target,
            fov,
            roll: 0.0,
        });
        self.keyframes
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Evaluate camera position/target at current playback time using smoothstep interpolation.
    pub fn evaluate(&self) -> Option<(Vec3, Vec3, f32)> {
        if self.keyframes.len() < 2 {
            return self
                .keyframes
                .first()
                .map(|k| (k.position, k.target, k.fov));
        }

        let total_time = self.keyframes.last().unwrap().time;
        let t = if self.looping {
            self.playback_time % total_time
        } else {
            self.playback_time.min(total_time)
        };

        // Find surrounding keyframes
        let mut i = 0;
        while i < self.keyframes.len() - 1 && self.keyframes[i + 1].time < t {
            i += 1;
        }

        let k0 = &self.keyframes[i];
        let k1 = &self.keyframes[(i + 1).min(self.keyframes.len() - 1)];

        let segment_t = if (k1.time - k0.time).abs() < 0.001 {
            0.0
        } else {
            (t - k0.time) / (k1.time - k0.time)
        };

        // Smooth interpolation (smoothstep)
        let smooth_t = segment_t * segment_t * (3.0 - 2.0 * segment_t);
        let pos = k0.position.lerp(k1.position, smooth_t);
        let target = k0.target.lerp(k1.target, smooth_t);
        let fov = k0.fov + (k1.fov - k0.fov) * smooth_t;

        Some((pos, target, fov))
    }

    /// Advance playback.
    pub fn tick(&mut self, dt: f32) {
        self.playback_time += dt * self.playback_speed;
    }

    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    pub fn is_finished(&self) -> bool {
        !self.looping && self.playback_time >= self.duration()
    }
}
