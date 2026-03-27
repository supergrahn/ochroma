use std::time::Instant;

/// Game loop timing with fixed timestep for simulation and variable for rendering.
pub struct GameClock {
    pub fixed_dt: f32,          // simulation timestep (seconds)
    pub accumulator: f32,
    pub time_scale: f32,        // 0.0 = paused, 1.0 = normal
    pub total_time: f64,        // total elapsed time (seconds)
    pub frame_count: u64,
    last_update: Instant,
}

impl GameClock {
    pub fn new(fixed_dt: f32) -> Self {
        Self {
            fixed_dt,
            accumulator: 0.0,
            time_scale: 1.0,
            total_time: 0.0,
            frame_count: 0,
            last_update: Instant::now(),
        }
    }

    /// Call at the start of each frame. Returns real dt.
    pub fn tick(&mut self) -> f32 {
        let now = Instant::now();
        let real_dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;
        self.frame_count += 1;

        let scaled_dt = real_dt * self.time_scale;
        self.accumulator += scaled_dt;
        self.total_time += scaled_dt as f64;

        real_dt
    }

    /// Check if a simulation step should run. Call in a loop.
    pub fn should_step(&mut self) -> bool {
        if self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            true
        } else {
            false
        }
    }

    /// Get interpolation factor for rendering between sim steps.
    pub fn interpolation_factor(&self) -> f32 {
        self.accumulator / self.fixed_dt
    }

    pub fn fps(&self) -> f32 {
        if self.total_time > 0.0 { self.frame_count as f32 / self.total_time as f32 } else { 0.0 }
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.time_scale = if paused { 0.0 } else { 1.0 };
    }

    pub fn is_paused(&self) -> bool { self.time_scale == 0.0 }
}

/// Game loop phase ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GamePhase {
    /// Input processing.
    Input,
    /// Physics and collision.
    Physics,
    /// Game simulation (citizens, economy, traffic).
    Simulation,
    /// AI and pathfinding.
    AI,
    /// Prepare render data (cull, LOD, gather).
    PreRender,
    /// Render.
    Render,
    /// Post-frame cleanup.
    PostFrame,
}

impl GamePhase {
    pub fn all_in_order() -> &'static [GamePhase] {
        &[
            GamePhase::Input,
            GamePhase::Physics,
            GamePhase::Simulation,
            GamePhase::AI,
            GamePhase::PreRender,
            GamePhase::Render,
            GamePhase::PostFrame,
        ]
    }
}
