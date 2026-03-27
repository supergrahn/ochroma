use std::time::{Duration, Instant};

pub struct HeadlessRunner {
    pub sim_speed: f32,
    pub tick_rate_hz: f32,
}

impl HeadlessRunner {
    pub fn new(sim_speed: f32) -> Self {
        Self { sim_speed, tick_rate_hz: 10.0 }
    }

    /// Run headless for a given duration (real seconds).
    /// Returns the number of simulation ticks completed.
    pub fn run_for(&self, duration: Duration, mut tick_fn: impl FnMut(f32)) -> u64 {
        let dt = self.sim_speed / self.tick_rate_hz;
        let tick_interval = Duration::from_secs_f32(1.0 / self.tick_rate_hz);
        let start = Instant::now();
        let mut ticks = 0u64;

        while start.elapsed() < duration {
            tick_fn(dt);
            ticks += 1;
            std::thread::sleep(tick_interval);
        }

        ticks
    }
}
