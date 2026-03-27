use std::collections::VecDeque;

/// Frame timing metrics.
#[derive(Debug, Clone, Default)]
pub struct FrameMetrics {
    pub frame_time_ms: f32,
    pub sort_time_ms: f32,
    pub rasterize_time_ms: f32,
    pub present_time_ms: f32,
    pub splat_count_visible: u32,
    pub splat_count_culled: u32,
    pub instance_count: u32,
    pub vram_usage_mb: f32,
}

/// Rolling telemetry collector.
pub struct TelemetryCollector {
    history: VecDeque<FrameMetrics>,
    max_history: usize,
}

impl TelemetryCollector {
    pub fn new(max_history: usize) -> Self {
        Self { history: VecDeque::new(), max_history }
    }

    pub fn record(&mut self, metrics: FrameMetrics) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }

    pub fn avg_frame_time_ms(&self) -> f32 {
        if self.history.is_empty() { return 0.0; }
        self.history.iter().map(|m| m.frame_time_ms).sum::<f32>() / self.history.len() as f32
    }

    pub fn avg_fps(&self) -> f32 {
        let avg = self.avg_frame_time_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    pub fn latest(&self) -> Option<&FrameMetrics> {
        self.history.back()
    }

    pub fn history(&self) -> &VecDeque<FrameMetrics> {
        &self.history
    }
}
