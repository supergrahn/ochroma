use crate::spectral_framebuffer::SpectralFramebuffer;

/// Temporal accumulation buffer — blends current frame with reprojected history.
pub struct TemporalAccumulator {
    pub width: u32,
    pub height: u32,
    /// Accumulated spectral history.
    history: Vec<[f32; 8]>,
    /// History depth (for nearest-depth rejection).
    history_depth: Vec<f32>,
    /// Number of frames accumulated per pixel.
    frame_count: Vec<u16>,
    /// Blend factor (0 = all history, 1 = all current).
    pub blend_alpha: f32,
    /// Maximum history frames before reset.
    pub max_accumulation: u16,
}

impl TemporalAccumulator {
    pub fn new(width: u32, height: u32) -> Self {
        let count = (width * height) as usize;
        Self {
            width,
            height,
            history: vec![[0.0; 8]; count],
            history_depth: vec![f32::MAX; count],
            frame_count: vec![0; count],
            blend_alpha: 0.1, // 10% current, 90% history
            max_accumulation: 64,
        }
    }

    /// Accumulate a new frame into the history.
    /// Uses motion vectors to reproject history, rejects disoccluded pixels.
    pub fn accumulate(&mut self, current: &SpectralFramebuffer) {
        assert_eq!(current.width, self.width);
        assert_eq!(current.height, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let i = (y * self.width + x) as usize;

                // Get motion vector to find where this pixel was in the previous frame
                let mv = current.motion[i];
                let prev_x = x as f32 - mv[0];
                let prev_y = y as f32 - mv[1];

                // Bounds check for reprojected position
                let in_bounds = prev_x >= 0.0
                    && prev_x < self.width as f32
                    && prev_y >= 0.0
                    && prev_y < self.height as f32;

                let current_depth = current.depth[i];
                let current_spectral = current.spectral[i];

                if !in_bounds || current.sample_count[i] == 0 {
                    // No valid reprojection — use current frame only
                    self.history[i] = current_spectral;
                    self.history_depth[i] = current_depth;
                    self.frame_count[i] = 1;
                    continue;
                }

                // Check depth consistency (reject disocclusion)
                let prev_i = (prev_y as u32 * self.width + prev_x as u32) as usize;
                let prev_i = prev_i.min(self.history.len() - 1);
                let depth_diff = (self.history_depth[prev_i] - current_depth).abs();
                let depth_threshold = current_depth * 0.1; // 10% tolerance

                if depth_diff > depth_threshold || self.frame_count[prev_i] == 0 {
                    // Depth discontinuity — reset history
                    self.history[i] = current_spectral;
                    self.history_depth[i] = current_depth;
                    self.frame_count[i] = 1;
                } else {
                    // Blend current with reprojected history
                    let alpha = self.blend_alpha;
                    let prev = self.history[prev_i];
                    for b in 0..8 {
                        self.history[i][b] =
                            prev[b] * (1.0 - alpha) + current_spectral[b] * alpha;
                    }
                    self.history_depth[i] = current_depth;
                    self.frame_count[i] =
                        (self.frame_count[prev_i] + 1).min(self.max_accumulation);
                }
            }
        }
    }

    /// Get the accumulated spectral value at a pixel.
    pub fn get(&self, x: u32, y: u32) -> [f32; 8] {
        let i = (y * self.width + x) as usize;
        if i < self.history.len() {
            self.history[i]
        } else {
            [0.0; 8]
        }
    }

    /// Copy accumulated result back into a framebuffer's spectral channel.
    pub fn write_to_framebuffer(&self, fb: &mut SpectralFramebuffer) {
        for i in 0..self.history.len().min(fb.spectral.len()) {
            fb.spectral[i] = self.history[i];
        }
    }

    /// Average accumulated frames across all pixels.
    pub fn avg_accumulated_frames(&self) -> f32 {
        let total: u64 = self.frame_count.iter().map(|&f| f as u64).sum();
        total as f32 / self.frame_count.len() as f32
    }

    /// Reset all history (e.g., on camera cut).
    pub fn reset(&mut self) {
        for h in &mut self.history {
            *h = [0.0; 8];
        }
        for d in &mut self.history_depth {
            *d = f32::MAX;
        }
        for f in &mut self.frame_count {
            *f = 0;
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let count = (width * height) as usize;
        self.history = vec![[0.0; 8]; count];
        self.history_depth = vec![f32::MAX; count];
        self.frame_count = vec![0; count];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_framebuffer::SpectralFramebuffer;

    #[test]
    fn first_frame_equals_input() {
        let mut ta = TemporalAccumulator::new(2, 2);
        let mut fb = SpectralFramebuffer::new(2, 2);
        fb.spectral[0] = [0.5; 8];
        fb.sample_count[0] = 1;
        fb.depth[0] = 10.0;

        ta.accumulate(&fb);
        let result = ta.get(0, 0);
        assert_eq!(result, [0.5; 8], "first frame should equal input (no history)");
    }

    #[test]
    fn reset_clears_history() {
        let mut ta = TemporalAccumulator::new(2, 2);
        let mut fb = SpectralFramebuffer::new(2, 2);
        fb.spectral[0] = [1.0; 8];
        fb.sample_count[0] = 1;
        fb.depth[0] = 5.0;
        ta.accumulate(&fb);

        ta.reset();
        let result = ta.get(0, 0);
        assert_eq!(result, [0.0; 8], "reset should zero all history");
        assert_eq!(ta.avg_accumulated_frames(), 0.0);
    }

    #[test]
    fn resize_changes_dimensions() {
        let mut ta = TemporalAccumulator::new(4, 4);
        assert_eq!(ta.width, 4);
        ta.resize(8, 8);
        assert_eq!(ta.width, 8);
        assert_eq!(ta.height, 8);
    }
}
