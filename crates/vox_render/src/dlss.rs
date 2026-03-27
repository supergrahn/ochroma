/// DLSS quality mode — determines internal render resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlssQuality {
    Off,               // Render at native resolution
    Quality,           // 67% of native (1.5x upscale)
    Balanced,          // 50% of native (2x upscale)
    Performance,       // 33% of native (3x upscale)
    UltraPerformance,  // 25% of native (4x upscale)
}

impl DlssQuality {
    /// Render resolution as fraction of display resolution.
    pub fn render_fraction(&self) -> f32 {
        match self {
            Self::Off => 1.0,
            Self::Quality => 0.67,
            Self::Balanced => 0.50,
            Self::Performance => 0.33,
            Self::UltraPerformance => 0.25,
        }
    }

    /// Internal resolution for a given display size.
    pub fn internal_resolution(&self, display_w: u32, display_h: u32) -> (u32, u32) {
        let f = self.render_fraction();
        ((display_w as f32 * f) as u32, (display_h as f32 * f) as u32)
    }
}

/// Frame generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameGeneration {
    Off,
    On,  // Generate 1 intermediate frame (doubles output FPS)
}

/// The DLSS pipeline manager.
pub struct DlssPipeline {
    pub quality: DlssQuality,
    pub frame_gen: FrameGeneration,
    pub display_width: u32,
    pub display_height: u32,
    /// Previous frame for frame generation.
    prev_frame: Option<Vec<[u8; 4]>>,
    prev_motion: Option<Vec<[f32; 2]>>,
    /// Stats
    pub frames_generated: u64,
    pub frames_upscaled: u64,
}

impl DlssPipeline {
    pub fn new(display_width: u32, display_height: u32, quality: DlssQuality) -> Self {
        Self {
            quality, frame_gen: FrameGeneration::Off,
            display_width, display_height,
            prev_frame: None, prev_motion: None,
            frames_generated: 0, frames_upscaled: 0,
        }
    }

    /// Get the resolution the renderer should render at.
    pub fn render_resolution(&self) -> (u32, u32) {
        self.quality.internal_resolution(self.display_width, self.display_height)
    }

    /// Upscale a rendered frame from internal to display resolution.
    /// On NVIDIA: this would call the DLSS SDK.
    /// Fallback: bilinear upscale.
    pub fn upscale(
        &mut self,
        pixels: &[[u8; 4]],
        src_w: u32, src_h: u32,
        _depth: &[f32],
        _motion: &[[f32; 2]],
    ) -> Vec<[u8; 4]> {
        self.frames_upscaled += 1;

        if self.quality == DlssQuality::Off || (src_w == self.display_width && src_h == self.display_height) {
            return pixels.to_vec();
        }

        // Bilinear upscale fallback
        bilinear_upscale(pixels, src_w, src_h, self.display_width, self.display_height)
    }

    /// Generate an intermediate frame between previous and current.
    /// On NVIDIA: this would use DLSS Frame Generation.
    /// Fallback: simple linear interpolation.
    pub fn generate_frame(
        &mut self,
        current: &[[u8; 4]],
        motion: &[[f32; 2]],
    ) -> Option<Vec<[u8; 4]>> {
        if self.frame_gen == FrameGeneration::Off { return None; }

        let result = if let Some(prev) = &self.prev_frame {
            // Simple 50/50 blend as fallback for real frame generation
            let blended: Vec<[u8; 4]> = prev.iter().zip(current.iter())
                .map(|(p, c)| [
                    ((p[0] as u16 + c[0] as u16) / 2) as u8,
                    ((p[1] as u16 + c[1] as u16) / 2) as u8,
                    ((p[2] as u16 + c[2] as u16) / 2) as u8,
                    255,
                ])
                .collect();
            self.frames_generated += 1;
            Some(blended)
        } else {
            None
        };

        self.prev_frame = Some(current.to_vec());
        self.prev_motion = Some(motion.to_vec());

        result
    }

    /// Effective output FPS multiplier.
    pub fn fps_multiplier(&self) -> f32 {
        let upscale_boost = 1.0 / (self.quality.render_fraction() * self.quality.render_fraction());
        let gen_boost = if self.frame_gen == FrameGeneration::On { 2.0 } else { 1.0 };
        upscale_boost * gen_boost
    }

    pub fn resize(&mut self, display_w: u32, display_h: u32) {
        self.display_width = display_w;
        self.display_height = display_h;
        self.prev_frame = None;
    }
}

/// Bilinear upscale (fallback for non-NVIDIA).
fn bilinear_upscale(
    src: &[[u8; 4]], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32,
) -> Vec<[u8; 4]> {
    let mut dst = vec![[0u8; 4]; (dst_w * dst_h) as usize];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx as f32 * src_w as f32 / dst_w as f32;
            let sy = dy as f32 * src_h as f32 / dst_h as f32;
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(src_w - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let fx = sx - sx.floor();
            let fy = sy - sy.floor();

            let p00 = src[(y0 * src_w + x0) as usize];
            let p10 = src[(y0 * src_w + x1) as usize];
            let p01 = src[(y1 * src_w + x0) as usize];
            let p11 = src[(y1 * src_w + x1) as usize];

            let mut pixel = [0u8; 4];
            for c in 0..4 {
                let v = p00[c] as f32 * (1.0-fx)*(1.0-fy) + p10[c] as f32 * fx*(1.0-fy)
                    + p01[c] as f32 * (1.0-fx)*fy + p11[c] as f32 * fx*fy;
                pixel[c] = v.clamp(0.0, 255.0) as u8;
            }
            dst[(dy * dst_w + dx) as usize] = pixel;
        }
    }
    dst
}
