/// Upscaling quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleQuality {
    /// No upscaling — render at native resolution.
    Native,
    /// Render at 77% resolution, upscale to native.
    Quality,
    /// Render at 67% resolution, upscale to native.
    Balanced,
    /// Render at 50% resolution, upscale to native.
    Performance,
    /// Render at 33% resolution, upscale to native.
    UltraPerformance,
}

impl UpscaleQuality {
    /// Scale factor: internal resolution / display resolution.
    pub fn scale_factor(&self) -> f32 {
        match self {
            Self::Native => 1.0,
            Self::Quality => 0.77,
            Self::Balanced => 0.67,
            Self::Performance => 0.50,
            Self::UltraPerformance => 0.33,
        }
    }

    /// Calculate internal render resolution from display resolution.
    pub fn internal_resolution(&self, display_width: u32, display_height: u32) -> (u32, u32) {
        let scale = self.scale_factor();
        (
            (display_width as f32 * scale).round() as u32,
            (display_height as f32 * scale).round() as u32,
        )
    }

    /// Estimated performance multiplier (higher = faster).
    pub fn performance_multiplier(&self) -> f32 {
        let s = self.scale_factor();
        1.0 / (s * s) // pixel count scales quadratically
    }
}

/// Simple bilinear upscaler (placeholder for future TAA/DLSS/FSR integration).
pub fn bilinear_upscale(
    input: &[[u8; 4]],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<[u8; 4]> {
    let mut output = vec![[0u8; 4]; (dst_width * dst_height) as usize];

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let sx = dx as f32 * src_width as f32 / dst_width as f32;
            let sy = dy as f32 * src_height as f32 / dst_height as f32;

            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);

            let fx = sx - sx.floor();
            let fy = sy - sy.floor();

            let p00 = input[(y0 * src_width + x0) as usize];
            let p10 = input[(y0 * src_width + x1) as usize];
            let p01 = input[(y1 * src_width + x0) as usize];
            let p11 = input[(y1 * src_width + x1) as usize];

            let mut result = [0u8; 4];
            for c in 0..4 {
                let v00 = p00[c] as f32;
                let v10 = p10[c] as f32;
                let v01 = p01[c] as f32;
                let v11 = p11[c] as f32;

                let v0 = v00 + (v10 - v00) * fx;
                let v1 = v01 + (v11 - v01) * fx;
                let v = v0 + (v1 - v0) * fy;

                result[c] = v.clamp(0.0, 255.0) as u8;
            }

            output[(dy * dst_width + dx) as usize] = result;
        }
    }

    output
}

/// Manages the upscaling pipeline.
pub struct UpscaleManager {
    pub quality: UpscaleQuality,
    pub display_width: u32,
    pub display_height: u32,
}

impl UpscaleManager {
    pub fn new(display_width: u32, display_height: u32, quality: UpscaleQuality) -> Self {
        Self { quality, display_width, display_height }
    }

    /// Get the resolution the renderer should use internally.
    pub fn render_resolution(&self) -> (u32, u32) {
        self.quality.internal_resolution(self.display_width, self.display_height)
    }

    /// Upscale a rendered frame from internal to display resolution.
    pub fn upscale(&self, pixels: &[[u8; 4]], src_width: u32, src_height: u32) -> Vec<[u8; 4]> {
        if self.quality == UpscaleQuality::Native {
            return pixels.to_vec();
        }
        bilinear_upscale(pixels, src_width, src_height, self.display_width, self.display_height)
    }
}
