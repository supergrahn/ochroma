/// Spectral-aware denoiser that reduces alpha blending shimmer.
/// Uses edge-aware bilateral filtering on the spectral framebuffer.
pub struct SpectralDenoiser {
    pub strength: f32,      // 0.0 = no denoising, 1.0 = maximum
    pub spatial_sigma: f32, // spatial kernel width in pixels
    pub spectral_sigma: f32, // spectral similarity threshold
}

impl SpectralDenoiser {
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            spatial_sigma: 2.0,
            spectral_sigma: 0.1,
        }
    }

    /// Denoise an RGBA8 framebuffer in-place.
    /// Uses bilateral filter: weight = exp(-spatial_dist²/2σ_s²) × exp(-color_dist²/2σ_c²)
    pub fn denoise(&self, pixels: &mut [[u8; 4]], width: u32, height: u32) {
        if self.strength <= 0.0 { return; }

        let radius = (self.spatial_sigma * 2.0).ceil() as i32;
        let original = pixels.to_vec();

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let idx = (y * width as i32 + x) as usize;
                let center = original[idx];

                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }

                        let nidx = (ny * width as i32 + nx) as usize;
                        let neighbor = original[nidx];

                        // Spatial weight
                        let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                        let spatial_w = (-spatial_dist_sq / (2.0 * self.spatial_sigma * self.spatial_sigma)).exp();

                        // Color/spectral weight (bilateral)
                        let color_dist_sq = (center[0] as f32 - neighbor[0] as f32).powi(2)
                            + (center[1] as f32 - neighbor[1] as f32).powi(2)
                            + (center[2] as f32 - neighbor[2] as f32).powi(2);
                        let color_w = (-color_dist_sq / (2.0 * self.spectral_sigma * self.spectral_sigma * 255.0 * 255.0)).exp();

                        let w = spatial_w * color_w;
                        sum_r += neighbor[0] as f32 * w;
                        sum_g += neighbor[1] as f32 * w;
                        sum_b += neighbor[2] as f32 * w;
                        weight_sum += w;
                    }
                }

                if weight_sum > 0.0 {
                    let blend = self.strength;
                    pixels[idx][0] = (center[0] as f32 * (1.0 - blend) + (sum_r / weight_sum) * blend).clamp(0.0, 255.0) as u8;
                    pixels[idx][1] = (center[1] as f32 * (1.0 - blend) + (sum_g / weight_sum) * blend).clamp(0.0, 255.0) as u8;
                    pixels[idx][2] = (center[2] as f32 * (1.0 - blend) + (sum_b / weight_sum) * blend).clamp(0.0, 255.0) as u8;
                    // Alpha unchanged
                }
            }
        }
    }
}
