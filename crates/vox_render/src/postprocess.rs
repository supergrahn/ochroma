/// Tone mapping method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapping {
    None,
    Reinhard,
    ACES,
}

/// Post-processing pipeline configuration.
#[derive(Debug, Clone)]
pub struct PostProcessPipeline {
    pub tone_mapping: ToneMapping,
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub vignette_enabled: bool,
    pub vignette_strength: f32,
}

impl Default for PostProcessPipeline {
    fn default() -> Self {
        Self {
            tone_mapping: ToneMapping::ACES,
            bloom_enabled: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.3,
            vignette_enabled: false,
            vignette_strength: 0.5,
        }
    }
}

impl PostProcessPipeline {
    /// Apply all enabled post-processing effects in order.
    pub fn apply(&self, pixels: &mut [[f32; 4]], width: usize, height: usize) {
        if self.bloom_enabled {
            apply_bloom(pixels, width, height, self.bloom_threshold, self.bloom_intensity);
        }
        apply_tone_mapping(pixels, self.tone_mapping);
        if self.vignette_enabled {
            apply_vignette(pixels, width, height, self.vignette_strength);
        }
    }
}

/// Apply tone mapping to each pixel in the buffer.
pub fn apply_tone_mapping(pixels: &mut [[f32; 4]], method: ToneMapping) {
    match method {
        ToneMapping::None => {}
        ToneMapping::Reinhard => {
            for px in pixels.iter_mut() {
                for c in 0..3 {
                    px[c] = px[c] / (1.0 + px[c]);
                }
            }
        }
        ToneMapping::ACES => {
            for px in pixels.iter_mut() {
                for c in 0..3 {
                    px[c] = aces(px[c]);
                }
            }
        }
    }
}

/// ACES filmic tone mapping curve.
fn aces(x: f32) -> f32 {
    let num = x * (2.51 * x + 0.03);
    let den = x * (2.43 * x + 0.59) + 0.14;
    (num / den).clamp(0.0, 1.0)
}

/// Extract bright pixels, blur them, and add back to simulate bloom.
pub fn apply_bloom(
    pixels: &mut [[f32; 4]],
    width: usize,
    height: usize,
    threshold: f32,
    intensity: f32,
) {
    let len = width * height;
    if pixels.len() != len {
        return;
    }

    // Extract bright pixels.
    let mut bright: Vec<[f32; 4]> = pixels
        .iter()
        .map(|px| {
            let lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            if lum > threshold {
                *px
            } else {
                [0.0, 0.0, 0.0, 0.0]
            }
        })
        .collect();

    // Simple box blur (horizontal then vertical, radius=2).
    let radius: i32 = 2;
    let mut temp = vec![[0.0f32; 4]; len];

    // Horizontal pass.
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0f32; 3];
            let mut count = 0.0f32;
            for dx in -radius..=radius {
                let nx = x as i32 + dx;
                if nx >= 0 && (nx as usize) < width {
                    let idx = y * width + nx as usize;
                    sum[0] += bright[idx][0];
                    sum[1] += bright[idx][1];
                    sum[2] += bright[idx][2];
                    count += 1.0;
                }
            }
            let idx = y * width + x;
            temp[idx] = [sum[0] / count, sum[1] / count, sum[2] / count, 0.0];
        }
    }

    // Vertical pass.
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0f32; 3];
            let mut count = 0.0f32;
            for dy in -radius..=radius {
                let ny = y as i32 + dy;
                if ny >= 0 && (ny as usize) < height {
                    let idx = ny as usize * width + x;
                    sum[0] += temp[idx][0];
                    sum[1] += temp[idx][1];
                    sum[2] += temp[idx][2];
                    count += 1.0;
                }
            }
            let idx = y * width + x;
            bright[idx] = [sum[0] / count, sum[1] / count, sum[2] / count, 0.0];
        }
    }

    // Add bloom back to original.
    for (px, bl) in pixels.iter_mut().zip(bright.iter()) {
        px[0] += bl[0] * intensity;
        px[1] += bl[1] * intensity;
        px[2] += bl[2] * intensity;
    }
}

/// Apply a vignette effect, darkening pixels based on distance from center.
pub fn apply_vignette(
    pixels: &mut [[f32; 4]],
    width: usize,
    height: usize,
    strength: f32,
) {
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let max_dist = (cx * cx + cy * cy).sqrt();

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let factor = 1.0 - strength * (dist / max_dist);
            let factor = factor.clamp(0.0, 1.0);

            let idx = y * width + x;
            if idx < pixels.len() {
                pixels[idx][0] *= factor;
                pixels[idx][1] *= factor;
                pixels[idx][2] *= factor;
            }
        }
    }
}
