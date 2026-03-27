/// CPU-side visual post-processing effects for the Ochroma engine.
///
/// Each effect operates on `&mut [[u8; 4]]` RGBA pixels in-place.
/// Depth buffer is `&[f32]` with the same length as pixels.

// ---------------------------------------------------------------------------
// Per-effect free functions
// ---------------------------------------------------------------------------

/// Blend each pixel toward `fog_color` based on its depth value.
///
/// Fog factor = clamp((depth - fog_start) * fog_density, 0, 1).
/// At factor 0 the pixel is unchanged; at factor 1 it equals fog_color.
pub fn apply_fog(
    pixels: &mut [[u8; 4]],
    depth_buffer: &[f32],
    fog_color: [u8; 3],
    fog_density: f32,
    fog_start: f32,
) {
    let fc = [fog_color[0] as f32, fog_color[1] as f32, fog_color[2] as f32];
    for (px, &d) in pixels.iter_mut().zip(depth_buffer.iter()) {
        let t = ((d - fog_start) * fog_density).clamp(0.0, 1.0);
        if t <= 0.0 {
            continue;
        }
        px[0] = lerp_u8(px[0], fc[0], t);
        px[1] = lerp_u8(px[1], fc[1], t);
        px[2] = lerp_u8(px[2], fc[2], t);
        // alpha channel unchanged
    }
}

/// Darken pixels near the edges of the frame.
///
/// The vignette factor is computed as `(1 - r^2)^2` where `r` is the
/// normalized distance from the frame centre, clamped to [0,1].
/// `strength` scales how much the factor attenuates the pixel
/// (0 = no effect, 1 = full darkening at extreme corners).
pub fn apply_vignette(
    pixels: &mut [[u8; 4]],
    width: u32,
    height: u32,
    strength: f32,
) {
    if strength <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let inv_cx = if cx > 0.0 { 1.0 / cx } else { 0.0 };
    let inv_cy = if cy > 0.0 { 1.0 / cy } else { 0.0 };

    let w = width as usize;
    let h = height as usize;
    let len = pixels.len().min(w * h);
    for i in 0..len {
        let x = (i % w) as f32;
        let y = (i / w) as f32;
        let dx = (x - cx) * inv_cx;
        let dy = (y - cy) * inv_cy;
        let r2 = (dx * dx + dy * dy).min(1.0);
        // smooth vignette: bright centre → dark edge
        let factor = (1.0 - r2) * (1.0 - r2);
        // blend between 1.0 (neutral) and factor (full dark) using strength
        let scale = 1.0 - strength * (1.0 - factor);
        let scale = scale.clamp(0.0, 1.0);
        let px = &mut pixels[i];
        px[0] = (px[0] as f32 * scale) as u8;
        px[1] = (px[1] as f32 * scale) as u8;
        px[2] = (px[2] as f32 * scale) as u8;
    }
}

/// Approximate screen-space ambient occlusion by comparing each pixel's depth
/// against its neighbours.  Pixels that sit in a "depth valley" (surrounded by
/// shallower neighbours) are darkened.
///
/// `radius` is the sampling half-extent in pixels (checked in a square window).
/// `strength` controls maximum darkening (0 = none, 1 = full black in extreme cases).
pub fn apply_ssao(
    pixels: &mut [[u8; 4]],
    depth_buffer: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    strength: f32,
) {
    if strength <= 0.0 || radius == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as isize;
    let len = pixels.len().min(w * h);

    // Build occlusion factors first so we don't mix read/write on depth_buffer.
    let mut occlusion = vec![0.0f32; len];
    let sample_count = ((2 * r + 1) * (2 * r + 1) - 1) as f32;

    for i in 0..len {
        let x = (i % w) as isize;
        let y = (i / w) as isize;
        let center_d = depth_buffer[i];
        let mut diff_sum = 0.0f32;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                let nd = depth_buffer[ni];
                // Only accumulate when the current pixel is deeper than its
                // neighbour (occluded behind a depth discontinuity).
                let diff = (center_d - nd).max(0.0);
                diff_sum += diff;
            }
        }

        // Normalize: average difference over maximum expected depth span.
        let avg = diff_sum / sample_count;
        // Sigmoid-ish clamped factor so small differences don't over-darken.
        occlusion[i] = (avg * 4.0).min(1.0);
    }

    for i in 0..len {
        let o = occlusion[i] * strength;
        if o <= 0.0 {
            continue;
        }
        let scale = 1.0 - o;
        let px = &mut pixels[i];
        px[0] = (px[0] as f32 * scale) as u8;
        px[1] = (px[1] as f32 * scale) as u8;
        px[2] = (px[2] as f32 * scale) as u8;
    }
}

/// Apply brightness, contrast, and saturation adjustments.
///
/// - `brightness`: 0–2, neutral = 1.0  (multiplies luminance)
/// - `contrast`:   0–2, neutral = 1.0  (scales around mid-grey 0.5)
/// - `saturation`: 0–2, neutral = 1.0  (blends toward greyscale at 0)
pub fn apply_color_grade(
    pixels: &mut [[u8; 4]],
    brightness: f32,
    contrast: f32,
    saturation: f32,
) {
    for px in pixels.iter_mut() {
        let mut r = px[0] as f32 / 255.0;
        let mut g = px[1] as f32 / 255.0;
        let mut b = px[2] as f32 / 255.0;

        // Brightness
        r *= brightness;
        g *= brightness;
        b *= brightness;

        // Contrast: scale around 0.5
        r = (r - 0.5) * contrast + 0.5;
        g = (g - 0.5) * contrast + 0.5;
        b = (b - 0.5) * contrast + 0.5;

        // Saturation: blend toward luminance
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        r = lum + (r - lum) * saturation;
        g = lum + (g - lum) * saturation;
        b = lum + (b - lum) * saturation;

        px[0] = (r.clamp(0.0, 1.0) * 255.0) as u8;
        px[1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
        px[2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

#[inline]
fn lerp_u8(a: u8, b: f32, t: f32) -> u8 {
    let v = a as f32 * (1.0 - t) + b * t;
    v.clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// PostProcessStack
// ---------------------------------------------------------------------------

/// Ordered stack of post-processing effects applied to a pixel buffer.
pub struct PostProcessStack {
    pub fog_enabled: bool,
    pub fog_color: [u8; 3],
    pub fog_density: f32,
    pub fog_start: f32,
    pub vignette_strength: f32,
    pub ssao_enabled: bool,
    pub ssao_radius: u32,
    pub ssao_strength: f32,
    /// 0–2, neutral = 1.0
    pub brightness: f32,
    /// 0–2, neutral = 1.0
    pub contrast: f32,
    /// 0–2, neutral = 1.0
    pub saturation: f32,
}

impl Default for PostProcessStack {
    fn default() -> Self {
        Self {
            fog_enabled: false,
            fog_color: [180, 190, 200],
            fog_density: 0.01,
            fog_start: 50.0,
            vignette_strength: 0.0,
            ssao_enabled: false,
            ssao_radius: 2,
            ssao_strength: 0.5,
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
        }
    }
}

impl PostProcessStack {
    /// Apply all enabled effects to `pixels` in sequence.
    ///
    /// Order: fog → vignette → SSAO → color grading.
    pub fn apply(&self, pixels: &mut [[u8; 4]], depth: &[f32], width: u32, height: u32) {
        if self.fog_enabled {
            apply_fog(pixels, depth, self.fog_color, self.fog_density, self.fog_start);
        }
        if self.vignette_strength > 0.0 {
            apply_vignette(pixels, width, height, self.vignette_strength);
        }
        if self.ssao_enabled {
            apply_ssao(pixels, depth, width, height, self.ssao_radius, self.ssao_strength);
        }
        if self.brightness != 1.0 || self.contrast != 1.0 || self.saturation != 1.0 {
            apply_color_grade(pixels, self.brightness, self.contrast, self.saturation);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a flat RGBA buffer filled with a single colour.
    fn solid(color: [u8; 4], n: usize) -> Vec<[u8; 4]> {
        vec![color; n]
    }

    // ── Fog ──────────────────────────────────────────────────────────────────

    #[test]
    fn fog_darkens_distant_pixels() {
        // A bright white pixel very far away should blend strongly toward black fog.
        let mut pixels = solid([255, 255, 255, 255], 1);
        let depth = vec![200.0f32]; // well beyond fog_start=10
        apply_fog(&mut pixels, &depth, [0, 0, 0], 0.1, 10.0);
        // factor = (200-10)*0.1 = 19 → clamped 1.0 → pixel should be [0,0,0]
        assert_eq!(pixels[0][0], 0);
        assert_eq!(pixels[0][1], 0);
        assert_eq!(pixels[0][2], 0);
        assert_eq!(pixels[0][3], 255, "alpha should be unchanged");
    }

    #[test]
    fn fog_leaves_near_pixels_unchanged() {
        let mut pixels = solid([100, 150, 200, 255], 1);
        let original = pixels[0];
        let depth = vec![5.0f32]; // below fog_start=10
        apply_fog(&mut pixels, &depth, [255, 0, 0], 0.1, 10.0);
        assert_eq!(pixels[0], original);
    }

    // ── Vignette ─────────────────────────────────────────────────────────────

    #[test]
    fn vignette_darkens_corners_more_than_centre() {
        // 3×3 image, all white.
        let n = 9;
        let mut pixels = solid([200, 200, 200, 255], n);
        apply_vignette(&mut pixels, 3, 3, 1.0);

        let centre = pixels[4]; // pixel (1,1)
        let corner = pixels[0]; // pixel (0,0)

        // Centre should be brighter (less darkened) than corner.
        assert!(centre[0] > corner[0], "centre {centre:?} should be brighter than corner {corner:?}");
    }

    #[test]
    fn vignette_zero_strength_is_noop() {
        let mut pixels = solid([128, 128, 128, 255], 9);
        let before = pixels.clone();
        apply_vignette(&mut pixels, 3, 3, 0.0);
        assert_eq!(pixels, before);
    }

    // ── SSAO ─────────────────────────────────────────────────────────────────

    #[test]
    fn ssao_darkens_depth_discontinuity() {
        // 3×3 image: all depths = 0.0 except the centre pixel which is 1.0.
        // The centre is "behind" an edge — its neighbours are shallower, so it
        // should be darkened.
        let n = 9;
        let mut pixels = solid([200, 200, 200, 255], n);
        let mut depth = vec![0.0f32; n];
        depth[4] = 1.0; // centre is deeper

        apply_ssao(&mut pixels, &depth, 3, 3, 1, 1.0);

        let centre = pixels[4];
        let edge = pixels[1]; // top-centre, depth=0, should be unaffected

        assert!(
            centre[0] < edge[0],
            "centre (deep) {centre:?} should be darker than shallow edge {edge:?}"
        );
    }

    #[test]
    fn ssao_zero_strength_is_noop() {
        let mut pixels = solid([200, 200, 200, 255], 9);
        let before = pixels.clone();
        let depth = vec![0.0f32; 9];
        apply_ssao(&mut pixels, &depth, 3, 3, 1, 0.0);
        assert_eq!(pixels, before);
    }

    // ── Color grading ─────────────────────────────────────────────────────────

    #[test]
    fn color_grade_brightness_increase_lightens_pixels() {
        let mut pixels = solid([100, 100, 100, 255], 1);
        apply_color_grade(&mut pixels, 2.0, 1.0, 1.0);
        // 100/255 * 2.0 * 255 ≈ 200
        assert!(pixels[0][0] > 100, "brightness=2.0 should increase pixel value");
    }

    #[test]
    fn color_grade_neutral_settings_are_noop() {
        let mut pixels = solid([80, 120, 200, 255], 1);
        let before = pixels.clone();
        apply_color_grade(&mut pixels, 1.0, 1.0, 1.0);
        // Allow ±1 for rounding.
        for ch in 0..3 {
            let diff = (pixels[0][ch] as i16 - before[0][ch] as i16).abs();
            assert!(diff <= 1, "channel {ch}: expected ~{} got {}", before[0][ch], pixels[0][ch]);
        }
        assert_eq!(pixels[0][3], before[0][3], "alpha should be unchanged");
    }

    // ── Stack-level ───────────────────────────────────────────────────────────

    #[test]
    fn empty_pixels_unaffected() {
        let mut pixels: Vec<[u8; 4]> = vec![];
        let depth: Vec<f32> = vec![];
        let stack = PostProcessStack {
            fog_enabled: true,
            ssao_enabled: true,
            vignette_strength: 1.0,
            brightness: 2.0,
            ..Default::default()
        };
        // Must not panic on empty input.
        stack.apply(&mut pixels, &depth, 0, 0);
        assert!(pixels.is_empty());
    }

    #[test]
    fn stack_default_is_noop() {
        let mut pixels = solid([80, 120, 200, 255], 4);
        let before = pixels.clone();
        let depth = vec![1.0f32; 4];
        // Default stack has all effects disabled / neutral.
        PostProcessStack::default().apply(&mut pixels, &depth, 2, 2);
        assert_eq!(pixels, before);
    }
}
