//! SpectralHUD — live 16-band spectral energy bar display.

use crate::vello_ctx::VelloCtxCpu;

/// Minimal spectral radiance cache for the UI layer.
#[derive(Debug, Clone, Copy)]
pub struct SpectralRadianceCache {
    pub band_energy: [f32; 16],
}

impl Default for SpectralRadianceCache {
    fn default() -> Self { Self { band_energy: [0.0; 16] } }
}

impl SpectralRadianceCache {
    /// Build a cache from a live `[f32; 16]` band-energy source.
    ///
    /// Values are stored as-is; `bar_rects` clamps to `[0, 1]` at draw time.
    pub fn from_f32(band_energy: [f32; 16]) -> Self {
        Self { band_energy }
    }

    /// Build a cache from a live `[u16; 16]` quantized band-energy source.
    ///
    /// Each band is dequantized as `value / u16::MAX`, so `u16::MAX` maps to
    /// `1.0` (full energy) and `0` maps to `0.0`. This is the natural input
    /// when band energy arrives quantized from a GPU readback or over the wire.
    pub fn from_u16(bands: [u16; 16]) -> Self {
        let mut band_energy = [0.0f32; 16];
        for b in 0..16 {
            band_energy[b] = bands[b] as f32 / u16::MAX as f32;
        }
        Self { band_energy }
    }
}

pub struct SpectralHUD;

const BAND_COLORS: [[f32; 4]; 16] = [
    [0.58, 0.00, 0.83, 1.0], // Band 0  — violet
    [0.35, 0.00, 1.00, 1.0], // Band 1  — violet-blue
    [0.00, 0.00, 1.00, 1.0], // Band 2  — blue-violet
    [0.00, 0.35, 1.00, 1.0], // Band 3  — blue
    [0.00, 0.60, 1.00, 1.0], // Band 4  — blue-cyan
    [0.00, 0.85, 0.85, 1.0], // Band 5  — cyan
    [0.00, 0.90, 0.40, 1.0], // Band 6  — cyan-green
    [0.30, 0.90, 0.00, 1.0], // Band 7  — green-yellow
    [1.00, 0.95, 0.00, 1.0], // Band 8  — yellow
    [1.00, 0.75, 0.00, 1.0], // Band 9  — yellow-orange
    [1.00, 0.55, 0.00, 1.0], // Band 10 — orange
    [1.00, 0.30, 0.00, 1.0], // Band 11 — orange-red
    [1.00, 0.10, 0.00, 1.0], // Band 12 — red
    [0.90, 0.00, 0.00, 1.0], // Band 13 — deep red
    [0.80, 0.00, 0.00, 1.0], // Band 14 — near-IR/red
    [0.70, 0.00, 0.00, 1.0], // Band 15 — near-IR
];

const BAR_GAP: f32 = 2.0;

impl SpectralHUD {
    pub fn band_colors() -> [[f32; 4]; 16] {
        BAND_COLORS
    }

    pub fn bar_rects(
        energy:      [f32; 16],
        pos:         [f32; 2],
        total_width: f32,
        max_height:  f32,
    ) -> Vec<[f32; 4]> {
        let bar_w = (total_width / 16.0 - BAR_GAP).max(1.0);
        (0..16usize).map(|b| {
            let h = (energy[b].clamp(0.0, 1.0) * max_height).max(0.0);
            let x = pos[0] + b as f32 * (bar_w + BAR_GAP);
            let y = pos[1] + (max_height - h);
            [x, y, bar_w, h]
        }).collect()
    }

    /// Bar geometry for a live `[u16; 16]` quantized band-energy source.
    ///
    /// Convenience over `bar_rects`: dequantizes each band (`value / u16::MAX`)
    /// before laying out bars, so a higher `u16` band yields a taller bar.
    /// Returns one `[x, y, w, h]` per band (16 total).
    pub fn bar_rects_u16(
        bands:       [u16; 16],
        pos:         [f32; 2],
        total_width: f32,
        max_height:  f32,
    ) -> Vec<[f32; 4]> {
        let cache = SpectralRadianceCache::from_u16(bands);
        Self::bar_rects(cache.band_energy, pos, total_width, max_height)
    }

    pub fn render_cpu(
        ctx:   &mut VelloCtxCpu,
        cache: &SpectralRadianceCache,
        pos:   [f32; 2],
    ) {
        let max_height  = 60.0;
        let total_width = 160.0;
        let bars        = Self::bar_rects(cache.band_energy, pos, total_width, max_height);
        let colors      = Self::band_colors();

        for (b, rect) in bars.iter().enumerate() {
            let bg_rect = [rect[0], pos[1], rect[2], max_height];
            ctx.fill_rect(bg_rect, [0.1, 0.1, 0.1, 0.7]);
            ctx.fill_rect(*rect, colors[b]);
        }
    }

    #[cfg(feature = "game-ui")]
    pub fn render(
        ctx:   &mut crate::vello_ctx::VelloCtx,
        cache: &SpectralRadianceCache,
        pos:   [f32; 2],
    ) {
        let max_height  = 60.0;
        let total_width = 160.0;
        let bars        = Self::bar_rects(cache.band_energy, pos, total_width, max_height);
        let colors      = Self::band_colors();

        for (b, rect) in bars.iter().enumerate() {
            let bg_rect = [rect[0], pos[1], rect[2], max_height];
            ctx.fill_rect(bg_rect, [0.1, 0.1, 0.1, 0.7]);
            ctx.fill_rect(*rect, colors[b]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(values: [f32; 16]) -> SpectralRadianceCache {
        SpectralRadianceCache { band_energy: values }
    }

    #[test]
    fn bar_heights_proportional_to_energy() {
        let cache = make_cache([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2]);
        let bars  = SpectralHUD::bar_rects(cache.band_energy, [0.0, 0.0], 160.0, 100.0);
        let heights: Vec<f32> = bars.iter().map(|r| r[3]).collect();
        println!("band8_h={} band0_h={}", heights[8], heights[0]);
        assert!(heights[8] > heights[0], "band8 h={} band0 h={}", heights[8], heights[0]);
        assert!(heights[8] > heights[1], "band8 h={} band1 h={}", heights[8], heights[1]);
    }

    #[test]
    fn zero_energy_bar_has_zero_height() {
        let cache = make_cache([0.0; 16]);
        let bars  = SpectralHUD::bar_rects(cache.band_energy, [0.0, 0.0], 160.0, 100.0);
        for bar in &bars {
            assert!((bar[3]).abs() < 1e-6, "zero energy bar should have zero height, got {}", bar[3]);
        }
    }

    #[test]
    fn full_energy_bar_fills_max_height() {
        let cache = make_cache([1.0; 16]);
        let bars  = SpectralHUD::bar_rects(cache.band_energy, [0.0, 0.0], 160.0, 100.0);
        for bar in &bars {
            assert!((bar[3] - 100.0).abs() < 1e-5, "full-energy bar should be max_height, got {}", bar[3]);
        }
    }

    #[test]
    fn band_colors_violet_to_red_gradient() {
        let colors = SpectralHUD::band_colors();
        println!("band0_blue={} band15_red={}", colors[0][2], colors[15][0]);
        assert!(colors[0][2] > 0.5, "band 0 should be violet (high blue), b={}", colors[0][2]);
        assert!(colors[15][0] > 0.5, "band 15 should be red (high red), r={}", colors[15][0]);
    }

    #[test]
    fn bar_rects_returns_exactly_16_rects() {
        let bars = SpectralHUD::bar_rects([0.5; 16], [0.0, 0.0], 160.0, 100.0);
        assert_eq!(bars.len(), 16);
    }

    #[test]
    fn render_cpu_emits_16_fill_rect_commands() {
        let cache = make_cache([0.5; 16]);
        let mut ctx = crate::vello_ctx::VelloCtxCpu::new(800, 600);
        SpectralHUD::render_cpu(&mut ctx, &cache, [10.0, 10.0]);
        assert!(ctx.commands().len() >= 16, "expected ≥16 draw commands, got {}", ctx.commands().len());
    }

    #[test]
    fn from_u16_dequantizes_full_scale_to_one() {
        // u16::MAX is full energy -> 1.0; 0 -> 0.0; midpoint -> ~0.5.
        let cache = SpectralRadianceCache::from_u16([
            u16::MAX, 0, 32768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        assert!((cache.band_energy[0] - 1.0).abs() < 1e-4, "full-scale u16 should map to 1.0, got {}", cache.band_energy[0]);
        assert!(cache.band_energy[1].abs() < 1e-6, "zero u16 should map to 0.0, got {}", cache.band_energy[1]);
        assert!((cache.band_energy[2] - 0.5).abs() < 1e-2, "mid u16 should map to ~0.5, got {}", cache.band_energy[2]);
    }

    #[test]
    fn bar_rects_u16_high_band_taller_than_low_band() {
        // Live quantized input: band 12 is full energy, band 3 is near-zero.
        let mut bands = [0u16; 16];
        bands[3]  = 256;        // very low energy
        bands[12] = u16::MAX;   // full energy
        let bars = SpectralHUD::bar_rects_u16(bands, [0.0, 0.0], 160.0, 100.0);
        assert_eq!(bars.len(), 16, "expected 16 bars, got {}", bars.len());
        let h_high = bars[12][3];
        let h_low  = bars[3][3];
        println!("u16 band12_h={} band3_h={}", h_high, h_low);
        assert!(h_high > h_low, "full-scale band12 h={} should be taller than low band3 h={}", h_high, h_low);
        // Full-scale band must reach (within epsilon) the max height.
        assert!((h_high - 100.0).abs() < 1e-3, "full-scale u16 band should fill max_height, got {}", h_high);
    }

    #[test]
    fn bar_rects_u16_matches_dequantized_f32_path() {
        // The u16 convenience path must agree with dequantize-then-bar_rects.
        let bands: [u16; 16] = [
            0, 4096, 8192, 12288, 16384, 20480, 24576, 28672,
            u16::MAX, 49152, 40960, 32768, 24576, 16384, 8192, 4096,
        ];
        let via_u16 = SpectralHUD::bar_rects_u16(bands, [5.0, 7.0], 160.0, 100.0);
        let cache   = SpectralRadianceCache::from_u16(bands);
        let via_f32 = SpectralHUD::bar_rects(cache.band_energy, [5.0, 7.0], 160.0, 100.0);
        for b in 0..16 {
            assert!((via_u16[b][3] - via_f32[b][3]).abs() < 1e-4,
                "band {} height mismatch: u16 path={} f32 path={}", b, via_u16[b][3], via_f32[b][3]);
        }
    }

    // --- Real GPU (Vello) SpectralHUD pixel test -------------------------
    //
    // Renders the live SpectralHUD through the actual vello::Renderer on a GPU
    // and reads pixels back. Self-skips when no adapter is present. Asserts
    // computed pixel outcomes: band 0 (violet, high blue) and band 15 (red,
    // high red) land in their bar columns with the expected dominant channel,
    // and the HUD region contains many distinct colours (the spectral gradient
    // actually rendered, not a flat fill).
    #[cfg(feature = "game-ui")]
    #[test]
    fn vello_gpu_spectral_hud_renders_band_hues_and_many_colors() {
        use crate::vello_ctx::VelloCtx;

        let w = 256u32;
        let h = 128u32;
        let Some(mut ctx) = VelloCtx::new_headless(w, h) else {
            eprintln!("[vello] no GPU adapter — skipping SpectralHUD GPU test");
            return;
        };

        // Full energy so every bar reaches max height.
        let cache = SpectralRadianceCache::from_f32([1.0; 16]);
        let pos = [8.0f32, 8.0f32];
        ctx.begin_frame();
        SpectralHUD::render(&mut ctx, &cache, pos);
        let pixels = ctx.render_to_rgba().expect("gpu hud render");
        assert_eq!(pixels.len(), (w * h) as usize);

        // Geometry of the bars (must match render()'s internal constants).
        let total_width = 160.0f32;
        let max_height  = 60.0f32;
        let bars = SpectralHUD::bar_rects([1.0; 16], pos, total_width, max_height);

        // Sample the centre of band 0's bar (violet: blue >> red) and band 15's
        // bar (red: red >> blue). Sample near the bottom where the bar is solid.
        let sample = |rect: [f32; 4]| -> [u8; 4] {
            let sx = (rect[0] + rect[2] * 0.5).round() as u32;
            let sy = (rect[1] + rect[3] - 4.0).round() as u32;
            pixels[(sy.min(h - 1) * w + sx.min(w - 1)) as usize]
        };
        let b0 = sample(bars[0]);   // violet
        let b15 = sample(bars[15]); // red
        println!("[vello] hud band0={:?} band15={:?}", b0, b15);
        assert!(b0[2] > b0[0] + 40, "band 0 should be violet (blue>>red), got {:?}", b0);
        assert!(b15[0] > b15[2] + 40, "band 15 should be red (red>>blue), got {:?}", b15);

        // Count distinct colours inside the HUD region — the 16-band gradient
        // plus the dark backdrop must yield many unique colours.
        let mut seen = std::collections::HashSet::new();
        let x0 = pos[0] as u32;
        let x1 = (pos[0] + total_width) as u32;
        let y0 = pos[1] as u32;
        let y1 = (pos[1] + max_height) as u32;
        let mut non_background = 0usize;
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                let p = pixels[(y * w + x) as usize];
                if p[0] > 16 || p[1] > 16 || p[2] > 16 {
                    non_background += 1;
                }
                // Quantise to 5 bits/channel so AA dithering doesn't inflate the count.
                seen.insert((p[0] >> 3, p[1] >> 3, p[2] >> 3));
            }
        }
        let distinct = seen.len();
        println!(
            "[vello] HUD {}x{} non_background_px={} distinct_colors={}",
            w, h, non_background, distinct,
        );
        assert!(non_background > 2000, "HUD region should be mostly painted, got {non_background}");
        // 16 bars of distinct hues -> comfortably more than 8 quantised colours.
        assert!(distinct >= 12, "expected >=12 distinct colours, got {distinct}");
    }
}
