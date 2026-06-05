//! GameHud — composes the in-game heads-up display as draw commands.
//!
//! This is the CPU/software-renderer path: it records `DrawCmd`s into a
//! [`VelloCtxCpu`] which a software rasteriser then composites into the game
//! frame via [`VelloCtxCpu::rasterize_into`].
//!
//! Layout here is done with plain arithmetic so the HUD works under the crate's
//! DEFAULT features. `taffy` is the layout engine for the windowed-GPU path and
//! lives behind the optional `game-ui` feature — it is intentionally NOT used on
//! this CPU path so the software renderer has no extra feature dependency.

use crate::spectral_hud::{SpectralHUD, SpectralRadianceCache};
use crate::vello_ctx::VelloCtxCpu;

/// Composes the full in-game HUD (spectral bars + orb progress) for the CPU
/// software-renderer path.
pub struct GameHud {
    width:  u32,
    height: u32,
    /// Pixel margin from the screen edges for anchored elements.
    margin: f32,
}

// --- Spectral bar geometry (matches SpectralHUD::render_cpu) ---------------
const BARS_TOTAL_WIDTH: f32 = 160.0;
const BARS_MAX_HEIGHT:  f32 = 60.0;
/// Padding between the backdrop panel edge and the bars inside it.
const PANEL_PAD: f32 = 8.0;

// --- Orb progress bar geometry ---------------------------------------------
const ORB_BAR_WIDTH:  f32 = 240.0;
const ORB_BAR_HEIGHT: f32 = 18.0;

impl GameHud {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, margin: 16.0 }
    }

    /// Records the full HUD into `ctx`:
    /// - a translucent backdrop panel bottom-left,
    /// - the 16-band spectral bars (bottom-left, inside the panel),
    /// - an orb-progress bar top-left (background track + proportional fill).
    ///
    /// `orbs_collected` is clamped to `orbs_total`; if `orbs_total == 0` the
    /// fill fraction is treated as 0.
    pub fn compose(
        &self,
        ctx:            &mut VelloCtxCpu,
        bands:          &SpectralRadianceCache,
        orbs_collected: u32,
        orbs_total:     u32,
    ) {
        self.compose_spectral_bars(ctx, bands);
        self.compose_orb_bar(ctx, orbs_collected, orbs_total);
    }

    /// Bottom-left translucent panel + 16-band spectral bars.
    fn compose_spectral_bars(&self, ctx: &mut VelloCtxCpu, bands: &SpectralRadianceCache) {
        let panel_w = BARS_TOTAL_WIDTH + PANEL_PAD * 2.0;
        let panel_h = BARS_MAX_HEIGHT + PANEL_PAD * 2.0;
        // Anchor the panel to the bottom-left corner.
        let panel_x = self.margin;
        let panel_y = self.height as f32 - self.margin - panel_h;

        // Translucent dark backdrop.
        ctx.fill_rect([panel_x, panel_y, panel_w, panel_h], [0.0, 0.0, 0.0, 0.55]);

        // Bars start inside the panel padding.
        let bars_pos = [panel_x + PANEL_PAD, panel_y + PANEL_PAD];
        SpectralHUD::render_cpu(ctx, bands, bars_pos);
    }

    /// Top-left orb-progress bar: a background track and a proportional fill.
    fn compose_orb_bar(&self, ctx: &mut VelloCtxCpu, collected: u32, total: u32) {
        let x = self.margin;
        let y = self.margin;

        // Background track (dark translucent).
        ctx.fill_rect([x, y, ORB_BAR_WIDTH, ORB_BAR_HEIGHT], [0.05, 0.05, 0.08, 0.7]);

        let frac = if total == 0 {
            0.0
        } else {
            (collected.min(total) as f32) / (total as f32)
        };
        let fill_w = (ORB_BAR_WIDTH * frac).max(0.0);
        if fill_w > 0.0 {
            // Warm amber fill so it reads against the dark track.
            ctx.fill_rect([x, y, fill_w, ORB_BAR_HEIGHT], [1.0, 0.78, 0.2, 0.95]);
        }
    }

    /// The orb-bar track rect `[x, y, w, h]` (background), useful for tests and
    /// for callers wanting hit-test geometry.
    pub fn orb_bar_track_rect(&self) -> [f32; 4] {
        [self.margin, self.margin, ORB_BAR_WIDTH, ORB_BAR_HEIGHT]
    }

    /// The filled portion rect `[x, y, w, h]` for the given orb counts.
    pub fn orb_bar_fill_rect(&self, collected: u32, total: u32) -> [f32; 4] {
        let frac = if total == 0 {
            0.0
        } else {
            (collected.min(total) as f32) / (total as f32)
        };
        [self.margin, self.margin, (ORB_BAR_WIDTH * frac).max(0.0), ORB_BAR_HEIGHT]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vello_ctx::DrawCmd;

    const W: u32 = 1280;
    const H: u32 = 720;

    #[test]
    fn compose_paints_bottom_left_bar_region() {
        let hud   = GameHud::new(W, H);
        let bands = SpectralRadianceCache::from_f32([1.0; 16]);
        let mut ctx = VelloCtxCpu::new(W, H);
        hud.compose(&mut ctx, &bands, 5, 10);

        let mut pixels = vec![[0u8, 0, 0, 255]; (W * H) as usize];
        ctx.rasterize_into(&mut pixels, W, H);

        // Bottom-left region: x in [16, 200), y in [H-100, H-16).
        let mut non_black = 0usize;
        for y in (H - 100)..(H - 16) {
            for x in 16..200u32 {
                let p = pixels[(y * W + x) as usize];
                if p[0] != 0 || p[1] != 0 || p[2] != 0 {
                    non_black += 1;
                }
            }
        }
        println!("non_black in bar region = {}", non_black);
        assert!(non_black > 0, "bottom-left bar region should have painted pixels");
    }

    #[test]
    fn orb_fill_is_half_track_width_at_five_of_ten() {
        let hud   = GameHud::new(W, H);
        let track = hud.orb_bar_track_rect();
        let fill  = hud.orb_bar_fill_rect(5, 10);
        println!("track_w={} fill_w={}", track[2], fill[2]);
        assert!(
            (fill[2] - track[2] * 0.5).abs() < 1e-3,
            "fill width {} should be half track width {}",
            fill[2], track[2],
        );
    }

    #[test]
    fn orb_fill_clamps_and_handles_zero_total() {
        let hud = GameHud::new(W, H);
        let track = hud.orb_bar_track_rect();
        // Over-collected: clamps to full track width.
        let full = hud.orb_bar_fill_rect(99, 10);
        assert!((full[2] - track[2]).abs() < 1e-3, "over-collected fill should clamp to full track");
        // Zero total: empty fill.
        let empty = hud.orb_bar_fill_rect(3, 0);
        assert!(empty[2].abs() < 1e-6, "zero total should produce empty fill, got {}", empty[2]);
    }

    #[test]
    fn compose_records_half_width_orb_fill_command() {
        let hud   = GameHud::new(W, H);
        let bands = SpectralRadianceCache::from_f32([0.5; 16]);
        let mut ctx = VelloCtxCpu::new(W, H);
        hud.compose(&mut ctx, &bands, 5, 10);

        // The orb fill command should be a FillRect at the top-left margin with
        // width ~= half the track width.
        let track_w = hud.orb_bar_track_rect()[2];
        let found = ctx.commands().iter().any(|c| match c {
            DrawCmd::FillRect { rect, .. } => {
                (rect[0] - 16.0).abs() < 1e-3
                    && (rect[1] - 16.0).abs() < 1e-3
                    && (rect[2] - track_w * 0.5).abs() < 1e-3
            }
        });
        assert!(found, "expected a recorded orb-fill FillRect at half track width");
    }

    #[test]
    fn orb_fill_rendered_pixels_span_half_track() {
        let hud   = GameHud::new(W, H);
        let bands = SpectralRadianceCache::from_f32([0.0; 16]);
        let mut ctx = VelloCtxCpu::new(W, H);
        hud.compose(&mut ctx, &bands, 5, 10);

        let mut pixels = vec![[0u8, 0, 0, 255]; (W * H) as usize];
        ctx.rasterize_into(&mut pixels, W, H);

        // Scan the orb-bar row for the bright amber fill run vs the darker track.
        let track = hud.orb_bar_track_rect();
        let row_y = (track[1] + track[3] * 0.5) as u32;
        let x_start = track[0] as u32;
        let x_end   = (track[0] + track[2]) as u32;

        // Amber fill has high red channel; dark track does not.
        let mut bright_run = 0usize;
        for x in x_start..x_end {
            let p = pixels[(row_y * W + x) as usize];
            if p[0] > 150 {
                bright_run += 1;
            }
        }
        let track_px = (x_end - x_start) as usize;
        println!("bright_run={} track_px={}", bright_run, track_px);
        // Fill should cover roughly half the track (allow a few px slack).
        let half = track_px / 2;
        assert!(
            (bright_run as i64 - half as i64).abs() <= 3,
            "bright fill run {} should be ~half the track {} px",
            bright_run, track_px,
        );
    }
}
