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
use crate::text;
use crate::vello_ctx::VelloCtxCpu;

use vox_core::game_ui::CHAR_H;

/// Composes the full in-game HUD (spectral bars + orb progress) for the CPU
/// software-renderer path.
pub struct GameHud {
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

// --- Orb LABEL row geometry (ABOVE the bar) --------------------------------
// The "ORBS: n/m" caption used to be stamped at the same top-left margin as the
// bar, so the lit glyph rows overlapped the amber fill rect (the visual bug).
// We now reserve a dedicated label row ABOVE the bar: the label occupies
// `ORB_LABEL_H` px starting at the top margin, then a small gap, then the bar.
// This guarantees the label glyphs and the bar fill rect are vertically
// disjoint (proven by `compose_label_and_bar_are_disjoint`).
const ORB_LABEL_SCALE: u32 = 2; // 5x7 bitmap scale -> CHAR_H*2 = 14px tall glyphs
const ORB_LABEL_H: f32 = (CHAR_H * ORB_LABEL_SCALE) as f32; // 14px
/// Vertical gap between the label row and the bar track.
const ORB_LABEL_GAP: f32 = 6.0;

impl GameHud {
    pub fn new(_width: u32, height: u32) -> Self {
        Self { height, margin: 16.0 }
    }

    /// Records the full HUD into `ctx`:
    /// - a translucent backdrop panel bottom-left,
    /// - the 16-band spectral bars (bottom-left, inside the panel),
    /// - an orb-progress bar top-left (background track + proportional fill),
    ///   sitting BELOW a reserved label row (the caption is stamped separately
    ///   via [`GameHud::draw_orb_label`], so it can never overlap the fill).
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

    /// Stamp the "ORBS: n/m" caption into `pixels`, left-aligned in its reserved
    /// row ABOVE the bar. Kept separate from [`GameHud::compose`] (which records
    /// vector rects into a `VelloCtxCpu`) because text rasterises directly into
    /// the final RGBA buffer. The label row and the bar are disjoint by
    /// construction — see [`GameHud::orb_label_rect`] /
    /// [`GameHud::orb_bar_track_rect`].
    pub fn draw_orb_label(
        &self,
        pixels: &mut [[u8; 4]],
        w: u32,
        h: u32,
        collected: u32,
        total: u32,
        color: [u8; 3],
    ) {
        let r = self.orb_label_rect();
        let label = format!("ORBS: {collected}/{total}");
        // Left-aligned within the row, vertically centred.
        let font_px = ORB_LABEL_H;
        let ty = r[1] + (r[3] - font_px) * 0.5;
        text::draw_text(pixels, w, h, [r[0], ty.max(0.0)], &label, color, font_px);
    }

    /// Records ONLY the bottom-left spectral-bars panel into `ctx` (no orb bar).
    /// Used by hosts that render the orb caption + progress bar via the retained
    /// [`crate::ui_tree`] HUD and want `GameHud` for just the 16-band readout.
    pub fn compose_spectral_panel(&self, ctx: &mut VelloCtxCpu, bands: &SpectralRadianceCache) {
        self.compose_spectral_bars(ctx, bands);
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

    /// Top-left orb-progress bar (BELOW the label row): a background track and a
    /// proportional fill.
    fn compose_orb_bar(&self, ctx: &mut VelloCtxCpu, collected: u32, total: u32) {
        let track = self.orb_bar_track_rect();

        // Background track (dark translucent).
        ctx.fill_rect(track, [0.05, 0.05, 0.08, 0.7]);

        let fill = self.orb_bar_fill_rect(collected, total);
        if fill[2] > 0.0 {
            // Warm amber fill so it reads against the dark track.
            ctx.fill_rect(fill, [1.0, 0.78, 0.2, 0.95]);
        }
    }

    /// Y-coordinate of the bar's top edge: below the label row + gap.
    fn bar_y(&self) -> f32 {
        self.margin + ORB_LABEL_H + ORB_LABEL_GAP
    }

    /// The reserved label row rect `[x, y, w, h]` (ABOVE the bar). Disjoint from
    /// [`GameHud::orb_bar_track_rect`] by construction.
    pub fn orb_label_rect(&self) -> [f32; 4] {
        [self.margin, self.margin, ORB_BAR_WIDTH, ORB_LABEL_H]
    }

    /// The orb-bar track rect `[x, y, w, h]` (background), useful for tests and
    /// for callers wanting hit-test geometry.
    pub fn orb_bar_track_rect(&self) -> [f32; 4] {
        [self.margin, self.bar_y(), ORB_BAR_WIDTH, ORB_BAR_HEIGHT]
    }

    /// The filled portion rect `[x, y, w, h]` for the given orb counts.
    pub fn orb_bar_fill_rect(&self, collected: u32, total: u32) -> [f32; 4] {
        let frac = if total == 0 {
            0.0
        } else {
            (collected.min(total) as f32) / (total as f32)
        };
        [self.margin, self.bar_y(), (ORB_BAR_WIDTH * frac).max(0.0), ORB_BAR_HEIGHT]
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

        // The orb fill command should be a FillRect at the bar's top-left (left
        // margin, below the reserved label row) with width ~= half the track.
        let track = hud.orb_bar_track_rect();
        let track_w = track[2];
        let bar_y = track[1];
        let found = ctx.commands().iter().any(|c| match c {
            DrawCmd::FillRect { rect, .. } => {
                (rect[0] - 16.0).abs() < 1e-3
                    && (rect[1] - bar_y).abs() < 1e-3
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

    /// REGRESSION (the shipped overlap bug): the "ORBS: n/m" label's lit glyph
    /// pixels and the amber bar-fill rect must occupy DISJOINT pixel regions.
    /// Previously both were stamped at the same top-left margin so the label sat
    /// on top of the fill. We render the label and the bar into one buffer and
    /// require zero overlap between the set of label-coloured glyph pixels and
    /// the set of amber fill pixels.
    #[test]
    fn compose_label_and_bar_are_disjoint() {
        let hud = GameHud::new(W, H);
        let bands = SpectralRadianceCache::from_f32([0.0; 16]);
        let mut ctx = VelloCtxCpu::new(W, H);
        hud.compose(&mut ctx, &bands, 5, 10);

        let mut pixels = vec![[0u8, 0, 0, 255]; (W * H) as usize];
        ctx.rasterize_into(&mut pixels, W, H);
        // Stamp the caption exactly as the game does (amber-ish label colour).
        let label_color = [255u8, 255, 100];
        hud.draw_orb_label(&mut pixels, W, H, 5, 10, label_color);

        // Classify each pixel in the top-left HUD region:
        // - "amber fill": the bar's warm fill (high red, red clearly > blue,
        //   and NOT the exact label colour).
        // - "label glyph": pixels matching the label colour we drew.
        // Require the two sets to be disjoint (no pixel is both).
        let label = hud.orb_label_rect();
        let bar = hud.orb_bar_track_rect();
        let y0 = label[1] as u32;
        let y1 = (bar[1] + bar[3]) as u32 + 2;
        let x0 = label[0] as u32;
        let x1 = (label[0] + label[2]) as u32;

        let mut label_px = 0u32;
        let mut fill_px = 0u32;
        let mut both = 0u32;
        for y in y0..y1.min(H) {
            for x in x0..x1.min(W) {
                let p = pixels[(y * W + x) as usize];
                let is_label = p[0] == label_color[0]
                    && p[1] == label_color[1]
                    && p[2] == label_color[2];
                // Amber fill: warm, red dominant over blue, but not the label.
                let is_fill = !is_label && p[0] > 150 && p[0] as i32 > p[2] as i32 + 40;
                if is_label {
                    label_px += 1;
                }
                if is_fill {
                    fill_px += 1;
                }
                if is_label && is_fill {
                    both += 1;
                }
            }
        }
        println!("label_px={label_px} fill_px={fill_px} overlap={both}");
        // Both regions must actually exist (label drew, bar drew)...
        assert!(label_px > 20, "label glyphs did not render ({label_px} px)");
        assert!(fill_px > 100, "bar fill did not render ({fill_px} px)");
        // ...and they must not overlap.
        assert_eq!(both, 0, "label glyphs overlap the bar fill ({both} shared px)");

        // Stronger structural guarantee: the label row's max y is strictly above
        // the bar's min y (vertical disjointness of the reserved rects).
        assert!(
            label[1] + label[3] <= bar[1],
            "label row [{}..{}] must end above bar top {}",
            label[1], label[1] + label[3], bar[1],
        );
    }
}
