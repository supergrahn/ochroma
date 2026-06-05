//! GameMenu — composes full-screen game menus (main / pause / win) as draw
//! commands for the CPU software-renderer path.
//!
//! Like [`crate::game_hud::GameHud`], this records `DrawCmd`s into a
//! [`VelloCtxCpu`] which the software rasteriser composites into the game frame
//! via [`VelloCtxCpu::rasterize_into`].
//!
//! Each menu draws:
//! - a full-screen translucent dim overlay (so the live scene is visibly dimmed),
//! - a centred panel,
//! - a 16-band spectral accent strip across the top of the panel (reusing
//!   [`SpectralHUD::band_colors`]), and
//! - one button-like highlight rect per selectable option, with the selected
//!   option drawn brighter than the rest.
//!
//! Text labels are NOT drawn here — the shell stamps bitmap text on top using
//! the rects exposed by [`GameMenu::option_rects`] (and [`GameMenu::title_rect`]
//! / [`GameMenu::accent_strip_rect`]) so labels land inside the highlight zones.

use crate::spectral_hud::SpectralHUD;
use crate::vello_ctx::VelloCtxCpu;

/// Composes full-screen game menus for the CPU software-renderer path.
#[derive(Debug, Clone, Copy)]
pub struct GameMenu {
    width:  u32,
    height: u32,
}

// --- Panel geometry --------------------------------------------------------
/// Panel width as a fraction of the screen width.
const PANEL_W_FRAC: f32 = 0.5;
/// Panel height as a fraction of the screen height.
const PANEL_H_FRAC: f32 = 0.6;
/// Inner padding between the panel edge and its contents.
const PANEL_PAD: f32 = 24.0;

// --- Accent strip ----------------------------------------------------------
/// Height of the 16-band spectral accent strip along the top of the panel.
const ACCENT_H: f32 = 16.0;
/// Gap below the accent strip before the title zone.
const ACCENT_GAP: f32 = 16.0;

// --- Title / option zones --------------------------------------------------
/// Height of the title label zone (a reserved band the shell draws text into).
const TITLE_H: f32 = 56.0;
/// Height of each selectable-option highlight rect.
const OPTION_H: f32 = 44.0;
/// Vertical gap between option rects.
const OPTION_GAP: f32 = 14.0;

// --- Colors ----------------------------------------------------------------
/// Full-screen dim overlay (translucent black). Alpha drives how much the live
/// scene behind the menu is darkened.
const OVERLAY: [f32; 4] = [0.0, 0.0, 0.0, 0.62];
/// Panel backdrop (dark, nearly opaque so options read clearly).
const PANEL_BG: [f32; 4] = [0.06, 0.07, 0.10, 0.92];
/// Unselected option highlight (dim slate).
const OPTION_UNSELECTED: [f32; 4] = [0.18, 0.20, 0.26, 0.85];
/// Selected option highlight (bright warm gold) — must read clearly brighter
/// than [`OPTION_UNSELECTED`].
const OPTION_SELECTED: [f32; 4] = [0.95, 0.78, 0.25, 0.95];

impl GameMenu {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// The centred panel rect `[x, y, w, h]`.
    pub fn panel_rect(&self) -> [f32; 4] {
        let w = self.width as f32 * PANEL_W_FRAC;
        let h = self.height as f32 * PANEL_H_FRAC;
        let x = (self.width as f32 - w) * 0.5;
        let y = (self.height as f32 - h) * 0.5;
        [x, y, w, h]
    }

    /// The 16-band spectral accent strip rect `[x, y, w, h]` along the top of
    /// the panel's content area.
    pub fn accent_strip_rect(&self) -> [f32; 4] {
        let panel = self.panel_rect();
        [
            panel[0] + PANEL_PAD,
            panel[1] + PANEL_PAD,
            panel[2] - PANEL_PAD * 2.0,
            ACCENT_H,
        ]
    }

    /// The title label zone rect `[x, y, w, h]` (below the accent strip). The
    /// shell centres its bitmap title text inside this band.
    pub fn title_rect(&self) -> [f32; 4] {
        let panel = self.panel_rect();
        let strip = self.accent_strip_rect();
        [
            panel[0] + PANEL_PAD,
            strip[1] + strip[3] + ACCENT_GAP,
            panel[2] - PANEL_PAD * 2.0,
            TITLE_H,
        ]
    }

    /// The highlight rects `[x, y, w, h]` for `n` selectable options, stacked
    /// below the title zone and centred horizontally in the panel. The shell
    /// centres each option's bitmap label inside the matching rect.
    pub fn option_rects(&self, n: usize) -> Vec<[f32; 4]> {
        let panel = self.panel_rect();
        let title = self.title_rect();
        let x = panel[0] + PANEL_PAD;
        let w = panel[2] - PANEL_PAD * 2.0;
        let mut y = title[1] + title[3] + OPTION_GAP;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push([x, y, w, OPTION_H]);
            y += OPTION_H + OPTION_GAP;
        }
        out
    }

    /// Compose the main menu: dim overlay, panel, accent strip, and `n` option
    /// highlights with `selected` drawn bright.
    pub fn compose_main(&self, ctx: &mut VelloCtxCpu, option_count: usize, selected: usize) {
        self.compose_common(ctx, option_count, selected);
    }

    /// Compose the pause menu — identical layout to the main menu.
    pub fn compose_pause(&self, ctx: &mut VelloCtxCpu, option_count: usize, selected: usize) {
        self.compose_common(ctx, option_count, selected);
    }

    /// Compose the win screen. Draws the same overlay/panel/accent strip plus a
    /// single highlighted "continue" option. `orbs_total` and `elapsed_s` are
    /// not rendered here (no text on this path); the shell stamps them as bitmap
    /// text inside [`GameMenu::title_rect`] / the option rect. They are accepted
    /// so the call site reads intentionally and to keep the signature stable.
    pub fn compose_win(
        &self,
        ctx:        &mut VelloCtxCpu,
        _orbs_total: u32,
        _elapsed_s:  f32,
        option_count: usize,
        selected:     usize,
    ) {
        self.compose_common(ctx, option_count, selected);
    }

    /// Shared composition for all three menus.
    fn compose_common(&self, ctx: &mut VelloCtxCpu, option_count: usize, selected: usize) {
        // 1. Full-screen dim overlay so the live scene behind reads darker.
        ctx.fill_rect([0.0, 0.0, self.width as f32, self.height as f32], OVERLAY);

        // 2. Centred panel backdrop.
        ctx.fill_rect(self.panel_rect(), PANEL_BG);

        // 3. 16-band spectral accent strip (reuse SpectralHUD::band_colors).
        self.compose_accent_strip(ctx);

        // 4. Option highlight rects — selected drawn brighter.
        for (i, rect) in self.option_rects(option_count).into_iter().enumerate() {
            let color = if i == selected { OPTION_SELECTED } else { OPTION_UNSELECTED };
            ctx.fill_rect(rect, color);
        }
    }

    /// Paint the 16-band spectral accent strip: one equal-width colored cell per
    /// band across the strip rect, using the SpectralHUD band palette.
    fn compose_accent_strip(&self, ctx: &mut VelloCtxCpu) {
        let strip = self.accent_strip_rect();
        let colors = SpectralHUD::band_colors();
        let cell_w = strip[2] / 16.0;
        for (b, color) in colors.iter().enumerate() {
            let x = strip[0] + b as f32 * cell_w;
            ctx.fill_rect([x, strip[1], cell_w, strip[3]], *color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 1280;
    const H: u32 = 720;

    fn mean_luma(pixels: &[[u8; 4]]) -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }
        let sum: f64 = pixels
            .iter()
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .sum();
        (sum / pixels.len() as f64) as f32
    }

    /// A mid-grey scene buffer so the overlay's dimming is measurable.
    fn grey_scene() -> Vec<[u8; 4]> {
        vec![[160u8, 160, 160, 255]; (W * H) as usize]
    }

    #[test]
    fn option_rects_returns_requested_count_stacked_downward() {
        let menu = GameMenu::new(W, H);
        let rects = menu.option_rects(3);
        assert_eq!(rects.len(), 3, "expected 3 option rects");
        // Each subsequent rect is strictly lower (greater y) than the previous.
        assert!(rects[1][1] > rects[0][1], "rect 1 y={} not below rect 0 y={}", rects[1][1], rects[0][1]);
        assert!(rects[2][1] > rects[1][1], "rect 2 y={} not below rect 1 y={}", rects[2][1], rects[1][1]);
        // All rects share the same x and width.
        assert!((rects[0][0] - rects[1][0]).abs() < 1e-3);
        assert!((rects[0][2] - rects[1][2]).abs() < 1e-3);
    }

    #[test]
    fn option_rects_lie_inside_panel() {
        let menu = GameMenu::new(W, H);
        let panel = menu.panel_rect();
        for r in menu.option_rects(2) {
            assert!(r[0] >= panel[0], "option x {} left of panel {}", r[0], panel[0]);
            assert!(r[0] + r[2] <= panel[0] + panel[2] + 1e-3, "option overflows panel right");
            assert!(r[1] >= panel[1], "option y above panel top");
            assert!(r[1] + r[3] <= panel[1] + panel[3] + 1e-3, "option overflows panel bottom");
        }
    }

    #[test]
    fn main_menu_overlay_dims_scene() {
        let menu = GameMenu::new(W, H);
        let untouched = grey_scene();
        let base_luma = mean_luma(&untouched);

        let mut ctx = VelloCtxCpu::new(W, H);
        menu.compose_main(&mut ctx, 2, 0);
        let mut pixels = untouched.clone();
        ctx.rasterize_into(&mut pixels, W, H);
        let menu_luma = mean_luma(&pixels);

        println!("base_luma={base_luma} menu_luma={menu_luma}");
        assert!(
            menu_luma < base_luma,
            "menu mean luminance {menu_luma} should be < untouched scene {base_luma} (overlay must dim)",
        );
    }

    #[test]
    fn accent_strip_shows_violet_at_band0_and_red_at_band15() {
        let menu = GameMenu::new(W, H);
        let mut ctx = VelloCtxCpu::new(W, H);
        menu.compose_main(&mut ctx, 2, 0);
        let mut pixels = vec![[0u8, 0, 0, 255]; (W * H) as usize];
        ctx.rasterize_into(&mut pixels, W, H);

        let strip = menu.accent_strip_rect();
        let cell_w = strip[2] / 16.0;
        let sample_y = (strip[1] + strip[3] * 0.5) as u32;
        // Sample the centre of band 0 (violet: high blue) and band 15 (red).
        let band0_x = (strip[0] + cell_w * 0.5) as u32;
        let band15_x = (strip[0] + cell_w * 15.5) as u32;
        let b0 = pixels[(sample_y * W + band0_x) as usize];
        let b15 = pixels[(sample_y * W + band15_x) as usize];
        println!("band0={b0:?} band15={b15:?}");
        // Band 0 violet: blue dominates red.
        assert!(b0[2] > b0[0], "band0 should be violet-ish (blue>red): {b0:?}");
        assert!(b0[2] > 100, "band0 blue channel should be strong: {b0:?}");
        // Band 15 red: red dominates blue.
        assert!(b15[0] > b15[2], "band15 should be red-ish (red>blue): {b15:?}");
        assert!(b15[0] > 100, "band15 red channel should be strong: {b15:?}");
    }

    #[test]
    fn selected_option_brighter_than_unselected() {
        let menu = GameMenu::new(W, H);
        let mut ctx = VelloCtxCpu::new(W, H);
        // 2 options, select index 0.
        menu.compose_main(&mut ctx, 2, 0);
        let mut pixels = vec![[0u8, 0, 0, 255]; (W * H) as usize];
        ctx.rasterize_into(&mut pixels, W, H);

        let rects = menu.option_rects(2);
        let center = |r: [f32; 4]| -> [u8; 4] {
            let cx = (r[0] + r[2] * 0.5) as u32;
            let cy = (r[1] + r[3] * 0.5) as u32;
            pixels[(cy * W + cx) as usize]
        };
        let sel = center(rects[0]);
        let unsel = center(rects[1]);
        let lum = |p: [u8; 4]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
        println!("selected={sel:?} unselected={unsel:?}");
        assert!(
            lum(sel) > lum(unsel) + 20.0,
            "selected option luma {} should exceed unselected {} by a clear margin",
            lum(sel),
            lum(unsel),
        );
    }

    #[test]
    fn selection_index_follows_selected_parameter() {
        let menu = GameMenu::new(W, H);
        // Select index 1 this time; the second option should now be the bright one.
        let mut ctx = VelloCtxCpu::new(W, H);
        menu.compose_main(&mut ctx, 2, 1);
        let mut pixels = vec![[0u8, 0, 0, 255]; (W * H) as usize];
        ctx.rasterize_into(&mut pixels, W, H);

        let rects = menu.option_rects(2);
        let center = |r: [f32; 4]| -> [u8; 4] {
            let cx = (r[0] + r[2] * 0.5) as u32;
            let cy = (r[1] + r[3] * 0.5) as u32;
            pixels[(cy * W + cx) as usize]
        };
        let lum = |p: [u8; 4]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
        let opt0 = lum(center(rects[0]));
        let opt1 = lum(center(rects[1]));
        println!("opt0={opt0} opt1={opt1}");
        assert!(opt1 > opt0 + 20.0, "option 1 (selected) should be brighter than option 0");
    }

    #[test]
    fn pause_menu_dims_and_paints_options() {
        let menu = GameMenu::new(W, H);
        let untouched = grey_scene();
        let base_luma = mean_luma(&untouched);
        let mut ctx = VelloCtxCpu::new(W, H);
        menu.compose_pause(&mut ctx, 2, 1);
        let mut pixels = untouched.clone();
        ctx.rasterize_into(&mut pixels, W, H);
        assert!(mean_luma(&pixels) < base_luma, "pause overlay must dim the scene");
    }

    #[test]
    fn win_screen_dims_and_accepts_stats() {
        let menu = GameMenu::new(W, H);
        let untouched = grey_scene();
        let base_luma = mean_luma(&untouched);
        let mut ctx = VelloCtxCpu::new(W, H);
        // 10 orbs, 42.5s elapsed, single "continue" option selected.
        menu.compose_win(&mut ctx, 10, 42.5, 1, 0);
        let mut pixels = untouched.clone();
        ctx.rasterize_into(&mut pixels, W, H);
        let win_luma = mean_luma(&pixels);
        assert!(win_luma < base_luma, "win overlay must dim the scene");
        // The single highlighted option must be present (bright fill somewhere
        // in the option rect): sample its centre and require it to be brighter
        // than the dimmed scene background.
        let rect = menu.option_rects(1)[0];
        let cx = (rect[0] + rect[2] * 0.5) as u32;
        let cy = (rect[1] + rect[3] * 0.5) as u32;
        let p = pixels[(cy * W + cx) as usize];
        let lum = 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
        assert!(lum > win_luma, "win continue-option {p:?} should be brighter than the dimmed scene");
    }
}
