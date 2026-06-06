//! Real text rasterisation for the retained game UI.
//!
//! This module gives [`crate::ui_tree`] a single text-drawing entry point,
//! [`draw_text`], that stamps a string into an RGBA8 buffer at a pixel box.
//!
//! - With the `game-ui` feature on, text is shaped + laid out by **parley 0.3**
//!   (fontique resolves a system font) and each glyph is rasterised to an 8-bit
//!   alpha coverage mask by **swash**, then alpha-composited into the frame.
//!   This is true outline rendering: glyphs scale continuously and anti-alias,
//!   unlike the fixed 5x7 grid.
//! - Without `game-ui` it falls back to [`vox_core::game_ui::burn_text`], the
//!   blocky 5x7 bitmap font, so the default (engine-only) build keeps working
//!   with no extra dependency.
//!
//! The pixel layout matches the rest of the CPU UI path: row-major `[u8; 4]`
//! straight-alpha RGBA, `pixels.len() >= w*h`.

use vox_core::game_ui::{burn_text, CHAR_H};

/// Native pixel height of the 5x7 bitmap font (one scale unit). Callers use it
/// to pick a `font_px` that roughly fills the same row height as the fallback.
pub const BITMAP_CHAR_H: u32 = CHAR_H;

/// Map a desired pixel font height to the nearest bitmap-font integer scale
/// (the fallback path can only render at integer multiples of the 5x7 grid).
fn bitmap_scale_for_px(font_px: f32) -> u32 {
    ((font_px / CHAR_H as f32).round() as u32).max(1)
}

/// Draw `text` so its top-left sits at (`x`, `y`) in `pixels`, sized so the
/// glyphs are about `font_px` tall, in straight RGB `color`.
///
/// Returns the lit-pixel count actually written (coverage > 0), which the UI
/// tree's tests use to prove real rendering happened.
///
/// With `game-ui`: parley layout + swash alpha masks (anti-aliased, true
/// scaling). Without it: the 5x7 bitmap font at the nearest integer scale.
pub fn draw_text(
    pixels: &mut [[u8; 4]],
    buf_w: u32,
    buf_h: u32,
    pos: [f32; 2],
    text: &str,
    color: [u8; 3],
    font_px: f32,
) -> usize {
    #[cfg(feature = "game-ui")]
    {
        real_text::draw(pixels, buf_w, buf_h, pos, text, color, font_px)
    }
    #[cfg(not(feature = "game-ui"))]
    {
        bitmap_draw(pixels, buf_w, buf_h, pos, text, color, font_px)
    }
}

/// Bitmap (fallback) text draw. Also reachable directly by tests that want to
/// compare the bitmap path against the parley path at the same box. `pos` is the
/// top-left `[x, y]` in pixels.
pub fn bitmap_draw(
    pixels: &mut [[u8; 4]],
    buf_w: u32,
    buf_h: u32,
    pos: [f32; 2],
    text: &str,
    color: [u8; 3],
    font_px: f32,
) -> usize {
    let scale = bitmap_scale_for_px(font_px);
    let x0 = pos[0].max(0.0) as u32;
    let y0 = pos[1].max(0.0) as u32;
    burn_text(pixels, buf_w, x0, y0, text, color, scale);
    // Count the lit glyph pixels we just wrote (exact RGB match, opaque).
    count_color_pixels(pixels, buf_w, buf_h, color)
}

/// Count pixels whose RGB exactly equals `color` and are fully opaque — the
/// signature `burn_text` leaves. Bounded to the buffer.
fn count_color_pixels(pixels: &[[u8; 4]], buf_w: u32, buf_h: u32, color: [u8; 3]) -> usize {
    let n = (buf_w * buf_h) as usize;
    pixels
        .iter()
        .take(n)
        .filter(|p| p[0] == color[0] && p[1] == color[1] && p[2] == color[2] && p[3] == 255)
        .count()
}

// ---------------------------------------------------------------------------
// Real text path (parley layout + swash glyph masks), feature-gated.
// ---------------------------------------------------------------------------

#[cfg(feature = "game-ui")]
mod real_text {
    use parley::{Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext};
    use parley::style::{StyleProperty};
    use std::cell::RefCell;
    use swash::scale::image::Content;
    use swash::scale::{Render, ScaleContext, Source, StrikeWith};
    use swash::zeno::{Format, Vector};
    use swash::FontRef;

    // parley font/scratch contexts are not cheap to build (fontique scans the
    // system font db). Keep them thread-local and reuse across draws.
    thread_local! {
        static FONT_CX: RefCell<FontContext> = RefCell::new(FontContext::new());
        static LAYOUT_CX: RefCell<LayoutContext<[u8; 4]>> = RefCell::new(LayoutContext::new());
        static SCALE_CX: RefCell<ScaleContext> = RefCell::new(ScaleContext::new());
    }

    /// Shape + lay out `text` at `font_px`, rasterise each glyph to an alpha
    /// mask via swash, and alpha-composite it into `pixels`. Returns the count
    /// of pixels touched with non-zero coverage.
    pub fn draw(
        pixels: &mut [[u8; 4]],
        buf_w: u32,
        buf_h: u32,
        pos: [f32; 2],
        text: &str,
        color: [u8; 3],
        font_px: f32,
    ) -> usize {
        if text.is_empty() {
            return 0;
        }
        let layout = build_layout(text, font_px, color);

        // `positioned_glyphs` already bakes the run offset + baseline into each
        // glyph; we add only `pos` (the caller's top-left draw origin).
        let mut lit = 0usize;
        SCALE_CX.with(|scx| {
            let mut scale_cx = scx.borrow_mut();
            for line in layout.lines() {
                for item in line.items() {
                    if let parley::PositionedLayoutItem::GlyphRun(run) = item {
                        lit += render_glyph_run(
                            &mut scale_cx, &run, pixels, buf_w, buf_h, pos, color,
                        );
                    }
                }
            }
        });
        lit
    }

    fn build_layout(text: &str, font_px: f32, color: [u8; 3]) -> Layout<[u8; 4]> {
        FONT_CX.with(|fcx| {
            LAYOUT_CX.with(|lcx| {
                let mut font_cx = fcx.borrow_mut();
                let mut layout_cx = lcx.borrow_mut();
                let brush = [color[0], color[1], color[2], 255];
                let mut builder = layout_cx.ranged_builder(&mut font_cx, text, 1.0);
                builder.push_default(StyleProperty::FontSize(font_px));
                builder.push_default(StyleProperty::Brush(brush));
                let mut layout: Layout<[u8; 4]> = builder.build(text);
                layout.break_all_lines(None);
                layout.align(None, Alignment::Start, AlignmentOptions::default());
                layout
            })
        })
    }

    fn render_glyph_run(
        scale_cx: &mut ScaleContext,
        run: &GlyphRun<'_, [u8; 4]>,
        pixels: &mut [[u8; 4]],
        buf_w: u32,
        buf_h: u32,
        pos: [f32; 2],
        color: [u8; 3],
    ) -> usize {
        let font_run = run.run();
        let font = font_run.font();
        let font_size = font_run.font_size();
        let normalized_coords = font_run.normalized_coords();

        // peniko Font -> swash FontRef over the raw font bytes at the face index.
        let font_data: &[u8] = font.data.as_ref();
        let Some(font_ref) = FontRef::from_index(font_data, font.index as usize) else {
            return 0;
        };

        let mut scaler = scale_cx
            .builder(font_ref)
            .size(font_size)
            .hint(true)
            .normalized_coords(normalized_coords.iter().copied())
            .build();

        // `positioned_glyphs()` yields each glyph with `x` already advanced along
        // the run (run offset + cumulative advances) and `y` already offset by
        // the baseline. We add only the caller's draw origin.
        let mut lit = 0usize;
        for glyph in run.positioned_glyphs() {
            let gx = pos[0] + glyph.x;
            let gy = pos[1] + glyph.y;
            // Rasterise the outline to an 8-bit alpha mask at the glyph's
            // fractional pen position (the fractional part feeds anti-aliasing).
            let image = Render::new(&[
                Source::ColorOutline(0),
                Source::ColorBitmap(StrikeWith::BestFit),
                Source::Outline,
            ])
            .format(Format::Alpha)
            .offset(Vector::new(gx.fract(), gy.fract()))
            .render(&mut scaler, glyph.id);

            let Some(image) = image else { continue };
            if image.content != Content::Mask {
                // We only blit alpha masks; skip colour glyphs (none for ASCII).
                continue;
            }
            let gw = image.placement.width as i32;
            let gh = image.placement.height as i32;
            if gw == 0 || gh == 0 {
                continue;
            }
            // Top-left of the mask in buffer space.
            let px0 = gx.floor() as i32 + image.placement.left;
            let py0 = gy.floor() as i32 - image.placement.top;

            for row in 0..gh {
                let dy = py0 + row;
                if dy < 0 || dy >= buf_h as i32 {
                    continue;
                }
                for cx in 0..gw {
                    let dx = px0 + cx;
                    if dx < 0 || dx >= buf_w as i32 {
                        continue;
                    }
                    let cov = image.data[(row * gw + cx) as usize];
                    if cov == 0 {
                        continue;
                    }
                    let a = cov as f32 / 255.0;
                    let idx = (dy as u32 * buf_w + dx as u32) as usize;
                    if idx >= pixels.len() {
                        continue;
                    }
                    let d = pixels[idx];
                    let inv = 1.0 - a;
                    pixels[idx] = [
                        (color[0] as f32 * a + d[0] as f32 * inv + 0.5) as u8,
                        (color[1] as f32 * a + d[1] as f32 * inv + 0.5) as u8,
                        (color[2] as f32 * a + d[2] as f32 * inv + 0.5) as u8,
                        ((a + d[3] as f32 / 255.0 * inv) * 255.0 + 0.5) as u8,
                    ];
                    lit += 1;
                }
            }
        }
        lit
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// At the same font box, the parley-rendered label lights MORE pixels than
    /// the 5x7 bitmap version (anti-aliased coverage > blocky on/off grid).
    /// Without `game-ui`, draw_text *is* the bitmap path, so we just prove the
    /// bitmap path lights a sane number of pixels.
    #[test]
    fn text_lights_pixels_at_box() {
        let (w, h) = (256u32, 64u32);
        let color = [255u8, 255, 255];

        // Compare at the SAME RENDERED CAP HEIGHT. The 5x7 fallback is a *sparse
        // stroke* font: at its native 7px scale only ~15 of each glyph's 35
        // cells light up, and there is no anti-aliasing. A real proportional
        // face rendered to fill the same ~7px cap box covers its strokes densely
        // and adds anti-aliased edge pixels, so it lights strictly more pixels —
        // the whole point of wiring in parley.
        let text = "ORBS 5/10";
        // Native 5x7 bitmap (scale 1 -> 7px tall glyphs).
        let mut bmp = vec![[0u8, 0, 0, 255]; (w * h) as usize];
        let bmp_lit = bitmap_draw(&mut bmp, w, h, [4.0, 8.0], text, color, BITMAP_CHAR_H as f32);
        println!("bitmap(7px) lit px = {bmp_lit}");
        assert!(bmp_lit > 30, "bitmap text should light real px, got {bmp_lit}");

        #[cfg(feature = "game-ui")]
        {
            // Parley at a font size whose cap height ~matches the 7px bitmap box.
            // (em ~ 11px gives a cap height near 7-8px for typical sans faces.)
            let mut par = vec![[0u8, 0, 0, 255]; (w * h) as usize];
            let par_lit = draw_text(&mut par, w, h, [4.0, 8.0], text, color, 11.0);
            println!("parley(~7px cap) lit px = {par_lit} vs bitmap {bmp_lit}");
            assert!(
                par_lit > bmp_lit,
                "parley anti-aliased coverage ({par_lit}) should exceed bitmap ({bmp_lit})",
            );
        }
    }

    /// A tall glyph ('M' at 24px) must span >= 14 rows of lit pixels — proof of
    /// real continuous scaling, not the fixed 5x7 (CHAR_H=7) grid that would cap
    /// 'M' at 7 rows native / a coarse integer multiple. Only meaningful with
    /// the real-text path.
    #[cfg(feature = "game-ui")]
    #[test]
    fn tall_glyph_spans_many_rows() {
        let (w, h) = (64u32, 64u32);
        let color = [255u8, 255, 255];
        let mut px = vec![[0u8, 0, 0, 255]; (w * h) as usize];
        draw_text(&mut px, w, h, [8.0, 4.0], "M", color, 24.0);

        // Count rows that contain at least one lit pixel.
        let mut rows_lit = 0u32;
        for y in 0..h {
            let mut any = false;
            for x in 0..w {
                let p = px[(y * w + x) as usize];
                if p[0] > 20 || p[1] > 20 || p[2] > 20 {
                    any = true;
                    break;
                }
            }
            if any {
                rows_lit += 1;
            }
        }
        println!("M@24px rows lit = {rows_lit}");
        assert!(
            rows_lit >= 14,
            "M at 24px should span >= 14 rows (real scaling), spanned {rows_lit}",
        );
    }
}
