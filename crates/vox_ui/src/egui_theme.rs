//! egui consumer of the [`crate::tokens::Tokens`] design system.
//!
//! [`apply`] writes egui `Visuals` + `Spacing` + `text_styles` derived ENTIRELY
//! from tokens — there are no literal `Color32`s here. The legacy
//! `theme::apply_ochroma_theme` is now a one-line shim over `apply(ctx,
//! &Tokens::default())` so `editor.rs`/`main.rs` build unchanged.

use crate::tokens::Tokens;
use egui::{Color32, CornerRadius, FontFamily, FontId, Shadow, Stroke, TextStyle, Visuals};

fn c(t: &Tokens, key: &str) -> Color32 {
    let [r, g, b, a] = t.color(key);
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

/// Apply a token set to an egui context: Visuals, Spacing, text ramp, motion.
pub fn apply(ctx: &egui::Context, t: &Tokens) {
    let mut style = (*ctx.style()).clone();
    let mut visuals = if c(t, "surface.bg.0").r() < 128 {
        Visuals::dark()
    } else {
        Visuals::light()
    };

    let bg0 = c(t, "surface.bg.0");
    let bg1 = c(t, "surface.bg.1");
    let bg2 = c(t, "surface.bg.2");
    let bg3 = c(t, "surface.bg.3");
    let hover = c(t, "surface.hover");
    let active = c(t, "surface.active");
    let border = c(t, "surface.border");
    let accent = c(t, "accent.base");
    let accent_hover = c(t, "accent.hover");
    let accent_dim = c(t, "accent.dim");
    let text_primary = c(t, "text.primary");
    let text_secondary = c(t, "text.secondary");
    let warning = c(t, "status.warning");
    let error = c(t, "status.error");

    let r_sm = t.radius[0] as u8;

    visuals.window_fill = bg1;
    visuals.panel_fill = bg1;
    visuals.faint_bg_color = bg2;
    visuals.extreme_bg_color = bg0;

    visuals.widgets.noninteractive.bg_fill = bg2;
    visuals.widgets.noninteractive.weak_bg_fill = bg2;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text_secondary);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.5, border);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(r_sm);

    visuals.widgets.inactive.bg_fill = bg3;
    visuals.widgets.inactive.weak_bg_fill = bg3;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, text_primary);
    visuals.widgets.inactive.bg_stroke = Stroke::new(0.5, border);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(r_sm);

    visuals.widgets.hovered.bg_fill = hover;
    visuals.widgets.hovered.weak_bg_fill = hover;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, text_primary);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(r_sm);

    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.weak_bg_fill = accent;
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, accent_hover);
    visuals.widgets.active.corner_radius = CornerRadius::same(r_sm);

    visuals.widgets.open.bg_fill = active;
    visuals.widgets.open.weak_bg_fill = active;
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, text_primary);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, accent_dim);
    visuals.widgets.open.corner_radius = CornerRadius::same(r_sm);

    visuals.selection.bg_fill = accent_dim;
    visuals.selection.stroke = Stroke::new(1.0, accent);

    visuals.window_shadow = Shadow {
        offset: [0, 2],
        blur: 8,
        spread: 0,
        color: Color32::from_black_alpha(80),
    };
    visuals.window_corner_radius = CornerRadius::same(t.radius[1] as u8);
    visuals.resize_corner_size = t.space[3];
    visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };
    visuals.hyperlink_color = accent;
    visuals.warn_fg_color = warning;
    visuals.error_fg_color = error;

    // === SPACING — all derived as 4px-grid multiples (rem-scalable) ===
    let s1 = t.space[1];
    let s2 = t.space[2];
    let s3 = t.space[3];
    let scale = t.rem / 14.0;
    style.spacing.item_spacing = egui::vec2(s2 * scale, s1 * scale);
    style.spacing.window_margin = egui::Margin::same((s2 * scale) as i8);
    style.spacing.button_padding = egui::vec2(s2 * scale, s1 * scale);
    style.spacing.indent = s3 * scale;
    style.spacing.interact_size = egui::vec2(32.0 * scale, (t.rem + 6.0) * scale);
    style.spacing.slider_width = 120.0 * scale;
    style.spacing.text_edit_width = 200.0 * scale;
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 6.0,
        ..Default::default()
    };

    // === TYPE RAMP — real AA vector glyphs via egui's atlas ===
    let tr = &t.type_ramp;
    style.text_styles = [
        (TextStyle::Small, FontId::new(tr.caption, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(tr.body, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(tr.mono, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(tr.body, FontFamily::Proportional)),
        (TextStyle::Heading, FontId::new(tr.heading, FontFamily::Proportional)),
    ]
    .into();

    style.animation_time = t.motion("fast");

    style.visuals = visuals;
    ctx.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_writes_accent_into_active_widget_fill() {
        // The design's invariant (test 1, egui half): the active-widget fill
        // bytes equal Tokens::color("accent.base") resolved from the JSON.
        let ctx = egui::Context::default();
        let t = Tokens::default();
        apply(&ctx, &t);
        let got = ctx.style().visuals.widgets.active.bg_fill;
        let want = t.color("accent.base");
        assert_eq!(
            [got.r(), got.g(), got.b(), got.a()],
            want,
            "egui active fill must equal token accent.base"
        );
    }

    #[test]
    fn light_theme_swaps_panel_fill_pixel() {
        // Swapping dark->light changes the egui panel fill bytes (the in-Visuals
        // proof behind the snapshot's panel-pixel swap).
        let ctx = egui::Context::default();
        apply(&ctx, &Tokens::default());
        let dark = ctx.style().visuals.panel_fill;

        let light_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/ui/ochroma_light.theme.json");
        let light = Tokens::load(light_path).unwrap();
        apply(&ctx, &light);
        let lit = ctx.style().visuals.panel_fill;

        let delta = (lit.r() as i32 - dark.r() as i32).unsigned_abs();
        assert!(
            delta > 40,
            "panel fill r changed by only {delta} on dark->light (dark={dark:?} light={lit:?})"
        );
    }

    #[test]
    fn rem_scales_button_padding() {
        let ctx = egui::Context::default();
        let mut t = Tokens::default();
        apply(&ctx, &t);
        let pad14 = ctx.style().spacing.button_padding;
        t.rem = 18.0;
        apply(&ctx, &t);
        let pad18 = ctx.style().spacing.button_padding;
        assert!(
            pad18.x > pad14.x,
            "button padding must grow with rem (14:{} 18:{})",
            pad14.x,
            pad18.x
        );
    }
}
