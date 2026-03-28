use egui::{Color32, CornerRadius, FontFamily, FontId, Shadow, Stroke, TextStyle, Visuals};

/// Apply the Ochroma 2026 dark theme to an egui context.
pub fn apply_ochroma_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let mut visuals = Visuals::dark();

    // === COLORS ===
    // Background: very dark blue-grey, not pure black
    let bg_darkest = Color32::from_rgb(18, 18, 24);
    let bg_dark = Color32::from_rgb(24, 24, 32);
    let bg_panel = Color32::from_rgb(30, 30, 40);
    let bg_widget = Color32::from_rgb(38, 38, 50);
    let bg_hover = Color32::from_rgb(48, 48, 65);
    let bg_active = Color32::from_rgb(55, 55, 75);

    // Accent: electric blue (Ochroma brand)
    let accent = Color32::from_rgb(60, 130, 255);
    let accent_hover = Color32::from_rgb(80, 150, 255);
    let accent_dim = Color32::from_rgb(40, 90, 180);

    // Text
    let text_primary = Color32::from_rgb(220, 222, 230);
    let text_secondary = Color32::from_rgb(140, 145, 160);
    let _text_disabled = Color32::from_rgb(80, 85, 95);

    // Borders: subtle, sharp
    let border = Color32::from_rgb(45, 45, 60);

    // Status colors
    let warning = Color32::from_rgb(255, 180, 50);
    let error = Color32::from_rgb(255, 70, 70);

    // === VISUALS ===
    visuals.window_fill = bg_dark;
    visuals.panel_fill = bg_dark;
    visuals.faint_bg_color = bg_panel;
    visuals.extreme_bg_color = bg_darkest;

    // Widgets
    visuals.widgets.noninteractive.bg_fill = bg_panel;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text_secondary);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.5, border);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(3);

    visuals.widgets.inactive.bg_fill = bg_widget;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, text_primary);
    visuals.widgets.inactive.bg_stroke = Stroke::new(0.5, border);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(3);

    visuals.widgets.hovered.bg_fill = bg_hover;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, text_primary);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(3);

    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, accent_hover);
    visuals.widgets.active.corner_radius = CornerRadius::same(3);

    visuals.widgets.open.bg_fill = bg_active;
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, text_primary);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, accent_dim);
    visuals.widgets.open.corner_radius = CornerRadius::same(3);

    // Selection
    visuals.selection.bg_fill = accent_dim;
    visuals.selection.stroke = Stroke::new(1.0, accent);

    // Window shadow (subtle, not Windows XP drop shadow)
    visuals.window_shadow = Shadow {
        offset: [0, 2],
        blur: 8,
        spread: 0,
        color: Color32::from_black_alpha(80),
    };

    // Window corner radius: sharp but not completely square
    visuals.window_corner_radius = CornerRadius::same(4);

    // Resize handle
    visuals.resize_corner_size = 8.0;

    // Scrollbar: thin, subtle
    visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };

    // Hyperlinks
    visuals.hyperlink_color = accent;

    // Warn/error
    visuals.warn_fg_color = warning;
    visuals.error_fg_color = error;

    // === SPACING ===
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.window_margin = egui::Margin::same(8);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.indent = 16.0;
    style.spacing.interact_size = egui::vec2(32.0, 20.0);
    style.spacing.slider_width = 120.0;
    style.spacing.text_edit_width = 200.0;
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 6.0,
        ..Default::default()
    };

    // === TEXT ===
    style.text_styles = [
        (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(13.0, FontFamily::Proportional)),
        (TextStyle::Heading, FontId::new(16.0, FontFamily::Proportional)),
    ]
    .into();

    // === ANIMATION ===
    style.animation_time = 0.1;

    // Apply
    style.visuals = visuals;
    ctx.set_style(style);
}

/// Branded colors for use throughout the editor UI.
pub struct OchromaColors;

impl OchromaColors {
    pub const BG_DARKEST: Color32 = Color32::from_rgb(18, 18, 24);
    pub const BG_DARK: Color32 = Color32::from_rgb(24, 24, 32);
    pub const BG_PANEL: Color32 = Color32::from_rgb(30, 30, 40);
    pub const BG_WIDGET: Color32 = Color32::from_rgb(38, 38, 50);
    pub const ACCENT: Color32 = Color32::from_rgb(60, 130, 255);
    pub const ACCENT_HOVER: Color32 = Color32::from_rgb(80, 150, 255);
    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(220, 222, 230);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(140, 145, 160);
    pub const SUCCESS: Color32 = Color32::from_rgb(50, 200, 120);
    pub const WARNING: Color32 = Color32::from_rgb(255, 180, 50);
    pub const ERROR: Color32 = Color32::from_rgb(255, 70, 70);

    // Gizmo axis colors (bright, distinct)
    pub const AXIS_X: Color32 = Color32::from_rgb(230, 60, 60);
    pub const AXIS_Y: Color32 = Color32::from_rgb(60, 200, 60);
    pub const AXIS_Z: Color32 = Color32::from_rgb(60, 100, 230);

    // Entity type colors for outliner icons
    pub const MESH_ICON: Color32 = Color32::from_rgb(100, 180, 255);
    pub const LIGHT_ICON: Color32 = Color32::from_rgb(255, 220, 100);
    pub const AUDIO_ICON: Color32 = Color32::from_rgb(100, 255, 180);
    pub const SCRIPT_ICON: Color32 = Color32::from_rgb(200, 150, 255);
    pub const CAMERA_ICON: Color32 = Color32::from_rgb(255, 150, 100);
    pub const TERRAIN_ICON: Color32 = Color32::from_rgb(120, 200, 100);
    pub const PARTICLE_ICON: Color32 = Color32::from_rgb(255, 180, 80);
}

/// Styled section header for editor panels.
pub fn section_header(ui: &mut egui::Ui, title: &str) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add(egui::Separator::default().horizontal());
        ui.label(
            egui::RichText::new(title)
                .strong()
                .color(OchromaColors::TEXT_PRIMARY)
                .size(13.0),
        );
        ui.add(egui::Separator::default().horizontal());
    });
    ui.add_space(2.0);
}

/// Styled property row: label on left, widget on right.
pub fn property_row(ui: &mut egui::Ui, label: &str) -> egui::InnerResponse<()> {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .color(OchromaColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.add_space((ui.available_width() - 160.0).max(0.0));
    })
}

/// Styled toolbar button.
pub fn toolbar_button(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let text = if active {
        egui::RichText::new(label).color(Color32::WHITE).strong()
    } else {
        egui::RichText::new(label).color(OchromaColors::TEXT_SECONDARY)
    };

    let button = if active {
        egui::Button::new(text).fill(OchromaColors::ACCENT)
    } else {
        egui::Button::new(text).fill(OchromaColors::BG_WIDGET)
    };

    ui.add(button).clicked()
}

/// Styled icon label for the outliner.
pub fn entity_icon(entity_type: &str) -> (char, Color32) {
    match entity_type {
        "mesh" | "model" => ('\u{25C6}', OchromaColors::MESH_ICON),
        "light" => ('\u{2600}', OchromaColors::LIGHT_ICON),
        "audio" => ('\u{266A}', OchromaColors::AUDIO_ICON),
        "script" => ('\u{26A1}', OchromaColors::SCRIPT_ICON),
        "camera" => ('\u{1F4F7}', OchromaColors::CAMERA_ICON),
        "terrain" => ('\u{25B2}', OchromaColors::TERRAIN_ICON),
        "particle" => ('\u{2726}', OchromaColors::PARTICLE_ICON),
        _ => ('\u{25CB}', OchromaColors::TEXT_SECONDARY),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_applies_without_panic() {
        let ctx = egui::Context::default();
        apply_ochroma_theme(&ctx);
        // If we got here, no panic occurred
    }

    #[test]
    fn color_constants_are_correct() {
        assert_eq!(OchromaColors::BG_DARKEST, Color32::from_rgb(18, 18, 24));
        assert_eq!(OchromaColors::ACCENT, Color32::from_rgb(60, 130, 255));
        assert_eq!(OchromaColors::ERROR, Color32::from_rgb(255, 70, 70));
    }

    #[test]
    fn section_header_renders() {
        let ctx = egui::Context::default();
        apply_ochroma_theme(&ctx);
        ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                section_header(ui, "Transform");
            });
        });
    }
}
