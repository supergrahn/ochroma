use crate::tokens::Tokens;
use egui::Color32;

/// Apply the Ochroma 2026 dark theme to an egui context.
///
/// This is now a thin SHIM over the token design system: it applies the default
/// (dark) [`Tokens`] via [`crate::egui_theme::apply`]. Call sites in
/// `editor.rs`/`main.rs` build unchanged through the migration. For a custom
/// theme (e.g. light, or a hot-reloaded JSON), call `egui_theme::apply` directly.
pub fn apply_ochroma_theme(ctx: &egui::Context) {
    crate::egui_theme::apply(ctx, &Tokens::default());
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
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                section_header(ui, "Transform");
            });
        });
    }
}
