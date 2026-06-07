//! Coherent vector icon family (egui-phosphor 0.9, egui-0.31 compatible).
//!
//! Replaces `theme::entity_icon`'s single placeholder Unicode codepoints with a
//! real vector font merged into the egui glyph atlas so icons sit inline with
//! proportional text. [`install`] merges the Phosphor Regular family (the
//! default egui-phosphor feature); [`icon`] maps editor roles to codepoints.

/// Merge the Phosphor icon font into an egui context so `icon::*` codepoints
/// render as vector glyphs inline with text. Idempotent-safe to call once at
/// startup.
pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}

/// Role -> codepoint map (the design's ICONS table). These are `&str` because
/// each Phosphor glyph is a multi-byte codepoint that composes into egui text.
pub mod icon {
    use egui_phosphor::regular as r;

    pub const MOVE: &str = r::ARROWS_OUT_CARDINAL;
    pub const ROTATE: &str = r::ARROWS_CLOCKWISE;
    pub const SCALE: &str = r::CORNERS_OUT;
    pub const WORLD: &str = r::GLOBE;
    pub const LOCAL: &str = r::CROSSHAIR;
    pub const SNAP: &str = r::MAGNET;
    pub const SHOW_FLAGS: &str = r::EYE;
    pub const PERF: &str = r::GAUGE;
    pub const SEARCH: &str = r::MAGNIFYING_GLASS;
    pub const MESH: &str = r::CUBE;
    pub const LIGHT: &str = r::LIGHTBULB;
    pub const AUDIO: &str = r::SPEAKER_HIGH;
    pub const SCRIPT: &str = r::CODE;
    pub const CAMERA: &str = r::VIDEO_CAMERA;
    pub const TERRAIN: &str = r::MOUNTAINS;
    pub const PARTICLE: &str = r::SPARKLE;
    pub const FOLDER: &str = r::FOLDER;
    pub const FOLDER_OPEN: &str = r::FOLDER_OPEN;
    pub const PLAY: &str = r::PLAY;
    pub const PAUSE: &str = r::PAUSE;
    pub const STOP: &str = r::STOP;
    pub const ADD: &str = r::PLUS;
    pub const SETTINGS: &str = r::GEAR;
    pub const NODE_GRAPH: &str = r::TREE_STRUCTURE;
    pub const HIERARCHY: &str = r::STACK;
    pub const INSPECTOR: &str = r::LIST;
    pub const FILE: &str = r::FILE;
    pub const IMAGE: &str = r::IMAGE;
    pub const CONSOLE: &str = r::TERMINAL_WINDOW;
    pub const WARNING: &str = r::WARNING;
}

/// Resolve an entity-type string to its (icon codepoint, dotted color token
/// key) pair — the token-aware replacement for `theme::entity_icon`.
pub fn entity_icon(entity_type: &str) -> (&'static str, &'static str) {
    match entity_type {
        "mesh" | "model" => (icon::MESH, "accent.base"),
        "light" => (icon::LIGHT, "status.warning"),
        "audio" => (icon::AUDIO, "status.success"),
        "script" => (icon::SCRIPT, "port.split_weights"),
        "camera" => (icon::CAMERA, "port.lod_mesh"),
        "terrain" => (icon::TERRAIN, "port.terrain"),
        "particle" => (icon::PARTICLE, "status.warning"),
        _ => (icon::FILE, "text.secondary"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icons_are_distinct_codepoints() {
        // >=6 role icons produce distinct codepoints (the precondition for the
        // distinct pixel signatures asserted in vox_app's snapshot tests).
        let set: std::collections::HashSet<&str> = [
            icon::TERRAIN,
            icon::MESH,
            icon::LIGHT,
            icon::AUDIO,
            icon::SCRIPT,
            icon::CAMERA,
            icon::SEARCH,
        ]
        .into_iter()
        .collect();
        assert!(set.len() >= 6, "expected >=6 distinct icon codepoints, got {}", set.len());
    }

    #[test]
    fn install_registers_phosphor_family() {
        let ctx = egui::Context::default();
        install(&ctx);
        // The merged font must produce a non-empty galley for an icon glyph —
        // i.e. the codepoint is in the atlas, not a tofu box.
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let galley = ui.painter().layout_no_wrap(
                    icon::TERRAIN.to_string(),
                    egui::FontId::proportional(20.0),
                    egui::Color32::WHITE,
                );
                assert!(galley.size().x > 1.0, "phosphor glyph laid out to zero width");
            });
        });
    }
}
