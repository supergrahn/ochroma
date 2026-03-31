use crate::editor_app::WorkspaceMode;

/// The left-rail vertical icon bar that switches the active workspace mode.
pub struct ModeStrip {
    active: WorkspaceMode,
}

impl ModeStrip {
    pub fn new() -> Self {
        Self { active: WorkspaceMode::Objects }
    }

    pub fn active_mode(&self) -> WorkspaceMode {
        self.active
    }

    /// Set the active mode. Returns `true` if the mode actually changed.
    pub fn set_mode(&mut self, mode: WorkspaceMode) -> bool {
        if self.active == mode {
            return false;
        }
        self.active = mode;
        true
    }

    /// Render the mode strip as a vertical egui panel.
    /// Returns `Some(mode)` if the user clicked a different mode, `None` otherwise.
    pub fn show(&mut self, ui: &mut egui::Ui) -> Option<WorkspaceMode> {
        let modes: &[(WorkspaceMode, &str, &str)] = &[
            (WorkspaceMode::Sculpt,   "⬡", "Sculpt"),
            (WorkspaceMode::Objects,  "⬜", "Objects"),
            (WorkspaceMode::Lighting, "☀", "Lighting"),
            (WorkspaceMode::Animate,  "▶", "Animate"),
            (WorkspaceMode::Logic,    "⬡", "Logic"),
            (WorkspaceMode::Simulate, "⏵", "Simulate"),
        ];

        let mut clicked = None;
        ui.vertical(|ui| {
            ui.set_width(44.0);
            for (mode, icon, label) in modes {
                let selected = self.active == *mode;
                let btn = egui::Button::new(*icon)
                    .min_size(egui::vec2(36.0, 36.0))
                    .selected(selected);
                let resp = ui.add(btn).on_hover_text(*label);
                if resp.clicked() && !selected {
                    self.active = *mode;
                    clicked = Some(*mode);
                }
            }
        });
        clicked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_strip_default_is_objects() {
        let strip = ModeStrip::new();
        assert_eq!(strip.active_mode(), WorkspaceMode::Objects);
    }

    #[test]
    fn mode_strip_set_mode_returns_event_on_change() {
        let mut strip = ModeStrip::new();
        let changed = strip.set_mode(WorkspaceMode::Lighting);
        assert!(changed);
        assert_eq!(strip.active_mode(), WorkspaceMode::Lighting);
    }

    #[test]
    fn mode_strip_set_mode_returns_false_if_same() {
        let mut strip = ModeStrip::new();
        let changed = strip.set_mode(WorkspaceMode::Objects);
        assert!(!changed);
    }

    #[test]
    fn mode_strip_all_six_modes_are_settable() {
        let mut strip = ModeStrip::new();
        for mode in [
            WorkspaceMode::Sculpt,
            WorkspaceMode::Objects,
            WorkspaceMode::Lighting,
            WorkspaceMode::Animate,
            WorkspaceMode::Logic,
            WorkspaceMode::Simulate,
        ] {
            strip.set_mode(mode);
            assert_eq!(strip.active_mode(), mode);
        }
    }
}
