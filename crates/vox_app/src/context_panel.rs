use crate::editor_app::WorkspaceMode;

/// The three fixed tabs in the right sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarTab {
    Context,
    Scene,
    Assets,
}

/// The right-side 3-tab sidebar.
pub struct ContextPanel {
    active_tab: SidebarTab,
}

impl ContextPanel {
    pub fn new() -> Self {
        Self { active_tab: SidebarTab::Context }
    }

    pub fn active_tab(&self) -> SidebarTab {
        self.active_tab
    }

    pub fn set_tab(&mut self, tab: SidebarTab) {
        self.active_tab = tab;
    }

    /// Returns the display label for the Context tab's content area given a mode.
    pub fn context_label_for_mode(mode: WorkspaceMode) -> &'static str {
        match mode {
            WorkspaceMode::Sculpt    => "Sculpt Tools",
            WorkspaceMode::Objects   => "Object Properties",
            WorkspaceMode::Lighting  => "Lighting Controls",
            WorkspaceMode::Animate   => "Animation",
            WorkspaceMode::Logic     => "Logic",
            WorkspaceMode::Simulate  => "Simulation",
        }
    }

    /// Render the full right sidebar panel.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        mode: WorkspaceMode,
        show_context: impl FnOnce(&mut egui::Ui),
        show_scene: impl FnOnce(&mut egui::Ui),
        show_assets: impl FnOnce(&mut egui::Ui),
    ) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                for (tab, label) in [
                    (SidebarTab::Context, "Context"),
                    (SidebarTab::Scene,   "Scene"),
                    (SidebarTab::Assets,  "Assets"),
                ] {
                    let selected = self.active_tab == tab;
                    if ui.selectable_label(selected, label).clicked() {
                        self.active_tab = tab;
                    }
                }
            });

            ui.separator();

            match self.active_tab {
                SidebarTab::Context => {
                    ui.label(Self::context_label_for_mode(mode));
                    ui.separator();
                    show_context(ui);
                }
                SidebarTab::Scene  => show_scene(ui),
                SidebarTab::Assets => show_assets(ui),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_panel_default_tab_is_context() {
        let panel = ContextPanel::new();
        assert_eq!(panel.active_tab(), SidebarTab::Context);
    }

    #[test]
    fn context_panel_tab_switch_stores_selection() {
        let mut panel = ContextPanel::new();
        panel.set_tab(SidebarTab::Assets);
        assert_eq!(panel.active_tab(), SidebarTab::Assets);
        panel.set_tab(SidebarTab::Scene);
        assert_eq!(panel.active_tab(), SidebarTab::Scene);
    }

    #[test]
    fn context_label_for_sculpt_mode_is_sculpt_tools() {
        assert_eq!(
            ContextPanel::context_label_for_mode(WorkspaceMode::Sculpt),
            "Sculpt Tools"
        );
    }

    #[test]
    fn context_label_for_simulate_mode_is_simulation() {
        assert_eq!(
            ContextPanel::context_label_for_mode(WorkspaceMode::Simulate),
            "Simulation"
        );
    }
}
