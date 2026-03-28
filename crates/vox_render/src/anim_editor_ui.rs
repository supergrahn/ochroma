//! Animation state machine editor window.

use vox_nodes::OchrGraph;
use vox_ui::node_graph_widget::NodeGraphWidget;

pub struct AnimEditorUi {
    pub open: bool,
    pub graph: OchrGraph,
    widget: NodeGraphWidget,
}

impl AnimEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: OchrGraph::new(),
            widget: NodeGraphWidget::new(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }
        egui::Window::new("Animation Editor")
            .default_size([950.0, 560.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Animation State Machine");
                    ui.separator();
                    ui.label(format!("{} states", self.graph.graph.node_count()));
                });
                ui.separator();
                let _actions = self.widget.show_egui(ui);
            });
    }
}

impl Default for AnimEditorUi {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anim_editor_has_ochrgraph() {
        let ed = AnimEditorUi::new();
        assert_eq!(ed.graph.graph.node_count(), 0);
    }
}
