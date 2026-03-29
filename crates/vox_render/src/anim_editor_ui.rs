//! Animation state machine editor window.

use vox_nodes::{OchrGraph, NodeId};
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

    /// Rebuild the NodeGraphWidget from current OchrGraph state.
    fn sync_widget(&mut self) {
        self.widget = NodeGraphWidget::new();
        for vn in self.graph.to_visual_nodes() {
            self.widget.add_node(vn);
        }
        for vc in self.graph.to_visual_connections() {
            self.widget.add_connection(vc);
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
                let actions = self.widget.show_egui(ui);
                for action in actions {
                    if let vox_ui::node_graph_widget::NodeGraphAction::NodeDeleted { id } = action {
                        let _ = self.graph.remove_node(NodeId(id));
                        self.sync_widget();
                    }
                }
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
