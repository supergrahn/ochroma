use crate::editor_app::{AssetId, GraphNodeId, NodeGraphDiff};

/// State for the node-graph change badge and on-demand reveal panel.
pub struct NodeGraphPanelState {
    badge_count: usize,
    revealed: bool,
    highlighted: Vec<GraphNodeId>,
    open_asset: Option<AssetId>,
}

impl NodeGraphPanelState {
    pub fn new() -> Self {
        Self {
            badge_count: 0,
            revealed: false,
            highlighted: Vec::new(),
            open_asset: None,
        }
    }

    pub fn badge_count(&self) -> usize { self.badge_count }

    pub fn is_revealed(&self) -> bool { self.revealed }

    pub fn highlighted_nodes(&self) -> &[GraphNodeId] { &self.highlighted }

    pub fn open_asset(&self) -> Option<AssetId> { self.open_asset }

    pub fn notify_diff(&mut self, diff: NodeGraphDiff) {
        self.badge_count = diff.changed_count();
        let mut highlights: Vec<GraphNodeId> = diff.added_nodes().to_vec();
        highlights.extend_from_slice(diff.modified_nodes());
        self.highlighted = highlights;
        self.open_asset = Some(diff.asset_id());
    }

    pub fn open_reveal(&mut self) {
        self.revealed = true;
    }

    pub fn close_reveal(&mut self) {
        self.revealed = false;
        self.badge_count = 0;
    }

    pub fn show_badge(&self, ui: &mut egui::Ui) -> bool {
        if self.badge_count == 0 {
            return false;
        }
        let label = format!("⬡ {} nodes changed", self.badge_count);
        ui.add(
            egui::Label::new(
                egui::RichText::new(label)
                    .color(egui::Color32::from_rgb(100, 160, 255))
                    .small()
            )
        ).clicked()
    }

    pub fn show_reveal_panel(&mut self, ui: &mut egui::Ui) {
        if !self.revealed {
            return;
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Node Graph")
                    .strong()
                    .color(egui::Color32::from_rgb(100, 160, 255))
            );
            if ui.small_button("✕").clicked() {
                self.close_reveal();
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
            for node_id in &self.highlighted {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("+ ")
                            .color(egui::Color32::from_rgb(100, 200, 100))
                    );
                    ui.label(format!("Node #{}", node_id.0));
                    ui.label(
                        egui::RichText::new("AI")
                            .small()
                            .color(egui::Color32::from_rgb(100, 160, 255))
                    );
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diff(added: usize, modified: usize) -> NodeGraphDiff {
        NodeGraphDiff::new(
            AssetId(1),
            (0..added).map(|i| GraphNodeId(i as u32)).collect(),
            (100..100 + modified).map(|i| GraphNodeId(i as u32)).collect(),
        )
    }

    #[test]
    fn panel_starts_with_no_badge() {
        let panel = NodeGraphPanelState::new();
        assert_eq!(panel.badge_count(), 0);
    }

    #[test]
    fn notify_diff_sets_badge_count() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(3, 2));
        assert_eq!(panel.badge_count(), 5);
    }

    #[test]
    fn panel_starts_closed() {
        let panel = NodeGraphPanelState::new();
        assert!(!panel.is_revealed());
    }

    #[test]
    fn open_reveal_sets_revealed() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(1, 0));
        panel.open_reveal();
        assert!(panel.is_revealed());
    }

    #[test]
    fn close_reveal_clears_revealed_and_badge() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(2, 1));
        panel.open_reveal();
        panel.close_reveal();
        assert!(!panel.is_revealed());
        assert_eq!(panel.badge_count(), 0);
    }

    #[test]
    fn highlighted_nodes_match_diff_after_open() {
        let mut panel = NodeGraphPanelState::new();
        let diff = make_diff(2, 1);
        panel.notify_diff(diff);
        panel.open_reveal();
        assert_eq!(panel.highlighted_nodes().len(), 3);
    }
}
