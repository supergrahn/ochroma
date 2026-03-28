//! egui window for the material node graph editor.
//! Uses OchrGraph (crucible-core backend) + NodeGraphWidget (egui rendering).

use vox_nodes::{OchrGraph, NodeId};
use vox_nodes::mat_nodes::{
    FloatConstNode, MaterialOutputNode, MultiplyNode, AddNode,
};
use vox_ui::node_graph_widget::{NodeGraphWidget, VisualPin, VisualPinType};

pub struct MaterialEditorUi {
    pub open: bool,
    pub name: String,
    pub graph: OchrGraph,
    widget: NodeGraphWidget,
    selected_node: Option<NodeId>,
}

impl MaterialEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            name: String::new(),
            graph: OchrGraph::new(),
            widget: NodeGraphWidget::new(),
            selected_node: None,
        }
    }

    /// Build a default Roughness + Metallic → MaterialOutput graph.
    pub fn create_default_graph(&mut self) {
        self.graph = OchrGraph::new();
        self.name = "New Material".to_string();

        let roughness_id = self.graph.add_node(
            "Roughness", Box::new(FloatConstNode::new(0.5)), [80.0, 60.0],
        );
        let metallic_id = self.graph.add_node(
            "Metallic", Box::new(FloatConstNode::new(0.0)), [80.0, 170.0],
        );
        let output_id = self.graph.add_node(
            "Output", Box::new(MaterialOutputNode), [320.0, 110.0],
        );

        let _ = self.graph.connect(roughness_id, "out", output_id, "roughness");
        let _ = self.graph.connect(metallic_id,  "out", output_id, "metallic");
        let _ = self.graph.graph.cook();
        self.sync_widget();
    }

    /// Rebuild the NodeGraphWidget from current OchrGraph state.
    fn sync_widget(&mut self) {
        self.widget = NodeGraphWidget::new();
        let snap = self.graph.graph.snapshot();

        for mut vn in self.graph.to_visual_nodes() {
            if let Some(ns) = snap.nodes.iter().find(|ns| ns.id == vn.id) {
                match ns.type_name.as_str() {
                    "FloatConst" => {
                        vn.outputs = vec![VisualPin {
                            name: "out".into(),
                            pin_type: VisualPinType::Float,
                            connected: false,
                        }];
                        vn.color = [45, 110, 65];
                        vn.size = [130.0, 55.0];
                    }
                    "Multiply" | "Add" => {
                        vn.inputs = vec![
                            VisualPin { name: "a".into(), pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "b".into(), pin_type: VisualPinType::Float, connected: false },
                        ];
                        vn.outputs = vec![VisualPin { name: "out".into(), pin_type: VisualPinType::Float, connected: false }];
                        vn.color = [80, 55, 110];
                        vn.size = [130.0, 80.0];
                    }
                    "OneMinus" => {
                        vn.inputs  = vec![VisualPin { name: "input".into(), pin_type: VisualPinType::Float, connected: false }];
                        vn.outputs = vec![VisualPin { name: "out".into(),   pin_type: VisualPinType::Float, connected: false }];
                        vn.color = [80, 55, 110];
                        vn.size = [130.0, 60.0];
                    }
                    "MaterialOutput" => {
                        vn.inputs = vec![
                            VisualPin { name: "base_r".into(),    pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "base_g".into(),    pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "base_b".into(),    pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "roughness".into(), pin_type: VisualPinType::Float, connected: false },
                            VisualPin { name: "metallic".into(),  pin_type: VisualPinType::Float, connected: false },
                        ];
                        vn.color = [140, 55, 55];
                        vn.size = [160.0, 175.0];
                    }
                    _ => {}
                }
            }
            self.widget.add_node(vn);
        }

        for vc in self.graph.to_visual_connections() {
            self.widget.add_connection(vc);
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        egui::Window::new("Material Editor")
            .default_size([950.0, 560.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if self.name.is_empty() {
                        if ui.button("New Material").clicked() {
                            self.create_default_graph();
                        }
                        return;
                    }
                    ui.heading(self.name.clone());
                    ui.separator();
                    ui.label(format!("{} nodes", self.graph.graph.node_count()));
                    ui.separator();
                    if ui.button("+ Float").clicked() {
                        let n = self.graph.graph.node_count() as f32;
                        let _ = self.graph.add_node(
                            "Float", Box::new(FloatConstNode::new(1.0)),
                            [50.0 + n * 20.0, 50.0 + n * 20.0],
                        );
                        let _ = self.graph.graph.cook();
                        self.sync_widget();
                    }
                    if ui.button("+ Multiply").clicked() {
                        let n = self.graph.graph.node_count() as f32;
                        let _ = self.graph.add_node(
                            "Multiply", Box::new(MultiplyNode),
                            [200.0 + n * 10.0, 50.0],
                        );
                        let _ = self.graph.graph.cook();
                        self.sync_widget();
                    }
                    if ui.button("+ Add").clicked() {
                        let n = self.graph.graph.node_count() as f32;
                        let _ = self.graph.add_node(
                            "Add", Box::new(AddNode), [200.0 + n * 10.0, 100.0],
                        );
                        let _ = self.graph.graph.cook();
                        self.sync_widget();
                    }
                    if ui.button("Cook").clicked() {
                        let _ = self.graph.graph.cook();
                    }
                });

                if self.name.is_empty() { return; }
                ui.separator();

                let actions = self.widget.show_egui(ui);

                for action in actions {
                    use vox_ui::node_graph_widget::NodeGraphAction;
                    match action {
                        NodeGraphAction::NodeMoved { id, new_pos } => {
                            self.graph.set_position(NodeId(id), new_pos);
                        }
                        NodeGraphAction::NodeSelected { id } => {
                            self.selected_node = Some(NodeId(id));
                        }
                        NodeGraphAction::ConnectionCreated { from_node, from_pin, to_node, to_pin } => {
                            let _ = self.graph.connect(
                                NodeId(from_node), &from_pin,
                                NodeId(to_node),   &to_pin,
                            );
                            let _ = self.graph.graph.cook();
                            self.sync_widget();
                        }
                        NodeGraphAction::NodeDeleted { id } => {
                            self.widget.remove_node(id);
                        }
                        _ => {}
                    }
                }
            });
    }
}

impl Default for MaterialEditorUi {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_editor_starts_empty() {
        let ed = MaterialEditorUi::new();
        assert!(!ed.open);
        assert_eq!(ed.graph.graph.node_count(), 0);
    }

    #[test]
    fn create_default_graph_has_nodes() {
        let mut ed = MaterialEditorUi::new();
        ed.create_default_graph();
        assert_eq!(ed.graph.graph.node_count(), 3);
    }

    #[test]
    fn create_default_graph_cooks_cleanly() {
        let mut ed = MaterialEditorUi::new();
        ed.create_default_graph();
        ed.graph.graph.cook().unwrap();
    }
}
