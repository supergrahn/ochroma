//! egui window for the material node graph editor.

use crate::material_editor::{MaterialGraph, MaterialEditorNode, MaterialNodeType, MaterialConnection};

pub struct MaterialEditorUi {
    pub open: bool,
    pub graph: Option<MaterialGraph>,
    selected_node: Option<u32>,
    zoom: f32,
    scroll: egui::Vec2,
}

impl MaterialEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: None,
            selected_node: None,
            zoom: 1.0,
            scroll: egui::Vec2::ZERO,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        egui::Window::new("Material Editor")
            .open(&mut self.open)
            .default_size([800.0, 500.0])
            .resizable(true)
            .show(ctx, |ui| {
                if self.graph.is_none() {
                    ui.centered_and_justified(|ui| {
                        ui.label("No material loaded.");
                    });
                    if ui.button("New Material").clicked() {
                        self.graph = Some(MaterialGraph::new("New Material"));
                    }
                    return;
                }

                let graph = self.graph.as_ref().unwrap();
                let node_count = graph.nodes.len();
                let conn_count = graph.connections.len();
                let graph_name = graph.name.clone();

                ui.horizontal(|ui| {
                    ui.heading(&graph_name);
                    ui.separator();
                    ui.label(format!("{} nodes  {} connections", node_count, conn_count));
                });
                ui.separator();

                // Collect node data to avoid borrow issues
                let nodes: Vec<(u32, String, [f32; 2])> = self.graph.as_ref().unwrap().nodes.iter()
                    .map(|n| (n.id, node_type_label(&n.node_type).to_string(), n.position))
                    .collect();
                let connections: Vec<(u32, u32)> = self.graph.as_ref().unwrap().connections.iter()
                    .map(|c| (c.from_node, c.to_node))
                    .collect();

                egui::SidePanel::left("mat_node_list").default_width(160.0).show_inside(ui, |ui| {
                    ui.label(egui::RichText::new("Nodes").strong());
                    for (id, label, _) in &nodes {
                        let selected = self.selected_node == Some(*id);
                        if ui.selectable_label(selected, format!("#{} {}", id, label)).clicked() {
                            self.selected_node = Some(*id);
                        }
                    }
                });

                egui::SidePanel::right("mat_props").default_width(200.0).show_inside(ui, |ui| {
                    ui.label(egui::RichText::new("Properties").strong());
                    if let Some(sel_id) = self.selected_node {
                        if let Some(node) = self.graph.as_mut().unwrap().nodes.iter_mut().find(|n| n.id == sel_id) {
                            ui.label(format!("#{} {}", node.id, node_type_label(&node.node_type)));
                            ui.separator();
                            node_properties_ui(ui, &mut node.node_type);
                        }
                    } else {
                        ui.label("Select a node.");
                    }
                });

                egui::CentralPanel::default().show_inside(ui, |ui| {
                    let painter = ui.painter();
                    let origin = ui.min_rect().min;

                    // Draw connections
                    for (from_id, to_id) in &connections {
                        if let (Some((_, _, from_pos)), Some((_, _, to_pos))) = (
                            nodes.iter().find(|(id, _, _)| id == from_id),
                            nodes.iter().find(|(id, _, _)| id == to_id),
                        ) {
                            let from = egui::pos2(origin.x + from_pos[0] + 100.0, origin.y + from_pos[1] + 20.0);
                            let to = egui::pos2(origin.x + to_pos[0], origin.y + to_pos[1] + 20.0);
                            painter.line_segment([from, to], egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 140, 200)));
                        }
                    }

                    // Draw node boxes
                    for (id, label, pos) in &nodes {
                        let x = origin.x + pos[0];
                        let y = origin.y + pos[1];
                        let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(100.0, 40.0));
                        let fill = if self.selected_node == Some(*id) {
                            egui::Color32::from_rgb(40, 70, 120)
                        } else {
                            egui::Color32::from_rgb(30, 35, 50)
                        };
                        painter.rect_filled(rect, 4.0, fill);
                        painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 90, 130)), egui::StrokeKind::Outside);
                        painter.text(rect.center(), egui::Align2::CENTER_CENTER, label,
                            egui::FontId::proportional(11.0), egui::Color32::WHITE);
                    }
                });
            });
    }
}

fn node_type_label(t: &MaterialNodeType) -> &'static str {
    match t {
        MaterialNodeType::MaterialOutput        => "Output",
        MaterialNodeType::SpectralConstant {..} => "Spectral",
        MaterialNodeType::FloatConstant {..}    => "Float",
        MaterialNodeType::ColorConstant {..}    => "Color",
        MaterialNodeType::TextureCoordinate     => "TexCoord",
        MaterialNodeType::TextureSample {..}    => "Texture",
        MaterialNodeType::Add                   => "Add",
        MaterialNodeType::Subtract              => "Subtract",
        MaterialNodeType::Multiply              => "Multiply",
        MaterialNodeType::Divide                => "Divide",
        MaterialNodeType::Lerp                  => "Lerp",
        MaterialNodeType::Power                 => "Power",
        MaterialNodeType::Sqrt                  => "Sqrt",
        MaterialNodeType::Abs                   => "Abs",
        MaterialNodeType::OneMinus              => "1-x",
        MaterialNodeType::SpectralBlend {..}    => "SpectralBlend",
        MaterialNodeType::SpectralShift {..}    => "SpectralShift",
        MaterialNodeType::WearBlend {..}        => "Wear",
        MaterialNodeType::FresnelEffect {..}    => "Fresnel",
        MaterialNodeType::Roughness {..}        => "Roughness",
        MaterialNodeType::Metallic {..}         => "Metallic",
        MaterialNodeType::Emission {..}         => "Emission",
        MaterialNodeType::Opacity {..}          => "Opacity",
        MaterialNodeType::PerlinNoise {..}      => "Perlin",
        MaterialNodeType::VoronoiNoise {..}     => "Voronoi",
        MaterialNodeType::Checker {..}          => "Checker",
        MaterialNodeType::Gradient {..}         => "Gradient",
        MaterialNodeType::Remap {..}            => "Remap",
        MaterialNodeType::SmoothStep {..}       => "SmoothStep",
        MaterialNodeType::Time                  => "Time",
    }
}

fn node_properties_ui(ui: &mut egui::Ui, node_type: &mut MaterialNodeType) {
    match node_type {
        MaterialNodeType::FloatConstant { value } => {
            ui.horizontal(|ui| { ui.label("Value:"); ui.add(egui::DragValue::new(value).speed(0.01)); });
        }
        MaterialNodeType::ColorConstant { r, g, b } => {
            ui.horizontal(|ui| { ui.label("R:"); ui.add(egui::DragValue::new(r).speed(0.01).range(0.0..=1.0)); });
            ui.horizontal(|ui| { ui.label("G:"); ui.add(egui::DragValue::new(g).speed(0.01).range(0.0..=1.0)); });
            ui.horizontal(|ui| { ui.label("B:"); ui.add(egui::DragValue::new(b).speed(0.01).range(0.0..=1.0)); });
        }
        MaterialNodeType::Roughness { value } => {
            ui.horizontal(|ui| { ui.label("Roughness:"); ui.add(egui::DragValue::new(value).speed(0.01).range(0.0..=1.0)); });
        }
        MaterialNodeType::Emission { intensity } => {
            ui.horizontal(|ui| { ui.label("Intensity:"); ui.add(egui::DragValue::new(intensity).speed(0.1)); });
        }
        MaterialNodeType::SpectralBlend { factor } => {
            ui.horizontal(|ui| { ui.label("Factor:"); ui.add(egui::DragValue::new(factor).speed(0.01).range(0.0..=1.0)); });
        }
        _ => { ui.label("No editable properties."); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_ui_default_open_false() {
        let ui = MaterialEditorUi::new();
        assert!(!ui.open);
        assert!(ui.graph.is_none());
    }
}
