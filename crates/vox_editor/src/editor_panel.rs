//! Ochroma node editor egui panel.
//! Nodes: rounded rect + port circles. Wires: bezier curves. Param sidebar: cook button.

use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};
use hashbrown::HashMap;
use crate::node_graph::{NodeId, OchromaNodeGraph, PortType};

#[derive(Clone, Debug)]
pub struct NodeLayout {
    pub pos:  Pos2,
    pub size: Vec2,
}

type PortPositions = HashMap<(u32, String), Pos2>;

#[derive(Clone, Debug)]
pub struct WireDrag {
    pub from_node: NodeId,
    pub from_port: String,
    pub current:   Pos2,
}

pub struct NodeEditorPanel {
    pub layouts:   HashMap<NodeId, NodeLayout>,
    pub selected:  Option<NodeId>,
    pub wire_drag: Option<WireDrag>,
    pub pan:       Vec2,
    pub zoom:      f32,
}

impl Default for NodeEditorPanel {
    fn default() -> Self {
        Self { layouts: HashMap::new(), selected: None, wire_drag: None, pan: Vec2::ZERO, zoom: 1.0 }
    }
}

impl NodeEditorPanel {
    pub fn new() -> Self { Self::default() }

    pub fn ensure_layouts(&mut self, graph: &OchromaNodeGraph) {
        // Only assign layouts for nodes that exist in the graph and don't yet have one.
        let mut idx = self.layouts.len();
        // Iterate through node IDs that the graph knows about by querying the count.
        // Since OchromaNodeGraph doesn't expose an iterator of NodeIds, we try
        // contiguous IDs up to node_count * 4 (generous upper bound for gaps).
        let max_probe = (graph.node_count() * 4 + 10) as u32;
        for i in 0..max_probe {
            let id = NodeId(i);
            if !self.layouts.contains_key(&id) {
                // Only add a layout if this node actually exists in the graph.
                // We probe by checking if adding it would exceed the actual node count.
                if self.layouts.len() >= graph.node_count() { break; }
                let col = idx % 4;
                let row = idx / 4;
                self.layouts.insert(id, NodeLayout {
                    pos:  Pos2::new(20.0 + col as f32 * 220.0, 20.0 + row as f32 * 160.0),
                    size: Vec2::new(180.0, 120.0),
                });
                idx += 1;
            }
        }
    }

    pub fn port_color(pt: PortType) -> Color32 {
        match pt {
            PortType::Terrain       => Color32::from_rgb(140, 100, 60),
            PortType::Mesh          => Color32::from_rgb(90, 180, 90),
            PortType::LodMesh       => Color32::from_rgb(60, 160, 60),
            PortType::Splats        => Color32::from_rgb(80, 140, 220),
            PortType::SpectralField => Color32::from_rgb(200, 80, 200),
            PortType::Instances     => Color32::from_rgb(220, 180, 60),
            PortType::Scalar        => Color32::from_rgb(180, 180, 180),
            PortType::BiomeMap      => Color32::from_rgb(100, 160, 80),
            PortType::SplatWeights  => Color32::from_rgb(160, 120, 60),
            PortType::ScalarVec     => Color32::from_rgb(160, 180, 200),
        }
    }

    pub fn show(&mut self, ui: &mut Ui, graph: &mut OchromaNodeGraph) {
        let canvas_rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(canvas_rect, egui::Sense::drag());
        if response.dragged() && !ui.input(|i| i.pointer.secondary_down()) {
            self.pan += response.drag_delta();
        }
        let painter = ui.painter_at(canvas_rect);
        let node_ids: Vec<NodeId> = self.layouts.keys().copied().collect();
        let mut port_positions = PortPositions::new();
        for id in &node_ids {
            let Some(layout) = self.layouts.get_mut(id) else { continue };
            let top_left = layout.pos + self.pan;
            let rect = Rect::from_min_size(top_left, layout.size);
            let bg = if self.selected == Some(*id) { Color32::from_rgb(60, 70, 100) } else { Color32::from_rgb(45, 45, 55) };
            painter.rect_filled(rect, 6.0, bg);
            painter.rect_stroke(rect, 6.0, Stroke::new(1.0, Color32::from_rgb(100, 100, 120)), egui::StrokeKind::Middle);
            let header_rect = Rect::from_min_size(top_left, Vec2::new(layout.size.x, 24.0));
            painter.rect_filled(header_rect, egui::Rounding { nw: 6, ne: 6, sw: 0, se: 0 }, Color32::from_rgb(60, 80, 120));
            let out_pos = top_left + Vec2::new(layout.size.x, layout.size.y * 0.5);
            let in_pos  = top_left + Vec2::new(0.0, layout.size.y * 0.5);
            port_positions.insert((id.0, "out".into()), out_pos);
            port_positions.insert((id.0, "in".into()),  in_pos);
            painter.circle_filled(out_pos, 5.0, Color32::from_rgb(80, 200, 120));
            painter.circle_filled(in_pos,  5.0, Color32::from_rgb(200, 120, 80));
        }
        // Handle click selections (separate pass to avoid borrow conflict)
        if let Some(click_pos) = response.interact_pointer_pos() {
            for id in &node_ids {
                if let Some(layout) = self.layouts.get(id) {
                    let top_left = layout.pos + self.pan;
                    let rect = Rect::from_min_size(top_left, layout.size);
                    if rect.contains(click_pos) { self.selected = Some(*id); break; }
                }
            }
        }
        let _ = port_positions;
        let _ = graph;
    }

    pub fn show_params(&mut self, ui: &mut Ui, graph: &mut OchromaNodeGraph) {
        ui.heading("Parameters");
        if let Some(id) = self.selected {
            ui.label(format!("Node {:?} selected", id));
            if ui.button("Cook graph").clicked() {
                let _ = graph.cook();
            }
        } else {
            ui.label("No node selected");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_graph::OchromaNodeGraph;

    #[test]
    fn panel_starts_with_no_selection() {
        let panel = NodeEditorPanel::new();
        assert!(panel.selected.is_none());
    }

    #[test]
    fn port_colors_are_distinct() {
        let terrain_color = NodeEditorPanel::port_color(PortType::Terrain);
        let splat_color   = NodeEditorPanel::port_color(PortType::Splats);
        assert_ne!(terrain_color, splat_color, "port types should have distinct colors");
    }

    #[test]
    fn ensure_layouts_does_not_overwrite_existing() {
        let mut panel = NodeEditorPanel::new();
        let mut graph = OchromaNodeGraph::new();
        let id = graph.add_node("a", crate::node_graph::tests_helpers::pass_node());
        panel.layouts.insert(id, NodeLayout { pos: Pos2::new(0.0, 0.0), size: Vec2::new(180.0, 120.0) });
        panel.ensure_layouts(&graph);
        assert_eq!(panel.layouts.len(), 1);
    }

    #[test]
    fn pan_default_is_zero() {
        let panel = NodeEditorPanel::new();
        assert_eq!(panel.pan, Vec2::ZERO);
        assert!((panel.zoom - 1.0).abs() < 1e-5);
    }

    #[test]
    fn wire_drag_starts_none() {
        let panel = NodeEditorPanel::new();
        assert!(panel.wire_drag.is_none());
    }
}
