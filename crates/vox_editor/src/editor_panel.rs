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

/// How close (in screen pixels) the pointer must be to a port circle to count
/// as hovering it for the purposes of starting / completing a wire drag.
pub const PORT_HIT_RADIUS: f32 = 9.0;

impl NodeEditorPanel {
    pub fn new() -> Self { Self::default() }

    /// World-space screen position of a node's single input ("in") port,
    /// accounting for the current pan. Returns `None` if the node has no layout.
    pub fn input_port_pos(&self, node: NodeId) -> Option<Pos2> {
        let layout = self.layouts.get(&node)?;
        let top_left = layout.pos + self.pan;
        Some(top_left + Vec2::new(0.0, layout.size.y * 0.5))
    }

    /// Screen position of a node's single output ("out") port (right edge),
    /// accounting for the current pan. Returns `None` if the node has no layout.
    pub fn output_port_pos(&self, node: NodeId) -> Option<Pos2> {
        let layout = self.layouts.get(&node)?;
        let top_left = layout.pos + self.pan;
        Some(top_left + Vec2::new(layout.size.x, layout.size.y * 0.5))
    }

    /// Find the node whose output port is within [`PORT_HIT_RADIUS`] of `pos`.
    pub fn output_port_at(&self, pos: Pos2) -> Option<NodeId> {
        self.layouts.keys().copied().find(|&id| {
            self.output_port_pos(id).is_some_and(|p| p.distance(pos) <= PORT_HIT_RADIUS)
        })
    }

    /// Find the node whose input port is within [`PORT_HIT_RADIUS`] of `pos`.
    pub fn input_port_at(&self, pos: Pos2) -> Option<NodeId> {
        self.layouts.keys().copied().find(|&id| {
            self.input_port_pos(id).is_some_and(|p| p.distance(pos) <= PORT_HIT_RADIUS)
        })
    }

    /// Begin dragging a wire out of node `from`'s output port.
    pub fn begin_wire_drag(&mut self, from: NodeId, start: Pos2) {
        self.wire_drag = Some(WireDrag {
            from_node: from,
            from_port: "out".into(),
            current:   start,
        });
    }

    /// Update the loose end of an in-progress wire drag.
    pub fn update_wire_drag(&mut self, pos: Pos2) {
        if let Some(drag) = self.wire_drag.as_mut() {
            drag.current = pos;
        }
    }

    /// Finish a wire drag at `release_pos`. If an input port lies under the
    /// release point and the connection is valid, a real edge is added to
    /// `graph` (output "out" -> input "in"). Clears the in-progress drag either way.
    ///
    /// Returns `true` iff a new edge was successfully created.
    pub fn complete_wire_drag(&mut self, graph: &mut OchromaNodeGraph, release_pos: Pos2) -> bool {
        let Some(drag) = self.wire_drag.take() else { return false };
        let Some(target) = self.input_port_at(release_pos) else { return false };
        // No self-loops: dragging a port back onto its own node does nothing.
        if target == drag.from_node { return false; }
        graph.connect(drag.from_node, &drag.from_port, target, "in").is_ok()
    }

    pub fn ensure_layouts(&mut self, graph: &OchromaNodeGraph) {
        // Remove stale layouts for nodes that no longer exist.
        let live_ids: std::collections::HashSet<NodeId> = graph.node_ids().collect();
        self.layouts.retain(|id, _| live_ids.contains(id));
        // Assign grid positions to newly added nodes.
        let start_idx = self.layouts.len();
        let mut ids: Vec<NodeId> = graph.node_ids()
            .filter(|id| !self.layouts.contains_key(id))
            .collect();
        ids.sort();
        for (idx, id) in (start_idx..).zip(ids) {
            let col = idx % 4;
            let row = idx / 4;
            self.layouts.insert(id, NodeLayout {
                pos:  Pos2::new(20.0 + col as f32 * 220.0, 20.0 + row as f32 * 160.0),
                size: Vec2::new(180.0, 120.0),
            });
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

        // --- Wire-drag lifecycle (takes priority over panning) -------------
        // A drag that starts on an output port draws a wire instead of panning.
        let pointer_pos = response.interact_pointer_pos();
        if response.drag_started() {
            if let Some(pos) = pointer_pos {
                if let Some(src) = self.output_port_at(pos) {
                    self.begin_wire_drag(src, pos);
                }
            }
        }
        let dragging_wire = self.wire_drag.is_some();
        if dragging_wire {
            if let Some(pos) = pointer_pos {
                self.update_wire_drag(pos);
            }
            if response.drag_stopped() {
                let release = pointer_pos.unwrap_or_else(|| {
                    self.wire_drag.as_ref().map(|d| d.current).unwrap_or(canvas_rect.center())
                });
                self.complete_wire_drag(graph, release);
            }
        } else if response.dragged() && !ui.input(|i| i.pointer.secondary_down()) {
            self.pan += response.drag_delta();
        }

        let painter = ui.painter_at(canvas_rect);

        // --- Draw existing edges as wires ----------------------------------
        for (from, _fp, to, _tp) in graph.edges() {
            let (Some(a), Some(b)) = (self.output_port_pos(from), self.input_port_pos(to)) else { continue };
            painter.line_segment([a, b], Stroke::new(2.0, Color32::from_rgb(180, 180, 200)));
        }
        // --- Draw the in-progress wire -------------------------------------
        if let Some(drag) = &self.wire_drag {
            if let Some(a) = self.output_port_pos(drag.from_node) {
                painter.line_segment([a, drag.current], Stroke::new(2.0, Color32::from_rgb(120, 220, 160)));
            }
        }

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
            painter.rect_filled(header_rect, egui::CornerRadius { nw: 6, ne: 6, sw: 0, se: 0 }, Color32::from_rgb(60, 80, 120));
            let out_pos = top_left + Vec2::new(layout.size.x, layout.size.y * 0.5);
            let in_pos  = top_left + Vec2::new(0.0, layout.size.y * 0.5);
            port_positions.insert((id.0, "out".into()), out_pos);
            port_positions.insert((id.0, "in".into()),  in_pos);
            painter.circle_filled(out_pos, 5.0, Color32::from_rgb(80, 200, 120));
            painter.circle_filled(in_pos,  5.0, Color32::from_rgb(200, 120, 80));
        }
        // Handle click selections (separate pass to avoid borrow conflict).
        // Skip while a wire is being dragged so wiring doesn't also re-select.
        if !dragging_wire {
            if let Some(click_pos) = response.interact_pointer_pos() {
                for id in &node_ids {
                    if let Some(layout) = self.layouts.get(id) {
                        let top_left = layout.pos + self.pan;
                        let rect = Rect::from_min_size(top_left, layout.size);
                        if rect.contains(click_pos) { self.selected = Some(*id); break; }
                    }
                }
            }
        }
        let _ = port_positions;
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

    #[test]
    fn wire_drag_from_out_to_in_creates_one_edge() {
        let mut panel = NodeEditorPanel::new();
        let mut graph = OchromaNodeGraph::new();
        let a = graph.add_node("a", crate::node_graph::tests_helpers::pass_node());
        let b = graph.add_node("b", crate::node_graph::tests_helpers::pass_node());

        // Place the two nodes at known positions so port coordinates are deterministic.
        panel.layouts.insert(a, NodeLayout { pos: Pos2::new(0.0,   0.0), size: Vec2::new(180.0, 120.0) });
        panel.layouts.insert(b, NodeLayout { pos: Pos2::new(400.0, 0.0), size: Vec2::new(180.0, 120.0) });

        let a_out = panel.output_port_pos(a).unwrap();
        let b_in  = panel.input_port_pos(b).unwrap();
        // A's output is on its right edge; B's input on its left edge — they differ.
        assert_eq!(a_out, Pos2::new(180.0, 60.0));
        assert_eq!(b_in,  Pos2::new(400.0, 60.0));

        assert_eq!(graph.edge_count(), 0, "no edges before drag");

        // Drag a wire from A's output port and release exactly on B's input port.
        panel.begin_wire_drag(a, a_out);
        assert!(panel.wire_drag.is_some(), "drag should be in progress");
        let created = panel.complete_wire_drag(&mut graph, b_in);

        assert!(created, "releasing on B's input port must create an edge");
        assert!(panel.wire_drag.is_none(), "drag must be cleared after completion");
        assert_eq!(graph.edge_count(), 1, "exactly one edge must exist");

        let edges: Vec<(u32, &str, u32, &str)> = graph.edges()
            .map(|(f, fp, t, tp)| (f.0, fp, t.0, tp))
            .collect();
        assert_eq!(edges, vec![(a.0, "out", b.0, "in")], "edge must be A.out -> B.in");
    }

    #[test]
    fn wire_drag_release_in_empty_space_creates_no_edge() {
        let mut panel = NodeEditorPanel::new();
        let mut graph = OchromaNodeGraph::new();
        let a = graph.add_node("a", crate::node_graph::tests_helpers::pass_node());
        graph.add_node("b", crate::node_graph::tests_helpers::pass_node());
        panel.layouts.insert(a, NodeLayout { pos: Pos2::new(0.0, 0.0), size: Vec2::new(180.0, 120.0) });

        let a_out = panel.output_port_pos(a).unwrap();
        panel.begin_wire_drag(a, a_out);
        // Release far from any input port.
        let created = panel.complete_wire_drag(&mut graph, Pos2::new(5000.0, 5000.0));
        assert!(!created, "releasing in empty space must not create an edge");
        assert_eq!(graph.edge_count(), 0);
        assert!(panel.wire_drag.is_none(), "drag must still be cleared");
    }
}
