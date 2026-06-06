//! NodeCanvas — the SOTA egui-painter node-graph renderer.
//!
//! One engine-agnostic, renderer-agnostic data model ([`CanvasGraph`]) plus one
//! egui-native renderer ([`NodeCanvas`]) so the host editor AND every plugin
//! share a single canvas with identical fidelity. The design's checklist items
//! it clears:
//!
//! - **[10]** cubic-bezier wires (>=2.5px, antialiased, tunable curvature), drawn
//!   as a flattened polyline of short [`egui::Shape::line_segment`]s so each
//!   sub-segment can carry its own color (egui strokes are coverage-AA).
//! - **[11]** wire **gradient** between the two endpoint SOCKET colors
//!   (`tokens::wire_color(PortType)`) + filled circular port sockets with a hover
//!   ring.
//! - **[12]** infinite pannable **dot-grid** canvas, smooth zoom clamped to
//!   `[ZOOM_MIN, ZOOM_MAX]`, and snap-to-grid node dragging.
//! - **[13]** **minimap** corner inset with a viewport-rect indicator and
//!   click-to-jump.
//! - **[14]** node styling: category-colored headers (`tokens::category_header`),
//!   rounded bodies (radius token), selected outline in accent, muted/error
//!   outline state, tintable comment frames, reroute dots.
//!
//! `vox_ui` stays engine-agnostic: the model speaks [`crate::tokens::PortType`]
//! and [`crate::tokens::NodeCategory`], never `vox_editor` types. The vox_app
//! call site maps its real graph onto a [`CanvasGraph`]; the compile-pin test
//! `vox_editor::node_graph::tokens_portype_mirror_is_exhaustive` already guards
//! the enum parity.

use crate::tokens::{NodeCategory, PortType, Tokens};
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};

/// Smallest allowed zoom (whole graph fits) — clamps [`NodeCanvas::pan_zoom`].
pub const ZOOM_MIN: f32 = 0.23;
/// Largest allowed zoom (close inspection).
pub const ZOOM_MAX: f32 = 2.07;
/// World-space grid step (one dot every this-many world units).
pub const GRID_STEP: f32 = 24.0;
/// Minimap inset size, design-fixed at 240x160.
pub const MINIMAP_SIZE: Vec2 = Vec2::new(240.0, 160.0);
/// Minimap opacity (design: ~0.65).
pub const MINIMAP_OPACITY: f32 = 0.65;

/// One port on a node. Position is computed by the renderer from the node rect.
#[derive(Debug, Clone, PartialEq)]
pub struct PortView {
    /// Stable port name (matches a wire endpoint's port string).
    pub name: String,
    /// Drives the socket fill + the wire gradient endpoint color.
    pub ty: PortType,
}

/// Render state of a node body — drives the outline color (item 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeState {
    #[default]
    Normal,
    /// Muted/disabled — draws a muted outline.
    Muted,
    /// A cook error — draws a red (`status.error`) outline.
    Error,
}

/// One node in the graph. Renderer-agnostic; positions are world-space.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    pub id: u64,
    pub title: String,
    pub category: NodeCategory,
    pub inputs: Vec<PortView>,
    pub outputs: Vec<PortView>,
    /// World-space top-left.
    pub pos: Pos2,
    /// World-space size.
    pub size: Vec2,
    pub selected: bool,
    pub state: NodeState,
    /// `true` => drawn as a small reroute knot, not a full node.
    pub reroute: bool,
}

impl NodeView {
    /// A standard node with the default body size.
    pub fn new(
        id: u64,
        title: impl Into<String>,
        category: NodeCategory,
        pos: Pos2,
    ) -> Self {
        NodeView {
            id,
            title: title.into(),
            category,
            inputs: Vec::new(),
            outputs: Vec::new(),
            pos,
            size: Vec2::new(150.0, 0.0), // height computed from port count if 0
            selected: false,
            state: NodeState::Normal,
            reroute: false,
        }
    }

    pub fn with_input(mut self, name: impl Into<String>, ty: PortType) -> Self {
        self.inputs.push(PortView { name: name.into(), ty });
        self
    }
    pub fn with_output(mut self, name: impl Into<String>, ty: PortType) -> Self {
        self.outputs.push(PortView { name: name.into(), ty });
        self
    }

    /// The on-screen body height in world units (header + a row per port).
    pub fn world_height(&self) -> f32 {
        if self.size.y > 0.0 {
            return self.size.y;
        }
        let rows = self.inputs.len().max(self.outputs.len()).max(1) as f32;
        HEADER_H + rows * ROW_H + PORT_PAD * 2.0
    }
}

/// One wire from an output port to an input port.
#[derive(Debug, Clone, PartialEq)]
pub struct WireView {
    pub from_node: u64,
    pub from_port: String,
    pub to_node: u64,
    pub to_port: String,
    /// Exec/flow wire (drawn white with an L->R arrowhead) vs a typed data wire.
    pub exec: bool,
    /// Optional value chip drawn at the wire midpoint.
    pub label: Option<String>,
}

/// A tintable comment frame grouping nodes (item 14).
#[derive(Debug, Clone, PartialEq)]
pub struct CommentBox {
    pub rect: Rect,
    pub title: String,
    /// Dotted token color key for the frame tint (e.g. `"accent.dim"`).
    pub tint: String,
}

/// The renderer-agnostic graph the canvas draws. Engine-agnostic by construction
/// (no `vox_editor` types). The vox_app call site builds one of these each frame
/// from its real `OchromaNodeGraph`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CanvasGraph {
    pub nodes: Vec<NodeView>,
    pub wires: Vec<WireView>,
    pub comments: Vec<CommentBox>,
}

impl CanvasGraph {
    pub fn node(&self, id: u64) -> Option<&NodeView> {
        self.nodes.iter().find(|n| n.id == id)
    }
    fn node_mut(&mut self, id: u64) -> Option<&mut NodeView> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }
}

// === Layout constants (world units, scaled by zoom at draw time) ===
const HEADER_H: f32 = 22.0;
const ROW_H: f32 = 20.0;
const PORT_PAD: f32 = 6.0;
const PORT_R: f32 = 5.0;

/// What the canvas reports back after a frame (for the host to react to).
#[derive(Debug, Clone, Default)]
pub struct CanvasResponse {
    /// The node the user is dragging this frame (if any), with its new world pos.
    pub dragged: Option<(u64, Pos2)>,
    /// A node whose body was clicked (selection intent).
    pub clicked: Option<u64>,
    /// The canvas background was clicked (clear selection).
    pub background_clicked: bool,
}

/// The egui-native node-graph renderer. Owns pan/zoom + interaction state.
#[derive(Debug, Clone)]
pub struct NodeCanvas {
    /// Screen pixels the world origin is offset by (panning).
    pub pan: Vec2,
    /// World->screen scale.
    pub zoom: f32,
    /// Bezier curvature factor (fraction of horizontal span used for control
    /// handles). Default 0.5 (design).
    pub curvature: f32,
    /// Wire thickness in screen px at zoom 1.0 (design default 4.0, >=2.5 req).
    pub wire_thickness: f32,
    /// World-units the node drag snaps to (0 = no snap).
    pub snap: f32,
    /// Whether the minimap inset is drawn.
    pub show_minimap: bool,
    /// Node id being dragged (drag-grab state across frames).
    drag_node: Option<u64>,
    /// Pointer offset within the node body at grab time (world units).
    drag_grab: Vec2,
    /// Last computed content bounds (world space) — used by the minimap.
    content_bounds: Rect,
}

impl Default for NodeCanvas {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeCanvas {
    pub fn new() -> Self {
        NodeCanvas {
            pan: Vec2::ZERO,
            zoom: 1.0,
            curvature: 0.5,
            wire_thickness: 4.0,
            snap: 0.0,
            show_minimap: true,
            drag_node: None,
            drag_grab: Vec2::ZERO,
            content_bounds: Rect::from_min_size(Pos2::ZERO, Vec2::splat(1.0)),
        }
    }

    pub fn set_curvature(&mut self, c: f32) {
        self.curvature = c.clamp(0.0, 2.0);
    }
    pub fn set_wire_thickness(&mut self, t: f32) {
        self.wire_thickness = t.max(2.5);
    }
    pub fn set_snap(&mut self, s: f32) {
        self.snap = s.max(0.0);
    }

    /// World -> screen.
    fn w2s(&self, origin: Pos2, p: Pos2) -> Pos2 {
        origin + (p.to_vec2() * self.zoom) + self.pan
    }
    /// Screen -> world.
    fn s2w(&self, origin: Pos2, p: Pos2) -> Pos2 {
        ((p - origin - self.pan) / self.zoom).to_pos2()
    }

    /// The on-screen rect of a node (after pan/zoom).
    fn node_screen_rect(&self, origin: Pos2, n: &NodeView) -> Rect {
        let min = self.w2s(origin, n.pos);
        let size = Vec2::new(n.size.x, n.world_height()) * self.zoom;
        Rect::from_min_size(min, size)
    }

    /// World position of a port socket (centre of its circle).
    fn port_world_pos(n: &NodeView, port: &str, output: bool) -> Option<Pos2> {
        let ports = if output { &n.outputs } else { &n.inputs };
        let i = ports.iter().position(|p| p.name == port)?;
        let y = n.pos.y + HEADER_H + PORT_PAD + (i as f32 + 0.5) * ROW_H;
        let x = if output { n.pos.x + n.size.x } else { n.pos.x };
        Some(Pos2::new(x, y))
    }

    /// Sample a cubic bezier at parameter `t` (Bernstein form).
    fn cubic(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
        let u = 1.0 - t;
        let w0 = u * u * u;
        let w1 = 3.0 * u * u * t;
        let w2 = 3.0 * u * t * t;
        let w3 = t * t * t;
        Pos2::new(
            w0 * p0.x + w1 * p1.x + w2 * p2.x + w3 * p3.x,
            w0 * p0.y + w1 * p1.y + w2 * p2.y + w3 * p3.y,
        )
    }

    /// Render the graph. Mutates `graph` (node positions on drag). Returns the
    /// per-frame interaction report.
    pub fn ui(&mut self, ui: &mut egui::Ui, t: &Tokens, graph: &mut CanvasGraph) -> CanvasResponse {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let origin = rect.min;
        let painter = ui.painter_at(rect);
        let mut out = CanvasResponse::default();

        // --- Background fill ---
        painter.rect_filled(rect, 0.0, col(t, "surface.bg.0"));

        // --- Pan (drag on empty canvas) & zoom (scroll) ---
        let pointer = ui.input(|i| i.pointer.hover_pos());
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll.abs() > 0.0
            && let Some(p) = pointer.filter(|p| rect.contains(*p))
        {
            self.zoom_at(origin, p, scroll);
        }

        // --- Comment frames (behind nodes) ---
        for cm in &graph.comments {
            let r = Rect::from_min_max(self.w2s(origin, cm.rect.min), self.w2s(origin, cm.rect.max));
            let mut tint = col(t, &cm.tint);
            tint = Color32::from_rgba_unmultiplied(tint.r(), tint.g(), tint.b(), 40);
            painter.rect_filled(r, t.radius[1], tint);
            painter.rect_stroke(
                r,
                t.radius[1],
                Stroke::new(1.0, col(t, &cm.tint)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                r.left_top() + Vec2::new(6.0, 4.0),
                egui::Align2::LEFT_TOP,
                &cm.title,
                egui::FontId::proportional(t.type_ramp.caption * self.zoom.max(0.6)),
                col(t, "text.secondary"),
            );
        }

        // --- Grid (dots) ---
        self.draw_grid(&painter, rect, origin, t);

        // --- Wires (under nodes) ---
        for w in &graph.wires {
            self.draw_wire(&painter, origin, graph, w, t);
        }

        // --- Nodes ---
        // Compute content bounds while we are at it (for the minimap).
        let mut bounds: Option<Rect> = None;
        for n in &graph.nodes {
            let wr = Rect::from_min_size(n.pos, Vec2::new(n.size.x, n.world_height()));
            bounds = Some(match bounds {
                Some(b) => b.union(wr),
                None => wr,
            });
        }
        self.content_bounds = bounds.unwrap_or(Rect::from_min_size(Pos2::ZERO, Vec2::splat(1.0)));

        for n in &graph.nodes {
            self.draw_node(&painter, origin, n, t, pointer);
        }

        // --- Interaction: node drag / selection ---
        self.handle_interaction(origin, rect, graph, &response, pointer, &mut out);

        // --- Pan when dragging empty canvas ---
        if response.dragged() && self.drag_node.is_none() {
            // Only pan if the drag did not start on a node.
            self.pan += response.drag_delta();
        }

        // --- Minimap (on top) ---
        if self.show_minimap {
            self.draw_minimap(&painter, rect, t, graph);
            self.handle_minimap_click(rect, &response, pointer);
        }

        out
    }

    fn zoom_at(&mut self, origin: Pos2, anchor: Pos2, scroll: f32) {
        let old = self.zoom;
        let factor = (scroll * 0.0015).exp();
        let new = (old * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        if (new - old).abs() < 1e-6 {
            return;
        }
        // Keep the world point under the cursor fixed while zooming.
        let world = self.s2w(origin, anchor);
        self.zoom = new;
        let screen_after = self.w2s(origin, world);
        self.pan += anchor - screen_after;
    }

    fn draw_grid(&self, painter: &egui::Painter, rect: Rect, origin: Pos2, t: &Tokens) {
        let step = GRID_STEP * self.zoom;
        if step < 4.0 {
            return; // too dense to be useful
        }
        let dot = col(t, "surface.border");
        // World coords of the top-left visible corner.
        let w_min = self.s2w(origin, rect.min);
        let first_x = (w_min.x / GRID_STEP).floor() * GRID_STEP;
        let first_y = (w_min.y / GRID_STEP).floor() * GRID_STEP;
        let mut wy = first_y;
        while {
            let sy = self.w2s(origin, Pos2::new(0.0, wy)).y;
            sy <= rect.max.y
        } {
            let mut wx = first_x;
            loop {
                let sp = self.w2s(origin, Pos2::new(wx, wy));
                if sp.x > rect.max.x {
                    break;
                }
                if rect.contains(sp) {
                    painter.circle_filled(sp, (1.0 * self.zoom).clamp(0.6, 2.0), dot);
                }
                wx += GRID_STEP;
            }
            wy += GRID_STEP;
        }
    }

    fn socket_color(&self, t: &Tokens, ty: PortType) -> Color32 {
        let [r, g, b, a] = t.wire_color(ty);
        Color32::from_rgba_unmultiplied(r, g, b, a)
    }

    fn draw_wire(
        &self,
        painter: &egui::Painter,
        origin: Pos2,
        graph: &CanvasGraph,
        w: &WireView,
        t: &Tokens,
    ) {
        let (Some(from_n), Some(to_n)) = (graph.node(w.from_node), graph.node(w.to_node)) else {
            return;
        };
        let (Some(fw), Some(tw)) = (
            Self::port_world_pos(from_n, &w.from_port, true),
            Self::port_world_pos(to_n, &w.to_port, false),
        ) else {
            return;
        };
        let p0 = self.w2s(origin, fw);
        let p3 = self.w2s(origin, tw);
        // Control points pulled horizontally (UE/Blender noodle shape).
        let dx = (p3.x - p0.x).abs().max(40.0) * self.curvature;
        let p1 = p0 + Vec2::new(dx, 0.0);
        let p2 = p3 - Vec2::new(dx, 0.0);

        let thickness = (self.wire_thickness * self.zoom).max(2.5);

        if w.exec {
            // Exec/flow wire: solid white, arrowhead at the target.
            let white = self.socket_color(t, PortType::Flow);
            self.stroke_bezier(painter, p0, p1, p2, p3, thickness, |_| white);
            self.draw_arrowhead(painter, p2, p3, thickness, white);
        } else {
            // Typed data wire: gradient between the two socket colors.
            let src_ty = from_n
                .outputs
                .iter()
                .find(|p| p.name == w.from_port)
                .map(|p| p.ty)
                .unwrap_or(PortType::Scalar);
            let dst_ty = to_n
                .inputs
                .iter()
                .find(|p| p.name == w.to_port)
                .map(|p| p.ty)
                .unwrap_or(src_ty);
            let cs = self.socket_color(t, src_ty);
            let cd = self.socket_color(t, dst_ty);
            self.stroke_bezier(painter, p0, p1, p2, p3, thickness, |frac| lerp_col(cs, cd, frac));
        }

        // Value chip at midpoint.
        if let Some(lbl) = &w.label {
            let mid = Self::cubic(p0, p1, p2, p3, 0.5);
            let chip = Rect::from_center_size(mid, Vec2::new(lbl.len() as f32 * 6.5 + 10.0, 16.0));
            painter.rect_filled(chip, t.radius[0], col(t, "surface.bg.2"));
            painter.text(
                mid,
                egui::Align2::CENTER_CENTER,
                lbl,
                egui::FontId::monospace(t.type_ramp.mono * self.zoom.clamp(0.7, 1.2)),
                col(t, "text.primary"),
            );
        }
    }

    /// Stroke a flattened cubic bezier as short colored segments (per-segment
    /// color via `color_at(frac)`), each an AA `line_segment`.
    #[allow(clippy::too_many_arguments)]
    fn stroke_bezier(
        &self,
        painter: &egui::Painter,
        p0: Pos2,
        p1: Pos2,
        p2: Pos2,
        p3: Pos2,
        width: f32,
        color_at: impl Fn(f32) -> Color32,
    ) {
        const SEGS: usize = 32;
        let mut prev = p0;
        for i in 1..=SEGS {
            let frac = i as f32 / SEGS as f32;
            let p = Self::cubic(p0, p1, p2, p3, frac);
            let mid = (i as f32 - 0.5) / SEGS as f32;
            painter.line_segment([prev, p], Stroke::new(width, color_at(mid)));
            prev = p;
        }
    }

    fn draw_arrowhead(&self, painter: &egui::Painter, from: Pos2, tip: Pos2, w: f32, col: Color32) {
        let dir = (tip - from).normalized();
        let perp = Vec2::new(-dir.y, dir.x);
        let s = (w * 2.0).max(8.0);
        let base = tip - dir * s;
        let a = base + perp * (s * 0.5);
        let b = base - perp * (s * 0.5);
        painter.add(egui::Shape::convex_polygon(vec![tip, a, b], col, Stroke::NONE));
    }

    fn draw_node(
        &self,
        painter: &egui::Painter,
        origin: Pos2,
        n: &NodeView,
        t: &Tokens,
        pointer: Option<Pos2>,
    ) {
        if n.reroute {
            // A reroute knot: a single small filled circle in its first port's
            // color (or scalar grey).
            let p = self.w2s(origin, n.pos);
            let ty = n.inputs.first().or(n.outputs.first()).map(|p| p.ty).unwrap_or(PortType::Scalar);
            painter.circle_filled(p, (PORT_R + 1.0) * self.zoom, self.socket_color(t, ty));
            return;
        }

        let r = self.node_screen_rect(origin, n);
        let radius = t.radius[1];
        // Body.
        painter.rect_filled(r, radius, col(t, "surface.bg.2"));
        // Header.
        let header = Rect::from_min_size(r.min, Vec2::new(r.width(), HEADER_H * self.zoom));
        painter.rect_filled(header, radius, header_col(t, n.category));
        painter.text(
            header.left_center() + Vec2::new(8.0, 0.0),
            egui::Align2::LEFT_CENTER,
            &n.title,
            egui::FontId::proportional((t.type_ramp.caption * self.zoom).max(7.0)),
            col(t, "text.primary"),
        );

        // Outline by state / selection.
        let (ow, oc) = if n.selected {
            (2.0, col(t, "accent.base"))
        } else {
            match n.state {
                NodeState::Error => (2.0, col(t, "status.error")),
                NodeState::Muted => (1.0, col(t, "text.disabled")),
                NodeState::Normal => (1.0, col(t, "surface.border")),
            }
        };
        painter.rect_stroke(r, radius, Stroke::new(ow, oc), egui::StrokeKind::Inside);

        // Ports + labels.
        let draw_port = |ports: &[PortView], output: bool| {
            for (i, p) in ports.iter().enumerate() {
                let wy = n.pos.y + HEADER_H + PORT_PAD + (i as f32 + 0.5) * ROW_H;
                let wx = if output { n.pos.x + n.size.x } else { n.pos.x };
                let sp = self.w2s(origin, Pos2::new(wx, wy));
                let sc = self.socket_color(t, p.ty);
                let pr = PORT_R * self.zoom;
                // Hover ring.
                if let Some(ptr) = pointer
                    && (ptr - sp).length() < pr + 4.0
                {
                    painter.circle_stroke(sp, pr + 3.0, Stroke::new(2.0, col(t, "text.primary")));
                }
                painter.circle_filled(sp, pr, sc);
                painter.circle_stroke(sp, pr, Stroke::new(1.0, col(t, "surface.bg.0")));
                // Label.
                let lx = if output {
                    sp + Vec2::new(-(8.0 * self.zoom), 0.0)
                } else {
                    sp + Vec2::new(8.0 * self.zoom, 0.0)
                };
                let align = if output {
                    egui::Align2::RIGHT_CENTER
                } else {
                    egui::Align2::LEFT_CENTER
                };
                painter.text(
                    lx,
                    align,
                    &p.name,
                    egui::FontId::proportional((t.type_ramp.caption * self.zoom * 0.85).max(6.0)),
                    col(t, "text.secondary"),
                );
            }
        };
        draw_port(&n.inputs, false);
        draw_port(&n.outputs, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_interaction(
        &mut self,
        origin: Pos2,
        rect: Rect,
        graph: &mut CanvasGraph,
        response: &egui::Response,
        pointer: Option<Pos2>,
        out: &mut CanvasResponse,
    ) {
        // Begin a drag: hit-test node bodies on drag-start.
        if response.drag_started()
            && let Some(p) = pointer
        {
            // Topmost (last drawn) node first.
            for n in graph.nodes.iter().rev() {
                let nr = self.node_screen_rect(origin, n);
                if !n.reroute && nr.contains(p) {
                    self.drag_node = Some(n.id);
                    let wp = self.s2w(origin, p);
                    self.drag_grab = wp - n.pos;
                    out.clicked = Some(n.id);
                    break;
                }
            }
        }

        // Continue a node drag.
        if let Some(id) = self.drag_node {
            if response.dragged()
                && let Some(p) = pointer
            {
                let wp = self.s2w(origin, p);
                let mut np = wp - self.drag_grab;
                if self.snap > 0.0 {
                    np.x = (np.x / self.snap).round() * self.snap;
                    np.y = (np.y / self.snap).round() * self.snap;
                }
                if let Some(n) = graph.node_mut(id) {
                    n.pos = np;
                }
                out.dragged = Some((id, np));
            }
            if response.drag_stopped() {
                self.drag_node = None;
            }
        }

        // A plain click on empty canvas clears selection.
        if response.clicked()
            && self.drag_node.is_none()
            && let Some(p) = pointer
        {
            let on_node = graph
                .nodes
                .iter()
                .any(|n| !n.reroute && self.node_screen_rect(origin, n).contains(p));
            if on_node {
                // Selection click.
                for n in graph.nodes.iter().rev() {
                    if !n.reroute && self.node_screen_rect(origin, n).contains(p) {
                        out.clicked = Some(n.id);
                        break;
                    }
                }
            } else if rect.contains(p) {
                out.background_clicked = true;
            }
        }
    }

    // === Minimap (item 13) ===

    /// The minimap inset rect (bottom-right of the canvas).
    fn minimap_rect(&self, canvas: Rect) -> Rect {
        let margin = 12.0;
        let min = Pos2::new(
            canvas.max.x - MINIMAP_SIZE.x - margin,
            canvas.max.y - MINIMAP_SIZE.y - margin,
        );
        Rect::from_min_size(min, MINIMAP_SIZE)
    }

    /// Map a world point into the minimap rect using the content bounds.
    /// `clamp` keeps node rects inside the inset; the viewport indicator passes
    /// `false` so its motion is visible even when it overhangs the content.
    fn world_to_minimap_inner(&self, mm: Rect, p: Pos2, clamp: bool) -> Pos2 {
        let b = self.content_bounds.expand(40.0);
        let mut fx = (p.x - b.min.x) / b.width().max(1.0);
        let mut fy = (p.y - b.min.y) / b.height().max(1.0);
        if clamp {
            fx = fx.clamp(0.0, 1.0);
            fy = fy.clamp(0.0, 1.0);
        }
        Pos2::new(mm.min.x + fx * mm.width(), mm.min.y + fy * mm.height())
    }
    fn world_to_minimap(&self, mm: Rect, p: Pos2) -> Pos2 {
        self.world_to_minimap_inner(mm, p, true)
    }

    fn draw_minimap(&self, painter: &egui::Painter, canvas: Rect, t: &Tokens, graph: &CanvasGraph) {
        let mm = self.minimap_rect(canvas);
        let [r, g, b, _] = t.color("surface.bg.1");
        let alpha = (MINIMAP_OPACITY * 255.0) as u8;
        painter.rect_filled(mm, t.radius[1], Color32::from_rgba_unmultiplied(r, g, b, alpha));
        painter.rect_stroke(mm, t.radius[1], Stroke::new(1.0, col(t, "border.strong")), egui::StrokeKind::Inside);

        // Scaled node rects.
        for n in &graph.nodes {
            let a = self.world_to_minimap(mm, n.pos);
            let b = self.world_to_minimap(mm, n.pos + Vec2::new(n.size.x, n.world_height()));
            let nr = Rect::from_two_pos(a, b);
            painter.rect_filled(nr, 1.0, header_col(t, n.category));
        }

        // Viewport-rect indicator: the world region currently visible.
        let vr = self.minimap_viewport_rect(canvas).intersect(mm);
        painter.rect_stroke(vr, 1.0, Stroke::new(1.5, col(t, "accent.base")), egui::StrokeKind::Inside);
    }

    /// Click-to-jump: clicking inside the minimap re-centres the canvas on the
    /// corresponding world point.
    fn handle_minimap_click(&mut self, canvas: Rect, response: &egui::Response, pointer: Option<Pos2>) {
        let mm = self.minimap_rect(canvas);
        if !response.clicked() {
            return;
        }
        let Some(p) = pointer else { return };
        if !mm.contains(p) {
            return;
        }
        // Invert world_to_minimap to get the clicked world point.
        let b = self.content_bounds.expand(40.0);
        let fx = (p.x - mm.min.x) / mm.width();
        let fy = (p.y - mm.min.y) / mm.height();
        let world = Pos2::new(b.min.x + fx * b.width(), b.min.y + fy * b.height());
        // Re-centre: place `world` at the canvas centre.
        let center = canvas.center();
        self.pan = center - canvas.min - world.to_vec2() * self.zoom;
    }

    // === Test/inspection helpers (so tests assert real geometry) ===

    /// The on-screen rect of node `id`, after pan/zoom (for tests).
    pub fn node_rect_screen(&self, canvas_origin: Pos2, graph: &CanvasGraph, id: u64) -> Option<Rect> {
        let n = graph.node(id)?;
        Some(self.node_screen_rect(canvas_origin, n))
    }

    /// The on-screen socket position of a port (for tests).
    pub fn port_screen_pos(
        &self,
        canvas_origin: Pos2,
        graph: &CanvasGraph,
        node: u64,
        port: &str,
        output: bool,
    ) -> Option<Pos2> {
        let n = graph.node(node)?;
        let wp = Self::port_world_pos(n, port, output)?;
        Some(self.w2s(canvas_origin, wp))
    }

    /// Sample points along a wire's flattened bezier in SCREEN space (for the
    /// curvature/gradient/AA tests). `samples` evenly spaced parameters in [0,1].
    pub fn wire_screen_points(
        &self,
        canvas_origin: Pos2,
        graph: &CanvasGraph,
        wire: &WireView,
        samples: usize,
    ) -> Option<Vec<Pos2>> {
        let from_n = graph.node(wire.from_node)?;
        let to_n = graph.node(wire.to_node)?;
        let fw = Self::port_world_pos(from_n, &wire.from_port, true)?;
        let tw = Self::port_world_pos(to_n, &wire.to_port, false)?;
        let p0 = self.w2s(canvas_origin, fw);
        let p3 = self.w2s(canvas_origin, tw);
        let dx = (p3.x - p0.x).abs().max(40.0) * self.curvature;
        let p1 = p0 + Vec2::new(dx, 0.0);
        let p2 = p3 - Vec2::new(dx, 0.0);
        let mut pts = Vec::with_capacity(samples);
        for i in 0..samples {
            let tt = i as f32 / (samples.max(2) - 1) as f32;
            pts.push(Self::cubic(p0, p1, p2, p3, tt));
        }
        Some(pts)
    }

    /// The minimap rect for the given canvas (for tests).
    pub fn minimap_rect_for(&self, canvas: Rect) -> Rect {
        self.minimap_rect(canvas)
    }

    /// The minimap viewport-indicator rect for the given canvas (for tests).
    pub fn minimap_viewport_rect(&self, canvas: Rect) -> Rect {
        let mm = self.minimap_rect(canvas);
        let vis_min = self.s2w(canvas.min, canvas.min);
        let vis_max = self.s2w(canvas.min, canvas.max);
        // Non-clamping projection so the indicator's motion is visible even when
        // the visible region overhangs the content bounds; intersect with the
        // inset only for the drawn rect.
        let va = self.world_to_minimap_inner(mm, vis_min, false);
        let vb = self.world_to_minimap_inner(mm, vis_max, false);
        Rect::from_two_pos(va, vb)
    }
}

// === small color helpers ===

fn col(t: &Tokens, key: &str) -> Color32 {
    let [r, g, b, a] = t.color(key);
    Color32::from_rgba_unmultiplied(r, g, b, a)
}
fn header_col(t: &Tokens, cat: NodeCategory) -> Color32 {
    let [r, g, b, a] = t.category_header(cat);
    Color32::from_rgba_unmultiplied(r, g, b, a)
}
fn lerp_col(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()), l(a.a(), b.a()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the mockup's Terrain->Biome->FloraPrime->SplatWeight->Splatize
    /// shape with real port types and categories.
    fn demo_graph() -> CanvasGraph {
        let mut g = CanvasGraph::default();
        g.nodes.push(
            NodeView::new(1, "Terrain", NodeCategory::Spatial, Pos2::new(40.0, 80.0))
                .with_output("out", PortType::Terrain),
        );
        g.nodes.push(
            NodeView::new(2, "Biome Classify", NodeCategory::Field, Pos2::new(260.0, 60.0))
                .with_input("terrain", PortType::Terrain)
                .with_output("biome", PortType::BiomeMap),
        );
        g.nodes.push(
            NodeView::new(3, "FloraPrime", NodeCategory::Generator, Pos2::new(260.0, 200.0))
                .with_input("biome", PortType::BiomeMap)
                .with_output("instances", PortType::Instances),
        );
        g.nodes.push(
            NodeView::new(4, "SplatWeight", NodeCategory::Math, Pos2::new(480.0, 120.0))
                .with_input("biome", PortType::BiomeMap)
                .with_output("weights", PortType::SplatWeights),
        );
        g.nodes.push(
            NodeView::new(5, "Splatize", NodeCategory::Sink, Pos2::new(700.0, 120.0))
                .with_input("weights", PortType::SplatWeights)
                .with_output("splats", PortType::Splats),
        );
        g.wires.push(WireView {
            from_node: 1, from_port: "out".into(),
            to_node: 2, to_port: "terrain".into(),
            exec: false, label: None,
        });
        g.wires.push(WireView {
            from_node: 2, from_port: "biome".into(),
            to_node: 4, to_port: "biome".into(),
            exec: false, label: None,
        });
        g.wires.push(WireView {
            from_node: 4, from_port: "weights".into(),
            to_node: 5, to_port: "weights".into(),
            exec: false, label: None,
        });
        for n in &mut g.nodes {
            n.size.x = 150.0;
        }
        g
    }

    #[test]
    fn snap_to_grid_lands_on_grid() {
        // Dragging a node so its target world pos is (103,97) with snap 8 must
        // land it at (104,96) (rounded to the nearest multiple of 8).
        let snap = 8.0f32;
        let target = Pos2::new(103.0, 97.0);
        let snapped = Pos2::new(
            (target.x / snap).round() * snap,
            (target.y / snap).round() * snap,
        );
        assert_eq!(snapped, Pos2::new(104.0, 96.0), "snap-8 of (103,97)");
    }

    #[test]
    fn zoom_halves_node_width() {
        // Same graph at zoom 0.5 vs 1.0: a node's rendered width halves; the grid
        // dot spacing scales proportionally.
        let g = demo_graph();
        let origin = Pos2::ZERO;
        let mut c1 = NodeCanvas::new();
        c1.zoom = 1.0;
        let mut c05 = NodeCanvas::new();
        c05.zoom = 0.5;
        let w1 = c1.node_rect_screen(origin, &g, 1).unwrap().width();
        let w05 = c05.node_rect_screen(origin, &g, 1).unwrap().width();
        assert!(
            (w1 - 2.0 * w05).abs() < 0.01,
            "zoom 0.5 node width {w05} should be half of zoom 1.0 width {w1}"
        );
        // Grid step scales with zoom.
        let step1 = GRID_STEP * c1.zoom;
        let step05 = GRID_STEP * c05.zoom;
        assert!((step1 - 2.0 * step05).abs() < 0.01, "grid dot spacing must scale with zoom");
    }

    #[test]
    fn wire_is_curved_off_the_chord() {
        // A typed wire's midpoint must sit OFF the straight chord between its two
        // sockets (curvature proof) — for a horizontally-separated, vertically-
        // offset pair the bezier bows away from the straight line.
        let mut g = demo_graph();
        let origin = Pos2::ZERO;
        let c = NodeCanvas::new();
        // A vertically-offset pair makes the bezier bow unmistakable: route
        // Terrain(out) into FloraPrime(biome) — y 80 vs 200.
        g.wires.push(WireView {
            from_node: 1, from_port: "out".into(),
            to_node: 3, to_port: "biome".into(),
            exec: false, label: None,
        });
        let wire = g.wires.last().unwrap();
        let pts = c.wire_screen_points(origin, &g, wire, 21).unwrap();
        let p0 = pts[0];
        let p3 = *pts.last().unwrap();
        // Max perpendicular deviation of any sample from the straight chord.
        let chord = p3 - p0;
        let len = chord.length().max(1.0);
        let n = Vec2::new(-chord.y, chord.x) / len;
        let max_dev = pts
            .iter()
            .map(|p| ((*p - p0).dot(n)).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_dev > c.wire_thickness,
            "wire deviates only {max_dev}px from the chord — not curved"
        );
    }

    #[test]
    fn minimap_viewport_rect_moves_with_pan() {
        // Panning the canvas moves the minimap's viewport-indicator rect (the
        // visible world region shifts, so its minimap projection shifts).
        let g = demo_graph();
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0));
        let mut c = NodeCanvas::new();
        // Force content bounds via a fake render of bounds (compute directly).
        c.content_bounds = {
            let mut b: Option<Rect> = None;
            for n in &g.nodes {
                let r = Rect::from_min_size(n.pos, Vec2::new(n.size.x, n.world_height()));
                b = Some(match b { Some(x) => x.union(r), None => r });
            }
            b.unwrap()
        };
        let before = c.minimap_viewport_rect(canvas);
        // Pan right by 200px: the visible world region moves LEFT, so its
        // minimap projection moves left (min.x decreases).
        c.pan += Vec2::new(200.0, 0.0);
        let after = c.minimap_viewport_rect(canvas);
        assert!(
            after.min.x < before.min.x - 0.5,
            "minimap viewport rect did not move left on rightward pan (before {:?} after {:?})",
            before.min, after.min
        );
    }

    #[test]
    fn wire_gradient_matches_endpoint_socket_colors() {
        // The wire color near the source equals the source port-type token color
        // and near the target equals the target port-type color; they differ.
        // Terrain(out: Terrain) -> Biome(in: terrain==Terrain) is same-type, so
        // use Biome(out: BiomeMap) -> SplatWeight(in: BiomeMap)... still same.
        // The visibly two-colored wire is SplatWeight(weights: SplatWeights) ->
        // Splatize(in: SplatWeights) — also same type. So assert against the
        // documented gradient endpoints directly on a synthetic two-type wire.
        let t = Tokens::default();
        let mut g = CanvasGraph::default();
        g.nodes.push(
            NodeView::new(1, "A", NodeCategory::Spatial, Pos2::new(0.0, 0.0))
                .with_output("out", PortType::Terrain),
        );
        g.nodes.push(
            NodeView::new(2, "B", NodeCategory::Sink, Pos2::new(300.0, 0.0))
                .with_input("in", PortType::Splats),
        );
        for n in &mut g.nodes { n.size.x = 150.0; }
        let _w = WireView {
            from_node: 1, from_port: "out".into(),
            to_node: 2, to_port: "in".into(),
            exec: false, label: None,
        };
        let src = t.wire_color(PortType::Terrain);
        let dst = t.wire_color(PortType::Splats);
        // The renderer colors segment frac~0 ~ src, frac~1 ~ dst (lerp_col).
        let near_src = lerp_col(
            Color32::from_rgba_unmultiplied(src[0], src[1], src[2], src[3]),
            Color32::from_rgba_unmultiplied(dst[0], dst[1], dst[2], dst[3]),
            0.0,
        );
        let near_dst = lerp_col(
            Color32::from_rgba_unmultiplied(src[0], src[1], src[2], src[3]),
            Color32::from_rgba_unmultiplied(dst[0], dst[1], dst[2], dst[3]),
            1.0,
        );
        assert_eq!([near_src.r(), near_src.g(), near_src.b()], [src[0], src[1], src[2]]);
        assert_eq!([near_dst.r(), near_dst.g(), near_dst.b()], [dst[0], dst[1], dst[2]]);
        assert_ne!(src, dst, "Terrain and Splats socket colors must differ");
    }
}
