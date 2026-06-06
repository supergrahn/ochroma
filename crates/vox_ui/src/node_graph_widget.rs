//! Reusable egui-based node graph renderer.
//! Used by visual scripting, material editor, animation state machine, etc.

/// A visual node in the graph widget.
#[derive(Debug, Clone)]
pub struct VisualNode {
    pub id: u32,
    pub title: String,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [u8; 3],
    pub inputs: Vec<VisualPin>,
    pub outputs: Vec<VisualPin>,
    pub selected: bool,
    pub collapsed: bool,
}

#[derive(Debug, Clone)]
pub struct VisualPin {
    pub name: String,
    pub pin_type: VisualPinType,
    pub connected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualPinType {
    Flow,
    Float,
    Vec3,
    Bool,
    String,
    Spectral,
    Any,
}

impl VisualPinType {
    pub fn color(&self) -> [u8; 3] {
        match self {
            Self::Flow => [255, 255, 255],
            Self::Float => [100, 200, 100],
            Self::Vec3 => [220, 200, 50],
            Self::Bool => [200, 80, 80],
            Self::String => [200, 100, 180],
            Self::Spectral => [100, 150, 255],
            Self::Any => [150, 150, 150],
        }
    }
}

/// A visual connection between two pins.
///
/// Note: `last_value` is intentionally NOT a field here so existing struct-literal
/// construction sites (e.g. engine_runner) keep compiling unchanged. The formatted
/// snapshot of the value that flowed through a wire during the last `evaluate()` is
/// stored side-band in [`NodeGraphWidget::wire_values`] and threaded via
/// [`NodeGraphWidget::set_wire_value`] / [`NodeGraphWidget::wire_value`].
#[derive(Debug, Clone)]
pub struct VisualConnection {
    pub from_node: u32,
    pub from_pin: String,
    pub to_node: u32,
    pub to_pin: String,
    pub color: [u8; 3],
}

/// Identity of a wire (connection) for value-inspection lookups.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WireKey {
    pub from_node: u32,
    pub from_pin: String,
    pub to_node: u32,
    pub to_pin: String,
}

impl WireKey {
    pub fn of(conn: &VisualConnection) -> Self {
        Self {
            from_node: conn.from_node,
            from_pin: conn.from_pin.clone(),
            to_node: conn.to_node,
            to_pin: conn.to_pin.clone(),
        }
    }
}

/// A comment box: a translucent tinted rectangle with a title strip that groups
/// member nodes. Moving the box moves all member nodes by the same delta — the
/// classic UE "comment" / Unity group-box authoring affordance.
#[derive(Debug, Clone)]
pub struct CommentBox {
    pub id: u32,
    pub title: String,
    pub position: [f32; 2],
    pub size: [f32; 2],
    /// RGBA tint; the alpha controls how translucent the body fill is.
    pub tint: [u8; 4],
    pub members: Vec<u32>,
}

/// The node graph widget state.
pub struct NodeGraphWidget {
    pub nodes: Vec<VisualNode>,
    pub connections: Vec<VisualConnection>,
    pub scroll_offset: [f32; 2],
    pub zoom: f32,
    pub dragging_node: Option<u32>,
    pub dragging_connection: Option<DragConnection>,
    pub selected_nodes: Vec<u32>,
    pub grid_size: f32,
    /// Comment boxes, drawn behind nodes.
    pub comments: Vec<CommentBox>,
    /// Formatted snapshot of the value last seen on each wire, keyed by [`WireKey`].
    /// Populated by the editor during `evaluate()` and rendered as a value chip.
    pub wire_values: std::collections::HashMap<WireKey, String>,
}

#[derive(Debug, Clone)]
pub struct DragConnection {
    pub from_node: u32,
    pub from_pin: String,
    pub is_output: bool,
    pub mouse_pos: [f32; 2],
}

/// Actions the widget produces for the host system to process.
#[derive(Debug, Clone)]
pub enum NodeGraphAction {
    NodeMoved { id: u32, new_pos: [f32; 2] },
    NodeSelected { id: u32 },
    NodeDeleted { id: u32 },
    ConnectionCreated { from_node: u32, from_pin: String, to_node: u32, to_pin: String },
    ConnectionDeleted { from_node: u32, from_pin: String },
    PanChanged { offset: [f32; 2] },
    ZoomChanged { zoom: f32 },
}

impl NodeGraphWidget {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            scroll_offset: [0.0; 2],
            zoom: 1.0,
            dragging_node: None,
            dragging_connection: None,
            selected_nodes: Vec::new(),
            grid_size: 20.0,
            comments: Vec::new(),
            wire_values: std::collections::HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: VisualNode) {
        self.nodes.push(node);
    }

    pub fn add_connection(&mut self, conn: VisualConnection) {
        self.connections.push(conn);
    }

    pub fn remove_node(&mut self, id: u32) {
        self.nodes.retain(|n| n.id != id);
        self.connections.retain(|c| c.from_node != id && c.to_node != id);
    }

    /// Calculate the screen position of a pin for connection drawing.
    pub fn pin_position(&self, node_id: u32, pin_name: &str, is_output: bool) -> Option<[f32; 2]> {
        let node = self.nodes.iter().find(|n| n.id == node_id)?;
        let pins = if is_output { &node.outputs } else { &node.inputs };
        let _pin_idx = pins.iter().position(|p| p.name == pin_name)?;

        let x = if is_output {
            node.position[0] + node.size[0]
        } else {
            node.position[0]
        };
        let y = node.position[1] + 30.0 + _pin_idx as f32 * 24.0;

        Some([x + self.scroll_offset[0], y + self.scroll_offset[1]])
    }

    /// Render the node graph to a pixel buffer (software renderer path).
    /// Draws: background grid, connections as lines, nodes as rectangles with pins.
    pub fn render_to_pixels(
        &self,
        pixels: &mut [[u8; 4]],
        width: u32,
        height: u32,
    ) {
        // Background grid
        for y in 0..height {
            for x in 0..width {
                let gx = (x as f32 + self.scroll_offset[0]) % (self.grid_size * self.zoom);
                let gy = (y as f32 + self.scroll_offset[1]) % (self.grid_size * self.zoom);
                let is_grid = gx < 1.0 || gy < 1.0;
                let idx = (y * width + x) as usize;
                if idx < pixels.len() {
                    pixels[idx] = if is_grid { [40, 40, 45, 255] } else { [30, 30, 35, 255] };
                }
            }
        }

        // Comment boxes — drawn behind nodes/wires: translucent tinted body + a
        // brighter title strip along the top edge (#9a).
        for comment in &self.comments {
            let cx = (comment.position[0] + self.scroll_offset[0]) as i32;
            let cy = (comment.position[1] + self.scroll_offset[1]) as i32;
            let cw = comment.size[0] as i32;
            let ch = comment.size[1] as i32;
            // Translucent body fill.
            fill_rect(pixels, width, height, cx, cy, cw, ch, comment.tint);
            // Title strip: same hue but fully opaque + brightened so it reads as a header.
            let strip = [
                comment.tint[0].saturating_add(60),
                comment.tint[1].saturating_add(60),
                comment.tint[2].saturating_add(60),
                255,
            ];
            fill_rect(pixels, width, height, cx, cy, cw, 16, strip);
            // Title text inside the strip.
            draw_text(pixels, width, height, cx + 3, cy + 5, &comment.title, [240, 240, 245, 255]);
            // Border so the group is visually bounded.
            draw_rect_outline(pixels, width, height, cx, cy, cw, ch, [
                comment.tint[0].saturating_add(90),
                comment.tint[1].saturating_add(90),
                comment.tint[2].saturating_add(90),
                255,
            ]);
        }

        // Draw connections as straight lines (Bezier curves in future)
        for conn in &self.connections {
            if let (Some(from_pos), Some(to_pos)) = (
                self.pin_position(conn.from_node, &conn.from_pin, true),
                self.pin_position(conn.to_node, &conn.to_pin, false),
            ) {
                draw_line(
                    pixels, width, height,
                    from_pos[0] as i32, from_pos[1] as i32,
                    to_pos[0] as i32, to_pos[1] as i32,
                    [conn.color[0], conn.color[1], conn.color[2], 255],
                );
            }
        }

        // Draw nodes
        for node in &self.nodes {
            let x = (node.position[0] + self.scroll_offset[0]) as i32;
            let y = (node.position[1] + self.scroll_offset[1]) as i32;
            let w = node.size[0] as i32;
            let h = node.size[1] as i32;

            // Node background
            let bg = if node.selected { [60, 60, 70, 240] } else { [45, 45, 55, 230] };
            fill_rect(pixels, width, height, x, y, w, h, bg);

            // Title bar
            let title_color = [node.color[0], node.color[1], node.color[2], 255];
            fill_rect(pixels, width, height, x, y, w, 24, title_color);

            // Pin circles
            for (i, pin) in node.inputs.iter().enumerate() {
                let py = y + 30 + i as i32 * 24;
                let color = pin.pin_type.color();
                fill_circle(pixels, width, height, x, py + 4, 5, [color[0], color[1], color[2], 255]);
            }
            for (i, pin) in node.outputs.iter().enumerate() {
                let py = y + 30 + i as i32 * 24;
                let color = pin.pin_type.color();
                fill_circle(pixels, width, height, x + w, py + 4, 5, [color[0], color[1], color[2], 255]);
            }

            // Selection border
            if node.selected {
                draw_rect_outline(pixels, width, height, x - 1, y - 1, w + 2, h + 2, [255, 180, 50, 255]);
            }
        }

        // Wire value chips (#9b) — drawn on top, near each wire's midpoint, only for
        // wires that carried a value during the last evaluate().
        for conn in &self.connections {
            let Some(value) = self.wire_value(conn) else { continue };
            if let (Some(from_pos), Some(to_pos)) = (
                self.pin_position(conn.from_node, &conn.from_pin, true),
                self.pin_position(conn.to_node, &conn.to_pin, false),
            ) {
                let mid_x = ((from_pos[0] + to_pos[0]) * 0.5) as i32;
                let mid_y = ((from_pos[1] + to_pos[1]) * 0.5) as i32;
                let text_w = (value.len() as i32) * (CHAR_W + 1);
                let chip_w = text_w + 6;
                let chip_h = CHAR_H + 6;
                let cx = mid_x - chip_w / 2;
                let cy = mid_y - chip_h / 2;
                // Chip background (opaque dark) + accent border in the wire's color.
                fill_rect(pixels, width, height, cx, cy, chip_w, chip_h, [20, 22, 30, 255]);
                draw_rect_outline(pixels, width, height, cx, cy, chip_w, chip_h, [
                    conn.color[0], conn.color[1], conn.color[2], 255,
                ]);
                draw_text(pixels, width, height, cx + 3, cy + 3, value, [225, 230, 240, 255]);
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    // --- Comment boxes (#9a) ---

    /// Add a comment box. Drawn behind nodes with a translucent tint + title strip.
    pub fn add_comment(&mut self, comment: CommentBox) {
        self.comments.push(comment);
    }

    pub fn comment_count(&self) -> usize {
        self.comments.len()
    }

    /// Move a comment box by `delta`, dragging every member node along with it by the
    /// same delta — the standard "move the group, move its contents" behavior.
    /// Returns `false` if no comment with `comment_id` exists.
    pub fn move_comment(&mut self, comment_id: u32, delta: [f32; 2]) -> bool {
        let members = match self.comments.iter_mut().find(|c| c.id == comment_id) {
            Some(c) => {
                c.position[0] += delta[0];
                c.position[1] += delta[1];
                c.members.clone()
            }
            None => return false,
        };
        for node in &mut self.nodes {
            if members.contains(&node.id) {
                node.position[0] += delta[0];
                node.position[1] += delta[1];
            }
        }
        true
    }

    // --- Wire value inspection (#9b) ---

    /// Record the formatted snapshot of the value that flowed through a wire during
    /// the last evaluate(). The editor calls this; the widget renders it as a chip.
    pub fn set_wire_value(&mut self, conn: &VisualConnection, value: impl Into<String>) {
        self.wire_values.insert(WireKey::of(conn), value.into());
    }

    /// Fetch the last recorded value snapshot for a wire, if any.
    pub fn wire_value(&self, conn: &VisualConnection) -> Option<&str> {
        self.wire_values.get(&WireKey::of(conn)).map(|s| s.as_str())
    }

    /// Clear all recorded wire value snapshots (e.g. before a fresh evaluate()).
    pub fn clear_wire_values(&mut self) {
        self.wire_values.clear();
    }

    /// Render the node graph interactively using egui.
    /// Returns actions produced by user interaction this frame.
    /// Call this inside an egui window or panel.
    pub fn show_egui(&mut self, ui: &mut egui::Ui) -> Vec<NodeGraphAction> {
        let mut actions = Vec::new();
        let avail = ui.available_rect_before_wrap();
        let painter = ui.painter_at(avail);

        // Allocate the full available area for input
        let response = ui.allocate_rect(avail, egui::Sense::click_and_drag());

        // Background
        painter.rect_filled(avail, 0.0, egui::Color32::from_rgb(28, 28, 35));

        // Grid lines
        let grid_px = self.grid_size * self.zoom;
        let ox = self.scroll_offset[0] % grid_px;
        let oy = self.scroll_offset[1] % grid_px;
        let grid_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(40, 42, 52));
        let mut gx = avail.min.x + ox;
        while gx < avail.max.x {
            painter.line_segment([egui::pos2(gx, avail.min.y), egui::pos2(gx, avail.max.y)], grid_stroke);
            gx += grid_px;
        }
        let mut gy = avail.min.y + oy;
        while gy < avail.max.y {
            painter.line_segment([egui::pos2(avail.min.x, gy), egui::pos2(avail.max.x, gy)], grid_stroke);
            gy += grid_px;
        }

        // Pan (middle mouse drag)
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            self.scroll_offset[0] += delta.x;
            self.scroll_offset[1] += delta.y;
            actions.push(NodeGraphAction::PanChanged { offset: self.scroll_offset });
        }

        // Zoom (scroll wheel)
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta.abs() > 0.1 {
            let old_zoom = self.zoom;
            self.zoom = (self.zoom * (1.0 + scroll_delta * 0.001)).clamp(0.2, 4.0);
            if (self.zoom - old_zoom).abs() > 1e-4 {
                actions.push(NodeGraphAction::ZoomChanged { zoom: self.zoom });
            }
        }

        // Draw connections as cubic bezier curves
        for conn in &self.connections {
            let from_pos = self.pin_screen_pos(conn.from_node, &conn.from_pin, true, avail.min);
            let to_pos   = self.pin_screen_pos(conn.to_node,   &conn.to_pin,   false, avail.min);
            if let (Some(fp), Some(tp)) = (from_pos, to_pos) {
                let ctrl_dx = ((tp.x - fp.x).abs() * 0.5).max(50.0);
                painter.add(egui::Shape::CubicBezier(egui::epaint::CubicBezierShape {
                    points: [
                        fp,
                        egui::pos2(fp.x + ctrl_dx, fp.y),
                        egui::pos2(tp.x - ctrl_dx, tp.y),
                        tp,
                    ],
                    closed: false,
                    fill: egui::Color32::TRANSPARENT,
                    stroke: egui::Stroke::new(2.0, egui::Color32::from_rgb(
                        conn.color[0], conn.color[1], conn.color[2],
                    )).into(),
                }));
            }
        }

        // Draw nodes — collect move/select actions separately to avoid borrow conflict
        let mut move_actions:   Vec<(u32, [f32; 2])> = Vec::new();
        let mut select_actions: Vec<u32>              = Vec::new();

        for node in &mut self.nodes {
            let sx = avail.min.x + node.position[0] * self.zoom + self.scroll_offset[0];
            let sy = avail.min.y + node.position[1] * self.zoom + self.scroll_offset[1];
            let sw = node.size[0] * self.zoom;
            let pin_rows = node.inputs.len().max(node.outputs.len()) as f32;
            let sh = (30.0 + pin_rows * 24.0 + 8.0) * self.zoom;

            let node_rect = egui::Rect::from_min_size(egui::pos2(sx, sy), egui::vec2(sw, sh));
            let node_resp = ui.allocate_rect(node_rect, egui::Sense::click_and_drag());

            // Body
            let bg = if node.selected {
                egui::Color32::from_rgb(55, 65, 90)
            } else {
                egui::Color32::from_rgb(36, 36, 48)
            };
            painter.rect_filled(node_rect, 4.0 * self.zoom, bg);

            // Title bar
            let title_h = 24.0 * self.zoom;
            let title_rect = egui::Rect::from_min_size(node_rect.min, egui::vec2(sw, title_h));
            painter.rect_filled(title_rect, 4.0 * self.zoom, egui::Color32::from_rgb(node.color[0], node.color[1], node.color[2]));
            painter.text(
                title_rect.center(),
                egui::Align2::CENTER_CENTER,
                &node.title,
                egui::FontId::proportional((11.0 * self.zoom).max(8.0)),
                egui::Color32::WHITE,
            );

            // Border
            let border = if node.selected {
                egui::Color32::from_rgb(100, 160, 255)
            } else {
                egui::Color32::from_rgb(58, 68, 100)
            };
            painter.rect_stroke(node_rect, 4.0 * self.zoom, egui::Stroke::new(1.5, border), egui::StrokeKind::Outside);

            // Input pins (left side)
            for (i, pin) in node.inputs.iter().enumerate() {
                let py = sy + (30.0 + i as f32 * 24.0 + 8.0) * self.zoom;
                let c = pin.pin_type.color();
                let pin_col = egui::Color32::from_rgb(c[0], c[1], c[2]);
                painter.circle_filled(egui::pos2(sx, py), 5.0 * self.zoom, pin_col);
                painter.text(
                    egui::pos2(sx + 10.0 * self.zoom, py),
                    egui::Align2::LEFT_CENTER,
                    &pin.name,
                    egui::FontId::proportional((10.0 * self.zoom).max(7.0)),
                    egui::Color32::from_rgb(190, 190, 200),
                );
            }

            // Output pins (right side)
            for (i, pin) in node.outputs.iter().enumerate() {
                let py = sy + (30.0 + i as f32 * 24.0 + 8.0) * self.zoom;
                let c = pin.pin_type.color();
                let pin_col = egui::Color32::from_rgb(c[0], c[1], c[2]);
                painter.circle_filled(egui::pos2(sx + sw, py), 5.0 * self.zoom, pin_col);
                painter.text(
                    egui::pos2(sx + sw - 10.0 * self.zoom, py),
                    egui::Align2::RIGHT_CENTER,
                    &pin.name,
                    egui::FontId::proportional((10.0 * self.zoom).max(7.0)),
                    egui::Color32::from_rgb(190, 190, 200),
                );
            }

            // Drag to move
            if node_resp.dragged_by(egui::PointerButton::Primary) {
                let delta = node_resp.drag_delta();
                let new_pos = [
                    node.position[0] + delta.x / self.zoom,
                    node.position[1] + delta.y / self.zoom,
                ];
                move_actions.push((node.id, new_pos));
            }

            // Click to select
            if node_resp.clicked() {
                select_actions.push(node.id);
            }
        }

        // Apply mutations after the borrow-loop ends
        for (id, pos) in move_actions {
            if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
                n.position = pos;
            }
            actions.push(NodeGraphAction::NodeMoved { id, new_pos: pos });
        }
        for id in select_actions {
            self.selected_nodes.clear();
            self.selected_nodes.push(id);
            for n in &mut self.nodes {
                n.selected = n.id == id;
            }
            actions.push(NodeGraphAction::NodeSelected { id });
        }

        actions
    }

    /// Screen-space position of a pin for connection curve drawing.
    fn pin_screen_pos(
        &self,
        node_id:   u32,
        pin_name:  &str,
        is_output: bool,
        origin:    egui::Pos2,
    ) -> Option<egui::Pos2> {
        let node = self.nodes.iter().find(|n| n.id == node_id)?;
        let pins = if is_output { &node.outputs } else { &node.inputs };
        let idx = pins.iter().position(|p| p.name == pin_name)?;
        let sx = origin.x + node.position[0] * self.zoom + self.scroll_offset[0];
        let sy = origin.y + node.position[1] * self.zoom + self.scroll_offset[1];
        let sw = node.size[0] * self.zoom;
        let x  = if is_output { sx + sw } else { sx };
        let y  = sy + (30.0 + idx as f32 * 24.0 + 8.0) * self.zoom;
        Some(egui::pos2(x, y))
    }
}

impl Default for NodeGraphWidget {
    fn default() -> Self {
        Self::new()
    }
}

// --- Helper drawing functions ---

#[allow(clippy::too_many_arguments)]
fn fill_rect(pixels: &mut [[u8; 4]], w: u32, h: u32, x: i32, y: i32, rw: i32, rh: i32, color: [u8; 4]) {
    for dy in 0..rh {
        for dx in 0..rw {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                let idx = (py * w as i32 + px) as usize;
                if idx < pixels.len() {
                    let a = color[3] as f32 / 255.0;
                    let dst = pixels[idx];
                    pixels[idx] = [
                        (color[0] as f32 * a + dst[0] as f32 * (1.0 - a)) as u8,
                        (color[1] as f32 * a + dst[1] as f32 * (1.0 - a)) as u8,
                        (color[2] as f32 * a + dst[2] as f32 * (1.0 - a)) as u8,
                        255,
                    ];
                }
            }
        }
    }
}

fn fill_circle(pixels: &mut [[u8; 4]], w: u32, h: u32, cx: i32, cy: i32, r: i32, color: [u8; 4]) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                    pixels[(py * w as i32 + px) as usize] = color;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_rect_outline(pixels: &mut [[u8; 4]], w: u32, h: u32, x: i32, y: i32, rw: i32, rh: i32, color: [u8; 4]) {
    for dx in 0..rw {
        set_pixel(pixels, w, h, x + dx, y, color);
        set_pixel(pixels, w, h, x + dx, y + rh - 1, color);
    }
    for dy in 0..rh {
        set_pixel(pixels, w, h, x, y + dy, color);
        set_pixel(pixels, w, h, x + rw - 1, y + dy, color);
    }
}

// --- Minimal 3x5 bitmap font for value chips & comment titles ---

const CHAR_W: i32 = 3;
const CHAR_H: i32 = 5;

/// Return the 5-row (top→bottom) 3-bit-wide glyph for a character. Unknown chars
/// render as a small dot. Covers 0-9, A-Z, and a handful of punctuation used in
/// formatted port values (., -, :, space, [, ], ,).
fn glyph(c: char) -> [u8; 5] {
    match c.to_ascii_uppercase() {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'A' => [0b111, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b111, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b111, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b101, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b111],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b111, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b111, 0b101, 0b111, 0b100, 0b100],
        'Q' => [0b111, 0b101, 0b101, 0b111, 0b011],
        'R' => [0b111, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ',' => [0b000, 0b000, 0b000, 0b010, 0b100],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '[' => [0b011, 0b010, 0b010, 0b010, 0b011],
        ']' => [0b110, 0b010, 0b010, 0b010, 0b110],
        '=' => [0b000, 0b111, 0b000, 0b111, 0b000],
        ' ' => [0b000, 0b000, 0b000, 0b000, 0b000],
        _   => [0b000, 0b000, 0b010, 0b000, 0b000],
    }
}

/// Draw a left-aligned text string using the 3x5 bitmap font, top-left at (x, y).
fn draw_text(pixels: &mut [[u8; 4]], w: u32, h: u32, x: i32, y: i32, text: &str, color: [u8; 4]) {
    let mut cursor = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..CHAR_W {
                // Bit (CHAR_W-1-col) is the leftmost column.
                if bits & (1 << (CHAR_W - 1 - col)) != 0 {
                    set_pixel(pixels, w, h, cursor + col, y + row as i32, color);
                }
            }
        }
        cursor += CHAR_W + 1;
    }
}

fn set_pixel(pixels: &mut [[u8; 4]], w: u32, h: u32, x: i32, y: i32, color: [u8; 4]) {
    if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
        pixels[(y * w as i32 + x) as usize] = color;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_line(pixels: &mut [[u8; 4]], w: u32, h: u32, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 4]) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        set_pixel(pixels, w, h, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_node(id: u32, x: f32, y: f32) -> VisualNode {
        VisualNode {
            id,
            title: format!("Node {}", id),
            position: [x, y],
            size: [160.0, 100.0],
            color: [80, 120, 200],
            inputs: vec![
                VisualPin { name: "In".into(), pin_type: VisualPinType::Float, connected: false },
            ],
            outputs: vec![
                VisualPin { name: "Out".into(), pin_type: VisualPinType::Float, connected: false },
            ],
            selected: false,
            collapsed: false,
        }
    }

    #[test]
    fn test_add_nodes() {
        let mut widget = NodeGraphWidget::new();
        widget.add_node(make_test_node(1, 0.0, 0.0));
        widget.add_node(make_test_node(2, 200.0, 0.0));
        assert_eq!(widget.node_count(), 2);
    }

    #[test]
    fn test_add_connections() {
        let mut widget = NodeGraphWidget::new();
        widget.add_node(make_test_node(1, 0.0, 0.0));
        widget.add_node(make_test_node(2, 200.0, 0.0));
        widget.add_connection(VisualConnection {
            from_node: 1,
            from_pin: "Out".into(),
            to_node: 2,
            to_pin: "In".into(),
            color: [100, 200, 100],
        });
        assert_eq!(widget.connection_count(), 1);
    }

    #[test]
    fn test_pin_position_computes() {
        let mut widget = NodeGraphWidget::new();
        widget.add_node(make_test_node(1, 50.0, 50.0));
        let pos = widget.pin_position(1, "Out", true);
        assert!(pos.is_some());
        let pos = pos.unwrap();
        // Output pin x = node.position.x + node.size.x + scroll_offset.x = 50 + 160 + 0 = 210
        assert!((pos[0] - 210.0).abs() < 0.01);
        // y = node.position.y + 30 + 0*24 + scroll_offset.y = 50 + 30 + 0 = 80
        assert!((pos[1] - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_render_to_pixels_non_empty() {
        let mut widget = NodeGraphWidget::new();
        widget.add_node(make_test_node(1, 10.0, 10.0));
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![[0u8; 4]; (w * h) as usize];
        widget.render_to_pixels(&mut pixels, w, h);
        // Check that not all pixels are zero (background was drawn)
        let non_zero = pixels.iter().any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0);
        assert!(non_zero, "render_to_pixels should produce non-empty output");
    }

    #[test]
    fn test_remove_node_cleans_connections() {
        let mut widget = NodeGraphWidget::new();
        widget.add_node(make_test_node(1, 0.0, 0.0));
        widget.add_node(make_test_node(2, 200.0, 0.0));
        widget.add_node(make_test_node(3, 400.0, 0.0));
        widget.add_connection(VisualConnection {
            from_node: 1, from_pin: "Out".into(), to_node: 2, to_pin: "In".into(), color: [100, 200, 100],
        });
        widget.add_connection(VisualConnection {
            from_node: 2, from_pin: "Out".into(), to_node: 3, to_pin: "In".into(), color: [100, 200, 100],
        });
        assert_eq!(widget.connection_count(), 2);
        widget.remove_node(2);
        assert_eq!(widget.node_count(), 2);
        assert_eq!(widget.connection_count(), 0); // both connections involved node 2
    }

    #[test]
    fn show_egui_method_exists() {
        // Compilation test — confirms show_egui signature is correct.
        let _: fn(&mut NodeGraphWidget, &mut egui::Ui) -> Vec<NodeGraphAction> =
            |w, ui| w.show_egui(ui);
    }

    #[test]
    fn new_widget_is_empty() {
        let w = NodeGraphWidget::new();
        assert_eq!(w.node_count(), 0);
        assert_eq!(w.connection_count(), 0);
        assert_eq!(w.comment_count(), 0);
    }

    // --- Comment box tests (#9a) ---

    #[test]
    fn move_comment_shifts_member_nodes_by_same_delta() {
        let mut w = NodeGraphWidget::new();
        w.add_node(make_test_node(1, 100.0, 100.0));
        w.add_node(make_test_node(2, 300.0, 100.0));
        w.add_node(make_test_node(3, 500.0, 500.0)); // NOT a member
        w.add_comment(CommentBox {
            id: 10,
            title: "Terrain".into(),
            position: [50.0, 50.0],
            size: [400.0, 200.0],
            tint: [60, 90, 140, 90],
            members: vec![1, 2],
        });

        let moved = w.move_comment(10, [40.0, -25.0]);
        assert!(moved, "move_comment should report success for an existing comment");

        // Member nodes shifted by exactly the delta.
        let n1 = w.nodes.iter().find(|n| n.id == 1).unwrap();
        let n2 = w.nodes.iter().find(|n| n.id == 2).unwrap();
        assert_eq!(n1.position, [140.0, 75.0]);
        assert_eq!(n2.position, [340.0, 75.0]);
        // Non-member untouched.
        let n3 = w.nodes.iter().find(|n| n.id == 3).unwrap();
        assert_eq!(n3.position, [500.0, 500.0]);
        // Comment itself moved.
        assert_eq!(w.comments[0].position, [90.0, 25.0]);
        // Unknown id is a no-op returning false.
        assert!(!w.move_comment(999, [1.0, 1.0]));
    }

    #[test]
    fn comment_box_renders_distinct_body_and_brighter_title_strip() {
        let mut w = NodeGraphWidget::new();
        // Big comment box, no nodes, so the comment fully owns its region.
        w.add_comment(CommentBox {
            id: 1,
            title: "X".into(),
            position: [10.0, 10.0],
            size: [40.0, 40.0],
            tint: [60, 90, 140, 200],
            members: vec![],
        });
        let (width, height) = (64u32, 64u32);
        let mut pixels = vec![[30u8, 30, 35, 255]; (width * height) as usize];
        // Capture a known background sample far from the comment.
        w.render_to_pixels(&mut pixels, width, height);
        let bg = pixels[(60 * width + 60) as usize];

        // A pixel inside the comment body (well below the title strip).
        let body = pixels[(35 * width + 30) as usize];
        // A pixel inside the title strip (top rows of the comment).
        let strip = pixels[(15 * width + 30) as usize];

        assert_ne!(body, bg, "comment body should differ from background");
        // The title strip is brighter than the body fill (it's strip = tint + 60).
        let body_lum = body[0] as u32 + body[1] as u32 + body[2] as u32;
        let strip_lum = strip[0] as u32 + strip[1] as u32 + strip[2] as u32;
        assert!(strip_lum > body_lum, "title strip ({strip_lum}) should be brighter than body ({body_lum})");
    }

    // --- Wire value chip tests (#9b) ---

    #[test]
    fn set_and_get_wire_value_round_trips() {
        let mut w = NodeGraphWidget::new();
        let conn = VisualConnection {
            from_node: 1, from_pin: "Out".into(), to_node: 2, to_pin: "In".into(), color: [200, 200, 80],
        };
        assert_eq!(w.wire_value(&conn), None);
        w.set_wire_value(&conn, "Terrain 1024 cells");
        assert_eq!(w.wire_value(&conn), Some("Terrain 1024 cells"));
    }

    #[test]
    fn wire_value_chip_renders_pixels_near_midpoint() {
        let mut w = NodeGraphWidget::new();
        // Two nodes on the same baseline so the wire (and its midpoint) is well
        // inside the buffer.
        w.add_node(make_test_node(1, 10.0, 20.0));
        w.add_node(make_test_node(2, 230.0, 20.0));
        let conn = VisualConnection {
            from_node: 1, from_pin: "Out".into(), to_node: 2, to_pin: "In".into(), color: [200, 80, 80],
        };
        w.add_connection(conn.clone());

        let (width, height) = (512u32, 96u32);
        let render = |w: &NodeGraphWidget| {
            let mut px = vec![[30u8, 30, 35, 255]; (width * height) as usize];
            w.render_to_pixels(&mut px, width, height);
            px
        };

        let without_chip = render(&w);
        w.set_wire_value(&conn, "42");
        let with_chip = render(&w);

        // The chip introduces new pixels not present without a value.
        let diff = without_chip.iter().zip(with_chip.iter()).filter(|(a, b)| a != b).count();
        assert!(diff > 10, "wire value chip should change a region of pixels, diff={diff}");

        // The chip background color [20,22,30] must appear somewhere (it is not a
        // color the grid/wire/background uses).
        let has_chip_bg = with_chip.iter().any(|p| p[0] == 20 && p[1] == 22 && p[2] == 30);
        assert!(has_chip_bg, "chip background fill should be present in the rendered buffer");
    }
}
