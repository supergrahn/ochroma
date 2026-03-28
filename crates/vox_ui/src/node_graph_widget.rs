/// Reusable egui-based node graph renderer.
/// Used by visual scripting, material editor, animation state machine, etc.

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
#[derive(Debug, Clone)]
pub struct VisualConnection {
    pub from_node: u32,
    pub from_pin: String,
    pub to_node: u32,
    pub to_pin: String,
    pub color: [u8; 3],
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
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

impl Default for NodeGraphWidget {
    fn default() -> Self {
        Self::new()
    }
}

// --- Helper drawing functions ---

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

fn set_pixel(pixels: &mut [[u8; 4]], w: u32, h: u32, x: i32, y: i32, color: [u8; 4]) {
    if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
        pixels[(y * w as i32 + x) as usize] = color;
    }
}

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
}
