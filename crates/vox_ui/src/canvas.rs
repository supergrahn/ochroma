use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UICanvas {
    pub name: String,
    pub width: f32,
    pub height: f32,
    pub elements: Vec<UICanvasElement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UICanvasElement {
    pub id: String,
    pub element_type: CanvasElementType,
    pub anchor: Anchor,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub visible: bool,
    pub children: Vec<UICanvasElement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanvasElementType {
    Text {
        content: String,
        font_size: f32,
        color: [u8; 4],
    },
    Image {
        path: String,
        tint: [u8; 4],
    },
    Panel {
        color: [u8; 4],
        border_radius: f32,
    },
    Button {
        label: String,
        on_click: String,
    },
    Slider {
        value: f32,
        min: f32,
        max: f32,
        on_change: String,
    },
    ProgressBar {
        value: f32,
        fill_color: [u8; 4],
        bg_color: [u8; 4],
    },
    VerticalLayout {
        spacing: f32,
    },
    HorizontalLayout {
        spacing: f32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Anchor {
    pub fn resolve(&self, canvas_w: f32, canvas_h: f32) -> [f32; 2] {
        match self {
            Self::TopLeft => [0.0, 0.0],
            Self::TopCenter => [canvas_w / 2.0, 0.0],
            Self::TopRight => [canvas_w, 0.0],
            Self::CenterLeft => [0.0, canvas_h / 2.0],
            Self::Center => [canvas_w / 2.0, canvas_h / 2.0],
            Self::CenterRight => [canvas_w, canvas_h / 2.0],
            Self::BottomLeft => [0.0, canvas_h],
            Self::BottomCenter => [canvas_w / 2.0, canvas_h],
            Self::BottomRight => [canvas_w, canvas_h],
        }
    }
}

impl UICanvas {
    pub fn new(name: &str, width: f32, height: f32) -> Self {
        Self {
            name: name.to_string(),
            width,
            height,
            elements: Vec::new(),
        }
    }

    pub fn add_element(&mut self, element: UICanvasElement) {
        self.elements.push(element);
    }

    pub fn find_element(&self, id: &str) -> Option<&UICanvasElement> {
        fn find_recursive<'a>(elements: &'a [UICanvasElement], id: &str) -> Option<&'a UICanvasElement> {
            for el in elements {
                if el.id == id {
                    return Some(el);
                }
                if let Some(found) = find_recursive(&el.children, id) {
                    return Some(found);
                }
            }
            None
        }
        find_recursive(&self.elements, id)
    }

    pub fn find_element_mut(&mut self, id: &str) -> Option<&mut UICanvasElement> {
        fn find_index_path(elements: &[UICanvasElement], id: &str) -> Option<Vec<usize>> {
            for (i, el) in elements.iter().enumerate() {
                if el.id == id {
                    return Some(vec![i]);
                }
                if let Some(mut path) = find_index_path(&el.children, id) {
                    path.insert(0, i);
                    return Some(path);
                }
            }
            None
        }

        let path = find_index_path(&self.elements, id)?;
        let mut current = &mut self.elements[path[0]];
        for &idx in &path[1..] {
            current = &mut current.children[idx];
        }
        Some(current)
    }

    pub fn set_text(&mut self, id: &str, text: &str) -> bool {
        if let Some(el) = self.find_element_mut(id) {
            if let CanvasElementType::Text { ref mut content, .. } = el.element_type {
                *content = text.to_string();
                return true;
            }
        }
        false
    }

    pub fn set_progress(&mut self, id: &str, value: f32) -> bool {
        if let Some(el) = self.find_element_mut(id) {
            if let CanvasElementType::ProgressBar { value: ref mut v, .. } = el.element_type {
                *v = value.clamp(0.0, 1.0);
                return true;
            }
        }
        false
    }

    pub fn set_visible(&mut self, id: &str, visible: bool) -> bool {
        if let Some(el) = self.find_element_mut(id) {
            el.visible = visible;
            return true;
        }
        false
    }

    pub fn element_count(&self) -> usize {
        fn count_recursive(elements: &[UICanvasElement]) -> usize {
            let mut count = elements.len();
            for el in elements {
                count += count_recursive(&el.children);
            }
            count
        }
        count_recursive(&self.elements)
    }

    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn load_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Render the canvas to pixel buffer (bitmap font).
    pub fn render(&self, pixels: &mut [[u8; 4]], screen_w: u32, screen_h: u32) {
        for element in &self.elements {
            if !element.visible {
                continue;
            }
            self.render_element(element, pixels, screen_w, screen_h);
        }
    }

    fn render_element(
        &self,
        element: &UICanvasElement,
        pixels: &mut [[u8; 4]],
        screen_w: u32,
        screen_h: u32,
    ) {
        let anchor_pos = element.anchor.resolve(self.width, self.height);
        let scale_x = screen_w as f32 / self.width;
        let scale_y = screen_h as f32 / self.height;

        let px = ((anchor_pos[0] + element.position[0]) * scale_x) as i32;
        let py = ((anchor_pos[1] + element.position[1]) * scale_y) as i32;
        let pw = (element.size[0] * scale_x) as i32;
        let ph = (element.size[1] * scale_y) as i32;

        let color = match &element.element_type {
            CanvasElementType::Panel { color, .. } => *color,
            CanvasElementType::Text { color, .. } => *color,
            CanvasElementType::Button { .. } => [128, 128, 128, 255],
            CanvasElementType::ProgressBar { value, fill_color, bg_color } => {
                // Render background then fill
                let fill_w = (pw as f32 * value) as i32;
                // Background
                for y in py..(py + ph) {
                    for x in px..(px + pw) {
                        if x >= 0 && y >= 0 && (x as u32) < screen_w && (y as u32) < screen_h {
                            let idx = (y as u32 * screen_w + x as u32) as usize;
                            if idx < pixels.len() {
                                pixels[idx] = *bg_color;
                            }
                        }
                    }
                }
                // Fill
                for y in py..(py + ph) {
                    for x in px..(px + fill_w) {
                        if x >= 0 && y >= 0 && (x as u32) < screen_w && (y as u32) < screen_h {
                            let idx = (y as u32 * screen_w + x as u32) as usize;
                            if idx < pixels.len() {
                                pixels[idx] = *fill_color;
                            }
                        }
                    }
                }
                // Children
                for child in &element.children {
                    if child.visible {
                        self.render_element(child, pixels, screen_w, screen_h);
                    }
                }
                return;
            }
            _ => [255, 255, 255, 255],
        };

        // Fill rectangle
        for y in py..(py + ph) {
            for x in px..(px + pw) {
                if x >= 0 && y >= 0 && (x as u32) < screen_w && (y as u32) < screen_h {
                    let idx = (y as u32 * screen_w + x as u32) as usize;
                    if idx < pixels.len() {
                        pixels[idx] = color;
                    }
                }
            }
        }

        // Render children
        for child in &element.children {
            if child.visible {
                self.render_element(child, pixels, screen_w, screen_h);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_text(id: &str, text: &str) -> UICanvasElement {
        UICanvasElement {
            id: id.to_string(),
            element_type: CanvasElementType::Text {
                content: text.to_string(),
                font_size: 16.0,
                color: [255, 255, 255, 255],
            },
            anchor: Anchor::TopLeft,
            position: [10.0, 10.0],
            size: [100.0, 20.0],
            visible: true,
            children: vec![],
        }
    }

    #[test]
    fn test_create_canvas() {
        let canvas = UICanvas::new("HUD", 1920.0, 1080.0);
        assert_eq!(canvas.name, "HUD");
        assert_eq!(canvas.width, 1920.0);
        assert_eq!(canvas.height, 1080.0);
        assert_eq!(canvas.element_count(), 0);
    }

    #[test]
    fn test_add_elements() {
        let mut canvas = UICanvas::new("HUD", 1920.0, 1080.0);
        canvas.add_element(make_text("title", "Hello World"));
        canvas.add_element(UICanvasElement {
            id: "panel".to_string(),
            element_type: CanvasElementType::Panel {
                color: [30, 30, 30, 200],
                border_radius: 4.0,
            },
            anchor: Anchor::Center,
            position: [0.0, 0.0],
            size: [400.0, 300.0],
            visible: true,
            children: vec![],
        });
        assert_eq!(canvas.element_count(), 2);
    }

    #[test]
    fn test_anchor_resolve() {
        let w = 1920.0_f32;
        let h = 1080.0_f32;
        assert_eq!(Anchor::TopLeft.resolve(w, h), [0.0, 0.0]);
        assert_eq!(Anchor::Center.resolve(w, h), [960.0, 540.0]);
        assert_eq!(Anchor::BottomRight.resolve(w, h), [1920.0, 1080.0]);
        assert_eq!(Anchor::TopCenter.resolve(w, h), [960.0, 0.0]);
        assert_eq!(Anchor::BottomCenter.resolve(w, h), [960.0, 1080.0]);
    }

    #[test]
    fn test_set_text_updates() {
        let mut canvas = UICanvas::new("HUD", 1920.0, 1080.0);
        canvas.add_element(make_text("score", "0"));
        assert!(canvas.set_text("score", "100"));
        let el = canvas.find_element("score").unwrap();
        if let CanvasElementType::Text { content, .. } = &el.element_type {
            assert_eq!(content, "100");
        } else {
            panic!("expected text element");
        }
        // Non-existent returns false
        assert!(!canvas.set_text("nonexistent", "fail"));
    }

    #[test]
    fn test_json_round_trip() {
        let mut canvas = UICanvas::new("Menu", 1920.0, 1080.0);
        canvas.add_element(make_text("title", "Main Menu"));
        canvas.add_element(UICanvasElement {
            id: "btn_start".to_string(),
            element_type: CanvasElementType::Button {
                label: "Start".to_string(),
                on_click: "on_start".to_string(),
            },
            anchor: Anchor::Center,
            position: [0.0, 50.0],
            size: [200.0, 40.0],
            visible: true,
            children: vec![],
        });

        let json = canvas.save_json().unwrap();
        let loaded = UICanvas::load_json(&json).unwrap();
        assert_eq!(loaded.name, "Menu");
        assert_eq!(loaded.element_count(), 2);
        assert!(loaded.find_element("btn_start").is_some());
    }

    #[test]
    fn test_render_produces_pixels() {
        let mut canvas = UICanvas::new("Test", 100.0, 100.0);
        canvas.add_element(UICanvasElement {
            id: "panel".to_string(),
            element_type: CanvasElementType::Panel {
                color: [255, 0, 0, 255],
                border_radius: 0.0,
            },
            anchor: Anchor::TopLeft,
            position: [0.0, 0.0],
            size: [50.0, 50.0],
            visible: true,
            children: vec![],
        });

        let mut pixels = vec![[0u8; 4]; 100 * 100];
        canvas.render(&mut pixels, 100, 100);

        // Top-left pixel should be red
        assert_eq!(pixels[0], [255, 0, 0, 255]);
        // Pixel at (60, 60) should be untouched (black)
        assert_eq!(pixels[60 * 100 + 60], [0, 0, 0, 0]);
    }
}
