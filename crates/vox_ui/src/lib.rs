use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    pub background_color: [f32; 4],
    pub foreground_color: [f32; 4],
    pub accent_color: [f32; 4],
    pub font_size: f32,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background_color: [0.1, 0.1, 0.1, 1.0],
            foreground_color: [1.0, 1.0, 1.0, 1.0],
            accent_color: [0.2, 0.6, 1.0, 1.0],
            font_size: 14.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UiNode {
    Panel {
        position: [f32; 2],
        size: [f32; 2],
        color: [f32; 4],
    },
    Text {
        position: [f32; 2],
        content: String,
        font_size: f32,
        color: [f32; 4],
    },
    Button {
        position: [f32; 2],
        size: [f32; 2],
        label: String,
    },
    Slider {
        position: [f32; 2],
        size: [f32; 2],
        value: f32,
        min: f32,
        max: f32,
    },
}

pub struct UiRoot {
    nodes: Vec<UiNode>,
    #[allow(dead_code)]
    theme: Theme,
}

impl UiRoot {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            theme: Theme::default(),
        }
    }

    pub fn add(&mut self, node: UiNode) {
        self.nodes.push(node);
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for UiRoot {
    fn default() -> Self {
        Self::new()
    }
}
