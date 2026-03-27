//! Game UI framework — HUD elements, menus, and bitmap font rendering.
//!
//! Scripts populate a [`GameUI`] resource with [`UIElement`]s; the renderer
//! calls [`GameUI::render_to_pixels`] each frame to stamp them into the
//! framebuffer using a 5×7 bitmap font.

use bevy_ecs::prelude::*;

// ---------------------------------------------------------------------------
// 5×7 bitmap font
// ---------------------------------------------------------------------------

const CHAR_W: u32 = 5;
const CHAR_H: u32 = 7;
/// Horizontal stride: character width + 1 px gap.
const CHAR_STRIDE: u32 = 6;

fn char_bitmap(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        'A' | 'a' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' | 'b' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' | 'c' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' | 'd' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' | 'e' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' | 'f' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' | 'g' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' | 'h' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' | 'i' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' | 'j' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' | 'k' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' | 'l' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' | 'm' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' | 'n' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' | 'o' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' | 'p' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' | 'q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' | 'r' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' | 's' => [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
        'T' | 't' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' | 'u' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' | 'v' => [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
        'W' | 'w' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' | 'x' => [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001],
        'Y' | 'y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' | 'z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '"' => [0b01010, 0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000],
        '\'' => [0b00100, 0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000],
        ' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111], // box
    }
}

/// Burn a single line of text into a pixel buffer at (`x`, `y`) with the
/// given `scale` (1 = native 5×7, 2 = 10×14, etc.).
fn burn_text(
    pixels: &mut [[u8; 4]],
    width: u32,
    x: u32,
    y: u32,
    text: &str,
    color: [u8; 3],
    scale: u32,
) {
    let stride = CHAR_STRIDE * scale;
    for (ci, ch) in text.chars().enumerate() {
        let bitmap = char_bitmap(ch);
        let base_x = x + ci as u32 * stride;
        for (row, &bits) in bitmap.iter().enumerate() {
            for col in 0..CHAR_W {
                if bits & (1 << (4 - col)) != 0 {
                    // Fill a `scale × scale` block for this pixel.
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = base_x + col * scale + sx;
                            let py = y + row as u32 * scale + sy;
                            if px < width {
                                let idx = (py * width + px) as usize;
                                if idx < pixels.len() {
                                    pixels[idx] = [color[0], color[1], color[2], 255];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Width in pixels of a rendered string at the given scale.
fn text_pixel_width(text: &str, scale: u32) -> u32 {
    let len = text.len() as u32;
    if len == 0 {
        return 0;
    }
    len * CHAR_STRIDE * scale - scale // last char has no trailing gap
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Screen-relative anchor for a UI element.
#[derive(Debug, Clone, PartialEq)]
pub enum UIPosition {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Custom { x: u32, y: u32 },
}

impl Default for UIPosition {
    fn default() -> Self {
        Self::TopLeft
    }
}

/// Font scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UISize {
    Small,
    Normal,
    Large,
}

impl Default for UISize {
    fn default() -> Self {
        Self::Normal
    }
}

impl UISize {
    fn scale(self) -> u32 {
        match self {
            Self::Small => 1,
            Self::Normal => 2,
            Self::Large => 3,
        }
    }
}

/// A single HUD text element.
#[derive(Debug, Clone)]
pub struct UIElement {
    pub id: String,
    pub text: String,
    pub position: UIPosition,
    pub color: [u8; 3],
    pub visible: bool,
    pub size: UISize,
}

impl UIElement {
    pub fn new(id: impl Into<String>, text: impl Into<String>, position: UIPosition) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            position,
            color: [255, 255, 255],
            visible: true,
            size: UISize::Normal,
        }
    }
}

/// Top-level game state (drives menu rendering).
#[derive(Debug, Clone, PartialEq)]
pub enum GameState {
    MainMenu,
    Playing,
    Paused,
    GameOver { message: String },
}

impl Default for GameState {
    fn default() -> Self {
        Self::MainMenu
    }
}

// ---------------------------------------------------------------------------
// GameUI resource
// ---------------------------------------------------------------------------

/// Bevy resource that scripts populate with HUD elements.
#[derive(Resource, Default)]
pub struct GameUI {
    pub elements: Vec<UIElement>,
    pub game_state: GameState,
    pub menu_selection: usize,
}

impl GameUI {
    /// Update the text of an existing element (identified by `id`).
    /// Returns `true` if the element was found.
    pub fn set_text(&mut self, id: &str, text: &str) -> bool {
        if let Some(el) = self.elements.iter_mut().find(|e| e.id == id) {
            el.text = text.to_owned();
            true
        } else {
            false
        }
    }

    /// Set visibility of an element by `id`.
    pub fn set_visible(&mut self, id: &str, visible: bool) -> bool {
        if let Some(el) = self.elements.iter_mut().find(|e| e.id == id) {
            el.visible = visible;
            true
        } else {
            false
        }
    }

    /// Add a new UI element (or replace one with the same id).
    pub fn add_element(&mut self, element: UIElement) {
        // Remove existing with same id, then push.
        self.elements.retain(|e| e.id != element.id);
        self.elements.push(element);
    }

    /// Remove an element by id. Returns `true` if it existed.
    pub fn remove_element(&mut self, id: &str) -> bool {
        let before = self.elements.len();
        self.elements.retain(|e| e.id != id);
        self.elements.len() < before
    }

    /// Render all visible elements (and the active menu overlay) into a
    /// pixel buffer of dimensions `width × height`.  Each entry in `pixels`
    /// is `[R, G, B, A]`.
    pub fn render_to_pixels(&self, pixels: &mut [[u8; 4]], width: u32, height: u32) {
        // 1. Menu overlay (if not Playing).
        self.render_menu(pixels, width, height);

        // 2. HUD elements (always rendered — game can hide them via visible flag).
        for el in &self.elements {
            if !el.visible || el.text.is_empty() {
                continue;
            }
            let scale = el.size.scale();
            let tw = text_pixel_width(&el.text, scale);
            let th = CHAR_H * scale;
            let margin: u32 = 10;

            let (x, y) = match &el.position {
                UIPosition::TopLeft => (margin, margin),
                UIPosition::TopCenter => (width.saturating_sub(tw) / 2, margin),
                UIPosition::TopRight => (width.saturating_sub(tw + margin), margin),
                UIPosition::CenterLeft => (margin, height.saturating_sub(th) / 2),
                UIPosition::Center => (
                    width.saturating_sub(tw) / 2,
                    height.saturating_sub(th) / 2,
                ),
                UIPosition::CenterRight => (
                    width.saturating_sub(tw + margin),
                    height.saturating_sub(th) / 2,
                ),
                UIPosition::BottomLeft => (margin, height.saturating_sub(th + margin)),
                UIPosition::BottomCenter => (
                    width.saturating_sub(tw) / 2,
                    height.saturating_sub(th + margin),
                ),
                UIPosition::BottomRight => (
                    width.saturating_sub(tw + margin),
                    height.saturating_sub(th + margin),
                ),
                UIPosition::Custom { x, y } => (*x, *y),
            };

            burn_text(pixels, width, x, y, &el.text, el.color, scale);
        }
    }

    // ------------------------------------------------------------------
    // Menu rendering helpers
    // ------------------------------------------------------------------

    fn render_menu(&self, pixels: &mut [[u8; 4]], width: u32, height: u32) {
        match &self.game_state {
            GameState::Playing => {}
            GameState::MainMenu => {
                let lines = [
                    ("OCHROMA ENGINE", UISize::Large),
                    ("Press ENTER to Play", UISize::Normal),
                    ("Press ESC to Quit", UISize::Normal),
                ];
                self.render_centered_lines(pixels, width, height, &lines, [255, 255, 255]);
            }
            GameState::Paused => {
                let lines = [
                    ("PAUSED", UISize::Large),
                    ("Press ESC to Resume", UISize::Normal),
                ];
                self.render_centered_lines(pixels, width, height, &lines, [255, 255, 200]);
            }
            GameState::GameOver { message } => {
                // We need to own the string to build the slice; use a small
                // helper closure instead.
                let scale_title = UISize::Large;
                let scale_sub = UISize::Normal;
                let color = [255, 100, 100];

                let total_h = CHAR_H * scale_title.scale()
                    + 12
                    + CHAR_H * scale_sub.scale()
                    + 12
                    + CHAR_H * scale_sub.scale();
                let mut cy = height.saturating_sub(total_h) / 2;

                // title line (the message)
                let tw = text_pixel_width(message, scale_title.scale());
                let cx = width.saturating_sub(tw) / 2;
                burn_text(pixels, width, cx, cy, message, color, scale_title.scale());
                cy += CHAR_H * scale_title.scale() + 12;

                // subtitle
                let sub = "Press ENTER to Restart";
                let tw2 = text_pixel_width(sub, scale_sub.scale());
                let cx2 = width.saturating_sub(tw2) / 2;
                burn_text(pixels, width, cx2, cy, sub, color, scale_sub.scale());
            }
        }
    }

    fn render_centered_lines(
        &self,
        pixels: &mut [[u8; 4]],
        width: u32,
        height: u32,
        lines: &[(&str, UISize)],
        color: [u8; 3],
    ) {
        let line_gap: u32 = 12;
        let total_h: u32 = lines
            .iter()
            .map(|(_, sz)| CHAR_H * sz.scale())
            .sum::<u32>()
            + line_gap * lines.len().saturating_sub(1) as u32;

        let mut cy = height.saturating_sub(total_h) / 2;
        for (text, size) in lines {
            let scale = size.scale();
            let tw = text_pixel_width(text, scale);
            let cx = width.saturating_sub(tw) / 2;
            burn_text(pixels, width, cx, cy, text, color, scale);
            cy += CHAR_H * scale + line_gap;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ui() -> GameUI {
        GameUI::default()
    }

    #[test]
    fn add_element_creates_entry() {
        let mut ui = make_ui();
        ui.add_element(UIElement::new("score", "Score: 0", UIPosition::TopLeft));
        assert_eq!(ui.elements.len(), 1);
        assert_eq!(ui.elements[0].id, "score");
        assert_eq!(ui.elements[0].text, "Score: 0");
    }

    #[test]
    fn set_text_updates_existing() {
        let mut ui = make_ui();
        ui.add_element(UIElement::new("hp", "HP: 100", UIPosition::TopRight));
        assert!(ui.set_text("hp", "HP: 50"));
        assert_eq!(ui.elements[0].text, "HP: 50");
    }

    #[test]
    fn set_text_returns_false_for_missing() {
        let mut ui = make_ui();
        assert!(!ui.set_text("nope", "nothing"));
    }

    #[test]
    fn remove_element_deletes_it() {
        let mut ui = make_ui();
        ui.add_element(UIElement::new("a", "AAA", UIPosition::Center));
        ui.add_element(UIElement::new("b", "BBB", UIPosition::Center));
        assert!(ui.remove_element("a"));
        assert_eq!(ui.elements.len(), 1);
        assert_eq!(ui.elements[0].id, "b");
    }

    #[test]
    fn remove_element_returns_false_for_missing() {
        let mut ui = make_ui();
        assert!(!ui.remove_element("ghost"));
    }

    #[test]
    fn render_to_pixels_produces_non_empty_for_visible() {
        let mut ui = make_ui();
        ui.game_state = GameState::Playing; // no menu overlay
        ui.add_element(UIElement::new("msg", "HI", UIPosition::TopLeft));

        let w = 128u32;
        let h = 64u32;
        let mut buf = vec![[0u8; 4]; (w * h) as usize];
        ui.render_to_pixels(&mut buf, w, h);

        let non_zero = buf.iter().filter(|p| p[3] != 0).count();
        assert!(non_zero > 0, "expected non-zero pixels for visible element");
    }

    #[test]
    fn invisible_element_produces_no_pixels() {
        let mut ui = make_ui();
        ui.game_state = GameState::Playing;
        let mut el = UIElement::new("hidden", "HELLO", UIPosition::TopLeft);
        el.visible = false;
        ui.add_element(el);

        let w = 128u32;
        let h = 64u32;
        let mut buf = vec![[0u8; 4]; (w * h) as usize];
        ui.render_to_pixels(&mut buf, w, h);

        let non_zero = buf.iter().filter(|p| p[3] != 0).count();
        assert_eq!(non_zero, 0, "invisible element should produce no pixels");
    }

    #[test]
    fn game_state_transitions() {
        let mut ui = make_ui();
        assert_eq!(ui.game_state, GameState::MainMenu);

        ui.game_state = GameState::Playing;
        assert_eq!(ui.game_state, GameState::Playing);

        ui.game_state = GameState::Paused;
        assert_eq!(ui.game_state, GameState::Paused);

        ui.game_state = GameState::GameOver {
            message: "YOU WIN!".into(),
        };
        assert!(matches!(ui.game_state, GameState::GameOver { .. }));
    }

    #[test]
    fn menu_rendering_main_menu() {
        let ui = make_ui(); // default = MainMenu
        let w = 320u32;
        let h = 240u32;
        let mut buf = vec![[0u8; 4]; (w * h) as usize];
        ui.render_to_pixels(&mut buf, w, h);

        let non_zero = buf.iter().filter(|p| p[3] != 0).count();
        assert!(
            non_zero > 0,
            "main menu should render text (OCHROMA ENGINE, etc.)"
        );
    }

    #[test]
    fn menu_rendering_paused() {
        let mut ui = make_ui();
        ui.game_state = GameState::Paused;

        let w = 320u32;
        let h = 240u32;
        let mut buf = vec![[0u8; 4]; (w * h) as usize];
        ui.render_to_pixels(&mut buf, w, h);

        let non_zero = buf.iter().filter(|p| p[3] != 0).count();
        assert!(non_zero > 0, "paused menu should render PAUSED text");
    }

    #[test]
    fn menu_rendering_game_over() {
        let mut ui = make_ui();
        ui.game_state = GameState::GameOver {
            message: "YOU WIN!".into(),
        };

        let w = 320u32;
        let h = 240u32;
        let mut buf = vec![[0u8; 4]; (w * h) as usize];
        ui.render_to_pixels(&mut buf, w, h);

        let non_zero = buf.iter().filter(|p| p[3] != 0).count();
        assert!(non_zero > 0, "game over screen should render message");
    }

    #[test]
    fn add_element_replaces_duplicate_id() {
        let mut ui = make_ui();
        ui.add_element(UIElement::new("x", "old", UIPosition::TopLeft));
        ui.add_element(UIElement::new("x", "new", UIPosition::Center));
        assert_eq!(ui.elements.len(), 1);
        assert_eq!(ui.elements[0].text, "new");
    }

    #[test]
    fn text_pixel_width_calculation() {
        // "HI" = 2 chars, scale 2 => (2 * 6 * 2) - 2 = 22
        assert_eq!(text_pixel_width("HI", 2), 22);
        assert_eq!(text_pixel_width("", 2), 0);
        assert_eq!(text_pixel_width("A", 1), 5); // 1*6*1 - 1 = 5
    }
}
