use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A game action (what the player wants to do).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameAction {
    CameraOrbit, CameraPan, CameraZoomIn, CameraZoomOut,
    Place, Select, Cancel, Delete,
    RotateCW, RotateCCW,
    Undo, Redo,
    Save, Load,
    Pause, SpeedNormal, SpeedFast, SpeedVeryFast,
    ToggleOverlay, CycleOverlay,
    ZoneMode, RoadMode, ServiceMode, PlaceMode, SelectMode,
}

/// Physical input sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputSource {
    Key(u32), // keycode
    MouseButton(u8),
    MouseScroll,
    GamepadButton(u8),
    GamepadAxis(u8),
}

/// Input state for the current frame.
#[derive(Debug, Default)]
pub struct InputState {
    pub pressed: HashMap<InputSource, bool>,
    pub just_pressed: HashMap<InputSource, bool>,
    pub just_released: HashMap<InputSource, bool>,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub scroll_delta: f32,
}

impl InputState {
    pub fn is_pressed(&self, source: InputSource) -> bool {
        *self.pressed.get(&source).unwrap_or(&false)
    }

    pub fn was_just_pressed(&self, source: InputSource) -> bool {
        *self.just_pressed.get(&source).unwrap_or(&false)
    }

    pub fn was_just_released(&self, source: InputSource) -> bool {
        *self.just_released.get(&source).unwrap_or(&false)
    }

    /// Call at end of frame to clear transient state.
    pub fn end_frame(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll_delta = 0.0;
    }

    pub fn press(&mut self, source: InputSource) {
        if !self.is_pressed(source) {
            self.just_pressed.insert(source, true);
        }
        self.pressed.insert(source, true);
    }

    pub fn release(&mut self, source: InputSource) {
        if self.is_pressed(source) {
            self.just_released.insert(source, true);
        }
        self.pressed.insert(source, false);
    }
}

/// Keybinding map: maps actions to input sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBindings {
    pub bindings: HashMap<GameAction, Vec<InputSource>>,
}

impl Default for KeyBindings {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(GameAction::Undo, vec![InputSource::Key(29)]); // Ctrl+Z (simplified)
        bindings.insert(GameAction::Redo, vec![InputSource::Key(21)]); // Ctrl+Y
        bindings.insert(GameAction::Save, vec![InputSource::Key(31)]); // Ctrl+S
        bindings.insert(GameAction::Place, vec![InputSource::MouseButton(0)]); // Left click
        bindings.insert(GameAction::Select, vec![InputSource::MouseButton(0)]);
        bindings.insert(GameAction::Cancel, vec![InputSource::Key(1)]); // Escape
        bindings.insert(GameAction::CameraZoomIn, vec![InputSource::MouseScroll]);
        Self { bindings }
    }
}

impl KeyBindings {
    pub fn is_action_active(&self, action: GameAction, state: &InputState) -> bool {
        if let Some(sources) = self.bindings.get(&action) {
            sources.iter().any(|s| state.is_pressed(*s))
        } else {
            false
        }
    }

    pub fn was_action_triggered(&self, action: GameAction, state: &InputState) -> bool {
        if let Some(sources) = self.bindings.get(&action) {
            sources.iter().any(|s| state.was_just_pressed(*s))
        } else {
            false
        }
    }

    pub fn rebind(&mut self, action: GameAction, sources: Vec<InputSource>) {
        self.bindings.insert(action, sources);
    }
}
