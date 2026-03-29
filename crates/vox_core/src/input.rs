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

/// Save key bindings to a TOML file.
pub fn save_bindings(bindings: &KeyBindings, path: &std::path::Path) -> Result<(), String> {
    let s = toml::to_string_pretty(bindings).map_err(|e| e.to_string())?;
    std::fs::write(path, s).map_err(|e| e.to_string())
}

/// Load key bindings from a TOML file.
/// Returns `KeyBindings::default()` if the file is missing or malformed.
pub fn load_bindings(path: &std::path::Path) -> KeyBindings {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn press_and_release() {
        let mut state = InputState::default();
        let key = InputSource::Key(42);
        assert!(!state.is_pressed(key));

        state.press(key);
        assert!(state.is_pressed(key));
        assert!(state.was_just_pressed(key));

        state.end_frame();
        assert!(state.is_pressed(key));
        assert!(!state.was_just_pressed(key));

        state.release(key);
        assert!(!state.is_pressed(key));
        assert!(state.was_just_released(key));
    }

    #[test]
    fn keybinding_lookup() {
        let bindings = KeyBindings::default();
        let mut state = InputState::default();
        // Undo is bound to Key(29)
        state.press(InputSource::Key(29));
        assert!(bindings.is_action_active(GameAction::Undo, &state));
        assert!(bindings.was_action_triggered(GameAction::Undo, &state));
    }

    #[test]
    fn input_state_just_pressed_clears_after_end_frame() {
        let mut state = InputState::default();
        state.press(InputSource::Key(42));
        assert!(state.was_just_pressed(InputSource::Key(42)));
        state.end_frame();
        assert!(!state.was_just_pressed(InputSource::Key(42)));
        assert!(state.is_pressed(InputSource::Key(42)));
    }

    #[test]
    fn unbound_action_is_inactive() {
        let bindings = KeyBindings::default();
        let state = InputState::default();
        // SpeedVeryFast has no default binding
        assert!(!bindings.is_action_active(GameAction::SpeedVeryFast, &state));
    }
}

#[cfg(test)]
mod keybinding_persist_tests {
    use super::*;

    #[test]
    fn key_bindings_roundtrip_toml() {
        let mut bindings = KeyBindings::default();
        bindings.rebind(GameAction::CameraZoomIn, vec![InputSource::Key(200)]);

        let path = std::env::temp_dir().join("ochroma_test_keybindings.toml");
        save_bindings(&bindings, &path).expect("save should succeed");

        let loaded = load_bindings(&path);
        let sources = loaded.bindings
            .get(&GameAction::CameraZoomIn)
            .expect("CameraZoomIn should be present");
        assert_eq!(sources[0], InputSource::Key(200));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_bindings_returns_default_on_missing_file() {
        let loaded = load_bindings(std::path::Path::new(
            "/tmp/does_not_exist_ochroma_keys_xyzzy.toml",
        ));
        assert!(
            !loaded.bindings.is_empty(),
            "default bindings should be non-empty"
        );
    }

    #[test]
    fn load_bindings_ignores_malformed_toml() {
        let path = std::env::temp_dir().join("ochroma_test_bad_keys.toml");
        std::fs::write(&path, "this is not valid toml ][[[").unwrap();
        let loaded = load_bindings(&path);
        drop(loaded);
        let _ = std::fs::remove_file(&path);
    }
}
