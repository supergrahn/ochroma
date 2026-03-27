use vox_core::input::*;

#[test]
fn press_and_release() {
    let mut state = InputState::default();
    let key = InputSource::Key(42);
    state.press(key);
    assert!(state.is_pressed(key));
    assert!(state.was_just_pressed(key));
    state.end_frame();
    assert!(state.is_pressed(key));
    assert!(!state.was_just_pressed(key)); // cleared after frame
    state.release(key);
    assert!(!state.is_pressed(key));
    assert!(state.was_just_released(key));
}

#[test]
fn keybinding_action_check() {
    let bindings = KeyBindings::default();
    let mut state = InputState::default();
    state.press(InputSource::Key(29)); // Ctrl+Z
    assert!(bindings.is_action_active(GameAction::Undo, &state));
}

#[test]
fn rebind_key() {
    let mut bindings = KeyBindings::default();
    bindings.rebind(GameAction::Undo, vec![InputSource::Key(100)]);
    let mut state = InputState::default();
    state.press(InputSource::Key(100));
    assert!(bindings.is_action_active(GameAction::Undo, &state));
}

#[test]
fn mouse_delta_clears_each_frame() {
    let mut state = InputState::default();
    state.mouse_dx = 10.0;
    state.mouse_dy = 5.0;
    state.end_frame();
    assert_eq!(state.mouse_dx, 0.0);
    assert_eq!(state.mouse_dy, 0.0);
}
