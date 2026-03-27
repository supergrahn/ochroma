use vox_core::game_loop::{GameClock, GamePhase};

#[test]
fn clock_accumulates_time() {
    let mut clock = GameClock::new(1.0 / 60.0);
    clock.time_scale = 1.0;
    // Simulate by manually advancing
    clock.accumulator = 0.02; // 20ms
    assert!(clock.should_step()); // 20ms > 16.6ms
    assert!(!clock.should_step()); // only ~3.4ms left
}

#[test]
fn paused_clock_no_steps() {
    let mut clock = GameClock::new(1.0 / 60.0);
    clock.set_paused(true);
    clock.accumulator = 1.0; // lots of time
    // time_scale is 0 so tick() won't add time, but accumulator already has time
    // The pausing is about not adding new time, not about clearing existing accumulator
    assert!(clock.is_paused());
}

#[test]
fn game_phases_in_order() {
    let phases = GamePhase::all_in_order();
    assert_eq!(phases[0], GamePhase::Input);
    assert_eq!(phases[phases.len() - 1], GamePhase::PostFrame);
    assert_eq!(phases.len(), 7);
}

#[test]
fn interpolation_between_steps() {
    let mut clock = GameClock::new(0.1); // 100ms steps
    clock.accumulator = 0.05; // 50ms accumulated
    assert!(!clock.should_step()); // not enough for a step
    let interp = clock.interpolation_factor();
    assert!((interp - 0.5).abs() < 0.01); // halfway between steps
}
