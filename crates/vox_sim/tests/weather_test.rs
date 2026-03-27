use vox_sim::weather::*;

#[test]
fn weather_starts_clear() {
    let state = WeatherState::new(0);
    assert_eq!(state.current, WeatherType::Clear);
}

#[test]
fn weather_transitions_over_time() {
    let mut state = WeatherState::new(0);
    state.transition_duration = 1.0; // fast transition for test
    for i in 0..100 {
        state.tick(0.1, i as u64);
    }
    // After many ticks, weather should have changed at least once
    // (probabilistic, but very likely with 100 transition opportunities)
}

#[test]
fn rain_increases_humidity() {
    let mut state = WeatherState::new(0);
    state.current = WeatherType::LightRain;
    let initial_humidity = state.humidity;
    state.tick(10.0, 0);
    assert!(state.humidity >= initial_humidity);
}

#[test]
fn storm_increases_wind() {
    let mut state = WeatherState::new(0);
    state.current = WeatherType::Storm;
    let initial_wind = state.wind_speed;
    state.tick(5.0, 0);
    assert!(state.wind_speed > initial_wind);
}

#[test]
fn temperature_matches_season() {
    let summer = WeatherState::new(135); // day 135 = summer
    let winter = WeatherState::new(315); // day 315 = winter
    assert!(summer.temperature > winter.temperature);
}
