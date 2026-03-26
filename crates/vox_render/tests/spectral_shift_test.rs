use vox_core::spectral::SpectralBands;
use vox_render::spectral_shift::{
    apply_wear_shift, apply_weather_shift, time_of_day_illuminant_blend, WeatherState,
};

fn flat_spd(value: f32) -> SpectralBands {
    SpectralBands([value; 8])
}

fn spds_equal(a: &SpectralBands, b: &SpectralBands) -> bool {
    a.0.iter().zip(b.0.iter()).all(|(x, y)| (x - y).abs() < f32::EPSILON)
}

// --- Weather tests ---

#[test]
fn clear_weather_is_identity() {
    let base = SpectralBands([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let result = apply_weather_shift(&base, WeatherState::Clear);
    assert!(spds_equal(&base, &result), "Clear should not modify SPD");
}

#[test]
fn rain_modifies_spd() {
    let base = flat_spd(1.0);
    let light_rain = apply_weather_shift(&base, WeatherState::LightRain);
    let heavy_rain = apply_weather_shift(&base, WeatherState::HeavyRain);

    // Both should differ from the base
    assert!(
        !spds_equal(&base, &light_rain),
        "LightRain should modify SPD"
    );
    assert!(
        !spds_equal(&base, &heavy_rain),
        "HeavyRain should modify SPD"
    );

    // Heavy rain should attenuate more than light rain
    let light_sum: f32 = light_rain.0.iter().sum();
    let heavy_sum: f32 = heavy_rain.0.iter().sum();
    assert!(
        heavy_sum < light_sum,
        "HeavyRain should attenuate more than LightRain: heavy={heavy_sum}, light={light_sum}"
    );
}

#[test]
fn overcast_attenuates_all_bands() {
    let base = flat_spd(1.0);
    let overcast = apply_weather_shift(&base, WeatherState::Overcast);
    for i in 0..8 {
        assert!(
            overcast.0[i] < base.0[i],
            "Overcast should attenuate band {i}: {} >= {}",
            overcast.0[i],
            base.0[i]
        );
    }
}

#[test]
fn fog_attenuates_short_wavelengths_more() {
    let base = flat_spd(1.0);
    let foggy = apply_weather_shift(&base, WeatherState::Fog);
    // Fog scatters short wavelengths more, so band 0 (380nm) should be attenuated more than band 7 (660nm)
    assert!(
        foggy.0[0] < foggy.0[7],
        "Fog should attenuate short wavelengths more: band0={}, band7={}",
        foggy.0[0],
        foggy.0[7]
    );
}

#[test]
fn snow_boosts_reflectance() {
    let base = flat_spd(1.0);
    let snowy = apply_weather_shift(&base, WeatherState::Snow);
    for i in 0..8 {
        assert!(
            snowy.0[i] > base.0[i],
            "Snow should boost band {i}: {} <= {}",
            snowy.0[i],
            base.0[i]
        );
    }
}

// --- Wear tests ---

#[test]
fn wear_zero_returns_fresh_spd() {
    let fresh = SpectralBands([0.9, 0.8, 0.85, 0.7, 0.75, 0.8, 0.85, 0.9]);
    let worn = SpectralBands([0.4, 0.3, 0.35, 0.3, 0.35, 0.4, 0.45, 0.5]);

    let result = apply_wear_shift(&fresh, &worn, 0.0);
    assert!(spds_equal(&fresh, &result), "wear=0.0 should return fresh SPD");
}

#[test]
fn wear_one_returns_worn_spd() {
    let fresh = SpectralBands([0.9, 0.8, 0.85, 0.7, 0.75, 0.8, 0.85, 0.9]);
    let worn = SpectralBands([0.4, 0.3, 0.35, 0.3, 0.35, 0.4, 0.45, 0.5]);

    let result = apply_wear_shift(&fresh, &worn, 1.0);
    assert!(spds_equal(&worn, &result), "wear=1.0 should return worn SPD");
}

#[test]
fn wear_interpolation_midpoint() {
    let fresh = SpectralBands([1.0; 8]);
    let worn = SpectralBands([0.0; 8]);

    let result = apply_wear_shift(&fresh, &worn, 0.5);
    for i in 0..8 {
        assert!(
            (result.0[i] - 0.5).abs() < f32::EPSILON,
            "wear=0.5 midpoint should be 0.5, got {} at band {i}",
            result.0[i]
        );
    }
}

#[test]
fn wear_interpolation_quarter() {
    let fresh = SpectralBands([1.0; 8]);
    let worn = SpectralBands([0.0; 8]);

    let result_25 = apply_wear_shift(&fresh, &worn, 0.25);
    let result_75 = apply_wear_shift(&fresh, &worn, 0.75);

    for i in 0..8 {
        assert!(
            (result_25.0[i] - 0.75).abs() < f32::EPSILON,
            "wear=0.25 should give 0.75, got {} at band {i}",
            result_25.0[i]
        );
        assert!(
            (result_75.0[i] - 0.25).abs() < f32::EPSILON,
            "wear=0.75 should give 0.25, got {} at band {i}",
            result_75.0[i]
        );
    }
}

#[test]
fn wear_clamped_outside_range() {
    let fresh = SpectralBands([1.0; 8]);
    let worn = SpectralBands([0.5; 8]);

    // Values outside [0, 1] should be clamped
    let result_neg = apply_wear_shift(&fresh, &worn, -0.5);
    let result_over = apply_wear_shift(&fresh, &worn, 1.5);

    assert!(spds_equal(&fresh, &result_neg), "wear < 0 should clamp to fresh");
    assert!(spds_equal(&worn, &result_over), "wear > 1 should clamp to worn");
}

// --- Time of day tests ---

#[test]
fn midday_is_pure_d65() {
    let (d65, d50, a) = time_of_day_illuminant_blend(12.0);
    assert!((d65 - 1.0).abs() < 1e-5, "Midday d65={d65}");
    assert!(d50.abs() < 1e-5, "Midday d50={d50}");
    assert!(a.abs() < 1e-5, "Midday a={a}");
}

#[test]
fn midnight_is_pure_a() {
    let (d65, d50, a) = time_of_day_illuminant_blend(0.0);
    assert!(d65.abs() < 1e-5, "Midnight d65={d65}");
    assert!(d50.abs() < 1e-5, "Midnight d50={d50}");
    assert!((a - 1.0).abs() < 1e-5, "Midnight a={a}");
}

#[test]
fn weights_always_sum_to_one() {
    // Test every 15-minute increment
    for quarter in 0..96 {
        let h = quarter as f32 * 0.25;
        let (d65, d50, a) = time_of_day_illuminant_blend(h);
        let sum = d65 + d50 + a;
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Weights don't sum to 1 at hour {h:.2}: d65={d65:.4}, d50={d50:.4}, a={a:.4}, sum={sum:.6}"
        );
    }
}

#[test]
fn weights_are_non_negative() {
    for i in 0..240 {
        let h = i as f32 / 10.0;
        let (d65, d50, a) = time_of_day_illuminant_blend(h);
        assert!(d65 >= 0.0, "d65 negative at hour {h}: {d65}");
        assert!(d50 >= 0.0, "d50 negative at hour {h}: {d50}");
        assert!(a >= 0.0, "a negative at hour {h}: {a}");
    }
}

#[test]
fn evening_has_high_a_weight() {
    // At 22:00 (10pm), incandescent should dominate
    let (d65, d50, a) = time_of_day_illuminant_blend(22.0);
    assert!(a > 0.8, "Evening should have high A weight, got a={a}");
    let _ = (d65, d50);
}
