use vox_core::spectral::SpectralBands;

/// Weather conditions that affect the spectral power distribution of reflected light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeatherState {
    Clear,
    Overcast,
    LightRain,
    HeavyRain,
    Fog,
    Snow,
}

/// Apply a weather-dependent multiplicative shift to a spectral power distribution.
///
/// Each weather state scales different wavelength bands to simulate the effect of
/// atmospheric scattering, absorption, and reflectance changes.
pub fn apply_weather_shift(base: &SpectralBands, weather: WeatherState) -> SpectralBands {
    // Wavelength band indices: 380, 420, 460, 500, 540, 580, 620, 660 nm
    let factors: [f32; 8] = match weather {
        WeatherState::Clear => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        // Overcast: Rayleigh scattering boosts short wavelengths, mid/long attenuated
        WeatherState::Overcast => [0.85, 0.88, 0.90, 0.92, 0.93, 0.92, 0.90, 0.88],
        // Light rain: blue-shift from water droplet scattering, slight overall attenuation
        WeatherState::LightRain => [0.75, 0.80, 0.88, 0.90, 0.88, 0.85, 0.82, 0.78],
        // Heavy rain: strong attenuation across all bands, more pronounced in long wavelengths
        WeatherState::HeavyRain => [0.55, 0.58, 0.62, 0.64, 0.63, 0.60, 0.57, 0.54],
        // Fog: uniform scattering of short wavelengths, long wavelengths penetrate better
        WeatherState::Fog => [0.60, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88],
        // Snow: high reflectance boost, slight blue shift from ice crystal scattering
        WeatherState::Snow => [1.15, 1.18, 1.20, 1.18, 1.15, 1.12, 1.10, 1.08],
    };

    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = base.0[i] * factors[i];
    }
    SpectralBands(result)
}

/// Linearly interpolate between a fresh and worn SPD by a wear factor in [0.0, 1.0].
///
/// - `wear = 0.0` returns the fresh SPD unchanged.
/// - `wear = 1.0` returns the worn SPD unchanged.
/// - Values in between interpolate linearly band-by-band.
pub fn apply_wear_shift(fresh: &SpectralBands, worn: &SpectralBands, wear: f32) -> SpectralBands {
    let t = wear.clamp(0.0, 1.0);
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = fresh.0[i] * (1.0 - t) + worn.0[i] * t;
    }
    SpectralBands(result)
}

/// Return (D65_weight, D50_weight, A_weight) for the time of day.
///
/// - Midday (12h): pure D65 (daylight 6500K)
/// - Late afternoon (16h): blend toward D50 (5000K warm daylight)
/// - Evening/night (20h–4h): dominated by illuminant A (2856K incandescent)
/// - Dawn/dusk transitions blend between states
///
/// Weights sum to 1.0.
pub fn time_of_day_illuminant_blend(hour: f32) -> (f32, f32, f32) {
    // Normalise hour to [0, 24)
    let h = hour.rem_euclid(24.0);

    // Key time points:
    //   0–5:   night   -> illuminant A dominant
    //   5–7:   dawn    -> transition A -> D65
    //   7–15:  day     -> D65 dominant
    //   15–18: dusk    -> transition D65 -> D50 -> A
    //   18–21: evening -> D50 -> A
    //   21–24: night   -> A dominant

    let (d65, d50, a) = if (7.0..15.0).contains(&h) {
        // Full day: pure D65
        (1.0f32, 0.0f32, 0.0f32)
    } else if (15.0..17.0).contains(&h) {
        // Late afternoon: blend D65 -> D50
        let t = (h - 15.0) / 2.0; // 0..1
        (1.0 - t, t, 0.0)
    } else if (17.0..19.0).contains(&h) {
        // Dusk: blend D50 -> A
        let t = (h - 17.0) / 2.0; // 0..1
        (0.0, 1.0 - t, t)
    } else if (19.0..21.0).contains(&h) {
        // Evening: mostly A, small D50 contribution fading out
        let t = (h - 19.0) / 2.0;
        (0.0, (1.0 - t) * 0.2, 1.0 - (1.0 - t) * 0.2)
    } else if !(5.0..21.0).contains(&h) {
        // Night: pure A
        (0.0, 0.0, 1.0)
    } else if (5.0..7.0).contains(&h) {
        // Dawn: blend A -> D65
        let t = (h - 5.0) / 2.0; // 0..1
        (t, 0.0, 1.0 - t)
    } else {
        (1.0, 0.0, 0.0)
    };

    (d65, d50, a)
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn illuminant_blend_weights_sum_to_one() {
        for h_tenth in 0..240 {
            let h = h_tenth as f32 / 10.0;
            let (d65, d50, a) = time_of_day_illuminant_blend(h);
            let sum = d65 + d50 + a;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Weights don't sum to 1 at hour {h}: d65={d65}, d50={d50}, a={a}, sum={sum}"
            );
        }
    }
}
