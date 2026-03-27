use vox_core::spectral::Illuminant;
use vox_render::spectral_shift::{WeatherState, time_of_day_illuminant_blend};

/// Current environmental state affecting rendering.
#[derive(Debug, Clone)]
pub struct EnvironmentState {
    pub weather: WeatherState,
    pub time_of_day: f32, // 0.0-24.0
}

impl Default for EnvironmentState {
    fn default() -> Self {
        Self {
            weather: WeatherState::Clear,
            time_of_day: 12.0,
        }
    }
}

impl EnvironmentState {
    /// Get the blended illuminant for the current time of day.
    pub fn current_illuminant(&self) -> Illuminant {
        let (d65_w, d50_w, a_w) = time_of_day_illuminant_blend(self.time_of_day);
        let d65 = Illuminant::d65();
        let d50 = Illuminant::d50();
        let a = Illuminant::a();

        Illuminant {
            bands: std::array::from_fn(|i| {
                d65.bands[i] * d65_w + d50.bands[i] * d50_w + a.bands[i] * a_w
            }),
        }
    }

    /// Get a descriptive label for UI.
    pub fn time_label(&self) -> &'static str {
        match self.time_of_day as u32 {
            0..=5 => "Night",
            6..=7 => "Dawn",
            8..=16 => "Day",
            17..=18 => "Sunset",
            19..=20 => "Dusk",
            _ => "Night",
        }
    }

    pub fn weather_label(&self) -> &'static str {
        match self.weather {
            WeatherState::Clear => "Clear",
            WeatherState::Overcast => "Overcast",
            WeatherState::LightRain => "Light Rain",
            WeatherState::HeavyRain => "Heavy Rain",
            WeatherState::Fog => "Fog",
            WeatherState::Snow => "Snow",
        }
    }
}
