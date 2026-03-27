use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeatherType {
    Clear,
    Cloudy,
    LightRain,
    HeavyRain,
    Storm,
    Fog,
    Snow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherState {
    pub current: WeatherType,
    pub temperature: f32,        // Celsius
    pub wind_speed: f32,         // m/s
    pub wind_direction: f32,     // radians
    pub humidity: f32,           // 0.0-1.0
    pub precipitation: f32,      // 0.0-1.0
    pub cloud_coverage: f32,     // 0.0-1.0
    pub transition_timer: f32,
    pub transition_duration: f32,
    pub next_weather: WeatherType,
}

impl WeatherState {
    pub fn new(season_day: u32) -> Self {
        let base_temp = match (season_day % 360) / 90 {
            0 => 12.0, // spring
            1 => 22.0, // summer
            2 => 14.0, // autumn
            _ => 3.0,  // winter
        };
        Self {
            current: WeatherType::Clear,
            temperature: base_temp,
            wind_speed: 2.0,
            wind_direction: 0.0,
            humidity: 0.5,
            precipitation: 0.0,
            cloud_coverage: 0.2,
            transition_timer: 0.0,
            transition_duration: 300.0, // 5 minutes transition
            next_weather: WeatherType::Clear,
        }
    }

    pub fn tick(&mut self, dt: f32, seed: u64) {
        self.transition_timer += dt;

        if self.transition_timer >= self.transition_duration {
            self.current = self.next_weather;
            self.transition_timer = 0.0;

            // Pick next weather based on current + random
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_add(self.transition_timer as u64));
            self.next_weather = self.pick_next(&mut rng);
            self.transition_duration = 120.0 + rng.random::<f32>() * 600.0;
        }

        // Update derived values
        match self.current {
            WeatherType::Clear => {
                self.cloud_coverage = (self.cloud_coverage - dt * 0.01).max(0.1);
                self.precipitation = 0.0;
            }
            WeatherType::Cloudy => {
                self.cloud_coverage = (self.cloud_coverage + dt * 0.01).min(0.7);
                self.precipitation = 0.0;
            }
            WeatherType::LightRain => {
                self.cloud_coverage = 0.8;
                self.precipitation = 0.3;
                self.humidity = (self.humidity + dt * 0.001).min(0.95);
            }
            WeatherType::HeavyRain => {
                self.cloud_coverage = 0.95;
                self.precipitation = 0.8;
                self.humidity = 0.95;
            }
            WeatherType::Storm => {
                self.cloud_coverage = 1.0;
                self.precipitation = 1.0;
                self.wind_speed = (self.wind_speed + dt * 0.1).min(25.0);
            }
            WeatherType::Fog => {
                self.cloud_coverage = 0.6;
                self.precipitation = 0.0;
                self.humidity = 0.95;
            }
            WeatherType::Snow => {
                self.cloud_coverage = 0.85;
                self.precipitation = 0.5;
                self.temperature = self.temperature.min(0.0);
            }
        }
    }

    fn pick_next(&self, rng: &mut StdRng) -> WeatherType {
        let roll: f32 = rng.random();
        match self.current {
            WeatherType::Clear => {
                if roll < 0.5 {
                    WeatherType::Clear
                } else if roll < 0.8 {
                    WeatherType::Cloudy
                } else {
                    WeatherType::Fog
                }
            }
            WeatherType::Cloudy => {
                if roll < 0.3 {
                    WeatherType::Clear
                } else if roll < 0.6 {
                    WeatherType::LightRain
                } else if roll < 0.8 {
                    WeatherType::Cloudy
                } else {
                    WeatherType::Fog
                }
            }
            WeatherType::LightRain => {
                if roll < 0.3 {
                    WeatherType::Cloudy
                } else if roll < 0.5 {
                    WeatherType::HeavyRain
                } else {
                    WeatherType::LightRain
                }
            }
            WeatherType::HeavyRain => {
                if roll < 0.3 {
                    WeatherType::LightRain
                } else if roll < 0.5 {
                    WeatherType::Storm
                } else {
                    WeatherType::HeavyRain
                }
            }
            WeatherType::Storm => {
                if roll < 0.5 {
                    WeatherType::HeavyRain
                } else {
                    WeatherType::Cloudy
                }
            }
            WeatherType::Fog => {
                if roll < 0.5 {
                    WeatherType::Clear
                } else {
                    WeatherType::Cloudy
                }
            }
            WeatherType::Snow => {
                if roll < 0.3 {
                    WeatherType::Cloudy
                } else if self.temperature > 2.0 {
                    WeatherType::LightRain
                } else {
                    WeatherType::Snow
                }
            }
        }
    }

    pub fn is_raining(&self) -> bool {
        matches!(
            self.current,
            WeatherType::LightRain | WeatherType::HeavyRain | WeatherType::Storm
        )
    }

    pub fn is_dangerous(&self) -> bool {
        matches!(self.current, WeatherType::Storm)
    }

    pub fn label(&self) -> &'static str {
        match self.current {
            WeatherType::Clear => "Clear",
            WeatherType::Cloudy => "Cloudy",
            WeatherType::LightRain => "Light Rain",
            WeatherType::HeavyRain => "Heavy Rain",
            WeatherType::Storm => "Storm",
            WeatherType::Fog => "Fog",
            WeatherType::Snow => "Snow",
        }
    }
}
