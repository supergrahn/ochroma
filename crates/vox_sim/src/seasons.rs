use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Season { Spring, Summer, Autumn, Winter }

impl Season {
    pub fn from_day(day: u32) -> Self {
        match (day % 360) / 90 {
            0 => Self::Spring,
            1 => Self::Summer,
            2 => Self::Autumn,
            _ => Self::Winter,
        }
    }

    pub fn heating_cost_multiplier(&self) -> f32 {
        match self { Self::Winter => 2.0, Self::Autumn | Self::Spring => 1.2, Self::Summer => 0.5 }
    }

    pub fn crop_growth_rate(&self) -> f32 {
        match self { Self::Spring => 1.5, Self::Summer => 1.0, Self::Autumn => 0.3, Self::Winter => 0.0 }
    }

    pub fn snow_coverage(&self) -> f32 {
        match self { Self::Winter => 0.8, Self::Autumn => 0.1, _ => 0.0 }
    }
}
