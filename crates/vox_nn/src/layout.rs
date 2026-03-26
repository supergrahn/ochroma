use thiserror::Error;
use uuid::Uuid;

use crate::scene_graph::{
    AtmosphereState, BuildingSlot, PropSlot, SceneGraph, Season, StreetLayout, VegetationSlot,
    Weather,
};

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("prompt was empty")]
    EmptyPrompt,
    #[error("failed to parse layout: {0}")]
    ParseError(String),
    #[error("unsupported layout style: {0}")]
    UnsupportedStyle(String),
}

pub trait LayoutInterpreter {
    fn interpret(&self, prompt: &str) -> Result<SceneGraph, LayoutError>;
}

/// A stub interpreter that always returns a hardcoded Victorian street scene.
pub struct StubLayoutInterpreter;

impl LayoutInterpreter for StubLayoutInterpreter {
    fn interpret(&self, prompt: &str) -> Result<SceneGraph, LayoutError> {
        if prompt.trim().is_empty() {
            return Err(LayoutError::EmptyPrompt);
        }

        let scene = SceneGraph {
            id: Uuid::new_v4(),
            street: StreetLayout {
                name: "Victorian High Street".to_string(),
                width_metres: 12.0,
                length_metres: 80.0,
                buildings: vec![
                    BuildingSlot {
                        id: Uuid::new_v4(),
                        position: [0.0, 0.0, 0.0],
                        style: "Victorian Terrace".to_string(),
                        height_metres: 9.0,
                        facade_material: "red_brick".to_string(),
                    },
                    BuildingSlot {
                        id: Uuid::new_v4(),
                        position: [8.0, 0.0, 0.0],
                        style: "Victorian Shopfront".to_string(),
                        height_metres: 7.5,
                        facade_material: "painted_stucco".to_string(),
                    },
                    BuildingSlot {
                        id: Uuid::new_v4(),
                        position: [-8.0, 0.0, 0.0],
                        style: "Victorian Terrace".to_string(),
                        height_metres: 9.0,
                        facade_material: "red_brick".to_string(),
                    },
                ],
                props: vec![
                    PropSlot {
                        id: Uuid::new_v4(),
                        position: [3.0, 0.0, 2.0],
                        kind: "gas_lamp".to_string(),
                        scale: 1.0,
                    },
                    PropSlot {
                        id: Uuid::new_v4(),
                        position: [-3.0, 0.0, 2.0],
                        kind: "gas_lamp".to_string(),
                        scale: 1.0,
                    },
                    PropSlot {
                        id: Uuid::new_v4(),
                        position: [0.0, 0.0, 5.0],
                        kind: "horse_trough".to_string(),
                        scale: 1.2,
                    },
                ],
                vegetation: vec![
                    VegetationSlot {
                        id: Uuid::new_v4(),
                        position: [5.0, 0.0, 4.0],
                        species: "london_plane".to_string(),
                        height_metres: 6.0,
                        canopy_radius_metres: 2.5,
                    },
                    VegetationSlot {
                        id: Uuid::new_v4(),
                        position: [-5.0, 0.0, 4.0],
                        species: "london_plane".to_string(),
                        height_metres: 5.5,
                        canopy_radius_metres: 2.2,
                    },
                ],
            },
            atmosphere: AtmosphereState {
                weather: Weather::PartlyCloudy,
                season: Season::Autumn,
                time_of_day_hour: 14.5,
                fog_density: 0.05,
            },
        };

        Ok(scene)
    }
}
