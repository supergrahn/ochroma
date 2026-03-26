use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGraph {
    pub id: Uuid,
    pub street: StreetLayout,
    pub atmosphere: AtmosphereState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreetLayout {
    pub name: String,
    pub width_metres: f32,
    pub length_metres: f32,
    pub buildings: Vec<BuildingSlot>,
    pub props: Vec<PropSlot>,
    pub vegetation: Vec<VegetationSlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingSlot {
    pub id: Uuid,
    pub position: [f32; 3],
    pub style: String,
    pub height_metres: f32,
    pub facade_material: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropSlot {
    pub id: Uuid,
    pub position: [f32; 3],
    pub kind: String,
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegetationSlot {
    pub id: Uuid,
    pub position: [f32; 3],
    pub species: String,
    pub height_metres: f32,
    pub canopy_radius_metres: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmosphereState {
    pub weather: Weather,
    pub season: Season,
    pub time_of_day_hour: f32,
    pub fog_density: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Weather {
    Clear,
    PartlyCloudy,
    Overcast,
    LightRain,
    HeavyRain,
    Fog,
    Snow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}
