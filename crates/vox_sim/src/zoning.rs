use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneType {
    ResidentialLow,
    ResidentialMed,
    ResidentialHigh,
    CommercialLocal,
    CommercialRegional,
    IndustrialLight,
    IndustrialHeavy,
    Office,
    MixedUse,
    Agricultural,
    Park,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZonePlot {
    pub id: u32,
    pub zone_type: ZoneType,
    pub area_sqm: f32,
    pub developed: bool,
    pub land_value: f32,
}

impl ZonePlot {
    pub fn new(id: u32, zone_type: ZoneType, area_sqm: f32) -> Self {
        Self {
            id,
            zone_type,
            area_sqm,
            developed: false,
            land_value: 100.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandMeter {
    pub residential: f32,
    pub commercial: f32,
    pub industrial: f32,
    pub office: f32,
}

impl Default for DemandMeter {
    fn default() -> Self {
        Self {
            residential: 0.5,
            commercial: 0.3,
            industrial: 0.2,
            office: 0.2,
        }
    }
}

impl DemandMeter {
    pub fn highest_demand(&self) -> ZoneType {
        let max = self
            .residential
            .max(self.commercial)
            .max(self.industrial)
            .max(self.office);
        if max == self.residential {
            ZoneType::ResidentialMed
        } else if max == self.commercial {
            ZoneType::CommercialLocal
        } else if max == self.industrial {
            ZoneType::IndustrialLight
        } else {
            ZoneType::Office
        }
    }
}
