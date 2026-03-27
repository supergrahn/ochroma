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
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub building_id: Option<u32>,
    pub district_id: Option<u32>,
}

impl ZonePlot {
    pub fn new(id: u32, zone_type: ZoneType, area_sqm: f32) -> Self {
        Self {
            id,
            zone_type,
            area_sqm,
            developed: false,
            land_value: 100.0,
            position: [0.0, 0.0],
            size: [0.0, 0.0],
            building_id: None,
            district_id: None,
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

pub struct ZoningManager {
    pub plots: Vec<ZonePlot>,
    pub demand: DemandMeter,
    next_id: u32,
}

impl ZoningManager {
    pub fn new() -> Self {
        Self { plots: Vec::new(), demand: DemandMeter::default(), next_id: 0 }
    }

    pub fn zone_plot(&mut self, zone_type: ZoneType, position: [f32; 2], size: [f32; 2]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let area_sqm = size[0] * size[1];
        self.plots.push(ZonePlot {
            id,
            zone_type,
            area_sqm,
            developed: false,
            land_value: 1.0,
            position,
            size,
            building_id: None,
            district_id: None,
        });
        id
    }

    /// Update demand meter based on current zone counts.
    pub fn update_demand(&mut self, citizen_count: u32) {
        let res_count = self
            .plots
            .iter()
            .filter(|p| {
                matches!(
                    p.zone_type,
                    ZoneType::ResidentialLow | ZoneType::ResidentialMed | ZoneType::ResidentialHigh
                )
            })
            .count();
        let com_count = self
            .plots
            .iter()
            .filter(|p| {
                matches!(p.zone_type, ZoneType::CommercialLocal | ZoneType::CommercialRegional)
            })
            .count();
        let ind_count = self
            .plots
            .iter()
            .filter(|p| {
                matches!(p.zone_type, ZoneType::IndustrialLight | ZoneType::IndustrialHeavy)
            })
            .count();

        let citizens_f = citizen_count as f32;
        self.demand.residential =
            (citizens_f / (res_count.max(1) as f32 * 10.0)).min(1.0);
        self.demand.commercial =
            (citizens_f / (com_count.max(1) as f32 * 50.0)).min(1.0);
        self.demand.industrial =
            (citizens_f / (ind_count.max(1) as f32 * 100.0)).min(1.0);
    }

    /// Get plots ready for development (zoned but no building yet).
    pub fn undeveloped_plots(&self) -> Vec<&ZonePlot> {
        self.plots.iter().filter(|p| p.building_id.is_none()).collect()
    }

    /// Mark a plot as developed.
    pub fn develop_plot(&mut self, plot_id: u32, building_id: u32) {
        if let Some(plot) = self.plots.iter_mut().find(|p| p.id == plot_id) {
            plot.building_id = Some(building_id);
            plot.developed = true;
        }
    }

    pub fn plot_count(&self) -> usize {
        self.plots.len()
    }
}

impl Default for ZoningManager {
    fn default() -> Self {
        Self::new()
    }
}
