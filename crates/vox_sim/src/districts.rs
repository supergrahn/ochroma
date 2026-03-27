use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistrictPolicy {
    pub tax_modifier: f32,        // +/- from base rate
    pub rent_control: bool,
    pub noise_ordinance: bool,
    pub speed_limit_kmh: Option<f32>,
    pub historical_preservation: bool,
}

impl Default for DistrictPolicy {
    fn default() -> Self {
        Self { tax_modifier: 0.0, rent_control: false, noise_ordinance: false, speed_limit_kmh: None, historical_preservation: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct District {
    pub id: u32,
    pub name: String,
    pub bounds: ([f32; 2], [f32; 2]), // min, max corners
    pub policy: DistrictPolicy,
}

pub struct DistrictManager {
    pub districts: Vec<District>,
    next_id: u32,
}

impl DistrictManager {
    pub fn new() -> Self { Self { districts: Vec::new(), next_id: 0 } }

    pub fn create_district(&mut self, name: &str, min: [f32; 2], max: [f32; 2]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.districts.push(District { id, name: name.to_string(), bounds: (min, max), policy: DistrictPolicy::default() });
        id
    }

    pub fn set_policy(&mut self, district_id: u32, policy: DistrictPolicy) {
        if let Some(d) = self.districts.iter_mut().find(|d| d.id == district_id) {
            d.policy = policy;
        }
    }

    pub fn district_at(&self, position: [f32; 2]) -> Option<&District> {
        self.districts.iter().find(|d| {
            position[0] >= d.bounds.0[0] && position[0] <= d.bounds.1[0]
                && position[1] >= d.bounds.0[1] && position[1] <= d.bounds.1[1]
        })
    }

    pub fn tax_modifier_at(&self, position: [f32; 2]) -> f32 {
        self.district_at(position).map(|d| d.policy.tax_modifier).unwrap_or(0.0)
    }
}

impl Default for DistrictManager {
    fn default() -> Self {
        Self::new()
    }
}
