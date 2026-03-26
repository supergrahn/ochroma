use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceType {
    Fire,
    Police,
    Hospital,
    School,
    University,
    Park,
    Library,
    WasteManagement,
    PowerPlant,
    WaterTreatment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceBuilding {
    pub id: u32,
    pub service_type: ServiceType,
    pub capacity: u32,
    pub current_users: u32,
    pub coverage_radius_m: f32,
    pub operational: bool,
}

impl ServiceBuilding {
    pub fn new(id: u32, service_type: ServiceType, capacity: u32, coverage_radius_m: f32) -> Self {
        Self {
            id,
            service_type,
            capacity,
            current_users: 0,
            coverage_radius_m,
            operational: true,
        }
    }

    pub fn utilisation(&self) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.current_users as f32 / self.capacity as f32
    }

    pub fn is_over_capacity(&self) -> bool {
        self.current_users > self.capacity
    }
}
