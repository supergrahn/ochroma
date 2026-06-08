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
    PrimarySchool,
    SecondarySchool,
    Clinic,
    FireStation,
    PoliceStation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceBuilding {
    pub id: u32,
    pub service_type: ServiceType,
    pub capacity: u32,
    pub current_users: u32,
    pub coverage_radius_m: f32,
    pub operational: bool,
    pub position: [f32; 2],
    pub coverage_radius: f32,
    pub current_load: u32,
    pub operational_cost: f64,
    pub staff_required: u32,
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
            position: [0.0, 0.0],
            coverage_radius: coverage_radius_m,
            current_load: 0,
            operational_cost: 0.0,
            staff_required: 0,
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

pub struct ServiceManager {
    pub buildings: Vec<ServiceBuilding>,
    next_id: u32,
}

impl ServiceManager {
    pub fn new() -> Self {
        Self { buildings: Vec::new(), next_id: 0 }
    }

    pub fn place_service(&mut self, service_type: ServiceType, position: [f32; 2]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let (radius, capacity, cost, staff) = match service_type {
            ServiceType::PrimarySchool => (1000.0, 500, 5000.0, 20),
            ServiceType::SecondarySchool => (2000.0, 1000, 10000.0, 40),
            ServiceType::University => (5000.0, 5000, 50000.0, 200),
            ServiceType::Clinic => (1000.0, 200, 3000.0, 10),
            ServiceType::Hospital => (3000.0, 1000, 30000.0, 100),
            ServiceType::FireStation | ServiceType::Fire => (2000.0, 0, 8000.0, 15),
            ServiceType::PoliceStation | ServiceType::Police => (2000.0, 0, 8000.0, 20),
            ServiceType::School => (1500.0, 600, 6000.0, 25),
            ServiceType::Park => (500.0, 0, 1000.0, 5),
            ServiceType::Library => (1000.0, 300, 4000.0, 10),
            ServiceType::WasteManagement => (5000.0, 0, 15000.0, 30),
            ServiceType::PowerPlant => (10000.0, 0, 100000.0, 50),
            ServiceType::WaterTreatment => (8000.0, 0, 80000.0, 40),
        };
        self.buildings.push(ServiceBuilding {
            id,
            service_type,
            capacity,
            current_users: 0,
            coverage_radius_m: radius,
            operational: true,
            position,
            coverage_radius: radius,
            current_load: 0,
            operational_cost: cost,
            staff_required: staff,
        });
        id
    }

    /// Check if a position is covered by a service type.
    pub fn is_covered(&self, position: [f32; 2], service_type: ServiceType) -> bool {
        self.buildings.iter().any(|b| {
            b.service_type == service_type && {
                let dx = b.position[0] - position[0];
                let dz = b.position[1] - position[1];
                (dx * dx + dz * dz).sqrt() <= b.coverage_radius
            }
        })
    }

    /// Total operational cost of all services.
    pub fn total_cost(&self) -> f64 {
        self.buildings.iter().map(|b| b.operational_cost).sum()
    }
}

impl Default for ServiceManager {
    fn default() -> Self {
        Self::new()
    }
}
