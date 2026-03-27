use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingType {
    Residential,
    Commercial,
    Industrial,
    Service,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Building {
    pub id: u32,
    pub building_type: BuildingType,
    pub position: [f32; 2],
    pub capacity: u32,
    pub occupants: u32,
    pub operational: bool,
}

pub struct BuildingManager {
    pub buildings: Vec<Building>,
    next_id: u32,
}

impl BuildingManager {
    pub fn new() -> Self {
        Self { buildings: Vec::new(), next_id: 0 }
    }

    pub fn add_building(&mut self, building_type: BuildingType, position: [f32; 2], capacity: u32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.buildings.push(Building {
            id,
            building_type,
            position,
            capacity,
            occupants: 0,
            operational: true,
        });
        id
    }

    pub fn find_nearest_with_vacancy(&self, position: [f32; 2], bt: BuildingType) -> Option<u32> {
        self.buildings
            .iter()
            .filter(|b| b.building_type == bt && b.occupants < b.capacity && b.operational)
            .min_by(|a, b| {
                let da = (a.position[0] - position[0]).powi(2) + (a.position[1] - position[1]).powi(2);
                let db = (b.position[0] - position[0]).powi(2) + (b.position[1] - position[1]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|b| b.id)
    }

    pub fn assign_occupant(&mut self, building_id: u32) -> bool {
        if let Some(b) = self.buildings.iter_mut().find(|b| b.id == building_id) {
            if b.occupants < b.capacity {
                b.occupants += 1;
                return true;
            }
        }
        false
    }

    pub fn total_housing(&self) -> u32 {
        self.buildings
            .iter()
            .filter(|b| b.building_type == BuildingType::Residential)
            .map(|b| b.capacity)
            .sum()
    }

    pub fn total_jobs(&self) -> u32 {
        self.buildings
            .iter()
            .filter(|b| matches!(b.building_type, BuildingType::Commercial | BuildingType::Industrial))
            .map(|b| b.capacity)
            .sum()
    }

    pub fn count(&self) -> usize {
        self.buildings.len()
    }
}

impl Default for BuildingManager {
    fn default() -> Self {
        Self::new()
    }
}
