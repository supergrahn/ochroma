use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VehicleType {
    Car,
    Bus,
    Truck,
    EmergencyVehicle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vehicle {
    pub id: u32,
    pub vehicle_type: VehicleType,
    pub position: Vec3,
    pub velocity: Vec3,
    pub speed: f32,
    pub max_speed: f32,
    pub road_segment_id: u32,
    pub progress: f32,    // 0.0-1.0 along current segment
    pub route: Vec<u32>,  // list of road segment IDs to follow
    pub route_index: usize,
    pub destination: Option<Vec3>,
    pub parked: bool,
}

pub struct VehicleManager {
    pub vehicles: Vec<Vehicle>,
    next_id: u32,
    pub max_vehicles: usize,
}

impl VehicleManager {
    pub fn new(max_vehicles: usize) -> Self {
        Self {
            vehicles: Vec::new(),
            next_id: 0,
            max_vehicles,
        }
    }

    pub fn spawn(
        &mut self,
        vehicle_type: VehicleType,
        position: Vec3,
        route: Vec<u32>,
    ) -> Option<u32> {
        if self.vehicles.len() >= self.max_vehicles {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let max_speed = match vehicle_type {
            VehicleType::Car => 13.9,              // 50 km/h
            VehicleType::Bus => 11.1,              // 40 km/h
            VehicleType::Truck => 8.3,             // 30 km/h
            VehicleType::EmergencyVehicle => 22.2, // 80 km/h
        };
        self.vehicles.push(Vehicle {
            id,
            vehicle_type,
            position,
            velocity: Vec3::ZERO,
            speed: 0.0,
            max_speed,
            road_segment_id: route.first().copied().unwrap_or(0),
            progress: 0.0,
            route,
            route_index: 0,
            destination: None,
            parked: false,
        });
        Some(id)
    }

    pub fn tick(&mut self, dt: f32) {
        for vehicle in &mut self.vehicles {
            if vehicle.parked {
                continue;
            }

            // Accelerate toward max speed
            vehicle.speed = (vehicle.speed + dt * 5.0).min(vehicle.max_speed);

            // Advance along route
            vehicle.progress += vehicle.speed * dt * 0.01; // scaled for segment length

            if vehicle.progress >= 1.0 {
                vehicle.progress = 0.0;
                vehicle.route_index += 1;
                if vehicle.route_index >= vehicle.route.len() {
                    // Reached destination — park
                    vehicle.parked = true;
                    vehicle.speed = 0.0;
                } else {
                    vehicle.road_segment_id = vehicle.route[vehicle.route_index];
                }
            }

            // Simple position update (actual road geometry would modify this)
            vehicle.position += vehicle.velocity.normalize_or_zero() * vehicle.speed * dt;
        }

        // Remove parked vehicles after some time (simplified: keep recent ones)
        let cutoff = self.next_id.saturating_sub(100);
        self.vehicles
            .retain(|v| !v.parked || v.id >= cutoff);
    }

    pub fn count(&self) -> usize {
        self.vehicles.len()
    }

    pub fn active_count(&self) -> usize {
        self.vehicles.iter().filter(|v| !v.parked).count()
    }

    /// Get vehicles on a specific road segment.
    pub fn on_segment(&self, segment_id: u32) -> Vec<&Vehicle> {
        self.vehicles
            .iter()
            .filter(|v| v.road_segment_id == segment_id && !v.parked)
            .collect()
    }
}
