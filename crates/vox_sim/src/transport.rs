use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportType {
    Bus,
    Tram,
    Metro,
    Rail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportStop {
    pub id: u32,
    pub name: String,
    pub position: [f32; 3],
    pub transport_type: TransportType,
    pub passengers_per_hour: u32,
}

impl TransportStop {
    pub fn new(id: u32, name: impl Into<String>, position: [f32; 3], transport_type: TransportType) -> Self {
        Self {
            id,
            name: name.into(),
            position,
            transport_type,
            passengers_per_hour: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportRoute {
    pub id: u32,
    pub name: String,
    pub transport_type: TransportType,
    pub stops: Vec<u32>,
    pub frequency_minutes: f32,
    pub capacity_per_vehicle: u32,
    pub active: bool,
}

impl TransportRoute {
    pub fn new(
        id: u32,
        name: impl Into<String>,
        transport_type: TransportType,
        frequency_minutes: f32,
        capacity_per_vehicle: u32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            transport_type,
            stops: Vec::new(),
            frequency_minutes,
            capacity_per_vehicle,
            active: true,
        }
    }

    pub fn add_stop(&mut self, stop_id: u32) {
        self.stops.push(stop_id);
    }

    pub fn stop_count(&self) -> usize {
        self.stops.len()
    }
}
