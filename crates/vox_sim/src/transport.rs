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
    pub position: [f32; 2],
    pub name: String,
    pub waiting_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportRoute {
    pub id: u32,
    pub transport_type: TransportType,
    pub stops: Vec<TransportStop>,
    pub vehicle_count: u32,
    pub frequency_minutes: f32,
}

impl TransportRoute {
    /// Calculate total route length (sum of distances between consecutive stops).
    pub fn route_length(&self) -> f32 {
        if self.stops.len() < 2 {
            return 0.0;
        }
        self.stops
            .windows(2)
            .map(|w| {
                let dx = w[1].position[0] - w[0].position[0];
                let dz = w[1].position[1] - w[0].position[1];
                (dx * dx + dz * dz).sqrt()
            })
            .sum()
    }

    /// Estimated travel time in minutes for the full route.
    pub fn travel_time_minutes(&self) -> f32 {
        let speed_kmh = match self.transport_type {
            TransportType::Bus => 25.0,
            TransportType::Tram => 30.0,
            TransportType::Metro => 40.0,
            TransportType::Rail => 80.0,
        };
        let length_km = self.route_length() / 1000.0;
        (length_km / speed_kmh) * 60.0
    }

    /// Revenue per hour based on passenger capacity and frequency.
    pub fn hourly_revenue(&self, fare: f32, load_factor: f32) -> f32 {
        let trips_per_hour = 60.0 / self.frequency_minutes.max(1.0);
        let capacity = match self.transport_type {
            TransportType::Bus => 40.0,
            TransportType::Tram => 100.0,
            TransportType::Metro => 200.0,
            TransportType::Rail => 500.0,
        };
        trips_per_hour * capacity * load_factor * fare * self.vehicle_count as f32
    }
}

pub struct TransportManager {
    pub routes: Vec<TransportRoute>,
    next_id: u32,
}

impl TransportManager {
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            next_id: 0,
        }
    }

    pub fn create_route(
        &mut self,
        transport_type: TransportType,
        frequency_minutes: f32,
        vehicle_count: u32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.routes.push(TransportRoute {
            id,
            transport_type,
            stops: Vec::new(),
            vehicle_count,
            frequency_minutes,
        });
        id
    }

    pub fn add_stop(&mut self, route_id: u32, position: [f32; 2], name: &str) {
        if let Some(route) = self.routes.iter_mut().find(|r| r.id == route_id) {
            let stop_id = route.stops.len() as u32;
            route.stops.push(TransportStop {
                id: stop_id,
                position,
                name: name.to_string(),
                waiting_count: 0,
            });
        }
    }

    pub fn total_hourly_revenue(&self, fare: f32, load_factor: f32) -> f32 {
        self.routes
            .iter()
            .map(|r| r.hourly_revenue(fare, load_factor))
            .sum()
    }
}

impl Default for TransportManager {
    fn default() -> Self {
        Self::new()
    }
}
