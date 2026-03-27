use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisasterType {
    Fire,
    Flood,
    Earthquake,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveDisaster {
    pub id: u32,
    pub disaster_type: DisasterType,
    pub position: [f32; 2],
    pub radius: f32,
    pub intensity: f32,          // 0.0-1.0
    pub time_remaining: f32,     // seconds until resolved
    pub responding_services: u32, // number of responding units
}

pub struct DisasterManager {
    pub active: Vec<ActiveDisaster>,
    next_id: u32,
    /// Random disaster probability per tick (0.0 = never, 1.0 = every tick).
    pub disaster_probability: f32,
}

impl DisasterManager {
    pub fn new() -> Self {
        Self {
            active: Vec::new(),
            next_id: 0,
            disaster_probability: 0.001,
        }
    }

    /// Trigger a disaster at a position.
    pub fn trigger(
        &mut self,
        disaster_type: DisasterType,
        position: [f32; 2],
        intensity: f32,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let radius = intensity * 100.0; // bigger intensity = bigger area
        let duration = intensity * 300.0; // bigger intensity = longer
        self.active.push(ActiveDisaster {
            id,
            disaster_type,
            position,
            radius,
            intensity,
            time_remaining: duration,
            responding_services: 0,
        });
        id
    }

    /// Assign a service unit to respond to a disaster.
    pub fn respond(&mut self, disaster_id: u32) {
        if let Some(d) = self.active.iter_mut().find(|d| d.id == disaster_id) {
            d.responding_services += 1;
            // Each responder reduces duration
            d.time_remaining *= 0.8;
        }
    }

    /// Tick: advance disasters, resolve completed ones.
    pub fn tick(&mut self, dt: f32) {
        for disaster in &mut self.active {
            disaster.time_remaining -= dt;
            // Responders reduce intensity
            if disaster.responding_services > 0 {
                disaster.intensity *=
                    1.0 - (0.01 * disaster.responding_services as f32 * dt);
                disaster.intensity = disaster.intensity.max(0.0);
            }
        }
        self.active
            .retain(|d| d.time_remaining > 0.0 && d.intensity > 0.01);
    }

    /// Check if a position is affected by any active disaster.
    pub fn is_affected(&self, position: [f32; 2]) -> Option<&ActiveDisaster> {
        self.active.iter().find(|d| {
            let dx = d.position[0] - position[0];
            let dz = d.position[1] - position[1];
            (dx * dx + dz * dz).sqrt() <= d.radius
        })
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}
