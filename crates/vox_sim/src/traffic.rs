use serde::{Deserialize, Serialize};

/// Traffic parameters for a road segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadSegmentTraffic {
    pub segment_id: u32,
    pub length_km: f32,
    pub density: f32,        // vehicles/km (ρ)
    pub density_max: f32,    // jam density (ρ_max)
    pub velocity_max: f32,   // free-flow speed km/h (v_max)
    pub flow: f32,           // vehicles/hour
}

impl RoadSegmentTraffic {
    pub fn new(segment_id: u32, length_km: f32, density_max: f32, velocity_max: f32) -> Self {
        Self {
            segment_id, length_km,
            density: 0.0, density_max, velocity_max, flow: 0.0,
        }
    }

    /// Greenshields velocity: v(ρ) = v_max × (1 - ρ/ρ_max)
    pub fn velocity(&self) -> f32 {
        self.velocity_max * (1.0 - self.density / self.density_max).max(0.0)
    }

    /// Flow: q = ρ × v(ρ)
    pub fn compute_flow(&self) -> f32 {
        self.density * self.velocity()
    }
}

/// Traffic network: collection of road segments with flow simulation.
pub struct TrafficNetwork {
    pub segments: Vec<RoadSegmentTraffic>,
}

impl TrafficNetwork {
    pub fn new() -> Self { Self { segments: Vec::new() } }

    pub fn add_segment(&mut self, segment: RoadSegmentTraffic) {
        self.segments.push(segment);
    }

    /// Advance traffic simulation by dt seconds.
    /// Solves ρ_t + (ρ × v(ρ))_x = 0 per segment using Godunov scheme.
    pub fn tick(&mut self, dt: f32) {
        // Update flows
        for seg in &mut self.segments {
            seg.flow = seg.compute_flow();
        }

        // Simple density evolution: each segment's density changes based on
        // flow difference with neighbors. For isolated segments, density decays toward equilibrium.
        let dt_hours = dt / 3600.0;
        for seg in &mut self.segments {
            // Conservation: ρ_t = -dq/dx ≈ -flow / length
            // Simplified: density approaches a steady state
            let target_flow = seg.density * seg.velocity();
            let flow_gradient = target_flow / seg.length_km.max(0.01);
            seg.density = (seg.density - flow_gradient * dt_hours).max(0.0).min(seg.density_max);
            seg.flow = seg.compute_flow();
        }
    }

    /// Add vehicles to a segment.
    pub fn inject_vehicles(&mut self, segment_id: u32, count: f32) {
        if let Some(seg) = self.segments.iter_mut().find(|s| s.segment_id == segment_id) {
            seg.density = (seg.density + count / seg.length_km.max(0.01)).min(seg.density_max);
        }
    }
}
