use vox_sim::citizen::CitizenManager;
use vox_sim::economy::CityBudget;
use vox_sim::zoning::ZoningManager;
use vox_sim::services::ServiceManager;
use vox_sim::traffic::TrafficNetwork;
use vox_sim::agent::AgentManager;
use bevy_ecs::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameSpeed {
    Paused,
    Normal,    // 1x
    Fast,      // 2x
    VeryFast,  // 4x
}

impl GameSpeed {
    pub fn multiplier(&self) -> f32 {
        match self {
            Self::Paused => 0.0,
            Self::Normal => 1.0,
            Self::Fast => 2.0,
            Self::VeryFast => 4.0,
        }
    }
}

/// Master game simulation state.
#[derive(Resource)]
pub struct SimulationState {
    pub citizens: CitizenManager,
    pub budget: CityBudget,
    pub zoning: ZoningManager,
    pub services: ServiceManager,
    pub traffic: TrafficNetwork,
    pub agents: AgentManager,
    pub game_speed: GameSpeed,
    pub game_time_hours: f64, // total hours elapsed
    pub tick_accumulator: f32, // fractional tick accumulator
}

impl SimulationState {
    pub fn new() -> Self {
        let mut citizens = CitizenManager::new();
        // Start with 10 citizens
        for i in 0..10 {
            citizens.spawn(20.0 + i as f32 * 3.0, None);
        }

        Self {
            citizens,
            budget: CityBudget::default(),
            zoning: ZoningManager::new(),
            services: ServiceManager::new(),
            traffic: TrafficNetwork::new(),
            agents: AgentManager::new(),
            game_speed: GameSpeed::Normal,
            game_time_hours: 8.0, // start at 8am
            tick_accumulator: 0.0,
        }
    }

    /// Advance simulation by real-time dt seconds.
    /// Ticks at 10Hz game time (one tick = 0.1 game hours at 1x speed).
    pub fn tick(&mut self, dt_real: f32) {
        let game_dt = dt_real * self.game_speed.multiplier();
        if game_dt <= 0.0 { return; }

        self.tick_accumulator += game_dt;
        let tick_interval = 0.1; // seconds of real time per sim tick

        while self.tick_accumulator >= tick_interval {
            self.tick_accumulator -= tick_interval;
            let dt_game_hours = tick_interval / 3600.0 * 100.0; // 100x time compression

            self.game_time_hours += dt_game_hours as f64;

            // Citizen lifecycle (age in years, compress heavily)
            self.citizens.tick(dt_game_hours / 8760.0); // hours → years

            // Economy: collect taxes based on citizen count
            let citizen_count = self.citizens.count() as u32;
            let commercial = self.zoning.plots.iter()
                .filter(|p| matches!(p.zone_type, vox_sim::zoning::ZoneType::CommercialLocal | vox_sim::zoning::ZoneType::CommercialRegional))
                .count() as u32;
            let industrial = self.zoning.plots.iter()
                .filter(|p| matches!(p.zone_type, vox_sim::zoning::ZoneType::IndustrialLight | vox_sim::zoning::ZoneType::IndustrialHeavy))
                .count() as u32;
            self.budget.tick(citizen_count, commercial, industrial);

            // Traffic
            self.traffic.tick(tick_interval);

            // Agents
            self.agents.tick(tick_interval);

            // Update zoning demand
            self.zoning.update_demand(citizen_count);
        }
    }

    /// Get current time-of-day as hour (0.0-24.0)
    pub fn time_of_day(&self) -> f32 {
        (self.game_time_hours % 24.0) as f32
    }

    /// Get current day count
    pub fn day(&self) -> u32 {
        (self.game_time_hours / 24.0) as u32
    }
}
