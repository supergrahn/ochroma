use crate::citizen::CitizenManager;

/// Calculate migration: citizens arrive if city is attractive, leave if not.
pub struct MigrationSystem {
    pub regional_satisfaction: f32, // what other cities offer (0.0-1.0)
    pub migration_cooldown: f32,
}

impl MigrationSystem {
    pub fn new() -> Self {
        Self { regional_satisfaction: 0.5, migration_cooldown: 0.0 }
    }

    /// Returns (arrivals, departures) counts.
    pub fn calculate_migration(
        &mut self,
        citizens: &CitizenManager,
        city_satisfaction: f32,
        available_housing: u32,
        dt: f32,
    ) -> (u32, u32) {
        self.migration_cooldown -= dt;
        if self.migration_cooldown > 0.0 { return (0, 0); }
        self.migration_cooldown = 10.0; // check every 10 game-seconds

        let satisfaction_delta = city_satisfaction - self.regional_satisfaction;

        let arrivals = if satisfaction_delta > 0.1 && available_housing > 0 {
            // Positive: people want to move in
            (satisfaction_delta * 5.0).ceil().min(available_housing as f32) as u32
        } else { 0 };

        let departures = if satisfaction_delta < -0.1 {
            // Negative: people leave
            let unhappy = citizens.count_unhappy(0.3);
            (unhappy as f32 * 0.1).ceil() as u32
        } else { 0 };

        (arrivals, departures)
    }
}

impl Default for MigrationSystem {
    fn default() -> Self {
        Self::new()
    }
}
