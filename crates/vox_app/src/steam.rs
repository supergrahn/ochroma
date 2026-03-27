use serde::{Deserialize, Serialize};

/// Steam achievement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub unlocked: bool,
}

/// Steam integration interface.
pub struct SteamIntegration {
    pub available: bool,
    pub achievements: Vec<Achievement>,
    pub player_name: String,
    pub rich_presence: String,
}

impl SteamIntegration {
    /// Try to initialize Steam. Returns mock if Steam not available.
    pub fn init() -> Self {
        // In production, this would call Steamworks API
        println!("[ochroma-steam] Steam SDK not linked — running in development mode");
        Self {
            available: false,
            achievements: Self::default_achievements(),
            player_name: "Developer".to_string(),
            rich_presence: String::new(),
        }
    }

    fn default_achievements() -> Vec<Achievement> {
        vec![
            Achievement {
                id: "FIRST_ROAD".into(),
                name: "Pathfinder".into(),
                description: "Build your first road".into(),
                unlocked: false,
            },
            Achievement {
                id: "POP_100".into(),
                name: "Hamlet".into(),
                description: "Reach 100 citizens".into(),
                unlocked: false,
            },
            Achievement {
                id: "POP_1000".into(),
                name: "Village Chief".into(),
                description: "Reach 1,000 citizens".into(),
                unlocked: false,
            },
            Achievement {
                id: "POP_10000".into(),
                name: "Mayor".into(),
                description: "Reach 10,000 citizens".into(),
                unlocked: false,
            },
            Achievement {
                id: "POP_100000".into(),
                name: "Megalopolis".into(),
                description: "Reach 100,000 citizens".into(),
                unlocked: false,
            },
            Achievement {
                id: "FIRST_SERVICE".into(),
                name: "Public Servant".into(),
                description: "Place your first service building".into(),
                unlocked: false,
            },
            Achievement {
                id: "BUDGET_100K".into(),
                name: "Wealthy".into(),
                description: "Accumulate $100,000".into(),
                unlocked: false,
            },
            Achievement {
                id: "SURVIVE_DISASTER".into(),
                name: "Resilient".into(),
                description: "Survive a disaster".into(),
                unlocked: false,
            },
            Achievement {
                id: "FIRST_MOD".into(),
                name: "Modder".into(),
                description: "Load a mod".into(),
                unlocked: false,
            },
            Achievement {
                id: "PLAY_1HOUR".into(),
                name: "Dedicated".into(),
                description: "Play for 1 hour".into(),
                unlocked: false,
            },
        ]
    }

    pub fn unlock_achievement(&mut self, id: &str) {
        if let Some(ach) = self.achievements.iter_mut().find(|a| a.id == id) {
            if !ach.unlocked {
                ach.unlocked = true;
                println!("[ochroma-steam] Achievement unlocked: {}", ach.name);
                // In production: call SteamUserStats::SetAchievement + StoreStats
            }
        }
    }

    pub fn set_rich_presence(&mut self, status: &str) {
        self.rich_presence = status.to_string();
        // In production: call SteamFriends::SetRichPresence
    }

    pub fn unlocked_count(&self) -> usize {
        self.achievements.iter().filter(|a| a.unlocked).count()
    }

    /// Check game state and unlock applicable achievements.
    pub fn check_achievements(
        &mut self,
        population: u32,
        funds: f64,
        roads_built: u32,
        services_placed: u32,
        disasters_survived: u32,
        playtime_hours: f32,
        mods_loaded: u32,
    ) {
        if roads_built > 0 {
            self.unlock_achievement("FIRST_ROAD");
        }
        if population >= 100 {
            self.unlock_achievement("POP_100");
        }
        if population >= 1000 {
            self.unlock_achievement("POP_1000");
        }
        if population >= 10000 {
            self.unlock_achievement("POP_10000");
        }
        if population >= 100000 {
            self.unlock_achievement("POP_100000");
        }
        if services_placed > 0 {
            self.unlock_achievement("FIRST_SERVICE");
        }
        if funds >= 100000.0 {
            self.unlock_achievement("BUDGET_100K");
        }
        if disasters_survived > 0 {
            self.unlock_achievement("SURVIVE_DISASTER");
        }
        if mods_loaded > 0 {
            self.unlock_achievement("FIRST_MOD");
        }
        if playtime_hours >= 1.0 {
            self.unlock_achievement("PLAY_1HOUR");
        }

        // Update rich presence
        self.set_rich_presence(&format!(
            "Building a city — {} citizens, ${:.0}",
            population, funds
        ));
    }
}
