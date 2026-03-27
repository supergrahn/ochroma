use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CityEra {
    Village,    // < 500 pop
    Town,       // 500-2000
    City,       // 2000-10000
    Metropolis, // 10000+
}

impl CityEra {
    pub fn from_population(pop: u32) -> Self {
        match pop {
            0..=499 => Self::Village,
            500..=1999 => Self::Town,
            2000..=9999 => Self::City,
            _ => Self::Metropolis,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Village => "Village",
            Self::Town => "Town",
            Self::City => "City",
            Self::Metropolis => "Metropolis",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub name: String,
    pub description: String,
    pub population_required: u32,
    pub achieved: bool,
    pub unlocks: Vec<String>, // what gets unlocked
}

pub struct MilestoneTracker {
    pub milestones: Vec<Milestone>,
    pub current_era: CityEra,
    pub notifications: Vec<String>, // pending UI notifications
}

impl MilestoneTracker {
    pub fn new() -> Self {
        Self {
            milestones: vec![
                Milestone {
                    name: "First Settlement".into(),
                    description: "Reach 100 citizens".into(),
                    population_required: 100,
                    achieved: false,
                    unlocks: vec!["Primary School".into()],
                },
                Milestone {
                    name: "Growing Village".into(),
                    description: "Reach 500 citizens".into(),
                    population_required: 500,
                    achieved: false,
                    unlocks: vec!["Fire Station".into(), "Police Station".into()],
                },
                Milestone {
                    name: "Town Charter".into(),
                    description: "Reach 2,000 citizens".into(),
                    population_required: 2000,
                    achieved: false,
                    unlocks: vec!["Hospital".into(), "Avenue roads".into()],
                },
                Milestone {
                    name: "City Status".into(),
                    description: "Reach 10,000 citizens".into(),
                    population_required: 10000,
                    achieved: false,
                    unlocks: vec!["University".into(), "Highway".into(), "Metro".into()],
                },
                Milestone {
                    name: "Metropolis".into(),
                    description: "Reach 50,000 citizens".into(),
                    population_required: 50000,
                    achieved: false,
                    unlocks: vec!["Airport".into(), "Stadium".into()],
                },
                Milestone {
                    name: "Megacity".into(),
                    description: "Reach 100,000 citizens".into(),
                    population_required: 100000,
                    achieved: false,
                    unlocks: vec!["Everything".into()],
                },
            ],
            current_era: CityEra::Village,
            notifications: Vec::new(),
        }
    }

    /// Check and award milestones based on current population.
    pub fn check(&mut self, population: u32) {
        let new_era = CityEra::from_population(population);
        if new_era > self.current_era {
            self.notifications
                .push(format!("City upgraded to {}!", new_era.label()));
            self.current_era = new_era;
        }

        for milestone in &mut self.milestones {
            if !milestone.achieved && population >= milestone.population_required {
                milestone.achieved = true;
                self.notifications.push(format!(
                    "Milestone: {} — Unlocked: {:?}",
                    milestone.name, milestone.unlocks
                ));
            }
        }
    }

    /// Take pending notifications (drains queue).
    pub fn take_notifications(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notifications)
    }

    pub fn achieved_count(&self) -> usize {
        self.milestones.iter().filter(|m| m.achieved).count()
    }

    pub fn next_milestone(&self) -> Option<&Milestone> {
        self.milestones.iter().find(|m| !m.achieved)
    }
}
