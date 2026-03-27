/// Advisor messages based on city conditions.
pub struct AdvisorSystem {
    pub messages: Vec<AdvisorMessage>,
    cooldown_ticks: u32,
}

#[derive(Debug, Clone)]
pub struct AdvisorMessage {
    pub category: AdvisorCategory,
    pub title: String,
    pub text: String,
    pub priority: u8, // 1=low, 5=critical
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvisorCategory {
    Budget,
    Population,
    Infrastructure,
    Services,
    Environment,
    Growth,
}

impl AdvisorSystem {
    pub fn new() -> Self {
        Self { messages: Vec::new(), cooldown_ticks: 0 }
    }

    /// Evaluate city state and generate advisor messages.
    pub fn evaluate(
        &mut self,
        population: u32,
        funds: f64,
        unemployment_rate: f32,
        housing_shortage: bool,
        power_deficit: bool,
        water_deficit: bool,
        high_crime: bool,
        high_pollution: bool,
        no_schools: bool,
        no_hospitals: bool,
        traffic_congestion: f32,
    ) {
        self.messages.clear();

        if self.cooldown_ticks > 0 {
            self.cooldown_ticks -= 1;
            return;
        }

        if funds < 0.0 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Budget, priority: 5,
                title: "Budget Crisis!".into(),
                text: "The city is in debt. Raise taxes or cut services immediately.".into(),
            });
        } else if funds < 5000.0 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Budget, priority: 3,
                title: "Low Funds".into(),
                text: "City funds are running low. Consider increasing tax rates.".into(),
            });
        }

        if housing_shortage {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Population, priority: 4,
                title: "Housing Shortage".into(),
                text: "Citizens need homes! Zone more residential areas.".into(),
            });
        }

        if unemployment_rate > 0.2 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Population, priority: 3,
                title: "High Unemployment".into(),
                text: format!("Unemployment at {:.0}%. Zone more commercial or industrial areas.", unemployment_rate * 100.0),
            });
        }

        if power_deficit {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Infrastructure, priority: 4,
                title: "Power Shortage".into(),
                text: "Some buildings don't have power. Build more power plants.".into(),
            });
        }

        if water_deficit {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Infrastructure, priority: 4,
                title: "Water Shortage".into(),
                text: "Water supply is insufficient. Expand the water network.".into(),
            });
        }

        if high_crime {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Services, priority: 3,
                title: "Rising Crime".into(),
                text: "Crime is increasing. Build more police stations.".into(),
            });
        }

        if high_pollution {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Environment, priority: 2,
                title: "Pollution Warning".into(),
                text: "Air quality is declining. Add parks and reduce industrial zones.".into(),
            });
        }

        if no_schools && population > 200 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Services, priority: 3,
                title: "No Schools".into(),
                text: "Children have nowhere to learn. Build a school.".into(),
            });
        }

        if no_hospitals && population > 500 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Services, priority: 3,
                title: "No Healthcare".into(),
                text: "Citizens need healthcare. Build a clinic or hospital.".into(),
            });
        }

        if traffic_congestion > 0.7 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Infrastructure, priority: 2,
                title: "Traffic Congestion".into(),
                text: "Roads are congested. Build wider roads or public transport.".into(),
            });
        }

        if population > 0 && population < 50 {
            self.messages.push(AdvisorMessage {
                category: AdvisorCategory::Growth, priority: 1,
                title: "Getting Started".into(),
                text: "Welcome! Build roads, then zone areas for residential and commercial use.".into(),
            });
        }

        // Sort by priority (highest first)
        self.messages.sort_by(|a, b| b.priority.cmp(&a.priority));

        if !self.messages.is_empty() {
            self.cooldown_ticks = 100; // don't spam
        }
    }

    pub fn top_message(&self) -> Option<&AdvisorMessage> {
        self.messages.first()
    }

    pub fn message_count(&self) -> usize { self.messages.len() }
}
