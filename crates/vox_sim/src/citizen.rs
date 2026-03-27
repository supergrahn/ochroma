use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleStage {
    Child,
    Student,
    Worker,
    Retired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EducationLevel {
    None,
    Primary,
    Secondary,
    University,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Needs {
    pub housing: f32,
    pub food: f32,
    pub health: f32,
    pub safety: f32,
    pub education: f32,
    pub employment: f32,
    pub leisure: f32,
}

impl Needs {
    pub fn satisfaction(&self) -> f32 {
        (self.housing
            + self.food
            + self.health
            + self.safety
            + self.education
            + self.employment
            + self.leisure)
            / 7.0
    }
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            housing: 0.5,
            food: 0.5,
            health: 0.5,
            safety: 0.5,
            education: 0.5,
            employment: 0.5,
            leisure: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citizen {
    pub id: u32,
    pub agent_id: u32,
    pub age: f32,
    pub lifecycle: LifecycleStage,
    pub education: EducationLevel,
    pub employment: Option<u32>,
    pub residence: Option<u32>,
    pub satisfaction: f32,
    pub needs: Needs,
}

impl Citizen {
    pub fn lifecycle_for_age(age: f32) -> LifecycleStage {
        if age < 6.0 {
            LifecycleStage::Child
        } else if age < 18.0 {
            LifecycleStage::Student
        } else if age < 65.0 {
            LifecycleStage::Worker
        } else {
            LifecycleStage::Retired
        }
    }
}

pub struct CitizenManager {
    citizens: Vec<Citizen>,
    next_id: u32,
}

impl CitizenManager {
    pub fn new() -> Self {
        Self { citizens: Vec::new(), next_id: 0 }
    }

    pub fn spawn(&mut self, age: f32, residence: Option<u32>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.citizens.push(Citizen {
            id,
            agent_id: id,
            age,
            lifecycle: Citizen::lifecycle_for_age(age),
            education: if age > 18.0 { EducationLevel::Secondary } else { EducationLevel::None },
            employment: None,
            residence,
            satisfaction: 0.5,
            needs: Needs::default(),
        });
        id
    }

    pub fn count(&self) -> usize {
        self.citizens.len()
    }

    pub fn get(&self, id: u32) -> Option<&Citizen> {
        self.citizens.iter().find(|c| c.id == id)
    }

    /// Advance all citizens by dt game-years.
    pub fn tick(&mut self, dt_years: f32) {
        let mut deaths = Vec::new();
        for citizen in &mut self.citizens {
            citizen.age += dt_years;
            citizen.lifecycle = Citizen::lifecycle_for_age(citizen.age);
            citizen.satisfaction = citizen.needs.satisfaction();

            // Death check (probabilistic after 70)
            if citizen.age > 70.0 {
                let death_prob = (citizen.age - 70.0) * 0.02 * dt_years;
                if death_prob > 0.5 {
                    deaths.push(citizen.id);
                }
            }
        }
        self.citizens.retain(|c| !deaths.contains(&c.id));
    }

    /// Count citizens that would migrate out (satisfaction below threshold for too long).
    pub fn count_unhappy(&self, threshold: f32) -> usize {
        self.citizens.iter().filter(|c| c.satisfaction < threshold).count()
    }

    pub fn all(&self) -> &[Citizen] {
        &self.citizens
    }
}

impl Default for CitizenManager {
    fn default() -> Self {
        Self::new()
    }
}
