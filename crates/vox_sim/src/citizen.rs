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
