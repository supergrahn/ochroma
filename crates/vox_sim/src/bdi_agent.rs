use serde::{Deserialize, Serialize};

use crate::citizen::Citizen;

/// A belief the agent holds about the world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Belief {
    /// Agent knows there is a job at the given building id.
    KnowsJobAt(u32),
    /// Agent knows there is housing at the given building id.
    KnowsHousingAt(u32),
    /// Agent knows there is a service at the given building id.
    KnowsServiceAt(u32),
    /// Agent knows the price level of a resource (0.0 = cheap, 1.0 = expensive).
    KnowsPrice { resource: String, level: f32 },
    /// Agent has a friend (citizen id).
    HasFriend(u32),
    /// Agent believes crime is high in a district.
    HighCrimeArea(u32),
    /// Agent believes an area is desirable.
    DesirableArea(u32),
}

/// A desire the agent wants to fulfil.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Desire {
    FindBetterJob,
    FindHousing,
    StartFamily,
    GetEducated,
    Socialize,
    ImproveHealth,
    IncreaseLeisure,
    ImproveSafety,
}

/// A single step in an agent's plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    GoTo(u32),
    ApplyForJob(u32),
    RentHousing(u32),
    AttendSchool(u32),
    VisitService(u32),
    MeetFriend(u32),
    SearchForHousing,
    SearchForJob,
    Idle,
}

/// An intention is a committed plan — a sequence of actions toward a desire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    pub desire: Desire,
    pub plan: Vec<Action>,
    pub current_step: usize,
}

impl Intention {
    pub fn new(desire: Desire, plan: Vec<Action>) -> Self {
        Self {
            desire,
            plan,
            current_step: 0,
        }
    }

    /// Advance to the next step. Returns true if plan is complete.
    pub fn advance(&mut self) -> bool {
        self.current_step += 1;
        self.current_step >= self.plan.len()
    }

    /// Current action, or None if plan is complete.
    pub fn current_action(&self) -> Option<&Action> {
        self.plan.get(self.current_step)
    }

    pub fn is_complete(&self) -> bool {
        self.current_step >= self.plan.len()
    }
}

/// BDI (Belief-Desire-Intention) agent wrapping a citizen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BdiAgent {
    pub citizen_id: u32,
    pub beliefs: Vec<Belief>,
    pub desires: Vec<Desire>,
    pub intention: Option<Intention>,
    pub satisfaction_threshold: f32,
}

impl BdiAgent {
    pub fn new(citizen_id: u32) -> Self {
        Self {
            citizen_id,
            beliefs: Vec::new(),
            desires: Vec::new(),
            intention: None,
            satisfaction_threshold: 0.4,
        }
    }

    /// Add a belief if not already present.
    pub fn add_belief(&mut self, belief: Belief) {
        if !self.beliefs.contains(&belief) {
            self.beliefs.push(belief);
        }
    }

    /// Remove a belief.
    pub fn remove_belief(&mut self, belief: &Belief) {
        self.beliefs.retain(|b| b != belief);
    }

    /// Generate desires based on the citizen's current state.
    pub fn generate_desires(&mut self, citizen: &Citizen) {
        self.desires.clear();

        // No residence => must find housing
        if citizen.residence.is_none() {
            self.desires.push(Desire::FindHousing);
        }

        // No job and is a worker => find a job
        if citizen.employment.is_none()
            && citizen.lifecycle == crate::citizen::LifecycleStage::Worker
        {
            self.desires.push(Desire::FindBetterJob);
        }

        // Low satisfaction triggers various desires
        if citizen.needs.housing < self.satisfaction_threshold {
            if !self.desires.contains(&Desire::FindHousing) {
                self.desires.push(Desire::FindHousing);
            }
        }

        if citizen.needs.employment < self.satisfaction_threshold {
            if !self.desires.contains(&Desire::FindBetterJob) {
                self.desires.push(Desire::FindBetterJob);
            }
        }

        if citizen.needs.education < self.satisfaction_threshold {
            self.desires.push(Desire::GetEducated);
        }

        if citizen.needs.health < self.satisfaction_threshold {
            self.desires.push(Desire::ImproveHealth);
        }

        if citizen.needs.leisure < self.satisfaction_threshold {
            self.desires.push(Desire::IncreaseLeisure);
        }

        if citizen.needs.safety < self.satisfaction_threshold {
            self.desires.push(Desire::ImproveSafety);
        }

        // Social desire when leisure is moderate but satisfaction is low overall
        if citizen.satisfaction < 0.5 && citizen.needs.leisure >= self.satisfaction_threshold {
            self.desires.push(Desire::Socialize);
        }
    }

    /// Pick the most urgent desire and create a plan.
    /// Priority: housing > job > health > safety > education > leisure > socialize > family.
    pub fn plan_next_action(&mut self, citizen: &Citizen) {
        self.generate_desires(citizen);

        if self.desires.is_empty() {
            self.intention = None;
            return;
        }

        // Priority ordering
        let priority = [
            Desire::FindHousing,
            Desire::FindBetterJob,
            Desire::ImproveHealth,
            Desire::ImproveSafety,
            Desire::GetEducated,
            Desire::IncreaseLeisure,
            Desire::Socialize,
            Desire::StartFamily,
        ];

        let chosen = priority
            .iter()
            .find(|d| self.desires.contains(d))
            .copied()
            .unwrap_or(self.desires[0]);

        let plan = self.build_plan(chosen);
        self.intention = Some(Intention::new(chosen, plan));
    }

    /// Build a concrete plan for the given desire using current beliefs.
    fn build_plan(&self, desire: Desire) -> Vec<Action> {
        match desire {
            Desire::FindHousing => {
                // Check if we know of any housing
                if let Some(building_id) = self.known_housing() {
                    vec![Action::GoTo(building_id), Action::RentHousing(building_id)]
                } else {
                    vec![Action::SearchForHousing]
                }
            }
            Desire::FindBetterJob => {
                if let Some(building_id) = self.known_job() {
                    vec![Action::GoTo(building_id), Action::ApplyForJob(building_id)]
                } else {
                    vec![Action::SearchForJob]
                }
            }
            Desire::GetEducated => {
                if let Some(service_id) = self.known_service() {
                    vec![Action::GoTo(service_id), Action::AttendSchool(service_id)]
                } else {
                    vec![Action::Idle]
                }
            }
            Desire::ImproveHealth => {
                if let Some(service_id) = self.known_service() {
                    vec![Action::GoTo(service_id), Action::VisitService(service_id)]
                } else {
                    vec![Action::Idle]
                }
            }
            Desire::Socialize => {
                if let Some(friend_id) = self.known_friend() {
                    vec![Action::MeetFriend(friend_id)]
                } else {
                    vec![Action::Idle]
                }
            }
            Desire::StartFamily | Desire::IncreaseLeisure | Desire::ImproveSafety => {
                vec![Action::Idle]
            }
        }
    }

    fn known_housing(&self) -> Option<u32> {
        self.beliefs.iter().find_map(|b| match b {
            Belief::KnowsHousingAt(id) => Some(*id),
            _ => None,
        })
    }

    fn known_job(&self) -> Option<u32> {
        self.beliefs.iter().find_map(|b| match b {
            Belief::KnowsJobAt(id) => Some(*id),
            _ => None,
        })
    }

    fn known_service(&self) -> Option<u32> {
        self.beliefs.iter().find_map(|b| match b {
            Belief::KnowsServiceAt(id) => Some(*id),
            _ => None,
        })
    }

    fn known_friend(&self) -> Option<u32> {
        self.beliefs.iter().find_map(|b| match b {
            Belief::HasFriend(id) => Some(*id),
            _ => None,
        })
    }
}
