/// Conditions that must be met to complete a tutorial step.
#[derive(Debug, Clone, PartialEq)]
pub enum TutorialCondition {
    PlaceRoad,
    ZoneResidential,
    ZoneCommercial,
    PlaceService,
    ReachPopulation(u32),
    OpenBudget,
    WaitSeconds(f32),
}

/// A single step in the tutorial sequence.
#[derive(Debug, Clone)]
pub struct TutorialStep {
    pub instruction: String,
    pub completion_condition: TutorialCondition,
    pub completed: bool,
}

impl TutorialStep {
    pub fn new(instruction: &str, condition: TutorialCondition) -> Self {
        Self {
            instruction: instruction.to_string(),
            completion_condition: condition,
            completed: false,
        }
    }
}

/// Snapshot of game state used to check tutorial conditions.
#[derive(Debug, Clone, Default)]
pub struct GameStateSnapshot {
    pub roads_placed: u32,
    pub residential_zones: u32,
    pub commercial_zones: u32,
    pub services_placed: u32,
    pub population: u32,
    pub budget_opened: bool,
    pub elapsed_seconds: f32,
}

/// Manages the interactive tutorial flow.
pub struct TutorialManager {
    steps: Vec<TutorialStep>,
    current_step: usize,
    active: bool,
    elapsed: f32,
}

impl TutorialManager {
    /// Create a new tutorial manager with the default 8-step tutorial.
    pub fn new() -> Self {
        let steps = vec![
            TutorialStep::new(
                "Welcome! Place your first road to get started.",
                TutorialCondition::PlaceRoad,
            ),
            TutorialStep::new(
                "Great! Now zone some residential area along the road.",
                TutorialCondition::ZoneResidential,
            ),
            TutorialStep::new(
                "Citizens need shops. Zone a commercial area.",
                TutorialCondition::ZoneCommercial,
            ),
            TutorialStep::new(
                "Place a fire station or police station to keep citizens safe.",
                TutorialCondition::PlaceService,
            ),
            TutorialStep::new(
                "Open the budget panel to review your finances.",
                TutorialCondition::OpenBudget,
            ),
            TutorialStep::new(
                "Wait a moment while citizens move in...",
                TutorialCondition::WaitSeconds(5.0),
            ),
            TutorialStep::new(
                "Grow your city to 100 citizens!",
                TutorialCondition::ReachPopulation(100),
            ),
            TutorialStep::new(
                "Place another road to expand your city.",
                TutorialCondition::PlaceRoad,
            ),
        ];

        Self {
            steps,
            current_step: 0,
            active: false,
            elapsed: 0.0,
        }
    }

    /// Start the tutorial from the beginning.
    pub fn start(&mut self) {
        self.current_step = 0;
        self.active = true;
        self.elapsed = 0.0;
        for step in &mut self.steps {
            step.completed = false;
        }
    }

    /// Check whether the current step's condition is met and advance if so.
    pub fn check_condition(&mut self, state: &GameStateSnapshot) {
        if !self.active || self.current_step >= self.steps.len() {
            return;
        }

        let met = match &self.steps[self.current_step].completion_condition {
            TutorialCondition::PlaceRoad => state.roads_placed > 0,
            TutorialCondition::ZoneResidential => state.residential_zones > 0,
            TutorialCondition::ZoneCommercial => state.commercial_zones > 0,
            TutorialCondition::PlaceService => state.services_placed > 0,
            TutorialCondition::ReachPopulation(target) => state.population >= *target,
            TutorialCondition::OpenBudget => state.budget_opened,
            TutorialCondition::WaitSeconds(secs) => state.elapsed_seconds >= *secs,
        };

        if met {
            self.steps[self.current_step].completed = true;
            self.current_step += 1;
        }
    }

    /// Get the instruction text for the current step.
    pub fn current_instruction(&self) -> Option<&str> {
        if !self.active {
            return None;
        }
        self.steps
            .get(self.current_step)
            .map(|s| s.instruction.as_str())
    }

    /// Whether all steps have been completed.
    pub fn is_complete(&self) -> bool {
        self.active && self.current_step >= self.steps.len()
    }

    /// Skip the current step.
    pub fn skip(&mut self) {
        if self.active && self.current_step < self.steps.len() {
            self.steps[self.current_step].completed = true;
            self.current_step += 1;
        }
    }

    /// Progress as a percentage (0.0 to 100.0).
    pub fn progress_percent(&self) -> f32 {
        if self.steps.is_empty() {
            return 100.0;
        }
        (self.current_step as f32 / self.steps.len() as f32) * 100.0
    }

    /// The current step index.
    pub fn current_step_index(&self) -> usize {
        self.current_step
    }

    /// Total number of steps.
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }

    /// Whether the tutorial is currently active.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Default for TutorialManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_tutorial() {
        let mut mgr = TutorialManager::new();
        assert!(!mgr.is_active());
        mgr.start();
        assert!(mgr.is_active());
        assert_eq!(mgr.current_step_index(), 0);
        assert_eq!(mgr.total_steps(), 8);
        assert!(!mgr.is_complete());
    }

    #[test]
    fn advance_through_steps() {
        let mut mgr = TutorialManager::new();
        mgr.start();

        // Step 0: PlaceRoad
        assert_eq!(
            mgr.current_instruction().unwrap(),
            "Welcome! Place your first road to get started."
        );

        let state = GameStateSnapshot {
            roads_placed: 1,
            ..Default::default()
        };
        mgr.check_condition(&state);
        assert_eq!(mgr.current_step_index(), 1);

        // Step 1: ZoneResidential
        let state = GameStateSnapshot {
            residential_zones: 1,
            ..Default::default()
        };
        mgr.check_condition(&state);
        assert_eq!(mgr.current_step_index(), 2);
    }

    #[test]
    fn skip_works() {
        let mut mgr = TutorialManager::new();
        mgr.start();
        assert_eq!(mgr.current_step_index(), 0);

        mgr.skip();
        assert_eq!(mgr.current_step_index(), 1);

        mgr.skip();
        assert_eq!(mgr.current_step_index(), 2);
    }

    #[test]
    fn completion_detection() {
        let mut mgr = TutorialManager::new();
        mgr.start();

        // Skip all steps.
        for _ in 0..mgr.total_steps() {
            mgr.skip();
        }

        assert!(mgr.is_complete());
        assert!((mgr.progress_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn progress_percent_tracks() {
        let mut mgr = TutorialManager::new();
        mgr.start();
        assert!((mgr.progress_percent() - 0.0).abs() < 0.01);

        mgr.skip(); // 1/8
        let expected = (1.0 / 8.0) * 100.0;
        assert!((mgr.progress_percent() - expected).abs() < 0.01);
    }

    #[test]
    fn condition_not_met_stays_on_step() {
        let mut mgr = TutorialManager::new();
        mgr.start();

        // Empty state should not advance past PlaceRoad.
        let state = GameStateSnapshot::default();
        mgr.check_condition(&state);
        assert_eq!(mgr.current_step_index(), 0);
    }
}
