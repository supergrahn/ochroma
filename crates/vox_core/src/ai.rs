use glam::Vec3;

/// AI agent state -- attached to entities that move autonomously.
#[derive(Debug, Clone)]
pub struct AIAgent {
    pub state: AIState,
    pub move_speed: f32,
    pub detection_radius: f32,
    pub current_path: Vec<Vec3>,
    pub path_index: usize,
    pub target_entity: Option<u32>,
    pub home_position: Vec3,
    pub behavior: AIBehavior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIState {
    Idle,
    Patrolling,
    Chasing,
    Fleeing,
    ReturningHome,
}

#[derive(Debug, Clone)]
pub enum AIBehavior {
    /// Stay still.
    Stationary,
    /// Walk between waypoints.
    Patrol { waypoints: Vec<Vec3>, current: usize },
    /// Chase target when in range, return home when lost.
    GuardArea { guard_radius: f32 },
    /// Always flee from target.
    Coward,
}

impl AIAgent {
    pub fn new_patrol(waypoints: Vec<Vec3>, speed: f32) -> Self {
        Self {
            state: AIState::Patrolling,
            move_speed: speed,
            detection_radius: 15.0,
            current_path: Vec::new(),
            path_index: 0,
            target_entity: None,
            home_position: waypoints.first().copied().unwrap_or(Vec3::ZERO),
            behavior: AIBehavior::Patrol { waypoints, current: 0 },
        }
    }

    pub fn new_guard(position: Vec3, radius: f32, speed: f32) -> Self {
        Self {
            state: AIState::Idle,
            move_speed: speed,
            detection_radius: radius,
            current_path: Vec::new(),
            path_index: 0,
            target_entity: None,
            home_position: position,
            behavior: AIBehavior::GuardArea { guard_radius: radius },
        }
    }

    /// Tick the AI agent. Returns the desired movement delta for this frame.
    pub fn tick(
        &mut self,
        my_position: Vec3,
        target_position: Option<Vec3>,
        dt: f32,
    ) -> Vec3 {
        let mut move_dir = Vec3::ZERO;

        // Check if target is in detection range
        let target_in_range = target_position
            .map(|tp| tp.distance(my_position) < self.detection_radius)
            .unwrap_or(false);

        match &mut self.behavior {
            AIBehavior::Stationary => {
                self.state = AIState::Idle;
            }
            AIBehavior::Patrol { waypoints, current } => {
                if target_in_range {
                    self.state = AIState::Chasing;
                    if let Some(tp) = target_position {
                        move_dir = (tp - my_position).normalize_or_zero();
                    }
                } else {
                    self.state = AIState::Patrolling;
                    if !waypoints.is_empty() {
                        let target = waypoints[*current];
                        let dist = target.distance(my_position);
                        if dist < 1.0 {
                            *current = (*current + 1) % waypoints.len();
                        }
                        move_dir = (waypoints[*current] - my_position).normalize_or_zero();
                    }
                }
            }
            AIBehavior::GuardArea { guard_radius } => {
                if target_in_range {
                    self.state = AIState::Chasing;
                    if let Some(tp) = target_position {
                        let dist_from_home = my_position.distance(self.home_position);
                        if dist_from_home < *guard_radius {
                            move_dir = (tp - my_position).normalize_or_zero();
                        } else {
                            // Too far from home, return
                            self.state = AIState::ReturningHome;
                            move_dir = (self.home_position - my_position).normalize_or_zero();
                        }
                    }
                } else if my_position.distance(self.home_position) > 1.0 {
                    self.state = AIState::ReturningHome;
                    move_dir = (self.home_position - my_position).normalize_or_zero();
                } else {
                    self.state = AIState::Idle;
                }
            }
            AIBehavior::Coward => {
                if target_in_range {
                    self.state = AIState::Fleeing;
                    if let Some(tp) = target_position {
                        move_dir = (my_position - tp).normalize_or_zero(); // run away
                    }
                } else {
                    self.state = AIState::Idle;
                }
            }
        }

        // Apply speed and delta time
        move_dir * self.move_speed * dt
    }

    pub fn is_chasing(&self) -> bool {
        self.state == AIState::Chasing
    }

    pub fn is_idle(&self) -> bool {
        self.state == AIState::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    #[test]
    fn ai_patrol_moves_toward_first_waypoint() {
        let waypoints = vec![Vec3::new(10.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 0.0)];
        let mut agent = AIAgent::new_patrol(waypoints, 5.0);
        let pos = Vec3::ZERO;
        let delta = agent.tick(pos, None, DT);
        // Should move in positive X direction toward first waypoint
        assert!(delta.x > 0.0, "agent should move toward first waypoint");
        assert_eq!(agent.state, AIState::Patrolling);
    }

    #[test]
    fn ai_patrol_advances_to_next_waypoint_when_close() {
        let waypoints = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let mut agent = AIAgent::new_patrol(waypoints, 5.0);
        // Position very close to first waypoint (within 1.0)
        let pos = Vec3::new(0.5, 0.0, 0.0);
        let delta = agent.tick(pos, None, DT);
        // After advancing, should now aim toward second waypoint (10, 0, 0)
        assert!(delta.x > 0.0, "should move toward second waypoint");
        // Verify the patrol index advanced by checking behavior
        if let AIBehavior::Patrol { current, .. } = &agent.behavior {
            assert_eq!(*current, 1);
        } else {
            panic!("expected Patrol behavior");
        }
    }

    #[test]
    fn ai_patrol_chases_target_in_detection_range() {
        let waypoints = vec![Vec3::new(10.0, 0.0, 0.0)];
        let mut agent = AIAgent::new_patrol(waypoints, 5.0);
        agent.detection_radius = 20.0;
        let pos = Vec3::ZERO;
        let target = Some(Vec3::new(0.0, 0.0, 15.0)); // within 20 units
        let delta = agent.tick(pos, target, DT);
        assert!(agent.is_chasing());
        assert!(delta.z > 0.0, "should move toward target");
    }

    #[test]
    fn ai_patrol_resumes_patrol_when_target_leaves_range() {
        let waypoints = vec![Vec3::new(10.0, 0.0, 0.0)];
        let mut agent = AIAgent::new_patrol(waypoints, 5.0);
        agent.detection_radius = 10.0;

        let pos = Vec3::ZERO;
        // First tick: target in range -> chase
        let target_close = Some(Vec3::new(5.0, 0.0, 0.0));
        agent.tick(pos, target_close, DT);
        assert!(agent.is_chasing());

        // Second tick: target out of range -> patrol
        let target_far = Some(Vec3::new(50.0, 0.0, 0.0));
        agent.tick(pos, target_far, DT);
        assert_eq!(agent.state, AIState::Patrolling);
    }

    #[test]
    fn ai_guard_returns_home_when_too_far() {
        let home = Vec3::ZERO;
        let mut agent = AIAgent::new_guard(home, 10.0, 5.0);
        // Agent is far from home and target is even farther away from home
        let pos = Vec3::new(15.0, 0.0, 0.0); // beyond guard_radius of 10
        let target = Some(Vec3::new(20.0, 0.0, 0.0)); // in detection range
        let delta = agent.tick(pos, target, DT);
        assert_eq!(agent.state, AIState::ReturningHome);
        assert!(delta.x < 0.0, "should move back toward home (negative X)");
    }

    #[test]
    fn ai_coward_flees_from_target() {
        let mut agent = AIAgent {
            state: AIState::Idle,
            move_speed: 5.0,
            detection_radius: 20.0,
            current_path: Vec::new(),
            path_index: 0,
            target_entity: None,
            home_position: Vec3::ZERO,
            behavior: AIBehavior::Coward,
        };
        let pos = Vec3::new(5.0, 0.0, 0.0);
        let target = Some(Vec3::ZERO); // target at origin, agent at x=5
        let delta = agent.tick(pos, target, DT);
        assert_eq!(agent.state, AIState::Fleeing);
        assert!(delta.x > 0.0, "should flee away from target (positive X)");
    }

    #[test]
    fn ai_stationary_agent_does_not_move() {
        let mut agent = AIAgent {
            state: AIState::Idle,
            move_speed: 5.0,
            detection_radius: 20.0,
            current_path: Vec::new(),
            path_index: 0,
            target_entity: None,
            home_position: Vec3::ZERO,
            behavior: AIBehavior::Stationary,
        };
        let pos = Vec3::ZERO;
        let target = Some(Vec3::new(5.0, 0.0, 0.0));
        let delta = agent.tick(pos, target, DT);
        assert_eq!(delta, Vec3::ZERO);
        assert!(agent.is_idle());
    }

    #[test]
    fn ai_detection_radius_out_of_range_no_chase() {
        let waypoints = vec![Vec3::new(10.0, 0.0, 0.0)];
        let mut agent = AIAgent::new_patrol(waypoints, 5.0);
        agent.detection_radius = 5.0;
        let pos = Vec3::ZERO;
        // Target at 10 units away, detection radius is 5
        let target = Some(Vec3::new(10.0, 0.0, 0.0));
        agent.tick(pos, target, DT);
        assert!(!agent.is_chasing(), "should not chase target outside detection radius");
        assert_eq!(agent.state, AIState::Patrolling);
    }

    #[test]
    fn ai_speed_scales_with_dt() {
        let waypoints = vec![Vec3::new(10.0, 0.0, 0.0)];
        let mut agent1 = AIAgent::new_patrol(waypoints.clone(), 5.0);
        let mut agent2 = AIAgent::new_patrol(waypoints, 5.0);
        let pos = Vec3::ZERO;

        let delta_small = agent1.tick(pos, None, 0.01);
        let delta_large = agent2.tick(pos, None, 0.1);

        // delta_large should be ~10x delta_small
        let ratio = delta_large.length() / delta_small.length();
        assert!(
            (ratio - 10.0).abs() < 0.01,
            "movement should scale linearly with dt, got ratio {}",
            ratio
        );
    }
}
