use vox_sim::milestones::{CityEra, MilestoneTracker};

#[test]
fn era_from_population() {
    assert_eq!(CityEra::from_population(50), CityEra::Village);
    assert_eq!(CityEra::from_population(500), CityEra::Town);
    assert_eq!(CityEra::from_population(5000), CityEra::City);
    assert_eq!(CityEra::from_population(50000), CityEra::Metropolis);
}

#[test]
fn milestone_awarded_at_threshold() {
    let mut tracker = MilestoneTracker::new();
    tracker.check(100);
    assert_eq!(tracker.achieved_count(), 1);
    let notifs = tracker.take_notifications();
    assert!(!notifs.is_empty());
}

#[test]
fn era_upgrades_with_population() {
    let mut tracker = MilestoneTracker::new();
    assert_eq!(tracker.current_era, CityEra::Village);
    tracker.check(600);
    assert_eq!(tracker.current_era, CityEra::Town);
}

#[test]
fn next_milestone_tracks_progress() {
    let mut tracker = MilestoneTracker::new();
    let next = tracker.next_milestone().unwrap();
    assert_eq!(next.population_required, 100);
    tracker.check(200);
    let next = tracker.next_milestone().unwrap();
    assert_eq!(next.population_required, 500);
}
