use vox_sim::citizen::{Citizen, LifecycleStage, Needs};

#[test]
fn lifecycle_stage_from_age() {
    assert_eq!(Citizen::lifecycle_for_age(3.0), LifecycleStage::Child);
    assert_eq!(Citizen::lifecycle_for_age(12.0), LifecycleStage::Student);
    assert_eq!(Citizen::lifecycle_for_age(30.0), LifecycleStage::Worker);
    assert_eq!(Citizen::lifecycle_for_age(70.0), LifecycleStage::Retired);
}

#[test]
fn needs_satisfaction_average() {
    let needs = Needs { housing: 1.0, food: 1.0, health: 1.0, safety: 1.0, education: 1.0, employment: 1.0, leisure: 1.0 };
    assert!((needs.satisfaction() - 1.0).abs() < 0.001);
    assert!((Needs::default().satisfaction() - 0.5).abs() < 0.001);
}
