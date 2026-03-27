use vox_sim::buildings::{BuildingManager, BuildingType};
use vox_sim::citizen::{CitizenManager, EducationLevel};
use vox_sim::employment::*;

#[test]
fn match_employment_assigns_jobs() {
    let mut citizens = CitizenManager::new();
    for _ in 0..10 {
        citizens.spawn(25.0, Some(0));
    }

    let mut buildings = BuildingManager::new();
    // Citizens spawned at age 25 get Secondary education, which maps to Commercial
    buildings.add_building(BuildingType::Commercial, [0.0, 0.0], 20);

    let matched = match_employment(citizens.all_mut(), &mut buildings);
    assert!(matched > 0, "Should match some workers to jobs");
}

#[test]
fn match_housing_assigns_homes() {
    let mut citizens = CitizenManager::new();
    for _ in 0..5 {
        citizens.spawn(25.0, None);
    }

    let mut buildings = BuildingManager::new();
    buildings.add_building(BuildingType::Residential, [0.0, 0.0], 10);

    let matched = match_housing(citizens.all_mut(), &mut buildings);
    assert_eq!(matched, 5);
}

#[test]
fn crime_increases_with_unemployment() {
    let high_employment = calculate_crime_rate(1000, 900, 0.5);
    let low_employment = calculate_crime_rate(1000, 200, 0.5);
    assert!(low_employment > high_employment);
}

#[test]
fn police_reduces_crime() {
    let no_police = calculate_crime_rate(1000, 500, 0.0);
    let full_police = calculate_crime_rate(1000, 500, 1.0);
    assert!(full_police < no_police);
}

#[test]
fn education_pipeline() {
    let mut citizens = CitizenManager::new();
    citizens.spawn(10.0, Some(0));

    let graduated = process_education(citizens.all_mut(), true, false, false);
    assert!(graduated > 0);
    assert_eq!(citizens.all()[0].education, EducationLevel::Primary);
}
