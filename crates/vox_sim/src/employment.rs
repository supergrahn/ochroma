use crate::buildings::{BuildingManager, BuildingType};
use crate::citizen::{Citizen, EducationLevel, LifecycleStage};
use crate::services::{ServiceManager, ServiceType};

/// Match citizens to available jobs based on education and proximity.
pub fn match_employment(citizens: &mut [Citizen], buildings: &mut BuildingManager) -> u32 {
    let mut matched = 0u32;

    for citizen in citizens.iter_mut() {
        if citizen.lifecycle != LifecycleStage::Worker {
            continue;
        }
        if citizen.employment.is_some() {
            continue;
        }

        let bt = match citizen.education {
            EducationLevel::None | EducationLevel::Primary => BuildingType::Industrial,
            EducationLevel::Secondary => BuildingType::Commercial,
            EducationLevel::University => BuildingType::Commercial,
        };

        let pos = citizen
            .residence
            .map(|r| [r as f32 * 10.0, 0.0])
            .unwrap_or([0.0, 0.0]);
        if let Some(building_id) = buildings.find_nearest_with_vacancy(pos, bt)
            && buildings.assign_occupant(building_id)
        {
            citizen.employment = Some(building_id);
            citizen.workplace = Some(building_id);
            citizen.needs.employment = 0.8;
            matched += 1;
        }
    }

    matched
}

/// Like [`match_employment`] but enforces the **dependent-care employment gate**:
/// a `Worker` who is the guardian of a dependent in `care_stage` cannot take a
/// job unless a `care_service` covers their home — there is no one to leave the
/// dependent with. This is the general city-builder primitive; the GAME picks the
/// stage + service (e.g. `Child` + `Childcare`). Equivalent to `match_employment`
/// for citizens with no dependents (so it never changes a childless workforce).
pub fn match_employment_gated(
    citizens: &mut [Citizen],
    buildings: &mut BuildingManager,
    services: &ServiceManager,
    care_stage: LifecycleStage,
    care_service: ServiceType,
) -> u32 {
    // Snapshot id -> lifecycle BEFORE the mutable pass, so a guardian can check
    // whether any of its dependents is in the care-requiring stage (the dependents
    // are other citizens in the same slice).
    let stage_of: std::collections::HashMap<u32, LifecycleStage> =
        citizens.iter().map(|c| (c.id, c.lifecycle)).collect();

    let mut matched = 0u32;
    for citizen in citizens.iter_mut() {
        if citizen.lifecycle != LifecycleStage::Worker {
            continue;
        }
        if citizen.employment.is_some() {
            continue;
        }

        // Home via the same residence heuristic `match_employment` uses.
        let home = citizen
            .residence
            .map(|r| [r as f32 * 10.0, 0.0])
            .unwrap_or([0.0, 0.0]);

        // THE GATE: a guardian of a dependent in `care_stage` is blocked from
        // employment unless a covering care service exists at home.
        let needs_care = citizen
            .dependents
            .iter()
            .any(|d| stage_of.get(d) == Some(&care_stage));
        if needs_care && !services.is_covered(home, care_service) {
            continue;
        }

        let bt = match citizen.education {
            EducationLevel::None | EducationLevel::Primary => BuildingType::Industrial,
            EducationLevel::Secondary => BuildingType::Commercial,
            EducationLevel::University => BuildingType::Commercial,
        };
        if let Some(building_id) = buildings.find_nearest_with_vacancy(home, bt)
            && buildings.assign_occupant(building_id)
        {
            citizen.employment = Some(building_id);
            citizen.workplace = Some(building_id);
            citizen.needs.employment = 0.8;
            matched += 1;
        }
    }

    matched
}

/// Match citizens to available housing.
pub fn match_housing(citizens: &mut [Citizen], buildings: &mut BuildingManager) -> u32 {
    let mut matched = 0u32;

    for citizen in citizens.iter_mut() {
        if citizen.residence.is_some() {
            continue;
        }

        let pos = [0.0, 0.0];
        if let Some(building_id) = buildings.find_nearest_with_vacancy(pos, BuildingType::Residential)
            && buildings.assign_occupant(building_id)
        {
            citizen.residence = Some(building_id);
            citizen.needs.housing = 0.8;
            matched += 1;
        }
    }

    matched
}

/// Calculate crime level based on police coverage and unemployment.
pub fn calculate_crime_rate(citizen_count: u32, employed_count: u32, police_coverage: f32) -> f32 {
    if citizen_count == 0 {
        return 0.0;
    }
    let unemployment_rate = 1.0 - (employed_count as f32 / citizen_count as f32);
    let base_crime = unemployment_rate * 0.5;
    let policed = base_crime * (1.0 - police_coverage * 0.8);
    policed.clamp(0.0, 1.0)
}

/// Education pipeline: children attend school, graduate to higher education.
pub fn process_education(
    citizens: &mut [Citizen],
    has_primary: bool,
    has_secondary: bool,
    has_university: bool,
) -> u32 {
    let mut graduated = 0u32;

    for citizen in citizens.iter_mut() {
        match citizen.lifecycle {
            LifecycleStage::Student => {
                if citizen.age >= 6.0 && citizen.age < 12.0 && has_primary
                    && citizen.education < EducationLevel::Primary
                {
                    citizen.education = EducationLevel::Primary;
                    citizen.needs.education = 0.7;
                    graduated += 1;
                }
                if citizen.age >= 12.0 && citizen.age < 18.0 && has_secondary
                    && citizen.education < EducationLevel::Secondary
                {
                    citizen.education = EducationLevel::Secondary;
                    citizen.needs.education = 0.8;
                    graduated += 1;
                }
            }
            LifecycleStage::Worker
                if citizen.age < 25.0
                    && has_university
                    && citizen.education < EducationLevel::University
                => {
                    citizen.education = EducationLevel::University;
                    citizen.needs.education = 0.9;
                    graduated += 1;
                }
            _ => {}
        }
    }

    graduated
}
