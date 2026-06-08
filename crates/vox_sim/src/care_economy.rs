//! Care economy — the **"dependent-care gates employment"** mechanic.
//!
//! The general primitive lives in [`crate::employment::match_employment_gated`]
//! (game-type level: a guardian of a dependent in *some* lifecycle stage can't
//! work without a covering care service). THIS module is the GAME binding:
//! **childcare** ([`ServiceType::Childcare`]) gates a parent of a
//! [`LifecycleStage::Child`] (the under-6s).
//!
//! The domino — and the reason it matters: **no childcare → a parent can't go to
//! work → no income → the household's economy and satisfaction fall → the family
//! eventually leaves.** Cities: Skylines never modelled who looks after the
//! children; Ochroma does, because the lifecycle + service + employment substrate
//! was already here — the mechanic is one gate on a loop that was already turning.

use crate::buildings::BuildingManager;
use crate::citizen::{Citizen, LifecycleStage};
use crate::employment::match_employment_gated;
use crate::services::{ServiceManager, ServiceType};

/// Employment matching with the CHILDCARE gate applied: a parent of a `Child`
/// must have `Childcare` coverage at home to take a job. The game-specific binding
/// of the general [`match_employment_gated`] primitive.
pub fn match_employment_childcare(
    citizens: &mut [Citizen],
    buildings: &mut BuildingManager,
    services: &ServiceManager,
) -> u32 {
    match_employment_gated(
        citizens,
        buildings,
        services,
        LifecycleStage::Child,
        ServiceType::Childcare,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buildings::BuildingType;
    use crate::citizen::CitizenManager;

    /// THE mechanic, proven: a parent of a toddler cannot take an available job
    /// while there is no childcare; place a kindergarten in range and they can.
    #[test]
    fn childcare_gates_employment() {
        let mut cm = CitizenManager::new();
        // Parent: a Worker (age 30), lives in residence-1 (home heuristic = [10,0]).
        let parent = cm.spawn(30.0, Some(1));
        // Toddler: a Child (age 3), same residence — the dependent.
        let toddler = cm.spawn(3.0, Some(1));
        cm.add_dependent(parent, toddler);

        // A commercial job exists right by the home, so the ONLY thing that can
        // block employment is the childcare gate — not job availability.
        let mut buildings = BuildingManager::new();
        buildings.add_building(BuildingType::Commercial, [10.0, 0.0], 5);

        let mut services = ServiceManager::new();

        // 1. No childcare anywhere → the parent is stranded, can't go to work.
        let m0 = match_employment_childcare(cm.all_mut(), &mut buildings, &services);
        assert_eq!(
            cm.get(parent).unwrap().employment,
            None,
            "no childcare → parent can't take the job (matched={m0})"
        );

        // 2. Open a kindergarten covering the home → the parent can finally work.
        services.place_service(ServiceType::Childcare, [10.0, 0.0]);
        let m1 = match_employment_childcare(cm.all_mut(), &mut buildings, &services);
        assert!(
            cm.get(parent).unwrap().employment.is_some(),
            "childcare in range → parent can finally go to work (matched={m1})"
        );

        println!(
            "OK: no childcare → parent UNEMPLOYED; kindergarten placed → parent EMPLOYED \
             — the mechanic CS2 forgot."
        );
    }

    /// A childless worker is unaffected by the gate — employs normally even with
    /// zero childcare in the city (the gate only bites guardians of children).
    #[test]
    fn childless_worker_unaffected_by_gate() {
        let mut cm = CitizenManager::new();
        let worker = cm.spawn(40.0, Some(1)); // no dependents
        let mut buildings = BuildingManager::new();
        buildings.add_building(BuildingType::Commercial, [10.0, 0.0], 5);
        let services = ServiceManager::new(); // no childcare at all
        match_employment_childcare(cm.all_mut(), &mut buildings, &services);
        assert!(
            cm.get(worker).unwrap().employment.is_some(),
            "a worker with no children employs normally regardless of childcare"
        );
    }

    /// The kindergarten must actually COVER the home: a childcare far outside the
    /// coverage radius does not unblock the parent (proves it's spatial, not a
    /// global flag).
    #[test]
    fn childcare_out_of_range_does_not_unblock() {
        let mut cm = CitizenManager::new();
        let parent = cm.spawn(30.0, Some(1)); // home heuristic [10,0]
        let toddler = cm.spawn(3.0, Some(1));
        cm.add_dependent(parent, toddler);
        let mut buildings = BuildingManager::new();
        buildings.add_building(BuildingType::Commercial, [10.0, 0.0], 5);
        let mut services = ServiceManager::new();
        // Childcare 9 km away — outside its ~1.5 km coverage radius.
        services.place_service(ServiceType::Childcare, [9000.0, 0.0]);
        match_employment_childcare(cm.all_mut(), &mut buildings, &services);
        assert_eq!(
            cm.get(parent).unwrap().employment,
            None,
            "childcare out of coverage range must NOT unblock the parent"
        );
    }
}
