use vox_sim::migration::MigrationSystem;
use vox_sim::citizen::CitizenManager;

#[test]
fn high_satisfaction_attracts() {
    let mut migration = MigrationSystem::new();
    let citizens = CitizenManager::new();
    let (arrivals, _) = migration.calculate_migration(&citizens, 0.8, 100, 15.0);
    assert!(arrivals > 0);
}

#[test]
fn low_satisfaction_no_arrivals() {
    let mut migration = MigrationSystem::new();
    let citizens = CitizenManager::new();
    let (arrivals, _) = migration.calculate_migration(&citizens, 0.3, 100, 15.0);
    assert_eq!(arrivals, 0);
}
