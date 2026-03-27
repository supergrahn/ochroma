use vox_sim::supply_chain::SupplyChainManager;
use vox_sim::economy::ResourceType;
use vox_sim::buildings::{BuildingManager, BuildingType};

#[test]
fn industrial_produces_resources() {
    let mut chain = SupplyChainManager::new();
    let mut buildings = BuildingManager::new();
    buildings.add_building(BuildingType::Industrial, [0.0, 0.0], 10);
    chain.tick(&buildings, 1.0);
    assert!(chain.stock_level(ResourceType::Timber) > 0.0);
}

#[test]
fn commercial_processes_resources() {
    let mut chain = SupplyChainManager::new();
    let mut buildings = BuildingManager::new();
    buildings.add_building(BuildingType::Industrial, [0.0, 0.0], 10);
    buildings.add_building(BuildingType::Commercial, [50.0, 0.0], 5);
    // Produce first
    chain.tick(&buildings, 1.0);
    // Process
    chain.tick(&buildings, 1.0);
    // Timber should be consumed, planks produced
    assert!(chain.stock_level(ResourceType::Planks) > 0.0);
}
