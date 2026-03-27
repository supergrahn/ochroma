use bevy_ecs::prelude::*;
use glam::{Vec3, Quat};
use uuid::Uuid;

use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};
use vox_sim::citizen::CitizenManager;
use vox_sim::economy::CityBudget;
use vox_sim::zoning::{ZoningManager, ZoneType};
use vox_sim::services::{ServiceManager, ServiceType};
use vox_sim::traffic::{TrafficNetwork, RoadSegmentTraffic};
use vox_sim::buildings::{BuildingManager, BuildingType};
use vox_sim::milestones::MilestoneTracker;
use vox_sim::pollution::PollutionGrid;
use vox_sim::disasters::DisasterManager;

/// Simulate a full city lifecycle: build roads, zone, grow, manage for 1000 ticks.
#[test]
fn full_city_lifecycle_1000_ticks() {
    let mut citizens = CitizenManager::new();
    let mut budget = CityBudget::default();
    let mut zoning = ZoningManager::new();
    let mut services = ServiceManager::new();
    let mut traffic = TrafficNetwork::new();
    let mut buildings = BuildingManager::new();
    let mut milestones = MilestoneTracker::new();
    let mut pollution = PollutionGrid::new(50, 50, 10.0);
    let mut disasters = DisasterManager::new();

    // Initial setup: some zones and roads
    for i in 0..20 {
        zoning.zone_plot(ZoneType::ResidentialLow, [i as f32 * 12.0, 0.0], [10.0, 10.0]);
    }
    for i in 0..5 {
        zoning.zone_plot(ZoneType::CommercialLocal, [i as f32 * 15.0, 20.0], [10.0, 10.0]);
    }
    for i in 0..3 {
        zoning.zone_plot(ZoneType::IndustrialLight, [i as f32 * 20.0, 40.0], [10.0, 10.0]);
    }

    services.place_service(ServiceType::PrimarySchool, [50.0, 50.0]);
    services.place_service(ServiceType::FireStation, [100.0, 0.0]);

    traffic.add_segment(RoadSegmentTraffic::new(0, 0.5, 200.0, 50.0));

    // Add buildings
    for i in 0..10 {
        buildings.add_building(BuildingType::Residential, [i as f32 * 12.0, 0.0], 20);
    }
    for i in 0..3 {
        buildings.add_building(BuildingType::Commercial, [i as f32 * 15.0, 20.0], 10);
    }
    buildings.add_building(BuildingType::Industrial, [0.0, 40.0], 15);

    // Spawn initial citizens
    for _ in 0..200 {
        citizens.spawn(25.0, Some(0));
    }

    // Run 1000 simulation ticks
    for tick in 0..1000 {
        let dt = 0.1;
        citizens.tick(dt / 8760.0); // age in years
        budget.tick(citizens.count() as u32, 3, 1);
        traffic.tick(dt);
        zoning.update_demand(citizens.count() as u32);
        milestones.check(citizens.count() as u32);
        pollution.decay(0.001);
        disasters.tick(dt);

        // Trigger a disaster at tick 50 (early enough that it resolves by tick 1000)
        if tick == 50 {
            disasters.trigger(vox_sim::disasters::DisasterType::Fire, [50.0, 50.0], 0.3);
        }
    }

    // Verify state is consistent
    assert!(citizens.count() > 0, "Some citizens should survive 1000 ticks");
    assert!(budget.funds != 50000.0, "Budget should have changed");
    assert!(milestones.achieved_count() >= 1, "Should have hit at least first milestone");

    // Disasters should have resolved
    assert_eq!(disasters.active_count(), 0, "Fire should have burned out");
}

/// Test ECS with 50,000 entities doesn't crash.
#[test]
fn ecs_50k_entities() {
    let mut world = World::new();
    let uuid = Uuid::new_v4();

    world.spawn(SplatAssetComponent {
        uuid,
        splat_count: 10,
        splats: Vec::new(),
    });

    for i in 0..50_000 {
        world.spawn(SplatInstanceComponent {
            asset_uuid: uuid,
            position: Vec3::new((i % 200) as f32, 0.0, (i / 200) as f32),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            instance_id: i as u32,
            lod: LodLevel::Full,
        });
    }

    let mut query = world.query::<&SplatInstanceComponent>();
    assert_eq!(query.iter(&world).count(), 50_000);
}

/// Test that all simulation systems can run together without panicking.
#[test]
fn all_systems_tick_without_panic() {
    use vox_sim::supply_chain::SupplyChainManager;
    use vox_sim::land_value::LandValueGrid;
    use vox_sim::districts::DistrictManager;
    use vox_sim::migration::MigrationSystem;
    use vox_sim::agent::AgentManager;

    let mut citizens = CitizenManager::new();
    let mut budget = CityBudget::default();
    let mut zoning = ZoningManager::new();
    let mut services = ServiceManager::new();
    let mut traffic = TrafficNetwork::new();
    let mut buildings = BuildingManager::new();
    let mut supply_chain = SupplyChainManager::new();
    let mut land_value = LandValueGrid::new(50, 50, 10.0);
    let mut districts = DistrictManager::new();
    let mut migration = MigrationSystem::new();
    let mut agents = AgentManager::new();
    let mut milestones = MilestoneTracker::new();
    let mut pollution = PollutionGrid::new(50, 50, 10.0);
    let mut disasters = DisasterManager::new();

    // Populate
    for _ in 0..500 { citizens.spawn(25.0, Some(0)); }
    zoning.zone_plot(ZoneType::ResidentialLow, [0.0, 0.0], [50.0, 50.0]);
    services.place_service(ServiceType::Hospital, [25.0, 25.0]);
    buildings.add_building(BuildingType::Industrial, [0.0, 0.0], 50);
    buildings.add_building(BuildingType::Commercial, [30.0, 0.0], 20);
    districts.create_district("Downtown", [0.0, 0.0], [100.0, 100.0]);

    // Run 100 ticks
    for _ in 0..100 {
        citizens.tick(0.001);
        budget.tick(citizens.count() as u32, 1, 1);
        traffic.tick(0.1);
        zoning.update_demand(citizens.count() as u32);
        supply_chain.tick(&buildings, 0.1);
        land_value.recalculate(&[(25.0, 25.0, 500.0)], &[]);
        agents.tick(0.1);
        milestones.check(citizens.count() as u32);
        pollution.diffuse(0.01);
        pollution.decay(0.001);
        disasters.tick(0.1);

        let (arrivals, _departures) = migration.calculate_migration(&citizens, 0.6, 100, 11.0);
        for _ in 0..arrivals { citizens.spawn(25.0, Some(0)); }
    }

    // Just verify no panics occurred
    assert!(citizens.count() > 0);
    assert!(buildings.count() > 0);
}

/// Save/load round-trip preserves data.
#[test]
fn save_load_full_state() {
    use vox_data::scene_serialize::WorldSnapshot;

    let mut snap = WorldSnapshot::new(12345.0);
    snap.simulation.citizen_count = 5000;
    snap.simulation.funds = 75000.0;
    snap.simulation.building_count = 200;
    snap.simulation.road_segment_count = 50;
    snap.simulation.zone_count = 100;

    for i in 0..500 {
        let entity = snap.add_entity(i);
        entity.components.insert("position".into(), serde_json::json!([i as f32, 0.0, 0.0]));
        entity.components.insert("type".into(), serde_json::json!("building"));
    }

    let bytes = snap.to_bytes().unwrap();
    let loaded = WorldSnapshot::from_bytes(&bytes).unwrap();

    assert_eq!(loaded.simulation.citizen_count, 5000);
    assert_eq!(loaded.simulation.building_count, 200);
    assert_eq!(loaded.entities.len(), 500);
}
