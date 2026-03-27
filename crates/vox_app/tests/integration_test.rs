use bevy_ecs::prelude::*;
use glam::{Vec3, Quat};
use uuid::Uuid;
use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};

#[test]
fn ecs_world_with_10000_instances() {
    let mut world = World::new();
    let asset_uuid = Uuid::new_v4();

    // Spawn one asset
    world.spawn(SplatAssetComponent {
        uuid: asset_uuid,
        splat_count: 100,
        splats: Vec::new(), // empty for speed
    });

    // Spawn 10,000 instances
    for i in 0..10_000 {
        world.spawn(SplatInstanceComponent {
            asset_uuid,
            position: Vec3::new((i % 100) as f32, 0.0, (i / 100) as f32),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            instance_id: i as u32,
            lod: LodLevel::Full,
        });
    }

    let mut query = world.query::<&SplatInstanceComponent>();
    assert_eq!(query.iter(&world).count(), 10_000);
}

#[test]
fn simulation_state_runs_without_panic() {
    use vox_sim::citizen::CitizenManager;
    use vox_sim::economy::CityBudget;
    use vox_sim::zoning::{ZoningManager, ZoneType};
    use vox_sim::services::{ServiceManager, ServiceType};
    use vox_sim::traffic::TrafficNetwork;

    let mut citizens = CitizenManager::new();
    for _ in 0..100 { citizens.spawn(25.0, None); }

    let mut budget = CityBudget::default();
    let mut zoning = ZoningManager::new();
    zoning.zone_plot(ZoneType::ResidentialLow, [0.0, 0.0], [10.0, 10.0]);
    zoning.zone_plot(ZoneType::CommercialLocal, [20.0, 0.0], [10.0, 10.0]);

    let mut services = ServiceManager::new();
    services.place_service(ServiceType::PrimarySchool, [50.0, 50.0]);

    let mut traffic = TrafficNetwork::new();

    // Run 1000 ticks without crashing
    for _ in 0..1000 {
        citizens.tick(0.001);
        budget.tick(citizens.count() as u32, 1, 1);
        zoning.update_demand(citizens.count() as u32);
        traffic.tick(0.1);
    }

    // Verify state is consistent
    assert!(citizens.count() > 0, "Some citizens should survive");
    assert!(budget.funds != 0.0, "Budget should have changed");
}

#[test]
fn save_load_round_trip() {
    use vox_data::save::{GameState, SaveHeader, save_game, load_game};

    let dir = std::env::temp_dir().join("ochroma_integration_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let state = GameState {
        header: SaveHeader {
            version: 1,
            city_name: "Integration Test City".into(),
            game_time_hours: 500.0,
            citizen_count: 10000,
            funds: 123456.78,
        },
        data: vec![42; 1000],
    };

    let path = dir.join("test.ochroma_save");
    save_game(&state, &path).unwrap();
    let loaded = load_game(&path).unwrap();

    assert_eq!(loaded.header.city_name, "Integration Test City");
    assert_eq!(loaded.header.citizen_count, 10000);
    assert_eq!(loaded.data.len(), 1000);

    let _ = std::fs::remove_dir_all(&dir);
}
