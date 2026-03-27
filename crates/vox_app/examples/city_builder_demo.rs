//! City Builder Demo -- showcases all 31 vox_sim modules running together.
//! This is a GAME built on the Ochroma engine, not part of the engine itself.

// === Import every single vox_sim module ===
use vox_sim::advisor::AdvisorSystem;
use vox_sim::agent::AgentManager;
use vox_sim::bdi_agent::BdiAgent;
use vox_sim::buildings::{BuildingManager, BuildingType};
use vox_sim::citizen::CitizenManager;
use vox_sim::city_council::CityCouncil;
use vox_sim::deterministic::SimulationRecorder;
use vox_sim::disasters::{DisasterManager, DisasterType};
use vox_sim::districts::DistrictManager;
use vox_sim::economy::CityBudget;
use vox_sim::ecosystem::{EcosystemManager, TreeSpecies};
use vox_sim::history::GameHistory;
use vox_sim::land_value::LandValueGrid;
use vox_sim::migration::MigrationSystem;
use vox_sim::milestones::MilestoneTracker;
use vox_sim::pollution::PollutionGrid;
use vox_sim::roads::RoadNetwork;
use vox_sim::seasons::Season;
use vox_sim::services::{ServiceManager, ServiceType};
use vox_sim::sharding::ShardManager;
use vox_sim::social_network::{RelationshipType, SocialNetwork};
use vox_sim::supply_chain::SupplyChainManager;
use vox_sim::trade::TradeSystem;
use vox_sim::traffic::{RoadSegmentTraffic, TrafficNetwork};
use vox_sim::transport::{TransportManager, TransportType};
use vox_sim::utilities::{UtilityNetwork, UtilityType};
use vox_sim::vehicles::{VehicleManager, VehicleType};
use vox_sim::weather::WeatherState;
use vox_sim::zoning::{ZoningManager, ZoneType};

use glam::Vec3;
use vox_core::lwc::WorldCoord;

fn main() {
    println!("=== City Builder Demo ===");
    println!("Initializing all 31 vox_sim subsystems...\n");

    // ---------------------------------------------------------------
    // 1. Initialize ALL 31 systems
    // ---------------------------------------------------------------

    // (1) Zoning
    let mut zoning = ZoningManager::new();
    zoning.zone_plot(ZoneType::ResidentialLow, [0.0, 0.0], [100.0, 100.0]);
    zoning.zone_plot(ZoneType::CommercialLocal, [100.0, 0.0], [100.0, 100.0]);
    zoning.zone_plot(ZoneType::IndustrialLight, [200.0, 0.0], [100.0, 100.0]);
    zoning.zone_plot(ZoneType::Office, [0.0, 100.0], [100.0, 100.0]);
    zoning.zone_plot(ZoneType::Park, [100.0, 100.0], [50.0, 50.0]);
    println!("[01/31] Zoning: {} plots laid out", zoning.plot_count());

    // (2) Buildings
    let mut buildings = BuildingManager::new();
    for i in 0..10 {
        buildings.add_building(BuildingType::Residential, [i as f32 * 10.0, 0.0], 20);
    }
    for i in 0..5 {
        buildings.add_building(BuildingType::Commercial, [100.0 + i as f32 * 10.0, 0.0], 15);
    }
    for i in 0..3 {
        buildings.add_building(BuildingType::Industrial, [200.0 + i as f32 * 10.0, 0.0], 30);
    }
    println!("[02/31] Buildings: {} placed", buildings.buildings.len());

    // (3) Roads
    let mut roads = RoadNetwork::new();
    roads.add_straight(
        vox_sim::roads::RoadType::Avenue,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(300.0, 0.0, 0.0),
    );
    roads.add_straight(
        vox_sim::roads::RoadType::LocalStreet,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 200.0),
    );
    roads.add_curve(
        vox_sim::roads::RoadType::Highway,
        Vec3::new(300.0, 0.0, 0.0),
        Vec3::new(350.0, 0.0, 100.0),
        Vec3::new(300.0, 0.0, 200.0),
    );
    println!(
        "[03/31] Roads: {} segments, {:.0}m total",
        roads.segment_count(),
        roads.total_length()
    );

    // (4) Services
    let mut services = ServiceManager::new();
    services.place_service(ServiceType::Fire, [50.0, 50.0]);
    services.place_service(ServiceType::Police, [150.0, 50.0]);
    services.place_service(ServiceType::Hospital, [250.0, 50.0]);
    services.place_service(ServiceType::School, [50.0, 150.0]);
    services.place_service(ServiceType::Park, [100.0, 100.0]);
    println!("[04/31] Services: 5 placed, cost={:.0}/tick", services.total_cost());

    // (5) Citizens
    let mut citizens = CitizenManager::new();
    for i in 0..50 {
        citizens.spawn(20.0 + (i as f32 * 0.5), Some(i % 10));
    }
    println!("[05/31] Citizens: {} spawned", citizens.count());

    // (6) Economy
    let mut budget = CityBudget::default();
    budget.residential_income = 500.0;
    budget.commercial_income = 300.0;
    budget.industrial_income = 200.0;
    budget.services_expenses = 150.0;
    let report = budget.generate_report(citizens.count() as u32);
    println!("[06/31] Economy: funds={:.0}, net={:.0}", report.funds, report.net);

    // (7) Traffic
    let mut traffic = TrafficNetwork::new();
    traffic.add_segment(RoadSegmentTraffic::new(0, 1.0, 200.0, 60.0));
    traffic.add_segment(RoadSegmentTraffic::new(1, 0.5, 150.0, 50.0));
    traffic.add_segment(RoadSegmentTraffic::new(2, 2.0, 300.0, 120.0));
    println!("[07/31] Traffic: {} segments", traffic.segments.len());

    // (8) Transport
    let mut transport = TransportManager::new();
    let bus_route = transport.create_route(TransportType::Bus, 10.0, 5);
    transport.add_stop(bus_route, [0.0, 0.0], "Central Station");
    transport.add_stop(bus_route, [100.0, 0.0], "Commercial District");
    transport.add_stop(bus_route, [200.0, 0.0], "Industrial Park");
    let metro_route = transport.create_route(TransportType::Metro, 5.0, 3);
    transport.add_stop(metro_route, [0.0, 0.0], "Metro Central");
    transport.add_stop(metro_route, [150.0, 0.0], "Metro East");
    println!(
        "[08/31] Transport: revenue={:.0}/hr",
        transport.total_hourly_revenue(2.5, 0.6)
    );

    // (9) Utilities -- power
    let mut power = UtilityNetwork::new(UtilityType::Power);
    let power_src = power.add_source([250.0, 50.0], 1000.0);
    let power_c1 = power.add_consumer([50.0, 0.0], 200.0);
    power.connect(power_src, power_c1, 500.0);
    println!("[09/31] Power: {} nodes", power.nodes.len());

    // (10) Utilities -- water
    let mut water_net = UtilityNetwork::new(UtilityType::Water);
    let water_src = water_net.add_source([0.0, 200.0], 800.0);
    let water_c1 = water_net.add_consumer([50.0, 0.0], 150.0);
    water_net.connect(water_src, water_c1, 400.0);
    println!("[10/31] Water: {} nodes, served={}", water_net.nodes.len(), water_net.is_served(water_c1));

    // (11) Disasters
    let mut disasters = DisasterManager::new();
    disasters.trigger(DisasterType::Fire, [120.0, 30.0], 0.3);
    println!("[11/31] Disasters: {} active", disasters.active.len());

    // (12) Pollution
    let mut pollution = PollutionGrid::new(64, 64, 10.0);
    pollution.add_source(200.0, 0.0, 100.0, 0.5, true);
    println!("[12/31] Pollution: air at industrial={:.2}", pollution.air_at(200.0, 0.0));

    // (13) Land value
    let mut land_value = LandValueGrid::new(64, 64, 10.0);
    land_value.recalculate(
        &[(50.0, 50.0, 200.0), (150.0, 50.0, 200.0)],
        &[(100.0, 100.0)],
    );
    println!("[13/31] Land value: center={:.2}", land_value.sample(0.0, 0.0));

    // (14) Districts
    let mut districts = DistrictManager::new();
    let d1 = districts.create_district("Downtown", [0.0, 0.0], [150.0, 150.0]);
    let d2 = districts.create_district("Industrial Zone", [150.0, 0.0], [300.0, 100.0]);
    let _ = (d1, d2);
    println!("[14/31] Districts: {} created", districts.districts.len());

    // (15) Advisor
    let mut advisor = AdvisorSystem::new();
    advisor.evaluate(
        citizens.count() as u32,
        budget.funds,
        0.1,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        0.2,
    );
    println!("[15/31] Advisor: {} messages", advisor.messages.len());

    // (16) Milestones
    let mut milestones = MilestoneTracker::new();
    milestones.check(citizens.count() as u32);
    println!(
        "[16/31] Milestones: era={}, notifications={}",
        milestones.current_era.label(),
        milestones.notifications.len()
    );

    // (17) Migration
    let mut migration = MigrationSystem::new();
    let (arrivals, departures) = migration.calculate_migration(&citizens, 0.7, 50, 100.0);
    println!("[17/31] Migration: +{} -{}", arrivals, departures);

    // (18) Seasons
    let season = Season::from_day(90);
    println!(
        "[18/31] Seasons: {:?}, heating_cost={:.1}x, crop_growth={:.1}",
        season,
        season.heating_cost_multiplier(),
        season.crop_growth_rate()
    );

    // (19) Weather
    let mut weather = WeatherState::new(90);
    weather.tick(1.0, 42);
    println!(
        "[19/31] Weather: {:?}, temp={:.1}C, wind={:.1}m/s",
        weather.current, weather.temperature, weather.wind_speed
    );

    // (20) Ecosystem
    let mut ecosystem = EcosystemManager::new();
    ecosystem.plant_tree([50.0, 120.0], TreeSpecies::Oak);
    ecosystem.plant_tree([60.0, 120.0], TreeSpecies::Pine);
    ecosystem.plant_tree([70.0, 120.0], TreeSpecies::Birch);
    ecosystem.tick(1.0, |_pos| 0.1, 0.5);
    println!("[20/31] Ecosystem: {} trees", ecosystem.count());

    // (21) Employment (module-level function)
    let matched_jobs = vox_sim::employment::match_employment(citizens.all_mut(), &mut buildings);
    let matched_housing = vox_sim::employment::match_housing(citizens.all_mut(), &mut buildings);
    println!("[21/31] Employment: {} jobs matched, {} housed", matched_jobs, matched_housing);

    // (22) History
    let mut history = GameHistory::new(1000);
    history.record_tick(0.0, citizens.count() as u32, budget.funds, 1000.0, 150.0, 0.7, 0.1, 0.2, 0.3);
    println!("[22/31] History: pop_avg={:.0}", history.population.average());

    // (23) Supply chain
    let mut supply_chain = SupplyChainManager::new();
    supply_chain.tick(&buildings, 1.0);
    println!("[23/31] Supply chain: {} resource types tracked", supply_chain.stocks.len());

    // (24) Trade
    let mut trade = TradeSystem::new();
    trade.update_prices();
    println!("[24/31] Trade: {} partners, balance={:.0}", trade.partners.len(), trade.trade_balance);

    // (25) Vehicles
    let mut vehicles = VehicleManager::new(500);
    vehicles.spawn(VehicleType::Car, Vec3::new(10.0, 0.0, 0.0), vec![0, 1]);
    vehicles.spawn(VehicleType::Bus, Vec3::new(0.0, 0.0, 0.0), vec![0, 1, 2]);
    vehicles.spawn(VehicleType::Truck, Vec3::new(200.0, 0.0, 0.0), vec![2]);
    vehicles.spawn(VehicleType::EmergencyVehicle, Vec3::new(120.0, 0.0, 30.0), vec![0]);
    println!("[25/31] Vehicles: {} active", vehicles.vehicles.len());

    // (26) Agents
    let mut agents = AgentManager::new();
    for i in 0..10 {
        agents.spawn(WorldCoord::from_absolute(i as f64 * 10.0, 0.0, 0.0), 5.0);
    }
    agents.tick(0.1);
    println!("[26/31] Agents: {} spawned", agents.count());

    // (27) BDI agents
    let mut bdi_agents: Vec<BdiAgent> = (0..5).map(|i| BdiAgent::new(i)).collect();
    for bdi in &mut bdi_agents {
        bdi.beliefs.push(vox_sim::bdi_agent::Belief::KnowsJobAt(5));
        bdi.desires.push(vox_sim::bdi_agent::Desire::FindBetterJob);
    }
    println!("[27/31] BDI agents: {} with beliefs and desires", bdi_agents.len());

    // (28) Social network
    let mut social = SocialNetwork::new();
    social.add_relationship(0, 1, RelationshipType::Friend, 0.8);
    social.add_relationship(1, 2, RelationshipType::Coworker, 0.5);
    social.add_relationship(0, 3, RelationshipType::Family, 1.0);
    println!("[28/31] Social network: relationships established");

    // (29) City council
    let mut council = CityCouncil::new();
    council.add_member(vox_sim::city_council::CouncilMember::new(
        0,
        "Mayor Smith".into(),
        vox_sim::city_council::Ideology::Progressive,
        vec![vox_sim::city_council::PolicyArea::Healthcare],
    ));
    council.add_member(vox_sim::city_council::CouncilMember::new(
        1,
        "Councilor Jones".into(),
        vox_sim::city_council::Ideology::Conservative,
        vec![vox_sim::city_council::PolicyArea::Infrastructure],
    ));
    println!("[29/31] City council: {} members", council.members.len());

    // (30) Sharding
    let mut shards = ShardManager::new();
    let shard_tiles = std::collections::HashSet::from([
        vox_sim::sharding::TileCoord { x: 0, y: 0 },
        vox_sim::sharding::TileCoord { x: 1, y: 0 },
    ]);
    let _shard_id = shards.create_shard(shard_tiles);
    println!("[30/31] Sharding: shard count={}", shards.shard_count());

    // (31) Deterministic replay
    let mut recorder = SimulationRecorder::new();
    recorder.record_tick(0, vec![], 42);
    recorder.record_tick(1, vec![], 43);
    let replayed = recorder.replay_tick(0);
    println!(
        "[31/31] Deterministic: tick 0 replay={}",
        if replayed.is_some() { "OK" } else { "MISSING" }
    );

    println!("\n--- All 31 systems initialized ---\n");

    // ---------------------------------------------------------------
    // 2. Run 100 simulation ticks
    // ---------------------------------------------------------------
    println!("Running 100 simulation ticks...");
    let dt = 1.0_f32;
    for tick in 0..100 {
        let time = tick as f64;

        // Citizens age
        citizens.tick(dt / 365.0);

        // Traffic simulation
        traffic.tick(dt);

        // Weather evolves
        weather.tick(dt, tick as u64);

        // Ecosystem grows
        ecosystem.tick(dt, |_pos| 0.1, season.crop_growth_rate());

        // Agent movement
        agents.tick(dt);

        // Routines (hour cycles through 0-23)
        let hour = (tick % 24) as f32;
        let _movements = vox_sim::routines::update_routines(citizens.all_mut(), hour);

        // Supply chain
        supply_chain.tick(&buildings, dt);

        // Budget tick
        budget.funds += budget.net() * (dt as f64 / 3600.0);

        // Disasters tick
        disasters.tick(dt);

        // Record history every 10 ticks
        if tick % 10 == 0 {
            history.record_tick(
                time,
                citizens.count() as u32,
                budget.funds,
                budget.total_income(),
                budget.total_expenses(),
                0.7,
                0.1,
                0.2,
                0.3,
            );
        }

        // Trade
        trade.update_prices();

        // Zoning demand
        zoning.update_demand(citizens.count() as u32);

        // Advisor (every 20 ticks)
        if tick % 20 == 0 {
            advisor.evaluate(
                citizens.count() as u32,
                budget.funds,
                0.1,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                0.2,
            );
        }

        // Milestones
        milestones.check(citizens.count() as u32);

        // Deterministic recorder
        recorder.record_tick(tick as u64, vec![], tick as u64 + 100);
    }

    println!("100 ticks complete.\n");

    // ---------------------------------------------------------------
    // 3. Print final stats from every system
    // ---------------------------------------------------------------
    println!("=== Final City Stats ===");
    println!("  Population:      {}", citizens.count());
    println!("  Buildings:       {}", buildings.buildings.len());
    println!("  Road segments:   {}", roads.segment_count());
    println!("  Road length:     {:.0}m", roads.total_length());
    println!("  Service cost:    {:.0}/tick", services.total_cost());
    println!("  Funds:           {:.0}", budget.funds);
    println!("  Net income:      {:.0}", budget.net());
    println!("  Traffic segs:    {}", traffic.segments.len());
    println!("  Transport rev:   {:.0}/hr", transport.total_hourly_revenue(2.5, 0.6));
    println!("  Power nodes:     {}", power.nodes.len());
    println!("  Water nodes:     {}", water_net.nodes.len());
    println!("  Active disasters:{}", disasters.active.len());
    println!("  Land value (0,0):{:.2}", land_value.sample(0.0, 0.0));
    println!("  Districts:       {}", districts.districts.len());
    println!("  Advisor msgs:    {}", advisor.messages.len());
    println!("  Era:             {}", milestones.current_era.label());
    println!("  Season:          {:?}", season);
    println!("  Weather:         {:?} {:.1}C", weather.current, weather.temperature);
    println!("  Trees:           {}", ecosystem.count());
    println!("  Resources:       {} types", supply_chain.stocks.len());
    println!("  Trade partners:  {}", trade.partners.len());
    println!("  Vehicles:        {}", vehicles.vehicles.len());
    println!("  Agents:          {}", agents.count());
    println!("  BDI agents:      {}", bdi_agents.len());
    println!("  Council members: {}", council.members.len());
    println!("  Shard count:     {}", shards.shard_count());
    println!("  History points:  {}", history.population.len());
    println!("  Replay ticks:    {}", if recorder.replay_tick(99).is_some() { "OK" } else { "FAIL" });
    println!("  Zone plots:      {}", zoning.plot_count());
    println!("  Pollution @ind:  {:.2}", pollution.air_at(200.0, 0.0));

    println!("\nAll 31 vox_sim modules exercised successfully.");
}
