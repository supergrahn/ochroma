//! `CitySim` — a high-level facade that constructs a small but *populated* city and
//! advances every major subsystem coherently on each tick.
//!
//! The point of this module is to stop the sim from ticking an empty map. A freshly
//! constructed [`CitySim`] already contains zones, developed buildings (housing + jobs),
//! services, a road network, citizens, spatial agents bound to those citizens, and a
//! city budget. [`CitySim::tick`] then drives all of those subsystems in a deterministic,
//! causally ordered sequence so that population, employment, the economy and agent
//! positions actually evolve.
//!
//! Tick ordering (each step feeds the next):
//!   1. housing/employment matching (citizens claim residences and jobs)
//!   2. education pipeline (students graduate, unlocking better jobs)
//!   3. citizen aging + needs/satisfaction update
//!   4. economy: tax income from population + commercial/industrial buildings
//!   5. migration: attract newcomers when the city is attractive and has vacant housing
//!   6. agents: every citizen-agent walks one step toward its assigned workplace
//!
//! This is engine-internal game logic (a city sim), deliberately living in `vox_sim`
//! (the game/sim layer), never in an engine crate.

use glam::Vec3;
use uuid::Uuid;
use vox_core::lwc::WorldCoord;

use crate::agent::AgentManager;
use crate::buildings::{BuildingManager, BuildingType};
use crate::citizen::CitizenManager;
use crate::economy::CityBudget;
use crate::employment::{
    calculate_crime_rate, match_employment, match_housing, process_education,
};
use crate::migration::MigrationSystem;
use crate::roads::{RoadNetwork, RoadType};
use crate::services::{ServiceManager, ServiceType};
use crate::zoning::{ZoneType, ZoningManager};

/// A snapshot of headline city metrics, returned after a batch of ticks so callers
/// (tests, future UI/binary wiring) can observe real evolution without poking internals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CityStats {
    pub population: u32,
    pub employed: u32,
    pub housed: u32,
    pub funds: f64,
    pub net_budget: f64,
    pub crime_rate: f32,
    pub mean_satisfaction: f32,
    /// Number of citizen-agents currently moving toward a workplace.
    pub agents_commuting: usize,
    /// Game-seconds elapsed since construction.
    pub elapsed_secs: f64,
}

/// Links a citizen to its spatial agent and the world position of its workplace,
/// so the agent has a concrete destination to commute toward.
struct Commuter {
    citizen_id: u32,
    agent_id: Uuid,
    home: WorldCoord,
    workplace: Option<WorldCoord>,
}

/// The populated city facade.
pub struct CitySim {
    pub zoning: ZoningManager,
    pub buildings: BuildingManager,
    pub services: ServiceManager,
    pub roads: RoadNetwork,
    pub citizens: CitizenManager,
    pub agents: AgentManager,
    pub budget: CityBudget,
    pub migration: MigrationSystem,

    /// Building id -> world position, so we can route agents to their job.
    building_positions: Vec<(u32, WorldCoord)>,
    commuters: Vec<Commuter>,

    next_residence_anchor: f32,
    elapsed_secs: f64,
    tick_index: u64,
}

impl CitySim {
    /// Game-seconds advanced per [`CitySim::tick`] step. One tick ~ a slice of a day;
    /// 100 ticks is a few in-game days at this scale, enough for visible evolution
    /// while keeping agent movement and aging meaningful.
    pub const SECONDS_PER_TICK: f32 = 60.0;

    /// Years of aging applied per tick (kept small so a 100-tick run nudges lifecycles
    /// without instantly killing the founding population).
    pub const YEARS_PER_TICK: f32 = 0.02;

    /// Construct a small, fully populated city: a grid of residential / commercial /
    /// industrial zones, developed buildings, core services, a road spine, founding
    /// citizens with bound spatial agents, and a seeded budget.
    pub fn new_small() -> Self {
        let mut sim = Self {
            zoning: ZoningManager::new(),
            buildings: BuildingManager::new(),
            services: ServiceManager::new(),
            roads: RoadNetwork::new(),
            citizens: CitizenManager::new(),
            agents: AgentManager::new(),
            budget: CityBudget::default(),
            migration: MigrationSystem::new(),
            building_positions: Vec::new(),
            commuters: Vec::new(),
            next_residence_anchor: 0.0,
            elapsed_secs: 0.0,
            tick_index: 0,
        };

        sim.build_zones_and_buildings();
        sim.build_roads();
        sim.build_services();
        sim.seed_population(24);

        sim
    }

    /// Lay out a compact zoned core and immediately develop a building on each plot,
    /// so housing and jobs exist from tick zero (no empty map).
    fn build_zones_and_buildings(&mut self) {
        // (zone, building, capacity, x grid offset). Lay plots along a row.
        let layout: [(ZoneType, BuildingType, u32); 8] = [
            (ZoneType::ResidentialMed, BuildingType::Residential, 6),
            (ZoneType::ResidentialMed, BuildingType::Residential, 6),
            (ZoneType::ResidentialLow, BuildingType::Residential, 4),
            (ZoneType::CommercialLocal, BuildingType::Commercial, 8),
            (ZoneType::CommercialLocal, BuildingType::Commercial, 8),
            (ZoneType::IndustrialLight, BuildingType::Industrial, 10),
            (ZoneType::IndustrialLight, BuildingType::Industrial, 10),
            (ZoneType::ResidentialMed, BuildingType::Residential, 6),
        ];

        for (i, (zone, btype, cap)) in layout.iter().enumerate() {
            let x = i as f32 * 50.0;
            let pos2 = [x, 0.0];
            let plot_id = self.zoning.zone_plot(*zone, pos2, [40.0, 40.0]);
            let building_id = self.buildings.add_building(*btype, pos2, *cap);
            self.zoning.develop_plot(plot_id, building_id);

            let world = WorldCoord::from_absolute(x as f64, 0.0, 0.0);
            self.building_positions.push((building_id, world));
        }
    }

    /// A road spine built as two avenue segments meeting at the city centre, plus a cross
    /// street whose endpoint coincides with that centre. The shared endpoints trigger the
    /// network's auto-intersection logic, so the road subsystem is genuinely connected.
    fn build_roads(&mut self) {
        let span = 8.0 * 50.0;
        let mid = span * 0.5;
        // West half then east half of the avenue, meeting exactly at the centre point.
        self.roads.add_straight(
            RoadType::Avenue,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(mid, 0.0, 0.0),
        );
        self.roads.add_straight(
            RoadType::Avenue,
            Vec3::new(mid, 0.0, 0.0),
            Vec3::new(span, 0.0, 0.0),
        );
        // Cross street whose end shares the centre endpoint -> intersection forms here.
        self.roads.add_straight(
            RoadType::LocalStreet,
            Vec3::new(mid, 0.0, -50.0),
            Vec3::new(mid, 0.0, 0.0),
        );
    }

    /// Place the services that the education + safety pipelines depend on.
    fn build_services(&mut self) {
        self.services
            .place_service(ServiceType::PrimarySchool, [100.0, 0.0]);
        self.services
            .place_service(ServiceType::SecondarySchool, [150.0, 0.0]);
        self.services
            .place_service(ServiceType::Clinic, [120.0, 0.0]);
        self.services
            .place_service(ServiceType::PoliceStation, [80.0, 0.0]);
    }

    /// Spawn founding citizens of working age, each with a bound spatial agent placed at
    /// its (eventual) home. Ages are spread deterministically across the working band.
    fn seed_population(&mut self, count: u32) {
        for i in 0..count {
            // Deterministic age spread across 19..=58, all working-age so they seek jobs.
            let age = 19.0 + (i % 40) as f32;
            let home_x = (i % 8) as f32 * 50.0;
            let home = WorldCoord::from_absolute(home_x as f64, 0.0, 0.0);

            let citizen_id = self.citizens.spawn(age, None);
            // Walking speed ~1.4 m/s.
            let agent_id = self.agents.spawn(home, 1.4);

            self.commuters.push(Commuter {
                citizen_id,
                agent_id,
                home,
                workplace: None,
            });
        }
        self.next_residence_anchor = count as f32 * 50.0;
    }

    /// How many residential vacancies remain right now.
    fn vacant_housing(&self) -> u32 {
        self.buildings
            .buildings
            .iter()
            .filter(|b| b.building_type == BuildingType::Residential)
            .map(|b| b.capacity.saturating_sub(b.occupants))
            .sum()
    }

    /// Number of citizens currently in employment.
    pub fn employed_count(&self) -> u32 {
        self.citizens
            .all()
            .iter()
            .filter(|c| c.employment.is_some())
            .count() as u32
    }

    /// Number of citizens with an assigned residence.
    pub fn housed_count(&self) -> u32 {
        self.citizens
            .all()
            .iter()
            .filter(|c| c.residence.is_some())
            .count() as u32
    }

    /// Mean satisfaction across the population (0 if empty).
    pub fn mean_satisfaction(&self) -> f32 {
        let all = self.citizens.all();
        if all.is_empty() {
            return 0.0;
        }
        all.iter().map(|c| c.satisfaction).sum::<f32>() / all.len() as f32
    }

    /// Count of commercial buildings (revenue drivers for the budget).
    fn commercial_count(&self) -> u32 {
        self.buildings
            .buildings
            .iter()
            .filter(|b| b.building_type == BuildingType::Commercial)
            .count() as u32
    }

    /// Count of industrial buildings.
    fn industrial_count(&self) -> u32 {
        self.buildings
            .buildings
            .iter()
            .filter(|b| b.building_type == BuildingType::Industrial)
            .count() as u32
    }

    /// Refresh each commuter's workplace destination from its citizen's current job, and
    /// (re)point its bound agent at that destination. Returns the number of commuters that
    /// now have a workplace to walk to.
    fn assign_commute_destinations(&mut self) -> usize {
        // Snapshot (citizen_id -> workplace building id) to avoid borrow conflicts.
        let jobs: Vec<(u32, Option<u32>)> = self
            .citizens
            .all()
            .iter()
            .map(|c| (c.id, c.workplace))
            .collect();

        let mut commuting = 0usize;
        for commuter in self.commuters.iter_mut() {
            let workplace_building = jobs
                .iter()
                .find(|(cid, _)| *cid == commuter.citizen_id)
                .and_then(|(_, w)| *w);

            let dest = workplace_building.and_then(|bid| {
                self.building_positions
                    .iter()
                    .find(|(id, _)| *id == bid)
                    .map(|(_, w)| *w)
            });

            commuter.workplace = dest;
            if let Some(dest) = dest
                && let Some(agent) = self.agents.get_mut(commuter.agent_id) {
                    // Only (re)assign if the agent isn't already heading there / parked there.
                    let at_dest = agent.destination.is_none()
                        && Self::same_spot(agent.position, dest);
                    if !at_dest {
                        agent.destination = Some(dest);
                    }
                    commuting += 1;
                }
        }
        commuting
    }

    fn same_spot(a: WorldCoord, b: WorldCoord) -> bool {
        let (ax, _, az) = a.to_absolute();
        let (bx, _, bz) = b.to_absolute();
        (ax - bx).abs() < 0.5 && (az - bz).abs() < 0.5
    }

    /// Whether `has_*` school services exist (drives the education pipeline).
    fn has_service(&self, st: ServiceType) -> bool {
        self.services
            .buildings
            .iter()
            .any(|b| b.service_type == st)
    }

    /// Police coverage fraction at the city centre (drives the crime calculation).
    fn police_coverage(&self) -> f32 {
        if self.has_service(ServiceType::PoliceStation) {
            0.6
        } else {
            0.0
        }
    }

    /// Advance the entire city by `n` ticks and return fresh headline stats.
    pub fn tick(&mut self, n: u32) -> CityStats {
        for _ in 0..n {
            self.step();
        }
        self.stats()
    }

    /// One coherent simulation step across all major subsystems.
    fn step(&mut self) {
        let dt_secs = Self::SECONDS_PER_TICK;
        let dt_years = Self::YEARS_PER_TICK;

        // 1. Housing then employment matching. Citizens must have a residence before a job
        //    so the proximity heuristic in `match_employment` has a meaningful anchor.
        {
            let citizens = self.citizens.all_mut();
            match_housing(citizens, &mut self.buildings);
        }
        {
            let citizens = self.citizens.all_mut();
            match_employment(citizens, &mut self.buildings);
        }

        // Mirror each citizen's new residence onto its commuter home position so that, if
        // they later lose/leave a job, the agent still has a sensible anchor.
        self.sync_commuter_homes();

        // 2. Education pipeline (students -> primary/secondary; young workers -> university).
        {
            let has_primary = self.has_service(ServiceType::PrimarySchool)
                || self.has_service(ServiceType::School);
            let has_secondary = self.has_service(ServiceType::SecondarySchool)
                || self.has_service(ServiceType::School);
            let has_university = self.has_service(ServiceType::University);
            let citizens = self.citizens.all_mut();
            process_education(citizens, has_primary, has_secondary, has_university);
        }

        // 3. Age citizens, recompute needs/satisfaction.
        self.citizens.tick(dt_years);

        // 4. Economy: tax income from current population + commercial/industrial activity.
        let population = self.citizens.count() as u32;
        let commercial = self.commercial_count();
        let industrial = self.industrial_count();
        self.budget.services_expenses = self.services.total_cost();
        self.budget
            .tick(population, commercial, industrial);

        // 5. Migration: attract newcomers when the city beats the region and has housing.
        let city_satisfaction = self.mean_satisfaction();
        let available_housing = self.vacant_housing();
        let (arrivals, departures) = self.migration.calculate_migration(
            &self.citizens,
            city_satisfaction,
            available_housing,
            dt_secs,
        );
        if arrivals > 0 {
            self.spawn_arrivals(arrivals);
        }
        if departures > 0 {
            self.remove_departures(departures);
        }

        // 6. Agents: point each employed citizen's agent at its workplace and step movement.
        self.assign_commute_destinations();
        self.agents.tick(dt_secs);

        self.elapsed_secs += dt_secs as f64;
        self.tick_index += 1;
    }

    /// Copy citizen residence anchors onto commuter homes (best-effort spatial anchor).
    fn sync_commuter_homes(&mut self) {
        let homes: Vec<(u32, Option<u32>)> = self
            .citizens
            .all()
            .iter()
            .map(|c| (c.id, c.residence))
            .collect();
        let positions = self.building_positions.clone();
        for commuter in self.commuters.iter_mut() {
            if let Some((_, Some(res_building))) =
                homes.iter().find(|(cid, _)| *cid == commuter.citizen_id)
                && let Some((_, w)) = positions.iter().find(|(id, _)| id == res_building) {
                    commuter.home = *w;
                }
        }
    }

    /// New residents arrive: spawn working-age citizens with bound agents at a fresh anchor.
    fn spawn_arrivals(&mut self, arrivals: u32) {
        for _ in 0..arrivals {
            let age = 22.0 + (self.tick_index % 30) as f32;
            let home_x = self.next_residence_anchor;
            self.next_residence_anchor += 50.0;
            let home = WorldCoord::from_absolute(home_x as f64, 0.0, 0.0);
            let citizen_id = self.citizens.spawn(age, None);
            let agent_id = self.agents.spawn(home, 1.4);
            self.commuters.push(Commuter {
                citizen_id,
                agent_id,
                home,
                workplace: None,
            });
        }
    }

    /// Departures: remove the least-satisfied citizens (and their bound agents/commuters),
    /// and free up the building occupancy they held.
    fn remove_departures(&mut self, departures: u32) {
        let mut order: Vec<(u32, f32)> = self
            .citizens
            .all()
            .iter()
            .map(|c| (c.id, c.satisfaction))
            .collect();
        order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let leaving: Vec<u32> = order
            .into_iter()
            .take(departures as usize)
            .map(|(id, _)| id)
            .collect();
        if leaving.is_empty() {
            return;
        }

        // Free building occupancy held by leavers.
        for cid in &leaving {
            if let Some(c) = self.citizens.get(*cid) {
                for bid in [c.residence, c.employment].into_iter().flatten() {
                    if let Some(b) =
                        self.buildings.buildings.iter_mut().find(|b| b.id == bid)
                    {
                        b.occupants = b.occupants.saturating_sub(1);
                    }
                }
            }
        }

        // Despawn their agents and drop their commuter records.
        self.commuters.retain(|cm| {
            if leaving.contains(&cm.citizen_id) {
                self.agents.remove(cm.agent_id);
                false
            } else {
                true
            }
        });

        // Remove the citizens themselves.
        self.citizens.remove_many(&leaving);
    }

    /// Headline metrics snapshot.
    pub fn stats(&self) -> CityStats {
        let population = self.citizens.count() as u32;
        let employed = self.employed_count();
        let crime_rate =
            calculate_crime_rate(population, employed, self.police_coverage());
        let agents_commuting = self
            .agents
            .iter()
            .filter(|a| a.destination.is_some())
            .count();

        CityStats {
            population,
            employed,
            housed: self.housed_count(),
            funds: self.budget.funds,
            net_budget: self.budget.net(),
            crime_rate,
            mean_satisfaction: self.mean_satisfaction(),
            agents_commuting,
            elapsed_secs: self.elapsed_secs,
        }
    }

    /// World position of a citizen's bound agent, for tests/inspection.
    pub fn agent_position_of(&self, citizen_id: u32) -> Option<WorldCoord> {
        let agent_id = self
            .commuters
            .iter()
            .find(|c| c.citizen_id == citizen_id)
            .map(|c| c.agent_id)?;
        self.agents.get(agent_id).map(|a| a.position)
    }

    /// The workplace destination assigned to a citizen's commute, if any.
    pub fn workplace_of(&self, citizen_id: u32) -> Option<WorldCoord> {
        self.commuters
            .iter()
            .find(|c| c.citizen_id == citizen_id)
            .and_then(|c| c.workplace)
    }

    /// First citizen id (founding population), for tests.
    pub fn first_citizen_id(&self) -> Option<u32> {
        self.citizens.all().first().map(|c| c.id)
    }
}

impl Default for CitySim {
    fn default() -> Self {
        Self::new_small()
    }
}
