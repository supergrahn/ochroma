pub mod advisor;
pub mod agent;
pub mod calendar;
pub mod city_sim;
pub mod buildings;
pub mod citizen;
pub mod disasters;
pub mod districts;
pub mod economy;
pub mod land_value;
pub mod migration;
pub mod milestones;
pub mod pollution;
pub mod roads;
pub mod routines;
pub mod seasons;
pub mod services;
pub mod supply_chain;
pub mod traffic;
pub mod transport;
pub mod utilities;
pub mod zoning;
pub mod ecosystem;
pub mod employment;
pub mod history;
pub mod trade;
pub mod vehicles;
pub mod weather;
pub mod bdi_agent;
pub mod social_network;
pub mod city_council;
pub mod sharding;
pub mod deterministic;
pub mod crowd;
pub mod spatial_hash;
pub mod sim_hash;
// The canonical shared sim vocabulary (beyond-CS2 cim-sim/economy/wellbeing
// seam, design 2026-06-24 §5 build-order A1): DistrictId/CohortId/Money/
// SkillTier/Relevance/EventKind/PresenceLedger/assert_population_conserved.
// Game-agnostic engine primitives the game re-exports for its cim/ callers.
pub mod sim_types;
