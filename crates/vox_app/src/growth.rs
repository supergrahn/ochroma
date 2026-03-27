use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use uuid::Uuid;
use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};
use vox_data::proc_gs::{
    SplatRule, RuleHeader, GeometryConfig, GeometryStrategy, MaterialZoneConfig, VariationConfig,
    emit_splats,
};
use crate::simulation::SimulationState;

static NEXT_BUILDING_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(5000);

/// Build a SplatRule for a residential building.
fn residential_rule() -> SplatRule {
    SplatRule {
        header: RuleHeader {
            asset_type: "House".to_string(),
            style: "residential".to_string(),
        },
        geometry: GeometryConfig {
            strategy: GeometryStrategy::StructuredPlacement,
            floor_count_min: 2,
            floor_count_max: 3,
            height_min: 6.0,
            height_max: 10.5,
            width_min: 5.0,
            width_max: 7.0,
            depth_min: 10.0,
            depth_max: 10.0,
            splats_per_sqm: 2.0,
        },
        material_zones: vec![
            MaterialZoneConfig {
                name: "facade".to_string(),
                material_tag: "brick_red".to_string(),
                zone_type: "wall".to_string(),
                coverage: 0.8,
            },
            MaterialZoneConfig {
                name: "roof".to_string(),
                material_tag: "slate_grey".to_string(),
                zone_type: "roof".to_string(),
                coverage: 1.0,
            },
        ],
        variation: VariationConfig {
            scale_min: 0.04,
            scale_max: 0.08,
            opacity_min: 0.8,
            opacity_max: 1.0,
        },
    }
}

/// Build a SplatRule for a commercial building.
fn commercial_rule() -> SplatRule {
    SplatRule {
        header: RuleHeader {
            asset_type: "Shop".to_string(),
            style: "commercial".to_string(),
        },
        geometry: GeometryConfig {
            strategy: GeometryStrategy::StructuredPlacement,
            floor_count_min: 1,
            floor_count_max: 2,
            height_min: 4.0,
            height_max: 9.0,
            width_min: 8.0,
            width_max: 12.0,
            depth_min: 15.0,
            depth_max: 15.0,
            splats_per_sqm: 1.5,
        },
        material_zones: vec![
            MaterialZoneConfig {
                name: "facade".to_string(),
                material_tag: "concrete_raw".to_string(),
                zone_type: "wall".to_string(),
                coverage: 0.8,
            },
            MaterialZoneConfig {
                name: "roof".to_string(),
                material_tag: "asphalt_dry".to_string(),
                zone_type: "roof".to_string(),
                coverage: 1.0,
            },
        ],
        variation: VariationConfig {
            scale_min: 0.05,
            scale_max: 0.09,
            opacity_min: 0.8,
            opacity_max: 1.0,
        },
    }
}

/// System: grow buildings on undeveloped zoned plots when demand is sufficient.
/// Called periodically (not every frame — every ~5 game seconds).
pub fn growth_tick(world: &mut World) {
    // Get simulation state to check demand and plots
    let sim = world.resource::<SimulationState>();
    let demand = sim.zoning.demand.clone();
    let undeveloped: Vec<(u32, [f32; 2], vox_sim::zoning::ZoneType)> = sim.zoning.undeveloped_plots()
        .iter()
        .map(|p| (p.id, p.position, p.zone_type))
        .collect();

    if undeveloped.is_empty() { return; }

    // Only grow if there's demand
    let should_grow = demand.residential > 0.3 || demand.commercial > 0.3 || demand.industrial > 0.3;
    if !should_grow { return; }

    // Try to grow one building per tick
    let plot = undeveloped[0];

    let rule = match plot.2 {
        vox_sim::zoning::ZoneType::ResidentialLow
        | vox_sim::zoning::ZoneType::ResidentialMed
        | vox_sim::zoning::ZoneType::ResidentialHigh => residential_rule(),
        vox_sim::zoning::ZoneType::CommercialLocal
        | vox_sim::zoning::ZoneType::CommercialRegional => commercial_rule(),
        _ => return, // Don't grow industrial/other yet
    };

    let seed = plot.0 as u64 * 12345 + 42;
    let splats = emit_splats(&rule, seed);
    if splats.is_empty() { return; }

    let uuid = Uuid::new_v4();
    let splat_count = splats.len() as u32;
    let building_id = NEXT_BUILDING_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Spawn asset + instance
    world.spawn(SplatAssetComponent { uuid, splat_count, splats });
    world.spawn(SplatInstanceComponent {
        asset_uuid: uuid,
        position: Vec3::new(plot.1[0], 0.0, plot.1[1]),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        instance_id: building_id,
        lod: LodLevel::Full,
    });

    // Mark plot as developed
    let mut sim = world.resource_mut::<SimulationState>();
    sim.zoning.develop_plot(plot.0, building_id);

    println!("[ochroma] Building grew on plot {} at ({:.0}, {:.0})", plot.0, plot.1[0], plot.1[1]);
}
