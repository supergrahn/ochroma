//! ECS integration for the Forge procedural generation system.
//!
//! ## Rule-driven generation
//! Attach `ProcGenComponent` to any entity. The next `forge_pcg_system` tick
//! will run `emit_splats` and insert a `ProcGenResultComponent` with the result.
//!
//! ## Organic / advanced generation
//! Attach `AdvancedProcGenComponent` to any entity. The next
//! `advanced_forge_system` tick will call the appropriate generator and insert
//! `ProcGenResultComponent`.
//!
//! Both systems are idempotent: they only process entities that do NOT yet have
//! a `ProcGenResultComponent`.

use bevy_ecs::prelude::*;
use vox_core::types::GaussianSplat;

use crate::proc_gs::{emit_splats, SplatRule};
use crate::proc_gs_advanced::{generate_bench, generate_tree};

// ── Components ─────────────────────────────────────────────────────────────

/// Marks an entity for rule-driven Gaussian splat generation.
///
/// Attach this component on spawn. `forge_pcg_system` will run `emit_splats`
/// once and insert `ProcGenResultComponent`. After that the entity is skipped.
#[derive(Component, Debug, Clone)]
pub struct ProcGenComponent {
    pub rule: SplatRule,
    pub seed: u64,
}

/// Selects which advanced (organic) generator to run.
#[derive(Component, Debug, Clone)]
pub enum AdvancedProcGenComponent {
    Tree {
        seed: u64,
        height: f32,
        canopy_radius: f32,
    },
    Bench {
        seed: u64,
    },
}

/// Written by `forge_pcg_system` or `advanced_forge_system` after generation.
///
/// Presence of this component on an entity means generation is complete.
/// Both systems use `Without<ProcGenResultComponent>` to skip already-generated
/// entities — so generation runs exactly once per entity.
#[derive(Component, Debug, Clone)]
pub struct ProcGenResultComponent {
    pub splats: Vec<GaussianSplat>,
}

// ── Systems ────────────────────────────────────────────────────────────────

/// For each entity with `ProcGenComponent` but no `ProcGenResultComponent`,
/// run `emit_splats` and insert the result.
///
/// Idempotent: entities that already have `ProcGenResultComponent` are skipped.
pub fn forge_pcg_system(
    mut commands: Commands,
    query: Query<(Entity, &ProcGenComponent), Without<ProcGenResultComponent>>,
) {
    for (entity, proc_gen) in query.iter() {
        let splats = emit_splats(&proc_gen.rule, proc_gen.seed);
        commands.entity(entity).insert(ProcGenResultComponent { splats });
    }
}

/// For each entity with `AdvancedProcGenComponent` but no `ProcGenResultComponent`,
/// dispatch to the appropriate organic generator and insert the result.
///
/// Idempotent: entities that already have `ProcGenResultComponent` are skipped.
pub fn advanced_forge_system(
    mut commands: Commands,
    query: Query<(Entity, &AdvancedProcGenComponent), Without<ProcGenResultComponent>>,
) {
    for (entity, advanced) in query.iter() {
        let splats = match advanced {
            AdvancedProcGenComponent::Tree { seed, height, canopy_radius } => {
                generate_tree(*seed, *height, *canopy_radius)
            }
            AdvancedProcGenComponent::Bench { seed } => {
                generate_bench(*seed)
            }
        };
        commands.entity(entity).insert(ProcGenResultComponent { splats });
    }
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that chains `forge_pcg_system` then `advanced_forge_system`
/// in `Update`.
///
/// Both systems are stateless — no Resources are inserted. Generation runs
/// exactly once per entity (idempotent via `Without<ProcGenResultComponent>`).
pub struct ForgePlugin;

impl bevy_app::Plugin for ForgePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            bevy_app::Update,
            (forge_pcg_system, advanced_forge_system).chain(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proc_gs::{
        GeometryConfig, GeometryStrategy, MaterialZoneConfig, RuleHeader, VariationConfig,
    };

    fn minimal_rule() -> SplatRule {
        SplatRule {
            header: RuleHeader {
                asset_type: "test".to_string(),
                style: "plain".to_string(),
            },
            geometry: GeometryConfig {
                strategy: GeometryStrategy::StructuredPlacement,
                floor_count_min: 1,
                floor_count_max: 1,
                height_min: 3.0,
                height_max: 3.0,
                width_min: 5.0,
                width_max: 5.0,
                depth_min: 5.0,
                depth_max: 5.0,
                splats_per_sqm: 1.0,
            },
            material_zones: vec![],
            variation: VariationConfig {
                scale_min: 0.1,
                scale_max: 0.2,
                opacity_min: 0.8,
                opacity_max: 1.0,
            },
        }
    }

    #[test]
    fn proc_gen_component_stores_rule_and_seed() {
        let rule = minimal_rule();
        let comp = ProcGenComponent {
            rule: rule.clone(),
            seed: 42,
        };
        assert_eq!(comp.seed, 42);
        assert_eq!(comp.rule.header.asset_type, "test");
    }

    #[test]
    fn advanced_proc_gen_tree_variant() {
        let comp = AdvancedProcGenComponent::Tree {
            seed: 7,
            height: 5.0,
            canopy_radius: 2.0,
        };
        if let AdvancedProcGenComponent::Tree { height, .. } = comp {
            assert_eq!(height, 5.0);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn proc_gen_result_holds_splats() {
        let result = ProcGenResultComponent {
            splats: vec![],
        };
        assert!(result.splats.is_empty());
    }

    #[test]
    fn forge_pcg_system_generates_result() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();

        let entity = world.spawn(ProcGenComponent {
            rule: minimal_rule(),
            seed: 1,
        }).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(forge_pcg_system);
        schedule.run(&mut world);

        let result = world.entity(entity).get::<ProcGenResultComponent>();
        assert!(result.is_some(), "ProcGenResultComponent should be inserted after system runs");
    }

    #[test]
    fn forge_pcg_system_is_idempotent() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();

        // Entity already has a result — should NOT be re-generated
        let entity = world.spawn((
            ProcGenComponent { rule: minimal_rule(), seed: 2 },
            ProcGenResultComponent { splats: vec![] },
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(forge_pcg_system);
        schedule.run(&mut world);

        // Result component still present (not replaced)
        let result = world.entity(entity).get::<ProcGenResultComponent>().unwrap();
        // splats remains empty because we pre-inserted an empty result
        assert!(result.splats.is_empty(), "pre-existing result should not be replaced");
    }

    #[test]
    fn advanced_forge_system_generates_tree() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();

        let entity = world.spawn(AdvancedProcGenComponent::Tree {
            seed: 10,
            height: 5.0,
            canopy_radius: 2.0,
        }).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(advanced_forge_system);
        schedule.run(&mut world);

        let result = world.entity(entity).get::<ProcGenResultComponent>();
        assert!(result.is_some(), "ProcGenResultComponent should be inserted for tree");
        assert!(
            !result.unwrap().splats.is_empty(),
            "tree generation should produce at least one splat"
        );
    }

    #[test]
    fn advanced_forge_system_generates_bench() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();

        let entity = world.spawn(AdvancedProcGenComponent::Bench { seed: 20 }).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(advanced_forge_system);
        schedule.run(&mut world);

        let result = world.entity(entity).get::<ProcGenResultComponent>();
        assert!(result.is_some(), "ProcGenResultComponent should be inserted for bench");
        assert!(
            !result.unwrap().splats.is_empty(),
            "bench generation should produce at least one splat"
        );
    }

    #[test]
    fn plugin_registers_systems() {
        use bevy_app::App;
        // Plugin should build without panicking.
        let mut app = App::new();
        app.add_plugins(ForgePlugin);
        // No assertion needed — panicking during build counts as failure.
    }
}
