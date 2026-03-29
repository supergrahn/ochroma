# Forge PCG ECS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `emit_splats` rule engine and `generate_tree`/`generate_bench` organic generators into bevy_ecs so any entity with a `ProcGenComponent` or `AdvancedProcGenComponent` automatically has its `ProcGenResultComponent` populated with generated `GaussianSplat`s.

**Architecture:** A new `vox_data::forge_ecs` module provides `ForgePlugin`, two systems (`forge_pcg_system` → `advanced_forge_system`), and three components (`ProcGenComponent`, `AdvancedProcGenComponent`, `ProcGenResultComponent`). Systems use a `Without<ProcGenResultComponent>` filter for idempotency — generation only runs once per entity. Both generators are stateless (no Resource needed), so `ForgePlugin` only wires the systems.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"` (added to `vox_data/Cargo.toml`), `vox_data::proc_gs::{emit_splats, SplatRule}`, `vox_data::proc_gs_advanced::{generate_tree, generate_bench}`, `vox_core::types::GaussianSplat`

---

## Key Files (read before editing)

- `crates/vox_data/src/proc_gs.rs` — `emit_splats(rule: &SplatRule, seed: u64) -> Vec<GaussianSplat>`; `SplatRule { header: RuleHeader, geometry: GeometryConfig, material_zones: Vec<MaterialZoneConfig>, variation: VariationConfig }`
- `crates/vox_data/src/proc_gs_advanced.rs` — `generate_tree(seed: u64, height: f32, canopy_radius: f32) -> Vec<GaussianSplat>`; `generate_bench(seed: u64) -> Vec<GaussianSplat>`
- `crates/vox_data/Cargo.toml` — add bevy_ecs + bevy_app
- `crates/vox_data/src/lib.rs` — add `pub mod forge_ecs;`

## File Structure

**Create:**
- `crates/vox_data/src/forge_ecs.rs` — components + systems + plugin (single file)

**Modify:**
- `crates/vox_data/Cargo.toml` — add `bevy_ecs` + `bevy_app` workspace deps
- `crates/vox_data/src/lib.rs` — add `pub mod forge_ecs;`

---

### Task 1: ProcGenComponent + AdvancedProcGenComponent + ProcGenResultComponent

**Files:**
- Modify: `crates/vox_data/Cargo.toml`
- Modify: `crates/vox_data/src/lib.rs`
- Create: `crates/vox_data/src/forge_ecs.rs`

- [ ] **Step 1: Add bevy_ecs + bevy_app to `crates/vox_data/Cargo.toml`**

Open the file. After the `rand = "0.9"` line, add:

```toml
bevy_ecs = { workspace = true }
bevy_app = { workspace = true }
```

- [ ] **Step 2: Add `pub mod forge_ecs;` to `crates/vox_data/src/lib.rs`**

Open the file. After `pub mod proc_gs_advanced;` add:

```rust
pub mod forge_ecs;
```

- [ ] **Step 3: Create `crates/vox_data/src/forge_ecs.rs`** with this exact content:

```rust
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
}
```

- [ ] **Step 4: Verify compile**

```bash
cargo check -p vox_data 2>&1 | tail -5
```
Expected: clean (no errors)

If there are compile errors about `GeometryConfig`, `RuleHeader`, `VariationConfig`, or `GeometryStrategy`, read `crates/vox_data/src/proc_gs.rs` to check exact field names and adjust.

- [ ] **Step 5: Run tests**

```bash
cargo test -p vox_data --lib -- forge_ecs 2>&1 | tail -5
```
Expected: `3 passed; 0 failed`

- [ ] **Step 6: Commit**

```bash
git add crates/vox_data/Cargo.toml crates/vox_data/src/lib.rs crates/vox_data/src/forge_ecs.rs
git commit -m "feat(forge): ProcGenComponent + AdvancedProcGenComponent + ProcGenResultComponent"
```

---

### Task 2: forge_pcg_system

**Files:**
- Modify: `crates/vox_data/src/forge_ecs.rs`

- [ ] **Step 1: Add failing tests** — append inside `#[cfg(test)] mod tests` BEFORE closing `}`:

```rust
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

        // Result component must be inserted
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
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_data --lib -- forge_ecs::tests::forge_pcg_system_generates_result 2>&1 | tail -5
```
Expected: FAIL — `forge_pcg_system` not defined

- [ ] **Step 3: Implement** — insert BEFORE `#[cfg(test)]` in `forge_ecs.rs`:

```rust
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
```

- [ ] **Step 4: Run all forge_ecs tests**

```bash
cargo test -p vox_data --lib -- forge_ecs 2>&1 | tail -8
```
Expected: `5 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/forge_ecs.rs
git commit -m "feat(forge): forge_pcg_system — emit_splats on ProcGenComponent entities"
```

---

### Task 3: advanced_forge_system + ForgePlugin

**Files:**
- Modify: `crates/vox_data/src/forge_ecs.rs`

- [ ] **Step 1: Add failing tests** — append inside `#[cfg(test)] mod tests` BEFORE closing `}`:

```rust
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
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_data --lib -- forge_ecs::tests::advanced_forge_system_generates_tree 2>&1 | tail -5
```
Expected: FAIL — `advanced_forge_system`, `ForgePlugin` not defined

- [ ] **Step 3: Implement** — insert after `forge_pcg_system` and BEFORE `#[cfg(test)]`:

```rust
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
```

- [ ] **Step 4: Run all forge_ecs tests**

```bash
cargo test -p vox_data --lib -- forge_ecs 2>&1 | tail -10
```
Expected: `8 passed; 0 failed`

- [ ] **Step 5: Full vox_data suite — no regressions**

```bash
cargo test -p vox_data --lib 2>&1 | grep "test result"
```
Expected: `0 failed`

- [ ] **Step 6: Commit**

```bash
git add crates/vox_data/src/forge_ecs.rs
git commit -m "feat(forge): advanced_forge_system + ForgePlugin — complete Forge PCG ECS integration"
```

---

## Self-Review

**Spec coverage:**
- ✅ `ProcGenComponent { rule: SplatRule, seed: u64 }` → Task 1
- ✅ `AdvancedProcGenComponent::Tree { seed, height, canopy_radius }` → Task 1
- ✅ `AdvancedProcGenComponent::Bench { seed }` → Task 1
- ✅ `ProcGenResultComponent { splats: Vec<GaussianSplat> }` → Task 1
- ✅ `forge_pcg_system` — emit_splats once per entity, idempotent → Task 2
- ✅ `advanced_forge_system` — tree/bench dispatch, idempotent → Task 3
- ✅ `ForgePlugin` — chains both systems in `Update` → Task 3
- ✅ Test: `forge_pcg_system_generates_result` → Task 2
- ✅ Test: `forge_pcg_system_is_idempotent` → Task 2
- ✅ Test: `advanced_forge_system_generates_tree` → Task 3
- ✅ Test: `advanced_forge_system_generates_bench` → Task 3
- ✅ Test: `plugin_registers_systems` → Task 3

**Placeholder scan:** No TBDs. All function bodies shown in full.

**Type consistency:**
- `ProcGenResultComponent` defined Task 1, used as Without filter in Tasks 2 & 3 ✅
- `emit_splats(&proc_gen.rule, proc_gen.seed)` matches `pub fn emit_splats(rule: &SplatRule, seed: u64)` ✅
- `generate_tree(*seed, *height, *canopy_radius)` matches `pub fn generate_tree(seed: u64, height: f32, canopy_radius: f32)` ✅
- `generate_bench(*seed)` matches `pub fn generate_bench(seed: u64)` ✅
- `AdvancedProcGenComponent` match arms dereference with `*seed`, `*height`, `*canopy_radius` — required because enum fields are references in pattern matching ✅
- `minimal_rule()` helper defined in tests Task 1, reused in Task 2 (same module, same file) ✅
