# Physics ECS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `RapierPhysicsWorld` into bevy_ecs so any entity with `ColliderComponent` + `TransformComponent` automatically gets physics simulation — position, gravity, collision.

**Architecture:** A new `vox_physics::ecs` module provides `PhysicsPlugin` (bevy_app), `PhysicsBodyComponent` (output component storing the Rapier handle), `PhysicsBodyTypeComponent` (Dynamic/Static/Kinematic intent), and three ordered systems: `spawn_physics_bodies_system` (inserts new entities into Rapier), `physics_step_system` (steps the world), `sync_transforms_system` (writes Rapier positions back to `TransformComponent`). The existing `RapierPhysicsWorld` is stored as a bevy_ecs `Resource` — no changes to its internal API.

**Tech Stack:** `rapier3d = "0.22"`, `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `vox_core::ecs::{TransformComponent, ColliderComponent, ColliderShape}`.

---

## Key File Paths (read before editing)

- `crates/vox_physics/Cargo.toml`
- `crates/vox_physics/src/lib.rs`
- `crates/vox_physics/src/rapier.rs` — `RapierPhysicsWorld` API
- `crates/vox_physics/tests/rapier_test.rs` — broken feature gate
- `crates/vox_core/src/ecs.rs` — `TransformComponent`, `ColliderComponent`, `ColliderShape`

## File Structure

**Create:**
- `crates/vox_physics/src/ecs.rs` — all ECS types and systems

**Modify:**
- `crates/vox_physics/Cargo.toml` — add `bevy_ecs`, `bevy_app`, `vox_core` deps
- `crates/vox_physics/src/lib.rs` — add `pub mod ecs;`
- `crates/vox_physics/tests/rapier_test.rs` — remove dead `#[cfg(feature = "rapier")]` wrapper

---

### Task 1: Fix rapier tests + add bevy deps

**Files:**
- Modify: `crates/vox_physics/Cargo.toml`
- Modify: `crates/vox_physics/tests/rapier_test.rs`

The rapier tests are wrapped in `#[cfg(feature = "rapier")]` but `vox_physics/Cargo.toml` defines no such feature. They never compile or run. Fix both files.

- [ ] **Step 1: Read the broken test file**

Run: `cat crates/vox_physics/tests/rapier_test.rs`

Confirm: the outer `#[cfg(feature = "rapier")] mod rapier_tests { ... }` wrapper means zero tests run under `cargo test -p vox_physics`.

- [ ] **Step 2: Run tests to confirm zero pass**

Run: `cargo test -p vox_physics 2>&1 | tail -5`
Expected output contains: `0 passed`

- [ ] **Step 3: Remove the dead cfg gate in rapier_test.rs**

Replace the entire `crates/vox_physics/tests/rapier_test.rs` with:

```rust
use vox_physics::RapierPhysicsWorld;

#[test]
fn create_world() {
    let world = RapierPhysicsWorld::new();
    assert_eq!(world.body_count(), 0);
    assert_eq!(world.collider_count(), 0);
}

#[test]
fn ball_falls_to_ground() {
    let mut world = RapierPhysicsWorld::new();
    world.add_static_collider([0.0, -1.0, 0.0], [100.0, 1.0, 100.0]);
    let (ball, _) = world.add_dynamic_sphere([0.0, 10.0, 0.0], 0.5, 1.0);
    for _ in 0..120 {
        world.step();
    }
    let pos = world.body_position(ball).unwrap();
    assert!(pos[1] < 5.0, "Ball should have fallen: y={}", pos[1]);
    assert!(pos[1] > -1.5, "Ball should be above ground: y={}", pos[1]);
}

#[test]
fn dynamic_box_falls() {
    let mut world = RapierPhysicsWorld::new();
    world.add_static_collider([0.0, -1.0, 0.0], [50.0, 1.0, 50.0]);
    let (bx, _) = world.add_dynamic_box([0.0, 5.0, 0.0], [0.5, 0.5, 0.5], 2.0);
    for _ in 0..60 {
        world.step();
    }
    let pos = world.body_position(bx).unwrap();
    assert!(pos[1] < 5.0, "Box should have fallen: y={}", pos[1]);
}

#[test]
fn character_controller() {
    let mut world = RapierPhysicsWorld::new();
    let (char_handle, _) = world.add_character_controller([0.0, 1.0, 0.0], 0.3, 1.8);
    world.set_kinematic_position(char_handle, [5.0, 1.0, 0.0]);
    world.step();
    let pos = world.body_position(char_handle).unwrap();
    assert!((pos[0] - 5.0).abs() < 0.1, "Character x={}", pos[0]);
}

#[test]
fn apply_force_changes_velocity() {
    let mut world = RapierPhysicsWorld::new();
    let (body, _) = world.add_dynamic_box([0.0, 5.0, 0.0], [0.5, 0.5, 0.5], 1.0);
    world.apply_force(body, [100.0, 0.0, 0.0]);
    world.step();
    let vel = world.body_velocity(body).unwrap();
    assert!(vel[0] > 0.0, "Body should have positive x velocity");
}

#[test]
fn raycast_hits_static() {
    let mut world = RapierPhysicsWorld::new();
    world.add_static_collider([0.0, 0.0, 0.0], [10.0, 0.5, 10.0]);
    let result = world.raycast([0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0);
    assert!(result.is_some(), "Ray should hit the ground collider");
    let (hit, dist) = result.unwrap();
    assert!(dist < 10.5, "Hit distance should be ~9.5, got {}", dist);
    assert!(hit[1].abs() < 1.0, "Hit y should be near 0, got {}", hit[1]);
}
```

- [ ] **Step 4: Add bevy_ecs, bevy_app, vox_core to Cargo.toml**

Open `crates/vox_physics/Cargo.toml`. Add to `[dependencies]`:

```toml
bevy_ecs  = { workspace = true }
bevy_app  = { workspace = true }
vox_core  = { path = "../vox_core" }
```

`vox_core` is already there — skip adding if present. Only add missing deps.

- [ ] **Step 5: Run rapier tests**

Run: `cargo test -p vox_physics 2>&1 | tail -8`
Expected: `6 passed; 0 failed` (the 6 rapier tests above plus any existing physics_test.rs tests)

- [ ] **Step 6: Commit**

```bash
git add crates/vox_physics/Cargo.toml crates/vox_physics/tests/rapier_test.rs
git commit -m "fix(physics): remove dead rapier feature gate, unlock 6 rapier tests; add bevy deps"
```

---

### Task 2: PhysicsBodyComponent + PhysicsBodyTypeComponent

**Files:**
- Create: `crates/vox_physics/src/ecs.rs`
- Modify: `crates/vox_physics/src/lib.rs`

Define the ECS components and their tests. The systems come in Task 3.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_physics/src/ecs.rs` with just the tests (no impl yet):

```rust
//! bevy_ecs integration for vox_physics.

use bevy_ecs::prelude::*;
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};

/// Body type intent, read by spawn_physics_bodies_system.
/// Default is Dynamic.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBodyTypeComponent {
    Dynamic,
    Static,
    Kinematic,
}

impl Default for PhysicsBodyTypeComponent {
    fn default() -> Self { Self::Dynamic }
}

/// Output component written by spawn_physics_bodies_system.
/// Presence signals the entity is already registered in RapierPhysicsWorld.
#[derive(Component, Debug, Clone)]
pub struct PhysicsBodyComponent {
    pub body_handle:     RigidBodyHandle,
    pub collider_handle: ColliderHandle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_type_default_is_dynamic() {
        assert_eq!(PhysicsBodyTypeComponent::default(), PhysicsBodyTypeComponent::Dynamic);
    }

    #[test]
    fn physics_body_component_stores_handles() {
        // Compile test: PhysicsBodyComponent can be inserted as a bevy_ecs Component.
        fn _check(_: PhysicsBodyComponent) {}
    }
}
```

- [ ] **Step 2: Add pub mod ecs to lib.rs**

Open `crates/vox_physics/src/lib.rs`. Add after the existing `pub mod rapier;` line:

```rust
pub mod ecs;
pub use ecs::{PhysicsBodyComponent, PhysicsBodyTypeComponent};
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p vox_physics ecs 2>&1 | tail -8`
Expected: `2 passed; 0 failed`

- [ ] **Step 4: Commit**

```bash
git add crates/vox_physics/src/ecs.rs crates/vox_physics/src/lib.rs
git commit -m "feat(physics): PhysicsBodyComponent + PhysicsBodyTypeComponent ECS components"
```

---

### Task 3: spawn_physics_bodies_system + physics_step_system + sync_transforms_system + PhysicsPlugin

**Files:**
- Modify: `crates/vox_physics/src/ecs.rs`

Add the three systems and the plugin. Then write an integration test that proves they work end-to-end.

- [ ] **Step 1: Write the failing integration test**

Replace `crates/vox_physics/src/ecs.rs` with:

```rust
//! bevy_ecs integration for vox_physics.

use bevy_ecs::prelude::*;
use bevy_app::{App, Plugin, Update};
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};
use vox_core::ecs::{ColliderComponent, ColliderShape, TransformComponent};
use crate::RapierPhysicsWorld;

/// Body type intent — read by spawn_physics_bodies_system.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBodyTypeComponent {
    Dynamic,
    Static,
    Kinematic,
}

impl Default for PhysicsBodyTypeComponent {
    fn default() -> Self { Self::Dynamic }
}

/// Output component added by spawn_physics_bodies_system.
/// Presence signals this entity is registered in RapierPhysicsWorld.
#[derive(Component, Debug, Clone)]
pub struct PhysicsBodyComponent {
    pub body_handle:     RigidBodyHandle,
    pub collider_handle: ColliderHandle,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Registers entities that have ColliderComponent + TransformComponent
/// but no PhysicsBodyComponent yet.
pub fn spawn_physics_bodies_system(
    mut commands: Commands,
    mut physics: ResMut<RapierPhysicsWorld>,
    query: Query<
        (Entity, &ColliderComponent, &TransformComponent, Option<&PhysicsBodyTypeComponent>),
        Without<PhysicsBodyComponent>,
    >,
) {
    for (entity, collider, transform, body_type) in query.iter() {
        let pos = [
            transform.position.x,
            transform.position.y,
            transform.position.z,
        ];
        let btype = body_type.copied().unwrap_or_default();

        let (body_handle, collider_handle) = match (&collider.shape, btype) {
            (ColliderShape::Box { half_extents }, PhysicsBodyTypeComponent::Static) => {
                let ch = physics.add_static_collider(pos, *half_extents);
                // Static colliders have no rigid body — use a sentinel
                // We still need a handle pair; static bodies don't have a RigidBodyHandle in
                // Rapier, so we use add_dynamic_box then immediately freeze it. Alternative:
                // store Option<RigidBodyHandle>. For simplicity here, we skip adding a body handle
                // for statics and instead produce a special-cased component.
                // Use a static kinematic body so the handle is valid:
                let (bh, _) = physics.add_character_controller(pos, 0.01, 0.01);
                // Remove the capsule collider just added, replace with the real static collider.
                // Actually just use the real collider handle we already have.
                let _ = bh; // drop — static colliders need no body
                (ch, ch) // both fields set to the collider handle as a placeholder for statics
            }
            (ColliderShape::Box { half_extents }, _) => {
                let (bh, ch) = physics.add_dynamic_box(pos, *half_extents, 1.0);
                (bh, ch)
            }
            (ColliderShape::Sphere { radius }, _) => {
                let (bh, ch) = physics.add_dynamic_sphere(pos, *radius, 1.0);
                (bh, ch)
            }
            (ColliderShape::Capsule { radius, height }, _) => {
                let (bh, ch) = physics.add_character_controller(pos, *radius, *height);
                (bh, ch)
            }
        };

        commands.entity(entity).insert(PhysicsBodyComponent {
            body_handle,
            collider_handle,
        });
    }
}

/// Step the Rapier simulation by one fixed timestep.
pub fn physics_step_system(mut physics: ResMut<RapierPhysicsWorld>) {
    physics.step();
}

/// Write Rapier body positions back to TransformComponent.
pub fn sync_transforms_system(
    physics: Res<RapierPhysicsWorld>,
    mut query: Query<(&PhysicsBodyComponent, &mut TransformComponent)>,
) {
    for (body, mut transform) in query.iter_mut() {
        if let Some(pos) = physics.body_position(body.body_handle) {
            transform.position.x = pos[0];
            transform.position.y = pos[1];
            transform.position.z = pos[2];
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Register with bevy_app to get automatic physics each update.
/// Usage: `app.add_plugins(PhysicsPlugin::default())`
pub struct PhysicsPlugin {
    pub gravity: [f32; 3],
}

impl Default for PhysicsPlugin {
    fn default() -> Self { Self { gravity: [0.0, -9.81, 0.0] } }
}

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        let mut world = RapierPhysicsWorld::new();
        world.set_gravity(self.gravity);
        app.insert_resource(world);
        app.add_systems(Update, (
            spawn_physics_bodies_system,
            physics_step_system,
            sync_transforms_system,
        ).chain());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;
    use glam::Vec3;
    use vox_core::ecs::{ColliderComponent, ColliderShape, TransformComponent};
    use crate::RapierPhysicsWorld;

    #[test]
    fn body_type_default_is_dynamic() {
        assert_eq!(PhysicsBodyTypeComponent::default(), PhysicsBodyTypeComponent::Dynamic);
    }

    #[test]
    fn spawn_system_registers_body_in_rapier() {
        // Build a bevy_ecs World manually (no full App needed for unit test).
        let mut world = World::new();
        world.insert_resource(RapierPhysicsWorld::new());

        let entity = world.spawn((
            TransformComponent {
                position: Vec3::new(0.0, 5.0, 0.0),
                ..Default::default()
            },
            ColliderComponent {
                shape: ColliderShape::Box { half_extents: [0.5, 0.5, 0.5] },
            },
            PhysicsBodyTypeComponent::Dynamic,
        )).id();

        // Run spawn system via a one-shot schedule
        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(spawn_physics_bodies_system);
        schedule.run(&mut world);

        // Entity should now have PhysicsBodyComponent
        assert!(
            world.entity(entity).contains::<PhysicsBodyComponent>(),
            "spawn_physics_bodies_system should have added PhysicsBodyComponent"
        );

        // Rapier world should have 1 body
        let physics = world.resource::<RapierPhysicsWorld>();
        assert!(physics.body_count() >= 1, "Rapier should have at least 1 body");
    }

    #[test]
    fn step_and_sync_moves_transform() {
        let mut world = World::new();
        world.insert_resource(RapierPhysicsWorld::new());

        // Spawn a dynamic sphere high up
        let entity = world.spawn((
            TransformComponent {
                position: Vec3::new(0.0, 10.0, 0.0),
                ..Default::default()
            },
            ColliderComponent {
                shape: ColliderShape::Sphere { radius: 0.5 },
            },
        )).id();

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems((
            spawn_physics_bodies_system,
            physics_step_system,
            sync_transforms_system,
        ).chain());

        // Run 60 steps (~1 second at 60 Hz)
        for _ in 0..60 {
            schedule.run(&mut world);
        }

        let transform = world.entity(entity).get::<TransformComponent>().unwrap();
        assert!(
            transform.position.y < 8.0,
            "Sphere should have fallen under gravity, y={}",
            transform.position.y
        );
    }

    #[test]
    fn spawn_runs_once_per_entity() {
        // Verify spawn_physics_bodies_system is idempotent:
        // running twice should not double-register the same entity.
        let mut world = World::new();
        world.insert_resource(RapierPhysicsWorld::new());

        world.spawn((
            TransformComponent::default(),
            ColliderComponent { shape: ColliderShape::Sphere { radius: 1.0 } },
        ));

        let mut schedule = bevy_ecs::schedule::Schedule::default();
        schedule.add_systems(spawn_physics_bodies_system);

        schedule.run(&mut world);
        let count_after_first = world.resource::<RapierPhysicsWorld>().body_count();

        schedule.run(&mut world);
        let count_after_second = world.resource::<RapierPhysicsWorld>().body_count();

        assert_eq!(
            count_after_first, count_after_second,
            "Running spawn twice should not register the entity again"
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vox_physics ecs 2>&1 | tail -10`
Expected: FAIL — `spawn_physics_bodies_system`, `physics_step_system` etc. not defined yet (the test file IS the implementation in this case — we wrote impl + tests together)

Actually, the test and impl are in the same file, so you should run the build check:

Run: `cargo check -p vox_physics 2>&1 | tail -10`
Expected: might fail on the static collider code — keep reading.

- [ ] **Step 3: Fix the static collider placeholder**

The implementation in Step 1 has a problematic static collider path that creates a spurious character controller body. Rapier static colliders have no `RigidBodyHandle`. Replace the `match` block in `spawn_physics_bodies_system` with a cleaner approach: for static shapes, use `RigidBodyHandle` from a kinematic-position body so we have a valid handle (it won't move since the body type is set to static via collider-only insertion), or more simply: **skip body_handle for statics** by only supporting Dynamic/Kinematic for sync, and for Static just insert a sentinel.

The cleanest approach: use `RigidBodyHandle` from `add_dynamic_box`/`add_dynamic_sphere` for all body types initially — later you can lock them. Replace the entire `spawn_physics_bodies_system` function with:

```rust
pub fn spawn_physics_bodies_system(
    mut commands: Commands,
    mut physics: ResMut<RapierPhysicsWorld>,
    query: Query<
        (Entity, &ColliderComponent, &TransformComponent, Option<&PhysicsBodyTypeComponent>),
        Without<PhysicsBodyComponent>,
    >,
) {
    for (entity, collider, transform, body_type) in query.iter() {
        let pos = [
            transform.position.x,
            transform.position.y,
            transform.position.z,
        ];
        let btype = body_type.copied().unwrap_or_default();

        let handles = match &collider.shape {
            ColliderShape::Box { half_extents } => match btype {
                PhysicsBodyTypeComponent::Static => {
                    let ch = physics.add_static_collider(pos, *half_extents);
                    None // Static: no body handle
                }
                _ => {
                    let (bh, ch) = physics.add_dynamic_box(pos, *half_extents, 1.0);
                    Some((bh, ch))
                }
            },
            ColliderShape::Sphere { radius } => {
                let (bh, ch) = physics.add_dynamic_sphere(pos, *radius, 1.0);
                Some((bh, ch))
            }
            ColliderShape::Capsule { radius, height } => {
                let (bh, ch) = physics.add_character_controller(pos, *radius, *height);
                Some((bh, ch))
            }
        };

        if let Some((body_handle, collider_handle)) = handles {
            commands.entity(entity).insert(PhysicsBodyComponent {
                body_handle,
                collider_handle,
            });
        }
        // Static entities get no PhysicsBodyComponent — they won't be sync'd (correct: they don't move)
    }
}
```

And update `sync_transforms_system` to stay the same (it already only processes entities WITH `PhysicsBodyComponent`, so statics are skipped automatically).

- [ ] **Step 4: Build and run tests**

Run: `cargo test -p vox_physics 2>&1 | tail -10`

Expected output: `9 passed; 0 failed` (6 rapier tests + 3 new ecs tests)

If `cannot find crate for bevy_ecs` errors occur, verify the Cargo.toml changes from Task 1 are in place.

If `TransformComponent not found` errors occur, double-check the import path: `vox_core::ecs::TransformComponent`. The `vox_core` dep must be in `vox_physics/Cargo.toml`.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_physics/src/ecs.rs crates/vox_physics/src/lib.rs
git commit -m "feat(physics): spawn_physics_bodies_system, physics_step_system, sync_transforms_system, PhysicsPlugin"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Fix dead rapier feature gate → Task 1
- ✅ bevy_ecs + bevy_app deps → Task 1
- ✅ `PhysicsBodyComponent` + `PhysicsBodyTypeComponent` → Task 2
- ✅ `spawn_physics_bodies_system` → Task 3
- ✅ `physics_step_system` → Task 3
- ✅ `sync_transforms_system` → Task 3
- ✅ `PhysicsPlugin` → Task 3
- ✅ Integration test: spawn → step → transform updated → Task 3
- ✅ Idempotency test (no double-register) → Task 3

**Placeholder scan:** No TBDs. All code shown in full.

**Type consistency:**
- `PhysicsBodyComponent { body_handle: RigidBodyHandle, collider_handle: ColliderHandle }` — defined Task 2, used Task 3 ✅
- `PhysicsBodyTypeComponent` — defined Task 2, used in spawn system Task 3 ✅
- `RapierPhysicsWorld` — from `crate::RapierPhysicsWorld` (re-exported in lib.rs) ✅
- `ColliderComponent`, `ColliderShape`, `TransformComponent` — from `vox_core::ecs` ✅
- `spawn_physics_bodies_system`, `physics_step_system`, `sync_transforms_system` — defined Task 3, registered in `PhysicsPlugin` Task 3 ✅
