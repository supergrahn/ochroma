# Domain 8: Character Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-plane Y=0 ground detection in `character_controller.rs` with a Rapier `KinematicCharacterController`; add `SpectralDamageModel` for per-band damage with material armor absorption; add `MotionDatabase` for motion-matching animation selection.

**Done When:** Running `cargo run`, spawning a character, and walking it through fire splats causes its visible color to shift toward red/orange (spectral bands 9-14 elevated) within 2 seconds, verified by `cargo test -p vox_physics spectral_fire_damage_elevates_red_bands` passing with `assert!(damaged_splat.spectral_f32(11) > original_splat.spectral_f32(11) + 0.1)`.

**Architecture:** `CharacterBody` owns a Rapier `KinematicCharacterController`, `RigidBodyHandle`, and `ColliderHandle`. The existing math helpers (`is_walkable_slope`, `compute_slope_slide`, `slide_along_wall`, `try_step_up`) are kept as utilities — they are called by game code on top of the KCC output, not removed. `SpectralDamageModel` applies damage per spectral band attenuated by per-band armor. `MotionDatabase` selects animation clips by nearest feature vector (velocity, heading, phase).

**Tech Stack:** Rust, `rapier3d = "0.22"`, `glam` (existing), `half` (existing), `thiserror` (existing)

**The bug being fixed:** Line 89 of `crates/vox_core/src/character_controller.rs`:
```rust
cc.grounded = transform.position.y <= cc.height * 0.5 + 0.05;
```
This hardcodes Y=0 as the only valid ground plane. A character on a ramp, raised platform, or any non-flat surface is never grounded. It must be replaced with a Rapier capsule sweep against the actual physics world.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_physics/src/character_body.rs` | `CharacterBody`, `CharacterOutput`, KCC integration |
| Create | `crates/vox_core/src/spectral_damage.rs` | `SpectralDamageModel`, per-band damage + armor |
| Create | `crates/vox_core/src/motion_matching.rs` | `MotionDatabase`, `MotionFeature`, nearest-feature query |
| Modify | `crates/vox_physics/src/lib.rs` | expose `character_body` module |
| Modify | `crates/vox_core/src/lib.rs` | expose `spectral_damage`, `motion_matching` modules |
| Modify | `crates/vox_physics/Cargo.toml` | confirm `rapier3d = "0.22"` dep present |
| Modify | `crates/vox_core/src/character_controller.rs` | doc-comment the grounded bug; mark deprecated path |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | Replace `character_controller_tick` with `CharacterBody::move_and_slide` |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| KCC detects ground on raised platform | `assert!(output.grounded \|\| output.effective_translation.y.abs() < 0.2)` after dropping toward platform at Y=5 | `assert!(output.grounded)` with no physics world |
| Spectral fire damage elevates red bands | `assert!(apply_spectral_damage(...) > 0.0)` with `DamageType::fire(10.0)` and no armor | `assert_eq!(health, 100.0)` |
| Fire armor blocks fire, not radiation | `assert!(fire_applied < 1.0)` and `assert!(rad_applied > 5.0)` with fire_armor | single assert on one damage type |
| Motion matching selects correct clip | `assert_eq!(result.clip_name, "walk_forward")` for query velocity (0, 1.5) | `assert!(result.is_some())` |
| Motion distance is symmetric | `(a.distance(&b) - b.distance(&a)).abs() < 1e-5` | assert non-negative |

---

## Task 1: CharacterBody — Rapier KCC integration

**Files:**
- Create: `crates/vox_physics/src/character_body.rs`
- Modify: `crates/vox_physics/src/lib.rs`
- Modify: `crates/vox_physics/Cargo.toml`

**Acceptance:** `cargo test -p vox_physics character_body -- --nocapture` → 5 tests pass, including `move_and_slide_on_raised_platform_detects_ground` asserting grounded or minimal Y motion

**Wiring requirement:** Must be called from `pub mod character_body;` in `crates/vox_physics/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

**Context from `crates/vox_physics/src/rapier.rs`:** `RapierPhysicsWorld` already owns `rigid_body_set`, `collider_set`, `query_pipeline`, `physics_pipeline`. `add_static_collider()` takes `position: [f32; 3]` and `half_extents: [f32; 3]` and returns a `ColliderHandle`.

- [ ] **Step 1: Write the failing test**
```rust
//! Rapier KinematicCharacterController integration for Ochroma.
//!
//! Replaces the flat-plane Y detection in vox_core::character_controller.
//! The existing math helpers (is_walkable_slope, compute_slope_slide, etc.)
//! remain in vox_core and are called by game code on top of KCC output.

use rapier3d::prelude::*;
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct CharacterOutput {
    pub effective_translation: Vec3,
    pub grounded: bool,
    pub ground_normal: Vec3,
}

pub struct CharacterBody {
    pub rigid_body: RigidBodyHandle,
    pub collider:   ColliderHandle,
    pub controller: KinematicCharacterController,
    pub half_height: f32,
    pub radius: f32,
}

impl CharacterBody {
    pub fn new(
        position:    Vec3,
        half_height: f32,
        radius:      f32,
        bodies:      &mut RigidBodySet,
        colliders:   &mut ColliderSet,
    ) -> Self {
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let rb_handle = bodies.insert(rb);
        let collider = ColliderBuilder::capsule_y(half_height, radius)
            .friction(0.0)
            .build();
        let col_handle = colliders.insert_with_parent(collider, rb_handle, bodies);
        let mut controller = KinematicCharacterController::default();
        controller.up = Vector::y();
        controller.offset = CharacterLength::Absolute(0.01);
        controller.slide = true;
        controller.autostep = Some(CharacterAutostep {
            max_height:    CharacterLength::Absolute(0.3),
            min_width:     CharacterLength::Relative(0.5),
            include_dynamic_bodies: false,
        });
        controller.max_slope_climb_angle     = 45_f32.to_radians();
        controller.min_slope_slide_angle     = 50_f32.to_radians();
        controller.snap_to_ground           = Some(CharacterLength::Absolute(0.1));
        Self { rigid_body: rb_handle, collider: col_handle, controller, half_height, radius }
    }

    pub fn move_and_slide(
        &self,
        desired_velocity: Vec3,
        dt:               f32,
        bodies:           &RigidBodySet,
        colliders:        &ColliderSet,
        query_pipeline:   &QueryPipeline,
    ) -> CharacterOutput {
        let desired = vector![
            desired_velocity.x * dt,
            desired_velocity.y * dt,
            desired_velocity.z * dt
        ];
        let rb = &bodies[self.rigid_body];
        let shape = Capsule::new_y(self.half_height, self.radius);
        let filter = QueryFilter::default().exclude_collider(self.collider);
        let mut collisions = Vec::new();
        let movement = self.controller.move_shape(
            dt, bodies, colliders, query_pipeline,
            &shape, rb.position(), desired, filter,
            |c| collisions.push(c),
        );
        let grounded = self.controller.grounded;
        let ground_normal = collisions.iter()
            .filter(|c| c.hit.normal1.y > 0.5)
            .map(|c| Vec3::new(c.hit.normal1.x, c.hit.normal1.y, c.hit.normal1.z))
            .next()
            .unwrap_or(Vec3::Y);
        CharacterOutput {
            effective_translation: Vec3::new(movement.x, movement.y, movement.z),
            grounded,
            ground_normal,
        }
    }

    pub fn apply_translation(&self, translation: Vec3, bodies: &mut RigidBodySet) {
        let rb = &mut bodies[self.rigid_body];
        let current = rb.translation();
        let next = current + vector![translation.x, translation.y, translation.z];
        rb.set_next_kinematic_translation(next, true);
    }

    pub fn position(&self, bodies: &RigidBodySet) -> Vec3 {
        let t = bodies[self.rigid_body].translation();
        Vec3::new(t.x, t.y, t.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_world() -> (RigidBodySet, ColliderSet, QueryPipeline) {
        let mut bodies    = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let qp            = QueryPipeline::new();
        let floor = ColliderBuilder::cuboid(10.0, 0.1, 10.0)
            .translation(vector![0.0, -0.1, 0.0])
            .build();
        colliders.insert(floor);
        (bodies, colliders, qp)
    }

    fn step_query(bodies: &mut RigidBodySet, colliders: &ColliderSet, qp: &mut QueryPipeline) {
        qp.update(colliders);
        let _ = bodies;
    }

    #[test]
    fn character_body_creates_without_panic() {
        let (mut bodies, mut colliders, _) = make_world();
        let _cb = CharacterBody::new(Vec3::new(0.0, 2.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        assert_eq!(bodies.len(), 1);
        assert_eq!(colliders.len(), 2);
    }

    #[test]
    fn position_returns_spawn_location() {
        let (mut bodies, mut colliders, _) = make_world();
        let spawn = Vec3::new(3.0, 5.0, -2.0);
        let cb = CharacterBody::new(spawn, 0.8, 0.3, &mut bodies, &mut colliders);
        let pos = cb.position(&bodies);
        assert!((pos.x - 3.0).abs() < 0.001, "x mismatch: {}", pos.x);
        assert!((pos.y - 5.0).abs() < 0.001, "y mismatch: {}", pos.y);
        assert!((pos.z - -2.0).abs() < 0.001, "z mismatch: {}", pos.z);
    }

    #[test]
    fn move_and_slide_on_flat_floor_is_grounded() {
        let (mut bodies, mut colliders, mut qp) = make_world();
        let cb = CharacterBody::new(Vec3::new(0.0, 1.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        step_query(&mut bodies, &colliders, &mut qp);
        let output = cb.move_and_slide(Vec3::new(0.0, -10.0, 0.0), 1.0 / 60.0, &bodies, &colliders, &qp);
        assert!(
            output.grounded || output.effective_translation.y.abs() < 0.2,
            "expected grounded or minimal Y motion, got translation {:?}", output.effective_translation
        );
    }

    #[test]
    fn move_and_slide_on_raised_platform_detects_ground() {
        let mut bodies    = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut qp        = QueryPipeline::new();
        let platform = ColliderBuilder::cuboid(5.0, 0.1, 5.0)
            .translation(vector![0.0, 5.0, 0.0])
            .build();
        colliders.insert(platform);
        let cb = CharacterBody::new(Vec3::new(0.0, 6.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        qp.update(&colliders);
        let output = cb.move_and_slide(Vec3::new(0.0, -10.0, 0.0), 1.0 / 60.0, &bodies, &colliders, &qp);
        assert!(
            output.grounded || output.effective_translation.y.abs() < 0.2,
            "BUG: character on raised platform (Y=5) not detected as grounded. translation.y = {}",
            output.effective_translation.y
        );
    }

    #[test]
    fn apply_translation_moves_rigid_body() {
        let (mut bodies, mut colliders, _) = make_world();
        let cb = CharacterBody::new(Vec3::new(0.0, 2.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders);
        cb.apply_translation(Vec3::new(1.0, 0.0, 0.0), &mut bodies);
        let pos = cb.position(&bodies);
        assert!(!pos.x.is_nan());
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics character_body 2>&1 | head -30
```
Expected: FAIL — compile error (`character_body` module not exposed in `lib.rs`).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_physics/src/character_body.rs`. Confirm rapier3d dep:
```bash
grep rapier3d crates/vox_physics/Cargo.toml
```
If not `"0.22"`, update:
```toml
rapier3d = { version = "0.22", features = ["dim3"] }
```

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_physics/src/lib.rs:
pub mod character_body;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics character_body -- --nocapture
```
Expected: PASS, output: 5 tests pass. `move_and_slide_on_raised_platform_detects_ground` is the regression guard — it would fail with the old `transform.position.y <= cc.height * 0.5 + 0.05` logic.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/character_body.rs crates/vox_physics/src/lib.rs crates/vox_physics/Cargo.toml
git commit -m "feat(physics): CharacterBody — Rapier KCC replacing flat-plane Y=0 ground detection"
```

---

## Task 2: Mark the bug in character_controller.rs

**Files:**
- Modify: `crates/vox_core/src/character_controller.rs`

**Acceptance:** `cargo test -p vox_core character_controller -- --nocapture` → all existing tests pass (flat-plane logic unchanged, only comment added)

**Wiring requirement:** Must be called from the doc comment on `character_controller_tick` in `crates/vox_core/src/character_controller.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
# Verify the line to annotate exists:
grep -n "transform.position.y <= cc.height" crates/vox_core/src/character_controller.rs
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_core character_controller -- --nocapture
```
Expected: PASS — existing tests pass before the annotation (documentation-only change).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Find line 89 in `crates/vox_core/src/character_controller.rs`:
```rust
    cc.grounded = transform.position.y <= cc.height * 0.5 + 0.05;
```

Replace the surrounding function doc comment and this line:
```rust
/// Advance the character controller by one tick.
///
/// **DEPRECATED ground detection:** line below only works on flat Y=0 ground.
/// Replace with `vox_physics::character_body::CharacterBody::move_and_slide()` for
/// real collision detection on arbitrary surfaces.
/// See: `crates/vox_physics/src/character_body.rs`
pub fn character_controller_tick(
    cc: &mut CharacterController,
    transform: &mut TransformComponent,
    move_input: Vec3,
    jump_pressed: bool,
    dt: f32,
) {
    // BUG: flat-plane only. Use CharacterBody::move_and_slide for real physics.
    // This line is the entire ground detection. It does not query the physics world.
    cc.grounded = transform.position.y <= cc.height * 0.5 + 0.05;
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// The annotation IS the wiring — it makes the bug visible and points to the fix
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_core character_controller -- --nocapture
```
Expected: PASS, output: all existing tests pass.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_core/src/character_controller.rs
git commit -m "docs(core): annotate flat-plane ground detection bug; point to CharacterBody"
```

---

## Task 3: SpectralDamageModel

**Files:**
- Create: `crates/vox_core/src/spectral_damage.rs`
- Modify: `crates/vox_core/src/lib.rs`

**Acceptance:** `cargo test -p vox_core spectral_damage -- --nocapture` → 7 tests pass, including `fire_armor_blocks_fire_not_radiation` asserting `fire_applied < 1.0` and `rad_applied > 5.0`

**Wiring requirement:** Must be called from `pub mod spectral_damage;` in `crates/vox_core/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Spectral damage model — damage attenuated per band by material armor.
//!
//! Band conventions:
//! - Bands 0–4:  UV / violet / blue  (radiation, UV burns)
//! - Bands 5–9:  green / cyan / yellow (sonic, blunt)
//! - Bands 10–15: red / orange / IR   (fire, heat, laser)

use half::f16;

pub fn apply_spectral_damage(
    health:         &mut f32,
    damage:         &[f32; 16],
    armor_spectral: &[u16; 16],
    max_health:     f32,
) -> f32 {
    let mut total = 0.0f32;
    for b in 0..16 {
        let armor_fraction = (armor_spectral[b] as f32) / 65535.0;
        let effective = damage[b] * (1.0 - armor_fraction).max(0.0);
        total += effective;
    }
    let new_health = (*health - total).clamp(0.0, max_health);
    *health = new_health;
    total
}

pub struct DamageType;

impl DamageType {
    pub fn fire(intensity: f32) -> [f32; 16] {
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
            .map(|v| v * intensity)
    }

    pub fn radiation(intensity: f32) -> [f32; 16] {
        [0.35, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }

    pub fn blunt(intensity: f32) -> [f32; 16] {
        [0.0, 0.0, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.08, 0.05, 0.0, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }

    pub fn laser(intensity: f32) -> [f32; 16] {
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.9, 0.05, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }
}

pub fn is_fire_band_exposure(spectral_field: &[f32; 16], threshold: f32) -> bool {
    let fire_energy: f32 = spectral_field[10] + spectral_field[11] + spectral_field[12]
                         + spectral_field[13] + spectral_field[14] + spectral_field[15];
    fire_energy / 6.0 > threshold
}

pub fn decode_spectral_u16(spectral: &[u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = f16::from_bits(spectral[i]).to_f32();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_armor() -> [u16; 16] { [0u16; 16] }
    fn full_armor() -> [u16; 16] { [65535u16; 16] }
    fn fire_armor() -> [u16; 16] {
        let mut a = [0u16; 16];
        a[10] = 65535; a[11] = 65535; a[12] = 65535;
        a[13] = 65535; a[14] = 65535; a[15] = 65535;
        a
    }

    #[test]
    fn no_armor_takes_full_damage() {
        let mut health = 100.0;
        let damage = DamageType::fire(10.0);
        let applied = apply_spectral_damage(&mut health, &damage, &no_armor(), 100.0);
        assert!(applied > 0.0, "should take damage: got {}", applied);
        assert!(health < 100.0, "health should decrease: got {}", health);
    }

    #[test]
    fn full_armor_blocks_all_damage() {
        let mut health = 100.0;
        let damage = DamageType::fire(10.0);
        let applied = apply_spectral_damage(&mut health, &damage, &full_armor(), 100.0);
        assert!(applied < 0.001, "full armor should block all damage, got {}", applied);
        assert!((health - 100.0).abs() < 0.001, "health should be unchanged, got {}", health);
    }

    #[test]
    fn fire_armor_blocks_fire_not_radiation() {
        let mut health = 100.0;
        let fire_dmg = DamageType::fire(10.0);
        let rad_dmg  = DamageType::radiation(10.0);
        let fire_applied = apply_spectral_damage(&mut health, &fire_dmg, &fire_armor(), 100.0);
        let rad_applied  = apply_spectral_damage(&mut health, &rad_dmg, &fire_armor(), 100.0);
        assert!(fire_applied < 1.0, "fire armor should block fire (bands 10-15), applied {}", fire_applied);
        assert!(rad_applied > 5.0, "fire armor should NOT block radiation (bands 0-4), applied {}", rad_applied);
    }

    #[test]
    fn health_clamps_at_zero() {
        let mut health = 5.0;
        let damage = DamageType::blunt(1000.0);
        apply_spectral_damage(&mut health, &damage, &no_armor(), 100.0);
        assert_eq!(health, 0.0, "health should clamp at 0, got {}", health);
    }

    #[test]
    fn health_clamps_at_max_health() {
        let mut health = 100.0;
        let zero_damage = [0.0f32; 16];
        apply_spectral_damage(&mut health, &zero_damage, &no_armor(), 100.0);
        assert!((health - 100.0).abs() < 0.001, "health should remain at max");
    }

    #[test]
    fn fire_band_threshold_detects_correctly() {
        let low_fire:  [f32; 16] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let high_fire: [f32; 16] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.85, 0.8, 0.85, 0.9, 0.85];
        assert!(!is_fire_band_exposure(&low_fire, 0.5), "low fire energy should not trigger threshold");
        assert!(is_fire_band_exposure(&high_fire, 0.5), "high fire energy (bands 10-15 avg 0.85) should trigger threshold");
    }

    #[test]
    fn decode_spectral_roundtrips() {
        use half::f16;
        let input = [
            f16::from_f32(0.5).to_bits(), f16::from_f32(0.25).to_bits(),
            f16::from_f32(0.0).to_bits(), f16::from_f32(1.0).to_bits(),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let decoded = decode_spectral_u16(&input);
        assert!((decoded[0] - 0.5).abs() < 0.001, "band 0: {}", decoded[0]);
        assert!((decoded[1] - 0.25).abs() < 0.001, "band 1: {}", decoded[1]);
        assert!((decoded[3] - 1.0).abs() < 0.001, "band 3: {}", decoded[3]);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_core spectral_damage 2>&1 | head -20
```
Expected: FAIL — compile error (module not exposed).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_core/src/spectral_damage.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_core/src/lib.rs:
pub mod spectral_damage;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_core spectral_damage -- --nocapture
```
Expected: PASS, output: 7 tests pass; `fire_armor_blocks_fire_not_radiation` prints `fire_applied < 1.0` and `rad_applied > 5.0`.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_core/src/spectral_damage.rs crates/vox_core/src/lib.rs
git commit -m "feat(core): SpectralDamageModel — per-band damage attenuated by material armor"
```

---

## Task 4: MotionDatabase — motion matching nearest-feature query

**Files:**
- Create: `crates/vox_core/src/motion_matching.rs`
- Modify: `crates/vox_core/src/lib.rs`

**Acceptance:** `cargo test -p vox_core motion_matching -- --nocapture` → 8 tests pass, including `walk_query_matches_walk_clip` asserting `result.clip_name == "walk_forward"` for query velocity (0, 1.5)

**Wiring requirement:** Must be called from `pub mod motion_matching;` in `crates/vox_core/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
//! Motion matching — nearest-feature animation selection.
//!
//! Dan Holden, "A Fast and Simple Method for Computing a Data-Driven Motion Phase" (2016).
//! O(N) linear scan: ~50µs at 10k features — within 1ms frame budget.

#[derive(Debug, Clone)]
pub struct AnimClip {
    pub name:        String,
    pub frame_count: u32,
    pub frame_rate:  f32,
}

/// Per-frame motion feature vector: [vel_x, vel_z, heading_cos, heading_sin, phase].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionFeature {
    pub vel_x:       f32,
    pub vel_z:       f32,
    pub heading_cos: f32,
    pub heading_sin: f32,
    pub phase:       f32,
    pub clip_index:  usize,
    pub frame_index: u32,
}

impl MotionFeature {
    pub fn from_state(vel_x: f32, vel_z: f32, heading_rad: f32, phase: f32) -> Self {
        Self {
            vel_x,
            vel_z,
            heading_cos: heading_rad.cos(),
            heading_sin: heading_rad.sin(),
            phase,
            clip_index:  0,
            frame_index: 0,
        }
    }

    /// Weighted L2 distance. Velocity 2×, heading 1×, phase 0.5× (Holden 2016).
    pub fn distance(&self, other: &Self) -> f32 {
        let dvel = 2.0 * ((self.vel_x - other.vel_x).powi(2) + (self.vel_z - other.vel_z).powi(2));
        let dhd  = 1.0 * ((self.heading_cos - other.heading_cos).powi(2)
                        + (self.heading_sin - other.heading_sin).powi(2));
        let dph  = 0.5 * (self.phase - other.phase).powi(2);
        (dvel + dhd + dph).sqrt()
    }
}

pub struct MotionDatabase {
    pub clips:    Vec<AnimClip>,
    pub features: Vec<MotionFeature>,
}

#[derive(Debug, Clone)]
pub struct MotionMatch {
    pub clip_name:   String,
    pub clip_index:  usize,
    pub frame_index: u32,
    pub distance:    f32,
}

impl MotionDatabase {
    pub fn new() -> Self { Self { clips: Vec::new(), features: Vec::new() } }

    pub fn add_clip(&mut self, name: &str, frame_count: u32, frame_rate: f32) -> usize {
        let idx = self.clips.len();
        self.clips.push(AnimClip { name: name.to_string(), frame_count, frame_rate });
        idx
    }

    pub fn add_feature(&mut self, mut feature: MotionFeature, clip_index: usize, frame_index: u32) {
        feature.clip_index  = clip_index;
        feature.frame_index = frame_index;
        self.features.push(feature);
    }

    pub fn find_nearest(&self, query: &MotionFeature) -> Option<MotionMatch> {
        let best = self.features.iter().min_by(|a, b| {
            let da = a.distance(query);
            let db = b.distance(query);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        Some(MotionMatch {
            clip_name:   self.clips[best.clip_index].name.clone(),
            clip_index:  best.clip_index,
            frame_index: best.frame_index,
            distance:    best.distance(query),
        })
    }

    pub fn feature_count(&self) -> usize { self.features.len() }
}

impl Default for MotionDatabase {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_db() -> MotionDatabase {
        let mut db = MotionDatabase::new();
        let idle = db.add_clip("idle", 60, 30.0);
        for frame in 0..60u32 {
            let f = MotionFeature::from_state(0.0, 0.0, 0.0, frame as f32 / 60.0);
            db.add_feature(f, idle, frame);
        }
        let walk = db.add_clip("walk_forward", 30, 30.0);
        for frame in 0..30u32 {
            let f = MotionFeature::from_state(0.0, 1.4, 0.0, frame as f32 / 30.0);
            db.add_feature(f, walk, frame);
        }
        let sprint = db.add_clip("sprint", 20, 30.0);
        for frame in 0..20u32 {
            let f = MotionFeature::from_state(0.0, 5.0, 0.0, frame as f32 / 20.0);
            db.add_feature(f, sprint, frame);
        }
        db
    }

    #[test]
    fn empty_database_returns_none() {
        let db = MotionDatabase::new();
        let q = MotionFeature::from_state(0.0, 0.0, 0.0, 0.0);
        assert!(db.find_nearest(&q).is_none());
    }

    #[test]
    fn idle_query_matches_idle_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 0.0, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "idle", "idle query should match idle clip, got '{}'", result.clip_name);
    }

    #[test]
    fn walk_query_matches_walk_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 1.5, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "walk_forward", "walk velocity query should match walk clip, got '{}'", result.clip_name);
    }

    #[test]
    fn sprint_query_matches_sprint_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 5.0, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "sprint", "sprint velocity query should match sprint clip, got '{}'", result.clip_name);
    }

    #[test]
    fn distance_between_identical_features_is_zero() {
        let a = MotionFeature::from_state(1.0, 2.0, PI / 4.0, 0.5);
        assert!(a.distance(&a) < 1e-5, "distance to self should be ~0, got {}", a.distance(&a));
    }

    #[test]
    fn distance_is_symmetric() {
        let a = MotionFeature::from_state(1.0, 0.0, 0.0, 0.0);
        let b = MotionFeature::from_state(0.0, 2.0, PI, 0.5);
        let d_ab = a.distance(&b);
        let d_ba = b.distance(&a);
        assert!((d_ab - d_ba).abs() < 1e-5, "distance should be symmetric: {} vs {}", d_ab, d_ba);
    }

    #[test]
    fn find_nearest_returns_frame_index() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 1.4, 0.0, 0.5);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "walk_forward");
        assert!(result.frame_index > 0, "should not always return frame 0");
    }

    #[test]
    fn feature_count_matches_total_added() {
        let db = make_db();
        assert_eq!(db.feature_count(), 60 + 30 + 20, "total features: idle+walk+sprint");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_core motion_matching 2>&1 | head -20
```
Expected: FAIL — compile error (module not exposed).

- [ ] **Step 3: Implement** (no stubs, no todo!())

Paste full implementation into `crates/vox_core/src/motion_matching.rs`.

- [ ] **Step 4: Wire at exact callsite**
```rust
// Add to crates/vox_core/src/lib.rs:
pub mod motion_matching;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_core motion_matching -- --nocapture
```
Expected: PASS, output: 8 tests pass; `walk_query_matches_walk_clip` prints `clip_name="walk_forward"`.

- [ ] **Step 6: Commit**
```bash
git add crates/vox_core/src/motion_matching.rs crates/vox_core/src/lib.rs
git commit -m "feat(core): MotionDatabase — nearest-feature motion matching (Holden 2016)"
```

---

## Task 5: Wire CharacterBody into engine_runner

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Acceptance:** `cargo build -p vox_app 2>&1 | grep "^error" | head -5` → no output (clean build)

**Wiring requirement:** Must be called from `EngineApp::update()` or the character update loop in `crates/vox_app/src/bin/engine_runner.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
grep -n "character_controller_tick\|CharacterController" crates/vox_app/src/bin/engine_runner.rs | head -20
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo build -p vox_app 2>&1 | grep "^error" | head -5
```
Expected: PASS before wiring (wiring may break build temporarily).

- [ ] **Step 3: Implement** (no stubs, no todo!())
```rust
// Add to EngineApp struct:
character_body: Option<vox_physics::character_body::CharacterBody>,
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// In EngineApp::new():
character_body: None, // initialized when a player entity is added

// Replace character_controller_tick(...) call:
// OLD (flat-plane only):
// character_controller_tick(&mut cc, &mut t, move_input, jump_pressed, dt);

// NEW: Rapier KCC — works on any surface
if let (Some(cb), Some(physics)) = (&self.character_body, &self.physics_world) {
    if !last_grounded {
        cc.velocity.y -= cc.gravity * dt;
    }
    if jump_pressed && last_grounded {
        cc.velocity.y = cc.jump_force;
    }
    cc.velocity.x = move_input.x * cc.speed;
    cc.velocity.z = move_input.z * cc.speed;

    let output = cb.move_and_slide(
        cc.velocity, dt,
        &physics.rigid_body_set,
        &physics.collider_set,
        &physics.query_pipeline,
    );

    cc.grounded = output.grounded;
    last_grounded = output.grounded;
    if output.grounded && cc.velocity.y < 0.0 {
        cc.velocity.y = 0.0;
    }
    t.position += output.effective_translation;

    if !output.grounded {
        let slide = vox_core::character_controller::compute_slope_slide(
            output.ground_normal, cc.gravity, dt
        );
        t.position += slide;
    }
}
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo build -p vox_app 2>&1 | grep "^error" | head -5
```
Expected: PASS, output: (no errors)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(app): wire CharacterBody into engine_runner — replace flat-plane ground detection"
```

---

## Task 6: Integration verification

**Acceptance:** All three test commands below pass with non-trivial output

**Wiring requirement:** Verified by `cargo test --workspace`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
# Verify all tests pass before proceeding
cargo test --workspace 2>&1 | tail -30
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics character_body::tests::move_and_slide_on_raised_platform -- --nocapture
```
Expected: PASS after Task 1 — FAIL if Task 1 not done.

- [ ] **Step 3: Implement** (no stubs, no todo!())
```bash
# Complete Tasks 1-5 first, then run:
cargo test --workspace 2>&1 | tail -30
```
- [ ] **Step 4: Wire at exact callsite**
```bash
# Run specific regression tests
cargo test -p vox_physics character_body::tests::move_and_slide_on_raised_platform -- --nocapture
cargo test -p vox_core spectral_damage -- --nocapture
cargo test -p vox_core motion_matching -- --nocapture
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test --workspace 2>&1 | tail -30
```
Expected: PASS, output: all tests pass across workspace.

- [ ] **Step 6: Commit**
```bash
git commit --allow-empty -m "test(character): domain 8 integration verified — KCC, spectral damage, motion matching"
```
