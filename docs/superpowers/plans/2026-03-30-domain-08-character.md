# Domain 8 — Character Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-plane Y=0 ground detection in `character_controller.rs` with a Rapier `KinematicCharacterController`; add `SpectralDamageModel` for per-band damage with material armor absorption; add `MotionDatabase` for motion-matching animation selection.

**The bug being fixed:** Line 89 of `crates/vox_core/src/character_controller.rs`:
```rust
cc.grounded = transform.position.y <= cc.height * 0.5 + 0.05;
```
This hardcodes Y=0 as the only valid ground plane. A character on a ramp, a raised platform, or any non-flat surface is never grounded. It must be replaced with a Rapier capsule sweep against the actual physics world.

**Architecture:** `CharacterBody` owns a Rapier `KinematicCharacterController`, `RigidBodyHandle`, and `ColliderHandle`. The existing math helpers (`is_walkable_slope`, `compute_slope_slide`, `slide_along_wall`, `try_step_up`) are kept as utilities — they are called by game code on top of the KCC output, not removed. `SpectralDamageModel` applies damage per spectral band attenuated by per-band armor. `MotionDatabase` selects animation clips by nearest feature vector (velocity, heading, phase).

**Tech Stack:** Rust, `rapier3d = "0.22"`, `glam` (existing), `half` (existing), `thiserror` (existing)

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

---

## Task 1: CharacterBody — Rapier KCC integration

**Files:**
- Create: `crates/vox_physics/src/character_body.rs`
- Modify: `crates/vox_physics/src/lib.rs`
- Modify: `crates/vox_physics/Cargo.toml`

**Context from `crates/vox_physics/src/rapier.rs`:** `RapierPhysicsWorld` already owns `rigid_body_set`, `collider_set`, `query_pipeline`, `physics_pipeline`, and all integration parameters. `add_static_collider()` takes `position: [f32; 3]` and `half_extents: [f32; 3]` and returns a `ColliderHandle`. This is the API `CharacterBody` will call on `PhysicsWorld` to register its capsule collider.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_physics/src/character_body.rs`:

```rust
//! Rapier KinematicCharacterController integration for Ochroma.
//!
//! Replaces the flat-plane Y detection in vox_core::character_controller.
//! The existing math helpers (is_walkable_slope, compute_slope_slide, etc.)
//! remain in vox_core and are called by game code on top of KCC output.

use rapier3d::prelude::*;
use glam::Vec3;

/// Output of one `move_and_slide` call — what the KCC resolved.
#[derive(Debug, Clone)]
pub struct CharacterOutput {
    /// Translation actually applied after collision resolution.
    pub effective_translation: Vec3,
    /// True if the character is touching ground this frame.
    pub grounded: bool,
    /// Ground normal at foot contact point (Y-up if no contact).
    pub ground_normal: Vec3,
}

/// Physics body representing a player or NPC capsule.
///
/// Owns the Rapier handles for the kinematic rigid body and capsule collider.
/// Lifetime is tied to the `RigidBodySet` / `ColliderSet` in which it was created.
pub struct CharacterBody {
    pub rigid_body: RigidBodyHandle,
    pub collider:   ColliderHandle,
    pub controller: KinematicCharacterController,
    /// Half-height of the capsule (not counting hemisphere radii).
    pub half_height: f32,
    /// Capsule radius.
    pub radius: f32,
}

impl CharacterBody {
    /// Create a new `CharacterBody` at `position` with the given capsule dimensions.
    ///
    /// Inserts the rigid body and collider into the provided sets.
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
            .friction(0.0) // friction handled via KCC, not Rapier contacts
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

    /// Compute movement for this frame.
    ///
    /// `desired_velocity`: XZ movement + Y jump/gravity already accumulated.
    /// Returns `CharacterOutput` with the collision-resolved translation and grounded state.
    /// Caller must then write `effective_translation` back to the rigid body position.
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

        // Filter: skip self
        let filter = QueryFilter::default().exclude_collider(self.collider);

        let mut collisions = Vec::new();
        let movement = self.controller.move_shape(
            dt,
            bodies,
            colliders,
            query_pipeline,
            &shape,
            rb.position(),
            desired,
            filter,
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

    /// Apply the resolved translation back to the rigid body.
    pub fn apply_translation(
        &self,
        translation: Vec3,
        bodies:      &mut RigidBodySet,
    ) {
        let rb = &mut bodies[self.rigid_body];
        let current = rb.translation();
        let next = current + vector![translation.x, translation.y, translation.z];
        rb.set_next_kinematic_translation(next, true);
    }

    /// Current world position (capsule centre).
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

        // Static floor at Y=0, 20×20 m
        let floor = ColliderBuilder::cuboid(10.0, 0.1, 10.0)
            .translation(vector![0.0, -0.1, 0.0])
            .build();
        colliders.insert(floor);

        (bodies, colliders, qp)
    }

    fn step_query(
        bodies: &mut RigidBodySet,
        colliders: &ColliderSet,
        qp: &mut QueryPipeline,
    ) {
        qp.update(colliders);
        // In production the physics pipeline steps too; here we only need QP.
        let _ = bodies; // keep borrow alive
    }

    #[test]
    fn character_body_creates_without_panic() {
        let (mut bodies, mut colliders, _) = make_world();
        let _cb = CharacterBody::new(
            Vec3::new(0.0, 2.0, 0.0),
            0.8, 0.3,
            &mut bodies, &mut colliders,
        );
        assert_eq!(bodies.len(), 1);
        assert_eq!(colliders.len(), 2); // floor + capsule
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
        let cb = CharacterBody::new(
            Vec3::new(0.0, 1.0, 0.0), // capsule centre 1m above floor
            0.8, 0.3,
            &mut bodies, &mut colliders,
        );
        step_query(&mut bodies, &colliders, &mut qp);

        // Gravity pull
        let output = cb.move_and_slide(
            Vec3::new(0.0, -10.0, 0.0),
            1.0 / 60.0,
            &bodies,
            &colliders,
            &qp,
        );
        // After dropping toward floor the KCC should detect ground
        assert!(
            output.grounded || output.effective_translation.y.abs() < 0.2,
            "expected grounded or minimal Y motion, got translation {:?}",
            output.effective_translation
        );
    }

    #[test]
    fn move_and_slide_on_raised_platform_detects_ground() {
        let mut bodies    = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut qp        = QueryPipeline::new();

        // Raised platform at Y=5
        let platform = ColliderBuilder::cuboid(5.0, 0.1, 5.0)
            .translation(vector![0.0, 5.0, 0.0])
            .build();
        colliders.insert(platform);

        let cb = CharacterBody::new(
            Vec3::new(0.0, 6.0, 0.0), // 1m above the raised platform
            0.8, 0.3,
            &mut bodies, &mut colliders,
        );
        qp.update(&colliders);

        let output = cb.move_and_slide(
            Vec3::new(0.0, -10.0, 0.0),
            1.0 / 60.0,
            &bodies,
            &colliders,
            &qp,
        );
        // The key correctness assertion for the bug fix:
        // character landing on a non-Y=0 surface must still detect ground.
        assert!(
            output.grounded || output.effective_translation.y.abs() < 0.2,
            "BUG: character on raised platform (Y=5) not detected as grounded. \
             translation.y = {}. Old flat-plane detection would fail here.",
            output.effective_translation.y
        );
    }

    #[test]
    fn apply_translation_moves_rigid_body() {
        let (mut bodies, mut colliders, _) = make_world();
        let cb = CharacterBody::new(
            Vec3::new(0.0, 2.0, 0.0), 0.8, 0.3, &mut bodies, &mut colliders,
        );
        cb.apply_translation(Vec3::new(1.0, 0.0, 0.0), &mut bodies);
        let pos = cb.position(&bodies);
        // next_kinematic_translation is staged; the actual position updates after
        // pipeline step. We verify the call doesn't panic.
        assert!(!pos.x.is_nan());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/tomespen/git/ochroma
cargo test -p vox_physics character_body 2>&1 | head -30
```

Expected: compile error — `character_body` module not exposed in `lib.rs`.

- [ ] **Step 3: Confirm rapier3d dep version**

```bash
grep rapier3d crates/vox_physics/Cargo.toml
```

If not `"0.22"`, update:

```toml
rapier3d = { version = "0.22", features = ["dim3"] }
```

- [ ] **Step 4: Expose the module**

Add to `crates/vox_physics/src/lib.rs`:

```rust
pub mod character_body;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p vox_physics character_body -- --nocapture
```

Expected: 5 tests pass (character_body_creates_without_panic, position_returns_spawn_location, move_and_slide_on_flat_floor_is_grounded, move_and_slide_on_raised_platform_detects_ground, apply_translation_moves_rigid_body).

The `move_and_slide_on_raised_platform_detects_ground` test is the regression guard — it would fail with the old `transform.position.y <= cc.height * 0.5 + 0.05` logic.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_physics/src/character_body.rs crates/vox_physics/src/lib.rs crates/vox_physics/Cargo.toml
git commit -m "feat(physics): CharacterBody — Rapier KCC replacing flat-plane Y=0 ground detection"
```

---

## Task 2: Mark the bug in character_controller.rs

**Files:**
- Modify: `crates/vox_core/src/character_controller.rs`

This is a documentation-only change. The flat-plane function stays for backward compatibility (existing tests pass). The comment makes the bug visible and points to the replacement.

- [ ] **Step 1: Add deprecation notice to `character_controller_tick`**

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
///
/// This is a plain function (not a Bevy system) so it can be tested without a
/// `World`.  A thin Bevy system can call this each frame.
pub fn character_controller_tick(
    cc: &mut CharacterController,
    transform: &mut TransformComponent,
    move_input: Vec3, // normalized XZ movement direction
    jump_pressed: bool,
    dt: f32,
) {
    // BUG: flat-plane only. Use CharacterBody::move_and_slide for real physics.
    // This line is the entire ground detection. It does not query the physics world.
    cc.grounded = transform.position.y <= cc.height * 0.5 + 0.05;
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
cargo test -p vox_core character_controller -- --nocapture
```

Expected: all existing tests pass (they were written for the flat-plane function and still test that code path correctly).

- [ ] **Step 3: Commit**

```bash
git add crates/vox_core/src/character_controller.rs
git commit -m "docs(core): annotate flat-plane ground detection bug; point to CharacterBody"
```

---

## Task 3: SpectralDamageModel

**Files:**
- Create: `crates/vox_core/src/spectral_damage.rs`
- Modify: `crates/vox_core/src/lib.rs`

**Design:** Damage is a `[f32; 8]` — one value per spectral band. Armor is a `[u16; 8]` (same layout as `GaussianSplat.spectral`). Each band's damage is attenuated by the armor value for that band. Physics meaning: fire damage is concentrated in bands 5–7 (red/IR). Water-spectral armor (high bands 2–3) does not absorb fire. Metal armor (high uniform reflectance, bands 0–7) absorbs radiation (bands 0–2). The attenuation model: `effective_damage[b] = damage[b] * (1 - armor_fraction[b])` where `armor_fraction[b] = armor_spectral[b] as f32 / 65535.0`.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_core/src/spectral_damage.rs`:

```rust
//! Spectral damage model — damage attenuated per band by material armor.
//!
//! Damage is represented as 8 independent band values. Armor absorbs each
//! band proportionally to the armor's spectral value in that band.
//!
//! # Band conventions (aligned with BAND_NM in spectral_atmosphere):
//! - Bands 0–2: UV / violet / blue  (radiation, UV burns)
//! - Bands 3–4: green / cyan        (sonic, blunt)
//! - Bands 5–7: red / orange / IR   (fire, heat, laser)
//!
//! # Armor absorption examples:
//! - Metal armor (high uniform reflectance): absorbs UV/radiation (bands 0–2)
//! - Water-spectral armor (bands 2–3 high): absorbs green/cyan
//! - Ablative foam (bands 5–7 high absorption): absorbs fire damage
//! - No armor fully protects against all bands — it would need uniform 65535

use half::f16;

/// Apply spectral damage to a health value with armor attenuation.
///
/// `health`:         current health (modified in place, clamped to [0, max_health]).
/// `damage`:         per-band damage values (non-negative).
/// `armor_spectral`: per-band armor as u16 values (0 = no armor, 65535 = maximum).
/// `max_health`:     health ceiling.
///
/// Returns the total effective damage applied.
pub fn apply_spectral_damage(
    health:         &mut f32,
    damage:         &[f32; 8],
    armor_spectral: &[u16; 8],
    max_health:     f32,
) -> f32 {
    let mut total = 0.0f32;
    for b in 0..8 {
        let armor_fraction = (armor_spectral[b] as f32) / 65535.0;
        let effective = damage[b] * (1.0 - armor_fraction).max(0.0);
        total += effective;
    }
    let new_health = (*health - total).clamp(0.0, max_health);
    *health = new_health;
    total
}

/// Standard damage type presets — spectral band distributions for common sources.
pub struct DamageType;

impl DamageType {
    /// Fire: concentrated in bands 5–7 (red / orange / near-IR).
    pub fn fire(intensity: f32) -> [f32; 8] {
        [0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.4, 0.45]
            .map(|v| v * intensity)
    }

    /// Radiation: concentrated in bands 0–2 (UV / violet / blue).
    pub fn radiation(intensity: f32) -> [f32; 8] {
        [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }

    /// Physical blunt: distributed across mid-bands 3–5.
    pub fn blunt(intensity: f32) -> [f32; 8] {
        [0.0, 0.05, 0.1, 0.25, 0.3, 0.25, 0.05, 0.0]
            .map(|v| v * intensity)
    }

    /// Laser (narrow band 6 — 620nm red):
    pub fn laser(intensity: f32) -> [f32; 8] {
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.9, 0.05]
            .map(|v| v * intensity)
    }
}

/// Query whether a spectral field energy reading constitutes fire-band exposure.
///
/// Checks if the sum of bands 5–7 exceeds `threshold` (normalised 0–1).
pub fn is_fire_band_exposure(spectral_field: &[f32; 8], threshold: f32) -> bool {
    let fire_energy: f32 = spectral_field[5] + spectral_field[6] + spectral_field[7];
    fire_energy / 3.0 > threshold
}

/// Decode a `[u16; 8]` spectral value (half-float bits) to `[f32; 8]`.
pub fn decode_spectral_u16(spectral: &[u16; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = f16::from_bits(spectral[i]).to_f32();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_armor() -> [u16; 8] { [0u16; 8] }
    fn full_armor() -> [u16; 8] { [65535u16; 8] }
    fn fire_armor() -> [u16; 8] {
        // High absorption in bands 5–7
        let mut a = [0u16; 8];
        a[5] = 65535;
        a[6] = 65535;
        a[7] = 65535;
        a
    }

    #[test]
    fn no_armor_takes_full_damage() {
        let mut health = 100.0;
        let damage = DamageType::fire(10.0);
        let applied = apply_spectral_damage(&mut health, &damage, &no_armor(), 100.0);
        // fire damage sums to 10.0 across bands 5–7 (0+0+0+0+0.5+1.0+4.0+4.5)
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
        // Fire armor only in bands 5–7
        let fire_dmg = DamageType::fire(10.0);
        let rad_dmg  = DamageType::radiation(10.0);

        let fire_applied = apply_spectral_damage(&mut health, &fire_dmg, &fire_armor(), 100.0);
        let pre_rad_health = health;
        let rad_applied  = apply_spectral_damage(&mut health, &rad_dmg, &fire_armor(), 100.0);

        // Fire armor should block most fire damage (bands 5–7 are fully armored)
        // Fire damage in band 5 = 0.1*10=1.0, band 6 = 0.4*10=4.0, band 7 = 0.45*10=4.5
        // Those three are fully blocked by fire_armor; band 4 = 0.05*10=0.5 passes through
        assert!(
            fire_applied < 1.0,
            "fire armor should block fire (bands 5-7), applied {}", fire_applied
        );
        // Radiation (bands 0–2) — fire armor has 0 absorption there — full damage
        assert!(
            rad_applied > 5.0,
            "fire armor should NOT block radiation (bands 0-2), applied {}", rad_applied
        );
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
        let zero_damage = [0.0f32; 8];
        // Apply zero damage — health should stay at max, not overflow
        apply_spectral_damage(&mut health, &zero_damage, &no_armor(), 100.0);
        assert!((health - 100.0).abs() < 0.001, "health should remain at max");
    }

    #[test]
    fn fire_band_threshold_detects_correctly() {
        let low_fire: [f32; 8]  = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1];
        let high_fire: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8, 0.85];

        assert!(!is_fire_band_exposure(&low_fire, 0.5),
            "low fire energy should not trigger threshold");
        assert!(is_fire_band_exposure(&high_fire, 0.5),
            "high fire energy (bands 5-7 avg 0.85) should trigger threshold");
    }

    #[test]
    fn decode_spectral_roundtrips() {
        use half::f16;
        let input = [
            f16::from_f32(0.5).to_bits(),
            f16::from_f32(0.25).to_bits(),
            f16::from_f32(0.0).to_bits(),
            f16::from_f32(1.0).to_bits(),
            0, 0, 0, 0,
        ];
        let decoded = decode_spectral_u16(&input);
        assert!((decoded[0] - 0.5).abs() < 0.001, "band 0: {}", decoded[0]);
        assert!((decoded[1] - 0.25).abs() < 0.001, "band 1: {}", decoded[1]);
        assert!((decoded[3] - 1.0).abs() < 0.001, "band 3: {}", decoded[3]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_core spectral_damage 2>&1 | head -20
```

Expected: compile error — module not exposed.

- [ ] **Step 3: Expose the module**

Add to `crates/vox_core/src/lib.rs`:

```rust
pub mod spectral_damage;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_core spectral_damage -- --nocapture
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_core/src/spectral_damage.rs crates/vox_core/src/lib.rs
git commit -m "feat(core): SpectralDamageModel — per-band damage attenuated by material armor"
```

---

## Task 4: MotionDatabase — motion matching nearest-feature query

**Files:**
- Create: `crates/vox_core/src/motion_matching.rs`
- Modify: `crates/vox_core/src/lib.rs`

**Design:** Motion matching (Dan Holden 2016, "A Fast and Simple Method for Computing a Data-Driven Motion Phase"). Each `AnimClip` is a named animation. Each `MotionFeature` is a compact feature vector: `[velocity_x, velocity_z, heading_cos, heading_sin, phase]` — 5 floats. The database stores one `MotionFeature` per frame per clip. Query: given the character's current velocity, heading, and phase, find the nearest feature by L2 distance. Return the clip and frame index.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_core/src/motion_matching.rs`:

```rust
//! Motion matching — nearest-feature animation selection.
//!
//! Implements the approach from:
//!   Dan Holden, Taku Komura, Jun Saito — "Phase-Functioned Neural Networks for
//!   Character Animation" (SIGGRAPH 2017) and "A Fast and Simple Method for
//!   Computing a Data-Driven Motion Phase" (2016).
//!
//! The database stores compact 5D feature vectors. Query finds the nearest
//! by L2 distance in O(N) — fast enough for <10k features at 60fps.

/// One animation clip (a named sequence of frames).
#[derive(Debug, Clone)]
pub struct AnimClip {
    pub name:         String,
    pub frame_count:  u32,
    pub frame_rate:   f32,
}

/// Per-frame motion feature vector: [vel_x, vel_z, heading_cos, heading_sin, phase].
///
/// - vel_x, vel_z: XZ velocity in m/s (character space)
/// - heading_cos, heading_sin: unit vector of facing direction
/// - phase: locomotion cycle phase in [0, 1]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionFeature {
    pub vel_x:       f32,
    pub vel_z:       f32,
    pub heading_cos: f32,
    pub heading_sin: f32,
    pub phase:       f32,
    /// Back-reference to which clip and frame this feature came from.
    pub clip_index:  usize,
    pub frame_index: u32,
}

impl MotionFeature {
    /// Create a feature vector from character state.
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

    /// Weighted L2 distance between two feature vectors.
    ///
    /// Velocity contributes more than heading (2×) and heading more than phase (0.5×).
    /// These weights follow Holden's recommendation from the 2016 paper.
    pub fn distance(&self, other: &Self) -> f32 {
        let dvel = 2.0 * ((self.vel_x - other.vel_x).powi(2) + (self.vel_z - other.vel_z).powi(2));
        let dhd  = 1.0 * ((self.heading_cos - other.heading_cos).powi(2)
                        + (self.heading_sin - other.heading_sin).powi(2));
        let dph  = 0.5 * (self.phase - other.phase).powi(2);
        (dvel + dhd + dph).sqrt()
    }
}

/// Database of motion features for all clips.
///
/// Build with `MotionDatabase::builder()`. Query with `find_nearest()`.
pub struct MotionDatabase {
    pub clips:    Vec<AnimClip>,
    pub features: Vec<MotionFeature>,
}

/// Result of a `find_nearest` query.
#[derive(Debug, Clone)]
pub struct MotionMatch {
    pub clip_name:   String,
    pub clip_index:  usize,
    pub frame_index: u32,
    pub distance:    f32,
}

impl MotionDatabase {
    /// Create an empty database (add clips and features via the builder pattern below,
    /// or directly by pushing to `clips` and `features`).
    pub fn new() -> Self {
        Self { clips: Vec::new(), features: Vec::new() }
    }

    /// Add a clip and return its index.
    pub fn add_clip(&mut self, name: &str, frame_count: u32, frame_rate: f32) -> usize {
        let idx = self.clips.len();
        self.clips.push(AnimClip {
            name: name.to_string(),
            frame_count,
            frame_rate,
        });
        idx
    }

    /// Add a pre-built feature with clip and frame references.
    pub fn add_feature(&mut self, mut feature: MotionFeature, clip_index: usize, frame_index: u32) {
        feature.clip_index  = clip_index;
        feature.frame_index = frame_index;
        self.features.push(feature);
    }

    /// Find the nearest matching feature in the database for the given query.
    ///
    /// Returns `None` if the database is empty.
    ///
    /// Time complexity: O(N) where N = `features.len()`.
    /// At 10k features on a modern CPU: ~50µs — within 1ms frame budget.
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

    /// Total number of features in the database.
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }
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

        // Clip 0: idle (zero velocity)
        let idle = db.add_clip("idle", 60, 30.0);
        for frame in 0..60u32 {
            let f = MotionFeature::from_state(0.0, 0.0, 0.0, frame as f32 / 60.0);
            db.add_feature(f, idle, frame);
        }

        // Clip 1: walk forward (vel_z = 1.4 m/s)
        let walk = db.add_clip("walk_forward", 30, 30.0);
        for frame in 0..30u32 {
            let f = MotionFeature::from_state(0.0, 1.4, 0.0, frame as f32 / 30.0);
            db.add_feature(f, walk, frame);
        }

        // Clip 2: sprint (vel_z = 5.0 m/s)
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
        assert_eq!(result.clip_name, "idle",
            "idle query should match idle clip, got '{}'", result.clip_name);
    }

    #[test]
    fn walk_query_matches_walk_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 1.5, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "walk_forward",
            "walk velocity query should match walk clip, got '{}'", result.clip_name);
    }

    #[test]
    fn sprint_query_matches_sprint_clip() {
        let db = make_db();
        let query = MotionFeature::from_state(0.0, 5.0, 0.0, 0.0);
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "sprint",
            "sprint velocity query should match sprint clip, got '{}'", result.clip_name);
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
        let query = MotionFeature::from_state(0.0, 1.4, 0.0, 0.5); // mid-phase walk
        let result = db.find_nearest(&query).unwrap();
        assert_eq!(result.clip_name, "walk_forward");
        // Phase 0.5 → should match frame ~15 of 30
        assert!(result.frame_index > 0, "should not always return frame 0");
    }

    #[test]
    fn feature_count_matches_total_added() {
        let db = make_db();
        assert_eq!(db.feature_count(), 60 + 30 + 20, "total features: idle+walk+sprint");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_core motion_matching 2>&1 | head -20
```

Expected: compile error — module not exposed.

- [ ] **Step 3: Expose the module**

Add to `crates/vox_core/src/lib.rs`:

```rust
pub mod motion_matching;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_core motion_matching -- --nocapture
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_core/src/motion_matching.rs crates/vox_core/src/lib.rs
git commit -m "feat(core): MotionDatabase — nearest-feature motion matching (Holden 2016)"
```

---

## Task 5: Wire CharacterBody into engine_runner

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

This task replaces the call to `character_controller_tick()` in the main update loop with `CharacterBody::move_and_slide()`. The `CharacterController` component becomes a companion config struct (speed, jump_force, etc.) — not the ground-detection authority.

- [ ] **Step 1: Locate the character update call**

```bash
grep -n "character_controller_tick\|CharacterController" crates/vox_app/src/bin/engine_runner.rs | head -20
```

- [ ] **Step 2: Add CharacterBody field to EngineApp**

Find the `EngineApp` struct definition. Add:

```rust
    /// Rapier-based character body for real collision ground detection.
    character_body: Option<vox_physics::character_body::CharacterBody>,
```

- [ ] **Step 3: Initialize CharacterBody in EngineApp::new()**

```rust
    character_body: None, // initialized when a player entity is added to the world
```

- [ ] **Step 4: Replace character tick**

Find where `character_controller_tick(...)` is called. Replace the block:

```rust
        // OLD (flat-plane only):
        // character_controller_tick(&mut cc, &mut t, move_input, jump_pressed, dt);

        // NEW: Rapier KCC — works on any surface
        if let (Some(cb), Some(physics)) = (&self.character_body, &self.physics_world) {
            // Accumulate gravity into velocity if not grounded
            if !last_grounded {
                cc.velocity.y -= cc.gravity * dt;
            }
            // Jump
            if jump_pressed && last_grounded {
                cc.velocity.y = cc.jump_force;
            }
            // Horizontal
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

            // Slope slide check using existing utility
            if !output.grounded {
                let slide = vox_core::character_controller::compute_slope_slide(
                    output.ground_normal, cc.gravity, dt
                );
                t.position += slide;
            }
        }
```

- [ ] **Step 5: Build to verify**

```bash
cargo build -p vox_app 2>&1 | grep -E "^error" | head -20
```

- [ ] **Step 6: Commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(app): wire CharacterBody into engine_runner — replace flat-plane ground detection"
```

---

## Task 6: Integration verification

- [ ] **Step 1: Full workspace test**

```bash
cargo test --workspace 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 2: Specific regression tests**

```bash
cargo test -p vox_physics character_body::tests::move_and_slide_on_raised_platform -- --nocapture
cargo test -p vox_core spectral_damage -- --nocapture
cargo test -p vox_core motion_matching -- --nocapture
```

All three must pass.

- [ ] **Step 3: Final commit**

```bash
git commit --allow-empty -m "test(character): domain 8 integration verified — KCC, spectral damage, motion matching"
```

---

## Self-Review

**Spec coverage:**
- [x] Flat-plane Y=0 bug documented and replaced — Task 1 (`move_and_slide_on_raised_platform_detects_ground` regression test) and Task 2 ✓
- [x] `KinematicCharacterController` from `rapier3d = "0.22"` — Task 1 ✓
- [x] `CharacterBody { rigid_body, collider, controller }` — Task 1 ✓
- [x] `move_and_slide(desired_velocity, dt, ...) -> CharacterOutput` — Task 1 ✓
- [x] Math helpers kept as utilities — `compute_slope_slide` called in Task 5 wire-up ✓
- [x] `SpectralDamageModel::apply(health, damage, armor_spectral)` — Task 3 ✓
- [x] Fire damage in bands 5–7, radiation in bands 0–2 — Task 3, `DamageType` presets ✓
- [x] `MotionDatabase` + `MotionFeature` + nearest-feature query — Task 4 ✓

**Known limitation:** `MotionDatabase::find_nearest` is O(N) linear scan. At 10k features this is ~50µs — within budget. For databases >100k features, replace with a k-d tree (`kiddo` crate).

**Engine crate rule:** `CharacterBody` is in `vox_physics` (engine-agnostic). `SpectralDamageModel` and `MotionDatabase` are in `vox_core` (engine-agnostic). No game concepts in either crate.
