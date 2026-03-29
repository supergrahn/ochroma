# Animation NPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a visible animated entity (procedural walk-cycle NPC) appear in the engine using the ECS animation system.

**Architecture:** Add `ProceduralWalkComponent` ECS component that stores base splat positions + time. An `animation_system` advances time and writes bobbing `GaussianSplat`s to `RenderBuffer` each frame. Wire into `engine_runner.rs` to spawn one NPC entity.

**Tech Stack:** bevy_ecs 0.16, bevy_app 0.16, glam, `vox_core::engine_runtime::{RenderBuffer, FrameTime}`

---

## Key Facts (read before implementing)

- `GaussianSplat` in `crates/vox_core/src/types.rs` has fields: `position: [f32; 3]`, `scale: [f32; 3]`, `rotation: [i16; 4]`, `opacity: u8`, `_pad: [u8; 3]`, `spectral: [u16; 8]`.
- `RenderBuffer` in `crates/vox_core/src/engine_runtime.rs` is a Bevy `Resource` with `pub splats: Vec<GaussianSplat>`. It is cleared and repopulated by `gather_splats_system` every frame — the animation system must run **after** `gather_splats_system` so its splats are not wiped.
- `FrameTime` in `crates/vox_core/src/engine_runtime.rs` is a Bevy `Resource` with `pub dt: f32`.
- `EngineRuntime::world` is a public `bevy_ecs::world::World`. Direct world spawning via `self.engine.world.spawn(...)` is valid.
- `engine_runner.rs` calls `self.engine.spawn("NPC").with_position(...)` for engine-tracked entities; for attaching custom Bevy components, use `self.engine.world.spawn(bundle)` directly.
- `gather_splats_system` in `engine_runtime.rs` clears `render_buffer.splats` at the start of each frame then repopulates from `SplatAssetComponent` entities. Our animation system must push additional splats **after** this clear, not before.
- The existing `AnimationDriver` in `crates/vox_render/src/animation_driver.rs` requires real GLTF data — do not use it here.

---

## Task 1 — Create `crates/vox_render/src/walk_animation.rs`

- [ ] Create a new file `crates/vox_render/src/walk_animation.rs` with the complete implementation below.

```rust
//! Procedural walk-cycle animation system.
//!
//! Provides `ProceduralWalkComponent` — an ECS component that stores a set of
//! base splat positions and an accumulated time. `animation_system` advances
//! time each frame, computes a sinusoidal vertical bob, and pushes the
//! resulting `GaussianSplat`s into the `RenderBuffer`.
//!
//! No skeleton or GLTF data required — suitable for demos and placeholder NPCs.

use bevy_ecs::prelude::*;
use glam::Vec3;
use vox_core::engine_runtime::{FrameTime, RenderBuffer};
use vox_core::types::GaussianSplat;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/// Procedural walk-cycle component. Attach to any entity that should appear as
/// an animated humanoid blob in the render output.
#[derive(Component, Debug, Clone)]
pub struct ProceduralWalkComponent {
    /// Base world positions of the NPC's splats (no transform applied).
    pub base_positions: Vec<[f32; 3]>,
    /// Accumulated animation time in seconds.
    pub time: f32,
    /// Vertical bob amplitude in meters.
    pub bob_amplitude: f32,
    /// Bob frequency in Hz.
    pub bob_frequency: f32,
}

impl ProceduralWalkComponent {
    /// Create a simple humanoid-ish blob: 8 splats arranged in a rough body shape.
    ///
    /// `center` is the world-space root position (hip level).
    pub fn humanoid_blob(center: Vec3) -> Self {
        let offsets: &[[f32; 3]] = &[
            [0.0,   0.0,  0.0],   // torso center
            [0.0,   0.4,  0.0],   // chest
            [0.0,   0.8,  0.0],   // head
            [-0.2,  0.2,  0.0],   // left shoulder
            [0.2,   0.2,  0.0],   // right shoulder
            [-0.15,-0.4,  0.0],   // left hip
            [0.15, -0.4,  0.0],   // right hip
            [0.0,  -0.7,  0.0],   // feet
        ];
        let base_positions = offsets
            .iter()
            .map(|o| [center.x + o[0], center.y + o[1], center.z + o[2]])
            .collect();
        Self {
            base_positions,
            time: 0.0,
            bob_amplitude: 0.05,
            bob_frequency: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// Advances each `ProceduralWalkComponent` by `dt` and appends its bobbing
/// splats to the `RenderBuffer`.
///
/// **Scheduling:** this system must run *after* `gather_splats_system` in the
/// engine frame schedule so that the per-entity splats pushed here are not
/// cleared by that system. Register it after calling `engine.tick()` flushes
/// the gather pass, or add it to the frame schedule after `gather_splats_system`.
pub fn animation_system(
    time: Res<FrameTime>,
    mut render_buffer: ResMut<RenderBuffer>,
    mut query: Query<&mut ProceduralWalkComponent>,
) {
    let dt = time.dt;
    for mut npc in query.iter_mut() {
        npc.time += dt;
        let bob = (npc.time * npc.bob_frequency * std::f32::consts::TAU).sin()
            * npc.bob_amplitude;
        for base in &npc.base_positions {
            render_buffer.splats.push(GaussianSplat {
                position: [base[0], base[1] + bob, base[2]],
                scale: [0.12, 0.12, 0.12],
                rotation: [0i16, 0, 0, 32767],
                opacity: 200,
                _pad: [0; 3],
                spectral: [15000u16; 8], // bright white-ish in all spectral bands
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::*;
    use vox_core::engine_runtime::{FrameTime, RenderBuffer};

    fn default_frame_time() -> FrameTime {
        FrameTime { dt: 0.016, total: 0.0, frame: 0 }
    }

    // -----------------------------------------------------------------------
    // Component shape tests
    // -----------------------------------------------------------------------

    #[test]
    fn walk_component_humanoid_blob_has_8_splats() {
        let comp = ProceduralWalkComponent::humanoid_blob(Vec3::ZERO);
        assert_eq!(
            comp.base_positions.len(),
            8,
            "humanoid_blob must produce exactly 8 base positions"
        );
    }

    #[test]
    fn walk_component_humanoid_blob_centers_on_given_position() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let comp = ProceduralWalkComponent::humanoid_blob(center);
        // Torso center offset is [0,0,0] so first splat == center
        let torso = comp.base_positions[0];
        assert!(
            (torso[0] - center.x).abs() < 1e-5,
            "torso x should match center.x"
        );
        assert!(
            (torso[1] - center.y).abs() < 1e-5,
            "torso y should match center.y"
        );
        assert!(
            (torso[2] - center.z).abs() < 1e-5,
            "torso z should match center.z"
        );
    }

    // -----------------------------------------------------------------------
    // System integration tests
    // -----------------------------------------------------------------------

    fn build_world_with_npc() -> World {
        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());
        world.insert_resource(default_frame_time());
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::ZERO));
        world
    }

    #[test]
    fn animation_system_pushes_splats_to_render_buffer() {
        let mut world = build_world_with_npc();

        let mut system = IntoSystem::into_system(animation_system);
        system.initialize(&mut world);
        system.run((), &mut world);
        system.apply_deferred(&mut world);

        let rb = world.resource::<RenderBuffer>();
        assert_eq!(
            rb.splats.len(),
            8,
            "animation_system must push 8 splats (one per base_position) into RenderBuffer"
        );
    }

    #[test]
    fn animation_system_splat_opacity_and_spectral_are_set() {
        let mut world = build_world_with_npc();

        let mut system = IntoSystem::into_system(animation_system);
        system.initialize(&mut world);
        system.run((), &mut world);
        system.apply_deferred(&mut world);

        let rb = world.resource::<RenderBuffer>();
        for splat in &rb.splats {
            assert_eq!(splat.opacity, 200, "opacity must be 200");
            assert_eq!(splat.spectral, [15000u16; 8], "spectral must be all-15000");
        }
    }

    #[test]
    fn bob_offset_changes_with_time() {
        // Run the system once at t=0.016, record Y positions.
        let mut world_a = build_world_with_npc();
        {
            let mut system = IntoSystem::into_system(animation_system);
            system.initialize(&mut world_a);
            system.run((), &mut world_a);
            system.apply_deferred(&mut world_a);
        }
        let y_at_dt: Vec<f32> = world_a
            .resource::<RenderBuffer>()
            .splats
            .iter()
            .map(|s| s.position[1])
            .collect();

        // Build a second world, set a very different time (quarter period of 2Hz bob
        // at TAU*0.5*2Hz = TAU => sin=0, so use 0.125s for sin(TAU*0.25) = 1.0 peak).
        let mut world_b = World::new();
        world_b.insert_resource(RenderBuffer::default());
        world_b.insert_resource(FrameTime { dt: 0.125, total: 0.0, frame: 0 });
        world_b.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::ZERO));
        {
            let mut system = IntoSystem::into_system(animation_system);
            system.initialize(&mut world_b);
            system.run((), &mut world_b);
            system.apply_deferred(&mut world_b);
        }
        let y_at_peak: Vec<f32> = world_b
            .resource::<RenderBuffer>()
            .splats
            .iter()
            .map(|s| s.position[1])
            .collect();

        // At dt=0.016s: time=0.016, bob=sin(0.016*2*TAU)*0.05 ≈ small positive
        // At dt=0.125s: time=0.125, bob=sin(0.125*2*TAU)*0.05 = sin(TAU/4)*0.05 = 0.05
        // The Y values must differ by more than floating-point noise.
        let any_differ = y_at_dt
            .iter()
            .zip(y_at_peak.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            any_differ,
            "Y positions must differ between different animation times; \
             got dt_y={:?}, peak_y={:?}",
            y_at_dt, y_at_peak
        );
    }

    #[test]
    fn animation_system_accumulates_time_on_component() {
        let mut world = build_world_with_npc();

        let mut system = IntoSystem::into_system(animation_system);
        system.initialize(&mut world);
        // Run twice
        system.run((), &mut world);
        system.apply_deferred(&mut world);
        system.run((), &mut world);
        system.apply_deferred(&mut world);

        let time_accumulated: f32 = {
            let mut q = world.query::<&ProceduralWalkComponent>();
            q.iter(&world).next().unwrap().time
        };
        let expected = 0.016 * 2.0;
        assert!(
            (time_accumulated - expected).abs() < 1e-5,
            "time should accumulate: expected {}, got {}",
            expected,
            time_accumulated
        );
    }

    #[test]
    fn multiple_npc_entities_all_push_splats() {
        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());
        world.insert_resource(default_frame_time());
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::new(0.0, 0.0, -3.0)));
        world.spawn(ProceduralWalkComponent::humanoid_blob(Vec3::new(5.0, 0.0, -3.0)));

        let mut system = IntoSystem::into_system(animation_system);
        system.initialize(&mut world);
        system.run((), &mut world);
        system.apply_deferred(&mut world);

        let rb = world.resource::<RenderBuffer>();
        assert_eq!(
            rb.splats.len(),
            16,
            "two NPC entities with 8 splats each must yield 16 total splats"
        );
    }
}
```

---

## Task 2 — Expose the module in `crates/vox_render/src/lib.rs`

- [ ] Open `crates/vox_render/src/lib.rs` and add one line after `pub mod animation_driver;` (line 64):

```rust
pub mod walk_animation;
```

The resulting block should look like:

```rust
pub mod animation_driver;
pub mod walk_animation;
pub mod multi_viewport;
```

---

## Task 3 — Wire into `engine_runner.rs`

- [ ] In `crates/vox_app/src/bin/engine_runner.rs`, add the import at the top of the use-block:

```rust
use vox_render::walk_animation::{animation_system, ProceduralWalkComponent};
```

- [ ] In `build_scene()`, after the last `self.engine.spawn(...)` call (approximately after the Light2 spawn), spawn one NPC entity directly into the Bevy world:

```rust
// Spawn procedural walk-cycle NPC at (0, 0, -3)
self.engine.world.spawn(
    ProceduralWalkComponent::humanoid_blob(glam::Vec3::new(0.0, 0.0, -3.0))
);
println!("[ochroma] Spawned procedural walk NPC at (0, 0, -3)");
```

- [ ] In `EngineApp`'s per-frame update (the method that calls `self.engine.tick(dt)` and then assembles splats for rendering), run the animation system **after** the engine tick so it appends to the already-cleared-and-gathered render buffer:

  Find the place where frame splats are assembled (the code that builds the list passed to the rasteriser). After `self.engine.tick(dt)` and before the splat read-back, run:

```rust
// Run procedural animation — must run after engine.tick() clears + gathers splats
{
    use bevy_ecs::system::IntoSystem;
    let mut sys = IntoSystem::into_system(animation_system);
    sys.initialize(&mut self.engine.world);
    sys.run((), &mut self.engine.world);
    sys.apply_deferred(&mut self.engine.world);
}
```

  Then read `self.engine.world.resource::<vox_core::engine_runtime::RenderBuffer>().splats` as the combined splat list (or extend `scene_splats` from it, depending on the existing frame assembly pattern).

  > **Note:** If the codebase already runs one-shot systems via a helper (e.g. `self.engine.run_system(...)`), use that instead of the raw `IntoSystem` dance. The raw form above is always valid with bevy_ecs 0.16.

---

## Verification

After implementation, confirm:

1. `cargo test -p vox_render walk_animation` — all 6 tests pass.
2. `cargo build --bin ochroma` — no compile errors.
3. Visually: running the engine shows a small cluster of bright white-ish blobs near the origin, gently bobbing up and down at ~2Hz.
