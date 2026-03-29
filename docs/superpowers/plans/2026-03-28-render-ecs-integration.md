# Render ECS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the stubbed `gather_splats_system` in `EngineRuntime` so entities with `SplatAssetComponent` + `Visible` automatically contribute world-space splats to `RenderBuffer` each frame, and expose a standalone `SplatRenderPlugin` for use without the full `EngineRuntime`.

**Architecture:** A new `pub fn transform_splat` helper converts a local-space `GaussianSplat` to world space using a `TransformComponent`. The existing `gather_splats_system` stub in `engine_runtime.rs` is completed to query `SplatAssetComponent + Visible` entities and write transformed splats into `RenderBuffer`. A new `vox_render::render_ecs` module provides a normal bevy system (`splat_gather_system`) and `SplatRenderPlugin` for callers that use bevy_app directly instead of `EngineRuntime`.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `vox_core::types::GaussianSplat`, `vox_core::ecs::{SplatAssetComponent, TransformComponent, Visible}`, `vox_core::engine_runtime::RenderBuffer`

---

## Key Files (read before editing)

- `crates/vox_core/src/engine_runtime.rs` — `gather_splats_system` (stub at line ~402), `RenderBuffer`, `FrameStats`, `CameraState`
- `crates/vox_core/src/ecs.rs` — `SplatAssetComponent { uuid, splat_count, splats: Vec<GaussianSplat> }`, `TransformComponent { position: Vec3, rotation: Quat, scale: Vec3 }`, `Visible` marker
- `crates/vox_core/src/types.rs` — `GaussianSplat { position: [f32;3], scale: [f32;3], rotation: [i16;4], opacity: u8, _pad: [u8;3], spectral: [u16;8] }`
- `crates/vox_render/src/lib.rs` — add `pub mod render_ecs;`
- `crates/vox_render/Cargo.toml` — already has `bevy_ecs` + `bevy_app` + `vox_core`

## File Structure

**Modify:**
- `crates/vox_core/src/engine_runtime.rs` — add `transform_splat`, complete `gather_splats_system`, update `FrameStats`

**Create:**
- `crates/vox_render/src/render_ecs.rs` — standalone `splat_gather_system` + `SplatRenderPlugin`

**Modify:**
- `crates/vox_render/src/lib.rs` — add `pub mod render_ecs;`

---

### Task 1: `transform_splat` helper

**Files:**
- Modify: `crates/vox_core/src/engine_runtime.rs`

`transform_splat` converts a `GaussianSplat` from an entity's local space to world space by applying the entity's `TransformComponent` (scale → rotate → translate). The splat's own orientation (`rotation: [i16;4]`) is preserved unchanged — only position and scale are affected by the entity transform.

- [ ] **Step 1: Write failing tests** — add inside a `#[cfg(test)] mod tests` block AT THE BOTTOM of `crates/vox_core/src/engine_runtime.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use crate::ecs::TransformComponent;
    use crate::types::GaussianSplat;

    fn zero_splat(pos: [f32; 3]) -> GaussianSplat {
        GaussianSplat {
            position: pos,
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        }
    }

    fn identity_transform() -> TransformComponent {
        TransformComponent {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    #[test]
    fn transform_splat_translates_position() {
        let splat = zero_splat([1.0, 0.0, 0.0]);
        let transform = TransformComponent {
            position: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        let out = transform_splat(splat, &transform);
        assert!((out.position[0] - 1.0).abs() < 1e-5, "x unchanged");
        assert!((out.position[1] - 5.0).abs() < 1e-5, "y shifted by entity position");
        assert!((out.position[2] - 0.0).abs() < 1e-5, "z unchanged");
    }

    #[test]
    fn transform_splat_scales_position_and_splat_scale() {
        let splat = zero_splat([1.0, 0.0, 0.0]);
        let transform = TransformComponent {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(2.0),
        };
        let out = transform_splat(splat, &transform);
        // Local position [1,0,0] * scale 2 = [2,0,0]
        assert!((out.position[0] - 2.0).abs() < 1e-5, "position scaled");
        // Splat scale [0.1,0.1,0.1] * 2 = [0.2,0.2,0.2]
        assert!((out.scale[0] - 0.2).abs() < 1e-4, "splat scale doubled");
    }
}
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_core --lib -- engine_runtime::tests::transform_splat_translates_position 2>&1 | tail -5
```
Expected: FAIL — `transform_splat` not defined

- [ ] **Step 3: Implement** — add BEFORE `gather_splats_system` in `engine_runtime.rs`:

```rust
/// Convert a `GaussianSplat` from entity-local space to world space.
///
/// Applies `TransformComponent` as: scale → rotate → translate.
/// The splat's own `rotation` field (encoded orientation of the Gaussian
/// ellipsoid) is preserved — only `position` and `scale` are modified.
pub fn transform_splat(splat: GaussianSplat, transform: &TransformComponent) -> GaussianSplat {
    // Scale local position, then rotate, then translate.
    let local_pos = Vec3::from(splat.position);
    let scaled_pos = local_pos * transform.scale;
    let world_pos = transform.rotation * scaled_pos + transform.position;

    // Scale the splat's own Gaussian radii by entity scale.
    let s = transform.scale;
    let new_scale = [
        splat.scale[0] * s.x,
        splat.scale[1] * s.y,
        splat.scale[2] * s.z,
    ];

    GaussianSplat {
        position: world_pos.into(),
        scale: new_scale,
        ..splat
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_core --lib -- engine_runtime::tests 2>&1 | tail -5
```
Expected: `2 passed; 0 failed`

- [ ] **Step 5: Commit**

```bash
git add crates/vox_core/src/engine_runtime.rs
git commit -m "feat(render): transform_splat — local→world space for GaussianSplat"
```

---

### Task 2: Complete `gather_splats_system`

**Files:**
- Modify: `crates/vox_core/src/engine_runtime.rs`

Replace the stub body with a real implementation: query all `SplatAssetComponent` entities that have `Visible`, apply `transform_splat` per splat, fill `RenderBuffer.splats`. Also update `FrameStats.splat_count` and `FrameStats.visible_splats` so the HUD shows accurate numbers.

- [ ] **Step 1: Add failing tests** — append inside `mod tests` in `engine_runtime.rs`:

```rust
    #[test]
    fn gather_splats_fills_render_buffer() {
        use bevy_ecs::world::World;
        use crate::ecs::{SplatAssetComponent, TransformComponent, Visible};
        use uuid::Uuid;

        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());
        world.insert_resource(FrameStats::default());

        // Spawn a visible entity with 2 splats.
        let splats = vec![
            zero_splat([0.0, 0.0, 0.0]),
            zero_splat([1.0, 0.0, 0.0]),
        ];
        world.spawn((
            SplatAssetComponent { uuid: Uuid::nil(), splat_count: 2, splats },
            TransformComponent {
                position: Vec3::new(0.0, 10.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            Visible,
        ));

        gather_splats_system(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 2, "both splats should be gathered");
        // Entity at y=10, splat at y=0 → world y=10
        assert!((buffer.splats[0].position[1] - 10.0).abs() < 1e-5,
            "splat should be translated to entity world position");
    }

    #[test]
    fn gather_splats_skips_non_visible() {
        use bevy_ecs::world::World;
        use crate::ecs::{SplatAssetComponent, TransformComponent};
        use uuid::Uuid;

        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());
        world.insert_resource(FrameStats::default());

        // Spawn entity WITHOUT Visible marker.
        world.spawn((
            SplatAssetComponent { uuid: Uuid::nil(), splat_count: 1, splats: vec![zero_splat([0.0, 0.0, 0.0])] },
            TransformComponent { position: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE },
            // No Visible component
        ));

        gather_splats_system(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 0, "non-visible entity splats should be skipped");
    }
```

Note: `FrameStats` needs to be `Default`. Check that `FrameStats` implements `Default` in the file (it already does via `#[derive(Default)]`). If `FrameStats` is not a Resource, add `impl Resource for FrameStats {}`.

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_core --lib -- engine_runtime::tests::gather_splats_fills_render_buffer 2>&1 | tail -5
```
Expected: FAIL (stub doesn't gather splats)

- [ ] **Step 3: Implement** — replace the body of `gather_splats_system` (lines ~402–426):

```rust
/// Gather splats from visible entities into the render buffer.
fn gather_splats_system(world: &mut World) {
    let mut render_buffer = world.resource_mut::<RenderBuffer>();
    render_buffer.splats.clear();
    render_buffer.lights.clear();

    // Gather lights from point light entities.
    let lights: Vec<LightData> = {
        let mut query = world.query::<(&TransformComponent, &PointLightComponent)>();
        query.iter(world)
            .map(|(t, l)| LightData {
                position: t.position,
                color: l.color,
                intensity: l.intensity,
                radius: l.radius,
            })
            .collect()
    };

    // Gather splats from visible entities that carry SplatAssetComponent.
    let gathered: Vec<GaussianSplat> = {
        let mut query = world.query_filtered::<
            (&SplatAssetComponent, &TransformComponent),
            With<Visible>,
        >();
        query.iter(world)
            .flat_map(|(asset, transform)| {
                asset.splats.iter().map(|&splat| transform_splat(splat, transform))
            })
            .collect()
    };

    let visible_count = gathered.len() as u32;

    let mut render_buffer = world.resource_mut::<RenderBuffer>();
    render_buffer.lights = lights;
    render_buffer.splats = gathered;

    // Update frame stats if available.
    if let Some(mut stats) = world.get_resource_mut::<FrameStats>() {
        stats.splat_count = visible_count;
        stats.visible_splats = visible_count;
    }
}
```

You'll need to add `Visible` to the imports at the top of the function scope, or ensure `use crate::ecs::*` already covers it (it should via the existing `use crate::ecs::*;` at the top of the module).

- [ ] **Step 4: Run all engine_runtime tests**

```bash
cargo test -p vox_core --lib -- engine_runtime::tests 2>&1 | tail -8
```
Expected: `4 passed; 0 failed`

- [ ] **Step 5: Full vox_core suite**

```bash
cargo test -p vox_core --lib 2>&1 | grep "test result"
```
Expected: `0 failed`

- [ ] **Step 6: Commit**

```bash
git add crates/vox_core/src/engine_runtime.rs
git commit -m "feat(render): complete gather_splats_system — SplatAssetComponent+Visible → RenderBuffer"
```

---

### Task 3: `SplatRenderPlugin` in `vox_render`

**Files:**
- Create: `crates/vox_render/src/render_ecs.rs`
- Modify: `crates/vox_render/src/lib.rs`

A standalone bevy_app Plugin that uses the normal (non-exclusive) system API for callers that manage a bevy_app directly rather than using `EngineRuntime`. Uses the same `transform_splat` helper from Task 1.

- [ ] **Step 1: Add `pub mod render_ecs;` to `crates/vox_render/src/lib.rs`**

Open the file. After `pub mod seq_ecs;` add:
```rust
pub mod render_ecs;
```

- [ ] **Step 2: Create `crates/vox_render/src/render_ecs.rs`**:

```rust
//! Standalone ECS integration for splat rendering.
//!
//! `SplatRenderPlugin` provides a `splat_gather_system` that reads
//! `SplatAssetComponent + Visible` entities and writes world-space splats
//! into the `RenderBuffer` resource each frame.
//!
//! Usage:
//! ```rust,ignore
//! app.insert_resource(RenderBuffer::default());
//! app.add_plugins(SplatRenderPlugin);
//! ```

use bevy_ecs::prelude::*;
use vox_core::ecs::{SplatAssetComponent, TransformComponent, Visible};
use vox_core::engine_runtime::{transform_splat, RenderBuffer};

// ── System ─────────────────────────────────────────────────────────────────

/// Gather world-space splats from all visible `SplatAssetComponent` entities
/// into `RenderBuffer`.
///
/// Clears `RenderBuffer.splats` each frame before re-gathering — call once
/// per frame before the render pass.
pub fn splat_gather_system(
    mut buffer: ResMut<RenderBuffer>,
    query: Query<(&SplatAssetComponent, &TransformComponent), With<Visible>>,
) {
    buffer.splats.clear();
    for (asset, transform) in query.iter() {
        for &splat in &asset.splats {
            buffer.splats.push(transform_splat(splat, transform));
        }
    }
}

// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that inserts `RenderBuffer` and registers `splat_gather_system`
/// in `Update`.
///
/// Callers must NOT also insert `RenderBuffer` manually — the plugin owns it.
pub struct SplatRenderPlugin;

impl bevy_app::Plugin for SplatRenderPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(RenderBuffer::default());
        app.add_systems(bevy_app::Update, splat_gather_system);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;
    use bevy_ecs::world::World;
    use glam::{Quat, Vec3};
    use uuid::Uuid;
    use vox_core::types::GaussianSplat;

    fn zero_splat() -> GaussianSplat {
        GaussianSplat {
            position: [0.0, 0.0, 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 255,
            _pad: [0; 3],
            spectral: [0; 8],
        }
    }

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(SplatRenderPlugin);
        // Panicking during build counts as test failure.
    }

    #[test]
    fn splat_gather_system_collects_visible_splats() {
        use bevy_ecs::schedule::Schedule;

        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());

        // Spawn visible entity with 3 splats at local origin.
        world.spawn((
            SplatAssetComponent {
                uuid: Uuid::nil(),
                splat_count: 3,
                splats: vec![zero_splat(), zero_splat(), zero_splat()],
            },
            TransformComponent {
                position: Vec3::new(5.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            Visible,
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(splat_gather_system);
        schedule.run(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 3, "all 3 splats should be gathered");
        // Entity at x=5, splat at x=0 → world x=5
        assert!(
            (buffer.splats[0].position[0] - 5.0).abs() < 1e-5,
            "splat x should match entity world x"
        );
    }

    #[test]
    fn splat_gather_system_ignores_invisible() {
        use bevy_ecs::schedule::Schedule;

        let mut world = World::new();
        world.insert_resource(RenderBuffer::default());

        // Entity WITHOUT Visible.
        world.spawn((
            SplatAssetComponent {
                uuid: Uuid::nil(),
                splat_count: 1,
                splats: vec![zero_splat()],
            },
            TransformComponent {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(splat_gather_system);
        schedule.run(&mut world);

        let buffer = world.resource::<RenderBuffer>();
        assert_eq!(buffer.splats.len(), 0, "invisible entity should be skipped");
    }
}
```

- [ ] **Step 3: Verify compile**

```bash
cargo check -p vox_render 2>&1 | tail -5
```
Expected: clean

- [ ] **Step 4: Run render_ecs tests**

```bash
cargo test -p vox_render --lib -- render_ecs 2>&1 | tail -8
```
Expected: `4 passed; 0 failed`

- [ ] **Step 5: Full vox_render suite**

```bash
cargo test -p vox_render --lib 2>&1 | grep "test result"
```
Expected: `0 failed`

- [ ] **Step 6: Commit**

```bash
git add crates/vox_render/src/render_ecs.rs crates/vox_render/src/lib.rs
git commit -m "feat(render): SplatRenderPlugin + splat_gather_system — standalone ECS render integration"
```

---

## Self-Review

**Spec coverage:**
- ✅ `transform_splat(splat, transform) -> GaussianSplat` — scale→rotate→translate, preserve spectral/opacity/rotation → Task 1
- ✅ Tests: translate, scale → Task 1
- ✅ `gather_splats_system` — queries `SplatAssetComponent + Visible`, calls `transform_splat`, writes to `RenderBuffer.splats` → Task 2
- ✅ `gather_splats_system` clears buffer each frame → Task 2
- ✅ `gather_splats_system` updates `FrameStats.splat_count` + `visible_splats` → Task 2
- ✅ Keeps light gathering (existing behavior preserved) → Task 2
- ✅ Tests: fills buffer, skips non-visible → Task 2
- ✅ `splat_gather_system` (normal system API) → Task 3
- ✅ `SplatRenderPlugin` inserts `RenderBuffer`, registers `splat_gather_system` → Task 3
- ✅ Tests: plugin builds, collects visible, ignores invisible → Task 3

**Placeholder scan:** No TBDs. All implementations shown in full.

**Type consistency:**
- `transform_splat(splat: GaussianSplat, transform: &TransformComponent) -> GaussianSplat` — defined Task 1, called in Task 2 (engine_runtime) and Task 3 (render_ecs) ✅
- `RenderBuffer { splats: Vec<GaussianSplat>, lights: Vec<LightData> }` — used in Task 2 and Task 3 ✅
- `SplatAssetComponent { uuid, splat_count, splats: Vec<GaussianSplat> }` — field names consistent ✅
- `FrameStats` — must implement `Resource`; in Task 2 we use `world.get_resource_mut::<FrameStats>()` (optional get, so it won't panic if absent) ✅
- `With<Visible>` filter — `Visible` is a marker component from `vox_core::ecs` covered by `use crate::ecs::*` ✅
