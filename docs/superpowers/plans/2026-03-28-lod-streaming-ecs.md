# LOD Streaming ECS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing CLAS, HierarchicalLOD, LodCrossfadeManager, MegaGeometry, and TileManager systems into a bevy_ecs `LodStreamingPlugin` so any entity with `TransformComponent + LodStateComponent` automatically gets distance-based LOD selection, crossfade transitions, and tile activation.

**Architecture:** A new `vox_render::lod_ecs` module provides `LodStreamingPlugin`, three ordered systems (`lod_select_system` → `lod_crossfade_system` → `tile_streaming_system`), and two resources (`CameraSettings`, `LodStreamingSettings`). The existing standalone systems (CLAS, HierarchicalLOD, MegaGeometry) are unchanged — this layer simply drives them from ECS. `LodStateComponent` is added to `vox_core` as the per-entity output written by the systems.

**Tech Stack:** `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `vox_render::hierarchical_lod::{select_lod_level, crossfade_factor}`, `vox_render::lod_crossfade::LodCrossfadeManager`, `vox_render::streaming::TileManager`, `vox_core::ecs::TransformComponent`, `vox_core::lwc::{TileCoord, WorldCoord, TILE_SIZE}`

---

## Key File Paths (read before editing)

- `crates/vox_core/src/ecs.rs` — add `LodStateComponent`
- `crates/vox_render/Cargo.toml` — add `bevy_ecs`, `bevy_app` deps
- `crates/vox_render/src/lib.rs` — add `pub mod lod_ecs;`
- `crates/vox_render/src/lod_ecs.rs` — **CREATE**: all resources, systems, plugin
- `crates/vox_render/src/hierarchical_lod.rs` — `select_lod_level(distance, screen_size) -> u32`
- `crates/vox_render/src/lod_crossfade.rs` — `LodCrossfadeManager::request_lod_change`, `tick`
- `crates/vox_render/src/streaming.rs` — `TileManager::update_camera(TileCoord)`
- `crates/vox_core/src/lwc.rs` — `WorldCoord::from_absolute`, `TileCoord`, `TILE_SIZE`

## File Structure

**Create:**
- `crates/vox_render/src/lod_ecs.rs` — resources + systems + plugin (single file, all ECS logic)

**Modify:**
- `crates/vox_core/src/ecs.rs` — add `LodStateComponent`
- `crates/vox_render/Cargo.toml` — add bevy_ecs + bevy_app
- `crates/vox_render/src/lib.rs` — add `pub mod lod_ecs;`

---

### Task 1: LodStateComponent in vox_core

**Files:**
- Modify: `crates/vox_core/src/ecs.rs`

- [ ] **Step 1: Read the file**

Run: `cat -n crates/vox_core/src/ecs.rs | tail -20`

Confirm `AudioEmitterComponent` is the last component (around line 94). Find the end of the file.

- [ ] **Step 2: Add LodStateComponent after AudioEmitterComponent**

Open `crates/vox_core/src/ecs.rs`. After the `AudioEmitterComponent` struct block and before `PointLightComponent`, insert:

```rust
/// Per-entity LOD state written by lod_select_system.
/// Attach this component to any entity that should receive automatic LOD selection.
/// `current_level` maps to the 4-level hierarchy in `vox_render::hierarchical_lod`:
///   0 = full detail, 1 = 40% splats, 2 = 10% splats, 3 = billboard.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct LodStateComponent {
    /// Current LOD level index (0–3).
    pub current_level: u8,
    /// Crossfade blending weight towards the next level (0.0 = stable, 1.0 = fully transitioned).
    pub crossfade: f32,
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_core 2>&1 | tail -8`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/vox_core/src/ecs.rs
git commit -m "feat(lod): LodStateComponent ECS component for 4-level LOD state"
```

---

### Task 2: Add bevy_ecs/bevy_app to vox_render + skeleton lod_ecs.rs

**Files:**
- Modify: `crates/vox_render/Cargo.toml`
- Modify: `crates/vox_render/src/lib.rs`
- Create: `crates/vox_render/src/lod_ecs.rs`

- [ ] **Step 1: Add deps to vox_render/Cargo.toml**

Open `crates/vox_render/Cargo.toml`. In `[dependencies]` add:

```toml
bevy_ecs = { workspace = true }
bevy_app = { workspace = true }
vox_core = { path = "../vox_core" }
```

`vox_core` is already present — skip adding it if so.

- [ ] **Step 2: Add pub mod lod_ecs to lib.rs**

Open `crates/vox_render/src/lib.rs`. After the `pub mod mega_geometry;` line add:

```rust
pub mod lod_ecs;
```

- [ ] **Step 3: Create lod_ecs.rs with resources only**

Create `crates/vox_render/src/lod_ecs.rs`:

```rust
//! ECS integration for LOD streaming.
//!
//! Provides `LodStreamingPlugin` which wires the existing CLAS, HierarchicalLOD,
//! LodCrossfadeManager, and TileManager systems into bevy_ecs.

use bevy_ecs::prelude::*;
use glam::Vec3;

use vox_core::lwc::{TileCoord, WorldCoord, TILE_SIZE};
use crate::lod_crossfade::LodCrossfadeManager;
use crate::streaming::TileManager;

// ── Resources ──────────────────────────────────────────────────────────────

/// Camera state used by lod_select_system to compute screen-space sizes.
/// Update this each frame from your camera transform before running systems.
#[derive(Resource, Debug, Clone)]
pub struct CameraSettings {
    /// Camera world position.
    pub position: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Render target height in pixels.
    pub screen_height: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            fov_y: std::f32::consts::FRAC_PI_4, // 45°
            screen_height: 1080.0,
        }
    }
}

/// Per-frame delta-time used by lod_crossfade_system.
/// Update this each frame with your frame duration in seconds.
#[derive(Resource, Debug, Clone, Copy)]
pub struct TimeStep(pub f32);

impl Default for TimeStep {
    fn default() -> Self { Self(1.0 / 60.0) }
}

// Allow these types to be stored as bevy_ecs Resources.
impl Resource for LodCrossfadeManager {}
impl Resource for TileManager {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_settings_default() {
        let s = CameraSettings::default();
        assert_eq!(s.position, Vec3::ZERO);
        assert!(s.fov_y > 0.0);
        assert!(s.screen_height > 0.0);
    }

    #[test]
    fn timestep_default_is_60hz() {
        let dt = TimeStep::default();
        assert!((dt.0 - 1.0 / 60.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 4: Verify compile**

Run: `cargo check -p vox_render 2>&1 | tail -8`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/Cargo.toml crates/vox_render/src/lib.rs crates/vox_render/src/lod_ecs.rs
git commit -m "feat(lod): lod_ecs skeleton with CameraSettings + TimeStep resources"
```

---

### Task 3: lod_select_system

**Files:**
- Modify: `crates/vox_render/src/lod_ecs.rs`

`lod_select_system` queries every entity with `TransformComponent + LodStateComponent`, computes distance to camera, projects a unit bounding sphere to screen pixels, calls `hierarchical_lod::select_lod_level(distance, screen_size)`, and if the level changed requests a crossfade transition.

- [ ] **Step 1: Write the failing tests first**

Add to the `#[cfg(test)] mod tests` block at the bottom of `lod_ecs.rs`:

```rust
    #[test]
    fn close_entity_selects_lod0() {
        let mut world = World::new();
        let mut cam = CameraSettings::default();
        cam.position = Vec3::ZERO;
        world.insert_resource(cam);
        world.insert_resource(LodCrossfadeManager { transitions: vec![], transition_duration: 0.5 });

        let entity = world.spawn((
            vox_core::ecs::TransformComponent {
                position: Vec3::new(0.0, 0.0, 10.0), // 10 m away
                ..Default::default()
            },
            vox_core::ecs::LodStateComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(lod_select_system);
        schedule.run(&mut world);

        let lod = world.entity(entity).get::<vox_core::ecs::LodStateComponent>().unwrap();
        assert_eq!(lod.current_level, 0, "10 m away should be LOD 0, got {}", lod.current_level);
    }

    #[test]
    fn distant_entity_selects_lod3() {
        let mut world = World::new();
        let mut cam = CameraSettings::default();
        cam.position = Vec3::ZERO;
        world.insert_resource(cam);
        world.insert_resource(LodCrossfadeManager { transitions: vec![], transition_duration: 0.5 });

        let entity = world.spawn((
            vox_core::ecs::TransformComponent {
                position: Vec3::new(0.0, 0.0, 500.0), // 500 m away → beyond 400 m threshold
                ..Default::default()
            },
            vox_core::ecs::LodStateComponent::default(),
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(lod_select_system);
        schedule.run(&mut world);

        let lod = world.entity(entity).get::<vox_core::ecs::LodStateComponent>().unwrap();
        assert_eq!(lod.current_level, 3, "500 m away should be LOD 3, got {}", lod.current_level);
    }
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cargo test -p vox_render lod_ecs 2>&1 | tail -10`
Expected: FAIL — `lod_select_system` not yet defined

- [ ] **Step 3: Implement lod_select_system**

Add these after the `impl Resource for TileManager {}` line in `lod_ecs.rs`:

```rust
// ── Systems ────────────────────────────────────────────────────────────────

/// Compute screen-space projected size in pixels for a unit-radius bounding sphere.
fn projected_pixels(distance: f32, fov_y: f32, screen_height: f32) -> f32 {
    if distance < 0.001 {
        return screen_height;
    }
    let half_fov_tan = (fov_y * 0.5).tan();
    // radius = 1.0 m (unit sphere); scale for assets with known bounds later
    (1.0 / (distance * half_fov_tan)) * (screen_height * 0.5)
}

/// For each entity with TransformComponent + LodStateComponent, select the appropriate
/// LOD level based on camera distance and projected screen size. Requests a crossfade
/// transition when the level changes.
pub fn lod_select_system(
    camera: Res<CameraSettings>,
    mut crossfade: ResMut<LodCrossfadeManager>,
    mut query: Query<(Entity, &vox_core::ecs::TransformComponent, &mut vox_core::ecs::LodStateComponent)>,
) {
    for (entity, transform, mut lod_state) in query.iter_mut() {
        let distance = (transform.position - camera.position).length();
        let screen_size = projected_pixels(distance, camera.fov_y, camera.screen_height);
        let new_level = crate::hierarchical_lod::select_lod_level(distance, screen_size);
        let new_level = new_level as u8;

        if new_level != lod_state.current_level {
            crossfade.request_lod_change(
                entity.index(),
                lod_state.current_level as u32,
                new_level as u32,
            );
            lod_state.current_level = new_level;
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_render lod_ecs 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/lod_ecs.rs
git commit -m "feat(lod): lod_select_system — distance-based 4-level LOD selection"
```

---

### Task 4: lod_crossfade_system

**Files:**
- Modify: `crates/vox_render/src/lod_ecs.rs`

`lod_crossfade_system` advances all active crossfade transitions by the current timestep and writes the resulting crossfade weight back into `LodStateComponent.crossfade`.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn crossfade_progresses_over_ticks() {
        let mut world = World::new();
        world.insert_resource(LodCrossfadeManager { transitions: vec![], transition_duration: 1.0 });
        world.insert_resource(TimeStep(0.1)); // 0.1 s per tick
        world.insert_resource(CameraSettings::default());

        // Spawn a high-LOD entity far away so lod_select picks LOD 3
        let entity = world.spawn((
            vox_core::ecs::TransformComponent {
                position: Vec3::new(0.0, 0.0, 500.0),
                ..Default::default()
            },
            vox_core::ecs::LodStateComponent { current_level: 0, crossfade: 0.0 },
        )).id();

        let mut schedule = Schedule::default();
        schedule.add_systems((lod_select_system, lod_crossfade_system).chain());

        // Tick once — lod_select requests a 0→3 transition; lod_crossfade advances it
        schedule.run(&mut world);

        let lod = world.entity(entity).get::<vox_core::ecs::LodStateComponent>().unwrap();
        // After one 0.1 s tick on a 1.0 s transition: progress ≈ 0.1
        assert!(
            lod.crossfade > 0.0 && lod.crossfade <= 1.0,
            "crossfade should be in (0, 1], got {}",
            lod.crossfade
        );
    }
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cargo test -p vox_render lod_ecs::tests::crossfade_progresses_over_ticks 2>&1 | tail -8`
Expected: FAIL — `lod_crossfade_system` not yet defined

- [ ] **Step 3: Implement lod_crossfade_system**

Add after `lod_select_system`:

```rust
/// Advance all active LOD crossfade transitions by `TimeStep.0` seconds,
/// then write the resulting crossfade weight back to each entity's LodStateComponent.
pub fn lod_crossfade_system(
    dt: Res<TimeStep>,
    mut crossfade: ResMut<LodCrossfadeManager>,
    mut query: Query<(Entity, &mut vox_core::ecs::LodStateComponent)>,
) {
    crossfade.tick(dt.0);

    for (entity, mut lod_state) in query.iter_mut() {
        if let Some(transition) = crossfade.get_transition(entity.index()) {
            lod_state.crossfade = transition.progress;
        } else {
            lod_state.crossfade = 0.0; // stable — no active transition
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_render lod_ecs 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/lod_ecs.rs
git commit -m "feat(lod): lod_crossfade_system — advance transitions + write crossfade to LodStateComponent"
```

---

### Task 5: tile_streaming_system + LodStreamingPlugin

**Files:**
- Modify: `crates/vox_render/src/lod_ecs.rs`

`tile_streaming_system` converts the camera position to a tile coordinate and calls `TileManager::update_camera` to activate nearby tiles and evict distant ones. `LodStreamingPlugin` wires all three systems together.

- [ ] **Step 1: Write the failing tests**

Add to `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn tile_streaming_activates_tiles_near_camera() {
        let mut world = World::new();
        let mut cam = CameraSettings::default();
        cam.position = Vec3::new(0.0, 0.0, 0.0);
        world.insert_resource(cam);
        world.insert_resource(TileManager::new());

        let mut schedule = Schedule::default();
        schedule.add_systems(tile_streaming_system);
        schedule.run(&mut world);

        let tm = world.resource::<TileManager>();
        let active = tm.active_tiles();
        assert!(!active.is_empty(), "At least the camera tile should be active");
        // Camera at (0,0,0) is in tile (0,0). With default active_radius=1, expect 3×3=9 tiles.
        assert_eq!(active.len(), 9, "Expected 3×3 tile grid, got {}", active.len());
    }

    #[test]
    fn plugin_inserts_resources() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(LodStreamingPlugin::default());
        // Resources must be present after plugin build.
        assert!(app.world().contains_resource::<CameraSettings>());
        assert!(app.world().contains_resource::<TimeStep>());
        assert!(app.world().contains_resource::<LodCrossfadeManager>());
        assert!(app.world().contains_resource::<TileManager>());
    }
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cargo test -p vox_render lod_ecs 2>&1 | tail -10`
Expected: FAIL — `tile_streaming_system`, `LodStreamingPlugin` not yet defined

- [ ] **Step 3: Implement tile_streaming_system**

Add after `lod_crossfade_system`:

```rust
/// Convert camera position to a TileCoord and call TileManager::update_camera
/// to activate the surrounding tile grid and evict distant tiles.
pub fn tile_streaming_system(
    camera: Res<CameraSettings>,
    mut tile_manager: ResMut<TileManager>,
) {
    let world_coord = WorldCoord::from_absolute(
        camera.position.x as f64,
        camera.position.y as f64,
        camera.position.z as f64,
    );
    tile_manager.update_camera(world_coord.tile);
}
```

- [ ] **Step 4: Implement LodStreamingPlugin**

Add after `tile_streaming_system`:

```rust
// ── Plugin ─────────────────────────────────────────────────────────────────

/// Bevy plugin that registers LOD streaming resources and systems.
///
/// Usage: `app.add_plugins(LodStreamingPlugin::default())`
///
/// After adding the plugin, update `CameraSettings` and `TimeStep` each frame
/// to drive LOD selection and crossfade timing.
pub struct LodStreamingPlugin {
    /// Crossfade transition duration in seconds (default 0.5 s).
    pub transition_duration: f32,
}

impl Default for LodStreamingPlugin {
    fn default() -> Self { Self { transition_duration: 0.5 } }
}

impl bevy_app::Plugin for LodStreamingPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(CameraSettings::default());
        app.insert_resource(TimeStep::default());
        app.insert_resource(LodCrossfadeManager {
            transitions: vec![],
            transition_duration: self.transition_duration,
        });
        app.insert_resource(TileManager::new());
        app.add_systems(
            bevy_app::Update,
            (lod_select_system, lod_crossfade_system, tile_streaming_system).chain(),
        );
    }
}
```

- [ ] **Step 5: Run all lod_ecs tests**

Run: `cargo test -p vox_render lod_ecs 2>&1 | tail -12`
Expected: all tests pass (7+ tests)

- [ ] **Step 6: Run full vox_render tests to check nothing regressed**

Run: `cargo test -p vox_render 2>&1 | grep -E "FAILED|^test result"`
Expected: `0 failed`

- [ ] **Step 7: Commit**

```bash
git add crates/vox_render/src/lod_ecs.rs
git commit -m "feat(lod): tile_streaming_system + LodStreamingPlugin — complete LOD ECS integration"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `LodStateComponent` (current_level 0–3 + crossfade) → Task 1
- ✅ `CameraSettings` resource → Task 2
- ✅ `TimeStep` resource → Task 2
- ✅ `lod_select_system` using `hierarchical_lod::select_lod_level` → Task 3
- ✅ `lod_crossfade_system` advancing `LodCrossfadeManager` → Task 4
- ✅ `tile_streaming_system` calling `TileManager::update_camera` → Task 5
- ✅ `LodStreamingPlugin` implementing `bevy_app::Plugin` → Task 5
- ✅ Integration test: close entity → LOD 0 → Task 3
- ✅ Integration test: distant entity → LOD 3 → Task 3
- ✅ Integration test: crossfade progresses → Task 4
- ✅ Integration test: tile activation → Task 5
- ✅ Integration test: plugin inserts resources → Task 5

**Placeholder scan:** No TBDs. All function bodies shown in full.

**Type consistency:**
- `LodStateComponent { current_level: u8, crossfade: f32 }` — defined Task 1, queried Tasks 3/4 ✅
- `CameraSettings { position: Vec3, fov_y: f32, screen_height: f32 }` — defined Task 2, used Task 3/5 ✅
- `TimeStep(f32)` — defined Task 2, used Task 4 ✅
- `LodCrossfadeManager { transitions, transition_duration }` — from `lod_crossfade.rs`, inserted Task 5 ✅
- `TileManager::new()` / `::active_tiles()` / `::update_camera()` — from `streaming.rs` ✅
- `hierarchical_lod::select_lod_level(distance: f32, screen_size: f32) -> u32` ✅
- `entity.index()` used as instance_id for LodCrossfadeManager ✅
- `WorldCoord::from_absolute(x: f64, y: f64, z: f64) -> WorldCoord` — from `lwc.rs` ✅
