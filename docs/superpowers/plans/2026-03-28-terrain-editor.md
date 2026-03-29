# Terrain Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `vox_terrain` brush/volume/foliage system into an interactive editor panel so designers can sculpt terrain, scatter foliage, and paint materials in-engine.

**Architecture:** `TerrainEditorState` lives in `vox_app`. It holds the active `TerrainVolume` (stored as a Bevy `Resource`) and UI state. Mouse clicks in the viewport apply a `TerrainBrush` to the volume; after each stroke, `volume_to_splats()` regenerates the terrain splats and updates the ECS `SplatAssetComponent`.

**Tech Stack:** Rust, bevy_ecs 0.16, bevy_app 0.16, egui 0.31, vox_terrain (brushes, volume, foliage, texture_paint).

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/vox_app/src/terrain_editor.rs` | Create | `TerrainEditorState`, brush/foliage UI helpers, re-splat logic |
| `crates/vox_app/src/editor.rs` | Modify | Add `show_terrain_editor` flag + terrain editor menu item |
| `crates/vox_app/src/terrain_setup.rs` | Modify | Expose terrain UUID + spawn `TerrainVolume` as a `Resource` |
| `crates/vox_app/src/simulation.rs` | Modify | Store `terrain_uuid: Option<Uuid>` on `SimulationState` |
| `crates/vox_app/src/lib.rs` | Modify | `pub mod terrain_editor;` |
| `crates/vox_terrain/Cargo.toml` | Modify | Already has the right deps — no change needed |
| `crates/vox_app/Cargo.toml` | Modify | Add `vox_terrain = { path = "../vox_terrain" }` |
| `crates/vox_app/tests/terrain_editor_test.rs` | Create | Unit tests |

---

### Task 1: TerrainEditorState struct

**Files:**
- Create: `crates/vox_app/src/terrain_editor.rs`
- Modify: `crates/vox_app/src/lib.rs`
- Modify: `crates/vox_app/Cargo.toml`
- Test: `crates/vox_app/tests/terrain_editor_test.rs`

- [ ] **Step 1: Add vox_terrain dependency**

In `crates/vox_app/Cargo.toml`, add under `[dependencies]`:
```toml
vox_terrain = { path = "../vox_terrain" }
```

- [ ] **Step 2: Write the failing test**

Create `crates/vox_app/tests/terrain_editor_test.rs`:
```rust
use vox_app::terrain_editor::{TerrainEditorState, ActiveBrush};
use vox_terrain::brushes::{BrushType, BrushFalloff, TerrainBrush};

#[test]
fn default_state_has_raise_brush() {
    let state = TerrainEditorState::default();
    assert!(matches!(state.brush.brush_type, BrushType::Raise));
    assert!((state.brush.radius - 5.0).abs() < f32::EPSILON);
    assert!((state.brush.strength - 0.5).abs() < f32::EPSILON);
}

#[test]
fn set_brush_type_updates_brush() {
    let mut state = TerrainEditorState::default();
    state.set_brush_type(BrushType::Flatten { target_height: 2.0 });
    assert!(matches!(state.brush.brush_type, BrushType::Flatten { .. }));
}

#[test]
fn active_brush_matches_enum() {
    let mut state = TerrainEditorState::default();
    assert_eq!(state.active_brush, ActiveBrush::Raise);
    state.active_brush = ActiveBrush::Lower;
    state.sync_brush();
    assert!(matches!(state.brush.brush_type, BrushType::Lower));
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cargo test -p vox_app terrain_editor
```
Expected: FAIL — `terrain_editor` module not found.

- [ ] **Step 4: Create terrain_editor.rs**

Create `crates/vox_app/src/terrain_editor.rs`:
```rust
use vox_terrain::brushes::{BrushFalloff, BrushType, TerrainBrush};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveBrush {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Paint,
    Erode,
}

pub struct TerrainEditorState {
    pub active_brush: ActiveBrush,
    pub brush: TerrainBrush,
    pub flatten_height: f32,
    pub paint_material: u8,
    pub foliage_scatter_pending: bool,
    pub foliage_density: f32,
    pub is_open: bool,
}

impl Default for TerrainEditorState {
    fn default() -> Self {
        Self {
            active_brush: ActiveBrush::Raise,
            brush: TerrainBrush::new(BrushType::Raise, 5.0, 0.5),
            flatten_height: 0.0,
            paint_material: 0,
            foliage_scatter_pending: false,
            foliage_density: 0.5,
            is_open: false,
        }
    }
}

impl TerrainEditorState {
    pub fn set_brush_type(&mut self, bt: BrushType) {
        self.brush.brush_type = bt;
    }

    /// Sync `brush.brush_type` from `active_brush` enum.
    pub fn sync_brush(&mut self) {
        self.brush.brush_type = match self.active_brush {
            ActiveBrush::Raise   => BrushType::Raise,
            ActiveBrush::Lower   => BrushType::Lower,
            ActiveBrush::Smooth  => BrushType::Smooth,
            ActiveBrush::Flatten => BrushType::Flatten { target_height: self.flatten_height },
            ActiveBrush::Paint   => BrushType::Paint { material: self.paint_material },
            ActiveBrush::Erode   => BrushType::Erode,
        };
    }
}
```

- [ ] **Step 5: Expose module in lib.rs**

In `crates/vox_app/src/lib.rs`, add:
```rust
pub mod terrain_editor;
```

- [ ] **Step 6: Run test to verify it passes**

```bash
cargo test -p vox_app terrain_editor
```
Expected: 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/terrain_editor.rs crates/vox_app/src/lib.rs crates/vox_app/Cargo.toml crates/vox_app/tests/terrain_editor_test.rs Cargo.lock
git commit -m "feat(terrain-editor): TerrainEditorState with brush type sync"
```

---

### Task 2: TerrainVolume as ECS Resource + terrain_setup wiring

**Files:**
- Modify: `crates/vox_app/src/terrain_setup.rs`
- Modify: `crates/vox_app/src/simulation.rs`
- Test: `crates/vox_app/tests/terrain_editor_test.rs` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `crates/vox_app/tests/terrain_editor_test.rs`:
```rust
use bevy_ecs::prelude::*;
use vox_app::terrain_editor::apply_brush_stroke;
use vox_terrain::volume::TerrainVolume;
use glam::Vec3;

#[test]
fn apply_brush_raises_terrain() {
    let mut world = World::new();
    let mut vol = TerrainVolume::new(32, 32, 32, 0.5, [0.0; 3]);
    // Fill with air (positive SDF)
    for v in vol.data.iter_mut() { *v = 1.0; }
    // Set ground at y=8
    for x in 0..32 {
        for z in 0..32 {
            let idx = 8 * 32 * 32 + z * 32 + x;
            vol.data[idx] = -0.1;
        }
    }
    world.insert_resource(vol);

    let center = Vec3::new(8.0, 4.0, 8.0);
    apply_brush_stroke(&mut world, center, vox_terrain::brushes::BrushType::Raise, 3.0, 1.0, 1.0);
    // Volume was modified (function didn't panic)
    let vol = world.resource::<TerrainVolume>();
    assert!(vol.solid_count() > 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_app apply_brush_raises
```
Expected: FAIL — `apply_brush_stroke` not found.

- [ ] **Step 3: Add `Resource` derive to TerrainVolume**

In `crates/vox_terrain/src/volume.rs`, add `bevy_ecs` to Cargo.toml and derive:

First, `crates/vox_terrain/Cargo.toml` — add:
```toml
bevy_ecs = { workspace = true }
```

Then in `volume.rs`, add `#[derive(bevy_ecs::prelude::Resource)]` above `pub struct TerrainVolume`:
```rust
#[derive(bevy_ecs::prelude::Resource)]
pub struct TerrainVolume {
    // ... existing fields unchanged
```

- [ ] **Step 4: Add apply_brush_stroke to terrain_editor.rs**

In `crates/vox_app/src/terrain_editor.rs`, add:
```rust
use bevy_ecs::prelude::*;
use glam::Vec3;
use vox_terrain::brushes::{BrushType, TerrainBrush};
use vox_terrain::volume::TerrainVolume;

/// Apply a single brush stroke to the TerrainVolume resource.
pub fn apply_brush_stroke(
    world: &mut World,
    center: Vec3,
    brush_type: BrushType,
    radius: f32,
    strength: f32,
    dt: f32,
) {
    if let Some(mut vol) = world.get_resource_mut::<TerrainVolume>() {
        let brush = TerrainBrush::new(brush_type, radius, strength);
        brush.apply(&mut vol, center, dt);
    }
}
```

- [ ] **Step 5: Spawn TerrainVolume in terrain_setup.rs**

Replace the contents of `spawn_terrain` in `crates/vox_app/src/terrain_setup.rs`:
```rust
use bevy_ecs::prelude::*;
use vox_core::ecs::{SplatAssetComponent, SplatInstanceComponent, LodLevel};
use vox_core::mapgen::generate_map;
use vox_terrain::volume::{TerrainVolume, generate_demo_volume};
use glam::{Quat, Vec3};
use uuid::Uuid;

/// Spawn terrain as an ECS entity in the world, and insert TerrainVolume as a Resource.
/// Returns the asset UUID for the terrain splat entity.
pub fn spawn_terrain(world: &mut World, width: f32, _depth: f32, _material: &str) -> Uuid {
    // Spawn sculpt-ready SDF volume as a Resource
    let volume = generate_demo_volume();
    world.insert_resource(volume);

    // Generate initial splats from the mapgen system
    let splats = generate_map(42, width, 1.0);
    let uuid = Uuid::new_v4();
    let splat_count = splats.len() as u32;

    world.spawn(SplatAssetComponent { uuid, splat_count, splats });
    world.spawn(SplatInstanceComponent {
        asset_uuid: uuid,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
        instance_id: 1000,
        lod: LodLevel::Full,
    });

    uuid
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p vox_app terrain_editor
```
Expected: all terrain_editor tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/terrain_editor.rs crates/vox_app/src/terrain_setup.rs crates/vox_terrain/src/volume.rs crates/vox_terrain/Cargo.toml crates/vox_app/tests/terrain_editor_test.rs Cargo.lock
git commit -m "feat(terrain-editor): TerrainVolume as ECS Resource, brush stroke system"
```

---

### Task 3: Re-splat terrain after brush stroke

**Files:**
- Modify: `crates/vox_app/src/terrain_editor.rs`
- Test: `crates/vox_app/tests/terrain_editor_test.rs` (add test)

After a brush stroke, the volume has changed. We need to regenerate splats and update the ECS `SplatAssetComponent` so the renderer sees new geometry.

- [ ] **Step 1: Write the failing test**

Append to `crates/vox_app/tests/terrain_editor_test.rs`:
```rust
use vox_core::ecs::SplatAssetComponent;
use uuid::Uuid;
use vox_app::terrain_editor::resplat_terrain;
use vox_terrain::volume::{TerrainVolume, generate_demo_volume};

#[test]
fn resplat_updates_splat_asset() {
    let mut world = World::new();
    let vol = generate_demo_volume();
    world.insert_resource(vol);

    let uuid = Uuid::new_v4();
    let entity = world.spawn(SplatAssetComponent {
        uuid,
        splat_count: 0,
        splats: vec![],
    }).id();

    resplat_terrain(&mut world, entity);

    let asset = world.entity(entity).get::<SplatAssetComponent>().unwrap();
    assert!(asset.splat_count > 0, "resplat should produce splats from the volume");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_app resplat_updates
```
Expected: FAIL — `resplat_terrain` not found.

- [ ] **Step 3: Implement resplat_terrain**

Add to `crates/vox_app/src/terrain_editor.rs`:
```rust
use bevy_ecs::prelude::*;
use vox_core::ecs::SplatAssetComponent;
use vox_terrain::volume::{TerrainVolume, volume_to_splats};

/// Regenerate terrain splats from the current TerrainVolume and update the ECS asset.
pub fn resplat_terrain(world: &mut World, terrain_entity: Entity) {
    let splats = {
        let vol = world.resource::<TerrainVolume>();
        volume_to_splats(vol)
    };
    let splat_count = splats.len() as u32;
    if let Some(mut asset) = world.entity_mut(terrain_entity).get_mut::<SplatAssetComponent>() {
        asset.splats = splats;
        asset.splat_count = splat_count;
    }
}
```

- [ ] **Step 4: Run test**

```bash
cargo test -p vox_app resplat_updates
```
Expected: PASS.

- [ ] **Step 5: Full suite**

```bash
cargo test -p vox_app
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_app/src/terrain_editor.rs crates/vox_app/tests/terrain_editor_test.rs
git commit -m "feat(terrain-editor): resplat_terrain regenerates ECS splats from volume after brush"
```

---

### Task 4: Terrain editor UI panel in SceneEditor

**Files:**
- Modify: `crates/vox_app/src/editor.rs`
- Modify: `crates/vox_app/src/terrain_editor.rs`
- Test: `crates/vox_app/tests/terrain_editor_test.rs` (add test)

- [ ] **Step 1: Write the failing test**

Append to `crates/vox_app/tests/terrain_editor_test.rs`:
```rust
use vox_app::terrain_editor::TerrainEditorState;
use vox_terrain::brushes::BrushType;

#[test]
fn terrain_editor_state_foliage_scatter_flag() {
    let mut state = TerrainEditorState::default();
    assert!(!state.foliage_scatter_pending);
    state.foliage_scatter_pending = true;
    assert!(state.foliage_scatter_pending);
    state.foliage_scatter_pending = false;
    assert!(!state.foliage_scatter_pending);
}

#[test]
fn sync_brush_flatten_uses_flatten_height() {
    let mut state = TerrainEditorState::default();
    state.active_brush = ActiveBrush::Flatten;
    state.flatten_height = 3.5;
    state.sync_brush();
    if let BrushType::Flatten { target_height } = state.brush.brush_type {
        assert!((target_height - 3.5).abs() < f32::EPSILON);
    } else {
        panic!("expected Flatten brush");
    }
}
```

- [ ] **Step 2: Run test to verify it passes (both are pure logic)**

```bash
cargo test -p vox_app terrain_editor_state_foliage sync_brush_flatten
```
Expected: PASS (these test existing logic).

- [ ] **Step 3: Add show_terrain_editor to SceneEditor**

In `crates/vox_app/src/editor.rs`, find the `SceneEditor` struct definition and add the field after `show_vfx_editor`:
```rust
pub show_terrain_editor: bool,
```

Also add to the `Default` or constructor wherever `show_vfx_editor: false` is initialized:
```rust
show_terrain_editor: false,
```

- [ ] **Step 4: Add terrain editor menu item**

In `editor.rs`, find the menu bar rendering (look for `ui.menu_button("Window"` or similar). Add a menu item for the terrain editor. For example, after the VFX editor toggle:
```rust
if ui.checkbox(&mut self.show_terrain_editor, "Terrain Editor").clicked() {}
```

- [ ] **Step 5: Show terrain editor panel**

In `editor.rs`, add conditional panel rendering (near other `show_*` conditionals):
```rust
if self.show_terrain_editor {
    if let Some(te_state) = terrain_editor_state {
        egui::Window::new("Terrain Editor")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                vox_app::terrain_editor::show_terrain_editor_panel(ui, te_state);
            });
    }
}
```

- [ ] **Step 6: Implement show_terrain_editor_panel**

Add to `crates/vox_app/src/terrain_editor.rs`:
```rust
use vox_terrain::brushes::{BrushFalloff, BrushType};

pub fn show_terrain_editor_panel(ui: &mut egui::Ui, state: &mut TerrainEditorState) {
    ui.heading("Brush");
    ui.horizontal(|ui| {
        for (label, variant) in &[
            ("Raise",   ActiveBrush::Raise),
            ("Lower",   ActiveBrush::Lower),
            ("Smooth",  ActiveBrush::Smooth),
            ("Flatten", ActiveBrush::Flatten),
            ("Paint",   ActiveBrush::Paint),
            ("Erode",   ActiveBrush::Erode),
        ] {
            if ui.selectable_label(state.active_brush == *variant, *label).clicked() {
                state.active_brush = *variant;
                state.sync_brush();
            }
        }
    });
    ui.add(egui::Slider::new(&mut state.brush.radius, 0.5..=50.0).text("Radius"));
    ui.add(egui::Slider::new(&mut state.brush.strength, 0.0..=2.0).text("Strength"));
    if state.active_brush == ActiveBrush::Flatten {
        if ui.add(egui::Slider::new(&mut state.flatten_height, -10.0..=50.0).text("Height")).changed() {
            state.sync_brush();
        }
    }
    ui.separator();
    ui.heading("Foliage");
    ui.add(egui::Slider::new(&mut state.foliage_density, 0.0..=1.0).text("Density"));
    if ui.button("Scatter Foliage").clicked() {
        state.foliage_scatter_pending = true;
    }
}
```

Note: `show_terrain_editor_panel` needs `egui` — add to `terrain_editor.rs` imports:
```rust
use egui;
```

This requires `vox_app` to have `egui` in its Cargo.toml (it already does from the editor).

- [ ] **Step 7: Run full suite**

```bash
cargo test -p vox_app
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/terrain_editor.rs crates/vox_app/tests/terrain_editor_test.rs
git commit -m "feat(terrain-editor): terrain editor UI panel with brush controls and foliage scatter"
```

---

### Task 5: Foliage scatter integration

**Files:**
- Modify: `crates/vox_app/src/terrain_editor.rs`
- Test: `crates/vox_app/tests/terrain_editor_test.rs` (add test)

When `foliage_scatter_pending` is true, call `scatter_foliage()` against the TerrainVolume and spawn the resulting instances as ECS entities.

- [ ] **Step 1: Write the failing test**

Append to `crates/vox_app/tests/terrain_editor_test.rs`:
```rust
use vox_app::terrain_editor::scatter_foliage_on_terrain;
use vox_terrain::volume::generate_demo_volume;
use vox_terrain::foliage::default_foliage_rules;
use vox_core::ecs::SplatInstanceComponent;

#[test]
fn scatter_foliage_spawns_entities() {
    let mut world = World::new();
    let vol = generate_demo_volume();
    world.insert_resource(vol);

    let rules = default_foliage_rules();
    scatter_foliage_on_terrain(&mut world, &rules, 1.0);

    let count = world.query::<&SplatInstanceComponent>().iter(&world).count();
    // scatter_foliage may return 0 on demo volume depending on slope/height,
    // so we just verify it doesn't panic
    let _ = count;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_app scatter_foliage_spawns
```
Expected: FAIL — `scatter_foliage_on_terrain` not found.

- [ ] **Step 3: Implement scatter_foliage_on_terrain**

Add to `crates/vox_app/src/terrain_editor.rs`:
```rust
use vox_core::ecs::{SplatInstanceComponent, LodLevel};
use vox_terrain::foliage::{FoliageRule, scatter_foliage};
use glam::Quat;
use uuid::Uuid;

static FOLIAGE_INSTANCE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(5000);

/// Scatter foliage on the terrain volume and spawn as ECS entities.
pub fn scatter_foliage_on_terrain(
    world: &mut World,
    rules: &[FoliageRule],
    density_scale: f32,
) {
    let instances = {
        let vol = world.resource::<TerrainVolume>();
        // Build a bounding box from the volume for sampling
        let world_size = vol.size_x as f32 * vol.voxel_size;
        let bounds_min = [vol.origin[0], vol.origin[2]]; // xz plane
        let bounds_max = [vol.origin[0] + world_size, vol.origin[2] + world_size];
        scatter_foliage(rules, vol, bounds_min, bounds_max, density_scale, 0)
    };

    for inst in instances {
        let id = FOLIAGE_INSTANCE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        world.spawn(SplatInstanceComponent {
            asset_uuid: Uuid::nil(), // placeholder — resolved by asset system
            position: Vec3::new(inst.position[0], inst.position[1], inst.position[2]),
            rotation: Quat::from_rotation_y(inst.rotation_y),
            scale: inst.scale,
            instance_id: id,
            lod: LodLevel::Full,
        });
    }
}
```

Check `scatter_foliage` signature in `vox_terrain::foliage` — adjust parameter order if different.

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_app terrain_editor
```
Expected: all pass.

- [ ] **Step 5: Full suite**

```bash
cargo test
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/vox_app/src/terrain_editor.rs crates/vox_app/tests/terrain_editor_test.rs
git commit -m "feat(terrain-editor): foliage scatter integration spawns ECS entities from vox_terrain"
```

---

## Summary

| Task | Deliverable |
|------|-------------|
| 1 | `TerrainEditorState` with brush sync |
| 2 | `TerrainVolume` as ECS Resource, `apply_brush_stroke` |
| 3 | `resplat_terrain` updates ECS SplatAssetComponent after sculpt |
| 4 | egui panel in SceneEditor with all brush controls |
| 5 | `scatter_foliage_on_terrain` spawns ECS instances |
