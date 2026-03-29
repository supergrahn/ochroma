# Shadow Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing CPU `ShadowMapper` into the render loop so splats in shadow appear visibly darker.

**Architecture:** `ShadowMapper` is added to `EngineApp`. Each frame it updates light VP matrices using the sun direction from `LightManager`, rasterises splat positions into CPU depth buffers, and the software rasteriser path checks `is_in_shadow()` per splat, applying a 0.3x shadow factor to opacity.

**Tech Stack:** `vox_render::shadows::ShadowMapper` (existing), `vox_render::lighting::LightManager` (existing), `SoftwareRasteriser` (existing in `vox_render::gpu::software_rasteriser`)

---

## Task 1: Add `ShadowMapper` to `EngineApp` and initialize it

**File:** `crates/vox_app/src/bin/engine_runner.rs`

### 1a. Add import

In the `use vox_render::...` import block (around line 34), add:

```rust
use vox_render::shadows::ShadowMapper;
```

### 1b. Add field to `EngineApp`

In the `EngineApp` struct (around line 71), add the field after `light_manager`:

```rust
    // Shadow mapping (CPU cascaded shadow maps, wired to sun direction)
    shadow_mapper: ShadowMapper,
```

### 1c. Initialize in `EngineApp::new` (or wherever the struct literal is built)

Find the `EngineApp { ... }` struct literal that initializes all fields. After the `light_manager` field initialization, add:

```rust
            shadow_mapper: ShadowMapper::new(512),
```

This creates 3 cascades (0-20m, 20-100m, 100-500m) each with a 512x512 depth buffer.

### 1d. Verify build

```bash
cd /home/tomespen/git/ochroma && cargo build -p vox_app 2>&1 | head -40
```

### 1e. Commit

```bash
cd /home/tomespen/git/ochroma && git add crates/vox_app/src/bin/engine_runner.rs && git commit -m "feat(shadows): add ShadowMapper field to EngineApp, initialize with 3×512 cascades"
```

---

## Task 2: Update and render shadow maps each frame

The shadow map must be built before the software rasteriser runs. The right place is immediately after the `render_splats` list is finalised (after particle splats are appended, around line 674) and before `self.rasteriser.render(...)` is called (line 681).

**File:** `crates/vox_app/src/bin/engine_runner.rs`

### 2a. Insert shadow map update + render pass

Find the block that ends with:
```rust
        // Add particle splats
        let particle_splats = self.particles.to_splats();
        render_splats.extend(&particle_splats);

        // Time-of-day illuminant
        let illuminant = illuminant_for_time(self.engine.time_of_day());

        // 1. Software rasterise at internal resolution
        let render_start = Instant::now();
        let fb = self.rasteriser.render(&render_splats, &render_camera, &illuminant);
```

Replace it with:

```rust
        // Add particle splats
        let particle_splats = self.particles.to_splats();
        render_splats.extend(&particle_splats);

        // --- Shadow map update ---
        // Derive sun direction from LightManager using current time of day.
        // Day 172 ≈ summer solstice; kept constant for now.
        let shadow_hour = self.engine.time_of_day();
        let sun_dir = self.light_manager.sun.sun_direction(shadow_hour, 172);

        // Camera forward: from position toward target (normalised).
        let cam_fwd = (self.camera.target - self.camera.position).normalize_or(glam::Vec3::NEG_Z);

        // Recompute light view-projection matrices for each cascade.
        self.shadow_mapper.update(self.camera.position, cam_fwd, sun_dir);

        // Rasterise splat occluders into the depth buffers.
        // Radius = average of per-axis scales (same approximation used by the rasteriser).
        let shadow_positions: Vec<glam::Vec3> = render_splats
            .iter()
            .map(|s| glam::Vec3::from(s.position))
            .collect();
        let shadow_radii: Vec<f32> = render_splats
            .iter()
            .map(|s| (s.scale[0].abs() + s.scale[1].abs() + s.scale[2].abs()) / 3.0)
            .collect();
        self.shadow_mapper.render_shadow_map(&shadow_positions, &shadow_radii);

        // Time-of-day illuminant
        let illuminant = illuminant_for_time(self.engine.time_of_day());

        // 1. Software rasterise at internal resolution
        let render_start = Instant::now();
        let fb = self.rasteriser.render(&render_splats, &render_camera, &illuminant);
```

### 2b. Verify build

```bash
cd /home/tomespen/git/ochroma && cargo build -p vox_app 2>&1 | head -40
```

### 2c. Commit

```bash
cd /home/tomespen/git/ochroma && git add crates/vox_app/src/bin/engine_runner.rs && git commit -m "feat(shadows): update + render shadow map each frame before software rasteriser"
```

---

## Task 3: Apply shadow darkening in the software rasteriser

The `SoftwareRasteriser::render` method must accept a reference to `ShadowMapper` and test each splat's world position before it is projected. Shadow factor: multiply `opacity` by `0.3` (implemented as integer multiply by `77u8 / 255` or via `f32` path already used in the method).

**File:** `crates/vox_render/src/gpu/software_rasteriser.rs`

### 3a. Add import for `ShadowMapper`

At the top of the file, after the existing `use` statements, add:

```rust
use crate::shadows::ShadowMapper;
```

### 3b. Change `render` signature to accept an optional shadow mapper

Change the `render` function signature from:

```rust
    pub fn render(
        &mut self,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        illuminant: &Illuminant,
    ) -> Framebuffer {
```

to:

```rust
    pub fn render(
        &mut self,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        illuminant: &Illuminant,
        shadow_mapper: Option<&ShadowMapper>,
    ) -> Framebuffer {
```

### 3c. Apply shadow factor during projection

Inside `render`, in the `filter_map` closure that builds `projected`, find the line that reads the opacity:

```rust
                let opacity = splat.opacity as f32 / 255.0;
```

Replace it with:

```rust
                // Apply shadow darkening (factor 0.3) if the splat is in shadow.
                let base_opacity = splat.opacity as f32 / 255.0;
                let shadow_factor = if let Some(sm) = shadow_mapper {
                    let world_pos = glam::Vec3::new(
                        splat.position[0],
                        splat.position[1],
                        splat.position[2],
                    );
                    if sm.is_in_shadow(world_pos, 0.005) {
                        0.3
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                let opacity = base_opacity * shadow_factor;
```

### 3d. Update all call sites for `rasteriser.render`

There are two call sites in `engine_runner.rs` (lines ~681 and ~1451). Both must be updated to pass `Some(&self.shadow_mapper)`.

**Call site 1** (main render path, ~line 681):

```rust
        let fb = self.rasteriser.render(&render_splats, &render_camera, &illuminant, Some(&self.shadow_mapper));
```

**Call site 2** (the secondary render path, ~line 1451):

```rust
            let fb = self.rasteriser.render(&render_splats, &render_camera, &illuminant, Some(&self.shadow_mapper));
```

If there are any test call sites in `vox_render` tests, pass `None` to maintain backward compat.

### 3e. Verify build and tests

```bash
cd /home/tomespen/git/ochroma && cargo build -p vox_render -p vox_app 2>&1 | head -60
cd /home/tomespen/git/ochroma && cargo test -p vox_render 2>&1 | tail -20
```

### 3f. Commit

```bash
cd /home/tomespen/git/ochroma && git add crates/vox_render/src/gpu/software_rasteriser.rs crates/vox_app/src/bin/engine_runner.rs && git commit -m "feat(shadows): apply 0.3x shadow factor to opacity in SoftwareRasteriser per-splat shadow test"
```

---

## Final verification

```bash
cd /home/tomespen/git/ochroma && cargo test 2>&1 | tail -30
```

Expected: all tests pass. At runtime, splats occluded from the sun will render at 30% opacity, creating visible shadows in the scene.

---

## Key API reference (confirmed from source)

| Symbol | Location | Signature |
|--------|----------|-----------|
| `ShadowMapper::new` | `crates/vox_render/src/shadows.rs:54` | `fn new(resolution: usize) -> Self` |
| `ShadowMapper::update` | `shadows.rs:74` | `fn update(&mut self, camera_pos: Vec3, camera_fwd: Vec3, sun_dir: Vec3)` |
| `ShadowMapper::render_shadow_map` | `shadows.rs:112` | `fn render_shadow_map(&mut self, splat_positions: &[Vec3], splat_radii: &[f32])` |
| `ShadowMapper::is_in_shadow` | `shadows.rs:175` | `fn is_in_shadow(&self, world_pos: Vec3, bias: f32) -> bool` |
| `SunModel::sun_direction` | `crates/vox_render/src/lighting.rs:14` | `fn sun_direction(&self, hour: f32, day_of_year: u32) -> Vec3` |
| `SoftwareRasteriser::render` | `crates/vox_render/src/gpu/software_rasteriser.rs:82` | `fn render(&mut self, splats: &[GaussianSplat], camera: &RenderCamera, illuminant: &Illuminant) -> Framebuffer` (to be extended) |
