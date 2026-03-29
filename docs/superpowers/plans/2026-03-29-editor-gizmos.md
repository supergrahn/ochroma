# Editor Gizmos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire GizmoRenderer into the render loop so translate/rotate/scale handles appear on the selected entity in the editor viewport, and mouse drag moves/transforms the entity along the active axis.

**Architecture:** After the software rasteriser fills the pixel buffer each frame, call `GizmoRenderer::draw_overlay()` with the selected entity position and view-projection matrix to overlay axis arrows. Mouse events call `GizmoRenderer::hit_test()` / `begin_drag()` / `update_drag()` / `end_drag()` to move entities.

**Tech Stack:** `GizmoRenderer` (`crates/vox_render/src/gizmos.rs`), `SoftwareRasteriser` pixel buffer (`crates/vox_render/src/gpu/software_rasteriser.rs`), winit mouse events (`crates/vox_app/src/bin/engine_runner.rs`)

---

## Current State Audit

Before implementing, note what is **already present** so tasks build on — not duplicate — existing work:

### Already implemented in `crates/vox_render/src/gizmos.rs`

| Item | Status |
|------|--------|
| `GizmoMode` enum (`Translate`, `Rotate`, `Scale`) | Done |
| `Axis` enum (`X`, `Y`, `Z`) | Done |
| `GizmoRenderer` struct with `mode`, `active_axis`, `dragging`, `drag_start_screen`, `drag_start_world` | Done |
| `project_to_screen(pos, view_proj, width, height) -> Option<(f32,f32)>` | Done (returns `Option`, handles behind-camera) |
| `world_arrow_length(entity_pos, view_proj, width, height) -> f32` | Done (scales arrow to ~`ARROW_PIXELS` regardless of depth) |
| `draw_line(pixels: &mut [[u8;4]], width, height, x0,y0,x1,y1, color)` | Done (Bresenham, pixel slice) |
| `draw_arrowhead(...)` | Done (triangle cap for translate mode) |
| `draw_cube_endpoint(...)` | Done (square cap for scale mode) |
| `draw_circle(...)` | Done (64-segment circle for rotate mode) |
| `point_to_segment_dist(px,py, ax,ay, bx,by) -> f32` | Done |
| `GizmoRenderer::draw_overlay(&self, pixels: &mut [[u8;4]], width, height, entity_world_pos, view_proj)` | Done (all three modes, arrowheads, active axis highlight) |
| `GizmoRenderer::hit_test(mouse_x,y, entity_world_pos, view_proj, sw, sh) -> Option<Axis>` | Done |
| `GizmoRenderer::begin_drag(&mut self, axis, mouse_x, mouse_y)` | Done |
| `GizmoRenderer::update_drag(&mut self, mouse_x,y, entity_pos, view_proj, sw,sh) -> Vec3` | Done (projects pixel delta onto world axis) |
| `GizmoRenderer::end_drag(&mut self)` | Done |
| `ARROW_PIXELS = 80.0`, `HIT_TOLERANCE = 8.0` | Done |
| Colors: `RED`, `GREEN`, `BLUE`, `YELLOW` | Done |

### Already wired in `crates/vox_app/src/bin/engine_runner.rs`

| Item | Status |
|------|--------|
| `EngineApp.gizmo: GizmoRenderer` field | Done |
| `EngineApp.left_mouse_held: bool` field | Done |
| Render loop calls `gizmo.draw_overlay()` after HUD, when `editor_visible && selected_entity` | Done (lines 770–791) |
| `handle_mouse_button` left-press: calls `gizmo.hit_test()` then `begin_drag()` | Done (lines 1084–1121) |
| `handle_mouse_button` left-release: calls `gizmo.end_drag()` | Done (lines 1077–1083) |
| `handle_mouse_move`: calls `gizmo.update_drag()` and `editor.move_selected(delta)` when dragging | Done (lines 1136–1163) |
| `draw_overlay` takes `&mut [[u8;4]]` (pixel slice), matches `final_pixels` type | Done |

### Pixel buffer type contract

`SoftwareRasteriser::render()` returns `Vec<[u8; 4]>`. The gizmo is drawn into `final_pixels: Vec<[u8; 4]>` (post-upscale display buffer) via `gizmo.draw_overlay(&mut final_pixels, display_w, display_h, entity.position, vp)`. The view-projection matrix used is re-derived from `display_w/display_h` aspect ratio, which is correct since `final_pixels` is at display resolution.

---

## What Remains To Do

The core gizmo system is complete and wired. The remaining work is:

1. **Tests** — the existing test suite covers only `project_to_screen_basic` and `draw_line_non_empty`. Three additional tests are required.
2. **Snap-to-grid** — `SceneEditor.snap_enabled` and `snap_grid` exist but `update_drag` does not apply snapping when the editor has snap enabled.
3. **Gizmo mode sync** — `SceneEditor.gizmo_mode` and `GizmoRenderer.mode` are separate; a change in the editor UI does not update the renderer.
4. **Thick line rendering** — arrows are single-pixel-wide Bresenham lines. A 2px-wide variant would improve usability.
5. **`draw` method alias** — the prompt specification requests a `draw(framebuffer: &mut [u8], ...)` signature taking a flat `&mut [u8]` (RGBA bytes, not `[[u8;4]]`). This alias is missing; the existing method uses `[[u8;4]]` slices. Adding the flat-byte variant broadens future API surface.

---

## Task 1: Add `draw` flat-byte alias and `draw_line_flat` helper

**File:** `crates/vox_render/src/gizmos.rs`

**Why:** The existing `draw_overlay` takes `&mut [[u8;4]]`. Some callers (e.g. wgpu texture upload paths) pass a flat `&mut [u8]` RGBA buffer. Provide a thin wrapper so both call sites work.

- [ ] Add a `draw_line_flat` free function that accepts `pixels: &mut [u8]` (flat RGBA) and converts pixel addressing: `idx = (y * width + x) * 4`.
- [ ] Add a `pub fn draw(...)` method on `GizmoRenderer` with signature:

```rust
pub fn draw(
    &self,
    framebuffer: &mut [u8],    // flat RGBA pixel buffer, width*height*4 bytes
    width: u32,
    height: u32,
    view_proj: glam::Mat4,
    origin: glam::Vec3,
) {
    // Reinterpret as pixel slice and delegate to draw_overlay
    assert_eq!(framebuffer.len(), (width * height * 4) as usize);
    // SAFETY: [u8;4] has same size and align as 4×u8
    let pixels: &mut [[u8; 4]] = unsafe {
        std::slice::from_raw_parts_mut(
            framebuffer.as_mut_ptr() as *mut [u8; 4],
            (width * height) as usize,
        )
    };
    self.draw_overlay(pixels, width, height, origin, view_proj);
}
```

Note: the argument order differs from `draw_overlay` (`origin` before `view_proj` in `draw`, but `entity_world_pos` before `view_proj` in `draw_overlay`). Match `draw_overlay`'s order internally.

- [ ] Alternatively, implement `draw` without unsafe by iterating over chunks of 4 bytes — avoids `unsafe` at the cost of a small copy. The safe version:

```rust
pub fn draw(
    &self,
    framebuffer: &mut [u8],
    width: u32,
    height: u32,
    view_proj: glam::Mat4,
    origin: glam::Vec3,
) {
    let n = (width * height) as usize;
    let mut pixels: Vec<[u8; 4]> = (0..n)
        .map(|i| [framebuffer[i*4], framebuffer[i*4+1], framebuffer[i*4+2], framebuffer[i*4+3]])
        .collect();
    self.draw_overlay(&mut pixels, width, height, origin, view_proj);
    for (i, p) in pixels.iter().enumerate() {
        framebuffer[i*4..i*4+4].copy_from_slice(p);
    }
}
```

**Prefer the unsafe reinterpret cast** (it is sound: `[u8;4]` and four consecutive `u8` bytes have identical layout). Add a comment explaining soundness.

---

## Task 2: Add three required unit tests

**File:** `crates/vox_render/src/gizmos.rs` — add inside the existing `#[cfg(test)] mod tests` block.

### Test 1: `project_to_screen_identity`

```rust
#[test]
fn project_to_screen_identity() {
    // With identity view_proj, NDC == world coords (before perspective divide).
    // A point at (0, 0, -1) in clip space with w=1 gives NDC (0, 0, -1),
    // which maps to screen center (400, 300) for an 800x600 window.
    // Use a simple orthographic-like matrix: identity (w stays 1).
    let vp = Mat4::IDENTITY;
    // With identity: clip = (x, y, z, 1). NDC_x = x/1 = x, NDC_y = y/1 = y.
    // sx = (0*0.5 + 0.5)*800 = 400, sy = (1 - (0*0.5+0.5))*600 = 300.
    let result = project_to_screen(Vec3::new(0.0, 0.0, -1.0), vp, 800, 600);
    assert!(result.is_some(), "identity projection should not be behind camera");
    let (sx, sy) = result.unwrap();
    assert!((sx - 400.0).abs() < 1.0, "expected sx≈400, got {sx}");
    assert!((sy - 300.0).abs() < 1.0, "expected sy≈300, got {sy}");
}
```

**Rationale:** With `Mat4::IDENTITY` as view_proj, any point with `w > 0` projects. The point `(0,0,-1)` has clip coords `(0,0,-1,1)`, giving NDC `(0,0,-1)`, mapping to screen `(400, 300)` for 800×600. The Z component does not affect X/Y screen position.

### Test 2: `draw_does_not_panic_on_empty_buffer`

```rust
#[test]
fn draw_does_not_panic_on_empty_buffer() {
    let gizmo = GizmoRenderer::new();
    // 4×4 RGBA buffer (64 bytes)
    let mut framebuffer = vec![0u8; 4 * 4 * 4];
    // Should not panic even though the projected gizmo likely falls outside bounds
    gizmo.draw(&mut framebuffer, 4, 4, Mat4::IDENTITY, Vec3::ZERO);
}
```

**Rationale:** Validates the `draw` method does not index out of bounds on a tiny buffer. `draw_line` already clamps to `width`/`height`, but this test documents the contract.

### Test 3: `handle_mouse_press_returns_none_when_no_entity`

```rust
#[test]
fn handle_mouse_press_returns_none_when_no_entity() {
    // A default GizmoRenderer with no active drag, testing hit_test.
    // With identity view_proj and Vec3::ZERO as entity origin, the entity
    // projects to (400, 300) on 800×600. Clicking at (400, 300) is at the
    // origin, not on any axis arm — all arms start at that point and extend
    // outward, so the origin itself is within HIT_TOLERANCE of every axis
    // segment base. To guarantee None, click far away from the gizmo.
    let gizmo = GizmoRenderer::new();
    let vp = {
        // Use a real perspective camera so the entity is visible
        let view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 1000.0,
        );
        proj * view
    };
    // Click at corner of screen — far from any gizmo arm
    let result = gizmo.hit_test(0.0, 0.0, Vec3::ZERO, vp, 800, 450);
    assert!(result.is_none(), "corner click should not hit any axis: {result:?}");
}
```

**Rationale:** Ensures `hit_test` returns `None` when the click is far from all three axis arms. Clicking at `(0, 0)` (top-left corner) will not be within `HIT_TOLERANCE = 8px` of any axis line if the entity projects near the screen center.

---

## Task 3: Sync `SceneEditor.gizmo_mode` to `GizmoRenderer.mode`

**File:** `crates/vox_app/src/bin/engine_runner.rs`

**Problem:** `SceneEditor.gizmo_mode` (type `vox_app::editor::GizmoMode`) is set by the egui toolbar (W/E/R keys or UI buttons) but never propagated to `EngineApp.gizmo.mode` (type `vox_render::gizmos::GizmoMode`). As a result, the rendered gizmo always shows Translate handles regardless of the editor mode.

**Note on duplicate enum:** `vox_app::editor::GizmoMode` and `vox_render::gizmos::GizmoMode` are separate types with identical variants. To avoid a dependency inversion, add a `From` conversion or match manually.

- [ ] At the start of `render_frame()`, before the gizmo `draw_overlay` call (or immediately before it in the `if self.editor_visible` block), add:

```rust
// Sync gizmo mode from editor to renderer
self.gizmo.mode = match self.editor.gizmo_mode {
    vox_app::editor::GizmoMode::Translate => vox_render::gizmos::GizmoMode::Translate,
    vox_app::editor::GizmoMode::Rotate    => vox_render::gizmos::GizmoMode::Rotate,
    vox_app::editor::GizmoMode::Scale     => vox_render::gizmos::GizmoMode::Scale,
};
```

- [ ] Apply the same sync in `handle_mouse_button` before calling `hit_test`, so the hit test shape (line vs. circle vs. cube endpoint) matches the displayed gizmo.

---

## Task 4: Apply snap-to-grid in `update_drag`

**File:** `crates/vox_render/src/gizmos.rs` (or call-site in `engine_runner.rs`)

**Problem:** `GizmoRenderer::update_drag` returns a raw world-space delta. `SceneEditor.snap_enabled` and `snap_grid` are never consulted, so holding snap has no effect during drag.

**Option A — apply snap at call-site** (preferred, keeps gizmos.rs engine-agnostic):

In `handle_mouse_move` in `engine_runner.rs`, after computing `delta`:

```rust
let delta = self.gizmo.update_drag(x as f32, y as f32, entity_pos, vp, dw, dh);

// Apply snap-to-grid if enabled
let delta = if self.editor.snap_enabled && self.editor.snap_grid > 0.0 {
    let grid = self.editor.snap_grid;
    // Round each axis component to nearest grid multiple
    // but only on the active axis to avoid clamping non-active axes to 0
    match self.gizmo.active_axis {
        Some(vox_render::gizmos::Axis::X) => Vec3::new(
            (delta.x / grid).round() * grid, 0.0, 0.0,
        ),
        Some(vox_render::gizmos::Axis::Y) => Vec3::new(
            0.0, (delta.y / grid).round() * grid, 0.0,
        ),
        Some(vox_render::gizmos::Axis::Z) => Vec3::new(
            0.0, 0.0, (delta.z / grid).round() * grid,
        ),
        None => delta,
    }
} else {
    delta
};
```

**Note:** Because `update_drag` updates `drag_start_screen` incrementally (each call advances the start), snapping introduces sub-pixel stutter. For smoother snap, track total displacement since drag start and snap the accumulated total; emit only the diff vs. last snapped position. This is a follow-up refinement, not required for initial correctness.

- [ ] Add snap-to-grid application at call-site in `handle_mouse_move`.
- [ ] Optionally add a `snap_threshold: Option<f32>` parameter to `update_drag` in a future refactor.

---

## Task 5: Thick line rendering (2px-wide arrows)

**File:** `crates/vox_render/src/gizmos.rs`

**Why:** Single-pixel Bresenham lines are hard to click on a high-resolution display. A 2px-wide variant makes the arrows more visible and improves hit-test reliability.

- [ ] Add a `draw_line_thick` helper that draws the Bresenham line then offsets by (+1, 0) and (0, +1) and repeats:

```rust
fn draw_line_thick(
    pixels: &mut [[u8; 4]],
    width: u32,
    height: u32,
    x0: i32, y0: i32,
    x1: i32, y1: i32,
    color: [u8; 4],
    thickness: i32,  // typically 2
) {
    for dx in 0..thickness {
        for dy in 0..thickness {
            draw_line(pixels, width, height, x0 + dx, y0 + dy, x1 + dx, y1 + dy, color);
        }
    }
}
```

- [ ] Replace `draw_line` calls in `draw_overlay` (for the axis shafts) with `draw_line_thick(..., 2)`. Leave arrowhead/circle/cube-endpoint helpers using single-pixel lines (they are already multi-line structures).

---

## Task 6: Add `draw_line_thick` unit test

**File:** `crates/vox_render/src/gizmos.rs`

```rust
#[test]
fn draw_line_thick_covers_more_pixels_than_thin() {
    let mut thin = vec![[0u8; 4]; 100 * 100];
    let mut thick = vec![[0u8; 4]; 100 * 100];
    draw_line(&mut thin, 100, 100, 10, 50, 90, 50, [255, 0, 0, 255]);
    draw_line_thick(&mut thick, 100, 100, 10, 50, 90, 50, [255, 0, 0, 255], 2);
    let thin_lit = thin.iter().filter(|p| p[0] == 255).count();
    let thick_lit = thick.iter().filter(|p| p[0] == 255).count();
    assert!(thick_lit > thin_lit, "thick should cover more pixels: thin={thin_lit}, thick={thick_lit}");
}
```

---

## Acceptance Criteria

- [ ] `cargo test -p vox_render gizmos` passes all 6 tests (2 existing + 4 new from Tasks 2 and 6, plus `project_to_screen_identity` and `handle_mouse_press_returns_none_when_no_entity`).
- [ ] Opening the editor (Tab key) and selecting an entity shows colored axis arrows on the entity in the viewport.
- [ ] Clicking an axis arm begins a drag; cursor movement translates the entity along that axis only.
- [ ] Axis arrow turns yellow while active/dragging; returns to base color on release.
- [ ] Pressing W/E/R (or clicking gizmo mode buttons in the egui toolbar) switches between Translate/Rotate/Scale gizmo modes and the correct shape is rendered.
- [ ] With snap enabled and `snap_grid = 1.0`, entities snap to integer world positions during drag.
- [ ] `cargo test` passes with no regressions.

---

## File Map

| File | Change |
|------|--------|
| `crates/vox_render/src/gizmos.rs` | Add `draw()` flat-byte alias (Task 1), add 3 tests (Task 2), add `draw_line_thick` (Task 5), add thick-line test (Task 6) |
| `crates/vox_app/src/bin/engine_runner.rs` | Sync `gizmo_mode` before draw and hit-test (Task 3), apply snap-to-grid in `handle_mouse_move` (Task 4) |

No new files needed. Tasks 3 and 4 are call-site changes only (< 20 lines each).
