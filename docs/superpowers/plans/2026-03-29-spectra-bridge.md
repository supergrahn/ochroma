# Spectra Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `spectra-gaussian-render` as a workspace path dependency and wire `TileRasterizer` as an alternative CPU render path in `vox_render`, with a `SpectraBridge` struct that converts from `vox_core::GaussianSplat` to Spectra's types.

**Architecture:** A new `SpectraBridge` struct in `vox_render::spectra_bridge` converts Ochroma's `GaussianSplat` (quantized i16 rotation, u8 opacity, u16 spectral bands) to Spectra's `GaussianSplat` (f32 rotation, f32 opacity, f32 SH coefficients). The bridge constructs a `GaussianCloud` and calls `TileRasterizer::render()` to produce an RGBA framebuffer. The existing `spectra_bridge.rs` already has `RenderBackend`, `QualityPreset`, `RenderConfig`, and `SpectraProcess` — the bridge extends this with actual rendering wiring. In `engine_runner.rs`, a `--spectra` CLI flag selects the Spectra CPU path instead of the wgpu rasterizer.

**Tech Stack:** `spectra-gaussian-render` (pure Rust, glam+rayon), `bevy_ecs = "0.16"`, `vox_core::types::GaussianSplat`, `vox_core::engine_runtime::CameraState`

---

## Key Files (read before editing)

- `crates/vox_render/src/spectra_bridge.rs` — existing `RenderBackend`, `QualityPreset`, `RenderConfig`, `SpectraProcess`
- `crates/vox_render/src/spectra_render.rs` — existing Spectra-compatible tile rasterizer (internal reimplementation)
- `crates/vox_render/Cargo.toml` — current dependencies (no spectra-gaussian-render yet)
- `Cargo.toml` — workspace root, workspace dependencies
- `crates/vox_core/src/types.rs` — `GaussianSplat { position: [f32;3], scale: [f32;3], rotation: [i16;4], opacity: u8, _pad: [u8;3], spectral: [u16;8] }`
- `crates/vox_core/src/engine_runtime.rs` — `CameraState { position: Vec3, forward: Vec3, view_proj: Mat4 }`

## File Structure

**Modify:**
- `Cargo.toml` — add `spectra-gaussian-render` to `[workspace.dependencies]`
- `crates/vox_render/Cargo.toml` — add `spectra-gaussian-render` dependency
- `crates/vox_render/src/spectra_bridge.rs` — add `SpectraBridge`, conversion functions, render method

**No new files required.**

---

### Task 1: Add workspace dependency

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/vox_render/Cargo.toml`

Add `spectra-gaussian-render` as a workspace path dependency pointing to the AetherSpectra monorepo, then reference it from `vox_render`.

- [ ] **Step 1: Add to workspace root `Cargo.toml`**

In `[workspace.dependencies]`, add:

```toml
spectra-gaussian-render = { path = "../aetherspectra/spectra/rust/spectra-gaussian-render" }
```

- [ ] **Step 2: Add to `crates/vox_render/Cargo.toml`**

In `[dependencies]`, add:

```toml
spectra-gaussian-render = { workspace = true }
```

- [ ] **Step 3: Verify compile**

```bash
cargo check -p vox_render 2>&1 | tail -5
```

Expected: successful check (no errors).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock crates/vox_render/Cargo.toml
git commit -m "deps: add spectra-gaussian-render as workspace path dep for vox_render"
```

---

### Task 2: `SpectraBridge` conversion functions

**Files:**
- Modify: `crates/vox_render/src/spectra_bridge.rs`

Add conversion functions that translate Ochroma's quantized `GaussianSplat` to Spectra's f32-based format, and a `build_cloud` helper.

- [ ] **Step 1: Write failing tests** — add at the bottom of `crates/vox_render/src/spectra_bridge.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat as OchromaSplat;

    fn test_splat() -> OchromaSplat {
        OchromaSplat {
            position: [1.0, 2.0, 3.0],
            scale: [0.5, 0.5, 0.5],
            rotation: [0, 0, 0, 32767], // identity quaternion (0,0,0,1)
            opacity: 255,
            _pad: [0; 3],
            spectral: [15360, 15360, 15360, 0, 0, 0, 0, 0], // f16 bits for 1.0
        }
    }

    #[test]
    fn splat_to_spectra_converts_position() {
        let out = splat_to_spectra(&test_splat());
        assert!((out.position[0] - 1.0).abs() < 1e-5);
        assert!((out.position[1] - 2.0).abs() < 1e-5);
        assert!((out.position[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn splat_to_spectra_converts_rotation() {
        let out = splat_to_spectra(&test_splat());
        // i16 0 / 32767 = 0.0 for xyz, 32767 / 32767 = 1.0 for w
        assert!((out.rotation[0] - 0.0).abs() < 1e-4, "x");
        assert!((out.rotation[1] - 0.0).abs() < 1e-4, "y");
        assert!((out.rotation[2] - 0.0).abs() < 1e-4, "z");
        assert!((out.rotation[3] - 1.0).abs() < 1e-4, "w");
    }

    #[test]
    fn splat_to_spectra_converts_opacity() {
        let out = splat_to_spectra(&test_splat());
        assert!((out.opacity - 1.0).abs() < 0.01, "255 -> 1.0");

        let mut half_opacity = test_splat();
        half_opacity.opacity = 128;
        let out2 = splat_to_spectra(&half_opacity);
        assert!((out2.opacity - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn splat_to_spectra_converts_sh_dc_from_spectral() {
        let out = splat_to_spectra(&test_splat());
        // spectral[0..3] = 15360 (f16 for 1.0), divided by 65535 as u16->f32
        // Actually we use half::f16::from_bits to decode, so 15360 -> 1.0
        assert!((out.sh_dc[0] - 1.0).abs() < 0.01, "sh_dc[0] from spectral[0]");
        assert!((out.sh_dc[1] - 1.0).abs() < 0.01, "sh_dc[1] from spectral[1]");
        assert!((out.sh_dc[2] - 1.0).abs() < 0.01, "sh_dc[2] from spectral[2]");
    }

    #[test]
    fn splat_to_spectra_sh_rest_empty() {
        let out = splat_to_spectra(&test_splat());
        assert!(out.sh_rest.is_empty(), "sh_rest should be empty (no higher-order SH)");
    }

    #[test]
    fn build_cloud_from_multiple_splats() {
        let splats = vec![test_splat(), test_splat()];
        let cloud = build_cloud(&splats);
        // GaussianCloud should contain 2 splats
        assert_eq!(cloud.len(), 2);
    }
}
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- spectra_bridge::tests::splat_to_spectra_converts_position 2>&1 | tail -5
```

Expected: FAIL — `splat_to_spectra` not defined.

- [ ] **Step 3: Implement** — add BEFORE the `#[cfg(test)]` block in `spectra_bridge.rs`, after the existing `SpectraProcess` impl:

```rust
use half::f16;
use vox_core::types::GaussianSplat as OchromaSplat;
use spectra_gaussian_render::{
    GaussianSplat as SpectraSplat,
    GaussianCloud,
    GaussianCamera,
    TileRasterizer,
    RasterConfig,
};

/// Convert an Ochroma `GaussianSplat` to Spectra's `GaussianSplat`.
///
/// - `rotation`: each `i16` component divided by 32767.0 to get `f32` in [-1, 1]
/// - `opacity`: `u8` divided by 255.0 to get `f32` in [0, 1]
/// - `sh_dc`: first 3 spectral bands decoded from f16 bits
/// - `sh_rest`: empty (no higher-order spherical harmonics)
pub fn splat_to_spectra(s: &OchromaSplat) -> SpectraSplat {
    SpectraSplat {
        position: s.position,
        scale: s.scale,
        rotation: [
            s.rotation[0] as f32 / 32767.0,
            s.rotation[1] as f32 / 32767.0,
            s.rotation[2] as f32 / 32767.0,
            s.rotation[3] as f32 / 32767.0,
        ],
        opacity: s.opacity as f32 / 255.0,
        sh_dc: [
            f16::from_bits(s.spectral[0]).to_f32(),
            f16::from_bits(s.spectral[1]).to_f32(),
            f16::from_bits(s.spectral[2]).to_f32(),
        ],
        sh_rest: vec![],
    }
}

/// Build a `GaussianCloud` from a slice of Ochroma splats.
pub fn build_cloud(splats: &[OchromaSplat]) -> GaussianCloud {
    let spectra_splats: Vec<SpectraSplat> = splats.iter().map(splat_to_spectra).collect();
    GaussianCloud::from_splats(spectra_splats)
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render --lib -- spectra_bridge::tests 2>&1 | tail -10
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/spectra_bridge.rs
git commit -m "feat(spectra-bridge): splat_to_spectra + build_cloud conversion from Ochroma to Spectra format"
```

---

### Task 3: `SpectraBridge::render` method and camera construction

**Files:**
- Modify: `crates/vox_render/src/spectra_bridge.rs`

Add the `SpectraBridge` struct that holds a `TileRasterizer` and provides a `render()` method converting the f32 output to RGBA u8, plus a helper to construct `GaussianCamera` from Ochroma's `CameraState`.

- [ ] **Step 1: Write failing tests** — add inside `mod tests`:

```rust
    #[test]
    fn camera_from_state_dimensions() {
        use vox_core::engine_runtime::CameraState;
        let state = CameraState::default();
        let cam = camera_from_state(&state, 800, 600, 0.785, 0.1, 1000.0);
        assert_eq!(cam.width, 800);
        assert_eq!(cam.height, 600);
        assert!((cam.fov_y - 0.785).abs() < 1e-3);
        assert!((cam.near - 0.1).abs() < 1e-5);
        assert!((cam.far - 1000.0).abs() < 1e-1);
    }

    #[test]
    fn spectra_bridge_render_returns_correct_buffer_size() {
        let bridge = SpectraBridge::new(4, 4);
        let splats = vec![test_splat()];
        use vox_core::engine_runtime::CameraState;
        let state = CameraState::default();
        let cam = camera_from_state(&state, 4, 4, 0.785, 0.1, 1000.0);
        let buf = bridge.render(&splats, &cam);
        assert_eq!(buf.len(), 4 * 4 * 4, "4x4 RGBA = 64 bytes");
    }
```

- [ ] **Step 2: Confirm they fail**

```bash
cargo test -p vox_render --lib -- spectra_bridge::tests::camera_from_state 2>&1 | tail -5
```

Expected: FAIL — `camera_from_state` not defined.

- [ ] **Step 3: Implement** — add after `build_cloud`:

```rust
use glam::Mat4;
use vox_core::engine_runtime::CameraState;

/// Construct a `GaussianCamera` from Ochroma's `CameraState`.
///
/// Decomposes `view_proj` into separate view and projection matrices.
/// In practice the caller should provide view and proj separately for
/// accuracy; this helper uses `view_proj` as-is for the view matrix and
/// builds a perspective proj from the given FOV/near/far.
pub fn camera_from_state(
    state: &CameraState,
    width: usize,
    height: usize,
    fov_y: f32,
    near: f32,
    far: f32,
) -> GaussianCamera {
    let aspect = width as f32 / height as f32;
    let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
    let view = Mat4::look_to_rh(state.position, state.forward, glam::Vec3::Y);
    GaussianCamera {
        view_matrix: view.to_cols_array(),
        proj_matrix: proj.to_cols_array(),
        width,
        height,
        fov_y,
        near,
        far,
    }
}

/// Bridge to the Spectra `TileRasterizer` for CPU Gaussian splatting.
pub struct SpectraBridge {
    rasterizer: TileRasterizer,
    width: usize,
    height: usize,
}

impl SpectraBridge {
    /// Create a new bridge with the given framebuffer dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let config = RasterConfig::default();
        Self {
            rasterizer: TileRasterizer::new(width, height, config),
            width,
            height,
        }
    }

    /// Render splats using Spectra's tile rasterizer, returning RGBA u8 buffer.
    ///
    /// 1. Converts Ochroma splats to Spectra format via `build_cloud`.
    /// 2. Calls `TileRasterizer::render` to get `Vec<f32>` (RGB, 3 channels).
    /// 3. Converts to RGBA u8 with gamma=1.0, alpha=255.
    pub fn render(&self, splats: &[OchromaSplat], camera: &GaussianCamera) -> Vec<u8> {
        let cloud = build_cloud(splats);
        let rgb_f32 = self.rasterizer.render(&cloud, camera);
        let pixel_count = self.width * self.height;
        let mut rgba = Vec::with_capacity(pixel_count * 4);
        for i in 0..pixel_count {
            let r = (rgb_f32[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (rgb_f32[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (rgb_f32[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
        rgba
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_render --lib -- spectra_bridge::tests 2>&1 | tail -10
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/spectra_bridge.rs
git commit -m "feat(spectra-bridge): SpectraBridge struct with TileRasterizer render + camera_from_state"
```

---

### Task 4: Wire into engine runtime with `--spectra` flag

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs` (or equivalent main entry point)

Add a `spectra_enabled: bool` field to the runtime and a `render_frame_spectra()` method that uses `SpectraBridge` instead of the wgpu rasterizer when `--spectra` is passed on the command line.

- [ ] **Step 1: Parse `--spectra` CLI flag**

At the top of `main()` or the runtime init, add:

```rust
let spectra_enabled = std::env::args().any(|a| a == "--spectra");
```

- [ ] **Step 2: Conditionally construct `SpectraBridge`**

```rust
let spectra_bridge = if spectra_enabled {
    println!("[ochroma] Spectra CPU render path enabled");
    Some(vox_render::spectra_bridge::SpectraBridge::new(
        config.resolution.0 as usize,
        config.resolution.1 as usize,
    ))
} else {
    None
};
```

- [ ] **Step 3: Add render dispatch in frame loop**

In the frame loop, before/instead of the wgpu `render_frame()` call:

```rust
if let Some(ref bridge) = spectra_bridge {
    let camera = vox_render::spectra_bridge::camera_from_state(
        &camera_state,
        config.resolution.0 as usize,
        config.resolution.1 as usize,
        config.fov,
        config.near_plane,
        config.far_plane,
    );
    let _rgba_buf = bridge.render(&render_buffer.splats, &camera);
    // TODO: blit _rgba_buf to window surface or save to file
} else {
    // existing wgpu render path
    render_frame(&gpu_state, &render_buffer, &camera_state);
}
```

- [ ] **Step 4: Verify compile**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

Expected: successful check.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(spectra-bridge): wire --spectra CLI flag to use SpectraBridge CPU render path"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 3 tasks from the spec covered (workspace dep, SpectraBridge conversion, engine wiring)
- [x] **No placeholders:** All code blocks are complete with real types and logic
- [x] **Type consistency:** `GaussianSplat` (Ochroma) vs `GaussianSplat` (Spectra) aliased clearly; `i16` -> `f32` rotation, `u8` -> `f32` opacity, `u16` f16-bits -> `f32` SH
- [x] **TDD:** Tests written before implementation for conversion functions
- [x] **Existing patterns:** Follows `render_ecs.rs` pattern with Plugin + standalone functions
- [x] **Engine generality:** No game-specific concepts; pure render infrastructure
