# Spectral Material Viewport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-band heatmap viewport mode — pressing `Tab` cycles through Full Color → Band 0 → Band 1 → … → Band 7 → Full Color, rendering the scene using the actual EWA splat pipeline with each spectral band mapped to a false-color gradient.

**Architecture:** Three layers. (1) `render_spectral_band_u8` in `spectra_render.rs` renders the scene identically to `render_with_spectra_u8` except each splat's color is replaced by `band_to_heatmap(splat.spectral_f32(band))` — no illuminant, no XYZ conversion, pure spectral energy visualization. (2) `SpectralViewportMode` enum in a new `spectral_viewport.rs` wraps the two render paths with a `cycle_next()` / `label()` API. (3) `engine_runner.rs` wires `Tab` to cycle modes, routes `render_frame` to the correct render path, and burns the mode label into the HUD.

**Why better than Unreal:** Unreal's Material Editor renders a sphere preview with simplified baked lighting — it is an approximation of real output. Ochroma's viewport IS the production EWA renderer. Spectral band view shows the literal per-splat spectral energy that drives audio, lighting, and GI — not a preview proxy. Engineers can visually verify spectral material authorship against the actual render pipeline in one keypress.

**Tech Stack:** Rust, `vox_core::types::GaussianSplat` (`spectral_f32`), `vox_render::spectra_render`, `half::f16` (already workspace dep).

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_render/src/spectra_render.rs` | Modify | Add `band_to_heatmap`, `render_spectral_band_u8` |
| `crates/vox_render/src/spectral_viewport.rs` | Create | `SpectralViewportMode` enum + `BAND_NAMES` |
| `crates/vox_render/src/lib.rs` | Modify | `pub mod spectral_viewport;` |
| `crates/vox_app/src/bin/engine_runner.rs` | Modify | Field + Tab keybinding + render path selection + HUD |

---

## Task 1: `band_to_heatmap` and `render_spectral_band_u8`

**Files:**
- Modify: `crates/vox_render/src/spectra_render.rs`

The existing private `ochroma_to_gaussian3d` converts GaussianSplat + Illuminant → Gaussian3D with full spectral→RGB. For band rendering we replace only the `color` field — geometry (position, log_scale, rotation, opacity) is computed identically.

- [ ] Add `band_to_heatmap` and `splat_to_gaussian3d_band` to `crates/vox_render/src/spectra_render.rs`, right after the `ochroma_to_gaussian3d` function:

```rust
/// Maps a scalar intensity [0, 1] to a 5-stop false-color gradient:
/// black → blue → cyan → green → yellow → red.
/// Identical to common scientific heatmaps (e.g. matplotlib "turbo" simplified).
pub fn band_to_heatmap(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    match () {
        _ if t < 0.2 => {
            let s = t / 0.2;
            [0.0, 0.0, s]                          // black → blue
        }
        _ if t < 0.4 => {
            let s = (t - 0.2) / 0.2;
            [0.0, s, 1.0]                          // blue → cyan
        }
        _ if t < 0.6 => {
            let s = (t - 0.4) / 0.2;
            [0.0, 1.0, 1.0 - s]                    // cyan → green
        }
        _ if t < 0.8 => {
            let s = (t - 0.6) / 0.2;
            [s, 1.0, 0.0]                          // green → yellow
        }
        _ => {
            let s = (t - 0.8) / 0.2;
            [1.0, 1.0 - s, 0.0]                    // yellow → red
        }
    }
}

/// Converts a GaussianSplat to Gaussian3D using one spectral band as false-color.
/// Geometry (position, scale, rotation, opacity) is identical to `ochroma_to_gaussian3d`.
fn splat_to_gaussian3d_band(splat: &GaussianSplat, band: usize) -> Gaussian3D {
    // --- geometry (mirrors ochroma_to_gaussian3d exactly) ---
    let log_scale = [
        splat.scale[0].max(1e-6).ln(),
        splat.scale[1].max(1e-6).ln(),
        splat.scale[2].max(1e-6).ln(),
    ];
    let qx = splat.rotation[0] as f32 / 32767.0;
    let qy = splat.rotation[1] as f32 / 32767.0;
    let qz = splat.rotation[2] as f32 / 32767.0;
    let qw = splat.rotation[3] as f32 / 32767.0;
    let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt().max(1e-8);
    let rotation = [qw / len, qx / len, qy / len, qz / len];
    let opacity = splat.opacity as f32 / 255.0;
    // --- color: single band → heatmap ---
    let intensity = splat.spectral_f32(band.min(7));
    let color = band_to_heatmap(intensity);
    Gaussian3D { position: splat.position, log_scale, rotation, color, opacity }
}

/// Render the scene as a false-color heatmap of a single spectral band.
/// `band` must be in `0..8`; values outside that range clamp to band 7.
///
/// Returns `width * height` RGBA pixels identical in format to `render_with_spectra_u8`.
pub fn render_spectral_band_u8(
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    width: u32,
    height: u32,
    band: usize,
) -> Vec<[u8; 4]> {
    let cam = ochroma_to_spectra_camera(camera, width, height);
    let gaussians: Vec<Gaussian3D> = splats
        .iter()
        .map(|s| splat_to_gaussian3d_band(s, band))
        .collect();
    let floats = render_cpu_internal(&gaussians, &cam);
    floats
        .chunks_exact(4)
        .map(|px| {
            [
                (px[0].clamp(0.0, 1.0) * 255.0) as u8,
                (px[1].clamp(0.0, 1.0) * 255.0) as u8,
                (px[2].clamp(0.0, 1.0) * 255.0) as u8,
                (px[3].clamp(0.0, 1.0) * 255.0) as u8,
            ]
        })
        .collect()
}
```

- [ ] Add tests for `band_to_heatmap` and `render_spectral_band_u8` inside the existing `#[cfg(test)]` module in `spectra_render.rs`:

```rust
#[test]
fn band_to_heatmap_zero_is_black() {
    let c = band_to_heatmap(0.0);
    assert_eq!(c, [0.0, 0.0, 0.0]);
}

#[test]
fn band_to_heatmap_one_is_red() {
    let c = band_to_heatmap(1.0);
    assert_eq!(c[0], 1.0);
    assert!(c[1] < 0.01, "green channel should be near 0 at t=1");
    assert_eq!(c[2], 0.0);
}

#[test]
fn band_to_heatmap_clamps_out_of_range() {
    assert_eq!(band_to_heatmap(-1.0), band_to_heatmap(0.0));
    assert_eq!(band_to_heatmap(2.0), band_to_heatmap(1.0));
}

#[test]
fn render_spectral_band_returns_correct_pixel_count() {
    let splats: Vec<GaussianSplat> = Vec::new(); // empty scene, black frame
    let camera = RenderCamera {
        view: glam::Mat4::IDENTITY,
        proj: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
    };
    let pixels = render_spectral_band_u8(&splats, &camera, 16, 16, 0);
    assert_eq!(pixels.len(), 16 * 16);
}

#[test]
fn render_spectral_band_differs_per_band_when_nonzero_spectral() {
    // A single splat with distinct per-band values
    let mut spectral = [0u16; 8];
    // Band 0 = 1.0 (high blue on heatmap), band 7 = 0.0 (black)
    spectral[0] = half::f16::from_f32(1.0).to_bits();
    spectral[7] = half::f16::from_f32(0.0).to_bits();
    let splat = GaussianSplat {
        position: [0.0, 0.0, -2.0],
        scale: [0.5, 0.5, 0.5],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        _pad: [0; 3],
        spectral,
    };
    let camera = RenderCamera {
        view: glam::Mat4::IDENTITY,
        proj: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
    };
    let pixels_b0 = render_spectral_band_u8(&[splat], &camera, 32, 32, 0);
    let pixels_b7 = render_spectral_band_u8(&[splat], &camera, 32, 32, 7);
    // Find any non-background pixel in each render
    let sum_b0: u32 = pixels_b0.iter().map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32).sum();
    let sum_b7: u32 = pixels_b7.iter().map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32).sum();
    assert_ne!(sum_b0, sum_b7, "band 0 (full) and band 7 (empty) should differ");
}
```

- [ ] Run:
```bash
cargo test -p vox_render band_to_heatmap render_spectral_band
```
Expected: 5 tests pass (3 heatmap + 2 render_spectral_band).

- [ ] Commit:
```bash
git commit -m "feat(render): band_to_heatmap + render_spectral_band_u8 for spectral viewport"
```

---

## Task 2: `SpectralViewportMode` enum

**Files:**
- Create: `crates/vox_render/src/spectral_viewport.rs`
- Modify: `crates/vox_render/src/lib.rs`

- [ ] Create `crates/vox_render/src/spectral_viewport.rs`:

```rust
//! Viewport mode for per-band spectral visualization.

/// Names for the 8 spectral bands as shown in the HUD.
/// Bands are ordered high-frequency (blue, electric) to low-frequency (red, bass).
pub const BAND_NAMES: [&str; 8] = [
    "Band 0 — 8 kHz (blue/electric)",
    "Band 1 — 4 kHz",
    "Band 2 — 2 kHz",
    "Band 3 — 1 kHz",
    "Band 4 — 500 Hz",
    "Band 5 — 250 Hz",
    "Band 6 — 125 Hz",
    "Band 7 — 80 Hz (red/bass)",
];

/// Controls which render path the engine viewport uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpectralViewportMode {
    /// Normal full-color EWA rendering (default).
    #[default]
    Full,
    /// False-color heatmap of a single spectral band (0–7).
    Band(usize),
}

impl SpectralViewportMode {
    /// Cycle to the next mode: Full → Band(0) → Band(1) → … → Band(7) → Full.
    pub fn cycle_next(self) -> Self {
        match self {
            Self::Full => Self::Band(0),
            Self::Band(b) if b < 7 => Self::Band(b + 1),
            Self::Band(_) => Self::Full,
        }
    }

    /// Short label for the HUD.
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "Viewport: Full Color",
            Self::Band(b) => BAND_NAMES[b.min(7)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_full_to_band0() {
        assert_eq!(SpectralViewportMode::Full.cycle_next(), SpectralViewportMode::Band(0));
    }

    #[test]
    fn cycle_band7_wraps_to_full() {
        assert_eq!(SpectralViewportMode::Band(7).cycle_next(), SpectralViewportMode::Full);
    }

    #[test]
    fn cycle_band3_to_band4() {
        assert_eq!(SpectralViewportMode::Band(3).cycle_next(), SpectralViewportMode::Band(4));
    }

    #[test]
    fn label_full() {
        assert_eq!(SpectralViewportMode::Full.label(), "Viewport: Full Color");
    }

    #[test]
    fn label_band0_contains_8khz() {
        assert!(SpectralViewportMode::Band(0).label().contains("8 kHz"));
    }

    #[test]
    fn label_band7_contains_bass() {
        assert!(SpectralViewportMode::Band(7).label().contains("bass"));
    }

    #[test]
    fn default_is_full() {
        assert_eq!(SpectralViewportMode::default(), SpectralViewportMode::Full);
    }
}
```

- [ ] Add `pub mod spectral_viewport;` to `crates/vox_render/src/lib.rs`.

- [ ] Run:
```bash
cargo test -p vox_render spectral_viewport
```
Expected: 7 tests pass.

- [ ] Commit:
```bash
git commit -m "feat(render): SpectralViewportMode — Full + Band(0..7) with cycle and HUD label"
```

---

## Task 3: Wire into engine_runner — Tab to cycle, HUD label, render path

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Add `spectral_viewport_mode` field to `EngineApp`. There are **two** `Self { ... }` construction sites — add to **both**:

```rust
// In struct definition, add:
spectral_viewport_mode: vox_render::spectral_viewport::SpectralViewportMode,

// In both Self { ... } construction sites, add:
spectral_viewport_mode: vox_render::spectral_viewport::SpectralViewportMode::default(),
```

- [ ] In the `keyboard_input` handler (where other key presses are handled), add `Tab` cycling:

```rust
if self.input_state.just_pressed(vox_core::input::Key::Tab) {
    self.spectral_viewport_mode = self.spectral_viewport_mode.cycle_next();
    println!("[ochroma] Spectral viewport: {}", self.spectral_viewport_mode.label());
}
```

- [ ] In `render_frame`, replace the existing SPECTRA PATH block (inside the `!self.spectral_bypass` branch) to route through spectral mode:

Find this block:
```rust
// SPECTRA PATH: tile-based EWA Gaussian splatting renderer
let fb = vox_render::spectra_render::render_with_spectra_u8(
    &render_splats,
    &render_camera,
    render_w,
    render_h,
    &illuminant,
);
```

Replace with:
```rust
// SPECTRA PATH: tile-based EWA Gaussian splatting renderer
let fb = match self.spectral_viewport_mode {
    vox_render::spectral_viewport::SpectralViewportMode::Full => {
        vox_render::spectra_render::render_with_spectra_u8(
            &render_splats,
            &render_camera,
            render_w,
            render_h,
            &illuminant,
        )
    }
    vox_render::spectral_viewport::SpectralViewportMode::Band(band) => {
        vox_render::spectra_render::render_spectral_band_u8(
            &render_splats,
            &render_camera,
            render_w,
            render_h,
            band,
        )
    }
};
```

- [ ] In the HUD overlay section of `render_frame` (where `burn_text` is called for stats), add the viewport mode label. After the existing FPS/camera stat burns, add:

```rust
// Spectral viewport mode label (top-left, below FPS)
if self.spectral_viewport_mode != vox_render::spectral_viewport::SpectralViewportMode::Full {
    burn_text(
        &mut final_pixels,
        display_w as usize,
        display_h as usize,
        self.spectral_viewport_mode.label(),
        4,
        32,
    );
}
```

- [ ] Verify compile:
```bash
cargo check --bin ochroma
```
Expected: zero errors.

- [ ] Run full test suite:
```bash
cargo test
```
Expected: all tests pass.

- [ ] Commit:
```bash
git commit -m "feat(app): spectral viewport mode — Tab cycles bands, HUD label, EWA band render path"
```

---

## Acceptance Criteria

| # | Test | Command |
|---|------|---------|
| 1 | Heatmap + band render pass | `cargo test -p vox_render band_to_heatmap render_spectral_band` |
| 2 | Mode cycle + labels | `cargo test -p vox_render spectral_viewport` |
| 3 | Engine compiles with wiring | `cargo check --bin ochroma` |
| 4 | Full workspace green | `cargo test` |
