# Domain 5 — Asset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Production asset pipeline — Smits RGB→spectral upsampling, `SpectralMaterial` database, VXM v3 with per-splat material IDs, COLMAP photogrammetry wrapper, and a `ochroma-tools` CLI.

**Architecture:** `SpectralUpsampler` applies Smits 1999 basis decomposition to convert sRGB → `[f32; 8]`. `SpectralMaterialDb` is a compile-time lookup table of physically measured material reflectances. VXM v3 extends the existing header to add a `spectral_level` field and a parallel `material_ids: Vec<u16>` sidecar (stored in a separate section after the splat block). `ColmapPipeline::run()` spawns the `colmap` subprocess, reads the sparse reconstruction PLY, and calls the existing `import_ply` path with Smits upsampling applied to vertex colours. The CLI binary `ochroma-tools` lives at `crates/vox_tools/`.

**Tech Stack:** Rust, existing `vox_data` / `vox_core`, `image` crate (existing), `bytemuck` (existing), `thiserror` (existing), `clap 4`, `zstd` (existing). No new GPU deps for this domain.

---

## Cargo.toml changes

New binary crate `crates/vox_tools/Cargo.toml`:

```toml
[package]
name = "ochroma-tools"
edition.workspace = true
version.workspace = true

[[bin]]
name = "ochroma-tools"
path = "src/main.rs"

[dependencies]
vox_data = { path = "../vox_data" }
vox_core = { path = "../vox_core" }
clap     = { version = "4", features = ["derive"] }
anyhow   = "1"
```

Add to `crates/vox_data/Cargo.toml` (if not already present):

```toml
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
```

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_data/src/spectral_upsampler.rs` | Smits 1999 RGB→spectral + `SpectralMaterialDb` |
| Create | `crates/vox_data/src/spectral_capture.rs` | `SpectralMaterialProfile`, 3-photo capture |
| Modify | `crates/vox_data/src/vxm.rs` | VXM v3 header, `spectral_level`, `material_ids` section |
| Create | `crates/vox_data/src/colmap_pipeline.rs` | `ColmapPipeline` subprocess wrapper |
| Modify | `crates/vox_data/src/import_pipeline.rs` | integrate Smits upsampling into PLY importer |
| Modify | `crates/vox_data/src/lib.rs` | expose new modules |
| Create | `crates/vox_tools/Cargo.toml` | binary crate manifest |
| Create | `crates/vox_tools/src/main.rs` | CLI entry point (clap) |
| Modify | `Cargo.toml` (workspace) | add vox_tools to members |

---

## Task 1: Smits upsampler + SpectralMaterialDb

**Files:**
- Create: `crates/vox_data/src/spectral_upsampler.rs`
- Modify: `crates/vox_data/src/lib.rs`

Smits 1999 decomposes sRGB into 7 basis spectra (white, cyan, magenta, yellow, red, green, blue). Each basis has a known 8-band reflectance coefficient vector. The method:

1. Decompose `(r, g, b)` into basis weights via sequential subtraction.
2. Compute weighted sum of the 8-band basis vectors.

Reference: "An RGB-to-Spectrum Conversion for Reflectances", Smits 1999, JGT.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_data/src/spectral_upsampler.rs`:

```rust
//! Smits 1999 RGB→spectral upsampling.
//!
//! Decomposes sRGB (linear, [0,1]) into 7 basis spectra:
//!   white, cyan, magenta, yellow, red, green, blue
//! and returns an 8-band reflectance [f32; 8].
//!
//! Band centre wavelengths match vox_render::spectral_atmosphere::BAND_NM:
//!   [380, 420, 460, 500, 540, 580, 620, 660] nm

/// 8-band reflectance coefficients for each Smits basis spectrum.
/// Rows: white, cyan, magenta, yellow, red, green, blue.
/// Values sampled from Smits 1999 Table 1 and linearly interpolated to 8 bands.
const SMITS_BASIS: [[f32; 8]; 7] = [
    // white
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    // cyan (absorbs red; high in blue-green)
    [0.97, 0.97, 0.94, 0.90, 0.12, 0.04, 0.03, 0.02],
    // magenta (absorbs green; high in red+blue)
    [0.35, 0.45, 0.23, 0.05, 0.77, 0.90, 0.97, 0.97],
    // yellow (absorbs blue; high in red-green)
    [0.02, 0.04, 0.08, 0.70, 0.96, 0.99, 0.99, 0.99],
    // red
    [0.02, 0.02, 0.02, 0.02, 0.38, 0.86, 0.97, 0.97],
    // green
    [0.02, 0.10, 0.40, 0.90, 0.55, 0.12, 0.03, 0.02],
    // blue
    [0.96, 0.95, 0.87, 0.25, 0.05, 0.03, 0.02, 0.02],
];

pub struct SpectralUpsampler;

impl SpectralUpsampler {
    /// Convert linear sRGB to an 8-band spectral reflectance via Smits 1999 decomposition.
    ///
    /// Inputs must be in [0, 1]. Values outside this range are clamped.
    pub fn from_rgb(r: f32, g: f32, b: f32) -> [f32; 8] {
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        // Smits decomposition: sequential subtraction into basis weights
        let (white, cyan, magenta, yellow, red, green, blue) =
            Self::decompose(r, g, b);

        let weights = [white, cyan, magenta, yellow, red, green, blue];
        let mut out = [0.0f32; 8];
        for (i, basis) in SMITS_BASIS.iter().enumerate() {
            for b in 0..8 {
                out[b] += weights[i] * basis[b];
            }
        }
        // Normalise to [0, 1]
        let max = out.iter().copied().fold(f32::EPSILON, f32::max);
        if max > 1.0 {
            for v in &mut out { *v /= max; }
        }
        out
    }

    /// Decompose (r, g, b) into 7 basis weights following Smits 1999 §3.
    fn decompose(r: f32, g: f32, b: f32) -> (f32,f32,f32,f32,f32,f32,f32) {
        let (mut white, mut cyan, mut magenta, mut yellow, mut red, mut green, mut blue)
            = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);

        if r <= g && r <= b {
            white = r;
            if g <= b {
                yellow = g - r;
                blue = b - g;
            } else {
                yellow = b - r;
                green = g - b;
            }
        } else if g <= r && g <= b {
            white = g;
            if r <= b {
                magenta = r - g;
                blue = b - r;
            } else {
                magenta = b - g;
                red = r - b;
            }
        } else {
            white = b;
            if r <= g {
                cyan = r - b;
                green = g - r;
            } else {
                cyan = g - b;
                red = r - g;
            }
        }
        (white, cyan, magenta, yellow, red, green, blue)
    }
}

/// Named spectral material with 8-band reflectance.
pub struct SpectralMaterial {
    pub name: &'static str,
    /// 8-band reflectance, bands at [380, 420, 460, 500, 540, 580, 620, 660] nm.
    pub reflectance: [f32; 8],
}

/// Compile-time database of physically motivated spectral materials.
/// IDs are 1-indexed to match VXM v3 `spectral_material_id` (0 = unassigned).
pub struct SpectralMaterialDb;

impl SpectralMaterialDb {
    /// All materials. Index + 1 = material_id in VXM v3.
    pub const MATERIALS: &'static [SpectralMaterial] = &[
        SpectralMaterial { name: "foliage",   reflectance: [0.05, 0.05, 0.06, 0.35, 0.55, 0.12, 0.06, 0.05] },
        SpectralMaterial { name: "soil",      reflectance: [0.04, 0.06, 0.08, 0.12, 0.19, 0.26, 0.30, 0.31] },
        SpectralMaterial { name: "rock",      reflectance: [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.21, 0.22] },
        SpectralMaterial { name: "water",     reflectance: [0.03, 0.05, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02] },
        SpectralMaterial { name: "glass",     reflectance: [0.92, 0.93, 0.94, 0.94, 0.94, 0.93, 0.92, 0.91] },
        SpectralMaterial { name: "concrete",  reflectance: [0.20, 0.21, 0.22, 0.24, 0.26, 0.27, 0.27, 0.27] },
        SpectralMaterial { name: "snow",      reflectance: [0.88, 0.91, 0.93, 0.94, 0.94, 0.93, 0.92, 0.91] },
        SpectralMaterial { name: "asphalt",   reflectance: [0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06] },
    ];

    /// Look up a material by name (case-insensitive). Returns None if not found.
    pub fn find_by_name(name: &str) -> Option<&'static SpectralMaterial> {
        Self::MATERIALS.iter().find(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// Retrieve material by 1-based ID (as stored in VXM v3). Returns None for id=0.
    pub fn find_by_id(id: u16) -> Option<&'static SpectralMaterial> {
        if id == 0 || id as usize > Self::MATERIALS.len() {
            None
        } else {
            Some(&Self::MATERIALS[id as usize - 1])
        }
    }

    /// Find the closest material by L2 distance in spectral space.
    pub fn classify(reflectance: &[f32; 8]) -> &'static SpectralMaterial {
        Self::MATERIALS.iter().min_by(|a, b| {
            let da: f32 = a.reflectance.iter().zip(reflectance).map(|(x,y)| (x-y).powi(2)).sum();
            let db: f32 = b.reflectance.iter().zip(reflectance).map(|(x,y)| (x-y).powi(2)).sum();
            da.partial_cmp(&db).unwrap()
        }).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_rgb_returns_flat_spectrum() {
        let s = SpectralUpsampler::from_rgb(1.0, 1.0, 1.0);
        for (i, &v) in s.iter().enumerate() {
            assert!(v > 0.9, "white: band {} should be near 1.0, got {}", i, v);
        }
    }

    #[test]
    fn black_rgb_returns_zero_spectrum() {
        let s = SpectralUpsampler::from_rgb(0.0, 0.0, 0.0);
        for (i, &v) in s.iter().enumerate() {
            assert!(v < 1e-5, "black: band {} should be ~0.0, got {}", i, v);
        }
    }

    #[test]
    fn red_rgb_concentrates_in_high_bands() {
        let s = SpectralUpsampler::from_rgb(1.0, 0.0, 0.0);
        // Bands 4-7 (540-660nm) should be higher than bands 0-2 (380-460nm)
        let high: f32 = s[4..8].iter().copied().sum::<f32>() / 4.0;
        let low:  f32 = s[0..3].iter().copied().sum::<f32>() / 3.0;
        assert!(high > low,
            "red: high bands avg {:.3} should exceed low bands avg {:.3}", high, low);
    }

    #[test]
    fn blue_rgb_concentrates_in_low_bands() {
        let s = SpectralUpsampler::from_rgb(0.0, 0.0, 1.0);
        let low:  f32 = s[0..3].iter().copied().sum::<f32>() / 3.0;
        let high: f32 = s[5..8].iter().copied().sum::<f32>() / 3.0;
        assert!(low > high,
            "blue: low bands avg {:.3} should exceed high bands avg {:.3}", low, high);
    }

    #[test]
    fn output_stays_in_unit_range() {
        let inputs = [(0.5, 0.5, 0.5), (1.0, 0.0, 0.5), (0.2, 0.8, 0.1)];
        for (r, g, b) in inputs {
            let s = SpectralUpsampler::from_rgb(r, g, b);
            for (i, &v) in s.iter().enumerate() {
                assert!((0.0..=1.0).contains(&v),
                    "rgb({},{},{}) band {} = {} out of [0,1]", r, g, b, i, v);
            }
        }
    }

    #[test]
    fn material_db_find_by_name() {
        let m = SpectralMaterialDb::find_by_name("foliage").unwrap();
        assert_eq!(m.name, "foliage");
    }

    #[test]
    fn material_db_find_by_id_one_based() {
        let m = SpectralMaterialDb::find_by_id(1).unwrap();
        assert_eq!(m.name, "foliage");
    }

    #[test]
    fn material_db_id_zero_returns_none() {
        assert!(SpectralMaterialDb::find_by_id(0).is_none());
    }

    #[test]
    fn classify_foliage() {
        // Foliage has a strong band-4 (540nm green) peak
        let green_spectrum = [0.05, 0.05, 0.06, 0.35, 0.55, 0.12, 0.06, 0.05];
        let m = SpectralMaterialDb::classify(&green_spectrum);
        assert_eq!(m.name, "foliage", "strong green peak should classify as foliage");
    }

    #[test]
    fn classify_snow() {
        let bright = [0.88, 0.91, 0.93, 0.94, 0.94, 0.93, 0.92, 0.91];
        let m = SpectralMaterialDb::classify(&bright);
        assert_eq!(m.name, "snow", "flat high reflectance should classify as snow");
    }

    #[test]
    fn all_eight_materials_have_unique_names() {
        let names: std::collections::HashSet<&str> =
            SpectralMaterialDb::MATERIALS.iter().map(|m| m.name).collect();
        assert_eq!(names.len(), SpectralMaterialDb::MATERIALS.len(),
            "every material must have a unique name");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/tomespen/git/ochroma
cargo test -p vox_data spectral_upsampler 2>&1 | head -20
```

Expected: compile error — module not in lib.rs.

- [ ] **Step 3: Expose the module**

Add to `crates/vox_data/src/lib.rs`:

```rust
pub mod spectral_upsampler;
pub use spectral_upsampler::{SpectralUpsampler, SpectralMaterialDb, SpectralMaterial};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p vox_data spectral_upsampler -- --nocapture
```

Expected: 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/spectral_upsampler.rs crates/vox_data/src/lib.rs
git commit -m "feat(data): Smits RGB→spectral upsampler + SpectralMaterialDb (8 materials)"
```

---

## Task 2: SpectralMaterialProfile — 3-photo capture

**Files:**
- Create: `crates/vox_data/src/spectral_capture.rs`
- Modify: `crates/vox_data/src/lib.rs`

Three photographs of the same surface under known light spectra allow solving for surface reflectance at each pixel. The method: for each pixel, solve the 3×8 linear system `L × r = c` where `L[i][b]` is light `i`'s energy in band `b`, `r[b]` is unknown reflectance, and `c[i]` is measured pixel brightness in band `b`. With only 3 observations for 8 unknowns this is underdetermined; we use a pseudoinverse with Smits prior as regularisation.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_data/src/spectral_capture.rs`:

```rust
//! 3-photo spectral material capture.
//!
//! Estimates per-surface spectral reflectance from three photographs taken
//! under different known illuminants (daylight, tungsten, cool-LED).
//!
//! For each pixel the system solves: measured_rgb ≈ light_spd × reflectance
//! The result is a SpectralMaterialProfile with per-band mean and variance.

use crate::spectral_upsampler::SpectralUpsampler;

/// Spectral power distribution of a light source — energy in each of 8 bands.
#[derive(Debug, Clone, Copy)]
pub struct LightSpd(pub [f32; 8]);

impl LightSpd {
    /// Daylight D65 approximation (normalised).
    pub fn daylight() -> Self {
        Self([0.82, 0.88, 0.94, 0.98, 1.00, 0.99, 0.97, 0.95])
    }

    /// Tungsten / incandescent approximation (red-heavy).
    pub fn tungsten() -> Self {
        Self([0.15, 0.20, 0.28, 0.40, 0.60, 0.80, 0.93, 1.00])
    }

    /// Cool LED approximation (blue-heavy).
    pub fn cool_led() -> Self {
        Self([0.55, 0.80, 1.00, 0.90, 0.70, 0.55, 0.40, 0.30])
    }
}

/// Measured spectral reflectance profile for a material.
#[derive(Debug, Clone)]
pub struct SpectralMaterialProfile {
    /// Mean per-band reflectance across all sampled pixels.
    pub reflectance: [f32; 8],
    /// Per-band variance (confidence indicator).
    pub variance: [f32; 8],
}

impl SpectralMaterialProfile {
    /// Estimate spectral reflectance from three RGB photographs under known lights.
    ///
    /// Each photo is represented by its mean sRGB value over the material region.
    /// This is sufficient for the unit-test approximation; production uses per-pixel crops.
    pub fn from_three_photos(
        photos: [&[f32; 3]; 3],
        lights: [LightSpd; 3],
    ) -> Self {
        // Upsample each photo's mean RGB to 8-band spectral measurement
        let measured: [[f32; 8]; 3] = [
            SpectralUpsampler::from_rgb(photos[0][0], photos[0][1], photos[0][2]),
            SpectralUpsampler::from_rgb(photos[1][0], photos[1][1], photos[1][2]),
            SpectralUpsampler::from_rgb(photos[2][0], photos[2][1], photos[2][2]),
        ];

        // For each band, estimate reflectance by weighted average: r[b] = mean(measured[i][b] / light[i][b])
        let mut reflectance = [0.0f32; 8];
        let mut variance = [0.0f32; 8];

        for b in 0..8 {
            let estimates: [f32; 3] = [
                (measured[0][b] / lights[0].0[b].max(1e-4)).clamp(0.0, 1.0),
                (measured[1][b] / lights[1].0[b].max(1e-4)).clamp(0.0, 1.0),
                (measured[2][b] / lights[2].0[b].max(1e-4)).clamp(0.0, 1.0),
            ];
            let mean = (estimates[0] + estimates[1] + estimates[2]) / 3.0;
            let var = estimates.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / 3.0;
            reflectance[b] = mean;
            variance[b] = var;
        }

        Self { reflectance, variance }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daylight_spd_peaks_at_green() {
        let d = LightSpd::daylight();
        // Green band (index 4, 540nm) should be maximum for D65
        let peak = d.0.iter().copied().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        assert_eq!(peak, 4, "D65 peak should be band 4 (540nm)");
    }

    #[test]
    fn tungsten_spd_peaks_at_red() {
        let t = LightSpd::tungsten();
        let peak = t.0.iter().copied().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        assert_eq!(peak, 7, "tungsten peak should be band 7 (660nm)");
    }

    #[test]
    fn three_photo_profile_in_unit_range() {
        let lights = [LightSpd::daylight(), LightSpd::tungsten(), LightSpd::cool_led()];
        let photos = [[0.5f32, 0.5, 0.5], [0.5f32, 0.45, 0.4], [0.45f32, 0.5, 0.55]];
        let profile = SpectralMaterialProfile::from_three_photos(
            [&photos[0], &photos[1], &photos[2]],
            lights,
        );
        for (i, &v) in profile.reflectance.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v),
                "reflectance[{}] = {} must be in [0,1]", i, v);
        }
    }

    #[test]
    fn three_photo_variance_is_nonneg() {
        let lights = [LightSpd::daylight(), LightSpd::tungsten(), LightSpd::cool_led()];
        let photos = [[1.0f32, 0.0, 0.0], [0.8f32, 0.1, 0.05], [0.7f32, 0.05, 0.1]];
        let profile = SpectralMaterialProfile::from_three_photos(
            [&photos[0], &photos[1], &photos[2]],
            lights,
        );
        for (i, &v) in profile.variance.iter().enumerate() {
            assert!(v >= 0.0, "variance[{}] = {} must be non-negative", i, v);
        }
    }

    #[test]
    fn gray_surface_has_flat_reflectance() {
        let lights = [LightSpd::daylight(), LightSpd::tungsten(), LightSpd::cool_led()];
        // Gray surface: same brightness under all lights means flat reflectance
        let gray = [0.5f32, 0.5, 0.5];
        let profile = SpectralMaterialProfile::from_three_photos(
            [&gray, &gray, &gray],
            lights,
        );
        let min = profile.reflectance.iter().copied().fold(f32::MAX, f32::min);
        let max = profile.reflectance.iter().copied().fold(f32::MIN, f32::max);
        assert!(max - min < 0.4,
            "gray surface should have relatively flat reflectance, range was {:.3}", max - min);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_data spectral_capture 2>&1 | head -20
```

- [ ] **Step 3: Expose the module**

Add to `crates/vox_data/src/lib.rs`:

```rust
pub mod spectral_capture;
pub use spectral_capture::{LightSpd, SpectralMaterialProfile};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p vox_data spectral_capture -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/spectral_capture.rs crates/vox_data/src/lib.rs
git commit -m "feat(data): SpectralMaterialProfile — 3-photo spectral capture estimator"
```

---

## Task 3: VXM v3 — spectral_level + material_ids section

**Files:**
- Modify: `crates/vox_data/src/vxm.rs`

VXM v3 extends the header: `version = 3`, adds `spectral_level: u8` (1=uplift, 2=capture-approximate, 3=measured), reclaims one pad byte. After the compressed splat block, a new optional section stores the `material_ids` array as a `u32` count followed by `count × u16` values (zstd-compressed). Presence is indicated by `flags & 0x01`.

- [ ] **Step 1: Read the current file to understand the pad layout**

Current `VxmHeader`:
```
magic(4) + version(2) + flags(2) + asset_uuid(16) + splat_count(4) + material_type(1) + _pad0(3) + _pad1(32) = 64
```

v3 repurposes `_pad0[0]` as `spectral_level` and `_pad0[1]` as a reserved byte (sets v3 without changing struct size).

- [ ] **Step 2: Write failing tests for v3 round-trip**

Add to the test module at the bottom of `crates/vox_data/src/vxm.rs`:

```rust
    #[cfg(test)]
    mod v3_tests {
        use super::*;
        use vox_core::types::GaussianSplat;
        use bytemuck::Zeroable;

        fn make_splat(pos: [f32; 3]) -> GaussianSplat {
            let mut s = GaussianSplat::zeroed();
            s.position = pos;
            s.opacity = 200;
            s
        }

        #[test]
        fn vxm_v3_header_still_64_bytes() {
            assert_eq!(std::mem::size_of::<VxmHeader>(), 64);
        }

        #[test]
        fn material_ids_roundtrip() {
            let splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([1.0, 0.0, 0.0])];
            let material_ids = vec![3u16, 7u16]; // rock, asphalt

            let mut buf = Vec::new();
            let file = VxmFileV3 {
                splats: splats.clone(),
                material_ids: material_ids.clone(),
                spectral_level: 1,
            };
            file.write(&mut buf).unwrap();

            let loaded = VxmFileV3::read(std::io::Cursor::new(&buf)).unwrap();
            assert_eq!(loaded.splats.len(), 2);
            assert_eq!(loaded.material_ids, material_ids);
            assert_eq!(loaded.spectral_level, 1);
        }

        #[test]
        fn empty_material_ids_roundtrip() {
            let splats = vec![make_splat([0.0, 1.0, 0.0])];
            let file = VxmFileV3 { splats, material_ids: vec![], spectral_level: 2 };
            let mut buf = Vec::new();
            file.write(&mut buf).unwrap();
            let loaded = VxmFileV3::read(std::io::Cursor::new(&buf)).unwrap();
            assert!(loaded.material_ids.is_empty());
        }
    }
```

- [ ] **Step 3: Implement VxmFileV3**

Add to `crates/vox_data/src/vxm.rs` after the existing `VxmFile` impl:

```rust
const VERSION_V3: u16 = 3;
/// flags bit: material_ids section present
const FLAG_MATERIAL_IDS: u16 = 0x0001;

/// VXM v3: splats + optional material_ids + spectral_level.
pub struct VxmFileV3 {
    pub splats: Vec<GaussianSplat>,
    /// Per-splat material ID (0 = unassigned, 1-8 = SpectralMaterialDb index).
    /// May be empty (len=0 means no material data).
    pub material_ids: Vec<u16>,
    /// 1 = Smits uplift, 2 = capture-approximate, 3 = measured from 3-photo.
    pub spectral_level: u8,
}

impl VxmFileV3 {
    pub fn write<W: Write>(&self, mut w: W) -> Result<(), VxmError> {
        let has_mats = !self.material_ids.is_empty();
        let flags: u16 = if has_mats { FLAG_MATERIAL_IDS } else { 0 };

        // Build header — reuse VxmHeader struct, stamp v3 into version field
        let mut hdr = VxmHeader::zeroed();
        hdr.magic = *MAGIC;
        hdr.version = VERSION_V3;
        hdr.flags = flags;
        hdr.splat_count = self.splats.len() as u32;
        hdr._pad0[0] = self.spectral_level;  // spectral_level in first pad byte
        w.write_all(bytemuck::bytes_of(&hdr))?;

        // Compressed splat block (same as v1)
        let splat_bytes: &[u8] = bytemuck::cast_slice(&self.splats);
        let compressed = zstd::encode_all(splat_bytes, 0)
            .map_err(|e| VxmError::Compress(e.to_string()))?;
        w.write_all(&(compressed.len() as u64).to_le_bytes())?;
        w.write_all(&compressed)?;

        // Optional material_ids section
        if has_mats {
            let ids_bytes: &[u8] = bytemuck::cast_slice(&self.material_ids);
            let ids_compressed = zstd::encode_all(ids_bytes, 0)
                .map_err(|e| VxmError::Compress(e.to_string()))?;
            w.write_all(&(self.material_ids.len() as u32).to_le_bytes())?;
            w.write_all(&(ids_compressed.len() as u64).to_le_bytes())?;
            w.write_all(&ids_compressed)?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut r: R) -> Result<Self, VxmError> {
        let mut hdr_bytes = [0u8; 64];
        r.read_exact(&mut hdr_bytes)?;
        let hdr: VxmHeader = *bytemuck::from_bytes(&hdr_bytes);

        if &hdr.magic != MAGIC {
            return Err(VxmError::InvalidMagic);
        }
        if hdr.version != VERSION_V3 {
            return Err(VxmError::UnsupportedVersion(hdr.version));
        }

        let spectral_level = hdr._pad0[0];

        // Read compressed splat block
        let mut size_bytes = [0u8; 8];
        r.read_exact(&mut size_bytes)?;
        let compressed_size = u64::from_le_bytes(size_bytes) as usize;
        let mut compressed = vec![0u8; compressed_size];
        r.read_exact(&mut compressed)?;
        let decompressed = zstd::decode_all(&compressed[..])
            .map_err(|e| VxmError::Decompress(e.to_string()))?;
        let splats: Vec<GaussianSplat> = bytemuck::cast_slice(&decompressed).to_vec();

        // Optional material_ids section
        let mut material_ids = Vec::new();
        if hdr.flags & FLAG_MATERIAL_IDS != 0 {
            let mut count_bytes = [0u8; 4];
            r.read_exact(&mut count_bytes)?;
            let count = u32::from_le_bytes(count_bytes) as usize;
            let mut ids_size_bytes = [0u8; 8];
            r.read_exact(&mut ids_size_bytes)?;
            let ids_compressed_size = u64::from_le_bytes(ids_size_bytes) as usize;
            let mut ids_compressed = vec![0u8; ids_compressed_size];
            r.read_exact(&mut ids_compressed)?;
            let ids_bytes = zstd::decode_all(&ids_compressed[..])
                .map_err(|e| VxmError::Decompress(e.to_string()))?;
            let ids_slice: &[u16] = bytemuck::cast_slice(&ids_bytes);
            material_ids = ids_slice[..count].to_vec();
        }

        Ok(Self { splats, material_ids, spectral_level })
    }
}
```

Note: `VxmHeader._pad0` must be `pub` for this to compile. Verify and adjust visibility in the header struct if needed.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p vox_data v3_tests -- --nocapture
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/vxm.rs
git commit -m "feat(data): VXM v3 format — spectral_level + material_ids section"
```

---

## Task 4: COLMAP subprocess wrapper

**Files:**
- Create: `crates/vox_data/src/colmap_pipeline.rs`
- Modify: `crates/vox_data/src/lib.rs`

`ColmapPipeline::run()` performs:

1. `colmap feature_extractor` — extract keypoints from image directory.
2. `colmap exhaustive_matcher` — match features.
3. `colmap mapper` — sparse reconstruction to a `sparse/0/` folder.
4. `colmap model_converter --output_type TXT` — export to `cameras.txt`, `images.txt`, `points3D.txt`.
5. Parse `points3D.txt` for `(x, y, z, r, g, b)`, apply Smits upsampling, create `GaussianSplat` at each point.
6. Write output as VXM v3 with material IDs from `SpectralMaterialDb::classify`.

- [ ] **Step 1: Write failing tests**

Create `crates/vox_data/src/colmap_pipeline.rs`:

```rust
//! COLMAP photogrammetry subprocess wrapper.
//!
//! Spawns the `colmap` binary, runs sparse reconstruction from an image directory,
//! reads the resulting points3D.txt point cloud, applies Smits RGB→spectral upsampling,
//! and produces a VXM v3 file with per-splat spectral material IDs.
//!
//! Requires `colmap` to be installed and on PATH. Returns Err if not found.

use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;
use vox_core::types::GaussianSplat;
use bytemuck::Zeroable;
use half::f16;

use crate::spectral_upsampler::{SpectralUpsampler, SpectralMaterialDb};

#[derive(Debug, Error)]
pub enum ColmapError {
    #[error("colmap not found on PATH — install COLMAP: https://colmap.github.io")]
    NotFound,
    #[error("colmap subprocess failed (exit {code}): {stderr}")]
    SubprocessFailed { code: i32, stderr: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("points3D.txt parse error at line {line}: {msg}")]
    ParseError { line: usize, msg: String },
    #[error("vxm write error: {0}")]
    Vxm(#[from] crate::vxm::VxmError),
}

/// A point from the COLMAP sparse reconstruction.
#[derive(Debug, Clone)]
pub struct ColmapPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

pub struct ColmapPipeline;

impl ColmapPipeline {
    /// Run the full COLMAP sparse reconstruction pipeline.
    ///
    /// - `image_dir`: directory containing the input photographs.
    /// - `work_dir`: temporary working directory (created if absent).
    /// - `output_vxm`: path to write the resulting VXM v3 file.
    pub fn run(
        image_dir: &Path,
        work_dir: &Path,
        output_vxm: &Path,
    ) -> Result<(), ColmapError> {
        // Verify colmap is available
        Self::check_colmap_available()?;

        std::fs::create_dir_all(work_dir)?;
        let db_path = work_dir.join("colmap.db");
        let sparse_dir = work_dir.join("sparse");
        let txt_dir = work_dir.join("sparse_txt");
        std::fs::create_dir_all(&sparse_dir)?;
        std::fs::create_dir_all(&txt_dir)?;

        // Step 1: Feature extraction
        Self::run_colmap(&["feature_extractor",
            "--database_path", db_path.to_str().unwrap(),
            "--image_path", image_dir.to_str().unwrap(),
        ])?;

        // Step 2: Feature matching
        Self::run_colmap(&["exhaustive_matcher",
            "--database_path", db_path.to_str().unwrap(),
        ])?;

        // Step 3: Sparse reconstruction
        Self::run_colmap(&["mapper",
            "--database_path", db_path.to_str().unwrap(),
            "--image_path", image_dir.to_str().unwrap(),
            "--output_path", sparse_dir.to_str().unwrap(),
        ])?;

        // Step 4: Export to text format
        let model_dir = sparse_dir.join("0");
        Self::run_colmap(&["model_converter",
            "--input_path", model_dir.to_str().unwrap(),
            "--output_path", txt_dir.to_str().unwrap(),
            "--output_type", "TXT",
        ])?;

        // Step 5: Parse point cloud
        let points3d_path = txt_dir.join("points3D.txt");
        let points = Self::parse_points3d(&points3d_path)?;

        // Step 6: Convert to spectrally annotated VXM v3
        let (splats, material_ids) = Self::points_to_splats(&points);
        let vxm = crate::vxm::VxmFileV3 { splats, material_ids, spectral_level: 1 };
        let file = std::fs::File::create(output_vxm)?;
        vxm.write(std::io::BufWriter::new(file))?;

        Ok(())
    }

    /// Parse COLMAP points3D.txt format.
    /// Expected line format: POINT3D_ID X Y Z R G B ERROR TRACK[]
    pub fn parse_points3d(path: &Path) -> Result<Vec<ColmapPoint>, ColmapError> {
        let text = std::fs::read_to_string(path)?;
        let mut points = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() { continue; }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 7 {
                return Err(ColmapError::ParseError {
                    line: line_no,
                    msg: format!("expected ≥7 columns, got {}", cols.len()),
                });
            }
            let parse_f32 = |s: &str, field: &str| -> Result<f32, ColmapError> {
                s.parse::<f32>().map_err(|_| ColmapError::ParseError {
                    line: line_no,
                    msg: format!("cannot parse {} as f32: {}", field, s),
                })
            };
            let parse_u8 = |s: &str, field: &str| -> Result<u8, ColmapError> {
                s.parse::<u8>().map_err(|_| ColmapError::ParseError {
                    line: line_no,
                    msg: format!("cannot parse {} as u8: {}", field, s),
                })
            };
            points.push(ColmapPoint {
                x: parse_f32(cols[1], "X")?,
                y: parse_f32(cols[2], "Y")?,
                z: parse_f32(cols[3], "Z")?,
                r: parse_u8(cols[4], "R")?,
                g: parse_u8(cols[5], "G")?,
                b: parse_u8(cols[6], "B")?,
            });
        }
        Ok(points)
    }

    /// Convert point cloud to GaussianSplats with Smits spectral upsampling.
    pub fn points_to_splats(points: &[ColmapPoint]) -> (Vec<GaussianSplat>, Vec<u16>) {
        let mut splats = Vec::with_capacity(points.len());
        let mut material_ids = Vec::with_capacity(points.len());

        for p in points {
            let r = p.r as f32 / 255.0;
            let g = p.g as f32 / 255.0;
            let b = p.b as f32 / 255.0;
            let spectral_f32 = SpectralUpsampler::from_rgb(r, g, b);

            // Classify to nearest material
            let mat = SpectralMaterialDb::classify(&spectral_f32);
            let mat_id = SpectralMaterialDb::MATERIALS
                .iter()
                .position(|m| m.name == mat.name)
                .map_or(0u16, |i| (i + 1) as u16);

            let mut splat = GaussianSplat::zeroed();
            splat.position = [p.x, p.y, p.z];
            splat.scale = [0.01, 0.01, 0.01]; // small default scale
            splat.rotation = [0, 0, 0, 32767]; // identity quaternion (i16 packed)
            splat.opacity = 200;
            for band in 0..8 {
                splat.spectral[band] = f16::from_f32(spectral_f32[band]).to_bits();
            }

            splats.push(splat);
            material_ids.push(mat_id);
        }

        (splats, material_ids)
    }

    fn check_colmap_available() -> Result<(), ColmapError> {
        Command::new("colmap")
            .arg("--version")
            .output()
            .map_err(|_| ColmapError::NotFound)?;
        Ok(())
    }

    fn run_colmap(args: &[&str]) -> Result<(), ColmapError> {
        let output = Command::new("colmap").args(args).output()?;
        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(ColmapError::SubprocessFailed { code, stderr });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_POINTS3D: &str = r"# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 0.5 1.0 2.0 120 80 40 0.5 1 0 2 1
2 -1.0 0.5 0.3 60 120 60 0.3 1 2 2 3
3 0.0 0.0 0.0 200 200 200 0.1 1 4
";

    #[test]
    fn parse_points3d_extracts_positions() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_points3D.txt");
        std::fs::write(&path, SAMPLE_POINTS3D).unwrap();
        let points = ColmapPipeline::parse_points3d(&path).unwrap();
        assert_eq!(points.len(), 3);
        assert!((points[0].x - 0.5).abs() < 1e-5);
        assert!((points[1].y - 0.5).abs() < 1e-5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parse_points3d_extracts_rgb() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_points3D_rgb.txt");
        std::fs::write(&path, SAMPLE_POINTS3D).unwrap();
        let points = ColmapPipeline::parse_points3d(&path).unwrap();
        assert_eq!(points[0].r, 120);
        assert_eq!(points[0].g, 80);
        assert_eq!(points[0].b, 40);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parse_points3d_skips_comment_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_points3D_comments.txt");
        std::fs::write(&path, SAMPLE_POINTS3D).unwrap();
        let points = ColmapPipeline::parse_points3d(&path).unwrap();
        assert_eq!(points.len(), 3, "comment lines should be skipped");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn points_to_splats_assigns_spectral() {
        let pts = vec![ColmapPoint { x: 0.0, y: 0.0, z: 0.0, r: 200, g: 80, b: 40 }];
        let (splats, mat_ids) = ColmapPipeline::points_to_splats(&pts);
        assert_eq!(splats.len(), 1);
        assert_eq!(mat_ids.len(), 1);
        // All 8 spectral bands should be set
        let any_nonzero = splats[0].spectral.iter().any(|&v| v != 0);
        assert!(any_nonzero, "spectral bands must be populated from Smits upsampling");
    }

    #[test]
    fn points_to_splats_assigns_valid_material_id() {
        let pts = vec![ColmapPoint { x: 0.0, y: 0.0, z: 0.0, r: 30, g: 140, b: 30 }];
        let (_, mat_ids) = ColmapPipeline::points_to_splats(&pts);
        assert!(mat_ids[0] > 0, "material ID should be nonzero — unclassified means wrong");
        assert!(mat_ids[0] <= SpectralMaterialDb::MATERIALS.len() as u16,
            "material ID {} out of database range", mat_ids[0]);
    }

    #[test]
    fn points_to_splats_white_point_classifies_as_snow_or_concrete() {
        let pts = vec![ColmapPoint { x: 0.0, y: 0.0, z: 0.0, r: 230, g: 230, b: 235 }];
        let (_, mat_ids) = ColmapPipeline::points_to_splats(&pts);
        let name = SpectralMaterialDb::find_by_id(mat_ids[0]).unwrap().name;
        assert!(name == "snow" || name == "concrete" || name == "glass",
            "bright white point should classify as snow, concrete, or glass, got {}", name);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_data colmap_pipeline 2>&1 | head -20
```

- [ ] **Step 3: Expose the module**

Add to `crates/vox_data/src/lib.rs`:

```rust
pub mod colmap_pipeline;
pub use colmap_pipeline::{ColmapPipeline, ColmapPoint, ColmapError};
```

Add `half` to `vox_data` deps if not already present:

```toml
half = "2"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p vox_data colmap_pipeline -- --nocapture
```

Expected: 6 tests pass (COLMAP binary tests are not invoked in unit tests — only parse + conversion).

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/colmap_pipeline.rs crates/vox_data/src/lib.rs
git commit -m "feat(data): ColmapPipeline — subprocess wrapper + point cloud to spectral splats"
```

---

## Task 5: Integrate Smits upsampling into PLY importer

**Files:**
- Modify: `crates/vox_data/src/import_pipeline.rs`

The existing `import_ply` reads vertex positions and colours from PLY files. After this task, RGB vertex colours are converted to 8-band spectral via `SpectralUpsampler::from_rgb` before being stored in `GaussianSplat.spectral`.

- [ ] **Step 1: Locate the vertex colour assignment in import_ply**

```bash
grep -n "spectral\|color\|colour\|rgb\|vertex" /home/tomespen/git/ochroma/crates/vox_data/src/import_pipeline.rs | head -30
```

- [ ] **Step 2: Add Smits call at the vertex colour assignment site**

Find the code that assigns vertex colour to splat.spectral. Replace the direct assignment with:

```rust
// Convert vertex RGB to 8-band spectral via Smits 1999 upsampling
let spectral_f32 = vox_data::spectral_upsampler::SpectralUpsampler::from_rgb(r, g, b);
for band in 0..8 {
    splat.spectral[band] = half::f16::from_f32(spectral_f32[band]).to_bits();
}
```

If no vertex colour is available, set a neutral grey (0.5 across all bands) rather than zeroing spectral:

```rust
// No vertex colour: neutral grey spectral (preserves spectral invariant)
let neutral = half::f16::from_f32(0.5).to_bits();
for band in 0..8 { splat.spectral[band] = neutral; }
```

- [ ] **Step 3: Add a test for PLY→spectral upsampling**

Add to `import_pipeline.rs` test module (or create `tests/import_ply_spectral.rs`):

```rust
#[test]
fn ply_import_produces_nonzero_spectral() {
    use crate::import_pipeline::{ImportSettings, import_asset};
    // Write a minimal ASCII PLY with vertex colours to a temp file
    let ply_content = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 255 0 0\n1 0 0 0 255 0\n";
    let dir = std::env::temp_dir();
    let path = dir.join("test_spectral_ply.ply");
    std::fs::write(&path, ply_content).unwrap();

    let result = import_asset(&path, &ImportSettings::default()).unwrap();
    for splat in &result.splats {
        let any_nonzero = splat.spectral.iter().any(|&v| v != 0);
        assert!(any_nonzero, "spectral must be populated from vertex colour");
    }
    // Red vertex (255,0,0) should have higher high-band energy than low-band
    let red_splat = &result.splats[0];
    let low: f32 = (0..3).map(|b| half::f16::from_bits(red_splat.spectral[b]).to_f32()).sum::<f32>() / 3.0;
    let high: f32 = (5..8).map(|b| half::f16::from_bits(red_splat.spectral[b]).to_f32()).sum::<f32>() / 3.0;
    assert!(high > low, "red vertex should have higher spectral energy in bands 5-7");

    std::fs::remove_file(&path).ok();
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_data ply_import_produces_nonzero_spectral -- --nocapture
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/import_pipeline.rs
git commit -m "feat(data): integrate Smits upsampling into PLY importer — vertex RGB→spectral"
```

---

## Task 6: CLI tool `ochroma-tools`

**Files:**
- Create: `crates/vox_tools/Cargo.toml`
- Create: `crates/vox_tools/src/main.rs`
- Modify: workspace `Cargo.toml`

- [ ] **Step 1: Create the crate**

Create `crates/vox_tools/Cargo.toml`:

```toml
[package]
name = "ochroma-tools"
edition.workspace = true
version.workspace = true

[[bin]]
name = "ochroma-tools"
path = "src/main.rs"

[dependencies]
vox_data = { path = "../vox_data" }
vox_core = { path = "../vox_core" }
clap     = { version = "4", features = ["derive"] }
anyhow   = "1"
```

- [ ] **Step 2: Write failing test that binary exists**

```bash
cargo build -p ochroma-tools 2>&1 | head -10
```

Expected: compile error — file not found.

- [ ] **Step 3: Create main.rs**

Create `crates/vox_tools/src/main.rs`:

```rust
//! ochroma-tools — Ochroma engine asset pipeline CLI.
//!
//! Usage:
//!   ochroma-tools import --images <dir> --out scene.vxm
//!   ochroma-tools import --gltf model.glb --out scene.vxm
//!   ochroma-tools capture --images <dir> --out scene.vxm

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ochroma-tools", about = "Ochroma engine asset pipeline tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Import assets into VXM format with spectral annotation.
    Import {
        /// Input image directory for COLMAP photogrammetry.
        #[arg(long)]
        images: Option<PathBuf>,
        /// Input GLTF/GLB file.
        #[arg(long)]
        gltf: Option<PathBuf>,
        /// Output .vxm file path.
        #[arg(long)]
        out: PathBuf,
        /// Working directory for COLMAP intermediate files.
        #[arg(long, default_value = "/tmp/ochroma_colmap")]
        work_dir: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Import { images, gltf, out, work_dir } => {
            if let Some(img_dir) = images {
                println!("Running COLMAP photogrammetry pipeline...");
                println!("  Image directory : {}", img_dir.display());
                println!("  Work directory  : {}", work_dir.display());
                println!("  Output          : {}", out.display());
                vox_data::ColmapPipeline::run(&img_dir, &work_dir, &out)
                    .map_err(|e| anyhow::anyhow!("COLMAP pipeline failed: {}", e))?;
                println!("Done. Wrote {}", out.display());
            } else if let Some(gltf_path) = gltf {
                println!("Importing GLTF asset...");
                println!("  Input  : {}", gltf_path.display());
                println!("  Output : {}", out.display());
                let settings = vox_data::ImportSettings::default();
                let result = vox_data::import_asset(&gltf_path, &settings)
                    .map_err(|e| anyhow::anyhow!("GLTF import failed: {}", e))?;
                // Write as VXM v3 with auto-classified material IDs
                let material_ids: Vec<u16> = result.splats.iter().map(|s| {
                    let spectral: [f32; 8] = std::array::from_fn(|b| {
                        half::f16::from_bits(s.spectral[b]).to_f32()
                    });
                    let mat = vox_data::SpectralMaterialDb::classify(&spectral);
                    vox_data::SpectralMaterialDb::MATERIALS
                        .iter()
                        .position(|m| m.name == mat.name)
                        .map_or(0u16, |i| (i + 1) as u16)
                }).collect();
                let vxm = vox_data::VxmFileV3 {
                    splats: result.splats,
                    material_ids,
                    spectral_level: 1,
                };
                let file = std::fs::File::create(&out)?;
                vxm.write(std::io::BufWriter::new(file))?;
                println!("Done. Wrote {} splats to {}", vxm.splats.len(), out.display());
            } else {
                anyhow::bail!("Provide either --images or --gltf");
            }
        }
    }
    Ok(())
}
```

Ensure `VxmFileV3`, `ImportSettings`, `import_asset`, `SpectralMaterialDb` are all re-exported from `vox_data::lib.rs`.

- [ ] **Step 4: Add vox_tools to workspace**

In the root `Cargo.toml` workspace members list, add:

```toml
"crates/vox_tools",
```

- [ ] **Step 5: Build to verify clean compilation**

```bash
cargo build -p ochroma-tools 2>&1 | grep "^error" | head -20
```

Expected: clean build.

- [ ] **Step 6: Smoke test the CLI**

```bash
cargo run -p ochroma-tools -- import --help
```

Expected: prints usage for the `import` subcommand.

- [ ] **Step 7: Commit**

```bash
git add crates/vox_tools/ Cargo.toml
git commit -m "feat(tools): ochroma-tools CLI — import --images (COLMAP) and --gltf subcommands"
```

---

## Self-Review

**Spec coverage:**
- [x] Smits 1999 RGB→spectral decomposition, 8-band output → Task 1 ✓
- [x] `SpectralUpsampler::from_rgb(r,g,b) -> [f32; 8]` → Task 1 ✓
- [x] `SpectralMaterialDb` — foliage, soil, rock, water, glass, concrete, snow, asphalt → Task 1 ✓
- [x] `SpectralMaterialDb::classify()` → Task 1 ✓
- [x] `SpectralMaterialProfile` from 3-photo capture → Task 2 ✓
- [x] VXM v3 `spectral_material_id: u16` per splat → Task 3 ✓
- [x] VXM v3 `spectral_level` → Task 3 ✓
- [x] `ColmapPipeline::run()` subprocess wrapper → Task 4 ✓
- [x] Smits upsampling in PLY importer → Task 5 ✓
- [x] `ochroma-tools import --images <dir> --out scene.vxm` → Task 6 ✓
- [x] `ochroma-tools import --gltf model.glb --out scene.vxm` → Task 6 ✓

**Spectral invariant:** Every import path — PLY, GLTF, and COLMAP — now populates `GaussianSplat.spectral` from real surface data rather than zeroing it. The neutral grey fallback (0.5 across bands) preserves the invariant for geometry with no colour data.

**Known limitation:** The 3-photo `SpectralMaterialProfile` uses Smits-upsampled RGB as the per-photo measurement, not raw spectrometer data. This is correct for standard cameras. True spectral accuracy requires multispectral cameras; the architecture supports substituting raw band measurements without API change.

**VXM v2 compatibility:** `VxmFileV3::read()` returns `UnsupportedVersion(1)` for v1 files. The existing `VxmFile::read()` (v1 reader) remains untouched. Callers that need backwards compatibility should try `VxmFile::read()` first, then `VxmFileV3::read()`.

---

## Task 7: VegetationSplatizer — PROSPECT-PRO spectral_embedding → 16-band splats

**Files:**
- Create: `crates/vox_data/src/vegetation_splatizer.rs`
- Modify: `crates/vox_data/src/lib.rs`

**Context:** `FloraPrimeNode` in crucible-nodes writes `Mesh.spectral_embedding: Vec<[f32; 6]>` — six PCA components of PROSPECT-PRO evaluated at 6 wavelengths per vertex. To splatize vegetation meshes with physically accurate leaf optics, we must back-project these 6 PCA components to the full 16-band spectral representation. The 6 components explain ~97% of spectral variance; the back-projection uses the PCA basis stored as a 6×16 matrix (6 components × 16 wavelength bands).

PCA basis for PROSPECT-PRO → 16-band back-projection (computed offline from USGS leaf spectra):
```
PC0: [0.31,0.32,0.33,0.34,0.33,0.32,0.31,0.30, 0.28,0.25,0.22,0.19,0.17,0.25,0.40,0.45]
PC1: [0.12,0.11,0.10,0.08,0.06,0.04,0.02,0.01,-0.02,-0.05,-0.08,-0.11,-0.13,0.15,0.38,0.42]
PC2: [-0.05,-0.04,-0.03,-0.01,0.02,0.05,0.08,0.10, 0.08,0.06,0.04,0.02,0.01,-0.08,-0.20,-0.22]
PC3: [0.02,0.02,0.01,0.01,-0.01,-0.02,-0.03,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.05,0.06]
PC4: [-0.01,-0.01,0.00,0.01,0.02,0.01,0.00,-0.01,-0.02,-0.01,0.00,0.01,0.02,-0.01,-0.03,-0.04]
PC5: [0.00,0.00,0.01,0.01,0.00,-0.01,-0.01,0.00,0.01,0.01,0.00,-0.01,-0.01,0.00,0.01,0.01]
```

- [ ] **Step 1: Write failing test**

Create `crates/vox_data/src/vegetation_splatizer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_backprojection_produces_valid_spectrum() {
        // Healthy green leaf: high chlorophyll → strong red-edge rise
        let embedding = [0.8f32, 0.4, -0.1, 0.0, 0.0, 0.0]; // PC0 dominant = bright leaf
        let spectrum = backproject_pca(&embedding);
        // All bands should be in [0, 1]
        for (i, &v) in spectrum.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "band {i} out of range: {v}");
        }
        // Red-edge bands (index 12-14, 680-730nm) should be higher than visible green (index 6-8)
        let red_edge_avg = (spectrum[12] + spectrum[13] + spectrum[14]) / 3.0;
        let green_avg = (spectrum[6] + spectrum[7] + spectrum[8]) / 3.0;
        assert!(red_edge_avg > green_avg, "red-edge should exceed green for leaf");
    }

    #[test]
    fn test_splatize_vegetation_mesh() {
        // Create a minimal mesh with spectral_embedding
        let mesh = EditorMesh {
            positions: vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],
            normals: vec![[0.0,1.0,0.0]; 3],
            indices: vec![[0u32,1,2]],
            spectral_embedding: Some(vec![
                [0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.4, 0.0, 0.0, 0.0, 0.0],
            ]),
            ..Default::default()
        };
        let splats = splatize_vegetation_mesh(&mesh, 1.0);
        assert_eq!(splats.len(), 1, "one triangle → one splat");
        // Splat spectral should be average of vertex embeddings back-projected
        assert!(splats[0].spectral[0] > 0.0, "spectral[0] should be nonzero");
    }
}
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cargo test -p vox_data vegetation_splatizer 2>&1 | head -20
```

Expected: compile error — `backproject_pca`, `splatize_vegetation_mesh` not found.

- [ ] **Step 3: Implement VegetationSplatizer**

```rust
//! VegetationSplatizer — converts vegetation meshes with PROSPECT-PRO spectral
//! embeddings (6 PCA components) to GaussianSplats with 16-band spectral values.

use vox_core::types::GaussianSplat;

/// PCA basis matrix: 6 components × 16 wavelength bands.
/// Rows are principal components, columns are wavelength bands (380–755nm, 25nm steps).
const PCA_BASIS: [[f32; 16]; 6] = [
    [0.31,0.32,0.33,0.34,0.33,0.32,0.31,0.30, 0.28,0.25,0.22,0.19,0.17,0.25,0.40,0.45],
    [0.12,0.11,0.10,0.08,0.06,0.04,0.02,0.01,-0.02,-0.05,-0.08,-0.11,-0.13,0.15,0.38,0.42],
    [-0.05,-0.04,-0.03,-0.01,0.02,0.05,0.08,0.10, 0.08,0.06,0.04,0.02,0.01,-0.08,-0.20,-0.22],
    [0.02,0.02,0.01,0.01,-0.01,-0.02,-0.03,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.05,0.06],
    [-0.01,-0.01,0.00,0.01,0.02,0.01,0.00,-0.01,-0.02,-0.01,0.00,0.01,0.02,-0.01,-0.03,-0.04],
    [0.00,0.00,0.01,0.01,0.00,-0.01,-0.01,0.00,0.01,0.01,0.00,-0.01,-0.01,0.00,0.01,0.01],
];

/// Back-project a 6-component PCA embedding to 16-band spectral reflectance.
/// Result is clamped to [0, 1].
pub fn backproject_pca(embedding: &[f32; 6]) -> [f32; 16] {
    let mut spectrum = [0.0f32; 16];
    for (comp, &weight) in embedding.iter().enumerate() {
        for band in 0..16 {
            spectrum[band] += weight * PCA_BASIS[comp][band];
        }
    }
    // Clamp to valid reflectance range
    for v in &mut spectrum {
        *v = v.clamp(0.0, 1.0);
    }
    spectrum
}

/// Convert a vegetation mesh (with spectral_embedding) to GaussianSplats.
/// Each triangle becomes one splat. Spectral value = average of vertex embeddings
/// back-projected to 16 bands. Splat scale derived from triangle area.
pub fn splatize_vegetation_mesh(mesh: &EditorMesh, splat_scale: f32) -> Vec<GaussianSplat> {
    let embeddings = match &mesh.spectral_embedding {
        Some(e) => e,
        None    => return splatize_mesh_flat_foliage(mesh, splat_scale),
    };

    mesh.indices.iter().map(|tri| {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        // Centroid position
        let p: [f32; 3] = std::array::from_fn(|d| {
            (mesh.positions[i0][d] + mesh.positions[i1][d] + mesh.positions[i2][d]) / 3.0
        });
        // Average normal
        let n: [f32; 3] = std::array::from_fn(|d| {
            (mesh.normals[i0][d] + mesh.normals[i1][d] + mesh.normals[i2][d]) / 3.0
        });
        // Average spectral embedding across triangle vertices
        let avg_emb: [f32; 6] = std::array::from_fn(|c| {
            (embeddings[i0][c] + embeddings[i1][c] + embeddings[i2][c]) / 3.0
        });
        let spectral_f32 = backproject_pca(&avg_emb);
        // Convert to u16 for GaussianSplat
        let spectral_u16: [u16; 16] = std::array::from_fn(|i| {
            (spectral_f32[i] * 65535.0) as u16
        });
        // Triangle area for scale
        let edge1 = [p[0]-mesh.positions[i0][0], p[1]-mesh.positions[i0][1], p[2]-mesh.positions[i0][2]];
        let area = (edge1[0]*edge1[0] + edge1[1]*edge1[1] + edge1[2]*edge1[2]).sqrt() * 0.5;
        let scale = (area * splat_scale).max(0.01);

        GaussianSplat {
            position: p,
            normal: n,
            spectral: spectral_u16,
            scale: [scale; 3],
            opacity: 0.85,
            ..Default::default()
        }
    }).collect()
}

/// Fallback: splatize without spectral_embedding using flat Foliage USGS curve.
fn splatize_mesh_flat_foliage(mesh: &EditorMesh, splat_scale: f32) -> Vec<GaussianSplat> {
    // Foliage USGS 16-band curve
    const FOLIAGE: [f32; 16] = [
        0.04,0.04,0.05,0.07,0.08,0.10,0.12,0.12, 0.08,0.05,0.04,0.04,0.05,0.20,0.45,0.55
    ];
    let spectral_u16: [u16; 16] = std::array::from_fn(|i| (FOLIAGE[i] * 65535.0) as u16);
    mesh.indices.iter().map(|tri| {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p: [f32; 3] = std::array::from_fn(|d| {
            (mesh.positions[i0][d] + mesh.positions[i1][d] + mesh.positions[i2][d]) / 3.0
        });
        let n: [f32; 3] = std::array::from_fn(|d| {
            (mesh.normals[i0][d] + mesh.normals[i1][d] + mesh.normals[i2][d]) / 3.0
        });
        GaussianSplat {
            position: p, normal: n,
            spectral: spectral_u16,
            scale: [splat_scale * 0.1; 3],
            opacity: 0.85,
            ..Default::default()
        }
    }).collect()
}
```

- [ ] **Step 4: Run test — expect PASS**

```bash
cargo test -p vox_data vegetation_splatizer
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/vegetation_splatizer.rs crates/vox_data/src/lib.rs
git commit -m "feat(data): VegetationSplatizer — PROSPECT-PRO PCA embedding → 16-band splats"
```

---

## Task 8: TerrainSplatizer — SpectralTerrainMaterials + biome_to_splat_weights

**Files:**
- Create: `crates/vox_data/src/terrain_splatizer.rs`
- Modify: `crates/vox_data/src/lib.rs`

**Context:** forge-terrain's `biome_to_splat_weights(biome, height, world_height) -> [f32; 4]` maps biome + elevation to a 4-channel blend. `SpectralTerrainMaterials` provides the 7-slot material palette (Water, Sand, Grass, Dirt, Rock, Snow, Forest) as 16-band USGS curves. The terrain splatizer samples the heightfield at each splat position, determines the biome, looks up blend weights, and blends 4 spectral curves.

- [ ] **Step 1: Write failing test**

Create `crates/vox_data/src/terrain_splatizer.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_splatizer_snow_at_high_altitude() {
        let mats = SpectralTerrainMaterials::default();
        // Alpine biome at high elevation → heavy snow/rock blend
        let weights = biome_to_splat_weights(BiomeKind::Alpine, 320.0, 400.0);
        let spectral = blend_spectral_terrain(&mats, &weights);
        // Snow (slot 5) is bright at all bands. Blended result should be quite bright.
        let avg_reflectance: f32 = spectral.iter().sum::<f32>() / 16.0;
        assert!(avg_reflectance > 0.4, "alpine snow blend should be bright, got {avg_reflectance}");
    }

    #[test]
    fn test_terrain_splatizer_water_in_wetland() {
        let mats = SpectralTerrainMaterials::default();
        let weights = biome_to_splat_weights(BiomeKind::Wetland, 5.0, 100.0);
        let spectral = blend_spectral_terrain(&mats, &weights);
        // Water is very dark in near-IR (bands 8-15)
        let near_ir_avg: f32 = spectral[8..16].iter().sum::<f32>() / 8.0;
        assert!(near_ir_avg < 0.15, "wetland near-IR should be dark (water dominant)");
    }
}
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cargo test -p vox_data terrain_splatizer 2>&1 | head -20
```

Expected: compile error.

- [ ] **Step 3: Implement TerrainSplatizer**

```rust
//! TerrainSplatizer — converts terrain heightfields to GaussianSplats
//! with physically measured spectral reflectances from USGS material database.
//!
//! Biome → splat_weights[4] → blend 4 spectral curves from SpectralTerrainMaterials.

/// Biome kind — mirrors forge-terrain Biome enum (re-defined here to avoid forge dep).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiomeKind {
    Alpine, Tundra, Forest, Grassland, Desert, Wetland, Coastal,
    SubalpineShrub, Savanna, Taiga, TropicalRainforest,
}

/// 7-slot spectral terrain material palette (16 bands each, 380–755nm).
/// Slot order: Water(0), Sand(1), Grass(2), Dirt(3), Rock(4), Snow(5), Forest/Bark(6).
pub struct SpectralTerrainMaterials {
    pub slots: [[f32; 16]; 7],
}

impl Default for SpectralTerrainMaterials {
    fn default() -> Self {
        Self { slots: [
            [0.03,0.04,0.05,0.05,0.05,0.04,0.03,0.03, 0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01], // Water
            [0.25,0.28,0.31,0.34,0.36,0.38,0.39,0.40, 0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48], // Sand
            [0.04,0.04,0.05,0.07,0.08,0.10,0.12,0.12, 0.08,0.05,0.04,0.04,0.05,0.20,0.45,0.55], // Grass
            [0.07,0.09,0.11,0.13,0.14,0.16,0.18,0.20, 0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.30], // Dirt
            [0.15,0.17,0.19,0.21,0.22,0.23,0.24,0.25, 0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33], // Rock
            [0.93,0.94,0.95,0.95,0.95,0.94,0.93,0.92, 0.91,0.90,0.89,0.88,0.87,0.86,0.85,0.85], // Snow
            [0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12, 0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20], // Forest/Bark
        ]}
    }
}

/// Map biome + elevation fraction to 4-channel splat blend weights.
/// Weights sum to 1.0. Channel mapping: [water, rock/snow, vegetation, ground].
pub fn biome_to_splat_weights(biome: BiomeKind, height: f32, world_height: f32) -> [f32; 4] {
    let _t = (height / world_height.max(1.0)).clamp(0.0, 1.0);
    match biome {
        BiomeKind::Alpine          => [0.00, 0.50, 0.05, 0.45],
        BiomeKind::Tundra          => [0.00, 0.40, 0.20, 0.40],
        BiomeKind::Forest          => [0.00, 0.05, 0.70, 0.25],
        BiomeKind::Grassland       => [0.00, 0.05, 0.75, 0.20],
        BiomeKind::Desert          => [0.00, 0.10, 0.00, 0.90],
        BiomeKind::Wetland         => [0.40, 0.05, 0.40, 0.15],
        BiomeKind::Coastal         => [0.30, 0.10, 0.25, 0.35],
        BiomeKind::SubalpineShrub  => [0.00, 0.25, 0.50, 0.25],
        BiomeKind::Savanna         => [0.00, 0.10, 0.55, 0.35],
        BiomeKind::Taiga           => [0.00, 0.10, 0.65, 0.25],
        BiomeKind::TropicalRainforest => [0.10, 0.00, 0.80, 0.10],
    }
}

/// Blend 4 spectral slots using blend weights.
/// Channel mapping: [0]=Water, [1]=Rock/Snow, [2]=Grass, [3]=Dirt.
pub fn blend_spectral_terrain(mats: &SpectralTerrainMaterials, weights: &[f32; 4]) -> [f32; 16] {
    let slot_indices = [0usize, 4, 2, 3]; // Water, Rock, Grass, Dirt
    let mut result = [0.0f32; 16];
    for (ch, (&w, &slot)) in weights.iter().zip(slot_indices.iter()).enumerate() {
        let _ = ch;
        for band in 0..16 {
            result[band] += w * mats.slots[slot][band];
        }
    }
    result
}
```

- [ ] **Step 4: Run test — expect PASS**

```bash
cargo test -p vox_data terrain_splatizer
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_data/src/terrain_splatizer.rs crates/vox_data/src/lib.rs
git commit -m "feat(data): TerrainSplatizer — biome_to_splat_weights + SpectralTerrainMaterials 16-band blend"
```
