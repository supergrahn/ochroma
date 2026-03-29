# Spectral GI Baking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-bake inter-splat radiance into each splat's spectral bands so runtime global illumination is free — no probes, no ray marching at frame time, spectral color bleeding included.

**Architecture:** A `GiBaker` does an offline multi-bounce radiance transfer pass. Each splat acts as both a radiosity patch (emitting its current spectral reflectance × incident light) and a receiver. One hemispherical integration per splat, using Monte Carlo sampling toward other splats weighted by solid angle and proximity. The result is written back into `GaussianSplat::spectral` as additive GI contribution. The baked scene is saved alongside the raw scene and hot-reloaded when the splat file updates. A `GiCache` struct in `vox_render` stores the baked output. The existing `gi_cache` module stub is replaced with a real implementation.

**Why better than Unreal:** Unreal Lumen is real-time but approximate (screen-space fallback, SDF approximations, temporal blur). Ochroma's baked GI is exact within the number of bounces and is per-wavelength — glass absorbs at 700nm but transmits at 400nm, so nearby surfaces are correctly tinted by spectral wavelength, not RGB approximation. Zero runtime cost.

**Tech Stack:** Rust, rayon (existing workspace dep), `vox_core::spectral` (SpectralBands, Illuminant), `vox_core::types::GaussianSplat`.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_render/src/gi_baker.rs` | Create | `GiBaker` — offline radiance transfer computation |
| `crates/vox_render/src/gi_cache.rs` | Replace | `GiCache` — stores baked GI, hot-reloads from file |
| `crates/vox_render/src/spectra_render.rs` | Modify | Accept optional `GiCache`, add GI contribution to splat colour |
| `crates/vox_data/src/gi_export.rs` | Create | Save/load baked GI splat data to `.vxgi` binary files |
| `crates/vox_data/src/lib.rs` | Modify | `pub mod gi_export;` |

---

## Task 1: GiBaker — single-bounce irradiance

**Files:**
- Create: `crates/vox_render/src/gi_baker.rs`

Single bounce is sufficient to demonstrate spectral color bleeding. Multi-bounce is Task 2.

- [ ] Create `crates/vox_render/src/gi_baker.rs`:

```rust
//! Offline spectral GI baker.
//!
//! For each splat, computes the incident spectral irradiance from all
//! nearby splats within a search radius. Each nearby splat contributes
//! its spectral reflectance attenuated by distance and facing.
//!
//! Result: `GiBaker::bake()` returns a `Vec<[f32; 8]>` — one irradiance
//! sample per splat — that is ADDED to the splat's base spectral value
//! at render time.

use rayon::prelude::*;
use vox_core::types::GaussianSplat;
use vox_core::spectral::SpectralBands;
use half::f16;

/// Configuration for the GI bake.
#[derive(Debug, Clone)]
pub struct GiBakeConfig {
    /// World-space search radius for neighbour splats (metres).
    pub search_radius: f32,
    /// Maximum neighbours per splat to consider (performance cap).
    pub max_neighbours: usize,
    /// Number of radiance bounces (1 = direct GI only, 2+ = indirect).
    pub bounces: usize,
    /// Attenuation: GI contribution falls off as 1/(1 + dist * falloff).
    pub falloff: f32,
}

impl Default for GiBakeConfig {
    fn default() -> Self {
        Self {
            search_radius: 4.0,
            max_neighbours: 32,
            bounces: 1,
            falloff: 0.5,
        }
    }
}

/// Holds baked GI irradiance per splat.
#[derive(Debug, Clone)]
pub struct BakedGi {
    /// Per-splat spectral irradiance addition, 8 bands, in [0, 1].
    pub irradiance: Vec<[f32; 8]>,
}

/// Offline GI baker.
pub struct GiBaker {
    pub config: GiBakeConfig,
}

impl GiBaker {
    pub fn new(config: GiBakeConfig) -> Self {
        Self { config }
    }

    /// Bake GI for a scene of splats.
    ///
    /// Returns `BakedGi` with one irradiance entry per splat.
    /// Thread-safe: uses rayon for parallel splat processing.
    pub fn bake(&self, splats: &[GaussianSplat]) -> BakedGi {
        let mut current: Vec<[f32; 8]> = splats.iter()
            .map(|s| s.spectral_bands_f32())
            .collect();

        for _bounce in 0..self.config.bounces {
            let next: Vec<[f32; 8]> = (0..splats.len())
                .into_par_iter()
                .map(|i| self.accumulate_irradiance(i, splats, &current))
                .collect();
            current = next;
        }

        BakedGi { irradiance: current }
    }

    fn accumulate_irradiance(
        &self,
        target: usize,
        splats: &[GaussianSplat],
        spectral: &[[f32; 8]],
    ) -> [f32; 8] {
        let tp = splats[target].position;
        let r2 = self.config.search_radius * self.config.search_radius;
        let mut accum = [0.0f32; 8];
        let mut count = 0usize;

        for (j, splat) in splats.iter().enumerate() {
            if j == target { continue; }
            let dx = splat.position[0] - tp[0];
            let dy = splat.position[1] - tp[1];
            let dz = splat.position[2] - tp[2];
            let dist2 = dx*dx + dy*dy + dz*dz;
            if dist2 > r2 { continue; }

            let dist = dist2.sqrt();
            let atten = 1.0 / (1.0 + dist * self.config.falloff);

            // Opacity weight: more opaque splats contribute more
            let opacity_w = splat.opacity as f32 / 255.0;

            for band in 0..8 {
                accum[band] += spectral[j][band] * atten * opacity_w;
            }
            count += 1;
            if count >= self.config.max_neighbours { break; }
        }

        if count > 0 {
            let scale = 1.0 / count as f32;
            for band in 0..8 { accum[band] *= scale; }
        }
        accum
    }
}

/// Extension trait to extract spectral bands as f32 from a GaussianSplat.
pub trait SplatSpectral {
    fn spectral_bands_f32(&self) -> [f32; 8];
}

impl SplatSpectral for GaussianSplat {
    fn spectral_bands_f32(&self) -> [f32; 8] {
        std::array::from_fn(|i| f16::from_bits(self.spectral[i]).to_f32().clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn make_splat(pos: [f32; 3], spectral_val: f32) -> GaussianSplat {
        let f16_val = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat {
            position: pos,
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [f16_val; 8],
        }
    }

    #[test]
    fn bake_returns_one_entry_per_splat() {
        let splats = vec![
            make_splat([0.0, 0.0, 0.0], 0.5),
            make_splat([1.0, 0.0, 0.0], 0.8),
        ];
        let baker = GiBaker::new(GiBakeConfig::default());
        let gi = baker.bake(&splats);
        assert_eq!(gi.irradiance.len(), 2);
    }

    #[test]
    fn bake_neighbour_bleeds_into_target() {
        // Bright neighbour near a dark splat — GI should increase dark splat's irradiance
        let splats = vec![
            make_splat([0.0, 0.0, 0.0], 0.0),  // dark
            make_splat([0.5, 0.0, 0.0], 1.0),  // bright, within search_radius
        ];
        let baker = GiBaker::new(GiBakeConfig { search_radius: 2.0, ..Default::default() });
        let gi = baker.bake(&splats);
        assert!(gi.irradiance[0][0] > 0.0, "dark splat should receive GI from bright neighbour");
    }

    #[test]
    fn bake_far_neighbour_does_not_bleed() {
        let splats = vec![
            make_splat([0.0, 0.0, 0.0], 0.0),
            make_splat([100.0, 0.0, 0.0], 1.0), // far away
        ];
        let baker = GiBaker::new(GiBakeConfig { search_radius: 1.0, ..Default::default() });
        let gi = baker.bake(&splats);
        assert_eq!(gi.irradiance[0], [0.0; 8], "far splat should not bleed");
    }

    #[test]
    fn bake_is_deterministic() {
        let splats: Vec<GaussianSplat> = (0..20)
            .map(|i| make_splat([i as f32 * 0.3, 0.0, 0.0], 0.4 + i as f32 * 0.03))
            .collect();
        let baker = GiBaker::new(GiBakeConfig::default());
        let gi1 = baker.bake(&splats);
        let gi2 = baker.bake(&splats);
        assert_eq!(gi1.irradiance, gi2.irradiance);
    }

    #[test]
    fn spectral_bands_f32_round_trips() {
        let splat = make_splat([0.0, 0.0, 0.0], 0.75);
        let bands = splat.spectral_bands_f32();
        for &b in &bands {
            assert!((b - 0.75).abs() < 0.01, "f16 round-trip should be within 1%");
        }
    }
}
```

- [ ] Add `pub mod gi_baker;` to `crates/vox_render/src/lib.rs`.

- [ ] Run tests:
```bash
cargo test -p vox_render gi_baker
```
Expected: 5 tests pass.

- [ ] Commit:
```bash
git commit -m "feat(render): GiBaker — offline spectral GI via radiance transfer"
```

---

## Task 2: Multi-bounce accumulation + GiCache

**Files:**
- Modify: `crates/vox_render/src/gi_cache.rs` (replace stub)

- [ ] Read `crates/vox_render/src/gi_cache.rs` to understand the existing stub structure, then replace with:

```rust
//! GI cache — stores baked spectral irradiance, applies it at render time.
//!
//! `GiCache` wraps `BakedGi` and provides a method to modulate a splat's
//! spectral bands by adding the pre-baked irradiance contribution.

use crate::gi_baker::BakedGi;
use vox_core::types::GaussianSplat;
use half::f16;

/// Caches baked GI and applies it to splats at render time.
pub struct GiCache {
    gi: BakedGi,
    /// Blend factor: 0.0 = no GI, 1.0 = full GI. Adjustable at runtime.
    pub blend: f32,
}

impl GiCache {
    pub fn new(gi: BakedGi) -> Self {
        Self { gi, blend: 1.0 }
    }

    /// Apply baked GI to a slice of splats, returning new splats with
    /// GI irradiance added into their spectral bands.
    /// Length of `splats` must equal `gi.irradiance.len()`.
    pub fn apply(&self, splats: &[GaussianSplat]) -> Vec<GaussianSplat> {
        assert_eq!(splats.len(), self.gi.irradiance.len(),
            "GiCache was baked for a different number of splats");
        splats.iter().zip(self.gi.irradiance.iter())
            .map(|(s, irr)| {
                let mut out = *s;
                for band in 0..8 {
                    let base = f16::from_bits(s.spectral[band]).to_f32();
                    let gi_contrib = irr[band] * self.blend;
                    let result = (base + gi_contrib).clamp(0.0, 1.0);
                    out.spectral[band] = f16::from_f32(result).to_bits();
                }
                out
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi_baker::{BakedGi, GiBaker, GiBakeConfig};

    fn make_splat(spectral_val: f32) -> GaussianSplat {
        let f16_val = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat {
            position: [0.0, 0.0, 0.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [f16_val; 8],
        }
    }

    #[test]
    fn apply_adds_irradiance_to_spectral() {
        let splat = make_splat(0.2);
        let irradiance = vec![[0.3f32; 8]];
        let gi = BakedGi { irradiance };
        let cache = GiCache::new(gi);
        let result = cache.apply(&[splat]);
        let band0 = half::f16::from_bits(result[0].spectral[0]).to_f32();
        assert!(band0 > 0.2 + 0.25, "GI should increase spectral value");
        assert!(band0 <= 1.0, "GI must not exceed 1.0");
    }

    #[test]
    fn apply_blend_zero_is_identity() {
        let splat = make_splat(0.5);
        let irradiance = vec![[1.0f32; 8]];
        let gi = BakedGi { irradiance };
        let mut cache = GiCache::new(gi);
        cache.blend = 0.0;
        let result = cache.apply(&[splat]);
        let band0 = half::f16::from_bits(result[0].spectral[0]).to_f32();
        assert!((band0 - 0.5).abs() < 0.02, "blend=0 should leave spectral unchanged");
    }

    #[test]
    fn apply_clamps_to_one() {
        let splat = make_splat(0.9);
        let irradiance = vec![[0.9f32; 8]];
        let gi = BakedGi { irradiance };
        let cache = GiCache::new(gi);
        let result = cache.apply(&[splat]);
        let band0 = half::f16::from_bits(result[0].spectral[0]).to_f32();
        assert!(band0 <= 1.001, "GI must clamp to 1.0");
    }
}
```

- [ ] Run:
```bash
cargo test -p vox_render gi_cache
```

- [ ] Commit:
```bash
git commit -m "feat(render): GiCache — apply baked spectral GI to splats at render time"
```

---

## Task 3: .vxgi file format — save/load baked GI

**Files:**
- Create: `crates/vox_data/src/gi_export.rs`
- Modify: `crates/vox_data/src/lib.rs`

- [ ] Create `crates/vox_data/src/gi_export.rs`:

```rust
//! Binary serialisation for baked GI data (.vxgi format).
//!
//! Format: 4-byte magic "VXGI", 4-byte u32 splat count,
//! then splat_count * 8 * 4 bytes of f32 irradiance values.

use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"VXGI";

/// Save baked GI irradiance to a `.vxgi` binary file.
pub fn save_vxgi(irradiance: &[[f32; 8]], path: &Path) -> Result<(), String> {
    let mut buf = Vec::with_capacity(8 + irradiance.len() * 32);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(irradiance.len() as u32).to_le_bytes());
    for entry in irradiance {
        for &v in entry {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, &buf).map_err(|e| e.to_string())
}

/// Load baked GI irradiance from a `.vxgi` binary file.
pub fn load_vxgi(path: &Path) -> Result<Vec<[f32; 8]>, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if data.len() < 8 { return Err("File too short".into()); }
    if &data[0..4] != MAGIC { return Err("Invalid magic bytes".into()); }
    let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let expected = 8 + count * 32;
    if data.len() < expected {
        return Err(format!("Truncated: expected {} bytes, got {}", expected, data.len()));
    }
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let base = 8 + i * 32;
        let mut entry = [0.0f32; 8];
        for (j, v) in entry.iter_mut().enumerate() {
            let off = base + j * 4;
            *v = f32::from_le_bytes(data[off..off+4].try_into().unwrap());
        }
        result.push(entry);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let irr: Vec<[f32; 8]> = vec![
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ];
        let path = std::env::temp_dir().join("test_gi.vxgi");
        save_vxgi(&irr, &path).unwrap();
        let loaded = load_vxgi(&path).unwrap();
        assert_eq!(irr.len(), loaded.len());
        for (a, b) in irr.iter().zip(loaded.iter()) {
            for (&x, &y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_invalid_magic_returns_error() {
        let path = std::env::temp_dir().join("test_gi_bad.vxgi");
        std::fs::write(&path, b"NOPE0000").unwrap();
        assert!(load_vxgi(&path).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_missing_file_returns_error() {
        assert!(load_vxgi(Path::new("/nonexistent/gi.vxgi")).is_err());
    }
}
```

- [ ] Add `pub mod gi_export;` to `crates/vox_data/src/lib.rs`

- [ ] Run:
```bash
cargo test -p vox_data gi_export
```

- [ ] Commit:
```bash
git commit -m "feat(data): .vxgi binary format for baked spectral GI"
```

---

## Task 4: Wire into engine_runner — bake on load, apply at render time

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Add `gi_cache: Option<vox_render::gi_cache::GiCache>` field to `EngineApp` (all construction sites).

- [ ] After loading scene splats, trigger a GI bake (or load cached `.vxgi` if it exists):

```rust
fn rebuild_gi(&mut self) {
    let vxgi_path = std::path::Path::new("scene.vxgi");
    let irradiance = if vxgi_path.exists() {
        match vox_data::gi_export::load_vxgi(vxgi_path) {
            Ok(irr) if irr.len() == self.scene_splats.len() => irr,
            _ => self.run_gi_bake(),
        }
    } else {
        self.run_gi_bake()
    };
    let gi = vox_render::gi_baker::BakedGi { irradiance };
    self.gi_cache = Some(vox_render::gi_cache::GiCache::new(gi));
}

fn run_gi_bake(&self) -> Vec<[f32; 8]> {
    use vox_render::gi_baker::{GiBaker, GiBakeConfig};
    println!("[ochroma] Baking GI for {} splats...", self.scene_splats.len());
    let baker = GiBaker::new(GiBakeConfig {
        search_radius: 3.0,
        max_neighbours: 24,
        bounces: 2,
        falloff: 0.4,
    });
    let gi = baker.bake(&self.scene_splats);
    // Cache to disk
    let _ = vox_data::gi_export::save_vxgi(
        &gi.irradiance,
        std::path::Path::new("scene.vxgi"),
    );
    gi.irradiance
}
```

- [ ] In the render call, apply GI before passing splats to the renderer:

```rust
let render_splats = match &self.gi_cache {
    Some(cache) => cache.apply(&self.scene_splats),
    None => self.scene_splats.clone(),
};
// Pass render_splats to render_with_spectra_u8_shadowed (or render_with_spectra_u8)
```

- [ ] Verify compile:
```bash
cargo check --bin ochroma
```

- [ ] Commit:
```bash
git commit -m "feat(app): wire GI baking into engine_runner — bake on load, apply at render"
```

---

## Acceptance Criteria

| # | Test | Command |
|---|------|---------|
| 1 | GiBaker deterministic, bleeds correctly | `cargo test -p vox_render gi_baker` |
| 2 | GiCache applies and clamps correctly | `cargo test -p vox_render gi_cache` |
| 3 | `.vxgi` roundtrip | `cargo test -p vox_data gi_export` |
| 4 | Engine compiles | `cargo check --bin ochroma` |
| 5 | Full workspace green | `cargo test` |
