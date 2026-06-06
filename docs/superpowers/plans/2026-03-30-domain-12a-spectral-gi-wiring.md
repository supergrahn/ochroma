# Domain 12a: Spectral GI Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **PREREQUISITE:** Domain 12 (Spectral Frontier) must be fully implemented and passing before starting Domain 12a. This plan extends Domain 12's `SpectralRadianceCache`, `spectral_gi.rs`, and `spectral_gi_pass.wgsl` — do not implement Domain 12a standalone.

**Goal:** Wire live spectral GI into the engine render loop — replace the static baked `GiCache` and RGB `illuminant_for_time()` with a per-frame `SpectralRadianceCache` seeded by a physically correct per-band atmosphere model.

**Done When:** Running `cargo run` with a baked `.vxgi` file shows visible indirect lighting contribution (shadowed areas are not pitch black but have a non-zero spectral contribution), verified by `cargo test -p vox_render gi_wiring_applies_indirect_to_render` passing with `assert!(lit_pixel.iter().sum::<f32>() > dark_pixel.iter().sum::<f32>() + 0.05)`.

**Architecture:** `SpectralAtmosphere` computes Rayleigh β(λ) = β_ref × (550/λ)⁴ for each of the 16 spectral bands, producing sky radiance that seeds the `SpectralRadianceCache`. The cache gathers radiance from nearby emissive splats each frame (inverse-square, temporal EMA). The result replaces the old `GiCache::apply()` at the render loop splice point (`engine_runner.rs:884`). GPU compute pass is added in Task 4 as a performance upgrade over the CPU path.

**Tech Stack:** Rust, wgpu, `half::f16` (existing), `bytemuck` (existing), WGSL compute

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `crates/vox_render/src/spectral_atmosphere.rs` | `SpectralAtmosphere` — already created by Domain 12; no changes needed in 12a |
| Modify | `crates/vox_render/src/spectral_gi.rs` | extend Domain 12's `SpectralRadianceCache`; add `GpuGiPass` with corrected `GpuSplatEntry` |
| Modify | `crates/vox_render/src/gpu/spectral_gi_pass.wgsl` | extend Domain 12's shader with corrected `GpuSplatEntry` struct layout |
| Modify | `crates/vox_render/src/lib.rs` | expose new modules (already partially done by Domain 12) |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | wire into render loop |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Rayleigh λ⁻⁴ | zenith `sky_radiance()` → `radiance[0] > radiance[15]` (violet > NIR) | `assert!(radiance.len() == 16)` |
| Horizon reddening | horizon `sky_radiance(0.05, 0.0)` → zenith_ratio (blue/NIR) > horizon_ratio | checking sky returns any values |
| Solar irradiance range | `solar_irradiance()` → all values in `[0.0, 1.0]` | asserting non-empty |
| GI propagation | near emissive splat → receiver `cache[1]` any value > 0 | checking cache length |
| GI apply modulates splat | cache = [0.5;16], splat band 0 = 0.1 → `result.spectral()[0]` decoded > 0.1 | asserting spectral changed at all |
| GPU struct sizes | `GpuSplatEntry` = 144 bytes, `GiParamsUniform` = 16 bytes | checking non-zero |

---

## Task 1: SpectralAtmosphere — per-band Rayleigh sky AND wire module

**Files:**
- Modify: `crates/vox_render/src/spectral_atmosphere.rs` (already created by Domain 12 — skip re-implementation; verify tests pass)
- Modify: `crates/vox_render/src/lib.rs`

**Acceptance:** `cargo test -p vox_render spectral_atmosphere -- --nocapture` → 3 tests pass, output shows `violet band 0 (1.000) > NIR band 15 (0.0XX)`.

**Wiring requirement:** `spectral_atmosphere` module is already exposed by Domain 12. `sky_radiance()` takes two arguments: `(view_zenith_rad: f32, view_azimuth_rad: f32)` — this is the signature defined by Domain 12 and must be used consistently here. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_sky_violet_exceeds_red() {
        let atmo = SpectralAtmosphere::earth();
        let radiance = atmo.sky_radiance(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(radiance[0] > radiance[15],
            "violet band 0 ({}) should exceed NIR band 15 ({}) — Rayleigh λ⁻⁴",
            radiance[0], radiance[15]);
    }

    #[test]
    fn horizon_is_redder_than_zenith() {
        let atmo = SpectralAtmosphere::earth();
        let zenith  = atmo.sky_radiance(std::f32::consts::FRAC_PI_2, 0.0);
        let horizon = atmo.sky_radiance(0.05, 0.0);
        let zenith_ratio  = zenith[0]  / (zenith[15]  + 1e-6);
        let horizon_ratio = horizon[0] / (horizon[15] + 1e-6);
        assert!(zenith_ratio > horizon_ratio,
            "zenith is bluer (ratio {:.2}) than horizon (ratio {:.2})", zenith_ratio, horizon_ratio);
    }

    #[test]
    fn solar_irradiance_in_unit_range() {
        let atmo = SpectralAtmosphere::earth();
        let irr = atmo.solar_irradiance();
        for (i, &v) in irr.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "band {} irradiance {} out of [0,1]", i, v);
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cd /home/tomespen/git/ochroma
cargo test -p vox_render spectral_atmosphere 2>&1 | head -20
```
Expected: FAIL — compile error, `SpectralAtmosphere` not yet in `lib.rs`

- [ ] **Step 3: Implement**
```rust
//! Physically correct per-wavelength sky radiance (Rayleigh + Mie).

pub const BAND_NM: [f32; 16] = [380.0, 405.0, 430.0, 455.0, 480.0, 505.0, 530.0, 555.0, 580.0, 605.0, 630.0, 655.0, 680.0, 705.0, 730.0, 755.0];
const BETA_R_REF: f32 = 5.8e-6;

pub struct SpectralAtmosphere {
    pub sun_elevation: f32,
    pub turbidity:     f32,
}

impl SpectralAtmosphere {
    pub fn earth() -> Self {
        Self { sun_elevation: std::f32::consts::FRAC_PI_4, turbidity: 2.0 }
    }

    pub fn beta_rayleigh(lambda_nm: f32) -> f32 {
        BETA_R_REF * (550.0 / lambda_nm).powi(4)
    }

    pub fn sky_radiance(&self, view_zenith_rad: f32, _view_azimuth_rad: f32) -> [f32; 16] {
        let cos_sun  = self.sun_elevation.sin().clamp(0.0, 1.0);
        let view_h   = view_zenith_rad.sin().clamp(0.0, 1.0);
        let path_len = 1.0 / (view_h + 0.01);
        let mut out = [0.0f32; 16];
        let mut max_val = f32::EPSILON;
        for i in 0..16 {
            let beta = Self::beta_rayleigh(BAND_NM[i]);
            let mie  = 21e-6 * self.turbidity * 0.1;
            let scatter = (beta + mie) * path_len;
            out[i] = cos_sun * scatter * (-scatter * 0.1).exp();
            if out[i] > max_val { max_val = out[i]; }
        }
        for v in &mut out { *v /= max_val; }
        out
    }

    pub fn solar_irradiance(&self) -> [f32; 16] {
        let elev     = self.sun_elevation.clamp(0.0, std::f32::consts::FRAC_PI_2);
        let cos_elev = elev.sin().max(0.0);
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            let atm_loss = Self::beta_rayleigh(BAND_NM[i]) * (1.0 / (cos_elev + 0.01));
            out[i] = (cos_elev * (-atm_loss).exp()).clamp(0.0, 1.0);
        }
        out
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_render/src/lib.rs — add:
pub mod spectral_atmosphere;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_render spectral_atmosphere -- --nocapture
```
Expected: PASS, 3 tests pass, output shows `violet band 0 (1.000) should exceed NIR band 15 (0.0XX)`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_render/src/spectral_atmosphere.rs crates/vox_render/src/lib.rs
git commit -m "feat(render): SpectralAtmosphere — per-band Rayleigh sky radiance"
```

---

## Task 2: SpectralRadianceCache — live per-frame CPU GI AND wire module

**Files:**
- Modify: `crates/vox_render/src/spectral_gi.rs` (created by Domain 12; this task extends it)
- Modify: `crates/vox_render/src/lib.rs`

**Acceptance:** `cargo test -p vox_render spectral_gi -- --nocapture` → 4 tests pass, output shows `receiver should have non-zero irradiance after propagation`.

**Wiring requirement:** Must be exposed from `pub mod spectral_gi;` in `crates/vox_render/src/lib.rs`. `propagate()` must gather from emissive splats — not leave cache at zero. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn make_splat(pos: [f32; 3], spectral_val: f32, opacity: u8) -> GaussianSplat {
        let v = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat::surface(pos.into(), [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.1, 0.1, opacity, [v; 16])
    }

    #[test]
    fn cache_initialises_empty() {
        let cache = SpectralRadianceCache::new(10);
        assert_eq!(cache.cache.len(), 10);
        assert!(cache.cache.iter().all(|c: &[f32; 16]| c.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn nearby_emissive_splat_adds_irradiance() {
        let mut cache = SpectralRadianceCache::new(2);
        cache.alpha = 0.0; // no temporal smoothing
        let emitter  = make_splat([0.0, 0.0, 0.0], 1.0, 255);
        let receiver = make_splat([1.0, 0.0, 0.0], 0.0,  50); // low opacity = not emissive
        cache.propagate(&[emitter, receiver], 100);
        assert!(cache.cache[1].iter().any(|&v| v > 0.0),
            "receiver should have non-zero irradiance after propagation");
    }

    #[test]
    fn apply_adds_gi_to_spectral() {
        let mut cache = SpectralRadianceCache::new(1);
        cache.cache[0] = [0.5f32; 16];
        let splat = make_splat([0.0, 0.0, 0.0], 0.1, 200);
        let result = cache.apply(&[splat]);
        let out_val = half::f16::from_bits(result[0].spectral()[0]).to_f32();
        assert!(out_val > 0.1, "GI should have added to splat spectral (got {})", out_val);
    }

    #[test]
    fn resize_on_splat_count_change() {
        let mut cache = SpectralRadianceCache::new(5);
        cache.resize(10);
        assert_eq!(cache.cache.len(), 10);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_render spectral_gi 2>&1 | head -20
```
Expected: FAIL — compile error, module not exposed

- [ ] **Step 3: Implement**
```rust
//! Live spectral GI — CPU propagation path.
//! Gathers radiance from nearby emissive splats each frame using inverse-square
//! distance weighting, then blends with an exponential moving average.

use vox_core::types::GaussianSplat;
use half::f16;
use crate::spectral_atmosphere::SpectralAtmosphere;

#[derive(Clone)]
pub struct SplatGiEntry { pub position: [f32; 3], pub emissive: [f32; 16] }

pub struct SpectralRadianceCache {
    pub cache:       Vec<[f32; 16]>,
    pub alpha:       f32,
    pub sky_ambient: [f32; 16],
}

impl SpectralRadianceCache {
    pub fn new(splat_count: usize) -> Self {
        Self { cache: vec![[0.0f32; 16]; splat_count], alpha: 0.9, sky_ambient: [0.0f32; 16] }
    }

    pub fn set_sky(&mut self, atmo: &SpectralAtmosphere) { self.sky_ambient = atmo.solar_irradiance(); }
    pub fn resize(&mut self, count: usize) { self.cache.resize(count, [0.0f32; 16]); }

    pub fn propagate(&mut self, splats: &[GaussianSplat], max_emitters: usize) {
        self.resize(splats.len());
        let emitters: Vec<SplatGiEntry> = splats.iter()
            .filter(|s| s.opacity() > 128)
            .take(max_emitters)
            .map(|s| SplatGiEntry { position: s.position(), emissive: decode_spectral(&s.spectral()) })
            .collect();

        let sky = self.sky_ambient;
        let alpha = self.alpha;
        for (i, splat) in splats.iter().enumerate() {
            let pos = splat.position();
            let mut incoming = sky;
            for emitter in &emitters {
                let dx = emitter.position[0] - pos[0];
                let dy = emitter.position[1] - pos[1];
                let dz = emitter.position[2] - pos[2];
                let dist_sq = (dx*dx + dy*dy + dz*dz).max(0.01);
                let weight  = 1.0 / dist_sq;
                for b in 0..16 { incoming[b] += emitter.emissive[b] * weight; }
            }
            let max_incoming = incoming.iter().copied().fold(f32::EPSILON, f32::max);
            let scale = if max_incoming > 1.0 { 1.0 / max_incoming } else { 1.0 };
            for b in 0..16 {
                self.cache[i][b] = alpha * self.cache[i][b]
                    + (1.0 - alpha) * (incoming[b] * scale).clamp(0.0, 1.0);
            }
        }
    }

    pub fn apply(&self, splats: &[GaussianSplat]) -> Vec<GaussianSplat> {
        splats.iter().enumerate().map(|(i, s)| {
            let irr = if i < self.cache.len() { self.cache[i] } else { self.sky_ambient };
            let mut out = *s;
            let spectral = decode_spectral(&s.spectral());
            for b in 0..16 {
                let modulated = (spectral[b] + irr[b] * 0.5).clamp(0.0, 1.0);
                out.spectral_mut()[b] = f16::from_f32(modulated).to_bits();
            }
            out
        }).collect()
    }
}

fn decode_spectral(s: &[u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 { out[i] = f16::from_bits(s[i]).to_f32(); }
    out
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_render/src/lib.rs — add:
pub mod spectral_gi;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_render spectral_gi -- --nocapture
```
Expected: PASS, 4 tests pass, output shows `receiver should have non-zero irradiance after propagation`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_render/src/spectral_gi.rs crates/vox_render/src/lib.rs
git commit -m "feat(render): SpectralRadianceCache — live per-frame CPU spectral GI"
```

---

## Task 3: Wire into engine_runner — replace static GiCache AND wire spectral atmosphere

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

**Acceptance:** `cargo build -p vox_app 2>&1 | grep "^error"` → empty. Time-of-day now drives spectral atmosphere; old `gi_cache` field replaced.

**Wiring requirement:** Must be called from `render_frame()` in `crates/vox_app/src/bin/engine_runner.rs` at line ~884. The static GiCache apply block must be replaced — not left alongside. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
# Build test — verify new fields exist:
cargo build -p vox_app 2>&1 | grep "^error"
```
Expected: FAIL — `spectral_atmosphere` field not yet in `EngineApp`

- [ ] **Step 2: Run to verify failure**
```bash
cargo build -p vox_app 2>&1 | grep "spectral_atmosphere\|spectral_gi" | head -5
```
Expected: no mentions yet (fields not added)

- [ ] **Step 3: Implement** — add fields to `EngineApp` struct:
```rust
// In EngineApp struct definition:
spectral_atmosphere: vox_render::spectral_atmosphere::SpectralAtmosphere,
spectral_gi:         vox_render::spectral_gi::SpectralRadianceCache,
```

Initialise in `EngineApp::new()`:
```rust
spectral_atmosphere: vox_render::spectral_atmosphere::SpectralAtmosphere::earth(),
spectral_gi:         vox_render::spectral_gi::SpectralRadianceCache::new(0),
```
- [ ] **Step 4: Wire at exact callsite**

In `render_frame()`, before the GI apply block (~line 884):
```rust
// Update spectral atmosphere from time of day
{
    let hour = self.engine.time_of_day();
    let norm = (hour % 24.0) / 24.0;
    self.spectral_atmosphere.sun_elevation =
        (std::f32::consts::PI * norm - std::f32::consts::FRAC_PI_2).sin()
            .max(0.0) * std::f32::consts::FRAC_PI_2;
    self.spectral_gi.set_sky(&self.spectral_atmosphere);
}
```

Replace static GI apply block:
```rust
// REPLACE this:
// let render_splats = match &self.gi_cache {
//     Some(cache) => cache.apply(&render_splats),
//     None => render_splats,
// };

// WITH this:
self.spectral_gi.propagate(&render_splats, 256);
let render_splats = self.spectral_gi.apply(&render_splats);
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo build -p vox_app 2>&1 | grep -E "^error" | head -10
```
Expected: PASS — empty output (clean build)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(app): wire SpectralAtmosphere + SpectralRadianceCache into render loop"
```

---

## Task 4: GPU GI Pass — wgpu compute for production performance AND wire into dispatch

> **2026-06-06 status:** a GPU spectral-GI compute path landed in `crates/vox_render/src/spectral_gi.rs` as `GpuGi` (commit 0e7bdb7) — not the `GpuGiPass` named here; it owns a headless wgpu device and `step(&[GaussianSplat], hour)`, proven on the RADV 780M (monotonic falloff, bit-identical reruns, 50k splats @ 67.75 ms/step). **EngineLoop / render-loop wiring is explicitly DEFERRED** — `GpuGi` is exercised only by its own tests; the live frame still runs the CPU `SpectralRadianceCache` path. Treat this task as "compute landed, dispatch-into-frame pending."

**Files:**
- Modify: `crates/vox_render/src/gpu/spectral_gi_pass.wgsl` (created by Domain 12; update struct layout to match corrected `GpuSplatEntry`)
- Modify: `crates/vox_render/src/spectral_gi.rs` (add `GpuGiPass` with corrected `GpuSplatEntry`)
- Modify: `crates/vox_render/src/gpu/mod.rs`

**Acceptance:** `cargo test -p vox_render gpu_tests -- --nocapture` → 2 tests pass, output shows `GpuSplatEntry = 80 bytes` and `GiParamsUniform = 16 bytes`.

**Wiring requirement:** Must be compiled via `wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl")` inside `GpuGiPass::new()` in `crates/vox_render/src/spectral_gi.rs`. `dispatch()` must call `pass.dispatch_workgroups()` — not be empty. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpu_splat_entry_size() {
        // 4 (position + pad) + 16 (radiance) + 16 (reflectance) = 36 floats × 4 bytes = 144 bytes
        assert_eq!(std::mem::size_of::<GpuSplatEntry>(), 144);
    }

    #[test]
    fn gi_params_size() {
        assert_eq!(std::mem::size_of::<GiParamsUniform>(), 16);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_render gpu_tests 2>&1 | head -20
```
Expected: FAIL — `GpuSplatEntry` not defined

- [ ] **Step 3: Implement**

Update `crates/vox_render/src/gpu/spectral_gi_pass.wgsl` (extends Domain 12's version — replace `SplatGiEntry` with the corrected layout):
```wgsl
struct GpuSplatEntry {
    position: vec3<f32>,
    _pad0: f32,
    radiance: array<f32, 16>,
    reflectance: array<f32, 16>,
};
struct GiParams { splat_count: u32, max_emitters: u32, alpha: f32, _pad: f32 }

@group(0) @binding(0) var<storage, read>       splats:   array<GpuSplatEntry>;
@group(0) @binding(1) var<storage, read_write> radiance: array<array<f32, 16>>;
@group(0) @binding(2) var<uniform>             params:   GiParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let receiver_idx = gid.x;
    if receiver_idx >= params.splat_count { return; }
    let pos = splats[receiver_idx].position;
    var incoming: array<f32, 16>;
    let candidate_count = min(params.splat_count, params.max_emitters);
    let stride = max(params.splat_count / candidate_count, 1u);
    for (var k = 0u; k < candidate_count; k++) {
        let emitter_idx = k * stride;
        if emitter_idx == receiver_idx { continue; }
        let ep = splats[emitter_idx].position;
        let dx = ep.x - pos.x; let dy = ep.y - pos.y; let dz = ep.z - pos.z;
        let dist_sq = max(dx*dx + dy*dy + dz*dz, 0.01);
        let weight = 1.0 / dist_sq;
        for (var b = 0u; b < 16u; b++) { incoming[b] += splats[emitter_idx].radiance[b] * weight; }
    }
    var max_val = 0.00001;
    for (var b = 0u; b < 16u; b++) { if incoming[b] > max_val { max_val = incoming[b]; } }
    let alpha = params.alpha;
    for (var b = 0u; b < 16u; b++) {
        let new_val = clamp(incoming[b] / max_val, 0.0, 1.0);
        radiance[receiver_idx][b] = alpha * radiance[receiver_idx][b] + (1.0 - alpha) * new_val;
    }
}
```

Add to `crates/vox_render/src/spectral_gi.rs`:
```rust
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct GpuSplatEntry {
    pub position: [f32; 3],
    pub _pad0: f32,          // pad to 16 bytes for WGSL vec3 alignment
    pub radiance: [f32; 16], // 64 bytes
    pub reflectance: [f32; 16], // 64 bytes
}                            // total: 144 bytes

const _: () = assert!(std::mem::size_of::<GpuSplatEntry>() == 144);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiParamsUniform { pub splat_count: u32, pub max_emitters: u32, pub alpha: f32, pub _pad: f32 }

pub struct GpuGiPass {
    pub splat_buffer:    wgpu::Buffer,
    pub radiance_buffer: wgpu::Buffer,
    pub params_buffer:   wgpu::Buffer,
    pipeline:            wgpu::ComputePipeline,
    bind_group:          wgpu::BindGroup,
    max_splats:          u32,
}

impl GpuGiPass {
    pub fn new(device: &wgpu::Device, max_splats: u32) -> Self {
        let splat_bytes    = max_splats as u64 * std::mem::size_of::<GpuSplatEntry>() as u64;
        let radiance_bytes = max_splats as u64 * 16 * 4;
        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_splat_buf"), size: splat_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let radiance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_radiance_buf"), size: radiance_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_params_buf"), size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true  }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_bg"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: splat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: radiance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi_pl"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi_pipeline"), layout: Some(&pipeline_layout), module: &shader,
            entry_point: Some("main"), cache: None, compilation_options: Default::default(),
        });
        Self { splat_buffer, radiance_buffer, params_buffer, pipeline, bind_group, max_splats }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, splats_gpu: &[GpuSplatEntry], splat_count: u32, alpha: f32) {
        let count = splat_count.min(self.max_splats);
        queue.write_buffer(&self.splat_buffer, 0, bytemuck::cast_slice(splats_gpu));
        let params = GiParamsUniform { splat_count: count, max_emitters: 256, alpha, _pad: 0.0 };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gi_pass"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups((count + 63) / 64, 1, 1);
    }
}
```
- [ ] **Step 4: Wire at exact callsite**

`GpuGiPass::new()` uses `wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl")` — compile-time wiring. `dispatch()` is called from the render loop when a GPU device is available.

- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_render gpu_tests -- --nocapture
```
Expected: PASS, 2 tests pass, output shows `GpuSplatEntry = 80` and `GiParamsUniform = 16`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_render/src/spectral_gi.rs crates/vox_render/src/gpu/spectral_gi_pass.wgsl
git commit -m "feat(render): GpuGiPass — wgpu compute for spectral GI propagation"
```

---

## Task 5: Verification and cleanup — remove static GiCache

**Acceptance:** `cargo test --workspace 2>&1 | tail -5` → all tests pass; `grep -n "gi_cache" crates/vox_app/src/bin/engine_runner.rs` → no matches (field removed).

**Wiring requirement:** `SpectralRadianceCache` must be the only GI path — `GiCache` must be deleted from the struct. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```bash
grep -n "gi_cache" crates/vox_app/src/bin/engine_runner.rs
```
Expected: currently has matches (old field present)

- [ ] **Step 2: Run to verify failure**
```bash
cargo test --workspace 2>&1 | tail -20
```
Expected: all tests pass (confirms no regressions before cleanup)

- [ ] **Step 3: Implement** — remove dead `gi_cache` field:
```rust
// DELETE from EngineApp struct:
// gi_cache: Option<vox_render::gi_cache::GiCache>,

// DELETE from EngineApp::new():
// gi_cache: None,
```
- [ ] **Step 4: Wire at exact callsite**

Remove all remaining references to `gi_cache` in `engine_runner.rs`.

- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test --workspace 2>&1 | tail -5
```
Expected: PASS — all tests pass; no `gi_cache` references remain

- [ ] **Step 6: Commit**
```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "refactor(app): remove static GiCache — replaced by live SpectralRadianceCache"
```

---

## Self-Review

**Spec coverage:**
- [x] Rayleigh β(λ) = β_ref × (550/λ)⁴ per band → Task 1
- [x] Sky seeds GI cache → Task 3 wires `set_sky()`
- [x] Temporal EMA → Task 2, `alpha = 0.9`
- [x] GPU compute pass → Task 4
- [x] Replace static GiCache → Tasks 3 + 5
- [x] Replace RGB illuminant_for_time → Task 3

**Known approximation:** The `sky_radiance()` single-scattering approximation is not a full Hosek-Wilkie model. It correctly orders bands (violet > NIR at zenith) across all 16 bands and handles path length attenuation, which is sufficient for GI seeding. Full Hosek-Wilkie can be substituted later without changing the interface.

**Known limitation:** CPU propagation in Task 2 is O(N × 256) per frame. At 500k splats this is 128M operations — too slow at 60fps. Task 4 (GPU pass) is required for production. The CPU path is correct and fast enough for development iteration.
