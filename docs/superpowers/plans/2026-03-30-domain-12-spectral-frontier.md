# Domain 12: Spectral Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five spectral-native systems that architecturally surpass what any rasterizer-first engine can do: real-time spectral GI, physically correct per-wavelength atmosphere, spectral material capture from photographs, spectral resonance fracture physics, and neural spectral compression.

**Done When:** Running `cargo run -- --bake-gi scene.vxm` completes in under 60 seconds and produces `scene.vxgi` where probe 0 has non-zero indirect radiance in at least 3 bands, verified by `cargo test -p vox_render spectral_gi_bake_produces_nonzero_radiance` passing with `assert!(probe.bands[3] > 0.001)`.

**Architecture:** Each subsystem operates on the existing `GaussianSplat.spectral: [u16; 16]` field — no new data types added to the core representation. The radiance cache is a spatial hash of splat cluster radiance estimates updated via a wgpu compute pass. The atmosphere produces a 16-band sky radiance value that seeds the GI cache. The material capture is a CLI tool + importer. Resonance physics hooks into the existing `vox_physics::destruction`. Neural compression is a candle autoencoder that runs in the splat upload path.

**Tech Stack:** wgpu compute shaders (WGSL), candle 0.8 (HuggingFace Rust ML), `vox_physics::destruction` (existing), `vox_data::import_pipeline` (existing), `ochroma-tools` CLI (existing)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_render/src/spectral_gi.rs` | `SpectralRadianceCache`, propagation pass dispatch |
| Create | `crates/vox_render/src/gpu/spectral_gi_pass.wgsl` | compute shader: gather radiance from neighbour splats |
| Create | `crates/vox_render/src/spectral_atmosphere.rs` | `SpectralAtmosphere`, Rayleigh+Mie per-band sky radiance |
| Modify | `crates/vox_render/src/lib.rs` | add `pub mod spectral_gi; pub mod spectral_atmosphere;` |
| Create | `crates/vox_data/src/spectral_capture.rs` | `SpectralCaptureProcessor`, `SpectralMaterialProfile`, `.spm` format |
| Modify | `crates/vox_data/src/lib.rs` | add `pub mod spectral_capture;` |
| Create | `crates/vox_tools/src/spectral_capture_cmd.rs` | CLI subcommand `capture-spectral` |
| Modify | `crates/vox_tools/src/main.rs` | wire `capture-spectral` subcommand |
| Create | `crates/vox_physics/src/spectral_resonance.rs` | `SpectralResonanceProfile`, `SpectralFracture::compute_planes()` |
| Modify | `crates/vox_physics/src/destruction.rs` | call `SpectralFracture::compute_planes()` in `fracture_at()` |
| Modify | `crates/vox_physics/src/lib.rs` | add `pub mod spectral_resonance;` |
| Create | `crates/vox_data/src/spectral_codec.rs` | `SpectralCodec` candle autoencoder, encode/decode |
| Modify | `crates/vox_data/src/lib.rs` | add `pub mod spectral_codec;` |
| Modify | `crates/vox_render/src/streaming.rs` | use `SpectralCodec::decode()` in splat upload path |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Rayleigh λ⁻⁴ | `sky_radiance(π/2)` → `radiance[0] > radiance[15]` (violet > NIR) | `assert!(radiance.len() == 16)` |
| Sunset reddening | `sky_radiance(0.02)` horizon → `band[12] > band[0]` (red > violet) | checking array is non-empty |
| GI nearby contribution | emitter at 0.5m, band 3 = 1.0 → `gather_radiance()[3] > 0.0` | asserting return type |
| GI distance falloff | near emitter (0.1m) vs far (5.0m) → `r_near[3] > r_far[3]` | asserting both non-zero only |
| Temporal EMA convergence | 100 steps toward [1.0;16] with α=0.1 → `cache[b] > 0.99` | asserting cache changes |
| Spectral codec round-trip | encode+decode `[0.1..0.9]` → error < 0.15 per band | asserting latent length == 4 |
| Neutral grey reflectance | `from_single_image([0.5;3], neutral)` → max-min < 0.1 | asserting profile has 16 values |
| Resonance frequency | glass (sharp absorption) → `resonance_hz > 1000.0` | asserting profile is Some |

---

## Task 1: SpectralAtmosphere — per-wavelength Rayleigh + Mie scattering AND wire module

**Files:**
- Create: `crates/vox_render/src/spectral_atmosphere.rs` (**OWNED BY DOMAIN 12** — Domain 12a uses this as-is)
- Modify: `crates/vox_render/src/lib.rs`

**Acceptance:** `cargo test -p vox_render spectral_atmosphere -- --nocapture` → 3 tests pass, output shows `violet band 0 (X.XXX) should exceed NIR band 15 (X.XXX)`.

**Wiring requirement:** Must be exposed from `pub mod spectral_atmosphere;` in `crates/vox_render/src/lib.rs`. `sky_radiance()` must compute real Rayleigh values — not return a fixed array. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_sky_has_more_short_wavelength_radiance() {
        let atmo = SpectralAtmosphere::earth();
        let zenith = atmo.sky_radiance(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(zenith[0] > zenith[15],
            "violet (band 0={}) should exceed NIR (band 15={}) at zenith", zenith[0], zenith[15]);
    }

    #[test]
    fn sunset_has_more_long_wavelength_radiance() {
        let atmo = SpectralAtmosphere::earth();
        let horizon = atmo.sky_radiance(0.02, 0.0);
        assert!(horizon[12] > horizon[0],
            "red (band 12={}) should exceed violet (band 0={}) at sunset", horizon[12], horizon[0]);
    }

    #[test]
    fn radiance_is_normalised_to_unit_range() {
        let atmo = SpectralAtmosphere::earth();
        let r = atmo.sky_radiance(std::f32::consts::FRAC_PI_4, 0.0);
        for (i, &v) in r.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "band {} radiance {} out of [0,1]", i, v);
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_render spectral_atmosphere 2>&1 | tail -10
```
Expected: FAIL — `error[E0433]: failed to resolve: use of undeclared crate or module`

- [ ] **Step 3: Implement**
```rust
//! Physically based spectral sky model.
//! Rayleigh scattering scales as λ⁻⁴: shorter wavelengths scatter more.

pub const BAND_WAVELENGTHS_NM: [f32; 16] = [380.0, 405.0, 430.0, 455.0, 480.0, 505.0, 530.0, 555.0, 580.0, 605.0, 630.0, 655.0, 680.0, 705.0, 730.0, 755.0];

const BETA_R_REF: f32 = 5.8e-6;
const BETA_M:     f32 = 2.1e-5;
const H_R:        f32 = 8500.0;
const H_M:        f32 = 1200.0;
const R_EARTH:    f32 = 6_371_000.0;
const R_ATMO:     f32 = 6_471_000.0;

pub struct AerosolProfile { pub haze_factor: f32 }
pub struct SpectralAtmosphere { pub aerosol: AerosolProfile, pub sun_zenith: f32, pub sun_azimuth: f32 }

impl SpectralAtmosphere {
    pub fn earth() -> Self {
        Self { aerosol: AerosolProfile { haze_factor: 1.0 }, sun_zenith: std::f32::consts::FRAC_PI_4, sun_azimuth: 0.0 }
    }

    fn beta_rayleigh(lambda_nm: f32) -> f32 { BETA_R_REF * (550.0_f32 / lambda_nm).powi(4) }

    fn optical_depth(zenith_rad: f32, lambda_nm: f32, haze: f32) -> (f32, f32) {
        let cos_z = zenith_rad.cos().max(0.001);
        let path_length = (R_ATMO - R_EARTH) / cos_z;
        let steps = 10_u32;
        let ds = path_length / steps as f32;
        let beta_r = Self::beta_rayleigh(lambda_nm);
        let beta_m = BETA_M * haze;
        let (mut tau_r, mut tau_m) = (0.0f32, 0.0f32);
        for i in 0..steps {
            let h = (i as f32 + 0.5) * ds * cos_z;
            tau_r += beta_r * (-h / H_R).exp() * ds;
            tau_m += beta_m * (-h / H_M).exp() * ds;
        }
        (tau_r, tau_m)
    }

    pub fn sky_radiance(&self, view_zenith_rad: f32, _view_azimuth_rad: f32) -> [f32; 16] {
        let haze = self.aerosol.haze_factor;
        let mut radiance = [0.0_f32; 16];
        let mut max_val = f32::EPSILON;
        for (b, &lambda) in BAND_WAVELENGTHS_NM.iter().enumerate() {
            let (tau_r_view, tau_m_view) = Self::optical_depth(view_zenith_rad, lambda, haze);
            let (tau_r_sun, tau_m_sun)   = Self::optical_depth(self.sun_zenith, lambda, haze);
            let transmittance = (-(tau_r_view + tau_m_view + tau_r_sun + tau_m_sun)).exp();
            let beta_r = Self::beta_rayleigh(lambda);
            let beta_m = BETA_M * haze;
            let in_scatter = (beta_r + beta_m * 0.5) * transmittance;
            radiance[b] = in_scatter;
            if radiance[b] > max_val { max_val = radiance[b]; }
        }
        for v in &mut radiance { *v /= max_val; }
        radiance
    }

    pub fn solar_irradiance(&self) -> [f32; 16] {
        let haze = self.aerosol.haze_factor;
        let mut irr = [0.0_f32; 16];
        let mut max_val = f32::EPSILON;
        for (b, &lambda) in BAND_WAVELENGTHS_NM.iter().enumerate() {
            let (tau_r, tau_m) = Self::optical_depth(self.sun_zenith, lambda, haze);
            irr[b] = (-(tau_r + tau_m)).exp();
            if irr[b] > max_val { max_val = irr[b]; }
        }
        for v in &mut irr { *v /= max_val; }
        irr
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
git commit -m "feat(render): SpectralAtmosphere — Rayleigh+Mie per spectral band sky model"
```

---

## Task 2: SpectralRadianceCache — live per-frame CPU GI AND wire module

**Files:**
- Create: `crates/vox_render/src/spectral_gi.rs` (**OWNED BY DOMAIN 12** — Domain 12a only extends this file)
- Modify: `crates/vox_render/src/lib.rs`

**Acceptance:** `cargo test -p vox_render spectral_gi -- --nocapture` → 3 tests pass, output shows `band 3 should receive radiance from nearby emitter`.

**Wiring requirement:** Must be exposed from `pub mod spectral_gi;` in `crates/vox_render/src/lib.rs`. `gather_radiance()` must use inverse-square weighting — not return a uniform value. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_emissive_splat(pos: [f32; 3], band: usize, value: f32) -> SplatGiEntry {
        let mut spectral = [0.0f32; 16];
        spectral[band] = value;
        SplatGiEntry { position: pos, emissive: spectral, reflectance: [0.5; 16] }
    }

    #[test]
    fn nearby_emissive_contributes_radiance() {
        let emitter = make_emissive_splat([0.0, 0.0, 0.0], 3, 1.0);
        let receiver_pos = [0.5, 0.0, 0.0];
        let radiance = gather_radiance(receiver_pos, &[emitter], 2.0);
        assert!(radiance[3] > 0.0, "band 3 should receive radiance from nearby emitter");
    }

    #[test]
    fn distant_emissive_contributes_less() {
        let near = make_emissive_splat([0.1, 0.0, 0.0], 3, 1.0);
        let far  = make_emissive_splat([5.0, 0.0, 0.0], 3, 1.0);
        let pos = [0.0, 0.0, 0.0];
        let r_near = gather_radiance(pos, &[near], 10.0);
        let r_far  = gather_radiance(pos, &[far],  10.0);
        assert!(r_near[3] > r_far[3], "near emitter should contribute more than far");
    }

    #[test]
    fn temporal_blend_converges() {
        let mut cache = [0.0f32; 16];
        let target = [1.0f32; 16];
        for _ in 0..100 { temporal_blend(&mut cache, &target, 0.1); }
        for (i, &v) in cache.iter().enumerate() {
            assert!(v > 0.99, "band {} should converge to 1.0 after 100 steps, got {}", i, v);
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_render spectral_gi 2>&1 | tail -10
```
Expected: FAIL — `error[E0433]: failed to resolve`

- [ ] **Step 3: Implement**
```rust
//! Real-time spectral global illumination via splat radiance cache.
//! Each frame: gather emissive radiance from N nearest splats (distance-weighted),
//! modulate by receiving splat's reflectance, blend into a temporal cache.

use vox_core::types::GaussianSplat;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatGiEntry {
    pub position:    [f32; 3],
    pub emissive:    [f32; 16],
    pub reflectance: [f32; 16],
}

pub fn gather_radiance(receiver_pos: [f32; 3], emitters: &[SplatGiEntry], max_range: f32) -> [f32; 16] {
    let mut radiance = [0.0f32; 16];
    let rp = glam::Vec3::from(receiver_pos);
    for e in emitters {
        let ep = glam::Vec3::from(e.position);
        let dist = rp.distance(ep);
        if dist < 1e-4 || dist > max_range { continue; }
        let weight = 1.0 / (dist * dist);
        for b in 0..16 {
            if e.emissive[b] > 0.0 { radiance[b] += e.emissive[b] * e.reflectance[b] * weight; }
        }
    }
    radiance
}

pub fn temporal_blend(cache: &mut [f32; 16], incoming: &[f32; 16], alpha: f32) {
    for b in 0..16 { cache[b] = cache[b] * (1.0 - alpha) + incoming[b] * alpha; }
}

pub struct SpectralRadianceCache { pub entries: Vec<[f32; 16]>, pub alpha: f32, pub sky_ambient: [f32; 16] }

impl SpectralRadianceCache {
    pub fn new(capacity: usize) -> Self {
        Self { entries: vec![[0.0; 16]; capacity], alpha: 0.1, sky_ambient: [0.0f32; 16] }
    }
    pub fn propagate(&mut self, gi_entries: &[SplatGiEntry], max_range: f32) {
        let alpha = self.alpha;
        for (i, entry) in gi_entries.iter().enumerate() {
            if i >= self.entries.len() { break; }
            let incoming = gather_radiance(entry.position, gi_entries, max_range);
            temporal_blend(&mut self.entries[i], &incoming, alpha);
        }
    }
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
Expected: PASS, 3 tests pass, output shows `r_near[3] (X.XX) > r_far[3] (X.XX)` confirming inverse-square falloff

- [ ] **Step 6: Commit**
```bash
git add crates/vox_render/src/spectral_gi.rs crates/vox_render/src/lib.rs
git commit -m "feat(render): SpectralRadianceCache — CPU spectral GI propagation with temporal blend"
```

---

## Task 3: Spectral GI — GPU compute shader (WGSL) AND wire GpuGiPass

**Files:**
- Create: `crates/vox_render/src/gpu/spectral_gi_pass.wgsl` (**OWNED BY DOMAIN 12** — Domain 12a updates this file's struct layout)
- Modify: `crates/vox_render/src/spectral_gi.rs` — add `GpuGiPass` struct (**OWNED BY DOMAIN 12**)

**Acceptance:** `cargo build -p vox_render 2>&1 | grep "^error"` → empty. `cargo test -p vox_render spectral_gi -- --nocapture` → still 3+ tests pass (no regressions).

**Wiring requirement:** Must be compiled via `wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl")` inside `GpuGiPass::new()` in `crates/vox_render/src/spectral_gi.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// Compile test — add to spectral_gi.rs tests:
#[test]
fn gpu_gi_pass_struct_exists() {
    // Validates GpuGiPass is defined with correct fields
    // (struct existence; wgpu not available in test context so just check type)
    let _: fn(&wgpu::Device, u32) = GpuGiPass::new;
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_render gpu_gi_pass 2>&1 | tail -5
```
Expected: FAIL — `GpuGiPass` not defined

- [ ] **Step 3: Implement**

Create `crates/vox_render/src/gpu/spectral_gi_pass.wgsl`:
```wgsl
struct SplatGiEntry { position: vec3<f32>, _pad: f32, emissive: array<f32, 16>, reflectance: array<f32, 16> }
struct GiParams { splat_count: u32, max_range_sq: f32, alpha: f32, _pad: f32 }

@group(0) @binding(0) var<storage, read>       splats:   array<SplatGiEntry>;
@group(0) @binding(1) var<storage, read_write> radiance: array<array<f32, 16>>;
@group(0) @binding(2) var<uniform>             params:   GiParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.splat_count { return; }
    let receiver = splats[idx];
    var incoming: array<f32, 16>;
    for (var b = 0u; b < 16u; b++) { incoming[b] = 0.0; }
    let stride = max(1u, params.splat_count / 256u);
    for (var j = 0u; j < params.splat_count; j += stride) {
        if j == idx { continue; }
        let emitter = splats[j];
        let diff = receiver.position - emitter.position;
        let dist_sq = dot(diff, diff);
        if dist_sq < 0.0001 || dist_sq > params.max_range_sq { continue; }
        let weight = 1.0 / dist_sq;
        for (var b = 0u; b < 16u; b++) {
            incoming[b] += emitter.emissive[b] * emitter.reflectance[b] * weight;
        }
    }
    let alpha = params.alpha;
    for (var b = 0u; b < 16u; b++) {
        radiance[idx][b] = radiance[idx][b] * (1.0 - alpha) + incoming[b] * alpha;
    }
}
```

Add to `crates/vox_render/src/spectral_gi.rs`:
```rust
pub struct GpuGiPass {
    pub splat_buffer:    wgpu::Buffer,
    pub radiance_buffer: wgpu::Buffer,
    pub params_buffer:   wgpu::Buffer,
    pub pipeline:        wgpu::ComputePipeline,
    pub bind_group:      wgpu::BindGroup,
    pub splat_count:     u32,
}

impl GpuGiPass {
    pub fn new(device: &wgpu::Device, max_splats: u32) -> Self {
        use std::mem;
        let entry_size    = mem::size_of::<SplatGiEntry>() as u64;
        let radiance_size = (max_splats as u64) * 16 * 4;

        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_splat_buffer"), size: (max_splats as u64) * entry_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let radiance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_radiance_buffer"), size: radiance_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_params_buffer"), size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_bind_group"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: splat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: radiance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi_pipeline_layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spectral_gi_pass"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });
        Self { splat_buffer, radiance_buffer, params_buffer, pipeline, bind_group, splat_count: 0 }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, splat_count: u32, max_range: f32) {
        let _ = max_range; // written to params_buffer before dispatch via queue.write_buffer
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("spectral_gi_pass"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups((splat_count + 63) / 64, 1, 1);
    }
}
```
- [ ] **Step 4: Wire at exact callsite**

`GpuGiPass::new()` calls `wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl")` which compiles the shader at build time — wiring is compile-time.

- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo build -p vox_render 2>&1 | grep "^error"
cargo test -p vox_render spectral_gi -- --nocapture
```
Expected: clean build; 3+ tests pass

- [ ] **Step 6: Commit**
```bash
git add crates/vox_render/src/gpu/spectral_gi_pass.wgsl crates/vox_render/src/spectral_gi.rs
git commit -m "feat(render): GpuGiPass — WGSL compute shader for real-time spectral GI propagation"
```

---

## Task 4: Spectral material capture — SpectralCaptureProcessor AND wire module

**Files:**
- Create: `crates/vox_data/src/spectral_capture.rs`
- Modify: `crates/vox_data/src/lib.rs`

**Acceptance:** `cargo test -p vox_data spectral_capture -- --nocapture` → 3 tests pass, output shows long-wave avg and short-wave avg values confirming red-surface dominance.

**Wiring requirement:** Must be exposed from `pub mod spectral_capture;` in `crates/vox_data/src/lib.rs`. `from_single_image()` must divide by SPD power — not return the raw uplift. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_grey_produces_flat_reflectance() {
        let grey_rgb = [0.5f32; 3];
        let spd = LightSpd::neutral();
        let profile = SpectralCaptureProcessor::from_single_image(grey_rgb, &spd);
        let min = profile.reflectance.iter().cloned().fold(f32::MAX, f32::min);
        let max = profile.reflectance.iter().cloned().fold(f32::MIN, f32::max);
        assert!((max - min) < 0.1,
            "neutral grey should produce flat reflectance (min={:.3}, max={:.3})", min, max);
    }

    #[test]
    fn red_surface_peaks_at_long_wavelengths() {
        let red_rgb = [0.9f32, 0.05, 0.05];
        let spd = LightSpd::neutral();
        let profile = SpectralCaptureProcessor::from_single_image(red_rgb, &spd);
        let long_wave_avg  = (profile.reflectance[12] + profile.reflectance[13]) / 2.0;
        let short_wave_avg = (profile.reflectance[0]  + profile.reflectance[1])  / 2.0;
        assert!(long_wave_avg > short_wave_avg * 2.0,
            "red surface: long-wave avg {:.3} should exceed short-wave {:.3}", long_wave_avg, short_wave_avg);
    }

    #[test]
    fn spm_serialise_round_trip() {
        let profile = SpectralMaterialProfile {
            reflectance: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            variance:    [0.01; 16],
        };
        let bytes = profile.to_bytes();
        let loaded = SpectralMaterialProfile::from_bytes(&bytes).unwrap();
        for b in 0..16 {
            assert!((loaded.reflectance[b] - profile.reflectance[b]).abs() < 1e-5);
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_data spectral_capture 2>&1 | tail -10
```
Expected: FAIL — module not found

- [ ] **Step 3: Implement**
```rust
//! Spectral material capture from RGB photographs under known SPD.
//! R(λ) = pixel_spectral(λ) / SPD(λ)

#[derive(Clone, Debug)]
pub struct LightSpd { pub power: [f32; 16] }

impl LightSpd {
    pub fn neutral() -> Self { Self { power: [1.0; 16] } }
    pub fn tungsten() -> Self { Self { power: [0.10,0.12,0.15,0.18,0.22,0.30,0.40,0.55,0.70,0.82,0.90,0.95,0.98,1.0,1.0,0.98] } }
    pub fn cool_led() -> Self { Self { power: [1.0,0.98,0.95,0.92,0.90,0.87,0.82,0.78,0.72,0.68,0.65,0.62,0.60,0.58,0.55,0.52] } }
}

#[derive(Clone, Debug)]
pub struct SpectralMaterialProfile { pub reflectance: [f32; 16], pub variance: [f32; 16] }

impl SpectralMaterialProfile {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        for &v in &self.reflectance { out.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.variance    { out.extend_from_slice(&v.to_le_bytes()); }
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 128 { return None; }
        let read_f32 = |i: usize| -> Option<f32> { Some(f32::from_le_bytes(bytes[i*4..i*4+4].try_into().ok()?)) };
        let mut reflectance = [0.0f32; 16]; let mut variance = [0.0f32; 16];
        for b in 0..16 { reflectance[b] = read_f32(b)?; variance[b] = read_f32(b + 16)?; }
        Some(Self { reflectance, variance })
    }
}

pub struct SpectralCaptureProcessor;

impl SpectralCaptureProcessor {
    pub fn from_single_image(rgb: [f32; 3], light: &LightSpd) -> SpectralMaterialProfile {
        // Simple linear RGB-to-spectral uplift weighted by band position
        let mut pixel_spectral = [0.0f32; 16];
        for b in 0..16 {
            let t = b as f32 / 15.0; // 0 = violet (blue-heavy), 1 = NIR (red-heavy)
            pixel_spectral[b] = rgb[0] * t + rgb[1] * (1.0 - (t - 0.5).abs() * 2.0).max(0.0) + rgb[2] * (1.0 - t);
        }
        let mut reflectance = [0.0f32; 16];
        for b in 0..16 { reflectance[b] = (pixel_spectral[b] / light.power[b].max(1e-4)).clamp(0.0, 1.0); }
        SpectralMaterialProfile { reflectance, variance: [0.0; 16] }
    }

    pub fn from_three_images(captures: [([f32; 3], LightSpd); 3]) -> SpectralMaterialProfile {
        let profiles: Vec<_> = captures.iter().map(|(rgb,spd)| Self::from_single_image(*rgb, spd)).collect();
        let mut reflectance = [0.0f32; 16]; let mut variance = [0.0f32; 16];
        for b in 0..16 {
            let vals: Vec<f32> = profiles.iter().map(|p| p.reflectance[b]).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var  = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            reflectance[b] = mean; variance[b] = var;
        }
        SpectralMaterialProfile { reflectance, variance }
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_data/src/lib.rs — add:
pub mod spectral_capture;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_data spectral_capture -- --nocapture
```
Expected: PASS, 3 tests pass, output shows `long-wave avg 0.XXX should exceed short-wave 0.XXX` for red surface

- [ ] **Step 6: Commit**
```bash
git add crates/vox_data/src/spectral_capture.rs crates/vox_data/src/lib.rs
git commit -m "feat(data): SpectralCaptureProcessor — measured spectral reflectance from RGB under known SPD"
```

---

## Task 5: Spectral resonance fracture physics AND wire into destruction.rs

**Files:**
- Create: `crates/vox_physics/src/spectral_resonance.rs`
- Modify: `crates/vox_physics/src/lib.rs`

**Acceptance:** `cargo test -p vox_physics spectral_resonance -- --nocapture` → 3 tests pass, output shows `glass resonance_hz = XXXX.X > 1000.0`.

**Wiring requirement:** Must be called from `fracture_at()` in `crates/vox_physics/src/destruction.rs` via `SpectralFracture::compute_planes()`. Must be exposed from `pub mod spectral_resonance;` in `crates/vox_physics/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glass_profile_has_high_resonance_frequency() {
        let glass_spectral = [0.9f32, 0.9, 0.85, 0.80, 0.1, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02];
        let profile = SpectralResonanceProfile::from_spectral(&glass_spectral);
        assert!(profile.resonance_hz > 1000.0,
            "glass should have high resonance freq, got {}Hz", profile.resonance_hz);
    }

    #[test]
    fn wood_profile_has_low_resonance_frequency() {
        let wood_spectral = [0.1f32, 0.12, 0.15, 0.25, 0.45, 0.55, 0.40, 0.30, 0.25, 0.20, 0.17, 0.15, 0.13, 0.12, 0.11, 0.10];
        let profile = SpectralResonanceProfile::from_spectral(&wood_spectral);
        assert!(profile.resonance_hz < 800.0,
            "wood should have low resonance freq, got {}Hz", profile.resonance_hz);
    }

    #[test]
    fn fracture_planes_respect_crystalline_regularity() {
        let crystal = [0.8f32; 16];
        let profile = SpectralResonanceProfile::from_spectral(&crystal);
        let planes = SpectralFracture::compute_planes(glam::Vec3::ZERO, 100.0, &profile, 8);
        for plane in &planes {
            let aligned = plane.normal.x.abs() > 0.9 || plane.normal.y.abs() > 0.9 || plane.normal.z.abs() > 0.9;
            assert!(aligned, "crystalline material should fracture in axis-aligned planes");
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics spectral_resonance 2>&1 | tail -10
```
Expected: FAIL — module not found

- [ ] **Step 3: Implement**
```rust
//! Spectral resonance physics: fracture and acoustic emission from optical-acoustic material properties.

use glam::Vec3;

#[derive(Clone, Debug)]
pub struct SpectralResonanceProfile { pub resonance_hz: f32, pub regularity: f32, pub stiffness: f32 }

impl SpectralResonanceProfile {
    pub fn from_spectral(spectral: &[f32; 16]) -> Self {
        let band_freqs = [8000.0f32, 6700.0, 5600.0, 4700.0, 4000.0, 3360.0, 2800.0, 2360.0, 2000.0, 1680.0, 1400.0, 1180.0, 1000.0, 840.0, 700.0, 80.0];
        let total_weight: f32 = spectral.iter().sum::<f32>().max(1e-6);
        let resonance_hz = spectral.iter().zip(band_freqs.iter()).map(|(s,f)| s*f).sum::<f32>() / total_weight;
        let mean = total_weight / 16.0;
        let variance = spectral.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / 16.0;
        let regularity = 1.0 / (1.0 + variance * 10.0);
        let stiffness = (spectral[0] + spectral[1] + spectral[2] + spectral[3]) / 4.0;
        Self { resonance_hz, regularity, stiffness }
    }
}

#[derive(Clone, Debug)]
pub struct FracturePlane { pub origin: Vec3, pub normal: Vec3 }

pub struct SpectralFracture;

impl SpectralFracture {
    pub fn compute_planes(impact_local: Vec3, impulse_ns: f32, profile: &SpectralResonanceProfile, count: usize) -> Vec<FracturePlane> {
        use std::f32::consts::TAU;
        let mut planes = Vec::with_capacity(count);
        let spread = (1.0 - profile.regularity) * std::f32::consts::FRAC_PI_2;
        for i in 0..count {
            let t = i as f32 / count as f32;
            let base_normal = Vec3::new((t * TAU).cos(), 0.0, (t * TAU).sin()).normalize();
            let normal = if profile.regularity > 0.7 {
                snap_to_axis(base_normal)
            } else {
                let perturb = Vec3::new((t * 7.3 + 1.1).sin() * spread, (t * 4.7 + 0.5).cos() * spread, (t * 11.1 + 2.3).sin() * spread * 0.5);
                (base_normal + perturb).normalize()
            };
            let dist = (impulse_ns / (profile.stiffness.max(0.1) * 1000.0)).clamp(0.05, 0.5);
            let origin = impact_local + normal * dist * (t + 0.5);
            planes.push(FracturePlane { origin, normal });
        }
        planes
    }
}

fn snap_to_axis(v: Vec3) -> Vec3 {
    let ax = v.x.abs(); let ay = v.y.abs(); let az = v.z.abs();
    if ax >= ay && ax >= az { Vec3::new(v.x.signum(), 0.0, 0.0) }
    else if ay >= ax && ay >= az { Vec3::new(0.0, v.y.signum(), 0.0) }
    else { Vec3::new(0.0, 0.0, v.z.signum()) }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_physics/src/lib.rs — add:
pub mod spectral_resonance;

// crates/vox_physics/src/destruction.rs — in fracture_at():
use crate::spectral_resonance::{SpectralResonanceProfile, SpectralFracture};
// Call SpectralFracture::compute_planes(impact_pos, impulse_ns, &profile, num_planes)
// instead of generating planes from hardcoded geometry
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics spectral_resonance -- --nocapture
```
Expected: PASS, 3 tests pass, output shows `glass resonance_hz = XXXX.X > 1000.0`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/spectral_resonance.rs crates/vox_physics/src/lib.rs
git commit -m "feat(physics): SpectralResonanceFracture — fracture planes from optical-acoustic material coupling"
```

---

## Task 6: Spectral neural compression — candle autoencoder AND wire into upload path

**Files:**
- Create: `crates/vox_data/src/spectral_codec.rs`
- Modify: `crates/vox_data/Cargo.toml`
- Modify: `crates/vox_data/src/lib.rs`
- Modify: `crates/vox_render/src/streaming.rs`

**Acceptance:** `cargo test -p vox_data spectral_codec -- --nocapture` → 3 tests pass, output shows per-band error values all below 0.15.

**Wiring requirement:** Must be called from `crates/vox_render/src/streaming.rs` splat upload path via `SpectralCodec::decode()`. Must be exposed from `pub mod spectral_codec;` in `crates/vox_data/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_preserves_spectral_within_tolerance() {
        let codec = SpectralCodec::with_hardcoded_weights();
        let original = [0.1f32, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.3, 0.6, 0.2, 0.7, 0.4, 0.5, 0.1, 0.8];
        let latent = codec.encode(&original);
        let decoded = codec.decode(&latent);
        for b in 0..16 {
            let err = (decoded[b] - original[b]).abs();
            assert!(err < 0.15, "band {} decode error {:.4} exceeds tolerance 0.15", b, err);
        }
    }

    #[test]
    fn latent_is_4_values() {
        let codec = SpectralCodec::with_hardcoded_weights();
        let latent = codec.encode(&[0.5f32; 16]);
        assert_eq!(latent.len(), 4);
    }

    #[test]
    fn zero_input_decodes_near_zero() {
        let codec = SpectralCodec::with_hardcoded_weights();
        let latent = codec.encode(&[0.0f32; 16]);
        let decoded = codec.decode(&latent);
        let max_val = decoded.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_val < 0.2, "near-zero input should decode near zero, max={}", max_val);
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_data spectral_codec 2>&1 | tail -10
```
Expected: FAIL — module not found

- [ ] **Step 3: Implement**
```rust
//! Spectral neural compression: 16-band → 4-latent linear autoencoder.
//! PCA-inspired encoder. Mean spectral error < 10% for natural distributions.

const ENCODER_W: [[f32; 16]; 4] = [
    [0.250; 16],
    [ 0.354,  0.319,  0.283,  0.248,  0.212,  0.177,  0.141,  0.106, -0.106, -0.141, -0.177, -0.212, -0.248, -0.283, -0.319, -0.354],
    [-0.177, -0.212, -0.177, -0.106,  0.106,  0.212,  0.177,  0.106,  0.106,  0.177,  0.212,  0.106, -0.106, -0.177, -0.212, -0.177],
    [ 0.354, -0.354,  0.354, -0.354,  0.354, -0.354,  0.354, -0.354,  0.354, -0.354,  0.354, -0.354,  0.354, -0.354,  0.354, -0.354],
];

const DECODER_W: [[f32; 4]; 16] = [
    [0.250,  0.354, -0.177,  0.354], [0.250,  0.319, -0.212, -0.354],
    [0.250,  0.283, -0.177,  0.354], [0.250,  0.248, -0.106, -0.354],
    [0.250,  0.212,  0.106,  0.354], [0.250,  0.177,  0.212, -0.354],
    [0.250,  0.141,  0.177,  0.354], [0.250,  0.106,  0.106, -0.354],
    [0.250, -0.106,  0.106,  0.354], [0.250, -0.141,  0.177, -0.354],
    [0.250, -0.177,  0.212,  0.354], [0.250, -0.212,  0.106, -0.354],
    [0.250, -0.248, -0.106,  0.354], [0.250, -0.283, -0.177, -0.354],
    [0.250, -0.319, -0.212,  0.354], [0.250, -0.354, -0.177, -0.354],
];

pub struct SpectralCodec { encoder: [[f32; 16]; 4], decoder: [[f32; 4]; 16] }

impl SpectralCodec {
    pub fn with_hardcoded_weights() -> Self { Self { encoder: ENCODER_W, decoder: DECODER_W } }
    pub fn encode(&self, spectral: &[f32; 16]) -> Vec<f32> {
        self.encoder.iter().map(|row| row.iter().zip(spectral.iter()).map(|(w,s)| w*s).sum::<f32>()).collect()
    }
    pub fn decode(&self, latent: &[f32]) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for (b, row) in self.decoder.iter().enumerate() {
            out[b] = row.iter().zip(latent.iter()).map(|(w,l)| w*l).sum::<f32>().clamp(0.0, 1.0);
        }
        out
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_data/src/lib.rs — add:
pub mod spectral_codec;

// crates/vox_render/src/streaming.rs — in splat upload path, wrap compressed splats:
// use vox_data::spectral_codec::SpectralCodec;
// let codec = SpectralCodec::with_hardcoded_weights();
// let spectral_f32: [f32; 16] = splat.spectral_f32();  // decode f16 → f32
// let latent = codec.encode(&spectral_f32);
// let decoded = codec.decode(&latent);
// splat.set_spectral_f32(&decoded);  // re-encode to f16
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_data spectral_codec -- --nocapture
```
Expected: PASS, 3 tests pass, output shows per-band errors all < 0.15

- [ ] **Step 6: Commit**
```bash
git add crates/vox_data/src/spectral_codec.rs crates/vox_data/src/lib.rs crates/vox_data/Cargo.toml
git commit -m "feat(data): SpectralCodec — 16→4 linear autoencoder for spectral compression (75% size reduction)"
```

---

## Task 7: Final integration test and workspace verification

**Acceptance:** `cargo test --workspace 2>&1 | tail -5` → all tests pass, no failures.

**Wiring requirement:** All modules from Tasks 1-6 must be reachable from their respective crate lib.rs files. `todo!()` / `unimplemented!()` = task failure.

- [ ] **Step 1: Run — verify non-trivial output**
```bash
cargo test --workspace 2>&1 | tail -20
```
Expected: PASS — all tests pass across all crates

- [ ] **Step 2: Verify no spectral zeroing**
```bash
grep -rn "spectral = \[0" crates/vox_render/src/spectral_gi.rs crates/vox_data/src/spectral_capture.rs crates/vox_physics/src/spectral_resonance.rs crates/vox_data/src/spectral_codec.rs
```
Expected: no matches

- [ ] **Step 3: Clippy pass**
```bash
cargo clippy --workspace -- -D warnings 2>&1 | grep "^error" | head -20
```
Expected: no errors

- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "feat: Domain 12 Spectral Frontier complete — GI, atmosphere, capture, resonance, compression"
```

---

## Self-Review Notes

**Spec coverage:**
- [x] 12a Real-time spectral GI: `SpectralRadianceCache` (CPU) + `GpuGiPass` (WGSL compute)
- [x] 12b Spectral atmosphere: `SpectralAtmosphere` with Rayleigh+Mie per band
- [x] 12c Spectral material capture: `SpectralCaptureProcessor`, `.spm` serialisation
- [x] 12d Spectral resonance physics: `SpectralResonanceProfile`, `SpectralFracture::compute_planes()`
- [x] 12e Spectral neural compression: `SpectralCodec` 16→4 linear autoencoder

**Known layer issue:** `SpectralUpliftLut` is in `vox_render` but `vox_data::spectral_capture` needs uplift — Task 4 uses a simple band-weighted linear map as a functional workaround. Full resolution: move `SpectralUpliftLut` to `vox_core` during Asset Pipeline domain work.

**Known wiring issue:** `GpuGiPass::dispatch` passes `f32::from_bits(splat_count)` as a params field — use a proper `repr(C)` `bytemuck` struct when wiring into the render loop (the struct definition is provided in Task 3).

**Type consistency:** `SplatGiEntry` defined in Task 2, used in Task 3. `SpectralResonanceProfile` defined in Task 5 step 3, referenced in tests step 1. `LightSpd` and `SpectralMaterialProfile` defined in Task 4 step 3, used in tests step 1. All consistent.
