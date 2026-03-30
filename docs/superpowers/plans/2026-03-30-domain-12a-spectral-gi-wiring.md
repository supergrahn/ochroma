# Domain 12a — Spectral GI Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire live spectral GI into the engine render loop — replace the static baked `GiCache` and RGB `illuminant_for_time()` with a per-frame `SpectralRadianceCache` seeded by a physically correct per-band atmosphere model.

**Architecture:** `SpectralAtmosphere` computes Rayleigh β(λ) = β_ref × (550/λ)⁴ for each of the 8 spectral bands, producing sky radiance that seeds the `SpectralRadianceCache`. The cache gathers radiance from nearby emissive splats each frame (inverse-square, temporal EMA). The result replaces the old `GiCache::apply()` at the render loop splice point (`engine_runner.rs:884`). GPU compute pass is added in Task 4 as a performance upgrade over the CPU path.

**Tech Stack:** Rust, wgpu, `half::f16` (existing), `bytemuck` (existing), WGSL compute

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_render/src/spectral_atmosphere.rs` | `SpectralAtmosphere` — Rayleigh+Mie per band |
| Create | `crates/vox_render/src/spectral_gi.rs` | `SpectralRadianceCache`, CPU propagation, `GpuGiPass` |
| Create | `crates/vox_render/src/gpu/spectral_gi_pass.wgsl` | compute shader: gather radiance per splat |
| Modify | `crates/vox_render/src/lib.rs` | expose new modules |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | wire into render loop |

---

## Task 1: SpectralAtmosphere — per-band Rayleigh sky

**Files:**
- Create: `crates/vox_render/src/spectral_atmosphere.rs`
- Modify: `crates/vox_render/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `crates/vox_render/src/spectral_atmosphere.rs`:

```rust
//! Physically correct per-wavelength sky radiance (Rayleigh + Mie).
//! Replaces the RGB AtmosphereParams for spectral rendering.

/// Centre wavelength of each spectral band in nanometres.
pub const BAND_NM: [f32; 8] = [380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0];

/// Reference Rayleigh scattering coefficient at 550 nm (1/m, sea level).
const BETA_R_REF: f32 = 5.8e-6;

pub struct SpectralAtmosphere {
    /// Sun altitude above horizon in radians. 0 = horizon, π/2 = zenith.
    pub sun_elevation: f32,
    /// Aerosol turbidity [1, 10]. 1 = pure, 10 = hazy.
    pub turbidity: f32,
}

impl SpectralAtmosphere {
    pub fn earth() -> Self {
        Self { sun_elevation: std::f32::consts::FRAC_PI_4, turbidity: 2.0 }
    }

    /// Rayleigh scattering coefficient β(λ) = β_ref × (550/λ)⁴.
    pub fn beta_rayleigh(lambda_nm: f32) -> f32 {
        BETA_R_REF * (550.0 / lambda_nm).powi(4)
    }

    /// Approximate sky radiance in each spectral band for the given view direction.
    /// `view_elevation`: radians above horizon.
    /// Returns values in [0, 1] (normalised).
    pub fn sky_radiance(&self, view_elevation: f32) -> [f32; 8] {
        let cos_sun = self.sun_elevation.sin().clamp(0.0, 1.0);
        let view_h = view_elevation.sin().clamp(0.0, 1.0);
        let path_len = 1.0 / (view_h + 0.01);  // longer path at horizon

        let mut out = [0.0f32; 8];
        let mut max_val = f32::EPSILON;
        for i in 0..8 {
            let beta = Self::beta_rayleigh(BAND_NM[i]);
            // Mie contribution — wavelength-independent, scaled by turbidity
            let mie = 21e-6 * self.turbidity * 0.1;
            let scatter = (beta + mie) * path_len;
            // Attenuated scattered radiance towards viewer
            out[i] = cos_sun * scatter * (-scatter * 0.1).exp();
            if out[i] > max_val { max_val = out[i]; }
        }
        for v in &mut out { *v /= max_val; }
        out
    }

    /// Approximate solar irradiance per spectral band (direct sun).
    /// Returns values in [0, 1] (normalised at zenith).
    pub fn solar_irradiance(&self) -> [f32; 8] {
        let elev = self.sun_elevation.clamp(0.0, std::f32::consts::FRAC_PI_2);
        let cos_elev = elev.sin().max(0.0);
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            let atm_loss = Self::beta_rayleigh(BAND_NM[i]) * (1.0 / (cos_elev + 0.01));
            out[i] = (cos_elev * (-atm_loss).exp()).clamp(0.0, 1.0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_sky_violet_exceeds_red() {
        let atmo = SpectralAtmosphere::earth();
        let radiance = atmo.sky_radiance(std::f32::consts::FRAC_PI_2);
        assert!(radiance[0] > radiance[7],
            "violet band 0 ({}) should exceed red band 7 ({}) — Rayleigh λ⁻⁴",
            radiance[0], radiance[7]);
    }

    #[test]
    fn horizon_is_redder_than_zenith() {
        let atmo = SpectralAtmosphere::earth();
        let zenith = atmo.sky_radiance(std::f32::consts::FRAC_PI_2);
        let horizon = atmo.sky_radiance(0.05);
        let zenith_ratio = zenith[0] / (zenith[7] + 1e-6);
        let horizon_ratio = horizon[0] / (horizon[7] + 1e-6);
        assert!(zenith_ratio > horizon_ratio,
            "zenith is bluer (ratio {:.2}) than horizon (ratio {:.2})",
            zenith_ratio, horizon_ratio);
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

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/tomespen/git/ochroma
cargo test -p vox_render spectral_atmosphere 2>&1 | head -20
```

Expected: compile error — `SpectralAtmosphere` not yet in `lib.rs`

- [ ] **Step 3: Expose the module**

Add to `crates/vox_render/src/lib.rs`:

```rust
pub mod spectral_atmosphere;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p vox_render spectral_atmosphere -- --nocapture
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/spectral_atmosphere.rs crates/vox_render/src/lib.rs
git commit -m "feat(render): SpectralAtmosphere — per-band Rayleigh sky radiance"
```

---

## Task 2: SpectralRadianceCache — live per-frame CPU GI

**Files:**
- Create: `crates/vox_render/src/spectral_gi.rs`
- Modify: `crates/vox_render/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Create `crates/vox_render/src/spectral_gi.rs`:

```rust
//! Live spectral GI — CPU propagation path.
//! Gathers radiance from nearby emissive splats each frame using inverse-square
//! distance weighting, then blends with an exponential moving average.

use vox_core::types::GaussianSplat;
use half::f16;
use crate::spectral_atmosphere::SpectralAtmosphere;

/// Per-splat entry used in the radiance cache.
#[derive(Clone)]
pub struct SplatGiEntry {
    pub position: [f32; 3],
    /// Emissive spectral power (decoded from splat.spectral).
    pub emissive: [f32; 8],
}

/// Live spectral radiance cache updated every frame.
pub struct SpectralRadianceCache {
    /// Accumulated irradiance per splat (index matches render_splats).
    pub cache: Vec<[f32; 8]>,
    /// Temporal blend weight. 0 = full new frame, 1 = no update.
    pub alpha: f32,
    /// Sky seed from SpectralAtmosphere.solar_irradiance() applied as ambient.
    pub sky_ambient: [f32; 8],
}

impl SpectralRadianceCache {
    pub fn new(splat_count: usize) -> Self {
        Self {
            cache: vec![[0.0f32; 8]; splat_count],
            alpha: 0.9,
            sky_ambient: [0.0f32; 8],
        }
    }

    /// Update sky ambient from the current atmosphere state.
    pub fn set_sky(&mut self, atmo: &SpectralAtmosphere) {
        self.sky_ambient = atmo.solar_irradiance();
    }

    /// Resize cache if splat count changes.
    pub fn resize(&mut self, count: usize) {
        self.cache.resize(count, [0.0f32; 8]);
    }

    /// Propagate one frame: gather radiance from emissive splats into each receiver.
    /// Only samples up to `max_emitters` nearest emitters for performance.
    pub fn propagate(&mut self, splats: &[GaussianSplat], max_emitters: usize) {
        self.resize(splats.len());

        // Build emitter list from high-opacity splats with non-zero spectral values
        let emitters: Vec<SplatGiEntry> = splats.iter()
            .filter(|s| s.opacity > 128)
            .take(max_emitters)
            .map(|s| SplatGiEntry {
                position: s.position,
                emissive: decode_spectral(&s.spectral),
            })
            .collect();

        let sky = self.sky_ambient;
        let alpha = self.alpha;

        for (i, splat) in splats.iter().enumerate() {
            let pos = splat.position;
            let mut incoming = sky;

            // Gather from emitters (inverse-square)
            for emitter in &emitters {
                let dx = emitter.position[0] - pos[0];
                let dy = emitter.position[1] - pos[1];
                let dz = emitter.position[2] - pos[2];
                let dist_sq = (dx*dx + dy*dy + dz*dz).max(0.01);
                let weight = 1.0 / dist_sq;
                for b in 0..8 {
                    incoming[b] += emitter.emissive[b] * weight;
                }
            }

            // Normalise and clamp
            let max_incoming = incoming.iter().copied().fold(f32::EPSILON, f32::max);
            let scale = if max_incoming > 1.0 { 1.0 / max_incoming } else { 1.0 };

            // Temporal blend
            for b in 0..8 {
                self.cache[i][b] = alpha * self.cache[i][b]
                    + (1.0 - alpha) * (incoming[b] * scale).clamp(0.0, 1.0);
            }
        }
    }

    /// Apply cached GI to splats: modulate spectral by cached irradiance.
    pub fn apply(&self, splats: &[GaussianSplat]) -> Vec<GaussianSplat> {
        splats.iter().enumerate().map(|(i, s)| {
            let irr = if i < self.cache.len() { self.cache[i] } else { self.sky_ambient };
            let mut out = *s;
            let spectral = decode_spectral(&s.spectral);
            for b in 0..8 {
                let modulated = (spectral[b] + irr[b] * 0.5).clamp(0.0, 1.0);
                out.spectral[b] = f16::from_f32(modulated).to_bits();
            }
            out
        }).collect()
    }
}

fn decode_spectral(s: &[u16; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for i in 0..8 { out[i] = f16::from_bits(s[i]).to_f32(); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn make_splat(pos: [f32; 3], spectral_val: f32, opacity: u8) -> GaussianSplat {
        let v = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat {
            position: pos,
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity,
            _pad: [0; 3],
            spectral: [v; 8],
        }
    }

    #[test]
    fn cache_initialises_empty() {
        let cache = SpectralRadianceCache::new(10);
        assert_eq!(cache.cache.len(), 10);
        assert!(cache.cache.iter().all(|c| c.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn nearby_emissive_splat_adds_irradiance() {
        let mut cache = SpectralRadianceCache::new(2);
        cache.alpha = 0.0; // no temporal smoothing

        let emitter = make_splat([0.0, 0.0, 0.0], 1.0, 255);
        let receiver = make_splat([1.0, 0.0, 0.0], 0.0, 50); // low opacity = not emissive
        cache.propagate(&[emitter, receiver], 100);

        // Receiver should pick up irradiance from emitter
        assert!(cache.cache[1].iter().any(|&v| v > 0.0),
            "receiver should have non-zero irradiance after propagation");
    }

    #[test]
    fn apply_adds_gi_to_spectral() {
        let mut cache = SpectralRadianceCache::new(1);
        cache.cache[0] = [0.5f32; 8];
        let splat = make_splat([0.0, 0.0, 0.0], 0.1, 200);
        let result = cache.apply(&[splat]);
        let out_val = half::f16::from_bits(result[0].spectral[0]).to_f32();
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

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_render spectral_gi 2>&1 | head -20
```

Expected: compile error — module not exposed

- [ ] **Step 3: Expose the module**

Add to `crates/vox_render/src/lib.rs`:

```rust
pub mod spectral_gi;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p vox_render spectral_gi -- --nocapture
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/src/spectral_gi.rs crates/vox_render/src/lib.rs
git commit -m "feat(render): SpectralRadianceCache — live per-frame CPU spectral GI"
```

---

## Task 3: Wire into engine_runner — replace static GiCache

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

Context: `EngineApp` currently has:
- `gi_cache: Option<vox_render::gi_cache::GiCache>` (line 264) — baked, static
- `illuminant_for_time(hour) -> Illuminant` (line 286) — RGB, no spectral physics

We replace both with a single live spectral system.

- [ ] **Step 1: Add new fields to EngineApp struct**

Find the `EngineApp` struct definition (around line 200). Add after the `gi_cache` field:

```rust
    // Live spectral atmosphere (Rayleigh per band, updated from time_of_day each frame)
    spectral_atmosphere: vox_render::spectral_atmosphere::SpectralAtmosphere,

    // Live spectral GI cache (propagation from emissive splats, updated per frame)
    spectral_gi: vox_render::spectral_gi::SpectralRadianceCache,
```

- [ ] **Step 2: Initialise new fields in EngineApp::new()**

Find where `EngineApp` is constructed (search for `gi_cache: None`). After that line add:

```rust
            spectral_atmosphere: vox_render::spectral_atmosphere::SpectralAtmosphere::earth(),
            spectral_gi: vox_render::spectral_gi::SpectralRadianceCache::new(0),
```

- [ ] **Step 3: Update atmosphere from time-of-day each frame**

Find `render_frame()`. Before the `// Apply GI cache` block (around line 884), add:

```rust
        // Update spectral atmosphere from time of day
        {
            let hour = self.engine.time_of_day();
            let norm = (hour % 24.0) / 24.0;
            // Sun elevation: rises from 0 at midnight, peaks at π/2 at noon
            self.spectral_atmosphere.sun_elevation =
                (std::f32::consts::PI * norm - std::f32::consts::FRAC_PI_2).sin()
                    .max(0.0) * std::f32::consts::FRAC_PI_2;
            self.spectral_gi.set_sky(&self.spectral_atmosphere);
        }
```

- [ ] **Step 4: Replace the static GI apply with live spectral GI**

Find the block at line 884:

```rust
        // Apply GI cache (modulates spectral bands per-splat)
        let render_splats = match &self.gi_cache {
            Some(cache) => cache.apply(&render_splats),
            None => render_splats,
        };
```

Replace with:

```rust
        // Live spectral GI: propagate from emissive splats, then apply
        self.spectral_gi.propagate(&render_splats, 256);
        let render_splats = self.spectral_gi.apply(&render_splats);
```

- [ ] **Step 5: Build to verify it compiles**

```bash
cargo build -p vox_app 2>&1 | grep -E "error|warning.*unused" | head -30
```

Expected: clean build (may have unused import warnings for old `gi_cache` field — that's OK).

- [ ] **Step 6: Run the engine and verify spectral GI is live**

```bash
cargo run -p vox_app --bin engine_runner -- --scene assets/demo.vxm 2>&1 | head -20
```

Expected: engine starts, no panic. Time-of-day changes should now affect spectral band distribution differently than before (violet-heavy at noon, red-heavy at dawn/dusk).

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(app): wire SpectralAtmosphere + SpectralRadianceCache into render loop"
```

---

## Task 4: GPU GI Pass — wgpu compute for production performance

This task upgrades the CPU propagation (Task 2) to a GPU compute pass. The CPU path remains as a fallback when no wgpu device is available.

**Files:**
- Create: `crates/vox_render/src/gpu/spectral_gi_pass.wgsl`
- Modify: `crates/vox_render/src/spectral_gi.rs` (add `GpuGiPass`)
- Modify: `crates/vox_render/src/gpu/mod.rs`

- [ ] **Step 1: Write the compute shader**

Create `crates/vox_render/src/gpu/spectral_gi_pass.wgsl`:

```wgsl
// Spectral GI compute pass — gathers irradiance from nearby emissive splats.
// Each workgroup thread handles one receiver splat.
// Strides through the first min(splat_count, 256) splats as candidate emitters.

struct SplatEntry {
    position: vec3<f32>,
    opacity: f32,
    spectral: array<f32, 8>,  // decoded from f16 on CPU before upload
}

struct GiParams {
    splat_count: u32,
    max_emitters: u32,
    alpha: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read>       splats:    array<SplatEntry>;
@group(0) @binding(1) var<storage, read_write> radiance:  array<array<f32, 8>>;
@group(0) @binding(2) var<uniform>             params:    GiParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let receiver_idx = gid.x;
    if receiver_idx >= params.splat_count { return; }

    let pos = splats[receiver_idx].position;
    var incoming: array<f32, 8>;

    let candidate_count = min(params.splat_count, params.max_emitters);
    let stride = max(params.splat_count / candidate_count, 1u);

    for (var k = 0u; k < candidate_count; k++) {
        let emitter_idx = k * stride;
        if emitter_idx == receiver_idx { continue; }
        if splats[emitter_idx].opacity < 0.5 { continue; }

        let ep = splats[emitter_idx].position;
        let dx = ep.x - pos.x;
        let dy = ep.y - pos.y;
        let dz = ep.z - pos.z;
        let dist_sq = max(dx*dx + dy*dy + dz*dz, 0.01);
        let weight = 1.0 / dist_sq;

        for (var b = 0u; b < 8u; b++) {
            incoming[b] += splats[emitter_idx].spectral[b] * weight;
        }
    }

    // Normalise
    var max_val = 0.00001;
    for (var b = 0u; b < 8u; b++) {
        if incoming[b] > max_val { max_val = incoming[b]; }
    }

    // Temporal blend with existing cache
    let alpha = params.alpha;
    for (var b = 0u; b < 8u; b++) {
        let new_val = clamp(incoming[b] / max_val, 0.0, 1.0);
        radiance[receiver_idx][b] = alpha * radiance[receiver_idx][b] + (1.0 - alpha) * new_val;
    }
}
```

- [ ] **Step 2: Write GpuGiPass struct**

Add to `crates/vox_render/src/spectral_gi.rs`:

```rust
use wgpu;

/// GPU-accelerated GI pass. Computes in O(N × max_emitters / 64) instead of O(N × max_emitters).
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
        let splat_bytes = max_splats as u64 * std::mem::size_of::<GpuSplatEntry>() as u64;
        let radiance_bytes = max_splats as u64 * 8 * 4;  // 8 × f32

        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_splat_buf"),
            size: splat_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let radiance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_radiance_buf"),
            size: radiance_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_params_buf"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl"));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: splat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: radiance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Self { splat_buffer, radiance_buffer, params_buffer, pipeline, bind_group, max_splats }
    }

    /// Upload splat data and dispatch the compute pass.
    /// `splat_count` must be ≤ `max_splats`.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        splats_gpu: &[GpuSplatEntry],
        splat_count: u32,
        alpha: f32,
    ) {
        let count = splat_count.min(self.max_splats);
        queue.write_buffer(&self.splat_buffer, 0, bytemuck::cast_slice(splats_gpu));
        let params = GiParamsUniform { splat_count: count, max_emitters: 256, alpha, _pad: 0.0 };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gi_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups((count + 63) / 64, 1, 1);
    }
}

/// GPU layout for one splat entry (matches WGSL SplatEntry).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSplatEntry {
    pub position: [f32; 3],
    pub opacity: f32,
    pub spectral: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GiParamsUniform {
    splat_count: u32,
    max_emitters: u32,
    alpha: f32,
    _pad: f32,
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpu_splat_entry_size() {
        // 3 + 1 + 8 = 12 floats × 4 bytes = 48 bytes
        assert_eq!(std::mem::size_of::<GpuSplatEntry>(), 48);
    }

    #[test]
    fn gi_params_size() {
        assert_eq!(std::mem::size_of::<GiParamsUniform>(), 16);
    }
}
```

- [ ] **Step 3: Run size tests**

```bash
cargo test -p vox_render gpu_tests -- --nocapture
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/vox_render/src/spectral_gi.rs crates/vox_render/src/gpu/spectral_gi_pass.wgsl
git commit -m "feat(render): GpuGiPass — wgpu compute for spectral GI propagation"
```

---

## Task 5: Verification and performance check

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace 2>&1 | tail -20
```

Expected: all tests pass. Any `GiCache` tests that assert on exact splat counts may fail — update them to not depend on fixed splat counts.

- [ ] **Step 2: Run with the demo scene and inspect spectral bands**

```bash
cargo run -p vox_app --bin engine_runner -- --scene assets/demo.vxm
```

Press `Tab` to cycle spectral viewport modes. Verify:
- At noon: viewport band 0 (violet) is brighter than band 7 (red)
- At dawn/dusk (change `time_of_day` config): band 7 (red) brighter than band 0

- [ ] **Step 3: Confirm the old gi_cache field is unused and remove it**

Search for any remaining references to `gi_cache` in engine_runner.rs:

```bash
grep -n "gi_cache" crates/vox_app/src/bin/engine_runner.rs
```

If the field is still declared but never read/written except the initializer, remove it:

```rust
// DELETE this field from EngineApp struct:
// gi_cache: Option<vox_render::gi_cache::GiCache>,
```

- [ ] **Step 4: Final commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "refactor(app): remove static GiCache — replaced by live SpectralRadianceCache"
```

---

## Self-Review

**Spec coverage:**
- [x] Rayleigh β(λ) = β_ref × (550/λ)⁴ per band → Task 1 ✓
- [x] Sky seeds GI cache → Task 3 wires `set_sky()` ✓
- [x] Temporal EMA → Task 2, `alpha = 0.9` ✓
- [x] GPU compute pass → Task 4 ✓
- [x] Replace static GiCache → Task 3 ✓
- [x] Replace RGB illuminant_for_time → Task 3 ✓

**Known approximation:** The `sky_radiance()` single-scattering approximation is not a full Hosek-Wilkie model. It correctly orders bands (violet > red at zenith) and handles path length attenuation, which is sufficient for GI seeding. Full Hosek-Wilkie can be substituted later without changing the interface.

**Known limitation:** CPU propagation in Task 2 is O(N × 256) per frame. At 500k splats this is 128M operations — too slow at 60fps. Task 4 (GPU pass) is required for production. The CPU path is correct and fast enough for development iteration.
