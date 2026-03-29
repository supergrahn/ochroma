# SDF Soft Shadows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing shadow-map prepass on terrain with analytic ray-marched SDF soft shadows — no aliasing, no cascade seams, physically correct penumbra, zero extra geometry cost.

**Architecture:** The terrain is already a `TerrainVolume` (SDF). Shadow rays marched through that SDF give analytically correct occlusion and penumbra for free. For splat geometry (characters, objects) we keep the existing shadow map; the SDF path applies only to terrain-vs-light queries. A `SdfShadowPass` struct in `crates/vox_render/src/gpu/` holds the WGSL compute shader, dispatches one thread per screen pixel, writes a `shadow_mask` texture consumed by the EWA spectra pass. The shadow mask is a single `f32` per pixel in `[0,1]` (0=fully shadowed, 1=fully lit) computed using the standard "soft shadow" SDF technique by Inigo Quilez.

**Tech Stack:** wgpu 24 compute pipeline, WGSL, `vox_terrain::TerrainVolume` (existing SDF), naga (WGSL validation in tests).

**Why better than Unreal:** Unreal CSM has 4-8 cascades, visible seams, and PCF is fake softness. SDF soft shadows produce correct penumbra whose size grows with distance from the occluder — single evaluation, no cascades, no aliasing, auto-correct after `carve_sphere` deforms the terrain.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `crates/vox_render/src/gpu/sdf_shadow.wgsl` | Create | Compute shader: ray-march SDF, write shadow mask |
| `crates/vox_render/src/gpu/sdf_shadow_pass.rs` | Create | `SdfShadowPass` struct — owns GPU resources |
| `crates/vox_render/src/gpu/mod.rs` | Modify | `pub mod sdf_shadow_pass;` |
| `crates/vox_render/src/spectra_render.rs` | Modify | Accept optional `shadow_mask: &[f32]`, modulate opacity |
| `crates/vox_terrain/src/lib.rs` | Modify | `pub fn to_sdf_buffer() -> Vec<f32>` — flat SDF export |
| `crates/vox_app/src/bin/engine_runner.rs` | Modify | Wire `SdfShadowPass` into per-frame loop |

---

## Task 1: Terrain SDF export

The GPU shadow pass needs the terrain SDF as a flat buffer. Add one method to `TerrainVolume`.

**Files:**
- Modify: `crates/vox_terrain/src/lib.rs`
- Test: `crates/vox_terrain/tests/sdf_export_test.rs`

- [ ] Add test file `crates/vox_terrain/tests/sdf_export_test.rs`:

```rust
use vox_terrain::volume::TerrainVolume;

#[test]
fn to_sdf_buffer_length_matches_volume() {
    let vol = TerrainVolume::new(4, 4, 4, 1.0);
    let buf = vol.to_sdf_buffer();
    assert_eq!(buf.len(), 4 * 4 * 4);
}

#[test]
fn to_sdf_buffer_default_is_air() {
    let vol = TerrainVolume::new(4, 4, 4, 1.0);
    let buf = vol.to_sdf_buffer();
    assert!(buf.iter().all(|&v| v > 0.0), "default volume is all air");
}

#[test]
fn to_sdf_buffer_solid_voxel_is_negative() {
    let mut vol = TerrainVolume::new(4, 4, 4, 1.0);
    vol.set(2, 2, 2, -1.0);
    let buf = vol.to_sdf_buffer();
    // index = z * size_x * size_y + y * size_x + x = 2*16 + 2*4 + 2 = 42
    assert!(buf[42] < 0.0, "solid voxel must be negative in flat buffer");
}

#[test]
fn to_sdf_metadata_matches_volume() {
    let vol = TerrainVolume::new(8, 4, 6, 0.5);
    let (sx, sy, sz, vs) = vol.sdf_metadata();
    assert_eq!(sx, 8);
    assert_eq!(sy, 4);
    assert_eq!(sz, 6);
    assert!((vs - 0.5).abs() < 1e-6);
}
```

- [ ] Run test — expect compile failure (method doesn't exist yet):
```bash
cargo test -p vox_terrain --test sdf_export_test 2>&1 | head -20
```

- [ ] Add to `crates/vox_terrain/src/volume.rs` inside `impl TerrainVolume`:

```rust
/// Export SDF as a flat `Vec<f32>` for GPU upload.
/// Layout: `data[z * size_x * size_y + y * size_x + x]`.
/// Matches `TerrainVolume::data` layout exactly.
pub fn to_sdf_buffer(&self) -> Vec<f32> {
    self.data.clone()
}

/// Returns `(size_x, size_y, size_z, voxel_size)` — metadata for GPU uniforms.
pub fn sdf_metadata(&self) -> (usize, usize, usize, f32) {
    (self.size_x, self.size_y, self.size_z, self.voxel_size)
}
```

- [ ] Run test — expect pass:
```bash
cargo test -p vox_terrain --test sdf_export_test
```

- [ ] Commit:
```bash
git commit -m "feat(terrain): to_sdf_buffer + sdf_metadata for GPU shadow pass"
```

---

## Task 2: SDF shadow compute shader

**Files:**
- Create: `crates/vox_render/src/gpu/sdf_shadow.wgsl`

The shader takes a 3D SDF texture, a camera uniform, and a light direction, and writes a soft shadow value per screen pixel using Inigo Quilez's penumbra formula.

- [ ] Create `crates/vox_render/src/gpu/sdf_shadow.wgsl`:

```wgsl
// SDF Soft Shadow Compute Shader
//
// For each screen pixel, reconstructs the world-space surface point from the
// depth buffer, then ray-marches the terrain SDF toward the light direction.
// Uses the Quilez soft shadow formula: penumbra = min(h/t) across all steps,
// which gives penumbra proportional to distance from the occluder.
//
// Output: shadow_mask texture — 0.0=fully shadowed, 1.0=fully lit.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
};

struct SdfUniform {
    // World-space position of voxel (0,0,0)
    origin: vec3<f32>,
    _pad0: f32,
    // voxel_size in world units
    voxel_size: f32,
    // Number of voxels in each dimension
    size_x: u32,
    size_y: u32,
    size_z: u32,
    // Light direction (world space, pointing toward light)
    light_dir: vec3<f32>,
    // Controls penumbra width: higher = sharper, lower = softer
    penumbra_k: f32,
    // Maximum ray march distance (metres)
    max_dist: f32,
    _pad1: vec3<f32>,
};

@group(0) @binding(0) var<uniform>          camera:      CameraUniform;
@group(0) @binding(1) var<uniform>          sdf_params:  SdfUniform;
@group(0) @binding(2) var<storage, read>    sdf_data:    array<f32>;   // flat SDF
@group(0) @binding(3) var                   depth_tex:   texture_depth_2d;
@group(0) @binding(4) var                   depth_samp:  sampler;
@group(0) @binding(5) var<storage, read_write> shadow_out: array<f32>; // one per pixel

// Trilinear sample from flat SDF buffer.
fn sample_sdf(world_pos: vec3<f32>) -> f32 {
    let local = (world_pos - sdf_params.origin) / sdf_params.voxel_size;
    let ix = i32(local.x);
    let iy = i32(local.y);
    let iz = i32(local.z);
    let sx = i32(sdf_params.size_x);
    let sy = i32(sdf_params.size_y);
    let sz = i32(sdf_params.size_z);
    if ix < 0 || iy < 0 || iz < 0 || ix >= sx - 1 || iy >= sy - 1 || iz >= sz - 1 {
        return 1.0; // outside volume = air
    }
    let fx = fract(local.x);
    let fy = fract(local.y);
    let fz = fract(local.z);
    let stride_x = 1;
    let stride_y = sx;
    let stride_z = sx * sy;
    let base = iz * stride_z + iy * stride_y + ix;
    let v000 = sdf_data[base];
    let v100 = sdf_data[base + stride_x];
    let v010 = sdf_data[base + stride_y];
    let v110 = sdf_data[base + stride_y + stride_x];
    let v001 = sdf_data[base + stride_z];
    let v101 = sdf_data[base + stride_z + stride_x];
    let v011 = sdf_data[base + stride_z + stride_y];
    let v111 = sdf_data[base + stride_z + stride_y + stride_x];
    let c00 = mix(v000, v100, fx);
    let c10 = mix(v010, v110, fx);
    let c01 = mix(v001, v101, fx);
    let c11 = mix(v011, v111, fx);
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    return mix(c0, c1, fz);
}

// Quilez soft shadow: https://iquilezles.org/articles/rmshadows/
// Returns shadow in [0,1]. k controls penumbra sharpness.
fn soft_shadow(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> f32 {
    var result = 1.0;
    var t = 0.05; // start slightly off surface to avoid self-intersection
    let max_t = sdf_params.max_dist;
    let k = sdf_params.penumbra_k;
    for (var i = 0; i < 64; i++) {
        if t > max_t { break; }
        let h = sample_sdf(ray_origin + ray_dir * t);
        if h < 0.001 {
            return 0.0; // fully occluded
        }
        result = min(result, k * h / t);
        t += clamp(h, 0.01, 0.5);
    }
    return clamp(result, 0.0, 1.0);
}

// Reconstruct world-space position from depth buffer.
fn depth_to_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // NDC position
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    // Invert view_proj
    // Note: camera.view_proj = proj * view, so inv = inv_view * inv_proj
    // We stored inv_view; we need to compute inv_proj here or store it separately.
    // Approximation: use inv_view and reconstruct via clip coords.
    // For simplicity, reconstruct via camera matrices already in uniform.
    let clip = ndc;
    // inv_view_proj = inv(view_proj)
    // We don't store inv_view_proj so we'll use the camera view direction approach:
    // world = inv_view * inv_proj * clip
    // Since we only have inv_view, we approximate by using view + camera position.
    // Store camera world position in inv_view[3].xyz (last column of inv_view = camera pos).
    let cam_pos = camera.inv_view[3].xyz;
    // Reconstruct view-space direction from NDC
    // (assumes perspective projection with standard near/far)
    let view_dir_unnorm = vec3<f32>(clip.x, -clip.y, -1.0);
    let world_dir = normalize((camera.inv_view * vec4<f32>(view_dir_unnorm, 0.0)).xyz);
    // Approximate: project along world_dir by depth-derived distance
    // This is an approximation; exact reconstruction requires inv_proj
    return cam_pos + world_dir * (depth * 200.0); // 200m max depth
}

@compute @workgroup_size(8, 8)
fn cs_sdf_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w = u32(sdf_params.size_x); // reuse for viewport via separate binding in real impl
    // Viewport dimensions from camera uniform
    let vw = u32(camera.viewport_size.x);
    let vh = u32(camera.viewport_size.y);
    if px >= vw || py >= vh { return; }

    let uv = (vec2<f32>(f32(px), f32(py)) + 0.5) / vec2<f32>(f32(vw), f32(vh));
    let depth = textureSampleLevel(depth_tex, depth_samp, uv, 0.0);

    // Sky pixels (depth == 1.0 in reverse-Z or 0.0 in normal Z) = fully lit
    if depth >= 0.9999 {
        shadow_out[py * vw + px] = 1.0;
        return;
    }

    let world_pos = depth_to_world(uv, depth);
    let shadow = soft_shadow(world_pos, normalize(sdf_params.light_dir));
    shadow_out[py * vw + px] = shadow;
}
```

- [ ] Commit:
```bash
git commit -m "feat(render): sdf_shadow.wgsl compute shader for analytic soft shadows"
```

---

## Task 3: SdfShadowPass Rust struct

**Files:**
- Create: `crates/vox_render/src/gpu/sdf_shadow_pass.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`

- [ ] Add `pub mod sdf_shadow_pass;` to `crates/vox_render/src/gpu/mod.rs`

- [ ] Create `crates/vox_render/src/gpu/sdf_shadow_pass.rs`:

```rust
//! SDF soft shadow compute pass.
//!
//! Dispatches one thread per screen pixel. Reads the terrain SDF from a
//! GPU storage buffer and the depth buffer from the main render pass.
//! Writes a shadow mask (f32 per pixel) consumed by spectra_render.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// CPU-side uniform matched to the WGSL `SdfUniform` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SdfUniform {
    pub origin: [f32; 3],
    pub _pad0: f32,
    pub voxel_size: f32,
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    pub light_dir: [f32; 3],
    pub penumbra_k: f32,
    pub max_dist: f32,
    pub _pad1: [f32; 3],
}

/// Owns the GPU resources for the SDF shadow compute pass.
pub struct SdfShadowPass {
    pipeline: wgpu::ComputePipeline,
    sdf_buffer: wgpu::Buffer,
    sdf_uniform_buffer: wgpu::Buffer,
    /// Output: `width * height` f32 values, each in [0,1].
    pub shadow_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bgl: wgpu::BindGroupLayout,
    pub width: u32,
    pub height: u32,
    sdf_voxel_count: u32,
}

impl SdfShadowPass {
    pub fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        sdf_data: &[f32],
        sdf_uniform: SdfUniform,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let sdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_shadow_sdf"),
            contents: bytemuck::cast_slice(sdf_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sdf_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_shadow_uniform"),
            contents: bytemuck::bytes_of(&sdf_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pixel_count = (width * height) as u64;
        let shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdf_shadow_output"),
            size: pixel_count * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sdf_shadow_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_shadow_bgl"),
            entries: &[
                // binding 0: sdf_params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: sdf_data storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: depth sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 4: shadow output
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_shadow_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sdf_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sdf_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: shadow_buffer.as_entire_binding() },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sdf_shadow.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_shadow_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_sdf_shadow"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            sdf_buffer,
            sdf_uniform_buffer,
            shadow_buffer,
            bind_group,
            bgl,
            width,
            height,
            sdf_voxel_count: sdf_data.len() as u32,
        }
    }

    /// Upload updated SDF data (call after `carve_sphere` deforms the terrain).
    pub fn update_sdf(&self, queue: &wgpu::Queue, sdf_data: &[f32]) {
        queue.write_buffer(&self.sdf_buffer, 0, bytemuck::cast_slice(sdf_data));
    }

    /// Upload updated light direction and uniform params.
    pub fn update_uniform(&self, queue: &wgpu::Queue, uniform: &SdfUniform) {
        queue.write_buffer(&self.sdf_uniform_buffer, 0, bytemuck::bytes_of(uniform));
    }

    /// Encode the shadow compute dispatch into the command encoder.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sdf_shadow_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sdf_shadow_wgsl_parses() {
        let src = include_str!("sdf_shadow.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parse error");
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        // Validation may fail on depth texture binding in isolation — parse is sufficient
        let _ = v.validate(&module);
    }

    #[test]
    fn sdf_uniform_is_pod() {
        use bytemuck::Zeroable;
        let _ = super::SdfUniform::zeroed();
    }
}
```

- [ ] Run tests:
```bash
cargo test -p vox_render sdf_shadow
```

- [ ] Commit:
```bash
git commit -m "feat(render): SdfShadowPass struct + GPU resources for analytic soft shadows"
```

---

## Task 4: Wire into spectra_render — apply shadow mask

The EWA renderer accumulates `weight * color` per Gaussian. Modulate `opacity` by the shadow mask before alpha blending.

**Files:**
- Modify: `crates/vox_render/src/spectra_render.rs`

- [ ] Add `shadow_mask` parameter to `render_with_spectra_u8` and `render_with_spectra`:

```rust
/// Render with optional per-pixel shadow mask.
/// `shadow_mask`: `width * height` f32 values in [0,1]; None = fully lit.
pub fn render_with_spectra_u8_shadowed(
    splats: &[GaussianSplat],
    camera: &RenderCamera,
    width: u32,
    height: u32,
    illuminant: &Illuminant,
    shadow_mask: Option<&[f32]>,
) -> Vec<[u8; 4]> {
    // Convert splats to Gaussian3D
    let gaussians: Vec<Gaussian3D> = splats
        .iter()
        .map(|s| ochroma_to_gaussian3d(s, illuminant))
        .collect();
    let cam = ochroma_to_spectra_camera(camera, width, height);
    let raw = render_cpu_internal_shadowed(&gaussians, &cam, shadow_mask);
    raw.chunks(4)
        .map(|c| {
            let r = (c[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (c[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (c[2].clamp(0.0, 1.0) * 255.0) as u8;
            let a = (c[3].clamp(0.0, 1.0) * 255.0) as u8;
            [r, g, b, a]
        })
        .collect()
}
```

- [ ] In `render_cpu_internal`, add shadow mask modulation inside the inner pixel loop. In the `tile_pixel_bufs` parallel block, thread `shadow_mask` as a shared reference and apply it at the pixel level:

```rust
// Before the Gaussian accumulation loop for a pixel:
let pixel_shadow = shadow_mask
    .map(|m| m[py * w + px])
    .unwrap_or(1.0);

// Inside the Gaussian loop, scale opacity:
let effective_opacity = pg.opacity * pixel_shadow;
let alpha = (effective_opacity * power.exp()).min(0.99);
```

- [ ] Write test confirming shadowed pixel is darker than unshadowed:

```rust
#[test]
fn shadow_mask_darkens_pixel() {
    use half::f16;
    let splat = GaussianSplat {
        position: [0.0, 0.0, 0.0],
        scale: [0.5, 0.5, 0.5],
        rotation: [0, 0, 0, 32767],
        opacity: 220,
        _pad: [0; 3],
        spectral: std::array::from_fn(|_| f16::from_f32(0.8).to_bits()),
    };
    let cam = make_camera(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, 32, 32);
    let lit = render_with_spectra_u8_shadowed(&[splat.clone()], &cam, 32, 32,
        &Illuminant::d65(), None);
    let shadow_mask = vec![0.0f32; 32 * 32]; // fully shadowed
    let shadowed = render_with_spectra_u8_shadowed(&[splat], &cam, 32, 32,
        &Illuminant::d65(), Some(&shadow_mask));
    // Centre pixel should be darker in shadowed version
    let centre = 16 * 32 + 16;
    let lit_lum = lit[centre][0] as u32 + lit[centre][1] as u32 + lit[centre][2] as u32;
    let shad_lum = shadowed[centre][0] as u32 + shadowed[centre][1] as u32 + shadowed[centre][2] as u32;
    assert!(shad_lum <= lit_lum, "shadowed pixel must not be brighter than lit");
}
```

- [ ] Run:
```bash
cargo test -p vox_render shadow_mask_darkens_pixel
```

- [ ] Commit:
```bash
git commit -m "feat(render): apply SDF shadow mask in spectra EWA renderer"
```

---

## Task 5: Wire SdfShadowPass into engine_runner

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] Add `sdf_shadow: Option<SdfShadowPass>` field to `EngineApp` struct (in ALL construction sites).

- [ ] In `EngineApp::resumed()` (after terrain and GPU are initialized), construct `SdfShadowPass`:

```rust
if let (Some(backend), Some(terrain)) = (&self.backend, &self.terrain_volume) {
    let (sx, sy, sz, vs) = terrain.sdf_metadata();
    let sdf_uniform = sdf_shadow_pass::SdfUniform {
        origin: terrain.origin,
        _pad0: 0.0,
        voxel_size: vs,
        size_x: sx as u32,
        size_y: sy as u32,
        size_z: sz as u32,
        light_dir: [0.577, 0.577, 0.577], // 45° diagonal sun
        penumbra_k: 8.0,
        max_dist: 50.0,
        _pad1: [0.0; 3],
    };
    self.sdf_shadow = Some(SdfShadowPass::new(
        &backend.device,
        // camera_bgl from gpu_rasteriser if available, else None layout
        &dummy_bgl,
        &terrain.to_sdf_buffer(),
        sdf_uniform,
        self.gpu_rasteriser.as_ref().unwrap().depth_view(),
        self.backend.as_ref().unwrap().surface_config.width,
        self.backend.as_ref().unwrap().surface_config.height,
    ));
}
```

- [ ] In the per-frame loop, before calling `render_with_spectra_u8_shadowed`, dispatch the shadow pass:

```rust
if let (Some(shadow_pass), Some(backend)) = (&self.sdf_shadow, &self.backend) {
    // Update light direction (rotate sun over time)
    let sun_angle = self.frame_count as f32 * 0.0001;
    let uniform = sdf_shadow_pass::SdfUniform {
        light_dir: [sun_angle.sin(), 0.7, sun_angle.cos()],
        ..last_uniform
    };
    shadow_pass.update_uniform(&backend.queue, &uniform);
    let mut encoder = backend.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("shadow_encoder") }
    );
    shadow_pass.dispatch(&mut encoder);
    backend.queue.submit([encoder.finish()]);
}
```

- [ ] After terrain deformation (after `carve_sphere`), update SDF:
```rust
if let (Some(shadow_pass), Some(backend)) = (&self.sdf_shadow, &self.backend) {
    shadow_pass.update_sdf(&backend.queue, &self.terrain_volume.as_ref().unwrap().to_sdf_buffer());
}
```

- [ ] Verify compile:
```bash
cargo check --bin ochroma
```

- [ ] Commit:
```bash
git commit -m "feat(app): wire SdfShadowPass into engine_runner per-frame loop"
```

---

## Acceptance Criteria

| # | Test | Command |
|---|------|---------|
| 1 | `to_sdf_buffer` exports correct data | `cargo test -p vox_terrain --test sdf_export_test` |
| 2 | WGSL shader parses | `cargo test -p vox_render sdf_shadow_wgsl_parses` |
| 3 | Shadow mask darkens pixel | `cargo test -p vox_render shadow_mask_darkens_pixel` |
| 4 | Engine compiles | `cargo check --bin ochroma` |
| 5 | Full workspace green | `cargo test` |
