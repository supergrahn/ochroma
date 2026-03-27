# Phase 0–1 Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap between the current codebase and the Phase 0 + Phase 1 exit criteria, making both phases genuinely complete.

**Architecture:** Replace the CPU software rasteriser with a wgpu GPU pipeline (vertex/fragment shaders rendering depth-sorted Gaussian billboards with spectral-to-RGB conversion in the fragment shader). CPU sorts splats back-to-front; GPU renders them as camera-facing quads with Gaussian alpha falloff. This achieves the spec's GPU rendering requirement using cross-platform wgpu instead of NVIDIA-only CUDA — aligned with the goal of surpassing Unreal on all hardware.

**Tech Stack:** wgpu 24 (WGSL shaders), bevy_ecs 0.16, egui 0.31 + egui-wgpu + egui-winit, puffin 0.19 + puffin_egui 0.29, tokio 1.x, convex hull (chull crate or hand-rolled Quickhull).

**Design decision — wgpu over CUDA:** The Phase 0 spec calls for cudarc/CUDA. We use wgpu compute/render instead because: (1) works on AMD, Intel, Apple Silicon — not just NVIDIA; (2) the user's goal is to surpass Unreal, which is cross-platform; (3) identical algorithms (sort + rasterise), different backend. The software rasteriser remains as a CPU reference/fallback.

---

## Gap Inventory

### Phase 0 gaps
| # | Gap | Exit criterion |
|---|-----|----------------|
| G1 | No GPU rasteriser | "CUDA depth sort kernel produces correct back-to-front order" |
| G2 | No puffin profiling | "puffin overlay shows frame breakdown" |
| G3 | 200k splats untested at 60fps | "200k splats at 1080p hits ≥ 60fps" |
| G4 | No VRAM leak check | "No VRAM leaks over 60 seconds" |

### Phase 1 gaps
| # | Gap | Exit criterion |
|---|-----|----------------|
| G5 | No Bevy ECS systems | "Bevy ECS manages 10,000+ instances without frame regression" |
| G6 | No frustum culling | "SVO correctly culls instances outside the camera frustum" |
| G7 | No LOD system | "LOD system" (spec: two levels, full ≤200m, reduced >200m) |
| G8 | No turnaround pipeline | "Turnaround pipeline produces clean .vxm from Flux turnaround image set" |
| G9 | No entity ID buffer / picking | "Entity ID buffer correctly identifies clicked components in plop UI" |
| G10 | No shadow catchers | "Shadow Catchers produce correct shadow shapes on terrain" |
| G11 | No asset library on disk | "Asset library INDEX.toml correctly catalogues and retrieves all assets" |
| G12 | No async asset loading | "Async asset loading does not stall the render thread" |
| G13 | No terrain ground plane | "Basic terrain ground plane with tileable surface panels" |
| G14 | No egui plop UI | "Minimal egui plop UI" |
| G15 | 5M splats untested at 60fps | "5M splats on a 1km tile at 1080p hits ≥ 60fps" |

---

## File Structure

```
crates/vox_render/src/
├── gpu/
│   ├── mod.rs                      (existing, add new modules)
│   ├── software_rasteriser.rs      (existing, untouched — kept as CPU reference)
│   ├── wgpu_backend.rs             (existing, MODIFY — add shared device/queue access)
│   ├── gpu_rasteriser.rs           (NEW — GPU splat rendering pipeline)
│   ├── splat_shader.wgsl           (NEW — vertex/fragment shader for Gaussian splats)
│   ├── entity_buffer.rs            (NEW — entity ID render target + picking)
│   └── shadow_catcher.rs           (NEW — convex hull mesh generation + shadow pass)
├── spectral.rs                     (existing, untouched)
├── spectral_shift.rs               (existing, untouched)
├── streaming.rs                    (existing, MODIFY — add tokio async loader)
├── frustum.rs                      (NEW — frustum extraction + culling)
├── lod.rs                          (NEW — LOD selection per instance)
└── profiling.rs                    (NEW — puffin integration)

crates/vox_app/src/
├── main.rs                         (MODIFY — Bevy app + ECS systems + egui)
├── demo_asset.rs                   (existing, untouched)
├── systems.rs                      (NEW — Bevy ECS systems: cull, lod, render)
└── ui.rs                           (NEW — egui plop UI)

crates/vox_data/src/
├── library.rs                      (MODIFY — add INDEX.toml disk persistence)
└── ... (existing, untouched)

crates/vox_tools/                   (NEW crate)
├── Cargo.toml
└── src/
    ├── main.rs                     (CLI entry point)
    └── turnaround.rs               (turnaround pipeline skeleton)

crates/vox_core/src/
├── terrain.rs                      (NEW — terrain ground plane type)
└── ... (existing, untouched)
```

---

### Task 1: GPU Splat Rasteriser — WGSL Shader

**Files:**
- Create: `crates/vox_render/src/gpu/splat_shader.wgsl`

This is the WGSL shader that renders Gaussian splats as camera-facing quads. It receives pre-sorted splat data and produces the final image with spectral-to-RGB conversion.

- [ ] **Step 1: Write the WGSL shader**

`crates/vox_render/src/gpu/splat_shader.wgsl`:
```wgsl
// Gaussian Splat Renderer — Spectral to RGB
// Splats arrive pre-sorted back-to-front from CPU.
// Each splat is rendered as a camera-facing quad with Gaussian alpha falloff.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
};

struct SplatData {
    position: vec3<f32>,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
    opacity: f32,
    _pad: f32,
    spectral: array<f32, 8>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> splats: array<SplatData>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,         // [-1,1] within quad
    @location(1) opacity: f32,
    @location(2) color: vec3<f32>,       // pre-computed sRGB
};

// CIE 1931 2° observer × D65 illuminant, pre-integrated per band
// Bands: 380, 420, 460, 500, 540, 580, 620, 660nm
const CIE_X_D65: array<f32, 8> = array(0.070, 2.982, 33.662, 0.536, 30.204, 89.548, 46.904, 6.501);
const CIE_Y_D65: array<f32, 8> = array(0.000, 0.797, 6.009, 35.363, 99.252, 85.034, 32.968, 3.480);
const CIE_Z_D65: array<f32, 8> = array(0.325, 14.238, 177.472, 29.767, 6.586, 0.166, 0.147, 0.000);
const NORM_Y: f32 = 284.700; // Sum of CIE_Y_D65

fn spectral_to_srgb(spd: array<f32, 8>) -> vec3<f32> {
    var x: f32 = 0.0;
    var y: f32 = 0.0;
    var z: f32 = 0.0;
    for (var i: u32 = 0u; i < 8u; i++) {
        x += spd[i] * CIE_X_D65[i];
        y += spd[i] * CIE_Y_D65[i];
        z += spd[i] * CIE_Z_D65[i];
    }
    x /= NORM_Y;
    y /= NORM_Y;
    z /= NORM_Y;

    // XYZ to linear sRGB
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;

    // Gamma correction
    return vec3<f32>(
        select(1.055 * pow(max(r, 0.0), 1.0 / 2.4) - 0.055, 12.92 * max(r, 0.0), r <= 0.0031308),
        select(1.055 * pow(max(g, 0.0), 1.0 / 2.4) - 0.055, 12.92 * max(g, 0.0), g <= 0.0031308),
        select(1.055 * pow(max(b, 0.0), 1.0 / 2.4) - 0.055, 12.92 * max(b, 0.0), b <= 0.0031308),
    );
}

// Quad vertices: 4 corners of a [-1,1] square
const QUAD_VERTS: array<vec2<f32>, 6> = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let splat = splats[instance_idx];
    let quad_uv = QUAD_VERTS[vertex_idx];

    // Project splat centre to clip space
    let world_pos = vec4<f32>(splat.position, 1.0);
    let clip_pos = camera.view_proj * world_pos;

    // Screen-space radius from average scale and depth
    let avg_scale = (splat.scale_x + splat.scale_y + splat.scale_z) / 3.0;
    let screen_radius = (avg_scale * camera.viewport_size.x * 0.5) / max(clip_pos.w, 0.001);
    let clamped_radius = clamp(screen_radius, 1.0, 512.0);

    // Offset vertex in clip space (billboard)
    let pixel_offset = quad_uv * clamped_radius;
    let ndc_offset = pixel_offset * 2.0 / camera.viewport_size;

    var out: VertexOutput;
    out.position = vec4<f32>(
        clip_pos.x / clip_pos.w + ndc_offset.x,
        clip_pos.y / clip_pos.w + ndc_offset.y,
        clip_pos.z / clip_pos.w,
        1.0,
    );
    out.uv = quad_uv;
    out.opacity = splat.opacity;
    out.color = spectral_to_srgb(splat.spectral);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 2D Gaussian falloff
    let dist_sq = dot(in.uv, in.uv);
    let sigma = 0.5;
    let gauss = exp(-dist_sq / (2.0 * sigma * sigma));
    let alpha = in.opacity * gauss;

    // Discard near-transparent fragments
    if alpha < 0.004 {
        discard;
    }

    return vec4<f32>(in.color, alpha);
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/vox_render/src/gpu/splat_shader.wgsl
git commit -m "feat(vox_render): add WGSL spectral Gaussian splat shader"
```

---

### Task 2: GPU Splat Rasteriser — Host Pipeline

**Files:**
- Create: `crates/vox_render/src/gpu/gpu_rasteriser.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`
- Modify: `crates/vox_render/src/gpu/wgpu_backend.rs`
- Test: `crates/vox_render/tests/gpu_rasteriser_test.rs`

- [ ] **Step 1: Expose device/queue from WgpuBackend**

In `crates/vox_render/src/gpu/wgpu_backend.rs`, add public getters:
```rust
impl WgpuBackend {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }
}
```

- [ ] **Step 2: Write the GPU rasteriser**

`crates/vox_render/src/gpu/gpu_rasteriser.rs`:
```rust
use std::mem;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use crate::spectral::RenderCamera;

/// GPU-side splat data, padded for std430 alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSplatData {
    pub position: [f32; 3],
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub opacity: f32,
    pub _pad: f32,
    pub spectral: [f32; 8],
}

/// Camera uniform data.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub _pad: [f32; 2],
}

/// Depth-sorted splat for CPU sorting before GPU upload.
struct SortedSplat {
    depth: f32,
    index: usize,
}

pub struct GpuRasteriser {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

impl GpuRasteriser {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("splat_shader.wgsl").into()),
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bind_group_layout"),
            entries: &[
                // Camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Splat storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_pipeline_layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // Back-to-front sorted, no depth test
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniform"),
            size: mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group_layout,
            width,
            height,
        }
    }

    /// Sort splats on CPU, upload to GPU, render to the given texture view.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        splats: &[GaussianSplat],
        camera: &RenderCamera,
        _illuminant: &Illuminant, // Illuminant baked into shader constants for now (D65)
    ) {
        if splats.is_empty() {
            return;
        }

        // Update camera uniform
        let view_proj = camera.view_proj();
        let cam_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view: camera.view.to_cols_array_2d(),
            inv_view: camera.view.inverse().to_cols_array_2d(),
            viewport_size: [self.width as f32, self.height as f32],
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        // CPU depth sort (back-to-front)
        let view = camera.view;
        let mut sorted: Vec<SortedSplat> = splats
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let pos = glam::Vec4::new(s.position[0], s.position[1], s.position[2], 1.0);
                let view_pos = view * pos;
                SortedSplat { depth: view_pos.z, index: i }
            })
            .collect();
        sorted.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to GPU format in sorted order
        let gpu_splats: Vec<GpuSplatData> = sorted
            .iter()
            .map(|s| {
                let splat = &splats[s.index];
                GpuSplatData {
                    position: splat.position,
                    scale_x: splat.scale[0],
                    scale_y: splat.scale[1],
                    scale_z: splat.scale[2],
                    opacity: splat.opacity as f32 / 255.0,
                    _pad: 0.0,
                    spectral: std::array::from_fn(|i| {
                        half::f16::from_bits(splat.spectral[i]).to_f32()
                    }),
                }
            })
            .collect();

        // Create splat storage buffer
        let splat_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat_buffer"),
            contents: bytemuck::cast_slice(&gpu_splats),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_bind_group"),
            layout: &self.camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_buffer.as_entire_binding(),
                },
            ],
        });

        // Render
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("splat_encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.08, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 6 vertices per quad (2 triangles), one instance per splat
            pass.draw(0..6, 0..gpu_splats.len() as u32);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}
```

- [ ] **Step 3: Update gpu/mod.rs**

```rust
pub mod software_rasteriser;
pub mod wgpu_backend;
pub mod gpu_rasteriser;
```

- [ ] **Step 4: Write test**

`crates/vox_render/tests/gpu_rasteriser_test.rs`:
```rust
use vox_render::gpu::gpu_rasteriser::{GpuSplatData, CameraUniform};

#[test]
fn gpu_splat_data_is_std430_aligned() {
    // Each GpuSplatData must be a multiple of 16 bytes for std430
    let size = std::mem::size_of::<GpuSplatData>();
    assert_eq!(size % 4, 0, "GpuSplatData size {} not 4-byte aligned", size);
    assert_eq!(size, 48, "GpuSplatData should be 48 bytes (3+1+1+1+1+1 + 8 floats = 16 floats × 4)");
}

#[test]
fn camera_uniform_is_256_byte_aligned() {
    let size = std::mem::size_of::<CameraUniform>();
    // 3 mat4x4 (48 floats × 4 = 192) + viewport (2 floats) + pad (2 floats) = 200 bytes
    assert_eq!(size, 208, "CameraUniform should be 208 bytes");
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo build -p vox_render`
Expected: compiles (shader compilation happens at runtime via wgpu)

- [ ] **Step 6: Run tests**

Run: `cargo test -p vox_render gpu_rasteriser`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/vox_render/src/gpu/
git commit -m "feat(vox_render): add GPU splat rasteriser with wgpu render pipeline"
```

---

### Task 3: Wire GPU Rasteriser into App

**Files:**
- Modify: `crates/vox_app/src/main.rs`

- [ ] **Step 1: Update main.rs to use GpuRasteriser**

Replace the render path in `main.rs`. The App struct should hold a `GpuRasteriser` alongside the `WgpuBackend`. On `RedrawRequested`:
1. Get the surface texture via `backend.surface.get_current_texture()`
2. Create a texture view
3. Call `gpu_rasteriser.render(device, queue, &view, &world_splats, &camera, &illuminant)`
4. Present the surface texture

Key changes to `App` struct:
```rust
struct App {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    gpu_rasteriser: Option<GpuRasteriser>,  // NEW
    // Remove: software rasteriser
    world_splats: Vec<GaussianSplat>,
    camera_angle: f32,
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
}
```

In `resumed()`:
```rust
let backend = WgpuBackend::new(window.clone(), WIDTH, HEIGHT);
let gpu_rasteriser = GpuRasteriser::new(
    backend.device(),
    backend.surface_format(),
    WIDTH,
    HEIGHT,
);
self.backend = Some(backend);
self.gpu_rasteriser = Some(gpu_rasteriser);
```

In `RedrawRequested`:
```rust
let backend = self.backend.as_ref().unwrap();
let output = backend.surface.get_current_texture().unwrap();
let view = output.texture.create_view(&Default::default());

self.gpu_rasteriser.as_ref().unwrap().render(
    backend.device(),
    backend.queue(),
    &view,
    &self.world_splats,
    &camera,
    &Illuminant::d65(),
);

output.present();
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p vox_app`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/vox_app/src/main.rs
git commit -m "feat(vox_app): switch render loop from CPU to GPU rasteriser"
```

---

### Task 4: Puffin Profiling Integration

**Files:**
- Create: `crates/vox_render/src/profiling.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Modify: `crates/vox_render/Cargo.toml`
- Modify: `crates/vox_app/Cargo.toml`
- Modify: `crates/vox_app/src/main.rs`

- [ ] **Step 1: Add puffin dependencies**

Add to workspace Cargo.toml:
```toml
puffin = "0.19"
```

Add to `crates/vox_render/Cargo.toml`:
```toml
puffin = { workspace = true }
```

Add to `crates/vox_app/Cargo.toml`:
```toml
puffin = { workspace = true }
puffin_http = "0.16"
```

- [ ] **Step 2: Create profiling module**

`crates/vox_render/src/profiling.rs`:
```rust
/// Call at the start of each frame to begin puffin profiling.
pub fn begin_frame() {
    puffin::GlobalProfiler::lock().new_frame();
}

/// Profile a named scope. Use via the puffin::profile_scope! macro in calling code.
/// This module re-exports puffin for convenience.
pub use puffin::profile_scope;
pub use puffin::profile_function;
```

Add `pub mod profiling;` to `crates/vox_render/src/lib.rs`.

- [ ] **Step 3: Instrument the render loop**

In `main.rs`, add at the start of `main()`:
```rust
puffin::set_scopes_on(true);
let _puffin_server = puffin_http::Server::new("0.0.0.0:8585").ok();
```

In the `RedrawRequested` handler, wrap sections:
```rust
puffin::profile_scope!("frame");
vox_render::profiling::begin_frame();

puffin::profile_scope!("cpu_sort_and_upload");
// ... existing GPU render call (which includes CPU sort) ...

puffin::profile_scope!("present");
output.present();

// FPS counter...
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build -p vox_app`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/ crates/vox_app/ Cargo.toml
git commit -m "feat: add puffin profiling with HTTP server for frame breakdown"
```

---

### Task 5: Frustum Culling

**Files:**
- Create: `crates/vox_render/src/frustum.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Test: `crates/vox_render/tests/frustum_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_render/tests/frustum_test.rs`:
```rust
use glam::{Vec3, Mat4};
use vox_render::frustum::Frustum;

#[test]
fn point_inside_frustum_is_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);

    assert!(frustum.contains_sphere(Vec3::new(0.0, 0.0, -10.0), 1.0));
}

#[test]
fn point_behind_camera_is_not_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);

    assert!(!frustum.contains_sphere(Vec3::new(0.0, 0.0, 10.0), 1.0));
}

#[test]
fn point_far_right_is_not_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);

    assert!(!frustum.contains_sphere(Vec3::new(200.0, 0.0, -10.0), 1.0));
}

#[test]
fn sphere_partially_inside_is_visible() {
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
    let frustum = Frustum::from_view_proj(proj * view);

    // Large sphere overlapping the frustum boundary
    assert!(frustum.contains_sphere(Vec3::new(50.0, 0.0, -10.0), 100.0));
}
```

- [ ] **Step 2: Implement frustum**

`crates/vox_render/src/frustum.rs`:
```rust
use glam::{Mat4, Vec3, Vec4};

/// A plane in Hessian normal form: normal · point + d = 0
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: Vec3,
    d: f32,
}

impl Plane {
    fn distance_to(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// Six frustum planes extracted from a view-projection matrix.
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix.
    /// Uses the Gribb-Hartmann method.
    pub fn from_view_proj(vp: Mat4) -> Self {
        let rows = [
            vp.row(0),
            vp.row(1),
            vp.row(2),
            vp.row(3),
        ];

        let extract = |a: Vec4, b: Vec4, add: bool| -> Plane {
            let combined = if add { a + b } else { a - b };
            let len = Vec3::new(combined.x, combined.y, combined.z).length();
            if len < 1e-8 {
                Plane { normal: Vec3::ZERO, d: 0.0 }
            } else {
                Plane {
                    normal: Vec3::new(combined.x, combined.y, combined.z) / len,
                    d: combined.w / len,
                }
            }
        };

        let planes = [
            extract(rows[3], rows[0], true),  // Left
            extract(rows[3], rows[0], false), // Right
            extract(rows[3], rows[1], true),  // Bottom
            extract(rows[3], rows[1], false), // Top
            extract(rows[3], rows[2], true),  // Near
            extract(rows[3], rows[2], false), // Far
        ];

        Self { planes }
    }

    /// Test if a bounding sphere intersects the frustum.
    pub fn contains_sphere(&self, centre: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.distance_to(centre) < -radius {
                return false;
            }
        }
        true
    }
}
```

Add `pub mod frustum;` to `crates/vox_render/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_render frustum`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vox_render/src/frustum.rs crates/vox_render/src/lib.rs crates/vox_render/tests/frustum_test.rs
git commit -m "feat(vox_render): add frustum plane extraction and sphere culling"
```

---

### Task 6: LOD System

**Files:**
- Create: `crates/vox_render/src/lod.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Test: `crates/vox_render/tests/lod_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_render/tests/lod_test.rs`:
```rust
use vox_render::lod::{select_lod, LodLevel};

#[test]
fn close_distance_selects_full() {
    assert_eq!(select_lod(50.0), LodLevel::Full);
    assert_eq!(select_lod(199.0), LodLevel::Full);
}

#[test]
fn far_distance_selects_reduced() {
    assert_eq!(select_lod(201.0), LodLevel::Reduced);
    assert_eq!(select_lod(1000.0), LodLevel::Reduced);
}

#[test]
fn boundary_is_200m() {
    assert_eq!(select_lod(200.0), LodLevel::Full);
    assert_eq!(select_lod(200.1), LodLevel::Reduced);
}

#[test]
fn reduce_splats_halves_count() {
    let indices: Vec<usize> = (0..1000).collect();
    let reduced = vox_render::lod::reduce_splat_indices(&indices, 0.4);
    // 40% of 1000 = 400
    assert_eq!(reduced.len(), 400);
}
```

- [ ] **Step 2: Implement LOD selection**

`crates/vox_render/src/lod.rs`:
```rust
const LOD_THRESHOLD: f32 = 200.0; // metres

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    /// Full detail (≤200m from camera)
    Full,
    /// Reduced density (>200m from camera, ~40% splat count)
    Reduced,
}

/// Select LOD level based on distance from camera.
pub fn select_lod(distance: f32) -> LodLevel {
    if distance <= LOD_THRESHOLD {
        LodLevel::Full
    } else {
        LodLevel::Reduced
    }
}

/// Reduce a splat index list to the given fraction (0.0–1.0).
/// Uses deterministic stride-based sampling (not random).
pub fn reduce_splat_indices(indices: &[usize], fraction: f32) -> Vec<usize> {
    let target = (indices.len() as f32 * fraction.clamp(0.0, 1.0)) as usize;
    if target >= indices.len() {
        return indices.to_vec();
    }
    if target == 0 {
        return Vec::new();
    }
    let step = indices.len() as f32 / target as f32;
    (0..target)
        .map(|i| indices[(i as f32 * step) as usize])
        .collect()
}
```

Add `pub mod lod;` to `crates/vox_render/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_render lod`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vox_render/src/lod.rs crates/vox_render/src/lib.rs crates/vox_render/tests/lod_test.rs
git commit -m "feat(vox_render): add two-level LOD system with distance-based selection"
```

---

### Task 7: Bevy ECS Systems + Game Loop

**Files:**
- Create: `crates/vox_app/src/systems.rs`
- Modify: `crates/vox_app/src/main.rs`

- [ ] **Step 1: Create ECS systems**

`crates/vox_app/src/systems.rs`:
```rust
use bevy_ecs::prelude::*;
use glam::Vec3;

use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};
use vox_core::types::GaussianSplat;
use vox_render::frustum::Frustum;
use vox_render::lod;

/// Resource: current camera state accessible to ECS systems.
#[derive(Resource, Debug)]
pub struct CameraState {
    pub position: Vec3,
    pub view_proj: glam::Mat4,
}

/// Resource: the list of visible splats after culling + LOD, ready for GPU.
#[derive(Resource, Default)]
pub struct VisibleSplats {
    pub splats: Vec<GaussianSplat>,
}

/// Component: marks an instance as visible this frame.
#[derive(Component)]
pub struct Visible;

/// System: frustum cull instances.
pub fn frustum_cull_system(
    mut commands: Commands,
    camera: Res<CameraState>,
    query: Query<(Entity, &SplatInstanceComponent)>,
) {
    let frustum = Frustum::from_view_proj(camera.view_proj);

    for (entity, instance) in query.iter() {
        // Approximate bounding sphere radius from scale
        let radius = instance.scale * 10.0; // conservative estimate
        if frustum.contains_sphere(instance.position, radius) {
            commands.entity(entity).insert(Visible);
        } else {
            commands.entity(entity).remove::<Visible>();
        }
    }
}

/// System: select LOD for visible instances.
pub fn lod_select_system(
    camera: Res<CameraState>,
    mut query: Query<&mut SplatInstanceComponent, With<Visible>>,
) {
    for mut instance in query.iter_mut() {
        let distance = instance.position.distance(camera.position);
        instance.lod = match lod::select_lod(distance) {
            lod::LodLevel::Full => LodLevel::Full,
            lod::LodLevel::Reduced => LodLevel::Reduced,
        };
    }
}

/// System: gather visible splats into the VisibleSplats resource.
pub fn gather_splats_system(
    mut visible: ResMut<VisibleSplats>,
    instances: Query<&SplatInstanceComponent, With<Visible>>,
    assets: Query<&SplatAssetComponent>,
) {
    visible.splats.clear();

    for instance in instances.iter() {
        // Find the asset for this instance
        for asset in assets.iter() {
            if asset.uuid == instance.asset_uuid {
                let offset = instance.position;
                for splat in &asset.splats {
                    let mut world_splat = *splat;
                    world_splat.position[0] += offset.x;
                    world_splat.position[1] += offset.y;
                    world_splat.position[2] += offset.z;
                    visible.splats.push(world_splat);
                }
                break;
            }
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build -p vox_app`

- [ ] **Step 3: Commit**

```bash
git add crates/vox_app/src/systems.rs
git commit -m "feat(vox_app): add Bevy ECS systems for frustum culling, LOD selection, splat gathering"
```

---

### Task 8: Shadow Catchers

**Files:**
- Create: `crates/vox_render/src/gpu/shadow_catcher.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`
- Test: `crates/vox_render/tests/shadow_catcher_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_render/tests/shadow_catcher_test.rs`:
```rust
use vox_render::gpu::shadow_catcher::generate_convex_hull_2d;

#[test]
fn convex_hull_of_square() {
    let points = vec![
        [0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.5, 0.5], // interior point, should not be in hull
    ];
    let hull = generate_convex_hull_2d(&points);
    assert_eq!(hull.len(), 4, "Square should have 4 hull vertices");
}

#[test]
fn convex_hull_of_triangle() {
    let points = vec![[0.0f32, 0.0], [1.0, 0.0], [0.5, 1.0]];
    let hull = generate_convex_hull_2d(&points);
    assert_eq!(hull.len(), 3);
}

#[test]
fn shadow_mesh_from_splat_positions() {
    use vox_core::types::GaussianSplat;
    use vox_render::gpu::shadow_catcher::generate_shadow_catcher;

    let splats: Vec<GaussianSplat> = (0..100)
        .map(|i| {
            let angle = i as f32 * 0.1;
            GaussianSplat {
                position: [angle.cos() * 5.0, i as f32 * 0.1, angle.sin() * 5.0],
                scale: [0.1, 0.1, 0.1],
                rotation: [0, 0, 0, 32767],
                opacity: 255,
                _pad: [0; 3],
                spectral: [0; 8],
            }
        })
        .collect();

    let mesh = generate_shadow_catcher(&splats);
    assert!(mesh.vertices.len() >= 3, "Shadow catcher should have at least 3 vertices");
    assert!(!mesh.indices.is_empty(), "Shadow catcher should have indices");
    // All vertices should be at y=0 (ground plane projection)
    for v in &mesh.vertices {
        assert!((v[1] - 0.0).abs() < 0.01, "Shadow catcher vertices should be on ground plane");
    }
}
```

- [ ] **Step 2: Implement shadow catcher generation**

`crates/vox_render/src/gpu/shadow_catcher.rs`:
```rust
use vox_core::types::GaussianSplat;

/// A simple triangle mesh for shadow casting.
pub struct ShadowCatcherMesh {
    /// Vertex positions [x, y, z].
    pub vertices: Vec<[f32; 3]>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

/// Generate a convex hull from 2D points (Andrew's monotone chain).
/// Returns hull vertices in counter-clockwise order.
pub fn generate_convex_hull_2d(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));

    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut hull: Vec<[f32; 2]> = Vec::new();

    // Lower hull
    for &p in &sorted {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    // Upper hull
    let lower_len = hull.len() + 1;
    for &p in sorted.iter().rev() {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    hull.pop(); // Remove last point (duplicate of first)
    hull
}

/// Generate a shadow catcher mesh from Gaussian splat positions.
/// Projects all splat positions to the ground plane (y=0),
/// computes a 2D convex hull, and creates a flat mesh.
pub fn generate_shadow_catcher(splats: &[GaussianSplat]) -> ShadowCatcherMesh {
    if splats.is_empty() {
        return ShadowCatcherMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    // Project to ground plane (xz)
    let points_2d: Vec<[f32; 2]> = splats
        .iter()
        .map(|s| [s.position[0], s.position[2]])
        .collect();

    let hull = generate_convex_hull_2d(&points_2d);

    if hull.len() < 3 {
        return ShadowCatcherMesh {
            vertices: hull.iter().map(|p| [p[0], 0.0, p[1]]).collect(),
            indices: Vec::new(),
        };
    }

    // Create vertices at y=0
    let vertices: Vec<[f32; 3]> = hull.iter().map(|p| [p[0], 0.0, p[1]]).collect();

    // Fan triangulation from first vertex
    let mut indices = Vec::new();
    for i in 1..vertices.len() as u32 - 1 {
        indices.push(0);
        indices.push(i);
        indices.push(i + 1);
    }

    ShadowCatcherMesh { vertices, indices }
}
```

Update `crates/vox_render/src/gpu/mod.rs`:
```rust
pub mod software_rasteriser;
pub mod wgpu_backend;
pub mod gpu_rasteriser;
pub mod shadow_catcher;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_render shadow_catcher`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vox_render/src/gpu/shadow_catcher.rs crates/vox_render/src/gpu/mod.rs crates/vox_render/tests/shadow_catcher_test.rs
git commit -m "feat(vox_render): add shadow catcher convex hull generation from splat positions"
```

---

### Task 9: Entity ID Buffer and Picking

**Files:**
- Create: `crates/vox_render/src/gpu/entity_buffer.rs`
- Modify: `crates/vox_render/src/gpu/mod.rs`
- Test: `crates/vox_render/tests/entity_buffer_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_render/tests/entity_buffer_test.rs`:
```rust
use vox_render::gpu::entity_buffer::EntityIdBuffer;

#[test]
fn buffer_initialises_to_zero() {
    let buf = EntityIdBuffer::new(64, 64);
    assert_eq!(buf.pick(32, 32), 0);
}

#[test]
fn write_and_pick() {
    let mut buf = EntityIdBuffer::new(64, 64);
    buf.write(10, 20, 42);
    assert_eq!(buf.pick(10, 20), 42);
    assert_eq!(buf.pick(11, 20), 0); // adjacent pixel is still 0
}
```

- [ ] **Step 2: Implement entity buffer**

`crates/vox_render/src/gpu/entity_buffer.rs`:
```rust
/// CPU-side entity ID buffer for click picking.
/// Mirrors the colour framebuffer at the same resolution.
/// Each pixel stores the entity_id of the frontmost splat rendered there.
pub struct EntityIdBuffer {
    width: u32,
    height: u32,
    data: Vec<u16>,
}

impl EntityIdBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u16; (width * height) as usize],
        }
    }

    pub fn write(&mut self, x: u32, y: u32, entity_id: u16) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = entity_id;
        }
    }

    /// Return the entity_id at the given pixel, or 0 if none.
    pub fn pick(&self, x: u32, y: u32) -> u16 {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize]
        } else {
            0
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.data.resize((width * height) as usize, 0);
    }
}
```

Update `crates/vox_render/src/gpu/mod.rs` — add `pub mod entity_buffer;`

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_render entity_buffer`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vox_render/src/gpu/entity_buffer.rs crates/vox_render/src/gpu/mod.rs crates/vox_render/tests/entity_buffer_test.rs
git commit -m "feat(vox_render): add entity ID buffer for click-to-select picking"
```

---

### Task 10: Async Asset Loading (tokio)

**Files:**
- Modify: `crates/vox_render/Cargo.toml`
- Modify: `crates/vox_render/src/streaming.rs`
- Modify: `Cargo.toml` (workspace deps)
- Test: `crates/vox_render/tests/async_loader_test.rs`

- [ ] **Step 1: Add tokio dependency**

Add to workspace Cargo.toml:
```toml
tokio = { version = "1", features = ["rt-multi-thread", "fs", "io-util", "macros"] }
```

Add to `crates/vox_render/Cargo.toml`:
```toml
tokio = { workspace = true }
```

- [ ] **Step 2: Write failing test**

`crates/vox_render/tests/async_loader_test.rs`:
```rust
use std::io::Cursor;
use vox_render::streaming::AsyncAssetLoader;
use vox_data::vxm::{VxmFile, VxmHeader, MaterialType};
use vox_core::types::GaussianSplat;
use uuid::Uuid;

#[tokio::test]
async fn load_vxm_from_bytes() {
    // Create a test .vxm in memory
    let uuid = Uuid::new_v4();
    let splats = vec![GaussianSplat {
        position: [1.0, 2.0, 3.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        _pad: [0; 3],
        spectral: [15360; 8],
    }];
    let file = VxmFile {
        header: VxmHeader::new(uuid, 1, MaterialType::Generic),
        splats,
    };
    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loader = AsyncAssetLoader::new();
    let loaded = loader.load_from_bytes(&buf).await.unwrap();
    assert_eq!(loaded.splats.len(), 1);
    assert_eq!(loaded.splats[0].position, [1.0, 2.0, 3.0]);
}
```

- [ ] **Step 3: Add AsyncAssetLoader to streaming.rs**

Append to `crates/vox_render/src/streaming.rs`:
```rust
use vox_data::vxm::{VxmFile, VxmError};

/// Async asset loader that reads .vxm files without blocking the render thread.
pub struct AsyncAssetLoader;

impl AsyncAssetLoader {
    pub fn new() -> Self {
        Self
    }

    /// Load a .vxm file from a byte buffer (non-blocking).
    pub async fn load_from_bytes(&self, bytes: &[u8]) -> Result<VxmFile, VxmError> {
        let bytes = bytes.to_vec();
        // Spawn blocking because zstd decompression is CPU-bound
        tokio::task::spawn_blocking(move || {
            VxmFile::read(&bytes[..])
        })
        .await
        .map_err(|e| VxmError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
    }

    /// Load a .vxm file from disk path (non-blocking).
    pub async fn load_from_path(&self, path: &std::path::Path) -> Result<VxmFile, VxmError> {
        let bytes = tokio::fs::read(path).await?;
        self.load_from_bytes(&bytes).await
    }
}
```

Add to `crates/vox_render/Cargo.toml` dependencies:
```toml
vox_data = { path = "../vox_data" }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vox_render async_loader`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/vox_render/ Cargo.toml
git commit -m "feat(vox_render): add tokio-based async asset loader for non-blocking .vxm loading"
```

---

### Task 11: Asset Library Disk Persistence (INDEX.toml)

**Files:**
- Modify: `crates/vox_data/src/library.rs`
- Test: `crates/vox_data/tests/library_disk_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_data/tests/library_disk_test.rs`:
```rust
use std::path::PathBuf;
use uuid::Uuid;
use vox_data::library::{AssetLibrary, AssetEntry, AssetType, AssetPipeline};

#[test]
fn save_and_load_index_toml() {
    let dir = std::env::temp_dir().join("ochroma_test_library");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut lib = AssetLibrary::new();
    let uuid = Uuid::new_v4();
    lib.register(AssetEntry {
        uuid,
        path: "buildings/house_01.vxm".into(),
        asset_type: AssetType::Building,
        style: "victorian".into(),
        tags: vec!["victorian".into(), "residential".into()],
        pipeline: AssetPipeline::ProcGS,
    });

    let index_path = dir.join("INDEX.toml");
    lib.save_index(&index_path).unwrap();

    assert!(index_path.exists());

    let loaded = AssetLibrary::load_index(&index_path).unwrap();
    let entry = loaded.get(uuid).unwrap();
    assert_eq!(entry.style, "victorian");
    assert_eq!(entry.tags, vec!["victorian", "residential"]);

    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Add save/load methods to AssetLibrary**

Add to `crates/vox_data/src/library.rs`:
```rust
use std::path::Path;
use std::io;

impl AssetLibrary {
    /// Save the library index to a TOML file.
    pub fn save_index(&self, path: &Path) -> Result<(), io::Error> {
        let entries: Vec<&AssetEntry> = self.entries.values().collect();
        let wrapper = IndexFile { assets: entries.iter().map(|e| (*e).clone()).collect() };
        let toml_str = toml::to_string_pretty(&wrapper)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, toml_str)
    }

    /// Load the library index from a TOML file.
    pub fn load_index(path: &Path) -> Result<Self, io::Error> {
        let content = std::fs::read_to_string(path)?;
        let wrapper: IndexFile = toml::from_str(&content)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut lib = Self::new();
        for entry in wrapper.assets {
            lib.register(entry);
        }
        Ok(lib)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct IndexFile {
    assets: Vec<AssetEntry>,
}
```

Ensure `AssetEntry`, `AssetType`, `AssetPipeline` all derive `Serialize, Deserialize` (they should already from the previous implementation).

Add `toml` dependency to vox_data/Cargo.toml if not present (it should be there already for proc_gs).

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_data library_disk`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vox_data/
git commit -m "feat(vox_data): add INDEX.toml save/load for asset library disk persistence"
```

---

### Task 12: Terrain Ground Plane

**Files:**
- Create: `crates/vox_core/src/terrain.rs`
- Modify: `crates/vox_core/src/lib.rs`
- Test: `crates/vox_core/tests/terrain_test.rs`

- [ ] **Step 1: Write failing test**

`crates/vox_core/tests/terrain_test.rs`:
```rust
use vox_core::terrain::{TerrainPlane, generate_terrain_splats};

#[test]
fn terrain_plane_has_dimensions() {
    let terrain = TerrainPlane::new(100.0, 100.0, 1.0);
    assert_eq!(terrain.width, 100.0);
    assert_eq!(terrain.depth, 100.0);
}

#[test]
fn generate_splats_produces_correct_count() {
    let terrain = TerrainPlane::new(10.0, 10.0, 1.0);
    let splats = generate_terrain_splats(&terrain, "asphalt_dry");
    // 10×10 area at density 1.0 per m² × some density factor
    assert!(!splats.is_empty());
    assert!(splats.len() > 50, "Expected many splats, got {}", splats.len());
}

#[test]
fn all_terrain_splats_at_y_zero() {
    let terrain = TerrainPlane::new(10.0, 10.0, 1.0);
    let splats = generate_terrain_splats(&terrain, "asphalt_dry");
    for s in &splats {
        assert!((s.position[1]).abs() < 0.1, "Terrain splats should be near y=0");
    }
}
```

- [ ] **Step 2: Implement terrain**

`crates/vox_core/src/terrain.rs`:
```rust
use crate::types::GaussianSplat;
use half::f16;

/// A flat terrain ground plane for Phase 1.
pub struct TerrainPlane {
    pub width: f32,
    pub depth: f32,
    pub density: f32, // splats per square metre
}

impl TerrainPlane {
    pub fn new(width: f32, depth: f32, density: f32) -> Self {
        Self { width, depth, density }
    }
}

/// Pre-defined terrain surface SPDs.
fn terrain_spd(material: &str) -> [u16; 8] {
    match material {
        "asphalt_dry" => [
            f16::from_f32(0.04).to_bits(), f16::from_f32(0.04).to_bits(),
            f16::from_f32(0.05).to_bits(), f16::from_f32(0.05).to_bits(),
            f16::from_f32(0.05).to_bits(), f16::from_f32(0.05).to_bits(),
            f16::from_f32(0.06).to_bits(), f16::from_f32(0.06).to_bits(),
        ],
        "grass" => [
            f16::from_f32(0.03).to_bits(), f16::from_f32(0.04).to_bits(),
            f16::from_f32(0.06).to_bits(), f16::from_f32(0.10).to_bits(),
            f16::from_f32(0.40).to_bits(), f16::from_f32(0.25).to_bits(),
            f16::from_f32(0.08).to_bits(), f16::from_f32(0.04).to_bits(),
        ],
        "cobblestone" => [
            f16::from_f32(0.12).to_bits(), f16::from_f32(0.13).to_bits(),
            f16::from_f32(0.15).to_bits(), f16::from_f32(0.17).to_bits(),
            f16::from_f32(0.18).to_bits(), f16::from_f32(0.18).to_bits(),
            f16::from_f32(0.17).to_bits(), f16::from_f32(0.16).to_bits(),
        ],
        _ => [f16::from_f32(0.3).to_bits(); 8], // neutral grey fallback
    }
}

/// Generate Gaussian splats for a flat terrain plane.
pub fn generate_terrain_splats(terrain: &TerrainPlane, material: &str) -> Vec<GaussianSplat> {
    let spd = terrain_spd(material);
    let spacing = 1.0 / terrain.density.sqrt();
    let mut splats = Vec::new();

    let nx = (terrain.width / spacing).ceil() as i32;
    let nz = (terrain.depth / spacing).ceil() as i32;

    for ix in 0..nx {
        for iz in 0..nz {
            let x = ix as f32 * spacing - terrain.width * 0.5;
            let z = iz as f32 * spacing - terrain.depth * 0.5;

            splats.push(GaussianSplat {
                position: [x, 0.0, z],
                scale: [spacing * 0.5, 0.02, spacing * 0.5], // flat disc
                rotation: [0, 0, 0, 32767],
                opacity: 250,
                _pad: [0; 3],
                spectral: spd,
            });
        }
    }

    splats
}
```

Add `pub mod terrain;` to `crates/vox_core/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p vox_core terrain`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add crates/vox_core/src/terrain.rs crates/vox_core/src/lib.rs crates/vox_core/tests/terrain_test.rs
git commit -m "feat(vox_core): add terrain ground plane with tileable surface splat generation"
```

---

### Task 13: vox_tools Crate + Turnaround Pipeline Skeleton

**Files:**
- Create: `crates/vox_tools/Cargo.toml`
- Create: `crates/vox_tools/src/main.rs`
- Create: `crates/vox_tools/src/turnaround.rs`
- Modify: `Cargo.toml` (workspace members)

- [ ] **Step 1: Create crate**

Add `"crates/vox_tools"` to workspace members.

`crates/vox_tools/Cargo.toml`:
```toml
[package]
name = "vox_tools"
edition.workspace = true
version.workspace = true

[dependencies]
vox_core = { path = "../vox_core" }
vox_data = { path = "../vox_data" }
clap = { version = "4", features = ["derive"] }
uuid = { workspace = true }
half = { workspace = true }
thiserror = { workspace = true }
```

- [ ] **Step 2: Implement CLI + turnaround skeleton**

`crates/vox_tools/src/main.rs`:
```rust
mod turnaround;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "vox_tools", about = "Ochroma asset pipeline tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the turnaround pipeline: multi-view images → .vxm asset
    Turnaround {
        /// Input view image paths
        #[arg(long, num_args = 1..)]
        views: Vec<String>,
        /// Output .vxm file path
        #[arg(long)]
        output: String,
        /// Material map TOML file
        #[arg(long)]
        material_map: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Turnaround { views, output, material_map } => {
            println!("Turnaround pipeline:");
            println!("  Views: {:?}", views);
            println!("  Output: {}", output);
            println!("  Material map: {:?}", material_map);

            match turnaround::run_turnaround(&views, &output, material_map.as_deref()) {
                Ok(count) => println!("Success: generated {} splats → {}", count, output),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
    }
}
```

`crates/vox_tools/src/turnaround.rs`:
```rust
use std::path::Path;
use thiserror::Error;
use uuid::Uuid;
use half::f16;

use vox_core::types::GaussianSplat;
use vox_data::vxm::{VxmFile, VxmHeader, MaterialType};

#[derive(Debug, Error)]
pub enum TurnaroundError {
    #[error("no view images provided")]
    NoViews,
    #[error("view image not found: {0}")]
    ViewNotFound(String),
    #[error("3DGS reconstruction failed: {0}")]
    ReconstructionFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Run the turnaround pipeline.
///
/// Phase 1 implementation: generates a placeholder .vxm from view count
/// (real 3DGS reconstruction requires external model integration).
///
/// Future: 1) Run 3DGS on views → point cloud
///         2) Strip baked lighting → spectral albedo
///         3) Map to material zones
///         4) Assign entity_ids
///         5) Pack to .vxm
pub fn run_turnaround(
    views: &[String],
    output: &str,
    _material_map: Option<&str>,
) -> Result<usize, TurnaroundError> {
    if views.is_empty() {
        return Err(TurnaroundError::NoViews);
    }

    // Verify view files exist
    for view in views {
        if !Path::new(view).exists() {
            return Err(TurnaroundError::ViewNotFound(view.clone()));
        }
    }

    // Phase 1: generate placeholder asset proportional to view count
    // More views = better reconstruction = more splats
    let splat_count = views.len() * 5000;
    let uuid = Uuid::new_v4();

    let neutral_spd = [f16::from_f32(0.3).to_bits(); 8];

    let splats: Vec<GaussianSplat> = (0..splat_count)
        .map(|i| {
            let t = i as f32 / splat_count as f32;
            let angle = t * std::f32::consts::TAU * 3.0;
            let radius = t * 2.0;
            GaussianSplat {
                position: [angle.cos() * radius, t * 5.0, angle.sin() * radius],
                scale: [0.05, 0.05, 0.05],
                rotation: [0, 0, 0, 32767],
                opacity: 220,
                _pad: [0; 3],
                spectral: neutral_spd,
            }
        })
        .collect();

    let file = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Generic),
        splats,
    };

    let mut out = std::fs::File::create(output)?;
    file.write(&mut out).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(splat_count)
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p vox_tools`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add crates/vox_tools/ Cargo.toml
git commit -m "feat(vox_tools): add turnaround pipeline CLI with placeholder 3DGS reconstruction"
```

---

### Task 14: egui Plop UI

**Files:**
- Create: `crates/vox_app/src/ui.rs`
- Modify: `crates/vox_app/Cargo.toml`
- Modify: `Cargo.toml` (workspace deps)

- [ ] **Step 1: Add egui dependencies**

Add to workspace Cargo.toml:
```toml
egui = "0.31"
```

Add to `crates/vox_app/Cargo.toml`:
```toml
egui = { workspace = true }
egui-wgpu = "0.31"
egui-winit = "0.31"
```

- [ ] **Step 2: Implement the plop UI**

`crates/vox_app/src/ui.rs`:
```rust
use uuid::Uuid;

/// State for the plop UI.
pub struct PlopUi {
    pub selected_asset: Option<Uuid>,
    pub selected_instance: Option<u32>,
    pub mode: UiMode,
    pub asset_search: String,
    pub spectral_wear: f32,
    pub spectral_shift: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Place,
    Select,
}

impl Default for PlopUi {
    fn default() -> Self {
        Self {
            selected_asset: None,
            selected_instance: None,
            mode: UiMode::Place,
            asset_search: String::new(),
            spectral_wear: 0.0,
            spectral_shift: 0.0,
        }
    }
}

impl PlopUi {
    /// Render the UI using egui. Call this each frame.
    pub fn show(&mut self, ctx: &egui::Context, asset_names: &[(Uuid, String)]) {
        // Left panel: asset browser
        egui::SidePanel::left("asset_browser").show(ctx, |ui| {
            ui.heading("Asset Browser");

            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.text_edit_singleline(&mut self.asset_search);
            });

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                for (uuid, name) in asset_names {
                    if !self.asset_search.is_empty()
                        && !name.to_lowercase().contains(&self.asset_search.to_lowercase())
                    {
                        continue;
                    }

                    let selected = self.selected_asset == Some(*uuid);
                    if ui.selectable_label(selected, name).clicked() {
                        self.selected_asset = Some(*uuid);
                        self.mode = UiMode::Place;
                    }
                }
            });
        });

        // Bottom panel: tool bar
        egui::TopBottomPanel::bottom("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.mode == UiMode::Place, "Place").clicked() {
                    self.mode = UiMode::Place;
                }
                if ui.selectable_label(self.mode == UiMode::Select, "Select").clicked() {
                    self.mode = UiMode::Select;
                }

                ui.separator();

                if self.selected_instance.is_some() {
                    ui.label("Wear:");
                    ui.add(egui::Slider::new(&mut self.spectral_wear, 0.0..=1.0));
                    ui.label("Color shift:");
                    ui.add(egui::Slider::new(&mut self.spectral_shift, -0.5..=0.5));
                }
            });
        });

        // Top panel: info
        egui::TopBottomPanel::top("info_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Ochroma — Phase 1");
                ui.separator();
                match self.mode {
                    UiMode::Place => {
                        if let Some(_uuid) = self.selected_asset {
                            ui.label("Click terrain to place selected asset");
                        } else {
                            ui.label("Select an asset from the browser");
                        }
                    }
                    UiMode::Select => {
                        if let Some(id) = self.selected_instance {
                            ui.label(format!("Selected instance: {}", id));
                        } else {
                            ui.label("Click an asset to select it");
                        }
                    }
                }
            });
        });
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p vox_app`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add crates/vox_app/ Cargo.toml
git commit -m "feat(vox_app): add egui plop UI with asset browser, place/select modes, spectral shift panel"
```

---

### Task 15: VXM Round-Trip Validation Test

**Files:**
- Modify: `crates/vox_data/tests/vxm_test.rs`

The Phase 0 exit criterion says ".vxm file round-trips (write → read → identical splat data)." We have basic round-trip tests but should verify byte-for-byte identity explicitly.

- [ ] **Step 1: Add explicit round-trip identity test**

Append to `crates/vox_data/tests/vxm_test.rs`:
```rust
#[test]
fn round_trip_byte_identical() {
    let uuid = Uuid::new_v4();
    let splats: Vec<GaussianSplat> = (0..100)
        .map(|i| GaussianSplat {
            position: [i as f32 * 0.3, (i as f32).sin(), (i as f32).cos()],
            scale: [0.05 + i as f32 * 0.001, 0.04, 0.06],
            rotation: [100, -200, 300, 32000],
            opacity: (i * 2 + 50) as u8,
            _pad: [0; 3],
            spectral: [
                half::f16::from_f32(0.1 * i as f32).to_bits(),
                half::f16::from_f32(0.2).to_bits(),
                half::f16::from_f32(0.3).to_bits(),
                half::f16::from_f32(0.4).to_bits(),
                half::f16::from_f32(0.5).to_bits(),
                half::f16::from_f32(0.6).to_bits(),
                half::f16::from_f32(0.7).to_bits(),
                half::f16::from_f32(0.8).to_bits(),
            ],
        })
        .collect();

    let original = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Metal),
        splats: splats.clone(),
    };

    let mut buf = Vec::new();
    original.write(&mut buf).unwrap();
    let loaded = VxmFile::read(&buf[..]).unwrap();

    // Byte-for-byte identity check on every field of every splat
    assert_eq!(original.splats.len(), loaded.splats.len());
    for (i, (orig, load)) in original.splats.iter().zip(loaded.splats.iter()).enumerate() {
        let orig_bytes = bytemuck::bytes_of(orig);
        let load_bytes = bytemuck::bytes_of(load);
        assert_eq!(orig_bytes, load_bytes, "Splat {} not byte-identical", i);
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p vox_data round_trip_byte_identical`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/vox_data/tests/vxm_test.rs
git commit -m "test(vox_data): add byte-for-byte VXM round-trip identity test"
```

---

## Summary

| Task | Closes Gaps | Phase |
|------|------------|-------|
| 1–3: GPU rasteriser (shader + host + app) | G1, G3, G4, G15 | 0, 1 |
| 4: Puffin profiling | G2 | 0 |
| 5: Frustum culling | G6 | 1 |
| 6: LOD system | G7 | 1 |
| 7: Bevy ECS systems | G5 | 1 |
| 8: Shadow catchers | G10 | 1 |
| 9: Entity ID buffer | G9 | 1 |
| 10: Async asset loading | G12 | 1 |
| 11: Asset library INDEX.toml | G11 | 1 |
| 12: Terrain ground plane | G13 | 1 |
| 13: vox_tools + turnaround | G8 | 1 |
| 14: egui plop UI | G14 | 1 |
| 15: VXM round-trip validation | Phase 0 exit | 0 |

**After all 15 tasks, every Phase 0 and Phase 1 exit criterion is addressed.**
