//! Hybrid mesh + Gaussian-splat compositor — GPU compute port of the CPU oracle
//! [`crate::hybrid_compose`] (`render_hybrid_lit` -> `rasterise_meshes` then
//! `composite_splats`).
//!
//! GPU is the foundation; the CPU [`crate::hybrid_compose::render_hybrid_lit`] is
//! the correctness ORACLE this mirrors bit-for-bit. One compute thread per pixel,
//! TWO DISPATCHES (mesh pass, then splat pass) sharing one depth buffer — the
//! exact two-pass structure of the oracle.
//!
//! Follows the [`crate::gpu::splat_rt_gpu::SplatRtGpu`] / [`crate::gpu::many_light_gpu`]
//! house pattern: construct-with-its-own-device, adapter-gated (never panics on a
//! missing GPU — returns [`HybridComposeGpuError::NoAdapter`]), `include_wgsl!`,
//! measured-then-asserted tolerances, bit-identical determinism.
//!
//! ## Host-side prep (identical-input principle)
//!
//! Two pieces of math are done ON THE HOST in Rust, byte-identical to the oracle,
//! and uploaded as pre-computed primitives — so the GPU cannot diverge on them:
//!
//! * **Triangle clipping + projection** — Sutherland–Hodgman near+far clipping,
//!   perspective projection, and screen-space fan into sub-triangles use code
//!   reproduced verbatim from [`crate::hybrid_compose`]'s private
//!   `clip_triangle_near` / `clip_polygon_far` / `to_camera_space` / `project_cam`
//!   / `edge` (those are not `pub`, and the oracle file is read-only, so they are
//!   re-derived here identically). We upload post-clip screen-space [`MeshTri`].
//! * **Splat projection** — splats are projected with the oracle's OWN
//!   [`project_gaussian`] (the same function the CPU path calls), filtered, and
//!   sorted front-to-back exactly as `composite_splats` does. We upload the
//!   resulting pre-projected [`SplatRec`] in that order. The shader never
//!   reprojects, so the EWA covariance / conic / radius math is shared verbatim.
//!
//! The WGSL reproduces ONLY the per-pixel rasterization (`mesh_pass`) and
//! compositing (`splat_pass`) inner loops. See `hybrid_compose_gpu.wgsl`.

use bytemuck::{Pod, Zeroable};
use half::f16;
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;

use spectra_gaussian_render::renderer::{
    project_gaussian, Gaussian3D, GaussianCamera, ALPHA_THRESHOLD, TRANSMITTANCE_THRESHOLD,
};

use crate::gpu::software_rasteriser::build_gaussian_camera;
use crate::hybrid_compose::{HybridScene, SunLight};
use crate::spectral::RenderCamera;

/// Dispatch params uniform (mirrors `Params` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    px_w: u32,
    px_h: u32,
    tri_count: u32,
    splat_count: u32,
}

const _: () = assert!(std::mem::size_of::<Params>() == 16);

/// One post-clip screen-space triangle for the GPU mesh pass. `a/b/c` are
/// `(screen_x, screen_y, cam_z)`; `inv_area2 = 1.0 / edge(a,b,c)`; `spectral` is
/// the already-shaded 16-band reflectance (4× `vec4` = 16 floats). 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshTri {
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    inv_area2: f32,
    _p0: f32,
    _p1: f32,
    spectral: [f32; 16],
}

const _: () = assert!(std::mem::size_of::<MeshTri>() == (9 + 3 + 16) * 4);

/// One pre-projected splat record (mirrors the CPU `SplatRecord`), already in
/// front-to-back (ascending depth) order. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SplatRec {
    screen_pos: [f32; 2],
    conic: [f32; 3],
    radius: f32,
    depth: f32,
    opacity: f32,
    spectral: [f32; 16],
}

const _: () = assert!(std::mem::size_of::<SplatRec>() == (2 + 3 + 1 + 1 + 1 + 16) * 4);

/// Error returned when the GPU compositor cannot be created or run. Never panics
/// on a missing/inadequate GPU — the caller can fall back to the CPU oracle.
#[derive(Debug, Clone)]
pub enum HybridComposeGpuError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed.
    DeviceCreation(String),
    /// Mapping the readback buffer failed.
    Readback(String),
}

impl std::fmt::Display for HybridComposeGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HybridComposeGpuError::NoAdapter => write!(f, "no GPU adapter available"),
            HybridComposeGpuError::DeviceCreation(e) => {
                write!(f, "GPU device creation failed: {e}")
            }
            HybridComposeGpuError::Readback(e) => write!(f, "GPU readback failed: {e}"),
        }
    }
}

impl std::error::Error for HybridComposeGpuError {}

/// The composed result of a GPU hybrid render: `width*height` pixels of 16-band
/// spectral plus the resolved per-pixel depth (cam_z), row-major (y from top).
#[derive(Debug, Clone)]
pub struct HybridGpuImage {
    pub width: u32,
    pub height: u32,
    /// 16 spectral bands per pixel, row-major.
    pub spectral: Vec<[f32; 16]>,
    /// Resolved depth (cam_z) per pixel; `f32::MAX` where nothing was drawn.
    pub depth: Vec<f32>,
}

/// Headless GPU hybrid mesh+splat compositor. Owns its own wgpu device/queue (no
/// window/surface). Sized for up to `max_tris` post-clip triangles, `max_splats`
/// splat records, and `max_pixels`.
pub struct HybridComposeGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mesh_pipeline: wgpu::ComputePipeline,
    splat_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    tri_buffer: wgpu::Buffer,
    splat_buffer: wgpu::Buffer,
    mesh_depth_buffer: wgpu::Buffer,
    mesh_spectral_buffer: wgpu::Buffer,
    out_spectral_buffer: wgpu::Buffer,
    out_depth_buffer: wgpu::Buffer,
    spec_readback: wgpu::Buffer,
    depth_readback: wgpu::Buffer,
    max_tris: u32,
    max_splats: u32,
    max_pixels: u32,
    /// Adapter human name, for diagnostics / benches.
    pub adapter_name: String,
}

impl HybridComposeGpu {
    /// Create a headless GPU compositor sized for up to `max_tris` post-clip
    /// triangles, `max_splats` splats, and `max_pixels` output pixels. Returns
    /// [`HybridComposeGpuError`] (never panics) if no adapter is found or device
    /// creation fails.
    pub fn new(
        max_tris: u32,
        max_splats: u32,
        max_pixels: u32,
    ) -> Result<Self, HybridComposeGpuError> {
        pollster::block_on(Self::new_async(max_tris, max_splats, max_pixels))
    }

    async fn new_async(
        max_tris: u32,
        max_splats: u32,
        max_pixels: u32,
    ) -> Result<Self, HybridComposeGpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(HybridComposeGpuError::NoAdapter)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hybrid_compose_gpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| HybridComposeGpuError::DeviceCreation(e.to_string()))?;

        let max_tris = max_tris.max(1);
        let max_splats = max_splats.max(1);
        let max_pixels = max_pixels.max(1);

        let tri_bytes = max_tris as u64 * std::mem::size_of::<MeshTri>() as u64;
        let splat_bytes = max_splats as u64 * std::mem::size_of::<SplatRec>() as u64;
        let spec_bytes = max_pixels as u64 * 16 * 4;
        let depth_bytes = max_pixels as u64 * 4;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tri_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_tris"),
            size: tri_bytes.max(96),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_splats"),
            size: splat_bytes.max(96),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mesh_depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_mesh_depth"),
            size: depth_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mesh_spectral_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_mesh_spectral"),
            size: spec_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_spectral_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_out_spectral"),
            size: spec_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_out_depth"),
            size: depth_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spec_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_spec_readback"),
            size: spec_bytes.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hybrid_depth_readback"),
            size: depth_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("hybrid_compose_gpu.wgsl"));

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid_bgl"),
            entries: &[
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
                storage_ro(1), // tris
                storage_ro(2), // splats
                storage_rw(3), // mesh_depth
                storage_rw(4), // mesh_spectral
                storage_rw(5), // out_spectral
                storage_rw(6), // out_depth
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let mesh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hybrid_mesh_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mesh_pass"),
            cache: None,
            compilation_options: Default::default(),
        });
        let splat_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hybrid_splat_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("splat_pass"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            mesh_pipeline,
            splat_pipeline,
            bgl,
            params_buffer,
            tri_buffer,
            splat_buffer,
            mesh_depth_buffer,
            mesh_spectral_buffer,
            out_spectral_buffer,
            out_depth_buffer,
            spec_readback,
            depth_readback,
            max_tris,
            max_splats,
            max_pixels,
            adapter_name,
        })
    }

    /// Render `scene` into a [`HybridGpuImage`] with default sun lighting,
    /// mirroring [`crate::hybrid_compose::render_hybrid`].
    pub fn render(
        &self,
        scene: &HybridScene,
        camera: &RenderCamera,
        illuminant: &Illuminant,
        width: u32,
        height: u32,
    ) -> Result<HybridGpuImage, HybridComposeGpuError> {
        self.render_lit(scene, camera, illuminant, &SunLight::default(), width, height)
    }

    /// Render `scene` with an explicit sun light, mirroring
    /// [`crate::hybrid_compose::render_hybrid_lit`].
    pub fn render_lit(
        &self,
        scene: &HybridScene,
        camera: &RenderCamera,
        illuminant: &Illuminant,
        sun: &SunLight,
        width: u32,
        height: u32,
    ) -> Result<HybridGpuImage, HybridComposeGpuError> {
        let _ = illuminant; // bands stay linear; illuminant applies at display (CPU parity).
        let pixels = (width as u64 * height as u64) as usize;
        if pixels == 0 {
            return Ok(HybridGpuImage {
                width,
                height,
                spectral: Vec::new(),
                depth: Vec::new(),
            });
        }
        assert!(
            (width * height) <= self.max_pixels,
            "render exceeds max_pixels: {} > {}",
            width * height,
            self.max_pixels
        );

        let gcam = build_gaussian_camera(camera, width as usize, height as usize);

        // Host-side prep: identical-input triangles and pre-projected splats.
        let tris = build_mesh_tris(&scene.meshes, &gcam, sun, width, height);
        let splats = build_splat_recs(scene.splats, &gcam);
        assert!(
            tris.len() as u32 <= self.max_tris,
            "scene exceeds max_tris: {} > {}",
            tris.len(),
            self.max_tris
        );
        assert!(
            splats.len() as u32 <= self.max_splats,
            "scene exceeds max_splats: {} > {}",
            splats.len(),
            self.max_splats
        );

        if !tris.is_empty() {
            self.queue
                .write_buffer(&self.tri_buffer, 0, bytemuck::cast_slice(&tris));
        }
        if !splats.is_empty() {
            self.queue
                .write_buffer(&self.splat_buffer, 0, bytemuck::cast_slice(&splats));
        }
        let params = Params {
            px_w: width,
            px_h: height,
            tri_count: tris.len() as u32,
            splat_count: splats.len() as u32,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tri_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.mesh_depth_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mesh_spectral_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.out_spectral_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.out_depth_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid_encoder"),
            });
        // Pass 1: mesh. Pass 2: splat. Two dispatches => pass 1 fully completes
        // (whole-image mesh depth) before pass 2 reads it — the oracle's order.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hybrid_mesh_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mesh_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hybrid_splat_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let spec_bytes = pixels as u64 * 16 * 4;
        let depth_bytes = pixels as u64 * 4;
        encoder.copy_buffer_to_buffer(
            &self.out_spectral_buffer,
            0,
            &self.spec_readback,
            0,
            spec_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.out_depth_buffer,
            0,
            &self.depth_readback,
            0,
            depth_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let spec_data = self.map_read(&self.spec_readback, spec_bytes)?;
        let spectral: Vec<[f32; 16]> = {
            let floats: &[f32] = bytemuck::cast_slice(&spec_data);
            (0..pixels)
                .map(|i| {
                    let mut px = [0.0f32; 16];
                    px.copy_from_slice(&floats[i * 16..i * 16 + 16]);
                    px
                })
                .collect()
        };
        self.spec_readback.unmap();

        let depth_data = self.map_read(&self.depth_readback, depth_bytes)?;
        let depth: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&depth_data)[..pixels].to_vec();
        self.depth_readback.unmap();

        Ok(HybridGpuImage {
            width,
            height,
            spectral,
            depth,
        })
    }

    fn map_read(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
    ) -> Result<Vec<u8>, HybridComposeGpuError> {
        let slice = buffer.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(HybridComposeGpuError::Readback(e.to_string())),
            Err(e) => return Err(HybridComposeGpuError::Readback(e.to_string())),
        }
        Ok(slice.get_mapped_range().to_vec())
    }
}

// ===========================================================================
// Host-side primitive prep — reproduced VERBATIM from the CPU oracle
// `crate::hybrid_compose` (its helpers are private + the file is read-only).
// Bit-identical Rust f32 code => bit-identical inputs to both paths.
// ===========================================================================

/// Mirror of the oracle's `normalised`.
fn normalised(d: [f32; 3]) -> [f32; 3] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if len > 1e-8 {
        [d[0] / len, d[1] / len, d[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Mirror of the oracle's `to_camera_space`.
fn to_camera_space(p: [f32; 3], gcam: &GaussianCamera) -> Option<[f32; 3]> {
    let v = &gcam.view_matrix;
    let cam_x = v[0] * p[0] + v[1] * p[1] + v[2] * p[2] + v[3];
    let cam_y = v[4] * p[0] + v[5] * p[1] + v[6] * p[2] + v[7];
    let cam_z = v[8] * p[0] + v[9] * p[1] + v[10] * p[2] + v[11];
    if !(cam_x.is_finite() && cam_y.is_finite() && cam_z.is_finite()) {
        return None;
    }
    Some([cam_x, cam_y, cam_z])
}

/// Mirror of the oracle's `project_cam`.
fn project_cam(cam: [f32; 3], gcam: &GaussianCamera) -> Option<(f32, f32, f32)> {
    let cam_z = cam[2];
    if cam_z < gcam.near || cam_z > gcam.far {
        return None;
    }
    let inv_z = 1.0 / cam_z;
    let sx = gcam.fx * cam[0] * inv_z + gcam.width as f32 * 0.5;
    let sy = gcam.fy * cam[1] * inv_z + gcam.height as f32 * 0.5;
    Some((sx, sy, cam_z))
}

/// Mirror of the oracle's `clip_triangle_near`.
fn clip_triangle_near(tri: [[f32; 3]; 3], near: f32) -> Vec<[f32; 3]> {
    let inside = |v: &[f32; 3]| v[2] >= near;
    let intersect = |a: &[f32; 3], b: &[f32; 3]| -> [f32; 3] {
        let t = (near - a[2]) / (b[2] - a[2]);
        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), near]
    };
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(4);
    for i in 0..3 {
        let cur = tri[i];
        let prev = tri[(i + 2) % 3];
        let cur_in = inside(&cur);
        let prev_in = inside(&prev);
        if cur_in {
            if !prev_in {
                out.push(intersect(&prev, &cur));
            }
            out.push(cur);
        } else if prev_in {
            out.push(intersect(&prev, &cur));
        }
    }
    out
}

/// Mirror of the oracle's `clip_polygon_far`.
fn clip_polygon_far(poly: &[[f32; 3]], far: f32) -> Vec<[f32; 3]> {
    let inside = |v: &[f32; 3]| v[2] <= far;
    let intersect = |a: &[f32; 3], b: &[f32; 3]| -> [f32; 3] {
        let t = (far - a[2]) / (b[2] - a[2]);
        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), far]
    };
    let n = poly.len();
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let cur = poly[i];
        let prev = poly[(i + n - 1) % n];
        let cur_in = inside(&cur);
        let prev_in = inside(&prev);
        if cur_in {
            if !prev_in {
                out.push(intersect(&prev, &cur));
            }
            out.push(cur);
        } else if prev_in {
            out.push(intersect(&prev, &cur));
        }
    }
    out
}

/// Mirror of the oracle's `edge`.
#[inline]
fn edge(a: (f32, f32), b: (f32, f32), px: f32, py: f32) -> f32 {
    (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
}

/// Reproduce `rasterise_meshes`' per-triangle setup (clip → project → fan →
/// shade) and emit only the screen-space sub-triangles the CPU actually
/// rasterizes (same area2 / bbox gates), each carrying its shaded spectrum.
fn build_mesh_tris(
    meshes: &[crate::hybrid_compose::HybridMesh],
    gcam: &GaussianCamera,
    sun: &SunLight,
    width: u32,
    height: u32,
) -> Vec<MeshTri> {
    let sun_dir = normalised(sun.direction);
    let ambient = sun.ambient.clamp(0.0, 1.0);
    let w_i32 = width as i32;
    let h_i32 = height as i32;
    let mut out: Vec<MeshTri> = Vec::new();

    for mesh in meshes {
        let nverts = mesh.positions.len();
        for tri in mesh.indices.chunks(3) {
            if tri.len() < 3 {
                continue;
            }
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= nverts || i1 >= nverts || i2 >= nverts {
                continue;
            }
            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];

            let finite = |v: [f32; 3]| v[0].is_finite() && v[1].is_finite() && v[2].is_finite();
            if !(finite(p0) && finite(p1) && finite(p2)) {
                continue;
            }

            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let nrm = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let nlen = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if nlen <= 1e-12 {
                continue;
            }
            let world_n = [nrm[0] / nlen, nrm[1] / nlen, nrm[2] / nlen];
            let ndl = sun_dir[0] * world_n[0] + sun_dir[1] * world_n[1] + sun_dir[2] * world_n[2];
            let diffuse = ndl.abs().clamp(0.0, 1.0);
            let shade = (ambient + (1.0 - ambient) * diffuse).clamp(0.0, 1.0);
            let shaded: [f32; 16] = std::array::from_fn(|k| mesh.reflectance[k] * shade);

            let (Some(ca), Some(cb), Some(cc)) = (
                to_camera_space(p0, gcam),
                to_camera_space(p1, gcam),
                to_camera_space(p2, gcam),
            ) else {
                continue;
            };
            let near_clipped = clip_triangle_near([ca, cb, cc], gcam.near);
            if near_clipped.len() < 3 {
                continue;
            }
            let clipped = clip_polygon_far(&near_clipped, gcam.far);
            if clipped.len() < 3 {
                continue;
            }

            let mut proj: Vec<(f32, f32, f32)> = Vec::with_capacity(clipped.len());
            let mut all_projected = true;
            for cv in &clipped {
                match project_cam(*cv, gcam) {
                    Some(s) => proj.push(s),
                    None => {
                        all_projected = false;
                        break;
                    }
                }
            }
            if !all_projected {
                continue;
            }

            // Fan: (0, i, i+1). Emit only sub-triangles the CPU rasterizes
            // (area2.abs() >= 1e-6 AND a non-empty clamped screen bbox).
            for i in 1..proj.len() - 1 {
                let a = proj[0];
                let b = proj[i];
                let c = proj[i + 1];
                let (sa, sb, sc) = ((a.0, a.1), (b.0, b.1), (c.0, c.1));
                let area2 = edge(sa, sb, sc.0, sc.1);
                if area2.abs() < 1e-6 {
                    continue;
                }
                let inv_area2 = 1.0 / area2;

                let min_x = (sa.0.min(sb.0).min(sc.0).floor() as i32).max(0);
                let max_x = (sa.0.max(sb.0).max(sc.0).ceil() as i32).min(w_i32 - 1);
                let min_y = (sa.1.min(sb.1).min(sc.1).floor() as i32).max(0);
                let max_y = (sa.1.max(sb.1).max(sc.1).ceil() as i32).min(h_i32 - 1);
                if min_x > max_x || min_y > max_y {
                    continue;
                }

                out.push(MeshTri {
                    a: [a.0, a.1, a.2],
                    b: [b.0, b.1, b.2],
                    c: [c.0, c.1, c.2],
                    inv_area2,
                    _p0: 0.0,
                    _p1: 0.0,
                    spectral: shaded,
                });
            }
        }
    }
    out
}

/// Reproduce `composite_splats`' projection + front-to-back sort, emitting the
/// records in the SAME order the CPU composites them. Uses the oracle's OWN
/// `project_gaussian`, so projection math is shared verbatim.
fn build_splat_recs(splats: &[GaussianSplat], gcam: &GaussianCamera) -> Vec<SplatRec> {
    let mut records: Vec<SplatRec> = Vec::with_capacity(splats.len());
    for splat in splats {
        let scales = splat.scales();
        let log_scale = [
            scales[0].max(1e-4).ln(),
            scales[1].max(1e-4).ln(),
            scales[2].max(1e-4).ln(),
        ];
        let q = splat.decoded_rotation();
        let g3d = Gaussian3D {
            position: splat.position(),
            log_scale,
            rotation: [q.w, q.x, q.y, q.z],
            color: [0.0, 0.0, 0.0],
            opacity: 1.0,
            sh_coeffs: None,
        };
        let Some(proj) = project_gaussian(&g3d, gcam) else {
            continue;
        };
        let spectral: [f32; 16] =
            std::array::from_fn(|i| f16::from_bits(splat.spectral()[i]).to_f32());
        records.push(SplatRec {
            screen_pos: proj.screen_pos,
            conic: proj.conic,
            radius: proj.radius,
            depth: proj.depth,
            opacity: splat.opacity() as f32 / 255.0,
            spectral,
        });
    }
    // Front-to-back (nearest first), exactly as the CPU sort.
    records.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    records
}

// Keep these thresholds referenced so the shared-constant intent is explicit and
// a future drift in the spectra crate trips a compile-time visibility here.
const _: f32 = ALPHA_THRESHOLD;
const _: f32 = TRANSMITTANCE_THRESHOLD;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_compose::{render_hybrid, render_hybrid_lit, HybridMesh};
    use crate::spectral_framebuffer::SpectralFramebuffer;
    use glam::{Mat4, Quat, Vec3};

    const W: u32 = 64;
    const H: u32 = 64;

    fn head_on_camera() -> RenderCamera {
        RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 0.0, 20.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, W as f32 / H as f32, 0.1, 500.0),
        }
    }

    fn illum() -> Illuminant {
        Illuminant::d65()
    }

    fn single_band(band: usize, value: f32) -> [u16; 16] {
        let mut s = [f16::from_f32(0.0).to_bits(); 16];
        s[band] = f16::from_f32(value).to_bits();
        s
    }

    fn single_band_f32(band: usize, value: f32) -> [f32; 16] {
        let mut s = [0.0f32; 16];
        s[band] = value;
        s
    }

    fn quad(z: f32, half: f32, refl: [f32; 16], object_id: u32) -> HybridMesh {
        HybridMesh {
            positions: vec![
                [-half, -half, z],
                [half, -half, z],
                [half, half, z],
                [-half, half, z],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            reflectance: refl,
            object_id,
        }
    }

    fn big_splat(z: f32, band: usize, value: f32, opacity: u8) -> GaussianSplat {
        GaussianSplat::volume(
            [0.0, 0.0, z],
            [3.0, 3.0, 3.0],
            Quat::IDENTITY,
            opacity,
            single_band(band, value),
        )
    }

    /// Skip a GPU test gracefully if this box truly has no GPU (CI without one).
    fn try_gpu(max_tris: u32, max_splats: u32, max_pixels: u32) -> Option<HybridComposeGpu> {
        match HybridComposeGpu::new(max_tris, max_splats, max_pixels) {
            Ok(g) => {
                eprintln!("[hybrid_compose_gpu test] adapter: {}", g.adapter_name);
                Some(g)
            }
            Err(HybridComposeGpuError::NoAdapter) => {
                eprintln!("[hybrid_compose_gpu test] no adapter — skipping GPU test");
                None
            }
            Err(e) => panic!("unexpected GPU init error on a box with a GPU: {e}"),
        }
    }

    /// Render the CPU oracle into a framebuffer (the literal oracle path).
    fn cpu_render(scene: &HybridScene, cam: &RenderCamera, il: &Illuminant) -> SpectralFramebuffer {
        let mut fb = SpectralFramebuffer::new(W, H);
        render_hybrid(scene, cam, il, &mut fb);
        fb
    }

    /// Max absolute / relative per-band deviation between CPU fb and GPU image.
    fn measure_dev(cpu: &SpectralFramebuffer, gpu: &HybridGpuImage) -> (f32, f32) {
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..(W * H) as usize {
            let c = cpu.spectral[i];
            let g = gpu.spectral[i];
            for b in 0..16 {
                let d = (c[b] - g[b]).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let denom = c[b].abs().max(1e-4);
                let rel = d / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        (max_abs, max_rel)
    }

    /// THE VALIDATION (house pattern): the oracle's own `occlusion_both_ways`
    /// scenes (mesh-occludes-splat-behind, splat-in-front) on BOTH paths, asserting
    /// per-pixel per-band agreement within a measured-then-reported tolerance.
    #[test]
    fn gpu_matches_cpu_occlusion_both_ways() {
        let cam = head_on_camera();
        let il = illum();
        let Some(gpu) = try_gpu(64, 4, (W * H) as u32) else { return };

        let wall = quad(0.0, 4.0, single_band_f32(11, 1.0), 7);

        // Splat behind the wall.
        let behind = big_splat(-8.0, 3, 1.0, 255);
        let scene_b = HybridScene {
            meshes: vec![wall.clone()],
            splats: std::slice::from_ref(&behind),
        };
        let cpu_b = cpu_render(&scene_b, &cam, &il);
        let gpu_b = gpu.render(&scene_b, &cam, &il, W, H).expect("gpu behind");
        let (abs_b, rel_b) = measure_dev(&cpu_b, &gpu_b);

        // Splat in front of the wall.
        let front = big_splat(8.0, 3, 1.0, 255);
        let scene_f = HybridScene {
            meshes: vec![wall],
            splats: std::slice::from_ref(&front),
        };
        let cpu_f = cpu_render(&scene_f, &cam, &il);
        let gpu_f = gpu.render(&scene_f, &cam, &il, W, H).expect("gpu front");
        let (abs_f, rel_f) = measure_dev(&cpu_f, &gpu_f);

        let max_abs = abs_b.max(abs_f);
        let max_rel = rel_b.max(rel_f);
        eprintln!(
            "[occlusion_both_ways] BEHIND abs={abs_b:e} rel={rel_b:e} | FRONT abs={abs_f:e} rel={rel_f:e}"
        );
        eprintln!(
            "[occlusion_both_ways] max_abs_dev={max_abs:e} max_rel_dev={max_rel:e} (asserting abs < 1e-4)"
        );
        // Identical f32 math; only the splat exp() and band reads remain — ULP.
        assert!(max_abs < 1e-4, "max abs band deviation {max_abs:e} exceeds 1e-4");
        assert!(max_rel < 1e-3, "max rel band deviation {max_rel:e} exceeds 1e-3");

        // Sanity: both paths lit something (and the front splat is present).
        let cpu_total: f32 = cpu_f.spectral.iter().flat_map(|p| p.iter()).sum();
        assert!(cpu_total > 0.0, "CPU oracle produced an empty image");
    }

    /// THE VALIDATION on the steep-triangle perspective-depth scene: the case
    /// where linear vs perspective-correct depth diverges, exercising the mesh
    /// pass' inv-z interpolation and the splat depth-test against it.
    #[test]
    fn gpu_matches_cpu_perspective_depth() {
        let cam = head_on_camera();
        let il = illum();
        let Some(gpu) = try_gpu(64, 4, (W * H) as u32) else { return };

        // Steep triangle apex cam_z~1 -> base cam_z~100 (the oracle's `steep_tri`).
        let tri = HybridMesh {
            positions: vec![
                [0.0, 0.0, 19.0],
                [-30.0, 20.0, -80.0],
                [30.0, 20.0, -80.0],
            ],
            indices: vec![0, 1, 2],
            reflectance: single_band_f32(11, 1.0),
            object_id: 1,
        };
        // Splat genuinely behind the true surface (cam_z 21.1) — must be rejected.
        let behind = GaussianSplat::volume(
            [0.0, -3.0, -1.1],
            [6.0, 6.0, 6.0],
            Quat::IDENTITY,
            255,
            single_band(3, 1.0),
        );
        let scene = HybridScene {
            meshes: vec![tri],
            splats: std::slice::from_ref(&behind),
        };
        let cpu = cpu_render(&scene, &cam, &il);
        let g = gpu.render(&scene, &cam, &il, W, H).expect("gpu render");
        let (max_abs, max_rel) = measure_dev(&cpu, &g);
        eprintln!(
            "[perspective_depth] max_abs_dev={max_abs:e} max_rel_dev={max_rel:e} (asserting abs < 1e-4)"
        );
        assert!(max_abs < 1e-4, "max abs band deviation {max_abs:e} exceeds 1e-4");
        assert!(max_rel < 1e-3, "max rel band deviation {max_rel:e} exceeds 1e-3");

        // The mesh surface (band 11) must actually be present in the apex region.
        let mesh_energy: f32 = cpu.spectral.iter().map(|p| p[11]).sum();
        assert!(mesh_energy > 1.0, "CPU oracle must light the steep mesh");
    }

    /// THE VALIDATION on the multi-triangle cube-over-splat-carpet end-to-end
    /// scene: many triangles (12 cube faces) + 49 splats, heavy overlap, both
    /// signatures, with CPU timing for the informational comparison.
    #[test]
    fn gpu_matches_cpu_cube_over_carpet() {
        let cam = RenderCamera {
            view: Mat4::look_at_rh(Vec3::new(0.0, 6.0, 18.0), Vec3::ZERO, Vec3::Y),
            proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, W as f32 / H as f32, 0.1, 500.0),
        };
        let il = illum();

        let cube = cube_mesh([0.0, 0.0, 0.0], 2.0, single_band_f32(11, 1.0), 1);
        let mut carpet = Vec::new();
        for gx in -3..=3 {
            for gz in -3..=3 {
                carpet.push(GaussianSplat::volume(
                    [gx as f32 * 2.0, -3.0, gz as f32 * 2.0],
                    [1.2, 0.2, 1.2],
                    Quat::IDENTITY,
                    220,
                    single_band(3, 1.0),
                ));
            }
        }
        let scene = HybridScene {
            meshes: vec![cube],
            splats: &carpet,
        };

        let Some(gpu) = try_gpu(64, carpet.len() as u32, (W * H) as u32) else { return };

        let t0 = std::time::Instant::now();
        let mut fb = SpectralFramebuffer::new(W, H);
        render_hybrid_lit(&scene, &cam, &il, &SunLight::default(), &mut fb);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Warm up (pipeline/first-submit overhead) then time a steady render.
        let _ = gpu.render(&scene, &cam, &il, W, H).expect("gpu warmup");
        let t1 = std::time::Instant::now();
        let g = gpu.render(&scene, &cam, &il, W, H).expect("gpu render");
        let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!("[timing] cube+49-splat 64x64: CPU={cpu_ms:.3}ms GPU(incl readback)={gpu_ms:.3}ms");

        let (max_abs, max_rel) = measure_dev(&fb, &g);
        eprintln!(
            "[cube_over_carpet] max_abs_dev={max_abs:e} max_rel_dev={max_rel:e} (asserting abs < 1e-4)"
        );
        assert!(max_abs < 1e-4, "max abs band deviation {max_abs:e} exceeds 1e-4");
        assert!(max_rel < 1e-3, "max rel band deviation {max_rel:e} exceeds 1e-3");

        // Both signatures present (proves real compositing happened, not zeros).
        let red: f32 = g.spectral.iter().map(|p| p[11]).sum();
        let blue: f32 = g.spectral.iter().map(|p| p[3]).sum();
        assert!(red > 1.0, "cube (band 11) must be present on GPU: {red}");
        assert!(blue > 1.0, "carpet (band 3) must be present on GPU: {blue}");
    }

    /// Determinism: two GPU renders of the same scene are bit-identical (spectral
    /// AND depth).
    #[test]
    fn gpu_is_deterministic() {
        let cam = head_on_camera();
        let il = illum();
        let Some(gpu) = try_gpu(64, 4, (W * H) as u32) else { return };

        let wall = quad(0.0, 4.0, single_band_f32(11, 1.0), 7);
        let front = big_splat(8.0, 3, 1.0, 255);
        let scene = HybridScene {
            meshes: vec![wall],
            splats: std::slice::from_ref(&front),
        };

        let a = gpu.render(&scene, &cam, &il, W, H).expect("render a");
        let b = gpu.render(&scene, &cam, &il, W, H).expect("render b");
        assert_eq!(a.spectral.len(), b.spectral.len());
        for (i, (pa, pb)) in a.spectral.iter().zip(b.spectral.iter()).enumerate() {
            for k in 0..16 {
                assert_eq!(
                    pa[k].to_bits(),
                    pb[k].to_bits(),
                    "GPU render must be bit-identical: pixel {i} band {k}: {} vs {}",
                    pa[k],
                    pb[k]
                );
            }
        }
        for (i, (da, db)) in a.depth.iter().zip(b.depth.iter()).enumerate() {
            assert_eq!(da.to_bits(), db.to_bits(), "depth must be bit-identical at {i}");
        }
    }

    /// Depth correctness vs the oracle: at the centre of the `depth_buffer_keeps_nearest`
    /// scene, GPU resolved depth equals the CPU framebuffer depth (near mesh cam_z).
    #[test]
    fn gpu_depth_matches_cpu() {
        let cam = head_on_camera();
        let il = illum();
        let Some(gpu) = try_gpu(64, 4, (W * H) as u32) else { return };

        let mesh = quad(5.0, 4.0, single_band_f32(11, 1.0), 1);
        let splat = big_splat(-5.0, 3, 1.0, 255);
        let scene = HybridScene {
            meshes: vec![mesh],
            splats: std::slice::from_ref(&splat),
        };
        let cpu = cpu_render(&scene, &cam, &il);
        let g = gpu.render(&scene, &cam, &il, W, H).expect("gpu render");

        let c = (H / 2 * W + W / 2) as usize;
        let cpu_d = cpu.depth[c];
        let gpu_d = g.depth[c];
        eprintln!("[depth] cpu={cpu_d:.6} gpu={gpu_d:.6}");
        assert!(
            (cpu_d - gpu_d).abs() < 1e-3,
            "GPU centre depth {gpu_d} must match CPU {cpu_d}"
        );
        // And it's the near mesh (~15), not the far splat (~25).
        assert!(gpu_d < 20.0, "near mesh depth must win on GPU: {gpu_d}");
    }

    fn cube_mesh(c: [f32; 3], h: f32, refl: [f32; 16], object_id: u32) -> HybridMesh {
        let [cx, cy, cz] = c;
        let positions = vec![
            [cx - h, cy - h, cz - h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz + h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx - h, cy + h, cz + h],
        ];
        let indices = vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 0, 4, 5, 0, 5,
            1, 3, 2, 6, 3, 6, 7,
        ];
        HybridMesh {
            positions,
            indices,
            reflectance: refl,
            object_id,
        }
    }
}


