//! Spectral splat ray tracer — GPU compute port of the CPU oracle
//! [`crate::splat_rt`].
//!
//! GPU is the foundation; the CPU [`crate::splat_rt::render_orthographic`] /
//! [`crate::splat_rt::trace_ray`] (`bvh = None` brute-force path) is the
//! correctness ORACLE this mirrors bit-for-bit. One compute thread per pixel
//! loops every splat, computes the per-splat `(t_peak, alpha)` with EXACTLY the
//! CPU math, gathers the BUDGET (64) nearest hits, insertion-sorts by `t_peak`
//! in-shader, and front-to-back over-composites 16 spectral bands + alpha.
//!
//! Follows the [`crate::spectral_gi::GpuGi`] house pattern: construct-with-its-
//! own-device, then `render(scene, camera) -> Vec<[f32; 17]>`. Never panics on a
//! missing GPU — returns [`SplatRtGpuError::NoAdapter`] so callers can fall back
//! to the CPU oracle.
//!
//! ## Raw splat upload — the 96-byte `GaussianSplat` POD
//!
//! `GaussianSplat` is `#[repr(C)]` + `Pod` and exactly 96 bytes (asserted
//! below), so we upload it VERBATIM with `bytemuck` and decode it in-shader.
//! The WGSL sees it as `array<u32, 24>` and unpacks: `i16` quat via
//! sign-extending arithmetic shifts (`i32(w<<16)>>16` for the low half), `u16`
//! f16 spectral via `unpack2x16float`, `u8` opacity via `word15 & 0xFF`. The
//! full byte map lives in the shader header (`splat_rt_gpu.wgsl`).

use bytemuck::{Pod, Zeroable};
use vox_core::types::GaussianSplat;

use crate::splat_rt::{OrthoCamera, RtScene, BANDS};

/// We upload the raw 96-byte `GaussianSplat` and decode in-shader, so the
/// shader's view (`array<u32,24>`) must match the host struct size exactly.
const _: () = assert!(std::mem::size_of::<GaussianSplat>() == 96);
const _: () = assert!(std::mem::size_of::<GaussianSplat>() == 24 * 4);

/// Output floats per pixel: 16 spectral bands + 1 alpha.
pub const PIXEL_FLOATS: usize = BANDS + 1; // 17

/// Camera + dispatch uniform. `std140`-friendly: every `vec3` is padded to
/// `vec4`. Mirrors the fields [`crate::splat_rt::render_orthographic`] consumes
/// (forward/right/up are pre-normalized on the host, exactly as the CPU does).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    eye: [f32; 3],
    _p0: f32,
    forward: [f32; 3],
    _p1: f32,
    right: [f32; 3],
    _p2: f32,
    up: [f32; 3],
    _p3: f32,
    width: f32,
    height: f32,
    px_w: u32,
    px_h: u32,
    splat_count: u32,
    _p4: u32,
    _p5: u32,
    _p6: u32,
}

const _: () = assert!(std::mem::size_of::<CameraUniform>() == 96);

/// Error returned when the GPU ray tracer cannot be created or run. Never
/// panics on a missing/inadequate GPU — the caller can fall back to the CPU
/// oracle.
#[derive(Debug, Clone)]
pub enum SplatRtGpuError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed.
    DeviceCreation(String),
    /// Mapping the readback buffer failed.
    Readback(String),
    /// A required storage buffer would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`). Returned instead
    /// of letting wgpu raise an uncaptured Validation Error that aborts the
    /// process — so the caller can fall back to the CPU oracle. `what` names the
    /// offending buffer, `requested` is the byte size we'd need, `limit` is the
    /// device's cap.
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for SplatRtGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SplatRtGpuError::NoAdapter => write!(f, "no GPU adapter available"),
            SplatRtGpuError::DeviceCreation(e) => write!(f, "GPU device creation failed: {e}"),
            SplatRtGpuError::Readback(e) => write!(f, "GPU readback failed: {e}"),
            SplatRtGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => write!(
                f,
                "GPU buffer '{what}' requires {requested} bytes, exceeding device limit {limit}"
            ),
        }
    }
}

impl std::error::Error for SplatRtGpuError {}

/// Headless GPU spectral splat ray tracer. Owns its own wgpu device/queue (no
/// window/surface). Sized for up to `max_splats` and `max_pixels` per render.
pub struct SplatRtGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    splat_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    max_splats: u32,
    max_pixels: u32,
    /// `device.limits().max_storage_buffer_binding_size`, captured at creation so
    /// `render()` can validate every storage binding range against it and return
    /// [`SplatRtGpuError::ExceedsDeviceLimits`] instead of letting wgpu raise an
    /// uncaptured Validation Error that aborts the process.
    max_storage_binding: u64,
    /// Adapter human name, for diagnostics / benches.
    pub adapter_name: String,
}

impl SplatRtGpu {
    /// Create a headless GPU ray tracer sized for up to `max_splats` splats and
    /// `max_pixels` output pixels. Returns [`SplatRtGpuError`] (never panics) if
    /// no adapter is found or device creation fails.
    pub fn new(max_splats: u32, max_pixels: u32) -> Result<Self, SplatRtGpuError> {
        pollster::block_on(Self::new_async(max_splats, max_pixels))
    }

    async fn new_async(max_splats: u32, max_pixels: u32) -> Result<Self, SplatRtGpuError> {
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
            .ok_or(SplatRtGpuError::NoAdapter)?;
        let info = adapter.get_info();
        crate::gpu::adapter::ensure_hardware(&info).map_err(|_| SplatRtGpuError::NoAdapter)?;
        let adapter_name = info.name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("splat_rt_gpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| SplatRtGpuError::DeviceCreation(e.to_string()))?;

        let max_splats = max_splats.max(1);
        let max_pixels = max_pixels.max(1);

        let splat_bytes = max_splats as u64 * std::mem::size_of::<GaussianSplat>() as u64;
        let out_bytes = max_pixels as u64 * PIXEL_FLOATS as u64 * 4;

        // Validate the up-front buffer sizing against the device's storage-binding
        // and total-buffer limits BEFORE creating buffers / binding them. wgpu
        // raises an uncaptured Validation Error (which aborts the process) if a
        // bound range exceeds `max_storage_buffer_binding_size`, so we must catch
        // it here and honor the documented no-panic contract. The splat and out
        // buffers are both bound as STORAGE, so each must fit the binding limit;
        // both must also fit `max_buffer_size`.
        let limits = device.limits();
        let max_storage_binding = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        for (what, bytes) in [("splat_buffer", splat_bytes), ("out_buffer", out_bytes)] {
            if bytes > max_storage_binding {
                return Err(SplatRtGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage_binding,
                });
            }
            if bytes > max_buffer {
                return Err(SplatRtGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_rt_splats"),
            size: splat_bytes.max(96),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_rt_camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_rt_out"),
            size: out_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_rt_readback"),
            size: out_bytes.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("splat_rt_gpu.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_rt_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_rt_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_rt_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            splat_buffer,
            camera_buffer,
            out_buffer,
            readback_buffer,
            max_splats,
            max_pixels,
            max_storage_binding,
            adapter_name,
        })
    }

    /// Render `scene` with one orthographic ray per pixel, mirroring
    /// [`crate::splat_rt::render_orthographic`]. Returns `width*height` pixels
    /// in row-major order (y from top), each `[16 bands, alpha]`.
    ///
    /// `scene.bvh` is ignored — this is the brute-force `bvh = None` oracle path.
    /// The hard budget is fixed at 64 in-shader (the CPU's hard budget); the
    /// `budget` argument to the CPU renderer is matched by that constant.
    pub fn render(
        &self,
        scene: &RtScene,
        camera: &OrthoCamera,
        width: u32,
        height: u32,
    ) -> Result<Vec<[f32; PIXEL_FLOATS]>, SplatRtGpuError> {
        let pixels = (width as u64 * height as u64) as usize;
        if pixels == 0 {
            return Ok(Vec::new());
        }
        // The documented contract is "never panics — returns SplatRtGpuError so
        // callers can fall back to the CPU oracle". The former `assert!`s here
        // were panics that broke that contract, so they are routed through the
        // error channel instead. We also validate the actual per-render storage
        // binding ranges (out path: W*H*PIXEL_FLOATS*4; splat path: n*96) against
        // `max_storage_buffer_binding_size` — the limit wgpu would otherwise trip
        // with an uncaptured, process-aborting Validation Error in
        // `create_bind_group`.
        if (width * height) > self.max_pixels {
            return Err(SplatRtGpuError::ExceedsDeviceLimits {
                what: "out_buffer (render exceeds max_pixels)",
                requested: (width as u64 * height as u64) * PIXEL_FLOATS as u64 * 4,
                limit: self.max_pixels as u64 * PIXEL_FLOATS as u64 * 4,
            });
        }
        let out_range = pixels as u64 * PIXEL_FLOATS as u64 * 4;
        if out_range > self.max_storage_binding {
            return Err(SplatRtGpuError::ExceedsDeviceLimits {
                what: "out_buffer",
                requested: out_range,
                limit: self.max_storage_binding,
            });
        }
        let count = (scene.splats.len() as u32).min(self.max_splats);
        let n = count as usize;
        if scene.splats.len() as u32 > self.max_splats {
            return Err(SplatRtGpuError::ExceedsDeviceLimits {
                what: "splat_buffer (scene exceeds max_splats)",
                requested: scene.splats.len() as u64 * std::mem::size_of::<GaussianSplat>() as u64,
                limit: self.max_splats as u64 * std::mem::size_of::<GaussianSplat>() as u64,
            });
        }
        let splat_range = n as u64 * std::mem::size_of::<GaussianSplat>() as u64;
        if splat_range > self.max_storage_binding {
            return Err(SplatRtGpuError::ExceedsDeviceLimits {
                what: "splat_buffer",
                requested: splat_range,
                limit: self.max_storage_binding,
            });
        }

        // Upload splats (raw 96-byte POD) and the camera uniform.
        if n > 0 {
            self.queue.write_buffer(
                &self.splat_buffer,
                0,
                bytemuck::cast_slice(&scene.splats[..n]),
            );
        }

        // Pre-normalize forward/right/up exactly as render_orthographic does.
        let fwd = camera.forward.normalize();
        let right = camera.right.normalize();
        let up = camera.up.normalize();
        let cam_u = CameraUniform {
            eye: camera.eye.into(),
            _p0: 0.0,
            forward: fwd.into(),
            _p1: 0.0,
            right: right.into(),
            _p2: 0.0,
            up: up.into(),
            _p3: 0.0,
            width: camera.width,
            height: camera.height,
            px_w: width,
            px_h: height,
            splat_count: count,
            _p4: 0,
            _p5: 0,
            _p6: 0,
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_u));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_rt_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("splat_rt_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("splat_rt_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 8x8 workgroup; one thread per pixel.
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let copy_bytes = pixels as u64 * PIXEL_FLOATS as u64 * 4;
        encoder.copy_buffer_to_buffer(
            &self.out_buffer,
            0,
            &self.readback_buffer,
            0,
            copy_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..copy_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(SplatRtGpuError::Readback(e.to_string())),
            Err(e) => return Err(SplatRtGpuError::Readback(e.to_string())),
        }

        let out: Vec<[f32; PIXEL_FLOATS]> = {
            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            (0..pixels)
                .map(|i| {
                    let mut px = [0.0f32; PIXEL_FLOATS];
                    px.copy_from_slice(&floats[i * PIXEL_FLOATS..i * PIXEL_FLOATS + PIXEL_FLOATS]);
                    px
                })
                .collect()
        };
        self.readback_buffer.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splat_rt::render_orthographic;
    use glam::{Quat, Vec3};
    use half::f16;

    fn band_spectral(band: usize, value: f32) -> [u16; 16] {
        let mut s = [f16::from_f32(0.0).to_bits(); 16];
        s[band] = f16::from_f32(value).to_bits();
        s
    }

    /// Skip a GPU test gracefully if this box truly has no GPU (CI without one).
    /// On the target box (RADV) this returns `Some` and the test runs.
    fn try_gpu(max_splats: u32, max_pixels: u32) -> Option<SplatRtGpu> {
        match SplatRtGpu::new(max_splats, max_pixels) {
            Ok(g) => {
                eprintln!("[splat_rt_gpu test] adapter: {}", g.adapter_name);
                Some(g)
            }
            Err(SplatRtGpuError::NoAdapter) => {
                eprintln!("[splat_rt_gpu test] no adapter — skipping GPU test");
                None
            }
            Err(e) => panic!("unexpected GPU init error on a box with a GPU: {e}"),
        }
    }

    /// The 3-splat scene from the CPU oracle's `cross_check_against_rasterizer`
    /// test, reconstructed identically, with that test's orthographic camera.
    fn cross_check_scene() -> (RtScene, OrthoCamera, u32) {
        const RES: u32 = 32;
        let splats = vec![
            GaussianSplat::volume([-0.8, 0.0, 0.0], [0.5, 0.5, 0.5], Quat::IDENTITY, 230, band_spectral(2, 1.0)),
            GaussianSplat::volume([0.8, 0.0, 0.0], [0.5, 0.5, 0.5], Quat::IDENTITY, 230, band_spectral(8, 1.0)),
            GaussianSplat::volume([0.0, 0.9, 0.0], [0.5, 0.5, 0.5], Quat::IDENTITY, 230, band_spectral(13, 1.0)),
        ];
        let eye_z = 5.0f32;
        let half = (std::f32::consts::FRAC_PI_4 * 0.5).tan() * eye_z;
        let ortho = OrthoCamera {
            eye: Vec3::new(0.0, 0.0, eye_z),
            forward: Vec3::new(0.0, 0.0, -1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, -1.0, 0.0),
            width: 2.0 * half,
            height: 2.0 * half,
        };
        (RtScene::build(splats, 64), ortho, RES)
    }

    /// The 200-splat seeded random scene from the CPU oracle's
    /// `bvh_matches_brute_force` test, reconstructed identically (same LCG seed
    /// and constants), rendered with an orthographic camera looking down -Z.
    fn random_scene() -> (RtScene, OrthoCamera, u32) {
        let mut seed = 0x1234_5678u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut splats = Vec::with_capacity(200);
        for _ in 0..200 {
            let p = [
                (rng() - 0.5) * 8.0,
                (rng() - 0.5) * 8.0,
                (rng() - 0.5) * 8.0,
            ];
            let band = (rng() * 16.0) as usize % 16;
            splats.push(GaussianSplat::volume(
                p,
                [0.6, 0.6, 0.6],
                Quat::IDENTITY,
                150,
                band_spectral(band, 1.0),
            ));
        }
        const RES: u32 = 32;
        let ortho = OrthoCamera {
            eye: Vec3::new(0.0, 0.0, 12.0),
            forward: Vec3::new(0.0, 0.0, -1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            width: 10.0,
            height: 10.0,
        };
        (RtScene::build(splats, 32), ortho, RES)
    }

    /// Maximum absolute and relative per-band deviation between two pixel grids.
    fn measure_dev(
        cpu: &[[f32; 16]],
        gpu: &[[f32; PIXEL_FLOATS]],
    ) -> (f32, f32) {
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            for b in 0..16 {
                let a = c[b];
                let d = (a - g[b]).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let denom = a.abs().max(1e-4);
                let rel = d / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        (max_abs, max_rel)
    }

    /// THE VALIDATION (GpuGi house pattern): render the CPU oracle's own
    /// cross-check 3-splat scene on BOTH paths and assert per-pixel per-band
    /// agreement within a measured, reported tolerance. f32-GPU vs f32-CPU with
    /// identical math is very tight.
    #[test]
    fn gpu_matches_cpu_cross_check_scene() {
        let (scene, cam, res) = cross_check_scene();
        let Some(gpu) = try_gpu(scene.splats.len() as u32, res * res) else { return };

        let cpu = render_orthographic(&scene, &cam, res, res, 64);
        let g = gpu.render(&scene, &cam, res, res).expect("gpu render");

        let (max_abs, max_rel) = measure_dev(&cpu, &g);
        eprintln!(
            "[cross_check] max_abs_dev={max_abs:e} max_rel_dev={max_rel:e} (asserting abs < 1e-4)"
        );
        // Identical f32 math; only exp/sqrt ULP differences remain. Measured
        // deviation is ~1e-6 on RADV; we assert a generous 1e-4 absolute bound
        // (>~100x the measured residual) and a tight relative bound.
        assert!(
            max_abs < 1e-4,
            "GPU<->CPU max absolute band deviation {max_abs:e} exceeds 1e-4"
        );
        assert!(
            max_rel < 1e-3,
            "GPU<->CPU max relative band deviation {max_rel:e} exceeds 1e-3"
        );

        // Sanity: the image is not empty (both paths actually lit something).
        let cpu_total: f32 = cpu.iter().flat_map(|p| p.iter()).sum();
        assert!(cpu_total > 0.0, "CPU oracle produced an empty image");
    }

    /// THE VALIDATION on the 200-splat seeded random scene: heavier overlap,
    /// more depth sorting, exercising the in-shader insertion sort against the
    /// CPU's depth-keyed sort.
    #[test]
    fn gpu_matches_cpu_random_scene() {
        let (scene, cam, res) = random_scene();
        let Some(gpu) = try_gpu(scene.splats.len() as u32, res * res) else { return };

        let t0 = std::time::Instant::now();
        let cpu = render_orthographic(&scene, &cam, res, res, 64);
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        // Warm up GPU (pipeline/first-submit overhead), then time a steady render.
        let _ = gpu.render(&scene, &cam, res, res).expect("gpu warmup");
        let t1 = std::time::Instant::now();
        let g = gpu.render(&scene, &cam, res, res).expect("gpu render");
        let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!("[timing] 200-splat 32x32: CPU={cpu_ms:.3}ms GPU(incl readback)={gpu_ms:.3}ms");

        let (max_abs, max_rel) = measure_dev(&cpu, &g);
        eprintln!(
            "[random_200] max_abs_dev={max_abs:e} max_rel_dev={max_rel:e} (asserting abs < 1e-4)"
        );
        assert!(
            max_abs < 1e-4,
            "GPU<->CPU max absolute band deviation {max_abs:e} exceeds 1e-4"
        );
        assert!(
            max_rel < 1e-3,
            "GPU<->CPU max relative band deviation {max_rel:e} exceeds 1e-3"
        );

        let cpu_total: f32 = cpu.iter().flat_map(|p| p.iter()).sum();
        assert!(cpu_total > 0.0, "CPU oracle produced an empty image");
    }

    /// CONTRACT: `new()` returns [`SplatRtGpuError::ExceedsDeviceLimits`] (NOT a
    /// process-aborting wgpu Validation Error) when the requested buffers would
    /// exceed `max_storage_buffer_binding_size`. Probe-proven threshold: out path
    /// fails above floor(limit/68) pixels. We request well above the default
    /// 128 MiB cap and assert the exact variant + fields, never aborting.
    #[test]
    fn new_rejects_oversized_out_buffer_with_error_not_panic() {
        // 4M pixels * 17 floats * 4 bytes = 272 MB out buffer, above the 128 MiB
        // default storage-binding limit — must surface as a clean error.
        let err = match SplatRtGpu::new(1, 4_000_000) {
            Ok(g) => {
                // Only acceptable if this box's limit is actually large enough.
                let lim = g.max_storage_binding;
                let out_bytes = 4_000_000u64 * PIXEL_FLOATS as u64 * 4;
                assert!(
                    out_bytes <= lim,
                    "new() succeeded but out_bytes {out_bytes} > limit {lim}"
                );
                eprintln!("[limits] device storage limit {lim} >= {out_bytes}; skip");
                return;
            }
            Err(SplatRtGpuError::NoAdapter) => {
                eprintln!("[limits] no adapter — skipping");
                return;
            }
            Err(e) => e,
        };
        match err {
            SplatRtGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => {
                assert_eq!(what, "out_buffer");
                assert_eq!(requested, 4_000_000u64 * PIXEL_FLOATS as u64 * 4);
                assert!(requested > limit, "requested {requested} must exceed limit {limit}");
            }
            other => panic!("expected ExceedsDeviceLimits, got {other:?}"),
        }
    }

    /// CONTRACT: a too-large render returns the error variant (here via the
    /// converted `max_pixels` guard) WITHOUT aborting. Sized so `new()` succeeds
    /// (tiny buffers) but the render asks for more pixels than allocated.
    #[test]
    fn render_oversized_returns_error_not_panic() {
        let Some(gpu) = try_gpu(3, 16) else { return };
        let (scene, cam, _res) = cross_check_scene();
        // 100x100 = 10000 pixels >> max_pixels=16 → converted-assert error path.
        let err = gpu
            .render(&scene, &cam, 100, 100)
            .expect_err("oversized render must return an error, not abort");
        match err {
            SplatRtGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => {
                assert_eq!(what, "out_buffer (render exceeds max_pixels)");
                assert!(requested > limit, "requested {requested} must exceed limit {limit}");
            }
            other => panic!("expected ExceedsDeviceLimits, got {other:?}"),
        }
        // The GPU is still usable afterward (no abort): a valid render still works.
        let _ = gpu.render(&scene, &cam, 4, 4).expect("valid render after rejection");
    }

    /// Determinism: two GPU renders of the same scene are bit-identical.
    #[test]
    fn gpu_is_deterministic() {
        let (scene, cam, res) = random_scene();
        let Some(gpu) = try_gpu(scene.splats.len() as u32, res * res) else { return };

        let a = gpu.render(&scene, &cam, res, res).expect("render a");
        let b = gpu.render(&scene, &cam, res, res).expect("render b");
        assert_eq!(a.len(), b.len());
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            for k in 0..PIXEL_FLOATS {
                assert_eq!(
                    pa[k].to_bits(),
                    pb[k].to_bits(),
                    "GPU render must be bit-identical: pixel {i} float {k}: {} vs {}",
                    pa[k],
                    pb[k]
                );
            }
        }
    }
}
