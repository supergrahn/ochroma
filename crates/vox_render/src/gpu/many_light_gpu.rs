//! Many-light reservoir sampler — GPU compute port of the CPU oracle
//! [`crate::many_light`].
//!
//! GPU is the foundation; the CPU [`crate::many_light::LightSampler::sample`]
//! (the `full` brute-force path over ALL lights) is the correctness ORACLE this
//! mirrors BIT-FOR-BIT. One compute thread per shade point — the GPU analogue
//! of per-pixel light sampling. Each thread runs the EXACT CPU reservoir loop
//! over every light, seeded per-thread EXACTLY as the CPU seeds per-call, so
//! the same `(shade_point, seed)` reproduces the same draws → the same chosen
//! light index BIT-EXACTLY.
//!
//! Follows the [`crate::gpu::splat_rt_gpu::SplatRtGpu`] / [`crate::spectral_gi`]
//! house pattern: construct-with-its-own-device, adapter-gated (never panics on
//! a missing GPU — returns [`ManyLightGpuError::NoAdapter`]), measured-then-
//! asserted tolerances, bit-identical determinism.
//!
//! ## The u64 LCG, emulated in WGSL via 2×u32
//!
//! WGSL has no `u64`, so the CPU `Lcg { state: u64 }` is emulated with two u32
//! limbs and carry-propagating mul/add. This is the load-bearing piece: a raw
//! draws test ([`tests::lcg_emulation_bit_exact`]) pins N GPU draws against the
//! CPU [`crate::many_light`] `Lcg` BEFORE the reservoir is trusted. See
//! `many_light_gpu.wgsl` for the limb arithmetic.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::lighting::PointLight;

/// GPU-side light: `pos.xyz + radius`, `color.rgb + intensity`. Matches the
/// `Light` struct in the shader (two `vec4<f32>`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    pos_radius: [f32; 4],
    color_intensity: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuLight>() == 32);

/// GPU-side shade point: `xyz` + a spare `w` (carries seed_lo for debugging).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuShadePoint {
    point_seedlo: [f32; 4],
}

/// Dispatch params uniform (mirrors `Params` in the shader).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    light_count: u32,
    point_count: u32,
    draw_count: u32,
    _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<Params>() == 16);

/// One selected light plus its unbiased RIS weight — the readback POD,
/// matching `LightSampleOut` in the shader. `light_index == u32::MAX` means no
/// light was chosen (every candidate had zero target).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuLightSample {
    /// Chosen light index, or `u32::MAX` if none chosen.
    pub light_index: u32,
    /// RIS weight `W = (1/p̂_s) · (Σp̂ / M)`.
    pub weight: f32,
    /// Target value `p̂_s` of the chosen light.
    pub target: f32,
    /// Candidate count `M` (lights inspected).
    pub m: u32,
}

const _: () = assert!(std::mem::size_of::<GpuLightSample>() == 16);

/// Error returned when the GPU sampler cannot be created or run. Never panics
/// on a missing/inadequate GPU — the caller can fall back to the CPU oracle.
#[derive(Debug, Clone)]
pub enum ManyLightGpuError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed.
    DeviceCreation(String),
    /// Mapping the readback buffer failed.
    Readback(String),
    /// A required storage buffer would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`). Returned instead
    /// of letting wgpu raise an uncaptured Validation Error that aborts the
    /// process — so the caller can fall back to the CPU oracle.
    ///
    /// NOTE: this hardens the SAME no-panic-contract class as
    /// [`crate::gpu::splat_rt_gpu::SplatRtGpuError::ExceedsDeviceLimits`]. A
    /// refuter rejected the parallel finding here on reachability-today (every
    /// shipped caller passes tiny sizes), but the contract docs are identical and
    /// the missing-validation gap is structurally the same, so we close it the
    /// same way.
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for ManyLightGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManyLightGpuError::NoAdapter => write!(f, "no GPU adapter available"),
            ManyLightGpuError::DeviceCreation(e) => {
                write!(f, "GPU device creation failed: {e}")
            }
            ManyLightGpuError::Readback(e) => write!(f, "GPU readback failed: {e}"),
            ManyLightGpuError::ExceedsDeviceLimits {
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

impl std::error::Error for ManyLightGpuError {}

/// Headless GPU many-light reservoir sampler. Owns its own wgpu device/queue
/// (no window/surface). Sized for up to `max_lights` and `max_points`.
pub struct ManyLightGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    sample_pipeline: wgpu::ComputePipeline,
    lcg_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    light_buffer: wgpu::Buffer,
    point_buffer: wgpu::Buffer,
    seed_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    out_readback: wgpu::Buffer,
    draw_buffer: wgpu::Buffer,
    draw_readback: wgpu::Buffer,
    max_lights: u32,
    max_points: u32,
    /// Maximum raw draws per seed the LCG-test buffer can hold (per point).
    max_draws_per_point: u32,
    /// `device.limits().max_storage_buffer_binding_size`, captured at creation so
    /// `sample()` / `lcg_draws()` can validate each storage binding range and
    /// return [`ManyLightGpuError::ExceedsDeviceLimits`] instead of letting wgpu
    /// raise an uncaptured, process-aborting Validation Error.
    max_storage_binding: u64,
    /// Adapter human name, for diagnostics / benches.
    pub adapter_name: String,
}

impl ManyLightGpu {
    /// Create a headless GPU sampler sized for up to `max_lights` lights and
    /// `max_points` shade points. `max_draws_per_point` sizes the LCG raw-draw
    /// test buffer. Returns [`ManyLightGpuError`] (never panics) if no adapter
    /// is found or device creation fails.
    pub fn new(
        max_lights: u32,
        max_points: u32,
        max_draws_per_point: u32,
    ) -> Result<Self, ManyLightGpuError> {
        pollster::block_on(Self::new_async(max_lights, max_points, max_draws_per_point))
    }

    async fn new_async(
        max_lights: u32,
        max_points: u32,
        max_draws_per_point: u32,
    ) -> Result<Self, ManyLightGpuError> {
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
            .ok_or(ManyLightGpuError::NoAdapter)?;
        let info = adapter.get_info();
        crate::gpu::adapter::ensure_hardware(&info).map_err(|_| ManyLightGpuError::NoAdapter)?;
        let adapter_name = info.name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("many_light_gpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| ManyLightGpuError::DeviceCreation(e.to_string()))?;

        let max_lights = max_lights.max(1);
        let max_points = max_points.max(1);
        let max_draws_per_point = max_draws_per_point.max(1);

        let light_bytes = max_lights as u64 * std::mem::size_of::<GpuLight>() as u64;
        let point_bytes = max_points as u64 * std::mem::size_of::<GpuShadePoint>() as u64;
        let seed_bytes = max_points as u64 * 8; // vec2<u32>
        let out_bytes = max_points as u64 * std::mem::size_of::<GpuLightSample>() as u64;
        let draw_bytes = max_points as u64 * max_draws_per_point as u64 * 4;

        // Validate every STORAGE buffer against the device's binding/total limits
        // BEFORE creating + binding them. Same no-panic-contract hardening as
        // splat_rt_gpu: a bound range over `max_storage_buffer_binding_size`
        // triggers an uncaptured, process-aborting wgpu Validation Error, so we
        // surface it as a returned error here instead.
        let limits = device.limits();
        let max_storage_binding = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        for (what, bytes) in [
            ("light_buffer", light_bytes),
            ("point_buffer", point_bytes),
            ("seed_buffer", seed_bytes),
            ("out_buffer", out_bytes),
            ("draw_buffer", draw_bytes),
        ] {
            if bytes > max_storage_binding {
                return Err(ManyLightGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage_binding,
                });
            }
            if bytes > max_buffer {
                return Err(ManyLightGpuError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_lights"),
            size: light_bytes.max(32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_points"),
            size: point_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let seed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_seeds"),
            size: seed_bytes.max(8),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_out"),
            size: out_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_out_readback"),
            size: out_bytes.max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_draws"),
            size: draw_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let draw_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("many_light_draws_readback"),
            size: draw_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("many_light_gpu.wgsl"));

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
            label: Some("many_light_bgl"),
            entries: &[
                storage_ro(0), // lights
                storage_ro(1), // points
                storage_ro(2), // seeds
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_rw(4), // out_samples
                storage_rw(5), // out_draws
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("many_light_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let sample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("many_light_sample_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });
        let lcg_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("many_light_lcg_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("lcg_test"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            sample_pipeline,
            lcg_pipeline,
            bgl,
            light_buffer,
            point_buffer,
            seed_buffer,
            params_buffer,
            out_buffer,
            out_readback,
            draw_buffer,
            draw_readback,
            max_lights,
            max_points,
            max_draws_per_point,
            max_storage_binding,
            adapter_name,
        })
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("many_light_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.seed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.draw_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Upload `lights` + per-point `(shade_point, seed)` pairs, dispatch one
    /// thread per shade point running the EXACT CPU reservoir over all lights,
    /// and read back one [`GpuLightSample`] per point.
    ///
    /// `seeds[i]` is the FULL u64 seed for point `i`, seeded exactly as the CPU
    /// passes to `LightSampler::sample(shade_point, seed)` (the GPU applies
    /// `Lcg::new`'s mul-add internally, mirroring the CPU).
    pub fn sample(
        &self,
        lights: &[PointLight],
        points: &[Vec3],
        seeds: &[u64],
    ) -> Result<Vec<GpuLightSample>, ManyLightGpuError> {
        assert_eq!(points.len(), seeds.len(), "points and seeds length mismatch");
        let n_points = points.len();
        if n_points == 0 {
            return Ok(Vec::new());
        }
        // No-panic contract: size overruns go through the error channel (the
        // converted `assert!`s were panics). Each over-limit also implies the
        // corresponding storage binding would exceed the device cap.
        if lights.len() as u32 > self.max_lights {
            return Err(ManyLightGpuError::ExceedsDeviceLimits {
                what: "light_buffer (scene exceeds max_lights)",
                requested: lights.len() as u64 * std::mem::size_of::<GpuLight>() as u64,
                limit: self.max_lights as u64 * std::mem::size_of::<GpuLight>() as u64,
            });
        }
        if n_points as u32 > self.max_points {
            return Err(ManyLightGpuError::ExceedsDeviceLimits {
                what: "out_buffer (dispatch exceeds max_points)",
                requested: n_points as u64 * std::mem::size_of::<GpuLightSample>() as u64,
                limit: self.max_points as u64 * std::mem::size_of::<GpuLightSample>() as u64,
            });
        }
        // Defensive: validate the actual bound ranges against the storage limit.
        let out_range = n_points as u64 * std::mem::size_of::<GpuLightSample>() as u64;
        if out_range > self.max_storage_binding {
            return Err(ManyLightGpuError::ExceedsDeviceLimits {
                what: "out_buffer",
                requested: out_range,
                limit: self.max_storage_binding,
            });
        }

        self.upload(lights, points, seeds, 0);

        let bind_group = self.bind_group();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("many_light_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("many_light_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n_points as u32).div_ceil(64), 1, 1);
        }
        let copy_bytes = n_points as u64 * std::mem::size_of::<GpuLightSample>() as u64;
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.out_readback, 0, copy_bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = self.map_read(&self.out_readback, copy_bytes)?;
        let out: Vec<GpuLightSample> = bytemuck::cast_slice::<u8, GpuLightSample>(&data)[..n_points]
            .to_vec();
        self.out_readback.unmap();
        Ok(out)
    }

    /// LCG bit-exactness probe: for each seed, emit `draws_per_seed` consecutive
    /// `next_unit()` values, exactly as the CPU `Lcg` (after `Lcg::new`) would
    /// produce. Returns a `Vec` of `seeds.len() * draws_per_seed` f32 in
    /// `[0,1)`, row-major (seed-major).
    pub fn lcg_draws(
        &self,
        seeds: &[u64],
        draws_per_seed: u32,
    ) -> Result<Vec<f32>, ManyLightGpuError> {
        let n_seeds = seeds.len();
        if n_seeds == 0 || draws_per_seed == 0 {
            return Ok(Vec::new());
        }
        if n_seeds as u32 > self.max_points {
            return Err(ManyLightGpuError::ExceedsDeviceLimits {
                what: "seed_buffer (lcg seeds exceed max_points)",
                requested: n_seeds as u64 * 8,
                limit: self.max_points as u64 * 8,
            });
        }
        if draws_per_seed > self.max_draws_per_point {
            return Err(ManyLightGpuError::ExceedsDeviceLimits {
                what: "draw_buffer (draws_per_seed exceeds max_draws_per_point)",
                requested: n_seeds as u64 * draws_per_seed as u64 * 4,
                limit: n_seeds as u64 * self.max_draws_per_point as u64 * 4,
            });
        }

        // Shade points are unused by lcg_test; pass zeros.
        let pts: Vec<Vec3> = vec![Vec3::ZERO; n_seeds];
        self.upload(&[], &pts, seeds, draws_per_seed);

        let bind_group = self.bind_group();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("many_light_lcg_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("many_light_lcg_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lcg_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n_seeds as u32).div_ceil(64), 1, 1);
        }
        let total = n_seeds * draws_per_seed as usize;
        let copy_bytes = total as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.draw_buffer, 0, &self.draw_readback, 0, copy_bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = self.map_read(&self.draw_readback, copy_bytes)?;
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data)[..total].to_vec();
        self.draw_readback.unmap();
        Ok(out)
    }

    /// Upload lights, points, seeds, and params for a dispatch. `point_count`
    /// is taken from `points.len()`; `light_count` from `lights.len()`.
    fn upload(&self, lights: &[PointLight], points: &[Vec3], seeds: &[u64], draw_count: u32) {
        if !lights.is_empty() {
            let gl: Vec<GpuLight> = lights
                .iter()
                .map(|l| GpuLight {
                    pos_radius: [l.position.x, l.position.y, l.position.z, l.radius],
                    color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
                })
                .collect();
            self.queue
                .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&gl));
        }
        let gp: Vec<GpuShadePoint> = points
            .iter()
            .zip(seeds.iter())
            .map(|(p, s)| GpuShadePoint {
                point_seedlo: [p.x, p.y, p.z, f32::from_bits(*s as u32)],
            })
            .collect();
        self.queue
            .write_buffer(&self.point_buffer, 0, bytemuck::cast_slice(&gp));

        let gs: Vec<[u32; 2]> = seeds
            .iter()
            .map(|s| [*s as u32, (*s >> 32) as u32])
            .collect();
        self.queue
            .write_buffer(&self.seed_buffer, 0, bytemuck::cast_slice(&gs));

        let params = Params {
            light_count: lights.len() as u32,
            point_count: points.len() as u32,
            draw_count,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn map_read(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
    ) -> Result<Vec<u8>, ManyLightGpuError> {
        let slice = buffer.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(ManyLightGpuError::Readback(e.to_string())),
            Err(e) => return Err(ManyLightGpuError::Readback(e.to_string())),
        }
        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }
}

/// Reference CPU u64 LCG, byte-identical to the private `Lcg` in
/// [`crate::many_light`] (which is not `pub`). Used by tests to cross-check the
/// GPU limb emulation. `new` mixes the seed; `next_unit` advances and reads the
/// top 24 bits.
/// The CPU LCG multiplier `C = 6364136223846793005`.
#[cfg(test)]
const LCG_C: u64 = 6364136223846793005;
/// The CPU LCG increment `A = 1442695040888963407`.
#[cfg(test)]
const LCG_A: u64 = 1442695040888963407;
/// The `sample_n` sub-seed multiplier (golden-ratio constant).
#[cfg(test)]
const SUBSEED_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

#[cfg(test)]
struct RefLcg {
    state: u64,
}

#[cfg(test)]
impl RefLcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(LCG_C).wrapping_add(LCG_A),
        }
    }
    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(LCG_C).wrapping_add(LCG_A);
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip a GPU test gracefully if this box truly has no GPU (CI without one).
    fn try_gpu(max_lights: u32, max_points: u32, max_draws: u32) -> Option<ManyLightGpu> {
        match ManyLightGpu::new(max_lights, max_points, max_draws) {
            Ok(g) => {
                eprintln!("[many_light_gpu test] adapter: {}", g.adapter_name);
                Some(g)
            }
            Err(ManyLightGpuError::NoAdapter) => {
                eprintln!("[many_light_gpu test] no adapter — skipping GPU test");
                None
            }
            Err(e) => panic!("unexpected GPU init error on a box with a GPU: {e}"),
        }
    }

    /// The CPU oracle's `make_scene`: `n` lights from a seeded `RefLcg`,
    /// reconstructed identically (same constants and field order).
    fn make_scene(n: usize, seed: u64) -> Vec<PointLight> {
        let mut rng = RefLcg::new(seed);
        (0..n)
            .map(|_| {
                let px = (rng.next_unit() - 0.5) * 20.0;
                let py = (rng.next_unit() - 0.5) * 20.0;
                let pz = (rng.next_unit() - 0.5) * 20.0;
                PointLight {
                    position: Vec3::new(px, py, pz),
                    color: [
                        0.2 + rng.next_unit(),
                        0.2 + rng.next_unit(),
                        0.2 + rng.next_unit(),
                    ],
                    intensity: 0.5 + rng.next_unit() * 2.0,
                    radius: 8.0 + rng.next_unit() * 12.0,
                }
            })
            .collect()
    }

    /// CPU oracle: one `LightSampler::sample` per (point, seed). Public API of
    /// `many_light`, so this is the literal oracle.
    fn cpu_samples(
        lights: &[PointLight],
        points: &[Vec3],
        seeds: &[u64],
    ) -> Vec<Option<crate::many_light::LightSample>> {
        let sampler = crate::many_light::LightSampler::new(lights);
        points
            .iter()
            .zip(seeds.iter())
            .map(|(p, s)| sampler.sample(*p, *s))
            .collect()
    }

    /// THE LOAD-BEARING PRIMITIVE TEST: the u64-via-2×u32 LCG emulation produces
    /// BIT-EXACT raw draws vs the CPU `Lcg`, before any reservoir is trusted.
    #[test]
    fn lcg_emulation_bit_exact() {
        const SEEDS: usize = 64;
        const DRAWS: u32 = 32;
        let Some(gpu) = try_gpu(1, SEEDS as u32, DRAWS) else { return };

        let seeds: Vec<u64> = (0..SEEDS as u64)
            .map(|i| {
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0xDEAD_BEEF)
            })
            .collect();

        let gpu_draws = gpu.lcg_draws(&seeds, DRAWS).expect("lcg draws");
        assert_eq!(gpu_draws.len(), SEEDS * DRAWS as usize);

        let mut total = 0usize;
        let mut bit_equal = 0usize;
        for (si, &seed) in seeds.iter().enumerate() {
            let mut rng = RefLcg::new(seed);
            for k in 0..DRAWS as usize {
                let cpu = rng.next_unit();
                let gpu_v = gpu_draws[si * DRAWS as usize + k];
                total += 1;
                if cpu.to_bits() == gpu_v.to_bits() {
                    bit_equal += 1;
                } else {
                    eprintln!(
                        "[lcg] MISMATCH seed#{si} draw#{k}: cpu={cpu} ({:#010x}) gpu={gpu_v} ({:#010x})",
                        cpu.to_bits(),
                        gpu_v.to_bits()
                    );
                }
            }
        }
        eprintln!("[lcg] {bit_equal}/{total} raw draws bit-equal");
        assert_eq!(
            bit_equal, total,
            "GPU LCG emulation not bit-exact: {bit_equal}/{total}"
        );
    }

    /// THE VALIDATION (house pattern): the CPU oracle's exact 64-light scene,
    /// same seeds → GPU chosen indices BIT-EQUAL to CPU `sample()` indices, W
    /// weights within a measured-then-asserted f32 tolerance.
    #[test]
    fn gpu_matches_cpu_indices_and_weights() {
        let lights = make_scene(64, 0xABCD_1234);
        // A spread of shade points across the scene volume, each its own seed.
        let mut points = Vec::new();
        let mut seeds = Vec::new();
        let mut sg = RefLcg::new(0x5151_5151);
        for k in 0..256u64 {
            points.push(Vec3::new(
                (sg.next_unit() - 0.5) * 16.0,
                (sg.next_unit() - 0.5) * 16.0,
                (sg.next_unit() - 0.5) * 16.0,
            ));
            seeds.push(
                k.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0x1234_5678),
            );
        }
        let Some(gpu) = try_gpu(lights.len() as u32, points.len() as u32, 1) else { return };

        let cpu = cpu_samples(&lights, &points, &seeds);
        let gpu_out = gpu.sample(&lights, &points, &seeds).expect("gpu sample");
        assert_eq!(cpu.len(), gpu_out.len());

        let mut idx_total = 0usize;
        let mut idx_equal = 0usize;
        let mut max_w_abs = 0.0f32;
        let mut max_w_rel = 0.0f32;
        for (c, g) in cpu.iter().zip(gpu_out.iter()) {
            idx_total += 1;
            match c {
                Some(cs) => {
                    if g.light_index as usize == cs.light_index {
                        idx_equal += 1;
                    } else {
                        eprintln!(
                            "[idx] MISMATCH cpu={} gpu={}",
                            cs.light_index, g.light_index
                        );
                    }
                    assert_eq!(g.m as usize, cs.m, "candidate count M mismatch");
                    let d = (cs.weight - g.weight).abs();
                    if d > max_w_abs {
                        max_w_abs = d;
                    }
                    let rel = d / cs.weight.abs().max(1e-6);
                    if rel > max_w_rel {
                        max_w_rel = rel;
                    }
                }
                None => {
                    // CPU chose nothing → GPU must agree (u32::MAX sentinel).
                    if g.light_index == u32::MAX {
                        idx_equal += 1;
                    }
                }
            }
        }
        eprintln!(
            "[validate] index equality {idx_equal}/{idx_total}; max_w_abs={max_w_abs:e} max_w_rel={max_w_rel:e}"
        );
        assert_eq!(
            idx_equal, idx_total,
            "GPU sample indices must be 100% bit-equal to CPU: {idx_equal}/{idx_total}"
        );
        // Identical f32 math + bit-exact draws → weights are ULP-tight. Measured
        // ~1e-6 relative on RADV; assert a generous, well-above-measured bound.
        assert!(
            max_w_rel < 1e-4,
            "GPU<->CPU max relative W deviation {max_w_rel:e} exceeds 1e-4"
        );
    }

    /// Unbiasedness on GPU: 4096 GPU reservoir samples of the CPU oracle's exact
    /// 64-light scene/point, accumulated with the same `weight·M` sum-estimator,
    /// match the brute-force reference within the CPU test's own 2% bound.
    #[test]
    fn gpu_unbiasedness_within_2pct() {
        let lights = make_scene(64, 0xABCD_1234);
        let shade_point = Vec3::new(1.0, 2.0, -3.0);
        let normal = Vec3::ZERO;
        const SAMPLES: usize = 4096;

        let Some(gpu) = try_gpu(lights.len() as u32, SAMPLES as u32, 1) else { return };

        // Brute-force reference: Σ over all lights of pure-radiance contribution.
        let reference = brute_force(shade_point, normal, &lights);

        // One reservoir per sample, sub-seeds derived EXACTLY as estimate_radiance.
        let base_seed = 0x5151_5151u64;
        let points = vec![shade_point; SAMPLES];
        let seeds: Vec<u64> = (0..SAMPLES as u64)
            .map(|k| base_seed.wrapping_mul(SUBSEED_MUL).wrapping_add(k))
            .collect();

        let t0 = std::time::Instant::now();
        let gpu_out = gpu.sample(&lights, &points, &seeds).expect("gpu sample");
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Time the CPU equivalent for the informational comparison.
        let t1 = std::time::Instant::now();
        let cpu_est =
            crate::many_light::estimate_radiance(shade_point, normal, &lights, SAMPLES, base_seed);
        let cpu_ms = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[timing] 4096 samples x 64 lights: CPU={cpu_ms:.3}ms GPU(incl readback)={gpu_ms:.3}ms"
        );
        let _ = cpu_est;

        // Accumulate the GPU samples with the same sum-estimator as the CPU.
        let mut acc = [0.0f32; 3];
        for g in &gpu_out {
            if g.light_index == u32::MAX {
                continue;
            }
            let w = g.weight * g.m as f32;
            let f = shade_contribution(&lights[g.light_index as usize], shade_point, normal);
            acc[0] += f[0] * w;
            acc[1] += f[1] * w;
            acc[2] += f[2] * w;
        }
        let inv = 1.0 / SAMPLES as f32;
        let estimate = [acc[0] * inv, acc[1] * inv, acc[2] * inv];

        for c in 0..3 {
            let rel = (estimate[c] - reference[c]).abs() / reference[c].max(1e-6);
            eprintln!(
                "[unbiased] channel {c}: gpu_est={} ref={} rel={:.4}",
                estimate[c], reference[c], rel
            );
            assert!(
                rel < 0.02,
                "channel {c}: GPU estimate {} vs reference {} rel err {:.4} >= 2%",
                estimate[c],
                reference[c],
                rel
            );
        }
    }

    /// Bit-identical determinism: two GPU dispatches of the same scene/seeds
    /// produce byte-identical samples.
    #[test]
    fn gpu_is_deterministic() {
        let lights = make_scene(64, 0xABCD_1234);
        let points: Vec<Vec3> = (0..128u64)
            .map(|i| Vec3::new(i as f32 * 0.05 - 3.0, 1.0, -2.0))
            .collect();
        let seeds: Vec<u64> = (0..128u64)
            .map(|k| k.wrapping_mul(SUBSEED_MUL).wrapping_add(99))
            .collect();
        let Some(gpu) = try_gpu(lights.len() as u32, points.len() as u32, 1) else { return };

        let a = gpu.sample(&lights, &points, &seeds).expect("dispatch a");
        let b = gpu.sample(&lights, &points, &seeds).expect("dispatch b");
        assert_eq!(a.len(), b.len());
        for (i, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(sa.light_index, sb.light_index, "index differs at {i}");
            assert_eq!(
                sa.weight.to_bits(),
                sb.weight.to_bits(),
                "weight not bit-identical at {i}"
            );
            assert_eq!(sa.target.to_bits(), sb.target.to_bits(), "target differs at {i}");
            assert_eq!(sa.m, sb.m, "M differs at {i}");
        }
    }

    /// CONTRACT (class-consistency with splat_rt_gpu): an over-`max_points`
    /// dispatch returns [`ManyLightGpuError::ExceedsDeviceLimits`] (the converted
    /// `assert!`), NOT a panic. `new()` allocates tiny buffers; `sample()` is then
    /// handed more points than allocated.
    #[test]
    fn sample_oversized_returns_error_not_panic() {
        let Some(gpu) = try_gpu(1, 4, 1) else { return };
        let lights = make_scene(1, 0xABCD_1234);
        let points = vec![Vec3::ZERO; 100]; // 100 > max_points=4
        let seeds = vec![0u64; 100];
        let err = gpu
            .sample(&lights, &points, &seeds)
            .expect_err("oversized dispatch must return an error, not abort");
        match err {
            ManyLightGpuError::ExceedsDeviceLimits {
                what,
                requested,
                limit,
            } => {
                assert_eq!(what, "out_buffer (dispatch exceeds max_points)");
                assert!(requested > limit, "requested {requested} must exceed limit {limit}");
            }
            other => panic!("expected ExceedsDeviceLimits, got {other:?}"),
        }
        // GPU still usable: a valid dispatch still works (no abort happened).
        let ok_pts = vec![Vec3::ZERO; 2];
        let ok_seeds = vec![1u64, 2u64];
        let _ = gpu.sample(&lights, &ok_pts, &ok_seeds).expect("valid sample after rejection");
    }

    // --- CPU reference helpers (mirror many_light.rs's private test helpers) ---

    fn attenuation(distance: f32, intensity: f32, radius: f32) -> f32 {
        if distance >= radius {
            return 0.0;
        }
        let d = distance / radius;
        intensity * (1.0 - d * d).max(0.0)
    }

    fn shade_contribution(light: &PointLight, shade_point: Vec3, normal: Vec3) -> [f32; 3] {
        let d = light.position.distance(shade_point);
        let att = attenuation(d, light.intensity, light.radius);
        let n = normal.normalize_or_zero();
        let cos = if n == Vec3::ZERO {
            1.0
        } else {
            let to_light = (light.position - shade_point).normalize_or_zero();
            n.dot(to_light).max(0.0)
        };
        let s = att * cos;
        [light.color[0] * s, light.color[1] * s, light.color[2] * s]
    }

    fn brute_force(shade_point: Vec3, normal: Vec3, lights: &[PointLight]) -> [f32; 3] {
        let mut acc = [0.0f32; 3];
        for l in lights {
            let f = shade_contribution(l, shade_point, normal);
            acc[0] += f[0];
            acc[1] += f[1];
            acc[2] += f[2];
        }
        acc
    }
}
