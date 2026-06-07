//! Runtime spectral relight kernel — GPU compute port of the CPU oracle
//! [`crate::relight`].
//!
//! GPU is the foundation; the CPU [`crate::relight::relight_scene`] (driving the
//! pure `derive_intrinsic`, `reilluminate_one`, and `encode_radiance` op chain)
//! is the correctness ORACLE this mirrors
//! BIT-FOR-BIT. One compute thread per splat — the GPU analogue of the oracle's
//! rayon per-splat rebake. Each thread runs the EXACT CPU band loop for the
//! ambient-only / no-shadow configuration (`with_sky_ambient(true)`,
//! `with_shadows(false)`) — the same configuration the headline
//! `relight_tungsten_to_daylight_is_bluer` metamer claim uses, where for
//! Preset/CIE illuminants `sun_direction()` is `None`, so `n_dot_l = 1`,
//! `shadow = 1`, and `emitter_gather = 0`.
//!
//! This is the 6th CPU-oracle→WGSL twin, following the
//! [`crate::gpu::many_light_gpu::ManyLightGpu`] / [`crate::gpu::splat_rt_gpu`]
//! house pattern: construct-with-its-own-device, adapter-gated (never panics on
//! a missing GPU — returns [`GpuRelightError::NoAdapter`]),
//! `ExceedsDeviceLimits` validation on every storage buffer, measured-then-
//! asserted tolerances, bit-identical determinism.
//!
//! ## What lives where (host vs GPU)
//!
//! The splat's BAKED radiance is decoded HOST-side via [`GaussianSplat::spectral_f32`]
//! — the SAME `half` decode the oracle's `read_radiance` uses — and uploaded as
//! f32, so the GPU INPUT is bit-identical to the CPU input. The intrinsic divide,
//! the multiply-add re-illumination, and the OUTPUT f16 store (`pack2x16float`,
//! round-to-nearest-even f32→f16 identical to `half::f16::from_f32`) all run
//! on-device. The output f16 quantization is the single load-bearing piece the
//! `<1e-6` bound watches; see `relight_gpu.wgsl`.

use bytemuck::{Pod, Zeroable};
use vox_core::types::GaussianSplat;

/// GPU-layout per-splat relight input: 16 baked-radiance f32 bands, decoded
/// host-side from the splat's f16 spectral field. 64 bytes (4×vec4). Positions /
/// normals are NOT needed in the ambient-only slice (no n_dot_l, no shadow rays).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRelightSplat {
    baked: [f32; 16],
}

const _: () = assert!(std::mem::size_of::<GpuRelightSplat>() == 64);

/// Relight compute uniform. std140/std430: 4×vec4 target_spd (64) +
/// 4×vec4 ambient (64) + 4×vec4 ref_spd (64) + (count, floor, f16_max, pad) (16)
/// = 208 bytes. `ambient` is bound pre-weighted by `AMBIENT_FILL_WEIGHT` (0.5).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RelightParams {
    target_spd: [f32; 16],
    ambient: [f32; 16],
    ref_spd: [f32; 16],
    splat_count: u32,
    floor: f32,
    f16_max: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<RelightParams>() == 208);

/// Error returned when the GPU relight kernel cannot be created or run. Never
/// panics on a missing/inadequate GPU — the caller can fall back to the CPU
/// oracle [`crate::relight::relight_scene`].
#[derive(Debug, Clone)]
pub enum GpuRelightError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed.
    DeviceCreation(String),
    /// Mapping the readback buffer failed.
    Readback(String),
    /// A required storage buffer would exceed a hard device limit
    /// (`max_storage_buffer_binding_size` / `max_buffer_size`). Returned instead
    /// of letting wgpu raise an uncaptured Validation Error that aborts the
    /// process — so the caller can fall back to the CPU oracle. Same no-panic
    /// contract class as
    /// [`crate::gpu::splat_rt_gpu::SplatRtGpuError::ExceedsDeviceLimits`].
    ExceedsDeviceLimits {
        what: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl std::fmt::Display for GpuRelightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuRelightError::NoAdapter => write!(f, "no GPU adapter available"),
            GpuRelightError::DeviceCreation(e) => {
                write!(f, "GPU device creation failed: {e}")
            }
            GpuRelightError::Readback(e) => write!(f, "GPU readback failed: {e}"),
            GpuRelightError::ExceedsDeviceLimits {
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

impl std::error::Error for GpuRelightError {}

/// Headless GPU spectral relight kernel. Owns its own wgpu device/queue (no
/// window/surface). Sized for up to `max_splats` splats.
pub struct GpuRelight {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    splat_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    out_readback: wgpu::Buffer,
    max_splats: u32,
    /// `device.limits().max_storage_buffer_binding_size`, captured at creation so
    /// `relight()` can validate each storage binding range and return
    /// [`GpuRelightError::ExceedsDeviceLimits`] instead of letting wgpu raise an
    /// uncaptured, process-aborting Validation Error.
    max_storage_binding: u64,
    /// Adapter human name, for diagnostics / benches.
    pub adapter_name: String,
}

/// 8 packed u32 per splat (16 f16 bands, two per u32).
const OUT_U32_PER_SPLAT: u64 = 8;

impl GpuRelight {
    /// Create a headless GPU relight kernel sized for up to `max_splats` splats.
    /// Returns [`GpuRelightError`] (never panics) if no adapter is found or
    /// device creation fails, so the caller can stay on the CPU oracle.
    pub fn new(max_splats: u32) -> Result<Self, GpuRelightError> {
        Self::new_with_limits(max_splats, wgpu::Limits::default())
    }

    /// Like [`GpuRelight::new`] but with caller-chosen device limits. Used by the
    /// fallback test to force device creation to fail with impossible limits.
    pub fn new_with_limits(
        max_splats: u32,
        required_limits: wgpu::Limits,
    ) -> Result<Self, GpuRelightError> {
        pollster::block_on(Self::new_async(max_splats, required_limits))
    }

    async fn new_async(
        max_splats: u32,
        required_limits: wgpu::Limits,
    ) -> Result<Self, GpuRelightError> {
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
            .ok_or(GpuRelightError::NoAdapter)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("relight_gpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuRelightError::DeviceCreation(e.to_string()))?;

        let max_splats = max_splats.max(1);

        let splat_bytes = max_splats as u64 * std::mem::size_of::<GpuRelightSplat>() as u64;
        let out_bytes = max_splats as u64 * OUT_U32_PER_SPLAT * 4;

        // Validate every STORAGE buffer against the device's binding/total limits
        // BEFORE creating + binding them. A bound range over
        // `max_storage_buffer_binding_size` triggers an uncaptured,
        // process-aborting wgpu Validation Error, so we surface it as a returned
        // error here — the no-panic contract (mirrors splat_rt_gpu / many_light_gpu).
        let limits = device.limits();
        let max_storage_binding = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;
        for (what, bytes) in [("splat_buffer", splat_bytes), ("out_buffer", out_bytes)] {
            if bytes > max_storage_binding {
                return Err(GpuRelightError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_storage_binding,
                });
            }
            if bytes > max_buffer {
                return Err(GpuRelightError::ExceedsDeviceLimits {
                    what,
                    requested: bytes,
                    limit: max_buffer,
                });
            }
        }

        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relight_gpu_splats"),
            size: splat_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relight_gpu_params"),
            size: std::mem::size_of::<RelightParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relight_gpu_out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relight_gpu_out_readback"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("relight_gpu.wgsl"));

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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relight_gpu_bgl"),
            entries: &[
                storage_ro(0), // splats (baked radiance)
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
            label: Some("relight_gpu_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relight_gpu_pipeline"),
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
            params_buffer,
            out_buffer,
            out_readback,
            max_splats,
            max_storage_binding,
            adapter_name,
        })
    }

    fn bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relight_gpu_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.out_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// One-shot relight matching `relight_scene(splats, settings)` for the
    /// ambient-only / no-shadow configuration (`with_sky_ambient(true)`,
    /// `with_shadows(false)`) across Preset/CIE illuminants where
    /// `sun_direction()` is `None`.
    ///
    /// Derives the per-splat intrinsic on-device (`baked ÷ max(ref_spd, floor)`,
    /// UNCLAMPED as the oracle is), multiplies by `target_spd`, adds the
    /// pre-weighted `ambient` (`solar_irradiance × AMBIENT_FILL_WEIGHT`, or zeros
    /// if sky off), f16-stores via `pack2x16float`, and reads back. Returns a
    /// fresh `Vec<GaussianSplat>` (input cloned, only the spectral field changed),
    /// exactly like `relight_scene`.
    ///
    /// `ref_spd` / `target_spd` come from [`crate::relight::IlluminantSpec::spd`];
    /// `ambient` must already be `× AMBIENT_FILL_WEIGHT` (or all zeros). `floor`
    /// is [`crate::relight::RelightSettings::floor`] (1e-3 default).
    ///
    /// Threading: call from any thread; internally one encoder, one submit, one
    /// poll. Panics: never. Over `max_splats` →
    /// [`GpuRelightError::ExceedsDeviceLimits`] (caller checks, like the GI twin).
    pub fn relight(
        &self,
        splats: &[GaussianSplat],
        ref_spd: &[f32; 16],
        target_spd: &[f32; 16],
        ambient: &[f32; 16],
        floor: f32,
    ) -> Result<Vec<GaussianSplat>, GpuRelightError> {
        let n = splats.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n as u32 > self.max_splats {
            return Err(GpuRelightError::ExceedsDeviceLimits {
                what: "splat_buffer (scene exceeds max_splats)",
                requested: n as u64 * std::mem::size_of::<GpuRelightSplat>() as u64,
                limit: self.max_splats as u64 * std::mem::size_of::<GpuRelightSplat>() as u64,
            });
        }
        // Defensive: validate the actual bound ranges against the storage limit.
        let out_range = n as u64 * OUT_U32_PER_SPLAT * 4;
        if out_range > self.max_storage_binding {
            return Err(GpuRelightError::ExceedsDeviceLimits {
                what: "out_buffer",
                requested: out_range,
                limit: self.max_storage_binding,
            });
        }

        // Decode each splat's baked radiance host-side with the SAME `half`
        // decode the oracle's `read_radiance` uses (`spectral_f32`), so the GPU
        // input is bit-identical to the CPU input.
        let gpu_splats: Vec<GpuRelightSplat> = splats
            .iter()
            .map(|s| GpuRelightSplat {
                baked: std::array::from_fn(|b| s.spectral_f32(b)),
            })
            .collect();
        self.queue.write_buffer(
            &self.splat_buffer,
            0,
            bytemuck::cast_slice(&gpu_splats),
        );

        let params = RelightParams {
            target_spd: *target_spd,
            ambient: *ambient,
            ref_spd: *ref_spd,
            splat_count: n as u32,
            floor,
            f16_max: half::f16::MAX.to_f32(),
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = self.bind_group();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("relight_gpu_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("relight_gpu_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        let copy_bytes = n as u64 * OUT_U32_PER_SPLAT * 4;
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.out_readback, 0, copy_bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = self.map_read(&self.out_readback, copy_bytes)?;
        let packed: &[u32] = bytemuck::cast_slice(&data);

        // Unpack the 8 u32 / splat into 16 f16 bits and write into cloned splats.
        let mut out = splats.to_vec();
        for (i, splat) in out.iter_mut().enumerate() {
            let base = i * OUT_U32_PER_SPLAT as usize;
            let bits = splat.spectral_mut();
            for p in 0..8usize {
                let word = packed[base + p];
                bits[p * 2] = (word & 0xFFFF) as u16;
                bits[p * 2 + 1] = (word >> 16) as u16;
            }
        }
        self.out_readback.unmap();
        Ok(out)
    }

    /// Adapter human name (for diagnostics / benches).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    fn map_read(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
    ) -> Result<Vec<u8>, GpuRelightError> {
        let slice = buffer.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuRelightError::Readback(e.to_string())),
            Err(e) => return Err(GpuRelightError::Readback(e.to_string())),
        }
        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relight::{
        relight_scene, CieIlluminant, IlluminantSpec, PresetIlluminant, RelightSettings,
    };

    /// Weight of the sky-ambient FILL term — must equal the oracle's private
    /// `AMBIENT_FILL_WEIGHT` (relight.rs:50). Pinned to 0.5 here so the host-side
    /// ambient matches the CPU `relight_scene` None-sun branch exactly; if the
    /// oracle ever changes this, `gpu_relight_matches_cpu_oracle` diverges and fails.
    const AMBIENT_FILL_WEIGHT: f32 = 0.5;
    use crate::spectral_atmosphere::SpectralAtmosphere;
    use glam::Quat;
    use half::f16;
    use vox_data::spectral_capture::{forward_rgb, LightSpd};

    const BANDS: usize = 16;

    fn f16_max() -> f32 {
        f16::MAX.to_f32()
    }

    /// Skip a GPU test gracefully if this box truly has no GPU (CI without one).
    fn try_gpu(max_splats: u32) -> Option<GpuRelight> {
        match GpuRelight::new(max_splats) {
            Ok(g) => {
                eprintln!("[relight_gpu test] adapter: {}", g.adapter_name);
                Some(g)
            }
            Err(GpuRelightError::NoAdapter) => {
                eprintln!("[gpu_relight] no adapter — skipping");
                None
            }
            Err(e) => panic!("unexpected GPU init error on a box with a GPU: {e}"),
        }
    }

    /// Encode per-band radiance to f16 bits — mirrors the oracle's private
    /// `encode_radiance` so the test scene is built exactly as `relight.rs`.
    fn encode_radiance(radiance: &[f32; 16]) -> [u16; 16] {
        let max = f16_max();
        std::array::from_fn(|b| {
            let r = radiance[b];
            let safe = if r.is_nan() { 0.0 } else { r.clamp(0.0, max) };
            f16::from_f32(safe).to_bits()
        })
    }

    fn forward_band(intrinsic: &[f32; 16], light: &[f32; 16]) -> [f32; 16] {
        std::array::from_fn(|b| intrinsic[b] * light[b])
    }

    fn splat_with_radiance(pos: [f32; 3], radiance: &[f32; 16]) -> GaussianSplat {
        let bits = encode_radiance(radiance);
        GaussianSplat::volume(pos, [0.1, 0.1, 0.1], Quat::IDENTITY, 255, bits)
    }

    /// The 0.5-weighted sky ambient the oracle computes for the None-sun branch:
    /// `SpectralAtmosphere::earth().solar_irradiance() × AMBIENT_FILL_WEIGHT`
    /// (relight.rs:519-526,609). Bound pre-multiplied so the shader does one add.
    fn earth_ambient() -> [f32; 16] {
        let irr = SpectralAtmosphere::earth().solar_irradiance();
        std::array::from_fn(|b| irr[b] * AMBIENT_FILL_WEIGHT)
    }

    /// Build the 100k surface-slab scene from `relight.rs`'s cost bench
    /// (`relight_100k_cost_budget`): grey-intrinsic ⊙ tungsten baked into a thin
    /// slab over a 100×100 XZ ground plane.
    fn build_100k_scene() -> Vec<GaussianSplat> {
        let tungsten = LightSpd::tungsten().0;
        let baked = forward_band(&[0.5; 16], &tungsten);
        let n = 100_000usize;
        let side = 316usize;
        (0..n)
            .map(|i| {
                let gx = (i % side) as f32 / side as f32 * 100.0;
                let gz = (i / side) as f32 / side as f32 * 100.0;
                let gy = ((i.wrapping_mul(2654435761)) % 200) as f32 / 100.0;
                splat_with_radiance([gx, gy, gz], &baked)
            })
            .collect()
    }

    /// THE VALIDATION (Done-When gate, correctness ONLY): the GPU relit output
    /// matches `relight_scene()`'s output to `<1e-6` per band across the 4
    /// non-identity illuminants the verifier named (tungsten ref → daylight,
    /// cool_led, neutral, d65), in the ambient-only / no-shadow config.
    ///
    /// Per the verifier's BINDING timing correction, this gate asserts ONLY
    /// correctness (bit-exact-vs-CPU within tolerance). Any millisecond/timing
    /// assertion lives in the SEPARATE `#[ignore]` perf bench below — the gate
    /// must never flake on wall-clock.
    #[test]
    fn gpu_relight_matches_cpu_oracle() {
        let splats = build_100k_scene();
        let Some(gpu) = try_gpu(splats.len() as u32) else {
            return;
        };

        let tungsten = IlluminantSpec::Preset(PresetIlluminant::Tungsten);
        let ref_spd = tungsten.spd();
        let ambient = earth_ambient();

        let targets: [IlluminantSpec; 4] = [
            IlluminantSpec::Preset(PresetIlluminant::Daylight),
            IlluminantSpec::Preset(PresetIlluminant::CoolLed),
            IlluminantSpec::Preset(PresetIlluminant::Neutral),
            IlluminantSpec::Cie(CieIlluminant::D65),
        ];

        let mut global_max_dev = 0.0f32;
        for target in &targets {
            // CPU oracle: the exact ambient-only / no-shadow config.
            let settings = RelightSettings::new(tungsten.clone(), target.clone())
                .with_sky_ambient(true)
                .with_shadows(false);
            let (cpu_out, _report) = relight_scene(&splats, &settings);

            let target_spd = target.spd();
            let gpu_out = gpu
                .relight(&splats, &ref_spd, &target_spd, &ambient, 1e-3)
                .expect("gpu relight");
            assert_eq!(cpu_out.len(), gpu_out.len());

            let mut max_dev = 0.0f32;
            for (c, g) in cpu_out.iter().zip(gpu_out.iter()) {
                for b in 0..BANDS {
                    let d = (g.spectral_f32(b) - c.spectral_f32(b)).abs();
                    if d > max_dev {
                        max_dev = d;
                    }
                }
            }
            eprintln!("[gpu_relight] target={} max|Δ/band|={max_dev:e}", target.name());
            if max_dev > global_max_dev {
                global_max_dev = max_dev;
            }
        }

        eprintln!(
            "[gpu_relight] illuminants={} splats={} max|Δ/band|={global_max_dev:e} ({})",
            targets.len(),
            splats.len(),
            gpu.adapter_name
        );
        // Decoded host-side input + identical f32 op order + pack2x16float f16
        // store that bit-matches half::f16::from_f32 → the only error source is
        // f16 quantization, applied IDENTICALLY by CPU and GPU. Measured ~0 on
        // RADV; assert the same <1e-6 bar the other twins hold.
        assert!(
            global_max_dev < 1e-6,
            "GPU<->CPU max per-band deviation {global_max_dev:e} exceeds 1e-6"
        );
    }

    /// Bit-identical determinism: two dispatches of the same scene/illuminant
    /// produce byte-identical f16 spectral fields.
    #[test]
    fn gpu_relight_is_deterministic() {
        let tungsten = LightSpd::tungsten().0;
        let baked = forward_band(&[0.5; 16], &tungsten);
        let splats: Vec<GaussianSplat> = (0..2048)
            .map(|i| splat_with_radiance([i as f32 * 0.01, 0.0, 0.0], &baked))
            .collect();
        let Some(gpu) = try_gpu(splats.len() as u32) else {
            return;
        };
        let ref_spd = IlluminantSpec::Preset(PresetIlluminant::Tungsten).spd();
        let target_spd = IlluminantSpec::Preset(PresetIlluminant::CoolLed).spd();
        let ambient = earth_ambient();

        let a = gpu
            .relight(&splats, &ref_spd, &target_spd, &ambient, 1e-3)
            .expect("dispatch a");
        let b = gpu
            .relight(&splats, &ref_spd, &target_spd, &ambient, 1e-3)
            .expect("dispatch b");
        assert_eq!(a.len(), b.len());
        for (i, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            for band in 0..BANDS {
                assert_eq!(
                    sa.spectral()[band],
                    sb.spectral()[band],
                    "band {band} differs at splat {i}: not bit-identical across dispatches"
                );
            }
        }
        eprintln!("[gpu_relight] determinism: 2048 splats byte-identical across two dispatches");
    }

    /// f16 store mirrors CPU `encode_radiance` (no inf, over-max → f16_max).
    /// Ports `relight_bright_band_clamps_to_f16_max_not_inf` (relight.rs:1011):
    /// a splat with baked b4=60000 (intrinsic b4 ≈ 214k under tungsten ref b4),
    /// relit to daylight, must store a FINITE f16_max — never +inf.
    #[test]
    fn gpu_relight_bright_band_clamps_to_f16_max() {
        let mut baked = [0.5f32; 16];
        baked[4] = 60000.0;
        let splats = vec![splat_with_radiance([0.0, 0.0, 0.0], &baked)];
        let Some(gpu) = try_gpu(8) else {
            return;
        };
        let ref_spd = IlluminantSpec::Preset(PresetIlluminant::Tungsten).spd();
        let target_spd = IlluminantSpec::Preset(PresetIlluminant::Daylight).spd();
        // Sky off for this clamp probe (mirror the oracle test's with_sky_ambient(false)).
        let ambient = [0.0f32; 16];

        let out = gpu
            .relight(&splats, &ref_spd, &target_spd, &ambient, 1e-3)
            .expect("gpu relight");
        let stored_b4 = out[0].spectral_f32(4);
        eprintln!("[gpu_relight] bright-band stored b4 = {stored_b4}");
        assert!(stored_b4.is_finite(), "stored b4 must be finite, got {stored_b4}");
        assert_eq!(stored_b4, f16_max(), "stored b4 must be clamped to f16 max");

        // Cross-check the GPU equals the CPU oracle on this exact clamp scene.
        let settings = RelightSettings::new(
            IlluminantSpec::Preset(PresetIlluminant::Tungsten),
            IlluminantSpec::Preset(PresetIlluminant::Daylight),
        )
        .with_sky_ambient(false)
        .with_shadows(false);
        let (cpu_out, _r) = relight_scene(&splats, &settings);
        assert_eq!(cpu_out[0].spectral_f32(4), stored_b4, "GPU clamp must equal CPU clamp");
    }

    /// Metamerism survives the GPU round-trip + f16 store: two intrinsic bases
    /// that are metamers under neutral light diverge under cool_led after the GPU
    /// relight pass. Ports the structurally-impossible-in-RGB property.
    #[test]
    fn gpu_relight_preserves_metamers() {
        // A sharp single/double-band metamer pair, searched as relight.rs's
        // metamer_pair() does: base = one-band spike, alt = two-band over a flat
        // 0.2 baseline, neutral-light RGB distance < 0.01, max cool_led divergence.
        let neutral = LightSpd::neutral();
        let cool = LightSpd::cool_led();
        let levels = [0.2f32, 0.4, 0.6, 0.8, 1.0];
        let mut best: Option<([f32; 16], [f32; 16], f32)> = None;
        for i in 2..=12 {
            let mut base = [0.2f32; 16];
            base[i] = (base[i] + 0.7).min(1.0);
            let rn = forward_rgb(&base, &neutral);
            for j in 2..=12 {
                for k in (j + 1)..=12 {
                    for &aj in &levels {
                        for &ak in &levels {
                            let mut alt = [0.2f32; 16];
                            alt[j] = (alt[j] + aj).min(1.0);
                            alt[k] = (alt[k] + ak).min(1.0);
                            let an = forward_rgb(&alt, &neutral);
                            let nd: f32 = (0..3).map(|c| (an[c] - rn[c]).powi(2)).sum::<f32>().sqrt();
                            if nd < 0.01 {
                                let rc = forward_rgb(&base, &cool);
                                let ac = forward_rgb(&alt, &cool);
                                let cd: f32 =
                                    (0..3).map(|c| (ac[c] - rc[c]).powi(2)).sum::<f32>().sqrt();
                                if best.as_ref().map(|(_, _, d)| cd > *d).unwrap_or(true) {
                                    best = Some((base, alt, cd));
                                }
                            }
                        }
                    }
                }
            }
        }
        let (base, alt, _) = best.expect("should find a sharp neutral-light metamer");

        let Some(gpu) = try_gpu(8) else {
            return;
        };

        // Bake each base ⊙ neutral into a splat. Relighting reference == neutral,
        // target == cool_led, sky OFF (a pure intrinsic·light multiply, so the
        // GPU output decodes back to base ⊙ cool_led — the same render path as
        // forward_rgb(base, cool)).
        let neutral_spd = neutral.0;
        let cool_spd = cool.0;
        let base_splat = splat_with_radiance([0.0, 0.0, 0.0], &forward_band(&base, &neutral_spd));
        let alt_splat = splat_with_radiance([1.0, 0.0, 0.0], &forward_band(&alt, &neutral_spd));
        let splats = vec![base_splat, alt_splat];

        let out = gpu
            .relight(&splats, &neutral_spd, &cool_spd, &[0.0; 16], 1e-3)
            .expect("gpu relight");

        // Decode the GPU-relit radiance back to intrinsic (÷ cool) and observe
        // through forward_rgb — the same render-consistent observer the oracle uses.
        let decode_intrinsic = |s: &GaussianSplat| -> [f32; 16] {
            std::array::from_fn(|b| s.spectral_f32(b) / cool_spd[b].max(1e-6))
        };
        let gpu_base_i = decode_intrinsic(&out[0]);
        let gpu_alt_i = decode_intrinsic(&out[1]);
        let rgb_base = forward_rgb(&gpu_base_i, &cool);
        let rgb_alt = forward_rgb(&gpu_alt_i, &cool);
        let led_dist: f32 = (0..3)
            .map(|c| (rgb_alt[c] - rgb_base[c]).powi(2))
            .sum::<f32>()
            .sqrt();

        // Neutral-light distance of the ORIGINAL intrinsic pair (the metamer
        // property they were searched to satisfy).
        let rn_base = forward_rgb(&base, &neutral);
        let rn_alt = forward_rgb(&alt, &neutral);
        let neutral_dist: f32 = (0..3)
            .map(|c| (rn_alt[c] - rn_base[c]).powi(2))
            .sum::<f32>()
            .sqrt();

        eprintln!(
            "[gpu_relight] metamer divergence (GPU): neutral={neutral_dist:.4} cool_led={led_dist:.4}"
        );
        assert!(
            neutral_dist < 0.012,
            "pair must be metameric under neutral light, got {neutral_dist:.4}"
        );
        assert!(
            led_dist > 0.03,
            "metamers must diverge under cool_led on the GPU output, got {led_dist:.4}"
        );
    }

    /// CONTRACT (no-panic): impossible device limits → `Err`, never a panic.
    /// Mirrors `spectral_gi::gpu_gi_falls_back_on_impossible_limits`.
    #[test]
    fn gpu_relight_falls_back_on_impossible_limits() {
        let bad = wgpu::Limits {
            max_storage_buffers_per_shader_stage: u32::MAX,
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u32::MAX,
            max_compute_workgroups_per_dimension: u32::MAX,
            ..wgpu::Limits::default()
        };
        let res = GpuRelight::new_with_limits(64, bad);
        match res {
            Err(GpuRelightError::DeviceCreation(_)) | Err(GpuRelightError::NoAdapter) => {}
            Err(other) => panic!("expected device-creation/no-adapter error, got {other}"),
            Ok(_) => panic!("impossible limits must not yield a working device"),
        }
    }

    /// CONTRACT (class-consistency with splat_rt_gpu / many_light_gpu): an
    /// over-`max_splats` relight returns [`GpuRelightError::ExceedsDeviceLimits`]
    /// (NOT a panic), and the kernel stays usable after the rejection.
    #[test]
    fn gpu_relight_oversized_returns_error_not_panic() {
        let Some(gpu) = try_gpu(4) else {
            return;
        };
        let tungsten = LightSpd::tungsten().0;
        let baked = forward_band(&[0.5; 16], &tungsten);
        let splats: Vec<GaussianSplat> = (0..100)
            .map(|i| splat_with_radiance([i as f32 * 0.01, 0.0, 0.0], &baked))
            .collect();
        let ref_spd = IlluminantSpec::Preset(PresetIlluminant::Tungsten).spd();
        let target_spd = IlluminantSpec::Preset(PresetIlluminant::Daylight).spd();
        let err = gpu
            .relight(&splats, &ref_spd, &target_spd, &earth_ambient(), 1e-3)
            .expect_err("oversized relight must return an error, not abort");
        match err {
            GpuRelightError::ExceedsDeviceLimits { what, requested, limit } => {
                assert_eq!(what, "splat_buffer (scene exceeds max_splats)");
                assert!(requested > limit, "requested {requested} must exceed limit {limit}");
            }
            other => panic!("expected ExceedsDeviceLimits, got {other:?}"),
        }
        // Still usable: a valid relight succeeds (no abort happened).
        let ok: Vec<GaussianSplat> = (0..3)
            .map(|i| splat_with_radiance([i as f32, 0.0, 0.0], &baked))
            .collect();
        let _ = gpu
            .relight(&ok, &ref_spd, &target_spd, &earth_ambient(), 1e-3)
            .expect("valid relight after rejection");
    }

    /// PERF BENCH (SEPARATE, `#[ignore]`): the 100k-splat dispatch timing,
    /// PRINTED, never asserted as a gate — per the verifier's binding correction
    /// (first-dispatch shader compile + buffer alloc + poll(Wait) make wall-clock
    /// timing non-deterministic across machines, so it must not gate the green
    /// test). Mirrors `relight_100k_cost_budget` / the atom_budget ignored bench.
    #[test]
    #[ignore = "perf bench — run explicitly with --ignored --nocapture"]
    fn gpu_relight_100k_dispatch_timing() {
        let splats = build_100k_scene();
        let Some(gpu) = try_gpu(splats.len() as u32) else {
            eprintln!("[gpu_relight] no adapter — cannot bench");
            return;
        };
        let ref_spd = IlluminantSpec::Preset(PresetIlluminant::Tungsten).spd();
        let target_spd = IlluminantSpec::Preset(PresetIlluminant::Daylight).spd();
        let ambient = earth_ambient();

        // Warm up (first dispatch pays shader compile + first buffer alloc).
        let _ = gpu
            .relight(&splats, &ref_spd, &target_spd, &ambient, 1e-3)
            .expect("warmup relight");

        let t0 = std::time::Instant::now();
        let _ = gpu
            .relight(&splats, &ref_spd, &target_spd, &ambient, 1e-3)
            .expect("timed relight");
        let dispatch_ms = t0.elapsed().as_secs_f64() * 1e3;

        eprintln!(
            "[gpu_relight] splats={} dispatch(incl readback)={dispatch_ms:.3} ms ({})",
            splats.len(),
            gpu.adapter_name
        );
    }
}
