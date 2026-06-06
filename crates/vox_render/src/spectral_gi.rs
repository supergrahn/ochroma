//! Real-time spectral global illumination via splat radiance cache.
//! Each frame: gather emissive radiance from N nearest splats (distance-weighted),
//! modulate by receiving splat's reflectance, blend into a temporal cache.

use vox_core::types::GaussianSplat;
use half::f16;

use crate::spectral_atmosphere::SpectralAtmosphere;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatGiEntry {
    pub position: [f32; 3],
    pub emissive: [f32; 16],
    pub reflectance: [f32; 16],
}

/// Accumulated radiance probe — result of a `gather_radiance` call stored for inspection or GI baking.
#[derive(Clone, Debug)]
pub struct GiProbe {
    pub bands: [f32; 16],
}

/// Shared hour → sun-zenith mapping. The SINGLE source of truth used by BOTH
/// the CPU `EngineLoop::step_gi` (to set `SpectralAtmosphere::sun_zenith`) and
/// the GPU `GpuGi::sky_ambient_for_hour`. Keeping the formula here means the two
/// GI backends can never silently drift apart in their sky-ambient term.
///
/// Maps `hour` (0..24) to a sun elevation that peaks at noon (`FRAC_PI_2 * 1.0`)
/// and is zero from dusk through dawn.
pub fn sun_zenith_for_hour(hour: f32) -> f32 {
    let norm = (hour % 24.0) / 24.0;
    (std::f32::consts::PI * norm - std::f32::consts::FRAC_PI_2)
        .sin()
        .max(0.0)
        * std::f32::consts::FRAC_PI_2
}

pub fn gather_radiance(
    receiver_pos: [f32; 3],
    emitters: &[SplatGiEntry],
    max_range: f32,
) -> [f32; 16] {
    let mut radiance = [0.0f32; 16];
    let rp = glam::Vec3::from(receiver_pos);
    for e in emitters {
        let ep = glam::Vec3::from(e.position);
        let dist = rp.distance(ep);
        if dist < 1e-4 || dist > max_range {
            continue;
        }
        let weight = 1.0 / (dist * dist);
        // Index `b` couples three arrays (radiance, e.emissive, e.reflectance); a zip obscures the kernel.
        #[allow(clippy::needless_range_loop)]
        for b in 0..16 {
            if e.emissive[b] > 0.0 {
                radiance[b] += e.emissive[b] * e.reflectance[b] * weight;
            }
        }
    }
    radiance
}

/// Blend `incoming` into `cache` using EMA.
/// `alpha` = retain-old weight: alpha=0.9 means 90% old value, 10% new.
/// This matches the `propagate` method convention in `SpectralRadianceCache`.
pub fn temporal_blend(cache: &mut [f32; 16], incoming: &[f32; 16], alpha: f32) {
    for b in 0..16 {
        cache[b] = cache[b] * alpha + incoming[b] * (1.0 - alpha);
    }
}

// ---------------------------------------------------------------------------
// SpectralRadianceCache — Domain 12 core + Domain 12a extensions
// ---------------------------------------------------------------------------

pub struct SpectralRadianceCache {
    pub cache: Vec<[f32; 16]>,
    pub alpha: f32,
    pub sky_ambient: [f32; 16],
    /// Domain 12 compat field (same data as cache)
    pub entries: Vec<[f32; 16]>,
}

impl SpectralRadianceCache {
    pub fn new(splat_count: usize) -> Self {
        Self {
            cache: vec![[0.0f32; 16]; splat_count],
            alpha: 0.9,
            sky_ambient: [0.0f32; 16],
            entries: vec![[0.0f32; 16]; splat_count],
        }
    }

    pub fn set_sky(&mut self, atmo: &SpectralAtmosphere) {
        self.sky_ambient = atmo.solar_irradiance();
    }

    /// Bake per-splat sky-lit ambient radiance into the cache by sampling the
    /// physically based spectral sky model along each splat's surface normal.
    ///
    /// This is the live call path for [`SpectralAtmosphere::sky_radiance`]: a
    /// render loop calls this once the atmosphere/sun state is known to fill the
    /// radiance cache with view-dependent sky lighting before applying it to the
    /// splats. The view elevation is taken from the splat normal's up-component
    /// (`asin(n.y)`), and the azimuth from `atan2(n.z, n.x)`, so splats facing
    /// different parts of the sky receive different per-band radiance.
    ///
    /// Returns the number of cache entries written.
    pub fn propagate_sky(&mut self, splats: &[GaussianSplat], atmo: &SpectralAtmosphere) -> usize {
        self.resize(splats.len());
        // Cache the overall sky-ambient (solar) term for fallback/apply use.
        self.sky_ambient = atmo.solar_irradiance();
        let alpha = self.alpha;

        for (i, splat) in splats.iter().enumerate() {
            let n = splat.normal();
            // Elevation above horizon from the up-component of the normal.
            let up = n[1].clamp(-1.0, 1.0);
            let elevation = up.asin(); // [-PI/2, PI/2]
            // Only the upper hemisphere sees sky; clamp to a small positive
            // floor so down-facing splats still get the (reddest) horizon band.
            let view_elev = elevation.max(0.001);
            let azimuth = n[2].atan2(n[0]);

            let sky = atmo.sky_radiance(view_elev, azimuth);
            for (c, &s) in self.cache[i].iter_mut().zip(sky.iter()) {
                *c = alpha * *c + (1.0 - alpha) * s;
            }
            self.entries[i] = self.cache[i];
        }
        splats.len()
    }

    pub fn resize(&mut self, count: usize) {
        self.cache.resize(count, [0.0f32; 16]);
        self.entries.resize(count, [0.0f32; 16]);
    }

    /// Domain 12 propagation: gather from SplatGiEntry slice using free function
    pub fn propagate_gi(&mut self, gi_entries: &[SplatGiEntry], max_range: f32) {
        let alpha = self.alpha;
        for (i, entry) in gi_entries.iter().enumerate() {
            if i >= self.entries.len() {
                break;
            }
            let incoming = gather_radiance(entry.position, gi_entries, max_range);
            temporal_blend(&mut self.entries[i], &incoming, alpha);
            self.cache[i] = self.entries[i];
        }
    }

    /// Domain 12a propagation: gather directly from GaussianSplat slice
    pub fn propagate(&mut self, splats: &[GaussianSplat], max_emitters: usize) {
        self.resize(splats.len());
        let emitters: Vec<SplatGiEntry> = splats
            .iter()
            .filter(|s| s.opacity() > 128)
            .take(max_emitters)
            .map(|s| {
                let emissive = decode_spectral(s.spectral());
                SplatGiEntry {
                    position: s.position(),
                    emissive,
                    reflectance: [0.5f32; 16],
                }
            })
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
                let dist_sq = (dx * dx + dy * dy + dz * dz).max(0.01);
                let weight = 1.0 / dist_sq;
                for (inc, &em) in incoming.iter_mut().zip(emitter.emissive.iter()) {
                    *inc += em * weight;
                }
            }
            let max_incoming = incoming.iter().copied().fold(f32::EPSILON, f32::max);
            let scale = if max_incoming > 1.0 { 1.0 / max_incoming } else { 1.0 };
            for (c, &inc) in self.cache[i].iter_mut().zip(incoming.iter()) {
                *c = alpha * *c + (1.0 - alpha) * (inc * scale).clamp(0.0, 1.0);
            }
            self.entries[i] = self.cache[i];
        }
    }

    pub fn apply(&self, splats: &[GaussianSplat]) -> Vec<GaussianSplat> {
        splats
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let irr = if i < self.cache.len() {
                    self.cache[i]
                } else {
                    self.sky_ambient
                };
                let mut out = *s;
                let spectral = decode_spectral(s.spectral());
                for b in 0..16 {
                    let modulated = (spectral[b] + irr[b] * 0.5).clamp(0.0, 1.0);
                    out.spectral_mut()[b] = f16::from_f32(modulated).to_bits();
                }
                out
            })
            .collect()
    }
}

fn decode_spectral(s: &[u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = f16::from_bits(s[i]).to_f32();
    }
    out
}

// ---------------------------------------------------------------------------
// GPU GI pass structs — Domain 12a Task 4
// ---------------------------------------------------------------------------

/// GPU-layout splat entry for compute shader.
/// Layout: 4 (position + pad) + 64 (radiance) + 64 (reflectance) = 144 bytes total.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct GpuSplatEntry {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub radiance: [f32; 16],
    pub reflectance: [f32; 16],
}

const _: () = assert!(std::mem::size_of::<GpuSplatEntry>() == 144);

/// Emitter-prefix bound shared by the CPU `propagate` (`take(MAX_EMITTERS)`) and
/// the GPU pass. Both paths sum over the FIRST `MAX_EMITTERS` emitter splats in
/// buffer order — the same documented selection rule, so they never diverge.
pub const MAX_EMITTERS: u32 = 256;

/// GI compute-pass uniform. Layout mirrors the WGSL `GiParams` (std140):
/// 4 × u32 header (16 bytes) followed by 16 sky-ambient bands packed as
/// 4 × vec4<f32> (64 bytes) = 80 bytes total.
///
/// There is no `alpha` field: `GpuGi` is stateless per call (it binds no
/// previous-frame radiance), so the CPU temporal-EMA `alpha` has no GPU
/// counterpart. The pass always fully replaces the radiance, exactly matching a
/// CPU `propagate` on a fresh (zeroed) cache — i.e. alpha = 0.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiParamsUniform {
    pub splat_count: u32,
    pub max_emitters: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    /// Per-band sky-ambient radiance (mirrors `SpectralRadianceCache::set_sky`
    /// → `SpectralAtmosphere::solar_irradiance`). 16 bands as 4 × vec4.
    pub sky_ambient: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<GiParamsUniform>() == 80);

pub struct GpuGiPass {
    pub splat_buffer: wgpu::Buffer,
    pub radiance_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    /// CPU-mappable staging buffer for reading the radiance back.
    pub readback_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pub max_splats: u32,
}

impl GpuGiPass {
    pub fn new(device: &wgpu::Device, max_splats: u32) -> Self {
        let splat_bytes = max_splats as u64 * std::mem::size_of::<GpuSplatEntry>() as u64;
        let radiance_bytes = max_splats as u64 * 16 * 4;

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
            size: std::mem::size_of::<GiParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi_readback_buf"),
            size: radiance_bytes.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("gpu/spectral_gi_pass.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: radiance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
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

        Self {
            splat_buffer,
            radiance_buffer,
            params_buffer,
            readback_buffer,
            pipeline,
            bind_group,
            max_splats,
        }
    }

    /// Encode the GI compute pass into `encoder` plus a copy of the radiance
    /// storage buffer into the CPU-mappable readback buffer. Caller submits the
    /// encoder and maps `readback_buffer` to retrieve results.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        splats_gpu: &[GpuSplatEntry],
        splat_count: u32,
        max_emitters: u32,
        sky_ambient: [f32; 16],
    ) {
        let count = splat_count.min(self.max_splats);
        queue.write_buffer(&self.splat_buffer, 0, bytemuck::cast_slice(splats_gpu));
        let mut sky_packed = [[0.0f32; 4]; 4];
        for b in 0..16 {
            sky_packed[b / 4][b % 4] = sky_ambient[b];
        }
        let params = GiParamsUniform {
            splat_count: count,
            max_emitters,
            _pad0: 0,
            _pad1: 0,
            sky_ambient: sky_packed,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let copy_bytes = count as u64 * 16 * 4;
        if copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.radiance_buffer,
                0,
                &self.readback_buffer,
                0,
                copy_bytes,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GpuGi — high-level, headless, drop-in GPU global illumination
// ---------------------------------------------------------------------------

/// Error returned when the GPU GI engine cannot be created or run. The caller
/// can use this to fall back to the CPU path — `GpuGi` never panics on a missing
/// or inadequate GPU.
#[derive(Debug, Clone)]
pub enum GpuGiError {
    /// No wgpu adapter (no GPU / no driver) could be found.
    NoAdapter,
    /// An adapter was found but device creation failed (e.g. limits too high).
    DeviceCreation(String),
    /// Mapping the readback buffer failed.
    Readback(String),
}

impl std::fmt::Display for GpuGiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuGiError::NoAdapter => write!(f, "no GPU adapter available"),
            GpuGiError::DeviceCreation(e) => write!(f, "GPU device creation failed: {e}"),
            GpuGiError::Readback(e) => write!(f, "GPU readback failed: {e}"),
        }
    }
}

impl std::error::Error for GpuGiError {}

/// Headless GPU spectral global illumination engine.
///
/// Owns its own wgpu device/queue (no window/surface needed) and exposes
/// [`GpuGi::step`], a drop-in replacement for the CPU `EngineLoop::step_gi`:
/// upload splats, run the GI compute pass, read back GI-lit splats.
pub struct GpuGi {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass: GpuGiPass,
    capacity: u32,
    /// Adapter human name, for diagnostics / benches.
    pub adapter_name: String,
}

impl GpuGi {
    /// Create a headless GPU GI engine sized for up to `max_splats` splats.
    ///
    /// Returns [`GpuGiError`] (never panics) if no adapter is found or device
    /// creation fails, so the caller can stay on the CPU path.
    pub fn new(max_splats: u32) -> Result<Self, GpuGiError> {
        Self::new_with_limits(max_splats, wgpu::Limits::default())
    }

    /// Like [`GpuGi::new`] but with caller-chosen device limits. Used by the
    /// fallback test to force device creation to fail with impossible limits.
    pub fn new_with_limits(
        max_splats: u32,
        required_limits: wgpu::Limits,
    ) -> Result<Self, GpuGiError> {
        pollster::block_on(Self::new_async(max_splats, required_limits))
    }

    /// Construct with deliberately impossible device limits so device creation
    /// fails (or no adapter is found). Used by downstream fallback tests that
    /// cannot depend on `wgpu` directly to assert the no-panic `Err` contract.
    /// Always returns `Err` on real hardware.
    pub fn new_failing_for_test() -> Result<Self, GpuGiError> {
        Self::new_with_limits(
            64,
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: u32::MAX,
                max_buffer_size: u64::MAX,
                max_storage_buffer_binding_size: u32::MAX,
                max_compute_workgroups_per_dimension: u32::MAX,
                ..wgpu::Limits::default()
            },
        )
    }

    async fn new_async(
        max_splats: u32,
        required_limits: wgpu::Limits,
    ) -> Result<Self, GpuGiError> {
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
            .ok_or(GpuGiError::NoAdapter)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu_gi_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuGiError::DeviceCreation(e.to_string()))?;

        let capacity = max_splats.max(1);
        let pass = GpuGiPass::new(&device, capacity);
        Ok(Self {
            device,
            queue,
            pass,
            capacity,
            adapter_name,
        })
    }

    /// Pack a `GaussianSplat` into a `GpuSplatEntry`. `radiance` carries the
    /// decoded spectral; `reflectance[0]` is the emitter flag (opacity > 128).
    fn pack(splat: &GaussianSplat) -> GpuSplatEntry {
        let spectral = decode_spectral(splat.spectral());
        let mut reflectance = [0.0f32; 16];
        reflectance[0] = if splat.opacity() > 128 { 1.0 } else { 0.0 };
        GpuSplatEntry {
            position: splat.position(),
            _pad0: 0.0,
            radiance: spectral,
            reflectance,
        }
    }

    /// Compute the per-band sky-ambient radiance for a given hour, bit-for-bit
    /// mirroring `EngineLoop::step_gi`: map `hour` → sun elevation, then take the
    /// atmosphere's `solar_irradiance()` (which is what `set_sky` caches as the
    /// `sky_ambient` term the CPU `propagate` seeds `incoming` with).
    pub fn sky_ambient_for_hour(hour: f32) -> [f32; 16] {
        let sun_zenith = sun_zenith_for_hour(hour);
        let mut atmo = SpectralAtmosphere::earth();
        atmo.sun_zenith = sun_zenith;
        atmo.sun_elevation = sun_zenith;
        atmo.solar_irradiance()
    }

    /// Drop-in GPU equivalent of `EngineLoop::step_gi`.
    ///
    /// Uploads `splats`, runs the spectral GI compute pass, and returns the
    /// GI-lit splats (positions/geometry preserved, per-band spectral lifted by
    /// indirect radiance). Mirrors the CPU semantics: bright opaque emitters
    /// (opacity > 128) cast spectral radiance onto receivers, weighted by
    /// inverse-square distance, and the result is written into each splat's
    /// spectral as `clamp(spectral + irr * 0.5, 0, 1)`.
    ///
    /// `hour` drives the sky-ambient term exactly as the CPU `EngineLoop::step_gi`
    /// does: `hour` → sun elevation → `SpectralAtmosphere::solar_irradiance`,
    /// which seeds each receiver's `incoming` before the emitter sum (mirroring
    /// `SpectralRadianceCache::set_sky` + the `let mut incoming = sky;` line in
    /// `propagate`).
    pub fn step(&self, splats: &[GaussianSplat], hour: f32) -> Result<Vec<GaussianSplat>, GpuGiError> {
        if splats.is_empty() {
            return Ok(Vec::new());
        }
        let count = (splats.len() as u32).min(self.capacity);
        let n = count as usize;

        let gpu_entries: Vec<GpuSplatEntry> = splats[..n].iter().map(Self::pack).collect();
        let sky_ambient = Self::sky_ambient_for_hour(hour);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_gi_encoder"),
            });
        self.pass.dispatch(
            &mut encoder,
            &self.queue,
            &gpu_entries,
            count,
            MAX_EMITTERS,
            sky_ambient,
        );
        self.queue.submit(Some(encoder.finish()));

        // Map the readback buffer and wait for the GPU.
        let slice = self.pass.readback_buffer.slice(..(n as u64 * 16 * 4));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuGiError::Readback(e.to_string())),
            Err(e) => return Err(GpuGiError::Readback(e.to_string())),
        }

        let lit: Vec<[f32; 16]> = {
            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            (0..n)
                .map(|i| {
                    let mut bands = [0.0f32; 16];
                    bands.copy_from_slice(&floats[i * 16..i * 16 + 16]);
                    bands
                })
                .collect()
        };
        self.pass.readback_buffer.unmap();

        // Write the GI-lit spectral back into copies of the input splats.
        let out: Vec<GaussianSplat> = splats
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut o = *s;
                if i < n {
                    for (b, &v) in lit[i].iter().enumerate() {
                        o.spectral_mut()[b] = f16::from_f32(v).to_bits();
                    }
                }
                o
            })
            .collect();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn make_splat(pos: [f32; 3], spectral_val: f32, opacity: u8) -> GaussianSplat {
        let v = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat::surface(
            pos.into(),
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            0.1,
            0.1,
            opacity,
            [v; 16],
        )
    }

    fn make_emissive_splat(pos: [f32; 3], band: usize, value: f32) -> SplatGiEntry {
        let mut spectral = [0.0f32; 16];
        spectral[band] = value;
        SplatGiEntry {
            position: pos,
            emissive: spectral,
            reflectance: [0.5; 16],
        }
    }

    // --- Domain 12 tests (free functions) ---

    #[test]
    fn nearby_emissive_contributes_radiance() {
        let emitter = make_emissive_splat([0.0, 0.0, 0.0], 3, 1.0);
        let receiver_pos = [0.5, 0.0, 0.0];
        let radiance = gather_radiance(receiver_pos, &[emitter], 2.0);
        // emitter at 0m, receiver at 0.5m, emissive[3]=1.0, reflectance=0.5
        // expected: 1.0 * 0.5 / (0.5^2) = 2.0
        assert!(
            radiance[3] > 1.0,
            "band 3 radiance should be > 1.0 for nearby emitter at 0.5m (got {})",
            radiance[3]
        );
    }

    #[test]
    fn distant_emissive_contributes_less() {
        let near = make_emissive_splat([0.1, 0.0, 0.0], 3, 1.0);
        let far = make_emissive_splat([5.0, 0.0, 0.0], 3, 1.0);
        let pos = [0.0, 0.0, 0.0];
        let r_near = gather_radiance(pos, &[near], 10.0);
        let r_far = gather_radiance(pos, &[far], 10.0);
        assert!(r_near[3] > r_far[3], "near emitter should contribute more than far");
    }

    #[test]
    fn temporal_blend_converges() {
        let mut cache = [0.0f32; 16];
        let target = [1.0f32; 16];
        for _ in 0..100 {
            temporal_blend(&mut cache, &target, 0.1);
        }
        for (i, &v) in cache.iter().enumerate() {
            assert!(v > 0.99, "band {} should converge to 1.0 after 100 steps, got {}", i, v);
        }
    }

    // --- Domain 12a tests (SpectralRadianceCache methods) ---

    #[test]
    fn cache_initialises_empty() {
        let cache = SpectralRadianceCache::new(10);
        assert_eq!(cache.cache.len(), 10);
        assert!(cache
            .cache
            .iter()
            .all(|c: &[f32; 16]| c.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn nearby_emissive_splat_adds_irradiance() {
        let mut cache = SpectralRadianceCache::new(2);
        cache.alpha = 0.0; // no temporal smoothing → new value replaces fully
        let emitter = make_splat([0.0, 0.0, 0.0], 1.0, 255);
        let receiver = make_splat([1.0, 0.0, 0.0], 0.0, 50); // low opacity = not emissive
        cache.propagate(&[emitter, receiver], 100);
        assert!(
            cache.cache[1].iter().any(|&v| v > 0.0),
            "receiver should have non-zero irradiance after propagation"
        );
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

    /// Build a surface splat whose geometric normal equals `n` (must be unit).
    /// Picks two tangents perpendicular to `n` whose cross product is `n`.
    fn splat_with_normal(pos: [f32; 3], n: [f32; 3]) -> GaussianSplat {
        let nv = glam::Vec3::from(n).normalize();
        // Any vector not parallel to nv.
        let helper = if nv.x.abs() < 0.9 { glam::Vec3::X } else { glam::Vec3::Y };
        let u = nv.cross(helper).normalize(); // u ⟂ n
        let v = nv.cross(u).normalize();       // v ⟂ n, and u × v = n
        GaussianSplat::surface(
            pos,
            u.into(),
            v.into(),
            0.1,
            0.1,
            255,
            [0u16; 16],
        )
    }

    #[test]
    fn gi_cache_is_live() {
        // The sky-radiance call path must actually run and produce DIFFERENT
        // per-band radiance for two splats that look at different parts of the
        // sky. Splat 0 faces straight up (zenith, bluest); splat 1 faces a
        // near-horizon direction (reddest). Verify the cache is not constant.
        let atmo = SpectralAtmosphere::earth();

        let up = splat_with_normal([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Near-horizon normal: mostly +X with a tiny upward tilt.
        let horizon = splat_with_normal([10.0, 0.0, 0.0], [1.0, 0.02, 0.0]);

        // Sanity: the constructed normals point where we expect.
        let n_up = up.normal();
        let n_hz = horizon.normal();
        assert!(n_up[1] > 0.99, "zenith splat normal should point up, got {:?}", n_up);
        assert!(n_hz[1] < 0.2, "horizon splat normal should be near-horizontal, got {:?}", n_hz);

        let mut cache = SpectralRadianceCache::new(0);
        cache.alpha = 0.0; // new sky value replaces fully — no temporal damping
        let written = cache.propagate_sky(&[up, horizon], &atmo);
        assert_eq!(written, 2, "propagate_sky should write both entries");

        let r_zenith = cache.cache[0];
        let r_horizon = cache.cache[1];

        // Cache must not be all-zero (sky_radiance actually ran).
        assert!(
            r_zenith.iter().any(|&v| v > 1e-3),
            "zenith radiance must be non-trivial, got {:?}",
            r_zenith
        );

        // At least one band must differ by > 1e-3 between the two lit positions —
        // proves the cache is live and view-dependent, not a constant fill.
        let max_band_delta = (0..16)
            .map(|b| (r_zenith[b] - r_horizon[b]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_band_delta > 1e-3,
            "two distinct lit positions must differ per-band by >1e-3, max delta={max_band_delta}\nzenith={r_zenith:?}\nhorizon={r_horizon:?}"
        );

        // Physical sanity: zenith is bluer (band 0 > band 15), horizon is redder
        // (its red-band fraction exceeds the zenith's), confirming sky_radiance
        // drove the per-position result rather than a uniform constant.
        assert!(
            r_zenith[0] > r_zenith[15],
            "zenith should be violet-dominant: b0={} b15={}",
            r_zenith[0], r_zenith[15]
        );
        let zenith_blue_ratio = r_zenith[0] / (r_zenith[15] + 1e-6);
        let horizon_blue_ratio = r_horizon[0] / (r_horizon[15] + 1e-6);
        assert!(
            zenith_blue_ratio > horizon_blue_ratio,
            "zenith ({zenith_blue_ratio:.3}) must be bluer than horizon ({horizon_blue_ratio:.3})"
        );
    }

    // --- GPU struct size tests ---

    #[test]
    fn gpu_splat_entry_size() {
        assert_eq!(std::mem::size_of::<GpuSplatEntry>(), 144);
    }

    #[test]
    fn gi_params_size() {
        // 4 × u32 header (16) + 4 × vec4<f32> sky bands (64) = 80 bytes.
        assert_eq!(std::mem::size_of::<GiParamsUniform>(), 80);
    }

    // --- GpuGi end-to-end tests (exercise the real GPU on this box) ---

    /// Bright opaque emitter at origin, dark receiver `d` metres away on +X.
    /// `emit` is the emitter's per-band spectral value.
    fn emitter_receiver_scene_emit(d: f32, emit: f32) -> Vec<GaussianSplat> {
        let emitter = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2],
            glam::Quat::IDENTITY,
            255,
            [f16::from_f32(emit).to_bits(); 16],
        );
        let receiver = GaussianSplat::volume(
            [d, 0.0, 0.0],
            [0.2, 0.2, 0.2],
            glam::Quat::IDENTITY,
            10,
            [f16::from_f32(0.0).to_bits(); 16],
        );
        vec![emitter, receiver]
    }

    fn emitter_receiver_scene(d: f32) -> Vec<GaussianSplat> {
        emitter_receiver_scene_emit(d, 1.0)
    }

    fn receiver_band(out: &[GaussianSplat], band: usize) -> f32 {
        f16::from_bits(out[1].spectral()[band]).to_f32()
    }

    /// Skip a GPU test gracefully if this box truly has no GPU (CI without one).
    /// On the target box (AMD 780M) this returns `Some` and the test runs.
    fn try_gpu(max_splats: u32) -> Option<GpuGi> {
        match GpuGi::new(max_splats) {
            Ok(g) => Some(g),
            Err(GpuGiError::NoAdapter) => {
                eprintln!("[gpu_gi test] no adapter — skipping GPU test");
                None
            }
            Err(e) => panic!("unexpected GPU init error on a box with a GPU: {e}"),
        }
    }

    #[test]
    fn gpu_gi_lifts_dark_receiver_off_zero() {
        let Some(gpu) = try_gpu(64) else { return };
        let scene = emitter_receiver_scene(0.5);
        // Receiver starts fully dark.
        assert_eq!(receiver_band(&scene, 8), 0.0);

        let out = gpu.step(&scene, 12.0).expect("gpu step");
        // Receiver band energies must be lifted off zero (DIRECTION matches CPU).
        let any_lit = (0..16).any(|b| receiver_band(&out, b) > 1e-3);
        assert!(any_lit, "receiver must be lit by the emitter, got {:?}",
            (0..16).map(|b| receiver_band(&out, b)).collect::<Vec<_>>());

        // Emitter stays bright (unchanged-or-similar): it was already saturated.
        let emitter_b8 = f16::from_bits(out[0].spectral()[8]).to_f32();
        assert!(emitter_b8 > 0.9, "emitter should stay bright, got {emitter_b8}");
    }

    #[test]
    fn gpu_gi_radiance_falls_off_with_distance() {
        let Some(gpu) = try_gpu(64) else { return };
        // Dim emitter (emit=0.2) so incoming stays below the normalization knee
        // (1.0) at all three distances and the inverse-square falloff is visible
        // rather than clamped — matches the CPU propagate normalization.
        let r05 = gpu.step(&emitter_receiver_scene_emit(0.5, 0.2), 12.0).expect("step");
        let r10 = gpu.step(&emitter_receiver_scene_emit(1.0, 0.2), 12.0).expect("step");
        let r20 = gpu.step(&emitter_receiver_scene_emit(2.0, 0.2), 12.0).expect("step");

        // Use a band that is not saturated at all three distances; band 8.
        let v05 = receiver_band(&r05, 8);
        let v10 = receiver_band(&r10, 8);
        let v20 = receiver_band(&r20, 8);
        assert!(v05 > 1e-3, "0.5m receiver must be lit, got {v05}");
        assert!(
            v05 > v10 && v10 > v20,
            "radiance must decrease monotonically with distance: 0.5m={v05} 1.0m={v10} 2.0m={v20}"
        );
    }

    #[test]
    fn gpu_gi_is_non_constant_and_position_dependent() {
        let Some(gpu) = try_gpu(64) else { return };
        // Two receivers at different distances in one scene.
        let emitter = GaussianSplat::volume(
            [0.0, 0.0, 0.0], [0.2, 0.2, 0.2], glam::Quat::IDENTITY, 255,
            [f16::from_f32(1.0).to_bits(); 16],
        );
        let near = GaussianSplat::volume(
            [0.4, 0.0, 0.0], [0.2, 0.2, 0.2], glam::Quat::IDENTITY, 10,
            [f16::from_f32(0.0).to_bits(); 16],
        );
        let far = GaussianSplat::volume(
            [3.0, 0.0, 0.0], [0.2, 0.2, 0.2], glam::Quat::IDENTITY, 10,
            [f16::from_f32(0.0).to_bits(); 16],
        );
        let out = gpu.step(&[emitter, near, far], 12.0).expect("step");
        let near_b = f16::from_bits(out[1].spectral()[8]).to_f32();
        let far_b = f16::from_bits(out[2].spectral()[8]).to_f32();
        assert!(
            near_b > far_b,
            "result must be position-dependent: near={near_b} far={far_b}"
        );
    }

    #[test]
    fn gpu_gi_is_deterministic() {
        let Some(gpu) = try_gpu(64) else { return };
        let scene = emitter_receiver_scene(0.5);
        let a = gpu.step(&scene, 12.0).expect("step a");
        let b = gpu.step(&scene, 12.0).expect("step b");
        for i in 0..scene.len() {
            for band in 0..16 {
                let va = f16::from_bits(a[i].spectral()[band]).to_f32();
                let vb = f16::from_bits(b[i].spectral()[band]).to_f32();
                assert!(
                    (va - vb).abs() < 1e-6,
                    "GPU GI must be deterministic: splat {i} band {band}: {va} vs {vb}"
                );
            }
        }
    }

    /// CPU reference for a single stateless `GpuGi::step`: mirrors
    /// `EngineLoop::step_gi` (set sky from `hour`, `propagate(splats, 256)`,
    /// `apply`) on a FRESH cache (alpha = 0 → full replace, no temporal damping),
    /// then quantizes through f16 exactly as the GPU readback path does. This is
    /// the semantic contract the GPU pass must reproduce per-band.
    fn cpu_step_reference(splats: &[GaussianSplat], hour: f32) -> Vec<GaussianSplat> {
        let mut cache = SpectralRadianceCache::new(splats.len());
        cache.alpha = 0.0; // stateless: fresh cache, full replace
        cache.sky_ambient = GpuGi::sky_ambient_for_hour(hour);
        cache.propagate(splats, MAX_EMITTERS as usize);
        let lit = cache.apply(splats);
        // `apply` already writes f16-quantized spectral, matching the GPU's
        // f16 store on readback. Return as-is.
        lit
    }

    /// Equivalence contract: for a scene LARGER than the emitter bound, with
    /// several emitters — including one whose index the OLD `n/cap` stride would
    /// have skipped — the GPU pass must agree per-band with the CPU `step_gi`
    /// reference within a tight f16-quantization-aware epsilon.
    ///
    /// Pre-fix this test fails for two independent reasons: (1) the old striding
    /// summed a strided emitter subset, so the dominant emitter adjacent to the
    /// probe receiver (placed at a non-stride index) was dropped → GPU radiance
    /// ≈ 0 there while CPU lights it strongly; (2) the old shader seeded
    /// `incoming = 0` instead of the sky-ambient term, so EVERY receiver's bands
    /// diverged from the CPU by the (non-zero, noon) sky contribution.
    #[test]
    fn gpu_gi_matches_cpu_step_for_large_strided_scene() {
        let n: usize = 1200;
        let Some(gpu) = try_gpu(n as u32) else { return };

        // Build a deterministic line of dark receivers along +X.
        let mut scene: Vec<GaussianSplat> = (0..n)
            .map(|i| {
                GaussianSplat::volume(
                    [i as f32 * 0.5, 0.0, 0.0],
                    [0.1, 0.1, 0.1],
                    glam::Quat::IDENTITY,
                    10, // dark receiver, opacity <= 128 → not an emitter
                    [f16::from_f32(0.0).to_bits(); 16],
                )
            })
            .collect();

        // A handful of emitters near the front (all within the first 256
        // emitters, so the prefix-take keeps them on both paths).
        let emit_band = 8usize;
        let put_emitter = |scene: &mut Vec<GaussianSplat>, idx: usize, pos: [f32; 3], v: f32| {
            let mut spectral = [f16::from_f32(0.0).to_bits(); 16];
            spectral[emit_band] = f16::from_f32(v).to_bits();
            scene[idx] = GaussianSplat::volume(
                pos,
                [0.1, 0.1, 0.1],
                glam::Quat::IDENTITY,
                255, // bright emitter
                spectral,
            );
        };
        put_emitter(&mut scene, 0, [0.0, 0.0, 0.0], 0.2);
        put_emitter(&mut scene, 7, [3.5, 0.0, 0.0], 0.2);

        // The dominant emitter for a specific probe receiver, placed at an index
        // the OLD stride (n/cap = 1200/256 = 4) would skip: 1001 % 4 == 1, so
        // k=1001 was never visited by the strided loop. It sits 0.2m off the
        // probe receiver at index 1000 → huge inverse-square contribution.
        let stride_old = (n as u32 / MAX_EMITTERS).max(1); // = 4
        assert_eq!(stride_old, 4, "scene sized so the old stride is 4");
        let dropped_emitter_idx = 1001usize;
        assert_ne!(
            dropped_emitter_idx as u32 % stride_old,
            0,
            "dominant emitter must sit at a non-stride index the old loop skipped"
        );
        let probe_receiver_idx = 1000usize;
        let probe_pos = [probe_receiver_idx as f32 * 0.5, 0.0, 0.0];
        put_emitter(
            &mut scene,
            dropped_emitter_idx,
            [probe_pos[0] + 0.2, 0.0, 0.0],
            0.5,
        );

        let hour = 12.0; // noon → non-zero sky ambient (exercises finding #2)
        // Sanity: the sky term is genuinely non-zero at this hour, so a shader
        // that seeds incoming=0 must diverge.
        let sky = GpuGi::sky_ambient_for_hour(hour);
        assert!(
            sky.iter().any(|&v| v > 1e-3),
            "noon sky-ambient must be non-trivial, got {sky:?}"
        );

        let gpu_out = gpu.step(&scene, hour).expect("gpu step");
        let cpu_out = cpu_step_reference(&scene, hour);
        assert_eq!(gpu_out.len(), cpu_out.len());

        // f16 has ~3-4 significant decimal digits; values are in [0,1]. A 2e-3
        // epsilon is tight yet quantization-safe near the top of the range.
        let eps = 2e-3f32;

        // Receivers to check: the probe (whose dominant emitter the old loop
        // dropped), its neighbours, an emitter, and a spread of plain receivers.
        let sample: Vec<usize> = {
            let mut s = vec![
                0,
                7,
                probe_receiver_idx,
                probe_receiver_idx + 1,
                dropped_emitter_idx,
                500,
                999,
                1100,
                n - 1,
            ];
            s.dedup();
            s
        };

        let mut max_delta = 0.0f32;
        for &i in &sample {
            for b in 0..16 {
                let g = f16::from_bits(gpu_out[i].spectral()[b]).to_f32();
                let c = f16::from_bits(cpu_out[i].spectral()[b]).to_f32();
                let d = (g - c).abs();
                if d > max_delta {
                    max_delta = d;
                }
                assert!(
                    d <= eps,
                    "CPU/GPU divergence at splat {i} band {b}: gpu={g} cpu={c} (|Δ|={d} > {eps})"
                );
            }
        }

        // Prove the probe receiver was actually lit by the would-be-dropped
        // emitter (not a vacuous pass): its emit-band must be clearly above the
        // dark floor on BOTH paths.
        let probe_g = f16::from_bits(gpu_out[probe_receiver_idx].spectral()[emit_band]).to_f32();
        let probe_c = f16::from_bits(cpu_out[probe_receiver_idx].spectral()[emit_band]).to_f32();
        assert!(
            probe_g > 0.05 && probe_c > 0.05,
            "probe receiver must be strongly lit by its adjacent emitter: gpu={probe_g} cpu={probe_c}"
        );

        eprintln!(
            "[gpu_gi equivalence] n={n} sample={} max|Δ|={max_delta:.2e} probe gpu={probe_g:.4} cpu={probe_c:.4}",
            sample.len()
        );
    }

    #[test]
    fn gpu_gi_falls_back_on_impossible_limits() {
        // Force device creation to fail with impossible limits → Err, never panic.
        let bad = wgpu::Limits {
            max_storage_buffers_per_shader_stage: u32::MAX,
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u32::MAX,
            max_compute_workgroups_per_dimension: u32::MAX,
            ..wgpu::Limits::default()
        };
        let res = GpuGi::new_with_limits(64, bad);
        match res {
            Err(GpuGiError::DeviceCreation(_)) | Err(GpuGiError::NoAdapter) => {}
            Err(other) => panic!("expected device-creation/no-adapter error, got {other}"),
            Ok(_) => panic!("impossible limits must not yield a working device"),
        }
    }

    #[test]
    #[ignore = "perf bench — run explicitly with --ignored --nocapture"]
    fn gpu_gi_50k_timing() {
        let Some(gpu) = try_gpu(60_000) else {
            eprintln!("no GPU — cannot bench");
            return;
        };
        // 50k splats: a grid of receivers with a few bright emitters seeded in.
        let mut scene = Vec::with_capacity(50_000);
        for i in 0..50_000u32 {
            let x = (i % 100) as f32 * 0.1;
            let y = ((i / 100) % 100) as f32 * 0.1;
            let z = (i / 10_000) as f32 * 0.1;
            let emitter = i % 500 == 0;
            scene.push(GaussianSplat::volume(
                [x, y, z],
                [0.05, 0.05, 0.05],
                glam::Quat::IDENTITY,
                if emitter { 255 } else { 10 },
                [f16::from_f32(if emitter { 1.0 } else { 0.0 }).to_bits(); 16],
            ));
        }
        // Warm up (shader compile, allocation).
        let _ = gpu.step(&scene, 12.0).expect("warmup");
        let runs = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..runs {
            let _ = gpu.step(&scene, 12.0).expect("bench step");
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        eprintln!(
            "GPU GI 50k splats on {}: {:.2} ms/step (avg of {runs})",
            gpu.adapter_name, ms
        );
    }

    #[test]
    fn spectral_gi_bake_produces_nonzero_radiance() {
        // Emitter at origin with band 3 = 1.0 (visible green), receiver at 0.5m
        let emitter = SplatGiEntry {
            position: [0.0, 0.0, 0.0],
            emissive: { let mut e = [0.0f32; 16]; e[3] = 1.0; e },
            reflectance: [0.5; 16],
        };
        let probe_pos = [0.5, 0.0, 0.0];
        let radiance = gather_radiance(probe_pos, &[emitter], 2.0);
        let probe = GiProbe { bands: radiance };
        // expected: 1.0 * 0.5 / (0.5^2) = 2.0
        assert!(
            probe.bands[3] > 0.001,
            "GI bake must produce non-zero indirect radiance in band 3, got {}",
            probe.bands[3]
        );
    }
}
