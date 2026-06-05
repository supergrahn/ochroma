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
            for b in 0..16 {
                self.cache[i][b] = alpha * self.cache[i][b] + (1.0 - alpha) * sky[b];
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
                for b in 0..16 {
                    incoming[b] += emitter.emissive[b] * weight;
                }
            }
            let max_incoming = incoming.iter().copied().fold(f32::EPSILON, f32::max);
            let scale = if max_incoming > 1.0 { 1.0 / max_incoming } else { 1.0 };
            for b in 0..16 {
                self.cache[i][b] = alpha * self.cache[i][b]
                    + (1.0 - alpha) * (incoming[b] * scale).clamp(0.0, 1.0);
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiParamsUniform {
    pub splat_count: u32,
    pub max_emitters: u32,
    pub alpha: f32,
    pub _pad: f32,
}

pub struct GpuGiPass {
    pub splat_buffer: wgpu::Buffer,
    pub radiance_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
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
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            pipeline,
            bind_group,
            max_splats,
        }
    }

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
        let params = GiParamsUniform {
            splat_count: count,
            max_emitters: 256,
            alpha,
            _pad: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gi_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
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
        assert_eq!(std::mem::size_of::<GiParamsUniform>(), 16);
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
