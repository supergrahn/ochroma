use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use vox_core::spectral::Illuminant;

use super::GpuContext;
use crate::spectral::RenderCamera;

const CIE_X: [f32; 16] = [
    0.01741, 0.08028, 0.26000, 0.21000, 0.00949, 0.00000, 0.11201, 0.38000, 0.74300, 1.02200,
    0.71600, 0.38100, 0.19700, 0.09020, 0.03400, 0.01180,
];
const CIE_Y: [f32; 16] = [
    0.00039, 0.00232, 0.01998, 0.09520, 0.17399, 0.46600, 0.69500, 0.94500, 0.86800, 0.65100,
    0.38100, 0.18000, 0.08000, 0.03300, 0.01200, 0.00400,
];
const CIE_Z: [f32; 16] = [
    0.08290, 0.38637, 1.29900, 1.24500, 0.45640, 0.05250, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
];
const BAND_SPACING: f32 = 25.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresentationLight {
    /// Unit direction toward the active light.
    pub dir: [f32; 3],
    /// Linear RGB light color.
    pub color: [f32; 3],
    /// 0..1 active light intensity after weather attenuation.
    pub intensity: f32,
    pub is_day: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresentationSky {
    pub light: PresentationLight,
    pub attenuation: f32,
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub ground: [f32; 3],
    pub fog: f32,
}

impl PresentationSky {
    fn lit(self) -> f32 {
        0.28 + 0.72 * self.light.intensity
    }

    fn light_core(self) -> [f32; 3] {
        if self.light.is_day {
            [255.0, 244.0, 214.0]
        } else {
            [200.0, 210.0, 235.0]
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PresentParams {
    inv_view_proj: [[f32; 4]; 4],
    light_dir_atten: [f32; 4],
    zenith: [f32; 4],
    horizon: [f32; 4],
    ground: [f32; 4],
    light_color_lit: [f32; 4],
    light_core: [f32; 4],
    dims_fog: [f32; 4],
    rgb_coeffs: [[f32; 4]; 8],
}

pub struct SpectralPresenter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    output_texture: wgpu::Texture,
    readback: wgpu::Buffer,
    coverage: wgpu::Buffer,
    coverage_readback: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl SpectralPresenter {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let device = ctx.device().clone();
        let queue = ctx.queue().clone();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spectral_present_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("spectral_present.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spectral_present_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
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
                storage_texture_entry(2),
                storage_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spectral_present_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spectral_present_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("present"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spectral_present_params"),
            size: std::mem::size_of::<PresentParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_texture = output_texture(&device, width, height);
        let readback = pixel_readback_buffer(&device, width, height);
        let coverage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spectral_present_coverage"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let coverage_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spectral_present_coverage_readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            queue,
            pipeline,
            layout,
            params,
            output_texture,
            readback,
            coverage,
            coverage_readback,
            width,
            height,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.output_texture = output_texture(&self.device, width, height);
        self.readback = pixel_readback_buffer(&self.device, width, height);
        self.width = width;
        self.height = height;
    }

    pub fn present(
        &mut self,
        spectral_texture: &wgpu::Texture,
        camera: &RenderCamera,
        sky: PresentationSky,
    ) -> Result<(Vec<[u8; 4]>, f32), String> {
        self.encode_present(spectral_texture, camera, sky, true)?;
        let pixels = read_pixels(&self.device, &self.readback, self.width, self.height)?;
        let coverage = self.read_coverage()?;
        Ok((pixels, coverage))
    }

    pub fn present_texture(
        &mut self,
        spectral_texture: &wgpu::Texture,
        camera: &RenderCamera,
        sky: PresentationSky,
    ) -> Result<(wgpu::Texture, f32), String> {
        self.encode_present(spectral_texture, camera, sky, false)?;
        let coverage = self.read_coverage()?;
        Ok((self.output_texture.clone(), coverage))
    }

    fn encode_present(
        &mut self,
        spectral_texture: &wgpu::Texture,
        camera: &RenderCamera,
        sky: PresentationSky,
        copy_pixels: bool,
    ) -> Result<(), String> {
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&present_params(camera, sky, self.width, self.height)),
        );
        self.queue.write_buffer(&self.coverage, 0, &[0, 0, 0, 0]);

        let spectral_view = spectral_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("spectral_present_spectral_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let output_view = self
            .output_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("spectral_present_output_view"),
                ..Default::default()
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spectral_present_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&spectral_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.coverage.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spectral_present_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spectral_present_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(16), self.height.div_ceil(16), 1);
        }
        if copy_pixels {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row(self.width)),
                        rows_per_image: Some(self.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        encoder.copy_buffer_to_buffer(&self.coverage, 0, &self.coverage_readback, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn read_coverage(&self) -> Result<f32, String> {
        let covered = read_u32(&self.device, &self.coverage_readback)?;
        let total = (self.width as u64 * self.height as u64).max(1) as f32;
        Ok(covered as f32 / total)
    }
}

fn storage_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn output_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spectral_present_output_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn pixel_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spectral_present_readback"),
        size: readback_bytes(width, height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn padded_bytes_per_row(width: u32) -> u32 {
    let row = width * 4;
    row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

fn readback_bytes(width: u32, height: u32) -> u64 {
    padded_bytes_per_row(width) as u64 * height as u64
}

fn present_params(
    camera: &RenderCamera,
    sky: PresentationSky,
    width: u32,
    height: u32,
) -> PresentParams {
    PresentParams {
        inv_view_proj: (camera.proj * camera.view).inverse().to_cols_array_2d(),
        light_dir_atten: [
            sky.light.dir[0],
            sky.light.dir[1],
            sky.light.dir[2],
            sky.attenuation,
        ],
        zenith: vec4_color(sky.zenith),
        horizon: vec4_color(sky.horizon),
        ground: vec4_color(sky.ground),
        light_color_lit: [
            sky.light.color[0],
            sky.light.color[1],
            sky.light.color[2],
            sky.lit(),
        ],
        light_core: vec4_color(sky.light_core()),
        dims_fog: [width as f32, height as f32, sky.fog, 0.0],
        rgb_coeffs: spectral_rgb_coeffs(),
    }
}

fn vec4_color(rgb: [f32; 3]) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], 0.0]
}

fn spectral_rgb_coeffs() -> [[f32; 4]; 8] {
    let illuminant = Illuminant::d65();
    let mut norm_x = 0.0;
    let mut norm_y = 0.0;
    let mut norm_z = 0.0;
    for i in 0..16 {
        norm_x += illuminant.bands[i] * CIE_X[i] * BAND_SPACING;
        norm_y += illuminant.bands[i] * CIE_Y[i] * BAND_SPACING;
        norm_z += illuminant.bands[i] * CIE_Z[i] * BAND_SPACING;
    }
    std::array::from_fn(|bin| {
        let mut xyz = [0.0; 3];
        for i in [bin * 2, bin * 2 + 1] {
            let power = illuminant.bands[i] * BAND_SPACING;
            xyz[0] += power * CIE_X[i] / norm_x * 0.9505;
            xyz[1] += power * CIE_Y[i] / norm_y;
            xyz[2] += power * CIE_Z[i] / norm_z * 1.0888;
        }
        let [x, y, z] = xyz;
        [
            3.2406 * x - 1.5372 * y - 0.4986 * z,
            -0.9689 * x + 1.8758 * y + 0.0415 * z,
            0.0557 * x - 0.2040 * y + 1.0570 * z,
            0.0,
        ]
    })
}

fn read_pixels(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> Result<Vec<[u8; 4]>, String> {
    let bytes = readback_bytes(width, height);
    let slice = readback.slice(0..bytes);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    if !matches!(rx.recv(), Ok(Ok(()))) {
        return Err("spectral present readback map failed".to_string());
    }
    let out = {
        let data = slice.get_mapped_range();
        let row_bytes = (width * 4) as usize;
        let padded = padded_bytes_per_row(width) as usize;
        let mut out = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height as usize {
            let row = &data[y * padded..y * padded + row_bytes];
            out.extend(row.chunks_exact(4).map(|p| [p[0], p[1], p[2], p[3]]));
        }
        out
    };
    readback.unmap();
    Ok(out)
}

fn read_u32(device: &wgpu::Device, readback: &wgpu::Buffer) -> Result<u32, String> {
    let slice = readback.slice(0..4);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    if !matches!(rx.recv(), Ok(Ok(()))) {
        return Err("spectral present coverage map failed".to_string());
    }
    let value = {
        let data = slice.get_mapped_range();
        u32::from_ne_bytes([data[0], data[1], data[2], data[3]])
    };
    readback.unmap();
    Ok(value)
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use super::*;

    fn presentation_sky() -> PresentationSky {
        PresentationSky {
            light: PresentationLight {
                dir: [0.0, 1.0, 0.0],
                color: [1.0, 0.95, 0.9],
                intensity: 0.8,
                is_day: true,
            },
            attenuation: 0.85,
            zenith: [70.0, 125.0, 210.0],
            horizon: [200.0, 212.0, 226.0],
            ground: [92.0, 138.0, 86.0],
            fog: 0.12,
        }
    }

    #[test]
    fn readback_rows_are_wgpu_aligned() {
        assert_eq!(
            padded_bytes_per_row(1),
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
        );
        assert_eq!(readback_bytes(3, 2), 512);
    }

    /// Real-hardware skip gate (matches the other `gpu::` GPU tests): build a
    /// `GpuContext` on the local HARDWARE adapter, skipping cleanly on no-adapter
    /// or a software rasteriser so headless CI without a GPU stays green.
    fn try_gpu_context(label: &str) -> Option<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                eprintln!("[{label}] no adapter — skipping GPU test");
                return None;
            }
        };
        let info = adapter.get_info();
        if crate::gpu::adapter::ensure_hardware(&info).is_err() {
            eprintln!("[{label}] software adapter ({}) — skipping GPU test", info.name);
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("spectral_present_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("device creation on a box with a GPU");
        Some(GpuContext::from_parts(&device, &queue, &info))
    }

    /// Build the 4-layer `Rgba32Float` spectral texture the presenter consumes
    /// (the exact format/layer-count `TiledSplatRenderer`'s `spectral_texture`
    /// uses). `bins` is the 8 spectral-bin values written uniformly into the
    /// rectangle `[x0,x1) × [y0,y1)`; the rest stays zero. Layer 0 carries bins
    /// 0..4 (p0.xyzw), layer 1 carries bins 4..8 (p1.xyzw) — matching
    /// `unpack_geometry_rgb` in `spectral_present.wgsl`.
    fn spectral_texture_with_rect(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rect: (u32, u32, u32, u32),
        bins: [f32; 8],
    ) -> wgpu::Texture {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spectral_present_test_input"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (x0, y0, x1, y1) = rect;
        // Two written layers (0 = bins 0..4, 1 = bins 4..8). Each texel is RGBA32F.
        for (layer, quad) in [[bins[0], bins[1], bins[2], bins[3]], [bins[4], bins[5], bins[6], bins[7]]]
            .into_iter()
            .enumerate()
        {
            let mut data = vec![0.0f32; (width * height * 4) as usize];
            for y in y0..y1 {
                for x in x0..x1 {
                    let base = ((y * width + x) * 4) as usize;
                    data[base..base + 4].copy_from_slice(&quad);
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 16),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        tex
    }

    /// ENGINE-SIDE GPU SMOKE TEST for `SpectralPresenter` (the extracted presenter
    /// the game wraps). Without this, the presenter's correctness was proven only
    /// TRANSITIVELY by the Urban Horizon suite — an engine-only CI run would pass while
    /// it was broken. This drives the REAL `present()` path on the 780M and asserts
    /// two computed outcomes, not `is_ok()`:
    ///   (1) a KNOWN SKY scene (zero spectral) → coverage == 0 (nothing passes the
    ///       geometry threshold) AND a top-of-frame pixel lands in the BLUE band
    ///       (B > R and B > G), because the zenith colour is [70,125,210];
    ///   (2) a KNOWN SPLAT scene (a bright neutral rectangle) → coverage > 0 AND a
    ///       pixel inside the rectangle is NON-BLACK and brighter than the same
    ///       pixel in the sky-only render (the splat path actually lit it).
    #[test]
    fn presenter_renders_known_sky_and_splat_scenes() {
        let Some(ctx) = try_gpu_context("spectral_present") else {
            return;
        };
        let device = ctx.device().clone();
        let queue = ctx.queue().clone();
        const W: u32 = 64;
        const H: u32 = 64;

        // Camera looking up and slightly forward so the upper rows sample the
        // zenith-facing sky (dir.y > 0 branch in backdrop_color).
        let camera = RenderCamera {
            view: Mat4::look_at_rh(
                glam::Vec3::new(0.0, 1.0, 0.0),
                glam::Vec3::new(0.0, 4.0, -1.0),
                glam::Vec3::Y,
            ),
            proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, W as f32 / H as f32, 0.1, 100.0),
        };
        let sky = presentation_sky();

        let mut presenter = SpectralPresenter::new(&ctx, W, H);

        // ── (1) SKY: all-zero spectral → no coverage, blue backdrop. ──────────
        let sky_tex =
            spectral_texture_with_rect(&device, &queue, W, H, (0, 0, 0, 0), [0.0; 8]);
        let (sky_px, sky_cov) = presenter
            .present(&sky_tex, &camera, sky)
            .expect("sky present on a box with a GPU");
        assert_eq!(
            sky_cov, 0.0,
            "zero-spectral scene must yield ZERO geometry coverage, got {sky_cov}"
        );
        // Top row, centre column: should be the zenith-facing sky (blue-dominant).
        let top = sky_px[(W / 2) as usize];
        assert!(
            top[2] > top[0] && top[2] > top[1] && top[2] > 60,
            "sky backdrop must be blue-dominant at the zenith (B>R, B>G, B bright), got {top:?}"
        );

        // ── (2) SPLAT: bright neutral rectangle → coverage > 0, lit pixels. ───
        let (rx0, ry0, rx1, ry1) = (W / 4, H / 4, 3 * W / 4, 3 * H / 4);
        let splat_tex = spectral_texture_with_rect(
            &device,
            &queue,
            W,
            H,
            (rx0, ry0, rx1, ry1),
            [0.2; 8], // ~RGB 51/51/51 geometry, sum 153 > the 18.0 coverage threshold
        );
        let (splat_px, splat_cov) = presenter
            .present(&splat_tex, &camera, sky)
            .expect("splat present on a box with a GPU");
        let expected_cov =
            ((rx1 - rx0) * (ry1 - ry0)) as f32 / (W * H) as f32;
        assert!(
            (splat_cov - expected_cov).abs() < 0.02,
            "rectangle coverage must match the lit area ~{expected_cov:.3}, got {splat_cov:.3}"
        );
        // A pixel inside the rectangle must be lit (non-black) AND land in the
        // expected NEUTRAL-GREY band: geometry RGB ~[51,51,51] (bin 0.2 over the
        // D65 reconstruction) scaled by `light_color_lit` (lit ≈ 0.856 × the
        // near-white [1,0.95,0.9] light) → roughly [44,41,39]. This is the splat
        // branch (`rgb = geom * light`), structurally distinct from the blue
        // backdrop the SAME pixel shows in the sky render.
        let cx = (ry0 + ry1) / 2 * W + (rx0 + rx1) / 2;
        let lit = splat_px[cx as usize];
        let bg = sky_px[cx as usize];
        let lit_sum = lit[0] as i32 + lit[1] as i32 + lit[2] as i32;
        assert!(
            lit_sum > 30,
            "lit splat pixel must be non-black, got {lit:?} (sum {lit_sum})"
        );
        // Neutral grey, slightly warm from the daylight tint: R >= G >= B, all in
        // a mid band. (A blue backdrop bleeding through would give B-dominant.)
        assert!(
            lit[0] >= lit[1] && lit[1] >= lit[2] && (20..=90).contains(&lit[0]),
            "splat pixel must be the warm neutral-grey lit band (R>=G>=B, mid), got {lit:?}"
        );
        // The covered splat pixel must NOT equal the backdrop the same coordinate
        // shows under the sky render — proving the geometry branch overrode it.
        assert!(
            lit != bg,
            "covered splat pixel {lit:?} must differ from the sky backdrop {bg:?} \
             at the same coordinate (the splat branch must override the backdrop)"
        );
        eprintln!(
            "[spectral_present] sky_cov={sky_cov} top={top:?} splat_cov={splat_cov:.3} \
             (expected {expected_cov:.3}) lit={lit:?} bg={bg:?}"
        );
    }

    #[test]
    fn present_params_carry_camera_environment_and_dimensions() {
        let camera = RenderCamera {
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
        };

        let params = present_params(&camera, presentation_sky(), 3840, 2160);

        assert_eq!(params.dims_fog, [3840.0, 2160.0, 0.12, 0.0]);
        assert_eq!(params.light_dir_atten, [0.0, 1.0, 0.0, 0.85]);
        assert!((params.light_color_lit[3] - 0.856).abs() < 1e-5);
        assert!(params.rgb_coeffs.iter().flatten().all(|v| v.is_finite()));
    }
}
