use std::sync::Arc;
use winit::window::Window;

/// A minimal wgpu backend that blits a software framebuffer to the window surface.
pub struct WgpuBackend {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
}

impl WgpuBackend {
    /// Create a new backend for the given window.
    ///
    /// Tries Vulkan first, then GL, then all available backends.
    /// Returns `Err(String)` if no suitable adapter or device can be created.
    pub fn new(window: Arc<Window>, width: u32, height: u32) -> Result<Self, String> {
        pollster::block_on(Self::new_async(window, width, height))
    }

    async fn new_async(window: Arc<Window>, width: u32, height: u32) -> Result<Self, String> {
        // Try backends in order: Vulkan → GL → all available
        let backend_attempts: &[(&str, wgpu::Backends)] = &[
            ("Vulkan", wgpu::Backends::VULKAN),
            ("GL", wgpu::Backends::GL),
            ("all", wgpu::Backends::all()),
        ];

        let mut last_error = String::from("no backends available");

        for (name, backends) in backend_attempts {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: *backends,
                ..Default::default()
            });

            let surface = match instance.create_surface(Arc::clone(&window)) {
                Ok(s) => s,
                Err(e) => {
                    last_error = format!("{name} backend: surface creation failed: {e}");
                    eprintln!("[wgpu] {last_error}");
                    continue;
                }
            };

            let adapter = match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
            {
                Some(a) => a,
                None => {
                    last_error = format!("{name} backend: no suitable adapter found");
                    eprintln!("[wgpu] {last_error}");
                    continue;
                }
            };

            let (device, queue) = match adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("ochroma-device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await
            {
                Ok(dq) => dq,
                Err(e) => {
                    last_error = format!("{name} backend: device creation failed: {e}");
                    eprintln!("[wgpu] {last_error}");
                    continue;
                }
            };

            let surface_caps = surface.get_capabilities(&adapter);
            if surface_caps.formats.is_empty() {
                last_error = format!("{name} backend: no surface formats available");
                eprintln!("[wgpu] {last_error}");
                continue;
            }

            // Prefer Bgra8UnormSrgb or Rgba8UnormSrgb; fall back to first available format.
            let format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|f| {
                    *f == wgpu::TextureFormat::Bgra8UnormSrgb
                        || *f == wgpu::TextureFormat::Rgba8UnormSrgb
                })
                .unwrap_or(surface_caps.formats[0]);

            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
            };
            surface.configure(&device, &config);

            eprintln!("[wgpu] Successfully initialised with {name} backend");
            return Ok(Self {
                surface,
                device,
                queue,
                config,
                width,
                height,
            });
        }

        Err(last_error)
    }

    /// Write raw RGBA8 pixel data into the current surface texture and present it.
    ///
    /// `pixels` must contain exactly `width * height` elements.
    pub fn present_framebuffer(&self, pixels: &[[u8; 4]], width: u32, height: u32) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("wgpu: get_current_texture failed: {e}");
                return;
            }
        };

        // Flatten [[u8;4]] to &[u8]
        let raw: &[u8] = bytemuck::cast_slice(pixels);

        // wgpu requires bytes_per_row to be a multiple of 256.
        // Each pixel is 4 bytes, so bytes_per_row = width * 4, padded up.
        let unpadded_bytes_per_row = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.div_ceil(align) * align;

        if padded_bytes_per_row == unpadded_bytes_per_row {
            // No padding needed — write directly.
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(unpadded_bytes_per_row),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            // Build a padded staging buffer row-by-row.
            let mut padded: Vec<u8> =
                vec![0u8; (padded_bytes_per_row * height) as usize];
            for row in 0..height as usize {
                let src_start = row * unpadded_bytes_per_row as usize;
                let dst_start = row * padded_bytes_per_row as usize;
                padded[dst_start..dst_start + unpadded_bytes_per_row as usize]
                    .copy_from_slice(&raw[src_start..src_start + unpadded_bytes_per_row as usize]);
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.submit(std::iter::empty());
        output.present();
    }

    /// Reconfigure the surface after a window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        // Query the device's max texture dimension, fall back to a safe limit
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let w = width.min(max_dim);
        let h = height.min(max_dim);
        self.width = w;
        self.height = h;
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns a reference to the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns a reference to the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Returns the surface texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Returns a reference to the wgpu surface.
    pub fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }
}
