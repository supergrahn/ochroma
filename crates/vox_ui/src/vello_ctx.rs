//! VelloCtx — wgpu-backed Vello scene renderer for game UI.
//!
//! VelloCtx owns a vello::Renderer and accumulates draw commands each frame.
//! VelloCtxCpu is a headless equivalent for unit tests (no GPU required).

// --- Draw command enum (shared by GPU and CPU paths) ----------------------

#[derive(Debug, Clone)]
pub enum DrawCmd {
    FillRect { rect: [f32; 4], color: [f32; 4] },
}

// --- Local-GPU policy (inlined; vox_ui has no vox_render dep) --------------

/// Is this adapter the CPU software rasteriser (llvmpipe / SwiftShader) rather
/// than a real GPU? Mirrors `vox_render::gpu::adapter::is_software`. "Use local
/// GPU": the UI canvas must run on real hardware, not emulate on the CPU.
#[cfg(feature = "game-ui")]
fn is_software_adapter(info: &vello::wgpu::AdapterInfo) -> bool {
    if info.device_type == vello::wgpu::DeviceType::Cpu {
        return true;
    }
    let name = info.name.to_ascii_lowercase();
    name.contains("llvmpipe")
        || name.contains("swiftshader")
        || name.contains("softpipe")
        || name.contains("software")
}

/// Is the software fallback explicitly permitted via `OCHROMA_ALLOW_SOFTWARE_GPU`?
#[cfg(feature = "game-ui")]
fn software_gpu_allowed() -> bool {
    matches!(
        std::env::var("OCHROMA_ALLOW_SOFTWARE_GPU")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on")
    )
}

// --- CPU test stub --------------------------------------------------------

/// Headless VelloCtx for unit tests — accumulates DrawCmd without a GPU.
pub struct VelloCtxCpu {
    commands: Vec<DrawCmd>,
    width:    u32,
    height:   u32,
}

impl VelloCtxCpu {
    pub fn new(width: u32, height: u32) -> Self {
        Self { commands: Vec::new(), width, height }
    }

    pub fn begin_frame(&mut self) {
        self.commands.clear();
    }

    pub fn fill_rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        self.commands.push(DrawCmd::FillRect { rect, color });
    }

    pub fn commands(&self) -> &[DrawCmd] {
        &self.commands
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width  = width;
        self.height = height;
    }

    pub fn width(&self)  -> u32 { self.width  }
    pub fn height(&self) -> u32 { self.height }

    /// Rasterize all recorded draw commands, in order, into an RGBA8 pixel
    /// buffer (`pixels.len()` must be at least `width * height`, row-major).
    ///
    /// Each `FillRect` alpha-blends its straight-alpha RGBA `color` (channels
    /// in `0..=1`) over the destination using the standard "over" operator
    /// `out = src*a + dst*(1-a)` per channel, clipped to the buffer bounds.
    /// Rects partially or fully off-screen are clipped; zero-size rects and
    /// alpha-0 colors are no-ops; alpha-1 colors overwrite the destination.
    pub fn rasterize_into(&self, pixels: &mut [[u8; 4]], width: u32, height: u32) {
        let w = width as i64;
        let h = height as i64;
        if w <= 0 || h <= 0 {
            return;
        }
        let needed = (w * h) as usize;
        let limit = needed.min(pixels.len());
        let pixels = &mut pixels[..limit];
        // If the buffer is smaller than width*height we still clamp safely
        // below by checking the linear index.

        for cmd in &self.commands {
            match cmd {
                DrawCmd::FillRect { rect, color } => {
                    let a = color[3].clamp(0.0, 1.0);
                    if a <= 0.0 {
                        continue; // fully transparent: no-op
                    }

                    // Rect is [x, y, w, h] in pixel space. Compute the integer
                    // pixel span [x0, x1) x [y0, y1), clipped to the buffer.
                    let rx = rect[0];
                    let ry = rect[1];
                    let rw = rect[2];
                    let rh = rect[3];
                    // NaN must be treated as a skip: `!(rw > 0.0)` is true for NaN,
                    // whereas `rw <= 0.0` is false for NaN — that difference matters here.
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if !(rw > 0.0) || !(rh > 0.0) {
                        continue; // zero-size (or NaN) rect: no-op
                    }

                    let x0 = rx.floor() as i64;
                    let y0 = ry.floor() as i64;
                    let x1 = (rx + rw).ceil() as i64;
                    let y1 = (ry + rh).ceil() as i64;

                    let x0 = x0.clamp(0, w);
                    let y0 = y0.clamp(0, h);
                    let x1 = x1.clamp(0, w);
                    let y1 = y1.clamp(0, h);
                    if x0 >= x1 || y0 >= y1 {
                        continue; // fully off-screen
                    }

                    let sr = color[0].clamp(0.0, 1.0);
                    let sg = color[1].clamp(0.0, 1.0);
                    let sb = color[2].clamp(0.0, 1.0);
                    let inv = 1.0 - a;

                    for y in y0..y1 {
                        for x in x0..x1 {
                            let idx = (y * w + x) as usize;
                            if idx >= pixels.len() {
                                continue;
                            }
                            let dst = pixels[idx];
                            let dr = dst[0] as f32 / 255.0;
                            let dg = dst[1] as f32 / 255.0;
                            let db = dst[2] as f32 / 255.0;
                            let da = dst[3] as f32 / 255.0;

                            let or = sr * a + dr * inv;
                            let og = sg * a + dg * inv;
                            let ob = sb * a + db * inv;
                            let oa = a + da * inv;

                            pixels[idx] = [
                                (or * 255.0 + 0.5) as u8,
                                (og * 255.0 + 0.5) as u8,
                                (ob * 255.0 + 0.5) as u8,
                                (oa * 255.0 + 0.5) as u8,
                            ];
                        }
                    }
                }
            }
        }
    }
}

// --- GPU VelloCtx (feature-gated) ----------------------------------------

/// A real GPU-backed Vello renderer.
///
/// Unlike [`VelloCtxCpu`] (which records `DrawCmd`s and software-rasterises),
/// `VelloCtx` drives an actual `vello::Renderer` over a `wgpu` device, the same
/// GPU vector path the windowed editor presents through. The same `fill_rect`
/// API records into a `vello::Scene`; the scene is flushed either to a caller's
/// surface texture ([`end_frame`](Self::end_frame)) or, for headless tests and
/// CLI verification, to an offscreen `Rgba8Unorm` texture that is read back to
/// CPU pixels by [`render_to_rgba`](Self::render_to_rgba).
#[cfg(feature = "game-ui")]
pub struct VelloCtx {
    renderer: vello::Renderer,
    scene:    vello::Scene,
    width:    u32,
    height:   u32,
    /// Device/queue owned only by the headless constructor. When the caller
    /// supplies their own device/queue (windowed path via [`new`](Self::new)),
    /// this is `None` and the caller passes device/queue to `end_frame`.
    owned: Option<(vello::wgpu::Device, vello::wgpu::Queue)>,
}

#[cfg(feature = "game-ui")]
impl VelloCtx {
    pub fn new(
        device: &vello::wgpu::Device,
        _queue: &vello::wgpu::Queue,
        width:  u32,
        height: u32,
        // vello 0.5 dropped `RendererOptions::surface_format`; the renderer is no
        // longer told the surface format up front. The param is retained so the
        // windowed-path call site is unchanged, but it is now advisory only.
        _surface_format: vello::wgpu::TextureFormat,
    ) -> Result<Self, vello::Error> {
        let renderer = vello::Renderer::new(
            device,
            vello::RendererOptions {
                use_cpu:        false,
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: std::num::NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )?;
        Ok(Self { renderer, scene: vello::Scene::new(), width, height, owned: None })
    }

    /// Build a fully self-contained headless `VelloCtx`: it requests its own
    /// `wgpu` instance/adapter/device/queue (no window, no surface) and a Vello
    /// renderer configured for offscreen `render_to_texture`. Returns `None` if
    /// no GPU adapter is available (e.g. CI with no Vulkan/GL) — callers should
    /// treat that as "skip GPU path", not a failure.
    pub fn new_headless(width: u32, height: u32) -> Option<Self> {
        use vello::wgpu;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))?;
        // Use local GPU: refuse the llvmpipe CPU software rasteriser for the UI
        // canvas (override: OCHROMA_ALLOW_SOFTWARE_GPU=1). vox_ui does not depend
        // on vox_render, so the policy is inlined here against vello's wgpu.
        if is_software_adapter(&adapter.get_info()) && !software_gpu_allowed() {
            eprintln!(
                "[vello] refusing software adapter '{}' — set OCHROMA_ALLOW_SOFTWARE_GPU=1 to override",
                adapter.get_info().name
            );
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vello-headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;

        let renderer = vello::Renderer::new(
            &device,
            vello::RendererOptions {
                // No surface — we only ever render_to_texture offscreen.
                // (vello 0.5 removed the `surface_format` option entirely.)
                use_cpu:        false,
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: std::num::NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .ok()?;

        Some(Self {
            renderer,
            scene: vello::Scene::new(),
            width,
            height,
            owned: Some((device, queue)),
        })
    }

    pub fn begin_frame(&mut self) {
        self.scene = vello::Scene::new();
    }

    pub fn fill_rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        use vello::kurbo::{Affine, Rect};
        use vello::peniko::{Brush, Color, Fill};
        // peniko::Color is AlphaColor<Srgb>; construct via AlphaColor::new([r, g, b, a])
        let vello_color = Color::new([color[0], color[1], color[2], color[3]]);
        let vello_rect = Rect::new(
            rect[0] as f64, rect[1] as f64,
            (rect[0] + rect[2]) as f64, (rect[1] + rect[3]) as f64,
        );
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            &Brush::Solid(vello_color),
            None,
            &vello_rect,
        );
    }

    pub fn end_frame(
        &mut self,
        device: &vello::wgpu::Device,
        queue:  &vello::wgpu::Queue,
        surface_view: &vello::wgpu::TextureView,
    ) -> Result<(), vello::Error> {
        self.renderer.render_to_texture(
            device,
            queue,
            &self.scene,
            surface_view,
            &vello::RenderParams {
                base_color:          vello::peniko::color::palette::css::BLACK,
                width:               self.width,
                height:              self.height,
                antialiasing_method: vello::AaConfig::Area,
            },
        )
    }

    /// Render the currently-recorded scene to an offscreen `Rgba8Unorm` texture
    /// and read it back to a row-major `Vec<[u8; 4]>` (length `width*height`).
    ///
    /// Only available on a headless context (built via [`new_headless`]). Uses
    /// the owned device/queue. This is the path the pixel-level tests and the
    /// `--vello-hud-selftest` CLI flag assert against: it proves the real Vello
    /// GPU pipeline executed and produced the expected pixels, not a CPU stub.
    pub fn render_to_rgba(&mut self) -> Result<Vec<[u8; 4]>, String> {
        use vello::wgpu;

        let (device, queue) = self
            .owned
            .as_ref()
            .ok_or_else(|| "render_to_rgba requires a headless VelloCtx (use new_headless)".to_string())?;

        let w = self.width;
        let h = self.height;

        // Vello's render_to_texture requires an Rgba8Unorm STORAGE_BINDING target.
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello-headless-target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer
            .render_to_texture(
                device,
                queue,
                &self.scene,
                &view,
                &vello::RenderParams {
                    // TRANSPARENT base: unrendered pixels read back with
                    // alpha 0, so compositors key on the real alpha channel
                    // instead of heuristically color-keying near-black (which
                    // dropped AA edge coverage and dark content).
                    base_color:          vello::peniko::color::palette::css::TRANSPARENT,
                    width:               w,
                    height:              h,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .map_err(|e| format!("vello render_to_texture failed: {e:?}"))?;

        // Copy texture -> buffer. Rows must be padded to COPY_BYTES_PER_ROW_ALIGNMENT.
        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let buffer_size = (padded_bpr * h) as wgpu::BufferAddress;

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vello-headless-readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vello-readback") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Map the readback buffer and block until ready.
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "map_async sender dropped".to_string())?
            .map_err(|e| format!("buffer map failed: {e:?}"))?;

        let data = slice.get_mapped_range();
        let mut pixels = vec![[0u8; 4]; (w * h) as usize];
        for y in 0..h {
            let row_off = (y * padded_bpr) as usize;
            for x in 0..w {
                let px = row_off + (x * 4) as usize;
                pixels[(y * w + x) as usize] = [data[px], data[px + 1], data[px + 2], data[px + 3]];
            }
        }
        drop(data);
        readback.unmap();

        Ok(pixels)
    }

    pub fn width(&self)  -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width  = width;
        self.height = height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vello_ctx_cpu_starts_empty() {
        let ctx = VelloCtxCpu::new(800, 600);
        assert_eq!(ctx.commands().len(), 0);
    }

    #[test]
    fn fill_rect_appends_command() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        ctx.fill_rect([10.0, 20.0, 100.0, 30.0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(ctx.commands().len(), 1);
        match &ctx.commands()[0] {
            DrawCmd::FillRect { rect, color } => {
                println!("rect[0]={} color[0]={}", rect[0], color[0]);
                assert!((rect[0] - 10.0).abs() < 1e-6);
                assert!((color[0] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn fill_rect_multiple_commands() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        for _ in 0..16 { ctx.fill_rect([0.0; 4], [0.0; 4]); }
        assert_eq!(ctx.commands().len(), 16);
    }

    #[test]
    fn begin_frame_clears_commands() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        ctx.fill_rect([0.0; 4], [0.0; 4]);
        assert_eq!(ctx.commands().len(), 1);
        ctx.begin_frame();
        assert_eq!(ctx.commands().len(), 0);
    }

    #[test]
    fn rasterize_opaque_red_fills_rect_and_leaves_outside_untouched() {
        let w = 16u32;
        let h = 16u32;
        let mut ctx = VelloCtxCpu::new(w, h);
        // Opaque red rect covering [4,4] .. [12,12).
        ctx.fill_rect([4.0, 4.0, 8.0, 8.0], [1.0, 0.0, 0.0, 1.0]);
        let mut pixels = vec![[0u8, 0, 0, 255]; (w * h) as usize];
        ctx.rasterize_into(&mut pixels, w, h);

        // Interior sample point (6, 6).
        let inside = pixels[(6 * w + 6) as usize];
        println!("inside={:?}", inside);
        assert_eq!(inside, [255, 0, 0, 255], "interior pixel should be opaque red");

        // Outside sample point (0, 0) must be untouched black.
        let outside = pixels[0];
        assert_eq!(outside, [0, 0, 0, 255], "outside pixel should remain black");

        // Just outside the right edge (x=12, y=6) is exclusive -> untouched.
        let edge = pixels[(6 * w + 12) as usize];
        assert_eq!(edge, [0, 0, 0, 255], "pixel at exclusive right edge untouched");
    }

    #[test]
    fn rasterize_half_alpha_white_over_black_is_mid_grey() {
        let w = 8u32;
        let h = 8u32;
        let mut ctx = VelloCtxCpu::new(w, h);
        ctx.fill_rect([0.0, 0.0, 8.0, 8.0], [1.0, 1.0, 1.0, 0.5]);
        let mut pixels = vec![[0u8, 0, 0, 255]; (w * h) as usize];
        ctx.rasterize_into(&mut pixels, w, h);

        let p = pixels[(3 * w + 3) as usize];
        println!("blended={:?}", p);
        // out = 1.0*0.5 + 0.0*0.5 = 0.5 -> 128 (with +0.5 rounding).
        assert!((p[0] as i32 - 127).abs() <= 1, "R should be ~127, got {}", p[0]);
        assert!((p[1] as i32 - 127).abs() <= 1, "G should be ~127, got {}", p[1]);
        assert!((p[2] as i32 - 127).abs() <= 1, "B should be ~127, got {}", p[2]);
    }

    #[test]
    fn rasterize_offscreen_rect_leaves_buffer_unchanged() {
        let w = 8u32;
        let h = 8u32;
        let mut ctx = VelloCtxCpu::new(w, h);
        // Entirely off the right/bottom of the buffer.
        ctx.fill_rect([100.0, 100.0, 10.0, 10.0], [1.0, 0.0, 0.0, 1.0]);
        // Entirely off the top-left (negative).
        ctx.fill_rect([-50.0, -50.0, 10.0, 10.0], [0.0, 1.0, 0.0, 1.0]);
        let original = vec![[3u8, 7, 11, 255]; (w * h) as usize];
        let mut pixels = original.clone();
        ctx.rasterize_into(&mut pixels, w, h);
        assert_eq!(pixels, original, "off-screen rects must not modify any pixel");
    }

    #[test]
    fn rasterize_alpha_zero_is_noop() {
        let w = 4u32;
        let h = 4u32;
        let mut ctx = VelloCtxCpu::new(w, h);
        ctx.fill_rect([0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 0.0, 0.0]);
        let original = vec![[10u8, 20, 30, 255]; (w * h) as usize];
        let mut pixels = original.clone();
        ctx.rasterize_into(&mut pixels, w, h);
        assert_eq!(pixels, original, "alpha-0 fill must be a no-op");
    }

    #[test]
    fn rasterize_partially_offscreen_clips_to_bounds() {
        let w = 8u32;
        let h = 8u32;
        let mut ctx = VelloCtxCpu::new(w, h);
        // Straddles the top-left corner: covers [-2,-2]..[3,3) -> visible [0,0]..[3,3).
        ctx.fill_rect([-2.0, -2.0, 5.0, 5.0], [0.0, 0.0, 1.0, 1.0]);
        let mut pixels = vec![[0u8, 0, 0, 255]; (w * h) as usize];
        ctx.rasterize_into(&mut pixels, w, h);

        // (0,0) is inside the clipped region -> blue.
        assert_eq!(pixels[0], [0, 0, 255, 255]);
        // (2,2) still inside.
        assert_eq!(pixels[(2 * w + 2) as usize], [0, 0, 255, 255]);
        // (3,3) is outside the [0,3) span -> untouched.
        assert_eq!(pixels[(3 * w + 3) as usize], [0, 0, 0, 255]);
    }

    #[test]
    fn resize_updates_dimensions() {
        let mut ctx = VelloCtxCpu::new(800, 600);
        ctx.resize(1920, 1080);
        println!("width={} height={}", ctx.width(), ctx.height());
        assert_eq!(ctx.width(), 1920);
        assert_eq!(ctx.height(), 1080);
    }

    // --- Real GPU (Vello) headless pixel tests ---------------------------
    //
    // These exercise the *actual* vello::Renderer over a wgpu device and read
    // the rendered texture back to CPU. They self-skip (return) when no GPU
    // adapter is present so headless CI without a GPU stays green; on a machine
    // with Vulkan/Metal/GL they assert real computed pixel values produced by
    // the GPU compute pipeline, not by the CPU `VelloCtxCpu` stub.
    #[cfg(feature = "game-ui")]
    #[test]
    fn vello_gpu_fill_rect_produces_red_pixels_on_gpu() {
        let Some(mut ctx) = VelloCtx::new_headless(64, 64) else {
            eprintln!("[vello] no GPU adapter — skipping GPU fill_rect test");
            return;
        };
        ctx.begin_frame();
        // Opaque red rect covering the centre.
        ctx.fill_rect([16.0, 16.0, 32.0, 32.0], [1.0, 0.0, 0.0, 1.0]);
        let pixels = ctx.render_to_rgba().expect("gpu render");
        assert_eq!(pixels.len(), 64 * 64, "pixel count must match width*height");

        // Centre pixel must be (near) opaque red — the GPU rasterised it.
        let centre = pixels[(32 * 64 + 32) as usize];
        println!("[vello] gpu centre pixel = {:?}", centre);
        assert!(centre[0] > 200, "centre R should be high (red), got {}", centre[0]);
        assert!(centre[1] < 64, "centre G should be low, got {}", centre[1]);
        assert!(centre[2] < 64, "centre B should be low, got {}", centre[2]);

        // A corner outside the rect must be black background.
        let corner = pixels[0];
        println!("[vello] gpu corner pixel = {:?}", corner);
        assert!(corner[0] < 32 && corner[1] < 32 && corner[2] < 32,
            "corner should be black background, got {:?}", corner);

        // The red region must actually cover a meaningful number of pixels.
        let red_px = pixels.iter()
            .filter(|p| p[0] > 200 && p[1] < 64 && p[2] < 64)
            .count();
        println!("[vello] gpu red_px = {}", red_px);
        // 32x32 rect = 1024 px; allow AA slack on the border.
        assert!(red_px > 900, "expected >900 red px, got {}", red_px);
    }

    // --- GPU-STACK UNIFICATION PROOF -------------------------------------
    //
    // The point of the vello 0.4 -> 0.5.1 bump: vello 0.5.1 pins wgpu 24.0.3,
    // the SAME major as vox_render's wgpu 24. Before the bump, vello dragged in
    // wgpu 23 and its GPU UI ran on a SECOND device, compositing across the CPU.
    // After the bump, ONE wgpu-24 device can serve BOTH vello and vox_render.
    //
    // VARIANT SHIPPED: (b) — "shared device satisfies both requirement sets".
    // We CANNOT use variant (a) (dev-dep on vox_render + call its API) because
    // vox_render already depends on vox_ui (see vox_render/Cargo.toml), so a
    // `vox_ui -> vox_render` dev-dependency would form a dependency cycle. Cargo
    // rejects that even for dev-deps in the same workspace edge.
    //
    // Instead we build ONE wgpu-24 `Device`/`Queue` configured with vox_render's
    // EXACT device requirements — `Features::empty()` plus the most demanding
    // limits any vox_render GPU entry point requests (the `GpuGi` compute path:
    // unbounded storage-buffer count/size and compute-workgroup dispatch, which
    // is a strict superset of the `downlevel_defaults()` used by the splat
    // backend and material eval). We then hand that device to a vello 0.5
    // `VelloCtx::new` (the windowed-path constructor that consumes a CALLER's
    // device), render a real scene through it, and read pixels back.
    //
    // The assertion is a real computed pixel outcome: the centre of a red rect
    // rendered by vello on the shared device reads back as opaque red (R high,
    // G/B low) and >900 of the rect's pixels are red. If vello 0.5 could not run
    // on a device built to vox_render's spec, `VelloCtx::new` or the render would
    // fail and these pixels would never appear — proving the unification.
    //
    // Self-skips when no GPU adapter is present (headless CI without Vulkan/GL),
    // matching the existing GPU tests in this module.
    #[cfg(feature = "game-ui")]
    #[test]
    fn shared_wgpu24_device_drives_both_vello_and_vox_render_requirements() {
        use vello::wgpu;

        let w = 64u32;
        let h = 64u32;

        // ONE device, built to vox_render's exact requirements (wgpu 24).
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let Some(adapter) = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        )) else {
            eprintln!("[vello] no GPU adapter — skipping shared-device unification proof");
            return;
        };
        if is_software_adapter(&adapter.get_info()) && !software_gpu_allowed() {
            eprintln!("[vello] software adapter — skipping shared-device unification proof");
            return;
        }

        // vox_render's most demanding device request (GpuGi compute path):
        //   features = empty, limits = default + unbounded storage/compute.
        // Mirrors crates/vox_render/src/spectral_gi.rs new_blocking().
        // Clamp the "unbounded" sentinels to what the adapter actually exposes
        // (u32::MAX/u64::MAX are requests, not guarantees — vox_render's own
        // GpuGi path is similarly bounded by the adapter). Everything else stays
        // at vox_render's request: default base limits + maxed storage/compute.
        let al = adapter.limits();
        let shared_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage,
            max_buffer_size: al.max_buffer_size,
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            ..wgpu::Limits::default()
        };

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("shared-vello-and-vox_render-device"),
                required_features: wgpu::Features::empty(),
                required_limits: shared_limits,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )) {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("[vello] shared device creation failed ({e}) — skipping proof");
                return;
            }
        };

        // (a) Drive VELLO on the shared device via the caller-supplied-device
        // constructor (the windowed path). vello 0.5 / wgpu 24 must accept it.
        let mut ctx = VelloCtx::new(
            &device,
            &queue,
            w,
            h,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("vello 0.5 Renderer must construct on a device built to vox_render's spec");

        ctx.begin_frame();
        // Opaque red rect over the centre.
        ctx.fill_rect([16.0, 16.0, 32.0, 32.0], [1.0, 0.0, 0.0, 1.0]);

        // Render vello's scene to an Rgba8Unorm target on the SHARED device, then
        // read it back. (We own the target here; end_frame takes the same device.)
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shared-device-vello-target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        ctx.end_frame(&device, &queue, &view)
            .expect("vello render_to_texture on the shared device must succeed");

        // Read the rendered texture back to CPU pixels.
        let unpadded_bpr = w * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared-device-readback"),
            size: (padded_bpr * h) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shared-device-copy"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map sender").expect("buffer map");
        let data = slice.get_mapped_range();
        let mut pixels = vec![[0u8; 4]; (w * h) as usize];
        for y in 0..h {
            let row = (y * padded_bpr) as usize;
            for x in 0..w {
                let p = row + (x * 4) as usize;
                pixels[(y * w + x) as usize] = [data[p], data[p + 1], data[p + 2], data[p + 3]];
            }
        }
        drop(data);
        readback.unmap();

        // (b) THE PROOF: vello rendered correctly on the device built to
        // vox_render's spec. Centre pixel is opaque red; the rect is solidly red.
        let centre = pixels[(32 * 64 + 32) as usize];
        println!("[vello] shared-device centre pixel = {:?}", centre);
        assert!(centre[0] > 200, "centre R should be high (red), got {}", centre[0]);
        assert!(centre[1] < 64, "centre G should be low, got {}", centre[1]);
        assert!(centre[2] < 64, "centre B should be low, got {}", centre[2]);

        let red_px = pixels.iter()
            .filter(|p| p[0] > 200 && p[1] < 64 && p[2] < 64)
            .count();
        println!("[vello] shared-device red_px = {}", red_px);
        assert!(
            red_px > 900,
            "vello on the vox_render-spec device must paint the 32x32 red rect; got {red_px} red px",
        );
    }
}
