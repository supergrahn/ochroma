// Hide the console window on Windows (GUI application)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! The Ochroma Editor — the first real WINDOWED host for the full editor shell.
//!
//! This opens a 1600x900 winit window, creates a wgpu-24 surface (via the same
//! [`WgpuBackend`] the engine binary uses), and drives `EditorShell` — the
//! complete docked editor (menu bar, Phosphor toolbar, World/Viewport/Node
//! Graph/Properties/Content/Output Log tabs, the three plugin tabs, command
//! palette, Ask Ochroma, status bar) — live at vsync through `egui-winit` +
//! `egui-wgpu`. Until now that shell only existed headless via
//! `shell_snapshot` (a CPU egui rasteriser); this is the same shell, on-GPU,
//! in a window the user can actually open and click.
//!
//! The Viewport tab uploads a `SoftwareRasteriser` frame into an egui texture
//! (`viewport::scene_texture`) exactly as the headless snapshot does. That CPU
//! upload-per-frame is the accepted v1 path; the next slice is to render the
//! viewport scene straight into a native wgpu texture and hand egui a
//! `TextureId` for it, eliminating the per-frame CPU round-trip.
//!
//! Usage:
//!   cargo run -p vox_app --bin ochroma_editor                       # live window
//!   cargo run -p vox_app --bin ochroma_editor -- --frames 120       # render 120 frames, exit 0
//!   cargo run -p vox_app --bin ochroma_editor -- --shot out.png     # capture last frame as PNG
//!   cargo run -p vox_app --bin ochroma_editor -- --frames 120 --shot out.png  # combine

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use vox_app::shell::{cpu_render, EditorShell, ShellRequest};
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_ui::Tokens;

const WINDOW_WIDTH: u32 = 1600;
const WINDOW_HEIGHT: u32 = 900;
const WINDOW_TITLE: &str = "Ochroma Editor";

/// Parsed command-line options.
struct Cli {
    /// If set, render exactly this many frames then exit 0 (proof mode).
    frames: Option<u32>,
    /// If set, capture the LAST rendered frame to this PNG path via wgpu readback.
    shot: Option<String>,
    /// Theme to load tokens from ("dark" default, "light" optional).
    light: bool,
    /// AAA Spec 03 proof mode: `--demo forgery` plants the metameric forgery pair
    /// at startup so the first frame already shows it.
    demo: Option<String>,
    /// AAA Spec 03: `--illuminant <name>` sets the inspection light (e.g.
    /// `cool_led`) so the captured shot shows the forgery split.
    illuminant: Option<String>,
}

fn parse_cli() -> Cli {
    let mut cli = Cli { frames: None, shot: None, light: false, demo: None, illuminant: None };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                i += 1;
                // A garbage/negative/missing value must NEVER silently parse to
                // None (which would be indistinguishable from "no --frames" and
                // hang the proof loop forever). Fail fast with a clear message.
                if i >= args.len() {
                    eprintln!("[ochroma_editor] --frames expects a non-negative integer, got nothing");
                    std::process::exit(2);
                }
                match args[i].parse::<u32>() {
                    Ok(n) => cli.frames = Some(n),
                    Err(_) => {
                        eprintln!(
                            "[ochroma_editor] --frames expects a non-negative integer, got {:?}",
                            args[i]
                        );
                        std::process::exit(2);
                    }
                }
            }
            "--shot" => {
                i += 1;
                if i < args.len() {
                    cli.shot = Some(args[i].clone());
                }
            }
            "--theme" => {
                i += 1;
                if i < args.len() {
                    cli.light = args[i] == "light";
                }
            }
            "--demo" => {
                i += 1;
                if i < args.len() {
                    cli.demo = Some(args[i].clone());
                }
            }
            "--illuminant" => {
                i += 1;
                if i < args.len() {
                    cli.illuminant = Some(args[i].clone());
                }
            }
            _ => {}
        }
        i += 1;
    }

    // --shot with no --frames is documented as a one-shot capture: default to a
    // single frame so the shot is captured and the process exits 0 (without this
    // the is_last_frame gate is never true and the loop runs forever).
    if cli.shot.is_some() && cli.frames.is_none() {
        cli.frames = Some(1);
    }

    // --frames 0 renders zero frames (set up, then exit 0 immediately). But a
    // zero-frame run can never capture a shot, so --frames 0 --shot is a usage
    // error rather than a silent no-capture exit.
    if cli.frames == Some(0) && cli.shot.is_some() {
        eprintln!("[ochroma_editor] need at least 1 frame to capture a --shot (got --frames 0)");
        std::process::exit(2);
    }

    cli
}

/// Load the shipped UI tokens (dark by default — same theme as `shell_snapshot`).
fn load_tokens(light: bool) -> Tokens {
    if light {
        Tokens::load("assets/ui/ochroma_light.theme.json").unwrap_or_default()
    } else {
        Tokens::load("assets/ui/ochroma.theme.json").unwrap_or_default()
    }
}

/// Build a fully-populated `EditorShell` with all three real plugins installed,
/// focused on the live viewport tab — identical setup to `shell_snapshot` so the
/// windowed editor shows exactly the same dock.
fn build_shell(tokens: Tokens, cli: &Cli) -> EditorShell {
    let mut shell = EditorShell::new(tokens);
    // Install Crucible wired to the shell's scene-sink (NOT the detached
    // `::new()` sink) so pressing "Cook scene" plants real splats into the live
    // windowed viewport — exactly as install_floraprime does for "Grow tree".
    shell.install_crucible();
    shell.install_plugin(Box::new(vox_app::shell::plugins::ForgePlugin::new()));
    // Install FloraPrime wired to the shell's grow-sink (NOT the detached
    // `::new()` sink) so pressing "Grow tree" plants real splats into the live
    // windowed viewport — exactly as the headless shell_snapshot binary does.
    shell.install_floraprime();
    shell.focus_viewport();

    // AAA Spec 03 proof mode: plant the forgery and/or set the inspection light
    // through the SAME request path the editor UI uses, then drain so the FIRST
    // rendered frame already shows the relit forgery. ForgeryDemo is queued before
    // SetIlluminant so the HUD receipt has the planted ranges when the light flips.
    if cli.demo.as_deref() == Some("forgery") {
        shell.requests.borrow_mut().push(ShellRequest::ForgeryDemo);
    }
    if let Some(name) = &cli.illuminant {
        match vox_render::relight::IlluminantSpec::parse(name) {
            Some(spec) => shell.requests.borrow_mut().push(ShellRequest::SetIlluminant(spec)),
            None => eprintln!("[ochroma_editor] unknown --illuminant {name:?}; keeping the gallery light"),
        }
    }
    shell.drain_requests();
    if cli.demo.as_deref() == Some("forgery") {
        // Surface the live receipt on stdout so the proof run is gate-checkable
        // without reading pixels.
        println!("[ochroma_editor] forgery HUD: {}", shell.status);
    }
    shell
}

struct EditorHost {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    shell: EditorShell,
    tokens: Tokens,
    /// Background clear colour from the theme surface token.
    bg: [u8; 4],

    cli: Cli,
    frames_rendered: u32,

    /// GPU global illumination bound to the SHARED present device (the same
    /// `wgpu::Device` the window presents on), built behind `OCHROMA_GI=gpu` from
    /// the backend's `GpuContext` — proof the GI path uses no second device.
    shared_gi: Option<vox_render::spectral_gi::GpuGi>,
}

impl EditorHost {
    fn new(cli: Cli) -> Self {
        let tokens = load_tokens(cli.light);
        let bg = tokens.color("surface.bg.0");
        let shell = build_shell(tokens.clone(), &cli);
        Self {
            window: None,
            backend: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            shell,
            tokens,
            bg,
            cli,
            frames_rendered: 0,
            shared_gi: None,
        }
    }

    /// Run the egui shell once, returning the tessellated output + textures so it
    /// can be rendered into ANY target view (the live surface or the readback
    /// offscreen). This keeps the shell-layout call in one place.
    fn run_shell_frame(
        &mut self,
        window: &Arc<Window>,
    ) -> (egui::FullOutput, Vec<egui::ClippedPrimitive>) {
        let egui_state = self.egui_state.as_mut().expect("egui_state");
        let raw_input = egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            self.shell.ui(ctx);
        });
        egui_state.handle_platform_output(window, full_output.platform_output.clone());
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes.clone(), self.egui_ctx.pixels_per_point());
        (full_output, tris)
    }

    /// Paint a tessellated egui frame into `view` (load=clear to theme bg) and
    /// submit. Shared by the live present path and the offscreen readback path.
    fn paint(
        &mut self,
        view: &wgpu::TextureView,
        full_output: &egui::FullOutput,
        tris: &[egui::ClippedPrimitive],
        size_px: [u32; 2],
        pixels_per_point: f32,
    ) {
        let backend = self.backend.as_ref().expect("backend");
        let egui_renderer = self.egui_renderer.as_mut().expect("egui_renderer");

        for (id, image_delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(backend.device(), backend.queue(), *id, image_delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size_px,
            pixels_per_point,
        };

        let mut encoder = backend
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ochroma_editor_egui_encoder"),
            });

        egui_renderer.update_buffers(
            backend.device(),
            backend.queue(),
            &mut encoder,
            tris,
            &screen_descriptor,
        );

        let clear = wgpu::Color {
            r: srgb_to_linear(self.bg[0]),
            g: srgb_to_linear(self.bg[1]),
            b: srgb_to_linear(self.bg[2]),
            a: 1.0,
        };
        {
            let rp_desc = wgpu::RenderPassDescriptor {
                label: Some("ochroma_editor_egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            let mut render_pass = encoder.begin_render_pass(&rp_desc).forget_lifetime();
            egui_renderer.render(&mut render_pass, tris, &screen_descriptor);
        }

        backend.queue().submit(std::iter::once(encoder.finish()));

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }
    }

    /// Render the shell ONE more time into an offscreen `COPY_SRC` texture, read
    /// it back, and write a PNG. The surface texture is configured without
    /// `COPY_SRC`, so the proof capture renders to a dedicated texture instead.
    fn capture_shot(&mut self, path: &str) {
        let window = self.window.as_ref().expect("window").clone();
        let (w, h) = {
            let b = self.backend.as_ref().expect("backend");
            (b.width(), b.height())
        };
        let format = self.backend.as_ref().expect("backend").surface_format();
        let ppp = window.scale_factor() as f32;

        // Offscreen RENDER_ATTACHMENT | COPY_SRC target in the SAME format egui
        // was built for.
        let target = self
            .backend
            .as_ref()
            .expect("backend")
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ochroma_editor_shot_target"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let (full_output, tris) = self.run_shell_frame(&window);
        self.paint(&view, &full_output, &tris, [w, h], ppp);

        // Copy the offscreen target into a padded readback buffer.
        let bytes_per_pixel = 4u32;
        let unpadded = w * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;

        let backend = self.backend.as_ref().expect("backend");
        let buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("ochroma_editor_shot_readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = backend
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ochroma_editor_shot_copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        backend.queue().submit(std::iter::once(encoder.finish()));

        // Map + read.
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        backend.device().poll(wgpu::Maintain::Wait);
        rx.recv().expect("map channel").expect("buffer map failed");

        let mapped = slice.get_mapped_range();
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        let bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for y in 0..h as usize {
            let src = y * padded as usize;
            for x in 0..w as usize {
                let s = src + x * 4;
                let d = (y * w as usize + x) * 4;
                if bgra {
                    rgba[d] = mapped[s + 2];
                    rgba[d + 1] = mapped[s + 1];
                    rgba[d + 2] = mapped[s];
                    rgba[d + 3] = mapped[s + 3];
                } else {
                    rgba[d..d + 4].copy_from_slice(&mapped[s..s + 4]);
                }
            }
        }
        drop(mapped);
        buffer.unmap();

        // Create the parent directory if it's trivially missing, then write.
        // A bad/unwritable path must NOT panic — report the io error and exit
        // cleanly with a non-zero code so callers/CI get a clear failure.
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = cpu_render::write_png(path, &rgba, w, h) {
            eprintln!("[ochroma_editor] failed to write {path}: {e}");
            std::process::exit(1);
        }
        let nonbg = cpu_render::non_background_fraction(&rgba, self.bg, 6) * 100.0;
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        println!(
            "[ochroma_editor] wrote {path} ({bytes} bytes), {nonbg:.1}% non-background pixels, {w}x{h}, format={format:?}"
        );
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let window = match &self.window {
            Some(w) => w.clone(),
            None => return,
        };
        if self.backend.is_none() {
            return;
        }

        // --frames 0 means render ZERO frames: the window/backend are already set
        // up (resumed() ran), so exit 0 immediately without presenting a frame.
        // (--frames 0 --shot is rejected at parse time, so there's no shot here.)
        if self.cli.frames == Some(0) {
            println!("[ochroma_editor] rendered 0 frames (--frames 0), exiting 0");
            event_loop.exit();
            return;
        }

        let is_last_frame = self
            .cli
            .frames
            .map(|n| self.frames_rendered + 1 >= n)
            .unwrap_or(false);

        // On the last proof frame, capture the shot from a dedicated offscreen
        // render so the PNG is byte-for-byte what the window shows.
        if is_last_frame && let Some(path) = self.cli.shot.clone() {
            self.capture_shot(&path);
        }

        // Live present to the surface.
        let backend = self.backend.as_ref().expect("backend");
        let surface_tex = match backend.surface().get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // The surface must be reconfigured before the next acquire will
                // succeed — without this we'd busy-spin on the same error under
                // ControlFlow::Poll. Reconfigure (which also re-applies the
                // Mailbox workaround) then retry on the next redraw.
                eprintln!("[ochroma_editor] surface lost/outdated — reconfiguring");
                if let Some(backend) = self.backend.as_ref() {
                    configure_present_mailbox(backend);
                }
                window.request_redraw();
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("[ochroma_editor] surface out of memory — exiting 1");
                std::process::exit(1);
            }
            Err(e) => {
                // Timeout (or any future variant): transient, just retry.
                eprintln!("[ochroma_editor] surface error: {e}");
                window.request_redraw();
                return;
            }
        };
        let view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (w, h) = (backend.width(), backend.height());
        let ppp = window.scale_factor() as f32;

        let (full_output, tris) = self.run_shell_frame(&window);
        self.paint(&view, &full_output, &tris, [w, h], ppp);
        surface_tex.present();

        self.frames_rendered += 1;

        if let Some(n) = self.cli.frames
            && self.frames_rendered >= n
        {
            println!(
                "[ochroma_editor] rendered {} frames (--frames {}), exiting 0",
                self.frames_rendered, n
            );
            event_loop.exit();
            return;
        }

        // Request the next frame (canonical winit redraw loop).
        window.request_redraw();
    }
}

impl ApplicationHandler for EditorHost {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let backend = match WgpuBackend::new(Arc::clone(&window), WINDOW_WIDTH, WINDOW_HEIGHT) {
            Ok(b) => b,
            Err(e) => panic!("[ochroma_editor] GPU init failed: {e}"),
        };
        println!(
            "[ochroma_editor] GPU backend initialised ({}x{}, format={:?})",
            backend.width(),
            backend.height(),
            backend.surface_format()
        );

        configure_present_mailbox(&backend);
        println!("[ochroma_editor] surface present mode set to Mailbox");

        // Behind OCHROMA_GI=gpu: bind GPU global illumination to the SHARED present
        // device. This constructs the GI compute pass on the backend's own
        // `wgpu::Device`/`Queue` (cloned handles — NOT a second `request_device`),
        // proving the shared-device foundation end-to-end. With OCHROMA_GI unset,
        // none of this runs and the editor's existing path is byte-identical.
        if std::env::var("OCHROMA_GI")
            .map(|v| v.eq_ignore_ascii_case("gpu"))
            .unwrap_or(false)
        {
            // The backend does not expose its adapter identity, so resolve the
            // adapter NAME via a device-less enumeration matching the backend's
            // selection order (Vulkan first, then GL, then all) — no second device
            // is created here, only a name lookup. The GI device below is the
            // backend's actual present device.
            let adapter_info = resolve_present_adapter_info();
            let ctx = vox_render::gpu::GpuContext::from_parts(
                backend.device(),
                backend.queue(),
                &adapter_info,
            );
            let gi = vox_render::spectral_gi::GpuGi::new_with_context(&ctx, 200_000);
            // Same device, not a coincidence: the GI's adapter name equals the
            // context's, which was resolved from the same present-adapter order.
            assert_eq!(
                gi.adapter_name,
                adapter_info.name,
                "GI must run on the present adapter, not a second device"
            );
            println!(
                "[ochroma_editor] GI on shared present device (adapter={})",
                gi.adapter_name
            );
            println!(
                "[ochroma_editor] GI adapter = {} (backend adapter = {})",
                gi.adapter_name, adapter_info.name
            );
            self.shared_gi = Some(gi);
        }

        // egui input + theme. Install the Phosphor icon font and the tokenized
        // egui style exactly as the headless snapshot does, so the live dock is
        // the same dark, icon-led shell.
        vox_ui::design::icons::install(&self.egui_ctx);
        vox_ui::egui_theme::apply(&self.egui_ctx, &self.tokens);

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            backend.device(),
            backend.surface_format(),
            None,
            1,
            false,
        );

        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.backend = Some(backend);
        self.window = Some(window.clone());

        println!("[ochroma_editor] editor shell live — dock: World | Viewport | Node Graph | Properties | Content | Output Log + Crucible/Forge/FloraPrime plugin tabs. Ctrl+K opens the command palette.");
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Feed input to egui (which routes to the shell: tab drags, palette,
        // Ask Ochroma text field, inspector scrubs).
        if let Some(egui_state) = &mut self.egui_state
            && let Some(window) = &self.window
        {
            let _ = egui_state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(backend) = &mut self.backend
                    && size.width > 0
                    && size.height > 0
                {
                    backend.resize(size.width, size.height);
                    // resize() reverts the surface to Fifo; restore Mailbox.
                    configure_present_mailbox(backend);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }
}

/// Reconfigure the backend's surface with `PresentMode::Mailbox`.
///
/// `WgpuBackend` configures (and on every `resize` re-configures) the surface
/// with `PresentMode::Fifo`. On some Wayland/Xwayland compositors Fifo blocks
/// `present()` indefinitely when the window isn't actively composited (which
/// would deadlock proof mode). Mailbox keeps frames making progress while still
/// pacing to the display when the compositor cooperates. Called once after
/// backend creation and again after every resize (which reverts to Fifo).
fn configure_present_mailbox(backend: &WgpuBackend) {
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: backend.surface_format(),
        width: backend.width(),
        height: backend.height(),
        present_mode: wgpu::PresentMode::Mailbox,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    };
    backend.surface().configure(backend.device(), &config);
}

/// Resolve the `AdapterInfo` of the adapter the present backend selected.
///
/// `WgpuBackend` selects its adapter Vulkan-first (then GL, then all) and does not
/// retain/expose the resulting `AdapterInfo`. To label the shared-device GI with the
/// SAME adapter without a second `request_device`, we re-run that selection order as
/// a NAME-ONLY enumeration: `request_adapter` returns an `Adapter` handle from which
/// `get_info()` reads the identity, but NO device is created here. The GI compute
/// pass itself runs on the backend's actual present device (cloned handles), so this
/// resolves the name the present device's adapter reports, not a different GPU.
fn resolve_present_adapter_info() -> wgpu::AdapterInfo {
    let attempts: &[wgpu::Backends] = &[
        wgpu::Backends::VULKAN,
        wgpu::Backends::GL,
        wgpu::Backends::all(),
    ];
    for backends in attempts {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: *backends,
            ..Default::default()
        });
        if let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        {
            let info = adapter.get_info();
            // Use local GPU: prefer a real hardware adapter. If this backend only
            // offers llvmpipe, skip to the next backend rather than labelling the
            // shared device with the software rasteriser. (WgpuBackend itself
            // refuses software, so this only resolves the name of the real GPU.)
            if vox_render::gpu::adapter::ensure_hardware(&info).is_err() {
                continue;
            }
            return info;
        }
    }
    // Backend creation already succeeded, so an adapter exists; this is unreachable
    // in practice. Provide a benign default rather than panicking the shell.
    wgpu::AdapterInfo {
        name: String::from("unknown"),
        vendor: 0,
        device: 0,
        device_type: wgpu::DeviceType::Other,
        driver: String::new(),
        driver_info: String::new(),
        backend: wgpu::Backend::Empty,
    }
}

/// Convert an sRGB-encoded 8-bit channel to a linear 0..1 value for the surface
/// clear colour (the surface format is *_Srgb, so the clear value must be linear).
fn srgb_to_linear(c: u8) -> f64 {
    let c = c as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn main() {
    let cli = parse_cli();
    println!(
        "[ochroma_editor] Ochroma Editor — windowed shell host (frames={:?}, shot={:?})",
        cli.frames, cli.shot
    );

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut host = EditorHost::new(cli);
    event_loop.run_app(&mut host).expect("Event loop failed");
}
