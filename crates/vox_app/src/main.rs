mod demo_asset;
pub mod headless;
pub mod systems;
pub mod ui;

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::gpu_rasteriser::GpuRasteriser;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use egui_wgpu::wgpu;
use ui::PlopUi;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// The render backend — either GPU-accelerated or CPU software fallback.
enum RenderMode {
    Gpu {
        backend: WgpuBackend,
        gpu_rasteriser: GpuRasteriser,
        egui_state: egui_winit::State,
        egui_renderer: egui_wgpu::Renderer,
        egui_ctx: egui::Context,
    },
    Software {
        backend: WgpuBackend,
        rasteriser: SoftwareRasteriser,
    },
    /// Pure CPU fallback — no GPU at all. The app still runs but cannot present
    /// to a window surface (headless-style). We print frames to console only.
    CpuOnly {
        rasteriser: SoftwareRasteriser,
    },
}

struct App {
    /// Set once `resumed` fires.
    window: Option<Arc<Window>>,
    render_mode: Option<RenderMode>,
    /// Combined splat list: two building instances side by side.
    world_splats: Vec<GaussianSplat>,
    /// Orbit angle in radians.
    camera_angle: f32,
    last_frame: Instant,
    /// Instant of last FPS print.
    fps_timer: Instant,
    frame_count: u64,
    plop_ui: PlopUi,
}

impl App {
    fn new() -> Self {
        // Build the demo asset once.
        let asset = demo_asset::generate_building();

        // Two instances: origin and (20, 0, 0).
        let offsets = [Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0)];
        let mut splats: Vec<GaussianSplat> = Vec::with_capacity(asset.splats.len() * 2);
        for offset in offsets {
            for s in &asset.splats {
                let mut copy = *s;
                copy.position[0] += offset.x;
                copy.position[1] += offset.y;
                copy.position[2] += offset.z;
                splats.push(copy);
            }
        }

        let now = Instant::now();
        Self {
            window: None,
            render_mode: None,
            world_splats: splats,
            camera_angle: 0.0,
            last_frame: now,
            fps_timer: now,
            frame_count: 0,
            plop_ui: PlopUi::default(),
        }
    }

}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma — Phase 0")
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        // Try GPU backend; fall back to software if it fails.
        let render_mode = match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => {
                // Try GPU rasteriser first; if shader compilation or pipeline creation
                // panics we catch it below via the software path.
                let gpu_rasteriser = GpuRasteriser::new(
                    backend.device(),
                    backend.surface_format(),
                    WIDTH,
                    HEIGHT,
                );

                // Initialise egui
                let egui_ctx = egui::Context::default();
                let egui_state = egui_winit::State::new(
                    egui_ctx.clone(),
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

                eprintln!("[ochroma] GPU mode initialised");
                RenderMode::Gpu {
                    backend,
                    gpu_rasteriser,
                    egui_state,
                    egui_renderer,
                    egui_ctx,
                }
            }
            Err(e) => {
                eprintln!("[ochroma] GPU init failed: {e}");
                // Try a minimal wgpu backend for surface blitting with software rasteriser
                match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
                    Ok(backend) => {
                        eprintln!("[ochroma] Software rasteriser mode (with wgpu surface blit)");
                        RenderMode::Software {
                            backend,
                            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
                        }
                    }
                    Err(e2) => {
                        eprintln!("[ochroma] All GPU paths failed: {e2}");
                        eprintln!("[ochroma] Running in CPU-only mode (no window rendering)");
                        RenderMode::CpuOnly {
                            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
                        }
                    }
                }
            }
        };

        self.window = Some(window);
        self.render_mode = Some(render_mode);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Pass events to egui if in GPU mode
        if let Some(RenderMode::Gpu { egui_state, .. }) = &mut self.render_mode {
            if let Some(window) = self.window.as_ref() {
                let _ = egui_state.on_window_event(window, &event);
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                let w = size.width.max(1);
                let h = size.height.max(1);
                match &mut self.render_mode {
                    Some(RenderMode::Gpu { backend, gpu_rasteriser, .. }) => {
                        backend.resize(w, h);
                        gpu_rasteriser.resize(w, h);
                    }
                    Some(RenderMode::Software { backend, rasteriser }) => {
                        backend.resize(w, h);
                        rasteriser.width = w;
                        rasteriser.height = h;
                    }
                    Some(RenderMode::CpuOnly { rasteriser }) => {
                        rasteriser.width = w;
                        rasteriser.height = h;
                    }
                    None => {}
                }
            }

            WindowEvent::RedrawRequested => {
                // --- Puffin: new frame ---
                puffin::GlobalProfiler::lock().new_frame();

                // --- Timing ---
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // --- Update orbit angle ---
                self.camera_angle += dt * 0.3;

                // Pre-compute values that need &self before we mutably borrow render_mode.
                let window_clone = self.window.clone();
                let world_splats = &self.world_splats;
                let plop_ui = &mut self.plop_ui;
                let camera_angle = self.camera_angle;

                // Helper: build camera inline to avoid borrowing self.
                let make_camera = |w: u32, h: u32| -> RenderCamera {
                    let target = Vec3::new(20.0, 7.5, 6.0);
                    let radius = 50.0_f32;
                    let cam_x = target.x + radius * camera_angle.cos();
                    let cam_z = target.z + radius * camera_angle.sin();
                    let cam_y = 18.0_f32;
                    let eye = Vec3::new(cam_x, cam_y, cam_z);
                    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
                    let aspect = w as f32 / h as f32;
                    let proj = Mat4::perspective_rh(
                        std::f32::consts::FRAC_PI_4,
                        aspect,
                        0.1,
                        200.0,
                    );
                    RenderCamera { view, proj }
                };

                match &mut self.render_mode {
                    Some(RenderMode::Gpu {
                        backend,
                        gpu_rasteriser,
                        egui_state,
                        egui_renderer,
                        egui_ctx,
                    }) => {
                        puffin::profile_scope!("render");

                        let camera = make_camera(backend.width(), backend.height());

                        // --- GPU splat render ---
                        let output = match backend.surface().get_current_texture() {
                            Ok(t) => t,
                            Err(_) => return,
                        };
                        let view_tex = output.texture.create_view(&Default::default());

                        gpu_rasteriser.render(
                            backend.device(),
                            backend.queue(),
                            &view_tex,
                            world_splats,
                            &camera,
                            &Illuminant::d65(),
                        );

                        // --- egui render on top ---
                        let window = window_clone.as_ref().unwrap();
                        let raw_input = egui_state.take_egui_input(window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            plop_ui.show(ctx, &[]);
                        });

                        egui_state.handle_platform_output(window, full_output.platform_output);

                        let tris = egui_ctx.tessellate(full_output.shapes, egui_ctx.pixels_per_point());
                        for (id, image_delta) in &full_output.textures_delta.set {
                            egui_renderer.update_texture(backend.device(), backend.queue(), *id, image_delta);
                        }

                        let screen_descriptor = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [backend.width(), backend.height()],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        let mut encoder = backend.device().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("egui_encoder"),
                            },
                        );

                        egui_renderer.update_buffers(
                            backend.device(),
                            backend.queue(),
                            &mut encoder,
                            &tris,
                            &screen_descriptor,
                        );

                        {
                            let rp_desc = wgpu::RenderPassDescriptor {
                                label: Some("egui_render_pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view_tex,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            };
                            let mut render_pass = encoder
                                .begin_render_pass(&rp_desc)
                                .forget_lifetime();
                            egui_renderer.render(&mut render_pass, &tris, &screen_descriptor);
                        }

                        backend.queue().submit(std::iter::once(encoder.finish()));

                        for id in &full_output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }

                        output.present();
                    }
                    Some(RenderMode::Software { backend, rasteriser }) => {
                        puffin::profile_scope!("render");
                        let camera = make_camera(rasteriser.width, rasteriser.height);
                        let fb = rasteriser.render(world_splats, &camera, &Illuminant::d65());
                        backend.present_framebuffer(&fb.pixels, fb.width, fb.height);
                    }
                    Some(RenderMode::CpuOnly { rasteriser }) => {
                        puffin::profile_scope!("render");
                        let camera = make_camera(rasteriser.width, rasteriser.height);
                        let _fb = rasteriser.render(world_splats, &camera, &Illuminant::d65());
                        // No surface to present to — frame computed but discarded.
                    }
                    None => {}
                }

                // --- FPS counter with frame time ---
                self.frame_count += 1;
                let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if elapsed >= 2.0 {
                    let fps = self.frame_count as f32 / elapsed;
                    let frame_ms = elapsed * 1000.0 / self.frame_count as f32;
                    println!("FPS: {fps:.1}  frame: {frame_ms:.2}ms");
                    self.frame_count = 0;
                    self.fps_timer = now;
                }

                // --- Request next frame ---
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    puffin::set_scopes_on(true);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop failed");
}
