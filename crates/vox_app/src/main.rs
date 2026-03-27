mod demo_asset;
pub mod systems;
pub mod ui;

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::gpu_rasteriser::GpuRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct App {
    /// Set once `resumed` fires.
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    gpu_rasteriser: Option<GpuRasteriser>,
    /// Combined splat list: two building instances side by side.
    world_splats: Vec<GaussianSplat>,
    /// Orbit angle in radians.
    camera_angle: f32,
    last_frame: Instant,
    /// Instant of last FPS print.
    fps_timer: Instant,
    frame_count: u64,
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
            backend: None,
            gpu_rasteriser: None,
            world_splats: splats,
            camera_angle: 0.0,
            last_frame: now,
            fps_timer: now,
            frame_count: 0,
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

        let backend = WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT);

        let gpu_rasteriser = GpuRasteriser::new(
            backend.device(),
            backend.surface_format(),
            WIDTH, HEIGHT,
        );
        self.gpu_rasteriser = Some(gpu_rasteriser);

        self.window = Some(window);
        self.backend = Some(backend);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(backend) = self.backend.as_mut() {
                    let w = size.width.max(1);
                    let h = size.height.max(1);
                    backend.resize(w, h);
                }
                if let Some(gpu) = &mut self.gpu_rasteriser {
                    gpu.resize(size.width, size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                // --- Timing ---
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // --- Update orbit angle ---
                self.camera_angle += dt * 0.3;

                // --- Build camera ---
                // Orbit around the midpoint between the two instances.
                let target = Vec3::new(20.0, 7.5, 6.0); // centre of both buildings
                let radius = 50.0_f32;
                let cam_x = target.x + radius * self.camera_angle.cos();
                let cam_z = target.z + radius * self.camera_angle.sin();
                let cam_y = 18.0_f32;
                let eye = Vec3::new(cam_x, cam_y, cam_z);

                let view = Mat4::look_at_rh(eye, target, Vec3::Y);
                let aspect = WIDTH as f32 / HEIGHT as f32;
                let proj = Mat4::perspective_rh(
                    std::f32::consts::FRAC_PI_4, // 45° fov
                    aspect,
                    0.1,
                    200.0,
                );
                let camera = RenderCamera { view, proj };

                // --- GPU rasterise + present ---
                let backend = self.backend.as_ref().unwrap();
                let output = match backend.surface().get_current_texture() {
                    Ok(t) => t,
                    Err(_) => return,
                };
                let view_tex = output.texture.create_view(&Default::default());

                self.gpu_rasteriser.as_ref().unwrap().render(
                    backend.device(),
                    backend.queue(),
                    &view_tex,
                    &self.world_splats,
                    &camera,
                    &Illuminant::d65(),
                );

                output.present();

                // --- FPS counter ---
                self.frame_count += 1;
                let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if elapsed >= 2.0 {
                    let fps = self.frame_count as f32 / elapsed;
                    println!("FPS: {fps:.1}");
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
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop failed");
}
