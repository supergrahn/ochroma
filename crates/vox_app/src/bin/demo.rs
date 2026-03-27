//! Ochroma Engine Demo -- a clean, working, interactive application.
//!
//! Usage:
//!   cargo run --bin demo                    # default scene
//!   cargo run --bin demo -- scene.ply       # load a .ply file

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::software_rasteriser::{Framebuffer, SoftwareRasteriser};
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct DemoApp {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    rasteriser: SoftwareRasteriser,

    // Scene
    scene_splats: Vec<GaussianSplat>,

    // Camera
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    cam_speed: f32,

    // Input state
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
    last_mouse: Option<(f64, f64)>,
    left_click_pending: bool,
    mouse_x: f64,
    mouse_y: f64,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    fps_display: f32,

    // Placed objects
    placed_objects: Vec<(Vec3, Vec<GaussianSplat>)>,

    // CLI arg: .ply file to load
    ply_path: Option<String>,
}

impl DemoApp {
    fn new(ply_path: Option<String>) -> Self {
        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            scene_splats: Vec::new(),
            cam_pos: Vec3::new(0.0, 10.0, 30.0),
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            cam_speed: 15.0,
            keys_held: std::collections::HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            left_click_pending: false,
            mouse_x: 0.0,
            mouse_y: 0.0,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            fps_display: 0.0,
            placed_objects: Vec::new(),
            ply_path,
        }
    }

    fn build_scene(&mut self) {
        // Try loading .ply file if provided
        if let Some(path) = &self.ply_path {
            match vox_data::ply_loader::load_ply(std::path::Path::new(path)) {
                Ok(splats) => {
                    println!("[ochroma] Loaded {} splats from {}", splats.len(), path);
                    self.scene_splats = splats;
                    return;
                }
                Err(e) => {
                    eprintln!("[ochroma] Failed to load {}: {}", path, e);
                    eprintln!("[ochroma] Falling back to default scene");
                }
            }
        }

        // Default scene: volumetric terrain with buildings
        println!("[ochroma] Building default scene...");

        // Volumetric terrain
        let vol = vox_terrain::volume::generate_demo_volume(42);
        let materials = vox_terrain::volume::default_volume_materials();
        let terrain_splats = vox_terrain::volume::volume_to_splats(&vol, &materials, 42);
        println!("[ochroma]   Terrain: {} splats", terrain_splats.len());

        // A few buildings
        let mut building_splats = Vec::new();
        for i in 0..4 {
            let b = vox_data::proc_gs_advanced::generate_detailed_building(
                42 + i,
                6.0,
                8.0,
                2 + (i % 3) as u32,
                "victorian",
            );
            for s in &b {
                let mut ws = *s;
                ws.position[0] += i as f32 * 10.0 - 15.0;
                ws.position[2] += 20.0;
                building_splats.push(ws);
            }
        }
        println!("[ochroma]   Buildings: {} splats", building_splats.len());

        // Trees
        let mut tree_splats = Vec::new();
        for i in 0..6 {
            let t = vox_data::proc_gs_advanced::generate_tree(100 + i, 6.0 + i as f32, 2.5);
            for s in &t {
                let mut ws = *s;
                ws.position[0] += i as f32 * 8.0 - 20.0;
                ws.position[2] += 10.0;
                tree_splats.push(ws);
            }
        }
        println!("[ochroma]   Trees: {} splats", tree_splats.len());

        self.scene_splats = terrain_splats;
        self.scene_splats.extend(building_splats);
        self.scene_splats.extend(tree_splats);
        println!("[ochroma] Total scene: {} splats", self.scene_splats.len());
    }

    fn camera_forward(&self) -> Vec3 {
        Vec3::new(
            self.cam_yaw.sin() * self.cam_pitch.cos(),
            self.cam_pitch.sin(),
            -self.cam_yaw.cos() * self.cam_pitch.cos(),
        )
        .normalize()
    }

    fn camera_right(&self) -> Vec3 {
        self.camera_forward().cross(Vec3::Y).normalize()
    }

    fn update_camera(&mut self, dt: f32) {
        let forward = self.camera_forward();
        let right = self.camera_right();
        let speed = self.cam_speed * dt;

        if self.keys_held.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * speed;
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * speed;
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * speed;
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            self.cam_pos += right * speed;
        }
        if self.keys_held.contains(&KeyCode::Space) {
            self.cam_pos.y += speed;
        }
        if self.keys_held.contains(&KeyCode::ShiftLeft) {
            self.cam_pos.y -= speed;
        }
    }

    fn render_frame(&mut self) -> Framebuffer {
        let forward = self.camera_forward();
        let target = self.cam_pos + forward;

        let camera = RenderCamera {
            view: Mat4::look_at_rh(self.cam_pos, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                self.rasteriser.width as f32 / self.rasteriser.height as f32,
                0.1,
                500.0,
            ),
        };

        // Combine scene + placed objects
        let mut all_splats = self.scene_splats.clone();
        for (pos, splats) in &self.placed_objects {
            for s in splats {
                let mut ws = *s;
                ws.position[0] += pos.x;
                ws.position[1] += pos.y;
                ws.position[2] += pos.z;
                all_splats.push(ws);
            }
        }

        self.rasteriser
            .render(&all_splats, &camera, &Illuminant::d65())
    }

    fn place_object_at_cursor(&mut self) {
        // Simple: place a tree at a fixed distance in front of camera
        let forward = self.camera_forward();
        let place_pos = self.cam_pos + forward * 15.0;
        let place_pos = Vec3::new(place_pos.x, 0.0, place_pos.z); // snap to ground

        let tree = vox_data::proc_gs_advanced::generate_tree(
            self.placed_objects.len() as u64 + 1000,
            7.0,
            3.0,
        );
        println!(
            "[ochroma] Placed tree at ({:.1}, {:.1}, {:.1})",
            place_pos.x, place_pos.y, place_pos.z
        );
        self.placed_objects.push((place_pos, tree));
    }
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma Engine Demo")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        // Try GPU backend, continue without if it fails
        match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => {
                println!("[ochroma] GPU backend initialised");
                self.backend = Some(backend);
            }
            Err(e) => {
                eprintln!("[ochroma] GPU init failed: {}", e);
                eprintln!("[ochroma] Running without GPU surface blit -- frames will render but may not display");
            }
        }

        self.window = Some(window);
        self.build_scene();

        println!("[ochroma] Controls:");
        println!("  WASD        -- move camera");
        println!("  Space/Shift -- up/down");
        println!("  Right-click -- capture mouse for look");
        println!("  Left-click  -- place a tree");
        println!("  Escape      -- release mouse / quit");
        println!("  F12         -- save screenshot");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys_held.insert(key);

                        match key {
                            KeyCode::Escape => {
                                if self.mouse_captured {
                                    self.mouse_captured = false;
                                    if let Some(w) = &self.window {
                                        w.set_cursor_visible(true);
                                    }
                                } else {
                                    event_loop.exit();
                                }
                            }
                            KeyCode::F12 => {
                                let fb = self.render_frame();
                                let dir = std::env::temp_dir().join("ochroma_visual");
                                std::fs::create_dir_all(&dir).ok();
                                let path = dir.join("screenshot.ppm");
                                let mut data =
                                    format!("P6\n{} {}\n255\n", fb.width, fb.height).into_bytes();
                                for p in &fb.pixels {
                                    data.push(p[0]);
                                    data.push(p[1]);
                                    data.push(p[2]);
                                }
                                std::fs::write(&path, &data).ok();
                                println!("[ochroma] Screenshot: {}", path.display());
                            }
                            _ => {}
                        }
                    } else {
                        self.keys_held.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Right => {
                    if state == ElementState::Pressed {
                        self.mouse_captured = true;
                        self.last_mouse = None;
                        if let Some(w) = &self.window {
                            w.set_cursor_visible(false);
                        }
                    } else {
                        self.mouse_captured = false;
                        if let Some(w) = &self.window {
                            w.set_cursor_visible(true);
                        }
                    }
                }
                MouseButton::Left if state == ElementState::Pressed => {
                    self.left_click_pending = true;
                }
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x;
                self.mouse_y = position.y;

                if self.mouse_captured {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.cam_yaw += dx * 0.003;
                        self.cam_pitch = (self.cam_pitch - dy * 0.003).clamp(-1.5, 1.5);
                    }
                    self.last_mouse = Some((position.x, position.y));
                }
            }

            WindowEvent::Resized(size) => {
                let w = size.width.max(1);
                let h = size.height.max(1);
                self.rasteriser = SoftwareRasteriser::new(w, h);
                if let Some(backend) = &mut self.backend {
                    backend.resize(w, h);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                // Update
                self.update_camera(dt);
                if self.left_click_pending {
                    self.place_object_at_cursor();
                    self.left_click_pending = false;
                }

                // Render
                let fb = self.render_frame();

                // Present to window (if GPU backend available)
                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&fb.pixels, fb.width, fb.height);
                }

                // FPS counter
                self.frame_count += 1;
                let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if elapsed >= 1.0 {
                    self.fps_display = self.frame_count as f32 / elapsed;
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "Ochroma Engine -- {:.0} FPS | {} splats | pos ({:.0},{:.0},{:.0})",
                            self.fps_display,
                            self.scene_splats.len()
                                + self
                                    .placed_objects
                                    .iter()
                                    .map(|(_, s)| s.len())
                                    .sum::<usize>(),
                            self.cam_pos.x,
                            self.cam_pos.y,
                            self.cam_pos.z
                        ));
                    }
                    self.frame_count = 0;
                    self.fps_timer = now;
                }

                // Request next frame
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    println!("======================================");
    println!("     Ochroma Engine Demo v0.1.0       ");
    println!("  Spectral Gaussian Splatting Engine   ");
    println!("======================================");

    let ply_path = std::env::args().nth(1);
    if let Some(ref path) = ply_path {
        println!("[ochroma] Loading scene from: {}", path);
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = DemoApp::new(ply_path);
    event_loop.run_app(&mut app).expect("Event loop failed");
}
