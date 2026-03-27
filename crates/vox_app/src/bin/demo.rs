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
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_render::spectral_tonemapper::{tonemap_spectral_framebuffer, ToneMapOperator, ToneMapSettings};
use vox_render::temporal::TemporalAccumulator;

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

    // Spectral pipeline
    spectral_fb: SpectralFramebuffer,
    temporal: TemporalAccumulator,
    tonemap_settings: ToneMapSettings,
    time_of_day: f32,  // 0-24 hours
    exposure: f32,
}

/// Map time-of-day (0-24) to an illuminant that shifts from warm sunrise
/// through neutral daylight to warm sunset and cool moonlight.
fn illuminant_for_time(hour: f32) -> Illuminant {
    let hour = hour % 24.0;
    // Blend between key illuminants based on time of day:
    //   6  = sunrise  (warm, Illuminant A-ish)
    //  12  = noon     (D65 daylight)
    //  18  = sunset   (warm, D50-ish)
    //   0  = midnight (cool moonlight)
    let d65 = Illuminant::d65();
    let warm = Illuminant::a();
    let cool = Illuminant {
        bands: [30.0, 45.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0],
    };

    let (a, b, t) = if (6.0..12.0).contains(&hour) {
        // Sunrise -> Noon
        (&warm, &d65, (hour - 6.0) / 6.0)
    } else if (12.0..18.0).contains(&hour) {
        // Noon -> Sunset
        (&d65, &warm, (hour - 12.0) / 6.0)
    } else if hour >= 18.0 {
        // Sunset -> Midnight
        let t = (hour - 18.0) / 6.0;
        (&warm, &cool, t)
    } else {
        // Midnight -> Sunrise
        (&cool, &warm, hour / 6.0)
    };

    let mut bands = [0.0f32; 8];
    for i in 0..8 {
        bands[i] = a.bands[i] * (1.0 - t) + b.bands[i] * t;
    }
    Illuminant { bands }
}

fn tonemap_operator_name(op: ToneMapOperator) -> &'static str {
    match op {
        ToneMapOperator::None => "Linear",
        ToneMapOperator::ACES => "ACES",
        ToneMapOperator::Reinhard => "Reinhard",
        ToneMapOperator::Filmic => "Filmic",
    }
}

fn next_tonemap_operator(op: ToneMapOperator) -> ToneMapOperator {
    match op {
        ToneMapOperator::None => ToneMapOperator::ACES,
        ToneMapOperator::ACES => ToneMapOperator::Reinhard,
        ToneMapOperator::Reinhard => ToneMapOperator::Filmic,
        ToneMapOperator::Filmic => ToneMapOperator::None,
    }
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
            spectral_fb: SpectralFramebuffer::new(WIDTH, HEIGHT),
            temporal: TemporalAccumulator::new(WIDTH, HEIGHT),
            tonemap_settings: ToneMapSettings::default(),
            time_of_day: 12.0, // Start at noon
            exposure: 1.0,
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

    fn render_frame(&mut self) -> Vec<[u8; 4]> {
        let forward = self.camera_forward();
        let target = self.cam_pos + forward;
        let w = self.rasteriser.width;
        let h = self.rasteriser.height;

        let camera = RenderCamera {
            view: Mat4::look_at_rh(self.cam_pos, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                w as f32 / h as f32,
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

        // Time-of-day illuminant
        let illuminant = illuminant_for_time(self.time_of_day);

        // 1. Render with software rasteriser (returns RGBA u8 framebuffer)
        let fb = self.rasteriser.render(&all_splats, &camera, &illuminant);

        // 2. Write to spectral framebuffer (approximate: convert RGB back to spectral bands)
        self.spectral_fb.clear();
        for (i, pixel) in fb.pixels.iter().enumerate() {
            let x = (i % w as usize) as u32;
            let y = (i / w as usize) as u32;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Approximate spectral distribution from RGB.
            // Maps RGB to 8 bands (380-660nm) using overlap model:
            //   380nm = violet (blue)
            //   420nm = blue
            //   460nm = cyan (blue+green)
            //   500nm = green-blue
            //   540nm = green
            //   580nm = yellow (green+red)
            //   620nm = orange-red
            //   660nm = red
            let spectral = [
                b * 0.3,                    // 380nm: violet
                b * 0.7,                    // 420nm: blue
                b * 0.8 + g * 0.1,          // 460nm: cyan-blue
                g * 0.4 + b * 0.2,          // 500nm: green-blue
                g * 0.9 + r * 0.05,         // 540nm: green
                r * 0.4 + g * 0.3,          // 580nm: yellow
                r * 0.8 + g * 0.05,         // 620nm: orange-red
                r * 0.6,                    // 660nm: red
            ];
            let albedo = spectral; // Approximate: albedo same as spectral for this path

            self.spectral_fb.write_sample(
                x, y,
                spectral,
                1.0,               // depth (approximate, not per-pixel from rasteriser)
                [0.0, 1.0, 0.0],  // normal (up)
                0,                 // object_id
                albedo,
            );
        }

        // 3. Temporal accumulation: blend with previous frame
        self.temporal.accumulate(&self.spectral_fb);
        self.temporal.write_to_framebuffer(&mut self.spectral_fb);

        // 4. Tone map spectral framebuffer to RGBA8 for display
        self.tonemap_settings.exposure = self.exposure;
        tonemap_spectral_framebuffer(&self.spectral_fb, &illuminant, &self.tonemap_settings)
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

    fn total_splat_count(&self) -> usize {
        self.scene_splats.len()
            + self.placed_objects.iter().map(|(_, s)| s.len()).sum::<usize>()
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
        println!("  T           -- advance time of day (+1 hour)");
        println!("  +/-         -- adjust exposure");
        println!("  M           -- cycle tone map operator");
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
                                let pixels = self.render_frame();
                                let dir = std::env::temp_dir().join("ochroma_visual");
                                std::fs::create_dir_all(&dir).ok();
                                let path = dir.join("screenshot.ppm");
                                let w = self.rasteriser.width;
                                let h = self.rasteriser.height;
                                let mut data =
                                    format!("P6\n{} {}\n255\n", w, h).into_bytes();
                                for p in &pixels {
                                    data.push(p[0]);
                                    data.push(p[1]);
                                    data.push(p[2]);
                                }
                                std::fs::write(&path, &data).ok();
                                println!("[ochroma] Screenshot: {}", path.display());
                            }
                            KeyCode::KeyT => {
                                self.time_of_day = (self.time_of_day + 1.0) % 24.0;
                                self.temporal.reset(); // Reset accumulation on lighting change
                                println!(
                                    "[ochroma] Time of day: {:.0}:00",
                                    self.time_of_day
                                );
                            }
                            KeyCode::Equal => {
                                // + key (=/+)
                                self.exposure = (self.exposure * 1.2).min(16.0);
                                println!("[ochroma] Exposure: {:.2}", self.exposure);
                            }
                            KeyCode::Minus => {
                                self.exposure = (self.exposure / 1.2).max(0.05);
                                println!("[ochroma] Exposure: {:.2}", self.exposure);
                            }
                            KeyCode::KeyM => {
                                self.tonemap_settings.operator =
                                    next_tonemap_operator(self.tonemap_settings.operator);
                                println!(
                                    "[ochroma] Tone map: {}",
                                    tonemap_operator_name(self.tonemap_settings.operator)
                                );
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
                self.spectral_fb = SpectralFramebuffer::new(w, h);
                self.temporal.resize(w, h);
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

                // Render through spectral pipeline
                let pixels = self.render_frame();

                // Present to window (if GPU backend available)
                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&pixels, self.rasteriser.width, self.rasteriser.height);
                }

                // FPS counter + title update
                self.frame_count += 1;
                let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if elapsed >= 1.0 {
                    self.fps_display = self.frame_count as f32 / elapsed;
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "Ochroma -- {:.0} FPS | {} splats | {:.0}:00 | EV {:.2} | {}",
                            self.fps_display,
                            self.total_splat_count(),
                            self.time_of_day,
                            self.exposure,
                            tonemap_operator_name(self.tonemap_settings.operator),
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
