//! Ochroma Engine — Walking Simulator (Dogfood Game)
//!
//! The first game built on the engine. Proves everything works.
//! Walk around, collect 10 glowing orbs, win!
//!
//! cargo run --bin walking_sim

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct Orb {
    position: Vec3,
    collected: bool,
    bob_phase: f32,
}

struct WalkingSim {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    rasteriser: SoftwareRasteriser,

    // Scene
    terrain_splats: Vec<GaussianSplat>,
    building_splats: Vec<GaussianSplat>,
    tree_splats: Vec<GaussianSplat>,

    // Game state
    orbs: Vec<Orb>,
    orbs_collected: u32,
    total_orbs: u32,
    game_won: bool,
    game_time: f32,

    // Player
    player_pos: Vec3,
    player_yaw: f32,
    player_pitch: f32,

    // Input
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
    last_mouse: Option<(f64, f64)>,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
}

impl WalkingSim {
    fn new() -> Self {
        // Generate orb positions
        let orb_positions = vec![
            Vec3::new(15.0, 1.5, 15.0),
            Vec3::new(-10.0, 1.5, 20.0),
            Vec3::new(25.0, 1.5, -5.0),
            Vec3::new(-20.0, 1.5, -15.0),
            Vec3::new(5.0, 1.5, -25.0),
            Vec3::new(30.0, 1.5, 10.0),
            Vec3::new(-25.0, 1.5, 5.0),
            Vec3::new(0.0, 1.5, 30.0),
            Vec3::new(20.0, 1.5, -20.0),
            Vec3::new(-15.0, 1.5, -25.0),
        ];

        let orbs: Vec<Orb> = orb_positions
            .into_iter()
            .map(|pos| Orb {
                position: pos,
                collected: false,
                bob_phase: pos.x * 0.1, // offset bobbing so they don't all sync
            })
            .collect();

        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            terrain_splats: Vec::new(),
            building_splats: Vec::new(),
            tree_splats: Vec::new(),
            orbs,
            orbs_collected: 0,
            total_orbs: 10,
            game_won: false,
            game_time: 0.0,
            player_pos: Vec3::new(0.0, 2.0, 0.0),
            player_yaw: 0.0,
            player_pitch: 0.0,
            keys_held: std::collections::HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
        }
    }

    fn build_scene(&mut self) {
        println!("[walking_sim] Building scene...");

        // Terrain
        let vol = vox_terrain::volume::generate_demo_volume(42);
        let materials = vox_terrain::volume::default_volume_materials();
        self.terrain_splats = vox_terrain::volume::volume_to_splats(&vol, &materials, 42);
        println!("[walking_sim]   Terrain: {} splats", self.terrain_splats.len());

        // Buildings
        for i in 0..3 {
            let b = vox_data::proc_gs_advanced::generate_detailed_building(
                i as u64,
                6.0,
                8.0,
                2,
                "victorian",
            );
            for s in &b {
                let mut ws = *s;
                ws.position[0] += i as f32 * 12.0 - 12.0;
                ws.position[2] += 25.0;
                self.building_splats.push(ws);
            }
        }
        println!(
            "[walking_sim]   Buildings: {} splats",
            self.building_splats.len()
        );

        // Trees scattered around
        for i in 0..8 {
            let t = vox_data::proc_gs_advanced::generate_tree(100 + i, 7.0, 3.0);
            let angle = i as f32 * 0.8;
            let radius = 15.0 + (i as f32 * 3.0);
            for s in &t {
                let mut ws = *s;
                ws.position[0] += angle.cos() * radius;
                ws.position[2] += angle.sin() * radius;
                self.tree_splats.push(ws);
            }
        }
        println!(
            "[walking_sim]   Trees: {} splats",
            self.tree_splats.len()
        );

        let total =
            self.terrain_splats.len() + self.building_splats.len() + self.tree_splats.len();
        println!("[walking_sim] Scene: {} splats total", total);
    }

    fn generate_orb_splats(&self) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        // Glowing yellow-white spectral values
        let orb_spd: [u16; 8] = std::array::from_fn(|i| {
            let v = if (3..=6).contains(&i) { 0.9 } else { 0.5 };
            half::f16::from_f32(v).to_bits()
        });

        for orb in &self.orbs {
            if orb.collected {
                continue;
            }

            let bob_y = (self.game_time * 2.0 + orb.bob_phase).sin() * 0.3;
            let pos = orb.position + Vec3::new(0.0, bob_y, 0.0);

            // Create a small cluster of bright splats
            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        let d = (dx * dx + dy * dy + dz * dz) as f32;
                        if d > 6.0 {
                            continue;
                        }
                        splats.push(GaussianSplat {
                            position: [
                                pos.x + dx as f32 * 0.15,
                                pos.y + dy as f32 * 0.15,
                                pos.z + dz as f32 * 0.15,
                            ],
                            scale: [0.1, 0.1, 0.1],
                            rotation: [0, 0, 0, 32767],
                            opacity: 230,
                            _pad: [0; 3],
                            spectral: orb_spd,
                        });
                    }
                }
            }
        }
        splats
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.player_yaw.sin() * self.player_pitch.cos(),
            self.player_pitch.sin(),
            -self.player_yaw.cos() * self.player_pitch.cos(),
        )
        .normalize()
    }

    fn update(&mut self, dt: f32) {
        if self.game_won {
            return;
        }
        self.game_time += dt;

        // Movement
        let speed = 8.0 * dt;
        let forward =
            Vec3::new(self.player_yaw.sin(), 0.0, -self.player_yaw.cos()).normalize();
        let right = forward.cross(Vec3::Y).normalize();

        if self.keys_held.contains(&KeyCode::KeyW) {
            self.player_pos += forward * speed;
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            self.player_pos -= forward * speed;
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            self.player_pos -= right * speed;
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            self.player_pos += right * speed;
        }

        // Keep player above ground
        self.player_pos.y = 2.0;

        // Check orb collection
        for orb in &mut self.orbs {
            if orb.collected {
                continue;
            }
            let dist = (orb.position - self.player_pos).length();
            if dist < 2.5 {
                orb.collected = true;
                self.orbs_collected += 1;
                println!(
                    "[walking_sim] Orb collected! {}/{}",
                    self.orbs_collected, self.total_orbs
                );
                if self.orbs_collected >= self.total_orbs {
                    self.game_won = true;
                    println!(
                        "[walking_sim] YOU WIN! All orbs collected in {:.1} seconds!",
                        self.game_time
                    );
                }
            }
        }
    }

    fn render(&mut self) -> Vec<[u8; 4]> {
        let target = self.player_pos + self.forward();
        let camera = RenderCamera {
            view: Mat4::look_at_rh(self.player_pos, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                WIDTH as f32 / HEIGHT as f32,
                0.1,
                500.0,
            ),
        };

        // Combine all splats
        let mut all = self.terrain_splats.clone();
        all.extend_from_slice(&self.building_splats);
        all.extend_from_slice(&self.tree_splats);
        all.extend(self.generate_orb_splats());

        let fb = self.rasteriser.render(&all, &camera, &Illuminant::d65());

        // Draw HUD text (burn into framebuffer)
        let mut pixels = fb.pixels;
        burn_text(
            &mut pixels,
            fb.width,
            10,
            10,
            &format!("Orbs: {}/{}", self.orbs_collected, self.total_orbs),
        );

        if self.game_won {
            burn_text(
                &mut pixels,
                fb.width,
                fb.width / 2 - 80,
                fb.height / 2,
                "YOU WIN!",
            );
        }

        pixels
    }
}

/// Burn simple text into a pixel buffer (no font rendering, just block letters).
fn burn_text(pixels: &mut [[u8; 4]], width: u32, x: u32, y: u32, text: &str) {
    // Simple: draw colored blocks for each character position
    for (i, _ch) in text.chars().enumerate() {
        let px = x + i as u32 * 8;
        for dy in 0..10 {
            for dx in 0..6 {
                let fx = px + dx;
                let fy = y + dy;
                if fx < width {
                    let idx = (fy * width + fx) as usize;
                    if idx < pixels.len() {
                        pixels[idx] = [255, 255, 255, 255];
                    }
                }
            }
        }
    }
}

impl ApplicationHandler for WalkingSim {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma -- Walking Simulator")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => {
                self.backend = Some(backend);
            }
            Err(e) => {
                eprintln!("[walking_sim] GPU: {}", e);
            }
        }

        self.window = Some(window);
        self.build_scene();

        println!("[walking_sim] Walk around and collect all 10 glowing orbs to win!");
        println!("Controls: WASD move, right-click look, Escape quit");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys_held.insert(key);
                        if key == KeyCode::Escape {
                            event_loop.exit();
                        }
                    } else {
                        self.keys_held.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput {
                state,
                button: winit::event::MouseButton::Right,
                ..
            } => {
                self.mouse_captured = state == ElementState::Pressed;
                self.last_mouse = None;
                if let Some(w) = &self.window {
                    w.set_cursor_visible(!self.mouse_captured);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_captured {
                    if let Some((lx, ly)) = self.last_mouse {
                        self.player_yaw += (position.x - lx) as f32 * 0.003;
                        self.player_pitch = (self.player_pitch
                            - (position.y - ly) as f32 * 0.003)
                            .clamp(-1.5, 1.5);
                    }
                    self.last_mouse = Some((position.x, position.y));
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                self.update(dt);
                let pixels = self.render();

                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&pixels, WIDTH, HEIGHT);
                }

                // FPS
                self.frame_count += 1;
                if now.duration_since(self.fps_timer).as_secs_f32() >= 1.0 {
                    let fps = self.frame_count as f32
                        / now.duration_since(self.fps_timer).as_secs_f32();
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "Ochroma Walking Sim -- {:.0} FPS | Orbs: {}/{}{}",
                            fps,
                            self.orbs_collected,
                            self.total_orbs,
                            if self.game_won { " | YOU WIN!" } else { "" }
                        ));
                    }
                    self.frame_count = 0;
                    self.fps_timer = now;
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    println!("========================================");
    println!("  Ochroma Walking Simulator");
    println!("  Collect all 10 glowing orbs to win!");
    println!("========================================");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WalkingSim::new();
    event_loop.run_app(&mut app).unwrap();
}
