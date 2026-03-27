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
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_render::spectral_tonemapper::{tonemap_spectral_framebuffer, ToneMapSettings};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

// ---------------------------------------------------------------------------
// Bitmap font (5x7 pixel glyphs)
// ---------------------------------------------------------------------------

const CHAR_WIDTH: u32 = 6;
#[allow(dead_code)]
const CHAR_HEIGHT: u32 = 8;

fn char_bitmap(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        'A' | 'a' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' | 'b' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' | 'c' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' | 'd' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' | 'e' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' | 'f' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' | 'g' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' | 'h' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' | 'i' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' | 'j' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' | 'k' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' | 'l' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' | 'm' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' | 'n' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' | 'o' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' | 'p' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' | 'r' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' | 's' => [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
        'T' | 't' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' | 'u' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' | 'v' => [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
        'W' | 'w' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' | 'x' => [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001],
        'Y' | 'y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' | 'z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
        ' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

fn burn_text(pixels: &mut [[u8; 4]], width: u32, x: u32, y: u32, text: &str, color: [u8; 3]) {
    for (ci, ch) in text.chars().enumerate() {
        let bitmap = char_bitmap(ch);
        let base_x = x + ci as u32 * CHAR_WIDTH;
        for (row, &bits) in bitmap.iter().enumerate() {
            for col in 0..5u32 {
                if bits & (1 << (4 - col)) != 0 {
                    let px = base_x + col;
                    let py = y + row as u32;
                    if px < width {
                        let idx = (py * width + px) as usize;
                        if idx < pixels.len() {
                            pixels[idx] = [color[0], color[1], color[2], 255];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AABB collision
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BoundingBox {
    min: Vec3,
    max: Vec3,
}

fn check_building_collision(player_pos: &mut Vec3, buildings: &[BoundingBox]) {
    let radius = 1.0;
    for bb in buildings {
        if player_pos.x > bb.min.x - radius
            && player_pos.x < bb.max.x + radius
            && player_pos.z > bb.min.z - radius
            && player_pos.z < bb.max.z + radius
            && player_pos.y < bb.max.y
        {
            // Push player out via nearest face
            let dx_min = (player_pos.x - (bb.min.x - radius)).abs();
            let dx_max = (player_pos.x - (bb.max.x + radius)).abs();
            let dz_min = (player_pos.z - (bb.min.z - radius)).abs();
            let dz_max = (player_pos.z - (bb.max.z + radius)).abs();

            let min_d = dx_min.min(dx_max).min(dz_min).min(dz_max);
            if min_d == dx_min {
                player_pos.x = bb.min.x - radius;
            } else if min_d == dx_max {
                player_pos.x = bb.max.x + radius;
            } else if min_d == dz_min {
                player_pos.z = bb.min.z - radius;
            } else {
                player_pos.z = bb.max.z + radius;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Game structs
// ---------------------------------------------------------------------------

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
    building_boxes: Vec<BoundingBox>,

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

    // Spectral pipeline
    spectral_fb: SpectralFramebuffer,
    tonemap_settings: ToneMapSettings,
}

impl WalkingSim {
    fn new() -> Self {
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
                bob_phase: pos.x * 0.1,
            })
            .collect();

        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            terrain_splats: Vec::new(),
            building_splats: Vec::new(),
            tree_splats: Vec::new(),
            building_boxes: Vec::new(),
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
            spectral_fb: SpectralFramebuffer::new(WIDTH, HEIGHT),
            tonemap_settings: ToneMapSettings::default(),
        }
    }

    fn build_scene(&mut self) {
        println!("[walking_sim] Building scene...");

        // Terrain
        let vol = vox_terrain::volume::generate_demo_volume(42);
        let materials = vox_terrain::volume::default_volume_materials();
        self.terrain_splats = vox_terrain::volume::volume_to_splats(&vol, &materials, 42);
        println!(
            "[walking_sim]   Terrain: {} splats",
            self.terrain_splats.len()
        );

        // Buildings (with bounding boxes for collision)
        let building_width = 6.0;
        let building_depth = 8.0;
        let building_height = 10.0; // approximate visual height
        for i in 0..3 {
            let bx = i as f32 * 12.0 - 12.0;
            let bz = 25.0;
            let b = vox_data::proc_gs_advanced::generate_detailed_building(
                i as u64,
                building_width,
                building_depth,
                2,
                "victorian",
            );
            for s in &b {
                let mut ws = *s;
                ws.position[0] += bx;
                ws.position[2] += bz;
                self.building_splats.push(ws);
            }
            // Store AABB for collision
            self.building_boxes.push(BoundingBox {
                min: Vec3::new(
                    bx - building_width * 0.5,
                    0.0,
                    bz - building_depth * 0.5,
                ),
                max: Vec3::new(
                    bx + building_width * 0.5,
                    building_height,
                    bz + building_depth * 0.5,
                ),
            });
        }
        println!(
            "[walking_sim]   Buildings: {} splats ({} collision boxes)",
            self.building_splats.len(),
            self.building_boxes.len(),
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

            let rotation_angle = self.game_time * 1.5 + orb.bob_phase;
            let cos_a = rotation_angle.cos();
            let sin_a = rotation_angle.sin();

            let pulse = 1.0 + (self.game_time * 3.0 + orb.bob_phase).sin() * 0.2;
            let scale = 0.1 * pulse;

            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        let d = (dx * dx + dy * dy + dz * dz) as f32;
                        if d > 6.0 {
                            continue;
                        }
                        let rx = dx as f32 * 0.15 * cos_a - dz as f32 * 0.15 * sin_a;
                        let rz = dx as f32 * 0.15 * sin_a + dz as f32 * 0.15 * cos_a;
                        splats.push(GaussianSplat {
                            position: [pos.x + rx, pos.y + dy as f32 * 0.15, pos.z + rz],
                            scale: [scale, scale, scale],
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

        // Building collision: push player out of any overlapping building
        check_building_collision(&mut self.player_pos, &self.building_boxes);

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

                // Generate collect sound WAV (proves audio pipeline works)
                let sound = vox_audio::synth::generate_collect_sound();
                let path = std::env::temp_dir()
                    .join(format!("ochroma_collect_{}.wav", self.orbs_collected));
                match vox_audio::synth::save_wav(&sound, 44100, &path) {
                    Ok(()) => println!("[walking_sim] Sound: {}", path.display()),
                    Err(e) => eprintln!("[walking_sim] Sound save failed: {}", e),
                }

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

        let illuminant = Illuminant::d65();

        // 1. Rasterise
        let fb = self.rasteriser.render(&all, &camera, &illuminant);

        // 2. Write to spectral framebuffer (approximate: RGB -> spectral bands)
        self.spectral_fb.clear();
        for (i, pixel) in fb.pixels.iter().enumerate() {
            let x = (i % WIDTH as usize) as u32;
            let y = (i / WIDTH as usize) as u32;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let spectral = [
                b * 0.3,
                b * 0.7,
                b * 0.8 + g * 0.1,
                g * 0.4 + b * 0.2,
                g * 0.9 + r * 0.05,
                r * 0.4 + g * 0.3,
                r * 0.8 + g * 0.05,
                r * 0.6,
            ];
            let albedo = spectral;

            self.spectral_fb.write_sample(
                x, y, spectral, 1.0, [0.0, 1.0, 0.0], 0, albedo,
            );
        }

        // 3. Tone map spectral framebuffer to RGBA8
        let mut pixels = tonemap_spectral_framebuffer(
            &self.spectral_fb,
            &illuminant,
            &self.tonemap_settings,
        );

        // 4. Draw HUD text (burn into framebuffer)
        let orb_text = format!(
            "ORBS: {}/{}",
            self.orbs_collected, self.total_orbs
        );
        burn_text(&mut pixels, WIDTH, 10, 10, &orb_text, [255, 255, 255]);

        if self.game_won {
            // Center "YOU WIN!" in yellow
            let text = "YOU WIN!";
            let text_w = text.len() as u32 * CHAR_WIDTH;
            let cx = WIDTH / 2 - text_w / 2;
            burn_text(&mut pixels, WIDTH, cx, HEIGHT / 2, text, [255, 255, 0]);
        }

        pixels
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
