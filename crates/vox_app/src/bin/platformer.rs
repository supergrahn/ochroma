//! Ochroma Engine — 3D Platformer
//!
//! The SECOND game built on the engine. Proves it's general-purpose:
//! jump between platforms, reach the gold platform to win.
//!
//! cargo run --bin platformer

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use vox_core::character_controller::{CharacterController, character_controller_tick};
use vox_core::ecs::TransformComponent;
use vox_core::game_ui::{GameState, GameUI, UIElement, UIPosition, UISize};
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_audio::spatial::SpatialAudioManager;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_render::spectral_tonemapper::{tonemap_spectral_framebuffer, ToneMapSettings};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

const PLATFORM_COUNT: usize = 10;
const FALL_THRESHOLD: f32 = -5.0;

// ---------------------------------------------------------------------------
// Platform
// ---------------------------------------------------------------------------

struct Platform {
    position: Vec3,
    size: Vec3, // half-extents
    splats: Vec<GaussianSplat>,
}

impl Platform {
    /// Check if a point (player feet) is standing on this platform.
    fn is_on(&self, pos: Vec3) -> bool {
        let top = self.position.y + self.size.y;
        let feet = pos.y;
        // Must be within horizontal bounds and close to the top surface
        pos.x >= self.position.x - self.size.x
            && pos.x <= self.position.x + self.size.x
            && pos.z >= self.position.z - self.size.z
            && pos.z <= self.position.z + self.size.z
            && feet >= top - 0.3
            && feet <= top + 1.5
    }
}

// ---------------------------------------------------------------------------
// Splat generation helpers
// ---------------------------------------------------------------------------

/// Make a spectral power distribution from approximate RGB.
fn spd_from_rgb(r: f32, g: f32, b: f32) -> [u16; 8] {
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
    std::array::from_fn(|i| half::f16::from_f32(spectral[i]).to_bits())
}

/// Platform colours per index.
fn platform_color(index: usize) -> (f32, f32, f32) {
    match index % 10 {
        0 => (0.7, 0.3, 0.3), // red
        1 => (0.3, 0.3, 0.8), // blue
        2 => (0.3, 0.7, 0.3), // green
        3 => (0.8, 0.8, 0.2), // yellow
        4 => (0.6, 0.2, 0.7), // purple
        5 => (0.2, 0.7, 0.7), // cyan
        6 => (0.8, 0.5, 0.2), // orange
        7 => (0.8, 0.3, 0.5), // pink
        8 => (0.4, 0.6, 0.3), // olive
        _ => (0.9, 0.8, 0.2), // gold (victory platform)
    }
}

/// Generate a flat rectangular platform of splats.
fn generate_platform(index: usize, position: Vec3, size: Vec3) -> Platform {
    let mut splats = Vec::new();
    let (r, g, b) = if index == PLATFORM_COUNT - 1 {
        (0.95, 0.85, 0.15) // gold for victory
    } else {
        platform_color(index)
    };
    let spd = spd_from_rgb(r, g, b);

    // Fill a flat box of splats
    let step = 0.4;
    let nx = (size.x * 2.0 / step).ceil() as i32;
    let nz = (size.z * 2.0 / step).ceil() as i32;
    let ny = (size.y * 2.0 / step).ceil() as i32;

    for ix in 0..=nx {
        for iz in 0..=nz {
            for iy in 0..=ny {
                let x = position.x - size.x + ix as f32 * step;
                let y = position.y - size.y + iy as f32 * step;
                let z = position.z - size.z + iz as f32 * step;
                splats.push(GaussianSplat {
                    position: [x, y, z],
                    scale: [0.2, 0.2, 0.2],
                    rotation: [0, 0, 0, 32767],
                    opacity: 220,
                    _pad: [0; 3],
                    spectral: spd,
                });
            }
        }
    }

    Platform {
        position,
        size,
        splats,
    }
}

/// Generate player splats: a small cube of bright white splats.
fn generate_player_splats(pos: Vec3) -> Vec<GaussianSplat> {
    let mut splats = Vec::new();
    let spd = spd_from_rgb(0.95, 0.95, 1.0); // bright white
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                splats.push(GaussianSplat {
                    position: [
                        pos.x + dx as f32 * 0.15,
                        pos.y + dy as f32 * 0.15,
                        pos.z + dz as f32 * 0.15,
                    ],
                    scale: [0.12, 0.12, 0.12],
                    rotation: [0, 0, 0, 32767],
                    opacity: 240,
                    _pad: [0; 3],
                    spectral: spd,
                });
            }
        }
    }
    splats
}

// ---------------------------------------------------------------------------
// PlatformerGame
// ---------------------------------------------------------------------------

struct PlatformerGame {
    // Window + backend
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    rasteriser: SoftwareRasteriser,

    // Game state
    player: CharacterController,
    player_transform: TransformComponent,
    platforms: Vec<Platform>,
    current_platform: usize,
    spawn_point: Vec3,
    won: bool,

    // Camera follows player
    camera_offset: Vec3,

    // HUD
    ui: GameUI,

    // Audio
    audio: SpatialAudioManager,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    current_fps: f32,

    // Input
    keys_held: std::collections::HashSet<KeyCode>,

    // Spectral pipeline
    spectral_fb: SpectralFramebuffer,
    tonemap_settings: ToneMapSettings,
}

impl PlatformerGame {
    fn new() -> Self {
        // Generate 10 platforms at increasing heights with varying gaps
        let mut platforms = Vec::new();
        let mut rng_x: f32 = 0.0;
        let mut rng_z: f32 = 0.0;
        for i in 0..PLATFORM_COUNT {
            // Pseudorandom offsets based on index
            rng_x += ((i as f32 * 1.7).sin() * 3.0).round();
            rng_z += ((i as f32 * 2.3).cos() * 2.0).round();
            let height = i as f32 * 2.5;

            // First platform is big, rest get smaller
            let size = if i == 0 {
                Vec3::new(3.0, 0.3, 3.0)
            } else if i == PLATFORM_COUNT - 1 {
                Vec3::new(2.5, 0.5, 2.5) // victory platform is slightly bigger & taller
            } else {
                Vec3::new(2.0, 0.3, 2.0)
            };

            let position = Vec3::new(rng_x, height, rng_z);
            platforms.push(generate_platform(i, position, size));
        }

        let spawn_point = Vec3::new(
            platforms[0].position.x,
            platforms[0].position.y + platforms[0].size.y + 1.0,
            platforms[0].position.z,
        );

        let mut player = CharacterController::default();
        player.speed = 6.0;
        player.jump_force = 10.0;
        player.gravity = 22.0;

        let player_transform = TransformComponent {
            position: spawn_point,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };

        let mut ui = GameUI::default();
        ui.game_state = GameState::MainMenu;

        let audio = SpatialAudioManager::new();
        if audio.is_available() {
            println!("[platformer] Spatial audio: available");
        } else {
            println!("[platformer] Spatial audio: silent mode");
        }

        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            player,
            player_transform,
            platforms,
            current_platform: 0,
            spawn_point,
            won: false,
            camera_offset: Vec3::new(0.0, 5.0, 10.0),
            ui,
            audio,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            keys_held: std::collections::HashSet::new(),
            spectral_fb: SpectralFramebuffer::new(WIDTH, HEIGHT),
            tonemap_settings: ToneMapSettings::default(),
        }
    }

    fn respawn(&mut self) {
        self.player_transform.position = self.spawn_point;
        self.player.velocity = Vec3::ZERO;
        self.player.grounded = false;
        self.current_platform = 0;
        println!("[platformer] Respawned at start!");
        // Play a low tone on death
        self.audio.play_tone(200.0, 0.4, 0.6);
    }

    fn update(&mut self, dt: f32) {
        if self.ui.game_state != GameState::Playing {
            return;
        }

        // ---------------------------------------------------------------
        // 1. Player input -> CharacterController
        // ---------------------------------------------------------------
        let mut move_input = Vec3::ZERO;
        if self.keys_held.contains(&KeyCode::KeyW) {
            move_input.z -= 1.0;
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            move_input.z += 1.0;
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            move_input.x -= 1.0;
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            move_input.x += 1.0;
        }
        if move_input.length_squared() > 0.0 {
            move_input = move_input.normalize();
        }

        let jump_pressed = self.keys_held.contains(&KeyCode::Space);

        // Before ticking, check platform grounding
        // Override the default ground detection in character_controller_tick
        let on_platform = self
            .platforms
            .iter()
            .any(|p| p.is_on(self.player_transform.position));

        if on_platform && self.player.velocity.y <= 0.0 {
            self.player.grounded = true;
            // Snap to platform top
            if let Some(p) = self
                .platforms
                .iter()
                .find(|p| p.is_on(self.player_transform.position))
            {
                let top = p.position.y + p.size.y + 0.01;
                if self.player_transform.position.y < top + 0.3 {
                    self.player_transform.position.y = top;
                    if self.player.velocity.y < 0.0 {
                        self.player.velocity.y = 0.0;
                    }
                }
            }
        }

        // Jump sound
        if jump_pressed && self.player.grounded {
            self.audio.play_tone(600.0, 0.15, 0.5);
        }

        character_controller_tick(
            &mut self.player,
            &mut self.player_transform,
            move_input,
            jump_pressed,
            dt,
        );

        // Override the default ground check again post-tick for platform awareness
        // (the built-in check only knows about y=0)
        let on_platform_post = self
            .platforms
            .iter()
            .any(|p| p.is_on(self.player_transform.position));
        if on_platform_post && self.player.velocity.y <= 0.0 {
            self.player.grounded = true;
        }
        // If not on any platform AND above y=0, mark as not grounded
        if !on_platform_post && self.player_transform.position.y > self.player.height * 0.5 + 0.1 {
            self.player.grounded = false;
        }

        // ---------------------------------------------------------------
        // 2. Fall detection
        // ---------------------------------------------------------------
        if self.player_transform.position.y < FALL_THRESHOLD {
            self.respawn();
            return;
        }

        // ---------------------------------------------------------------
        // 3. Platform progress tracking
        // ---------------------------------------------------------------
        for (i, p) in self.platforms.iter().enumerate() {
            if p.is_on(self.player_transform.position) && i > self.current_platform {
                self.current_platform = i;
                println!(
                    "[platformer] Reached platform {}/{}",
                    i + 1,
                    PLATFORM_COUNT
                );
                // Ascending tone for progress
                let freq = 440.0 + i as f32 * 80.0;
                self.audio.play_tone(freq, 0.2, 0.7);
            }
        }

        // ---------------------------------------------------------------
        // 4. Victory check
        // ---------------------------------------------------------------
        if !self.won && self.current_platform == PLATFORM_COUNT - 1 {
            self.won = true;
            println!("[platformer] *** YOU WIN! ***");
            self.audio.play_tone(880.0, 0.5, 0.9);
            self.ui.game_state = GameState::GameOver {
                message: "YOU WIN!".to_string(),
            };
        }

        // ---------------------------------------------------------------
        // 5. Audio listener
        // ---------------------------------------------------------------
        self.audio.set_listener(
            self.player_transform.position,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
        );
        self.audio.tick(dt);

        // ---------------------------------------------------------------
        // 6. Update HUD
        // ---------------------------------------------------------------
        let pos = self.player_transform.position;
        self.ui.set_text(
            "level",
            &format!("Level: 1  |  Platform: {}/{}", self.current_platform + 1, PLATFORM_COUNT),
        );
        self.ui.set_text(
            "pos",
            &format!("X:{:.1} Y:{:.1} Z:{:.1}", pos.x, pos.y, pos.z),
        );
        self.ui
            .set_text("fps", &format!("FPS: {:.0}", self.current_fps));
    }

    fn render(&mut self) -> Vec<[u8; 4]> {
        // Camera follows player from behind and above
        let player_pos = self.player_transform.position;
        let eye = player_pos + self.camera_offset;
        let target = player_pos;

        let camera = RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                WIDTH as f32 / HEIGHT as f32,
                0.1,
                500.0,
            ),
        };

        // Combine all splats
        let mut all: Vec<GaussianSplat> = Vec::new();
        for p in &self.platforms {
            all.extend_from_slice(&p.splats);
        }
        // Player splats at current position
        all.extend(generate_player_splats(player_pos));

        let illuminant = Illuminant::d65();

        // 1. Rasterise
        let fb = self.rasteriser.render(&all, &camera, &illuminant);

        // 2. Write to spectral framebuffer
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
            self.spectral_fb
                .write_sample(x, y, spectral, 1.0, [0.0, 1.0, 0.0], 0, albedo);
        }

        // 3. Tone map
        let mut pixels = tonemap_spectral_framebuffer(
            &self.spectral_fb,
            &illuminant,
            &self.tonemap_settings,
        );

        // 4. Render GameUI HUD
        self.ui.render_to_pixels(&mut pixels, WIDTH, HEIGHT);

        pixels
    }
}

impl ApplicationHandler for PlatformerGame {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma -- 3D Platformer")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => {
                self.backend = Some(backend);
            }
            Err(e) => {
                eprintln!("[platformer] GPU: {}", e);
            }
        }

        self.window = Some(window);

        // Set up HUD elements
        let mut level_el = UIElement::new(
            "level",
            &format!("Level: 1  |  Platform: 1/{}", PLATFORM_COUNT),
            UIPosition::TopLeft,
        );
        level_el.size = UISize::Normal;
        level_el.color = [255, 255, 100];
        self.ui.add_element(level_el);

        let pos = self.player_transform.position;
        let mut pos_el = UIElement::new(
            "pos",
            &format!("X:{:.1} Y:{:.1} Z:{:.1}", pos.x, pos.y, pos.z),
            UIPosition::BottomLeft,
        );
        pos_el.size = UISize::Small;
        pos_el.color = [180, 255, 180];
        self.ui.add_element(pos_el);

        let mut fps_el = UIElement::new("fps", "FPS: --", UIPosition::TopRight);
        fps_el.size = UISize::Small;
        fps_el.color = [180, 220, 255];
        self.ui.add_element(fps_el);

        println!("[platformer] {} platforms generated", self.platforms.len());
        let total_splats: usize = self.platforms.iter().map(|p| p.splats.len()).sum();
        println!("[platformer] Scene: {} platform splats", total_splats);
        println!("[platformer] Press ENTER to start. WASD to move, SPACE to jump.");
        println!("[platformer] Reach the GOLD platform to win!");
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
                        match key {
                            KeyCode::Escape => {
                                match &self.ui.game_state {
                                    GameState::Playing => {
                                        self.ui.game_state = GameState::Paused;
                                    }
                                    GameState::Paused => {
                                        event_loop.exit();
                                    }
                                    _ => event_loop.exit(),
                                }
                            }
                            KeyCode::Enter => {
                                match &self.ui.game_state {
                                    GameState::MainMenu | GameState::Paused => {
                                        self.ui.game_state = GameState::Playing;
                                    }
                                    GameState::GameOver { .. } => {
                                        // Restart
                                        self.won = false;
                                        self.current_platform = 0;
                                        self.player = CharacterController::default();
                                        self.player.speed = 6.0;
                                        self.player.jump_force = 10.0;
                                        self.player.gravity = 22.0;
                                        self.player_transform.position = self.spawn_point;
                                        self.ui.game_state = GameState::Playing;
                                    }
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                    } else {
                        self.keys_held.remove(&key);
                    }
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

                // FPS tracking
                self.frame_count += 1;
                let fps_elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if fps_elapsed >= 1.0 {
                    self.current_fps = self.frame_count as f32 / fps_elapsed;
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "Ochroma Platformer -- {:.0} FPS | Platform: {}/{} | {}",
                            self.current_fps,
                            self.current_platform + 1,
                            PLATFORM_COUNT,
                            match &self.ui.game_state {
                                GameState::Playing => "Playing",
                                GameState::MainMenu => "Main Menu",
                                GameState::Paused => "Paused",
                                GameState::GameOver { .. } => "YOU WIN!",
                            }
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
    println!("  Ochroma 3D Platformer");
    println!("  Jump between platforms to reach GOLD!");
    println!("  Press ENTER to start");
    println!("========================================");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = PlatformerGame::new();
    event_loop.run_app(&mut app).unwrap();
}
