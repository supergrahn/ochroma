//! Ochroma Engine — Walking Simulator (Dogfood Game)
//!
//! The first game built on the engine. Proves everything works.
//! Walk around, collect 10 glowing orbs, win!
//!
//! cargo run --bin walking_sim

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
use vox_render::clas;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::rigid_animation::RigidClip;
use vox_render::shadows::ShadowMapper;
use vox_render::spectral::RenderCamera;
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_render::spectral_tonemapper::{tonemap_spectral_framebuffer, ToneMapSettings};
use vox_physics::rapier::RapierPhysicsWorld;
use vox_script::rhai_runtime::RhaiRuntime;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

// Sun direction (normalized, pointing toward ground = positive Y component negative)
const SUN_DIR: Vec3 = Vec3::new(0.4, -0.8, 0.3);

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

/// Windmill entity — base splats + blade splats with rigid animation.
struct Windmill {
    base_splats: Vec<GaussianSplat>,
    blade_splats_local: Vec<GaussianSplat>,   // splats in local space (around origin)
    blade_splats_world: Vec<GaussianSplat>,   // splats after applying animated rotation
    position: Vec3,
    clip: RigidClip,
    anim_time: f32,
}

impl Windmill {
    fn new(position: Vec3) -> Self {
        // Build base: a column of splats
        let mut base_splats = Vec::new();
        let base_spd: [u16; 8] = std::array::from_fn(|i| {
            let v: f32 = if i < 4 { 0.6 } else { 0.3 };
            half::f16::from_f32(v).to_bits()
        });
        for iy in 0..8 {
            let y = iy as f32 * 0.5;
            base_splats.push(GaussianSplat {
                position: [position.x, position.y + y, position.z],
                scale: [0.25, 0.25, 0.25],
                rotation: [0, 0, 0, 32767],
                opacity: 200,
                _pad: [0; 3],
                spectral: base_spd,
            });
        }

        // Build blade splats in local space (4 blades radiating outward)
        let mut blade_splats_local = Vec::new();
        let blade_height = 4.0; // attach blades at this height on the base
        let blade_spd: [u16; 8] = std::array::from_fn(|i| {
            let v: f32 = if (2..=5).contains(&i) { 0.85 } else { 0.4 };
            half::f16::from_f32(v).to_bits()
        });

        for blade in 0..4 {
            let angle = blade as f32 * std::f32::consts::FRAC_PI_2;
            // Each blade is a row of splats along one radial direction
            for r in 1..=5 {
                let radius = r as f32 * 0.4;
                blade_splats_local.push(GaussianSplat {
                    position: [radius * angle.cos(), 0.0, radius * angle.sin()],
                    scale: [0.18, 0.18, 0.18],
                    rotation: [0, 0, 0, 32767],
                    opacity: 210,
                    _pad: [0; 3],
                    spectral: blade_spd,
                });
            }
        }

        let blade_splats_world = blade_splats_local.clone();

        // Spin at 0.4 rotations/second around Y axis
        let clip = RigidClip::rotation_loop(Vec3::Y, 0.4);

        Windmill {
            base_splats,
            blade_splats_local,
            blade_splats_world,
            position: position + Vec3::new(0.0, blade_height, 0.0),
            clip,
            anim_time: 0.0,
        }
    }

    /// Advance animation and recompute world-space blade splats.
    fn tick(&mut self, dt: f32) {
        self.anim_time += dt;
        let kf = self.clip.sample(self.anim_time);
        let rot = kf.rotation;

        // Apply rotation + translation to each blade splat
        for (src, dst) in self
            .blade_splats_local
            .iter()
            .zip(self.blade_splats_world.iter_mut())
        {
            let local_pos = Vec3::from(src.position);
            let rotated = rot * local_pos;
            let world_pos = rotated + self.position;
            *dst = GaussianSplat {
                position: [world_pos.x, world_pos.y, world_pos.z],
                ..*src
            };
        }
    }
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
    game_time: f32,

    // Character controller (replaces ad-hoc player state)
    cc: CharacterController,
    cc_transform: TransformComponent,
    player_yaw: f32,
    player_pitch: f32,

    // Game UI
    game_ui: GameUI,
    fps_display: f32,

    // Spatial audio
    audio: SpatialAudioManager,

    // Shadow mapper
    shadow_mapper: ShadowMapper,

    // Windmill
    windmill: Windmill,

    // Input
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
    last_mouse: Option<(f64, f64)>,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    current_fps: f32,

    // Spectral pipeline
    spectral_fb: SpectralFramebuffer,
    tonemap_settings: ToneMapSettings,

    // CLAS clustering stats
    clas_cluster_count: usize,
    clas_bvh_depth: u32,

    // Scripting
    rhai: RhaiRuntime,

    // Rapier physics world
    physics: RapierPhysicsWorld,
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

        // Character controller — start at eye level above ground
        let mut cc = CharacterController::default();
        cc.speed = 8.0;
        let cc_transform = TransformComponent {
            position: Vec3::new(0.0, cc.height * 0.5, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };

        // Game UI — start on main menu; player presses Enter to begin
        let mut game_ui = GameUI::default();
        game_ui.game_state = GameState::MainMenu;

        // Spatial audio — gracefully silent if no hardware
        let audio = SpatialAudioManager::new();
        if audio.is_available() {
            println!("[walking_sim] Spatial audio: available");
        } else {
            println!("[walking_sim] Spatial audio: silent mode (no hardware or rodio feature)");
        }

        // Shadow mapper — small resolution for software performance
        let shadow_mapper = ShadowMapper::new(128);

        // Windmill placed at the side of the scene
        let windmill = Windmill::new(Vec3::new(18.0, 0.0, -8.0));

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
            game_time: 0.0,
            cc,
            cc_transform,
            player_yaw: 0.0,
            player_pitch: 0.0,
            game_ui,
            fps_display: 0.0,
            audio,
            shadow_mapper,
            windmill,
            keys_held: std::collections::HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            spectral_fb: SpectralFramebuffer::new(WIDTH, HEIGHT),
            tonemap_settings: ToneMapSettings::default(),
            clas_cluster_count: 0,
            clas_bvh_depth: 0,
            rhai: RhaiRuntime::new(),
            physics: {
                let mut phys = RapierPhysicsWorld::new();
                // Ground plane at y=0
                phys.add_static_collider([0.0, -0.5, 0.0], [500.0, 0.5, 500.0]);
                println!("[walking_sim] Physics: Rapier3D world initialised (ground plane)");
                phys
            },
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
        let building_height = 10.0;
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
            // Register building in Rapier physics world
            self.physics.add_static_collider(
                [bx, building_height * 0.5, bz],
                [building_width * 0.5, building_height * 0.5, building_depth * 0.5],
            );
        }
        println!(
            "[walking_sim]   Buildings: {} splats ({} collision boxes, {} Rapier colliders)",
            self.building_splats.len(),
            self.building_boxes.len(),
            self.physics.collider_count(),
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

        // CLAS clustering
        let mut all_splats = self.terrain_splats.clone();
        all_splats.extend_from_slice(&self.building_splats);
        all_splats.extend_from_slice(&self.tree_splats);
        let clusters = clas::build_clusters(&all_splats, 128);
        let bvh = clas::build_cluster_bvh(&clusters);
        let stats = clas::compute_stats(&clusters, &bvh);
        self.clas_cluster_count = stats.cluster_count;
        self.clas_bvh_depth = stats.bvh_depth;
        println!(
            "[walking_sim] CLAS: {} clusters, BVH depth {}, avg {:.0} splats/cluster",
            stats.cluster_count, stats.bvh_depth, stats.avg_splats_per_cluster,
        );

        // Prime the shadow mapper with the initial camera state
        let sun_dir = SUN_DIR.normalize();
        self.shadow_mapper.update(
            self.cc_transform.position,
            Vec3::new(self.player_yaw.sin(), 0.0, -self.player_yaw.cos()).normalize(),
            sun_dir,
        );
        // Render shadow map from building splat positions
        let occluder_positions: Vec<Vec3> = self
            .building_splats
            .iter()
            .map(|s| Vec3::from(s.position))
            .collect();
        let occluder_radii: Vec<f32> = self
            .building_splats
            .iter()
            .map(|s| s.scale[0])
            .collect();
        self.shadow_mapper
            .render_shadow_map(&occluder_positions, &occluder_radii);
        println!("[walking_sim] Shadow mapper primed.");
    }

    fn player_pos(&self) -> Vec3 {
        self.cc_transform.position
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.player_yaw.sin() * self.player_pitch.cos(),
            self.player_pitch.sin(),
            -self.player_yaw.cos() * self.player_pitch.cos(),
        )
        .normalize()
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

    fn update(&mut self, dt: f32) {
        // Only update game logic when Playing
        if self.game_ui.game_state != GameState::Playing {
            return;
        }
        self.game_time += dt;

        // ---------------------------------------------------------------
        // 1. CharacterController movement (replaces ad-hoc WASD code)
        // ---------------------------------------------------------------
        let forward_xz =
            Vec3::new(self.player_yaw.sin(), 0.0, -self.player_yaw.cos()).normalize();
        let right_xz = forward_xz.cross(Vec3::Y).normalize();

        let mut move_input = Vec3::ZERO;
        if self.keys_held.contains(&KeyCode::KeyW) {
            move_input += forward_xz;
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            move_input -= forward_xz;
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            move_input -= right_xz;
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            move_input += right_xz;
        }
        if move_input.length_squared() > 0.0 {
            move_input = move_input.normalize();
        }

        let jump_pressed = self.keys_held.contains(&KeyCode::Space);

        character_controller_tick(
            &mut self.cc,
            &mut self.cc_transform,
            move_input,
            jump_pressed,
            dt,
        );

        // Step Rapier physics world
        self.physics.step();

        // Use Rapier ground raycast for ground detection: cast ray downward from feet
        let feet_y = self.cc_transform.position.y - self.cc.height * 0.5;
        if let Some((_hit_pos, dist)) = self.physics.raycast(
            [self.cc_transform.position.x, feet_y + 0.1, self.cc_transform.position.z],
            [0.0, -1.0, 0.0],
            0.3, // short ray: just checking if ground is within 0.3m below feet
        ) {
            // Ground detected by Rapier — snap to ground if falling
            if self.cc.velocity.y <= 0.0 {
                let ground_y = feet_y + 0.1 - dist;
                self.cc_transform.position.y = ground_y + self.cc.height * 0.5;
                self.cc.grounded = true;
                if self.cc.velocity.y < 0.0 {
                    self.cc.velocity.y = 0.0;
                }
            }
        }

        // Building collision still applied post-integration (AABB fallback alongside Rapier)
        check_building_collision(&mut self.cc_transform.position, &self.building_boxes);

        // ---------------------------------------------------------------
        // 2. Shadow mapper update each frame
        // ---------------------------------------------------------------
        let sun_dir = SUN_DIR.normalize();
        let cam_fwd = self.forward();
        self.shadow_mapper
            .update(self.cc_transform.position, cam_fwd, sun_dir);

        // ---------------------------------------------------------------
        // 3. Windmill animation tick
        // ---------------------------------------------------------------
        self.windmill.tick(dt);

        // ---------------------------------------------------------------
        // 4. Spatial audio listener update
        // ---------------------------------------------------------------
        self.audio
            .set_listener(self.cc_transform.position, self.forward(), Vec3::Y);
        self.audio.tick(dt);

        // ---------------------------------------------------------------
        // 5. Orb collection
        // ---------------------------------------------------------------
        let player_pos = self.cc_transform.position;
        for orb in &mut self.orbs {
            if orb.collected {
                continue;
            }
            let dist = (orb.position - player_pos).length();
            if dist < 2.5 {
                orb.collected = true;
                self.orbs_collected += 1;
                println!(
                    "[walking_sim] Orb collected! {}/{}",
                    self.orbs_collected, self.total_orbs
                );

                // SpatialAudio: play a tone on collect
                // Each successive orb plays a slightly higher frequency
                let freq = 440.0 + self.orbs_collected as f32 * 110.0;
                self.audio.play_tone(freq, 0.3, 0.8);

                // Also save a WAV for proof (legacy path)
                let sound = vox_audio::synth::generate_collect_sound();
                let path = std::env::temp_dir()
                    .join(format!("ochroma_collect_{}.wav", self.orbs_collected));
                match vox_audio::synth::save_wav(&sound, 44100, &path) {
                    Ok(()) => println!("[walking_sim] Sound: {}", path.display()),
                    Err(e) => eprintln!("[walking_sim] Sound save failed: {}", e),
                }

                if self.orbs_collected >= self.total_orbs {
                    println!(
                        "[walking_sim] YOU WIN! All orbs collected in {:.1} seconds!",
                        self.game_time
                    );
                    self.game_ui.game_state = GameState::GameOver {
                        message: "YOU WIN!".to_string(),
                    };
                }
            }
        }

        // ---------------------------------------------------------------
        // 6. Update HUD elements with current values
        // ---------------------------------------------------------------
        let pos = self.cc_transform.position;
        self.game_ui.set_text(
            "orbs",
            &format!("ORBS: {}/{}", self.orbs_collected, self.total_orbs),
        );
        self.game_ui.set_text(
            "pos",
            &format!("X:{:.0} Y:{:.1} Z:{:.0}", pos.x, pos.y, pos.z),
        );
        self.game_ui
            .set_text("fps", &format!("FPS: {:.0}", self.current_fps));
    }

    fn render(&mut self) -> Vec<[u8; 4]> {
        let eye = self.player_pos();
        let target = eye + self.forward();
        let camera = RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                WIDTH as f32 / HEIGHT as f32,
                0.1,
                500.0,
            ),
        };

        // Combine all scene splats
        let mut all = self.terrain_splats.clone();
        all.extend_from_slice(&self.building_splats);
        all.extend_from_slice(&self.tree_splats);
        all.extend(self.generate_orb_splats());
        // Windmill base + animated blades
        all.extend_from_slice(&self.windmill.base_splats);
        all.extend_from_slice(&self.windmill.blade_splats_world);

        let illuminant = Illuminant::d65();

        // 1. Rasterise
        let fb = self.rasteriser.render(&all, &camera, &illuminant);

        // 2. Write to spectral framebuffer with shadow darkening on terrain
        self.spectral_fb.clear();
        for (i, pixel) in fb.pixels.iter().enumerate() {
            let x = (i % WIDTH as usize) as u32;
            let y = (i / WIDTH as usize) as u32;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Shadow: check if this terrain pixel's world position is shadowed.
            // We reconstruct an approximate world position from pixel coordinates
            // via the camera's inverse projection (simple ground-plane approximation).
            // For a software renderer this is an approximation — good enough for a demo.
            let shadow_factor = {
                // Map screen pixel to NDC
                let ndc_x = (x as f32 / WIDTH as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (y as f32 / HEIGHT as f32) * 2.0;

                // Only check pixels in the lower half of the screen (ground region heuristic)
                if ndc_y < -0.1 {
                    // Use a world-space grid probe at estimated ground position
                    let ray_dir_view = Vec3::new(
                        ndc_x / camera.proj.col(0)[0],
                        ndc_y / camera.proj.col(1)[1],
                        -1.0,
                    )
                    .normalize();
                    // Transform ray direction to world space
                    let view_inv = camera.view.inverse();
                    let world_dir =
                        (view_inv * ray_dir_view.extend(0.0)).truncate().normalize();

                    // Intersect with y=0 ground plane
                    let t = if world_dir.y.abs() > 1e-4 {
                        -eye.y / world_dir.y
                    } else {
                        -1.0
                    };
                    if t > 0.0 && t < 200.0 {
                        let world_pos = eye + world_dir * t;
                        if self.shadow_mapper.is_in_shadow(world_pos, 0.01) {
                            0.6 // darken shadowed terrain by 40%
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            };

            let r = r * shadow_factor;
            let g = g * shadow_factor;
            let b = b * shadow_factor;

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

        // 3. Tone map spectral framebuffer to RGBA8
        let mut pixels = tonemap_spectral_framebuffer(
            &self.spectral_fb,
            &illuminant,
            &self.tonemap_settings,
        );

        // 4. Render GameUI HUD (orbs, position, fps, game-over overlay)
        self.game_ui.render_to_pixels(&mut pixels, WIDTH, HEIGHT);

        pixels
    }
}

impl ApplicationHandler for WalkingSim {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma -- Walking Simulator")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));

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

        // Set up GameUI HUD elements (shown when Playing)
        let mut orb_el = UIElement::new("orbs", "ORBS: 0/10", UIPosition::TopLeft);
        orb_el.size = UISize::Normal;
        orb_el.color = [255, 255, 100];
        self.game_ui.add_element(orb_el);

        let pos = self.cc_transform.position;
        let mut pos_el = UIElement::new(
            "pos",
            &format!("X:{:.0} Y:{:.1} Z:{:.0}", pos.x, pos.y, pos.z),
            UIPosition::BottomLeft,
        );
        pos_el.size = UISize::Small;
        pos_el.color = [180, 255, 180];
        self.game_ui.add_element(pos_el);

        let mut fps_el = UIElement::new("fps", "FPS: --", UIPosition::TopRight);
        fps_el.size = UISize::Small;
        fps_el.color = [180, 220, 255];
        self.game_ui.add_element(fps_el);

        // Load game config via Rhai scripting
        let config_script = r#"
            let orb_count = 10;
            let player_speed = 8.0;
            let collect_distance = 2.5;
            let orb_bob_speed = 2.0;
            let orb_pulse_speed = 3.0;
            log("Game config loaded via Rhai!");
            orb_count
        "#;
        match self.rhai.load_script("config", config_script) {
            Ok(idx) => {
                if let Ok(()) = self.rhai.run(idx) {
                    println!("[walking_sim] Rhai config loaded");
                }
            }
            Err(e) => eprintln!("[walking_sim] Rhai error: {}", e),
        }

        match self.rhai.eval("2 + 2") {
            Ok(result) => println!("[walking_sim] Rhai eval test: 2 + 2 = {}", result),
            Err(e) => eprintln!("[walking_sim] Rhai eval error: {}", e),
        }

        println!("[walking_sim] Walk around and collect all 10 glowing orbs to win!");
        println!("Controls: ENTER start, WASD move, SPACE jump, right-click look, ` Rhai eval, Escape quit");
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
                                // If playing, pause; if paused, quit (or resume on second press)
                                match &self.game_ui.game_state {
                                    GameState::Playing => {
                                        self.game_ui.game_state = GameState::Paused;
                                    }
                                    GameState::Paused => {
                                        event_loop.exit();
                                    }
                                    _ => event_loop.exit(),
                                }
                            }
                            KeyCode::Enter => {
                                match &self.game_ui.game_state {
                                    GameState::MainMenu | GameState::Paused => {
                                        self.game_ui.game_state = GameState::Playing;
                                    }
                                    GameState::GameOver { .. } => {
                                        // Restart: reset orbs, player position, game time
                                        for orb in &mut self.orbs {
                                            orb.collected = false;
                                        }
                                        self.orbs_collected = 0;
                                        self.game_time = 0.0;
                                        self.cc = CharacterController::default();
                                        self.cc.speed = 8.0;
                                        self.cc_transform.position =
                                            Vec3::new(0.0, self.cc.height * 0.5, 0.0);
                                        self.game_ui.game_state = GameState::Playing;
                                    }
                                    _ => {}
                                }
                            }
                            KeyCode::Backquote => {
                                let expr = "42 * 2 + 1";
                                match self.rhai.eval(expr) {
                                    Ok(result) => {
                                        println!("[rhai-console] {} = {}", expr, result)
                                    }
                                    Err(e) => eprintln!("[rhai-console] Error: {}", e),
                                }
                            }
                            _ => {}
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

                // FPS tracking
                self.frame_count += 1;
                let fps_elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if fps_elapsed >= 1.0 {
                    self.current_fps = self.frame_count as f32 / fps_elapsed;
                    self.fps_display = self.current_fps;
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "Ochroma Walking Sim -- {:.0} FPS | Orbs: {}/{} | CLAS: {} clusters | {}",
                            self.current_fps,
                            self.orbs_collected,
                            self.total_orbs,
                            self.clas_cluster_count,
                            match &self.game_ui.game_state {
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
    println!("  Ochroma Walking Simulator");
    println!("  Collect all 10 glowing orbs to win!");
    println!("  Press ENTER on the main menu to start");
    println!("========================================");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WalkingSim::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
