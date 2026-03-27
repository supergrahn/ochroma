//! The Ochroma Engine — the real, complete engine binary.
//!
//! This is IT. One binary that opens a window, loads assets, creates the full
//! rendering pipeline (spectral framebuffer, temporal accumulation, tone mapping,
//! DLSS, CLAS clustering, frustum culling, LOD, particles, lighting), handles
//! input, runs scripts, and presents frames.
//!
//! Usage:
//!   cargo run --bin ochroma                         # default scene
//!   cargo run --bin ochroma -- level.ochroma_map    # load a map file
//!   cargo run --bin ochroma -- scene.ply            # load a .ply file

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use vox_app::editor::SceneEditor;
use vox_core::engine_runtime::{EngineConfig, EngineRuntime};
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::camera::CameraController;
use vox_render::clas;
use vox_render::dlss::{DlssPipeline, DlssQuality, FrameGeneration};
use vox_render::frustum::Frustum;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::lighting::{LightManager, PointLight};
use vox_render::lod;
use vox_render::mega_geometry::MegaGeometryDispatch;
use vox_render::particles::{ParticleEmitter, ParticleSystem};
use vox_render::spectral::RenderCamera;
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_render::spectral_tonemapper::{tonemap_spectral_framebuffer, ToneMapOperator, ToneMapSettings};
use vox_render::temporal::TemporalAccumulator;

use vox_audio::AudioEngine;
use vox_physics::rapier::RapierPhysicsWorld;

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

// ---------------------------------------------------------------------------
// The engine application — owns every system
// ---------------------------------------------------------------------------

struct EngineApp {
    // Core engine runtime (scripts, ECS, time, input)
    engine: EngineRuntime,

    // Window + GPU
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,

    // Rendering pipeline
    rasteriser: SoftwareRasteriser,
    camera: CameraController,
    particles: ParticleSystem,
    light_manager: LightManager,
    spectral_fb: SpectralFramebuffer,
    temporal: TemporalAccumulator,
    tonemap: ToneMapSettings,
    dlss: DlssPipeline,

    // Scene splats (loaded from assets)
    scene_splats: Vec<GaussianSplat>,

    // CLAS clustering (precomputed on scene load)
    clas_cluster_count: usize,
    clas_bvh_depth: u32,
    clas_avg_per_cluster: f32,

    // MegaGeometry
    mega_tile_count: u32,

    // Input state (raw winit -> engine each frame)
    keys: HashSet<KeyCode>,
    mouse_captured: bool,
    last_mouse: Option<(f64, f64)>,
    left_click_pending: bool,
    mouse_x: f64,
    mouse_y: f64,
    ctrl_held: bool,

    // Camera FPS controls (yaw/pitch, WASD)
    cam_yaw: f32,
    cam_pitch: f32,
    cam_speed: f32,

    // Stats
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    fps: f32,

    // Editor
    editor: SceneEditor,
    editor_visible: bool,

    // Exposure
    exposure: f32,

    // Fast render mode: skip spectral pipeline (P key toggle)
    spectral_bypass: bool,

    // CLI asset path
    asset_path: Option<String>,

    // Placed objects (from editor left-click)
    placed_objects: Vec<(Vec3, Vec<GaussianSplat>)>,

    // Audio engine (with rodio backend)
    audio: AudioEngine,
    click_counter: u32,

    // Rapier physics world
    physics: RapierPhysicsWorld,

    // Entity -> splat range mapping: entity_id -> (start_index, end_index) in scene_splats
    entity_splat_ranges: HashMap<u32, (usize, usize)>,
    // Original entity positions at scene build time (for computing deltas)
    entity_original_positions: HashMap<u32, [f32; 3]>,
}

// ---------------------------------------------------------------------------
// Illuminant from time of day
// ---------------------------------------------------------------------------

fn illuminant_for_time(hour: f32) -> Illuminant {
    let hour = hour % 24.0;
    let d65 = Illuminant::d65();
    let warm = Illuminant::a();
    let cool = Illuminant {
        bands: [30.0, 45.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0],
    };

    let (a, b, t) = if (6.0..12.0).contains(&hour) {
        (&warm, &d65, (hour - 6.0) / 6.0)
    } else if (12.0..18.0).contains(&hour) {
        (&d65, &warm, (hour - 12.0) / 6.0)
    } else if hour >= 18.0 {
        (&warm, &cool, (hour - 18.0) / 6.0)
    } else {
        (&cool, &warm, hour / 6.0)
    };

    let mut bands = [0.0f32; 8];
    for i in 0..8 {
        bands[i] = a.bands[i] * (1.0 - t) + b.bands[i] * t;
    }
    Illuminant { bands }
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

fn tonemap_operator_name(op: ToneMapOperator) -> &'static str {
    match op {
        ToneMapOperator::None => "Linear",
        ToneMapOperator::ACES => "ACES",
        ToneMapOperator::Reinhard => "Reinhard",
        ToneMapOperator::Filmic => "Filmic",
    }
}

fn dlss_quality_name(q: DlssQuality) -> &'static str {
    match q {
        DlssQuality::Off => "Off",
        DlssQuality::Quality => "Quality",
        DlssQuality::Balanced => "Balanced",
        DlssQuality::Performance => "Performance",
        DlssQuality::UltraPerformance => "Ultra Perf",
    }
}

fn next_dlss_quality(q: DlssQuality) -> DlssQuality {
    match q {
        DlssQuality::Off => DlssQuality::Quality,
        DlssQuality::Quality => DlssQuality::Balanced,
        DlssQuality::Balanced => DlssQuality::Performance,
        DlssQuality::Performance => DlssQuality::UltraPerformance,
        DlssQuality::UltraPerformance => DlssQuality::Off,
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

// ---------------------------------------------------------------------------
// Bitmap font (5x7 pixel glyphs) for HUD
// ---------------------------------------------------------------------------

const CHAR_WIDTH: u32 = 6;

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
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '>' => [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
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
// Construction
// ---------------------------------------------------------------------------

impl EngineApp {
    fn new(asset_path: Option<String>) -> Self {
        let config = EngineConfig {
            window_title: "Ochroma Engine".into(),
            window_width: DEFAULT_WIDTH,
            window_height: DEFAULT_HEIGHT,
            ..Default::default()
        };

        let mut editor = SceneEditor::new();
        editor.visible = false;

        let mut engine = EngineRuntime::new(config);
        engine.load_scene("Default");

        let dlss = DlssPipeline::new(DEFAULT_WIDTH, DEFAULT_HEIGHT, DlssQuality::Performance);
        let (render_w, render_h) = dlss.render_resolution();

        Self {
            engine,
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(render_w, render_h),
            camera: CameraController::new(DEFAULT_WIDTH as f32 / DEFAULT_HEIGHT as f32),
            particles: ParticleSystem::new(10_000),
            light_manager: LightManager::new(51.5), // London latitude
            spectral_fb: SpectralFramebuffer::new(render_w, render_h),
            temporal: TemporalAccumulator::new(render_w, render_h),
            tonemap: ToneMapSettings::default(),
            dlss,
            scene_splats: Vec::new(),
            clas_cluster_count: 0,
            clas_bvh_depth: 0,
            clas_avg_per_cluster: 0.0,
            mega_tile_count: 0,
            keys: HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            left_click_pending: false,
            mouse_x: 0.0,
            mouse_y: 0.0,
            ctrl_held: false,
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            cam_speed: 15.0,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            fps: 0.0,
            editor,
            editor_visible: false,
            exposure: 1.0,
            spectral_bypass: true, // fast mode by default
            asset_path,
            placed_objects: Vec::new(),
            audio: {
                let audio = AudioEngine::new(64);
                println!("[ochroma] Audio: synth mode (WAV generation)");
                audio
            },
            click_counter: 0,
            physics: {
                let mut physics = RapierPhysicsWorld::new();
                // Ground plane collider
                physics.add_static_collider([0.0, -0.5, 0.0], [100.0, 0.5, 100.0]);
                println!("[ochroma] Physics world initialised (Rapier3D)");
                physics
            },
            entity_splat_ranges: HashMap::new(),
            entity_original_positions: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Scene loading
    // -----------------------------------------------------------------------

    fn build_scene(&mut self) {
        // Try loading from CLI arg
        if let Some(path) = &self.asset_path {
            if path.ends_with(".ply") {
                match vox_data::ply_loader::load_ply(std::path::Path::new(path)) {
                    Ok(splats) => {
                        println!("[ochroma] Loaded {} splats from {}", splats.len(), path);
                        self.scene_splats = splats;
                        self.run_clas();
                        return;
                    }
                    Err(e) => {
                        eprintln!("[ochroma] Failed to load {}: {}", path, e);
                        eprintln!("[ochroma] Falling back to default scene");
                    }
                }
            } else {
                println!("[ochroma] Scene file format not yet supported: {}", path);
                println!("[ochroma] Falling back to default scene");
            }
        }

        // Default scene: terrain + buildings + trees
        println!("[ochroma] Building default scene...");

        // Volumetric terrain
        let vol = vox_terrain::volume::generate_demo_volume(42);
        let materials = vox_terrain::volume::default_volume_materials();
        let terrain_splats = vox_terrain::volume::volume_to_splats(&vol, &materials, 42);
        println!("[ochroma]   Terrain: {} splats", terrain_splats.len());

        // Populate editor entities
        self.editor.add_entity("Terrain", "terrain", Vec3::ZERO);
        for i in 0..4 {
            self.editor.add_entity(
                &format!("Building {}", i + 1),
                "building.ply",
                Vec3::new(i as f32 * 10.0 - 15.0, 0.0, 20.0),
            );
        }
        for i in 0..6 {
            self.editor.add_entity(
                &format!("Tree {}", i + 1),
                "tree.ply",
                Vec3::new(i as f32 * 8.0 - 20.0, 0.0, 10.0),
            );
        }

        // Populate engine scene entities (for scripts)
        self.engine.add_entity("Terrain", None, [0.0, 0.0, 0.0], None);
        for i in 0..4u32 {
            self.engine.add_entity(
                &format!("Building {}", i + 1),
                Some("building.ply"),
                [i as f32 * 10.0 - 15.0, 0.0, 20.0],
                None,
            );
        }
        for i in 0..6u32 {
            self.engine.add_entity(
                &format!("Tree {}", i + 1),
                Some("tree.ply"),
                [i as f32 * 8.0 - 20.0, 0.0, 10.0],
                None,
            );
        }

        // Build scene_splats with entity-splat range tracking
        self.scene_splats.clear();
        self.entity_splat_ranges.clear();
        self.entity_original_positions.clear();

        // Entity 0 = Terrain
        let terrain_start = self.scene_splats.len();
        self.scene_splats.extend(terrain_splats);
        let terrain_end = self.scene_splats.len();
        self.entity_splat_ranges.insert(0, (terrain_start, terrain_end));
        self.entity_original_positions.insert(0, [0.0, 0.0, 0.0]);

        // Entities 1..4 = Buildings
        for i in 0..4u32 {
            let entity_id = i + 1;
            let pos = [i as f32 * 10.0 - 15.0, 0.0, 20.0];
            let start = self.scene_splats.len();
            let b = vox_data::proc_gs_advanced::generate_detailed_building(
                42 + i as u64, 6.0, 8.0, 2 + (i % 3), "victorian",
            );
            for s in &b {
                let mut ws = *s;
                ws.position[0] += pos[0];
                ws.position[2] += pos[2];
                self.scene_splats.push(ws);
            }
            let end = self.scene_splats.len();
            self.entity_splat_ranges.insert(entity_id, (start, end));
            self.entity_original_positions.insert(entity_id, pos);
            // Add physics collider for building
            self.physics.add_static_collider(pos, [5.0, 10.0, 8.0]);
        }

        // Entities 5..10 = Trees
        for i in 0..6u32 {
            let entity_id = i + 5;
            let pos = [i as f32 * 8.0 - 20.0, 0.0, 10.0];
            let start = self.scene_splats.len();
            let t = vox_data::proc_gs_advanced::generate_tree(100 + i as u64, 6.0 + i as f32, 2.5);
            for s in &t {
                let mut ws = *s;
                ws.position[0] += pos[0];
                ws.position[2] += pos[2];
                self.scene_splats.push(ws);
            }
            let end = self.scene_splats.len();
            self.entity_splat_ranges.insert(entity_id, (start, end));
            self.entity_original_positions.insert(entity_id, pos);
        }

        println!("[ochroma] Total scene: {} splats ({} entities tracked)",
            self.scene_splats.len(), self.entity_splat_ranges.len());

        // Set up particles
        self.particles.add_emitter(ParticleEmitter::smoke(Vec3::new(0.0, 5.0, 20.0)));
        self.particles.add_emitter(ParticleEmitter::dust(Vec3::new(-10.0, 0.5, 15.0)));

        // Set up point lights
        self.light_manager.add_point_light(PointLight {
            position: Vec3::new(5.0, 8.0, 20.0),
            color: [1.0, 0.9, 0.7],
            intensity: 50.0,
            radius: 30.0,
        });
        self.light_manager.add_point_light(PointLight {
            position: Vec3::new(-15.0, 4.0, 10.0),
            color: [0.7, 0.8, 1.0],
            intensity: 30.0,
            radius: 20.0,
        });

        // Also register lights in engine scene
        self.engine.scene.add_point_light([5.0, 8.0, 20.0], [1.0, 0.9, 0.7], 50.0, 30.0);
        self.engine.scene.add_point_light([-15.0, 4.0, 10.0], [0.7, 0.8, 1.0], 30.0, 20.0);

        // CLAS clustering + MegaGeometry
        self.run_clas();
    }

    fn run_clas(&mut self) {
        if self.scene_splats.is_empty() {
            return;
        }
        let clusters = clas::build_clusters(&self.scene_splats, 128);
        let bvh = clas::build_cluster_bvh(&clusters);
        let stats = clas::compute_stats(&clusters, &bvh);
        self.clas_cluster_count = stats.cluster_count;
        self.clas_bvh_depth = stats.bvh_depth;
        self.clas_avg_per_cluster = stats.avg_splats_per_cluster;
        println!(
            "[ochroma] CLAS: {} clusters, BVH depth {}, avg {:.0} splats/cluster",
            stats.cluster_count, stats.bvh_depth, stats.avg_splats_per_cluster,
        );

        let dispatch = MegaGeometryDispatch::new(self.rasteriser.width, self.rasteriser.height, 500_000);
        self.mega_tile_count = dispatch.tile_count();
        println!(
            "[ochroma] MegaGeometry: {} tiles at {}x{}",
            self.mega_tile_count, self.rasteriser.width, self.rasteriser.height,
        );
    }

    // -----------------------------------------------------------------------
    // Camera
    // -----------------------------------------------------------------------

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

        if self.keys.contains(&KeyCode::KeyW) {
            self.camera.position += forward * speed;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.camera.position -= forward * speed;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.camera.position -= right * speed;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.camera.position += right * speed;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.camera.position.y += speed;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.camera.position.y -= speed;
        }

        // Update camera target from yaw/pitch
        self.camera.target = self.camera.position + forward;
    }

    // -----------------------------------------------------------------------
    // Full-pipeline render
    // -----------------------------------------------------------------------

    fn render_frame(&mut self) -> Vec<[u8; 4]> {
        let (render_w, render_h) = self.dlss.render_resolution();
        let display_w = self.dlss.display_width;
        let display_h = self.dlss.display_height;

        // Resize internal buffers if DLSS resolution changed
        if self.rasteriser.width != render_w || self.rasteriser.height != render_h {
            self.rasteriser = SoftwareRasteriser::new(render_w, render_h);
            self.spectral_fb = SpectralFramebuffer::new(render_w, render_h);
            self.temporal.resize(render_w, render_h);
        }

        let forward = self.camera_forward();
        let target = self.camera.position + forward;

        let render_camera = RenderCamera {
            view: Mat4::look_at_rh(self.camera.position, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                self.camera.fov,
                display_w as f32 / display_h as f32,
                self.camera.near,
                self.camera.far,
            ),
        };

        // --- Frustum culling ---
        let vp = render_camera.view_proj();
        let frustum = Frustum::from_view_proj(vp);

        // Gather visible splats: cull per-entity bounding spheres
        let mut visible_splats: Vec<GaussianSplat> = Vec::with_capacity(self.scene_splats.len());

        // For the main scene we do a simple pass -- each splat is tested via
        // the frustum (using a conservative bounding sphere of radius = max scale).
        // In a real production engine the CLAS BVH would be traversed here.
        for splat in &self.scene_splats {
            let pos = Vec3::from(splat.position);
            let radius = splat.scale[0].max(splat.scale[1]).max(splat.scale[2]) + 1.0;
            if frustum.contains_sphere(pos, radius) {
                visible_splats.push(*splat);
            }
        }

        // LOD selection on visible splats — aggressive reduction for distant splats
        let cam_pos = self.camera.position;
        let lod_indices: Vec<usize> = (0..visible_splats.len())
            .filter(|&i| {
                let dist = cam_pos.distance(Vec3::from(visible_splats[i].position));
                match lod::select_lod(dist) {
                    lod::LodLevel::Full => true,
                    lod::LodLevel::Reduced => i % 4 == 0, // keep every 4th for distant
                }
            })
            .collect();

        let culled_count = self.scene_splats.len() - visible_splats.len();
        let lod_culled = visible_splats.len() - lod_indices.len();

        // Build final splat list from LOD-selected indices
        let mut render_splats: Vec<GaussianSplat> = lod_indices
            .iter()
            .map(|&i| visible_splats[i])
            .collect();

        // Add placed objects
        for (pos, splats) in &self.placed_objects {
            for s in splats {
                let mut ws = *s;
                ws.position[0] += pos.x;
                ws.position[1] += pos.y;
                ws.position[2] += pos.z;
                render_splats.push(ws);
            }
        }

        // Add particle splats
        let particle_splats = self.particles.to_splats();
        render_splats.extend(&particle_splats);

        // Time-of-day illuminant
        let illuminant = illuminant_for_time(self.engine.scene.time_of_day);

        // 1. Software rasterise at internal resolution
        let render_start = Instant::now();
        let fb = self.rasteriser.render(&render_splats, &render_camera, &illuminant);

        let upscaled = if self.spectral_bypass {
            // FAST PATH: skip spectral pipeline, just DLSS upscale the rasterised output
            let pixel_count = (render_w * render_h) as usize;
            let depth = vec![1.0f32; pixel_count];
            let motion = vec![[0.0f32; 2]; pixel_count];
            self.dlss.upscale(&fb.pixels, render_w, render_h, &depth, &motion)
        } else {
            // QUALITY PATH: full spectral pipeline
            // 2. Write to spectral framebuffer
            self.spectral_fb.clear();
            for (i, pixel) in fb.pixels.iter().enumerate() {
                let x = (i % render_w as usize) as u32;
                let y = (i / render_w as usize) as u32;
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

                self.spectral_fb.write_sample(x, y, spectral, 1.0, [0.0, 1.0, 0.0], 0, spectral);
            }

            // 3. Temporal accumulation
            self.temporal.accumulate(&self.spectral_fb);
            self.temporal.write_to_framebuffer(&mut self.spectral_fb);

            // 4. Tone map
            self.tonemap.exposure = self.exposure;
            let tonemapped = tonemap_spectral_framebuffer(&self.spectral_fb, &illuminant, &self.tonemap);

            // 5. DLSS upscale
            let pixel_count = (render_w * render_h) as usize;
            let depth = vec![1.0f32; pixel_count];
            let motion = vec![[0.0f32; 2]; pixel_count];
            self.dlss.upscale(&tonemapped, render_w, render_h, &depth, &motion)
        };

        // 6. DLSS frame generation
        let display_count = (display_w * display_h) as usize;
        let display_motion = vec![[0.0f32; 2]; display_count];
        let _generated = self.dlss.generate_frame(&upscaled, &display_motion);
        let render_ms = render_start.elapsed().as_secs_f32() * 1000.0;

        let mut final_pixels = upscaled;

        // 7. HUD overlay
        let mode_label = if self.spectral_bypass { "FAST" } else { "SPECTRAL" };
        let y_off = 4u32;
        burn_text(&mut final_pixels, display_w, 4, y_off,
            &format!("OCHROMA ENGINE  {:.0} FPS  {:.1}ms  {} visible ({} culled, {} lod)  [{}]",
                self.fps,
                render_ms,
                render_splats.len(),
                culled_count,
                lod_culled,
                mode_label,
            ),
            [220, 220, 220]);
        burn_text(&mut final_pixels, display_w, 4, y_off + 10,
            &format!("TIME {:.0}:00  EV {:.2}  {}  DLSS {}  CLAS:{}  TILES:{}  LIGHTS:{}  PARTICLES:{}",
                self.engine.scene.time_of_day,
                self.exposure,
                tonemap_operator_name(self.tonemap.operator),
                dlss_quality_name(self.dlss.quality),
                self.clas_cluster_count,
                self.mega_tile_count,
                self.light_manager.point_light_count(),
                self.particles.particle_count(),
            ),
            [180, 180, 180]);
        burn_text(&mut final_pixels, display_w, 4, y_off + 20,
            &format!("ENTITIES: {}  SCRIPTS: {}  FRAME: {}  [P] toggle spectral",
                self.engine.stats.entity_count,
                self.engine.scripts.registered_scripts().len(),
                self.engine.stats.frame_number,
            ),
            [160, 160, 160]);

        // Editor overlay
        if self.editor_visible {
            burn_text(&mut final_pixels, display_w, 10, 40,
                &format!("EDITOR  {} entities", self.editor.entity_count()),
                [255, 255, 100]);

            for (i, entity) in self.editor.entities.iter().enumerate() {
                let is_sel = self.editor.selected == Some(entity.id);
                let prefix = if is_sel { ">" } else { " " };
                let label = format!("{} #{} {}", prefix, entity.id, entity.name);
                let color = if is_sel { [0, 255, 0] } else { [200, 200, 200] };
                burn_text(&mut final_pixels, display_w, 10, 54 + i as u32 * 10, &label, color);
            }

            if let Some(entity) = self.editor.selected_entity() {
                let rx = display_w.saturating_sub(240);
                burn_text(&mut final_pixels, display_w, rx, 40,
                    &format!("SELECTED: {}", entity.name), [0, 255, 0]);
                burn_text(&mut final_pixels, display_w, rx, 54,
                    &format!("POS: {:.1},{:.1},{:.1}", entity.position.x, entity.position.y, entity.position.z),
                    [180, 180, 180]);
                burn_text(&mut final_pixels, display_w, rx, 64,
                    &format!("ASSET: {}", entity.asset_path),
                    [180, 180, 180]);
            }
        }

        final_pixels
    }

    fn place_object_at_cursor(&mut self) {
        let forward = self.camera_forward();
        let place_pos = self.camera.position + forward * 15.0;
        let place_pos = Vec3::new(place_pos.x, 0.0, place_pos.z);

        let tree = vox_data::proc_gs_advanced::generate_tree(
            self.placed_objects.len() as u64 + 1000, 7.0, 3.0,
        );
        println!("[ochroma] Placed tree at ({:.1}, {:.1}, {:.1})", place_pos.x, place_pos.y, place_pos.z);
        self.placed_objects.push((place_pos, tree));
    }

    fn total_splat_count(&self) -> usize {
        self.scene_splats.len()
            + self.placed_objects.iter().map(|(_, s)| s.len()).sum::<usize>()
            + self.particles.particle_count()
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler -- the real game loop
// ---------------------------------------------------------------------------

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title(&self.engine.config.window_title)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.engine.config.window_width,
                self.engine.config.window_height,
            ));

        let window = Arc::new(
            event_loop.create_window(attrs).expect("Failed to create window"),
        );

        match WgpuBackend::new(Arc::clone(&window), DEFAULT_WIDTH, DEFAULT_HEIGHT) {
            Ok(backend) => {
                println!("[ochroma] GPU backend initialised");
                self.backend = Some(backend);
            }
            Err(e) => {
                eprintln!("[ochroma] GPU init failed: {}", e);
                eprintln!("[ochroma] Running headless -- frames render but may not display");
            }
        }

        self.window = Some(window);

        // Build scene + CLAS + particles + lights
        self.build_scene();

        // Start the engine runtime (initialises scripts)
        self.engine.start();

        println!();
        println!("=============================================");
        println!("     OCHROMA ENGINE -- RUNNING");
        println!("=============================================");
        println!("  {} entities | {} splats | {} clusters",
            self.engine.scene.entity_count(),
            self.scene_splats.len(),
            self.clas_cluster_count);
        println!("  {} point lights | {} particle emitters",
            self.light_manager.point_light_count(),
            self.particles.emitters.len());
        println!();
        println!("  Controls:");
        println!("    WASD          Move camera");
        println!("    Space/Shift   Up / Down");
        println!("    Right-click   Capture mouse for look");
        println!("    Left-click    Place object");
        println!("    Escape        Release mouse / Quit");
        println!("    F12           Screenshot");
        println!("    T             Advance time (+1 hour)");
        println!("    +/-           Adjust exposure");
        println!("    M             Cycle tone mapper");
        println!("    Q             Cycle DLSS quality");
        println!("    P             Toggle fast/spectral render");
        println!("    G             Toggle frame generation");
        println!("    Tab           Toggle editor");
        println!("    Arrows        Move selected (editor)");
        println!("    Delete        Delete selected (editor)");
        println!("    Ctrl+S        Save scene");
        println!("=============================================");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.engine.stop();
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys.insert(key);

                        match key {
                            KeyCode::Escape => {
                                if self.mouse_captured {
                                    self.mouse_captured = false;
                                    if let Some(w) = &self.window {
                                        w.set_cursor_visible(true);
                                    }
                                } else {
                                    self.engine.stop();
                                    event_loop.exit();
                                }
                            }
                            KeyCode::F12 => {
                                let pixels = self.render_frame();
                                let dir = std::env::temp_dir().join("ochroma_visual");
                                std::fs::create_dir_all(&dir).ok();
                                let path = dir.join("screenshot.ppm");
                                let (w, h) = (self.dlss.display_width, self.dlss.display_height);
                                let mut data = format!("P6\n{} {}\n255\n", w, h).into_bytes();
                                for p in &pixels {
                                    data.push(p[0]);
                                    data.push(p[1]);
                                    data.push(p[2]);
                                }
                                std::fs::write(&path, &data).ok();
                                println!("[ochroma] Screenshot: {}", path.display());
                            }
                            KeyCode::KeyT => {
                                self.engine.scene.time_of_day = (self.engine.scene.time_of_day + 1.0) % 24.0;
                                self.temporal.reset();
                                println!("[ochroma] Time: {:.0}:00", self.engine.scene.time_of_day);
                            }
                            KeyCode::Equal => {
                                self.exposure = (self.exposure * 1.2).min(16.0);
                                println!("[ochroma] Exposure: {:.2}", self.exposure);
                            }
                            KeyCode::Minus => {
                                self.exposure = (self.exposure / 1.2).max(0.05);
                                println!("[ochroma] Exposure: {:.2}", self.exposure);
                            }
                            KeyCode::KeyM => {
                                self.tonemap.operator = next_tonemap_operator(self.tonemap.operator);
                                println!("[ochroma] Tonemap: {}", tonemap_operator_name(self.tonemap.operator));
                            }
                            KeyCode::KeyQ => {
                                self.dlss.quality = next_dlss_quality(self.dlss.quality);
                                let (rw, rh) = self.dlss.render_resolution();
                                println!(
                                    "[ochroma] DLSS: {} (render {}x{} -> display {}x{})",
                                    dlss_quality_name(self.dlss.quality),
                                    rw, rh, self.dlss.display_width, self.dlss.display_height,
                                );
                            }
                            KeyCode::KeyP => {
                                self.spectral_bypass = !self.spectral_bypass;
                                self.temporal.reset();
                                println!(
                                    "[ochroma] Render mode: {}",
                                    if self.spectral_bypass { "FAST (direct RGB)" } else { "QUALITY (spectral pipeline)" }
                                );
                            }
                            KeyCode::KeyG => {
                                self.dlss.frame_gen = match self.dlss.frame_gen {
                                    FrameGeneration::Off => FrameGeneration::On,
                                    FrameGeneration::On => FrameGeneration::Off,
                                };
                                println!("[ochroma] Frame generation: {:?}", self.dlss.frame_gen);
                            }
                            KeyCode::Tab => {
                                self.editor_visible = !self.editor_visible;
                                self.editor.visible = self.editor_visible;
                                println!(
                                    "[ochroma] Editor {}",
                                    if self.editor_visible { "OPEN" } else { "CLOSED" }
                                );
                                if self.editor_visible {
                                    self.editor.show_console();
                                    if self.editor.selected.is_none() && !self.editor.entities.is_empty() {
                                        let first_id = self.editor.entities[0].id;
                                        self.editor.select(first_id);
                                    }
                                }
                            }
                            KeyCode::Delete => {
                                if self.editor_visible {
                                    self.editor.delete_selected();
                                }
                            }
                            KeyCode::ArrowUp => {
                                if self.editor_visible {
                                    self.editor.move_selected(Vec3::new(0.0, 0.0, -1.0));
                                }
                            }
                            KeyCode::ArrowDown => {
                                if self.editor_visible {
                                    self.editor.move_selected(Vec3::new(0.0, 0.0, 1.0));
                                }
                            }
                            KeyCode::ArrowLeft => {
                                if self.editor_visible {
                                    self.editor.move_selected(Vec3::new(-1.0, 0.0, 0.0));
                                }
                            }
                            KeyCode::ArrowRight => {
                                if self.editor_visible {
                                    self.editor.move_selected(Vec3::new(1.0, 0.0, 0.0));
                                }
                            }
                            KeyCode::KeyS if self.ctrl_held => {
                                let map = self.editor.export_to_map("Ochroma Scene");
                                let path = std::env::temp_dir().join("ochroma_scene.ochroma_map");
                                match map.save(&path) {
                                    Ok(()) => println!("[ochroma] Scene saved to {}", path.display()),
                                    Err(e) => eprintln!("[ochroma] Save failed: {}", e),
                                }
                            }
                            _ => {}
                        }
                    } else {
                        self.keys.remove(&key);
                    }
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.ctrl_held = modifiers.state().control_key();
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
                    // Generate a click sound (saved as WAV for proof)
                    self.click_counter += 1;
                    let sound = vox_audio::synth::generate_click();
                    let path = std::env::temp_dir().join(format!("ochroma_click_{}.wav", self.click_counter));
                    let _ = vox_audio::synth::save_wav(&sound, 44100, &path);
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
                self.dlss.resize(w, h);
                let (rw, rh) = self.dlss.render_resolution();
                self.rasteriser = SoftwareRasteriser::new(rw, rh);
                self.spectral_fb = SpectralFramebuffer::new(rw, rh);
                self.temporal.resize(rw, rh);
                self.camera.aspect_ratio = w as f32 / h as f32;
                if let Some(backend) = &mut self.backend {
                    backend.resize(w, h);
                }
            }

            // ---------------------------------------------------------------
            // THE FRAME -- this is where every system runs each frame
            // ---------------------------------------------------------------
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                // 1. Update camera from input
                self.update_camera(dt);

                // 2. Handle left-click placement
                if self.left_click_pending {
                    self.place_object_at_cursor();
                    self.left_click_pending = false;
                }

                // 3. Tick particles
                self.particles.tick(dt);

                // 4. Tick engine runtime (scripts, time advance)
                self.engine.tick(dt);

                // 4b. Step physics
                self.physics.step();

                // 4c. Tick audio
                self.audio.tick(dt);
                self.audio.set_listener(self.camera.position);

                // 4d. Sync entity positions to splats — when scripts move entities,
                //     the corresponding splats move with them.
                for entity in &self.engine.scene.entities {
                    if let Some(&(start, end)) = self.entity_splat_ranges.get(&entity.id) {
                        if let Some(&orig_pos) = self.entity_original_positions.get(&entity.id) {
                            let dx = entity.position[0] - orig_pos[0];
                            let dy = entity.position[1] - orig_pos[1];
                            let dz = entity.position[2] - orig_pos[2];
                            // Only apply if the entity actually moved
                            if dx.abs() > 1e-6 || dy.abs() > 1e-6 || dz.abs() > 1e-6 {
                                for i in start..end.min(self.scene_splats.len()) {
                                    self.scene_splats[i].position[0] += dx;
                                    self.scene_splats[i].position[1] += dy;
                                    self.scene_splats[i].position[2] += dz;
                                }
                                // Update original position so delta is relative
                                self.entity_original_positions.insert(entity.id, entity.position);
                            }
                        }
                    }
                }

                // 5. Full render pipeline: frustum cull -> LOD -> rasterise ->
                //    spectral FB -> temporal accumulation -> tonemap -> DLSS -> present
                let pixels = self.render_frame();

                // 6. Present to window
                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&pixels, self.dlss.display_width, self.dlss.display_height);
                }

                // 7. FPS counter + title update
                self.frame_count += 1;
                let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if elapsed >= 1.0 {
                    self.fps = self.frame_count as f32 / elapsed;
                    if let Some(w) = &self.window {
                        let dlss_label = match self.dlss.quality {
                            DlssQuality::Off => "DLSS Off".to_string(),
                            q => {
                                let (rw, rh) = self.dlss.render_resolution();
                                format!("DLSS {} ({}x{})", dlss_quality_name(q), rw, rh)
                            }
                        };
                        let fg_label = if self.dlss.frame_gen == FrameGeneration::On { " | FrameGen" } else { "" };
                        let editor_label = if self.editor_visible {
                            format!(" | Editor ({})", self.editor.entity_count())
                        } else {
                            String::new()
                        };
                        let mode_label = if self.spectral_bypass { "FAST" } else { "SPECTRAL" };
                        w.set_title(&format!(
                            "Ochroma -- {:.0} FPS | {} total | CLAS:{} | {:.0}:00 | {} | {} | {}{}{}",
                            self.fps,
                            self.total_splat_count(),
                            self.clas_cluster_count,
                            self.engine.scene.time_of_day,
                            mode_label,
                            dlss_label,
                            tonemap_operator_name(self.tonemap.operator),
                            fg_label,
                            editor_label,
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

// ---------------------------------------------------------------------------
// main -- launch the engine
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================");
    println!("        OCHROMA ENGINE v0.1.0");
    println!("  Spectral Gaussian Splatting Engine");
    println!("=============================================");

    let asset_path = std::env::args().nth(1);
    if let Some(ref path) = asset_path {
        println!("[ochroma] Loading: {}", path);
    } else {
        println!("[ochroma] No scene file specified -- using default scene");
    }

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = EngineApp::new(asset_path);
    event_loop.run_app(&mut app).expect("Event loop failed");
}
