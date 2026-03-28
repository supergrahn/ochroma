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

use vox_app::content_browser::ContentBrowser;
use vox_app::editor::SceneEditor;
use vox_core::engine_runtime::{EngineConfig, EngineRuntime};
use vox_core::game_ui::burn_text;
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::camera::CameraController;
use vox_render::clas;
use vox_render::dlss::{DlssPipeline, DlssQuality, FrameGeneration};
use vox_render::frustum::Frustum;
use vox_render::gpu::gpu_rasteriser::GpuRasteriser;
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
use vox_audio::SpatialAudioManager;
use vox_physics::rapier::RapierPhysicsWorld;
use vox_render::gizmos::GizmoRenderer;
use vox_ui::theme::apply_ochroma_theme;

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
    gpu_rasteriser: Option<GpuRasteriser>,

    // Rendering pipeline (software fallback)
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
    content_browser: ContentBrowser,

    // egui integration
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

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

    // ECS entity index -> Rapier body handle (for entities with ColliderComponent)
    entity_rapier_bodies: HashMap<u32, vox_physics::RigidBodyHandle>,
    // Rapier collider handle -> ECS entity index (for raycast picking)
    collider_to_entity: HashMap<vox_physics::ColliderHandle, u32>,

    // Entity -> splat range mapping: entity_id -> (start_index, end_index) in scene_splats
    entity_splat_ranges: HashMap<u32, (usize, usize)>,
    // Original entity positions at scene build time (for computing deltas)
    entity_original_positions: HashMap<u32, [f32; 3]>,

    // Gizmo renderer for translate/rotate/scale handles
    gizmo: GizmoRenderer,
    // Track left mouse button held state for gizmo dragging
    left_mouse_held: bool,

    // Spatial audio manager (3D positional audio with distance attenuation)
    spatial_audio: SpatialAudioManager,
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

        let engine = EngineRuntime::new(config);

        let dlss = DlssPipeline::new(DEFAULT_WIDTH, DEFAULT_HEIGHT, DlssQuality::Performance);
        let (render_w, render_h) = dlss.render_resolution();

        Self {
            engine,
            window: None,
            backend: None,
            gpu_rasteriser: None,
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
            content_browser: ContentBrowser::new(std::path::Path::new(".")),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            exposure: 1.0,
            spectral_bypass: true, // fast mode by default
            asset_path,
            placed_objects: Vec::new(),
            audio: {
                let mut audio = AudioEngine::new(64);
                audio.init_backend();
                println!("[ochroma] Audio: rodio backend initialised");
                audio
            },
            click_counter: 0,
            physics: {
                let mut physics = RapierPhysicsWorld::new();
                // Ground plane collider — 1km x 1km
                physics.add_static_collider([0.0, -0.5, 0.0], [500.0, 0.5, 500.0]);
                println!("[ochroma] Physics: Rapier3D world initialised (ground plane 1000x1000)");
                physics
            },
            entity_rapier_bodies: HashMap::new(),
            collider_to_entity: HashMap::new(),
            entity_splat_ranges: HashMap::new(),
            entity_original_positions: HashMap::new(),
            gizmo: GizmoRenderer::new(),
            left_mouse_held: false,
            spatial_audio: {
                let mgr = SpatialAudioManager::new();
                println!("[ochroma] Spatial audio manager initialised (available: {})", mgr.is_available());
                mgr
            },
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
        self.engine.spawn("Terrain").with_position(Vec3::ZERO);
        for i in 0..4u32 {
            self.engine.spawn(&format!("Building {}", i + 1))
                .with_asset("building.ply")
                .with_position(Vec3::new(i as f32 * 10.0 - 15.0, 0.0, 20.0));
        }
        for i in 0..6u32 {
            self.engine.spawn(&format!("Tree {}", i + 1))
                .with_asset("tree.ply")
                .with_position(Vec3::new(i as f32 * 8.0 - 20.0, 0.0, 10.0));
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
            // Add Rapier static collider for building and track entity mapping
            let col_handle = self.physics.add_static_collider(pos, [5.0, 10.0, 8.0]);
            self.collider_to_entity.insert(col_handle, entity_id);
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
            // Add Rapier static collider for tree trunk
            let tree_height = 6.0 + i as f32;
            let col_handle = self.physics.add_static_collider(
                [pos[0], tree_height * 0.5, pos[2]],
                [1.5, tree_height * 0.5, 1.5],
            );
            self.collider_to_entity.insert(col_handle, entity_id);
        }

        println!("[ochroma] Physics: {} bodies, {} colliders in Rapier world",
            self.physics.body_count(), self.physics.collider_count());
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

        // Also register lights in engine world
        self.engine.spawn("Light1")
            .with_position(Vec3::new(5.0, 8.0, 20.0))
            .with_light([1.0, 0.9, 0.7], 50.0, 30.0);
        self.engine.spawn("Light2")
            .with_position(Vec3::new(-15.0, 4.0, 10.0))
            .with_light([0.7, 0.8, 1.0], 30.0, 20.0);

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
        let illuminant = illuminant_for_time(self.engine.time_of_day());

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
            [220, 220, 220], 1);
        burn_text(&mut final_pixels, display_w, 4, y_off + 10,
            &format!("TIME {:.0}:00  EV {:.2}  {}  DLSS {}  CLAS:{}  TILES:{}  LIGHTS:{}  PARTICLES:{}",
                self.engine.time_of_day(),
                self.exposure,
                tonemap_operator_name(self.tonemap.operator),
                dlss_quality_name(self.dlss.quality),
                self.clas_cluster_count,
                self.mega_tile_count,
                self.light_manager.point_light_count(),
                self.particles.particle_count(),
            ),
            [180, 180, 180], 1);
        burn_text(&mut final_pixels, display_w, 4, y_off + 20,
            &format!("ENTITIES: {}  SCRIPTS: {}  FRAME: {}  [P] toggle spectral",
                self.engine.stats.entity_count,
                self.engine.registered_script_count(),
                self.engine.stats.frame_number,
            ),
            [160, 160, 160], 1);

        // Gizmo overlay (drawn on top of scene in editor mode)
        if self.editor_visible {
            if let Some(entity) = self.editor.selected_entity() {
                let forward = self.camera_forward();
                let target = self.camera.position + forward;
                let view = Mat4::look_at_rh(self.camera.position, target, Vec3::Y);
                let proj = Mat4::perspective_rh(
                    self.camera.fov,
                    display_w as f32 / display_h as f32,
                    self.camera.near,
                    self.camera.far,
                );
                let vp = proj * view;
                self.gizmo.draw_overlay(
                    &mut final_pixels,
                    display_w,
                    display_h,
                    entity.position,
                    vp,
                );
            }
        }

        // Editor overlay
        if self.editor_visible {
            burn_text(&mut final_pixels, display_w, 10, 40,
                &format!("EDITOR  {} entities", self.editor.entity_count()),
                [255, 255, 100], 1);

            for (i, entity) in self.editor.entities.iter().enumerate() {
                let is_sel = self.editor.selected == Some(entity.id);
                let prefix = if is_sel { ">" } else { " " };
                let label = format!("{} #{} {}", prefix, entity.id, entity.name);
                let color = if is_sel { [0, 255, 0] } else { [200, 200, 200] };
                burn_text(&mut final_pixels, display_w, 10, 54 + i as u32 * 10, &label, color, 1);
            }

            if let Some(entity) = self.editor.selected_entity() {
                let rx = display_w.saturating_sub(240);
                burn_text(&mut final_pixels, display_w, rx, 40,
                    &format!("SELECTED: {}", entity.name), [0, 255, 0], 1);
                burn_text(&mut final_pixels, display_w, rx, 54,
                    &format!("POS: {:.1},{:.1},{:.1}", entity.position.x, entity.position.y, entity.position.z),
                    [180, 180, 180], 1);
                burn_text(&mut final_pixels, display_w, rx, 64,
                    &format!("ASSET: {}", entity.asset_path),
                    [180, 180, 180], 1);
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
// Input + redraw helpers (extracted from window_event)
// ---------------------------------------------------------------------------

impl EngineApp {
    fn handle_keyboard_event(&mut self, event: &winit::event::KeyEvent, event_loop: &ActiveEventLoop) {
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
                        let new_hour = (self.engine.time_of_day() + 1.0) % 24.0;
                        self.engine.set_time_of_day(new_hour);
                        self.temporal.reset();
                        println!("[ochroma] Time: {:.0}:00", self.engine.time_of_day());
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

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match button {
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
            MouseButton::Left if state == ElementState::Released => {
                // End gizmo drag on release
                if self.gizmo.dragging {
                    self.gizmo.end_drag();
                }
                self.left_mouse_held = false;
            }
            MouseButton::Left if state == ElementState::Pressed => {
                self.left_mouse_held = true;

                // In editor mode, check gizmo hit first before entity picking
                if self.editor_visible {
                    if let Some(sel_entity) = self.editor.selected_entity() {
                        let forward = self.camera_forward();
                        let cam_target = self.camera.position + forward;
                        let view = Mat4::look_at_rh(self.camera.position, cam_target, Vec3::Y);
                        let (dw, dh) = (self.dlss.display_width, self.dlss.display_height);
                        let proj = Mat4::perspective_rh(
                            self.camera.fov,
                            dw as f32 / dh as f32,
                            self.camera.near,
                            self.camera.far,
                        );
                        let vp = proj * view;
                        let entity_pos = sel_entity.position;
                        if let Some(axis) = self.gizmo.hit_test(
                            self.mouse_x as f32,
                            self.mouse_y as f32,
                            entity_pos,
                            vp,
                            dw,
                            dh,
                        ) {
                            self.gizmo.begin_drag(axis, self.mouse_x as f32, self.mouse_y as f32);
                            // Don't also do entity picking when starting a gizmo drag
                            self.left_click_pending = false;
                        } else {
                            self.left_click_pending = true;
                        }
                    } else {
                        self.left_click_pending = true;
                    }
                } else {
                    self.left_click_pending = true;
                }
                // Play click sound through speakers (both legacy and spatial audio)
                self.click_counter += 1;
                self.audio.play_sine_backend(self.click_counter, 800.0, 0.05, 0.3);
                self.spatial_audio.play_tone(800.0, 0.05, 0.3);
            }
            _ => {}
        }
    }

    fn handle_mouse_move(&mut self, x: f64, y: f64) {
        self.mouse_x = x;
        self.mouse_y = y;

        // Gizmo drag: move the selected entity along the constrained axis
        if self.gizmo.dragging && self.editor_visible {
            if let Some(sel_id) = self.editor.selected {
                if let Some(sel_entity) = self.editor.entities.iter().find(|e| e.id == sel_id) {
                    let entity_pos = sel_entity.position;
                    let forward = self.camera_forward();
                    let cam_target = self.camera.position + forward;
                    let view = Mat4::look_at_rh(self.camera.position, cam_target, Vec3::Y);
                    let (dw, dh) = (self.dlss.display_width, self.dlss.display_height);
                    let proj = Mat4::perspective_rh(
                        self.camera.fov,
                        dw as f32 / dh as f32,
                        self.camera.near,
                        self.camera.far,
                    );
                    let vp = proj * view;
                    let delta = self.gizmo.update_drag(
                        x as f32,
                        y as f32,
                        entity_pos,
                        vp,
                        dw,
                        dh,
                    );
                    if delta.length_squared() > 1e-8 {
                        self.editor.move_selected(delta);
                    }
                }
            }
        } else if self.mouse_captured {
            if let Some((lx, ly)) = self.last_mouse {
                let dx = (x - lx) as f32;
                let dy = (y - ly) as f32;
                self.cam_yaw += dx * 0.003;
                self.cam_pitch = (self.cam_pitch - dy * 0.003).clamp(-1.5, 1.5);
            }
            self.last_mouse = Some((x, y));
        }
    }

    fn handle_resize(&mut self, width: u32, height: u32) {
        // Cap surface size to GPU texture limits (wgpu max is typically 2048 or 8192
        // depending on adapter — use 2048 as safe minimum for software rasteriser)
        let max_dim = 4096;
        let w = width.max(1).min(max_dim);
        let h = height.max(1).min(max_dim);
        self.dlss.resize(w, h);
        let (rw, rh) = self.dlss.render_resolution();
        self.rasteriser = SoftwareRasteriser::new(rw, rh);
        self.spectral_fb = SpectralFramebuffer::new(rw, rh);
        self.temporal.resize(rw, rh);
        self.camera.aspect_ratio = w as f32 / h as f32;
        if let Some(backend) = &mut self.backend {
            backend.resize(w, h);
        }
        if let Some(gpu_rast) = &mut self.gpu_rasteriser {
            gpu_rast.resize(w, h);
        }
    }

    fn handle_redraw(&mut self, _event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // 1. Update camera from input
        self.update_camera(dt);

        // 2. Handle left-click: Rapier raycast for editor pick, or object placement
        if self.left_click_pending {
            if self.editor_visible {
                // Unproject screen coords to get a world-space ray
                let forward = self.camera_forward();
                let cam_target = self.camera.position + forward;
                let view = Mat4::look_at_rh(self.camera.position, cam_target, Vec3::Y);
                let (dw, dh) = (self.dlss.display_width, self.dlss.display_height);
                let proj = Mat4::perspective_rh(
                    self.camera.fov,
                    dw as f32 / dh as f32,
                    self.camera.near,
                    self.camera.far,
                );
                let inv_vp = (proj * view).inverse();
                let ndc_x = (2.0 * self.mouse_x as f32 / dw as f32) - 1.0;
                let ndc_y = 1.0 - (2.0 * self.mouse_y as f32 / dh as f32);
                let unproject = |ndc_z: f32| -> Vec3 {
                    let clip = glam::Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
                    let world = inv_vp * clip;
                    Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w)
                };
                let near_pt = unproject(-1.0);
                let far_pt = unproject(1.0);
                let ray_dir = (far_pt - near_pt).normalize();

                // Try Rapier raycast first for precise physics-based picking
                let mut picked = false;
                if let Some((col_handle, _hit_pos, _dist)) = self.physics.raycast_with_collider(
                    [near_pt.x, near_pt.y, near_pt.z],
                    [ray_dir.x, ray_dir.y, ray_dir.z],
                    1000.0,
                ) {
                    if let Some(&entity_id) = self.collider_to_entity.get(&col_handle) {
                        self.editor.select(entity_id);
                        if let Some(name) = self.editor.selected_name() {
                            println!("[ochroma] Rapier pick: '{}' (id={})", name, entity_id);
                        }
                        picked = true;
                    }
                }

                // Fall back to editor ray-sphere test if Rapier didn't match an entity
                if !picked {
                    if let Some(id) = self.editor.pick_entity_at_screen_pos(
                        self.mouse_x as f32,
                        self.mouse_y as f32,
                        dw,
                        dh,
                        inv_vp,
                    ) {
                        self.editor.select(id);
                        if let Some(name) = self.editor.selected_name() {
                            println!("[ochroma] Editor pick: '{}' (id={})", name, id);
                        }
                    }
                }
            } else {
                self.place_object_at_cursor();
            }
            self.left_click_pending = false;
        }

        // 3. Tick particles
        self.particles.tick(dt);

        // 4. Tick engine runtime (scripts, time advance)
        self.engine.tick(dt);

        // 4b. Step Rapier physics and sync dynamic bodies back to ECS
        self.physics.step();
        // Sync: read positions from Rapier dynamic bodies back into ECS transforms
        {
            use vox_core::ecs::TransformComponent;
            let body_map: Vec<(u32, vox_physics::RigidBodyHandle)> =
                self.entity_rapier_bodies.iter().map(|(&e, &h)| (e, h)).collect();
            for (eid, handle) in body_map {
                if let Some(pos) = self.physics.body_position(handle) {
                    // Update the ECS transform from Rapier
                    let mut query = self.engine.world.query::<(bevy_ecs::prelude::Entity, &mut TransformComponent)>();
                    for (entity, mut transform) in query.iter_mut(&mut self.engine.world) {
                        if entity.index() == eid {
                            transform.position = Vec3::new(pos[0], pos[1], pos[2]);
                            break;
                        }
                    }
                }
            }
        }

        // 4c. Tick audio (legacy AudioEngine + spatial audio manager)
        self.audio.tick(dt);
        self.audio.set_listener(self.camera.position);

        // Spatial audio: update listener position/orientation from camera, tick
        let cam_fwd = self.camera_forward();
        self.spatial_audio.set_listener(self.camera.position, cam_fwd, Vec3::Y);
        self.spatial_audio.tick(dt);

        // Process pending script commands for audio playback
        {
            use vox_core::script_interface::ScriptCommand;
            let pending = std::mem::take(
                &mut self.engine.world.resource_mut::<vox_core::engine_runtime::PendingScriptCommands>().commands,
            );
            for (_entity, commands) in &pending {
                for cmd in commands {
                    match cmd {
                        ScriptCommand::PlaySound { clip, volume, .. } => {
                            // Play via spatial audio manager as a procedural tone
                            // 440Hz for generic, 800Hz for click, 600Hz ascending for collect
                            let freq = match clip.as_str() {
                                "click" => 800.0,
                                "collect" => 600.0,
                                "jump" => 400.0,
                                _ => 440.0,
                            };
                            self.spatial_audio.play_tone(freq, 0.2, *volume);
                        }
                        _ => {}
                    }
                }
            }
            // Put unprocessed commands back (other systems may need them)
            self.engine.world.resource_mut::<vox_core::engine_runtime::PendingScriptCommands>().commands = pending;
        }

        // 4d. Sync entity positions to splats — when scripts move entities,
        //     the corresponding splats move with them.
        {
            use vox_core::ecs::TransformComponent;
            let mut query = self.engine.world.query::<(bevy_ecs::prelude::Entity, &TransformComponent)>();
            let entities: Vec<(u32, [f32; 3])> = query.iter(&self.engine.world)
                .map(|(e, t)| (e.index(), [t.position.x, t.position.y, t.position.z]))
                .collect();
            for (eid, pos) in entities {
                if let Some(&(start, end)) = self.entity_splat_ranges.get(&eid) {
                    if let Some(&orig_pos) = self.entity_original_positions.get(&eid) {
                        let dx = pos[0] - orig_pos[0];
                        let dy = pos[1] - orig_pos[1];
                        let dz = pos[2] - orig_pos[2];
                        if dx.abs() > 1e-6 || dy.abs() > 1e-6 || dz.abs() > 1e-6 {
                            for i in start..end.min(self.scene_splats.len()) {
                                self.scene_splats[i].position[0] += dx;
                                self.scene_splats[i].position[1] += dy;
                                self.scene_splats[i].position[2] += dz;
                            }
                            self.entity_original_positions.insert(eid, pos);
                        }
                    }
                }
            }
        }

        // 5. Render + present
        //    Primary path: GPU rasteriser directly to surface texture.
        //    Fallback: software rasteriser -> blit via backend.
        if self.gpu_rasteriser.is_some() && self.backend.is_some() {
            // --- GPU primary render path ---
            let backend = self.backend.as_ref().expect("backend checked above");
            let surface_tex = match backend.surface().get_current_texture() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[ochroma] GPU surface error: {}", e);
                    // Fall through to software path below
                    let pixels = self.render_frame();
                    if let Some(b) = &self.backend {
                        b.present_framebuffer(&pixels, self.dlss.display_width, self.dlss.display_height);
                    }
                    // skip to FPS counter
                    self.frame_count += 1;
                    let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                    if elapsed >= 1.0 {
                        self.fps = self.frame_count as f32 / elapsed;
                        self.frame_count = 0;
                        self.fps_timer = now;
                    }
                    if let Some(w) = &self.window { w.request_redraw(); }
                    return;
                }
            };
            let view = surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Build visible splats (frustum cull + LOD) for GPU path
            let forward = self.camera_forward();
            let target = self.camera.position + forward;
            let render_camera = RenderCamera {
                view: Mat4::look_at_rh(self.camera.position, target, Vec3::Y),
                proj: Mat4::perspective_rh(
                    self.camera.fov,
                    self.dlss.display_width as f32 / self.dlss.display_height as f32,
                    self.camera.near,
                    self.camera.far,
                ),
            };
            let vp = render_camera.view_proj();
            let frustum = Frustum::from_view_proj(vp);
            let cam_pos = self.camera.position;

            let mut visible_splats: Vec<GaussianSplat> = Vec::with_capacity(self.scene_splats.len());
            for splat in &self.scene_splats {
                let pos = Vec3::from(splat.position);
                let radius = splat.scale[0].max(splat.scale[1]).max(splat.scale[2]) + 1.0;
                if frustum.contains_sphere(pos, radius) {
                    visible_splats.push(*splat);
                }
            }
            // LOD
            let lod_splats: Vec<GaussianSplat> = visible_splats.iter().enumerate()
                .filter(|&(i, s)| {
                    let dist = cam_pos.distance(Vec3::from(s.position));
                    match lod::select_lod(dist) {
                        lod::LodLevel::Full => true,
                        lod::LodLevel::Reduced => i % 4 == 0,
                    }
                })
                .map(|(_, s)| *s)
                .collect();

            // Add placed objects + particles
            let mut render_splats = lod_splats;
            for (pos, splats) in &self.placed_objects {
                for s in splats {
                    let mut ws = *s;
                    ws.position[0] += pos.x;
                    ws.position[1] += pos.y;
                    ws.position[2] += pos.z;
                    render_splats.push(ws);
                }
            }
            render_splats.extend(&self.particles.to_splats());

            let illuminant = illuminant_for_time(self.engine.time_of_day());

            let gpu_rast = self.gpu_rasteriser.as_ref().expect("gpu_rasteriser checked above");
            gpu_rast.render(
                backend.device(),
                backend.queue(),
                &view,
                &render_splats,
                &render_camera,
                &illuminant,
            );

            // --- egui render on top of scene ---
            if let (Some(egui_state), Some(egui_renderer)) =
                (&mut self.egui_state, &mut self.egui_renderer)
            {
                let window = self.window.as_ref().expect("window exists during redraw");
                let raw_input = egui_state.take_egui_input(window);
                let splat_count = render_splats.len();
                let fps = self.fps;
                let dlss_mode = dlss_quality_name(self.dlss.quality);
                let editor_visible = self.editor_visible;
                let full_output = self.egui_ctx.run(raw_input, |ctx| {
                    if editor_visible {
                        self.editor.show(ctx);
                        self.content_browser.show(ctx);
                    }
                    // Always show HUD
                    egui::Area::new(egui::Id::new("hud_overlay"))
                        .fixed_pos(egui::pos2(4.0, 4.0))
                        .show(ctx, |ui| {
                            egui::Frame::NONE
                                .fill(egui::Color32::from_rgba_premultiplied(18, 18, 24, 200))
                                .corner_radius(egui::CornerRadius::same(4))
                                .inner_margin(egui::Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{:.0} FPS | {} splats | DLSS {}",
                                            fps, splat_count, dlss_mode,
                                        ))
                                        .color(egui::Color32::from_rgb(140, 145, 160))
                                        .size(11.0)
                                        .family(egui::FontFamily::Monospace),
                                    );
                                });
                        });
                });

                egui_state.handle_platform_output(window, full_output.platform_output);
                let tris = self.egui_ctx.tessellate(
                    full_output.shapes,
                    self.egui_ctx.pixels_per_point(),
                );
                for (id, image_delta) in &full_output.textures_delta.set {
                    egui_renderer.update_texture(
                        backend.device(),
                        backend.queue(),
                        *id,
                        image_delta,
                    );
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
                        color_attachments: &[Some(
                            wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            },
                        )],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    };
                    let mut render_pass = encoder
                        .begin_render_pass(&rp_desc)
                        .forget_lifetime();
                    egui_renderer.render(
                        &mut render_pass,
                        &tris,
                        &screen_descriptor,
                    );
                }

                backend.queue().submit(std::iter::once(encoder.finish()));

                for id in &full_output.textures_delta.free {
                    egui_renderer.free_texture(id);
                }
            }

            surface_tex.present();
        } else {
            // --- Software fallback render path (bitmap font HUD) ---
            let pixels = self.render_frame();
            if let Some(backend) = &self.backend {
                backend.present_framebuffer(&pixels, self.dlss.display_width, self.dlss.display_height);
            }
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
                    self.engine.time_of_day(),
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
                let gpu_rast = GpuRasteriser::new(
                    backend.device(),
                    backend.surface_format(),
                    DEFAULT_WIDTH,
                    DEFAULT_HEIGHT,
                );
                println!("[ochroma] GPU rasteriser created (primary render path)");
                self.gpu_rasteriser = Some(gpu_rast);
                self.backend = Some(backend);
            }
            Err(e) => {
                eprintln!("[ochroma] GPU init failed: {}", e);
                eprintln!("[ochroma] Falling back to software rasteriser");
            }
        }

        // Initialise egui on top of the GPU backend
        if let Some(backend) = &self.backend {
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
            apply_ochroma_theme(&self.egui_ctx);
            println!("[ochroma] egui overlay initialised (Ochroma 2026 theme)");
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
            self.engine.entity_count(),
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
        // Forward events to egui
        if let Some(egui_state) = &mut self.egui_state {
            if let Some(window) = &self.window {
                let _ = egui_state.on_window_event(window, &event);
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                self.engine.stop();
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => self.handle_keyboard_event(&event, event_loop),
            WindowEvent::ModifiersChanged(modifiers) => {
                self.ctrl_held = modifiers.state().control_key();
            }
            WindowEvent::MouseInput { state, button, .. } => self.handle_mouse_button(button, state),
            WindowEvent::CursorMoved { position, .. } => self.handle_mouse_move(position.x, position.y),
            WindowEvent::Resized(size) => self.handle_resize(size.width, size.height),
            WindowEvent::RedrawRequested => self.handle_redraw(event_loop),
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
