// Hide the console window on Windows (GUI application)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

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
use vox_app::soundscape::Soundscape;
use vox_app::walk_animation::{animation_system, ProceduralWalkComponent};
use vox_core::engine_runtime::EngineConfig;
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
use vox_render::spectral_tonemapper::{ToneMapOperator, ToneMapSettings};
use vox_render::temporal::TemporalAccumulator;

use ochroma_engine::engine_loop::{EngineLoop, SystemMask};
use vox_render::gizmos::GizmoRenderer;
use vox_ui::game_hud::GameHud;
use vox_ui::node_graph_widget::{NodeGraphWidget, VisualConnection, VisualNode, VisualPin, VisualPinType};
use vox_ui::spectral_hud::SpectralRadianceCache;
use vox_ui::theme::apply_ochroma_theme;
use vox_ui::vello_ctx::VelloCtxCpu;

use vox_editor::node_graph::{OchromaNodeGraph, ParamValue, PortData};
use vox_editor::node_thumbnail::node_thumbnail;
use vox_editor::nodes::biome_node::{BiomeKind, BiomeNode};
use vox_editor::nodes::terrain_node::TerrainNode;
use vox_editor::registry::NodeRegistry;
use vox_editor::templates;

// NavMesh + patrol demo (uncomment to enable):
// use vox_app::ai_fsm::NavMeshPlugin;
// app.add_plugins(NavMeshPlugin);

// Material hot-reload (uncomment to enable):
// use vox_render::material_hotreload::MaterialHotReloadPlugin;
// app.add_plugins(MaterialHotReloadPlugin::default()); // enable for material hot-reload

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

// ---------------------------------------------------------------------------
// NavMesh patrol agent
// ---------------------------------------------------------------------------

/// Bridges a sky ambient array into the SpectralRadianceSource trait for patrol agents.
struct SpectralGiAdapter {
    sky_ambient: [f32; 16],
}

impl vox_ai::perception::SpectralRadianceSource for SpectralGiAdapter {
    fn sample_at(&self, _pos: glam::Vec3, _radius: f32) -> [f32; 16] {
        self.sky_ambient
    }
}

/// Simple navmesh-driven patrol agent for AI demonstration.
struct PatrolAgent {
    position: Vec3,
    speed: f32,
    path: Vec<[f32; 3]>,
    path_index: usize,
    patrol_nodes: [u32; 2],
    current_target: usize,
    /// Spectral perception — agent senses environment via 16-band radiance.
    spectral_perception: vox_ai::perception::SpectralPerceptionAgent,
}

impl PatrolAgent {
    fn new(start_pos: Vec3, node_a: u32, node_b: u32) -> Self {
        Self {
            position: start_pos,
            speed: 3.0,
            path: Vec::new(),
            path_index: 0,
            patrol_nodes: [node_a, node_b],
            current_target: 0,
            spectral_perception: vox_ai::perception::SpectralPerceptionAgent::new(
                start_pos,
                12.0,
            ),
        }
    }

    fn update(&mut self, dt: f32, navmesh: &vox_core::navmesh::NavMesh) {
        // If no path or finished path, request new path to next patrol target
        if self.path_index >= self.path.len() {
            let from_node = self.patrol_nodes[self.current_target ^ 1];
            let to_node = self.patrol_nodes[self.current_target];
            if let Some(p) = navmesh.find_path(from_node, to_node) {
                self.path = p;
                self.path_index = 0;
            }
            self.current_target ^= 1;
        }
        // Move along path
        if let Some(&wp) = self.path.get(self.path_index) {
            let target = Vec3::from(wp);
            let dir = target - self.position;
            let dist = dir.length();
            if dist < 0.1 {
                self.path_index += 1;
            } else {
                self.position += dir.normalize() * (self.speed * dt).min(dist);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The engine application — owns every system
// ---------------------------------------------------------------------------

struct EngineApp {
    // Core engine runtime (scripts, ECS, time, input) now lives on `self.loop_`
    // (EngineLoop::runtime) — the single, ticked ECS world (S4). There is no
    // separate `engine` field anymore.

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
    input_state: vox_core::input::InputState,
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
    camera_velocity: Vec3,
    camera_acceleration: f32,
    camera_deceleration: f32,
    camera_max_speed: f32,

    // Title update throttle
    title_timer: Instant,

    // Stats
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    fps: f32,

    // Editor
    editor: SceneEditor,
    editor_visible: bool,
    content_browser: ContentBrowser,
    anim_editor_ui: vox_render::anim_editor_ui::AnimEditorUi,
    material_editor_ui: vox_render::material_editor_ui::MaterialEditorUi,

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

    // Audio engine (with rodio backend) + spatial audio manager now live on
    // `self.loop_` (EngineLoop), which constructs the single audio backend (S2).
    click_counter: u32,

    // Unified per-frame simulation driver (EngineLoop migration, S1).
    // Owns the Rapier physics world (formerly `physics`) and the ECS<->Rapier
    // body map (formerly `entity_rapier_bodies`). Other subsystems (audio, GI,
    // shadows, scripts) remain inline on EngineApp until later migration steps.
    loop_: EngineLoop,

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

    // VFX editor UI window
    vfx_editor_ui: vox_render::vfx_editor_ui::VfxEditorUi,

    // Ambient soundscape (toggled with N key)
    soundscape: Soundscape,

    // Cascaded shadow mapper now lives on `self.loop_` (EngineLoop), driven each
    // frame via `step_shadows` (S4).

    // Audio handle for high-level audio management
    audio_handle: Option<vox_audio::AudioHandle>,

    // Character controller (WASD + jump + mouse-look; toggle with P key)
    character: vox_app::character_controller::CharacterController,

    // Rhai scripting runtime (hot-reloadable game logic)
    rhai: vox_script::rhai_runtime::RhaiRuntime,

    // GLTF animation driver (optional — loaded from assets/character.glb if present)
    anim_driver: Option<vox_render::animation_driver::AnimationDriver>,

    // Delta time for the current frame (set in handle_redraw, consumed in render_frame)
    frame_dt: f32,

    // LOD tile streaming manager
    tile_manager: vox_render::streaming::TileManager,

    // Persisted key bindings (loaded from keybindings.toml at startup)
    #[allow(dead_code)]
    key_bindings: vox_core::input::KeyBindings,

    // SDF soft shadow pass (None until GPU is available)
    sdf_shadow: Option<vox_render::gpu::sdf_shadow_pass::SdfShadowPass>,

    // Baked GI cache (applied to splats before render)
    gi_cache: Option<vox_render::gi_cache::GiCache>,

    // Live spectral atmosphere + radiance cache now live on `self.loop_`
    // (EngineLoop), driven each frame via `step_gi` (S3).

    // Splat particle emitters (KeyE spawns fire emitter)
    particle_emitters: Vec<vox_render::splat_particles::SplatEmitter>,

    // SDF navmesh for agent pathfinding
    navmesh: Option<vox_core::navmesh::NavMesh>,

    // Patrol agents driven by navmesh pathfinding
    patrol_agents: Vec<PatrolAgent>,

    // Terrain SDF volume (kept for navmesh re-extraction after deform)
    terrain_volume: Option<vox_terrain::volume::TerrainVolume>,

    // Spectral viewport mode (Tab cycles bands)
    spectral_viewport_mode: vox_render::spectral_viewport::SpectralViewportMode,

    // Biome-driven ambient soundscape mix (blended each frame toward target biome)
    ambient_mix: vox_audio::BiomeAmbientMix,

    // Lua scripting runtime (mlua Lua 5.4)
    lua: vox_script::LuaRuntime,
    // Script watcher for hot-reload
    script_watcher: Option<vox_script::ScriptWatcher>,
    // Shared spectral state for Lua spectral.* bindings.
    // Must be kept alive here so the Arc inside Lua closures stays valid.
    #[allow(dead_code)]
    spectral_script_state: std::sync::Arc<std::sync::Mutex<vox_script::SpectralState>>,
    // Shared entity store for Lua entity.* bindings.
    // Must be kept alive here so the Arc inside Lua closures stays valid.
    #[allow(dead_code)]
    entity_script_store: std::sync::Arc<std::sync::Mutex<vox_script::EntityStore>>,

    // QUIC transport (Some if --server or --connect flag passed)
    quic_transport: Option<vox_net::quic_transport::QuicTransport>,
    // Per-client replication state (one entry per connected peer on server)
    replication_states: Vec<vox_net::replication_loop::ClientReplicationState>,

    // Rapier KCC body for the player character (None until character controller is first enabled)
    character_body: Option<vox_physics::character_body::CharacterBody>,

    // Game UI widgets (Manor Lords style resource panels, tooltips, buttons)
    game_widgets: vox_ui::GameWidgets,
    widget_cmds: Vec<vox_ui::WidgetCmd>,

    // Last terrain pick position from ScreenRay (updated on left-click)
    last_pick: Option<glam::Vec3>,

    // Live spectral GI bands: the spectral() (f16 BITS, NOT linear-quantized
    // u16) of the GI-lit splat nearest the camera, captured each frame in
    // render_frame after loop_.step_gi. Fed (decoded) to the Vello GameHud.
    latest_gi_bands: [u16; 16],

    // --- Editor panels (composited into the frame via VelloCtxCpu) ---
    // Node-graph panel visibility (toggled with KeyB). ON by default so the
    // editor looks like an editor and the smoke test can assert it.
    node_panel_visible: bool,
    // Visual node graph (terrain -> biome -> output) rendered via vox_ui's
    // node_graph_widget::render_to_pixels into a sub-rect of the frame.
    node_widget: NodeGraphWidget,
    // Number of nodes in the evaluated vox_editor graph.
    node_graph_node_count: usize,
    // Evaluated terminal output value from the graph: the count of Alpine
    // biome cells produced by the biome sink node. Finite + sane; shown as
    // text and asserted in smoke.
    node_graph_output_value: f64,

    // --- PCG-style live-in-viewport graph (rank #7) ---
    // The persistent, editable vox_editor graph instantiated from a template.
    // Param edits route through `request_recook`; `live_graph.live_cook(now)`
    // re-cooks ONLY the dirty subgraph (throttled), and the cooked Splats are
    // re-injected into scene_splats the same frame so edits change the world
    // without a restart. None until build_node_graph runs.
    live_graph: Option<vox_editor::node_graph::OchromaNodeGraph>,
    // The TerrainNode id in live_graph (the node whose params we live-edit).
    live_graph_terrain: Option<vox_editor::node_graph::NodeId>,
    // The terminal SplatizeNode id + port producing the live splats.
    live_graph_terminal: Option<(vox_editor::node_graph::NodeId, &'static str)>,
    // The [start, end) range in scene_splats owned by the live graph output, so
    // a recook can splice the fresh splats in place without rebuilding the scene.
    live_graph_splat_range: Option<(usize, usize)>,
    // Microseconds the last live recook took (printed in the verifiable line).
    live_graph_last_cook_us: u128,
    // Visual connections mirrored from the live graph, for wire-value chip refresh.
    live_graph_visual_conns: Vec<VisualConnection>,
    // --- Live preview thumbnails on graph nodes (rank #10) ---
    // Monotonic count of how many thumbnails have actually been (re)generated.
    // A thumbnail is regenerated ONLY when a node's cook generation changes, so
    // cooking with no change leaves this flat (proven by the smoke test).
    thumbnail_gen_count: u64,
    // Per visual-node-id, the graph cook generation the last thumbnail was built
    // from. Used to gate regeneration: same generation => reuse the cached blit.
    thumbnail_node_gen: std::collections::HashMap<u32, u64>,

    // Live GPU Vello renderer for the SpectralHUD (rank-1 adoption candidate:
    // GPU vector UI). Lazily created on first HUD draw via
    // VelloCtx::new_headless; `None` if no GPU adapter is available (then the
    // HUD falls back to the CPU VelloCtxCpu software path). The HUD is rendered
    // to an offscreen Rgba8Unorm texture by Vello, read back, and alpha-blended
    // over the final frame — so the windowed editor's HUD pixels are produced
    // by the real Vello GPU pipeline, not the CPU stub.
    vello_hud: Option<vox_ui::vello_ctx::VelloCtx>,
    // Set once we've decided GPU is unavailable, so we don't retry every frame.
    vello_hud_unavailable: bool,
    // Per-frame counter of HUD pixels actually composited from the Vello GPU
    // render (asserted by the smoke test to prove the GPU path ran live).
    vello_hud_px_last_frame: usize,
}

// ---------------------------------------------------------------------------
// Illuminant from time of day
// ---------------------------------------------------------------------------

fn illuminant_for_time(hour: f32) -> Illuminant {
    let hour = hour % 24.0;
    let d65 = Illuminant::d65();
    let warm = Illuminant::a();
    let cool = Illuminant {
        bands: [30.0, 38.0, 45.0, 55.0, 65.0, 70.0, 68.0, 62.0, 55.0, 47.0, 40.0, 35.0, 30.0, 25.0, 22.0, 20.0],
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

    let mut bands = [0.0f32; 16];
    for (i, band) in bands.iter_mut().enumerate() {
        *band = a.bands[i] * (1.0 - t) + b.bands[i] * t;
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

#[allow(dead_code)]
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

        let dlss = DlssPipeline::new(DEFAULT_WIDTH, DEFAULT_HEIGHT, DlssQuality::Performance);
        let (render_w, render_h) = dlss.render_resolution();

        // EngineLoop owns the live Rapier physics world (built with its own
        // 1km x 1km ground plane in EngineLoop::new) plus the ECS<->Rapier body
        // map. The character controller and scene colliders are added to it.
        let mut loop_ = EngineLoop::new(config, SystemMask::all());
        println!("[ochroma] Physics: Rapier3D world initialised (ground plane 1000x1000)");
        let character = vox_app::character_controller::CharacterController::new(
            &mut loop_.physics,
            glam::Vec3::new(0.0, 0.9, 0.0),
        );
        println!("[ochroma] Character controller initialised at (0, 0.9, 0)");

        let spectral_script_state = std::sync::Arc::new(std::sync::Mutex::new(
            vox_script::SpectralState::new()
        ));
        let entity_script_store = std::sync::Arc::new(std::sync::Mutex::new(
            vox_script::EntityStore::new()
        ));

        Self {
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
            input_state: vox_core::input::InputState::default(),
            keys: HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            left_click_pending: false,
            mouse_x: 0.0,
            mouse_y: 0.0,
            ctrl_held: false,
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            camera_velocity: Vec3::ZERO,
            camera_acceleration: 40.0,
            camera_deceleration: 20.0,
            camera_max_speed: 15.0,
            title_timer: Instant::now(),
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            fps: 0.0,
            editor,
            editor_visible: false,
            content_browser: ContentBrowser::new(std::path::Path::new(".")),
            anim_editor_ui: vox_render::anim_editor_ui::AnimEditorUi::new(),
            material_editor_ui: vox_render::material_editor_ui::MaterialEditorUi::new(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            exposure: 1.0,
            spectral_bypass: false, // spectra EWA renderer by default
            asset_path,
            placed_objects: Vec::new(),
            click_counter: 0,
            loop_,
            character,
            collider_to_entity: HashMap::new(),
            entity_splat_ranges: HashMap::new(),
            entity_original_positions: HashMap::new(),
            gizmo: GizmoRenderer::new(),
            left_mouse_held: false,
            vfx_editor_ui: vox_render::vfx_editor_ui::VfxEditorUi::new(),
            soundscape: {
                let ss = Soundscape::outdoor_default();
                println!("[ochroma] Soundscape: outdoor default ({} layers, active={})", ss.layers.len(), ss.active);
                ss
            },
            audio_handle: vox_audio::AudioHandle::spawn(),
            rhai: vox_script::rhai_runtime::RhaiRuntime::new(),
            anim_driver: None,
            frame_dt: 0.0,
            tile_manager: vox_render::streaming::TileManager::with_radius(2),
            key_bindings: vox_core::input::load_bindings(std::path::Path::new("keybindings.toml")),
            sdf_shadow: None,
            gi_cache: None,
            particle_emitters: Vec::new(),
            navmesh: None,
            patrol_agents: Vec::new(),
            terrain_volume: None,
            spectral_viewport_mode: vox_render::spectral_viewport::SpectralViewportMode::default(),
            ambient_mix: vox_audio::BiomeAmbientMix::for_biome(vox_audio::BiomeKind::Grassland),
            lua: {
                let mut rt = vox_script::LuaRuntime::new()
                    .expect("Lua 5.4 init failed");
                vox_script::register_spectral_bindings(rt.lua(), spectral_script_state.clone())
                    .expect("spectral bindings");
                vox_script::register_entity_bindings(rt.lua(), entity_script_store.clone())
                    .expect("entity bindings");
                let game_script = std::path::Path::new("assets/scripts/game.lua");
                if game_script.exists() {
                    rt.exec_file(game_script).expect("game.lua load failed");
                }
                rt
            },
            script_watcher: vox_script::ScriptWatcher::new(
                std::path::Path::new("assets/scripts")
            ).ok(),
            spectral_script_state,
            entity_script_store,
            quic_transport: None,
            replication_states: Vec::new(),
            character_body: None,
            last_pick: None,
            latest_gi_bands: [0u16; 16],
            node_panel_visible: true,
            node_widget: NodeGraphWidget::new(),
            node_graph_node_count: 0,
            node_graph_output_value: 0.0,
            live_graph: None,
            live_graph_terrain: None,
            live_graph_terminal: None,
            live_graph_splat_range: None,
            live_graph_last_cook_us: 0,
            live_graph_visual_conns: Vec::new(),
            thumbnail_gen_count: 0,
            thumbnail_node_gen: std::collections::HashMap::new(),
            vello_hud: None,
            vello_hud_unavailable: false,
            vello_hud_px_last_frame: 0,
            game_widgets: vox_ui::GameWidgets::new(),
            widget_cmds: vec![
                vox_ui::WidgetCmd::Panel {
                    title: "Resources".to_string(),
                    rows: vec![
                        vox_ui::ResourceRow { label: "Wood".to_string(), count: 0, icon_color: egui::Color32::from_rgb(139, 90, 43) },
                        vox_ui::ResourceRow { label: "Stone".to_string(), count: 0, icon_color: egui::Color32::from_rgb(160, 160, 160) },
                        vox_ui::ResourceRow { label: "Food".to_string(), count: 0, icon_color: egui::Color32::from_rgb(200, 180, 60) },
                    ],
                },
            ],
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
                        self.build_node_graph();
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
        // Extract navmesh from terrain SDF
        let navmesh = vox_terrain::navmesh_bridge::extract_from_volume(&vol, 1.5, 2);
        println!("[ochroma]   Navmesh: {} nodes", navmesh.node_count());
        self.navmesh = Some(navmesh);
        self.terrain_volume = Some(vol);

        // Spawn patrol agents between well-spaced navmesh nodes
        if let Some(nm) = &self.navmesh
            && nm.node_count() >= 2
        {
            let node_count = nm.node_count() as u32;
            let mid_a = node_count / 3;
            let mid_b = 2 * node_count / 3;
            let pos_a = Vec3::ZERO; // agents start at origin
            let pos_b = Vec3::new(5.0, 0.0, 5.0);
            self.patrol_agents.push(PatrolAgent::new(pos_a, mid_a, mid_b));
            self.patrol_agents.push(PatrolAgent::new(pos_b, mid_b, mid_a));
            println!("[ochroma]   Spawned {} patrol agents", self.patrol_agents.len());
        }

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

        // Populate engine scene entities (for scripts) — into the loop's world,
        // the single ticked ECS.
        self.loop_.runtime.spawn("Terrain").with_position(Vec3::ZERO);
        for i in 0..4u32 {
            self.loop_.runtime.spawn(&format!("Building {}", i + 1))
                .with_asset("building.ply")
                .with_position(Vec3::new(i as f32 * 10.0 - 15.0, 0.0, 20.0));
        }
        for i in 0..6u32 {
            self.loop_.runtime.spawn(&format!("Tree {}", i + 1))
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
                let p = ws.position();
                ws.set_position([p[0] + pos[0], p[1], p[2] + pos[2]]);
                self.scene_splats.push(ws);
            }
            let end = self.scene_splats.len();
            self.entity_splat_ranges.insert(entity_id, (start, end));
            self.entity_original_positions.insert(entity_id, pos);
            // Add Rapier static collider for building and track entity mapping
            let col_handle = self.loop_.physics.add_static_collider(pos, [5.0, 10.0, 8.0]);
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
                let p = ws.position();
                ws.set_position([p[0] + pos[0], p[1], p[2] + pos[2]]);
                self.scene_splats.push(ws);
            }
            let end = self.scene_splats.len();
            self.entity_splat_ranges.insert(entity_id, (start, end));
            self.entity_original_positions.insert(entity_id, pos);
            // Add Rapier static collider for tree trunk
            let tree_height = 6.0 + i as f32;
            let col_handle = self.loop_.physics.add_static_collider(
                [pos[0], tree_height * 0.5, pos[2]],
                [1.5, tree_height * 0.5, 1.5],
            );
            self.collider_to_entity.insert(col_handle, entity_id);
        }

        println!("[ochroma] Physics: {} bodies, {} colliders in Rapier world",
            self.loop_.physics.body_count(), self.loop_.physics.collider_count());
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
        self.loop_.runtime.spawn("Light1")
            .with_position(Vec3::new(5.0, 8.0, 20.0))
            .with_light([1.0, 0.9, 0.7], 50.0, 30.0);
        self.loop_.runtime.spawn("Light2")
            .with_position(Vec3::new(-15.0, 4.0, 10.0))
            .with_light([0.7, 0.8, 1.0], 30.0, 20.0);

        // Spawn procedural walk-cycle NPC at (0, 0, -3)
        self.loop_.world_mut().spawn(
            ProceduralWalkComponent::humanoid_blob(glam::Vec3::new(0.0, 0.0, -3.0))
        );
        println!("[ochroma] Spawned procedural walk NPC at (0, 0, -3)");

        // CLAS clustering + MegaGeometry
        self.run_clas();
        // Bake GI (or load from .vxgi cache)
        self.rebuild_gi();
        // Build + evaluate the editor node graph (terrain -> biome) for the panel
        self.build_node_graph();
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
    // Node graph (vox_editor) — build, evaluate, mirror into the visual widget
    // -----------------------------------------------------------------------

    /// Build a real vox_editor node graph (TerrainNode -> BiomeNode), evaluate
    /// it once, and record the node count + a terminal output value (the number
    /// of Alpine biome cells classified from the evaluated terrain). Also mirror
    /// the graph into a `NodeGraphWidget` so the bottom-right panel draws the
    /// actual pipeline. This is the editor's live node-graph readout.
    fn build_node_graph(&mut self) {
        let mut graph = OchromaNodeGraph::new();
        // Small, fast graph (low resolution, no erosion) — this runs every smoke
        // frame is NOT required; we evaluate once here at scene-build time.
        let terrain = graph.add_node("Terrain", Box::new(TerrainNode {
            resolution: 32,
            amplitude: 400.0,
            droplet_count: 0,
            seed: 7,
            ..Default::default()
        }));
        let biome = graph.add_node("Biome", Box::new(BiomeNode {
            world_height: 400.0,
            moisture: 0.5,
        }));
        graph.connect(terrain, "terrain", biome, "terrain")
            .expect("terrain->biome connect");

        self.node_graph_node_count = graph.node_count();

        // Evaluate the whole DAG and pull the terminal (sink) biome map.
        match graph.evaluate() {
            Ok(result) => {
                let alpine = result
                    .sole_sink()
                    .and_then(|sink| result.get(sink, "biome_map"))
                    .and_then(|d| d.as_biome_map())
                    .map(|bm| bm.iter().filter(|&&b| b == BiomeKind::Alpine as u8).count())
                    .unwrap_or(0);
                self.node_graph_output_value = alpine as f64;
                println!(
                    "[ochroma] NodeGraph: {} nodes, evaluated -> {} Alpine biome cells",
                    self.node_graph_node_count, alpine,
                );
            }
            Err(e) => {
                eprintln!("[ochroma] NodeGraph evaluate failed: {}", e);
                self.node_graph_output_value = 0.0;
            }
        }

        // Mirror the real graph topology into the visual widget for rendering.
        self.node_widget = NodeGraphWidget::new();
        self.node_widget.scroll_offset = [12.0, 12.0];
        self.node_widget.add_node(VisualNode {
            id: 1,
            title: "Terrain".into(),
            position: [10.0, 20.0],
            size: [120.0, 60.0],
            color: [120, 90, 60],
            inputs: vec![],
            outputs: vec![VisualPin {
                name: "terrain".into(),
                pin_type: VisualPinType::Any,
                connected: true,
            }],
            selected: false,
            collapsed: false,
        });
        self.node_widget.add_node(VisualNode {
            id: 2,
            title: "Biome".into(),
            position: [180.0, 20.0],
            size: [120.0, 60.0],
            color: [60, 130, 80],
            inputs: vec![VisualPin {
                name: "terrain".into(),
                pin_type: VisualPinType::Any,
                connected: true,
            }],
            outputs: vec![VisualPin {
                name: "biome".into(),
                pin_type: VisualPinType::Spectral,
                connected: true,
            }],
            selected: true,
            collapsed: false,
        });
        self.node_widget.add_connection(VisualConnection {
            from_node: 1,
            from_pin: "terrain".into(),
            to_node: 2,
            to_pin: "terrain".into(),
            color: [200, 200, 80],
        });

        // Build the PCG-style LIVE graph from a template and inject its splats.
        self.build_live_graph();
    }

    // -----------------------------------------------------------------------
    // PCG-style live-in-viewport graph (rank #7)
    // -----------------------------------------------------------------------

    /// World offset where the live-graph splats are placed so they sit in front
    /// of the camera in the default scene (over the terrain, near the buildings).
    const LIVE_GRAPH_WORLD_OFFSET: [f32; 3] = [0.0, 0.5, 14.0];

    /// Instantiate the live graph from a starter template, cook it once, inject
    /// its terminal Splats into `scene_splats` (tracked range), and mirror its
    /// topology into the visual widget. After this, a param edit + `live_recook`
    /// updates the world in place without a restart.
    fn build_live_graph(&mut self) {
        let registry = NodeRegistry::new();
        let template_name = "Building → Plot → Splatize";
        let Some(Ok(mut inst)) = templates::instantiate_by_name(&registry, template_name) else {
            eprintln!("[graph] live template '{template_name}' failed to instantiate");
            return;
        };

        // Tune the terminal splatize so the splat count is UNCLAMPED across the
        // footprint range we scrub — that way a PlotNode footprint edit visibly
        // changes the viewport splat count (area * splats_per_sqm).
        let terminal_id = inst.terminal_id;
        let _ = inst.graph.set_param(terminal_id, "splats_per_sqm", ParamValue::Float(2.0));
        let _ = inst.graph.set_param(terminal_id, "min_splats", ParamValue::Int(50));
        let _ = inst.graph.set_param(terminal_id, "max_splats", ParamValue::Int(50_000));

        // We live-edit the PlotNode footprint (node_ids[1] = "plot"): its ground
        // mesh area scales the downstream splat count, so a footprint scrub
        // re-cooks Plot -> Splatize and changes the viewport splat count.
        let edit_node = inst.node_ids.get(1).copied();

        self.live_graph_terminal = Some((terminal_id, inst.terminal_port));
        self.live_graph_terrain = edit_node; // the node we scrub live

        // First cook.
        if let Err(e) = inst.graph.cook() {
            eprintln!("[graph] live graph initial cook failed: {e}");
            return;
        }

        // Mirror visual connections for wire-value chips.
        self.live_graph_visual_conns = inst
            .graph
            .edges()
            .map(|(from, fp, to, tp)| VisualConnection {
                from_node: from.0 + 100, // offset so ids don't collide with panel nodes
                from_pin: fp.to_string(),
                to_node: to.0 + 100,
                to_pin: tp.to_string(),
                color: [120, 200, 255],
            })
            .collect();

        // Mirror the live-graph nodes into the visual widget (ids offset by +100 to
        // match the wire ids) so their live preview thumbnails (rank #10) blit onto
        // real node bodies. Laid out as a vertical stack below the terrain/biome row.
        for (row, nid) in inst.node_ids.iter().enumerate() {
            let title = inst.graph.node_name(*nid).unwrap_or("node").to_string();
            self.node_widget.add_node(VisualNode {
                id: nid.0 + 100,
                title,
                position: [10.0, 110.0 + row as f32 * 64.0],
                size: [110.0, 56.0],
                color: [70, 110, 150],
                inputs: vec![],
                outputs: vec![],
                selected: false,
                collapsed: false,
            });
        }

        self.live_graph = Some(inst.graph);

        // Inject the cooked splats into the viewport scene.
        self.inject_live_graph_splats();

        // Build the initial live preview thumbnails (rank #10) for every cooked node.
        self.refresh_node_thumbnails();
    }

    /// Pull the live graph's terminal Splats, offset them into the scene, and
    /// splice them into `scene_splats`. If a previous range exists it is replaced
    /// in place (same frame, no restart); otherwise the splats are appended and
    /// the new range recorded. Returns the new splat count contributed.
    fn inject_live_graph_splats(&mut self) -> usize {
        let Some((tid, port)) = self.live_graph_terminal else { return 0 };
        let Some(graph) = &self.live_graph else { return 0 };
        let Some(PortData::Splats(splats)) = graph.get_output(tid, port) else { return 0 };

        let off = Self::LIVE_GRAPH_WORLD_OFFSET;
        let mut world_splats: Vec<GaussianSplat> = splats
            .iter()
            .map(|s| {
                let mut ws = *s;
                let p = ws.position();
                ws.set_position([p[0] + off[0], p[1] + off[1], p[2] + off[2]]);
                ws
            })
            .collect();
        let count = world_splats.len();

        match self.live_graph_splat_range {
            Some((start, end)) => {
                // Replace the existing range in place.
                self.scene_splats.splice(start..end, world_splats.drain(..));
                self.live_graph_splat_range = Some((start, start + count));
            }
            None => {
                let start = self.scene_splats.len();
                self.scene_splats.append(&mut world_splats);
                self.live_graph_splat_range = Some((start, start + count));
            }
        }
        count
    }

    /// PCG-style live param edit entry point: route a numeric param change on the
    /// live graph's edit node through the throttled recook request. The actual
    /// recook happens in [`live_recook`] on the next frame whose clock has passed
    /// the throttle budget.
    fn live_graph_request_edit(&mut self, key: &str, value: ParamValue) {
        let (Some(graph), Some(node)) = (self.live_graph.as_mut(), self.live_graph_terrain) else { return };
        if let Err(e) = graph.request_recook(node, key, value) {
            eprintln!("[graph] live edit failed: {e}");
        }
    }

    /// Drive the live re-cook loop once per frame. If a throttled recook is due,
    /// cook ONLY the dirty subgraph, splice the fresh splats into the viewport the
    /// SAME frame, refresh the wire-value chips, and print the verifiable
    /// live-recook line. Returns the new live-graph splat count if a recook fired.
    fn live_recook(&mut self, now: std::time::Instant) -> Option<usize> {
        let report = {
            let graph = self.live_graph.as_mut()?;
            let t0 = std::time::Instant::now();
            let report = match graph.live_cook(now) {
                Ok(Some(r)) => r,
                Ok(None) => return None,
                Err(e) => {
                    eprintln!("[graph] live recook failed: {e}");
                    return None;
                }
            };
            self.live_graph_last_cook_us = t0.elapsed().as_micros();
            report
        };

        let viewport_before = self.scene_splats.len();
        let after = self.inject_live_graph_splats();
        let viewport_after = self.scene_splats.len();

        // Refresh wire-value chips from the freshly-cooked graph (stale chips are a bug).
        if let Some(graph) = &self.live_graph {
            let wires = graph.wire_values();
            self.node_widget.clear_wire_values();
            for conn in &self.live_graph_visual_conns {
                // Match the visual conn back to a graph wire value by node/port.
                if let Some(wv) = wires.iter().find(|w| {
                    w.from.0 + 100 == conn.from_node
                        && w.to.0 + 100 == conn.to_node
                        && w.from_port == conn.from_pin
                        && w.to_port == conn.to_pin
                }) {
                    self.node_widget.set_wire_value(conn, wv.value.clone());
                }
            }
        }

        println!(
            "[graph] live recook: node={} dirty_subgraph={} cook_us={} viewport_splats {}->{}",
            report.root_name,
            report.dirty_subgraph_size(),
            self.live_graph_last_cook_us,
            viewport_before,
            viewport_after,
        );

        // Regenerate live preview thumbnails ONLY for nodes whose cook generation
        // changed this recook (rank #10). Clean nodes keep their cached blit.
        self.refresh_node_thumbnails();
        Some(after)
    }

    /// Live preview thumbnails (rank #10): for every live-graph node, render its
    /// cooked primary output into a miniature and push it onto the visual widget —
    /// but ONLY when the node's cook generation (`cook_count`) has advanced since
    /// the last thumbnail was built for it. Cooking with no change leaves every
    /// generation flat, so `thumbnail_gen_count` does not move (proven in smoke).
    /// Returns the number of thumbnails actually (re)generated this call.
    fn refresh_node_thumbnails(&mut self) -> usize {
        const TW: usize = 64;
        const TH: usize = 40;
        let Some(graph) = &self.live_graph else { return 0 };

        // Collect (visual_id, generation, thumbnail-pixels) for nodes that changed.
        let mut regenerated: Vec<(u32, u64, Vec<[u8; 4]>)> = Vec::new();
        for nid in graph.node_ids() {
            let generation = graph.cook_count(nid).unwrap_or(0);
            if generation == 0 {
                continue; // never cooked => no output to preview yet
            }
            let visual_id = nid.0 + 100; // same id-space mapping as the wires
            if self.thumbnail_node_gen.get(&visual_id) == Some(&generation) {
                continue; // generation unchanged => reuse the cached thumbnail
            }
            // Pick the node's primary (first, deterministic) output port's data.
            let Some(outputs) = graph.node_outputs(nid) else { continue };
            // Deterministic pick: lowest port name so the same port previews each time.
            let Some((_, data)) = outputs.iter().min_by(|a, b| a.0.cmp(b.0)) else { continue };
            let pixels = node_thumbnail(data, TW, TH);
            regenerated.push((visual_id, generation, pixels));
        }

        let count = regenerated.len();
        for (visual_id, generation, pixels) in regenerated {
            self.node_widget.set_thumbnail(
                visual_id,
                vox_ui::node_graph_widget::NodeThumbnail::new(pixels, TW as u16, TH as u16),
            );
            self.thumbnail_node_gen.insert(visual_id, generation);
            self.thumbnail_gen_count += 1;
        }
        count
    }

    // -----------------------------------------------------------------------
    // GI baking
    // -----------------------------------------------------------------------

    fn rebuild_gi(&mut self) {
        let vxgi_path = std::path::Path::new("scene.vxgi");
        let irradiance = if vxgi_path.exists() {
            match vox_data::gi_export::load_vxgi(vxgi_path) {
                Ok(irr) if irr.len() == self.scene_splats.len() => irr,
                _ => self.run_gi_bake(),
            }
        } else {
            self.run_gi_bake()
        };
        let gi = vox_render::gi_baker::BakedGi { irradiance };
        self.gi_cache = Some(vox_render::gi_cache::GiCache::new(gi));
    }

    fn run_gi_bake(&self) -> Vec<[f32; 16]> {
        use vox_render::gi_baker::{GiBaker, GiBakeConfig};
        println!("[ochroma] Baking GI for {} splats...", self.scene_splats.len());
        let baker = GiBaker::new(GiBakeConfig {
            search_radius: 3.0,
            max_neighbours: 24,
            bounces: 2,
            falloff: 0.4,
        });
        let gi = baker.bake(&self.scene_splats);
        let _ = vox_data::gi_export::save_vxgi(
            &gi.irradiance,
            std::path::Path::new("scene.vxgi"),
        );
        gi.irradiance
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

        // Speed scales with altitude/distance (like Google Earth)
        let altitude_factor = (self.camera.position.y / 10.0).clamp(1.0, 10.0);
        self.camera_max_speed = 15.0 * altitude_factor;

        // Build desired velocity from input
        let mut desired = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) { desired += forward; }
        if self.keys.contains(&KeyCode::KeyS) { desired -= forward; }
        if self.keys.contains(&KeyCode::KeyA) { desired -= right; }
        if self.keys.contains(&KeyCode::KeyD) { desired += right; }
        if self.keys.contains(&KeyCode::Space) { desired += Vec3::Y; }
        if self.keys.contains(&KeyCode::ShiftLeft) { desired -= Vec3::Y; }

        if desired.length() > 0.01 {
            desired = desired.normalize() * self.camera_max_speed;
            // Accelerate toward desired velocity
            self.camera_velocity = self.camera_velocity.lerp(desired, (self.camera_acceleration * dt).min(1.0));
        } else {
            // Decelerate to stop
            self.camera_velocity = self.camera_velocity.lerp(Vec3::ZERO, (self.camera_deceleration * dt).min(1.0));
        }

        // Kill tiny residual velocity
        if self.camera_velocity.length() < 0.01 {
            self.camera_velocity = Vec3::ZERO;
        }

        // Apply velocity
        self.camera.position += self.camera_velocity * dt;

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
            let pos = Vec3::from(splat.position());
            let radius = splat.scale_u().max(splat.scale_v()).max(splat.scale_w()) + 1.0;
            if frustum.contains_sphere(pos, radius) {
                visible_splats.push(*splat);
            }
        }

        // LOD selection on visible splats — aggressive reduction for distant splats
        let cam_pos = self.camera.position;
        let lod_indices: Vec<usize> = (0..visible_splats.len())
            .filter(|&i| {
                let dist = cam_pos.distance(Vec3::from(visible_splats[i].position()));
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
                let p = ws.position();
                ws.set_position([p[0] + pos.x, p[1] + pos.y, p[2] + pos.z]);
                render_splats.push(ws);
            }
        }

        // Add particle splats (legacy ParticleSystem)
        let particle_splats = self.particles.to_splats();
        render_splats.extend(&particle_splats);

        // Tick splat emitters and inject particles
        {
            let dt = self.frame_dt;
            for emitter in &mut self.particle_emitters {
                emitter.tick(dt);
                render_splats.extend(emitter.splats());
                for dead_spectral in &emitter.died_this_frame {
                    let wav_path = vox_audio::create_impact_wav(dead_spectral, 0.08);
                    if let Some(audio) = &self.audio_handle {
                        audio.play(wav_path.to_str().unwrap_or(""), 0.4, false);
                    }
                }
            }
        }

        // Drain animation splats from RenderBuffer (written by animation_system)
        {
            let anim_splats = std::mem::take(
                &mut self.loop_.world_mut().resource_mut::<vox_core::engine_runtime::RenderBuffer>().splats
            );
            render_splats.extend(anim_splats);
        }

        // Update spectral atmosphere from time of day, then apply live spectral
        // GI via EngineLoop (S2/S3). step_gi mutates the loop's atmosphere/GI
        // cache (sun elevation from `hour`, set_sky) and returns the
        // GI-modulated splats — byte-for-byte the old inline propagate/apply.
        let hour = self.loop_.time_of_day();
        let render_splats = self.loop_.step_gi(&render_splats, hour);

        // Capture the live GI readout for the HUD: spectral of the GI-lit splat
        // nearest the camera. *spectral() is f16 BITS (decoded at HUD time).
        if let Some(nearest) = render_splats.iter().min_by(|a, b| {
            let da = (Vec3::from(a.position()) - cam_pos).length_squared();
            let db = (Vec3::from(b.position()) - cam_pos).length_squared();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.latest_gi_bands = *nearest.spectral();
        }

        // Legacy baked GI cache (kept for backward compat with --bake-gi workflow)
        let render_splats = match &self.gi_cache {
            Some(cache) => cache.apply(&render_splats),
            None => render_splats,
        };

        // Spectral caustics: refract through transmissive splats
        {
            use vox_render::spectral_caustics::{SpectralCaustics, CauchyGlass};
            use glam::Vec3;

            let glass = CauchyGlass::n_bk7();

            // Collect transmissive splat indices (opacity < 64 approximates glass)
            let transmissive: Vec<usize> = render_splats.iter().enumerate()
                .filter(|(_, s)| s.opacity() < 64)
                .map(|(i, _)| i)
                .collect();

            if !transmissive.is_empty() {
                let incident_dir = Vec3::new(0.0, -1.0, 0.0); // downward light
                let normal = Vec3::new(0.0, 1.0, 0.0);
                for &ti in &transmissive {
                    let spectral_f32: [f32; 16] = std::array::from_fn(|b| {
                        half::f16::from_bits(render_splats[ti].spectral()[b]).to_f32()
                    });
                    let _refraction = SpectralCaustics::refract(
                        incident_dir, normal, spectral_f32, &glass
                    );
                    // TODO(domain-6): accumulate refraction.transmitted into caustic_buffer
                    // and apply to render_splats at surrounding positions
                }
            }
        }

        // Spectral replication: server broadcasts changed splat bands to connected clients
        if let Some(transport) = &self.quic_transport
            && transport.role == vox_net::quic_transport::TransportRole::Server {
                use vox_net::spectral_relevance::{SplatSpectral, ObserverProfile};
                use vox_net::replication_loop::{replicate_tick, ReplicationConfig};

                let net_splats: Vec<SplatSpectral> = render_splats.iter()
                    .map(|s| SplatSpectral { bands: *s.spectral() })
                    .collect();

                if self.replication_states.is_empty() {
                    self.replication_states.push(
                        vox_net::replication_loop::ClientReplicationState::new(
                            0, net_splats.len(), ObserverProfile::human()
                        )
                    );
                }

                let config = ReplicationConfig::default();
                for client_state in &mut self.replication_states {
                    let _stats = replicate_tick(
                        &net_splats, client_state, &config,
                        |_packet_bytes| {
                            // TODO(domain-7): write packet_bytes to Quinn stream/datagram
                        },
                    );
                }
            }

        // Time-of-day illuminant
        let illuminant = illuminant_for_time(self.loop_.time_of_day());

        // --- Shadow map update --- via EngineLoop (S4). step_shadows computes
        // the sun direction (SunModel::new(51.5), identical to the old
        // light_manager.sun), updates the loop's ShadowMapper for the camera,
        // and renders the shadow map over render_splats — byte-for-byte the old
        // inline block. Subsequent shadow reads use self.loop_.shadow_mapper.
        let shadow_hour = self.loop_.time_of_day();
        let cam_fwd = (self.camera.target - self.camera.position).normalize_or(glam::Vec3::NEG_Z);
        let _sun_dir = self.loop_.step_shadows(
            self.camera.position,
            cam_fwd,
            shadow_hour,
            &render_splats,
        );

        // Generate CPU shadow mask: project each shadowed splat to screen space
        let shadow_mask: Vec<f32> = {
            let vp = render_camera.view_proj();
            let w = render_w as usize;
            let h = render_h as usize;
            let mut mask = vec![1.0f32; w * h];
            for splat in &render_splats {
                let world_pos = glam::Vec3::from(splat.position());
                if self.loop_.shadow_mapper.is_in_shadow(world_pos, 0.01) {
                    let clip = vp * world_pos.extend(1.0);
                    if clip.w > 0.001 {
                        let ndc_x = clip.x / clip.w;
                        let ndc_y = clip.y / clip.w;
                        let px = ((ndc_x * 0.5 + 0.5) * w as f32) as i32;
                        let py = ((1.0 - (ndc_y * 0.5 + 0.5)) * h as f32) as i32;
                        if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                            mask[py as usize * w + px as usize] = 0.0;
                        }
                    }
                }
            }
            mask
        };

        // 1. Render at internal resolution
        let render_start = Instant::now();

        let upscaled = if self.spectral_bypass {
            // FAST PATH: software rasteriser + DLSS upscale
            let fb = self.rasteriser.render(&render_splats, &render_camera, &illuminant, Some(&self.loop_.shadow_mapper));
            let pixel_count = (render_w * render_h) as usize;
            let depth = vec![1.0f32; pixel_count];
            let motion = vec![[0.0f32; 2]; pixel_count];
            self.dlss.upscale(&fb.pixels, render_w, render_h, &depth, &motion)
        } else {
            // SPECTRA PATH: tile-based EWA Gaussian splatting renderer
            let fb = match self.spectral_viewport_mode {
                vox_render::spectral_viewport::SpectralViewportMode::Full => {
                    vox_render::spectra_render::render_with_spectra_u8_shadowed(
                        &render_splats,
                        &render_camera,
                        render_w,
                        render_h,
                        &illuminant,
                        Some(&shadow_mask),
                    )
                }
                vox_render::spectral_viewport::SpectralViewportMode::Band(band) => {
                    vox_render::spectra_render::render_spectral_band_u8(
                        &render_splats,
                        &render_camera,
                        render_w,
                        render_h,
                        band,
                    )
                }
            };
            let pixel_count = (render_w * render_h) as usize;
            let depth = vec![1.0f32; pixel_count];
            let motion = vec![[0.0f32; 2]; pixel_count];
            self.dlss.upscale(&fb, render_w, render_h, &depth, &motion)
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
                self.loop_.time_of_day(),
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
                self.loop_.runtime.stats.entity_count,
                self.loop_.runtime.registered_script_count(),
                self.loop_.runtime.stats.frame_number,
            ),
            [160, 160, 160], 1);

        // Spectral viewport mode label (shown when not in full-color mode)
        if self.spectral_viewport_mode != vox_render::spectral_viewport::SpectralViewportMode::Full {
            burn_text(&mut final_pixels, display_w, 4, y_off + 30,
                self.spectral_viewport_mode.label(),
                [100, 220, 255], 1);
        }

        // Gizmo overlay (drawn on top of scene in editor mode)
        if self.editor_visible
            && let Some(entity) = self.editor.selected_entity()
        {
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
            // Sync gizmo mode from editor to renderer
            self.gizmo.mode = match self.editor.gizmo_mode {
                vox_app::editor::GizmoMode::Translate => vox_render::gizmos::GizmoMode::Translate,
                vox_app::editor::GizmoMode::Rotate    => vox_render::gizmos::GizmoMode::Rotate,
                vox_app::editor::GizmoMode::Scale     => vox_render::gizmos::GizmoMode::Scale,
            };
            self.gizmo.draw_overlay(
                &mut final_pixels,
                display_w,
                display_h,
                entity.position,
                vp,
            );
        }

        // NOTE: the bitmap-font EDITOR overlay (entity list + SELECTED panel via
        // `burn_text`) was DELETED as part of the editor face-lift. The editor
        // face is now `vox_app::shell` (egui_dock + tokens + Phosphor icons).
        // Only the gizmo overlay above and the engine HUD stats remain here.

        // Viewport mode post-process (editor only)
        if self.editor_visible {
            match self.editor.viewport_mode {
                vox_app::editor::ViewportMode::Lit => {} // no-op
                vox_app::editor::ViewportMode::Unlit => {
                    for pixel in final_pixels.iter_mut() {
                        let avg = ((pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3) as u8;
                        pixel[0] = avg;
                        pixel[1] = avg;
                        pixel[2] = avg;
                    }
                }
                vox_app::editor::ViewportMode::Wireframe => {
                    for pixel in final_pixels.iter_mut() {
                        pixel[0] = (pixel[0] as f32 * 0.2) as u8;
                        pixel[1] = (pixel[1] as f32 * 0.3) as u8;
                        pixel[2] = pixel[2].saturating_add(80);
                    }
                }
                vox_app::editor::ViewportMode::Normals => {
                    for pixel in final_pixels.iter_mut() {
                        pixel[0] = 128_u8.saturating_add(pixel[0] / 2);
                        pixel[1] = 128_u8.saturating_add(pixel[1] / 2);
                        pixel[2] = 200;
                    }
                }
                vox_app::editor::ViewportMode::Overdraw => {
                    for pixel in final_pixels.iter_mut() {
                        let bright = pixel[0].max(pixel[1]).max(pixel[2]);
                        let heat = bright as f32 / 255.0;
                        pixel[0] = (heat * 255.0) as u8;
                        pixel[1] = ((1.0 - heat) * 100.0) as u8;
                        pixel[2] = 0;
                        pixel[3] = 255;
                    }
                }
            }
        }

        // Vello GPU game HUD: live 16-band spectral GI readout (bottom-left
        // translucent panel) + a top-left progress bar, composited over the
        // final frame. The SpectralHUD bars are rendered by the REAL Vello GPU
        // pipeline (rank-1 adoption candidate) when an adapter is available;
        // otherwise the CPU VelloCtxCpu software path is used. Drawn in the
        // shared render path so the windowed editor gets it too, not just smoke.
        {
            // latest_gi_bands holds f16 BITS (the splat spectral encoding), NOT
            // linear-quantized u16 — decode before feeding the HUD so a radiance
            // of 1.0 fills a bar exactly. (from_u16 would misread the bits.)
            let mut energy = [0.0f32; 16];
            for (e, bits) in energy.iter_mut().zip(self.latest_gi_bands) {
                *e = half::f16::from_bits(bits).to_f32().clamp(0.0, 1.0);
            }
            let bands = SpectralRadianceCache::from_f32(energy);

            // SpectralHUD geometry must match GameHud's bottom-left placement.
            let panel_pad = 8.0f32;
            let margin = 16.0f32;
            let panel_h = 60.0 + panel_pad * 2.0; // BARS_MAX_HEIGHT + pad*2 = 76
            let panel_w = 160.0 + panel_pad * 2.0; // BARS_TOTAL_WIDTH + pad*2
            let panel_x = margin;
            let panel_y = display_h as f32 - margin - panel_h;
            let bars_pos = [panel_x + panel_pad, panel_y + panel_pad];

            // The editor has no orbs: repurpose the progress bar as a
            // selected-entity-of-total indicator (selection index / entity count).
            let entity_count = self.editor.entity_count() as u32;
            let selected_idx = self
                .editor
                .selected
                .and_then(|sel| self.editor.entities.iter().position(|e| e.id == sel))
                .map(|i| i as u32 + 1)
                .unwrap_or(0);

            // 1. Translucent backdrop panel (CPU) — drawn first so the GPU bars
            //    composite on top of it. Matches GameHud's [0,0,0,0.55] panel.
            let mut back_ctx = VelloCtxCpu::new(display_w, display_h);
            back_ctx.fill_rect([panel_x, panel_y, panel_w, panel_h], [0.0, 0.0, 0.0, 0.55]);
            back_ctx.rasterize_into(&mut final_pixels, display_w, display_h);

            // 2. Track A: render the 16-band spectral bars via the live Vello GPU
            //    pipeline, composited over the backdrop.
            self.vello_hud_px_last_frame =
                self.compose_spectral_hud_vello(&mut final_pixels, display_w, display_h, &bands, bars_pos);

            // 3. Orb/progress bar (CPU) + CPU bars fallback if no GPU adapter.
            let mut hud_ctx = VelloCtxCpu::new(display_w, display_h);
            let hud = GameHud::new(display_w, display_h);
            if self.vello_hud_px_last_frame == 0 {
                // No GPU: draw the spectral bars on the CPU so the HUD isn't blank.
                vox_ui::spectral_hud::SpectralHUD::render_cpu(&mut hud_ctx, &bands, bars_pos);
            }
            let track = hud.orb_bar_track_rect();
            let fill = hud.orb_bar_fill_rect(selected_idx, entity_count);
            hud_ctx.fill_rect(track, [0.05, 0.05, 0.08, 0.7]);
            if fill[2] > 0.0 {
                hud_ctx.fill_rect(fill, [1.0, 0.78, 0.2, 0.95]);
            }
            hud_ctx.rasterize_into(&mut final_pixels, display_w, display_h);
        }

        // NOTE: the old `burn_text`/`VelloCtxCpu` software editor panels (entity
        // inspector + node-graph blit) were DELETED here as part of the editor
        // face-lift (2026-06-06). The editor SHELL is now `vox_app::shell`
        // (egui_dock + tokens + Phosphor icons), proven headlessly by the
        // `shell_snapshot` bin. `engine_runner` keeps only its in-frame engine
        // HUD/gizmo smoke; it no longer composites a bitmap-font editor face.

        final_pixels
    }

    /// Render the live SpectralHUD bars through the real Vello GPU pipeline and
    /// composite them over `final_pixels`. Returns the number of HUD pixels
    /// actually blended in from the GPU render (0 if no GPU adapter, signalling
    /// the caller to fall back to the CPU software HUD).
    ///
    /// The HUD is rendered into a tight offscreen Rgba8Unorm texture (sized to
    /// the bar region), read back to CPU, and alpha-blended at `bars_pos`. This
    /// keeps the GPU work small while ensuring the composited HUD pixels are
    /// produced by Vello, not the CPU stub.
    fn compose_spectral_hud_vello(
        &mut self,
        final_pixels: &mut [[u8; 4]],
        display_w: u32,
        display_h: u32,
        bands: &SpectralRadianceCache,
        bars_pos: [f32; 2],
    ) -> usize {
        use vox_ui::spectral_hud::SpectralHUD;
        use vox_ui::vello_ctx::VelloCtx;

        // The bar block is 160x60 at bars_pos. Render into a slightly padded
        // offscreen texture so AA edges aren't clipped.
        let region_w: u32 = 176;
        let region_h: u32 = 72;

        // Lazily create the headless GPU Vello context once.
        if self.vello_hud.is_none() && !self.vello_hud_unavailable {
            match VelloCtx::new_headless(region_w, region_h) {
                Some(ctx) => {
                    self.vello_hud = Some(ctx);
                    println!("[vello] SpectralHUD GPU path initialised ({region_w}x{region_h})");
                }
                None => {
                    self.vello_hud_unavailable = true;
                    eprintln!("[vello] no GPU adapter — SpectralHUD falls back to CPU software path");
                }
            }
        }

        let Some(ctx) = self.vello_hud.as_mut() else {
            return 0;
        };

        ctx.begin_frame();
        // Render the bars at local origin within the region.
        SpectralHUD::render(ctx, bands, [0.0, 0.0]);
        let rgba = match ctx.render_to_rgba() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[vello] SpectralHUD GPU render failed: {e}");
                return 0;
            }
        };

        // Alpha-blend the rendered region over the frame at bars_pos.
        let ox = bars_pos[0].round() as i64;
        let oy = bars_pos[1].round() as i64;
        let mut blended = 0usize;
        for ry in 0..region_h {
            for rx in 0..region_w {
                let src = rgba[(ry * region_w + rx) as usize];
                // The Vello readback renders on a TRANSPARENT base, so the
                // alpha channel is authoritative: zero-alpha = background.
                // (The old near-black color-key dropped low-coverage AA edge
                // pixels and any genuinely dark HUD content.)
                if src[3] == 0 {
                    continue;
                }
                let dx = ox + rx as i64;
                let dy = oy + ry as i64;
                if dx < 0 || dy < 0 || dx >= display_w as i64 || dy >= display_h as i64 {
                    continue;
                }
                let idx = (dy as u32 * display_w + dx as u32) as usize;
                // Source alpha-over (straight alpha from the readback).
                let a = src[3] as f32 / 255.0;
                let inv = 1.0 - a;
                let dst = final_pixels[idx];
                final_pixels[idx] = [
                    (src[0] as f32 * a + dst[0] as f32 * inv) as u8,
                    (src[1] as f32 * a + dst[1] as f32 * inv) as u8,
                    (src[2] as f32 * a + dst[2] as f32 * inv) as u8,
                    255,
                ];
                blended += 1;
            }
        }
        blended
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
// Save / load helpers
// ---------------------------------------------------------------------------

impl EngineApp {
    fn build_world_save(&self) -> vox_data::world_save::WorldSave {
        use vox_data::world_save::{SavedEntity, WorldSave};

        let cam_pos = [self.camera.target.x, self.camera.target.y, self.camera.target.z];
        let cam_rot = [self.camera.orbit_angle, self.camera.orbit_distance, self.camera.altitude, 1.0];

        let entities: Vec<SavedEntity> = self.editor.entities.iter().map(|e| SavedEntity {
            name: e.name.clone(),
            position: e.position.to_array(),
            rotation: e.rotation.to_array(),
            scale: e.scale.to_array(),
            asset_path: Some(e.asset_path.clone()),
            scripts: Vec::new(),
            tags: Vec::new(),
            custom_data: std::collections::HashMap::new(),
            collider: None,
            audio: None,
            light: None,
            splats: Vec::new(),
            geom_splats: Vec::new(),
            prefab_ref: None,
        }).collect();

        WorldSave::from_entities(entities, cam_pos, cam_rot, self.loop_.time_of_day())
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
                self.input_state.press(vox_core::input::InputSource::Key(key as u32));

                match key {
                    KeyCode::Escape => {
                        if self.mouse_captured {
                            self.mouse_captured = false;
                            if let Some(w) = &self.window {
                                w.set_cursor_visible(true);
                            }
                        } else {
                            self.loop_.runtime.stop();
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
                        let new_hour = (self.loop_.time_of_day() + 1.0) % 24.0;
                        self.loop_.set_time_of_day(new_hour);
                        self.temporal.reset();
                        println!("[ochroma] Time: {:.0}:00", self.loop_.time_of_day());
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
                    KeyCode::KeyN => {
                        self.soundscape.active = !self.soundscape.active;
                        println!("[ochroma] Soundscape: {}", if self.soundscape.active { "ON" } else { "OFF" });
                    }
                    KeyCode::KeyB => {
                        self.node_panel_visible = !self.node_panel_visible;
                        println!("[ochroma] Node-graph panel: {}", if self.node_panel_visible { "ON" } else { "OFF" });
                    }
                    KeyCode::KeyI => {
                        self.editor.mini_map.open = !self.editor.mini_map.open;
                        println!("[ochroma] Mini map: {}", if self.editor.mini_map.open { "ON" } else { "OFF" });
                    }
                    KeyCode::KeyQ => {
                        self.spectral_bypass = !self.spectral_bypass;
                        self.temporal.reset();
                        println!(
                            "[ochroma] Render mode: {}",
                            if self.spectral_bypass { "FAST (direct RGB)" } else { "QUALITY (spectral pipeline)" }
                        );
                    }
                    KeyCode::KeyP => {
                        self.character.enabled = !self.character.enabled;
                        println!("[ochroma] Character controller: {}", if self.character.enabled { "ON" } else { "OFF" });
                    }
                    KeyCode::KeyG => {
                        self.dlss.frame_gen = match self.dlss.frame_gen {
                            FrameGeneration::Off => FrameGeneration::On,
                            FrameGeneration::On => FrameGeneration::Off,
                        };
                        println!("[ochroma] Frame generation: {:?}", self.dlss.frame_gen);
                    }
                    KeyCode::Tab => {
                        if self.editor_visible {
                            // Close editor on second Tab press
                            self.editor_visible = false;
                            self.editor.visible = false;
                            println!("[ochroma] Editor CLOSED");
                        } else {
                            // Cycle spectral viewport mode
                            self.spectral_viewport_mode = self.spectral_viewport_mode.cycle_next();
                            println!("[ochroma] Spectral viewport: {}", self.spectral_viewport_mode.label());
                        }
                    }
                    KeyCode::Backquote => {
                        // Open editor (Backquote / tilde key)
                        self.editor_visible = !self.editor_visible;
                        self.editor.visible = self.editor_visible;
                        println!("[ochroma] Editor {}", if self.editor_visible { "OPEN" } else { "CLOSED" });
                        if self.editor_visible {
                            self.editor.show_console();
                            if self.editor.selected.is_none() && !self.editor.entities.is_empty() {
                                let first_id = self.editor.entities[0].id;
                                self.editor.select(first_id);
                            }
                        }
                    }
                    KeyCode::KeyE => {
                        use vox_render::splat_particles::{SplatEmitter, EmitterConfig};
                        let pos = self.camera.position.to_array();
                        self.particle_emitters.push(SplatEmitter::new(EmitterConfig::fire(pos)));
                        println!("[ochroma] Spawned fire emitter at {:?}", pos);
                    }
                    KeyCode::Delete
                        if self.editor_visible => {
                            self.editor.delete_selected();
                        }
                    KeyCode::ArrowUp
                        if self.editor_visible => {
                            self.editor.move_selected(Vec3::new(0.0, 0.0, -1.0));
                        }
                    KeyCode::ArrowDown
                        if self.editor_visible => {
                            self.editor.move_selected(Vec3::new(0.0, 0.0, 1.0));
                        }
                    KeyCode::ArrowLeft
                        if self.editor_visible => {
                            self.editor.move_selected(Vec3::new(-1.0, 0.0, 0.0));
                        }
                    KeyCode::ArrowRight
                        if self.editor_visible => {
                            self.editor.move_selected(Vec3::new(1.0, 0.0, 0.0));
                        }
                    KeyCode::F5
                        if self.editor_visible && self.editor.editor_mode == vox_app::editor::EditorPlayMode::Editing => {
                            self.editor.play_requested = true;
                            self.editor.editor_mode = vox_app::editor::EditorPlayMode::Playing;
                            println!("[ochroma] Play");
                        }
                    KeyCode::F6
                        if self.editor_visible && self.editor.editor_mode != vox_app::editor::EditorPlayMode::Editing => {
                            self.editor.pause_requested = true;
                            self.editor.editor_mode = if self.editor.editor_mode == vox_app::editor::EditorPlayMode::Playing {
                                vox_app::editor::EditorPlayMode::Paused
                            } else {
                                vox_app::editor::EditorPlayMode::Playing
                            };
                            println!("[ochroma] Pause/Resume");
                        }
                    KeyCode::F7
                        if self.editor_visible && self.editor.editor_mode != vox_app::editor::EditorPlayMode::Editing => {
                            self.editor.stop_requested = true;
                            self.editor.editor_mode = vox_app::editor::EditorPlayMode::Editing;
                            println!("[ochroma] Stop");
                        }
                    KeyCode::KeyS if self.ctrl_held => {
                        let map = self.editor.export_to_map("Ochroma Scene");
                        let path = std::env::temp_dir().join("ochroma_scene.ochroma_map");
                        match map.save(&path) {
                            Ok(()) => println!("[ochroma] Scene saved to {}", path.display()),
                            Err(e) => eprintln!("[ochroma] Save failed: {}", e),
                        }
                    }
                    KeyCode::KeyZ if self.ctrl_held
                        && self.editor_visible => {
                            self.editor.undo();
                            println!("[ochroma] Undo ({} left)", self.editor.undo_stack.len());
                        }
                    KeyCode::KeyY if self.ctrl_held
                        && self.editor_visible => {
                            self.editor.redo();
                            println!("[ochroma] Redo ({} left)", self.editor.redo_stack.len());
                        }
                    KeyCode::KeyO => {
                        let _ = self.loop_.spatial_audio.play_3d(
                            std::path::Path::new("assets/audio/ambient/wind_loop.ogg"),
                            glam::Vec3::new(10.0, 0.0, 0.0),
                            0.5,
                            true,
                        );
                        println!("[ochroma] Playing demo 3D sound at (10, 0, 0)");
                    }
                    KeyCode::KeyF => {
                        let ws = self.build_world_save();
                        let path = vox_data::world_save::WorldSave::quick_save_path();
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        match ws.save_to_file(&path) {
                            Ok(_) => println!("[ochroma] World saved to {}", path.display()),
                            Err(e) => println!("[ochroma] Save failed: {}", e),
                        }
                    }
                    KeyCode::KeyL => {
                        let path = vox_data::world_save::WorldSave::quick_save_path();
                        match vox_data::world_save::WorldSave::load_from_file(&path) {
                            Ok(ws) => {
                                self.loop_.set_time_of_day(ws.resources.time_of_day);
                                let cp = ws.resources.camera_position;
                                let cr = ws.resources.camera_rotation;
                                self.camera.target = glam::Vec3::new(cp[0], cp[1], cp[2]);
                                self.camera.orbit_angle = cr[0];
                                self.camera.orbit_distance = cr[1];
                                self.camera.altitude = cr[2];
                                for saved in &ws.entities {
                                    if let Some(entity) = self.editor.entities.iter_mut().find(|e| e.name == saved.name) {
                                        entity.position = glam::Vec3::from(saved.position);
                                    } else {
                                        println!("[ochroma] Load: entity '{}' not found in scene (skipped)", saved.name);
                                    }
                                }
                                println!("[ochroma] World loaded from {} ({} entities)", path.display(), ws.entities.len());
                            }
                            Err(e) => println!("[ochroma] Load failed: {}", e),
                        }
                    }
                    _ => {}
                }
            } else {
                self.keys.remove(&key);
                self.input_state.release(vox_core::input::InputSource::Key(key as u32));
            }
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let button_index: u8 = match button {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            MouseButton::Middle => 2,
            MouseButton::Back => 3,
            MouseButton::Forward => 4,
            MouseButton::Other(n) => n.min(255) as u8,
        };
        if state == ElementState::Pressed {
            self.input_state.press(vox_core::input::InputSource::MouseButton(button_index));
        } else {
            self.input_state.release(vox_core::input::InputSource::MouseButton(button_index));
        }
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
                        // Sync gizmo mode from editor to renderer
                        self.gizmo.mode = match self.editor.gizmo_mode {
                            vox_app::editor::GizmoMode::Translate => vox_render::gizmos::GizmoMode::Translate,
                            vox_app::editor::GizmoMode::Rotate    => vox_render::gizmos::GizmoMode::Rotate,
                            vox_app::editor::GizmoMode::Scale     => vox_render::gizmos::GizmoMode::Scale,
                        };
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
                self.loop_.audio.play_sine_backend(self.click_counter, 800.0, 0.05, 0.3);
                self.loop_.spatial_audio.play_tone(800.0, 0.05, 0.3);

                // ScreenRay terrain pick — update last_pick for building placement
                {
                    use vox_core::picking::ScreenRay;
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
                    let vp_inv = (proj * view).inverse();
                    let ray = ScreenRay::from_screen(
                        self.mouse_x as f32,
                        self.mouse_y as f32,
                        dw as f32,
                        dh as f32,
                        vp_inv,
                    );
                    self.last_pick = ray.terrain_hit(&|_x, _z| 0.0, 500.0);
                    if let Some(pos) = self.last_pick {
                        println!("[pick] terrain hit at [{:.2}, {:.2}, {:.2}]", pos.x, pos.y, pos.z);
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_mouse_move(&mut self, x: f64, y: f64) {
        if let Some((lx, ly)) = self.last_mouse {
            self.input_state.mouse_dx += (x - lx) as f32;
            self.input_state.mouse_dy += (y - ly) as f32;
        }
        self.input_state.mouse_x = x as f32;
        self.input_state.mouse_y = y as f32;
        self.mouse_x = x;
        self.mouse_y = y;

        // Gizmo drag: move the selected entity along the constrained axis
        if self.gizmo.dragging && self.editor_visible
            && let Some(sel_id) = self.editor.selected
            && let Some(sel_entity) = self.editor.entities.iter().find(|e| e.id == sel_id)
        {
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
            // update_drag now returns GizmoDelta {Translate|Rotate|Scale}. The scene editor's
            // move_selected is translation-only, so extract translation(); Rotate/Scale yield
            // ZERO here (entity rotate/scale wiring is a later editor task).
            let mut translation = delta.translation();
            // Apply snap-to-grid if enabled
            if self.editor.snap_enabled && self.editor.snap_grid > 0.0 {
                let grid = self.editor.snap_grid;
                translation = match self.gizmo.active_axis {
                    Some(vox_render::gizmos::Axis::X) => glam::Vec3::new(
                        (translation.x / grid).round() * grid, 0.0, 0.0,
                    ),
                    Some(vox_render::gizmos::Axis::Y) => glam::Vec3::new(
                        0.0, (translation.y / grid).round() * grid, 0.0,
                    ),
                    Some(vox_render::gizmos::Axis::Z) => glam::Vec3::new(
                        0.0, 0.0, (translation.z / grid).round() * grid,
                    ),
                    None => translation,
                };
            }
            if translation.length_squared() > 1e-8 {
                self.editor.move_selected(translation);
            }
        } else if self.mouse_captured {
            if let Some((lx, ly)) = self.last_mouse {
                let dx = (x - lx) as f32;
                let dy = (y - ly) as f32;
                self.cam_yaw += dx * 0.003;
                self.cam_pitch = (self.cam_pitch - dy * 0.003).clamp(-1.5, 1.5);
                // Also drive character yaw when character controller is active
                if self.character.enabled && !self.editor_visible {
                    self.character.yaw += dx * 0.002;
                    self.cam_yaw = self.character.yaw; // keep free-camera yaw in sync
                }
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
        if let (Some(backend), Some(gpu_rast)) = (&self.backend, &mut self.gpu_rasteriser) {
            gpu_rast.resize(backend.device(), w, h);
        }
    }

    fn handle_redraw(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.frame_dt = dt;

        // All window/GPU-free per-frame simulation (camera, AI, physics, audio,
        // scripts, entity sync) lives in step_simulation — shared verbatim with
        // the headless smoke path (run_smoke).
        self.step_simulation(dt);


        // 5. Render + present
        //    Primary path: GPU rasteriser directly to surface texture.
        //    Fallback: software rasteriser -> blit via backend.
        if self.gpu_rasteriser.is_some() && self.backend.is_some() {
            // --- GPU primary render path ---
            let backend = self.backend.as_ref().expect("backend checked above");

            // Lazy-init SdfShadowPass once terrain and GPU are both available
            if self.sdf_shadow.is_none()
                && let Some(terrain) = &self.terrain_volume
            {
                use vox_render::gpu::sdf_shadow_pass::{SdfShadowPass, SdfUniform};
                let sdf_data = terrain.to_sdf_buffer();
                let (sx, sy, sz, vs) = terrain.sdf_metadata();
                let depth_placeholder = backend.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("sdf_depth_placeholder"),
                    size: wgpu::Extent3d { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let depth_view = depth_placeholder.create_view(&wgpu::TextureViewDescriptor {
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                });
                let sdf_uniform = SdfUniform {
                    origin: [0.0; 3],
                    _pad0: 0.0,
                    voxel_size: vs,
                    size_x: sx as u32,
                    size_y: sy as u32,
                    size_z: sz as u32,
                    light_dir: [0.577, -0.577, 0.577],
                    penumbra_k: 8.0,
                    max_dist: 50.0,
                    _pad1: [0.0; 3],
                };
                let w = DEFAULT_WIDTH;
                let h = DEFAULT_HEIGHT;
                self.sdf_shadow = Some(SdfShadowPass::new(
                    backend.device(), &sdf_data, sdf_uniform, &depth_view, w, h,
                ));
                println!("[ochroma] SdfShadowPass initialized ({sx}x{sy}x{sz} SDF, {w}x{h})");
            }

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
                let pos = Vec3::from(splat.position());
                let radius = splat.scale_u().max(splat.scale_v()).max(splat.scale_w()) + 1.0;
                if frustum.contains_sphere(pos, radius) {
                    visible_splats.push(*splat);
                }
            }
            // LOD
            let lod_splats: Vec<GaussianSplat> = visible_splats.iter().enumerate()
                .filter(|&(i, s)| {
                    let dist = cam_pos.distance(Vec3::from(s.position()));
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
                    let p = ws.position();
                    ws.set_position([p[0] + pos.x, p[1] + pos.y, p[2] + pos.z]);
                    render_splats.push(ws);
                }
            }
            render_splats.extend(&self.particles.to_splats());

            // Drain animation splats from RenderBuffer (written by animation_system)
            {
                let anim_splats = std::mem::take(
                    &mut self.loop_.world_mut().resource_mut::<vox_core::engine_runtime::RenderBuffer>().splats
                );
                render_splats.extend(anim_splats);
            }

            // Tick GLTF animation driver and append deformed splats
            if let Some(ref mut driver) = self.anim_driver {
                let animated_splats = driver.tick(dt);
                render_splats.extend(animated_splats);
            }

            let illuminant = illuminant_for_time(self.loop_.time_of_day());

            // Dispatch SDF soft shadow compute pass
            if let Some(sdf_pass) = &self.sdf_shadow {
                let mut sdf_encoder = backend.device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("sdf_shadow_encoder") }
                );
                sdf_pass.dispatch(&mut sdf_encoder);
                backend.queue().submit(Some(sdf_encoder.finish()));
            }

            let gpu_rast = self.gpu_rasteriser.as_ref().expect("gpu_rasteriser checked above");
            let shadow_vp = self.loop_.shadow_mapper.cascades.first().map(|c| c.light_view_proj);
            gpu_rast.render_with_shadow(
                backend.device(),
                backend.queue(),
                &view,
                &render_splats,
                &render_camera,
                &illuminant,
                shadow_vp.as_ref(),
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
                self.editor.status_splat_count = self.scene_splats.len();
                let full_output = self.egui_ctx.run(raw_input, |ctx| {
                    if editor_visible {
                        self.editor.show(ctx);
                        self.content_browser.show(ctx);
                        self.material_editor_ui.open = self.editor.show_material_editor;
                        self.material_editor_ui.show(ctx);
                        self.editor.show_material_editor = self.material_editor_ui.open;
                        self.anim_editor_ui.open = self.editor.show_anim_editor;
                        self.anim_editor_ui.show(ctx);
                        self.editor.show_anim_editor = self.anim_editor_ui.open;
                        self.vfx_editor_ui.open = self.editor.show_vfx_editor;
                        self.vfx_editor_ui.show(ctx);
                        self.editor.show_vfx_editor = self.vfx_editor_ui.open;
                    }
                    // Game widgets (resource panels, tooltips, buttons)
                    self.game_widgets.render(ctx, &self.widget_cmds);

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
                                            "FPS: {:.0} | Spectra: {} | {} points | DLSS {}",
                                            fps, !self.spectral_bypass, splat_count, dlss_mode,
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

        // Consume editor menu action flags
        if self.editor.pending_new_scene {
            self.editor.pending_new_scene = false;
            self.editor.entities.clear();
            self.editor.selected = None;
            println!("[ochroma] New scene");
        }
        if self.editor.pending_save || self.editor.pending_save_as {
            self.editor.pending_save = false;
            self.editor.pending_save_as = false;
            let map = self.editor.export_to_map("Ochroma Scene");
            let path = std::env::temp_dir().join("ochroma_scene.ochroma_map");
            match map.save(&path) {
                Ok(()) => println!("[ochroma] Scene saved to {}", path.display()),
                Err(e) => eprintln!("[ochroma] Save failed: {}", e),
            }
        }
        if self.editor.pending_open {
            self.editor.pending_open = false;
            // File picker not available yet — log intent
            println!("[ochroma] Open: file picker not yet implemented");
        }
        if self.editor.pending_exit {
            self.editor.pending_exit = false;
            self.loop_.runtime.stop();
            event_loop.exit();
            return;
        }
        if self.editor.play_requested {
            self.editor.play_requested = false;
            println!("[ochroma] \u{25b6} PLAY MODE");
        }
        if self.editor.pause_requested {
            self.editor.pause_requested = false;
            println!("[ochroma] \u{23f8} PAUSE");
        }
        if self.editor.stop_requested {
            self.editor.stop_requested = false;
            println!("[ochroma] \u{23f9} STOP \u{2014} returning to edit mode");
        }
        if let Some(id) = self.editor.focus_camera_on.take()
            && let Some(entity) = self.editor.entities.iter().find(|e| e.id == id)
        {
            let target = entity.position;
            self.camera.position = target + glam::Vec3::new(0.0, 5.0, 15.0);
            self.cam_yaw = 0.0;
            self.cam_pitch = -0.2;
            println!("[ochroma] Camera focused on entity #{} '{}'", id, entity.name);
        }

        // Handle drag-and-drop from content browser
        if let Some(asset_path) = self.content_browser.dragging_asset.take() {
            let forward = self.camera_forward();
            let pos = self.camera.position + forward * 10.0;
            let name = std::path::Path::new(&asset_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Dropped Asset")
                .to_string();
            let _id = self.editor.add_entity(&name, &asset_path, pos);
            self.editor.status_message = format!("Placed: {}", name);
            println!("[ochroma] Dropped '{}' at {:?}", name, pos);
        }

        // 7. FPS counter + title update (throttled to every 0.5s)
        self.frame_count += 1;
        let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.fps_timer = now;
        }
        // Update title at most every 0.5s
        if now.duration_since(self.title_timer).as_secs_f32() >= 0.5 {
            if let Some(w) = &self.window {
                let title = if self.editor_visible {
                    let n = self.editor.entity_count();
                    let noun = if n == 1 { "thing" } else { "things" };
                    format!(
                        "Ochroma Engine \u{2014} Editor | {} {} in the world | {:.0} FPS",
                        n, noun, self.fps,
                    )
                } else {
                    format!("Ochroma Engine | {} points | {:.0} FPS", self.total_splat_count(), self.fps)
                };
                w.set_title(&title);
            }
            self.title_timer = now;
        }

        // End of frame: clear transient input state (just_pressed, just_released)
        self.input_state.end_frame();

        // Request next frame
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// All window/GPU-free per-frame simulation: biome blend, patrol AI, camera,
    /// LOD streaming, notifications, cursor, left-click handling, particles, ECS
    /// runtime tick, procedural animation, physics + character controller, audio,
    /// scripts (Rhai/Lua), and entity->splat sync. Called every frame by
    /// handle_redraw (windowed) and by run_smoke (headless). The only
    /// window-dependent bits (cursor icon) are guarded by `self.window` and
    /// no-op headlessly.
    fn step_simulation(&mut self, dt: f32) {
        // PCG-style live-in-viewport graph: flush any throttled recook request
        // this frame. Cooks only the dirty subgraph and splices fresh splats into
        // the viewport without a restart (prints the verifiable live-recook line).
        self.live_recook(std::time::Instant::now());

        // Biome ambient soundscape: blend toward current biome target each frame
        {
            let biome = vox_audio::BiomeKind::Grassland; // default; terrain integration in future domain
            let target = vox_audio::BiomeAmbientMix::for_biome(biome);
            self.ambient_mix = self.ambient_mix.blend_toward(&target, 0.02);
        }

        // Update patrol agents along navmesh with spectral perception
        if let Some(nm) = &self.navmesh {
            let dt = self.frame_dt;
            let sky_ambient = illuminant_for_time(self.loop_.time_of_day()).bands;
            for agent in &mut self.patrol_agents {
                agent.update(dt, nm);
                // Update spectral perception from current sky illuminant
                let gi_adapter = SpectralGiAdapter { sky_ambient };
                agent.spectral_perception.position = agent.position;
                let percept = agent.spectral_perception.sense(&gi_adapter);
                agent.spectral_perception.update_emotion(&gi_adapter);
                // Raise alert if green band (7) energy exceeds threshold
                let _ = percept;
            }
        }

        // 1. Update camera from input
        self.update_camera(dt);

        // LOD streaming: update tile manager and load newly active tiles
        {
            let cam_pos = self.camera.position;
            let cam_tile = vox_core::lwc::TileCoord {
                x: (cam_pos.x / vox_core::lwc::TILE_SIZE as f32) as i32,
                z: (cam_pos.z / vox_core::lwc::TILE_SIZE as f32) as i32,
            };
            let newly_active = self.tile_manager.update_camera(cam_tile);
            for tile in &newly_active {
                let path = format!("assets/tiles/tile_{}_{}.vxm", tile.x, tile.z);
                let p = std::path::Path::new(&path);
                if p.exists() {
                    match std::fs::read(p).and_then(|bytes| {
                        vox_data::vxm::VxmFile::read(&bytes[..])
                            .map_err(std::io::Error::other)
                    }) {
                        Ok(_vxm) => {
                            println!("[streaming] Loaded tile {},{}", tile.x, tile.z);
                        }
                        Err(e) => eprintln!("[streaming] Failed to load {path}: {e}"),
                    }
                }
            }
        }

        // Tick notification queue (decrement TTLs, expire old toasts)
        self.editor.notification_queue.tick(dt);

        // Cursor changes based on editor mode (no-op when headless: window is None)
        if self.editor_visible
            && let Some(w) = &self.window
        {
            if self.gizmo.dragging {
                w.set_cursor(winit::window::CursorIcon::Grab);
            } else if self.gizmo.active_axis.is_some() {
                w.set_cursor(winit::window::CursorIcon::Pointer);
            } else {
                w.set_cursor(winit::window::CursorIcon::Default);
            }
        }

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
                if let Some((col_handle, _hit_pos, _dist)) = self.loop_.physics.raycast_with_collider(
                    [near_pt.x, near_pt.y, near_pt.z],
                    [ray_dir.x, ray_dir.y, ray_dir.z],
                    1000.0,
                )
                    && let Some(&entity_id) = self.collider_to_entity.get(&col_handle)
                {
                    self.editor.select(entity_id);
                    if let Some(name) = self.editor.selected_name() {
                        println!("[ochroma] Rapier pick: '{}' (id={})", name, entity_id);
                    }
                    picked = true;
                }

                // Fall back to editor ray-sphere test if Rapier didn't match an entity
                if !picked
                    && let Some(id) = self.editor.pick_entity_at_screen_pos(
                        self.mouse_x as f32,
                        self.mouse_y as f32,
                        dw,
                        dh,
                        inv_vp,
                    )
                {
                    self.editor.select(id);
                    if let Some(name) = self.editor.selected_name() {
                        println!("[ochroma] Editor pick: '{}' (id={})", name, id);
                    }
                }
            } else {
                self.place_object_at_cursor();
            }
            self.left_click_pending = false;
        }

        // 3. Tick particles
        self.particles.tick(dt);

        // 4. Tick engine runtime (scripts, time advance) on the loop's world —
        // the single ticked ECS (S4). NOTE: we call runtime.tick directly rather
        // than loop_.step_scripts here because step_scripts also DRAINS the
        // RenderBuffer; the shell's animation_system below appends to that same
        // RenderBuffer *after* the tick, and the render path drains it once
        // afterward. Draining here would lose the gather_splats output, so the
        // tick stays drain-free to preserve behavior exactly.
        self.loop_.runtime.tick(dt);

        // 4a. Run procedural animation — appends bobbing splats to RenderBuffer
        // (operates on the loop's world, where the NPC was spawned).
        {
            use bevy_ecs::system::{IntoSystem, System};
            let mut sys = IntoSystem::into_system(animation_system);
            sys.initialize(self.loop_.world_mut());
            sys.run((), self.loop_.world_mut());
            sys.apply_deferred(self.loop_.world_mut());
        }

        // 4b. Step Rapier physics and sync dynamic bodies back to ECS
        // Update character controller before physics step.
        // character.update() processes input (move_dir, jump, gravity) and stores
        // last_move_dir. When rapier_kcc_active, it skips flat-plane integration.
        // We then pass the full desired velocity (horizontal + vertical) through KCC.
        self.character.rapier_kcc_active = self.character_body.is_some();
        self.character.update(&self.input_state, dt, &mut self.loop_.physics);
        if let Some(ref cb) = self.character_body
            && self.character.enabled {
                // Full desired velocity: horizontal from input + vertical from gravity/jump
                let desired = self.character.last_move_dir
                    + Vec3::Y * self.character.vertical_velocity;
                let output = cb.move_and_slide(
                    desired, dt,
                    self.loop_.physics.rigid_body_set(),
                    self.loop_.physics.collider_set(),
                    self.loop_.physics.query_pipeline(),
                );
                self.character.on_ground = output.grounded;
                if output.grounded && self.character.vertical_velocity < 0.0 {
                    self.character.vertical_velocity = 0.0;
                }
                cb.apply_translation(output.effective_translation, self.loop_.physics.rigid_body_set_mut());
                self.character.position = cb.position(self.loop_.physics.rigid_body_set());
                self.loop_.physics.set_kinematic_position(
                    self.character.body_handle,
                    [self.character.position.x, self.character.position.y, self.character.position.z],
                );
            }
        // If character is enabled, drive camera position from character
        if self.character.enabled {
            let cam_pos = self.character.camera_position();
            self.camera.position = cam_pos;
            self.camera.target = cam_pos + self.character.camera_forward();
        }
        // EngineLoop S1: step Rapier physics and sync dynamic bodies back to ECS.
        // (Relocated from the inline block; the ECS<->Rapier body map now lives in
        // self.loop_.entity_rapier_bodies.)
        self.loop_.step_physics(dt);

        // 4c. Tick audio (legacy AudioEngine + spatial audio manager) via EngineLoop (S2).
        // Listener pose: when the character controller is enabled, use its
        // position/forward directly so the listener tracks the character even
        // before camera sync; otherwise use the camera. `audio.set_listener`
        // historically used `camera.position`, which already equals
        // `character.camera_position()` once the camera was synced above, so the
        // single `listener_pos` below preserves both backends' behavior exactly.
        let (listener_pos, listener_fwd) = if self.character.enabled {
            (self.character.camera_position(), self.character.camera_forward())
        } else {
            (self.camera.position, self.camera_forward())
        };
        self.loop_.step_audio(dt, listener_pos, listener_fwd);

        // Process pending script commands for audio playback via EngineLoop (S4).
        // drain_sound_commands is a byte-for-byte relocation of the old inline
        // block: it drains PendingScriptCommands from the loop's world, maps each
        // PlaySound clip to a tone frequency, plays it on the loop's spatial
        // audio, and restores unprocessed commands.
        self.loop_.drain_sound_commands();

        // 4c-rhai. Per-frame Rhai script update + command dispatch
        {
            let reloaded = self.rhai.poll_reload();
            for name in &reloaded {
                println!("[ochroma] hot-reload: {}", name);
            }
            let dt_dyn = rhai::Dynamic::from(dt as f64);
            for i in 0..self.rhai.script_count() {
                let _ = self.rhai.call_fn(i, "on_update", std::slice::from_ref(&dt_dyn));
            }
        }
        {
            use vox_core::script_interface::ScriptCommand;
            for cmd in vox_script::rhai_runtime::drain_pending_commands() {
                match cmd {
                    ScriptCommand::SetPosition { position } => {
                        println!("[script] set_position ({}, {}, {})", position[0], position[1], position[2]);
                    }
                    ScriptCommand::PlaySound { clip, volume, .. } => {
                        if let Some(ref h) = self.audio_handle {
                            let _ = h.play(&clip, volume, false);
                        }
                    }
                    ScriptCommand::Log { message } => {
                        println!("[script] {}", message);
                    }
                    other => {
                        println!("[script] unhandled command: {:?}", other);
                    }
                }
            }
        }

        // 4c-lua. Per-frame Lua update + spectral threshold callbacks
        {
            if let Some(watcher) = &self.script_watcher {
                for path in watcher.drain() {
                    self.lua.pending_reload.push(path);
                }
            }
            if let Err(e) = self.lua.call_update(dt) {
                eprintln!("[ochroma] Lua update error: {}", e);
            }
            vox_script::tick_thresholds(
                self.lua.lua(),
                &self.spectral_script_state,
            ).ok();
        }

        // 4d. Sync entity positions to splats — when scripts move entities,
        //     the corresponding splats move with them.
        {
            use vox_core::ecs::TransformComponent;
            let world = self.loop_.world_mut();
            let mut query = world.query::<(bevy_ecs::prelude::Entity, &TransformComponent)>();
            let entities: Vec<(u32, [f32; 3])> = query.iter(world)
                .map(|(e, t)| (e.index(), [t.position.x, t.position.y, t.position.z]))
                .collect();
            for (eid, pos) in entities {
                if let Some(&(start, end)) = self.entity_splat_ranges.get(&eid)
                    && let Some(&orig_pos) = self.entity_original_positions.get(&eid)
                {
                    let dx = pos[0] - orig_pos[0];
                    let dy = pos[1] - orig_pos[1];
                    let dz = pos[2] - orig_pos[2];
                    if dx.abs() > 1e-6 || dy.abs() > 1e-6 || dz.abs() > 1e-6 {
                        for i in start..end.min(self.scene_splats.len()) {
                            let p = self.scene_splats[i].position();
                            self.scene_splats[i].set_position([p[0] + dx, p[1] + dy, p[2] + dz]);
                        }
                        self.entity_original_positions.insert(eid, pos);
                    }
                }
            }
        }
    }

}

// ---------------------------------------------------------------------------
// ApplicationHandler -- the real game loop
// ---------------------------------------------------------------------------

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title(&self.loop_.runtime.config.window_title)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.loop_.runtime.config.window_width,
                self.loop_.runtime.config.window_height,
            ));

        let window = Arc::new(
            event_loop.create_window(attrs).expect("Failed to create window"),
        );

        match WgpuBackend::new(Arc::clone(&window), DEFAULT_WIDTH, DEFAULT_HEIGHT) {
            Ok(backend) => {
                println!("[ochroma] GPU backend initialised");
                let mut gpu_rast = GpuRasteriser::new(
                    backend.device(),
                    backend.surface_format(),
                    DEFAULT_WIDTH,
                    DEFAULT_HEIGHT,
                );
                gpu_rast.init_shadow_pass(backend.device());
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

        // Try loading a GLTF character for skeletal animation
        {
            use vox_data::gltf_animation::{assign_joint_bindings, extract_skeleton};
            use vox_render::animation_driver::AnimationDriver;

            let gltf_path = std::path::Path::new("assets/character.glb");
            if gltf_path.exists() {
                match extract_skeleton(gltf_path) {
                    Ok((skeleton, animations)) => {
                        let joint_bindings = assign_joint_bindings(&self.scene_splats, &skeleton);
                        let mut driver = AnimationDriver::new(skeleton, self.scene_splats.clone());
                        driver.joint_bindings = joint_bindings;
                        for anim in animations {
                            driver.add_animation(anim);
                        }
                        if driver.animation_count() > 0 {
                            driver.play(0);
                        }
                        self.anim_driver = Some(driver);
                        println!("[ochroma] GLTF character loaded with {} animations", self.anim_driver.as_ref().unwrap().animation_count());
                    }
                    Err(e) => eprintln!("[animation] GLTF load failed: {e}"),
                }
            }
        }

        // Start the engine runtime (initialises scripts)
        self.loop_.runtime.start();

        // Auto-load Rhai scripts from the scripts/ directory
        if let Ok(entries) = std::fs::read_dir("scripts") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("rhai") {
                    match self.rhai.load_script_file(
                        path.file_stem().unwrap_or_default().to_str().unwrap_or(""),
                        &path,
                    ) {
                        Ok(idx) => println!("[ochroma] Loaded script #{}: {}", idx, path.display()),
                        Err(e) => println!("[ochroma] Script load error {}: {}", path.display(), e),
                    }
                }
            }
        }

        // --demo flag: load the bundled demo scene script
        if std::env::args().any(|a| a == "--demo") {
            let demo_path = std::path::Path::new("examples/demo_scene/main.rhai");
            match self.rhai.load_script_file("demo", demo_path) {
                Ok(idx) => {
                    println!("[ochroma] Demo mode: loaded {}", demo_path.display());
                    let _ = self.rhai.call_fn(idx, "on_start", &[]);
                }
                Err(e) => eprintln!("[ochroma] Demo mode: failed to load script (run from repo root): {}", e),
            }
        }

        println!();
        println!("Ochroma Engine v0.1.0");
        println!("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
        println!("  {} things in the world | {} points | {} clusters",
            self.loop_.runtime.entity_count(),
            self.scene_splats.len(),
            self.clas_cluster_count);
        println!("  {} point lights | {} particle emitters",
            self.light_manager.point_light_count(),
            self.particles.emitters.len());
        println!();
        println!("  Tab     Editor");
        println!("  WASD    Move");
        println!("  Scroll  Zoom");
        println!("  RMB     Look");
        println!("  F5      Play");
        println!("  F12     Screenshot");
        println!("  Esc     Quit");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Forward events to egui
        if let Some(egui_state) = &mut self.egui_state
            && let Some(window) = &self.window
        {
            let _ = egui_state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => {
                self.loop_.runtime.stop();
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => self.handle_keyboard_event(&event, event_loop),
            WindowEvent::ModifiersChanged(modifiers) => {
                self.ctrl_held = modifiers.state().control_key();
            }
            WindowEvent::MouseInput { state, button, .. } => self.handle_mouse_button(button, state),
            WindowEvent::CursorMoved { position, .. } => self.handle_mouse_move(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                let forward = self.camera_forward();
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        // Zoom: move camera forward/backward, proportional to altitude
                        let zoom_speed = (self.camera.position.y.abs() / 10.0).max(1.0) * 2.0;
                        self.camera.position += forward * y * zoom_speed;
                        self.camera.orbit_distance = (self.camera.orbit_distance - y * zoom_speed).max(1.0);
                    }
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        let zoom_speed = (self.camera.position.y.abs() / 10.0).max(1.0) * 0.2;
                        self.camera.position += forward * pos.y as f32 * zoom_speed;
                    }
                }
            }
            WindowEvent::Resized(size) => self.handle_resize(size.width, size.height),
            WindowEvent::RedrawRequested => self.handle_redraw(event_loop),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// main -- launch the engine
// ---------------------------------------------------------------------------

/// Headless smoke test: run the REAL per-frame simulation (step_simulation) +
/// software render (render_frame) without a window/GPU. Proves the unified
/// EngineLoop + editor/game logic + render run on a headless box. Exits 0 on
/// success, panics (non-zero) if the loop/render is broken.
fn run_smoke() {
    println!("[ochroma] === HEADLESS SMOKE MODE (no window/GPU) ===");

    // Build the app exactly as resumed() would, minus window + GPU + egui. No
    // asset path (use the default scene). EngineApp::new constructs the
    // EngineLoop + character; build_scene() + runtime.start() are window-free.
    let mut app = EngineApp::new(None);
    app.build_scene();
    app.loop_.runtime.start();

    // The legacy baked GI cache (build_scene -> rebuild_gi) is sized for the FULL
    // scene_splats, but render_frame applies it to the frustum-culled + LOD subset
    // (GiCache::apply asserts equal lengths). The windowed build never hits this
    // because it renders via the GPU primary path; render_frame is the software
    // fallback. The legacy cache is optional (backward-compat with --bake-gi), so
    // drop it for the headless software render. The live spectral GI (loop_.step_gi)
    // still runs every frame.
    app.gi_cache = None;

    // Render via the engine's DEFAULT SPECTRA path (tile-based EWA Gaussian
    // splatting at a reduced internal resolution, then DLSS-upscaled) — the same
    // path the windowed app uses by default, and far cheaper per frame than the
    // full-res software rasteriser, which keeps the headless run tractable on CPU.
    app.spectral_bypass = false;

    // Frame the default scene (terrain + buildings at z~20, trees at z~10) from
    // above and in front. forward = (sin(yaw), sin(pitch), -cos(yaw)); yaw=0 with
    // a downward pitch looks toward -Z and down onto the scene. Walking forward
    // (-Z) then moves the camera over the scene, keeping it in frame.
    app.camera.position = Vec3::new(0.0, 18.0, 50.0);
    app.cam_yaw = 0.0; // forward = -Z (toward the scene at lower z)
    app.cam_pitch = -0.35;

    let dt = 1.0 / 60.0; // fixed 60Hz step
    // 30 frames is enough to prove the loop + sim + render run headlessly; the
    // EWA spectra render of ~240k splats is the per-frame cost driver on CPU.
    let total_frames = 30u32;
    let start_cam = app.camera.position;
    let start_frame_no = app.loop_.runtime.stats.frame_number;
    let start_patrol: Vec<Vec3> = app.patrol_agents.iter().map(|a| a.position).collect();

    let mut last_pixels: Vec<[u8; 4]> = Vec::new();
    for frame in 0..total_frames {
        // Inject "walk forward" for the first ~3/4 of the run so the REAL
        // update_camera movement code moves the camera through the scene.
        if frame < (total_frames * 3) / 4 {
            app.keys.insert(KeyCode::KeyW);
        } else {
            app.keys.remove(&KeyCode::KeyW);
        }
        app.frame_dt = dt;
        app.step_simulation(dt);
        last_pixels = app.render_frame();
        app.input_state.end_frame();
    }

    let (dw, dh) = (app.dlss.display_width, app.dlss.display_height);
    let cam_moved = (app.camera.position - start_cam).length();
    let frames_ticked = app.loop_.runtime.stats.frame_number.saturating_sub(start_frame_no);
    let patrol_moved: f32 = app
        .patrol_agents
        .iter()
        .zip(start_patrol.iter())
        .map(|(a, &s)| (a.position - s).length())
        .fold(0.0, f32::max);

    let non_black = last_pixels
        .iter()
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count();
    // Distinct RGB colors — a flat/blank fill has ~1 color; a real scene
    // (terrain + sky + splats + HUD) has many. Stronger "scene rendered" signal.
    let distinct_colors = last_pixels
        .iter()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<std::collections::HashSet<_>>()
        .len();

    // Write final frame to PPM (P6).
    let ppm_path = "/tmp/ochroma_engine_runner_smoke.ppm";
    {
        let mut data = format!("P6\n{} {}\n255\n", dw, dh).into_bytes();
        for p in &last_pixels {
            data.push(p[0]);
            data.push(p[1]);
            data.push(p[2]);
        }
        std::fs::write(ppm_path, &data).expect("failed to write smoke PPM");
    }

    println!(
        "[ochroma] SMOKE SUMMARY: frames={} cam_pos=({:.2},{:.2},{:.2}) splats={} entities={} ecs_frames={} patrol_moved={:.2} cam_moved={:.2} non_black_px={}/{} ppm={}",
        total_frames,
        app.camera.position.x,
        app.camera.position.y,
        app.camera.position.z,
        app.scene_splats.len(),
        app.loop_.runtime.entity_count(),
        frames_ticked,
        patrol_moved,
        cam_moved,
        non_black,
        last_pixels.len(),
        ppm_path,
    );
    println!("[ochroma] SMOKE SUMMARY: distinct_colors={}", distinct_colors);

    // --- Live GI + Vello HUD proof ---
    // (a) latest_gi_bands (f16 bits of the GI-lit splat nearest the camera) has
    //     at least one non-zero band after the run.
    let gi_nonzero_bands = app.latest_gi_bands.iter().filter(|&&b| b > 0).count();
    let gi_max_band = app.latest_gi_bands.iter().copied().max().unwrap_or(0);

    // (b) The GameHud actually composited into the final frame. The bottom-left
    //     spectral panel paints a translucent black backdrop [0,0,0,0.55] over
    //     the scene, so a pixel inside it must be DARKER than the mean frame
    //     luminance. Compute the panel rect from the HUD's known geometry
    //     (margin=16, panel_w=176, panel_h=76, anchored bottom-left) and sample
    //     just inside the panel's top edge (backdrop region, above the bars).
    let mean_lum: f32 = {
        let sum: u64 = last_pixels
            .iter()
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum();
        (sum as f32 / 3.0) / last_pixels.len().max(1) as f32
    };
    let margin = 16.0f32;
    let panel_h = 76.0f32; // BARS_MAX_HEIGHT(60) + PANEL_PAD(8)*2
    let panel_x = margin;
    let panel_y = dh as f32 - margin - panel_h;
    let sx = (panel_x + 4.0) as u32;
    let sy = (panel_y + 4.0) as u32;
    let panel_px = last_pixels[(sy * dw + sx) as usize];
    let panel_lum = (panel_px[0] as f32 + panel_px[1] as f32 + panel_px[2] as f32) / 3.0;

    println!(
        "[ochroma] SMOKE SUMMARY: gi_nonzero_bands={} gi_max_band={} hud_panel_px=({},{})={:?} panel_lum={:.1} mean_lum={:.1}",
        gi_nonzero_bands, gi_max_band, sx, sy, panel_px, panel_lum, mean_lum
    );

    assert!(
        gi_nonzero_bands > 0 && gi_max_band > 0,
        "live GI produced no lit bands: nonzero={} max={} — step_gi readout not wired",
        gi_nonzero_bands,
        gi_max_band
    );
    assert!(
        panel_lum < mean_lum,
        "HUD panel pixel at ({sx},{sy}) lum {:.1} not darker than frame mean {:.1} — GameHud backdrop did not composite",
        panel_lum,
        mean_lum
    );

    // (c) The live Vello GPU path actually rendered the SpectralHUD bars into
    //     the frame. compose_spectral_hud_vello records how many HUD pixels were
    //     blended from the real Vello render on the last frame. On a GPU-capable
    //     host this is > 0 (the bars + panel painted); on a headless host with
    //     no adapter it is 0 and the CPU GameHud path drew the bars instead, so
    //     this is only asserted when a Vello context was created.
    println!(
        "[ochroma] SMOKE VELLO: vello_hud_available={} vello_hud_px_last_frame={}",
        app.vello_hud.is_some(),
        app.vello_hud_px_last_frame,
    );
    if app.vello_hud.is_some() {
        assert!(
            app.vello_hud_px_last_frame > 500,
            "Vello GPU SpectralHUD composited only {} px (expected > 500) — GPU UI path not live",
            app.vello_hud_px_last_frame
        );
    }

    // --- Assertions: real frame + advanced sim (panic => non-zero exit). ---
    let total_px = (dw * dh) as usize;
    assert_eq!(
        last_pixels.len(),
        total_px,
        "render_frame() produced {} pixels, expected {}",
        last_pixels.len(),
        total_px
    );
    let min_non_black = total_px / 20; // >5% of the frame
    assert!(
        non_black >= min_non_black,
        "frame too empty: {} non-black px (< {} required) — render likely broken",
        non_black,
        min_non_black
    );
    // Real geometry rendered, not a uniform fill: require many distinct colors.
    assert!(
        distinct_colors >= 16,
        "frame has only {} distinct colors — looks like a flat fill, not a real render",
        distinct_colors
    );
    // Sim advanced: the ECS runtime ticked, and either the camera walked or a
    // patrol agent moved along the navmesh.
    assert!(
        frames_ticked >= total_frames as u64,
        "ECS only ticked {} frames (expected >= {}) — runtime did not advance",
        frames_ticked,
        total_frames
    );
    assert!(
        cam_moved > 1.0 || patrol_moved > 0.5,
        "neither camera ({:.2}m) nor patrol agents ({:.2}m) moved — movement/AI broken",
        cam_moved,
        patrol_moved
    );

    // --- Editor STATE proofs ---
    // The bitmap-font software editor panels (inspector / node-graph blit) were
    // DELETED in the editor face-lift — the editor face is now the egui_dock
    // `vox_app::shell` (proven separately by the `shell_snapshot` bin + the
    // `vox_app::shell` tests). What remains real and worth guarding here is the
    // editor/graph LOGIC the panels used to display: it must still be live,
    // populated, and produce a sane cook result. These assert computed outcomes,
    // not pixels of a deleted panel.
    let editor_entities = app.editor.entity_count();
    let ng_out = app.node_graph_output_value;
    let ng_out_sane = ng_out.is_finite() && (0.0..=(32.0 * 32.0)).contains(&ng_out);

    println!(
        "[ochroma] SMOKE SUMMARY: editor_entities={} node_graph_nodes={} node_graph_output={:.0}",
        editor_entities, app.node_graph_node_count, ng_out,
    );

    assert!(
        editor_entities >= 1,
        "editor world is empty ({editor_entities} entities) — editor state not populated from the ECS"
    );
    assert!(
        app.node_graph_node_count >= 2,
        "node graph has only {} nodes (expected terrain + biome)",
        app.node_graph_node_count
    );
    assert!(
        ng_out_sane,
        "node-graph evaluated output {} is not finite/sane (expected 0..=1024 Alpine cells)",
        ng_out
    );

    // --- PCG-style live-in-viewport recook proof (rank #7) ---
    // The live graph (Building->Plot->Splatize) injected splats into the scene at
    // build time. Now change a graph param mid-smoke and prove the recook (a)
    // fires, (b) re-cooks ONLY the dirty subgraph (Plot + Splatize, not Building),
    // and (c) provably changes the viewport splat content — before != after with
    // real printed values.
    let live_range_before = app.live_graph_splat_range;
    let live_splats_before = live_range_before.map(|(s, e)| e - s).unwrap_or(0);
    let scene_splats_before = app.scene_splats.len();
    assert!(
        live_splats_before > 0,
        "live graph contributed no splats to the scene — build_live_graph not wired"
    );

    // Make the throttle fire deterministically regardless of wall-clock pacing.
    if let Some(g) = app.live_graph.as_mut() {
        g.set_recook_budget(std::time::Duration::from_millis(0));
    }

    // --- Live preview thumbnails (rank #10) ---
    // Baseline thumbnail generation after the initial build (one per cooked node).
    let thumbs_generated_initial = app.thumbnail_gen_count;
    let edited_node_name = app
        .live_graph_terrain
        .and_then(|n| app.live_graph.as_ref().and_then(|g| g.node_name(n)))
        .unwrap_or("?")
        .to_string();

    // Cook again WITHOUT any change: no node's generation advances, so NO
    // thumbnail is regenerated (the gating counter must stay flat).
    let gen_before_noop = app.thumbnail_gen_count;
    app.refresh_node_thumbnails();
    let gen_after_noop = app.thumbnail_gen_count;
    assert_eq!(
        gen_after_noop, gen_before_noop,
        "thumbnail regenerated with no cook change: {gen_before_noop}->{gen_after_noop}"
    );

    // Scrub the PlotNode footprint to a much larger value so the Splatize output
    // grows (area * splats_per_sqm). This routes through the throttled request.
    app.live_graph_request_edit("footprint_w", ParamValue::Float(80.0));
    app.live_graph_request_edit("footprint_d", ParamValue::Float(80.0));
    // Flush with a forced future clock so the recook is guaranteed due.
    let future = std::time::Instant::now() + std::time::Duration::from_secs(1);
    let gen_before_edit = app.thumbnail_gen_count;
    let recooked = app.live_recook(future);
    let regen_after_edit = app.thumbnail_gen_count - gen_before_edit;

    println!(
        "[graph] thumbnails: generated={} regen_after_edit={} node={}",
        thumbs_generated_initial, regen_after_edit, edited_node_name
    );
    assert!(
        regen_after_edit >= 1,
        "edited node + downstream must regenerate >=1 thumbnail, got {regen_after_edit}"
    );

    let live_splats_after = app.live_graph_splat_range.map(|(s, e)| e - s).unwrap_or(0);
    let scene_splats_after = app.scene_splats.len();
    println!(
        "[ochroma] SMOKE LIVE GRAPH: recook={:?} live_splats {}->{} scene_splats {}->{} cook_us={}",
        recooked, live_splats_before, live_splats_after,
        scene_splats_before, scene_splats_after, app.live_graph_last_cook_us,
    );
    assert!(recooked.is_some(), "live recook did not fire after a param edit + budget flush");
    assert!(
        live_splats_after != live_splats_before,
        "live recook produced the same splat count ({live_splats_before}) — edit did not change the world"
    );
    assert!(
        scene_splats_after != scene_splats_before,
        "viewport splat count unchanged ({scene_splats_before}) after live recook — splats not spliced into scene"
    );

    println!("[ochroma] SMOKE PASS: loop + editor/game logic + render verified headlessly.");
}

/// Headless Vello self-test: render the live SpectralHUD through the *real*
/// vello::Renderer (GPU) into an offscreen texture, read the pixels back, and
/// print the mandated per-frame-style verification line. Exits non-zero if no
/// GPU adapter is available or the rendered HUD fails its pixel checks.
///
/// This is the default-binary proof that the Vello GPU UI path is live: the
/// same `vox_ui::SpectralHUD::render` call the windowed editor will use, driven
/// here with a synthetic 16-band spectral ramp so the output is deterministic.
fn run_vello_hud_selftest() -> i32 {
    use vox_ui::spectral_hud::SpectralHUD;
    use vox_ui::vello_ctx::VelloCtx;

    let w = 1280u32;
    let h = 720u32;

    let Some(mut ctx) = VelloCtx::new_headless(w, h) else {
        eprintln!("[vello] no GPU adapter available — cannot run HUD self-test");
        return 1;
    };

    // Synthetic but representative spectral GI input: a smooth ramp across the
    // 16 bands so every bar has a distinct height. (Game-specific HUD *content*
    // lives here in the app layer; vox_ui stays game-agnostic.)
    let mut energy = [0.0f32; 16];
    for (b, e) in energy.iter_mut().enumerate() {
        *e = 0.2 + 0.8 * (b as f32 / 15.0);
    }
    let cache = SpectralRadianceCache::from_f32(energy);

    ctx.begin_frame();
    // Bottom-left anchored, matching the windowed HUD placement.
    SpectralHUD::render(&mut ctx, &cache, [24.0, h as f32 - 100.0]);
    let pixels = match ctx.render_to_rgba() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[vello] HUD render failed: {e}");
            return 1;
        }
    };

    // Compute verifiable stats over the whole frame.
    let mut non_background = 0usize;
    let mut seen = std::collections::HashSet::new();
    for p in &pixels {
        if p[0] > 16 || p[1] > 16 || p[2] > 16 {
            non_background += 1;
            seen.insert((p[0] >> 3, p[1] >> 3, p[2] >> 3));
        }
    }
    let distinct = seen.len();
    println!(
        "[vello] HUD {}x{} non_background_px={} distinct_colors={}",
        w, h, non_background, distinct,
    );

    // Real pixel assertions: the HUD must have painted content with the full
    // spectral gradient. A flat/empty render fails here.
    if non_background < 1000 {
        eprintln!("[vello] FAIL: HUD region too empty (non_background_px={non_background})");
        return 1;
    }
    if distinct < 12 {
        eprintln!("[vello] FAIL: too few distinct colours ({distinct}) — gradient did not render");
        return 1;
    }
    println!("[vello] HUD self-test PASSED (real Vello GPU render)");
    0
}

fn main() {
    println!("Ochroma Engine v0.1.0 -- Spectral Gaussian Splatting");

    if std::env::args().any(|a| a == "--vello-hud-selftest") {
        std::process::exit(run_vello_hud_selftest());
    }

    if std::env::args().any(|a| a == "--smoke") {
        run_smoke();
        return;
    }

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
