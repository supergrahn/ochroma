//! Render Showcase -- exercises every vox_render, vox_core, vox_audio,
//! vox_net, vox_nn, vox_data, and vox_script module in one binary.

use glam::{Mat4, Quat, Vec3};
use std::path::PathBuf;

// === vox_render modules ===
use vox_render::camera::CameraController;
use vox_render::spectral_shift::{
    apply_wear_shift, apply_weather_shift, time_of_day_illuminant_blend,
    WeatherState as RenderWeather,
};
use vox_render::frustum::Frustum;
use vox_render::lod::{reduce_splat_indices, select_lod};
use vox_render::destruction::{apply_destruction_masks, generate_debris, DestructionMask};
use vox_render::particles::{ParticleEmitter, ParticleSystem};
use vox_render::animation::{AnimationClip, AnimationPlayer, Bone, Keyframe, Skeleton};
use vox_render::postprocess::{apply_tone_mapping, PostProcessPipeline, ToneMapping};
use vox_render::material_graph::{MaterialNode, SpectralMaterialGraph};
use vox_render::lighting::{sky_color, LightManager, PointLight, SunModel};
use vox_render::denoiser::SpectralDenoiser;
use vox_render::lod_crossfade::LodCrossfadeManager;
use vox_render::water::WaterSurface;
use vox_render::atmosphere::{compute_fog, compute_god_ray_intensity, compute_sky_color, AtmosphereParams};
use vox_render::subsurface::SubsurfaceProfile;
use vox_render::spectra_render::render_with_spectra_u8;
use vox_render::perf_inspector::{PerfInspector, PerfSnapshot, FrameBreakdown, VramBreakdown, EntityBreakdown};
use vox_render::spectra_bridge::{QualityPreset, RenderConfig};
use vox_render::vr::VrSession;
use vox_render::upscaling::{UpscaleManager, UpscaleQuality};
use vox_render::svt::{TileCache, TileId};
use vox_render::cinematic::CinematicCamera;
use vox_render::gi_cache::GICache;
use vox_render::spatial_ui::{PanelContent, SpatialUIManager};
use vox_render::hand_tracking::{GestureRecognizer, HandState, InteractionManager};
use vox_render::ar_placement::ARSession;
use vox_render::hierarchical_lod::{generate_lod_chain, MicroDetailGenerator};
use vox_render::benchmark::{generate_benchmark_splats, BenchmarkSuite};
use vox_render::memory_pool::{BufferPool, RingBuffer};
use vox_render::comparison::{compare_engines, ochroma_profile, unreal5_profile};
use vox_render::web_renderer::Platform;
use vox_render::telemetry::{FrameMetrics, TelemetryCollector};
use vox_render::streaming::TileManager;
use vox_render::profiling;
use vox_render::gpu::shadow_catcher::generate_shadow_catcher;
use vox_render::gpu::entity_buffer::EntityIdBuffer;
use vox_render::gpu::instancing::InstanceManager;

// === vox_core modules ===
use vox_core::undo::UndoStack;
use vox_core::ecs::{LodLevel as EcsLod, SplatInstanceComponent};
use vox_core::error::{recover, EngineError};
use vox_core::svo::SpatialHash;
use vox_core::lwc::{TileCoord, WorldCoord};
use vox_core::terrain::{generate_terrain_splats, TerrainPlane};
use vox_core::mapgen::generate_map;
use vox_core::input::{InputState, KeyBindings};
use vox_core::navmesh::NavMesh;
use vox_core::game_loop::{GameClock, GamePhase};
use vox_core::i18n::{I18nManager, Locale};
use vox_core::script_interface::ScriptRegistry;
use vox_core::spectral::{Illuminant, SpectralBands};

// === vox_audio ===
use vox_audio::acoustic_raytracer::{trace_sound, AcousticScene};

// === vox_net ===
use vox_net::lobby::LobbyState;
use vox_net::crdt::OperationLog;
use vox_net::world_hosting::WorldHost;

// === vox_nn ===
use vox_nn::history_gen::generate_history;
use vox_nn::scene_query::SceneQueryEngine;
use vox_nn::nl_commands::parse_command;
use vox_nn::text_to_city::generate_district_from_prompt;
use vox_nn::llm_client::{LlmClient, LlmProvider};

// === vox_data ===
use vox_data::marketplace::MarketplaceCache;
use vox_data::templates::available_templates;
use vox_data::creator_tools::{BrushStroke, TerrainSculptOp};
use vox_data::neural_compress::NeuralCompressor;
use vox_data::osm_import;
use vox_data::hot_reload::AssetWatcher;
use vox_data::asset_catalog::default_catalog;

// === vox_script ===
use vox_script::plugin_system::PluginManager;
use vox_script::visual_script::VisualScript;

fn main() {
    println!("=== Render Showcase ===");
    println!("Exercising every module across vox_render, vox_core, vox_audio, vox_net, vox_nn, vox_data, and vox_script.\n");

    let mut count = 0u32;

    // ---------------------------------------------------------------
    // vox_render modules
    // ---------------------------------------------------------------

    // 1. camera
    let mut camera = CameraController::new(16.0 / 9.0);
    camera.orbit(0.1);
    camera.zoom(-5.0);
    let vp = camera.view_proj();
    count += 1;
    println!("[{:02}] camera::CameraController -- view_proj computed", count);

    // 2. spectral_shift
    let base = SpectralBands([0.5; 8]);
    let shifted = apply_weather_shift(&base, RenderWeather::Overcast);
    let worn = SpectralBands([0.3; 8]);
    let _wear = apply_wear_shift(&base, &worn, 0.5);
    let (r, g, b) = time_of_day_illuminant_blend(12.0);
    let _ = (shifted, r, g, b);
    count += 1;
    println!("[{:02}] spectral_shift -- weather/wear/time-of-day shifts", count);

    // 3. frustum
    let frustum = Frustum::from_view_proj(vp);
    let inside = frustum.contains_sphere(Vec3::new(0.0, 0.0, -10.0), 1.0);
    count += 1;
    println!("[{:02}] frustum::Frustum -- contains_sphere={}", count, inside);

    // 4. lod
    let lod = select_lod(50.0);
    let indices: Vec<usize> = (0..100).collect();
    let reduced = reduce_splat_indices(&indices, 0.5);
    count += 1;
    println!("[{:02}] lod -- select_lod={:?}, reduced {} to {}", count, lod, indices.len(), reduced.len());

    // 5. destruction
    let splats = generate_benchmark_splats(20);
    let mask = DestructionMask {
        instance_id: 0,
        impact_point: Vec3::ZERO,
        radius: 5.0,
        progression: 0.5,
    };
    let damaged = apply_destruction_masks(&splats, &[mask]);
    let debris = generate_debris(Vec3::ZERO, 3.0, 10, 42);
    count += 1;
    println!("[{:02}] destruction -- {} damaged splats, {} debris", count, damaged.len(), debris.len());

    // 6. particles
    let mut particles = ParticleSystem::new(1000);
    particles.add_emitter(ParticleEmitter::smoke(Vec3::new(0.0, 5.0, 0.0)));
    particles.add_emitter(ParticleEmitter::dust(Vec3::new(10.0, 0.0, 0.0)));
    particles.tick(0.016);
    let particle_splats = particles.to_splats();
    count += 1;
    println!("[{:02}] particles -- {} particles, {} splats", count, particles.particle_count(), particle_splats.len());

    // 7. animation
    let mut skeleton = Skeleton::new();
    skeleton.add_bone(Bone {
        id: 0,
        name: "root".into(),
        parent_id: None,
        local_transform: Mat4::IDENTITY,
    });
    let mut clip = AnimationClip::new("idle", 2.0);
    clip.add_keyframe(0, Keyframe {
        time: 0.0,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    });
    clip.add_keyframe(0, Keyframe {
        time: 1.0,
        position: Vec3::new(0.0, 1.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    });
    let mut player = AnimationPlayer::new(vec![clip]);
    player.update(0.5);
    count += 1;
    println!("[{:02}] animation -- skeleton with {} bones, player updated", count, skeleton.compute_world_transforms().len());

    // 8. postprocess
    let pipeline = PostProcessPipeline::default();
    let mut pixels = vec![[0.8_f32, 0.6, 0.4, 1.0]; 64];
    pipeline.apply(&mut pixels, 8, 8);
    apply_tone_mapping(&mut pixels, ToneMapping::ACES);
    count += 1;
    println!("[{:02}] postprocess -- tone mapping + bloom applied to {} pixels", count, pixels.len());

    // 9. material_graph
    let mat = SpectralMaterialGraph {
        name: "showcase_mat".into(),
        albedo: MaterialNode::Constant { spd: [0.8, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002] },
        roughness: 0.5,
        metallic: 0.0,
        emission: None,
    };
    let albedo = mat.evaluate_albedo();
    let _ = albedo;
    count += 1;
    println!("[{:02}] material_graph -- evaluated albedo SPD", count);

    // 10. lighting
    let sun = SunModel::new(45.0);
    let sun_dir = sun.sun_direction(14.0, 180);
    let sky = sky_color(sun_dir, Vec3::new(0.0, 1.0, 0.0));
    let mut lights = LightManager::new(45.0);
    lights.add_point_light(PointLight {
        position: Vec3::new(10.0, 5.0, 0.0),
        intensity: 100.0,
        radius: 20.0,
        color: [1.0, 0.9, 0.7],
    });
    count += 1;
    println!("[{:02}] lighting -- sun_dir={:?}, sky=[{:.2},{:.2},{:.2}], {} point lights",
        count, sun_dir, sky[0], sky[1], sky[2], lights.point_light_count());

    // 11. denoiser
    let denoiser = SpectralDenoiser::new(0.5);
    let mut img = vec![[128u8, 128, 128, 255]; 16];
    denoiser.denoise(&mut img, 4, 4);
    count += 1;
    println!("[{:02}] denoiser -- denoised 4x4 image", count);

    // 12. lod_crossfade
    let mut crossfade = LodCrossfadeManager::new(0.5);
    crossfade.request_lod_change(0, 0, 1);
    crossfade.tick(0.25);
    count += 1;
    println!("[{:02}] lod_crossfade -- transition in progress", count);

    // 13. water
    let lake = WaterSurface::lake(Vec3::new(50.0, 0.0, 50.0), 30.0);
    let water_splats = lake.generate_splats(0.0);
    count += 1;
    println!("[{:02}] water -- lake generated {} splats", count, water_splats.len());

    // 14. atmosphere
    let atmo = AtmosphereParams::default();
    let sky_c = compute_sky_color(Vec3::new(0.0, 1.0, 0.0), sun_dir, &atmo);
    let (fog_color, fog_factor) = compute_fog(500.0, 0.002, [0.7, 0.7, 0.8]);
    let god_ray = compute_god_ray_intensity(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 0.3, -1.0), sun_dir, 100.0, 16);
    let _ = fog_color;
    count += 1;
    println!("[{:02}] atmosphere -- sky=[{:.2},{:.2},{:.2}], fog={:.2}, god_ray={:.2}",
        count, sky_c[0], sky_c[1], sky_c[2], fog_factor, god_ray);

    // 15. subsurface
    let sss = SubsurfaceProfile::vegetation();
    let transmitted = sss.transmit(&base, 0.5);
    let _ = transmitted;
    count += 1;
    println!("[{:02}] subsurface -- vegetation profile transmit computed", count);

    // 16. spectra_render
    let render_cam = vox_render::spectral::RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y),
        proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0),
    };
    let illuminant = Illuminant::d65();
    let frame = render_with_spectra_u8(&splats, &render_cam, 16, 16, &illuminant);
    count += 1;
    println!("[{:02}] spectra_render -- rendered {}x{} = {} pixels", count, 16, 16, frame.len());

    // 17. perf_inspector
    let mut perf = PerfInspector::new();
    perf.record_frame(PerfSnapshot {
        frame: FrameBreakdown {
            total_ms: 16.6,
            sort_ms: 2.0,
            cull_ms: 1.5,
            render_ms: 8.0,
            ui_ms: 1.0,
            sim_ms: 4.1,
        },
        vram: VramBreakdown {
            total_mb: 896.0,
            splats_mb: 512.0,
            textures_mb: 256.0,
            buffers_mb: 128.0,
        },
        entities: EntityBreakdown {
            total: 1_000_000,
            buildings: 500,
            citizens: 10_000,
            vehicles: 2_000,
            trees: 5_000,
            props: 3_000,
        },
    });
    count += 1;
    println!("[{:02}] perf_inspector -- {} frames recorded", count, perf.frame_count());

    // 18. spectra_bridge
    let _config = RenderConfig::default();
    let preset = QualityPreset::Medium;
    count += 1;
    println!("[{:02}] spectra_bridge -- denoiser={}, shadows={}", count, preset.enable_denoiser(), preset.enable_shadows());

    // 19. vr
    let mut vr = VrSession::new_simulated();
    vr.simulate_head_turn(0.1, 0.0);
    let stereo = vr.needs_stereo();
    count += 1;
    println!("[{:02}] vr -- simulated session, stereo={}, target_ms={:.1}", count, stereo, vr.target_frame_ms());

    // 20. upscaling
    let upscaler = UpscaleManager::new(1920, 1080, UpscaleQuality::Balanced);
    let (rw, rh) = upscaler.render_resolution();
    let small_pixels = vec![[128u8, 128, 128, 255]; (rw * rh) as usize];
    let upscaled = upscaler.upscale(&small_pixels, rw, rh);
    count += 1;
    println!("[{:02}] upscaling -- {}x{} -> {} upscaled pixels", count, rw, rh, upscaled.len());

    // 21. svt
    let mut tile_cache = TileCache::new(256);
    let tid = TileId { x: 0, y: 0, mip_level: 0 };
    tile_cache.touch(tid);
    tile_cache.mark_loaded(tid);
    count += 1;
    println!("[{:02}] svt -- {} tiles cached, loaded={}", count, tile_cache.tile_count(), tile_cache.is_loaded(tid));

    // 22. cinematic
    let mut cine = CinematicCamera::new();
    cine.add_keyframe(0.0, Vec3::new(0.0, 10.0, 20.0), Vec3::ZERO, 60.0);
    cine.add_keyframe(5.0, Vec3::new(20.0, 5.0, 0.0), Vec3::ZERO, 45.0);
    cine.tick(2.5);
    let dof = &cine.dof;
    let _coc = dof.coc_radius(15.0, 36.0);
    count += 1;
    println!("[{:02}] cinematic -- duration={:.1}s, DOF focal={}", count, cine.duration(), cine.dof.focal_distance);

    // 23. gi_cache
    let mut gi = GICache::new(2.0, 2);
    gi.add_bounce(Vec3::new(5.0, 0.0, 5.0), SpectralBands([0.5; 8]), 0);
    let irr = gi.query(Vec3::new(5.0, 0.0, 5.0));
    count += 1;
    println!("[{:02}] gi_cache -- {} cells, query={}", count, gi.cell_count(), irr.is_some());

    // 24. spatial_ui
    let mut spatial_ui = SpatialUIManager::new();
    let panel_id = spatial_ui.add_panel(Vec3::new(0.0, 2.0, -3.0), [1.0, 0.5], PanelContent::Text);
    let _ = panel_id;
    count += 1;
    println!("[{:02}] spatial_ui -- panel placed in world", count);

    // 25. hand_tracking
    let recognizer = GestureRecognizer::new();
    let hand = HandState::default();
    let gesture = recognizer.recognize(&hand);
    let interaction_mgr = InteractionManager::new();
    let action = interaction_mgr.update(&hand);
    let _ = (gesture, action);
    count += 1;
    println!("[{:02}] hand_tracking -- gesture recognized, interaction computed", count);

    // 26. ar_placement
    let mut ar = ARSession::new();
    ar.start();
    let _surface_id = ar.detect_horizontal(Vec3::new(0.0, 0.0, 0.0), [2.0, 2.0]);
    ar.stop();
    count += 1;
    println!("[{:02}] ar_placement -- session start/detect/stop", count);

    // 27. hierarchical_lod
    let lod_chain = generate_lod_chain(&splats);
    let micro = MicroDetailGenerator::generate_brick_detail(42);
    count += 1;
    println!("[{:02}] hierarchical_lod -- chain levels={}, brick detail={} splats",
        count, lod_chain.levels.len(), micro.len());

    // 28. benchmark
    let _bench = BenchmarkSuite::new();
    let bench_splats = generate_benchmark_splats(1000);
    count += 1;
    println!("[{:02}] benchmark -- generated {} splats for benchmarking", count, bench_splats.len());

    // 29. memory_pool
    let mut pool = BufferPool::new(4, 1024);
    let buf = pool.acquire();
    assert!(buf.is_some());
    let stats = pool.stats();
    let mut ring = RingBuffer::new(4096);
    let offset = ring.write(&[1, 2, 3, 4]);
    count += 1;
    println!("[{:02}] memory_pool -- pool: {} allocated, ring offset={}", count, stats.total_allocated, offset);

    // 30. comparison
    let ochroma = ochroma_profile();
    let unreal = unreal5_profile();
    let report = compare_engines(&ochroma, &unreal);
    count += 1;
    println!("[{:02}] comparison -- Ochroma vs Unreal5: {} advantages", count, report.advantages.len());

    // 31. web_renderer
    let web_config = Platform::WebBrowser.recommended_config();
    let fits = web_config.fits_vram_budget(2048.0);
    count += 1;
    println!("[{:02}] web_renderer -- {}x{} @ {}x DPR, fits_budget={}", count,
        web_config.canvas_width, web_config.canvas_height, web_config.pixel_ratio, fits);

    // 32. telemetry
    let mut telemetry = TelemetryCollector::new(100);
    telemetry.record(FrameMetrics {
        frame_time_ms: 16.6,
        sort_time_ms: 2.0,
        rasterize_time_ms: 8.0,
        present_time_ms: 1.0,
        splat_count_visible: 500_000,
        splat_count_culled: 200_000,
        instance_count: 100,
        vram_usage_mb: 512.0,
    });
    count += 1;
    println!("[{:02}] telemetry -- avg_fps={:.1}", count, telemetry.avg_fps());

    // 33. streaming
    let mut tile_mgr = TileManager::new();
    tile_mgr.update_camera(TileCoord { x: 0, z: 0 });
    let active = tile_mgr.active_tiles();
    count += 1;
    println!("[{:02}] streaming -- {} active tiles", count, active.len());

    // 34. profiling
    profiling::begin_frame();
    count += 1;
    println!("[{:02}] profiling -- frame begun", count);

    // 35. gpu::shadow_catcher
    let shadow_mesh = generate_shadow_catcher(&splats);
    count += 1;
    println!("[{:02}] gpu::shadow_catcher -- {} vertices", count, shadow_mesh.vertices.len());

    // 36. gpu::entity_buffer
    let mut entity_buf = EntityIdBuffer::new(64, 64);
    entity_buf.write(10, 10, 42);
    let picked = entity_buf.pick(10, 10);
    count += 1;
    println!("[{:02}] gpu::entity_buffer -- wrote entity {}, picked={}", count, 42, picked);

    // 37. gpu::instancing
    let mut instancer = InstanceManager::new();
    instancer.register_asset(uuid::Uuid::new_v4(), 100);
    count += 1;
    println!("[{:02}] gpu::instancing -- assets registered", count);

    // ---------------------------------------------------------------
    // vox_core modules
    // ---------------------------------------------------------------

    // 38. undo
    let _undo: UndoStack = UndoStack::new(100);
    count += 1;
    println!("[{:02}] vox_core::undo -- stack created", count);

    // 39. ecs
    let _ecs_lod = EcsLod::Full;
    let _splat_comp = SplatInstanceComponent {
        asset_uuid: uuid::Uuid::new_v4(),
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
        instance_id: 0,
        lod: EcsLod::Full,
    };
    count += 1;
    println!("[{:02}] vox_core::ecs -- SplatInstanceComponent created", count);

    // 40. error
    let recovered: f32 = recover(EngineError::AssetMissing { uuid: "test-uuid".into() });
    let _ = recovered;
    count += 1;
    println!("[{:02}] vox_core::error -- recovered from AssetMissing", count);

    // 41. svo
    let _spatial = SpatialHash::new(10.0);
    count += 1;
    println!("[{:02}] vox_core::svo -- spatial hash created", count);

    // 42. lwc
    let wc = WorldCoord::from_absolute(1000.0, 0.0, 2000.0);
    let tc = wc.tile;
    count += 1;
    println!("[{:02}] vox_core::lwc -- world -> tile({},{})", count, tc.x, tc.z);

    // 43. terrain
    let terrain = TerrainPlane::new(100.0, 100.0, 0.5);
    let terrain_splats = generate_terrain_splats(&terrain, "grass");
    count += 1;
    println!("[{:02}] vox_core::terrain -- {} terrain splats", count, terrain_splats.len());

    // 44. mapgen
    let map_splats = generate_map(42, 50.0, 0.5);
    count += 1;
    println!("[{:02}] vox_core::mapgen -- generated {} splats", count, map_splats.len());

    // 45. input
    let _input = InputState::default();
    let _bindings = KeyBindings::default();
    count += 1;
    println!("[{:02}] vox_core::input -- input state and bindings ready", count);

    // 46. navmesh
    let _nav = NavMesh::new();
    count += 1;
    println!("[{:02}] vox_core::navmesh -- navmesh created", count);

    // 47. game_loop
    let _clock = GameClock::new(1.0 / 60.0);
    let _phase = GamePhase::Input;
    count += 1;
    println!("[{:02}] vox_core::game_loop -- clock created, phase={:?}", count, GamePhase::Input);

    // 48. i18n
    let _i18n = I18nManager::new(Locale::En);
    count += 1;
    println!("[{:02}] vox_core::i18n -- locale={:?}", count, Locale::En);

    // 49. script_interface
    let _registry = ScriptRegistry::new();
    count += 1;
    println!("[{:02}] vox_core::script_interface -- registry created", count);

    // ---------------------------------------------------------------
    // vox_audio
    // ---------------------------------------------------------------

    // 50. acoustic_raytracer
    let scene = AcousticScene { surfaces: vec![] };
    let result = trace_sound(Vec3::new(0.0, 1.0, 0.0), Vec3::new(10.0, 1.0, 0.0), &scene, 3);
    count += 1;
    println!("[{:02}] vox_audio::acoustic_raytracer -- rt60={:.2}s", count, result.rt60);

    // ---------------------------------------------------------------
    // vox_net
    // ---------------------------------------------------------------

    // 51. lobby
    let mut lobby = LobbyState::new("Host", "TestCity", 8);
    let _pid = lobby.add_player("Player1", vox_net::lobby::PlayerRole::Spectator);
    count += 1;
    println!("[{:02}] vox_net::lobby -- {} players", count, lobby.player_count());

    // 52. crdt
    let mut crdt = OperationLog::new(1, 1000);
    crdt.apply_local(1, "position", vox_net::crdt::OpType::Set, vec![1, 2, 3, 4]);
    count += 1;
    println!("[{:02}] vox_net::crdt -- {} operations", count, crdt.operation_count());

    // 53. world_hosting
    let mut world_host = WorldHost::new();
    let world_id = world_host.create_world("TestWorld", "admin", 16);
    count += 1;
    println!("[{:02}] vox_net::world_hosting -- world created={}", count, world_id.is_ok());

    // ---------------------------------------------------------------
    // vox_nn
    // ---------------------------------------------------------------

    // 54. history_gen
    let history = generate_history(42);
    count += 1;
    println!("[{:02}] vox_nn::history_gen -- {} eras generated", count, history.eras.len());

    // 55. scene_query
    let scene_query = SceneQueryEngine::new();
    count += 1;
    println!("[{:02}] vox_nn::scene_query -- engine created, {} entities", count, scene_query.entity_count());

    // 56. nl_commands
    let cmd = parse_command("build a hospital at the center");
    count += 1;
    println!("[{:02}] vox_nn::nl_commands -- parsed={}", count, cmd.is_some());

    // 57. text_to_city
    let client = LlmClient::new(LlmProvider::Mock);
    let district = generate_district_from_prompt(&client, "a cozy residential neighborhood", 42);
    count += 1;
    println!("[{:02}] vox_nn::text_to_city -- generated={}", count, district.is_ok());

    // 58. llm_client
    // (already created above)
    let _ = &client;
    count += 1;
    println!("[{:02}] vox_nn::llm_client -- mock client created", count);

    // ---------------------------------------------------------------
    // vox_data
    // ---------------------------------------------------------------

    // 59. marketplace
    let cache = MarketplaceCache::new();
    let results = cache.search("house");
    count += 1;
    println!("[{:02}] vox_data::marketplace -- {} results for 'house'", count, results.len());

    // 60. templates
    let templates = available_templates();
    count += 1;
    println!("[{:02}] vox_data::templates -- {} templates available", count, templates.len());

    // 61. creator_tools
    let brush = BrushStroke {
        position: [0.0, 0.0, 0.0],
        radius: 5.0,
        material: "grass".into(),
        pressure: 0.8,
        timestamp: 0.0,
    };
    let sculpt = TerrainSculptOp::Raise;
    let _ = (brush.radius, sculpt);
    count += 1;
    println!("[{:02}] vox_data::creator_tools -- brush and sculpt ops ready", count);

    // 62. neural_compress
    let compressor = NeuralCompressor::new(vox_data::neural_compress::CompressionQuality::Balanced);
    let est = compressor.estimate_compressed_size(10000);
    count += 1;
    println!("[{:02}] vox_data::neural_compress -- estimated {}B for 10k splats", count, est);

    // 63. osm_import
    let parsed = osm_import::parse_osm_json("{}");
    count += 1;
    println!("[{:02}] vox_data::osm_import -- parse result={}", count, if parsed.is_ok() { "ok" } else { "empty" });

    // 64. hot_reload
    let mut watcher = AssetWatcher::new(1.0);
    watcher.watch(PathBuf::from("/tmp/test_asset.vxm"));
    count += 1;
    println!("[{:02}] vox_data::hot_reload -- watching {} paths", count, watcher.watched_count());

    // 65. asset_catalog
    let catalog = default_catalog();
    count += 1;
    println!("[{:02}] vox_data::asset_catalog -- {} entries", count, catalog.len());

    // ---------------------------------------------------------------
    // vox_script
    // ---------------------------------------------------------------

    // 66. plugin_system
    let _plugin_mgr = PluginManager::new("0.1.0");
    count += 1;
    println!("[{:02}] vox_script::plugin_system -- manager created", count);

    // 67. visual_script
    let mut vs = VisualScript::new("test_script");
    let node_id = vs.add_node(vox_script::visual_script::ScriptNode::Event("tick".into()));
    count += 1;
    println!("[{:02}] vox_script::visual_script -- script with node {}", count, node_id);

    // ---------------------------------------------------------------
    // Save a PPM image from the rendered frame
    // ---------------------------------------------------------------
    let ppm_path = "render_showcase_output.ppm";
    save_ppm(ppm_path, 16, 16, &frame);
    println!("\nFrame saved to {}", ppm_path);

    // ---------------------------------------------------------------
    // Final tally
    // ---------------------------------------------------------------
    println!("\n=== All {} modules exercised successfully ===", count);
}

fn save_ppm(path: &str, width: u32, height: u32, pixels: &[[u8; 4]]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create PPM file");
    write!(f, "P6\n{} {}\n255\n", width, height).unwrap();
    for px in pixels {
        f.write_all(&[px[0], px[1], px[2]]).unwrap();
    }
}
