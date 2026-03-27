mod demo_asset;
pub mod autosave;
pub mod debug_console;
pub mod render_to_file;
pub mod screenshot;
pub mod shortcut_help;
pub mod daytime;
pub mod growth;
pub mod headless;
pub mod overlays;
pub mod persistence;
pub mod placement;
pub mod road_builder;
pub mod simulation;
pub mod steam;
pub mod systems;
pub mod terrain_setup;
pub mod ui;
pub mod undo_integration;
pub mod notifications;
pub mod minimap;
pub mod settings;
pub mod soundscape;
pub mod tutorial;

use std::sync::Arc;
use std::time::Instant;

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};
use uuid::Uuid;
use vox_core::ecs::{LodLevel, SplatAssetComponent, SplatInstanceComponent};
use vox_core::types::GaussianSplat;
use vox_render::camera::CameraController;
use vox_render::gpu::gpu_rasteriser::GpuRasteriser;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use egui_wgpu::wgpu;
use ui::PlopUi;

use daytime::EnvironmentState;
use simulation::SimulationState;
use systems::{
    CameraState, VisibleSplats, frustum_cull_system, gather_splats_system, lod_select_system,
};
use undo_integration::GameUndoSystem;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// The render backend — either GPU-accelerated or CPU software fallback.
enum RenderMode {
    Gpu {
        backend: WgpuBackend,
        gpu_rasteriser: GpuRasteriser,
        egui_state: egui_winit::State,
        egui_renderer: egui_wgpu::Renderer,
        egui_ctx: egui::Context,
    },
    Software {
        backend: WgpuBackend,
        rasteriser: SoftwareRasteriser,
    },
    /// Pure CPU fallback — no GPU at all. The app still runs but cannot present
    /// to a window surface (headless-style). We print frames to console only.
    CpuOnly {
        rasteriser: SoftwareRasteriser,
    },
}

struct App {
    /// Set once `resumed` fires.
    window: Option<Arc<Window>>,
    render_mode: Option<RenderMode>,
    /// Bevy ECS world — owns all entities and resources.
    world: World,
    /// Bevy ECS schedule — runs frustum cull, LOD select, and gather systems.
    schedule: Schedule,
    /// Interactive camera controller.
    camera: CameraController,
    last_frame: Instant,
    /// Instant of last FPS print.
    fps_timer: Instant,
    frame_count: u64,
    plop_ui: PlopUi,
    /// The UUID of the demo building asset, for populating the asset browser.
    demo_asset_uuid: Uuid,
    /// Mouse state for interactive camera.
    middle_pressed: bool,
    right_pressed: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
    /// Undo/redo system for game actions.
    undo_system: GameUndoSystem,
    /// Environment state: time of day and weather.
    environment: EnvironmentState,
    /// Tracks whether Ctrl is currently held.
    ctrl_pressed: bool,
    /// Accumulates real time for growth ticks (fires every 2 seconds).
    growth_timer: f32,
}

impl App {
    fn new() -> Self {
        // Build the demo asset once.
        let asset = demo_asset::generate_building();

        // Set up ECS world with resources.
        let mut world = World::new();
        world.insert_resource(CameraState {
            position: Vec3::ZERO,
            view_proj: Mat4::IDENTITY,
        });
        world.insert_resource(VisibleSplats::default());
        world.insert_resource(SimulationState::new());

        // Spawn asset entity.
        let asset_uuid = asset.header.uuid();
        world.spawn(SplatAssetComponent {
            uuid: asset_uuid,
            splat_count: asset.splats.len() as u32,
            splats: asset.splats.clone(),
        });

        // Spawn instance entities: two buildings side by side.
        let offsets = [Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0)];
        for (i, offset) in offsets.iter().enumerate() {
            world.spawn(SplatInstanceComponent {
                asset_uuid,
                position: *offset,
                rotation: glam::Quat::IDENTITY,
                scale: 1.0,
                instance_id: i as u32,
                lod: LodLevel::Full,
            });
        }

        // Spawn terrain ground plane under the buildings.
        terrain_setup::spawn_terrain(&mut world, 200.0, 200.0, "grass");

        // Set up ECS schedule with chained systems.
        let mut schedule = Schedule::default();
        schedule.add_systems(
            (frustum_cull_system, lod_select_system, gather_splats_system).chain(),
        );

        // Set up interactive camera pointing at the buildings.
        let mut camera = CameraController::new(WIDTH as f32 / HEIGHT as f32);
        camera.target = Vec3::new(10.0, 7.5, 6.0);
        camera.orbit_distance = 50.0;
        camera.altitude = 18.0;
        camera.orbit_angle = 0.0;
        camera.update_position_public();

        let now = Instant::now();
        Self {
            window: None,
            render_mode: None,
            world,
            schedule,
            camera,
            last_frame: now,
            fps_timer: now,
            frame_count: 0,
            plop_ui: PlopUi::default(),
            demo_asset_uuid: asset_uuid,
            middle_pressed: false,
            right_pressed: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            undo_system: GameUndoSystem::new(),
            environment: EnvironmentState::default(),
            ctrl_pressed: false,
            growth_timer: 0.0,
        }
    }

    fn egui_wants_input(&self) -> bool {
        match &self.render_mode {
            Some(RenderMode::Gpu { egui_ctx, .. }) => {
                egui_ctx.wants_pointer_input() || egui_ctx.wants_keyboard_input()
            }
            _ => false,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma — Phase 0")
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        // Try GPU backend; fall back to software if it fails.
        let render_mode = match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => {
                // Try GPU rasteriser first; if shader compilation or pipeline creation
                // panics we catch it below via the software path.
                let gpu_rasteriser = GpuRasteriser::new(
                    backend.device(),
                    backend.surface_format(),
                    WIDTH,
                    HEIGHT,
                );

                // Initialise egui
                let egui_ctx = egui::Context::default();
                let egui_state = egui_winit::State::new(
                    egui_ctx.clone(),
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

                eprintln!("[ochroma] GPU mode initialised");
                RenderMode::Gpu {
                    backend,
                    gpu_rasteriser,
                    egui_state,
                    egui_renderer,
                    egui_ctx,
                }
            }
            Err(e) => {
                eprintln!("[ochroma] GPU init failed: {e}");
                // Try a minimal wgpu backend for surface blitting with software rasteriser
                match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
                    Ok(backend) => {
                        eprintln!(
                            "[ochroma] Software rasteriser mode (with wgpu surface blit)"
                        );
                        RenderMode::Software {
                            backend,
                            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
                        }
                    }
                    Err(e2) => {
                        eprintln!("[ochroma] All GPU paths failed: {e2}");
                        eprintln!("[ochroma] Running in CPU-only mode (no window rendering)");
                        RenderMode::CpuOnly {
                            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
                        }
                    }
                }
            }
        };

        self.window = Some(window);
        self.render_mode = Some(render_mode);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Pass events to egui if in GPU mode
        if let Some(RenderMode::Gpu { egui_state, .. }) = &mut self.render_mode {
            if let Some(window) = self.window.as_ref() {
                let _ = egui_state.on_window_event(window, &event);
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                let w = size.width.max(1);
                let h = size.height.max(1);
                self.camera.aspect_ratio = w as f32 / h as f32;
                match &mut self.render_mode {
                    Some(RenderMode::Gpu {
                        backend,
                        gpu_rasteriser,
                        ..
                    }) => {
                        backend.resize(w, h);
                        gpu_rasteriser.resize(w, h);
                    }
                    Some(RenderMode::Software { backend, rasteriser }) => {
                        backend.resize(w, h);
                        rasteriser.width = w;
                        rasteriser.height = h;
                    }
                    Some(RenderMode::CpuOnly { rasteriser }) => {
                        rasteriser.width = w;
                        rasteriser.height = h;
                    }
                    None => {}
                }
            }

            // --- Camera input: mouse wheel for zoom ---
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => {
                    self.camera.zoom(-y * 5.0);
                }
                winit::event::MouseScrollDelta::PixelDelta(pos) => {
                    self.camera.zoom(-pos.y as f32 * 0.5);
                }
            },

            // --- Camera input + placement: mouse button tracking ---
            WindowEvent::MouseInput { state, button, .. } => match button {
                winit::event::MouseButton::Middle => {
                    self.middle_pressed = state.is_pressed();
                }
                winit::event::MouseButton::Right => {
                    self.right_pressed = state.is_pressed();
                }
                winit::event::MouseButton::Left => {
                    if state.is_pressed() && !self.egui_wants_input() {
                        let inv_vp = self.camera.view_proj().inverse();
                        let (origin, dir) = placement::screen_to_ray(
                            self.last_mouse_x,
                            self.last_mouse_y,
                            WIDTH,
                            HEIGHT,
                            inv_vp,
                        );
                        if let Some(ground_pos) = placement::ray_ground_intersection(origin, dir) {
                            self.plop_ui.handle_viewport_click(ground_pos, None);
                        }
                    }
                }
                _ => {}
            },

            // --- Camera input: mouse drag for orbit/pan ---
            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x as f32 - self.last_mouse_x;
                let dy = position.y as f32 - self.last_mouse_y;
                self.last_mouse_x = position.x as f32;
                self.last_mouse_y = position.y as f32;

                if self.middle_pressed {
                    self.camera.orbit(dx * 0.005);
                    self.camera
                        .set_altitude(self.camera.altitude - dy * 0.5);
                }
                if self.right_pressed {
                    self.camera.pan(dx * 0.1, dy * 0.1);
                }
            }

            // --- Track modifier keys ---
            WindowEvent::ModifiersChanged(modifiers) => {
                self.ctrl_pressed = modifiers.state().control_key();
            }

            // --- Camera input: WASD keyboard pan + Ctrl shortcuts ---
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    match event.physical_key {
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyW,
                        ) if !self.ctrl_pressed => self.camera.pan(0.0, -2.0),
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyS,
                        ) if self.ctrl_pressed => {
                            let (hours, pop, funds) = {
                                let sim = self.world.resource::<SimulationState>();
                                (sim.game_time_hours, sim.citizens.count() as u32, sim.budget.funds)
                            };
                            let _ = persistence::save_current(
                                "MyCity",
                                hours,
                                pop,
                                funds,
                                "quicksave",
                            );
                        }
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyS,
                        ) => self.camera.pan(0.0, 2.0),
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyA,
                        ) => self.camera.pan(-2.0, 0.0),
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyD,
                        ) => self.camera.pan(2.0, 0.0),
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyZ,
                        ) if self.ctrl_pressed => {
                            self.undo_system.undo();
                        }
                        winit::keyboard::PhysicalKey::Code(
                            winit::keyboard::KeyCode::KeyY,
                        ) if self.ctrl_pressed => {
                            self.undo_system.redo();
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // --- Puffin: new frame ---
                puffin::GlobalProfiler::lock().new_frame();

                // --- Timing ---
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // --- Tick simulation ---
                self.world.resource_mut::<SimulationState>().tick(dt);

                // --- Growth tick every 2 seconds ---
                self.growth_timer += dt;
                if self.growth_timer >= 2.0 {
                    self.growth_timer = 0.0;
                    growth::growth_tick(&mut self.world);
                }

                // --- Update ECS camera state resource ---
                if let Some(mut cam_state) = self.world.get_resource_mut::<CameraState>() {
                    cam_state.position = self.camera.position;
                    cam_state.view_proj = self.camera.view_proj();
                }

                // --- Run ECS systems (frustum cull -> LOD select -> gather splats) ---
                self.schedule.run(&mut self.world);

                // --- Extract visible splats from ECS for rendering ---
                let splats_to_render: Vec<GaussianSplat> = {
                    let visible = self.world.resource::<VisibleSplats>();
                    visible.splats.clone()
                };

                // --- Process pending UI actions (placements, selections) ---
                let actions = self.plop_ui.take_actions();
                if !actions.is_empty() {
                    placement::process_actions(&mut self.world, &actions);
                }

                // Asset names list for the browser: expose the demo building.
                let asset_names = vec![(self.demo_asset_uuid, "Demo Building".to_string())];

                // Pre-compute values that need &self before we mutably borrow render_mode.
                let window_clone = self.window.clone();
                let plop_ui = &mut self.plop_ui;
                let env = &self.environment;

                // Extract simulation display data from world resource before borrowing render_mode.
                let (sim_day, sim_tod, sim_pop, sim_funds, sim_info, mut sim_speed) = {
                    let sim = self.world.resource::<SimulationState>();
                    let citizens = sim.citizens.all();
                    let avg_sat = if citizens.is_empty() {
                        0.0f32
                    } else {
                        citizens.iter().map(|c| c.satisfaction).sum::<f32>() / citizens.len() as f32
                    };
                    let info = ui::SimInfo {
                        funds: sim.budget.funds,
                        total_income: sim.budget.total_income(),
                        total_expenses: sim.budget.total_expenses(),
                        net: sim.budget.net(),
                        citizen_count: sim.citizens.count() as u32,
                        avg_satisfaction: avg_sat,
                        demand_residential: sim.zoning.demand.residential,
                        demand_commercial: sim.zoning.demand.commercial,
                        demand_industrial: sim.zoning.demand.industrial,
                    };
                    (sim.day(), sim.time_of_day(), sim.citizens.count(), sim.budget.funds, info, sim.game_speed)
                };

                // Current illuminant from time-of-day environment.
                let illuminant = env.current_illuminant();
                let time_label = env.time_label();
                let weather_label = env.weather_label();

                // Build camera matrices from the interactive controller.
                let camera = RenderCamera {
                    view: self.camera.view_matrix(),
                    proj: self.camera.proj_matrix(),
                };

                match &mut self.render_mode {
                    Some(RenderMode::Gpu {
                        backend,
                        gpu_rasteriser,
                        egui_state,
                        egui_renderer,
                        egui_ctx,
                    }) => {
                        puffin::profile_scope!("render");

                        // --- GPU splat render ---
                        let output = match backend.surface().get_current_texture() {
                            Ok(t) => t,
                            Err(_) => return,
                        };
                        let view_tex = output.texture.create_view(&Default::default());

                        gpu_rasteriser.render(
                            backend.device(),
                            backend.queue(),
                            &view_tex,
                            &splats_to_render,
                            &camera,
                            &illuminant,
                        );

                        // --- egui render on top ---
                        let window = window_clone.as_ref().unwrap();
                        let raw_input = egui_state.take_egui_input(window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            egui::TopBottomPanel::top("game_info").show(ctx, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(format!("Day {} — {:02}:{:02}",
                                        sim_day,
                                        sim_tod as u32,
                                        ((sim_tod % 1.0) * 60.0) as u32,
                                    ));
                                    ui.separator();
                                    ui.label(format!("Pop: {}", sim_pop));
                                    ui.separator();
                                    ui.label(format!("${:.0}", sim_funds));
                                    ui.separator();
                                    ui.label(format!("{} | {}", time_label, weather_label));
                                    ui.separator();
                                    if ui.button("⏸").clicked() { sim_speed = simulation::GameSpeed::Paused; }
                                    if ui.button("▶").clicked() { sim_speed = simulation::GameSpeed::Normal; }
                                    if ui.button("▶▶").clicked() { sim_speed = simulation::GameSpeed::Fast; }
                                    if ui.button("▶▶▶").clicked() { sim_speed = simulation::GameSpeed::VeryFast; }
                                });
                            });
                            plop_ui.show(ctx, &asset_names, Some(&sim_info));
                        });

                        egui_state
                            .handle_platform_output(window, full_output.platform_output);

                        let tris = egui_ctx
                            .tessellate(full_output.shapes, egui_ctx.pixels_per_point());
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
                                        view: &view_tex,
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

                        output.present();
                    }
                    Some(RenderMode::Software { backend, rasteriser }) => {
                        puffin::profile_scope!("render");
                        let fb = rasteriser.render(
                            &splats_to_render,
                            &camera,
                            &illuminant,
                        );
                        backend.present_framebuffer(&fb.pixels, fb.width, fb.height);
                    }
                    Some(RenderMode::CpuOnly { rasteriser }) => {
                        puffin::profile_scope!("render");
                        let _fb = rasteriser.render(
                            &splats_to_render,
                            &camera,
                            &illuminant,
                        );
                        // No surface to present to — frame computed but discarded.
                    }
                    None => {}
                }

                // --- Apply game speed changes from egui ---
                self.world.resource_mut::<SimulationState>().game_speed = sim_speed;

                // --- FPS counter with frame time ---
                self.frame_count += 1;
                let elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if elapsed >= 2.0 {
                    let fps = self.frame_count as f32 / elapsed;
                    let frame_ms = elapsed * 1000.0 / self.frame_count as f32;
                    println!("FPS: {fps:.1}  frame: {frame_ms:.2}ms");
                    self.frame_count = 0;
                    self.fps_timer = now;
                }

                // --- Request next frame ---
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    puffin::set_scopes_on(true);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop failed");
}
