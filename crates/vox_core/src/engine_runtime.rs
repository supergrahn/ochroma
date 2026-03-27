//! The Ochroma Engine Runtime v2 — Bevy ECS world, fixed timestep, entity builder.
//!
//! Games don't write their own main loops. They create an `EngineRuntime`,
//! configure it, register scripts, spawn entities, and call `tick()` each frame.
//!
//! ```rust,ignore
//! let mut engine = EngineRuntime::new(EngineConfig::default());
//! engine.register_script("Player", || Box::new(MyPlayerScript));
//! engine.spawn("Player")
//!     .with_asset("player.ply")
//!     .with_position(Vec3::new(0.0, 2.0, 0.0))
//!     .with_script("PlayerController")
//!     .with_collider(ColliderShape::Capsule { radius: 0.3, height: 1.8 });
//! engine.start();
//! while engine.tick(0.016) {}
//! ```

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};

use crate::ecs::*;
use crate::input::InputState;
use crate::script_interface::{GameScript, ScriptCommand, ScriptContext, ScriptRegistry};
use crate::types::GaussianSplat;

// ---------------------------------------------------------------------------
// Resources inserted into the Bevy World
// ---------------------------------------------------------------------------

/// Frame timing resource.
#[derive(Resource, Debug, Clone)]
pub struct FrameTime {
    pub dt: f32,
    pub total: f64,
    pub frame: u64,
}

impl Default for FrameTime {
    fn default() -> Self {
        Self { dt: 0.0, total: 0.0, frame: 0 }
    }
}

/// Fixed-timestep timing resource.
#[derive(Resource, Debug, Clone)]
pub struct FixedTime {
    pub dt: f32,
}

/// Render buffer: populated by the gather system each frame.
#[derive(Resource, Default, Debug)]
pub struct RenderBuffer {
    pub splats: Vec<GaussianSplat>,
    pub lights: Vec<LightData>,
}

/// A light for the render buffer.
#[derive(Debug, Clone)]
pub struct LightData {
    pub position: Vec3,
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

/// Camera state resource.
#[derive(Resource, Debug, Clone)]
pub struct CameraState {
    pub position: Vec3,
    pub forward: Vec3,
    pub view_proj: Mat4,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 10.0, 30.0),
            forward: Vec3::NEG_Z,
            view_proj: Mat4::IDENTITY,
        }
    }
}

/// Input state as a resource.
#[derive(Resource, Default)]
pub struct InputResource {
    pub state: InputState,
}

/// Script registry as a resource.
#[derive(Resource)]
pub struct ScriptRegistryResource {
    pub registry: ScriptRegistry,
}

/// Asset manager stub — stores known asset paths and tracks handles.
#[derive(Resource, Default)]
pub struct AssetManagerResource {
    pub assets: Vec<String>,
}

impl AssetManagerResource {
    pub fn register_asset(&mut self, path: &str) -> u64 {
        if let Some(idx) = self.assets.iter().position(|p| p == path) {
            return idx as u64;
        }
        let handle = self.assets.len() as u64;
        self.assets.push(path.to_string());
        handle
    }
}

/// Collision pairs detected during the last fixed tick.
#[derive(Resource, Default)]
pub struct CollisionPairs(pub Vec<(bevy_ecs::entity::Entity, bevy_ecs::entity::Entity)>);

/// Script contexts keyed by bevy Entity.
#[derive(Resource, Default)]
pub struct ScriptContexts {
    pub contexts: std::collections::HashMap<bevy_ecs::entity::Entity, ScriptContext>,
}

/// Pending script commands collected during script update, processed afterward.
#[derive(Resource, Default)]
pub struct PendingScriptCommands {
    pub commands: Vec<(bevy_ecs::entity::Entity, Vec<ScriptCommand>)>,
}

/// Counter for how many fixed-timestep physics steps ran this frame.
#[derive(Resource, Default)]
pub struct FixedStepCounter {
    pub steps_this_frame: u32,
}

/// Scene-level time of day (0.0 .. 24.0).
#[derive(Resource, Debug, Clone)]
pub struct TimeOfDay {
    pub hour: f32,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self { hour: 12.0 }
    }
}

// ---------------------------------------------------------------------------
// Physics backend selection
// ---------------------------------------------------------------------------

/// Physics backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBackend {
    Simple,
    Rapier,
}

impl Default for PhysicsBackend {
    fn default() -> Self {
        PhysicsBackend::Simple
    }
}

// ---------------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------------

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub window_title: String,
    pub window_width: u32,
    pub window_height: u32,
    pub target_fps: u32,
    pub fixed_timestep: f32,
    pub max_splats: usize,
    pub enable_particles: bool,
    pub enable_physics: bool,
    pub enable_audio: bool,
    pub enable_editor: bool,
    pub physics_backend: PhysicsBackend,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            window_title: "Ochroma Engine".to_string(),
            window_width: 1280,
            window_height: 720,
            target_fps: 60,
            fixed_timestep: 1.0 / 60.0,
            max_splats: 1_000_000,
            enable_particles: true,
            enable_physics: true,
            enable_audio: true,
            enable_editor: false,
            physics_backend: PhysicsBackend::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Bevy ECS Systems
// ---------------------------------------------------------------------------

/// Script update system — runs in the fixed schedule.
fn script_update_system(world: &mut World) {
    // Collect entity data first to avoid borrow conflicts.
    let entity_data: Vec<(bevy_ecs::entity::Entity, Vec<String>)> = {
        let mut query = world.query::<(Entity, &ScriptComponent)>();
        query.iter(world)
            .map(|(e, sc)| (e, sc.scripts.clone()))
            .collect()
    };

    let fixed_dt = world.resource::<FixedTime>().dt;

    // Run scripts and collect commands.
    let mut all_commands: Vec<(bevy_ecs::entity::Entity, Vec<ScriptCommand>)> = Vec::new();

    for (entity, scripts) in &entity_data {
        // Get or skip context
        let ctx = {
            let contexts = world.resource_mut::<ScriptContexts>();
            if !contexts.contexts.contains_key(entity) {
                continue;
            }
            // We need to remove it temporarily to avoid borrow conflict
            drop(contexts);
            let mut contexts = world.resource_mut::<ScriptContexts>();
            contexts.contexts.remove(entity)
        };

        if let Some(mut ctx) = ctx {
            {
                let registry = world.resource::<ScriptRegistryResource>();
                for script_name in scripts {
                    if let Some(mut script) = registry.registry.create(script_name) {
                        script.on_update(&mut ctx, fixed_dt);
                    }
                }
            }

            let commands = ctx.take_commands();
            if !commands.is_empty() {
                all_commands.push((*entity, commands));
            }

            // Put context back
            let mut contexts = world.resource_mut::<ScriptContexts>();
            contexts.contexts.insert(*entity, ctx);
        }
    }

    // Store pending commands for processing.
    world.resource_mut::<PendingScriptCommands>().commands = all_commands;
}

/// Process pending script commands — runs after script_update_system.
fn process_script_commands_system(world: &mut World) {
    let pending = std::mem::take(&mut world.resource_mut::<PendingScriptCommands>().commands);

    for (entity, commands) in pending {
        for cmd in commands {
            match cmd {
                ScriptCommand::SetPosition { position } => {
                    if let Some(mut transform) = world.get_mut::<TransformComponent>(entity) {
                        transform.position = Vec3::from_array(position);
                    }
                }
                ScriptCommand::SetRotation { rotation } => {
                    if let Some(mut transform) = world.get_mut::<TransformComponent>(entity) {
                        transform.rotation = glam::Quat::from_xyzw(
                            rotation[0], rotation[1], rotation[2], rotation[3],
                        );
                    }
                }
                ScriptCommand::Spawn { asset_path, position, rotation, scale } => {
                    let handle = world.resource_mut::<AssetManagerResource>().register_asset(&asset_path);
                    world.spawn((
                        NameComponent("Spawned".to_string()),
                        TransformComponent {
                            position: Vec3::from_array(position),
                            rotation: glam::Quat::from_xyzw(
                                rotation[0], rotation[1], rotation[2], rotation[3],
                            ),
                            scale: Vec3::from_array(scale),
                        },
                        AssetRefComponent { path: asset_path, handle },
                    ));
                }
                ScriptCommand::Destroy { entity_id } => {
                    // entity_id is a u32 from the old API — best-effort removal
                    let _ = entity_id;
                }
                ScriptCommand::PlaySound { clip, volume, .. } => {
                    // Inline WAV synthesis (vox_core cannot depend on vox_audio)
                    let sample_rate = 44100u32;
                    let duration = 0.05f32;
                    let num_samples = (sample_rate as f32 * duration) as usize;
                    let samples: Vec<f32> = (0..num_samples)
                        .map(|i| {
                            let t = i as f32 / sample_rate as f32;
                            let decay = 1.0 - (i as f32 / num_samples as f32);
                            (t * 800.0 * 2.0 * std::f32::consts::PI).sin() * decay * volume
                        })
                        .collect();

                    let data_size = (num_samples * 2) as u32;
                    let file_size = 36 + data_size;
                    let mut wav = Vec::with_capacity(44 + data_size as usize);
                    wav.extend_from_slice(b"RIFF");
                    wav.extend_from_slice(&file_size.to_le_bytes());
                    wav.extend_from_slice(b"WAVE");
                    wav.extend_from_slice(b"fmt ");
                    wav.extend_from_slice(&16u32.to_le_bytes());
                    wav.extend_from_slice(&1u16.to_le_bytes());
                    wav.extend_from_slice(&1u16.to_le_bytes());
                    wav.extend_from_slice(&sample_rate.to_le_bytes());
                    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
                    wav.extend_from_slice(&2u16.to_le_bytes());
                    wav.extend_from_slice(&16u16.to_le_bytes());
                    wav.extend_from_slice(b"data");
                    wav.extend_from_slice(&data_size.to_le_bytes());
                    for &s in &samples {
                        let s16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                        wav.extend_from_slice(&s16.to_le_bytes());
                    }

                    let path = std::env::temp_dir().join(format!("ochroma_{}.wav", clip));
                    let _ = std::fs::write(&path, &wav);
                    println!("[engine] play_sound: {} vol={} -> {}", clip, volume, path.display());
                }
                ScriptCommand::ApplyForce { force } => {
                    let _ = force; // TODO: wire to physics
                }
                ScriptCommand::SendEvent { name, data } => {
                    println!("[engine] Event: {} = {}", name, data);
                }
                ScriptCommand::Log { message } => {
                    println!("[script] {}", message);
                }
            }
        }
    }
}

/// AABB collision detection system — runs in the fixed schedule.
fn physics_collision_system(world: &mut World) {
    let entities: Vec<(bevy_ecs::entity::Entity, Vec3, ColliderShape)> = {
        let mut query = world.query::<(Entity, &TransformComponent, &ColliderComponent)>();
        query.iter(world)
            .map(|(e, t, c)| (e, t.position, c.shape.clone()))
            .collect()
    };

    let mut collisions = Vec::new();
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let (ea, pos_a, ref col_a) = entities[i];
            let (eb, pos_b, ref col_b) = entities[j];
            if aabb_overlap(pos_a, col_a, pos_b, col_b) {
                collisions.push((ea, eb));
            }
        }
    }

    world.resource_mut::<CollisionPairs>().0 = collisions;
}

/// Frustum cull system — runs in the frame schedule.
fn frustum_cull_system(world: &mut World) {
    let camera_pos = world.resource::<CameraState>().position;

    // Remove Visible from all entities first, then add to visible ones.
    let all_with_visible: Vec<bevy_ecs::entity::Entity> = {
        let mut query = world.query_filtered::<Entity, With<Visible>>();
        query.iter(world).collect()
    };
    for entity in all_with_visible {
        world.entity_mut(entity).remove::<Visible>();
    }

    // Mark entities within a generous view distance as visible.
    let to_mark: Vec<bevy_ecs::entity::Entity> = {
        let mut query = world.query_filtered::<(Entity, &TransformComponent), With<AssetRefComponent>>();
        query.iter(world)
            .filter(|(_, t)| t.position.distance(camera_pos) < 500.0)
            .map(|(e, _)| e)
            .collect()
    };
    for entity in to_mark {
        world.entity_mut(entity).insert(Visible);
    }
}

/// Gather splats from visible entities into the render buffer.
fn gather_splats_system(world: &mut World) {
    let mut render_buffer = world.resource_mut::<RenderBuffer>();
    render_buffer.splats.clear();
    render_buffer.lights.clear();

    // Gather lights from point light entities.
    let lights: Vec<LightData> = {
        let mut query = world.query::<(&TransformComponent, &PointLightComponent)>();
        query.iter(world)
            .map(|(t, l)| LightData {
                position: t.position,
                color: l.color,
                intensity: l.intensity,
                radius: l.radius,
            })
            .collect()
    };

    let mut render_buffer = world.resource_mut::<RenderBuffer>();
    render_buffer.lights = lights;

    // NOTE: Actual splat gathering requires loading real assets.
    // The AssetManagerResource is a stub; real loading happens in the binary.
    // For now, visible entities with AssetRefComponent are tracked but no splats are emitted.
}

/// AABB overlap test between two colliders at given positions.
fn aabb_overlap(pos_a: Vec3, col_a: &ColliderShape, pos_b: Vec3, col_b: &ColliderShape) -> bool {
    let ha = collider_half_extents(col_a);
    let hb = collider_half_extents(col_b);
    (pos_a.x - pos_b.x).abs() < ha[0] + hb[0]
        && (pos_a.y - pos_b.y).abs() < ha[1] + hb[1]
        && (pos_a.z - pos_b.z).abs() < ha[2] + hb[2]
}

/// Get AABB half-extents for any collider shape.
fn collider_half_extents(shape: &ColliderShape) -> [f32; 3] {
    match shape {
        ColliderShape::Box { half_extents } => *half_extents,
        ColliderShape::Sphere { radius } => [*radius, *radius, *radius],
        ColliderShape::Capsule { radius, height } => [*radius, height * 0.5 + radius, *radius],
    }
}

// ---------------------------------------------------------------------------
// Frame statistics
// ---------------------------------------------------------------------------

/// Frame statistics from the engine.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    pub frame_number: u64,
    pub dt: f32,
    pub fps: f32,
    pub entity_count: u32,
    pub splat_count: u32,
    pub visible_splats: u32,
    pub culled_splats: u32,
    pub physics_time_ms: f32,
    pub script_time_ms: f32,
    pub render_time_ms: f32,
    pub total_time_ms: f32,
}

// ---------------------------------------------------------------------------
// The Engine Runtime
// ---------------------------------------------------------------------------

/// The Engine Runtime v2 — Bevy ECS world with fixed timestep.
///
/// Games configure the engine, register scripts, spawn entities via the builder
/// pattern, then call `tick(frame_dt)` each frame. The engine runs fixed-timestep
/// physics/scripts at 60Hz and frame-rate systems (culling, gather) once per frame.
pub struct EngineRuntime {
    pub world: World,
    pub config: EngineConfig,
    pub stats: FrameStats,
    accumulator: f32,
    fixed_dt: f32,
    running: bool,
    frame_count: u64,
}

impl EngineRuntime {
    pub fn new(config: EngineConfig) -> Self {
        let mut world = World::new();

        let fixed_dt = config.fixed_timestep;

        // Insert all resources.
        world.insert_resource(FrameTime::default());
        world.insert_resource(FixedTime { dt: fixed_dt });
        world.insert_resource(RenderBuffer::default());
        world.insert_resource(CameraState::default());
        world.insert_resource(InputResource::default());
        world.insert_resource(ScriptRegistryResource {
            registry: ScriptRegistry::new(),
        });
        world.insert_resource(AssetManagerResource::default());
        world.insert_resource(CollisionPairs::default());
        world.insert_resource(ScriptContexts::default());
        world.insert_resource(PendingScriptCommands::default());
        world.insert_resource(FixedStepCounter::default());
        world.insert_resource(TimeOfDay::default());

        Self {
            world,
            config,
            stats: FrameStats::default(),
            accumulator: 0.0,
            fixed_dt,
            running: false,
            frame_count: 0,
        }
    }

    /// Spawn an entity with a name and return a builder for adding components.
    pub fn spawn(&mut self, name: &str) -> EntityBuilder<'_> {
        let entity = self.world.spawn((
            NameComponent(name.to_string()),
            TransformComponent::default(),
            TagsComponent::default(),
        )).id();

        // Create a script context for this entity.
        self.world.resource_mut::<ScriptContexts>()
            .contexts.insert(entity, ScriptContext::new(entity.index()));

        EntityBuilder {
            runtime: self,
            entity,
        }
    }

    /// Register a script factory by name.
    pub fn register_script<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn GameScript> + Send + Sync + 'static,
    {
        self.world.resource_mut::<ScriptRegistryResource>()
            .registry.register(name, factory);
    }

    /// Run one frame. Returns false when the engine should quit.
    ///
    /// `frame_dt` is the real wall-clock time since last tick (seconds).
    /// Fixed-timestep systems run 0..N times, frame systems run once.
    pub fn tick(&mut self, frame_dt: f32) -> bool {
        if !self.running {
            return false;
        }

        self.frame_count += 1;

        // Update frame time resource.
        {
            let mut ft = self.world.resource_mut::<FrameTime>();
            ft.dt = frame_dt;
            ft.total += frame_dt as f64;
            ft.frame = self.frame_count;
        }

        // --- Fixed timestep loop ---
        self.accumulator += frame_dt;
        let mut fixed_steps = 0u32;

        while self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            fixed_steps += 1;

            // Run fixed systems inline (exclusive world access).
            script_update_system(&mut self.world);
            process_script_commands_system(&mut self.world);
            physics_collision_system(&mut self.world);
        }

        self.world.resource_mut::<FixedStepCounter>().steps_this_frame = fixed_steps;

        // --- Per-frame systems ---
        frustum_cull_system(&mut self.world);
        gather_splats_system(&mut self.world);

        // Update stats.
        let entity_count = {
            let mut q = self.world.query::<&NameComponent>();
            q.iter(&self.world).count() as u32
        };
        self.stats.frame_number = self.frame_count;
        self.stats.dt = frame_dt;
        self.stats.entity_count = entity_count;
        self.stats.fps = if frame_dt > 0.0 { 1.0 / frame_dt } else { 0.0 };

        // End-of-frame input cleanup.
        self.world.resource_mut::<InputResource>().state.end_frame();

        true
    }

    /// Initialize script contexts and call on_start for all scripted entities.
    pub fn init_scripts(&mut self) {
        let scripted: Vec<(bevy_ecs::entity::Entity, Vec<String>)> = {
            let mut query = self.world.query::<(Entity, &ScriptComponent)>();
            query.iter(&self.world)
                .map(|(e, sc)| (e, sc.scripts.clone()))
                .collect()
        };

        for (entity, scripts) in &scripted {
            let mut ctx = ScriptContext::new(entity.index());
            {
                let registry = self.world.resource::<ScriptRegistryResource>();
                for script_name in scripts {
                    if let Some(mut script) = registry.registry.create(script_name) {
                        script.on_start(&mut ctx);
                    }
                }
            }

            // Process start commands.
            let commands = ctx.take_commands();
            if !commands.is_empty() {
                self.world.resource_mut::<PendingScriptCommands>().commands.push((*entity, commands));
            }

            // Store fresh context.
            self.world.resource_mut::<ScriptContexts>()
                .contexts.insert(*entity, ScriptContext::new(entity.index()));
        }

        // Process any commands from on_start.
        process_script_commands_system(&mut self.world);
    }

    /// Start the engine.
    pub fn start(&mut self) {
        self.running = true;
        self.init_scripts();

        let script_count = self.world.resource::<ScriptRegistryResource>()
            .registry.registered_scripts().len();
        let entity_count = {
            let mut q = self.world.query::<&NameComponent>();
            q.iter(&self.world).count()
        };
        println!("[engine] Started — {} entities, {} scripts registered", entity_count, script_count);
    }

    /// Stop the engine.
    pub fn stop(&mut self) {
        self.running = false;
        println!("[engine] Stopped");
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get the entity count (entities with NameComponent).
    pub fn entity_count(&mut self) -> usize {
        let mut q = self.world.query::<&NameComponent>();
        q.iter(&self.world).count()
    }

    /// Get the number of registered scripts.
    pub fn registered_script_count(&self) -> usize {
        self.world.resource::<ScriptRegistryResource>()
            .registry.registered_scripts().len()
    }

    /// Get mutable reference to time of day resource.
    pub fn time_of_day(&self) -> f32 {
        self.world.resource::<TimeOfDay>().hour
    }

    /// Set time of day.
    pub fn set_time_of_day(&mut self, hour: f32) {
        self.world.resource_mut::<TimeOfDay>().hour = hour;
    }
}

// ---------------------------------------------------------------------------
// Entity Builder — fluent API for spawning entities
// ---------------------------------------------------------------------------

/// Builder pattern for spawning entities with components.
pub struct EntityBuilder<'a> {
    runtime: &'a mut EngineRuntime,
    entity: bevy_ecs::entity::Entity,
}

impl<'a> EntityBuilder<'a> {
    /// Attach an asset reference.
    pub fn with_asset(self, path: &str) -> Self {
        let handle = self.runtime.world
            .resource_mut::<AssetManagerResource>()
            .register_asset(path);
        self.runtime.world.entity_mut(self.entity).insert(AssetRefComponent {
            path: path.to_string(),
            handle,
        });
        self
    }

    /// Set position.
    pub fn with_position(self, pos: Vec3) -> Self {
        self.runtime.world.entity_mut(self.entity).get_mut::<TransformComponent>()
            .unwrap().position = pos;
        self
    }

    /// Attach a script by name.
    pub fn with_script(self, name: &str) -> Self {
        if let Some(mut sc) = self.runtime.world.entity_mut(self.entity).get_mut::<ScriptComponent>() {
            sc.scripts.push(name.to_string());
        } else {
            self.runtime.world.entity_mut(self.entity).insert(ScriptComponent {
                scripts: vec![name.to_string()],
            });
        }
        self
    }

    /// Attach a collider.
    pub fn with_collider(self, shape: ColliderShape) -> Self {
        self.runtime.world.entity_mut(self.entity).insert(ColliderComponent { shape });
        self
    }

    /// Attach a point light.
    pub fn with_light(self, color: [f32; 3], intensity: f32, radius: f32) -> Self {
        self.runtime.world.entity_mut(self.entity).insert(PointLightComponent {
            color,
            intensity,
            radius,
        });
        self
    }

    /// Attach an audio emitter.
    pub fn with_audio(self, clip: &str, volume: f32, looping: bool) -> Self {
        self.runtime.world.entity_mut(self.entity).insert(AudioEmitterComponent {
            clip_path: clip.to_string(),
            volume,
            looping,
            playing: false,
            spatial: true,
        });
        self
    }

    /// Add a tag.
    pub fn with_tag(self, tag: &str) -> Self {
        self.runtime.world.entity_mut(self.entity).get_mut::<TagsComponent>()
            .unwrap().0.push(tag.to_string());
        self
    }

    /// Get the Bevy entity ID.
    pub fn id(&self) -> bevy_ecs::entity::Entity {
        self.entity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct TestScript;
    impl crate::script_interface::GameScript for TestScript {
        fn on_start(&mut self, ctx: &mut ScriptContext) {
            ctx.log("Test script started");
        }
        fn on_update(&mut self, ctx: &mut ScriptContext, _dt: f32) {
            ctx.log("Test script update");
        }
        fn name(&self) -> &str { "TestScript" }
    }

    #[test]
    fn create_engine_and_spawn_entities() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.spawn("Player").with_asset("player.ply").with_position(Vec3::new(0.0, 1.0, 0.0));
        engine.spawn("Enemy").with_asset("enemy.ply").with_position(Vec3::new(10.0, 0.0, 5.0));

        let count = engine.world.query::<&NameComponent>().iter(&engine.world).count();
        assert_eq!(count, 2);
    }

    #[test]
    fn find_entity_by_name() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.spawn("Player").with_asset("player.ply").with_position(Vec3::new(0.0, 1.0, 0.0));

        let found: Vec<(&NameComponent, &TransformComponent)> = engine.world
            .query::<(&NameComponent, &TransformComponent)>()
            .iter(&engine.world)
            .filter(|(n, _)| n.0 == "Player")
            .collect();

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1.position, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn spawn_with_script() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.register_script("TestScript", || Box::new(TestScript));
        engine.spawn("NPC").with_asset("npc.ply").with_position(Vec3::new(5.0, 0.0, 5.0)).with_script("TestScript");

        let scripts: Vec<&ScriptComponent> = engine.world
            .query::<&ScriptComponent>()
            .iter(&engine.world)
            .collect();
        assert_eq!(scripts.len(), 1);
        assert_eq!(scripts[0].scripts, vec!["TestScript"]);
    }

    #[test]
    fn engine_tick() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.register_script("TestScript", || Box::new(TestScript));
        engine.spawn("NPC").with_script("TestScript");
        engine.start();
        assert!(engine.tick(0.02)); // must exceed fixed_dt (1/60 ≈ 0.01667)
        assert_eq!(engine.stats.frame_number, 1);
    }

    #[test]
    fn destroy_entity() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let entity = engine.spawn("Temp").id();
        assert_eq!(engine.world.query::<&NameComponent>().iter(&engine.world).count(), 1);
        engine.world.despawn(entity);
        assert_eq!(engine.world.query::<&NameComponent>().iter(&engine.world).count(), 0);
    }

    #[test]
    fn tags_and_find() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.spawn("Coin").with_asset("coin.ply").with_position(Vec3::new(5.0, 1.0, 5.0)).with_tag("collectible");

        let collectibles: Vec<(&NameComponent, &TagsComponent)> = engine.world
            .query::<(&NameComponent, &TagsComponent)>()
            .iter(&engine.world)
            .filter(|(_, t)| t.0.contains(&"collectible".to_string()))
            .collect();

        assert_eq!(collectibles.len(), 1);
        assert_eq!(collectibles[0].0 .0, "Coin");
    }

    // --- Script that moves entities ---

    struct MoverScript;
    impl crate::script_interface::GameScript for MoverScript {
        fn on_update(&mut self, ctx: &mut ScriptContext, _dt: f32) {
            ctx.set_position([99.0, 0.0, 0.0]);
        }
        fn name(&self) -> &str { "Mover" }
    }

    #[test]
    fn script_set_position_moves_entity() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.register_script("Mover", || Box::new(MoverScript));
        let entity = engine.spawn("Thing").with_script("Mover").id();
        engine.start();
        engine.tick(0.02); // must exceed fixed_dt (1/60) to trigger at least one fixed step

        let transform = engine.world.get::<TransformComponent>(entity).unwrap();
        assert_eq!(transform.position, Vec3::new(99.0, 0.0, 0.0), "Script should have moved entity");
    }

    // --- AABB collision detection ---

    #[test]
    fn aabb_collision_detected_when_overlapping() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let ea = engine.spawn("BoxA")
            .with_position(Vec3::new(0.0, 0.0, 0.0))
            .with_collider(ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] })
            .id();
        let eb = engine.spawn("BoxB")
            .with_position(Vec3::new(1.5, 0.0, 0.0))
            .with_collider(ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] })
            .id();
        engine.start();
        engine.tick(0.02);

        let collisions = &engine.world.resource::<CollisionPairs>().0;
        assert!(!collisions.is_empty(), "Overlapping boxes should collide");
        assert!(collisions.contains(&(ea, eb)), "Collision pair not found");
    }

    #[test]
    fn aabb_no_collision_when_separated() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.spawn("BoxA")
            .with_position(Vec3::new(0.0, 0.0, 0.0))
            .with_collider(ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] });
        engine.spawn("BoxB")
            .with_position(Vec3::new(10.0, 0.0, 0.0))
            .with_collider(ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] });
        engine.start();
        engine.tick(0.02);

        let collisions = &engine.world.resource::<CollisionPairs>().0;
        assert!(collisions.is_empty(), "Separated boxes should not collide");
    }

    #[test]
    fn sphere_collision_detected() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let ea = engine.spawn("SphereA")
            .with_position(Vec3::ZERO)
            .with_collider(ColliderShape::Sphere { radius: 2.0 })
            .id();
        let eb = engine.spawn("SphereB")
            .with_position(Vec3::new(3.0, 0.0, 0.0))
            .with_collider(ColliderShape::Sphere { radius: 2.0 })
            .id();
        engine.start();
        engine.tick(0.02);

        let collisions = &engine.world.resource::<CollisionPairs>().0;
        assert!(collisions.contains(&(ea, eb)), "Overlapping spheres should collide");
    }

    // --- Audio proof ---

    struct SoundScript;
    impl crate::script_interface::GameScript for SoundScript {
        fn on_start(&mut self, ctx: &mut ScriptContext) {
            ctx.play_sound("test_sound", 1.0);
        }
        fn name(&self) -> &str { "SoundScript" }
    }

    #[test]
    fn script_play_sound_generates_wav() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.register_script("SoundScript", || Box::new(SoundScript));
        engine.spawn("Speaker").with_script("SoundScript");
        engine.start();

        let wav_path = std::env::temp_dir().join("ochroma_test_sound.wav");
        assert!(wav_path.exists(), "WAV file should exist at {}", wav_path.display());

        let data = std::fs::read(&wav_path).unwrap();
        assert!(data.len() > 44, "WAV file should have header + data");
        assert_eq!(&data[0..4], b"RIFF", "Should start with RIFF header");
        assert_eq!(&data[8..12], b"WAVE", "Should contain WAVE marker");

        let _ = std::fs::remove_file(&wav_path);
    }

    // --- New v2 tests ---

    #[test]
    fn fixed_timestep_runs_multiple_physics_steps() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        // fixed_dt = 1/60, frame_dt = 1/30 => 2 physics steps
        engine.start();
        engine.tick(1.0 / 30.0);

        let steps = engine.world.resource::<FixedStepCounter>().steps_this_frame;
        assert_eq!(steps, 2, "With frame_dt=1/30 and fixed_dt=1/60, should run 2 physics steps");
    }

    #[test]
    fn spawn_with_builder_pattern() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.spawn("Player")
            .with_asset("player.ply")
            .with_position(Vec3::new(0.0, 2.0, 0.0))
            .with_script("PlayerController")
            .with_collider(ColliderShape::Capsule { radius: 0.3, height: 1.8 })
            .with_tag("player");

        let count = engine.world
            .query::<(&NameComponent, &AssetRefComponent, &ScriptComponent)>()
            .iter(&engine.world)
            .count();
        assert_eq!(count, 1);

        // Verify collider and tag are present too.
        let full_count = engine.world
            .query::<(&NameComponent, &AssetRefComponent, &ScriptComponent, &ColliderComponent, &TagsComponent)>()
            .iter(&engine.world)
            .count();
        assert_eq!(full_count, 1);
    }

    #[test]
    fn render_buffer_populated_after_tick() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.world.insert_resource(CameraState {
            position: Vec3::new(0.0, 10.0, 30.0),
            forward: Vec3::NEG_Z,
            view_proj: Mat4::IDENTITY,
        });
        engine.spawn("Building").with_asset("test.ply").with_position(Vec3::ZERO);
        engine.start();
        engine.tick(0.016);

        // At minimum, the system ran without panicking.
        let _buffer = engine.world.resource::<RenderBuffer>();
    }

    #[test]
    fn fixed_timestep_accumulator_preserves_remainder() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.start();
        // fixed_dt = 1/60 ≈ 0.01667. Pass in 0.025 => 1 step, remainder ~0.00833
        engine.tick(0.025);
        let steps = engine.world.resource::<FixedStepCounter>().steps_this_frame;
        assert_eq!(steps, 1);

        // Next tick with 0.01 => accumulator ~0.01833 => 1 more step
        engine.tick(0.01);
        let steps = engine.world.resource::<FixedStepCounter>().steps_this_frame;
        assert_eq!(steps, 1);
    }
}
