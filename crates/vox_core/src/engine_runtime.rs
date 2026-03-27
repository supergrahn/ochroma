//! The Ochroma Engine Runtime — the central orchestrator that games use.
//!
//! Games don't write their own main loops. They create an `EngineRuntime`,
//! configure it, register scripts, load a scene, and call `run()`.
//!
//! ```rust,ignore
//! let mut engine = EngineRuntime::new(EngineConfig::default());
//! engine.scripts.register("Player", || Box::new(MyPlayerScript));
//! engine.load_scene("maps/level1.ochroma_map");
//! engine.run(); // blocks until quit
//! ```

use crate::game_loop::{GameClock, GamePhase};
use crate::input::{InputState, KeyBindings, InputSource, GameAction};
use crate::script_interface::{ScriptRegistry, ScriptContext, ScriptCommand};
use crate::undo::{UndoStack, UndoEntry};
use crate::error::EngineError;

/// Physics backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBackend {
    /// Built-in AABB collision detection.
    Simple,
    /// Full Rapier3D physics (when `rapier` feature is enabled in vox_physics).
    Rapier,
}

impl Default for PhysicsBackend {
    fn default() -> Self {
        PhysicsBackend::Simple
    }
}

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

/// A scene entity managed by the engine.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: u32,
    pub name: String,
    pub active: bool,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub asset_path: Option<String>,
    pub scripts: Vec<String>,
    pub collider: Option<ColliderShape>,
    pub tags: Vec<String>,
}

/// Collider shapes for physics.
#[derive(Debug, Clone)]
pub enum ColliderShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { radius: f32, height: f32 },
}

/// Light in the scene.
#[derive(Debug, Clone)]
pub struct SceneLight {
    pub light_type: LightType,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

/// The engine's scene state.
pub struct Scene {
    pub name: String,
    pub entities: Vec<Entity>,
    pub lights: Vec<SceneLight>,
    pub ambient_light: [f32; 3],
    pub gravity: f32,
    pub time_of_day: f32,
    next_id: u32,
}

impl Scene {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            entities: Vec::new(),
            lights: vec![SceneLight {
                light_type: LightType::Directional,
                position: [0.0, 100.0, 0.0],
                direction: [0.3, -1.0, 0.2],
                color: [1.0, 0.95, 0.9],
                intensity: 1.0,
                radius: 0.0,
            }],
            ambient_light: [0.1, 0.1, 0.12],
            gravity: 9.81,
            time_of_day: 12.0,
            next_id: 0,
        }
    }

    /// Spawn an entity. Returns its ID.
    pub fn spawn(&mut self, name: &str, asset_path: Option<&str>, position: [f32; 3]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(Entity {
            id,
            name: name.to_string(),
            active: true,
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            asset_path: asset_path.map(|s| s.to_string()),
            scripts: Vec::new(),
            collider: None,
            tags: Vec::new(),
        });
        id
    }

    /// Spawn with a script attached.
    pub fn spawn_with_script(&mut self, name: &str, asset: Option<&str>, pos: [f32; 3], script: &str) -> u32 {
        let id = self.spawn(name, asset, pos);
        self.entities.last_mut().unwrap().scripts.push(script.to_string());
        id
    }

    /// Spawn with a collider.
    pub fn spawn_with_collider(&mut self, name: &str, asset: Option<&str>, pos: [f32; 3], collider: ColliderShape) -> u32 {
        let id = self.spawn(name, asset, pos);
        self.entities.last_mut().unwrap().collider = Some(collider);
        id
    }

    /// Find entity by ID.
    pub fn get(&self, id: u32) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Find entity by ID (mutable).
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    /// Find entities by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.tags.contains(&tag.to_string())).collect()
    }

    /// Find entity by name.
    pub fn find_by_name(&self, name: &str) -> Option<&Entity> {
        self.entities.iter().find(|e| e.name == name)
    }

    /// Destroy an entity.
    pub fn destroy(&mut self, id: u32) {
        self.entities.retain(|e| e.id != id);
    }

    /// Entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.iter().filter(|e| e.active).count()
    }

    /// Add a point light.
    pub fn add_point_light(&mut self, position: [f32; 3], color: [f32; 3], intensity: f32, radius: f32) {
        self.lights.push(SceneLight {
            light_type: LightType::Point,
            position, direction: [0.0, -1.0, 0.0],
            color, intensity, radius,
        });
    }
}

/// AABB overlap test between two colliders at given positions.
fn aabb_overlap(pos_a: [f32; 3], col_a: &ColliderShape, pos_b: [f32; 3], col_b: &ColliderShape) -> bool {
    let ha = collider_half_extents(col_a);
    let hb = collider_half_extents(col_b);
    (pos_a[0] - pos_b[0]).abs() < ha[0] + hb[0]
        && (pos_a[1] - pos_b[1]).abs() < ha[1] + hb[1]
        && (pos_a[2] - pos_b[2]).abs() < ha[2] + hb[2]
}

/// Get AABB half-extents for any collider shape.
fn collider_half_extents(shape: &ColliderShape) -> [f32; 3] {
    match shape {
        ColliderShape::Box { half_extents } => *half_extents,
        ColliderShape::Sphere { radius } => [*radius, *radius, *radius],
        ColliderShape::Capsule { radius, height } => [*radius, height * 0.5 + radius, *radius],
    }
}

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

/// The Engine Runtime — orchestrates all systems.
///
/// This is what game developers interact with. They don't write their own
/// game loops. They configure the engine, register scripts, load scenes,
/// and the engine handles everything.
pub struct EngineRuntime {
    pub config: EngineConfig,
    pub scene: Scene,
    pub scripts: ScriptRegistry,
    pub input: InputState,
    pub bindings: KeyBindings,
    pub clock: GameClock,
    pub undo: UndoStack,
    pub stats: FrameStats,

    /// Script contexts for each scripted entity.
    script_contexts: std::collections::HashMap<u32, ScriptContext>,

    /// Collision pairs detected last tick.
    pub last_collisions: Vec<(u32, u32)>,

    /// Whether the engine is running.
    running: bool,
}

impl EngineRuntime {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            clock: GameClock::new(config.fixed_timestep),
            config,
            scene: Scene::new("Untitled"),
            scripts: ScriptRegistry::new(),
            input: InputState::default(),
            bindings: KeyBindings::default(),
            undo: UndoStack::new(100),
            stats: FrameStats::default(),
            script_contexts: std::collections::HashMap::new(),
            last_collisions: Vec::new(),
            running: false,
        }
    }

    /// Load a scene by name and populate it externally.
    /// Game code calls this then adds entities via scene.spawn().
    pub fn load_scene(&mut self, name: &str) {
        self.scene = Scene::new(name);
        println!("[engine] Created scene '{}'", name);
    }

    /// Convenience: add an entity with a script.
    pub fn add_entity(&mut self, name: &str, asset: Option<&str>, pos: [f32; 3], script: Option<&str>) -> u32 {
        let id = self.scene.spawn(name, asset, pos);
        if let Some(s) = script {
            self.scene.get_mut(id).unwrap().scripts.push(s.to_string());
        }
        id
    }

    /// Initialize script contexts for all scripted entities.
    pub fn init_scripts(&mut self) {
        self.script_contexts.clear();
        let scripted: Vec<(u32, Vec<String>)> = self.scene.entities.iter()
            .filter(|e| !e.scripts.is_empty())
            .map(|e| (e.id, e.scripts.clone()))
            .collect();

        for (id, scripts) in &scripted {
            let mut ctx = ScriptContext::new(*id);
            for script_name in scripts {
                if let Some(mut script) = self.scripts.create(script_name) {
                    script.on_start(&mut ctx);
                }
            }
            // Process start commands
            let commands = ctx.take_commands();
            self.process_commands(*id, commands);
            self.script_contexts.insert(*id, ScriptContext::new(*id));
        }
    }

    /// Run one frame of the engine. Call this from your game loop.
    /// Returns false when the engine should quit.
    pub fn tick(&mut self, dt: f32) -> bool {
        if !self.running { return false; }

        self.stats.frame_number += 1;
        self.stats.dt = dt;
        self.stats.entity_count = self.scene.entity_count() as u32;

        // Phase 1: Input (handled externally, input state already updated)

        // Phase 2: Scripts
        let script_start = std::time::Instant::now();
        let scripted: Vec<(u32, Vec<String>)> = self.scene.entities.iter()
            .filter(|e| !e.scripts.is_empty() && e.active)
            .map(|e| (e.id, e.scripts.clone()))
            .collect();

        for (id, scripts) in &scripted {
            if let Some(ctx) = self.script_contexts.get_mut(id) {
                for script_name in scripts {
                    if let Some(mut script) = self.scripts.create(script_name) {
                        script.on_update(ctx, dt);
                    }
                }
                let commands = ctx.take_commands();
                self.process_commands(*id, commands);
            }
        }
        self.stats.script_time_ms = script_start.elapsed().as_secs_f32() * 1000.0;

        // Phase 3: Simple AABB collision detection
        let entities_with_colliders: Vec<(u32, [f32; 3], ColliderShape)> = self.scene.entities.iter()
            .filter(|e| e.active && e.collider.is_some())
            .map(|e| (e.id, e.position, e.collider.clone().unwrap()))
            .collect();

        self.last_collisions.clear();
        for i in 0..entities_with_colliders.len() {
            for j in (i + 1)..entities_with_colliders.len() {
                let (id_a, pos_a, ref col_a) = entities_with_colliders[i];
                let (id_b, pos_b, ref col_b) = entities_with_colliders[j];
                if aabb_overlap(pos_a, col_a, pos_b, col_b) {
                    self.last_collisions.push((id_a, id_b));
                }
            }
        }

        // Notify scripts of collisions
        for (id_a, id_b) in &self.last_collisions.clone() {
            if let Some(ctx) = self.script_contexts.get_mut(id_a) {
                let scripts: Vec<String> = self.scene.get(*id_a)
                    .map(|e| e.scripts.clone()).unwrap_or_default();
                for script_name in &scripts {
                    if let Some(mut script) = self.scripts.create(script_name) {
                        script.on_collision(ctx, *id_b);
                    }
                }
            }
            if let Some(ctx) = self.script_contexts.get_mut(id_b) {
                let scripts: Vec<String> = self.scene.get(*id_b)
                    .map(|e| e.scripts.clone()).unwrap_or_default();
                for script_name in &scripts {
                    if let Some(mut script) = self.scripts.create(script_name) {
                        script.on_collision(ctx, *id_a);
                    }
                }
            }
        }

        self.stats.physics_time_ms = script_start.elapsed().as_secs_f32() * 1000.0 - self.stats.script_time_ms;

        // Phase 4: Advance time
        self.scene.time_of_day = (self.scene.time_of_day + dt * 0.01) % 24.0; // slow time advance

        self.input.end_frame();
        true
    }

    /// Process commands issued by scripts.
    fn process_commands(&mut self, entity_id: u32, commands: Vec<ScriptCommand>) {
        for cmd in commands {
            match cmd {
                ScriptCommand::Spawn { asset_path, position, rotation, scale } => {
                    let id = self.scene.spawn("Spawned", Some(&asset_path), position);
                    if let Some(e) = self.scene.get_mut(id) {
                        e.rotation = rotation;
                        e.scale = scale;
                    }
                }
                ScriptCommand::Destroy { entity_id: target } => {
                    self.scene.destroy(target);
                }
                ScriptCommand::SetPosition { position } => {
                    if let Some(entity) = self.scene.get_mut(entity_id) {
                        entity.position = position;
                    }
                }
                ScriptCommand::SetRotation { rotation } => {
                    if let Some(entity) = self.scene.get_mut(entity_id) {
                        entity.rotation = rotation;
                    }
                }
                ScriptCommand::ApplyForce { force } => {
                    // TODO: wire to physics when Rapier is enabled
                    let _ = force;
                }
                ScriptCommand::PlaySound { clip, volume, .. } => {
                    // Generate WAV proof — inline synthesis (vox_core cannot depend on vox_audio)
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

                    // Write minimal WAV
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
                ScriptCommand::SendEvent { name, data } => {
                    println!("[engine] Event: {} = {}", name, data);
                }
                ScriptCommand::Log { message } => {
                    println!("[script] {}", message);
                }
            }
        }
    }

    /// Start the engine.
    pub fn start(&mut self) {
        self.running = true;
        self.init_scripts();
        println!("[engine] Started — {} entities, {} scripts registered",
            self.scene.entity_count(), self.scripts.registered_scripts().len());
    }

    /// Stop the engine.
    pub fn stop(&mut self) {
        self.running = false;
        println!("[engine] Stopped");
    }

    pub fn is_running(&self) -> bool { self.running }

    /// Get the default spawn position.
    pub fn default_spawn_position(&self) -> [f32; 3] {
        self.scene.find_by_tag("default_spawn")
            .first()
            .map(|e| e.position)
            .unwrap_or([0.0, 2.0, 0.0])
    }
}

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
    fn create_engine_and_scene() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.scene.spawn("Player", Some("player.ply"), [0.0, 1.0, 0.0]);
        engine.scene.spawn("Enemy", Some("enemy.ply"), [10.0, 0.0, 5.0]);
        assert_eq!(engine.scene.entity_count(), 2);
    }

    #[test]
    fn find_entity_by_name() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.scene.spawn("Player", Some("player.ply"), [0.0, 1.0, 0.0]);
        let player = engine.scene.find_by_name("Player");
        assert!(player.is_some());
        assert_eq!(player.unwrap().position, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn spawn_with_script() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.scripts.register("TestScript", || Box::new(TestScript));
        engine.scene.spawn_with_script("NPC", Some("npc.ply"), [5.0, 0.0, 5.0], "TestScript");
        assert_eq!(engine.scene.entities[0].scripts, vec!["TestScript"]);
    }

    #[test]
    fn engine_tick() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.scripts.register("TestScript", || Box::new(TestScript));
        engine.scene.spawn_with_script("NPC", None, [0.0, 0.0, 0.0], "TestScript");
        engine.start();
        assert!(engine.tick(0.016));
        assert_eq!(engine.stats.frame_number, 1);
    }

    #[test]
    fn load_scene_and_populate() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.load_scene("Test Level");
        engine.add_entity("House", Some("house.ply"), [10.0, 0.0, 5.0], None);
        engine.scene.add_point_light([5.0, 3.0, 0.0], [1.0, 0.9, 0.8], 50.0, 30.0);
        assert_eq!(engine.scene.entity_count(), 1);
        assert!(engine.scene.lights.len() >= 2); // default directional + point
    }

    #[test]
    fn destroy_entity() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let id = engine.scene.spawn("Temp", None, [0.0, 0.0, 0.0]);
        assert_eq!(engine.scene.entity_count(), 1);
        engine.scene.destroy(id);
        assert_eq!(engine.scene.entity_count(), 0);
    }

    #[test]
    fn tags_and_find() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let id = engine.scene.spawn("Coin", Some("coin.ply"), [5.0, 1.0, 5.0]);
        engine.scene.get_mut(id).unwrap().tags.push("collectible".to_string());

        let collectibles = engine.scene.find_by_tag("collectible");
        assert_eq!(collectibles.len(), 1);
        assert_eq!(collectibles[0].name, "Coin");
    }

    // --- Fix 3: Scripts that visibly move entities ---

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
        engine.scripts.register("Mover", || Box::new(MoverScript));
        let id = engine.add_entity("Thing", None, [0.0, 0.0, 0.0], Some("Mover"));
        engine.start();
        engine.tick(0.016);

        let entity = engine.scene.get(id).unwrap();
        assert_eq!(entity.position, [99.0, 0.0, 0.0], "Script should have moved entity");
    }

    // --- Fix 4: AABB collision detection ---

    #[test]
    fn aabb_collision_detected_when_overlapping() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let a = engine.scene.spawn_with_collider(
            "BoxA", None, [0.0, 0.0, 0.0],
            ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] },
        );
        let b = engine.scene.spawn_with_collider(
            "BoxB", None, [1.5, 0.0, 0.0],
            ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] },
        );
        engine.start();
        engine.tick(0.016);

        assert!(!engine.last_collisions.is_empty(),
            "Overlapping boxes should collide: {:?}", engine.last_collisions);
        assert!(engine.last_collisions.contains(&(a, b)),
            "Collision pair ({}, {}) not found in {:?}", a, b, engine.last_collisions);
    }

    #[test]
    fn aabb_no_collision_when_separated() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        engine.scene.spawn_with_collider(
            "BoxA", None, [0.0, 0.0, 0.0],
            ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] },
        );
        engine.scene.spawn_with_collider(
            "BoxB", None, [10.0, 0.0, 0.0],
            ColliderShape::Box { half_extents: [1.0, 1.0, 1.0] },
        );
        engine.start();
        engine.tick(0.016);

        assert!(engine.last_collisions.is_empty(),
            "Separated boxes should not collide: {:?}", engine.last_collisions);
    }

    #[test]
    fn sphere_collision_detected() {
        let mut engine = EngineRuntime::new(EngineConfig::default());
        let a = engine.scene.spawn_with_collider(
            "SphereA", None, [0.0, 0.0, 0.0],
            ColliderShape::Sphere { radius: 2.0 },
        );
        let b = engine.scene.spawn_with_collider(
            "SphereB", None, [3.0, 0.0, 0.0],
            ColliderShape::Sphere { radius: 2.0 },
        );
        engine.start();
        engine.tick(0.016);

        assert!(engine.last_collisions.contains(&(a, b)),
            "Overlapping spheres should collide");
    }

    // --- Fix 5: Audio proof (WAV on script command) ---

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
        engine.scripts.register("SoundScript", || Box::new(SoundScript));
        engine.add_entity("Speaker", None, [0.0, 0.0, 0.0], Some("SoundScript"));
        engine.start();

        // Check that a WAV file was created
        let wav_path = std::env::temp_dir().join("ochroma_test_sound.wav");
        assert!(wav_path.exists(), "WAV file should exist at {}", wav_path.display());

        // Verify it's a valid WAV file (starts with RIFF header)
        let data = std::fs::read(&wav_path).unwrap();
        assert!(data.len() > 44, "WAV file should have header + data");
        assert_eq!(&data[0..4], b"RIFF", "Should start with RIFF header");
        assert_eq!(&data[8..12], b"WAVE", "Should contain WAVE marker");

        println!("WAV file generated: {} bytes at {}", data.len(), wav_path.display());
        let _ = std::fs::remove_file(&wav_path);
    }
}
