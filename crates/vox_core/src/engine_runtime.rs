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
            self.process_commands(commands);
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
                self.process_commands(commands);
            }
        }
        self.stats.script_time_ms = script_start.elapsed().as_secs_f32() * 1000.0;

        // Phase 3: Physics (simple collision — push entities out of colliders)
        // TODO: Rapier integration when feature enabled

        // Phase 4: Advance time
        self.scene.time_of_day = (self.scene.time_of_day + dt * 0.01) % 24.0; // slow time advance

        self.input.end_frame();
        true
    }

    /// Process commands issued by scripts.
    fn process_commands(&mut self, commands: Vec<ScriptCommand>) {
        for cmd in commands {
            match cmd {
                ScriptCommand::Spawn { asset_path, position, .. } => {
                    self.scene.spawn("Spawned", Some(&asset_path), position);
                }
                ScriptCommand::Destroy { entity_id } => {
                    self.scene.destroy(entity_id);
                }
                ScriptCommand::SetPosition { position } => {
                    // Applied to the entity that issued the command
                    // (simplified — in reality we'd track which entity issued it)
                }
                ScriptCommand::PlaySound { clip, volume, .. } => {
                    // TODO: wire to audio system
                    println!("[engine] play_sound: {} vol={}", clip, volume);
                }
                ScriptCommand::Log { message } => {
                    println!("[script] {}", message);
                }
                _ => {}
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
}
