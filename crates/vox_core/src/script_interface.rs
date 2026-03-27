/// The interface that game scripts implement.
/// This is the equivalent of Unity's MonoBehaviour or Unreal's AActor.
pub trait GameScript: Send + Sync {
    /// Called once when the entity is created.
    fn on_start(&mut self, _ctx: &mut ScriptContext) {}
    /// Called every frame.
    fn on_update(&mut self, _ctx: &mut ScriptContext, _dt: f32) {}
    /// Called when the entity is destroyed.
    fn on_destroy(&mut self, _ctx: &mut ScriptContext) {}
    /// Called when this entity collides with another.
    fn on_collision(&mut self, _ctx: &mut ScriptContext, _other_entity: u32) {}
    /// Script name for identification.
    fn name(&self) -> &str;
}

/// Context passed to scripts -- their window into the engine.
pub struct ScriptContext {
    pub entity_id: u32,
    /// Pending commands from this script (processed by engine after update).
    pub commands: Vec<ScriptCommand>,
}

/// Commands a script can issue to the engine.
#[derive(Debug, Clone)]
pub enum ScriptCommand {
    /// Spawn a new entity with the given asset.
    Spawn { asset_path: String, position: [f32; 3], rotation: [f32; 4], scale: [f32; 3] },
    /// Destroy an entity.
    Destroy { entity_id: u32 },
    /// Move this entity.
    SetPosition { position: [f32; 3] },
    /// Set rotation.
    SetRotation { rotation: [f32; 4] },
    /// Play a sound.
    PlaySound { clip: String, volume: f32, spatial: bool },
    /// Apply a force to a rigid body.
    ApplyForce { force: [f32; 3] },
    /// Send a custom event.
    SendEvent { name: String, data: String },
    /// Log a message.
    Log { message: String },
}

impl ScriptContext {
    pub fn new(entity_id: u32) -> Self {
        Self { entity_id, commands: Vec::new() }
    }

    pub fn spawn(&mut self, asset: &str, position: [f32; 3]) {
        self.commands.push(ScriptCommand::Spawn {
            asset_path: asset.to_string(),
            position,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        });
    }

    pub fn destroy(&mut self, entity_id: u32) {
        self.commands.push(ScriptCommand::Destroy { entity_id });
    }

    pub fn set_position(&mut self, pos: [f32; 3]) {
        self.commands.push(ScriptCommand::SetPosition { position: pos });
    }

    pub fn play_sound(&mut self, clip: &str, volume: f32) {
        self.commands.push(ScriptCommand::PlaySound {
            clip: clip.to_string(), volume, spatial: true,
        });
    }

    pub fn log(&mut self, msg: &str) {
        self.commands.push(ScriptCommand::Log { message: msg.to_string() });
    }

    /// Take all pending commands (called by engine after update).
    pub fn take_commands(&mut self) -> Vec<ScriptCommand> {
        std::mem::take(&mut self.commands)
    }
}

/// Registry of available script types.
pub struct ScriptRegistry {
    factories: std::collections::HashMap<String, Box<dyn Fn() -> Box<dyn GameScript>>>,
}

impl ScriptRegistry {
    pub fn new() -> Self {
        Self { factories: std::collections::HashMap::new() }
    }

    /// Register a script type so it can be attached to entities by name.
    pub fn register<F: Fn() -> Box<dyn GameScript> + 'static>(&mut self, name: &str, factory: F) {
        self.factories.insert(name.to_string(), Box::new(factory));
    }

    /// Create a script instance by name.
    pub fn create(&self, name: &str) -> Option<Box<dyn GameScript>> {
        self.factories.get(name).map(|f| f())
    }

    pub fn registered_scripts(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}
