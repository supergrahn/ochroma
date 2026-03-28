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
    /// Set UI text element.
    UISetText { id: String, text: String },
    /// Set UI progress bar value.
    UISetProgress { id: String, value: f32 },
    /// Show a UI notification.
    UINotification { message: String },
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

    pub fn set_ui_text(&mut self, id: &str, text: &str) {
        self.commands.push(ScriptCommand::UISetText { id: id.to_string(), text: text.to_string() });
    }

    pub fn set_ui_progress(&mut self, id: &str, value: f32) {
        self.commands.push(ScriptCommand::UISetProgress { id: id.to_string(), value });
    }

    pub fn show_notification(&mut self, message: &str) {
        self.commands.push(ScriptCommand::UINotification { message: message.to_string() });
    }

    /// Take all pending commands (called by engine after update).
    pub fn take_commands(&mut self) -> Vec<ScriptCommand> {
        std::mem::take(&mut self.commands)
    }
}

/// Registry of available script types.
pub struct ScriptRegistry {
    factories: std::collections::HashMap<String, Box<dyn Fn() -> Box<dyn GameScript> + Send + Sync>>,
}

// Safety: All factory closures are Send + Sync (enforced by register signature).
unsafe impl Send for ScriptRegistry {}
unsafe impl Sync for ScriptRegistry {}

impl Default for ScriptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptRegistry {
    pub fn new() -> Self {
        Self { factories: std::collections::HashMap::new() }
    }

    /// Register a script type so it can be attached to entities by name.
    pub fn register<F: Fn() -> Box<dyn GameScript> + Send + Sync + 'static>(&mut self, name: &str, factory: F) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_context_ui_set_text() {
        let mut ctx = ScriptContext::new(1);
        ctx.set_ui_text("health_label", "HP: 100");
        let cmds = ctx.take_commands();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            ScriptCommand::UISetText { id, text } => {
                assert_eq!(id, "health_label");
                assert_eq!(text, "HP: 100");
            }
            _ => panic!("Expected UISetText"),
        }
    }

    #[test]
    fn script_context_ui_set_progress() {
        let mut ctx = ScriptContext::new(1);
        ctx.set_ui_progress("xp_bar", 0.5);
        let cmds = ctx.take_commands();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            ScriptCommand::UISetProgress { id, value } => {
                assert_eq!(id, "xp_bar");
                assert!((value - 0.5).abs() < f32::EPSILON);
            }
            _ => panic!("Expected UISetProgress"),
        }
    }

    #[test]
    fn script_context_ui_notification() {
        let mut ctx = ScriptContext::new(1);
        ctx.show_notification("Quest complete!");
        let cmds = ctx.take_commands();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            ScriptCommand::UINotification { message } => {
                assert_eq!(message, "Quest complete!");
            }
            _ => panic!("Expected UINotification"),
        }
    }
}
