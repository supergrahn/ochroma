use bevy_ecs::prelude::*;

/// Resource that scripts use to control the in-game UI.
#[derive(Resource, Default)]
pub struct UIBridge {
    pub commands: Vec<UICommand>,
}

#[derive(Debug, Clone)]
pub enum UICommand {
    SetText { id: String, text: String },
    SetProgress { id: String, value: f32 },
    SetVisible { id: String, visible: bool },
    ShowNotification { message: String, duration: f32 },
    SetGameState { state: String }, // "menu", "playing", "paused", "gameover"
}

impl UIBridge {
    pub fn set_text(&mut self, id: &str, text: &str) {
        self.commands.push(UICommand::SetText {
            id: id.to_string(),
            text: text.to_string(),
        });
    }

    pub fn set_progress(&mut self, id: &str, value: f32) {
        self.commands.push(UICommand::SetProgress {
            id: id.to_string(),
            value,
        });
    }

    pub fn show_notification(&mut self, message: &str, duration: f32) {
        self.commands.push(UICommand::ShowNotification {
            message: message.to_string(),
            duration,
        });
    }

    pub fn set_visible(&mut self, id: &str, visible: bool) {
        self.commands.push(UICommand::SetVisible {
            id: id.to_string(),
            visible,
        });
    }

    pub fn take_commands(&mut self) -> Vec<UICommand> {
        std::mem::take(&mut self.commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_bridge_queues_commands() {
        let mut bridge = UIBridge::default();
        bridge.set_text("health", "100");
        bridge.set_progress("xp_bar", 0.75);
        bridge.show_notification("Level up!", 3.0);
        bridge.set_visible("minimap", false);
        assert_eq!(bridge.commands.len(), 4);
    }

    #[test]
    fn take_commands_drains() {
        let mut bridge = UIBridge::default();
        bridge.set_text("score", "42");
        bridge.set_progress("hp", 0.5);
        let cmds = bridge.take_commands();
        assert_eq!(cmds.len(), 2);
        assert!(bridge.commands.is_empty());
    }
}
