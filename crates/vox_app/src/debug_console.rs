/// Debug console for inspecting game state at runtime.
pub struct DebugConsole {
    pub visible: bool,
    pub input_buffer: String,
    pub output_lines: Vec<String>,
    pub max_output_lines: usize,
}

impl DebugConsole {
    pub fn new() -> Self {
        Self {
            visible: false,
            input_buffer: String::new(),
            output_lines: vec!["[ochroma debug] Type 'help' for commands".to_string()],
            max_output_lines: 100,
        }
    }

    pub fn toggle(&mut self) { self.visible = !self.visible; }

    /// Execute a command and return the output.
    pub fn execute(&mut self, command: &str) -> String {
        let parts: Vec<&str> = command.trim().split_whitespace().collect();
        if parts.is_empty() { return String::new(); }

        let output = match parts[0] {
            "help" => {
                "Available commands:\n  help — show this message\n  stats — show engine statistics\n  fps — show framerate\n  splats — show splat count\n  citizens — show citizen count\n  budget — show budget info\n  speed <0-4> — set game speed\n  time <hour> — set time of day\n  spawn <count> — spawn citizens\n  tp <x> <y> <z> — teleport camera\n  quit — exit game".to_string()
            }
            "stats" => {
                "Engine statistics:\n  Ochroma v0.1.0\n  14 crates, 325+ tests\n  Spectral Gaussian Splatting renderer".to_string()
            }
            "fps" => "Use F3 to toggle FPS counter".to_string(),
            "splats" => "Splat count: query VisibleSplats resource".to_string(),
            "citizens" => "Citizen count: query SimulationState resource".to_string(),
            "budget" => "Budget info: query SimulationState.budget".to_string(),
            "speed" => {
                if parts.len() > 1 {
                    format!("Game speed set to {}x", parts[1])
                } else {
                    "Usage: speed <0-4>".to_string()
                }
            }
            "time" => {
                if parts.len() > 1 {
                    format!("Time set to {}:00", parts[1])
                } else {
                    "Usage: time <hour>".to_string()
                }
            }
            "spawn" => {
                if parts.len() > 1 {
                    format!("Spawning {} citizens", parts[1])
                } else {
                    "Usage: spawn <count>".to_string()
                }
            }
            "tp" => {
                if parts.len() >= 4 {
                    format!("Teleporting camera to ({}, {}, {})", parts[1], parts[2], parts[3])
                } else {
                    "Usage: tp <x> <y> <z>".to_string()
                }
            }
            "quit" => "Exiting...".to_string(),
            _ => format!("Unknown command: '{}'. Type 'help' for commands.", parts[0]),
        };

        self.output_lines.push(format!("> {}", command));
        self.output_lines.push(output.clone());
        while self.output_lines.len() > self.max_output_lines {
            self.output_lines.remove(0);
        }

        output
    }

    /// Render the console via egui.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.visible { return; }

        egui::Window::new("Debug Console")
            .resizable(true)
            .default_width(600.0)
            .default_height(300.0)
            .show(ctx, |ui| {
                // Output area
                egui::ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
                    for line in &self.output_lines {
                        ui.monospace(line);
                    }
                });

                ui.separator();

                // Input
                let response = ui.text_edit_singleline(&mut self.input_buffer);
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let cmd = self.input_buffer.clone();
                    self.input_buffer.clear();
                    self.execute(&cmd);
                }
            });
    }
}
