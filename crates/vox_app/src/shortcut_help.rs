/// Keyboard shortcut reference overlay.
pub struct ShortcutHelp {
    pub visible: bool,
}

impl ShortcutHelp {
    pub fn new() -> Self { Self { visible: false } }
    pub fn toggle(&mut self) { self.visible = !self.visible; }

    pub fn show(&self, ctx: &egui::Context) {
        if !self.visible { return; }

        egui::Window::new("Keyboard Shortcuts")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts").striped(true).show(ui, |ui| {
                    let shortcuts = [
                        ("F1", "Toggle this help"),
                        ("F3", "Toggle FPS counter"),
                        ("F12", "Take screenshot"),
                        ("~", "Toggle debug console"),
                        ("Esc", "Cancel / Deselect"),
                        ("Ctrl+S", "Quick save"),
                        ("Ctrl+Z", "Undo"),
                        ("Ctrl+Y", "Redo"),
                        ("", ""),
                        ("WASD", "Pan camera"),
                        ("Middle Mouse", "Orbit camera"),
                        ("Right Mouse", "Pan camera (alt)"),
                        ("Scroll", "Zoom in/out"),
                        ("", ""),
                        ("1", "Place mode"),
                        ("2", "Select mode"),
                        ("3", "Zone mode"),
                        ("4", "Service mode"),
                        ("5", "Road mode"),
                        ("", ""),
                        ("Space", "Pause/Resume"),
                        ("+", "Speed up"),
                        ("-", "Slow down"),
                    ];

                    for (key, desc) in &shortcuts {
                        if key.is_empty() {
                            ui.separator();
                            ui.separator();
                        } else {
                            ui.strong(*key);
                            ui.label(*desc);
                        }
                        ui.end_row();
                    }
                });
            });
    }
}
