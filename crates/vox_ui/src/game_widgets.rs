//! Immediate-mode in-game UI widgets rendered via egui.

use egui::{Color32, Context, RichText, Ui};

/// One resource row in a `WidgetCmd::Panel`.
#[derive(Debug, Clone)]
pub struct ResourceRow {
    pub label: String,
    pub count: u32,
    pub icon_color: Color32,
}

/// Commands pushed each frame to describe what to render.
#[derive(Debug, Clone)]
pub enum WidgetCmd {
    /// Floating panel with a title and resource rows.
    Panel { title: String, rows: Vec<ResourceRow> },
    /// Hover tooltip — shown at current mouse position.
    Tooltip { text: String },
    /// Clickable button (rendering only — click state tracked externally).
    Button { label: String, id: String },
    /// Full-width status bar at the bottom of the screen.
    StatusBar { text: String },
}

/// Renders game UI widgets each frame.
pub struct GameWidgets;

impl GameWidgets {
    pub fn new() -> Self {
        Self
    }

    /// Render all widget commands into the egui context for this frame.
    pub fn render(&self, ctx: &Context, cmds: &[WidgetCmd]) {
        for cmd in cmds {
            match cmd {
                WidgetCmd::Panel { title, rows } => {
                    egui::Window::new(title.as_str())
                        .resizable(false)
                        .collapsible(false)
                        .show(ctx, |ui| {
                            for row in rows {
                                ui.horizontal(|ui| {
                                    let dot = egui::Shape::circle_filled(
                                        ui.cursor().min + egui::vec2(6.0, 6.0),
                                        5.0,
                                        row.icon_color,
                                    );
                                    ui.painter().add(dot);
                                    ui.add_space(14.0);
                                    ui.label(
                                        RichText::new(format!("{}: {}", row.label, row.count))
                                            .size(13.0),
                                    );
                                });
                            }
                        });
                }
                WidgetCmd::Tooltip { text } => {
                    egui::show_tooltip_at_pointer(
                        ctx,
                        egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("game_tooltip_layer")),
                        egui::Id::new("game_tooltip"),
                        |ui| {
                            ui.label(text.as_str());
                        },
                    );
                }
                WidgetCmd::Button { label, id } => {
                    egui::Window::new(id.as_str())
                        .title_bar(false)
                        .resizable(false)
                        .show(ctx, |ui: &mut Ui| {
                            let _ = ui.button(label.as_str());
                        });
                }
                WidgetCmd::StatusBar { text } => {
                    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
                        ui.label(text.as_str());
                    });
                }
            }
        }
    }
}

impl Default for GameWidgets {
    fn default() -> Self {
        Self::new()
    }
}
