//! egui window for the animation state machine editor.

use crate::anim_editor::{AnimGraphDefinition, AnimParameter, AnimState, AnimTransition};

pub struct AnimEditorUi {
    pub open: bool,
    pub graph: Option<AnimGraphDefinition>,
    selected_state: Option<String>,
}

impl AnimEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: None,
            selected_state: None,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }

        egui::Window::new("Animation Editor")
            .open(&mut self.open)
            .default_size([700.0, 450.0])
            .resizable(true)
            .show(ctx, |ui| {
                if self.graph.is_none() {
                    ui.centered_and_justified(|ui| {
                        ui.label("No animation graph loaded.");
                    });
                    return;
                }

                let graph = self.graph.as_ref().unwrap();
                let graph_name = graph.name.clone();
                let state_count = graph.states.len();
                let trans_count = graph.transitions.len();
                let param_count = graph.parameters.len();
                let default_state = graph.default_state.clone();

                ui.horizontal(|ui| {
                    ui.heading(&graph_name);
                    ui.separator();
                    ui.label(format!(
                        "{} states  {} transitions  {} params",
                        state_count, trans_count, param_count
                    ));
                });
                ui.separator();

                // Collect state data
                let states: Vec<(String, String, [f32; 2], bool)> = self
                    .graph
                    .as_ref()
                    .unwrap()
                    .states
                    .iter()
                    .map(|s| {
                        (
                            s.name.clone(),
                            s.clip_path.clone(),
                            s.position,
                            s.name == default_state,
                        )
                    })
                    .collect();
                let transitions: Vec<(String, String, f32)> = self
                    .graph
                    .as_ref()
                    .unwrap()
                    .transitions
                    .iter()
                    .map(|t| (t.from.clone(), t.to.clone(), t.blend_duration))
                    .collect();
                let params: Vec<(String, String)> = self
                    .graph
                    .as_ref()
                    .unwrap()
                    .parameters
                    .iter()
                    .map(|p| (p.name.clone(), format!("{:?}", p.param_type)))
                    .collect();

                egui::SidePanel::left("anim_states")
                    .default_width(180.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("States").strong());
                        for (name, _, _, is_default) in &states {
                            let is_sel =
                                self.selected_state.as_deref() == Some(name.as_str());
                            let label = if *is_default {
                                format!("★ {}", name)
                            } else {
                                format!("  {}", name)
                            };
                            if ui.selectable_label(is_sel, &label).clicked() {
                                self.selected_state = Some(name.clone());
                            }
                        }
                        ui.separator();
                        ui.label(egui::RichText::new("Parameters").strong());
                        for (name, type_str) in &params {
                            ui.label(format!("  {} ({})", name, type_str));
                        }
                    });

                egui::SidePanel::right("anim_props")
                    .default_width(200.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("Properties").strong());
                        if let Some(sel) = &self.selected_state {
                            if let Some((name, clip, _, _)) =
                                states.iter().find(|(n, _, _, _)| n == sel)
                            {
                                ui.label(format!("State: {}", name));
                                ui.label(format!("Clip: {}", clip));
                                ui.separator();
                                let outgoing: Vec<_> = transitions
                                    .iter()
                                    .filter(|(from, _, _)| from == sel)
                                    .collect();
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} transitions:",
                                        outgoing.len()
                                    ))
                                    .strong(),
                                );
                                for (_, to, blend) in &outgoing {
                                    ui.label(format!("  → {} ({:.2}s)", to, blend));
                                }
                            }
                        } else {
                            ui.label("Select a state.");
                        }
                    });

                egui::CentralPanel::default().show_inside(ui, |ui| {
                    let painter = ui.painter();
                    let origin = ui.min_rect().min;

                    // Draw transition arrows
                    for (from_name, to_name, _) in &transitions {
                        if let (Some((_, _, from_pos, _)), Some((_, _, to_pos, _))) = (
                            states.iter().find(|(n, _, _, _)| n == from_name),
                            states.iter().find(|(n, _, _, _)| n == to_name),
                        ) {
                            let from_c = egui::pos2(
                                origin.x + from_pos[0] + 75.0,
                                origin.y + from_pos[1] + 25.0,
                            );
                            let to_c = egui::pos2(
                                origin.x + to_pos[0] + 75.0,
                                origin.y + to_pos[1] + 25.0,
                            );
                            painter.line_segment(
                                [from_c, to_c],
                                egui::Stroke::new(
                                    1.5,
                                    egui::Color32::from_rgb(150, 150, 200),
                                ),
                            );
                            // Arrowhead
                            let dir = (to_c - from_c).normalized();
                            let mid = from_c + (to_c - from_c) * 0.6;
                            let perp = egui::vec2(-dir.y, dir.x) * 5.0;
                            painter.line_segment(
                                [mid, mid - dir * 8.0 + perp],
                                egui::Stroke::new(
                                    1.5,
                                    egui::Color32::from_rgb(150, 150, 200),
                                ),
                            );
                            painter.line_segment(
                                [mid, mid - dir * 8.0 - perp],
                                egui::Stroke::new(
                                    1.5,
                                    egui::Color32::from_rgb(150, 150, 200),
                                ),
                            );
                        }
                    }

                    // Draw state boxes
                    for (name, _, pos, is_default) in &states {
                        let x = origin.x + pos[0];
                        let y = origin.y + pos[1];
                        let rect = egui::Rect::from_min_size(
                            egui::pos2(x, y),
                            egui::vec2(150.0, 50.0),
                        );
                        let is_sel =
                            self.selected_state.as_deref() == Some(name.as_str());
                        let fill = if is_sel {
                            egui::Color32::from_rgb(40, 80, 130)
                        } else if *is_default {
                            egui::Color32::from_rgb(40, 80, 60)
                        } else {
                            egui::Color32::from_rgb(35, 40, 55)
                        };
                        painter.rect_filled(rect, 6.0, fill);
                        painter.rect_stroke(
                            rect,
                            6.0,
                            egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 110, 160)),
                            egui::StrokeKind::Outside,
                        );
                        painter.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            name,
                            egui::FontId::proportional(13.0),
                            egui::Color32::WHITE,
                        );
                        if *is_default {
                            painter.text(
                                rect.min + egui::vec2(4.0, 2.0),
                                egui::Align2::LEFT_TOP,
                                "★",
                                egui::FontId::proportional(10.0),
                                egui::Color32::from_rgb(200, 200, 80),
                            );
                        }
                    }
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anim_ui_default_closed() {
        let ui = AnimEditorUi::new();
        assert!(!ui.open);
        assert!(ui.graph.is_none());
    }
}
