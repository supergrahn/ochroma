//! VFX Editor UI — egui window for inspecting and editing VFX assets.

use egui::{Context, Window};

use crate::vfx_editor::{VfxAsset, VfxCategory};

const ALL_CATEGORIES: &[VfxCategory] = &[
    VfxCategory::Fire,
    VfxCategory::Smoke,
    VfxCategory::Explosion,
    VfxCategory::Weather,
    VfxCategory::Magic,
    VfxCategory::Environment,
    VfxCategory::UI,
    VfxCategory::Custom,
];

pub struct VfxEditorUi {
    pub open: bool,
    pub asset_library: Vec<VfxAsset>,
    pub selected_asset: Option<usize>,
}

impl VfxEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            asset_library: Vec::new(),
            selected_asset: None,
        }
    }

    pub fn show(&mut self, ctx: &Context) {
        if !self.open {
            return;
        }

        let mut open = self.open;
        Window::new("VFX Editor")
            .open(&mut open)
            .resizable(true)
            .min_width(520.0)
            .show(ctx, |ui| {
                egui::SidePanel::left("vfx_asset_list")
                    .resizable(true)
                    .min_width(160.0)
                    .show_inside(ui, |ui| {
                        ui.heading("Assets");
                        ui.separator();
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (i, asset) in self.asset_library.iter().enumerate() {
                                let label = format!(
                                    "[{}] {}",
                                    format!("{:?}", asset.category),
                                    asset.name
                                );
                                let selected = self.selected_asset == Some(i);
                                if ui.selectable_label(selected, &label).clicked() {
                                    self.selected_asset = Some(i);
                                }
                            }
                        });
                    });

                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if let Some(idx) = self.selected_asset {
                        if let Some(asset) = self.asset_library.get_mut(idx) {
                            ui.heading("Properties");
                            ui.separator();

                            egui::Grid::new("vfx_props_grid")
                                .num_columns(2)
                                .spacing([8.0, 4.0])
                                .show(ui, |ui| {
                                    ui.label("Name");
                                    ui.text_edit_singleline(&mut asset.name);
                                    ui.end_row();

                                    ui.label("Category");
                                    egui::ComboBox::from_id_salt("vfx_category_combo")
                                        .selected_text(format!("{:?}", asset.category))
                                        .show_ui(ui, |ui| {
                                            for &cat in ALL_CATEGORIES {
                                                ui.selectable_value(
                                                    &mut asset.category,
                                                    cat,
                                                    format!("{:?}", cat),
                                                );
                                            }
                                        });
                                    ui.end_row();

                                    ui.label("Description");
                                    ui.text_edit_multiline(&mut asset.description);
                                    ui.end_row();

                                    ui.label("Preview Dist.");
                                    ui.add(
                                        egui::DragValue::new(&mut asset.preview_camera_distance)
                                            .speed(0.1)
                                            .range(0.1_f32..=500.0_f32)
                                            .suffix(" m"),
                                    );
                                    ui.end_row();
                                });

                            ui.add_space(8.0);
                            ui.heading("Effect");
                            ui.separator();

                            // Show per-emitter read-only stats
                            let emitter_count = asset.effect.emitters.len();
                            ui.label(format!("Emitters: {}", emitter_count));

                            for (i, emitter) in asset.effect.emitters.iter().enumerate() {
                                egui::CollapsingHeader::new(format!("Emitter {}", i))
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        egui::Grid::new(format!("vfx_emitter_grid_{}", i))
                                            .num_columns(2)
                                            .spacing([8.0, 2.0])
                                            .show(ui, |ui| {
                                                ui.label("Max Particles");
                                                ui.label(emitter.max_particles.to_string());
                                                ui.end_row();

                                                ui.label("Emit Rate");
                                                ui.label(format!("{:.1}/s", emitter.rate));
                                                ui.end_row();

                                                ui.label("Lifetime");
                                                ui.label(format!(
                                                    "{:.2} – {:.2} s",
                                                    emitter.lifetime.min, emitter.lifetime.max
                                                ));
                                                ui.end_row();

                                                ui.label("Speed Range");
                                                ui.label(format!(
                                                    "{:.2} – {:.2} m/s",
                                                    emitter.velocity.speed.min,
                                                    emitter.velocity.speed.max
                                                ));
                                                ui.end_row();
                                            });
                                    });
                            }
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("Select a VFX asset from the list.");
                        });
                    }
                });
            });

        self.open = open;
    }
}

impl Default for VfxEditorUi {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vfx_ui_default_closed() {
        let ui = VfxEditorUi::new();
        assert!(!ui.open);
        assert!(ui.selected_asset.is_none());
    }
}
