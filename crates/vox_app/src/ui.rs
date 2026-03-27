use uuid::Uuid;

pub struct PlopUi {
    pub selected_asset: Option<Uuid>,
    pub selected_instance: Option<u32>,
    pub mode: UiMode,
    pub asset_search: String,
    pub spectral_wear: f32,
    pub spectral_shift: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Place,
    Select,
}

impl Default for PlopUi {
    fn default() -> Self {
        Self {
            selected_asset: None,
            selected_instance: None,
            mode: UiMode::Place,
            asset_search: String::new(),
            spectral_wear: 0.0,
            spectral_shift: 0.0,
        }
    }
}

impl PlopUi {
    pub fn show(&mut self, ctx: &egui::Context, asset_names: &[(Uuid, String)]) {
        // Left panel: asset browser
        egui::SidePanel::left("asset_browser").show(ctx, |ui| {
            ui.heading("Asset Browser");
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.text_edit_singleline(&mut self.asset_search);
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (uuid, name) in asset_names {
                    if !self.asset_search.is_empty()
                        && !name.to_lowercase().contains(&self.asset_search.to_lowercase())
                    {
                        continue;
                    }
                    let selected = self.selected_asset == Some(*uuid);
                    if ui.selectable_label(selected, name).clicked() {
                        self.selected_asset = Some(*uuid);
                        self.mode = UiMode::Place;
                    }
                }
            });
        });

        // Bottom panel: tool bar
        egui::TopBottomPanel::bottom("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.mode == UiMode::Place, "Place").clicked() {
                    self.mode = UiMode::Place;
                }
                if ui.selectable_label(self.mode == UiMode::Select, "Select").clicked() {
                    self.mode = UiMode::Select;
                }
                ui.separator();
                if self.selected_instance.is_some() {
                    ui.label("Wear:");
                    ui.add(egui::Slider::new(&mut self.spectral_wear, 0.0..=1.0));
                    ui.label("Color shift:");
                    ui.add(egui::Slider::new(&mut self.spectral_shift, -0.5..=0.5));
                }
            });
        });

        // Top panel: info
        egui::TopBottomPanel::top("info_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Ochroma — Phase 1");
                ui.separator();
                match self.mode {
                    UiMode::Place => {
                        if self.selected_asset.is_some() {
                            ui.label("Click terrain to place selected asset");
                        } else {
                            ui.label("Select an asset from the browser");
                        }
                    }
                    UiMode::Select => {
                        if let Some(id) = self.selected_instance {
                            ui.label(format!("Selected instance: {}", id));
                        } else {
                            ui.label("Click an asset to select it");
                        }
                    }
                }
            });
        });
    }
}
