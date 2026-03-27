use glam::Vec3;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub enum UiAction {
    PlaceAsset { asset_uuid: Uuid, position: Vec3 },
    SelectInstance { instance_id: u32 },
    Deselect,
    PlaceService { service_type: String, position: Vec3 },
    ZoneArea { zone_type: String, position: Vec3 },
    BuildRoad { start: Vec3, end: Vec3 },
    ChangeGameSpeed { speed: u8 }, // 0=pause, 1=normal, 2=fast, 3=veryfast
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Place,
    Select,
    Zone,
    Service,
    Road,
}

/// Simulation state snapshot populated by main.rs each frame.
pub struct SimInfo {
    pub funds: f64,
    pub total_income: f64,
    pub total_expenses: f64,
    pub net: f64,
    pub citizen_count: u32,
    pub avg_satisfaction: f32,
    pub demand_residential: f32,
    pub demand_commercial: f32,
    pub demand_industrial: f32,
}

pub struct PlopUi {
    pub selected_asset: Option<Uuid>,
    pub selected_instance: Option<u32>,
    pub mode: UiMode,
    pub asset_search: String,
    pub spectral_wear: f32,
    pub spectral_shift: f32,
    pub actions: Vec<UiAction>,
    pub click_position: Option<Vec3>,
    pub selected_zone_type: String,
    pub selected_service_type: String,
    pub road_start: Option<Vec3>,
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
            actions: Vec::new(),
            click_position: None,
            selected_zone_type: "Residential Low".to_string(),
            selected_service_type: "School".to_string(),
            road_start: None,
        }
    }
}

impl PlopUi {
    /// Handle a click in the 3D viewport at the given world position.
    pub fn handle_viewport_click(&mut self, world_pos: Vec3, instance_at_cursor: Option<u32>) {
        self.click_position = Some(world_pos);
        match self.mode {
            UiMode::Place => {
                if let Some(uuid) = self.selected_asset {
                    self.actions.push(UiAction::PlaceAsset {
                        asset_uuid: uuid,
                        position: world_pos,
                    });
                }
            }
            UiMode::Select => {
                if let Some(id) = instance_at_cursor {
                    self.selected_instance = Some(id);
                    self.actions.push(UiAction::SelectInstance { instance_id: id });
                } else {
                    self.selected_instance = None;
                    self.actions.push(UiAction::Deselect);
                }
            }
            UiMode::Zone => {
                self.actions.push(UiAction::ZoneArea {
                    zone_type: self.selected_zone_type.clone(),
                    position: world_pos,
                });
            }
            UiMode::Service => {
                self.actions.push(UiAction::PlaceService {
                    service_type: self.selected_service_type.clone(),
                    position: world_pos,
                });
            }
            UiMode::Road => {
                if let Some(start) = self.road_start.take() {
                    self.actions.push(UiAction::BuildRoad { start, end: world_pos });
                    println!("[ochroma] Road segment: ({:.1}, {:.1}) -> ({:.1}, {:.1})", start.x, start.z, world_pos.x, world_pos.z);
                } else {
                    self.road_start = Some(world_pos);
                    println!("[ochroma] Road start set at ({:.1}, {:.1})", world_pos.x, world_pos.z);
                }
            }
        }
    }

    /// Take pending actions (drains the queue).
    pub fn take_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.actions)
    }

    pub fn show(&mut self, ctx: &egui::Context, asset_names: &[(Uuid, String)], sim_info: Option<&SimInfo>) {
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

        // Right panel: budget and simulation info
        egui::SidePanel::right("budget_panel").resizable(true).default_width(200.0).show(ctx, |ui| {
            ui.heading("Budget");
            if let Some(sim) = sim_info {
                ui.label(format!("Funds: ${:.0}", sim.funds));
                ui.separator();
                ui.label(format!("Income: ${:.0}/mo", sim.total_income));
                ui.label(format!("Expenses: ${:.0}/mo", sim.total_expenses));
                ui.label(format!("Net: ${:.0}/mo", sim.net));
                ui.separator();
                ui.heading("Population");
                ui.label(format!("Citizens: {}", sim.citizen_count));
                ui.label(format!("Satisfaction: {:.0}%", sim.avg_satisfaction * 100.0));
                ui.separator();
                ui.heading("Demand");
                ui.add(egui::ProgressBar::new(sim.demand_residential).text("Residential"));
                ui.add(egui::ProgressBar::new(sim.demand_commercial).text("Commercial"));
                ui.add(egui::ProgressBar::new(sim.demand_industrial).text("Industrial"));
            } else {
                ui.label("No simulation running.");
            }
        });

        // Bottom panel: toolbar
        egui::TopBottomPanel::bottom("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.mode == UiMode::Place, "Place").clicked() {
                    self.mode = UiMode::Place;
                }
                if ui.selectable_label(self.mode == UiMode::Select, "Select").clicked() {
                    self.mode = UiMode::Select;
                }
                if ui.selectable_label(self.mode == UiMode::Zone, "Zone").clicked() {
                    self.mode = UiMode::Zone;
                }
                if ui.selectable_label(self.mode == UiMode::Service, "Service").clicked() {
                    self.mode = UiMode::Service;
                }
                if ui.selectable_label(self.mode == UiMode::Road, "Road").clicked() {
                    self.mode = UiMode::Road;
                }

                ui.separator();

                // Zone type selector (when in Zone mode)
                if self.mode == UiMode::Zone {
                    ui.label("Zone:");
                    egui::ComboBox::from_id_salt("zone_type")
                        .selected_text(&self.selected_zone_type)
                        .show_ui(ui, |ui| {
                            for zt in &["Residential Low", "Residential Med", "Residential High", "Commercial", "Industrial"] {
                                ui.selectable_value(&mut self.selected_zone_type, zt.to_string(), *zt);
                            }
                        });
                }

                // Service type selector (when in Service mode)
                if self.mode == UiMode::Service {
                    ui.label("Service:");
                    egui::ComboBox::from_id_salt("service_type")
                        .selected_text(&self.selected_service_type)
                        .show_ui(ui, |ui| {
                            for st in &["School", "Hospital", "Fire Station", "Police Station"] {
                                ui.selectable_value(&mut self.selected_service_type, st.to_string(), *st);
                            }
                        });
                }

                // Spectral controls when selecting
                if self.mode == UiMode::Select && self.selected_instance.is_some() {
                    ui.separator();
                    ui.label("Wear:");
                    ui.add(egui::Slider::new(&mut self.spectral_wear, 0.0..=1.0));
                    ui.label("Color shift:");
                    ui.add(egui::Slider::new(&mut self.spectral_shift, -0.5..=0.5));
                }
            });
        });

        // Top panel: info bar
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
                    UiMode::Zone => {
                        ui.label(format!("Click to zone: {}", self.selected_zone_type));
                    }
                    UiMode::Service => {
                        ui.label(format!("Click to place service: {}", self.selected_service_type));
                    }
                    UiMode::Road => {
                        if self.road_start.is_some() {
                            ui.label("Click to set road end point");
                        } else {
                            ui.label("Click to set road start point");
                        }
                    }
                }
            });
        });
    }
}
