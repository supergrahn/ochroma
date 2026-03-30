use crate::terrain_editor::TerrainEditorState;
use glam::{Quat, Vec3};

/// An entity in the editor's scene tree.
#[derive(Debug, Clone)]
pub struct EditorEntity {
    pub id: u32,
    pub name: String,
    pub asset_path: String,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub visible: bool,
    pub locked: bool,
    pub scripts: Vec<String>,
    pub parent: Option<u32>,
    pub children: Vec<u32>,
}

/// The scene editor state.
pub struct SceneEditor {
    pub entities: Vec<EditorEntity>,
    pub selected: Option<u32>,
    pub visible: bool,
    next_id: u32,

    // Gizmo state
    pub gizmo_mode: GizmoMode,
    pub snap_enabled: bool,
    pub snap_grid: f32,

    // Undo
    pub undo_stack: Vec<EditorAction>,
    pub redo_stack: Vec<EditorAction>,

    // Menu bar state
    pub pending_new_scene: bool,
    pub pending_open: bool,
    pub pending_save: bool,
    pub pending_save_as: bool,
    pub show_history: bool,
    pub show_material_editor: bool,
    pub show_anim_editor: bool,
    pub show_vfx_editor: bool,
    pub show_terrain_editor: bool,
    pub terrain_editor_state: TerrainEditorState,
    pub show_perf_stats: bool,
    pub pending_exit: bool,

    // Transform space
    pub transform_space: TransformSpace,

    // Viewport mode
    pub viewport_mode: ViewportMode,
    pub viewport_layout: ViewportLayout,

    // Play mode
    pub play_requested: bool,
    pub pause_requested: bool,
    pub stop_requested: bool,
    pub editor_mode: EditorPlayMode,

    // Status bar
    pub status_splat_count: usize,
    pub status_message: String,

    // Camera focus request from context menu
    pub focus_camera_on: Option<u32>,

    // Notification toasts
    pub notification_queue: crate::notifications::NotificationQueue,

    // Mini map
    pub mini_map: crate::minimap::MiniMap,

    // Settings panel
    pub show_settings: bool,
    pub app_settings: crate::settings::AppSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorPlayMode {
    Editing,
    Playing,
    Paused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformSpace {
    World,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportMode {
    Lit,
    Unlit,
    Wireframe,
    Normals,
    Overdraw,
}

impl ViewportMode {
    pub fn label(self) -> &'static str {
        match self {
            ViewportMode::Lit       => "Lit",
            ViewportMode::Unlit     => "Unlit",
            ViewportMode::Wireframe => "Wireframe",
            ViewportMode::Normals   => "Normals",
            ViewportMode::Overdraw  => "Overdraw",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportLayout {
    Single,
    HSplit,
    VSplit,
    Quad,
}

impl ViewportLayout {
    pub fn label(self) -> &'static str {
        match self {
            ViewportLayout::Single => "1 View",
            ViewportLayout::HSplit => "2 Views H",
            ViewportLayout::VSplit => "2 Views V",
            ViewportLayout::Quad   => "4 Views",
        }
    }
}

#[derive(Debug, Clone)]
pub enum EditorAction {
    MoveEntity {
        id: u32,
        old_pos: Vec3,
        new_pos: Vec3,
    },
    RotateEntity {
        id: u32,
        old_rot: Quat,
        new_rot: Quat,
    },
    ScaleEntity {
        id: u32,
        old_scale: Vec3,
        new_scale: Vec3,
    },
    AddEntity {
        id: u32,
    },
    DeleteEntity {
        id: u32,
        entity: EditorEntity,
    },
    RenameEntity {
        id: u32,
        old_name: String,
        new_name: String,
    },
}

impl EditorAction {
    pub fn label(&self) -> String {
        match self {
            EditorAction::MoveEntity { id, .. }   => format!("Move #{}", id),
            EditorAction::RotateEntity { id, .. } => format!("Rotate #{}", id),
            EditorAction::ScaleEntity { id, .. }  => format!("Scale #{}", id),
            EditorAction::AddEntity { id }         => format!("Add #{}", id),
            EditorAction::DeleteEntity { id, .. } => format!("Delete #{}", id),
            EditorAction::RenameEntity { id, old_name, new_name } =>
                format!("Rename #{}: {} \u{2192} {}", id, old_name, new_name),
        }
    }
}

impl Default for SceneEditor {
    fn default() -> Self { Self::new() }
}

impl SceneEditor {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            selected: None,
            visible: true,
            next_id: 0,
            gizmo_mode: GizmoMode::Translate,
            snap_enabled: false,
            snap_grid: 1.0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            pending_new_scene: false,
            pending_open: false,
            pending_save: false,
            pending_save_as: false,
            show_history: false,
            show_material_editor: false,
            show_anim_editor: false,
            show_vfx_editor: false,
            show_terrain_editor: false,
            terrain_editor_state: TerrainEditorState::default(),
            show_perf_stats: false,
            pending_exit: false,
            play_requested: false,
            pause_requested: false,
            stop_requested: false,
            editor_mode: EditorPlayMode::Editing,
            transform_space: TransformSpace::World,
            viewport_mode: ViewportMode::Lit,
            viewport_layout: ViewportLayout::Single,
            status_splat_count: 0,
            status_message: String::from("Ready"),
            focus_camera_on: None,
            notification_queue: crate::notifications::NotificationQueue::new(5),
            mini_map: crate::minimap::MiniMap::default(),
            show_settings: false,
            app_settings: crate::settings::load_settings(std::path::Path::new("settings.toml")),
        }
    }

    pub fn status_text(&self) -> String {
        format!(
            "{} entities | {} splats | {}",
            self.entities.len(),
            self.status_splat_count,
            self.status_message
        )
    }

    /// Add a new entity to the scene.
    pub fn add_entity(&mut self, name: &str, asset_path: &str, position: Vec3) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(EditorEntity {
            id,
            name: name.to_string(),
            asset_path: asset_path.to_string(),
            position,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            visible: true,
            locked: false,
            scripts: Vec::new(),
            parent: None,
            children: Vec::new(),
        });
        self.undo_stack.push(EditorAction::AddEntity { id });
        self.redo_stack.clear();
        id
    }

    /// Delete the selected entity.
    pub fn delete_selected(&mut self) {
        if let Some(id) = self.selected
            && let Some(idx) = self.entities.iter().position(|e| e.id == id)
        {
            let entity = self.entities.remove(idx);
            self.undo_stack
                .push(EditorAction::DeleteEntity { id, entity });
            self.redo_stack.clear();
            self.selected = None;
        }
    }

    /// Move the selected entity by a delta.
    pub fn move_selected(&mut self, delta: Vec3) {
        if let Some(id) = self.selected
            && let Some(entity) = self.entities.iter_mut().find(|e| e.id == id)
        {
            if entity.locked {
                return;
            }
            let old_pos = entity.position;
            entity.position += delta;
            if self.snap_enabled {
                entity.position.x =
                    (entity.position.x / self.snap_grid).round() * self.snap_grid;
                entity.position.y =
                    (entity.position.y / self.snap_grid).round() * self.snap_grid;
                entity.position.z =
                    (entity.position.z / self.snap_grid).round() * self.snap_grid;
            }
            self.undo_stack.push(EditorAction::MoveEntity {
                id,
                old_pos,
                new_pos: entity.position,
            });
            self.redo_stack.clear();
        }
    }

    /// Undo the last action.
    pub fn undo(&mut self) {
        if let Some(action) = self.undo_stack.pop() {
            match &action {
                EditorAction::MoveEntity { id, old_pos, .. } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.position = *old_pos;
                    }
                }
                EditorAction::AddEntity { id } => {
                    self.entities.retain(|e| e.id != *id);
                    if self.selected == Some(*id) {
                        self.selected = None;
                    }
                }
                EditorAction::DeleteEntity { entity, .. } => {
                    self.entities.push(entity.clone());
                }
                EditorAction::RenameEntity { id, old_name, .. } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.name = old_name.clone();
                    }
                }
                EditorAction::RotateEntity { id, old_rot, .. } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.rotation = *old_rot;
                    }
                }
                EditorAction::ScaleEntity {
                    id, old_scale, ..
                } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.scale = *old_scale;
                    }
                }
            }
            self.redo_stack.push(action);
        }
    }

    /// Redo the last undone action.
    pub fn redo(&mut self) {
        if let Some(action) = self.redo_stack.pop() {
            match &action {
                EditorAction::MoveEntity { id, new_pos, .. } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.position = *new_pos;
                    }
                }
                EditorAction::AddEntity { id } => {
                    // Re-add a default entity with that ID if it no longer exists
                    if !self.entities.iter().any(|e| e.id == *id) {
                        self.entities.push(EditorEntity {
                            id: *id,
                            name: format!("Entity {}", id),
                            asset_path: String::new(),
                            position: Vec3::ZERO,
                            rotation: glam::Quat::IDENTITY,
                            scale: Vec3::ONE,
                            visible: true,
                            locked: false,
                            scripts: Vec::new(),
                            parent: None,
                            children: Vec::new(),
                        });
                    }
                }
                EditorAction::DeleteEntity { id, .. } => {
                    self.entities.retain(|e| e.id != *id);
                }
                EditorAction::RenameEntity { id, new_name, .. } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.name = new_name.clone();
                    }
                }
                EditorAction::RotateEntity { id, new_rot, .. } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.rotation = *new_rot;
                    }
                }
                EditorAction::ScaleEntity {
                    id, new_scale, ..
                } => {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.scale = *new_scale;
                    }
                }
            }
            self.undo_stack.push(action);
        }
    }

    /// Select entity by ID.
    pub fn select(&mut self, id: u32) {
        self.selected = Some(id);
    }

    /// Get the selected entity.
    pub fn selected_entity(&self) -> Option<&EditorEntity> {
        self.selected
            .and_then(|id| self.entities.iter().find(|e| e.id == id))
    }

    /// Get mutable reference to selected entity.
    pub fn selected_entity_mut(&mut self) -> Option<&mut EditorEntity> {
        let id = self.selected?;
        self.entities.iter_mut().find(|e| e.id == id)
    }

    /// Duplicate the selected entity.
    pub fn duplicate_selected(&mut self) -> Option<u32> {
        let entity = self.selected_entity()?.clone();
        let new_pos = entity.position + Vec3::new(2.0, 0.0, 0.0); // offset
        Some(self.add_entity(
            &format!("{} (copy)", entity.name),
            &entity.asset_path,
            new_pos,
        ))
    }

    /// Rename an entity by ID, recording an undo action.
    pub fn rename_entity(&mut self, id: u32, new_name: &str) {
        if let Some(entity) = self.entities.iter_mut().find(|e| e.id == id) {
            let old_name = entity.name.clone();
            entity.name = new_name.to_string();
            self.undo_stack.push(EditorAction::RenameEntity {
                id,
                old_name,
                new_name: new_name.to_string(),
            });
            self.redo_stack.clear();
        }
    }

    /// Export scene to a MapFile.
    pub fn export_to_map(&self, name: &str) -> vox_data::map_file::MapFile {
        let mut map = vox_data::map_file::MapFile::new(name);
        for entity in &self.entities {
            let obj = vox_data::map_file::PlacedObject {
                name: entity.name.clone(),
                asset_path: entity.asset_path.clone(),
                position: [entity.position.x, entity.position.y, entity.position.z],
                rotation: [
                    entity.rotation.x,
                    entity.rotation.y,
                    entity.rotation.z,
                    entity.rotation.w,
                ],
                scale: [entity.scale.x, entity.scale.y, entity.scale.z],
                scripts: entity.scripts.clone(),
                properties: std::collections::HashMap::new(),
            };
            map.placed_objects.push(obj);
        }
        map
    }

    /// Import from a MapFile.
    pub fn import_from_map(&mut self, map: &vox_data::map_file::MapFile) {
        self.entities.clear();
        self.next_id = 0;
        for obj in &map.placed_objects {
            let id = self.next_id;
            self.next_id += 1;
            self.entities.push(EditorEntity {
                id,
                name: obj.name.clone(),
                asset_path: obj.asset_path.clone(),
                position: Vec3::new(obj.position[0], obj.position[1], obj.position[2]),
                rotation: Quat::from_xyzw(
                    obj.rotation[0],
                    obj.rotation[1],
                    obj.rotation[2],
                    obj.rotation[3],
                ),
                scale: Vec3::new(obj.scale[0], obj.scale[1], obj.scale[2]),
                visible: true,
                locked: false,
                scripts: obj.scripts.clone(),
                parent: None,
                children: Vec::new(),
            });
        }
    }

    /// Render the editor UI using egui.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Scene          Ctrl+N").clicked() {
                        self.pending_new_scene = true;
                        ui.close_menu();
                    }
                    if ui.button("Open...            Ctrl+O").clicked() {
                        self.pending_open = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save               Ctrl+S").clicked() {
                        self.pending_save = true;
                        ui.close_menu();
                    }
                    if ui.button("Save As...  Ctrl+Shift+S").clicked() {
                        self.pending_save_as = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        self.pending_exit = true;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo  Ctrl+Z").clicked() {
                        self.undo();
                        ui.close_menu();
                    }
                    if ui.button("Redo  Ctrl+Y").clicked() {
                        self.redo();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Duplicate  Ctrl+D").clicked() {
                        self.duplicate_selected();
                        ui.close_menu();
                    }
                    if ui.button("Delete  Del").clicked() {
                        self.delete_selected();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("History").clicked() {
                        self.show_history = !self.show_history;
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    if ui.button("Material Editor").clicked() {
                        self.show_material_editor = !self.show_material_editor;
                        ui.close_menu();
                    }
                    if ui.button("Animation Editor").clicked() {
                        self.show_anim_editor = !self.show_anim_editor;
                        ui.close_menu();
                    }
                    if ui.button("VFX Editor").clicked() {
                        self.show_vfx_editor = !self.show_vfx_editor;
                        ui.close_menu();
                    }
                    if ui.button("Terrain Editor").clicked() {
                        self.show_terrain_editor = !self.show_terrain_editor;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Performance Stats").clicked() {
                        self.show_perf_stats = !self.show_perf_stats;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Mini Map").clicked() {
                        self.mini_map.open = !self.mini_map.open;
                        ui.close_menu();
                    }
                    if ui.button("Settings").clicked() {
                        self.show_settings = !self.show_settings;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    ui.label("Ochroma Engine v0.1.0");
                    ui.separator();
                    ui.label("Tab — toggle editor");
                    ui.label("WASD — move camera");
                    ui.label("RMB drag — look");
                    ui.label("Scroll — zoom");
                    ui.label("F5 — play  F6 — pause  F7 — stop");
                });
            });
        });

        // Toolbar (top)
        egui::TopBottomPanel::top("editor_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Gizmo:");
                if ui
                    .selectable_label(self.gizmo_mode == GizmoMode::Translate, "Move (W)")
                    .clicked()
                {
                    self.gizmo_mode = GizmoMode::Translate;
                }
                if ui
                    .selectable_label(self.gizmo_mode == GizmoMode::Rotate, "Rotate (E)")
                    .clicked()
                {
                    self.gizmo_mode = GizmoMode::Rotate;
                }
                if ui
                    .selectable_label(self.gizmo_mode == GizmoMode::Scale, "Scale (R)")
                    .clicked()
                {
                    self.gizmo_mode = GizmoMode::Scale;
                }
                ui.separator();
                if ui.selectable_label(self.transform_space == TransformSpace::World, "World").clicked() {
                    self.transform_space = TransformSpace::World;
                }
                if ui.selectable_label(self.transform_space == TransformSpace::Local, "Local").clicked() {
                    self.transform_space = TransformSpace::Local;
                }
                ui.separator();
                ui.checkbox(&mut self.snap_enabled, "Snap");
                if self.snap_enabled {
                    ui.add(
                        egui::DragValue::new(&mut self.snap_grid)
                            .speed(0.1)
                            .prefix("Grid: "),
                    );
                }
                ui.separator();

                // Play button — only enabled in Editing mode
                let play_color = if self.editor_mode == EditorPlayMode::Editing {
                    egui::Color32::from_rgb(80, 200, 80)
                } else {
                    egui::Color32::GRAY
                };
                if ui.add_enabled(
                    self.editor_mode == EditorPlayMode::Editing,
                    egui::Button::new(egui::RichText::new("\u{25b6} Play").color(play_color)),
                ).clicked() {
                    self.play_requested = true;
                    self.editor_mode = EditorPlayMode::Playing;
                }

                // Pause button — only enabled when Playing or Paused
                let pause_label = if self.editor_mode == EditorPlayMode::Paused { "\u{25b6}\u{25b6} Resume" } else { "\u{23f8} Pause" };
                if ui.add_enabled(
                    self.editor_mode != EditorPlayMode::Editing,
                    egui::Button::new(pause_label),
                ).clicked() {
                    self.pause_requested = true;
                    self.editor_mode = if self.editor_mode == EditorPlayMode::Playing {
                        EditorPlayMode::Paused
                    } else {
                        EditorPlayMode::Playing
                    };
                }

                // Stop button — only enabled when Playing or Paused
                if ui.add_enabled(
                    self.editor_mode != EditorPlayMode::Editing,
                    egui::Button::new(egui::RichText::new("\u{23f9} Stop").color(egui::Color32::from_rgb(200, 80, 80))),
                ).clicked() {
                    self.stop_requested = true;
                    self.editor_mode = EditorPlayMode::Editing;
                }

                ui.separator();
                egui::ComboBox::from_id_salt("viewport_mode")
                    .selected_text(self.viewport_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [
                            ViewportMode::Lit,
                            ViewportMode::Unlit,
                            ViewportMode::Wireframe,
                            ViewportMode::Normals,
                            ViewportMode::Overdraw,
                        ] {
                            ui.selectable_value(&mut self.viewport_mode, mode, mode.label());
                        }
                    });
                egui::ComboBox::from_id_salt("viewport_layout")
                    .selected_text(self.viewport_layout.label())
                    .show_ui(ui, |ui| {
                        for layout in [
                            ViewportLayout::Single,
                            ViewportLayout::HSplit,
                            ViewportLayout::VSplit,
                            ViewportLayout::Quad,
                        ] {
                            ui.selectable_value(&mut self.viewport_layout, layout, layout.label());
                        }
                    });
                ui.separator();
                ui.label(format!("{} entities", self.entities.len()));
                if let Some(id) = self.selected {
                    ui.label(format!("| Selected: {}", id));
                }
            });
        });

        // Status bar (bottom)
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(self.status_text())
                        .size(11.0)
                        .color(egui::Color32::from_rgb(160, 165, 180)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(match self.editor_mode {
                            EditorPlayMode::Editing => "● EDITING",
                            EditorPlayMode::Playing => "▶ PLAYING",
                            EditorPlayMode::Paused  => "⏸ PAUSED",
                        })
                        .size(11.0)
                        .color(match self.editor_mode {
                            EditorPlayMode::Editing => egui::Color32::from_rgb(100, 150, 200),
                            EditorPlayMode::Playing => egui::Color32::from_rgb(80, 200, 80),
                            EditorPlayMode::Paused  => egui::Color32::from_rgb(220, 180, 60),
                        }),
                    );
                    if let Some(e) = self.selected_entity() {
                        ui.label(
                            egui::RichText::new(format!("Selected: {}", e.name))
                                .size(11.0)
                                .color(egui::Color32::from_rgb(140, 180, 140)),
                        );
                    }
                });
            });
        });

        // Scene hierarchy panel (left)
        egui::SidePanel::left("scene_hierarchy")
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Scene");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("+ Add").clicked() {
                        self.add_entity("New Entity", "default.ply", Vec3::ZERO);
                    }
                    if ui.button("Delete").clicked() {
                        self.delete_selected();
                    }
                    if ui.button("Duplicate").clicked() {
                        self.duplicate_selected();
                    }
                });

                ui.separator();

                // Collect to avoid borrow conflict with mut methods in context_menu
                let entity_items: Vec<(u32, String, bool)> = self.entities.iter().map(|e| {
                    let is_sel = self.selected == Some(e.id);
                    let lbl = if e.visible {
                        format!("\u{1f441} {}", e.name)
                    } else {
                        format!("  {}", e.name)
                    };
                    (e.id, lbl, is_sel)
                }).collect();

                for (entity_id, label, is_selected) in entity_items {
                    let response = ui.selectable_label(is_selected, &label);
                    if response.clicked() {
                        self.selected = Some(entity_id);
                    }
                    response.context_menu(|ui| {
                        if ui.button("Select").clicked() {
                            self.selected = Some(entity_id);
                            ui.close_menu();
                        }
                        if ui.button("Duplicate").clicked() {
                            self.selected = Some(entity_id);
                            self.duplicate_selected();
                            ui.close_menu();
                        }
                        if ui.button("Delete").clicked() {
                            self.selected = Some(entity_id);
                            self.delete_selected();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Focus Camera").clicked() {
                            self.focus_camera_on = Some(entity_id);
                            ui.close_menu();
                        }
                    });
                }
            });

        // Property inspector (right)
        egui::SidePanel::right("inspector")
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Inspector");
                ui.separator();

                if let Some(id) = self.selected {
                    if let Some(entity) = self.entities.iter_mut().find(|e| e.id == id) {
                        ui.label(format!("ID: {}", entity.id));
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.text_edit_singleline(&mut entity.name);
                        });
                        ui.horizontal(|ui| {
                            ui.label("Asset:");
                            ui.text_edit_singleline(&mut entity.asset_path);
                        });

                        ui.separator();
                        ui.label("Transform");

                        ui.horizontal(|ui| {
                            ui.label("X:");
                            ui.add(egui::DragValue::new(&mut entity.position.x).speed(0.1));
                            ui.label("Y:");
                            ui.add(egui::DragValue::new(&mut entity.position.y).speed(0.1));
                            ui.label("Z:");
                            ui.add(egui::DragValue::new(&mut entity.position.z).speed(0.1));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Scale:");
                            ui.add(egui::DragValue::new(&mut entity.scale.x).speed(0.01));
                            ui.add(egui::DragValue::new(&mut entity.scale.y).speed(0.01));
                            ui.add(egui::DragValue::new(&mut entity.scale.z).speed(0.01));
                        });

                        ui.separator();
                        ui.checkbox(&mut entity.visible, "Visible");
                        ui.checkbox(&mut entity.locked, "Locked");

                        ui.separator();
                        ui.label("Scripts");
                        for script in &entity.scripts {
                            ui.label(format!("  - {}", script));
                        }
                    } else {
                        ui.label("Entity not found");
                    }
                } else {
                    ui.label("No entity selected");
                }
            });

        if self.show_terrain_editor {
            egui::Window::new("Terrain Editor")
                .resizable(true)
                .default_width(240.0)
                .open(&mut self.show_terrain_editor)
                .show(ctx, |ui| {
                    crate::terrain_editor::show_terrain_editor_panel(ui, &mut self.terrain_editor_state);
                });
        }

        if self.terrain_editor_state.foliage_scatter_pending {
            self.terrain_editor_state.foliage_scatter_pending = false;
            println!("[ochroma-terrain] Foliage scatter requested (density={})", self.terrain_editor_state.foliage_density);
        }

        if self.show_history {
            egui::Window::new("History")
                .default_size([220.0, 300.0])
                .show(ctx, |ui| {
                    ui.label(egui::RichText::new("Undo Stack").strong());
                    egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        for action in self.undo_stack.iter().rev() {
                            ui.label(action.label());
                        }
                        if self.undo_stack.is_empty() {
                            ui.label(egui::RichText::new("(empty)").italics());
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Undo  Ctrl+Z").clicked() { self.undo(); }
                        if ui.button("Redo  Ctrl+Y").clicked() { self.redo(); }
                    });
                });
        }

        // Notification toasts (top-right corner)
        self.notification_queue.show(ctx);

        // Mini map
        let camera_pos = glam::Vec3::ZERO; // placeholder; caller can set via mini_map directly
        let entities: Vec<crate::minimap::MiniMapEntity> = self.entities.iter().map(|e| {
            crate::minimap::MiniMapEntity {
                position: e.position,
                color: egui::Color32::WHITE,
            }
        }).collect();
        self.mini_map.show(ctx, &entities, camera_pos);

        // Settings panel
        if self.show_settings {
            let changed = crate::settings::show_settings_panel(ctx, &mut self.app_settings, &mut self.show_settings);
            if changed {
                let _ = crate::settings::save_settings(&self.app_settings, std::path::Path::new("settings.toml"));
            }
        }
    }

    /// Cast a ray from camera and find the nearest entity hit.
    pub fn pick_entity_at_screen_pos(
        &self,
        screen_x: f32,
        screen_y: f32,
        screen_width: u32,
        screen_height: u32,
        inv_view_proj: glam::Mat4,
    ) -> Option<u32> {
        // Convert screen coords to NDC
        let ndc_x = (2.0 * screen_x / screen_width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / screen_height as f32);

        // Unproject to get ray using two NDC depths
        // Use Vec4 with manual perspective divide for robustness across all projection conventions.
        let unproject = |ndc_z: f32| -> glam::Vec3 {
            let clip = glam::Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
            let world = inv_view_proj * clip;
            glam::Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w)
        };
        let near = unproject(-1.0);
        let far = unproject(1.0);
        let ray_dir = (far - near).normalize();
        let ray_origin = near;

        // Test against all entity bounding spheres
        let mut best_hit: Option<(u32, f32)> = None;

        for entity in &self.entities {
            if !entity.visible {
                continue;
            }

            let entity_pos = entity.position;
            let radius = entity.scale.max_element() * 5.0; // approximate bounding sphere

            // Ray-sphere intersection
            let oc = ray_origin - entity_pos;
            let a = ray_dir.dot(ray_dir);
            let b = 2.0 * oc.dot(ray_dir);
            let c = oc.dot(oc) - radius * radius;
            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let t1 = (-b - sqrt_disc) / (2.0 * a);
                let t2 = (-b + sqrt_disc) / (2.0 * a);
                // Use the nearest positive intersection
                let t = if t1 > 0.0 { t1 } else { t2 };
                if t > 0.0
                    && best_hit.as_ref().is_none_or(|&(_, best_t)| t < best_t)
                {
                    best_hit = Some((entity.id, t));
                }
            }
        }

        best_hit.map(|(id, _)| id)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get the name of the currently selected entity (if any).
    pub fn selected_name(&self) -> Option<&str> {
        self.selected_entity().map(|e| e.name.as_str())
    }

    /// Print the entity list to the console (for when egui is not available).
    pub fn show_console(&self) {
        if !self.visible {
            return;
        }
        println!("[editor] === Scene Hierarchy ({} entities) ===", self.entities.len());
        for entity in &self.entities {
            let sel = if self.selected == Some(entity.id) { " <-- SELECTED" } else { "" };
            let vis = if entity.visible { "V" } else { " " };
            let lock = if entity.locked { "L" } else { " " };
            println!(
                "[editor]  [{}][{}] #{}: {} @ ({:.1}, {:.1}, {:.1}) [{}]{}",
                vis, lock, entity.id, entity.name,
                entity.position.x, entity.position.y, entity.position.z,
                entity.asset_path, sel,
            );
        }
        println!("[editor] ================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_select_entity() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("House", "house.ply", Vec3::new(10.0, 0.0, 5.0));
        editor.select(id);
        assert_eq!(editor.selected, Some(id));
        assert_eq!(editor.selected_entity().unwrap().name, "House");
    }

    #[test]
    fn delete_selected() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Tree", "tree.ply", Vec3::ZERO);
        editor.select(id);
        editor.delete_selected();
        assert_eq!(editor.entity_count(), 0);
        assert_eq!(editor.selected, None);
    }

    #[test]
    fn move_entity_with_snap() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Box", "box.ply", Vec3::ZERO);
        editor.select(id);
        editor.snap_enabled = true;
        editor.snap_grid = 2.0;
        editor.move_selected(Vec3::new(3.3, 0.0, 1.7));
        let pos = editor.selected_entity().unwrap().position;
        assert_eq!(pos.x, 4.0); // snapped to nearest 2.0
        assert_eq!(pos.z, 2.0);
    }

    #[test]
    fn undo_move() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Box", "box.ply", Vec3::ZERO);
        editor.select(id);
        editor.move_selected(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(editor.selected_entity().unwrap().position.x, 10.0);
        editor.undo();
        assert_eq!(editor.selected_entity().unwrap().position.x, 0.0);
    }

    #[test]
    fn undo_redo() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Box", "box.ply", Vec3::ZERO);
        editor.select(id);
        editor.move_selected(Vec3::new(5.0, 0.0, 0.0));
        editor.undo();
        assert_eq!(editor.selected_entity().unwrap().position.x, 0.0);
        editor.redo();
        assert_eq!(editor.selected_entity().unwrap().position.x, 5.0);
    }

    #[test]
    fn duplicate_entity() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Original", "thing.ply", Vec3::new(1.0, 2.0, 3.0));
        editor.select(id);
        let dup_id = editor.duplicate_selected().unwrap();
        assert_eq!(editor.entity_count(), 2);
        let dup = editor.entities.iter().find(|e| e.id == dup_id).unwrap();
        assert!(dup.name.contains("copy"));
    }

    #[test]
    fn export_import_map() {
        let mut editor = SceneEditor::new();
        editor.add_entity("House", "house.ply", Vec3::new(10.0, 0.0, 5.0));
        editor.add_entity("Tree", "tree.ply", Vec3::new(20.0, 0.0, -3.0));

        let map = editor.export_to_map("Test Map");
        assert_eq!(map.object_count(), 2);

        let mut editor2 = SceneEditor::new();
        editor2.import_from_map(&map);
        assert_eq!(editor2.entity_count(), 2);
        assert_eq!(editor2.entities[0].name, "House");
    }

    #[test]
    fn pick_entity_with_ray() {
        use glam::Mat4;

        let mut editor = SceneEditor::new();
        editor.add_entity("Near", "near.ply", Vec3::new(0.0, 0.0, -5.0));
        editor.add_entity("Far", "far.ply", Vec3::new(0.0, 0.0, -20.0));

        // Camera at origin looking down -Z
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let inv_vp = (proj * view).inverse();

        // Click center of screen -- should hit the near entity
        let hit = editor.pick_entity_at_screen_pos(128.0, 128.0, 256, 256, inv_vp);
        assert!(hit.is_some(), "Should hit an entity");
        // Near entity should be picked (closest)
        let picked = editor.entities.iter().find(|e| e.id == hit.unwrap()).unwrap();
        assert_eq!(picked.name, "Near", "Should pick the nearest entity");
    }

    #[test]
    fn menu_actions_default_none() {
        let editor = SceneEditor::new();
        // All menu action flags start false
        assert!(!editor.pending_new_scene);
        assert!(!editor.pending_open);
        assert!(!editor.pending_save);
        assert!(!editor.pending_save_as);
        assert!(!editor.show_history);
        assert!(!editor.show_material_editor);
        assert!(!editor.show_anim_editor);
        assert!(!editor.show_vfx_editor);
        assert!(!editor.show_perf_stats);
        assert!(!editor.pending_exit);
    }

    #[test]
    fn undo_redo_add_entity_round_trips() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("A", "a.ply", Vec3::ZERO);
        assert_eq!(editor.entity_count(), 1);
        editor.undo(); // undo the add
        assert_eq!(editor.entity_count(), 0);
        editor.redo(); // redo the add
        assert_eq!(editor.entity_count(), 1, "entity should be restored after redo");
        assert!(editor.entities.iter().any(|e| e.id == id));
    }

    #[test]
    fn locked_entity_cannot_move() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Locked", "thing.ply", Vec3::ZERO);
        editor.select(id);
        editor.selected_entity_mut().unwrap().locked = true;
        editor.move_selected(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(editor.selected_entity().unwrap().position.x, 0.0);
    }

    #[test]
    fn status_text_shows_entity_count() {
        let mut editor = SceneEditor::new();
        editor.add_entity("A", "a.ply", Vec3::ZERO);
        editor.add_entity("B", "b.ply", Vec3::ZERO);
        editor.status_splat_count = 500;
        assert_eq!(editor.status_text(), "2 entities | 500 splats | Ready");
    }

    #[test]
    fn play_mode_flag_set() {
        let mut editor = SceneEditor::new();
        assert_eq!(editor.editor_mode, EditorPlayMode::Editing);
        assert!(!editor.play_requested);

        // Simulate clicking Play
        editor.play_requested = true;
        editor.editor_mode = EditorPlayMode::Playing;
        assert!(editor.play_requested);
        assert_eq!(editor.editor_mode, EditorPlayMode::Playing);

        // Simulate consuming the flag
        editor.play_requested = false;
        assert!(!editor.play_requested);
        assert_eq!(editor.editor_mode, EditorPlayMode::Playing);

        // Simulate clicking Stop
        editor.stop_requested = true;
        editor.editor_mode = EditorPlayMode::Editing;
        assert_eq!(editor.editor_mode, EditorPlayMode::Editing);
    }

    #[test]
    fn transform_space_defaults_to_world() {
        let editor = SceneEditor::new();
        assert_eq!(editor.transform_space, TransformSpace::World);
    }

    #[test]
    fn transform_space_toggle() {
        let mut editor = SceneEditor::new();
        assert_eq!(editor.transform_space, TransformSpace::World);
        editor.transform_space = TransformSpace::Local;
        assert_eq!(editor.transform_space, TransformSpace::Local);
        editor.transform_space = TransformSpace::World;
        assert_eq!(editor.transform_space, TransformSpace::World);
    }

    #[test]
    fn history_panel_label_for_action() {
        let action = EditorAction::MoveEntity {
            id: 3,
            old_pos: Vec3::ZERO,
            new_pos: Vec3::new(1.0, 0.0, 0.0),
        };
        assert_eq!(action.label(), "Move #3");
    }

    #[test]
    fn ctrl_z_undo_via_label() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("Box", "box.ply", Vec3::ZERO);
        editor.select(id);
        editor.move_selected(Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(editor.selected_entity().unwrap().position.x, 5.0);
        // Check label before undo
        assert!(editor.undo_stack.last().unwrap().label().contains("Move"));
        editor.undo();
        assert_eq!(editor.selected_entity().unwrap().position.x, 0.0);
    }

    #[test]
    fn context_menu_rename_tracked() {
        let mut editor = SceneEditor::new();
        let id = editor.add_entity("OldName", "asset.ply", Vec3::ZERO);
        editor.rename_entity(id, "NewName");
        assert_eq!(editor.entities[0].name, "NewName");
        assert!(matches!(editor.undo_stack.last(), Some(EditorAction::RenameEntity { .. })));
    }

    #[test]
    fn focus_camera_on_starts_none() {
        let editor = SceneEditor::new();
        assert!(editor.focus_camera_on.is_none());
    }

    #[test]
    fn viewport_mode_defaults_lit() {
        let editor = SceneEditor::new();
        assert_eq!(editor.viewport_mode, ViewportMode::Lit);
    }

    #[test]
    fn viewport_mode_all_variants_have_labels() {
        assert_eq!(ViewportMode::Lit.label(), "Lit");
        assert_eq!(ViewportMode::Wireframe.label(), "Wireframe");
        assert_eq!(ViewportMode::Overdraw.label(), "Overdraw");
    }

    #[test]
    fn viewport_layout_defaults_single() {
        let editor = SceneEditor::new();
        assert_eq!(editor.viewport_layout, ViewportLayout::Single);
    }

    #[test]
    fn viewport_layout_all_labels() {
        assert_eq!(ViewportLayout::Single.label(), "1 View");
        assert_eq!(ViewportLayout::Quad.label(), "4 Views");
    }
}
