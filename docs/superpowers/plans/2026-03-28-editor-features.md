# Editor Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete game engine editor UI covering menu bar, play controls, status bar, viewport modes, multi-viewport layout selector, context menus, drag & drop, undo/redo shortcuts, material editor, animation graph editor, VFX editor, and world/local space toggle.

**Architecture:** All editor UI is rendered via egui on top of the wgpu scene. `editor.rs` owns panel layout and entity state. New sub-editor windows (material, anim, vfx) each get their own file. `engine_runner.rs` wires keyboard shortcuts and passes state to render.

**Tech Stack:** egui 0.31, wgpu 24, Rust, `vox_app`, `vox_render`, `vox_core`

---

## File Map

**Modify:**
- `crates/vox_app/src/editor.rs` — Add menu bar, play toolbar, status bar, world/local toggle, context menus, history window, drag drop target, viewport mode
- `crates/vox_app/src/bin/engine_runner.rs` — Wire Ctrl+Z/Y, F5/F6/F7, EditorStateMachine, viewport mode to render
- `crates/vox_app/src/content_browser.rs` — Add drag source state

**Create:**
- `crates/vox_render/src/material_editor_ui.rs` — egui material node graph window
- `crates/vox_render/src/anim_editor_ui.rs` — egui animation state machine window
- `crates/vox_render/src/vfx_editor_ui.rs` — egui VFX asset inspector window

---

## Task 1: Menu Bar (File / Edit / View / Help)

**Files:**
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Write the failing test**

```rust
// In crates/vox_app/src/editor.rs, add to mod tests:
#[test]
fn menu_actions_default_none() {
    let editor = SceneEditor::new();
    assert!(editor.pending_new_scene == false);
    assert!(editor.pending_open == false);
    assert!(editor.pending_save == false);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd crates/vox_app && cargo test menu_actions_default_none 2>&1
```
Expected: compile error — fields don't exist yet.

- [ ] **Step 3: Add menu state fields to SceneEditor**

In `editor.rs`, add to the `SceneEditor` struct:
```rust
// Menu action flags — set during show(), consumed by engine_runner.rs
pub pending_new_scene: bool,
pub pending_open: bool,
pub pending_save: bool,
pub pending_save_as: bool,
pub show_history: bool,
pub show_material_editor: bool,
pub show_anim_editor: bool,
pub show_vfx_editor: bool,
pub show_perf_stats: bool,
```

Initialize all to `false` in `SceneEditor::new()`:
```rust
pending_new_scene: false,
pending_open: false,
pending_save: false,
pending_save_as: false,
show_history: false,
show_material_editor: false,
show_anim_editor: false,
show_vfx_editor: false,
show_perf_stats: false,
```

- [ ] **Step 4: Add menu bar panel in `show()`**

In `editor.rs` `show()`, add BEFORE the existing `TopBottomPanel::top("editor_toolbar")`:

```rust
// Menu bar — must be registered before toolbar and side panels
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
                std::process::exit(0);
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
            ui.separator();
            if ui.button("Performance Stats").clicked() {
                self.show_perf_stats = !self.show_perf_stats;
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
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
cd crates/vox_app && cargo test menu_actions_default_none 2>&1
```
Expected: PASS

- [ ] **Step 6: Build to confirm no compile errors**

```bash
cargo build 2>&1 | grep -E "^error"
```
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/editor.rs
git commit -m "feat(editor): add File/Edit/View/Help menu bar"
```

---

## Task 2: Play / Pause / Stop Buttons

**Files:**
- Modify: `crates/vox_app/src/editor.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Write failing test**

```rust
// Add to editor.rs tests:
#[test]
fn play_mode_flag_set() {
    let mut editor = SceneEditor::new();
    assert!(!editor.play_requested);
    assert!(!editor.pause_requested);
    assert!(!editor.stop_requested);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd crates/vox_app && cargo test play_mode_flag_set 2>&1
```
Expected: compile error.

- [ ] **Step 3: Add play state fields to SceneEditor**

Add to `SceneEditor` struct in `editor.rs`:
```rust
pub play_requested: bool,
pub pause_requested: bool,
pub stop_requested: bool,
pub editor_mode: EditorPlayMode,
```

Add enum above the struct:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorPlayMode {
    Editing,
    Playing,
    Paused,
}
```

Initialize in `new()`:
```rust
play_requested: false,
pause_requested: false,
stop_requested: false,
editor_mode: EditorPlayMode::Editing,
```

- [ ] **Step 4: Add play/pause/stop buttons to the toolbar**

In the existing `TopBottomPanel::top("editor_toolbar")` `show()`, add after the snap controls and before the entity count label:

```rust
ui.separator();

let (play_label, play_color) = match self.editor_mode {
    EditorPlayMode::Editing => ("▶ Play", egui::Color32::from_rgb(80, 200, 80)),
    EditorPlayMode::Playing => ("▶ Play", egui::Color32::from_rgb(80, 200, 80)),
    EditorPlayMode::Paused => ("▶ Play", egui::Color32::GRAY),
};
if ui.add(egui::Button::new(
    egui::RichText::new(play_label).color(play_color)
)).clicked() && self.editor_mode == EditorPlayMode::Editing {
    self.play_requested = true;
    self.editor_mode = EditorPlayMode::Playing;
}

let pause_label = if self.editor_mode == EditorPlayMode::Paused { "▶▶ Resume" } else { "⏸ Pause" };
if ui.add(egui::Button::new(pause_label))
    .clicked() && self.editor_mode != EditorPlayMode::Editing
{
    self.pause_requested = true;
    self.editor_mode = if self.editor_mode == EditorPlayMode::Playing {
        EditorPlayMode::Paused
    } else {
        EditorPlayMode::Playing
    };
}

if ui.add(egui::Button::new(
    egui::RichText::new("⏹ Stop").color(egui::Color32::from_rgb(200, 80, 80))
)).clicked() && self.editor_mode != EditorPlayMode::Editing {
    self.stop_requested = true;
    self.editor_mode = EditorPlayMode::Editing;
}
```

- [ ] **Step 5: Wire F5/F6/F7 in engine_runner.rs keyboard handler**

In `handle_keyboard_event()` in `engine_runner.rs`, add inside the `Pressed` match:

```rust
KeyCode::F5 => {
    if self.editor_visible {
        self.editor.play_requested = true;
        self.editor.editor_mode = vox_app::editor::EditorPlayMode::Playing;
    }
}
KeyCode::F6 => {
    if self.editor_visible && self.editor.editor_mode != vox_app::editor::EditorPlayMode::Editing {
        self.editor.pause_requested = true;
        self.editor.editor_mode = if self.editor.editor_mode == vox_app::editor::EditorPlayMode::Playing {
            vox_app::editor::EditorPlayMode::Paused
        } else {
            vox_app::editor::EditorPlayMode::Playing
        };
    }
}
KeyCode::F7 => {
    if self.editor_visible && self.editor.editor_mode != vox_app::editor::EditorPlayMode::Editing {
        self.editor.stop_requested = true;
        self.editor.editor_mode = vox_app::editor::EditorPlayMode::Editing;
    }
}
```

- [ ] **Step 6: Consume the flags in the game loop (engine_runner.rs `update_frame()`)**

In the main update path (wherever `self.engine.tick()` is called), add:

```rust
// Handle play/pause/stop from editor
if self.editor.play_requested {
    self.editor.play_requested = false;
    println!("[ochroma] PLAY");
}
if self.editor.pause_requested {
    self.editor.pause_requested = false;
    println!("[ochroma] PAUSE");
}
if self.editor.stop_requested {
    self.editor.stop_requested = false;
    println!("[ochroma] STOP");
}
```

- [ ] **Step 7: Run test**

```bash
cd crates/vox_app && cargo test play_mode_flag_set 2>&1
```
Expected: PASS

- [ ] **Step 8: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 9: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): play/pause/stop buttons and F5/F6/F7 shortcuts"
```

---

## Task 3: Status Bar

**Files:**
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Write failing test**

```rust
// Add to editor.rs tests:
#[test]
fn status_text_shows_entity_count() {
    let mut editor = SceneEditor::new();
    editor.add_entity("A", "a.ply", Vec3::ZERO);
    editor.add_entity("B", "b.ply", Vec3::ZERO);
    assert_eq!(editor.status_text(), "2 entities | 0 splats | Ready");
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test status_text_shows_entity_count 2>&1
```
Expected: compile error — `status_text()` doesn't exist.

- [ ] **Step 3: Add status fields and method**

Add to `SceneEditor` struct:
```rust
pub status_splat_count: usize,
pub status_message: String,
```

Initialize in `new()`:
```rust
status_splat_count: 0,
status_message: String::from("Ready"),
```

Add method to `SceneEditor`:
```rust
pub fn status_text(&self) -> String {
    format!(
        "{} entities | {} splats | {}",
        self.entities.len(),
        self.status_splat_count,
        self.status_message
    )
}
```

- [ ] **Step 4: Add bottom panel in `show()`**

In `editor.rs` `show()`, add AFTER all side panels and BEFORE (or after) the central content:

```rust
// Status bar — must be registered before CentralPanel
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
```

- [ ] **Step 5: Pass splat count from engine_runner.rs**

In `engine_runner.rs`, in the egui render block where `self.editor.show(ctx)` is called, add before it:
```rust
self.editor.status_splat_count = self.scene_splats.len();
```

- [ ] **Step 6: Run test**

```bash
cd crates/vox_app && cargo test status_text_shows_entity_count 2>&1
```
Expected: PASS

- [ ] **Step 7: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 8: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): status bar with entity count, splat count, mode indicator"
```

---

## Task 4: World / Local Space Toggle

**Files:**
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn transform_space_defaults_to_world() {
    let editor = SceneEditor::new();
    assert_eq!(editor.transform_space, TransformSpace::World);
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test transform_space_defaults_to_world 2>&1
```
Expected: compile error.

- [ ] **Step 3: Add TransformSpace enum and field**

Add near the top of `editor.rs` (after other enums):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformSpace {
    World,
    Local,
}
```

Add to `SceneEditor` struct:
```rust
pub transform_space: TransformSpace,
```

Initialize in `new()`:
```rust
transform_space: TransformSpace::World,
```

- [ ] **Step 4: Add toggle button to toolbar**

In `TopBottomPanel::top("editor_toolbar")` in `show()`, add after the gizmo mode buttons:
```rust
ui.separator();
if ui.selectable_label(
    self.transform_space == TransformSpace::World, "World"
).clicked() {
    self.transform_space = TransformSpace::World;
}
if ui.selectable_label(
    self.transform_space == TransformSpace::Local, "Local"
).clicked() {
    self.transform_space = TransformSpace::Local;
}
```

- [ ] **Step 5: Run test**

```bash
cd crates/vox_app && cargo test transform_space_defaults_to_world 2>&1
```
Expected: PASS

- [ ] **Step 6: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/editor.rs
git commit -m "feat(editor): World/Local space toggle in toolbar"
```

---

## Task 5: Undo / Redo Keyboard Shortcuts + History Panel

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Wire Ctrl+Z and Ctrl+Y in engine_runner.rs**

In `handle_keyboard_event()`, add to the `Pressed` match arms (the undo/redo logic already exists in `editor.rs`, just needs keyboard wiring):

```rust
KeyCode::KeyZ if self.ctrl_held => {
    if self.editor_visible {
        self.editor.undo();
        println!("[ochroma] Undo ({} actions remaining)", self.editor.undo_stack.len());
    }
}
KeyCode::KeyY if self.ctrl_held => {
    if self.editor_visible {
        self.editor.redo();
        println!("[ochroma] Redo ({} actions remaining)", self.editor.redo_stack.len());
    }
}
```

- [ ] **Step 2: Write failing test for history display**

```rust
// In editor.rs tests:
#[test]
fn history_panel_label_for_action() {
    let action = EditorAction::MoveEntity {
        id: 0,
        old_pos: Vec3::ZERO,
        new_pos: Vec3::new(1.0, 0.0, 0.0),
    };
    assert_eq!(action.label(), "Move entity #0");
}
```

- [ ] **Step 3: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test history_panel_label_for_action 2>&1
```
Expected: compile error — `label()` doesn't exist.

- [ ] **Step 4: Add `label()` method to EditorAction**

```rust
impl EditorAction {
    pub fn label(&self) -> String {
        match self {
            EditorAction::MoveEntity { id, .. }   => format!("Move entity #{}", id),
            EditorAction::RotateEntity { id, .. } => format!("Rotate entity #{}", id),
            EditorAction::ScaleEntity { id, .. }  => format!("Scale entity #{}", id),
            EditorAction::AddEntity { id }         => format!("Add entity #{}", id),
            EditorAction::DeleteEntity { id, .. } => format!("Delete entity #{}", id),
            EditorAction::RenameEntity { id, old_name, new_name } =>
                format!("Rename #{}: {} → {}", id, old_name, new_name),
        }
    }
}
```

- [ ] **Step 5: Add history window to `show()`**

Add at the bottom of `show()` (after all panels):
```rust
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
```

- [ ] **Step 6: Run test**

```bash
cd crates/vox_app && cargo test history_panel_label_for_action 2>&1
```
Expected: PASS

- [ ] **Step 7: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 8: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): Ctrl+Z/Y undo-redo shortcuts and history panel"
```

---

## Task 6: Context Menus on Entities

**Files:**
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Write failing test**

```rust
// In editor.rs tests:
#[test]
fn context_menu_rename_tracked() {
    let mut editor = SceneEditor::new();
    let id = editor.add_entity("OldName", "asset.ply", Vec3::ZERO);
    editor.rename_entity(id, "NewName");
    assert_eq!(editor.entities[0].name, "NewName");
    // Undo stack should have the rename
    assert!(matches!(editor.undo_stack.last(), Some(EditorAction::RenameEntity { .. })));
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test context_menu_rename_tracked 2>&1
```
Expected: compile error — `rename_entity()` doesn't exist.

- [ ] **Step 3: Add `rename_entity()` method**

```rust
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
```

- [ ] **Step 4: Add context menus to entity rows in the hierarchy panel**

In `show()`, replace the selectable label in the hierarchy loop:

```rust
// Replace:
//   if ui.selectable_label(is_selected, &label).clicked() {
//       self.selected = Some(entity.id);
//   }
// With:
let response = ui.selectable_label(is_selected, &label);
if response.clicked() {
    self.selected = Some(entity.id);
}
let entity_id = entity.id;
response.context_menu(|ui| {
    if ui.button("Select").clicked() {
        self.selected = Some(entity_id);
        ui.close_menu();
    }
    if ui.button("Duplicate").clicked() {
        // Can't call self.duplicate_selected() here (borrow), defer via flag
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
        // Flag for engine_runner to act on
        self.focus_camera_on = Some(entity_id);
        ui.close_menu();
    }
});
```

- [ ] **Step 5: Add `focus_camera_on` field to SceneEditor**

```rust
// In struct:
pub focus_camera_on: Option<u32>,
// In new():
focus_camera_on: None,
```

- [ ] **Step 6: Consume `focus_camera_on` in engine_runner.rs**

In the egui run block, after `self.editor.show(ctx)`:
```rust
if let Some(id) = self.editor.focus_camera_on.take() {
    if let Some(entity) = self.editor.entities.iter().find(|e| e.id == id) {
        // Move camera to look at entity
        let target = entity.position;
        self.camera.position = target + glam::Vec3::new(0.0, 5.0, 10.0);
        self.cam_yaw = 0.0;
        self.cam_pitch = -0.2;
        println!("[ochroma] Camera focused on entity #{}", id);
    }
}
```

- [ ] **Step 7: Run test**

```bash
cd crates/vox_app && cargo test context_menu_rename_tracked 2>&1
```
Expected: PASS

- [ ] **Step 8: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 9: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): right-click context menus on entities with focus camera"
```

---

## Task 7: Viewport Modes (Lit / Unlit / Wireframe / Normals)

**Files:**
- Modify: `crates/vox_app/src/editor.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Write failing test**

```rust
// In editor.rs tests:
#[test]
fn viewport_mode_defaults_lit() {
    let editor = SceneEditor::new();
    assert_eq!(editor.viewport_mode, ViewportMode::Lit);
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test viewport_mode_defaults_lit 2>&1
```
Expected: compile error.

- [ ] **Step 3: Add ViewportMode enum and field**

```rust
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
```

Add to `SceneEditor` struct:
```rust
pub viewport_mode: ViewportMode,
```

Initialize in `new()`:
```rust
viewport_mode: ViewportMode::Lit,
```

- [ ] **Step 4: Add viewport mode dropdown to toolbar**

In the toolbar panel in `show()`, add:
```rust
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
```

- [ ] **Step 5: Make render_frame() in engine_runner.rs respect viewport mode**

In `render_frame()`, after splat rendering, find where HUD colors are applied. Add a viewport mode tint pass:

```rust
// Viewport mode post-process
match self.editor.viewport_mode {
    ViewportMode::Lit => {} // no-op
    ViewportMode::Unlit => {
        // Desaturate lighting: average RGB channels to remove light contribution
        for pixel in final_pixels.iter_mut() {
            let avg = ((pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32) / 3) as u8;
            *pixel = [avg, avg, avg, pixel[3]];
        }
    }
    ViewportMode::Wireframe => {
        // Tint dark blue for wireframe hint (actual wireframe needs GPU splat outlines)
        for pixel in final_pixels.iter_mut() {
            pixel[0] = (pixel[0] as f32 * 0.2) as u8;
            pixel[1] = (pixel[1] as f32 * 0.3) as u8;
            pixel[2] = (pixel[2] as f32 * 0.6 + 80.0) as u8;
        }
    }
    ViewportMode::Normals => {
        // Remap to normals-style pastel
        for pixel in final_pixels.iter_mut() {
            pixel[0] = 128 + pixel[0] / 2;
            pixel[1] = 128 + pixel[1] / 2;
            pixel[2] = 200;
        }
    }
    ViewportMode::Overdraw => {
        // Heat map: brighter = more overdraw (approximate with brightness)
        for pixel in final_pixels.iter_mut() {
            let bright = pixel[0].max(pixel[1]).max(pixel[2]);
            let heat = (bright as f32 / 255.0 * 4.0).min(1.0);
            pixel[0] = (heat * 255.0) as u8;
            pixel[1] = ((1.0 - heat) * 100.0) as u8;
            pixel[2] = 0;
            pixel[3] = 255;
        }
    }
}
```

Apply this only when `self.editor_visible` to avoid affecting game mode.

- [ ] **Step 6: Run test**

```bash
cd crates/vox_app && cargo test viewport_mode_defaults_lit 2>&1
```
Expected: PASS

- [ ] **Step 7: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 8: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): viewport modes — Lit/Unlit/Wireframe/Normals/Overdraw"
```

---

## Task 8: Drag & Drop from Content Browser to Scene

**Files:**
- Modify: `crates/vox_app/src/content_browser.rs`
- Modify: `crates/vox_app/src/editor.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Write failing test**

```rust
// In content_browser.rs tests:
#[test]
fn drag_state_starts_none() {
    let browser = ContentBrowser::new(std::path::Path::new("."));
    assert!(browser.dragging_asset.is_none());
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test drag_state_starts_none 2>&1
```
Expected: compile error.

- [ ] **Step 3: Add drag state to ContentBrowser**

Add to `ContentBrowser` struct:
```rust
pub dragging_asset: Option<String>, // asset path being dragged
```

Initialize in `new()`:
```rust
dragging_asset: None,
```

- [ ] **Step 4: Add drag source to content browser `show()`**

In the `ContentBrowser::show()` method, find where individual entry rows are rendered. Replace the plain `ui.label()` for each entry with:

```rust
let response = ui.add(egui::Label::new(
    format!("{} {} ({})", entry.entry_type.label(), entry.name, format_size(entry.size_bytes))
).sense(egui::Sense::click_and_drag()));

if response.drag_started() {
    self.dragging_asset = Some(entry.path.to_string_lossy().to_string());
}
if response.drag_stopped() {
    // dragging_asset is consumed by the drop target; clear if not consumed
    if self.dragging_asset.is_some() {
        self.dragging_asset = None;
    }
}
```

- [ ] **Step 5: Add drop zone to scene hierarchy in editor.rs**

In the hierarchy panel in `show()`, after the entity list, add:

```rust
// Drop zone at the bottom of the hierarchy
let drop_zone = ui.allocate_rect(
    ui.available_rect_before_wrap(),
    egui::Sense::hover(),
);
if drop_zone.hovered() {
    ui.painter().rect_filled(
        drop_zone.rect,
        0.0,
        egui::Color32::from_rgba_premultiplied(60, 120, 200, 40),
    );
    ui.label(egui::RichText::new("Drop asset here").color(egui::Color32::from_rgb(100, 160, 220)));
}
// The actual drop is handled by engine_runner after show() returns
```

Add to `SceneEditor` struct:
```rust
pub drop_pending_asset: Option<String>,
```
Initialize: `drop_pending_asset: None`

- [ ] **Step 6: Consume drop in engine_runner.rs**

After `self.editor.show(ctx)` and `self.content_browser.show(ctx)`:
```rust
// Handle drag-and-drop from content browser into scene
if let Some(asset_path) = self.content_browser.dragging_asset.take() {
    // If mouse is over the hierarchy area, place at origin; otherwise place in front of camera
    let forward = self.camera_forward();
    let pos = self.camera.position + forward * 10.0;
    let name = std::path::Path::new(&asset_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Dropped Asset")
        .to_string();
    let _id = self.editor.add_entity(&name, &asset_path, pos);
    println!("[ochroma] Dropped asset: {} at {:?}", asset_path, pos);
}
```

- [ ] **Step 7: Run test**

```bash
cd crates/vox_app && cargo test drag_state_starts_none 2>&1
```
Expected: PASS

- [ ] **Step 8: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 9: Commit**

```bash
git add crates/vox_app/src/content_browser.rs crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): drag asset from content browser to drop into scene"
```

---

## Task 9: Multiple Viewport Layout Selector

**Files:**
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn viewport_layout_defaults_single() {
    let editor = SceneEditor::new();
    assert_eq!(editor.viewport_layout, ViewportLayout::Single);
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_app && cargo test viewport_layout_defaults_single 2>&1
```
Expected: compile error.

- [ ] **Step 3: Add ViewportLayout enum and field**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportLayout {
    Single,
    HSplit,  // left/right
    VSplit,  // top/bottom
    Quad,    // 4-way split
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
```

Add to `SceneEditor` struct:
```rust
pub viewport_layout: ViewportLayout,
```
Initialize: `viewport_layout: ViewportLayout::Single`

- [ ] **Step 4: Add layout selector to toolbar**

In the toolbar `show()`:
```rust
ui.separator();
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
```

Note: Multi-viewport rendering (actually rendering multiple camera views) requires significant work in the render pipeline. This task implements the layout *selector* UI and enum. Actual multi-camera rendering is a separate render pipeline task.

- [ ] **Step 5: Run test**

```bash
cd crates/vox_app && cargo test viewport_layout_defaults_single 2>&1
```
Expected: PASS

- [ ] **Step 6: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/editor.rs
git commit -m "feat(editor): viewport layout selector (Single/2H/2V/Quad)"
```

---

## Task 10: Material Editor UI

**Files:**
- Create: `crates/vox_render/src/material_editor_ui.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 1: Write failing test**

```rust
// Create crates/vox_render/src/material_editor_ui.rs with:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_ui_default_open_false() {
        let ui = MaterialEditorUi::new();
        assert!(!ui.open);
        assert!(ui.graph.is_none());
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_render && cargo test material_ui_default_open_false 2>&1
```
Expected: file not found / compile error.

- [ ] **Step 3: Create `material_editor_ui.rs`**

```rust
//! egui window for the material node graph editor.

use crate::material_editor::{MaterialGraph, MaterialEditorNode, MaterialNodeType, MaterialConnection};

pub struct MaterialEditorUi {
    pub open: bool,
    pub graph: Option<MaterialGraph>,
    selected_node: Option<u32>,
    scroll_offset: egui::Vec2,
    zoom: f32,
}

impl MaterialEditorUi {
    pub fn new() -> Self {
        Self {
            open: false,
            graph: None,
            selected_node: None,
            scroll_offset: egui::Vec2::ZERO,
            zoom: 1.0,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        egui::Window::new("Material Editor")
            .open(&mut self.open)
            .default_size([800.0, 500.0])
            .resizable(true)
            .show(ctx, |ui| {
                if self.graph.is_none() {
                    ui.label("No material loaded. Open a material from the Content Browser.");
                    if ui.button("New Material").clicked() {
                        self.graph = Some(MaterialGraph::new("New Material"));
                    }
                    return;
                }

                let graph = self.graph.as_mut().unwrap();

                ui.horizontal(|ui| {
                    ui.heading(&graph.name);
                    ui.separator();
                    if ui.button("Save").clicked() {
                        // Flag for engine to save — could serialize to JSON
                    }
                    if ui.button("Add Node ▾").clicked() {}
                    ui.label(format!("{} nodes  {} connections", graph.nodes.len(), graph.connections.len()));
                });

                ui.separator();

                // Split: node list on left, canvas in center, properties on right
                egui::SidePanel::left("mat_node_list")
                    .default_width(160.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("Nodes").strong());
                        for node in &graph.nodes {
                            let label = format!("#{} {}", node.id, node_type_label(&node.node_type));
                            let selected = self.selected_node == Some(node.id);
                            if ui.selectable_label(selected, label).clicked() {
                                self.selected_node = Some(node.id);
                            }
                        }
                    });

                egui::SidePanel::right("mat_properties")
                    .default_width(200.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("Properties").strong());
                        if let Some(sel_id) = self.selected_node {
                            if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == sel_id) {
                                ui.label(format!("Node: #{}", node.id));
                                ui.label(node_type_label(&node.node_type));
                                ui.separator();
                                node_properties_ui(ui, &mut node.node_type);
                            }
                        } else {
                            ui.label("Select a node to edit properties.");
                        }
                    });

                // Central canvas — draw nodes as boxes with connections
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    let painter = ui.painter();
                    let canvas_origin = ui.min_rect().min;

                    // Draw connection lines first
                    for conn in &graph.connections {
                        if let (Some(from_node), Some(to_node)) = (
                            graph.nodes.iter().find(|n| n.id == conn.from_node),
                            graph.nodes.iter().find(|n| n.id == conn.to_node),
                        ) {
                            let from_pos = egui::pos2(
                                canvas_origin.x + from_node.position[0] * self.zoom + self.scroll_offset.x + 100.0,
                                canvas_origin.y + from_node.position[1] * self.zoom + self.scroll_offset.y + 20.0,
                            );
                            let to_pos = egui::pos2(
                                canvas_origin.x + to_node.position[0] * self.zoom + self.scroll_offset.x,
                                canvas_origin.y + to_node.position[1] * self.zoom + self.scroll_offset.y + 20.0,
                            );
                            painter.line_segment(
                                [from_pos, to_pos],
                                egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 140, 200)),
                            );
                        }
                    }

                    // Draw nodes as boxes
                    for node in &graph.nodes {
                        let x = canvas_origin.x + node.position[0] * self.zoom + self.scroll_offset.x;
                        let y = canvas_origin.y + node.position[1] * self.zoom + self.scroll_offset.y;
                        let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(100.0, 40.0));
                        let is_selected = self.selected_node == Some(node.id);
                        let fill = if is_selected {
                            egui::Color32::from_rgb(40, 70, 120)
                        } else {
                            egui::Color32::from_rgb(30, 35, 50)
                        };
                        painter.rect_filled(rect, 4.0, fill);
                        painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 90, 130)));
                        painter.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            node_type_label(&node.node_type),
                            egui::FontId::proportional(12.0),
                            egui::Color32::WHITE,
                        );
                    }
                });
            });
    }
}

fn node_type_label(t: &MaterialNodeType) -> &'static str {
    match t {
        MaterialNodeType::MaterialOutput        => "Output",
        MaterialNodeType::SpectralConstant {..} => "Spectral Const",
        MaterialNodeType::FloatConstant {..}    => "Float",
        MaterialNodeType::ColorConstant {..}    => "Color",
        MaterialNodeType::TextureCoordinate     => "TexCoord",
        MaterialNodeType::TextureSample {..}    => "Texture",
        MaterialNodeType::Add                   => "Add",
        MaterialNodeType::Subtract              => "Subtract",
        MaterialNodeType::Multiply              => "Multiply",
        MaterialNodeType::Divide                => "Divide",
        MaterialNodeType::Lerp                  => "Lerp",
        MaterialNodeType::Power                 => "Power",
        MaterialNodeType::Sqrt                  => "Sqrt",
        MaterialNodeType::Abs                   => "Abs",
        MaterialNodeType::OneMinus              => "1 - x",
        MaterialNodeType::SpectralBlend {..}    => "Spectral Blend",
        MaterialNodeType::SpectralShift {..}    => "Spectral Shift",
        MaterialNodeType::WearBlend {..}        => "Wear",
        MaterialNodeType::FresnelEffect {..}    => "Fresnel",
        MaterialNodeType::Roughness {..}        => "Roughness",
        MaterialNodeType::Metallic {..}         => "Metallic",
        MaterialNodeType::Emission {..}         => "Emission",
        MaterialNodeType::Opacity {..}          => "Opacity",
        MaterialNodeType::PerlinNoise {..}      => "Perlin Noise",
        MaterialNodeType::VoronoiNoise {..}     => "Voronoi",
        MaterialNodeType::Checker {..}          => "Checker",
        MaterialNodeType::Gradient {..}         => "Gradient",
        _                                       => "Node",
    }
}

fn node_properties_ui(ui: &mut egui::Ui, node_type: &mut MaterialNodeType) {
    match node_type {
        MaterialNodeType::FloatConstant { value } => {
            ui.horizontal(|ui| {
                ui.label("Value:");
                ui.add(egui::DragValue::new(value).speed(0.01));
            });
        }
        MaterialNodeType::ColorConstant { r, g, b } => {
            ui.horizontal(|ui| {
                ui.label("R:"); ui.add(egui::DragValue::new(r).speed(0.01).range(0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("G:"); ui.add(egui::DragValue::new(g).speed(0.01).range(0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("B:"); ui.add(egui::DragValue::new(b).speed(0.01).range(0.0..=1.0));
            });
        }
        MaterialNodeType::Roughness { value } => {
            ui.horizontal(|ui| {
                ui.label("Roughness:");
                ui.add(egui::DragValue::new(value).speed(0.01).range(0.0..=1.0));
            });
        }
        MaterialNodeType::Emission { intensity } => {
            ui.horizontal(|ui| {
                ui.label("Intensity:");
                ui.add(egui::DragValue::new(intensity).speed(0.1));
            });
        }
        MaterialNodeType::SpectralBlend { factor } => {
            ui.horizontal(|ui| {
                ui.label("Blend:");
                ui.add(egui::DragValue::new(factor).speed(0.01).range(0.0..=1.0));
            });
        }
        _ => {
            ui.label("No editable properties.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_ui_default_open_false() {
        let ui = MaterialEditorUi::new();
        assert!(!ui.open);
        assert!(ui.graph.is_none());
    }
}
```

- [ ] **Step 4: Export from `crates/vox_render/src/lib.rs`**

Add to lib.rs:
```rust
pub mod material_editor_ui;
```

- [ ] **Step 5: Wire into engine_runner.rs**

Add field to `EngineApp`:
```rust
material_editor_ui: vox_render::material_editor_ui::MaterialEditorUi,
```
Initialize in `new()`:
```rust
material_editor_ui: vox_render::material_editor_ui::MaterialEditorUi::new(),
```

In the egui run block:
```rust
// Sync open flag from editor menu
self.material_editor_ui.open = self.editor.show_material_editor;
self.material_editor_ui.show(ctx);
self.editor.show_material_editor = self.material_editor_ui.open;
```

- [ ] **Step 6: Run test**

```bash
cd crates/vox_render && cargo test material_ui_default_open_false 2>&1
```
Expected: PASS

- [ ] **Step 7: Build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 8: Commit**

```bash
git add crates/vox_render/src/material_editor_ui.rs crates/vox_render/src/lib.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): material node graph editor UI window"
```

---

## Task 11: Animation Graph Editor UI

**Files:**
- Create: `crates/vox_render/src/anim_editor_ui.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Write failing test**

```rust
// In anim_editor_ui.rs:
#[test]
fn anim_ui_default_closed() {
    let ui = AnimEditorUi::new();
    assert!(!ui.open);
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_render && cargo test anim_ui_default_closed 2>&1
```

- [ ] **Step 3: Create `anim_editor_ui.rs`**

```rust
//! egui window for the animation state machine editor.

use crate::anim_editor::{AnimGraphDefinition, AnimState, AnimTransition};

pub struct AnimEditorUi {
    pub open: bool,
    pub graph: Option<AnimGraphDefinition>,
    selected_state: Option<String>,
}

impl AnimEditorUi {
    pub fn new() -> Self {
        Self { open: false, graph: None, selected_state: None }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        egui::Window::new("Animation Editor")
            .open(&mut self.open)
            .default_size([700.0, 450.0])
            .resizable(true)
            .show(ctx, |ui| {
                if self.graph.is_none() {
                    ui.label("No animation graph loaded.");
                    return;
                }

                let graph = self.graph.as_ref().unwrap();

                ui.horizontal(|ui| {
                    ui.heading(&graph.name);
                    ui.separator();
                    ui.label(format!(
                        "{} states  {} transitions  {} params",
                        graph.states.len(), graph.transitions.len(), graph.parameters.len()
                    ));
                });

                ui.separator();

                egui::SidePanel::left("anim_states_list")
                    .default_width(180.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("States").strong());
                        for state in &graph.states {
                            let is_default = state.name == graph.default_state;
                            let is_selected = self.selected_state.as_deref() == Some(&state.name);
                            let label = if is_default {
                                format!("★ {}", state.name)
                            } else {
                                format!("  {}", state.name)
                            };
                            if ui.selectable_label(is_selected, &label).clicked() {
                                self.selected_state = Some(state.name.clone());
                            }
                        }
                        ui.separator();
                        ui.label(egui::RichText::new("Parameters").strong());
                        for param in &graph.parameters {
                            ui.label(format!("  {} ({:?})", param.name, param.param_type));
                        }
                    });

                egui::SidePanel::right("anim_state_props")
                    .default_width(200.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("Properties").strong());
                        if let Some(sel) = &self.selected_state {
                            if let Some(state) = graph.states.iter().find(|s| &s.name == sel) {
                                ui.label(format!("State: {}", state.name));
                                ui.label(format!("Clip: {}", state.clip_path));
                                ui.label(format!("Speed: {:.2}", state.speed));
                                ui.label(format!("Loop: {}", state.looping));
                                ui.separator();
                                let transitions: Vec<_> = graph.transitions.iter()
                                    .filter(|t| t.from == state.name)
                                    .collect();
                                ui.label(egui::RichText::new(format!("{} outgoing transitions:", transitions.len())).strong());
                                for t in transitions {
                                    ui.label(format!("  → {} ({:.2}s blend)", t.to, t.blend_duration));
                                }
                            }
                        }
                    });

                // Canvas: draw state machine as boxes and arrows
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    let painter = ui.painter();
                    let origin = ui.min_rect().min;
                    let graph = self.graph.as_ref().unwrap();

                    // Draw transition arrows first
                    for transition in &graph.transitions {
                        if let (Some(from), Some(to)) = (
                            graph.states.iter().find(|s| s.name == transition.from),
                            graph.states.iter().find(|s| s.name == transition.to),
                        ) {
                            let from_center = egui::pos2(
                                origin.x + from.position[0] + 75.0,
                                origin.y + from.position[1] + 25.0,
                            );
                            let to_center = egui::pos2(
                                origin.x + to.position[0] + 75.0,
                                origin.y + to.position[1] + 25.0,
                            );
                            painter.line_segment(
                                [from_center, to_center],
                                egui::Stroke::new(1.5, egui::Color32::from_rgb(150, 150, 200)),
                            );
                            // Arrowhead at midpoint direction
                            let dir = (to_center - from_center).normalized();
                            let mid = from_center + (to_center - from_center) * 0.6;
                            let perp = egui::vec2(-dir.y, dir.x) * 5.0;
                            painter.line_segment([mid, mid - dir * 8.0 + perp], egui::Stroke::new(1.5, egui::Color32::from_rgb(150, 150, 200)));
                            painter.line_segment([mid, mid - dir * 8.0 - perp], egui::Stroke::new(1.5, egui::Color32::from_rgb(150, 150, 200)));
                        }
                    }

                    // Draw state boxes
                    for state in &graph.states {
                        let x = origin.x + state.position[0];
                        let y = origin.y + state.position[1];
                        let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(150.0, 50.0));
                        let is_default = state.name == graph.default_state;
                        let is_selected = self.selected_state.as_deref() == Some(&state.name);
                        let fill = if is_selected {
                            egui::Color32::from_rgb(40, 80, 130)
                        } else if is_default {
                            egui::Color32::from_rgb(40, 80, 60)
                        } else {
                            egui::Color32::from_rgb(35, 40, 55)
                        };
                        painter.rect_filled(rect, 6.0, fill);
                        painter.rect_stroke(rect, 6.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 110, 160)));
                        painter.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            &state.name,
                            egui::FontId::proportional(13.0),
                            egui::Color32::WHITE,
                        );
                        if is_default {
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
    }
}
```

- [ ] **Step 4: Export from lib.rs**

```rust
pub mod anim_editor_ui;
```

- [ ] **Step 5: Wire into engine_runner.rs**

Add field to `EngineApp`:
```rust
anim_editor_ui: vox_render::anim_editor_ui::AnimEditorUi,
```
Initialize: `anim_editor_ui: vox_render::anim_editor_ui::AnimEditorUi::new()`

In egui block:
```rust
self.anim_editor_ui.open = self.editor.show_anim_editor;
self.anim_editor_ui.show(ctx);
self.editor.show_anim_editor = self.anim_editor_ui.open;
```

- [ ] **Step 6: Run test**

```bash
cd crates/vox_render && cargo test anim_ui_default_closed 2>&1
```
Expected: PASS

- [ ] **Step 7: Build and commit**

```bash
cargo build 2>&1 | grep "^error"
git add crates/vox_render/src/anim_editor_ui.rs crates/vox_render/src/lib.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): animation state machine editor UI window"
```

---

## Task 12: VFX Editor UI

**Files:**
- Create: `crates/vox_render/src/vfx_editor_ui.rs`
- Modify: `crates/vox_render/src/lib.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Write failing test**

```rust
// In vfx_editor_ui.rs:
#[test]
fn vfx_ui_default_closed() {
    let ui = VfxEditorUi::new();
    assert!(!ui.open);
    assert!(ui.selected_asset.is_none());
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd crates/vox_render && cargo test vfx_ui_default_closed 2>&1
```

- [ ] **Step 3: Create `vfx_editor_ui.rs`**

```rust
//! egui window for the VFX asset inspector.

use crate::vfx_editor::{VfxAsset, VfxCategory};
use crate::vfx::VfxEffect;

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

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }

        egui::Window::new("VFX Editor")
            .open(&mut self.open)
            .default_size([600.0, 400.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("VFX Library");
                    ui.separator();
                    ui.label(format!("{} effects", self.asset_library.len()));
                    if ui.button("+ New Effect").clicked() {
                        self.asset_library.push(VfxAsset {
                            name: format!("Effect_{}", self.asset_library.len()),
                            effect: VfxEffect::default(),
                            thumbnail_path: None,
                            category: VfxCategory::Custom,
                            description: String::new(),
                            tags: Vec::new(),
                            preview_camera_distance: 5.0,
                        });
                        self.selected_asset = Some(self.asset_library.len() - 1);
                    }
                });

                ui.separator();

                egui::SidePanel::left("vfx_library_list")
                    .default_width(200.0)
                    .show_inside(ui, |ui| {
                        ui.label(egui::RichText::new("Effects").strong());
                        for (i, asset) in self.asset_library.iter().enumerate() {
                            let selected = self.selected_asset == Some(i);
                            let label = format!("[{:?}] {}", asset.category, asset.name);
                            if ui.selectable_label(selected, &label).clicked() {
                                self.selected_asset = Some(i);
                            }
                        }
                        if self.asset_library.is_empty() {
                            ui.label(egui::RichText::new("No effects. Click '+ New Effect'.").italics());
                        }
                    });

                egui::CentralPanel::default().show_inside(ui, |ui| {
                    if let Some(idx) = self.selected_asset {
                        if let Some(asset) = self.asset_library.get_mut(idx) {
                            ui.heading(&asset.name);
                            ui.separator();

                            egui::Grid::new("vfx_props").num_columns(2).show(ui, |ui| {
                                ui.label("Name:");
                                ui.text_edit_singleline(&mut asset.name);
                                ui.end_row();

                                ui.label("Category:");
                                egui::ComboBox::from_id_salt("vfx_cat")
                                    .selected_text(format!("{:?}", asset.category))
                                    .show_ui(ui, |ui| {
                                        for cat in [VfxCategory::Fire, VfxCategory::Smoke,
                                                    VfxCategory::Explosion, VfxCategory::Weather,
                                                    VfxCategory::Magic, VfxCategory::Environment,
                                                    VfxCategory::UI, VfxCategory::Custom] {
                                            ui.selectable_value(&mut asset.category, cat, format!("{:?}", cat));
                                        }
                                    });
                                ui.end_row();

                                ui.label("Description:");
                                ui.text_edit_multiline(&mut asset.description);
                                ui.end_row();

                                ui.label("Preview Distance:");
                                ui.add(egui::DragValue::new(&mut asset.preview_camera_distance).speed(0.1).range(0.5..=50.0));
                                ui.end_row();
                            });

                            ui.separator();
                            ui.label(egui::RichText::new("Effect Parameters").strong());

                            egui::Grid::new("vfx_effect_params").num_columns(2).show(ui, |ui| {
                                ui.label("Max Particles:");
                                ui.label(format!("{}", asset.effect.max_particles));
                                ui.end_row();
                                ui.label("Emit Rate:");
                                ui.label(format!("{:.1}/s", asset.effect.emit_rate));
                                ui.end_row();
                                ui.label("Lifetime:");
                                ui.label(format!("{:.1}s", asset.effect.lifetime));
                                ui.end_row();
                                ui.label("Speed:");
                                ui.label(format!("{:.1} – {:.1}", asset.effect.speed_min, asset.effect.speed_max));
                                ui.end_row();
                            });
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("Select a VFX effect to inspect").italics());
                        });
                    }
                });
            });
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
```

- [ ] **Step 4: Export from lib.rs**

```rust
pub mod vfx_editor_ui;
```

- [ ] **Step 5: Check what fields `VfxEffect` has**

```bash
grep -n "pub " crates/vox_render/src/vfx.rs | head -30
```

Adjust field names in `vfx_editor_ui.rs` to match actual `VfxEffect` struct. Common fields to look for: `max_particles`, `emit_rate`, `lifetime`, `speed_min`, `speed_max`. If names differ, update the `egui::Grid` display to match.

- [ ] **Step 6: Wire into engine_runner.rs**

Add field to `EngineApp`:
```rust
vfx_editor_ui: vox_render::vfx_editor_ui::VfxEditorUi,
```
Initialize: `vfx_editor_ui: vox_render::vfx_editor_ui::VfxEditorUi::new()`

In egui block:
```rust
self.vfx_editor_ui.open = self.editor.show_vfx_editor;
self.vfx_editor_ui.show(ctx);
self.editor.show_vfx_editor = self.vfx_editor_ui.open;
```

- [ ] **Step 7: Run test**

```bash
cd crates/vox_render && cargo test vfx_ui_default_closed 2>&1
```
Expected: PASS

- [ ] **Step 8: Full build**

```bash
cargo build 2>&1 | grep "^error"
```

- [ ] **Step 9: Commit**

```bash
git add crates/vox_render/src/vfx_editor_ui.rs crates/vox_render/src/lib.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): VFX asset inspector editor window"
```

---

## Final Integration Check

- [ ] **Run all tests**

```bash
cargo test 2>&1 | tail -20
```
Expected: all pass, 0 failures.

- [ ] **Full release build**

```bash
cargo build --release 2>&1 | grep "^error"
```

- [ ] **Tag the release**

```bash
git tag editor-features-v1
```
