# Production UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three UI windows to `vox_app`: (1) `NotificationToast` -- timed on-screen messages with fade-out, (2) `MiniMap` -- 2D top-down entity dot map, (3) `SettingsPanel` -- resolution/vsync/audio volume sliders with persist to `settings.toml`.

**Architecture:** The existing `notifications.rs` has a `NotificationManager` with `Instant`-based timing and egui rendering. The existing `minimap.rs` has a pixel-buffer `Minimap`. The existing `settings.rs` has a `GameSettings` with toml save/load. This plan extends each module: notifications get a `NotificationQueue` with `tick(dt)` (no `Instant`, pure `f32` TTL decrement for testability) and top-right anchored `egui::Area` stack; minimap gets a `MiniMap` with entity-dot rendering via `egui::Painter`; settings get an `AppSettings` with `show_settings_panel` returning a `bool` dirty flag. All three are wired into `editor.rs` / engine runner.

**Tech Stack:** `egui = "0.31"`, `serde = "1"`, `toml = "0.8"`, `bevy_ecs = "0.16"`, `glam`

---

## Key Files (read before editing)

- `crates/vox_app/src/notifications.rs` -- existing `NotificationManager` with `Instant`-based timing
- `crates/vox_app/src/minimap.rs` -- existing pixel-buffer `Minimap`
- `crates/vox_app/src/settings.rs` -- existing `GameSettings` with toml persistence
- `crates/vox_app/src/editor.rs` -- `SceneEditor` with menu bar, `EditorEntity`
- `crates/vox_app/src/lib.rs` -- already has `pub mod notifications;`, `pub mod minimap;`, `pub mod settings;`
- `crates/vox_app/Cargo.toml` -- already has `egui`, `serde`, `toml`, `tempfile` (dev)

## File Structure

**Modify:**
- `crates/vox_app/src/notifications.rs` -- add `NotificationQueue` with `tick(dt)` and top-right `show`
- `crates/vox_app/src/minimap.rs` -- add `MiniMap` with egui painter entity dots
- `crates/vox_app/src/settings.rs` -- add `AppSettings` with `load_settings`, `save_settings`, `show_settings_panel`

**No new files required.**

---

### Task 1: `NotificationQueue` with `tick(dt)` and egui rendering

**Files:**
- Modify: `crates/vox_app/src/notifications.rs`

Add a `NotificationQueue` alongside the existing `NotificationManager`. The queue uses pure `f32` TTL (no `Instant`) for deterministic testing, and renders a top-right anchored stack of fading labels.

- [ ] **Step 1: Write the additions with tests** -- add at the end of `notifications.rs`, after the existing `NotificationManager::show`:

```rust
// ── NotificationQueue (f32-based TTL, testable) ───────────────────────────

/// A single queued notification with a remaining time-to-live.
#[derive(Debug, Clone)]
pub struct QueuedNotification {
    pub message: String,
    pub ttl: f32,
    pub initial_ttl: f32,
}

/// A notification queue that uses pure f32 TTL for deterministic testing.
///
/// Unlike `NotificationManager` (which uses `Instant`), this queue decrements
/// TTL each `tick(dt)` call, making it fully testable without real clocks.
pub struct NotificationQueue {
    pub items: std::collections::VecDeque<QueuedNotification>,
    pub max_visible: usize,
}

impl NotificationQueue {
    pub fn new(max_visible: usize) -> Self {
        Self {
            items: std::collections::VecDeque::new(),
            max_visible,
        }
    }

    /// Push a new notification with a time-to-live in seconds.
    pub fn push(&mut self, message: String, ttl: f32) {
        self.items.push_back(QueuedNotification {
            message,
            ttl,
            initial_ttl: ttl,
        });
    }

    /// Advance time by `dt` seconds. Removes expired notifications.
    pub fn tick(&mut self, dt: f32) {
        for item in self.items.iter_mut() {
            item.ttl -= dt;
        }
        self.items.retain(|n| n.ttl > 0.0);
    }

    /// Render notifications as a top-right anchored stack of fading labels.
    pub fn show(&self, ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        let right_x = screen.max.x - 10.0;
        let mut y = 50.0;

        for (i, notif) in self.items.iter().rev().take(self.max_visible).enumerate() {
            let alpha = (notif.ttl / notif.initial_ttl).clamp(0.0, 1.0);
            let color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, (alpha * 255.0) as u8);

            egui::Area::new(egui::Id::new(format!("toast_{}", i)))
                .fixed_pos(egui::pos2(right_x - 250.0, y))
                .show(ctx, |ui| {
                    ui.colored_label(color, &notif.message);
                });
            y += 25.0;
        }
    }
}

impl Default for NotificationQueue {
    fn default() -> Self {
        Self::new(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_queue_push_adds_items() {
        let mut q = NotificationQueue::new(5);
        q.push("Hello".into(), 3.0);
        assert_eq!(q.items.len(), 1);
        q.push("World".into(), 2.0);
        assert_eq!(q.items.len(), 2);
    }

    #[test]
    fn notification_queue_tick_removes_expired() {
        let mut q = NotificationQueue::new(5);
        q.push("Short".into(), 1.0);
        q.push("Long".into(), 5.0);
        assert_eq!(q.items.len(), 2);

        q.tick(1.5); // Short expires (ttl was 1.0), Long remains (ttl = 3.5)
        assert_eq!(q.items.len(), 1, "expired notification should be removed");
        assert_eq!(q.items[0].message, "Long");
    }

    #[test]
    fn notification_queue_tick_decrements_ttl() {
        let mut q = NotificationQueue::new(5);
        q.push("Test".into(), 5.0);
        q.tick(2.0);
        assert!((q.items[0].ttl - 3.0).abs() < 1e-5, "ttl should be 3.0 after 2s tick");
    }

    #[test]
    fn notification_queue_empty_after_all_expire() {
        let mut q = NotificationQueue::new(5);
        q.push("A".into(), 1.0);
        q.push("B".into(), 2.0);
        q.tick(3.0);
        assert_eq!(q.items.len(), 0, "all should be expired");
    }

    #[test]
    fn notification_manager_push_and_visible() {
        let mut m = NotificationManager::new(3);
        m.push("Info".into(), NotificationCategory::Info);
        assert_eq!(m.notifications.len(), 1);
        assert_eq!(m.visible().count(), 1);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_app --lib -- notifications::tests 2>&1 | tail -10
```

Expected: all 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_app/src/notifications.rs
git commit -m "feat(notifications): NotificationQueue with f32 TTL tick and top-right egui rendering"
```

---

### Task 2: `MiniMap` with egui painter entity dots

**Files:**
- Modify: `crates/vox_app/src/minimap.rs`

Add a `MiniMap` struct alongside the existing `Minimap` (pixel-buffer). The new `MiniMap` uses `egui::Painter` to draw entity dots and a camera indicator in a fixed-size window.

- [ ] **Step 1: Write additions with tests** -- add at the end of `minimap.rs`:

```rust
// ── MiniMap (egui painter-based) ──────────────────────────────────────────

use glam::Vec3;

/// A scene entity position for the minimap (stripped down for rendering).
#[derive(Debug, Clone)]
pub struct MiniMapEntity {
    pub position: Vec3,
    pub color: egui::Color32,
}

/// An egui-based minimap that draws entity dots on a 2D top-down view.
pub struct MiniMap {
    /// World-space radius shown on the minimap.
    pub radius: f32,
    /// Whether the minimap window is open.
    pub open: bool,
    /// Size of the minimap widget in screen pixels.
    pub widget_size: f32,
}

impl MiniMap {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            open: true,
            widget_size: 200.0,
        }
    }

    /// Convert a world XZ position to minimap screen position.
    fn world_to_map(&self, world_pos: Vec3, camera_pos: Vec3, rect: egui::Rect) -> egui::Pos2 {
        let center = rect.center();
        let scale = self.widget_size / (2.0 * self.radius);
        let dx = (world_pos.x - camera_pos.x) * scale;
        let dz = (world_pos.z - camera_pos.z) * scale;
        egui::pos2(center.x + dx, center.y + dz)
    }

    /// Render the minimap window with entity dots and camera indicator.
    ///
    /// - Entities are drawn as white (or custom color) dots.
    /// - Camera position is drawn as a yellow dot at the center.
    pub fn show(&mut self, ctx: &egui::Context, entities: &[MiniMapEntity], camera_pos: Vec3) {
        if !self.open {
            return;
        }

        egui::Window::new("Mini Map")
            .resizable(false)
            .default_size([self.widget_size, self.widget_size])
            .show(ctx, |ui| {
                let (rect, _response) = ui.allocate_exact_size(
                    egui::vec2(self.widget_size, self.widget_size),
                    egui::Sense::hover(),
                );

                let painter = ui.painter_at(rect);

                // Background
                painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(20, 30, 20));

                // Entity dots
                for entity in entities {
                    let pos = self.world_to_map(entity.position, camera_pos, rect);
                    if rect.contains(pos) {
                        painter.circle_filled(pos, 2.0, entity.color);
                    }
                }

                // Camera indicator (center, yellow)
                let cam_pos = self.world_to_map(camera_pos, camera_pos, rect);
                painter.circle_filled(cam_pos, 4.0, egui::Color32::from_rgb(255, 255, 0));
            });
    }
}

impl Default for MiniMap {
    fn default() -> Self {
        Self::new(500.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimap_pixel_buffer_new() {
        let m = Minimap::new(128, 1000.0);
        assert_eq!(m.size, 128);
        assert_eq!(m.pixels.len(), 128 * 128);
    }

    #[test]
    fn minimap_set_pixel_in_bounds() {
        let mut m = Minimap::new(64, 100.0);
        m.set_pixel(0.0, 0.0, [255, 0, 0, 255]);
        // Center pixel should be set
        let idx = (32 * 64 + 32) as usize;
        assert_eq!(m.pixels[idx], [255, 0, 0, 255]);
    }

    #[test]
    fn minimap_egui_default_radius() {
        let mm = MiniMap::default();
        assert_eq!(mm.radius, 500.0);
        assert!(mm.open);
    }

    #[test]
    fn minimap_world_to_map_center() {
        let mm = MiniMap::new(100.0);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let pos = mm.world_to_map(Vec3::new(10.0, 0.0, 10.0), Vec3::new(10.0, 0.0, 10.0), rect);
        // Entity at same position as camera -> center of rect
        assert!((pos.x - 100.0).abs() < 1.0, "x should be at center; got {}", pos.x);
        assert!((pos.y - 100.0).abs() < 1.0, "y should be at center; got {}", pos.y);
    }

    #[test]
    fn minimap_world_to_map_offset() {
        let mm = MiniMap::new(100.0);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 200.0));
        let pos = mm.world_to_map(Vec3::new(50.0, 0.0, 0.0), Vec3::ZERO, rect);
        // 50 units east with radius 100 and widget 200px -> center + 50px
        assert!((pos.x - 150.0).abs() < 1.0, "x should be 150; got {}", pos.x);
    }

    #[test]
    fn minimap_toggle_open() {
        let mut mm = MiniMap::default();
        assert!(mm.open);
        mm.open = false;
        assert!(!mm.open);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_app --lib -- minimap::tests 2>&1 | tail -10
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_app/src/minimap.rs
git commit -m "feat(minimap): MiniMap with egui painter entity dots + camera indicator"
```

---

### Task 3: `AppSettings` with load/save and settings panel

**Files:**
- Modify: `crates/vox_app/src/settings.rs`

Add an `AppSettings` struct (engine-level, not game-specific) alongside the existing `GameSettings`, with `load_settings`, `save_settings`, and `show_settings_panel`.

- [ ] **Step 1: Write additions with tests** -- add at the end of `settings.rs`:

```rust
// ── AppSettings (engine-level) ────────────────────────────────────────────

/// Engine-level application settings with toml persistence.
///
/// Unlike `GameSettings` which includes game-specific fields (camera_speed,
/// edge_scroll), `AppSettings` covers only engine-level concerns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub resolution: (u32, u32),
    pub vsync: bool,
    pub master_volume: f32,
    pub fullscreen: bool,
    pub render_quality: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            vsync: true,
            master_volume: 0.8,
            fullscreen: false,
            render_quality: "High".to_string(),
        }
    }
}

/// Load settings from a toml file, falling back to `Default` if the file
/// is missing or malformed.
pub fn load_settings(path: &std::path::Path) -> AppSettings {
    match std::fs::read_to_string(path) {
        Ok(content) => toml::from_str(&content).unwrap_or_default(),
        Err(_) => AppSettings::default(),
    }
}

/// Save settings to a toml file.
pub fn save_settings(settings: &AppSettings, path: &std::path::Path) -> Result<(), String> {
    let toml_str = toml::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, toml_str).map_err(|e| e.to_string())
}

/// Render a settings panel in an egui window. Returns `true` if any setting changed.
pub fn show_settings_panel(ctx: &egui::Context, settings: &mut AppSettings, open: &mut bool) -> bool {
    let mut changed = false;

    egui::Window::new("Settings")
        .open(open)
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading("Graphics");

            // Resolution
            ui.horizontal(|ui| {
                ui.label("Resolution:");
                let mut w = settings.resolution.0 as f32;
                let mut h = settings.resolution.1 as f32;
                if ui.add(egui::DragValue::new(&mut w).range(640.0..=3840.0).prefix("W: ")).changed() {
                    settings.resolution.0 = w as u32;
                    changed = true;
                }
                if ui.add(egui::DragValue::new(&mut h).range(480.0..=2160.0).prefix("H: ")).changed() {
                    settings.resolution.1 = h as u32;
                    changed = true;
                }
            });

            // VSync
            if ui.checkbox(&mut settings.vsync, "VSync").changed() {
                changed = true;
            }

            // Fullscreen
            if ui.checkbox(&mut settings.fullscreen, "Fullscreen").changed() {
                changed = true;
            }

            // Quality
            ui.horizontal(|ui| {
                ui.label("Quality:");
                egui::ComboBox::from_id_salt("quality_combo")
                    .selected_text(&settings.render_quality)
                    .show_ui(ui, |ui| {
                        for q in &["Low", "Medium", "High", "Ultra"] {
                            if ui.selectable_label(settings.render_quality == *q, *q).clicked() {
                                settings.render_quality = q.to_string();
                                changed = true;
                            }
                        }
                    });
            });

            ui.separator();
            ui.heading("Audio");

            // Master volume
            ui.horizontal(|ui| {
                ui.label("Master Volume:");
                if ui.add(egui::Slider::new(&mut settings.master_volume, 0.0..=1.0)).changed() {
                    changed = true;
                }
            });
        });

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_settings_default_values() {
        let s = GameSettings::default();
        assert_eq!(s.resolution, (1920, 1080));
        assert!(s.vsync);
        assert_eq!(s.quality, "High");
    }

    #[test]
    fn game_settings_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_settings.toml");
        let mut s = GameSettings::default();
        s.resolution = (2560, 1440);
        s.save(&path).unwrap();
        let loaded = GameSettings::load(&path).unwrap();
        assert_eq!(loaded.resolution, (2560, 1440));
    }

    #[test]
    fn app_settings_default() {
        let s = AppSettings::default();
        assert_eq!(s.resolution, (1920, 1080));
        assert!(s.vsync);
        assert!((s.master_volume - 0.8).abs() < 1e-5);
    }

    #[test]
    fn load_settings_falls_back_to_default_on_missing_file() {
        let path = std::path::Path::new("/tmp/nonexistent_ochroma_settings_test.toml");
        let s = load_settings(path);
        assert_eq!(s.resolution, (1920, 1080));
        assert!(s.vsync);
    }

    #[test]
    fn save_and_load_settings_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("app_settings.toml");
        let mut s = AppSettings::default();
        s.resolution = (3840, 2160);
        s.vsync = false;
        s.master_volume = 0.5;
        save_settings(&s, &path).unwrap();
        let loaded = load_settings(&path);
        assert_eq!(loaded.resolution, (3840, 2160));
        assert!(!loaded.vsync);
        assert!((loaded.master_volume - 0.5).abs() < 1e-5);
    }

    #[test]
    fn load_settings_falls_back_on_malformed_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_settings.toml");
        std::fs::write(&path, "this is not valid toml {{{{").unwrap();
        let s = load_settings(&path);
        // Should fall back to default, not panic
        assert_eq!(s.resolution, (1920, 1080));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vox_app --lib -- settings::tests 2>&1 | tail -10
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/vox_app/src/settings.rs
git commit -m "feat(settings): AppSettings with load/save toml persistence + show_settings_panel"
```

---

### Task 4: Wire all three into editor and engine runner

**Files:**
- Modify: `crates/vox_app/src/editor.rs`
- Modify: `crates/vox_app/src/bin/engine_runner.rs` (or equivalent)

Add `NotificationQueue` as a field on the editor/runtime, `MiniMap` toggled by N key, `SettingsPanel` under the View menu.

- [ ] **Step 1: Add fields to `SceneEditor`**

In `editor.rs`, add to the `SceneEditor` struct:

```rust
pub notification_queue: crate::notifications::NotificationQueue,
pub mini_map: crate::minimap::MiniMap,
pub show_settings: bool,
pub app_settings: crate::settings::AppSettings,
```

- [ ] **Step 2: Initialize in constructor**

In `SceneEditor::new()` or the default impl:

```rust
notification_queue: crate::notifications::NotificationQueue::new(5),
mini_map: crate::minimap::MiniMap::default(),
show_settings: false,
app_settings: crate::settings::load_settings(std::path::Path::new("settings.toml")),
```

- [ ] **Step 3: Add to View menu in the editor UI**

In the menu bar rendering section:

```rust
if ui.button("Mini Map (N)").clicked() {
    self.mini_map.open = !self.mini_map.open;
}
if ui.button("Settings").clicked() {
    self.show_settings = !self.show_settings;
}
```

- [ ] **Step 4: Call `show` methods in the frame render**

In the UI rendering section:

```rust
// Notification toasts
self.notification_queue.tick(dt);
self.notification_queue.show(ctx);

// Mini map
let entities: Vec<crate::minimap::MiniMapEntity> = self.entities.iter().map(|e| {
    crate::minimap::MiniMapEntity {
        position: e.position,
        color: if e.visible { egui::Color32::WHITE } else { egui::Color32::DARK_GRAY },
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
```

- [ ] **Step 5: Add N key toggle in input handler**

In the input handling section:

```rust
// N key toggles minimap
KeyCode::KeyN => {
    self.mini_map.open = !self.mini_map.open;
}
```

- [ ] **Step 6: Verify compile**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

Expected: successful check.

- [ ] **Step 7: Commit**

```bash
git add crates/vox_app/src/editor.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(ui): wire NotificationQueue, MiniMap (N key), SettingsPanel into editor"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 4 sub-features covered (notifications, minimap, settings, wiring)
- [x] **No placeholders:** All code blocks are complete with real egui API calls
- [x] **Type consistency:** `Vec3` positions throughout, `egui::Color32` for colors, `(u32, u32)` for resolution
- [x] **TDD:** 17 tests across 3 modules covering tick/expiry, coordinate mapping, save/load roundtrip, malformed fallback
- [x] **Existing patterns:** Extends existing modules rather than replacing them; `GameSettings` preserved alongside new `AppSettings`
- [x] **Engine generality:** `AppSettings` is engine-level (no game-specific fields); `MiniMapEntity` is generic; `NotificationQueue` is game-agnostic
- [x] **toml/serde:** Already workspace deps, no new dependencies needed
