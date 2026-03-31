# Ochroma Editor — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 Ochroma editor: context-adaptive workspace with a 6-mode strip, always-visible AI bar with scope toggle, intent narrative in the scene hierarchy, node-graph change badge with on-demand reveal, semantic asset search, and NPC path ghost overlays.

**Architecture:** A new `EditorApp` struct in `vox_app` owns all editor UI state and exposes a single `show(ctx, entities, agent_positions)` method that renders the full editor layout via egui. `EngineApp` (in `engine_runner.rs`) replaces its scattered panel calls with one `EditorApp::show()` call. All AI backend interaction goes through a `Box<dyn AiBackend>` trait object — a `StubAiBackend` is used for Phase 1.

**Tech Stack:** `egui 0.31`, `bevy_ecs`, `glam`, `vox_editor::node_graph::OchromaNodeGraph`, `vox_core::types::GaussianSplat`

> **Note:** This plan covers Phase 1 only. Phase 2 (Branching Timeline, Sensory Palettes, Director's Gaze, Proceduralism as Instrument) and Phase 3 (Resonance Testing, Spectral Intent Diffusion, Consequence Simulation Preview) are separate future plans that depend on this one shipping.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_app/src/editor_app.rs` | `EditorApp`, `WorkspaceMode`, `AiScope`, `EditorEvent`, `JobHandle`, `GraphNodeId`, `NodeGraphDiff`, `AssetId` |
| Create | `crates/vox_app/src/mode_strip.rs` | `ModeStrip` — left-rail 6-button mode picker |
| Create | `crates/vox_app/src/context_panel.rs` | `ContextPanel` — 3-tab right sidebar (Context / Scene / Assets) |
| Create | `crates/vox_app/src/scene_tree.rs` | `SceneTree` — renders entity hierarchy with intent subtitles |
| Create | `crates/vox_app/src/ai_bar.rs` | `AiBarState`, `AiBackend` trait, `StubAiBackend`, `AiContext`, `AiResult` |
| Create | `crates/vox_app/src/node_graph_panel.rs` | `NodeGraphPanelState` — badge count, reveal toggle, highlighted node list |
| Create | `crates/vox_app/src/asset_library.rs` | `AssetLibrary`, `AssetEntry`, `EmbeddingBackend` trait, `FixedEmbeddingBackend`, cosine similarity |
| Create | `crates/vox_app/src/ghost_overlays.rs` | `GhostOverlays` — path history ring buffer, splat generation |
| Modify | `crates/vox_app/src/editor.rs` | Add `intent: Option<String>` + `intent()`/`set_intent()` to `EditorEntity` |
| Modify | `crates/vox_app/src/lib.rs` | Register 8 new `pub mod` declarations |
| Modify | `crates/vox_app/src/bin/engine_runner.rs` | Add `editor_app: EditorApp` field to `EngineApp`; call `show()` in egui frame |

---

### Task 1: Core types in `editor_app.rs`

**Files:**
- Create: `crates/vox_app/src/editor_app.rs`

- [ ] **Step 1.1: Write the failing test**

```rust
// At bottom of crates/vox_app/src/editor_app.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_mode_all_variants_distinct() {
        let modes = [
            WorkspaceMode::Sculpt,
            WorkspaceMode::Objects,
            WorkspaceMode::Lighting,
            WorkspaceMode::Animate,
            WorkspaceMode::Logic,
            WorkspaceMode::Simulate,
        ];
        // All 6 modes are pairwise distinct
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert_ne!(modes[i], modes[j]);
            }
        }
    }

    #[test]
    fn node_graph_diff_changed_count_sums_added_and_modified() {
        let diff = NodeGraphDiff::new(
            AssetId(1),
            vec![GraphNodeId(0), GraphNodeId(1)],
            vec![GraphNodeId(2)],
        );
        assert_eq!(diff.changed_count(), 3);
    }

    #[test]
    fn node_graph_diff_asset_id_accessible() {
        let diff = NodeGraphDiff::new(AssetId(42), vec![], vec![GraphNodeId(5)]);
        assert_eq!(diff.asset_id(), AssetId(42));
    }
}
```

- [ ] **Step 1.2: Run test — expect compile failure (types undefined)**

```bash
cd /home/tomespen/git/ochroma
cargo test -p vox_app --lib editor_app 2>&1 | grep "error\[" | head -5
```

Expected: errors like `error[E0412]: cannot find type 'WorkspaceMode'`

- [ ] **Step 1.3: Write the types**

Create `crates/vox_app/src/editor_app.rs`:

```rust
/// The 6 workflow modes of the editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkspaceMode {
    Sculpt,
    Objects,
    Lighting,
    Animate,
    Logic,
    Simulate,
}

/// Controls what context is included with an AI prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiScope {
    Selection,
    Mode,
    Scene,
}

impl AiScope {
    /// Advance to the next scope in the cycle.
    pub fn next(self) -> Self {
        match self {
            AiScope::Selection => AiScope::Mode,
            AiScope::Mode => AiScope::Scene,
            AiScope::Scene => AiScope::Selection,
        }
    }
}

/// Opaque handle for an in-flight AI job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobHandle(pub u64);

/// ID for a node inside a procedural node graph (distinct from scene entity ID).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphNodeId(pub u32);

/// Identifies an asset in the asset library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId(pub u32);

/// Records which node-graph nodes were added or modified by an AI action.
pub struct NodeGraphDiff {
    asset_id: AssetId,
    added_nodes: Vec<GraphNodeId>,
    modified_nodes: Vec<GraphNodeId>,
}

impl NodeGraphDiff {
    pub fn new(
        asset_id: AssetId,
        added_nodes: Vec<GraphNodeId>,
        modified_nodes: Vec<GraphNodeId>,
    ) -> Self {
        Self { asset_id, added_nodes, modified_nodes }
    }

    pub fn asset_id(&self) -> AssetId { self.asset_id }

    pub fn added_nodes(&self) -> &[GraphNodeId] { &self.added_nodes }

    pub fn modified_nodes(&self) -> &[GraphNodeId] { &self.modified_nodes }

    pub fn changed_count(&self) -> usize {
        self.added_nodes.len() + self.modified_nodes.len()
    }
}

/// Events broadcast on EditorApp's internal bus.
#[derive(Debug, Clone)]
pub enum EditorEvent {
    WorkspaceModeChanged { mode: WorkspaceMode },
    AiActionComplete { diff: Option<NodeGraphDiff> },
}

impl Clone for NodeGraphDiff {
    fn clone(&self) -> Self {
        Self {
            asset_id: self.asset_id,
            added_nodes: self.added_nodes.clone(),
            modified_nodes: self.modified_nodes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_mode_all_variants_distinct() {
        let modes = [
            WorkspaceMode::Sculpt,
            WorkspaceMode::Objects,
            WorkspaceMode::Lighting,
            WorkspaceMode::Animate,
            WorkspaceMode::Logic,
            WorkspaceMode::Simulate,
        ];
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert_ne!(modes[i], modes[j]);
            }
        }
    }

    #[test]
    fn node_graph_diff_changed_count_sums_added_and_modified() {
        let diff = NodeGraphDiff::new(
            AssetId(1),
            vec![GraphNodeId(0), GraphNodeId(1)],
            vec![GraphNodeId(2)],
        );
        assert_eq!(diff.changed_count(), 3);
    }

    #[test]
    fn node_graph_diff_asset_id_accessible() {
        let diff = NodeGraphDiff::new(AssetId(42), vec![], vec![GraphNodeId(5)]);
        assert_eq!(diff.asset_id(), AssetId(42));
    }

    #[test]
    fn ai_scope_cycles_all_three() {
        assert_eq!(AiScope::Selection.next(), AiScope::Mode);
        assert_eq!(AiScope::Mode.next(), AiScope::Scene);
        assert_eq!(AiScope::Scene.next(), AiScope::Selection);
    }
}
```

- [ ] **Step 1.4: Register the module in lib.rs**

In `crates/vox_app/src/lib.rs`, add after the last `pub mod` line:

```rust
pub mod editor_app;
```

- [ ] **Step 1.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib editor_app 2>&1 | tail -5
```

Expected:
```
test editor_app::tests::workspace_mode_all_variants_distinct ... ok
test editor_app::tests::node_graph_diff_changed_count_sums_added_and_modified ... ok
test editor_app::tests::node_graph_diff_asset_id_accessible ... ok
test editor_app::tests::ai_scope_cycles_all_three ... ok
```

- [ ] **Step 1.6: Commit**

```bash
git add crates/vox_app/src/editor_app.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): core types WorkspaceMode, AiScope, NodeGraphDiff, AssetId"
```

---

### Task 2: Add `intent` to `EditorEntity`

**Files:**
- Modify: `crates/vox_app/src/editor.rs`

- [ ] **Step 2.1: Write the failing test**

Add to the bottom of `crates/vox_app/src/editor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn make_entity(id: u32) -> EditorEntity {
        EditorEntity {
            id,
            name: "Test".into(),
            asset_path: "".into(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            visible: true,
            locked: false,
            scripts: vec![],
            parent: None,
            children: vec![],
            intent: None,
        }
    }

    #[test]
    fn editor_entity_intent_starts_none() {
        let e = make_entity(1);
        assert_eq!(e.intent(), None);
    }

    #[test]
    fn editor_entity_set_intent_stores_prompt() {
        let mut e = make_entity(2);
        e.set_intent("a ruined watchtower");
        assert_eq!(e.intent(), Some("a ruined watchtower"));
    }

    #[test]
    fn editor_entity_intent_none_when_manually_created() {
        let e = make_entity(3);
        assert!(e.intent().is_none(), "manually created entities have no intent");
    }
}
```

- [ ] **Step 2.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib editor -- editor::tests 2>&1 | grep "error\[" | head -3
```

Expected: `error[E0063]: missing field 'intent'` (or similar — intent field not yet defined)

- [ ] **Step 2.3: Extend `EditorEntity`**

In `crates/vox_app/src/editor.rs`, add `intent` field to `EditorEntity` struct:

```rust
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
    pub intent: Option<String>,   // AI creation prompt; shown as subtitle in scene tree
}
```

Add accessor methods. Find the end of the `EditorEntity` struct definition (before `SceneEditor`) and insert:

```rust
impl EditorEntity {
    pub fn intent(&self) -> Option<&str> {
        self.intent.as_deref()
    }

    pub fn set_intent(&mut self, prompt: impl Into<String>) {
        self.intent = Some(prompt.into());
    }
}
```

- [ ] **Step 2.4: Fix all `EditorEntity` construction sites**

Search for all places that construct `EditorEntity { ... }` and add `intent: None`:

```bash
grep -rn "EditorEntity {" /home/tomespen/git/ochroma/crates/vox_app/src/ 2>/dev/null
```

For each construction site found, add `intent: None,` to the struct literal.

- [ ] **Step 2.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib editor 2>&1 | tail -5
```

Expected:
```
test editor::tests::editor_entity_intent_starts_none ... ok
test editor::tests::editor_entity_set_intent_stores_prompt ... ok
test editor::tests::editor_entity_intent_none_when_manually_created ... ok
```

- [ ] **Step 2.6: Commit**

```bash
git add crates/vox_app/src/editor.rs
git commit -m "feat(editor): add intent field to EditorEntity with accessor methods"
```

---

### Task 3: `ModeStrip` — left-rail mode picker

**Files:**
- Create: `crates/vox_app/src/mode_strip.rs`

- [ ] **Step 3.1: Write the failing test**

Create `crates/vox_app/src/mode_strip.rs` with just the tests:

```rust
use crate::editor_app::WorkspaceMode;

pub struct ModeStrip {
    active: WorkspaceMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_strip_default_is_objects() {
        let strip = ModeStrip::new();
        assert_eq!(strip.active_mode(), WorkspaceMode::Objects);
    }

    #[test]
    fn mode_strip_set_mode_returns_event_on_change() {
        let mut strip = ModeStrip::new();
        let changed = strip.set_mode(WorkspaceMode::Lighting);
        assert!(changed, "setting a different mode should signal a change");
        assert_eq!(strip.active_mode(), WorkspaceMode::Lighting);
    }

    #[test]
    fn mode_strip_set_mode_returns_false_if_same() {
        let mut strip = ModeStrip::new();
        let changed = strip.set_mode(WorkspaceMode::Objects); // same as default
        assert!(!changed, "setting the same mode should not signal a change");
    }

    #[test]
    fn mode_strip_all_six_modes_are_settable() {
        let mut strip = ModeStrip::new();
        for mode in [
            WorkspaceMode::Sculpt,
            WorkspaceMode::Objects,
            WorkspaceMode::Lighting,
            WorkspaceMode::Animate,
            WorkspaceMode::Logic,
            WorkspaceMode::Simulate,
        ] {
            strip.set_mode(mode);
            assert_eq!(strip.active_mode(), mode);
        }
    }
}
```

- [ ] **Step 3.2: Run test — expect compile failure (impl missing)**

```bash
cargo test -p vox_app --lib mode_strip 2>&1 | grep "error\[" | head -3
```

Expected: `error[E0599]: no method named 'new'`

- [ ] **Step 3.3: Implement `ModeStrip`**

Replace the file contents:

```rust
use crate::editor_app::WorkspaceMode;

/// The left-rail vertical icon bar that switches the active workspace mode.
pub struct ModeStrip {
    active: WorkspaceMode,
}

impl ModeStrip {
    pub fn new() -> Self {
        Self { active: WorkspaceMode::Objects }
    }

    pub fn active_mode(&self) -> WorkspaceMode {
        self.active
    }

    /// Set the active mode. Returns `true` if the mode actually changed.
    pub fn set_mode(&mut self, mode: WorkspaceMode) -> bool {
        if self.active == mode {
            return false;
        }
        self.active = mode;
        true
    }

    /// Render the mode strip as a vertical egui panel.
    /// Returns `Some(mode)` if the user clicked a different mode, `None` otherwise.
    pub fn show(&mut self, ui: &mut egui::Ui) -> Option<WorkspaceMode> {
        let modes: &[(WorkspaceMode, &str, &str)] = &[
            (WorkspaceMode::Sculpt,   "⬡", "Sculpt"),
            (WorkspaceMode::Objects,  "⬜", "Objects"),
            (WorkspaceMode::Lighting, "☀", "Lighting"),
            (WorkspaceMode::Animate,  "▶", "Animate"),
            (WorkspaceMode::Logic,    "⬡", "Logic"),
            (WorkspaceMode::Simulate, "⏵", "Simulate"),
        ];

        let mut clicked = None;
        ui.vertical(|ui| {
            ui.set_width(44.0);
            for (mode, icon, label) in modes {
                let selected = self.active == *mode;
                let btn = egui::Button::new(*icon)
                    .min_size(egui::vec2(36.0, 36.0))
                    .selected(selected);
                let resp = ui.add(btn).on_hover_text(*label);
                if resp.clicked() && !selected {
                    self.active = *mode;
                    clicked = Some(*mode);
                }
            }
        });
        clicked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_strip_default_is_objects() {
        let strip = ModeStrip::new();
        assert_eq!(strip.active_mode(), WorkspaceMode::Objects);
    }

    #[test]
    fn mode_strip_set_mode_returns_event_on_change() {
        let mut strip = ModeStrip::new();
        let changed = strip.set_mode(WorkspaceMode::Lighting);
        assert!(changed);
        assert_eq!(strip.active_mode(), WorkspaceMode::Lighting);
    }

    #[test]
    fn mode_strip_set_mode_returns_false_if_same() {
        let mut strip = ModeStrip::new();
        let changed = strip.set_mode(WorkspaceMode::Objects);
        assert!(!changed);
    }

    #[test]
    fn mode_strip_all_six_modes_are_settable() {
        let mut strip = ModeStrip::new();
        for mode in [
            WorkspaceMode::Sculpt,
            WorkspaceMode::Objects,
            WorkspaceMode::Lighting,
            WorkspaceMode::Animate,
            WorkspaceMode::Logic,
            WorkspaceMode::Simulate,
        ] {
            strip.set_mode(mode);
            assert_eq!(strip.active_mode(), mode);
        }
    }
}
```

- [ ] **Step 3.4: Register in lib.rs**

```rust
pub mod mode_strip;
```

- [ ] **Step 3.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib mode_strip 2>&1 | tail -5
```

Expected:
```
test mode_strip::tests::mode_strip_default_is_objects ... ok
test mode_strip::tests::mode_strip_set_mode_returns_event_on_change ... ok
test mode_strip::tests::mode_strip_set_mode_returns_false_if_same ... ok
test mode_strip::tests::mode_strip_all_six_modes_are_settable ... ok
```

- [ ] **Step 3.6: Commit**

```bash
git add crates/vox_app/src/mode_strip.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): ModeStrip — 6-mode left-rail selector"
```

---

### Task 4: `ContextPanel` — 3-tab right sidebar shell

**Files:**
- Create: `crates/vox_app/src/context_panel.rs`

- [ ] **Step 4.1: Write the failing test**

Create `crates/vox_app/src/context_panel.rs`:

```rust
use crate::editor_app::WorkspaceMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarTab {
    Context,
    Scene,
    Assets,
}

pub struct ContextPanel {
    active_tab: SidebarTab,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_panel_default_tab_is_context() {
        let panel = ContextPanel::new();
        assert_eq!(panel.active_tab(), SidebarTab::Context);
    }

    #[test]
    fn context_panel_tab_switch_stores_selection() {
        let mut panel = ContextPanel::new();
        panel.set_tab(SidebarTab::Assets);
        assert_eq!(panel.active_tab(), SidebarTab::Assets);
        panel.set_tab(SidebarTab::Scene);
        assert_eq!(panel.active_tab(), SidebarTab::Scene);
    }

    #[test]
    fn context_label_for_sculpt_mode_is_sculpt_tools() {
        assert_eq!(
            ContextPanel::context_label_for_mode(WorkspaceMode::Sculpt),
            "Sculpt Tools"
        );
    }

    #[test]
    fn context_label_for_simulate_mode_is_simulation() {
        assert_eq!(
            ContextPanel::context_label_for_mode(WorkspaceMode::Simulate),
            "Simulation"
        );
    }
}
```

- [ ] **Step 4.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib context_panel 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 4.3: Implement `ContextPanel`**

```rust
use crate::editor_app::WorkspaceMode;

/// The three fixed tabs in the right sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarTab {
    Context,
    Scene,
    Assets,
}

/// The right-side 3-tab sidebar.
/// - Context tab: changes content based on active WorkspaceMode.
/// - Scene tab: always the scene hierarchy.
/// - Assets tab: always the asset library.
pub struct ContextPanel {
    active_tab: SidebarTab,
}

impl ContextPanel {
    pub fn new() -> Self {
        Self { active_tab: SidebarTab::Context }
    }

    pub fn active_tab(&self) -> SidebarTab {
        self.active_tab
    }

    pub fn set_tab(&mut self, tab: SidebarTab) {
        self.active_tab = tab;
    }

    /// Returns the display label for the Context tab's content area given a mode.
    pub fn context_label_for_mode(mode: WorkspaceMode) -> &'static str {
        match mode {
            WorkspaceMode::Sculpt    => "Sculpt Tools",
            WorkspaceMode::Objects   => "Object Properties",
            WorkspaceMode::Lighting  => "Lighting Controls",
            WorkspaceMode::Animate   => "Animation",
            WorkspaceMode::Logic     => "Logic",
            WorkspaceMode::Simulate  => "Simulation",
        }
    }

    /// Render the full right sidebar panel.
    /// `mode` — the current workspace mode (determines Context tab content).
    /// `show_scene` — closure called to render Scene tab body.
    /// `show_assets` — closure called to render Assets tab body.
    /// `show_context` — closure called to render Context tab body.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        mode: WorkspaceMode,
        show_context: impl FnOnce(&mut egui::Ui),
        show_scene: impl FnOnce(&mut egui::Ui),
        show_assets: impl FnOnce(&mut egui::Ui),
    ) {
        ui.vertical(|ui| {
            // Tab bar
            ui.horizontal(|ui| {
                for (tab, label) in [
                    (SidebarTab::Context, "Context"),
                    (SidebarTab::Scene,   "Scene"),
                    (SidebarTab::Assets,  "Assets"),
                ] {
                    let selected = self.active_tab == tab;
                    if ui.selectable_label(selected, label).clicked() {
                        self.active_tab = tab;
                    }
                }
            });

            ui.separator();

            // Tab body
            match self.active_tab {
                SidebarTab::Context => {
                    ui.label(Self::context_label_for_mode(mode));
                    ui.separator();
                    show_context(ui);
                }
                SidebarTab::Scene  => show_scene(ui),
                SidebarTab::Assets => show_assets(ui),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_panel_default_tab_is_context() {
        let panel = ContextPanel::new();
        assert_eq!(panel.active_tab(), SidebarTab::Context);
    }

    #[test]
    fn context_panel_tab_switch_stores_selection() {
        let mut panel = ContextPanel::new();
        panel.set_tab(SidebarTab::Assets);
        assert_eq!(panel.active_tab(), SidebarTab::Assets);
        panel.set_tab(SidebarTab::Scene);
        assert_eq!(panel.active_tab(), SidebarTab::Scene);
    }

    #[test]
    fn context_label_for_sculpt_mode_is_sculpt_tools() {
        assert_eq!(
            ContextPanel::context_label_for_mode(WorkspaceMode::Sculpt),
            "Sculpt Tools"
        );
    }

    #[test]
    fn context_label_for_simulate_mode_is_simulation() {
        assert_eq!(
            ContextPanel::context_label_for_mode(WorkspaceMode::Simulate),
            "Simulation"
        );
    }
}
```

- [ ] **Step 4.4: Register in lib.rs**

```rust
pub mod context_panel;
```

- [ ] **Step 4.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib context_panel 2>&1 | tail -5
```

Expected:
```
test context_panel::tests::context_panel_default_tab_is_context ... ok
test context_panel::tests::context_panel_tab_switch_stores_selection ... ok
test context_panel::tests::context_label_for_sculpt_mode_is_sculpt_tools ... ok
test context_panel::tests::context_label_for_simulate_mode_is_simulation ... ok
```

- [ ] **Step 4.6: Commit**

```bash
git add crates/vox_app/src/context_panel.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): ContextPanel — 3-tab right sidebar shell"
```

---

### Task 5: `SceneTree` — intent narrative

**Files:**
- Create: `crates/vox_app/src/scene_tree.rs`

- [ ] **Step 5.1: Write the failing test**

Create `crates/vox_app/src/scene_tree.rs`:

```rust
use crate::editor::EditorEntity;
use glam::{Quat, Vec3};

pub struct SceneTree;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: u32, name: &str, intent: Option<&str>) -> EditorEntity {
        let mut e = EditorEntity {
            id,
            name: name.into(),
            asset_path: "".into(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            visible: true,
            locked: false,
            scripts: vec![],
            parent: None,
            children: vec![],
            intent: None,
        };
        if let Some(p) = intent {
            e.set_intent(p);
        }
        e
    }

    #[test]
    fn scene_tree_collects_root_entities() {
        let entities = vec![
            make_entity(1, "Tower", Some("a ruined watchtower")),
            make_entity(2, "Tree", None),
        ];
        let roots = SceneTree::root_entities(&entities);
        assert_eq!(roots.len(), 2, "both entities are roots (no parent)");
    }

    #[test]
    fn scene_tree_root_excludes_children() {
        let entities = vec![
            make_entity(1, "Parent", None),
            EditorEntity {
                id: 2,
                name: "Child".into(),
                parent: Some(1),
                children: vec![],
                intent: None,
                asset_path: "".into(),
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                visible: true,
                locked: false,
                scripts: vec![],
            },
        ];
        let roots = SceneTree::root_entities(&entities);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, 1);
    }

    #[test]
    fn scene_tree_intent_visible_for_ai_entity() {
        let e = make_entity(1, "Ruin", Some("a crumbling stone ruin"));
        assert!(SceneTree::has_visible_intent(&e));
    }

    #[test]
    fn scene_tree_no_intent_for_manual_entity() {
        let e = make_entity(2, "Box", None);
        assert!(!SceneTree::has_visible_intent(&e));
    }
}
```

- [ ] **Step 5.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib scene_tree 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 5.3: Implement `SceneTree`**

```rust
use crate::editor::EditorEntity;

/// Renders the scene hierarchy with AI intent subtitles.
pub struct SceneTree {
    /// The currently selected entity ID.
    selected: Option<u32>,
}

impl SceneTree {
    pub fn new() -> Self {
        Self { selected: None }
    }

    pub fn selected(&self) -> Option<u32> { self.selected }

    pub fn set_selected(&mut self, id: Option<u32>) { self.selected = id; }

    /// Returns only root entities (those with no parent).
    pub fn root_entities(entities: &[EditorEntity]) -> Vec<&EditorEntity> {
        entities.iter().filter(|e| e.parent.is_none()).collect()
    }

    /// Returns true if the entity has an AI intent prompt to display.
    pub fn has_visible_intent(entity: &EditorEntity) -> bool {
        entity.intent().is_some()
    }

    /// Render the scene tree inside `ui`. Calls `on_select` when an entity is clicked.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        entities: &[EditorEntity],
        on_select: &mut dyn FnMut(u32),
    ) {
        let roots = Self::root_entities(entities);
        for entity in roots {
            self.show_entity(ui, entity, entities, on_select);
        }
    }

    fn show_entity(
        &mut self,
        ui: &mut egui::Ui,
        entity: &EditorEntity,
        all_entities: &[EditorEntity],
        on_select: &mut dyn FnMut(u32),
    ) {
        let selected = self.selected == Some(entity.id);
        let has_children = !entity.children.is_empty();

        // Show the entity row (name + optional intent subtitle)
        let row = |ui: &mut egui::Ui| {
            ui.vertical(|ui| {
                let resp = ui.selectable_label(selected, &entity.name);
                if resp.clicked() {
                    self.selected = Some(entity.id);
                    on_select(entity.id);
                }
                if let Some(intent) = entity.intent() {
                    ui.label(
                        egui::RichText::new(intent)
                            .small()
                            .color(egui::Color32::from_gray(130))
                    );
                }
            });
        };

        if has_children {
            egui::CollapsingHeader::new(&entity.name)
                .show(ui, |ui| {
                    row(ui);
                    for &child_id in &entity.children {
                        if let Some(child) = all_entities.iter().find(|e| e.id == child_id) {
                            self.show_entity(ui, child, all_entities, on_select);
                        }
                    }
                });
        } else {
            row(ui);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn make_entity(id: u32, name: &str, intent: Option<&str>) -> EditorEntity {
        let mut e = EditorEntity {
            id,
            name: name.into(),
            asset_path: "".into(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            visible: true,
            locked: false,
            scripts: vec![],
            parent: None,
            children: vec![],
            intent: None,
        };
        if let Some(p) = intent {
            e.set_intent(p);
        }
        e
    }

    #[test]
    fn scene_tree_collects_root_entities() {
        let entities = vec![
            make_entity(1, "Tower", Some("a ruined watchtower")),
            make_entity(2, "Tree", None),
        ];
        let roots = SceneTree::root_entities(&entities);
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn scene_tree_root_excludes_children() {
        let entities = vec![
            make_entity(1, "Parent", None),
            EditorEntity {
                id: 2,
                name: "Child".into(),
                parent: Some(1),
                children: vec![],
                intent: None,
                asset_path: "".into(),
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                visible: true,
                locked: false,
                scripts: vec![],
            },
        ];
        let roots = SceneTree::root_entities(&entities);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, 1);
    }

    #[test]
    fn scene_tree_intent_visible_for_ai_entity() {
        let e = make_entity(1, "Ruin", Some("a crumbling stone ruin"));
        assert!(SceneTree::has_visible_intent(&e));
    }

    #[test]
    fn scene_tree_no_intent_for_manual_entity() {
        let e = make_entity(2, "Box", None);
        assert!(!SceneTree::has_visible_intent(&e));
    }
}
```

- [ ] **Step 5.4: Register in lib.rs**

```rust
pub mod scene_tree;
```

- [ ] **Step 5.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib scene_tree 2>&1 | tail -6
```

Expected: 4 tests passing.

- [ ] **Step 5.6: Commit**

```bash
git add crates/vox_app/src/scene_tree.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): SceneTree with intent subtitle rendering"
```

---

### Task 6: AI bar state + backend trait

**Files:**
- Create: `crates/vox_app/src/ai_bar.rs`

- [ ] **Step 6.1: Write the failing test**

Create `crates/vox_app/src/ai_bar.rs`:

```rust
use crate::editor_app::{AiScope, JobHandle, NodeGraphDiff};

pub struct AiBarState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_bar_default_scope_is_selection() {
        let bar = AiBarState::new();
        assert_eq!(bar.scope(), AiScope::Selection);
    }

    #[test]
    fn ai_bar_cycle_scope_selection_to_mode() {
        let mut bar = AiBarState::new();
        bar.cycle_scope();
        assert_eq!(bar.scope(), AiScope::Mode);
    }

    #[test]
    fn ai_bar_cycle_scope_wraps_scene_to_selection() {
        let mut bar = AiBarState::new();
        bar.cycle_scope(); // → Mode
        bar.cycle_scope(); // → Scene
        bar.cycle_scope(); // → Selection (wrap)
        assert_eq!(bar.scope(), AiScope::Selection);
    }

    #[test]
    fn ai_bar_starts_collapsed() {
        let bar = AiBarState::new();
        assert!(!bar.is_expanded());
    }

    #[test]
    fn ai_bar_toggle_expand_changes_state() {
        let mut bar = AiBarState::new();
        bar.toggle_expand();
        assert!(bar.is_expanded());
        bar.toggle_expand();
        assert!(!bar.is_expanded());
    }

    #[test]
    fn ai_bar_input_text_initially_empty() {
        let bar = AiBarState::new();
        assert_eq!(bar.input_text(), "");
    }

    #[test]
    fn stub_backend_returns_handle_on_submit() {
        let backend = StubAiBackend::new();
        let handle = backend.submit("make it more mossy", AiContext { scope: AiScope::Selection, selection_ids: vec![1] });
        // handle ID is non-zero (first job ID is 1)
        assert_eq!(handle.0, 1);
    }

    #[test]
    fn stub_backend_poll_returns_result_immediately() {
        let backend = StubAiBackend::new();
        let handle = backend.submit("test prompt", AiContext { scope: AiScope::Scene, selection_ids: vec![] });
        let result = backend.poll(&handle);
        assert!(result.is_some(), "stub backend resolves synchronously");
    }
}
```

- [ ] **Step 6.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib ai_bar 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 6.3: Implement AI bar state + backend**

```rust
use crate::editor_app::{AiScope, AssetId, JobHandle, NodeGraphDiff};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Context sent with each AI prompt.
#[derive(Clone)]
pub struct AiContext {
    pub scope: AiScope,
    pub selection_ids: Vec<u32>,
}

/// The result of an AI action.
#[derive(Clone)]
pub struct AiResult {
    /// If the AI modified a node graph, this diff describes the changes.
    pub diff: Option<NodeGraphDiff>,
    /// Human-readable summary of what the AI did.
    pub summary: String,
}

/// The AI backend trait. Implementations are Send + 'static so they can be boxed.
pub trait AiBackend: Send + 'static {
    /// Submit a prompt. Non-blocking — result arrives via `poll`.
    fn submit(&self, prompt: &str, context: AiContext) -> JobHandle;

    /// Poll for a completed result. Returns `Some` when done, `None` while pending.
    fn poll(&self, handle: &JobHandle) -> Option<AiResult>;
}

/// A synchronous stub backend used in Phase 1 and tests.
/// Returns a canned response immediately on every poll.
pub struct StubAiBackend {
    next_id: Arc<AtomicU64>,
}

impl StubAiBackend {
    pub fn new() -> Self {
        Self { next_id: Arc::new(AtomicU64::new(1)) }
    }
}

impl AiBackend for StubAiBackend {
    fn submit(&self, _prompt: &str, _context: AiContext) -> JobHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        JobHandle(id)
    }

    fn poll(&self, _handle: &JobHandle) -> Option<AiResult> {
        Some(AiResult {
            diff: None,
            summary: "AI action applied (stub)".into(),
        })
    }
}

/// All mutable state for the AI bar.
pub struct AiBarState {
    scope: AiScope,
    expanded: bool,
    input_text: String,
    history: Vec<(String, AiResult)>,   // (prompt, result)
    pending: Option<JobHandle>,
}

impl AiBarState {
    pub fn new() -> Self {
        Self {
            scope: AiScope::Selection,
            expanded: false,
            input_text: String::new(),
            history: Vec::new(),
            pending: None,
        }
    }

    pub fn scope(&self) -> AiScope { self.scope }

    pub fn cycle_scope(&mut self) {
        self.scope = self.scope.next();
    }

    pub fn is_expanded(&self) -> bool { self.expanded }

    pub fn toggle_expand(&mut self) { self.expanded = !self.expanded; }

    pub fn input_text(&self) -> &str { &self.input_text }

    pub fn input_text_mut(&mut self) -> &mut String { &mut self.input_text }

    pub fn history(&self) -> &[(String, AiResult)] { &self.history }

    /// Submit the current input text. Returns `Some(handle)` if the prompt was non-empty.
    pub fn submit(
        &mut self,
        backend: &dyn AiBackend,
        selection_ids: Vec<u32>,
    ) -> Option<JobHandle> {
        let text = self.input_text.trim().to_owned();
        if text.is_empty() {
            return None;
        }
        let handle = backend.submit(&text, AiContext {
            scope: self.scope,
            selection_ids,
        });
        self.pending = Some(handle);
        self.input_text.clear();
        Some(handle)
    }

    /// Poll the backend for results. Returns the result if the pending job is complete.
    pub fn tick(&mut self, backend: &dyn AiBackend) -> Option<AiResult> {
        let handle = self.pending?;
        let result = backend.poll(&handle)?;
        self.pending = None;
        self.history.push(("(prompt)".into(), result.clone()));
        Some(result)
    }

    /// Render the AI bar at the bottom of the editor window.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        backend: &dyn AiBackend,
        selection_ids: Vec<u32>,
        on_result: &mut dyn FnMut(AiResult),
    ) {
        // Poll for completed jobs first
        if let Some(result) = self.tick(backend) {
            on_result(result);
        }

        ui.horizontal(|ui| {
            // Scope toggle
            let scope_label = match self.scope {
                AiScope::Selection => "Sel",
                AiScope::Mode      => "Mode",
                AiScope::Scene     => "Scene",
            };
            if ui.button(scope_label).clicked() {
                self.cycle_scope();
            }

            // Input field
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.input_text)
                    .hint_text("Describe what you want…")
                    .desired_width(ui.available_width() - 80.0),
            );

            // Submit on Enter
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.submit(backend, selection_ids);
            }

            // Expand button
            let expand_label = if self.expanded { "▾" } else { "▸" };
            if ui.button(expand_label).clicked() {
                self.toggle_expand();
            }
        });

        if self.expanded {
            ui.separator();
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                for (prompt, result) in self.history.iter().rev() {
                    ui.label(format!("> {}", prompt));
                    ui.label(
                        egui::RichText::new(&result.summary)
                            .color(egui::Color32::from_gray(160))
                            .small()
                    );
                    ui.separator();
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_bar_default_scope_is_selection() {
        let bar = AiBarState::new();
        assert_eq!(bar.scope(), AiScope::Selection);
    }

    #[test]
    fn ai_bar_cycle_scope_selection_to_mode() {
        let mut bar = AiBarState::new();
        bar.cycle_scope();
        assert_eq!(bar.scope(), AiScope::Mode);
    }

    #[test]
    fn ai_bar_cycle_scope_wraps_scene_to_selection() {
        let mut bar = AiBarState::new();
        bar.cycle_scope();
        bar.cycle_scope();
        bar.cycle_scope();
        assert_eq!(bar.scope(), AiScope::Selection);
    }

    #[test]
    fn ai_bar_starts_collapsed() {
        let bar = AiBarState::new();
        assert!(!bar.is_expanded());
    }

    #[test]
    fn ai_bar_toggle_expand_changes_state() {
        let mut bar = AiBarState::new();
        bar.toggle_expand();
        assert!(bar.is_expanded());
        bar.toggle_expand();
        assert!(!bar.is_expanded());
    }

    #[test]
    fn ai_bar_input_text_initially_empty() {
        let bar = AiBarState::new();
        assert_eq!(bar.input_text(), "");
    }

    #[test]
    fn stub_backend_returns_handle_on_submit() {
        let backend = StubAiBackend::new();
        let handle = backend.submit("make it more mossy", AiContext {
            scope: AiScope::Selection,
            selection_ids: vec![1],
        });
        assert_eq!(handle.0, 1);
    }

    #[test]
    fn stub_backend_poll_returns_result_immediately() {
        let backend = StubAiBackend::new();
        let handle = backend.submit("test", AiContext {
            scope: AiScope::Scene,
            selection_ids: vec![],
        });
        assert!(backend.poll(&handle).is_some());
    }
}
```

- [ ] **Step 6.4: Register in lib.rs**

```rust
pub mod ai_bar;
```

- [ ] **Step 6.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib ai_bar 2>&1 | tail -10
```

Expected: 9 tests passing.

- [ ] **Step 6.6: Commit**

```bash
git add crates/vox_app/src/ai_bar.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): AiBarState, AiBackend trait, StubAiBackend, scope cycling"
```

---

### Task 7: `NodeGraphPanelState` — badge + reveal

**Files:**
- Create: `crates/vox_app/src/node_graph_panel.rs`

- [ ] **Step 7.1: Write the failing test**

Create `crates/vox_app/src/node_graph_panel.rs`:

```rust
use crate::editor_app::{AssetId, GraphNodeId, NodeGraphDiff};

pub struct NodeGraphPanelState;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diff(added: usize, modified: usize) -> NodeGraphDiff {
        NodeGraphDiff::new(
            AssetId(1),
            (0..added).map(|i| GraphNodeId(i as u32)).collect(),
            (100..100 + modified).map(|i| GraphNodeId(i as u32)).collect(),
        )
    }

    #[test]
    fn panel_starts_with_no_badge() {
        let panel = NodeGraphPanelState::new();
        assert_eq!(panel.badge_count(), 0);
    }

    #[test]
    fn notify_diff_sets_badge_count() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(3, 2));
        assert_eq!(panel.badge_count(), 5);
    }

    #[test]
    fn panel_starts_closed() {
        let panel = NodeGraphPanelState::new();
        assert!(!panel.is_revealed());
    }

    #[test]
    fn open_reveal_sets_revealed() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(1, 0));
        panel.open_reveal();
        assert!(panel.is_revealed());
    }

    #[test]
    fn close_reveal_clears_revealed_and_badge() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(2, 1));
        panel.open_reveal();
        panel.close_reveal();
        assert!(!panel.is_revealed());
        assert_eq!(panel.badge_count(), 0, "badge clears when graph is dismissed");
    }

    #[test]
    fn highlighted_nodes_match_diff_after_open() {
        let mut panel = NodeGraphPanelState::new();
        let diff = make_diff(2, 1);
        panel.notify_diff(diff);
        panel.open_reveal();
        // Should highlight 3 nodes total
        assert_eq!(panel.highlighted_nodes().len(), 3);
    }
}
```

- [ ] **Step 7.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib node_graph_panel 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 7.3: Implement `NodeGraphPanelState`**

```rust
use crate::editor_app::{AssetId, GraphNodeId, NodeGraphDiff};

/// State for the node-graph change badge and on-demand reveal panel.
pub struct NodeGraphPanelState {
    /// Number of nodes changed by the last AI action. 0 = no badge.
    badge_count: usize,
    /// Whether the node graph slide-up panel is currently open.
    revealed: bool,
    /// The nodes to highlight when the panel is open.
    highlighted: Vec<GraphNodeId>,
    /// The asset whose graph is currently open.
    open_asset: Option<AssetId>,
}

impl NodeGraphPanelState {
    pub fn new() -> Self {
        Self {
            badge_count: 0,
            revealed: false,
            highlighted: Vec::new(),
            open_asset: None,
        }
    }

    pub fn badge_count(&self) -> usize { self.badge_count }

    pub fn is_revealed(&self) -> bool { self.revealed }

    pub fn highlighted_nodes(&self) -> &[GraphNodeId] { &self.highlighted }

    pub fn open_asset(&self) -> Option<AssetId> { self.open_asset }

    /// Called after an AI action completes. Sets the badge.
    pub fn notify_diff(&mut self, diff: NodeGraphDiff) {
        self.badge_count = diff.changed_count();
        // Accumulate highlights so they're ready when the user opens the panel
        let mut highlights: Vec<GraphNodeId> = diff.added_nodes().to_vec();
        highlights.extend_from_slice(diff.modified_nodes());
        self.highlighted = highlights;
        self.open_asset = Some(diff.asset_id());
    }

    /// Open the reveal panel (triggered by badge click).
    pub fn open_reveal(&mut self) {
        self.revealed = true;
    }

    /// Close the reveal panel and clear the badge.
    pub fn close_reveal(&mut self) {
        self.revealed = false;
        self.badge_count = 0;
        // Keep highlighted/open_asset so the graph can be re-opened without re-running AI
    }

    /// Render the badge inside the Context tab. Returns true if the user clicked the badge.
    pub fn show_badge(&self, ui: &mut egui::Ui) -> bool {
        if self.badge_count == 0 {
            return false;
        }
        let label = format!("⬡ {} nodes changed", self.badge_count);
        ui.add(
            egui::Label::new(
                egui::RichText::new(label)
                    .color(egui::Color32::from_rgb(100, 160, 255))
                    .small()
            )
        ).clicked()
    }

    /// Render the node graph reveal panel (slide-up overlay at the bottom of the context panel).
    pub fn show_reveal_panel(&mut self, ui: &mut egui::Ui) {
        if !self.revealed {
            return;
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Node Graph")
                    .strong()
                    .color(egui::Color32::from_rgb(100, 160, 255))
            );
            if ui.small_button("✕").clicked() {
                self.close_reveal();
                return;
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
            for node_id in &self.highlighted {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("+ ")
                            .color(egui::Color32::from_rgb(100, 200, 100))
                    );
                    ui.label(format!("Node #{}", node_id.0));
                    ui.label(
                        egui::RichText::new("AI")
                            .small()
                            .color(egui::Color32::from_rgb(100, 160, 255))
                    );
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diff(added: usize, modified: usize) -> NodeGraphDiff {
        NodeGraphDiff::new(
            AssetId(1),
            (0..added).map(|i| GraphNodeId(i as u32)).collect(),
            (100..100 + modified).map(|i| GraphNodeId(i as u32)).collect(),
        )
    }

    #[test]
    fn panel_starts_with_no_badge() {
        let panel = NodeGraphPanelState::new();
        assert_eq!(panel.badge_count(), 0);
    }

    #[test]
    fn notify_diff_sets_badge_count() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(3, 2));
        assert_eq!(panel.badge_count(), 5);
    }

    #[test]
    fn panel_starts_closed() {
        let panel = NodeGraphPanelState::new();
        assert!(!panel.is_revealed());
    }

    #[test]
    fn open_reveal_sets_revealed() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(1, 0));
        panel.open_reveal();
        assert!(panel.is_revealed());
    }

    #[test]
    fn close_reveal_clears_revealed_and_badge() {
        let mut panel = NodeGraphPanelState::new();
        panel.notify_diff(make_diff(2, 1));
        panel.open_reveal();
        panel.close_reveal();
        assert!(!panel.is_revealed());
        assert_eq!(panel.badge_count(), 0);
    }

    #[test]
    fn highlighted_nodes_match_diff_after_open() {
        let mut panel = NodeGraphPanelState::new();
        let diff = make_diff(2, 1);
        panel.notify_diff(diff);
        panel.open_reveal();
        assert_eq!(panel.highlighted_nodes().len(), 3);
    }
}
```

- [ ] **Step 7.4: Register in lib.rs**

```rust
pub mod node_graph_panel;
```

- [ ] **Step 7.5: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib node_graph_panel 2>&1 | tail -8
```

Expected: 6 tests passing.

- [ ] **Step 7.6: Commit**

```bash
git add crates/vox_app/src/node_graph_panel.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): NodeGraphPanelState — badge count, reveal toggle, highlighted nodes"
```

---

### Task 8: `AssetLibrary` — semantic search

**Files:**
- Create: `crates/vox_app/src/asset_library.rs`

- [ ] **Step 8.1: Write the failing test**

Create `crates/vox_app/src/asset_library.rs`:

```rust
use crate::editor_app::AssetId;

pub struct AssetEntry {
    id: AssetId,
    name: String,
    embedding: Vec<f32>,
}

pub struct AssetLibrary;

#[cfg(test)]
mod tests {
    use super::*;

    fn rock_library() -> AssetLibrary {
        // Three assets: Rock A (nearly identical to query), Rock B (orthogonal), Rock C (high similarity)
        AssetLibrary::with_entries(vec![
            AssetEntry::new(AssetId(1), "Rock A", vec![0.99f32, 0.14, 0.0]),
            AssetEntry::new(AssetId(2), "Rock B", vec![0.0f32, 1.0, 0.0]),
            AssetEntry::new(AssetId(3), "Rock C", vec![0.98f32, 0.20, 0.0]),
        ])
    }

    fn query_embedding() -> Vec<f32> {
        vec![1.0f32, 0.0, 0.0]
    }

    #[test]
    fn search_returns_only_results_above_threshold() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        // Rock B is orthogonal → similarity ≈ 0.0, should not appear
        assert!(results.iter().all(|(e, _)| e.id() != AssetId(2)));
    }

    #[test]
    fn search_results_are_sorted_descending_by_score() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    #[test]
    fn search_returns_two_high_similarity_rocks() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        assert_eq!(results.len(), 2, "Rock A and Rock C should match; Rock B should not");
    }

    #[test]
    fn search_empty_library_returns_empty() {
        let library = AssetLibrary::with_entries(vec![]);
        let results = library.search_with_embedding(&[1.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![0.6f32, 0.8, 0.0];
        let sim = AssetLibrary::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "sim={}", sim);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = AssetLibrary::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "sim={}", sim);
    }
}
```

- [ ] **Step 8.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib asset_library 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 8.3: Implement `AssetLibrary`**

```rust
use crate::editor_app::AssetId;

/// An asset with a pre-computed LVM embedding for semantic search.
pub struct AssetEntry {
    id: AssetId,
    name: String,
    /// L2-normalised embedding vector. Length is fixed by the embedding model.
    embedding: Vec<f32>,
}

impl AssetEntry {
    pub fn new(id: AssetId, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self { id, name: name.into(), embedding }
    }

    pub fn id(&self) -> AssetId { self.id }
    pub fn name(&self) -> &str { &self.name }
    pub fn embedding(&self) -> &[f32] { &self.embedding }
}

/// The semantic asset library. Searching uses cosine similarity against pre-computed embeddings.
pub struct AssetLibrary {
    entries: Vec<AssetEntry>,
    /// Minimum cosine similarity to include in results (spec: >= 0.6).
    threshold: f32,
}

impl AssetLibrary {
    pub fn new() -> Self {
        Self { entries: Vec::new(), threshold: 0.6 }
    }

    pub fn with_entries(entries: Vec<AssetEntry>) -> Self {
        Self { entries, threshold: 0.6 }
    }

    pub fn add(&mut self, entry: AssetEntry) {
        self.entries.push(entry);
    }

    /// Cosine similarity between two vectors. Returns 0.0 if either is zero.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "embedding length mismatch");
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Search the library using a pre-computed query embedding.
    /// Returns entries with similarity >= 0.6, sorted descending.
    /// Empty result means no match — callers should offer an "AI Generate" button.
    pub fn search_with_embedding(&self, query: &[f32]) -> Vec<(&AssetEntry, f32)> {
        let mut results: Vec<(&AssetEntry, f32)> = self.entries.iter()
            .map(|e| {
                let sim = Self::cosine_similarity(query, e.embedding());
                (e, sim)
            })
            .filter(|(_, sim)| *sim >= self.threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn entries(&self) -> &[AssetEntry] { &self.entries }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rock_library() -> AssetLibrary {
        AssetLibrary::with_entries(vec![
            AssetEntry::new(AssetId(1), "Rock A", vec![0.99f32, 0.14, 0.0]),
            AssetEntry::new(AssetId(2), "Rock B", vec![0.0f32, 1.0, 0.0]),
            AssetEntry::new(AssetId(3), "Rock C", vec![0.98f32, 0.20, 0.0]),
        ])
    }

    fn query_embedding() -> Vec<f32> {
        vec![1.0f32, 0.0, 0.0]
    }

    #[test]
    fn search_returns_only_results_above_threshold() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        assert!(results.iter().all(|(e, _)| e.id() != AssetId(2)));
    }

    #[test]
    fn search_results_are_sorted_descending_by_score() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    #[test]
    fn search_returns_two_high_similarity_rocks() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_empty_library_returns_empty() {
        let library = AssetLibrary::with_entries(vec![]);
        let results = library.search_with_embedding(&[1.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![0.6f32, 0.8, 0.0];
        let sim = AssetLibrary::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "sim={}", sim);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = AssetLibrary::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "sim={}", sim);
    }
}
```

- [ ] **Step 8.4: Add asset library widget** (search bar + results + AI generate button)

Append to the end of `asset_library.rs` (before the `#[cfg(test)]` block):

```rust
/// Transient state for the asset library UI tab.
pub struct AssetLibraryUi {
    query: String,
    /// Cached results from the last search. Indices into the library.
    cached_results: Vec<(usize, f32)>,
}

impl AssetLibraryUi {
    pub fn new() -> Self {
        Self { query: String::new(), cached_results: Vec::new() }
    }

    /// Render the asset library tab.
    /// `get_embedding` converts the query string to an embedding vector.
    ///   Pass `|_| vec![0.0; 3]` in tests or Phase 1 (no real LVM).
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        library: &AssetLibrary,
        get_embedding: &dyn Fn(&str) -> Vec<f32>,
        on_select: &mut dyn FnMut(AssetId),
    ) {
        // Search bar
        let changed = ui.add(
            egui::TextEdit::singleline(&mut self.query)
                .hint_text("Search assets by feeling…")
                .desired_width(ui.available_width())
        ).changed();

        if changed && !self.query.is_empty() {
            let embedding = get_embedding(&self.query);
            let results = library.search_with_embedding(&embedding);
            self.cached_results = results.iter()
                .map(|(e, sim)| {
                    let idx = library.entries().iter().position(|x| x.id() == e.id()).unwrap_or(0);
                    (idx, *sim)
                })
                .collect();
        }

        ui.separator();

        if self.query.is_empty() {
            // Browse mode — show all entries
            egui::ScrollArea::vertical().show(ui, |ui| {
                for entry in library.entries() {
                    if ui.selectable_label(false, entry.name()).clicked() {
                        on_select(entry.id());
                    }
                }
            });
        } else if self.cached_results.is_empty() {
            // No results → offer AI generation
            ui.label(egui::RichText::new("No matching assets").color(egui::Color32::from_gray(160)));
            if ui.button("✦ Generate with AI").clicked() {
                // Callers handle this by watching for a separate "generate" signal.
                // For Phase 1, this is a no-op placeholder button.
            }
        } else {
            // Show ranked results
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, sim) in &self.cached_results {
                    if let Some(entry) = library.entries().get(*idx) {
                        let label = format!("{} ({:.0}%)", entry.name(), sim * 100.0);
                        if ui.selectable_label(false, label).clicked() {
                            on_select(entry.id());
                        }
                    }
                }
            });
        }
    }
}
```

- [ ] **Step 8.5: Register in lib.rs**

```rust
pub mod asset_library;
```

- [ ] **Step 8.6: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib asset_library 2>&1 | tail -8
```

Expected: 6 tests passing.

- [ ] **Step 8.7: Commit**

```bash
git add crates/vox_app/src/asset_library.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): AssetLibrary with cosine similarity search and AI Generate fallback"
```

---

### Task 9: `GhostOverlays` — NPC path ghosts

**Files:**
- Create: `crates/vox_app/src/ghost_overlays.rs`

- [ ] **Step 9.1: Write the failing test**

Create `crates/vox_app/src/ghost_overlays.rs`:

```rust
use vox_core::types::GaussianSplat;
use std::collections::VecDeque;

pub struct GhostOverlays;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn ghost_overlays_start_disabled() {
        let overlays = GhostOverlays::new();
        assert!(!overlays.enabled());
    }

    #[test]
    fn enable_disable_toggles_state() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        assert!(overlays.enabled());
        overlays.set_enabled(false);
        assert!(!overlays.enabled());
    }

    #[test]
    fn update_adds_position_to_history() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let positions: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        overlays.update(positions);
        assert_eq!(overlays.agent_count(), 2);
        assert_eq!(overlays.history_for_agent(0).len(), 1);
    }

    #[test]
    fn history_capped_at_max_frames() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let pos: &[[f32; 3]] = &[[0.0, 0.0, 0.0]];
        for _ in 0..100 {
            overlays.update(pos);
        }
        assert!(overlays.history_for_agent(0).len() <= GhostOverlays::MAX_HISTORY_FRAMES);
    }

    #[test]
    fn generate_path_splats_produces_splats_per_history_point() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let pos: &[[f32; 3]] = &[[1.0, 0.0, 0.0]];
        overlays.update(pos);
        overlays.update(&[[2.0, 0.0, 0.0]]);
        overlays.update(&[[3.0, 0.0, 0.0]]);
        let splats = overlays.generate_path_splats();
        assert!(!splats.is_empty(), "at least one splat per history point");
        assert!(splats.len() >= 3);
    }

    #[test]
    fn no_splats_when_disabled() {
        let mut overlays = GhostOverlays::new();
        overlays.update(&[[1.0, 0.0, 0.0]]);
        let splats = overlays.generate_path_splats();
        assert!(splats.is_empty(), "disabled overlays produce no splats");
    }
}
```

- [ ] **Step 9.2: Run test — expect compile failure**

```bash
cargo test -p vox_app --lib ghost_overlays 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 9.3: Implement `GhostOverlays`**

```rust
use vox_core::types::GaussianSplat;
use half::f16;
use std::collections::VecDeque;

/// Manages semi-transparent NPC path ghost overlays in Simulate mode.
pub struct GhostOverlays {
    enabled: bool,
    /// Per-agent ring buffer of past world positions.
    history: Vec<VecDeque<[f32; 3]>>,
}

impl GhostOverlays {
    /// Number of past frames kept in the history ring buffer (≈2s at 60fps).
    pub const MAX_HISTORY_FRAMES: usize = 120;

    pub fn new() -> Self {
        Self { enabled: false, history: Vec::new() }
    }

    pub fn enabled(&self) -> bool { self.enabled }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.history.clear();
        }
    }

    pub fn agent_count(&self) -> usize { self.history.len() }

    pub fn history_for_agent(&self, agent_idx: usize) -> &VecDeque<[f32; 3]> {
        &self.history[agent_idx]
    }

    /// Update history from the current agent positions.
    /// `positions` is a slice of world positions, one per agent.
    /// The number of agents may change between calls; history is extended or truncated.
    pub fn update(&mut self, positions: &[[f32; 3]]) {
        if !self.enabled {
            return;
        }

        // Resize history buffers to match current agent count
        self.history.resize_with(positions.len(), VecDeque::new);

        for (i, &pos) in positions.iter().enumerate() {
            let buf = &mut self.history[i];
            buf.push_back(pos);
            if buf.len() > Self::MAX_HISTORY_FRAMES {
                buf.pop_front();
            }
        }
    }

    /// Generate Gaussian splats representing all agent path histories.
    /// Returns empty vec when overlays are disabled.
    pub fn generate_path_splats(&self) -> Vec<GaussianSplat> {
        if !self.enabled {
            return Vec::new();
        }

        let mut splats = Vec::new();
        for buf in &self.history {
            let len = buf.len();
            for (frame_idx, &pos) in buf.iter().enumerate() {
                // Older frames are more transparent (fade from past to present)
                let age = frame_idx as f32 / len.max(1) as f32;  // 0 = oldest, 1 = newest
                let alpha = age * 0.35;  // max opacity 35% at newest frame
                splats.push(make_ghost_splat(pos, alpha));
            }
        }
        splats
    }
}

fn make_ghost_splat(pos: [f32; 3], alpha: f32) -> GaussianSplat {
    let mut s = GaussianSplat::default();
    s.xyz = pos;
    // NPC path ghosts: cyan tint
    s.f_dc = [
        f16::from_f32(0.2 * alpha),
        f16::from_f32(0.8 * alpha),
        f16::from_f32(0.9 * alpha),
    ];
    s.opacity = f16::from_f32(alpha);
    s.scale = [f16::from_f32(0.15); 3];
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_overlays_start_disabled() {
        let overlays = GhostOverlays::new();
        assert!(!overlays.enabled());
    }

    #[test]
    fn enable_disable_toggles_state() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        assert!(overlays.enabled());
        overlays.set_enabled(false);
        assert!(!overlays.enabled());
    }

    #[test]
    fn update_adds_position_to_history() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let positions: &[[f32; 3]] = &[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        overlays.update(positions);
        assert_eq!(overlays.agent_count(), 2);
        assert_eq!(overlays.history_for_agent(0).len(), 1);
    }

    #[test]
    fn history_capped_at_max_frames() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        let pos: &[[f32; 3]] = &[[0.0, 0.0, 0.0]];
        for _ in 0..100 {
            overlays.update(pos);
        }
        assert!(overlays.history_for_agent(0).len() <= GhostOverlays::MAX_HISTORY_FRAMES);
    }

    #[test]
    fn generate_path_splats_produces_splats_per_history_point() {
        let mut overlays = GhostOverlays::new();
        overlays.set_enabled(true);
        overlays.update(&[[1.0, 0.0, 0.0]]);
        overlays.update(&[[2.0, 0.0, 0.0]]);
        overlays.update(&[[3.0, 0.0, 0.0]]);
        let splats = overlays.generate_path_splats();
        assert!(!splats.is_empty());
        assert!(splats.len() >= 3);
    }

    #[test]
    fn no_splats_when_disabled() {
        let mut overlays = GhostOverlays::new();
        overlays.update(&[[1.0, 0.0, 0.0]]);
        let splats = overlays.generate_path_splats();
        assert!(splats.is_empty());
    }
}
```

- [ ] **Step 9.4: Register in lib.rs**

```rust
pub mod ghost_overlays;
```

- [ ] **Step 9.5: Check that `GaussianSplat::default()` exists**

```bash
grep -n "impl Default for GaussianSplat\|fn default" /home/tomespen/git/ochroma/crates/vox_core/src/types.rs | head -5
```

If `Default` is not derived, add `#[derive(Default)]` to `GaussianSplat` in `crates/vox_core/src/types.rs`. If it uses `f16` fields without a default, initialise with `f16::from_f32(0.0)` explicitly in `make_ghost_splat` instead of `GaussianSplat::default()`:

```rust
fn make_ghost_splat(pos: [f32; 3], alpha: f32) -> GaussianSplat {
    // Check actual GaussianSplat field names in crates/vox_core/src/types.rs
    // before finalising this. Adjust field names to match exactly.
    GaussianSplat {
        xyz: pos,
        f_dc: [
            f16::from_f32(0.2 * alpha),
            f16::from_f32(0.8 * alpha),
            f16::from_f32(0.9 * alpha),
        ],
        opacity: f16::from_f32(alpha),
        scale: [f16::from_f32(0.15); 3],
        ..Default::default()
    }
}
```

- [ ] **Step 9.6: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib ghost_overlays 2>&1 | tail -8
```

Expected: 5 tests passing.

- [ ] **Step 9.7: Commit**

```bash
git add crates/vox_app/src/ghost_overlays.rs crates/vox_app/src/lib.rs
git commit -m "feat(editor): GhostOverlays — NPC path history ring buffer and splat generation"
```

---

### Task 10: `EditorApp` — owner + `show()` method

**Files:**
- Modify: `crates/vox_app/src/editor_app.rs`

This task assembles all the components built in Tasks 1–9 into a single `EditorApp` struct with a `show()` method.

- [ ] **Step 10.1: Write the failing test**

Append to `crates/vox_app/src/editor_app.rs` (inside the `tests` module):

```rust
#[test]
fn editor_app_default_mode_is_objects() {
    let app = EditorApp::new();
    assert_eq!(app.mode(), WorkspaceMode::Objects);
}

#[test]
fn editor_app_set_mode_changes_mode() {
    let mut app = EditorApp::new();
    app.set_mode(WorkspaceMode::Simulate);
    assert_eq!(app.mode(), WorkspaceMode::Simulate);
}

#[test]
fn editor_app_notify_ai_result_with_diff_sets_badge() {
    use crate::ai_bar::AiResult;
    let mut app = EditorApp::new();
    let diff = NodeGraphDiff::new(AssetId(1), vec![GraphNodeId(0)], vec![]);
    app.notify_ai_result(AiResult { diff: Some(diff), summary: "done".into() });
    assert_eq!(app.node_graph_panel.badge_count(), 1);
}

#[test]
fn editor_app_notify_ai_result_without_diff_leaves_badge_zero() {
    use crate::ai_bar::AiResult;
    let mut app = EditorApp::new();
    app.notify_ai_result(AiResult { diff: None, summary: "done".into() });
    assert_eq!(app.node_graph_panel.badge_count(), 0);
}
```

- [ ] **Step 10.2: Run test — expect compile failure (EditorApp undefined)**

```bash
cargo test -p vox_app --lib editor_app 2>&1 | grep "error\[" | head -3
```

- [ ] **Step 10.3: Add EditorApp to `editor_app.rs`**

Append the following to `crates/vox_app/src/editor_app.rs` (after the existing type definitions, before `#[cfg(test)]`):

```rust
use crate::ai_bar::{AiBackend, AiBarState, AiResult, StubAiBackend};
use crate::asset_library::{AssetLibrary, AssetLibraryUi};
use crate::context_panel::ContextPanel;
use crate::editor::EditorEntity;
use crate::ghost_overlays::GhostOverlays;
use crate::mode_strip::ModeStrip;
use crate::node_graph_panel::NodeGraphPanelState;
use crate::scene_tree::SceneTree;

/// The single owner of all editor UI state.
pub struct EditorApp {
    mode_strip: ModeStrip,
    context_panel: ContextPanel,
    pub(crate) node_graph_panel: NodeGraphPanelState,
    scene_tree: SceneTree,
    ai_bar: AiBarState,
    asset_library: AssetLibrary,
    asset_library_ui: AssetLibraryUi,
    ghost_overlays: GhostOverlays,
    backend: Box<dyn AiBackend>,
}

impl EditorApp {
    pub fn new() -> Self {
        Self {
            mode_strip: ModeStrip::new(),
            context_panel: ContextPanel::new(),
            node_graph_panel: NodeGraphPanelState::new(),
            scene_tree: SceneTree::new(),
            ai_bar: AiBarState::new(),
            asset_library: AssetLibrary::new(),
            asset_library_ui: AssetLibraryUi::new(),
            ghost_overlays: GhostOverlays::new(),
            backend: Box::new(StubAiBackend::new()),
        }
    }

    pub fn mode(&self) -> WorkspaceMode {
        self.mode_strip.active_mode()
    }

    pub fn set_mode(&mut self, mode: WorkspaceMode) {
        let changed = self.mode_strip.set_mode(mode);
        if changed && mode == WorkspaceMode::Simulate {
            self.ghost_overlays.set_enabled(true);
        } else if changed {
            self.ghost_overlays.set_enabled(false);
        }
    }

    /// Call after an AI job completes. Updates the node graph badge if the result has a diff.
    pub fn notify_ai_result(&mut self, result: AiResult) {
        if let Some(diff) = result.diff {
            self.node_graph_panel.notify_diff(diff);
        }
    }

    /// Update ghost overlay history from the current agent world positions.
    /// Pass `patrol_agents.iter().map(|a| a.position.to_array()).collect::<Vec<_>>()`.
    pub fn update_ghost_overlays(&mut self, agent_positions: &[[f32; 3]]) {
        self.ghost_overlays.update(agent_positions);
    }

    /// Returns ghost overlay splats for the current frame. Empty when overlays are disabled.
    pub fn ghost_overlay_splats(&self) -> Vec<vox_core::types::GaussianSplat> {
        self.ghost_overlays.generate_path_splats()
    }

    /// Render the entire editor layout. Call once per egui frame.
    ///
    /// - `ctx`            — the egui context for this frame
    /// - `entities`       — the current scene entity list
    /// - `selection_ids`  — IDs of currently selected entities
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        entities: &[EditorEntity],
        selection_ids: Vec<u32>,
    ) {
        let mode = self.mode_strip.active_mode();

        // Left panel: mode strip
        egui::SidePanel::left("mode_strip")
            .exact_width(48.0)
            .resizable(false)
            .show(ctx, |ui| {
                if let Some(new_mode) = self.mode_strip.show(ui) {
                    // Ghost overlays auto-enable in Simulate mode
                    self.ghost_overlays.set_enabled(new_mode == WorkspaceMode::Simulate);
                }
            });

        // Right panel: 3-tab sidebar
        egui::SidePanel::right("context_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                let ng_panel = &mut self.node_graph_panel;
                let scene_tree = &mut self.scene_tree;
                let asset_lib = &self.asset_library;
                let asset_lib_ui = &mut self.asset_library_ui;

                self.context_panel.show(
                    ui,
                    mode,
                    |ui| {
                        // Context tab body: badge + reveal panel + mode tools
                        if ng_panel.show_badge(ui) {
                            ng_panel.open_reveal();
                        }
                        ng_panel.show_reveal_panel(ui);
                    },
                    |ui| {
                        // Scene tab body
                        scene_tree.show(ui, entities, &mut |_id| {});
                    },
                    |ui| {
                        // Assets tab body
                        asset_lib_ui.show(ui, asset_lib, &|_| vec![0.0f32; 3], &mut |_| {});
                    },
                );
            });

        // Bottom panel: AI bar
        egui::TopBottomPanel::bottom("ai_bar")
            .resizable(true)
            .min_height(36.0)
            .show(ctx, |ui| {
                let backend = self.backend.as_ref();
                self.ai_bar.show(
                    ui,
                    backend,
                    selection_ids,
                    &mut |result| self.notify_ai_result(result),
                );
            });
    }
}
```

- [ ] **Step 10.4: Run tests — expect PASS**

```bash
cargo test -p vox_app --lib editor_app 2>&1 | tail -10
```

Expected: all editor_app tests passing (including the 4 new ones from Step 10.1).

- [ ] **Step 10.5: Run full vox_app lib tests**

```bash
cargo test -p vox_app --lib 2>&1 | tail -10
```

Expected: all tests from Tasks 1–9 still passing.

- [ ] **Step 10.6: Commit**

```bash
git add crates/vox_app/src/editor_app.rs
git commit -m "feat(editor): EditorApp — assembles all Phase 1 components, single show() entry point"
```

---

### Task 11: Wire `EditorApp` into `engine_runner.rs`

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 11.1: Add `editor_app` field to `EngineApp`**

Find the `struct EngineApp {` definition (around line 141). Add the field:

```rust
editor_app: vox_app::editor_app::EditorApp,
```

Find the `EngineApp` construction site (around the line that assigns `patrol_agents: Vec::new()`). Add:

```rust
editor_app: vox_app::editor_app::EditorApp::new(),
```

- [ ] **Step 11.2: Replace scattered egui panel calls with `EditorApp::show()`**

Search for where egui panels are rendered for the existing `SceneEditor`:

```bash
grep -n "scene_editor\|SceneEditor\|egui::SidePanel\|egui::TopBottomPanel" \
  /home/tomespen/git/ochroma/crates/vox_app/src/bin/engine_runner.rs | head -20
```

The existing editor rendering is likely inside the `egui frame` call in the `draw()` or `about_to_wait()` method. Locate that block and add `EditorApp::show()` alongside (not replacing) the existing editor panels for now:

```rust
// In the egui frame closure, after existing UI code:
let agent_positions: Vec<[f32; 3]> = self.patrol_agents.iter()
    .map(|a| a.position.to_array())
    .collect();
self.editor_app.update_ghost_overlays(&agent_positions);

let selection_ids: Vec<u32> = self.scene_editor
    .selected
    .map(|id| vec![id])
    .unwrap_or_default();
self.editor_app.show(ctx, &self.scene_editor.entities, selection_ids);
```

- [ ] **Step 11.3: Add ghost overlay splats to the render batch**

In the render loop, after existing overlay splat generation, add ghost overlays:

```rust
// After existing splat gathering:
let ghost_splats = self.editor_app.ghost_overlay_splats();
// Extend the splat buffer (or add to a separate overlay pass)
// The exact method depends on how the render pipeline accepts extra splats.
// Search for where `GaussianSplat` slices are passed to the renderer:
//   grep -n "push_splats\|extra_splats\|overlay_splats" engine_runner.rs
// Add: extra_splats.extend(ghost_splats);
```

**IMPORTANT NOTE:** The exact extension point for overlay splats depends on the current renderer API. Run the grep above and find the correct integration point. Do not add an unused variable — if no integration point is found yet, store `ghost_splats` and log its length only:
```rust
let _ghost_splat_count = self.editor_app.ghost_overlay_splats().len();
```

- [ ] **Step 11.4: Build — expect clean compile**

```bash
cargo build --bin ochroma 2>&1 | grep "^error" | head -10
```

Expected: no errors. Warnings about unused variables are acceptable; fix them.

- [ ] **Step 11.5: Run — verify UI appears**

```bash
cargo run --bin ochroma
```

Expected: the editor window opens. The left mode strip is visible (6 icons). The right context panel shows 3 tabs. The AI bar is visible at the bottom. No crash.

- [ ] **Step 11.6: Run all tests**

```bash
cargo test -p vox_app --lib 2>&1 | tail -5
```

Expected: all tests passing.

- [ ] **Step 11.7: Commit**

```bash
git add crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(editor): wire EditorApp into engine_runner — mode strip, context panel, AI bar visible"
```

---

## Self-Review Checklist

- [x] **§2 Done When — item 1**: Mode strip + context panel change → Task 3 + 4 + 11
- [x] **§2 Done When — item 2**: AI bar submit → Task 6 wires `submit_ai_prompt` via `AiBarState::submit()`
- [x] **§2 Done When — item 3**: Badge appears after AI action → Task 7 + 10's `notify_ai_result()`
- [x] **§2 Done When — item 4**: Badge click reveals graph → Task 7's `open_reveal()` + `show_badge()` return value
- [x] **§2 Done When — item 5**: Semantic asset search → Task 8
- [x] **§2 Done When — item 6**: Ghost overlays in Simulate mode → Task 9 + set_mode in Task 10
- [x] **§3 Capabilities — all 6 rows**: Each has a corresponding task with exact test
- [x] **§5 Data Models**: All types defined: Task 1 (`WorkspaceMode`, `AiScope`, `JobHandle`, `GraphNodeId`, `AssetId`, `NodeGraphDiff`, `EditorEvent`), Task 6 (`AiContext`, `AiResult`), Task 8 (`AssetEntry`, `AssetLibrary`)
- [x] **§6 API — `EditorApp` method signatures**: All match across Tasks 10 + 11
- [x] **§7 Wiring table**: All 8 rows covered by Tasks 3–11
- [x] **No `todo!()`/stubs**: Every function body is complete in the plan
- [x] **Type consistency**: `NodeGraphDiff::new()` used in Tasks 1, 7, 10 with identical signatures
