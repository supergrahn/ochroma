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
#[derive(Debug)]
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

impl Clone for NodeGraphDiff {
    fn clone(&self) -> Self {
        Self {
            asset_id: self.asset_id,
            added_nodes: self.added_nodes.clone(),
            modified_nodes: self.modified_nodes.clone(),
        }
    }
}

/// Events broadcast on EditorApp's internal bus.
#[derive(Debug, Clone)]
pub enum EditorEvent {
    WorkspaceModeChanged { mode: WorkspaceMode },
    AiActionComplete { diff: Option<NodeGraphDiff> },
}

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
    pub fn update_ghost_overlays(&mut self, agent_positions: &[[f32; 3]]) {
        self.ghost_overlays.update(agent_positions);
    }

    /// Returns ghost overlay splats for the current frame. Empty when overlays are disabled.
    pub fn ghost_overlay_splats(&self) -> Vec<vox_core::types::GaussianSplat> {
        self.ghost_overlays.generate_path_splats()
    }

    /// Render the entire editor layout. Call once per egui frame.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        entities: &[EditorEntity],
        selection_ids: Vec<u32>,
    ) {
        let mode = self.mode_strip.active_mode();

        egui::SidePanel::left("mode_strip")
            .exact_width(48.0)
            .resizable(false)
            .show(ctx, |ui| {
                if let Some(new_mode) = self.mode_strip.show(ui) {
                    self.ghost_overlays.set_enabled(new_mode == WorkspaceMode::Simulate);
                }
            });

        egui::SidePanel::right("context_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                let ng_panel = &mut self.node_graph_panel;
                let scene_tree = &mut self.scene_tree;
                let asset_lib = &self.asset_library;
                let asset_lib_ui = &mut self.asset_library_ui;
                let context_panel = &mut self.context_panel;

                context_panel.show(
                    ui,
                    mode,
                    |ui| {
                        if ng_panel.show_badge(ui) {
                            ng_panel.open_reveal();
                        }
                        ng_panel.show_reveal_panel(ui);
                    },
                    |ui| {
                        scene_tree.show(ui, entities, &mut |_id| {});
                    },
                    |ui| {
                        asset_lib_ui.show(ui, asset_lib, &|_| vec![0.0f32; 3], &mut |_| {});
                    },
                );
            });

        egui::TopBottomPanel::bottom("ai_bar")
            .resizable(true)
            .min_height(36.0)
            .show(ctx, |ui| {
                if let Some(result) = self.ai_bar.tick(self.backend.as_ref()) {
                    self.notify_ai_result(result);
                }
                let backend = self.backend.as_ref();
                let ai_bar = &mut self.ai_bar;
                ai_bar.show(
                    ui,
                    backend,
                    selection_ids.clone(),
                    &mut |_result| { /* result already handled via tick above */ },
                );
            });
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
}
