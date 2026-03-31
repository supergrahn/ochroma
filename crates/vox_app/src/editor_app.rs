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
