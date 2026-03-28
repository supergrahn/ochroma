/// Drag-and-drop state machine for the editor.

#[derive(Debug, Clone)]
pub struct DragDropState {
    pub active: bool,
    pub payload: Option<DragPayload>,
    pub source: DragSource,
    pub mouse_pos: [f32; 2],
}

#[derive(Debug, Clone)]
pub enum DragPayload {
    Asset { path: String, asset_type: String },
    Entity { id: u32 },
    Prefab { name: String, path: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragSource {
    None,
    ContentBrowser,
    Outliner,
    External,
}

/// What happens when a drag ends.
#[derive(Debug, Clone)]
pub enum DropAction {
    /// Drop asset into viewport -- spawn entity at world position.
    SpawnAssetAtPosition { asset_path: String, world_position: [f32; 3] },
    /// Drop entity onto another -- reparent.
    ReparentEntity { entity_id: u32, new_parent: u32 },
    /// Drop prefab -- instantiate.
    InstantiatePrefab { prefab_path: String, world_position: [f32; 3] },
    /// Cancelled.
    Cancelled,
}

impl DragDropState {
    pub fn new() -> Self {
        Self {
            active: false,
            payload: None,
            source: DragSource::None,
            mouse_pos: [0.0; 2],
        }
    }

    pub fn begin(&mut self, payload: DragPayload, source: DragSource) {
        self.active = true;
        self.payload = Some(payload);
        self.source = source;
    }

    pub fn update_mouse(&mut self, x: f32, y: f32) {
        self.mouse_pos = [x, y];
    }

    pub fn end_drop(&mut self, world_position: [f32; 3]) -> DropAction {
        self.active = false;
        let action = match self.payload.take() {
            Some(DragPayload::Asset { path, .. }) => {
                DropAction::SpawnAssetAtPosition { asset_path: path, world_position }
            }
            Some(DragPayload::Prefab { path, .. }) => {
                DropAction::InstantiatePrefab { prefab_path: path, world_position }
            }
            Some(DragPayload::Entity { .. }) => DropAction::Cancelled,
            None => DropAction::Cancelled,
        };
        self.source = DragSource::None;
        action
    }

    pub fn cancel(&mut self) {
        self.active = false;
        self.payload = None;
        self.source = DragSource::None;
    }

    pub fn is_dragging(&self) -> bool {
        self.active
    }
}

impl Default for DragDropState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_end_drag() {
        let mut state = DragDropState::new();
        assert!(!state.is_dragging());
        state.begin(
            DragPayload::Asset { path: "mesh.vxm".into(), asset_type: "mesh".into() },
            DragSource::ContentBrowser,
        );
        assert!(state.is_dragging());
        assert_eq!(state.source, DragSource::ContentBrowser);
        let action = state.end_drop([1.0, 2.0, 3.0]);
        assert!(!state.is_dragging());
        match action {
            DropAction::SpawnAssetAtPosition { asset_path, world_position } => {
                assert_eq!(asset_path, "mesh.vxm");
                assert_eq!(world_position, [1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected SpawnAssetAtPosition"),
        }
    }

    #[test]
    fn test_asset_drop_spawns() {
        let mut state = DragDropState::new();
        state.begin(
            DragPayload::Asset { path: "textures/brick.png".into(), asset_type: "texture".into() },
            DragSource::ContentBrowser,
        );
        let action = state.end_drop([10.0, 0.0, -5.0]);
        match action {
            DropAction::SpawnAssetAtPosition { asset_path, .. } => {
                assert_eq!(asset_path, "textures/brick.png");
            }
            _ => panic!("Expected SpawnAssetAtPosition"),
        }
    }

    #[test]
    fn test_cancel_clears() {
        let mut state = DragDropState::new();
        state.begin(
            DragPayload::Entity { id: 42 },
            DragSource::Outliner,
        );
        assert!(state.is_dragging());
        state.cancel();
        assert!(!state.is_dragging());
        assert!(state.payload.is_none());
        assert_eq!(state.source, DragSource::None);
    }

    #[test]
    fn test_mouse_position_tracked() {
        let mut state = DragDropState::new();
        state.begin(
            DragPayload::Asset { path: "test.vxm".into(), asset_type: "mesh".into() },
            DragSource::ContentBrowser,
        );
        state.update_mouse(100.0, 200.0);
        assert_eq!(state.mouse_pos, [100.0, 200.0]);
        state.update_mouse(300.0, 400.0);
        assert_eq!(state.mouse_pos, [300.0, 400.0]);
    }
}
