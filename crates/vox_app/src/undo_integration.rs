use vox_core::undo::{UndoStack, UndoEntry};

/// Wraps UndoStack with game-specific action tracking.
pub struct GameUndoSystem {
    pub stack: UndoStack,
}

impl GameUndoSystem {
    pub fn new() -> Self {
        Self { stack: UndoStack::new(100) }
    }

    /// Record a placement action.
    pub fn record_placement(&mut self, instance_id: u32, position: [f32; 3]) {
        self.stack.push(UndoEntry {
            description: format!("Place building at ({:.0}, {:.0}, {:.0})", position[0], position[1], position[2]),
            undo_data: instance_id.to_le_bytes().to_vec(),
            redo_data: Vec::new(),
        });
    }

    /// Record a zone action.
    pub fn record_zone(&mut self, plot_id: u32, zone_type: &str) {
        self.stack.push(UndoEntry {
            description: format!("Zone plot {} as {}", plot_id, zone_type),
            undo_data: plot_id.to_le_bytes().to_vec(),
            redo_data: Vec::new(),
        });
    }

    /// Record a road action.
    pub fn record_road(&mut self, segment_id: u32) {
        self.stack.push(UndoEntry {
            description: format!("Build road segment {}", segment_id),
            undo_data: segment_id.to_le_bytes().to_vec(),
            redo_data: Vec::new(),
        });
    }

    pub fn undo(&mut self) -> Option<UndoEntry> {
        let entry = self.stack.undo();
        if let Some(ref e) = entry {
            println!("[ochroma] Undo: {}", e.description);
        }
        entry
    }

    pub fn redo(&mut self) -> Option<UndoEntry> {
        let entry = self.stack.redo();
        if let Some(ref e) = entry {
            println!("[ochroma] Redo: {}", e.description);
        }
        entry
    }
}
