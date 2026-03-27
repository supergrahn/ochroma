/// A reversible command.
pub trait Command: std::fmt::Debug {
    fn execute(&self) -> Vec<u8>; // Returns serialized state change
    fn undo_data(&self) -> Vec<u8>; // Returns data needed to undo
    fn description(&self) -> &str;
}

/// Undo/redo stack using command pattern.
pub struct UndoStack {
    undo_stack: Vec<UndoEntry>,
    redo_stack: Vec<UndoEntry>,
    max_depth: usize,
}

#[derive(Debug, Clone)]
pub struct UndoEntry {
    pub description: String,
    pub undo_data: Vec<u8>,
    pub redo_data: Vec<u8>,
}

impl UndoStack {
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
        }
    }

    pub fn push(&mut self, entry: UndoEntry) {
        self.redo_stack.clear(); // New action invalidates redo history
        self.undo_stack.push(entry);
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    pub fn undo(&mut self) -> Option<UndoEntry> {
        if let Some(entry) = self.undo_stack.pop() {
            self.redo_stack.push(entry.clone());
            Some(entry)
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<UndoEntry> {
        if let Some(entry) = self.redo_stack.pop() {
            self.undo_stack.push(entry.clone());
            Some(entry)
        } else {
            None
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}
