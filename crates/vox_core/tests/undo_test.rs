use vox_core::undo::{UndoEntry, UndoStack};

#[test]
fn push_and_undo() {
    let mut stack = UndoStack::new(100);
    stack.push(UndoEntry {
        description: "place building".into(),
        undo_data: vec![1],
        redo_data: vec![2],
    });
    assert!(stack.can_undo());
    let entry = stack.undo().unwrap();
    assert_eq!(entry.description, "place building");
    assert!(!stack.can_undo());
    assert!(stack.can_redo());
}

#[test]
fn undo_then_redo() {
    let mut stack = UndoStack::new(100);
    stack.push(UndoEntry {
        description: "a".into(),
        undo_data: vec![],
        redo_data: vec![],
    });
    stack.undo();
    let entry = stack.redo().unwrap();
    assert_eq!(entry.description, "a");
}

#[test]
fn new_action_clears_redo() {
    let mut stack = UndoStack::new(100);
    stack.push(UndoEntry {
        description: "a".into(),
        undo_data: vec![],
        redo_data: vec![],
    });
    stack.undo();
    assert!(stack.can_redo());
    stack.push(UndoEntry {
        description: "b".into(),
        undo_data: vec![],
        redo_data: vec![],
    });
    assert!(!stack.can_redo());
}

#[test]
fn max_depth_evicts_oldest() {
    let mut stack = UndoStack::new(3);
    for i in 0..5 {
        stack.push(UndoEntry {
            description: format!("{}", i),
            undo_data: vec![],
            redo_data: vec![],
        });
    }
    assert_eq!(stack.undo_count(), 3);
}
