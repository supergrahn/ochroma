use vox_net::crdt::*;

#[test]
fn local_operations_recorded() {
    let mut log = OperationLog::new(1, 1000);
    log.apply_local(0, "position", OpType::Set, vec![1, 2, 3]);
    assert_eq!(log.operation_count(), 1);
    assert!(log.get(0, "position").is_some());
}

#[test]
fn last_writer_wins() {
    let mut log = OperationLog::new(1, 1000);
    log.apply_local(0, "position", OpType::Set, vec![1]);
    log.apply_local(0, "position", OpType::Set, vec![2]);
    assert_eq!(log.get(0, "position").unwrap(), &vec![2u8]);
}

#[test]
fn remote_operation_merges() {
    let mut log1 = OperationLog::new(1, 1000);
    let mut log2 = OperationLog::new(2, 1000);

    let op1 = log1.apply_local(0, "position", OpType::Set, vec![10]);
    let op2 = log2.apply_local(0, "position", OpType::Set, vec![20]);

    // Apply each other's operations
    log1.apply_remote(op2.clone());
    log2.apply_remote(op1.clone());

    // Both should converge to the same value (higher timestamp or higher author wins)
    assert_eq!(log1.get(0, "position"), log2.get(0, "position"));
}

#[test]
fn replay_returns_recent() {
    let mut log = OperationLog::new(1, 1000);
    for i in 0..10 {
        log.apply_local(i, "test", OpType::Set, vec![i as u8]);
    }
    let recent = log.replay(3);
    assert_eq!(recent.len(), 3);
}

#[test]
fn lamport_clock_advances() {
    let mut clock = LamportClock::new(1);
    assert_eq!(clock.tick(), 1);
    assert_eq!(clock.tick(), 2);
    clock.merge(10);
    assert_eq!(clock.tick(), 12); // max(2, 10) + 1 = 11, then tick = 12
}
