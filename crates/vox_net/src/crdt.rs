use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A CRDT operation on the world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtOperation {
    pub id: u64,
    pub timestamp: u64, // Lamport clock
    pub author: u32,    // player ID
    pub entity_id: u32,
    pub component: String,
    pub op_type: OpType,
    pub data: Vec<u8>, // serialised value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpType {
    Set,    // last-writer-wins register
    Add,    // counter increment
    Remove, // tombstone
}

/// Lamport clock for causal ordering.
#[derive(Debug, Clone, Default)]
pub struct LamportClock {
    pub counter: u64,
    pub node_id: u32,
}

impl LamportClock {
    pub fn new(node_id: u32) -> Self {
        Self {
            counter: 0,
            node_id,
        }
    }
    pub fn tick(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }
    pub fn merge(&mut self, other: u64) {
        self.counter = self.counter.max(other) + 1;
    }
}

/// Operation log with conflict resolution.
pub struct OperationLog {
    pub operations: Vec<CrdtOperation>,
    pub clock: LamportClock,
    pub max_operations: usize,
    /// Last-writer-wins register state per (entity, component).
    state: HashMap<(u32, String), CrdtOperation>,
}

impl OperationLog {
    pub fn new(node_id: u32, max_operations: usize) -> Self {
        Self {
            operations: Vec::new(),
            clock: LamportClock::new(node_id),
            max_operations,
            state: HashMap::new(),
        }
    }

    /// Apply a local operation.
    pub fn apply_local(
        &mut self,
        entity_id: u32,
        component: &str,
        op_type: OpType,
        data: Vec<u8>,
    ) -> CrdtOperation {
        let timestamp = self.clock.tick();
        let op = CrdtOperation {
            id: self.operations.len() as u64,
            timestamp,
            author: self.clock.node_id,
            entity_id,
            component: component.to_string(),
            op_type,
            data,
        };
        self.apply_operation(op.clone());
        op
    }

    /// Apply a remote operation (from another node).
    pub fn apply_remote(&mut self, op: CrdtOperation) {
        self.clock.merge(op.timestamp);
        self.apply_operation(op);
    }

    fn apply_operation(&mut self, op: CrdtOperation) {
        let key = (op.entity_id, op.component.clone());
        // Last-writer-wins: higher timestamp wins, tie-break on author ID
        let should_apply = match self.state.get(&key) {
            Some(existing) => {
                op.timestamp > existing.timestamp
                    || (op.timestamp == existing.timestamp && op.author > existing.author)
            }
            None => true,
        };

        if should_apply {
            self.state.insert(key, op.clone());
        }

        self.operations.push(op);
        if self.operations.len() > self.max_operations {
            self.operations.remove(0);
        }
    }

    /// Get the current value for an entity's component.
    pub fn get(&self, entity_id: u32, component: &str) -> Option<&Vec<u8>> {
        self.state
            .get(&(entity_id, component.to_string()))
            .map(|op| &op.data)
    }

    /// Replay last N operations (for time-travel debugging).
    pub fn replay(&self, count: usize) -> &[CrdtOperation] {
        let start = self.operations.len().saturating_sub(count);
        &self.operations[start..]
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}
