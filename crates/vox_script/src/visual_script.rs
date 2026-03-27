use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comparison operators for condition nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
}

/// Math operations for MathOp nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MathOperation {
    Add,
    Sub,
    Mul,
    Div,
}

/// Variable access mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VarAccess {
    Get(String),
    Set(String, f64),
}

/// A node in the visual script graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScriptNode {
    /// Trigger that starts execution when a named event fires.
    Event(String),
    /// Evaluates a condition: variable name, comparison op, threshold value.
    Condition(String, ComparisonOp, f64),
    /// Executes a named game action with optional string parameter.
    Action(String, Option<String>),
    /// If/else branch: index of true-path first node, index of false-path first node.
    Branch(usize, usize),
    /// Loop a fixed number of times; body is the next connected node.
    Loop(u32),
    /// Get or set a script variable.
    Variable(VarAccess),
    /// Perform a math operation: (variable_name, operation, operand).
    MathOp(String, MathOperation, f64),
}

/// An edge connecting two nodes in the graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeConnection {
    pub from: usize,
    pub to: usize,
}

/// A visual script is a directed graph of connected nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualScript {
    pub name: String,
    pub nodes: Vec<ScriptNode>,
    pub connections: Vec<NodeConnection>,
}

/// The result of executing a visual script.
#[derive(Debug, Clone, Default)]
pub struct ExecutionResult {
    /// Actions that were triggered during execution, as (action_name, parameter).
    pub triggered_actions: Vec<(String, Option<String>)>,
    /// Final variable state after execution.
    pub variables: HashMap<String, f64>,
}

impl VisualScript {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, node: ScriptNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Connect two nodes by index.
    pub fn connect(&mut self, from: usize, to: usize) {
        self.connections.push(NodeConnection { from, to });
    }

    /// Find all Event nodes that match the given trigger name.
    pub fn find_entry_points(&self, trigger: &str) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| match node {
                ScriptNode::Event(name) if name == trigger => Some(i),
                _ => None,
            })
            .collect()
    }

    /// Get all outgoing connections from a given node index.
    fn outgoing(&self, from: usize) -> Vec<usize> {
        self.connections
            .iter()
            .filter(|c| c.from == from)
            .map(|c| c.to)
            .collect()
    }

    /// Serialise the script graph to JSON.
    pub fn compile_to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialise a script graph from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Execute a visual script starting from all Event nodes matching `trigger`.
/// `variables` provides the initial variable state (e.g. game world values).
pub fn execute_script(
    script: &VisualScript,
    trigger: &str,
    initial_variables: &HashMap<String, f64>,
) -> ExecutionResult {
    let mut result = ExecutionResult {
        triggered_actions: Vec::new(),
        variables: initial_variables.clone(),
    };

    let entry_points = script.find_entry_points(trigger);
    for entry in entry_points {
        let successors = script.outgoing(entry);
        for next in successors {
            walk_node(script, next, &mut result, 0);
        }
    }

    result
}

const MAX_DEPTH: u32 = 1000;

fn walk_node(
    script: &VisualScript,
    node_idx: usize,
    result: &mut ExecutionResult,
    depth: u32,
) {
    if depth > MAX_DEPTH || node_idx >= script.nodes.len() {
        return;
    }

    match &script.nodes[node_idx] {
        ScriptNode::Event(_) => {
            // Event nodes are entry points; if reached mid-graph, just continue.
            for next in script.outgoing(node_idx) {
                walk_node(script, next, result, depth + 1);
            }
        }
        ScriptNode::Action(name, param) => {
            result
                .triggered_actions
                .push((name.clone(), param.clone()));
            for next in script.outgoing(node_idx) {
                walk_node(script, next, result, depth + 1);
            }
        }
        ScriptNode::Condition(var_name, op, threshold) => {
            let val = result.variables.get(var_name).copied().unwrap_or(0.0);
            let passed = match op {
                ComparisonOp::Equal => (val - threshold).abs() < f64::EPSILON,
                ComparisonOp::NotEqual => (val - threshold).abs() >= f64::EPSILON,
                ComparisonOp::GreaterThan => val > *threshold,
                ComparisonOp::LessThan => val < *threshold,
                ComparisonOp::GreaterOrEqual => val >= *threshold,
                ComparisonOp::LessOrEqual => val <= *threshold,
            };
            if passed {
                for next in script.outgoing(node_idx) {
                    walk_node(script, next, result, depth + 1);
                }
            }
        }
        ScriptNode::Branch(true_idx, false_idx) => {
            // Branch expects the previous condition result. We look at the last
            // condition evaluation by checking the incoming connection's condition.
            // For simplicity, Branch always evaluates both paths based on its
            // direct predecessor condition. We use a simple heuristic: check if
            // there's a Condition node that connects to this Branch.
            let true_idx = *true_idx;
            let false_idx = *false_idx;

            // Find condition nodes that connect to this branch and evaluate them.
            // Only Condition predecessors are considered; non-condition predecessors
            // are ignored. If no condition predecessor exists, default to true.
            let condition_results: Vec<bool> = script
                .connections
                .iter()
                .filter(|c| c.to == node_idx)
                .filter_map(|c| {
                    if let Some(ScriptNode::Condition(var, op, threshold)) =
                        script.nodes.get(c.from)
                    {
                        let val = result.variables.get(var).copied().unwrap_or(0.0);
                        Some(match op {
                            ComparisonOp::Equal => (val - threshold).abs() < f64::EPSILON,
                            ComparisonOp::NotEqual => (val - threshold).abs() >= f64::EPSILON,
                            ComparisonOp::GreaterThan => val > *threshold,
                            ComparisonOp::LessThan => val < *threshold,
                            ComparisonOp::GreaterOrEqual => val >= *threshold,
                            ComparisonOp::LessOrEqual => val <= *threshold,
                        })
                    } else {
                        None // skip non-condition predecessors
                    }
                })
                .collect();

            let condition_passed = if condition_results.is_empty() {
                true // no condition predecessor => default true
            } else {
                condition_results.iter().all(|&v| v)
            };

            if condition_passed {
                walk_node(script, true_idx, result, depth + 1);
            } else {
                walk_node(script, false_idx, result, depth + 1);
            }
        }
        ScriptNode::Loop(count) => {
            let count = *count;
            let successors = script.outgoing(node_idx);
            for _ in 0..count {
                for &next in &successors {
                    walk_node(script, next, result, depth + 1);
                }
            }
        }
        ScriptNode::Variable(access) => {
            match access {
                VarAccess::Get(_name) => {
                    // Get just makes the variable available; value is already in the map.
                }
                VarAccess::Set(name, value) => {
                    result.variables.insert(name.clone(), *value);
                }
            }
            for next in script.outgoing(node_idx) {
                walk_node(script, next, result, depth + 1);
            }
        }
        ScriptNode::MathOp(var_name, op, operand) => {
            let current = result.variables.get(var_name).copied().unwrap_or(0.0);
            let new_val = match op {
                MathOperation::Add => current + operand,
                MathOperation::Sub => current - operand,
                MathOperation::Mul => current * operand,
                MathOperation::Div => {
                    if operand.abs() < f64::EPSILON {
                        f64::NAN
                    } else {
                        current / operand
                    }
                }
            };
            result.variables.insert(var_name.clone(), new_val);
            for next in script.outgoing(node_idx) {
                walk_node(script, next, result, depth + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_event_action() {
        let mut script = VisualScript::new("test");
        let event = script.add_node(ScriptNode::Event("BuildingPlaced".into()));
        let action = script.add_node(ScriptNode::Action(
            "spawn_celebration".into(),
            Some("fireworks".into()),
        ));
        script.connect(event, action);

        let vars = HashMap::new();
        let result = execute_script(&script, "BuildingPlaced", &vars);
        assert_eq!(result.triggered_actions.len(), 1);
        assert_eq!(result.triggered_actions[0].0, "spawn_celebration");
    }

    #[test]
    fn test_condition_gates_action() {
        let mut script = VisualScript::new("cond_test");
        let event = script.add_node(ScriptNode::Event("Tick".into()));
        let cond = script.add_node(ScriptNode::Condition(
            "population".into(),
            ComparisonOp::GreaterThan,
            100.0,
        ));
        let action = script.add_node(ScriptNode::Action("celebrate".into(), None));
        script.connect(event, cond);
        script.connect(cond, action);

        // Population below threshold - no action
        let mut vars = HashMap::new();
        vars.insert("population".into(), 50.0);
        let result = execute_script(&script, "Tick", &vars);
        assert!(result.triggered_actions.is_empty());

        // Population above threshold - action fires
        vars.insert("population".into(), 200.0);
        let result = execute_script(&script, "Tick", &vars);
        assert_eq!(result.triggered_actions.len(), 1);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut script = VisualScript::new("json_test");
        script.add_node(ScriptNode::Event("Start".into()));
        script.add_node(ScriptNode::Action("do_thing".into(), None));
        script.connect(0, 1);

        let json = script.compile_to_json().unwrap();
        let restored = VisualScript::from_json(&json).unwrap();
        assert_eq!(restored.name, "json_test");
        assert_eq!(restored.nodes.len(), 2);
        assert_eq!(restored.connections.len(), 1);
    }
}
