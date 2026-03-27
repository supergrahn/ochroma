use std::collections::HashMap;

/// Status returned by behavior tree node evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTStatus {
    Success,
    Failure,
    Running,
}

/// A node in the behavior tree.
pub enum BTNode {
    /// Run children in order until one fails.
    Sequence(Vec<BTNode>),
    /// Run children in order until one succeeds.
    Selector(Vec<BTNode>),
    /// Run child repeatedly.
    Repeater {
        child: Box<BTNode>,
        count: Option<u32>,
    },
    /// Invert child result.
    Inverter(Box<BTNode>),
    /// Always succeed.
    AlwaysSucceed(Box<BTNode>),
    /// Leaf: execute an action.
    Action(String),
    /// Leaf: check a condition.
    Condition(String),
}

/// A behavior tree with a root node.
pub struct BehaviorTree {
    pub root: BTNode,
}

/// Execution context for behavior tree evaluation.
pub struct BTContext {
    pub action_results: HashMap<String, BTStatus>,
    pub condition_results: HashMap<String, bool>,
    pub blackboard: HashMap<String, String>,
}

impl BTContext {
    pub fn new() -> Self {
        Self {
            action_results: HashMap::new(),
            condition_results: HashMap::new(),
            blackboard: HashMap::new(),
        }
    }
}

impl Default for BTContext {
    fn default() -> Self {
        Self::new()
    }
}

impl BehaviorTree {
    pub fn new(root: BTNode) -> Self {
        Self { root }
    }

    pub fn tick(&self, ctx: &BTContext) -> BTStatus {
        evaluate(&self.root, ctx)
    }
}

fn evaluate(node: &BTNode, ctx: &BTContext) -> BTStatus {
    match node {
        BTNode::Sequence(children) => {
            for child in children {
                match evaluate(child, ctx) {
                    BTStatus::Failure => return BTStatus::Failure,
                    BTStatus::Running => return BTStatus::Running,
                    BTStatus::Success => continue,
                }
            }
            BTStatus::Success
        }
        BTNode::Selector(children) => {
            for child in children {
                match evaluate(child, ctx) {
                    BTStatus::Success => return BTStatus::Success,
                    BTStatus::Running => return BTStatus::Running,
                    BTStatus::Failure => continue,
                }
            }
            BTStatus::Failure
        }
        BTNode::Inverter(child) => match evaluate(child, ctx) {
            BTStatus::Success => BTStatus::Failure,
            BTStatus::Failure => BTStatus::Success,
            BTStatus::Running => BTStatus::Running,
        },
        BTNode::AlwaysSucceed(child) => {
            let _ = evaluate(child, ctx);
            BTStatus::Success
        }
        BTNode::Repeater { child, count: _ } => {
            // Simplified: just run once per tick
            evaluate(child, ctx)
        }
        BTNode::Action(name) => ctx
            .action_results
            .get(name)
            .copied()
            .unwrap_or(BTStatus::Failure),
        BTNode::Condition(name) => {
            if *ctx.condition_results.get(name).unwrap_or(&false) {
                BTStatus::Success
            } else {
                BTStatus::Failure
            }
        }
    }
}

/// Builder helpers
pub fn sequence(children: Vec<BTNode>) -> BTNode {
    BTNode::Sequence(children)
}
pub fn selector(children: Vec<BTNode>) -> BTNode {
    BTNode::Selector(children)
}
pub fn action(name: &str) -> BTNode {
    BTNode::Action(name.to_string())
}
pub fn condition(name: &str) -> BTNode {
    BTNode::Condition(name.to_string())
}
pub fn inverter(child: BTNode) -> BTNode {
    BTNode::Inverter(Box::new(child))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_stops_on_failure() {
        let tree = BehaviorTree::new(sequence(vec![
            action("walk"),
            action("fail_action"),
            action("attack"),
        ]));
        let mut ctx = BTContext::new();
        ctx.action_results
            .insert("walk".into(), BTStatus::Success);
        ctx.action_results
            .insert("fail_action".into(), BTStatus::Failure);
        ctx.action_results
            .insert("attack".into(), BTStatus::Success);

        assert_eq!(tree.tick(&ctx), BTStatus::Failure);
    }

    #[test]
    fn sequence_succeeds_when_all_succeed() {
        let tree = BehaviorTree::new(sequence(vec![action("a"), action("b")]));
        let mut ctx = BTContext::new();
        ctx.action_results.insert("a".into(), BTStatus::Success);
        ctx.action_results.insert("b".into(), BTStatus::Success);
        assert_eq!(tree.tick(&ctx), BTStatus::Success);
    }

    #[test]
    fn selector_stops_on_success() {
        let tree = BehaviorTree::new(selector(vec![
            action("fail_action"),
            action("succeed_action"),
            action("never_reached"),
        ]));
        let mut ctx = BTContext::new();
        ctx.action_results
            .insert("fail_action".into(), BTStatus::Failure);
        ctx.action_results
            .insert("succeed_action".into(), BTStatus::Success);
        // never_reached not in context => Failure, but should never be evaluated
        assert_eq!(tree.tick(&ctx), BTStatus::Success);
    }

    #[test]
    fn selector_fails_when_all_fail() {
        let tree = BehaviorTree::new(selector(vec![action("a"), action("b")]));
        let mut ctx = BTContext::new();
        ctx.action_results.insert("a".into(), BTStatus::Failure);
        ctx.action_results.insert("b".into(), BTStatus::Failure);
        assert_eq!(tree.tick(&ctx), BTStatus::Failure);
    }

    #[test]
    fn inverter_flips_success_to_failure() {
        let tree = BehaviorTree::new(inverter(action("ok")));
        let mut ctx = BTContext::new();
        ctx.action_results.insert("ok".into(), BTStatus::Success);
        assert_eq!(tree.tick(&ctx), BTStatus::Failure);
    }

    #[test]
    fn inverter_flips_failure_to_success() {
        let tree = BehaviorTree::new(inverter(action("bad")));
        let mut ctx = BTContext::new();
        ctx.action_results.insert("bad".into(), BTStatus::Failure);
        assert_eq!(tree.tick(&ctx), BTStatus::Success);
    }

    #[test]
    fn condition_reads_from_context() {
        let tree = BehaviorTree::new(condition("has_ammo"));
        let mut ctx = BTContext::new();
        ctx.condition_results.insert("has_ammo".into(), true);
        assert_eq!(tree.tick(&ctx), BTStatus::Success);

        ctx.condition_results.insert("has_ammo".into(), false);
        assert_eq!(tree.tick(&ctx), BTStatus::Failure);
    }

    #[test]
    fn action_reads_from_context() {
        let tree = BehaviorTree::new(action("shoot"));
        let mut ctx = BTContext::new();
        ctx.action_results
            .insert("shoot".into(), BTStatus::Running);
        assert_eq!(tree.tick(&ctx), BTStatus::Running);
    }

    #[test]
    fn nested_tree_evaluates_correctly() {
        // Selector: try (sequence: check ammo AND shoot), else (reload)
        let tree = BehaviorTree::new(selector(vec![
            sequence(vec![condition("has_ammo"), action("shoot")]),
            action("reload"),
        ]));

        // Case 1: has ammo => shoot succeeds
        let mut ctx = BTContext::new();
        ctx.condition_results.insert("has_ammo".into(), true);
        ctx.action_results
            .insert("shoot".into(), BTStatus::Success);
        ctx.action_results
            .insert("reload".into(), BTStatus::Success);
        assert_eq!(tree.tick(&ctx), BTStatus::Success);

        // Case 2: no ammo => sequence fails, selector tries reload
        ctx.condition_results.insert("has_ammo".into(), false);
        assert_eq!(tree.tick(&ctx), BTStatus::Success); // reload succeeds
    }
}
