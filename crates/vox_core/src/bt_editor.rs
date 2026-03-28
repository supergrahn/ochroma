use serde::{Deserialize, Serialize};

/// A visual behavior tree definition (editor-friendly format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTDefinition {
    pub name: String,
    pub root: BTEditorNode,
    pub blackboard_keys: Vec<BlackboardKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTEditorNode {
    pub id: u32,
    pub node_type: BTEditorNodeType,
    pub position: [f32; 2],
    pub children: Vec<BTEditorNode>,
    pub comment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BTEditorNodeType {
    // Composites
    Sequence,
    Selector,
    Parallel { required_successes: u32 },
    // Decorators
    Inverter,
    Repeater { count: Option<u32> },
    Cooldown { seconds: f32 },
    Timeout { seconds: f32 },
    AlwaysSucceed,
    AlwaysFail,
    // Conditions (leaf)
    IsInRange { key: String, range: f32 },
    HasLineOfSight { key: String },
    IsHealthBelow { threshold: f32 },
    BlackboardCheck { key: String, expected: String },
    // Actions (leaf)
    MoveTo { key: String },
    Attack { key: String },
    Flee { key: String },
    Patrol { waypoints_key: String },
    Wait { seconds: f32 },
    PlayAnimation { clip: String },
    SetBlackboard { key: String, value: String },
    PlaySound { clip: String },
    Custom { action_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackboardKey {
    pub name: String,
    pub key_type: BlackboardType,
    pub default_value: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BlackboardType {
    Entity,
    Vec3,
    Float,
    Bool,
    String,
}

impl BTEditorNode {
    fn count(&self) -> usize {
        1 + self.children.iter().map(|c| c.count()).sum::<usize>()
    }

    fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }
}

impl BTDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            root: BTEditorNode {
                id: 0,
                node_type: BTEditorNodeType::Sequence,
                position: [0.0, 0.0],
                children: Vec::new(),
                comment: String::new(),
            },
            blackboard_keys: Vec::new(),
        }
    }

    pub fn count_nodes(&self) -> usize {
        self.root.count()
    }

    pub fn max_depth(&self) -> usize {
        self.root.depth()
    }

    pub fn save_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn load_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_tree() {
        let bt = BTDefinition::new("test_tree");
        assert_eq!(bt.name, "test_tree");
        assert_eq!(bt.root.id, 0);
    }

    #[test]
    fn count_nodes() {
        let mut bt = BTDefinition::new("patrol_ai");
        bt.root.children.push(BTEditorNode {
            id: 1,
            node_type: BTEditorNodeType::MoveTo {
                key: "target".into(),
            },
            position: [100.0, 50.0],
            children: Vec::new(),
            comment: String::new(),
        });
        bt.root.children.push(BTEditorNode {
            id: 2,
            node_type: BTEditorNodeType::Attack {
                key: "enemy".into(),
            },
            position: [200.0, 50.0],
            children: Vec::new(),
            comment: String::new(),
        });
        assert_eq!(bt.count_nodes(), 3);
    }

    #[test]
    fn max_depth() {
        let mut bt = BTDefinition::new("deep_tree");
        bt.root.children.push(BTEditorNode {
            id: 1,
            node_type: BTEditorNodeType::Selector,
            position: [0.0, 0.0],
            children: vec![BTEditorNode {
                id: 2,
                node_type: BTEditorNodeType::Wait { seconds: 1.0 },
                position: [0.0, 0.0],
                children: Vec::new(),
                comment: String::new(),
            }],
            comment: String::new(),
        });
        assert_eq!(bt.max_depth(), 3);
    }

    #[test]
    fn json_round_trip() {
        let mut bt = BTDefinition::new("round_trip");
        bt.root.children.push(BTEditorNode {
            id: 1,
            node_type: BTEditorNodeType::Patrol {
                waypoints_key: "wp".into(),
            },
            position: [10.0, 20.0],
            children: Vec::new(),
            comment: "patrol node".into(),
        });
        let json = bt.save_json().unwrap();
        let loaded = BTDefinition::load_json(&json).unwrap();
        assert_eq!(loaded.name, "round_trip");
        assert_eq!(loaded.count_nodes(), 2);
    }

    #[test]
    fn blackboard_keys_stored() {
        let mut bt = BTDefinition::new("bb_test");
        bt.blackboard_keys.push(BlackboardKey {
            name: "target_entity".into(),
            key_type: BlackboardType::Entity,
            default_value: String::new(),
        });
        bt.blackboard_keys.push(BlackboardKey {
            name: "health".into(),
            key_type: BlackboardType::Float,
            default_value: "100.0".into(),
        });
        assert_eq!(bt.blackboard_keys.len(), 2);
        assert_eq!(bt.blackboard_keys[0].name, "target_entity");
        assert!(matches!(bt.blackboard_keys[1].key_type, BlackboardType::Float));
    }
}
