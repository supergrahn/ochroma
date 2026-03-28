use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A visual script graph — nodes connected by wires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualGraph {
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub connections: Vec<Connection>,
}

/// A node in the visual script graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: u32,
    pub node_type: NodeType,
    pub position: [f32; 2],
    pub inputs: Vec<Pin>,
    pub outputs: Vec<Pin>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pin {
    pub name: String,
    pub pin_type: PinType,
    pub default_value: Option<PinValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinType {
    Flow,
    Float,
    Vec3,
    Bool,
    String,
    Entity,
    Any,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PinValue {
    Float(f32),
    Vec3([f32; 3]),
    Bool(bool),
    String(String),
    Entity(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    // Events (entry points)
    EventStart,
    EventUpdate,
    EventCollision,
    EventInput { action: String },
    // Flow control
    Branch,
    ForLoop { count: u32 },
    Delay { seconds: f32 },
    Sequence,
    // Math
    Add,
    Subtract,
    Multiply,
    Divide,
    Clamp,
    Lerp,
    Random,
    // Comparison
    Equal,
    NotEqual,
    Greater,
    Less,
    And,
    Or,
    Not,
    // Entity
    GetPosition,
    SetPosition,
    GetRotation,
    SetRotation,
    Spawn { asset: String },
    Destroy,
    FindByTag { tag: String },
    // Physics
    Raycast,
    ApplyForce,
    IsGrounded,
    // Audio
    PlaySound { clip: String },
    StopSound,
    // Variables
    GetVariable { name: String },
    SetVariable { name: String },
    // Output
    Print { message: String },
    // Custom
    Custom { name: String, code: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from_node: u32,
    pub from_pin: String,
    pub to_node: u32,
    pub to_pin: String,
}

/// Execution context for graph evaluation.
pub struct GraphContext {
    pub variables: HashMap<String, PinValue>,
    pub entity_id: u32,
    pub dt: f32,
    pub time: f32,
}

/// Actions produced by graph execution.
#[derive(Debug, Clone)]
pub enum GraphAction {
    SetPosition(u32, [f32; 3]),
    Spawn(String, [f32; 3]),
    Destroy(u32),
    PlaySound(String),
    Print(String),
}

impl VisualGraph {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node_type: NodeType, position: [f32; 2]) -> u32 {
        let id = self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        let (inputs, outputs) = default_pins_for(&node_type);
        self.nodes.push(GraphNode {
            id,
            node_type,
            position,
            inputs,
            outputs,
        });
        id
    }

    pub fn connect(&mut self, from_node: u32, from_pin: &str, to_node: u32, to_pin: &str) {
        self.connections.push(Connection {
            from_node,
            from_pin: from_pin.to_string(),
            to_node,
            to_pin: to_pin.to_string(),
        });
    }

    pub fn remove_node(&mut self, id: u32) {
        self.nodes.retain(|n| n.id != id);
        self.connections
            .retain(|c| c.from_node != id && c.to_node != id);
    }

    pub fn remove_connection(&mut self, from_node: u32, from_pin: &str) {
        self.connections
            .retain(|c| !(c.from_node == from_node && c.from_pin == from_pin));
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    pub fn load_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Execute the graph starting from EventStart.
    pub fn execute(&self, context: &mut GraphContext) -> Vec<GraphAction> {
        let mut actions = Vec::new();
        for node in &self.nodes {
            if matches!(node.node_type, NodeType::EventStart) {
                self.execute_node(node.id, context, &mut actions);
            }
        }
        actions
    }

    fn execute_node(&self, node_id: u32, ctx: &mut GraphContext, actions: &mut Vec<GraphAction>) {
        let node = match self.nodes.iter().find(|n| n.id == node_id) {
            Some(n) => n,
            None => return,
        };
        match &node.node_type {
            NodeType::EventStart | NodeType::EventUpdate => {
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::Print { message } => {
                actions.push(GraphAction::Print(message.clone()));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::SetPosition => {
                if let Some(PinValue::Vec3(pos)) = self.read_input(node_id, "position", ctx) {
                    actions.push(GraphAction::SetPosition(ctx.entity_id, pos));
                }
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::Spawn { asset } => {
                let pos = self
                    .read_input(node_id, "position", ctx)
                    .and_then(|v| {
                        if let PinValue::Vec3(p) = v {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .unwrap_or([0.0; 3]);
                actions.push(GraphAction::Spawn(asset.clone(), pos));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::Branch => {
                let cond = self
                    .read_input(node_id, "condition", ctx)
                    .and_then(|v| {
                        if let PinValue::Bool(b) = v {
                            Some(b)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(false);
                let pin = if cond { "true" } else { "false" };
                for conn in &self.connections {
                    if conn.from_node == node_id && conn.from_pin == pin {
                        self.execute_node(conn.to_node, ctx, actions);
                    }
                }
            }
            NodeType::SetVariable { name } => {
                if let Some(val) = self.read_input(node_id, "value", ctx) {
                    ctx.variables.insert(name.clone(), val);
                }
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::GetVariable { .. } => {
                // Output is handled by read_input on connected nodes
            }
            NodeType::Destroy => {
                actions.push(GraphAction::Destroy(ctx.entity_id));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::PlaySound { clip } => {
                actions.push(GraphAction::PlaySound(clip.clone()));
                self.follow_flow(node_id, ctx, actions);
            }
            _ => {
                self.follow_flow(node_id, ctx, actions);
            }
        }
    }

    fn follow_flow(&self, node_id: u32, ctx: &mut GraphContext, actions: &mut Vec<GraphAction>) {
        for conn in &self.connections {
            if conn.from_node == node_id && conn.from_pin == "flow_out" {
                self.execute_node(conn.to_node, ctx, actions);
            }
        }
    }

    fn read_input(&self, node_id: u32, pin_name: &str, ctx: &GraphContext) -> Option<PinValue> {
        // Find connection to this input
        for conn in &self.connections {
            if conn.to_node == node_id && conn.to_pin == pin_name {
                let src = self.nodes.iter().find(|n| n.id == conn.from_node)?;
                return match &src.node_type {
                    NodeType::GetVariable { name } => ctx.variables.get(name).cloned(),
                    _ => None,
                };
            }
        }
        // Use default value from pin
        let node = self.nodes.iter().find(|n| n.id == node_id)?;
        node.inputs
            .iter()
            .find(|p| p.name == pin_name)?
            .default_value
            .clone()
    }
}

fn default_pins_for(node_type: &NodeType) -> (Vec<Pin>, Vec<Pin>) {
    match node_type {
        NodeType::EventStart | NodeType::EventUpdate | NodeType::EventCollision => (
            vec![],
            vec![Pin {
                name: "flow_out".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
        ),
        NodeType::Branch => (
            vec![
                Pin {
                    name: "flow_in".to_string(),
                    pin_type: PinType::Flow,
                    default_value: None,
                },
                Pin {
                    name: "condition".to_string(),
                    pin_type: PinType::Bool,
                    default_value: Some(PinValue::Bool(false)),
                },
            ],
            vec![
                Pin {
                    name: "true".to_string(),
                    pin_type: PinType::Flow,
                    default_value: None,
                },
                Pin {
                    name: "false".to_string(),
                    pin_type: PinType::Flow,
                    default_value: None,
                },
            ],
        ),
        NodeType::Print { .. } => (
            vec![Pin {
                name: "flow_in".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
            vec![Pin {
                name: "flow_out".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
        ),
        NodeType::SetPosition => (
            vec![
                Pin {
                    name: "flow_in".to_string(),
                    pin_type: PinType::Flow,
                    default_value: None,
                },
                Pin {
                    name: "position".to_string(),
                    pin_type: PinType::Vec3,
                    default_value: Some(PinValue::Vec3([0.0; 3])),
                },
            ],
            vec![Pin {
                name: "flow_out".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
        ),
        NodeType::SetVariable { .. } => (
            vec![
                Pin {
                    name: "flow_in".to_string(),
                    pin_type: PinType::Flow,
                    default_value: None,
                },
                Pin {
                    name: "value".to_string(),
                    pin_type: PinType::Any,
                    default_value: None,
                },
            ],
            vec![Pin {
                name: "flow_out".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
        ),
        NodeType::GetVariable { .. } => (
            vec![],
            vec![Pin {
                name: "value".to_string(),
                pin_type: PinType::Any,
                default_value: None,
            }],
        ),
        _ => (
            vec![Pin {
                name: "flow_in".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
            vec![Pin {
                name: "flow_out".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph_and_add_nodes() {
        let mut graph = VisualGraph::new("TestGraph");
        assert_eq!(graph.name, "TestGraph");
        assert_eq!(graph.node_count(), 0);

        let id1 = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let id2 = graph.add_node(NodeType::Print { message: "hello".into() }, [100.0, 0.0]);
        assert_eq!(graph.node_count(), 2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_add_connections() {
        let mut graph = VisualGraph::new("ConnTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let print = graph.add_node(NodeType::Print { message: "hi".into() }, [100.0, 0.0]);
        graph.connect(start, "flow_out", print, "flow_in");
        assert_eq!(graph.connection_count(), 1);

        graph.remove_connection(start, "flow_out");
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn test_execute_prints() {
        let mut graph = VisualGraph::new("ExecTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let p1 = graph.add_node(NodeType::Print { message: "first".into() }, [100.0, 0.0]);
        let p2 = graph.add_node(NodeType::Print { message: "second".into() }, [200.0, 0.0]);
        graph.connect(start, "flow_out", p1, "flow_in");
        graph.connect(p1, "flow_out", p2, "flow_in");

        let mut ctx = GraphContext {
            variables: HashMap::new(),
            entity_id: 1,
            dt: 0.016,
            time: 0.0,
        };
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 2);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "first"));
        assert!(matches!(&actions[1], GraphAction::Print(m) if m == "second"));
    }

    #[test]
    fn test_branch_true_false() {
        let mut graph = VisualGraph::new("BranchTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let get_var = graph.add_node(NodeType::GetVariable { name: "flag".into() }, [50.0, 50.0]);
        let branch = graph.add_node(NodeType::Branch, [100.0, 0.0]);
        let print_true = graph.add_node(NodeType::Print { message: "yes".into() }, [200.0, -50.0]);
        let print_false = graph.add_node(NodeType::Print { message: "no".into() }, [200.0, 50.0]);

        graph.connect(start, "flow_out", branch, "flow_in");
        graph.connect(get_var, "value", branch, "condition");
        graph.connect(branch, "true", print_true, "flow_in");
        graph.connect(branch, "false", print_false, "flow_in");

        // Test with true
        let mut ctx = GraphContext {
            variables: HashMap::from([("flag".to_string(), PinValue::Bool(true))]),
            entity_id: 1,
            dt: 0.016,
            time: 0.0,
        };
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "yes"));

        // Test with false
        ctx.variables.insert("flag".to_string(), PinValue::Bool(false));
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "no"));
    }

    #[test]
    fn test_variable_set_get() {
        let mut graph = VisualGraph::new("VarTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let set_var = graph.add_node(NodeType::SetVariable { name: "score".into() }, [100.0, 0.0]);
        // Connect a GetVariable as the value source
        let get_var = graph.add_node(NodeType::GetVariable { name: "input_val".into() }, [50.0, 50.0]);

        graph.connect(start, "flow_out", set_var, "flow_in");
        graph.connect(get_var, "value", set_var, "value");

        let mut ctx = GraphContext {
            variables: HashMap::from([("input_val".to_string(), PinValue::Float(42.0))]),
            entity_id: 1,
            dt: 0.016,
            time: 0.0,
        };
        graph.execute(&mut ctx);
        assert!(matches!(ctx.variables.get("score"), Some(PinValue::Float(v)) if (*v - 42.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_json_round_trip() {
        let mut graph = VisualGraph::new("SerdeTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let print = graph.add_node(NodeType::Print { message: "hello".into() }, [100.0, 0.0]);
        graph.connect(start, "flow_out", print, "flow_in");

        let json = graph.save_json().unwrap();
        let loaded = VisualGraph::load_json(&json).unwrap();
        assert_eq!(loaded.name, "SerdeTest");
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.connection_count(), 1);
    }
}
