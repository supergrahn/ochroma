use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    pub entity_position: [f32; 3],
    pub entity_rotation: [f32; 4],
    pub grounded: bool,
    pub collision_other: Option<u32>,
    pub input_actions: HashSet<String>,
}

/// Actions produced by graph execution.
#[derive(Debug, Clone)]
pub enum GraphAction {
    SetPosition(u32, [f32; 3]),
    SetRotation(u32, [f32; 4]),
    Spawn(String, [f32; 3]),
    Destroy(u32),
    PlaySound(String),
    StopSound(String),
    Print(String),
    ApplyForce(u32, [f32; 3]),
    Raycast { origin: [f32; 3], direction: [f32; 3] },
    FindByTag(String),
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
            // ── Events ──────────────────────────────────────────────
            NodeType::EventStart | NodeType::EventUpdate => {
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::EventCollision => {
                // Entry point for collision events — just follow flow
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::EventInput { .. } => {
                // Entry point for input events — just follow flow
                self.follow_flow(node_id, ctx, actions);
            }

            // ── Output ──────────────────────────────────────────────
            NodeType::Print { message } => {
                actions.push(GraphAction::Print(message.clone()));
                self.follow_flow(node_id, ctx, actions);
            }

            // ── Entity: Transform ───────────────────────────────────
            NodeType::SetPosition => {
                if let Some(PinValue::Vec3(pos)) = self.read_input(node_id, "position", ctx) {
                    actions.push(GraphAction::SetPosition(ctx.entity_id, pos));
                }
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::GetPosition => {
                // Output is read via read_input — store result in node output cache
                // The value is produced by evaluate_node instead
                // (pure data node, no flow execution needed)
            }
            NodeType::SetRotation => {
                if let Some(PinValue::Vec3(rot)) = self.read_input(node_id, "rotation", ctx) {
                    actions.push(GraphAction::SetRotation(ctx.entity_id, [rot[0], rot[1], rot[2], 1.0]));
                }
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::GetRotation => {
                // Pure data node — output read via evaluate_node
            }

            // ── Entity: Lifecycle ───────────────────────────────────
            NodeType::Spawn { asset } => {
                let pos = self
                    .read_input(node_id, "position", ctx)
                    .and_then(|v| if let PinValue::Vec3(p) = v { Some(p) } else { None })
                    .unwrap_or([0.0; 3]);
                actions.push(GraphAction::Spawn(asset.clone(), pos));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::Destroy => {
                actions.push(GraphAction::Destroy(ctx.entity_id));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::FindByTag { tag } => {
                actions.push(GraphAction::FindByTag(tag.clone()));
                // Pure data node, but also has flow
            }

            // ── Audio ───────────────────────────────────────────────
            NodeType::PlaySound { clip } => {
                actions.push(GraphAction::PlaySound(clip.clone()));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::StopSound => {
                let clip = self
                    .read_input(node_id, "clip", ctx)
                    .and_then(|v| if let PinValue::String(s) = v { Some(s) } else { None })
                    .unwrap_or_default();
                actions.push(GraphAction::StopSound(clip));
                self.follow_flow(node_id, ctx, actions);
            }

            // ── Flow Control ────────────────────────────────────────
            NodeType::Branch => {
                let cond = self
                    .read_input(node_id, "condition", ctx)
                    .and_then(|v| if let PinValue::Bool(b) = v { Some(b) } else { None })
                    .unwrap_or(false);
                let pin = if cond { "true" } else { "false" };
                for conn in &self.connections {
                    if conn.from_node == node_id && conn.from_pin == pin {
                        self.execute_node(conn.to_node, ctx, actions);
                    }
                }
            }
            NodeType::ForLoop { count } => {
                let n = *count;
                for i in 0..n {
                    ctx.variables.insert("__loop_index".to_string(), PinValue::Float(i as f32));
                    // Execute the "loop_body" flow output each iteration
                    for conn in &self.connections {
                        if conn.from_node == node_id && conn.from_pin == "loop_body" {
                            self.execute_node(conn.to_node, ctx, actions);
                        }
                    }
                }
                // After loop completes, follow "completed" output
                for conn in &self.connections {
                    if conn.from_node == node_id && conn.from_pin == "completed" {
                        self.execute_node(conn.to_node, ctx, actions);
                    }
                }
            }
            NodeType::Sequence => {
                // Execute each numbered output in order: "then_0", "then_1", etc.
                let mut idx = 0;
                loop {
                    let pin_name = format!("then_{}", idx);
                    let mut found = false;
                    for conn in &self.connections {
                        if conn.from_node == node_id && conn.from_pin == pin_name {
                            self.execute_node(conn.to_node, ctx, actions);
                            found = true;
                        }
                    }
                    if !found {
                        break;
                    }
                    idx += 1;
                }
            }
            NodeType::Delay { seconds } => {
                // Simplified: mark as delayed, then follow flow
                actions.push(GraphAction::Print(format!("[delay {:.2}s]", seconds)));
                self.follow_flow(node_id, ctx, actions);
            }

            // ── Variables ───────────────────────────────────────────
            NodeType::SetVariable { name } => {
                if let Some(val) = self.read_input(node_id, "value", ctx) {
                    ctx.variables.insert(name.clone(), val);
                }
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::GetVariable { .. } => {
                // Pure data node — output read via read_input/evaluate_node
            }

            // ── Math (pure data nodes) ──────────────────────────────
            NodeType::Add | NodeType::Subtract | NodeType::Multiply | NodeType::Divide
            | NodeType::Clamp | NodeType::Lerp | NodeType::Random => {
                // Pure data nodes — results consumed via evaluate_node
                // If they have flow connections, follow them
                self.follow_flow(node_id, ctx, actions);
            }

            // ── Comparison / Logic (pure data nodes) ────────────────
            NodeType::Equal | NodeType::NotEqual | NodeType::Greater | NodeType::Less
            | NodeType::And | NodeType::Or | NodeType::Not => {
                // Pure data nodes
                self.follow_flow(node_id, ctx, actions);
            }

            // ── Physics ─────────────────────────────────────────────
            NodeType::Raycast => {
                let origin = self
                    .read_input(node_id, "origin", ctx)
                    .and_then(|v| if let PinValue::Vec3(p) = v { Some(p) } else { None })
                    .unwrap_or(ctx.entity_position);
                let direction = self
                    .read_input(node_id, "direction", ctx)
                    .and_then(|v| if let PinValue::Vec3(d) = v { Some(d) } else { None })
                    .unwrap_or([0.0, -1.0, 0.0]);
                actions.push(GraphAction::Raycast { origin, direction });
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::ApplyForce => {
                let force = self
                    .read_input(node_id, "force", ctx)
                    .and_then(|v| if let PinValue::Vec3(f) = v { Some(f) } else { None })
                    .unwrap_or([0.0; 3]);
                actions.push(GraphAction::ApplyForce(ctx.entity_id, force));
                self.follow_flow(node_id, ctx, actions);
            }
            NodeType::IsGrounded => {
                // Pure data node — result via evaluate_node
            }

            // ── Custom ──────────────────────────────────────────────
            NodeType::Custom { name, code } => {
                // Evaluate embedded Rhai code (simplified: just print the code)
                actions.push(GraphAction::Print(format!("[custom:{}] {}", name, code)));
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
                return self.evaluate_node(conn.from_node, &conn.from_pin, ctx);
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

    /// Evaluate a pure data node and return its output value.
    fn evaluate_node(&self, node_id: u32, output_pin: &str, ctx: &GraphContext) -> Option<PinValue> {
        let node = self.nodes.iter().find(|n| n.id == node_id)?;
        match &node.node_type {
            NodeType::GetVariable { name } => ctx.variables.get(name).cloned(),

            // ── Entity data ─────────────────────────────────────
            NodeType::GetPosition => Some(PinValue::Vec3(ctx.entity_position)),
            NodeType::GetRotation => Some(PinValue::Vec3([
                ctx.entity_rotation[0],
                ctx.entity_rotation[1],
                ctx.entity_rotation[2],
            ])),
            NodeType::IsGrounded => Some(PinValue::Bool(ctx.grounded)),
            NodeType::FindByTag { .. } => {
                // Simplified: return current entity
                Some(PinValue::Entity(ctx.entity_id))
            }

            // ── Math ────────────────────────────────────────────
            NodeType::Add => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Float(a + b))
            }
            NodeType::Subtract => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Float(a - b))
            }
            NodeType::Multiply => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Float(a * b))
            }
            NodeType::Divide => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                if b.abs() < 1e-10 {
                    Some(PinValue::Float(0.0))
                } else {
                    Some(PinValue::Float(a / b))
                }
            }
            NodeType::Clamp => {
                let val = self.read_input_float(node_id, "value", ctx);
                let min = self.read_input_float(node_id, "min", ctx);
                let max = self.read_input_float(node_id, "max", ctx);
                Some(PinValue::Float(val.clamp(min, max)))
            }
            NodeType::Lerp => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                let t = self.read_input_float(node_id, "t", ctx);
                Some(PinValue::Float(a + (b - a) * t.clamp(0.0, 1.0)))
            }
            NodeType::Random => {
                // Deterministic pseudo-random based on time and entity_id
                let seed = (ctx.time * 1000.0 + ctx.entity_id as f32).sin() * 0.5 + 0.5;
                Some(PinValue::Float(seed))
            }

            // ── Comparison ──────────────────────────────────────
            NodeType::Equal => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Bool((a - b).abs() < 1e-6))
            }
            NodeType::NotEqual => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Bool((a - b).abs() >= 1e-6))
            }
            NodeType::Greater => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Bool(a > b))
            }
            NodeType::Less => {
                let a = self.read_input_float(node_id, "a", ctx);
                let b = self.read_input_float(node_id, "b", ctx);
                Some(PinValue::Bool(a < b))
            }

            // ── Logic ───────────────────────────────────────────
            NodeType::And => {
                let a = self.read_input_bool(node_id, "a", ctx);
                let b = self.read_input_bool(node_id, "b", ctx);
                Some(PinValue::Bool(a && b))
            }
            NodeType::Or => {
                let a = self.read_input_bool(node_id, "a", ctx);
                let b = self.read_input_bool(node_id, "b", ctx);
                Some(PinValue::Bool(a || b))
            }
            NodeType::Not => {
                let a = self.read_input_bool(node_id, "a", ctx);
                Some(PinValue::Bool(!a))
            }

            // For any other node type, try default pin value
            _ => {
                let node = self.nodes.iter().find(|n| n.id == node_id)?;
                node.outputs.iter().find(|p| p.name == output_pin)?.default_value.clone()
            }
        }
    }

    /// Read a float input, with fallback to 0.0.
    fn read_input_float(&self, node_id: u32, pin_name: &str, ctx: &GraphContext) -> f32 {
        self.read_input(node_id, pin_name, ctx)
            .and_then(|v| match v {
                PinValue::Float(f) => Some(f),
                _ => None,
            })
            .unwrap_or(0.0)
    }

    /// Read a bool input, with fallback to false.
    fn read_input_bool(&self, node_id: u32, pin_name: &str, ctx: &GraphContext) -> bool {
        self.read_input(node_id, pin_name, ctx)
            .and_then(|v| match v {
                PinValue::Bool(b) => Some(b),
                _ => None,
            })
            .unwrap_or(false)
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
        NodeType::EventInput { .. } => (
            vec![],
            vec![Pin {
                name: "flow_out".to_string(),
                pin_type: PinType::Flow,
                default_value: None,
            }],
        ),
        NodeType::Branch => (
            vec![
                Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "condition".into(), pin_type: PinType::Bool, default_value: Some(PinValue::Bool(false)) },
            ],
            vec![
                Pin { name: "true".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "false".into(), pin_type: PinType::Flow, default_value: None },
            ],
        ),
        NodeType::ForLoop { .. } => (
            vec![Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None }],
            vec![
                Pin { name: "loop_body".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "completed".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "index".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
            ],
        ),
        NodeType::Sequence => (
            vec![Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None }],
            vec![
                Pin { name: "then_0".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "then_1".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "then_2".into(), pin_type: PinType::Flow, default_value: None },
            ],
        ),
        NodeType::Print { .. } | NodeType::Destroy | NodeType::StopSound
        | NodeType::Delay { .. } | NodeType::Custom { .. } => (
            vec![Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None }],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
        ),
        NodeType::SetPosition | NodeType::SetRotation => (
            vec![
                Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "position".into(), pin_type: PinType::Vec3, default_value: Some(PinValue::Vec3([0.0; 3])) },
            ],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
        ),
        NodeType::GetPosition | NodeType::GetRotation => (
            vec![],
            vec![Pin { name: "value".into(), pin_type: PinType::Vec3, default_value: None }],
        ),
        NodeType::IsGrounded => (
            vec![],
            vec![Pin { name: "value".into(), pin_type: PinType::Bool, default_value: None }],
        ),
        NodeType::Spawn { .. } => (
            vec![
                Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "position".into(), pin_type: PinType::Vec3, default_value: Some(PinValue::Vec3([0.0; 3])) },
            ],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
        ),
        NodeType::PlaySound { .. } => (
            vec![Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None }],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
        ),
        NodeType::FindByTag { .. } => (
            vec![],
            vec![Pin { name: "entity".into(), pin_type: PinType::Entity, default_value: None }],
        ),
        NodeType::SetVariable { .. } => (
            vec![
                Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "value".into(), pin_type: PinType::Any, default_value: None },
            ],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
        ),
        NodeType::GetVariable { .. } => (
            vec![],
            vec![Pin { name: "value".into(), pin_type: PinType::Any, default_value: None }],
        ),
        // Math: two float inputs, one float output
        NodeType::Add | NodeType::Subtract | NodeType::Multiply | NodeType::Divide => (
            vec![
                Pin { name: "a".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
                Pin { name: "b".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
            ],
            vec![Pin { name: "result".into(), pin_type: PinType::Float, default_value: None }],
        ),
        NodeType::Clamp => (
            vec![
                Pin { name: "value".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
                Pin { name: "min".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
                Pin { name: "max".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(1.0)) },
            ],
            vec![Pin { name: "result".into(), pin_type: PinType::Float, default_value: None }],
        ),
        NodeType::Lerp => (
            vec![
                Pin { name: "a".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
                Pin { name: "b".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(1.0)) },
                Pin { name: "t".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.5)) },
            ],
            vec![Pin { name: "result".into(), pin_type: PinType::Float, default_value: None }],
        ),
        NodeType::Random => (
            vec![],
            vec![Pin { name: "result".into(), pin_type: PinType::Float, default_value: None }],
        ),
        // Comparison: two float inputs, one bool output
        NodeType::Equal | NodeType::NotEqual | NodeType::Greater | NodeType::Less => (
            vec![
                Pin { name: "a".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
                Pin { name: "b".into(), pin_type: PinType::Float, default_value: Some(PinValue::Float(0.0)) },
            ],
            vec![Pin { name: "result".into(), pin_type: PinType::Bool, default_value: None }],
        ),
        // Logic: two bool inputs, one bool output (Not has one input)
        NodeType::And | NodeType::Or => (
            vec![
                Pin { name: "a".into(), pin_type: PinType::Bool, default_value: Some(PinValue::Bool(false)) },
                Pin { name: "b".into(), pin_type: PinType::Bool, default_value: Some(PinValue::Bool(false)) },
            ],
            vec![Pin { name: "result".into(), pin_type: PinType::Bool, default_value: None }],
        ),
        NodeType::Not => (
            vec![
                Pin { name: "a".into(), pin_type: PinType::Bool, default_value: Some(PinValue::Bool(false)) },
            ],
            vec![Pin { name: "result".into(), pin_type: PinType::Bool, default_value: None }],
        ),
        // Physics
        NodeType::Raycast => (
            vec![
                Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "origin".into(), pin_type: PinType::Vec3, default_value: Some(PinValue::Vec3([0.0; 3])) },
                Pin { name: "direction".into(), pin_type: PinType::Vec3, default_value: Some(PinValue::Vec3([0.0, -1.0, 0.0])) },
            ],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
        ),
        NodeType::ApplyForce => (
            vec![
                Pin { name: "flow_in".into(), pin_type: PinType::Flow, default_value: None },
                Pin { name: "force".into(), pin_type: PinType::Vec3, default_value: Some(PinValue::Vec3([0.0; 3])) },
            ],
            vec![Pin { name: "flow_out".into(), pin_type: PinType::Flow, default_value: None }],
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
            entity_position: [0.0; 3],
            entity_rotation: [0.0, 0.0, 0.0, 1.0],
            grounded: false,
            collision_other: None,
            input_actions: HashSet::new(),
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
            entity_position: [0.0; 3],
            entity_rotation: [0.0, 0.0, 0.0, 1.0],
            grounded: false,
            collision_other: None,
            input_actions: HashSet::new(),
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
            entity_position: [0.0; 3],
            entity_rotation: [0.0, 0.0, 0.0, 1.0],
            grounded: false,
            collision_other: None,
            input_actions: HashSet::new(),
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

    fn make_ctx() -> GraphContext {
        GraphContext {
            variables: HashMap::new(),
            entity_id: 1,
            dt: 0.016,
            time: 1.0,
            entity_position: [10.0, 20.0, 30.0],
            entity_rotation: [0.0, 0.7071, 0.0, 0.7071],
            grounded: true,
            collision_other: None,
            input_actions: HashSet::new(),
        }
    }

    #[test]
    fn test_for_loop_executes_n_times() {
        let mut graph = VisualGraph::new("ForLoopTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let for_node = graph.add_node(NodeType::ForLoop { count: 3 }, [100.0, 0.0]);
        let print = graph.add_node(NodeType::Print { message: "tick".into() }, [200.0, 0.0]);

        graph.connect(start, "flow_out", for_node, "flow_in");
        graph.connect(for_node, "loop_body", print, "flow_in");

        let mut ctx = make_ctx();
        let actions = graph.execute(&mut ctx);
        let print_count = actions.iter().filter(|a| matches!(a, GraphAction::Print(m) if m == "tick")).count();
        assert_eq!(print_count, 3);
    }

    #[test]
    fn test_add_computes_correctly() {
        let mut graph = VisualGraph::new("AddTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let set_a = graph.add_node(NodeType::SetVariable { name: "a".into() }, [50.0, 0.0]);
        let set_b = graph.add_node(NodeType::SetVariable { name: "b".into() }, [100.0, 0.0]);
        let get_a = graph.add_node(NodeType::GetVariable { name: "a".into() }, [50.0, 50.0]);
        let get_b = graph.add_node(NodeType::GetVariable { name: "b".into() }, [100.0, 50.0]);
        let add = graph.add_node(NodeType::Add, [150.0, 0.0]);
        let set_result = graph.add_node(NodeType::SetVariable { name: "result".into() }, [200.0, 0.0]);

        // Flow: start -> set_a -> set_b -> set_result
        graph.connect(start, "flow_out", set_a, "flow_in");
        graph.connect(set_a, "flow_out", set_b, "flow_in");
        graph.connect(set_b, "flow_out", set_result, "flow_in");

        // Data: add reads a and b, set_result reads add output
        graph.connect(get_a, "value", add, "a");
        graph.connect(get_b, "value", add, "b");
        graph.connect(add, "result", set_result, "value");

        let mut ctx = make_ctx();
        ctx.variables.insert("a".to_string(), PinValue::Float(10.0));
        ctx.variables.insert("b".to_string(), PinValue::Float(25.0));
        // set_a will read "value" input which is not connected so it keeps existing var
        // Let's wire it properly: set_a sets a=10 from get_input_val
        // Actually simpler: just pre-set the variables and test the add node
        let actions = graph.execute(&mut ctx);
        assert!(matches!(ctx.variables.get("result"), Some(PinValue::Float(v)) if (*v - 35.0).abs() < 0.001),
            "Expected 35.0, got {:?}", ctx.variables.get("result"));
        let _ = actions;
    }

    #[test]
    fn test_equal_returns_true_false() {
        let mut graph = VisualGraph::new("EqualTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let get_a = graph.add_node(NodeType::GetVariable { name: "a".into() }, [50.0, 50.0]);
        let get_b = graph.add_node(NodeType::GetVariable { name: "b".into() }, [50.0, 100.0]);
        let equal = graph.add_node(NodeType::Equal, [100.0, 50.0]);
        let branch = graph.add_node(NodeType::Branch, [150.0, 0.0]);
        let print_yes = graph.add_node(NodeType::Print { message: "equal".into() }, [200.0, -50.0]);
        let print_no = graph.add_node(NodeType::Print { message: "not_equal".into() }, [200.0, 50.0]);

        graph.connect(start, "flow_out", branch, "flow_in");
        graph.connect(get_a, "value", equal, "a");
        graph.connect(get_b, "value", equal, "b");
        graph.connect(equal, "result", branch, "condition");
        graph.connect(branch, "true", print_yes, "flow_in");
        graph.connect(branch, "false", print_no, "flow_in");

        // Test equal case
        let mut ctx = make_ctx();
        ctx.variables.insert("a".to_string(), PinValue::Float(5.0));
        ctx.variables.insert("b".to_string(), PinValue::Float(5.0));
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "equal"));

        // Test not-equal case
        ctx.variables.insert("b".to_string(), PinValue::Float(7.0));
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "not_equal"));
    }

    #[test]
    fn test_and_or_not() {
        let graph = VisualGraph::new("LogicTest");
        let ctx = GraphContext {
            variables: HashMap::new(),
            entity_id: 1,
            dt: 0.016,
            time: 0.0,
            entity_position: [0.0; 3],
            entity_rotation: [0.0, 0.0, 0.0, 1.0],
            grounded: false,
            collision_other: None,
            input_actions: HashSet::new(),
        };

        // Test And directly via evaluate_node
        let mut g = VisualGraph::new("AndTest");
        let and_id = g.add_node(NodeType::And, [0.0, 0.0]);
        // With default false/false inputs, And should be false
        let result = g.evaluate_node(and_id, "result", &ctx);
        assert!(matches!(result, Some(PinValue::Bool(false))));

        // Test Or
        let mut g2 = VisualGraph::new("OrTest");
        let or_id = g2.add_node(NodeType::Or, [0.0, 0.0]);
        let result = g2.evaluate_node(or_id, "result", &ctx);
        assert!(matches!(result, Some(PinValue::Bool(false))));

        // Test Not (default false -> true)
        let mut g3 = VisualGraph::new("NotTest");
        let not_id = g3.add_node(NodeType::Not, [0.0, 0.0]);
        let result = g3.evaluate_node(not_id, "result", &ctx);
        assert!(matches!(result, Some(PinValue::Bool(true))));

        let _ = graph;
    }

    #[test]
    fn test_get_position_reads_from_context() {
        let mut graph = VisualGraph::new("GetPosTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let get_pos = graph.add_node(NodeType::GetPosition, [50.0, 50.0]);
        let set_pos = graph.add_node(NodeType::SetPosition, [100.0, 0.0]);

        graph.connect(start, "flow_out", set_pos, "flow_in");
        graph.connect(get_pos, "value", set_pos, "position");

        let mut ctx = make_ctx(); // position = [10, 20, 30]
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::SetPosition(1, pos) if pos == &[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_math_branch_graph() {
        // Full graph: get two vars, multiply them, compare > 100, branch to different prints
        let mut graph = VisualGraph::new("MathBranchTest");
        let start = graph.add_node(NodeType::EventStart, [0.0, 0.0]);
        let get_a = graph.add_node(NodeType::GetVariable { name: "speed".into() }, [50.0, 50.0]);
        let get_b = graph.add_node(NodeType::GetVariable { name: "factor".into() }, [50.0, 100.0]);
        let mul = graph.add_node(NodeType::Multiply, [100.0, 50.0]);
        let get_threshold = graph.add_node(NodeType::GetVariable { name: "threshold".into() }, [100.0, 100.0]);
        let gt = graph.add_node(NodeType::Greater, [150.0, 50.0]);
        let branch = graph.add_node(NodeType::Branch, [200.0, 0.0]);
        let print_fast = graph.add_node(NodeType::Print { message: "fast".into() }, [300.0, -50.0]);
        let print_slow = graph.add_node(NodeType::Print { message: "slow".into() }, [300.0, 50.0]);

        graph.connect(start, "flow_out", branch, "flow_in");
        graph.connect(get_a, "value", mul, "a");
        graph.connect(get_b, "value", mul, "b");
        graph.connect(mul, "result", gt, "a");
        graph.connect(get_threshold, "value", gt, "b");
        graph.connect(gt, "result", branch, "condition");
        graph.connect(branch, "true", print_fast, "flow_in");
        graph.connect(branch, "false", print_slow, "flow_in");

        let mut ctx = make_ctx();
        ctx.variables.insert("speed".to_string(), PinValue::Float(15.0));
        ctx.variables.insert("factor".to_string(), PinValue::Float(8.0));
        ctx.variables.insert("threshold".to_string(), PinValue::Float(100.0));

        // 15 * 8 = 120 > 100 → "fast"
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "fast"));

        // Change so product < threshold: 5 * 8 = 40 < 100 → "slow"
        ctx.variables.insert("speed".to_string(), PinValue::Float(5.0));
        let actions = graph.execute(&mut ctx);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], GraphAction::Print(m) if m == "slow"));
    }
}
