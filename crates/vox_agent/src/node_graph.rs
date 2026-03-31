use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub type NodeId = u32;
pub type PinId  = u32;

#[derive(Debug, Clone)]
pub struct AgentNode {
    id:       NodeId,
    kind:     AgentNodeKind,
    position: [f32; 2],
    inputs:   Vec<PinId>,
    outputs:  Vec<PinId>,
}

impl AgentNode {
    pub fn id(&self) -> NodeId           { self.id }
    pub fn kind(&self) -> &AgentNodeKind { &self.kind }
    pub fn input_pins(&self) -> &[PinId] { &self.inputs }
    pub fn output_pins(&self) -> &[PinId] { &self.outputs }
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub src_node: NodeId,
    pub src_pin:  PinId,
    pub dst_node: NodeId,
    pub dst_pin:  PinId,
}

#[derive(Debug, thiserror::Error)]
pub enum ConnectionError {
    #[error("source node {0} not found")]
    SrcNotFound(NodeId),
    #[error("destination node {0} not found")]
    DstNotFound(NodeId),
}

#[derive(Debug, thiserror::Error)]
pub enum CycleError {
    #[error("graph contains a cycle")]
    Cycle,
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("node {0}: uses feature not enabled in AgentStateDesc ({1})")]
    FeatureNotEnabled(NodeId, &'static str),
}

/// IR for a complete agent behavior program.
pub struct AgentNodeGraph {
    nodes:       Vec<AgentNode>,
    connections: Vec<Connection>,
    name:        String,
    next_id:     NodeId,
    next_pin:    PinId,
}

impl AgentNodeGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self { nodes: Vec::new(), connections: Vec::new(),
               name: name.into(), next_id: 1, next_pin: 1 }
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn nodes(&self) -> &[AgentNode] { &self.nodes }
    pub fn connections(&self) -> &[Connection] { &self.connections }

    pub fn add_node(&mut self, kind: AgentNodeKind, position: [f32; 2]) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let (n_in, n_out) = kind.pin_counts();
        let inputs: Vec<PinId> = (0..n_in).map(|_| { let p = self.next_pin; self.next_pin += 1; p }).collect();
        let outputs: Vec<PinId> = (0..n_out).map(|_| { let p = self.next_pin; self.next_pin += 1; p }).collect();
        self.nodes.push(AgentNode { id, kind, position, inputs, outputs });
        id
    }

    pub fn connect(
        &mut self,
        src_node: NodeId, src_pin: PinId,
        dst_node: NodeId, dst_pin: PinId,
    ) -> Result<(), ConnectionError> {
        if !self.nodes.iter().any(|n| n.id == src_node) {
            return Err(ConnectionError::SrcNotFound(src_node));
        }
        if !self.nodes.iter().any(|n| n.id == dst_node) {
            return Err(ConnectionError::DstNotFound(dst_node));
        }
        self.connections.push(Connection { src_node, src_pin, dst_node, dst_pin });
        Ok(())
    }

    /// Kahn's algorithm. Returns nodes in topological order or Err if cycle detected.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, CycleError> {
        let mut in_degree: HashMap<NodeId, usize> = self.nodes.iter().map(|n| (n.id, 0)).collect();
        for c in &self.connections {
            *in_degree.entry(c.dst_node).or_insert(0) += 1;
        }
        let mut queue: std::collections::VecDeque<NodeId> =
            in_degree.iter().filter(|&(_, d)| *d == 0).map(|(&id, _)| id).collect();
        let mut order = Vec::new();
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for c in self.connections.iter().filter(|c| c.src_node == id) {
                let d = in_degree.entry(c.dst_node).or_insert(0);
                *d = d.saturating_sub(1);
                if *d == 0 { queue.push_back(c.dst_node); }
            }
        }
        if order.len() == self.nodes.len() { Ok(order) } else { Err(CycleError::Cycle) }
    }

    /// Type-checks connections and validates feature requirements against desc.
    pub fn validate(
        &self,
        _registry: &AgentNodeRegistry,
        desc: &crate::desc::AgentStateDesc,
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        for node in &self.nodes {
            match &node.kind {
                AgentNodeKind::SampleSpectral { .. }
                | AgentNodeKind::OnSpectralThreshold { .. }
                | AgentNodeKind::SpectralDot
                | AgentNodeKind::SampleSpectralCurve
                | AgentNodeKind::SpectralBand { .. } => {
                    if !desc.spectral {
                        errors.push(ValidationError::FeatureNotEnabled(node.id, "spectral"));
                    }
                }
                AgentNodeKind::QueryNeighbours { .. }
                | AgentNodeKind::NeighbourCount
                | AgentNodeKind::NeighbourPosition { .. } => {
                    if desc.spatial_hash.is_none() {
                        errors.push(ValidationError::FeatureNotEnabled(node.id, "spatial_hash"));
                    }
                }
                AgentNodeKind::ReadCustom { .. } | AgentNodeKind::WriteCustom { .. } => {
                    if desc.custom_floats == 0 {
                        errors.push(ValidationError::FeatureNotEnabled(node.id, "custom_floats"));
                    }
                }
                _ => {}
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

/// Slang fragment template for a custom node.
#[derive(Clone)]
pub struct SlangFragment(pub String);

/// Registry of game-registered custom node kinds.
pub struct AgentNodeRegistry {
    custom: HashMap<String, SlangFragment>,
}

impl AgentNodeRegistry {
    pub fn new() -> Self { Self { custom: HashMap::new() } }

    pub fn register(&mut self, kind_name: impl Into<String>, fragment: SlangFragment) {
        self.custom.insert(kind_name.into(), fragment);
    }

    pub fn get(&self, kind_name: &str) -> Option<&SlangFragment> {
        self.custom.get(kind_name)
    }
}

/// Built-in engine node kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentNodeKind {
    // Entry
    OnUpdate,
    OnSpectralThreshold { band: u32, threshold: f32 },
    // Read
    GetPosition,
    GetVelocity,
    AgentId,
    GetTime,
    ReadCustom { slot: u32 },
    SampleSpectral { band: u32 },
    QueryNeighbours { radius: f32 },
    NeighbourCount,
    NeighbourPosition { index: u32 },
    // Math
    Add, Sub, Mul, Div,
    Lerp, Clamp, Normalize, Length, Distance,
    Select, Noise,
    // Logic
    Compare { op: CompareOp },
    And, Or, Not, Branch,
    // Write
    SetVelocity, AddVelocity,
    WriteCustom { slot: u32 },
    RequestCpuAttention,
    // Spectral
    SpectralDot, SampleSpectralCurve, SpectralBand { band: u32 },
    // Game-registered
    Custom { kind_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompareOp { Lt, Le, Gt, Ge, Eq, Ne }

impl AgentNodeKind {
    /// Returns (input pin count, output pin count).
    pub fn pin_counts(&self) -> (usize, usize) {
        match self {
            AgentNodeKind::OnUpdate => (0, 1),
            AgentNodeKind::OnSpectralThreshold { .. } => (0, 1),
            AgentNodeKind::GetPosition | AgentNodeKind::GetVelocity => (0, 1),
            AgentNodeKind::AgentId | AgentNodeKind::GetTime => (0, 1),
            AgentNodeKind::ReadCustom { .. } => (0, 1),
            AgentNodeKind::SampleSpectral { .. } => (0, 1),
            AgentNodeKind::QueryNeighbours { .. } => (0, 1),
            AgentNodeKind::NeighbourCount => (1, 1),
            AgentNodeKind::NeighbourPosition { .. } => (1, 1),
            AgentNodeKind::Add | AgentNodeKind::Sub
            | AgentNodeKind::Mul | AgentNodeKind::Div => (2, 1),
            AgentNodeKind::Lerp => (3, 1),
            AgentNodeKind::Clamp => (3, 1),
            AgentNodeKind::Normalize | AgentNodeKind::Length => (1, 1),
            AgentNodeKind::Distance => (2, 1),
            AgentNodeKind::Select => (3, 1),
            AgentNodeKind::Noise => (1, 1),
            AgentNodeKind::Compare { .. } => (2, 1),
            AgentNodeKind::And | AgentNodeKind::Or => (2, 1),
            AgentNodeKind::Not => (1, 1),
            AgentNodeKind::Branch => (1, 2),
            AgentNodeKind::SetVelocity | AgentNodeKind::AddVelocity => (1, 0),
            AgentNodeKind::WriteCustom { .. } => (1, 0),
            AgentNodeKind::RequestCpuAttention => (0, 0),
            AgentNodeKind::SpectralDot => (2, 1),
            AgentNodeKind::SampleSpectralCurve => (1, 1),
            AgentNodeKind::SpectralBand { .. } => (1, 1),
            AgentNodeKind::Custom { .. } => (1, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::AgentStateDesc;

    fn registry() -> AgentNodeRegistry { AgentNodeRegistry::new() }

    fn desc_minimal() -> AgentStateDesc {
        AgentStateDesc { agent_count: 100, custom_floats: 0, spectral: false, spatial_hash: None }
    }

    #[test]
    fn add_node_returns_unique_ids() {
        let mut g = AgentNodeGraph::new("test");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::SetVelocity, [100.0, 0.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn connect_valid_nodes_succeeds() {
        let mut g = AgentNodeGraph::new("test");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::SetVelocity, [100.0, 0.0]);
        let src_pin = g.nodes()[0].output_pins()[0];
        let dst_pin = g.nodes()[1].input_pins()[0];
        assert!(g.connect(a, src_pin, b, dst_pin).is_ok());
    }

    #[test]
    fn connect_invalid_src_returns_err() {
        let mut g = AgentNodeGraph::new("test");
        let b = g.add_node(AgentNodeKind::SetVelocity, [0.0, 0.0]);
        let dst_pin = g.nodes()[0].input_pins()[0];
        assert!(matches!(g.connect(999, 0, b, dst_pin), Err(ConnectionError::SrcNotFound(999))));
    }

    #[test]
    fn topological_order_linear_graph() {
        let mut g = AgentNodeGraph::new("test");
        let a = g.add_node(AgentNodeKind::GetPosition, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::Normalize,   [100.0, 0.0]);
        let c = g.add_node(AgentNodeKind::SetVelocity, [200.0, 0.0]);
        let a_out = g.nodes().iter().find(|n| n.id() == a).unwrap().output_pins()[0];
        let b_in  = g.nodes().iter().find(|n| n.id() == b).unwrap().input_pins()[0];
        let b_out = g.nodes().iter().find(|n| n.id() == b).unwrap().output_pins()[0];
        let c_in  = g.nodes().iter().find(|n| n.id() == c).unwrap().input_pins()[0];
        g.connect(a, a_out, b, b_in).unwrap();
        g.connect(b, b_out, c, c_in).unwrap();
        let order = g.topological_order().expect("no cycle");
        assert_eq!(order[0], a, "GetPosition must come first");
        assert_eq!(order[2], c, "SetVelocity must come last");
    }

    #[test]
    fn topological_order_detects_cycle() {
        let mut g = AgentNodeGraph::new("cyclic");
        let a = g.add_node(AgentNodeKind::Add, [0.0, 0.0]);
        let b = g.add_node(AgentNodeKind::Add, [100.0, 0.0]);
        let a_out = g.nodes().iter().find(|n| n.id() == a).unwrap().output_pins()[0];
        let b_out = g.nodes().iter().find(|n| n.id() == b).unwrap().output_pins()[0];
        let a_in  = g.nodes().iter().find(|n| n.id() == a).unwrap().input_pins()[0];
        let b_in  = g.nodes().iter().find(|n| n.id() == b).unwrap().input_pins()[0];
        g.connect(a, a_out, b, b_in).unwrap();
        g.connect(b, b_out, a, a_in).unwrap();
        assert!(matches!(g.topological_order(), Err(CycleError::Cycle)));
    }

    #[test]
    fn validate_rejects_spectral_node_when_spectral_disabled() {
        let mut g = AgentNodeGraph::new("test");
        g.add_node(AgentNodeKind::SampleSpectral { band: 5 }, [0.0, 0.0]);
        let errors = g.validate(&registry(), &desc_minimal()).unwrap_err();
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn validate_accepts_spectral_node_when_spectral_enabled() {
        let desc = AgentStateDesc { spectral: true, ..desc_minimal() };
        let mut g = AgentNodeGraph::new("test");
        g.add_node(AgentNodeKind::SampleSpectral { band: 5 }, [0.0, 0.0]);
        assert!(g.validate(&registry(), &desc).is_ok());
    }
}
