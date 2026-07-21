//! Retained, game-agnostic scene graph primitives.
//!
//! `vox_scene` is intentionally CPU-only. Game code produces ordered
//! [`SceneDelta`] batches; renderer code mirrors the graph to GPU-owned BLAS/TLAS
//! state. The graph itself never learns game concepts.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(u32);

impl NodeId {
    pub const fn from_raw(raw: u32) -> Option<Self> {
        if raw == 0 { None } else { Some(Self(raw)) }
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    fn slot(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProtoId(u32);

impl ProtoId {
    pub const fn from_raw(raw: u32) -> Option<Self> {
        if raw == 0 { None } else { Some(Self(raw)) }
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    fn slot(self) -> usize {
        self.0 as usize
    }
}

pub trait SceneId: Copy + Ord {
    fn from_nonzero_raw(raw: u32) -> Self;
    fn raw(self) -> u32;
}

impl SceneId for NodeId {
    fn from_nonzero_raw(raw: u32) -> Self {
        Self(raw)
    }

    fn raw(self) -> u32 {
        self.0
    }
}

impl SceneId for ProtoId {
    fn from_nonzero_raw(raw: u32) -> Self {
        Self(raw)
    }

    fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneIdAllocator<T: SceneId> {
    next: u32,
    _marker: PhantomData<T>,
}

impl<T: SceneId> Default for SceneIdAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SceneId> SceneIdAllocator<T> {
    pub const fn new() -> Self {
        Self {
            next: 1,
            _marker: PhantomData,
        }
    }

    pub fn allocate(&mut self) -> T {
        let raw = self.next;
        self.next = self
            .next
            .checked_add(1)
            .expect("scene id allocator exhausted");
        T::from_nonzero_raw(raw)
    }

    pub fn observe(&mut self, id: T) {
        self.next = self.next.max(id.raw().saturating_add(1));
    }

    pub const fn next_raw(&self) -> u32 {
        self.next
    }
}

pub type NodeIdAllocator = SceneIdAllocator<NodeId>;
pub type ProtoIdAllocator = SceneIdAllocator<ProtoId>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneTransform {
    pub rows: [[f32; 4]; 3],
}

impl SceneTransform {
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    pub const fn from_rows(rows: [[f32; 4]; 3]) -> Self {
        Self { rows }
    }

    pub fn from_translation(x: f32, y: f32, z: f32) -> Self {
        Self {
            rows: [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z]],
        }
    }
}

impl Default for SceneTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtoMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    pub material_ids: Vec<u32>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

impl ProtoMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
}

impl Default for ProtoMesh {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_ids: Vec::new(),
            aabb_min: [0.0; 3],
            aabb_max: [0.0; 3],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderComponent {
    pub proto: ProtoId,
    pub material_base: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub transform: SceneTransform,
    pub render: Option<RenderComponent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SceneDelta {
    AddProto {
        proto: ProtoId,
        mesh: ProtoMesh,
    },
    EditProto {
        proto: ProtoId,
        mesh: ProtoMesh,
    },
    AddNode {
        id: NodeId,
        proto: ProtoId,
        transform: SceneTransform,
        material_base: u32,
    },
    RemoveNode {
        id: NodeId,
    },
    SetTransform {
        id: NodeId,
        transform: SceneTransform,
    },
    SetMaterialBase {
        id: NodeId,
        material_base: u32,
    },
    SetProto {
        id: NodeId,
        proto: ProtoId,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApplyStats {
    pub protos_added: usize,
    pub protos_edited: usize,
    pub nodes_added: usize,
    pub nodes_removed: usize,
    pub transforms_set: usize,
    pub materials_set: usize,
    pub protos_set: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneGraphError {
    DuplicateProto(ProtoId),
    MissingProto(ProtoId),
    DuplicateNode(NodeId),
    MissingNode(NodeId),
}

impl fmt::Display for SceneGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateProto(id) => write!(f, "duplicate proto {}", id.raw()),
            Self::MissingProto(id) => write!(f, "missing proto {}", id.raw()),
            Self::DuplicateNode(id) => write!(f, "duplicate node {}", id.raw()),
            Self::MissingNode(id) => write!(f, "missing node {}", id.raw()),
        }
    }
}

impl Error for SceneGraphError {}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneGraph {
    nodes: Vec<Option<SceneNode>>,
    protos: Vec<Option<ProtoMesh>>,
    live_nodes: usize,
    live_protos: usize,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, deltas: &[SceneDelta]) -> Result<ApplyStats, SceneGraphError> {
        let mut stats = ApplyStats::default();
        for delta in deltas {
            match delta {
                SceneDelta::AddProto { proto, mesh } => {
                    self.add_proto(*proto, mesh.clone())?;
                    stats.protos_added += 1;
                }
                SceneDelta::EditProto { proto, mesh } => {
                    self.edit_proto(*proto, mesh.clone())?;
                    stats.protos_edited += 1;
                }
                SceneDelta::AddNode {
                    id,
                    proto,
                    transform,
                    material_base,
                } => {
                    self.add_node(*id, *proto, *transform, *material_base)?;
                    stats.nodes_added += 1;
                }
                SceneDelta::RemoveNode { id } => {
                    self.remove_node(*id)?;
                    stats.nodes_removed += 1;
                }
                SceneDelta::SetTransform { id, transform } => {
                    self.set_transform(*id, *transform)?;
                    stats.transforms_set += 1;
                }
                SceneDelta::SetMaterialBase { id, material_base } => {
                    self.set_material_base(*id, *material_base)?;
                    stats.materials_set += 1;
                }
                SceneDelta::SetProto { id, proto } => {
                    self.set_proto(*id, *proto)?;
                    stats.protos_set += 1;
                }
            }
        }
        Ok(stats)
    }

    pub fn add_proto(&mut self, proto: ProtoId, mesh: ProtoMesh) -> Result<(), SceneGraphError> {
        let slot = proto.slot();
        ensure_len(&mut self.protos, slot + 1);
        if self.protos[slot].is_some() {
            return Err(SceneGraphError::DuplicateProto(proto));
        }
        self.protos[slot] = Some(mesh);
        self.live_protos += 1;
        Ok(())
    }

    pub fn edit_proto(&mut self, proto: ProtoId, mesh: ProtoMesh) -> Result<(), SceneGraphError> {
        let slot = proto.slot();
        let existing = self
            .protos
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or(SceneGraphError::MissingProto(proto))?;
        *existing = mesh;
        Ok(())
    }

    pub fn add_node(
        &mut self,
        id: NodeId,
        proto: ProtoId,
        transform: SceneTransform,
        material_base: u32,
    ) -> Result<(), SceneGraphError> {
        if !self.contains_proto(proto) {
            return Err(SceneGraphError::MissingProto(proto));
        }
        let slot = id.slot();
        ensure_len(&mut self.nodes, slot + 1);
        if self.nodes[slot].is_some() {
            return Err(SceneGraphError::DuplicateNode(id));
        }
        self.nodes[slot] = Some(SceneNode {
            id,
            parent: None,
            transform,
            render: Some(RenderComponent {
                proto,
                material_base,
            }),
        });
        self.live_nodes += 1;
        Ok(())
    }

    pub fn remove_node(&mut self, id: NodeId) -> Result<SceneNode, SceneGraphError> {
        let slot = id.slot();
        let removed = self
            .nodes
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or(SceneGraphError::MissingNode(id))?;
        self.live_nodes -= 1;
        Ok(removed)
    }

    pub fn set_transform(
        &mut self,
        id: NodeId,
        transform: SceneTransform,
    ) -> Result<(), SceneGraphError> {
        let node = self.node_mut(id)?;
        node.transform = transform;
        Ok(())
    }

    pub fn set_proto(&mut self, id: NodeId, proto: ProtoId) -> Result<(), SceneGraphError> {
        if !self.contains_proto(proto) {
            return Err(SceneGraphError::MissingProto(proto));
        }
        let node = self.node_mut(id)?;
        let render = node
            .render
            .as_mut()
            .ok_or(SceneGraphError::MissingNode(id))?;
        render.proto = proto;
        Ok(())
    }

    pub fn set_material_base(
        &mut self,
        id: NodeId,
        material_base: u32,
    ) -> Result<(), SceneGraphError> {
        let node = self.node_mut(id)?;
        let render = node
            .render
            .as_mut()
            .ok_or(SceneGraphError::MissingNode(id))?;
        render.material_base = material_base;
        Ok(())
    }

    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(id.slot()).and_then(Option::as_ref)
    }

    pub fn proto(&self, id: ProtoId) -> Option<&ProtoMesh> {
        self.protos.get(id.slot()).and_then(Option::as_ref)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &SceneNode> {
        self.nodes.iter().filter_map(Option::as_ref)
    }

    pub fn protos(&self) -> impl Iterator<Item = (ProtoId, &ProtoMesh)> {
        self.protos.iter().enumerate().filter_map(|(slot, mesh)| {
            let raw = slot as u32;
            Some((ProtoId::from_raw(raw)?, mesh.as_ref()?))
        })
    }

    pub fn live_node_count(&self) -> usize {
        self.live_nodes
    }

    pub fn live_proto_count(&self) -> usize {
        self.live_protos
    }

    pub fn contains_node(&self, id: NodeId) -> bool {
        self.node(id).is_some()
    }

    pub fn contains_proto(&self, id: ProtoId) -> bool {
        self.proto(id).is_some()
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut SceneNode, SceneGraphError> {
        self.nodes
            .get_mut(id.slot())
            .and_then(Option::as_mut)
            .ok_or(SceneGraphError::MissingNode(id))
    }
}

fn ensure_len<T>(slots: &mut Vec<Option<T>>, len: usize) {
    if slots.len() < len {
        slots.resize_with(len, || None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proto(raw: u32) -> ProtoId {
        ProtoId::from_raw(raw).unwrap()
    }

    fn node(raw: u32) -> NodeId {
        NodeId::from_raw(raw).unwrap()
    }

    fn mesh(triangles: usize) -> ProtoMesh {
        ProtoMesh {
            positions: vec![[0.0, 0.0, 0.0]; triangles * 3],
            normals: vec![[0.0, 1.0, 0.0]; triangles * 3],
            uvs: vec![[0.0, 0.0]; triangles * 3],
            indices: (0..triangles)
                .map(|i| {
                    let base = (i * 3) as u32;
                    [base, base + 1, base + 2]
                })
                .collect(),
            material_ids: vec![0; triangles],
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn ordered_deltas_add_proto_and_node() {
        let mut graph = SceneGraph::new();
        let stats = graph
            .apply(&[
                SceneDelta::AddProto {
                    proto: proto(1),
                    mesh: mesh(2),
                },
                SceneDelta::AddNode {
                    id: node(1),
                    proto: proto(1),
                    transform: SceneTransform::from_translation(10.0, 2.0, -4.0),
                    material_base: 7,
                },
            ])
            .unwrap();

        assert_eq!(stats.protos_added, 1);
        assert_eq!(stats.nodes_added, 1);
        assert_eq!(graph.live_proto_count(), 1);
        assert_eq!(graph.live_node_count(), 1);
        let n = graph.node(node(1)).unwrap();
        assert_eq!(
            n.transform,
            SceneTransform::from_translation(10.0, 2.0, -4.0)
        );
        assert_eq!(
            n.render,
            Some(RenderComponent {
                proto: proto(1),
                material_base: 7,
            })
        );
    }

    #[test]
    fn set_transform_updates_node_without_touching_proto_count() {
        let mut graph = SceneGraph::new();
        graph
            .apply(&[
                SceneDelta::AddProto {
                    proto: proto(1),
                    mesh: mesh(1),
                },
                SceneDelta::AddNode {
                    id: node(1),
                    proto: proto(1),
                    transform: SceneTransform::IDENTITY,
                    material_base: 0,
                },
                SceneDelta::SetTransform {
                    id: node(1),
                    transform: SceneTransform::from_translation(3.0, 0.0, 9.0),
                },
            ])
            .unwrap();

        assert_eq!(graph.live_proto_count(), 1);
        assert_eq!(
            graph.node(node(1)).unwrap().transform,
            SceneTransform::from_translation(3.0, 0.0, 9.0)
        );
    }

    #[test]
    fn set_material_base_updates_only_render_binding() {
        let mut graph = SceneGraph::new();
        let transform = SceneTransform::from_translation(1.0, 2.0, 3.0);
        let stats = graph
            .apply(&[
                SceneDelta::AddProto {
                    proto: proto(1),
                    mesh: mesh(1),
                },
                SceneDelta::AddNode {
                    id: node(1),
                    proto: proto(1),
                    transform,
                    material_base: 4,
                },
                SceneDelta::SetMaterialBase {
                    id: node(1),
                    material_base: 19,
                },
            ])
            .unwrap();

        let n = graph.node(node(1)).unwrap();
        assert_eq!(n.transform, transform);
        assert_eq!(n.render.unwrap().material_base, 19);
        assert_eq!(stats.materials_set, 1);
    }

    #[test]
    fn structural_errors_are_explicit() {
        let mut graph = SceneGraph::new();
        let err = graph
            .apply(&[SceneDelta::AddNode {
                id: node(1),
                proto: proto(7),
                transform: SceneTransform::IDENTITY,
                material_base: 0,
            }])
            .unwrap_err();
        assert_eq!(err, SceneGraphError::MissingProto(proto(7)));
    }

    #[test]
    fn remove_node_releases_only_that_node() {
        let mut graph = SceneGraph::new();
        graph
            .apply(&[
                SceneDelta::AddProto {
                    proto: proto(1),
                    mesh: mesh(1),
                },
                SceneDelta::AddNode {
                    id: node(1),
                    proto: proto(1),
                    transform: SceneTransform::IDENTITY,
                    material_base: 0,
                },
                SceneDelta::AddNode {
                    id: node(2),
                    proto: proto(1),
                    transform: SceneTransform::IDENTITY,
                    material_base: 0,
                },
                SceneDelta::RemoveNode { id: node(1) },
            ])
            .unwrap();

        assert!(!graph.contains_node(node(1)));
        assert!(graph.contains_node(node(2)));
        assert_eq!(graph.live_node_count(), 1);
        assert_eq!(graph.live_proto_count(), 1);
    }

    #[test]
    fn allocators_are_monotonic_and_observe_external_ids() {
        let mut nodes = NodeIdAllocator::new();
        assert_eq!(nodes.allocate().raw(), 1);
        nodes.observe(node(10));
        assert_eq!(nodes.allocate().raw(), 11);

        let mut protos = ProtoIdAllocator::new();
        assert_eq!(protos.allocate().raw(), 1);
        assert_eq!(protos.allocate().raw(), 2);
    }
}
