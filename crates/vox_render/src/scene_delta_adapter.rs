//! Renderer-side planning for retained scene graph deltas.
//!
//! The CPU scene graph emits game-agnostic [`vox_scene::SceneDelta`] batches.
//! This adapter owns the renderer mirror's stable `NodeId -> instance_index`
//! mapping and turns transform-only batches into TLAS refits. Structural edits
//! deliberately fall back to the existing full-scene upload path until the GPU
//! mirror owns prototype/node allocation directly.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use vox_scene::{NodeId, SceneDelta as GraphSceneDelta};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetainedDeltaStats {
    pub structural_deltas: usize,
    pub transform_deltas: usize,
    pub refits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetainedDeltaPlan {
    pub structural_rebuild: bool,
    pub refits: Vec<(usize, [[f32; 4]; 3])>,
    pub stats: RetainedDeltaStats,
}

impl RetainedDeltaPlan {
    pub fn requires_scene_rebuild(&self) -> bool {
        self.structural_rebuild
    }

    pub fn is_empty(&self) -> bool {
        !self.structural_rebuild && self.refits.is_empty()
    }

    #[cfg(feature = "spectra-native")]
    pub fn queue_resident_refits(
        &self,
        renderer: &mut crate::resident_renderer::ResidentCityRenderer,
    ) -> Result<(), RetainedDeltaError> {
        if self.structural_rebuild {
            return Err(RetainedDeltaError::StructuralRebuildRequired);
        }
        for (instance_index, transform) in &self.refits {
            renderer.update_instance_transform(*instance_index, *transform);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedDeltaError {
    DuplicateNode(NodeId),
    UnknownNode(NodeId),
    StructuralRebuildRequired,
}

impl fmt::Display for RetainedDeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(f, "duplicate retained node {}", id.raw()),
            Self::UnknownNode(id) => write!(f, "unknown retained node {}", id.raw()),
            Self::StructuralRebuildRequired => write!(f, "structural scene rebuild required"),
        }
    }
}

impl Error for RetainedDeltaError {}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetainedRenderMirror {
    node_to_instance: BTreeMap<NodeId, usize>,
    instance_to_node: Vec<NodeId>,
}

impl RetainedRenderMirror {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace_instances_in_node_order<I>(&mut self, nodes: I) -> Result<(), RetainedDeltaError>
    where
        I: IntoIterator<Item = NodeId>,
    {
        let mut node_to_instance = BTreeMap::new();
        let mut instance_to_node = Vec::new();

        for (instance_index, node) in nodes.into_iter().enumerate() {
            if node_to_instance.insert(node, instance_index).is_some() {
                return Err(RetainedDeltaError::DuplicateNode(node));
            }
            instance_to_node.push(node);
        }

        self.node_to_instance = node_to_instance;
        self.instance_to_node = instance_to_node;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.node_to_instance.clear();
        self.instance_to_node.clear();
    }

    pub fn len(&self) -> usize {
        self.instance_to_node.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instance_to_node.is_empty()
    }

    pub fn instance_index(&self, node: NodeId) -> Option<usize> {
        self.node_to_instance.get(&node).copied()
    }

    pub fn node_for_instance(&self, instance_index: usize) -> Option<NodeId> {
        self.instance_to_node.get(instance_index).copied()
    }

    pub fn plan_deltas(
        &self,
        deltas: &[GraphSceneDelta],
    ) -> Result<RetainedDeltaPlan, RetainedDeltaError> {
        let mut stats = RetainedDeltaStats::default();

        for delta in deltas {
            if delta_requires_rebuild(delta) {
                stats.structural_deltas += 1;
            } else if matches!(delta, GraphSceneDelta::SetTransform { .. }) {
                stats.transform_deltas += 1;
            }
        }

        if stats.structural_deltas > 0 {
            return Ok(RetainedDeltaPlan {
                structural_rebuild: true,
                refits: Vec::new(),
                stats,
            });
        }

        let mut refits = BTreeMap::new();
        for delta in deltas {
            if let GraphSceneDelta::SetTransform { id, transform } = delta {
                let instance_index = self
                    .instance_index(*id)
                    .ok_or(RetainedDeltaError::UnknownNode(*id))?;
                refits.insert(instance_index, transform.rows);
            }
        }

        stats.refits = refits.len();
        Ok(RetainedDeltaPlan {
            structural_rebuild: false,
            refits: refits.into_iter().collect(),
            stats,
        })
    }
}

fn delta_requires_rebuild(delta: &GraphSceneDelta) -> bool {
    matches!(
        delta,
        GraphSceneDelta::AddProto { .. }
            | GraphSceneDelta::EditProto { .. }
            | GraphSceneDelta::AddNode { .. }
            | GraphSceneDelta::RemoveNode { .. }
            | GraphSceneDelta::SetProto { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_scene::{ProtoId, ProtoMesh, SceneTransform};

    fn node(raw: u32) -> NodeId {
        NodeId::from_raw(raw).unwrap()
    }

    fn proto(raw: u32) -> ProtoId {
        ProtoId::from_raw(raw).unwrap()
    }

    fn transform(x: f32) -> SceneTransform {
        SceneTransform::from_translation(x, 0.0, 0.0)
    }

    #[test]
    fn rebuild_result_installs_stable_instance_mapping() {
        let mut mirror = RetainedRenderMirror::new();
        mirror
            .replace_instances_in_node_order([node(3), node(1), node(7)])
            .unwrap();

        assert_eq!(mirror.len(), 3);
        assert_eq!(mirror.instance_index(node(3)), Some(0));
        assert_eq!(mirror.instance_index(node(1)), Some(1));
        assert_eq!(mirror.instance_index(node(7)), Some(2));
        assert_eq!(mirror.node_for_instance(1), Some(node(1)));
    }

    #[test]
    fn transform_only_delta_maps_to_instance_refit() {
        let mut mirror = RetainedRenderMirror::new();
        mirror
            .replace_instances_in_node_order([node(10), node(20)])
            .unwrap();

        let plan = mirror
            .plan_deltas(&[GraphSceneDelta::SetTransform {
                id: node(20),
                transform: transform(4.0),
            }])
            .unwrap();

        assert!(!plan.requires_scene_rebuild());
        assert_eq!(plan.refits, vec![(1, transform(4.0).rows)]);
        assert_eq!(
            plan.stats,
            RetainedDeltaStats {
                structural_deltas: 0,
                transform_deltas: 1,
                refits: 1,
            }
        );
    }

    #[test]
    fn transform_refits_are_latest_wins_and_instance_sorted() {
        let mut mirror = RetainedRenderMirror::new();
        mirror
            .replace_instances_in_node_order([node(5), node(9), node(2)])
            .unwrap();

        let plan = mirror
            .plan_deltas(&[
                GraphSceneDelta::SetTransform {
                    id: node(2),
                    transform: transform(2.0),
                },
                GraphSceneDelta::SetTransform {
                    id: node(5),
                    transform: transform(1.0),
                },
                GraphSceneDelta::SetTransform {
                    id: node(5),
                    transform: transform(5.0),
                },
            ])
            .unwrap();

        assert_eq!(
            plan.refits,
            vec![(0, transform(5.0).rows), (2, transform(2.0).rows)]
        );
        assert_eq!(plan.stats.transform_deltas, 3);
        assert_eq!(plan.stats.refits, 2);
    }

    #[test]
    fn structural_delta_uses_full_rebuild_fallback() {
        let mirror = RetainedRenderMirror::new();
        let structural_deltas = [
            GraphSceneDelta::AddProto {
                proto: proto(1),
                mesh: ProtoMesh::default(),
            },
            GraphSceneDelta::EditProto {
                proto: proto(1),
                mesh: ProtoMesh::default(),
            },
            GraphSceneDelta::AddNode {
                id: node(1),
                proto: proto(1),
                transform: SceneTransform::IDENTITY,
                material_base: 0,
            },
            GraphSceneDelta::RemoveNode { id: node(1) },
            GraphSceneDelta::SetProto {
                id: node(1),
                proto: proto(2),
            },
        ];

        for delta in structural_deltas {
            let plan = mirror.plan_deltas(&[delta]).unwrap();
            assert!(plan.requires_scene_rebuild());
            assert!(plan.refits.is_empty());
            assert_eq!(plan.stats.structural_deltas, 1);
        }
    }

    #[test]
    fn structural_batches_do_not_emit_stale_refits() {
        let mut mirror = RetainedRenderMirror::new();
        mirror.replace_instances_in_node_order([node(1)]).unwrap();

        let plan = mirror
            .plan_deltas(&[
                GraphSceneDelta::SetTransform {
                    id: node(1),
                    transform: transform(1.0),
                },
                GraphSceneDelta::RemoveNode { id: node(1) },
            ])
            .unwrap();

        assert!(plan.requires_scene_rebuild());
        assert!(plan.refits.is_empty());
        assert_eq!(plan.stats.transform_deltas, 1);
        assert_eq!(plan.stats.structural_deltas, 1);
    }

    #[test]
    fn unknown_transform_node_is_explicit() {
        let mirror = RetainedRenderMirror::new();
        let err = mirror
            .plan_deltas(&[GraphSceneDelta::SetTransform {
                id: node(42),
                transform: transform(1.0),
            }])
            .unwrap_err();

        assert_eq!(err, RetainedDeltaError::UnknownNode(node(42)));
    }

    #[test]
    fn duplicate_rebuild_mapping_is_rejected_without_mutating_mirror() {
        let mut mirror = RetainedRenderMirror::new();
        mirror.replace_instances_in_node_order([node(9)]).unwrap();

        let err = mirror
            .replace_instances_in_node_order([node(1), node(1)])
            .unwrap_err();

        assert_eq!(err, RetainedDeltaError::DuplicateNode(node(1)));
        assert_eq!(mirror.instance_index(node(9)), Some(0));
    }
}
