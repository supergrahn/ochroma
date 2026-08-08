//! Engine-generic immutable frontier meshes for progressive geometry reveal.
//!
//! A caller derives one cap prototype from each authored reveal section, admits
//! it through the normal runtime-geometry path, and instances that shared BLAS
//! for active objects. No hit shader synthesizes triangles and no active object
//! owns a private mesh.

use std::collections::BTreeMap;

use crate::runtime_geometry::{RuntimeGeometryKey, RuntimeGeometrySource};
use vox_scene::{
    NodeId, NodeIdAllocator, ProtoId, ProtoIdAllocator, ProtoMesh, SceneDelta, SceneTransform,
};

const EPS: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressiveRevealCapError {
    TooFewVertices,
    NonFiniteVertex,
    NonPlanarPolygon,
    DegenerateEdge,
    DegeneratePolygon,
    SelfIntersectingPolygon,
    TriangulationFailed,
    MissingPrototype,
    DuplicateInstance,
    MissingInstance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressiveRevealCapAdmission {
    pub key: RuntimeGeometryKey,
    pub proto: ProtoId,
    pub newly_admitted: bool,
}

/// Retained-scene ownership for shared reveal caps. Prototype identity is the
/// canonical runtime-geometry content key; active project identity owns only a
/// cheap TLAS node. Completing a project removes its node, never the shared
/// prototype/BLAS used by other projects.
#[derive(Debug, Default)]
pub struct ProgressiveRevealCapRegistry {
    prototypes: BTreeMap<RuntimeGeometryKey, ProtoId>,
    instances: BTreeMap<u64, NodeId>,
}

impl ProgressiveRevealCapRegistry {
    pub fn admit_section(
        &mut self,
        section_xz: &[[f32; 2]],
        material_id: u32,
        proto_ids: &mut ProtoIdAllocator,
    ) -> Result<(ProgressiveRevealCapAdmission, Option<SceneDelta>), ProgressiveRevealCapError>
    {
        let mesh = progressive_reveal_cap_prototype(section_xz, material_id)?;
        let key = RuntimeGeometrySource::from_proto_mesh(&mesh)
            .map_err(|_| ProgressiveRevealCapError::TriangulationFailed)?
            .key();
        if let Some(&proto) = self.prototypes.get(&key) {
            return Ok((
                ProgressiveRevealCapAdmission {
                    key,
                    proto,
                    newly_admitted: false,
                },
                None,
            ));
        }
        let proto = proto_ids.allocate();
        self.prototypes.insert(key, proto);
        Ok((
            ProgressiveRevealCapAdmission {
                key,
                proto,
                newly_admitted: true,
            },
            Some(SceneDelta::AddProto { proto, mesh }),
        ))
    }

    pub fn admit_frontier(
        &mut self,
        vertices: &[[f32; 3]],
        material_id: u32,
        proto_ids: &mut ProtoIdAllocator,
    ) -> Result<(ProgressiveRevealCapAdmission, Option<SceneDelta>), ProgressiveRevealCapError>
    {
        let mesh = progressive_reveal_frontier_prototype(vertices, material_id)?;
        let key = RuntimeGeometrySource::from_proto_mesh(&mesh)
            .map_err(|_| ProgressiveRevealCapError::TriangulationFailed)?
            .key();
        if let Some(&proto) = self.prototypes.get(&key) {
            return Ok((
                ProgressiveRevealCapAdmission {
                    key,
                    proto,
                    newly_admitted: false,
                },
                None,
            ));
        }
        let proto = proto_ids.allocate();
        self.prototypes.insert(key, proto);
        Ok((
            ProgressiveRevealCapAdmission {
                key,
                proto,
                newly_admitted: true,
            },
            Some(SceneDelta::AddProto { proto, mesh }),
        ))
    }

    pub fn begin_instance(
        &mut self,
        project_id: u64,
        prototype: RuntimeGeometryKey,
        transform: SceneTransform,
        material_base: u32,
        node_ids: &mut NodeIdAllocator,
    ) -> Result<SceneDelta, ProgressiveRevealCapError> {
        let &proto = self
            .prototypes
            .get(&prototype)
            .ok_or(ProgressiveRevealCapError::MissingPrototype)?;
        if self.instances.contains_key(&project_id) {
            return Err(ProgressiveRevealCapError::DuplicateInstance);
        }
        let id = node_ids.allocate();
        self.instances.insert(project_id, id);
        Ok(SceneDelta::AddNode {
            id,
            proto,
            transform,
            material_base,
        })
    }

    pub fn set_instance_transform(
        &self,
        project_id: u64,
        transform: SceneTransform,
    ) -> Result<SceneDelta, ProgressiveRevealCapError> {
        let &id = self
            .instances
            .get(&project_id)
            .ok_or(ProgressiveRevealCapError::MissingInstance)?;
        Ok(SceneDelta::SetTransform { id, transform })
    }

    pub fn complete_instance(
        &mut self,
        project_id: u64,
    ) -> Result<SceneDelta, ProgressiveRevealCapError> {
        let id = self
            .instances
            .remove(&project_id)
            .ok_or(ProgressiveRevealCapError::MissingInstance)?;
        Ok(SceneDelta::RemoveNode { id })
    }

    pub fn prototype_count(&self) -> usize {
        self.prototypes.len()
    }

    pub fn active_instance_count(&self) -> usize {
        self.instances.len()
    }
}

/// Triangulate an authored X/Z section polygon into an immutable upward-facing
/// cap prototype. Concave simple polygons are supported; holes are represented
/// as separate authored sections rather than guessed here.
pub fn progressive_reveal_cap_prototype(
    section_xz: &[[f32; 2]],
    material_id: u32,
) -> Result<ProtoMesh, ProgressiveRevealCapError> {
    let vertices = section_xz
        .iter()
        .map(|point| [point[0], 0.0, point[1]])
        .collect::<Vec<_>>();
    progressive_reveal_frontier_prototype(&vertices, material_id)
}

/// Triangulate a simple authored planar 3-D frontier. This is shared by
/// horizontal rise caps and vertical horizontal-reveal frontiers.
pub fn progressive_reveal_frontier_prototype(
    vertices: &[[f32; 3]],
    material_id: u32,
) -> Result<ProtoMesh, ProgressiveRevealCapError> {
    if vertices.len() < 3 {
        return Err(ProgressiveRevealCapError::TooFewVertices);
    }
    if vertices
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(ProgressiveRevealCapError::NonFiniteVertex);
    }
    let origin = vertices[0];
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross3 = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |a: [f32; 3]| dot(a, a).sqrt();
    let edge = sub(vertices[1], origin);
    let edge_len = norm(edge);
    if edge_len <= EPS as f32 {
        return Err(ProgressiveRevealCapError::DegenerateEdge);
    }
    let u = [edge[0] / edge_len, edge[1] / edge_len, edge[2] / edge_len];
    let mut normal = vertices
        .iter()
        .skip(2)
        .map(|&point| cross3(u, sub(point, origin)))
        .find(|candidate| norm(*candidate) > EPS as f32)
        .ok_or(ProgressiveRevealCapError::DegeneratePolygon)?;
    let normal_len = norm(normal);
    normal = [
        normal[0] / normal_len,
        normal[1] / normal_len,
        normal[2] / normal_len,
    ];
    let dominant = if normal[0].abs() >= normal[1].abs() && normal[0].abs() >= normal[2].abs() {
        normal[0]
    } else if normal[1].abs() >= normal[2].abs() {
        normal[1]
    } else {
        normal[2]
    };
    if dominant < 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    if vertices
        .iter()
        .any(|&point| dot(sub(point, origin), normal).abs() > 1.0e-4)
    {
        return Err(ProgressiveRevealCapError::NonPlanarPolygon);
    }
    let v = cross3(normal, u);
    let projected = vertices
        .iter()
        .map(|&point| {
            let d = sub(point, origin);
            [dot(d, u), dot(d, v)]
        })
        .collect::<Vec<_>>();
    for i in 0..projected.len() {
        let a = projected[i];
        let b = projected[(i + 1) % projected.len()];
        if distance_squared(a, b) <= EPS {
            return Err(ProgressiveRevealCapError::DegenerateEdge);
        }
    }
    if polygon_self_intersects(&projected) {
        return Err(ProgressiveRevealCapError::SelfIntersectingPolygon);
    }

    let signed_area = polygon_signed_area(&projected);
    if signed_area.abs() <= EPS {
        return Err(ProgressiveRevealCapError::DegeneratePolygon);
    }
    let mut order: Vec<usize> = (0..projected.len()).collect();
    if signed_area < 0.0 {
        order.reverse();
    }
    let triangles = ear_clip(&projected, order)?;

    let min_x = projected
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min);
    let max_x = projected
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let min_z = projected
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min);
    let max_z = projected
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let width = (max_x - min_x).max(f32::EPSILON);
    let depth = (max_z - min_z).max(f32::EPSILON);

    Ok(ProtoMesh {
        positions: vertices.to_vec(),
        normals: vec![normal; vertices.len()],
        uvs: projected
            .iter()
            .map(|point| [(point[0] - min_x) / width, (point[1] - min_z) / depth])
            .collect(),
        material_ids: vec![material_id; triangles.len()],
        indices: triangles,
        aabb_min: std::array::from_fn(|axis| {
            vertices
                .iter()
                .map(|p| p[axis])
                .fold(f32::INFINITY, f32::min)
        }),
        aabb_max: std::array::from_fn(|axis| {
            vertices
                .iter()
                .map(|p| p[axis])
                .fold(f32::NEG_INFINITY, f32::max)
        }),
    })
}

fn ear_clip(
    points: &[[f32; 2]],
    mut polygon: Vec<usize>,
) -> Result<Vec<[u32; 3]>, ProgressiveRevealCapError> {
    let mut triangles = Vec::with_capacity(points.len() - 2);
    while polygon.len() > 3 {
        let mut clipped = false;
        for cursor in 0..polygon.len() {
            let previous = polygon[(cursor + polygon.len() - 1) % polygon.len()];
            let current = polygon[cursor];
            let next = polygon[(cursor + 1) % polygon.len()];
            if cross(points[previous], points[current], points[next]) <= EPS {
                continue;
            }
            if polygon.iter().copied().any(|candidate| {
                candidate != previous
                    && candidate != current
                    && candidate != next
                    && point_in_triangle(
                        points[candidate],
                        points[previous],
                        points[current],
                        points[next],
                    )
            }) {
                continue;
            }
            triangles.push([previous as u32, current as u32, next as u32]);
            polygon.remove(cursor);
            clipped = true;
            break;
        }
        if !clipped {
            return Err(ProgressiveRevealCapError::TriangulationFailed);
        }
    }
    triangles.push([polygon[0] as u32, polygon[1] as u32, polygon[2] as u32]);
    Ok(triangles)
}

fn polygon_signed_area(points: &[[f32; 2]]) -> f64 {
    (0..points.len())
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            f64::from(a[0]) * f64::from(b[1]) - f64::from(b[0]) * f64::from(a[1])
        })
        .sum::<f64>()
        * 0.5
}

fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f64 {
    (f64::from(b[0]) - f64::from(a[0])) * (f64::from(c[1]) - f64::from(a[1]))
        - (f64::from(b[1]) - f64::from(a[1])) * (f64::from(c[0]) - f64::from(a[0]))
}

fn point_in_triangle(point: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    cross(a, b, point) >= -EPS && cross(b, c, point) >= -EPS && cross(c, a, point) >= -EPS
}

fn distance_squared(a: [f32; 2], b: [f32; 2]) -> f64 {
    let dx = f64::from(a[0]) - f64::from(b[0]);
    let dz = f64::from(a[1]) - f64::from(b[1]);
    dx * dx + dz * dz
}

fn polygon_self_intersects(points: &[[f32; 2]]) -> bool {
    for a in 0..points.len() {
        let a_next = (a + 1) % points.len();
        for b in (a + 1)..points.len() {
            let b_next = (b + 1) % points.len();
            if a == b || a_next == b || b_next == a {
                continue;
            }
            if segments_intersect(points[a], points[a_next], points[b], points[b_next]) {
                return true;
            }
        }
    }
    false
}

fn segments_intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> bool {
    let ab_c = cross(a, b, c);
    let ab_d = cross(a, b, d);
    let cd_a = cross(c, d, a);
    let cd_b = cross(c, d, b);
    ab_c * ab_d < -EPS && cd_a * cd_b < -EPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_scene::SceneGraph;

    #[test]
    fn concave_section_becomes_one_shared_closed_frontier_mesh() {
        let cap = progressive_reveal_cap_prototype(
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [2.0, 1.5], [0.0, 3.0]],
            17,
        )
        .unwrap();
        assert_eq!(cap.positions.len(), 5);
        assert_eq!(cap.indices.len(), 3);
        assert_eq!(cap.material_ids, vec![17; 3]);
        assert!(cap.normals.iter().all(|normal| *normal == [0.0, 1.0, 0.0]));
        assert_eq!(cap.aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(cap.aabb_max, [4.0, 0.0, 3.0]);
    }

    #[test]
    fn clockwise_authored_section_is_normalized_to_upward_triangles() {
        let cap =
            progressive_reveal_cap_prototype(&[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], 0)
                .unwrap();
        assert_eq!(cap.indices.len(), 2);
        for triangle in &cap.indices {
            let a = cap.positions[triangle[0] as usize];
            let b = cap.positions[triangle[1] as usize];
            let c = cap.positions[triangle[2] as usize];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let normal_y = ab[2] * ac[0] - ab[0] * ac[2];
            assert!(normal_y > 0.0);
        }
    }

    #[test]
    fn vertical_frontier_is_planar_and_faces_consistently() {
        let frontier = progressive_reveal_frontier_prototype(
            &[
                [0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 3.0, 4.0],
                [0.0, 0.0, 4.0],
            ],
            9,
        )
        .unwrap();
        assert_eq!(frontier.indices.len(), 2);
        assert_eq!(frontier.material_ids, vec![9, 9]);
        assert!(
            frontier
                .normals
                .iter()
                .all(|normal| *normal == [1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn self_intersecting_section_fails_closed() {
        assert_eq!(
            progressive_reveal_cap_prototype(&[[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]], 0,),
            Err(ProgressiveRevealCapError::SelfIntersectingPolygon)
        );
    }

    #[test]
    fn identical_cap_sections_share_one_runtime_geometry_key() {
        let first =
            progressive_reveal_cap_prototype(&[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], 17)
                .unwrap();
        let second = first.clone();
        let first_source = RuntimeGeometrySource::from_proto_mesh(&first).unwrap();
        let second_source = RuntimeGeometrySource::from_proto_mesh(&second).unwrap();
        assert_eq!(first_source.key(), second_source.key());
    }

    #[test]
    fn two_projects_share_one_cap_proto_and_complete_independently() {
        let section = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]];
        let mut registry = ProgressiveRevealCapRegistry::default();
        let mut proto_ids = ProtoIdAllocator::new();
        let mut node_ids = NodeIdAllocator::new();
        let mut graph = SceneGraph::new();

        let (first, add_proto) = registry
            .admit_section(&section, 17, &mut proto_ids)
            .unwrap();
        graph.apply(&[add_proto.unwrap()]).unwrap();
        let (second, duplicate_proto) = registry
            .admit_section(&section, 17, &mut proto_ids)
            .unwrap();
        assert_eq!(first.key, second.key);
        assert_eq!(first.proto, second.proto);
        assert!(duplicate_proto.is_none());

        let first_node = registry
            .begin_instance(
                101,
                first.key,
                SceneTransform::from_translation(10.0, 8.0, 20.0),
                0,
                &mut node_ids,
            )
            .unwrap();
        let second_node = registry
            .begin_instance(
                102,
                first.key,
                SceneTransform::from_translation(40.0, 12.0, 50.0),
                0,
                &mut node_ids,
            )
            .unwrap();
        graph.apply(&[first_node, second_node]).unwrap();
        assert_eq!(registry.prototype_count(), 1);
        assert_eq!(registry.active_instance_count(), 2);
        assert_eq!(graph.live_proto_count(), 1);
        assert_eq!(graph.live_node_count(), 2);

        let move_first = registry
            .set_instance_transform(101, SceneTransform::from_translation(10.0, 16.0, 20.0))
            .unwrap();
        graph.apply(&[move_first]).unwrap();
        graph
            .apply(&[registry.complete_instance(101).unwrap()])
            .unwrap();
        assert_eq!(registry.active_instance_count(), 1);
        assert_eq!(graph.live_proto_count(), 1);
        assert_eq!(graph.live_node_count(), 1);
    }
}
