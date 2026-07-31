use vox_data::geometry_clusters::{
    ReadyGeometryCluster, ReadyGeometryClusters, ReadyGeometryClustersError,
};

fn cluster(id: u32, start: u32, count: u32, min_x: f32, max_x: f32) -> ReadyGeometryCluster {
    ReadyGeometryCluster::new(id, start, count, [min_x, 0.0, 0.0], [max_x, 1.0, 1.0]).unwrap()
}

#[test]
fn exact_source_partition_round_trips_deterministically() {
    let ready = ReadyGeometryClusters::new(
        8,
        vec![4, 5, 6, 7, 0, 1, 2, 3],
        vec![cluster(0, 0, 4, 1.0, 2.0), cluster(1, 4, 4, 0.0, 1.0)],
    )
    .unwrap();
    let first = serde_json::to_vec(&ready).unwrap();
    let decoded: ReadyGeometryClusters = serde_json::from_slice(&first).unwrap();
    decoded.validate().unwrap();
    assert_eq!(first, serde_json::to_vec(&decoded).unwrap());
    assert_eq!(decoded.source_triangle_count(), 8);
    assert_eq!(decoded.clusters().len(), 2);
}

#[test]
fn duplicate_source_triangle_is_rejected() {
    let error = ReadyGeometryClusters::new(
        8,
        vec![0, 1, 2, 3, 4, 5, 6, 6],
        vec![cluster(0, 0, 4, 0.0, 1.0), cluster(1, 4, 4, 1.0, 2.0)],
    )
    .unwrap_err();
    assert_eq!(error, ReadyGeometryClustersError::DuplicateTriangle(6));
}

#[test]
fn cluster_ranges_must_tile_the_source_order() {
    let error = ReadyGeometryClusters::new(
        8,
        (0..8).collect(),
        vec![cluster(0, 0, 4, 0.0, 1.0), cluster(1, 3, 4, 1.0, 2.0)],
    )
    .unwrap_err();
    assert_eq!(
        error,
        ReadyGeometryClustersError::NonContiguousClusterRange {
            id: 1,
            expected_start: 4,
            actual_start: 3,
        }
    );
}

#[test]
fn ochroma_derives_a_deterministic_partition_from_the_source_mesh() {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for triangle in 0..513 {
        let first = positions.len() as u32;
        let x = (triangle % 31) as f32;
        let z = (triangle / 31) as f32;
        positions.extend_from_slice(&[[x, 0.0, z], [x + 0.25, 0.0, z], [x, 0.5, z]]);
        indices.push([first, first + 1, first + 2]);
    }
    let first = ReadyGeometryClusters::from_source_mesh(&positions, &indices).unwrap();
    let second = ReadyGeometryClusters::from_source_mesh(&positions, &indices).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.source_triangle_count(), 513);
    assert_eq!(first.clusters().len(), 3);
    assert!(
        first
            .clusters()
            .iter()
            .all(|cluster| (4..=256).contains(&cluster.triangle_count()))
    );
}

#[test]
fn source_partition_rejects_an_index_outside_the_authoritative_mesh() {
    let error =
        ReadyGeometryClusters::from_source_mesh(&[[0.0, 0.0, 0.0]], &[[0, 1, 0]]).unwrap_err();
    assert_eq!(
        error,
        ReadyGeometryClustersError::VertexOutOfRange {
            triangle: 0,
            vertex: 1
        }
    );
}
