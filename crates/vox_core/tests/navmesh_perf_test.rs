use std::time::Instant;
use glam::Vec2;
use vox_core::navmesh::NavMesh;

#[test]
fn navmesh_nearest_node_1000_nodes_under_1ms() {
    let mut nm = NavMesh::new();
    for i in 0u32..1000 {
        let x = (i % 32) as f32 * 3.16;
        let z = (i / 32) as f32 * 3.16;
        nm.add_node(i, Vec2::new(x, z), true);
    }
    nm.rebuild_grid();

    let query_pos = Vec2::new(50.0, 50.0);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = nm.nearest_node(query_pos);
    }
    let elapsed_us = start.elapsed().as_micros();
    let per_query_us = elapsed_us / 100;
    println!("nearest_node 1000 nodes in {}µs per query", per_query_us);
    assert!(
        per_query_us < 1000,
        "nearest_node must be < 1ms per query on 1000 nodes, got {}µs",
        per_query_us
    );

    let result = nm.nearest_node(query_pos).unwrap();
    let found = nm.nodes.iter().find(|n| n.id == result).unwrap();
    let dist = (found.position - query_pos).length();
    assert!(dist < 4.0, "nearest node must be within 4m of query, got {:.2}m", dist);
}
