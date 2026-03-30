use std::time::Instant;
use vox_core::navmesh::{NavMesh, NavNode};

#[test]
fn navmesh_nearest_node_1000_nodes_under_1ms() {
    let mut nm = NavMesh::new();
    for i in 0u32..1000 {
        let x = (i % 32) as f32 * 3.16;
        let z = (i / 32) as f32 * 3.16;
        nm.nodes.push(NavNode { id: i, world_pos: [x, 0.0, z], neighbours: vec![] });
    }
    nm.rebuild_grid();

    let query_pos = [50.0f32, 0.0, 50.0];
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
    let dist = ((found.world_pos[0] - 50.0).powi(2) + (found.world_pos[2] - 50.0).powi(2)).sqrt();
    assert!(dist < 4.0, "nearest node must be within 4m of query, got {:.2}m", dist);
}
