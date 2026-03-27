use vox_core::navmesh::NavMesh;
use glam::Vec2;

#[test]
fn find_direct_path() {
    let mut mesh = NavMesh::new();
    mesh.add_node(0, Vec2::new(0.0, 0.0), true);
    mesh.add_node(1, Vec2::new(10.0, 0.0), true);
    mesh.add_edge(0, 1);
    let path = mesh.find_path(0, 1).unwrap();
    assert_eq!(path, vec![0, 1]);
}

#[test]
fn find_multi_hop_path() {
    let mut mesh = NavMesh::new();
    mesh.add_node(0, Vec2::new(0.0, 0.0), true);
    mesh.add_node(1, Vec2::new(5.0, 0.0), true);
    mesh.add_node(2, Vec2::new(10.0, 0.0), true);
    mesh.add_edge(0, 1);
    mesh.add_edge(1, 2);
    let path = mesh.find_path(0, 2).unwrap();
    assert_eq!(path, vec![0, 1, 2]);
}

#[test]
fn no_path_when_disconnected() {
    let mut mesh = NavMesh::new();
    mesh.add_node(0, Vec2::new(0.0, 0.0), true);
    mesh.add_node(1, Vec2::new(10.0, 0.0), true);
    // No edge
    assert!(mesh.find_path(0, 1).is_none());
}

#[test]
fn avoids_unwalkable_nodes() {
    let mut mesh = NavMesh::new();
    mesh.add_node(0, Vec2::new(0.0, 0.0), true);
    mesh.add_node(1, Vec2::new(5.0, 0.0), false); // blocked!
    mesh.add_node(2, Vec2::new(10.0, 0.0), true);
    mesh.add_node(3, Vec2::new(5.0, 5.0), true);  // detour
    mesh.add_edge(0, 1);
    mesh.add_edge(1, 2);
    mesh.add_edge(0, 3);
    mesh.add_edge(3, 2);
    let path = mesh.find_path(0, 2).unwrap();
    assert!(!path.contains(&1), "Should avoid unwalkable node 1");
    assert!(path.contains(&3), "Should go through detour node 3");
}

#[test]
fn nearest_node_finds_closest() {
    let mut mesh = NavMesh::new();
    mesh.add_node(0, Vec2::new(0.0, 0.0), true);
    mesh.add_node(1, Vec2::new(100.0, 0.0), true);
    assert_eq!(mesh.nearest_node(Vec2::new(10.0, 0.0)), Some(0));
    assert_eq!(mesh.nearest_node(Vec2::new(90.0, 0.0)), Some(1));
}

#[test]
fn a_star_finds_shortest_path() {
    let mut mesh = NavMesh::new();
    // Diamond shape: 0 -> 1 (short) -> 3, or 0 -> 2 (long) -> 3
    mesh.add_node(0, Vec2::new(0.0, 0.0), true);
    mesh.add_node(1, Vec2::new(1.0, 1.0), true);   // short path
    mesh.add_node(2, Vec2::new(-5.0, 5.0), true);   // long detour
    mesh.add_node(3, Vec2::new(2.0, 0.0), true);
    mesh.add_edge(0, 1);
    mesh.add_edge(1, 3);
    mesh.add_edge(0, 2);
    mesh.add_edge(2, 3);
    let path = mesh.find_path(0, 3).unwrap();
    assert!(path.contains(&1), "Should take the short path through node 1");
}
