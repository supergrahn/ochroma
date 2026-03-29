use vox_core::navmesh::{NavMesh, NavNode};

#[test]
fn find_direct_path() {
    let mut mesh = NavMesh::new();
    mesh.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![1] });
    mesh.nodes.push(NavNode { id: 1, world_pos: [10.0, 0.0, 0.0], neighbours: vec![0] });
    let path = mesh.find_path(0, 1).unwrap();
    assert_eq!(path.len(), 2);
    assert!((path[0][0] - 0.0).abs() < 0.01);
    assert!((path[1][0] - 10.0).abs() < 0.01);
}

#[test]
fn find_multi_hop_path() {
    let mut mesh = NavMesh::new();
    mesh.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![1] });
    mesh.nodes.push(NavNode { id: 1, world_pos: [5.0, 0.0, 0.0], neighbours: vec![0, 2] });
    mesh.nodes.push(NavNode { id: 2, world_pos: [10.0, 0.0, 0.0], neighbours: vec![1] });
    let path = mesh.find_path(0, 2).unwrap();
    assert_eq!(path.len(), 3);
}

#[test]
fn no_path_when_disconnected() {
    let mut mesh = NavMesh::new();
    mesh.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![] });
    mesh.nodes.push(NavNode { id: 1, world_pos: [10.0, 0.0, 0.0], neighbours: vec![] });
    assert!(mesh.find_path(0, 1).is_none());
}

#[test]
fn a_star_finds_shortest_path() {
    // Diamond: 0 -> 1 (short, x=1) -> 3, or 0 -> 2 (long, x=-5) -> 3
    let mut mesh = NavMesh::new();
    mesh.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![1, 2] });
    mesh.nodes.push(NavNode { id: 1, world_pos: [1.0, 1.0, 0.0], neighbours: vec![0, 3] });
    mesh.nodes.push(NavNode { id: 2, world_pos: [-5.0, 5.0, 0.0], neighbours: vec![0, 3] });
    mesh.nodes.push(NavNode { id: 3, world_pos: [2.0, 0.0, 0.0], neighbours: vec![1, 2] });
    let path = mesh.find_path(0, 3).unwrap();
    // Short path 0->1->3 visits positions [0,0,0], [1,1,0], [2,0,0]
    let visits_node1 = path.iter().any(|p| (p[0] - 1.0).abs() < 0.01 && (p[1] - 1.0).abs() < 0.01);
    assert!(visits_node1, "A* should take the short path through node 1 at (1,1,0)");
}

#[test]
fn nearest_node_finds_closest() {
    let mut mesh = NavMesh::new();
    mesh.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![] });
    mesh.nodes.push(NavNode { id: 1, world_pos: [100.0, 0.0, 0.0], neighbours: vec![] });
    assert_eq!(mesh.nearest_node([10.0, 0.0, 0.0]), Some(0));
    assert_eq!(mesh.nearest_node([90.0, 0.0, 0.0]), Some(1));
}
