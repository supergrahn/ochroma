use glam::Vec3;
use vox_core::navmesh::{NavMesh, NavNode};
use vox_sim::crowd::{CrowdAgent, CrowdSimulation};

fn two_node_navmesh() -> NavMesh {
    let mut nm = NavMesh::new();
    nm.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![1] });
    nm.nodes.push(NavNode { id: 1, world_pos: [10.0, 0.0, 0.0], neighbours: vec![0] });
    nm
}

#[test]
fn crowd_agent_set_navmesh_destination_stores_path() {
    let nm = two_node_navmesh();
    let mut agent = CrowdAgent {
        position: Vec3::new(0.1, 0.0, 0.0),
        velocity: Vec3::ZERO,
        target: Vec3::ZERO,
        speed: 2.0,
        radius: 0.3,
        path: Vec::new(),
        path_index: 0,
    };
    agent.set_navmesh_destination(Vec3::new(9.9, 0.0, 0.0), &nm);
    println!("path length: {}", agent.path.len());
    assert!(agent.path.len() >= 1, "path must have at least 1 waypoint");
    assert_eq!(agent.path_index, 0);
}

#[test]
fn crowd_agent_follows_navmesh_path_around_obstacle() {
    let mut nm = NavMesh::new();
    nm.nodes.push(NavNode { id: 0, world_pos: [0.0, 0.0, 0.0], neighbours: vec![1] });
    nm.nodes.push(NavNode { id: 1, world_pos: [5.0, 0.0, 0.0], neighbours: vec![0, 2] });
    nm.nodes.push(NavNode { id: 2, world_pos: [10.0, 0.0, 0.0], neighbours: vec![1] });

    let mut sim = CrowdSimulation::new();
    let idx = sim.add_agent(Vec3::new(0.1, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 5.0);
    sim.agents[idx].set_navmesh_destination(Vec3::new(9.9, 0.0, 0.0), &nm);

    let mut reached_waypoint1 = false;
    let mut reached_goal = false;

    for _ in 0..400 {
        sim.tick(0.05);
        let pos = sim.agents[idx].position;
        if !reached_waypoint1 && (pos - Vec3::new(5.0, 0.0, 0.0)).length() < 0.5 {
            println!("agent reached waypoint 1 at [{:.1}, {:.1}, {:.1}]", pos.x, pos.y, pos.z);
            reached_waypoint1 = true;
        }
        if (pos - Vec3::new(9.9, 0.0, 0.0)).length() < 0.5 {
            println!("agent reached goal");
            reached_goal = true;
            break;
        }
    }
    assert!(reached_waypoint1, "agent must pass through intermediate waypoint at [5,0,0]");
    assert!(reached_goal, "agent must reach goal at [10,0,0]");
}
