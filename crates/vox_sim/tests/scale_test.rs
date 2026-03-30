use std::time::Instant;
use glam::Vec3;
use vox_sim::crowd::CrowdSimulation;

#[test]
fn crowd_500_agents_tick_under_16ms() {
    let mut sim = CrowdSimulation::new();
    for i in 0..500 {
        let x = (i % 25) as f32 * 2.0;
        let z = (i / 25) as f32 * 2.0;
        sim.add_agent(
            Vec3::new(x, 0.0, z),
            Vec3::new(50.0 - x, 0.0, 50.0 - z),
            2.0,
        );
    }
    sim.tick(1.0 / 60.0);
    let start = Instant::now();
    sim.tick(1.0 / 60.0);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("500 agents ticked in {:.2}ms", elapsed_ms);
    assert!(
        elapsed_ms < 16.0,
        "500 agents must tick under 16ms, took {:.2}ms",
        elapsed_ms
    );
}

#[test]
fn spatial_hash_neighbours_correct() {
    use vox_sim::spatial_hash::SpatialHash;
    let mut hash = SpatialHash::new(2.0);
    hash.insert(0, Vec3::new(0.0, 0.0, 0.0));
    hash.insert(1, Vec3::new(1.5, 0.0, 0.0));
    hash.insert(2, Vec3::new(3.5, 0.0, 0.0));
    hash.insert(3, Vec3::new(10.0, 0.0, 0.0));

    let mut neighbours = hash.neighbours(Vec3::new(0.0, 0.0, 0.0), 2.5);
    neighbours.sort();
    println!("neighbours: {:?}", neighbours);
    assert!(neighbours.contains(&1), "agent 1 at 1.5m must be in neighbours");
    assert!(neighbours.contains(&2), "agent 2 at 3.5m must be in neighbours (adjacent cell)");
    assert!(!neighbours.contains(&3), "agent 3 at 10m must NOT be in neighbours");
}
