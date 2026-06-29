//! Focused scaling bench for `CrowdSimulation::tick` — the per-agent avoidance
//! pass that the spatial-hash-reuse + rayon optimization targets. Measures
//! wall-time per tick at several densely-packed populations.

use std::time::Instant;

use glam::Vec3;
use vox_sim::crowd::CrowdSimulation;

fn make_crowd(n: usize) -> CrowdSimulation {
    let mut sim = CrowdSimulation::new();
    for i in 0..n {
        // 0.8 m spacing -> dense neighbour cells (real avoidance work).
        let x = (i % 256) as f32 * 0.8;
        let z = (i / 256) as f32 * 0.8;
        sim.add_agent(
            Vec3::new(x, 0.0, z),
            Vec3::new(5000.0, 0.0, 5000.0),
            1.5 + (i % 3) as f32 * 0.25,
        );
    }
    sim
}

fn time_it(label: &str, iters: u32, mut f: impl FnMut()) {
    f(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total = start.elapsed();
    let per = total.as_secs_f64() / iters as f64;
    println!(
        "{label:<34} {:>10.3} ms/tick   ({} iters, {:.1} ms total)",
        per * 1000.0,
        iters,
        total.as_secs_f64() * 1000.0
    );
}

fn main() {
    println!("=== CrowdSimulation::tick scaling ===\n");
    for &n in &[1_000usize, 10_000, 50_000, 100_000, 250_000] {
        let mut sim = make_crowd(n);
        let iters = if n >= 100_000 { 20 } else { 40 };
        time_it(&format!("crowd.tick  n={n}"), iters, || {
            sim.tick(1.0 / 60.0);
        });
    }
}
