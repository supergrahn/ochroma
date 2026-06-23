//! Determinism SHIP GATE for the crowd-simulation hot loop.
//!
//! The crowd avoidance pass was optimized to (a) reuse the spatial hash across
//! ticks and (b) run the per-agent steering in parallel via rayon above a size
//! threshold. Both are only legal if the simulated result stays byte-identical:
//! each agent writes only its own delta slot and reads immutable pre-tick state,
//! so task scheduling order cannot change the outcome.
//!
//! These tests assert a bit-exact (`to_bits`) FNV-1a hash of every agent's
//! position+velocity after a fixed number of ticks is:
//!   - identical across two independent same-seed runs (replay determinism),
//!   - identical between the SERIAL and PARALLEL code paths (the parallelism
//!     introduced no nondeterminism), and
//!   - equal to a recorded golden (regression: the optimization changed speed,
//!     not the simulated result).

use glam::Vec3;
use vox_sim::crowd::CrowdSimulation;

/// Build a deterministic packed crowd of `n` agents on a grid, all heading to a
/// common far target so the avoidance pass does real work every tick.
fn make_crowd(n: usize) -> CrowdSimulation {
    let mut sim = CrowdSimulation::new();
    for i in 0..n {
        // Tight 0.8 m spacing so neighbour cells are densely populated.
        let x = (i % 128) as f32 * 0.8;
        let z = (i / 128) as f32 * 0.8;
        sim.add_agent(
            Vec3::new(x, 0.0, z),
            Vec3::new(500.0, 0.0, 500.0),
            1.5 + (i % 3) as f32 * 0.25,
        );
    }
    sim
}

/// FNV-1a over the bit patterns of every agent's position + velocity, in agent
/// index order (the store is index-stable), so the hash is order-deterministic.
fn crowd_hash(sim: &CrowdSimulation) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bits: u32| {
        for b in bits.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for a in &sim.agents {
        mix(a.position.x.to_bits());
        mix(a.position.y.to_bits());
        mix(a.position.z.to_bits());
        mix(a.velocity.x.to_bits());
        mix(a.velocity.y.to_bits());
        mix(a.velocity.z.to_bits());
    }
    h
}

fn run(n: usize, ticks: u32) -> u64 {
    let mut sim = make_crowd(n);
    for _ in 0..ticks {
        sim.tick(1.0 / 60.0);
    }
    crowd_hash(&sim)
}

#[test]
fn crowd_replay_is_byte_identical_across_runs() {
    // 4096 agents is above PARALLEL_THRESHOLD (2048), so this exercises the
    // rayon path; two same-seed runs must produce identical state.
    let a = run(4096, 60);
    let b = run(4096, 60);
    assert_eq!(a, b, "same-seed crowd runs must be byte-identical (parallel path)");
}

#[test]
fn crowd_serial_equals_parallel() {
    // The strongest guard: tick two IDENTICALLY-seeded crowds the same number of
    // times, one FORCING the serial steering path and one FORCING the parallel
    // path. If the rayon reduction introduced any nondeterminism, the per-agent
    // state would diverge. They must be bit-for-bit identical.
    let mut serial = make_crowd(4096);
    let mut parallel = make_crowd(4096);
    for _ in 0..60 {
        serial.tick_force_serial(1.0 / 60.0);
        parallel.tick_force_parallel(1.0 / 60.0);
    }
    assert_eq!(
        crowd_hash(&serial),
        crowd_hash(&parallel),
        "parallel steering path must be byte-identical to the serial reference"
    );
}

#[test]
fn crowd_replay_hash_matches_golden() {
    // GOLDEN regression value (parallel path, 4096 agents, 60 ticks). Recorded
    // after the spatial-hash-reuse + rayon optimization. A change here means the
    // simulated crowd outcome moved — forbidden for a pure performance change.
    const GOLDEN: u64 = 0x2487_6237_c23a_8c6a;
    let h = run(4096, 60);
    assert_eq!(
        h, GOLDEN,
        "crowd state hash changed (got {h:#018x}); an optimization altered \
         the simulated result, not just its speed — determinism moat broken"
    );
}
