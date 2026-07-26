//! Measures the per-frame CPU cost the wind->sea-state wiring adds to the present
//! loop. Run: `cargo run -p vox_water --release --example cost`
use std::time::Instant;
use vox_water::WindResponse;

fn main() {
    let response = WindResponse::default();
    let n = 1_000_000u32;
    let mut sink = 0.0f32;
    for i in 0..10_000u32 {
        sink += response.drive(4.0 + (i % 20) as f32).amplitude_scale;
    }
    let t0 = Instant::now();
    for i in 0..n {
        sink += response.drive(4.0 + (i % 20) as f32).amplitude_scale;
    }
    let elapsed = t0.elapsed();
    println!("sink={sink:.1}");
    println!(
        "WindResponse::drive x{n}: {:?} total, {:.1} ns/call",
        elapsed,
        elapsed.as_secs_f64() * 1e9 / n as f64
    );
    println!(
        "PER-FRAME cost (1 call/frame): {:.6} ms  (budget 33.3 ms)",
        elapsed.as_secs_f64() * 1e3 / n as f64
    );
}
