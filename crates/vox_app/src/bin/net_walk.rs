//! Headless 2-player networked walk demo over vox_net's real QUIC loopback.
//!
//! Run with: `cargo run --bin net_walk`
//!
//! A host and a client each own one walking player (position + 16-band spectral
//! signature). Over 60 ticks (~16 ms apart) the host walks +X at 2 m/s and the
//! client walks +Z at 3 m/s; every tick each side replicates its own state to the
//! peer through vox_net's QUIC transport (real TLS 1.3 handshake, real loopback
//! socket, real per-stream packets). Afterwards each side asserts it tracked the
//! OTHER player's final position to within one tick of movement and that it
//! received more than 30 packets. On success it prints the NET SUMMARY line and
//! exits 0; any failed assertion panics and exits non-zero.

use vox_net::{run_loopback_walk_demo, WalkDemoConfig};

fn main() {
    let cfg = WalkDemoConfig::default();

    let report = run_loopback_walk_demo(cfg).expect("[net_walk] QUIC loopback walk demo failed");

    // Predicted final positions from the deterministic movement math.
    // host:  2.0 m/s * 0.016 s * 60 ticks = 1.92 m along +X
    // client: 3.0 m/s * 0.016 s * 60 ticks = 2.88 m along +Z
    let host_exp = cfg.expected_host_final();
    let client_exp = cfg.expected_client_final();
    let host_tol = cfg.host_tick_step(); // one tick of host X movement (0.032 m)
    let client_tol = cfg.client_tick_step(); // one tick of client Z movement (0.048 m)

    // Print both sides' view of both players before asserting, for transparency.
    println!(
        "[net_walk] host own final     = ({:.4}, {:.4}, {:.4})",
        report.host_final[0], report.host_final[1], report.host_final[2]
    );
    println!(
        "[net_walk] client own final   = ({:.4}, {:.4}, {:.4})",
        report.client_final[0], report.client_final[1], report.client_final[2]
    );
    println!(
        "[net_walk] host saw client    = ({:.4}, {:.4}, {:.4})  spectral[0]={}",
        report.host_saw_client[0], report.host_saw_client[1], report.host_saw_client[2],
        report.host_saw_client_spectral[0]
    );
    println!(
        "[net_walk] client saw host    = ({:.4}, {:.4}, {:.4})  spectral[0]={}",
        report.client_saw_host[0], report.client_saw_host[1], report.client_saw_host[2],
        report.client_saw_host_spectral[0]
    );
    println!(
        "[net_walk] expected: host.x={:.4}  client.z={:.4}  (tol host={:.4} client={:.4})",
        host_exp[0], client_exp[2], host_tol, client_tol
    );

    // Each side must have received more than 30 packets from its peer.
    assert!(
        report.msgs_host > 30,
        "[net_walk] host received only {} packets (need > 30)",
        report.msgs_host
    );
    assert!(
        report.msgs_client > 30,
        "[net_walk] client received only {} packets (need > 30)",
        report.msgs_client
    );

    // Own simulated positions must match the deterministic math exactly.
    assert!(
        (report.host_final[0] - host_exp[0]).abs() < 1e-4,
        "[net_walk] host own x {} != expected {}",
        report.host_final[0], host_exp[0]
    );
    assert!(
        (report.client_final[2] - client_exp[2]).abs() < 1e-4,
        "[net_walk] client own z {} != expected {}",
        report.client_final[2], client_exp[2]
    );

    // Each side tracked the OTHER's final position to within one tick of movement.
    assert!(
        (report.client_saw_host[0] - host_exp[0]).abs() <= host_tol + 1e-4,
        "[net_walk] client_saw_host.x {} not within one tick of host.x {}",
        report.client_saw_host[0], host_exp[0]
    );
    assert!(
        (report.host_saw_client[2] - client_exp[2]).abs() <= client_tol + 1e-4,
        "[net_walk] host_saw_client.z {} not within one tick of client.z {}",
        report.host_saw_client[2], client_exp[2]
    );

    println!(
        "[net_walk] NET SUMMARY: host_saw_client=({:.2},{:.2}) client_saw_host=({:.2},{:.2}) msgs_host={} msgs_client={} PASS",
        report.host_saw_client[0],
        report.host_saw_client[2],
        report.client_saw_host[0],
        report.client_saw_host[2],
        report.msgs_host,
        report.msgs_client
    );
}
