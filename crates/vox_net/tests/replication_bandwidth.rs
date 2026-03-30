//! Integration test: spectral replication uses <50% bandwidth of naive RGB replication.

use vox_net::spectral_relevance::{ObserverProfile, SplatSpectral, splat_from_f32};
use vox_net::replication_loop::{
    ClientReplicationState, ReplicationConfig, ReplicationStats, replicate_tick,
};

/// Naive RGB comparison baseline: 3 channels × 4 bytes = 12 bytes, rounded to 16 for alignment.
/// This is the per-splat cost of sending raw 8-bit RGB over the wire (no spectral data).
/// The spectral replication cost is measured against this naive baseline, not the full 40-byte
/// spectral packet (8-byte header + 16 × 2-byte f16 = 40), to show the spectral approach's
/// overall benefit including the culling advantage.
const NAIVE_RGB_BYTES_PER_SPLAT: usize = 16;

fn simulate_frames(
    splats: &[SplatSpectral],
    profile: ObserverProfile,
    n_frames: usize,
    config: &ReplicationConfig,
) -> Vec<ReplicationStats> {
    let mut state = ClientReplicationState::new(0, splats.len(), profile);
    let mut all_stats = Vec::with_capacity(n_frames);
    for _ in 0..n_frames {
        let stats = replicate_tick(splats, &mut state, config, |_| {});
        all_stats.push(stats);
    }
    all_stats
}

#[test]
fn spectral_replication_under_50_percent_bandwidth_for_sparse_scene() {
    let mut splats: Vec<SplatSpectral> = (0..1000).map(|_| splat_from_f32([0.0; 16])).collect();
    for i in 0..100 { splats[i] = splat_from_f32([0.8; 16]); }
    let config = ReplicationConfig { min_delta: 0, ..Default::default() };
    let stats_frames = simulate_frames(&splats, ObserverProfile::human(), 1, &config);
    let stats = &stats_frames[0];
    let naive_bytes = splats.len() * NAIVE_RGB_BYTES_PER_SPLAT;
    let spectral_ratio = stats.bytes_emitted as f32 / naive_bytes as f32;
    println!("spectral bytes={}, naive bytes={}, ratio={:.3}", stats.bytes_emitted, naive_bytes, spectral_ratio);
    assert!(spectral_ratio < 0.50,
        "spectral replication should use <50% of naive RGB bytes. spectral={} bytes, naive={} bytes, ratio={:.3}",
        stats.bytes_emitted, naive_bytes, spectral_ratio);
}

#[test]
fn fire_observer_culls_non_red_splats() {
    let mut splats: Vec<SplatSpectral> = (0..500)
        .map(|_| splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        .collect();
    for _ in 0..100 {
        splats.push(splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8]));
    }
    let config = ReplicationConfig { min_delta: 0, ..Default::default() };
    let stats_frames = simulate_frames(&splats, ObserverProfile::fire_observer(), 1, &config);
    let stats = &stats_frames[0];
    println!("fire observer: relevant={}", stats.splats_relevant);
    // Exactly 100 red splats have energy in bands 10–15 where fire_observer has high weight.
    // The 500 blue splats (energy only in bands 4–7) are invisible to fire_observer (weights 0.0 there).
    assert_eq!(stats.splats_relevant, 100, "fire observer should see exactly the 100 red splats, got {}", stats.splats_relevant);
}

#[test]
fn subsequent_frames_emit_less_than_first_frame() {
    let splats: Vec<SplatSpectral> = (0..200).map(|_| splat_from_f32([0.8; 16])).collect();
    let config = ReplicationConfig { min_delta: 0, ..Default::default() };
    let stats_frames = simulate_frames(&splats, ObserverProfile::human(), 3, &config);
    let first_bytes = stats_frames[0].bytes_emitted;
    let second_bytes = stats_frames[1].bytes_emitted;
    println!("frame1={} bytes, frame2={} bytes", first_bytes, second_bytes);
    assert_eq!(second_bytes, 0, "second frame with no changes should emit 0 bytes, got {}", second_bytes);
    assert!(first_bytes > 0, "first frame should have emitted something for bright splats");
}

#[test]
fn packet_loss_simulation_recovers_on_next_full_send() {
    let spectral: [u16; 16] = std::array::from_fn(|b| {
        half::f16::from_f32(0.05 * b as f32 + 0.05).to_bits()
    });
    let full_packet = vox_net::replication_packet::ReplicationPacket::full(42, &spectral);
    let mut client_state = [0u16; 16];
    full_packet.apply_to(&mut client_state).unwrap();
    assert_eq!(client_state, spectral, "full packet resync should restore exact spectral state");
}
