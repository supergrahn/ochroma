//! Server-side replication loop.
//!
//! Each tick: filter splats by spectral relevance per client, delta-compress
//! changed bands, encode packets, and dispatch via transport callback.

use crate::spectral_relevance::{ObserverProfile, SpectralRelevanceFilter, SplatSpectral};
use crate::replication_packet::ReplicationPacket;

#[derive(Debug, Clone)]
pub struct ClientReplicationState {
    pub entity_id_offset: u32,
    pub last_sent: Vec<[u16; 16]>,
    pub observer_profile: ObserverProfile,
}

impl ClientReplicationState {
    pub fn new(entity_id_offset: u32, splat_count: usize, profile: ObserverProfile) -> Self {
        Self { entity_id_offset, last_sent: vec![[0u16; 16]; splat_count], observer_profile: profile }
    }

    pub fn resize(&mut self, count: usize) {
        self.last_sent.resize(count, [0u16; 16]);
    }
}

pub struct ReplicationConfig {
    pub min_delta: u16,
    pub relevance_threshold: f32,
    pub max_packets_per_tick: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self { min_delta: 32, relevance_threshold: 0.05, max_packets_per_tick: 1024 }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ReplicationStats {
    pub splats_total: usize,
    pub splats_relevant: usize,
    pub packets_emitted: usize,
    pub bytes_emitted: usize,
    pub bytes_unculled: usize,
}

impl ReplicationStats {
    pub fn bandwidth_ratio(&self) -> f32 {
        if self.bytes_unculled == 0 { return 0.0; }
        self.bytes_emitted as f32 / self.bytes_unculled as f32
    }
}

pub fn replicate_tick<F>(
    splats: &[SplatSpectral],
    client_state: &mut ClientReplicationState,
    config: &ReplicationConfig,
    mut send: F,
) -> ReplicationStats
where
    F: FnMut(Vec<u8>),
{
    client_state.resize(splats.len());
    let filter = SpectralRelevanceFilter::new(config.relevance_threshold);
    let relevant_indices = filter.filter_indices(splats, &client_state.observer_profile);
    let mut stats = ReplicationStats {
        splats_total: splats.len(),
        splats_relevant: relevant_indices.len(),
        bytes_unculled: splats.len() * 40,
        ..Default::default()
    };
    let mut emitted = 0;
    for &idx in &relevant_indices {
        if emitted >= config.max_packets_per_tick { break; }
        let current = &splats[idx].bands;
        let previous = &client_state.last_sent[idx];
        let packet = ReplicationPacket::from_delta(
            client_state.entity_id_offset + idx as u32,
            previous, current, config.min_delta,
        );
        if packet.changed_bands == 0 { continue; }
        let encoded = packet.encode();
        stats.bytes_emitted += encoded.len();
        stats.packets_emitted += 1;
        client_state.last_sent[idx] = *current;
        send(encoded);
        emitted += 1;
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_relevance::splat_from_f32;

    fn make_state(count: usize) -> ClientReplicationState {
        ClientReplicationState::new(0, count, ObserverProfile::human())
    }

    #[test]
    fn first_tick_with_bright_splats_emits_packets() {
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.8; 16])).collect();
        let mut state = make_state(10);
        let config = ReplicationConfig::default();
        let mut packets = Vec::new();
        let stats = replicate_tick(&splats, &mut state, &config, |p| packets.push(p));
        assert!(stats.packets_emitted > 0, "first tick with bright splats should emit packets");
    }

    #[test]
    fn second_tick_no_change_emits_nothing() {
        let splats: Vec<_> = (0..5).map(|_| splat_from_f32([0.8; 16])).collect();
        let mut state = make_state(5);
        let config = ReplicationConfig { min_delta: 0, ..Default::default() };
        replicate_tick(&splats, &mut state, &config, |_| {});
        let mut packets = Vec::new();
        let stats = replicate_tick(&splats, &mut state, &config, |p| packets.push(p));
        assert_eq!(stats.packets_emitted, 0, "second tick with unchanged data should emit 0 packets");
    }

    #[test]
    fn dark_splats_are_culled_entirely() {
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.0; 16])).collect();
        let mut state = make_state(10);
        let config = ReplicationConfig::default();
        let mut packets = Vec::new();
        let stats = replicate_tick(&splats, &mut state, &config, |p| packets.push(p));
        assert_eq!(stats.splats_relevant, 0, "all-dark splats should be culled");
        assert_eq!(stats.packets_emitted, 0);
    }

    #[test]
    fn bandwidth_ratio_is_below_fifty_percent_for_sparse_changes() {
        let mut splats: Vec<_> = (0..100).map(|_| splat_from_f32([0.0; 16])).collect();
        for i in 0..5 { splats[i] = splat_from_f32([0.8; 16]); }
        let mut state = make_state(100);
        let config = ReplicationConfig { min_delta: 0, ..Default::default() };
        let stats = replicate_tick(&splats, &mut state, &config, |_| {});
        assert!(stats.bandwidth_ratio() < 0.5, "5/100 splats bright should give <50% bandwidth ratio, got {:.3}", stats.bandwidth_ratio());
    }

    #[test]
    fn max_packets_per_tick_is_respected() {
        let splats: Vec<_> = (0..200).map(|_| splat_from_f32([0.9; 16])).collect();
        let mut state = make_state(200);
        let config = ReplicationConfig { max_packets_per_tick: 10, min_delta: 0, ..Default::default() };
        let stats = replicate_tick(&splats, &mut state, &config, |_| {});
        assert!(stats.packets_emitted <= 10, "should respect max_packets_per_tick=10, got {}", stats.packets_emitted);
    }

    #[test]
    fn changed_band_triggers_new_packet_on_subsequent_tick() {
        let mut splat_data = splat_from_f32([0.8; 16]);
        let mut state = make_state(1);
        let config = ReplicationConfig { min_delta: 0, ..Default::default() };
        replicate_tick(&[splat_data], &mut state, &config, |_| {});
        splat_data.bands[3] = half::f16::from_f32(0.1).to_bits();
        let mut packets = Vec::new();
        let stats = replicate_tick(&[splat_data], &mut state, &config, |p| packets.push(p));
        assert_eq!(stats.packets_emitted, 1, "changed band should trigger one new packet");
        let decoded = crate::replication_packet::ReplicationPacket::decode(&packets[0]).unwrap();
        assert_eq!(decoded.changed_bands, 1 << 3, "only band 3 should be in the packet");
    }
}
