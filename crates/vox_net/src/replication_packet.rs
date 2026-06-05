//! Delta-compressed spectral replication packet.
//!
//! Wire format (little-endian):
//!   [entity_id: u32][changed_bands: u32][value_0: u16][value_1: u16]...
//! where value_N is present only if bit N of changed_bands is set.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PacketError {
    #[error("buffer too short: need {needed} bytes, got {have}")]
    BufferTooShort { needed: usize, have: usize },
    #[error("values count {values} does not match popcount of changed_bands {expected}")]
    BandCountMismatch { values: usize, expected: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationPacket {
    pub entity_id: u32,
    pub changed_bands: u32,
    pub values: Vec<u16>,
}

impl ReplicationPacket {
    pub fn from_delta(entity_id: u32, before: &[u16; 16], after: &[u16; 16], min_delta: u16) -> Self {
        let mut changed_bands: u32 = 0;
        let mut values = Vec::with_capacity(16);
        for b in 0..16 {
            let delta = before[b].abs_diff(after[b]);
            if delta > min_delta {
                changed_bands |= 1 << b;
                values.push(after[b]);
            }
        }
        Self { entity_id, changed_bands, values }
    }

    pub fn full(entity_id: u32, spectral: &[u16; 16]) -> Self {
        Self { entity_id, changed_bands: 0xFFFF, values: spectral.to_vec() }
    }

    pub fn apply_to(&self, spectral: &mut [u16; 16]) -> Result<(), PacketError> {
        let expected = self.changed_bands.count_ones() as usize;
        if self.values.len() != expected {
            return Err(PacketError::BandCountMismatch { values: self.values.len(), expected });
        }
        let mut value_idx = 0;
        for b in 0..16 {
            if self.changed_bands & (1 << b) != 0 {
                spectral[b] = self.values[value_idx];
                value_idx += 1;
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.values.len() * 2);
        buf.extend_from_slice(&self.entity_id.to_le_bytes());
        buf.extend_from_slice(&self.changed_bands.to_le_bytes());
        for &v in &self.values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, PacketError> {
        if buf.len() < 8 {
            return Err(PacketError::BufferTooShort { needed: 8, have: buf.len() });
        }
        let entity_id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let changed_bands = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        // Only the lower 16 bits are valid band flags. Bits 16–31 are reserved.
        let band_count = (changed_bands & 0xFFFF).count_ones() as usize;
        let needed = 8 + band_count * 2;
        if buf.len() < needed {
            return Err(PacketError::BufferTooShort { needed, have: buf.len() });
        }
        let mut values = Vec::with_capacity(band_count);
        for i in 0..band_count {
            let offset = 8 + i * 2;
            values.push(u16::from_le_bytes([buf[offset], buf[offset + 1]]));
        }
        Ok(Self { entity_id, changed_bands, values })
    }

    pub fn wire_size(&self) -> usize {
        8 + self.values.len() * 2
    }

    pub fn bandwidth_ratio(&self) -> f32 {
        self.wire_size() as f32 / 40.0
    }
}

/// Full player-state replication packet: world position plus the complete
/// 16-band spectral signature for one networked entity.
///
/// `ReplicationPacket` deliberately carries only spectral band deltas (it has no
/// notion of position). A networked walking player needs to replicate *where* it
/// is, so this packet pairs a 3-component world position with the full spectral
/// signature. Both sides decode a byte-identical struct.
///
/// Wire format (little-endian, fixed 4 + 12 + 32 = 48 bytes):
///   [entity_id: u32][pos_x: f32][pos_y: f32][pos_z: f32][band_0: u16]...[band_15: u16]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerStatePacket {
    pub entity_id: u32,
    pub position: [f32; 3],
    pub spectral: [u16; 16],
}

impl PlayerStatePacket {
    /// Fixed encoded size in bytes: 4 (id) + 12 (3×f32 position) + 32 (16×u16 bands).
    pub const WIRE_SIZE: usize = 4 + 12 + 32;

    pub fn new(entity_id: u32, position: [f32; 3], spectral: [u16; 16]) -> Self {
        Self { entity_id, position, spectral }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::WIRE_SIZE);
        buf.extend_from_slice(&self.entity_id.to_le_bytes());
        for c in &self.position {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        for &v in &self.spectral {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self, PacketError> {
        if buf.len() < Self::WIRE_SIZE {
            return Err(PacketError::BufferTooShort { needed: Self::WIRE_SIZE, have: buf.len() });
        }
        let entity_id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let mut position = [0.0f32; 3];
        for (i, c) in position.iter_mut().enumerate() {
            let o = 4 + i * 4;
            *c = f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        }
        let mut spectral = [0u16; 16];
        for (i, v) in spectral.iter_mut().enumerate() {
            let o = 16 + i * 2;
            *v = u16::from_le_bytes([buf[o], buf[o + 1]]);
        }
        Ok(Self { entity_id, position, spectral })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f16_bits(v: f32) -> u16 {
        half::f16::from_f32(v).to_bits()
    }

    #[test]
    fn from_delta_only_includes_changed_bands() {
        let before: [u16; 16] = [1000; 16];
        let mut after = before;
        after[2] = 2000;
        after[5] = 3000;
        let packet = ReplicationPacket::from_delta(42, &before, &after, 0);
        assert_eq!(packet.changed_bands, (1 << 2) | (1 << 5), "only bands 2 and 5 should be marked changed");
        assert_eq!(packet.values.len(), 2);
        assert_eq!(packet.values[0], 2000, "first value should be band 2");
        assert_eq!(packet.values[1], 3000, "second value should be band 5");
    }

    #[test]
    fn from_delta_with_min_delta_suppresses_noise() {
        let before: [u16; 16] = [1000; 16];
        let mut after = before;
        after[0] = 1010;
        after[3] = 2000;
        let packet = ReplicationPacket::from_delta(1, &before, &after, 50);
        assert_eq!(packet.changed_bands, 1 << 3, "only band 3 should pass min_delta=50 filter");
        assert_eq!(packet.values.len(), 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let after: [u16; 16] = [
            make_f16_bits(0.1), make_f16_bits(0.2), make_f16_bits(0.3), make_f16_bits(0.4),
            make_f16_bits(0.5), make_f16_bits(0.6), make_f16_bits(0.7), make_f16_bits(0.8),
            make_f16_bits(0.1), make_f16_bits(0.2), make_f16_bits(0.3), make_f16_bits(0.4),
            make_f16_bits(0.5), make_f16_bits(0.6), make_f16_bits(0.7), make_f16_bits(0.8),
        ];
        let packet = ReplicationPacket::full(99, &after);
        let encoded = packet.encode();
        let decoded = ReplicationPacket::decode(&encoded).unwrap();
        assert_eq!(decoded.entity_id, 99);
        assert_eq!(decoded.changed_bands, 0xFFFF);
        assert_eq!(decoded.values, packet.values);
    }

    #[test]
    fn apply_to_only_modifies_changed_bands() {
        let mut spectral = [1000u16; 16];
        let packet = ReplicationPacket { entity_id: 7, changed_bands: 0b00001010, values: vec![2222, 4444] };
        packet.apply_to(&mut spectral).unwrap();
        assert_eq!(spectral[0], 1000);
        assert_eq!(spectral[1], 2222);
        assert_eq!(spectral[2], 1000);
        assert_eq!(spectral[3], 4444);
        for b in 4..16 { assert_eq!(spectral[b], 1000, "band {} should be unchanged", b); }
    }

    #[test]
    fn wire_size_scales_with_band_count() {
        let zero = ReplicationPacket { entity_id: 0, changed_bands: 0, values: vec![] };
        let one_band = ReplicationPacket { entity_id: 0, changed_bands: 1, values: vec![0] };
        let all_bands = ReplicationPacket::full(0, &[0u16; 16]);
        assert_eq!(zero.wire_size(), 8);
        assert_eq!(one_band.wire_size(), 10);
        assert_eq!(all_bands.wire_size(), 40);
    }

    #[test]
    fn bandwidth_ratio_full_packet_is_one() {
        let packet = ReplicationPacket::full(0, &[0u16; 16]);
        assert!((packet.bandwidth_ratio() - 1.0).abs() < 1e-5, "full packet bandwidth ratio should be 1.0");
    }

    #[test]
    fn bandwidth_ratio_two_bands_is_under_half() {
        let before = [0u16; 16];
        let mut after = [0u16; 16];
        after[2] = 1000;
        after[6] = 2000;
        let packet = ReplicationPacket::from_delta(0, &before, &after, 0);
        assert!(packet.bandwidth_ratio() < 0.5, "2-band packet should use <50% bandwidth, got {:.2}", packet.bandwidth_ratio());
    }

    #[test]
    fn decode_truncated_buffer_returns_error() {
        let buf = [0u8; 3];
        assert!(matches!(ReplicationPacket::decode(&buf), Err(PacketError::BufferTooShort { .. })));
    }

    #[test]
    fn apply_to_band_count_mismatch_returns_error() {
        let mut spectral = [0u16; 16];
        let packet = ReplicationPacket { entity_id: 0, changed_bands: 0xFFFF, values: vec![1, 2] };
        assert!(matches!(packet.apply_to(&mut spectral), Err(PacketError::BandCountMismatch { .. })));
    }

    #[test]
    fn player_state_encode_decode_roundtrip_preserves_position_and_bands() {
        let spectral: [u16; 16] = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let sent = PlayerStatePacket::new(0xBEEF, [1.92, -3.5, 2.88], spectral);
        let bytes = sent.encode();
        assert_eq!(bytes.len(), PlayerStatePacket::WIRE_SIZE, "fixed 48-byte wire format");

        let got = PlayerStatePacket::decode(&bytes).unwrap();
        assert_eq!(got.entity_id, 0xBEEF, "entity_id must round-trip exactly");
        assert_eq!(got.position[0], 1.92f32, "pos.x must round-trip bit-exact");
        assert_eq!(got.position[1], -3.5f32, "pos.y must round-trip bit-exact");
        assert_eq!(got.position[2], 2.88f32, "pos.z must round-trip bit-exact");
        assert_eq!(got.spectral, spectral, "all 16 bands must round-trip exactly");
        assert_eq!(got, sent, "decoded packet must equal the sent packet field-for-field");
    }

    #[test]
    fn player_state_decode_truncated_buffer_returns_error() {
        let buf = [0u8; PlayerStatePacket::WIRE_SIZE - 1];
        assert!(matches!(
            PlayerStatePacket::decode(&buf),
            Err(PacketError::BufferTooShort { needed: PlayerStatePacket::WIRE_SIZE, .. })
        ));
    }
}
