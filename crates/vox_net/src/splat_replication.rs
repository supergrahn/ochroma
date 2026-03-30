//! Splat delta replication.
//!
//! Transmits only changed splats between frames, using 16-bit quantized deltas.
//! A full scene of 100k splats = 4.8MB. Delta compression with 1% change rate = 48KB.
//!
//! Delta packet: `[u32 splat_id, i8×3 d_pos_q, i8×3 d_scale_q, u8 d_opacity, u8 spectral_changed_mask, u16×K d_spectral]`
//! where spectral_changed_mask is a bitmask of which of the 8 spectral bands changed.

use std::collections::HashMap;

/// Snapshot of a single Gaussian splat for delta comparison.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplatSnapshot {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [i16; 4],
    pub opacity: u8,
    pub spectral: [u16; 8],
}

/// A quantized delta packet representing changes to a single splat.
///
/// `spectral_changed` is a bitmask of which of the 8 spectral bands changed.
/// `d_spectral` has one entry per set bit in `spectral_changed`.
#[derive(Clone, Debug, PartialEq)]
pub struct SplatDeltaPacket {
    pub splat_id: u32,
    pub d_position: [i8; 3],
    pub d_scale: [i8; 3],
    pub d_opacity: i8,
    pub spectral_changed: u8,
    pub d_spectral: Vec<i16>,
}

/// 1 unit of i8 = 1/512 m (about 2mm precision).
pub const POSITION_QUANT: f32 = 512.0;
/// Scale quantization factor.
pub const SCALE_QUANT: f32 = 256.0;
/// Spectral quantization factor (i16 for more precision).
pub const SPECTRAL_QUANT: f32 = 128.0;

/// Compute the delta between two splat snapshots.
///
/// Returns `None` if the splat is unchanged (all deltas zero, no spectral changes).
pub fn diff_splat(
    prev: &SplatSnapshot,
    curr: &SplatSnapshot,
    splat_id: u32,
) -> Option<SplatDeltaPacket> {
    let d_position = [
        ((curr.position[0] - prev.position[0]) * POSITION_QUANT)
            .round()
            .clamp(-127.0, 127.0) as i8,
        ((curr.position[1] - prev.position[1]) * POSITION_QUANT)
            .round()
            .clamp(-127.0, 127.0) as i8,
        ((curr.position[2] - prev.position[2]) * POSITION_QUANT)
            .round()
            .clamp(-127.0, 127.0) as i8,
    ];

    let d_scale = [
        ((curr.scale[0] - prev.scale[0]) * SCALE_QUANT)
            .round()
            .clamp(-127.0, 127.0) as i8,
        ((curr.scale[1] - prev.scale[1]) * SCALE_QUANT)
            .round()
            .clamp(-127.0, 127.0) as i8,
        ((curr.scale[2] - prev.scale[2]) * SCALE_QUANT)
            .round()
            .clamp(-127.0, 127.0) as i8,
    ];

    let d_opacity = (curr.opacity as i16 - prev.opacity as i16).clamp(-127, 127) as i8;

    let mut spectral_changed: u8 = 0;
    let mut d_spectral: Vec<i16> = Vec::new();
    for b in 0..8usize {
        if prev.spectral[b] != curr.spectral[b] {
            spectral_changed |= 1 << b;
            let delta = (curr.spectral[b] as i16).wrapping_sub(prev.spectral[b] as i16);
            d_spectral.push(delta);
        }
    }

    // Return None if nothing changed
    if d_position == [0i8; 3]
        && d_scale == [0i8; 3]
        && d_opacity == 0
        && spectral_changed == 0
    {
        return None;
    }

    Some(SplatDeltaPacket {
        splat_id,
        d_position,
        d_scale,
        d_opacity,
        spectral_changed,
        d_spectral,
    })
}

/// Reconstruct a splat snapshot by applying a delta to a base snapshot.
pub fn apply_delta(base: &SplatSnapshot, delta: &SplatDeltaPacket) -> SplatSnapshot {
    let position = [
        base.position[0] + delta.d_position[0] as f32 / POSITION_QUANT,
        base.position[1] + delta.d_position[1] as f32 / POSITION_QUANT,
        base.position[2] + delta.d_position[2] as f32 / POSITION_QUANT,
    ];

    let scale = [
        base.scale[0] + delta.d_scale[0] as f32 / SCALE_QUANT,
        base.scale[1] + delta.d_scale[1] as f32 / SCALE_QUANT,
        base.scale[2] + delta.d_scale[2] as f32 / SCALE_QUANT,
    ];

    let opacity = (base.opacity as i16 + delta.d_opacity as i16).clamp(0, 255) as u8;

    let mut spectral = base.spectral;
    let mut idx = 0usize;
    #[allow(clippy::needless_range_loop)]
    for b in 0..8usize {
        if delta.spectral_changed & (1 << b) != 0 {
            spectral[b] = spectral[b].wrapping_add(delta.d_spectral[idx] as u16);
            idx += 1;
        }
    }

    SplatSnapshot {
        position,
        scale,
        rotation: base.rotation,
        opacity,
        spectral,
    }
}

/// Replication state holding the latest snapshot for each tracked splat.
pub struct SplatReplicationState {
    pub snapshots: HashMap<u32, SplatSnapshot>,
    pub frame: u64,
}

impl SplatReplicationState {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            frame: 0,
        }
    }

    /// Store (or update) the snapshot for a splat.
    pub fn update_splat(&mut self, id: u32, splat: SplatSnapshot) {
        self.snapshots.insert(id, splat);
    }

    /// Compute delta packets for all splats that changed relative to `prev_state`.
    pub fn compute_deltas(&self, prev_state: &SplatReplicationState) -> Vec<SplatDeltaPacket> {
        let mut deltas = Vec::new();
        for (&id, curr) in &self.snapshots {
            if let Some(prev) = prev_state.snapshots.get(&id)
                && let Some(pkt) = diff_splat(prev, curr, id)
            {
                deltas.push(pkt);
            }
            // Splats not present in prev_state are new; full-snapshot delivery
            // is handled by a separate mechanism (not delta replication).
        }
        deltas
    }

    /// Apply received delta packets to local state.
    pub fn apply_deltas(&mut self, deltas: &[SplatDeltaPacket]) {
        for delta in deltas {
            if let Some(base) = self.snapshots.get(&delta.splat_id).copied() {
                let updated = apply_delta(&base, delta);
                self.snapshots.insert(delta.splat_id, updated);
            }
        }
    }
}

impl Default for SplatReplicationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate wire bytes for a slice of delta packets.
///
/// Per packet: 4 (id) + 3 (pos) + 3 (scale) + 1 (opacity) + 1 (mask) + 2 * popcount(spectral_changed)
pub fn estimate_delta_bytes(deltas: &[SplatDeltaPacket]) -> usize {
    deltas.iter().map(|pkt| {
        4 + 3 + 3 + 1 + 1 + 2 * pkt.spectral_changed.count_ones() as usize
    }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_splat(pos: [f32; 3]) -> SplatSnapshot {
        SplatSnapshot {
            position: pos,
            scale: [1.0, 1.0, 1.0],
            rotation: [0, 0, 0, 16383],
            opacity: 200,
            spectral: [100, 200, 300, 400, 500, 600, 700, 800],
        }
    }

    #[test]
    fn identical_splats_produce_no_delta() {
        let s = make_splat([0.0, 0.0, 0.0]);
        assert_eq!(diff_splat(&s, &s, 0), None);
    }

    #[test]
    fn changed_position_produces_delta() {
        let prev = make_splat([0.0, 0.0, 0.0]);
        let curr = make_splat([0.1, 0.0, 0.0]);
        let pkt = diff_splat(&prev, &curr, 42).expect("should produce delta");
        assert_eq!(pkt.splat_id, 42);
        assert_ne!(pkt.d_position[0], 0);
        assert_eq!(pkt.d_position[1], 0);
        assert_eq!(pkt.d_position[2], 0);
    }

    #[test]
    fn delta_round_trip() {
        let base = make_splat([1.0, 2.0, 3.0]);
        let mut modified = base;
        modified.position[0] += 0.05;
        modified.position[2] -= 0.02;
        modified.opacity = 180;

        let pkt = diff_splat(&base, &modified, 1).expect("should have delta");
        let reconstructed = apply_delta(&base, &pkt);

        // Check within quantization error (1/POSITION_QUANT ≈ 0.002)
        for i in 0..3 {
            let err = (reconstructed.position[i] - modified.position[i]).abs();
            assert!(err < 1.0 / POSITION_QUANT + 1e-6, "pos[{i}] error {err} too large");
        }
        // Opacity exact (integer arithmetic)
        assert_eq!(reconstructed.opacity, modified.opacity);
    }

    #[test]
    fn spectral_change_bitmask_correct() {
        let mut prev = make_splat([0.0, 0.0, 0.0]);
        let mut curr = prev;
        // Change bands 0 and 3
        curr.spectral[0] = prev.spectral[0] + 10;
        curr.spectral[3] = prev.spectral[3] + 20;

        let pkt = diff_splat(&prev, &curr, 5).expect("should have delta");
        // bits 0 and 3 set → 0b00001001 = 9
        assert_eq!(pkt.spectral_changed, 0b00001001);
        assert_eq!(pkt.d_spectral.len(), 2);

        // Also verify values
        assert_eq!(pkt.d_spectral[0], 10);
        assert_eq!(pkt.d_spectral[1], 20);

        // Suppress unused-mut warnings appeased — needed for the mutation above
        let _ = &mut prev;
    }

    #[test]
    fn estimate_bytes_no_spectral_change() {
        let prev = make_splat([0.0, 0.0, 0.0]);
        let mut curr = prev;
        curr.position[0] += 0.1;

        let pkt = diff_splat(&prev, &curr, 7).expect("delta");
        assert_eq!(pkt.spectral_changed, 0);
        // 4 + 3 + 3 + 1 + 1 + 0 = 12
        assert_eq!(estimate_delta_bytes(&[pkt]), 12);
    }

    #[test]
    fn replication_state_compute_deltas() {
        let splat_a = make_splat([0.0, 0.0, 0.0]);
        let mut splat_b = splat_a;
        splat_b.position[0] = 1.0;

        let mut prev = SplatReplicationState::new();
        prev.update_splat(1, splat_a);

        let mut curr = SplatReplicationState::new();
        curr.update_splat(1, splat_b);

        let deltas = curr.compute_deltas(&prev);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].splat_id, 1);
        assert_ne!(deltas[0].d_position[0], 0);
    }
}
