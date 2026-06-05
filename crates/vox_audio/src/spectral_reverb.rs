//! SpectralReverb — derives room impulse response from surrounding Gaussian splat reflectance.

/// Reverb profile derived from splat reflectance.
#[derive(Debug, Clone)]
pub struct SpectralReverb {
    pub tail_length_secs: f32,
    pub band_rt60: [f32; 16],
    pub mean_reflectance: [f32; 16],
}

/// Compact, value-typed reverb parameters for a room.
///
/// This is the clean reachable entry point for the rest of the engine: given the
/// 16-band spectral reflectance of the surfaces *surrounding* a listener (walls,
/// floor, ceiling — i.e. the inside surface of a room), produce a reverb tail
/// length (RT60), a wet/dry mix, and a renderable impulse response.
///
/// Highly-reflective uniform surfaces (stone, tile) yield a long tail; absorptive
/// surfaces (fabric, foliage) yield a short, dead tail.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbParams {
    /// Broadband RT60 (time for the tail to decay 60 dB) in seconds.
    pub rt60_secs: f32,
    /// Suggested wet/dry mix in `[0, 1]` — more reflective rooms sound wetter.
    pub wet_mix: f32,
    /// Mean broadband reflectance of the surrounding surfaces in `[0, 1]`.
    pub mean_reflectance: f32,
}

impl ReverbParams {
    /// Number of samples a rendered impulse response will occupy at `sample_rate`.
    pub fn tail_samples_len(&self, sample_rate: u32) -> usize {
        (self.rt60_secs * sample_rate as f32).round() as usize
    }
}

impl SpectralReverb {
    pub fn from_splat_reflectance(splats: &[[u16; 16]]) -> Self {
        if splats.is_empty() {
            return Self::default_dead_room();
        }

        let mut mean = [0.0f32; 16];
        for s in splats {
            for band in 0..16usize {
                mean[band] += half::f16::from_bits(s[band]).to_f32().max(0.0);
            }
        }
        for m in &mut mean { *m /= splats.len() as f32; }

        let overall_mean: f32 = mean.iter().sum::<f32>() / 16.0;
        let tail_length_secs = 0.05 + overall_mean.powi(2) * 7.95;

        let hf_penalty = [0.70f32, 0.72, 0.75, 0.78, 0.82, 0.86, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00];
        let band_rt60 = std::array::from_fn(|b| {
            let r   = mean[b].clamp(0.0, 1.0);
            let rt  = 0.05 + r.powi(2) * 7.95;
            rt * hf_penalty[b]
        });

        Self { tail_length_secs, band_rt60, mean_reflectance: mean }
    }

    /// Build a reverb from a global-illumination band vector (already linear `f32`
    /// reflectance per band, as produced by the GI / spectral lighting pass).
    ///
    /// Mirrors [`Self::from_splat_reflectance`] but skips the `f16` decode so the GI
    /// pipeline can drive reverb directly without round-tripping through splats.
    pub fn from_gi_bands(bands: &[f32; 16]) -> Self {
        let mean: [f32; 16] = std::array::from_fn(|b| bands[b].max(0.0));

        let overall_mean: f32 = mean.iter().sum::<f32>() / 16.0;
        let tail_length_secs = 0.05 + overall_mean.powi(2) * 7.95;

        let hf_penalty = [0.70f32, 0.72, 0.75, 0.78, 0.82, 0.86, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00];
        let band_rt60 = std::array::from_fn(|b| {
            let r  = mean[b].clamp(0.0, 1.0);
            let rt = 0.05 + r.powi(2) * 7.95;
            rt * hf_penalty[b]
        });

        Self { tail_length_secs, band_rt60, mean_reflectance: mean }
    }

    /// Collapse this reverb into compact value-typed [`ReverbParams`].
    pub fn to_params(&self) -> ReverbParams {
        let mean: f32 = (self.mean_reflectance.iter().sum::<f32>() / 16.0).clamp(0.0, 1.0);
        // Wetter rooms sound more reflective; cap so a dry room stays mostly dry.
        let wet_mix = (mean * 0.6).clamp(0.0, 0.6);
        ReverbParams {
            rt60_secs: self.tail_length_secs,
            wet_mix,
            mean_reflectance: mean,
        }
    }

    pub fn tail_samples(&self, sample_rate: u32) -> Vec<f32> {
        let n = (self.tail_length_secs * sample_rate as f32).round() as usize;
        let decay_rate = -6.9 / self.tail_length_secs;

        let mut state = 0x12345678u32;
        let lcg_next = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (*s as i32 as f32) / i32::MAX as f32
        };

        (0..n).map(|i| {
            let t        = i as f32 / sample_rate as f32;
            let envelope = (decay_rate * t).exp();
            envelope * lcg_next(&mut state)
        }).collect()
    }

    fn default_dead_room() -> Self {
        Self {
            tail_length_secs: 0.05,
            band_rt60:        [0.05; 16],
            mean_reflectance: [0.0; 16],
        }
    }
}

// ---------------------------------------------------------------------------
// Reachable room-reverb entry points
// ---------------------------------------------------------------------------

/// Clean reachable entry point: compute reverb parameters for a room from the
/// 16-band spectral reflectance of its *surrounding surfaces* (walls/floor/ceiling).
///
/// A room lined with high, uniform reflectance (stone/tile) yields a LONGER tail
/// (larger `rt60_secs`) than a room lined with absorptive material (fabric/foliage).
pub fn reverb_for_room(surface_spectra: &[[u16; 16]]) -> ReverbParams {
    SpectralReverb::from_splat_reflectance(surface_spectra).to_params()
}

/// Render the room's reverb impulse response (decaying noise tail) at `sample_rate`.
///
/// Length scales with `rt60_secs`, so a stone room produces a longer IR than a
/// fabric room. Returns an empty buffer for a fully dead room with a zero tail.
pub fn room_impulse(surface_spectra: &[[u16; 16]], sample_rate: u32) -> Vec<f32> {
    SpectralReverb::from_splat_reflectance(surface_spectra).tail_samples(sample_rate)
}

/// As [`reverb_for_room`], but driven directly from a GI band vector (linear `f32`
/// reflectance per band) instead of `f16`-encoded splats.
pub fn reverb_for_room_from_gi(gi_bands: &[f32; 16]) -> ReverbParams {
    SpectralReverb::from_gi_bands(gi_bands).to_params()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_high_reflectance() -> Vec<[u16; 16]> {
        let v = half::f16::from_f32(0.9).to_bits();
        vec![[v; 16]; 16]
    }

    fn make_low_reflectance() -> Vec<[u16; 16]> {
        let v = half::f16::from_f32(0.05).to_bits();
        vec![[v; 16]; 16]
    }

    #[test]
    fn high_reflectance_gives_longer_tail_than_low() {
        let high = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
        let low  = SpectralReverb::from_splat_reflectance(&make_low_reflectance());
        println!("high={} low={}", high.tail_length_secs, low.tail_length_secs);
        assert!(high.tail_length_secs > low.tail_length_secs,
            "high={} low={}", high.tail_length_secs, low.tail_length_secs);
    }

    #[test]
    fn tail_length_within_physical_bounds() {
        let reverb = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
        assert!(reverb.tail_length_secs >= 0.05);
        assert!(reverb.tail_length_secs <= 10.0);
    }

    #[test]
    fn empty_splat_list_yields_default_reverb() {
        let reverb = SpectralReverb::from_splat_reflectance(&[]);
        assert!(reverb.tail_length_secs > 0.0);
    }

    #[test]
    fn tail_samples_length_matches_tail_length() {
        let reverb = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
        let ir     = reverb.tail_samples(44_100);
        let expected = (reverb.tail_length_secs * 44_100.0) as usize;
        assert!((ir.len() as isize - expected as isize).abs() <= 1);
    }

    #[test]
    fn tail_samples_decays_to_near_zero() {
        let reverb = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
        let ir     = reverb.tail_samples(44_100);
        let last   = ir.last().copied().unwrap_or(0.0).abs();
        assert!(last < 0.01, "IR should decay to near-zero, last={last}");
    }

    #[test]
    fn per_band_rt60_high_reflectance_vs_low() {
        let high = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
        let low  = SpectralReverb::from_splat_reflectance(&make_low_reflectance());
        for band in 0..16usize {
            assert!(high.band_rt60[band] > low.band_rt60[band],
                "band {band}: high_rt60={} low_rt60={}", high.band_rt60[band], low.band_rt60[band]);
        }
    }

    /// A stone-like room (high uniform reflectance ~0.9) must yield a LONGER reverb
    /// tail — both RT60 seconds and rendered sample count — than a fabric-like
    /// (mid-absorption ~0.3) room. Asserts real ordered numeric values.
    #[test]
    fn reverb_for_room_stone_longer_tail_than_fabric() {
        // Stone: high, uniform reflectance across all 16 bands.
        let stone_v  = half::f16::from_f32(0.90).to_bits();
        // Fabric: mid-absorption (low reflectance) across all bands.
        let fabric_v = half::f16::from_f32(0.30).to_bits();

        let stone_room:  Vec<[u16; 16]> = vec![[stone_v;  16]; 16];
        let fabric_room: Vec<[u16; 16]> = vec![[fabric_v; 16]; 16];

        let stone  = reverb_for_room(&stone_room);
        let fabric = reverb_for_room(&fabric_room);

        let sr = 44_100u32;
        let stone_n  = stone.tail_samples_len(sr);
        let fabric_n = fabric.tail_samples_len(sr);

        // Cross-check rendered IR length matches the reported param length.
        let stone_ir  = room_impulse(&stone_room,  sr);
        let fabric_ir = room_impulse(&fabric_room, sr);

        println!(
            "stone rt60={:.3}s ({stone_n} samples, ir.len={}) | fabric rt60={:.3}s ({fabric_n} samples, ir.len={})",
            stone.rt60_secs, stone_ir.len(), fabric.rt60_secs, fabric_ir.len()
        );

        // Ordered numeric assertions on real computed values.
        assert!(stone.rt60_secs > fabric.rt60_secs,
            "stone rt60={:.3}s must exceed fabric rt60={:.3}s", stone.rt60_secs, fabric.rt60_secs);
        assert!(stone_n > fabric_n,
            "stone tail samples {stone_n} must exceed fabric tail samples {fabric_n}");
        assert!(stone_ir.len() > fabric_ir.len(),
            "stone IR {} must be longer than fabric IR {}", stone_ir.len(), fabric_ir.len());
        // Concrete magnitude sanity: a 0.9-reflectance stone room has a multi-second tail.
        assert!(stone.rt60_secs > 1.0,
            "stone room should ring for >1s, got {:.3}s", stone.rt60_secs);
        assert!(fabric.rt60_secs < 1.0,
            "fabric room should be relatively dead (<1s), got {:.3}s", fabric.rt60_secs);
    }

    #[test]
    fn to_params_wet_mix_higher_for_reflective_room() {
        let stone  = reverb_for_room(&make_high_reflectance());
        let fabric = reverb_for_room(&make_low_reflectance());
        println!("stone wet={:.3} fabric wet={:.3}", stone.wet_mix, fabric.wet_mix);
        assert!(stone.wet_mix > fabric.wet_mix,
            "reflective room should be wetter: stone={:.3} fabric={:.3}", stone.wet_mix, fabric.wet_mix);
        assert!(stone.wet_mix > 0.0 && stone.wet_mix <= 0.6, "wet_mix out of range: {}", stone.wet_mix);
    }

    #[test]
    fn gi_band_reverb_matches_equivalent_splat_reverb() {
        // 0.5 reflectance everywhere, expressed as GI bands vs f16 splats.
        let gi = [0.5f32; 16];
        let splat_v = half::f16::from_f32(0.5).to_bits();
        let splats: Vec<[u16; 16]> = vec![[splat_v; 16]; 4];

        let from_gi    = reverb_for_room_from_gi(&gi);
        let from_splat = reverb_for_room(&splats);

        println!("gi rt60={:.4} splat rt60={:.4}", from_gi.rt60_secs, from_splat.rt60_secs);
        // f16(0.5) is exactly representable, so these must match closely.
        assert!((from_gi.rt60_secs - from_splat.rt60_secs).abs() < 1e-3,
            "gi={:.4} splat={:.4}", from_gi.rt60_secs, from_splat.rt60_secs);
    }
}
