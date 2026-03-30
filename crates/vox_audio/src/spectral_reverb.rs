//! SpectralReverb — derives room impulse response from surrounding Gaussian splat reflectance.

/// Reverb profile derived from splat reflectance.
#[derive(Debug, Clone)]
pub struct SpectralReverb {
    pub tail_length_secs: f32,
    pub band_rt60: [f32; 16],
    pub mean_reflectance: [f32; 16],
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
}
