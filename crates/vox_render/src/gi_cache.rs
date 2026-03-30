//! GI cache — stores baked spectral irradiance, applies it at render time.

use crate::gi_baker::BakedGi;
use vox_core::types::GaussianSplat;
use half::f16;

pub struct GiCache {
    gi: BakedGi,
    pub blend: f32,
}

impl GiCache {
    pub fn new(gi: BakedGi) -> Self {
        Self { gi, blend: 1.0 }
    }

    /// Apply baked GI to a slice of splats, returning new splats with
    /// GI irradiance added into their spectral bands.
    /// Panics if `splats.len() != gi.irradiance.len()`.
    pub fn apply(&self, splats: &[GaussianSplat]) -> Vec<GaussianSplat> {
        assert_eq!(splats.len(), self.gi.irradiance.len(),
            "GiCache was baked for a different number of splats");
        splats.iter().zip(self.gi.irradiance.iter())
            .map(|(s, irr)| {
                let mut out = *s;
                for (band, &irr_val) in irr.iter().enumerate() {
                    let base = f16::from_bits(s.spectral()[band]).to_f32();
                    let gi_contrib = irr_val * self.blend;
                    let result = (base + gi_contrib).clamp(0.0, 1.0);
                    out.spectral_mut()[band] = f16::from_f32(result).to_bits();
                }
                out
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi_baker::BakedGi;

    fn make_splat(spectral_val: f32) -> GaussianSplat {
        let f16_val = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat::volume([0.0, 0.0, 0.0], [0.1, 0.1, 0.1], glam::Quat::IDENTITY, 200, [f16_val; 16])
    }

    #[test]
    fn apply_adds_irradiance_to_spectral() {
        let splat = make_splat(0.2);
        let irradiance = vec![[0.3f32; 16]];
        let gi = BakedGi { irradiance };
        let cache = GiCache::new(gi);
        let result = cache.apply(&[splat]);
        let band0 = half::f16::from_bits(result[0].spectral()[0]).to_f32();
        assert!(band0 > 0.2 + 0.25, "GI should increase spectral value");
        assert!(band0 <= 1.0, "GI must not exceed 1.0");
    }

    #[test]
    fn apply_blend_zero_is_identity() {
        let splat = make_splat(0.5);
        let irradiance = vec![[1.0f32; 16]];
        let gi = BakedGi { irradiance };
        let mut cache = GiCache::new(gi);
        cache.blend = 0.0;
        let result = cache.apply(&[splat]);
        let band0 = half::f16::from_bits(result[0].spectral()[0]).to_f32();
        assert!((band0 - 0.5).abs() < 0.02, "blend=0 should leave spectral unchanged");
    }

    #[test]
    fn apply_clamps_to_one() {
        let splat = make_splat(0.9);
        let irradiance = vec![[0.9f32; 16]];
        let gi = BakedGi { irradiance };
        let cache = GiCache::new(gi);
        let result = cache.apply(&[splat]);
        let band0 = half::f16::from_bits(result[0].spectral()[0]).to_f32();
        assert!(band0 <= 1.001, "GI must clamp to 1.0");
    }
}
