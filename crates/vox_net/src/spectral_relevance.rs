//! Spectral relevance filtering for network replication.
//!
//! Replaces geometry-based "interest volume" culling with a physics-based check:
//! a splat is relevant to a client if its spectral energy in any band exceeds the
//! client's perceptual threshold for that band.

use half::f16;

/// Observer spectral sensitivity profile.
#[derive(Debug, Clone)]
pub struct ObserverProfile {
    pub weights: [f32; 16],
}

impl ObserverProfile {
    /// Standard human photopic sensitivity (CIE V(λ) approximated at 16 band centres).
    pub fn human() -> Self {
        Self {
            weights: [0.004, 0.010, 0.030, 0.100, 0.230, 0.450, 0.710, 0.954, 0.995, 0.870, 0.757, 0.550, 0.265, 0.120, 0.061, 0.020],
        }
    }

    /// Observer tuned for fire detection (high bands 10–15: red/near-IR).
    /// Bands 0–7 are kept at 0.0 so blue/UV splats are never falsely triggered.
    /// Bands 8–9 provide low-weight near-IR sensitivity (580–605nm, hot embers).
    pub fn fire_observer() -> Self {
        Self { weights: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.40, 0.8, 0.9, 1.0, 0.95, 0.9, 0.85] }
    }

    /// Observer tuned for underwater visibility (high bands 4–8: blue/cyan/green).
    /// Bands 10–15 are at ≤0.05 so red/NIR splats do not trigger this observer.
    pub fn underwater() -> Self {
        Self { weights: [0.05, 0.1, 0.3, 0.5, 1.0, 0.9, 0.85, 0.7, 0.6, 0.4, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0] }
    }

    /// Custom profile from raw weights. Values are clamped to [0, 1].
    pub fn custom(weights: [f32; 16]) -> Self {
        Self { weights: std::array::from_fn(|i| weights[i].clamp(0.0, 1.0)) }
    }
}

/// A single Gaussian splat's spectral data for relevance testing.
#[derive(Debug, Clone, Copy)]
pub struct SplatSpectral {
    pub bands: [u16; 16],
}

impl SplatSpectral {
    pub fn decode(&self, b: usize) -> f32 {
        f16::from_bits(self.bands[b]).to_f32()
    }

    pub fn decode_all(&self) -> [f32; 16] {
        std::array::from_fn(|i| self.decode(i))
    }
}

/// Spectral relevance filter — determines if a splat should be replicated.
pub struct SpectralRelevanceFilter {
    pub threshold: f32,
}

impl SpectralRelevanceFilter {
    pub fn new(threshold: f32) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0) }
    }

    pub fn default_filter() -> Self {
        Self::new(0.05)
    }

    pub fn is_relevant(&self, splat: &SplatSpectral, observer_profile: &ObserverProfile) -> bool {
        for b in 0..16 {
            let energy = splat.decode(b);
            let weighted = energy * observer_profile.weights[b];
            if weighted > self.threshold {
                return true;
            }
        }
        false
    }

    pub fn filter_indices(
        &self,
        splats: &[SplatSpectral],
        observer_profile: &ObserverProfile,
    ) -> Vec<usize> {
        splats.iter().enumerate()
            .filter(|(_, s)| self.is_relevant(s, observer_profile))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn cull_fraction(
        &self,
        splats: &[SplatSpectral],
        observer_profile: &ObserverProfile,
    ) -> f32 {
        if splats.is_empty() { return 0.0; }
        let relevant = self.filter_indices(splats, observer_profile).len();
        1.0 - (relevant as f32 / splats.len() as f32)
    }
}

/// Construct a SplatSpectral from f32 band values.
pub fn splat_from_f32(bands: [f32; 16]) -> SplatSpectral {
    SplatSpectral {
        bands: std::array::from_fn(|i| f16::from_f32(bands[i].clamp(0.0, 1.0)).to_bits()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bright_splat_is_relevant_to_human() {
        let filter = SpectralRelevanceFilter::default_filter();
        let profile = ObserverProfile::human();
        let splat = splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(filter.is_relevant(&splat, &profile),
            "bright green splat should be relevant to human observer");
    }

    #[test]
    fn dark_splat_is_not_relevant() {
        let filter = SpectralRelevanceFilter::new(0.1);
        let profile = ObserverProfile::human();
        let splat = splat_from_f32([0.01; 16]);
        assert!(!filter.is_relevant(&splat, &profile),
            "near-black splat should not be relevant (below threshold)");
    }

    #[test]
    fn red_splat_is_relevant_to_fire_observer_but_not_underwater() {
        let filter = SpectralRelevanceFilter::default_filter();
        let fire_profile = ObserverProfile::fire_observer();
        let water_profile = ObserverProfile::underwater();
        let splat = splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
        assert!(filter.is_relevant(&splat, &fire_profile), "red splat should be relevant to fire observer");
        assert!(!filter.is_relevant(&splat, &water_profile), "red splat should NOT be relevant to underwater observer");
    }

    #[test]
    fn blue_splat_is_relevant_to_underwater_not_fire() {
        let filter = SpectralRelevanceFilter::default_filter();
        let fire_profile = ObserverProfile::fire_observer();
        let water_profile = ObserverProfile::underwater();
        let splat = splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(filter.is_relevant(&splat, &water_profile), "blue splat should be relevant to underwater observer");
        assert!(!filter.is_relevant(&splat, &fire_profile), "blue splat should NOT be relevant to fire observer");
    }

    #[test]
    fn filter_indices_returns_only_relevant_subset() {
        let filter = SpectralRelevanceFilter::default_filter();
        let profile = ObserverProfile::human();
        let splats = vec![
            splat_from_f32([0.9; 16]),
            splat_from_f32([0.01; 16]),
            splat_from_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            splat_from_f32([0.0; 16]),
        ];
        let indices = filter.filter_indices(&splats, &profile);
        assert_eq!(indices, vec![0, 2], "expected indices 0 and 2 to be relevant: {:?}", indices);
    }

    #[test]
    fn cull_fraction_all_dark_is_one() {
        let filter = SpectralRelevanceFilter::new(0.05);
        let profile = ObserverProfile::human();
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.0; 16])).collect();
        let fraction = filter.cull_fraction(&splats, &profile);
        assert!((fraction - 1.0).abs() < 1e-5, "all-dark splats should give cull_fraction=1.0, got {}", fraction);
    }

    #[test]
    fn cull_fraction_all_bright_is_zero() {
        let filter = SpectralRelevanceFilter::new(0.05);
        let profile = ObserverProfile::human();
        let splats: Vec<_> = (0..10).map(|_| splat_from_f32([0.9; 16])).collect();
        let fraction = filter.cull_fraction(&splats, &profile);
        assert!((fraction - 0.0).abs() < 1e-5, "all-bright splats should give cull_fraction=0.0, got {}", fraction);
    }

    #[test]
    fn observer_profile_custom_clamped_to_unit() {
        let profile = ObserverProfile::custom([2.0, -0.5, 1.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        for (i, &w) in profile.weights.iter().enumerate() {
            assert!((0.0..=1.0).contains(&w), "weight[{}] = {} should be clamped to [0,1]", i, w);
        }
    }
}
