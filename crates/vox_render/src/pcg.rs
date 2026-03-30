//! Procedural Content Generation — spectral-aware placement rules.
//! FilterBySpectralBand: place content only where a spectral band exceeds a threshold.

use glam::Vec3;
use vox_core::types::GaussianSplat;
use half::f16;

/// A 3D point in the PCG candidate set.
#[derive(Debug, Clone)]
pub struct PcgPoint {
    pub position: Vec3,
    pub spectral: [f32; 8],
    pub weight: f32,
}

impl PcgPoint {
    pub fn from_splat(splat: &GaussianSplat) -> Self {
        let spectral = std::array::from_fn(|b| f16::from_bits(splat.spectral()[b]).to_f32());
        Self { position: Vec3::from(splat.position()), spectral, weight: splat.opacity() as f32 / 255.0 }
    }
}

/// PCG filter: keep only points where spectral band `band` exceeds `threshold`.
/// This is the unique Ochroma PCG node — no Unreal equivalent.
pub struct FilterBySpectralBand {
    pub band: usize,       // 0–7
    pub threshold: f32,    // [0.0, 1.0]
    pub invert: bool,      // if true, keep points BELOW threshold
}

impl FilterBySpectralBand {
    pub fn new(band: usize, threshold: f32) -> Self {
        Self { band: band.min(7), threshold, invert: false }
    }

    pub fn inverted(mut self) -> Self { self.invert = true; self }

    pub fn filter(&self, points: &[PcgPoint]) -> Vec<PcgPoint> {
        points.iter().filter(|p| {
            let val = p.spectral[self.band];
            let passes = val >= self.threshold;
            if self.invert { !passes } else { passes }
        }).cloned().collect()
    }
}

/// PCG scatter: distribute instances within a region based on spectral density.
/// High spectral band value = denser scatter. Unique to Ochroma.
pub struct ScatterBySpectralDensity {
    pub band: usize,
    pub base_density: f32,   // instances per square metre at band value 0
    pub max_density: f32,    // instances per square metre at band value 1.0
    pub min_separation: f32, // minimum distance between placed instances
}

impl ScatterBySpectralDensity {
    pub fn new(band: usize, base_density: f32, max_density: f32, min_separation: f32) -> Self {
        Self { band: band.min(7), base_density, max_density, min_separation }
    }

    /// Scatter instances within the given points, returning placement positions.
    pub fn scatter(&self, candidates: &[PcgPoint]) -> Vec<Vec3> {
        let mut placed: Vec<Vec3> = Vec::new();

        // Simple Poisson-disk-like scatter: iterate candidates in a deterministic order,
        // place if spectral value triggers placement and no nearby placed point exists.
        for (i, point) in candidates.iter().enumerate() {
            let band_val = point.spectral[self.band];
            let density = self.base_density + (self.max_density - self.base_density) * band_val;

            // Deterministic Bernoulli trial based on point index and density
            let hash = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
            let rand_val = (hash as f32) / (u32::MAX as f32);

            // Probability of placement per unit area at this density (approximate)
            if rand_val > density * self.min_separation * self.min_separation {
                continue;
            }

            // Check separation from already placed points
            let too_close = placed.iter().any(|p| (*p - point.position).length() < self.min_separation);
            if !too_close {
                placed.push(point.position);
            }
        }

        placed
    }
}

/// PCG spectral weather: affects the spectral emission of splats based on weather state.
#[derive(Debug, Clone)]
pub struct SpectralWeather {
    pub rain_intensity: f32,    // [0, 1] — modulates UV/blue bands (rain absorbs high-freq)
    pub fog_density: f32,       // [0, 1] — reduces all bands uniformly
    pub snow_coverage: f32,     // [0, 1] — shifts all bands toward arctic white
}

impl SpectralWeather {
    pub fn clear() -> Self { Self { rain_intensity: 0.0, fog_density: 0.0, snow_coverage: 0.0 } }

    /// Apply weather modulation to a spectral value array.
    pub fn modulate(&self, spectral: &mut [f32; 8]) {
        // Rain: reduce high-frequency bands (water absorbs UV/blue)
        for val in spectral[..3].iter_mut() {
            *val *= 1.0 - self.rain_intensity * 0.5;
        }
        // Fog: uniform attenuation
        for val in spectral.iter_mut() {
            *val *= 1.0 - self.fog_density * 0.3;
        }
        // Snow: push toward flat white
        for val in spectral.iter_mut() {
            *val = *val * (1.0 - self.snow_coverage) + 0.85 * self.snow_coverage;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(band0: f32) -> PcgPoint {
        let mut spectral = [0.0f32; 8];
        spectral[0] = band0;
        PcgPoint { position: Vec3::ZERO, spectral, weight: 1.0 }
    }

    #[test]
    fn filter_by_spectral_band_passes_high() {
        let filter = FilterBySpectralBand::new(0, 0.4);
        let points = vec![make_point(0.6), make_point(0.8), make_point(0.5)];
        let result = filter.filter(&points);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn filter_by_spectral_band_blocks_low() {
        let filter = FilterBySpectralBand::new(0, 0.4);
        let points = vec![make_point(0.1), make_point(0.1)];
        let result = filter.filter(&points);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn filter_inverted() {
        let filter = FilterBySpectralBand::new(0, 0.4).inverted();
        let points = vec![make_point(0.1), make_point(0.2), make_point(0.8)];
        let result = filter.filter(&points);
        // Only points below 0.4 should pass
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|p| p.spectral[0] < 0.4));
    }

    #[test]
    fn spectral_weather_snow_brightens() {
        let weather = SpectralWeather { rain_intensity: 0.0, fog_density: 0.0, snow_coverage: 1.0 };
        let mut spectral = [0.1f32; 8];
        weather.modulate(&mut spectral);
        for b in 0..8 {
            assert!((spectral[b] - 0.85).abs() < 0.01, "band {} = {}", b, spectral[b]);
        }
    }
}
