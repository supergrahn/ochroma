//! Offline spectral GI baker.
//!
//! For each splat, computes the incident spectral irradiance from all
//! nearby splats within a search radius. Each nearby splat contributes
//! its spectral reflectance attenuated by distance and facing.
//!
//! Result: `GiBaker::bake()` returns a `Vec<[f32; 8]>` — one irradiance
//! sample per splat — that is ADDED to the splat's base spectral value
//! at render time.

use rayon::prelude::*;
use vox_core::types::GaussianSplat;
use half::f16;

#[derive(Debug, Clone)]
pub struct GiBakeConfig {
    pub search_radius: f32,
    pub max_neighbours: usize,
    pub bounces: usize,
    pub falloff: f32,
}

impl Default for GiBakeConfig {
    fn default() -> Self {
        Self {
            search_radius: 4.0,
            max_neighbours: 32,
            bounces: 1,
            falloff: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BakedGi {
    pub irradiance: Vec<[f32; 16]>,
}

pub struct GiBaker {
    pub config: GiBakeConfig,
}

impl GiBaker {
    pub fn new(config: GiBakeConfig) -> Self {
        Self { config }
    }

    pub fn bake(&self, splats: &[GaussianSplat]) -> BakedGi {
        let mut current: Vec<[f32; 16]> = splats.iter()
            .map(|s| s.spectral_bands_f32())
            .collect();

        for _bounce in 0..self.config.bounces {
            let next: Vec<[f32; 16]> = (0..splats.len())
                .into_par_iter()
                .map(|i| self.accumulate_irradiance(i, splats, &current))
                .collect();
            current = next;
        }

        BakedGi { irradiance: current }
    }

    fn accumulate_irradiance(
        &self,
        target: usize,
        splats: &[GaussianSplat],
        spectral: &[[f32; 16]],
    ) -> [f32; 16] {
        let tp = splats[target].position();
        let r2 = self.config.search_radius * self.config.search_radius;
        let mut accum = [0.0f32; 16];
        let mut count = 0usize;

        for (j, splat) in splats.iter().enumerate() {
            if j == target { continue; }
            let sp = splat.position();
            let dx = sp[0] - tp[0];
            let dy = sp[1] - tp[1];
            let dz = sp[2] - tp[2];
            let dist2 = dx*dx + dy*dy + dz*dz;
            if dist2 > r2 { continue; }

            let dist = dist2.sqrt();
            let atten = 1.0 / (1.0 + dist * self.config.falloff);
            let opacity_w = splat.opacity() as f32 / 255.0;

            for band in 0..16 {
                accum[band] += spectral[j][band] * atten * opacity_w;
            }
            count += 1;
            if count >= self.config.max_neighbours { break; }
        }

        if count > 0 {
            let scale = 1.0 / count as f32;
            for val in accum.iter_mut() { *val *= scale; }
        }
        accum
    }
}

pub trait SplatSpectral {
    fn spectral_bands_f32(&self) -> [f32; 16];
}

impl SplatSpectral for GaussianSplat {
    fn spectral_bands_f32(&self) -> [f32; 16] {
        std::array::from_fn(|i| f16::from_bits(self.spectral()[i]).to_f32().clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn make_splat(pos: [f32; 3], spectral_val: f32) -> GaussianSplat {
        let f16_val = half::f16::from_f32(spectral_val).to_bits();
        GaussianSplat::volume(pos, [0.1, 0.1, 0.1], glam::Quat::IDENTITY, 200, [f16_val; 16])
    }

    #[test]
    fn bake_returns_one_entry_per_splat() {
        let splats = vec![
            make_splat([0.0, 0.0, 0.0], 0.5),
            make_splat([1.0, 0.0, 0.0], 0.8),
        ];
        let baker = GiBaker::new(GiBakeConfig::default());
        let gi = baker.bake(&splats);
        assert_eq!(gi.irradiance.len(), 2);
    }

    #[test]
    fn bake_neighbour_bleeds_into_target() {
        let splats = vec![
            make_splat([0.0, 0.0, 0.0], 0.0),
            make_splat([0.5, 0.0, 0.0], 1.0),
        ];
        let baker = GiBaker::new(GiBakeConfig { search_radius: 2.0, ..Default::default() });
        let gi = baker.bake(&splats);
        assert!(gi.irradiance[0][0] > 0.0, "dark splat should receive GI from bright neighbour");
    }

    #[test]
    fn bake_far_neighbour_does_not_bleed() {
        let splats = vec![
            make_splat([0.0, 0.0, 0.0], 0.0),
            make_splat([100.0, 0.0, 0.0], 1.0),
        ];
        let baker = GiBaker::new(GiBakeConfig { search_radius: 1.0, ..Default::default() });
        let gi = baker.bake(&splats);
        assert_eq!(gi.irradiance[0], [0.0; 16], "far splat should not bleed");
    }

    #[test]
    fn bake_is_deterministic() {
        let splats: Vec<GaussianSplat> = (0..20)
            .map(|i| make_splat([i as f32 * 0.3, 0.0, 0.0], 0.4 + i as f32 * 0.03))
            .collect();
        let baker = GiBaker::new(GiBakeConfig::default());
        let gi1 = baker.bake(&splats);
        let gi2 = baker.bake(&splats);
        assert_eq!(gi1.irradiance, gi2.irradiance);
    }

    #[test]
    fn spectral_bands_f32_round_trips() {
        let splat = make_splat([0.0, 0.0, 0.0], 0.75);
        let bands = splat.spectral_bands_f32();
        for &b in &bands {
            assert!((b - 0.75).abs() < 0.01, "f16 round-trip should be within 1%");
        }
    }
}
