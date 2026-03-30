//! Thermal dynamics for spectral splats.
//! Hot objects emit in bands 9–14 (580–730 nm red/near-IR).
//! Heat diffuses via inverse-square, tracked per frame.

use glam::Vec3;
use half::f16;
use vox_core::types::GaussianSplat;

#[derive(Debug, Clone)]
pub struct HeatSource {
    pub position: Vec3,
    pub power: f32,
    pub cooling_rate: f32,
    pub age_seconds: f32,
}

pub struct ThermalEmitter {
    pub heat: Vec<f32>,
    pub emit_threshold: f32,
    pub diffusion_radius: f32,
    pub cooling_per_frame: f32,
}

impl ThermalEmitter {
    pub fn new(splat_count: usize) -> Self {
        Self {
            heat: vec![0.0f32; splat_count],
            emit_threshold: 0.2,
            diffusion_radius: 0.5,
            cooling_per_frame: 0.005,
        }
    }

    pub fn resize(&mut self, count: usize) {
        self.heat.resize(count, 0.0);
    }

    /// Update heat values and spectral bands for each splat given heat sources.
    /// Each heat source is `(position, power)`.
    pub fn update(&mut self, splats: &mut Vec<GaussianSplat>, heat_sources: &[(Vec3, f32)]) {
        self.resize(splats.len());
        let r2_limit = self.diffusion_radius * self.diffusion_radius;

        // Apply heat from sources
        for (i, splat) in splats.iter().enumerate() {
            let pos = Vec3::from_array(splat.position());
            for &(src_pos, power) in heat_sources {
                let dist_sq = (pos - src_pos).length_squared();
                if dist_sq < r2_limit {
                    let attenuation = 1.0 - (dist_sq / r2_limit).sqrt();
                    self.heat[i] =
                        (self.heat[i] + power * attenuation * 0.1).clamp(0.0, 1.0);
                }
            }
        }

        // Elevate spectral bands 9-14 for hot splats, then cool
        for (i, splat) in splats.iter_mut().enumerate() {
            let h = self.heat[i];
            if h > self.emit_threshold {
                let excess = h - self.emit_threshold;
                for b in 9..15usize {
                    let current = f16::from_bits(splat.spectral()[b]).to_f32();
                    let elevated = (current + excess * 0.5).clamp(0.0, 1.0);
                    splat.spectral_mut()[b] = f16::from_f32(elevated).to_bits();
                }
            }
            self.heat[i] = (h - self.cooling_per_frame).max(0.0);
        }
    }

    /// Iterate over splats whose heat exceeds the emit threshold, yielding
    /// `(world_position, spectral_radiance_16bands)` for GI seeding.
    pub fn hot_emitters<'a>(
        &'a self,
        splats: &'a [GaussianSplat],
    ) -> impl Iterator<Item = (Vec3, [f32; 16])> + 'a {
        splats
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < self.heat.len() && self.heat[*i] > self.emit_threshold)
            .map(|(i, splat)| {
                let pos = Vec3::from_array(splat.position());
                let mut spectral = [0.0f32; 16];
                for b in 0..16 {
                    spectral[b] =
                        f16::from_bits(splat.spectral()[b]).to_f32() * self.heat[i];
                }
                (pos, spectral)
            })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_splat(pos: [f32; 3]) -> GaussianSplat {
        let spectral = [half::f16::from_f32(0.0).to_bits(); 16];
        GaussianSplat::surface(
            pos,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            0.1,
            0.1,
            200,
            spectral,
        )
    }

    #[test]
    fn heat_source_elevates_bands_9_to_14() {
        let mut emitter = ThermalEmitter::new(0);
        let mut splats = vec![make_splat([0.0, 0.0, 0.0])];
        let source = (Vec3::ZERO, 1.0);
        for _ in 0..20 {
            emitter.update(&mut splats, &[source]);
        }
        let b9 = half::f16::from_bits(splats[0].spectral()[9]).to_f32();
        let b11 = half::f16::from_bits(splats[0].spectral()[11]).to_f32();
        let b14 = half::f16::from_bits(splats[0].spectral()[14]).to_f32();
        assert!(
            b9 > 0.0 || b11 > 0.0 || b14 > 0.0,
            "bands 9-14 must be elevated after heat application (b9={b9}, b11={b11}, b14={b14})"
        );
    }

    #[test]
    fn distant_splat_receives_less_heat() {
        let mut emitter = ThermalEmitter::new(0);
        let mut splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([10.0, 0.0, 0.0])];
        let source = (Vec3::ZERO, 1.0);
        for _ in 0..10 {
            emitter.update(&mut splats, &[source]);
        }
        assert!(
            emitter.heat[0] >= emitter.heat[1],
            "close splat must have >= heat than distant splat"
        );
    }

    #[test]
    fn heat_cools_without_source() {
        let mut emitter = ThermalEmitter::new(1);
        emitter.heat[0] = 0.8;
        let mut splats = vec![make_splat([0.0, 0.0, 0.0])];
        for _ in 0..50 {
            emitter.update(&mut splats, &[]);
        }
        assert!(
            emitter.heat[0] < 0.8,
            "heat should cool without source, got {}",
            emitter.heat[0]
        );
    }

    #[test]
    fn cold_splats_bands_unchanged() {
        let mut emitter = ThermalEmitter::new(0);
        let b11_init = half::f16::from_f32(0.5).to_bits();
        let mut spectral = [half::f16::from_f32(0.0).to_bits(); 16];
        spectral[11] = b11_init;
        let mut splats = vec![GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            0.1,
            0.1,
            200,
            spectral,
        )];
        let source = (Vec3::new(100.0, 0.0, 0.0), 0.001);
        emitter.update(&mut splats, &[source]);
        let b11_after = half::f16::from_bits(splats[0].spectral()[11]).to_f32();
        let b11_before = half::f16::from_bits(b11_init).to_f32();
        assert!(
            (b11_after - b11_before).abs() < 0.01,
            "distant source must not change band 11: before={} after={}",
            b11_before,
            b11_after
        );
    }

    #[test]
    fn hot_emitters_yields_above_threshold() {
        let mut emitter = ThermalEmitter::new(2);
        emitter.heat[0] = 0.9;
        emitter.heat[1] = 0.05;
        let splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([1.0, 0.0, 0.0])];
        let emitters: Vec<_> = emitter.hot_emitters(&splats).collect();
        assert_eq!(emitters.len(), 1, "only the hot splat should be yielded");
    }
}
