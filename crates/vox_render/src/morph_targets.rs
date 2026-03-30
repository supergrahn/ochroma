//! Spectral morph targets — blend shapes with spectral channel deformation.
//! SplatDelta includes d_spectral[8] for per-band emission changes.
//! MorphComputePass: GPU compute that applies up to 16 active targets.

use vox_core::types::GaussianSplat;
use half::f16;
use std::sync::Arc;

/// Per-splat delta for a morph target. Sparse — only changed splats stored.
#[derive(Debug, Clone)]
pub struct SplatDelta {
    pub splat_index: u32,
    pub d_position: [f32; 3],
    pub d_scale: [f32; 3],
    /// Spectral delta: f16 bits, signed delta stored as offset from 0 (use f16 directly).
    /// Positive = emission increase, negative = absorption increase.
    pub d_spectral: [u16; 16],
}

impl SplatDelta {
    /// Create a delta for position change only.
    pub fn position(splat_index: u32, d_pos: [f32; 3]) -> Self {
        Self {
            splat_index,
            d_position: d_pos,
            d_scale: [0.0; 3],
            d_spectral: [0; 16],
        }
    }

    /// Create a delta with a spectral shift (signed f32 per band, packed to f16 bits).
    pub fn with_spectral(mut self, d_spectral_f32: [f32; 16]) -> Self {
        for (b, &v) in d_spectral_f32.iter().enumerate() {
            self.d_spectral[b] = f16::from_f32(v).to_bits();
        }
        self
    }

    /// Unpack d_spectral to f32 (for CPU application).
    pub fn d_spectral_f32(&self) -> [f32; 16] {
        std::array::from_fn(|b| f16::from_bits(self.d_spectral[b]).to_f32())
    }
}

pub struct MorphTarget {
    pub name: String,
    pub deltas: Vec<SplatDelta>,
}

impl MorphTarget {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), deltas: Vec::new() }
    }

    pub fn add_delta(&mut self, delta: SplatDelta) {
        self.deltas.push(delta);
    }
}

pub struct MorphTargetSet {
    pub targets: Vec<MorphTarget>,
    pub base_splats: Arc<Vec<GaussianSplat>>,
}

impl MorphTargetSet {
    pub fn new(base_splats: Vec<GaussianSplat>) -> Self {
        Self {
            targets: Vec::new(),
            base_splats: Arc::new(base_splats),
        }
    }

    pub fn add_target(&mut self, target: MorphTarget) -> usize {
        let idx = self.targets.len();
        self.targets.push(target);
        idx
    }

    /// CPU evaluation: apply morph target weights to base_splats.
    /// `weights`: parallel to `targets`, weight in [0.0, 1.0].
    pub fn evaluate(&self, weights: &[f32]) -> Vec<GaussianSplat> {
        let mut result = (*self.base_splats).clone();

        for (target_idx, target) in self.targets.iter().enumerate() {
            let w = if target_idx < weights.len() { weights[target_idx] } else { 0.0 };
            if w < 1e-5 { continue; }

            for delta in &target.deltas {
                let idx = delta.splat_index as usize;
                if idx >= result.len() { continue; }

                let splat = &mut result[idx];
                {
                    let p = splat.position_mut();
                    p[0] += delta.d_position[0] * w;
                    p[1] += delta.d_position[1] * w;
                    p[2] += delta.d_position[2] * w;
                }
                splat.set_scales(
                    (splat.scale_u() + delta.d_scale[0] * w).max(0.0001),
                    (splat.scale_v() + delta.d_scale[1] * w).max(0.0001),
                    (splat.scale_w() + delta.d_scale[2] * w).max(0.0001),
                );

                // Apply spectral delta
                let d_spec = delta.d_spectral_f32();
                for (b, &ds) in d_spec.iter().enumerate() {
                    let current = f16::from_bits(splat.spectral()[b]).to_f32();
                    let new_val = (current + ds * w).clamp(0.0, 1.0);
                    splat.spectral_mut()[b] = f16::from_f32(new_val).to_bits();
                }
            }
        }

        result
    }

    /// Diff two splat arrays to produce a MorphTarget.
    /// `name`: morph target name.
    /// `base` and `deformed` must be the same length.
    /// Only splats with meaningful changes are included (pos change > 1e-4 or spectral change > 0.001).
    pub fn compute_diff(name: impl Into<String>, base: &[GaussianSplat], deformed: &[GaussianSplat]) -> MorphTarget {
        assert_eq!(base.len(), deformed.len(), "base and deformed splat arrays must have equal length");
        let mut target = MorphTarget::new(name);

        for (i, (b, d)) in base.iter().zip(deformed.iter()).enumerate() {
            let dp = [
                d.position()[0] - b.position()[0],
                d.position()[1] - b.position()[1],
                d.position()[2] - b.position()[2],
            ];
            let ds = [
                d.scale_u() - b.scale_u(),
                d.scale_v() - b.scale_v(),
                d.scale_w() - b.scale_w(),
            ];

            let pos_changed = dp.iter().any(|v| v.abs() > 1e-4);
            let scale_changed = ds.iter().any(|v| v.abs() > 1e-4);

            let mut d_spectral = [0u16; 16];
            let mut spec_changed = false;
            for (band, ds_val) in d_spectral.iter_mut().enumerate() {
                let bv = f16::from_bits(b.spectral()[band]).to_f32();
                let dv = f16::from_bits(d.spectral()[band]).to_f32();
                let diff = dv - bv;
                if diff.abs() > 0.001 { spec_changed = true; }
                *ds_val = f16::from_f32(diff).to_bits();
            }

            if pos_changed || scale_changed || spec_changed {
                target.deltas.push(SplatDelta {
                    splat_index: i as u32,
                    d_position: dp,
                    d_scale: ds,
                    d_spectral,
                });
            }
        }

        target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn make_splat(position: [f32; 3], spectral: [u16; 16]) -> GaussianSplat {
        GaussianSplat::volume(position, [1.0, 1.0, 1.0], Quat::IDENTITY, 255, spectral)
    }

    #[test]
    fn splat_delta_position_only() {
        let delta = SplatDelta::position(0, [1.0, 0.0, 0.0]);
        assert_eq!(delta.splat_index, 0);
        assert_eq!(delta.d_position, [1.0, 0.0, 0.0]);
        assert_eq!(delta.d_spectral, [0u16; 16]);
    }

    #[test]
    fn splat_delta_with_spectral_roundtrip() {
        let delta = SplatDelta::position(0, [0.0; 3]).with_spectral([0.1; 16]);
        let unpacked = delta.d_spectral_f32();
        for v in unpacked {
            assert!((v - 0.1).abs() < 0.001, "expected ~0.1, got {v}");
        }
    }

    #[test]
    fn morph_target_set_evaluate_zero_weight() {
        let base_spectral = [f16::from_f32(0.5).to_bits(); 16];
        let base = vec![make_splat([0.0, 0.0, 0.0], base_spectral)];
        let mut set = MorphTargetSet::new(base.clone());

        let mut target = MorphTarget::new("test");
        target.add_delta(SplatDelta::position(0, [5.0, 5.0, 5.0]));
        set.add_target(target);

        let result = set.evaluate(&[0.0]);
        assert_eq!(result[0].position(), base[0].position());
    }

    #[test]
    fn morph_target_set_evaluate_full_weight() {
        let base_spectral = [f16::from_f32(0.5).to_bits(); 16];
        let base = vec![make_splat([0.0, 0.0, 0.0], base_spectral)];
        let mut set = MorphTargetSet::new(base);

        let d_pos = [1.0f32, 2.0, 3.0];
        let mut target = MorphTarget::new("test");
        target.add_delta(SplatDelta::position(0, d_pos));
        set.add_target(target);

        let result = set.evaluate(&[1.0]);
        assert!((result[0].position()[0] - 1.0).abs() < 1e-5);
        assert!((result[0].position()[1] - 2.0).abs() < 1e-5);
        assert!((result[0].position()[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn compute_diff_finds_changed_splats() {
        let spec = [f16::from_f32(0.5).to_bits(); 16];
        let base = vec![
            make_splat([0.0, 0.0, 0.0], spec),
            make_splat([1.0, 0.0, 0.0], spec),
        ];
        let mut deformed = base.clone();
        deformed[0].position_mut()[0] += 1.0;

        let target = MorphTargetSet::compute_diff("diff", &base, &deformed);
        assert_eq!(target.deltas.len(), 1);
        assert_eq!(target.deltas[0].splat_index, 0);
        assert!((target.deltas[0].d_position[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn compute_diff_spectral_change() {
        let spec_base = [f16::from_f32(0.2).to_bits(); 16];
        let mut spec_deformed: [u16; 16] = spec_base;
        spec_deformed[3] = f16::from_f32(0.8).to_bits();

        let base = vec![make_splat([0.0, 0.0, 0.0], spec_base)];
        let deformed = vec![make_splat([0.0, 0.0, 0.0], spec_deformed)];

        let target = MorphTargetSet::compute_diff("spec_diff", &base, &deformed);
        assert_eq!(target.deltas.len(), 1);
        let d_spec = target.deltas[0].d_spectral_f32();
        assert!(d_spec[3].abs() > 0.001, "expected nonzero spectral delta at band 3, got {}", d_spec[3]);
    }
}
