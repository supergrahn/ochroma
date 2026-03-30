//! Spectral damage model.
//!
//! Tracks damage to objects with spectral consequences: a burning wall darkens
//! toward char (high red→low all), a rusting pipe shifts from metal grey to
//! rust orange, a cracking stone develops grey-to-dark transitions.
//! Damage is applied as a weighted lerp from the pristine spectral toward
//! a damage-state target spectral.

use vox_core::types::GaussianSplat;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DamageType {
    Fire,
    Rust,
    Crack,
    Frost,
    Acid,
}

pub struct DamageState {
    pub health: f32,
    pub damage_type: DamageType,
    pub accumulated: f32,
}

impl DamageState {
    pub fn new(damage_type: DamageType) -> Self {
        Self {
            health: 1.0,
            damage_type,
            accumulated: 0.0,
        }
    }
}

pub fn damage_spectral_target(dtype: DamageType) -> [f32; 16] {
    match dtype {
        DamageType::Fire  => [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.18, 0.12],
        DamageType::Rust  => [0.04, 0.04, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.65, 0.72, 0.70, 0.65, 0.60, 0.55, 0.45, 0.35],
        DamageType::Crack => [0.13, 0.13, 0.14, 0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14],
        DamageType::Frost => [0.80, 0.85, 0.88, 0.90, 0.90, 0.88, 0.85, 0.83, 0.80, 0.77, 0.73, 0.68, 0.63, 0.60, 0.57, 0.55],
        DamageType::Acid  => [0.25, 0.40, 0.55, 0.45, 0.30, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
    }
}

pub fn apply_damage_to_spectral(pristine: &[f32; 16], damage: &DamageState) -> [f32; 16] {
    let target = damage_spectral_target(damage.damage_type);
    let t = damage.accumulated.clamp(0.0, 1.0);
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = pristine[i] * (1.0 - t) + target[i] * t;
    }
    out
}

pub fn apply_damage_to_splats(splats: &mut [GaussianSplat], damage: &DamageState) {
    for splat in splats.iter_mut() {
        let mut pristine = [0.0f32; 16];
        for (b, val) in pristine.iter_mut().enumerate() {
            *val = half::f16::from_bits(splat.spectral()[b]).to_f32();
        }
        let result = apply_damage_to_spectral(&pristine, damage);
        for (b, &r) in result.iter().enumerate() {
            splat.spectral_mut()[b] = half::f16::from_f32(r).to_bits();
        }
    }
}

pub struct SpectralDamageComponent {
    pub states: Vec<DamageState>,
}

impl SpectralDamageComponent {
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    pub fn add_zone(&mut self, dtype: DamageType) -> usize {
        let idx = self.states.len();
        self.states.push(DamageState::new(dtype));
        idx
    }

    pub fn apply_damage(&mut self, zone: usize, amount: f32) {
        let state = &mut self.states[zone];
        state.health = (state.health - amount).max(0.0);
        state.accumulated = (state.accumulated + amount).clamp(0.0, 1.0);
    }

    pub fn is_destroyed(&self, zone: usize) -> bool {
        self.states[zone].health <= 0.0
    }
}

impl Default for SpectralDamageComponent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn bright_spectral() -> [f32; 16] {
        [1.0; 16]
    }

    #[test]
    fn fire_damage_darkens_spectral() {
        let pristine = bright_spectral();
        let damage = DamageState { health: 0.0, damage_type: DamageType::Fire, accumulated: 1.0 };
        let result = apply_damage_to_spectral(&pristine, &damage);
        for i in 0..16 {
            assert!(result[i] < pristine[i], "band {} should be darker after full fire damage", i);
        }
    }

    #[test]
    fn frost_damage_shifts_toward_blue() {
        let pristine = [0.1f32; 16];
        let damage = DamageState { health: 0.5, damage_type: DamageType::Frost, accumulated: 1.0 };
        let result = apply_damage_to_spectral(&pristine, &damage);
        // Band 0 (blue end) should increase since frost target[0] = 0.8 > pristine[0] = 0.1
        assert!(result[0] > pristine[0], "band 0 should increase toward frost blue-white");
    }

    #[test]
    fn zero_damage_unchanged_spectral() {
        let pristine = [0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.1, 0.8,
                        0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.1, 0.8];
        let damage = DamageState { health: 1.0, damage_type: DamageType::Rust, accumulated: 0.0 };
        let result = apply_damage_to_spectral(&pristine, &damage);
        for i in 0..16 {
            assert!((result[i] - pristine[i]).abs() < 1e-6, "band {} should be unchanged at 0 accumulated damage", i);
        }
    }

    #[test]
    fn full_damage_is_target_spectral() {
        let pristine = [0.9f32; 16];
        let dtype = DamageType::Acid;
        let damage = DamageState { health: 0.0, damage_type: dtype, accumulated: 1.0 };
        let target = damage_spectral_target(dtype);

        // apply_damage_to_spectral directly (no splat needed for pure logic check)
        let result = apply_damage_to_spectral(&pristine, &damage);
        for b in 0..16 {
            assert!((result[b] - target[b]).abs() < 1e-6, "band {} result={} target={}", b, result[b], target[b]);
        }

        // Also verify f16 round-trip via apply_damage_to_splats
        let mut splat = GaussianSplat::volume(
            [0.0; 3],
            [1.0; 3],
            Quat::IDENTITY,
            255,
            [0u16; 16],
        );
        for b in 0..16 {
            splat.spectral_mut()[b] = half::f16::from_f32(pristine[b]).to_bits();
        }
        apply_damage_to_splats(std::slice::from_mut(&mut splat), &damage);
        for b in 0..16 {
            let decoded = half::f16::from_bits(splat.spectral()[b]).to_f32();
            // f16 round-trip tolerance ~0.002
            assert!((decoded - target[b]).abs() < 0.002, "band {} decoded={} target={}", b, decoded, target[b]);
        }
    }

    #[test]
    fn component_health_decrements() {
        let mut comp = SpectralDamageComponent::new();
        let zone = comp.add_zone(DamageType::Crack);
        comp.apply_damage(zone, 0.3);
        assert!((comp.states[zone].health - 0.7).abs() < 1e-6);
    }

    #[test]
    fn component_destroyed_at_zero_health() {
        let mut comp = SpectralDamageComponent::new();
        let zone = comp.add_zone(DamageType::Fire);
        assert!(!comp.is_destroyed(zone));
        comp.apply_damage(zone, 1.5); // more than 1.0 to ensure it hits zero
        assert!(comp.is_destroyed(zone));
    }
}
