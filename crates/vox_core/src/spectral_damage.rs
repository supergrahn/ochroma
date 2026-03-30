//! Spectral damage model — damage attenuated per band by material armor.
//!
//! Band conventions:
//! - Bands 0–4:  UV / violet / blue  (radiation, UV burns)
//! - Bands 5–9:  green / cyan / yellow (sonic, blunt)
//! - Bands 10–15: red / orange / IR   (fire, heat, laser)

use half::f16;

pub fn apply_spectral_damage(
    health:         &mut f32,
    damage:         &[f32; 16],
    armor_spectral: &[u16; 16],
    max_health:     f32,
) -> f32 {
    let mut total = 0.0f32;
    for b in 0..16 {
        let armor_fraction = (armor_spectral[b] as f32) / 65535.0;
        let effective = damage[b] * (1.0 - armor_fraction).max(0.0);
        total += effective;
    }
    let new_health = (*health - total).clamp(0.0, max_health);
    *health = new_health;
    total
}

pub struct DamageType;

impl DamageType {
    pub fn fire(intensity: f32) -> [f32; 16] {
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
            .map(|v| v * intensity)
    }

    pub fn radiation(intensity: f32) -> [f32; 16] {
        [0.35, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }

    pub fn blunt(intensity: f32) -> [f32; 16] {
        [0.0, 0.0, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.08, 0.05, 0.0, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }

    pub fn laser(intensity: f32) -> [f32; 16] {
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.9, 0.05, 0.0, 0.0, 0.0]
            .map(|v| v * intensity)
    }
}

pub fn is_fire_band_exposure(spectral_field: &[f32; 16], threshold: f32) -> bool {
    let fire_energy: f32 = spectral_field[10] + spectral_field[11] + spectral_field[12]
                         + spectral_field[13] + spectral_field[14] + spectral_field[15];
    fire_energy / 6.0 > threshold
}

pub fn decode_spectral_u16(spectral: &[u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = f16::from_bits(spectral[i]).to_f32();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_armor() -> [u16; 16] { [0u16; 16] }
    fn full_armor() -> [u16; 16] { [65535u16; 16] }
    fn fire_armor() -> [u16; 16] {
        let mut a = [0u16; 16];
        a[10] = 65535; a[11] = 65535; a[12] = 65535;
        a[13] = 65535; a[14] = 65535; a[15] = 65535;
        a
    }

    #[test]
    fn no_armor_takes_full_damage() {
        let mut health = 100.0;
        let damage = DamageType::fire(10.0);
        let applied = apply_spectral_damage(&mut health, &damage, &no_armor(), 100.0);
        assert!(applied > 0.0, "should take damage: got {}", applied);
        assert!(health < 100.0, "health should decrease: got {}", health);
    }

    #[test]
    fn full_armor_blocks_all_damage() {
        let mut health = 100.0;
        let damage = DamageType::fire(10.0);
        let applied = apply_spectral_damage(&mut health, &damage, &full_armor(), 100.0);
        assert!(applied < 0.001, "full armor should block all damage, got {}", applied);
        assert!((health - 100.0).abs() < 0.001, "health should be unchanged, got {}", health);
    }

    #[test]
    fn fire_armor_blocks_fire_not_radiation() {
        let mut health = 100.0;
        let fire_dmg = DamageType::fire(10.0);
        let rad_dmg  = DamageType::radiation(10.0);
        let fire_applied = apply_spectral_damage(&mut health, &fire_dmg, &fire_armor(), 100.0);
        let rad_applied  = apply_spectral_damage(&mut health, &rad_dmg, &fire_armor(), 100.0);
        assert!(fire_applied < 1.0, "fire armor should block fire (bands 10-15), applied {}", fire_applied);
        assert!(rad_applied > 5.0, "fire armor should NOT block radiation (bands 0-4), applied {}", rad_applied);
    }

    #[test]
    fn health_clamps_at_zero() {
        let mut health = 5.0;
        let damage = DamageType::blunt(1000.0);
        apply_spectral_damage(&mut health, &damage, &no_armor(), 100.0);
        assert_eq!(health, 0.0, "health should clamp at 0, got {}", health);
    }

    #[test]
    fn health_clamps_at_max_health() {
        let mut health = 100.0;
        let zero_damage = [0.0f32; 16];
        apply_spectral_damage(&mut health, &zero_damage, &no_armor(), 100.0);
        assert!((health - 100.0).abs() < 0.001, "health should remain at max");
    }

    #[test]
    fn fire_band_threshold_detects_correctly() {
        let low_fire:  [f32; 16] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let high_fire: [f32; 16] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.85, 0.8, 0.85, 0.9, 0.85];
        assert!(!is_fire_band_exposure(&low_fire, 0.5), "low fire energy should not trigger threshold");
        assert!(is_fire_band_exposure(&high_fire, 0.5), "high fire energy (bands 10-15 avg 0.85) should trigger threshold");
    }

    #[test]
    fn decode_spectral_roundtrips() {
        use half::f16;
        let input = [
            f16::from_f32(0.5).to_bits(), f16::from_f32(0.25).to_bits(),
            f16::from_f32(0.0).to_bits(), f16::from_f32(1.0).to_bits(),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let decoded = decode_spectral_u16(&input);
        assert!((decoded[0] - 0.5).abs() < 0.001, "band 0: {}", decoded[0]);
        assert!((decoded[1] - 0.25).abs() < 0.001, "band 1: {}", decoded[1]);
        assert!((decoded[3] - 1.0).abs() < 0.001, "band 3: {}", decoded[3]);
    }
}
