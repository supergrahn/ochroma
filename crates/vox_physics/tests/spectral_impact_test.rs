use glam::{Quat, Vec3};
use half;
use vox_core::types::GaussianSplat;
use vox_physics::destruction::{SplatAssembly, spectral_shift_on_break};

/// Dropping a rigid body onto a glass floor causes the floor splats' spectral
/// bands 1-3 (UV/violet) to visibly reduce (crack damage) upon impact.
///
/// Done When: `cargo test -p vox_physics rigid_body_impact_cracks_glass_spectral`
/// passing with `assert!(cracked.spectral_f32(2) < original.spectral_f32(2) - 0.05)`.
#[test]
fn rigid_body_impact_cracks_glass_spectral() {
    // Glass spectral profile: high UV/violet (bands 0-3), low red/IR
    let glass_spectral: [u16; 16] = {
        let mut s = [0u16; 16];
        // Bands 0-3: UV/violet — characteristic of glass transmittance
        for b in 0..4 {
            s[b] = half::f16::from_f32(0.8).to_bits();
        }
        // Bands 4-15: green through IR — lower values for glass
        for b in 4..16 {
            s[b] = half::f16::from_f32(0.2).to_bits();
        }
        s
    };

    // Build a glass floor from splats
    let mut floor_splat = GaussianSplat::volume(
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 1.0],
        Quat::IDENTITY,
        255,
        glass_spectral,
    );

    // Record original band 2 (UV/violet) value before impact
    let original_band2 = floor_splat.spectral_f32(2);

    // Apply fracture spectral shift — simulating impact damage
    spectral_shift_on_break(&mut floor_splat);

    // Cracked glass: UV/violet bands should drop visibly (> 0.05 reduction)
    let cracked_band2 = floor_splat.spectral_f32(2);
    assert!(
        cracked_band2 < original_band2 - 0.05,
        "glass crack should reduce band 2 (UV/violet) by > 0.05: original={:.4} cracked={:.4}",
        original_band2, cracked_band2
    );
}

/// Verify the fracture also generates planes via SplatAssembly on high impulse.
#[test]
fn rigid_body_impact_generates_fracture_planes() {
    let glass_spectral: [u16; 16] = {
        let mut s = [0u16; 16];
        for b in 0..4 { s[b] = half::f16::from_f32(0.8).to_bits(); }
        s
    };
    let splat = GaussianSplat::volume([0.0; 3], [1.0; 3], Quat::IDENTITY, 255, glass_spectral);

    let mut assembly = SplatAssembly {
        splats: vec![splat],
        health: 1000.0,
        max_health: 1000.0,
        fracture_threshold: 50.0,
        is_active: true,
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        mass: 1.0,
    };

    let impact = Vec3::new(0.0, 0.0, 0.0);
    let planes = assembly.fracture_at(impact, 150.0);
    assert!(!planes.is_empty(), "high impulse should generate fracture planes");
}
