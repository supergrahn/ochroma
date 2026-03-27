use glam::Vec3;
use vox_core::types::GaussianSplat;
use vox_render::destruction::{DestructionMask, apply_destruction_masks, generate_debris};

fn make_splat(pos: [f32; 3]) -> GaussianSplat {
    GaussianSplat {
        position: pos, scale: [0.1, 0.1, 0.1], rotation: [0, 0, 0, 32767],
        opacity: 255, _pad: [0; 3], spectral: [15360; 8],
    }
}

#[test]
fn destruction_reduces_opacity_at_impact() {
    let splats = vec![make_splat([0.0, 0.0, 0.0])];
    let masks = vec![DestructionMask {
        instance_id: 0, impact_point: Vec3::ZERO, radius: 5.0, progression: 1.0,
    }];
    let result = apply_destruction_masks(&splats, &masks);
    assert!(result[0].opacity < 255, "Opacity should be reduced at impact");
}

#[test]
fn destruction_does_not_affect_distant_splats() {
    let splats = vec![make_splat([100.0, 0.0, 0.0])];
    let masks = vec![DestructionMask {
        instance_id: 0, impact_point: Vec3::ZERO, radius: 5.0, progression: 1.0,
    }];
    let result = apply_destruction_masks(&splats, &masks);
    assert_eq!(result[0].opacity, 255, "Distant splat should be unaffected");
}

#[test]
fn zero_progression_leaves_intact() {
    let splats = vec![make_splat([0.0, 0.0, 0.0])];
    let masks = vec![DestructionMask {
        instance_id: 0, impact_point: Vec3::ZERO, radius: 5.0, progression: 0.0,
    }];
    let result = apply_destruction_masks(&splats, &masks);
    assert_eq!(result[0].opacity, 255);
}

#[test]
fn debris_generates_correct_count() {
    let debris = generate_debris(Vec3::ZERO, 3.0, 50, 42);
    assert_eq!(debris.len(), 50);
}
