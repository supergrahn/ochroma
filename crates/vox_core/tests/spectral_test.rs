use vox_core::spectral::{SpectralBands, Illuminant, spectral_to_xyz, xyz_to_srgb};

#[test]
fn d65_illuminant_has_8_bands() {
    let d65 = Illuminant::d65();
    assert_eq!(d65.bands.len(), 8);
}

#[test]
fn flat_white_spd_under_d65_is_near_white() {
    let spd = SpectralBands([1.0; 8]);
    let d65 = Illuminant::d65();
    let xyz = spectral_to_xyz(&spd, &d65);
    let rgb = xyz_to_srgb(xyz);
    assert!(rgb[0] > 0.8, "R={}", rgb[0]);
    assert!(rgb[1] > 0.8, "G={}", rgb[1]);
    assert!(rgb[2] > 0.8, "B={}", rgb[2]);
}

#[test]
fn zero_spd_produces_black() {
    let spd = SpectralBands([0.0; 8]);
    let d65 = Illuminant::d65();
    let xyz = spectral_to_xyz(&spd, &d65);
    let rgb = xyz_to_srgb(xyz);
    assert_eq!(rgb, [0.0, 0.0, 0.0]);
}
