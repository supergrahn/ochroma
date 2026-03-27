use vox_render::subsurface::*;

#[test]
fn vegetation_transmits_red_more_than_blue() {
    let profile = SubsurfaceProfile::vegetation();
    let transmitted = profile.spectral_shift(1.0);
    // Red/NIR bands (index 6,7) should transmit more than blue (index 0,1)
    assert!(transmitted.0[6] > transmitted.0[0],
        "Red should transmit more than blue through leaves: red={}, blue={}", transmitted.0[6], transmitted.0[0]);
}

#[test]
fn thicker_material_transmits_less() {
    let profile = SubsurfaceProfile::wax();
    let thin = profile.spectral_shift(0.1);
    let thick = profile.spectral_shift(1.0);
    for i in 0..8 {
        assert!(thin.0[i] >= thick.0[i], "Thicker material should transmit less at band {}", i);
    }
}

#[test]
fn opaque_profile_transmits_nothing() {
    let mut profile = SubsurfaceProfile::wax();
    profile.translucency = 0.0;
    let transmitted = profile.spectral_shift(1.0);
    for v in &transmitted.0 {
        assert_eq!(*v, 0.0, "Zero translucency should transmit nothing");
    }
}

#[test]
fn skin_has_red_shift() {
    let profile = SubsurfaceProfile::skin();
    let shift = profile.spectral_shift(0.5);
    // Skin transmits red more -> warm colour when backlit
    assert!(shift.0[6] > shift.0[2], "Skin should shift toward red");
}
