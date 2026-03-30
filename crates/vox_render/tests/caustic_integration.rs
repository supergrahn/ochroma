//! Integration test: glass prism produces band-separated caustic.
//! Verifies that Snell+Cauchy correctly orders spectral bands by refraction angle.

use vox_render::spectral_caustics::{CauchyGlass, SpectralCaustics, BAND_UM};
use glam::Vec3;

#[test]
fn prism_separates_white_light_violet_to_red() {
    let glass = CauchyGlass::n_bk7();
    let angle = 30.0_f32.to_radians();
    let incident = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let white_light = [1.0f32; 16];

    let refraction = SpectralCaustics::refract(incident, normal, white_light, &glass);

    let x_components: Vec<f32> = refraction.directions.iter().map(|d| d.x).collect();

    let violet_x = x_components[0];
    let red_x = x_components[15];
    println!("violet x={:.5}, red x={:.5}", violet_x, red_x);

    assert!(
        violet_x < red_x,
        "violet (x={:.5}) should bend more than red (x={:.5}) through glass prism",
        violet_x, red_x
    );

    // Verify monotonic ordering: each band bends at least as much as the next longer wavelength
    for b in 0..15 {
        assert!(
            x_components[b] <= x_components[b + 1] + 1e-5,
            "band {} ({}nm, x={:.5}) should refract at least as much as band {} ({}nm, x={:.5})",
            b, (BAND_UM[b] * 1000.0) as u32,
            x_components[b],
            b + 1, (BAND_UM[b + 1] * 1000.0) as u32,
            x_components[b + 1]
        );
    }
}

#[test]
fn chromatic_spread_increases_with_incidence_angle() {
    let glass = CauchyGlass::n_bk7();
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let white = [1.0f32; 16];

    let spread_10 = {
        let a = 10.0_f32.to_radians();
        let inc = Vec3::new(a.sin(), -a.cos(), 0.0);
        SpectralCaustics::chromatic_spread(&SpectralCaustics::refract(inc, normal, white, &glass))
    };
    let spread_45 = {
        let a = 45.0_f32.to_radians();
        let inc = Vec3::new(a.sin(), -a.cos(), 0.0);
        SpectralCaustics::chromatic_spread(&SpectralCaustics::refract(inc, normal, white, &glass))
    };

    println!("spread at 10°: {:.6} rad, at 45°: {:.6} rad", spread_10, spread_45);
    assert!(
        spread_45 > spread_10,
        "chromatic spread should increase with incidence angle: 10°→{:.6}rad, 45°→{:.6}rad",
        spread_10, spread_45
    );
}

#[test]
fn n_bk7_dispersion_matches_published_values() {
    let glass = CauchyGlass::n_bk7();
    let n_blue = glass.ior(0.460);
    let n_yellow = glass.ior(0.580);
    assert!(
        n_blue > n_yellow,
        "blue IOR ({:.4}) should exceed yellow IOR ({:.4}) for normal dispersion",
        n_blue, n_yellow
    );
    // Overall dispersion (n_violet - n_red) should be ~0.017 for N-BK7
    let dispersion = glass.ior(0.380) - glass.ior(0.660);
    println!("N-BK7 dispersion: {:.4}", dispersion);
    assert!(
        dispersion > 0.010 && dispersion < 0.030,
        "N-BK7 dispersion should be ~0.017, got {:.4}", dispersion
    );
}
