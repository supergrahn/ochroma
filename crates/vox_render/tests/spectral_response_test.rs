//! R14 "realize spectral" — non-vacuous gate.
//!
//! Proves the engine's spectral surface response is REAL physics, not RGB
//! luminance fakery:
//!
//!   1. Illuminant response — one fixed reflectance SPD renders to DIFFERENT
//!      colours under a warm (2700 K) vs cool (6500 K) illuminant.
//!   2. Metamerism — a constructed metameric pair (two distinct SPDs that match
//!      under illuminant A) DIVERGES under illuminant B. RGB-luminance fakery is
//!      mathematically incapable of this: with fakery the colour is a function of
//!      the RGB triple alone, so two SPDs sharing an XYZ under A would track each
//!      other under every illuminant. Divergence proves true per-wavelength
//!      reflectance × illuminant integration.
//!
//! Deterministic: pure arithmetic, no RNG, no GPU dispatch. The Spectra GPU
//! megakernel mirrors this exact algorithm per hero wavelength (see
//! `spectral_response.rs` module docs), so this is a faithful proxy.

use vox_render::spectral_response::{
    reflectance_to_rgb, reflectance_to_xyz, IlluminantSpd, SPD_BANDS,
};

/// CIE76-ish perceptual-ish delta: Euclidean distance in linear sRGB. Crude but
/// monotone with colour difference; thresholds chosen well above float noise.
fn color_delta(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn spectral_material_responds_to_illuminant() {
    // A reddish surface: high reflectance in the long-wavelength bands, low in
    // the short. Authored as a real per-band reflectance, NOT an RGB triple.
    let red_reflectance: [f32; SPD_BANDS] =
        [0.04, 0.05, 0.06, 0.10, 0.45, 0.85, 0.92, 0.90];

    let warm = IlluminantSpd::warm_incandescent(); // 2700 K
    let cool = IlluminantSpd::cool_daylight(); // 6500 K

    let rgb_warm = reflectance_to_rgb(&red_reflectance, &warm);
    let rgb_cool = reflectance_to_rgb(&red_reflectance, &cool);

    let delta = color_delta(rgb_warm, rgb_cool);
    println!(
        "[illuminant response] same red SPD:\n  warm 2700K -> {:?}\n  cool 6500K -> {:?}\n  delta = {:.5}",
        rgb_warm, rgb_cool, delta
    );

    // A real spectral surface MUST shift colour with the illuminant. RGB fakery
    // (luminance smear) would shift only brightness, not chromaticity.
    assert!(
        delta > 0.02,
        "spectral material must respond to illuminant colour: delta {:.5} <= 0.02 \
         (would indicate RGB-luminance fakery)",
        delta
    );

    // And the warm illuminant must push a red surface warmer (more red, less
    // blue) than the cool illuminant — a directional, physical check.
    assert!(
        rgb_warm[0] / rgb_warm[2].max(1e-4) > rgb_cool[0] / rgb_cool[2].max(1e-4),
        "warm illuminant must raise the red/blue ratio vs cool: warm {:?}, cool {:?}",
        rgb_warm,
        rgb_cool
    );
}

/// Build a metameric partner of `base` under `illum_a`: a DIFFERENT SPD whose
/// XYZ under `illum_a` equals `base`'s. We perturb `base` along a direction in
/// the black space of the CIE-under-A response operator (i.e. a spectral change
/// that the A-illuminated observer cannot see), found by projecting an
/// arbitrary "wiggle" out of the 3D XYZ-response subspace via Gram-Schmidt.
fn metameric_partner(base: &[f32; SPD_BANDS], illum_a: &IlluminantSpd) -> [f32; SPD_BANDS] {
    // The three response vectors r_k[i] = CMF_k(λ_i) · L_A(λ_i). A spectral
    // change Δ is invisible under A iff Δ · r_k = 0 for k in {X,Y,Z}.
    // Reconstruct r_k by finite differences of reflectance_to_xyz (linear in R).
    let basis = |chan: usize| -> [f32; SPD_BANDS] {
        std::array::from_fn(|i| {
            let mut e = [0.0f32; SPD_BANDS];
            e[i] = 1.0;
            reflectance_to_xyz(&e, illum_a)[chan]
        })
    };
    let dot = |a: &[f32; SPD_BANDS], b: &[f32; SPD_BANDS]| -> f32 {
        (0..SPD_BANDS).map(|i| a[i] * b[i]).sum()
    };
    let project_out = |v: &mut [f32; SPD_BANDS], u: &[f32; SPD_BANDS]| {
        let uu = dot(u, u);
        if uu > 1e-12 {
            let c = dot(v, u) / uu;
            for i in 0..SPD_BANDS {
                v[i] -= c * u[i];
            }
        }
    };

    // Orthonormalise the three response vectors first (Gram-Schmidt) so that
    // projecting `wiggle` out of each ORTHOGONAL basis vector truly removes its
    // component in span{rx,ry,rz}. (Sequential projection against the raw,
    // non-orthogonal rx/ry/rz does NOT.)
    let o1 = basis(0);
    let mut o2 = basis(1);
    let mut o3 = basis(2);
    project_out(&mut o2, &o1);
    project_out(&mut o3, &o1);
    project_out(&mut o3, &o2);

    // An arbitrary spectral wiggle with structure (alternating sign) so it has
    // components both inside and outside the visible subspace.
    let mut wiggle: [f32; SPD_BANDS] =
        std::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
    // Remove the visible component => remainder is invisible under A.
    project_out(&mut wiggle, &o1);
    project_out(&mut wiggle, &o2);
    project_out(&mut wiggle, &o3);

    // Scale the invisible direction so the partner stays a plausible reflectance
    // in [0,1] without clamping (clamping would break the metamerism).
    let max_abs = wiggle.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let headroom = (0..SPD_BANDS)
        .map(|i| {
            let w = wiggle[i] / max_abs;
            if w > 0.0 {
                (1.0 - base[i]) / w
            } else if w < 0.0 {
                (0.0 - base[i]) / w
            } else {
                f32::INFINITY
            }
        })
        .fold(f32::INFINITY, f32::min);
    // Use 80% of headroom for a sizeable, unclamped perturbation.
    let scale = 0.8 * headroom / max_abs;

    std::array::from_fn(|i| (base[i] + scale * wiggle[i]).clamp(0.0, 1.0))
}

#[test]
fn spectral_metameric_pair_matches_under_a_diverges_under_b() {
    let illum_a = IlluminantSpd::cool_daylight(); // matching illuminant (D65-like)
    let illum_b = IlluminantSpd::warm_incandescent(); // test illuminant (2700 K)

    // A mid-grey-ish base reflectance with some structure.
    let base: [f32; SPD_BANDS] = [0.45, 0.50, 0.48, 0.52, 0.50, 0.47, 0.49, 0.46];
    let partner = metameric_partner(&base, &illum_a);

    // The two SPDs must be genuinely DIFFERENT spectra.
    let spectral_diff: f32 = (0..SPD_BANDS)
        .map(|i| (base[i] - partner[i]).abs())
        .sum::<f32>();
    println!(
        "[metamer] base    = {:?}\n          partner = {:?}\n          Σ|Δ reflectance| = {:.4}",
        base, partner, spectral_diff
    );
    assert!(
        spectral_diff > 0.1,
        "metameric partner must be a distinct spectrum (Σ|Δ| = {:.4})",
        spectral_diff
    );

    // Under illuminant A they must MATCH (constructed to be metameric).
    let a_base = reflectance_to_rgb(&base, &illum_a);
    let a_partner = reflectance_to_rgb(&partner, &illum_a);
    let delta_a = color_delta(a_base, a_partner);

    // Under illuminant B they must DIVERGE (true spectral; fakery cannot do this).
    let b_base = reflectance_to_rgb(&base, &illum_b);
    let b_partner = reflectance_to_rgb(&partner, &illum_b);
    let delta_b = color_delta(b_base, b_partner);

    println!(
        "[metamer] under A (6500K): base {:?} vs partner {:?}  delta_A = {:.6}",
        a_base, a_partner, delta_a
    );
    println!(
        "[metamer] under B (2700K): base {:?} vs partner {:?}  delta_B = {:.6}",
        b_base, b_partner, delta_b
    );
    println!(
        "[metamer] divergence ratio delta_B / delta_A = {:.1}x",
        delta_b / delta_a.max(1e-9)
    );

    assert!(
        delta_a < 0.005,
        "metameric pair must match under illuminant A: delta_A {:.6} >= 0.005",
        delta_a
    );
    assert!(
        delta_b > 0.02,
        "metameric pair must DIVERGE under illuminant B: delta_B {:.6} <= 0.02 \
         (RGB fakery would keep them matched)",
        delta_b
    );
    assert!(
        delta_b > 10.0 * delta_a.max(1e-9),
        "divergence under B must dwarf the match under A: delta_B {:.6} not >> delta_A {:.6}",
        delta_b,
        delta_a
    );
}

#[test]
fn spectral_white_reflector_is_illuminant_invariant_grey() {
    // A perfect white reflector (R ≡ 1) must read as neutral grey under ANY
    // illuminant — this is the white-balance the shading kernel applies, and it
    // is what makes a coloured material's shift *visible* (the reference stays
    // white). This guards that the illuminant response is chromatic, not just a
    // global tint applied to everything.
    let white = [1.0f32; SPD_BANDS];
    let warm = reflectance_to_rgb(&white, &IlluminantSpd::warm_incandescent());
    let cool = reflectance_to_rgb(&white, &IlluminantSpd::cool_daylight());
    println!("[white] warm {:?}  cool {:?}", warm, cool);

    for (label, rgb) in [("warm", warm), ("cool", cool)] {
        let maxc = rgb[0].max(rgb[1]).max(rgb[2]);
        let minc = rgb[0].min(rgb[1]).min(rgb[2]);
        assert!(
            maxc > 0.5,
            "white reflector under {label} must be bright (max channel {maxc:.3})"
        );
        assert!(
            (maxc - minc) < 0.20 * maxc,
            "white reflector under {label} must stay near-neutral: spread {:.3} of max {:.3}",
            maxc - minc,
            maxc
        );
    }
}
