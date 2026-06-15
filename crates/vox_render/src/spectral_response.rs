//! Canonical spectral surface-response model (R14 "realize spectral").
//!
//! This is the single source of truth for how a spectral reflectance SPD
//! interacts with an illuminant SPD to produce a colour. The Spectra GPU
//! megakernel (`spectra/slang/megakernel.slang`, SDF surface accumulation)
//! mirrors this *exact* algorithm per hero wavelength, so a deterministic CPU
//! test of this module is a faithful proxy for the GPU spectral path.
//!
//! The physics, in one line:
//!
//! ```text
//! colour_XYZ = Σ_λ  R(λ) · L(λ) · CMF(λ)      (then ÷ CIE_Y integral → XYZ → sRGB)
//! ```
//!
//! where `R(λ)` is the material's per-band *reflectance* (NOT an RGB triple
//! upsampled by fakery) and `L(λ)` is the illuminant's per-band *power*.
//!
//! Because the same geometry+reflectance is multiplied by a *different* `L(λ)`
//! under a different light, the resulting colour genuinely shifts (metameric /
//! illuminant response). RGB-luminance fakery — `lum = luminance(albedo·light)`
//! smeared flat across wavelengths — physically cannot do this, which is the
//! `luminance(contrib)` smear R14 removes from the SDF surface path.
//!
//! ## Band grid
//!
//! 8 wavelength samples spanning 380–780 nm, identical to
//! `spectra/slang/spectral.slang` (`WAVELENGTH_SAMPLES = 8`):
//! 380, 437, 494, 551, 608, 665, 722, 780 nm.

/// Number of reflectance bands carried per material on the real-time GPU path.
/// Matches `WAVELENGTH_SAMPLES` in `spectra/slang/spectral.slang`.
pub const SPD_BANDS: usize = 8;

/// The 8 band-centre wavelengths (nm). Identical to the GPU grid.
pub const SPD_WAVELENGTHS: [f32; SPD_BANDS] =
    [380.0, 437.0, 494.0, 551.0, 608.0, 665.0, 722.0, 780.0];

// ---------------------------------------------------------------------------
// CIE 1931 2° observer — Wyman/Sloan/Shirley (2013) Gaussian approximation.
// Byte-for-byte the same functions the GPU uses (spectral_common.slang:20-44),
// so CPU and GPU integrate to the same XYZ for a given spectrum.
// ---------------------------------------------------------------------------

fn cie_x(wl: f32) -> f32 {
    let t1 = (wl - 442.0) * if wl < 442.0 { 0.0624 } else { 0.0374 };
    let t2 = (wl - 599.8) * if wl < 599.8 { 0.0264 } else { 0.0323 };
    let t3 = (wl - 501.1) * if wl < 501.1 { 0.0490 } else { 0.0382 };
    0.362 * (-0.5 * t1 * t1).exp() + 1.056 * (-0.5 * t2 * t2).exp()
        - 0.065 * (-0.5 * t3 * t3).exp()
}

fn cie_y(wl: f32) -> f32 {
    let t1 = (wl - 568.8) * if wl < 568.8 { 0.0213 } else { 0.0247 };
    let t2 = (wl - 530.9) * if wl < 530.9 { 0.0613 } else { 0.0322 };
    0.821 * (-0.5 * t1 * t1).exp() + 0.286 * (-0.5 * t2 * t2).exp()
}

fn cie_z(wl: f32) -> f32 {
    let t1 = (wl - 437.0) * if wl < 437.0 { 0.0845 } else { 0.0278 };
    let t2 = (wl - 459.0) * if wl < 459.0 { 0.0385 } else { 0.0725 };
    1.217 * (-0.5 * t1 * t1).exp() + 0.681 * (-0.5 * t2 * t2).exp()
}

/// CIE XYZ → linear sRGB (D65 matrix). Matches `xyz_to_srgb` on the GPU.
pub fn xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz;
    [
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ]
}

// ---------------------------------------------------------------------------
// Illuminant SPDs from real physics (Planck's law).
// ---------------------------------------------------------------------------

/// A per-band illuminant power distribution on the 8-band grid.
///
/// Carries the *relative* spectral power at each band centre. Absolute scale is
/// irrelevant: shading normalises by the illuminant's own luminance so a white
/// (R≡1) surface always integrates to a constant brightness regardless of the
/// light's colour temperature — only the *chromaticity* of the response shifts.
#[derive(Debug, Clone, Copy)]
pub struct IlluminantSpd {
    pub power: [f32; SPD_BANDS],
}

/// Planck's blackbody spectral radiance (relative), wavelength in nm,
/// temperature in Kelvin. Constant factors cancel under normalisation, so we
/// keep only the wavelength/temperature dependence:
///   M(λ,T) ∝ 1 / (λ^5 · (exp(hc / λ k T) − 1))
fn planck_relative(wl_nm: f32, temp_k: f32) -> f32 {
    // hc/k in nm·K (1.4388e7 nm·K is the second radiation constant c2).
    const C2_NM_K: f32 = 1.438_777e7;
    let l = wl_nm as f64;
    let t = temp_k as f64;
    let c2 = C2_NM_K as f64;
    let num = 1.0;
    let den = l.powi(5) * ((c2 / (l * t)).exp() - 1.0);
    (num / den) as f32
}

impl IlluminantSpd {
    /// Blackbody illuminant at the given colour temperature, sampled on the
    /// 8-band grid and normalised so its peak band is 1.0.
    pub fn blackbody(temp_k: f32) -> Self {
        let mut power = [0.0f32; SPD_BANDS];
        let mut peak = 0.0f32;
        for i in 0..SPD_BANDS {
            power[i] = planck_relative(SPD_WAVELENGTHS[i], temp_k);
            if power[i] > peak {
                peak = power[i];
            }
        }
        if peak > 0.0 {
            for p in power.iter_mut() {
                *p /= peak;
            }
        }
        Self { power }
    }

    /// Warm incandescent illuminant (~2700 K tungsten): red-heavy, blue-starved.
    pub fn warm_incandescent() -> Self {
        Self::blackbody(2700.0)
    }

    /// Cool daylight illuminant (~6500 K, D65-like blackbody): blue-rich.
    pub fn cool_daylight() -> Self {
        Self::blackbody(6500.0)
    }

    /// Equal-energy (flat) illuminant — every band equally bright. Useful as a
    /// neutral reference; not a real light.
    pub fn equal_energy() -> Self {
        Self {
            power: [1.0; SPD_BANDS],
        }
    }
}

// ---------------------------------------------------------------------------
// The shading kernel: reflectance × illuminant → XYZ → sRGB.
// ---------------------------------------------------------------------------

/// Integrate a reflectance SPD under an illuminant to CIE XYZ.
///
/// This is the algorithm the GPU megakernel performs per hero wavelength:
/// `radiance(λ) = R(λ) · L(λ)`, accumulated against the CIE observer and
/// divided by the illuminant's own Y integral so a perfect white reflector
/// (R≡1) lands at constant luminance under any illuminant — exactly the white
/// balance that makes an *illuminant colour shift* visible while a white card
/// stays white.
pub fn reflectance_to_xyz(reflectance: &[f32; SPD_BANDS], illuminant: &IlluminantSpd) -> [f32; 3] {
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut z = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    let mut norm_z = 0.0f32;
    for i in 0..SPD_BANDS {
        let wl = SPD_WAVELENGTHS[i];
        let l = illuminant.power[i];
        let power = reflectance[i] * l;
        let (cx, cy, cz) = (cie_x(wl), cie_y(wl), cie_z(wl));
        x += power * cx;
        y += power * cy;
        z += power * cz;
        norm_x += l * cx;
        norm_y += l * cy;
        norm_z += l * cz;
    }
    if norm_y <= 1e-6 {
        return [0.0, 0.0, 0.0];
    }
    // Per-channel white balance to the illuminant, then scale X and Z to the
    // CIE D65 white point so a flat (R≡1) reflector lands at the sRGB white
    // point [≈1,1,1] under ANY illuminant. Identical structure to
    // `vox_core::spectral::spectral_to_xyz` — the engine's authoritative
    // 16-band integrator — so CPU and the 16-band path agree.
    let nx = if norm_x > 0.0 { norm_x } else { norm_y };
    let nz = if norm_z > 0.0 { norm_z } else { norm_y };
    [(x / nx) * 0.9505, y / norm_y, (z / nz) * 1.0888]
}

/// Full surface response: reflectance SPD under an illuminant → linear sRGB.
pub fn reflectance_to_rgb(reflectance: &[f32; SPD_BANDS], illuminant: &IlluminantSpd) -> [f32; 3] {
    let rgb = xyz_to_srgb(reflectance_to_xyz(reflectance, illuminant));
    [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
}

// ---------------------------------------------------------------------------
// Reflectance construction (the REAL path — no RGB-luminance fakery).
// ---------------------------------------------------------------------------

/// Resample the engine's 16-band reflectance SPD (380–755 nm @ 25 nm,
/// `vox_core::spectral::BAND_WAVELENGTHS`) onto the GPU's 8-band grid by linear
/// interpolation in wavelength. This is the bridge that carries a *real*
/// per-band reflectance from the engine's authored material to the GPU — it
/// does NOT collapse to RGB and re-upsample.
pub fn reflectance_from_bands16(bands16: &[f32; 16]) -> [f32; SPD_BANDS] {
    use vox_core::spectral::BAND_WAVELENGTHS;
    let lo = BAND_WAVELENGTHS[0];
    let hi = BAND_WAVELENGTHS[15];
    std::array::from_fn(|i| {
        let wl = SPD_WAVELENGTHS[i].clamp(lo, hi);
        // Locate the bracketing 16-band samples.
        let mut j = 0;
        while j + 1 < 16 && BAND_WAVELENGTHS[j + 1] < wl {
            j += 1;
        }
        let w0 = BAND_WAVELENGTHS[j];
        let w1 = BAND_WAVELENGTHS[(j + 1).min(15)];
        if (w1 - w0).abs() < 1e-6 {
            bands16[j]
        } else {
            let t = ((wl - w0) / (w1 - w0)).clamp(0.0, 1.0);
            bands16[j] * (1.0 - t) + bands16[(j + 1).min(15)] * t
        }
    })
}

/// The Smits (1999) RGB→reflectance upsample, mirroring
/// `rgb_to_spectral` in `spectra/slang/spectral.slang`. This is the *fallback*
/// used only when a material supplies no authored SPD — it yields a plausible
/// smooth reflectance (which DOES respond to illuminant, unlike the luminance
/// smear). Materials with a real SPD bypass this entirely.
pub fn reflectance_from_rgb(rgb: [f32; 3]) -> [f32; SPD_BANDS] {
    const WHITE: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    const RED: [f32; 8] = [0.05, 0.02, 0.0, 0.0, 0.1, 0.8, 0.97, 1.0];
    const GREEN: [f32; 8] = [0.0, 0.05, 0.3, 0.85, 0.6, 0.1, 0.0, 0.0];
    const BLUE: [f32; 8] = [0.95, 0.9, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0];

    let (mut r, mut g, mut b) = (rgb[0], rgb[1], rgb[2]);
    let mut spec = [0.0f32; 8];

    let mn = r.min(g).min(b);
    for i in 0..8 {
        spec[i] += mn * WHITE[i];
    }
    r -= mn;
    g -= mn;
    b -= mn;

    if r > 0.0 && g > 0.0 {
        let k = r.min(g);
        for i in 0..8 {
            spec[i] += k * (RED[i] + GREEN[i]) * 0.5;
        }
        r -= k;
        g -= k;
    }
    if g > 0.0 && b > 0.0 {
        let k = g.min(b);
        for i in 0..8 {
            spec[i] += k * (GREEN[i] + BLUE[i]) * 0.5;
        }
        g -= k;
        b -= k;
    }
    if r > 0.0 && b > 0.0 {
        let k = r.min(b);
        for i in 0..8 {
            spec[i] += k * (RED[i] + BLUE[i]) * 0.5;
        }
        r -= k;
        b -= k;
    }
    for i in 0..8 {
        spec[i] += r * RED[i] + g * GREEN[i] + b * BLUE[i];
        spec[i] = spec[i].max(0.0);
    }
    spec
}
