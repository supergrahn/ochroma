pub const BAND_WAVELENGTHS: [f32; 8] = [380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0];
pub const BAND_SPACING: f32 = 40.0;

#[derive(Debug, Clone, Copy)]
pub struct SpectralBands(pub [f32; 8]);

#[derive(Debug, Clone)]
pub struct Illuminant {
    pub bands: [f32; 8],
}

// CIE 1931 2° observer at our 8 bands
const CIE_X: [f32; 8] = [0.0014, 0.0434, 0.3362, 0.0049, 0.2904, 0.9163, 0.5419, 0.0874];
const CIE_Y: [f32; 8] = [0.0000, 0.0116, 0.0600, 0.3230, 0.9540, 0.8700, 0.3810, 0.0468];
const CIE_Z: [f32; 8] = [0.0065, 0.2074, 1.7721, 0.2720, 0.0633, 0.0017, 0.0017, 0.0000];

impl Illuminant {
    pub fn d65() -> Self {
        Self { bands: [49.98, 68.70, 100.15, 109.35, 104.05, 97.74, 86.56, 74.35] }
    }
    pub fn d50() -> Self {
        Self { bands: [25.83, 52.93, 86.68, 100.00, 100.76, 97.74, 84.34, 70.06] }
    }
    pub fn a() -> Self {
        Self { bands: [9.80, 17.68, 29.49, 45.78, 66.29, 90.01, 115.92, 142.08] }
    }
    pub fn f11() -> Self {
        Self { bands: [3.00, 15.00, 60.00, 40.00, 80.00, 120.00, 55.00, 15.00] }
    }
}

pub fn spectral_to_xyz(spd: &SpectralBands, illuminant: &Illuminant) -> [f32; 3] {
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut z = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    let mut norm_z = 0.0f32;

    for i in 0..8 {
        let power = spd.0[i] * illuminant.bands[i];
        x += power * CIE_X[i] * BAND_SPACING;
        y += power * CIE_Y[i] * BAND_SPACING;
        z += power * CIE_Z[i] * BAND_SPACING;
        norm_x += illuminant.bands[i] * CIE_X[i] * BAND_SPACING;
        norm_y += illuminant.bands[i] * CIE_Y[i] * BAND_SPACING;
        norm_z += illuminant.bands[i] * CIE_Z[i] * BAND_SPACING;
    }

    if norm_y > 0.0 {
        let nx = if norm_x > 0.0 { norm_x } else { norm_y };
        let nz = if norm_z > 0.0 { norm_z } else { norm_y };
        // Normalize each channel by illuminant's own XYZ integral, then
        // scale to the CIE D65 white point so the sRGB matrix maps white to [1,1,1].
        let xn = (x / nx) * 0.9505;
        let yn = y / norm_y;
        let zn = (z / nz) * 1.0888;
        [xn, yn, zn]
    } else {
        [0.0, 0.0, 0.0]
    }
}

pub fn xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz;
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

pub fn linear_to_srgb_gamma(c: f32) -> f32 {
    if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}
