use crate::spectral_framebuffer::SpectralFramebuffer;
use vox_core::spectral::{spectral_to_xyz, xyz_to_srgb, linear_to_srgb_gamma, SpectralBands, Illuminant};

/// Tone mapping operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapOperator {
    None,    // Linear clamp
    Reinhard, // x / (1 + x)
    ACES,    // Academy Color Encoding System
    Filmic,  // Uncharted 2 filmic curve
}

/// Settings for the tone mapper.
#[derive(Debug, Clone)]
pub struct ToneMapSettings {
    pub operator: ToneMapOperator,
    pub exposure: f32,     // EV adjustment (1.0 = neutral)
    pub white_point: f32,  // Brightest value that maps to white
    pub gamma: f32,        // Display gamma (2.2 for sRGB)
    pub saturation: f32,   // 0 = greyscale, 1 = normal, >1 = boosted
}

impl Default for ToneMapSettings {
    fn default() -> Self {
        Self {
            operator: ToneMapOperator::ACES,
            exposure: 1.0,
            white_point: 1.0,
            gamma: 2.2,
            saturation: 1.0,
        }
    }
}

/// ACES tone mapping curve.
fn aces_tonemap(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// Reinhard tone mapping.
fn reinhard_tonemap(x: f32, white: f32) -> f32 {
    (x * (1.0 + x / (white * white))) / (1.0 + x)
}

/// Filmic tone mapping (Uncharted 2).
fn filmic_tonemap(x: f32) -> f32 {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
}

/// Apply tone mapping to a single RGB value.
fn apply_tonemap(r: f32, g: f32, b: f32, settings: &ToneMapSettings) -> [f32; 3] {
    // Exposure
    let r = r * settings.exposure;
    let g = g * settings.exposure;
    let b = b * settings.exposure;

    // Saturation
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let r = luma + (r - luma) * settings.saturation;
    let g = luma + (g - luma) * settings.saturation;
    let b = luma + (b - luma) * settings.saturation;

    // Tone map
    let (r, g, b) = match settings.operator {
        ToneMapOperator::None => (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)),
        ToneMapOperator::Reinhard => (
            reinhard_tonemap(r, settings.white_point),
            reinhard_tonemap(g, settings.white_point),
            reinhard_tonemap(b, settings.white_point),
        ),
        ToneMapOperator::ACES => (aces_tonemap(r), aces_tonemap(g), aces_tonemap(b)),
        ToneMapOperator::Filmic => {
            let w = filmic_tonemap(11.2); // white scale
            (
                filmic_tonemap(r) / w,
                filmic_tonemap(g) / w,
                filmic_tonemap(b) / w,
            )
        }
    };

    [r.max(0.0), g.max(0.0), b.max(0.0)]
}

/// Convert a spectral framebuffer to displayable RGBA8 pixels.
/// This is the final step before presenting to the screen.
pub fn tonemap_spectral_framebuffer(
    fb: &SpectralFramebuffer,
    illuminant: &Illuminant,
    settings: &ToneMapSettings,
) -> Vec<[u8; 4]> {
    let mut output = Vec::with_capacity(fb.pixel_count());

    for i in 0..fb.pixel_count() {
        let spectral = SpectralBands(fb.spectral[i]);

        // Spectral -> XYZ -> linear RGB
        let xyz = spectral_to_xyz(&spectral, illuminant);
        let linear_rgb = xyz_to_srgb(xyz);

        // Tone map
        let mapped = apply_tonemap(linear_rgb[0], linear_rgb[1], linear_rgb[2], settings);

        // Gamma correction
        let r = linear_to_srgb_gamma(mapped[0]);
        let g = linear_to_srgb_gamma(mapped[1]);
        let b_val = linear_to_srgb_gamma(mapped[2]);

        output.push([
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b_val.clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]);
    }

    output
}

/// Convert spectral framebuffer to HDR f32 RGB (for DLSS input).
pub fn tonemap_spectral_to_hdr(
    fb: &SpectralFramebuffer,
    illuminant: &Illuminant,
    settings: &ToneMapSettings,
) -> Vec<[f32; 4]> {
    let mut output = Vec::with_capacity(fb.pixel_count());

    for i in 0..fb.pixel_count() {
        let spectral = SpectralBands(fb.spectral[i]);
        let xyz = spectral_to_xyz(&spectral, illuminant);
        let linear_rgb = xyz_to_srgb(xyz);
        let mapped = apply_tonemap(linear_rgb[0], linear_rgb[1], linear_rgb[2], settings);
        output.push([mapped[0], mapped[1], mapped[2], 1.0]);
    }

    output
}
