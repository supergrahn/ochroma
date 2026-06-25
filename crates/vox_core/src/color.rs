//! Colour-space conversion utilities shared across engine crates.

/// Exact piecewise sRGB EOTF (encoded → linear) for one channel in [0, 1].
///
/// Matches the IEC 61966-2-1 standard ("sRGB color space"). Used wherever
/// a pixel stored as 8-bit sRGB (e.g. a PNG / JPEG) needs to be converted
/// to a linear radiance value before path-traced shading.
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear → sRGB (inverse EOTF) for one channel in [0, 1].
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_to_linear_roundtrip() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert_eq!(srgb_to_linear(1.0), 1.0);
        // Values in the linear segment.
        assert!((srgb_to_linear(0.04045) - 0.04045 / 12.92).abs() < 1e-7);
        // Mid-tone: well-known value.
        assert!((srgb_to_linear(0.5) - 0.214041).abs() < 1e-5);
        // Round-trip.
        let v = 0.73_f32;
        assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
    }
}
