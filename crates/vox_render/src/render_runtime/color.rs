//! Color/tint math helpers (from render_gpu::materials).

/// Relative luminance of a linear RGB triple (Rec. 709 weights).
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

/// Linear interpolation between `a` and `b` by `t`.
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// Lift each channel toward white by `amount`, clamped into the display range.
pub fn lift_rgb(rgb: [f32; 3], amount: f32) -> [f32; 3] {
    [
        mix(rgb[0], 1.0, amount).clamp(0.02, 1.0),
        mix(rgb[1], 1.0, amount).clamp(0.02, 1.0),
        mix(rgb[2], 1.0, amount).clamp(0.02, 1.0),
    ]
}

/// Scale each channel by `scale`, clamped into the display range.
pub fn scale_rgb(rgb: [f32; 3], scale: f32) -> [f32; 3] {
    [
        (rgb[0] * scale).clamp(0.02, 1.0),
        (rgb[1] * scale).clamp(0.02, 1.0),
        (rgb[2] * scale).clamp(0.02, 1.0),
    ]
}

/// Tint a base colour toward a texture's local luminance detail (from
/// `render_gpu::materials`). `strength` blends between the flat base and the
/// per-texel detail lift.
pub fn texture_luma_tint_rgb(
    base: [f32; 3],
    texture_rgb: [f32; 3],
    texture_average: [f32; 3],
    strength: f32,
) -> [f32; 3] {
    let detail = (luminance(texture_rgb) / luminance(texture_average).max(0.02)).clamp(0.68, 1.38);
    let lift = mix(1.0, detail, strength.clamp(0.0, 1.0));
    [
        (base[0] * lift).clamp(0.02, 1.0),
        (base[1] * lift).clamp(0.02, 1.0),
        (base[2] * lift).clamp(0.02, 1.0),
    ]
}

/// Glass display tint: blends `base` toward a glass tint and adds a sky-sheen
/// term (from `render_gpu::materials`).
pub fn glass_display_rgb(base: [f32; 3], roughness: f32, upwardness: f32) -> [f32; 3] {
    let tint = [0.14, 0.68, 0.82];
    let sheen = (1.0 - roughness).clamp(0.0, 1.0);
    let sky = (0.18 + upwardness * 0.10 + sheen * 0.18).clamp(0.0, 0.42);
    [
        mix(base[0], tint[0], 0.72) + sky,
        mix(base[1], tint[1], 0.72) + sky,
        mix(base[2], tint[2], 0.72) + sky,
    ]
}
