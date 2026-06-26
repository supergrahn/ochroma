//! Pure-math render-runtime helpers — deterministic hashing and RGB color utilities.
//!
//! Moved verbatim from the game's `render_gpu::buildings` / `render_gpu::materials`
//! so a new game never re-implements these. No game-type dependencies (f32/u64/[f32; 3]).

// ---------------------------------------------------------------------------
// Deterministic hash helpers (from render_gpu::buildings)
// ---------------------------------------------------------------------------

/// splitmix64 mix — the repo's standard inline deterministic hash (no `rand`).
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One independent 64-bit channel of an instance seed, keyed by `salt`.
pub fn variation_bits(seed: u64, salt: u64) -> u64 {
    splitmix64(seed ^ splitmix64(salt))
}

/// Deterministic uniform in `[0, 1)` for `(seed, salt)`.
pub fn variation_unit(seed: u64, salt: u64) -> f32 {
    (variation_bits(seed, salt) >> 40) as f32 / (1u64 << 24) as f32
}

/// Deterministic uniform in `[-1, 1)` for `(seed, salt)`.
pub fn variation_signed(seed: u64, salt: u64) -> f32 {
    variation_unit(seed, salt) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Colour utilities (from render_gpu::materials)
// ---------------------------------------------------------------------------

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
