//! Deterministic hashing/variation helpers (from render_gpu::buildings).

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
