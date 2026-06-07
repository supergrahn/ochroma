// Runtime spectral relight kernel — GPU compute mirror of the CPU oracle in
// `relight.rs` (`derive_intrinsic` + `reilluminate_one` + `encode_radiance`,
// driven by `relight_scene`'s ambient-only / no-shadow configuration). ONE
// THREAD PER SPLAT. Bit-equivalence target: the CPU `f32` band loop reproduced
// here with IDENTICAL operation order and no fast-math, so the GPU stored f16
// bit-matches the CPU `half::f16::from_f32` round-trip on RADV.
//
// =============================  THE OP CHAIN  ================================
// For each splat, for each of the 16 bands b (the EXACT op order of
// `reilluminate_one` with `n_dot_l = 1`, `shadow = 1`, `emitter_gather = 0`):
//
//   intrinsic[b] = baked[b] / max(ref_spd[b], floor)        // derive_intrinsic (UNCLAMPED)
//   direct_scale = max(1.0 * 1.0, 0.0)                       // = 1.0 (ambient-only)
//   incident[b]  = target_spd[b] * direct_scale + ambient[b]
//   out[b]       = clamp(intrinsic[b] * incident + 0.0, 0.0, f16_max)
//   stored[b]    = f16(nan? 0.0 : clamp(out[b], 0.0, f16_max))   // encode_radiance
//
// `baked[b]` is decoded HOST-side from the splat's f16 spectral field via the
// SAME `half` crate the oracle's `read_radiance` uses, and uploaded as f32 — so
// the GPU input is bit-identical to the CPU input. Only the OUTPUT f16 store is
// reproduced on-device (via `pack2x16float`), which is the single load-bearing
// quantization the `<1e-6` bound watches.
//
// `derive_intrinsic` is intentionally UN-clamped (relight.rs:357) — a bright
// band crossing illuminants can exceed 65504 and would encode to +inf without
// the clamp in reilluminate_one + encode_radiance (wave-12 critical finding).
// =============================================================================

struct RelightSplat {
    // 16 baked-radiance f32 bands (decoded host-side from the splat's f16
    // spectral field). Laid out as 4 vec4 so the WGSL struct is std430-tight
    // at 64 bytes and matches the host `GpuRelightSplat`.
    baked: array<vec4<f32>, 4>,
};

struct RelightParams {
    // target illuminant SPD (IlluminantSpec::spd of the target), 16 bands.
    target_spd: array<vec4<f32>, 4>,
    // 0.5-weighted sky ambient SPD (solar_irradiance * AMBIENT_FILL_WEIGHT), or
    // zeros if sky ambient is off. Bound pre-multiplied so the shader does one add.
    ambient: array<vec4<f32>, 4>,
    // reference illuminant SPD (IlluminantSpec::spd of the bake-time light).
    ref_spd: array<vec4<f32>, 4>,
    splat_count: u32,
    floor: f32,     // RelightSettings::floor (1e-3 default)
    f16_max: f32,   // half::f16::MAX.to_f32() — the clamp ceiling
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> splats: array<RelightSplat>;
@group(0) @binding(1) var<uniform> params: RelightParams;
// Output: 16 f16 bands packed two-per-u32 (8 u32 per splat). bits[2*p] holds
// bands (2p, 2p+1) as pack2x16float(vec2(out[2p], out[2p+1])).
@group(0) @binding(2) var<storage, read_write> out_bits: array<u32>;

// Read band b out of a 4×vec4 array (std430 16-float block).
fn band(a: array<vec4<f32>, 4>, b: u32) -> f32 {
    let v = a[b >> 2u];
    let lane = b & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// f32 -> f16 bits with ROUND-TO-NEAREST, TIES-TO-EVEN — bit-identical to the
// CPU `half::f16::from_f32` (relight.rs `encode_radiance`).
//
// SPEC-VS-REALITY (load-bearing): the WGSL builtin `pack2x16float` on RADV
// rounds f32->f16 toward ZERO (truncation), NOT round-to-nearest-even, so it
// produces f16 bits one ULP BELOW `half::f16::from_f32` on ~half of values
// (measured: a constant 4.88e-4 = one f16 ULP at [0.25,0.5) deviation). The
// verifier flagged exactly this assertion as the one most likely to need
// care. So we do the IEEE-754 round-to-nearest-even conversion explicitly in
// integer bit ops, which bit-matches `half` for every value. Inputs here are
// already clamped to [0, f16_max] (no inf/NaN, no overflow-to-inf), but the
// conversion handles the full normal/subnormal/round-to-even cases for safety.
fn f32_to_f16_rne(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    let sign = (bits >> 16u) & 0x8000u;
    let exp = (bits >> 23u) & 0xFFu;        // 8-bit f32 exponent
    let mant = bits & 0x7FFFFFu;            // 23-bit f32 mantissa

    // NaN -> a quiet NaN (not reached: encode_band maps NaN->0 first).
    if (exp == 0xFFu) {
        if (mant != 0u) {
            return sign | 0x7E00u; // qNaN
        }
        return sign | 0x7C00u;     // +/- inf
    }

    // Unbiased exponent. f32 bias 127, f16 bias 15.
    let e = i32(exp) - 127 + 15;

    if (e >= 0x1F) {
        // Overflow -> inf (not reached for clamped inputs).
        return sign | 0x7C00u;
    }

    if (e <= 0) {
        // Subnormal f16 or zero. Shift the implicit-1 mantissa right by (1 - e)
        // extra bits, with round-to-nearest-even on the discarded bits.
        if (e < -10) {
            // Too small even for the smallest subnormal -> +/- 0.
            return sign;
        }
        // Mantissa with implicit leading 1, in the 24-bit (1.23) form.
        let m = mant | 0x800000u;
        // Total right shift to land the value in the 10-bit f16 fraction:
        //   13 (23->10) + (1 - e).
        let shift = u32(14 - e); // = 13 + (1 - e)
        let half_bits = m >> shift;
        // Round-to-nearest-even on the bits shifted out.
        let remainder = m & ((1u << shift) - 1u);
        let halfway = 1u << (shift - 1u);
        var rounded = half_bits;
        if (remainder > halfway || (remainder == halfway && (half_bits & 1u) == 1u)) {
            rounded = rounded + 1u;
        }
        return sign | rounded;
    }

    // Normal f16. Take the top 10 mantissa bits and round-to-nearest-even on
    // the remaining 13. A round that carries into bit 10 increments the exponent
    // (handled automatically by adding into the combined exp|mant field).
    let half_exp = u32(e) << 10u;
    let half_mant = mant >> 13u;
    var combined = half_exp | half_mant;
    let remainder = mant & 0x1FFFu;       // low 13 bits discarded
    let halfway = 0x1000u;                // 1 << 12
    if (remainder > halfway || (remainder == halfway && (half_mant & 1u) == 1u)) {
        combined = combined + 1u;         // may carry mant->exp; bit-correct
    }
    return sign | combined;
}

// CPU `encode_radiance` for one band: NaN -> 0.0, else clamp to [0, f16_max],
// then the f16 round-trip (round-to-nearest-even).
fn encode_band(r: f32) -> u32 {
    var safe: f32;
    // `r != r` is the WGSL NaN test (no isnan builtin in the core profile).
    if (r != r) {
        safe = 0.0;
    } else {
        safe = clamp(r, 0.0, params.f16_max);
    }
    return f32_to_f16_rne(safe);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.splat_count) {
        return;
    }

    let s = splats[idx];
    let base = idx * 8u; // 8 u32 of packed f16 per splat (16 bands / 2)

    // Ambient-only / no-shadow config: n_dot_l = 1, shadow = 1, emitter = 0.
    let direct_scale = max(1.0 * 1.0, 0.0);

    for (var p: u32 = 0u; p < 8u; p = p + 1u) {
        let b0 = p * 2u;
        let b1 = b0 + 1u;

        // --- band b0 ---
        let baked0 = band(s.baked, b0);
        let ref0 = band(params.ref_spd, b0);
        let intrinsic0 = baked0 / max(ref0, params.floor);      // derive_intrinsic (unclamped)
        let incident0 = band(params.target_spd, b0) * direct_scale + band(params.ambient, b0);
        let out0 = clamp(intrinsic0 * incident0 + 0.0, 0.0, params.f16_max);

        // --- band b1 ---
        let baked1 = band(s.baked, b1);
        let ref1 = band(params.ref_spd, b1);
        let intrinsic1 = baked1 / max(ref1, params.floor);
        let incident1 = band(params.target_spd, b1) * direct_scale + band(params.ambient, b1);
        let out1 = clamp(intrinsic1 * incident1 + 0.0, 0.0, params.f16_max);

        // encode_radiance per band (NaN->0, clamp, f16), then pack the pair.
        let bits0 = encode_band(out0);
        let bits1 = encode_band(out1);
        out_bits[base + p] = bits0 | (bits1 << 16u);
    }
}
