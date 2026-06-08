// GI → raster residency fold: writes the GI radiance buffer's first 8 bands
// DIRECTLY into the tiled renderer's persistent splat buffer, on-device, with
// NO CPU readback between the GI compute pass and the rasterizer. ONE THREAD
// PER SPLAT.
//
// The GI producer (`GpuGiPass.radiance_buffer`) is `array<array<f32,16>>` — 16
// f32 of GI-lit per-band radiance per splat. The raster consumer
// (`TiledSplatRenderer.splat_buf`) is `array<GpuSplatFull>` (80 bytes), whose
// `spectral: array<f32,8>` field at byte offset 48 carries the 8 GPU spectral
// bands the raster shader reads. This kernel folds `radiance[i][0..8]` into
// `splats[i].spectral[0..8]`, dropping bands 8..15 — exactly the first-8-bands
// rule the host-side `gaussian_splat_to_gpu_full` uses when it packs splats.
//
// ============================  CORRECTNESS  ================================
// The readback oracle (`GpuGi::step` → `gaussian_splat_to_gpu_full`) round-trips
// each band through f16: `half::f16::from_f32(v).to_bits()` then
// `half::f16::from_bits(bits).to_f32()` (round-to-nearest, ties-to-even). To be
// BIT-IDENTICAL to that oracle we must reproduce the SAME f16 quantization on
// device. The WGSL builtin `pack2x16float` on RADV TRUNCATES toward zero (NOT
// round-to-even — measured in Spec 02), so it diverges by one f16 ULP on ~half
// of values. We therefore COPY the validated `f32_to_f16_rne` (verbatim from
// `relight_gpu.wgsl`) which does the IEEE-754 round-to-nearest-even conversion
// in integer bit ops, then `unpack2x16float` the resulting f16 bits back to f32
// (an EXACT widening) — giving the identical f16 round-trip value the oracle
// stores.
// =============================================================================

// Mirror of host `GpuSplatFull` (splat_buffer.rs), 80 bytes, std430-tight:
//   [0..16]  position_depth: vec4<f32>
//   [16..28] conic:          vec3<f32>  (12 bytes, 16-aligned)
//   [28..32] _pad0:          f32
//   [32..48] opacity_color:  vec4<f32>
//   [48..80] spectral:       array<f32, 8>
struct GpuSplatFull {
    position_depth: vec4<f32>,
    conic: vec3<f32>,
    _pad0: f32,
    opacity_color: vec4<f32>,
    spectral: array<f32, 8>,
};

struct CombineParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       radiance: array<array<f32, 16>>;
@group(0) @binding(1) var<storage, read_write> splats:   array<GpuSplatFull>;
@group(0) @binding(2) var<uniform>             params:   CombineParams;

// f32 -> f16 bits with ROUND-TO-NEAREST, TIES-TO-EVEN — bit-identical to the
// CPU `half::f16::from_f32`. COPIED VERBATIM from `relight_gpu.wgsl` (validated
// bit-exact against the `half` crate in Spec 02). `pack2x16float` on RADV
// truncates toward zero, so we do the conversion explicitly in integer bit ops.
fn f32_to_f16_rne(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    let sign = (bits >> 16u) & 0x8000u;
    let exp = (bits >> 23u) & 0xFFu;        // 8-bit f32 exponent
    let mant = bits & 0x7FFFFFu;            // 23-bit f32 mantissa

    // NaN -> a quiet NaN (not reached: inputs are clamped to [0,1]).
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

// Reproduce the oracle's f16 round-trip EXACTLY: f32 -> f16 bits (RNE) -> f32.
// `unpack2x16float` widens the low 16 bits (an f16) to f32 exactly, so the
// result equals `half::f16::from_bits(half::f16::from_f32(x).to_bits()).to_f32()`.
fn quantize_f16(x: f32) -> f32 {
    let bits = f32_to_f16_rne(x) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    // Fold the GI radiance's first 8 bands into the splat's spectral, f16-
    // quantized to match the readback oracle. Bands 8..15 of radiance are
    // dropped (the first-8-bands rule of `gaussian_splat_to_gpu_full`).
    for (var b = 0u; b < 8u; b = b + 1u) {
        splats[i].spectral[b] = quantize_f16(radiance[i][b]);
    }
}
