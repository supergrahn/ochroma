// Many-light reservoir sampler — GPU compute mirror of the CPU oracle in
// `many_light.rs` (`LightSampler::sample` / `reservoir`, the `full` brute-force
// path over ALL lights). ONE THREAD PER SHADE POINT. Bit-equivalence target:
// the CPU `f32` reservoir math reproduced here with identical operation order,
// driven by a bit-exact emulation of the CPU u64 LCG.
//
// ============================  u64 LCG EMULATION  ============================
// WGSL has no u64, so the CPU `Lcg { state: u64 }` is emulated as two u32
// limbs `(lo, hi)` (little-endian: state = hi * 2^32 + lo). The CPU does, per
// step, `state = state * C + A` (wrapping u64) where
//
//   C = 6364136223846793005 = 0x5851F42D_4C957F2D
//   A = 1442695040888963407 = 0x14057B7E_F767814F
//
// and seeds with `Lcg::new(seed)`: state = seed * C + A (one mul-add). The
// reservoir then draws `next_unit() = (state >> 40) / 2^24`, i.e. mix once
// more THEN read the top 24 bits of the high limb (`hi >> 8`). All of this is
// reproduced bit-for-bit below; see the LCG unit test in the host module which
// pins a buffer of raw draws against the CPU `Lcg`.
//
// `sample_n` derives a per-reservoir sub-seed
//   sub_seed = rng_seed * 0x9E3779B9_7F4A7C15 + k
// (wrapping u64), which we emulate with the same mul-add limbs (different
// constant). One GPU thread reproduces one CPU `sample(shade_point, sub_seed)`.
// =============================================================================

struct Light {
    // position.xyz + radius in w
    pos_radius: vec4<f32>,
    // color.rgb + intensity in w
    color_intensity: vec4<f32>,
};

struct ShadePoint {
    // xyz shade point; w = seed_lo (low 32 bits of the per-thread u64 seed)
    point_seedlo: vec4<f32>,
};

struct Params {
    light_count: u32,
    point_count: u32,
    draw_count: u32, // LCG-test: raw draws per seed
    _pad1: u32,
};

struct LightSampleOut {
    light_index: u32,    // chosen index, or 0xFFFFFFFF if none chosen
    weight: f32,         // RIS weight W = (1/p̂_s) · (Σp̂ / M)
    chosen_target: f32,  // p̂_s of the chosen light (`target` is WGSL-reserved)
    m: u32,              // candidate count M
};

@group(0) @binding(0) var<storage, read> lights: array<Light>;
@group(0) @binding(1) var<storage, read> points: array<ShadePoint>;
// Per-thread full u64 seed, two limbs each (lo, hi). points[i].w carries a
// redundant lo for debugging; the authoritative seed comes from this buffer.
@group(0) @binding(2) var<storage, read> seeds: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> out_samples: array<LightSampleOut>;
// LCG-test only: raw f32 draws, draw_count per seed, written contiguously.
@group(0) @binding(5) var<storage, read_write> out_draws: array<f32>;

// --- u64 (lo, hi) arithmetic --------------------------------------------------

// 32x32 -> 64 unsigned multiply via 16-bit halves (no u32 overflow). Returns
// (lo, hi) limbs of the 64-bit product.
fn mul_u32_wide(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;          // < 2^32
    let lh = a_lo * b_hi;          // < 2^32
    let hl = a_hi * b_lo;          // < 2^32
    let hh = a_hi * b_hi;          // < 2^32

    // Accumulate the middle terms into bits [16..48). Track carry into hi.
    // cross = lh + hl, may overflow 32 bits.
    let cross = lh + hl;
    let cross_carry = select(0u, 1u, cross < lh); // carry out of lh+hl (bit 32)

    // low 64 = ll + (cross << 16) + (hh << 32)
    let cross_lo = cross << 16u;          // contributes to low limb
    let cross_hi = (cross >> 16u) | (cross_carry << 16u); // contributes to hi limb

    let lo = ll + cross_lo;
    let lo_carry = select(0u, 1u, lo < ll);

    let hi = hh + cross_hi + lo_carry;
    return vec2<u32>(lo, hi);
}

// 64-bit add: (a) + (b), returning low 64 limbs (wrapping, mod 2^64).
fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}

// state * mul + inc  (mod 2^64), all 64-bit limbs. mul/inc passed as limbs.
fn lcg_muladd(state: vec2<u32>, mul: vec2<u32>, inc: vec2<u32>) -> vec2<u32> {
    // 64x64 -> low 64 of product:
    //   lo limb  = (state.lo * mul.lo) mod 2^32, plus carries
    //   hi limb  = state.lo*mul.hi + state.hi*mul.lo + high(state.lo*mul.lo)
    let ll = mul_u32_wide(state.x, mul.x); // full 64-bit low*low product
    // cross products only need their low 32 bits (they land in the hi limb).
    let lo_hi = state.x * mul.y; // (state.lo * mul.hi) mod 2^32
    let hi_lo = state.y * mul.x; // (state.hi * mul.lo) mod 2^32

    let prod_lo = ll.x;
    let prod_hi = ll.y + lo_hi + hi_lo; // mod 2^32 (wrapping)

    return add_u64(vec2<u32>(prod_lo, prod_hi), inc);
}

// The CPU C and A constants, as limbs.
const C_LIMBS = vec2<u32>(0x4C957F2Du, 0x5851F42Du); // 6364136223846793005
const A_LIMBS = vec2<u32>(0xF767814Fu, 0x14057B7Eu); // 1442695040888963407

// Advance one LCG step and return the next f32 in [0,1): (state >> 40) / 2^24.
struct Rng { state: vec2<u32> };

fn rng_new(seed: vec2<u32>) -> Rng {
    // Lcg::new — one mul-add seed mix.
    var r: Rng;
    r.state = lcg_muladd(seed, C_LIMBS, A_LIMBS);
    return r;
}

fn rng_next_unit(r: ptr<function, Rng>) -> f32 {
    (*r).state = lcg_muladd((*r).state, C_LIMBS, A_LIMBS);
    // state >> 40  ==  top 24 bits of the high limb  ==  hi >> 8.
    let top24 = (*r).state.y >> 8u;
    return f32(top24) / 16777216.0; // / 2^24
}

// --- target function (mirrors rgb_target / attenuation / luminance) -----------

fn luminance(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

fn attenuation(distance: f32, intensity: f32, radius: f32) -> f32 {
    if (distance >= radius) {
        return 0.0;
    }
    let d = distance / radius;
    return intensity * max(1.0 - d * d, 0.0);
}

fn rgb_target(li: Light, shade_point: vec3<f32>) -> f32 {
    let pos = li.pos_radius.xyz;
    let radius = li.pos_radius.w;
    let color = li.color_intensity.xyz;
    let intensity = li.color_intensity.w;
    let dist = distance(pos, shade_point);
    let att = attenuation(dist, intensity, radius);
    return att * luminance(color);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.point_count) {
        return;
    }

    let shade_point = points[idx].point_seedlo.xyz;
    var rng = rng_new(seeds[idx]);

    // EXACT mirror of `LightSampler::reservoir` over the full light range.
    var w_sum: f32 = 0.0;
    var m: u32 = 0u;
    var chosen: u32 = 0xFFFFFFFFu;
    var chosen_target: f32 = 0.0;

    let n = params.light_count;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let p_hat = rgb_target(lights[i], shade_point);
        m = m + 1u;
        if (p_hat <= 0.0) {
            continue;
        }
        w_sum = w_sum + p_hat;
        // Keep candidate i with probability p̂_i / w_sum.
        if (rng_next_unit(&rng) * w_sum <= p_hat) {
            chosen = i;
            chosen_target = p_hat;
        }
    }

    var result: LightSampleOut;
    if (chosen == 0xFFFFFFFFu) {
        result.light_index = 0xFFFFFFFFu;
        result.weight = 0.0;
        result.chosen_target = 0.0;
        result.m = m;
    } else {
        let weight = (1.0 / chosen_target) * (w_sum / f32(m));
        result.light_index = chosen;
        result.weight = weight;
        result.chosen_target = chosen_target;
        result.m = m;
    }
    out_samples[idx] = result;
}

// LCG bit-exactness test: one thread per seed, emit `draw_count` consecutive
// `next_unit()` draws to out_draws[seed*draw_count + k]. Pins the u64 LCG
// emulation against the CPU `Lcg` BEFORE any reservoir logic is trusted.
@compute @workgroup_size(64)
fn lcg_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.point_count) {
        return;
    }
    var rng = rng_new(seeds[idx]);
    let dc = params.draw_count;
    let base = idx * dc;
    for (var k: u32 = 0u; k < dc; k = k + 1u) {
        out_draws[base + k] = rng_next_unit(&rng);
    }
}
