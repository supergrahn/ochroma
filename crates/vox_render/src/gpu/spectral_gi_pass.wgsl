// Spectral GI compute pass — gathers radiance from nearby emissive splats and
// writes back the GI-lit per-band spectral for each receiver splat.
//
// Mirrors the CPU path (SpectralRadianceCache::propagate + apply) exactly:
//   incoming = sky_ambient + sum_over_emitters(emissive / max(dist_sq, 0.01))
//   scale    = (max_incoming > 1) ? 1/max_incoming : 1
//   irr[b]   = clamp(incoming[b] * scale, 0, 1)          (propagate, fresh cache)
//   out[b]   = clamp(spectral[b] + irr[b] * 0.5, 0, 1)   (apply)
//
// Emitter selection mirrors the CPU's `splats.iter().filter(opacity>128)
// .take(max_emitters)`: the FIRST `max_emitters` emitter splats in buffer order
// contribute (a prefix, NOT a stride). The receiver is NOT skipped — if it is
// itself an emitter it contributes to its own incoming, exactly as the CPU
// emitter list (built independently of the receiver) does.
//
// GpuSplatEntry layout: position(vec3) + _pad(f32) + radiance([f32;16]) +
// reflectance([f32;16]) = 36 floats × 4 = 144 bytes. `radiance` carries the
// splat's decoded spectral; `reflectance.x` carries an emitter flag (1.0 if the
// splat's opacity > 128, else 0.0).
//
// Sky ambient is supplied once per dispatch in `params.sky_ambient` (the host
// computes it from the atmosphere's solar irradiance, mirroring
// `SpectralRadianceCache::set_sky`). It is NOT read from any splat's reflectance
// bands; reflectance holds only the emitter flag in slot 0.

struct GpuSplatEntry {
    position: vec3<f32>,
    _pad0: f32,
    radiance: array<f32, 16>,
    reflectance: array<f32, 16>,
};

struct GiParams {
    splat_count: u32,
    max_emitters: u32,
    _pad0: u32,
    _pad1: u32,
    // Per-band sky-ambient radiance (mirrors set_sky -> solar_irradiance).
    // vec4-aligned for std140 uniform layout: 16 bands = 4 × vec4.
    sky_ambient: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<storage, read>       splats:   array<GpuSplatEntry>;
@group(0) @binding(1) var<storage, read_write> radiance: array<array<f32, 16>>;
@group(0) @binding(2) var<uniform>             params:   GiParams;

fn sky_band(b: u32) -> f32 {
    let v = params.sky_ambient[b / 4u];
    let lane = b % 4u;
    if lane == 0u { return v.x; }
    if lane == 1u { return v.y; }
    if lane == 2u { return v.z; }
    return v.w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let receiver_idx = gid.x;
    if receiver_idx >= params.splat_count { return; }

    let pos = splats[receiver_idx].position;

    // incoming starts at the sky-ambient term (mirrors CPU `let mut incoming =
    // sky;` in SpectralRadianceCache::propagate).
    var incoming: array<f32, 16>;
    for (var b = 0u; b < 16u; b++) {
        incoming[b] = sky_band(b);
    }

    // Gather from the FIRST `max_emitters` emitter splats in buffer order,
    // bit-for-bit matching the CPU's `filter(opacity>128).take(max_emitters)`.
    // We scan splats in order, counting emitters, and stop once `max_emitters`
    // emitters have been accumulated. The receiver is NOT skipped.
    let n = params.splat_count;
    let cap = params.max_emitters;
    var emitters_seen = 0u;
    for (var k = 0u; k < n; k++) {
        if emitters_seen >= cap { break; }
        // Emitter flag lives in reflectance[0]; non-emitters are not counted.
        if splats[k].reflectance[0] < 0.5 { continue; }
        emitters_seen += 1u;

        let ep = splats[k].position;
        let dx = ep.x - pos.x;
        let dy = ep.y - pos.y;
        let dz = ep.z - pos.z;
        let dist_sq = max(dx * dx + dy * dy + dz * dz, 0.01);
        let weight = 1.0 / dist_sq;
        for (var b = 0u; b < 16u; b++) {
            incoming[b] += splats[k].radiance[b] * weight;
        }
    }

    // Normalize incoming (matches CPU propagate's max-based scale).
    var max_incoming = 1.1920929e-7; // f32::EPSILON
    for (var b = 0u; b < 16u; b++) {
        if incoming[b] > max_incoming { max_incoming = incoming[b]; }
    }
    var scale = 1.0;
    if max_incoming > 1.0 { scale = 1.0 / max_incoming; }

    // apply(): out = clamp(spectral + irr * 0.5, 0, 1). `radiance` holds the
    // receiver's own decoded spectral.
    for (var b = 0u; b < 16u; b++) {
        let irr = clamp(incoming[b] * scale, 0.0, 1.0);
        let spectral = splats[receiver_idx].radiance[b];
        radiance[receiver_idx][b] = clamp(spectral + irr * 0.5, 0.0, 1.0);
    }
}
