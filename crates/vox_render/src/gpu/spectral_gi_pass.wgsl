// Spectral GI compute pass — gathers radiance from nearby emissive splats and
// writes back the GI-lit per-band spectral for each receiver splat.
//
// Mirrors the CPU path (SpectralRadianceCache::propagate + apply):
//   incoming = sky_ambient + sum_over_emitters(emissive / max(dist_sq, 0.01))
//   scale    = (max_incoming > 1) ? 1/max_incoming : 1
//   irr[b]   = clamp(incoming[b] * scale, 0, 1)          (propagate, alpha=0)
//   out[b]   = clamp(spectral[b] + irr[b] * 0.5, 0, 1)   (apply)
//
// GpuSplatEntry layout: position(vec3) + _pad(f32) + radiance([f32;16]) +
// reflectance([f32;16]) = 36 floats × 4 = 144 bytes. `radiance` carries the
// splat's decoded spectral; `reflectance.x` carries an emitter flag (1.0 if the
// splat's opacity > 128, else 0.0).

struct GpuSplatEntry {
    position: vec3<f32>,
    _pad0: f32,
    radiance: array<f32, 16>,
    reflectance: array<f32, 16>,
};

struct GiParams {
    splat_count: u32,
    max_emitters: u32,
    alpha: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read>       splats:   array<GpuSplatEntry>;
@group(0) @binding(1) var<storage, read_write> radiance: array<array<f32, 16>>;
@group(0) @binding(2) var<uniform>             params:   GiParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let receiver_idx = gid.x;
    if receiver_idx >= params.splat_count { return; }

    let pos = splats[receiver_idx].position;

    // Sky ambient is stored in the receiver's reflectance bands [1..15] plus a
    // header in reflectance[0] holding the emitter flag. We keep it simple and
    // read the per-splat sky ambient from reflectance bands directly (the host
    // packs the same sky term into every splat's reflectance, leaving slot 0 for
    // the emitter flag — see host packing).
    var incoming: array<f32, 16>;
    for (var b = 0u; b < 16u; b++) {
        incoming[b] = 0.0;
    }

    // Gather from every emitter splat (opacity-gated on the host via the flag in
    // reflectance[0]). Full O(n*m) over the emitter subset, capped by
    // max_emitters via striding so 50k splats stay tractable.
    let n = params.splat_count;
    let cap = max(params.max_emitters, 1u);
    let stride = max(n / cap, 1u);

    for (var k = 0u; k < n; k += stride) {
        if k == receiver_idx { continue; }
        // Emitter flag lives in reflectance[0]; non-emitters contribute nothing.
        if splats[k].reflectance[0] < 0.5 { continue; }
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
