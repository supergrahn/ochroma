// Spectral GI compute pass — gathers radiance from nearby emissive splats.
// GpuSplatEntry layout: position(vec3) + _pad(f32) + radiance([f32;16]) + reflectance([f32;16])
// = 4 + 1 + 16 + 16 = 37 floats, but padded to 36 → 144 bytes in Rust repr(C).

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
    var incoming: array<f32, 16>;

    let candidate_count = min(params.splat_count, params.max_emitters);
    let stride = max(params.splat_count / max(candidate_count, 1u), 1u);

    for (var k = 0u; k < candidate_count; k++) {
        let emitter_idx = k * stride;
        if emitter_idx == receiver_idx { continue; }
        let ep = splats[emitter_idx].position;
        let dx = ep.x - pos.x;
        let dy = ep.y - pos.y;
        let dz = ep.z - pos.z;
        let dist_sq = max(dx * dx + dy * dy + dz * dz, 0.01);
        let weight = 1.0 / dist_sq;
        for (var b = 0u; b < 16u; b++) {
            incoming[b] += splats[emitter_idx].radiance[b] * weight;
        }
    }

    var max_val = 0.00001;
    for (var b = 0u; b < 16u; b++) {
        if incoming[b] > max_val { max_val = incoming[b]; }
    }

    let alpha = params.alpha;
    for (var b = 0u; b < 16u; b++) {
        let new_val = clamp(incoming[b] / max_val, 0.0, 1.0);
        radiance[receiver_idx][b] = alpha * radiance[receiver_idx][b] + (1.0 - alpha) * new_val;
    }
}
