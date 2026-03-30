// Morph target application compute shader.
// morph → skin → EWA (pipeline order per spec).

struct GaussianSplat {
    position: vec3<f32>,
    scale: vec3<f32>,
    rotation: vec4<i32>,  // i16 quaternion stored as i32 for alignment
    opacity: u32,
    spectral: array<u32, 4>,  // 8 x u16 packed as 4 x u32 (pairs)
}

struct PackedSplatDelta {
    splat_index: u32,
    d_position: vec3<f32>,
    d_scale: vec3<f32>,
    _pad0: f32,
    d_spectral: array<u32, 4>,  // 8 x f16 packed as 4 x u32
    _pad1: vec4<u32>,
}

struct TargetOffsets {
    start: u32,
    end: u32,
    _pad: vec2<u32>,
}

struct MorphWeights {
    weights: array<f32, 16>,
    target_count: u32,
    splat_count: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read>       base_splats:    array<GaussianSplat>;
@group(0) @binding(1) var<storage, read>       delta_buffer:   array<PackedSplatDelta>;
@group(0) @binding(2) var<uniform>             morph_params:   MorphWeights;
@group(0) @binding(3) var<storage, read>       target_offsets: array<TargetOffsets>;
@group(0) @binding(4) var<storage, read_write> output_splats:  array<GaussianSplat>;

fn unpack_f16_pair(packed: u32) -> vec2<f32> {
    let lo = f32(u32(packed & 0xFFFFu));
    let hi = f32(u32(packed >> 16u));
    // Simple unpack: treat as f16 via bit manipulation
    // For simplicity, normalize to [-1, 1] range from u16 value
    return vec2((lo / 32768.0) - 1.0, (hi / 32768.0) - 1.0);
}

@compute @workgroup_size(256)
fn apply_morph_targets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_id = gid.x;
    if splat_id >= morph_params.splat_count { return; }

    var result = base_splats[splat_id];

    for (var t = 0u; t < morph_params.target_count; t++) {
        let w = morph_params.weights[t];
        if w < 0.0001 { continue; }

        let start = target_offsets[t].start;
        let end = target_offsets[t].end;

        for (var d = start; d < end; d++) {
            let delta = delta_buffer[d];
            if delta.splat_index != splat_id { continue; }

            result.position += delta.d_position * w;
            result.scale += delta.d_scale * w;

            // Apply spectral deltas (4 pairs of f16)
            for (var p = 0u; p < 4u; p++) {
                let packed_delta = delta.d_spectral[p];
                let packed_base = result.spectral[p];

                let lo_d = f32(i32(packed_delta & 0xFFFFu) - 32768) / 32768.0;
                let hi_d = f32(i32(packed_delta >> 16u) - 32768) / 32768.0;
                let lo_b = f32(packed_base & 0xFFFFu) / 65535.0;
                let hi_b = f32(packed_base >> 16u) / 65535.0;

                let lo_new = clamp(lo_b + lo_d * w, 0.0, 1.0);
                let hi_new = clamp(hi_b + hi_d * w, 0.0, 1.0);

                result.spectral[p] = (u32(lo_new * 65535.0) & 0xFFFFu) | (u32(hi_new * 65535.0) << 16u);
            }

            break; // Each splat appears at most once per target
        }
    }

    output_splats[splat_id] = result;
}
