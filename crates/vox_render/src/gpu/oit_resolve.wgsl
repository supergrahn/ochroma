// OIT resolve: reconstruct transmittance from moments and composite over opaque.
// Uses b_0 (total alpha) as the coverage weight — simple single-layer approximation.
//
// Dispatch: (ceil(width*height / 256), 1, 1)

struct OitParams {
    width: u32,
    height: u32,
    transparent_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var opaque_tex:       texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var moments_tex:      texture_storage_2d<rgba32float, read>;
@group(0) @binding(2) var transmit_tex:     texture_storage_2d<r32float,    read>;
@group(0) @binding(3) var output_tex:       texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> oit_params: OitParams;

@compute @workgroup_size(256)
fn oit_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    if pixel_idx >= oit_params.width * oit_params.height { return; }

    let px = pixel_idx % oit_params.width;
    let py = pixel_idx / oit_params.width;
    let coord = vec2<i32>(i32(px), i32(py));

    // b_0 is total accumulated alpha (first moment, stored in .x).
    let total_coverage = min(textureLoad(moments_tex, coord).x, 0.99);
    let transmittance  = 1.0 - total_coverage;

    // Composite: attenuate opaque layer by transparent coverage.
    let opaque = textureLoad(opaque_tex, coord);
    textureStore(output_tex, coord, opaque * transmittance);
}
