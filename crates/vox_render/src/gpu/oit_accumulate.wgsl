// OIT accumulation: for each transparent splat touching a pixel,
// accumulate power moments of depth distribution.
// b_0 = sum(alpha_i)
// b_1 = sum(alpha_i * z_i)
// b_2 = sum(alpha_i * z_i^2)
// b_3 = sum(alpha_i * z_i^3)
//
// Dispatch: (ceil(transparent_count / 256), 1, 1)

struct GpuSplatFull {
    position_depth: vec4<f32>,
    conic: vec3<f32>,
    _pad0: f32,
    opacity_color: vec4<f32>,
    spectral: array<f32, 8>,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
}

struct OitParams {
    width: u32,
    height: u32,
    transparent_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform>            camera:          CameraUniform;
@group(0) @binding(1) var<storage, read_only> splats:          array<GpuSplatFull>;
@group(0) @binding(2) var<storage, read_only> sorted_vals:     array<u32>;
@group(0) @binding(3) var                     moments_tex:     texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var                     transmit_tex:    texture_storage_2d<r32float,    read_write>;
@group(0) @binding(5) var<uniform>            oit_params:      OitParams;

@compute @workgroup_size(256)
fn oit_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_idx = gid.x;
    if splat_idx >= oit_params.transparent_count { return; }

    let real_idx = sorted_vals[splat_idx];
    let splat    = splats[real_idx];

    let clip = camera.view_proj * splat.position_depth;
    if clip.w <= 0.0 { return; }
    let ndc    = clip.xy / clip.w;
    let screen = (ndc * 0.5 + vec2(0.5)) * vec2(f32(oit_params.width), f32(oit_params.height));

    let z       = clip.w;   // view-space depth
    let opacity = splat.opacity_color.w;

    // Affect pixels in 3-sigma radius (simplified: just the centre pixel).
    let px = u32(screen.x);
    let py = u32(screen.y);
    if px >= oit_params.width || py >= oit_params.height { return; }

    let coord = vec2<i32>(i32(px), i32(py));

    let z2 = z * z;
    let z3 = z2 * z;

    // Additive accumulation of moments (b0, b1, b2, b3).
    let prev_m = textureLoad(moments_tex, coord);
    let delta_m = vec4(opacity, opacity * z, opacity * z2, opacity * z3);
    textureStore(moments_tex, coord, prev_m + delta_m);

    // Additive accumulation of total transmittance.
    let prev_t = textureLoad(transmit_tex, coord).r;
    textureStore(transmit_tex, coord, vec4(prev_t + opacity, 0.0, 0.0, 0.0));
}
