// SDF Soft Shadow Compute Shader
//
// For each screen pixel, reconstructs the world-space surface point from the
// depth buffer, then ray-marches the terrain SDF toward the light direction.
// Uses the Quilez soft shadow formula: penumbra = min(h/t) across all steps,
// which gives penumbra proportional to distance from the occluder.
//
// Output: shadow_mask texture — 0.0=fully shadowed, 1.0=fully lit.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
};

struct SdfUniform {
    // World-space position of voxel (0,0,0)
    origin: vec3<f32>,
    _pad0: f32,
    // voxel_size in world units
    voxel_size: f32,
    // Number of voxels in each dimension
    size_x: u32,
    size_y: u32,
    size_z: u32,
    // Light direction (world space, pointing toward light)
    light_dir: vec3<f32>,
    // Controls penumbra width: higher = sharper, lower = softer
    penumbra_k: f32,
    // Maximum ray march distance (metres)
    max_dist: f32,
    _pad1: vec3<f32>,
};

@group(0) @binding(0) var<uniform>          camera:      CameraUniform;
@group(0) @binding(1) var<uniform>          sdf_params:  SdfUniform;
@group(0) @binding(2) var<storage, read>    sdf_data:    array<f32>;   // flat SDF
@group(0) @binding(3) var                   depth_tex:   texture_depth_2d;
@group(0) @binding(4) var                   depth_samp:  sampler;
@group(0) @binding(5) var<storage, read_write> shadow_out: array<f32>; // one per pixel

// Trilinear sample from flat SDF buffer.
fn sample_sdf(world_pos: vec3<f32>) -> f32 {
    let local = (world_pos - sdf_params.origin) / sdf_params.voxel_size;
    let ix = i32(local.x);
    let iy = i32(local.y);
    let iz = i32(local.z);
    let sx = i32(sdf_params.size_x);
    let sy = i32(sdf_params.size_y);
    let sz = i32(sdf_params.size_z);
    if ix < 0 || iy < 0 || iz < 0 || ix >= sx - 1 || iy >= sy - 1 || iz >= sz - 1 {
        return 1.0; // outside volume = air
    }
    let fx = fract(local.x);
    let fy = fract(local.y);
    let fz = fract(local.z);
    let stride_x = 1;
    let stride_y = sx;
    let stride_z = sx * sy;
    let base = iz * stride_z + iy * stride_y + ix;
    let v000 = sdf_data[base];
    let v100 = sdf_data[base + stride_x];
    let v010 = sdf_data[base + stride_y];
    let v110 = sdf_data[base + stride_y + stride_x];
    let v001 = sdf_data[base + stride_z];
    let v101 = sdf_data[base + stride_z + stride_x];
    let v011 = sdf_data[base + stride_z + stride_y];
    let v111 = sdf_data[base + stride_z + stride_y + stride_x];
    let c00 = mix(v000, v100, fx);
    let c10 = mix(v010, v110, fx);
    let c01 = mix(v001, v101, fx);
    let c11 = mix(v011, v111, fx);
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    return mix(c0, c1, fz);
}

// Quilez soft shadow: https://iquilezles.org/articles/rmshadows/
// Returns shadow in [0,1]. k controls penumbra sharpness.
fn soft_shadow(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> f32 {
    var result = 1.0;
    var t = 0.05; // start slightly off surface to avoid self-intersection
    let max_t = sdf_params.max_dist;
    let k = sdf_params.penumbra_k;
    for (var i = 0; i < 64; i++) {
        if t > max_t { break; }
        let h = sample_sdf(ray_origin + ray_dir * t);
        if h < 0.001 {
            return 0.0; // fully occluded
        }
        result = min(result, k * h / t);
        t += clamp(h, 0.01, 0.5);
    }
    return clamp(result, 0.0, 1.0);
}

// Reconstruct world-space position from depth buffer.
fn depth_to_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let cam_pos = camera.inv_view[3].xyz;
    let view_dir_unnorm = vec3<f32>(ndc.x, -ndc.y, -1.0);
    let world_dir = normalize((camera.inv_view * vec4<f32>(view_dir_unnorm, 0.0)).xyz);
    return cam_pos + world_dir * (depth * 200.0);
}

@compute @workgroup_size(8, 8)
fn cs_sdf_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let vw = u32(camera.viewport_size.x);
    let vh = u32(camera.viewport_size.y);
    if px >= vw || py >= vh { return; }

    let uv = (vec2<f32>(f32(px), f32(py)) + 0.5) / vec2<f32>(f32(vw), f32(vh));
    let depth = textureSampleLevel(depth_tex, depth_samp, uv, 0.0);

    if depth >= 0.9999 {
        shadow_out[py * vw + px] = 1.0;
        return;
    }

    let world_pos = depth_to_world(uv, depth);
    let shadow = soft_shadow(world_pos, normalize(sdf_params.light_dir));
    shadow_out[py * vw + px] = shadow;
}
