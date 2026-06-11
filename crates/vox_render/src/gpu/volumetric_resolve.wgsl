// Froxel volumetric resolve — applies the scattered froxel volume to the
// splat HDR target. Dispatch: (ceil(froxel_w/8), ceil(froxel_h/8), 1) —
// one thread per froxel COLUMN; each thread covers its column's pixel block.
//
// This entry point cannot live in scatter_compute.wgsl: that module's globals
// at bindings 1–3 (params uniform / froxels read-write / sdf read-only buffer)
// conflict with the resolve bind group layout below.
//
// Bindings mirror `resolve_bgl` in volumetric_pass.rs:
//   0: camera uniform (declared for layout parity; the march is froxel-space)
//   1: froxels        (storage, read) — written by scatter_compute
//   2: out_color      (rgba32float storage texture, WRITE-only: baseline wgpu
//                      forbids read-write rgba32float storage textures, so the
//                      pre-resolve colour arrives via binding 4, a copy that
//                      dispatch_resolve records before this pass)
//   3: depth          (depth texture — view-space limit of the march)
//   4: in_color       (rgba32float sampled copy of the pre-resolve target)
//   5: vol_params     (the pass's params uniform: froxel dims + z range)

// Isotropic phase function 1/(4π): the M1 single-scatter source term is
// albedo · phase · unit sun. Anisotropic Mie phase arrives with the Tier-2
// clipmap wave.
const PHASE_ISO: f32 = 0.07957747;

struct FroxelVoxel {
    scatter: array<f32, 8>,
    transmittance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct VolumetricParams {
    sun_direction: vec3<f32>,
    mie_coeff: f32,
    z_near: f32,
    z_far: f32,
    froxel_width: u32,
    froxel_height: u32,
    froxel_depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> froxels: array<FroxelVoxel>;
@group(0) @binding(2) var out_color: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var in_color: texture_2d<f32>;
@group(0) @binding(5) var<uniform> vol_params: VolumetricParams;

fn froxel_idx(x: u32, y: u32, z: u32) -> u32 {
    return x + vol_params.froxel_width * (y + vol_params.froxel_height * z);
}

@compute @workgroup_size(8, 8, 1)
fn volumetric_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fw = vol_params.froxel_width;
    let fh = vol_params.froxel_height;
    let fd = vol_params.froxel_depth;
    if gid.x >= fw || gid.y >= fh {
        return;
    }

    let tex_dims = textureDimensions(out_color);
    // Pixel block covered by this froxel column (proportional mapping).
    let px0 = (gid.x * tex_dims.x) / fw;
    let px1 = ((gid.x + 1u) * tex_dims.x) / fw;
    let py0 = (gid.y * tex_dims.y) / fh;
    let py1 = ((gid.y + 1u) * tex_dims.y) / fh;

    let zn = vol_params.z_near;
    let zf = vol_params.z_far;

    for (var py = py0; py < py1; py++) {
        for (var px = px0; px < px1; px++) {
            let coord = vec2<i32>(i32(px), i32(py));
            let d = textureLoad(depth_tex, coord, 0);

            // [0,1] depth -> view-space z (standard wgpu [0,1] projection),
            // then -> exponential slice limit (FroxelVolume::slice_z inverse).
            let view_z = zn * zf / max(zf - d * (zf - zn), 1e-6);
            let k_f = f32(fd) * log(max(view_z, zn) / zn) / log(zf / zn);
            let k_limit = min(u32(ceil(k_f)), fd);

            // Front-to-back march: T = product of voxel transmittances,
            // S = energy-conserving single-scatter with mean-band in-scatter.
            var t_total = 1.0;
            var s_total = 0.0;
            for (var k = 0u; k < k_limit; k++) {
                let v = froxels[froxel_idx(gid.x, gid.y, k)];
                var mean_scatter = 0.0;
                for (var b = 0u; b < 8u; b++) {
                    mean_scatter += v.scatter[b];
                }
                mean_scatter /= 8.0;
                let tr = clamp(v.transmittance, 1e-6, 1.0);
                // scatter_compute writes tr = exp(-extinction * 0.5).
                let extinction = -2.0 * log(tr);
                let albedo = clamp(mean_scatter / max(extinction, 1e-6), 0.0, 1.0);
                s_total += t_total * (1.0 - tr) * albedo * PHASE_ISO;
                t_total *= tr;
            }

            let rgba = textureLoad(in_color, coord, 0);
            textureStore(
                out_color,
                coord,
                vec4(rgba.rgb * t_total + vec3(s_total), rgba.a),
            );
        }
    }
}
