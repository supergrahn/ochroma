// Froxel scatter + transmittance computation.
// Dispatch: (ceil(w/8), ceil(h/8), ceil(d/4))

const RAYLEIGH_BASE: array<f32, 8> = array<f32, 8>(
    1.0000, 0.7211, 0.5258, 0.3868, 0.2878, 0.2160, 0.1634, 0.1250
);

struct FroxelVoxel {
    scatter: array<f32, 8>,
    transmittance: f32,
    _pad: vec3<f32>,
}

struct VolumetricParams {
    sun_direction: vec3<f32>,
    mie_coeff: f32,
    z_near: f32,
    z_far: f32,
    froxel_width: u32,
    froxel_height: u32,
    froxel_depth: u32,
    _pad: vec3<u32>,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
}

struct SdfParams {
    origin: vec3<f32>,
    cell_size: f32,
    dims: vec3<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> vol_params: VolumetricParams;
@group(0) @binding(2) var<storage, read_write> froxels: array<FroxelVoxel>;
@group(0) @binding(3) var<storage, read_only> sdf_volume: array<f32>;

fn froxel_idx(x: u32, y: u32, z: u32) -> u32 {
    return x + vol_params.froxel_width * (y + vol_params.froxel_height * z);
}

fn froxel_world_pos(x: u32, y: u32, z: u32) -> vec3<f32> {
    // Convert froxel (x,y,z) to world position via exponential Z distribution.
    let u = (f32(x) + 0.5) / f32(vol_params.froxel_width);
    let v = (f32(y) + 0.5) / f32(vol_params.froxel_height);
    let depth = vol_params.z_near * pow(vol_params.z_far / vol_params.z_near,
                                        f32(z) / f32(vol_params.froxel_depth));
    // NDC -> world via inv_view (camera-space reconstruction).
    let ndc = vec3(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0);
    let view_pos = vec3(ndc.x * depth, ndc.y * depth, -depth);
    let world_pos = (camera.inv_view * vec4(view_pos, 1.0)).xyz;
    return world_pos;
}

fn sample_sdf_at(p: vec3<f32>, sdf_origin: vec3<f32>, cell_size: f32, dims: vec3<u32>) -> f32 {
    let rel = (p - sdf_origin) / cell_size;
    let idx = vec3<i32>(rel);
    if any(idx < vec3(0, 0, 0)) || any(idx >= vec3<i32>(dims) - vec3(1, 1, 1)) {
        return 1.0;
    }
    let flat_idx = idx.x + idx.y * i32(dims.x) + idx.z * i32(dims.x * dims.y);
    return sdf_volume[flat_idx];
}

@compute @workgroup_size(8, 8, 4)
fn scatter_compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= vol_params.froxel_width || gid.y >= vol_params.froxel_height || gid.z >= vol_params.froxel_depth {
        return;
    }

    // Use a fixed SDF origin/cell_size/dims (caller uploads sdf_volume).
    // These defaults produce a "no obstacles" result when no SDF is bound.
    let sdf_origin = vec3<f32>(0.0, 0.0, 0.0);
    let cell_size = 1.0;
    let dims = vec3<u32>(1u, 1u, 1u);

    let world_pos = froxel_world_pos(gid.x, gid.y, gid.z);
    let sdf_val = sample_sdf_at(world_pos, sdf_origin, cell_size, dims);
    // Density is positive inside objects (fog / atmosphere model).
    let density = max(0.0, -sdf_val);

    let idx = froxel_idx(gid.x, gid.y, gid.z);
    var voxel: FroxelVoxel;

    // Rayleigh scattering per spectral band.
    var total_scatter = 0.0;
    for (var b = 0u; b < 8u; b++) {
        let scatter_b = RAYLEIGH_BASE[b] * (1.0 + density);
        voxel.scatter[b] = scatter_b + vol_params.mie_coeff;
        total_scatter += scatter_b;
    }

    // Transmittance: Beer-Lambert approximation over 1 froxel depth unit.
    let extinction = total_scatter / 8.0 + vol_params.mie_coeff;
    voxel.transmittance = exp(-extinction * 0.5);
    voxel._pad = vec3(0.0);

    froxels[idx] = voxel;
}
