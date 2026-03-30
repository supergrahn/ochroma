// Rolling spectral GI probe update via SDF cone marching.
// One workgroup per face of each selected probe (6 workgroups per probe).
// Dispatched as (num_selected_probes * 6, 1, 1).

struct ProbeUpdateParams {
    probe_indices: array<u32, 4>,
    probe_count: u32,
    _pad0: array<u32, 3>,
    grid_origin: vec3<f32>,
    grid_spacing: f32,
    grid_dims: vec3<u32>,
    _pad1: u32,
    frame_index: u32,
    _pad2: array<u32, 3>,
}

struct GpuProbe {
    radiance: array<f32, 48>,
    world_pos: vec3<f32>,
    _pad: f32,
}

struct SdfParams {
    origin: vec3<f32>,
    cell_size: f32,
    dims: vec3<u32>,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: ProbeUpdateParams;
@group(0) @binding(1) var<storage, read_write> probes: array<GpuProbe>;
@group(0) @binding(2) var<storage, read> sdf_volume: array<f32>;
@group(0) @binding(3) var<uniform> sdf_params: SdfParams;

@compute @workgroup_size(1)
fn probe_update_face(@builtin(workgroup_id) wg: vec3<u32>) {
    let dispatch_idx = wg.x; // 0..num_selected_probes*6
    let probe_slot = dispatch_idx / 6u;
    let face = dispatch_idx % 6u;

    if probe_slot >= params.probe_count { return; }
    let probe_idx = params.probe_indices[probe_slot];

    // Face direction vectors (+X, -X, +Y, -Y, +Z, -Z)
    let face_dir: vec3<f32> = face_direction(face);
    let probe_pos = probes[probe_idx].world_pos;

    // March N=16 cones in the hemisphere around face_dir
    var accum: array<f32, 8>;
    accum[0] = 0.0; accum[1] = 0.0; accum[2] = 0.0; accum[3] = 0.0;
    accum[4] = 0.0; accum[5] = 0.0; accum[6] = 0.0; accum[7] = 0.0;
    var total_weight = 0.0;

    for (var i = 0u; i < 16u; i++) {
        let sample_dir = hemisphere_sample(face_dir, i, 16u);
        let occlusion = sdf_march(probe_pos, sample_dir, 20.0);
        if occlusion < 1.0 {
            let neighbor_rad = sample_neighbor_probes(probe_pos + sample_dir * 1.0, sample_dir);
            let w = 1.0 - occlusion;
            for (var b = 0u; b < 8u; b++) {
                accum[b] += w * neighbor_rad[b];
            }
            total_weight += w;
        }
    }

    if total_weight > 0.0 {
        for (var b = 0u; b < 8u; b++) {
            accum[b] /= total_weight;
        }
    }

    // EMA: new = old * 0.9 + computed * 0.1
    let base = face * 8u;
    for (var b = 0u; b < 8u; b++) {
        probes[probe_idx].radiance[base + b] =
            probes[probe_idx].radiance[base + b] * 0.9 + accum[b] * 0.1;
    }
}

fn face_direction(face: u32) -> vec3<f32> {
    if face == 0u { return vec3(1.0, 0.0, 0.0); }
    if face == 1u { return vec3(-1.0, 0.0, 0.0); }
    if face == 2u { return vec3(0.0, 1.0, 0.0); }
    if face == 3u { return vec3(0.0, -1.0, 0.0); }
    if face == 4u { return vec3(0.0, 0.0, 1.0); }
    return vec3(0.0, 0.0, -1.0);
}

// SDF sphere-trace: returns 0.0 if unoccluded, 1.0 if hit within max_dist
fn sdf_march(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> f32 {
    var p = origin;
    var t = 0.0;
    for (var step = 0u; step < 32u; step++) {
        let d = sample_sdf(p);
        if d < 0.01 { return 1.0; }
        t += d;
        if t >= max_dist { return 0.0; }
        p = origin + dir * t;
    }
    return 0.0;
}

// Sample SDF at world position — nearest-neighbor lookup
fn sample_sdf(p: vec3<f32>) -> f32 {
    let rel = (p - sdf_params.origin) / sdf_params.cell_size;
    let idx = vec3<i32>(i32(rel.x), i32(rel.y), i32(rel.z));
    if idx.x < 0 || idx.y < 0 || idx.z < 0 {
        return 1.0;
    }
    let dims = vec3<i32>(i32(sdf_params.dims.x), i32(sdf_params.dims.y), i32(sdf_params.dims.z));
    if idx.x >= dims.x - 1 || idx.y >= dims.y - 1 || idx.z >= dims.z - 1 {
        return 1.0;
    }
    let i000 = idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;
    return sdf_volume[i000];
}

fn sample_neighbor_probes(pos: vec3<f32>, normal: vec3<f32>) -> array<f32, 8> {
    // Placeholder ambient radiance — full trilinear probe interpolation would go here
    var result: array<f32, 8>;
    result[0] = 0.01; result[1] = 0.01; result[2] = 0.01; result[3] = 0.01;
    result[4] = 0.01; result[5] = 0.01; result[6] = 0.01; result[7] = 0.01;
    return result;
}

fn hemisphere_sample(normal: vec3<f32>, i: u32, n: u32) -> vec3<f32> {
    // Stratified hemisphere sampling using Hammersley-like sequence
    let phi = 2.0 * 3.14159265 * f32(i) / f32(n);
    let cos_theta = 1.0 - f32(i + 1u) / f32(n + 1u);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let local = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    // Build tangent frame aligned to normal
    let up = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(normal.x) < 0.9);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return tangent * local.x + bitangent * local.y + normal * local.z;
}
