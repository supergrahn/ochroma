// Pass 2: spatial hashing, build neighbor lists
struct Particle {
    position:      vec3<f32>, density: f32,
    predicted_pos: vec3<f32>, lambda:  f32,
    velocity:      vec3<f32>, pressure: f32,
    spectral:      array<f32, 16>,
}
struct PbfParams {
    dt: f32, rest_density: f32, smoothing_h: f32, epsilon: f32,
    particle_count: u32, max_neighbors: u32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: PbfParams;
// neighbor_buf: for each particle, [neighbor_count, n0, n1, ..., n63]
@group(0) @binding(2) var<storage, read_write> neighbors: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count { return; }
    let h2 = params.smoothing_h * params.smoothing_h;
    let stride = params.max_neighbors + 1u;
    let base = i * stride;
    var count: u32 = 0u;
    for (var j: u32 = 0u; j < params.particle_count; j = j + 1u) {
        if j == i { continue; }
        let d = particles[i].predicted_pos - particles[j].predicted_pos;
        let r2 = dot(d, d);
        if r2 < h2 && count < params.max_neighbors {
            neighbors[base + 1u + count] = j;
            count = count + 1u;
        }
    }
    neighbors[base] = count;
}
