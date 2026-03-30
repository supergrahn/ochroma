// Pass 1: predict positions from velocity (apply gravity)
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count { return; }
    var v = particles[i].velocity;
    v.y -= 9.81 * params.dt;
    particles[i].velocity = v;
    particles[i].predicted_pos = particles[i].position + v * params.dt;
}
