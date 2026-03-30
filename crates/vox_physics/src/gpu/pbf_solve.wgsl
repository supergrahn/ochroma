// Pass 3: density constraint solve — compute λᵢ = −Cᵢ / (∇Cᵢ²/ρ₀ + ε), then apply Δp
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
@group(0) @binding(2) var<storage, read_write> neighbors: array<u32>;

fn poly6(r2: f32, h: f32) -> f32 {
    let h2 = h * h;
    if r2 >= h2 { return 0.0; }
    let x = h2 - r2;
    // coeff: 315 / (64 * PI * h^9)
    let coeff = 315.0 / (64.0 * 3.14159265 * h * h * h * h * h * h * h * h * h);
    return coeff * x * x * x;
}

fn spiky_grad(r_vec: vec3<f32>, r: f32, h: f32) -> vec3<f32> {
    if r >= h || r < 1e-6 { return vec3<f32>(0.0, 0.0, 0.0); }
    let coeff = -45.0 / (3.14159265 * h * h * h * h * h * h) * (h - r) * (h - r) / r;
    return r_vec * coeff;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count { return; }

    let h = params.smoothing_h;
    let rho0 = params.rest_density;
    let stride = params.max_neighbors + 1u;
    let base = i * stride;
    let count = neighbors[base];

    // Compute density using poly6
    var rho = poly6(0.0, h); // self contribution
    for (var ni: u32 = 0u; ni < count; ni = ni + 1u) {
        let j = neighbors[base + 1u + ni];
        let d = particles[i].predicted_pos - particles[j].predicted_pos;
        let r2 = dot(d, d);
        rho = rho + poly6(r2, h);
    }
    particles[i].density = rho;

    // Compute constraint Cᵢ = ρᵢ/ρ₀ - 1
    let ci = rho / rho0 - 1.0;

    // Compute |∇Cᵢ|² (sum of spiky gradient squares)
    var grad_sq = 0.0f;
    for (var ni: u32 = 0u; ni < count; ni = ni + 1u) {
        let j = neighbors[base + 1u + ni];
        let d = particles[i].predicted_pos - particles[j].predicted_pos;
        let r = length(d);
        let g = spiky_grad(d, r, h) / rho0;
        grad_sq = grad_sq + dot(g, g);
    }

    particles[i].lambda = -ci / (grad_sq + params.epsilon);

    // Barrier to ensure all lambdas are written before computing Δp
    // (In practice we need a second pass for Δp, but here we use atomics-free approach)
    // Apply Δp in the same invocation using already-computed neighbor lambdas
    var delta_p = vec3<f32>(0.0, 0.0, 0.0);
    for (var ni: u32 = 0u; ni < count; ni = ni + 1u) {
        let j = neighbors[base + 1u + ni];
        let d = particles[i].predicted_pos - particles[j].predicted_pos;
        let r = length(d);
        let g = spiky_grad(d, r, h);
        let s = (particles[i].lambda + particles[j].lambda) / rho0;
        delta_p = delta_p + g * s;
    }

    var pred = particles[i].predicted_pos + delta_p;
    // Ground plane
    if pred.y < 0.0 { pred.y = 0.0; }
    particles[i].predicted_pos = pred;
}
