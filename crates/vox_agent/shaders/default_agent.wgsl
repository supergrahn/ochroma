// Default behavior: integrate velocity. Replace with game-specific shader.
// Bind group layout must match AgentComputeLayer::bind_group_layout_source().

struct AgentUniforms {
    agent_count:  u32,
    custom_floats: u32,
    dt:            f32,
    time:          f32,
    grid_width:    u32,
    cell_size:     f32,
    _pad0:         f32,
    _pad1:         f32,
}

@group(0) @binding(0) var<storage, read>       positions_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> positions_out: array<f32>;
@group(0) @binding(2) var<storage, read>       velocities_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocities_out:array<f32>;
@group(0) @binding(4) var<storage, read_write> agent_flags:   array<u32>;
@group(0) @binding(5) var<uniform>             uniforms:      AgentUniforms;

@compute @workgroup_size(64)
fn agent_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= uniforms.agent_count { return; }
    if (agent_flags[i] & 1u) == 0u { return; }

    let vx = velocities_in[i * 3u];
    let vy = velocities_in[i * 3u + 1u];
    let vz = velocities_in[i * 3u + 2u];

    positions_out[i * 3u]      = positions_in[i * 3u]      + vx * uniforms.dt;
    positions_out[i * 3u + 1u] = positions_in[i * 3u + 1u] + vy * uniforms.dt;
    positions_out[i * 3u + 2u] = positions_in[i * 3u + 2u] + vz * uniforms.dt;

    velocities_out[i * 3u]      = vx;
    velocities_out[i * 3u + 1u] = vy;
    velocities_out[i * 3u + 2u] = vz;
}
