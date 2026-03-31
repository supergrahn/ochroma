// Pass 3: scatter agent indices into cell_data in cell order.
// Reuses cell_counts as atomic write cursors (reset to 0 before this pass).

struct SpatialUniforms {
    agent_count:  u32,
    grid_width:   u32,
    cell_size:    f32,
    origin_x:     f32,
    origin_z:     f32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

@group(0) @binding(0) var<storage, read>       positions:    array<f32>;
@group(0) @binding(1) var<storage, read_write> cell_counts:  array<atomic<u32>>;
@group(0) @binding(2) var<storage, read>       cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_data:    array<u32>;
@group(0) @binding(4) var<storage, read_write> spatial_cell: array<u32>;
@group(0) @binding(5) var<uniform>             su:           SpatialUniforms;

fn world_to_cell(x: f32, z: f32) -> u32 {
    let cx = u32(clamp((x - su.origin_x) / su.cell_size,
                       0.0, f32(su.grid_width - 1u)));
    let cz = u32(clamp((z - su.origin_z) / su.cell_size,
                       0.0, f32(su.grid_width - 1u)));
    return cz * su.grid_width + cx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= su.agent_count { return; }
    let x = positions[i * 3u];
    let z = positions[i * 3u + 2u];
    let cell = world_to_cell(x, z);
    spatial_cell[i] = cell;
    let slot = atomicAdd(&cell_counts[cell], 1u);
    cell_data[cell_offsets[cell] + slot] = i;
}
