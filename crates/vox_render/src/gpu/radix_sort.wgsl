struct SortEntry {
    depth: f32,
    index: u32,
};

struct Params {
    count: u32,
    stage: u32,
    step: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<SortEntry>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn bitonic_sort_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }

    let half_block = 1u << params.step;

    let group = idx / half_block;
    let ascending = ((group / (1u << (params.stage - params.step))) % 2u) == 0u;

    let partner = idx ^ half_block;
    if partner <= idx || partner >= params.count { return; }

    let a = data[idx];
    let b = data[partner];

    let should_swap = select(a.depth < b.depth, a.depth > b.depth, ascending);
    if should_swap {
        data[idx] = b;
        data[partner] = a;
    }
}
