// Pass 2: convert cell_counts[] into exclusive prefix sums stored in cell_offsets[].
// cell_offsets has cell_count+1 entries; the last entry = total agent count.
// Single-thread pass (workgroup_size(1)). Adequate for cell_count up to ~2M.

struct PrefixUniforms {
    cell_count: u32,
    _pad: array<u32, 3>,
}

@group(0) @binding(0) var<storage, read>       cell_counts:  array<u32>;
@group(0) @binding(1) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(2) var<uniform>             pu:           PrefixUniforms;

@compute @workgroup_size(1)
fn main() {
    var running: u32 = 0u;
    for (var c: u32 = 0u; c < pu.cell_count; c = c + 1u) {
        cell_offsets[c] = running;
        running = running + cell_counts[c];
    }
    cell_offsets[pu.cell_count] = running;
}
