// write_indirect_args.wgsl
// Single-invocation compute shader that converts a raw tile entry count into
// an indirect dispatch argument buffer suitable for `dispatchWorkgroupsIndirect`.
//
// The output format mirrors `wgpu::util::DispatchIndirectArgs`:
//   indirect_args[0] = ceil(tile_count / 256)   — workgroup X count
//   indirect_args[1] = 1                          — workgroup Y count
//   indirect_args[2] = 1                          — workgroup Z count

@group(0) @binding(0) var<storage, read>       tile_count:    u32;
@group(0) @binding(1) var<storage, read_write> indirect_args: array<u32, 3>;

@compute @workgroup_size(1)
fn write_indirect_args(@builtin(global_invocation_id) _gid: vec3<u32>) {
    let count = tile_count;
    // Ceiling division: (count + 255) / 256
    let wg_x = (count + 255u) / 256u;
    indirect_args[0] = wg_x;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
}
