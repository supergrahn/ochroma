// tile_range_build.wgsl — build per-tile [start, end) spans from radix-sorted
// tile keys. The keystone that fills splat_raster's `tile_ranges` binding (3):
// after the radix sort groups entries by tile id (ascending), each tile's
// contiguous run in the sorted array is [start, end). Mirrors the CPU boundary
// scan in `spectra_render.rs` (the `cpu_tile_ranges` oracle), op-for-op.
//
// Two entry points share one bind group layout:
//   clear_ranges — one thread per tile, zeroes ranges[t] = (0,0) (empty tiles
//                  stay (0,0); start == end => the raster skips them).
//   build_ranges — one thread per sorted entry i. At a tile boundary (cur != prev)
//                  the SAME thread closes prev (ranges[prev].y = i) and opens cur
//                  (ranges[cur].x = i); thread 0 opens the first tile; the last
//                  thread closes its tile (ranges[cur].y = count). Each component
//                  of each ranges[t] is written by exactly one thread (.x and .y
//                  are distinct 4-byte locations), so there is no data race.

struct TileRangeParams {
    count: u32,      // number of sorted entries (host-side, from tile_count readback)
    num_tiles: u32,  // tiles_x * tiles_y
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform>             params:         TileRangeParams;
@group(0) @binding(1) var<storage, read>       sorted_keys_hi: array<u32>;     // tile id per sorted entry
@group(0) @binding(2) var<storage, read_write> ranges:         array<vec2<u32>>; // [start, end) per tile

@compute @workgroup_size(256)
fn clear_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.num_tiles) {
        return;
    }
    ranges[t] = vec2<u32>(0u, 0u);
}

@compute @workgroup_size(256)
fn build_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let cur = sorted_keys_hi[i];
    if (cur >= params.num_tiles) {
        return; // defensive: out-of-range tile id is ignored, never OOB-writes
    }

    if (i == 0u) {
        // First sorted entry opens its tile.
        ranges[cur].x = 0u;
    } else {
        let prev = sorted_keys_hi[i - 1u];
        if (cur != prev) {
            // Boundary: prev tile ends here, cur tile starts here (same thread).
            if (prev < params.num_tiles) {
                ranges[prev].y = i;
            }
            ranges[cur].x = i;
        }
    }

    // Last sorted entry closes its tile.
    if (i == params.count - 1u) {
        ranges[cur].y = params.count;
    }
}
