// 4-pass 8-bit GPU radix sort — histogram, prefix sum, scatter.
// All three entry points share the same bind group layout (group 0).
//
// Binding map:
//   0: RadixParams uniform
//   1: keys_lo  (storage, read_only)  — low 32 bits of sort key
//   2: keys_hi  (storage, read_only)  — high 32 bits of sort key
//   3: vals     (storage, read_only)  — splat index per entry
//   4: histogram (storage, read_write) — 256 u32 bucket counts
//   5: prefix   (storage, read_write) — 256 u32 exclusive prefix sums
//   6: keys_lo_out (storage, read_write)
//   7: keys_hi_out (storage, read_write)
//   8: vals_out    (storage, read_write)

struct RadixParams {
    count:      u32,
    bit_shift:  u32,   // 0 / 8 / 16 / 24
    use_hi_key: u32,   // 0 = keys_lo, 1 = keys_hi
    pass_idx:   u32,
}

@group(0) @binding(0) var<uniform>           sort_params:  RadixParams;
@group(0) @binding(1) var<storage, read>     keys_lo_in:   array<u32>;
@group(0) @binding(2) var<storage, read>     keys_hi_in:   array<u32>;
@group(0) @binding(3) var<storage, read>     vals_in:      array<u32>;
@group(0) @binding(4) var<storage, read_write> histogram:  array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> prefix:     array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> keys_lo_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> keys_hi_out: array<u32>;
@group(0) @binding(8) var<storage, read_write> vals_out:   array<u32>;

// ---------------------------------------------------------------------------
// Shared workgroup histogram (256 buckets × 4 bytes = 1 KiB)
// ---------------------------------------------------------------------------
var<workgroup> wg_hist: array<atomic<u32>, 256>;

// ---------------------------------------------------------------------------
// Pass 1: radix_histogram
//   Each workgroup tallies a local 256-bucket histogram for its slice of keys,
//   then flushes the local counts into the global histogram with atomicAdd.
//   The global histogram must be zero-initialised by the host before dispatch.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn radix_histogram(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
) {
    // Zero the workgroup histogram.
    atomicStore(&wg_hist[lid.x], 0u);
    workgroupBarrier();

    let idx = gid.x;
    if idx < sort_params.count {
        let key = select(keys_lo_in[idx], keys_hi_in[idx], sort_params.use_hi_key != 0u);
        let bucket = (key >> sort_params.bit_shift) & 0xFFu;
        atomicAdd(&wg_hist[bucket], 1u);
    }

    workgroupBarrier();

    // Each thread flushes one bucket from the workgroup histogram to global.
    atomicAdd(&histogram[lid.x], atomicLoad(&wg_hist[lid.x]));
}

// ---------------------------------------------------------------------------
// Pass 2: radix_prefix_sum
//   Exclusive prefix sum over the 256-entry histogram — Hillis-Steele scan.
//   Must be dispatched as exactly 1 workgroup of 256 threads.
// ---------------------------------------------------------------------------
var<workgroup> scan_buf: array<u32, 256>;

@compute @workgroup_size(256)
fn radix_prefix_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = lid.x;
    // Load histogram counts into shared memory.
    scan_buf[i] = atomicLoad(&histogram[i]);
    workgroupBarrier();

    // Up-sweep / down-sweep exclusive scan (Hillis-Steele variant).
    // After this the scan_buf holds the exclusive prefix sum.

    // ---- up-sweep (reduce) ----
    var offset: u32 = 1u;
    var d: u32 = 128u;
    loop {
        if d == 0u { break; }
        workgroupBarrier();
        if i < d {
            let ai = offset * (2u * i + 1u) - 1u;
            let bi = offset * (2u * i + 2u) - 1u;
            scan_buf[bi] += scan_buf[ai];
        }
        offset = offset << 1u;
        d = d >> 1u;
    }

    // Zero the last element for exclusive scan.
    if i == 0u {
        scan_buf[255] = 0u;
    }

    // ---- down-sweep ----
    offset = 128u;
    d = 1u;
    loop {
        if offset == 0u { break; }
        workgroupBarrier();
        if i < d {
            let ai = offset * (2u * i + 1u) - 1u;
            let bi = offset * (2u * i + 2u) - 1u;
            let tmp     = scan_buf[ai];
            scan_buf[ai] = scan_buf[bi];
            scan_buf[bi] = scan_buf[bi] + tmp;
        }
        offset = offset >> 1u;
        d = d << 1u;
    }

    workgroupBarrier();
    // Write exclusive prefix into output prefix buffer (used as per-bucket cursor).
    atomicStore(&prefix[i], scan_buf[i]);
}

// ---------------------------------------------------------------------------
// Pass 3: radix_scatter
//   Each thread reads its key, computes the 8-bit bucket, atomically claims
//   a slot in prefix[bucket], and writes (key_lo, key_hi, val) to the output.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx >= sort_params.count { return; }

    let lo  = keys_lo_in[idx];
    let hi  = keys_hi_in[idx];
    let val = vals_in[idx];

    let key    = select(lo, hi, sort_params.use_hi_key != 0u);
    let bucket = (key >> sort_params.bit_shift) & 0xFFu;

    // Atomically claim the next available output slot for this bucket.
    let dest = atomicAdd(&prefix[bucket], 1u);

    keys_lo_out[dest] = lo;
    keys_hi_out[dest] = hi;
    vals_out[dest]    = val;
}
