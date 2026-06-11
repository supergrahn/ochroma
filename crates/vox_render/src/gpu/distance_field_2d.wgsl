// Masked, integer-deterministic 2D jump-flood distance transform.
//
// Determinism contract (load-bearing — the sim feeds employment gating and
// replay from the read-back grid):
//   * gather-style ping-pong: every pass reads only the previous cost state
//     and writes only its own cells — a pure function of the previous
//     buffer, no atomics, no rasterization order anywhere;
//   * all state is integer: parent waypoint coords packed in the rg16uint
//     bit layout ((y << 16) | x) and an accumulated chamfer cost in tenths
//     of a cell (axis step = 10, diagonal = 14 — the same integer metric an
//     8-neighbour integer Dijkstra uses), saturating at 0xFFFE (a ~6553-cell
//     geodesic cap; 0xFFFF is the no-value sentinel);
//   * candidates are scanned in a fixed 8-direction order and adopted only
//     on strict integer `<` improvement, so ties resolve identically on
//     every GPU;
//   * jump-tap positions use 24.8 fixed-point integer arithmetic — no
//     floating-point anywhere in the propagation.
//
// Geodesic semantics: each cell's payload is the jump-source neighbour it
// was reached from (a waypoint chain back to a seed), not the seed itself —
// a plain nearest-seed-coord JFA can only ever report straight-line
// distance, which is wrong the moment a wall forces a detour. The chain's
// accumulated integer cost is the geodesic approximation; the CPU readback
// recomputes exact per-segment lengths from the integer coords.
//
// Memory layout (the 780M is memory-bound at 1024²): costs pack two u16
// cells per u32 word and one thread owns one word, so there are no
// half-word write races; the parent grid is a SINGLE buffer (never read by
// candidate evaluation — a candidate's parent is the jump neighbour itself)
// written only on improvement, halving per-pass traffic again.

struct Globals {
    size: u32,
    n_seeds: u32,
    _pad0: u32,
    _pad1: u32,
}

struct StepU {
    step: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> step_u: StepU;
// u8 cost mask packed 4 cells per word; 255 = impassable.
@group(0) @binding(2) var<storage, read> mask_words: array<u32>;
@group(0) @binding(3) var<storage, read> seeds: array<vec2<u32>>;
// Chamfer cost in tenths of a cell, two u16 cells per word (x-major).
@group(0) @binding(4) var<storage, read> src_cost: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_cost: array<u32>;
// rg16uint-packed parent waypoint per cell; single buffer, write-on-improve.
@group(0) @binding(6) var<storage, read_write> parent: array<u32>;
// Resolve outputs: f16 distance pairs (cell units, debug/render only) and
// u16 seed-id pairs, copied buffer→texture into r16float / r16uint.
@group(0) @binding(7) var<storage, read_write> dist_pack: array<u32>;
@group(0) @binding(8) var<storage, read_write> id_pack: array<u32>;

const SENTINEL: u32 = 0xffffffffu;
const SENT16: u32 = 0xffffu;
const COST_CAP: u32 = 0xfffeu;
const IMPASSABLE: u32 = 255u;
const AXIS_COST: u32 = 10u;
const DIAG_COST: u32 = 14u;

fn mask_at(x: u32, y: u32) -> u32 {
    let idx = y * globals.size + x;
    return (mask_words[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu;
}

fn cost_at(x: u32, y: u32) -> u32 {
    let idx = y * globals.size + x;
    return (src_cost[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xffffu;
}

// One packed cost word, SENTINEL (both halves no-value) outside the grid.
fn cost_word(wc: i32, y: i32) -> u32 {
    let half = i32(globals.size / 2u);
    if (wc < 0 || wc >= half || y < 0 || y >= i32(globals.size)) {
        return SENTINEL;
    }
    return src_cost[u32(y) * u32(half) + u32(wc)];
}

// rg16uint bit layout: r (low 16) = x, g (high 16) = y.
fn pack_cell(x: u32, y: u32) -> u32 {
    return (y << 16u) | (x & 0xffffu);
}

// Fixed 8-direction scan order (axis first, then diagonals). A switch keeps
// the table in registers/branches — a local `var` array would be
// dynamically indexed and spill to per-thread scratch memory, which costs
// more than the whole pass.
fn dir_of(d: u32) -> vec2<i32> {
    switch d {
        case 0u: { return vec2<i32>(1, 0); }
        case 1u: { return vec2<i32>(-1, 0); }
        case 2u: { return vec2<i32>(0, 1); }
        case 3u: { return vec2<i32>(0, -1); }
        case 4u: { return vec2<i32>(1, 1); }
        case 5u: { return vec2<i32>(1, -1); }
        case 6u: { return vec2<i32>(-1, 1); }
        default: { return vec2<i32>(-1, -1); }
    }
}

// Seed placement is fused into init (one thread per cost word): a seed
// kernel writing single cells would race on the shared u16 halves.
@compute @workgroup_size(16, 16)
fn init_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = globals.size;
    let half = size / 2u;
    if (gid.x >= half || gid.y >= size) {
        return;
    }
    var word = SENTINEL; // both halves = no value
    for (var k = 0u; k < 2u; k = k + 1u) {
        let x = gid.x * 2u + k;
        parent[gid.y * size + x] = SENTINEL;
        for (var s = 0u; s < globals.n_seeds; s = s + 1u) {
            let sd = seeds[s];
            if (sd.y != gid.y || sd.x != x) {
                continue;
            }
            // Impassable cells never hold seeds.
            if (mask_at(x, gid.y) == IMPASSABLE) {
                continue;
            }
            word = word & ~(SENT16 << (k * 16u)); // cost 0
            // A root points at itself — the chain-walk termination.
            parent[gid.y * size + x] = pack_cell(x, gid.y);
            break;
        }
    }
    dst_cost[gid.y * half + gid.x] = word;
}

@compute @workgroup_size(16, 16)
fn jfa_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = globals.size;
    let half = size / 2u;
    if (gid.x >= half || gid.y >= size) {
        return;
    }
    let wx = i32(gid.x);
    let y = i32(gid.y);
    let step = i32(step_u.step);
    let taps = min(step_u.step, 16u);
    // All candidate costs for both cells live in a 3x3 word neighbourhood
    // (word columns wx, wx ± max(step/2, 1); rows y, y ± step) — loaded
    // up-front into registers so the latency overlaps and nothing is read
    // twice. Every step in the pinned schedule is even except 1, and both
    // shapes resolve inside these nine words.
    let w_off = max(step >> 1u, 1);
    let r_m = vec3<u32>(
        cost_word(wx - w_off, y - step),
        cost_word(wx, y - step),
        cost_word(wx + w_off, y - step),
    );
    let r_0 = vec3<u32>(
        cost_word(wx - w_off, y),
        src_cost[gid.y * half + gid.x],
        cost_word(wx + w_off, y),
    );
    let r_p = vec3<u32>(
        cost_word(wx - w_off, y + step),
        cost_word(wx, y + step),
        cost_word(wx + w_off, y + step),
    );
    let own = r_0.y;
    var word = 0u;
    for (var k = 0u; k < 2u; k = k + 1u) {
        let cx = i32(gid.x * 2u + k);
        // Impassable cells never adopt seeds.
        if (mask_at(u32(cx), gid.y) == IMPASSABLE) {
            word = word | (SENT16 << (k * 16u));
            continue;
        }
        var best = (own >> (k * 16u)) & 0xffffu;
        var best_parent = 0u;
        var improved = false;
        let c = vec2<i32>(cx, y);
        // Cheapest-first candidate processing: repeatedly select the
        // unprocessed candidate with the lowest (cost, direction) key and
        // tap-check it; the first tap-clear one wins and every remaining
        // candidate costs at least as much, so the loop ends. This computes
        // exactly "min over tap-clear candidates, ties to the lowest
        // direction index" — identical to a fixed-order scan (the result is
        // order-independent for strict `<` adoption) but runs the expensive
        // tap loop ~once per improving cell instead of up to eight times
        // during flood passes. Selection keys are integers, so the outcome
        // stays bit-deterministic.
        var processed = 0u;
        loop {
            var sel_d = 8u;
            var sel_cost = best;
            var sel_n = c;
            for (var d = 0u; d < 8u; d = d + 1u) {
                if ((processed & (1u << d)) != 0u) {
                    continue;
                }
                let dir = dir_of(d);
                let n = c + dir * step;
                if (n.x < 0 || n.y < 0 || n.x >= i32(size) || n.y >= i32(size)) {
                    continue;
                }
                let row = select(select(r_0, r_m, dir.y < 0), r_p, dir.y > 0);
                let wcol = n.x >> 1u;
                let wn = select(select(1, 0, wcol < wx), 2, wcol > wx);
                let ncost = (row[wn] >> (u32(n.x & 1) * 16u)) & 0xffffu;
                if (ncost == SENT16) {
                    continue;
                }
                let unit = select(AXIS_COST, DIAG_COST, dir.x != 0 && dir.y != 0);
                let cand = min(ncost + unit * step_u.step, COST_CAP);
                // Strict integer improvement; `<` keeps the lowest d on ties.
                if (cand < sel_cost) {
                    sel_cost = cand;
                    sel_d = d;
                    sel_n = n;
                }
            }
            if (sel_d == 8u) {
                break;
            }
            processed = processed | (1u << sel_d);
            // Pinned jump acceptance: sample the mask at min(step, 16)
            // evenly spaced taps along c -> n; any impassable tap rejects
            // the jump. Taps advance by a 24.8 fixed-point stride — one
            // integer division per candidate instead of two per tap (GPUs
            // have no hardware integer divide), still integer-exact and
            // therefore bit-identical on every device: tap i sits at
            // (c*256 + stride*i + 128) >> 8, an arithmetic-shift floor that
            // rounds half-up for both signs.
            let delta = sel_n - c;
            let stride = (delta * 256) / i32(taps + 1u);
            let base = c * 256 + vec2<i32>(128, 128);
            var tap_blocked = false;
            for (var i = 1u; i <= taps; i = i + 1u) {
                let p = (base + stride * i32(i)) >> vec2<u32>(8u, 8u);
                if (mask_at(u32(p.x), u32(p.y)) == IMPASSABLE) {
                    tap_blocked = true;
                    break;
                }
            }
            if (tap_blocked) {
                continue;
            }
            best = sel_cost;
            best_parent = pack_cell(u32(sel_n.x), u32(sel_n.y));
            improved = true;
            break;
        }
        if (improved) {
            parent[gid.y * size + u32(cx)] = best_parent;
        }
        word = word | (best << (k * 16u));
    }
    dst_cost[gid.y * half + gid.x] = word;
}

// One thread resolves the two cells of one cost word so the f16 distance
// and u16 id outputs pack one u32 word each (rg16uint/r16float/r16uint are
// not storage-texture-capable in core WebGPU — the textures are filled by
// buffer→texture copies instead). Binds the FINAL cost buffer as src.
@compute @workgroup_size(16, 16)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = globals.size;
    let half = size / 2u;
    if (gid.x >= half || gid.y >= size) {
        return;
    }
    var ds = vec2<f32>(0.0, 0.0);
    var ids = vec2<u32>(SENT16, SENT16);
    for (var k = 0u; k < 2u; k = k + 1u) {
        let x = gid.x * 2u + k;
        let cost = cost_at(x, gid.y);
        if (cost == SENT16) {
            ds[k] = 65504.0; // f16 max = "unreached" debug value
            continue;
        }
        ds[k] = f32(cost) * 0.1; // chamfer tenths -> cell units
        // Walk the waypoint chain to its root (parent == self), then match
        // the seed list. Costs strictly decrease along the chain, so the
        // walk terminates; the guard is defensive only.
        var cur = gid.y * size + x;
        var guard = 0u;
        loop {
            let p = parent[cur];
            let pidx = (p >> 16u) * size + (p & 0xffffu);
            if (pidx == cur) {
                break;
            }
            cur = pidx;
            guard = guard + 1u;
            if (guard > 8192u) {
                break;
            }
        }
        let root = parent[cur];
        for (var s = 0u; s < globals.n_seeds; s = s + 1u) {
            if (pack_cell(seeds[s].x, seeds[s].y) == root) {
                ids[k] = s;
                break;
            }
        }
    }
    let w = gid.y * half + gid.x;
    dist_pack[w] = pack2x16float(ds);
    id_pack[w] = (ids.y << 16u) | ids.x;
}
