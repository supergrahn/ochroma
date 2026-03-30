// Spectral depth-of-field with per-band chromatic bokeh.
// Band 0 (violet) defocuses more than band 7 (red) — Abbe dispersion.

struct DofParams {
    focus_distance: f32,
    aperture: f32,
    focal_length: f32,
    abbe_coeff: f32,
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       spectral_lo: array<vec4<f32>>;  // bands 0-3
@group(0) @binding(1) var<storage, read>       spectral_hi: array<vec4<f32>>;  // bands 4-7
@group(0) @binding(2) var<storage, read>       depth_buf:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out_lo:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out_hi:      array<vec4<f32>>;
@group(0) @binding(5) var<uniform>             params:      DofParams;

// 18-point hexagonal Poisson disk (unit radius)
const DISK_COUNT: i32 = 18;
const DISK: array<vec2<f32>, 18> = array<vec2<f32>, 18>(
    vec2<f32>( 0.000000,  1.000000),
    vec2<f32>( 0.500000,  0.866025),
    vec2<f32>( 0.866025,  0.500000),
    vec2<f32>( 1.000000,  0.000000),
    vec2<f32>( 0.866025, -0.500000),
    vec2<f32>( 0.500000, -0.866025),
    vec2<f32>( 0.000000, -1.000000),
    vec2<f32>(-0.500000, -0.866025),
    vec2<f32>(-0.866025, -0.500000),
    vec2<f32>(-1.000000,  0.000000),
    vec2<f32>(-0.866025,  0.500000),
    vec2<f32>(-0.500000,  0.866025),
    vec2<f32>( 0.000000,  0.500000),
    vec2<f32>( 0.433013,  0.250000),
    vec2<f32>( 0.433013, -0.250000),
    vec2<f32>( 0.000000, -0.500000),
    vec2<f32>(-0.433013, -0.250000),
    vec2<f32>(-0.433013,  0.250000),
);

fn pixel_idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

fn clamp_coord(v: i32, limit: u32) -> u32 {
    return u32(clamp(v, 0, i32(limit) - 1));
}

@compute @workgroup_size(16, 16)
fn dof_compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }

    let idx   = pixel_idx(gid.x, gid.y);
    let depth = depth_buf[idx];

    // Circle of Confusion radius (in pixels)
    let fd  = params.focus_distance;
    let fl  = params.focal_length;
    let apt = params.aperture;
    let coc_base = apt * fl * abs(depth - fd) / max(depth * abs(fd - fl), 0.0001);

    // Accumulate one colour per spectral half (lo = bands 0-3, hi = bands 4-7)
    var sum_lo = vec4<f32>(0.0);
    var sum_hi = vec4<f32>(0.0);

    // We iterate the four spectral channels independently to model Abbe dispersion.
    // band index b in [0,7]; bands 0-3 live in lo.{x,y,z,w}, 4-7 in hi.{x,y,z,w}
    for (var b: i32 = 0; b < 8; b++) {
        // Per-band CoC: violet (b=0) has the largest defocus
        let band_coc = coc_base * (1.0 + params.abbe_coeff * f32(7 - b) / 7.0);

        var acc = 0.0;
        var wt  = 0.0;

        for (var s: i32 = 0; s < DISK_COUNT; s++) {
            let offset = DISK[s] * band_coc;
            let sx = clamp_coord(i32(gid.x) + i32(offset.x), params.width);
            let sy = clamp_coord(i32(gid.y) + i32(offset.y), params.height);
            let sidx = pixel_idx(sx, sy);

            var sample_val: f32;
            if b < 4 {
                let sv = spectral_lo[sidx];
                sample_val = array<f32, 4>(sv.x, sv.y, sv.z, sv.w)[b];
            } else {
                let sv = spectral_hi[sidx];
                sample_val = array<f32, 4>(sv.x, sv.y, sv.z, sv.w)[b - 4];
            }
            acc += sample_val;
            wt  += 1.0;
        }

        let result = acc / max(wt, 1.0);

        if b < 4 {
            switch b {
                case 0: { sum_lo.x = result; }
                case 1: { sum_lo.y = result; }
                case 2: { sum_lo.z = result; }
                default: { sum_lo.w = result; }
            }
        } else {
            switch b - 4 {
                case 0: { sum_hi.x = result; }
                case 1: { sum_hi.y = result; }
                case 2: { sum_hi.z = result; }
                default: { sum_hi.w = result; }
            }
        }
    }

    out_lo[idx] = sum_lo;
    out_hi[idx] = sum_hi;
}
