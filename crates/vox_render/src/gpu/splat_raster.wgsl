// splat_raster.wgsl — EWA tile rasterizer for spectral Gaussian splats.
//
// One workgroup per 16×16 tile. Workgroup size: 256 (16×16).
// Splats are iterated front-to-back within each tile using pre-sorted indices.
// Output: 4-layer rgba32float texture array holding 8 spectral bands + transmittance.

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct CameraUniform {
    view_proj:     mat4x4<f32>,
    view:          mat4x4<f32>,
    inv_view:      mat4x4<f32>,
    viewport_size: vec2<f32>,
    _pad:          vec2<f32>,
}

struct GpuSplatFull {
    position_depth: vec4<f32>,        // xyz = world pos, w = view-space depth
    conic:          vec3<f32>,        // EWA conic coefficients
    _pad0:          f32,
    opacity_color:  vec4<f32>,        // w = opacity (0..1)
    spectral:       array<f32, 8>,    // 8 spectral bands
}

struct TileRangesBuffer {
    ranges: array<vec2<u32>>,         // x = start, y = end (index into sorted_vals)
}

struct RasterParams {
    width:       u32,
    height:      u32,
    num_tiles_x: u32,
    _pad:        u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform>        camera:      CameraUniform;
@group(0) @binding(1) var<storage, read>  splats:      array<GpuSplatFull>;
@group(0) @binding(2) var<storage, read>  sorted_vals: array<u32>;
@group(0) @binding(3) var<storage, read>  tile_ranges: TileRangesBuffer;
@group(0) @binding(4) var                 output:      texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var<uniform>        params:      RasterParams;

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id)         wg:  vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let tile_x = wg.x;
    let tile_y = wg.y;
    let tile_id = tile_x + tile_y * params.num_tiles_x;

    let px = tile_x * 16u + lid.x;
    let py = tile_y * 16u + lid.y;

    // Out-of-bounds guard — still need to run to avoid divergence in uniform flow,
    // but we skip the expensive inner loop and early-return before any stores.
    let in_bounds = px < params.width && py < params.height;

    // Load tile range.
    let range = tile_ranges.ranges[tile_id];
    let start = range.x;
    let end   = range.y;

    // Per-thread accumulators.
    var transmittance: f32 = 1.0;
    var accum: array<f32, 8>;
    for (var b = 0; b < 8; b++) {
        accum[b] = 0.0;
    }

    if in_bounds {
        let pixel_center = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
        let w = f32(params.width);
        let h = f32(params.height);

        for (var i = start; i < end; i++) {
            let splat_idx = sorted_vals[i];
            let sp = splats[splat_idx];

            // Project splat world position to screen.
            let clip   = camera.view_proj * vec4<f32>(sp.position_depth.xyz, 1.0);
            let ndc    = clip.xy / clip.w;
            let screen = (ndc * 0.5 + 0.5) * vec2<f32>(w, h);

            let dx = pixel_center.x - screen.x;
            let dy = pixel_center.y - screen.y;

            // EWA Gaussian evaluation: exponent = -0.5 * (a*dx² + 2b*dx*dy + c*dy²)
            let exponent = -0.5 * (sp.conic.x * dx * dx
                                 + 2.0 * sp.conic.y * dx * dy
                                 + sp.conic.z * dy * dy);

            if exponent < -8.0 {
                continue;
            }

            let alpha = min(0.99, sp.opacity_color.w * exp(exponent));
            if alpha < (1.0 / 255.0) {
                continue;
            }

            let weight = alpha * transmittance;
            for (var b = 0; b < 8; b++) {
                accum[b] += weight * sp.spectral[b];
            }

            transmittance *= (1.0 - alpha);
            if transmittance < 0.001 {
                break;
            }
        }
    }

    if !in_bounds {
        return;
    }

    let coord = vec2<i32>(i32(px), i32(py));

    // Layer 0: spectral bands 0–3
    textureStore(output, coord, 0, vec4<f32>(accum[0], accum[1], accum[2], accum[3]));
    // Layer 1: spectral bands 4–7
    textureStore(output, coord, 1, vec4<f32>(accum[4], accum[5], accum[6], accum[7]));
    // Layer 3: transmittance (r channel)
    textureStore(output, coord, 3, vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
