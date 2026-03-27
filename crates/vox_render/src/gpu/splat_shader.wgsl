// WGSL Gaussian Splat Rasteriser Shader
// Renders Gaussian splats as camera-facing billboard quads with spectral-to-sRGB conversion.

// ---------------------------------------------------------------------------
// Uniform / storage bindings
// ---------------------------------------------------------------------------

struct CameraUniform {
    view_proj   : mat4x4<f32>,
    view        : mat4x4<f32>,
    inv_view    : mat4x4<f32>,
    viewport_size : vec2<f32>,
    _pad        : vec2<f32>,
}

struct SplatData {
    position : vec3<f32>,
    scale_x  : f32,
    scale_y  : f32,
    scale_z  : f32,
    opacity  : f32,
    _pad     : f32,
    spectral : array<f32, 8>,
}

@group(0) @binding(0) var<uniform>           camera : CameraUniform;
@group(0) @binding(1) var<storage, read>     splats : array<SplatData>;

// ---------------------------------------------------------------------------
// Spectral → sRGB helpers
// ---------------------------------------------------------------------------

// CIE 1931 2° observer × D65 illuminant × 40 nm band spacing, at 8 bands:
// 380, 420, 460, 500, 540, 580, 620, 660 nm
const CIE_X_D65 : array<f32, 8> = array<f32, 8>(
    0.070, 2.982, 33.662, 0.536, 30.204, 89.548, 46.904, 6.501
);
const CIE_Y_D65 : array<f32, 8> = array<f32, 8>(
    0.000, 0.797, 6.009, 35.363, 99.252, 85.034, 32.968, 3.480
);
const CIE_Z_D65 : array<f32, 8> = array<f32, 8>(
    0.325, 14.238, 177.472, 29.767, 6.586, 0.166, 0.147, 0.000
);

// Sum of CIE_Y_D65 — used to normalise Y to [0, 1] for a perfect white reflector
const NORM_Y : f32 = 262.903;

fn spectral_to_linear_srgb(spectral: array<f32, 8>) -> vec3<f32> {
    // Integrate SPD against CIE observer × D65 to obtain XYZ
    var x = 0.0;
    var y = 0.0;
    var z = 0.0;

    for (var i = 0; i < 8; i++) {
        let s = spectral[i];
        x += s * CIE_X_D65[i];
        y += s * CIE_Y_D65[i];
        z += s * CIE_Z_D65[i];
    }

    // Normalise by D65 white (Y integral).  This maps a flat reflector to
    // Y = 1, matching the CPU path's per-illuminant normalisation.
    let inv_norm = 1.0 / NORM_Y;
    // Additional white-point scaling so XYZ→sRGB matrix maps white to [1,1,1].
    // The CPU side normalises X/norm_x * 0.9505 and Z/norm_z * 1.0888; here
    // we pre-absorb those factors into equivalent NORM constants.
    // norm_x (sum of CIE_X_D65) ≈ 310.317, so scale = 0.9505 / (310.317 / 262.903)
    // norm_z (sum of CIE_Z_D65) ≈ 228.701, so scale = 1.0888 / (228.701 / 262.903)
    let norm_x_d65 = 0.070 + 2.982 + 33.662 + 0.536 + 30.204 + 89.548 + 46.904 + 6.501; // 310.407
    let norm_z_d65 = 0.325 + 14.238 + 177.472 + 29.767 + 6.586 + 0.166 + 0.147 + 0.000; // 228.701

    let xn = (x / norm_x_d65) * 0.9505;
    let yn = y * inv_norm;
    let zn = (z / norm_z_d65) * 1.0888;

    // XYZ → linear sRGB (IEC 61966-2-1 / sRGB standard matrix)
    let r = max(0.0,  3.2406 * xn - 1.5372 * yn - 0.4986 * zn);
    let g = max(0.0, -0.9689 * xn + 1.8758 * yn + 0.0415 * zn);
    let b = max(0.0,  0.0557 * xn - 0.2040 * yn + 1.0570 * zn);

    return vec3<f32>(r, g, b);
}

fn linear_to_srgb_gamma(c: f32) -> f32 {
    if c <= 0.0031308 {
        return 12.92 * c;
    } else {
        return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
}

fn linear_srgb_to_srgb(lin: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_gamma(lin.x),
        linear_to_srgb_gamma(lin.y),
        linear_to_srgb_gamma(lin.z),
    );
}

// ---------------------------------------------------------------------------
// Quad corner offsets (two triangles, CCW winding, forming a [-1,1] square)
// ---------------------------------------------------------------------------

const QUAD_OFFSETS : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

// ---------------------------------------------------------------------------
// Vertex shader output
// ---------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0)       uv       : vec2<f32>,
    @location(1)       opacity  : f32,
    @location(2)       color    : vec3<f32>,
}

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_main(
    @builtin(vertex_index)   vertex_index   : u32,
    @builtin(instance_index) instance_index : u32,
) -> VertexOutput {
    let splat  = splats[instance_index];
    let corner = QUAD_OFFSETS[vertex_index];

    // Project splat centre to clip space
    let world_pos = vec4<f32>(splat.position, 1.0);
    let clip      = camera.view_proj * world_pos;

    // Average scale → world-space radius of the billboard
    let avg_scale = (splat.scale_x + splat.scale_y + splat.scale_z) / 3.0;

    // Convert world-space radius to NDC radius.
    // We use clip.w (the view-space depth) as the perspective divisor.
    // viewport_size.y controls the vertical extent; use the smaller dimension
    // for a square footprint in pixels.
    let ndc_radius = avg_scale / max(clip.w, 1e-6);

    // Scale NDC offset so the quad appears square in screen pixels.
    // Aspect ratio correction: NDC x spans [-1,1] over viewport_size.x pixels
    // while NDC y spans [-1,1] over viewport_size.y pixels.
    let aspect = camera.viewport_size.x / camera.viewport_size.y;
    let ndc_offset = vec2<f32>(corner.x * ndc_radius / aspect,
                               corner.y * ndc_radius);

    // Offset the clip-space position (after perspective divide we're in NDC,
    // so multiply offset back by clip.w to stay in clip space).
    var out_clip = clip;
    out_clip.x += ndc_offset.x * clip.w;
    out_clip.y += ndc_offset.y * clip.w;

    // Spectral → sRGB colour (computed once per vertex, interpolated to frag)
    let linear_rgb = spectral_to_linear_srgb(splat.spectral);
    let srgb       = linear_srgb_to_srgb(linear_rgb);

    var out : VertexOutput;
    out.position = out_clip;
    out.uv       = corner;            // already in [-1,1]
    out.opacity  = splat.opacity;
    out.color    = srgb;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

// Gaussian sigma for the splat footprint; the billboard covers [-1,1] in UV
// so sigma = 0.5 gives the density ≈ 0 at the quad edges.
const SIGMA : f32 = 0.5;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 2D isotropic Gaussian falloff
    let sigma2    = 2.0 * SIGMA * SIGMA;
    let gauss     = exp(-dot(in.uv, in.uv) / sigma2);
    let alpha     = gauss * in.opacity;

    if alpha < 0.004 {
        discard;
    }

    return vec4<f32>(in.color, alpha);
}
