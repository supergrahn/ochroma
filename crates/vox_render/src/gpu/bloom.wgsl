// Dual-Kawase bloom for spectral framebuffer.
// Three entry points: bloom_extract, bloom_downsample, bloom_upsample

struct BloomParams {
    width: u32,
    height: u32,
    level: u32,         // current mip level (0 = full res)
    threshold: f32,
    strength: f32,
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<storage, read>       input_lo:  array<vec4<f32>>;  // bands 0-3
@group(0) @binding(1) var<storage, read>       input_hi:  array<vec4<f32>>;  // bands 4-7
@group(0) @binding(2) var<storage, read_write> output_lo: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output_hi: array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params:    BloomParams;

fn pixel_idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Extract bright regions: keep only pixels above threshold
@compute @workgroup_size(16, 16)
fn bloom_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }
    let idx = pixel_idx(gid.x, gid.y);
    let lo = input_lo[idx];
    let hi = input_hi[idx];
    let luminance = dot(lo.xyz, vec3<f32>(0.2126, 0.7152, 0.0722))
                  + dot(hi.xyz, vec3<f32>(0.05, 0.02, 0.01));
    let weight = max(0.0, luminance - params.threshold) / (luminance + 0.001);
    output_lo[idx] = lo * weight;
    output_hi[idx] = hi * weight;
}

// Dual-Kawase downsample: sample 4 neighbors at half-pixel offsets
@compute @workgroup_size(16, 16)
fn bloom_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_w = params.width;
    let out_h = params.height;
    if gid.x >= out_w || gid.y >= out_h { return; }

    let in_w  = out_w * 2u;
    let in_h2 = out_h * 2u;
    // 4-sample box filter from input at 2x resolution
    let sx = gid.x * 2u;
    let sy = gid.y * 2u;
    let i00 = sy * in_w + sx;
    let i10 = sy * in_w + min(sx + 1u, in_w - 1u);
    let i01 = min(sy + 1u, in_h2 - 1u) * in_w + sx;
    let i11 = min(sy + 1u, in_h2 - 1u) * in_w + min(sx + 1u, in_w - 1u);

    let out_idx = gid.y * out_w + gid.x;
    output_lo[out_idx] = (input_lo[i00] + input_lo[i10] + input_lo[i01] + input_lo[i11]) * 0.25;
    output_hi[out_idx] = (input_hi[i00] + input_hi[i10] + input_hi[i01] + input_hi[i11]) * 0.25;
}

// Dual-Kawase upsample + additive combine
@compute @workgroup_size(16, 16)
fn bloom_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }
    let idx = pixel_idx(gid.x, gid.y);
    // Add upsampled bloom into output with strength weight
    output_lo[idx] += input_lo[idx] * params.strength;
    output_hi[idx] += input_hi[idx] * params.strength;
}
