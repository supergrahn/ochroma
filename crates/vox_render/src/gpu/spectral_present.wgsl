struct PresentParams {
    inv_view_proj: mat4x4<f32>,
    light_dir_atten: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
    light_color_lit: vec4<f32>,
    light_core: vec4<f32>,
    dims_fog: vec4<f32>,
    rgb0: vec4<f32>,
    rgb1: vec4<f32>,
    rgb2: vec4<f32>,
    rgb3: vec4<f32>,
    rgb4: vec4<f32>,
    rgb5: vec4<f32>,
    rgb6: vec4<f32>,
    rgb7: vec4<f32>,
};

struct Coverage {
    covered: atomic<u32>,
};

@group(0) @binding(0) var spectral_tex: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> params: PresentParams;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read_write> coverage: Coverage;

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn unpack_geometry_rgb(px: vec2<i32>) -> vec3<f32> {
    let p0 = textureLoad(spectral_tex, px, 0, 0);
    let p1 = textureLoad(spectral_tex, px, 1, 0);
    var rgb = vec3<f32>(0.0);
    rgb += p0.x * params.rgb0.xyz;
    rgb += p0.y * params.rgb1.xyz;
    rgb += p0.z * params.rgb2.xyz;
    rgb += p0.w * params.rgb3.xyz;
    rgb += p1.x * params.rgb4.xyz;
    rgb += p1.y * params.rgb5.xyz;
    rgb += p1.z * params.rgb6.xyz;
    rgb += p1.w * params.rgb7.xyz;
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0;
}

fn backdrop_color(x: u32, y: u32) -> vec3<f32> {
    let width = params.dims_fog.x;
    let height = params.dims_fog.y;
    let fog = clamp(params.dims_fog.z, 0.0, 1.0);
    let ndc = vec2<f32>(
        2.0 * (f32(x) + 0.5) / width - 1.0,
        1.0 - 2.0 * (f32(y) + 0.5) / height,
    );
    let near4 = params.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    let far4 = params.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let near = near4.xyz / near4.w;
    let far = far4.xyz / far4.w;
    let dir = normalize(far - near);

    if (dir.y < -0.001) {
        let t = near.y / -dir.y;
        let dist = max(t * length(dir), 0.0);
        let f = 1.0 - exp(-dist * (0.0009 + 0.004 * fog));
        return lerp3(params.ground.xyz, params.horizon.xyz, clamp(f, 0.0, 1.0));
    }

    let up = pow(clamp(dir.y, 0.0, 1.0), 0.65);
    var color = lerp3(params.horizon.xyz, params.zenith.xyz, up);
    let d = clamp(dot(dir, params.light_dir_atten.xyz), -1.0, 1.0);
    if (d > 0.0) {
        let disk = pow(d, 2200.0);
        let glow = pow(d, 12.0) * 0.5;
        let bright = (disk + glow) * params.light_dir_atten.w;
        color += params.light_core.xyz * bright;
    }
    return color;
}

@compute @workgroup_size(16, 16, 1)
fn present(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(params.dims_fog.x);
    let height = u32(params.dims_fog.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let geom = unpack_geometry_rgb(vec2<i32>(i32(gid.x), i32(gid.y)));
    var rgb = backdrop_color(gid.x, gid.y);

    if (geom.x + geom.y + geom.z > 18.0) {
        atomicAdd(&coverage.covered, 1u);
        rgb = geom * params.light_color_lit.w * params.light_color_lit.xyz;
    }

    let out_rgb = clamp(rgb / 255.0, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(out_rgb, 1.0));
}
