// shadow_shader.wgsl — depth-only pass for sun shadow map
struct LightUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> light: LightUniform;

struct SplatData {
    position: vec3<f32>,
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
    opacity: f32,
    _pad: f32,
    spectral: array<f32, 8>,
};
@group(0) @binding(1) var<storage, read> splats: array<SplatData>;

@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let splat_idx = vi / 6u;
    let pos = vec4(splats[splat_idx].position, 1.0);
    return light.view_proj * pos;
}
