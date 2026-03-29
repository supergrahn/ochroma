// Blend Tree Skinning Compute Shader
//
// Blends up to 4 animation poses on GPU before skinning.
// One thread per splat. Workgroup size 64.

const MAX_POSES: u32 = 4u;

struct GpuSkinSplat {
    position: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,
    opacity: f32,
    rotation: vec4<f32>,
    spectral: array<f32, 8>,
};

struct JointTransform {
    skin_matrix: mat4x4<f32>,
};

struct BlendUniform {
    weights: vec4<f32>,   // blend weight for each pose (must sum to 1.0)
    joint_count: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read>       base_splats:    array<GpuSkinSplat>;
@group(0) @binding(1) var<storage, read>       joint_bindings: array<u32>;
@group(0) @binding(2) var<storage, read>       pose0:          array<JointTransform>;
@group(0) @binding(3) var<storage, read>       pose1:          array<JointTransform>;
@group(0) @binding(4) var<storage, read>       pose2:          array<JointTransform>;
@group(0) @binding(5) var<storage, read>       pose3:          array<JointTransform>;
@group(0) @binding(6) var<uniform>             blend:          BlendUniform;
@group(0) @binding(7) var<storage, read_write> skinned_splats: array<GpuSkinSplat>;

// Matrix lerp: weighted sum of 4 matrices
fn blend_matrices(
    m0: mat4x4<f32>, w0: f32,
    m1: mat4x4<f32>, w1: f32,
    m2: mat4x4<f32>, w2: f32,
    m3: mat4x4<f32>, w3: f32,
) -> mat4x4<f32> {
    return m0 * w0 + m1 * w1 + m2 * w2 + m3 * w3;
}

fn mat4_to_quat(m: mat4x4<f32>) -> vec4<f32> {
    let sx = length(m[0].xyz);
    let sy = length(m[1].xyz);
    let sz = length(m[2].xyz);
    let r = mat3x3<f32>(m[0].xyz / sx, m[1].xyz / sy, m[2].xyz / sz);
    let trace = r[0][0] + r[1][1] + r[2][2];
    var q: vec4<f32>;
    if trace > 0.0 {
        let s = 0.5 / sqrt(trace + 1.0);
        q = vec4<f32>((r[2][1]-r[1][2])*s, (r[0][2]-r[2][0])*s, (r[1][0]-r[0][1])*s, 0.25/s);
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]);
        q = vec4<f32>(0.25*s, (r[0][1]+r[1][0])/s, (r[0][2]+r[2][0])/s, (r[2][1]-r[1][2])/s);
    } else if r[1][1] > r[2][2] {
        let s = 2.0 * sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]);
        q = vec4<f32>((r[0][1]+r[1][0])/s, 0.25*s, (r[1][2]+r[2][1])/s, (r[0][2]-r[2][0])/s);
    } else {
        let s = 2.0 * sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]);
        q = vec4<f32>((r[0][2]+r[2][0])/s, (r[1][2]+r[2][1])/s, 0.25*s, (r[1][0]-r[0][1])/s);
    }
    return normalize(q);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}

@compute @workgroup_size(64)
fn cs_blend_skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&base_splats) { return; }

    let splat = base_splats[idx];
    let joint_idx = joint_bindings[idx];

    // Blend 4 pose matrices for this joint
    let blended = blend_matrices(
        pose0[joint_idx].skin_matrix, blend.weights.x,
        pose1[joint_idx].skin_matrix, blend.weights.y,
        pose2[joint_idx].skin_matrix, blend.weights.z,
        pose3[joint_idx].skin_matrix, blend.weights.w,
    );

    var out = splat;
    out.position = (blended * vec4<f32>(splat.position, 1.0)).xyz;
    let joint_q = mat4_to_quat(blended);
    out.rotation = normalize(quat_mul(joint_q, splat.rotation));

    skinned_splats[idx] = out;
}
