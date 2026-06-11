// expand_draws.wgsl
// One thread per SELECTED atom: binary-search the owning draw by its
// prefix-summed atom_offset, fetch the asset-local packed library atom through
// the flat index table, transform it by the draw's instance (rotate-then-
// translate; instance quat applied AFTER the atom quat), and write the result
// straight into the tiled chain's persistent buffers:
//   * out_splats[i]            — GpuSplatFull (conic / position_depth.w left 0;
//                                tile_assign fills them in-device)
//   * out_transforms[2i, 2i+1] — [scale.xyz, 0] then the world quat xyzw
//                                (the gaussian_splats_to_transforms layout)
//
// The quaternion math mirrors glam EXACTLY (mul_vec3 / mul_quat) so the CPU
// oracle test can hold the readback to < 1e-5 absolute deviation.

struct ExpandParams {
    // Total selected atoms (== threads doing real work).
    total: u32,
    // Number of entries in `draws`.
    draw_count: u32,
    _pad0: u32,
    _pad1: u32,
}

// Packed asset-local library atom — 64 B, uploaded once at pass creation.
// spectral packs the 8 pair-averaged f16 bins two-per-u32 (lo = bin 2j,
// hi = bin 2j+1), exactly the `gaussian_splat_to_gpu_full` binning.
struct LibraryAtom {
    // xyz = asset-local position, w = opacity 0..1
    pos_opacity: vec4<f32>,
    // xyz = scale, w = 0
    scale: vec4<f32>,
    // decoded rotation quaternion, xyzw
    quat: vec4<f32>,
    // 8 spectral bins as packed f16 pairs
    spectral: vec4<u32>,
}

// One selected draw unit — 32 B POD, re-uploaded per encode.
struct ExpandDraw {
    // First output slot (prefix sum over emission order — disjoint slots).
    atom_offset: u32,
    atom_count: u32,
    // Offset of this unit's slice in the concatenated atom index list.
    index_offset: u32,
    // Offset of the owning asset's atoms in lib_atoms (indices are asset-local).
    atom_base: u32,
    // Index into instance_xforms (×2 vec4 per instance).
    instance: u32,
    // LOD crossfade multiplier applied to opacity.
    opacity_scale: f32,
    _pad0: u32,
    _pad1: u32,
}

// Must match `crate::gpu::splat_buffer::GpuSplatFull` (80 B) and the frozen
// `tile_assign.wgsl` declaration byte-for-byte.
struct GpuSplatFull {
    position_depth: vec4<f32>,
    conic: vec3<f32>,
    _pad0: f32,
    opacity_color: vec4<f32>,
    spectral0: vec4<f32>,
    spectral1: vec4<f32>,
}

// Per-instance visual parameters (48 B, matches `InstanceVisualParams`):
// asset-local componentwise scale (positions AND extents, applied BEFORE the
// instance quat), an opacity multiplier, and the 8-bin spectral filter split
// across two vec4s (f0 = bins 0..3, f1 = bins 4..7). Identity entries are
// bit-transparent: `x * 1.0 == x` for every finite IEEE input, so the math
// below runs unconditionally — no branch.
struct InstanceVisual {
    scale: vec3<f32>,
    opacity_scale: f32,
    f0: vec4<f32>,
    f1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: ExpandParams;
@group(0) @binding(1) var<storage, read> lib_atoms: array<LibraryAtom>;
@group(0) @binding(2) var<storage, read> atom_indices: array<u32>;
@group(0) @binding(3) var<storage, read> draws: array<ExpandDraw>;
// Two vec4 per instance: [position.xyz, 0] then [quat x, y, z, w].
@group(0) @binding(4) var<storage, read> instance_xforms: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_splats: array<GpuSplatFull>;
@group(0) @binding(6) var<storage, read_write> out_transforms: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> visuals: array<InstanceVisual>;

// glam Quat::mul_quat (xyzw), instance quat `a` applied AFTER atom quat `b`.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// glam Quat::mul_vec3: v' = v(w² − b·b) + b(2 v·b) + (b × v)(2w).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    return v * (q.w * q.w - dot(b, b)) + b * (dot(v, b) * 2.0) + cross(b, v) * (q.w * 2.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }

    // Binary search: the LAST draw with atom_offset <= i owns this thread
    // (offsets are a strictly increasing prefix sum over non-empty draws).
    var lo = 0u;
    var hi = params.draw_count - 1u;
    loop {
        if lo >= hi { break; }
        let mid = (lo + hi + 1u) / 2u;
        if draws[mid].atom_offset <= i {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    let d = draws[lo];
    let local = i - d.atom_offset;

    let atom = lib_atoms[d.atom_base + atom_indices[d.index_offset + local]];
    let inst_pos = instance_xforms[d.instance * 2u].xyz;
    let inst_quat = instance_xforms[d.instance * 2u + 1u];
    let vis = visuals[d.instance];

    // Visual scale is asset-local: applied to the LOCAL position (and the
    // extents below) BEFORE the instance quat.
    let local_pos = atom.pos_opacity.xyz * vis.scale;
    let world_pos = quat_rotate(inst_quat, local_pos) + inst_pos;
    let world_quat = quat_mul(inst_quat, atom.quat);

    // Unpack the 8 f16 bins (exact f16→f32; bit-equal to the host round-trip).
    let s0 = unpack2x16float(atom.spectral.x);
    let s1 = unpack2x16float(atom.spectral.y);
    let s2 = unpack2x16float(atom.spectral.z);
    let s3 = unpack2x16float(atom.spectral.w);

    var splat: GpuSplatFull;
    // w (view depth) and conic are DEVICE-FILLED by the frozen tile_assign.
    splat.position_depth = vec4<f32>(world_pos, 0.0);
    splat.conic = vec3<f32>(0.0, 0.0, 0.0);
    splat._pad0 = 0.0;
    splat.opacity_color =
        vec4<f32>(0.0, 0.0, 0.0, atom.pos_opacity.w * d.opacity_scale * vis.opacity_scale);
    splat.spectral0 = vec4<f32>(s0, s1) * vis.f0;
    splat.spectral1 = vec4<f32>(s2, s3) * vis.f1;
    out_splats[i] = splat;

    out_transforms[i * 2u] = vec4<f32>(atom.scale.xyz * vis.scale, 0.0);
    out_transforms[i * 2u + 1u] = world_quat;
}
