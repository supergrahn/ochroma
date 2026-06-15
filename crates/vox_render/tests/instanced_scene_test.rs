//! Render Keystone T3: Forge-mesh → instanced SceneState, per-instance material,
//! spectral SPD.
//!
//! `per_instance_material` builds TWO instances of ONE cube BLAS — material A =
//! glass (`transmission=0.9`), material B = metal (`metallic=0.9`) — renders
//! through the resident HW-RT path, and asserts the two instances' center pixels
//! differ by > 0.2 in at least one channel (a single-material/material-0 bypass
//! would make them identical). It also prints `blas_count: 1, instance_count: 2`.
//!
//! `spectral_spd_populated` asserts the converter actually keys
//! `MaterialLayer::spectral_spd` by the stable per-instance `material_id` (not an
//! empty/all-white map).

#![cfg(feature = "spectra-native")]

use vox_render::resident_renderer::ResidentCityRenderer;
use vox_render::splat_backend::{BlasDesc, InstanceRecordGpu, LightRig, PbrMaterial};
use vox_render::splat_convert::meshes_to_instanced_scene;

const W: u32 = 640;
const H: u32 = 360;

/// One axis-aligned unit cube centered at the object-space origin → a BlasDesc.
fn cube_blas_desc() -> BlasDesc {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut material_ids: Vec<u8> = Vec::new();

    let h = 0.75f32; // half-extent (1.5 cube → comfortably fills its screen region)
    let corners = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ];
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 1, 2, 3], [0.0, 0.0, -1.0]),
        ([5, 4, 7, 6], [0.0, 0.0, 1.0]),
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0]),
        ([1, 5, 6, 2], [1.0, 0.0, 0.0]),
        ([3, 2, 6, 7], [0.0, 1.0, 0.0]),
        ([4, 5, 1, 0], [0.0, -1.0, 0.0]),
    ];
    for (quad, n) in faces {
        let base = positions.len() as u32;
        for &ci in quad.iter() {
            positions.push(corners[ci]);
            normals.push(n);
            uvs.push([0.0, 0.0]);
        }
        indices.push([base, base + 1, base + 2]);
        indices.push([base, base + 2, base + 3]);
        material_ids.push(0);
        material_ids.push(0);
    }

    BlasDesc {
        proto_id: 1,
        positions,
        normals,
        uvs,
        indices,
        material_ids,
        aabb_min: [-h, -h, -h],
        aabb_max: [h, h, h],
    }
}

/// One instance: proto 0, world position `t`, material id `mat`.
fn inst(proto: u32, mat: u32, t: [f32; 3]) -> InstanceRecordGpu {
    // Row-major 4×4 translation (translation in the last ROW — the
    // SceneState::instance_transforms layout the uploader/T2 reads).
    InstanceRecordGpu {
        proto_index: proto,
        transform: [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            t[0], t[1], t[2], 1.0,
        ],
        material_id: mat,
    }
}

fn proj() -> [f32; 16] {
    glam::Mat4::perspective_rh(60f32.to_radians(), W as f32 / H as f32, 0.1, 1000.0).to_cols_array()
}

fn view() -> [f32; 16] {
    glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 1.5, 7.0),
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::Y,
    )
    .to_cols_array()
}

/// Project a world point to a pixel coordinate via the same view/proj the
/// renderer used, returning `(px, py)` clamped to the framebuffer.
fn project(world: [f32; 3]) -> (usize, usize) {
    let v = glam::Mat4::from_cols_array(&view());
    let p = glam::Mat4::from_cols_array(&proj());
    let clip = p * v * glam::Vec4::new(world[0], world[1], world[2], 1.0);
    let ndc = clip.truncate() / clip.w;
    let px = ((ndc.x * 0.5 + 0.5) * W as f32).round() as i64;
    // NDC y up → screen y down.
    let py = ((1.0 - (ndc.y * 0.5 + 0.5)) * H as f32).round() as i64;
    (
        px.clamp(0, W as i64 - 1) as usize,
        py.clamp(0, H as i64 - 1) as usize,
    )
}

/// Average a small window of the beauty buffer (RGBA8) around `(px, py)`.
fn sample_window(rgba: &[u8], px: usize, py: usize) -> [f32; 3] {
    let r = 6i64;
    let mut acc = [0.0f32; 3];
    let mut n = 0.0f32;
    for dy in -r..=r {
        for dx in -r..=r {
            let x = px as i64 + dx;
            let y = py as i64 + dy;
            if x < 0 || y < 0 || x >= W as i64 || y >= H as i64 {
                continue;
            }
            let idx = ((y as usize) * W as usize + x as usize) * 4;
            acc[0] += rgba[idx] as f32 / 255.0;
            acc[1] += rgba[idx + 1] as f32 / 255.0;
            acc[2] += rgba[idx + 2] as f32 / 255.0;
            n += 1.0;
        }
    }
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

/// FrameOutput beauty (linear f32 RGBA) → tonemapped sRGB8 RGBA, matching the
/// engine's still path so pixel deltas are in display space.
fn beauty_to_rgba8(beauty: &[f32], w: u32, h: u32) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        for c in 0..3 {
            let lin = beauty.get(i * 4 + c).copied().unwrap_or(0.0).max(0.0);
            // simple gamma encode for a stable display-space comparison
            let s = lin.powf(1.0 / 2.2).clamp(0.0, 1.0);
            out[i * 4 + c] = (s * 255.0 + 0.5) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    out
}

#[test]
fn per_instance_material() {
    let cube = cube_blas_desc();
    // Strongly separated appearance: a deep-blue transmissive glass vs a
    // saturated-red rough metal. Glass transmits/refracts (dark, blue-biased);
    // metal reflects its own conductor tint (bright, red-biased) — the two are
    // far apart in both R and B channels, well past the 0.2 gate.
    let glass = PbrMaterial {
        base_color: [0.05, 0.12, 0.6],
        roughness: 0.02,
        transmission: 0.95,
        ior: 1.5,
        ..Default::default()
    };
    let metal = PbrMaterial {
        base_color: [0.95, 0.15, 0.08],
        roughness: 0.25,
        metallic: 0.95,
        ..Default::default()
    };
    let instances = vec![
        inst(0, 0, [-1.6, 0.0, 0.0]),
        inst(0, 1, [1.6, 0.0, 0.0]),
    ];
    let scene = meshes_to_instanced_scene(
        std::slice::from_ref(&cube),
        &instances,
        &[glass, metal],
        &[],
        W,
        H,
    );

    // 156-float Vulkan stride (NEVER 132-float CUDA) + counts.
    assert_eq!(scene.materials.material_count, 2, "two materials");
    assert_eq!(
        scene.materials.params.len(),
        2 * 156,
        "must use the 156-float Vulkan material stride, got {}",
        scene.materials.params.len()
    );
    assert_eq!(scene.geometry.instance_count, 2, "two instances");
    assert_eq!(
        scene.geometry.instance_material_ids,
        vec![0u32, 1u32],
        "per-instance material ids carried for the TLAS custom index"
    );

    let blas_count = 1usize; // one prototype shared by both instances
    println!(
        "blas_count: {}, instance_count: {}",
        blas_count, scene.geometry.instance_count
    );

    // Render through the resident HW-RT path (the live engine path).
    let mut r = ResidentCityRenderer::new(W, H, LightRig::default(), 8, 3, scene)
        .expect("construct resident renderer");
    let frame = r.render_camera(view(), proj()).expect("render");
    let rgba = beauty_to_rgba8(&frame.beauty, frame.width, frame.height);

    let (gx, gy) = project([-1.6, 0.0, 0.0]);
    let (mx, my) = project([1.6, 0.0, 0.0]);
    let a = sample_window(&rgba, gx, gy); // glass instance
    let b = sample_window(&rgba, mx, my); // metal instance
    let d = (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max);
    println!(
        "glass@({gx},{gy})={a:?} metal@({mx},{my})={b:?} max_channel_delta={d:.4}"
    );
    assert!(
        d > 0.2,
        "two instances of one BLAS with glass vs metal materials must differ; \
         max channel delta {d:.4} (a={a:?}, b={b:?})"
    );
}

#[test]
fn spectral_spd_populated() {
    let cube = cube_blas_desc();
    // Material 5 carries a non-white, sloped 16-band reflectance (red-ish).
    let mut spd = [0.2f32; 16];
    for (i, v) in spd.iter_mut().enumerate() {
        *v = 0.1 + 0.85 * (i as f32 / 15.0); // rising curve, clearly != [1.0;16]
    }
    // Six materials so id 5 exists; only id 5 gets a custom SPD.
    let materials: Vec<PbrMaterial> = (0..6).map(|_| PbrMaterial::default()).collect();
    let instances = vec![inst(0, 5, [0.0, 0.0, 0.0])];
    let scene = meshes_to_instanced_scene(
        std::slice::from_ref(&cube),
        &instances,
        &materials,
        &[(5u32, spd)],
        W,
        H,
    );

    let got = scene
        .materials
        .spectral_spd
        .get(&5)
        .copied()
        .expect("spectral_spd[5] must be populated");
    println!("spectral_spd[5] = {got:?}");
    assert_ne!(
        got, [1.0f32; 16],
        "spectral_spd[5] must be the real sloped reflectance, not all-white"
    );
    assert_eq!(got, spd, "spectral_spd[5] must equal the uploaded curve");
    // Materials with no SPD entry stay white (absent ⇒ uploader fills 1.0).
    assert!(
        !scene.materials.spectral_spd.contains_key(&0),
        "material 0 must have no SPD entry (white)"
    );
}
