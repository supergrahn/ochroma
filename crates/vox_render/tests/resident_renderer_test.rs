//! Render Keystone T1: ResidentSceneRenderer reuse behavior.
//!
//! Proves the renderer is constructed ONCE and a camera-only frame (then an
//! identical-scene re-upload) reuses the GPU scene instead of rebuilding it.

#![cfg(feature = "spectra-native")]

use spectra_scene_state::{MaterialLayer, SceneState};
use vox_render::resident_renderer::ResidentSceneRenderer;
use vox_render::splat_backend::{LightRig, PbrMaterial, pack_vulkan_mesh_material};

/// Build a tiny 3-cube instanced scene with distinct transforms + materials.
/// Deterministic: same call yields byte-identical geometry/materials.
fn test_instanced_scene() -> SceneState {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut material_ids: Vec<u8> = Vec::new();

    let materials = [
        PbrMaterial {
            base_color: [0.6, 0.18, 0.12],
            roughness: 0.85,
            ..Default::default()
        },
        PbrMaterial {
            base_color: [0.6, 0.7, 0.85],
            roughness: 0.05,
            transmission: 0.85,
            ior: 1.5,
            ..Default::default()
        },
        PbrMaterial {
            base_color: [0.8, 0.8, 0.82],
            roughness: 0.15,
            metallic: 0.9,
            ..Default::default()
        },
    ];

    for (i, &x) in [-2.0f32, 0.0, 2.0].iter().enumerate() {
        push_cube(
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut indices,
            &mut material_ids,
            [x, 0.0, 0.0],
            1.0,
            i as u8,
        );
    }

    let mut scene = SceneState::new(640, 360);
    scene.geometry.vertex_count = positions.len();
    scene.geometry.triangle_count = indices.len();
    scene.geometry.positions = positions.iter().flat_map(|p| *p).collect();
    scene.geometry.normals = normals.iter().flat_map(|n| *n).collect();
    scene.geometry.uvs = uvs.iter().flat_map(|t| *t).collect();
    scene.geometry.indices = indices.iter().flat_map(|t| *t).collect();
    scene.geometry.material_ids = material_ids.iter().map(|&m| m as u32).collect();
    scene.geometry.instance_count = 1;
    scene.geometry.instance_transforms = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let mut params = Vec::new();
    for m in &materials {
        params.extend_from_slice(&pack_vulkan_mesh_material(*m));
    }
    scene.materials = MaterialLayer {
        params,
        spectral_spd: Default::default(),
        material_count: materials.len(),
    };

    scene.camera.view_matrix = view_a();
    scene.camera.fov_y_radians = 60f32.to_radians();
    scene.camera.width = 640;
    scene.camera.height = 360;
    scene.mark_geometry_changed();
    scene.mark_materials_changed();
    scene.mark_camera_changed();
    scene
}

#[test]
fn camera_only_frame_reuses_scene() {
    let scene = test_instanced_scene();
    let mut r =
        ResidentSceneRenderer::new(640, 360, LightRig::default(), 1, 2, scene).expect("construct");

    let f0 = r.render_camera(view_a(), proj()).expect("frame a");
    assert!(f0.width == 640 && f0.height == 360, "frame a dims");

    // Camera-only move — no set_scene.
    let f = r.render_camera(view_b(), proj()).expect("frame b");
    assert!(f.width == 640 && f.height == 360, "frame b dims");

    // Re-apply identical scene: must report reuse, not rebuild.
    let delta = r.set_scene(test_instanced_scene()).expect("set_scene");
    assert_eq!(
        delta.layers_rebuilt(),
        0,
        "identical scene must not rebuild layers (got {})",
        delta.layers_rebuilt()
    );
    assert!(delta.layers_reused() > 0, "expected reused layers, got 0");
}

/// Task 3 determinism witness: replaying the same `SceneDelta` log from scratch
/// must produce a byte-identical `download_resident_instances` buffer.
///
/// `drain_scene_deltas` only QUEUES the transform into the delta ring; the ring
/// is applied to the resident `g_instances` buffer by the flush. `render_camera`
/// performs that flush but also runs a full trace/present, which CUDA-faults on
/// this minimal 2-node scene. `flush_pending_deltas` runs the EXACT same flush
/// (ring drain → `apply_scene_delta` g_instances patch → IAS refit / legacy
/// fallback) with NO trace/present, so the witness reads the same patched buffer
/// without needing a render-complete scene.
#[cfg(feature = "spectra-native")]
#[test]
fn scene_delta_replay_exact() {
    use vox_scene::{NodeId, SceneDelta as GraphSceneDelta, SceneTransform};

    let scene = test_instanced_scene();
    // The scene has instance_count=1. Assign one NodeId to instance 0.
    let node_id = NodeId::from_raw(1).expect("node id 1 must be valid");

    // ── renderer 1 ──
    use vox_render::resident_renderer::FidelityTier;

    let mut renderer = ResidentSceneRenderer::new_with_tier(
        256,
        256,
        LightRig::default(),
        FidelityTier::Performance,
        None,
        scene.clone(),
    )
    .expect("renderer construction must succeed");

    renderer
        .reset_retained_mirror(std::iter::once(node_id))
        .expect("reset_retained_mirror must not fail with unique nodes");

    // Capture the instance buffer BEFORE the transform delta lands (pre-flush).
    // `drain_scene_deltas` below only QUEUES into the ring; the buffer is only
    // mutated by `flush_pending_deltas`, so this is the upload-time snapshot.
    let pre_buf = renderer
        .download_resident_instances()
        .expect("download must succeed before drain");

    // Build a transform delta and drain it (QUEUES into the ring; g_instances
    // is NOT yet patched).
    let delta = GraphSceneDelta::SetTransform {
        id: node_id,
        transform: SceneTransform::from_translation(1.0, 2.0, 3.0),
    };
    let mut deltas1 = vec![delta.clone()];
    renderer
        .drain_scene_deltas(&mut deltas1)
        .expect("transform-only drain must succeed");
    assert!(deltas1.is_empty(), "drain must clear the vec on success");

    // CRITICAL: flush_pending_deltas applies the queued ring into the GPU
    // g_instances buffer (the same `apply_scene_delta` patch render_camera runs,
    // minus the trace/present). Without it the ring has NOT been applied and the
    // download would be a stale pre-transform snapshot (false-positive pass).
    renderer
        .flush_pending_deltas()
        .expect("flush_pending_deltas must succeed after drain");

    let post_buf = renderer
        .download_resident_instances()
        .expect("download must succeed after flush_pending_deltas");

    // (a) The transform delta must have changed the buffer.
    assert_ne!(
        pre_buf, post_buf,
        "post-flush buffer must differ from pre-flush buffer — \
         transform delta did not land in the GPU resident buffer"
    );
    assert!(
        !post_buf.is_empty(),
        "instance buffer must be non-empty after scene upload"
    );

    // ── renderer 2 (replay) ──
    let mut renderer2 = ResidentSceneRenderer::new_with_tier(
        256,
        256,
        LightRig::default(),
        FidelityTier::Performance,
        None,
        scene.clone(),
    )
    .expect("second renderer construction must succeed");

    renderer2
        .reset_retained_mirror(std::iter::once(node_id))
        .expect("reset_retained_mirror must not fail on second renderer");

    let mut deltas2 = vec![delta];
    renderer2
        .drain_scene_deltas(&mut deltas2)
        .expect("replay drain must succeed");

    renderer2
        .flush_pending_deltas()
        .expect("replay flush_pending_deltas must succeed");

    let buf2 = renderer2
        .download_resident_instances()
        .expect("replay download must succeed");

    // (b) Replay of the same delta log must produce byte-identical output.
    assert_eq!(
        buf2.len(),
        post_buf.len(),
        "instance buffer lengths must match"
    );
    let replay_matches = post_buf == buf2;
    println!(
        "instance_buf[0]={} instance_buf[1]={} replay_matches={replay_matches}",
        post_buf[0],
        post_buf.get(1).copied().unwrap_or(0)
    );
    assert!(
        replay_matches,
        "replay of same SceneDelta log must produce byte-identical instance buffer"
    );
}

// ── geometry + camera helpers ──

fn push_cube(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<[u32; 3]>,
    material_ids: &mut Vec<u8>,
    center: [f32; 3],
    size: f32,
    mat_id: u8,
) {
    let h = size * 0.5;
    let c = center;
    let corners = [
        [c[0] - h, c[1] - h, c[2] - h],
        [c[0] + h, c[1] - h, c[2] - h],
        [c[0] + h, c[1] + h, c[2] - h],
        [c[0] - h, c[1] + h, c[2] - h],
        [c[0] - h, c[1] - h, c[2] + h],
        [c[0] + h, c[1] - h, c[2] + h],
        [c[0] + h, c[1] + h, c[2] + h],
        [c[0] - h, c[1] + h, c[2] + h],
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
        material_ids.push(mat_id);
        material_ids.push(mat_id);
    }
}

fn proj() -> [f32; 16] {
    glam::Mat4::perspective_rh(60f32.to_radians(), 640.0 / 360.0, 0.1, 1000.0).to_cols_array()
}

fn view_a() -> [f32; 16] {
    glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 3.0, 8.0),
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::Y,
    )
    .to_cols_array()
}

fn view_b() -> [f32; 16] {
    glam::Mat4::look_at_rh(
        glam::Vec3::new(3.0, 3.0, 7.0),
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::Y,
    )
    .to_cols_array()
}
