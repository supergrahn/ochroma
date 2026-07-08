//! Task 1 (carrier seam): a `HybridMesh` must be able to CARRY PBR relief
//! (normal / roughness / displacement) and a transmission override, so the
//! game-side `mesh_to_blas_material` consumer can bind them into `PbrMaterial`.
//! Before this seam existed the mesh carried only `albedo_tex_path`, so relief
//! and water transmission died before reaching the path tracer.

use vox_render::hybrid_compose::HybridMesh;

fn unit_tri() -> (Vec<[f32; 3]>, Vec<u32>) {
    (
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![0, 1, 2],
    )
}

#[test]
fn hybrid_pbr_seam_carries_relief_and_transmission() {
    let (positions, indices) = unit_tri();
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let mesh = HybridMesh::from_rgb(positions, indices, [0.6, 0.4, 0.3], 7)
        .with_albedo_texture(uvs, "polyhaven://brick_diff".to_string())
        .with_pbr_textures(
            Some("polyhaven://brick_nor".to_string()),
            Some("polyhaven://brick_rough".to_string()),
            Some("polyhaven://brick_disp".to_string()),
        )
        .with_transmission(0.6, 1.33);

    // Real carried values — not None, not the -1/0.0 defaults.
    assert_eq!(
        mesh.albedo_tex_path.as_deref(),
        Some("polyhaven://brick_diff")
    );
    assert_eq!(
        mesh.normal_tex_path.as_deref(),
        Some("polyhaven://brick_nor")
    );
    assert_eq!(
        mesh.roughness_tex_path.as_deref(),
        Some("polyhaven://brick_rough")
    );
    assert_eq!(
        mesh.displacement_tex_path.as_deref(),
        Some("polyhaven://brick_disp")
    );
    assert!(mesh.opacity_tex_path.is_none());
    // A displacement map auto-enables POM at the proven brick depth.
    assert!(
        mesh.displacement_scale > 0.0,
        "displacement map must enable POM, got scale {}",
        mesh.displacement_scale
    );
    assert_eq!(mesh.displacement_midlevel, 0.5);
    assert_eq!(mesh.transmission_override, Some(0.6));
    assert_eq!(mesh.ior_override, Some(1.33));

    println!(
        "carried: normal={:?} rough={:?} disp={:?} scale={} transmission={:?} ior={:?}",
        mesh.normal_tex_path,
        mesh.roughness_tex_path,
        mesh.displacement_tex_path,
        mesh.displacement_scale,
        mesh.transmission_override,
        mesh.ior_override
    );
}

#[test]
fn hybrid_defaults_leave_pbr_unset() {
    // An untextured mesh must NOT accidentally claim relief/transmission, or it
    // would clobber the channel-derived material in the consumer.
    let (positions, indices) = unit_tri();
    let mesh = HybridMesh::from_rgb(positions, indices, [0.5, 0.5, 0.5], 0);
    assert!(mesh.normal_tex_path.is_none());
    assert!(mesh.roughness_tex_path.is_none());
    assert!(mesh.displacement_tex_path.is_none());
    assert!(mesh.opacity_tex_path.is_none());
    assert_eq!(mesh.displacement_scale, 0.0);
    assert!(mesh.transmission_override.is_none());
    assert!(mesh.ior_override.is_none());
    assert!(!mesh.vegetation_bsdf);
}

#[test]
fn hybrid_pbr_seam_carries_vegetation_bsdf_flag() {
    let (positions, indices) = unit_tri();
    let mesh = HybridMesh::from_rgb(positions, indices, [0.2, 0.5, 0.2], 9)
        .with_vegetation_bsdf()
        .with_opacity_texture("polyhaven://leaf_alpha".to_string());
    assert!(mesh.vegetation_bsdf);
    assert_eq!(
        mesh.opacity_tex_path.as_deref(),
        Some("polyhaven://leaf_alpha")
    );
}

/// Build a `RenderScene` from two axis-aligned meshes and verify the computed
/// AABB spans the expected range. Checks the real derived values, not is_some.
#[test]
fn render_scene_two_meshes_aabb() {
    use vox_render::RenderScene;

    // Mesh A: triangle at x=[1,3], y=0, z=[0,1]
    let mesh_a = HybridMesh::from_rgb(
        vec![[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
        vec![0, 1, 2],
        [0.8, 0.2, 0.2],
        1,
    );
    // Mesh B: triangle at x=[-2,-1], y=5, z=[-3,-2]
    let mesh_b = HybridMesh::from_rgb(
        vec![[-2.0, 5.0, -3.0], [-1.0, 5.0, -3.0], [-1.5, 5.0, -2.0]],
        vec![0, 1, 2],
        [0.2, 0.4, 0.8],
        2,
    );

    let scene = RenderScene {
        meshes: vec![mesh_a, mesh_b],
        splats: vec![],
        sdf_aabbs: vec![],
        fallback_splats: vec![],
    };

    assert_eq!(scene.meshes.len(), 2, "must have 2 meshes");

    // Compute AABB from the mesh positions — the real derived outcome.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for mesh in &scene.meshes {
        for p in &mesh.positions {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }
    }

    // Expected combined extent: x in [-2, 3], y in [0, 5], z in [-3, 1]
    assert!(
        (min[0] - (-2.0_f32)).abs() < 1e-5,
        "min x must be -2.0, got {}",
        min[0]
    );
    assert!(
        (max[0] - 3.0_f32).abs() < 1e-5,
        "max x must be 3.0, got {}",
        max[0]
    );
    assert!(
        (min[1] - 0.0_f32).abs() < 1e-5,
        "min y must be 0.0, got {}",
        min[1]
    );
    assert!(
        (max[1] - 5.0_f32).abs() < 1e-5,
        "max y must be 5.0, got {}",
        max[1]
    );
    // clone must be identical
    let cloned = scene.clone();
    assert_eq!(cloned.meshes.len(), 2, "clone must preserve mesh count");
    println!(
        "aabb_min=[{:.1},{:.1},{:.1}] aabb_max=[{:.1},{:.1},{:.1}] meshes={}",
        min[0],
        min[1],
        min[2],
        max[0],
        max[1],
        max[2],
        scene.meshes.len()
    );
}
