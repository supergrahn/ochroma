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
    assert_eq!(mesh.albedo_tex_path.as_deref(), Some("polyhaven://brick_diff"));
    assert_eq!(mesh.normal_tex_path.as_deref(), Some("polyhaven://brick_nor"));
    assert_eq!(
        mesh.roughness_tex_path.as_deref(),
        Some("polyhaven://brick_rough")
    );
    assert_eq!(
        mesh.displacement_tex_path.as_deref(),
        Some("polyhaven://brick_disp")
    );
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
    assert_eq!(mesh.displacement_scale, 0.0);
    assert!(mesh.transmission_override.is_none());
    assert!(mesh.ior_override.is_none());
}
