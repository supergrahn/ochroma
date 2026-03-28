//! Integration tests for GLTF import.
//!
//! NOTE: GLTF import produces REFERENCE QUALITY splat clouds, not production quality.

use std::path::Path;
use vox_core::spectral::rgb_to_spectral;
use vox_data::gltf_import::{import_gltf, ImportError};

#[test]
fn test_rgb_to_spectral_coloured_input_nonzero() {
    let spectral = rgb_to_spectral(0.8, 0.5, 0.3);
    let any_nonzero = spectral.iter().any(|&v| v != 0);
    assert!(any_nonzero, "spectral bands should be non-zero for coloured input");
}

#[test]
fn test_rgb_to_spectral_red_high_bands() {
    let spectral = rgb_to_spectral(1.0, 0.0, 0.0);
    let band6 = half::f16::from_bits(spectral[6]).to_f32();
    let band7 = half::f16::from_bits(spectral[7]).to_f32();
    assert!(band6 > 0.9, "red should have high band 6, got {band6}");
    assert!(band7 > 0.5, "red should have high band 7, got {band7}");
}

#[test]
fn test_rgb_to_spectral_green_high_band4() {
    let spectral = rgb_to_spectral(0.0, 1.0, 0.0);
    let band4 = half::f16::from_bits(spectral[4]).to_f32();
    assert!(band4 > 0.9, "green should have high band 4, got {band4}");
}

#[test]
fn test_import_error_display_io() {
    let err = ImportError::IoError("disk full".to_string());
    assert_eq!(format!("{err}"), "IO error: disk full");
}

#[test]
fn test_import_error_display_parse() {
    let err = ImportError::ParseError("unexpected token".to_string());
    assert_eq!(format!("{err}"), "Parse error: unexpected token");
}

#[test]
fn test_import_error_display_no_meshes() {
    let err = ImportError::NoMeshes;
    assert_eq!(format!("{err}"), "No meshes found in GLTF file");
}

#[test]
fn test_import_nonexistent_file() {
    let result = import_gltf(Path::new("/tmp/does_not_exist_gltf_test.glb"));
    assert!(result.is_err());
}

/// Test importing a real .glb file if available on the system.
/// This test is ignored by default — run with `cargo test -- --ignored` if you have the file.
#[test]
fn test_import_real_glb_boombox() {
    let path = Path::new("/home/tomespen/usd-26.03/resources/Geometry/boombox.glb");
    if !path.exists() {
        eprintln!("Skipping: boombox.glb not found at {}", path.display());
        return;
    }

    let result = import_gltf(path).expect("should import boombox.glb");
    assert!(result.mesh_count > 0, "should have at least one mesh");
    assert!(result.triangle_count > 0, "should have triangles");
    assert!(result.vertex_count > 0, "should have vertices");
    assert!(!result.splats.is_empty(), "should produce splats");

    println!(
        "boombox.glb: {} meshes, {} tris, {} verts -> {} splats",
        result.mesh_count, result.triangle_count, result.vertex_count, result.splats.len()
    );
}
