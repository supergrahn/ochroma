//! Round-trip tests across the VXM writer/reader seam.
//!
//! These tests exercise the WHOLE seam that ochroma-tools depends on: a file is
//! WRITTEN via the V3 writer (the ochroma-tools code path) and then READ BACK
//! via the canonical loader (`VxmFile::read`) and the import pipeline
//! (`import_asset`). Before the version-dispatch fix, `VxmFile::read` rejected
//! anything but version 1, so v3 (and v2) files were unloadable.

use glam::Quat;
use uuid::Uuid;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{MaterialType, VxmError, VxmFile, VxmFileV3, VxmHeader};
use vox_data::vxm_v2::{GaussianSplatV2, VxmFileV2, VxmHeaderV2};
use vox_data::{import_asset, ImportSettings};

fn band16(values: [f32; 16]) -> [u16; 16] {
    std::array::from_fn(|i| half::f16::from_f32(values[i]).to_bits())
}

/// V3 (ochroma-tools) writer -> canonical VxmFile::read.
#[test]
fn v3_write_then_vxmfile_read_recovers_positions() {
    let splats = vec![
        GaussianSplat::volume([1.5, -2.0, 3.25], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, band16([0.3; 16])),
        GaussianSplat::volume([10.0, 20.0, 30.0], [0.2, 0.2, 0.2], Quat::IDENTITY, 255, band16([0.7; 16])),
        GaussianSplat::volume([-5.0, 0.0, 7.0], [0.05, 0.05, 0.05], Quat::IDENTITY, 128, band16([0.1; 16])),
    ];
    let original_positions: Vec<[f32; 3]> = splats.iter().map(|s| s.position()).collect();

    let v3 = VxmFileV3 {
        splats: splats.clone(),
        material_ids: vec![3, 7, 1],
        spectral_level: 1,
    };
    let mut buf = Vec::new();
    v3.write(&mut buf).expect("v3 write should succeed");

    // Read back via the CANONICAL loader that the game / import pipeline use.
    let loaded = VxmFile::read(&buf[..]).expect("VxmFile::read must accept v3 output");

    assert_eq!(loaded.header.version, 3, "header version preserved");
    assert!(loaded.header.splat_count > 0, "splat_count must be > 0");
    assert_eq!(loaded.header.splat_count, 3);
    assert_eq!(loaded.splats.len(), 3);
    for (i, (got, want)) in loaded.splats.iter().zip(&original_positions).enumerate() {
        assert_eq!(got.position(), *want, "splat {i} position mismatch");
    }
    // Spectral survives the round trip (sanity on splat payload, not just count).
    assert!((loaded.splats[1].spectral_f32(0) - 0.7).abs() < 1e-2,
        "spectral band 0 of splat 1 = {}", loaded.splats[1].spectral_f32(0));
}

/// V3 (ochroma-tools) writer -> import pipeline (`import_asset` on a .vxm path).
#[test]
fn v3_write_then_import_pipeline_loads_splats() {
    let splats = vec![
        GaussianSplat::volume([2.0, 4.0, 6.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, band16([0.5; 16])),
        GaussianSplat::volume([8.0, 16.0, 24.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, band16([0.5; 16])),
    ];
    let want_first = splats[0].position();

    let v3 = VxmFileV3 { splats, material_ids: vec![], spectral_level: 1 };

    let dir = std::env::temp_dir().join("ochroma_vxm_v3_import_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scene_v3.vxm");
    {
        let file = std::fs::File::create(&path).unwrap();
        v3.write(std::io::BufWriter::new(file)).unwrap();
    }

    let result = import_asset(&path, &ImportSettings::default())
        .expect("import pipeline must load v3 .vxm written by ochroma-tools");

    assert!(!result.splats.is_empty(), "import produced no splats from v3 file");
    assert_eq!(result.splats.len(), 2);
    assert_eq!(result.splats[0].position(), want_first);
    // Collision box is computed from real splat positions.
    let (mn, mx) = result.collision_box.expect("collision box expected");
    assert_eq!(mn, [2.0, 4.0, 6.0]);
    assert_eq!(mx, [8.0, 16.0, 24.0]);

    std::fs::remove_dir_all(&dir).ok();
}

/// V1 writer -> canonical VxmFile::read (must still work — backward compat).
#[test]
fn v1_write_then_vxmfile_read_still_works() {
    let uuid = Uuid::new_v4();
    let splats = vec![GaussianSplat::volume(
        [1.0, 2.0, 3.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 255, band16([0.4; 16]),
    )];
    let file = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete),
        splats,
    };
    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loaded = VxmFile::read(&buf[..]).unwrap();
    assert_eq!(loaded.header.version, 1);
    assert_eq!(loaded.splats.len(), 1);
    assert_eq!(loaded.splats[0].position(), [1.0, 2.0, 3.0]);
}

/// V2 writer -> canonical VxmFile::read (upcast 52-byte splats to 96-byte).
#[test]
fn v2_write_then_vxmfile_read_upcasts() {
    let uuid = Uuid::new_v4();
    let v2_splats = vec![
        GaussianSplatV2 {
            position: [3.0, 6.0, 9.0],
            scale: [0.1, 0.2, 0.3],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            semantic_zone: 0,
            entity_id: 0,
            spectral: [half::f16::from_f32(0.6).to_bits(); 8],
        },
        GaussianSplatV2 {
            position: [-1.0, -2.0, -3.0],
            scale: [0.05, 0.05, 0.05],
            rotation: [0, 0, 0, 32767],
            opacity: 128,
            semantic_zone: 1,
            entity_id: 42,
            spectral: [half::f16::from_f32(0.2).to_bits(); 8],
        },
    ];
    let v2 = VxmFileV2 {
        header: VxmHeaderV2::new(uuid, v2_splats.len() as u32),
        splats: v2_splats,
    };
    let mut buf = Vec::new();
    v2.write(&mut buf).unwrap();

    let loaded = VxmFile::read(&buf[..]).expect("VxmFile::read must accept v2 output");
    assert_eq!(loaded.header.version, 2, "version reported as 2");
    assert_eq!(loaded.splats.len(), 2);
    assert_eq!(loaded.splats[0].position(), [3.0, 6.0, 9.0]);
    assert_eq!(loaded.splats[1].position(), [-1.0, -2.0, -3.0]);
    assert_eq!(loaded.splats[0].opacity(), 200);
    // First 8 spectral bands carried over from the v2 8-band layout.
    assert!((loaded.splats[0].spectral_f32(0) - 0.6).abs() < 1e-2,
        "upcast spectral band 0 = {}", loaded.splats[0].spectral_f32(0));
    // Scales carried over.
    assert_eq!(loaded.splats[0].scales(), [0.1, 0.2, 0.3]);
}

/// Garbage / unknown version is still rejected cleanly.
#[test]
fn unknown_version_rejected() {
    // Build a 64-byte header with a bogus version (99) but valid magic.
    let mut header = VxmHeader::new(Uuid::new_v4(), 0, MaterialType::Generic);
    header.version = 99;
    let mut buf = bytemuck::bytes_of(&header).to_vec();
    buf.extend_from_slice(&0u64.to_le_bytes()); // empty compressed block size
    let err = VxmFile::read(&buf[..]).err().expect("unknown version must error");
    assert!(matches!(err, VxmError::UnsupportedVersion(99)),
        "expected UnsupportedVersion(99), got {err:?}");
}
