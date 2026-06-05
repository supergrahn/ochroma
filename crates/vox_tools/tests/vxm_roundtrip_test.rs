//! End-to-end VXM seam test for the `ochroma-tools` writer path.
//!
//! `ochroma-tools import --gltf ... --out scene.vxm` (see
//! `crates/vox_tools/src/ochroma_main.rs`) builds a `VxmFileV3` by:
//!   1. importing splats,
//!   2. classifying a per-splat spectral material id via `SpectralMaterialDb`,
//!   3. writing `VxmFileV3` (version 3).
//!
//! This test reproduces that exact construction and then proves the file is
//! LOADABLE by the canonical reader (`VxmFile::read`) and the import pipeline
//! (`import_asset`) — the round trip that was broken because the reader only
//! accepted version 1.

use glam::Quat;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{VxmFile, VxmFileV3};
use vox_data::{import_asset, ImportSettings, SpectralMaterialDb};

fn band16(values: [f32; 16]) -> [u16; 16] {
    std::array::from_fn(|i| half::f16::from_f32(values[i]).to_bits())
}

/// Replicates the material-id classification done in `ochroma_main.rs`.
fn classify_material_ids(splats: &[GaussianSplat]) -> Vec<u16> {
    splats
        .iter()
        .map(|s| {
            let spectral: [f32; 16] = std::array::from_fn(|b| s.spectral_f32(b));
            let mat = SpectralMaterialDb::classify(&spectral);
            SpectralMaterialDb::MATERIALS
                .iter()
                .position(|m| m.name == mat.name)
                .map_or(0u16, |i| (i + 1) as u16)
        })
        .collect()
}

#[test]
fn ochroma_tools_v3_path_writes_loadable_file() {
    // Build splats the way an import would, with real spectral content so the
    // classifier produces non-zero material ids (exercising the v3 mat section).
    let splats = vec![
        GaussianSplat::volume([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, band16([0.9; 16])),
        GaussianSplat::volume([4.0, 5.0, 6.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, band16([0.05; 16])),
        GaussianSplat::volume([-7.0, 8.0, -9.0], [0.1, 0.1, 0.1], Quat::IDENTITY, 200, band16([0.5; 16])),
    ];
    let want_positions: Vec<[f32; 3]> = splats.iter().map(|s| s.position()).collect();

    // --- ochroma-tools construction (mirrors ochroma_main.rs) ---
    let material_ids = classify_material_ids(&splats);
    let splat_count = splats.len();
    let vxm = VxmFileV3 {
        splats,
        material_ids: material_ids.clone(),
        spectral_level: 1,
    };

    let dir = std::env::temp_dir().join("ochroma_tools_vxm_roundtrip_test");
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("scene.vxm");
    {
        let file = std::fs::File::create(&out).unwrap();
        vxm.write(std::io::BufWriter::new(file)).unwrap();
    }

    // --- Canonical reader must accept the v3 output ---
    let bytes = std::fs::read(&out).unwrap();
    let loaded = VxmFile::read(&bytes[..])
        .expect("VxmFile::read must load ochroma-tools v3 output");
    assert_eq!(loaded.header.version, 3);
    assert!(loaded.header.splat_count > 0, "splat_count must be > 0");
    assert_eq!(loaded.header.splat_count as usize, splat_count);
    assert_eq!(loaded.splats.len(), splat_count);
    for (i, (got, want)) in loaded.splats.iter().zip(&want_positions).enumerate() {
        assert_eq!(got.position(), *want, "splat {i} position mismatch after reload");
    }

    // --- Import pipeline (what the game uses) must load it too ---
    let result = import_asset(&out, &ImportSettings::default())
        .expect("import_asset must load ochroma-tools v3 output");
    assert_eq!(result.splats.len(), splat_count);
    assert_eq!(result.splats[0].position(), want_positions[0]);

    std::fs::remove_dir_all(&dir).ok();
}
