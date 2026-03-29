use std::io::Write;
use vox_data::import_pipeline::{import_asset, ImportSettings};

fn write_binary_ply(path: &std::path::Path, count: usize) {
    use std::io::BufWriter;
    let mut f = BufWriter::new(std::fs::File::create(path).unwrap());
    write!(f, "ply\nformat binary_little_endian 1.0\nelement vertex {}\n", count).unwrap();
    write!(f, "property float x\nproperty float y\nproperty float z\n").unwrap();
    write!(f, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n").unwrap();
    write!(f, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n").unwrap();
    write!(f, "property float opacity\n").unwrap();
    write!(f, "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n").unwrap();
    write!(f, "end_header\n").unwrap();
    for i in 0..count {
        let x = (i as f32 * 0.1f32).to_le_bytes();
        let scale = (-2.3f32).to_le_bytes();
        let rot_w = 1.0f32.to_le_bytes();
        let rot_zero = 0.0f32.to_le_bytes();
        let opacity = 0.0f32.to_le_bytes();
        let color = 0.5f32.to_le_bytes();
        f.write_all(&x).unwrap();
        for _ in 0..2 { f.write_all(&0.0f32.to_le_bytes()).unwrap(); }
        for _ in 0..3 { f.write_all(&scale).unwrap(); }
        f.write_all(&rot_w).unwrap();
        for _ in 0..3 { f.write_all(&rot_zero).unwrap(); }
        f.write_all(&opacity).unwrap();
        for _ in 0..3 { f.write_all(&color).unwrap(); }
    }
}

#[test]
fn import_ply_produces_real_splats() {
    let dir = std::env::temp_dir().join("ochroma_import_test_real");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("real.ply");
    write_binary_ply(&path, 10);

    let settings = ImportSettings::default();
    let result = import_asset(&path, &settings).unwrap();
    assert_eq!(result.splats.len(), 10, "PLY import should produce one splat per vertex");
    assert!((result.splats[0].position[0]).abs() < 0.01, "first splat near x=0");
    let scale = result.splats[0].scale[0];
    assert!(scale > 0.05 && scale < 0.2,
        "scale should be ~exp(-2.3)≈0.1, not dummy 0.01, got {}", scale);

    std::fs::remove_dir_all(&dir).ok();
}

fn write_minimal_glb(path: &std::path::Path) {
    // Minimal GLB: JSON chunk + BIN chunk with 1 triangle (3 vertices)
    // POSITION accessor requires min/max for gltf crate validation
    let json = r#"{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteLength":36,"byteOffset":0},{"buffer":0,"byteLength":6,"byteOffset":36}],"buffers":[{"byteLength":42}]}"#;
    let json_bytes = json.as_bytes();
    let padded_json = ((json_bytes.len() + 3) / 4) * 4;
    // BIN: 3 vec3 positions + 3 u16 indices
    let mut bin = vec![0u8; 42];
    // v1: (1,0,0)
    bin[12..16].copy_from_slice(&1.0f32.to_le_bytes());
    // v2: (0,1,0)
    bin[28..32].copy_from_slice(&1.0f32.to_le_bytes());
    // indices: 0,1,2
    bin[36] = 0; bin[38] = 1; bin[40] = 2;
    let padded_bin = ((bin.len() + 3) / 4) * 4;
    let total = 12 + 8 + padded_json + 8 + padded_bin;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(padded_json as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json_bytes);
    out.resize(12 + 8 + padded_json, 0x20);
    out.extend_from_slice(&(padded_bin as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out.resize(total, 0);
    std::fs::write(path, &out).unwrap();
}

#[test]
fn import_gltf_produces_real_splats() {
    let dir = std::env::temp_dir().join("ochroma_gltf_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.glb");
    write_minimal_glb(&path);

    let settings = ImportSettings::default();
    let result = import_asset(&path, &settings).unwrap();
    assert!(!result.splats.is_empty(), "GLTF import should produce splats");
    // Splats should be within the triangle's bounds [0,1] on x and y
    for s in &result.splats {
        assert!(s.position[0] >= -0.01 && s.position[0] <= 1.01,
            "splat x={} out of triangle range", s.position[0]);
        assert!(s.position[1] >= -0.01 && s.position[1] <= 1.01,
            "splat y={} out of triangle range", s.position[1]);
    }
    std::fs::remove_dir_all(&dir).ok();
}

use vox_data::vxm::{VxmFile, VxmHeader, MaterialType};
use vox_core::types::GaussianSplat;
use uuid::Uuid;

#[test]
fn import_vxm_produces_exact_splats() {
    let dir = std::env::temp_dir().join("ochroma_vxm_import_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.vxm");

    // Write a real VXM with 5 known splats
    let file = VxmFile {
        header: VxmHeader::new(Uuid::new_v4(), 5, MaterialType::Generic),
        splats: (0..5).map(|i| GaussianSplat {
        position: [i as f32, 0.0, 0.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 16384],
        opacity: 200,
        _pad: [0; 3],
        spectral: [100, 200, 150, 100, 80, 60, 40, 20],
    }).collect(),
    };
    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();
    std::fs::write(&path, &buf).unwrap();

    let settings = ImportSettings::default();
    let result = import_asset(&path, &settings).unwrap();
    assert_eq!(result.splats.len(), 5, "VXM import should produce exactly 5 splats");
    assert!((result.splats[0].position[0]).abs() < 0.01, "first splat at x=0");
    assert!((result.splats[4].position[0] - 4.0).abs() < 0.01, "fifth splat at x=4");

    std::fs::remove_dir_all(&dir).ok();
}
