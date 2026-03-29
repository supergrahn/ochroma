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
