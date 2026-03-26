use std::io::Cursor;
use half::f16;
use uuid::Uuid;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{MaterialType, VxmFile, VxmHeader};

#[test]
fn write_to_file_read_back_identical() {
    let uuid = Uuid::new_v4();
    let splats: Vec<GaussianSplat> = (0..500)
        .map(|i| GaussianSplat {
            position: [i as f32 * 0.1, (i as f32 * 0.7).sin(), 0.0],
            scale: [0.05, 0.05, 0.05],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [f16::from_f32(0.5).to_bits(); 8],
        })
        .collect();

    let original = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete),
        splats: splats.clone(),
    };

    let mut buf = Vec::new();
    original.write(&mut buf).unwrap();

    // Verify compression reduced size
    let uncompressed_size = 64 + 52 * 500;
    assert!(
        buf.len() < uncompressed_size,
        "Expected compression: {} < {}",
        buf.len(),
        uncompressed_size
    );

    let loaded = VxmFile::read(Cursor::new(&buf)).unwrap();
    assert_eq!(loaded.header.uuid(), uuid);
    assert_eq!(loaded.header.splat_count, 500);

    for (i, (orig, load)) in splats.iter().zip(loaded.splats.iter()).enumerate() {
        assert_eq!(orig.position, load.position, "splat {} position", i);
        assert_eq!(orig.scale, load.scale, "splat {} scale", i);
        assert_eq!(orig.rotation, load.rotation, "splat {} rotation", i);
        assert_eq!(orig.opacity, load.opacity, "splat {} opacity", i);
        assert_eq!(orig.spectral, load.spectral, "splat {} spectral", i);
    }
}
