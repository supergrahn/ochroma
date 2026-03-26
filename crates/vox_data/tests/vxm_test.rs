use uuid::Uuid;
use vox_core::types::GaussianSplat;
use vox_data::vxm::{MaterialType, VxmFile, VxmHeader};

#[test]
fn header_is_64_bytes() {
    assert_eq!(std::mem::size_of::<VxmHeader>(), 64);
}

#[test]
fn round_trip_write_read() {
    let uuid = Uuid::new_v4();
    let splats = vec![GaussianSplat {
        position: [1.0, 2.0, 3.0],
        scale: [0.1, 0.1, 0.1],
        rotation: [0, 0, 0, 32767],
        opacity: 255,
        _pad: [0; 3],
        spectral: [15360; 8],
    }];

    let file = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Generic),
        splats: splats.clone(),
    };

    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loaded = VxmFile::read(&buf[..]).unwrap();
    assert_eq!(loaded.header.magic, *b"VXMF");
    assert_eq!(loaded.header.version, 1);
    assert_eq!(loaded.header.splat_count, 1);
    assert_eq!(loaded.splats.len(), 1);
    assert_eq!(loaded.splats[0].position, [1.0, 2.0, 3.0]);
    assert_eq!(loaded.splats[0].opacity, 255);
}

#[test]
fn round_trip_many_splats() {
    let uuid = Uuid::new_v4();
    let splats: Vec<GaussianSplat> = (0..1000)
        .map(|i| GaussianSplat {
            position: [i as f32, 0.0, 0.0],
            scale: [0.05, 0.05, 0.05],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            _pad: [0; 3],
            spectral: [15360; 8],
        })
        .collect();

    let file = VxmFile {
        header: VxmHeader::new(uuid, splats.len() as u32, MaterialType::Concrete),
        splats,
    };

    let mut buf = Vec::new();
    file.write(&mut buf).unwrap();

    let loaded = VxmFile::read(&buf[..]).unwrap();
    assert_eq!(loaded.splats.len(), 1000);
    assert_eq!(loaded.splats[999].position[0], 999.0);
}
