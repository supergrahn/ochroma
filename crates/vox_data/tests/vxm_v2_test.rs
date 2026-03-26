use std::io::Cursor;
use uuid::Uuid;
use vox_data::vxm_v2::{GaussianSplatV2, VxmFileV2, VxmHeaderV2};

#[test]
fn gaussian_splat_v2_size() {
    assert_eq!(std::mem::size_of::<GaussianSplatV2>(), 52);
}

#[test]
fn vxm_header_v2_size() {
    assert_eq!(std::mem::size_of::<VxmHeaderV2>(), 64);
}

#[test]
fn round_trip_with_entity_id() {
    let uuid = Uuid::new_v4();
    let splats = vec![
        GaussianSplatV2 {
            position: [1.0, 2.0, 3.0],
            scale: [0.1, 0.1, 0.1],
            rotation: [0, 0, 0, 32767],
            opacity: 200,
            semantic_zone: 3,
            entity_id: 42,
            spectral: [100, 200, 300, 400, 500, 600, 700, 800],
        },
        GaussianSplatV2 {
            position: [4.0, 5.0, 6.0],
            scale: [0.2, 0.2, 0.2],
            rotation: [100, 200, 300, 32000],
            opacity: 128,
            semantic_zone: 1,
            entity_id: 9999,
            spectral: [10, 20, 30, 40, 50, 60, 70, 80],
        },
    ];

    let header = VxmHeaderV2::new(uuid, splats.len() as u32);
    let file = VxmFileV2 { header, splats };

    let mut buf = Vec::new();
    file.write(Cursor::new(&mut buf)).expect("write failed");

    let loaded = VxmFileV2::read(Cursor::new(&buf)).expect("read failed");

    assert_eq!(loaded.splats.len(), 2);
    assert_eq!(loaded.splats[0].entity_id, 42);
    assert_eq!(loaded.splats[1].entity_id, 9999);
    assert_eq!(loaded.splats[0].semantic_zone, 3);
    assert_eq!(loaded.splats[1].semantic_zone, 1);
    assert_eq!(loaded.header.uuid(), uuid);
}
