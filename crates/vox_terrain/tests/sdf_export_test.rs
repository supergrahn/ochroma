use vox_terrain::volume::TerrainVolume;

#[test]
fn to_sdf_buffer_length_matches_volume() {
    let vol = TerrainVolume::new(4, 4, 4, 1.0);
    let buf = vol.to_sdf_buffer();
    assert_eq!(buf.len(), 4 * 4 * 4);
}

#[test]
fn to_sdf_buffer_default_is_air() {
    let vol = TerrainVolume::new(4, 4, 4, 1.0);
    let buf = vol.to_sdf_buffer();
    assert!(buf.iter().all(|&v| v > 0.0), "default volume is all air");
}

#[test]
fn to_sdf_buffer_solid_voxel_is_negative() {
    let mut vol = TerrainVolume::new(4, 4, 4, 1.0);
    vol.set(2, 2, 2, -1.0);
    let buf = vol.to_sdf_buffer();
    // index = z * size_x * size_y + y * size_x + x = 2*16 + 2*4 + 2 = 42
    assert!(buf[42] < 0.0, "solid voxel must be negative in flat buffer");
}

#[test]
fn to_sdf_metadata_matches_volume() {
    let vol = TerrainVolume::new(8, 4, 6, 0.5);
    let (sx, sy, sz, vs) = vol.sdf_metadata();
    assert_eq!(sx, 8);
    assert_eq!(sy, 4);
    assert_eq!(sz, 6);
    assert!((vs - 0.5).abs() < 1e-6);
}
