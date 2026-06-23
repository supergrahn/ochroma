use vox_terrain::heightmap::*;

#[test]
fn flat_terrain_constant_height() {
    let hm = Heightmap::flat(10, 10, 1.0, 5.0);
    assert!((hm.sample(5.0, 5.0) - 5.0).abs() < 0.01);
}

#[test]
fn bilinear_interpolation() {
    let mut data = vec![0.0f32; 4];
    data[0] = 0.0; // (0,0)
    data[1] = 10.0; // (1,0)
    data[2] = 0.0; // (0,1)
    data[3] = 10.0; // (1,1)
    let hm = Heightmap::from_data(2, 2, data, 1.0);
    let mid = hm.sample(0.5, 0.5);
    assert!((mid - 5.0).abs() < 0.1, "Midpoint should be ~5.0, got {}", mid);
}

#[test]
fn to_splats_produces_correct_count() {
    let hm = Heightmap::flat(10, 10, 1.0, 0.0);
    let zones = default_zones();
    let splats = hm.to_splats(&zones, 1);
    assert_eq!(splats.len(), 100); // 10x10 cells x 1 splat each
}

#[test]
fn material_zones_assign_by_height() {
    let mut data = vec![0.0f32; 4];
    data[0] = -1.0; // water
    data[1] = 0.0; // sand
    data[2] = 5.0; // grass
    data[3] = 20.0; // snow
    let hm = Heightmap::from_data(2, 2, data, 10.0);
    let zones = default_zones();
    let splats = hm.to_splats(&zones, 1);
    // Different heights should produce different spectral values
    assert_ne!(splats[0].spectral(), splats[3].spectral());
}

#[test]
fn normal_points_up_on_flat() {
    let hm = Heightmap::flat(10, 10, 1.0, 0.0);
    let n = hm.normal_at(5.0, 5.0);
    assert!(n[1] > 0.99, "Normal on flat should point up: {:?}", n);
}

#[test]
fn slope_is_zero_on_flat() {
    let hm = Heightmap::flat(10, 10, 1.0, 0.0);
    assert!(
        hm.slope_at(5.0, 5.0) < 1.0,
        "Flat terrain should have ~0 slope"
    );
}

#[test]
fn test_heightmap_area() {
    let hm = Heightmap::flat(100, 100, 2.0, 0.0);
    assert!((hm.area() - 40000.0).abs() < 0.1); // 200m x 200m
}

#[test]
fn generate_test_heightmap_has_variation() {
    let hm = generate_test_heightmap(64, 64, 1.0, 42);
    let min = hm.data.iter().cloned().fold(f32::MAX, f32::min);
    let max = hm.data.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        max - min > 1.0,
        "Generated terrain should have height variation"
    );
}

// --- determinism witnesses (the replay-hash moat) -----------------------
// These lock the BYTE-EXACT output of the parallelized gen paths. The hashes
// were captured from the pre-parallel serial implementation; any change that
// alters a produced float (reduction order, FMA contraction, a different
// arithmetic expression) changes the hash and fails here. Do NOT "rebaseline"
// without a deliberate, documented reason.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[test]
fn generate_test_heightmap_is_byte_exact() {
    let hm = generate_test_heightmap(2048, 2048, 1.0, 7);
    assert_eq!(hm.data.len(), 2048 * 2048);
    let hash = fnv1a(bytemuck::cast_slice(&hm.data));
    assert_eq!(
        hash, 0x7098e23a5f4380ef,
        "generate_test_heightmap output changed (determinism moat); got {hash:#018x}"
    );
}

#[test]
fn to_splats_is_byte_exact() {
    let hm = generate_test_heightmap(512, 512, 1.0, 7);
    let zones = default_zones();
    let splats = hm.to_splats(&zones, 1);
    assert_eq!(splats.len(), 512 * 512);
    let hash = fnv1a(bytemuck::cast_slice(&splats));
    assert_eq!(
        hash, 0x4f7e9e92b40f70fd,
        "to_splats output changed (determinism moat); got {hash:#018x}"
    );
}
