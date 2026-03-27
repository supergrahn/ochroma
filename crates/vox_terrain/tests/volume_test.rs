use vox_terrain::volume::*;

#[test]
fn empty_volume_is_all_air() {
    let vol = TerrainVolume::new(16, 16, 16, 1.0);
    assert_eq!(vol.solid_count(), 0);
}

#[test]
fn add_sphere_creates_solid() {
    let mut vol = TerrainVolume::new(32, 32, 32, 1.0);
    sculpt::add_sphere(&mut vol, [0.0, 0.0, 0.0], 5.0, 0);
    assert!(vol.solid_count() > 100, "Sphere should create solid voxels: {}", vol.solid_count());
}

#[test]
fn remove_sphere_creates_cave() {
    let mut vol = TerrainVolume::new(32, 32, 32, 1.0);
    sculpt::add_ground_plane(&mut vol, 8.0, 0); // fill lower half
    let solid_before = vol.solid_count();
    sculpt::remove_sphere(&mut vol, [0.0, 4.0, 0.0], 4.0);
    let solid_after = vol.solid_count();
    assert!(solid_after < solid_before, "Cave should remove solid: {} -> {}", solid_before, solid_after);
}

#[test]
fn cliff_has_overhang() {
    let mut vol = TerrainVolume::new(32, 32, 32, 1.0);
    sculpt::add_cliff(&mut vol, [0.0, -8.0, 0.0], 12.0, 5.0, 6.0, 0);

    // Check that there's solid above air (overhang)
    // At the overhang position, top should be solid while below-overhang should be air
    let has_overhang = (0..vol.size_z).any(|z| {
        (0..vol.size_x).any(|x| {
            // Find a column where solid is above air
            let mut found_air = false;
            let mut found_solid_above_air = false;
            for y in 0..vol.size_y {
                let v = vol.get(x, y, z);
                if v > 0.0 { found_air = true; }
                if found_air && v <= 0.0 { found_solid_above_air = true; break; }
            }
            found_solid_above_air
        })
    });
    assert!(has_overhang, "Cliff should have an overhang");
}

#[test]
fn cave_creates_through_tunnel() {
    let mut vol = TerrainVolume::new(32, 32, 32, 1.0);
    // Fill everything solid first
    for i in 0..vol.data.len() { vol.data[i] = -1.0; }
    // Carve a cave
    sculpt::add_cave(&mut vol, [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0], 3.0);
    // The center of the cave should be air
    let center_val = vol.sample_world(0.0, 0.0, 0.0);
    assert!(center_val > 0.0, "Cave center should be air, got {}", center_val);
}

#[test]
fn arch_creates_solid() {
    let mut vol = TerrainVolume::new(32, 32, 32, 1.0);
    sculpt::add_arch(&mut vol, [0.0, -8.0, 0.0], 10.0, 8.0, 2.0, 0);
    assert!(vol.solid_count() > 50, "Arch should create solid voxels");
}

#[test]
fn volume_to_splats_produces_surface_only() {
    let mut vol = TerrainVolume::new(16, 16, 16, 1.0);
    sculpt::add_sphere(&mut vol, [0.0, 0.0, 0.0], 5.0, 0);
    let materials = default_volume_materials();
    let splats = volume_to_splats(&vol, &materials, 42);
    assert!(!splats.is_empty(), "Should produce splats on the surface");
    // Surface splats should be fewer than total solid voxels
    assert!(splats.len() < vol.solid_count(), "Splats should only be on surface, not interior");
}

#[test]
fn demo_volume_has_all_features() {
    let vol = generate_demo_volume(42);
    assert!(vol.solid_count() > 1000, "Demo should have significant solid");
    let surface = vol.surface_count();
    assert!(surface > 200, "Should have visible surface: {}", surface);
}

#[test]
fn gradient_points_outward() {
    let mut vol = TerrainVolume::new(32, 32, 32, 1.0);
    sculpt::add_sphere(&mut vol, [0.0, 0.0, 0.0], 5.0, 0);
    // Gradient at a surface point should point outward (away from center)
    let (cx, cy, cz) = vol.world_to_voxel(5.0, 0.0, 0.0); // surface of sphere in +X
    let g = vol.gradient(cx, cy, cz);
    assert!(g[0] > 0.0, "Gradient should point outward in +X: {:?}", g);
}

#[test]
fn materials_assigned_correctly() {
    let mut vol = TerrainVolume::new(16, 16, 16, 1.0);
    sculpt::add_sphere(&mut vol, [0.0, 0.0, 0.0], 4.0, 3); // material 3 = sand
    let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
    assert_eq!(vol.get_material(cx, cy, cz), 3);
}
