//! Real-time SDF terrain deformation.
//!
//! Thin wrappers around `volume::sculpt` that present a world-space API
//! for runtime terrain carving and filling.
//!
//! After any deformation call, regenerate terrain splats by calling
//! `vox_terrain::volume_to_splats(&volume, &materials, seed)`.

use crate::volume::{sculpt, TerrainVolume};

/// Carve a sphere-shaped hole into the SDF terrain.
pub fn carve_sphere(volume: &mut TerrainVolume, center: [f32; 3], radius: f32) {
    sculpt::remove_sphere(volume, center, radius);
}

/// Fill a sphere-shaped region with solid terrain.
pub fn fill_sphere(volume: &mut TerrainVolume, center: [f32; 3], radius: f32, material: u8) {
    sculpt::add_sphere(volume, center, radius, material);
}

/// Carve a tunnel (capsule shape) between two world-space points.
pub fn carve_tunnel(volume: &mut TerrainVolume, start: [f32; 3], end: [f32; 3], radius: f32) {
    sculpt::add_cave(volume, start, end, radius);
}

/// Apply a spherical explosion deformation at `center` with `radius`.
pub fn apply_explosion(volume: &mut TerrainVolume, center: [f32; 3], radius: f32) {
    carve_sphere(volume, center, radius);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::TerrainVolume;

    fn make_solid_volume() -> TerrainVolume {
        let mut vol = TerrainVolume::new(16, 16, 16, 1.0);
        for z in 0..16usize {
            for y in 0..16usize {
                for x in 0..16usize {
                    vol.set(x, y, z, -1.0);
                }
            }
        }
        vol
    }

    #[test]
    fn carve_sphere_makes_center_air() {
        let mut vol = make_solid_volume();
        carve_sphere(&mut vol, [0.0, 0.0, 0.0], 2.0);
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let val = vol.get(cx, cy, cz);
        assert!(val > -1.0, "SDF at sphere center should increase after carving, got {}", val);
    }

    #[test]
    fn carve_sphere_does_not_affect_far_voxels() {
        let mut vol = make_solid_volume();
        carve_sphere(&mut vol, [0.0, 0.0, 0.0], 1.0);
        let (fx, fy, fz) = vol.world_to_voxel(6.0, 0.0, 0.0);
        let val = vol.get(fx, fy, fz);
        assert!(val < 0.0, "Far voxels should remain solid after a small carve, got {}", val);
    }

    #[test]
    fn fill_sphere_makes_region_solid() {
        let mut vol = TerrainVolume::new(16, 16, 16, 1.0); // starts as all-air (SDF = 1.0 by default)
        fill_sphere(&mut vol, [0.0, 0.0, 0.0], 2.0, 1);
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let val = vol.get(cx, cy, cz);
        assert!(val < 0.0, "SDF at sphere center should be solid (< 0) after fill, got {}", val);
    }

    #[test]
    fn carve_then_fill_restores_solid() {
        let mut vol = make_solid_volume();
        let center = [0.0f32, 0.0, 0.0];
        carve_sphere(&mut vol, center, 2.0);
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let after_carve = vol.get(cx, cy, cz);
        assert!(after_carve > -1.0, "carve should make center less solid");
        fill_sphere(&mut vol, center, 2.0, 0);
        let after_fill = vol.get(cx, cy, cz);
        assert!(after_fill < 0.0, "fill after carve should restore solid (< 0), got {}", after_fill);
    }

    #[test]
    fn apply_explosion_same_as_carve_sphere() {
        let mut vol_a = make_solid_volume();
        let mut vol_b = make_solid_volume();
        carve_sphere(&mut vol_a, [0.0, 0.0, 0.0], 3.0);
        apply_explosion(&mut vol_b, [0.0, 0.0, 0.0], 3.0);
        assert_eq!(vol_a.data, vol_b.data, "apply_explosion should produce identical result to carve_sphere");
    }

    #[test]
    fn carve_tunnel_connects_two_points() {
        let mut vol = make_solid_volume();
        carve_tunnel(&mut vol, [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0], 1.0);
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        let val = vol.get(cx, cy, cz);
        assert!(val > -1.0, "tunnel center should be air after carve_tunnel, got {}", val);
    }
}
