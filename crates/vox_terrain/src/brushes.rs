//! Terrain sculpting brushes for the editor.
//!
//! In-editor brush tools for sculpting SDF terrain volumes.

use glam::Vec3;

use crate::volume::TerrainVolume;

#[derive(Debug, Clone, Copy)]
pub enum BrushType {
    Raise,
    Lower,
    Smooth,
    Flatten { target_height: f32 },
    Paint { material: u8 },
    Erode,
}

#[derive(Debug, Clone)]
pub struct TerrainBrush {
    pub brush_type: BrushType,
    pub radius: f32,
    pub strength: f32,
    pub falloff: BrushFalloff,
}

#[derive(Debug, Clone, Copy)]
pub enum BrushFalloff {
    Linear,
    Smooth,
    Sharp,
}

impl TerrainBrush {
    pub fn new(brush_type: BrushType, radius: f32, strength: f32) -> Self {
        Self {
            brush_type,
            radius,
            strength,
            falloff: BrushFalloff::Smooth,
        }
    }

    /// Apply the brush to the terrain volume at the given world position.
    pub fn apply(&self, volume: &mut TerrainVolume, center: Vec3, dt: f32) {
        let r_voxels = (self.radius / volume.voxel_size).ceil() as i32 + 1;
        let (cx, cy, cz) = volume.world_to_voxel(center.x, center.y, center.z);

        for dz in -r_voxels..=r_voxels {
            for dy in -r_voxels..=r_voxels {
                for dx in -r_voxels..=r_voxels {
                    let x = (cx as i32 + dx).max(0) as usize;
                    let y = (cy as i32 + dy).max(0) as usize;
                    let z = (cz as i32 + dz).max(0) as usize;
                    if x >= volume.size_x || y >= volume.size_y || z >= volume.size_z {
                        continue;
                    }

                    let wp = volume.voxel_to_world(x, y, z);
                    let dist = ((wp[0] - center.x).powi(2)
                        + (wp[1] - center.y).powi(2)
                        + (wp[2] - center.z).powi(2))
                    .sqrt();
                    if dist > self.radius {
                        continue;
                    }

                    let falloff = self.compute_falloff(dist / self.radius);
                    let effect = self.strength * falloff * dt;

                    match self.brush_type {
                        BrushType::Raise => {
                            let v = volume.get(x, y, z);
                            volume.set(x, y, z, v - effect);
                        }
                        BrushType::Lower => {
                            let v = volume.get(x, y, z);
                            volume.set(x, y, z, v + effect);
                        }
                        BrushType::Smooth => {
                            let avg = self.neighbor_average(volume, x, y, z);
                            let v = volume.get(x, y, z);
                            volume.set(x, y, z, v + (avg - v) * effect * 0.5);
                        }
                        BrushType::Flatten { target_height } => {
                            let v = volume.get(x, y, z);
                            let target_sdf = wp[1] - target_height;
                            volume.set(x, y, z, v + (target_sdf - v) * effect);
                        }
                        BrushType::Paint { material } => {
                            if volume.get(x, y, z) <= 0.0 {
                                volume.set_material(x, y, z, material);
                            }
                        }
                        BrushType::Erode => {
                            let slope = self.compute_slope(volume, x, y, z);
                            let v = volume.get(x, y, z);
                            volume.set(x, y, z, v + effect * slope);
                        }
                    }
                }
            }
        }
    }

    fn compute_falloff(&self, t: f32) -> f32 {
        match self.falloff {
            BrushFalloff::Linear => 1.0 - t,
            BrushFalloff::Smooth => {
                let t = 1.0 - t;
                t * t * (3.0 - 2.0 * t)
            }
            BrushFalloff::Sharp => 1.0,
        }
    }

    fn neighbor_average(&self, vol: &TerrainVolume, x: usize, y: usize, z: usize) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for d in &[-1i32, 0, 1] {
            let nx = (x as i32 + d).max(0) as usize;
            let ny = (y as i32 + d).max(0) as usize;
            let nz = (z as i32 + d).max(0) as usize;
            if nx < vol.size_x && ny < vol.size_y && nz < vol.size_z {
                sum += vol.get(nx, ny, nz);
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    fn compute_slope(&self, vol: &TerrainVolume, x: usize, y: usize, z: usize) -> f32 {
        let g = vol.gradient(x, y, z);
        (1.0 - g[1].abs()).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::TerrainVolume;

    fn make_volume() -> TerrainVolume {
        let mut vol = TerrainVolume::new(16, 16, 16, 1.0);
        // Fill bottom half as solid (negative SDF), top as air (positive)
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let wp = vol.voxel_to_world(x, y, z);
                    let sdf = wp[1]; // below y=0 is solid
                    vol.set(x, y, z, sdf);
                }
            }
        }
        vol
    }

    #[test]
    fn test_raise_increases_solidity() {
        let mut vol = make_volume();
        let center = Vec3::new(0.0, 0.0, 0.0);
        let (cx, cy, cz) = vol.world_to_voxel(center.x, center.y, center.z);
        let before = vol.get(cx, cy, cz);

        let brush = TerrainBrush::new(BrushType::Raise, 3.0, 1.0);
        brush.apply(&mut vol, center, 1.0);

        let after = vol.get(cx, cy, cz);
        // Raise decreases SDF (more solid)
        assert!(after < before, "Raise should decrease SDF: before={}, after={}", before, after);
    }

    #[test]
    fn test_lower_decreases_solidity() {
        let mut vol = make_volume();
        let center = Vec3::new(0.0, 0.0, 0.0);
        let (cx, cy, cz) = vol.world_to_voxel(center.x, center.y, center.z);
        let before = vol.get(cx, cy, cz);

        let brush = TerrainBrush::new(BrushType::Lower, 3.0, 1.0);
        brush.apply(&mut vol, center, 1.0);

        let after = vol.get(cx, cy, cz);
        // Lower increases SDF (more air)
        assert!(after > before, "Lower should increase SDF: before={}, after={}", before, after);
    }

    #[test]
    fn test_smooth_reduces_variation() {
        let mut vol = make_volume();
        // Create a spike
        let (cx, cy, cz) = vol.world_to_voxel(0.0, 0.0, 0.0);
        vol.set(cx, cy, cz, -5.0); // big solid spike

        let before = vol.get(cx, cy, cz);
        let brush = TerrainBrush::new(BrushType::Smooth, 3.0, 2.0);
        brush.apply(&mut vol, Vec3::new(0.0, 0.0, 0.0), 1.0);

        let after = vol.get(cx, cy, cz);
        // Smooth should move it toward neighbor average (less extreme)
        assert!(
            after.abs() < before.abs(),
            "Smooth should reduce extreme values: before={}, after={}",
            before,
            after
        );
    }

    #[test]
    fn test_flatten_moves_toward_target() {
        let mut vol = make_volume();
        let target = 2.0;
        let center = Vec3::new(0.0, 0.0, 0.0);
        let (cx, cy, cz) = vol.world_to_voxel(center.x, center.y, center.z);
        let before = vol.get(cx, cy, cz);
        let wp = vol.voxel_to_world(cx, cy, cz);
        let target_sdf = wp[1] - target;

        let brush = TerrainBrush::new(BrushType::Flatten { target_height: target }, 3.0, 1.0);
        brush.apply(&mut vol, center, 1.0);

        let after = vol.get(cx, cy, cz);
        // Should be closer to target_sdf than before
        let dist_before = (before - target_sdf).abs();
        let dist_after = (after - target_sdf).abs();
        assert!(
            dist_after <= dist_before,
            "Flatten should move toward target: dist_before={}, dist_after={}",
            dist_before,
            dist_after
        );
    }

    #[test]
    fn test_paint_sets_material() {
        let mut vol = make_volume();
        // Pick a point that is solid (below surface)
        let center = Vec3::new(0.0, -2.0, 0.0);
        let (cx, cy, cz) = vol.world_to_voxel(center.x, center.y, center.z);
        assert!(vol.get(cx, cy, cz) <= 0.0, "Should be solid");
        assert_eq!(vol.get_material(cx, cy, cz), 0);

        let brush = TerrainBrush::new(BrushType::Paint { material: 5 }, 3.0, 1.0);
        brush.apply(&mut vol, center, 1.0);

        assert_eq!(vol.get_material(cx, cy, cz), 5);
    }

    #[test]
    fn test_falloff_smooth_vs_linear() {
        let brush_smooth = TerrainBrush {
            brush_type: BrushType::Raise,
            radius: 5.0,
            strength: 1.0,
            falloff: BrushFalloff::Smooth,
        };
        let brush_linear = TerrainBrush {
            brush_type: BrushType::Raise,
            radius: 5.0,
            strength: 1.0,
            falloff: BrushFalloff::Linear,
        };

        // At t=0 (center) both should be 1.0
        assert!((brush_smooth.compute_falloff(0.0) - 1.0).abs() < 1e-6);
        assert!((brush_linear.compute_falloff(0.0) - 1.0).abs() < 1e-6);

        // At t=1 (edge) both should be 0.0
        assert!((brush_smooth.compute_falloff(1.0) - 0.0).abs() < 1e-6);
        assert!((brush_linear.compute_falloff(1.0) - 0.0).abs() < 1e-6);

        // At t=0.25, smooth and linear should differ
        let smooth_val = brush_smooth.compute_falloff(0.25);
        let linear_val = brush_linear.compute_falloff(0.25);
        assert!(
            (smooth_val - linear_val).abs() > 0.01,
            "Smooth and linear falloff should differ at t=0.25: smooth={}, linear={}",
            smooth_val,
            linear_val
        );
    }
}
