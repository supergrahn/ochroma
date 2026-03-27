use glam::Vec3;
use vox_core::types::GaussianSplat;
use half::f16;

/// Water surface properties.
#[derive(Debug, Clone)]
pub struct WaterSurface {
    pub center: Vec3,
    pub width: f32,
    pub depth: f32,
    pub water_level: f32,
    pub flow_direction: Vec3,
    pub flow_speed: f32,
    pub wave_amplitude: f32,
    pub wave_frequency: f32,
}

impl WaterSurface {
    pub fn river(center: Vec3, width: f32, length: f32, flow_speed: f32) -> Self {
        Self {
            center,
            width,
            depth: length,
            water_level: center.y,
            flow_direction: Vec3::new(1.0, 0.0, 0.0),
            flow_speed,
            wave_amplitude: 0.05,
            wave_frequency: 2.0,
        }
    }

    pub fn lake(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            width: radius * 2.0,
            depth: radius * 2.0,
            water_level: center.y,
            flow_direction: Vec3::ZERO,
            flow_speed: 0.0,
            wave_amplitude: 0.02,
            wave_frequency: 1.0,
        }
    }

    /// Generate water surface splats with wave animation offset.
    pub fn generate_splats(&self, time: f32) -> Vec<GaussianSplat> {
        let water_spd: [u16; 8] = [
            f16::from_f32(0.01).to_bits(), f16::from_f32(0.03).to_bits(),
            f16::from_f32(0.08).to_bits(), f16::from_f32(0.12).to_bits(),
            f16::from_f32(0.10).to_bits(), f16::from_f32(0.06).to_bits(),
            f16::from_f32(0.03).to_bits(), f16::from_f32(0.01).to_bits(),
        ];

        let spacing = 0.5;
        let nx = (self.width / spacing).ceil() as i32;
        let nz = (self.depth / spacing).ceil() as i32;
        let mut splats = Vec::with_capacity((nx * nz) as usize);

        for ix in 0..nx {
            for iz in 0..nz {
                let x = self.center.x + (ix as f32 - nx as f32 * 0.5) * spacing;
                let z = self.center.z + (iz as f32 - nz as f32 * 0.5) * spacing;

                // Wave displacement
                let wave_phase = (x * self.wave_frequency + time * self.flow_speed).sin();
                let wave_phase2 = (z * self.wave_frequency * 0.7 + time * self.flow_speed * 0.5).cos();
                let y = self.water_level + wave_phase * self.wave_amplitude + wave_phase2 * self.wave_amplitude * 0.5;

                // Fresnel-like opacity: more opaque at steep angles (simplified)
                let opacity = 180;

                splats.push(GaussianSplat {
                    position: [x, y, z],
                    scale: [spacing * 0.5, 0.01, spacing * 0.5],
                    rotation: [0, 0, 0, 32767],
                    opacity,
                    _pad: [0; 3],
                    spectral: water_spd,
                });
            }
        }

        splats
    }

    /// Compute reflection direction for a given view direction at the water surface.
    pub fn reflect(view_dir: Vec3) -> Vec3 {
        let normal = Vec3::Y; // water surface normal is up
        view_dir - 2.0 * view_dir.dot(normal) * normal
    }

    /// Fresnel reflectance (Schlick approximation).
    pub fn fresnel(view_dir: Vec3, ior: f32) -> f32 {
        let cos_i = view_dir.dot(Vec3::Y).abs();
        let r0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cos_i).powi(5)
    }
}
