//! Spectral resonance fracture — fracture plane geometry derived from
//! optical-acoustic material coupling.
//! Band variance drives regularity: low variance → crystalline → axis-aligned planes.

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct FracturePlane {
    pub origin: Vec3,
    pub normal: Vec3,
    pub curvature: f32,
}

pub struct SpectralResonanceFracture;

impl SpectralResonanceFracture {
    pub fn compute_planes(
        impact_pos: Vec3,
        impulse_ns: f32,
        spectral: &[u16; 16],
    ) -> Vec<FracturePlane> {
        let profile = decode_spectral(spectral);
        let total_energy: f32 = profile.iter().sum();
        let threshold = total_energy * 0.5;
        if impulse_ns < threshold {
            return Vec::new();
        }
        let mean = total_energy / 16.0;
        let variance: f32 =
            profile.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 16.0;
        let regularity = (1.0 - (variance * 4.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let num_planes =
            ((impulse_ns / threshold).sqrt() * 3.0).clamp(1.0, 8.0) as usize;
        let mut planes = Vec::with_capacity(num_planes);
        for k in 0..num_planes {
            let angle =
                (k as f32) * std::f32::consts::TAU / (num_planes as f32);
            let raw_normal =
                Vec3::new(angle.cos(), 0.3 * regularity, angle.sin()).normalize();
            let normal = if regularity > 0.7 {
                snap_to_axis(raw_normal)
            } else {
                raw_normal
            };
            let curvature = 1.0 - regularity;
            let origin = impact_pos + normal * 0.05;
            planes.push(FracturePlane { origin, normal, curvature });
        }
        planes
    }

    pub fn fracture_threshold(spectral: &[u16; 16]) -> f32 {
        let profile = decode_spectral(spectral);
        profile.iter().sum::<f32>() * 0.5
    }
}

fn snap_to_axis(v: Vec3) -> Vec3 {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    if ax >= ay && ax >= az {
        Vec3::new(v.x.signum(), 0.0, 0.0)
    } else if ay >= ax && ay >= az {
        Vec3::new(0.0, v.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, v.z.signum())
    }
}

fn decode_spectral(s: &[u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = (s[i] as f32) / 65535.0;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn metal_spectral() -> [u16; 16] {
        [60000u16; 16]
    }
    fn glass_spectral() -> [u16; 16] {
        [
            60000, 100, 60000, 100, 60000, 100, 60000, 100, 60000, 100, 60000,
            100, 60000, 100, 60000, 100,
        ]
    }

    #[test]
    fn low_impulse_produces_no_planes() {
        let planes =
            SpectralResonanceFracture::compute_planes(Vec3::ZERO, 0.001, &metal_spectral());
        assert!(planes.is_empty(), "impulse below threshold must produce 0 planes");
    }

    #[test]
    fn high_impulse_produces_planes() {
        let planes =
            SpectralResonanceFracture::compute_planes(Vec3::ZERO, 100.0, &metal_spectral());
        assert!(!planes.is_empty(), "strong impact must produce fracture planes");
    }

    #[test]
    fn crystalline_planes_are_axis_aligned() {
        let planes =
            SpectralResonanceFracture::compute_planes(Vec3::ZERO, 100.0, &metal_spectral());
        for plane in &planes {
            let n = plane.normal;
            let is_axis = (n.x.abs() - 1.0).abs() < 0.01
                || (n.y.abs() - 1.0).abs() < 0.01
                || (n.z.abs() - 1.0).abs() < 0.01;
            assert!(is_axis, "crystalline normal {:?} must be axis-aligned", n);
        }
    }

    #[test]
    fn amorphous_planes_have_curvature() {
        let planes =
            SpectralResonanceFracture::compute_planes(Vec3::ZERO, 100.0, &glass_spectral());
        assert!(!planes.is_empty());
        let max_curvature = planes.iter().map(|p| p.curvature).fold(0.0f32, f32::max);
        assert!(
            max_curvature > 0.1,
            "amorphous material must produce curved fracture planes (got {})",
            max_curvature
        );
    }

    #[test]
    fn plane_normals_are_unit_length() {
        let planes = SpectralResonanceFracture::compute_planes(
            Vec3::new(1.0, 2.0, 3.0),
            50.0,
            &glass_spectral(),
        );
        for plane in &planes {
            let len = plane.normal.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "fracture plane normal must be unit-length, got {}",
                len
            );
        }
    }
}
