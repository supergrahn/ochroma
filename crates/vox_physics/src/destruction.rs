//! Splat fracture — destructible splat assemblies.
//! On impact, a SplatAssembly breaks into SplatParticle instances with spectral shift.

use glam::{self, Vec3};
use vox_core::types::GaussianSplat;

/// A rigid group of splats that fractures as one unit.
#[derive(Debug, Clone)]
pub struct SplatAssembly {
    pub splats: Vec<GaussianSplat>,
    pub position: Vec3,
    pub velocity: Vec3,
    pub health: f32,
    pub max_health: f32,
    pub mass: f32,
    pub is_active: bool,
    /// Impact threshold: minimum impulse to trigger fracture
    pub fracture_threshold: f32,
}

/// A single fracture particle ejected from a broken assembly.
#[derive(Debug, Clone)]
pub struct FractureParticle {
    pub splat: GaussianSplat,
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32, // seconds until despawn
    pub age: f32,
}

/// Manages fracture simulation for a set of assemblies.
pub struct FractureSystem {
    pub assemblies: Vec<SplatAssembly>,
    pub particles: Vec<FractureParticle>,
    /// Camera position for LOD culling: no fracture beyond this distance
    pub camera_pos: Vec3,
    pub max_fracture_distance: f32, // default 30.0 m
    /// Reusable particle pool (avoid alloc per frame)
    particle_pool: Vec<FractureParticle>,
}

const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

impl FractureSystem {
    pub fn new() -> Self {
        Self {
            assemblies: Vec::new(),
            particles: Vec::new(),
            camera_pos: Vec3::ZERO,
            max_fracture_distance: 30.0,
            particle_pool: Vec::new(),
        }
    }

    /// Register an assembly for fracture tracking.
    pub fn add_assembly(&mut self, assembly: SplatAssembly) -> usize {
        let idx = self.assemblies.len();
        self.assemblies.push(assembly);
        idx
    }

    /// Apply an impact impulse to assembly `idx` at `impact_point`.
    /// If impulse exceeds fracture_threshold AND assembly is within max_fracture_distance, fracture.
    pub fn apply_impact(&mut self, idx: usize, impulse: f32, impact_point: Vec3) {
        if idx >= self.assemblies.len() {
            return;
        }
        let threshold = self.assemblies[idx].fracture_threshold;
        let is_active = self.assemblies[idx].is_active;

        if !is_active {
            return;
        }
        if impulse < threshold {
            return;
        }

        // Check LOD distance
        let dist = self.assemblies[idx].position.distance(self.camera_pos);
        if dist > self.max_fracture_distance {
            return;
        }

        self.fracture_assembly(idx, impact_point, impulse);
    }

    /// Advance simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let mut i = 0;
        while i < self.particles.len() {
            let p = &mut self.particles[i];
            p.age += dt;

            if p.age >= p.lifetime {
                // Return to pool
                let dead = self.particles.swap_remove(i);
                self.particle_pool.push(dead);
                continue;
            }

            // Gravity
            p.velocity += GRAVITY * dt;

            // Integrate position
            p.position += p.velocity * dt;

            // Ground plane bounce
            if p.position.y < 0.0 {
                p.position.y = 0.0;
                p.velocity.y *= -0.3;
                p.velocity.x *= 0.9;
                p.velocity.z *= 0.9;
            }

            i += 1;
        }
    }

    /// Produce fracture particles from an assembly, applying spectral shift on break.
    fn fracture_assembly(&mut self, idx: usize, impact_point: Vec3, impulse: f32) {
        // LOD check: skip if assembly is outside max_fracture_distance from camera
        let dist = self.assemblies[idx].position.distance(self.camera_pos);
        if dist > self.max_fracture_distance {
            return;
        }

        let assembly = &mut self.assemblies[idx];
        assembly.is_active = false;

        // Collect splat data to avoid borrow issues
        let splats: Vec<GaussianSplat> = assembly.splats.clone();
        let assembly_position = assembly.position;

        for (splat_idx, mut splat) in splats.into_iter().enumerate() {
            // World position: assembly_position + splat local offset (position field)
            let world_pos = assembly_position
                + Vec3::from(splat.position());

            // Radial velocity from impact point
            let radial_dir = (world_pos - impact_point).normalize_or_zero();

            // Deterministic pseudo-random jitter per splat
            let jitter_seed = (idx * 1664525 + splat_idx * 22695477 + 1013904223) & 0xFFFF;
            let jitter = jitter_seed as f32 / 65535.0 - 0.5;
            let jitter_vec = Vec3::new(jitter, jitter * 0.7 + 0.3, jitter * -0.5);

            let velocity = radial_dir * impulse * 0.5 + jitter_vec;

            // Spectral shift: burn/crumble effect
            spectral_shift_on_break(&mut splat);

            // Lifetime: 3.0 + pseudo-rand(0..2) using a different seed
            let lifetime_seed = ((idx * 22695477 + splat_idx * 1664525 + 1013904223) & 0xFFFF) as f32
                / 65535.0;
            let lifetime = 3.0 + lifetime_seed * 2.0;

            let particle = if let Some(mut pooled) = self.particle_pool.pop() {
                pooled.splat = splat;
                pooled.position = world_pos;
                pooled.velocity = velocity;
                pooled.lifetime = lifetime;
                pooled.age = 0.0;
                pooled
            } else {
                FractureParticle {
                    splat,
                    position: world_pos,
                    velocity,
                    lifetime,
                    age: 0.0,
                }
            };

            self.particles.push(particle);
        }
    }
}

use crate::spectral_fracture::{FracturePlane, SpectralResonanceFracture};

impl SplatAssembly {
    /// Compute spectral resonance fracture planes for this assembly at the
    /// given impact point and impulse (Ns). Returns planes; also reduces
    /// health and deactivates assembly when health reaches zero.
    pub fn fracture_at(&mut self, impact_pos: glam::Vec3, impulse_ns: f32) -> Vec<FracturePlane> {
        if !self.is_active {
            return Vec::new();
        }
        if impulse_ns < self.fracture_threshold {
            return Vec::new();
        }
        let spectral = self.mean_spectral_profile();
        let planes =
            SpectralResonanceFracture::compute_planes(impact_pos, impulse_ns, &spectral);
        if !planes.is_empty() {
            self.health -= impulse_ns;
            if self.health <= 0.0 {
                self.is_active = false;
            }
        }
        planes
    }

    /// Compute mean spectral profile across all splats in the assembly.
    pub fn mean_spectral_profile(&self) -> [u16; 16] {
        if self.splats.is_empty() {
            return [32767u16; 16];
        }
        let mut acc = [0u32; 16];
        for splat in &self.splats {
            for b in 0..16 {
                acc[b] += splat.spectral()[b] as u32;
            }
        }
        let mut out = [0u16; 16];
        for b in 0..16 {
            out[b] = (acc[b] / self.splats.len() as u32) as u16;
        }
        out
    }
}

/// Spectral shift applied to splats on fracture (burn/crumble effect).
/// Reduces high-frequency bands (0-1), slightly increases mid bands (3-4).
pub fn spectral_shift_on_break(splat: &mut GaussianSplat) {
    // bands 0-2 (UV/violet): reduce by 40% (loss of UV/blue emission — char/ash; glass cracking)
    for b in 0..3 {
        let val = half::f16::from_bits(splat.spectral()[b]).to_f32();
        splat.spectral_mut()[b] = half::f16::from_f32(val * 0.6).to_bits();
    }
    // band 3-4: slight boost (thermal glow at break point)
    for b in 3..5 {
        let val = half::f16::from_bits(splat.spectral()[b]).to_f32();
        splat.spectral_mut()[b] = half::f16::from_f32((val * 1.15).min(1.0)).to_bits();
    }
}

impl Default for FractureSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn make_test_splat(spectral_val: f32) -> GaussianSplat {
    let bits = half::f16::from_f32(spectral_val).to_bits();
    GaussianSplat::volume(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        glam::Quat::IDENTITY,
        255,
        [bits; 16],
    )
}

#[allow(dead_code)]
fn make_assembly_with_splat(pos: Vec3, threshold: f32, splat_val: f32) -> SplatAssembly {
    SplatAssembly {
        splats: vec![make_test_splat(splat_val)],
        position: pos,
        velocity: Vec3::ZERO,
        health: 100.0,
        max_health: 100.0,
        mass: 1.0,
        is_active: true,
        fracture_threshold: threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_shift_reduces_band0() {
        let mut splat = make_test_splat(0.8);
        let before = half::f16::from_bits(splat.spectral()[0]).to_f32();
        spectral_shift_on_break(&mut splat);
        let after = half::f16::from_bits(splat.spectral()[0]).to_f32();
        assert!(
            after < before,
            "Band 0 should be reduced after spectral shift: before={before}, after={after}"
        );
        // Should be approximately 0.8 * 0.6 = 0.48
        assert!(
            (after - before * 0.6).abs() < 0.01,
            "Band 0 reduction should be ~40%: expected ~{}, got {after}",
            before * 0.6
        );
    }

    #[test]
    fn spectral_shift_boosts_band3() {
        let mut splat = make_test_splat(0.5);
        let before = half::f16::from_bits(splat.spectral()[3]).to_f32();
        spectral_shift_on_break(&mut splat);
        let after = half::f16::from_bits(splat.spectral()[3]).to_f32();
        assert!(
            after > before,
            "Band 3 should be boosted after spectral shift: before={before}, after={after}"
        );
        assert!(
            (after - (before * 1.15).min(1.0)).abs() < 0.01,
            "Band 3 boost should be ~15%: expected ~{}, got {after}",
            before * 1.15
        );
    }

    #[test]
    fn apply_impact_below_threshold_no_fracture() {
        let mut sys = FractureSystem::new();
        sys.camera_pos = Vec3::ZERO;
        let asm = make_assembly_with_splat(Vec3::ZERO, 50.0, 0.5);
        let idx = sys.add_assembly(asm);

        // Impulse below threshold
        sys.apply_impact(idx, 10.0, Vec3::ZERO);

        assert!(
            sys.particles.is_empty(),
            "No particles should be emitted for sub-threshold impulse"
        );
        assert!(
            sys.assemblies[idx].is_active,
            "Assembly should still be active after sub-threshold impact"
        );
    }

    #[test]
    fn apply_impact_above_threshold_fractures() {
        let mut sys = FractureSystem::new();
        sys.camera_pos = Vec3::ZERO;
        sys.max_fracture_distance = 30.0;
        let asm = make_assembly_with_splat(Vec3::new(0.0, 0.0, 5.0), 50.0, 0.5);
        let idx = sys.add_assembly(asm);

        // Impulse above threshold, camera close
        sys.apply_impact(idx, 100.0, Vec3::ZERO);

        assert!(
            !sys.particles.is_empty(),
            "Particles should be emitted for above-threshold impulse"
        );
        assert!(
            !sys.assemblies[idx].is_active,
            "Assembly should be deactivated after fracture"
        );
    }

    #[test]
    fn lod_culling_prevents_distant_fracture() {
        let mut sys = FractureSystem::new();
        sys.camera_pos = Vec3::ZERO;
        sys.max_fracture_distance = 30.0;
        // Place assembly far from camera
        let asm = make_assembly_with_splat(Vec3::new(0.0, 0.0, 100.0), 50.0, 0.5);
        let idx = sys.add_assembly(asm);

        // Very high impulse but assembly is outside max_fracture_distance
        sys.apply_impact(idx, 9999.0, Vec3::ZERO);

        assert!(
            sys.particles.is_empty(),
            "No particles should be emitted for distant assembly (LOD culling)"
        );
        assert!(
            sys.assemblies[idx].is_active,
            "Assembly should remain active when culled by LOD distance"
        );
    }

    #[test]
    fn step_advances_particle_age() {
        let mut sys = FractureSystem::new();
        sys.camera_pos = Vec3::ZERO;
        let asm = make_assembly_with_splat(Vec3::ZERO, 10.0, 0.5);
        let idx = sys.add_assembly(asm);
        sys.apply_impact(idx, 100.0, Vec3::new(1.0, 0.0, 0.0));

        assert!(!sys.particles.is_empty(), "Need at least one particle to test age");
        let age_before = sys.particles[0].age;

        sys.step(0.1);

        let age_after = sys.particles[0].age;
        assert!(
            (age_after - (age_before + 0.1)).abs() < 1e-5,
            "Particle age should increase by dt: expected ~{}, got {age_after}",
            age_before + 0.1
        );
    }
}
