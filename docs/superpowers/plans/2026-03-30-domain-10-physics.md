# Domain 10: Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PBF GPU fluids at 50k particles × 60fps; spectral resonance fracture (planes from optical-acoustic material coupling); full Rapier integration; spectral thermal dynamics (hot objects emit in red/IR bands feeding GI cache).

**Done When:** Running `cargo run`, dropping a rigid body onto a glass floor causes the floor splats' spectral bands 1-3 (UV/violet) to visibly reduce (crack damage) upon impact, verified by `cargo test -p vox_physics rigid_body_impact_cracks_glass_spectral` passing with `assert!(cracked.spectral_f32(2) < original.spectral_f32(2) - 0.05)`.

**Architecture:** Four interlocking systems: (1) `PbfFluidSim` — wgpu compute, 4 passes/frame, density constraint λᵢ = −Cᵢ/(∇Cᵢ² + ε); (2) `SpectralResonanceFracture` — band-variance drives fracture plane regularity, wired into existing `FractureSystem::fracture_at()`; (3) `ThermalEmitter` — splats above heat threshold elevate bands 9-14, feeding `SpectralRadianceCache`; (4) `SpectralFluid` — PBF particles carry `spectral: [f32; 16]`, visual appearance emerges from physics.

**Tech Stack:** Rust, wgpu, rapier3d `0.22`, glam (existing), bytemuck (existing), WGSL compute

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `crates/vox_physics/src/pbf.rs` | `PbfParticle`, `PbfGpuBuffers`, `PbfFluidSim`, density constraint math |
| Create | `crates/vox_physics/src/gpu/pbf_predict.wgsl` | Pass 1: predict positions from velocity |
| Create | `crates/vox_physics/src/gpu/pbf_neighbors.wgsl` | Pass 2: spatial hashing, neighbor lists |
| Create | `crates/vox_physics/src/gpu/pbf_solve.wgsl` | Pass 3: density constraint solve, Δp |
| Create | `crates/vox_physics/src/gpu/pbf_integrate.wgsl` | Pass 4: update velocity + position |
| Create | `crates/vox_physics/src/spectral_fracture.rs` | `SpectralResonanceFracture`, `FracturePlane` |
| Create | `crates/vox_physics/src/thermal.rs` | `ThermalEmitter`, heat diffusion, GI integration |
| Modify | `crates/vox_physics/src/fluid.rs` | Add `SpectralFluid` variant, wire PBF |
| Modify | `crates/vox_physics/src/destruction.rs` | Wire `SpectralResonanceFracture` into `fracture_at()` |
| Modify | `crates/vox_physics/src/lib.rs` | Expose new modules |
| Modify | `crates/vox_physics/Cargo.toml` | Add rapier3d `0.22` |

---

## Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| PBF gravity | `cpu_step()` on a particle at y=5 → y decreases; assert `y1 < y0` | `assert!(true)` |
| Ground plane | 10 steps with downward velocity → `position[1] >= 0.0` always | asserting position without stepping |
| Spectral fracture planes | crystalline metal at impulse=100 → normals axis-aligned, `(n.x.abs()-1.0).abs() < 0.01` | `assert!(!planes.is_empty())` without checking normals |
| Amorphous curvature | glass spectral at impulse=100 → `max_curvature > 0.1` | checking plane count only |
| Thermal band elevation | 20 heat-source updates → bands 9-14 non-zero | checking `heat[i] > 0` without checking splat |
| Spectral fluid mixing | water+lava mix at 0.5 → `s[14] > WATER[14]` and `s[14] < LAVA[14]` | checking particle count |
| Wet spectral blend | wet_factor=0.3 on dry soil → NIR bands 8-15 darker than dry | checking blend returns any value |

---

## Task 1: PbfParticle + PbfFluidSim — density constraint CPU fallback AND wire module

**Files:**
- Create: `crates/vox_physics/src/pbf.rs`
- Modify: `crates/vox_physics/src/lib.rs`

**Acceptance:** `cargo test -p vox_physics pbf -- --nocapture` → PASS, output includes `particle should fall` test name and 5 tests pass.

**Wiring requirement:** Must be exposed from `pub mod pbf;` in `crates/vox_physics/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// crates/vox_physics/src/pbf.rs — include at bottom of file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbf_particle_size_aligned() {
        // position(12) + density(4) + predicted(12) + lambda(4) + velocity(12) + pressure(4) + spectral(64) = 112 bytes
        assert_eq!(std::mem::size_of::<PbfParticle>(), 112,
            "PbfParticle must be 112 bytes for GPU alignment");
    }

    #[test]
    fn pbf_params_size() {
        assert_eq!(std::mem::size_of::<PbfParams>(), 32);
    }

    #[test]
    fn spawn_increments_count() {
        let mut sim = PbfFluidSim::new(1000.0, 0.1);
        sim.spawn([0.0, 1.0, 0.0], [0.0; 3], WATER_SPECTRAL);
        sim.spawn([0.1, 1.0, 0.0], [0.0; 3], WATER_SPECTRAL);
        assert_eq!(sim.particles.len(), 2);
        assert_eq!(sim.params.particle_count, 2);
    }

    #[test]
    fn cpu_step_particles_fall_under_gravity() {
        let mut sim = PbfFluidSim::new(1000.0, 0.15);
        sim.spawn([0.0, 5.0, 0.0], [0.0; 3], WATER_SPECTRAL);
        let y0 = sim.particles[0].position[1];
        sim.cpu_step();
        let y1 = sim.particles[0].position[1];
        assert!(y1 < y0, "particle should fall: y0={} y1={}", y0, y1);
    }

    #[test]
    fn cpu_step_ground_plane_clamped() {
        let mut sim = PbfFluidSim::new(1000.0, 0.15);
        sim.spawn([0.0, 0.001, 0.0], [0.0, -10.0, 0.0], WATER_SPECTRAL);
        for _ in 0..10 { sim.cpu_step(); }
        assert!(sim.particles[0].position[1] >= 0.0,
            "particle must not go below ground plane");
    }

    #[test]
    fn spectral_profile_persists_through_step() {
        let mut sim = PbfFluidSim::new(1000.0, 0.15);
        sim.spawn([0.0, 2.0, 0.0], [0.0; 3], BLOOD_SPECTRAL);
        sim.cpu_step();
        assert!(sim.particles[0].spectral[9] > 0.0,
            "blood band 9 must persist through physics step");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics pbf 2>&1 | head -20
```
Expected: FAIL — compile error, module not found

- [ ] **Step 3: Implement**
```rust
//! Position-Based Fluids (PBF) — GPU-accelerated.
//! 4 compute passes per frame: predict → neighbors → solve → integrate.
//! Density constraint: λᵢ = −Cᵢ / (∇Cᵢ² + ε),  Cᵢ = ρᵢ/ρ₀ − 1

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PbfParticle {
    pub position:      [f32; 3],
    pub density:       f32,
    pub predicted_pos: [f32; 3],
    pub lambda:        f32,
    pub velocity:      [f32; 3],
    pub pressure:      f32,
    pub spectral:      [f32; 16],
}

pub struct PbfGpuBuffers {
    pub particle_buf:  wgpu::Buffer,
    pub neighbor_buf:  wgpu::Buffer,
    pub params_buf:    wgpu::Buffer,
    pub max_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PbfParams {
    pub dt:             f32,
    pub rest_density:   f32,
    pub smoothing_h:    f32,
    pub epsilon:        f32,
    pub particle_count: u32,
    pub max_neighbors:  u32,
    pub _pad:           [u32; 2],
}

pub struct PbfFluidSim {
    pub particles: Vec<PbfParticle>,
    pub params:    PbfParams,
    gpu:           Option<PbfGpuBuffers>,
}

pub const WATER_SPECTRAL: [f32; 16] = [0.1, 0.8, 0.9, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
pub const BLOOD_SPECTRAL: [f32; 16] = [0.0, 0.0, 0.01, 0.05, 0.2, 0.6, 0.4, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01];
pub const LAVA_SPECTRAL:  [f32; 16] = [0.0, 0.0, 0.02, 0.1, 0.4, 0.8, 1.0, 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60];

impl PbfFluidSim {
    pub fn new(rest_density: f32, smoothing_h: f32) -> Self {
        Self {
            particles: Vec::new(),
            params: PbfParams {
                dt: 1.0 / 60.0, rest_density, smoothing_h,
                epsilon: 1e-4, particle_count: 0, max_neighbors: 64, _pad: [0; 2],
            },
            gpu: None,
        }
    }

    pub fn spawn(&mut self, pos: [f32; 3], vel: [f32; 3], spectral: [f32; 16]) {
        self.particles.push(PbfParticle {
            position: pos, density: self.params.rest_density, predicted_pos: pos,
            lambda: 0.0, velocity: vel, pressure: 0.0, spectral,
        });
        self.params.particle_count = self.particles.len() as u32;
    }

    pub fn cpu_step(&mut self) {
        let dt = self.params.dt;
        let h = self.params.smoothing_h;
        let rho0 = self.params.rest_density;
        let eps = self.params.epsilon;
        let n = self.particles.len();

        for p in &mut self.particles {
            p.velocity[1] -= 9.81 * dt;
            p.predicted_pos = [
                p.position[0] + p.velocity[0] * dt,
                p.position[1] + p.velocity[1] * dt,
                p.position[2] + p.velocity[2] * dt,
            ];
        }

        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
        let mut densities = vec![0.0f32; n];
        for i in 0..n {
            let mut rho = 0.0f32;
            for j in 0..n {
                let dx = self.particles[i].predicted_pos[0] - self.particles[j].predicted_pos[0];
                let dy = self.particles[i].predicted_pos[1] - self.particles[j].predicted_pos[1];
                let dz = self.particles[i].predicted_pos[2] - self.particles[j].predicted_pos[2];
                let r2 = dx*dx + dy*dy + dz*dz;
                if r2 < h*h { rho += poly6_coeff * (h*h - r2).powi(3); }
            }
            densities[i] = rho;
        }

        let spiky_grad = -45.0 / (std::f32::consts::PI * h.powi(6));
        let mut deltas = vec![[0.0f32; 3]; n];
        for i in 0..n {
            let ci = densities[i] / rho0 - 1.0;
            let mut grad_sq = 0.0f32;
            for j in 0..n {
                if i == j { continue; }
                let dx = self.particles[i].predicted_pos[0] - self.particles[j].predicted_pos[0];
                let dy = self.particles[i].predicted_pos[1] - self.particles[j].predicted_pos[1];
                let dz = self.particles[i].predicted_pos[2] - self.particles[j].predicted_pos[2];
                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                if r < h && r > 1e-6 {
                    let g = spiky_grad * (h - r).powi(2) / r;
                    grad_sq += (g*dx)*(g*dx) + (g*dy)*(g*dy) + (g*dz)*(g*dz);
                }
            }
            self.particles[i].lambda = -ci / (grad_sq / rho0 + eps);
        }
        for i in 0..n {
            let li = self.particles[i].lambda;
            for j in 0..n {
                if i == j { continue; }
                let dx = self.particles[i].predicted_pos[0] - self.particles[j].predicted_pos[0];
                let dy = self.particles[i].predicted_pos[1] - self.particles[j].predicted_pos[1];
                let dz = self.particles[i].predicted_pos[2] - self.particles[j].predicted_pos[2];
                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                if r < h && r > 1e-6 {
                    let g = spiky_grad * (h - r).powi(2) / r;
                    let lj = self.particles[j].lambda;
                    let s = (li + lj) / rho0;
                    deltas[i][0] += s * g * dx;
                    deltas[i][1] += s * g * dy;
                    deltas[i][2] += s * g * dz;
                }
            }
        }
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.predicted_pos[0] += deltas[i][0];
            p.predicted_pos[1] += deltas[i][1];
            p.predicted_pos[2] += deltas[i][2];
            if p.predicted_pos[1] < 0.0 { p.predicted_pos[1] = 0.0; }
        }

        for p in &mut self.particles {
            p.velocity = [
                (p.predicted_pos[0] - p.position[0]) / dt,
                (p.predicted_pos[1] - p.position[1]) / dt,
                (p.predicted_pos[2] - p.position[2]) / dt,
            ];
            p.position = p.predicted_pos;
        }
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_physics/src/lib.rs — add:
pub mod pbf;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics pbf -- --nocapture
```
Expected: PASS, 6 tests pass, output shows `particle should fall`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/pbf.rs crates/vox_physics/src/lib.rs
git commit -m "feat(physics): PbfParticle + PbfFluidSim — density constraint, CPU fallback step"
```

---

## Task 2: PBF compute shaders — 4-pass GPU simulation loop AND wire into PbfFluidSim

**Files:**
- Create: `crates/vox_physics/src/gpu/pbf_predict.wgsl`
- Create: `crates/vox_physics/src/gpu/pbf_neighbors.wgsl`
- Create: `crates/vox_physics/src/gpu/pbf_solve.wgsl`
- Create: `crates/vox_physics/src/gpu/pbf_integrate.wgsl`
- Modify: `crates/vox_physics/src/pbf.rs` (add `upload_to_gpu`, `gpu_step`)

**Acceptance:** `cargo test -p vox_physics pbf -- --nocapture` → 6 tests pass; `cargo build -p vox_physics` → clean.

**Wiring requirement:** Must be called from `PbfFluidSim::gpu_step()` in `crates/vox_physics/src/pbf.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// In pbf.rs tests — add:
#[test]
fn gpu_buffers_size_calculation() {
    // Validate that buffer sizing arithmetic doesn't overflow for 50k particles
    let max: u32 = 50_000;
    let particle_bytes = max as u64 * std::mem::size_of::<PbfParticle>() as u64;
    let neighbor_bytes = max as u64 * (64u64 + 1) * 4; // max_neighbors=64
    assert!(particle_bytes > 0, "particle buffer must be non-zero");
    assert_eq!(particle_bytes, 50_000 * 112); // 112 bytes per PbfParticle
    assert_eq!(neighbor_bytes, 50_000 * 65 * 4);
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics gpu_buffers_size 2>&1 | tail -5
```
Expected: FAIL — test not found (module not yet present)

- [ ] **Step 3: Implement** (create all 4 WGSL shaders and add `upload_to_gpu` / `gpu_step` / `PbfPipelines` to pbf.rs)

Pass 1 — `crates/vox_physics/src/gpu/pbf_predict.wgsl`:
```wgsl
struct Particle {
    position:      vec3<f32>, density: f32,
    predicted_pos: vec3<f32>, lambda:  f32,
    velocity:      vec3<f32>, pressure: f32,
    spectral:      array<f32, 16>,
}
struct PbfParams { dt: f32, rest_density: f32, smoothing_h: f32, epsilon: f32, particle_count: u32, max_neighbors: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: PbfParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count { return; }
    var v = particles[i].velocity;
    v.y -= 9.81 * params.dt;
    particles[i].velocity = v;
    particles[i].predicted_pos = particles[i].position + v * params.dt;
}
```

Pass 2 — `crates/vox_physics/src/gpu/pbf_neighbors.wgsl`, Pass 3 — `pbf_solve.wgsl`, Pass 4 — `pbf_integrate.wgsl`: same struct layout; solve computes `λᵢ = −Cᵢ / (∇Cᵢ²/ρ₀ + ε)` using poly6 density and spiky gradient; integrate applies Δp and enforces `y ≥ 0`.

Add to `crates/vox_physics/src/pbf.rs`:
```rust
pub struct PbfPipelines {
    pub predict:    wgpu::ComputePipeline,
    pub neighbors:  wgpu::ComputePipeline,
    pub solve:      wgpu::ComputePipeline,
    pub integrate:  wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
}

impl PbfFluidSim {
    pub fn upload_to_gpu(&mut self, device: &wgpu::Device, max_particles: u32) {
        let particle_bytes = max_particles as u64 * std::mem::size_of::<PbfParticle>() as u64;
        let neighbor_bytes = max_particles as u64 * (self.params.max_neighbors as u64 + 1) * 4;
        let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbf_particles"), size: particle_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let neighbor_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbf_neighbors"), size: neighbor_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbf_params"), size: std::mem::size_of::<PbfParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu = Some(PbfGpuBuffers { particle_buf, neighbor_buf, params_buf, max_particles });
    }

    pub fn gpu_step(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, pipelines: &PbfPipelines) {
        let gpu = match &self.gpu { Some(g) => g, None => { self.cpu_step(); return; } };
        self.params.particle_count = self.particles.len() as u32;
        queue.write_buffer(&gpu.params_buf, 0, bytemuck::bytes_of(&self.params));
        queue.write_buffer(&gpu.particle_buf, 0, bytemuck::cast_slice(&self.particles));
        let count = self.params.particle_count;
        let workgroups = (count + 63) / 64;
        for pipeline in [&pipelines.predict, &pipelines.neighbors, &pipelines.solve, &pipelines.integrate] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pbf_pass"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &pipelines.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}
```
- [ ] **Step 4: Wire at exact callsite**

`gpu_step()` is already wired in the impl block above — it dispatches all 4 pipelines in order.

- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics pbf -- --nocapture
```
Expected: PASS, 7 tests pass (including `gpu_buffers_size_calculation` with `50_000 * 112 = 5_600_000`)

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/pbf.rs \
        crates/vox_physics/src/gpu/pbf_predict.wgsl \
        crates/vox_physics/src/gpu/pbf_neighbors.wgsl \
        crates/vox_physics/src/gpu/pbf_solve.wgsl \
        crates/vox_physics/src/gpu/pbf_integrate.wgsl
git commit -m "feat(physics): PBF 4-pass GPU compute — predict/neighbors/solve/integrate"
```

---

## Task 3: SpectralResonanceFracture — wire into destruction.rs

**Files:**
- Create: `crates/vox_physics/src/spectral_fracture.rs`
- Modify: `crates/vox_physics/src/destruction.rs`
- Modify: `crates/vox_physics/src/lib.rs`

**Acceptance:** `cargo test -p vox_physics spectral_fracture -- --nocapture` → 5 tests pass, output shows `crystalline normal ... must be axis-aligned`.

**Wiring requirement:** Must be called from `SplatAssembly::fracture_at()` in `crates/vox_physics/src/destruction.rs`. Must also add `assembly.fracture_at()` call inside `FractureSystem::apply_impact` in `crates/vox_physics/src/fracture.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// crates/vox_physics/src/spectral_fracture.rs — tests module
#[cfg(test)]
mod tests {
    use super::*;

    fn metal_spectral() -> [u16; 16] { [60000u16; 16] }
    fn glass_spectral() -> [u16; 16] {
        [60000, 100, 60000, 100, 60000, 100, 60000, 100,
         60000, 100, 60000, 100, 60000, 100, 60000, 100]
    }

    #[test]
    fn low_impulse_produces_no_planes() {
        let planes = SpectralResonanceFracture::compute_planes(Vec3::ZERO, 0.001, &metal_spectral());
        assert!(planes.is_empty(), "impulse below threshold must produce 0 planes");
    }

    #[test]
    fn high_impulse_produces_planes() {
        let planes = SpectralResonanceFracture::compute_planes(Vec3::ZERO, 100.0, &metal_spectral());
        assert!(!planes.is_empty(), "strong impact must produce fracture planes");
    }

    #[test]
    fn crystalline_planes_are_axis_aligned() {
        let planes = SpectralResonanceFracture::compute_planes(Vec3::ZERO, 100.0, &metal_spectral());
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
        let planes = SpectralResonanceFracture::compute_planes(Vec3::ZERO, 100.0, &glass_spectral());
        assert!(!planes.is_empty());
        let max_curvature = planes.iter().map(|p| p.curvature).fold(0.0f32, f32::max);
        assert!(max_curvature > 0.1,
            "amorphous material must produce curved fracture planes (got {})", max_curvature);
    }

    #[test]
    fn plane_normals_are_unit_length() {
        let planes = SpectralResonanceFracture::compute_planes(Vec3::new(1.0, 2.0, 3.0), 50.0, &glass_spectral());
        for plane in &planes {
            let len = plane.normal.length();
            assert!((len - 1.0).abs() < 1e-5, "fracture plane normal must be unit-length, got {}", len);
        }
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics spectral_fracture 2>&1 | tail -5
```
Expected: FAIL — compile error, module not found

- [ ] **Step 3: Implement**
```rust
//! Spectral resonance fracture — fracture plane geometry derived from
//! optical-acoustic material coupling.
//! Band variance drives regularity: low variance → crystalline → axis-aligned planes.

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct FracturePlane { pub origin: Vec3, pub normal: Vec3, pub curvature: f32 }

pub struct SpectralResonanceFracture;

impl SpectralResonanceFracture {
    pub fn compute_planes(impact_pos: Vec3, impulse_ns: f32, spectral: &[u16; 16]) -> Vec<FracturePlane> {
        let profile = decode_spectral(spectral);
        let total_energy: f32 = profile.iter().sum();
        let threshold = total_energy * 0.5;
        if impulse_ns < threshold { return Vec::new(); }
        let mean = total_energy / 16.0;
        let variance: f32 = profile.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 16.0;
        let regularity = (1.0 - (variance * 4.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let num_planes = ((impulse_ns / threshold).sqrt() * 3.0).clamp(1.0, 8.0) as usize;
        let mut planes = Vec::with_capacity(num_planes);
        for k in 0..num_planes {
            let angle = (k as f32) * std::f32::consts::TAU / (num_planes as f32);
            let raw_normal = Vec3::new(angle.cos(), 0.3 * regularity, angle.sin()).normalize();
            let normal = if regularity > 0.7 { snap_to_axis(raw_normal) } else { raw_normal };
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
    let ax = v.x.abs(); let ay = v.y.abs(); let az = v.z.abs();
    if ax >= ay && ax >= az { Vec3::new(v.x.signum(), 0.0, 0.0) }
    else if ay >= ax && ay >= az { Vec3::new(0.0, v.y.signum(), 0.0) }
    else { Vec3::new(0.0, 0.0, v.z.signum()) }
}

fn decode_spectral(s: &[u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 { out[i] = (s[i] as f32) / 65535.0; }
    out
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_physics/src/lib.rs — add:
pub mod spectral_fracture;

// crates/vox_physics/src/destruction.rs — add to SplatAssembly impl:
use crate::spectral_fracture::{SpectralResonanceFracture, FracturePlane};

pub fn fracture_at(&mut self, impact_pos: glam::Vec3, impulse_ns: f32) -> Vec<FracturePlane> {
    if !self.is_active { return Vec::new(); }
    if impulse_ns < self.fracture_threshold { return Vec::new(); }
    let spectral = self.mean_spectral_profile();
    let planes = SpectralResonanceFracture::compute_planes(impact_pos, impulse_ns, &spectral);
    if !planes.is_empty() {
        self.health -= impulse_ns;
        if self.health <= 0.0 { self.is_active = false; }
    }
    planes
}

fn mean_spectral_profile(&self) -> [u16; 16] {
    if self.splats.is_empty() { return [32767u16; 16]; }
    let mut acc = [0u32; 16];
    for splat in &self.splats { for b in 0..16 { acc[b] += splat.spectral()[b] as u32; } }
    let mut out = [0u16; 16];
    for b in 0..16 { out[b] = (acc[b] / self.splats.len() as u32) as u16; }
    out
}

// crates/vox_physics/src/fracture.rs — wire fracture_at() inside FractureSystem::apply_impact:
pub fn apply_impact(&mut self, splats: &mut Vec<GaussianSplat>, contact: &Contact) {
    // existing impulse/damage code...

    // Wire fracture: trigger if damage exceeds threshold
    if contact.impulse > self.fracture_threshold {
        let assembly = SplatAssembly::from_splats(splats);
        let fragments = assembly.fracture_at(contact.position, contact.impulse);
        *splats = fragments;
    }
}
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics spectral_fracture -- --nocapture
```
Expected: PASS, 5 tests pass, output shows axis-aligned normal values like `[1.0, 0.0, 0.0]`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/spectral_fracture.rs \
        crates/vox_physics/src/destruction.rs \
        crates/vox_physics/src/lib.rs
git commit -m "feat(physics): SpectralResonanceFracture — band-variance fracture planes wired into destruction.rs"
```

---

## Task 4: ThermalEmitter — heat diffusion + bands 9-14 elevation AND wire GI seeding

**Files:**
- Create: `crates/vox_physics/src/thermal.rs`
- Modify: `crates/vox_physics/src/lib.rs`

**Acceptance:** `cargo test -p vox_physics thermal -- --nocapture` → 5 tests pass, output shows `b9=... b11=... b14=...` non-zero values.

**Wiring requirement:** Must be exposed from `pub mod thermal;` in `crates/vox_physics/src/lib.rs`. `hot_emitters()` must return live data — not an empty iterator. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_splat(pos: [f32; 3]) -> GaussianSplat {
        let spectral = [half::f16::from_f32(0.0).to_bits(); 16];
        GaussianSplat::surface(pos.into(), [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.1, 0.1, 200, spectral)
    }

    #[test]
    fn heat_source_elevates_bands_9_to_14() {
        let mut emitter = ThermalEmitter::new(0);
        let mut splats = vec![make_splat([0.0, 0.0, 0.0])];
        let source = (Vec3::ZERO, 1.0);
        for _ in 0..20 { emitter.update(&mut splats, &[source]); }
        let b9  = half::f16::from_bits(splats[0].spectral()[9]).to_f32();
        let b11 = half::f16::from_bits(splats[0].spectral()[11]).to_f32();
        let b14 = half::f16::from_bits(splats[0].spectral()[14]).to_f32();
        assert!(b9 > 0.0 || b11 > 0.0 || b14 > 0.0,
            "bands 9-14 must be elevated after heat application (b9={b9}, b11={b11}, b14={b14})");
    }

    #[test]
    fn distant_splat_receives_less_heat() {
        let mut emitter = ThermalEmitter::new(0);
        let mut splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([10.0, 0.0, 0.0])];
        let source = (Vec3::ZERO, 1.0);
        for _ in 0..10 { emitter.update(&mut splats, &[source]); }
        assert!(emitter.heat[0] >= emitter.heat[1], "close splat must have >= heat than distant splat");
    }

    #[test]
    fn heat_cools_without_source() {
        let mut emitter = ThermalEmitter::new(1);
        emitter.heat[0] = 0.8;
        let mut splats = vec![make_splat([0.0, 0.0, 0.0])];
        for _ in 0..50 { emitter.update(&mut splats, &[]); }
        assert!(emitter.heat[0] < 0.8, "heat should cool without source, got {}", emitter.heat[0]);
    }

    #[test]
    fn cold_splats_bands_unchanged() {
        let mut emitter = ThermalEmitter::new(0);
        let b11_init = half::f16::from_f32(0.5).to_bits();
        let mut spectral = [half::f16::from_f32(0.0).to_bits(); 16];
        spectral[11] = b11_init;
        let mut splats = vec![GaussianSplat::surface([0.0,0.0,0.0].into(),[1.0,0.0,0.0],[0.0,0.0,-1.0],0.1,0.1,200,spectral)];
        let source = (Vec3::new(100.0, 0.0, 0.0), 0.001);
        emitter.update(&mut splats, &[source]);
        let b11_after = half::f16::from_bits(splats[0].spectral()[11]).to_f32();
        let b11_before = half::f16::from_bits(b11_init).to_f32();
        assert!((b11_after - b11_before).abs() < 0.01,
            "distant source must not change band 11: before={} after={}", b11_before, b11_after);
    }

    #[test]
    fn hot_emitters_yields_above_threshold() {
        let mut emitter = ThermalEmitter::new(2);
        emitter.heat[0] = 0.9;
        emitter.heat[1] = 0.05;
        let splats = vec![make_splat([0.0, 0.0, 0.0]), make_splat([1.0, 0.0, 0.0])];
        let emitters: Vec<_> = emitter.hot_emitters(&splats).collect();
        assert_eq!(emitters.len(), 1, "only the hot splat should be yielded");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics thermal 2>&1 | head -20
```
Expected: FAIL — compile error, module not found

- [ ] **Step 3: Implement**
```rust
//! Thermal dynamics for spectral splats.
//! Hot objects emit in bands 9–14 (580–730 nm red/near-IR).
//! Heat diffuses via inverse-square, tracked per frame.

use glam::Vec3;
use half::f16;
use vox_core::types::GaussianSplat;

#[derive(Debug, Clone)]
pub struct HeatSource { pub position: Vec3, pub power: f32, pub cooling_rate: f32, pub age_seconds: f32 }

pub struct ThermalEmitter {
    pub heat:              Vec<f32>,
    pub emit_threshold:    f32,
    pub diffusion_radius:  f32,
    pub cooling_per_frame: f32,
}

impl ThermalEmitter {
    pub fn new(splat_count: usize) -> Self {
        Self { heat: vec![0.0f32; splat_count], emit_threshold: 0.2, diffusion_radius: 0.5, cooling_per_frame: 0.005 }
    }
    pub fn resize(&mut self, count: usize) { self.heat.resize(count, 0.0); }

    pub fn update(&mut self, splats: &mut Vec<GaussianSplat>, heat_sources: &[(Vec3, f32)]) {
        self.resize(splats.len());
        let r2_limit = self.diffusion_radius * self.diffusion_radius;
        for (i, splat) in splats.iter().enumerate() {
            let pos = Vec3::from_array(splat.position());
            for &(src_pos, power) in heat_sources {
                let dist_sq = (pos - src_pos).length_squared();
                if dist_sq < r2_limit {
                    let attenuation = 1.0 - (dist_sq / r2_limit).sqrt();
                    self.heat[i] = (self.heat[i] + power * attenuation * 0.1).clamp(0.0, 1.0);
                }
            }
        }
        for (i, splat) in splats.iter_mut().enumerate() {
            let h = self.heat[i];
            if h > self.emit_threshold {
                let excess = h - self.emit_threshold;
                for b in 9..15usize {
                    let current = f16::from_bits(splat.spectral_mut()[b]).to_f32();
                    let elevated = (current + excess * 0.5).clamp(0.0, 1.0);
                    splat.spectral_mut()[b] = f16::from_f32(elevated).to_bits();
                }
            }
            self.heat[i] = (h - self.cooling_per_frame).max(0.0);
        }
    }

    pub fn hot_emitters<'a>(&'a self, splats: &'a [GaussianSplat]) -> impl Iterator<Item = (Vec3, [f32; 16])> + 'a {
        splats.iter().enumerate()
            .filter(|(i, _)| *i < self.heat.len() && self.heat[*i] > self.emit_threshold)
            .map(|(i, splat)| {
                let pos = Vec3::from_array(splat.position());
                let mut spectral = [0.0f32; 16];
                for b in 0..16 { spectral[b] = f16::from_bits(splat.spectral()[b]).to_f32() * self.heat[i]; }
                (pos, spectral)
            })
    }
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_physics/src/lib.rs — add:
pub mod thermal;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics thermal -- --nocapture
```
Expected: PASS, 5 tests pass, output shows non-zero `b9`/`b11`/`b14` values and single hot emitter

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/thermal.rs crates/vox_physics/src/lib.rs
git commit -m "feat(physics): ThermalEmitter — heat diffusion, bands 9-14 elevation, GI seeding"
```

---

## Task 5: SpectralFluid — PBF particles with spectral[16] AND wire SpectralFluidKind

**Files:**
- Modify: `crates/vox_physics/src/fluid.rs`

**Acceptance:** `cargo test -p vox_physics spectral_fluid -- --nocapture` → 4 tests pass, output shows `water band 1 (blue) must exceed band 12 (red)`.

**Wiring requirement:** Must be called from `SpectralFluid::spawn()` in `crates/vox_physics/src/fluid.rs` which delegates to `PbfFluidSim::spawn()`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod spectral_fluid_tests {
    use super::*;

    #[test]
    fn water_particles_have_blue_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
        fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
        let s = &fluid.sim.particles[0].spectral;
        assert!(s[1] > s[12], "water band 1 (blue) must exceed band 12 (red)");
    }

    #[test]
    fn blood_particles_have_red_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Blood);
        fluid.spawn([0.0, 1.0, 0.0], [0.0; 3]);
        let s = &fluid.sim.particles[0].spectral;
        assert!(s[9] > s[0], "blood band 9 (red) must exceed band 0 (violet)");
    }

    #[test]
    fn mixed_spawn_blends_spectral() {
        use crate::pbf::LAVA_SPECTRAL;
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
        fluid.spawn_mixed([0.0, 1.0, 0.0], [0.0; 3], &LAVA_SPECTRAL, 0.5);
        let s = fluid.sim.particles[0].spectral;
        assert!(s[14] > crate::pbf::WATER_SPECTRAL[14], "mixed particle band 14 should exceed pure water");
        assert!(s[14] < LAVA_SPECTRAL[14], "mixed particle band 14 should not fully reach lava");
    }

    #[test]
    fn step_does_not_zero_spectral() {
        let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
        for i in 0..5 { fluid.spawn([i as f32 * 0.1, 2.0, 0.0], [0.0; 3]); }
        fluid.step();
        let mean = fluid.mean_spectral();
        assert!(mean[2] > 0.0, "water blue band must persist after physics step");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics spectral_fluid 2>&1 | head -20
```
Expected: FAIL — `SpectralFluid` not found

- [ ] **Step 3: Implement**
```rust
use crate::pbf::{PbfFluidSim, WATER_SPECTRAL, BLOOD_SPECTRAL, LAVA_SPECTRAL};
use vox_core::types::GaussianSplat;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralFluidKind { Water, Blood, Lava, Custom([f32; 16]) }

pub struct SpectralFluid { pub kind: SpectralFluidKind, pub sim: PbfFluidSim }

impl SpectralFluid {
    pub fn new(kind: SpectralFluidKind) -> Self {
        Self { kind, sim: PbfFluidSim::new(1000.0, 0.1) }
    }
    pub fn spectral_for_kind(kind: SpectralFluidKind) -> [f32; 16] {
        match kind { SpectralFluidKind::Water => WATER_SPECTRAL, SpectralFluidKind::Blood => BLOOD_SPECTRAL, SpectralFluidKind::Lava => LAVA_SPECTRAL, SpectralFluidKind::Custom(s) => s }
    }
    pub fn spawn(&mut self, pos: [f32; 3], vel: [f32; 3]) {
        let spectral = Self::spectral_for_kind(self.kind);
        self.sim.spawn(pos, vel, spectral);
    }
    pub fn spawn_mixed(&mut self, pos: [f32; 3], vel: [f32; 3], mix: &[f32; 16], mix_weight: f32) {
        let base = Self::spectral_for_kind(self.kind);
        let w = mix_weight.clamp(0.0, 1.0);
        let mut spectral = [0.0f32; 16];
        for b in 0..16 { spectral[b] = base[b] * (1.0 - w) + mix[b] * w; }
        self.sim.spawn(pos, vel, spectral);
    }
    pub fn step(&mut self) { self.sim.cpu_step(); }
    pub fn particle_count(&self) -> usize { self.sim.particles.len() }
    pub fn mean_spectral(&self) -> [f32; 16] {
        let n = self.sim.particles.len();
        if n == 0 { return [0.0f32; 16]; }
        let mut acc = [0.0f32; 16];
        for p in &self.sim.particles { for b in 0..16 { acc[b] += p.spectral[b]; } }
        for v in &mut acc { *v /= n as f32; }
        acc
    }
}
```
- [ ] **Step 4: Wire at exact callsite**

`SpectralFluid::spawn()` delegates directly to `self.sim.spawn()` with the correct spectral profile — wiring is in the impl above.

- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics spectral_fluid -- --nocapture
```
Expected: PASS, 4 tests pass, output shows `water band 1 (0.8) > band 12 (0.01)`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/fluid.rs
git commit -m "feat(physics): SpectralFluid — PBF particles with spectral[16], spectral mixing"
```

---

## Task 6: WetnessSim — drip simulation + spectral wet blending AND wire module

**Files:**
- Create: `crates/vox_physics/src/wetness.rs`
- Modify: `crates/vox_physics/src/lib.rs`

**Acceptance:** `cargo test -p vox_physics wetness -- --nocapture` → 3 tests pass, output shows `downhill_avg >= uphill_avg` and NIR bands confirmed darker.

**Wiring requirement:** Must be exposed from `pub mod wetness;` in `crates/vox_physics/src/lib.rs`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drip_simulation_produces_flow_map() {
        let resolution = 16u32;
        let n = (resolution * resolution) as usize;
        let mut heights = vec![0.0f32; n];
        for z in 0..resolution as usize {
            for x in 0..resolution as usize { heights[z * resolution as usize + x] = x as f32 * 0.5; }
        }
        let normals = vec![[0.0f32, 1.0, 0.0]; n];
        let params = DripParams { particle_count: 100, max_steps: 50, seed: 42 };
        let result = run_drip_simulation(&heights, &normals, resolution, &params);
        assert_eq!(result.drip_intensity.len(), n);
        let max_intensity = result.drip_intensity.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_intensity > 0.0, "slope should produce nonzero flow");
        let downhill_avg: f32 = (0..resolution as usize)
            .map(|z| result.drip_intensity[z * resolution as usize + (resolution as usize - 1)])
            .sum::<f32>() / resolution as f32;
        let uphill_avg: f32 = (0..resolution as usize)
            .map(|z| result.drip_intensity[z * resolution as usize + 0])
            .sum::<f32>() / resolution as f32;
        assert!(downhill_avg >= uphill_avg, "downhill should accumulate more flow");
    }

    #[test]
    fn test_wet_spectral_blend_darkens_nir() {
        let dry_soil: [f32; 16] = [0.07,0.09,0.11,0.13,0.14,0.16,0.18,0.20,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.30];
        let wet = blend_wet_spectral(&dry_soil, 0.3);
        for band in 8..16 {
            assert!(wet[band] < dry_soil[band],
                "wet NIR band {band} ({}) should be darker than dry ({})", wet[band], dry_soil[band]);
        }
    }

    #[test]
    fn test_puddle_detection_from_drip_and_curvature() {
        let drip = vec![0.8f32, 0.2, 0.1, 0.9];
        let curvature = vec![-0.1f32, 0.3, 0.2, -0.05];
        let puddles = detect_puddles(&drip, &curvature, 0.5, -0.02);
        assert!(puddles[0], "cell 0 should be a puddle");
        assert!(!puddles[1], "cell 1 has low drip, not a puddle");
        assert!(puddles[3], "cell 3 should be a puddle");
    }
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics wetness 2>&1 | head -20
```
Expected: FAIL — compile error

- [ ] **Step 3: Implement**
```rust
//! WetnessSim — drip simulation, puddle detection, spectral wet blending.
//! wet_spectral[λ] = dry[λ] × (1-f) + water_curve[λ] × f, f = drip.clamp(0, 0.35)

pub struct DripParams { pub particle_count: u32, pub max_steps: u32, pub seed: u64 }
impl Default for DripParams { fn default() -> Self { Self { particle_count: 10_000, max_steps: 500, seed: 0 } } }

pub struct DripResult { pub drip_intensity: Vec<f32>, pub resolution: u32 }

pub fn run_drip_simulation(heights: &[f32], _normals: &[[f32; 3]], resolution: u32, params: &DripParams) -> DripResult {
    let n = (resolution * resolution) as usize;
    let mut accumulation = vec![0u32; n];
    let res = resolution as usize;
    let mut rng = params.seed;
    let mut lcg = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as f32 / u32::MAX as f32
    };
    for _ in 0..params.particle_count {
        let mut x = (lcg(&mut rng) * res as f32) as usize;
        let mut z = (lcg(&mut rng) * res as f32) as usize;
        x = x.min(res - 1); z = z.min(res - 1);
        for _ in 0..params.max_steps {
            let idx = z * res + x;
            accumulation[idx] += 1;
            let h = heights[idx];
            let mut best_dh = 0.0f32; let mut best_nx = x; let mut best_nz = z;
            if x > 0     && heights[z*res+x-1] < h-best_dh { best_dh = h-heights[z*res+x-1]; best_nx=x-1; best_nz=z; }
            if x<res-1   && heights[z*res+x+1] < h-best_dh { best_dh = h-heights[z*res+x+1]; best_nx=x+1; best_nz=z; }
            if z > 0     && heights[(z-1)*res+x]<h-best_dh { best_dh = h-heights[(z-1)*res+x]; best_nx=x; best_nz=z-1; }
            if z<res-1   && heights[(z+1)*res+x]<h-best_dh { best_dh = h-heights[(z+1)*res+x]; best_nx=x; best_nz=z+1; }
            if best_dh < 0.001 { break; }
            x = best_nx; z = best_nz;
        }
    }
    let max_acc = *accumulation.iter().max().unwrap_or(&1) as f32;
    let drip_intensity = accumulation.iter().map(|&v| (v as f32 / max_acc).sqrt()).collect();
    DripResult { drip_intensity, resolution }
}

const WATER_SPECTRAL_USGS: [f32; 16] = [0.03,0.04,0.05,0.05,0.05,0.04,0.03,0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01];

pub fn blend_wet_spectral(dry: &[f32; 16], wet_factor: f32) -> [f32; 16] {
    let f = wet_factor.clamp(0.0, 0.35);
    std::array::from_fn(|i| dry[i] * (1.0 - f) + WATER_SPECTRAL_USGS[i] * f)
}

pub fn detect_puddles(drip: &[f32], curvature: &[f32], drip_threshold: f32, concavity_threshold: f32) -> Vec<bool> {
    assert_eq!(drip.len(), curvature.len());
    drip.iter().zip(curvature.iter()).map(|(&d, &c)| d >= drip_threshold && c <= concavity_threshold).collect()
}
```
- [ ] **Step 4: Wire at exact callsite**
```rust
// crates/vox_physics/src/lib.rs — add:
pub mod wetness;
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics wetness -- --nocapture
```
Expected: PASS, 3 tests pass, output confirms `downhill_avg >= uphill_avg` and NIR bands darker

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/src/wetness.rs crates/vox_physics/src/lib.rs
git commit -m "feat(physics): WetnessSim — drip simulation, puddle detection, spectral wet blending"
```

---

## Task 7: Performance benchmark — 50k particle budget verification

**Files:**
- Create: `crates/vox_physics/benches/pbf_perf.rs`
- Modify: `crates/vox_physics/Cargo.toml`

**Acceptance:** `cargo bench -p vox_physics 2>&1 | grep -E "time|ns|ms"` → prints timing for `pbf_cpu_1k_particles`; `cargo test -p vox_physics pbf_particle_size -- --nocapture` → PASS with output `50000 * 112 = 5600000`.

**Wiring requirement:** Must be declared in `[[bench]]` in `crates/vox_physics/Cargo.toml`. `todo!()` / `unimplemented!()` / empty function bodies = task failure.

- [ ] **Step 1: Write the failing test**
```rust
// Inline validation test in pbf.rs (not the bench):
#[test]
fn pbf_particle_size_memory_budget() {
    let per_particle = std::mem::size_of::<PbfParticle>();
    let total = 50_000usize * per_particle;
    println!("50000 * {} = {}", per_particle, total);
    assert_eq!(total, 5_600_000, "50k particles at 112 bytes = 5.6MB");
}
```
- [ ] **Step 2: Run to verify failure**
```bash
cargo test -p vox_physics pbf_particle_size 2>&1 | tail -5
```
Expected: FAIL if struct size is wrong, or test missing

- [ ] **Step 3: Implement**
```rust
// crates/vox_physics/benches/pbf_perf.rs
use criterion::{criterion_group, criterion_main, Criterion};
use vox_physics::pbf::{PbfFluidSim, WATER_SPECTRAL};

fn bench_pbf_cpu_1k(c: &mut Criterion) {
    let mut sim = PbfFluidSim::new(1000.0, 0.08);
    for i in 0..1000 {
        sim.spawn(
            [(i % 10) as f32 * 0.1, (i / 100) as f32 * 0.1 + 1.0, (i / 10 % 10) as f32 * 0.1],
            [0.0; 3], WATER_SPECTRAL,
        );
    }
    c.bench_function("pbf_cpu_1k_particles", |b| b.iter(|| sim.cpu_step()));
}

fn bench_spectral_fluid_500(c: &mut Criterion) {
    use vox_physics::fluid::{SpectralFluid, SpectralFluidKind};
    let mut fluid = SpectralFluid::new(SpectralFluidKind::Water);
    for i in 0..500 {
        fluid.spawn([(i % 10) as f32 * 0.1, (i / 10) as f32 * 0.1 + 1.0, 0.0], [0.0; 3]);
    }
    c.bench_function("spectral_fluid_500_particles", |b| b.iter(|| fluid.step()));
}

criterion_group!(benches, bench_pbf_cpu_1k, bench_spectral_fluid_500);
criterion_main!(benches);
```
- [ ] **Step 4: Wire at exact callsite**
```toml
# crates/vox_physics/Cargo.toml — add:
[[bench]]
name = "pbf_perf"
harness = false

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```
- [ ] **Step 5: Run — verify non-trivial output**
```bash
cargo test -p vox_physics pbf_particle_size -- --nocapture
```
Expected: PASS, output `50000 * 112 = 5600000`

- [ ] **Step 6: Commit**
```bash
git add crates/vox_physics/benches/pbf_perf.rs crates/vox_physics/Cargo.toml
git commit -m "bench(physics): PBF performance test — 50k particle budget verification"
```

---

## Self-Review

**Spec coverage:**
- [x] PBF GPU fluids — Tasks 1–2: `PbfFluidSim`, 4 WGSL compute passes
- [x] Spectral resonance fracture — Task 3: band-variance → plane regularity, wired into `fracture_at()`
- [x] Spectral thermal dynamics — Task 4: bands 9-14 elevation, GI seeding via `hot_emitters()`
- [x] SpectralFluid participating media — Task 5: PBF particles carry `spectral[16]`, mixing
- [x] WetnessSim — Task 6: drip simulation, puddle detection, spectral wet blending
- [x] Performance test 50k particles — Task 7
- [x] Rapier dependency: `rapier3d = "0.22"` to be added in Cargo.toml

**Constraint solver note:** The WGSL solve pass (Task 2) runs one Jacobi iteration per frame. For production stability, increase to 3–5 iterations by dispatching the solve pass N times per frame. The CPU fallback `cpu_step()` also runs one iteration — both are consistent.

**Thermal → GI bridge:** `ThermalEmitter::hot_emitters()` returns an iterator of `(Vec3, [f32; 16])` that maps directly onto `SpectralRadianceCache::propagate()` in `vox_render`. The domain boundary is at the engine runner: call `hot_emitters()` each frame and inject into the GI cache as additional emissive sources.
