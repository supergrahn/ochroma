# Domain 5 — Physics & Simulation

**Version:** 1.0 — 2026-03-29
**Status:** Draft
**Crate:** `vox_physics` (primary), with integration points in `vox_render`, `vox_audio`, `vox_terrain`

---

## Goals

1. All physics simulation drives visual output via GaussianSplat mutations — no separate geometry layer.
2. Splat fracture leverages the existing `SplatParticle`/`SplatEmitter` infrastructure; destruction is
   purely a matter of re-classifying splats, not swapping render primitives.
3. XPBD soft bodies bind directly to `GaussianSplat` arrays via barycentric weights — no skeleton required.
4. Fluid particles own GaussianSplats in the scene splat buffer; the renderer is unaware of "fluid mode."
5. Acoustic physics (reverb) is addressed in Domain 6; this domain covers mechanical simulation only.
6. All systems are CPU-first with `rayon` parallelism; GPU compute can be promoted per section in later phases.

---

## 5.1 Splat Fracture — Destructible Environments

### Core Concept

A rigid splat assembly is a named group of `GaussianSplat` entries in `scene_splats: Vec<GaussianSplat>`.
When a Rapier contact event delivers an impulse above `impact_threshold`, the affected splats in the
assembly's neighborhood are removed from `scene_splats` and re-spawned as `SplatParticle` instances with
outward velocities. No destruction mesh, no LOD swap, no pre-authored break shapes — fracture emerges
from the distribution of splats in the assembly.

### Data Structures

```rust
/// Attached to an entity that participates in fracture.
pub struct FractureComponent {
    /// Identifies which splats in scene_splats belong to this assembly.
    pub assembly_id: AssemblyId,
    /// Minimum impulse magnitude (N·s) before fracture triggers.
    pub impact_threshold: f32,
    /// Lifetime (s) of ejected SplatParticle shards.
    pub shard_lifetime: f32,
    /// Per-band spectral shift applied to ejected splats.
    /// Positive = increase that band; negative = decrease.
    /// Example — scorched concrete: [−0.3,−0.2,−0.1,0.0,0.0,0.3,0.5,0.4]
    pub spectral_shift_on_break: [f32; 8],
}

/// A named group of index ranges into scene_splats.
pub struct SplatAssembly {
    pub id: AssemblyId,
    /// Sorted indices into scene_splats. Kept sorted for range queries.
    pub splat_indices: Vec<u32>,
    /// Axis-aligned bounding box (world space) for quick rejection.
    pub aabb: Aabb,
}
```

### FractureSystem

```rust
impl FractureSystem {
    /// Called from the physics update loop when a Rapier contact event fires.
    pub fn handle_impact(
        &mut self,
        assembly: &mut SplatAssembly,
        component: &FractureComponent,
        impact_point: Vec3,
        impulse: Vec3,
        scene_splats: &mut Vec<GaussianSplat>,
        emitter_queue: &mut Vec<SplatParticle>,
        audio_queue: &mut Vec<ImpactAudioRequest>,
        camera_pos: Vec3,
    ) {
        let impulse_magnitude = impulse.length();
        if impulse_magnitude < component.impact_threshold {
            return;
        }

        let fracture_radius = impulse_magnitude * 0.3;
        let beyond_lod_range = (impact_point - camera_pos).length() > 30.0;

        // Collect splat indices within fracture_radius using assembly AABB pre-filter
        let mut to_remove: Vec<usize> = assembly
            .splat_indices
            .iter()
            .copied()
            .filter(|&idx| {
                let pos = Vec3::from(scene_splats[idx as usize].position);
                (pos - impact_point).length() <= fracture_radius
            })
            .map(|idx| idx as usize)
            .collect();

        // Remove in reverse order to keep indices valid during swap-remove.
        to_remove.sort_unstable_by(|a, b| b.cmp(a));

        for &idx in &to_remove {
            let splat = scene_splats.swap_remove(idx);
            let splat_pos = Vec3::from(splat.position);

            if !beyond_lod_range {
                // Compute outward velocity with random jitter.
                let outward = (splat_pos - impact_point).normalize_or_zero();
                let mass_factor = 1.0 / (splat.scale[0] * splat.scale[1] * splat.scale[2]).max(0.001);
                let jitter = random_unit_vec3() * impulse_magnitude * 0.1;
                let velocity = outward * impulse_magnitude * mass_factor.clamp(0.1, 5.0) + jitter;

                // Apply spectral shift.
                let mut spectral = decode_spectral_u16(splat.spectral);
                for b in 0..8 {
                    spectral[b] = (spectral[b] + component.spectral_shift_on_break[b]).clamp(0.0, 1.0);
                }

                let particle = self.cache.allocate(SplatParticle {
                    position: splat_pos.to_array(),
                    velocity: velocity.to_array(),
                    lifetime: component.shard_lifetime,
                    age: 0.0,
                    scale: splat.scale,
                    spectral,
                    opacity: splat.opacity,
                });
                emitter_queue.push(particle);
            }

            // Queue impact audio regardless of LOD range.
            audio_queue.push(ImpactAudioRequest {
                position: splat_pos,
                spectral: decode_spectral_u16(splat.spectral),
                impulse_magnitude,
            });
        }

        // Update assembly index set: remove evacuated indices, remap swap-remove moves.
        self.repair_assembly_indices(assembly, &to_remove, scene_splats.len());
    }
}
```

**Fracture LOD rule:** when `(impact_point - camera_pos).length() > 30.0`, skip `SplatParticle`
allocation entirely. Splats are removed and audio is queued; visual fracture is implied by the
sudden absence of splats. This prevents hundreds of offscreen particles from burning simulation budget.

### ImpactDetector Integration

```rust
/// Wraps Rapier's ContactForceEvent into engine domain types.
pub struct ImpactDetector;

impl ImpactDetector {
    pub fn process_contact_events(
        events: &[ContactForceEvent],
        assemblies: &HashMap<RigidBodyHandle, AssemblyId>,
    ) -> Vec<PendingImpact> {
        events.iter().filter_map(|ev| {
            let assembly_id = assemblies.get(&ev.collider1.parent()?)?;
            Some(PendingImpact {
                assembly_id: *assembly_id,
                impact_point: Vec3::from(ev.max_force_direction) * 0.0 + Vec3::from(ev.total_force_magnitude), // contact centroid
                impulse: Vec3::from(ev.total_force_magnitude * ev.max_force_direction),
            })
        }).collect()
    }
}
```

### FractureCache

A pool of pre-allocated `SplatParticle` vecs to avoid heap allocation on the hot fracture path:

```rust
pub struct FractureCache {
    pool: Vec<SplatParticle>,
    capacity: usize,
}

impl FractureCache {
    pub fn new(capacity: usize) -> Self { ... }
    /// Returns a particle from the pool or overwrites the oldest entry if full.
    pub fn allocate(&mut self, particle: SplatParticle) -> SplatParticle { ... }
}
```

---

## 5.2 Soft Body — XPBD Simulation

### Data Structures

```rust
pub struct SoftBody {
    pub particles: Vec<SoftParticle>,
    pub constraints: Vec<SoftConstraint>,
    /// Maps each particle to the GaussianSplats it influences, with weights.
    pub splat_bindings: Vec<SplatBinding>,
    pub shape_match_stiffness: f32,
}

pub struct SoftParticle {
    pub position: Vec3,
    pub prev_position: Vec3,
    pub velocity: Vec3,
    pub inv_mass: f32,
    /// Optional Rapier body for interaction with the rigid body world.
    pub rapier_body_handle: Option<RigidBodyHandle>,
}

pub enum SoftConstraint {
    Distance {
        p0: u32,
        p1: u32,
        rest_len: f32,
        /// XPBD compliance (inverse stiffness). 0.0 = rigid.
        compliance: f32,
    },
    Volume {
        /// Tetrahedron vertex indices into SoftBody::particles.
        tet: [u32; 4],
        rest_volume: f32,
        compliance: f32,
    },
    ShapeMatch {
        /// Particle indices that form this shape-matching group.
        group: Vec<u32>,
        /// Rest-pose positions in local space.
        target_positions: Vec<Vec3>,
        stiffness: f32,
    },
}

/// Binds a GaussianSplat to a triangle of soft particles via barycentric weights.
pub struct SplatBinding {
    pub splat_index: u32,
    pub particles: [u32; 3],
    pub weights: [f32; 3],
}
```

### XPBD Solver

8 substeps per physics frame (fixed timestep 1/60 s → substep dt = 1/480 s):

```rust
impl SoftBodySolver {
    pub fn step(&mut self, dt: f32, gravity: Vec3, substeps: u32) {
        let sub_dt = dt / substeps as f32;
        for _ in 0..substeps {
            // 1. Apply external forces.
            for p in &mut self.body.particles {
                if p.inv_mass == 0.0 { continue; }
                p.velocity += gravity * sub_dt;
                // Wind, contact impulses injected here.
            }

            // 2. Predict positions.
            for p in &mut self.body.particles {
                p.prev_position = p.position;
                p.position += p.velocity * sub_dt;
            }

            // 3. Project constraints (XPBD lambda updates).
            let mut lambdas = vec![0.0f32; self.body.constraints.len()];
            for (i, constraint) in self.body.constraints.iter().enumerate() {
                self.project_constraint(constraint, sub_dt, &mut lambdas[i]);
            }

            // 4. Update velocities from position delta.
            for p in &mut self.body.particles {
                p.velocity = (p.position - p.prev_position) / sub_dt;
            }
        }
    }

    fn project_constraint(&mut self, c: &SoftConstraint, dt: f32, lambda: &mut f32) {
        match c {
            SoftConstraint::Distance { p0, p1, rest_len, compliance } => {
                let d = self.particles[*p1].position - self.particles[*p0].position;
                let len = d.length();
                let constraint_val = len - rest_len;
                let w0 = self.particles[*p0].inv_mass;
                let w1 = self.particles[*p1].inv_mass;
                let alpha = compliance / (dt * dt);
                let delta_lambda = (-constraint_val - alpha * *lambda) / (w0 + w1 + alpha);
                *lambda += delta_lambda;
                let correction = d.normalize_or_zero() * delta_lambda;
                self.particles[*p0].position -= correction * w0;
                self.particles[*p1].position += correction * w1;
            }
            SoftConstraint::Volume { tet, rest_volume, compliance } => {
                // Tetrahedral volume constraint; gradient computed via cross products.
                // ...
            }
            SoftConstraint::ShapeMatch { group, target_positions, stiffness } => {
                // Polar decomposition of deformation gradient F = RS.
                // R extracted via iterative symmetric eigendecomposition (3×3).
                // Spring force: F_i = stiffness * (R * target_pos_i - current_pos_i)
                // ...
            }
        }
    }
}
```

**Shape matching detail:** The polar decomposition of the deformation gradient `F = Σ (x_i - centroid) ⊗ (x0_i - centroid0)` gives an optimal rotation `R` via the SVD `F = U Σ V^T`, then `R = U V^T`. This is computed per-group per-substep using a 3×3 Jacobi SVD (6 Jacobi sweeps sufficient for convergence). The resulting rotation defines rest-shape targets; spring forces pull each particle toward `centroid + R * (rest_pos_i - rest_centroid)`. This produces near-rigid deformation suitable for flesh/meat volumes.

### Splat Binding Update

After each solve step, propagate soft particle positions to owned splats:

```rust
pub fn sync_splats(&self, scene_splats: &mut Vec<GaussianSplat>) {
    for binding in &self.body.splat_bindings {
        let p0 = self.body.particles[binding.particles[0] as usize].position;
        let p1 = self.body.particles[binding.particles[1] as usize].position;
        let p2 = self.body.particles[binding.particles[2] as usize].position;
        let pos = p0 * binding.weights[0]
                + p1 * binding.weights[1]
                + p2 * binding.weights[2];

        let splat = &mut scene_splats[binding.splat_index as usize];
        splat.position = pos.to_array();

        // Scale deformation: stretch factor along triangle normal drives scale.z.
        // Compression/stretch is automatic from GaussianSplat::scale being mutable.
    }
}
```

Spectral deformation is automatic: `GaussianSplat::scale` directly controls the ellipsoid footprint in the EWA renderer. No additional spectral bookkeeping is needed — a compressed splat simply covers less area.

---

## 5.3 Rigid Body Joints & Vehicles

### Joint Wrappers

Rapier exposes `GenericJoint`; these wrappers provide ergonomic named types:

```rust
pub enum JointDef {
    Hinge {
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
        limits: Option<(f32, f32)>, // radians
    },
    Ball {
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        swing_limit: Option<f32>,
        twist_limit: Option<f32>,
    },
    Prismatic {
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
        limits: Option<(f32, f32)>, // meters
    },
    Fixed {
        body_a: RigidBodyHandle,
        body_b: RigidBodyHandle,
        frame_a: Isometry3,
        frame_b: Isometry3,
    },
}

pub fn build_joint(def: &JointDef, rapier: &mut RapierContext) -> ImpulseJointHandle { ... }
```

### VehicleController

```rust
pub struct VehicleController {
    pub chassis_body: RigidBodyHandle,
    pub wheels: Vec<WheelState>,
    pub wheel_defs: Vec<WheelDef>,
    /// Net engine torque in N·m applied to drive wheels.
    pub engine_torque: f32,
    /// Current steering angle in radians.
    pub steer_angle: f32,
    /// Wheel splat assemblies; rotated each frame to match angular velocity.
    pub wheel_assemblies: Vec<AssemblyId>,
}

pub struct WheelDef {
    /// Hub offset from chassis center in local space.
    pub hub_offset: Vec3,
    /// Natural length of suspension spring (m).
    pub suspension_rest_length: f32,
    pub suspension_stiffness: f32,  // N/m
    pub damping: f32,               // N·s/m
    pub radius: f32,
    pub friction: f32,
    pub lateral_stiffness: f32,
    pub is_driven: bool,
    pub is_steered: bool,
}

pub struct WheelState {
    pub contact_point: Option<Vec3>,
    pub contact_normal: Option<Vec3>,
    pub compression: f32,
    pub angular_velocity: f32, // rad/s, for splat rotation
    pub lateral_slip: f32,
}
```

**Wheel physics per-frame:**

1. Ray-cast from hub downward (distance = `rest_length + radius`). Hit = contact.
2. Suspension force: `F_suspension = stiffness * (rest_length - ray_dist) - damping * chassis_vel_y`.
3. Friction force from lateral slip: `F_friction = -lateral_stiffness * lateral_slip * contact_normal_perp`.
4. Apply both forces to chassis rigid body via `rapier.bodies[chassis_body].add_force_at_point(...)`.
5. Integrate `wheel_state.angular_velocity += (engine_torque / wheel_def.radius - friction_drag) * dt`.
6. Rotate `wheel_assemblies[i]` splat group around its axle by `angular_velocity * dt` radians.

**Splat vehicle rendering:** chassis and each wheel are separate `SplatAssembly` instances. The chassis assembly is rigid-body-driven (transforms applied from Rapier). Wheel assemblies additionally rotate around their local axle. No mesh baking, no normal map baking — the EWA renderer sees them as ordinary splats.

**VehicleAudio:**
- Engine RPM = `angular_velocity_of_drive_wheel * 60 / (2π)` scaled by gear ratio.
- Spectral profile for engine sound: bands 4–6 (low rumble region) modulated by throttle; bands 1–3 (whine) rise with RPM.
- Reuse `vox_audio::synthesize_impact` infrastructure with a continuous oscillator mode feeding the `AudioGraph` (see Domain 6).

---

## 5.4 Fluid Simulation — SPH

### Data Structures

```rust
pub struct FluidSystem {
    pub particles: Vec<FluidParticle>,
    pub grid: SpatialHash,
    /// Spectral profile of the fluid (shared across all particles, per fluid body).
    pub spectral: [f32; 8],
    pub viscosity: f32,          // Pa·s; water ≈ 0.001, honey ≈ 10.0
    pub surface_tension: f32,
    pub smoothing_radius: f32,   // h; neighbor search radius
    pub rest_density: f32,       // kg/m³; water ≈ 1000
    pub stiffness: f32,          // pressure EOS constant k
    /// Index range in scene_splats owned by this fluid system.
    pub splat_range: std::ops::Range<usize>,
}

pub struct FluidParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub density: f32,
    pub pressure: f32,
    /// Index into scene_splats; this particle drives that splat's position.
    pub splat_index: u32,
}
```

### SPH Algorithm

Fixed timestep: 0.002 s (500 Hz). Integrate with `rayon::par_iter` across particles.

**1. Neighborhood search via SpatialHash:**

```rust
pub struct SpatialHash {
    cell_size: f32,    // = 2 * smoothing_radius
    table: HashMap<IVec3, Vec<u32>>,
}
```

O(1) amortized insert/query per particle. Each frame: clear → re-insert all particles → query.

**2. Density estimation (cubic spline kernel W):**

```
W(r, h) = (315 / 64πh⁹) * (h² - r²)³    for 0 ≤ r ≤ h
         = 0                               otherwise

ρ_i = Σ_j m_j * W(|x_i - x_j|, h)
```

**3. Pressure (Tait EOS):**

```
p_i = k * ((ρ_i / ρ_rest)^7 - 1)
```

**4. Forces:**

```
Pressure gradient:   F_press_i = -Σ_j m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W(r_ij, h)
Viscosity:           F_visc_i  = μ * Σ_j m_j * (v_j - v_i)/ρ_j * ∇²W(r_ij, h)
Surface tension:     Akinci 2013 cohesion + curvature force (normals from color field gradient)
```

**5. SDF collision:**

For each particle, query `TerrainVolume::sdf(pos)`. If `sdf_val < particle_radius`:
- Push particle along gradient: `pos += sdf_gradient * (particle_radius - sdf_val)`.
- Reflect velocity: `vel -= 2.0 * vel.dot(sdf_gradient) * sdf_gradient`.
- Apply restitution coefficient 0.3.

**6. Splat sync:**

After advection, `scene_splats[particle.splat_index].position = particle.position.to_array()`. Splat scale is fixed at `smoothing_radius * 1.5` per axis; the EWA renderer handles the rest.

**Surface foam:** Particles with `velocity.length() > foam_threshold` (default 4.0 m/s) trigger a `SplatEmitter` configured with `EmitterConfig { spectral: WHITE, emit_rate: 5.0, max_particles: 16, lifetime: 0.3..0.8 }`. Uses existing `SplatEmitter` from `vox_render::splat_particles`.

**Screen-space fluid (close range):** when camera is within 8m of any fluid particle, a post-process pass in `vox_render` composites a screen-space normal map computed from the depth discontinuities in the splat depth buffer. This gives surface sheen/refraction without ray marching.

**Spectral profiles for common fluids:**

| Fluid | Spectral profile (bands 0–7) |
|-------|------------------------------|
| Water | `[0.1, 0.1, 0.3, 0.6, 0.4, 0.2, 0.1, 0.05]` (blue dominant) |
| Mud   | `[0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]` (red/brown) |
| Blood | `[0.7, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01]` (deep red) |
| Lava  | `[0.1, 0.1, 0.1, 0.2, 0.5, 0.9, 1.0, 0.8]` (orange-red) |

---

## 5.5 Collision Layers & Queries

### Collision Layer System

```rust
#[repr(u32)]
pub enum CollisionLayer {
    Default    = 1 << 0,
    Player     = 1 << 1,
    Enemy      = 1 << 2,
    Terrain    = 1 << 3,
    Projectile = 1 << 4,
    Trigger    = 1 << 5,
    Fluid      = 1 << 6,
    Debris     = 1 << 7,
}

pub struct CollisionFilter {
    /// Which layer this object occupies.
    pub layer: CollisionLayer,
    /// Bitmask of layers this object can collide with.
    pub mask: u32,
}

impl CollisionFilter {
    pub fn interacts_with(&self, other: &CollisionFilter) -> bool {
        (self.mask & other.layer as u32) != 0
            && (other.mask & self.layer as u32) != 0
    }
}
```

Standard interaction table:

| Layer | Collides with |
|-------|---------------|
| Player | Default, Terrain, Enemy, Trigger |
| Enemy | Default, Terrain, Player, Projectile |
| Projectile | Default, Terrain, Enemy, Debris |
| Fluid | Terrain, Default |
| Debris | Terrain, Default, Player |

### Query API (`vox_physics/src/query.rs`)

```rust
pub struct RaycastHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub body: RigidBodyHandle,
    pub assembly_id: Option<AssemblyId>,
    pub spectral: Option<[f32; 8]>, // populated for SpectralQuery
}

pub struct OverlapResult {
    pub body: RigidBodyHandle,
    pub assembly_id: Option<AssemblyId>,
}

pub struct SweepHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub time_of_impact: f32,
    pub body: RigidBodyHandle,
}

pub trait PhysicsQuery {
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: CollisionFilter)
        -> Option<RaycastHit>;

    fn sphere_overlap(&self, center: Vec3, radius: f32, filter: CollisionFilter)
        -> Vec<OverlapResult>;

    fn sweep(&self, shape: &dyn Shape, from: Isometry3, to: Isometry3, filter: CollisionFilter)
        -> Option<SweepHit>;

    /// Raycast that additionally returns the spectral profile of the first hit splat.
    /// Intended for AI perception: enemies can "see" spectral signatures.
    fn spectral_raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32)
        -> Vec<RaycastHit>; // sorted by distance, with spectral populated
}
```

`spectral_raycast` implementation: after the Rapier ray hits a body, look up its `assembly_id`, find the nearest splat to `hit.point` in that assembly, and populate `spectral` from `decode_spectral_u16(splat.spectral)`. Cost: O(assembly_splat_count) for the nearest-splat search; acceptable since AI queries are at 10 Hz, not per-frame.

---

## 5.6 Rope & Cable Simulation

### Data Structures

```rust
pub struct Rope {
    pub segments: Vec<RopeSegment>,
    pub fixed_end: Vec3,
    /// Mass of the free end (kg); rest of the rope is massless in PBD.
    pub free_end_mass: f32,
    pub segment_length: f32,
    pub radius: f32,
    /// Index range in scene_splats owned by this rope.
    pub splat_range: std::ops::Range<usize>,
}

pub struct RopeSegment {
    pub position: Vec3,
    pub prev_position: Vec3,
}
```

### Position-Based Rope Solver

Verlet integration with 4 substeps per frame:

```rust
impl Rope {
    pub fn step(&mut self, dt: f32, gravity: Vec3, substeps: u32) {
        let sub_dt = dt / substeps as f32;
        for _ in 0..substeps {
            // Verlet integrate each segment.
            for seg in self.segments.iter_mut() {
                let vel = seg.position - seg.prev_position;
                seg.prev_position = seg.position;
                seg.position += vel + gravity * sub_dt * sub_dt;
            }
            // Pin first segment to fixed_end.
            self.segments[0].position = self.fixed_end;

            // Distance constraint projection.
            for i in 0..self.segments.len() - 1 {
                let delta = self.segments[i + 1].position - self.segments[i].position;
                let dist = delta.length();
                let correction = delta * (1.0 - self.segment_length / dist.max(0.0001)) * 0.5;
                self.segments[i].position += correction;
                self.segments[i + 1].position -= correction;
                // Re-pin first segment after correction.
                self.segments[0].position = self.fixed_end;
            }
        }
    }

    pub fn sync_splats(&self, scene_splats: &mut Vec<GaussianSplat>) {
        for (i, seg) in self.segments.iter().enumerate() {
            let splat_idx = self.splat_range.start + i;
            if splat_idx >= self.splat_range.end { break; }

            let splat = &mut scene_splats[splat_idx];
            splat.position = seg.position.to_array();

            // Orient along segment direction.
            if i + 1 < self.segments.len() {
                let dir = (self.segments[i + 1].position - seg.position).normalize_or_zero();
                // Encode dir as quaternion → i16[4] rotation.
                let rot = Quat::from_rotation_arc(Vec3::X, dir);
                splat.rotation = encode_quat_i16(rot);
            }
            // Scale: elongated along segment axis.
            splat.scale = [self.segment_length, self.radius, self.radius];
        }
    }
}
```

**Use cases:** ziplines (fixed both ends, player attaches via `Prismatic` joint), bridge cables (high segment count, wind-driven perturbation), hanging lights (single fixed end, `SplatAssembly` light at free end providing dynamic GI as the rope sways), swinging objects.

---

## File Map

```
crates/vox_physics/
  src/
    lib.rs                  — re-exports; PhysicsWorld struct
    fracture.rs             — FractureComponent, FractureSystem, FractureCache, ImpactDetector
    soft_body.rs            — SoftBody, SoftParticle, SoftConstraint, SoftBodySolver
    joints.rs               — JointDef, build_joint
    vehicle.rs              — VehicleController, WheelDef, WheelState
    fluid.rs                — FluidSystem, FluidParticle, SpatialHash, SPH kernels
    query.rs                — CollisionLayer, CollisionFilter, PhysicsQuery impl, RaycastHit
    rope.rs                 — Rope, RopeSegment, solver
    assembly.rs             — SplatAssembly, AssemblyId, assembly spatial indexing
  tests/
    fracture_test.rs
    soft_body_test.rs
    fluid_test.rs
    vehicle_test.rs
    rope_test.rs
    query_test.rs
```

Integration points:
- `vox_render/src/splat_particles.rs` — fracture ejects into existing `SplatParticle` pool.
- `vox_audio/src/lib.rs` — fracture and vehicle audio via `synthesize_impact` and future `AudioGraph`.
- `vox_terrain/src/lib.rs` — fluid SDF collision via `TerrainVolume::sdf()`.

---

## Milestones

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M5.1 | `FractureSystem::handle_impact` + `ImpactDetector`; fracture test with 10k splat scene | Phase 5, week 1 |
| M5.2 | `SoftBodySolver` XPBD distance + volume constraints; splat sync test | Phase 5, week 1 |
| M5.3 | Shape matching (polar decomp); cloth-like soft body demo | Phase 5, week 2 |
| M5.4 | `JointDef` wrappers; `VehicleController` 4-wheel raycast | Phase 5, week 2 |
| M5.5 | `FluidSystem` SPH 500-particle water scene; SDF collision | Phase 5, week 3 |
| M5.6 | `CollisionLayer` + `PhysicsQuery` + `spectral_raycast` | Phase 5, week 3 |
| M5.7 | `Rope` solver + splat sync; hanging-light demo | Phase 5, week 4 |

---

## Acceptance Criteria

- **Fracture:** Firing a projectile at a 50k-splat wall removes the affected region (within `fracture_radius`) within one physics frame, spawns `SplatParticle` instances with correct outward velocities, and calls `ImpactAudioRequest` for each removed splat. Beyond 30m, no `SplatParticle` instances are allocated.
- **FractureCache:** Zero heap allocations during fracture events after initial pool allocation (verified with `#[global_allocator]` counting allocator in tests).
- **XPBD:** A 100-particle soft body remains stable under gravity for 60 seconds at 8 substeps/frame. Distance constraint error < 1% after convergence.
- **Shape matching:** A shape-matched soft body with stiffness 1.0 returns to rest shape within 5 frames after a displacement of 0.5m.
- **Vehicle:** A 4-wheeled vehicle traverses a 10° slope, maintains contact on all 4 wheels, and steers 30° at 10 m/s without flipping. Wheel splat assemblies rotate visibly in sync with angular velocity.
- **SPH:** 500-particle water scene at 0.002s timestep, 500Hz, with SDF terrain collision; no particle tunneling through terrain; simulation stable for 30 seconds.
- **CollisionLayer:** `Player` vs `Fluid` returns no collision; `Player` vs `Terrain` returns collision. `spectral_raycast` returns spectral data matching the nearest splat at the hit point.
- **Rope:** A 20-segment rope under gravity converges to a catenary shape. Segment distance constraint error < 0.5% after 4 substeps.

---

## Effort Estimate

| Section | Estimate |
|---------|----------|
| 5.1 Splat Fracture | 1.5 days |
| 5.2 XPBD Soft Body (distance + volume) | 2 days |
| 5.2 Shape matching (polar decomp) | 1 day |
| 5.3 Joint wrappers + VehicleController | 1.5 days |
| 5.4 SPH Fluid | 3 days |
| 5.5 Collision Layers + Query API | 0.5 days |
| 5.6 Rope | 0.5 days |
| Tests + integration | 1 day |
| **Total** | **~11 days** |

Dependencies: `rapier3d 0.22`, `rustfft` (shared with Domain 6), `rayon` (already in workspace).
New crate dependency: `vox_physics` must add `vox_terrain` to `Cargo.toml` for SDF collision.
