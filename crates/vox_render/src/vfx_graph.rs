//! Node-based splat-particle VFX — Ochroma's spectral analogue of UE Niagara /
//! Unity VFX Graph, where particles **are** Gaussian splats with spectral
//! emission.
//!
//! # Why this module exists
//!
//! `vox_render` historically carried five overlapping particle paths:
//!
//! - [`crate::particles`]      — `ParticleEmitter`, RGB-free 16-band spectral.
//! - [`crate::splat_particles`]— `SplatEmitter`, splats-as-particles + audio.
//! - [`crate::gpu_particles`]  — `GpuParticleSystem`, GPU-buffer SoA + CPU fallback.
//! - [`crate::particle_ecs`]   — bevy ECS components.
//! - [`crate::vfx`]            — declarative `VfxEffect` (rate/shape/curve emitters).
//!
//! This module **unifies** them under one authoring model: a typed node DAG
//! ([`VfxGraph`]) that describes an effect, plus a deterministic interpreter
//! ([`VfxGraphInstance`]) that advances particle state in Struct-of-Arrays
//! storage and emits a `Vec<GaussianSplat>` per frame.
//!
//! The existing modules are **reused, not rewritten**:
//!
//! - The 16-band spectral particle concept comes straight from
//!   [`crate::particles::Particle::spectral`] and
//!   [`crate::splat_particles::SplatParticle::spectral`].
//! - Splat construction reuses [`GaussianSplat::volume`], exactly as every
//!   existing path does (`particles::to_splats`, `splat_particles::to_splat`,
//!   `gpu_particles::to_splats`, `particle_ecs::particles_to_splats`).
//! - The emit-accumulator + xorshift/LCG RNG idiom is the same deterministic
//!   pattern used by `SplatEmitter` and `VfxInstance`.
//! - The graph node/validation idiom mirrors the editor's `node_graph` (typed
//!   ports, cycle rejection) but is **self-contained** — no `vox_editor` dep,
//!   keeping the engine crate independent.
//!
//! # Determinism
//!
//! Given the same `seed` and the same `dt` sequence, a [`VfxGraphInstance`]
//! produces bit-identical splats. RNG is a per-instance LCG seeded from the
//! graph seed; no wall-clock, no global state. This makes effects replay- and
//! netcode-safe.

use glam::{Quat, Vec3};
use half::f16;
use vox_core::spectral::BAND_WAVELENGTHS;
use vox_core::types::GaussianSplat;

// ===========================================================================
// Node graph model
// ===========================================================================

/// Spawn region geometry for new particles.
#[derive(Debug, Clone, PartialEq)]
pub enum SpawnShape {
    /// All particles spawn at the effect origin.
    Point,
    /// Uniformly within a sphere of `radius`.
    Sphere { radius: f32 },
    /// Within a cone opening `angle` degrees around +Y, base `radius`.
    Cone { angle_deg: f32, radius: f32 },
}

/// How spectral emission is authored for a particle.
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralEmission {
    /// Explicit 16-band spectral power distribution (values in [0, 1]).
    Spd([f32; 16]),
    /// Planckian blackbody radiator at `kelvin`, normalised so the brightest
    /// band == 1.0. Low kelvin (1800K) → red/IR dominant; high → blue.
    Blackbody { kelvin: f32 },
}

impl SpectralEmission {
    /// Resolve to a concrete 16-band SPD in [0, 1].
    pub fn to_spd(&self) -> [f32; 16] {
        match self {
            SpectralEmission::Spd(s) => *s,
            SpectralEmission::Blackbody { kelvin } => blackbody_spd(*kelvin),
        }
    }
}

/// A single typed node in a [`VfxGraph`].
///
/// Nodes form a DAG validated into four stages: exactly one [`Spawn`], then
/// [`Init`]s, then [`Update`]s, then exactly one [`Output`]. Edges are implicit
/// in declaration order (Niagara's "stack" model) but the typing is enforced by
/// [`VfxGraph::validate`].
///
/// [`Spawn`]: VfxNode::Spawn
/// [`Init`]: VfxNode::Init
/// [`Update`]: VfxNode::Update
/// [`Output`]: VfxNode::Output
#[derive(Debug, Clone, PartialEq)]
pub enum VfxNode {
    // --- Spawn stage ---
    /// Continuous emission at `rate` particles/second from `shape`.
    Spawn { rate: f32, shape: SpawnShape },
    /// One-shot burst of `count` particles at t=0 from `shape`.
    SpawnBurst { count: u32, shape: SpawnShape },

    // --- Init stage (run once when a particle is born) ---
    /// Initial velocity: `direction * speed`, perturbed by `spread`
    /// (radians-ish cone half-width in each axis).
    InitVelocity { direction: [f32; 3], speed: f32, spread: f32 },
    /// Lifetime sampled uniformly in `[min, max]` seconds.
    InitLifetime { min: f32, max: f32 },
    /// Constant initial splat half-axis size.
    InitSize { size: f32 },
    /// Initial spectral emission.
    InitSpectral { emission: SpectralEmission },

    // --- Update stage (run every step) ---
    /// Constant acceleration (m/s²), e.g. `[0,-9.81,0]` gravity.
    Gravity { accel: [f32; 3] },
    /// Velocity damping: `v *= (1 - coefficient*dt)`.
    Drag { coefficient: f32 },
    /// Hash-noise turbulence: per-particle pseudo-random acceleration of
    /// magnitude `strength`, varying with position and `frequency`.
    Turbulence { strength: f32, frequency: f32 },
    /// Multiply total spectral energy by `1.0` at birth down to `end_scale` at
    /// death (linear in normalised age).
    SpectralFadeOverLife { end_scale: f32 },
    /// Scale size from `1.0` at birth to `end_scale` at death.
    SizeOverLife { end_scale: f32 },

    // --- Output stage ---
    /// Emit one [`GaussianSplat`] per live particle per frame. `opacity` is the
    /// base opacity (0..=255) modulated by remaining-life fraction.
    Output { base_opacity: u8 },
}

/// Stage classification for typed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStage {
    Spawn,
    Init,
    Update,
    Output,
}

impl VfxNode {
    /// The pipeline stage this node belongs to.
    pub fn stage(&self) -> NodeStage {
        match self {
            VfxNode::Spawn { .. } | VfxNode::SpawnBurst { .. } => NodeStage::Spawn,
            VfxNode::InitVelocity { .. }
            | VfxNode::InitLifetime { .. }
            | VfxNode::InitSize { .. }
            | VfxNode::InitSpectral { .. } => NodeStage::Init,
            VfxNode::Gravity { .. }
            | VfxNode::Drag { .. }
            | VfxNode::Turbulence { .. }
            | VfxNode::SpectralFadeOverLife { .. }
            | VfxNode::SizeOverLife { .. } => NodeStage::Update,
            VfxNode::Output { .. } => NodeStage::Output,
        }
    }
}

/// An explicit directed edge between two nodes (by index). The interpreter does
/// not need edges (stage order is canonical), but they let an editor draw the
/// graph and let [`VfxGraph::validate`] reject cycles and stage-violating wires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VfxEdge {
    pub from: usize,
    pub to: usize,
}

/// A typed node DAG describing one splat-particle effect.
#[derive(Debug, Clone)]
pub struct VfxGraph {
    pub name: String,
    pub nodes: Vec<VfxNode>,
    pub edges: Vec<VfxEdge>,
    /// Determinism seed.
    pub seed: u64,
}

/// Errors produced by [`VfxGraph::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfxGraphError {
    /// Graph has no spawn node — nothing would ever be born.
    NoSpawn,
    /// Graph has no output node — nothing would render.
    NoOutput,
    /// More than one spawn node (ambiguous emission).
    MultipleSpawn,
    /// More than one output node.
    MultipleOutput,
    /// An edge references an out-of-range node index.
    EdgeOutOfBounds { edge: VfxEdge },
    /// An edge wires a later stage back into an earlier one (type mismatch):
    /// the stage order Spawn→Init→Update→Output is violated.
    StageMismatch { from: NodeStage, to: NodeStage },
    /// The directed graph contains a cycle.
    Cycle,
}

impl std::fmt::Display for VfxGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfxGraphError::NoSpawn => write!(f, "graph has no Spawn node"),
            VfxGraphError::NoOutput => write!(f, "graph has no Output node"),
            VfxGraphError::MultipleSpawn => write!(f, "graph has more than one Spawn node"),
            VfxGraphError::MultipleOutput => write!(f, "graph has more than one Output node"),
            VfxGraphError::EdgeOutOfBounds { edge } => {
                write!(f, "edge {}->{} references a missing node", edge.from, edge.to)
            }
            VfxGraphError::StageMismatch { from, to } => {
                write!(f, "type mismatch: cannot wire {from:?} stage into {to:?} stage")
            }
            VfxGraphError::Cycle => write!(f, "graph contains a cycle"),
        }
    }
}

impl std::error::Error for VfxGraphError {}

impl VfxGraph {
    /// Validate node/edge typing and acyclicity. Returns the canonical stage
    /// order required by [`VfxGraphInstance`] on success.
    pub fn validate(&self) -> Result<(), VfxGraphError> {
        let spawn_count = self
            .nodes
            .iter()
            .filter(|n| n.stage() == NodeStage::Spawn)
            .count();
        let output_count = self
            .nodes
            .iter()
            .filter(|n| n.stage() == NodeStage::Output)
            .count();

        if spawn_count == 0 {
            return Err(VfxGraphError::NoSpawn);
        }
        if spawn_count > 1 {
            return Err(VfxGraphError::MultipleSpawn);
        }
        if output_count == 0 {
            return Err(VfxGraphError::NoOutput);
        }
        if output_count > 1 {
            return Err(VfxGraphError::MultipleOutput);
        }

        // Edge bounds + stage-order typing.
        let stage_rank = |s: NodeStage| match s {
            NodeStage::Spawn => 0,
            NodeStage::Init => 1,
            NodeStage::Update => 2,
            NodeStage::Output => 3,
        };
        for edge in &self.edges {
            if edge.from >= self.nodes.len() || edge.to >= self.nodes.len() {
                return Err(VfxGraphError::EdgeOutOfBounds { edge: *edge });
            }
            let from = self.nodes[edge.from].stage();
            let to = self.nodes[edge.to].stage();
            // Data flows strictly forward through stages. Equal-stage wires are
            // allowed (e.g. chaining two Update nodes). A backward wire is a
            // type mismatch.
            if stage_rank(to) < stage_rank(from) {
                return Err(VfxGraphError::StageMismatch { from, to });
            }
        }

        if self.has_cycle() {
            return Err(VfxGraphError::Cycle);
        }
        Ok(())
    }

    /// Kahn's-algorithm cycle detection over the declared edges.
    fn has_cycle(&self) -> bool {
        let n = self.nodes.len();
        let mut indeg = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges {
            if e.from < n && e.to < n {
                adj[e.from].push(e.to);
                indeg[e.to] += 1;
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut visited = 0usize;
        while let Some(node) = queue.pop() {
            visited += 1;
            for &next in &adj[node] {
                indeg[next] -= 1;
                if indeg[next] == 0 {
                    queue.push(next);
                }
            }
        }
        visited != n
    }

    // --- node accessors used by the interpreter ---

    fn spawn(&self) -> &VfxNode {
        self.nodes
            .iter()
            .find(|n| n.stage() == NodeStage::Spawn)
            .expect("validated graph has a spawn node")
    }

    fn output_opacity(&self) -> u8 {
        for n in &self.nodes {
            if let VfxNode::Output { base_opacity } = n {
                return *base_opacity;
            }
        }
        255
    }
}

// ===========================================================================
// Interpreter — deterministic SoA particle state
// ===========================================================================

/// Live particle state in Struct-of-Arrays form (cache-friendly, mirrors the
/// GPU layout in [`crate::gpu_particles::GpuParticle`]).
#[derive(Debug, Clone, Default)]
struct ParticleSoa {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,
    age: Vec<f32>,
    lifetime: Vec<f32>,
    size: Vec<f32>,
    /// Base (birth) spectral SPD, never mutated; fade is applied at emit time.
    spectral: Vec<[f32; 16]>,
}

impl ParticleSoa {
    fn len(&self) -> usize {
        self.age.len()
    }

    fn push(&mut self, pos: Vec3, vel: Vec3, lifetime: f32, size: f32, spectral: [f32; 16]) {
        self.pos_x.push(pos.x);
        self.pos_y.push(pos.y);
        self.pos_z.push(pos.z);
        self.vel_x.push(vel.x);
        self.vel_y.push(vel.y);
        self.vel_z.push(vel.z);
        self.age.push(0.0);
        self.lifetime.push(lifetime);
        self.size.push(size);
        self.spectral.push(spectral);
    }

    fn swap_remove(&mut self, i: usize) {
        self.pos_x.swap_remove(i);
        self.pos_y.swap_remove(i);
        self.pos_z.swap_remove(i);
        self.vel_x.swap_remove(i);
        self.vel_y.swap_remove(i);
        self.vel_z.swap_remove(i);
        self.age.swap_remove(i);
        self.lifetime.swap_remove(i);
        self.size.swap_remove(i);
        self.spectral.swap_remove(i);
    }
}

/// A running instance of a [`VfxGraph`]. Holds particle SoA state and a
/// deterministic RNG. Call [`step`](Self::step) each frame, then
/// [`emit_splats`](Self::emit_splats) to get the frame's renderable splats.
pub struct VfxGraphInstance {
    graph: VfxGraph,
    origin: Vec3,
    particles: ParticleSoa,
    accumulator: f32,
    burst_fired: bool,
    rng: u64,
    time: f32,
    max_particles: usize,
    // resolved Init parameters (cached at construction)
    init_dir: Vec3,
    init_speed: f32,
    init_spread: f32,
    life_min: f32,
    life_max: f32,
    init_size: f32,
    init_spectral: [f32; 16],
    // resolved Update parameters
    gravity: Vec3,
    drag: f32,
    turbulence_strength: f32,
    turbulence_freq: f32,
    spectral_fade_end: f32,
    size_over_life_end: f32,
}

impl VfxGraphInstance {
    /// Build an instance from a **validated** graph. Panics if `graph` fails
    /// [`VfxGraph::validate`] — call that first (or use [`Self::try_new`]).
    pub fn new(graph: VfxGraph, origin: Vec3) -> Self {
        Self::try_new(graph, origin).expect("VfxGraphInstance::new requires a valid graph")
    }

    /// Build an instance, validating first.
    pub fn try_new(graph: VfxGraph, origin: Vec3) -> Result<Self, VfxGraphError> {
        graph.validate()?;

        // Resolve node parameters into flat fields (defaults match the
        // conventions used by the legacy SplatEmitter/VfxEmitter).
        let mut init_dir = Vec3::Y;
        let mut init_speed = 1.0;
        let mut init_spread = 0.0;
        let mut life_min = 1.0;
        let mut life_max = 1.0;
        let mut init_size = 0.1;
        let mut init_spectral = [0.0f32; 16];
        let mut gravity = Vec3::ZERO;
        let mut drag = 0.0;
        let mut turbulence_strength = 0.0;
        let mut turbulence_freq = 1.0;
        let mut spectral_fade_end = 1.0;
        let mut size_over_life_end = 1.0;
        let mut max_particles = 4096usize;

        for node in &graph.nodes {
            match node {
                VfxNode::InitVelocity { direction, speed, spread } => {
                    init_dir = Vec3::from(*direction).normalize_or_zero();
                    init_speed = *speed;
                    init_spread = *spread;
                }
                VfxNode::InitLifetime { min, max } => {
                    life_min = *min;
                    life_max = *max;
                }
                VfxNode::InitSize { size } => init_size = *size,
                VfxNode::InitSpectral { emission } => init_spectral = emission.to_spd(),
                VfxNode::Gravity { accel } => gravity = Vec3::from(*accel),
                VfxNode::Drag { coefficient } => drag = *coefficient,
                VfxNode::Turbulence { strength, frequency } => {
                    turbulence_strength = *strength;
                    turbulence_freq = *frequency;
                }
                VfxNode::SpectralFadeOverLife { end_scale } => spectral_fade_end = *end_scale,
                VfxNode::SizeOverLife { end_scale } => size_over_life_end = *end_scale,
                _ => {}
            }
        }

        // Cap pool to keep frame cost bounded: rate * max_lifetime headroom.
        if let VfxNode::Spawn { rate, .. } = graph.spawn() {
            let est = (rate * life_max.max(0.001) * 1.5).ceil() as usize;
            max_particles = est.clamp(1, 1 << 16);
        }

        let seed = graph.seed;
        Ok(Self {
            graph,
            origin,
            particles: ParticleSoa::default(),
            accumulator: 0.0,
            burst_fired: false,
            // LCG seeding identical in spirit to vfx.rs (avoid a zero state).
            rng: seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407),
            time: 0.0,
            max_particles,
            init_dir,
            init_speed,
            init_spread,
            life_min,
            life_max,
            init_size,
            init_spectral,
            gravity,
            drag,
            turbulence_strength,
            turbulence_freq,
            spectral_fade_end,
            size_over_life_end,
        })
    }

    /// Number of live particles.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Elapsed simulated time in seconds.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Advance the simulation by `dt` seconds: integrate live particles, kill
    /// expired ones, then spawn new ones. Deterministic given the seed.
    pub fn step(&mut self, dt: f32) {
        self.time += dt;

        // --- Update existing particles (swap-remove dead) ---
        let mut i = 0;
        while i < self.particles.len() {
            self.particles.age[i] += dt;
            if self.particles.age[i] >= self.particles.lifetime[i] {
                self.particles.swap_remove(i);
                continue; // swapped element now at i, re-examine
            }

            // Gravity + turbulence acceleration.
            let mut vx = self.particles.vel_x[i] + self.gravity.x * dt;
            let mut vy = self.particles.vel_y[i] + self.gravity.y * dt;
            let mut vz = self.particles.vel_z[i] + self.gravity.z * dt;

            if self.turbulence_strength != 0.0 {
                let f = self.turbulence_freq;
                let (tx, ty, tz) = turbulence_accel(
                    self.particles.pos_x[i] * f,
                    self.particles.pos_y[i] * f,
                    self.particles.pos_z[i] * f,
                );
                vx += tx * self.turbulence_strength * dt;
                vy += ty * self.turbulence_strength * dt;
                vz += tz * self.turbulence_strength * dt;
            }

            // Drag.
            if self.drag != 0.0 {
                let damp = (1.0 - self.drag * dt).clamp(0.0, 1.0);
                vx *= damp;
                vy *= damp;
                vz *= damp;
            }

            self.particles.vel_x[i] = vx;
            self.particles.vel_y[i] = vy;
            self.particles.vel_z[i] = vz;
            self.particles.pos_x[i] += vx * dt;
            self.particles.pos_y[i] += vy * dt;
            self.particles.pos_z[i] += vz * dt;
            i += 1;
        }

        // --- Burst spawn (once) ---
        if !self.burst_fired {
            if let VfxNode::SpawnBurst { count, shape } = self.graph.spawn() {
                let (count, shape) = (*count, shape.clone());
                for _ in 0..count {
                    if self.particles.len() >= self.max_particles {
                        break;
                    }
                    self.spawn_one(&shape);
                }
            }
            self.burst_fired = true;
        }

        // --- Rate spawn ---
        if let VfxNode::Spawn { rate, shape } = self.graph.spawn() {
            let (rate, shape) = (*rate, shape.clone());
            self.accumulator += rate * dt;
            while self.accumulator >= 1.0 && self.particles.len() < self.max_particles {
                self.accumulator -= 1.0;
                self.spawn_one(&shape);
            }
            // Clamp runaway accumulation (matches SplatEmitter guard).
            if self.accumulator > rate.max(1.0) {
                self.accumulator = 0.0;
            }
        }
    }

    /// Yield this frame's renderable splats (one volume splat per live
    /// particle). Spectral-fade and size-over-life are applied here from the
    /// immutable birth state, so the function is pure w.r.t. particle state.
    pub fn emit_splats(&self) -> Vec<GaussianSplat> {
        let base_opacity = self.graph.output_opacity() as f32;
        let mut out = Vec::with_capacity(self.particles.len());
        for i in 0..self.particles.len() {
            let life_t = (self.particles.age[i] / self.particles.lifetime[i]).clamp(0.0, 1.0);
            let remaining = 1.0 - life_t;

            // Spectral fade: scale every band toward `spectral_fade_end`.
            let fade = 1.0 + (self.spectral_fade_end - 1.0) * life_t;
            let spd = self.particles.spectral[i];
            let spectral: [u16; 16] = std::array::from_fn(|b| {
                f16::from_f32((spd[b] * fade).clamp(0.0, 1.0)).to_bits()
            });

            // Size over life.
            let size_scale = 1.0 + (self.size_over_life_end - 1.0) * life_t;
            let size = self.particles.size[i] * size_scale;

            let opacity = (base_opacity * remaining).clamp(0.0, 255.0) as u8;

            out.push(GaussianSplat::volume(
                [self.particles.pos_x[i], self.particles.pos_y[i], self.particles.pos_z[i]],
                [size, size, size],
                Quat::IDENTITY,
                opacity,
                spectral,
            ));
        }
        out
    }

    /// Mean velocity over all live particles (diagnostic / test helper).
    pub fn mean_velocity(&self) -> Vec3 {
        let n = self.particles.len();
        if n == 0 {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for i in 0..n {
            sum += Vec3::new(
                self.particles.vel_x[i],
                self.particles.vel_y[i],
                self.particles.vel_z[i],
            );
        }
        sum / n as f32
    }

    fn spawn_one(&mut self, shape: &SpawnShape) {
        let offset = self.sample_shape(shape);
        let dir = self.sample_velocity_dir();
        let vel = dir * self.init_speed;
        let life_t = self.rand_unit();
        let lifetime = self.life_min + (self.life_max - self.life_min) * life_t;
        self.particles.push(
            self.origin + offset,
            vel,
            lifetime.max(1e-3),
            self.init_size,
            self.init_spectral,
        );
    }

    fn sample_velocity_dir(&mut self) -> Vec3 {
        if self.init_spread <= 0.0 {
            return self.init_dir;
        }
        let s = self.init_spread;
        let rx = (self.rand_unit() - 0.5) * 2.0 * s;
        let ry = (self.rand_unit() - 0.5) * 2.0 * s;
        let rz = (self.rand_unit() - 0.5) * 2.0 * s;
        (self.init_dir + Vec3::new(rx, ry, rz)).normalize_or_zero()
    }

    fn sample_shape(&mut self, shape: &SpawnShape) -> Vec3 {
        match shape {
            SpawnShape::Point => Vec3::ZERO,
            SpawnShape::Sphere { radius } => {
                let x = self.rand_unit() - 0.5;
                let y = self.rand_unit() - 0.5;
                let z = self.rand_unit() - 0.5;
                let dir = Vec3::new(x, y, z).normalize_or_zero();
                dir * self.rand_unit() * *radius
            }
            SpawnShape::Cone { angle_deg, radius } => {
                let a = self.rand_unit() * std::f32::consts::TAU;
                let r = self.rand_unit() * *radius;
                let spread = angle_deg.to_radians().sin() * r;
                Vec3::new(a.cos() * spread, 0.0, a.sin() * spread)
            }
        }
    }

    /// Deterministic LCG in [0, 1) — same recurrence as `vfx.rs::next_random`.
    fn rand_unit(&mut self) -> f32 {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng >> 33) as f32 / (1u64 << 31) as f32
    }
}

// ===========================================================================
// Hash-noise turbulence (deterministic, position-driven)
// ===========================================================================

/// Integer hash → f32 in [-1, 1]. Wang-style mix; deterministic.
fn hash_to_signed(mut x: u32) -> f32 {
    x = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    let h = (word >> 22) ^ word;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Position-hashed turbulence acceleration, each component in roughly [-1, 1].
fn turbulence_accel(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let xi = (x * 16.0) as i32 as u32;
    let yi = (y * 16.0) as i32 as u32;
    let zi = (z * 16.0) as i32 as u32;
    let base = xi
        .wrapping_mul(73856093)
        ^ yi.wrapping_mul(19349663)
        ^ zi.wrapping_mul(83492791);
    (
        hash_to_signed(base),
        hash_to_signed(base ^ 0x9E3779B9),
        hash_to_signed(base ^ 0x85EBCA6B),
    )
}

// ===========================================================================
// Blackbody spectral emission
// ===========================================================================

/// Planckian blackbody spectral radiance over the 16 USGS bands, normalised so
/// the peak band == 1.0. Uses Planck's law with the band centre wavelengths
/// from [`BAND_WAVELENGTHS`].
pub fn blackbody_spd(kelvin: f32) -> [f32; 16] {
    // Planck's law constants (SI), wavelength form:
    //   B(λ,T) = (2hc²/λ⁵) / (exp(hc/(λ k T)) - 1)
    const HC_OVER_K: f64 = 0.0143877688; // h*c/k_B  (m·K)
    let t = kelvin.max(1.0) as f64;
    let mut spd = [0.0f64; 16];
    let mut max = 0.0f64;
    for (i, wl_nm) in BAND_WAVELENGTHS.iter().enumerate() {
        let lambda = *wl_nm as f64 * 1e-9; // nm → m
        let l5 = lambda.powi(5);
        let expo = (HC_OVER_K / (lambda * t)).exp() - 1.0;
        let radiance = 1.0 / (l5 * expo);
        spd[i] = radiance;
        if radiance > max {
            max = radiance;
        }
    }
    let inv = if max > 0.0 { 1.0 / max } else { 0.0 };
    std::array::from_fn(|i| (spd[i] * inv) as f32)
}

// ===========================================================================
// Library effects (DATA, not code paths)
// ===========================================================================

/// Construct a library effect by name. Returns `None` for unknown names.
/// Known: `"fire"`, `"fountain"`, `"smoke"`.
pub fn effect_by_name(name: &str, seed: u64) -> Option<VfxGraph> {
    match name {
        "fire" => Some(graph_fire(seed)),
        "fountain" => Some(graph_fountain(seed)),
        "smoke" => Some(graph_smoke(seed)),
        _ => None,
    }
}

/// Fire — blackbody 1800K, cone upward, turbulence, fading.
pub fn graph_fire(seed: u64) -> VfxGraph {
    VfxGraph {
        name: "fire".into(),
        nodes: vec![
            VfxNode::Spawn { rate: 60.0, shape: SpawnShape::Cone { angle_deg: 15.0, radius: 0.2 } },
            VfxNode::InitVelocity { direction: [0.0, 1.0, 0.0], speed: 2.5, spread: 0.25 },
            VfxNode::InitLifetime { min: 0.8, max: 1.6 },
            VfxNode::InitSize { size: 0.12 },
            VfxNode::InitSpectral { emission: SpectralEmission::Blackbody { kelvin: 1800.0 } },
            VfxNode::Gravity { accel: [0.0, 0.6, 0.0] }, // hot air buoyancy (rises)
            VfxNode::Turbulence { strength: 1.5, frequency: 2.0 },
            VfxNode::SpectralFadeOverLife { end_scale: 0.05 },
            VfxNode::SizeOverLife { end_scale: 0.4 },
            VfxNode::Output { base_opacity: 200 },
        ],
        // Linear stack: 0->1->2->...->9.
        edges: (0..9).map(|i| VfxEdge { from: i, to: i + 1 }).collect(),
        seed,
    }
}

/// Fountain — sphere burst-ish spray, gravity pulls it back down, blue-white SPD.
pub fn graph_fountain(seed: u64) -> VfxGraph {
    let blue_white = [
        0.85, 0.90, 0.95, 1.00, 0.95, 0.85, 0.75, 0.65, 0.55, 0.50, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34,
    ];
    VfxGraph {
        name: "fountain".into(),
        nodes: vec![
            VfxNode::Spawn { rate: 80.0, shape: SpawnShape::Sphere { radius: 0.1 } },
            VfxNode::InitVelocity { direction: [0.0, 1.0, 0.0], speed: 6.0, spread: 0.3 },
            VfxNode::InitLifetime { min: 1.0, max: 1.8 },
            VfxNode::InitSize { size: 0.05 },
            VfxNode::InitSpectral { emission: SpectralEmission::Spd(blue_white) },
            VfxNode::Gravity { accel: [0.0, -9.81, 0.0] },
            VfxNode::Drag { coefficient: 0.1 },
            VfxNode::Output { base_opacity: 220 },
        ],
        edges: (0..7).map(|i| VfxEdge { from: i, to: i + 1 }).collect(),
        seed,
    }
}

/// Smoke — slow rise, drag, grey fade.
pub fn graph_smoke(seed: u64) -> VfxGraph {
    VfxGraph {
        name: "smoke".into(),
        nodes: vec![
            VfxNode::Spawn { rate: 12.0, shape: SpawnShape::Sphere { radius: 0.2 } },
            VfxNode::InitVelocity { direction: [0.0, 1.0, 0.0], speed: 1.0, spread: 0.4 },
            VfxNode::InitLifetime { min: 2.0, max: 3.5 },
            VfxNode::InitSize { size: 0.25 },
            VfxNode::InitSpectral { emission: SpectralEmission::Spd([0.3; 16]) },
            VfxNode::Gravity { accel: [0.0, 0.4, 0.0] },
            VfxNode::Drag { coefficient: 0.5 },
            VfxNode::SpectralFadeOverLife { end_scale: 0.1 },
            VfxNode::SizeOverLife { end_scale: 3.0 },
            VfxNode::Output { base_opacity: 120 },
        ],
        edges: (0..9).map(|i| VfxEdge { from: i, to: i + 1 }).collect(),
        seed,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn band_sum(splat: &GaussianSplat) -> f32 {
        (0..16).map(|b| splat.spectral_f32(b)).sum()
    }

    fn spectral_argmax(splat: &GaussianSplat) -> usize {
        let mut best = 0;
        let mut bestv = splat.spectral_f32(0);
        for b in 1..16 {
            let v = splat.spectral_f32(b);
            if v > bestv {
                bestv = v;
                best = b;
            }
        }
        best
    }

    #[test]
    fn blackbody_1800k_peaks_in_red_ir() {
        let spd = blackbody_spd(1800.0);
        let mut argmax = 0;
        for i in 1..16 {
            if spd[i] > spd[argmax] {
                argmax = i;
            }
        }
        // Bands 13..15 are 705/730/755 nm — deep red / near-IR.
        println!(
            "blackbody 1800K argmax band={} (wl={}nm), spd={:?}",
            argmax, BAND_WAVELENGTHS[argmax], spd
        );
        assert!(
            argmax >= 13,
            "1800K should peak in red/IR (band>=13), got band {} ({}nm)",
            argmax,
            BAND_WAVELENGTHS[argmax]
        );
    }

    #[test]
    fn fire_effect_velocity_and_spectrum_at_1s() {
        let mut inst = VfxGraphInstance::new(graph_fire(7), Vec3::ZERO);
        // Step to t≈1s in 60 frames.
        for _ in 0..60 {
            inst.step(1.0 / 60.0);
        }
        let splats = inst.emit_splats();
        assert!(!splats.is_empty(), "fire should have live particles at 1s");

        // Mean velocity has +Y dominance.
        let mv = inst.mean_velocity();
        println!(
            "fire @1s: count={}, mean_vel=({:.3},{:.3},{:.3})",
            splats.len(),
            mv.x,
            mv.y,
            mv.z
        );
        assert!(mv.y > mv.x.abs() && mv.y > mv.z.abs(), "fire mean velocity must be +Y dominant: {mv:?}");
        assert!(mv.y > 0.0, "fire must rise");

        // Spectral argmax in red/IR bands for the brightest (youngest) splat.
        // Track the max band across splats (fade can dim old ones uniformly).
        let young = splats
            .iter()
            .max_by(|a, b| band_sum(a).partial_cmp(&band_sum(b)).unwrap())
            .unwrap();
        let am = spectral_argmax(young);
        println!("fire @1s: brightest splat spectral argmax band={am} ({}nm)", BAND_WAVELENGTHS[am]);
        assert!(am >= 13, "fire blackbody argmax must be red/IR band>=13, got {am}");

        // Particle count within spawn-rate*lifetime band.
        // rate=60/s, lifetime in [0.8,1.6] (mean ~1.2) → steady-state ~ 60*1.2=72,
        // bounded by [60*0.8, 60*1.6] = [48, 96] with spawn jitter headroom.
        let n = splats.len();
        assert!(
            (40..=110).contains(&n),
            "fire steady-state count {n} outside expected [40,110] (rate*lifetime)"
        );
    }

    #[test]
    fn spectral_fade_over_life_decreases_energy() {
        // Single-particle burst so we can track ONE particle across its life.
        let graph = VfxGraph {
            name: "fade_probe".into(),
            nodes: vec![
                VfxNode::SpawnBurst { count: 1, shape: SpawnShape::Point },
                VfxNode::InitVelocity { direction: [0.0, 1.0, 0.0], speed: 0.0, spread: 0.0 },
                VfxNode::InitLifetime { min: 1.0, max: 1.0 },
                VfxNode::InitSize { size: 0.1 },
                VfxNode::InitSpectral { emission: SpectralEmission::Spd([0.8; 16]) },
                VfxNode::SpectralFadeOverLife { end_scale: 0.0 },
                VfxNode::Output { base_opacity: 255 },
            ],
            edges: (0..6).map(|i| VfxEdge { from: i, to: i + 1 }).collect(),
            seed: 1,
        };
        let mut inst = VfxGraphInstance::new(graph, Vec3::ZERO);
        inst.step(0.0); // fire the burst

        inst.step(0.1); // age ≈ 0.1 (10% life)
        let early = band_sum(&inst.emit_splats()[0]);

        for _ in 0..8 {
            inst.step(0.1); // age ≈ 0.9 (90% life)
        }
        let late = band_sum(&inst.emit_splats()[0]);

        println!("spectral fade: energy@10%life={early:.4}  @90%life={late:.4}");
        assert!(
            late < early,
            "spectral energy must strictly decrease over life: {early} -> {late}"
        );
    }

    #[test]
    fn gravity_decreases_fountain_mean_vy() {
        let mut inst = VfxGraphInstance::new(graph_fountain(3), Vec3::ZERO);
        // Prime a population.
        for _ in 0..10 {
            inst.step(1.0 / 60.0);
        }
        let vy_start = inst.mean_velocity().y;
        let mut prev = vy_start;
        let mut monotonic = true;
        for _ in 0..60 {
            inst.step(1.0 / 60.0);
            let vy = inst.mean_velocity().y;
            if vy > prev + 1e-3 {
                monotonic = false;
            }
            prev = vy;
        }
        let vy_end = inst.mean_velocity().y;
        println!("fountain mean vy: start={vy_start:.4} end={vy_end:.4} monotonic_down={monotonic}");
        assert!(
            vy_end < vy_start,
            "gravity must pull fountain mean vy down: {vy_start} -> {vy_end}"
        );
    }

    #[test]
    fn determinism_same_seed_bit_identical() {
        let mut a = VfxGraphInstance::new(graph_fire(12345), Vec3::new(1.0, 2.0, 3.0));
        let mut b = VfxGraphInstance::new(graph_fire(12345), Vec3::new(1.0, 2.0, 3.0));
        let dts = [1.0 / 60.0, 1.0 / 90.0, 1.0 / 50.0];
        for k in 0..100 {
            let dt = dts[k % dts.len()];
            a.step(dt);
            b.step(dt);
        }
        let sa = a.emit_splats();
        let sb = b.emit_splats();
        assert_eq!(sa.len(), sb.len(), "same seed must yield same particle count");
        let bytes_a: &[u8] = bytemuck::cast_slice(&sa);
        let bytes_b: &[u8] = bytemuck::cast_slice(&sb);
        println!(
            "determinism: {} splats, {} bytes each, identical={}",
            sa.len(),
            bytes_a.len(),
            bytes_a == bytes_b
        );
        assert_eq!(bytes_a, bytes_b, "same seed + same dt sequence must be bit-identical");
    }

    #[test]
    fn different_seed_differs() {
        let mut a = VfxGraphInstance::new(graph_fire(1), Vec3::ZERO);
        let mut b = VfxGraphInstance::new(graph_fire(2), Vec3::ZERO);
        for _ in 0..30 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        let sa = a.emit_splats();
        let sb = b.emit_splats();
        let bytes_a: &[u8] = bytemuck::cast_slice(&sa);
        let bytes_b: &[u8] = bytemuck::cast_slice(&sb);
        assert_ne!(bytes_a, bytes_b, "different seeds should diverge");
    }

    #[test]
    fn cycle_is_rejected() {
        let graph = VfxGraph {
            name: "cyclic".into(),
            nodes: vec![
                VfxNode::Spawn { rate: 1.0, shape: SpawnShape::Point },
                VfxNode::Gravity { accel: [0.0, -1.0, 0.0] },
                VfxNode::Output { base_opacity: 255 },
            ],
            // 1 -> 2 -> 1 forms a cycle among Update/Output-ish wiring.
            edges: vec![
                VfxEdge { from: 0, to: 1 },
                VfxEdge { from: 1, to: 2 },
                VfxEdge { from: 2, to: 1 }, // back-edge → cycle (and stage mismatch)
            ],
            seed: 0,
        };
        let err = graph.validate().unwrap_err();
        println!("cycle graph rejected with: {err}");
        // Stage check runs before cycle check; a back-edge Output->Update is a
        // StageMismatch. Either error is a valid rejection of this bad graph.
        assert!(
            matches!(err, VfxGraphError::Cycle | VfxGraphError::StageMismatch { .. }),
            "expected Cycle or StageMismatch, got {err:?}"
        );
    }

    #[test]
    fn pure_cycle_detected() {
        // A real cycle among same-stage Update nodes (no stage mismatch).
        let graph = VfxGraph {
            name: "pure_cycle".into(),
            nodes: vec![
                VfxNode::Spawn { rate: 1.0, shape: SpawnShape::Point },
                VfxNode::Drag { coefficient: 0.1 },
                VfxNode::Turbulence { strength: 1.0, frequency: 1.0 },
                VfxNode::Output { base_opacity: 255 },
            ],
            edges: vec![
                VfxEdge { from: 1, to: 2 },
                VfxEdge { from: 2, to: 1 }, // Update<->Update cycle, ranks equal
            ],
            seed: 0,
        };
        assert_eq!(graph.validate(), Err(VfxGraphError::Cycle));
    }

    #[test]
    fn type_mismatch_is_rejected() {
        // Wire Output (rank 3) back into Init (rank 1): stage mismatch.
        let graph = VfxGraph {
            name: "mismatch".into(),
            nodes: vec![
                VfxNode::Spawn { rate: 1.0, shape: SpawnShape::Point },
                VfxNode::InitSize { size: 0.1 },
                VfxNode::Output { base_opacity: 255 },
            ],
            edges: vec![VfxEdge { from: 2, to: 1 }],
            seed: 0,
        };
        let err = graph.validate().unwrap_err();
        println!("type mismatch rejected with: {err}");
        assert_eq!(
            err,
            VfxGraphError::StageMismatch { from: NodeStage::Output, to: NodeStage::Init }
        );
    }

    #[test]
    fn missing_spawn_and_output_rejected() {
        let no_spawn = VfxGraph {
            name: "no_spawn".into(),
            nodes: vec![VfxNode::Output { base_opacity: 255 }],
            edges: vec![],
            seed: 0,
        };
        assert_eq!(no_spawn.validate(), Err(VfxGraphError::NoSpawn));

        let no_output = VfxGraph {
            name: "no_output".into(),
            nodes: vec![VfxNode::Spawn { rate: 1.0, shape: SpawnShape::Point }],
            edges: vec![],
            seed: 0,
        };
        assert_eq!(no_output.validate(), Err(VfxGraphError::NoOutput));
    }

    #[test]
    fn edge_out_of_bounds_rejected() {
        let graph = VfxGraph {
            name: "oob".into(),
            nodes: vec![
                VfxNode::Spawn { rate: 1.0, shape: SpawnShape::Point },
                VfxNode::Output { base_opacity: 255 },
            ],
            edges: vec![VfxEdge { from: 0, to: 99 }],
            seed: 0,
        };
        assert_eq!(
            graph.validate(),
            Err(VfxGraphError::EdgeOutOfBounds { edge: VfxEdge { from: 0, to: 99 } })
        );
    }

    #[test]
    fn library_effects_validate() {
        for name in ["fire", "fountain", "smoke"] {
            let g = effect_by_name(name, 1).expect("known effect");
            g.validate().unwrap_or_else(|e| panic!("{name} invalid: {e}"));
        }
        assert!(effect_by_name("nope", 1).is_none());
    }

    #[test]
    fn render_proof_fire_warm_pixels() {
        use crate::gpu::software_rasteriser::SoftwareRasteriser;
        use crate::spectral::RenderCamera;
        use glam::Mat4;
        use vox_core::spectral::Illuminant;

        // Build a dense, settled fire population.
        let mut inst = VfxGraphInstance::new(graph_fire(99), Vec3::new(0.0, 0.0, 0.0));
        for _ in 0..120 {
            inst.step(1.0 / 60.0);
        }
        let splats = inst.emit_splats();
        assert!(splats.len() > 20, "need a populated fire, got {}", splats.len());

        let width = 128u32;
        let height = 128u32;
        let eye = Vec3::new(0.0, 1.0, 4.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let camera = RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                width as f32 / height as f32,
                0.1,
                100.0,
            ),
        };
        let illuminant = Illuminant::d65();
        let mut raster = SoftwareRasteriser::new(width, height);
        let fb = raster.render(&splats, &camera, &illuminant, None);

        // Count non-black pixels and warm (R>B) lit pixels.
        let mut non_black = 0usize;
        let mut warm = 0usize;
        let mut sum_r = 0u64;
        let mut sum_b = 0u64;
        for px in &fb.pixels {
            let (r, g, b) = (px[0] as u32, px[1] as u32, px[2] as u32);
            if r + g + b > 0 {
                non_black += 1;
                sum_r += r as u64;
                sum_b += b as u64;
                if r > b {
                    warm += 1;
                }
            }
        }
        println!(
            "render-proof fire: {} splats -> {non_black} non-black px, {warm} warm px, sum_r={sum_r} sum_b={sum_b}",
            splats.len()
        );
        assert!(non_black > 50, "expected >50 lit pixels, got {non_black}");
        assert!(
            warm > non_black / 2,
            "warm pixels ({warm}) must dominate lit region ({non_black})"
        );
        assert!(sum_r > sum_b, "total red ({sum_r}) must exceed total blue ({sum_b})");
    }
}
