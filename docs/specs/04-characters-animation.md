# Domain 4 — Characters & Animation

**Status:** Spec v1.0 — 2026-03-29
**Crate scope:** `vox_core`, `vox_render`, `vox_physics`, `vox_audio`, `vox_app`
**Dependencies:** GaussianSplat pipeline, BlendSkinningCompute (4-pose), AnimationDriver, GltfSkeleton, Rapier, wgpu 24, rayon, glam

---

## Goals

Ochroma must support production-quality character animation entirely within the spectral Gaussian Splat paradigm. No polygon meshes are used for character surfaces; all deformation and rendering is splat-based. The character pipeline must:

- Support full multi-joint skinning with 4 blend weights per splat at 60+ FPS for 10 simultaneous characters at 4K.
- Support spectral morph targets: deformation that modulates both geometry and spectral emission, enabling facial expressions and body deformation impossible in RGB pipelines.
- Match animation to motion capture databases without explicit blend trees, using motion matching.
- Place feet accurately on uneven SDF terrain using IK.
- Simulate cloth and hair as splat fields driven by position-based dynamics.
- Maintain engine generality: no game-specific character types in engine crates.

---

## 4.1 Skeletal Mesh Pipeline (Production-Grade)

### Current State and Gap

The engine has `AnimationDriver`, `GltfSkeleton`, and `BlendSkinningCompute` (4 input poses, quaternion slerp on GPU). The gap: each splat is assigned to exactly one joint. Real character meshes require up to 4 joint influences per vertex/splat to prevent hard crease artefacts at joints.

### SplatSkinData

```rust
// vox_core::skinning
pub struct SplatSkinData {
    pub joint_indices: [u8; 4],    // indices into GltfSkeleton::joints
    pub joint_weights: [f32; 4],   // must sum to 1.0; trailing zeros for < 4 influences
}
```

`SplatSkinData` is a parallel array to `GaussianSplat`, not embedded inside `GaussianSplat`. This keeps the hot splat data (used by the EWA tile renderer) compact and cache-efficient. `SplatSkinData` is only bound when the skinning compute pass runs.

### GPU Skinning Buffer Layout

```wgsl
// In blend_skinning.wgsl (extension of existing shader)
struct SplatSkinData {
    joint_indices: vec4<u32>,
    joint_weights: vec4<f32>,
}

@group(0) @binding(0) var<storage, read>       base_splats:    array<GaussianSplat>;
@group(0) @binding(1) var<storage, read>       skin_data:      array<SplatSkinData>;
@group(0) @binding(2) var<storage, read>       joint_matrices: array<mat4x4<f32>>;  // world-space skinning matrices
@group(0) @binding(3) var<storage, read_write> output_splats:  array<GaussianSplat>;
```

Each compute invocation handles one splat. The skinning matrix for a splat is:

```wgsl
var skin_matrix = mat4x4<f32>(0.0);
for (var i = 0u; i < 4u; i++) {
    let idx = skin_data[id].joint_indices[i];
    let w   = skin_data[id].joint_weights[i];
    skin_matrix += joint_matrices[idx] * w;
}
let skinned_pos = skin_matrix * vec4<f32>(base_splats[id].position, 1.0);
```

Rotation (stored as `[i16; 4]` quaternion) is transformed by extracting the rotation from `skin_matrix` via polar decomposition (`mat3_to_quat` utility function in the shader). Scale is transformed by the scale component of `skin_matrix`.

`joint_matrices[k]` = `current_joint_world_transform[k] * inverse_bind_matrix[k]`. Inverse bind matrices are uploaded once at character load time into a persistent `wgpu::Buffer`.

### Skeleton Retargeter

```rust
pub struct SkeletonRetargeter {
    pub source_skeleton: Arc<GltfSkeleton>,
    pub target_skeleton: Arc<GltfSkeleton>,
    pub joint_map: HashMap<JointName, JointName>,
}

impl SkeletonRetargeter {
    pub fn retarget_pose(&self, source_pose: &SkeletonPose) -> SkeletonPose { ... }
}
```

Retargeting maps source joint local rotations to target joint local rotations by name. For joints not in `joint_map`, the target uses its bind pose rotation. Proportional correction: if the source and target T-pose upper arm lengths differ, scale the resulting arm reach proportionally so that hand targets remain correct. This is implemented by computing a per-joint chain length ratio and scaling translation components of terminal joints.

`JointName` is a `SmallString<[u8; 32]>` to avoid heap allocation for typical joint names.

### Animation Compression: B-Spline Clips

```rust
pub struct AnimationClip {
    pub name: String,
    pub duration_secs: f32,
    pub joint_curves: Vec<JointCurve>,    // one per joint
    pub root_motion: Option<RootMotionCurve>,
}

pub struct JointCurve {
    pub joint_index: u16,
    pub rotation_spline: BSpline<Quat>,
    pub translation_spline: BSpline<Vec3>,
    pub scale_spline: BSpline<Vec3>,
}

pub struct BSpline<T> {
    pub control_points: Vec<T>,
    pub knot_vector: Vec<f32>,
    pub degree: u8,                // 3 for cubic
}
```

Clip evaluation: `JointCurve::sample(t: f32) -> JointTransform` evaluates each B-spline at `t` using De Boor's algorithm. Memory: a 60s clip at 30 joints with 10 control points per joint per channel = 30 × 3 curves × 10 control points = 900 `Vec3`/`Quat` values ≈ 30 KB. An equivalent 30fps keyframe clip = 30 joints × 1800 keyframes × 3 channels = 4.8 MB. B-spline storage is ~160× smaller for smooth motions.

### Root Motion Extraction

```rust
impl AnimationClip {
    pub fn extract_root_motion(&mut self) -> RootMotionCurve {
        // removes root joint XZ translation and Y rotation from joint_curves[root_joint_index]
        // returns them as RootMotionCurve for use by CharacterController
    }
}

pub struct RootMotionCurve {
    pub translation_xz: BSpline<Vec2>,
    pub rotation_y: BSpline<f32>,
}
```

The `CharacterController` (Rapier-backed) advances the character's world transform each frame by the delta of `RootMotionCurve::sample(t + dt) - sample(t)`, rotated by the character's current facing direction. This ensures locomotion animations drive actual world movement without sliding.

---

## 4.2 Morph Targets (Blend Shapes)

### Types

```rust
pub struct SplatDelta {
    pub splat_index: u32,
    pub d_position: [f32; 3],
    pub d_scale:    [f32; 3],
    pub d_spectral: [u16; 8],    // f16 bits; signed delta, stored as offset from 32768 = 0.0
}

pub struct MorphTarget {
    pub name: String,
    pub deltas: Vec<SplatDelta>,
}

pub struct MorphTargetSet {
    pub targets: Vec<MorphTarget>,
    pub base_splats: Arc<Vec<GaussianSplat>>,
}
```

`d_spectral` uses f16 for storage efficiency. At runtime, each delta is unpacked to `f32` in the compute shader. The signed delta representation: `d_spectral[i]` as f16 can be negative (absorption) or positive (emission increase). For example, a `smile_l` morph target on cheek splats has positive `d_spectral[5..7]` (red band) to represent increased subsurface scattering in the compressed cheek tissue.

### MorphComputePass

WGSL compute shader `morph_targets.wgsl`:

```wgsl
@group(0) @binding(0) var<storage, read>       base_splats:   array<GaussianSplat>;
@group(0) @binding(1) var<storage, read>       delta_buffer:  array<PackedSplatDelta>;
@group(0) @binding(2) var<uniform>             weights:       array<f32, 16>;
@group(0) @binding(3) var<storage, read>       active_target_offsets: array<TargetOffsets, 16>;
@group(0) @binding(4) var<storage, read_write> output_splats: array<GaussianSplat>;
```

Each compute dispatch iterates over active morph targets (up to 16) for each splat. Inactive targets (weight = 0.0) are skipped via early exit. `PackedSplatDelta` stores `splat_index` + compressed delta fields; the buffer is the concatenation of all active targets' delta arrays.

`MorphComputePass` runs before `BlendSkinningCompute` in the frame pipeline: morph → skin → EWA render.

### Authoring

In the editor, `MorphTargetAuthorer` compares a base-pose character's splat array against a deformed-pose splat array (authored by the artist by brushing splat positions/scales/spectrals in the editor). The diff is computed as `SplatDelta` for each splat that changed beyond a threshold (`d_position.length() > 1e-4` or any `|d_spectral[i]| > 0.001`). This workflow reuses the terrain sculpt brush tool adapted for character splat editing.

---

## 4.3 Facial Animation System

### FacialRig

```rust
pub struct FacialRig {
    pub action_units: Vec<ActionUnit>,   // 44 FACS action units
    pub au_to_morph: Vec<AuMorphMapping>,
}

pub struct ActionUnit {
    pub id: u8,
    pub name: String,     // e.g. "AU06_cheek_raiser"
    pub weight: f32,      // [0.0, 1.0]
}

pub struct AuMorphMapping {
    pub au_id: u8,
    pub morph_name: String,
    pub influence: f32,   // how much this AU drives this morph
}
```

`FacialRig::compute_morph_weights() -> Vec<(MorphTargetIndex, f32)>` collapses the AU weights into final morph weights by summing `au.weight * mapping.influence` across all mappings for each morph target, then clamping to `[0.0, 1.0]`.

### Audio-Driven Lip Sync

```rust
pub struct AudioLipSync {
    pub phoneme_classifier: PhonemeClassifier,
    pub target_rig: FacialRig,
    pub viseme_table: VisemeTable,
}

pub struct PhonemeClassifier {
    weights: Box<[f32]>,       // 50 KB embedded model parameters (MFCC → phoneme)
    frame_buffer: VecDeque<[f32; 13]>,  // 13 MFCC coefficients, 25ms frames
}

pub enum Phoneme {
    Silence, P, B, M, F, V, Th, D, T, N, S, Z, Sh, Ch, Jh, G, K, Ng,
    Ah, Ae, Ey, Ih, Iy, Oh, Ow, Uh, Uw, Er, Aa, Aw, Oy, Ay,
}

pub struct VisemeTable {
    phoneme_to_au_weights: HashMap<Phoneme, Vec<(u8, f32)>>,
}
```

`PhonemeClassifier::classify_frame(audio_pcm: &[f32]) -> Phoneme`:
1. Compute 13 MFCC coefficients from the 25ms audio frame using a Hann window + FFT (via `rustfft` crate) + mel filterbank + DCT.
2. Concatenate 3 frames (75ms context window) into a 39-element feature vector.
3. Multiply by the embedded weight matrix (single-layer linear classifier, `39 → 32` hidden → phoneme logits), apply softmax, return argmax phoneme.

The 50 KB model is embedded at compile time via `include_bytes!`. A 3-layer classifier is sufficient for real-time phoneme recognition when trained on a balanced phoneme corpus; this is not speaker-dependent.

`VisemeTable` maps each phoneme to a set of FACS action unit weights. Example: phoneme `Uw` (as in "food") → `{AU20: 0.8, AU25: 0.4}` (lip stretcher + lips parted).

Lip sync runs 1 frame ahead of audio playback (pre-classified at audio frame submission) to account for animation smoothing latency.

### SpectralEmotionMapping

```rust
pub struct SpectralEmotionMapping {
    pub emotion: EmotionState,
    pub spectral_bias: [f32; 8],    // additive bias to all splats' spectral emission
}

pub enum EmotionState { Neutral, Anger, Sadness, Fear, Joy }
```

Mapping values (physically motivated):
- `Anger`: `spectral_bias = [0.0, 0.0, 0.0, 0.02, 0.03, 0.02, 0.0, 0.0]` — slight red-band elevation from increased surface blood flow.
- `Joy`: `spectral_bias = [0.0, 0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0]` — warmer skin tone from mild exercise.
- `Fear`: `spectral_bias = [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` — reduced near-UV (pallor from vasoconstriction).
- `Sadness`: zero bias (physiologically neutral).

The bias is applied in `MorphComputePass` as a post-morph addition to all face-region splats identified by a tag mask (`is_face_splat: bool` stored in `SplatMetadata`). Bias magnitude is subtle (≤ 3% of band maximum) and is intended as subconscious emotional signalling, not cartoon-level coloration.

---

## 4.4 Motion Matching

### PoseDatabase

```rust
pub struct PoseDatabase {
    pub poses: Vec<DatabasePose>,
    pub feature_dim: usize,          // 9
    pub kd_tree: KdTree<f32, u32, 9>, // kdtree crate; maps feature vec to pose index
}

pub struct DatabasePose {
    pub joint_positions:  Vec<Vec3>,
    pub joint_velocities: Vec<Vec3>,
    pub trajectory:       [Vec3; 4],     // predicted positions at +0.1s, +0.3s, +0.5s, +1.0s
    pub foot_contacts:    [bool; 2],     // left, right
    pub phase:            f32,           // [0, 2π], from foot contact pattern
    pub time_in_clip:     f32,
    pub clip_id:          u32,
}
```

Feature vector layout (9 dimensions):
```
[vel_x, vel_z, dir_x, dir_z,
 foot_l_pos.x, foot_l_pos.z,
 foot_r_pos.x, foot_r_pos.z,
 hip_vel.y]
```

Positions are in character-local space (relative to current character transform). This makes the database independent of character world position and facing direction.

### Query and Nearest Neighbour

Each frame: `MotionMatcher::query(desired: FeatureVec9) -> &DatabasePose`:
1. Build `desired` from controller input: `vel_x/z` from analog stick, `dir_x/z` from look direction, foot positions from previous frame's IK result, `hip_vel.y` from vertical controller velocity.
2. KD-tree L2 query returns the nearest `DatabasePose` index.
3. If the nearest pose is the current pose (continuation is natural), advance `time_in_clip` by `dt` and sample the current clip directly.
4. If switching to a new pose, hand off to `InertialBlender`.

KD-tree is rebuilt via `PoseDatabase::rebuild_tree()` only when the database changes (new animation clips added). Tree build cost at 10,000 poses: ~2ms on one thread; negligible.

### Inertial Blending

```rust
pub struct InertialBlender {
    pub current_pose: SkeletonPose,
    pub offset:       Vec<Vec3>,      // per-joint position offset currently being damped
    pub velocity:     Vec<Vec3>,      // per-joint velocity of offset
    pub half_life:    f32,            // default 0.1s
}
```

`InertialBlender::update(target_pose: &SkeletonPose, dt: f32) -> SkeletonPose`:

On each call to transition to a new `target_pose`:
1. Compute `offset[j] = current_pose.joint_pos[j] - target_pose.joint_pos[j]` for each joint.
2. Each frame, damp `offset[j]` toward zero using a critically-damped spring: `offset[j] += velocity[j] * dt; velocity[j] += -offset[j] / half_life² - 2.0 * velocity[j] / half_life` (spring-damper, no overshooting).
3. Output pose = `target_pose.joint_pos[j] + offset[j]`.

This produces smooth, physically-plausible transitions without explicit blend tree state machines. No blend weights to tune. No transition durations to author. The system converges to the target pose automatically.

### Phase Signal

Phase `φ ∈ [0, 2π]` is computed per clip by detecting left and right foot contact events (foot velocity below threshold + downward position). Each contact cycle is one full period. Phase is stored per `DatabasePose` at database build time. At query time, phase-matching adds a phase-compatibility term to the feature distance:

```
d_total = d_feature + λ * min_angular_distance(φ_candidate, φ_desired)
```

`λ = 0.3` (tunable). This prevents transitioning mid-swing to a pose that is mid-stance, which causes foot-skating artefacts.

### Database Authoring

`PoseDatabaseBuilder::ingest_clip(clip: &AnimationClip, skeleton: &GltfSkeleton)`:
1. Evaluate the clip at 10 Hz.
2. At each sample, compute `DatabasePose`: forward kinematics for joint world positions, finite difference for velocities, foot contact from `joint_pos[left_toe].y < contact_threshold`.
3. Compute trajectory: sample the clip's root motion at `+0.1s, +0.3s, +0.5s, +1.0s` from the current time.
4. Compute phase via foot contact detection.
5. Append to `poses`.

---

## 4.5 Inverse Kinematics

### IkChain

```rust
pub struct IkChain {
    pub joints:       Vec<JointId>,
    pub target:       Vec3,
    pub pole_vector:  Option<Vec3>,
    pub max_reach:    f32,           // sum of bone lengths; cached
    pub iterations:   u8,            // default 8
}
```

### FABRIK Solver

`FabrikSolver::solve(chain: &mut IkChain, joint_positions: &mut [Vec3])`:

Forward pass (tip to root):
```
positions[n] = chain.target
for i in (0..n).rev():
    dir = (positions[i] - positions[i+1]).normalize()
    positions[i] = positions[i+1] + dir * bone_lengths[i]
```

Backward pass (root to tip):
```
positions[0] = root_position  // pin root
for i in 0..n:
    dir = (positions[i+1] - positions[i]).normalize()
    positions[i+1] = positions[i] + dir * bone_lengths[i]
```

8 iterations is sufficient: FABRIK converges to < 1mm error within 4 iterations for most game configurations; 8 is a conservative ceiling. Convergence check: `if (positions[n] - target).length() < 0.001 { break }`.

Pole vector constraint (for knee/elbow direction): after backward pass, project the mid-joint onto the plane whose normal is `(root → tip)`, then rotate the projected position toward the pole vector. This keeps knees pointing forward and elbows pointing backward without additional iterations.

### Two-Bone Analytical IK

For chains of exactly 2 bones (upper arm + forearm, upper leg + lower leg), use the exact law-of-cosines solution instead of FABRIK:

```rust
fn two_bone_ik(root: Vec3, mid: Vec3, tip_target: Vec3,
               l0: f32, l1: f32, pole: Vec3) -> (Quat, Quat) {
    let d = (tip_target - root).length().min(l0 + l1 - 1e-4);
    let cos_angle_mid = (l0*l0 + l1*l1 - d*d) / (2.0 * l0 * l1);
    let angle_mid = cos_angle_mid.clamp(-1.0, 1.0).acos();
    // ... compute joint rotations from angle and pole vector
}
```

This gives an exact, single-step solution. Used at runtime; FABRIK is reserved for longer chains (spine, tail).

### Foot Placement

```rust
pub struct FootPlacement {
    pub left_foot_chain:  IkChain,
    pub right_foot_chain: IkChain,
    pub hip_compensation: f32,       // fraction of foot offset applied to hip
    pub raycast_offset:   f32,       // distance above ankle to start SDF raycast
}
```

`FootPlacement::update(terrain: &TerrainVolume, pose: &mut SkeletonPose)`:

For each foot:
1. Raycast downward from `ankle_world_pos + Vec3::Y * raycast_offset` against the terrain SDF using sphere-marching: `pos += sdf_value * direction` until `sdf_value < 0.005m` or max steps (64) reached.
2. If hit: set `IkChain::target = hit_point`. Compute `foot_offset = hit_point.y - original_ankle_y`.
3. Solve `two_bone_ik` for the leg chain.
4. Apply hip compensation: `hip_joint.translation.y -= foot_offset * hip_compensation` to avoid leg hyperextension when one foot is on a raised surface. `hip_compensation = 0.5` means the hip moves halfway to meet the foot.
5. Foot rotation: align the foot bone's forward axis to the terrain normal at the hit point using a `Quat::from_rotation_arc` correction.

### Hand IK

`HandIk::grab(target_pos: Vec3, grab_normal: Vec3, arm: Arm)`: Sets up a 2-bone IK chain from wrist to shoulder. Target is the object's registered grab point. After IK, the wrist is rotated to align the palm normal to `grab_normal` using a secondary `Quat::from_rotation_arc`.

### IK Stack Order

Frame pipeline for a character: (1) motion matching → raw pose, (2) inertial blending → smooth raw pose, (3) `FootPlacement` IK, (4) `HandIk` grab IK, (5) `BlendSkinningCompute` → skinned splats, (6) `MorphComputePass` → morphed splats, (7) EWA tile render.

---

## 4.6 Cloth Simulation

### Types

```rust
pub struct ClothMesh {
    pub particles:     Vec<ClothParticle>,
    pub constraints:   Vec<ClothConstraint>,
    pub splat_bindings: Vec<SplatBinding>,
    pub wind_field:    WindField,
}

pub struct ClothParticle {
    pub position:      Vec3,
    pub prev_position: Vec3,
    pub inv_mass:      f32,   // 0.0 = pinned particle
}

pub enum ClothConstraint {
    Distance { p0: u32, p1: u32, rest_length: f32, stiffness: f32 },
    Bend     { p0: u32, p1: u32, p2: u32, p3: u32, rest_angle: f32, stiffness: f32 },
    Pin      { particle_id: u32, world_pos: Vec3 },
}

pub struct SplatBinding {
    pub splat_index: u32,
    pub p0: u32, pub p1: u32, pub p2: u32,
    pub barycentric: [f32; 3],
}

pub struct WindField {
    pub direction: Vec3,
    pub speed: f32,
    pub turbulence: f32,
}
```

`Pin` constraints attach the cloth to the character's skeleton: waist particles are pinned to the hip bone's world position, updated each frame before XPBD solve.

### XPBD Solver

`ClothMesh::simulate(dt: f32, substeps: u32 = 8)`:

Per substep (`sub_dt = dt / substeps`):
1. **Predict positions**: `predicted = position + (position - prev_position) + gravity * sub_dt²`. Add wind force: `predicted += wind_force(position, wind_field) * sub_dt²`.
2. **Project constraints** (in order: Pin, Distance, Bend):
   - `Distance`: `delta = predicted[p1] - predicted[p0]; error = delta.length() - rest_length; correction = error * stiffness / (inv_mass[p0] + inv_mass[p1]); predicted[p0] += correction * inv_mass[p0] * normalize(delta); predicted[p1] -= correction * inv_mass[p1] * normalize(delta)`.
   - `Bend`: dihedral angle constraint between two triangles sharing an edge (p1-p2). Compute current dihedral angle, compare to `rest_angle`, apply correction to all 4 particles weighted by `inv_mass`.
3. **Update velocity**: `prev_position = position; position = predicted`.

XPBD guarantees stability independent of `stiffness` values (unlike PBD), allowing stiff cloth without explosions.

`wind_force(pos, wind_field)`:
```rust
let turbulence_noise = simplex3d(pos * wind_field.turbulence, time);
wind_field.direction * wind_field.speed + turbulence_noise * wind_field.turbulence
```

Simplex noise is evaluated per particle per substep using a fast inline implementation (no external crate needed; 50-line implementation).

### Splat Binding Update

After each frame's simulation (all substeps complete), `ClothMesh::update_splats(splats: &mut [GaussianSplat])`:

For each `SplatBinding`:
```rust
let p = bary[0] * particles[p0].position
      + bary[1] * particles[p1].position
      + bary[2] * particles[p2].position;
splats[splat_index].position = p.into();
```

Scale and rotation of cloth splats are updated from the local frame of the triangle (tangent/bitangent/normal computed from particle positions). This runs on rayon parallel iter for performance.

### Self-Collision

`ClothSelfCollision::resolve(particles: &mut [ClothParticle], cell_size: f32)`:
1. Build a `HashMap<(i32,i32,i32), Vec<u32>>` spatial hash grid. Key = `floor(position / cell_size)` per axis.
2. For each particle, check its own cell and 26 neighbours.
3. For each pair closer than `2 * particle_radius`, apply repulsion: `delta = p_j.position - p_i.position; if delta.length() < 2r { correction = (2r - delta.length()) * 0.5 * normalize(delta); p_i.position -= correction * inv_mass_i / (inv_mass_i + inv_mass_j); p_j.position += correction * inv_mass_j / (inv_mass_i + inv_mass_j); }`.

---

## 4.7 Hair Rendering & Simulation

### HairStrand

```rust
pub struct HairStrand {
    pub root_pos:        Vec3,
    pub control_points:  Vec<Vec3>,   // 8 points per strand
    pub width_curve:     Vec<f32>,    // width at each control point; tapers toward tip
    pub spectral_melanin: [f32; 8],   // melanin absorption model per spectral band
}

pub struct HairGroom {
    pub strands:    Vec<HairStrand>,
    pub lod_radius: HairLodRadius,
}

pub struct HairLodRadius {
    pub full_sim:    f32,   // default 5.0m
    pub rigid:       f32,   // default 20.0m; beyond = culled
}
```

### Spectral Melanin Model

`HairStrand::compute_spectral_melanin(eumelanin_density: f32, pheomelanin_density: f32) -> [f32; 8]`:

Spectral reflectance of hair is modelled from first principles (after d'Eon et al.):

- **Eumelanin** absorption coefficient spectrum (per unit density): high across all bands, peak in UV-blue. Absorption array (relative): `[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]` for bands 0–7.
- **Pheomelanin** absorption coefficient spectrum: concentrated in UV/blue, negligible above 600nm. Absorption array: `[0.8, 0.7, 0.5, 0.2, 0.05, 0.02, 0.01, 0.0]`.

Combined absorption: `A[b] = eumelanin_density * A_eu[b] + pheomelanin_density * A_ph[b]`.

Reflectance: `spectral_melanin[b] = exp(-A[b])`.

Example outputs:
- Jet black (`eu = 3.0, ph = 0.0`): `spectral_melanin ≈ [0.07, 0.09, 0.13, 0.17, 0.22, 0.30, 0.35, 0.41]` — dark everywhere.
- Red (`eu = 0.3, ph = 1.5`): `spectral_melanin ≈ [0.10, 0.14, 0.28, 0.72, 0.95, 0.97, 0.99, 1.0]` — absorbs UV/blue, reflects red.
- Blonde (`eu = 0.1, ph = 0.1`): `spectral_melanin ≈ [0.83, 0.85, 0.88, 0.97, 0.99, 0.99, 0.99, 1.0]` — high reflectance across bands.

This is authored via two float sliders in the editor; the full 8-band spectral profile is computed automatically.

### Splat Rendering of Strands

For each strand, `HairSplatGenerator::generate(strand: &HairStrand) -> Vec<GaussianSplat>`:

Evaluate the strand's cubic Catmull-Rom spline through its 8 control points at `N_splats = 16` intervals. At each point:
- `position` = spline point
- `rotation` = quaternion aligning splat Z axis to spline tangent
- `scale` = `[width_at_t / 2, width_at_t / 2, segment_length / 2]` — elongated along tangent
- `spectral` = `strand.spectral_melanin` converted to f16
- `opacity` = 220 (slightly transparent to allow sub-strand light scattering)

Total splats per character head: `strand_count * 16`. For 5000 strands: 80,000 hair splats.

### Hair Simulation

```rust
pub struct HairSimulation {
    pub strands: Vec<HairParticleStrand>,
}

pub struct HairParticleStrand {
    pub particles:    [Vec3; 8],
    pub prev_pos:     [Vec3; 8],
    pub rest_dirs:    [Vec3; 7],     // rest-pose direction for each segment (shape constraint)
    pub root_joint:   JointId,       // follows character skeleton
}
```

`HairSimulation::simulate(dt: f32, wind: &WindField, skeleton_pose: &SkeletonPose)`:

Per strand, 4 XPBD substeps:
1. Pin particle 0 to `skeleton_pose.joint_world_pos[root_joint]`.
2. Predict all particles with gravity + wind.
3. Project distance constraints (rest length = original segment length, stiffness = 0.9).
4. Project shape constraints: each segment tries to maintain its rest-pose direction relative to the previous segment (stiffness = 0.3). This gives hair body without full rigidity.
5. Update positions.

After simulation, regenerate strand splats from particle positions via `HairSplatGenerator::generate_from_particles`.

### Hair LOD

`HairLodSelector::select(strand: &HairStrand, camera_dist: f32) -> HairLodMode`:
- `camera_dist < full_sim_radius`: full per-strand XPBD simulation.
- `full_sim_radius <= camera_dist < rigid_radius`: rigid body approximation — entire strand transforms rigidly with root joint, no simulation.
- `camera_dist >= rigid_radius`: strand culled from render list entirely.

At the full→rigid transition, `InertialBlender` is applied to the strand particle positions over 0.2s to avoid popping.

---

## File Map

```
vox_core/src/
  skinning.rs           — SplatSkinData, SkeletonRetargeter, JointName, JointCurve, BSpline<T>
  animation_clip.rs     — AnimationClip, BSpline evaluation (De Boor), RootMotionCurve
  morph_target.rs       — SplatDelta, MorphTarget, MorphTargetSet
  facial_rig.rs         — FacialRig, ActionUnit, AuMorphMapping, EmotionState, SpectralEmotionMapping
  motion_matching.rs    — PoseDatabase, DatabasePose, FeatureVec9, MotionMatcher, InertialBlender
  ik.rs                 — IkChain, FabrikSolver, two_bone_ik, FootPlacement, HandIk
  cloth.rs              — ClothMesh, ClothParticle, ClothConstraint, SplatBinding, WindField
  hair.rs               — HairStrand, HairGroom, HairParticleStrand, HairSimulation, HairLodRadius

vox_render/src/
  morph_compute.rs      — MorphComputePass, wgpu pass setup, weight uniform upload
  hair_splat_gen.rs     — HairSplatGenerator, splat array generation from strand/particles
  cloth_splat_update.rs — ClothMesh::update_splats rayon parallel impl

vox_render/shaders/
  morph_targets.wgsl    — morph compute shader (base + delta + weights → output)
  blend_skinning.wgsl   — extended with SplatSkinData bind group, 4-weight skin matrix

vox_audio/src/
  phoneme_classifier.rs — PhonemeClassifier, MFCC, embedded weights, frame buffer
  lip_sync.rs           — AudioLipSync, VisemeTable, Phoneme enum

vox_app/src/
  character.rs          — CharacterController integration: motion matching, IK, cloth, hair per-frame update
  morph_target_authorer.rs — editor: diff base vs deformed pose → SplatDelta array
  hair_editor.rs        — groom painting, strand density, melanin sliders
  foot_placement.rs     — FootPlacement system, SDF raycast, hip compensation
```

---

## Milestones

### M1 — Multi-Weight Skinning (2 days)
- `SplatSkinData` type; parallel array serialisation in `.vxm` format.
- `blend_skinning.wgsl` updated: 4-weight skin matrix accumulation; bind group extended.
- Inverse bind matrix upload at character load.
- **Acceptance:** elbow joint of test character shows smooth deformation with 2 bone weights; no crease artefact visible at joint bend > 90°.

### M2 — Morph Targets (3 days)
- `SplatDelta`, `MorphTarget`, `MorphTargetSet` types.
- `morph_targets.wgsl` compute shader; `MorphComputePass` wgpu integration.
- Morph authoring tool in editor (diff-based from deformed pose).
- `FacialRig`, `SpectralEmotionMapping`, `AudioLipSync` with `PhonemeClassifier`.
- **Acceptance:** smile morph target on test head character increases red-band emission of cheek splats by ≥ 15%; audio lip sync matches phoneme onset within ±2 frames at 60 FPS.

### M3 — Motion Matching (3 days)
- `PoseDatabase`, `DatabasePose`, KD-tree construction and query.
- `MotionMatcher::query` with phase-weighted distance.
- `InertialBlender` spring-damper.
- Root motion extraction and `CharacterController` integration.
- **Acceptance:** locomotion test: character walks, runs, turns on flat ground with no foot-skating; transition latency < 2 frames; foot contact preserved through transitions.

### M4 — Inverse Kinematics (2 days)
- `FabrikSolver`, `two_bone_ik` (analytical).
- `FootPlacement`: SDF raycast per foot, hip compensation.
- `HandIk` grab IK.
- IK layer integrated into per-frame character update pipeline.
- **Acceptance:** character stands on uneven SDF terrain (slope up to 30°) with both feet planted correctly; no knee hyperextension up to 0.5m height difference between feet.

### M5 — Cloth Simulation (3 days)
- `ClothMesh`, `ClothParticle`, `ClothConstraint`, `SplatBinding`.
- XPBD solver: distance + bend + pin constraints; 8 substeps.
- `WindField` turbulence (simplex noise).
- Self-collision spatial hash.
- Splat binding rayon parallel update.
- **Acceptance:** cape cloth on character falls correctly under gravity; cloth does not self-intersect at rest; wind response visually convincing at 3 m/s wind speed; simulation stable at 60 FPS.

### M6 — Hair Rendering & Simulation (3 days)
- `HairStrand`, melanin model, `HairSplatGenerator`.
- `HairSimulation` XPBD per-strand with shape constraints.
- LOD selection: full sim / rigid / cull.
- Melanin editor sliders → spectral array.
- **Acceptance:** character with 5000 hair strands renders at ≥ 60 FPS at 3m distance; hair responds to wind; red/black/blonde melanin variants have perceptually correct spectral hue under neutral illumination.

**Total estimated effort:** 16 engineering-days.

---

## Acceptance Criteria (System-Level)

1. 10 simultaneous characters with full skinning, morph targets active, cloth, and hair render at ≥ 60 FPS at 1440p on RTX 4070.
2. Skinning: no crease artefacts at any joint bend angle in [0°, 150°] range.
3. Morph targets: up to 16 simultaneous active morph targets per character with no measurable FPS regression vs 0 active targets (verify with GPU profiler).
4. Motion matching: character transitions between locomotion states without any foot-skating frame detectable by visual inspection.
5. Foot IK: both feet planted correctly on SDF terrain with slope up to 35°, height difference up to 1.0m, verified across 100 random terrain positions.
6. Cloth: XPBD stable at all legal `dt` values (30–120 FPS equivalent); no particle explosion.
7. Hair melanin model: red, blonde, black, and brown presets match physical reference photographs by spectral band under D65 illumination (visual match, not formal measurement).
8. Spectral emotion mapping: angry-state character has measurably higher red-band emission (≥ 1% above neutral) confirmed by reading back GPU output splat buffer.
9. Audio lip sync: phoneme onset error ≤ 40ms relative to audio timeline at 60 FPS.

---

## Effort Summary

| Area | Days |
|------|------|
| M1 Multi-Weight Skinning | 2 |
| M2 Morph Targets + Facial | 3 |
| M3 Motion Matching | 3 |
| M4 Inverse Kinematics | 2 |
| M5 Cloth Simulation | 3 |
| M6 Hair | 3 |
| **Total** | **16** |
