# Domain 9 — AI & Gameplay Systems

**Status:** Draft — 2026-03-29
**Scope:** Production-grade ECS, behavior trees, environment query system, AI perception, navmesh extensions, gameplay framework
**Engine:** Ochroma spectral Gaussian Splatting — Rust workspace, wgpu 24, WGSL shaders, rayon, glam, egui, rhai, rapier3d

---

## Goals

Domain 9 delivers the complete runtime AI and gameplay stack on top of the engine primitives established in Domains 1–8. The existing `EngineRuntime` with its ad-hoc `spawn`/`with_position`/`with_asset` API is replaced by an archetype-based ECS capable of parallel system execution. A full behavior tree runtime, environment query system, and multi-modal AI perception stack — including Ochroma-unique spectral perception — are built on top. The NavMesh A* implementation gains ORCA crowd simulation, dynamic obstacles, path smoothing, and jump links. A thin but complete gameplay framework (GameMode, PlayerState, SaveGame, input mapping, achievements) closes the gap between the engine and playable experiences.

The entire domain must integrate with the existing crates: `vox_core`, `vox_render`, `vox_terrain`, `vox_physics`, `vox_audio`, `vox_script`. Engine crates must remain game-agnostic; all domain-specific implementations (game modes, achievement lists) live in `vox_app` or `vox_sim`.

**Performance targets:**
- 10,000 active entities at 60 Hz with all systems running, on a single 8-core CPU
- 500 simultaneous AI agents with full behavior trees and perception at 60 Hz
- EQS query (100 points, 4 tests) completes in < 2 ms on rayon threadpool
- Archetype query over 100K entities with 3 components: < 0.5 ms

---

## Architecture

```
vox_core/
  ecs/
    archetype.rs        — Archetype, ComponentColumn, ArchetypeLocation
    world.rs            — World, EntityId, entity_index
    query.rs            — Query<(A,B,C)>, QueryFilter, QueryIter
    system.rs           — System trait, SystemMeta, SystemScheduler
    resources.rs        — Resources, ResourceCell
    events.rs           — EventWriter<T>, EventReader<T>, EventQueue
  ai/
    behavior_tree.rs    — BehaviorNode, BehaviorStatus, BehaviorContext
    blackboard.rs       — Blackboard, BlackboardValue
    nodes/
      composite.rs      — Sequence, Selector, Parallel
      decorator.rs      — Inverter, Repeater, Cooldown, ForceSuccess, ForceFailure
      leaf.rs           — MoveTo, LookAt, PlayAnimation, Wait, SetBlackboard,
                          FireRayCast, SpectralCheck
    eqs.rs              — EnvQuery, QueryGenerator, QueryTest, QueryPoint, EqsContext
    generators.rs       — GridGenerator, NavmeshGenerator, PathfindableGenerator
    tests.rs            — DistanceTest, LineOfSightTest, CoverTest,
                          SpectralBandTest, NavMeshReachabilityTest
    perception.rs       — PerceptionComponent, PerceivedStimulus, PerceptionSystem
    team.rs             — Team, TeamRegistry
    crowd.rs            — CrowdManager, OrcaAgent, OrcaSolver
  gameplay/
    game_mode.rs        — GameMode trait, WinResult
    player_state.rs     — PlayerState, TeamId
    save_game.rs        — SaveGame, WorldSnapshot
    input_map.rs        — InputActionMap, InputBinding, ActionId
    achievements.rs     — AchievementSystem, Achievement, AchievementId

vox_app/
  modes/
    deathmatch.rs
    coop.rs
    sandbox.rs
    spectral_hunt.rs
  achievements/
    spectral_achievements.rs
```

---

## 9.1 Entity Component System (Production-Grade)

### Current State

`EngineRuntime` in `vox_core/src/runtime.rs` maintains a flat `Vec<Entity>` with per-entity `HashMap<TypeId, Box<dyn Any>>`. This is correct but slow: component access is a double indirection (entity index → HashMap lookup → Box downcast), cache locality is zero, and there is no mechanism for parallel system execution.

### Archetype Storage Model

The replacement is an archetype-based ECS following the sparse-set / archetype hybrid model described by the Bevy ECS design document (Adams, 2020). Components are stored in contiguous column arrays — SoA layout — so iterating a single component type produces sequential memory access with no stride.

```rust
// vox_core/src/ecs/archetype.rs

use std::alloc::{alloc, dealloc, Layout};
use std::any::TypeId;
use std::collections::HashMap;

pub struct ComponentColumn {
    data:     *mut u8,
    len:      usize,
    capacity: usize,
    item_size: usize,
    item_align: usize,
    drop_fn:  unsafe fn(*mut u8),   // type-erased destructor
}

impl ComponentColumn {
    pub fn new<T: 'static>() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
            item_size: std::mem::size_of::<T>(),
            item_align: std::mem::align_of::<T>(),
            drop_fn: |ptr| unsafe { std::ptr::drop_in_place(ptr as *mut T) },
        }
    }

    /// Push a value by moving raw bytes. Caller must ensure T matches column type.
    pub unsafe fn push_raw(&mut self, src: *const u8) { /* grow + copy */ }

    /// Get pointer to element at row `r`.
    pub unsafe fn get_ptr(&self, r: usize) -> *mut u8 {
        self.data.add(r * self.item_size)
    }

    /// Swap-remove row `r`; returns the entity that was displaced (if any).
    pub unsafe fn swap_remove(&mut self, r: usize) { /* memmove last → r */ }
}

pub struct Archetype {
    pub component_types: Vec<TypeId>,
    pub columns: HashMap<TypeId, ComponentColumn>,
    pub entity_ids: Vec<EntityId>,   // parallel array: row → entity
}

impl Archetype {
    pub fn matches(&self, required: &[TypeId], excluded: &[TypeId]) -> bool {
        required.iter().all(|t| self.component_types.contains(t))
            && !excluded.iter().any(|t| self.component_types.contains(t))
    }
}
```

`ComponentColumn` allocates its backing store via the global allocator with explicit `Layout`, bypassing Vec to avoid monomorphization for each component type. `item_size` and `item_align` are captured at column creation from `size_of::<T>()` and `align_of::<T>()`. The `drop_fn` is a type-erased destructor pointer so `ComponentColumn::drop()` can call component destructors without knowing the type at call site.

### World and Entity Index

```rust
// vox_core/src/ecs/world.rs

pub type EntityId = u64;   // generation (high 32) | index (low 32)

pub struct ArchetypeLocation {
    pub archetype_id: u32,
    pub row: u32,
}

pub struct World {
    archetypes:   Vec<Archetype>,
    entity_index: HashMap<EntityId, ArchetypeLocation>,
    next_entity:  std::sync::atomic::AtomicU64,
    free_list:    Vec<EntityId>,   // recycled entity slots
}

impl World {
    pub fn spawn(&mut self) -> EntityBuilder<'_> { ... }
    pub fn despawn(&mut self, id: EntityId) { ... }

    pub fn get_component<T: 'static>(&self, id: EntityId) -> Option<&T> {
        let loc = self.entity_index.get(&id)?;
        let arch = &self.archetypes[loc.archetype_id as usize];
        let col = arch.columns.get(&TypeId::of::<T>())?;
        Some(unsafe { &*(col.get_ptr(loc.row as usize) as *const T) })
    }

    pub fn get_component_mut<T: 'static>(&mut self, id: EntityId) -> Option<&mut T> { ... }

    /// Move entity to new archetype when adding/removing a component.
    fn migrate(&mut self, id: EntityId, new_type_set: Vec<TypeId>) { ... }
}
```

Entity IDs pack a 32-bit generation counter into the high word to catch use-after-despawn bugs. When an entity is despawned its slot goes on `free_list`; the next spawn from that slot increments the generation, invalidating all stale `EntityId` copies.

### Query System

```rust
// vox_core/src/ecs/query.rs

pub struct Query<'w, Q: QueryParam> {
    world: &'w World,
    _marker: PhantomData<Q>,
}

pub trait QueryParam {
    type Item<'a>;
    fn type_ids() -> Vec<TypeId>;
    unsafe fn fetch<'a>(arch: &'a Archetype, row: usize) -> Self::Item<'a>;
}

// Blanket impl for tuples up to 8 via macro:
// impl<A: Component, B: Component> QueryParam for (A, B) { ... }

pub struct QueryFilter<Include, Exclude> {
    _i: PhantomData<Include>,
    _e: PhantomData<Exclude>,
}

pub struct With<T>(PhantomData<T>);
pub struct Without<T>(PhantomData<T>);
```

`Query<(A, B, C)>` iterates all archetypes in `World` that contain `TypeId::of::<A>()`, `TypeId::of::<B>()`, and `TypeId::of::<C>()`. For each matching archetype it iterates rows, calling `unsafe { transmute }` from the column's raw bytes to typed references. This is safe because:

1. The column was constructed with `ComponentColumn::new::<T>()` capturing the correct `item_size` and `item_align`.
2. The archetype's `component_types` is the ground truth; if `TypeId::of::<T>()` is in the set, the column bytes are valid `T`.
3. Lifetime is bounded by `'w` (the world borrow), preventing concurrent mutation via the query.

`QueryFilter<With<A>, Without<B>>` adds a filter stage in `Archetype::matches()`: archetypes missing `A` or containing `B` are skipped entirely.

### System Scheduling and Parallelism

```rust
// vox_core/src/ecs/system.rs

pub trait System: Send + Sync {
    fn reads(&self) -> Vec<TypeId>;
    fn writes(&self) -> Vec<TypeId>;
    fn run(&self, world: &World, resources: &Resources);
}

pub struct SystemMeta {
    pub id:     SystemId,
    pub reads:  Vec<TypeId>,
    pub writes: Vec<TypeId>,
    pub deps:   Vec<SystemId>,   // explicit ordering constraints
}

pub struct SystemScheduler {
    systems:          Vec<(SystemMeta, Box<dyn System>)>,
    dependency_graph: petgraph::Graph<SystemId, ()>,
    parallel_groups:  Vec<Vec<SystemId>>,  // computed once at build time
}

impl SystemScheduler {
    pub fn build(&mut self) {
        // Topological sort dependency_graph → stages.
        // Within each stage, group systems whose write sets are disjoint → parallel group.
        // Systems with overlapping write sets are sequenced by topological order within stage.
        self.parallel_groups = self.compute_parallel_groups();
    }

    pub fn run_frame(&self, world: &World, resources: &Resources) {
        for group in &self.parallel_groups {
            rayon::scope(|s| {
                for &sys_id in group {
                    let sys = &self.systems[sys_id as usize].1;
                    s.spawn(|_| sys.run(world, resources));
                }
            });
        }
    }
}
```

`SystemScheduler::build()` runs once at startup. The algorithm:

1. Topological sort `dependency_graph` (petgraph `toposort`) to produce an ordered list of systems.
2. Greedily group consecutive systems: two systems A and B can share a rayon scope if `A.writes ∩ B.writes = ∅` and `A.reads ∩ B.writes = ∅` and `B.reads ∩ A.writes = ∅`.
3. Systems that write the same component type are always sequenced.

This is the Bevy "stage" model generalized to dynamic write-set intersection. At runtime, each parallel group dispatches into `rayon::scope` — rayon's work-stealing scheduler handles load balancing across the CPU thread pool. No synchronization primitives are needed during execution because the scheduler has already proven non-overlapping write sets at build time.

### Resources

```rust
// vox_core/src/ecs/resources.rs

pub struct Resources {
    map: HashMap<TypeId, ResourceCell>,
}

pub struct ResourceCell {
    data:  *mut u8,
    rw:    std::sync::RwLock<()>,  // guards concurrent read/write
    drop:  unsafe fn(*mut u8),
}

impl Resources {
    pub fn insert<T: 'static + Send + Sync>(&mut self, val: T) { ... }
    pub fn get<T: 'static>(&self) -> Option<ResourceRef<'_, T>> { ... }
    pub fn get_mut<T: 'static>(&mut self) -> Option<ResourceMut<'_, T>> { ... }
}
```

Resources hold globally shared non-component data: `Time { delta: f32, total: f64 }`, `InputState`, `SpatialAudioManager`, `RenderCamera`. Systems declare resource reads/writes the same way as component reads/writes; the scheduler uses them to determine parallelism.

### Events

```rust
// vox_core/src/ecs/events.rs

pub struct EventQueue<T> {
    current: Vec<T>,
    previous: Vec<T>,
}

pub struct EventWriter<'a, T> { queue: &'a mut EventQueue<T> }
pub struct EventReader<'a, T> { queue: &'a EventQueue<T> }

impl<T> EventQueue<T> {
    pub fn flush(&mut self) {
        std::mem::swap(&mut self.current, &mut self.previous);
        self.current.clear();
    }
}
```

Events are double-buffered: `EventWriter` pushes to `current`; `EventReader` reads from `previous` (written last frame). At end-of-frame, `flush()` swaps the buffers and clears current. Events live for exactly one frame — same model as Bevy events. `EventQueue<T>` is stored in `Resources` keyed by `TypeId::of::<EventQueue<T>>()`.

### Migration from EngineRuntime

`EngineRuntime` is wrapped: `LegacyRuntime { world: World, ... }` implements the old `spawn`/`with_position`/`with_asset` methods by translating them into `world.spawn().with::<Position>(...)`. Subsystems migrate incrementally: each subsystem is moved to a `System` impl when its component types are stable. The old HashMap-per-entity storage is removed subsystem by subsystem over the migration sprint.

---

## 9.2 Behavior Tree System

### Core Traits

```rust
// vox_core/src/ai/behavior_tree.rs

pub enum BehaviorStatus { Success, Failure, Running }

pub trait BehaviorNode: Send + Sync {
    fn tick(&self, ctx: &mut BehaviorContext<'_>) -> BehaviorStatus;
    fn reset(&self, ctx: &mut BehaviorContext<'_>) {}   // called when parent aborts
}

pub struct BehaviorContext<'w> {
    pub entity:     EntityId,
    pub world:      &'w World,
    pub blackboard: &'w mut Blackboard,
    pub navmesh:    &'w NavMesh,
    pub physics:    &'w PhysicsWorld,
    pub dt:         f32,
}
```

### Blackboard

```rust
// vox_core/src/ai/blackboard.rs

pub enum BlackboardValue {
    Float(f32),
    Vec3(glam::Vec3),
    EntityRef(EntityId),
    Bool(bool),
    SpectralProfile([f32; 8]),
}

pub struct Blackboard {
    entries: HashMap<String, BlackboardValue>,
}

impl Blackboard {
    pub fn set(&mut self, key: impl Into<String>, val: BlackboardValue) { ... }
    pub fn get(&self, key: &str) -> Option<&BlackboardValue> { ... }
    pub fn get_vec3(&self, key: &str) -> Option<glam::Vec3> { ... }
    pub fn get_float(&self, key: &str) -> Option<f32> { ... }
}
```

`Blackboard` is the sole communication channel between `BehaviorNode` implementations. No node holds mutable state itself — all inter-tick state is written to and read from the blackboard. This makes the tree stateless and serializable: save the blackboard, save the tree position (via a running-node stack), and the AI state is fully captured.

### Composite Nodes

**Sequence:** Ticks children left-to-right. Returns `Running` while the currently-executing child returns `Running`. Returns `Failure` on the first child `Failure`. Returns `Success` only when all children return `Success`. Remembers the currently-running child index in the blackboard under a generated key (`__seq_{node_id}_idx`) so it resumes correctly on the next tick without re-evaluating completed children.

**Selector:** Ticks children left-to-right. Returns `Running` while the currently-executing child returns `Running`. Returns `Success` on the first child `Success`. Returns `Failure` only when all children return `Failure`.

**Parallel:** Ticks all children every tick regardless of individual status. `success_policy: NRequired(n)` — returns `Success` when `n` children have returned `Success`. `failure_policy: NRequired(n)` — returns `Failure` when `n` children have returned `Failure`. `Parallel { success_policy: NRequired(all), failure_policy: NRequired(1) }` is a common "all succeed or any fails" configuration for coordinated multi-condition gates.

### Decorator Nodes

- `Inverter`: returns `Success` on child `Failure`, `Failure` on child `Success`, `Running` passthrough.
- `Repeater { n: Option<u32> }`: repeats child `n` times (or infinitely if `None`); resets child after each `Success`; propagates `Failure` immediately.
- `Cooldown { duration: f32 }`: after child `Success`, blocks re-entry for `duration` seconds (time tracked in blackboard under `__cd_{node_id}_remaining`); returns `Failure` while on cooldown.
- `ForceSuccess`: runs child; always returns `Success`.
- `ForceFailure`: runs child; always returns `Failure`.

### Leaf Task Nodes

```rust
// vox_core/src/ai/nodes/leaf.rs

pub struct MoveTo {
    pub target_key:     String,   // blackboard key → Vec3
    pub arrival_radius: f32,
    pub path_key:       String,   // blackboard key → Vec<Vec3> (stored path)
}

impl BehaviorNode for MoveTo {
    fn tick(&self, ctx: &mut BehaviorContext<'_>) -> BehaviorStatus {
        let target = ctx.blackboard.get_vec3(&self.target_key)?;  // Failure if missing
        let agent_pos = ctx.world.get_component::<Position>(ctx.entity)?.0;

        if agent_pos.distance(target) <= self.arrival_radius {
            return BehaviorStatus::Success;
        }

        // Re-path if no stored path or target moved significantly
        let path_stale = /* check blackboard path last-target vs current target */;
        if path_stale {
            let path = ctx.navmesh.find_path(agent_pos, target)?;
            let smoothed = funnel_smooth(&path, ctx.navmesh);
            ctx.blackboard.set(&self.path_key, BlackboardValue::Path(smoothed));
        }

        let next_waypoint = /* first point in stored path beyond arrival_radius */;
        let dir = (next_waypoint - agent_pos).normalize();

        // Drive rapier3d character controller
        let cc = ctx.world.get_component_mut::<CharacterController>(ctx.entity)?;
        cc.desired_velocity = dir * cc.max_speed;

        BehaviorStatus::Running
    }
}
```

`MoveTo` calls `NavMesh::find_path()` only when the cached path is stale (first tick, or target moved > 1 m). It then drives the `rapier3d` `KinematicCharacterController` via the `CharacterController` component rather than setting position directly, so physics handles step-ups, slopes, and CCD.

**SpectralCheck** — unique to Ochroma:

```rust
pub struct SpectralCheck {
    pub assembly_key: String,     // blackboard key → EntityId (a SplatAssembly)
    pub band:         usize,      // 0–7
    pub threshold:    f32,        // [0, 1]
    pub comparator:   Comparator, // GreaterThan, LessThan, EqualEpsilon(f32)
}

pub enum Comparator { GreaterThan, LessThan, EqualEpsilon(f32) }

impl BehaviorNode for SpectralCheck {
    fn tick(&self, ctx: &mut BehaviorContext<'_>) -> BehaviorStatus {
        let asm_entity = ctx.blackboard.get_entity(&self.assembly_key)?;
        let assembly = ctx.world.get_component::<SplatAssembly>(asm_entity)?;
        let band_val = assembly.mean_spectral[self.band];
        let passes = match self.comparator {
            Comparator::GreaterThan    => band_val > self.threshold,
            Comparator::LessThan       => band_val < self.threshold,
            Comparator::EqualEpsilon(e) => (band_val - self.threshold).abs() < e,
        };
        if passes { BehaviorStatus::Success } else { BehaviorStatus::Failure }
    }
}
```

`SpectralCheck` enables agents that react to the spectral composition of scene objects: a guard patrols toward warm-light sources (high band 5–7), an insect swarms UV emitters (band 0), a creature flees IR-bright fire (band 7). This has no Unreal equivalent; Unreal agents can only react to gameplay tags or numeric properties manually set by designers.

### Authoring and Serialization

`BehaviorTreeEditor` in `vox_app/src/editor/bt_editor.rs` is an egui panel. Nodes appear as rounded rectangles; drag-and-drop reorders children. Each node type has an egui property inspector autogenerated via a `BehaviorNodeInspect` derive macro that reads field names and types.

Trees serialize to RON (Rusty Object Notation) via `serde`'s `#[derive(Serialize, Deserialize)]` on all node types dispatched through a `typetag`-registered trait object. Example:

```ron
Sequence(
    children: [
        SpectralCheck(assembly_key: "nearest_light", band: 5, threshold: 0.6,
                      comparator: GreaterThan),
        MoveTo(target_key: "patrol_waypoint", arrival_radius: 0.5,
               path_key: "__move_path"),
        Wait(duration: 2.0),
    ]
)
```

Hot-reloading: the `vox_script::rhai_runtime::RhaiRuntime` file-watch mechanism is re-used — when a `.ron` tree file changes on disk, `BehaviorTreeAsset` reloads and replaces the `BehaviorTree` component on all agents using that tree. Agent blackboards are preserved across hot-reloads.

---

## 9.3 Environment Query System (EQS)

### Overview

EQS finds optimal positions or entities for AI decision-making by generating candidate points, running scored tests over them, and returning the best result. It is the AI equivalent of the spatial query API in `vox_terrain`.

```rust
// vox_core/src/ai/eqs.rs

pub struct EnvQuery {
    pub generator:    Box<dyn QueryGenerator>,
    pub tests:        Vec<Box<dyn QueryTest>>,
    pub scoring_mode: ScoringMode,
}

pub struct QueryPoint {
    pub position: glam::Vec3,
    pub score:    f32,
    pub metadata: HashMap<String, f32>,
}

pub enum ScoringMode { Max, Average, Product }

pub struct EqsContext<'w> {
    pub querier_pos:    glam::Vec3,
    pub querier_entity: EntityId,
    pub world:          &'w World,
    pub navmesh:        &'w NavMesh,
    pub gi_probes:      &'w ProbeGrid,
    pub physics:        &'w PhysicsWorld,
}
```

### Generators

**GridGenerator:** Produces a uniform grid of points at `spacing` intervals within `radius` of the querier. Points outside the NavMesh AABB are discarded. Generates O((radius/spacing)²) points.

**NavmeshGenerator:** Calls `navmesh.nearest_node(querier_pos)` then BFS-expands the NavMesh graph to all nodes within `radius` meters (graph distance, not Euclidean). Returns each node's centroid as a `QueryPoint`. Produces exactly the set of reachable navigation positions — no off-mesh garbage.

**PathfindableGenerator:** Extension of `NavmeshGenerator` that additionally verifies each point with `navmesh.find_path(querier_pos, point_pos).is_some()` — eliminates nodes reachable in the graph but physically disconnected due to dynamic obstacle blocking. Slower but produces strictly valid results.

### Tests

```rust
pub trait QueryTest: Send + Sync {
    /// Returns a score in [0.0, 1.0]. 0.0 = worst, 1.0 = best.
    fn score(&self, point: &QueryPoint, ctx: &EqsContext<'_>) -> f32;
}
```

**DistanceTest:** Score is a function of distance from `to` (entity position or fixed world point) to `point.position`. `DistanceScoringEquation` enum: `Linear { min_dist, max_dist }`, `InverseLinear { ... }`, `Constant(f32)`. Linear scoring maps `[min_dist, max_dist]` onto `[0, 1]` (or `[1, 0]` for InverseLinear).

**LineOfSightTest:** Casts a ray from the `from` target position to `point.position` via `physics.cast_ray()`. Score is `1.0` if no hit within `max_dist`, `0.0` otherwise. Binary test.

**CoverTest:** Casts a ray from `threat_position` to `point.position`. Score is `1.0` if the ray hits a solid surface before reaching the point (point is in cover). Also checks that the point is reachable from querier (not trapped behind wall). Internally calls `TerrainVolume::sdf_value()` at midpoint of the ray: negative SDF = inside geometry = cover.

**SpectralBandTest** — unique to Ochroma:

```rust
pub struct SpectralBandTest {
    pub band: usize,
    pub min:  f32,
    pub max:  f32,
}

impl QueryTest for SpectralBandTest {
    fn score(&self, point: &QueryPoint, ctx: &EqsContext<'_>) -> f32 {
        let irradiance = ctx.gi_probes.sample_spectral(point.position);
        let band_val = irradiance[self.band];
        // Score 1.0 if in [min, max], falls off linearly outside
        linear_remap_clamp(band_val, self.min, self.max)
    }
}
```

`SpectralBandTest` queries `ProbeGrid::sample_spectral()` — the trilinearly-interpolated GI probe value at `point.position`. This lets AI agents find positions with specific spectral lighting conditions: seek warm cover (band 5–7 GI > 0.5), avoid UV-lit areas (band 0 GI > 0.3), cluster in shadow (all bands GI < 0.1). No Unreal equivalent exists; Unreal agents have no access to lighting data at query time.

### Asynchronous Evaluation

```rust
impl EnvQuery {
    pub fn run_async<'w>(
        self,
        ctx: EqsContext<'w>,
    ) -> impl Future<Output = Option<QueryPoint>> {
        async move {
            let points = self.generator.generate(&ctx);
            let scored: Vec<QueryPoint> = points
                .into_par_iter()    // rayon parallel iterator
                .map(|mut p| {
                    let scores: Vec<f32> =
                        self.tests.iter().map(|t| t.score(&p, &ctx)).collect();
                    p.score = match self.scoring_mode {
                        ScoringMode::Max     => scores.iter().cloned().fold(0.0_f32, f32::max),
                        ScoringMode::Average => scores.iter().sum::<f32>() / scores.len() as f32,
                        ScoringMode::Product => scores.iter().product(),
                    };
                    p
                })
                .collect();
            scored.into_iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        }
    }
}
```

The EQS system uses `rayon::par_iter` over the points (test evaluation is embarrassingly parallel since each point is independent). The result is a `Future` polled by the ECS event loop; the calling behavior tree leaf stores the query handle in the blackboard and returns `Running` until the result arrives next frame.

---

## 9.4 AI Perception System

### Components

```rust
// vox_core/src/ai/perception.rs

pub struct PerceptionComponent {
    pub sight:    Option<SightConfig>,
    pub hearing:  Option<HearingConfig>,
    pub spectral: Option<SpectralConfig>,
    pub stimuli:  Vec<PerceivedStimulus>,   // updated by PerceptionSystem each frame
}

pub struct SightConfig    { pub range: f32, pub fov_deg: f32, pub max_age_secs: f32 }
pub struct HearingConfig  { pub range: f32, pub min_loudness: f32 }

pub struct SpectralConfig {
    pub band:                usize,   // which spectral band to sense
    pub detection_threshold: f32,    // minimum band value to detect
    pub range:               f32,
}

pub struct PerceivedStimulus {
    pub source_entity:  EntityId,
    pub position:       glam::Vec3,
    pub stimulus_type:  StimulusType,
    pub confidence:     f32,  // 0.0–1.0; decreases with distance/age
    pub age:            f32,  // seconds since last confirmed observation
}

pub enum StimulusType {
    Sight,
    Sound { loudness: f32 },
    SpectralEmission { band: usize, intensity: f32 },
    Vibration { magnitude: f32 },
}
```

### PerceptionSystem Update

`PerceptionSystem` runs as a registered `System` in `SystemScheduler`. It reads `PerceptionComponent`, `Position`, `Velocity`, `Team`, `SplatAssembly`. It writes `PerceptionComponent.stimuli` only (no other writes), making it fully parallelizable with most other systems.

**Sight pipeline per agent:**

1. Gather all entities within `sight.range` via the spatial index (`kdtree` crate, `KdTree<f32, EntityId, [f32; 3]>`).
2. For each candidate: compute angle from agent forward direction to direction-to-target. If angle > `fov_deg / 2`, skip.
3. Cast a ray from agent eye position (entity position + `[0, 1.8, 0]`) to target center via `PhysicsWorld::cast_ray_filter()` (filter excludes the querying entity). If ray hits terrain or other geometry before reaching target distance, mark occluded.
4. Unoccluded candidates become `StimulusType::Sight` stimuli with `confidence = 1.0 - (dist / range)`.

**Hearing pipeline:**

Noise-generating events (`FootstepEvent { position, loudness }`, `ImpactEvent { position, loudness }` from `vox_audio::synthesize_impact`) are `EventWriter`-pushed into `EventQueue<HearingStimulus>`. `PerceptionSystem` reads the previous frame's `HearingStimulus` events and for each perceiving agent, finds stimuli within `hearing.range` where `loudness > min_loudness`.

**Spectral perception** — unique to Ochroma:

```rust
if let Some(spectral_cfg) = &perceiver.spectral {
    for candidate_entity in nearby {
        if let Some(assembly) = world.get_component::<SplatAssembly>(candidate_entity) {
            let intensity = assembly.mean_spectral[spectral_cfg.band];
            if intensity > spectral_cfg.detection_threshold {
                let dist = /* distance */;
                if dist < spectral_cfg.range {
                    stimuli.push(PerceivedStimulus {
                        source_entity: candidate_entity,
                        position: candidate_pos,
                        stimulus_type: StimulusType::SpectralEmission {
                            band: spectral_cfg.band,
                            intensity,
                        },
                        confidence: (1.0 - dist / spectral_cfg.range) * intensity,
                        age: 0.0,
                    });
                }
            }
        }
    }
}
```

Spectral perception bypasses both frustum culling and line-of-sight — it models perception mechanisms (thermal sensing, UV sensitivity) that can detect through certain materials. `SplatAssembly::mean_spectral` is the per-frame average of all splats in the assembly across each band, computed by `SpectralAssemblySystem` and cached as a component field.

### Stimulus Aging and Decay

At the start of each `PerceptionSystem::update()`, all existing stimuli have their `age += dt`. Stimuli with `age > max_age_secs` are removed. `confidence` decays linearly from its initial value to `0.0` over `max_age_secs`. Behavior trees reading `confidence` can distinguish "currently seeing" from "last seen 3 seconds ago at that position."

### Team System

```rust
pub struct Team { pub id: u8, pub enemy_teams: Vec<u8> }
```

`PerceptionSystem` only generates stimuli where the source entity belongs to a team listed in `perceiver.team.enemy_teams`. Friendly agents are invisible to the perception system (they don't generate threat stimuli). `TeamRegistry` in `Resources` maps `TeamId → Team`.

---

## 9.5 NavMesh & Pathfinding Extensions

### ORCA Crowd Simulation

The existing A*-over-`NavMesh` handles single-agent pathfinding. For groups of agents navigating simultaneously, per-agent A* paths produce collisions. `CrowdManager` solves this with ORCA (Optimal Reciprocal Collision Avoidance, van den Berg et al. 2011).

```rust
// vox_core/src/ai/crowd.rs

pub struct OrcaAgent {
    pub entity:              EntityId,
    pub position:            glam::Vec2,   // XZ plane
    pub velocity:            glam::Vec2,
    pub preferred_velocity:  glam::Vec2,   // from NavMesh path direction
    pub radius:              f32,
    pub max_speed:           f32,
    pub time_horizon:        f32,          // seconds to look ahead for collisions (default: 2.0)
    pub time_horizon_obst:   f32,          // seconds for static obstacles (default: 0.5)
}

pub struct OrcaSolver {
    pub agents: Vec<OrcaAgent>,
    spatial_index: KdTree<f32, usize, [f32; 2]>,  // kdtree crate
}

impl OrcaSolver {
    pub fn step(&mut self, dt: f32) {
        // 1. Rebuild spatial index (agents move each frame)
        self.rebuild_spatial();

        // 2. For each agent, find neighbors within max_speed * time_horizon + 2*radius
        // 3. For each neighbor pair, compute ORCA half-plane constraint in velocity space
        //    (van den Berg 2011, eq. 6–11)
        // 4. Solve 2D linear program: find velocity closest to preferred_velocity
        //    that satisfies all half-plane constraints
        //    LP solver: randomized incremental (O(n) expected) from RVO2 paper
        // 5. Apply solved velocity to agent position
        for agent in &mut self.agents {
            agent.position += agent.velocity * dt;
        }
    }
}
```

`CrowdManager` bridges ORCA back to the ECS: each frame it reads `Position`, `CharacterController`, and `NavPath` components from all agents; converts to `OrcaAgent` structs; runs `OrcaSolver::step()`; writes back the solved velocities to `CharacterController.desired_velocity`. The `rapier3d` character controller then applies the velocity with physics-correct sliding and stepping.

### Dynamic Obstacles

`DynamicObstacle { body_handle: RigidBodyHandle, affected_radius: f32, last_pos: Vec3 }` is a component on physics bodies that block navigation. `DynamicObstacleSystem` runs after `PhysicsSystem`:

1. For each `DynamicObstacle`, read current body position from `PhysicsWorld::body_position(handle)`.
2. If `(current_pos - last_pos).length() > 0.1`, call `navmesh.invalidate_region(current_pos, affected_radius)` and `navmesh.invalidate_region(last_pos, affected_radius)`.
3. Call `navmesh_bridge::extract_region(terrain_volume, invalidated_aabb)` to re-extract NavMesh in affected region.
4. Call `navmesh.merge(new_region)`.
5. Update `last_pos`.

This uses the already-implemented `invalidate_region` and `merge` APIs from Domain 6.

### Path Smoothing — Funnel Algorithm

`NavMesh::find_path()` returns a sequence of NavMesh polygon centroids. Traversing these directly produces jerky zigzag movement. The funnel (string-pull) algorithm (Lee and Preparata 1984; implemented in `navmesh/funnel.rs`) produces a minimal set of straight-line waypoints through the navigation corridor:

```rust
pub fn funnel_smooth(raw_path: &[NodeId], navmesh: &NavMesh) -> Vec<glam::Vec3> {
    // 1. Extract portal edges: for each consecutive node pair, find the shared edge
    //    (the portal between polygons)
    // 2. Run the funnel: maintain a left and right "funnel" vertex
    //    slide the funnel apex forward when a portal narrows the funnel to a point
    // 3. Each apex advance produces a waypoint in the smoothed path
    // Output: significantly fewer waypoints, straight where possible, curved at corners
}
```

### Jump Links

```rust
pub struct NavLink {
    pub from_node:    NodeId,
    pub to_node:      NodeId,
    pub link_type:    NavLinkType,
    pub animation_tag: String,   // e.g. "jump_high", "ladder_climb", "vault"
    pub traversal_cost: f32,     // extra A* cost for this link
}

pub enum NavLinkType { JumpLink, LadderLink, VaultLink, TeleportLink }
```

`NavLink` entries are stored in `NavMesh::links: Vec<NavLink>`. The A* implementation in `navmesh/pathfind.rs` treats links as directed edges with `traversal_cost` added to the path cost. When the path includes a `NavLink` edge, `MoveTo::tick()` detects the link traversal and fires an `AnimationEvent { entity, tag: link.animation_tag }` to the animation system, then sets `CharacterController` to kinematic mode for the duration of the jump arc.

### Area Costs

```rust
pub struct NavArea { pub id: u8, pub traversal_cost: f32, pub name: String }
```

NavMesh nodes carry an `area_id: u8`. `NavMesh::find_path()` weights edge costs by `NavArea::traversal_cost`: `water (cost=5.0)`, `road (cost=0.7)`, `default (cost=1.0)`. The A* heuristic (straight-line Euclidean) remains admissible because `traversal_cost >= 0` and the Euclidean heuristic never overestimates.

---

## 9.6 Gameplay Framework

### GameMode Trait

```rust
// vox_core/src/gameplay/game_mode.rs

pub enum WinResult { TeamWin(TeamId), Draw, Continue }

pub trait GameMode: Send + Sync {
    fn on_player_spawn(&mut self, player: EntityId, world: &mut World, res: &mut Resources);
    fn on_player_death(&mut self, player: EntityId, world: &mut World, res: &mut Resources);
    fn check_win_condition(&self, world: &World, res: &Resources) -> WinResult;
    fn on_tick(&mut self, dt: f32, world: &mut World, res: &mut Resources);
    fn name(&self) -> &'static str;
}
```

**DeathmatchMode:** tracks kill counts per player via `EventReader<DeathEvent>`; `check_win_condition` returns `TeamWin` when any player's kill count reaches `kill_limit`.

**CoopMode:** all players share a health pool against waves of AI agents; win when all waves cleared.

**SandboxMode:** no win condition; `on_player_death` respawns immediately with full health.

**SpectralHuntMode** — unique to Ochroma: players must `SpectralScanner`-scan world objects and find all 8 dominant spectral signatures (one per band). `check_win_condition` queries `AchievementSystem::progress(AchievementId::DiscoverAllBands)` — returns `TeamWin` when progress reaches `1.0`.

### PlayerState

```rust
pub struct PlayerState {
    pub entity_id:        EntityId,
    pub health:           f32,
    pub spectral_affinity: [f32; 8],  // per-band sensitivity (0.0–1.0)
    pub score:            u32,
    pub team:             TeamId,
    pub kills:            u32,
    pub deaths:           u32,
}
```

`spectral_affinity` is the Ochroma-unique player attribute. It determines:
- **AI detection radius:** agents using `SpectralConfig` perceive this player at `range * affinity[band]` — a player with high band-0 affinity glows in the UV and is detectable from further away.
- **Spectral resonance footsteps:** `vox_audio::synthesize_impact` receives `affinity` as a modulation vector — footsteps have richer spectral content for high-affinity players, audible to `HearingConfig` agents at greater range.
- Affinities are initialized randomly on spawn (values drawn from `U[0.1, 0.9]`) and shown in a "spectral character sheet" UI panel.

### SaveGame

```rust
// vox_core/src/gameplay/save_game.rs

pub struct SaveGame {
    pub world_snapshot:    WorldSnapshot,
    pub player_states:     Vec<PlayerState>,
    pub cell_dirty_flags:  Vec<(CellCoord, bool)>,   // which terrain cells were modified
    pub timestamp:         u64,   // Unix seconds
    pub game_mode_state:   Vec<u8>,  // bincode-encoded Box<dyn GameMode>
}

pub struct WorldSnapshot {
    pub entities: Vec<EntitySnapshot>,
}

pub struct EntitySnapshot {
    pub id:         EntityId,
    pub components: Vec<ComponentBlob>,   // (TypeId, bincode bytes)
}
```

Save is triggered by `SaveGame::write(path)` which serializes to binary via `bincode`. Each component type that participates in saving implements `Saveable: serde::Serialize + serde::Deserialize`. `TerrainVolume` cells are saved per-cell: only cells with `dirty = true` are written; clean cells are reconstructed from the procedural generator on load.

### Input Action Mapping

```rust
// vox_core/src/gameplay/input_map.rs

pub enum ActionId {
    Move, Jump, Fire, Interact, Crouch, Sprint,
    CycleSpectralView,   // step through spectral viewport band overlays
    ActivateScanner,     // hold to show SpectralScanner UI
    OpenGameMode,
}

pub struct InputBinding {
    pub key:      winit::event::VirtualKeyCode,
    pub modifier: Option<winit::event::ModifiersState>,
}

pub struct InputActionMap {
    pub bindings: HashMap<ActionId, Vec<InputBinding>>,
}
```

`InputActionMap` is stored in `Resources`. `InputSystem` reads `winit` events, resolves them against the map, and pushes `ActionEvent { action: ActionId, pressed: bool }` events each frame. Rhai scripts can query `input.is_action_pressed("Fire")` via a bound `RhaiInputModule`.

### Achievements

```rust
// vox_core/src/gameplay/achievements.rs

pub struct Achievement {
    pub id:          AchievementId,
    pub name:        String,
    pub description: String,
    pub target:      f32,   // progress value at which achievement unlocks (usually 1.0)
}

pub struct AchievementSystem {
    pub achievements: Vec<Achievement>,
    pub progress:     HashMap<AchievementId, f32>,
    pub unlocked:     HashSet<AchievementId>,
}

impl AchievementSystem {
    pub fn increment(&mut self, id: AchievementId, delta: f32) { ... }
    pub fn set_progress(&mut self, id: AchievementId, val: f32) { ... }
}
```

`SpectralAchievement::DiscoverAllBands` tracks whether the player has scanned objects with each of the 8 distinct dominant spectral bands as their primary emission. `AchievementSystem::increment(DiscoverAllBands, 1.0 / 8.0)` is called each time a new dominant band is discovered via `SpectralScanner`. The achievement unlocks at `progress = 1.0` (all 8 bands found).

---

## File Map

```
crates/
  vox_core/
    src/
      ecs/
        mod.rs
        archetype.rs        — ComponentColumn, Archetype
        world.rs            — World, EntityId, ArchetypeLocation
        query.rs            — Query, QueryFilter, With, Without
        system.rs           — System trait, SystemMeta, SystemScheduler
        resources.rs        — Resources, ResourceCell, ResourceRef
        events.rs           — EventQueue, EventWriter, EventReader
      ai/
        mod.rs
        behavior_tree.rs    — BehaviorNode, BehaviorStatus, BehaviorContext
        blackboard.rs       — Blackboard, BlackboardValue
        nodes/
          composite.rs      — Sequence, Selector, Parallel
          decorator.rs      — Inverter, Repeater, Cooldown, ForceSuccess, ForceFailure
          leaf.rs           — MoveTo, LookAt, PlayAnimation, Wait, SetBlackboard,
                              FireRayCast, SpectralCheck
        eqs.rs              — EnvQuery, QueryGenerator, QueryTest, QueryPoint, EqsContext
        generators.rs       — GridGenerator, NavmeshGenerator, PathfindableGenerator
        tests.rs            — DistanceTest, LineOfSightTest, CoverTest,
                              SpectralBandTest, NavMeshReachabilityTest
        perception.rs       — PerceptionComponent, PerceivedStimulus, PerceptionSystem,
                              SightConfig, HearingConfig, SpectralConfig
        crowd.rs            — CrowdManager, OrcaAgent, OrcaSolver
        team.rs             — Team, TeamRegistry
      gameplay/
        mod.rs
        game_mode.rs        — GameMode trait, WinResult
        player_state.rs     — PlayerState, TeamId
        save_game.rs        — SaveGame, WorldSnapshot, EntitySnapshot, ComponentBlob
        input_map.rs        — InputActionMap, InputBinding, ActionId
        achievements.rs     — AchievementSystem, Achievement, AchievementId

  vox_app/
    src/
      modes/
        deathmatch.rs
        coop.rs
        sandbox.rs
        spectral_hunt.rs
      achievements/
        spectral_achievements.rs
      editor/
        bt_editor.rs        — BehaviorTreeEditor (egui panel)
        eqs_debug.rs        — EQS debug visualizer (scored points as colored splats)
        perception_debug.rs — Perception cone/range visualizer

  vox_core/
    src/
      navmesh/
        funnel.rs           — funnel_smooth(), portal edge extraction
        jump_links.rs       — NavLink, NavLinkType
        area_costs.rs       — NavArea
        crowd_bridge.rs     — CrowdManager ↔ ECS bridge
```

---

## Milestones

### M9.1 — Archetype ECS Core (2 days)
- `ComponentColumn`, `Archetype`, `World`, `EntityId` generation/recycling
- `EntityBuilder` fluent API; `get_component`, `get_component_mut`
- `Query<(A,)>` single-component iteration
- Unit tests: spawn 100K entities, query all, verify correct components returned

### M9.2 — System Scheduler (1 day)
- `System` trait, `SystemMeta`, `SystemScheduler::build()`
- Parallel group computation (write-set intersection)
- `rayon::scope` dispatch
- Integration test: 4 systems with non-overlapping writes run in parallel; verify no data races under MIRI

### M9.3 — ECS Completion (1 day)
- Multi-component `Query<(A, B, C)>` via macro
- `QueryFilter<With<A>, Without<B>>`
- `Resources` with `RwLock`-guarded cells
- `EventQueue<T>` double-buffer; `EventWriter`/`EventReader`
- EngineRuntime migration wrapper

### M9.4 — Behavior Tree Runtime (2 days)
- All composite and decorator nodes
- All leaf nodes including `MoveTo` (NavMesh + CharacterController integration) and `SpectralCheck`
- `Blackboard` with all value types
- RON serialization round-trip tests for all node types

### M9.5 — BehaviorTree Editor (1 day)
- egui panel: node graph, drag/drop children, property inspector
- Hot-reload on `.ron` file change via `vox_script` file watcher

### M9.6 — EQS (1.5 days)
- All generators and tests including `SpectralBandTest` with `ProbeGrid` integration
- `rayon::par_iter` async evaluation
- Performance test: 100-point query, 4 tests, completes in < 2 ms

### M9.7 — AI Perception (1.5 days)
- `PerceptionComponent` with sight, hearing, spectral configs
- `PerceptionSystem` full pipeline (frustum, raycast, hearing events, spectral scan)
- Stimulus aging and decay
- `TeamRegistry` team-based filtering
- Integration test: agent with UV spectral perception detects emissive splat assembly through fog

### M9.8 — Crowd + NavMesh Extensions (2 days)
- `OrcaSolver` ORCA implementation; integration test: 50 agents converge without clipping
- Dynamic obstacle NavMesh invalidation bridge
- `funnel_smooth()` path smoothing
- `NavLink` jump links; A* traversal + animation event dispatch
- `NavArea` area costs

### M9.9 — Gameplay Framework (1.5 days)
- `GameMode` trait and all 4 implementations
- `PlayerState` with `spectral_affinity` and its effects on AI perception radius and audio
- `SaveGame` serialization (bincode) and deserialization
- `InputActionMap` with winit binding
- `AchievementSystem` with `DiscoverAllBands`

**Total estimated effort: ~13.5 developer-days**

---

## Acceptance Criteria

1. **ECS correctness:** spawn 1M entities with 5 components each; query all with `Query<(A, B, C)>`; verify zero incorrect results under `cargo test --release`.
2. **ECS performance:** 10K entities, 8 systems (4 parallel pairs), full frame tick completes in < 2 ms on an 8-core host.
3. **Behavior tree determinism:** given the same blackboard initial state and same dt sequence, a tree produces identical `BehaviorStatus` sequences across 1000 runs (no hidden mutable state in nodes).
4. **BehaviorTree RON round-trip:** serialize a tree with all node types to RON and deserialize; deep-equality check passes.
5. **EQS performance:** `NavmeshGenerator { radius: 20.0 }` + `DistanceTest` + `LineOfSightTest` + `SpectralBandTest` + `CoverTest` over a 100-node NavMesh completes in < 2 ms on rayon threadpool (8 threads).
6. **ORCA correctness:** 50 agents approaching the same point from random directions; after 5 simulated seconds, no two agents overlap (pairwise distances > `r_a + r_b`).
7. **Spectral perception:** agent with `SpectralConfig { band: 0, detection_threshold: 0.5, range: 30.0 }` detects a `SplatAssembly` with `mean_spectral[0] = 0.8` at 25 m range; does not detect one with `mean_spectral[0] = 0.3`.
8. **SaveGame round-trip:** spawn 500 entities with varied components, save, reload, verify all components match original values; verify `cell_dirty_flags` only contains cells that were carved.
9. **Hot-reload:** edit a `.ron` behavior tree file while the engine is running; within one file-watch poll cycle (< 1 s), all agents using that tree switch to the new tree without crashing.
10. **SpectralHunt win condition:** `SpectralHuntMode::check_win_condition` returns `Continue` until all 8 band achievements are incremented; returns `TeamWin` immediately after the 8th.

---

## Effort Summary

| Milestone | Scope | Days |
|-----------|-------|------|
| M9.1 | Archetype ECS core | 2.0 |
| M9.2 | System scheduler + parallelism | 1.0 |
| M9.3 | ECS completion + migration | 1.0 |
| M9.4 | Behavior tree runtime | 2.0 |
| M9.5 | BehaviorTree editor (egui) | 1.0 |
| M9.6 | Environment Query System | 1.5 |
| M9.7 | AI Perception | 1.5 |
| M9.8 | Crowd + NavMesh extensions | 2.0 |
| M9.9 | Gameplay framework | 1.5 |
| **Total** | | **13.5 days** |

Risk factors: `SpectralBandTest` depends on `ProbeGrid::sample_spectral()` from Domain 7 being stable; if that API changes, EQS tests need updates. ORCA correctness is non-trivial — the 2D LP solver must handle degenerate cases (agent surrounded on all sides). Budget 0.5 days for ORCA edge-case debugging.
