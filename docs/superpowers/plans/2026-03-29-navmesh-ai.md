# NavMesh + AI FSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the `navmesh` crate (pure Rust, no Recast FFI) to generate walkable surfaces from the terrain SDF, and add an `AiFsm` trait + `PatrolAgent` component that uses A* pathfinding to move entities via `TransformComponent`.

**Architecture:** `vox_terrain::navmesh` samples `TerrainVolume` to find walkable surface points (SDF solid-to-air transitions on the XZ grid), triangulates them, and builds a `navmesh::NavMesh`. The navmesh is stored as a `NavMeshResource` bevy_ecs Resource. A `PatrolAgent` ECS component holds waypoints and an `AiState` FSM. A `patrol_system` queries the navmesh for A* paths and drives `TransformComponent` along them each frame. A `NavMeshPlugin` wires startup generation + update system.

**Tech Stack:** `navmesh = "0.9"`, `bevy_ecs = "0.16"`, `bevy_app = "0.16"`, `vox_core::ecs::TransformComponent`, `vox_terrain::volume::TerrainVolume`.

---

## Key File Paths (read before editing)

- `crates/vox_terrain/Cargo.toml` — add `navmesh` dep
- `crates/vox_terrain/src/lib.rs` — add `pub mod navmesh;`
- `crates/vox_terrain/src/volume.rs` — `TerrainVolume` API: `get()`, `voxel_to_world()`, `is_surface()`, `sample_world()`
- `crates/vox_core/src/ecs.rs` — `TransformComponent { position: Vec3, rotation: Quat, scale: Vec3 }`
- `crates/vox_core/src/engine_runtime.rs` — `FrameTime { dt, total, frame }`, `RenderBuffer`
- `crates/vox_app/Cargo.toml` — game layer deps
- `crates/vox_app/src/bin/engine_runner.rs` — where plugins are wired

## File Structure

**Create:**
- `crates/vox_terrain/src/navmesh.rs` — navmesh generation from `TerrainVolume`
- `crates/vox_app/src/ai_fsm.rs` — `AiState`, `PatrolAgent` component, `patrol_system`, `NavMeshPlugin`

**Modify:**
- `crates/vox_terrain/Cargo.toml` — add `navmesh = "0.9"`, `bevy_app` deps
- `crates/vox_terrain/src/lib.rs` — add `pub mod navmesh;`
- `crates/vox_app/Cargo.toml` — (already has `vox_terrain` dep)
- `crates/vox_app/src/lib.rs` — add `pub mod ai_fsm;`

---

### Task 1: NavMesh generation from TerrainVolume

**Files:**
- Modify: `crates/vox_terrain/Cargo.toml`
- Modify: `crates/vox_terrain/src/lib.rs`
- Create: `crates/vox_terrain/src/navmesh.rs`

- [ ] **Step 1: Add dependencies to vox_terrain/Cargo.toml**

Add `navmesh` and `bevy_app` to dependencies:

```toml
[dependencies]
vox_core = { path = "../vox_core" }
glam = { workspace = true }
serde = { workspace = true }
half = { workspace = true }
rand = "0.9"
bevy_ecs = { workspace = true }
bevy_app = { workspace = true }
navmesh = "0.9"
```

- [ ] **Step 2: Add module to vox_terrain/src/lib.rs**

Add `pub mod navmesh;` to the module list.

- [ ] **Step 3: Create `crates/vox_terrain/src/navmesh.rs`**

```rust
//! NavMesh generation from `TerrainVolume` SDF data.
//!
//! Samples the terrain volume's XZ grid to find walkable surface points
//! (where the SDF transitions from solid to air along the Y axis), then
//! builds a `navmesh::NavMesh` polygon mesh for A* pathfinding.

use glam::Vec3;
use navmesh::NavMesh;

use crate::volume::TerrainVolume;

/// A walkable surface point with world-space position and surface normal Y.
#[derive(Debug, Clone, Copy)]
pub struct SurfacePoint {
    pub position: Vec3,
    pub normal_y: f32,
}

/// Walk the XZ grid of the terrain volume at the given step size (in world
/// metres) and find the first Y where the SDF transitions from solid (<=0)
/// to air (>0). Returns the world-space surface points.
///
/// `step` controls the sampling density — smaller values yield more points.
/// `max_slope_angle` filters out points whose surface is steeper than this
/// angle in degrees (default: 45 degrees for walkable surfaces).
pub fn sample_walkable_surface(
    vol: &TerrainVolume,
    step: f32,
    max_slope_angle: f32,
) -> Vec<SurfacePoint> {
    let mut points = Vec::new();
    let cos_max = max_slope_angle.to_radians().cos();

    let world_min_x = vol.origin[0];
    let world_min_z = vol.origin[2];
    let world_max_x = vol.origin[0] + vol.size_x as f32 * vol.voxel_size;
    let world_max_z = vol.origin[2] + vol.size_z as f32 * vol.voxel_size;
    let world_max_y = vol.origin[1] + vol.size_y as f32 * vol.voxel_size;
    let world_min_y = vol.origin[1];

    let mut wx = world_min_x;
    while wx < world_max_x {
        let mut wz = world_min_z;
        while wz < world_max_z {
            // Walk top-down to find the highest surface point at this XZ
            let mut wy = world_max_y;
            let y_step = vol.voxel_size;
            let mut prev_sdf = vol.sample_world(wx, wy, wz);

            while wy > world_min_y {
                wy -= y_step;
                let sdf = vol.sample_world(wx, wy, wz);

                // Transition: previous was air (>0), current is solid (<=0)
                if prev_sdf > 0.0 && sdf <= 0.0 {
                    // Interpolate to find the exact surface Y
                    let t = prev_sdf / (prev_sdf - sdf);
                    let surface_y = (wy + y_step) - t * y_step;

                    // Check slope: compute SDF gradient at this point
                    let (vx, vy, vz) = vol.world_to_voxel(wx, surface_y, wz);
                    let grad = vol.gradient(vx, vy, vz);
                    let normal_y = grad[1]; // Y component of surface normal

                    // Walkable if normal points mostly upward
                    if normal_y >= cos_max {
                        points.push(SurfacePoint {
                            position: Vec3::new(wx, surface_y, wz),
                            normal_y,
                        });
                    }
                    break; // Only take the highest surface at this XZ
                }

                prev_sdf = sdf;
            }

            wz += step;
        }
        wx += step;
    }

    points
}

/// Configuration for navmesh generation.
#[derive(Debug, Clone)]
pub struct NavMeshConfig {
    /// Sampling step in world metres (smaller = denser mesh).
    pub sample_step: f32,
    /// Agent radius in metres — used for navmesh shrinkage.
    pub agent_radius: f32,
    /// Maximum slope angle in degrees for walkable surfaces.
    pub max_slope_angle: f32,
}

impl Default for NavMeshConfig {
    fn default() -> Self {
        Self {
            sample_step: 2.0,
            agent_radius: 0.5,
            max_slope_angle: 45.0,
        }
    }
}

/// Build a `navmesh::NavMesh` from the terrain volume.
///
/// 1. Samples walkable surface points from the SDF.
/// 2. Triangulates the points using a simple grid-based approach.
/// 3. Builds the NavMesh polygon structure.
///
/// Returns `None` if the terrain has no walkable surface.
pub fn build_navmesh(vol: &TerrainVolume, config: &NavMeshConfig) -> Option<NavMesh> {
    let points = sample_walkable_surface(vol, config.sample_step, config.max_slope_angle);
    if points.len() < 3 {
        return None;
    }

    // Build a grid-indexed point map for triangulation.
    // We index by (grid_x, grid_z) where grid coords = floor(world / step).
    let step = config.sample_step;
    let mut grid: std::collections::HashMap<(i32, i32), usize> = std::collections::HashMap::new();
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(points.len());

    for (idx, p) in points.iter().enumerate() {
        let gx = (p.position.x / step).floor() as i32;
        let gz = (p.position.z / step).floor() as i32;
        grid.insert((gx, gz), idx);
        vertices.push([p.position.x, p.position.y, p.position.z]);
    }

    // Generate triangles by connecting adjacent grid cells.
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for (&(gx, gz), &idx) in &grid {
        // Try to form two triangles per grid quad:
        //   (gx,gz) -- (gx+1,gz)
        //       |  \       |
        //   (gx,gz+1) -- (gx+1,gz+1)
        let right = grid.get(&(gx + 1, gz)).copied();
        let below = grid.get(&(gx, gz + 1)).copied();
        let diag = grid.get(&(gx + 1, gz + 1)).copied();

        // Upper-left triangle
        if let (Some(r), Some(b)) = (right, below) {
            let dy1 = (vertices[idx][1] - vertices[r][1]).abs();
            let dy2 = (vertices[idx][1] - vertices[b][1]).abs();
            if dy1 < step * 2.0 && dy2 < step * 2.0 {
                triangles.push([idx as u32, r as u32, b as u32]);
            }
        }

        // Lower-right triangle
        if let (Some(r), Some(b), Some(d)) = (right, below, diag) {
            let dy1 = (vertices[r][1] - vertices[d][1]).abs();
            let dy2 = (vertices[b][1] - vertices[d][1]).abs();
            if dy1 < step * 2.0 && dy2 < step * 2.0 {
                triangles.push([r as u32, d as u32, b as u32]);
            }
        }
    }

    if triangles.is_empty() {
        return None;
    }

    // Flatten vertices for navmesh crate: expects Vec<Vec3> as [x, y, z] tuples.
    // The navmesh crate uses NavMesh::new(vertices, polygons) where polygons
    // are Vec<Vec<u32>> (each polygon = list of vertex indices).
    let nav_vertices: Vec<navmesh::NavVec3> = vertices
        .iter()
        .map(|v| navmesh::NavVec3::new(v[0], v[1], v[2]))
        .collect();

    let polygons: Vec<Vec<u32>> = triangles
        .iter()
        .map(|t| vec![t[0], t[1], t[2]])
        .collect();

    NavMesh::new(nav_vertices, polygons).ok()
}

/// Result of a navmesh path query.
#[derive(Debug, Clone)]
pub struct NavPath {
    /// Waypoints from start to end in world space.
    pub waypoints: Vec<Vec3>,
}

/// Query a path on the navmesh from `start` to `end`.
///
/// Returns `None` if no path exists (e.g., start/end not on the mesh).
pub fn find_path(mesh: &NavMesh, start: Vec3, end: Vec3) -> Option<NavPath> {
    let from = navmesh::NavVec3::new(start.x, start.y, start.z);
    let to = navmesh::NavVec3::new(end.x, end.y, end.z);

    // NavMesh::find_path returns Option<Vec<NavVec3>>
    let query = navmesh::NavPathMode::Accuracy;
    match mesh.find_path(from, to, query, query) {
        Some(path) => {
            let waypoints: Vec<Vec3> = path
                .iter()
                .map(|p| Vec3::new(p.x, p.y, p.z))
                .collect();
            if waypoints.is_empty() {
                None
            } else {
                Some(NavPath { waypoints })
            }
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::{sculpt, TerrainVolume};

    /// Create a simple flat terrain for testing.
    fn flat_terrain() -> TerrainVolume {
        let mut vol = TerrainVolume::new(32, 16, 32, 1.0);
        sculpt::add_ground_plane(&mut vol, 0.0, 0);
        vol
    }

    #[test]
    fn sample_walkable_surface_on_flat_terrain_returns_points() {
        let vol = flat_terrain();
        let points = sample_walkable_surface(&vol, 2.0, 45.0);
        assert!(
            !points.is_empty(),
            "Flat terrain should have walkable surface points, got 0"
        );
        // All points should be near Y=0 (the ground plane)
        for p in &points {
            assert!(
                (p.position.y).abs() < 2.0,
                "Surface point Y={} should be near 0.0",
                p.position.y
            );
        }
    }

    #[test]
    fn sample_walkable_surface_empty_volume_returns_empty() {
        let vol = TerrainVolume::new(8, 8, 8, 1.0); // all air
        let points = sample_walkable_surface(&vol, 1.0, 45.0);
        assert!(points.is_empty(), "All-air volume should have no walkable surface");
    }

    #[test]
    fn build_navmesh_on_flat_terrain_succeeds() {
        let vol = flat_terrain();
        let config = NavMeshConfig {
            sample_step: 4.0,
            agent_radius: 0.5,
            max_slope_angle: 45.0,
        };
        let mesh = build_navmesh(&vol, &config);
        assert!(mesh.is_some(), "Flat terrain should produce a valid navmesh");
    }

    #[test]
    fn build_navmesh_empty_volume_returns_none() {
        let vol = TerrainVolume::new(8, 8, 8, 1.0);
        let config = NavMeshConfig::default();
        let mesh = build_navmesh(&vol, &config);
        assert!(mesh.is_none(), "All-air volume should not produce a navmesh");
    }

    #[test]
    fn navmesh_config_default_is_reasonable() {
        let config = NavMeshConfig::default();
        assert!(config.sample_step > 0.0);
        assert!(config.agent_radius > 0.0);
        assert!(config.max_slope_angle > 0.0 && config.max_slope_angle <= 90.0);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_terrain -- navmesh 2>&1 | tail -20
```

Expected: all 5 tests pass.

**Commit:** `feat(terrain): NavMesh generation from TerrainVolume SDF — sample_walkable_surface + build_navmesh`

---

### Task 2: AiFsm trait + PatrolAgent component

**Files:**
- Modify: `crates/vox_app/src/lib.rs`
- Create: `crates/vox_app/src/ai_fsm.rs`

- [ ] **Step 1: Read `crates/vox_app/src/lib.rs` to find the module list**

- [ ] **Step 2: Add `pub mod ai_fsm;` to `crates/vox_app/src/lib.rs`**

- [ ] **Step 3: Create `crates/vox_app/src/ai_fsm.rs`**

```rust
//! AI Finite State Machine + Patrol Agent for Ochroma Engine.
//!
//! Provides `PatrolAgent` ECS component with an `AiState` FSM, and a
//! `patrol_system` that queries the `NavMeshResource` for A* paths and
//! drives entity `TransformComponent` along them.

use bevy_ecs::prelude::*;
use glam::Vec3;

use vox_core::ecs::TransformComponent;
use vox_core::engine_runtime::FrameTime;

// ---------------------------------------------------------------------------
// AI State Machine
// ---------------------------------------------------------------------------

/// Finite state machine states for AI agents.
#[derive(Debug, Clone, PartialEq)]
pub enum AiState {
    /// Standing still, waiting.
    Idle,
    /// Following patrol waypoints in order.
    Patrol { waypoint_idx: usize },
    /// Chasing a target position.
    Chase { target: Vec3 },
    /// Fleeing from a threat position.
    Flee { from: Vec3 },
}

impl Default for AiState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// PatrolAgent Component
// ---------------------------------------------------------------------------

/// An AI agent that patrols between waypoints using navmesh pathfinding.
///
/// Attach alongside `TransformComponent` to make an entity patrol.
#[derive(Component, Debug, Clone)]
pub struct PatrolAgent {
    /// World-space waypoints to visit in order.
    pub waypoints: Vec<Vec3>,
    /// Current FSM state.
    pub state: AiState,
    /// Movement speed in metres per second.
    pub speed: f32,
    /// Current computed path from navmesh (world-space points).
    pub path: Vec<Vec3>,
    /// Index into `path` — the next point to walk toward.
    pub path_idx: usize,
    /// Distance threshold to consider a waypoint "reached" (metres).
    pub reach_threshold: f32,
}

impl Default for PatrolAgent {
    fn default() -> Self {
        Self {
            waypoints: Vec::new(),
            state: AiState::Idle,
            speed: 3.0,
            path: Vec::new(),
            path_idx: 0,
            reach_threshold: 0.5,
        }
    }
}

impl PatrolAgent {
    /// Create a new patrol agent with the given waypoints and speed.
    pub fn new(waypoints: Vec<Vec3>, speed: f32) -> Self {
        let state = if waypoints.is_empty() {
            AiState::Idle
        } else {
            AiState::Patrol { waypoint_idx: 0 }
        };
        Self {
            waypoints,
            state,
            speed,
            path: Vec::new(),
            path_idx: 0,
            reach_threshold: 0.5,
        }
    }

    /// Advance to the next waypoint, wrapping around to the start.
    pub fn next_waypoint(&mut self) {
        if self.waypoints.is_empty() {
            self.state = AiState::Idle;
            return;
        }
        let next_idx = match &self.state {
            AiState::Patrol { waypoint_idx } => (*waypoint_idx + 1) % self.waypoints.len(),
            _ => 0,
        };
        self.state = AiState::Patrol { waypoint_idx: next_idx };
        self.path.clear();
        self.path_idx = 0;
    }

    /// Get the current target waypoint position, if patrolling.
    pub fn current_target(&self) -> Option<Vec3> {
        match &self.state {
            AiState::Patrol { waypoint_idx } => self.waypoints.get(*waypoint_idx).copied(),
            AiState::Chase { target } => Some(*target),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NavMesh Resource
// ---------------------------------------------------------------------------

/// Wraps a `navmesh::NavMesh` as a bevy_ecs Resource for system access.
#[derive(Resource)]
pub struct NavMeshResource {
    pub mesh: Option<navmesh::NavMesh>,
}

impl Default for NavMeshResource {
    fn default() -> Self {
        Self { mesh: None }
    }
}

// ---------------------------------------------------------------------------
// patrol_system
// ---------------------------------------------------------------------------

/// Drives `PatrolAgent` entities along navmesh paths each frame.
///
/// For each patrolling agent:
/// 1. If `path` is empty, query the navmesh for a path from current position
///    to the target waypoint.
/// 2. Advance along `path` by `speed * dt`, popping waypoints as reached.
/// 3. At destination, call `next_waypoint()` to cycle to the next one.
pub fn patrol_system(
    time: Res<FrameTime>,
    nav: Res<NavMeshResource>,
    mut query: Query<(&mut PatrolAgent, &mut TransformComponent)>,
) {
    let dt = time.dt;

    for (mut agent, mut transform) in query.iter_mut() {
        // Only process agents in Patrol state
        let target = match agent.current_target() {
            Some(t) => t,
            None => continue,
        };

        // If no path computed yet, query the navmesh
        if agent.path.is_empty() {
            if let Some(ref mesh) = nav.mesh {
                if let Some(nav_path) =
                    vox_terrain::navmesh::find_path(mesh, transform.position, target)
                {
                    agent.path = nav_path.waypoints;
                    agent.path_idx = 0;
                } else {
                    // No path found — skip to next waypoint
                    agent.next_waypoint();
                    continue;
                }
            } else {
                // No navmesh — use direct line to target as fallback
                agent.path = vec![target];
                agent.path_idx = 0;
            }
        }

        // Advance along the path
        if agent.path_idx < agent.path.len() {
            let next_point = agent.path[agent.path_idx];
            let to_next = next_point - transform.position;
            let dist = to_next.length();

            if dist < agent.reach_threshold {
                // Reached this path point, advance
                agent.path_idx += 1;
            } else {
                // Move toward the next point
                let direction = to_next / dist;
                let move_dist = (agent.speed * dt).min(dist);
                transform.position += direction * move_dist;

                // Face movement direction (Y-axis rotation only)
                if direction.x.abs() > 1e-6 || direction.z.abs() > 1e-6 {
                    let yaw = direction.z.atan2(direction.x);
                    transform.rotation =
                        glam::Quat::from_rotation_y(-yaw + std::f32::consts::FRAC_PI_2);
                }
            }
        }

        // Check if we've finished the entire path
        if agent.path_idx >= agent.path.len() {
            agent.next_waypoint();
        }
    }
}

// ---------------------------------------------------------------------------
// NavMeshPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that generates a navmesh from `TerrainVolume` on startup and
/// registers the `patrol_system` in `Update`.
///
/// Requires `TerrainVolume` to be inserted as a Resource before this plugin
/// builds. If no `TerrainVolume` exists, inserts an empty `NavMeshResource`.
pub struct NavMeshPlugin;

impl bevy_app::Plugin for NavMeshPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<NavMeshResource>();
        app.add_systems(bevy_app::Startup, build_navmesh_startup);
        app.add_systems(bevy_app::Update, patrol_system);
    }
}

/// Startup system: builds navmesh from `TerrainVolume` if present.
fn build_navmesh_startup(
    mut nav: ResMut<NavMeshResource>,
    terrain: Option<Res<vox_terrain::volume::TerrainVolume>>,
) {
    if let Some(vol) = terrain {
        let config = vox_terrain::navmesh::NavMeshConfig::default();
        nav.mesh = vox_terrain::navmesh::build_navmesh(&vol, &config);
        if nav.mesh.is_some() {
            log_info("NavMesh built from TerrainVolume");
        } else {
            log_info("NavMesh: no walkable surface found in TerrainVolume");
        }
    } else {
        log_info("NavMesh: no TerrainVolume resource — skipping navmesh generation");
    }
}

fn log_info(msg: &str) {
    #[cfg(debug_assertions)]
    eprintln!("[NavMesh] {}", msg);
    let _ = msg;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;
    use bevy_ecs::schedule::Schedule;
    use bevy_ecs::world::World;
    use glam::Vec3;

    #[test]
    fn patrol_agent_next_waypoint_advances_and_wraps() {
        let mut agent = PatrolAgent::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 10.0),
            ],
            3.0,
        );
        assert!(matches!(agent.state, AiState::Patrol { waypoint_idx: 0 }));

        agent.next_waypoint();
        assert!(matches!(agent.state, AiState::Patrol { waypoint_idx: 1 }));

        agent.next_waypoint();
        assert!(matches!(agent.state, AiState::Patrol { waypoint_idx: 2 }));

        // Should wrap around
        agent.next_waypoint();
        assert!(matches!(agent.state, AiState::Patrol { waypoint_idx: 0 }));
    }

    #[test]
    fn patrol_agent_empty_waypoints_stays_idle() {
        let mut agent = PatrolAgent::new(vec![], 3.0);
        assert!(matches!(agent.state, AiState::Idle));
        agent.next_waypoint();
        assert!(matches!(agent.state, AiState::Idle));
    }

    #[test]
    fn patrol_agent_current_target_returns_correct_waypoint() {
        let agent = PatrolAgent::new(
            vec![Vec3::new(5.0, 0.0, 5.0), Vec3::new(15.0, 0.0, 15.0)],
            2.0,
        );
        let target = agent.current_target().unwrap();
        assert!((target - Vec3::new(5.0, 0.0, 5.0)).length() < 1e-5);
    }

    #[test]
    fn patrol_system_moves_toward_waypoint() {
        let mut world = World::new();
        world.insert_resource(FrameTime { dt: 1.0, total: 0.0, frame: 0 });
        world.insert_resource(NavMeshResource::default()); // no mesh — direct path fallback

        let mut agent = PatrolAgent::new(
            vec![Vec3::new(10.0, 0.0, 0.0)],
            5.0,
        );
        // Start at origin
        world.spawn((
            agent,
            TransformComponent {
                position: Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(patrol_system);
        schedule.run(&mut world);

        // After 1 second at speed 5, should have moved 5 metres toward (10,0,0)
        let mut query = world.query::<&TransformComponent>();
        let transform = query.iter(&world).next().unwrap();
        assert!(
            transform.position.x > 4.0,
            "Agent should have moved toward target, x={}",
            transform.position.x
        );
    }

    #[test]
    fn navmesh_plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(bevy_app::ScheduleRunnerPlugin::default());
        app.add_plugins(NavMeshPlugin);
        // Run one update to trigger startup systems
        app.update();
    }

    #[test]
    fn navmesh_resource_default_has_no_mesh() {
        let res = NavMeshResource::default();
        assert!(res.mesh.is_none());
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_app -- ai_fsm 2>&1 | tail -20
```

Expected: all 6 tests pass.

**Commit:** `feat(ai): PatrolAgent ECS component + AiState FSM + patrol_system + NavMeshPlugin`

---

### Task 3: patrol_system drive + NavMesh A* path query

This is already included in the `ai_fsm.rs` file from Task 2. The `patrol_system` function:

1. Checks if the agent has a target waypoint
2. If `path` is empty, queries `NavMeshResource` via `vox_terrain::navmesh::find_path()`
3. Falls back to direct-line pathfinding if no navmesh exists
4. Advances `TransformComponent.position` along the path at `speed * dt`
5. Rotates the entity to face the movement direction
6. At path end, calls `next_waypoint()` to cycle

No additional file creation needed.

- [ ] **Step 1: Verify the `find_path` function in `vox_terrain::navmesh` compiles**

```bash
cargo check -p vox_terrain 2>&1 | tail -5
```

- [ ] **Step 2: Verify `patrol_system` compiles in vox_app**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

---

### Task 4: Wire into vox_app — demo patrol entity

**Files:**
- Modify: `crates/vox_app/src/bin/engine_runner.rs`

- [ ] **Step 1: Read `engine_runner.rs` to find the plugin setup section**

Look for where `app.add_plugins(...)` is called or where the `EngineRuntime` is built.

- [ ] **Step 2: Add NavMeshPlugin and spawn a demo PatrolAgent**

Add to the engine initialization section:

```rust
use vox_app::ai_fsm::{NavMeshPlugin, PatrolAgent};
```

In the setup code (where terrain is inserted), add after terrain setup:

```rust
// --- NavMesh + AI demo ---
// NavMeshPlugin builds the navmesh from TerrainVolume on startup
// and registers patrol_system in Update.
// (Uncomment when ready to test:)
// app.add_plugins(NavMeshPlugin);
//
// // Spawn a demo patrol agent with 4 waypoints
// world.spawn((
//     PatrolAgent::new(
//         vec![
//             Vec3::new(-10.0, 1.0, -10.0),
//             Vec3::new(10.0, 1.0, -10.0),
//             Vec3::new(10.0, 1.0, 10.0),
//             Vec3::new(-10.0, 1.0, 10.0),
//         ],
//         3.0,
//     ),
//     TransformComponent {
//         position: Vec3::new(-10.0, 1.0, -10.0),
//         rotation: Quat::IDENTITY,
//         scale: Vec3::ONE,
//     },
// ));
```

- [ ] **Step 3: Verify the full crate compiles**

```bash
cargo check -p vox_app 2>&1 | tail -5
```

**Commit:** `feat(ai): NavMesh from terrain SDF + PatrolAgent ECS component + patrol_system`

---

## Summary

| Task | File | What |
|------|------|------|
| 1 | `crates/vox_terrain/src/navmesh.rs` | NavMesh generation from SDF |
| 2 | `crates/vox_app/src/ai_fsm.rs` | AiState FSM + PatrolAgent component |
| 3 | (same as Task 2) | patrol_system + find_path integration |
| 4 | `crates/vox_app/src/bin/engine_runner.rs` | Wire NavMeshPlugin + demo entity |

**Final commit:** `feat(ai): NavMesh from terrain SDF + PatrolAgent ECS component + patrol_system`
