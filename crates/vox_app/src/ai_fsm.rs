//! AI FSM + PatrolAgent for Ochroma Engine.

use bevy_ecs::prelude::*;
use glam::Vec3;

use vox_core::ecs::TransformComponent;
use vox_core::engine_runtime::FrameTime;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum AiState {
    #[default]
    Idle,
    Patrol { waypoint_idx: usize },
    Chase { target: Vec3 },
    Flee { from: Vec3 },
}

#[derive(Component, Debug, Clone)]
pub struct PatrolAgent {
    pub waypoints: Vec<Vec3>,
    pub state: AiState,
    pub speed: f32,
    pub path: Vec<Vec3>,
    pub path_idx: usize,
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
    pub fn new(waypoints: Vec<Vec3>, speed: f32) -> Self {
        let state = if waypoints.is_empty() {
            AiState::Idle
        } else {
            AiState::Patrol { waypoint_idx: 0 }
        };
        Self { waypoints, state, speed, path: Vec::new(), path_idx: 0, reach_threshold: 0.5 }
    }

    pub fn next_waypoint(&mut self) {
        if self.waypoints.is_empty() { self.state = AiState::Idle; return; }
        let next_idx = match &self.state {
            AiState::Patrol { waypoint_idx } => (*waypoint_idx + 1) % self.waypoints.len(),
            _ => 0,
        };
        self.state = AiState::Patrol { waypoint_idx: next_idx };
        self.path.clear();
        self.path_idx = 0;
    }

    pub fn current_target(&self) -> Option<Vec3> {
        match &self.state {
            AiState::Patrol { waypoint_idx } => self.waypoints.get(*waypoint_idx).copied(),
            AiState::Chase { target } => Some(*target),
            _ => None,
        }
    }
}

#[derive(Resource, Default)]
pub struct NavMeshResource {
    pub mesh: Option<navmesh::NavMesh>,
}

pub fn patrol_system(
    time: Res<FrameTime>,
    nav: Res<NavMeshResource>,
    mut query: Query<(&mut PatrolAgent, &mut TransformComponent)>,
) {
    let dt = time.dt;
    for (mut agent, mut transform) in query.iter_mut() {
        let target = match agent.current_target() { Some(t) => t, None => continue };

        if agent.path.is_empty() {
            if let Some(ref mesh) = nav.mesh {
                if let Some(nav_path) = vox_terrain::navmesh::find_path(mesh, transform.position, target) {
                    agent.path = nav_path.waypoints;
                    agent.path_idx = 0;
                } else {
                    agent.next_waypoint();
                    continue;
                }
            } else {
                agent.path = vec![target];
                agent.path_idx = 0;
            }
        }

        if agent.path_idx < agent.path.len() {
            let next_point = agent.path[agent.path_idx];
            let to_next = next_point - transform.position;
            let dist = to_next.length();

            if dist < agent.reach_threshold {
                agent.path_idx += 1;
            } else {
                let direction = to_next / dist;
                let move_dist = (agent.speed * dt).min(dist);
                transform.position += direction * move_dist;

                if direction.x.abs() > 1e-6 || direction.z.abs() > 1e-6 {
                    let yaw = direction.z.atan2(direction.x);
                    transform.rotation = glam::Quat::from_rotation_y(-yaw + std::f32::consts::FRAC_PI_2);
                }
            }
        }

        if agent.path_idx >= agent.path.len() {
            agent.next_waypoint();
        }
    }
}

pub struct NavMeshPlugin;

impl bevy_app::Plugin for NavMeshPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<NavMeshResource>();
        app.add_systems(bevy_app::Startup, build_navmesh_startup);
        app.add_systems(bevy_app::Update, patrol_system);
    }
}

fn build_navmesh_startup(
    mut nav: ResMut<NavMeshResource>,
    terrain: Option<Res<vox_terrain::volume::TerrainVolume>>,
) {
    if let Some(vol) = terrain {
        let config = vox_terrain::navmesh::NavMeshConfig::default();
        nav.mesh = vox_terrain::navmesh::build_navmesh(&vol, &config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::schedule::Schedule;
    use bevy_ecs::world::World;

    #[test]
    fn patrol_agent_advances_waypoints() {
        let mut agent = PatrolAgent::new(
            vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 10.0)],
            3.0,
        );
        assert!(matches!(agent.state, AiState::Patrol { waypoint_idx: 0 }));
        agent.next_waypoint();
        assert!(matches!(agent.state, AiState::Patrol { waypoint_idx: 1 }));
        agent.next_waypoint();
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
    fn patrol_agent_current_target() {
        let agent = PatrolAgent::new(vec![Vec3::new(5.0, 0.0, 5.0)], 2.0);
        let target = agent.current_target().unwrap();
        assert!((target - Vec3::new(5.0, 0.0, 5.0)).length() < 1e-5);
    }

    #[test]
    fn patrol_system_moves_toward_waypoint() {
        let mut world = World::new();
        world.insert_resource(FrameTime { dt: 1.0, total: 0.0, frame: 0 });
        world.insert_resource(NavMeshResource::default());

        world.spawn((
            PatrolAgent::new(vec![Vec3::new(10.0, 0.0, 0.0)], 5.0),
            TransformComponent {
                position: Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(patrol_system);
        schedule.run(&mut world);

        let mut q = world.query::<&TransformComponent>();
        let t = q.iter(&world).next().unwrap();
        assert!(t.position.x > 4.0, "should have moved, x={}", t.position.x);
    }

    #[test]
    fn navmesh_resource_default_has_no_mesh() {
        let res = NavMeshResource::default();
        assert!(res.mesh.is_none());
    }
}
