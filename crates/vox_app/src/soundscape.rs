//! Generic ambient soundscape system.

use bevy_ecs::prelude::*;

// ── Data types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SoundLayer {
    pub name: String,
    pub clip_path: String,
    pub volume: f32,
    pub looping: bool,
    pub spatial: bool,
}

#[derive(Debug, Clone)]
pub struct Soundscape {
    pub layers: Vec<SoundLayer>,
    pub active: bool,
}

impl Soundscape {
    pub fn new() -> Self {
        Self { layers: Vec::new(), active: true }
    }

    pub fn add_layer(&mut self, layer: SoundLayer) {
        self.layers.push(layer);
    }

    pub fn remove_layer(&mut self, name: &str) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.name != name);
        self.layers.len() < before
    }

    pub fn outdoor_default() -> Self {
        Self {
            layers: vec![
                SoundLayer {
                    name: "wind".into(),
                    clip_path: "audio/ambient/wind_loop.ogg".into(),
                    volume: 0.3,
                    looping: true,
                    spatial: false,
                },
                SoundLayer {
                    name: "distant_traffic".into(),
                    clip_path: "audio/ambient/traffic_distant.ogg".into(),
                    volume: 0.15,
                    looping: true,
                    spatial: false,
                },
                SoundLayer {
                    name: "birds".into(),
                    clip_path: "audio/ambient/birds_morning.ogg".into(),
                    volume: 0.2,
                    looping: true,
                    spatial: true,
                },
            ],
            active: true,
        }
    }
}

impl Default for Soundscape {
    fn default() -> Self { Self::new() }
}

// ── ECS Integration ───────────────────────────────────────────────────────

#[derive(Resource)]
pub struct SoundscapeResource(pub Soundscape);

#[derive(Component, Debug)]
pub struct SoundscapeLayerMarker {
    pub layer_name: String,
}

pub fn soundscape_sync_system(
    mut commands: Commands,
    soundscape: Res<SoundscapeResource>,
    existing: Query<(Entity, &SoundscapeLayerMarker)>,
) {
    let active = soundscape.0.active;

    let existing_names: Vec<(Entity, String)> = existing
        .iter()
        .map(|(e, m)| (e, m.layer_name.clone()))
        .collect();

    for (entity, name) in &existing_names {
        if !soundscape.0.layers.iter().any(|l| l.name == *name) {
            commands.entity(*entity).despawn();
        }
    }

    for layer in &soundscape.0.layers {
        let already_spawned = existing_names.iter().any(|(_, n)| *n == layer.name);
        if !already_spawned {
            commands.spawn((
                SoundscapeLayerMarker { layer_name: layer.name.clone() },
                vox_core::ecs::AudioEmitterComponent {
                    clip_path: layer.clip_path.clone(),
                    volume: layer.volume,
                    looping: layer.looping,
                    playing: active,
                    spatial: layer.spatial,
                },
                vox_core::ecs::TransformComponent::default(),
            ));
        }
    }
}

pub fn soundscape_toggle_system(
    soundscape: Res<SoundscapeResource>,
    mut query: Query<&mut vox_core::ecs::AudioEmitterComponent, With<SoundscapeLayerMarker>>,
) {
    let active = soundscape.0.active;
    for mut emitter in query.iter_mut() {
        emitter.playing = active;
    }
}

pub struct SoundscapePlugin {
    pub initial: Soundscape,
}

impl SoundscapePlugin {
    pub fn new(initial: Soundscape) -> Self { Self { initial } }
    pub fn with_outdoor_default() -> Self { Self::new(Soundscape::outdoor_default()) }
}

impl bevy_app::Plugin for SoundscapePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.insert_resource(SoundscapeResource(self.initial.clone()));
        app.add_systems(
            bevy_app::Update,
            (soundscape_sync_system, soundscape_toggle_system).chain(),
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outdoor_default_has_3_layers() {
        let s = Soundscape::outdoor_default();
        assert_eq!(s.layers.len(), 3);
        assert!(s.active);
    }

    #[test]
    fn add_layer_increases_count() {
        let mut s = Soundscape::new();
        s.add_layer(SoundLayer {
            name: "test".into(),
            clip_path: "test.ogg".into(),
            volume: 0.5,
            looping: false,
            spatial: false,
        });
        assert_eq!(s.layers.len(), 1);
    }

    #[test]
    fn remove_layer_decreases_count() {
        let mut s = Soundscape::outdoor_default();
        assert!(s.remove_layer("wind"));
        assert_eq!(s.layers.len(), 2);
    }

    #[test]
    fn remove_nonexistent_layer_returns_false() {
        let mut s = Soundscape::outdoor_default();
        assert!(!s.remove_layer("nonexistent"));
        assert_eq!(s.layers.len(), 3);
    }

    #[test]
    fn soundscape_plugin_builds_without_panic() {
        use bevy_app::App;
        let mut app = App::new();
        app.add_plugins(SoundscapePlugin::with_outdoor_default());
        assert!(app.world().contains_resource::<SoundscapeResource>());
    }

    #[test]
    fn soundscape_sync_spawns_entities() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        world.insert_resource(SoundscapeResource(Soundscape::outdoor_default()));

        let mut schedule = Schedule::default();
        schedule.add_systems(soundscape_sync_system);
        schedule.run(&mut world);
        world.flush();

        let count = world.query::<&SoundscapeLayerMarker>().iter(&world).count();
        assert_eq!(count, 3);
    }

    #[test]
    fn soundscape_toggle_sets_playing() {
        use bevy_ecs::schedule::Schedule;
        use bevy_ecs::world::World;

        let mut world = World::new();
        let mut ss = Soundscape::outdoor_default();
        ss.active = false;
        world.insert_resource(SoundscapeResource(ss));

        world.spawn((
            SoundscapeLayerMarker { layer_name: "wind".into() },
            vox_core::ecs::AudioEmitterComponent {
                clip_path: "wind.ogg".into(),
                volume: 0.3,
                looping: true,
                playing: true,
                spatial: false,
            },
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(soundscape_toggle_system);
        schedule.run(&mut world);

        let emitter = world
            .query::<&vox_core::ecs::AudioEmitterComponent>()
            .iter(&world)
            .next()
            .unwrap();
        assert!(!emitter.playing);
    }
}
