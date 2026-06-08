//! # Play-in-Editor — Step 1, the play core (AAA Spec 12)
//!
//! Press Play: snapshot the authored [`EditorShell`] state (a cheap `Vec` clone),
//! run a fresh [`EngineLoop`] driving the authored splats via a script in a
//! headless simulation, Stop drops the session so the authored state is restored
//! EXACTLY (the sim only ever mutated a COPY — the snapshot stays pristine).
//!
//! Step 1 is proven headless against [`PlayController`] directly: there is no
//! viewport composite and no [`EditorShell`] field change, so it carries zero
//! risk to the existing shell tests. The only `mod.rs` change is `pub mod play;`.
//!
//! ## The spawn tuple — why `AssetRefComponent` is present
//!
//! `EngineLoop::step_scripts` calls `runtime.tick(dt)`, and `tick` runs
//! `frustum_cull_system` BEFORE `gather_splats_system`. The cull system REMOVES
//! `Visible` from every entity, then re-adds it only to entities that carry an
//! `AssetRefComponent` (and are within view distance). The gather query that
//! actually collects splats is `(&SplatAssetComponent, &TransformComponent)`
//! filtered `With<Visible>`.
//!
//! So an entity needs BOTH: `SplatAssetComponent` (so gather picks up its splats)
//! AND `AssetRefComponent` (so the per-frame cull keeps it `Visible`). Spawning a
//! bare `Visible` marker WITHOUT `AssetRefComponent` would have its `Visible`
//! stripped on the very first tick and gather zero splats — the empty-
//! `render_splats` failure the headline test guards against. The spawn tuple
//! below therefore adds `AssetRefComponent` to the proven gather tuple from
//! `engine_runtime.rs`'s `gather_splats_fills_render_buffer` test.

use vox_core::ecs::{
    AssetRefComponent, NameComponent, ScriptComponent, SplatAssetComponent, TransformComponent,
    Visible,
};
use vox_core::script_interface::{GameScript, ScriptContext};
use vox_core::types::GaussianSplat;

use glam::{Quat, Vec3};
use uuid::Uuid;

use ochroma_engine::engine_loop::{EngineLoop, SystemMask};
use vox_core::engine_runtime::EngineConfig;

/// Editor / play state machine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlayMode {
    Editing,
    Playing,
    Paused,
}

/// A cheap, pristine copy of the authored editor state taken at Press-Play.
///
/// Both fields are plain `Vec`s, so the clone is a memcpy-class copy with no
/// engine side effects. The running simulation operates on a SEPARATE world
/// built from this data, so the snapshot is never mutated and `stop()` can hand
/// it straight back to restore the authored scene exactly.
#[derive(Clone)]
pub struct AuthoredSnapshot {
    pub entities: Vec<crate::shell::ShellEntity>,
    pub overlay: Vec<GaussianSplat>,
}

/// Live state of one play session: the simulation loop, the frame counter, and
/// the most recent frame's render splats (drained out of the engine's
/// `RenderBuffer` by `step_scripts`).
pub struct PlaySession {
    pub loop_: EngineLoop,
    pub frame: u64,
    pub render_splats: Vec<GaussianSplat>,
}

/// Drives Press-Play / Pause / Resume / Stop. Owns the pristine snapshot taken at
/// Play and the live [`PlaySession`] while Playing/Paused.
pub struct PlayController {
    pub mode: PlayMode,
    pub session: Option<PlaySession>,
    pub snapshot: Option<AuthoredSnapshot>,
}

impl PlayController {
    pub fn new() -> Self {
        Self {
            mode: PlayMode::Editing,
            session: None,
            snapshot: None,
        }
    }

    /// Enter Play: store a pristine snapshot, build a fresh simulation loop,
    /// register the demo mover script, spawn the authored overlay as a scripted
    /// entity, and start the runtime. Returns a short human-readable status line.
    ///
    /// Uses `SystemMask::all()` + `EngineConfig::default()` — the SAME
    /// construction the engine's own `engine_loop` tests run headless without
    /// panicking (audio backend init, GI cache, ShadowMapper all succeed on the
    /// headless/GPU-less test path). `scripts` MUST be true (it is, in `all()`),
    /// otherwise `OrbitMover` would never run.
    pub fn enter_play(&mut self, snap: AuthoredSnapshot) -> String {
        // Keep the pristine copy for restore-on-stop.
        self.snapshot = Some(snap.clone());

        // Fresh, fully-enabled simulation loop. `all()` keeps scripts=true so the
        // mover ticks; constructs headless without panic (proven by the engine's
        // own engine_loop tests, which use this exact pair).
        let mut loop_ = EngineLoop::new(EngineConfig::default(), SystemMask::all());

        // Register the demo mover BEFORE start() so init_scripts can instantiate
        // it for the scripted entity.
        loop_
            .runtime
            .register_script("OrbitMover", || Box::new(OrbitMover { start_x: 5.0, t: 0.0 }));

        // Spawn the authored overlay as the EXACT raw tuple the gather query
        // needs (SplatAssetComponent + TransformComponent + Visible), PLUS
        // AssetRefComponent so frustum_cull_system keeps the entity Visible each
        // tick (see module docs), and ScriptComponent so OrbitMover drives it.
        let splats = snap.overlay.clone();
        let splat_count = splats.len() as u32;
        loop_.runtime.world.spawn((
            NameComponent("PlayOverlay".to_string()),
            SplatAssetComponent {
                uuid: Uuid::nil(),
                splat_count,
                splats,
            },
            TransformComponent {
                position: Vec3::new(5.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            Visible,
            // Required for frustum_cull_system to re-mark the entity Visible each
            // frame — without it gather collects zero splats (see module docs).
            AssetRefComponent {
                path: "play://overlay".to_string(),
                handle: 0,
            },
            ScriptComponent {
                scripts: vec!["OrbitMover".to_string()],
            },
        ));

        // start() internally calls init_scripts — do NOT call init_scripts here.
        loop_.runtime.start();

        self.session = Some(PlaySession {
            loop_,
            frame: 0,
            render_splats: Vec::new(),
        });
        self.mode = PlayMode::Playing;

        format!(
            "Playing — {} authored overlay splats, OrbitMover registered",
            splat_count
        )
    }

    /// Advance the simulation one frame. No-op unless Playing.
    ///
    /// `step_scripts` ticks the runtime AND drains the `RenderBuffer` (via
    /// `mem::take`), returning the moved splats — we read `render_splats` ONLY
    /// from that return value, never separately from the buffer (it was already
    /// drained).
    pub fn tick(&mut self, dt: f32) {
        if self.mode != PlayMode::Playing {
            return;
        }
        if let Some(session) = self.session.as_mut() {
            let s = session.loop_.step_scripts(dt);
            session.render_splats = s;
            session.loop_.step_physics(dt);
            session.frame += 1;
        }
    }

    /// Playing → Paused. No effect otherwise.
    pub fn pause(&mut self) {
        if self.mode == PlayMode::Playing {
            self.mode = PlayMode::Paused;
        }
    }

    /// Paused → Playing. No effect otherwise.
    pub fn resume(&mut self) {
        if self.mode == PlayMode::Paused {
            self.mode = PlayMode::Playing;
        }
    }

    /// Stop: drop the simulated world (it ceases to exist — auto-restore), return
    /// to Editing, and hand back the pristine authored snapshot for the shell to
    /// restore. The sim mutated only its own COPY, so the returned snapshot is
    /// exactly the authored state.
    pub fn stop(&mut self) -> Option<AuthoredSnapshot> {
        self.session = None;
        self.mode = PlayMode::Editing;
        self.snapshot.take()
    }
}

impl Default for PlayController {
    fn default() -> Self {
        Self::new()
    }
}

/// Demo script: walks the scripted entity along +X at 2 units/sec from
/// `start_x`. Stateful (`t` accumulates), so the engine's per-entity instance
/// caching is what makes the motion add up across frames.
pub struct OrbitMover {
    pub start_x: f32,
    pub t: f32,
}

impl GameScript for OrbitMover {
    fn on_update(&mut self, ctx: &mut ScriptContext, dt: f32) {
        self.t += dt;
        ctx.set_position([self.start_x + self.t * 2.0, 0.0, 0.0]);
    }

    fn name(&self) -> &str {
        "OrbitMover"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Author ~200 real volume splats all at x=5.0 (the authored overlay).
    fn authored_overlay() -> Vec<GaussianSplat> {
        (0..200)
            .map(|i| {
                let y = i as f32 * 0.01;
                GaussianSplat::volume(
                    [5.0, y, 0.0],
                    [0.1, 0.1, 0.1],
                    Quat::IDENTITY,
                    200,
                    [f16::from_f32(0.5).to_bits(); 16],
                )
            })
            .collect()
    }

    /// Press Play → tick → the scripted entity MOVES, and Stop restores the
    /// authored snapshot EXACTLY (the sim mutated a copy; the snapshot is
    /// pristine).
    #[test]
    fn tick_moves_a_scripted_entity_and_stop_restores() {
        let overlay = authored_overlay();
        // Keep a separate clone of the authored overlay to assert restore against.
        let authored = overlay.clone();
        assert_eq!(authored.len(), 200, "authored overlay has 200 splats");
        assert_eq!(
            authored[0].position()[0],
            5.0,
            "authored splats start at x=5.0"
        );

        let snap = AuthoredSnapshot {
            entities: Vec::new(),
            overlay,
        };

        let mut pc = PlayController::new();
        pc.enter_play(snap.clone());

        // ONE tick: proves the spawn tuple actually gathers (the FATAL-fix check —
        // if Visible were stripped by frustum cull, this would be empty).
        pc.tick(1.0 / 60.0);
        assert!(
            !pc.session.as_ref().unwrap().render_splats.is_empty(),
            "render_splats must be non-empty after one tick — the spawn tuple \
             (SplatAssetComponent + AssetRefComponent + Visible) gathers splats"
        );
        let p1 = pc.session.as_ref().unwrap().render_splats[0].position()[0];

        // 119 more ticks — the stateful OrbitMover keeps walking the entity +X.
        for _ in 0..119 {
            pc.tick(1.0 / 60.0);
        }
        let p120 = pc.session.as_ref().unwrap().render_splats[0].position()[0];

        // Stop drops the simulated world and returns the pristine snapshot.
        let restored = pc.stop().unwrap();
        let restored_x = restored.overlay[0].position()[0];

        println!("frame1_x={p1:.3} frame120_x={p120:.3} restored_x={restored_x:.3}");

        // The script MOVED the entity by more than 1 unit over 120 frames.
        assert!(
            (p120 - p1).abs() > 1.0,
            "OrbitMover must move the entity >1.0 over 120 frames: \
             frame1_x={p1}, frame120_x={p120}"
        );

        // Stop restored the authored state EXACTLY (the sim mutated only a copy).
        assert_eq!(
            restored.overlay.len(),
            authored.len(),
            "stop() restores the full authored overlay"
        );
        assert_eq!(
            restored_x, 5.0,
            "authored splats restored to their authored x=5.0 (snapshot is pristine)"
        );
        assert_eq!(pc.mode, PlayMode::Editing, "stop() returns to Editing");
    }
}
