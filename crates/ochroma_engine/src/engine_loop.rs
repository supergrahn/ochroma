//! # EngineLoop — unified per-frame simulation driver
//!
//! `EngineLoop` owns the CPU-side per-frame subsystems that a shell (editor or
//! game) previously wired inline: the ECS/script fixed-step ([`EngineRuntime`]),
//! Rapier physics + ECS sync, audio, spectral GI, and shadow updates. It
//! **composes** [`vox_core::engine_runtime::EngineRuntime`] (the ECS) as a field
//! rather than replacing it, so all of EngineRuntime's behavior and tests are
//! preserved by construction.
//!
//! ## What lives here vs. in the shell
//!
//! The loop owns only GPU-free *simulation*. Window/`wgpu`/input/UI/render and
//! the **character controller (KCC)** stay shell-side — the two shells use
//! different character types, so pulling KCC in here would force them to
//! converge (a behavior change). Do **not** add KCC to `EngineLoop`.
//!
//! The sub-steps are byte-for-byte relocations of the corresponding
//! `engine_runner.rs` blocks (see method docs for the source line ranges), so a
//! call site can be swapped to a sub-step with no behavior change.

use std::collections::HashMap;

use glam::Vec3;
use vox_core::engine_runtime::{EngineConfig, EngineRuntime};
use vox_core::types::GaussianSplat;

use vox_audio::{AudioEngine, SpatialAudioManager};
use vox_physics::rapier::RapierPhysicsWorld;
use vox_render::lighting::SunModel;
use vox_render::shadows::ShadowMapper;
use vox_render::spectral_atmosphere::SpectralAtmosphere;
use vox_render::spectral_gi::{GpuGi, SpectralRadianceCache};

/// Per-phase ordering vocabulary. The loop runs enabled phases in this fixed
/// order; these are stable names so call sites and tests can describe *when*
/// work happens. This is not a scheduler.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EnginePhase {
    Scripts,
    Physics,
    Audio,
    Gi,
    Shadows,
}

/// Opt-out flags so a shell can share code paths without adopting new behavior.
/// `walking_sim` starts with `gi=false, scripts=false` (it has neither today).
#[derive(Clone, Copy, Debug)]
pub struct SystemMask {
    pub scripts: bool,
    pub physics: bool,
    pub audio: bool,
    pub animation: bool,
    pub gi: bool,
    pub shadows: bool,
}

impl SystemMask {
    /// Every system on (the editor shell's behavior).
    pub fn all() -> Self {
        Self {
            scripts: true,
            physics: true,
            audio: true,
            animation: true,
            gi: true,
            shadows: true,
        }
    }

    /// Game-minimal: physics + audio + animation + shadows, no GI/scripts.
    pub fn game_minimal() -> Self {
        Self {
            scripts: false,
            physics: true,
            audio: true,
            animation: true,
            gi: false,
            shadows: true,
        }
    }
}

impl Default for SystemMask {
    fn default() -> Self {
        Self::all()
    }
}

/// Typed cross-sub-step blackboard so sub-steps share data (e.g. the post-GI
/// render-splat list, listener pose) without a god-struct or extra clones.
/// The loop is single-threaded, so stored values need no `Send` bound.
#[derive(Default)]
pub struct Blackboard {
    map: HashMap<std::any::TypeId, Box<dyn std::any::Any>>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: 'static>(&mut self, v: T) {
        self.map.insert(std::any::TypeId::of::<T>(), Box::new(v));
    }

    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.map
            .get(&std::any::TypeId::of::<T>())
            .and_then(|b| b.downcast_ref())
    }

    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&std::any::TypeId::of::<T>())
            .and_then(|b| b.downcast_mut())
    }
}

/// What the shell hands in each frame. Camera is passed **by value** (`Vec3`
/// copies) to avoid borrow friction with the shell's `&mut self`.
pub struct FrameCtx<'a> {
    pub dt: f32,
    pub input: &'a vox_core::input::InputState,
    pub camera_pos: Vec3,
    pub camera_fwd: Vec3,
    /// Time of day, 0..24, drives GI sky + shadow sun.
    pub hour: f32,
}

/// Side outputs the shell needs back for its own rendering. The loop never
/// renders.
#[derive(Default)]
pub struct FrameOutput {
    /// Splats drained from the [`RenderBuffer`] (via `mem::take`).
    ///
    /// [`RenderBuffer`]: vox_core::engine_runtime::RenderBuffer
    pub anim_splats: Vec<GaussianSplat>,
    pub sun_dir: Vec3,
    pub fixed_steps: u32,
}

/// Which spectral-GI implementation `step_gi` routes through.
///
/// The CPU path ([`SpectralRadianceCache`]) is the proven default and is always
/// present (the `spectral_gi` field on [`EngineLoop`]); it also serves as the
/// permanent fallback if the GPU path fails mid-run. When `Gpu` is active,
/// `step_gi` uploads the splats to the headless [`GpuGi`] device and reads back
/// the GI-lit result — proven bit-equivalent to the CPU reference by
/// `vox_render`'s `gpu_gi_matches_cpu_step_for_large_strided_scene` test.
enum GiBackend {
    Cpu,
    Gpu(Box<GpuGi>),
}

/// Unified per-frame simulation driver. Composes [`EngineRuntime`] (untouched)
/// and owns the CPU-side subsystems engine_runner already wired.
pub struct EngineLoop {
    /// Existing vox_core ECS — **untouched**, composed not replaced.
    pub runtime: EngineRuntime,
    pub physics: RapierPhysicsWorld,
    pub audio: AudioEngine,
    pub spatial_audio: SpatialAudioManager,
    pub shadow_mapper: ShadowMapper,
    pub spectral_atmosphere: SpectralAtmosphere,
    pub spectral_gi: SpectralRadianceCache,
    /// Sun model for shadow sun-direction (mirrors `LightManager::new(51.5)`).
    pub sun: SunModel,
    /// Pure-sim ECS<->Rapier maps relocated off the shell.
    pub entity_rapier_bodies: HashMap<u32, vox_physics::RigidBodyHandle>,
    pub mask: SystemMask,
    pub blackboard: Blackboard,
    /// Active GI backend. Private: callers select it via [`EngineLoop::use_gpu_gi`]
    /// or the `OCHROMA_GI` env override, and read which one is live via
    /// [`EngineLoop::gi_backend`]. `step_gi`'s signature/semantics are unchanged
    /// regardless of which backend is active.
    gi_backend: GiBackend,
    /// Wall-clock duration of the most recent `step_gi` call, in microseconds.
    /// `None` until the first `step_gi`. Surfaced via [`EngineLoop::last_gi_us`].
    last_gi_us: Option<u64>,
    /// Capacity the GPU backend was sized for. A frame whose splat count
    /// exceeds it routes through the CPU path for that call (the GPU pass
    /// would clamp and leave the tail unlit — a silent CPU/GPU divergence);
    /// the GPU backend stays selected for subsequent smaller frames.
    gpu_gi_capacity: u32,
}

/// Sizing for the headless GPU GI device. Large enough for the smoke scenes and
/// typical editor views; a frame beyond this runs the (unlimited) CPU path for
/// that call rather than letting the GPU clamp diverge from the CPU mirror.
const GPU_GI_CAPACITY: u32 = 200_000;

impl EngineLoop {
    /// Build the loop. Mirrors `EngineApp::new` (engine_runner.rs:425-516):
    /// Rapier ground plane, audio backend init, GI cache, ShadowMapper(512),
    /// earth atmosphere, London-latitude sun. Does **not** create window/GPU.
    pub fn new(config: EngineConfig, mask: SystemMask) -> Self {
        let runtime = EngineRuntime::new(config);

        // Rapier physics world with a 1km x 1km ground plane (engine_runner:428-434).
        let physics = {
            let mut p = RapierPhysicsWorld::new();
            p.add_static_collider([0.0, -0.5, 0.0], [500.0, 0.5, 500.0]);
            p
        };

        // Audio engine with rodio backend (engine_runner:497-502).
        let audio = {
            let mut a = AudioEngine::new(64);
            a.init_backend();
            a
        };

        // Spatial audio manager (engine_runner:512-515).
        let spatial_audio = SpatialAudioManager::new();

        // GI backend selection. Precedence (read once, at construction):
        //   1. OCHROMA_GI=gpu  → try the GPU path; on adapter/device failure log
        //      one eprintln and fall back to Cpu (never panics).
        //   2. OCHROMA_GI=cpu  → force the proven CPU path.
        //   3. unset / other   → default Cpu (the proven path).
        // A later `use_gpu_gi()` call can still upgrade an env-default-Cpu loop.
        let gi_env = std::env::var("OCHROMA_GI").ok();
        let gi_backend = match gi_env.as_deref() {
            // Case-insensitive: "gpu"/"GPU"/"Gpu" all select the GPU path.
            Some(v) if v.eq_ignore_ascii_case("gpu") => match GpuGi::new(GPU_GI_CAPACITY) {
                Ok(g) => GiBackend::Gpu(Box::new(g)),
                Err(e) => {
                    eprintln!(
                        "[ochroma_engine] OCHROMA_GI=gpu requested but GPU GI init failed \
                         ({e}); falling back to CPU spectral GI."
                    );
                    GiBackend::Cpu
                }
            },
            Some(v) if v.eq_ignore_ascii_case("cpu") => GiBackend::Cpu,
            // An unrecognized value silently defaulting would hide typos
            // (OCHROMA_GI=Gpu used to mean "cpu" without a word) — warn once.
            Some(other) => {
                eprintln!(
                    "[ochroma_engine] unrecognized OCHROMA_GI value '{other}' \
                     (expected gpu|cpu); using the CPU path."
                );
                GiBackend::Cpu
            }
            None => GiBackend::Cpu,
        };

        Self {
            runtime,
            physics,
            audio,
            spatial_audio,
            // engine_runner:523 — ShadowMapper::new(512).
            shadow_mapper: ShadowMapper::new(512),
            // engine_runner:532 — SpectralAtmosphere::earth().
            spectral_atmosphere: SpectralAtmosphere::earth(),
            // engine_runner:533 — SpectralRadianceCache::new(0).
            spectral_gi: SpectralRadianceCache::new(0),
            // engine_runner:456 — LightManager::new(51.5) — London latitude.
            sun: SunModel::new(51.5),
            entity_rapier_bodies: HashMap::new(),
            mask,
            blackboard: Blackboard::new(),
            gi_backend,
            last_gi_us: None,
            gpu_gi_capacity: GPU_GI_CAPACITY,
        }
    }

    /// Switch `step_gi` to the headless GPU spectral-GI backend.
    ///
    /// Returns `Err` (and leaves the loop on whatever backend it had) if no wgpu
    /// adapter is available or device creation fails — never panics. If the env
    /// override `OCHROMA_GI=gpu` already activated the GPU path at construction,
    /// this is idempotent (a fresh device is created and swapped in).
    ///
    /// Precedence note: an explicit `use_gpu_gi()` call wins over the
    /// construction-time env default, because it runs later. `OCHROMA_GI=cpu`
    /// only sets the *default*; it does not forbid a later explicit upgrade.
    pub fn use_gpu_gi(&mut self) -> Result<(), String> {
        match GpuGi::new(self.gpu_gi_capacity) {
            Ok(g) => {
                self.gi_backend = GiBackend::Gpu(Box::new(g));
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Name of the currently active GI backend: `"cpu"` or `"gpu"`. Shells can
    /// surface this in a HUD without owning the selection logic.
    pub fn gi_backend(&self) -> &'static str {
        match self.gi_backend {
            GiBackend::Cpu => "cpu",
            GiBackend::Gpu(_) => "gpu",
        }
    }

    /// Wall-clock time of the most recent [`step_gi`](Self::step_gi) call, in
    /// microseconds. `None` until `step_gi` has run at least once.
    pub fn last_gi_us(&self) -> Option<u64> {
        self.last_gi_us
    }

    /// Test-only: shrink the GPU capacity so the over-capacity CPU routing is
    /// exercisable without building a 200k-splat scene. Not for production —
    /// the real capacity is fixed at construction.
    #[doc(hidden)]
    pub fn set_gpu_gi_capacity_for_test(&mut self, capacity: u32) {
        self.gpu_gi_capacity = capacity;
    }

    /// Builder-style mask override.
    pub fn with_mask(mut self, mask: SystemMask) -> Self {
        self.mask = mask;
        self
    }

    // -----------------------------------------------------------------------
    // Re-exposed engine state (for shells that read it inline today)
    // -----------------------------------------------------------------------

    pub fn world(&mut self) -> &mut bevy_ecs::world::World {
        &mut self.runtime.world
    }

    /// Mutable access to the composed ECS world. Same as [`world`](Self::world);
    /// named `world_mut` so call sites that need an explicit `&mut World` (e.g.
    /// running a one-off bevy system, building a query) read clearly.
    pub fn world_mut(&mut self) -> &mut bevy_ecs::world::World {
        &mut self.runtime.world
    }

    pub fn time_of_day(&self) -> f32 {
        self.runtime.time_of_day()
    }

    pub fn set_time_of_day(&mut self, hour: f32) {
        self.runtime.set_time_of_day(hour);
    }

    // -----------------------------------------------------------------------
    // Sub-steps — public so shells call them à la carte during migration.
    // Each body is a relocation of the cited engine_runner.rs block.
    // -----------------------------------------------------------------------

    /// Run the ECS/script fixed-step (`runtime.tick(dt)`, engine_runner:1922)
    /// and drain animation splats written to the [`RenderBuffer`] into
    /// `FrameOutput.anim_splats` (engine_runner:971-977).
    ///
    /// NOTE: the shell's procedural `animation_system` is a game component
    /// (`vox_app::walk_animation::ProceduralWalkComponent`) and stays shell-side;
    /// the shell runs it (it appends to the RenderBuffer) and this drain picks up
    /// whatever was written there.
    ///
    /// [`RenderBuffer`]: vox_core::engine_runtime::RenderBuffer
    pub fn step_scripts(&mut self, dt: f32) -> Vec<GaussianSplat> {
        self.runtime.tick(dt);
        std::mem::take(
            &mut self
                .runtime
                .world
                .resource_mut::<vox_core::engine_runtime::RenderBuffer>()
                .splats,
        )
    }

    /// Step Rapier physics and sync dynamic bodies back to ECS transforms.
    /// Relocated from engine_runner.rs:1969-1987.
    pub fn step_physics(&mut self, _dt: f32) {
        self.physics.step();
        // Sync: read positions from Rapier dynamic bodies back into ECS transforms.
        {
            use vox_core::ecs::TransformComponent;
            let body_map: Vec<(u32, vox_physics::RigidBodyHandle)> = self
                .entity_rapier_bodies
                .iter()
                .map(|(&e, &h)| (e, h))
                .collect();
            for (eid, handle) in body_map {
                if let Some(pos) = self.physics.body_position(handle) {
                    let mut query = self
                        .runtime
                        .world
                        .query::<(bevy_ecs::prelude::Entity, &mut TransformComponent)>();
                    for (entity, mut transform) in query.iter_mut(&mut self.runtime.world) {
                        if entity.index() == eid {
                            transform.position = Vec3::new(pos[0], pos[1], pos[2]);
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Tick the legacy audio engine + spatial audio manager and set the
    /// listener pose. Relocated from engine_runner.rs:1990-2006.
    pub fn step_audio(&mut self, dt: f32, listener_pos: Vec3, listener_fwd: Vec3) {
        self.audio.tick(dt);
        self.audio.set_listener(listener_pos);
        self.spatial_audio
            .set_listener(listener_pos, listener_fwd, Vec3::Y);
        self.spatial_audio.tick(dt);
    }

    /// Drain pending `PlaySound` script commands and dispatch them to the
    /// spatial audio manager as procedural tones. Relocated from
    /// engine_runner.rs:2008-2031.
    pub fn drain_sound_commands(&mut self) {
        use vox_core::script_interface::ScriptCommand;
        let pending = std::mem::take(
            &mut self
                .runtime
                .world
                .resource_mut::<vox_core::engine_runtime::PendingScriptCommands>()
                .commands,
        );
        for (_entity, commands) in &pending {
            for cmd in commands {
                if let ScriptCommand::PlaySound { clip, volume, .. } = cmd {
                    let freq = match clip.as_str() {
                        "click" => 800.0,
                        "collect" => 600.0,
                        "jump" => 400.0,
                        _ => 440.0,
                    };
                    self.spatial_audio.play_tone(freq, 0.2, *volume);
                }
            }
        }
        // Put unprocessed commands back (other systems may need them).
        self.runtime
            .world
            .resource_mut::<vox_core::engine_runtime::PendingScriptCommands>()
            .commands = pending;
    }

    /// Update the spectral atmosphere from time of day, set the GI sky, then
    /// propagate + apply live spectral GI to `splats`. Returns the GI-modulated
    /// splat list. Relocated from engine_runner.rs:980-993.
    pub fn step_gi(&mut self, splats: &[GaussianSplat], hour: f32) -> Vec<GaussianSplat> {
        let t0 = std::time::Instant::now();
        // The GPU device is sized for `gpu_gi_capacity` splats and CLAMPS past
        // it — the tail would silently stay unlit while the CPU path lights
        // everything (exactly the divergence class the equivalence work
        // killed). An over-capacity frame therefore routes to the CPU path,
        // which has no size limit; the backend selection is untouched so a
        // later smaller frame uses the GPU again.
        let over_capacity = matches!(self.gi_backend, GiBackend::Gpu(_))
            && splats.len() > self.gpu_gi_capacity as usize;
        let out = match &self.gi_backend {
            GiBackend::Cpu => self.step_gi_cpu(splats, hour),
            GiBackend::Gpu(_) if over_capacity => self.step_gi_cpu(splats, hour),
            GiBackend::Gpu(gpu) => match gpu.step(splats, hour) {
                Ok(lit) => lit,
                Err(e) => {
                    // One log line, then permanently fall back to CPU so the rest
                    // of the run never spams per-frame errors. Callers still get
                    // valid splats THIS frame: we run the CPU path below.
                    eprintln!(
                        "[ochroma_engine] GPU spectral GI failed ({e}); permanently \
                         falling back to CPU spectral GI for the rest of this run."
                    );
                    self.gi_backend = GiBackend::Cpu;
                    self.step_gi_cpu(splats, hour)
                }
            },
        };
        self.last_gi_us = Some(t0.elapsed().as_micros() as u64);
        out
    }

    /// CPU spectral-GI path — the proven reference and permanent fallback.
    /// Update the spectral atmosphere from `hour`, set the GI sky, then run
    /// propagate and apply. The hour → sun-zenith mapping is the SHARED
    /// [`vox_render::spectral_gi::sun_zenith_for_hour`] so it can never drift from
    /// the GPU path's `sky_ambient_for_hour`.
    fn step_gi_cpu(&mut self, splats: &[GaussianSplat], hour: f32) -> Vec<GaussianSplat> {
        self.spectral_atmosphere.sun_zenith =
            vox_render::spectral_gi::sun_zenith_for_hour(hour);
        self.spectral_atmosphere.sun_elevation = self.spectral_atmosphere.sun_zenith;
        self.spectral_gi.set_sky(&self.spectral_atmosphere);
        // The emitter bound is the SHARED constant the GPU pass also uses —
        // a literal here would silently diverge the CPU/GPU GI mirror.
        self.spectral_gi
            .propagate(splats, vox_render::spectral_gi::MAX_EMITTERS as usize);
        self.spectral_gi.apply(splats)
    }

    /// Update the shadow map for the given camera + time of day, render the
    /// shadow map over `splats`, and return the sun direction. Relocated from
    /// engine_runner.rs:1063-1077.
    pub fn step_shadows(
        &mut self,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        hour: f32,
        splats: &[GaussianSplat],
    ) -> Vec3 {
        let sun_dir = self.sun.sun_direction(hour, 172);
        self.shadow_mapper.update(cam_pos, cam_fwd, sun_dir);

        let shadow_positions: Vec<Vec3> =
            splats.iter().map(|s| Vec3::from(s.position())).collect();
        let shadow_radii: Vec<f32> = splats
            .iter()
            .map(|s| (s.scale_u().abs() + s.scale_v().abs() + s.scale_w().abs()) / 3.0)
            .collect();
        self.shadow_mapper
            .render_shadow_map(&shadow_positions, &shadow_radii);
        sun_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use vox_core::engine_runtime::{EngineConfig, FixedStepCounter};
    use vox_core::script_interface::{GameScript, ScriptContext};

    fn test_loop() -> EngineLoop {
        EngineLoop::new(EngineConfig::default(), SystemMask::all())
    }

    /// (1) step_physics drops a dynamic body under gravity and the result
    /// matches a direct `RapierPhysicsWorld::step()` reference run.
    #[test]
    fn step_physics_drops_body_under_gravity_matching_reference() {
        let mut lp = test_loop();
        let (body, _) = lp.physics.add_dynamic_box([0.0, 10.0, 0.0], [0.5, 0.5, 0.5], 1.0);

        for _ in 0..60 {
            lp.step_physics(1.0 / 60.0);
        }
        let y_loop = lp.physics.body_position(body).expect("body exists")[1];

        // Direct reference: same world setup, same number of steps.
        let mut reference = {
            let mut p = RapierPhysicsWorld::new();
            p.add_static_collider([0.0, -0.5, 0.0], [500.0, 0.5, 500.0]);
            p
        };
        let (ref_body, _) =
            reference.add_dynamic_box([0.0, 10.0, 0.0], [0.5, 0.5, 0.5], 1.0);
        for _ in 0..60 {
            reference.step();
        }
        let y_ref = reference.body_position(ref_body).expect("ref body exists")[1];

        assert!(
            y_loop < 10.0,
            "body must fall under gravity: y_loop={y_loop} should be < 10.0"
        );
        assert!(
            (y_loop - y_ref).abs() < 1e-4,
            "step_physics must match direct step(): y_loop={y_loop}, y_ref={y_ref}"
        );
    }

    /// (2) step_gi raises a band of a dark receiver splat sitting next to a
    /// bright opaque emitter, per the propagate/apply formula
    /// (`out = spectral + irr * 0.5`).
    #[test]
    fn step_gi_brightens_receiver_band_near_emitter() {
        let mut lp = test_loop();

        // Emitter: opacity > 128 so propagate() treats it as a GI emitter,
        // bright in every band.
        let emitter = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2],
            glam::Quat::IDENTITY,
            255,
            [f16::from_f32(1.0).to_bits(); 16],
        );
        // Receiver: low opacity (not an emitter), starts fully dark.
        let receiver = GaussianSplat::volume(
            [0.5, 0.0, 0.0],
            [0.2, 0.2, 0.2],
            glam::Quat::IDENTITY,
            10,
            [f16::from_f32(0.0).to_bits(); 16],
        );

        let input = vec![emitter, receiver];
        let in_band = f16::from_bits(input[1].spectral()[8]).to_f32();
        assert_eq!(in_band, 0.0, "receiver band 8 must start at 0.0");

        let out = lp.step_gi(&input, 12.0);
        let out_band = f16::from_bits(out[1].spectral()[8]).to_f32();

        assert!(
            out_band > in_band,
            "GI must brighten the receiver: out_band={out_band} should exceed in_band={in_band}"
        );
        // The emitter is bright + close; with the formula out = clamp(0 + irr*0.5),
        // and irr being the (scaled, alpha-blended) incoming radiance, the band
        // must rise by a non-trivial amount.
        assert!(
            out_band > 0.001,
            "GI band lift must be non-trivial: out_band={out_band}"
        );
    }

    /// (3) step_scripts ticks the ECS fixed-step: a registered counting
    /// GameScript advances once per tick and FixedStepCounter reflects it.
    #[test]
    fn step_scripts_advances_counting_script_and_fixed_steps() {
        struct CounterScript {
            ticks: u32,
        }
        impl GameScript for CounterScript {
            fn on_update(&mut self, _ctx: &mut ScriptContext, _dt: f32) {
                self.ticks += 1;
            }
            fn name(&self) -> &str {
                "Counter"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let mut lp = test_loop();
        lp.runtime
            .register_script("Counter", || Box::new(CounterScript { ticks: 0 }));
        let entity = lp.runtime.spawn("Ticker").with_script("Counter").id();
        lp.runtime.start();

        // 0.02s > fixed_dt (1/60) => exactly one fixed step per tick.
        for _ in 0..3 {
            lp.step_scripts(0.02);
        }

        let steps = lp
            .runtime
            .world
            .resource::<FixedStepCounter>()
            .steps_this_frame;
        assert_eq!(steps, 1, "last tick of 0.02s must run exactly one fixed step");

        let instances = lp
            .runtime
            .world
            .resource::<vox_core::engine_runtime::ScriptInstances>();
        let script = instances
            .scripts
            .get(&(entity, "Counter".to_string()))
            .expect("cached script instance should exist");
        let counter = script
            .as_any()
            .downcast_ref::<CounterScript>()
            .expect("instance should be a CounterScript");
        assert_eq!(
            counter.ticks, 3,
            "script counter must persist and reach 3 across 3 ticks"
        );
    }

    /// (4) Blackboard round-trip: insert a Vec<GaussianSplat>, mutate via
    /// get_mut, and observe the mutation through get.
    #[test]
    fn blackboard_round_trip_persists_mutation() {
        let mut bb = Blackboard::new();
        let splat = GaussianSplat::volume(
            [1.0, 2.0, 3.0],
            [0.1, 0.1, 0.1],
            glam::Quat::IDENTITY,
            128,
            [0u16; 16],
        );
        bb.insert(vec![splat]);

        {
            let v = bb
                .get_mut::<Vec<GaussianSplat>>()
                .expect("inserted vec should be retrievable");
            v[0].set_position([9.0, 8.0, 7.0]);
        }

        let v = bb
            .get::<Vec<GaussianSplat>>()
            .expect("vec should still be present");
        assert_eq!(
            v[0].position(),
            [9.0, 8.0, 7.0],
            "mutation through get_mut must persist and be visible via get"
        );
    }
}
