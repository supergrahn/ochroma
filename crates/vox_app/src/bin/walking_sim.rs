// Hide the console window on Windows (GUI application)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Ochroma Engine — Walking Simulator (Dogfood Game)
//!
//! The first game built on the engine. Proves everything works.
//! Walk around, collect 10 glowing orbs, win!
//!
//! cargo run --bin walking_sim

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use ochroma_engine::engine_loop::{EngineLoop, SystemMask};
use vox_ai::perception::{BehaviorState, SpectralPerceptionAgent, ZonedRadianceSource};
use vox_audio::biome_soundscape::{BiomeAmbientMix, BiomeKind};
use vox_audio::AudioCommand;
use vox_core::character_controller::{CharacterController, character_controller_tick};
use vox_core::ecs::TransformComponent;
use vox_core::engine_runtime::EngineConfig;
use vox_core::game_ui::{
    burn_text, GameState, GameUI, UIElement, UIPosition, UISize, CHAR_H, CHAR_STRIDE,
};
use vox_ui::game_hud::GameHud;
use vox_ui::game_menu::GameMenu;
use vox_ui::spectral_hud::SpectralRadianceCache;
use vox_ui::vello_ctx::VelloCtxCpu;
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::atom_budget::{AtomBudgetSelector, Selection};
use vox_render::clas;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::rigid_animation::RigidClip;
use vox_render::shadows::ShadowMapper;
use vox_render::spectral::RenderCamera;
use vox_render::spectral_framebuffer::SpectralFramebuffer;
use vox_render::spectral_tonemapper::{tonemap_spectral_framebuffer, ToneMapSettings};
use vox_script::rhai_runtime::RhaiRuntime;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// Resolve the path to the live game script `assets/scripts/walking_sim.rhai`.
/// Tries (in order): the current working directory (workspace root when run via
/// `cargo run`), then the workspace root derived from this crate's
/// `CARGO_MANIFEST_DIR` (`crates/vox_app` -> two levels up). Returns the first
/// path that exists, else the cwd-relative path (so the error message is sane).
fn resolve_script_path() -> std::path::PathBuf {
    const REL: &str = "assets/scripts/walking_sim.rhai";
    let cwd_rel = std::path::PathBuf::from(REL);
    if cwd_rel.exists() {
        return cwd_rel;
    }
    // crates/vox_app -> workspace root is two parents up.
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(ws_root) = manifest.parent().and_then(|p| p.parent()) {
        let p = ws_root.join(REL);
        if p.exists() {
            return p;
        }
    }
    cwd_rel
}

// ---------------------------------------------------------------------------
// Game-menu option labels (selectable rows the shell draws over GameMenu rects)
// ---------------------------------------------------------------------------
const MAIN_MENU_OPTIONS: [&str; 2] = ["START", "QUIT"];
const PAUSE_MENU_OPTIONS: [&str; 2] = ["RESUME", "QUIT"];
const WIN_MENU_OPTIONS: [&str; 1] = ["CONTINUE"];

// Sun direction (normalized, pointing toward ground = positive Y component negative)
const SUN_DIR: Vec3 = Vec3::new(0.4, -0.8, 0.3);

// ---------------------------------------------------------------------------
// Day/night cycle
// ---------------------------------------------------------------------------

// Real seconds per in-game hour during normal windowed play. 30 s/hour means a
// full 24h day takes 12 real minutes — slow enough to feel like a cycle, fast
// enough that a few minutes of play visibly moves the sun.
const REAL_SECS_PER_GAME_HOUR: f32 = 30.0;

// In the 160-frame smoke (≈2.6 s of sim at 60 Hz) the normal rate would barely
// move the clock, so we accelerate time in smoke by this multiplier to exercise
// a visible swing of the day/night cycle inside the run.
const SMOKE_TIME_MULTIPLIER: f32 = 400.0;

/// Sun direction as a function of time of day. The sun arcs east->up->west:
/// rises around 6h, peaks (straight overhead-ish) at noon, sets around 18h, and
/// is below the horizon (pointing up = no direct light reaches the ground) at
/// night. Returned vector points *from the sun toward the ground* (so its Y is
/// negative during the day).
fn sun_dir_for_hour(hour: f32) -> Vec3 {
    // Map [0,24) to a full circle; noon = sun highest.
    let t = (hour / 24.0) * std::f32::consts::TAU;
    // Sun elevation: +1 at noon, -1 at midnight.
    let elevation = -(t).cos(); // hour 12 -> cos(pi) = -1 -> elevation +1
    // Sun azimuth sweeps east (-x) to west (+x) across the day.
    let azimuth = (t).sin();
    // Direction the light travels (toward ground): negative Y when sun is up.
    Vec3::new(azimuth * 0.5, -elevation.max(0.05), 0.3).normalize()
}

/// Sky/sun brightness multiplier in [night_floor, 1.0] as a function of hour.
/// Peaks at noon, bottoms out at night. Used to scale rendered luminance so
/// night frames are measurably darker than noon frames.
fn sky_brightness_for_hour(hour: f32) -> f32 {
    // Daylight curve: a clamped cosine centred on noon. Below the horizon the
    // value is clamped to a small night floor (moonlight) so the scene never
    // goes fully black.
    let t = (hour / 24.0) * std::f32::consts::TAU;
    let day = (-(t).cos()).max(0.0); // 0 at midnight, 1 at noon
    let night_floor = 0.18;
    night_floor + (1.0 - night_floor) * day
}

// Live spectral GI is expensive (propagate is O(N) per splat over a 256-cap
// neighbourhood). Full-scene GI every frame blows the smoke time budget, so we
// recompute the GI-lit scene only every GI_CADENCE frames and reuse the cached
// result in between. The cadence keeps indirect light visibly updating while
// the 160-frame smoke finishes in well under 2 minutes.
const GI_CADENCE: u32 = 10;

// Number of nearest scene splats fed to the GI step. Full terrain is tens of
// thousands of splats; GI over all of them per cadence is still too slow for the
// budget, so we run GI on the nearest-K subset around the player each time. This
// is an honest performance limit — distant indirect light is not recomputed.
const GI_NEAREST_K: usize = 2000;

// Per-frame atom budget for static-scene splat selection. The static scene
// holds ~65k splats; the AtomBudgetSelector caps the rendered static set to
// this many per frame (≈ ⅓ of the scene) — a visible win with no visible loss
// at the demo's draw distances. Game (vox_app) owns this constant; the engine
// selector is budget-agnostic.
const ATOM_BUDGET: usize = 24_000;

// ---------------------------------------------------------------------------
// AABB collision
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BoundingBox {
    min: Vec3,
    max: Vec3,
}

fn check_building_collision(player_pos: &mut Vec3, buildings: &[BoundingBox]) {
    let radius = 1.0;
    for bb in buildings {
        if player_pos.x > bb.min.x - radius
            && player_pos.x < bb.max.x + radius
            && player_pos.z > bb.min.z - radius
            && player_pos.z < bb.max.z + radius
            && player_pos.y < bb.max.y
        {
            // Push player out via nearest face
            let dx_min = (player_pos.x - (bb.min.x - radius)).abs();
            let dx_max = (player_pos.x - (bb.max.x + radius)).abs();
            let dz_min = (player_pos.z - (bb.min.z - radius)).abs();
            let dz_max = (player_pos.z - (bb.max.z + radius)).abs();

            let min_d = dx_min.min(dx_max).min(dz_min).min(dz_max);
            if min_d == dx_min {
                player_pos.x = bb.min.x - radius;
            } else if min_d == dx_max {
                player_pos.x = bb.max.x + radius;
            } else if min_d == dz_min {
                player_pos.z = bb.min.z - radius;
            } else {
                player_pos.z = bb.max.z + radius;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Game structs
// ---------------------------------------------------------------------------

struct Orb {
    position: Vec3,
    collected: bool,
    bob_phase: f32,
}

/// A dynamic physics box the player dropped with KeyQ. Falls under gravity,
/// rendered as a single splat at its live Rapier body position, and fractures
/// (shrinks/splits) on ground impact via SpectralPhysics.
struct DroppedBox {
    handle: vox_physics::RigidBodyHandle,
    /// Material spectral profile (drives both fracture brittleness + impact audio).
    material: [u16; 16],
    half_extents: Vec3,
    spawn_y: f32,
    /// Live until it has fractured (then it stays as shrunken fragments but is no
    /// longer eligible to fracture again).
    fractured: bool,
    /// Render scale, shrunk on fracture per the impact result.
    render_scale: f32,
}

/// Windmill entity — base splats + blade splats with rigid animation.
struct Windmill {
    base_splats: Vec<GaussianSplat>,
    blade_splats_local: Vec<GaussianSplat>,   // splats in local space (around origin)
    blade_splats_world: Vec<GaussianSplat>,   // splats after applying animated rotation
    position: Vec3,
    clip: RigidClip,
    anim_time: f32,
}

impl Windmill {
    fn new(position: Vec3) -> Self {
        // Build base: a column of splats
        let mut base_splats = Vec::new();
        let base_spd: [u16; 16] = std::array::from_fn(|i| {
            let v: f32 = if i < 8 { 0.6 } else { 0.3 };
            half::f16::from_f32(v).to_bits()
        });
        for iy in 0..8 {
            let y = iy as f32 * 0.5;
            base_splats.push(GaussianSplat::volume(
                [position.x, position.y + y, position.z],
                [0.25, 0.25, 0.25],
                Quat::IDENTITY,
                200,
                base_spd,
            ));
        }

        // Build blade splats in local space (4 blades radiating outward)
        let mut blade_splats_local = Vec::new();
        let blade_height = 4.0; // attach blades at this height on the base
        let blade_spd: [u16; 16] = std::array::from_fn(|i| {
            let v: f32 = if (4..=11).contains(&i) { 0.85 } else { 0.4 };
            half::f16::from_f32(v).to_bits()
        });

        for blade in 0..4 {
            let angle = blade as f32 * std::f32::consts::FRAC_PI_2;
            // Each blade is a row of splats along one radial direction
            for r in 1..=5 {
                let radius = r as f32 * 0.4;
                blade_splats_local.push(GaussianSplat::volume(
                    [radius * angle.cos(), 0.0, radius * angle.sin()],
                    [0.18, 0.18, 0.18],
                    Quat::IDENTITY,
                    210,
                    blade_spd,
                ));
            }
        }

        let blade_splats_world = blade_splats_local.clone();

        // Spin at 0.4 rotations/second around Y axis
        let clip = RigidClip::rotation_loop(Vec3::Y, 0.4);

        Windmill {
            base_splats,
            blade_splats_local,
            blade_splats_world,
            position: position + Vec3::new(0.0, blade_height, 0.0),
            clip,
            anim_time: 0.0,
        }
    }

    /// Advance animation and recompute world-space blade splats.
    fn tick(&mut self, dt: f32) {
        self.anim_time += dt;
        let kf = self.clip.sample(self.anim_time);
        let rot = kf.rotation;

        // Apply rotation + translation to each blade splat
        for (src, dst) in self
            .blade_splats_local
            .iter()
            .zip(self.blade_splats_world.iter_mut())
        {
            let local_pos = Vec3::from(src.position());
            let rotated = rot * local_pos;
            let world_pos = rotated + self.position;
            *dst = *src;
            dst.set_position([world_pos.x, world_pos.y, world_pos.z]);
        }
    }
}

/// A wandering AI NPC. It drives the real vox_ai perception->decision path:
/// each tick it builds a [`ZonedRadianceSource`] in which the player radiates a
/// hot fire-band spectral signature within FLEE_RADIUS, calls
/// `SpectralPerceptionAgent::sense()` to perceive the local radiance, then
/// `assess_threat()` to turn that percept into a `BehaviorState`. When the
/// player is close the fire-band energy classifies as `Flee` and the NPC
/// kinematically steers away; otherwise it wanders between waypoints.
struct Npc {
    /// vox_ai perception agent — the source of the retrievable behaviour decision.
    agent: SpectralPerceptionAgent,
    position: Vec3,
    /// Patrol waypoints (XZ). The NPC cycles through these while wandering.
    waypoints: Vec<Vec3>,
    current_waypoint: usize,
    speed: f32,
    /// Last behaviour state, to detect transitions (Wander/Patrol <-> Flee).
    last_state: BehaviorState,
    /// Number of times the AI behaviour state changed over the run.
    state_changes: u32,
    /// Total ground distance travelled (metres).
    distance_travelled: f32,
    /// Spawn position, so the smoke can assert net displacement.
    spawn_pos: Vec3,
    /// Spectral profile for rendering the NPC body (a cool teal so it reads as a
    /// distinct character against terrain).
    body_spectral: [u16; 16],
}

// Player perceived as a threat within this radius (metres). Inside it the NPC's
// percept picks up the player's hot fire-band signature -> Flee.
const NPC_FLEE_RADIUS: f32 = 8.0;

impl Npc {
    fn new(spawn: Vec3, waypoints: Vec<Vec3>) -> Self {
        // Cool teal body: energy concentrated in the green/cyan bands (5..9),
        // low in the fire bands, so the NPC itself is never mistaken for a threat.
        let body_spectral: [u16; 16] = std::array::from_fn(|i| {
            let v: f32 = if (4..=8).contains(&i) { 0.85 } else { 0.15 };
            half::f16::from_f32(v).to_bits()
        });
        Npc {
            agent: SpectralPerceptionAgent::new(spawn, NPC_FLEE_RADIUS + 4.0),
            position: spawn,
            waypoints,
            current_waypoint: 0,
            speed: 4.0,
            last_state: BehaviorState::Idle,
            state_changes: 0,
            distance_travelled: 0.0,
            spawn_pos: spawn,
            body_spectral,
        }
    }

    /// Drive perception -> decision -> kinematic steering for one tick.
    fn tick(&mut self, dt: f32, player_pos: Vec3) {
        let prev_pos = self.position;
        self.agent.position = self.position;

        // Build the radiance field the NPC perceives: a calm green ambient
        // everywhere, with the player projected as a hot fire-band (10..15) zone
        // of radius FLEE_RADIUS. This is the GI the NPC senses.
        let calm_ambient: [f32; 16] = std::array::from_fn(|i| if (5..=8).contains(&i) { 0.25 } else { 0.05 });
        let player_hot: [f32; 16] = std::array::from_fn(|i| if (10..=15).contains(&i) { 0.9 } else { 0.05 });
        let gi = ZonedRadianceSource {
            zones: vec![(player_pos, NPC_FLEE_RADIUS, player_hot)],
            background: calm_ambient,
        };

        // Perceive (records into spectral memory) and classify into a decision.
        let percept = self.agent.sense(&gi);
        // Re-encode the perceived radiance as the f16 spectral the classifier
        // expects, then assess. assess_threat stores the retrievable decision.
        let spectral: [u16; 16] =
            std::array::from_fn(|i| half::f16::from_f32(percept.radiance[i]).to_bits());
        let assessment = self.agent.assess_threat(&spectral);
        let state = assessment.behavior;

        if state != self.last_state {
            self.state_changes += 1;
            self.last_state = state;
        }

        // Kinematic steering on the ground plane (no physics body).
        let target = if state >= BehaviorState::Investigate {
            // Threatened: flee directly away from the player.
            let away = self.position - player_pos;
            let away = if away.length_squared() > 1e-4 { away.normalize() } else { Vec3::Z };
            self.position + away * 10.0
        } else {
            // Calm: wander toward the current waypoint, advancing when reached.
            let wp = self.waypoints[self.current_waypoint];
            if (wp - self.position).length() < 1.5 {
                self.current_waypoint = (self.current_waypoint + 1) % self.waypoints.len();
            }
            self.waypoints[self.current_waypoint]
        };

        let to_target = Vec3::new(target.x - self.position.x, 0.0, target.z - self.position.z);
        if to_target.length_squared() > 1e-4 {
            let step = to_target.normalize() * self.speed * dt;
            self.position.x += step.x;
            self.position.z += step.z;
        }
        // Keep the NPC at eye/character height like orbs and the windmill base.
        self.position.y = 1.0;

        self.distance_travelled += (self.position - prev_pos).length();
    }

    /// Net displacement from spawn (metres).
    fn net_moved(&self) -> f32 {
        (self.position - self.spawn_pos).length()
    }

    /// Render the NPC as a small cluster of splats so it is visible in-frame.
    fn splats(&self) -> Vec<GaussianSplat> {
        let mut out = Vec::with_capacity(8);
        // A small vertical capsule of splats — a recognisable little character.
        for iy in 0..4 {
            let y = self.position.y + iy as f32 * 0.4;
            let scale = if iy >= 2 { 0.35 } else { 0.28 };
            out.push(GaussianSplat::volume(
                [self.position.x, y, self.position.z],
                [scale, scale, scale],
                Quat::IDENTITY,
                235,
                self.body_spectral,
            ));
        }
        out
    }
}

struct WalkingSim {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    rasteriser: SoftwareRasteriser,

    // Scene
    terrain_splats: Vec<GaussianSplat>,
    building_splats: Vec<GaussianSplat>,
    tree_splats: Vec<GaussianSplat>,
    building_boxes: Vec<BoundingBox>,

    // Game state
    orbs: Vec<Orb>,
    orbs_collected: u32,
    total_orbs: u32,
    game_time: f32,

    // Character controller (replaces ad-hoc player state)
    cc: CharacterController,
    cc_transform: TransformComponent,
    player_yaw: f32,
    player_pitch: f32,

    // Game UI
    game_ui: GameUI,
    fps_display: f32,
    /// game_time captured when the player won (for the win screen's elapsed
    /// readout). 0.0 until a win occurs.
    win_time: f32,

    // Windmill
    windmill: Windmill,

    // Wandering AI NPC (vox_ai perception->decision path).
    npc: Npc,

    // ---- Day/night cycle ----
    // Smoke accelerates time; the windowed path advances at REAL_SECS_PER_GAME_HOUR.
    time_accel: f32,

    // ---- Biome soundscape ----
    // Channel the biome mixer queues PlaySynth commands onto. Headless-safe: the
    // receiver is drained on the game side; with no audio device nothing is heard
    // but the call path (mix -> reverb -> queue) still runs and is counted.
    audio_tx: std::sync::mpsc::Sender<AudioCommand>,
    audio_rx: std::sync::mpsc::Receiver<AudioCommand>,
    /// Count of biome soundscape beds queued (counted regardless of device).
    soundscape_events: u32,
    /// Human-readable label of the currently active biome.
    current_biome: BiomeKind,
    /// Game-time at which the last soundscape bed was queued (for refresh cadence).
    last_soundscape_t: f32,

    // Dropped physics boxes (KeyQ). Each falls under Rapier gravity and
    // fractures (spectral impact) when it settles on the ground.
    dropped_boxes: Vec<DroppedBox>,
    fracture_events: u32,
    /// Count of impact/collect audio dispatch attempts (counted even when no
    /// output device is present, so headless CI still verifies the call path).
    audio_events: u32,

    // Live spectral GI state. GI-lit scene splats are recomputed at a reduced
    // cadence (see GI_CADENCE) and reused for rendering between updates so the
    // smoke stays within its time budget.
    gi_lit_splats: Vec<GaussianSplat>,
    gi_frame_counter: u32,
    /// Spectral bands of the GI-lit splat nearest the player (for the HUD task).
    pub latest_gi_bands: [u16; 16],

    // Input
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
    last_mouse: Option<(f64, f64)>,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    current_fps: f32,

    // Spectral pipeline
    spectral_fb: SpectralFramebuffer,
    tonemap_settings: ToneMapSettings,

    // CLAS clustering stats
    clas_cluster_count: usize,
    clas_bvh_depth: u32,

    // ---- Atom-budget splat selection ----
    // The static scene (terrain + buildings + trees) concatenated once into one
    // Vec, with an AtomBudgetSelector built over it. Per frame, the selector
    // yields ≤ ATOM_BUDGET indices into `static_splats`; dynamic splats are
    // appended unbudgeted. Replaces the per-frame full-scene clone.
    static_splats: Vec<GaussianSplat>,
    selector: Option<AtomBudgetSelector>,
    /// Reused per-frame selection output (indices + crossfade opacity scale).
    selection: Selection,

    // Scripting
    rhai: RhaiRuntime,
    /// Path to the live, hot-reloadable game script (assets/scripts/walking_sim.rhai).
    script_path: std::path::PathBuf,
    /// Index of the loaded game script in `rhai`, if it loaded successfully.
    game_script_idx: Option<usize>,
    /// Last good orb bob amplitude (metres) read from the script. Cached so a
    /// transient script call failure keeps the last value instead of snapping to 0.
    orb_bob_amplitude: f32,
    /// Last good orb bob speed (rad/s) read from the script.
    orb_bob_speed: f32,
    /// Last good orb pulse depth (fraction of base size) read from the script.
    orb_pulse_depth: f32,
    /// Last good windmill spin multiplier read from the script (1.0 = base rate).
    windmill_speed_mult: f32,
    /// Notification text surfaced when the script fails to recompile (last error).
    script_notification: Option<String>,

    // Unified per-frame simulation driver (EngineLoop). Owns the shared
    // physics world, audio backends (legacy + spatial), and shadow mapper.
    // walking_sim drives the game-minimal subset: physics + audio + animation
    // + shadows (no GI, no scripts).
    loop_: EngineLoop,
}

impl WalkingSim {
    fn new() -> Self {
        let orb_positions = vec![
            Vec3::new(15.0, 1.5, 15.0),
            Vec3::new(-10.0, 1.5, 20.0),
            Vec3::new(25.0, 1.5, -5.0),
            Vec3::new(-20.0, 1.5, -15.0),
            Vec3::new(5.0, 1.5, -25.0),
            Vec3::new(30.0, 1.5, 10.0),
            Vec3::new(-25.0, 1.5, 5.0),
            Vec3::new(0.0, 1.5, 30.0),
            Vec3::new(20.0, 1.5, -20.0),
            Vec3::new(-15.0, 1.5, -25.0),
        ];

        let orbs: Vec<Orb> = orb_positions
            .into_iter()
            .map(|pos| Orb {
                position: pos,
                collected: false,
                bob_phase: pos.x * 0.1,
            })
            .collect();

        // Character controller — start at eye level above ground
        let cc = CharacterController { speed: 8.0, ..Default::default() };
        let cc_transform = TransformComponent {
            position: Vec3::new(0.0, cc.height * 0.5, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };

        // Game UI — start on main menu; player presses Enter to begin
        let game_ui = GameUI { game_state: GameState::MainMenu, ..Default::default() };

        // Unified per-frame simulation driver. EngineLoop owns the shared
        // physics world (built with its own 1km x 1km ground plane), the audio
        // backends, and the shadow mapper — the same instances the editor shell
        // (engine_runner) drives. walking_sim uses the game-minimal mask:
        // physics + audio + animation + shadows, no GI/scripts.
        // Enable live spectral GI on top of the game-minimal subset. GI is run
        // at a reduced cadence in update() to stay within the smoke time budget.
        let mut mask = SystemMask::game_minimal();
        mask.gi = true;
        let mut loop_ = EngineLoop::new(EngineConfig::default(), mask);

        // Match walking_sim's original shadow resolution (128) instead of the
        // loop's editor default (512) so the software-rendered shadow look is
        // unchanged. walking_sim drives this mapper directly with its own fixed
        // SUN_DIR and building-only occluders (see build_scene / update).
        loop_.shadow_mapper = ShadowMapper::new(128);

        // Spatial audio — gracefully silent if no hardware. Reuse the single
        // backend the loop already constructed.
        if loop_.spatial_audio.is_available() {
            println!("[walking_sim] Spatial audio: available");
        } else {
            println!("[walking_sim] Spatial audio: silent mode (no hardware or rodio feature)");
        }

        // Static colliders live on the loop's physics world (the ground plane
        // was already added by EngineLoop::new). Buildings are registered later
        // in build_scene against loop_.physics.
        println!("[walking_sim] Physics: Rapier3D world initialised (ground plane)");

        // Windmill placed at the side of the scene
        let windmill = Windmill::new(Vec3::new(18.0, 0.0, -8.0));

        // Wandering NPC. Spawned near the start->first-orb path (first orb is at
        // (15,_,15)) so the smoke's player walk brings the player within
        // NPC_FLEE_RADIUS and triggers the Patrol->Flee transition. Waypoints
        // form a small patrol loop around the spawn.
        let npc_spawn = Vec3::new(10.0, 1.0, 10.0);
        let npc = Npc::new(
            npc_spawn,
            vec![
                Vec3::new(10.0, 1.0, 10.0),
                Vec3::new(14.0, 1.0, 6.0),
                Vec3::new(8.0, 1.0, 4.0),
                Vec3::new(5.0, 1.0, 9.0),
            ],
        );

        // Biome soundscape command channel (headless-safe; see field docs).
        let (audio_tx, audio_rx) = std::sync::mpsc::channel::<AudioCommand>();

        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            terrain_splats: Vec::new(),
            building_splats: Vec::new(),
            tree_splats: Vec::new(),
            building_boxes: Vec::new(),
            orbs,
            orbs_collected: 0,
            total_orbs: 10,
            game_time: 0.0,
            cc,
            cc_transform,
            player_yaw: 0.0,
            player_pitch: 0.0,
            game_ui,
            fps_display: 0.0,
            win_time: 0.0,
            windmill,
            npc,
            time_accel: 1.0 / REAL_SECS_PER_GAME_HOUR,
            audio_tx,
            audio_rx,
            soundscape_events: 0,
            current_biome: BiomeKind::Grassland,
            last_soundscape_t: -1000.0,
            dropped_boxes: Vec::new(),
            fracture_events: 0,
            audio_events: 0,
            gi_lit_splats: Vec::new(),
            gi_frame_counter: 0,
            latest_gi_bands: [0u16; 16],
            keys_held: std::collections::HashSet::new(),
            mouse_captured: false,
            last_mouse: None,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            spectral_fb: SpectralFramebuffer::new(WIDTH, HEIGHT),
            tonemap_settings: ToneMapSettings::default(),
            clas_cluster_count: 0,
            clas_bvh_depth: 0,
            static_splats: Vec::new(),
            selector: None,
            selection: Selection::new(),
            rhai: RhaiRuntime::new(),
            script_path: resolve_script_path(),
            game_script_idx: None,
            // Defaults mirror the script's shipped constants so behaviour is
            // sane even if the script file is missing on disk.
            orb_bob_amplitude: 0.30,
            orb_bob_speed: 2.0,
            orb_pulse_depth: 0.20,
            windmill_speed_mult: 1.0,
            script_notification: None,
            loop_,
        }
    }

    /// Load (or reload from scratch) the live game script and prime the cached
    /// script-driven values from it. Idempotent-ish: pushes a new script into the
    /// runtime; intended to be called once during setup. Poll-driven hot reload
    /// thereafter swaps the AST in place (see `tick_script`).
    fn load_game_script(&mut self) {
        // Poll for changes ~2x/second (cheap mtime check, not per-frame hashing).
        self.rhai.reload_interval = std::time::Duration::from_millis(500);
        match self.rhai.load_script_file("walking_sim", &self.script_path) {
            Ok(idx) => {
                self.game_script_idx = Some(idx);
                println!(
                    "[walking_sim] Loaded game script: {}",
                    self.script_path.display()
                );
                self.tick_script(); // prime cached values from the freshly loaded script
            }
            Err(e) => {
                self.game_script_idx = None;
                self.script_notification = Some(format!("script load failed: {e}"));
                eprintln!("[walking_sim] {e}");
            }
        }
    }

    /// Hot-reload poll + read script-driven values. Called every frame from
    /// update(). The mtime poll is internally rate-limited (reload_interval), so
    /// this is cheap. On a compile error the runtime keeps the last-good AST and
    /// increments `script_errors`; we surface its message as a notification. On a
    /// per-call failure we keep the last cached value (never snap to 0).
    fn tick_script(&mut self) {
        let reloaded = self.rhai.poll_reload();
        if !reloaded.is_empty() {
            // A successful swap clears any prior error notification.
            self.script_notification = None;
            println!(
                "[walking_sim] Hot-reloaded script (reloads={})",
                self.rhai.script_reloads
            );
        }
        if let Some(err) = &self.rhai.last_error {
            self.script_notification = Some(format!("SCRIPT ERROR (last-good kept): {err}"));
        }

        let Some(idx) = self.game_script_idx else {
            return;
        };
        // Read each tunable via its direct accessor. Keep the last cached value if
        // a call fails (missing/renamed fn) so a partially-broken-but-compiling
        // script can never blank the behaviour.
        if let Some(v) = self.call_script_number(idx, "bob_amplitude") {
            self.orb_bob_amplitude = v;
        }
        if let Some(v) = self.call_script_number(idx, "bob_speed") {
            self.orb_bob_speed = v;
        }
        if let Some(v) = self.call_script_number(idx, "pulse_depth") {
            self.orb_pulse_depth = v;
        }
        if let Some(v) = self.call_script_number(idx, "windmill_speed") {
            self.windmill_speed_mult = v;
        }
    }

    /// Call a zero-arg script function and coerce its return to f32, accepting
    /// either a Rhai float or integer. Returns None on any error (missing fn,
    /// wrong type), so the caller can keep its last-good cached value.
    fn call_script_number(&mut self, idx: usize, fn_name: &str) -> Option<f32> {
        let v = self.rhai.call_fn(idx, fn_name, &[]).ok()?;
        if let Ok(f) = v.as_float() {
            Some(f as f32)
        } else {
            v.as_int().ok().map(|i| i as f32)
        }
    }

    fn build_scene(&mut self) {
        println!("[walking_sim] Building scene...");

        // Terrain
        let vol = vox_terrain::volume::generate_demo_volume(42);
        let materials = vox_terrain::volume::default_volume_materials();
        self.terrain_splats = vox_terrain::volume::volume_to_splats(&vol, &materials, 42);
        println!(
            "[walking_sim]   Terrain: {} splats",
            self.terrain_splats.len()
        );

        // Buildings (with bounding boxes for collision)
        let building_width = 6.0;
        let building_depth = 8.0;
        let building_height = 10.0;
        for i in 0..3 {
            let bx = i as f32 * 12.0 - 12.0;
            let bz = 25.0;
            let b = vox_data::proc_gs_advanced::generate_detailed_building(
                i as u64,
                building_width,
                building_depth,
                2,
                "victorian",
            );
            for s in &b {
                let mut ws = *s;
                ws.position_mut()[0] += bx;
                ws.position_mut()[2] += bz;
                self.building_splats.push(ws);
            }
            self.building_boxes.push(BoundingBox {
                min: Vec3::new(
                    bx - building_width * 0.5,
                    0.0,
                    bz - building_depth * 0.5,
                ),
                max: Vec3::new(
                    bx + building_width * 0.5,
                    building_height,
                    bz + building_depth * 0.5,
                ),
            });
            // Register building in the loop's Rapier physics world
            self.loop_.physics.add_static_collider(
                [bx, building_height * 0.5, bz],
                [building_width * 0.5, building_height * 0.5, building_depth * 0.5],
            );
        }
        println!(
            "[walking_sim]   Buildings: {} splats ({} collision boxes, {} Rapier colliders)",
            self.building_splats.len(),
            self.building_boxes.len(),
            self.loop_.physics.collider_count(),
        );

        // Trees scattered around
        for i in 0..8 {
            let t = vox_data::proc_gs_advanced::generate_tree(100 + i, 7.0, 3.0);
            let angle = i as f32 * 0.8;
            let radius = 15.0 + (i as f32 * 3.0);
            for s in &t {
                let mut ws = *s;
                ws.position_mut()[0] += angle.cos() * radius;
                ws.position_mut()[2] += angle.sin() * radius;
                self.tree_splats.push(ws);
            }
        }
        println!(
            "[walking_sim]   Trees: {} splats",
            self.tree_splats.len()
        );

        let total =
            self.terrain_splats.len() + self.building_splats.len() + self.tree_splats.len();
        println!("[walking_sim] Scene: {} splats total", total);

        // CLAS clustering + atom-budget selector. The concatenated static scene
        // (terrain + buildings + trees) is kept in `static_splats` for the life
        // of the run; the selector indexes into it per frame. The CLAS stats log
        // is preserved (same clusters, now load-bearing via the selector).
        let mut all_splats = self.terrain_splats.clone();
        all_splats.extend_from_slice(&self.building_splats);
        all_splats.extend_from_slice(&self.tree_splats);
        let clusters = clas::build_clusters(&all_splats, 128);
        let bvh = clas::build_cluster_bvh(&clusters);
        let stats = clas::compute_stats(&clusters, &bvh);
        self.clas_cluster_count = stats.cluster_count;
        self.clas_bvh_depth = stats.bvh_depth;
        println!(
            "[walking_sim] CLAS: {} clusters, BVH depth {}, avg {:.0} splats/cluster",
            stats.cluster_count, stats.bvh_depth, stats.avg_splats_per_cluster,
        );

        // Build the atom-budget selector over the static scene (same 128-splat
        // target cluster size as the CLAS log above).
        self.selector = Some(AtomBudgetSelector::build(&all_splats, 128));
        self.static_splats = all_splats;

        // Prime the loop's shadow mapper with the initial camera state.
        // walking_sim drives the mapper directly (not via step_shadows) so the
        // fixed SUN_DIR and building-only occluders preserve the original look.
        let sun_dir = SUN_DIR.normalize();
        self.loop_.shadow_mapper.update(
            self.cc_transform.position,
            Vec3::new(self.player_yaw.sin(), 0.0, -self.player_yaw.cos()).normalize(),
            sun_dir,
        );
        // Render shadow map from building splat positions
        let occluder_positions: Vec<Vec3> = self
            .building_splats
            .iter()
            .map(|s| Vec3::from(s.position()))
            .collect();
        let occluder_radii: Vec<f32> = self
            .building_splats
            .iter()
            .map(|s| s.scale_u())
            .collect();
        self.loop_.shadow_mapper
            .render_shadow_map(&occluder_positions, &occluder_radii);
        println!("[walking_sim] Shadow mapper primed.");

        // Load the live, hot-reloadable game script. Drives orb bob/pulse +
        // windmill speed; editing the file mid-run changes the game (see
        // tick_script / update). Both the windowed flow and the smoke reach this.
        self.load_game_script();
    }

    fn player_pos(&self) -> Vec3 {
        self.cc_transform.position
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.player_yaw.sin() * self.player_pitch.cos(),
            self.player_pitch.sin(),
            -self.player_yaw.cos() * self.player_pitch.cos(),
        )
        .normalize()
    }

    fn generate_orb_splats(&self) -> Vec<GaussianSplat> {
        let mut splats = Vec::new();
        let orb_spd: [u16; 16] = std::array::from_fn(|i| {
            let v = if (6..=13).contains(&i) { 0.9 } else { 0.5 };
            half::f16::from_f32(v).to_bits()
        });

        for orb in &self.orbs {
            if orb.collected {
                continue;
            }

            // Bob amplitude + speed come from the live game script (hot-reloadable).
            let bob_y = (self.game_time * self.orb_bob_speed + orb.bob_phase).sin()
                * self.orb_bob_amplitude;
            let pos = orb.position + Vec3::new(0.0, bob_y, 0.0);

            let rotation_angle = self.game_time * 1.5 + orb.bob_phase;
            let cos_a = rotation_angle.cos();
            let sin_a = rotation_angle.sin();

            let pulse =
                1.0 + (self.game_time * 3.0 + orb.bob_phase).sin() * self.orb_pulse_depth;
            let scale = 0.1 * pulse;

            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        let d = (dx * dx + dy * dy + dz * dz) as f32;
                        if d > 6.0 {
                            continue;
                        }
                        let rx = dx as f32 * 0.15 * cos_a - dz as f32 * 0.15 * sin_a;
                        let rz = dx as f32 * 0.15 * sin_a + dz as f32 * 0.15 * cos_a;
                        splats.push(GaussianSplat::volume(
                            [pos.x + rx, pos.y + dy as f32 * 0.15, pos.z + rz],
                            [scale, scale, scale],
                            Quat::IDENTITY,
                            230,
                            orb_spd,
                        ));
                    }
                }
            }
        }
        splats
    }

    /// Recompute live spectral GI over the nearest-K scene splats around the
    /// player and cache the GI-lit result for rendering. Also stores the
    /// spectral of the GI-lit splat nearest the player into `latest_gi_bands`.
    fn recompute_gi(&mut self) {
        let player = self.player_pos();

        // Nearest-K static subset around the player via the cluster BVH
        // (perf budget — see GI_NEAREST_K). Replaces the old full-scene distance
        // sort: nearest_clusters returns the cluster ids covering ≥ GI_NEAREST_K
        // static splats, and we gather just those clusters' splats. O(k + log V)
        // instead of O(scene · log scene) per GI step.
        let mut scene: Vec<GaussianSplat> = Vec::new();
        if let Some(selector) = self.selector.as_ref() {
            for cid in selector.nearest_clusters(player, GI_NEAREST_K) {
                for &idx in selector.cluster_indices(cid) {
                    scene.push(self.static_splats[idx as usize]);
                }
            }
        } else {
            scene.extend_from_slice(&self.static_splats);
        }

        // Dynamic splats (orbs + windmill) are appended unbudgeted as before.
        scene.extend(self.generate_orb_splats());
        scene.extend_from_slice(&self.windmill.base_splats);
        scene.extend_from_slice(&self.windmill.blade_splats_world);

        // Keep the GI_NEAREST_K budget: trim the gathered subset (now ordered by
        // cluster proximity) by exact distance to the player.
        if scene.len() > GI_NEAREST_K {
            scene.sort_by(|a, b| {
                let da = (Vec3::from(a.position()) - player).length_squared();
                let db = (Vec3::from(b.position()) - player).length_squared();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            scene.truncate(GI_NEAREST_K);
        }

        let hour = self.loop_.time_of_day();
        let lit = self.loop_.step_gi(&scene, hour);

        // Store the spectral of the GI-lit splat nearest the player.
        if let Some(nearest) = lit.iter().min_by(|a, b| {
            let da = (Vec3::from(a.position()) - player).length_squared();
            let db = (Vec3::from(b.position()) - player).length_squared();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.latest_gi_bands = *nearest.spectral();
        }

        self.gi_lit_splats = lit;
    }

    /// Spawn a dynamic physics box 2m in front of and 1.5m above the player.
    /// The box falls under Rapier gravity (stepped in update()) and is drawn as
    /// a splat at its live body position; on ground impact it fractures.
    fn drop_box(&mut self) {
        let pos = self.player_pos() + self.forward() * 2.0 + Vec3::new(0.0, 1.5, 0.0);
        let half_extents = Vec3::splat(0.4);
        let (handle, _collider) = self.loop_.physics.add_dynamic_box(
            [pos.x, pos.y, pos.z],
            [half_extents.x, half_extents.y, half_extents.z],
            2.0,
        );

        // Brittle, glass-like material: sharp alternating bands, low total
        // energy. Brittleness drives the fracture fragment count and the impact
        // audio timbre.
        let material: [u16; 16] = std::array::from_fn(|i| {
            let v: f32 = if i % 2 == 0 { 0.9 } else { 0.02 };
            half::f16::from_f32(v).to_bits()
        });

        self.dropped_boxes.push(DroppedBox {
            handle,
            material,
            half_extents,
            spawn_y: pos.y,
            fractured: false,
            render_scale: half_extents.x,
        });
        println!(
            "[walking_sim] Dropped box #{} at ({:.1},{:.1},{:.1})",
            self.dropped_boxes.len(),
            pos.x,
            pos.y,
            pos.z
        );
    }

    /// Detect dropped boxes that have settled on the ground and apply a spectral
    /// impact fracture to each, exactly once. Returns nothing; updates
    /// `fracture_events`, `audio_events`, and the box's render scale.
    fn update_dropped_boxes(&mut self) {
        for b in &mut self.dropped_boxes {
            if b.fractured {
                continue;
            }
            let Some(pos) = self.loop_.physics.body_position(b.handle) else {
                continue;
            };
            let Some(vel) = self.loop_.physics.body_velocity(b.handle) else {
                continue;
            };
            let fell = b.spawn_y - pos[1] > 1.0;
            // Ground is the y=0 plane; the box has half-extent ~0.4 so it rests
            // near y≈0.4. Treat it as impacted once it has fallen and its
            // vertical speed has decayed to near zero (settled on the ground).
            let settled = pos[1] <= b.half_extents.y + 0.15 && vel[1].abs() < 0.5;
            if !(fell && settled) {
                continue;
            }

            // Impulse ≈ mass * pre-impact speed. The box fell a known height, so
            // derive a real impulse from the fall (v = sqrt(2 g h)) scaled to the
            // Ns range the spectral fracture threshold expects.
            let fall_h = (b.spawn_y - pos[1]).max(0.0);
            let impact_speed = (2.0 * 9.81 * fall_h).sqrt();
            let impulse_ns = (2.0 * impact_speed * 5000.0).max(1.0);

            let impact_pos = Vec3::new(pos[0], pos[1], pos[2]);
            // Build a renderable splat from the material so the fracture can shift
            // its spectral in place, then read back the result.
            let mut splat = GaussianSplat::volume(
                [pos[0], pos[1], pos[2]],
                [b.render_scale, b.render_scale, b.render_scale],
                Quat::IDENTITY,
                230,
                b.material,
            );
            let result = vox_physics::spectral_physics::SpectralPhysics::apply_impact_to_splat(
                &mut splat,
                impact_pos,
                impulse_ns,
            );

            // Visualise the fracture: the box splits into fragments, so each
            // visible fragment shrinks. Scale down by 1/cbrt(fragments) so total
            // volume is roughly conserved across the shattering.
            let frag = result.fragment_count.max(1) as f32;
            b.render_scale *= 1.0 / frag.cbrt();
            // Adopt the spectral-shifted (darkened/cracked) material for rendering.
            b.material = *splat.spectral();
            b.fractured = true;
            self.fracture_events += 1;
            println!(
                "[walking_sim] Box fractured: brittleness={:.2} fragments={} cracks={} impulse={:.0}Ns",
                result.brittleness,
                result.fragment_count,
                result.crack_count(),
                impulse_ns
            );

            // Impact audio: synthesise + play from the material spectral. Guarded
            // (returns 0 with no device); counted regardless so headless CI still
            // exercises the call path.
            let _samples =
                vox_audio::synthesize_and_play_spectral(&b.material, impulse_ns.clamp(0.3, 1.0));
            self.audio_events += 1;
        }
    }

    /// Render splats for the dropped boxes at their live Rapier body positions.
    fn dropped_box_splats(&self) -> Vec<GaussianSplat> {
        let mut out = Vec::with_capacity(self.dropped_boxes.len());
        for b in &self.dropped_boxes {
            if let Some(pos) = self.loop_.physics.body_position(b.handle) {
                out.push(GaussianSplat::volume(
                    pos,
                    [b.render_scale, b.render_scale, b.render_scale],
                    Quat::IDENTITY,
                    230,
                    b.material,
                ));
            }
        }
        out
    }

    /// Classify the biome at the player's position with a simple, honest 3-way
    /// heuristic over scene geometry:
    /// - Forest: near scattered tree splats (within ~10m of a tree).
    /// - Wetland: low-lying terrain near the origin "basin" (radius < 12m), a
    ///   stand-in for the water/marsh at the map centre.
    /// - Grassland: everywhere else (open ground).
    ///
    /// This is deliberately coarse — there is no biome map in the demo terrain,
    /// so we derive an approximate biome from where the dynamic scene props are.
    fn classify_biome(&self) -> BiomeKind {
        let p = self.player_pos();
        // Forest: any tree splat within 10m of the player (XZ).
        let near_tree = self.tree_splats.iter().any(|s| {
            let sp = Vec3::from(s.position());
            let dx = sp.x - p.x;
            let dz = sp.z - p.z;
            (dx * dx + dz * dz) < 10.0 * 10.0
        });
        if near_tree {
            return BiomeKind::Forest;
        }
        // Wetland basin near the map centre.
        if (p.x * p.x + p.z * p.z) < 12.0 * 12.0 {
            return BiomeKind::Wetland;
        }
        BiomeKind::Grassland
    }

    /// Start or refresh the biome ambient soundscape based on the player's
    /// current biome. Queues a fresh ambient bed when the biome changes or the
    /// previous bed has elapsed. Headless-safe: `play_biome_soundscape` only
    /// queues onto our channel (drained below) — no device required.
    fn refresh_soundscape(&mut self) {
        let biome = self.classify_biome();
        let bed_secs = 1.0f32; // each bed is ~1s of ambient; refresh on expiry
        let changed = biome != self.current_biome;
        let expired = self.game_time - self.last_soundscape_t >= bed_secs;
        if !(changed || expired) {
            return;
        }
        self.current_biome = biome;
        let mix = BiomeAmbientMix::for_biome(biome);
        // Outdoor mix (no enclosing room surfaces) -> dead-room dry ambient bed.
        let _len = vox_audio::play_biome_soundscape(&mix, bed_secs, 0.4, &[], &self.audio_tx);
        self.soundscape_events += 1;
        self.last_soundscape_t = self.game_time;
        // Drain the queued command so the channel doesn't grow unbounded. With no
        // audio device there is nothing further to do; this just exercises the
        // full mix->queue path headlessly.
        while self.audio_rx.try_recv().is_ok() {}
    }

    fn update(&mut self, dt: f32) {
        // Only update game logic when Playing
        if self.game_ui.game_state != GameState::Playing {
            return;
        }
        self.game_time += dt;

        // ---------------------------------------------------------------
        // 0a. Live game script: cheap mtime-poll hot-reload (~2x/sec) + read the
        //     script-driven tunables (orb bob amplitude/speed, pulse depth,
        //     windmill speed) into cached host fields. A compile error keeps the
        //     last-good script running and surfaces a notification; it never
        //     crashes the game.
        // ---------------------------------------------------------------
        self.tick_script();

        // ---------------------------------------------------------------
        // 0. Day/night cycle: advance the EngineLoop's time-of-day clock
        //    continuously. `time_accel` is game-hours per real second (1/30 in
        //    windowed play; the smoke bumps it for a visible swing). Wrap at 24h.
        // ---------------------------------------------------------------
        let new_hour = (self.loop_.time_of_day() + dt * self.time_accel).rem_euclid(24.0);
        self.loop_.set_time_of_day(new_hour);

        // ---------------------------------------------------------------
        // 1. CharacterController movement (replaces ad-hoc WASD code)
        // ---------------------------------------------------------------
        let forward_xz =
            Vec3::new(self.player_yaw.sin(), 0.0, -self.player_yaw.cos()).normalize();
        let right_xz = forward_xz.cross(Vec3::Y).normalize();

        let mut move_input = Vec3::ZERO;
        if self.keys_held.contains(&KeyCode::KeyW) {
            move_input += forward_xz;
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            move_input -= forward_xz;
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            move_input -= right_xz;
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            move_input += right_xz;
        }
        if move_input.length_squared() > 0.0 {
            move_input = move_input.normalize();
        }

        let jump_pressed = self.keys_held.contains(&KeyCode::Space);

        character_controller_tick(
            &mut self.cc,
            &mut self.cc_transform,
            move_input,
            jump_pressed,
            dt,
        );

        // Step Rapier physics world (via the shared EngineLoop)
        self.loop_.step_physics(dt);

        // Use Rapier ground raycast for ground detection: cast ray downward from feet
        let feet_y = self.cc_transform.position.y - self.cc.height * 0.5;
        if let Some((_hit_pos, dist)) = self.loop_.physics.raycast(
            [self.cc_transform.position.x, feet_y + 0.1, self.cc_transform.position.z],
            [0.0, -1.0, 0.0],
            0.3, // short ray: just checking if ground is within 0.3m below feet
        ) {
            // Ground detected by Rapier — snap to ground if falling
            if self.cc.velocity.y <= 0.0 {
                let ground_y = feet_y + 0.1 - dist;
                self.cc_transform.position.y = ground_y + self.cc.height * 0.5;
                self.cc.grounded = true;
                if self.cc.velocity.y < 0.0 {
                    self.cc.velocity.y = 0.0;
                }
            }
        }

        // Building collision still applied post-integration (AABB fallback alongside Rapier)
        check_building_collision(&mut self.cc_transform.position, &self.building_boxes);

        // ---------------------------------------------------------------
        // 2. Shadow mapper update each frame (via the shared EngineLoop's
        //    shadow mapper). Driven directly with walking_sim's fixed SUN_DIR
        //    rather than step_shadows. The sun now follows the day/night cycle
        //    (see sun_dir_for_hour) so shadows swing with time of day.
        // ---------------------------------------------------------------
        let sun_dir = sun_dir_for_hour(self.loop_.time_of_day());
        let cam_fwd = self.forward();
        self.loop_.shadow_mapper
            .update(self.cc_transform.position, cam_fwd, sun_dir);

        // ---------------------------------------------------------------
        // 3. Windmill animation tick. The script-driven speed multiplier scales
        //    the advance rate so editing WINDMILL_SPEED in the .rhai file changes
        //    how fast the blades spin without a restart.
        self.windmill.tick(dt * self.windmill_speed_mult);

        // ---------------------------------------------------------------
        // 3b. AI NPC: perception -> decision -> kinematic steering. The NPC
        //     senses the player as a hot fire-band radiance zone and flees when
        //     within NPC_FLEE_RADIUS; otherwise it wanders its patrol loop.
        // ---------------------------------------------------------------
        self.npc.tick(dt, self.cc_transform.position);

        // ---------------------------------------------------------------
        // 3c. Biome soundscape: classify the player's biome and (re)start the
        //     ambient bed via the vox_audio mixer. Headless-safe.
        // ---------------------------------------------------------------
        self.refresh_soundscape();

        // ---------------------------------------------------------------
        // 4. Spatial audio listener + tick (via the shared EngineLoop). This
        //    drives the loop's legacy AudioEngine + SpatialAudioManager and
        //    sets the listener pose from the player camera.
        // ---------------------------------------------------------------
        self.loop_
            .step_audio(dt, self.cc_transform.position, self.forward());

        // ---------------------------------------------------------------
        // 4b. Live spectral GI (reduced cadence + nearest-K subset).
        //     Runs step_gi on the nearest GI_NEAREST_K scene splats around the
        //     player every GI_CADENCE frames, caches the GI-lit result for the
        //     renderer, and stores the spectral of the GI-lit splat nearest the
        //     player into latest_gi_bands (for the HUD task).
        // ---------------------------------------------------------------
        if self.gi_frame_counter.is_multiple_of(GI_CADENCE) {
            self.recompute_gi();
        }
        self.gi_frame_counter = self.gi_frame_counter.wrapping_add(1);

        // ---------------------------------------------------------------
        // 4c. Dropped-box physics fracture (KeyQ boxes settling on the ground).
        // ---------------------------------------------------------------
        self.update_dropped_boxes();

        // ---------------------------------------------------------------
        // 5. Orb collection
        // ---------------------------------------------------------------
        let player_pos = self.cc_transform.position;
        for orb in &mut self.orbs {
            if orb.collected {
                continue;
            }
            let dist = (orb.position - player_pos).length();
            if dist < 2.5 {
                orb.collected = true;
                self.orbs_collected += 1;
                println!(
                    "[walking_sim] Orb collected! {}/{}",
                    self.orbs_collected, self.total_orbs
                );

                // SpatialAudio: play a tone on collect (via the shared loop's
                // spatial audio manager). Each successive orb plays a slightly
                // higher frequency.
                let freq = 440.0 + self.orbs_collected as f32 * 110.0;
                self.loop_.spatial_audio.play_tone(freq, 0.3, 0.8);

                // Modern spectral audio path: synthesise + play an impact tone
                // from the orb's material spectral. Guarded (returns 0 with no
                // device); counted regardless so headless CI verifies the call.
                let orb_spd: [u16; 16] = std::array::from_fn(|i| {
                    let v = if (6..=13).contains(&i) { 0.9 } else { 0.5 };
                    half::f16::from_f32(v).to_bits()
                });
                let _samples = vox_audio::synthesize_and_play_spectral(&orb_spd, 0.6);
                self.audio_events += 1;

                // Also save a WAV for proof (legacy path)
                let sound = vox_audio::synth::generate_collect_sound();
                let path = std::env::temp_dir()
                    .join(format!("ochroma_collect_{}.wav", self.orbs_collected));
                match vox_audio::synth::save_wav(&sound, 44100, &path) {
                    Ok(()) => println!("[walking_sim] Sound: {}", path.display()),
                    Err(e) => eprintln!("[walking_sim] Sound save failed: {}", e),
                }

                if self.orbs_collected >= self.total_orbs {
                    println!(
                        "[walking_sim] YOU WIN! All orbs collected in {:.1} seconds!",
                        self.game_time
                    );
                    self.win_time = self.game_time;
                    self.game_ui.menu_selection = 0;
                    self.game_ui.game_state = GameState::GameOver {
                        message: "YOU WIN!".to_string(),
                    };
                }
            }
        }

        // ---------------------------------------------------------------
        // 6. Update HUD elements with current values
        // ---------------------------------------------------------------
        let pos = self.cc_transform.position;
        self.game_ui.set_text(
            "orbs",
            &format!("ORBS: {}/{}", self.orbs_collected, self.total_orbs),
        );
        self.game_ui.set_text(
            "pos",
            &format!("X:{:.0} Y:{:.1} Z:{:.0}", pos.x, pos.y, pos.z),
        );
        self.game_ui
            .set_text("fps", &format!("FPS: {:.0}", self.current_fps));
    }

    /// Number of selectable options in the currently active menu (0 when
    /// Playing — there is no menu to navigate).
    fn menu_option_count(&self) -> usize {
        match &self.game_ui.game_state {
            GameState::Playing => 0,
            GameState::MainMenu => MAIN_MENU_OPTIONS.len(),
            GameState::Paused => PAUSE_MENU_OPTIONS.len(),
            GameState::GameOver { .. } => WIN_MENU_OPTIONS.len(),
        }
    }

    /// Reset orbs, player, and clocks to a fresh playthrough.
    fn reset_game(&mut self) {
        for orb in &mut self.orbs {
            orb.collected = false;
        }
        self.orbs_collected = 0;
        self.game_time = 0.0;
        self.win_time = 0.0;
        self.cc = CharacterController::default();
        self.cc.speed = 8.0;
        self.cc_transform.position = Vec3::new(0.0, self.cc.height * 0.5, 0.0);
    }

    /// Apply the active menu's selected option (ENTER). Drives the real game
    /// state transitions: main-menu Start/Quit, pause Resume/Quit, win Continue.
    fn activate_menu_selection(&mut self, event_loop: &ActiveEventLoop) {
        let sel = self.game_ui.menu_selection;
        match &self.game_ui.game_state {
            GameState::Playing => {}
            GameState::MainMenu => match sel {
                0 => {
                    // START
                    self.game_ui.game_state = GameState::Playing;
                }
                _ => event_loop.exit(), // QUIT
            },
            GameState::Paused => match sel {
                0 => {
                    // RESUME
                    self.game_ui.game_state = GameState::Playing;
                }
                _ => event_loop.exit(), // QUIT
            },
            GameState::GameOver { .. } => {
                // CONTINUE -> restart a fresh playthrough.
                self.reset_game();
                self.game_ui.menu_selection = 0;
                self.game_ui.game_state = GameState::Playing;
            }
        }
    }

    /// Pixel width of a string at the given bitmap-font scale (mirrors the
    /// private helper in vox_core::game_ui: `len*stride*scale - scale`, last
    /// char has no trailing gap).
    fn label_px_width(text: &str, scale: u32) -> u32 {
        let len = text.chars().count() as u32;
        if len == 0 {
            return 0;
        }
        len * CHAR_STRIDE * scale - scale
    }

    /// Draw a single bitmap label centered horizontally and vertically inside
    /// `rect` ([x, y, w, h]), at the given font scale and color.
    fn draw_label_centered(
        pixels: &mut [[u8; 4]],
        rect: [f32; 4],
        text: &str,
        color: [u8; 3],
        scale: u32,
    ) {
        let tw = Self::label_px_width(text, scale);
        let th = CHAR_H * scale;
        let x = (rect[0] + (rect[2] - tw as f32) * 0.5).max(0.0) as u32;
        let y = (rect[1] + (rect[3] - th as f32) * 0.5).max(0.0) as u32;
        burn_text(pixels, WIDTH, x, y, text, color, scale);
    }

    /// Composite the active game menu (main / pause / win) over `pixels`.
    ///
    /// Shared code path used by BOTH the windowed flow and the smoke. Builds a
    /// [`GameMenu`] layout via the CPU Vello context (dim overlay + panel +
    /// 16-band accent strip + option highlights), rasterises it into the frame,
    /// then stamps bitmap text labels on top inside the menu's title/option
    /// rects. No-op when Playing.
    fn render_menu_overlay(&self, pixels: &mut [[u8; 4]]) {
        let menu = GameMenu::new(WIDTH, HEIGHT);
        let sel = self.game_ui.menu_selection;
        let mut ctx = VelloCtxCpu::new(WIDTH, HEIGHT);

        // Title label + per-option labels chosen per state.
        let (title, options): (String, Vec<String>) = match &self.game_ui.game_state {
            GameState::Playing => return,
            GameState::MainMenu => {
                menu.compose_main(&mut ctx, MAIN_MENU_OPTIONS.len(), sel);
                (
                    "OCHROMA".to_string(),
                    MAIN_MENU_OPTIONS.iter().map(|s| s.to_string()).collect(),
                )
            }
            GameState::Paused => {
                menu.compose_pause(&mut ctx, PAUSE_MENU_OPTIONS.len(), sel);
                (
                    "PAUSED".to_string(),
                    PAUSE_MENU_OPTIONS.iter().map(|s| s.to_string()).collect(),
                )
            }
            GameState::GameOver { message } => {
                menu.compose_win(
                    &mut ctx,
                    self.total_orbs,
                    self.win_time,
                    WIN_MENU_OPTIONS.len(),
                    sel,
                );
                // Title shows the win message; the option row label includes the
                // orb count + elapsed time so the stats are visible on screen.
                (
                    message.clone(),
                    vec![format!(
                        "ORBS {}/{}  {:.0}S  CONTINUE",
                        self.total_orbs, self.total_orbs, self.win_time
                    )],
                )
            }
        };

        // Rasterise the menu layout into the frame.
        ctx.rasterize_into(pixels, WIDTH, HEIGHT);

        // Bitmap labels on top, centered in the menu's published rects.
        Self::draw_label_centered(pixels, menu.title_rect(), &title, [255, 255, 255], 3);
        let rects = menu.option_rects(options.len());
        for (i, label) in options.iter().enumerate() {
            // Selected option: dark text on the bright gold highlight; others:
            // light text on the dim slate highlight.
            let color = if i == sel { [20, 20, 20] } else { [220, 220, 230] };
            Self::draw_label_centered(pixels, rects[i], label, color, 2);
        }
    }

    fn render(&mut self) -> Vec<[u8; 4]> {
        let eye = self.player_pos();
        let target = eye + self.forward();
        let camera = RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                WIDTH as f32 / HEIGHT as f32,
                0.1,
                500.0,
            ),
        };

        // Atom-budget selection of the STATIC scene (terrain + buildings +
        // trees). The selector frustum-culls + LOD-degrades into ≤ ATOM_BUDGET
        // indices, replacing the old per-frame full-scene clone. Dynamic splats
        // are appended unbudgeted below. (Falls back to the raw static set only
        // if the selector was never built, e.g. an empty scene.)
        let mut all: Vec<GaussianSplat> = Vec::new();
        if let Some(selector) = self.selector.as_mut() {
            let mut selection = std::mem::take(&mut self.selection);
            selector.select(&camera, ATOM_BUDGET, &mut selection);
            all.reserve(selection.indices.len());
            for (k, &idx) in selection.indices.iter().enumerate() {
                let mut s = self.static_splats[idx as usize];
                let scale = selection.opacity_scale[k].clamp(0.0, 1.0);
                if scale < 1.0 {
                    s.set_opacity((s.opacity() as f32 * scale).round() as u8);
                }
                all.push(s);
            }
            self.selection = selection;
        } else {
            all.extend_from_slice(&self.static_splats);
        }
        all.extend(self.generate_orb_splats());
        // Windmill base + animated blades
        all.extend_from_slice(&self.windmill.base_splats);
        all.extend_from_slice(&self.windmill.blade_splats_world);
        // Wandering AI NPC body.
        all.extend(self.npc.splats());

        // Live spectral GI: overlay the GI-lit nearest-K subset on top of the
        // raw scene so injected indirect light actually shows in the frame. The
        // GI-lit splats carry the propagated radiance (their spectral differs
        // from the raw input); rendering them brightens the near field.
        all.extend_from_slice(&self.gi_lit_splats);

        // Dropped physics boxes, drawn at their live Rapier body positions.
        all.extend(self.dropped_box_splats());

        let illuminant = Illuminant::d65();

        // Day/night: derive a global sky brightness from the live time-of-day.
        // Applied per-pixel below so night frames are measurably darker than
        // noon. Also tint toward cool blue at night (warm at noon).
        let hour = self.loop_.time_of_day();
        let sky = sky_brightness_for_hour(hour);
        // Warm/cool tint: 1.0 = full warm (noon), 0.0 = cool (night).
        let warmth = ((sky - 0.18) / (1.0 - 0.18)).clamp(0.0, 1.0);
        let tint_r = 0.85 + 0.15 * warmth;
        let tint_g = 0.80 + 0.20 * warmth;
        let tint_b = 0.75 + 0.05 * warmth + 0.20 * (1.0 - warmth);

        // 1. Rasterise
        let fb = self.rasteriser.render(&all, &camera, &illuminant, None);

        // 2. Write to spectral framebuffer with shadow darkening on terrain
        self.spectral_fb.clear();
        for (i, pixel) in fb.pixels.iter().enumerate() {
            let x = (i % WIDTH as usize) as u32;
            let y = (i / WIDTH as usize) as u32;
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            // Shadow: check if this terrain pixel's world position is shadowed.
            // We reconstruct an approximate world position from pixel coordinates
            // via the camera's inverse projection (simple ground-plane approximation).
            // For a software renderer this is an approximation — good enough for a demo.
            let shadow_factor = {
                // Map screen pixel to NDC
                let ndc_x = (x as f32 / WIDTH as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (y as f32 / HEIGHT as f32) * 2.0;

                // Only check pixels in the lower half of the screen (ground region heuristic)
                if ndc_y < -0.1 {
                    // Use a world-space grid probe at estimated ground position
                    let ray_dir_view = Vec3::new(
                        ndc_x / camera.proj.col(0)[0],
                        ndc_y / camera.proj.col(1)[1],
                        -1.0,
                    )
                    .normalize();
                    // Transform ray direction to world space
                    let view_inv = camera.view.inverse();
                    let world_dir =
                        (view_inv * ray_dir_view.extend(0.0)).truncate().normalize();

                    // Intersect with y=0 ground plane
                    let t = if world_dir.y.abs() > 1e-4 {
                        -eye.y / world_dir.y
                    } else {
                        -1.0
                    };
                    if t > 0.0 && t < 200.0 {
                        let world_pos = eye + world_dir * t;
                        if self.loop_.shadow_mapper.is_in_shadow(world_pos, 0.01) {
                            0.6 // darken shadowed terrain by 40%
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            };

            // Combine shadow occlusion with the global day/night sky brightness
            // + warm/cool tint so the whole frame darkens and cools at night.
            let r = r * shadow_factor * sky * tint_r;
            let g = g * shadow_factor * sky * tint_g;
            let b = b * shadow_factor * sky * tint_b;

            let spectral = [
                b * 0.30,
                b * 0.55,
                b * 0.70,
                b * 0.80 + g * 0.05,
                b * 0.50 + g * 0.30,
                g * 0.55 + b * 0.10,
                g * 0.80 + r * 0.03,
                g * 0.90 + r * 0.05,
                g * 0.60 + r * 0.20,
                r * 0.45 + g * 0.25,
                r * 0.75 + g * 0.05,
                r * 0.80,
                r * 0.70,
                r * 0.65,
                r * 0.60,
                r * 0.55,
            ];
            let albedo = spectral;

            self.spectral_fb
                .write_sample(x, y, spectral, 1.0, [0.0, 1.0, 0.0], 0, albedo);
        }

        // 3. Tone map spectral framebuffer to RGBA8
        let mut pixels = tonemap_spectral_framebuffer(
            &self.spectral_fb,
            &illuminant,
            &self.tonemap_settings,
        );

        // 4. Render GameUI HUD text (orbs, position, fps) — only while Playing.
        //    Menu states are drawn by the Vello game-menu overlay below, so the
        //    old bitmap-text HUD elements are suppressed behind a menu.
        if self.game_ui.game_state == GameState::Playing {
            self.game_ui.render_to_pixels(&mut pixels, WIDTH, HEIGHT);
        } else {
            // MainMenu / Paused / GameOver: real Vello game menus (dim overlay +
            // panel + 16-band spectral accent strip + selectable option
            // highlights) with bitmap labels stamped on top. Shared with the
            // windowed flow.
            self.render_menu_overlay(&mut pixels);
        }

        // 5. Vello-style game HUD: live 16-band spectral GI readout (bottom-left)
        //    + orb progress bar (top-left), composited over the frame. The band
        //    energies come straight from the GI step (`latest_gi_bands`), so the
        //    HUD shows the actual indirect radiance at the player's position.
        if self.game_ui.game_state == GameState::Playing {
            let mut hud_ctx = VelloCtxCpu::new(WIDTH, HEIGHT);
            // latest_gi_bands holds f16 BITS (the splat spectral encoding), not
            // linear-quantized u16 — decode before feeding the HUD so a radiance
            // of 1.0 fills a bar exactly.
            let mut energy = [0.0f32; 16];
            for (e, bits) in energy.iter_mut().zip(self.latest_gi_bands) {
                *e = half::f16::from_bits(bits).to_f32().clamp(0.0, 1.0);
            }
            let bands = SpectralRadianceCache::from_f32(energy);
            GameHud::new(WIDTH, HEIGHT).compose(
                &mut hud_ctx,
                &bands,
                self.orbs_collected,
                self.total_orbs,
            );
            hud_ctx.rasterize_into(&mut pixels, WIDTH, HEIGHT);
        }

        // 6. Script error/notification banner. When the live game script fails to
        //    recompile, the game keeps running on the last-good version and this
        //    red banner surfaces the error + the running error count so the
        //    designer sees exactly what broke without the game crashing.
        if self.game_ui.game_state == GameState::Playing
            && let Some(note) = &self.script_notification
        {
            let banner = format!("{}  [errors={}]", note, self.rhai.script_errors);
            burn_text(&mut pixels, WIDTH, 20, HEIGHT - 60, &banner, [255, 80, 80], 2);
        }

        pixels
    }
}

impl ApplicationHandler for WalkingSim {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Ochroma -- Walking Simulator")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("Failed to create window"));

        match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => {
                self.backend = Some(backend);
            }
            Err(e) => {
                eprintln!("[walking_sim] GPU: {}", e);
            }
        }

        self.window = Some(window);
        self.build_scene();

        // Set up GameUI HUD elements (shown when Playing)
        let mut orb_el = UIElement::new("orbs", "ORBS: 0/10", UIPosition::TopLeft);
        orb_el.size = UISize::Normal;
        orb_el.color = [255, 255, 100];
        self.game_ui.add_element(orb_el);

        let pos = self.cc_transform.position;
        let mut pos_el = UIElement::new(
            "pos",
            format!("X:{:.0} Y:{:.1} Z:{:.0}", pos.x, pos.y, pos.z),
            UIPosition::BottomLeft,
        );
        pos_el.size = UISize::Small;
        pos_el.color = [180, 255, 180];
        self.game_ui.add_element(pos_el);

        let mut fps_el = UIElement::new("fps", "FPS: --", UIPosition::TopRight);
        fps_el.size = UISize::Small;
        fps_el.color = [180, 220, 255];
        self.game_ui.add_element(fps_el);

        // Load game config via Rhai scripting
        let config_script = r#"
            let orb_count = 10;
            let player_speed = 8.0;
            let collect_distance = 2.5;
            let orb_bob_speed = 2.0;
            let orb_pulse_speed = 3.0;
            log("Game config loaded via Rhai!");
            orb_count
        "#;
        match self.rhai.load_script("config", config_script) {
            Ok(idx) => {
                if let Ok(()) = self.rhai.run(idx) {
                    println!("[walking_sim] Rhai config loaded");
                }
            }
            Err(e) => eprintln!("[walking_sim] Rhai error: {}", e),
        }

        match self.rhai.eval("2 + 2") {
            Ok(result) => println!("[walking_sim] Rhai eval test: 2 + 2 = {}", result),
            Err(e) => eprintln!("[walking_sim] Rhai eval error: {}", e),
        }

        println!("[walking_sim] Walk around and collect all 10 glowing orbs to win!");
        println!("Controls: UP/DOWN select, ENTER confirm, WASD move, SPACE jump, right-click look, ` Rhai eval, ESC pause/quit");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys_held.insert(key);
                        match key {
                            KeyCode::Escape => {
                                // Playing -> pause (reset selection to Resume).
                                // On a menu, Escape quits.
                                match &self.game_ui.game_state {
                                    GameState::Playing => {
                                        self.game_ui.menu_selection = 0;
                                        self.game_ui.game_state = GameState::Paused;
                                    }
                                    _ => event_loop.exit(),
                                }
                            }
                            // Menu navigation: Up/Down (and W/S as a convenience)
                            // move the selection within the active menu's options.
                            KeyCode::ArrowUp | KeyCode::ArrowDown => {
                                let n = self.menu_option_count();
                                if n > 0 {
                                    let sel = self.game_ui.menu_selection;
                                    self.game_ui.menu_selection = match key {
                                        KeyCode::ArrowUp => (sel + n - 1) % n,
                                        _ => (sel + 1) % n,
                                    };
                                }
                            }
                            KeyCode::Enter => {
                                self.activate_menu_selection(event_loop);
                            }
                            KeyCode::KeyQ
                                // Drop a dynamic physics box in front of the
                                // player (only while actually playing).
                                if self.game_ui.game_state == GameState::Playing => {
                                    self.drop_box();
                                }
                            KeyCode::Backquote => {
                                let expr = "42 * 2 + 1";
                                match self.rhai.eval(expr) {
                                    Ok(result) => {
                                        println!("[rhai-console] {} = {}", expr, result)
                                    }
                                    Err(e) => eprintln!("[rhai-console] Error: {}", e),
                                }
                            }
                            _ => {}
                        }
                    } else {
                        self.keys_held.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput {
                state,
                button: winit::event::MouseButton::Right,
                ..
            } => {
                self.mouse_captured = state == ElementState::Pressed;
                self.last_mouse = None;
                if let Some(w) = &self.window {
                    w.set_cursor_visible(!self.mouse_captured);
                }
            }

            WindowEvent::CursorMoved { position, .. }
                if self.mouse_captured => {
                    if let Some((lx, ly)) = self.last_mouse {
                        self.player_yaw += (position.x - lx) as f32 * 0.003;
                        self.player_pitch = (self.player_pitch
                            - (position.y - ly) as f32 * 0.003)
                            .clamp(-1.5, 1.5);
                    }
                    self.last_mouse = Some((position.x, position.y));
                }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                self.update(dt);
                let pixels = self.render();

                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&pixels, WIDTH, HEIGHT);
                }

                // FPS tracking
                self.frame_count += 1;
                let fps_elapsed = now.duration_since(self.fps_timer).as_secs_f32();
                if fps_elapsed >= 1.0 {
                    self.current_fps = self.frame_count as f32 / fps_elapsed;
                    self.fps_display = self.current_fps;
                    if let Some(w) = &self.window {
                        w.set_title(&format!(
                            "Ochroma Walking Sim -- {:.0} FPS | Orbs: {}/{} | CLAS: {} clusters | {}",
                            self.current_fps,
                            self.orbs_collected,
                            self.total_orbs,
                            self.clas_cluster_count,
                            match &self.game_ui.game_state {
                                GameState::Playing => "Playing",
                                GameState::MainMenu => "Main Menu",
                                GameState::Paused => "Paused",
                                GameState::GameOver { .. } => "YOU WIN!",
                            }
                        ));
                    }
                    self.frame_count = 0;
                    self.fps_timer = now;
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

/// Headless smoke test: run the REAL per-frame simulation + software render
/// without a window/GPU. Proves the unified EngineLoop + game logic + render
/// actually run on a headless box. Exits 0 on success, panics (non-zero) if the
/// loop/render is broken.
fn run_smoke() {
    println!("[walking_sim] === HEADLESS SMOKE MODE (no window/GPU) ===");

    // Build the app exactly as `resumed()` would, minus the window + GPU surface.
    // WalkingSim::new() already constructs the EngineLoop + scene state; only the
    // window/backend are created in resumed(), and render() never touches them.
    let mut app = WalkingSim::new();
    app.build_scene();

    // Replicate the non-window HUD setup from resumed() so render() draws the HUD.
    {
        let mut orb_el = UIElement::new("orbs", "ORBS: 0/10", UIPosition::TopLeft);
        orb_el.size = UISize::Normal;
        orb_el.color = [255, 255, 100];
        app.game_ui.add_element(orb_el);

        let pos = app.cc_transform.position;
        let mut pos_el = UIElement::new(
            "pos",
            format!("X:{:.0} Y:{:.1} Z:{:.0}", pos.x, pos.y, pos.z),
            UIPosition::BottomLeft,
        );
        pos_el.size = UISize::Small;
        pos_el.color = [180, 255, 180];
        app.game_ui.add_element(pos_el);

        let mut fps_el = UIElement::new("fps", "FPS: --", UIPosition::TopRight);
        fps_el.size = UISize::Small;
        fps_el.color = [180, 220, 255];
        app.game_ui.add_element(fps_el);
    }

    // Bypass the ENTER-on-menu gate: start the game directly.
    app.game_ui.game_state = GameState::Playing;

    // Accelerate the day/night clock so a visible swing happens inside the
    // 160-frame run (windowed play uses the slow real-time rate).
    app.time_accel = SMOKE_TIME_MULTIPLIER / REAL_SECS_PER_GAME_HOUR;

    let spawn_pos = app.cc_transform.position;
    let windmill_anim_start = app.windmill.anim_time;
    let dt = 1.0 / 60.0; // fixed 60Hz step
    let total_frames = 160u32;
    // Inject "walk forward" input for nearly the whole run so the real
    // CharacterController movement code carries the player into an orb (nearest
    // orb is ~21m out; at 8 m/s collection radius 2.5m is reached by ~frame 140).
    let walk_frames = total_frames - 5;

    // Capture GI-lit splats input vs output to prove radiance was injected.
    // After the first GI step, latest_gi_bands should be non-zero and the
    // GI-lit splats differ from the raw scene input.

    let mut box_spawn_y = 0.0f32;
    for frame in 0..total_frames {
        // Drop a physics box early so it has time to fall + fracture before the
        // run ends. The windowed path triggers this from the KeyQ handler; the
        // smoke calls the same shared drop_box() directly.
        if frame == 10 {
            app.drop_box();
            box_spawn_y = app
                .dropped_boxes
                .last()
                .map(|b| b.spawn_y)
                .expect("box should exist after drop_box");
        }
        if frame < walk_frames {
            // Aim yaw at the nearest uncollected orb, then hold W. Movement is
            // performed by the REAL CharacterController code inside update().
            let player = app.cc_transform.position;
            if let Some(orb) = app
                .orbs
                .iter()
                .filter(|o| !o.collected)
                .min_by(|a, b| {
                    let da = (a.position - player).length_squared();
                    let db = (b.position - player).length_squared();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let to_orb = orb.position - player;
                // forward_xz used by update() is (sin(yaw), 0, -cos(yaw)); solve
                // for the yaw that points it at the orb in the XZ plane.
                app.player_yaw = to_orb.x.atan2(-to_orb.z);
            }
            app.keys_held.insert(KeyCode::KeyW);
        } else {
            app.keys_held.remove(&KeyCode::KeyW);
        }

        app.update(dt);
        // Render every frame to exercise the full compositor path with evolving
        // state; the frame checked below is re-captured after the script phases.
        app.render();
    }

    // ===================================================================
    // ATOM BUDGET SMOKE
    // -------------------------------------------------------------------
    // Exercise the AtomBudgetSelector on the live static scene from the
    // player's current camera. Prints the two "Done When" lines (budget 24000
    // then 2000) and asserts: selected ≤ budget, frustum culling did real work,
    // ≥ 2 non-zero LOD histogram buckets, and select_us < 2000.
    // ===================================================================
    {
        // Probe camera: inside the scene at eye height, looking across it. The
        // forward half of the static scene falls in the frustum (the rest is
        // culled, so C > 0), near clusters render at L0 and far ones degrade —
        // and the summed L0 count exceeds the budget, so budget=2000 must
        // degrade LOD globally vs budget=24000.
        let eye = Vec3::new(0.0, 6.0, -30.0);
        let target = Vec3::new(0.0, 4.0, 40.0);
        let camera = RenderCamera {
            view: Mat4::look_at_rh(eye, target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                WIDTH as f32 / HEIGHT as f32,
                0.1,
                500.0,
            ),
        };
        let selector = app
            .selector
            .as_mut()
            .expect("selector must be built by build_scene");
        let total_clusters = selector.cluster_count();
        let total_static = app.static_splats.len();

        // Warm-up: prime the selector's internal scratch vecs + CPU caches for
        // this camera's working-set size (the per-frame loop above selected from
        // the player's near-empty view). The design's < 2 ms budget is a
        // steady-state figure; the measured selects below reflect that.
        let mut warm = Selection::new();
        selector.select(&camera, 24_000, &mut warm);

        for &budget in &[24_000usize, 2_000usize] {
            let mut sel = Selection::new();
            let stats = selector.select(&camera, budget, &mut sel);
            println!(
                "[walking_sim] ATOM BUDGET: budget={} selected={} of {} clusters_visible={}/{} clusters_culled={} lod_histogram=[L0:{} L1:{} L2:{} L3:{}] select_us={}",
                stats.budget,
                stats.selected,
                total_static,
                stats.clusters_visible,
                total_clusters,
                stats.clusters_culled,
                stats.lod_histogram[0],
                stats.lod_histogram[1],
                stats.lod_histogram[2],
                stats.lod_histogram[3],
                stats.select_us,
            );

            assert!(
                stats.selected <= budget,
                "selected {} exceeds budget {}",
                stats.selected,
                budget
            );
            assert!(
                stats.clusters_culled > 0 || stats.clusters_visible < total_clusters,
                "frustum culling did no work: visible={}/{} culled={}",
                stats.clusters_visible,
                total_clusters,
                stats.clusters_culled
            );
            let nonzero_lod_buckets =
                stats.lod_histogram.iter().filter(|&&c| c > 0).count();
            assert!(
                nonzero_lod_buckets >= 2,
                "LOD histogram has {} non-zero buckets (< 2): {:?} — distance-LOD not driving selection",
                nonzero_lod_buckets,
                stats.lod_histogram
            );
            assert!(
                stats.select_us < 2000,
                "select took {} us (>= 2000 us / 2 ms budget)",
                stats.select_us
            );
        }
    }

    // ===================================================================
    // SCRIPT HOT-RELOAD SMOKE
    // -------------------------------------------------------------------
    // Prove that editing assets/scripts/walking_sim.rhai mid-run changes the
    // game without restart, that a broken script never crashes the game (it
    // keeps the last-good behaviour + counts the error), and measure the REAL
    // script-driven quantity (orb bob amplitude) before/after.
    // ===================================================================

    // In the smoke, frames are instant (no 60Hz wall-clock), so the 500ms mtime
    // poll gate would never fire. Drop the interval to 0 so every tick_script
    // polls. (Windowed play keeps the cheap 500ms cadence.)
    app.rhai.reload_interval = std::time::Duration::from_millis(0);

    // Back up the real script so the smoke can rewrite it in place (the runtime
    // watches this exact path) and restore it afterwards — we never leave the
    // repo's shipped script clobbered.
    let script_path = app.script_path.clone();
    let original_script =
        std::fs::read_to_string(&script_path).expect("game script must exist for hot-reload smoke");

    // Guarantee orb #0 is present (uncollected) so the amplitude measurement
    // always has a splat cluster to read — the main 160-frame walk may have
    // already collected it.
    app.orbs[0].collected = false;
    app.orbs[0].bob_phase = 0.0; // crest is reachable within the sweep window

    // Measure the script-driven orb bob amplitude from the REAL rendered orb
    // splats: isolate orb #0 (mark every other orb collected for the duration of
    // the sweep), sweep game_time over a full bob period, and take the peak
    // excursion of orb #0's splat-cluster CENTROID from its rest height. The
    // cluster's local dy offsets are symmetric, so the centroid Y tracks the pure
    // bob — no contamination from the cluster shape. This is the on-screen motion
    // the script drives, not the cached constant.
    let measure_bob_amplitude = |app: &mut WalkingSim| -> f32 {
        // Snapshot + isolate orb #0.
        let saved_collected: Vec<bool> = app.orbs.iter().map(|o| o.collected).collect();
        for (i, o) in app.orbs.iter_mut().enumerate() {
            o.collected = i != 0;
        }
        let rest_y = app.orbs[0].position.y;
        let saved_t = app.game_time;
        let mut peak = 0.0f32;
        let mut t = 0.0f32;
        while t < 4.0 {
            app.game_time = t;
            let splats = app.generate_orb_splats(); // only orb #0 emits now
            if !splats.is_empty() {
                let sum_y: f32 = splats.iter().map(|s| s.position()[1]).sum();
                let centroid_y = sum_y / splats.len() as f32;
                peak = peak.max((centroid_y - rest_y).abs());
            }
            t += 1.0 / 120.0;
        }
        // Restore.
        app.game_time = saved_t;
        for (o, &c) in app.orbs.iter_mut().zip(saved_collected.iter()) {
            o.collected = c;
        }
        app.orbs[0].collected = false; // keep orb #0 alive for later phases
        peak
    };

    // --- Phase (a): shipped script. Run ~40 frames, then measure amplitude. ---
    for _ in 0..40 {
        app.update(dt);
    }
    let bob_amp_before = measure_bob_amplitude(&mut app);
    let reloads_before = app.rhai.script_reloads;

    // --- Phase (b): rewrite the script with a CHANGED amplitude, run ~40 more
    //     frames, then re-measure. The measured behaviour must change accordingly.
    // Rewrite the literal returned by `bob_amplitude()` (the single source of
    // truth the host reads): 0.30 -> 1.20.
    let changed_script = original_script.replace("    0.30\n", "    1.20\n");
    assert_ne!(
        changed_script, original_script,
        "smoke could not find the bob_amplitude literal (0.30) to rewrite"
    );
    // Sleep before the write so the new mtime is strictly greater than the one
    // recorded at load time (some filesystems have coarse mtime resolution).
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&script_path, &changed_script).expect("rewrite script");
    for _ in 0..40 {
        app.update(dt);
    }
    let bob_amp_after = measure_bob_amplitude(&mut app);
    let reloads_after = app.rhai.script_reloads;
    let script_reloads_observed = reloads_after.saturating_sub(reloads_before);

    // --- Phase (c): write a deliberately BROKEN script. The game must keep
    //     running on the last-good (amplitude 1.20) behaviour and script_errors
    //     must rise.
    let _errors_before = app.rhai.script_errors;
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&script_path, "fn bob_amplitude() { 1.20 ") // unbalanced brace
        .expect("write broken script");
    for _ in 0..40 {
        app.update(dt); // must NOT panic
    }
    let bob_amp_broken = measure_bob_amplitude(&mut app);
    let script_errors_observed = app.rhai.script_errors;

    // --- Phase (d): restore the good script. ---
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&script_path, &original_script).expect("restore script");
    app.update(dt);
    let bob_amp_restored = measure_bob_amplitude(&mut app);

    // The script phases above ran ~120 extra updates — the player kept walking
    // and may have collected more orbs since `last_pixels` was captured, so the
    // HUD orb-bar check below would compare a CURRENT fill rect against a STALE
    // frame. Re-render so the checked frame matches the current counters.
    let last_pixels = app.render();

    let final_pos = app.cc_transform.position;
    let moved = (final_pos - spawn_pos).length();
    let windmill_advanced = (app.windmill.anim_time - windmill_anim_start).abs();

    // Dropped-box fall distance (proof the box actually fell under gravity).
    let box_final_y = app
        .dropped_boxes
        .last()
        .and_then(|b| app.loop_.physics.body_position(b.handle))
        .map(|p| p[1])
        .unwrap_or(box_spawn_y);
    let box_fell = box_spawn_y - box_final_y;

    // GI proof: how many of latest_gi_bands are non-zero, and the max band.
    let gi_nonzero_bands = app.latest_gi_bands.iter().filter(|&&b| b > 0).count();
    let gi_max_band = app.latest_gi_bands.iter().copied().max().unwrap_or(0);

    // Verify GI actually injected radiance: run one more GI step on a known
    // scene snapshot and confirm the GI-lit output differs from the input.
    let gi_input = {
        let mut s = app.terrain_splats.clone();
        s.extend_from_slice(&app.building_splats);
        s.truncate(GI_NEAREST_K);
        s
    };
    let gi_output = app.loop_.step_gi(&gi_input, app.loop_.time_of_day());
    let gi_changed = gi_input
        .iter()
        .zip(gi_output.iter())
        .any(|(a, b)| a.spectral() != b.spectral());

    // Count non-black pixels in the final frame.
    let non_black = last_pixels
        .iter()
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count();
    // Count distinct RGB colors — a flat/blank fill has ~1 color; a real scene
    // (terrain + sky + splats + HUD) has many. This is a much stronger "the scene
    // actually rendered" signal than non-black alone.
    let distinct_colors = last_pixels
        .iter()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<std::collections::HashSet<_>>()
        .len();

    // --- Day/night luminance: render one noon frame and one midnight frame
    //     from the SAME camera/scene state and compare mean luminance. Uses the
    //     shared render() path (set_time_of_day drives sky brightness in render).
    let mean_luma = |pixels: &[[u8; 4]]| -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }
        let sum: f64 = pixels
            .iter()
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .sum();
        (sum / pixels.len() as f64) as f32
    };
    // Force Playing so the day/night comparison frames carry no menu overlay
    // (if all orbs were collected the state would be GameOver, which now dims).
    app.game_ui.game_state = GameState::Playing;
    app.loop_.set_time_of_day(12.0);
    let noon_pixels = app.render();
    let noon_luma = mean_luma(&noon_pixels);
    app.loop_.set_time_of_day(0.0);
    let midnight_pixels = app.render();
    let midnight_luma = mean_luma(&midnight_pixels);
    let day_night_ratio = if midnight_luma > 0.01 { noon_luma / midnight_luma } else { f32::INFINITY };

    // --- Game menus: render the SAME scene/camera under three states from a
    //     fixed time-of-day and compare. Proves the Vello game-menu overlay
    //     (dim overlay + 16-band spectral accent strip + selected-option
    //     highlight) composites over the live frame in MainMenu and Paused.
    app.loop_.set_time_of_day(12.0);
    // Baseline Playing frame (no menu overlay).
    app.game_ui.game_state = GameState::Playing;
    let playing_pixels = app.render();
    let playing_luma = mean_luma(&playing_pixels);

    // MainMenu frame (Start selected = index 0).
    app.game_ui.game_state = GameState::MainMenu;
    app.game_ui.menu_selection = 0;
    let mainmenu_pixels = app.render();
    let mainmenu_luma = mean_luma(&mainmenu_pixels);

    // Paused frame (Resume selected = index 0).
    app.game_ui.game_state = GameState::Paused;
    app.game_ui.menu_selection = 0;
    let paused_pixels = app.render();
    let paused_luma = mean_luma(&paused_pixels);

    // Restore Playing so any later state queries are consistent.
    app.game_ui.game_state = GameState::Playing;

    // Accent-strip saturation: sample the centre of band 8 (yellow — strongly
    // saturated) in the main-menu frame and confirm it is a saturated color
    // (max channel - min channel large), proving the spectral strip painted.
    let menu = GameMenu::new(WIDTH, HEIGHT);
    let strip = menu.accent_strip_rect();
    let cell_w = strip[2] / 16.0;
    let accent_y = (strip[1] + strip[3] * 0.5) as u32;
    let accent_x = (strip[0] + cell_w * 8.5) as u32; // band 8 centre
    let accent_px = mainmenu_pixels[(accent_y * WIDTH + accent_x) as usize];
    let menu_accent_sat = accent_px.iter().take(3).copied().max().unwrap_or(0) as i32
        - accent_px.iter().take(3).copied().min().unwrap_or(0) as i32;

    // Selected-option highlight: the selected (index 0) option rect must be
    // brighter than an unselected one (index 1) in the main-menu frame.
    let opt_rects = menu.option_rects(MAIN_MENU_OPTIONS.len());
    // Sample near the left edge of each option rect (highlight fill, clear of
    // the centered bitmap label) so the label glyphs don't skew the reading.
    let opt_fill_luma = |r: [f32; 4], pixels: &[[u8; 4]]| -> f32 {
        let cx = (r[0] + r[2] * 0.08) as u32;
        let cy = (r[1] + r[3] * 0.5) as u32;
        let p = pixels[(cy * WIDTH + cx) as usize];
        0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
    };
    let menu_sel_luma = opt_fill_luma(opt_rects[0], &mainmenu_pixels);
    let menu_unsel_luma = opt_fill_luma(opt_rects[1], &mainmenu_pixels);

    // --- AI NPC stats (perception->decision path). ---
    let npc_moved = app.npc.net_moved();
    let npc_distance = app.npc.distance_travelled;
    let npc_state_changes = app.npc.state_changes;
    let npc_final_state = app.npc.last_state.as_str();

    // --- Biome soundscape stats. ---
    let biome_label = format!("{:?}", app.current_biome);
    let soundscape_events = app.soundscape_events;

    // Write the final frame to a PPM (P6).
    let ppm_path = "/tmp/ochroma_walking_sim_smoke.ppm";
    {
        let mut data = format!("P6\n{} {}\n255\n", WIDTH, HEIGHT).into_bytes();
        for p in &last_pixels {
            data.push(p[0]);
            data.push(p[1]);
            data.push(p[2]);
        }
        std::fs::write(ppm_path, &data).expect("failed to write smoke PPM");
    }

    println!(
        "[walking_sim] SMOKE SUMMARY: frames={} final_pos=({:.2},{:.2},{:.2}) orbs={}/{} non_black_px={}/{} distinct_colors={} windmill_dt={:.2} moved={:.2} gi_nonzero_bands={} gi_max_band={} gi_changed={} drops={} box_fell={:.2}m fracture_events={} audio_events={} noon_luma={:.2} midnight_luma={:.2} day_night_ratio={:.2}x biome={} soundscape_events={} npc_moved={:.2} npc_distance={:.2} npc_state_changes={} npc_state={} menu_playing_luma={:.2} menu_mainmenu_luma={:.2} menu_paused_luma={:.2} menu_accent_sat={} menu_sel_luma={:.2} menu_unsel_luma={:.2} script_reloads={} script_errors={} bob_amp_before={:.3} bob_amp_after={:.3} bob_amp_broken={:.3} bob_amp_restored={:.3} ppm={}",
        total_frames,
        final_pos.x,
        final_pos.y,
        final_pos.z,
        app.orbs_collected,
        app.total_orbs,
        non_black,
        last_pixels.len(),
        distinct_colors,
        windmill_advanced,
        moved,
        gi_nonzero_bands,
        gi_max_band,
        gi_changed,
        app.dropped_boxes.len(),
        box_fell,
        app.fracture_events,
        app.audio_events,
        noon_luma,
        midnight_luma,
        day_night_ratio,
        biome_label,
        soundscape_events,
        npc_moved,
        npc_distance,
        npc_state_changes,
        npc_final_state,
        playing_luma,
        mainmenu_luma,
        paused_luma,
        menu_accent_sat,
        menu_sel_luma,
        menu_unsel_luma,
        script_reloads_observed,
        script_errors_observed,
        bob_amp_before,
        bob_amp_after,
        bob_amp_broken,
        bob_amp_restored,
        ppm_path,
    );

    // --- Assertions: real frame + advanced sim (panic => non-zero exit). ---
    let total_px = (WIDTH * HEIGHT) as usize;
    assert_eq!(
        last_pixels.len(),
        total_px,
        "render() produced {} pixels, expected {}",
        last_pixels.len(),
        total_px
    );
    // Scene actually rendered (not a blank buffer): require a substantial fraction
    // of the frame to be non-black.
    let min_non_black = total_px / 20; // >5% of the frame
    assert!(
        non_black >= min_non_black,
        "frame too empty: {} non-black px (< {} required) — render likely broken",
        non_black,
        min_non_black
    );
    // Real geometry rendered, not a uniform fill: require many distinct colors
    // (terrain shading + sky + splats + HUD text all contribute).
    assert!(
        distinct_colors >= 16,
        "frame has only {} distinct colors — looks like a flat fill, not a real render",
        distinct_colors
    );
    // Vello HUD composited: sample the center of the orb progress bar's FILL
    // (>=1 orb collected by now) and require the amber fill to dominate — proves
    // GameHud::compose + rasterize_into actually wrote into the final frame.
    if app.orbs_collected > 0 {
        let fill = GameHud::new(WIDTH, HEIGHT)
            .orb_bar_fill_rect(app.orbs_collected, app.total_orbs);
        // The software rasteriser composites the amber fill with per-pixel alpha
        // coverage, so an individual centre pixel can land in a gap. Scan the
        // fill's mid-row and require that a substantial fraction of it reads amber
        // (high red, red > blue) — proves GameHud::compose + rasterize_into
        // actually painted the bar, robustly to single-pixel coverage gaps.
        let cy = (fill[1] + fill[3] / 2.0) as u32;
        let x0 = fill[0] as u32;
        let x1 = (fill[0] + fill[2]) as u32;
        let mut amber = 0u32;
        let mut total = 0u32;
        for x in x0..x1 {
            let px = last_pixels[(cy * WIDTH + x) as usize];
            total += 1;
            if px[0] > 120 && px[0] > px[2] {
                amber += 1;
            }
        }
        assert!(
            total > 0 && amber * 4 >= total,
            "orb-bar fill row at y={cy} has only {amber}/{total} amber px — HUD compositing broken",
        );
    }
    // Sim advanced: the windmill always animates, the player walked, and orbs were
    // collected. Require an evolving quantity to have changed.
    assert!(
        windmill_advanced > 0.0,
        "windmill did not animate — sim did not advance"
    );
    assert!(
        moved > 1.0 || app.orbs_collected > 0,
        "player did not move ({:.2}m) and collected no orbs — movement/physics broken",
        moved
    );

    // --- New feature assertions (GI, drop, fracture, audio). ---
    // 1. Live spectral GI: at least one band non-zero AND GI actually changed
    //    the splats (radiance was injected, output != input).
    assert!(
        gi_nonzero_bands > 0 && gi_max_band > 0,
        "GI produced no lit bands: nonzero={} max={} — GI not running",
        gi_nonzero_bands,
        gi_max_band
    );
    assert!(
        gi_changed,
        "GI-lit splats are identical to input — no radiance was injected"
    );

    // 2 + 3. Drop + fracture: the box fell > 1m AND fractured at least once.
    assert!(
        box_fell > 1.0,
        "dropped box did not fall: only {:.2}m (spawn_y={:.2}, final_y={:.2})",
        box_fell,
        box_spawn_y,
        box_final_y
    );
    assert!(
        app.fracture_events >= 1,
        "no fracture events: box never registered a ground impact"
    );

    // 4. Impact + collect audio: at least one fracture impact + one orb collect.
    assert!(
        app.audio_events >= 2,
        "audio_events={} (< 2 required: need >=1 fracture + >=1 orb collect)",
        app.audio_events
    );

    // --- Day/night cycle: the noon frame must be measurably brighter than the
    //     midnight frame (>= 1.5x mean luminance), proving the sky-brightness
    //     modulation in the shared render() path actually fires off time-of-day.
    assert!(
        noon_luma > 0.0 && midnight_luma > 0.0,
        "day/night luminance broken: noon={:.3} midnight={:.3}",
        noon_luma,
        midnight_luma
    );
    assert!(
        day_night_ratio >= 1.5,
        "noon ({:.2}) not >= 1.5x brighter than midnight ({:.2}); ratio={:.2}x — day/night not driving render",
        noon_luma,
        midnight_luma,
        day_night_ratio
    );

    // --- Biome soundscape: at least one ambient bed was queued via the mixer.
    assert!(
        soundscape_events >= 1,
        "soundscape_events={} (< 1) — biome soundscape mixer never ran",
        soundscape_events
    );

    // --- AI NPC: it moved a real distance AND its AI state changed at least
    //     once (Patrol/Wander <-> Flee as the player approached/left).
    assert!(
        npc_moved > 1.0,
        "NPC net displacement {:.2}m (<= 1m) — NPC steering did not run",
        npc_moved
    );
    assert!(
        npc_state_changes >= 1,
        "NPC behaviour state never changed ({} changes) — perception->decision path inert",
        npc_state_changes
    );

    // --- Game menus: the MainMenu and Paused frames must be measurably DIMMER
    //     than the same scene rendered while Playing (the full-screen dim
    //     overlay), the spectral accent strip must show a saturated color, and
    //     the selected option highlight must be brighter than an unselected one.
    assert!(
        playing_luma > 0.0,
        "playing baseline frame is black ({playing_luma}) — render broken"
    );
    assert!(
        mainmenu_luma < playing_luma,
        "main-menu frame ({mainmenu_luma:.2}) not dimmer than playing ({playing_luma:.2}) — dim overlay missing",
    );
    assert!(
        paused_luma < playing_luma,
        "paused frame ({paused_luma:.2}) not dimmer than playing ({playing_luma:.2}) — dim overlay missing",
    );
    assert!(
        menu_accent_sat > 50,
        "menu accent strip pixel not saturated (max-min={menu_accent_sat} <= 50) — 16-band spectral strip missing",
    );
    assert!(
        menu_sel_luma > menu_unsel_luma + 20.0,
        "selected option ({menu_sel_luma:.2}) not clearly brighter than unselected ({menu_unsel_luma:.2}) — selection highlight broken",
    );

    // --- Script hot-reload: editing the .rhai file mid-run changed the measured
    //     orb bob amplitude (0.30 -> 1.20 in the file) via a real reload, a broken
    //     edit kept the last-good behaviour (game did not crash, error counted),
    //     and the restore brought the original amplitude back.
    assert!(
        bob_amp_before > 0.0,
        "shipped-script orb bob amplitude measured as 0 — script not driving orbs"
    );
    assert!(
        (bob_amp_before - 0.30).abs() < 0.05,
        "shipped amplitude {:.3} != ~0.30 — script constant not applied",
        bob_amp_before
    );
    assert!(
        script_reloads_observed >= 1,
        "no hot-reload observed ({script_reloads_observed}) — file edit did not reload"
    );
    // The CHANGED file (0.30 -> 1.20) must roughly quadruple the measured bob.
    assert!(
        bob_amp_after > bob_amp_before * 2.0,
        "edited amplitude {:.3} not clearly larger than original {:.3} — hot-reload had no effect",
        bob_amp_after,
        bob_amp_before
    );
    assert!(
        (bob_amp_after - 1.20).abs() < 0.1,
        "edited amplitude {:.3} != ~1.20 — reloaded script not applied",
        bob_amp_after
    );
    // The BROKEN edit must NOT change behaviour (last-good 1.20 retained) and
    // must have incremented the error counter — the game kept running.
    assert!(
        script_errors_observed >= 1,
        "broken script did not raise script_errors ({script_errors_observed}) — error path inert"
    );
    assert!(
        (bob_amp_broken - bob_amp_after).abs() < 0.05,
        "broken script changed behaviour ({:.3} vs last-good {:.3}) — last-good not preserved",
        bob_amp_broken,
        bob_amp_after
    );
    // Restoring the good script returns the original amplitude.
    assert!(
        (bob_amp_restored - bob_amp_before).abs() < 0.05,
        "restored amplitude {:.3} != original {:.3} — restore reload failed",
        bob_amp_restored,
        bob_amp_before
    );

    println!("[walking_sim] SMOKE PASS: loop + game logic + render verified headlessly.");
}

fn main() {
    if std::env::args().any(|a| a == "--smoke") {
        run_smoke();
        return;
    }

    println!("========================================");
    println!("  Ochroma Walking Simulator");
    println!("  Collect all 10 glowing orbs to win!");
    println!("  Press ENTER on the main menu to start");
    println!("========================================");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = WalkingSim::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
