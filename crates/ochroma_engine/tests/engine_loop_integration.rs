//! Integration test for the shared `EngineLoop` — the simulation driver that BOTH
//! binaries now depend on (the `ochroma` editor via `SystemMask::all()` and
//! `walking_sim` via `SystemMask::game_minimal()`).
//!
//! The per-sub-step unit tests in `engine_loop.rs` prove each system in isolation.
//! This test proves the loop runs a COHERENT MULTI-FRAME simulation with physics +
//! scripts + audio + GI all advancing together, asserting cross-system end state.
//! It is the headless equivalent of "the windowed binary's loop runs without
//! diverging" — the binary-loop coverage gap the audit flagged.

use half::f16;
use ochroma_engine::engine_loop::{EngineLoop, SystemMask};
use vox_core::engine_runtime::{EngineConfig, FixedStepCounter, ScriptInstances};
use vox_core::script_interface::{GameScript, ScriptContext};
use vox_core::types::GaussianSplat;

/// A stateful script: if the loop re-created it each frame (the original bug),
/// `frames` would never exceed 1.
struct FrameCounter {
    frames: u32,
}
impl GameScript for FrameCounter {
    fn on_update(&mut self, _ctx: &mut ScriptContext, _dt: f32) {
        self.frames += 1;
    }
    fn name(&self) -> &str {
        "FrameCounter"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn engine_loop_runs_coherent_multiframe_simulation() {
    let mut lp = EngineLoop::new(EngineConfig::default(), SystemMask::all());

    // --- Physics: a box dropped from y=20 must fall and SETTLE on the ground plane. ---
    let (body, _) = lp.physics.add_dynamic_box([0.0, 20.0, 0.0], [0.5, 0.5, 0.5], 1.0);
    let y_start = lp.physics.body_position(body).expect("body exists")[1];
    assert!((y_start - 20.0).abs() < 1e-3, "body should start at y=20, got {y_start}");

    // --- Scripts: a stateful counter that must advance once per fixed step. ---
    lp.runtime
        .register_script("FrameCounter", || Box::new(FrameCounter { frames: 0 }));
    let entity = lp.runtime.spawn("Ticker").with_script("FrameCounter").id();
    lp.runtime.start();

    // --- GI scene: a bright opaque emitter next to a fully-dark receiver. ---
    let emitter = GaussianSplat::volume(
        [0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2],
        glam::Quat::IDENTITY,
        255,
        [f16::from_f32(1.0).to_bits(); 16],
    );
    let receiver = GaussianSplat::volume(
        [0.5, 0.0, 0.0],
        [0.2, 0.2, 0.2],
        glam::Quat::IDENTITY,
        10,
        [f16::from_f32(0.0).to_bits(); 16],
    );

    // --- Drive 120 frames of the FULL loop, in a realistic per-frame order. ---
    let dt = 1.0 / 60.0;
    let listener_pos = glam::Vec3::new(0.0, 1.0, 0.0);
    let listener_fwd = glam::Vec3::new(0.0, 0.0, -1.0);
    let mut gi_out = vec![emitter, receiver];
    for _ in 0..120 {
        lp.step_scripts(dt);
        lp.step_physics(dt);
        lp.step_audio(dt, listener_pos, listener_fwd);
        gi_out = lp.step_gi(&[emitter, receiver], 12.0);
    }

    // ===== Assert coherent end state across ALL systems =====

    // Physics: fell a long way AND settled on the ground (didn't tunnel through, isn't NaN).
    let y_end = lp.physics.body_position(body).expect("body still exists")[1];
    assert!(
        y_end < y_start - 10.0,
        "body must fall far under gravity over 120 frames: {y_start} -> {y_end}"
    );
    assert!(
        y_end.is_finite() && y_end > -2.0,
        "body must rest on/near the ground plane, not tunnel through: y_end={y_end}"
    );

    // Scripts: the cached instance ran across MANY frames (stateful), not reset to 1 each frame.
    let instances = lp.runtime.world.resource::<ScriptInstances>();
    let cached = instances
        .scripts
        .get(&(entity, "FrameCounter".to_string()))
        .expect("cached script instance must persist across frames");
    let frames = cached
        .as_any()
        .downcast_ref::<FrameCounter>()
        .expect("instance is a FrameCounter")
        .frames;
    assert!(
        frames >= 100,
        "stateful script must accumulate across the 120-frame run (got {frames}); \
         a value of 1 would mean the loop re-creates the script every frame"
    );

    // The fixed-step accumulator advanced each frame.
    let steps = lp
        .runtime
        .world
        .resource::<FixedStepCounter>()
        .steps_this_frame;
    assert_eq!(steps, 1, "the final frame ran exactly one fixed step");

    // GI: the dark receiver was brightened by the emitter's radiance (band 8 lifted off zero).
    let recv_band = f16::from_bits(gi_out[1].spectral()[8]).to_f32();
    assert!(
        recv_band > 0.001,
        "GI must brighten the dark receiver over the run: receiver band 8 = {recv_band}"
    );
}
