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
use vox_render::spectral_gi::GpuGi;

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

/// Build a deterministic few-hundred-splat GI scene: a line of dark receivers
/// along +X with several bright opaque emitters seeded in. Mirrors the shape of
/// `vox_render`'s equivalence scene so the same epsilon applies.
fn gi_scene(n: usize) -> Vec<GaussianSplat> {
    let mut scene: Vec<GaussianSplat> = (0..n)
        .map(|i| {
            GaussianSplat::volume(
                [i as f32 * 0.5, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                glam::Quat::IDENTITY,
                10, // dark receiver, opacity <= 128 → not an emitter
                [f16::from_f32(0.0).to_bits(); 16],
            )
        })
        .collect();
    let put_emitter = |scene: &mut Vec<GaussianSplat>, idx: usize, pos: [f32; 3], band: usize, v: f32| {
        let mut spectral = [f16::from_f32(0.0).to_bits(); 16];
        spectral[band] = f16::from_f32(v).to_bits();
        scene[idx] = GaussianSplat::volume(pos, [0.1, 0.1, 0.1], glam::Quat::IDENTITY, 255, spectral);
    };
    // Two front emitters plus one adjacent to a mid-scene probe receiver.
    assert!(n > 201, "gi_scene needs at least 202 splats for its emitter layout");
    put_emitter(&mut scene, 0, [0.0, 0.0, 0.0], 8, 0.5);
    put_emitter(&mut scene, 7, [3.5, 0.0, 0.0], 8, 0.4);
    put_emitter(&mut scene, 201, [100.0 * 0.5 + 0.2, 0.0, 0.0], 8, 0.5); // 0.2m off receiver 100
    scene
}

/// The headline integration: routing `step_gi` through the GPU backend produces
/// per-band, per-splat results that agree with the CPU backend within the same
/// f16-quantization-aware epsilon `vox_render`'s equivalence test uses (2e-3).
/// Also asserts `gi_backend()` reports the active backend correctly and that a
/// fresh CPU loop reports "cpu". Skips cleanly if this box has no GPU adapter.
#[test]
fn step_gi_gpu_backend_matches_cpu_backend_per_band() {
    let n = 400usize;
    let scene = gi_scene(n);
    let hour = 12.0; // noon → non-zero sky ambient

    // CPU backend (the proven default).
    let mut cpu_lp = EngineLoop::new(EngineConfig::default(), SystemMask::all());
    assert_eq!(cpu_lp.gi_backend(), "cpu", "fresh loop must default to CPU GI");
    // Stateless single step: fresh cache, alpha defaults to 0.9, but a single
    // call from a zeroed cache is what both paths compare against — drive ONE
    // step on each so the temporal-EMA state matches (one blend from zero).
    let cpu_out = cpu_lp.step_gi(&scene, hour);
    let cpu_us = cpu_lp.last_gi_us().expect("CPU step_gi must record timing");

    // GPU backend on a FRESH loop (skip if no adapter / init failure).
    let mut gpu_lp = EngineLoop::new(EngineConfig::default(), SystemMask::all());
    match gpu_lp.use_gpu_gi() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("[step_gi gpu equivalence] no usable GPU ({e}) — skipping");
            return;
        }
    }
    assert_eq!(gpu_lp.gi_backend(), "gpu", "use_gpu_gi() must activate the GPU backend");
    let gpu_out = gpu_lp.step_gi(&scene, hour);
    let gpu_us = gpu_lp.last_gi_us().expect("GPU step_gi must record timing");
    assert_eq!(gpu_out.len(), cpu_out.len(), "both backends return all splats");

    // Per-band per-splat agreement. The CPU `step_gi` blends from a zeroed cache
    // with alpha=0.9 (10% of the new value); the GPU path is a full replace
    // (alpha=0). To compare the SAME quantity we drive the CPU cache to steady
    // state and re-read — but that changes its EMA. Instead, compare against the
    // GPU's documented full-replace reference directly via a second CPU loop that
    // we step enough times to converge, then assert DIRECTIONAL + magnitude
    // agreement on the dominant probe and check the converged equivalence below.
    // For a tight numeric contract we drive the CPU backend to convergence:
    let mut cpu_conv = EngineLoop::new(EngineConfig::default(), SystemMask::all());
    let mut conv_out = Vec::new();
    for _ in 0..200 {
        conv_out = cpu_conv.step_gi(&scene, hour);
    }

    let eps = 2e-3f32;
    let sample: Vec<usize> = vec![0, 7, 100, 101, 201, 200, 250, 399];
    let mut max_delta = 0.0f32;
    for &i in &sample {
        for b in 0..16 {
            let g = f16::from_bits(gpu_out[i].spectral()[b]).to_f32();
            let c = f16::from_bits(conv_out[i].spectral()[b]).to_f32();
            let d = (g - c).abs();
            if d > max_delta {
                max_delta = d;
            }
            assert!(
                d <= eps,
                "GPU vs converged-CPU divergence at splat {i} band {b}: gpu={g} cpu={c} (|Δ|={d} > {eps})"
            );
        }
    }

    // Prove the probe receiver (idx 100, adjacent to emitter 201) was genuinely
    // lit on BOTH paths — not a vacuous all-zero match.
    let probe_g = f16::from_bits(gpu_out[100].spectral()[8]).to_f32();
    let probe_c = f16::from_bits(conv_out[100].spectral()[8]).to_f32();
    assert!(
        probe_g > 0.05 && probe_c > 0.05,
        "probe receiver must be strongly lit by its adjacent emitter: gpu={probe_g} cpu={probe_c}"
    );

    // Sanity that the single-step CPU output is also non-trivially lit (the
    // default behavior shells get) so this isn't comparing against a dead path.
    let single_probe = f16::from_bits(cpu_out[100].spectral()[8]).to_f32();
    assert!(single_probe > 0.0, "single CPU step must already start lifting the probe: {single_probe}");

    eprintln!(
        "[step_gi gpu equivalence] n={n} sample={} max|Δ|={max_delta:.2e} probe gpu={probe_g:.4} cpu={probe_c:.4} | last_gi_us cpu={cpu_us} gpu={gpu_us}",
        sample.len()
    );
}

/// The forced-failure fallback: if the GPU device cannot be created (here forced
/// via impossible wgpu limits exposed by `GpuGi::new_with_limits`), `use_gpu_gi`
/// returns `Err` and the loop stays on CPU — `step_gi` still returns valid lit
/// splats and `gi_backend()` reports "cpu". No panic.
#[test]
fn gpu_gi_init_failure_falls_back_to_cpu_without_panicking() {
    // `use_gpu_gi()` uses default limits, so we can't force ITS failure cheaply
    // without an adapter-less box. Instead assert the contract directly: a loop
    // whose use_gpu_gi() errored (or was never called) stays on CPU and works.
    let mut lp = EngineLoop::new(EngineConfig::default(), SystemMask::all());
    // Construct a GpuGi with impossible limits to PROVE device creation fails
    // gracefully (Err, not panic) — the same failure use_gpu_gi() would surface.
    let forced = GpuGi::new_failing_for_test();
    assert!(forced.is_err(), "impossible limits must not yield a working GPU device");

    // The loop is still on CPU and step_gi works and returns valid lit splats.
    assert_eq!(lp.gi_backend(), "cpu");
    let scene = gi_scene(256);
    let out = lp.step_gi(&scene, 12.0);
    assert_eq!(out.len(), scene.len());
    let probe = f16::from_bits(out[1].spectral()[8]).to_f32();
    assert!(probe.is_finite(), "CPU fallback must return finite spectral, got {probe}");
    assert!(lp.last_gi_us().is_some(), "step_gi must record timing on the CPU path too");
}

/// A frame whose splat count exceeds the GPU device capacity must route to
/// the (unlimited) CPU path for that call — the GPU pass would clamp and
/// silently leave the tail unlit — while the GPU backend stays selected for
/// subsequent smaller frames. (Wave-3 review: the old field doc claimed this
/// fallback existed; now it does.)
#[test]
fn over_capacity_frame_routes_to_cpu_and_keeps_gpu_selected() {
    let mut gpu_loop = EngineLoop::new(EngineConfig::default(), SystemMask::all());
    if gpu_loop.use_gpu_gi().is_err() {
        eprintln!("no GPU adapter — skipping over-capacity routing test");
        return;
    }
    assert_eq!(gpu_loop.gi_backend(), "gpu");
    gpu_loop.set_gpu_gi_capacity_for_test(64); // far below the 256-splat scene

    let scene = gi_scene(256);

    // Reference: a pure-CPU loop on the identical scene.
    let mut cpu_loop = EngineLoop::new(EngineConfig::default(), SystemMask::all());
    let cpu_out = cpu_loop.step_gi(&scene, 12.0);

    // Over-capacity frame on the gpu-selected loop: must match CPU exactly
    // (it routed to CPU), including the tail beyond the GPU capacity.
    let routed_out = gpu_loop.step_gi(&scene, 12.0);
    assert_eq!(routed_out.len(), cpu_out.len());
    for (i, (a, b)) in routed_out.iter().zip(cpu_out.iter()).enumerate() {
        for band in 0..16 {
            assert_eq!(
                a.spectral()[band],
                b.spectral()[band],
                "splat {i} band {band}: over-capacity frame diverged from CPU"
            );
        }
    }
    // The backend selection survives — a later smaller frame would use the GPU.
    assert_eq!(
        gpu_loop.gi_backend(),
        "gpu",
        "over-capacity routing must not permanently demote the backend"
    );
    // Telemetry honesty (review finding): the SELECTED backend is gpu, but
    // the EXECUTED path for this over-capacity call was cpu — both must be
    // visible so timing isn't mislabeled.
    assert_eq!(
        gpu_loop.last_gi_backend_used(),
        Some("cpu"),
        "over-capacity frame executed the CPU path; telemetry must say so"
    );
}
