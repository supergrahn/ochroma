//! `city_ceiling` — an HONEST scale-ceiling benchmark for Ochroma's city-builder
//! path. No hype: it measures where each stage of a Cities-Skylines-/Manor-Lords-
//! like loop blows the 60 fps frame budget (16.67 ms) on THIS box, today.
//!
//! Five axes:
//!   1. Agent stepping (`vox_sim::AgentManager`) — destination-seeking citizens.
//!   2. Crowd sim (`vox_sim::CrowdSimulation`) — with collision avoidance.
//!   3. GPU render (`TiledSplatRenderer`) — a city's worth of splats on-device.
//!   4. Spectral relight (`relight_scene`) — re-lighting the whole city.
//!   5. The REAL `CitySim` causal loop (zoning→jobs→economy→migration→agents),
//!      proving the integrated game logic actually runs (today: small-scale).
//!
//! Run: `cargo run --release -p vox_app --bin city_ceiling`

use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use vox_core::lwc::WorldCoord;
use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::tiled_splat_renderer::TiledSplatRenderer;
use vox_render::gpu::GpuContext;
use vox_render::relight::{relight_scene, IlluminantSpec, RelightSettings};
use vox_render::spectral::RenderCamera;
use vox_sim::agent::AgentManager;
use vox_sim::city_sim::CitySim;
use vox_sim::crowd::CrowdSimulation;

/// 60 fps frame budget. A stage that exceeds this can't run every frame at 60 fps.
const FRAME_BUDGET_MS: f64 = 16.67;
const RENDER_W: u32 = 640;
const RENDER_H: u32 = 360;

fn verdict(ms: f64) -> &'static str {
    if ms <= FRAME_BUDGET_MS {
        "OK  @60fps"
    } else if ms <= 33.3 {
        "tight @30fps"
    } else {
        "OVER budget"
    }
}

/// Axis 1 — citizen agents seeking a destination, stepped once per frame.
fn bench_agents(counts: &[usize]) {
    println!("\n── 1. Agent stepping  (AgentManager, destination-seeking) ─────────────");
    println!("   {:>9}  {:>11}  {:>10}", "agents", "ms / tick", "verdict");
    for &n in counts {
        let mut am = AgentManager::new();
        let ids: Vec<_> = (0..n)
            .map(|i| {
                let p = WorldCoord::from_absolute((i % 1000) as f64, 0.0, (i / 1000) as f64);
                am.spawn(p, 1.4)
            })
            .collect();
        // A far destination so every agent keeps moving every tick (sustained cost).
        for id in &ids {
            if let Some(a) = am.get_mut(*id) {
                a.destination = Some(WorldCoord::from_absolute(50_000.0, 0.0, 50_000.0));
            }
        }
        let k = 30u32;
        let t = Instant::now();
        for _ in 0..k {
            am.tick(1.0 / 60.0);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / k as f64;
        println!("   {n:>9}  {ms:>9.3}  {:>10}", verdict(ms));
    }
}

/// Axis 2 — crowd agents with collision avoidance (the heavier per-agent model).
fn bench_crowd(counts: &[usize]) {
    println!("\n── 2. Crowd sim  (CrowdSimulation, with collision avoidance) ──────────");
    println!("   {:>9}  {:>11}  {:>10}", "agents", "ms / tick", "verdict");
    for &n in counts {
        let mut crowd = CrowdSimulation::new();
        for i in 0..n {
            let p = Vec3::new((i % 200) as f32, 0.0, (i / 200) as f32);
            crowd.add_agent(p, Vec3::new(500.0, 0.0, 500.0), 1.4);
        }
        let k = 10u32;
        let t = Instant::now();
        for _ in 0..k {
            crowd.tick(1.0 / 60.0);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / k as f64;
        println!("   {n:>9}  {ms:>9.3}  {:>10}", verdict(ms));
    }
}

/// Build a city-like grid of opaque volume splats (a stand-in for buildings/props).
fn city_splats(n: usize) -> Vec<GaussianSplat> {
    let side = (n as f64).sqrt().ceil() as usize;
    let spd_green: [u16; 16] = std::array::from_fn(|i| {
        let v = if (5..=9).contains(&i) { 0.85f32 } else { 0.2 };
        half::f16::from_f32(v).to_bits()
    });
    let spd_amber: [u16; 16] = std::array::from_fn(|i| {
        let v = if (9..=14).contains(&i) { 0.9f32 } else { 0.25 };
        half::f16::from_f32(v).to_bits()
    });
    (0..n)
        .map(|i| {
            let gx = (i % side) as f32 - side as f32 * 0.5;
            let gz = (i / side) as f32 - side as f32 * 0.5;
            let spd = if i % 3 == 0 { spd_amber } else { spd_green };
            GaussianSplat::volume(
                [gx * 0.6, 0.0, gz * 0.6 - 6.0],
                [0.25, 0.25, 0.25],
                Quat::IDENTITY,
                255,
                spd,
            )
        })
        .collect()
}

fn city_camera() -> RenderCamera {
    RenderCamera {
        view: Mat4::look_at_rh(Vec3::new(0.0, 40.0, 60.0), Vec3::new(0.0, 0.0, -6.0), Vec3::Y),
        proj: Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_3,
            RENDER_W as f32 / RENDER_H as f32,
            0.1,
            500.0,
        ),
    }
}

/// Axis 3 — render N city splats on-device through the tiled rasterizer.
fn bench_render(ctx: &GpuContext, counts: &[usize]) {
    println!("\n── 3. GPU render  (TiledSplatRenderer, {RENDER_W}x{RENDER_H}, on the local GPU) ──");
    println!("   {:>9}  {:>11}  {:>9}  {:>10}", "splats", "render ms", "non-black", "verdict");
    let cam = city_camera();
    for &n in counts {
        let splats = city_splats(n);
        let mut renderer = match TiledSplatRenderer::new(ctx.clone(), &splats, RENDER_W, RENDER_H) {
            Ok(r) => r,
            Err(e) => {
                println!("   {n:>9}  {:>11}  (ExceedsDeviceLimits / {e}) — RENDER CEILING", "—");
                break;
            }
        };
        let frame = match renderer.render(&cam) {
            Ok(f) => f,
            Err(e) => {
                println!("   {n:>9}  render error: {e}");
                continue;
            }
        };
        let (_px, non_black) = frame.resolve_to_srgb(ctx, &Illuminant::d65());
        let ms = frame.raster_gpu_ms.map(|m| m as f64).unwrap_or(frame.wall_ms as f64);
        let pct = non_black as f64 / (RENDER_W * RENDER_H) as f64 * 100.0;
        println!("   {n:>9}  {ms:>9.3}*  {pct:>7.1}%  {:>10}", verdict(ms));
    }
    println!("   (* wall-clock of the whole chain incl. one tile_count readback — see Spec 08/11)");
}

/// Axis 4 — spectral relight of N city splats (the wedge applied city-wide).
fn bench_relight(counts: &[usize]) {
    println!("\n── 4. Spectral relight  (relight_scene, whole-city re-illumination, CPU) ──");
    println!("   {:>9}  {:>11}  {:>10}", "splats", "relight ms", "verdict");
    let settings = RelightSettings::new(
        IlluminantSpec::parse("daylight").unwrap(),
        IlluminantSpec::parse("tungsten").unwrap(),
    )
    .with_sky_ambient(false)
    .with_shadows(false);
    for &n in counts {
        let splats = city_splats(n);
        let t = Instant::now();
        let (_relit, _r) = relight_scene(&splats, &settings);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("   {n:>9}  {ms:>9.3}  {:>10}", verdict(ms));
    }
}

/// Axis 5 — the REAL integrated CitySim causal loop. Proves the game logic runs
/// (zoning → jobs → economy → migration → agents); reports its scale + tick cost.
fn bench_city_sim() {
    println!("\n── 5. Integrated CitySim causal loop  (the real game logic) ───────────");
    let mut sim = CitySim::new_small();
    let s0 = sim.stats();
    let t = Instant::now();
    let ticks = 200u32;
    let s = sim.tick(ticks);
    let ms_total = t.elapsed().as_secs_f64() * 1000.0;
    let ms_per_tick = ms_total / ticks as f64;
    println!(
        "   start: pop={}  → after {ticks} ticks: pop={} employed={} commuting={} budget-driven economy evolved",
        s0.population, s.population, s.employed, s.agents_commuting
    );
    println!(
        "   integrated tick cost: {ms_per_tick:.4} ms/tick  [{}]",
        verdict(ms_per_tick)
    );
    println!(
        "   NOTE: new_small() is a FIXED small city (8 buildings, ~58 housing cap). The 100k\n   \
         milestone is a design goal — reaching it needs a scale constructor (zone/develop\n   \
         thousands of plots). That constructor is the real next gap for CS2/Manor-Lords scale."
    );
}

/// Headless hardware GPU context (skips on a no-GPU / software lane).
fn headless_context() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let info = adapter.get_info();
    if vox_render::gpu::adapter::ensure_hardware(&info).is_err() {
        return None;
    }
    let ts = if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("city_ceiling"),
            required_features: ts,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some(GpuContext::from_parts(&device, &queue, &info))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Ochroma city-builder SCALE CEILING — honest 60fps (16.67ms) budget    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let agent_counts = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000];
    let crowd_counts = [500, 2_000, 10_000, 25_000, 50_000];
    let render_counts = [10_000, 100_000, 500_000, 1_000_000, 2_000_000];

    bench_agents(&agent_counts);
    bench_crowd(&crowd_counts);
    match headless_context() {
        Some(ctx) => {
            println!("\n[gpu] {}", ctx.adapter_name());
            bench_render(&ctx, &render_counts);
        }
        None => println!("\n── 3. GPU render — SKIPPED (no hardware GPU adapter) ──"),
    }
    bench_relight(&render_counts);
    bench_city_sim();

    println!("\n── Verdict ────────────────────────────────────────────────────────────");
    println!("   The numbers above are the ceiling TODAY on this box. Read them as:");
    println!("   how many citizens / splats each stage sustains inside one 60fps frame.");
}
