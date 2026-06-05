// Hide the console window on Windows (GUI application)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Ochroma Engine — Gaussian Scene Viewer (`splat_view`)
//!
//! Loads a Gaussian scene (`.ply` standard 3DGS, or Ochroma `.vxm`) and renders
//! it with an orbit camera through the engine's true anisotropic 3DGS software
//! rasteriser (16-band spectral compositing — the same
//! [`SoftwareRasteriser`] `walking_sim` uses).
//!
//! Usage:
//! ```text
//! splat_view <scene.ply|scene.vxm> [--smoke]
//! splat_view demo [--smoke]
//! ```
//!
//! - Windowed: orbit the camera with the left mouse drag or the arrow keys;
//!   zoom with the scroll wheel or `+` / `-`. The camera auto-frames the scene's
//!   bounding sphere on load.
//! - `--smoke`: headless. Renders 8 orbit frames at 45° steps, writes the LAST
//!   to `/tmp/ochroma_splat_view_smoke.ppm`, and asserts the frames are
//!   non-trivial AND change as the camera orbits. Exits non-zero on failure.
//! - `demo`: generates a colourful Gaussian scene in-process, round-trips it
//!   through the standard-3DGS PLY writer + `load_ply`, and renders THAT.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use vox_core::spectral::Illuminant;
use vox_core::types::GaussianSplat;
use vox_render::gpu::software_rasteriser::SoftwareRasteriser;
use vox_render::gpu::wgpu_backend::WgpuBackend;
use vox_render::spectral::RenderCamera;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const SMOKE_PPM: &str = "/tmp/ochroma_splat_view_smoke.ppm";

// ---------------------------------------------------------------------------
// Scene loading
// ---------------------------------------------------------------------------

/// Load a scene by path. Dispatches on extension: `.vxm` -> VxmFile (its
/// `splats` field), anything else -> standard 3DGS PLY via `load_ply`.
fn load_scene(path: &Path) -> Result<Vec<GaussianSplat>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "vxm" {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let vxm = vox_data::vxm::VxmFile::read(reader)
            .map_err(|e| format!("read vxm {}: {e:?}", path.display()))?;
        Ok(vxm.splats)
    } else {
        vox_data::ply_loader::load_ply(path)
            .map_err(|e| format!("load_ply {}: {e}", path.display()))
    }
}

/// Build a colourful demo scene: a spherical shell of Gaussians whose hue sweeps
/// with latitude/longitude, with anisotropic scales and varied rotations so the
/// true 3DGS path has something interesting (and orientation-dependent) to show.
/// Returns RGB-derived spectral splats — the same metamer path a real PLY uses.
fn build_demo_scene() -> Vec<GaussianSplat> {
    use vox_data::spectral_upsampler::SpectralUpsampler;

    let mut splats = Vec::new();
    let radius = 6.0f32;
    let rings = 14;
    let per_ring = 26;

    for i in 0..rings {
        // Latitude from pole to pole.
        let lat = (i as f32 + 0.5) / rings as f32; // (0,1)
        let theta = lat * std::f32::consts::PI;
        let y = radius * theta.cos();
        let ring_r = radius * theta.sin();

        for j in 0..per_ring {
            let lon = (j as f32 / per_ring as f32) * std::f32::consts::TAU;
            let x = ring_r * lon.cos();
            let z = ring_r * lon.sin();

            // Hue cycles around longitude; brightness rises toward the equator.
            let hue = lon / std::f32::consts::TAU;
            let bright = 0.55 + 0.45 * theta.sin();
            let (r, g, b) = hsv_to_rgb(hue, 0.85, bright);
            let spectral_f32 = SpectralUpsampler::from_rgb(r, g, b);
            let spectral: [u16; 16] =
                std::array::from_fn(|k| half::f16::from_f32(spectral_f32[k]).to_bits());

            // Anisotropic ellipsoid tangent to the shell: stretched along the
            // longitude direction, thin radially — gives orientation-dependent
            // footprints the orbit will visibly rotate.
            let tangent = Vec3::new(-lon.sin(), 0.0, lon.cos());
            let rot = Quat::from_rotation_arc(Vec3::X, tangent.normalize());
            let scale = [0.55, 0.18, 0.30];

            splats.push(GaussianSplat::volume(
                [x, y, z],
                scale,
                rot,
                235,
                spectral,
            ));
        }
    }

    // A bright core cluster so the centre is never empty.
    let core = SpectralUpsampler::from_rgb(1.0, 0.95, 0.7);
    let core_spec: [u16; 16] = std::array::from_fn(|k| half::f16::from_f32(core[k]).to_bits());
    for k in 0..40 {
        let a = k as f32 * 2.399_963; // golden-angle scatter
        let rr = 0.9 * (k as f32 / 40.0).sqrt();
        splats.push(GaussianSplat::volume(
            [rr * a.cos(), (k as f32 * 0.05) - 1.0, rr * a.sin()],
            [0.35, 0.35, 0.35],
            Quat::IDENTITY,
            255,
            core_spec,
        ));
    }

    splats
}

/// HSV -> RGB in [0,1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = (h.rem_euclid(1.0)) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------

/// Orbit camera that frames a target point at a given distance. Yaw/pitch orbit
/// around `target`; `distance` zooms.
#[derive(Clone, Copy)]
struct OrbitCamera {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitCamera {
    /// Auto-frame: centre on the scene centroid, distance set so the whole
    /// bounding sphere fits in the vertical FOV.
    fn auto_frame(splats: &[GaussianSplat]) -> Self {
        let (center, radius) = scene_bounds(splats);
        // Fit the bounding sphere into the vertical FOV (FRAC_PI_4) with margin.
        let half_fov = std::f32::consts::FRAC_PI_4 * 0.5;
        let fit = radius / half_fov.sin();
        let distance = (fit * 1.3).max(radius + 1.0).max(2.0);
        OrbitCamera {
            target: center,
            distance,
            yaw: 0.0,
            pitch: 0.35,
        }
    }

    /// Eye position from yaw/pitch/distance around the target.
    fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let dir = Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), self.yaw.cos() * cp);
        self.target + dir * self.distance
    }

    fn render_camera(&self) -> RenderCamera {
        let eye = self.eye();
        RenderCamera {
            view: Mat4::look_at_rh(eye, self.target, Vec3::Y),
            proj: Mat4::perspective_rh(
                std::f32::consts::FRAC_PI_4,
                WIDTH as f32 / HEIGHT as f32,
                0.05,
                self.distance * 10.0 + 100.0,
            ),
        }
    }
}

/// Centroid + bounding radius (sphere enclosing all splat centres) of the scene.
fn scene_bounds(splats: &[GaussianSplat]) -> (Vec3, f32) {
    if splats.is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let mut center = Vec3::ZERO;
    for s in splats {
        center += Vec3::from(s.position());
    }
    center /= splats.len() as f32;
    let mut radius = 0.0f32;
    for s in splats {
        let d = (Vec3::from(s.position()) - center).length();
        // Include the splat's own extent so big splats aren't clipped.
        let ext = d + s.scales().iter().copied().fold(0.0, f32::max);
        radius = radius.max(ext);
    }
    (center, radius.max(0.5))
}

/// Render the scene through the true-Gaussian software rasteriser and return
/// RGBA8 pixels (already 16-band spectral-composited + tonemapped by the
/// rasteriser's spectral resolve).
fn render_frame(
    rasteriser: &mut SoftwareRasteriser,
    splats: &[GaussianSplat],
    cam: &OrbitCamera,
    illuminant: &Illuminant,
) -> Vec<[u8; 4]> {
    let camera = cam.render_camera();
    let fb = rasteriser.render(splats, &camera, illuminant, None);
    fb.pixels
}

// ---------------------------------------------------------------------------
// Windowed application
// ---------------------------------------------------------------------------

struct SplatView {
    window: Option<Arc<Window>>,
    backend: Option<WgpuBackend>,
    rasteriser: SoftwareRasteriser,
    illuminant: Illuminant,
    splats: Vec<GaussianSplat>,
    cam: OrbitCamera,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
    keys_held: std::collections::HashSet<KeyCode>,
    last_frame: Instant,
    scene_label: String,
}

impl SplatView {
    fn new(splats: Vec<GaussianSplat>, scene_label: String) -> Self {
        let cam = OrbitCamera::auto_frame(&splats);
        Self {
            window: None,
            backend: None,
            rasteriser: SoftwareRasteriser::new(WIDTH, HEIGHT),
            illuminant: Illuminant::d65(),
            splats,
            cam,
            dragging: false,
            last_mouse: None,
            keys_held: std::collections::HashSet::new(),
            last_frame: Instant::now(),
            scene_label,
        }
    }

    /// Apply continuous keyboard orbit/zoom (arrow keys + `+`/`-`).
    fn apply_keys(&mut self, dt: f32) {
        let orbit = 1.5 * dt;
        if self.keys_held.contains(&KeyCode::ArrowLeft) {
            self.cam.yaw -= orbit;
        }
        if self.keys_held.contains(&KeyCode::ArrowRight) {
            self.cam.yaw += orbit;
        }
        if self.keys_held.contains(&KeyCode::ArrowUp) {
            self.cam.pitch = (self.cam.pitch + orbit).clamp(-1.5, 1.5);
        }
        if self.keys_held.contains(&KeyCode::ArrowDown) {
            self.cam.pitch = (self.cam.pitch - orbit).clamp(-1.5, 1.5);
        }
        let zoom = 1.0 + 1.5 * dt;
        if self.keys_held.contains(&KeyCode::Equal) || self.keys_held.contains(&KeyCode::NumpadAdd) {
            self.cam.distance = (self.cam.distance / zoom).max(0.5);
        }
        if self.keys_held.contains(&KeyCode::Minus)
            || self.keys_held.contains(&KeyCode::NumpadSubtract)
        {
            self.cam.distance *= zoom;
        }
    }
}

impl ApplicationHandler for SplatView {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title(format!("Ochroma splat_view -- {}", self.scene_label))
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );
        match WgpuBackend::new(Arc::clone(&window), WIDTH, HEIGHT) {
            Ok(backend) => self.backend = Some(backend),
            Err(e) => eprintln!("[splat_view] GPU backend init failed: {e}"),
        }
        self.window = Some(window);
        self.last_frame = Instant::now();
        println!("[splat_view] {} splats loaded.", self.splats.len());
        println!("[splat_view] Controls: drag / arrows orbit, scroll / +- zoom, ESC quit.");
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        if key == KeyCode::Escape {
                            event_loop.exit();
                        }
                        self.keys_held.insert(key);
                    } else {
                        self.keys_held.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                self.last_mouse = None;
            }

            WindowEvent::CursorMoved { position, .. } if self.dragging => {
                if let Some((lx, ly)) = self.last_mouse {
                    self.cam.yaw += (position.x - lx) as f32 * 0.01;
                    self.cam.pitch =
                        (self.cam.pitch - (position.y - ly) as f32 * 0.01).clamp(-1.5, 1.5);
                }
                self.last_mouse = Some((position.x, position.y));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                let factor = (1.0 - scroll * 0.1).clamp(0.5, 1.5);
                self.cam.distance = (self.cam.distance * factor).max(0.5);
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                self.apply_keys(dt);

                let pixels =
                    render_frame(&mut self.rasteriser, &self.splats, &self.cam, &self.illuminant);
                if let Some(backend) = &self.backend {
                    backend.present_framebuffer(&pixels, WIDTH, HEIGHT);
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke test (headless)
// ---------------------------------------------------------------------------

/// Fraction of non-black pixels in a frame.
fn non_black_fraction(pixels: &[[u8; 4]]) -> f32 {
    let n = pixels
        .iter()
        .filter(|p| p[0] > 4 || p[1] > 4 || p[2] > 4)
        .count();
    n as f32 / pixels.len() as f32
}

/// Number of distinct RGB colours (quantised to 5 bits/channel to ignore noise).
fn distinct_colors(pixels: &[[u8; 4]]) -> usize {
    let mut set = std::collections::HashSet::new();
    for p in pixels {
        if p[0] <= 4 && p[1] <= 4 && p[2] <= 4 {
            continue; // ignore background
        }
        let key = ((p[0] >> 3) as u32) << 10 | ((p[1] >> 3) as u32) << 5 | (p[2] >> 3) as u32;
        set.insert(key);
    }
    set.len()
}

/// Fraction of pixels that differ (any channel by >16) between two frames.
fn frame_diff_fraction(a: &[[u8; 4]], b: &[[u8; 4]]) -> f32 {
    let mut diff = 0usize;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let d = (pa[0] as i32 - pb[0] as i32).abs()
            .max((pa[1] as i32 - pb[1] as i32).abs())
            .max((pa[2] as i32 - pb[2] as i32).abs());
        if d > 16 {
            diff += 1;
        }
    }
    diff as f32 / a.len() as f32
}

/// Write RGBA8 pixels as a binary PPM (P6).
fn write_ppm(path: &str, pixels: &[[u8; 4]], width: u32, height: u32) -> std::io::Result<()> {
    let mut out = Vec::with_capacity((width * height * 3) as usize + 32);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for p in pixels {
        out.push(p[0]);
        out.push(p[1]);
        out.push(p[2]);
    }
    std::fs::write(path, out)
}

/// Headless smoke: render 8 orbit frames (45° steps) of `splats`, write the last
/// to the smoke PPM, and assert non-triviality + orbit-induced change. Returns
/// Err(msg) on any failed assertion.
fn run_smoke(splats: &[GaussianSplat], scene_label: &str) -> Result<(), String> {
    println!("[splat_view] === HEADLESS SMOKE MODE (no window/GPU) ===");
    println!("[splat_view] Scene: {scene_label} ({} splats)", splats.len());

    let mut rasteriser = SoftwareRasteriser::new(WIDTH, HEIGHT);
    let illuminant = Illuminant::d65();
    let mut cam = OrbitCamera::auto_frame(splats);

    let mut frames: Vec<Vec<[u8; 4]>> = Vec::with_capacity(8);
    for i in 0..8 {
        cam.yaw = (i as f32) * std::f32::consts::FRAC_PI_4; // 45° steps
        let pixels = render_frame(&mut rasteriser, splats, &cam, &illuminant);

        let nb = non_black_fraction(&pixels);
        let nc = distinct_colors(&pixels);
        println!(
            "[splat_view] frame {i}: yaw={:>5.1}deg  non_black={:.2}%  distinct_colors={}",
            cam.yaw.to_degrees(),
            nb * 100.0,
            nc
        );
        if nb < 0.01 {
            return Err(format!(
                "frame {i} too sparse: {:.3}% non-black (need >=1%)",
                nb * 100.0
            ));
        }
        if nc < 8 {
            return Err(format!("frame {i} only {nc} distinct colours (need >=8)"));
        }
        frames.push(pixels);
    }

    // Orbit proof: frame 0 (yaw 0deg) vs frame 4 (yaw 180deg) must differ in
    // >10% of pixels — the camera genuinely moved and the projection responded.
    let diff = frame_diff_fraction(&frames[0], &frames[4]);
    println!(
        "[splat_view] orbit check: frame0 vs frame4 differ in {:.1}% of pixels (need >10%)",
        diff * 100.0
    );
    if diff <= 0.10 {
        return Err(format!(
            "orbit produced no motion: frame0 vs frame4 differ in only {:.1}% of pixels",
            diff * 100.0
        ));
    }

    write_ppm(SMOKE_PPM, frames.last().unwrap(), WIDTH, HEIGHT)
        .map_err(|e| format!("write PPM {SMOKE_PPM}: {e}"))?;
    println!("[splat_view] Wrote last frame to {SMOKE_PPM}");
    println!("[splat_view] SMOKE PASSED");
    Ok(())
}

// ---------------------------------------------------------------------------
// Scene resolution: `demo` round-trips through the PLY writer; a path loads.
// ---------------------------------------------------------------------------

/// Resolve the scene argument into splats + a human label.
///
/// `demo`: build the in-process demo scene, write it to a temp standard-3DGS
/// PLY, then load THAT back via `load_ply` and render it — exercising the real
/// PLY write->read path end to end. Any other arg is treated as a scene path.
fn resolve_scene(arg: &str) -> Result<(Vec<GaussianSplat>, String), String> {
    if arg == "demo" {
        let demo = build_demo_scene();
        let tmp: PathBuf = std::env::temp_dir().join("ochroma_splat_view_demo.ply");
        vox_data::ply_loader::write_ply(&demo, &tmp)
            .map_err(|e| format!("write demo PLY {}: {e}", tmp.display()))?;
        println!(
            "[splat_view] Demo: {} splats -> {} (standard 3DGS PLY), reloading via load_ply",
            demo.len(),
            tmp.display()
        );
        let loaded = vox_data::ply_loader::load_ply(&tmp)
            .map_err(|e| format!("reload demo PLY {}: {e}", tmp.display()))?;
        if loaded.is_empty() {
            return Err("demo PLY reloaded to zero splats".into());
        }
        Ok((loaded, format!("demo ({})", tmp.display())))
    } else {
        let path = Path::new(arg);
        let splats = load_scene(path)?;
        if splats.is_empty() {
            return Err(format!("scene {} loaded to zero splats", path.display()));
        }
        Ok((splats, arg.to_string()))
    }
}

fn print_usage() {
    eprintln!("Usage: splat_view <scene.ply|scene.vxm|demo> [--smoke]");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args.iter().any(|a| a == "--smoke");
    let scene_arg = args.iter().skip(1).find(|a| !a.starts_with("--"));

    let Some(scene_arg) = scene_arg else {
        print_usage();
        std::process::exit(2);
    };

    let (splats, label) = match resolve_scene(scene_arg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[splat_view] ERROR: {e}");
            std::process::exit(1);
        }
    };

    if smoke {
        match run_smoke(&splats, &label) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("[splat_view] SMOKE FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    // Windowed mode.
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = SplatView::new(splats, label);
    event_loop.run_app(&mut app).expect("run app");
}
