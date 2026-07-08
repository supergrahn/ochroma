//! Determinism gate for the cinematic offline export.
//!
//! `render_cine_frame` must produce a byte-identical beauty buffer for two
//! renders of the same `(scene, cam, seed, spp)` — the contract trailer / Steam
//! capsule capture depends on. We compare SHA-256 of the linear-f32 beauty
//! bytes from two back-to-back renders on the same backend.
//!
//! Requires the Spectra Vulkan backend; if no GPU/driver is present the test
//! skips (prints SKIPPED) rather than failing, matching the other GPU tests.

#![cfg(feature = "spectra-native")]

use sha2::{Digest, Sha256};
use spectra_gpu::VulkanSlangBackend;
use spectra_renderer::{RenderConfig, Renderer};
use spectra_scene_state::{CameraLayer, LightLayer, SceneState};
use vox_core::types::GaussianSplat;
use vox_render::splat_convert::{camera_layer, splats_to_lit_scene};

const W: u32 = 96;
const H: u32 = 64;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A few opaque, brightly-spectral surface splats spread in depth so a non-zero
/// `lens_radius` actually blurs the out-of-focus ones (exercises the DoF path).
fn test_scene_with_bright_point(eye: [f32; 3]) -> SceneState {
    let spectral = [40000u16; 16]; // bright, ~flat (white-ish) reflectance
    let mk = |pos: [f32; 3]| {
        GaussianSplat::surface(
            pos,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.6,
            0.6,
            255,
            spectral,
        )
    };
    let splats = vec![
        mk([0.0, 0.0, 0.0]),    // near (in focus)
        mk([1.5, 0.5, -4.0]),   // mid
        mk([-1.0, -0.4, -9.0]), // far (out of focus -> blurred by DoF)
    ];
    let mut scene = splats_to_lit_scene(&splats, W, H, eye);
    scene.lights = LightLayer {
        // A directional light toward the splats (packing handled by the convert
        // module's own light path; here a simple downward sun is enough).
        light_data: vec![
            0.3, -0.8, -0.3, 0.0, // direction
            1.0, 0.97, 0.92, 5.0, // color + intensity
        ],
        light_count: 1,
    };
    scene
}

fn test_cam_layer(eye: [f32; 3], lens_radius: f32, focus: f32, bokeh: i32) -> CameraLayer {
    let view = glam::Mat4::look_at_rh(glam::Vec3::from(eye), glam::Vec3::ZERO, glam::Vec3::Y)
        .to_cols_array();
    let mut cam = camera_layer(view, std::f32::consts::FRAC_PI_4, W, H, lens_radius, focus);
    cam.bokeh_blades = bokeh;
    cam.shutter_open = 0.0;
    cam.shutter_close = 0.0;
    cam
}

fn test_backend() -> Option<Renderer<VulkanSlangBackend>> {
    let gpu = match VulkanSlangBackend::new(0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIPPED cine_export_deterministic: no Vulkan backend: {e:?}");
            return None;
        }
    };
    let mut config = RenderConfig::near_realtime(W, H);
    // Point the renderer at the in-repo Slang kernels.
    if let Ok(d) = std::env::var("SPECTRA_SLANG_DIR") {
        config.slang_kernel_dir = Some(std::path::PathBuf::from(d));
    } else {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spectra/slang");
        if p.is_dir() {
            config.slang_kernel_dir = Some(p);
        }
    }
    Some(Renderer::new(gpu, config))
}

#[test]
fn cine_export_deterministic() {
    let Some(mut backend) = test_backend() else {
        return;
    };
    let eye = [3.0, 2.0, 6.0];
    let scene = test_scene_with_bright_point(eye);
    let cam = test_cam_layer(
        eye, /*lens_radius*/ 5.0, /*focus*/ 8.0, /*bokeh*/ 6,
    );

    let a = vox_render::render_cine_frame(
        &mut backend,
        &scene,
        &cam,
        /*seed*/ 1234,
        /*spp*/ 64,
    )
    .expect("render a");
    let b = vox_render::render_cine_frame(
        &mut backend,
        &scene,
        &cam,
        /*seed*/ 1234,
        /*spp*/ 64,
    )
    .expect("render b");

    assert_eq!(a.beauty.len(), (W * H * 4) as usize);
    let ha = sha256_hex(bytemuck::cast_slice(&a.beauty));
    let hb = sha256_hex(bytemuck::cast_slice(&b.beauty));
    println!("sha256(frame_a) = {ha}");
    println!("sha256(frame_b) = {hb}");
    assert_eq!(ha, hb, "non-deterministic: {ha} vs {hb}");

    // Sanity: the frame is not entirely black (the splats actually rendered).
    let lit = a.beauty.iter().any(|&v| v > 1e-4);
    assert!(lit, "frame is entirely black — scene did not render");
}
