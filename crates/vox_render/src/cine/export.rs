//! Deterministic single-frame cinematic render + PNG encode.
//!
//! [`render_cine_frame`] is the offline-export primitive: it renders **one**
//! converged frame through the Spectra path tracer with a fixed seed, a fixed
//! sample budget, and ReSTIR resampling disabled, so two calls with identical
//! inputs produce **byte-identical** beauty buffers. This is the determinism
//! contract the trailer/Steam-capsule PNG sequence depends on.
//!
//! It deliberately bypasses the realtime `near_realtime` preset's stochastic
//! reservoir resampling (`ResampleMode::TemporalSpatial`), which carries
//! frame-to-frame history and would make a still non-reproducible. Per export
//! frame we [`Renderer::reconfigure`] (which re-seeds the frame RNG from
//! `config.seed` and builds a fresh render state) and then `load_scene_state`
//! (which reallocates the per-pixel reservoirs from scratch) — the verified
//! determinism recipe from the design doc.
//!
//! The pixels always route through the Spectra backend; never the rasteriser.

use std::path::Path;

use spectra_renderer::{FrameOutput, RenderConfig, ResampleMode};
use spectra_scene_state::{CameraLayer, SceneState};

/// Default seed used by export when a caller does not override it. Any fixed
/// value works; what matters for reproducibility is that it is constant.
pub const CINE_BASE_SEED: u64 = 0xC1_4E_5E_ED;

/// Render one deterministic, converged cinematic frame.
///
/// Sets `RenderConfig.seed = seed`, `target_spp = spp`, and
/// `resample_mode = None` (ReSTIR off → no temporal history), reconfigures the
/// renderer (re-seeding its frame RNG and resetting render state), loads the
/// scene with `cam` as its camera, and renders to the configured sample budget.
///
/// The returned [`FrameOutput::beauty`] is linear-f32 RGBA; the caller tonemaps
/// + encodes (see [`write_png`]).
///
/// Determinism contract: for a fixed `(scene, cam, seed, spp)`, the returned
/// `beauty` is byte-identical across calls (verified by
/// `tests/cine_export_deterministic.rs`).
///
/// Generic over the GPU backend so it works on both the Vulkan and CUDA
/// `Renderer<G>` without a second code path.
pub fn render_cine_frame<G: spectra_gpu::GpuBackend>(
    renderer: &mut spectra_renderer::Renderer<G>,
    scene: &SceneState,
    cam: &CameraLayer,
    seed: u64,
    spp: u32,
) -> Result<FrameOutput, String> {
    // Build the deterministic config off the renderer's current one (preserving
    // resolution + look + slang kernel dir) and override exactly the three
    // determinism levers from the design doc.
    let mut config: RenderConfig = renderer
        .config()
        .cloned()
        .unwrap_or_else(|| RenderConfig::near_realtime(cam.width, cam.height));
    config.seed = seed;
    config.target_spp = spp;
    // ReSTIR carries per-pixel temporal history across frames; disabling it (and
    // the reconfigure/reset below) removes the only source of frame-to-frame
    // nondeterminism in a converged still.
    config.resample_mode = ResampleMode::None;
    config.use_restir = false;

    // Replace the config (re-seeds the frame RNG from config.seed, resets render
    // state) without recreating the GPU backend.
    renderer.reconfigure(config);

    // Inject the cinematic camera into the scene and (re)load it. Cloning keeps
    // the caller's `&SceneState` immutable per the API contract.
    let mut scene = scene.clone();
    scene.camera = cam.clone();
    renderer
        .load_scene_state(scene)
        .map_err(|e| format!("load_scene_state: {e:?}"))?;
    renderer.set_camera_view_matrix(cam.view_matrix);
    renderer.set_view_proj(cam.view_matrix);

    // NOTE: no `renderer.reset()` here — `reconfigure` already builds a fresh
    // `RenderState` (re-seeding the frame RNG from `config.seed`) and the
    // subsequent `load_scene_state` reallocates per-pixel reservoirs from
    // scratch (`scene.rs`: `reservoirs = vec![default; pixels]; prev = None`).
    // Calling `reset()` after `load_scene_state` would wipe the just-uploaded
    // scene (`*state = RenderState::new(..)`), yielding `render: NoScene`.

    renderer.render().map_err(|e| format!("render: {e:?}"))
}

/// Tonemap a linear-f32 RGBA beauty buffer to 8-bit and write a PNG.
///
/// `beauty` is `width*height*4` linear-f32 values (the renderer already applies
/// the configured tonemap into linear display values). We clamp to `[0,1]` and
/// quantize with round-to-nearest — a pure, deterministic mapping so identical
/// `beauty` always yields a byte-identical PNG.
pub fn write_png(beauty: &[f32], width: u32, height: u32, path: &Path) -> Result<(), String> {
    let rgba = beauty_to_rgba8(beauty, width, height);
    let img: image::RgbaImage = image::ImageBuffer::from_raw(width, height, rgba)
        .ok_or_else(|| "beauty buffer size mismatch".to_string())?;
    img.save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| format!("png encode {}: {e}", path.display()))
}

/// Tonemap a linear-f32 RGBA beauty buffer to 16-bit and write a PNG.
///
/// 16-bit preserves more tonal range for grading the captured frame. Same pure
/// quantization (round-to-nearest over `[0,65535]`), so it is equally
/// reproducible.
pub fn write_png16(beauty: &[f32], width: u32, height: u32, path: &Path) -> Result<(), String> {
    let n = (width as usize) * (height as usize) * 4;
    if beauty.len() < n {
        return Err(format!("beauty buffer too small: {} < {}", beauty.len(), n));
    }
    let mut rgba: Vec<u16> = Vec::with_capacity(n);
    for &v in &beauty[..n] {
        rgba.push((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
    }
    let img: image::ImageBuffer<image::Rgba<u16>, Vec<u16>> =
        image::ImageBuffer::from_raw(width, height, rgba)
            .ok_or_else(|| "beauty buffer size mismatch".to_string())?;
    img.save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| format!("png16 encode {}: {e}", path.display()))
}

/// Pure linear-f32 → 8-bit RGBA quantization (clamp + round-to-nearest).
///
/// Factored out so it is independently testable without touching the GPU. The
/// length is `width*height*4`; any shortfall in `beauty` is zero-filled so the
/// function never panics.
pub fn beauty_to_rgba8(beauty: &[f32], width: u32, height: u32) -> Vec<u8> {
    let n = (width as usize) * (height as usize) * 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = beauty.get(i).copied().unwrap_or(0.0);
        out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beauty_quantizes_round_to_nearest() {
        // 0.5 → 128 (0.5*255+0.5 = 128.0), 1.0 → 255, over-1 clamps, negatives clamp.
        let beauty = [0.5_f32, 1.0, 2.0, -1.0];
        let out = beauty_to_rgba8(&beauty, 1, 1);
        assert_eq!(out, vec![128u8, 255, 255, 0]);
    }
}
