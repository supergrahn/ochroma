//! Spectra-native frame conversion helpers — engine-side so games don't
//! re-implement them. Gated on `spectra-native` because they touch `FrameOutput`.

use crate::resident_renderer::FrameOutput;

/// Convert a `FrameOutput` to RGBA8 bytes.
pub fn beauty_to_rgba8(
    frame: &FrameOutput,
) -> (Vec<u8>, u32, u32) {
    let px = (frame.width * frame.height) as usize;
    let mut out: Vec<u8> = Vec::with_capacity(px * 4);
    for i in 0..px {
        out.push((frame.beauty[i * 4].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((frame.beauty[i * 4 + 1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((frame.beauty[i * 4 + 2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        out.push((frame.beauty[i * 4 + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    }
    (out, frame.width, frame.height)
}

/// Output dimensions from an interop frame (falls back to internal res).
pub fn interop_dims(
    frame: &FrameOutput,
    iw: u32,
    ih: u32,
) -> (u32, u32) {
    if frame.render_width > 0 { (frame.render_width, frame.render_height) } else { (iw, ih) }
}
