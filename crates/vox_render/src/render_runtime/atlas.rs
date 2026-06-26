//! Texture / HDRI loader utilities — engine-side so games don't re-implement
//! them. Gated on `spectra-native` because they return `splat_backend::TextureImage`.

use vox_core::srgb_to_linear;

/// Load an equirectangular HDRI (Radiance `.hdr`) as interleaved LINEAR RGB f32
/// (channels=3) + dimensions, ready for `ResidentSceneRenderer::set_hdri`. HDR is
/// already linear (no sRGB decode). Returns `None` if the file is missing/bad so
/// the renderer falls back to the procedural sky.
pub fn load_hdri_rgb_f32(path: &str) -> Option<(Vec<f32>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    if w == 0 || h == 0 {
        return None;
    }
    Some((rgb.into_raw(), w, h))
}

/// Load a base-colour texture file into a LINEAR RGBA `TextureImage` for the path
/// tracer's flat atlas. Returns `None` if the file is missing/undecodable.
///
/// RGBA (not RGB) so the source ALPHA channel survives: PolyHaven foliage leaf
/// cards carry the leaf cutout in the base-color alpha. The path tracer's
/// alpha-cutout test reads this `.w` (megakernel.slang ~2345) ONLY for materials
/// whose `opacity_tex == albedo_tex` (the scatter/vegetation foliage signal);
/// opaque facades keep `opacity_tex = -1` so their (typically opaque, alpha=1)
/// channel is ignored — buildings render identically. sRGB is decoded on RGB;
/// the alpha channel is a LINEAR coverage signal and is NOT gamma-decoded.
pub fn load_linear_texture(path: &str) -> Option<crate::splat_backend::TextureImage> {
    let img = image::open(path).ok()?.to_rgba8();
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return None;
    }
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for px in img.pixels() {
        data.push(srgb_to_linear(px[0] as f32 / 255.0));
        data.push(srgb_to_linear(px[1] as f32 / 255.0));
        data.push(srgb_to_linear(px[2] as f32 / 255.0));
        data.push(px[3] as f32 / 255.0); // alpha = linear coverage (leaf cutout)
    }
    Some(crate::splat_backend::TextureImage {
        width: w,
        height: h,
        channels: 4,
        data,
    })
}

/// Load a DATA texture (normal / roughness / displacement) into a LINEAR atlas
/// image WITHOUT sRGB decoding — these maps are linear data, not colour, and
/// gamma-decoding them corrupts the surface (a normal map decoded as sRGB tilts
/// every normal). `channels` selects the layout: 3 = tangent-space normal (RGB),
/// 1 = single-channel roughness/height (luma). Returns `None` on a bad file so a
/// missing map falls back to flat (slot -1) and never corrupts sampling.
pub fn load_data_texture(
    path: &str,
    channels: u32,
) -> Option<crate::splat_backend::TextureImage> {
    let img = image::open(path).ok()?;
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return None;
    }
    let data: Vec<f32> = if channels == 1 {
        img.to_luma8()
            .pixels()
            .map(|p| p[0] as f32 / 255.0)
            .collect()
    } else {
        img.to_rgb8()
            .pixels()
            .flat_map(|p| {
                [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect()
    };
    Some(crate::splat_backend::TextureImage {
        width: w,
        height: h,
        channels,
        data,
    })
}
