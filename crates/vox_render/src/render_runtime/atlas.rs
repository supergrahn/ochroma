//! Texture / HDRI loader utilities — engine-side so games don't re-implement
//! them. Gated on `spectra-native` because they return `splat_backend::TextureImage`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use vox_core::srgb_to_linear;

const TEX_CACHE_MAGIC: &[u8; 8] = b"UHTEX01\0";

type TextureCache = HashMap<String, Arc<crate::splat_backend::TextureImage>>;

#[derive(Clone, Copy)]
enum DdsMirrorMode {
    SrgbColor,
    LinearData,
}

fn texture_memory_cache() -> &'static Mutex<TextureCache> {
    static CACHE: OnceLock<Mutex<TextureCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn texture_cache_enabled() -> bool {
    std::env::var("OCHROMA_TEXTURE_CACHE")
        .map(|v| v.trim() != "0")
        .unwrap_or(true)
}

fn texture_disk_cache_enabled() -> bool {
    std::env::var("OCHROMA_TEXTURE_DISK_CACHE")
        .map(|v| !matches!(v.trim(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}

fn texture_cache_dir() -> PathBuf {
    std::env::var_os("OCHROMA_TEXTURE_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if let Ok(cwd) = std::env::current_dir() {
                let dir = cwd
                    .join("game_data")
                    .join("cache")
                    .join("texture_decode_v1");
                if dir.parent().is_some_and(|p| p.exists()) {
                    return dir;
                }
            }
            if let Ok(exe) = std::env::current_exe()
                && let Some(exe_dir) = exe.parent()
            {
                let dir = exe_dir
                    .join("game_data")
                    .join("cache")
                    .join("texture_decode_v1");
                if dir.parent().is_some_and(|p| p.exists()) {
                    return dir;
                }
            }
            std::env::temp_dir().join("ochroma_texture_cache_v1")
        })
}

fn file_stamp(path: &Path) -> Option<(u64, u64, u32)> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let dur = modified
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::ZERO);
    Some((meta.len(), dur.as_secs(), dur.subsec_nanos()))
}

fn texture_cache_key(path: &Path, mode: &str) -> Option<String> {
    let (len, secs, nanos) = file_stamp(path)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut h);
    mode.hash(&mut h);
    len.hash(&mut h);
    secs.hash(&mut h);
    nanos.hash(&mut h);
    Some(format!("{:016x}", h.finish()))
}

fn is_dds_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("dds"))
}

fn dds_format_to_gpu(
    format: image_dds::ImageFormat,
) -> Option<(spectra_gpu::GpuTextureFormat, u32, u32, usize)> {
    use image_dds::ImageFormat as D;
    use spectra_gpu::GpuTextureFormat as G;
    match format {
        D::Rgba8Unorm => Some((G::Rgba8Unorm, 4, 1, 4)),
        D::Rgba8UnormSrgb => Some((G::Rgba8UnormSrgb, 4, 1, 4)),
        D::BC1RgbaUnorm => Some((G::Bc1Unorm, 4, 4, 8)),
        D::BC1RgbaUnormSrgb => Some((G::Bc1UnormSrgb, 4, 4, 8)),
        D::BC3RgbaUnorm => Some((G::Bc3Unorm, 4, 4, 16)),
        D::BC3RgbaUnormSrgb => Some((G::Bc3UnormSrgb, 4, 4, 16)),
        D::BC4RUnorm => Some((G::Bc4Unorm, 1, 4, 8)),
        D::BC5RgUnorm => Some((G::Bc5Unorm, 2, 4, 16)),
        D::BC7RgbaUnorm => Some((G::Bc7Unorm, 4, 4, 16)),
        D::BC7RgbaUnormSrgb => Some((G::Bc7UnormSrgb, 4, 4, 16)),
        _ => None,
    }
}

fn dds_format_is_srgb(format: image_dds::ImageFormat) -> bool {
    matches!(
        format,
        image_dds::ImageFormat::Rgba8UnormSrgb
            | image_dds::ImageFormat::BC1RgbaUnormSrgb
            | image_dds::ImageFormat::BC3RgbaUnormSrgb
            | image_dds::ImageFormat::BC7RgbaUnormSrgb
    )
}

fn dds_format_is_bc5(format: image_dds::ImageFormat) -> bool {
    matches!(format, image_dds::ImageFormat::BC5RgUnorm)
}

fn dds_native_mips(
    surface: &image_dds::Surface<&[u8]>,
    block_extent: u32,
    block_bytes: usize,
) -> Option<Vec<crate::RendererTextureMip>> {
    let mut mips = Vec::with_capacity(surface.mipmaps as usize);
    for level in 0..surface.mipmaps {
        let width = image_dds::mip_dimension(surface.width, level);
        let height = image_dds::mip_dimension(surface.height, level);
        let blocks_w = width.div_ceil(block_extent).max(1);
        let blocks_h = height.div_ceil(block_extent).max(1);
        let row_pitch_bytes = blocks_w as usize * block_bytes;
        let bytes = surface.get(0, 0, level)?.to_vec();
        if bytes.len() != row_pitch_bytes.checked_mul(blocks_h as usize)? {
            return None;
        }
        mips.push(crate::RendererTextureMip {
            width,
            height,
            row_pitch_bytes,
            data: bytes,
        });
    }
    Some(mips)
}

fn load_dds_texture(
    path: &Path,
    requested_channels: u32,
    mirror_mode: DdsMirrorMode,
) -> Option<crate::splat_backend::TextureImage> {
    if !(1..=4).contains(&requested_channels) {
        return None;
    }
    let mut f = std::fs::File::open(path).ok()?;
    let dds = image_dds::ddsfile::Dds::read(&mut f).ok()?;
    let surface = image_dds::Surface::from_dds(&dds).ok()?;
    if surface.depth != 1 || surface.layers != 1 || surface.width == 0 || surface.height == 0 {
        return None;
    }
    let (gpu_format, logical_channels, block_extent, block_bytes) =
        dds_format_to_gpu(surface.image_format)?;
    if requested_channels > logical_channels {
        let bc5_normal_as_rgb = requested_channels == 3
            && matches!(mirror_mode, DdsMirrorMode::LinearData)
            && dds_format_is_bc5(surface.image_format);
        if !bc5_normal_as_rgb {
            return None;
        }
    }
    if block_extent > 1 && (surface.width % block_extent != 0 || surface.height % block_extent != 0)
    {
        eprintln!(
            "[texture-dds] rejecting non-block-aligned BCn texture {}: {}x{}",
            path.display(),
            surface.width,
            surface.height
        );
        return None;
    }

    let native = crate::RendererTexture2D::new(
        gpu_format,
        surface.width,
        surface.height,
        requested_channels,
        dds_native_mips(&surface, block_extent, block_bytes)?,
    );
    let decoded = surface.decode_layers_mipmaps_rgbaf32(0..1, 0..1).ok()?;
    let mut data = Vec::with_capacity(
        surface.width as usize * surface.height as usize * requested_channels as usize,
    );
    let apply_srgb =
        matches!(mirror_mode, DdsMirrorMode::SrgbColor) && dds_format_is_srgb(surface.image_format);
    let reconstruct_bc5_normal = requested_channels == 3
        && matches!(mirror_mode, DdsMirrorMode::LinearData)
        && dds_format_is_bc5(surface.image_format);
    for rgba in decoded.data.chunks_exact(4) {
        let mut px = [rgba[0], rgba[1], rgba[2], rgba[3]];
        if apply_srgb {
            px[0] = srgb_to_linear(px[0]);
            px[1] = srgb_to_linear(px[1]);
            px[2] = srgb_to_linear(px[2]);
        }
        if reconstruct_bc5_normal {
            let x = px[0] * 2.0 - 1.0;
            let y = px[1] * 2.0 - 1.0;
            let z = (1.0 - x * x - y * y).max(0.0).sqrt();
            data.extend_from_slice(&[px[0], px[1], z * 0.5 + 0.5]);
        } else {
            data.extend_from_slice(&px[..requested_channels as usize]);
        }
    }

    eprintln!(
        "[texture-dds] loaded native {} format={:?} mips={} mirror_ch={}",
        path.display(),
        gpu_format,
        native.mips.len(),
        requested_channels
    );
    Some(crate::splat_backend::TextureImage {
        width: surface.width,
        height: surface.height,
        channels: requested_channels,
        data,
        native: Some(native),
    })
}

fn load_dds_texture_arc(
    path: &Path,
    mode: &str,
    requested_channels: u32,
    mirror_mode: DdsMirrorMode,
) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let key = texture_cache_key(path, mode)?;
    let key = format!("dds:{key}");
    if texture_cache_enabled()
        && let Ok(cache) = texture_memory_cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return Some(hit.clone());
    }
    let tex = Arc::new(load_dds_texture(path, requested_channels, mirror_mode)?);
    if texture_cache_enabled()
        && let Ok(mut cache) = texture_memory_cache().lock()
    {
        cache.insert(key, tex.clone());
    }
    Some(tex)
}

fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn read_texture_cache_file(path: &Path) -> Option<crate::splat_backend::TextureImage> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != TEX_CACHE_MAGIC {
        return None;
    }
    let width = read_u32(&mut f)?;
    let height = read_u32(&mut f)?;
    let channels = read_u32(&mut f)?;
    let len = read_u64(&mut f)? as usize;
    let expected = width as usize * height as usize * channels as usize;
    if width == 0 || height == 0 || channels == 0 || len != expected {
        return None;
    }
    let mut bytes = vec![0u8; len.checked_mul(4)?];
    f.read_exact(&mut bytes).ok()?;
    let data = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Some(crate::splat_backend::TextureImage {
        width,
        height,
        channels,
        data,
        native: None,
    })
}

fn write_texture_cache_file(key: &str, tex: &crate::splat_backend::TextureImage) {
    let dir = texture_cache_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let final_path = dir.join(format!("{key}.uhtx"));
    let tmp_path = dir.join(format!("{key}.tmp"));
    let Ok(mut f) = std::fs::File::create(&tmp_path) else {
        return;
    };
    if f.write_all(TEX_CACHE_MAGIC).is_err()
        || f.write_all(&tex.width.to_le_bytes()).is_err()
        || f.write_all(&tex.height.to_le_bytes()).is_err()
        || f.write_all(&tex.channels.to_le_bytes()).is_err()
        || f.write_all(&(tex.data.len() as u64).to_le_bytes()).is_err()
    {
        let _ = std::fs::remove_file(&tmp_path);
        return;
    }
    for v in &tex.data {
        if f.write_all(&v.to_le_bytes()).is_err() {
            let _ = std::fs::remove_file(&tmp_path);
            return;
        }
    }
    if f.flush().is_ok() {
        match std::fs::rename(&tmp_path, &final_path) {
            Ok(()) => {}
            Err(_) => {
                let _ = std::fs::remove_file(&final_path);
                let _ = std::fs::rename(tmp_path, final_path);
            }
        }
    }
}

fn load_cached_texture_arc(
    path: &str,
    mode: &str,
    decode: impl FnOnce(&Path) -> Option<crate::splat_backend::TextureImage>,
) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let path_ref = Path::new(path);
    if !texture_cache_enabled() {
        return decode(path_ref).map(Arc::new);
    }
    let key = texture_cache_key(path_ref, mode)?;
    if let Some(hit) = texture_memory_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .cloned()
    {
        return Some(hit);
    }
    if texture_disk_cache_enabled() {
        let disk_path = texture_cache_dir().join(format!("{key}.uhtx"));
        if let Some(tex) = read_texture_cache_file(&disk_path) {
            let tex = Arc::new(tex);
            texture_memory_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key, Arc::clone(&tex));
            return Some(tex);
        }
    }
    let tex = decode(path_ref)?;
    if texture_disk_cache_enabled() {
        write_texture_cache_file(&key, &tex);
    }
    let tex = Arc::new(tex);
    texture_memory_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key, Arc::clone(&tex));
    Some(tex)
}

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

/// Load a base-colour texture file into a LINEAR RGBA `TextureImage` for resident
/// material texture upload. Returns `None` if the file is missing/undecodable.
///
/// RGBA (not RGB) so the source ALPHA channel survives: PolyHaven foliage leaf
/// cards carry the leaf cutout in the base-color alpha. The path tracer's
/// alpha-cutout test reads this `.w` (megakernel.slang ~2345) ONLY for materials
/// whose `opacity_tex == albedo_tex` (the scatter/vegetation foliage signal);
/// opaque facades keep `opacity_tex = -1` so their (typically opaque, alpha=1)
/// channel is ignored — buildings render identically. sRGB is decoded on RGB;
/// the alpha channel is a LINEAR coverage signal and is NOT gamma-decoded.
pub fn load_linear_texture(path: &str) -> Option<crate::splat_backend::TextureImage> {
    load_linear_texture_arc(path).map(|tex| (*tex).clone())
}

/// Shared-handle variant of [`load_linear_texture`]. Live scene assembly uses
/// this so RAM-cache hits do not clone the decoded f32 texture payload before
/// atlas construction reads it.
pub fn load_linear_texture_arc(path: &str) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let path_ref = Path::new(path);
    if is_dds_path(path_ref) {
        return load_dds_texture_arc(path_ref, "linear_rgba_dds_v1", 4, DdsMirrorMode::SrgbColor);
    }
    load_cached_texture_arc(path, "linear_rgba_v1", |path| {
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
            native: None,
        })
    })
}

/// Load a DATA texture (normal / roughness / displacement) into a LINEAR atlas
/// image WITHOUT sRGB decoding — these maps are linear data, not colour, and
/// gamma-decoding them corrupts the surface (a normal map decoded as sRGB tilts
/// every normal). `channels` selects the layout: 3 = tangent-space normal (RGB),
/// 1 = single-channel roughness/height (luma). Returns `None` on a bad file so a
/// missing map falls back to flat (slot -1) and never corrupts sampling.
pub fn load_data_texture(path: &str, channels: u32) -> Option<crate::splat_backend::TextureImage> {
    load_data_texture_arc(path, channels).map(|tex| (*tex).clone())
}

/// Shared-handle variant of [`load_data_texture`]. Avoids cloning cached normal,
/// roughness, and displacement payloads on structural scene rebuilds.
pub fn load_data_texture_arc(
    path: &str,
    channels: u32,
) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let path_ref = Path::new(path);
    if is_dds_path(path_ref) {
        let mode = format!("data_ch{channels}_dds_v1");
        return load_dds_texture_arc(path_ref, &mode, channels, DdsMirrorMode::LinearData);
    }
    let mode = format!("data_ch{channels}_v1");
    load_cached_texture_arc(path, &mode, |path| {
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
            native: None,
        })
    })
}
