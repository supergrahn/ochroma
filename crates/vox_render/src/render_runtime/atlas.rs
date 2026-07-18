//! Texture / HDRI loader utilities — engine-side so games don't re-implement
//! them. Gated on `spectra-native` because they return `splat_backend::TextureImage`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
#[cfg(any(test, not(feature = "aot-shaders")))]
use std::io::{Read, Write};
use std::path::Path;
#[cfg(any(test, not(feature = "aot-shaders")))]
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

#[cfg(any(test, not(feature = "aot-shaders")))]
use vox_core::{linear_to_srgb, srgb_to_linear};

#[cfg(any(test, not(feature = "aot-shaders")))]
const TEX_CACHE_MAGIC: &[u8; 8] = b"UHTEX01\0";
/// v2 adds an optional GPU-native BCn payload after the f32 mirror so the
/// load-time BC7/BC5/BC4 compression cost is paid once, not per run.
#[cfg(any(test, not(feature = "aot-shaders")))]
const TEX_CACHE_MAGIC_V2: &[u8; 8] = b"UHTEX02\0";

#[cfg(any(test, not(feature = "aot-shaders")))]
type TextureCache = HashMap<String, Arc<crate::splat_backend::TextureImage>>;

/// Immutable GPU-native material texture loaded from an offline-cooked DDS.
///
/// Unlike [`crate::splat_backend::TextureImage`], this carrier deliberately has
/// no decoded f32 mirror. Product runtime code may parse and validate the DDS
/// container, retain its authored mip bytes, and upload them; it must not
/// recreate texture data or render mips.
#[derive(Debug, Clone)]
pub struct CookedNativeTexture {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub texture: crate::RendererTexture2D,
}

type CookedNativeTextureCache = HashMap<String, Arc<CookedNativeTexture>>;

#[cfg(any(test, not(feature = "aot-shaders")))]
#[derive(Clone, Copy)]
enum DdsMirrorMode {
    SrgbColor,
    LinearData,
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn texture_memory_cache() -> &'static Mutex<TextureCache> {
    static CACHE: OnceLock<Mutex<TextureCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cooked_native_texture_cache() -> &'static Mutex<CookedNativeTextureCache> {
    static CACHE: OnceLock<Mutex<CookedNativeTextureCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn texture_cache_enabled() -> bool {
    std::env::var("OCHROMA_TEXTURE_CACHE")
        .map(|v| v.trim() != "0")
        .unwrap_or(true)
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn texture_disk_cache_enabled() -> bool {
    std::env::var("OCHROMA_TEXTURE_DISK_CACHE")
        .map(|v| !matches!(v.trim(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}

#[cfg(any(test, not(feature = "aot-shaders")))]
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

#[cfg(any(test, not(feature = "aot-shaders")))]
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

/// Semantic class of a texture for load-time BCn compression when no cooked
/// DDS exists. Mirrors the cook contract (`material_texture_cook`): color →
/// BC7 sRGB, tangent-space normal XY → BC5, single-channel scalar → BC4.
#[cfg(any(test, not(feature = "aot-shaders")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeCompressKind {
    /// sRGB-encoded colour (albedo). Skipped when the image carries a real
    /// alpha cutout so the coverage-preserving RGBA8 mip path keeps distant
    /// foliage dense.
    SrgbColor,
    /// Tangent-space normal; X/Y survive, Z is reconstructed at sample time.
    NormalXy,
    /// Roughness / displacement / AO — single channel.
    Scalar,
}

/// Whether runtime BCn compression of JPEG/PNG sources is enabled
/// (`OCHROMA_TEXTURE_BCN=0` disables; default on).
#[cfg(any(test, not(feature = "aot-shaders")))]
fn runtime_bcn_enabled() -> bool {
    std::env::var("OCHROMA_TEXTURE_BCN")
        .map(|v| !matches!(v.trim(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}

/// Format-selection helper: which BCn target (image_dds encode format + GPU
/// upload format) a decoded texture compresses to, or `None` when it must stay
/// on the uncompressed RGBA8 fallback (cutout alpha, non-block-aligned size).
#[cfg(any(test, not(feature = "aot-shaders")))]
pub(crate) fn select_native_compress_format(
    kind: NativeCompressKind,
    width: u32,
    height: u32,
    has_cutout_alpha: bool,
) -> Option<(image_dds::ImageFormat, spectra_gpu::GpuTextureFormat)> {
    if width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0 {
        return None;
    }
    match kind {
        NativeCompressKind::SrgbColor if has_cutout_alpha => None,
        NativeCompressKind::SrgbColor => Some((
            image_dds::ImageFormat::BC7RgbaUnormSrgb,
            spectra_gpu::GpuTextureFormat::Bc7UnormSrgb,
        )),
        NativeCompressKind::NormalXy => Some((
            image_dds::ImageFormat::BC5RgUnorm,
            spectra_gpu::GpuTextureFormat::Bc5Unorm,
        )),
        NativeCompressKind::Scalar => Some((
            image_dds::ImageFormat::BC4RUnorm,
            spectra_gpu::GpuTextureFormat::Bc4Unorm,
        )),
    }
}

/// True when a colour mirror carries a real alpha cutout (foliage leaf cards):
/// those must keep the coverage-preserving RGBA8 mip path.
#[cfg(any(test, not(feature = "aot-shaders")))]
pub(crate) fn texture_has_cutout_alpha(tex: &crate::splat_backend::TextureImage) -> bool {
    tex.channels == 4
        && tex
            .data
            .chunks_exact(4)
            .any(|px| px[3] < 254.5 / 255.0)
}

/// Probe for a precompressed DDS sibling next to a JPEG/PNG source
/// (`foo.jpg` → `foo.dds`). Cook outputs that land next to the source win over
/// runtime compression.
#[cfg(any(test, not(feature = "aot-shaders")))]
pub(crate) fn sibling_dds_path(path: &Path) -> Option<PathBuf> {
    if is_dds_path(path) {
        return None;
    }
    let candidate = path.with_extension("dds");
    (candidate != path && candidate.is_file()).then_some(candidate)
}

/// Compress a decoded f32 mirror into a GPU-native BCn payload with a full
/// generated mip chain (`Quality::Fast` intel_tex ISPC encode). Returns `None`
/// when the texture must stay on the RGBA8 fallback.
#[cfg(any(test, not(feature = "aot-shaders")))]
fn compress_native_bcn(
    tex: &crate::splat_backend::TextureImage,
    kind: NativeCompressKind,
) -> Option<crate::RendererTexture2D> {
    let has_cutout =
        kind == NativeCompressKind::SrgbColor && texture_has_cutout_alpha(tex);
    let (image_format, gpu_format) =
        select_native_compress_format(kind, tex.width, tex.height, has_cutout)?;
    let texels = tex.width as usize * tex.height as usize;
    let channels = tex.channels as usize;
    if tex.data.len() != texels * channels {
        return None;
    }
    let mut rgba = vec![0u8; texels * 4];
    for (i, px) in tex.data.chunks_exact(channels).enumerate() {
        let dst = i * 4;
        match kind {
            NativeCompressKind::SrgbColor => {
                // Mirror is linear; the BCn payload stores sRGB bytes and the
                // GPU sRGB sampler decodes back to linear (same contract as
                // cooked BC7 sRGB DDS).
                rgba[dst] = f32_to_unorm8(linear_to_srgb(px[0]));
                rgba[dst + 1] = f32_to_unorm8(linear_to_srgb(*px.get(1).unwrap_or(&0.0)));
                rgba[dst + 2] = f32_to_unorm8(linear_to_srgb(*px.get(2).unwrap_or(&0.0)));
                rgba[dst + 3] = px.get(3).map_or(255, |a| f32_to_unorm8(*a));
            }
            NativeCompressKind::NormalXy => {
                rgba[dst] = f32_to_unorm8(px[0]);
                rgba[dst + 1] = f32_to_unorm8(*px.get(1).unwrap_or(&0.5));
                rgba[dst + 2] = 0;
                rgba[dst + 3] = 255;
            }
            NativeCompressKind::Scalar => {
                let v = f32_to_unorm8(px[0]);
                rgba[dst] = v;
                rgba[dst + 1] = v;
                rgba[dst + 2] = v;
                rgba[dst + 3] = 255;
            }
        }
    }
    let surface = image_dds::SurfaceRgba8 {
        width: tex.width,
        height: tex.height,
        depth: 1,
        layers: 1,
        mipmaps: 1,
        data: rgba.as_slice(),
    };
    let encoded = surface
        .encode(
            image_format,
            image_dds::Quality::Fast,
            image_dds::Mipmaps::GeneratedAutomatic,
        )
        .ok()?;
    let (_, _, block_extent, block_bytes) = dds_format_to_gpu(encoded.image_format)?;
    let mips = dds_native_mips(&encoded, block_extent, block_bytes)?;
    Some(crate::RendererTexture2D::new(
        gpu_format,
        tex.width,
        tex.height,
        tex.channels,
        mips,
    ))
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn f32_to_unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn dds_native_mips<T: AsRef<[u8]>>(
    surface: &image_dds::Surface<T>,
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

#[cfg(any(test, not(feature = "aot-shaders")))]
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

#[cfg(any(test, not(feature = "aot-shaders")))]
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

/// Read one offline-cooked DDS without decoding it to pixels or generating
/// mips. The complete authored mip chain is retained byte-for-byte for GPU
/// upload. A non-DDS path, malformed container, unsupported format, mismatched
/// logical channel contract, or incomplete/non-block-aligned payload fails
/// closed with `None`.
fn load_cooked_native_dds(path: &Path, requested_channels: u32) -> Option<CookedNativeTexture> {
    if !is_dds_path(path) || !(1..=4).contains(&requested_channels) {
        return None;
    }
    let mut file = std::fs::File::open(path).ok()?;
    let dds = image_dds::ddsfile::Dds::read(&mut file).ok()?;
    let surface = image_dds::Surface::from_dds(&dds).ok()?;
    if surface.depth != 1
        || surface.layers != 1
        || surface.width == 0
        || surface.height == 0
        || surface.mipmaps == 0
    {
        return None;
    }
    let expected_mips = u32::BITS - surface.width.max(surface.height).leading_zeros();
    if surface.mipmaps != expected_mips {
        return None;
    }
    let (gpu_format, logical_channels, block_extent, block_bytes) =
        dds_format_to_gpu(surface.image_format)?;
    let bc5_normal_as_rgb = requested_channels == 3 && dds_format_is_bc5(surface.image_format);
    if requested_channels > logical_channels && !bc5_normal_as_rgb {
        return None;
    }
    if block_extent > 1 && (surface.width % block_extent != 0 || surface.height % block_extent != 0)
    {
        return None;
    }
    let mips = dds_native_mips(&surface, block_extent, block_bytes)?;
    if mips.len() != surface.mipmaps as usize {
        return None;
    }
    let texture = crate::RendererTexture2D::new(
        gpu_format,
        surface.width,
        surface.height,
        requested_channels,
        mips,
    );
    eprintln!(
        "[texture-dds] loaded cooked native {} format={:?} mips={} mirror=none",
        path.display(),
        gpu_format,
        texture.mips.len()
    );
    Some(CookedNativeTexture {
        width: surface.width,
        height: surface.height,
        channels: requested_channels,
        texture,
    })
}

fn load_cooked_native_dds_arc(
    path: &Path,
    mode: &str,
    requested_channels: u32,
) -> Option<Arc<CookedNativeTexture>> {
    let key = format!("cooked-native:{}", texture_cache_key(path, mode)?);
    if let Ok(cache) = cooked_native_texture_cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return Some(Arc::clone(hit));
    }
    let texture = Arc::new(load_cooked_native_dds(path, requested_channels)?);
    if let Ok(mut cache) = cooked_native_texture_cache().lock() {
        cache.insert(key, Arc::clone(&texture));
    }
    Some(texture)
}

/// Load an exact offline-cooked sRGB colour DDS for direct GPU upload.
/// Source JPEG/PNG files and sibling discovery are intentionally unsupported:
/// the product resolver must provide the manifest-pinned cooked path.
pub fn load_cooked_linear_texture_arc(path: &str) -> Option<Arc<CookedNativeTexture>> {
    let texture = load_cooked_native_dds_arc(Path::new(path), "cooked_linear_dds_v1", 4)?;
    matches!(
        texture.texture.format,
        spectra_gpu::GpuTextureFormat::Rgba8UnormSrgb
            | spectra_gpu::GpuTextureFormat::Bc1UnormSrgb
            | spectra_gpu::GpuTextureFormat::Bc3UnormSrgb
            | spectra_gpu::GpuTextureFormat::Bc7UnormSrgb
    )
    .then_some(texture)
}

/// Load an exact offline-cooked linear-data DDS for direct GPU upload.
pub fn load_cooked_data_texture_arc(path: &str, channels: u32) -> Option<Arc<CookedNativeTexture>> {
    let texture = load_cooked_native_dds_arc(
        Path::new(path),
        &format!("cooked_data_ch{channels}_dds_v1"),
        channels,
    )?;
    match channels {
        1 if texture.texture.format == spectra_gpu::GpuTextureFormat::Bc4Unorm => Some(texture),
        3 if texture.texture.format == spectra_gpu::GpuTextureFormat::Bc5Unorm => Some(texture),
        _ => None,
    }
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// Stable on-disk tag for a `GpuTextureFormat` in the v2 texture cache.
#[cfg(any(test, not(feature = "aot-shaders")))]
fn gpu_format_cache_tag(format: spectra_gpu::GpuTextureFormat) -> Option<u32> {
    use spectra_gpu::GpuTextureFormat as G;
    Some(match format {
        G::Rgba8Unorm => 1,
        G::Rgba8UnormSrgb => 2,
        G::Bc1Unorm => 3,
        G::Bc1UnormSrgb => 4,
        G::Bc3Unorm => 5,
        G::Bc3UnormSrgb => 6,
        G::Bc4Unorm => 7,
        G::Bc5Unorm => 8,
        G::Bc7Unorm => 9,
        G::Bc7UnormSrgb => 10,
    })
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn gpu_format_from_cache_tag(tag: u32) -> Option<spectra_gpu::GpuTextureFormat> {
    use spectra_gpu::GpuTextureFormat as G;
    Some(match tag {
        1 => G::Rgba8Unorm,
        2 => G::Rgba8UnormSrgb,
        3 => G::Bc1Unorm,
        4 => G::Bc1UnormSrgb,
        5 => G::Bc3Unorm,
        6 => G::Bc3UnormSrgb,
        7 => G::Bc4Unorm,
        8 => G::Bc5Unorm,
        9 => G::Bc7Unorm,
        10 => G::Bc7UnormSrgb,
        _ => return None,
    })
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn read_native_cache_section<R: Read>(r: &mut R) -> Option<Option<crate::RendererTexture2D>> {
    let mut flag = [0u8; 1];
    r.read_exact(&mut flag).ok()?;
    if flag[0] == 0 {
        return Some(None);
    }
    let format = gpu_format_from_cache_tag(read_u32(r)?)?;
    let width = read_u32(r)?;
    let height = read_u32(r)?;
    let channels = read_u32(r)?;
    let mip_count = read_u32(r)? as usize;
    if width == 0 || height == 0 || !(1..=4).contains(&channels) || !(1..=20).contains(&mip_count) {
        return None;
    }
    let mut mips = Vec::with_capacity(mip_count);
    for _ in 0..mip_count {
        let mw = read_u32(r)?;
        let mh = read_u32(r)?;
        let pitch = read_u32(r)? as usize;
        let len = read_u64(r)? as usize;
        if mw == 0 || mh == 0 || len == 0 || len > (1usize << 30) {
            return None;
        }
        let mut data = vec![0u8; len];
        r.read_exact(&mut data).ok()?;
        mips.push(crate::RendererTextureMip {
            width: mw,
            height: mh,
            row_pitch_bytes: pitch,
            data,
        });
    }
    Some(Some(crate::RendererTexture2D::new(
        format, width, height, channels, mips,
    )))
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn read_texture_cache_file(path: &Path) -> Option<crate::splat_backend::TextureImage> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    let has_native_section = if &magic == TEX_CACHE_MAGIC_V2 {
        true
    } else if &magic == TEX_CACHE_MAGIC {
        false
    } else {
        return None;
    };
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
    let native = if has_native_section {
        read_native_cache_section(&mut f)?
    } else {
        None
    };
    Some(crate::splat_backend::TextureImage {
        width,
        height,
        channels,
        data,
        native,
    })
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn write_native_cache_section<W: Write>(
    w: &mut W,
    native: Option<&crate::RendererTexture2D>,
) -> std::io::Result<()> {
    let Some(native) = native else {
        return w.write_all(&[0u8]);
    };
    let Some(tag) = gpu_format_cache_tag(native.format) else {
        return w.write_all(&[0u8]);
    };
    w.write_all(&[1u8])?;
    w.write_all(&tag.to_le_bytes())?;
    w.write_all(&native.width.to_le_bytes())?;
    w.write_all(&native.height.to_le_bytes())?;
    w.write_all(&native.channels.to_le_bytes())?;
    w.write_all(&(native.mips.len() as u32).to_le_bytes())?;
    for mip in &native.mips {
        w.write_all(&mip.width.to_le_bytes())?;
        w.write_all(&mip.height.to_le_bytes())?;
        w.write_all(&(mip.row_pitch_bytes as u32).to_le_bytes())?;
        w.write_all(&(mip.data.len() as u64).to_le_bytes())?;
        w.write_all(&mip.data)?;
    }
    Ok(())
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn write_texture_cache_file(key: &str, tex: &crate::splat_backend::TextureImage) {
    let dir = texture_cache_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let final_path = dir.join(format!("{key}.uhtx"));
    let tmp_path = dir.join(format!("{key}.tmp"));
    let Ok(file) = std::fs::File::create(&tmp_path) else {
        return;
    };
    let mut f = std::io::BufWriter::new(file);
    if f.write_all(TEX_CACHE_MAGIC_V2).is_err()
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
    if write_native_cache_section(&mut f, tex.native.as_ref()).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return;
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

/// Attach a load-time-compressed BCn native payload when the decoded texture
/// has none and runtime compression is enabled. Returns whether a payload was
/// added (so callers know to refresh the disk cache).
#[cfg(any(test, not(feature = "aot-shaders")))]
fn ensure_native_bcn(
    tex: &mut crate::splat_backend::TextureImage,
    compress: Option<NativeCompressKind>,
) -> bool {
    let Some(kind) = compress else {
        return false;
    };
    if tex.native.is_some() || !runtime_bcn_enabled() {
        return false;
    }
    match compress_native_bcn(tex, kind) {
        Some(native) => {
            tex.native = Some(native);
            true
        }
        None => false,
    }
}

#[cfg(any(test, not(feature = "aot-shaders")))]
fn load_cached_texture_arc(
    path: &str,
    mode: &str,
    compress: Option<NativeCompressKind>,
    decode: impl FnOnce(&Path) -> Option<crate::splat_backend::TextureImage>,
) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let path_ref = Path::new(path);
    if !texture_cache_enabled() {
        let mut tex = decode(path_ref)?;
        ensure_native_bcn(&mut tex, compress);
        return Some(Arc::new(tex));
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
        if let Some(mut tex) = read_texture_cache_file(&disk_path) {
            // A v1 (or pre-BCn) cache entry lacks the native payload: compress
            // once and upgrade the file in place.
            if ensure_native_bcn(&mut tex, compress) {
                write_texture_cache_file(&key, &tex);
            }
            let tex = Arc::new(tex);
            texture_memory_cache()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(key, Arc::clone(&tex));
            return Some(tex);
        }
    }
    let mut tex = decode(path_ref)?;
    ensure_native_bcn(&mut tex, compress);
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
#[cfg(any(test, not(feature = "aot-shaders")))]
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
#[cfg(any(test, not(feature = "aot-shaders")))]
pub fn load_linear_texture(path: &str) -> Option<crate::splat_backend::TextureImage> {
    load_linear_texture_arc(path).map(|tex| (*tex).clone())
}

/// Shared-handle variant of [`load_linear_texture`]. Live scene assembly uses
/// this so RAM-cache hits do not clone the decoded f32 texture payload before
/// atlas construction reads it.
#[cfg(any(test, not(feature = "aot-shaders")))]
pub fn load_linear_texture_arc(path: &str) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let path_ref = Path::new(path);
    if is_dds_path(path_ref) {
        return load_dds_texture_arc(path_ref, "linear_rgba_dds_v1", 4, DdsMirrorMode::SrgbColor);
    }
    // A precompressed DDS sibling (cook output copied next to the source) wins
    // over runtime compression.
    if let Some(dds) = sibling_dds_path(path_ref)
        && let Some(tex) =
            load_dds_texture_arc(&dds, "linear_rgba_dds_v1", 4, DdsMirrorMode::SrgbColor)
    {
        return Some(tex);
    }
    load_cached_texture_arc(
        path,
        "linear_rgba_v1",
        Some(NativeCompressKind::SrgbColor),
        |path| {
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
        },
    )
}

/// Load a DATA texture (normal / roughness / displacement) into a LINEAR atlas
/// image WITHOUT sRGB decoding — these maps are linear data, not colour, and
/// gamma-decoding them corrupts the surface (a normal map decoded as sRGB tilts
/// every normal). `channels` selects the layout: 3 = tangent-space normal (RGB),
/// 1 = single-channel roughness/height (luma). Returns `None` on a bad file so a
/// missing map falls back to flat (slot -1) and never corrupts sampling.
#[cfg(any(test, not(feature = "aot-shaders")))]
pub fn load_data_texture(path: &str, channels: u32) -> Option<crate::splat_backend::TextureImage> {
    load_data_texture_arc(path, channels).map(|tex| (*tex).clone())
}

/// Shared-handle variant of [`load_data_texture`]. Avoids cloning cached normal,
/// roughness, and displacement payloads on structural scene rebuilds.
#[cfg(any(test, not(feature = "aot-shaders")))]
pub fn load_data_texture_arc(
    path: &str,
    channels: u32,
) -> Option<Arc<crate::splat_backend::TextureImage>> {
    let path_ref = Path::new(path);
    if is_dds_path(path_ref) {
        let mode = format!("data_ch{channels}_dds_v1");
        return load_dds_texture_arc(path_ref, &mode, channels, DdsMirrorMode::LinearData);
    }
    // A precompressed DDS sibling (cook output copied next to the source) wins
    // over runtime compression.
    if let Some(dds) = sibling_dds_path(path_ref) {
        let mode = format!("data_ch{channels}_dds_v1");
        if let Some(tex) = load_dds_texture_arc(&dds, &mode, channels, DdsMirrorMode::LinearData) {
            return Some(tex);
        }
    }
    let mode = format!("data_ch{channels}_v1");
    // `channels == 3` is the tangent-space-normal layout (see doc comment);
    // single-channel data compresses to BC4. Other channel counts stay RGBA8.
    let compress = match channels {
        3 => Some(NativeCompressKind::NormalXy),
        1 => Some(NativeCompressKind::Scalar),
        _ => None,
    };
    load_cached_texture_arc(path, &mode, compress, |path| {
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

#[cfg(test)]
mod tests {
    use super::*;
    use spectra_gpu::GpuTextureFormat as G;

    fn solid_texture(
        width: u32,
        height: u32,
        channels: u32,
        px: &[f32],
    ) -> crate::splat_backend::TextureImage {
        assert_eq!(px.len(), channels as usize);
        let mut data = Vec::with_capacity((width * height * channels) as usize);
        for _ in 0..width * height {
            data.extend_from_slice(px);
        }
        crate::splat_backend::TextureImage {
            width,
            height,
            channels,
            data,
            native: None,
        }
    }

    fn write_encoded_dds(
        path: &Path,
        format: image_dds::ImageFormat,
        mipmaps: image_dds::Mipmaps,
    ) {
        let rgba = vec![128_u8, 64, 32, 255].repeat(8 * 8);
        let surface = image_dds::SurfaceRgba8 {
            width: 8,
            height: 8,
            depth: 1,
            layers: 1,
            mipmaps: 1,
            data: rgba.as_slice(),
        };
        let dds = surface
            .encode(format, image_dds::Quality::Fast, mipmaps)
            .unwrap()
            .to_dds()
            .unwrap();
        let mut bytes = Vec::new();
        dds.write(&mut bytes).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn format_selection_matches_cook_contract() {
        assert_eq!(
            select_native_compress_format(NativeCompressKind::SrgbColor, 1024, 1024, false),
            Some((image_dds::ImageFormat::BC7RgbaUnormSrgb, G::Bc7UnormSrgb))
        );
        assert_eq!(
            select_native_compress_format(NativeCompressKind::NormalXy, 1024, 1024, false),
            Some((image_dds::ImageFormat::BC5RgUnorm, G::Bc5Unorm))
        );
        assert_eq!(
            select_native_compress_format(NativeCompressKind::Scalar, 1024, 1024, false),
            Some((image_dds::ImageFormat::BC4RUnorm, G::Bc4Unorm))
        );
        // Cutout colour stays on the coverage-preserving RGBA8 path.
        assert_eq!(
            select_native_compress_format(NativeCompressKind::SrgbColor, 1024, 1024, true),
            None
        );
        // Non-block-aligned sizes cannot upload as BCn.
        assert_eq!(
            select_native_compress_format(NativeCompressKind::SrgbColor, 1022, 1024, false),
            None
        );
    }

    #[test]
    fn cutout_alpha_detection_reads_alpha_channel() {
        let opaque = solid_texture(4, 4, 4, &[0.2, 0.4, 0.6, 1.0]);
        assert!(!texture_has_cutout_alpha(&opaque));
        let mut cutout = opaque.clone();
        cutout.data[7] = 0.3; // one texel's alpha
        assert!(texture_has_cutout_alpha(&cutout));
        // 3-channel data textures never count as cutout.
        let data3 = solid_texture(4, 4, 3, &[0.5, 0.5, 1.0]);
        assert!(!texture_has_cutout_alpha(&data3));
    }

    #[test]
    fn compress_native_bcn_produces_bc7_mip_chain_that_decodes_back() {
        let tex = solid_texture(8, 8, 4, &[0.5, 0.25, 0.125, 1.0]);
        let native = compress_native_bcn(&tex, NativeCompressKind::SrgbColor)
            .expect("opaque block-aligned colour must compress");
        assert_eq!(native.format, G::Bc7UnormSrgb);
        // 8x8 -> 4 mips (8,4,2,1); every level present for ray-cone LOD.
        assert_eq!(native.mips.len(), 4);
        assert_eq!(
            (native.mips[0].width, native.mips[0].height),
            (8, 8)
        );
        // 8x8 = 2x2 BC7 blocks of 16 bytes.
        assert_eq!(native.mips[0].data.len(), 4 * 16);
        assert_eq!(native.mips[0].row_pitch_bytes, 2 * 16);

        // Decode the top mip and check the sRGB-encoded payload reproduces the
        // linear mirror colour after sRGB decode.
        let surface = image_dds::Surface {
            width: 8,
            height: 8,
            depth: 1,
            layers: 1,
            mipmaps: 1,
            image_format: image_dds::ImageFormat::BC7RgbaUnormSrgb,
            data: native.mips[0].data.as_slice(),
        };
        let decoded = surface.decode_layers_mipmaps_rgbaf32(0..1, 0..1).unwrap();
        let px = &decoded.data[..4];
        // Decoded texels are raw sRGB-encoded values in [0,1].
        assert!((srgb_to_linear(px[0]) - 0.5).abs() < 0.03, "r={}", px[0]);
        assert!((srgb_to_linear(px[1]) - 0.25).abs() < 0.03, "g={}", px[1]);
        assert!((srgb_to_linear(px[2]) - 0.125).abs() < 0.03, "b={}", px[2]);
    }

    #[test]
    fn compress_native_bcn_normal_and_scalar_formats() {
        let normal = solid_texture(8, 8, 3, &[0.5, 0.5, 1.0]);
        let native = compress_native_bcn(&normal, NativeCompressKind::NormalXy).unwrap();
        assert_eq!(native.format, G::Bc5Unorm);
        // BC5: 16 bytes per 4x4 block.
        assert_eq!(native.mips[0].data.len(), 4 * 16);

        let rough = solid_texture(8, 8, 1, &[0.7]);
        let native = compress_native_bcn(&rough, NativeCompressKind::Scalar).unwrap();
        assert_eq!(native.format, G::Bc4Unorm);
        // BC4: 8 bytes per 4x4 block.
        assert_eq!(native.mips[0].data.len(), 4 * 8);
        let surface = image_dds::Surface {
            width: 8,
            height: 8,
            depth: 1,
            layers: 1,
            mipmaps: 1,
            image_format: image_dds::ImageFormat::BC4RUnorm,
            data: native.mips[0].data.as_slice(),
        };
        let decoded = surface.decode_layers_mipmaps_rgbaf32(0..1, 0..1).unwrap();
        assert!((decoded.data[0] - 0.7).abs() < 0.01, "r={}", decoded.data[0]);
    }

    #[test]
    fn cutout_colour_falls_back_to_uncompressed() {
        let mut tex = solid_texture(8, 8, 4, &[0.5, 0.25, 0.125, 1.0]);
        tex.data[3] = 0.4; // real cutout alpha
        assert!(compress_native_bcn(&tex, NativeCompressKind::SrgbColor).is_none());
    }

    #[test]
    fn sibling_dds_probe_finds_cook_output_next_to_source() {
        let dir = tempfile::tempdir().unwrap();
        let jpg = dir.path().join("diffuse.jpg");
        std::fs::write(&jpg, b"not a real jpeg").unwrap();
        assert_eq!(sibling_dds_path(&jpg), None);
        let dds = dir.path().join("diffuse.dds");
        std::fs::write(&dds, b"not a real dds").unwrap();
        assert_eq!(sibling_dds_path(&jpg), Some(dds.clone()));
        // A DDS source never probes for a sibling of itself.
        assert_eq!(sibling_dds_path(&dds), None);
    }

    #[test]
    fn cooked_native_loader_keeps_authored_bc7_mips_without_float_decode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("albedo.dds");
        write_encoded_dds(
            &path,
            image_dds::ImageFormat::BC7RgbaUnormSrgb,
            image_dds::Mipmaps::GeneratedAutomatic,
        );

        let cooked = load_cooked_linear_texture_arc(path.to_str().unwrap()).unwrap();
        assert_eq!(cooked.texture.format, G::Bc7UnormSrgb);
        assert_eq!(cooked.texture.mips.len(), 4);
        assert_eq!((cooked.width, cooked.height, cooked.channels), (8, 8, 4));
    }

    #[test]
    fn cooked_native_loader_rejects_missing_mip_chain_and_wrong_semantic_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("incomplete.dds");
        write_encoded_dds(
            &path,
            image_dds::ImageFormat::BC5RgUnorm,
            image_dds::Mipmaps::Disabled,
        );
        assert!(load_cooked_data_texture_arc(path.to_str().unwrap(), 3).is_none());

        let scalar_path = dir.path().join("scalar.dds");
        write_encoded_dds(
            &scalar_path,
            image_dds::ImageFormat::BC4RUnorm,
            image_dds::Mipmaps::GeneratedAutomatic,
        );
        assert!(load_cooked_data_texture_arc(scalar_path.to_str().unwrap(), 3).is_none());
    }

    #[test]
    fn disk_cache_v2_roundtrips_native_payload_and_reads_v1() {
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: test-local env override for the cache directory.
        unsafe { std::env::set_var("OCHROMA_TEXTURE_CACHE_DIR", dir.path()) };

        let mut tex = solid_texture(8, 8, 4, &[0.5, 0.25, 0.125, 1.0]);
        tex.native = compress_native_bcn(&tex, NativeCompressKind::SrgbColor);
        assert!(tex.native.is_some());

        write_texture_cache_file("roundtrip", &tex);
        let path = texture_cache_dir().join("roundtrip.uhtx");
        let read = read_texture_cache_file(&path).expect("v2 cache entry must parse");
        assert_eq!(read.width, 8);
        assert_eq!(read.data, tex.data);
        let native = read.native.expect("native payload survives the cache");
        let orig = tex.native.as_ref().unwrap();
        assert_eq!(native.format, orig.format);
        assert_eq!(native.mips.len(), orig.mips.len());
        assert_eq!(native.mips[0].data, orig.mips[0].data);

        // Legacy v1 entries (no native section) still read as native: None.
        let mut v1 = Vec::new();
        v1.extend_from_slice(TEX_CACHE_MAGIC);
        v1.extend_from_slice(&2u32.to_le_bytes());
        v1.extend_from_slice(&2u32.to_le_bytes());
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&4u64.to_le_bytes());
        for v in [0.1f32, 0.2, 0.3, 0.4] {
            v1.extend_from_slice(&v.to_le_bytes());
        }
        let v1_path = dir.path().join("legacy.uhtx");
        std::fs::write(&v1_path, v1).unwrap();
        let legacy = read_texture_cache_file(&v1_path).expect("v1 cache entry must parse");
        assert_eq!(legacy.channels, 1);
        assert!((legacy.data[3] - 0.4).abs() < 1e-6);
        assert!(legacy.native.is_none());

        unsafe { std::env::remove_var("OCHROMA_TEXTURE_CACHE_DIR") };
    }
}
