//! Test harness for the spectra-native path-trace entry points,
//! split by domain from the in-file `mod tests` block (same module
//! tree: children reach `splat_backend` privates via `use super::super`).
//! Shared loaders/rigs/rasterizer/PNG helpers live in this file.

#![allow(unused_imports)]

use super::*;

mod core;
mod materials;
mod mesh_render;
mod perf;
mod perturbation;
mod sdf;
mod terrain_carve;

use super::SpectraRenderBackend;

// ---- M0: native SDF-volume primitive (confetti → wall) ----------------

/// Minimal dependency-free PNG writer (RGBA8, zlib stored blocks). Mirrors
/// vox_app::shell::cpu_render::write_png so the SDF still can be inspected.
#[cfg(feature = "spectra-native")]
fn write_png_rgba(path: &str, rgba: &[u8], w: u32, h: u32) {
    use std::io::Write;
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in bytes {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut crc_in = Vec::with_capacity(4 + data.len());
        crc_in.extend_from_slice(kind);
        crc_in.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    }
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    let stride = (w * 4) as usize;
    for y in 0..h as usize {
        raw.push(0u8);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }
    // zlib stored blocks (no deflate dependency).
    let mut comp = vec![0x78u8, 0x01u8];
    let mut i = 0;
    while i < raw.len() {
        let block = (raw.len() - i).min(0xFFFF);
        let is_last = i + block >= raw.len();
        comp.push(if is_last { 1 } else { 0 });
        comp.extend_from_slice(&(block as u16).to_le_bytes());
        comp.extend_from_slice(&(!(block as u16)).to_le_bytes());
        comp.extend_from_slice(&raw[i..i + block]);
        i += block;
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    comp.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &comp);
    chunk(&mut png, b"IEND", &[]);
    let mut f = std::fs::File::create(path).expect("create png");
    f.write_all(&png).expect("write png");
}

/// A pixel is "covered" (part of the rendered building surface) when its
/// luma clearly exceeds the dark sky-gradient background. The SDF is lit by
/// the sun + fills; the gradient sky is dark-ish at the horizon, so a simple
/// luma threshold separates solid surface from sky.
#[cfg(feature = "spectra-native")]
fn luma(px: &[u8]) -> f32 {
    0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32
}

/// Solidity = how SOLID (vs confetti) the rendered surface is, independent of
/// the building's (non-rectangular) silhouette. For every scanline that
/// contains surface pixels, measure the filled fraction within that row's
/// span (first→last surface pixel). A continuous wall fills its span ≈ 1.0;
/// sparse splat confetti leaves gaps inside the span and scores far lower.
/// Returns (mean_span_fill, total_surface_pixels). Rows whose span is shorter
/// than `min_span` (slivers / antialias fringe) are ignored.
#[cfg(feature = "spectra-native")]
fn surface_solidity(rgba: &[u8], w: u32, h: u32, thr: f32, min_span: u32) -> (f64, u64) {
    let mut sum_fill = 0.0f64;
    let mut rows = 0u64;
    let mut total = 0u64;
    for y in 0..h {
        let mut first = None;
        let mut last = 0u32;
        let mut count = 0u32;
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if luma(&rgba[i..i + 4]) > thr {
                first.get_or_insert(x);
                last = x;
                count += 1;
            }
        }
        total += count as u64;
        if let Some(f) = first {
            let span = last - f + 1;
            if span >= min_span {
                sum_fill += count as f64 / span as f64;
                rows += 1;
            }
        }
    }
    let mean = if rows > 0 {
        sum_fill / rows as f64
    } else {
        0.0
    };
    (mean, total)
}

/// Load one cooked atoms.json SDF into a `SdfVolumeInput`, decoding snorm16
/// distances to metres exactly as `vox_physics::sdf::from_snorm16_grid`.
#[cfg(feature = "spectra-native")]
fn load_atoms_sdf(path: &std::path::Path) -> super::SdfVolumeInput {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
    let sdf = &json["sdf"];
    let res: Vec<u64> = sdf["resolution"]
        .as_array()
        .expect("resolution")
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let origin: Vec<f32> = sdf["origin"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let voxel_size = sdf["voxel_size"].as_f64().unwrap() as f32;
    let narrow_band = sdf["narrow_band"].as_f64().unwrap() as f32;
    let snorm: Vec<i64> = sdf["distances_snorm16"]
        .as_array()
        .expect("distances_snorm16")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let distances: Vec<f32> = snorm
        .iter()
        .map(|&v| {
            let n = if v as i32 == i16::MIN as i32 {
                -1.0
            } else {
                v as f32 / i16::MAX as f32
            };
            n.clamp(-1.0, 1.0) * narrow_band
        })
        .collect();
    super::SdfVolumeInput {
        resolution: [res[0] as u32, res[1] as u32, res[2] as u32],
        origin: [origin[0], origin[1], origin[2]],
        voxel_size,
        narrow_band,
        distances,
    }
}

/// Load the cooked ATOMS (position + channel + colour) from an atoms.json
/// into `SdfSceneAtom`s in the asset's LOCAL space (same space as the SDF
/// grid). `force_glass_opaque` reclassifies glass→facade (the control render
/// that proves windows differ from walls). Returns (atoms, raw_glass_count).
#[cfg(feature = "spectra-native")]
fn load_atoms_material(
    path: &std::path::Path,
    force_glass_opaque: bool,
) -> (Vec<super::SdfSceneAtom>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
    let arr = json["atoms"].as_array().expect("atoms array");
    let mut atoms = Vec::with_capacity(arr.len());
    let mut glass = 0usize;
    for a in arr {
        let p = a["position"].as_array().expect("position");
        let c = a["color"].as_array().expect("color");
        let ch_str = a["channel"].as_str().unwrap_or("other");
        let mut channel = super::sdf_atom_channel_from_str(ch_str);
        if channel == super::SDF_ATOM_CH_GLASS {
            glass += 1;
            if force_glass_opaque {
                channel = super::SDF_ATOM_CH_FACADE;
            }
        }
        atoms.push(super::SdfSceneAtom {
            position: [
                p[0].as_f64().unwrap() as f32,
                p[1].as_f64().unwrap() as f32,
                p[2].as_f64().unwrap() as f32,
            ],
            color: [
                c[0].as_f64().unwrap() as f32,
                c[1].as_f64().unwrap() as f32,
                c[2].as_f64().unwrap() as f32,
            ],
            channel,
        });
    }
    (atoms, glass)
}

/// Wrap a single SdfVolumeInput in a 1-element slice for the scene API.
#[cfg(feature = "spectra-native")]
fn volume_slice(v: &super::SdfVolumeInput) -> Vec<super::SdfVolumeInput> {
    vec![v.clone()]
}

/// The cook's forge_box_uv constants for one cooked asset, read from the
/// payload's forge_description footprint (`width`/`depth` in metres) —
/// `offset_x = w/2`, `offset_z = d/2`, 1 texture tile per 2.5 m. These are
/// the SAME constants `game_asset_cook.rs::forge_box_uv` used to sample
/// the atom colours, so the kernel's box projection lands texel-exact on
/// the cooked appearance.
#[cfg(feature = "spectra-native")]
fn load_forge_uv_params(path: &std::path::Path) -> super::SdfUvParams {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
    let fp = &json["forge_description"]["footprint"];
    let width = fp["width"].as_f64().expect("forge footprint width") as f32;
    let depth = fp["depth"].as_f64().expect("forge footprint depth") as f32;
    super::SdfUvParams {
        offset_x: width * 0.5,
        offset_z: depth * 0.5,
        tile_recip: 1.0 / 2.5,
    }
}

/// Shared wave-1 craftsman harness: the cooked SDF + atoms, the M2 glass
/// test's placement (base on y=0, centred at the origin) and its exact
/// camera/rig, so the parity gate, the textured quality gates and the M2
/// glass test all measure the SAME frame.
#[cfg(feature = "spectra-native")]
struct CraftsmanHarness {
    asset_path: std::path::PathBuf,
    volume: super::SdfVolumeInput,
    atoms: Vec<super::SdfSceneAtom>,
    raw_glass: usize,
    instance: super::SdfSceneInstance,
    eye: [f32; 3],
    center: [f32; 3],
    fov_y: f32,
    w: u32,
    h: u32,
    rig: super::LightRig,
}

#[cfg(feature = "spectra-native")]
fn craftsman_harness() -> CraftsmanHarness {
    use super::{LightRig, SdfSceneInstance};

    let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join("Ochroma/projects/urban_horizon/assets/buildings/forge_starter/atoms");
    let asset_path = atoms_dir.join("forge.house.craftsman.atoms.json");

    let volume = load_atoms_sdf(&asset_path);
    let (atoms, raw_glass) = load_atoms_material(&asset_path, false);

    // Place the single building so its base sits on y=0, centred at origin
    // (identical to the M2 glass test).
    let ground_y = -volume.origin[1];
    let instance = SdfSceneInstance {
        volume_index: 0,
        position: [
            -(volume.origin[0] + (volume.resolution[0] - 1) as f32 * volume.voxel_size * 0.5),
            ground_y,
            -(volume.origin[2] + (volume.resolution[2] - 1) as f32 * volume.voxel_size * 0.5),
        ],
        rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
        uniform_scale: 1.0,
        albedo: [0.7, 0.7, 0.7],
        reflectance_spd: crate::spectral_response::reflectance_from_rgb([0.7, 0.7, 0.7]),
    };

    // The M2 front-facade camera + bright-sky rig.
    let rig = LightRig {
        sun_dir: [0.3, 0.6, 0.7],
        sun_intensity: 2.2,
        sky_intensity: 0.0,
        camera_fill: 0.0,
        rim_fill: 0.0,
        sky_dome_intensity: 1.0,
        sky_dome_zenith: [0.45, 0.62, 0.95],
        sky_dome_horizon: [0.80, 0.86, 0.95],
        ..Default::default()
    };

    CraftsmanHarness {
        asset_path,
        volume,
        atoms,
        raw_glass,
        instance,
        eye: [3.0, 6.0, 22.0],
        center: [0.0, 4.5, 0.0],
        fov_y: std::f32::consts::FRAC_PI_4,
        w: 320,
        h: 320,
        rig,
    }
}

// --- Test-local copy of the game's TextureCache resolver pattern --------
// (urban_horizon/src/asset/textures.rs — the engine cannot depend on the
// game crate, so the quality test replicates the exact load semantics:
// logical `polyhaven://<stem>_<kind>` URIs resolve through the pinned
// stem->set table; diffuse texels are sRGB-decoded to linear and
// mean-normalized toward the cooked base_color_factor tint; normal and
// roughness maps load raw; everything box-downsamples to <=256 (diffuse/
// rough) / <=128 (normal) in linear space.)

/// Logical URI stem -> on-disk PolyHaven set (the game's pinned table).
#[cfg(feature = "spectra-native")]
const POLYHAVEN_STEM_SETS: &[(&str, &str)] = &[
    ("clapboard", "brown_planks_05"),
    ("painted_trim_primary", "beige_wall_001"),
    ("painted_trim_shadow", "beige_wall_001"),
    ("wood_door", "brown_planks_03"),
    ("slate_roof", "roof_slates_02"),
    ("stucco", "concrete_wall_003"),
    ("brick", "brick_wall_001"),
    ("painted_siding", "blue_painted_planks"),
    ("painted_siding_worn", "distressed_painted_planks"),
    ("roof_tiles", "clay_roof_tiles"),
    ("roof_shingles", "red_slate_roof_tiles_01"),
    ("metal_roof", "corrugated_iron_02"),
    ("asphalt", "asphalt_02"),
    ("lawn", "leafy_grass"),
    ("concrete_pavers", "concrete_pavers"),
    ("brick_common", "brick_wall_001"),
    ("brick_painted", "painted_worn_brick"),
    ("plaster", "painted_plaster_wall"),
];

#[cfg(feature = "spectra-native")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TexKind {
    Diffuse,
    Normal,
    Roughness,
    /// Single-channel height/displacement map (PolyHaven `displacement.jpg`).
    /// Loaded linear, native res — drives POM in the megakernel.
    Height,
}

/// Resolve a cooked texture URI to (on-disk set, kind). Handles both the
/// logical `polyhaven://<stem>_<kind>` form and the direct relative
/// `textures/polyhaven/<set>/1k/<kind>.jpg` form the cook also emits.
#[cfg(feature = "spectra-native")]
fn resolve_cooked_texture_uri(uri: &str) -> Option<(String, TexKind)> {
    if let Some(rest) = uri.strip_prefix("polyhaven://") {
        let (stem, kind) = if let Some(s) = rest.strip_suffix("_diff") {
            (s, TexKind::Diffuse)
        } else if let Some(s) = rest.strip_suffix("_nor") {
            (s, TexKind::Normal)
        } else if let Some(s) = rest.strip_suffix("_rough") {
            (s, TexKind::Roughness)
        } else if let Some(s) = rest.strip_suffix("_disp") {
            (s, TexKind::Height)
        } else {
            return None;
        };
        let set = POLYHAVEN_STEM_SETS
            .iter()
            .find(|(s, _)| *s == stem)
            .map(|(_, set)| *set)?;
        return Some((set.to_string(), kind));
    }
    // Direct path: textures/polyhaven/<set>/1k/{diffuse,normal,roughness}.jpg
    let rest = uri.strip_prefix("textures/polyhaven/")?;
    let mut parts = rest.split('/');
    let set = parts.next()?;
    let _res = parts.next()?; // "1k"
    let kind = match parts.next()? {
        "diffuse.jpg" => TexKind::Diffuse,
        "normal.jpg" => TexKind::Normal,
        "roughness.jpg" => TexKind::Roughness,
        "displacement.jpg" => TexKind::Height,
        _ => return None,
    };
    Some((set.to_string(), kind))
}

// sRGB decode lives in vox_core::srgb_to_linear; import it so the
// spectra-native texture tests below can call it unqualified.
#[cfg(feature = "spectra-native")]
use vox_core::srgb_to_linear;

/// Integer box-filter reduction to `<= max_size` per side (linear space).
#[cfg(feature = "spectra-native")]
fn box_downsample(
    data: Vec<f32>,
    width: u32,
    height: u32,
    channels: u32,
    max_size: u32,
) -> (Vec<f32>, u32, u32) {
    let factor = (width.max(height)).div_ceil(max_size).max(1);
    if factor <= 1 {
        return (data, width, height);
    }
    let (ow, oh) = (width / factor, height / factor);
    let c = channels as usize;
    let mut out = vec![0.0f32; (ow * oh) as usize * c];
    let inv = 1.0 / (factor * factor) as f32;
    for oy in 0..oh {
        for ox in 0..ow {
            let base = ((oy * ow + ox) as usize) * c;
            for sy in 0..factor {
                for sx in 0..factor {
                    let src = (((oy * factor + sy) * width + ox * factor + sx) as usize) * c;
                    for ch in 0..c {
                        out[base + ch] += data[src + ch];
                    }
                }
            }
            for ch in 0..c {
                out[base + ch] *= inv;
            }
        }
    }
    (out, ow, oh)
}

/// Build a RELAXED cone-step map from a 1-channel height field (height in
/// `[0,1]`, 1 = surface plane, 0 = deepest recess), returning one cone ratio
/// per texel in `[0,1]`.
///
/// For each texel `t` the cone ratio is the steepest empty cone rising from
/// `t` toward the surface that does NOT pierce any higher neighbour:
///   cone[t] = min over all other texels s with height[s] > height[t] of
///             horizontal_dist(s,t) / (height[s] - height[t])
/// i.e. `horizontal / vertical` — the cone's slope. A flat-above region gives
/// a large ratio (clamped to 1 = open cone, big march step); a tall wall just
/// above gives a small ratio (tight cone, small step). The march advances by
/// the largest distance that keeps the ray outside this cone, so it never
/// overshoots the surface (Policarpo cone step / Dummer relaxed cone step).
///
/// RELAXED: we keep the standard min-ratio (allow the cone to just touch the
/// surface above) rather than a stricter safety margin — fewer march steps;
/// the kernel's `u_cone_relax_bias` and the one-lerp contact refinement absorb
/// any over-step from EWA blending.
///
/// This is REAL computation over the decoded displacement texels, not a
/// placeholder. To stay tractable at 1k it uses a BOUNDED-RADIUS sweep: a
/// higher texel `s` constrains `t` only within a horizontal window whose
/// radius is set by the max relief depth (`max_depth`, the render-time
/// displacement_scale upper bound) — beyond that radius a cone of ratio <= 1
/// would have to rise more than the entire relief band to be pierced, so it
/// never constrains. Cost is O(n * window^2), not O(n^2).
///
/// Horizontal distance is measured in UV-normalized units (texel index /
/// dimension) to match the kernel, which marches the UV offset; vertical drop
/// is in height units (`[0,1]`). The ratio is clamped to `[0,1]`: 1 means the
/// surface above is far/flat enough that the cone is effectively open.
#[cfg(feature = "spectra-native")]
fn build_relaxed_cone_map(height: &[f32], w: u32, h: u32, max_depth: f32) -> Vec<f32> {
    let wu = w as usize;
    let hu = h as usize;
    assert_eq!(height.len(), wu * hu, "height must be w*h single channel");
    // Window radius in texels. A higher neighbour s constrains t's cone only
    // if horiz/vert <= 1, i.e. horiz <= vert <= max_depth (vertical drop can
    // be at most the relief band). Convert that max horizontal UV distance to
    // texels. Guard against a degenerate tiny radius.
    let dim = w.max(h) as f32;
    let radius = ((max_depth.max(1e-3) * dim).ceil() as i32).clamp(2, dim as i32);
    let inv_dim = 1.0f32 / dim; // UV per texel (square texels assumed)

    let mut cone = vec![1.0f32; wu * hu]; // start fully open (ratio 1)
    for ty in 0..hu {
        for tx in 0..wu {
            let ti = ty * wu + tx;
            let ht = height[ti];
            let mut min_ratio = 1.0f32;
            let y0 = (ty as i32 - radius).max(0);
            let y1 = (ty as i32 + radius).min(hu as i32 - 1);
            let x0 = (tx as i32 - radius).max(0);
            let x1 = (tx as i32 + radius).min(wu as i32 - 1);
            for sy in y0..=y1 {
                for sx in x0..=x1 {
                    let si = sy as usize * wu + sx as usize;
                    let hs = height[si];
                    if hs <= ht {
                        continue; // only HIGHER neighbours constrain the cone
                    }
                    let dx = (sx - tx as i32) as f32 * inv_dim;
                    let dy = (sy - ty as i32) as f32 * inv_dim;
                    let horiz = (dx * dx + dy * dy).sqrt();
                    let vert = hs - ht;
                    if vert <= 1e-6 {
                        continue;
                    }
                    let ratio = horiz / vert;
                    if ratio < min_ratio {
                        min_ratio = ratio;
                    }
                }
            }
            cone[ti] = min_ratio.clamp(0.0, 1.0);
        }
    }
    cone
}

/// Load one PolyHaven map exactly like the game's TextureCache: diffuse is
/// sRGB->linear, mean-normalized toward `tint` (`clamp(tint/mean, 0.25,
/// 4.0)` per channel, the ALL-P1-1 no-double-darkening rule), <=256;
/// normal raw 3ch <=128; roughness raw 1ch <=256.
#[cfg(feature = "spectra-native")]
fn load_polyhaven_map(
    polyhaven_root: &std::path::Path,
    set: &str,
    kind: TexKind,
    tint: [f32; 3],
) -> super::TextureImage {
    let file = match kind {
        TexKind::Diffuse => "diffuse.jpg",
        TexKind::Normal => "normal.jpg",
        TexKind::Roughness => "roughness.jpg",
        TexKind::Height => "displacement.jpg",
    };
    let path = polyhaven_root.join(set).join("1k").join(file);
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
        .to_rgb8();
    let (width, height) = img.dimensions();
    let (channels, data, max_size): (u32, Vec<f32>, u32) = match kind {
        TexKind::Diffuse => {
            let linear: Vec<f32> = img
                .pixels()
                .flat_map(|p| [0, 1, 2].map(|c| srgb_to_linear(p.0[c] as f32 / 255.0)))
                .collect();
            // The PolyHaven diffuse IS the albedo. The previous "normalize the
            // texture mean toward the per-instance tint" step DESATURATED a
            // saturated texture: e.g. a red brick (low G/B) tinted toward a
            // less-saturated factor boosted G/B per channel and pulled the
            // whole wall to grey (measured [164,161,158] on the sunlit face).
            // Fix: keep the texture's OWN linear color and contrast verbatim;
            // apply only a HUE-PRESERVING brightness nudge toward the tint's
            // luma (never a per-channel re-balance), so the brick stays the
            // brick's red instead of collapsing to grey.
            let mut mean = [0.0f64; 3];
            for texel in linear.chunks_exact(3) {
                for (m, &v) in mean.iter_mut().zip(texel) {
                    *m += v as f64;
                }
            }
            let n = (linear.len() / 3).max(1) as f64;
            let tex_luma =
                (0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]) as f32 / n as f32;
            let tint_luma = 0.2126 * tint[0] + 0.7152 * tint[1] + 0.0722 * tint[2];
            // Single scalar gain (clamped, gentle) — same on all channels, so
            // chroma is untouched. Nudge ~50% of the way toward the tint luma.
            let gain = {
                let g = (tint_luma / tex_luma.max(1e-4)).clamp(0.5, 2.0);
                1.0 + (g - 1.0) * 0.5
            };
            // Modest chroma lift: PolyHaven weathered brick is a muted
            // brown ([0.14,0.077,0.051] linear); under a bright neutral sun
            // it reads as warm-grey. Push saturation ~1.35x about each
            // texel's own luma (hue + brightness preserved) so the brick
            // reads as brick, not tan stucco — without re-tinting it.
            const SAT: f32 = 1.35;
            let data = linear
                .chunks_exact(3)
                .flat_map(|t| {
                    let l = 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2];
                    [0, 1, 2].map(|c| {
                        let v = (l + (t[c] - l) * SAT) * gain;
                        v.clamp(0.0, 1.0)
                    })
                })
                .collect();
            (
                3, data,
                // Keep native 1k diffuse (was 256 -> mushy brick over a facade).
                1024,
            )
        }
        TexKind::Normal => (
            3,
            img.pixels()
                .flat_map(|p| [0, 1, 2].map(|c| p.0[c] as f32 / 255.0))
                .collect(),
            128,
        ),
        TexKind::Roughness => (
            1,
            img.pixels().map(|p| p.0[0] as f32 / 255.0).collect(),
            256,
        ),
        // Height/displacement: single linear channel, kept at native 1k so
        // the relief march reads crisp mortar gaps (downsampling smears them).
        // PolyHaven stores it grayscale; take the red channel as height.
        //
        // Cone-step needs BOTH height AND a per-texel cone ratio at the same
        // uv every march iteration. We pack the cone ratio into the G channel
        // of THIS height texture (R = height, G = relaxed cone ratio) so a
        // single EWA tap returns both — ZERO new bindings / material slots.
        // The cone map is REALLY computed from the decoded displacement texels
        // below (build_relaxed_cone_map), not a placeholder. See the early
        // return: the Height arm returns a 2-channel TextureImage and does NOT
        // fall through to the generic 1ch path.
        TexKind::Height => {
            let height_1ch: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
            // Downsample the height FIRST (to the final 1k resolution) so the
            // O(n*window) cone sweep runs over the smaller, final texel grid —
            // keeps the precompute tractable and the cone map aligned to the
            // exact texels the kernel samples.
            let (h_data, w, h) = box_downsample(height_1ch, width, height, 1, 1024);
            // displacement_scale used at render time is ~0.02-0.08 (UV-height
            // units). The cone only needs to be conservative within the relief
            // band; bound the sweep window so we never do full O(n^2). 0.08 of
            // the texture span is a safe upper bound on how far a cone can run
            // before the surface above it stops constraining it.
            let cone = build_relaxed_cone_map(&h_data, w, h, 0.08);
            debug_assert_eq!(cone.len(), h_data.len());
            // Interleave R = height, G = cone into a 2-channel buffer.
            let mut data2 = vec![0.0f32; h_data.len() * 2];
            for i in 0..h_data.len() {
                data2[i * 2] = h_data[i];
                data2[i * 2 + 1] = cone[i];
            }
            return super::TextureImage {
                width: w,
                height: h,
                channels: 2,
                data: data2,
                native: None,
            };
        }
    };
    let (data, width, height) = box_downsample(data, width, height, channels, max_size);
    super::TextureImage {
        width,
        height,
        channels,
        data,
        native: None,
    }
}

/// Load a PolyHaven displacement/height map for POM, tolerant of where it
/// lives. The per-project cooked packs shipped only diffuse/normal/roughness
/// (no `displacement.jpg`), but the global PolyHaven cache
/// (`~/.cache/aetherspectra/polyhaven/<set>/1k/displacement.jpg`) carries the
/// height map. Try the pack root first, then the cache; return `None` (POM
/// off for that material) if neither exists — never panic, so a pack without
/// height data still renders flat instead of failing the gate.
#[cfg(feature = "spectra-native")]
fn try_load_height_map(polyhaven_root: &std::path::Path, set: &str) -> Option<super::TextureImage> {
    // Candidate roots in priority order: the pack's own polyhaven dir, then
    // the shared aetherspectra cache.
    let cache_root = dirs_next_cache().map(|c| c.join("aetherspectra/polyhaven"));
    let roots: Vec<std::path::PathBuf> = std::iter::once(polyhaven_root.to_path_buf())
        .chain(cache_root)
        .collect();
    for root in roots {
        let path = root.join(set).join("1k").join("displacement.jpg");
        if path.exists() {
            return Some(load_polyhaven_map(
                &root,
                set,
                TexKind::Height,
                [1.0, 1.0, 1.0],
            ));
        }
    }
    None
}

/// Resolve the OS cache dir (`$XDG_CACHE_HOME` or `~/.cache`) without pulling
/// in an extra crate — just enough for the PolyHaven height fallback.
#[cfg(feature = "spectra-native")]
fn dirs_next_cache() -> Option<std::path::PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return Some(std::path::PathBuf::from(xdg));
        }
    }
    std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".cache"))
}

/// Parse the cooked payload's per-channel `ReadyAssetPbrMaterial` list
/// (serde, the SAME atoms.json the M2 test reads) and build the engine
/// inputs: `PbrMaterial`s + deduped `TextureImage`s through the resolver
/// above, plus the per-instance channel->material table. Glass keeps -1 so
/// windows stay on the M2 glass BSDF route.
#[cfg(feature = "spectra-native")]
#[allow(clippy::type_complexity)]
fn load_craftsman_pbr(
    asset_path: &std::path::Path,
) -> (
    Vec<super::PbrMaterial>,
    Vec<super::TextureImage>,
    super::SdfChannelMaterials,
) {
    #[derive(serde::Deserialize, Default)]
    struct CookedTextureSet {
        #[serde(default)]
        base_color: Option<String>,
        #[serde(default)]
        normal: Option<String>,
        #[serde(default)]
        roughness: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct CookedPbrMaterial {
        id: String,
        channel: String,
        base_color_factor: [f32; 4],
        metallic_factor: f32,
        roughness_factor: f32,
        #[serde(default)]
        textures: CookedTextureSet,
    }

    let bytes =
        std::fs::read(asset_path).unwrap_or_else(|e| panic!("read {}: {e}", asset_path.display()));
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
    let cooked: Vec<CookedPbrMaterial> = serde_json::from_value(json["materials"].clone())
        .expect("parse cooked ReadyAssetPbrMaterial list");
    assert!(
        !cooked.is_empty(),
        "cooked payload carries no per-channel PBR materials"
    );

    // textures/polyhaven root sits beside the atoms dir in the pack.
    let polyhaven_root = asset_path
        .parent()
        .and_then(|p| p.parent())
        .expect("atoms dir parent")
        .join("textures/polyhaven");

    let mut materials: Vec<super::PbrMaterial> = Vec::new();
    let mut textures: Vec<super::TextureImage> = Vec::new();
    // Dedupe by (set, kind, quantized tint) like the game's cache.
    let mut tex_index: std::collections::HashMap<(String, TexKind, [u32; 3]), i32> =
        std::collections::HashMap::new();
    let mut table = super::SdfChannelMaterials {
        material_for_channel: [-1i32; 9],
    };

    for cm in &cooked {
        let slot = super::sdf_atom_channel_from_str(&cm.channel) as usize;
        if slot == super::SDF_ATOM_CH_GLASS as usize {
            continue; // glass stays -1: the M2 BSDF route, never textured
        }
        let tint = [
            cm.base_color_factor[0],
            cm.base_color_factor[1],
            cm.base_color_factor[2],
        ];
        let mut resolve = |uri: &Option<String>, expect: TexKind| -> i32 {
            let Some(uri) = uri.as_deref() else { return -1 };
            let Some((set, kind)) = resolve_cooked_texture_uri(uri) else {
                eprintln!("[sdf_textured] unmapped texture uri '{uri}' (flat colour)");
                return -1;
            };
            assert_eq!(kind, expect, "uri '{uri}' resolved to the wrong map kind");
            // Tint only keys diffuse entries (normal/rough are tint-free).
            let tkey = match kind {
                TexKind::Diffuse => [
                    (tint[0].max(0.0) * 1024.0).round() as u32,
                    (tint[1].max(0.0) * 1024.0).round() as u32,
                    (tint[2].max(0.0) * 1024.0).round() as u32,
                ],
                _ => [0; 3],
            };
            let key = (set.clone(), kind, tkey);
            if let Some(&idx) = tex_index.get(&key) {
                return idx;
            }
            let img = load_polyhaven_map(&polyhaven_root, &set, kind, tint);
            let idx = textures.len() as i32;
            textures.push(img);
            tex_index.insert(key, idx);
            idx
        };
        let albedo_tex = resolve(&cm.textures.base_color, TexKind::Diffuse);
        let roughness_tex = resolve(&cm.textures.roughness, TexKind::Roughness);
        let normal_tex = resolve(&cm.textures.normal, TexKind::Normal);

        // POM height map: the cooked CookedTextureSet doesn't carry a
        // displacement URI, but every PolyHaven set on disk ships a
        // `displacement.jpg` beside diffuse/normal/roughness. Derive the set
        // from any present map URI and load its height map as a TexKind::Height
        // texture (linear, native 1k). Deduped on (set, Height) — height is
        // tint-free so the tint key is [0;3].
        let displacement_tex = {
            let set = cm
                .textures
                .normal
                .as_deref()
                .or(cm.textures.base_color.as_deref())
                .or(cm.textures.roughness.as_deref())
                .and_then(resolve_cooked_texture_uri)
                .map(|(set, _)| set);
            match set {
                Some(set) => {
                    let key = (set.clone(), TexKind::Height, [0u32; 3]);
                    if let Some(&idx) = tex_index.get(&key) {
                        idx
                    } else if let Some(img) = try_load_height_map(&polyhaven_root, &set) {
                        let idx = textures.len() as i32;
                        textures.push(img);
                        tex_index.insert(key, idx);
                        idx
                    } else {
                        // No height map shipped for this set: POM stays off
                        // (flat normal-map shading), never a hard failure.
                        -1
                    }
                }
                None => -1,
            }
        };

        let mat_id = materials.len() as i32;
        materials.push(super::PbrMaterial {
            base_color: tint,
            roughness: cm.roughness_factor,
            metallic: cm.metallic_factor,
            emission_strength: 0.0,
            albedo_tex,
            modulate_base_color_texture: false,
            opacity_tex: -1,
            vegetation_bsdf: false,
            roughness_tex,
            modulate_roughness_texture: false,
            normal_tex,
            displacement_tex,
            // ~3 cm relief reads as real brick/mortar depth at street
            // distance without over-marching; midlevel 0.5 = surface plane.
            displacement_scale: if displacement_tex >= 0 { 0.03 } else { 0.0 },
            displacement_midlevel: 0.5,
            uv_scale: [1.0, 1.0],
            // SDF path: glass stays opaque here — its windows route
            // through the dedicated SDF glass branch in the megakernel,
            // not through MAT_GLASS mesh materials.
            ..super::PbrMaterial::default()
        });
        if table.material_for_channel[slot] < 0 {
            table.material_for_channel[slot] = mat_id;
        }
        eprintln!(
            "[sdf_textured] cooked material '{}' channel '{}' -> slot {slot} mat {mat_id} \
                 (albedo_tex={albedo_tex} rough_tex={roughness_tex} normal_tex={normal_tex} \
                 disp_tex={displacement_tex} disp_scale={} roughness={})",
            cm.id,
            cm.channel,
            if displacement_tex >= 0 { 0.03 } else { 0.0 },
            cm.roughness_factor
        );
    }
    (materials, textures, table)
}

// ===== Mesh M0: the cooked craftsman as a REAL textured, lit triangle =====
// ===== mesh through the proven single-mesh path tracer ====================

/// The cooked craftsman's triangle mesh exactly as the pack carries it
/// (`ReadyAssetPayload.mesh`): per-vertex positions/normals/uvs (the
/// cook's box-projected world-scale UVs, multiply convention already
/// applied — used as-is with `uv_scale = [1,1]`, textures wrap) and
/// PER-TRIANGLE material ids indexing the payload's cooked `materials`
/// list in cooked order.
#[cfg(feature = "spectra-native")]
struct CookedMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<[u32; 3]>,
    material_ids: Vec<u8>,
}

/// Load `ReadyAssetPayload.mesh` from a cooked atoms.json — the SAME pack
/// file the SDF craftsman tests read; the triangle mesh lives beside the
/// SDF/atoms in that payload. Validates shape invariants (parallel vertex
/// arrays, per-triangle material ids, in-range indices) so a drifted cook
/// fails loudly here, not as GPU garbage.
#[cfg(feature = "spectra-native")]
fn load_craftsman_mesh(asset_path: &std::path::Path) -> CookedMesh {
    let bytes =
        std::fs::read(asset_path).unwrap_or_else(|e| panic!("read {}: {e}", asset_path.display()));
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
    let mesh = &json["mesh"];
    assert!(
        mesh.is_object(),
        "cooked payload {} carries no `mesh` object",
        asset_path.display()
    );
    let f32s_n = |key: &str, n: usize| -> Vec<Vec<f32>> {
        mesh[key]
            .as_array()
            .unwrap_or_else(|| panic!("mesh.{key} missing"))
            .iter()
            .map(|v| {
                let a = v.as_array().unwrap_or_else(|| panic!("mesh.{key} row"));
                assert_eq!(a.len(), n, "mesh.{key} row arity");
                a.iter().map(|x| x.as_f64().unwrap() as f32).collect()
            })
            .collect()
    };
    let positions: Vec<[f32; 3]> = f32s_n("positions", 3)
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let normals: Vec<[f32; 3]> = f32s_n("normals", 3)
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let uvs: Vec<[f32; 2]> = f32s_n("uvs", 2).into_iter().map(|v| [v[0], v[1]]).collect();
    let indices: Vec<[u32; 3]> = mesh["indices"]
        .as_array()
        .expect("mesh.indices")
        .iter()
        .map(|v| {
            let a = v.as_array().expect("triangle");
            assert_eq!(a.len(), 3, "triangle arity");
            [
                a[0].as_u64().unwrap() as u32,
                a[1].as_u64().unwrap() as u32,
                a[2].as_u64().unwrap() as u32,
            ]
        })
        .collect();
    let material_ids: Vec<u8> = mesh["material_ids"]
        .as_array()
        .expect("mesh.material_ids")
        .iter()
        .map(|v| {
            let m = v.as_u64().expect("material id");
            assert!(m < 256, "material id {m} exceeds u8");
            m as u8
        })
        .collect();
    assert_eq!(positions.len(), normals.len(), "positions/normals parallel");
    assert_eq!(positions.len(), uvs.len(), "positions/uvs parallel");
    assert_eq!(
        indices.len(),
        material_ids.len(),
        "material_ids must be PER-TRIANGLE"
    );
    assert!(
        indices
            .iter()
            .all(|t| t.iter().all(|&i| (i as usize) < positions.len())),
        "triangle index out of range"
    );
    CookedMesh {
        positions,
        normals,
        uvs,
        indices,
        material_ids,
    }
}

/// Shared cooked-material loader: parse the payload's
/// `ReadyAssetPbrMaterial` list IN COOKED ORDER, loading every referenced
/// PolyHaven map through the game's TextureCache pattern above (diffuse
/// sRGB→linear + tint-normalized, normal/roughness raw, box-downsampled,
/// deduped). HONORS the cooked `transmission`/`ior`/`thin_walled` fields
/// (the cook sets them on curtain-wall vision glass; absent fields default
/// to the historical opaque values). Returns
/// (materials, textures, per-material channel names).
#[cfg(feature = "spectra-native")]
fn load_cooked_pbr_materials(
    asset_path: &std::path::Path,
) -> (
    Vec<super::PbrMaterial>,
    Vec<super::TextureImage>,
    Vec<String>,
) {
    #[derive(serde::Deserialize, Default)]
    struct CookedTextureSet {
        #[serde(default)]
        base_color: Option<String>,
        #[serde(default)]
        normal: Option<String>,
        #[serde(default)]
        roughness: Option<String>,
    }
    fn default_cooked_ior() -> f32 {
        1.5
    }
    #[derive(serde::Deserialize)]
    struct CookedPbrMaterial {
        id: String,
        channel: String,
        base_color_factor: [f32; 4],
        metallic_factor: f32,
        roughness_factor: f32,
        #[serde(default)]
        textures: CookedTextureSet,
        /// > 0 = the cook tagged this material REAL transmissive glass.
        #[serde(default)]
        transmission: f32,
        #[serde(default = "default_cooked_ior")]
        ior: f32,
        #[serde(default)]
        thin_walled: bool,
    }

    let bytes =
        std::fs::read(asset_path).unwrap_or_else(|e| panic!("read {}: {e}", asset_path.display()));
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse atoms.json");
    let cooked: Vec<CookedPbrMaterial> = serde_json::from_value(json["materials"].clone())
        .expect("parse cooked ReadyAssetPbrMaterial list");
    assert!(
        !cooked.is_empty(),
        "cooked payload carries no per-channel PBR materials"
    );

    // textures/polyhaven root sits beside the atoms dir in the pack.
    let polyhaven_root = asset_path
        .parent()
        .and_then(|p| p.parent())
        .expect("atoms dir parent")
        .join("textures/polyhaven");

    let mut materials: Vec<super::PbrMaterial> = Vec::new();
    let mut textures: Vec<super::TextureImage> = Vec::new();
    let mut channels: Vec<String> = Vec::new();
    // Dedupe by (set, kind, quantized tint) like the game's cache.
    let mut tex_index: std::collections::HashMap<(String, TexKind, [u32; 3]), i32> =
        std::collections::HashMap::new();

    for (i, cm) in cooked.iter().enumerate() {
        let tint = [
            cm.base_color_factor[0],
            cm.base_color_factor[1],
            cm.base_color_factor[2],
        ];
        let mut resolve = |uri: &Option<String>, expect: TexKind| -> i32 {
            let Some(uri) = uri.as_deref() else { return -1 };
            let Some((set, kind)) = resolve_cooked_texture_uri(uri) else {
                eprintln!("[mesh_m0] unmapped texture uri '{uri}' (flat colour)");
                return -1;
            };
            assert_eq!(kind, expect, "uri '{uri}' resolved to the wrong map kind");
            let tkey = match kind {
                TexKind::Diffuse => [
                    (tint[0].max(0.0) * 1024.0).round() as u32,
                    (tint[1].max(0.0) * 1024.0).round() as u32,
                    (tint[2].max(0.0) * 1024.0).round() as u32,
                ],
                _ => [0; 3],
            };
            let key = (set.clone(), kind, tkey);
            if let Some(&idx) = tex_index.get(&key) {
                return idx;
            }
            let img = load_polyhaven_map(&polyhaven_root, &set, kind, tint);
            let idx = textures.len() as i32;
            textures.push(img);
            tex_index.insert(key, idx);
            idx
        };
        let albedo_tex = resolve(&cm.textures.base_color, TexKind::Diffuse);
        let roughness_tex = resolve(&cm.textures.roughness, TexKind::Roughness);
        let normal_tex = resolve(&cm.textures.normal, TexKind::Normal);

        // POM height map for opaque facades (glass keeps -1 — windows get no
        // relief). Derive the PolyHaven set from any present map URI; load
        // its displacement.jpg as a linear single-channel height texture.
        let displacement_tex = if cm.transmission > 0.0 {
            -1
        } else {
            let set = cm
                .textures
                .normal
                .as_deref()
                .or(cm.textures.base_color.as_deref())
                .or(cm.textures.roughness.as_deref())
                .and_then(resolve_cooked_texture_uri)
                .map(|(set, _)| set);
            match set {
                Some(set) => {
                    let key = (set.clone(), TexKind::Height, [0u32; 3]);
                    if let Some(&idx) = tex_index.get(&key) {
                        idx
                    } else if let Some(img) = try_load_height_map(&polyhaven_root, &set) {
                        let idx = textures.len() as i32;
                        textures.push(img);
                        tex_index.insert(key, idx);
                        idx
                    } else {
                        // No height map shipped for this set: POM stays off
                        // (flat normal-map shading), never a hard failure.
                        -1
                    }
                }
                None => -1,
            }
        };

        materials.push(super::PbrMaterial {
            base_color: tint,
            roughness: cm.roughness_factor,
            metallic: cm.metallic_factor,
            emission_strength: 0.0,
            albedo_tex,
            modulate_base_color_texture: false,
            opacity_tex: -1,
            vegetation_bsdf: false,
            roughness_tex,
            modulate_roughness_texture: false,
            normal_tex,
            displacement_tex,
            // ~3 cm relief; midlevel 0.5 = surface plane (POM gate is the id).
            displacement_scale: if displacement_tex >= 0 { 0.03 } else { 0.0 },
            displacement_midlevel: 0.5,
            uv_scale: [1.0, 1.0],
            // The cooked glass tag travels with the material: curtain-wall
            // vision glass arrives transmissive FROM THE COOK, everything
            // else keeps the historical opaque defaults.
            transmission: cm.transmission,
            ior: cm.ior,
            thin_walled: cm.thin_walled,
            absorption_color: [0.0, 0.0, 0.0],
            absorption_depth: 1.0,
            is_water: false,
        });
        channels.push(cm.channel.clone());
        eprintln!(
            "[mesh_m0] cooked material {i} '{}' channel '{}' \
                 (albedo_tex={albedo_tex} rough_tex={roughness_tex} normal_tex={normal_tex} \
                 disp_tex={displacement_tex} disp_scale={} roughness={} metallic={} transmission={})",
            cm.id,
            cm.channel,
            if displacement_tex >= 0 { 0.03 } else { 0.0 },
            cm.roughness_factor,
            cm.metallic_factor,
            cm.transmission
        );
    }
    (materials, textures, channels)
}

/// GENERALIZED cooked-building material loader: a material table indexed
/// by the FORGE MESH MATERIAL IDS the payload's `mesh.material_ids`
/// actually carry (forge `facade/extrude.rs`: 0 wall, 1 roof, 2 glass,
/// 3 reveal, 4 trim, 5 cornice, 6 door), each id bound to the cooked
/// material of its CHANNEL — the same forge-id → channel collapse the
/// cook's `forge_material_channel` uses for atoms. Cooked
/// `transmission`/`ior`/`thin_walled` tags are KEPT, so a curtain-wall
/// payload's vision glass (forge id 2 → the cooked "glass" channel
/// material) renders transmissive exactly as cooked. Returns
/// (materials[forge_id], textures, channel name per forge_id).
#[cfg(feature = "spectra-native")]
fn load_building_mesh_pbr_by_forge_id(
    asset_path: &std::path::Path,
) -> (
    Vec<super::PbrMaterial>,
    Vec<super::TextureImage>,
    Vec<String>,
) {
    let (materials, textures, channels) = load_cooked_pbr_materials(asset_path);
    let by_channel = |name: &str| -> super::PbrMaterial {
        let idx = channels
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("cooked payload has no '{name}' channel material"));
        materials[idx]
    };
    // Forge mesh material id -> cooked channel (the cook's
    // forge_material_channel mapping: reveal/trim/cornice all collapse
    // onto the trim channel).
    let forge_channels = [
        "facade",
        "roof",
        "glass",
        "trim",
        "trim",
        "trim",
        "door",
        "glass_lit",
    ];
    let table: Vec<super::PbrMaterial> = forge_channels
        .iter()
        .map(|ch| {
            if *ch == "glass_lit" {
                // Forge id 7 (MAT_GLASS_LIT): the cooked payload carries
                // one glass material; lit panes are the glass clone with
                // a warm interior glow so a deterministic subset of
                // windows reads inhabited instead of dead.
                let mut lit = by_channel("glass");
                lit.base_color = [1.0, 0.82, 0.58];
                lit.emission_strength = 1.6;
                lit.transmission = 0.0;
                lit
            } else if *ch == "glass" {
                // Make the window channel REAL reflective glass. The cook
                // ships this factory's glass as transmission=0 (opaque),
                // which renders dead-flat panes. Promote it to a thin-walled
                // Fresnel dielectric (MAT_GLASS) so panes reflect the sky at
                // grazing angles and transmit toward the interior — the
                // physically-correct architectural-glass look. Thin-walled
                // (single pane, no refraction bend) + low roughness for a
                // crisp sky reflection; faint cool tint via base_color.
                let mut g = by_channel("glass");
                g.transmission = 1.0;
                g.ior = 1.5;
                g.thin_walled = true;
                g.roughness = 0.04;
                g.base_color = [0.78, 0.86, 0.92];
                g
            } else {
                by_channel(ch)
            }
        })
        .collect();
    (
        table,
        textures,
        forge_channels.iter().map(|ch| ch.to_string()).collect(),
    )
}

/// CPU-side depth-buffered material-id rasterization of the cooked mesh
/// with the EXACT pinhole the Spectra camera kernel uses
/// (`camera.slang`: `ndc_x = (2(x+0.5)/w − 1)·aspect`,
/// `ndc_y = 1 − 2(y+0.5)/h`, `ray = fwd + ndc·tanHalf·basis`), so the
/// per-material pixel masks line up with the GPU render — the gates then
/// measure exactly the wall/roof/trim pixels, no hand-tuned crops. The
/// test cross-checks this projection against the GPU image (coverage
/// agreement) before any gate uses it. Returns per pixel: (cooked
/// material id, triangle index) of the nearest triangle, or (−1, −1)
/// where nothing covers the pixel. The triangle index lets the
/// detail-energy gate restrict itself to SAME-FACE pixel pairs — the
/// forge cook models each clapboard course as real geometry, so
/// face-boundary shading steps would otherwise dominate the flat
/// control's gradient energy and hide the texture signal the gate
/// measures. Depth is perspective-correct (interpolated 1/z).
#[cfg(feature = "spectra-native")]
fn rasterize_material_masks(
    mesh: &CookedMesh,
    eye: [f32; 3],
    target: [f32; 3],
    fov_y: f32,
    w: u32,
    h: u32,
) -> (Vec<i32>, Vec<i32>) {
    use glam::Vec3;
    let eye_v = Vec3::from(eye);
    let fwd = (Vec3::from(target) - eye_v).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let tan_half = (fov_y * 0.5).tan();
    let aspect = w as f32 / h as f32;
    // World point -> (pixel x, pixel y, camera-space depth). Integer pixel
    // coordinates are pixel CENTERS (the -0.5 below mirrors the kernel's
    // gx + 0.5 sampling).
    let project = |p: Vec3| -> Option<(f32, f32, f32)> {
        let d = p - eye_v;
        let cz = d.dot(fwd);
        if cz <= 1e-3 {
            return None;
        }
        let ndc_x = d.dot(right) / (cz * tan_half);
        let ndc_y = d.dot(up) / (cz * tan_half);
        let sx = (ndc_x / aspect + 1.0) * 0.5 * w as f32 - 0.5;
        let sy = (1.0 - ndc_y) * 0.5 * h as f32 - 0.5;
        Some((sx, sy, cz))
    };
    let edge = |a: (f32, f32), b: (f32, f32), px: f32, py: f32| -> f32 {
        (b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)
    };
    let mut mat = vec![-1i32; (w * h) as usize];
    let mut tri_id = vec![-1i32; (w * h) as usize];
    let mut depth = vec![f32::INFINITY; (w * h) as usize];
    for (t_idx, (tri, &mid)) in mesh.indices.iter().zip(&mesh.material_ids).enumerate() {
        let p0 = Vec3::from(mesh.positions[tri[0] as usize]);
        let p1 = Vec3::from(mesh.positions[tri[1] as usize]);
        let p2 = Vec3::from(mesh.positions[tri[2] as usize]);
        let (Some(a), Some(b), Some(c)) = (project(p0), project(p1), project(p2)) else {
            continue; // behind the camera — the building never is
        };
        let area = edge((a.0, a.1), (b.0, b.1), c.0, c.1);
        if area.abs() < 1e-6 {
            continue; // degenerate in screen space
        }
        let x0 = a.0.min(b.0).min(c.0).floor().max(0.0) as u32;
        let x1 = (a.0.max(b.0).max(c.0).ceil() as i64).clamp(0, w as i64 - 1) as u32;
        let y0 = a.1.min(b.1).min(c.1).floor().max(0.0) as u32;
        let y1 = (a.1.max(b.1).max(c.1).ceil() as i64).clamp(0, h as i64 - 1) as u32;
        if x0 > x1 || y0 > y1 {
            continue;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f32, y as f32);
                let wa = edge((b.0, b.1), (c.0, c.1), px, py) / area;
                let wb = edge((c.0, c.1), (a.0, a.1), px, py) / area;
                let wc = edge((a.0, a.1), (b.0, b.1), px, py) / area;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue; // outside (normalizing by signed area makes
                    // inside-test winding-independent)
                }
                let inv_z = wa / a.2 + wb / b.2 + wc / c.2;
                if inv_z <= 0.0 {
                    continue;
                }
                let z = 1.0 / inv_z;
                let idx = (y * w + x) as usize;
                if z < depth[idx] {
                    depth[idx] = z;
                    mat[idx] = mid as i32;
                    tri_id[idx] = t_idx as i32;
                }
            }
        }
    }
    (mat, tri_id)
}

/// Connected components of one forge material id over 0.1 mm-welded
/// vertices — the real measured geometry the curtain-wall grid gates
/// count. Returns each component's AABB.
#[cfg(feature = "spectra-native")]
fn mesh_material_components(mesh: &CookedMesh, mat: u8) -> Vec<([f32; 3], [f32; 3])> {
    use std::collections::HashMap;
    let weld: Vec<u64> = {
        let mut ids: HashMap<[i64; 3], u64> = HashMap::new();
        mesh.positions
            .iter()
            .map(|p| {
                let key = [
                    (p[0] as f64 * 1.0e4).round() as i64,
                    (p[1] as f64 * 1.0e4).round() as i64,
                    (p[2] as f64 * 1.0e4).round() as i64,
                ];
                let next = ids.len() as u64;
                *ids.entry(key).or_insert(next)
            })
            .collect()
    };
    let tris: Vec<usize> = (0..mesh.indices.len())
        .filter(|&t| mesh.material_ids[t] == mat)
        .collect();
    let mut parent: Vec<usize> = (0..tris.len()).collect();
    fn find(parent: &mut Vec<usize>, mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    let mut owner: HashMap<u64, usize> = HashMap::new();
    for (ti, &t) in tris.iter().enumerate() {
        for &v in &mesh.indices[t] {
            match owner.get(&weld[v as usize]) {
                Some(&other) => {
                    let (ra, rb) = (find(&mut parent, ti), find(&mut parent, other));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
                None => {
                    owner.insert(weld[v as usize], ti);
                }
            }
        }
    }
    let mut boxes: HashMap<usize, ([f32; 3], [f32; 3])> = HashMap::new();
    for (ti, &t) in tris.iter().enumerate() {
        let root = find(&mut parent, ti);
        let entry = boxes
            .entry(root)
            .or_insert(([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]));
        for &v in &mesh.indices[t] {
            let p = mesh.positions[v as usize];
            for a in 0..3 {
                entry.0[a] = entry.0[a].min(p[a]);
                entry.1[a] = entry.1[a].max(p[a]);
            }
        }
    }
    boxes.into_values().collect()
}

/// Mean Sobel gradient-magnitude of luminance over mask pixels whose full
/// 3x3 neighbourhood is inside the mask — the facade detail-energy
/// measure (edges from mullions/transoms/spandrel-glass alternation),
/// silhouette-immune by construction.
#[cfg(feature = "spectra-native")]
fn masked_sobel_energy(rgba: &[u8], mask: &[bool], w: u32, h: u32) -> f64 {
    let luma = |i: usize| -> f64 {
        0.2126 * rgba[i * 4] as f64
            + 0.7152 * rgba[i * 4 + 1] as f64
            + 0.0722 * rgba[i * 4 + 2] as f64
    };
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for y in 1..h - 1 {
        'px: for x in 1..w - 1 {
            let i = (y * w + x) as usize;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let j = ((y as i32 + dy) * w as i32 + (x as i32 + dx)) as usize;
                    if !mask[j] {
                        continue 'px;
                    }
                }
            }
            let l =
                |dx: i32, dy: i32| luma(((y as i32 + dy) * w as i32 + (x as i32 + dx)) as usize);
            let gx = (l(1, -1) + 2.0 * l(1, 0) + l(1, 1)) - (l(-1, -1) + 2.0 * l(-1, 0) + l(-1, 1));
            let gy = (l(-1, 1) + 2.0 * l(0, 1) + l(1, 1)) - (l(-1, -1) + 2.0 * l(0, -1) + l(1, -1));
            sum += (gx * gx + gy * gy).sqrt();
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

/// Signed solid angle of triangle (p0,p1,p2) from `q` (van Oosterom–
/// Strackee) — the same oracle the forge seal tests and the cook's GWN
/// sign gate use.
#[cfg(feature = "spectra-native")]
fn tri_solid_angle(q: glam::DVec3, p0: glam::DVec3, p1: glam::DVec3, p2: glam::DVec3) -> f64 {
    let (a, b, c) = (p0 - q, p1 - q, p2 - q);
    let (la, lb, lc) = (a.length(), b.length(), c.length());
    if la < 1e-12 || lb < 1e-12 || lc < 1e-12 {
        return 0.0;
    }
    let det = a.dot(b.cross(c));
    let denom = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
    2.0 * det.atan2(denom)
}

/// Generalized winding number of a cooked mesh at `q`.
#[cfg(feature = "spectra-native")]
fn cooked_gwn(mesh: &CookedMesh, q: glam::DVec3) -> f64 {
    let mut sum = 0.0;
    for tri in &mesh.indices {
        let p0 = glam::DVec3::from(mesh.positions[tri[0] as usize].map(f64::from));
        let p1 = glam::DVec3::from(mesh.positions[tri[1] as usize].map(f64::from));
        let p2 = glam::DVec3::from(mesh.positions[tri[2] as usize].map(f64::from));
        sum += tri_solid_angle(q, p0, p1, p2);
    }
    sum / (4.0 * std::f64::consts::PI)
}

/// Replica of the cook's GWN closed gate (`SdfBaker::bake`): 4x4x4
/// stratified jittered probes (SplitMix stream 0xC001_BA5E, bit-for-bit),
/// |w| in (0.15, 0.85) anywhere => OPEN (Err carries the min fractional
/// |w|); otherwise Ok(interior probe count), which must be >= 1 for the
/// Closed sign to be provable. This is the gate that decides
/// `sign=gwn closed=yes` at recook time.
#[cfg(feature = "spectra-native")]
fn cooked_probe_gate(mesh: &CookedMesh) -> Result<usize, f32> {
    struct SplitMix(u64);
    impl SplitMix {
        fn unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32
        }
    }
    let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            mn[a] = mn[a].min(p[a]);
            mx[a] = mx[a].max(p[a]);
        }
    }
    let extent = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    let mut rng = SplitMix(0xC001_BA5E);
    let (mut interior, mut min_abs, mut fractional) = (0usize, f32::INFINITY, false);
    for cz in 0..4u32 {
        for cy in 0..4u32 {
            for cx in 0..4u32 {
                let u = (cx as f32 + rng.unit()) / 4.0;
                let v = (cy as f32 + rng.unit()) / 4.0;
                let s = (cz as f32 + rng.unit()) / 4.0;
                let q = glam::DVec3::new(
                    (mn[0] + u * extent[0]) as f64,
                    (mn[1] + v * extent[1]) as f64,
                    (mn[2] + s * extent[2]) as f64,
                );
                let w = cooked_gwn(mesh, q).abs() as f32;
                if w < 0.15 {
                    continue;
                }
                min_abs = min_abs.min(w);
                if w <= 0.85 {
                    fractional = true;
                } else {
                    interior += 1;
                }
            }
        }
    }
    if fractional {
        return Err(min_abs);
    }
    if interior == 0 {
        return Err(0.0);
    }
    Ok(interior)
}

/// Per-floor-slice skin width profile of a cooked building: max X extent
/// of vertical MAT_WALL(0) / MAT_GLASS(2) skin triangles per slice.
/// Mullions/cornices/roofs are excluded by material, horizontal sheets by
/// normal — what remains IS the massing silhouette the shape gates
/// measure. Returns (slice widths, building height).
#[cfg(feature = "spectra-native")]
fn skin_width_profile(mesh: &CookedMesh, slices: usize) -> (Vec<f32>, f32) {
    use glam::Vec3;
    let height = mesh
        .positions
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut min_x = vec![f32::INFINITY; slices];
    let mut max_x = vec![f32::NEG_INFINITY; slices];
    for (t, tri) in mesh.indices.iter().enumerate() {
        let m = mesh.material_ids[t];
        if m != 0 && m != 2 {
            continue;
        }
        let p0 = Vec3::from(mesh.positions[tri[0] as usize]);
        let p1 = Vec3::from(mesh.positions[tri[1] as usize]);
        let p2 = Vec3::from(mesh.positions[tri[2] as usize]);
        let g = (p2 - p0).cross(p1 - p0);
        if g.length() < 1e-9 || g.normalize().y.abs() > 0.5 {
            continue; // horizontal sheet (roof cap, underside, plate)
        }
        let cy = (p0.y + p1.y + p2.y) / 3.0;
        let s = ((cy / height) * slices as f32).floor() as usize;
        if s >= slices {
            continue;
        }
        for p in [p0, p1, p2] {
            min_x[s] = min_x[s].min(p.x);
            max_x[s] = max_x[s].max(p.x);
        }
    }
    let widths = (0..slices)
        .map(|s| {
            if max_x[s] > min_x[s] {
                max_x[s] - min_x[s]
            } else {
                0.0
            }
        })
        .collect();
    (widths, height)
}

/// Count massing volumes from a width profile: a new volume starts where
/// the slice width steps by more than 1 m (empty seam slices are skipped).
#[cfg(feature = "spectra-native")]
fn count_width_plateaus(widths: &[f32]) -> usize {
    let mut plateaus = 0usize;
    let mut last: Option<f32> = None;
    for &w in widths {
        if w <= 0.0 {
            continue;
        }
        if last.is_none_or(|l| (w - l).abs() > 1.0) {
            plateaus += 1;
        }
        last = Some(w);
    }
    plateaus
}

/// Normalized front-silhouette mask of a cooked building: the CPU
/// rasterizer (the GPU-cross-checked projection) with a camera fitted so
/// every building fills the frame the same way — pairwise mask deltas
/// then measure SHAPE, not size.
#[cfg(feature = "spectra-native")]
fn fitted_silhouette(mesh: &CookedMesh, res: u32) -> Vec<bool> {
    let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            mn[a] = mn[a].min(p[a]);
            mx[a] = mx[a].max(p[a]);
        }
    }
    let centre = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];
    let half = ((mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2])) * 0.5;
    let fov_y = std::f32::consts::FRAC_PI_4;
    let dist = half / (fov_y * 0.5).tan() * 1.25;
    let eye = [centre[0], centre[1], centre[2] + dist];
    let (mats_px, _) = rasterize_material_masks(mesh, eye, centre, fov_y, res, res);
    mats_px.iter().map(|&v| v >= 0).collect()
}

/// Exact mesh equality (f32 bit patterns) — the box byte-identity oracle.
#[cfg(feature = "spectra-native")]
fn pt_eq_mesh(a: &CookedMesh, b: &CookedMesh) -> bool {
    a.positions.len() == b.positions.len()
        && a.indices.len() == b.indices.len()
        && a.material_ids == b.material_ids
        && a.indices == b.indices
        && a.positions
            .iter()
            .zip(&b.positions)
            .all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
        && a.normals
            .iter()
            .zip(&b.normals)
            .all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
        && a.uvs
            .iter()
            .zip(&b.uvs)
            .all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
}

/// RGB → hue in degrees [0,360). Used to measure per-surface colour variety.
#[cfg(feature = "spectra-native")]
fn rgb_hue(r: f32, g: f32, b: f32) -> f32 {
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let d = mx - mn;
    if d <= 0.0 {
        return 0.0;
    }
    let h = if mx == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if mx == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    if h < 0.0 { h + 360.0 } else { h }
}

/// M1 ACCEPTANCE: the engine-owned Spectra path tracer natively renders a
/// CLUSTER of cooked Urban Horizon buildings as a small city block — MANY SDF
/// instances referencing a MULTI-VOLUME atlas, sphere-traced as solid
/// surfaces in megakernel.slang, occluding each other correctly, on flat
/// ground. Per-instance flat albedo so buildings are visually distinct.
/// Writes a PNG, prints instance count + coverage + seconds, and asserts
/// >= 8 instances and that each building's projected footprint is solidly
/// filled (high coverage over the footprints — not confetti). All native
/// SDF: no quad/triangle fallback for surfaces (only the off-frame sentinel
/// triangle for BVH validity).
///
/// Run alone (GPU, seconds/frame is fine):
///   SPECTRA_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
///   SPECTRA_SLANG_DIR=$HOME/src/spectra/slang SLANG_DIR=$HOME/slang-sdk \
///     scripts/build-spectra-native.sh test -p vox_render --features spectra-native \
///     --lib sdf_city_block_renders_multiple_solid_buildings -- --nocapture --test-threads=1
/// The shared M1 city-block scene: 12 instances over 6 cooked volumes on a
/// 3x4 grid, the tight city-builder oblique camera, dark-sky sun rig and
/// 320x240 framing. Single source of truth for the M1 milestone test AND
/// the perf-breakdown harness so both always measure the same workload.
#[cfg(feature = "spectra-native")]
#[allow(clippy::type_complexity)]
fn city_block_scene() -> (
    Vec<super::SdfVolumeInput>,
    Vec<super::SdfSceneInstance>,
    [f32; 3],   // eye
    [f32; 3],   // center / look target
    f32,        // fov_y
    (u32, u32), // (w, h)
    super::LightRig,
) {
    use super::{LightRig, SdfSceneInstance};

    let atoms_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join("Ochroma/projects/urban_horizon/assets/buildings/forge_starter/atoms");

    // --- Multi-volume atlas: distinct cooked Urban Horizon fields. ------------
    let asset_names = [
        "forge.house.craftsman",
        "forge.house.victorian",
        "city.res_med.l3.3x4.rowhouse_02",
        "civic.school",
        "civic.police",
        "city.res_low.l1.2x2.cottage_01",
    ];
    let volumes: Vec<super::SdfVolumeInput> = asset_names
        .iter()
        .map(|n| load_atoms_sdf(&atoms_dir.join(format!("{n}.atoms.json"))))
        .collect();
    assert!(
        volumes.iter().all(|v| v.distances.iter().any(|d| *d < 0.0)),
        "every cooked field must contain interior (negative) samples"
    );
    for (n, v) in asset_names.iter().zip(&volumes) {
        eprintln!(
            "[sdf_block] volume {n}: res={:?} voxel={:.4} band={:.4}",
            v.resolution, v.voxel_size, v.narrow_band
        );
    }

    // Per-volume horizontal footprint (metres) so we can lay instances out
    // on a non-overlapping grid (the X/Z extent of the cooked grid).
    let footprint = |v: &super::SdfVolumeInput| -> (f32, f32) {
        (
            (v.resolution[0] - 1) as f32 * v.voxel_size,
            (v.resolution[2] - 1) as f32 * v.voxel_size,
        )
    };
    // Ground placement: each cooked field's local Y origin is at v.origin[1];
    // translate so the field's base sits on y=0 (flat ground).
    let ground_y = |v: &super::SdfVolumeInput| -> f32 { -v.origin[1] };

    // Distinct per-instance albedos (warm/cool variety so buildings read as
    // separate masses in the PNG).
    let palette = [
        [0.78, 0.40, 0.34], // brick red
        [0.46, 0.58, 0.74], // slate blue
        [0.80, 0.74, 0.52], // sand
        [0.55, 0.70, 0.52], // sage
        [0.74, 0.62, 0.46], // tan
        [0.62, 0.52, 0.66], // muted violet
        [0.84, 0.78, 0.66], // cream
        [0.50, 0.64, 0.68], // teal grey
    ];

    // --- Block layout: 3 rows x 4 columns = 12 instances on a grid. ------
    let cols = 4usize;
    let rows = 3usize;
    let gap = 6.0f32; // metres between footprints
    // Column/row pitch from the widest footprint so nothing overlaps.
    let max_w = volumes
        .iter()
        .map(|v| footprint(v).0)
        .fold(0.0f32, f32::max);
    let max_d = volumes
        .iter()
        .map(|v| footprint(v).1)
        .fold(0.0f32, f32::max);
    let pitch_x = max_w + gap;
    let pitch_z = max_d + gap;

    let mut instances: Vec<SdfSceneInstance> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let vol = idx % volumes.len();
            let v = &volumes[vol];
            let (fw, fd) = footprint(v);
            // Cell centre on the grid, then offset so the field's own grid
            // origin lands such that the building is centred in its cell.
            let cell_x = (c as f32 - (cols as f32 - 1.0) * 0.5) * pitch_x;
            let cell_z = (r as f32 - (rows as f32 - 1.0) * 0.5) * pitch_z;
            let position = [
                cell_x - (v.origin[0] + fw * 0.5),
                ground_y(v),
                cell_z - (v.origin[2] + fd * 0.5),
            ];
            instances.push(SdfSceneInstance {
                volume_index: vol as u32,
                position,
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                uniform_scale: 1.0,
                albedo: palette[idx % palette.len()],
                reflectance_spd: crate::spectral_response::reflectance_from_rgb(
                    palette[idx % palette.len()],
                ),
            });
        }
    }
    let n_instances = instances.len();
    assert!(
        n_instances >= 8,
        "need >= 8 instances for the cluster proof, have {n_instances}"
    );

    // --- Camera: city-builder oblique looking down at the block. ---------
    // Scene horizontal centre is the origin (grid is symmetric about 0). The
    // block spans ~(cols-1)*pitch in X and ~(rows-1)*pitch in Z plus one
    // footprint; frame it tightly so the buildings fill the view.
    let block_w = (cols as f32 - 1.0) * pitch_x + max_w;
    let block_d = (rows as f32 - 1.0) * pitch_z + max_d;
    let center = [0.0f32, 5.0, 0.0];
    let span = block_w.max(block_d);
    // Pull in close for a filled frame; a flatter oblique (eye height ~0.5
    // span) reads as the classic city-builder 3/4 view.
    let dist = span * 0.62;
    let eye = [
        center[0] + dist * 0.55,
        center[1] + dist * 0.55,
        center[2] + dist * 0.95,
    ];
    let (w, h) = (320u32, 240u32);
    let fov_y = std::f32::consts::FRAC_PI_4;

    // Dark sky / dark fills so only the lit SDF surfaces register; the SDF's
    // view-facing fill (megakernel) lights every hit pixel above the sky.
    let rig = LightRig {
        sun_dir: [0.4, 0.75, 0.5],
        sun_intensity: 2.4,
        sky_intensity: 0.0,
        camera_fill: 0.0,
        rim_fill: 0.0,
        sky_dome_intensity: 0.0,
        sky_dome_zenith: [0.0, 0.0, 0.0],
        sky_dome_horizon: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    (volumes, instances, eye, center, fov_y, (w, h), rig)
}

/// Aggregate raw per-dispatch timings into (label, count, total_ms) rows,
/// sorted by total descending.
#[cfg(feature = "spectra-native")]
fn aggregate_kernel_ms(raw: &[(String, f32)]) -> Vec<(String, u32, f32)> {
    let mut map: std::collections::HashMap<String, (u32, f32)> = Default::default();
    for (label, ms) in raw {
        let e = map.entry(label.clone()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += *ms;
    }
    let mut rows: Vec<(String, u32, f32)> = map
        .into_iter()
        .map(|(label, (n, total))| (label, n, total))
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    rows
}
