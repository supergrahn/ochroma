//! `KHR_gaussian_splatting` glTF export/import for Gaussian splats.
//!
//! Implements the Khronos draft extension `KHR_gaussian_splatting`
//! (<https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting>).
//!
//! Per the draft, splat attributes live on a **mesh primitive whose `mode` is
//! `POINTS` (0)**. The standard `POSITION` accessor carries centroids; the
//! extension adds these primitive attribute semantics (colon-prefixed, NOT
//! underscore-prefixed):
//!
//! | semantic                                   | accessor | component |
//! |--------------------------------------------|----------|-----------|
//! | `POSITION`                                 | VEC3     | float     |
//! | `KHR_gaussian_splatting:ROTATION`          | VEC4     | float (unit quat, glTF xyzw order) |
//! | `KHR_gaussian_splatting:SCALE`             | VEC3     | float     |
//! | `KHR_gaussian_splatting:OPACITY`           | SCALAR   | float     |
//! | `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0`| VEC3     | float (SH DC color coefficient) |
//!
//! The extension is declared in the root `extensionsUsed` array and referenced
//! in the primitive's `extensions` object under the `KHR_gaussian_splatting` key.
//!
//! We emit all attributes as `float` (componentType 5126) for maximum
//! interoperability — the draft permits quantized variants but float is always
//! legal and lossless for the geometry.
//!
//! ## Color round-trip honesty
//! Ochroma carries a 16-band spectrum, not SH coefficients. On export we collapse
//! the spectrum to RGB (same band grouping as `ply_loader::write_ply`) and store
//! it as the SH **DC** coefficient `c` where display-RGB = `0.5 + C0*c`. On
//! import we invert that and re-upsample RGB→spectrum via the SAME Smits
//! `SpectralUpsampler` the PLY/SPZ loaders use. Positions, scales, rotations and
//! opacity survive exactly (float); color survives only up to the RGB bottleneck.

use std::path::Path;

use half::f16;
use vox_core::types::GaussianSplat;
use vox_data::SpectralUpsampler;

/// SH band-0 constant `1/(2*sqrt(pi))`, the DC SH→color factor.
const SH_C0: f32 = 0.282_094_8;

const EXT_NAME: &str = "KHR_gaussian_splatting";
const ATTR_ROTATION: &str = "KHR_gaussian_splatting:ROTATION";
const ATTR_SCALE: &str = "KHR_gaussian_splatting:SCALE";
const ATTR_OPACITY: &str = "KHR_gaussian_splatting:OPACITY";
const ATTR_SH0: &str = "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0";

#[derive(Debug, thiserror::Error)]
pub enum Splats2GltfError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("gltf parse error: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("input has no {EXT_NAME} extension")]
    NotASplatGltf,
    #[error("malformed {EXT_NAME} gltf: {0}")]
    Malformed(String),
}

/// Collapse a splat's 16-band spectrum to approximate linear RGB (same band
/// grouping as `ply_loader::write_ply`).
fn spectrum_to_rgb(s: &GaussianSplat) -> [f32; 3] {
    let mut rsum = 0.0f32;
    let mut gsum = 0.0f32;
    let mut bsum = 0.0f32;
    for band in 0..GaussianSplat::BANDS {
        let v = s.spectral_f32(band);
        if band < 5 {
            bsum += v;
        } else if band < 11 {
            gsum += v;
        } else {
            rsum += v;
        }
    }
    [
        (rsum / 5.0).clamp(0.0, 1.0),
        (gsum / 6.0).clamp(0.0, 1.0),
        (bsum / 5.0).clamp(0.0, 1.0),
    ]
}

/// Serialize splats to a `.gltf` (JSON + embedded base64 buffer) carrying the
/// `KHR_gaussian_splatting` extension. Returns the JSON text.
pub fn splats_to_gltf_json(splats: &[GaussianSplat]) -> String {
    let n = splats.len();

    // Build interleaved-free, per-attribute contiguous float buffers in this
    // order: POSITION(3), ROTATION(4), SCALE(3), OPACITY(1), SH0(3).
    let mut bin: Vec<u8> = Vec::new();

    let push_f32 = |bin: &mut Vec<u8>, v: f32| bin.extend_from_slice(&v.to_le_bytes());

    let pos_off = bin.len();
    let mut pmin = [f32::INFINITY; 3];
    let mut pmax = [f32::NEG_INFINITY; 3];
    for s in splats {
        let p = s.position();
        for k in 0..3 {
            pmin[k] = pmin[k].min(p[k]);
            pmax[k] = pmax[k].max(p[k]);
            push_f32(&mut bin, p[k]);
        }
    }
    let pos_len = bin.len() - pos_off;

    let rot_off = bin.len();
    for s in splats {
        let q = s.decoded_rotation().normalize();
        for v in [q.x, q.y, q.z, q.w] {
            push_f32(&mut bin, v);
        }
    }
    let rot_len = bin.len() - rot_off;

    let scale_off = bin.len();
    for s in splats {
        for v in s.scales() {
            push_f32(&mut bin, v);
        }
    }
    let scale_len = bin.len() - scale_off;

    let op_off = bin.len();
    for s in splats {
        push_f32(&mut bin, s.opacity() as f32 / 255.0);
    }
    let op_len = bin.len() - op_off;

    let sh_off = bin.len();
    for s in splats {
        let rgb = spectrum_to_rgb(s);
        // store SH DC coefficient c such that display-RGB = 0.5 + C0*c
        for c in rgb {
            push_f32(&mut bin, (c - 0.5) / SH_C0);
        }
    }
    let sh_len = bin.len() - sh_off;

    let b64 = base64_encode(&bin);
    let buffer_uri = format!("data:application/octet-stream;base64,{b64}");

    // Handle the empty case: min/max would be Inf; emit zeros.
    if n == 0 {
        pmin = [0.0; 3];
        pmax = [0.0; 3];
    }

    // accessors: 0=POSITION 1=ROTATION 2=SCALE 3=OPACITY 4=SH0
    // bufferViews mirror them 1:1.
    serde_json::json!({
        "asset": { "version": "2.0", "generator": "vox_tools::splats2gltf" },
        "extensionsUsed": [EXT_NAME],
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": {
                    "POSITION": 0,
                    ATTR_ROTATION: 1,
                    ATTR_SCALE: 2,
                    ATTR_OPACITY: 3,
                    ATTR_SH0: 4
                },
                "extensions": { EXT_NAME: {} }
            }]
        }],
        "buffers": [{ "byteLength": bin.len(), "uri": buffer_uri }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": pos_off, "byteLength": pos_len },
            { "buffer": 0, "byteOffset": rot_off, "byteLength": rot_len },
            { "buffer": 0, "byteOffset": scale_off, "byteLength": scale_len },
            { "buffer": 0, "byteOffset": op_off, "byteLength": op_len },
            { "buffer": 0, "byteOffset": sh_off, "byteLength": sh_len }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
              "min": pmin, "max": pmax },
            { "bufferView": 1, "componentType": 5126, "count": n, "type": "VEC4" },
            { "bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3" },
            { "bufferView": 3, "componentType": 5126, "count": n, "type": "SCALAR" },
            { "bufferView": 4, "componentType": 5126, "count": n, "type": "VEC3" }
        ]
    })
    .to_string()
}

/// Write a `KHR_gaussian_splatting` `.gltf` to disk. Returns splat count.
pub fn write_splats_gltf(splats: &[GaussianSplat], path: &Path) -> Result<usize, Splats2GltfError> {
    let json = splats_to_gltf_json(splats);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json)?;
    Ok(splats.len())
}

/// Detect whether a parsed glTF document declares `KHR_gaussian_splatting`.
pub fn document_has_splat_extension(document: &gltf::Document) -> bool {
    document.extensions_used().any(|e| e == EXT_NAME)
}

/// Import splats from a glTF carrying `KHR_gaussian_splatting`.
///
/// Reads the first POINTS primitive that declares the extension and pulls the
/// splat attributes straight from its accessors (no surface sampling).
///
/// The `KHR_gaussian_splatting` attribute semantics are colon-prefixed custom
/// names (e.g. `KHR_gaussian_splatting:ROTATION`). The typed `gltf` 1.4 API
/// only surfaces underscore-prefixed custom semantics (`Semantic::Extras`) and
/// gates per-primitive `extensions()` behind a feature flag, so neither the
/// extension presence nor these attributes are reachable through the typed
/// `Primitive`. Worse, a `Document` loaded with these names holds them as
/// `Checked::Invalid`, so `Document::into_json()` cannot be re-serialized.
///
/// We therefore take the document's raw glTF JSON (`raw_json`, exactly the bytes
/// that were parsed) to recover the per-primitive `attributes` map (name →
/// accessor index) and `extensions` marker, then resolve each index to a typed
/// `gltf::Accessor` via `document.accessors().nth(idx)` for the actual buffer
/// reads. The `document`/`buffers` still drive all numeric decoding.
pub fn import_splat_gltf(
    document: &gltf::Document,
    raw_json: &serde_json::Value,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<GaussianSplat>, Splats2GltfError> {
    if !document_has_splat_extension(document) {
        return Err(Splats2GltfError::NotASplatGltf);
    }

    let meshes = raw_json.get("meshes").and_then(|m| m.as_array());

    if let Some(meshes) = meshes {
        for mesh in meshes {
            let prims = match mesh.get("primitives").and_then(|p| p.as_array()) {
                Some(p) => p,
                None => continue,
            };
            for prim in prims {
                // Must be a POINTS primitive (mode 0; absent mode defaults to 4).
                let mode = prim.get("mode").and_then(|m| m.as_u64()).unwrap_or(4);
                if mode != 0 {
                    continue;
                }
                // Must declare the extension on the primitive.
                let has_ext = prim
                    .get("extensions")
                    .and_then(|e| e.get(EXT_NAME))
                    .is_some();
                if !has_ext {
                    continue;
                }
                return read_splat_primitive(document, prim, buffers);
            }
        }
    }
    Err(Splats2GltfError::Malformed(
        "no POINTS primitive declares the extension".into(),
    ))
}

/// Read the splat attributes from a single primitive, using its raw JSON
/// `attributes` map (name → accessor index) and the document's typed accessors.
fn read_splat_primitive(
    document: &gltf::Document,
    prim: &serde_json::Value,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<GaussianSplat>, Splats2GltfError> {
    let attributes = prim
        .get("attributes")
        .and_then(|a| a.as_object())
        .ok_or_else(|| Splats2GltfError::Malformed("primitive has no attributes".into()))?;

    // Resolve an attribute semantic name to a typed accessor via its index.
    let get_accessor = |name: &str| -> Option<gltf::Accessor> {
        let idx = attributes.get(name)?.as_u64()? as usize;
        document.accessors().nth(idx)
    };

    let pos_acc = get_accessor("POSITION")
        .ok_or_else(|| Splats2GltfError::Malformed("missing POSITION".into()))?;
    let rot_acc = get_accessor(ATTR_ROTATION);
    let scale_acc = get_accessor(ATTR_SCALE);
    let op_acc = get_accessor(ATTR_OPACITY);
    let sh_acc = get_accessor(ATTR_SH0);

    let positions = read_vec3(&pos_acc, buffers)?;
    let n = positions.len();
    let rotations = rot_acc.map(|a| read_vec4(&a, buffers)).transpose()?;
    let scales = scale_acc.map(|a| read_vec3(&a, buffers)).transpose()?;
    let opacities = op_acc.map(|a| read_scalar(&a, buffers)).transpose()?;
    let sh0 = sh_acc.map(|a| read_vec3(&a, buffers)).transpose()?;

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let p = positions[i];
        let s = scales.as_ref().map(|v| v[i]).unwrap_or([0.01, 0.01, 0.01]);
        let q = rotations
            .as_ref()
            .map(|v| glam::Quat::from_xyzw(v[i][0], v[i][1], v[i][2], v[i][3]).normalize())
            .unwrap_or(glam::Quat::IDENTITY);
        let opacity = opacities
            .as_ref()
            .map(|v| (v[i].clamp(0.0, 1.0) * 255.0).round() as u8)
            .unwrap_or(255);
        let rgb = sh0
            .as_ref()
            .map(|v| {
                [
                    (0.5 + SH_C0 * v[i][0]).clamp(0.0, 1.0),
                    (0.5 + SH_C0 * v[i][1]).clamp(0.0, 1.0),
                    (0.5 + SH_C0 * v[i][2]).clamp(0.0, 1.0),
                ]
            })
            .unwrap_or([0.5, 0.5, 0.5]);

        let spectral_f32 = SpectralUpsampler::from_rgb(rgb[0], rgb[1], rgb[2]);
        let spectral: [u16; 16] =
            std::array::from_fn(|b| f16::from_f32(spectral_f32[b]).to_bits());

        splats.push(GaussianSplat::volume(p, s, q, opacity, spectral));
    }
    Ok(splats)
}

// --- accessor readers (tightly-packed float buffer views) -------------------

fn accessor_slice<'a>(
    acc: &gltf::Accessor,
    buffers: &'a [gltf::buffer::Data],
) -> Result<&'a [u8], Splats2GltfError> {
    let view = acc
        .view()
        .ok_or_else(|| Splats2GltfError::Malformed("accessor has no bufferView".into()))?;
    let buf = &buffers[view.buffer().index()];
    let start = view.offset() + acc.offset();
    let len = acc.count() * acc.size();
    if start + len > buf.0.len() {
        return Err(Splats2GltfError::Malformed("accessor out of range".into()));
    }
    Ok(&buf.0[start..start + len])
}

fn read_f32_at(data: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
}

fn read_vec3(
    acc: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<[f32; 3]>, Splats2GltfError> {
    let data = accessor_slice(acc, buffers)?;
    let n = acc.count();
    Ok((0..n)
        .map(|i| {
            let b = i * 12;
            [read_f32_at(data, b), read_f32_at(data, b + 4), read_f32_at(data, b + 8)]
        })
        .collect())
}

fn read_vec4(
    acc: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<[f32; 4]>, Splats2GltfError> {
    let data = accessor_slice(acc, buffers)?;
    let n = acc.count();
    Ok((0..n)
        .map(|i| {
            let b = i * 16;
            [
                read_f32_at(data, b),
                read_f32_at(data, b + 4),
                read_f32_at(data, b + 8),
                read_f32_at(data, b + 12),
            ]
        })
        .collect())
}

fn read_scalar(
    acc: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<f32>, Splats2GltfError> {
    let data = accessor_slice(acc, buffers)?;
    let n = acc.count();
    Ok((0..n).map(|i| read_f32_at(data, i * 4)).collect())
}

// --- minimal base64 (standard alphabet, padded) -----------------------------

fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Parse a `KHR_gaussian_splatting` `.gltf`/`.glb` from raw bytes and import its
/// splats.
///
/// We deliberately use `gltf::Gltf::from_slice_without_validation` rather than
/// `gltf::import_slice`: the extension's attribute semantics are colon-prefixed
/// custom names (`KHR_gaussian_splatting:ROTATION`, …), which the gltf crate's
/// validator rejects as `<invalid semantic name>` (it only accepts standard or
/// underscore-prefixed semantics). Skipping validation lets the document load;
/// embedded data-URI buffers are then decoded via the public `import_buffers`.
pub fn import_splat_gltf_bytes(bytes: &[u8]) -> Result<Vec<GaussianSplat>, Splats2GltfError> {
    let gltf = gltf::Gltf::from_slice_without_validation(bytes)?;
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob.clone())?;
    // Recover the raw glTF JSON. For GLB the JSON lives in the first (JSON)
    // chunk; for a plain `.gltf` the whole slice is the JSON document.
    let json_bytes: &[u8] = if bytes.starts_with(b"glTF") {
        // GLB: 12-byte header, then chunk(s): [u32 len][u32 type][data].
        if bytes.len() < 20 {
            return Err(Splats2GltfError::Malformed("GLB header truncated".into()));
        }
        let chunk_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let start: usize = 20;
        let end = start
            .checked_add(chunk_len)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| Splats2GltfError::Malformed("GLB JSON chunk out of range".into()))?;
        &bytes[start..end]
    } else {
        bytes
    };
    let raw_json: serde_json::Value = serde_json::from_slice(json_bytes)?;
    import_splat_gltf(&gltf.document, &raw_json, &buffers)
}

/// CLI entry: import a `KHR_gaussian_splatting` `.gltf`/`.glb` into a `.vxm`.
pub fn gltf2splats_import(input: &Path, output: &Path) -> Result<usize, Splats2GltfError> {
    let bytes = std::fs::read(input)?;
    let splats = import_splat_gltf_bytes(&bytes)?;
    let count = splats.len();
    let vxm = vox_data::vxm::VxmFile {
        header: vox_data::vxm::VxmHeader::new(
            uuid::Uuid::new_v4(),
            count as u32,
            vox_data::vxm::MaterialType::Generic,
        ),
        splats,
    };
    let mut out = std::fs::File::create(output)?;
    vxm.write(&mut out)
        .map_err(|e| Splats2GltfError::Malformed(format!("vxm write: {e}")))?;
    Ok(count)
}

/// CLI entry: convert a `.vxm` or `.ply` into a `KHR_gaussian_splatting .gltf`.
pub fn splats2gltf(input: &Path, output: &Path) -> Result<usize, Splats2GltfError> {
    let splats = load_input_splats(input)?;
    write_splats_gltf(&splats, output)
}

/// Load splats from a `.vxm` or `.ply` for export.
fn load_input_splats(input: &Path) -> Result<Vec<GaussianSplat>, Splats2GltfError> {
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "ply" => vox_data::ply_loader::load_ply(input)
            .map_err(|e| Splats2GltfError::Malformed(format!("ply load: {e}"))),
        "vxm" => {
            let file = std::fs::File::open(input)?;
            let vxm = vox_data::vxm::VxmFile::read(file)
                .map_err(|e| Splats2GltfError::Malformed(format!("vxm load: {e}")))?;
            Ok(vxm.splats)
        }
        other => Err(Splats2GltfError::Malformed(format!(
            "unsupported input extension '{other}' (expected .vxm or .ply)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_data::SpectralUpsampler;

    fn splat_rgb(pos: [f32; 3], scale: [f32; 3], r: f32, g: f32, b: f32, op: u8) -> GaussianSplat {
        let spectral_f32 = SpectralUpsampler::from_rgb(r, g, b);
        let spectral: [u16; 16] = std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits());
        GaussianSplat::volume(pos, scale, glam::Quat::IDENTITY, op, spectral)
    }

    #[test]
    fn json_declares_extension_and_attributes() {
        let splats = vec![splat_rgb([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0.9, 0.1, 0.1, 230)];
        let json = splats_to_gltf_json(&splats);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        // extensionsUsed contains the extension
        let used = v["extensionsUsed"].as_array().unwrap();
        assert!(used.iter().any(|e| e == EXT_NAME), "extensionsUsed must list {EXT_NAME}");

        // primitive mode is POINTS (0) and declares the extension + attributes
        let prim = &v["meshes"][0]["primitives"][0];
        assert_eq!(prim["mode"], 0, "primitive mode must be POINTS (0)");
        assert!(prim["extensions"].get(EXT_NAME).is_some());
        let attrs = &prim["attributes"];
        assert!(attrs.get("POSITION").is_some());
        assert!(attrs.get(ATTR_ROTATION).is_some());
        assert!(attrs.get(ATTR_SCALE).is_some());
        assert!(attrs.get(ATTR_OPACITY).is_some());
        assert!(attrs.get(ATTR_SH0).is_some());
    }

    #[test]
    fn export_import_roundtrip_geometry_and_color() {
        let original = vec![
            splat_rgb([1.0, 2.0, 3.0], [0.10, 0.20, 0.30], 0.95, 0.05, 0.05, 230),
            splat_rgb([-4.5, 0.0, 7.25], [0.50, 0.40, 0.30], 0.10, 0.85, 0.20, 128),
            splat_rgb([10.0, -3.0, -2.0], [0.05, 0.07, 0.09], 0.10, 0.10, 0.90, 64),
        ];
        let json = splats_to_gltf_json(&original);
        // The KHR splat attribute semantics are colon-prefixed custom names that
        // gltf's validator rejects, so we import without validation (this is what
        // `import_splat_gltf_bytes` does) instead of `gltf::import_slice`.
        let gltf = gltf::Gltf::from_slice_without_validation(json.as_bytes())
            .expect("emitted gltf must parse");
        let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob.clone())
            .expect("buffers must decode");
        let raw_json: serde_json::Value =
            serde_json::from_str(&json).expect("raw json must parse");

        assert!(document_has_splat_extension(&gltf.document));
        let loaded = import_splat_gltf(&gltf.document, &raw_json, &buffers).expect("import");

        assert_eq!(loaded.len(), original.len(), "count must match");
        for (o, l) in original.iter().zip(loaded.iter()) {
            // positions are float — exact (within f32 epsilon)
            let op = o.position();
            let lp = l.position();
            for k in 0..3 {
                assert!((op[k] - lp[k]).abs() < 1e-5, "pos[{k}] {} vs {}", op[k], lp[k]);
            }
            // scales float — exact
            let os = o.scales();
            let ls = l.scales();
            for k in 0..3 {
                assert!((os[k] - ls[k]).abs() < 1e-5, "scale[{k}] {} vs {}", os[k], ls[k]);
            }
            // opacity through /255*255 round-trips exactly
            assert_eq!(o.opacity(), l.opacity(), "opacity must round-trip");
        }

        // Color survives the RGB bottleneck: the red splat stays red-dominant.
        let s = &loaded[0];
        let blue: f32 = (0..5).map(|b| s.spectral_f32(b)).sum();
        let red: f32 = (11..16).map(|b| s.spectral_f32(b)).sum();
        assert!(red > blue, "red splat stays red-dominant: red {red} vs blue {blue}");
    }

    #[test]
    fn base64_matches_known_vector() {
        // "Man" → "TWFu", "Ma" → "TWE=", "M" → "TQ=="
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"M"), "TQ==");
    }
}
