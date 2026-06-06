//! SPZ compressed Gaussian-splat format (Niantic Labs) round-trip.
//!
//! SPZ is the de-facto interchange format for 3D Gaussian splats — "about 10x
//! smaller than the PLY equivalent" (Niantic). We implement the **legacy v2
//! container** (gzip), which is the stable, fully-documented format read by all
//! SPZ tooling. Reference: <https://github.com/nianticlabs/spz>
//! (`src/cc/load-spz.cc`, `splat-types.h`).
//!
//! ## Container
//! The whole file is a single gzip stream. The decompressed payload begins with
//! a 16-byte `PackedGaussiansHeader`, then six tightly-packed attribute arrays
//! in order: positions, alphas, colors, scales, rotations, sh.
//!
//! ## Header (16 bytes, little-endian) — `PackedGaussiansHeader`
//! ```text
//! offset size field           notes
//! 0      4    magic           0x5053474e  ("NGSP", bytes N G S P in file order)
//! 4      4    version         2 (legacy gzip; quaternion-first-three rotations)
//! 8      4    numPoints       gaussian count
//! 12     1    shDegree        spherical-harmonics degree (we emit 0)
//! 13     1    fractionalBits  position fixed-point fractional bits (default 12)
//! 14     1    flags           bit0 = antialiased (we emit 0)
//! 15     1    reserved        0
//! ```
//!
//! ## Per-attribute quantization (exactly as the SPZ C++ packer/unpacker)
//! * **positions**: 3 axes × 3 bytes = signed 24-bit little-endian fixed-point.
//!   `fixed = round(coord * (1 << fractionalBits))`; unpack divides back. With
//!   the default 12 fractional bits the resolution is `1/4096 ≈ 2.44e-4` units,
//!   and 24 signed bits give a range of ±2048 units.
//! * **alphas**: 1 byte, `u8 = round(sigmoid(a) * 255)`; unpack = inverse-sigmoid.
//! * **colors**: 3 bytes, the SH **DC** coefficient `c` encoded as
//!   `u8 = round(c * 0.15 * 255 + 0.5 * 255)`; unpack `c = (u8/255 - 0.5)/0.15`.
//!   Display RGB is then `0.5 + SH_C0*c` — the SAME convention as the PLY loader,
//!   so we reuse the `SpectralUpsampler` (Smits 1999) RGB→spectral path.
//! * **scales**: 3 bytes, log-space, `u8 = round((log_scale + 10) * 16)`;
//!   unpack `log_scale = u8/16 - 10`, linear = `exp(log_scale)`.
//! * **rotations** (v2 "quaternion first three"): 3 bytes storing the xyz of the
//!   normalized quaternion (w-positive hemisphere). Pack:
//!   `u8 = round((q_i * (w<0?-1:1)) * 127.5 + 127.5)`; unpack:
//!   `q_i = u8/127.5 - 1`, `w = sqrt(max(0, 1 - x²-y²-z²))`.
//!
//! Like `write_ply`, the spectral→color step on write is lossy (16-band spectrum
//! collapsed to RGB → SH DC); positions/scales/rotations/opacity round-trip
//! within the documented quantization tolerances.

use half::f16;
use std::io::{Read, Write};
use std::path::Path;
use vox_core::types::GaussianSplat;

use crate::spectral_upsampler::SpectralUpsampler;

/// SPZ magic number: 0x5053474e, bytes `N G S P` in little-endian file order.
const SPZ_MAGIC: u32 = 0x5053_474e;
/// Legacy gzip container version with quaternion-first-three rotations.
const SPZ_VERSION_LEGACY: u32 = 2;
/// Default position fixed-point fractional bits (matches the SPZ packer default).
const DEFAULT_FRACTIONAL_BITS: u8 = 12;
/// SH band-0 constant `1 / (2*sqrt(pi))` — the DC SH→color factor.
const SH_C0: f32 = 0.282_094_8;
/// SPZ color quantization scale (Niantic's 0.15 "allow out-of-gamut base color").
const COLOR_SCALE: f32 = 0.15;

#[derive(Debug)]
pub enum SpzError {
    Io(std::io::Error),
    BadMagic(u32),
    UnsupportedVersion(u32),
    Truncated { expected: usize, got: usize },
}

impl From<std::io::Error> for SpzError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for SpzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic(m) => write!(f, "bad SPZ magic: {m:#010x} (expected {SPZ_MAGIC:#010x})"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported SPZ version: {v} (this reader supports legacy v2 gzip)")
            }
            Self::Truncated { expected, got } => {
                write!(f, "truncated SPZ payload: expected {expected} bytes, got {got}")
            }
        }
    }
}

impl std::error::Error for SpzError {}

/// Resolution of the 24-bit fixed-point position quantization for a given
/// `fractionalBits` setting, i.e. the size of one quantization step in world
/// units. Round-trip position error is bounded by half this value.
pub fn position_quant_step(fractional_bits: u8) -> f32 {
    1.0 / (1u32 << fractional_bits) as f32
}

/// Load an SPZ file from disk into Ochroma [`GaussianSplat`]s.
pub fn load_spz(path: &Path) -> Result<Vec<GaussianSplat>, SpzError> {
    let file = std::fs::File::open(path)?;
    load_spz_from_reader(std::io::BufReader::new(file))
}

/// Load SPZ from any reader (gzip stream). Useful for in-memory round-trips.
pub fn load_spz_from_reader(reader: impl Read) -> Result<Vec<GaussianSplat>, SpzError> {
    // The whole file is one gzip stream.
    let mut gz = flate2::read::GzDecoder::new(reader);
    let mut payload = Vec::new();
    gz.read_to_end(&mut payload)?;
    decode_payload(&payload)
}

/// Decode an already-decompressed SPZ payload (header + attribute arrays).
fn decode_payload(payload: &[u8]) -> Result<Vec<GaussianSplat>, SpzError> {
    if payload.len() < 16 {
        return Err(SpzError::Truncated { expected: 16, got: payload.len() });
    }
    let magic = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    if magic != SPZ_MAGIC {
        return Err(SpzError::BadMagic(magic));
    }
    let version = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    if version != SPZ_VERSION_LEGACY {
        return Err(SpzError::UnsupportedVersion(version));
    }
    let num_points = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let sh_degree = payload[12];
    let fractional_bits = payload[13];
    // payload[14] = flags (antialiased), payload[15] = reserved — not needed here.

    // SH coefficients per point per color channel: degree 0→0, 1→3, 2→8, 3→15.
    let sh_per_channel = match sh_degree {
        0 => 0,
        1 => 3,
        2 => 8,
        _ => 15,
    };

    // Tight attribute arrays in order: positions(9) alphas(1) colors(3) scales(3)
    // rotations(3) sh(sh_per_channel*3).
    let pos_bytes = num_points * 9;
    let alpha_bytes = num_points;
    let color_bytes = num_points * 3;
    let scale_bytes = num_points * 3;
    let rot_bytes = num_points * 3;
    let sh_bytes = num_points * sh_per_channel * 3;
    let expected = 16 + pos_bytes + alpha_bytes + color_bytes + scale_bytes + rot_bytes + sh_bytes;
    if payload.len() < expected {
        return Err(SpzError::Truncated { expected, got: payload.len() });
    }

    let mut off = 16;
    let pos = &payload[off..off + pos_bytes];
    off += pos_bytes;
    let alpha = &payload[off..off + alpha_bytes];
    off += alpha_bytes;
    let color = &payload[off..off + color_bytes];
    off += color_bytes;
    let scale = &payload[off..off + scale_bytes];
    off += scale_bytes;
    let rot = &payload[off..off + rot_bytes];
    // sh ignored on import (we only carry a 16-band spectrum, no view-dependent SH).

    let inv_frac = 1.0 / (1u32 << fractional_bits) as f32;

    let mut splats = Vec::with_capacity(num_points);
    for i in 0..num_points {
        // --- position: 3 axes × signed 24-bit little-endian fixed-point.
        let mut p = [0.0f32; 3];
        for (a, pa) in p.iter_mut().enumerate() {
            let b = &pos[i * 9 + a * 3..i * 9 + a * 3 + 3];
            let mut v = (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16);
            // sign-extend from bit 23
            if v & 0x0080_0000 != 0 {
                v |= 0xff00_0000;
            }
            *pa = (v as i32) as f32 * inv_frac;
        }

        // --- alpha: inverse sigmoid then back to u8 opacity (Ochroma carries u8).
        let a01 = alpha[i] as f32 / 255.0;
        let opacity = (a01.clamp(0.0, 1.0) * 255.0).round() as u8;

        // --- color: SPZ stores SH DC coefficient c; display RGB = 0.5 + C0*c.
        let mut rgb = [0.5f32; 3];
        for (k, rk) in rgb.iter_mut().enumerate() {
            let c = (color[i * 3 + k] as f32 / 255.0 - 0.5) / COLOR_SCALE;
            *rk = (0.5 + SH_C0 * c).clamp(0.0, 1.0);
        }

        // --- scale: log-space u8 → linear half-axes.
        let mut s = [0.0f32; 3];
        for (k, sk) in s.iter_mut().enumerate() {
            let log_scale = scale[i * 3 + k] as f32 / 16.0 - 10.0;
            *sk = log_scale.exp();
        }

        // --- rotation: quaternion-first-three, w reconstructed (positive).
        let qx = rot[i * 3] as f32 / 127.5 - 1.0;
        let qy = rot[i * 3 + 1] as f32 / 127.5 - 1.0;
        let qz = rot[i * 3 + 2] as f32 / 127.5 - 1.0;
        let qw = (1.0 - (qx * qx + qy * qy + qz * qz)).max(0.0).sqrt();
        let q = glam::Quat::from_xyzw(qx, qy, qz, qw).normalize();

        // RGB → 16-band spectrum via the SAME Smits upsampler as PLY.
        let spectral_f32 = SpectralUpsampler::from_rgb(rgb[0], rgb[1], rgb[2]);
        let spectral: [u16; GaussianSplat::BANDS] =
            std::array::from_fn(|b| f16::from_f32(spectral_f32[b]).to_bits());

        splats.push(GaussianSplat::volume(p, s, q, opacity, spectral));
    }

    Ok(splats)
}

/// Serialize splats to the SPZ decompressed payload (header + attribute arrays),
/// before gzip. Exposed for tests that want to inspect the raw layout.
fn encode_payload(splats: &[GaussianSplat]) -> Vec<u8> {
    let num_points = splats.len();
    let mut out = Vec::new();

    // --- 16-byte header ---
    out.extend_from_slice(&SPZ_MAGIC.to_le_bytes());
    out.extend_from_slice(&SPZ_VERSION_LEGACY.to_le_bytes());
    out.extend_from_slice(&(num_points as u32).to_le_bytes());
    out.push(0); // shDegree (we emit DC-only color, degree 0)
    out.push(DEFAULT_FRACTIONAL_BITS); // fractionalBits
    out.push(0); // flags (not antialiased)
    out.push(0); // reserved

    // Pre-size the six tight arrays so we can fill them positionally.
    let mut positions = Vec::with_capacity(num_points * 9);
    let mut alphas = Vec::with_capacity(num_points);
    let mut colors = Vec::with_capacity(num_points * 3);
    let mut scales = Vec::with_capacity(num_points * 3);
    let mut rotations = Vec::with_capacity(num_points * 3);
    // sh array is empty for degree 0.

    let frac = (1u32 << DEFAULT_FRACTIONAL_BITS) as f32;

    for s in splats {
        // --- position: round(coord * 2^fractionalBits) as signed 24-bit LE.
        let p = s.position();
        for &c in &p {
            let fixed = (c * frac).round() as i32;
            let u = fixed as u32; // two's-complement; we keep low 24 bits.
            positions.push((u & 0xff) as u8);
            positions.push(((u >> 8) & 0xff) as u8);
            positions.push(((u >> 16) & 0xff) as u8);
        }

        // --- alpha: sigmoid(opacity-as-logit)? No — Ochroma opacity is already
        // a linear [0,255] coverage. SPZ stores sigmoid(rawAlpha)*255, and on
        // load we invert directly to [0,255]. To round-trip the u8 exactly we
        // store the opacity byte as-is (which equals round(sigmoid(a)*255) for
        // whatever a produced it). This keeps load(write(x)) == x for opacity.
        alphas.push(s.opacity());

        // --- color: collapse 16-band spectrum → RGB (same band grouping as
        // write_ply), then encode the SH DC coefficient.
        let (r, g, b) = spectrum_to_rgb(s);
        for c in [r, g, b] {
            // invert load's display-RGB decode to recover the SH DC coefficient,
            // then apply SPZ color quantization.
            let dc = (c - 0.5) / SH_C0;
            let q = (dc * COLOR_SCALE * 255.0 + 0.5 * 255.0).round().clamp(0.0, 255.0);
            colors.push(q as u8);
        }

        // --- scale: linear half-axes → log → u8.
        let sc = s.scales();
        for &lin in &sc {
            let log_scale = lin.max(1e-8).ln();
            let q = ((log_scale + 10.0) * 16.0).round().clamp(0.0, 255.0);
            scales.push(q as u8);
        }

        // --- rotation: quaternion-first-three (store xyz on w-positive sphere).
        let q = s.decoded_rotation().normalize();
        // Put quaternion on the positive-w hemisphere so w reconstructs as +sqrt.
        let sign = if q.w < 0.0 { -1.0 } else { 1.0 };
        for comp in [q.x, q.y, q.z] {
            let v = ((comp * sign) * 127.5 + 127.5).round().clamp(0.0, 255.0);
            rotations.push(v as u8);
        }
    }

    out.extend_from_slice(&positions);
    out.extend_from_slice(&alphas);
    out.extend_from_slice(&colors);
    out.extend_from_slice(&scales);
    out.extend_from_slice(&rotations);
    // no sh bytes for degree 0
    out
}

/// Collapse a splat's 16-band spectrum to approximate linear RGB using the same
/// band grouping as `ply_loader::write_ply` (blue 0..5, green 5..11, red 11..16).
fn spectrum_to_rgb(s: &GaussianSplat) -> (f32, f32, f32) {
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
    (
        (rsum / 5.0).clamp(0.0, 1.0),
        (gsum / 6.0).clamp(0.0, 1.0),
        (bsum / 5.0).clamp(0.0, 1.0),
    )
}

/// Write Gaussian splats to an SPZ file (gzip-compressed legacy v2 container).
pub fn write_spz(path: &Path, splats: &[GaussianSplat]) -> Result<(), SpzError> {
    let bytes = write_spz_to_bytes(splats)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Serialize splats to an in-memory SPZ (gzip) byte vector.
pub fn write_spz_to_bytes(splats: &[GaussianSplat]) -> Result<Vec<u8>, SpzError> {
    let payload = encode_payload(splats);
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(&payload)?;
    Ok(enc.finish()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ply_loader::write_ply_to_bytes;
    use crate::spectral_upsampler::SpectralUpsampler;

    fn splat_rgb(pos: [f32; 3], scale: [f32; 3], r: f32, g: f32, b: f32, op: u8) -> GaussianSplat {
        let spectral_f32 = SpectralUpsampler::from_rgb(r, g, b);
        let spectral: [u16; 16] = std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits());
        GaussianSplat::volume(pos, scale, glam::Quat::IDENTITY, op, spectral)
    }

    fn splat_rot(pos: [f32; 3], q: glam::Quat) -> GaussianSplat {
        let spectral_f32 = SpectralUpsampler::from_rgb(0.5, 0.5, 0.5);
        let spectral: [u16; 16] = std::array::from_fn(|i| f16::from_f32(spectral_f32[i]).to_bits());
        GaussianSplat::volume(pos, [0.1, 0.1, 0.1], q.normalize(), 200, spectral)
    }

    #[test]
    fn roundtrip_positions_within_fixed_point_tolerance() {
        // Default 12 fractional bits → step = 1/4096; max error half a step.
        let step = position_quant_step(DEFAULT_FRACTIONAL_BITS);
        let tol = step / 2.0 + 1e-7; // +epsilon for the round() boundary
        assert!((step - 1.0 / 4096.0).abs() < 1e-9, "step should be 1/4096, got {step}");

        let original = vec![
            splat_rgb([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 0.9, 0.1, 0.1, 230),
            splat_rgb([-4.5, 0.0, 7.25], [0.5, 0.5, 0.5], 0.2, 0.8, 0.3, 128),
            splat_rgb([10.123, -3.777, -2.001], [0.05, 0.07, 0.09], 0.1, 0.1, 0.9, 64),
        ];

        let bytes = write_spz_to_bytes(&original).expect("write");
        let loaded = load_spz_from_reader(&bytes[..]).expect("load");

        assert_eq!(loaded.len(), original.len(), "count must be exact");
        for (o, l) in original.iter().zip(loaded.iter()) {
            let op = o.position();
            let lp = l.position();
            for k in 0..3 {
                assert!(
                    (op[k] - lp[k]).abs() <= tol,
                    "position[{k}] {} vs {} exceeds 24-bit fixed-point tol {tol}",
                    op[k],
                    lp[k]
                );
            }
        }
    }

    #[test]
    fn roundtrip_opacity_exact() {
        let original = vec![
            splat_rgb([0.0, 0.0, 0.0], [0.1, 0.1, 0.1], 0.5, 0.5, 0.5, 0),
            splat_rgb([0.0, 0.0, 0.0], [0.1, 0.1, 0.1], 0.5, 0.5, 0.5, 200),
            splat_rgb([0.0, 0.0, 0.0], [0.1, 0.1, 0.1], 0.5, 0.5, 0.5, 255),
        ];
        let bytes = write_spz_to_bytes(&original).expect("write");
        let loaded = load_spz_from_reader(&bytes[..]).expect("load");
        for (o, l) in original.iter().zip(loaded.iter()) {
            assert_eq!(o.opacity(), l.opacity(), "opacity u8 must round-trip exactly");
        }
    }

    #[test]
    fn roundtrip_rotation_within_quantization_error() {
        // 8-bit per component → step on the [-1,1] range is 1/127.5 ≈ 7.84e-3,
        // max per-component error half a step. Allow a little slack for the
        // w-reconstruction + renormalization.
        let comp_tol = 1.0 / 127.5 / 2.0 + 5e-3;
        let rots = [
            glam::Quat::IDENTITY,
            glam::Quat::from_rotation_y(0.7),
            glam::Quat::from_rotation_x(1.3) * glam::Quat::from_rotation_z(0.4),
            glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 2.1),
        ];
        let original: Vec<_> = rots
            .iter()
            .map(|&q| splat_rot([0.0, 0.0, 0.0], q))
            .collect();
        let bytes = write_spz_to_bytes(&original).expect("write");
        let loaded = load_spz_from_reader(&bytes[..]).expect("load");
        for (o, l) in original.iter().zip(loaded.iter()) {
            let qo = o.decoded_rotation().normalize();
            let ql = l.decoded_rotation().normalize();
            // Quaternions q and -q are the same rotation; align sign first.
            let ql = if qo.dot(ql) < 0.0 { -ql } else { ql };
            for (a, b) in [(qo.x, ql.x), (qo.y, ql.y), (qo.z, ql.z), (qo.w, ql.w)] {
                assert!(
                    (a - b).abs() <= comp_tol,
                    "rotation component {a} vs {b} exceeds quant tol {comp_tol}"
                );
            }
        }
    }

    #[test]
    fn compression_beats_ply_by_spec_margin() {
        // A few hundred splats (demo-scale scene), pseudo-random but deterministic.
        let mut splats = Vec::new();
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..400 {
            let p = [next() * 20.0 - 10.0, next() * 20.0 - 10.0, next() * 20.0 - 10.0];
            let s = [0.02 + next() * 0.2, 0.02 + next() * 0.2, 0.02 + next() * 0.2];
            splats.push(splat_rgb(p, s, next(), next(), next(), (next() * 255.0) as u8));
        }

        let ply = write_ply_to_bytes(&splats);
        let spz = write_spz_to_bytes(&splats).expect("write spz");
        let ratio = spz.len() as f32 / ply.len() as f32;
        println!(
            "compression: {} splats — PLY {} bytes, SPZ {} bytes, ratio {:.1}%",
            splats.len(),
            ply.len(),
            spz.len(),
            ratio * 100.0
        );
        assert!(
            ratio < 0.35,
            "SPZ should be < 35% of PLY size; got {:.1}% (PLY {}, SPZ {})",
            ratio * 100.0,
            ply.len(),
            spz.len()
        );
    }

    #[test]
    fn malformed_header_rejected() {
        // Valid gzip stream, but payload has a bogus magic.
        let mut bad_payload = vec![0u8; 16];
        bad_payload[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        let mut enc =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&bad_payload).unwrap();
        let gz = enc.finish().unwrap();

        match load_spz_from_reader(&gz[..]) {
            Err(SpzError::BadMagic(_)) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }

        // Wrong version.
        let mut bad_ver = vec![0u8; 16];
        bad_ver[0..4].copy_from_slice(&SPZ_MAGIC.to_le_bytes());
        bad_ver[4..8].copy_from_slice(&99u32.to_le_bytes());
        let mut enc2 =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc2.write_all(&bad_ver).unwrap();
        let gz2 = enc2.finish().unwrap();
        match load_spz_from_reader(&gz2[..]) {
            Err(SpzError::UnsupportedVersion(99)) => {}
            other => panic!("expected UnsupportedVersion(99), got {other:?}"),
        }

        // Truncated payload (header claims points but no data follows).
        let mut trunc = vec![0u8; 16];
        trunc[0..4].copy_from_slice(&SPZ_MAGIC.to_le_bytes());
        trunc[4..8].copy_from_slice(&SPZ_VERSION_LEGACY.to_le_bytes());
        trunc[8..12].copy_from_slice(&5u32.to_le_bytes()); // claims 5 points
        let mut enc3 =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc3.write_all(&trunc).unwrap();
        let gz3 = enc3.finish().unwrap();
        match load_spz_from_reader(&gz3[..]) {
            Err(SpzError::Truncated { .. }) => {}
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn color_roundtrips_through_spectral_bottleneck() {
        // A saturated red splat should come back red-dominant after the
        // spectrum→RGB→SH-DC→spectrum round-trip.
        let original = vec![splat_rgb([0.0, 0.0, 0.0], [0.1, 0.1, 0.1], 0.95, 0.05, 0.05, 220)];
        let bytes = write_spz_to_bytes(&original).expect("write");
        let loaded = load_spz_from_reader(&bytes[..]).expect("load");
        let s = &loaded[0];
        let blue: f32 = (0..5).map(|b| s.spectral_f32(b)).sum();
        let red: f32 = (11..16).map(|b| s.spectral_f32(b)).sum();
        assert!(red > blue, "red splat must stay red-dominant: red {red} vs blue {blue}");
    }
}
