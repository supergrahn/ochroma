//! The LOSSLESS Gaussian-splat codec for project save/load (AAA Spec 06).
//!
//! [`to_saved_geom`] / [`from_saved_geom`] convert a runtime [`GaussianSplat`] to
//! and from the serializable [`SavedSplatGeom`] record WITHOUT decoding the
//! quantized fields. The 16-band spectral signature is stored as the RAW `u16`
//! (f16 bits) it already is, and the quaternion as the RAW `i16` quantization —
//! so the on-disk record carries the exact in-memory bytes and reconstruction is
//! bit-identical, not approximate.
//!
//! The one subtlety is that [`GaussianSplat::volume`] re-quantizes its rotation
//! from a `Quat` (`comp * 32767.0 as i16`). Because that map is a saturating
//! `f32 → i16` cast and `i16 as f32 / 32767.0` lands every i16 back on a value
//! that re-quantizes to itself, feeding the stored `i16` back through the `Quat`
//! constructor yields the SAME `i16` — the volume round-trip is bit-exact for
//! every possible rotation, not just the identity.

use crate::world_save::SavedSplatGeom;
use glam::Quat;
use vox_core::types::GaussianSplat;

/// Serialize a runtime splat into a [`SavedSplatGeom`] for the project file.
///
/// Every quantized field (the 16-band `spectral` `[u16;16]` and the `[i16;4]`
/// rotation) is copied VERBATIM — no f16/quaternion decode — so the record holds
/// the exact in-memory bytes and the inverse [`from_saved_geom`] reproduces the
/// splat bit-for-bit.
pub fn to_saved_geom(s: &GaussianSplat) -> SavedSplatGeom {
    SavedSplatGeom {
        position: s.position(),
        kind: s.kind(),
        tangent_u: s.tangent_u(),
        scale_u: s.scale_u(),
        tangent_v: s.tangent_v(),
        scale_v: s.scale_v(),
        rotation: s.rotation_raw(),
        scale_w: s.scale_w(),
        opacity: s.opacity(),
        spectral: *s.spectral(),
    }
}

/// Replace a non-finite scale with `0.0` so a corrupt save can never feed a NaN
/// or infinity into a splat constructor (NO-PANIC contract).
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Reconstruct a runtime splat from a saved record (the inverse of
/// [`to_saved_geom`]).
///
/// `kind == 0` rebuilds a 2DGS surface; ANY other kind (including an unknown
/// value from a corrupt file) rebuilds a 3DGS volume. The stored `i16` rotation
/// is decoded to a `Quat` and handed to [`GaussianSplat::volume`], which
/// re-quantizes it back to the IDENTICAL `i16` (the cast round-trips exactly), so
/// `rotation_raw()` matches the original bit-for-bit. The spectral `[u16;16]` is
/// passed through untouched, so it too is bit-identical. Non-finite scales are
/// sanitized to `0.0` before construction (NO-PANIC).
pub fn from_saved_geom(g: &SavedSplatGeom) -> GaussianSplat {
    let su = finite_or_zero(g.scale_u);
    let sv = finite_or_zero(g.scale_v);
    let sw = finite_or_zero(g.scale_w);
    if g.kind == 0 {
        GaussianSplat::surface(
            g.position,
            g.tangent_u,
            g.tangent_v,
            su,
            sv,
            g.opacity,
            g.spectral,
        )
    } else {
        // Decode the raw i16 quantization back to a Quat; `volume` re-quantizes it
        // to the SAME i16 (saturating cast round-trips), so rotation_raw() is exact.
        let rot = Quat::from_xyzw(
            g.rotation[0] as f32 / 32767.0,
            g.rotation[1] as f32 / 32767.0,
            g.rotation[2] as f32 / 32767.0,
            g.rotation[3] as f32 / 32767.0,
        );
        GaussianSplat::volume(g.position, [su, sv, sw], rot, g.opacity, g.spectral)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Build a distinctive 16-band spectral pattern as raw f16 bits, so a dropped
    /// or reordered band makes the bit-exact assertion fail.
    fn spectral_bits(seed: f32) -> [u16; 16] {
        std::array::from_fn(|b| f16::from_f32(b as f32 * 0.0625 + seed).to_bits())
    }

    #[test]
    fn codec_round_trips_volume_and_surface_bit_exact() {
        // --- Volume splat with a NON-identity, normalized rotation ---
        let rot = Quat::from_axis_angle(glam::Vec3::new(0.3, 0.7, 0.2).normalize(), 1.1);
        let vol = GaussianSplat::volume(
            [1.5, -2.25, 3.75],
            [0.4, 0.8, 1.2],
            rot,
            200,
            spectral_bits(0.5),
        );
        let vol_rt = from_saved_geom(&to_saved_geom(&vol));
        assert_eq!(vol_rt.spectral(), vol.spectral(), "volume spectral must be bit-identical");
        assert_eq!(vol_rt.rotation_raw(), vol.rotation_raw(), "volume rotation i16 must be bit-identical");
        assert_eq!(vol_rt.position(), vol.position(), "volume position must match");
        assert_eq!(vol_rt.scales(), vol.scales(), "volume scales must match");
        assert_eq!(vol_rt.opacity(), vol.opacity(), "volume opacity must match");
        assert_eq!(vol_rt.kind(), vol.kind(), "volume kind must match");
        // Guard: the rotation is genuinely non-identity (the test would be hollow
        // if it accidentally quantized to the identity [0,0,0,32767]).
        assert_ne!(vol.rotation_raw(), [0, 0, 0, 32767], "test rotation must be non-identity");

        // --- Surface splat with a DISTINCTIVE spectral pattern ---
        let surf = GaussianSplat::surface(
            [-0.5, 4.0, 7.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.6,
            0.9,
            128,
            spectral_bits(0.125),
        );
        let surf_rt = from_saved_geom(&to_saved_geom(&surf));
        assert_eq!(surf_rt.spectral(), surf.spectral(), "surface spectral must be bit-identical");
        assert_eq!(surf_rt.rotation_raw(), surf.rotation_raw(), "surface rotation i16 must be bit-identical");
        assert_eq!(surf_rt.position(), surf.position(), "surface position must match");
        assert_eq!(surf_rt.scales(), surf.scales(), "surface scales must match");
        assert_eq!(surf_rt.opacity(), surf.opacity(), "surface opacity must match");
        assert_eq!(surf_rt.kind(), surf.kind(), "surface kind must match");
        // The two splats must carry genuinely different spectra (not aliased).
        assert_ne!(surf.spectral(), vol.spectral(), "the two test spectra must differ");
    }

    #[test]
    fn from_saved_geom_sanitizes_non_finite_scale() {
        // A corrupt save with a NaN/inf scale must NOT panic and must clamp to 0.0.
        let mut g = to_saved_geom(&GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            Quat::IDENTITY,
            255,
            spectral_bits(0.0),
        ));
        g.scale_u = f32::NAN;
        g.scale_v = f32::INFINITY;
        let s = from_saved_geom(&g);
        assert_eq!(s.scales(), [0.0, 0.0, 1.0], "non-finite scales must be sanitized to 0.0");
    }
}
