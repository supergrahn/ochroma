//! Conversion from Ochroma's `GaussianSplat` to Spectra's `GaussianDisc` / `GaussianBlob`.
//!
//! `GaussianSplat.kind == 0` → `GaussianDisc` (2DGS surface)
//! `GaussianSplat.kind == 1` → `GaussianBlob` (3DGS volume)
//!
//! Spectral: Ochroma has 16 bands (380–755 nm); Spectra takes bands 0–7 (380–660 nm).
//! Bands 8–15 (680–755 nm) are discarded.

use spectra_renderer::{GaussianBlob, GaussianDisc};
use vox_core::types::GaussianSplat;

/// Convert a surface splat (kind == 0) to a [`GaussianDisc`].
///
/// Normal = cross(tangent_u, tangent_v), already unit from GaussianSplat invariants.
/// Spectral bands 8–15 are discarded (only 380–660 nm range retained).
/// `roughness = 128` (≈0.5) and `metalness = 0` are PBR defaults until Ochroma adds these fields.
pub fn splat_to_disc(s: &GaussianSplat) -> GaussianDisc {
    debug_assert_eq!(s.kind(), 0, "splat_to_disc called on a volume splat");

    let tu = s.tangent_u();
    let tv = s.tangent_v();

    // Normal = cross(tangent_u, tangent_v).
    let nx = tu[1] * tv[2] - tu[2] * tv[1];
    let ny = tu[2] * tv[0] - tu[0] * tv[2];
    let nz = tu[0] * tv[1] - tu[1] * tv[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-8);

    let normal = [
        ((nx / len) * 32767.0) as i16,
        ((ny / len) * 32767.0) as i16,
        ((nz / len) * 32767.0) as i16,
    ];

    // First 8 spectral bands (380–660 nm); bands 8–15 discarded.
    let src = s.spectral();
    let albedo = [
        src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
    ];
    let emission = [0u16; 8]; // GaussianSplat has no emission channel

    GaussianDisc {
        position: s.position(),
        normal,
        _pad0: [0; 2],
        radius: [s.scale_u(), s.scale_v()],
        opacity: s.opacity(),
        roughness: 128, // 0.5 default
        metalness: 0,
        _pad1: 0,
        albedo,
        emission,
    }
}

/// Convert a volume splat (kind == 1) to a [`GaussianBlob`].
///
/// `extinction` is set from opacity (density proxy).
/// Phase H-G: single forward lobe default (`phase_g2 = 0`, `phase_blend = 255`).
pub fn splat_to_blob(s: &GaussianSplat) -> GaussianBlob {
    debug_assert_eq!(s.kind(), 1, "splat_to_blob called on a surface splat");

    // First 8 spectral bands (380–660 nm).
    let src = s.spectral();
    let albedo = [
        src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
    ];
    let emission = [0u16; 8];

    GaussianBlob {
        position: s.position(),
        scale: s.scales(),
        rotation: s.rotation_raw(),
        extinction: s.opacity(),
        phase_g1: 90,    // ~0.7 forward lobe (generic fog/cloud default)
        phase_g2: 0,     // single-lobe default
        phase_blend: 255, // pure g1
        _pad: [0; 4],
        albedo,
        emission,
    }
}

/// Convert a slice of [`GaussianSplat`] into `(surfaces, volumes)`.
///
/// Surface splats (kind == 0) become [`GaussianDisc`]; volume splats (kind == 1)
/// become [`GaussianBlob`]. Any unknown kind value is silently skipped.
pub fn convert_splats(splats: &[GaussianSplat]) -> (Vec<GaussianDisc>, Vec<GaussianBlob>) {
    let mut surfaces = Vec::new();
    let mut volumes = Vec::new();
    for s in splats {
        if s.is_surface() {
            surfaces.push(splat_to_disc(s));
        } else if s.is_volume() {
            volumes.push(splat_to_blob(s));
        }
    }
    (surfaces, volumes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vox_core::types::GaussianSplat;

    fn zero_spectral() -> [u16; 16] {
        [0u16; 16]
    }

    #[test]
    fn surface_normal_is_z_axis() {
        // tangent_u = X, tangent_v = Y  →  normal = Z
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let disc = splat_to_disc(&s);
        // nz should be close to 32767
        assert!(disc.normal[2] > 32_000, "nz = {} expected > 32000", disc.normal[2]);
        assert_eq!(disc.normal[0], 0);
        assert_eq!(disc.normal[1], 0);
    }

    #[test]
    fn surface_radii_copied() {
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.25,
            0.75,
            200,
            zero_spectral(),
        );
        let disc = splat_to_disc(&s);
        assert_eq!(disc.radius[0], 0.25);
        assert_eq!(disc.radius[1], 0.75);
        assert_eq!(disc.opacity, 200);
    }

    #[test]
    fn surface_spectral_bands_0_to_7_copied() {
        let mut spectral = [0u16; 16];
        spectral[0] = 1000;
        spectral[7] = 2000;
        spectral[8] = 9999; // must be discarded
        let s = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            spectral,
        );
        let disc = splat_to_disc(&s);
        assert_eq!(disc.albedo[0], 1000);
        assert_eq!(disc.albedo[7], 2000);
        // Only 8 bands — band 8 from source is not present in output
        assert_eq!(disc.albedo.len(), 8);
    }

    #[test]
    fn volume_extinction_from_opacity() {
        let s = GaussianSplat::volume(
            [1.0, 2.0, 3.0],
            [0.5, 0.5, 0.5],
            glam::Quat::IDENTITY,
            180,
            zero_spectral(),
        );
        let blob = splat_to_blob(&s);
        assert_eq!(blob.extinction, 180);
    }

    #[test]
    fn volume_scale_copied() {
        let s = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            glam::Quat::IDENTITY,
            255,
            zero_spectral(),
        );
        let blob = splat_to_blob(&s);
        assert_eq!(blob.scale[0], 1.0);
        assert_eq!(blob.scale[1], 2.0);
        assert_eq!(blob.scale[2], 3.0);
    }

    #[test]
    fn convert_splats_dispatches_by_kind() {
        let surf = GaussianSplat::surface(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            255,
            zero_spectral(),
        );
        let vol = GaussianSplat::volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            glam::Quat::IDENTITY,
            128,
            zero_spectral(),
        );
        let (surfs, vols) = convert_splats(&[surf, vol]);
        assert_eq!(surfs.len(), 1);
        assert_eq!(vols.len(), 1);
    }
}
