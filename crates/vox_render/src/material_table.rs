//! `MaterialTable` — the scene-wide shared material slot allocator.
//!
//! Design: the game-side spec lives at
//! `urban_horizon/docs/superpowers/specs/2026-06-18-solid-id-and-material-system.md`
//! §4.5. This extracts the inline `HashMap` dedup + slot allocation that used to
//! live in the game's `spectra_frame` (the static-city material loop, the cim
//! palette append, and the scatter proto material push) into a single typed
//! allocator so material-slot allocation happens in exactly one place.
//!
//! M1+M2 scope (behaviour-preserving): the dedup key, the colour bucket
//! quantization, and the slot value (`= materials.len() as u32`) are identical to
//! the prior inline code. The 256-material cap (the old `>= 255 ⇒ alias to 254`
//! clamp) is DELETED — slots are `u32` and the table grows to whatever the scene
//! needs. Nothing past slot 254 is aliased anymore.

// The `spectra-native` feature gate mirrors the game's `#[cfg(feature = "spectra")]`
// pattern — PbrMaterial is only compiled when Spectra is compiled in.
#![cfg(feature = "spectra-native")]

use std::collections::HashMap;

use crate::splat_backend::PbrMaterial;

/// Dedup key for a shared material slot.
///
/// Mirrors the historical inline tuple `(channel, albedo_tex, normal_tex,
/// roughness_tex, displacement_tex, color_bucket)` for opaque materials, with
/// opacity/vegetation added for cutout cards so masked foliage never aliases an
/// opaque material. Textured
/// materials share regardless of base colour (the texture IS the colour, so
/// `color` is the zero bucket); untextured materials carry a coarse 2-level /
/// channel colour bucket so flat buildings still read varied without exploding
/// the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialKey {
    pub channel: u8,
    pub albedo_tex: i32,
    pub normal_tex: i32,
    pub roughness_tex: i32,
    pub displacement_tex: i32,
    pub opacity_tex: i32,
    pub vegetation_bsdf: bool,
    pub color_bucket: u32,
}

impl MaterialKey {
    /// Build the dedup key for a resolved material exactly as the prior inline
    /// code did: textured ⇒ `color_bucket = 0`; untextured ⇒ a 3-bit
    /// 2-level/channel bucket of the base colour.
    pub fn from_material(
        channel: u8,
        mat: &PbrMaterial,
        albedo_tex: i32,
        normal_tex: i32,
        roughness_tex: i32,
        displacement_tex: i32,
    ) -> Self {
        let color_bucket = if albedo_tex >= 0 {
            0u32
        } else {
            let q = |c: f32| ((c.clamp(0.0, 1.0) * 2.0) as u32).min(1);
            (q(mat.base_color[0]) << 2) | (q(mat.base_color[1]) << 1) | q(mat.base_color[2])
        };
        Self {
            channel,
            albedo_tex,
            normal_tex,
            roughness_tex,
            displacement_tex,
            opacity_tex: mat.opacity_tex,
            vegetation_bsdf: mat.vegetation_bsdf,
            color_bucket,
        }
    }
}

/// Owns the scene-wide shared material table + its dedup index and the parallel
/// spectral SPD list. Slot ids are `u32`; there is NO 256 cap and NO aliasing.
pub struct MaterialTable {
    materials: Vec<PbrMaterial>,
    /// Parallel `(slot, spd)` list the scene upload consumes (same shape as the
    /// old `spectral_spd` vec).
    spectral: Vec<(u32, [f32; 16])>,
    lookup: HashMap<MaterialKey, u32>,
}

impl MaterialTable {
    /// New empty table. `cap` pre-reserves the backing vectors (was
    /// `Vec::with_capacity(meshes.len())` at the call site).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            materials: Vec::with_capacity(cap),
            spectral: Vec::with_capacity(cap),
            lookup: HashMap::new(),
        }
    }

    /// Dedup + intern; returns the `u32` slot. On a key hit returns the existing
    /// slot; on a miss pushes `mat`/`spd` and returns the new slot
    /// (`= self.len()` pre-push). NO 256/254 clamp — the table grows freely.
    pub fn intern(&mut self, key: MaterialKey, mat: PbrMaterial, spd: [f32; 16]) -> u32 {
        if let Some(&s) = self.lookup.get(&key) {
            return s;
        }
        let s = self.materials.len() as u32;
        self.materials.push(mat);
        self.spectral.push((s, spd));
        self.lookup.insert(key, s);
        s
    }

    /// Push a material WITHOUT dedup (the scatter proto path: each proto gets its
    /// own slot regardless of key). Returns the new slot.
    pub fn push_undeduped(&mut self, mat: PbrMaterial, spd: [f32; 16]) -> u32 {
        let s = self.materials.len() as u32;
        self.materials.push(mat);
        self.spectral.push((s, spd));
        s
    }

    /// Reserve a contiguous block of palette slots (the cim clothing palette).
    /// Returns the base slot; appended materials carry NO spectral SPD entry
    /// (matching the prior inline cim-palette append, which pushed only into
    /// `materials`).
    pub fn reserve_palette(&mut self, palette: &[PbrMaterial]) -> u32 {
        let base = self.materials.len() as u32;
        self.materials.extend_from_slice(palette);
        base
    }

    /// The shared material table (consumed by `meshes_to_instanced_scene`).
    pub fn materials(&self) -> &[PbrMaterial] {
        &self.materials
    }

    /// The parallel `(slot, spd)` spectral list.
    pub fn spectral(&self) -> &[(u32, [f32; 16])] {
        &self.spectral
    }

    /// Number of allocated slots.
    pub fn len(&self) -> u32 {
        self.materials.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}
