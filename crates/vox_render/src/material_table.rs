//! `MaterialTable` — the scene-wide shared material slot allocator.
//!
//! Design: the game-side spec lives at
//! `urban_horizon/docs/superpowers/specs/2026-06-18-solid-id-and-material-system.md`
//! §4.5. This extracts the inline `HashMap` dedup + slot allocation that used to
//! live in the game's `spectra_frame` (the static-city material loop, the cim
//! palette append, and the scatter proto material push) into a single typed
//! allocator so material-slot allocation happens in exactly one place.
//!
//! Slots are `u32` and the table grows to whatever the scene needs. The key uses
//! exact authored shading identity rather than the former three-bit colour bucket,
//! so visually distinct surfaces cannot collapse onto the first material that
//! happened to enter the table.

// The `spectra-native` feature gate mirrors the game's `#[cfg(feature = "spectra")]`
// pattern — PbrMaterial is only compiled when Spectra is compiled in.
#![cfg(feature = "spectra-native")]

use std::collections::HashMap;

use crate::splat_backend::PbrMaterial;

/// Dedup key for a shared material slot.
///
/// Captures every authored value that changes GPU shading. Textured materials
/// share regardless of fallback `base_color` only when the sampled albedo
/// replaces it. Factor x texture materials retain exact authored colour. This prevents two surfaces
/// using the same texture slots from aliasing when their roughness, relief, glass
/// optics, or UV policy differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialKey {
    pub channel: u8,
    pub albedo_tex: i32,
    pub normal_tex: i32,
    pub roughness_tex: i32,
    pub displacement_tex: i32,
    pub opacity_tex: i32,
    pub vegetation_bsdf: bool,
    pub modulate_base_color_texture: bool,
    pub modulate_roughness_texture: bool,
    pub base_color_bits: [u32; 3],
    /// `[roughness, metallic, emission, displacement_scale,
    /// displacement_midlevel, uv_scale.x, uv_scale.y, transmission, ior,
    /// absorption.r, absorption.g, absorption.b, absorption_depth]`.
    pub surface_param_bits: [u32; 13],
    pub thin_walled: bool,
    pub is_water: bool,
}

impl MaterialKey {
    /// Build an exact deterministic GPU-shading identity. Values are keyed by
    /// their IEEE-754 bits so no epsilon or hash iteration order can merge two
    /// authored materials accidentally.
    pub fn from_material(
        channel: u8,
        mat: &PbrMaterial,
        albedo_tex: i32,
        normal_tex: i32,
        roughness_tex: i32,
        displacement_tex: i32,
    ) -> Self {
        let base_color_bits = if albedo_tex >= 0 && !mat.modulate_base_color_texture {
            [0; 3]
        } else {
            mat.base_color.map(f32::to_bits)
        };
        Self {
            channel,
            albedo_tex,
            normal_tex,
            roughness_tex,
            displacement_tex,
            opacity_tex: mat.opacity_tex,
            vegetation_bsdf: mat.vegetation_bsdf,
            modulate_base_color_texture: mat.modulate_base_color_texture,
            modulate_roughness_texture: mat.modulate_roughness_texture,
            base_color_bits,
            surface_param_bits: [
                mat.roughness.to_bits(),
                mat.metallic.to_bits(),
                mat.emission_strength.to_bits(),
                mat.displacement_scale.to_bits(),
                mat.displacement_midlevel.to_bits(),
                mat.uv_scale[0].to_bits(),
                mat.uv_scale[1].to_bits(),
                mat.transmission.to_bits(),
                mat.ior.to_bits(),
                mat.absorption_color[0].to_bits(),
                mat.absorption_color[1].to_bits(),
                mat.absorption_color[2].to_bits(),
                mat.absorption_depth.to_bits(),
            ],
            thin_walled: mat.thin_walled,
            is_water: mat.is_water,
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

    /// Reserve slot 0 as a GUARANTEED-OPAQUE fallback material (neutral gray,
    /// `opacity_tex = -1`, no transmission, no vegetation BSDF). Call ONCE right
    /// after construction, BEFORE any intern/push.
    ///
    /// Why: the megakernel clamps every out-of-range resolved material id
    /// (`instance base + per-triangle relative id`) to slot 0 as its OOB safety
    /// net. If slot 0 happened to be a cutout/vegetation/glass material, that
    /// clamp turned a material-id bug into INVISIBLE geometry (alpha-cutout
    /// re-launches the ray through the surface — the thin-slab-buildings class).
    /// Reserving an opaque slot 0 makes the clamp target safe by construction:
    /// a resolution bug now shows as flat gray (or magenta under isolation mode
    /// 11), never as vanished geometry.
    ///
    /// Panics if the table is non-empty (the reservation MUST be slot 0).
    pub fn reserve_opaque_fallback_slot(&mut self) -> u32 {
        assert!(
            self.materials.is_empty(),
            "reserve_opaque_fallback_slot must run before any intern/push (table has {} slots)",
            self.materials.len()
        );
        // Neutral opaque gray; flat white SPD (spectrally neutral).
        self.push_undeduped(PbrMaterial::default(), [1.0; 16])
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(mat: &PbrMaterial) -> MaterialKey {
        MaterialKey::from_material(
            0,
            mat,
            mat.albedo_tex,
            mat.normal_tex,
            mat.roughness_tex,
            mat.displacement_tex,
        )
    }

    #[test]
    fn uv_policy_and_surface_parameters_are_part_of_material_identity() {
        let mut plain = PbrMaterial {
            albedo_tex: 7,
            normal_tex: 8,
            roughness_tex: 9,
            ..PbrMaterial::default()
        };
        let mut antitiled = plain;
        antitiled.uv_scale = [1.0, -1.0];
        assert_ne!(key(&plain), key(&antitiled));

        plain.roughness = 0.42;
        assert_ne!(key(&plain), key(&antitiled));
    }

    #[test]
    fn untextured_authored_colours_do_not_collapse_to_three_bits() {
        let a = PbrMaterial {
            base_color: [0.20, 0.25, 0.30],
            ..PbrMaterial::default()
        };
        let b = PbrMaterial {
            base_color: [0.35, 0.40, 0.45],
            ..PbrMaterial::default()
        };
        assert_ne!(key(&a), key(&b));
    }

    #[test]
    fn factor_times_texture_keeps_authored_colour_in_material_identity() {
        let a = PbrMaterial {
            base_color: [0.20, 0.25, 0.30],
            albedo_tex: 7,
            modulate_base_color_texture: true,
            ..PbrMaterial::default()
        };
        let b = PbrMaterial {
            base_color: [0.35, 0.40, 0.45],
            ..a
        };
        assert_ne!(key(&a), key(&b));
    }
}
