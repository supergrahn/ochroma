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
    pub nee_emitter: bool,
    pub nee_downlight: bool,
    pub base_color_bits: [u32; 3],
    /// `[roughness, metallic, emission, displacement_scale,
    /// displacement_midlevel, uv_scale.x, uv_scale.y, transmission, ior,
    /// exterior_reflectance, absorption.r, absorption.g, absorption.b,
    /// absorption_depth]`.
    pub surface_param_bits: [u32; 14],
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
            nee_emitter: mat.nee_emitter,
            nee_downlight: mat.nee_downlight,
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
                mat.exterior_reflectance.to_bits(),
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
    /// Dedup index for CONTIGUOUS runs (`intern_run`), keyed by the whole
    /// ordered zone tuple. Separate from `lookup` because a run's identity is
    /// the sequence, not any single material.
    run_lookup: HashMap<Vec<MaterialKey>, u32>,
}

impl MaterialTable {
    /// New empty table. `cap` pre-reserves the backing vectors (was
    /// `Vec::with_capacity(meshes.len())` at the call site).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            materials: Vec::with_capacity(cap),
            spectral: Vec::with_capacity(cap),
            lookup: HashMap::new(),
            run_lookup: HashMap::new(),
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

    /// Intern a CONTIGUOUS, deduplicated run of material slots for one
    /// multi-zone shared prototype. Returns the base slot; zone `k` of the
    /// prototype resolves in the closest-hit as `base + k`, matching
    /// `InstanceRecordGpu::material_base`'s `final = base + tri.material_id`.
    ///
    /// Why this exists: a building prototype carries ~20 material zones. Giving
    /// each zone its own BLAS and its own TLAS instance multiplies instance
    /// count (and therefore TLAS traversal cost) by the zone count. Folding the
    /// zones into ONE prototype needs their slots adjacent so a single
    /// `material_base` addresses all of them.
    ///
    /// Dedup is on the WHOLE ordered tuple, not per material, so two placements
    /// of the same asset with the same zone materials share one run while a
    /// different colour variation gets its own — preserving the per-instance
    /// colour/emission variation that shared prototypes rely on.
    ///
    /// Unlike [`Self::reserve_palette`], every reserved slot records its
    /// spectral SPD: building zones are spectrally shaded and dropping the SPD
    /// would silently change their look.
    ///
    /// Panics if the three slices disagree in length, or if `keys` is empty.
    pub fn intern_run(
        &mut self,
        keys: &[MaterialKey],
        mats: &[PbrMaterial],
        spds: &[[f32; 16]],
    ) -> u32 {
        assert!(!keys.is_empty(), "intern_run needs at least one zone");
        assert!(
            keys.len() == mats.len() && keys.len() == spds.len(),
            "intern_run slice lengths disagree: {} keys, {} materials, {} spds",
            keys.len(),
            mats.len(),
            spds.len()
        );
        if let Some(&base) = self.run_lookup.get(keys) {
            return base;
        }
        let base = self.materials.len() as u32;
        for (mat, spd) in mats.iter().zip(spds.iter()) {
            // push_undeduped, inlined: a run MUST stay contiguous, so it can
            // never take the dedup shortcut that `intern` would.
            let s = self.materials.len() as u32;
            self.materials.push(*mat);
            self.spectral.push((s, *spd));
        }
        self.run_lookup.insert(keys.to_vec(), base);
        base
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

    // ---- intern_run: contiguous multi-zone prototype runs -----------------
    //
    // These two properties are what let a ~20-zone building prototype collapse
    // from 20 TLAS instances to 1. If either breaks, zones shade with the wrong
    // material rather than failing loudly, so they are pinned here.

    fn zone(i: u8) -> (MaterialKey, PbrMaterial, [f32; 16]) {
        let mat = PbrMaterial {
            base_color: [i as f32 / 32.0, 0.5, 0.25],
            ..PbrMaterial::default()
        };
        (key(&mat), mat, [i as f32; 16])
    }

    fn split(zones: &[(MaterialKey, PbrMaterial, [f32; 16])]) -> (Vec<MaterialKey>, Vec<PbrMaterial>, Vec<[f32; 16]>) {
        (
            zones.iter().map(|z| z.0).collect(),
            zones.iter().map(|z| z.1).collect(),
            zones.iter().map(|z| z.2).collect(),
        )
    }

    #[test]
    fn intern_run_slots_are_contiguous_so_one_material_base_addresses_every_zone() {
        let mut table = MaterialTable::with_capacity(8);
        table.reserve_opaque_fallback_slot();
        // An unrelated intern first, so the run cannot accidentally start at 0.
        let (k, m, s) = zone(99);
        table.intern(k, m, s);

        let zones: Vec<_> = (0..20).map(zone).collect();
        let (keys, mats, spds) = split(&zones);
        let base = table.intern_run(&keys, &mats, &spds);

        // `final = base + tri.material_id` must land on exactly this zone's
        // material for every zone.
        for (k, expected) in zones.iter().enumerate() {
            let slot = base as usize + k;
            assert_eq!(
                table.materials()[slot].base_color, expected.1.base_color,
                "zone {k} did not resolve at base+{k}"
            );
        }
        assert_eq!(table.len(), base + 20, "run must not leave gaps");
    }

    #[test]
    fn intern_run_records_a_spectral_spd_for_every_reserved_slot() {
        // reserve_palette deliberately records none; a building zone that lost
        // its SPD would shade non-spectrally and change look silently.
        let mut table = MaterialTable::with_capacity(4);
        let zones: Vec<_> = (0..5).map(zone).collect();
        let (keys, mats, spds) = split(&zones);
        let base = table.intern_run(&keys, &mats, &spds);

        for (k, expected) in zones.iter().enumerate() {
            let slot = base + k as u32;
            let found = table
                .spectral()
                .iter()
                .find(|(s, _)| *s == slot)
                .unwrap_or_else(|| panic!("zone {k} (slot {slot}) has no spectral entry"));
            assert_eq!(found.1, expected.2);
        }
    }

    #[test]
    fn intern_run_dedups_on_the_whole_tuple_but_keeps_colour_variations_apart() {
        let mut table = MaterialTable::with_capacity(8);
        let zones: Vec<_> = (0..6).map(zone).collect();
        let (keys, mats, spds) = split(&zones);

        let first = table.intern_run(&keys, &mats, &spds);
        let repeat = table.intern_run(&keys, &mats, &spds);
        assert_eq!(first, repeat, "same zone tuple must share one run");
        assert_eq!(table.len(), 6, "the repeat must not allocate again");

        // One zone recoloured = a different variation = its own run, so two
        // placements of the same asset can differ in colour.
        let mut varied = zones.clone();
        varied[3] = zone(31);
        let (vkeys, vmats, vspds) = split(&varied);
        let other = table.intern_run(&vkeys, &vmats, &vspds);
        assert_ne!(first, other, "a colour variation must get its own run");
        assert_eq!(table.len(), 12);
    }

    #[test]
    fn intern_run_ordering_is_part_of_run_identity() {
        // base+k indexes by POSITION, so a permuted tuple is a different run.
        let mut table = MaterialTable::with_capacity(8);
        let zones: Vec<_> = (0..4).map(zone).collect();
        let (keys, mats, spds) = split(&zones);
        let forward = table.intern_run(&keys, &mats, &spds);

        let mut rev = zones.clone();
        rev.reverse();
        let (rkeys, rmats, rspds) = split(&rev);
        let backward = table.intern_run(&rkeys, &rmats, &rspds);
        assert_ne!(forward, backward);
    }
}
