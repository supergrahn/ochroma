//! `TerrainScene` — a single callable facade that composes the existing terrain
//! pieces into one pipeline a binary can drive directly:
//!
//! ```text
//! generate SDF TerrainVolume  ->  apply sculpt/brush deform  ->  volume_to_splats
//! ```
//!
//! Before this facade existed, the deform + splat path was reachable only from
//! per-module unit tests; a binary had to reach into `volume::sculpt`,
//! `deform`, `volume_to_splats` and the material table by hand. `TerrainScene`
//! owns the volume and the material palette and exposes the whole pipeline as a
//! handful of methods, so the runtime-deform + re-splat loop is live-wireable
//! without touching any binary.

use crate::deform;
use crate::foliage::{scatter_foliage, FoliageInstance, FoliageRule};
use crate::heightmap::Heightmap;
use crate::texture_paint::SplatMap;
use crate::volume::{
    default_volume_materials, sculpt, volume_to_splats, TerrainVolume, VolumeMaterial,
};
use vox_core::types::GaussianSplat;

/// Owns a `TerrainVolume`, its material palette, and the splat seed, and
/// exposes the generate -> deform -> splat pipeline as one object.
pub struct TerrainScene {
    volume: TerrainVolume,
    materials: Vec<VolumeMaterial>,
    /// Seed used for the (jittered) splat generation so re-splats are
    /// deterministic across deforms.
    splat_seed: u64,
    /// Optional per-texel material-weight splat map used for texture painting.
    /// `None` until [`TerrainScene::init_splat_map`] is called; this keeps the
    /// existing constructors and their behaviour unchanged for callers that
    /// never paint.
    splat_map: Option<SplatMap>,
}

impl TerrainScene {
    /// Create an empty scene: an all-air SDF volume plus the default material
    /// palette. Nothing is solid yet — call [`TerrainScene::add_ground_plane`]
    /// or one of the sculpt/deform helpers to introduce surface.
    pub fn new(size_x: usize, size_y: usize, size_z: usize, voxel_size: f32, splat_seed: u64) -> Self {
        Self {
            volume: TerrainVolume::new(size_x, size_y, size_z, voxel_size),
            materials: default_volume_materials(),
            splat_seed,
            splat_map: None,
        }
    }

    /// Convenience constructor: an empty volume seeded with a flat ground plane
    /// at `ground_height` of the given `material`. This is the minimal "world a
    /// binary would start from" before any interactive sculpting.
    pub fn with_ground(
        size_x: usize,
        size_y: usize,
        size_z: usize,
        voxel_size: f32,
        ground_height: f32,
        material: u8,
        splat_seed: u64,
    ) -> Self {
        let mut scene = Self::new(size_x, size_y, size_z, voxel_size, splat_seed);
        scene.add_ground_plane(ground_height, material);
        scene
    }

    /// Read-only access to the underlying SDF volume (for sampling / GPU upload).
    pub fn volume(&self) -> &TerrainVolume {
        &self.volume
    }

    /// Mutable access to the underlying SDF volume (for advanced callers).
    pub fn volume_mut(&mut self) -> &mut TerrainVolume {
        &mut self.volume
    }

    /// The material palette used when generating splats.
    pub fn materials(&self) -> &[VolumeMaterial] {
        &self.materials
    }

    /// The seed used for splat jitter.
    pub fn splat_seed(&self) -> u64 {
        self.splat_seed
    }

    // --- generation -------------------------------------------------------

    /// Add a flat solid ground plane at `height` of the given `material`.
    pub fn add_ground_plane(&mut self, height: f32, material: u8) {
        sculpt::add_ground_plane(&mut self.volume, height, material);
    }

    // --- deform / sculpt --------------------------------------------------

    /// Apply a spherical "fill" sculpt deform: make the region inside the sphere
    /// solid (the SDF at affected voxels decreases toward / below zero).
    ///
    /// Returns the number of voxels whose SDF value actually changed.
    pub fn sculpt_fill_sphere(&mut self, center: [f32; 3], radius: f32, material: u8) -> usize {
        let before = self.volume.data.clone();
        deform::fill_sphere(&mut self.volume, center, radius, material);
        count_changed(&before, &self.volume.data)
    }

    /// Apply a spherical "carve" sculpt deform: remove solid terrain inside the
    /// sphere (the SDF at affected voxels increases toward / above zero).
    ///
    /// Returns the number of voxels whose SDF value actually changed.
    pub fn sculpt_carve_sphere(&mut self, center: [f32; 3], radius: f32) -> usize {
        let before = self.volume.data.clone();
        deform::carve_sphere(&mut self.volume, center, radius);
        count_changed(&before, &self.volume.data)
    }

    /// Carve a capsule-shaped tunnel between two world points.
    ///
    /// Returns the number of voxels whose SDF value actually changed.
    pub fn sculpt_carve_tunnel(&mut self, start: [f32; 3], end: [f32; 3], radius: f32) -> usize {
        let before = self.volume.data.clone();
        deform::carve_tunnel(&mut self.volume, start, end, radius);
        count_changed(&before, &self.volume.data)
    }

    // --- splat extraction -------------------------------------------------

    /// Generate Gaussian surface splats from the current volume's surface
    /// voxels using the scene's material palette and seed. This is the final
    /// stage of the pipeline a renderer would upload.
    pub fn build_splats(&self) -> Vec<GaussianSplat> {
        volume_to_splats(&self.volume, &self.materials, self.splat_seed)
    }

    // --- foliage scatter --------------------------------------------------

    /// Extract a top-surface [`Heightmap`] from the current SDF volume.
    ///
    /// For each `(x, z)` voxel column the highest solid voxel (SDF <= 0) is
    /// found and its world-space top is recorded as the terrain height for that
    /// cell. Columns that are entirely air fall back to the volume's bottom edge
    /// (`origin.y`), so foliage rules with a positive `min_height` naturally
    /// reject them. The resulting heightmap shares the volume's horizontal
    /// origin and uses the volume's `voxel_size` as its cell size, so foliage
    /// positions come back in the same world frame as [`build_splats`].
    ///
    /// [`build_splats`]: TerrainScene::build_splats
    pub fn build_surface_heightmap(&self) -> Heightmap {
        let vol = &self.volume;
        let w = vol.size_x;
        let h = vol.size_z;
        let mut data = vec![vol.origin[1]; w * h];

        for z in 0..h {
            for x in 0..w {
                // Scan top-down for the highest solid voxel in this column.
                let mut top: Option<usize> = None;
                for y in (0..vol.size_y).rev() {
                    if vol.get(x, y, z) <= 0.0 {
                        top = Some(y);
                        break;
                    }
                }
                if let Some(ty) = top {
                    // World-space top face of that voxel.
                    let wy = vol.voxel_to_world(x, ty, z)[1] + vol.voxel_size * 0.5;
                    data[z * w + x] = wy;
                }
            }
        }

        let mut hm = Heightmap::from_data(w, h, data, vol.voxel_size);
        // Share the volume's horizontal world origin so foliage lands in the
        // same frame as the splats.
        hm.origin = [vol.origin[0], vol.origin[2]];
        hm
    }

    /// Scatter foliage across the current terrain surface.
    ///
    /// Derives a [`Heightmap`] from the live volume (via
    /// [`build_surface_heightmap`]) and runs the crate's
    /// [`scatter_foliage`](crate::foliage::scatter_foliage) over the supplied
    /// rules, honouring each rule's density, height band, slope limit, scale
    /// range and rotation. Returns the placed [`FoliageInstance`]s — the same
    /// list a binary would feed to the asset instancer. Deterministic for a
    /// given `seed` and surface.
    ///
    /// [`build_surface_heightmap`]: TerrainScene::build_surface_heightmap
    pub fn scatter_foliage(&self, rules: &[FoliageRule], seed: u64) -> Vec<FoliageInstance> {
        let hm = self.build_surface_heightmap();
        scatter_foliage(&hm, rules, seed)
    }

    // --- texture paint ----------------------------------------------------

    /// Create the scene's per-texel material [`SplatMap`] at the given
    /// resolution, replacing any existing one. Layers are added afterwards with
    /// [`add_material_layer`]; the first layer added becomes the fully-weighted
    /// base layer everywhere (matching [`SplatMap`] semantics).
    ///
    /// [`add_material_layer`]: TerrainScene::add_material_layer
    pub fn init_splat_map(&mut self, width: usize, height: usize) {
        self.splat_map = Some(SplatMap::new(width, height));
    }

    /// Add a material layer to the scene's splat map and return its layer index.
    ///
    /// Panics if [`init_splat_map`](TerrainScene::init_splat_map) has not been
    /// called yet — a splat map must exist before layers can be painted.
    pub fn add_material_layer(&mut self, name: &str, material: &str, spectral: [f32; 8]) -> usize {
        self.splat_map
            .as_mut()
            .expect("init_splat_map must be called before add_material_layer")
            .add_layer(name, material, spectral)
    }

    /// Paint a material `layer` into the splat map with a circular brush.
    ///
    /// `(texel_x, texel_z)` is the brush centre in splat-map texels, `strength`
    /// is the added (falloff-scaled) weight, and `radius` is the brush radius in
    /// texels. Weights are clamped and re-normalised per texel by the underlying
    /// [`SplatMap`]. No-op if no splat map has been initialised.
    pub fn paint_material(
        &mut self,
        texel_x: usize,
        texel_z: usize,
        layer: usize,
        strength: f32,
        radius: usize,
    ) {
        if let Some(map) = self.splat_map.as_mut() {
            map.paint(texel_x, texel_z, layer, strength, radius);
        }
    }

    /// Read-only access to the scene's splat map, if one has been initialised.
    pub fn splat_map(&self) -> Option<&SplatMap> {
        self.splat_map.as_ref()
    }

    /// The normalised material weight of `layer` at splat-map texel
    /// `(texel_x, texel_z)`, or `None` if no splat map exists or the layer index
    /// is out of range. This is the painted value a shader would blend with.
    pub fn material_weight_at(&self, texel_x: usize, texel_z: usize, layer: usize) -> Option<f32> {
        let map = self.splat_map.as_ref()?;
        if layer >= map.layer_count() {
            return None;
        }
        let x = texel_x.min(map.width.saturating_sub(1));
        let z = texel_z.min(map.height.saturating_sub(1));
        Some(map.weights[layer][z * map.width + x])
    }

    /// The blended spectral coefficients at splat-map texel `(texel_x,
    /// texel_z)`, or `None` if no splat map exists. Mirrors
    /// [`SplatMap::sample`].
    pub fn sample_material_spectral(&self, texel_x: usize, texel_z: usize) -> Option<[f32; 8]> {
        self.splat_map.as_ref().map(|m| m.sample(texel_x, texel_z))
    }
}

/// Count how many entries differ between two equal-length SDF buffers.
fn count_changed(before: &[f32], after: &[f32]) -> usize {
    debug_assert_eq!(before.len(), after.len());
    before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end pipeline test: build a volume with ground, apply a sphere
    /// sculpt at a known world location, and assert that BOTH (a) the SDF
    /// values changed at the expected voxels and (b) the resulting splat
    /// set changed accordingly. Every assertion checks a real computed value.
    #[test]
    fn fill_sculpt_changes_sdf_and_splats() {
        // 32^3 volume, 1 m voxels, ground plane just below the world origin so
        // the sphere we add pokes up above it and creates fresh surface.
        let mut scene = TerrainScene::with_ground(32, 32, 32, 1.0, -4.0, /*grass*/ 1, /*seed*/ 7);

        // --- BEFORE: snapshot SDF + splats ---
        let center = [3.0f32, 1.0, -2.0];
        let radius = 4.0f32;
        let (vx, vy, vz) = scene.volume().world_to_voxel(center[0], center[1], center[2]);
        let sdf_before = scene.volume().get(vx, vy, vz);
        let splats_before = scene.build_splats();
        let count_before = splats_before.len();

        // The center voxel sits above the ground plane, so before the fill it
        // must be air (SDF > 0). This guards the test against a no-op fill.
        assert!(
            sdf_before > 0.0,
            "center voxel must be air before fill, got SDF {sdf_before}"
        );

        // --- DEFORM: apply a sphere fill at a known location ---
        let changed = scene.sculpt_fill_sphere(center, radius, /*rock*/ 0);

        // (a) Assert the SDF actually changed at the expected voxel and that a
        //     non-trivial number of voxels were affected.
        let sdf_after = scene.volume().get(vx, vy, vz);
        assert!(
            sdf_after < sdf_before,
            "SDF at sphere center must decrease after fill: before {sdf_before}, after {sdf_after}"
        );
        assert!(
            sdf_after < 0.0,
            "sphere center must become solid (SDF < 0) after fill, got {sdf_after}"
        );

        // A radius-4 sphere at 1 m voxels touches well over a hundred voxels.
        assert!(
            changed > 100,
            "fill should change >100 voxels, only {changed} changed"
        );

        // The newly solid center voxel must carry the rock material we filled.
        assert_eq!(
            scene.volume().get_material(vx, vy, vz),
            0,
            "filled center voxel must be tagged with the rock material id"
        );

        // (b) Assert the resulting splat set changed accordingly. Adding a solid
        //     dome above flat ground introduces new surface voxels, so the
        //     splat count must strictly increase.
        let splats_after = scene.build_splats();
        let count_after = splats_after.len();
        assert!(
            count_after > count_before,
            "splat count must grow after adding a solid dome: before {count_before}, after {count_after}"
        );

        // And new splats must physically appear near the sculpt center —
        // i.e. the surface of the dome we just added. Verify at least one
        // splat lands within `radius + 1 voxel` of the sculpt center, and
        // that no such near-center splat existed beforehand on the flat
        // ground (the dome top is above the original ground surface).
        let near = |splats: &[GaussianSplat]| -> usize {
            splats
                .iter()
                .filter(|s| {
                    let p = s.position();
                    let dx = p[0] - center[0];
                    let dy = p[1] - center[1];
                    let dz = p[2] - center[2];
                    (dx * dx + dy * dy + dz * dz).sqrt() <= radius + 1.0
                })
                .count()
        };
        let near_before = near(&splats_before);
        let near_after = near(&splats_after);
        assert!(
            near_after > near_before,
            "more splats must appear near the sculpt center after fill: before {near_before}, after {near_after}"
        );

        // Finally, the highest splat in the scene must now sit at or above the
        // top of the dome — proof the new geometry, not just noise, drove the
        // splat change.
        let max_y_after = splats_after
            .iter()
            .map(|s| s.position()[1])
            .fold(f32::MIN, f32::max);
        let dome_top = center[1] + radius;
        assert!(
            max_y_after >= dome_top - scene.volume().voxel_size,
            "tallest splat ({max_y_after}) must reach near the dome top ({dome_top})"
        );
    }

    /// Carving into existing solid ground must raise the SDF (toward air) at the
    /// carve center and reduce the splat count is not guaranteed (carving opens
    /// new interior surfaces), but the SDF change and splat *delta* are real.
    #[test]
    fn carve_sculpt_raises_sdf_and_changes_splats() {
        // Solid block of ground filling the lower half of the volume.
        let mut scene = TerrainScene::with_ground(24, 24, 24, 1.0, 6.0, /*dirt*/ 2, /*seed*/ 11);

        let center = [0.0f32, 0.0, 0.0]; // well inside the solid ground
        let (vx, vy, vz) = scene.volume().world_to_voxel(center[0], center[1], center[2]);
        let sdf_before = scene.volume().get(vx, vy, vz);
        assert!(
            sdf_before < 0.0,
            "carve center must start solid (SDF < 0), got {sdf_before}"
        );

        let splats_before = scene.build_splats().len();

        let changed = scene.sculpt_carve_sphere(center, 3.0);
        assert!(changed > 50, "carve should change >50 voxels, only {changed}");

        let sdf_after = scene.volume().get(vx, vy, vz);
        assert!(
            sdf_after > sdf_before,
            "SDF at carve center must increase (toward air): before {sdf_before}, after {sdf_after}"
        );
        assert!(
            sdf_after > 0.0,
            "carve center must become air (SDF > 0), got {sdf_after}"
        );

        // The splat set must differ — carving an internal cavity exposes new
        // surface voxels, so the count must strictly increase from the flat slab.
        let splats_after = scene.build_splats().len();
        assert!(
            splats_after > splats_before,
            "carving a cavity must expose new surface splats: before {splats_before}, after {splats_after}"
        );
    }
}
