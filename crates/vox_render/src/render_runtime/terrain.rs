//! `TerrainUpload` — the engine-side, GPU-free terrain material payload.
//!
//! Plain data (no GPU / spectra types) describing a map's ground-cover spray
//! field + slope/height layered material + per-pixel curvature, resolved against
//! a built texture atlas (slots are `i32` atlas indices, `-1` = absent). A game
//! fills this once per map and hands it to [`crate::render_runtime::RenderRuntime::set_terrain`];
//! the runtime forwards each piece to the resident renderer. ALWAYS-BUILT (it
//! carries no renderer handles), so the game layer can construct it without the
//! `spectra-native` feature.

/// Terrain material upload payload, resolved against the built texture atlas.
/// `packed` empty => the renderer disables the field (single-slot ground path).
#[derive(Clone)]
pub struct TerrainUpload {
    /// 2 u32 per cell (the `SprayField::pack_u32()` bytes).
    pub packed: Vec<u32>,
    pub res: [u32; 2],
    pub origin: [f32; 2],
    pub cell_size: f32,
    /// 8 i32 per spray channel — MUST match the megakernel `SPRAY_CH_STRIDE` and
    /// its `g_spray_channels[c*8+k]` reads. Per-channel layout:
    ///   [0] base albedo   [1] base normal   [2] base rough   [3] base disp
    ///   [4] overlay albedo [5] overlay alpha [6] overlay normal [7] overlay disp
    /// (-1 = none). The game fills [1..3]/[4..7] only when its ground-detail
    /// toggles are on; otherwise they are -1 and the kernel skips that path.
    pub channel_slots: Vec<i32>,
    /// GROUND MACRO VARIATION: atlas slot of ONE global aerial-scale albedo the
    /// ground lerps toward with distance (keeps meso-scale patchiness resolving
    /// at aerial range where the ~1 m detail tile mips to a flat average). `-1`
    /// = off (byte-identical). `tile_m` = world metres per macro tile; `blend` =
    /// max lerp weight at full distance fade [0,1].
    pub ground_macro_slot: i32,
    pub ground_macro_tile_m: f32,
    pub ground_macro_blend: f32,
    /// SLOPE-LAYER atlas slots (#33): steep-face ROCK + transition DIRT
    /// albedo/normal, resolved from the map's `TerrainSurfaceSet.cliff/transition`.
    /// `-1` = absent (that layer rolls back into the biome ground in the kernel).
    pub slope_rock_albedo: i32,
    pub slope_rock_normal: i32,
    pub slope_dirt_albedo: i32,
    pub slope_dirt_normal: i32,
    /// Height-aware blend (Tier C): rock/dirt DISPLACEMENT atlas slots (-1 = none).
    pub slope_rock_disp: i32,
    pub slope_dirt_disp: i32,
    /// HIGH-ALTITUDE SNOW cap (alpine set): snow surface albedo/normal/disp atlas
    /// slots + the per-map snow line. `-1` slot / huge `snow_height` = snow OFF
    /// (lowland maps render byte-identical green ground).
    pub slope_snow_albedo: i32,
    pub slope_snow_normal: i32,
    pub slope_snow_disp: i32,
    pub slope_height_snow: f32,
    pub slope_height_snow_band: f32,
    /// PER-PIXEL CURVATURE grid baked from the heightmap (mean curvature, 1/m,
    /// row-major). Sampled bilinearly in the kernel for a smooth per-pixel
    /// curvature → no per-triangle scree squares. Empty = field off (legacy
    /// per-triangle κ, byte-identical). origin/cell_size/res match the heightmap.
    pub curvature_values: Vec<f32>,
    pub curvature_res: [u32; 2],
    pub curvature_origin: [f32; 2],
    pub curvature_cell_size: f32,
    /// WATER-DEPTH FIELD (P3, ROOT 4): per-cell STATIC water-column depth in metres
    /// (`max(0, sea_level - bed)`), row-major `iz*res_x+ix`. Sampled bilinearly in
    /// the megakernel at the ground hit XZ so the bed material can be blended by
    /// depth (wet_sand → shallow_mud → deep_silt) instead of reading flat Coastal
    /// sand. `0` = dry. Empty = field OFF (byte-identical: no underwater override,
    /// no wet band). origin/cell_size/res match the heightmap. v1 is the static
    /// initial-flood plane; dynamic WaterField ponds are a follow-up refit.
    pub depth_values: Vec<f32>,
    pub depth_res: [u32; 2],
    pub depth_origin: [f32; 2],
    pub depth_cell_size: f32,
    /// World-Y of the static sea plane (drives the intertidal WET BAND above the
    /// waterline). Only consulted when `depth_values` is non-empty.
    pub depth_sea_level: f32,
    /// WATER FLOW FIELD: live `WaterField` per-cell velocity `(vel_x, vel_z)` in
    /// m/s, interleaved 2 f32/cell (`[vx0,vz0, vx1,vz1, ...]`), row-major
    /// `iz*res_x+ix` on the SAME grid as `depth_values` (origin/cell_size/res reuse
    /// the depth-field uniforms — no separate grid). Drives flow-aligned wave scroll
    /// + white-water on the MAT_WATER surface, so rivers actually flow. Empty = flow
    /// OFF (byte-identical to still procedural ripples). Length must be
    /// `2 * depth_res[0] * depth_res[1]` when non-empty.
    pub flow_values: Vec<f32>,
    /// UNDERWATER BED atlas slots (P3): the submerged bed materials the megakernel
    /// blends by `depth_values` — `wet_sand` (~0 m) → `shallow_mud` (~2 m) →
    /// `deep_silt` (~8 m+), plus `riverbed` (driven by flow, follow-up). Each is an
    /// (albedo, normal, roughness) atlas-slot triple resolved against the SAME atlas
    /// as the meshes (`-1` = absent → that bed rolls back to the next-shallower one,
    /// finally to the spray ground → byte-identical). Set via `set_underwater_layers`.
    pub uw_wet_albedo: i32,
    pub uw_wet_normal: i32,
    pub uw_wet_rough: i32,
    pub uw_mud_albedo: i32,
    pub uw_mud_normal: i32,
    pub uw_mud_rough: i32,
    pub uw_silt_albedo: i32,
    pub uw_silt_normal: i32,
    pub uw_silt_rough: i32,
    pub uw_bed_albedo: i32,
    pub uw_bed_normal: i32,
    pub uw_bed_rough: i32,
}

impl Default for TerrainUpload {
    fn default() -> Self {
        // CONFIG-FIRST: the slope-layer + snow-cap fallback slots/heights are the
        // runtime config blocks `slope_layers` / `terrain` (`config/ochroma.ron`).
        // The shipped defaults equal the old literals (slope slots -1, snow slots
        // -1, snow line 1e9, band 120) so a snow-OFF lowland map renders byte-
        // identically. The map's `TerrainSurfaceSet` still overrides these at load.
        // Slope slots default to -1 (absent) — NOT 0, which is a valid atlas slot
        // that would wrongly fire the rock blend on a field-less ground.
        let cfg = vox_config::config();
        Self {
            packed: Vec::new(),
            res: [0, 0],
            origin: [0.0, 0.0],
            cell_size: 0.0,
            channel_slots: Vec::new(),
            // Macro layer OFF by default (-1 slot / 0 blend) — byte-identical for
            // any game that does not fill it.
            ground_macro_slot: -1,
            ground_macro_tile_m: 30.0,
            ground_macro_blend: 0.0,
            slope_rock_albedo: cfg.slope_layers.rock_albedo_slot,
            slope_rock_normal: cfg.slope_layers.rock_normal_slot,
            slope_dirt_albedo: cfg.slope_layers.dirt_albedo_slot,
            slope_dirt_normal: cfg.slope_layers.dirt_normal_slot,
            slope_rock_disp: cfg.slope_layers.rock_disp_slot,
            slope_dirt_disp: cfg.slope_layers.dirt_disp_slot,
            // Snow OFF by default: -1 slot + a huge snow line so the cap never fires.
            slope_snow_albedo: cfg.terrain.snow_albedo_slot,
            slope_snow_normal: cfg.terrain.snow_normal_slot,
            slope_snow_disp: cfg.terrain.snow_disp_slot,
            slope_height_snow: cfg.terrain.snow_line_m,
            slope_height_snow_band: cfg.terrain.snow_band_m,
            curvature_values: Vec::new(),
            curvature_res: [0, 0],
            curvature_origin: [0.0, 0.0],
            curvature_cell_size: 0.0,
            // Depth field OFF by default: empty values + a sentinel sea plane, so a
            // map that does not fill it renders byte-identically (no underwater
            // override, no wet band).
            depth_values: Vec::new(),
            depth_res: [0, 0],
            depth_origin: [0.0, 0.0],
            depth_cell_size: 0.0,
            depth_sea_level: 0.0,
            // Flow field OFF by default (empty → still procedural ripples,
            // byte-identical to a map that does not fill it).
            flow_values: Vec::new(),
            // Underwater bed slots default to -1 (absent) — NOT 0, which is a valid
            // atlas slot that would wrongly fire the bed blend on a field-less ground.
            uw_wet_albedo: -1,
            uw_wet_normal: -1,
            uw_wet_rough: -1,
            uw_mud_albedo: -1,
            uw_mud_normal: -1,
            uw_mud_rough: -1,
            uw_silt_albedo: -1,
            uw_silt_normal: -1,
            uw_silt_rough: -1,
            uw_bed_albedo: -1,
            uw_bed_normal: -1,
            uw_bed_rough: -1,
        }
    }
}
