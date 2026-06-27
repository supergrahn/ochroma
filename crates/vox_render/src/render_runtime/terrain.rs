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
    /// 4 i32 per spray channel `[albedo, normal, rough, disp]` atlas slot.
    pub channel_slots: Vec<i32>,
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
        }
    }
}
