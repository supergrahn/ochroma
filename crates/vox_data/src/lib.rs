pub mod asset_catalog;
pub mod creator_tools;
pub mod forge_ecs;
pub mod gi_export;
pub mod gltf_animation;
pub mod gltf_import;
pub mod hot_reload;
pub mod library;
pub mod map_file;
pub mod marketplace;
pub mod materials;
pub mod mega_geometry;
pub mod neural_compress;
pub mod osm_import;
pub mod ply_loader;
pub mod proc_gs;
pub mod proc_gs_advanced;
pub mod save;
pub mod scene_format;
pub mod scene_serialize;
pub mod splat_codec;
pub mod spz;
pub mod templates;
pub mod tile_streamer;
pub mod vxm;
pub mod vxm_v2;
/// The `.vxp` (vox pack) shipping asset container: a deterministic STORE-only
/// ZIP of content-addressed entries with BINARY geometry.
pub mod vxp;
pub mod world_save;
pub use splat_codec::{from_saved_geom, to_saved_geom};
pub mod import_helpers;
pub mod import_pipeline;
pub mod material_system;
pub mod prefab;
pub mod spectral_codec;
pub mod spectral_upsampler;
pub use spectral_upsampler::{SpectralMaterial, SpectralMaterialDb, SpectralUpsampler};
pub mod spectral_capture;
pub use spectral_capture::{LightSpd, SpectralMaterialProfile};
pub use vxm::VxmFileV3;
pub mod colmap_pipeline;
pub use colmap_pipeline::{ColmapError, ColmapPipeline, ColmapPoint};
pub use import_pipeline::{ImportResult, ImportSettings, import_asset};
pub mod asset_validate;
pub use asset_validate::{
    Severity, ValidationBudget, ValidationIssue, ValidationReport, validate_splats,
};
pub mod vegetation_splatizer;
pub use vegetation_splatizer::{backproject_pca, splatize_vegetation_mesh};
pub mod terrain_splatizer;
pub use terrain_splatizer::{
    BiomeKind, SpectralTerrainMaterials, biome_to_splat_weights, blend_spectral_terrain,
};
