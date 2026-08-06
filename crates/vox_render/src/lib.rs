pub mod animation;
pub mod atmosphere;
pub mod atom_budget;
pub mod atom_instances;
pub mod camera;
pub mod cine;
pub mod clas;
pub mod contact_decals;
pub mod dlss;
pub mod frustum;
pub mod gi_cache;
pub mod gpu;
pub mod hierarchical_lod;
pub mod hybrid_compose;
pub mod lighting;
pub mod lod;
pub mod lod_crossfade;
#[cfg(feature = "bevy")]
pub mod lod_ecs;
pub mod many_light;
pub mod material_compiler;
pub mod material_gpu_eval;
pub mod material_graph;
pub mod mega_geometry;
pub mod memory_pool;
pub mod mesh_detail;
pub mod mesh_program;
pub mod mesh_simplify;
pub mod naga_builder;
pub mod particles;
#[allow(unexpected_cfgs)]
pub mod perf_inspector;
pub mod postprocess;
pub mod profiling;
pub mod progressive_reveal;
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::ProgressiveRevealStateGpu;
#[cfg(feature = "bevy")]
pub mod render_ecs;
pub mod render_graph;
pub mod runtime_geometry;
pub mod scene_delta_adapter;
pub mod sdf_scene;
#[cfg(feature = "bevy")]
pub mod seq_ecs;
pub mod species_view;
pub mod spectra_bridge;
pub mod spectra_render;
pub mod spectral;
pub mod spectral_framebuffer;
pub mod spectral_shift;
pub mod spectral_tonemapper;
pub mod splat_rt;
pub mod streaming;
pub mod svt;
pub mod telemetry;
pub mod temporal;
pub mod upscaling;
pub mod vfx;
pub mod water;
pub mod web_renderer;
/// NVIDIA DLSS quality-mode spec, engine-side and always built (pure config). The
/// game picks a preset via `vox_render::DlssPreset`; the scale factors and the
/// even-dimension rule are owned here.
pub use dlss::DlssPreset;
pub mod animation_driver;
/// ONE BINARY: the acceleration backend is probed from the device at runtime
/// (NVIDIA -> OptiX, AMD/Intel -> Vulkan KHR, Apple -> Metal), never chosen by
/// a compile flag.
#[cfg(feature = "spectra-native")]
pub mod backend_select;
pub mod biome;
pub mod frame_debugger;
pub mod gi_baker;
pub mod gizmos;
pub mod gpu_particles;
pub mod importance;
pub mod level_streaming;
#[cfg(feature = "bevy")]
pub mod material_hotreload;
pub mod material_nodes;
#[cfg(feature = "bevy")]
pub mod particle_ecs;
pub mod pcg;
pub mod platform_profiles;
pub mod relight;
#[cfg(feature = "spectra-native")]
pub mod resident_renderer;
pub mod rigid_animation;
pub mod sequencer;
pub mod shadows;
pub mod skinning;
pub mod spectral_atmosphere;
pub mod spectral_gi;
pub mod spectral_response;
pub mod spectral_uplift;
pub mod spectral_viewport;
#[cfg(feature = "spectra-native")]
pub mod splat_backend;
pub mod splat_buffer_pool;
#[cfg(feature = "spectra-native")]
pub mod splat_convert;
pub mod splat_particles;
pub mod spline;
pub mod visual_effects;
pub mod world_partition;
#[cfg(feature = "spectra-native")]
pub use backend_select::{ResidentBackendKind, select_resident_backend};
#[cfg(feature = "spectra-native")]
pub use resident_renderer::{ResidentSceneRenderer, SceneSyncReport};
#[cfg(feature = "spectra-native")]
pub use spectra_renderer::{RendererTexture2D, RendererTextureMip};
// The native renderer consumes `spectra_scene_state::CameraLayer` (column-major
// view matrix + FOV); the old `spectra_renderer::CameraParams` type was removed.
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::CameraLayer as SpectraCameraParams;
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::GeometryStreamingLayer;
/// Re-export the native scene-state type so downstream crates (the game's live
/// frame seam) can name the type returned by the instanced-scene builder without
/// taking a direct dependency on `spectra-scene-state`.
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::SceneState;
/// The analytic terrain SDF layer (heightfield base + CSG feature list). Engine-
/// generic: a heightfield and CSG primitives carry no game concept, so this stays
/// on the engine side of the split while the game decides what to carve.
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::TerrainSdfLayer;
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::{GpuPageDescriptor, PAGE_FLAG_REQUIRED_BASE, PAGE_NONE};
#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::{MegaGeometryExecution, MegaGeometryLayer};

#[cfg(feature = "spectra-native")]
pub mod material_table;
pub mod render_ids;

pub mod render_runtime;
pub mod util;
pub use cine::{BokehShape, CameraSequence, CinePose, CinematicConfig, FocusMode};
#[cfg(feature = "spectra-native")]
pub use cine::{CINE_BASE_SEED, beauty_to_rgba8, render_cine_frame, write_png, write_png16};
pub use cine::{Channel, FrameRate, Interp, Key};
pub use cine::{CineFraming, KeyRef, eye_from_framing};
pub use hybrid_compose::RenderScene;
