mod demo_asset;
pub mod avatar_motion;
pub mod character_controller;
pub mod ai_fsm;
pub mod building_placement;
pub mod autosave;
pub mod debug_console;
pub mod content_browser;
pub mod editor;
pub mod shell;
// render_to_file renders through the banned CPU software rasterizer; it is a
// legacy offline utility (PPM/turntable). Gated OFF by default so the lib (and
// thus the editor/game) compiles with no rasterizer in its graph.
#[cfg(feature = "legacy-raster")]
pub mod render_to_file;
pub mod screenshot;
pub mod shortcut_help;
pub mod daytime;
pub mod growth;
pub mod headless;
pub mod overlays;
pub mod persistence;
pub mod placement;
pub mod road_builder;
pub mod simulation;
pub mod steam;
pub mod systems;
pub mod terrain_setup;
pub mod ui;
pub mod undo_integration;
pub mod notifications;
pub mod minimap;
pub mod settings;
pub mod soundscape;
pub mod tutorial;
pub mod terrain_editor;
pub mod walk_animation;
