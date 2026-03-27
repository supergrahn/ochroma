//! # Ochroma Engine
//!
//! A spectral Gaussian splatting game engine.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ochroma_engine::prelude::*;
//! ```
//!
//! ## Architecture
//!
//! - **Core**: Types, math, ECS components, spatial data structures
//! - **Data**: Asset formats (.vxm), procedural generation, materials
//! - **Render**: GPU rasterisation, spectral pipeline, post-processing
//! - **Sim**: Game simulation (if building a simulation game)
//! - **Audio**: Spatial audio with distance attenuation
//! - **Physics**: Rigid body simulation with AABB collision
//! - **Net**: Multiplayer networking with CRDT replication
//! - **Script**: Wasm mod runtime with event system

/// Core engine types, math, and ECS components.
pub mod core {
    pub use vox_core::*;
}

/// Asset formats, procedural generation, and materials.
pub mod data {
    pub use vox_data::*;
}

/// GPU rendering, spectral pipeline, and post-processing.
pub mod render {
    pub use vox_render::*;
}

/// Game simulation systems.
pub mod sim {
    pub use vox_sim::*;
}

/// Audio engine.
pub mod audio {
    pub use vox_audio::*;
}

/// Physics engine.
pub mod physics {
    pub use vox_physics::*;
}

/// Terrain system.
pub mod terrain {
    pub use vox_terrain::*;
}

/// UI framework.
pub mod ui {
    pub use vox_ui::*;
}

/// Networking.
pub mod net {
    pub use vox_net::*;
}

/// Scripting runtime.
pub mod script {
    pub use vox_script::*;
}

/// Neural/AI systems.
pub mod nn {
    pub use vox_nn::*;
}

/// Commonly used types re-exported for convenience.
pub mod prelude {
    // Core
    pub use vox_core::types::GaussianSplat;
    pub use vox_core::spectral::{SpectralBands, Illuminant};
    pub use vox_core::ecs::{SplatInstanceComponent, SplatAssetComponent, LodLevel};
    pub use vox_core::lwc::{WorldCoord, TileCoord};
    pub use vox_core::input::{GameAction, InputState, KeyBindings};
    pub use vox_core::game_loop::{GameClock, GamePhase};
    pub use vox_core::error::EngineError;
    pub use vox_core::undo::UndoStack;

    // Render
    pub use vox_render::spectral::RenderCamera;
    pub use vox_render::camera::CameraController;
    pub use vox_render::frustum::Frustum;
    pub use vox_render::lod::{LodLevel as RenderLodLevel, select_lod};
    pub use vox_render::particles::ParticleSystem;
    pub use vox_render::spectra_bridge::{QualityPreset, RenderConfig};

    // Data
    pub use vox_data::vxm::VxmFile;
    pub use vox_data::materials::MaterialLibrary;
    pub use vox_data::library::AssetLibrary;
    pub use vox_data::proc_gs_advanced::{generate_tree, generate_bench, generate_lamp_post};

    // Math (re-export glam)
    pub use glam::{Vec2, Vec3, Vec4, Mat4, Quat};
    pub use uuid::Uuid;
}
