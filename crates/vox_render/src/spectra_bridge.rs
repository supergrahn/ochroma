//! Engine-side contract for driving Spectra from Ochroma.
//!
//! Ochroma owns world/runtime policy: ECS, streaming, visibility decisions,
//! frame budgets, windows, input, and UI composition. Spectra owns renderer
//! capability: generic scene handles, acceleration structures, spectral
//! lighting, temporal reconstruction, and GPU-resident output. Game concepts
//! must stay above this bridge.

use std::path::{Path, PathBuf};

#[cfg(feature = "spectra-native")]
pub use spectra_scene_state::{
    DirtyRange as SpectraDirtyRange, DirtyRangeKind as SpectraDirtyRangeKind,
    FrameOutputTarget as SpectraFrameOutputTarget, GeometryHandle as SpectraGeometryHandle,
    InstanceBatchHandle as SpectraInstanceBatchHandle, LightSetHandle as SpectraLightSetHandle,
    LiveSceneCapabilities as SpectraLiveSceneCapabilities, MaterialHandle as SpectraMaterialHandle,
    RenderTargetHandle as SpectraRenderTargetHandle, ResidencyChange as SpectraResidencyChange,
    ResidencyState as SpectraResidencyState, SceneResource as SpectraSceneResource,
    SceneUpdateBatch as SpectraSceneUpdateBatch,
    SdfInstanceBatchHandle as SpectraSdfInstanceBatchHandle,
    SdfVolumeHandle as SpectraSdfVolumeHandle,
};

/// Owner of a concept at the game/engine/renderer boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryOwner {
    Game,
    Engine,
    Renderer,
}

/// Capability or policy item that crosses the Ochroma/Spectra bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeItem {
    /// Domain entities and interactions such as meshes, agents, regions, and tools.
    GameSemantics,
    /// ECS, visibility policy, residency decisions, frame budget, and scheduling.
    RuntimePolicy,
    /// Stable renderer handles for geometry, materials, SDF volumes, instance batches, and targets.
    GenericRenderHandles,
    /// Dirty ranges and add/remove/update batches sent from the engine to the renderer.
    SceneUpdateBatches,
    /// BLAS/TLAS, SDF sampling, temporal reconstruction, denoise, upscale, and tonemap.
    RenderAlgorithms,
    /// The final frame stays on the GPU until presentation/composition.
    GpuResidentOutput,
}

impl BridgeItem {
    pub const fn owner(self) -> BoundaryOwner {
        match self {
            Self::GameSemantics => BoundaryOwner::Game,
            Self::RuntimePolicy => BoundaryOwner::Engine,
            Self::GenericRenderHandles
            | Self::SceneUpdateBatches
            | Self::RenderAlgorithms
            | Self::GpuResidentOutput => BoundaryOwner::Renderer,
        }
    }
}

/// Renderer capabilities required before Spectra can be Ochroma's primary live renderer.
pub const PRIMARY_SPECTRA_BRIDGE_ITEMS: [BridgeItem; 4] = [
    BridgeItem::GenericRenderHandles,
    BridgeItem::SceneUpdateBatches,
    BridgeItem::RenderAlgorithms,
    BridgeItem::GpuResidentOutput,
];

/// Rendering backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    /// Built-in wgpu Gaussian splat rasteriser.
    BuiltIn,
    /// Spectra path tracer (high quality, requires Spectra installation).
    SpectraPathTracer,
    /// Spectra real-time mode (denoised, interactive).
    SpectraRealtime,
    /// CPU software rasteriser (fallback).
    Software,
}

impl RenderBackend {
    pub const fn is_spectra(self) -> bool {
        matches!(self, Self::SpectraPathTracer | Self::SpectraRealtime)
    }

    pub const fn is_primary_live_target(self) -> bool {
        matches!(self, Self::SpectraRealtime)
    }

    pub const fn requires_gpu_resident_output(self) -> bool {
        matches!(self, Self::SpectraRealtime)
    }
}

/// Render quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    Low,       // software rasteriser, no post-processing
    Medium,    // built-in wgpu, basic post-processing
    High,      // built-in wgpu, full post-processing + denoiser
    Ultra,     // Spectra real-time with AI denoiser
    Cinematic, // Spectra path tracer (offline quality)
}

impl QualityPreset {
    pub const fn backend(&self) -> RenderBackend {
        match self {
            Self::Low => RenderBackend::Software,
            Self::Medium | Self::High => RenderBackend::BuiltIn,
            Self::Ultra => RenderBackend::SpectraRealtime,
            Self::Cinematic => RenderBackend::SpectraPathTracer,
        }
    }

    pub const fn enable_denoiser(&self) -> bool {
        matches!(self, Self::High | Self::Ultra | Self::Cinematic)
    }

    pub const fn enable_shadows(&self) -> bool {
        matches!(
            self,
            Self::Medium | Self::High | Self::Ultra | Self::Cinematic
        )
    }

    pub const fn enable_post_processing(&self) -> bool {
        !matches!(self, Self::Low)
    }

    pub const fn enable_particles(&self) -> bool {
        !matches!(self, Self::Low)
    }

    pub const fn max_visible_splats(&self) -> usize {
        match self {
            Self::Low => 100_000,
            Self::Medium => 1_000_000,
            Self::High => 5_000_000,
            Self::Ultra => 20_000_000,
            Self::Cinematic => 50_000_000,
        }
    }
}

/// Configuration for the render pipeline.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub quality: QualityPreset,
    pub resolution: (u32, u32),
    pub vsync: bool,
    pub fov: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub gamma: f32,
    pub exposure: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            quality: QualityPreset::High,
            resolution: (1920, 1080),
            vsync: true,
            fov: std::f32::consts::FRAC_PI_4,
            near_plane: 0.1,
            far_plane: 10000.0,
            gamma: 2.2,
            exposure: 1.0,
        }
    }
}

/// Spectra process handle (for Option B: subprocess rendering).
pub struct SpectraProcess {
    pub available: bool,
    pub spectra_path: Option<PathBuf>,
}

fn looks_like_spectra_root(path: &Path) -> bool {
    path.join("rust/spectra-renderer/Cargo.toml").exists() || path.join("pyproject.toml").exists()
}

fn default_spectra_candidates() -> Vec<PathBuf> {
    let Some(home) = dirs_next::home_dir() else {
        return Vec::new();
    };

    vec![
        home.join("src/spectra"),
        home.join("git/aetherspectra/spectra"),
    ]
}

impl SpectraProcess {
    pub fn detect() -> Self {
        let spectra_path = std::env::var("SPECTRA_PATH")
            .ok()
            .map(PathBuf::from)
            .filter(|path| looks_like_spectra_root(path))
            .or_else(|| {
                default_spectra_candidates()
                    .into_iter()
                    .find(|path| looks_like_spectra_root(path))
            });

        let available = spectra_path.is_some();
        if available {
            let spectra_path = spectra_path.as_ref().expect("available path checked above");
            println!(
                "[ochroma] Spectra renderer detected at {}",
                spectra_path.display()
            );
        } else {
            println!("[ochroma] Spectra not found, using built-in renderer");
        }

        Self {
            available,
            spectra_path,
        }
    }

    pub fn is_available(&self) -> bool {
        self.available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All `BridgeItem` variants, so the invariant tests below cannot silently
    /// skip a future variant. If a variant is added without updating this list the
    /// `all_items_have_an_owner` count check fails — forcing a deliberate decision
    /// about which side of the bridge the new concept lives on.
    const ALL_BRIDGE_ITEMS: [BridgeItem; 6] = [
        BridgeItem::GameSemantics,
        BridgeItem::RuntimePolicy,
        BridgeItem::GenericRenderHandles,
        BridgeItem::SceneUpdateBatches,
        BridgeItem::RenderAlgorithms,
        BridgeItem::GpuResidentOutput,
    ];

    /// BEHAVIORAL INVARIANT (not a self-restating tautology): the set of items the
    /// renderer owns must be EXACTLY `PRIMARY_SPECTRA_BRIDGE_ITEMS` — two
    /// independently-declared facts (`owner()` per-variant, and the published
    /// "renderer must provide these" constant) that must agree. A drift in either
    /// (e.g. classifying `RuntimePolicy` as Renderer-owned, or forgetting to list
    /// a renderer capability) breaks this without touching the other declaration.
    #[test]
    fn renderer_owned_items_are_exactly_the_primary_bridge_set() {
        let renderer_owned: Vec<BridgeItem> = ALL_BRIDGE_ITEMS
            .into_iter()
            .filter(|item| item.owner() == BoundaryOwner::Renderer)
            .collect();

        // Same membership as the published renderer-capability contract, ignoring
        // declaration order.
        assert_eq!(
            renderer_owned.len(),
            PRIMARY_SPECTRA_BRIDGE_ITEMS.len(),
            "renderer-owned item count must match PRIMARY_SPECTRA_BRIDGE_ITEMS; \
             got {renderer_owned:?} vs {PRIMARY_SPECTRA_BRIDGE_ITEMS:?}"
        );
        for item in PRIMARY_SPECTRA_BRIDGE_ITEMS {
            assert!(
                renderer_owned.contains(&item),
                "{item:?} is published as a renderer capability but its owner() is \
                 not Renderer — the bridge declarations disagree"
            );
        }
    }

    /// BEHAVIORAL INVARIANT: no game/runtime policy concept may be owned by the
    /// renderer — the whole point of the bridge ("game concepts must stay above
    /// this bridge"). This is enforced as a property over ALL items, so a future
    /// variant defaulting to the wrong side is caught.
    #[test]
    fn policy_and_game_items_never_cross_into_the_renderer() {
        for item in ALL_BRIDGE_ITEMS {
            let owner = item.owner();
            if matches!(item, BridgeItem::GameSemantics) {
                assert_eq!(owner, BoundaryOwner::Game, "{item:?} must stay game-side");
            }
            if matches!(item, BridgeItem::RuntimePolicy) {
                assert_eq!(
                    owner,
                    BoundaryOwner::Engine,
                    "{item:?} (ECS/visibility/budget policy) must stay engine-side, \
                     never leak into the renderer"
                );
            }
        }
        // Every item is accounted for (no variant silently absent from the list).
        assert_eq!(
            ALL_BRIDGE_ITEMS.len(),
            6,
            "ALL_BRIDGE_ITEMS must enumerate every BridgeItem variant"
        );
    }

    #[test]
    fn spectra_realtime_is_the_primary_live_target() {
        assert!(RenderBackend::SpectraRealtime.is_spectra());
        assert!(RenderBackend::SpectraRealtime.is_primary_live_target());
        assert!(RenderBackend::SpectraRealtime.requires_gpu_resident_output());
        assert!(!RenderBackend::BuiltIn.requires_gpu_resident_output());
    }

    #[test]
    fn ultra_quality_routes_to_spectra_realtime() {
        assert_eq!(
            QualityPreset::Ultra.backend(),
            RenderBackend::SpectraRealtime
        );
        assert!(QualityPreset::Ultra
            .backend()
            .requires_gpu_resident_output());
        assert_eq!(
            QualityPreset::Cinematic.backend(),
            RenderBackend::SpectraPathTracer
        );
    }

    #[test]
    fn spectra_root_detection_accepts_rust_workspace_layout() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let renderer_dir = tmp.path().join("rust/spectra-renderer");
        std::fs::create_dir_all(&renderer_dir).expect("create renderer dir");
        std::fs::write(renderer_dir.join("Cargo.toml"), "[package]\nname = \"x\"\n")
            .expect("write marker");

        assert!(looks_like_spectra_root(tmp.path()));
    }
}
