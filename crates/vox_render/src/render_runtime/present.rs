//! Engine-owned present/runtime facade for games.
//!
//! Games should depend on `vox_render`/Ochroma runtime contracts, not on Spectra
//! leaf crates. The concrete implementation is still Spectra, but that detail is
//! owned here so applications do not wire renderer internals directly.

pub use spectra_gpu::GpuTextureFormat;
pub use spectra_gpu::VulkanSlangBackend;
pub use spectra_present::{
    clear_frame_generation_request_override, frame_generation_request, select_present,
    select_present_rr, set_frame_generation_request, FrameGenerationKind, FrameGenerationRequest,
    FrameGenerationStatus, GpuUiAtlas, GpuUiLayer, GpuUiVertex, Present, PresentBackend,
    PresentChoice, PresentError, PresentFrame, PresentGradeParams, PresentKind, RrGuides,
    UpscalerKind,
};

#[cfg(feature = "spectra-native-optix")]
pub use spectra_present::{headless_rr_eval, HeadlessRrOutput};
