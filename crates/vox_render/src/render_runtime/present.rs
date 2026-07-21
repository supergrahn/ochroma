//! Engine-owned present/runtime facade for games.
//!
//! Games should depend on `vox_render`/Ochroma runtime contracts, not on Spectra
//! leaf crates. The concrete implementation is still Spectra, but that detail is
//! owned here so applications do not wire renderer internals directly.

pub use spectra_gpu::GpuTextureFormat;
#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
pub use spectra_gpu::SharedMetalBackend;
pub use spectra_gpu::{GpuBackend, SharedVulkanBackend, VulkanSlangBackend};
pub use spectra_present::{
    DeviceFrame, DevicePixelOrder, DeviceTemporalFrame, FrameGenerationKind,
    FrameGenerationRequest, FrameGenerationStatus, GpuUiAtlas, GpuUiLayer, GpuUiVertex,
    MetalPresentOptions, Present, PresentBackend, PresentChoice, PresentError, PresentFrame,
    PresentGradeParams, PresentKind, ReconstructionCamera, ReconstructionFrameV1,
    ReconstructionGuides, ReconstructionModeRequest, RrGuides, UpscalerKind,
    VulkanDeviceFrame, VulkanPixelOrder,
    VulkanPresentOptions, VulkanTemporalFrame, clear_frame_generation_request_override,
    frame_generation_request, select_present, select_present_offscreen_shared,
    select_present_offscreen_shared_with_options, select_present_rr, select_vulkan_present_shared,
    select_vulkan_present_shared_with_options, set_frame_generation_request,
};

#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
pub use spectra_present::{select_metal_present_shared, select_metal_present_shared_with_options};

#[cfg(feature = "spectra-native-optix")]
pub use spectra_present::{HeadlessRrOutput, headless_rr_eval};
