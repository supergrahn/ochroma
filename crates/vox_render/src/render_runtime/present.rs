//! Engine-owned present/runtime facade for games.
//!
//! Games should depend on `vox_render`/Ochroma runtime contracts, not on Spectra
//! leaf crates. The concrete implementation is still Spectra, but that detail is
//! owned here so applications do not wire renderer internals directly.

pub use spectra_gpu::GpuTextureFormat;
pub use spectra_gpu::{GpuBackend, SharedVulkanBackend, VulkanSlangBackend};
#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
pub use spectra_gpu::SharedMetalBackend;
pub use spectra_present::{
    clear_frame_generation_request_override, frame_generation_request, select_present,
    select_present_offscreen_shared, select_present_offscreen_shared_with_options,
    select_present_rr, select_vulkan_present_shared, select_vulkan_present_shared_with_options,
    set_frame_generation_request, FrameGenerationKind, FrameGenerationRequest,
    DeviceFrame, DevicePixelOrder, DeviceTemporalFrame, FrameGenerationStatus, GpuUiAtlas,
    GpuUiLayer, GpuUiVertex, MetalPresentOptions, Present, PresentBackend, PresentChoice,
    PresentError, PresentFrame, PresentGradeParams, PresentKind, RrGuides, UpscalerKind,
    VulkanDeviceFrame, VulkanPixelOrder, VulkanPresentOptions, VulkanTemporalFrame,
};

#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
pub use spectra_present::{
    select_metal_present_shared, select_metal_present_shared_with_options,
};

#[cfg(feature = "spectra-native-optix")]
pub use spectra_present::{headless_rr_eval, HeadlessRrOutput};
