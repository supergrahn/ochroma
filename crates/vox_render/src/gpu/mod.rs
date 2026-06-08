pub mod adapter;
pub mod gpu_timing;
pub mod tile_range_build;
pub mod tiled_splat_renderer;
pub mod blend_skinning_compute;
pub mod radix_sort_pass;
pub mod compute_sort;
pub mod depth_prepass;
pub mod skinning_compute;
pub mod entity_buffer;
pub mod gpu_rasteriser;
pub mod instancing;
pub mod shadow_catcher;
pub mod sdf_shadow_pass;
pub mod software_rasteriser;
pub mod splat_raster;
pub mod wgpu_backend;
pub mod splat_buffer;
pub mod tile_assign;
pub mod gi_probe;
pub mod shadow_atlas;
pub mod volumetric_pass;
pub mod oit_pass;
pub mod bloom_pass;
pub mod dof_pass;
pub mod morph_compute;
pub mod splat_rt_gpu;
pub mod many_light_gpu;
pub mod hybrid_compose_gpu;
pub mod atom_budget_gpu;
pub mod relight_gpu;

use std::sync::Arc;

/// The single wgpu device + queue every render and compute module binds against.
///
/// Created once at surface bring-up (editor `resumed()` / engine present init) and
/// cloned — cheaply, since wgpu 24's [`wgpu::Device`]/[`wgpu::Queue`] are themselves
/// `Arc`-backed handles (`#[derive(Clone)]`) — into each pass constructor. Because
/// every pass binds buffers against the SAME device, a buffer produced by one pass
/// binds directly into the next with no CPU round-trip.
///
/// This is THE shared-device handle the GPU twins (`GpuGi`, `splat_rt_gpu`,
/// `many_light_gpu`, `hybrid_compose_gpu`, `atom_budget_gpu`, `relight_gpu`) take —
/// via their `new_with_context` constructors — instead of each owning its own device
/// through a private `Instance::new`. The standalone `new()` constructors are kept
/// intact for the CPU-oracle validation twins; `GpuContext` is added alongside them,
/// never replacing the validated own-device path.
///
/// `Arc<Device>`/`Arc<Queue>` (rather than bare clones) make the shared-ownership
/// contract explicit at every call site and give downstream residency/buffer-pool
/// work a device-lifetime handle to hang long-lived GPU resources off of. The clone
/// in [`GpuContext::from_parts`] is a refcount bump, NOT a second `request_device`.
#[derive(Clone)]
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_name: String,
    adapter_backend: wgpu::Backend,
}

impl GpuContext {
    /// Build from an already-created device/queue/adapter — the present path.
    ///
    /// `wgpu` 24's `Device`/`Queue` derive `Clone` (they are internally `Arc`-backed),
    /// so wrapping the borrowed handles in an `Arc` is a handle bump, not a device
    /// re-creation. The caller (e.g. the editor's `WgpuBackend`) keeps owning the
    /// originals; this `GpuContext` shares them.
    pub fn from_parts(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        info: &wgpu::AdapterInfo,
    ) -> Self {
        Self {
            device: Arc::new(device.clone()),
            queue: Arc::new(queue.clone()),
            adapter_name: info.name.clone(),
            adapter_backend: info.backend,
        }
    }

    /// The shared device. Returned by reference so pass constructors
    /// (`GpuGiPass::new(&device, …)`) can borrow it directly.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The shared queue, for `write_buffer`/`submit`.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Human-readable adapter name, for the shared-device honesty assertion and
    /// diagnostics (must equal the present backend's adapter name).
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The graphics backend (Vulkan/GL/…) backing the shared device — part of the
    /// device identity used by device-lost recovery and per-pass HUD labelling.
    pub fn adapter_backend(&self) -> wgpu::Backend {
        self.adapter_backend
    }
}
