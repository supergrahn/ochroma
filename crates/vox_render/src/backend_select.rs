//! ONE BINARY, and the acceleration backend is chosen FROM THE DEVICE at
//! runtime — never from a compile flag.
//!
//! # What was wrong
//!
//! `resident_renderer.rs` used to name its backend with a `type` alias selected
//! by `target_os` + cargo feature:
//!
//! ```ignore
//! #[cfg(all(target_os = "windows", feature = "spectra-native-optix"))]
//! type ResidentBackend = CudarcSlangBackend;   // ...else SharedVulkanBackend
//! ```
//!
//! That is a decision taken before the binary has ever seen the player's GPU.
//! A Windows build carrying the OptiX feature demanded CUDA on an AMD or Intel
//! machine; a build without it left an RTX card on Vulkan. Neither can be fixed
//! by the player, and shipping two binaries is not a fix — it is the same bug
//! with a download page in front of it.
//!
//! # Why the stakes are higher than "a bit slower"
//!
//! The software-BVH fallback **silently drops every instanced prototype**. A
//! measured OptiX-off frame on the box had no buildings and no trees at all. So
//! hardware traversal is not an optimisation here; it is the difference between
//! rendering the scene and rendering an empty plane. That is why every path in
//! this module ANNOUNCES what it chose and why, and why a fallback to a backend
//! without hardware traversal is a loud, unmissable warning rather than a
//! quietly wrong frame.
//!
//! # The choice
//!
//! | device                   | backend                       | traversal      |
//! |--------------------------|-------------------------------|----------------|
//! | NVIDIA, OptiX usable     | `CudarcSlangBackend` + OptiX  | OptiX RT cores |
//! | AMD / Intel (and NVIDIA  | `SharedVulkanBackend`         | `VK_KHR_ray_query` |
//! | without a usable OptiX)  |                               | per-proto BLAS + instanced TLAS |
//! | Apple                    | `SharedMetalBackend`          | Metal ray tracing |
//!
//! Probe order is deliberate: OptiX first, because when it is available it is
//! the only path with a cluster-AS primitive, and because `spectra_renderer::
//! optix_hw_rt_usable()` is a cheap library/symbol probe that needs no CUDA
//! context and answers `false` instantly on every non-NVIDIA machine.
//!
//! NOTE the NVIDIA-without-OptiX case lands on Vulkan KHR, not on CUDA. CUDA
//! without OptiX means the software BVH — i.e. no buildings — so Vulkan's
//! hardware traversal is strictly the better answer there.

#![cfg(feature = "spectra-native")]

use spectra_gpu::GpuBackend;

/// Which acceleration backend this run is actually using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentBackendKind {
    /// NVIDIA: CUDA compute + OptiX hardware traversal (RT cores), the only
    /// path with hardware cluster-AS build (CLAS), OMM and SER.
    OptixCuda,
    /// AMD / Intel / NVIDIA-without-OptiX: Vulkan compute + `VK_KHR_ray_query`
    /// hardware traversal. This IS the OptiX alternative.
    VulkanKhr,
    /// Apple: Metal compute + Metal hardware ray tracing.
    Metal,
}

impl ResidentBackendKind {
    pub fn label(self) -> &'static str {
        match self {
            ResidentBackendKind::OptixCuda => "OptiX/CUDA",
            ResidentBackendKind::VulkanKhr => "Vulkan KHR",
            ResidentBackendKind::Metal => "Metal",
        }
    }
}

/// The chosen backend, boxed for the renderer plus the un-boxed handle for the
/// device-sharing paths that genuinely need the concrete type.
///
/// WHY BOTH. The path tracer only ever needs `GpuBackend`, so `gpu` is enough
/// for `Renderer<Box<dyn GpuBackend>>`. But the same-device present path hands
/// its own `VkDevice`/`MTLDevice` to the swapchain so a frame is presented
/// without a host round-trip — that is real Vulkan/Metal interop and cannot be
/// expressed through a backend-neutral trait. These handles are CLONES of the
/// very object inside `gpu` (both are `Arc`-backed), never a second device: a
/// second device would present a black frame from a buffer the presenter cannot
/// see. They are `None` when that backend is not what got chosen, so a caller
/// must handle the mismatch rather than receive a wrong device.
pub struct SelectedBackend {
    pub gpu: Box<dyn GpuBackend>,
    pub kind: ResidentBackendKind,
    /// The same Vulkan backend, un-boxed — `Some` iff [`ResidentBackendKind::VulkanKhr`].
    pub vulkan: Option<spectra_gpu::SharedVulkanBackend>,
    /// The same Metal backend, un-boxed — `Some` iff [`ResidentBackendKind::Metal`].
    #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
    pub metal: Option<spectra_gpu::SharedMetalBackend>,
}

/// How the backend got picked. Printed so a support log can tell a genuine
/// hardware answer from an operator override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// Probed the device.
    Hardware,
    /// `SPECTRA_BACKEND` named it explicitly.
    EnvOverride,
}

/// Probe the device and build the acceleration backend it should run.
///
/// `SPECTRA_BACKEND` (`cuda`/`optix`, `vulkan`/`vk`, `metal`, `auto`) forces a
/// choice for A/B diagnosis. A forced choice that fails is a hard error — it is
/// never silently substituted, because a silent substitution is exactly how a
/// frame ends up on the software BVH with nobody knowing.
///
/// Returns the boxed backend plus what was chosen. Boxing is what makes this
/// possible at all: `Renderer<G>` is generic over `G: GpuBackend`, so one
/// renderer type means one `G` — and `Box<dyn GpuBackend>` is itself a
/// `GpuBackend` (`spectra-gpu/src/dyn_backend.rs`, which forwards all 50 trait
/// methods including the 31 whose defaults would otherwise silently degrade a
/// capability to "unsupported").
pub fn select_resident_backend() -> Result<SelectedBackend, String> {
    let forced_raw = std::env::var("SPECTRA_BACKEND").ok();
    let forced = forced_raw.as_deref().map(str::to_ascii_lowercase);
    let forced = match forced.as_deref() {
        None | Some("") | Some("auto") => None,
        other => other,
    };

    let (selected, source) = match forced {
        Some("cuda") | Some("optix") => {
            if !spectra_renderer::optix_hw_rt_usable() {
                eprintln!(
                    "[backend] WARNING SPECTRA_BACKEND={} forces CUDA, but OptiX is NOT usable in \
                     this binary on this machine — traversal will fall back to the SOFTWARE BVH, \
                     which DROPS every instanced prototype (no buildings, no trees).",
                    forced_raw.as_deref().unwrap_or("cuda")
                );
            }
            (new_cuda()?, Source::EnvOverride)
        }
        Some("vulkan") | Some("vk") => (new_vulkan()?, Source::EnvOverride),
        Some("metal") => (new_metal()?, Source::EnvOverride),
        Some(other) => {
            return Err(format!(
                "SPECTRA_BACKEND={other:?} is not a backend (expected auto|cuda|optix|vulkan|vk|metal)"
            ));
        }
        None => (auto_select()?, Source::Hardware),
    };

    announce(&selected, source);
    Ok(selected)
}

/// The hardware-driven path. Every arm states its evidence.
fn auto_select() -> Result<SelectedBackend, String> {
    // 1. NVIDIA with a usable OptiX. The probe is a library/symbol check, not a
    //    vendor-name guess and not a compile flag: it answers false on every
    //    non-NVIDIA device and on an NVIDIA device whose driver has no real
    //    nvoptix, without needing a CUDA context first.
    if spectra_renderer::optix_hw_rt_usable() {
        eprintln!(
            "[backend] probe: OptiX hardware traversal is usable on this device — trying CUDA compute"
        );
        match new_cuda() {
            Ok(sel) => return Ok(sel),
            Err(e) => {
                // Loud, because this is the silent-degradation trap: OptiX says
                // yes but the CUDA context will not come up, so we are about to
                // render on a different backend than the machine deserves.
                eprintln!(
                    "[backend] WARNING OptiX probe said yes but the CUDA backend failed to \
                     initialise ({e}); FALLING BACK to Vulkan KHR — hardware traversal is still \
                     hardware, but CLAS/OMM/SER are lost on this run."
                );
            }
        }
    }

    // 2. Apple.
    #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
    {
        eprintln!("[backend] probe: macOS — selecting Metal");
        return new_metal();
    }

    // 3. AMD / Intel / NVIDIA-without-OptiX: Vulkan KHR ray query.
    #[cfg(not(all(target_os = "macos", feature = "spectra-native-metal")))]
    {
        new_vulkan()
    }
}

/// The single unmissable line, plus a scream when the chosen backend has no
/// hardware traversal at all.
fn announce(sel: &SelectedBackend, source: Source) {
    // On OptiX/CUDA, traversal is owned by the OptiX layer, so the compute
    // backend's `is_hw_rt_available()` (the KHR question) is false by design and
    // is NOT evidence of a software fallback. Everywhere else it is the answer.
    let hw_rt = match sel.kind {
        ResidentBackendKind::OptixCuda => true,
        _ => sel.gpu.is_hw_rt_available(),
    };
    eprintln!(
        "[backend] resident path-tracer backend = {} device=\"{}\" vram={} GiB source={} hw_traversal={}",
        sel.kind.label(),
        sel.gpu.device_name(),
        sel.gpu.total_vram() >> 30,
        match source {
            Source::Hardware => "device-probe",
            Source::EnvOverride => "SPECTRA_BACKEND override",
        },
        if hw_rt { "YES" } else { "NO" },
    );
    if !hw_rt {
        eprintln!(
            "[backend] *** NO HARDWARE RAY TRAVERSAL *** — this run traverses the SOFTWARE BVH, \
             which drops every instanced prototype: expect a frame with NO buildings and NO trees. \
             This is a broken run, not a slow one."
        );
    }
}

/// Build a `SelectedBackend` with no concrete interop handle. Correct for
/// CUDA: the same-device present path is Vulkan/Metal only.
fn boxed_only(gpu: Box<dyn GpuBackend>, kind: ResidentBackendKind) -> SelectedBackend {
    SelectedBackend {
        gpu,
        kind,
        vulkan: None,
        #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
        metal: None,
    }
}

fn new_cuda() -> Result<SelectedBackend, String> {
    spectra_gpu::CudarcSlangBackend::new(0)
        .map(|b| boxed_only(Box::new(b), ResidentBackendKind::OptixCuda))
        .map_err(|e| format!("CUDA backend init: {e:?}"))
}

#[cfg(not(all(target_os = "macos", feature = "spectra-native-metal")))]
fn new_metal() -> Result<SelectedBackend, String> {
    Err(
        "Metal backend is not compiled into this binary (needs macOS + \
         vox_render/spectra-native-metal)"
            .to_string(),
    )
}

#[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
fn new_metal() -> Result<SelectedBackend, String> {
    let b = spectra_gpu::SharedMetalBackend::new(0)
        .map_err(|e| format!("Metal backend init: {e:?}"))?;
    // CLONE, not a second device — `SharedMetalBackend` is Arc-backed, so this
    // handle and the boxed one are the SAME MTLDevice. A second device would
    // hand the presenter a buffer it cannot see (a black window).
    Ok(SelectedBackend {
        gpu: Box::new(b.clone()),
        kind: ResidentBackendKind::Metal,
        vulkan: None,
        metal: Some(b),
    })
}

fn new_vulkan() -> Result<SelectedBackend, String> {
    let b = spectra_gpu::SharedVulkanBackend::new(0)
        .map_err(|e| format!("Vulkan backend init: {e:?}"))?;
    // CLONE, not a second device — see `new_metal`. The same-device present path
    // binds its swapchain to THIS VkDevice.
    Ok(SelectedBackend {
        gpu: Box::new(b.clone()),
        kind: ResidentBackendKind::VulkanKhr,
        vulkan: Some(b),
        #[cfg(all(target_os = "macos", feature = "spectra-native-metal"))]
        metal: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind must print as something a support log can grep. A label that
    /// silently changed would break the one line that says what ran.
    #[test]
    fn every_backend_kind_has_a_distinct_label() {
        let labels = [
            ResidentBackendKind::OptixCuda.label(),
            ResidentBackendKind::VulkanKhr.label(),
            ResidentBackendKind::Metal.label(),
        ];
        assert_eq!(labels, ["OptiX/CUDA", "Vulkan KHR", "Metal"]);
    }

    /// A nonsense `SPECTRA_BACKEND` must be a hard error naming the accepted
    /// values, never a silent fall-through to whatever the probe would pick —
    /// a typo'd override that quietly did something else is how a measurement
    /// gets attributed to the wrong backend.
    #[test]
    fn an_unknown_forced_backend_is_an_error_that_names_the_valid_values() {
        // SAFETY: single-threaded test, restored before returning.
        let prev = std::env::var("SPECTRA_BACKEND").ok();
        unsafe { std::env::set_var("SPECTRA_BACKEND", "rtx-please") };
        let err = select_resident_backend().err().expect("must be an error");
        match prev {
            Some(v) => unsafe { std::env::set_var("SPECTRA_BACKEND", v) },
            None => unsafe { std::env::remove_var("SPECTRA_BACKEND") },
        }
        assert!(err.contains("rtx-please"), "got: {err}");
        assert!(
            err.contains("vulkan"),
            "must list what IS accepted, got: {err}"
        );
        assert!(
            err.contains("metal"),
            "must list what IS accepted, got: {err}"
        );
    }
}
