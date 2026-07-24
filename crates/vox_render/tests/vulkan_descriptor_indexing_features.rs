//! Runtime-backend contract: the Vulkan device must ENABLE the descriptor-indexing
//! features that the megakernel's SPIR-V DECLARES.
//!
//! WHY THIS EXISTS (witnessed defect, AMD Radeon 780M / RADV Vulkan, 2026-07-25)
//! ---------------------------------------------------------------------------
//! This is the un-landed second half of the `NonUniformResourceIndex` fix
//! (spectra commit 4358568, guarded by `native_texture_nonuniform_index.rs`).
//!
//! That fix changed `texture_atlas.slang` to select the material-texture
//! descriptor array as
//!
//! ```slang
//! g_native_textures[NonUniformResourceIndex(tex_id)]
//! ```
//!
//! which makes Slang emit, into EVERY megakernel SPIR-V module the Vulkan
//! backend dispatches:
//!
//! ```spirv
//! OpCapability ShaderNonUniform
//! OpDecorate %n NonUniform          ; x281 in the shipped shade_and_bounce module
//! ```
//!
//! Vulkan requires the corresponding device features to be enabled at
//! `vkCreateDevice` time for such a module to be legal:
//!
//!   * `shaderSampledImageArrayDynamicIndexing`   (core `VkPhysicalDeviceFeatures`)
//!     — needed for ANY non-constant index into a sampled-image array;
//!   * `shaderSampledImageArrayNonUniformIndexing` (`VkPhysicalDeviceVulkan12Features`)
//!     — needed when that index is DIVERGENT across a wave, which `tex_id` is
//!       (it is the material of the surface each lane hit).
//!
//! `VulkanSlangBackend::new` requested NEITHER. It enabled only
//! `sampler_anisotropy` + `shader_int64` (core) and `scalar_block_layout` +
//! `buffer_device_address` + `shader_float16` (Vulkan 1.2). So the renderer
//! shipped a shader module that uses a capability the device was never asked to
//! grant — undefined behaviour that RADV happens to tolerate.
//!
//! Why it stayed invisible: no crash, no error return, a correct image on the
//! one GPU we witness on, and no Khronos validation layer installed on the dev
//! box to report it. A driver that is instead permitted to scalarize the
//! descriptor index (`readFirstLane`) reintroduces the ORIGINAL defect the
//! qualifier was added to cure: 64 consecutive pixels sampling one lane's
//! material texture, i.e. tall multi-material facades smearing into mud.
//!
//! The companion test `native_texture_nonuniform_index.rs` greps the Slang
//! SOURCE and therefore passes no matter what the device enables — it cannot
//! see this half of the contract. This test closes that gap by grading the
//! backend source that actually creates the device.
//!
//! No GPU is required: this is a source-level contract on the device-creation
//! path, so it fires on every platform and in CI.

use std::path::PathBuf;

/// `vulkan_backend.rs` in the spectra tree THIS BUILD WOULD ACTUALLY RUN.
///
/// Resolution mirrors the kernel-tree contract next door: the explicit
/// `OCHROMA_SLANG_KERNEL_DIR` override names a kernel tree, and the Rust backend
/// that consumes it lives at `../rust/spectra-gpu/src/` relative to it. Falling
/// back to the `spectra-gpu` path dependency declared in `Cargo.toml`
/// (`../../../spectra/rust/spectra-gpu`) grades the crate we link against. If
/// the file is missing the crate could not have been built, so a hard failure is
/// correct — never skip.
fn vulkan_backend_rs() -> PathBuf {
    if let Ok(dir) = std::env::var("OCHROMA_SLANG_KERNEL_DIR") {
        let candidate = PathBuf::from(dir)
            .join("../rust/spectra-gpu/src/vulkan_backend.rs")
            .canonicalize();
        if let Ok(path) = candidate {
            if path.is_file() {
                return path;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../spectra/rust/spectra-gpu/src/vulkan_backend.rs")
}

/// Strip `//` line comments so a feature named only in prose (the exact trap
/// this test must not fall into — the pre-fix file DISCUSSED descriptor arrays
/// at length in comments while enabling nothing) can never satisfy the contract.
fn code_without_line_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| match line.find("//") {
            Some(idx) => &line[..idx],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn vulkan_device_enables_the_descriptor_indexing_features_the_megakernel_declares() {
    let path = vulkan_backend_rs();
    let source = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "runtime Vulkan backend source {} must be readable (it is the device this crate \
             renders on): {error}",
            path.display()
        )
    });
    let code = code_without_line_comments(&source);

    // Guard against a vacuous pass: if the backend is ever restructured so that
    // it no longer creates the device here, this contract must fail loudly
    // rather than assert over a file that cannot contain the enables.
    assert!(
        code.contains("PhysicalDeviceVulkan12Features") && code.contains("create_device"),
        "{} no longer looks like the Vulkan device-creation path (expected both \
         PhysicalDeviceVulkan12Features and create_device) — RETARGET this contract instead \
         of deleting it",
        path.display()
    );

    // The megakernel SPIR-V indexes `g_native_textures[]` with a per-lane
    // material slot, so BOTH features are mandatory. `.feature(false)` must not
    // count as enabling it, hence the `(true)` in the needle.
    let required: [(&str, &str); 2] = [
        (
            "shader_sampled_image_array_dynamic_indexing(true)",
            "any non-constant index into the g_native_textures[] sampled-image array",
        ),
        (
            "shader_sampled_image_array_non_uniform_indexing(true)",
            "the DIVERGENT (per-lane material) index that texture_atlas.slang emits via \
             NonUniformResourceIndex, which makes Slang declare OpCapability ShaderNonUniform",
        ),
    ];

    let missing: Vec<&(&str, &str)> = required
        .iter()
        .filter(|(needle, _)| !code.contains(needle))
        .collect();

    assert!(
        missing.is_empty(),
        "{} creates the Vulkan device WITHOUT enabling {} required descriptor-indexing \
         feature(s):\n{}\n\
         The megakernel SPIR-V declares `OpCapability ShaderNonUniform` and carries NonUniform \
         decorations on the g_native_textures[] selection. Using a capability the device was \
         never asked to grant is undefined behaviour: RADV tolerates it, but a driver that \
         scalarizes the descriptor index instead makes 64 consecutive pixels sample one lane's \
         material texture — the exact facade smear that NonUniformResourceIndex was added to \
         cure. Enable each feature at vkCreateDevice (and fail closed when unsupported).",
        path.display(),
        missing.len(),
        missing
            .iter()
            .map(|(needle, why)| format!("  - `{needle}` — needed for {why}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
