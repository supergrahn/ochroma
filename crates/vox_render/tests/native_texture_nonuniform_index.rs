//! Runtime-kernel contract: the material-texture descriptor array must be
//! indexed with a NON-UNIFORM-qualified index.
//!
//! WHY THIS EXISTS (witnessed defect, AMD Radeon 780M / RADV Vulkan, 2026-07-24)
//! --------------------------------------------------------------------------
//! `texture_atlas.slang` declares the shipped material textures as a descriptor
//! array:
//!
//! ```slang
//! Texture2D<float4> g_native_textures[SPECTRA_MAX_NATIVE_TEXTURES];
//! ```
//!
//! and every sampling helper selects a slot with `tex_id`, which is
//! `mat.albedo_tex` / `mat.normal_tex` / … — i.e. a value that comes from the
//! material of the surface THAT LANE hit. It therefore varies from lane to lane
//! inside one wave.
//!
//! Vulkan requires such an index to carry the `NonUniformEXT` decoration.
//! Slang emits it only for `NonUniformResourceIndex(...)`. Without it the index
//! is treated as wave-uniform: RADV loads the image descriptor into scalar
//! registers from ONE lane and applies it to all 64 lanes of the wave, so a
//! whole run of 64 consecutive pixels samples whichever material texture the
//! first lane happened to own.
//!
//! Live-path witness of the unqualified build (setback masonry office tower,
//! Spectra Vulkan present, balanced tier, albedo AOV
//! `SPECTRA_NO_ALPHA_CUTOUT=12`, denoiser off): **56.4 %** of horizontal albedo
//! transitions landed exactly on linear-pixel-index ≡ 0 (mod 64) — the RADV
//! wave64 boundary — against 1.6 % expected by chance. Visually the brick,
//! limestone reveal and window-glass albedos smeared into each other in
//! horizontal 1-px runs, so tall facades (many material changes per scanline)
//! read as dark, muddy noise while a 3-storey walk-up (broad uniform spans,
//! most waves single-material) stayed clean. With the index qualified the same
//! measurement drops to **1.7 %** — chance level — and the facade reads as real
//! brick with legible windows and trim.
//!
//! This test is a SOURCE-LEVEL tripwire on that defect class: it fires the
//! moment any `g_native_textures[...]` selection is written without the
//! qualifier, on every platform, with no GPU required.

use std::path::PathBuf;

/// `texture_atlas.slang` in the kernel tree THIS BUILD WOULD ACTUALLY RUN.
///
/// Resolution mirrors the runtime's own order (`splat_backend::
/// resolve_slang_kernel_dir`): the explicit `OCHROMA_SLANG_KERNEL_DIR` override
/// first, then the `spectra-renderer` path dependency declared in `Cargo.toml`
/// (`../../../spectra/rust/spectra-renderer`). Grading the tree the renderer
/// loads is the point — a contract checked against a tree we never dispatch
/// would prove nothing. If the file is missing the crate could not have been
/// built against it, so a hard failure is correct — never skip.
fn texture_atlas_slang() -> PathBuf {
    if let Ok(dir) = std::env::var("OCHROMA_SLANG_KERNEL_DIR") {
        let candidate = PathBuf::from(dir).join("texture_atlas.slang");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spectra/slang/texture_atlas.slang")
}

/// Every `g_native_textures[<index>]` selection in the runtime kernel source,
/// as the raw `<index>` text (bracket-balanced).
fn descriptor_array_indices(source: &str) -> Vec<String> {
    const ARRAY: &str = "g_native_textures[";
    let mut out = Vec::new();
    let bytes = source.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = source[search_from..].find(ARRAY) {
        let open = search_from + rel + ARRAY.len();
        let mut depth = 1usize;
        let mut cursor = open;
        while cursor < bytes.len() && depth > 0 {
            match bytes[cursor] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            cursor += 1;
        }
        assert!(
            depth == 0,
            "unbalanced '[' in g_native_textures index at byte {open}"
        );
        out.push(source[open..cursor - 1].trim().to_string());
        search_from = cursor;
    }
    out
}

#[test]
fn native_material_texture_array_is_indexed_non_uniformly() {
    let path = texture_atlas_slang();
    let source = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "runtime kernel source {} must be readable (it is the texture path this crate ships): {error}",
            path.display()
        )
    });

    // The declaration itself uses a compile-time constant extent; it is not a
    // selection and must not be graded against the contract.
    let indices: Vec<String> = descriptor_array_indices(&source)
        .into_iter()
        .filter(|index| index != "SPECTRA_MAX_NATIVE_TEXTURES")
        .collect();

    // Guard against a vacuous pass: if the array is ever renamed or the helpers
    // move, this test must fail loudly rather than assert over an empty set.
    assert!(
        indices.len() >= 4,
        "expected the material-texture descriptor array to be selected in at least the four \
         sampling helpers of {}, found {} selection(s): {:?} — if the array was renamed or \
         moved, RETARGET this contract instead of deleting it",
        path.display(),
        indices.len(),
        indices
    );

    let unqualified: Vec<&String> = indices
        .iter()
        .filter(|index| !index.starts_with("NonUniformResourceIndex("))
        .collect();

    assert!(
        unqualified.is_empty(),
        "{} selects the material-texture descriptor array with {} per-lane index/indices that are \
         NOT wrapped in NonUniformResourceIndex(...): {:?}\n\
         The index is the hit surface's material texture slot, so it differs between lanes of one \
         wave. Unqualified, RADV (AMD, wave64) loads ONE lane's image descriptor into scalar \
         registers for the whole wave: 64 consecutive pixels sample the wrong material's texture \
         and tall facades render as dark, muddy horizontal smear. Wrap each index in \
         NonUniformResourceIndex(...).",
        path.display(),
        unqualified.len(),
        unqualified
    );
}
