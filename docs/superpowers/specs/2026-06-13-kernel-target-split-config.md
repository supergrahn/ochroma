# Design: One main kernel → CUDA + Vulkan, one config object for both

Status: APPROVED (2026-06-13). Governing law: see memory `config-first-no-hardcode-directive`.

## Goal

1. **One hardware-agnostic main kernel** (`spectra/slang/megakernel.slang` + its imported modules) that compiles to **both** a CUDA kernel (PTX) and a Vulkan kernel (SPIR-V). At runtime we use whichever the hardware supports. The main kernel never names a backend.
2. **One config object** shared by both backends (and Ochroma, and the game), carrying **every possible value with its default**.
3. **Config-first debugging:** when something is wrong, the config is the first place we look.
4. **Nothing hardcoded:** if a value can or should change, it is a config value.

All of this is implemented in **Spectra** (the renderer crate stack), consumed by Ochroma and the game.

## Current state (grounded — file:line)

- `slang/megakernel.slang` (3,800 LOC) is **already ~target-pure**: built from Slang `import` modules (`math_utils`, `material_dispatch`, `lights`, `sampling`, …); the only preprocessor lines are `SHADE_BLOCK_SIZE` / `SPECTRAL_USE_TILING` guards; the `cuda_sm_2_0` lines are boilerplate warning-suppression in every kernel, not CUDA code.
- Hardware-bound ray tracing is isolated in `slang/rt_query.slang` — but it is **Vulkan-flavored** (`RayQuery<>` over a `VK_KHR_acceleration_structure` TLAS). A CUDA build needs OptiX traversal.
- NVIDIA-only intrinsics leak into otherwise-shared files: `slang/lss_intersect.slang:25` (`optix_intersect_round_linear`), `slang/neural_material.slang:193` / `slang/material_nodes.slang:68` (`sm_6_9` / cooperative-vector).
- Host backends are **already split**: `spectra-gpu/src/vulkan_backend.rs` (slang→SPIR-V, ray_query) and `cudarc_backend.rs` (slang→PTX). Runtime selection already tries Vulkan then CUDA in `vox_render/splat_backend.rs`.
- `spectra-renderer/RenderConfig` is serde-serializable and already holds `tonemap`/`exposure_ev` — but render entry points never set them (ACES-by-omission). The CUDA-only feature kernels are gated by **lying empty `= []` feature flags** in `spectra-renderer/Cargo.toml` (audit A3).

So we are ~80% there: the work is to (a) abstract the one hardware-bound seam, (b) unify the config, (c) make the flags honest, (d) prove no dead settings.

## Design

### A. Kernel target abstraction
- Define a Slang **`interface IRayTracer { TraceHit trace_closest(Ray); bool trace_shadow(Ray); }`**. The megakernel calls the interface — never a backend.
- Two implementations: `RayQueryTracer` (Vulkan, the current `rt_query.slang` body) and `OptixTracer` (CUDA, OptiX traversal). Select per target.
- The few NVIDIA intrinsics (`optix_intersect_round_linear`, `sm_6_9` cooperative-vector) move behind Slang **`__target_switch`** / capability guards **with a generic fallback** so the shared file compiles cleanly on both targets.
- CUDA-only feature kernels (restir, nrc, niv, path-guide, …) stay separate files, gated by **real** feature flags (kill the empty `= []` aliases — A3).

### A2. Realtime capability matrix (the hardware-bound seams)

The RT traversal is not the only hardware-bound seam — realtime needs an **upscaler** and a **denoiser** too, and on NVIDIA those are DLSS/OptiX. Each seam is the same shape: a cross-vendor implementation (works on our AMD/Vulkan dev box) and an NVIDIA-only implementation, behind one trait, selected at runtime by the backend probe and overridable from the config. The main kernel and the renderer core never name a vendor.

| Capability | Vulkan / cross-vendor (AMD dev box) | NVIDIA / CUDA | Abstraction | Config key |
|---|---|---|---|---|
| RT traversal | `RayQuery` (VK_KHR) | OptiX | `IRayTracer` | `backend` |
| Realtime upscale | **FSR** | **DLSS** | `IUpscaler` | `render.upscaler` (auto/fsr/dlss/none + quality) |
| Denoise | SVGF / OIDN | OptiX denoiser | `IDenoiser` | `render.denoiser` |
| Neural (NRC/NIV) | (off / generic) | cooperative-vector `sm_6_9` | feature-gated | `features.*` |

DLSS/OptiX stub on the AMD box today (no CUDA) — so our realtime track is **FSR + SVGF**; DLSS + OptiX-denoise light up automatically on an NVIDIA build with no code change, because they're selected behind the trait. "Use whatever the hardware supports" applies per-capability, not just per-kernel.

### B. One config object (`spectra-types::RenderSettings`)
- Typed Rust struct hierarchy, `#[derive(Serialize, Deserialize)]`, `#[serde(deny_unknown_fields)]`, a `version: u32`, and a `validate()` (range-check) run on load.
- Sections (every leaf has a default): `backend` (auto / force-cuda / force-vulkan), `render` (resolution, spp, max_bounces, denoiser{enabled,strength}), `look` (exposure_ev, tonemap{operator|lut, strength}, color/white_balance/levels — the Lumina `{enabled,value}` pattern), `lighting` (sun, sky_dome, atmosphere{enabled,mie,turbidity,hdri}), `features` (restir/nrc/… each `{enabled}`).
- `resolve(defaults ⊕ preset ⊕ user_overrides) -> EffectiveSettings`, in one place. The game persists JSON; Ochroma + both backends read the resolved struct. **No call-site defaults.**
- Migrate today's scattered fields into it: the `LightRig` additions (atmosphere_enabled, sun_radiance, mie, turbidity), `LookPreset`, per-material `uv_scale`, the weathering enable — all become `RenderSettings` leaves.

### C. Backend selection
- `backend: Auto` probes Vulkan (KHR ray_query) first, then CUDA, then CPU — formalize the existing `splat_backend.rs` selector against the config (`force-*` overrides for diagnostics).

### D. No dead settings (the enforcement)
- **Perturbation coverage test (CI gate):** for every leaf field of `RenderSettings`, set it off-default and assert the rendered output measurably changes (hash / mean-luma — `look_preset_demo` generalized). A field bound nowhere → identical output → test fails, naming the dead setting. This is the antidote to the codebase's dominant failure mode (built-but-unwired systems).

## Phases & Done-When

1. **`RenderSettings` schema + validate + resolve** in `spectra-types`. Done-When: `cargo test -p spectra-types` prints `[settings] N leaf fields, all defaulted; resolve(defaults)==RenderConfig::default()` and round-trips JSON with `deny_unknown_fields` rejecting a typo'd key.
2. **Consumers read only resolved settings** (delete call-site defaults in `vox_render` entry points + `spectra-renderer`). Done-When: grep finds zero `RenderConfig::near_realtime` mutations of tonemap/exposure; a render with `look.exposure_ev=+2` is visibly brighter (mean-luma delta > 40).
3. **`IRayTracer` interface + RayQuery/OptiX impls**; megakernel calls the interface. Done-When: `slangc --target spirv megakernel` and `slangc --target cuda megakernel` both compile with zero target-specific code in `megakernel.slang` (grep clean).
4. **Honest feature flags** (A3): no `= []` aliases; `--no-default-features` on AMD builds & runs. Done-When: `cargo build -p spectra-bin` on the AMD box with default features = Vulkan only (already true after A1); each feature flag gates a real sub-crate feature.
5. **Perturbation coverage gate.** Done-When: `cargo test -p vox_render --features spectra-native perturbation_coverage` asserts every `RenderSettings` leaf moves the output; a deliberately-unwired field makes it FAIL.

## IMPORTANT NOTES (real signatures)
- `spectra_renderer::RenderConfig` — already `pub tonemap: ToneMapper, pub exposure_ev: f32`; `RenderSettings` supersets it.
- `spectra_tonemap::ToneMapper { Aces, Reinhard, ReinhardLuma, Filmic, None, AgX }`; color grading already exists as `Cdl` / `LiftGammaGain` (currently dead in the native path — wire via `look.color`/`levels`).
- `vox_render::LookPreset::resolve() -> (ToneMapper, f32)` — folds into `RenderSettings::look`.
- Engine/game split: `RenderSettings` schema + resolve + kernels = engine (rendering concept, not game). The game owns the JSON file, the UI, the preset library, and hot-reload.
