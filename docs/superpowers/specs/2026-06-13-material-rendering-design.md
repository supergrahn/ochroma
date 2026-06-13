# Design: Render Materials Correctly — From "Flat/Dead" to Path-Traced Super-Real (2026-06-13)

**Status:** Draft
**Scope:** Diagnose why path-traced building materials read flat/dead (glass with no reflections, brick/stucco blotchy and washed out) and design the shading/material-feed fixes that make them photoreal. Touches `vox_render` (material packing + texture load) and the Spectra Slang shade path (`material_dispatch.slang`, `megakernel.slang`).
**Related:** memory `super-realism-northstar.md` (leg 2 = PBR materials), `facade-photorealism-roadmap`, design `2026-06-12-sdf-housing-quality-design.md`.

---

## 1. Problem Statement

Concrete, observed-this-session symptoms, each tied to a root cause in code:

- **Opaque surfaces have NO specular layer at all.** Every non-glass material is packed as `MAT_LAMBERT` — `crates/vox_render/src/splat_backend.rs:2516` (`a[0] = pack_u32(if glass { 3 } else { 1 })`). `MAT_LAMBERT` evaluates pure diffuse (Oren-Nayar / Lambert, `material_dispatch.slang:192-210`); it adds **zero** specular highlight and **zero** Fresnel grazing-angle brightening. Brick, stucco, concrete, painted trim, and metal trim therefore read as flat matte paint. (Web benchmark: "Using only Lambert diffuse without proper specular layering and Fresnel misses the critical view-dependent reflection behavior that makes materials appear three-dimensional." — Babylon/OpenPBR.)
- **`metallic` is dead data.** `pack_vulkan_mesh_material` writes `a[9] = m.metallic` (`splat_backend.rs:2521`) but the material type is hard-forced to `MAT_LAMBERT`, and `MAT_LAMBERT` never reads `metallic`. Cooked metal trim (`metallic_factor`, `mod.rs:642`) renders as diffuse grey.
- **Glass renders correctly at the BSDF level but is starved of what it can reflect.** `MAT_GLASS` (`material_dispatch.slang:668-700`) is a proper Fresnel reflect/refract dielectric (`sample_glass`, `brdf_glass.slang:227`). It looks "flat/dead" because: (a) the SDF building path keeps glass **opaque** (`mod.rs:648-651` — windows route through a separate SDF branch that does NOT call `sample_glass`), and (b) when glass IS used, absorption/tint and roughness are hardcoded to clear+default (`splat_backend.rs:2536-2539`), and reflections only appear if `sky_intensity > 0` and bounces are spent (see below).
- **Diffuse textures are mean-normalized toward a flat per-instance tint, destroying contrast.** `load_polyhaven_map` decodes sRGB correctly (`mod.rs:490`) but then multiplies every texel by `clamp(tint_c / mean_c, 0.25, 4.0)` (`mod.rs:500-509`). This pulls the whole texture toward a single average colour, flattening the mortar-vs-brick and dirt-vs-clean variation that reads as "real". This is the primary "washed out / blotchy" cause (NOT a missing sRGB decode — that part is correct).
- **The opaque path spends bounces only on diffuse GI; there is no dedicated specular reflection.** `dispatch_sample` for `MAT_LAMBERT` returns a cosine-hemisphere diffuse ray (`material_dispatch.slang:631`). With low `max_bounces` (near-realtime = few) and diffuse-only sampling, surfaces get soft ambient GI but never a crisp environment reflection.
- **SDF building hits bypass the BSDF and normal maps entirely.** The SDF shade block (`megakernel.slang:1032-1078`) does `albedo * NdotL` direct lighting, never calls `dispatch_eval`/`dispatch_sample`, never applies a normal map, and writes no continuation ray. SDF buildings therefore cannot have specular, GI, or normal-mapped surface relief at all.

What is already CORRECT (do not "fix"): sRGB→linear decode on diffuse load (`mod.rs:418-422, 490`); normal-map decode + on-the-fly TBN for the **mesh** path (`megakernel.slang:1934-1996`, `normal_map.slang`); HDRI/gradient sky on ray miss for all bounces (`megakernel.slang:1137-1188`); ACES tonemap + EV0 default look (`splat_backend.rs:312`).

---

## 2. Done When

Running the material render-gate suite produces these exact, human-visible outcomes (each gate is a `cargo test ... -- --nocapture` that writes a PNG and asserts a measured pixel statistic; a human can open the PNG and confirm):

`cargo test -p vox_render --features spectra-native material_realism_gates -- --nocapture` produces:

1. **Specular brick gate:** a brick wall lit by a low sun shows a visible grazing-angle sheen band — measured: mean luminance of the top 10% of wall rows facing the sun at a grazing angle is ≥ 1.4× the mean of the front-facing rows (Fresnel rim). Today this ratio is ≈ 1.0 (flat).
2. **Reflective glass gate:** a glass curtain-wall facing a coloured sky/HDRI shows the sky colour in the glass — measured: the glass region's mean hue is within 15° of the sky hue and its variance across the pane is ≥ 2× a matte control. Today the glass region is near-uniform dark.
3. **Metal trim gate:** a `metallic = 1.0, roughness = 0.2` trim strip shows a bright specular streak — measured: peak luminance in the trip strip ≥ 2× its diffuse-only render.
4. **Texture contrast gate:** the rendered brick wall's per-channel pixel std-dev is ≥ 0.8× the source `diffuse.jpg`'s linear std-dev (i.e. the mortar/brick contrast survives). Today, post mean-normalization, it is < 0.5×.

---

## 3. Capabilities

| Capability | Real behavior test | Stub test (forbidden) |
|---|---|---|
| Opaque surfaces get a dielectric specular lobe + Fresnel | `assert!(grazing_band_lum / front_lum > 1.4)` on a rendered brick PNG | `assert!(pixels.len() > 0)` |
| `metallic`/`roughness` drive the BSDF | `assert!(metal_peak_lum > 2.0 * diffuse_peak_lum)` comparing two renders | `assert!(mat.metallic == 1.0)` (checks the field, not the render) |
| Glass reflects the environment | `assert!(hue_delta(glass_mean, sky_hue) < 15.0 && glass_var > 2.0 * matte_var)` | `assert!(glass_region_nonblack())` |
| Texture contrast preserved | `assert!(render_std >= 0.8 * source_linear_std)` per channel | `assert!(texture.data.iter().any(|&v| v != texture.data[0]))` |
| SDF building hits use the full BSDF + normal map | render two SDF walls (smooth vs normal-mapped); `assert!(normalmapped_lum_std > 1.5 * smooth_lum_std)` | `assert!(sdf_hit_returned_color())` |

---

## 4. Architecture

### 4.1 Route opaque materials through `MAT_OPENPBR` (or `MAT_METAL`), not `MAT_LAMBERT` — biggest win

`MAT_OPENPBR` already exists and is a full uber-shader with a diffuse base + dielectric specular lobe + Fresnel + metallic + roughness, handling its own layering (`material_dispatch.slang:463-470` eval, `847-852` sample, implemented in `openpbr.slang`). The fix is almost entirely in the **packer**, not the shader: `pack_vulkan_mesh_material` (`splat_backend.rs:2504`) must select the material type by content instead of hard-forcing `MAT_LAMBERT`:

- `transmission > 0.0` → `MAT_GLASS` (3) — unchanged.
- `metallic > 0.5` → `MAT_METAL` (2) — already a correct Cook-Torrance GGX conductor (`material_dispatch.slang:213-275, 635-665`), reads `albedo`/`metallic`/`roughness`/optional complex IOR.
- otherwise → `MAT_OPENPBR` (16) — dielectric with a 4% F0 specular layer over the diffuse base. This is the path that turns brick/stucco/concrete from "matte paint" into "real surface with grazing sheen".

The packer must also fill the OpenPBR slots `eval_openpbr`/`sample_openpbr` read (specular weight, specular roughness, F0/specular IOR, diffuse weight). Slot indices come from the Vulkan `MaterialData` SPIR-V reflection layout already used in this packer (diffuse_weight `a[74]`, specular_weight `a[76]` are present today at `splat_backend.rs:2567-2568` — confirm they are the slots `openpbr.slang` consumes; this is an OPEN QUESTION to verify against `openpbr.slang` field offsets before implementation).

Decision: default the dielectric specular IOR to 1.5 (F0 ≈ 0.04, the standard non-metal value the web benchmark cites as "2%–5% at normal incidence"); expose it later via `PbrMaterial`.

### 4.2 Stop flattening diffuse textures — replace mean-normalize-to-tint with a gentle multiply

In `load_polyhaven_map` (`mod.rs:486-512`) the `clamp(tint/mean, 0.25, 4.0)` per-channel factor recolours the whole texture toward the flat instance tint, collapsing intra-texture contrast. Replace with: keep the sRGB→linear texels as the albedo, and apply `tint` as a **subtle** tint multiply (e.g. `lerp(texel, texel * tint_normalized, k)` with small `k`, or apply tint as a separate `base_color` tint in the shader where `mat.albedo *= mat.base_color`). The texture's own variation (mortar lines, colour breakup) must dominate. The render-gate (§2.4) measures std-dev preservation so this can't regress to flat. Note `megakernel.slang:1967` does `mat.albedo = t.xyz` (texel REPLACES base_color), so per-instance tint variety must be reintroduced as a multiply in the shade path or in the packer's `base_color`.

### 4.3 Glass: feed it tint + roughness + use it on buildings

Two sub-fixes. (a) In `pack_vulkan_mesh_material` (`splat_backend.rs:2529-2540`) stop hardcoding clear/no-absorption: pass `base_color` as a faint Beer-Lambert tint and respect `roughness` so glass can be slightly frosted/rough where authored (the `sample_glass` rough path already exists, `brdf_glass.slang:227`). (b) For SDF-cooked buildings, the window channel is currently kept opaque (`mod.rs:648-651`). Once §4.4 routes SDF hits through the BSDF, give window channels `transmission > 0` so they become real `MAT_GLASS` and reflect the sky. Until §4.4 lands, keep the existing dedicated SDF glass branch but verify it actually evaluates Fresnel reflection (not just transmission).

### 4.4 SDF building hits must use the full BSDF + normal map (unlocks SDF specular/GI/relief)

The SDF shade block (`megakernel.slang:1032-1078`) is a `albedo * NdotL` direct-light dead end: no BSDF, no normal map, no continuation ray. Refactor it to mirror the mesh hit path: sample albedo/roughness/normal textures (sRGB note: SDF albedo at `megakernel.slang:1040` uses the raw texel — but textures are uploaded already-linear from Rust, so that is consistent; do NOT add a second decode), apply the SDF gradient normal perturbed by the normal map via `apply_normal_map`, build a `MaterialData`, and route through `dispatch_eval`/`dispatch_sample` so the SDF surface spawns the same diffuse-GI + specular continuation rays the mesh path does. This is the largest shader change and should be a separate wave after §4.1.

### 4.5 Bounce budget + exposure for material readability

Specular reflections need bounce budget. `near_realtime` config max_bounces is low (`splat_backend.rs:91` reads `config.max_bounces`). For asset-review stills, raise to ≥ 6 bounces so glass reflects sky and metals show 2-bounce environment. Confirm `sky_intensity > 0` in the default `LightRig` (`splat_backend.rs:230`) so the miss handler (`megakernel.slang:1137-1188`) actually contributes reflectable radiance — if `sky_intensity == 0` the background is black (`megakernel.slang:1145`) and glass/metal reflect nothing, reading "dead". Keep ACES/EV0 default look (`splat_backend.rs:312`); add a per-gate `LookPreset::SoftReview` option for flat asset-inspection where needed.

---

## 5. Data Models

`PbrMaterial` (`splat_backend.rs:153-187`) gains optional dielectric-specular control so the packer can drive `MAT_OPENPBR` without inventing values in the shader:

```rust
pub struct PbrMaterial {
    // ...existing fields (base_color, roughness, metallic, emission_strength,
    // albedo_tex, roughness_tex, normal_tex, uv_scale, transmission, ior, thin_walled)...

    /// Dielectric specular reflectance at normal incidence as an IOR
    /// (1.5 = F0≈0.04, the standard non-metal value). Drives the MAT_OPENPBR
    /// specular lobe for opaque non-metals. Default 1.5.
    specular_ior: f32,        // private — accessor specular_ior()
    /// Specular layer weight [0,1]. Default 1.0 (full Fresnel specular).
    specular_weight: f32,     // private — accessor specular_weight()
    /// Optional ambient-occlusion texture index (-1 = none). Multiplies the
    /// diffuse + low-frequency specular. Default -1.
    ao_tex: i32,
    /// Optional height/displacement texture index (-1 = none) for parallax
    /// occlusion mapping on the mesh path. Default -1.
    height_tex: i32,
}
```

AO and height are listed for the roadmap (leg 2 of the north star calls for "full normal+roughness+AO+height (parallax)") but are OUT OF SCOPE for wave 1 (see §9) — only the type-selection + texture-contrast + glass fixes are wave 1.

---

## 6. API

```rust
// splat_backend.rs — the packer selects type by content (replaces the hard MAT_LAMBERT).
// No signature change; behaviour change only.
fn pack_vulkan_mesh_material(m: PbrMaterial) -> [f32; VULKAN_MATERIAL_FLOATS];
// Slot a[0]: 3 if m.transmission > 0 (MAT_GLASS);
//            2 if m.metallic > 0.5    (MAT_METAL);
//            16 otherwise             (MAT_OPENPBR).
// Fills OpenPBR specular slots from m.specular_ior / m.specular_weight when type==16.

// mod.rs — texture loader no longer mean-normalizes diffuse toward tint.
fn load_polyhaven_map(root: &Path, set: &str, kind: TexKind, tint: [f32;3]) -> TextureImage;
// Diffuse: sRGB->linear texels preserved; `tint` applied as a gentle multiply only.
```

```hlsl
// material_dispatch.slang — NO new code; MAT_OPENPBR/MAT_METAL/MAT_GLASS cases
// already exist (lines 213, 278, 463, 635, 668, 847). The change is purely that
// the host now emits these types for opaque/metal/glass instead of MAT_LAMBERT.

// megakernel.slang (wave 2) — SDF hit block (lines ~1032-1078) refactored to
// build MaterialData + call dispatch_eval / dispatch_sample + apply_normal_map,
// matching the mesh hit path (lines ~1917-1996, 2775).
```

---

## 7. Wiring

| Component | Called from | File | Notes |
|---|---|---|---|
| `pack_vulkan_mesh_material` type-select | `pathtrace_mesh_textured_to_rgba` / SDF textured packer | `crates/vox_render/src/splat_backend.rs:2504` | every material upload — already wired, behaviour change only |
| `load_polyhaven_map` no-flatten | cooked-building material resolve | `crates/vox_render/src/splat_backend_tests/mod.rs:470` | move into the game `TextureCache` once proven (tests mirror it) |
| OpenPBR slot fill | inside `pack_vulkan_mesh_material` | `splat_backend.rs:2567` (specular_weight slot already present) | verify offsets vs `openpbr.slang` |
| SDF BSDF route | SDF hit block | `~/src/spectra/slang/megakernel.slang:1032` | wave 2; READ-ONLY repo today — coordinate before editing |
| `max_bounces ≥ 6` for review | `RenderConfig` in the gate | `splat_backend.rs:91` | per-gate, not a global default change |

---

## 8. Open Questions

- [ ] Exact OpenPBR `MaterialData` slot offsets `eval_openpbr`/`sample_openpbr` read — confirm against `~/src/spectra/slang/openpbr.slang` before filling `pack_vulkan_mesh_material` (avoid inventing offsets; the packer comment at `splat_backend.rs:2511` documents the verification method).
- [ ] Does the existing dedicated **SDF window** glass branch evaluate Fresnel **reflection** of the sky, or only transmission? (Determines whether §4.3b is needed before §4.4.)
- [ ] Is `metallic > 0.5` the right MAT_METAL cutoff, or should metallic-roughness blend stay inside `MAT_OPENPBR` (which handles a metallic parameter natively)? Prefer OpenPBR-handles-metallic if its conductor lobe matches `MAT_METAL` quality — fewer code paths.
- [ ] `sky_intensity` default in `LightRig::default()` (`splat_backend.rs:230`) — confirm > 0 so reflections have something to catch.

---

## 9. Out of Scope

- AO-texture and height/parallax-occlusion consumption (data-model stubs added in §5 but not wired) — a later facade-photorealism wave.
- Clearcoat/sheen layering, anisotropy, and per-instance procedural weathering blend (already partly wired elsewhere).
- The wgpu interactive `TiledSplatRenderer` path — this design targets the Spectra path-traced still/cinematic path.
- Changing the tonemap operator set or DLSS/denoiser.
- Editing `~/src/forge` (a concurrent node-DAG build owns it).

---

## 10. Related Plans / Designs

- Depends on: Spectra `material_dispatch.slang` MAT_OPENPBR/MAT_METAL/MAT_GLASS cases (already implemented).
- Related: `super-realism-northstar.md` (leg 2 PBR materials), `facade-photorealism-roadmap`, `2026-06-12-sdf-housing-quality-design.md`.
- Required before: a facade beauty-rig wave (window glow, depth/banding, weathering) — those build on a correct base material.

### Benchmark sources (what "correct" is)
- OpenPBR Surface Shading Model — dielectric base = rough GGX microfacet with a 4% F0 specular layer over diffuse: <https://academysoftwarefoundation.github.io/OpenPBR/>
- Babylon.js Mastering PBR — Lambert-only misses view-dependent reflection that gives surfaces depth: <https://doc.babylonjs.com/features/featuresDeepDive/materials/using/masterPBR>
- Chaos/Enscape glass best practices for arch-viz (roughness + IOR + thin-pane vs solid): <https://blog.chaos.com/best-practices-glass-in-architectural-design/>
- Demofox path-tracing Fresnel + rough refraction + absorption: <https://blog.demofox.org/2020/06/14/casual-shadertoy-path-tracing-3-fresnel-rough-refraction-absorption-orbit-camera/>
