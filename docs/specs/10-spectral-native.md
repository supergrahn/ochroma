# Domain 10 — Spectral-Native Capabilities (Surpassing Unreal)

**Status:** Draft — 2026-03-29
**Scope:** Neural spectral uplifting, material fingerprinting, spectral damage & storytelling, cinematography tools, procedural generation, audio-visual synchronization, photogrammetry pipeline
**Engine:** Ochroma spectral Gaussian Splatting — Rust workspace, wgpu 24, WGSL shaders, rayon, glam, egui, tract-onnx, burn

---

## Goals

Domain 10 delivers the capabilities that are structurally impossible in Unreal Engine, Unity, or any RGB-native renderer. These features work because every fundamental rendering primitive in Ochroma — `GaussianSplat.spectral: [u16; 8]` — carries per-splat spectral data at 8 bands covering ~380–720 nm. RGB engines cannot retrofit these features without a complete rewrite of their geometry representation; Ochroma has them for free as a consequence of its architecture.

The goal is not merely technical novelty. Each feature must be useful to a game developer:
- **Neural uplifting** solves the practical problem of importing existing 3DGS photogrammetry captures into Ochroma.
- **Material fingerprinting** gives designers an automatic material identification and acoustic assignment tool.
- **Spectral damage** gives world designers free environmental storytelling without placing triggers.
- **Spectral cinematography** gives cinematographers an entirely new creative axis with no re-lighting cost.
- **Spectral PCG** lets procedural artists define biomes and scatter rules in terms of physical material properties.
- **Audio-visual sync** gives audio designers a real-time reactive music system driven by scene lighting.
- **Photogrammetry pipeline** closes the loop from physical-world capture to playable game content.

**Performance targets:**
- Neural uplifter: 50K splat conversion in < 1 second on CPU (single-threaded rayon worker)
- `SpectralDamageSystem`: 100K damaged splats updated per frame in < 1 ms
- `SpectralAVSync` band aggregation: full screen coverage-weighted average in < 0.5 ms (GPU compute)
- `SpectralScanner` fingerprint match: top-5 from 500-entry `MaterialDatabase` in < 0.1 ms

---

## Architecture

```
crates/
  vox_data/
    src/
      uplift/
        mod.rs              — SpectralUplifter, MaterialClass
        onnx_runner.rs      — tract-onnx inference wrapper
        inpainter.rs        — SpectralInpainter (near-IR band estimation)
      fingerprint/
        mod.rs              — MaterialFingerprint, MaterialDatabase, MaterialEntry
        kl_divergence.rs    — KL divergence fingerprint matching
        webcam_import.rs    — MobileScanner, ColorCheckerCalibration
      damage/
        mod.rs              — SpectralDamageComponent, DamageEvent
        system.rs           — SpectralDamageSystem
        narrative.rs        — SpectralNarrative, NarrativeCheckpoint

  vox_render/
    src/
      cinematography/
        camera_preset.rs    — SpectralCameraPreset, preset library
        time_of_day.rs      — SpectralTimeOfDay, illuminant_for_time() extension
        sequencer.rs        — CinematicSequencer integration
      avsync/
        mod.rs              — SpectralAVSync, AvSyncBand
        audio_driver.rs     — SpectralAudioDriver
        music_reactive.rs   — SpectralReactiveMusicLayer

  vox_core/
    src/
      pcg/
        spectral_nodes.rs   — FilterBySpectralBand, ScatterBySpectralDensity,
                              SpectralBiomeTransition
      weather/
        spectral_weather.rs — SpectralWeather, rain/snow/fog/dust integration

  vox_app/
    src/
      tools/
        spectral_scanner.rs — SpectralScanner held item
        fingerprinter.rs    — FingerprinterTool (egui widget)
      capture/
        ochroma_capture.rs  — OchoraCapture pipeline entry point
```

---

## 10.1 Neural Spectral Uplifting (3DGS → 8-band)

### Problem

A standard 3DGS capture stores `color: [f32; 3]` (sRGB) per splat. Ochroma requires `spectral: [u16; 8]` (f16, 8 bands ~380–720 nm). Uplifting from RGB to spectral is an ill-posed inverse problem — infinitely many spectral power distributions metamerisize to the same RGB triplet under D65. A neural network trained on real spectroradiometer measurements provides the most probable spectral reconstruction for each material class.

### SpectralUplifter Architecture

```rust
// vox_data/src/uplift/mod.rs

pub enum MaterialClass {
    Diffuse,      // lambertian: wood, concrete, fabric, skin
    Metallic,     // specular: iron, copper, gold, aluminium
    Translucent,  // subsurface: wax, skin thin sections, foliage
    Emissive,     // self-luminous: LED, fire
}

pub struct UpliftInput {
    pub rgb:            [f32; 3],    // linear (NOT sRGB gamma)
    pub material_class: MaterialClass,
    pub roughness:      f32,         // Beckmann roughness, 0.0–1.0
}

pub struct SpectralUplifter {
    diffuse_model:     tract_onnx::prelude::SimplePlan<...>,
    metallic_model:    tract_onnx::prelude::SimplePlan<...>,
    translucent_model: tract_onnx::prelude::SimplePlan<...>,
    emissive_model:    tract_onnx::prelude::SimplePlan<...>,
}

impl SpectralUplifter {
    pub fn load(model_dir: &Path) -> Result<Self, UpliftError> {
        // Load 4 ONNX models from model_dir/diffuse.onnx etc.
        // Each model: 5 inputs (R, G, B, roughness, class_embed) → 8 outputs
    }

    pub fn uplift(&self, input: &UpliftInput) -> [f32; 8] {
        let model = match input.material_class {
            MaterialClass::Diffuse     => &self.diffuse_model,
            MaterialClass::Metallic    => &self.metallic_model,
            MaterialClass::Translucent => &self.translucent_model,
            MaterialClass::Emissive    => &self.emissive_model,
        };
        // Build tract tensor from input; run inference; extract 8-float output
        // Apply sigmoid clamp to [0, 1]
        let raw: [f32; 8] = /* tract inference */;
        raw.map(|v| v.clamp(0.0, 1.0))
    }

    /// Batch uplift: processes 50K splats in parallel using rayon.
    pub fn uplift_batch(&self, inputs: &[UpliftInput]) -> Vec<[f32; 8]> {
        inputs.par_iter().map(|i| self.uplift(i)).collect()
    }
}
```

**Network architecture per model:**
- Input: 5 neurons `[R, G, B, roughness, material_class_embedding]`
- Hidden: 3 layers × 64 neurons, ReLU activation
- Output: 8 neurons, sigmoid activation → values in `[0, 1]`
- Total parameters per model: `5×64 + 64×64 + 64×64 + 64×8 = ~8.9K params` — fits in L1 cache

**Training (offline, Python + PyTorch, exported to ONNX):**
- Dataset: 200 materials × 6 illuminant conditions × 5 roughness levels = 6,000 samples from a Konica Minolta CS-2000 spectroradiometer, measured at 1 nm resolution, resampled to 8 bands (Smits 1999 reconstruction basis for the resampling kernel).
- Loss: MSE on 8 output bands + perceptual regularization term (reconstructed RGB from output bands must match input RGB under D65 CIE 1931 observer).
- Validation RMSE target: < 0.05 per band on held-out test set (20% split).
- Export: `torch.onnx.export()` with `opset_version=17`; loaded by `tract-onnx` at runtime.

**Import pipeline integration:**

```rust
// vox_data/src/import/ply_import.rs (extension)

pub fn import_3dgs_ply(path: &Path, uplifter: &SpectralUplifter) -> Result<Vec<GaussianSplat>, ImportError> {
    let ply = ply_rs::Parser::new().read_ply(/* file */)?;
    let raw_splats: Vec<RawSplat3dgs> = parse_ply_splats(&ply)?;

    // Detect material class per splat from RGB + SH coefficients
    let inputs: Vec<UpliftInput> = raw_splats.iter().map(|rs| {
        UpliftInput {
            rgb:            linear_from_srgb(rs.color),
            material_class: classify_material(rs.color, rs.opacity, rs.scale),
            roughness:      estimate_roughness_from_sh(rs.sh_coeffs),
        }
    }).collect();

    let spectral_batch = uplifter.uplift_batch(&inputs);

    raw_splats.iter().zip(spectral_batch).map(|(rs, spectral)| {
        let spectral_u16 = spectral.map(|v| half::f16::from_f32(v).to_bits());
        // Apply SpectralInpainter for band 7 (near-IR)
        let spectral_u16 = SpectralInpainter::fix_ir_band(spectral_u16, &inputs[i]);
        Ok(GaussianSplat {
            position: rs.position,
            scale:    rs.scale,
            rotation: rs.rotation,
            opacity:  rs.opacity,
            spectral: spectral_u16,
        })
    }).collect()
}
```

### SpectralInpainter

Near-IR (band 7, ~700–720 nm) has no direct RGB equivalent. The inpainter applies two priors:

1. **General prior:** `band_7 = mean(spectral[4..=6]) * 0.8` — near-IR broadly correlates with visible red/orange reflectance.
2. **Foliage exception:** detect foliage by `rgb.g / (rgb.r + epsilon) > 1.4` (chlorophyll absorption of red makes green relatively dominant). For foliage, `band_7 = mean(spectral[4..=6]) * 2.1` — the chlorophyll red-edge causes anomalously high near-IR reflectance (Knipling 1970).
3. **Metal exception:** metals detected by low roughness + high overall brightness → `band_7 = mean(spectral[0..=7])` (flat spectral curve).

```rust
// vox_data/src/uplift/inpainter.rs

pub struct SpectralInpainter;

impl SpectralInpainter {
    pub fn fix_ir_band(mut spectral: [u16; 8], input: &UpliftInput) -> [u16; 8] {
        let as_f32: [f32; 8] = spectral.map(|v| half::f16::from_bits(v).to_f32());
        let ir = if is_foliage(input.rgb) {
            (as_f32[4] + as_f32[5] + as_f32[6]) / 3.0 * 2.1
        } else if matches!(input.material_class, MaterialClass::Metallic) {
            as_f32.iter().sum::<f32>() / 8.0
        } else {
            (as_f32[4] + as_f32[5] + as_f32[6]) / 3.0 * 0.8
        };
        spectral[7] = half::f16::from_f32(ir.clamp(0.0, 1.0)).to_bits();
        spectral
    }
}
```

---

## 10.2 Spectral Material Fingerprinting & Scanning

### MaterialFingerprint

```rust
// vox_data/src/fingerprint/mod.rs

pub struct MaterialFingerprint {
    pub bands:    [f32; 8],   // mean spectral value per band
    pub variance: [f32; 8],   // variance per band (for KL divergence matching)
}

pub struct MaterialEntry {
    pub name:        String,
    pub fingerprint: MaterialFingerprint,
    pub category:    MaterialCategory,
    pub acoustic:    AcousticProfile,   // for SDF reverb auto-assignment
}

pub enum MaterialCategory {
    Stone, Wood, Metal, Fabric, Glass, Foliage, Soil, Water, Concrete, Plaster, Ceramic
}

pub struct MaterialDatabase {
    pub entries: Vec<MaterialEntry>,
    kd_index:   kdtree::KdTree<f32, usize, [f32; 8]>,   // for fast nearest-neighbor
}
```

The database ships with 500 entries pre-populated from the same spectroradiometer dataset used for uplifter training, plus synthetic entries generated from known spectral reflectance databases (Colour & Vision Research Laboratory spectral dataset, University College London). The `kdtree` crate provides O(log n) nearest-neighbor lookup in the 8-dimensional spectral space.

### KL Divergence Matching

Standard Euclidean distance in 8D spectral space is not perceptually meaningful — two materials that differ only in band 7 (near-IR) look identical to a human. The matching metric is symmetric KL divergence treating each fingerprint as an 8-dimensional Gaussian:

```rust
// vox_data/src/fingerprint/kl_divergence.rs

/// Symmetric KL divergence between two 8-band Gaussian distributions.
pub fn spectral_kl_divergence(a: &MaterialFingerprint, b: &MaterialFingerprint) -> f32 {
    // KL(P||Q) + KL(Q||P) for diagonal Gaussians
    // = 0.5 * sum_i [ (var_a[i]/var_b[i]) + (var_b[i]/var_a[i])
    //               + (mean_b[i] - mean_a[i])^2 * (1/var_a[i] + 1/var_b[i]) - 2 ]
    let eps = 1e-6_f32;
    (0..8).map(|i| {
        let va = a.variance[i].max(eps);
        let vb = b.variance[i].max(eps);
        let dm = b.bands[i] - a.bands[i];
        0.5 * (va / vb + vb / va + dm * dm * (1.0 / va + 1.0 / vb) - 2.0)
    }).sum()
}

impl MaterialDatabase {
    /// Returns top-k matches sorted by ascending KL divergence.
    pub fn find_closest(&self, fp: &MaterialFingerprint, k: usize) -> Vec<(f32, &MaterialEntry)> {
        // Brute-force for 500 entries is < 0.1 ms; no need for approximate search at this scale
        let mut scored: Vec<(f32, &MaterialEntry)> = self.entries.iter()
            .map(|e| (spectral_kl_divergence(fp, &e.fingerprint), e))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);
        scored
    }
}
```

### FingerprinterTool (Editor)

`FingerprinterTool` in `vox_app/src/tools/fingerprinter.rs` is an egui side panel:

1. User selects a `SplatAssembly` entity in the scene hierarchy.
2. Clicking "Fingerprint" computes `MaterialFingerprint` from the assembly's splats: `bands[i] = mean of splat.spectral[i] (f16 → f32) across all splats in assembly`; `variance[i] = variance of same`.
3. Calls `MaterialDatabase::find_closest(fingerprint, 5)`.
4. Displays: "This material resembles: Concrete (KL=0.12), Stone (KL=0.18), Plaster (KL=0.31), ..."
5. "Apply Acoustic Profile" button: copies `MaterialEntry.acoustic` to the assembly's `AcousticProfileComponent` — setting reverb time, absorption, and diffusion for the SDF reverb system (Domain 4) automatically, without designer manual input.

### Runtime SpectralScanner

`SpectralScanner` is a held item (equippable entity component) usable during gameplay:

```rust
// vox_app/src/tools/spectral_scanner.rs

pub fn scanner_use(
    player_entity: EntityId,
    world: &World,
    res: &Resources,
) -> Option<ScanResult> {
    let camera = res.get::<RenderCamera>()?;
    let ray = camera.screen_center_ray();
    let hit = res.get::<PhysicsWorld>()?.cast_ray(ray, 50.0, /*filter*/)?;

    // Find the GaussianSplat closest to the hit point
    let hit_splat = res.get::<SplatBvh>()?.nearest_splat(hit.position)?;
    let spectral = hit_splat.spectral.map(|v| half::f16::from_bits(v).to_f32());

    let fp = MaterialFingerprint {
        bands:    spectral,
        variance: [0.0; 8],  // single-splat scan: no variance
    };
    let matches = res.get::<MaterialDatabase>()?.find_closest(&fp, 3);

    // Achievement integration
    let dominant_band = spectral.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i)?;
    res.get_mut::<AchievementSystem>()?
       .increment_band_discovery(dominant_band);

    Some(ScanResult { fingerprint: fp, matches, dominant_band })
}
```

### Webcam-Based Real Material Import

`MobileScanner` characterizes a real surface from a photograph:

1. **ColorChecker calibration:** User photographs an X-Rite ColorChecker Classic (24 patches with known spectral reflectances) under the target illuminant. `ColorCheckerCalibration::compute_sensor_matrix()` solves the least-squares system `M × camera_rgb = known_spectral_bands` (8×3 matrix, solved via `nalgebra::linalg::SVD`).
2. **Material import:** User photographs the target surface. Apply `M × pixel_rgb` per-pixel → approximate spectral reflectance for 8 bands. Average over the selected region → `MaterialFingerprint`. Export as a new `MaterialEntry`.

The sensor matrix `M` is a 8×3 matrix mapping camera RGB to spectral bands. It is camera-specific and illuminant-specific; the ColorChecker calibration step must be repeated if either changes.

---

## 10.3 Spectral Damage & Environmental Storytelling

### SpectralDamageComponent

```rust
// vox_data/src/damage/mod.rs

pub struct SpectralDamageComponent {
    pub base_spectral:    [f32; 8],   // original material spectral (read-only after spawn)
    pub current_spectral: [f32; 8],   // live value; written back to GaussianSplat.spectral
    pub damage_events:    Vec<DamageEvent>,
}

pub enum DamageEvent {
    Fire       { intensity: f32, elapsed: f32 },
    Water      { saturation: f32 },
    Radiation  { dose: f32 },
    Age        { elapsed_years: f32 },
    Oxidation  { rate: f32, elapsed: f32 },
}
```

### Damage Transfer Functions

Each `DamageEvent` variant defines a transfer function over the 8-band spectral array:

**Fire** — combustion carbon deposition (charring). Physically, charring increases broadband absorption while the remaining reflectance shifts toward longer wavelengths (thermal emission at high temperatures adds to bands 5–7):

```rust
DamageEvent::Fire { intensity, elapsed } => {
    let char_factor = (intensity * elapsed * 0.1).min(1.0);
    for i in 0..3 { s[i] *= 1.0 - char_factor; }          // destroy UV/blue bands
    for i in 3..5 { s[i] *= 1.0 - char_factor * 0.5; }    // partial loss in green
    for i in 5..8 { s[i] = s[i] * (1.0 - char_factor * 0.2) + char_factor * 0.4; }  // red/IR boost from ember glow
}
```

**Water** — wetting increases absorption coefficient (Kubelka-Munk theory; Berns et al. 2000). Wet surfaces also develop a thin-film specular component at blue wavelengths:

```rust
DamageEvent::Water { saturation } => {
    for i in 0..8 { s[i] *= 1.0 - saturation * 0.3; }   // darkening via absorption
    s[1] += saturation * 0.05;                            // thin-film blue specular
}
```

**Radiation** — ionizing radiation causes material fluorescence in UV band (band 0) and molecular damage that reduces inter-band correlation:

```rust
DamageEvent::Radiation { dose } => {
    let uv_boost = (dose * 0.01).min(0.8);
    s[0] = (s[0] + uv_boost).min(1.0);     // UV glow (visible in spectral viewport UV overlay)
    // Small random perturbation to all bands: radiation randomizes molecular structure
    for i in 1..8 {
        s[i] = (s[i] + (dose * 0.002) * pseudo_noise(i as f32)).clamp(0.0, 1.0);
    }
}
```

**Age** — photodegradation reduces inter-band variance (all colors drift toward grey); long exposure causes yellowing (Saleh & Teich 2007, photobleaching model):

```rust
DamageEvent::Age { elapsed_years } => {
    let grey_factor = (elapsed_years * 0.02).min(0.8);
    let grey = s.iter().sum::<f32>() / 8.0;
    for i in 0..8 { s[i] = s[i] * (1.0 - grey_factor) + grey * grey_factor; }
    // Yellowing: boost bands 4–6 at high age
    if elapsed_years > 20.0 {
        let yellow = ((elapsed_years - 20.0) * 0.005).min(0.3);
        for i in 4..7 { s[i] = (s[i] + yellow).min(1.0); }
    }
}
```

**Oxidation** — metal rusting shifts spectral from the flat metallic curve toward iron-oxide peaks at bands 5–6 (hematite ~610 nm peak):

```rust
DamageEvent::Oxidation { rate, elapsed } => {
    let rust_factor = (rate * elapsed).min(1.0);
    let rust_spectral = [0.05, 0.06, 0.08, 0.12, 0.25, 0.55, 0.60, 0.45_f32];
    for i in 0..8 {
        s[i] = s[i] * (1.0 - rust_factor) + rust_spectral[i] * rust_factor;
    }
}
```

### SpectralDamageSystem

```rust
// vox_data/src/damage/system.rs

pub struct SpectralDamageSystem;

impl System for SpectralDamageSystem {
    fn reads(&self)  -> Vec<TypeId> { vec![TypeId::of::<GaussianSplatRef>()] }
    fn writes(&self) -> Vec<TypeId> { vec![TypeId::of::<SpectralDamageComponent>()] }

    fn run(&self, world: &World, res: &Resources) {
        let dt = res.get::<Time>().unwrap().delta;
        Query::<(SpectralDamageComponent, GaussianSplatMut)>::new(world)
            .par_for_each(|(damage, splat_mut)| {
                let mut s = damage.base_spectral;
                for event in &mut damage.damage_events {
                    apply_damage_event(event, &mut s, dt);
                }
                damage.current_spectral = s;
                // Write back to the splat's spectral field
                splat_mut.spectral = s.map(|v| half::f16::from_f32(v).to_bits());
            });
    }
}
```

The system runs on the rayon threadpool via `par_for_each` (the ECS parallel query iterator). Each `DamageEvent::Fire { elapsed }` has its `elapsed` incremented by `dt` each frame; `DamageEvent::Age { elapsed_years }` increments by `dt / 31_557_600.0` (seconds per year). Events are removed when their effect is fully saturated (e.g. `char_factor >= 1.0`).

### Environmental Storytelling Examples

`SpectralNarrative` is a zero-code mechanism: designers do not place triggers. Instead they configure `DamageEvent` components on `SplatAssembly` entities in the editor, setting initial `elapsed` values to represent in-world history:

- **Burned dungeon (3-day fire):** `DamageEvent::Fire { intensity: 0.8, elapsed: 259_200.0 }` on all wall/floor assemblies. All surfaces render with charred warm-shift. No trigger code needed.
- **Haunted house (UV-active spirits):** `DamageEvent::Radiation { dose: 50.0 }` on select furnishing assemblies. `band[0]` elevated → these objects glow subtly in the spectral UV overlay. Invisible in normal RGB view; only players with a spectral viewer or high `spectral_affinity[0]` notice.
- **Ancient ruin (200-year age):** `DamageEvent::Age { elapsed_years: 200.0 }` on all stone assemblies. All surfaces rendered slightly grey with yellow undertone. No artist re-paint required.
- **Sunken ship (saltwater oxidation):** `DamageEvent::Oxidation { rate: 0.003, elapsed: 1_576_800.0 }` (50 years × rate) on all metal assemblies. All iron and steel surfaces render as deep rust-red.

---

## 10.4 Spectral Cinematography Tools

### SpectralCameraPreset

```rust
// vox_render/src/cinematography/camera_preset.rs

pub struct SpectralCameraPreset {
    pub name:             &'static str,
    pub sensor_response:  [f32; 8],   // per-band sensor gain (0.0–2.0; 1.0 = flat)
    pub lens_absorption:  [f32; 8],   // per-band transmission (0.0–1.0; 1.0 = clear)
    pub emulsion_curve:   AnimCurve,  // S-curve applied per band after sensor × absorption
}
```

The preset is applied in the final tonemap pass (`tonemap.wgsl`). The spectral accumulation buffer (8-band f16 per pixel) is multiplied element-wise by `sensor_response * lens_absorption`, then the `emulsion_curve` S-curve is applied per band before conversion to RGB via the Smits (1999) basis reconstruction:

```wgsl
// tonemap.wgsl (extension for preset support)
fn apply_preset(spectral: array<f32, 8>, preset: CameraPresetUniform) -> array<f32, 8> {
    var s = spectral;
    for (var i = 0u; i < 8u; i++) {
        s[i] = s[i] * preset.sensor_response[i] * preset.lens_absorption[i];
        s[i] = evaluate_curve(s[i], preset.emulsion_curve_coeffs[i]);
    }
    return s;
}
```

### Preset Library

| Preset | sensor_response | lens_absorption | Notes |
|--------|----------------|-----------------|-------|
| `StandardDigital` | `[1.0; 8]` | `[1.0; 8]` | Flat reference; equivalent to standard sRGB render |
| `VintageFilm` | `[0.6,0.8,1.0,1.1,1.2,1.3,1.2,0.9]` | `[0.7,0.8,0.9,1.0,1.0,1.0,0.9,0.7]` | Low UV/blue, warm mid, rolled-off near-IR; emulates 1970s Kodachrome |
| `NightVision` | `[0.0,0.0,0.0,0.0,0.0,0.0,0.5,2.0]` | `[1.0; 8]` | Only band 6–7 (near-IR) captured; renders as phosphor green in tonemap |
| `UltraViolet` | `[2.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0]` | `[0.9,0.5,0.1,0.0,0.0,0.0,0.0,0.0]` | UV-only sensor with UV-opaque lens (blocks visible); shows fluorescence |
| `FalseColor` | per-band distinct hues in tonemap | `[1.0; 8]` | Each band → distinct color channel; scientific visualization mode |

Swapping presets requires only updating the `CameraPresetUniform` buffer — zero re-lighting, zero asset re-export.

### SpectralTimeOfDay

Extension of the existing `illuminant_for_time(hour: f32) -> SpectralPowerDistribution` function:

```rust
// vox_render/src/cinematography/time_of_day.rs

pub fn illuminant_for_time(hour: f32) -> [f32; 8] {
    // Correlated Color Temperature varies from ~2000K (dawn/dusk) to ~6500K (noon)
    // using Bruneton & Neyret (2008) sky model parameterization
    let t = hour / 24.0;
    match hour {
        h if h < 5.5  => lerp(moonlight(), dawn_start(), (h / 5.5).powf(2.0)),
        h if h < 7.0  => lerp(dawn_start(), dawn_peak(), (h - 5.5) / 1.5),  // blue-pink
        h if h < 9.0  => lerp(dawn_peak(), d65_noon(),   (h - 7.0) / 2.0),
        h if h < 16.0 => d65_noon(),                                          // flat D65
        h if h < 18.0 => lerp(d65_noon(), golden_hour(), (h - 16.0) / 2.0),
        h if h < 19.5 => lerp(golden_hour(), dusk(),     (h - 18.0) / 1.5), // golden/red
        h if h < 21.0 => lerp(dusk(), night_sky(),        (h - 19.5) / 1.5),
        _             => night_sky(),
    }
}

fn dawn_start() -> [f32; 8] { [0.9, 0.95, 1.0, 0.85, 0.6, 0.4, 0.3, 0.2] }  // blue-pink
fn golden_hour() -> [f32; 8] { [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 0.9, 0.7] }   // golden orange
fn dusk() -> [f32; 8]        { [0.05, 0.1, 0.2, 0.35, 0.6, 0.8, 0.7, 0.4] } // purple-red
fn night_sky() -> [f32; 8]   { [0.001; 8].map(|v| v) }                        // ~D65 × 0.001
```

**Automatic audio correlation:** `SpectralAudioDriver` (see §10.6) listens to `AvSyncBand` values. Since `illuminant_for_time()` modulates the dominant bands, audio drivers respond automatically: dawn (high bands 1–2) triggers `AmbientSoundLayer::Dawn` (bird calls modulated by band-1 magnitude); night (all bands near-zero) triggers `AmbientSoundLayer::Night` (cricket synthesis driven by band-0 near-zero threshold).

### CinematicSequencer

`CinematicSequencer` stores a `Vec<PresetKeyframe>`:

```rust
pub struct PresetKeyframe {
    pub time_secs:  f32,
    pub preset:     SpectralCameraPreset,
    pub blend_mode: PresetBlendMode,
}

pub enum PresetBlendMode {
    Cut,
    LinearBlend { duration: f32 },
    SpectralDissolve { duration: f32 },  // blend band-by-band for chromatic separation effect
}
```

`SpectralDissolve` blends each of the 8 bands at a slightly different rate (band 0 transitions fastest, band 7 slowest), creating a chromatic dispersion effect as the camera preset changes — impossible in an RGB renderer where "the image" is a monolithic 3-channel value.

---

## 10.5 Spectral Procedural Generation

### PCG Spectral Nodes

The PCG graph system (Domain 3) is extended with spectral-aware node types. All nodes take a `SpectralTerrainSample { position: Vec3, spectral: [f32; 8] }` as input, sourced from the TerrainVolume's surface voxel spectral data.

**FilterBySpectralBand:**

```rust
pub struct FilterBySpectralBand {
    pub band:  usize,
    pub range: (f32, f32),  // [min, max] — points in range pass; others filtered
}

impl PcgNode for FilterBySpectralBand {
    fn evaluate(&self, points: Vec<PcgPoint>, ctx: &PcgContext) -> Vec<PcgPoint> {
        points.into_iter().filter(|p| {
            let band_val = ctx.terrain_spectral_at(p.position)[self.band];
            band_val >= self.range.0 && band_val <= self.range.1
        }).collect()
    }
}
```

Use case: `FilterBySpectralBand { band: 3, range: (0.0, 0.1) }` keeps only dark surfaces (low band-3 green reflectance) — place mushrooms and mosses exclusively on shadowed terrain with correct spectral appearance, without any artist-painted mask.

**ScatterBySpectralDensity:**

```rust
pub struct ScatterBySpectralDensity {
    pub band:     usize,
    pub assembly: AssetId,  // which SplatAssembly to scatter
}

impl PcgNode for ScatterBySpectralDensity {
    fn evaluate(&self, points: Vec<PcgPoint>, ctx: &PcgContext) -> Vec<PcgPoint> {
        points.into_iter().filter_map(|p| {
            let density = ctx.terrain_spectral_at(p.position)[self.band];
            // Poisson-disc sample probability proportional to density
            if ctx.rng.gen::<f32>() < density {
                Some(PcgPoint { assembly: self.assembly, ..p })
            } else {
                None
            }
        }).collect()
    }
}
```

`ScatterBySpectralDensity { band: 3, assembly: FOLIAGE }` places denser vegetation on greener (high band-3) terrain. `ScatterBySpectralDensity { band: 0, assembly: SNOW }` places snow where reflectance is brightest (high all-bands → high band-0). This is physically motivated scatter, not noise-based. No Unreal biome painting step required.

**SpectralBiomeTransition:**

Biome boundaries are defined by spectral gradient magnitude rather than arbitrary Voronoi or noise:

```rust
pub struct SpectralBiomeTransition {
    pub biome_a: BiomeId,
    pub biome_b: BiomeId,
    pub transition_band: usize,      // which band defines the boundary
    pub threshold: f32,              // transition center value
    pub width: f32,                  // soft transition width (both sides of threshold)
}
```

For a `Forest → Grassland` transition: `transition_band: 3` (green reflectance), `threshold: 0.4`, `width: 0.1`. Terrain points with `band_3 > 0.5` are fully forest; `band_3 < 0.3` fully grassland; `0.3..0.5` blends both biomes. The transition follows the actual spectral variation of the terrain — if a river valley has lower green reflectance due to wet soil, the biome boundary follows the valley naturally.

### SpectralWeather

```rust
// vox_core/src/weather/spectral_weather.rs

pub struct SpectralWeather {
    pub rain_rate:       f32,   // mm/hour
    pub snow_rate:       f32,   // mm/hour water equivalent
    pub fog_density:     f32,   // extinction coefficient
    pub dust_density:    f32,   // particles/m^3
}

impl System for SpectralWeatherSystem {
    fn run(&self, world: &World, res: &Resources) {
        let weather = res.get::<SpectralWeather>().unwrap();
        let terrain = res.get::<TerrainVolume>().unwrap();
        let dt = res.get::<Time>().unwrap().delta;

        if weather.rain_rate > 0.0 {
            // Find top-facing surface voxels (SDF gradient points up)
            // Apply Water damage event to their SplatAssembly at rate proportional to rain_rate
            // wet_rate = rain_rate * 0.01 * dt
            apply_weather_damage_to_surface(DamageEvent::Water {
                saturation: (weather.rain_rate * 0.01 * dt).min(1.0)
            }, world, &terrain, SurfaceDirection::UpFacing);
        }

        if weather.snow_rate > 0.0 {
            // Spawn new snow splats on top-facing surfaces
            // Snow spectral: high reflectance all bands [0.9, 0.88, 0.85, 0.83, 0.80, 0.78, 0.75, 0.72]
            spawn_weather_splats(SNOW_SPECTRAL, weather.snow_rate, world, &terrain);
        }

        if weather.fog_density > 0.0 {
            // Modify volumetric scattering uniforms:
            // increase bands 0–2 (Rayleigh scatter: blue-preferential)
            let vol = res.get_mut::<VolumetricScatterUniforms>().unwrap();
            vol.band_scatter[0] += weather.fog_density * 0.8 * dt;
            vol.band_scatter[1] += weather.fog_density * 0.5 * dt;
            vol.band_scatter[2] += weather.fog_density * 0.3 * dt;
        }

        if weather.dust_density > 0.0 {
            // Spawn dust splats near camera in bands 4–6 (brown ~580–640nm peak)
            let dust_spectral = [0.05, 0.08, 0.12, 0.20, 0.45, 0.60, 0.55, 0.35_f32];
            spawn_dust_volume(dust_spectral, weather.dust_density, world);
        }
    }
}
```

---

## 10.6 Spectral Audio-Visual Synchronization

### AvSyncBand Computation

`SpectralAVSync` runs as a GPU compute pass immediately after the splat rasterize pass, before the tonemap. It reads the spectral accumulation buffer (the per-pixel 8-band f16 buffer output by `splat_rasterize.wgsl`) and computes a coverage-weighted average of each band across all rendered pixels:

```wgsl
// avsync_reduce.wgsl

@group(0) @binding(0) var spectral_buffer: texture_storage_2d<rgba16float, read>;  // bands 0-3
@group(0) @binding(1) var spectral_buffer_b: texture_storage_2d<rgba16float, read>; // bands 4-7
@group(0) @binding(2) var<storage, read_write> av_sync_out: AvSyncOutput;

struct AvSyncOutput { bands: array<f32, 8>, pixel_count: u32 }

@compute @workgroup_size(16, 16)
fn av_sync_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Two-pass parallel reduction: first workgroup-local sums, then atomic add to global
    var local_sum: array<f32, 8>;
    let coords = gid.xy;
    let s_a = textureLoad(spectral_buffer, coords, 0);
    let s_b = textureLoad(spectral_buffer_b, coords, 0);
    for (var i = 0u; i < 4u; i++) { local_sum[i] += s_a[i]; }
    for (var i = 0u; i < 4u; i++) { local_sum[i+4u] += s_b[i]; }
    // ... workgroup reduction, then atomicAdd to av_sync_out.bands
}
```

Result is read back to CPU as `AvSyncBand [8]` each frame, mapped to `[0, 1]` by the maximum possible band value. Total computation time < 0.5 ms at 4K (hardware reduction is O(N/workgroup_size²) with two-pass reduction).

### SpectralAudioDriver

```rust
// vox_render/src/avsync/audio_driver.rs

pub struct SpectralAudioDriver {
    pub band_map: [(usize, AudioDriverTarget); 8],
}

pub enum AudioDriverTarget {
    AmbientTone    { freq_hz: f32, max_gain: f32 },
    AmbientLayer   { layer_id: SoundLayerId, max_gain: f32 },
    ReverbAmount   { max_reverb: f32 },
    SubBassRumble  { max_gain: f32 },
    TransientCheck { threshold_delta: f32, event: SoundEventId },
}

impl SpectralAudioDriver {
    pub fn update(&mut self, av_sync: &[f32; 8], audio: &mut SpatialAudioManager) {
        for (band, target) in &self.band_map {
            let v = av_sync[*band];
            match target {
                AudioDriverTarget::AmbientTone { freq_hz, max_gain } =>
                    audio.set_ambient_tone_gain(*freq_hz, v * max_gain),
                AudioDriverTarget::SubBassRumble { max_gain } =>
                    audio.set_sub_bass_gain(v * max_gain),
                AudioDriverTarget::TransientCheck { threshold_delta, event } => {
                    if (v - self.prev_av[*band]).abs() > *threshold_delta {
                        audio.play_event(*event, glam::Vec3::ZERO);
                    }
                }
                // ...
            }
        }
        self.prev_av = *av_sync;
    }
}
```

Default band mappings:
- Band 0 (UV) → `AmbientTone { freq_hz: 8000.0 }` — electric shimmer/crystalline hum
- Band 3 (green/mid) → `AmbientLayer { layer_id: ROOM_PRESENCE }` — room warmth, body, presence
- Band 7 (near-IR) → `SubBassRumble` — earth, weight, power
- Delta on any band > 0.3 per frame → `TransientCheck` → `SpectralImpactEvent` — sudden scene change emits a spectral impact sound via `vox_audio::synthesize_impact`

### SpectralReactiveMusicLayer

```rust
// vox_render/src/avsync/music_reactive.rs

pub struct SpectralReactiveMusicLayer {
    pub target_band:     usize,
    pub frequency_range: (f32, f32),  // audio frequency range this layer covers
    pub track_id:        MusicTrackId,
}

impl SpectralReactiveMusicLayer {
    pub fn update(&self, av_sync: &[f32; 8], audio: &mut SpatialAudioManager) {
        let volume = av_sync[self.target_band];
        audio.set_music_layer_volume(self.track_id, volume);
    }
}
```

A game can define 8 music layers (one per spectral band) each covering a different frequency range of the music mix. As the scene's visual spectral composition shifts — fire making everything warm, rain making everything cool, UV sources flickering — the music mix responds automatically without any game code. This is a real-time audio-visual synchronization mechanism with no Unreal analogue.

**SpectralFeedback Loop:** for experimental/art games, `SpectralFeedbackSystem` allows player input to modulate `SplatAssembly.spectral` directly. These changes feed back into `AvSyncBand` next frame → `SpectralAudioDriver` responds → audio changes → player hears result. The environment becomes an instrument. Implementation: `ActionId::ModulateSpectralBand(band, delta)` binding reads held-item `SpectralModulator { band, gain }` component; applies delta to targeted assembly bands each frame while held.

---

## 10.7 Photogrammetry-Native Pipeline

### Import Pipeline

The native import path from standard 3DGS tools:

```
3DGS training (COLMAP + gaussian-splatting) → output .ply
→ vox_data::import::import_3dgs_ply()
    → SpectralUplifter::uplift_batch()        (50K splats < 1 sec)
    → SpectralInpainter::fix_ir_band()
    → MaterialFingerprint per assembly
    → MaterialDatabase::find_closest() → AcousticProfile auto-assign
→ WorldCell { splats: Vec<GaussianSplat>, ... }
→ serialize to .vxm (Domain 2 binary format)
```

The entire import is triggered from the editor via "Import 3DGS PLY" in the asset browser. Progress is reported via an egui progress bar bound to a `rayon::ThreadPool` completion counter.

### OchoraCapture

`OchoraCapture` (separate binary in `crates/ochroma_capture/`) is the end-to-end photogrammetry-to-game-content pipeline:

```
Mobile device records video (HEVC, 4K, 60fps)
→ OchoraCapture::upload_to_endpoint(frames, session_id)
    → HTTPS POST to training server (local or cloud GPU)
    → Server runs COLMAP for camera poses (SfM)
    → Server trains 3DGS for 20K iterations (~25 min on RTX 3090)
    → Server exports .ply
→ OchoraCapture::download_and_import(session_id, project_path)
    → download .ply → import_3dgs_ply() → save as WorldCell .vxm
→ Editor hot-reloads WorldCell → visible in scene immediately
```

Total pipeline: < 30 minutes from capture to in-engine content. The training server is configurable (local GPU workstation, cloud VM, or a dedicated Ochroma cloud service endpoint). Authentication is handled via project API key stored in `~/.ochroma/config.toml`.

### DifferentiableRendering (Long-Term Research Track)

The EWA tile accumulation loop in `spectra_render.rs` is differentiable with respect to splat parameters `(position, scale, rotation, opacity, spectral)` because alpha-composite blending is smooth and differentiable at all points except exactly-zero-alpha splats. This means gradient-based scene optimization is theoretically possible post-import:

```rust
// Research prototype — not in M10 scope
pub fn optimize_scene(
    splats: &mut Vec<GaussianSplat>,
    reference_images: &[ReferenceView],
    iterations: u32,
) {
    // For each iteration:
    // 1. Render current splats with spectra_render (differentiable path)
    // 2. Compute L1 + SSIM loss vs reference
    // 3. Backprop through EWA accumulation (requires reverse-mode AD)
    //    — burn crate provides autodiff on CPU tensors
    // 4. Update splat parameters by Adam optimizer
}
```

This is a research track; the `burn` crate's autodiff would need to be plumbed through the EWA accumulator. Target milestone: proof-of-concept in Domain 12 (post-launch research).

### Spectral PLY Extension Format

Ochroma proposes `.spectral_ply` as an open format extension for photogrammetric captures with native spectral sensors (multispectral cameras, spectroradiometer arrays):

**Format:** Standard PLY binary (little-endian) with additional vertex properties:

```
property float spectral_0    // ~380nm
property float spectral_1    // ~430nm
property float spectral_2    // ~480nm
property float spectral_3    // ~530nm
property float spectral_4    // ~570nm
property float spectral_5    // ~610nm
property float spectral_6    // ~650nm
property float spectral_7    // ~700nm
```

A conforming reader that does not understand spectral properties ignores them (standard PLY forward-compatibility). A conforming writer from standard 3DGS uses the `SpectralUplifter` to populate these fields from RGB splat color. The format registration will be proposed to the Open3D and standard 3DGS communities.

---

## File Map

```
crates/
  vox_data/
    src/
      uplift/
        mod.rs              — SpectralUplifter, UpliftInput, MaterialClass
        onnx_runner.rs      — tract-onnx model loading and inference wrapper
        inpainter.rs        — SpectralInpainter, foliage/metal detection
      fingerprint/
        mod.rs              — MaterialFingerprint, MaterialDatabase, MaterialEntry,
                              MaterialCategory, AcousticProfile auto-assign
        kl_divergence.rs    — spectral_kl_divergence(), MaterialDatabase::find_closest()
        webcam_import.rs    — MobileScanner, ColorCheckerCalibration
      damage/
        mod.rs              — SpectralDamageComponent, DamageEvent
        system.rs           — SpectralDamageSystem (ECS System impl)
        transfer_fns.rs     — apply_damage_event() for all DamageEvent variants
        narrative.rs        — SpectralNarrative, NarrativeCheckpoint
      import/
        ply_import.rs       — import_3dgs_ply(), classify_material(), spectral PLY extension

  vox_render/
    src/
      cinematography/
        camera_preset.rs    — SpectralCameraPreset, AnimCurve, preset library constants
        time_of_day.rs      — illuminant_for_time() extension, all illuminant functions
        sequencer.rs        — PresetKeyframe, PresetBlendMode, CinematicSequencer integration
      avsync/
        mod.rs              — SpectralAVSync, AvSyncBand
        audio_driver.rs     — SpectralAudioDriver, AudioDriverTarget
        music_reactive.rs   — SpectralReactiveMusicLayer, SpectralFeedbackSystem
      shaders/
        avsync_reduce.wgsl  — GPU compute: spectral buffer → AvSyncBand[]
        tonemap.wgsl        — extension: apply_preset(), SpectralCameraPreset uniform

  vox_core/
    src/
      pcg/
        spectral_nodes.rs   — FilterBySpectralBand, ScatterBySpectralDensity,
                              SpectralBiomeTransition, PcgNode impls
      weather/
        mod.rs
        spectral_weather.rs — SpectralWeather, SpectralWeatherSystem, weather damage events

  vox_app/
    src/
      tools/
        spectral_scanner.rs — SpectralScanner item logic, scan_result UI
        fingerprinter.rs    — FingerprinterTool egui panel, acoustic auto-assign button
      capture/
        ochroma_capture.rs  — OchoraCapture pipeline, upload/download, progress bar

  crates/
    ochroma_capture/        — Standalone binary for mobile video capture pipeline
      src/
        main.rs
        upload.rs
        session.rs

  assets/
    models/
      uplift/
        diffuse.onnx
        metallic.onnx
        translucent.onnx
        emissive.onnx
    data/
      material_database.bin  — bincode-serialized Vec<MaterialEntry> (500 entries)
```

---

## Milestones

### M10.1 — Neural Uplifter (2 days)
- `SpectralUplifter` struct with 4 `tract-onnx` model slots
- ONNX model loading from asset path; inference for single `UpliftInput`
- `uplift_batch()` using `rayon::par_iter`
- `SpectralInpainter` with foliage/metal/general cases
- Performance test: 50K splat batch completes in < 1 sec (CI benchmark)
- ONNX model files included in repo under `assets/models/uplift/` (pre-trained weights from Python training script in `tools/train_uplifter/`)

### M10.2 — 3DGS PLY Import (1 day)
- `import_3dgs_ply()` full pipeline: PLY parse (ply-rs crate) → `RawSplat3dgs` → uplifter → `GaussianSplat`
- `classify_material()` heuristic from RGB + SH coefficients
- Integration test: import a 10K-splat PLY; verify all splats have non-zero spectral[7]
- Spectral PLY extension: writer adds `spectral_0..spectral_7` properties; round-trip test

### M10.3 — Material Fingerprinting (1.5 days)
- `MaterialFingerprint`, `MaterialDatabase` with `kdtree` integration
- `spectral_kl_divergence()` implementation and validation against known materials
- `MaterialDatabase::find_closest()` top-k matching
- `MaterialDatabase` pre-population from reference dataset (500 entries)
- `FingerprinterTool` egui panel: assembly → fingerprint → top-5 matches → acoustic auto-assign
- Performance test: top-5 from 500 entries in < 0.1 ms

### M10.4 — Runtime SpectralScanner (0.5 days)
- `SpectralScanner` item: raycast → nearest splat → fingerprint → match → `ScanResult` UI
- Achievement integration: `increment_band_discovery(dominant_band)` on scan
- Integration test: scan all 8 types of assembly → `DiscoverAllBands` achievement unlocks

### M10.5 — Webcam Import (1 day)
- `ColorCheckerCalibration::compute_sensor_matrix()` via `nalgebra` SVD
- `MobileScanner` webcam photo → spectral profile (single region average)
- Integration test: synthetic test with known illuminant + known material → output bands within 0.1 of ground truth

### M10.6 — Spectral Damage System (2 days)
- `SpectralDamageComponent`, all `DamageEvent` variants
- All transfer functions: Fire, Water, Radiation, Age, Oxidation
- `SpectralDamageSystem` parallel ECS update; `write_back` to `GaussianSplat.spectral`
- Performance test: 100K damaged splats update in < 1 ms
- Environmental storytelling test: burned dungeon scene → visual diff confirms warm shift; ancient ruin → grey-yellow shift

### M10.7 — Spectral Cinematography (1.5 days)
- `SpectralCameraPreset` struct; all 5 preset constants
- `apply_preset()` in `tonemap.wgsl`; `CameraPresetUniform` wgpu buffer
- `illuminant_for_time()` full day/night cycle with all lighting phases (Bruneton model)
- `CinematicSequencer` `PresetKeyframe` list + `SpectralDissolve` blend mode
- Editor: camera preset picker dropdown in viewport toolbar; live preview

### M10.8 — Spectral PCG Nodes (1 day)
- `FilterBySpectralBand`, `ScatterBySpectralDensity`, `SpectralBiomeTransition` PCG nodes
- Integration with Domain 3 PCG graph executor
- Integration test: forest/grassland biome transition follows spectral gradient, not noise

### M10.9 — Spectral Weather (1 day)
- `SpectralWeather` resource; `SpectralWeatherSystem`
- Rain → Water damage on surface splats; snow → new snow splats; fog → volumetric scatter; dust → near-camera splats
- Integration test: rain for 60 simulated seconds → surface splat band values demonstrably lower than dry baseline

### M10.10 — SpectralAVSync + Audio Driver (1.5 days)
- `avsync_reduce.wgsl` GPU compute pass; `AvSyncBand` readback
- `SpectralAudioDriver` with default 8-band→audio mappings
- `SpectralReactiveMusicLayer` volume driver
- `SpectralFeedbackSystem` player modulation loop
- Integration test: splat scene with high band-7 → SubBassRumble gain > 0.5; sudden band change → TransientCheck fires impact sound

### M10.11 — OchoraCapture Pipeline (2 days)
- `ochroma_capture` binary: video upload, session polling, PLY download, auto-import
- Progress bar UI in editor
- HTTPS client via `reqwest` crate; async via `tokio`
- End-to-end test: mock server returns a pre-trained PLY; verify import completes and cell appears in scene

**Total estimated effort: ~15 developer-days**

---

## Acceptance Criteria

1. **Uplifter performance:** `uplift_batch(&inputs)` with 50,000 `UpliftInput` entries completes in < 1 second on a 4-core host (verified in CI via `cargo bench`).
2. **Uplifter quality:** on the held-out spectroradiometer test set, per-band RMSE < 0.05 for all 4 material class networks.
3. **Foliage inpainting:** a grass/leaf splat assembly has `mean_spectral[7] > mean_spectral[5] * 1.5` after inpainting (chlorophyll red-edge effect present).
4. **Fingerprint matching:** scanning a `MaterialEntry` assembly returns that entry as the top match (KL divergence rank-1 correct) for all 500 pre-populated materials.
5. **Fingerprint performance:** `MaterialDatabase::find_closest(fp, 5)` completes in < 0.1 ms for any query fingerprint.
6. **Fire damage correctness:** a `DamageEvent::Fire { intensity: 1.0, elapsed: 3600.0 }` reduces `band[0]` by at least 0.5 and increases `band[7]` by at least 0.2 relative to an unburned baseline of `[0.6; 8]`.
7. **Age damage correctness:** `DamageEvent::Age { elapsed_years: 200.0 }` on a colorful material (`bands` with variance > 0.1) produces output where inter-band variance is reduced by at least 60% and `mean(bands[4..7]) > mean(bands[0..3])` (yellowing).
8. **Camera preset:** switching from `StandardDigital` to `NightVision` preset produces a render where only bands 6–7 contribute to the final image (band 0–5 pixel contributions are < 0.01 after tonemap).
9. **AvSyncBand correctness:** a scene with a single high band-7 assembly covering 80% of screen area produces `AvSyncBand[7] > 0.7` in the readback buffer.
10. **SpectralReactiveMusicLayer:** a music layer targeting band 3 reports volume `> 0.8` when `AvSyncBand[3] > 0.9`; reports volume `< 0.1` when `AvSyncBand[3] < 0.05`.
11. **Weather damage:** 60 seconds of simulated rain at `rain_rate = 10.0` produces measurable reduction (> 5%) in top-facing surface splat band values relative to pre-rain baseline.
12. **PCG scatter:** `ScatterBySpectralDensity { band: 3, assembly: FOLIAGE }` produces at least 2× more scatter points on terrain regions with `band_3 > 0.6` than on regions with `band_3 < 0.2` (statistically verified over 10 random terrain seeds).

---

## Effort Summary

| Milestone | Scope | Days |
|-----------|-------|------|
| M10.1 | Neural uplifter + inpainter | 2.0 |
| M10.2 | 3DGS PLY import + spectral PLY format | 1.0 |
| M10.3 | Material fingerprinting + database | 1.5 |
| M10.4 | Runtime SpectralScanner | 0.5 |
| M10.5 | Webcam material import | 1.0 |
| M10.6 | Spectral damage system | 2.0 |
| M10.7 | Spectral cinematography | 1.5 |
| M10.8 | Spectral PCG nodes | 1.0 |
| M10.9 | Spectral weather | 1.0 |
| M10.10 | SpectralAVSync + audio driver | 1.5 |
| M10.11 | OchoraCapture pipeline | 2.0 |
| **Total** | | **15.0 days** |

Risk factors: `tract-onnx` ONNX opset compatibility — verify the Python export `opset_version` matches what `tract` supports before committing the training pipeline. The `avsync_reduce.wgsl` readback has a 1-frame GPU→CPU latency; ensure the audio driver accounts for this (acceptable lag: 16.7 ms at 60 Hz). Webcam import quality depends on ColorChecker availability; ship a synthetic calibration fallback using the D65 observer as default sensor matrix.
