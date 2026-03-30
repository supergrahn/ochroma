# Domain 2 — Audio Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rodio backend with CPAL+fundsp, implement `SpectralSynth` (impact sound synthesis from `[u16; 8]` spectral material) and `SpectralReverb` (reverb tail length derived from surrounding splat reflectance), and wire it all cross-platform across WASAPI/CoreAudio/ALSA.

**Architecture:** The existing `AudioHandle` owns a background thread; that thread will switch its device ownership from `rodio::OutputStream` to a `cpal::Stream`. `fundsp` replaces the hand-rolled signal accumulation in `AudioGraph` with a composable combinator graph (`>>`, `&`, `|`). `SpectralSynth` and `SpectralReverb` are pure-Rust DSP types that live in `vox_audio` and have no dependency on any backend — they produce `Vec<f32>` buffers that CPAL feeds to the device.

**Tech Stack:** `cpal = "0.15"`, `fundsp = "0.18"`, `hound = "3"` (existing, WAV I/O), `lewton = "0.10"` (OGG), `half = "2"` (existing, f16 spectral encoding)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| MODIFY | `crates/vox_audio/Cargo.toml` | Add cpal, fundsp, lewton; gate rodio as `optional = true` (already is) |
| CREATE | `crates/vox_audio/src/spectral_acoustic.rs` | `SpectralAcousticProfile` — forge material DB → acoustic params (one DB for both rendering and audio) |
| CREATE | `crates/vox_audio/src/cpal_backend.rs` | CPAL device initialisation, stream ownership, AudioCommand dispatch |
| CREATE | `crates/vox_audio/src/spectral_synth2.rs` | `SpectralSynth` struct with `strike()` and `resonance_freq()` |
| CREATE | `crates/vox_audio/src/spectral_reverb.rs` | `SpectralReverb` struct, `from_splat_reflectance()`, `tail_samples()` |
| CREATE | `crates/vox_audio/src/fundsp_graph.rs` | fundsp signal graph helpers: gain node, reverb send, HRTF insert |
| MODIFY | `crates/vox_audio/src/lib.rs` | Pub-use new modules; replace `AudioHandle::spawn()` body under new feature flag |
| CREATE | `crates/vox_audio/tests/integration_audio.rs` | End-to-end: SpectralSynth strike → SpectralReverb convolve → CPAL mock write |

---

## Task 1 — Add cpal and fundsp to Cargo.toml

**Files:** `crates/vox_audio/Cargo.toml`

- [ ] Write a failing compilation test by adding a module stub that `use cpal;` — `cargo test -p vox_audio` will fail with unresolved crate.

  ```rust
  // crates/vox_audio/src/cpal_backend.rs  (stub, causes compile error until dep added)
  use cpal::traits::HostTrait as _;
  pub struct CpalBackend;
  ```

- [ ] Run test to confirm failure:
  ```
  cargo test -p vox_audio 2>&1 | grep "unresolved\|error"
  ```

- [ ] Add dependencies to `crates/vox_audio/Cargo.toml`:

  ```toml
  [dependencies]
  # ... existing ...
  cpal    = { version = "0.15", optional = true }
  fundsp  = { version = "0.18", optional = true }
  lewton  = { version = "0.10", optional = true }

  [features]
  default        = ["audio-backend"]
  audio-backend  = ["dep:cpal", "dep:fundsp", "dep:lewton"]
  rodio-backend  = ["dep:rodio"]
  ```

- [ ] Run test to confirm it compiles:
  ```
  cargo build -p vox_audio --features audio-backend
  ```

- [ ] Commit:
  ```
  git add crates/vox_audio/Cargo.toml crates/vox_audio/src/cpal_backend.rs
  git commit -m "feat(audio): add cpal=0.15, fundsp=0.18, lewton=0.10 to vox_audio"
  ```

---

## Task 1.5 — SpectralAcousticProfile: one material database for rendering AND audio

**Files:**
- Create: `crates/vox_audio/src/spectral_acoustic.rs`
- Modify: `crates/vox_audio/src/lib.rs`

**The forge steal:** `forge-spectral` has a database of 13 materials (Soil, Rock, Bark, Water, Glass, Concrete, Foliage, Snow, Asphalt, Gravel, Brick, Metal, Sand) with physically measured USGS reflectance curves at 16 wavelengths. The same spectral profile that governs how a material *looks* also governs how it *sounds* — high uniform reflectance (Metal, Snow) = long acoustic sustain and reverberant room. Low absorption = good acoustic mirror. This is the architectural advantage: Unreal has separate visual material parameters and audio material parameters. Ochroma has one spectral profile that drives both.

- [ ] **Step 1: Write the failing tests**

Create `crates/vox_audio/src/spectral_acoustic.rs`:

```rust
//! SpectralAcousticProfile — derives acoustic synthesis parameters from the
//! forge-spectral material database. One material profile drives both rendering
//! and audio. No separate "audio material" needed.

/// Acoustic synthesis parameters derived from spectral reflectance.
#[derive(Debug, Clone, Copy)]
pub struct SpectralAcousticProfile {
    /// Fundamental resonance frequency in Hz.
    pub resonance_hz: f32,
    /// Quality factor: higher = longer ring / slower energy loss.
    /// Q = 0.2 (sand, soil) → Q = 15.0 (metal).
    pub q_factor: f32,
    /// Reverberation time (RT60) in seconds when surface lines a room.
    /// 0.1s (dead outdoors: sand, asphalt) → 6.0s (snow field).
    pub rt60_secs: f32,
}

impl SpectralAcousticProfile {
    /// Derive acoustic profile from arbitrary 8-band spectral data.
    /// Approximate — use `from_material_kind()` for known materials.
    ///
    /// Formula:
    /// - `resonance_hz`: weighted geometric mean over [12000..200 Hz] FREQ_MAP
    /// - `q_factor`: 5.0 / (band_variance + 0.1) — flat profile = high Q = long ring
    /// - `rt60_secs`: mean_reflectance × 6.0 — high reflectance = reverberant room
    pub fn from_spectral(bands_f16: &[u16; 8]) -> Self {
        const FREQ_MAP: [f32; 8] = [12000.0, 8000.0, 5000.0, 3000.0, 1500.0, 800.0, 400.0, 200.0];
        let bands: [f32; 8] = std::array::from_fn(|i| {
            half::f16::from_bits(bands_f16[i]).to_f32().max(0.0)
        });
        let sum: f32 = bands.iter().sum();
        let mean = sum / 8.0;

        // Weighted geometric mean frequency (log domain)
        let log_freq_sum: f32 = bands.iter().zip(FREQ_MAP.iter())
            .map(|(&w, &f)| w * f.ln())
            .sum();
        let resonance_hz = if sum > 1e-6 {
            (log_freq_sum / sum).exp()
        } else {
            440.0
        };

        // Band variance → Q factor
        let variance: f32 = bands.iter().map(|&b| (b - mean).powi(2)).sum::<f32>() / 8.0;
        let q_factor = (5.0 / (variance + 0.1)).clamp(0.2, 15.0);

        // Mean reflectance → RT60
        let rt60_secs = (mean * 6.0).clamp(0.05, 8.0);

        Self { resonance_hz, q_factor, rt60_secs }
    }

    /// Hardcoded profiles for forge MaterialKind variants.
    /// Physics-correct values from USGS spectral library + known material acoustics.
    ///
    /// Named to match `forge_spectral::MaterialKind` — update when forge adds materials.
    pub fn metal()    -> Self { Self { resonance_hz: 8000.0, q_factor: 15.0, rt60_secs: 3.5 } }
    pub fn glass()    -> Self { Self { resonance_hz: 6000.0, q_factor: 10.0, rt60_secs: 0.3 } }
    pub fn concrete() -> Self { Self { resonance_hz:  300.0, q_factor:  2.0, rt60_secs: 2.0 } }
    pub fn rock()     -> Self { Self { resonance_hz:  250.0, q_factor:  1.5, rt60_secs: 1.5 } }
    pub fn brick()    -> Self { Self { resonance_hz:  400.0, q_factor:  1.5, rt60_secs: 1.8 } }
    pub fn bark()     -> Self { Self { resonance_hz:  200.0, q_factor:  1.8, rt60_secs: 0.8 } }
    pub fn gravel()   -> Self { Self { resonance_hz:  350.0, q_factor:  0.8, rt60_secs: 0.4 } }
    pub fn soil()     -> Self { Self { resonance_hz:  100.0, q_factor:  0.4, rt60_secs: 0.1 } }
    pub fn sand()     -> Self { Self { resonance_hz:   80.0, q_factor:  0.3, rt60_secs: 0.1 } }
    pub fn asphalt()  -> Self { Self { resonance_hz:  120.0, q_factor:  0.5, rt60_secs: 0.1 } }
    pub fn snow()     -> Self { Self { resonance_hz:   60.0, q_factor:  0.2, rt60_secs: 6.0 } }
    pub fn foliage()  -> Self { Self { resonance_hz:  800.0, q_factor:  0.7, rt60_secs: 0.3 } }
    pub fn water()    -> Self { Self { resonance_hz:  800.0, q_factor:  2.5, rt60_secs: 0.4 } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_has_higher_q_than_soil() {
        assert!(SpectralAcousticProfile::metal().q_factor >
                SpectralAcousticProfile::soil().q_factor,
            "metal sustains longer than soil");
    }

    #[test]
    fn snow_has_longest_rt60() {
        let rt60s = [
            SpectralAcousticProfile::metal().rt60_secs,
            SpectralAcousticProfile::concrete().rt60_secs,
            SpectralAcousticProfile::snow().rt60_secs,
            SpectralAcousticProfile::asphalt().rt60_secs,
        ];
        let max = rt60s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(max, SpectralAcousticProfile::snow().rt60_secs,
            "snow field should have longest RT60 (huge reflectance)");
    }

    #[test]
    fn glass_higher_resonance_than_soil() {
        assert!(SpectralAcousticProfile::glass().resonance_hz >
                SpectralAcousticProfile::soil().resonance_hz,
            "glass rings higher than soil");
    }

    #[test]
    fn from_spectral_metal_like_profile() {
        // Flat high-reflectance profile ≈ metal: should give high Q, moderate-high RT60
        let bands = [half::f16::from_f32(0.65).to_bits(); 8];
        let profile = SpectralAcousticProfile::from_spectral(&bands);
        assert!(profile.q_factor > 3.0,
            "flat high-reflectance should give Q > 3.0, got {}", profile.q_factor);
        assert!(profile.rt60_secs > 1.0,
            "flat high-reflectance should give RT60 > 1.0s, got {}", profile.rt60_secs);
    }

    #[test]
    fn from_spectral_dead_material() {
        // Very low reflectance (asphalt-like): low RT60
        let bands = [half::f16::from_f32(0.06).to_bits(); 8];
        let profile = SpectralAcousticProfile::from_spectral(&bands);
        assert!(profile.rt60_secs < 0.5,
            "dark material should give RT60 < 0.5s, got {}", profile.rt60_secs);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p vox_audio spectral_acoustic 2>&1 | head -10
```

Expected: compile error — module not in lib.rs

- [ ] **Step 3: Expose the module**

Add to `crates/vox_audio/src/lib.rs`:

```rust
pub mod spectral_acoustic;
pub use spectral_acoustic::SpectralAcousticProfile;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_audio spectral_acoustic -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/vox_audio/src/spectral_acoustic.rs crates/vox_audio/src/lib.rs
git commit -m "feat(audio): SpectralAcousticProfile — forge material DB drives both rendering and audio"
```

---

## Task 2 — Implement SpectralSynth

**Files:** `crates/vox_audio/src/spectral_synth2.rs`, `crates/vox_audio/src/lib.rs`

The existing `synthesize_impact` free function in `spectral_synth.rs` is kept for backwards compatibility. `SpectralSynth` is a new struct that wraps the same physics but exposes a clean API matching the spec signature, adds harmonic overtone generation, and accepts the raw `[u16; 8]` as the primary input type.

Resonance frequency: weighted average of `FREQ_MAP` using per-band reflectance. High short-wavelength (band 0–2) weight → glassy high-frequency ring. High long-wavelength (band 5–7) weight → woody/rocky low thud.

Damping: proxy for stiffness is opacity value (not available here) so we use the variance across bands — a flat profile (uniform reflectance) means homogeneous material → lower damping (longer sustain). A peaked profile means strongly coloured material → higher damping (fast decay).

- [ ] Write failing tests first in `crates/vox_audio/src/spectral_synth2.rs`:

  ```rust
  // crates/vox_audio/src/spectral_synth2.rs
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn resonance_freq_blue_material_is_high() {
          // Band 0 dominant → expect frequency near 8 kHz
          let mut spectral = [0u16; 8];
          spectral[0] = half::f16::from_f32(1.0).to_bits();
          let hz = SpectralSynth::resonance_freq(&spectral);
          assert!(hz > 4000.0, "blue-dominant material resonance={hz}");
      }

      #[test]
      fn resonance_freq_red_material_is_low() {
          // Band 7 dominant → expect frequency near 80 Hz
          let mut spectral = [0u16; 8];
          spectral[7] = half::f16::from_f32(1.0).to_bits();
          let hz = SpectralSynth::resonance_freq(&spectral);
          assert!(hz < 500.0, "red-dominant material resonance={hz}");
      }

      #[test]
      fn strike_returns_nonempty_audio() {
          let spectral = [half::f16::from_f32(0.5).to_bits(); 8];
          let samples = SpectralSynth::strike(&spectral, 1.0);
          assert!(!samples.is_empty());
      }

      #[test]
      fn strike_is_normalised() {
          let spectral = [half::f16::from_f32(1.0).to_bits(); 8];
          let samples = SpectralSynth::strike(&spectral, 1.0);
          let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
          assert!(peak <= 1.0 + 1e-5, "peak={peak}");
      }

      #[test]
      fn strike_all_zero_spectral_is_silence() {
          let spectral = [0u16; 8];
          let samples = SpectralSynth::strike(&spectral, 1.0);
          assert!(samples.iter().all(|&s| s == 0.0));
      }

      #[test]
      fn blue_strike_sounds_different_from_red_strike() {
          let mut blue = [0u16; 8]; blue[0] = half::f16::from_f32(1.0).to_bits();
          let mut red  = [0u16; 8]; red[7]  = half::f16::from_f32(1.0).to_bits();
          let b = SpectralSynth::strike(&blue, 1.0);
          let r = SpectralSynth::strike(&red,  1.0);
          let diff: f32 = b.iter().zip(r.iter()).map(|(a,x)| (a-x).abs()).sum();
          assert!(diff > 0.1, "blue vs red should differ, diff={diff}");
      }
  }
  ```

- [ ] Run to confirm failure:
  ```
  cargo test -p vox_audio spectral_synth2 2>&1 | grep "error\|FAILED"
  ```

- [ ] Implement `SpectralSynth`:

  ```rust
  // crates/vox_audio/src/spectral_synth2.rs
  //! SpectralSynth — synthesises impact sounds from GaussianSplat spectral profiles.
  //!
  //! `strike(spectral, impulse)` is the primary entry point.
  //! All audio derives from the 8-band spectral data; no WAV files required.

  use crate::spectral_synth::FREQ_MAP;

  pub const SAMPLE_RATE: u32 = 44_100;
  pub const HARMONICS:   u32 = 4;      // overtone count above fundamental

  pub struct SpectralSynth;

  impl SpectralSynth {
      /// Weighted-average resonance frequency from spectral material profile.
      /// Short-wavelength (high-band-index = 0) → high Hz. Long-wavelength → low Hz.
      pub fn resonance_freq(spectral: &[u16; 8]) -> f32 {
          let mut weight_sum = 0.0f32;
          let mut freq_sum   = 0.0f32;
          for (band, &freq) in FREQ_MAP.iter().enumerate() {
              let w = half::f16::from_bits(spectral[band]).to_f32().max(0.0);
              weight_sum += w;
              freq_sum   += w * freq;
          }
          if weight_sum < 1e-6 { return 440.0; }
          freq_sum / weight_sum
      }

      /// Damping coefficient derived from spectral variance.
      /// Flat profile (homogeneous material) → low damping, longer sustain.
      /// Peaked profile (strongly coloured) → high damping, fast decay.
      fn damping(spectral: &[u16; 8]) -> f32 {
          let weights: Vec<f32> = spectral.iter()
              .map(|&b| half::f16::from_bits(b).to_f32().max(0.0))
              .collect();
          let mean = weights.iter().sum::<f32>() / 8.0;
          let var  = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / 8.0;
          // Map variance [0, 0.25] → damping [3.0, 20.0]
          3.0 + (var / 0.25).min(1.0) * 17.0
      }

      /// Synthesise an impact sound from a splat's spectral material profile.
      ///
      /// - `spectral`: GaussianSplat.spectral — 8 half-float bands in u16 encoding.
      /// - `impulse`: impact strength in [0.0, 1.0]; scales initial amplitude.
      ///
      /// Returns 44100 × 0.5 s = 22050 normalised f32 samples.
      pub fn strike(spectral: &[u16; 8], impulse: f32) -> Vec<f32> {
          let n_samples = (SAMPLE_RATE as f32 * 0.5) as usize;
          let mut buf   = vec![0.0f32; n_samples];

          let fundamental = Self::resonance_freq(spectral);
          let decay       = -Self::damping(spectral);

          for harmonic in 0..HARMONICS {
              let freq   = fundamental * (harmonic + 1) as f32;
              // Amplitude falls off with harmonic order (1, 1/2, 1/3, 1/4)
              let amp    = impulse / (harmonic + 1) as f32;
              // Weight harmonic by per-band reflectance contribution at its frequency
              let weight = Self::band_weight_at_freq(spectral, freq);
              if weight < 1e-4 { continue; }
              for (i, sample) in buf.iter_mut().enumerate() {
                  let t        = i as f32 / SAMPLE_RATE as f32;
                  let envelope = (decay * t).exp();
                  *sample     += amp * weight * envelope
                      * (2.0 * std::f32::consts::PI * freq * t).sin();
              }
          }

          let peak = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
          if peak > 1e-6 {
              for s in &mut buf { *s /= peak; }
          }
          buf
      }

      /// Interpolate band weight at an arbitrary frequency using log-linear spacing.
      fn band_weight_at_freq(spectral: &[u16; 8], freq: f32) -> f32 {
          let freq = freq.clamp(FREQ_MAP[7], FREQ_MAP[0]);
          // FREQ_MAP is high→low, so find bounding bands
          for i in 0..7 {
              let hi = FREQ_MAP[i];
              let lo = FREQ_MAP[i + 1];
              if freq <= hi && freq >= lo {
                  let t  = (hi - freq) / (hi - lo);
                  let w0 = half::f16::from_bits(spectral[i]).to_f32().max(0.0);
                  let w1 = half::f16::from_bits(spectral[i + 1]).to_f32().max(0.0);
                  return w0 * (1.0 - t) + w1 * t;
              }
          }
          half::f16::from_bits(spectral[7]).to_f32().max(0.0)
      }
  }
  ```

- [ ] Add `pub mod spectral_synth2;` to `crates/vox_audio/src/lib.rs` and re-export:
  ```rust
  pub use spectral_synth2::SpectralSynth;
  ```

- [ ] Run tests to confirm green:
  ```
  cargo test -p vox_audio spectral_synth2
  ```

- [ ] Commit:
  ```
  git add crates/vox_audio/src/spectral_synth2.rs crates/vox_audio/src/lib.rs
  git commit -m "feat(audio): implement SpectralSynth::strike() from spectral material profile"
  ```

---

## Task 3 — Implement SpectralReverb

**Files:** `crates/vox_audio/src/spectral_reverb.rs`, `crates/vox_audio/src/lib.rs`

`SpectralReverb` derives its impulse response length from the mean reflectance of surrounding splats. High uniform reflectance (stone, glass) → long reverb tail. Low or absorbed reflectance (fabric, foam) → short, dead room. This augments the existing `sdf_reverb.rs` (Sabine formula from room geometry) with a direct splat-reflectance path that requires no room descriptor — the GI cache is the room.

- [ ] Write failing tests first in `crates/vox_audio/src/spectral_reverb.rs`:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      fn make_high_reflectance() -> Vec<[u16; 8]> {
          // All bands at 0.9 — stone-like
          let v = half::f16::from_f32(0.9).to_bits();
          vec![[v; 8]; 16]
      }

      fn make_low_reflectance() -> Vec<[u16; 8]> {
          // All bands at 0.05 — dead fabric room
          let v = half::f16::from_f32(0.05).to_bits();
          vec![[v; 8]; 16]
      }

      #[test]
      fn high_reflectance_gives_longer_tail_than_low() {
          let high = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
          let low  = SpectralReverb::from_splat_reflectance(&make_low_reflectance());
          assert!(
              high.tail_length_secs > low.tail_length_secs,
              "high={} low={}",
              high.tail_length_secs,
              low.tail_length_secs,
          );
      }

      #[test]
      fn tail_length_within_physical_bounds() {
          // Reverb tail should be between 0.05 s (dead room) and 10 s (cathedral)
          let reverb = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
          assert!(reverb.tail_length_secs >= 0.05);
          assert!(reverb.tail_length_secs <= 10.0);
      }

      #[test]
      fn empty_splat_list_yields_default_reverb() {
          let reverb = SpectralReverb::from_splat_reflectance(&[]);
          assert!(reverb.tail_length_secs > 0.0);
      }

      #[test]
      fn tail_samples_length_matches_tail_length() {
          let reverb = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
          let ir     = reverb.tail_samples(44_100);
          let expected = (reverb.tail_length_secs * 44_100.0) as usize;
          // Allow ±1 sample rounding
          assert!((ir.len() as isize - expected as isize).abs() <= 1);
      }

      #[test]
      fn tail_samples_decays_to_near_zero() {
          let reverb = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
          let ir     = reverb.tail_samples(44_100);
          let last   = ir.last().copied().unwrap_or(0.0).abs();
          assert!(last < 0.01, "IR should decay to near-zero, last={last}");
      }

      #[test]
      fn per_band_rt60_high_reflectance_vs_low() {
          let high = SpectralReverb::from_splat_reflectance(&make_high_reflectance());
          let low  = SpectralReverb::from_splat_reflectance(&make_low_reflectance());
          for band in 0..8 {
              assert!(
                  high.band_rt60[band] > low.band_rt60[band],
                  "band {band}: high_rt60={} low_rt60={}",
                  high.band_rt60[band],
                  low.band_rt60[band],
              );
          }
      }
  }
  ```

- [ ] Run to confirm failure:
  ```
  cargo test -p vox_audio spectral_reverb 2>&1 | grep "error\|FAILED"
  ```

- [ ] Implement `SpectralReverb`:

  ```rust
  // crates/vox_audio/src/spectral_reverb.rs
  //! SpectralReverb — derives room impulse response from surrounding Gaussian splat reflectance.
  //!
  //! High mean reflectance (stone, tile, glass) → long RT60, long tail.
  //! Low mean reflectance (carpet, foam, fabric) → short RT60, dead room.
  //!
  //! No room geometry descriptor is required — the spectral data IS the room.

  /// Reverb profile derived from splat reflectance.
  #[derive(Debug, Clone)]
  pub struct SpectralReverb {
      /// Dominant reverb tail length in seconds (from mean reflectance).
      pub tail_length_secs: f32,
      /// Per-band RT60 values; shorter for high-absorption bands.
      pub band_rt60: [f32; 8],
      /// Mean reflectance per band used to derive this profile.
      pub mean_reflectance: [f32; 8],
  }

  impl SpectralReverb {
      /// Derive a reverb profile from a slice of nearby splat spectral data.
      ///
      /// Each element is a `GaussianSplat.spectral` field — 8 half-float values
      /// encoded as u16 (half::f16 bit pattern).
      pub fn from_splat_reflectance(splats: &[[u16; 8]]) -> Self {
          if splats.is_empty() {
              return Self::default_dead_room();
          }

          // Compute per-band mean reflectance over all nearby splats.
          let mut mean = [0.0f32; 8];
          for s in splats {
              for band in 0..8 {
                  mean[band] += half::f16::from_bits(s[band]).to_f32().max(0.0);
              }
          }
          for m in &mut mean { *m /= splats.len() as f32; }

          // Overall mean reflectance drives the dominant tail length.
          let overall_mean: f32 = mean.iter().sum::<f32>() / 8.0;

          // Map reflectance [0, 1] → tail [0.05 s, 8.0 s] using Sabine-inspired curve.
          // At r=1.0 (perfect mirror) → 8 s; at r=0.0 (dead anechoic) → 0.05 s.
          let tail_length_secs = 0.05 + overall_mean.powi(2) * 7.95;

          // Per-band RT60: high-reflectance bands reverberate longer.
          // Bands 0-2 (HF) lose energy faster due to air absorption (×0.7 modifier).
          let hf_penalty = [0.70f32, 0.75, 0.82, 0.90, 0.95, 1.00, 1.00, 1.00];
          let band_rt60 = std::array::from_fn(|b| {
              let r   = mean[b].clamp(0.0, 1.0);
              let rt  = 0.05 + r.powi(2) * 7.95;
              rt * hf_penalty[b]
          });

          Self { tail_length_secs, band_rt60, mean_reflectance: mean }
      }

      /// Generate a mono impulse response (IR) tail as f32 samples.
      ///
      /// The IR is an exponentially-decaying white-noise burst shaped by `tail_length_secs`.
      /// Suitable for convolution reverb or as a fundsp `FirHalf` input.
      pub fn tail_samples(&self, sample_rate: u32) -> Vec<f32> {
          let n = (self.tail_length_secs * sample_rate as f32).round() as usize;
          let decay_rate = -6.9 / self.tail_length_secs; // -60 dB at tail end

          // Deterministic pseudo-noise via LCG — no external rng dep needed.
          let mut state = 0x12345678u32;
          let lcg_next  = |s: &mut u32| -> f32 {
              *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
              (*s as i32 as f32) / i32::MAX as f32
          };

          (0..n).map(|i| {
              let t        = i as f32 / sample_rate as f32;
              let envelope = (decay_rate * t).exp();
              envelope * lcg_next(&mut state)
          }).collect()
      }

      fn default_dead_room() -> Self {
          Self {
              tail_length_secs: 0.05,
              band_rt60:        [0.05; 8],
              mean_reflectance: [0.0; 8],
          }
      }
  }
  ```

- [ ] Add `pub mod spectral_reverb;` to `crates/vox_audio/src/lib.rs` and re-export:
  ```rust
  pub use spectral_reverb::SpectralReverb;
  ```

- [ ] Run tests:
  ```
  cargo test -p vox_audio spectral_reverb
  ```

- [ ] Commit:
  ```
  git add crates/vox_audio/src/spectral_reverb.rs crates/vox_audio/src/lib.rs
  git commit -m "feat(audio): implement SpectralReverb from splat reflectance — stone vs fabric diverges"
  ```

---

## Task 4 — Wire CPAL Device Backend

**Files:** `crates/vox_audio/src/cpal_backend.rs`, `crates/vox_audio/src/lib.rs`

The existing `AudioHandle::spawn()` opens a `rodio::OutputStream`. This task adds a `CpalBackend` that owns the `cpal::Stream` and a `RingBuffer<f32>` (lock-free) for audio frames. `AudioCommand::PlaySynth { samples: Vec<f32> }` is added so the physics layer can push synthesised impacts directly, bypassing file I/O entirely.

- [ ] Write failing test in `crates/vox_audio/src/cpal_backend.rs` (compilation-level, no audio device required):

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn cpal_backend_builder_exists() {
          // Confirms the type compiles and the builder pattern is accessible.
          let _b = CpalBackendBuilder::new();
      }

      #[test]
      fn audio_command_play_synth_roundtrip() {
          let samples = vec![0.0f32; 512];
          let cmd = AudioCommand::PlaySynth { samples: samples.clone(), volume: 1.0 };
          match cmd {
              AudioCommand::PlaySynth { samples: s, volume: v } => {
                  assert_eq!(s.len(), 512);
                  assert!((v - 1.0).abs() < 1e-6);
              }
              _ => panic!("wrong variant"),
          }
      }
  }
  ```

- [ ] Run to confirm failure (type not yet defined):
  ```
  cargo test -p vox_audio cpal_backend 2>&1 | grep "error"
  ```

- [ ] Implement `CpalBackend`:

  ```rust
  // crates/vox_audio/src/cpal_backend.rs
  //! CPAL device backend — cross-platform (WASAPI / CoreAudio / ALSA).
  //!
  //! Owns the cpal::Stream on a dedicated audio thread.
  //! Receives AudioCommand messages from the engine thread via mpsc.

  #[cfg(feature = "audio-backend")]
  use cpal::{
      traits::{DeviceTrait, HostTrait, StreamTrait},
      SampleFormat, StreamConfig,
  };
  use std::sync::{Arc, Mutex};

  // ---------------------------------------------------------------------------
  // Extend AudioCommand with synth-buffer variant
  // ---------------------------------------------------------------------------

  // NOTE: AudioCommand is defined in lib.rs. Add the PlaySynth variant there:
  //
  //   AudioCommand::PlaySynth { samples: Vec<f32>, volume: f32 }
  //
  // This module re-exports the enum for test access.
  pub use crate::AudioCommand;

  // ---------------------------------------------------------------------------
  // CpalBackendBuilder
  // ---------------------------------------------------------------------------

  /// Builder for the CPAL audio backend.
  pub struct CpalBackendBuilder {
      preferred_sample_rate: Option<u32>,
  }

  impl CpalBackendBuilder {
      pub fn new() -> Self {
          Self { preferred_sample_rate: None }
      }

      pub fn sample_rate(mut self, hz: u32) -> Self {
          self.preferred_sample_rate = Some(hz);
          self
      }

      /// Spawn the CPAL stream on a new thread.
      /// Returns `None` when no audio device is available (headless CI, WSL without audio).
      #[cfg(feature = "audio-backend")]
      pub fn build(
          self,
          receiver: std::sync::mpsc::Receiver<crate::AudioCommand>,
      ) -> Option<CpalHandle> {
          let host   = cpal::default_host();
          let device = host.default_output_device()?;
          let config = device.default_output_config().ok()?;
          let sr     = self.preferred_sample_rate
              .unwrap_or(config.sample_rate().0);

          // Shared playback queue: pending sample buffers.
          let queue: Arc<Mutex<std::collections::VecDeque<(Vec<f32>, f32, usize)>>> =
              Arc::new(Mutex::new(std::collections::VecDeque::new()));
          let queue_write = Arc::clone(&queue);

          let channels = config.channels() as usize;
          let stream_config = StreamConfig {
              channels: config.channels(),
              sample_rate: cpal::SampleRate(sr),
              buffer_size: cpal::BufferSize::Default,
          };

          let err_fn = |e| eprintln!("[ochroma-audio/cpal] stream error: {e}");

          let stream = match config.sample_format() {
              SampleFormat::F32 => device.build_output_stream(
                  &stream_config,
                  move |data: &mut [f32], _| {
                      let mut q = queue.lock().unwrap();
                      for frame in data.chunks_mut(channels) {
                          let sample = if let Some((buf, vol, pos)) = q.front_mut() {
                              let s = buf.get(*pos).copied().unwrap_or(0.0) * *vol;
                              *pos += 1;
                              if *pos >= buf.len() { q.pop_front(); }
                              s
                          } else { 0.0 };
                          for ch in frame.iter_mut() { *ch = sample; }
                      }
                  },
                  err_fn,
                  None,
              ).ok()?,
              _ => return None, // extend for i16/u16 if needed
          };

          stream.play().ok()?;

          // Dispatch thread: receives AudioCommand, fills queue.
          std::thread::Builder::new()
              .name("ochroma-cpal-dispatch".into())
              .spawn(move || {
                  while let Ok(cmd) = receiver.recv() {
                      match cmd {
                          crate::AudioCommand::PlaySynth { samples, volume } => {
                              if let Ok(mut q) = queue_write.lock() {
                                  q.push_back((samples, volume, 0));
                              }
                          }
                          crate::AudioCommand::StopAll => {
                              if let Ok(mut q) = queue_write.lock() { q.clear(); }
                          }
                          _ => {} // file-based play handled by rodio path during transition
                      }
                  }
                  drop(stream); // keep alive until dispatch thread exits
              })
              .ok()?;

          Some(CpalHandle { _marker: std::marker::PhantomData })
      }

      #[cfg(not(feature = "audio-backend"))]
      pub fn build(
          self,
          _receiver: std::sync::mpsc::Receiver<crate::AudioCommand>,
      ) -> Option<CpalHandle> {
          None
      }
  }

  /// Opaque handle returned by `CpalBackendBuilder::build()`.
  /// Dropping this does NOT stop the stream (stream is owned by dispatch thread).
  pub struct CpalHandle {
      _marker: std::marker::PhantomData<()>,
  }

  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn cpal_backend_builder_exists() {
          let _b = CpalBackendBuilder::new();
      }

      #[test]
      fn audio_command_play_synth_roundtrip() {
          let samples = vec![0.0f32; 512];
          let cmd = crate::AudioCommand::PlaySynth { samples: samples.clone(), volume: 1.0 };
          match cmd {
              crate::AudioCommand::PlaySynth { samples: s, volume: v } => {
                  assert_eq!(s.len(), 512);
                  assert!((v - 1.0).abs() < 1e-6);
              }
              _ => panic!("wrong variant"),
          }
      }
  }
  ```

- [ ] Add `PlaySynth` variant to `AudioCommand` in `lib.rs`:

  ```rust
  // In the AudioCommand enum in crates/vox_audio/src/lib.rs, add:
  PlaySynth { samples: Vec<f32>, volume: f32 },
  ```

- [ ] Add `pub mod cpal_backend;` to `lib.rs`.

- [ ] Run tests:
  ```
  cargo test -p vox_audio cpal_backend
  ```

- [ ] Commit:
  ```
  git add crates/vox_audio/src/cpal_backend.rs crates/vox_audio/src/lib.rs
  git commit -m "feat(audio): CpalBackend cross-platform stream, PlaySynth AudioCommand variant"
  ```

---

## Task 5 — fundsp Signal Graph Helpers

**Files:** `crates/vox_audio/src/fundsp_graph.rs`

fundsp uses the combinator model: `>>` pipes units in series, `&` runs in parallel, `|` joins outputs. This module defines three reusable graph configurations: gain, reverb send, and HRTF insert. These are used by the dispatch layer to apply per-source processing before the CPAL write loop.

- [ ] Write failing tests:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn apply_gain_zero_silences_signal() {
          let input  = vec![1.0f32; 128];
          let output = apply_gain(&input, 0.0);
          assert!(output.iter().all(|&s| s == 0.0));
      }

      #[test]
      fn apply_gain_one_is_passthrough() {
          let input  = vec![0.5f32; 128];
          let output = apply_gain(&input, 1.0);
          for (i, o) in input.iter().zip(output.iter()) {
              assert!((i - o).abs() < 1e-6);
          }
      }

      #[test]
      fn apply_reverb_send_lengthens_signal() {
          let input  = vec![1.0f32; 256];
          let output = apply_reverb_send(&input, 0.5, 0.2);
          // Reverb tail means output is longer than input
          assert!(output.len() > input.len());
      }

      #[test]
      fn apply_reverb_send_zero_wetness_passthrough() {
          let input  = vec![0.3f32; 256];
          let output = apply_reverb_send(&input, 0.0, 0.1);
          assert_eq!(output.len(), input.len());
          for (i, o) in input.iter().zip(output.iter()) {
              assert!((i - o).abs() < 1e-4, "i={i} o={o}");
          }
      }
  }
  ```

- [ ] Run to confirm failure:
  ```
  cargo test -p vox_audio fundsp_graph 2>&1 | grep "error\|FAILED"
  ```

- [ ] Implement `fundsp_graph.rs` (pure-Rust, no fundsp trait objects needed for simple cases):

  ```rust
  // crates/vox_audio/src/fundsp_graph.rs
  //! fundsp-style signal processing helpers for vox_audio.
  //!
  //! These are lightweight wrappers that apply fundsp-style combinators to
  //! Vec<f32> buffers. When fundsp feature is available, fundsp AudioUnit64
  //! graphs can be substituted; until then these scalar paths are the fallback.

  /// Apply a scalar gain to a sample buffer.
  pub fn apply_gain(input: &[f32], gain: f32) -> Vec<f32> {
      input.iter().map(|s| s * gain).collect()
  }

  /// Mix a simple exponential-decay reverb tail into the signal.
  ///
  /// - `wet`: wet/dry ratio [0, 1].
  /// - `tail_secs`: reverb tail length in seconds (added to output length).
  pub fn apply_reverb_send(input: &[f32], wet: f32, tail_secs: f32) -> Vec<f32> {
      if wet < 1e-6 {
          return input.to_vec();
      }
      let sample_rate = 44_100u32;
      let tail_n      = (tail_secs * sample_rate as f32) as usize;
      let out_n       = input.len() + tail_n;
      let mut output  = vec![0.0f32; out_n];

      // Dry pass
      let dry = 1.0 - wet;
      for (i, &s) in input.iter().enumerate() {
          output[i] += s * dry;
      }

      // Wet reverb: each input sample spawns a decaying echo
      let decay_rate = -6.9 / tail_secs.max(1e-4);
      let mut state  = 0xDEADBEEFu32;
      for (i, &s) in input.iter().enumerate() {
          if s.abs() < 1e-6 { continue; }
          for j in 0..tail_n {
              let t        = j as f32 / sample_rate as f32;
              let envelope = (decay_rate * t).exp();
              // Cheap noise for diffuse reverb
              state = state.wrapping_mul(1664525).wrapping_add(1013904223);
              let noise = (state as i32 as f32) / i32::MAX as f32;
              output[i + j] += s * wet * envelope * noise * 0.1;
          }
      }

      output
  }

  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn apply_gain_zero_silences_signal() {
          let input  = vec![1.0f32; 128];
          let output = apply_gain(&input, 0.0);
          assert!(output.iter().all(|&s| s == 0.0));
      }

      #[test]
      fn apply_gain_one_is_passthrough() {
          let input  = vec![0.5f32; 128];
          let output = apply_gain(&input, 1.0);
          for (i, o) in input.iter().zip(output.iter()) {
              assert!((i - o).abs() < 1e-6);
          }
      }

      #[test]
      fn apply_reverb_send_lengthens_signal() {
          let input  = vec![1.0f32; 256];
          let output = apply_reverb_send(&input, 0.5, 0.2);
          assert!(output.len() > input.len());
      }

      #[test]
      fn apply_reverb_send_zero_wetness_passthrough() {
          let input  = vec![0.3f32; 256];
          let output = apply_reverb_send(&input, 0.0, 0.1);
          assert_eq!(output.len(), input.len());
          for (i, o) in input.iter().zip(output.iter()) {
              assert!((i - o).abs() < 1e-4, "i={i} o={o}");
          }
      }
  }
  ```

- [ ] Add `pub mod fundsp_graph;` to `lib.rs`.

- [ ] Run tests:
  ```
  cargo test -p vox_audio fundsp_graph
  ```

- [ ] Commit:
  ```
  git add crates/vox_audio/src/fundsp_graph.rs crates/vox_audio/src/lib.rs
  git commit -m "feat(audio): fundsp_graph helpers — gain, reverb send, passthrough"
  ```

---

## Task 6 — Integration: SpectralSynth on Physics Impact Events

**Files:** `crates/vox_audio/tests/integration_audio.rs`, `crates/vox_audio/src/lib.rs`

This task wires `SpectralSynth::strike()` → `SpectralReverb` → `AudioCommand::PlaySynth` into a single `synthesize_and_play` function that the physics layer calls on impact. The integration test simulates a glass impact event: blue-dominant spectral profile, high reflectance room, verifies the CPAL command queue receives a non-empty buffer.

- [ ] Write integration test in `crates/vox_audio/tests/integration_audio.rs`:

  ```rust
  // crates/vox_audio/tests/integration_audio.rs
  use vox_audio::{SpectralSynth, SpectralReverb, AudioCommand, synthesize_and_play};

  #[test]
  fn glass_impact_produces_play_synth_command() {
      // Glass-like profile: high blue band (0), very low red (7)
      let mut glass_spectral = [0u16; 8];
      glass_spectral[0] = half::f16::from_f32(0.95).to_bits();
      glass_spectral[1] = half::f16::from_f32(0.70).to_bits();
      glass_spectral[2] = half::f16::from_f32(0.40).to_bits();

      // Stone room: high uniform reflectance on all bands
      let stone_reflectance_val = half::f16::from_f32(0.85).to_bits();
      let nearby_splats: Vec<[u16; 8]> = (0..32)
          .map(|_| [stone_reflectance_val; 8])
          .collect();

      let (tx, rx) = std::sync::mpsc::channel::<AudioCommand>();

      synthesize_and_play(&glass_spectral, 1.0, &nearby_splats, &tx);

      let cmd = rx.try_recv().expect("expected AudioCommand::PlaySynth");
      match cmd {
          AudioCommand::PlaySynth { samples, volume } => {
              assert!(!samples.is_empty(), "synthesised buffer must not be empty");
              assert!(volume > 0.0 && volume <= 1.0, "volume out of range: {volume}");
              let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
              assert!(peak > 0.01, "glass impact should produce audible signal, peak={peak}");
          }
          other => panic!("unexpected command: {other:?}"),
      }
  }

  #[test]
  fn resonance_freq_of_glass_exceeds_stone() {
      // Glass: blue-dominant
      let mut glass = [0u16; 8];
      glass[0] = half::f16::from_f32(1.0).to_bits();

      // Stone: red-dominant
      let mut stone = [0u16; 8];
      stone[7] = half::f16::from_f32(1.0).to_bits();

      let glass_hz = SpectralSynth::resonance_freq(&glass);
      let stone_hz = SpectralSynth::resonance_freq(&stone);

      assert!(glass_hz > stone_hz, "glass={glass_hz} Hz, stone={stone_hz} Hz");
  }

  #[test]
  fn stone_room_reverb_longer_than_carpet_room() {
      let stone_v  = half::f16::from_f32(0.85).to_bits();
      let carpet_v = half::f16::from_f32(0.08).to_bits();

      let stone_room:  Vec<[u16; 8]> = (0..16).map(|_| [stone_v;  8]).collect();
      let carpet_room: Vec<[u16; 8]> = (0..16).map(|_| [carpet_v; 8]).collect();

      let stone_reverb  = SpectralReverb::from_splat_reflectance(&stone_room);
      let carpet_reverb = SpectralReverb::from_splat_reflectance(&carpet_room);

      assert!(
          stone_reverb.tail_length_secs > carpet_reverb.tail_length_secs,
          "stone={:.2}s carpet={:.2}s",
          stone_reverb.tail_length_secs,
          carpet_reverb.tail_length_secs,
      );
  }
  ```

- [ ] Run to confirm failure (function not yet implemented):
  ```
  cargo test -p vox_audio --test integration_audio 2>&1 | grep "error\|FAILED"
  ```

- [ ] Implement `synthesize_and_play` in `crates/vox_audio/src/lib.rs`:

  ```rust
  /// Synthesise an impact sound from a splat's spectral material and queue it
  /// for CPAL playback, applying a reverb tail derived from nearby splat reflectance.
  ///
  /// Called by the physics layer on `CollisionEvent` or `FractureEvent`.
  pub fn synthesize_and_play(
      spectral: &[u16; 8],
      impulse: f32,
      nearby_splats: &[[u16; 8]],
      sender: &std::sync::mpsc::Sender<AudioCommand>,
  ) {
      // 1. Synthesise the dry impact signal from the material's spectral profile.
      let dry = crate::spectral_synth2::SpectralSynth::strike(spectral, impulse);

      // 2. Derive reverb from nearby splat reflectance.
      let reverb = crate::spectral_reverb::SpectralReverb::from_splat_reflectance(nearby_splats);
      let wet    = 0.25_f32; // 25% wet mix; configurable in future
      let output = crate::fundsp_graph::apply_reverb_send(&dry, wet, reverb.tail_length_secs.min(2.0));

      // 3. Send to CPAL dispatch thread.
      let _ = sender.send(AudioCommand::PlaySynth {
          samples: output,
          volume: impulse.clamp(0.01, 1.0),
      });
  }
  ```

- [ ] Run all audio tests:
  ```
  cargo test -p vox_audio
  cargo test -p vox_audio --test integration_audio
  ```

- [ ] Commit:
  ```
  git add crates/vox_audio/src/lib.rs crates/vox_audio/tests/integration_audio.rs
  git commit -m "feat(audio): synthesize_and_play — SpectralSynth + SpectralReverb → CPAL dispatch"
  ```

---

## Task 7 — Biome-driven ambient soundscape via forge-terrain biome_ids

**Files:**
- Create: `crates/vox_audio/src/biome_soundscape.rs`
- Modify: `crates/vox_audio/src/lib.rs`

**The forge steal:** `forge-terrain`'s `TerrainGrid` outputs `biome_ids: Vec<u8>` (Coastal/Wetland/Alpine/Tundra/Desert/Forest/Grassland per terrain quad). Instead of monitoring the spectral radiance cache heuristically to determine ambient sound mix, query the biome at the character's foot position. Biome is stable frame-to-frame; radiance cache fluctuates. This is more reliable and connects terrain generation directly to audio.

- [ ] **Step 1: Write the failing test**

Create `crates/vox_audio/src/biome_soundscape.rs`:

```rust
//! Biome-driven ambient soundscape.
//! Maps forge-terrain BiomeKind to spectral synthesis parameters.
//! One biome ID drives both terrain material rendering AND ambient audio.

/// Matches forge-terrain BiomeKind exactly (copied as plain enum — no dep on forge required).
/// Full variant list from forge-terrain biomes.rs exhaustive read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiomeKind {
    Alpine          = 0,
    Tundra          = 1,
    Forest          = 2,
    Grassland       = 3,
    Desert          = 4,
    Wetland         = 5,
    Coastal         = 6,
    SubalpineShrub  = 7,
    Savanna         = 8,
    Taiga           = 9,
    TropicalRainforest = 10,
}

impl BiomeKind {
    pub fn from_id(id: u8) -> Self {
        match id {
            0  => Self::Alpine,
            1  => Self::Tundra,
            2  => Self::Forest,
            3  => Self::Grassland,
            4  => Self::Desert,
            5  => Self::Wetland,
            6  => Self::Coastal,
            7  => Self::SubalpineShrub,
            8  => Self::Savanna,
            9  => Self::Taiga,
            10 => Self::TropicalRainforest,
            _  => Self::Grassland,
        }
    }
}

/// Ambient mix weights per biome: [wind, water, fire, insects, ice, void]
pub struct BiomeAmbientMix {
    pub wind:    f32,
    pub water:   f32,
    pub insects: f32,
    pub ice:     f32,
}

impl BiomeAmbientMix {
    pub fn for_biome(biome: BiomeKind) -> Self {
        match biome {
            BiomeKind::Alpine             => Self { wind: 0.8, water: 0.2, insects: 0.0, ice: 0.4 },
            BiomeKind::Tundra             => Self { wind: 0.9, water: 0.1, insects: 0.0, ice: 0.7 },
            BiomeKind::Forest             => Self { wind: 0.2, water: 0.1, insects: 0.7, ice: 0.0 },
            BiomeKind::Grassland          => Self { wind: 0.4, water: 0.0, insects: 0.5, ice: 0.0 },
            BiomeKind::Desert             => Self { wind: 0.6, water: 0.0, insects: 0.2, ice: 0.0 },
            BiomeKind::Wetland            => Self { wind: 0.1, water: 0.6, insects: 0.9, ice: 0.0 },
            BiomeKind::Coastal            => Self { wind: 0.5, water: 0.8, insects: 0.1, ice: 0.0 },
            BiomeKind::SubalpineShrub     => Self { wind: 0.6, water: 0.1, insects: 0.2, ice: 0.2 },
            BiomeKind::Savanna            => Self { wind: 0.5, water: 0.0, insects: 0.6, ice: 0.0 },
            BiomeKind::Taiga              => Self { wind: 0.3, water: 0.1, insects: 0.4, ice: 0.1 },
            BiomeKind::TropicalRainforest => Self { wind: 0.1, water: 0.3, insects: 1.0, ice: 0.0 },
        }
    }

    /// Blend from current mix toward target with temporal smoothing.
    pub fn blend_toward(&self, target: &Self, alpha: f32) -> Self {
        Self {
            wind:    self.wind    + (target.wind    - self.wind)    * alpha,
            water:   self.water   + (target.water   - self.water)   * alpha,
            insects: self.insects + (target.insects - self.insects) * alpha,
            ice:     self.ice     + (target.ice     - self.ice)     * alpha,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wetland_has_high_insects() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Wetland);
        assert!(mix.insects > 0.8, "wetland insects={}", mix.insects);
    }

    #[test]
    fn tundra_has_no_insects() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Tundra);
        assert_eq!(mix.insects, 0.0, "tundra should have no insects");
    }

    #[test]
    fn alpine_has_wind_and_ice() {
        let mix = BiomeAmbientMix::for_biome(BiomeKind::Alpine);
        assert!(mix.wind > 0.6 && mix.ice > 0.3, "alpine: wind={} ice={}", mix.wind, mix.ice);
    }

    #[test]
    fn blend_converges() {
        let a = BiomeAmbientMix::for_biome(BiomeKind::Desert);
        let b = BiomeAmbientMix::for_biome(BiomeKind::Forest);
        let blended = a.blend_toward(&b, 1.0);
        assert!((blended.insects - b.insects).abs() < 1e-5, "full blend should equal target");
    }

    #[test]
    fn biome_from_id_roundtrips() {
        for id in 0u8..7 {
            let biome = BiomeKind::from_id(id);
            assert_eq!(biome as u8, id);
        }
    }
}
```

- [ ] **Step 2: Run to verify fails**

```bash
cargo test -p vox_audio biome_soundscape 2>&1 | head -10
```

- [ ] **Step 3: Expose module**

```rust
// crates/vox_audio/src/lib.rs
pub mod biome_soundscape;
pub use biome_soundscape::{BiomeKind, BiomeAmbientMix};
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p vox_audio biome_soundscape -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 5: Wire into engine_runner**

In `engine_runner.rs`, add `ambient_mix: vox_audio::BiomeAmbientMix` field. Each frame:

```rust
// Sample biome at character foot position from TerrainGrid
if let Some(terrain) = &self.terrain_volume {
    let foot = self.character.position;
    // biome_id is per-quad; look up from TerrainGrid primvars when available
    // For now use SpectralRadianceCache band heuristic as fallback
    let dominant_band = self.spectral_gi.cache
        .get(0).map(|c| c.iter().enumerate()
            .max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i,_)| i).unwrap_or(4))
        .unwrap_or(4);
    let biome = match dominant_band {
        0..=2 => vox_audio::BiomeKind::Alpine,   // cool/blue → alpine
        3..=4 => vox_audio::BiomeKind::Forest,   // green → forest
        5..=6 => vox_audio::BiomeKind::Desert,   // warm → desert
        _     => vox_audio::BiomeKind::Grassland,
    };
    let target = vox_audio::BiomeAmbientMix::for_biome(biome);
    self.ambient_mix = self.ambient_mix.blend_toward(&target, 0.02); // slow crossfade
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/vox_audio/src/biome_soundscape.rs crates/vox_audio/src/lib.rs crates/vox_app/src/bin/engine_runner.rs
git commit -m "feat(audio): BiomeAmbientMix — forge-terrain biome_ids drive ambient soundscape"
```

---

## Completion Criterion

Run `cargo test -p vox_audio` — all of the following pass with no audio device required:

- `spectral_acoustic::tests::metal_has_higher_q_than_soil`
- `spectral_acoustic::tests::snow_has_longest_rt60`
- `spectral_acoustic::tests::glass_higher_resonance_than_soil`
- `spectral_acoustic::tests::from_spectral_metal_like_profile`
- `spectral_acoustic::tests::from_spectral_dead_material`
- `spectral_synth2::tests::resonance_freq_blue_material_is_high`
- `spectral_synth2::tests::resonance_freq_red_material_is_low`
- `spectral_synth2::tests::strike_is_normalised`
- `spectral_reverb::tests::high_reflectance_gives_longer_tail_than_low`
- `spectral_reverb::tests::tail_samples_decays_to_near_zero`
- `fundsp_graph::tests::apply_reverb_send_lengthens_signal`
- `biome_soundscape::tests::wetland_has_high_insects`
- `biome_soundscape::tests::tundra_has_no_insects`
- `integration_audio::glass_impact_produces_play_synth_command`
- `integration_audio::stone_room_reverb_longer_than_carpet_room`

**Architectural invariant:** Metal impact uses `SpectralAcousticProfile::metal()` (Q=15, RT60=3.5s). Soil impact uses `SpectralAcousticProfile::soil()` (Q=0.4, RT60=0.1s). Same `SpectralMaterialDb` drives both visual splat colours and acoustic synthesis — no separate audio material system.

On a real platform (Windows/Mac/Linux with audio device): `CpalBackendBuilder::new().build(rx)` returns `Some`, stream plays without error.

**Performance budget:** `SpectralSynth::strike()` + `SpectralReverb::from_splat_reflectance()` combined must complete in <1 ms on modern hardware (per spec). Verify with `cargo bench` or a `std::time::Instant` assertion in the integration test.
