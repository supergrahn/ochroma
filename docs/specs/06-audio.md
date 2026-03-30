# Domain 6 — Audio — Spectral Advantage

**Version:** 1.0 — 2026-03-29
**Status:** Draft
**Crate:** `vox_audio` (primary), with integration points in `vox_terrain`, `vox_render`, `vox_physics`

---

## Goals

1. Room acoustics are derived directly from the terrain SDF at runtime — no pre-baking, no static reverb
   zones, no artist-placed audio volumes.
2. Every material sound (footsteps, impacts, ambient) is synthesized from spectral profiles — no audio
   sample assets required for core gameplay. Sample assets are optional layering on top.
3. The spectral profile of visible splat assemblies directly modulates synthesis parameters via
   `SpectralLookup` audio graph nodes — the sound IS the visual light energy. Unreal cannot do this.
4. HRTF, occlusion, diffraction, and Doppler are all implemented against the same SDF that drives
   rendering; there is no separate acoustic geometry.
5. The music system reads dominant spectral bands from the live scene as an intensity signal — the
   visual world tonality drives the musical tonality.

---

## 6.1 Spectral Room Acoustics — SDF Reverb

### Core Concept

Traditional engines (Unreal Audio, FMOD) bake reverb into static zones or use a single broadband IR.
Ochroma derives per-band impulse responses from the terrain SDF at runtime. Because absorption is
wavelength-dependent, a concrete corridor sounds different from the same shape carved in carpet — the
spectral profile of the surface material feeds directly into band-specific absorption coefficients.

### Data Structures

```rust
/// Measurement point in world space with a per-band impulse response.
pub struct AcousticProbe {
    pub position: Vec3,
    /// impulse_response[band][sample_index] = (delay_seconds, amplitude).
    /// delay is quantized to the audio block rate (512 samples / 44100 Hz ≈ 11.6ms per block).
    pub impulse_response: Vec<Vec<(f32, f32)>>, // [8 bands][N reflection events]
    pub is_dirty: bool,
}

/// Absorption and scattering coefficients per spectral band for a surface material.
pub struct MaterialAcoustics {
    /// Fraction of incident energy absorbed per bounce, per band. Range [0, 1].
    pub absorption: [f32; 8],
    /// Fraction of incident energy scattered (vs specularly reflected), per band.
    pub scattering: [f32; 8],
}
```

Predefined `MaterialAcoustics` constants:

| Material | Absorption (bands 0–7) | Notes |
|----------|------------------------|-------|
| Concrete | `[0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.07]` | Bright reverb, long tail |
| Carpet   | `[0.08, 0.10, 0.20, 0.35, 0.50, 0.60, 0.65, 0.70]` | Kills high bands fast |
| Water    | `[0.35, 0.30, 0.25, 0.15, 0.05, 0.03, 0.03, 0.03]` | Reflective high bands |
| Foliage  | `[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.45, 0.50]` | High scattering all bands |
| Glass    | `[0.03, 0.03, 0.03, 0.04, 0.05, 0.04, 0.03, 0.03]` | Bright, minimal absorption |
| Wood     | `[0.10, 0.12, 0.14, 0.15, 0.20, 0.25, 0.30, 0.35]` | Warm: kills highs moderately |

### AudioRayMarcher

```rust
pub struct AudioRayMarcher {
    /// Number of audio rays emitted per probe measurement.
    pub ray_count: usize,         // default 64
    pub max_bounces: usize,       // default 20
    pub max_distance: f32,        // default 50.0 m
    pub speed_of_sound: f32,      // 343.0 m/s
}

impl AudioRayMarcher {
    pub fn measure(
        &self,
        probe_pos: Vec3,
        terrain: &TerrainVolume,
        material_acoustics: &HashMap<TerrainMaterial, MaterialAcoustics>,
    ) -> Vec<Vec<(f32, f32)>> {
        // Returns impulse_response[8 bands][N reflections]
        let mut ir: Vec<Vec<(f32, f32)>> = vec![Vec::new(); 8];

        let directions = fibonacci_sphere(self.ray_count); // deterministic hemispherical distribution

        for dir in directions {
            let mut ray_pos = probe_pos;
            let mut ray_dir = dir;
            let mut energy = [1.0f32; 8]; // per-band energy
            let mut travel_dist = 0.0f32;

            for _bounce in 0..self.max_bounces {
                // SDF sphere-march to next surface.
                let hit = sdf_march(ray_pos, ray_dir, terrain, self.max_distance - travel_dist);
                let Some(hit) = hit else { break };

                travel_dist += hit.distance;
                if travel_dist > self.max_distance { break; }

                let delay = travel_dist / self.speed_of_sound;
                let mat = terrain.material_at(hit.point);
                let acoustics = material_acoustics
                    .get(&mat)
                    .unwrap_or(&CONCRETE_ACOUSTICS);

                // Accumulate energy at this delay into the IR.
                for band in 0..8 {
                    let absorbed_energy = energy[band] * acoustics.absorption[band];
                    ir[band].push((delay, absorbed_energy));
                    energy[band] *= 1.0 - acoustics.absorption[band]; // attenuate for next bounce
                    energy[band] *= 1.0 - acoustics.scattering[band] * 0.5; // scatter loss
                }

                // Reflect ray using SDF gradient as surface normal.
                let normal = terrain.sdf_gradient(hit.point).normalize_or_zero();
                let specular = ray_dir - 2.0 * ray_dir.dot(normal) * normal;
                let scattered = random_hemisphere_dir(normal);
                let avg_scattering = acoustics.scattering.iter().sum::<f32>() / 8.0;
                ray_dir = specular.lerp(scattered, avg_scattering).normalize_or_zero();
                ray_pos = hit.point + normal * 0.01; // push off surface

                if energy.iter().all(|&e| e < 0.001) { break; } // energy depleted
            }
        }

        // Sort each band's IR by delay for convolution.
        for band in &mut ir {
            band.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        ir
    }
}
```

### ReverberationEngine

```rust
pub struct ReverberationEngine {
    pub probes: Vec<AcousticProbe>,
    pub listener_probe: AcousticProbe,
    pub last_listener_pos: Vec3,
    pub update_distance_threshold: f32, // default 1.0 m
    pub fft_size: usize,                // default 2048
}

impl ReverberationEngine {
    /// Call from audio update thread. Re-measures listener_probe if player has moved.
    pub fn update(&mut self, listener_pos: Vec3, terrain: &TerrainVolume, ...) {
        if (listener_pos - self.last_listener_pos).length() > self.update_distance_threshold {
            self.listener_probe.impulse_response =
                self.ray_marcher.measure(listener_pos, terrain, &self.material_acoustics);
            self.last_listener_pos = listener_pos;
        }
    }

    /// Convolve a mono audio block with the current IR for a given spectral band.
    /// Uses overlap-add FFT convolution (rustfft).
    pub fn convolve_band(&self, audio: &[f32], band: usize) -> Vec<f32> {
        // IR for this band as a discrete FIR filter sampled at audio_sample_rate.
        let ir_samples = self.ir_to_samples(&self.listener_probe.impulse_response[band], 44100);
        overlap_add_convolve(audio, &ir_samples, self.fft_size)
    }
}

fn overlap_add_convolve(signal: &[f32], ir: &[f32], fft_size: usize) -> Vec<f32> {
    // Standard overlap-add: partition signal into blocks of (fft_size - ir.len() + 1),
    // FFT each block, multiply by FFT(ir), IFFT, overlap-add.
    // Implemented via rustfft::FftPlanner.
    // Cost: O(N log N) per band per frame where N = fft_size.
}
```

**Dynamic update on terrain carve:** `TerrainVolume::carve_sphere(center, radius)` sends a
`TerrainCarveEvent` to the audio system. `ReverberationEngine` re-measures all probes within
`2 * carve_radius` of `center` on the next audio update tick. This keeps reverb accurate after
explosions, tunneling, or terrain modification.

**Competitive advantage:** Unreal's Reverb Submission system uses a single convolution IR per reverb
zone (broadband, artist-baked). Ochroma's IR is per-spectral-band, physically derived from the SDF at
runtime, and updates when terrain changes. A dynamically-carved cave sounds like a cave the moment the
last SDF voxel is removed.

---

## 6.2 Spectral Audio Synthesis — Extending vox_audio

### MaterialSoundLibrary

```rust
pub struct MaterialSoundLibrary {
    pub table: HashMap<MaterialId, MaterialSoundProfile>,
}

pub struct MaterialSoundProfile {
    pub acoustics: MaterialAcoustics,
    /// Dominant resonant frequency range in Hz derived from spectral profile.
    /// High-band materials (glass, band 0–2 high): resonant_freq_range = (800, 4000).
    /// Low-band materials (wood, band 5–7 high):   resonant_freq_range = (80, 400).
    pub resonant_freq_range: (f32, f32),
    /// Quality factor Q of the dominant resonance (sharpness of peak).
    pub resonance_q: f32,
    /// Impact decay time constant (s). Hard materials: 0.05; soft: 0.5.
    pub decay_tau: f32,
}
```

### SpectralSynthesizer

Extends `synthesize_impact` with material-aware resonance and spectral coloring:

```rust
pub struct SpectralSynthesizer {
    pub sample_rate: u32,         // 44100
    pub library: MaterialSoundLibrary,
}

impl SpectralSynthesizer {
    /// Synthesize an impact audio buffer (PCM f32, mono).
    /// All parameters derived from spectral profile — no audio files required.
    pub fn synthesize_impact(
        &self,
        spectral: &[f32; 8],
        impulse_magnitude: f32,
        material: MaterialId,
    ) -> Vec<f32> {
        let profile = &self.library.table[&material];

        // Derive resonant frequency from spectral band weights.
        // Band 0 = 20Hz, Band 7 = 20kHz (logarithmic spacing).
        let dominant_band = spectral.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(4);
        let freq = band_to_frequency(dominant_band); // logarithmic interpolation

        // Volume envelope: attack 1ms, decay = decay_tau, sustain 0, release = reverb tail.
        let volume = impulse_magnitude.clamp(0.0, 1.0);
        let n_samples = (profile.decay_tau * 5.0 * self.sample_rate as f32) as usize;
        let mut buffer = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f32 / self.sample_rate as f32;
            let envelope = if t < 0.001 {
                t / 0.001 // attack
            } else {
                (-t / profile.decay_tau).exp() // exponential decay
            };
            // Damped sinusoidal oscillator at resonant frequency.
            let carrier = (2.0 * std::f32::consts::PI * freq * t).sin();
            // Noise burst modulated by envelope for initial transient.
            let noise_burst = if t < 0.005 { white_noise() * 0.3 } else { 0.0 };
            buffer.push((carrier + noise_burst) * envelope * volume);
        }

        // Spectral coloring: apply per-band bandpass filters.
        // Transfer function = product of (1 + spectral[b] * biquad_bandpass(band_freq[b], Q)).
        for band in 0..8 {
            if spectral[band] > 0.05 {
                apply_biquad_bandpass(
                    &mut buffer,
                    band_to_frequency(band),
                    profile.resonance_q * spectral[band],
                    self.sample_rate,
                );
            }
        }

        buffer
    }
}
```

### FootstepSynthesizer

```rust
pub struct FootstepSynthesizer {
    pub synthesizer: SpectralSynthesizer,
}

impl FootstepSynthesizer {
    /// Synthesize a footstep sound. No audio samples needed.
    pub fn synthesize(
        &self,
        surface: MaterialId,
        weight: f32,   // character mass in kg
        velocity: f32, // character speed in m/s
    ) -> Vec<f32> {
        let profile = &self.synthesizer.library.table[&surface];
        // Impact magnitude scales with weight and velocity.
        let magnitude = (weight / 80.0) * (velocity / 3.0).clamp(0.5, 2.0);
        let spectral = material_id_to_spectral(surface);
        let mut base = self.synthesizer.synthesize_impact(&spectral, magnitude, surface);

        // Material-specific layering:
        match surface {
            MaterialId::GRASS => {
                // High-frequency rustle: bandpass white noise at bands 2–4.
                let rustle = bandpass_noise(600.0, 3000.0, 0.05, self.synthesizer.sample_rate);
                mix_into(&mut base, &rustle, 0.4);
            }
            MaterialId::METAL_GRATE => {
                // Resonant ring-down: second oscillator at 3× freq with slow decay.
                let freq = band_to_frequency(2);
                let ring = synthesize_damped_sine(freq * 3.0, 0.8, magnitude * 0.6, self.synthesizer.sample_rate);
                mix_into(&mut base, &ring, 0.7);
            }
            MaterialId::GRAVEL => {
                // Stochastic: 8–12 low-energy sub-impacts offset by random delays.
                let count = 8 + (white_noise() * 4.0).abs() as usize;
                for _ in 0..count {
                    let delay = (white_noise().abs() * 0.03 * self.synthesizer.sample_rate as f32) as usize;
                    let sub = self.synthesizer.synthesize_impact(&spectral, magnitude * 0.15, surface);
                    mix_at_offset(&mut base, &sub, delay, 1.0);
                }
            }
            _ => {}
        }
        base
    }
}
```

### AmbientSynthesizer

```rust
pub struct AmbientSynthesizer {
    pub update_interval: f32,  // default 1.0 s
    pub sample_rate: u32,
    pub block_duration: f32,   // default 1.0 s of audio per update
}

impl AmbientSynthesizer {
    /// Scan nearby splat assemblies, derive spectral dominance, compose ambient drone.
    /// Called at 1 Hz. Output is a 1-second f32 PCM buffer that the audio engine
    /// crossfades into the ambient mix.
    pub fn update(&self, nearby_assemblies: &[&SplatAssembly], scene_splats: &[GaussianSplat]) -> Vec<f32> {
        let mut band_energy = [0.0f32; 8];
        for assembly in nearby_assemblies {
            for &idx in &assembly.splat_indices {
                let s = decode_spectral_u16(scene_splats[idx as usize].spectral);
                for b in 0..8 { band_energy[b] += s[b]; }
            }
        }
        // Normalize.
        let total: f32 = band_energy.iter().sum();
        if total > 0.0 { for b in &mut band_energy { *b /= total; } }

        let n = (self.block_duration * self.sample_rate as f32) as usize;
        let mut buffer = vec![0.0f32; n];

        // Foliage / green-band (bands 1–3 high): bird-frequency component 2–6 kHz.
        let green = band_energy[1] + band_energy[2] + band_energy[3];
        if green > 0.15 {
            mix_into(&mut buffer, &bandpass_noise(2000.0, 6000.0, green * 0.3, self.sample_rate), green);
        }

        // Water / glass (bands 4–6 high): trickle component — stochastic impulse train.
        let blue = band_energy[4] + band_energy[5] + band_energy[6];
        if blue > 0.15 {
            let trickle = synthesize_impulse_train(12.0 + blue * 30.0, 0.02, self.sample_rate, n);
            mix_into(&mut buffer, &trickle, blue * 0.5);
        }

        // Fire / warm (bands 0, 6–7 high): crackling component — burst noise, low frequency.
        let red = band_energy[0] + band_energy[6] + band_energy[7];
        if red > 0.15 {
            let crackle = synthesize_burst_noise(red * 20.0, 0.005, 200.0, self.sample_rate, n);
            mix_into(&mut buffer, &crackle, red * 0.4);
        }

        buffer
    }
}
```

### WindSynthesizer

```rust
pub struct WindSynthesizer {
    pub sample_rate: u32,
}

impl WindSynthesizer {
    pub fn synthesize(
        &self,
        speed: f32,
        direction: Vec3,
        obstacles: &[Aabb],
        terrain: &TerrainVolume,
        listener_pos: Vec3,
    ) -> Vec<f32> {
        let n = (0.1 * self.sample_rate as f32) as usize; // 100ms block
        let mut buffer = pink_noise(speed * 0.1, self.sample_rate, n);

        for obstacle in obstacles {
            // Wind shadow: turbulence post-obstacle, modeled as AM noise burst.
            if is_downwind(listener_pos, obstacle, direction) {
                let turbulence = white_noise_burst(speed * 0.05, n);
                mix_into(&mut buffer, &turbulence, 0.3);
            }
        }

        // Narrow SDF passages: compute min SDF value along wind direction near listener.
        let gap_width = terrain.sdf(listener_pos);
        if gap_width < 0.5 {
            // Whistle frequency rises as gap narrows: f ≈ speed / (2 * gap_width).
            let whistle_freq = (speed / (2.0 * gap_width.max(0.05))).clamp(200.0, 8000.0);
            let whistle = synthesize_damped_sine(whistle_freq, 0.05, speed * 0.2, self.sample_rate);
            mix_into(&mut buffer, &whistle[..n.min(whistle.len())], 0.6);
        }

        buffer
    }
}
```

---

## 6.3 3D Spatial Audio

### Current State Extension

`SpatialAudioManager` provides distance attenuation. The following layers are added:

### HRTF

```rust
pub struct HrtfTable {
    /// Indexed by [elevation_index][azimuth_index][ear: 0=left, 1=right].
    /// Filters are FIR impulse responses (128 taps each).
    /// Source: MIT KEMAR HRTF database, 14 elevations × 72 azimuths.
    pub filters: Vec<Vec<[[f32; 128]; 2]>>,
    pub elevation_count: usize, // 14
    pub azimuth_count: usize,   // 72
}

impl HrtfTable {
    /// Look up nearest HRTF filter for a source at the given azimuth/elevation.
    pub fn get(&self, azimuth_deg: f32, elevation_deg: f32) -> &[[f32; 128]; 2] {
        let el_idx = ((elevation_deg + 90.0) / 180.0 * self.elevation_count as f32) as usize;
        let az_idx = (azimuth_deg / 360.0 * self.azimuth_count as f32) as usize;
        &self.filters[el_idx.min(self.elevation_count - 1)][az_idx % self.azimuth_count]
    }
}
```

HRTF dataset is embedded as a compressed binary blob (`include_bytes!("../data/kemar_hrtf.bin")`),
loaded at startup into `HrtfTable`. No runtime filesystem access required.

### Occlusion

```rust
pub fn compute_occlusion(
    source: Vec3,
    listener: Vec3,
    terrain: &TerrainVolume,
) -> f32 {
    // March a ray from source to listener through the SDF.
    // Accumulate path length inside solid (SDF < 0).
    let dir = (listener - source).normalize_or_zero();
    let dist = (listener - source).length();
    let mut occlusion_length = 0.0f32;
    let mut t = 0.0f32;
    while t < dist {
        let pos = source + dir * t;
        let sdf_val = terrain.sdf(pos);
        if sdf_val < 0.0 { occlusion_length += -sdf_val; }
        t += sdf_val.abs().max(0.02); // sphere-march step
    }
    // Attenuation: -6dB per 0.3m of occluder thickness.
    let attenuation_db = -6.0 * (occlusion_length / 0.3);
    db_to_linear(attenuation_db)
}
```

### Diffraction

```rust
pub fn compute_diffraction_gain(
    source: Vec3,
    listener: Vec3,
    terrain: &TerrainVolume,
) -> f32 {
    // Find the nearest SDF edge (saddle point) on the direct path.
    // Openness: how much of the hemisphere is unoccluded at the diffraction point.
    let edge = find_nearest_sdf_edge(source, listener, terrain);
    let openness = estimate_openness(edge, terrain); // 0 = fully blocked, 1 = open
    // Diffraction gain (UTD approximation): 0.5 - 0.5 * cos(π * openness).
    0.5 - 0.5 * (std::f32::consts::PI * openness).cos()
}
```

### Doppler

```rust
pub fn compute_doppler_pitch(
    source_velocity: Vec3,
    listener_velocity: Vec3,
    source_to_listener: Vec3,
) -> f32 {
    let c = 343.0f32;
    let dir = source_to_listener.normalize_or_zero();
    let v_source = source_velocity.dot(-dir);    // positive = approaching
    let v_listener = listener_velocity.dot(dir); // positive = approaching
    (c + v_listener) / (c + v_source).max(0.01)
}
```

Applied per `AudioSource`; pitch shift implemented via linear resampling on the audio block.

### Reverb Send

Each `AudioSource` has a `reverb_send: f32` weight `[0, 1]`. The wet signal is `dry * (1 - reverb_send) + convolve(dry) * reverb_send`. This is mixed per band and then summed. Sources marked as `reverb_send = 0.0` (e.g., UI sounds, music) bypass the convolution path entirely.

### Integrated SpatialAudioManager (extended)

```rust
pub struct SpatialAudioSource {
    pub position: Vec3,
    pub velocity: Vec3,
    pub audio_buffer: Vec<f32>,
    pub reverb_send: f32,
    pub collision_filter: CollisionFilter, // for occlusion queries
    pub hrtf_enabled: bool,
}

impl SpatialAudioManager {
    pub fn process_source(
        &self,
        source: &SpatialAudioSource,
        listener: &Listener,
        terrain: &TerrainVolume,
        hrtf_table: &HrtfTable,
        reverb: &ReverberationEngine,
        output: &mut StereoBlock,
    ) {
        let to_listener = listener.position - source.position;
        let dist = to_listener.length();
        let attenuation = distance_attenuation(dist, source.falloff_start, source.falloff_end);
        let occlusion = compute_occlusion(source.position, listener.position, terrain);
        let diffraction = if occlusion < 0.5 { compute_diffraction_gain(...) } else { 1.0 };
        let pitch = compute_doppler_pitch(source.velocity, listener.velocity, to_listener);

        let (az, el) = world_to_hrtf_angles(to_listener, listener.orientation);
        let hrtf_filters = hrtf_table.get(az, el);

        let mut mono = resample(&source.audio_buffer, pitch); // doppler
        apply_gain(&mut mono, attenuation * occlusion * diffraction);

        // HRTF spatialization (headphone mode).
        if source.hrtf_enabled {
            output.left  += convolve_fir(&mono, &hrtf_filters[0]);
            output.right += convolve_fir(&mono, &hrtf_filters[1]);
        } else {
            let (l, r) = pan_law(az);
            output.left  += &mono * l;
            output.right += &mono * r;
        }

        // Reverb send.
        if source.reverb_send > 0.01 {
            let wet = reverb.convolve_all_bands(&mono); // returns stereo
            output.left  += &wet.left  * source.reverb_send;
            output.right += &wet.right * source.reverb_send;
        }
    }
}
```

---

## 6.4 Spectral Audio Graph — MetaSounds Equivalent

### AudioGraph

```rust
pub struct AudioGraph {
    pub nodes: HashMap<NodeId, AudioNode>,
    pub edges: Vec<AudioEdge>,  // (from_node, from_port) → (to_node, to_port)
}

pub struct AudioEdge {
    pub from: (NodeId, u8),
    pub to: (NodeId, u8),
}
```

### AudioNode

```rust
pub enum AudioNode {
    Oscillator {
        waveform: Waveform, // Sine, Square, Saw, Triangle
        freq_hz: f32,
        freq_mod_input: Option<NodeId>, // modulates freq if connected
    },
    NoiseGenerator {
        color: NoiseColor, // White, Pink, Brown, Spectral(MaterialId)
    },
    BandpassFilter {
        center_band: usize, // 0–7 maps to frequency via band_to_frequency()
        q: f32,
    },
    Envelope {
        attack: f32,
        decay: f32,
        sustain: f32,
        release: f32,
        trigger_input: NodeId, // 0→1 triggers attack; 1→0 triggers release
    },
    /// THE OCHROMA-SPECIFIC NODE:
    /// Reads the live spectral value of band `band` from a named SplatAssembly.
    /// Output is a scalar [0, 1] that can drive any graph parameter.
    SpectralLookup {
        assembly_id: AssemblyId,
        band: usize,
    },
    Spatializer {
        position: Vec3,
    },
    Reverb {
        probe_ref: AcousticProbeRef,
    },
    Mixer {
        input_count: usize,
        gains: Vec<f32>,
    },
    Output, // stereo out
}

pub enum Waveform { Sine, Square, Saw, Triangle }
pub enum NoiseColor { White, Pink, Brown, Spectral(MaterialId) }
pub struct AcousticProbeRef(pub usize); // index into ReverberationEngine::probes
```

**SpectralLookup example use case:** A fire splat assembly has high values in band 6–7 (orange-red).
A `SpectralLookup { assembly_id: FIRE_PIT, band: 7 }` node outputs a scalar that modulates an
`Oscillator`'s amplitude. As the fire grows (more splats added, higher opacity), band 7 rises, the
oscillator gets louder, and the low-frequency rumble intensifies. The audio IS the visual light energy —
no manual sync, no event triggers, no animation curves.

### Graph Evaluation

Graphs execute at block rate (512 samples, 44100 Hz → 11.6ms per block):

```rust
impl AudioGraph {
    /// Topologically sort nodes, then evaluate each in order.
    /// Returns stereo output block.
    pub fn evaluate(&self, sample_rate: u32, block_size: usize, context: &AudioContext) -> StereoBlock {
        let order = self.topological_sort();
        let mut node_outputs: HashMap<NodeId, Vec<f32>> = HashMap::new();

        for node_id in order {
            let inputs: Vec<&[f32]> = self.incoming_edges(node_id)
                .map(|(from_id, _)| node_outputs[&from_id].as_slice())
                .collect();

            let output = match &self.nodes[&node_id] {
                AudioNode::Oscillator { waveform, freq_hz, freq_mod_input } => {
                    let freq = freq_mod_input
                        .and_then(|id| node_outputs.get(&id))
                        .map(|mod_buf| freq_hz + mod_buf[0] * 100.0) // FM: ±100 Hz
                        .unwrap_or(*freq_hz);
                    synthesize_oscillator(*waveform, freq, sample_rate, block_size)
                }
                AudioNode::SpectralLookup { assembly_id, band } => {
                    let val = context.read_assembly_band(*assembly_id, *band);
                    vec![val; block_size] // constant scalar block
                }
                // ... other nodes
                _ => vec![0.0; block_size],
            };
            node_outputs.insert(node_id, output);
        }

        // Collect Output node.
        let output_node = self.find_output_node();
        let left = node_outputs[&output_node].clone();
        StereoBlock { left, right: left } // spatializer adds stereo spread
    }
}
```

**Compile to AudioClosure:** graphs are validated at author time, then compiled to a flat `Vec<AudioOp>`
enum sequence (akin to bytecode). `AudioClosure::evaluate()` iterates this vec with no HashMap lookups,
achieving zero-overhead runtime evaluation. The compiled form is serialized to `vox_data` asset format
and hot-reloaded in editor.

```rust
pub enum AudioOp {
    Oscillator { freq: f32, waveform: Waveform, out: u8 },
    SpectralLookup { assembly_id: AssemblyId, band: u8, out: u8 },
    BandpassFilter { in_: u8, center_freq: f32, q: f32, out: u8 },
    Mul { a: u8, b: u8, out: u8 },
    Mix { a: u8, b: u8, gain_a: f32, gain_b: f32, out: u8 },
    Output { in_: u8 },
}

pub struct AudioClosure {
    pub ops: Vec<AudioOp>,
    pub register_count: u8, // number of f32 block registers needed
}
```

---

## 6.5 Music System

### Adaptive Music

```rust
pub struct AdaptiveMusic {
    pub layers: Vec<MusicLayer>,
    pub current_intensity: f32,
    pub beat_clock: BeatClock,
    pub crossfade_duration: f32, // seconds
}

pub struct MusicLayer {
    pub clip: AudioClip,
    pub min_intensity: f32,
    pub max_intensity: f32,
    /// Maps intensity (normalized within min/max) to volume [0, 1].
    pub volume_curve: AnimCurve,
    pub current_volume: f32,
    pub target_volume: f32,
}

pub struct BeatClock {
    pub bpm: f32,
    pub current_beat: f32,
    pub samples_per_beat: usize,
    pub sample_counter: usize,
}

impl BeatClock {
    pub fn advance(&mut self, block_size: usize) -> bool {
        self.sample_counter += block_size;
        let beat_samples = (60.0 / self.bpm * 44100.0) as usize;
        if self.sample_counter >= beat_samples {
            self.sample_counter -= beat_samples;
            self.current_beat += 1.0;
            true // downbeat fired
        } else { false }
    }
}
```

**Intensity sources** (all normalized 0–1, blended with weights):

| Source | Weight | Derivation |
|--------|--------|------------|
| Player health | 0.30 | `1.0 - health / max_health` |
| Enemy count (30m radius) | 0.25 | `(enemy_count / 5.0).min(1.0)` |
| Player velocity | 0.15 | `(speed / 8.0).min(1.0)` |
| SpectralIntensity (scene) | 0.30 | see below |

**SpectralIntensity:** Sum the dominant-band energy across all visible splat assemblies within 50m.
High blue-band dominance → calm (low intensity). High red/orange-band (bands 5–7) → intense. Computed
at 4 Hz (same pass as `AmbientSynthesizer`):

```rust
pub fn compute_spectral_intensity(
    nearby_assemblies: &[&SplatAssembly],
    scene_splats: &[GaussianSplat],
) -> f32 {
    let mut band_energy = [0.0f32; 8];
    for assembly in nearby_assemblies {
        for &idx in &assembly.splat_indices {
            let s = decode_spectral_u16(scene_splats[idx as usize].spectral);
            for b in 0..8 { band_energy[b] += s[b]; }
        }
    }
    // Red/warm = bands 5–7; blue/cool = bands 1–3.
    let warm = band_energy[5] + band_energy[6] + band_energy[7];
    let cool = band_energy[1] + band_energy[2] + band_energy[3];
    let total = (warm + cool).max(0.001);
    (warm / total).clamp(0.0, 1.0)
}
```

**Beat-synced crossfade:** when `current_intensity` crosses a layer's `min_intensity` or `max_intensity`
threshold, the layer is marked for fade. The fade begins on the next downbeat (as reported by
`BeatClock::advance`), preventing rhythmic glitches. Crossfade is a linear ramp over `crossfade_duration`.

---

## 6.6 Voice & Dialogue

### Data Structures

```rust
pub struct DialogueLine {
    pub id: DialogueId,
    pub text: String,
    /// Optional pre-recorded audio. If None, TTS stub or silence.
    pub audio_path: Option<std::path::PathBuf>,
    pub lip_sync_data: Option<LipSyncData>,
    pub duration_seconds: f32,
}

pub struct LipSyncData {
    /// Phoneme events with start time and duration (seconds).
    pub phonemes: Vec<PhonemeEvent>,
}

pub struct PhonemeEvent {
    pub start: f32,
    pub duration: f32,
    pub phoneme: Phoneme,
}

/// CMU Pronouncing Dictionary phoneme set (39 phonemes → 15 visemes).
pub enum Phoneme {
    AA, AE, AH, AO, AW, AY, B, CH, D, DH,
    EH, ER, EY, F, G, HH, IH, IY, JH, K,
    L, M, N, NG, OW, OY, P, R, S, SH,
    T, TH, UH, UW, V, W, Y, Z, ZH,
}

/// Reduced viseme set for facial rig driving.
pub enum Viseme {
    Closed, SlightlyOpen, Open, Wide,
    RoundedNarrow, RoundedWide,
    TeethTogether, UpperTeeth, LowerTeeth,
    Lips, Neutral,
}

pub fn phoneme_to_viseme(p: Phoneme) -> Viseme { ... }
```

### DialoguePlayer

```rust
pub struct DialoguePlayer {
    pub active_line: Option<ActiveDialogue>,
    pub subtitle_display: SubtitleDisplay,
    pub facial_rig: Option<FacialRigHandle>,
}

pub struct ActiveDialogue {
    pub line: DialogueLine,
    pub elapsed: f32,
    pub phoneme_cursor: usize,
}

impl DialoguePlayer {
    pub fn play(&mut self, line: DialogueLine, audio: &mut AudioEngine) {
        if let Some(path) = &line.audio_path {
            audio.play_2d(path, PlaybackParams::default());
        }
        self.subtitle_display.show(&line.text, line.duration_seconds);
        self.active_line = Some(ActiveDialogue { line, elapsed: 0.0, phoneme_cursor: 0 });
    }

    pub fn update(&mut self, dt: f32, facial_rig: Option<&mut FacialRig>) {
        let Some(active) = &mut self.active_line else { return };
        active.elapsed += dt;

        if let (Some(lip_sync), Some(rig)) = (&active.line.lip_sync_data, facial_rig) {
            // Advance phoneme cursor to current time.
            while active.phoneme_cursor < lip_sync.phonemes.len() {
                let ev = &lip_sync.phonemes[active.phoneme_cursor];
                if active.elapsed >= ev.start + ev.duration {
                    active.phoneme_cursor += 1;
                } else { break; }
            }
            // Drive rig with current viseme.
            if let Some(ev) = lip_sync.phonemes.get(active.phoneme_cursor) {
                if active.elapsed >= ev.start {
                    rig.set_viseme(phoneme_to_viseme(ev.phoneme.clone()));
                }
            }
        }

        if active.elapsed >= active.line.duration_seconds {
            self.active_line = None;
        }
    }
}
```

### Localization

```rust
pub struct DialogueTable {
    pub entries: HashMap<Locale, HashMap<DialogueId, DialogueLine>>,
    pub active_locale: Locale,
}

impl DialogueTable {
    /// Hot-swap locale at runtime. No restart required.
    /// In-flight dialogue finishes in the old locale; next play() uses new.
    pub fn set_locale(&mut self, locale: Locale) {
        self.active_locale = locale;
    }

    pub fn get(&self, id: DialogueId) -> Option<&DialogueLine> {
        self.entries.get(&self.active_locale)?.get(&id)
    }
}
```

**LipSync pre-computation (offline):** run a phoneme classifier (e.g., `wav2vec2`-based CMU aligner)
on each audio clip at asset build time. Output is serialized to `LipSyncData` and stored alongside the
audio in `vox_data`. At runtime, no speech analysis occurs.

---

## File Map

```
crates/vox_audio/
  src/
    lib.rs                   — AudioEngine, re-exports
    synthesis.rs             — SpectralSynthesizer, FootstepSynthesizer, synthesize_impact
    ambient.rs               — AmbientSynthesizer, WindSynthesizer
    reverb.rs                — AcousticProbe, AudioRayMarcher, ReverberationEngine
    materials.rs             — MaterialAcoustics, MaterialSoundLibrary, MaterialSoundProfile
    spatial.rs               — SpatialAudioManager (extended), HRTF, occlusion, diffraction, Doppler
    graph.rs                 — AudioGraph, AudioNode, AudioEdge, AudioClosure, AudioOp
    music.rs                 — AdaptiveMusic, MusicLayer, BeatClock, compute_spectral_intensity
    dialogue.rs              — DialogueLine, DialoguePlayer, LipSyncData, DialogueTable
    hrtf_data.rs             — HrtfTable, include_bytes KEMAR embed
    dsp.rs                   — overlap_add_convolve, biquad_bandpass, bandpass_noise, mix_into, ...
  data/
    kemar_hrtf.bin           — MIT KEMAR HRTF dataset (compressed, embedded)
  tests/
    reverb_test.rs
    synthesis_test.rs
    spatial_test.rs
    graph_test.rs
    music_test.rs
    dialogue_test.rs
```

Integration points:
- `vox_terrain/src/lib.rs` — `AudioRayMarcher` uses `TerrainVolume::sdf()`, `sdf_gradient()`, `material_at()`.
- `vox_physics/src/fracture.rs` — `ImpactAudioRequest` consumed by `SpectralSynthesizer`.
- `vox_render/src/lib.rs` — `AmbientSynthesizer` reads `SplatAssembly` spectral data from scene splats.
- `vox_physics/src/query.rs` — `compute_occlusion` uses same SDF as physics raycast.

---

## Milestones

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M6.1 | `AudioRayMarcher::measure` + `MaterialAcoustics` table; unit test: measure concrete box → verify IR tail length | Phase 6, week 1 |
| M6.2 | `ReverberationEngine` FFT convolution; spatial audio source plays through per-band reverb | Phase 6, week 1 |
| M6.3 | `SpectralSynthesizer::synthesize_impact` + `FootstepSynthesizer`; all 3 surface variants tested | Phase 6, week 2 |
| M6.4 | `AmbientSynthesizer` + `WindSynthesizer`; integration test: fire scene produces crackling ambient | Phase 6, week 2 |
| M6.5 | HRTF table embedded + `SpatialAudioManager` extended with occlusion, diffraction, Doppler | Phase 6, week 3 |
| M6.6 | `AudioGraph` + `SpectralLookup` node; demo: fire splat drives oscillator amplitude in real-time | Phase 6, week 3 |
| M6.7 | `AdaptiveMusic` + `BeatClock` + `compute_spectral_intensity`; 3-layer music demo | Phase 6, week 4 |
| M6.8 | `DialoguePlayer` + `LipSyncData` + `DialogueTable`; locale hot-swap test | Phase 6, week 4 |

---

## Acceptance Criteria

- **SDF Reverb:** Measuring an `AcousticProbe` inside a 5×5×3m concrete room produces an impulse response with RT60 > 0.8s in band 0 (low freq) and RT60 < 0.4s in band 7 (high freq). Carpet lining reduces RT60 in band 7 by > 50% compared to concrete.
- **Dynamic reverb update:** After `TerrainVolume::carve_sphere(pos, 2.0)`, all probes within 4m re-measure within the next two audio frames. Old IR is not audible after re-measure.
- **FootstepSynthesizer:** Grass, metal grate, and gravel footsteps are spectrally distinguishable (confirmed by FFT analysis of outputs: peak frequencies differ by > 1 octave). Zero audio file assets loaded.
- **SpectralLookup graph node:** A graph with `SpectralLookup { band: 7 }` → `Oscillator amplitude` produces silence when band 7 = 0.0 and audible tone when band 7 = 1.0. Evaluated within one audio block (11.6ms) of assembly spectral data changing.
- **HRTF:** A source at azimuth 90° (directly right) produces left-channel output at least 12 dB lower than right-channel when HRTF is enabled.
- **Occlusion:** A source separated from the listener by 0.5m of concrete (SDF solid) is attenuated by ≥ 10 dB compared to line-of-sight at the same distance.
- **AdaptiveMusic:** Intensity increase from 0.2 to 0.8 triggers a layer crossfade that begins on the next downbeat (within one beat period) and completes within `crossfade_duration` seconds.
- **DialoguePlayer:** Phoneme cursor advances correctly over a 3-second clip; viseme transitions match phoneme timings within ±16ms (one audio block). Locale hot-swap between EN and DE completes within one frame with no crash.

---

## Effort Estimate

| Section | Estimate |
|---------|----------|
| 6.1 SDF Reverb (ray marcher + FFT convolution) | 2.5 days |
| 6.2 SpectralSynthesizer + FootstepSynthesizer | 1.5 days |
| 6.3 AmbientSynthesizer + WindSynthesizer | 1 day |
| 6.4 HRTF embed + occlusion + diffraction + Doppler | 2 days |
| 6.5 AudioGraph + SpectralLookup + AudioClosure compile | 2.5 days |
| 6.6 AdaptiveMusic + BeatClock | 1 day |
| 6.7 DialoguePlayer + LipSync + Localization | 1.5 days |
| Tests + integration | 1 day |
| **Total** | **~13 days** |

Dependencies: `rustfft 6` (FFT convolution), `rapier3d` (shared with Domain 5 for physics queries),
`rayon` (parallel probe measurement). KEMAR HRTF dataset: public domain, MIT Media Lab.
New crate dependency: `vox_audio` must add `vox_terrain` to `Cargo.toml` for SDF material lookup.
